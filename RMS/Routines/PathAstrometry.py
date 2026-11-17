""" Astrometry of a faint point source which moves along a known path, measured on the raw frames.

A point source too faint to be seen in single frames can still be measured when its path across the sky is known in
advance, e.g. from an ephemeris: the frames are averaged into stacks, the stars are removed from every stack, and the
stacks are added along the predicted path, so that the light of the source adds up while the noise averages down.
The measured position, compared with an independent prediction of high accuracy, shows how well the astrometry of
the camera (the plate, the times of the frames, the conversions of the coordinates) puts a moving source where it is.

The measurement:
    1) The frames are averaged into stacks of stack_frames frames, with the dark and the flat applied as in the
        processing.
    2) The stars of every stack are extracted and the pointing of the platepar is recalibrated on them, as the
        processing does for every chunk of frames.
    3) The stacks of a window of window_stacks stacks are mapped onto the middle stack of the window through their
        recalibrated platepars, so that the stars are at the same place in all of them while the source moves. The
        median of the mapped stacks is the image of the stars, which is mapped back onto every stack and subtracted.
    4) The stacks of a group of group_stacks stacks are shifted so that the source predicted by the path is at the
        same place in all of them, and averaged. The source is centroided with a Gaussian near the position the
        path predicts for it in the middle stack of the group.
    5) The centroid is converted to RA and Dec with the recalibrated platepar of that stack at its time, as a
        measurement (as the detections are converted), so it can be compared with the path directly.

The source has to move against the stars over a window by several times the width of the stars, otherwise the
median of step 3 keeps its light and the subtraction removes it.

The raw frames are read through the frame interface (RMS.Formats.FrameInterface, see RMS.Routines.TrackShape.
openRawSource), so every raw input of RMS can be measured, and the frames are timed exactly as the processing timed
them (frame_time_offset, the repair of the timestamps).

Command line:
    python -m RMS.Routines.PathAstrometry raw_path path.csv -c config -p platepar [--dark bias.png] [--flat flat.png]
        [-o measurements.csv]
path.csv has the columns time (UTC, ISO format, or Unix seconds), ra and dec (degrees, J2000, the direction of the
source as a measurement of the camera would give it), and the measurements are written to a CSV file.
"""

from __future__ import print_function, division, absolute_import

import argparse
import csv
import datetime
import logging
import os

import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit


log = logging.getLogger("Logger")


# Frames averaged into a stack (about the length of a chunk of frames of the processing at 32 frames/s)
STACK_FRAMES = 80

# Stacks of a window, whose median is the image of the stars, and stacks of a group which are added along the path
#   (a window holds whole groups)
WINDOW_STACKS = 24
GROUP_STACKS = 8

# Fewest recalibrated stacks of a window for the median of the stars, and fewest stacks of a group for a measurement
MIN_WINDOW_STACKS = 7
MIN_GROUP_STACKS = 4

# The source has to move against the stars over a window by at least this many times the width of the box of the
#   fit, so that the median of the window doesn't keep its light
MIN_WINDOW_MOTION = 2.0

# Half size of the box in which the source is fitted (px), and the largest distance between the fitted and the
#   predicted position (px)
FIT_HALF = 4
SEARCH_RADIUS = 3.0

# Fewest signal-to-noise ratio of a measurement, and the limits of the Gaussian sigma (px)
MIN_SNR = 5.0
SIGMA_LIMITS = (0.3, 3.0)

# Grid of image positions on which the mapping between two stacks is computed (points per axis)
MAP_GRID = 17

# Columns of the CSV file of the command line
CSV_COLUMNS = ['time_utc', 'time_unix', 'ra', 'dec', 'sigma_ra', 'sigma_dec', 'x', 'y', 'sigma_x', 'sigma_y',
               'x_pred', 'y_pred', 'ra_pred', 'dec_pred', 'snr', 'amplitude', 'fwhm', 'n_stacks', 'n_frames',
               'n_stars', 'aberration']



### The path ###

def unitVectors(ra, dec):
    """ Unit vectors of directions.

    Arguments:
        ra: [ndarray] Right ascensions (deg).
        dec: [ndarray] Declinations (deg).

    Return:
        [ndarray] Unit vectors, one row per direction.
    """

    ra, dec = np.radians(np.asarray(ra, dtype=np.float64)), np.radians(np.asarray(dec, dtype=np.float64))

    return np.column_stack([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])



def raDecOfVectors(u):
    """ RA and Dec of vectors.

    Arguments:
        u: [ndarray] Vectors, one row per direction.

    Return:
        (ra, dec): [tuple of ndarrays] Right ascensions and declinations (deg).
    """

    u = np.atleast_2d(u)
    u = u/np.linalg.norm(u, axis=1)[:, None]

    return np.degrees(np.arctan2(u[:, 1], u[:, 0]))%360, np.degrees(np.arcsin(np.clip(u[:, 2], -1, 1)))



class Path(object):
    def __init__(self, times, ra, dec):
        """ The predicted path of the source on the sky, interpolated between its points.

        Arguments:
            times: [ndarray] Unix times of the points (s).
            ra: [ndarray] Right ascensions of the points (deg).
            dec: [ndarray] Declinations of the points (deg).
        """

        # The points in time order, at least two of them
        order = np.argsort(times)
        self.times = np.asarray(times, dtype=np.float64)[order]
        self.vectors = unitVectors(np.asarray(ra)[order], np.asarray(dec)[order])
        if len(self.times) < 2:
            raise ValueError('A path needs at least two points')


    def at(self, t):
        """ The direction of the path at the given times, interpolated linearly between the unit vectors of its
            points (the points are close enough that the arc between them is a straight line), NaN outside of it.

        Arguments:
            t: [ndarray] Unix times (s).

        Return:
            (ra, dec): [tuple of ndarrays] Right ascensions and declinations (deg).
        """

        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        u = np.column_stack([np.interp(t, self.times, self.vectors[:, i]) for i in range(3)])
        ra, dec = raDecOfVectors(u)

        # No direction outside of the path
        outside = (t < self.times[0]) | (t > self.times[-1])
        ra[outside], dec[outside] = np.nan, np.nan

        return ra, dec



### Stacks ###

def unixToJD(t):
    """ Julian date of a Unix time.

    Arguments:
        t: [float] Unix time (s).

    Return:
        [float] Julian date.
    """

    return float(t)/86400.0 + 2440587.5



def stackName(station_id, t_start):
    """ The name of an FF file which begins at the given time, which the recalibration takes the time from.

    Arguments:
        station_id: [str] Station code.
        t_start: [float] Unix time of the beginning of the stack (s).

    Return:
        [str] The name.
    """

    from RMS.Formats.FFfile import constructFFName

    dt = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=float(t_start))

    return constructFFName(station_id, dt)



def stacksOfFrames(source, indices, times, stack_frames=STACK_FRAMES, dark=None, flat_struct=None):
    """ Given frames averaged into stacks of consecutive frames, with the dark and the flat applied.

    Arguments:
        source: [RawFrames] The raw frames (see RMS.Routines.TrackShape.openRawSource), or any object with the
            method frame.
        indices: [ndarray] Frame numbers, in time order.
        times: [ndarray] Their times (Unix s).

    Keyword arguments:
        stack_frames: [int] Frames per stack. STACK_FRAMES by default.
        dark: [ndarray] Dark frame, None (default) for none.
        flat_struct: [Flat struct] Flat field, None (default) for none.

    Return:
        [list] Stacks, each a dict with the image ('img', float32), the mean time of its frames ('t', Unix s), the
            time of its first and last frame ('t_first', 't_last') and the number of frames ('n_frames'). Frames
            left over at the end which don't fill a stack are not used.
    """

    from RMS.Routines.Image import applyDark, applyFlat

    stacks = []
    for k in range(0, len(indices) - stack_frames + 1, stack_frames):

        # The mean of the frames of the stack (frames which can't be read are left out)
        acc, n, t_sum = None, 0, 0.0
        for i, t in zip(indices[k:k + stack_frames], times[k:k + stack_frames]):
            try:
                frame = source.frame(int(i)).astype(np.float64)
            except (IOError, ValueError, IndexError) as e:
                log.warning('Frame {:d} could not be read: {}'.format(int(i), e))
                continue
            acc = frame if acc is None else acc + frame
            n += 1
            t_sum += t
        if n < stack_frames//2:
            continue
        img = (acc/n).astype(np.float32)

        # The dark and the flat applied to the mean, as to the images of the processing
        if dark is not None:
            img = applyDark(img, dark.astype(np.float32))
        if flat_struct is not None:
            img = applyFlat(img, flat_struct)

        stacks.append(dict(img=img, t=t_sum/n, t_first=float(times[k]), t_last=float(times[k + stack_frames - 1]),
                           n_frames=n))

    return stacks



def makeStacks(source, t0, t1, stack_frames=STACK_FRAMES, dark=None, flat_struct=None):
    """ The frames between t0 and t1 averaged into stacks of consecutive frames, with the dark and the flat
        applied (see stacksOfFrames).

    Arguments:
        source: [RawFrames] The raw frames (see RMS.Routines.TrackShape.openRawSource), or any object with the
            methods framesBetween and frame.
        t0: [float] Beginning (Unix s).
        t1: [float] End (Unix s).

    Keyword arguments:
        stack_frames: [int] Frames per stack. STACK_FRAMES by default.
        dark: [ndarray] Dark frame, None (default) for none.
        flat_struct: [Flat struct] Flat field, None (default) for none.

    Return:
        [list] Stacks (see stacksOfFrames).
    """

    indices, times = source.framesBetween(t0, t1)

    return stacksOfFrames(source, indices, times, stack_frames=stack_frames, dark=dark, flat_struct=flat_struct)



def stackStars(img, config, mask=None):
    """ The stars of a stack, extracted with the settings of the config, as rows of a CALSTARS file.

    Arguments:
        img: [ndarray] The stack.
        config: [Config] Configuration of the camera.

    Keyword arguments:
        mask: [MaskStructure] Mask of the camera, None (default) for none.

    Return:
        [ndarray] Stars (y, x, intensity, amplitude, FWHM, background, SNR, saturated), empty if none.
    """

    from RMS.ExtractStars import extractStars

    result = extractStars(img.astype(np.float32), img_median=float(np.median(img)), mask=mask, gamma=config.gamma,
                          max_star_candidates=config.max_stars, border=10,
                          neighborhood_size=config.neighborhood_size, intensity_threshold=config.intensity_threshold,
                          segment_radius=config.segment_radius, roundness_threshold=config.roundness_threshold,
                          max_feature_ratio=config.max_feature_ratio, bit_depth=config.bit_depth)
    if result is False:
        return np.zeros((0, 8))

    x, y, amplitude, intensity, fwhm, background, snr, saturated = [np.asarray(v, dtype=np.float64)
                                                                     for v in result]

    return np.column_stack([y, x, intensity, amplitude, fwhm, background, snr, saturated])



def recalibrateStacks(stacks, platepar, config, catalog_stars, stack_frames=STACK_FRAMES, mask=None):
    """ Recalibrate the pointing of the platepar on the stars of every stack, as the processing does for every
        chunk of frames. Every stack starts from the given platepar.

    Arguments:
        stacks: [list] Stacks (see makeStacks), each gets its platepar ('platepar', None if the recalibration
            failed) and its number of stars ('n_stars').
        platepar: [Platepar] Platepar to start from.
        config: [Config] Configuration of the camera.
        catalog_stars: [ndarray] Catalog stars (ra, dec, mag).

    Keyword arguments:
        stack_frames: [int] Frames per stack. STACK_FRAMES by default.
        mask: [MaskStructure] Mask of the camera, None (default) for none.

    Return:
        None
    """

    from RMS.Astrometry.ApplyRecalibrate import recalibrateSingleFF

    # The recalibration logs every step: kept quiet here
    rms_log = logging.getLogger("rmslogger")
    level = rms_log.level
    rms_log.setLevel(logging.WARNING)
    try:
        for stack in stacks:
            stars = stackStars(stack['img'], config, mask=mask)
            stack['n_stars'] = len(stars)
            stack['platepar'] = None
            if len(stars) < config.min_matched_stars:
                continue

            # The stack is named by the time of its first frame and timed by the mean time of its frames
            name = stackName(config.stationID, stack['t_first'])
            recalibrated, ok = recalibrateSingleFF(platepar, name, stars, catalog_stars, config,
                                                   ff_frames=stack_frames, jd=unixToJD(stack['t']))
            if ok:
                stack['platepar'] = recalibrated
    finally:
        rms_log.setLevel(level)



### Directions as measurements ###

def measurementToXY(ra, dec, jd, platepar):
    """ Image positions of directions given as measurements of the camera, i.e. as xyToRaDecPP gives them with
        measurement=True: the inverse of that conversion. A plate fitted without the refraction whose measurements
        are corrected for it afterwards (measurement_apparent_to_true_refraction) gets the refraction put back
        before the projection, which raDecToXYPP doesn't do.

    Arguments:
        ra: [ndarray] Right ascensions (deg).
        dec: [ndarray] Declinations (deg).
        jd: [float] Julian date.
        platepar: [Platepar] Platepar.

    Return:
        (x, y): [tuple of ndarrays] Image coordinates.
    """

    from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, refractionScale
    from RMS.Astrometry.CyFunctions import eqRefractionTrueToApparent

    ra = np.array(np.atleast_1d(ra), dtype=np.float64)
    dec = np.array(np.atleast_1d(dec), dtype=np.float64)

    # The refraction which the conversion of the measurements removed afterwards, put back
    if (not platepar.refraction) and getattr(platepar, 'measurement_apparent_to_true_refraction', False):
        scale = refractionScale(platepar.elev)
        for i in range(len(ra)):
            ra_app, dec_app = eqRefractionTrueToApparent(np.radians(ra[i]), np.radians(dec[i]), float(jd),
                                                         np.radians(platepar.lat), np.radians(platepar.lon), scale)
            ra[i], dec[i] = np.degrees(ra_app), np.degrees(dec_app)

    return raDecToXYPP(ra, dec, jd, platepar, measurement=True)



### Mapping between stacks ###

def stackMapping(pp_from, t_from, pp_to, t_to, width, height, grid=MAP_GRID):
    """ The mapping from the pixels of one stack to the image positions of another stack which see the same stars,
        through their platepars: a grid of positions of the first stack is converted to the sky (as stars) and back
        to the image of the second, and interpolated between the grid points with bicubic splines (the mapping
        follows the distortion of the lens, which an affine mapping doesn't over a wide field).

    Arguments:
        pp_from: [Platepar] Platepar of the first stack.
        t_from: [float] Time of the first stack (Unix s).
        pp_to: [Platepar] Platepar of the second stack.
        t_to: [float] Time of the second stack (Unix s).
        width: [int] Image width (px).
        height: [int] Image height (px).

    Keyword arguments:
        grid: [int] Points per axis of the grid. MAP_GRID by default.

    Return:
        (rows, cols): [tuple of ndarrays] For every pixel of the first stack (arrays of the image's shape), the row
            and the column of the same sky position in the second.
    """

    from scipy.interpolate import RectBivariateSpline

    from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, xyToRaDecPP

    # The grid over the whole image, edges included
    x_axis = np.linspace(0, width - 1, grid)
    y_axis = np.linspace(0, height - 1, grid)
    xg, yg = np.meshgrid(x_axis, y_axis)

    # The sky positions of the grid in the first stack, and their image positions in the second
    _, ra, dec, _ = xyToRaDecPP(np.full(xg.size, unixToJD(t_from)), xg.ravel(), yg.ravel(), np.ones(xg.size),
                                pp_from, extinction_correction=False, jd_time=True)
    x2, y2 = raDecToXYPP(np.array(ra), np.array(dec), unixToJD(t_to), pp_to)

    # Every pixel interpolated between the grid points
    rows = RectBivariateSpline(y_axis, x_axis, np.reshape(y2, xg.shape))(np.arange(height), np.arange(width))
    cols = RectBivariateSpline(y_axis, x_axis, np.reshape(x2, xg.shape))(np.arange(height), np.arange(width))

    return rows, cols



def mapStack(img, rows, cols, order=3):
    """ An image resampled through a mapping (see stackMapping), NaN where it falls outside.

    Arguments:
        img: [ndarray] Image.
        rows: [ndarray] Row in the image of every output pixel.
        cols: [ndarray] Column in the image of every output pixel.

    Keyword arguments:
        order: [int] Order of the spline interpolation. 3 by default.

    Return:
        [ndarray] The resampled image.
    """

    return ndimage.map_coordinates(img.astype(np.float64), [rows, cols], order=order, mode='constant',
                                   cval=np.nan, prefilter=(order > 1))



def removeStars(stacks, width, height):
    """ Subtract the image of the stars from the stacks of a window: the stacks are mapped onto the middle stack
        (the stars are fixed on the sky), the median of the mapped stacks is the image of the stars, which is mapped
        back onto every stack and subtracted.

    Arguments:
        stacks: [list] Recalibrated stacks of the window (see recalibrateStacks), each gets its image without the
            stars ('residual', NaN where the median is not known).
        width: [int] Image width (px).
        height: [int] Image height (px).

    Return:
        None
    """

    ref = stacks[len(stacks)//2]

    # The stacks mapped onto the middle stack: each pixel of the middle stack taken from the same sky position of
    #   the other stack
    mapped = []
    for stack in stacks:
        rows, cols = stackMapping(ref['platepar'], ref['t'], stack['platepar'], stack['t'], width, height)
        mapped.append(mapStack(stack['img'], rows, cols))

    # The median keeps the stars, which are at the same place in all mapped stacks, and not the moving source
    with np.errstate(all='ignore'):
        stars = np.nanmedian(np.array(mapped), axis=0)

    # Where no stack covers the middle stack the image of the stars is unknown: it is filled for the resampling (the
    #   spline of a resampling spreads a NaN over the whole image) and those pixels are left unknown afterwards
    known = np.isfinite(stars)
    stars = np.where(known, stars, np.nanmedian(stars) if np.any(known) else 0.0)

    # The image of the stars mapped back onto every stack and subtracted
    for stack in stacks:
        rows, cols = stackMapping(stack['platepar'], stack['t'], ref['platepar'], ref['t'], width, height)
        residual = stack['img'].astype(np.float64) - mapStack(stars, rows, cols)
        covered = mapStack(known.astype(np.float64), rows, cols, order=1)
        residual[~(covered > 0.999)] = np.nan
        stack['residual'] = residual



### Measurement of the source ###

def _gaussian2D(xy, amplitude, x0, y0, sigma, background):
    """ A circular 2D Gaussian on a constant background.

    Arguments:
        xy: [tuple] Arrays of x and y.
        amplitude: [float] Peak above the background.
        x0: [float] Centre x.
        y0: [float] Centre y.
        sigma: [float] Sigma (px).
        background: [float] Constant background.

    Return:
        [ndarray] The values at the positions.
    """

    x, y = xy

    return amplitude*np.exp(-((x - x0)**2 + (y - y0)**2)/(2*sigma**2)) + background



def fitSource(img, x, y, fit_half=FIT_HALF, search_radius=SEARCH_RADIUS, min_snr=MIN_SNR):
    """ Fit the source with a circular Gaussian in a box around its predicted position.

    Arguments:
        img: [ndarray] Image without the stars (NaN where unknown).
        x: [float] Predicted x (px).
        y: [float] Predicted y (px).

    Keyword arguments:
        fit_half: [int] Half size of the box of the fit (px). FIT_HALF by default.
        search_radius: [float] Largest distance of the fitted centre from the prediction (px). SEARCH_RADIUS by
            default.
        min_snr: [float] Fewest signal-to-noise ratio. MIN_SNR by default.

    Return:
        [dict] The centre ('x', 'y'), its uncertainties ('sigma_x', 'sigma_y', px), the amplitude, the FWHM and the
            signal-to-noise ratio of the peak ('snr'), None if no source meets the limits.
    """

    xi, yi = int(round(x)), int(round(y))
    half = fit_half + int(np.ceil(search_radius))

    # The box must be inside the image and known everywhere
    if (xi - half < 0) or (yi - half < 0) or (xi + half >= img.shape[1]) or (yi + half >= img.shape[0]):
        return None
    box = img[yi - half:yi + half + 1, xi - half:xi + half + 1]
    if not np.all(np.isfinite(box)):
        return None

    # The noise from the pixels around the prediction outside of the source (median absolute deviation)
    yy, xx = np.mgrid[yi - half:yi + half + 1, xi - half:xi + half + 1]
    ring = np.hypot(xx - x, yy - y) > fit_half
    noise = 1.4826*np.median(np.abs(box[ring] - np.median(box[ring])))
    if not (noise > 0):
        return None

    # The Gaussian, starting at the brightest pixel near the prediction
    near = np.hypot(xx - x, yy - y) <= search_radius
    k = np.argmax(np.where(near, box, -np.inf))
    p0 = (box.ravel()[k] - np.median(box), xx.ravel()[k], yy.ravel()[k], 1.0, np.median(box))
    try:
        p, cov = curve_fit(_gaussian2D, (xx.ravel(), yy.ravel()), box.ravel(), p0=p0, maxfev=4000)
    except (RuntimeError, ValueError):
        return None
    errors = np.sqrt(np.diag(cov)) if np.all(np.isfinite(cov)) else np.full(5, np.nan)

    amplitude, xc, yc, sigma, _ = p
    sigma = abs(sigma)
    snr = amplitude/noise

    # A source near the prediction, of the size of a point source and above the noise
    if (amplitude <= 0) or (np.hypot(xc - x, yc - y) > search_radius) or (not SIGMA_LIMITS[0] < sigma
                                                                          < SIGMA_LIMITS[1]) \
            or (snr < min_snr) or (not np.all(np.isfinite(errors[1:3]))):
        return None

    return dict(x=float(xc), y=float(yc), sigma_x=float(errors[1]), sigma_y=float(errors[2]),
                amplitude=float(amplitude), fwhm=float(2.3548*sigma), snr=float(snr))



def _cutout(img, x, y, half):
    """ A square cutout of an image centred on the nearest pixel of a position, NaN outside of the image.

    Arguments:
        img: [ndarray] Image.
        x: [float] Centre x (px).
        y: [float] Centre y (px).
        half: [int] Half size of the cutout (px).

    Return:
        (cutout, x0, y0): [tuple] The cutout and the image position of its first pixel.
    """

    xi, yi = int(round(x)), int(round(y))
    out = np.full((2*half + 1, 2*half + 1), np.nan)
    y_lo, y_hi = max(yi - half, 0), min(yi + half + 1, img.shape[0])
    x_lo, x_hi = max(xi - half, 0), min(xi + half + 1, img.shape[1])
    if (y_lo < y_hi) and (x_lo < x_hi):
        out[y_lo - (yi - half):y_hi - (yi - half), x_lo - (xi - half):x_hi - (xi - half)] = img[y_lo:y_hi,
                                                                                                x_lo:x_hi]

    return out, xi - half, yi - half



def measureGroup(stacks, path, fit_half=FIT_HALF, search_radius=SEARCH_RADIUS, min_snr=MIN_SNR):
    """ Measure the source in a group of stacks without the stars: the stacks are shifted so that the predicted
        source is where it is predicted in the middle stack, averaged, and the source is fitted there.

    Arguments:
        stacks: [list] Stacks of the group with their residuals (see removeStars).
        path: [Path] Predicted path of the source.

    Keyword arguments:
        fit_half: [int] Half size of the box of the fit (px). FIT_HALF by default.
        search_radius: [float] Largest distance of the fitted centre from the prediction (px). SEARCH_RADIUS by
            default.
        min_snr: [float] Fewest signal-to-noise ratio. MIN_SNR by default.

    Return:
        [dict] The measurement (see CSV_COLUMNS), None if the source isn't found. 'aberration' is 1 if the
            platepar models the annual aberration, so that RA and Dec are the direction of the source in the
            Earth's frame (as for the detections), 0 if it has absorbed it (RA and Dec relative to the stars).
    """

    from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP

    # The predicted position of the source in every stack, as a measurement of the camera
    pred = []
    for stack in stacks:
        ra, dec = path.at(stack['t'])
        if not np.isfinite(ra[0]):
            return None
        x, y = measurementToXY(ra, dec, unixToJD(stack['t']), stack['platepar'])
        pred.append((float(x[0]), float(y[0])))

    # The middle stack of the group, which the others are shifted onto
    m = len(stacks)//2
    xm, ym = pred[m]
    half = fit_half + int(np.ceil(search_radius)) + 2

    # Every residual cut around its predicted source and shifted so the source is at the middle stack's position
    cutouts = []
    for stack, (x, y) in zip(stacks, pred):
        cut, x0, y0 = _cutout(stack['residual'], x, y, half + 2)
        shift = ((ym - y0) - (y - y0) - (round(ym) - round(y)), (xm - x0) - (x - x0) - (round(xm) - round(x)))
        filled = np.where(np.isfinite(cut), cut, 0.0)
        valid = ndimage.shift(np.isfinite(cut).astype(np.float64), shift, order=1, mode='constant', cval=0.0)
        moved = ndimage.shift(filled, shift, order=3, mode='constant', cval=0.0)
        moved[valid < 0.999] = np.nan
        cutouts.append(moved[2:-2, 2:-2])

    # The mean of the shifted cutouts, in the image coordinates of the middle stack
    with np.errstate(all='ignore'):
        mean = np.nanmean(np.array(cutouts), axis=0)
    img = np.full((stacks[m]['residual'].shape[0], stacks[m]['residual'].shape[1]), np.nan)
    xi, yi = int(round(xm)), int(round(ym))
    y_lo, x_lo = yi - half, xi - half
    if (y_lo < 0) or (x_lo < 0) or (yi + half >= img.shape[0]) or (xi + half >= img.shape[1]):
        return None
    img[y_lo:yi + half + 1, x_lo:xi + half + 1] = mean

    fit = fitSource(img, xm, ym, fit_half=fit_half, search_radius=search_radius, min_snr=min_snr)
    if fit is None:
        return None

    # The centre on the sky with the middle stack's platepar at its time, as a measurement, and the uncertainty
    #   of RA (times cos Dec) and Dec from the local scale of the plate
    pp, t = stacks[m]['platepar'], stacks[m]['t']
    jd = unixToJD(t)
    xs = np.array([fit['x'], fit['x'] + 1.0, fit['x']])
    ys = np.array([fit['y'], fit['y'], fit['y'] + 1.0])
    _, ra, dec, _ = xyToRaDecPP(np.full(3, jd), xs, ys, np.ones(3), pp, extinction_correction=False,
                                measurement=True, jd_time=True)
    ra, dec = np.array(ra), np.array(dec)
    cos_dec = np.cos(np.radians(dec[0]))
    d_ra = ((ra[1:] - ra[0] + 180)%360 - 180)*cos_dec*3600
    d_dec = (dec[1:] - dec[0])*3600
    sigma_ra = float(np.hypot(d_ra[0]*fit['sigma_x'], d_ra[1]*fit['sigma_y']))
    sigma_dec = float(np.hypot(d_dec[0]*fit['sigma_x'], d_dec[1]*fit['sigma_y']))
    ra_pred, dec_pred = path.at(t)

    return dict(time_utc=(datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=t)).isoformat(),
                time_unix=float(t), ra=float(ra[0]), dec=float(dec[0]), sigma_ra=sigma_ra, sigma_dec=sigma_dec,
                x=fit['x'], y=fit['y'], sigma_x=fit['sigma_x'], sigma_y=fit['sigma_y'], x_pred=xm, y_pred=ym,
                ra_pred=float(ra_pred[0]), dec_pred=float(dec_pred[0]), snr=fit['snr'], amplitude=fit['amplitude'],
                fwhm=fit['fwhm'], n_stacks=len(stacks), n_frames=int(sum(s['n_frames'] for s in stacks)),
                n_stars=int(np.median([s['n_stars'] for s in stacks])),
                aberration=int(bool(getattr(pp, 'aberration', False))))



def windowMotion(stacks, path):
    """ How far the predicted source moves against the stars over the stacks of a window (px): its predicted
        positions mapped onto the middle stack through the platepars.

    Arguments:
        stacks: [list] Recalibrated stacks of the window.
        path: [Path] Predicted path of the source.

    Return:
        [float] The largest distance between the mapped positions (px).
    """

    ref = stacks[len(stacks)//2]
    ref_jd = unixToJD(ref['t'])

    # The source's sky positions as stars seen at the middle stack's time: the motion against the stars
    ra, dec = path.at(np.array([s['t'] for s in stacks]))
    ok = np.isfinite(ra)
    if np.count_nonzero(ok) < 2:
        return 0.0
    x, y = measurementToXY(ra[ok], dec[ok], ref_jd, ref['platepar'])

    return float(np.max(np.hypot(x - x[0], y - y[0])))



def measurePath(source, path, platepar, config, catalog_stars=None, dark=None, flat_struct=None, mask=None,
                stack_frames=STACK_FRAMES, window_stacks=WINDOW_STACKS, group_stacks=GROUP_STACKS,
                fit_half=FIT_HALF, search_radius=SEARCH_RADIUS, min_snr=MIN_SNR):
    """ Measure a faint point source along its predicted path in the raw frames (see the description of the
        module).

    Arguments:
        source: [RawFrames] The raw frames (see RMS.Routines.TrackShape.openRawSource).
        path: [Path] Predicted path of the source.
        platepar: [Platepar] Platepar of the camera, whose pointing is recalibrated on every stack.
        config: [Config] Configuration of the camera.

    Keyword arguments:
        catalog_stars: [ndarray] Catalog stars (ra, dec, mag) of the recalibration, None (default) to load them as
            the processing does.
        dark: [ndarray] Dark frame, None (default) for none.
        flat_struct: [Flat struct] Flat field, None (default) for none.
        mask: [MaskStructure] Mask of the camera, None (default) for none.
        stack_frames: [int] Frames per stack. STACK_FRAMES by default.
        window_stacks: [int] Stacks per window of the image of the stars. WINDOW_STACKS by default.
        group_stacks: [int] Stacks per measurement. GROUP_STACKS by default.
        fit_half: [int] Half size of the box of the fit (px). FIT_HALF by default.
        search_radius: [float] Largest distance of the fitted centre from the prediction (px). SEARCH_RADIUS by
            default.
        min_snr: [float] Fewest signal-to-noise ratio. MIN_SNR by default.

    Return:
        (measurements, info): [tuple] The measurements (list of dicts, see CSV_COLUMNS) in time order, and a dict
            with the numbers of stacks ('stacks'), recalibrated stacks ('recalibrated'), windows ('windows') and
            windows in which the source doesn't move enough against the stars ('windows_static').
    """

    from RMS.Astrometry.AutoPlatepar import loadCatalogStars

    # The catalog of the recalibration as the processing loads it
    t0, t1 = path.times[0], path.times[-1]
    if catalog_stars is None:
        catalog_stars = loadCatalogStars(config, config.catalog_mag_limit, jd=unixToJD(0.5*(t0 + t1)))

    # The frames of the path, read a window at a time so that only the stacks of one window are kept
    indices, times = source.framesBetween(t0, t1)
    window_frames = window_stacks*stack_frames
    info = dict(stacks=0, recalibrated=0, windows=0, windows_static=0)

    width, height = int(platepar.X_res), int(platepar.Y_res)
    measurements = []
    for k in range(0, len(indices), window_frames):

        # The stacks of the window, recalibrated; those whose recalibration failed are left out
        stacks = stacksOfFrames(source, indices[k:k + window_frames], times[k:k + window_frames],
                                stack_frames=stack_frames, dark=dark, flat_struct=flat_struct)
        recalibrateStacks(stacks, platepar, config, catalog_stars, stack_frames=stack_frames, mask=mask)
        window = [s for s in stacks if s['platepar'] is not None]
        info['stacks'] += len(stacks)
        info['recalibrated'] += len(window)
        if len(window) < MIN_WINDOW_STACKS:
            continue
        info['windows'] += 1

        # The median of a window keeps a source which doesn't move enough against the stars
        if windowMotion(window, path) < MIN_WINDOW_MOTION*(2*fit_half + 1):
            info['windows_static'] += 1
            continue

        removeStars(window, width, height)
        for g in range(0, len(window), group_stacks):
            group = window[g:g + group_stacks]
            if len(group) < MIN_GROUP_STACKS:
                continue
            meas = measureGroup(group, path, fit_half=fit_half, search_radius=search_radius, min_snr=min_snr)
            if meas is not None:
                measurements.append(meas)

        # The images of the window are not needed any more
        for stack in stacks:
            stack.pop('img', None)
            stack.pop('residual', None)

    return measurements, info



### Command line ###

def readPath(path_file):
    """ Read a path from a CSV file with the columns time (UTC, ISO format, or Unix seconds), ra and dec (deg).

    Arguments:
        path_file: [str] Path to the file.

    Return:
        [Path] The path.
    """

    times, ra, dec = [], [], []
    with open(path_file) as f:
        for row in csv.DictReader(f):
            value = row['time'].strip()
            try:
                t = float(value)
            except ValueError:
                dt = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=datetime.timezone.utc)
                t = dt.timestamp()
            times.append(t)
            ra.append(float(row['ra']))
            dec.append(float(row['dec']))

    return Path(np.array(times), np.array(ra), np.array(dec))



def main(argv=None):
    """ Measure a source along the path of a CSV file in a raw recording and write the measurements to a CSV
        file. """

    arg_parser = argparse.ArgumentParser(description="Astrometry of a faint point source along a predicted path, "
                                         "measured on the raw frames.")
    arg_parser.add_argument('raw_path', help="Raw recording: a .vid or frame block file, a directory of them, or a "
                            "directory of image frames.")
    arg_parser.add_argument('path_file', help="CSV file of the path: columns time (UTC, ISO, or Unix seconds), ra, "
                            "dec (deg).")
    arg_parser.add_argument('-c', '--config', required=True, help="Configuration file of the camera.")
    arg_parser.add_argument('-p', '--platepar', required=True, help="Platepar of the camera.")
    arg_parser.add_argument('--dark', help="Dark frame (path), none by default.")
    arg_parser.add_argument('--flat', help="Flat field (path), none by default.")
    arg_parser.add_argument('--mask', help="Mask (path), none by default.")
    arg_parser.add_argument('-o', '--output', help="Output CSV file, the name of the path file with _measured by "
                            "default.")
    arg_parser.add_argument('--stack', type=int, default=STACK_FRAMES, help="Frames per stack.")
    arg_parser.add_argument('--group', type=int, default=GROUP_STACKS, help="Stacks per measurement.")
    arg_parser.add_argument('--window', type=int, default=WINDOW_STACKS, help="Stacks per window of the image of "
                            "the stars.")
    cml_args = arg_parser.parse_args(argv)

    import RMS.ConfigReader as cr
    from RMS.Formats.Platepar import Platepar
    from RMS.Routines.Image import loadDark, loadFlat
    from RMS.Routines.MaskImage import loadMask
    from RMS.Routines.TrackShape import openRawSource

    config = cr.parse(cml_args.config)
    platepar = Platepar()
    platepar.read(cml_args.platepar, use_flat=config.use_flat)
    path = readPath(cml_args.path_file)

    source = openRawSource(cml_args.raw_path, path.times[0], path.times[-1], config=config)
    if source is None:
        print("No raw frames of the path in {:s}".format(cml_args.raw_path))
        return 1

    try:
        # The dark and the flat as the input reads its frames (byteswapped for some inputs)
        handles = getattr(source, 'img_handles', None) or [None]
        byteswap = bool(getattr(handles[0], 'byteswap', False))
        dark = flat_struct = mask = None
        if cml_args.dark:
            dark = loadDark(*os.path.split(os.path.abspath(cml_args.dark)), byteswap=byteswap)
        if cml_args.flat:
            flat_struct = loadFlat(*os.path.split(os.path.abspath(cml_args.flat)), byteswap=byteswap)
        if cml_args.mask:
            mask = loadMask(cml_args.mask)

        measurements, info = measurePath(source, path, platepar, config, dark=dark, flat_struct=flat_struct,
                                         mask=mask, stack_frames=cml_args.stack, window_stacks=cml_args.window,
                                         group_stacks=cml_args.group)
    finally:
        source.close()

    output = cml_args.output or os.path.splitext(cml_args.path_file)[0] + '_measured.csv'
    with open(output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for meas in measurements:
            writer.writerow(meas)

    print("{:d} measurements ({:d} stacks, {:d} recalibrated, {:d} windows, {:d} without enough motion against the "
          "stars) written to {:s}".format(len(measurements), info['stacks'], info['recalibrated'], info['windows'],
                                         info['windows_static'], output))

    return 0



if __name__ == "__main__":

    import sys

    sys.exit(main())
