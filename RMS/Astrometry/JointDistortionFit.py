""" Joint fit of the lens distortion of a camera on the stars of many nights.

The platepar of a single night is fitted on the stars of that night alone, so the distortion is constrained only
where that night had stars and only as well as that night's stars measure it. The distortion of the lens doesn't
change between nights, but the pointing does: the camera drifts, is moved and is refocused over weeks and months.
Here one distortion is fitted on the stars of many chunks of many nights, while the pointing (and the scale) of
every chunk is fitted separately. The two are fitted in turn:

    1. Every chunk is recalibrated with the current distortion, as the processing does (only the pointing is
       refitted: ApplyRecalibrate.recalibrateSingleFF), and its stars are matched to the catalog.
    2. The distortion is fitted on the matched stars of all chunks together (Platepar.fitAstrometry). Each star is
       first moved into the image of a reference chunk (the one in the middle of the time span): by the change of
       the sidereal time between the two chunks, and by the difference between the pointing of its own chunk and
       of the reference chunk.

The recalibration of every chunk starts from the pointing of the platepar in the directory of its night (if there
is one, otherwise from the starting platepar), carried over to the current distortion (transferPointing). Every
N-th night can be held out of the fit, to check the fitted distortion on stars it wasn't fitted on.

The model of the fitted distortion is that of the starting platepar unless another one is asked for: another
distortion type, and for the radial types the decentering correction (Platepar.decentering_corr) switched on or
off. The model is applied to the platepar of the reference chunk before every fit of the distortion, as RMS sets a
model (AutoPlatepar.applyPlateparModel): the decentering flag first (Platepar.remapCoeffsForFlagChange, which adds
its coefficients as zeros or drops them, keeping the others), then the type. Another type is fitted from no
distortion, with the decentering correction if it is asked for (or if the starting platepar has it and the new type
is radial). The decentering correction applies only to the radial types.

The stars are those of the CALSTARS files, which RMS measures on dark-subtracted and flat-fielded images (if the
flat is used, see use_flat in the config). The photometry of the starting platepar (including its vignetting
coefficient) is kept as it is, it isn't fitted here.
"""

from __future__ import print_function, division, absolute_import

import argparse
import copy
import logging
import multiprocessing
import numbers
import os
import sys

import numpy as np
import scipy.optimize
from scipy.spatial import cKDTree

from RMS.Astrometry.ApplyAstrometry import getFOVSelectionRadius, raDecToXYPP, xyToRaDecPP
from RMS.Astrometry.ApplyRecalibrate import recalibrateSingleFF
from RMS.Astrometry.AutoPlatepar import loadCatalogStars
from RMS.Astrometry.Conversions import date2JD, jd2Date, JD2LST
from RMS.Astrometry.CyFunctions import subsetCatalog
from RMS.Formats import CALSTARS
from RMS.Formats.FFfile import getMiddleTimeFF
from RMS.Formats.Platepar import Platepar
from RMS.Logger import getLogger
from RMS.Math import angularSeparation


log = getLogger("rmslogger")


# Columns of the stars of a chunk (rows of a CALSTARS file)
STAR_Y, STAR_X, STAR_INTENSITY, STAR_AMPLITUDE, STAR_FWHM, STAR_BACKGROUND, STAR_SNR, STAR_SATURATED = range(8)

# The parameters of a platepar which describe where the camera looks (the pointing, its reference time, its
#   rotation and the scale, which changes slightly with focus), as opposed to the distortion of the lens
POINTING_PARAMETERS = ('RA_d', 'dec_d', 'JD', 'Ho', 'UT_corr', 'pos_angle_ref', 'rotation_from_horiz', 'az_centre',
                       'alt_centre', 'F_scale')

# The parameters of the photometry, which are kept from the starting platepar
PHOTOMETRY_PARAMETERS = ('mag_lev', 'mag_lev_stddev', 'vignetting_coeff', 'vignetting_fixed', 'extinction_scale')

# The pointing is carried over to another distortion on a grid of this many points along each axis, this far (px)
#   from the edges of the image. If the two platepars still differ by more than TRANSFER_MAX_RMS (px rms) on the
#   grid, they don't describe the same camera (or one of them is broken) and the pointing isn't carried over
TRANSFER_GRID = 15
TRANSFER_MARGIN = 12
TRANSFER_MAX_RMS = 1.0

# Catalog stars are matched only within this distance (px) from the edges of the image, and only if no other
#   catalog star is within this many match radii (a blended pair has no clean position)
MATCH_EDGE_MARGIN = 5
ISOLATION_RADII = 3

# Default match radius (px) for an image of 1280x720 px, scaled up with the image diagonal for larger images
MATCH_RADIUS = 1.0

# A chunk is used in the fit of the distortion only with at least this many matched stars
MIN_CHUNK_STARS = 15

# The stars of the fit are spread over the image: at most an equal number of stars in every cell of a grid of
#   this many cells along each axis
FIT_GRID = 16

# Only the stars which, moved to the reference chunk, are within this factor of the distance from the centre to the
#   corners of the image are fitted (the reverse polynomial folds stars far outside the image back into it)
FIT_FOV_FACTOR = 1.1

# The corners of the image are its outer share along both axes, and the radial profile of the residuals has this
#   many rings of equal width out to the corners
CORNER_SHARE = 0.15
RADIAL_BINS = 6
ZONE_GRID = 3
CORNER_NAMES = ('TL', 'TR', 'BL', 'BR')

# A warning is given if the residuals of the final distortion are larger than those of the best iteration by more
#   than this share
WORSE_SHARE = 0.02

# A chunk whose median residual is more than this many times the median of the chunks (a passing cloud, dew, a
#   chunk recalibrated to a wrong pointing) is left out of the next fit of the distortion, and so is a night whose
#   median residual is more than this many times the median of the nights (with at least 3 nights)
OUTLIER_CHUNK_FACTOR = 3.0
OUTLIER_NIGHT_FACTOR = 2.0

# The quality control of the fit (see qualityControl). It passes only if all of its checks pass:
#   - enough data: chunks recalibrated in the fit, held-out chunks, and stars of the fit in every zone of the grid
#       and in every corner (and the hours of clear sky and their span if asked for);
#   - better on the held-out stars: their median and rms residual no larger than with the starting platepars (by
#       at most a share), and the mean residual of no zone or corner larger by more than a fraction of a pixel;
#   - the judgement of the automatic calibration (AutoCalibration.judgeScore) passes on the held-out chunks;
#   - converged: the mapping of the image changed by less than a fraction of a pixel (rms) in the last fit, and
#       few chunks were left out as outliers;
#   - a sane model: the mapping changed from the starting platepar's by less than a share of the image diagonal,
#       it doesn't fold over anywhere in the image, and its forward and reverse directions agree.
QC_DEFAULTS = {
    'min_chunks': 20,
    'min_held_chunks': 5,
    'min_zone_stars': 10,
    'min_hours': None,
    'min_span_hours': None,
    'max_worse_share': 0.02,
    'max_zone_worse_px': 0.05,
    'judge': True,
    'judge_chunks': 12,
    'max_change_px': 0.05,
    'max_outlier_share': 0.1,
    'max_model_change_diag': 0.005,
    'max_roundtrip_px': 0.5,
}

# The two directions of the mapping are compared within this share of the distance from the centre to the corners:
#   in the outermost corners, where there are few stars, they disagree by a pixel or more also for good platepars
ROUNDTRIP_RADIUS = 0.9



def readPlatepar(path, use_flat):
    """ Read a platepar as the processing reads it, and check that it describes an image.

    Arguments:
        path: [str] Path of the platepar.
        use_flat: [bool] Whether the images are flat fielded (then a platepar without a vignetting coefficient gets
            none, see Platepar.addVignettingCoeff).

    Return:
        [Platepar] The platepar, None if the file doesn't exist, can't be read, or has no positive scale and image
            size.
    """

    if (path is None) or (not os.path.isfile(path)):
        return None

    # A file which can't be parsed as a platepar is none
    platepar = Platepar()
    try:
        if platepar.read(path, use_flat=use_flat) is False:
            return None

    except Exception as e:
        log.warning("The platepar {:s} can't be read: {:s}".format(path, repr(e)))
        return None

    # A platepar without a positive scale or image size maps nothing
    valid = [platepar.F_scale, platepar.X_res, platepar.Y_res]
    if not all(np.isfinite(value) and (value > 0) for value in valid):
        log.warning("The platepar {:s} has no positive scale and image size".format(path))
        return None

    return platepar


def transferPointing(platepar, pointing_platepar, n_grid=TRANSFER_GRID, margin=TRANSFER_MARGIN,
                     max_rms=TRANSFER_MAX_RMS):
    """ Carry the pointing of a platepar over to the distortion of another platepar of the same camera.

        Copying the pointing parameters is not enough: every distortion model has its own centre and offset terms,
        so the same pointing angles with another distortion put the stars pixels away. After the copy, the
        reference RA, Dec, position angle and scale are fitted by least squares, so that the platepar puts the sky
        positions of a grid of image points at the same image positions as the pointing platepar does at its
        reference time. Both directions are those from the sky to the image (which the recalibration and the
        matching of the stars use), so a platepar carried over to itself doesn't change.

    Arguments:
        platepar: [Platepar] Platepar with the distortion to keep (it is not changed).
        pointing_platepar: [Platepar] Platepar whose pointing is carried over.

    Keyword arguments:
        n_grid: [int] Number of grid points along each axis. TRANSFER_GRID by default.
        margin: [float] Distance of the grid from the edges of the image (px). TRANSFER_MARGIN by default.
        max_rms: [float] Largest rms difference of the two platepars on the grid (px) after the fit; a larger one
            raises a ValueError. TRANSFER_MAX_RMS by default, None not to check it.

    Return:
        (platepar, rms): [tuple] The platepar with the pointing carried over, and the rms distance (px) between
            the image positions of the grid's sky positions with the two platepars, which shows how different the
            two distortions are.
    """

    # Both platepars need a positive scale, which is fitted only within a factor of 2 of the pointing platepar's
    for pp in (platepar, pointing_platepar):
        if not (np.isfinite(pp.F_scale) and (pp.F_scale > 0)):
            raise ValueError("A platepar without a positive scale ({:.3f}) can't be used".format(pp.F_scale))

    # The distortion of the platepar with the pointing parameters of the pointing platepar
    transferred = copy.deepcopy(platepar)
    for name in POINTING_PARAMETERS:
        if hasattr(pointing_platepar, name):
            setattr(transferred, name, copy.deepcopy(getattr(pointing_platepar, name)))

    # Sky positions over the whole image: those of a grid of image points with the pointing platepar at its
    #   reference time, and their image positions with it
    jd = pointing_platepar.JD
    grid_x, grid_y = np.meshgrid(np.linspace(margin, platepar.X_res - margin, n_grid),
                                 np.linspace(margin, platepar.Y_res - margin, n_grid))
    grid_x, grid_y = grid_x.ravel(), grid_y.ravel()
    _, ra, dec, _ = xyToRaDecPP(len(grid_x)*[jd], grid_x, grid_y, len(grid_x)*[1], pointing_platepar,
                                extinction_correction=False, jd_time=True)
    ra, dec = np.array(ra), np.array(dec)
    x_pointing, y_pointing = raDecToXYPP(ra, dec, jd, pointing_platepar)

    keys = ('RA_d', 'dec_d', 'pos_angle_ref', 'F_scale')

    def _residuals(params):
        """ Distances (px) between the image positions of the sky positions with the two platepars.

        Arguments:
            params: [list] Reference RA, Dec, position angle (deg) and scale (px/deg) of the platepar.

        Return:
            [ndarray] Differences along x, then along y (px).
        """

        for key, value in zip(keys, params):
            setattr(transferred, key, value)
        x, y = raDecToXYPP(ra, dec, jd, transferred)

        return np.concatenate([x - x_pointing, y - y_pointing])

    # Fit the pointing, with steps of a similar size in all parameters and the scale kept positive
    p0 = [getattr(transferred, key) for key in keys]
    lower = [-np.inf, -np.inf, -np.inf, 0.5*p0[3]]
    upper = [np.inf, np.inf, np.inf, 2.0*p0[3]]
    fit = scipy.optimize.least_squares(_residuals, p0, x_scale=[1e-3, 1e-3, 1e-3, 1e-3*transferred.F_scale],
                                       bounds=(lower, upper))
    residuals = _residuals(fit.x)
    rms = float(np.sqrt(np.mean(residuals**2)))

    # Two platepars which still differ a lot don't describe the same camera
    if (not np.isfinite(rms)) or ((max_rms is not None) and (rms > max_rms)):
        raise ValueError("The pointing can't be carried over: the platepars differ by {:.2f} px rms (more than "
                         "{:.2f} px) after the fit".format(rms, max_rms if max_rms is not None else np.inf))

    # The reference alt/az and the rotation from the horizon follow the fitted pointing
    transferred.updateRefAltAz()

    return transferred, rms


def mappingDifference(platepar, other, n_grid=TRANSFER_GRID, margin=TRANSFER_MARGIN):
    """ How differently two platepars of the same camera map the image apart from the pointing: the pointing of the
        other platepar is carried over to the first one (transferPointing), and the image positions of the sky
        positions of a grid with the two are compared.

    Arguments:
        platepar: [Platepar] Platepar.
        other: [Platepar] Other platepar.

    Keyword arguments:
        n_grid: [int] Number of grid points along each axis. TRANSFER_GRID by default.
        margin: [float] Distance of the grid from the edges of the image (px). TRANSFER_MARGIN by default.

    Return:
        (rms, max): [tuple] The rms and the largest distance (px) between the image positions.
    """

    transferred, _ = transferPointing(platepar, other, n_grid=n_grid, margin=margin, max_rms=None)

    # The sky positions of a grid with the other platepar, and their image positions with both
    jd = other.JD
    grid_x, grid_y = np.meshgrid(np.linspace(margin, other.X_res - margin, n_grid),
                                 np.linspace(margin, other.Y_res - margin, n_grid))
    grid_x, grid_y = grid_x.ravel(), grid_y.ravel()
    _, ra, dec, _ = xyToRaDecPP(len(grid_x)*[jd], grid_x, grid_y, len(grid_x)*[1], other,
                                extinction_correction=False, jd_time=True)
    ra, dec = np.array(ra), np.array(dec)
    x_other, y_other = raDecToXYPP(ra, dec, jd, other)
    x, y = raDecToXYPP(ra, dec, jd, transferred)
    dist = np.hypot(x - x_other, y - y_other)

    return float(np.sqrt(np.mean(dist**2))), float(np.max(dist))


def modelSanity(platepar, n_grid=25):
    """ Whether the mapping of a platepar is usable over the whole image, also where it had no stars: it must not
        fold over (the sky direction of a grid of image positions turns the same way everywhere) and its two
        directions must agree (image to sky and back gives the same image position) within ROUNDTRIP_RADIUS.

    Arguments:
        platepar: [Platepar] Platepar.

    Keyword arguments:
        n_grid: [int] Number of grid points along each axis, from edge to edge. 25 by default.

    Return:
        (folds, roundtrip): [tuple] The number of grid cells which fold over, and the largest distance (px) between
            an image position within ROUNDTRIP_RADIUS and the image position of its sky position.
    """

    # The sky directions of a grid of image positions, edge to edge
    jd = platepar.JD
    xs, ys = np.meshgrid(np.linspace(0, platepar.X_res, n_grid), np.linspace(0, platepar.Y_res, n_grid))
    _, ra, dec, _ = xyToRaDecPP(xs.size*[jd], xs.ravel(), ys.ravel(), xs.size*[1], platepar,
                                extinction_correction=False, jd_time=True)
    ra, dec = np.radians(np.array(ra)), np.radians(np.array(dec))
    v = np.column_stack([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)]).reshape(n_grid, n_grid, 3)

    # The orientation of every cell: the triple product of its sides along x and y and its direction, whose sign
    #   is the same everywhere for a mapping which doesn't fold over
    side_x = v[:-1, 1:] - v[:-1, :-1]
    side_y = v[1:, :-1] - v[:-1, :-1]
    orientation = np.sum(np.cross(side_x, side_y)*v[:-1, :-1], axis=2)
    sign = np.sign(np.median(orientation))
    folds = int(np.count_nonzero(np.sign(orientation) != sign))

    # Back from the sky to the image
    x_back, y_back = raDecToXYPP(np.degrees(ra), np.degrees(dec), jd, platepar)
    radius = np.hypot(xs.ravel() - platepar.X_res/2.0, ys.ravel() - platepar.Y_res/2.0) \
        /np.hypot(platepar.X_res/2.0, platepar.Y_res/2.0)
    inner = radius <= ROUNDTRIP_RADIUS
    roundtrip = float(np.nanmax(np.hypot(x_back - xs.ravel(), y_back - ys.ravel())[inner]))
    if not np.isfinite(roundtrip):
        roundtrip = np.inf

    return folds, roundtrip


def _spreadKeys(keys, n):
    """ At most n of the keys of chunks, spread evenly over them (they are in time order).

    Arguments:
        keys: [list] Keys of the chunks.
        n: [int] Largest number of keys.

    Return:
        [list] The kept keys, in the same order.
    """

    if len(keys) <= n:
        return keys

    return [keys[i] for i in np.unique(np.round(np.linspace(0, len(keys) - 1, n)).astype(int))]


def readChunks(dir_path, config):
    """ Read the stars of every chunk (FF file or frame chunk) from the CALSTARS files in a directory.

    Arguments:
        dir_path: [str] Directory with the CALSTARS files.
        config: [Config] Configuration.

    Return:
        [list] Chunks as (name, jd, stars, ff_frames) sorted by time, jd the middle of the chunk (the time of the
            recalibration) and stars an array of the CALSTARS rows.
    """

    # The CALSTARS files of the directory
    names = sorted(name for name in os.listdir(dir_path) if name.startswith('CALSTARS') and name.endswith('.txt'))

    # The chunks of all files, every chunk once. A chunk whose name has no time is skipped
    chunks = {}
    for file_name in names:

        calstars = CALSTARS.readCALSTARS(dir_path, file_name)
        if not calstars:
            continue
        star_list, ff_frames = calstars

        for ff_name, stars in star_list:
            if ff_name in chunks:
                continue
            try:
                jd = date2JD(*getMiddleTimeFF(ff_name, config.fps, ret_milliseconds=True, ff_frames=ff_frames))
            except Exception:
                continue
            chunks[ff_name] = (ff_name, jd, np.array(stars, dtype=np.float64).reshape(-1, 8), ff_frames)

    return sorted(chunks.values(), key=lambda chunk: chunk[1])


def readNights(dir_paths, config, min_stars=0, snr_min=0):
    """ Read the chunks and the platepar of every directory (a night) and sort the nights by time.

    Arguments:
        dir_paths: [list] Directories with CALSTARS files and optionally a platepar (config.platepar_name).
        config: [Config] Configuration.

    Keyword arguments:
        min_stars: [int] Only the chunks with at least this many stars are kept. 0 by default.
        snr_min: [float] If positive, only the chunks with at least MIN_CHUNK_STARS stars of at least this SNR are
            kept. 0 by default.

    Return:
        [list] Nights as dictionaries with the directory ('dir'), the chunks ('chunks', see readChunks) and the
            platepar of the directory ('platepar', None if there is none or it can't be used), sorted by the time
            of the first chunk. Directories without chunks are left out.
    """

    nights = []
    for dir_path in dir_paths:

        # The chunks with enough stars (a night without them doesn't count, also not among the held-out nights)
        all_chunks = readChunks(dir_path, config)
        chunks = [chunk for chunk in all_chunks if len(chunk[2]) >= min_stars]
        if not chunks:
            log.warning("No chunks with at least {:d} stars in {:s}, the directory is not used".format(
                min_stars, dir_path))
            continue

        # The chunks with enough stars of the minimum SNR. The SNR of the CALSTARS files of older versions of RMS
        #   is on another scale (much lower), so their nights may have none
        if (snr_min is not None) and (snr_min > 0):
            snr = np.concatenate([chunk[2][:, STAR_SNR] for chunk in chunks])
            chunks = [chunk for chunk in chunks if np.count_nonzero(chunk[2][:, STAR_SNR] >= snr_min)
                      >= MIN_CHUNK_STARS]
            if not chunks:
                log.warning("No chunks with {:d} stars of SNR >= {:.1f} in {:s} (99th percentile of the SNR of its "
                            "stars {:.1f}; older CALSTARS files have the SNR on another scale), the directory is "
                            "not used".format(MIN_CHUNK_STARS, snr_min, dir_path, np.percentile(snr, 99)))
                continue

        # The platepar of the night, read as the processing reads it (a flat-fielded camera gets no default
        #   vignetting coefficient). One which can't be read is not used
        platepar = None
        platepar_path = os.path.join(dir_path, config.platepar_name)
        if os.path.isfile(platepar_path):
            platepar = readPlatepar(platepar_path, config.use_flat)
            if platepar is None:
                log.warning("The platepar {:s} can't be used, the night starts from the starting platepar".format(
                    platepar_path))

        nights.append({'dir': dir_path, 'chunks': chunks, 'platepar': platepar})

    return sorted(nights, key=lambda night: night['chunks'][0][1])


def heldOutNights(nights, holdout):
    """ The nights which are held out of the fit.

    Arguments:
        nights: [list] Nights sorted by time (see readNights).
        holdout: [tuple, list or None] (N, K): every N-th night from the K-th (counted from 0), or a list of the
            directories of the held-out nights, or None to hold out no night.

    Return:
        [set] Indices of the held-out nights.
    """

    if (holdout is None) or (len(holdout) == 0):
        return set()

    # Every N-th night from the K-th
    if (len(holdout) == 2) and all(isinstance(value, numbers.Integral) for value in holdout):
        n, k = int(holdout[0]), int(holdout[1])
        if (n < 1) or (k < 0):
            raise ValueError("The held-out nights must be every N-th from the K-th with N >= 1 and K >= 0")
        return set(range(k, len(nights), n))

    # The given directories, those which are not among the nights are reported
    def _norm(dir_path):
        """ The normalized absolute path of a directory.

        Arguments:
            dir_path: [str] Directory.

        Return:
            [str] Its normalized absolute path.
        """

        return os.path.normpath(os.path.abspath(dir_path))

    held_dirs = set(_norm(dir_path) for dir_path in holdout)
    night_dirs = set(_norm(night['dir']) for night in nights)
    for dir_path in sorted(held_dirs - night_dirs):
        log.warning("The held-out directory {:s} is not among the nights with usable chunks".format(dir_path))

    return set(i for i, night in enumerate(nights) if _norm(night['dir']) in held_dirs)


def matchStars(platepar, jd, stars, catalog_stars, match_radius, lim_mag):
    """ Match the stars of a chunk one to one to isolated catalog stars.

    Arguments:
        platepar: [Platepar] Platepar recalibrated on the chunk.
        jd: [float] Julian date of the chunk.
        stars: [ndarray] Stars of the chunk (CALSTARS rows).
        catalog_stars: [ndarray] Catalog stars (ra, dec, mag).
        match_radius: [float] A star is matched to a catalog star within this distance (px).
        lim_mag: [float] Limiting magnitude of the catalog stars.

    Return:
        [ndarray] The matched stars, one row per star: x, y, intensity, SNR, catalog RA, Dec, magnitude, and the
            catalog star's position on the image (x, y).
    """

    # Without stars nothing is matched
    empty = np.zeros((0, 9))
    stars = np.asarray(stars, dtype=np.float64).reshape(-1, 8)
    if len(stars) == 0:
        return empty

    # The catalog stars in the field of view (above the horizon)
    fov_radius = getFOVSelectionRadius(platepar)
    _, ra_c, dec_c, _ = xyToRaDecPP([jd2Date(jd)], [platepar.X_res/2], [platepar.Y_res/2], [1], platepar,
                                    extinction_correction=False)
    _, catalog_fov = subsetCatalog(np.asarray(catalog_stars[:, :3], dtype=np.float64), ra_c[0], dec_c[0], jd,
                                   platepar.lat, platepar.lon, fov_radius, lim_mag)
    if len(catalog_fov) == 0:
        return empty

    # Their positions on the image, those well inside it
    x_cat, y_cat = raDecToXYPP(catalog_fov[:, 0], catalog_fov[:, 1], jd, platepar)
    inside = (x_cat > MATCH_EDGE_MARGIN) & (x_cat < platepar.X_res - MATCH_EDGE_MARGIN) \
        & (y_cat > MATCH_EDGE_MARGIN) & (y_cat < platepar.Y_res - MATCH_EDGE_MARGIN)
    if np.count_nonzero(inside) < 2:
        return empty
    x_cat, y_cat, catalog_fov = x_cat[inside], y_cat[inside], catalog_fov[inside]

    # The isolated catalog stars: no other catalog star nearby
    tree = cKDTree(np.column_stack([x_cat, y_cat]))
    isolated = np.array([len(neighbours) == 1 for neighbours in
                         tree.query_ball_point(np.column_stack([x_cat, y_cat]), ISOLATION_RADII*match_radius)])

    # The nearest catalog star of every star within the match radius, if it is isolated
    dist, idx = tree.query(np.column_stack([stars[:, STAR_X], stars[:, STAR_Y]]), k=1,
                           distance_upper_bound=match_radius)
    matched = np.isfinite(dist)
    matched[matched] &= isolated[idx[matched]]

    # Only catalog stars matched by a single star
    counts = np.bincount(idx[matched], minlength=len(catalog_fov))
    matched[matched] &= counts[idx[matched]] == 1

    # The matched stars with their catalog stars
    j = idx[matched]
    s = stars[matched]

    return np.column_stack([s[:, STAR_X], s[:, STAR_Y], s[:, STAR_INTENSITY], s[:, STAR_SNR], catalog_fov[j, 0],
                            catalog_fov[j, 1], catalog_fov[j, 2], x_cat[j], y_cat[j]])


# The data the recalibration workers share, set once per worker (see _setWorkerData)
_worker_data = {}


def _setWorkerData(catalog_stars, match_catalog_stars, config, match_radius, lim_mag):
    """ Set the data which the recalibration of the chunks shares (see _recalibrateWorker).

    Arguments:
        catalog_stars: [ndarray] Catalog stars of the recalibration (ra, dec, mag).
        match_catalog_stars: [ndarray] Catalog stars of the matching (ra, dec, mag).
        config: [Config] Configuration.
        match_radius: [float] Match radius of the stars (px).
        lim_mag: [float] Limiting magnitude of the matched catalog stars.

    Return:
        None
    """

    _worker_data.clear()
    _worker_data.update(catalog_stars=catalog_stars, match_catalog_stars=match_catalog_stars, config=config,
                        match_radius=match_radius, lim_mag=lim_mag)


def _initPoolWorker(*args):
    """ Initialize a process of the pool of workers: set its data (see _setWorkerData) and keep the log of its many
        recalibrations quiet (only in the worker process).

    Arguments:
        args: [tuple] Arguments of _setWorkerData.

    Return:
        None
    """

    _setWorkerData(*args)
    logging.getLogger("rmslogger").setLevel(logging.WARNING)


def _recalibrateWorker(task):
    """ Recalibrate a chunk and match its stars (see recalibrateChunks).

    Arguments:
        task: [tuple] (key, platepar, name, jd, stars, ff_frames).

    Return:
        (key, platepar, matched): [tuple] The recalibrated platepar and the matched stars (see matchStars), or
            (key, None, None) if the recalibration failed.
    """

    key, platepar, name, jd, stars, ff_frames = task
    data = _worker_data

    # The pointing recalibrated on the stars of the chunk as the processing does
    recalibrated, ok = recalibrateSingleFF(platepar, name, stars, data['catalog_stars'], data['config'],
                                           ff_frames=ff_frames)
    if not ok:
        return key, None, None

    # The matched stars are kept separately, the platepar doesn't need its list of the recalibration's stars
    recalibrated.star_list = []
    matched = matchStars(recalibrated, jd, stars, data['match_catalog_stars'], data['match_radius'],
                         data['lim_mag'])

    return key, recalibrated, matched


def recalibrateChunks(tasks, pool=None):
    """ Recalibrate chunks and match their stars, in parallel if a pool of workers is given.

    Arguments:
        tasks: [list] Tasks (key, platepar to start from, chunk name, jd, stars, frames of the chunk).

    Keyword arguments:
        pool: [multiprocessing.Pool] Pool of workers initialized by _initPoolWorker. None by default, in which case
            the chunks are recalibrated here (_setWorkerData must have been called).

    Return:
        [dict] For every key the recalibrated platepar and the matched stars (None, None if the recalibration
            failed).
    """

    # Recalibrated here: the log of the recalibrations is kept quiet only while they run
    if pool is None:
        rms_log = logging.getLogger("rmslogger")
        previous_level = rms_log.level
        rms_log.setLevel(logging.WARNING)
        try:
            results = [_recalibrateWorker(task) for task in tasks]
        finally:
            rms_log.setLevel(previous_level)

    else:
        results = pool.map(_recalibrateWorker, tasks, chunksize=4)

    return {key: (platepar, matched) for key, platepar, matched in results}


def residualStats(matched, scale, width, height):
    """ Statistics of the residuals of matched stars, in arcseconds.

    Arguments:
        matched: [ndarray] Matched stars (see matchStars).
        scale: [ndarray] Plate scale at every star (arcsec/px).
        width: [int] Image width (px).
        height: [int] Image height (px).

    Return:
        [dict] The number of stars ('n'), the median distance ('median'), the rms distance ('rms'), the standard
            deviations along x and y ('std_x', 'std_y'), the mean radial residual (outwards positive) in rings of
            the distance from the centre ('radial': list of (inner, outer radius in px, mean, n)), the mean residual
            in a 3x3 grid of zones ('zones': rows from the top, list of (dx, dy, n)) and in the four corners
            ('corners': name -> (dx, dy, n)). None without stars.
    """

    if len(matched) == 0:
        return None

    x, y = matched[:, 0], matched[:, 1]
    dx, dy = (matched[:, 0] - matched[:, 7])*scale, (matched[:, 1] - matched[:, 8])*scale
    dist = np.hypot(dx, dy)

    def _mean(values, selection):
        """ Mean of the selected values.

        Arguments:
            values: [ndarray] Values.
            selection: [ndarray] Boolean selection of the values.

        Return:
            [float] The mean, nan if none is selected.
        """

        return float(np.mean(values[selection])) if np.any(selection) else np.nan

    # The mean radial residual in rings out to the corners
    cx, cy = width/2.0, height/2.0
    r = np.hypot(x - cx, y - cy)
    radial = ((x - cx)*dx + (y - cy)*dy)/np.maximum(r, 1e-6)
    edges = np.linspace(0, np.hypot(cx, cy), RADIAL_BINS + 1)
    rings = []
    for inner, outer in zip(edges[:-1], edges[1:]):
        in_ring = (r >= inner) & (r < outer)
        rings.append((float(inner), float(outer), _mean(radial, in_ring), int(np.count_nonzero(in_ring))))

    # The mean residual in the zones of a grid, rows from the top
    zones = []
    for j in range(ZONE_GRID):
        row = []
        for i in range(ZONE_GRID):
            in_zone = (x >= i*width/ZONE_GRID) & (x < (i + 1)*width/ZONE_GRID) & (y >= j*height/ZONE_GRID) \
                & (y < (j + 1)*height/ZONE_GRID)
            row.append((_mean(dx, in_zone), _mean(dy, in_zone), int(np.count_nonzero(in_zone))))
        zones.append(row)

    # The mean residual in the corners
    left, right = x < CORNER_SHARE*width, x > (1 - CORNER_SHARE)*width
    top, bottom = y < CORNER_SHARE*height, y > (1 - CORNER_SHARE)*height
    corners = {}
    for name, in_corner in zip(CORNER_NAMES, (top & left, top & right, bottom & left, bottom & right)):
        corners[name] = (_mean(dx, in_corner), _mean(dy, in_corner), int(np.count_nonzero(in_corner)))

    return {'n': len(matched), 'median': float(np.median(dist)), 'rms': float(np.sqrt(np.mean(dist**2))),
            'std_x': float(np.std(dx)), 'std_y': float(np.std(dy)), 'radial': rings, 'zones': zones,
            'corners': corners}


def _goodStars(matched, snr_min):
    """ The matched stars of at least the minimum SNR.

    Arguments:
        matched: [ndarray] Matched stars (see matchStars), or None.
        snr_min: [float] Minimum SNR, not applied if not positive.

    Return:
        [ndarray] The matched stars of at least the minimum SNR (None if none were given).
    """

    if (matched is None) or (not snr_min) or (snr_min <= 0):
        return matched

    return matched[matched[:, 3] >= snr_min]


def _collect(results, keys, snr_min):
    """ The matched stars of the recalibrated chunks with the plate scale at every star (arcsec/px).

    Arguments:
        results: [dict] Results of recalibrateChunks.
        keys: [list] Keys of the chunks to collect.
        snr_min: [float] Minimum SNR of the stars.

    Return:
        (matched, scale, n_recalibrated): [tuple] The matched stars, the scale and the number of recalibrated
            chunks.
    """

    matched, scale, n_recalibrated = [np.zeros((0, 9))], [np.zeros(0)], 0
    for key in keys:
        platepar, stars = results.get(key, (None, None))
        if platepar is None:
            continue
        n_recalibrated += 1
        stars = _goodStars(stars, snr_min)
        matched.append(stars)
        scale.append(np.full(len(stars), 3600.0/platepar.F_scale))

    return np.concatenate(matched), np.concatenate(scale), n_recalibrated


def setDistortionModel(platepar, distortion_type, decentering):
    """ Set the distortion model of a platepar, as RMS sets a model (AutoPlatepar.applyPlateparModel): the
        decentering flag first, which moves the coefficients to its layout, then the distortion type.

        With the same type the coefficients are kept, and the coefficients of the decentering correction are added
        as zeros or dropped. Another type starts from no distortion.

    Arguments:
        platepar: [Platepar] Platepar, changed in place.
        distortion_type: [str] Distortion type (see Platepar.setDistortionType).
        decentering: [bool] Decentering correction of the radial distortion types.

    Return:
        None
    """

    # The decentering correction exists only for the radial types
    if decentering and (not distortion_type.startswith("radial")):
        raise ValueError("The decentering correction applies only to the radial distortion types, not to "
                         "{:s}".format(distortion_type))

    # Another type from no distortion, with the flag set before the type gives the length of the coefficients
    if platepar.distortion_type != distortion_type:
        platepar.remapCoeffsForFlagChange('decentering_corr', decentering)
        platepar.setDistortionType(distortion_type, reset_params=True)

    # The same type with the decentering switched on or off, the other coefficients kept
    elif bool(getattr(platepar, 'decentering_corr', False)) != bool(decentering):
        platepar.remapCoeffsForFlagChange('decentering_corr', decentering)


def fitDistortion(platepars, matched_stars, jds, distortion_type, max_stars, seed=0, decentering=False):
    """ Fit one distortion on the matched stars of many recalibrated chunks.

        Every star is moved into the image of the reference chunk (the one in the middle of the time span): its
        catalog position is moved by the change of the sidereal time between its chunk and the reference chunk (the
        camera is fixed to the ground), and its measured position by the difference between the positions of its
        catalog star with the platepar of its chunk and with the platepar of the reference chunk. The difference
        of the measured position from the model of its own chunk, which the distortion has to explain, is kept.

    Arguments:
        platepars: [dict] Recalibrated platepar of every chunk (key -> platepar).
        matched_stars: [dict] Matched stars of every chunk (key -> matched stars, see matchStars).
        jds: [dict] Julian date of every chunk.
        distortion_type: [str] Distortion type of the fit (see Platepar.setDistortionType).
        max_stars: [int] Maximum number of stars of the fit, spread over the image.

    Keyword arguments:
        seed: [int] Seed of the random choice of the stars. 0 by default.
        decentering: [bool] Fit the decentering correction (radial types only, see setDistortionModel). False by
            default.

    Return:
        (platepar, n_stars, reference_key): [tuple] The fitted platepar with the pointing of the reference chunk,
            the number of fitted stars and the key of the reference chunk.
    """

    # The reference chunk is the one in the middle of the time span
    keys = sorted(platepars, key=lambda key: jds[key])
    reference_key = keys[len(keys)//2]
    pp_ref, jd_ref = platepars[reference_key], jds[reference_key]
    lst_ref = JD2LST(jd_ref, pp_ref.lon)[0]
    width, height = pp_ref.X_res, pp_ref.Y_res

    # Every star moved into the image of the reference chunk
    x_all, y_all, ra_all, dec_all, mag_all, intensity_all = [], [], [], [], [], []
    for key in keys:

        stars = matched_stars[key]
        ra, dec = stars[:, 4], stars[:, 5]

        # The same direction relative to the ground at the time of the reference chunk
        ra_ref = (ra + lst_ref - JD2LST(jds[key], pp_ref.lon)[0])%360

        # The measured position moved by the difference of the two models
        x_chunk, y_chunk = raDecToXYPP(ra, dec, jds[key], platepars[key])
        x_ref, y_ref = raDecToXYPP(ra_ref, dec, jd_ref, pp_ref)
        x_all.append(stars[:, 0] + x_ref - x_chunk)
        y_all.append(stars[:, 1] + y_ref - y_chunk)

        ra_all.append(ra_ref)
        dec_all.append(dec)
        mag_all.append(stars[:, 6])
        intensity_all.append(stars[:, 2])

    x, y, ra, dec, mag, intensity = [np.concatenate(values) for values in (x_all, y_all, ra_all, dec_all, mag_all,
                                                                           intensity_all)]

    # Only the stars inside the image and within the field of view at the reference time
    _, ra_c, dec_c, _ = xyToRaDecPP([jd2Date(jd_ref)], [width/2.0], [height/2.0], [1], pp_ref,
                                    extinction_correction=False)
    separation = np.degrees(angularSeparation(np.radians(ra), np.radians(dec), np.radians(ra_c[0]),
                                              np.radians(dec_c[0])))
    inside = (separation < FIT_FOV_FACTOR*np.hypot(width, height)/2.0/pp_ref.F_scale) & (x >= 0) & (x < width) \
        & (y >= 0) & (y < height)

    # At most max_stars stars, an equal number from every cell of a grid over the image
    idx = np.nonzero(inside)[0]
    cell = np.floor(x[idx]/(width/FIT_GRID)).astype(int)*FIT_GRID + np.floor(y[idx]/(height/FIT_GRID)).astype(int)
    cells = np.unique(cell)
    per_cell = max(1, max_stars//max(len(cells), 1))
    rng = np.random.RandomState(seed)
    chosen = np.concatenate([rng.choice(idx[cell == c], min(per_cell, np.count_nonzero(cell == c)), replace=False)
                             for c in cells]) if len(cells) else np.zeros(0, dtype=int)

    # The distortion fitted with the pointing of the reference chunk as the start, in the asked model. A different
    #   distortion type starts from no distortion
    platepar = copy.deepcopy(pp_ref)
    setDistortionModel(platepar, distortion_type, decentering)
    platepar.fitAstrometry(jd_ref, np.column_stack([x[chosen], y[chosen], intensity[chosen]]),
                           np.column_stack([ra[chosen], dec[chosen], mag[chosen]]))
    platepar.auto_recalibrated = False
    platepar.star_list = []

    return platepar, len(chosen), reference_key


def jointDistortionFit(dir_paths, platepar, config, distortion_type=None, holdout=None, n_iter=4, snr_min=10.0,
                       max_stars=20000, chunk_step=2, match_radius=None, lim_mag=None, nproc=None, decentering=None,
                       holdout_chunks=None, max_chunks=None, qc=True, qc_options=None):
    """ Fit one distortion on the stars of many nights, with the pointing of every chunk fitted separately (see
        the description of the module).

    Arguments:
        dir_paths: [list] Directories of the nights, with CALSTARS files and optionally a platepar
            (config.platepar_name), whose pointing is the start of the recalibration of the chunks of the night.
        platepar: [Platepar] Starting platepar, its distortion is the start of the fit and its pointing the start
            of the nights without a platepar.
        config: [Config] Configuration.

    Keyword arguments:
        distortion_type: [str] Distortion type of the fit (see Platepar.setDistortionType). None by default, in
            which case the type of the starting platepar is kept. A different type is fitted from no distortion.
        holdout: [tuple or list] Nights held out of the fit (see heldOutNights). None by default.
        n_iter: [int] Number of fits of the distortion, at least 1. 4 by default.
        snr_min: [float] Minimum SNR of the stars (not applied if not positive). 10 by default.
        max_stars: [int] Maximum number of stars in a fit of the distortion. 20000 by default.
        chunk_step: [int] Every chunk_step-th chunk of the nights of the fit is used. 2 by default.
        match_radius: [float] Match radius of the stars (px). None by default, in which case MATCH_RADIUS is
            scaled to the image size.
        lim_mag: [float] Limiting magnitude of the catalog stars which are matched (a deeper catalog leaves out
            more stars with a neighbour). None by default, in which case config.catalog_mag_limit is used.
        nproc: [int] Number of parallel workers. None by default, in which case all cores are used.
        decentering: [bool] Fit the decentering correction of the radial distortion types (True) or not (False).
            None by default, in which case the setting of the starting platepar is kept if the fitted type is
            radial (otherwise it is off). True with a type which isn't radial raises a ValueError.
        holdout_chunks: [int] Also hold out every N-th chunk of every night of the fit (from the N//2-th), so the
            fit can be checked within the nights it is fitted on (e.g. a single night). None by default.
        max_chunks: [int] At most this many chunks in the fit and this many held out, spread evenly in time over
            them (after chunk_step), which bounds the time of the fit. None by default (no limit).
        qc: [bool] Run the quality control of the fit (see qualityControl), its verdict is in the report. True by
            default.
        qc_options: [dict] Limits of the quality control which replace those of QC_DEFAULTS. None by default.

    Return:
        (platepar, report): [tuple] The fitted platepar (the distortion of the last fit with the pointing of the
            reference chunk, the photometry of the starting platepar), and the report of the fit as a dictionary:
            the residual statistics (see residualStats) of the chunks of the fit after every recalibration
            ('iterations'), of every night of the fit with the final distortion ('night_stats') and of the
            held-out nights with their own platepars and with the fitted distortion ('held_out_stats'), the
            matched stars behind them ('stars'), and the verdict of the quality control ('qc', see
            qualityControl). formatReport writes it as text.
    """

    if n_iter < 1:
        raise ValueError("At least one fit of the distortion is needed (n_iter = {:d})".format(n_iter))

    # The distortion type is the one of the starting platepar unless another one is asked for, which is then fitted
    #   from no distortion
    if distortion_type is None:
        distortion_type = platepar.distortion_type
    if distortion_type != platepar.distortion_type:
        log.info("The distortion type changes from {:s} (starting platepar) to {:s}, which is fitted from no "
                 "distortion".format(platepar.distortion_type, distortion_type))

    # The decentering correction is that of the starting platepar unless it is switched on or off, and exists only
    #   for the radial types
    start_decentering = bool(getattr(platepar, 'decentering_corr', False))
    radial = distortion_type.startswith("radial")
    if decentering is None:
        decentering = start_decentering and radial
    decentering = bool(decentering)
    if decentering and (not radial):
        raise ValueError("The decentering correction applies only to the radial distortion types, not to "
                         "{:s}".format(distortion_type))
    if decentering != start_decentering:
        log.info("The decentering correction is switched {:s} for the fit (starting platepar: {:s})".format(
            'on' if decentering else 'off', 'on' if start_decentering else 'off'))

    # The catalog depth and number of workers not given are those of the config and the machine
    if lim_mag is None:
        lim_mag = config.catalog_mag_limit

    if nproc is None:
        nproc = multiprocessing.cpu_count()

    # The match radius scaled to the image size as in the recalibration
    if match_radius is None:
        match_radius = MATCH_RADIUS*max(1.0, np.hypot(platepar.X_res, platepar.Y_res)/np.hypot(1280, 720))

    # The nights in time order, with the chunks with enough stars to be recalibrated and to be used in the fit, and
    #   the held-out nights
    min_stars = max(config.ff_min_stars, config.min_matched_stars)
    nights = readNights(dir_paths, config, min_stars=min_stars, snr_min=snr_min)
    if not nights:
        raise ValueError("No chunks with at least {:d} stars (and {:d} of SNR >= {:.1f}) in the given "
                         "directories".format(min_stars, MIN_CHUNK_STARS, snr_min))
    held = heldOutNights(nights, holdout)

    # The platepars of the nights whose pointing can't be carried over to the starting distortion (another camera,
    #   or a broken platepar) are not used
    for night in nights:
        if night['platepar'] is not None:
            try:
                transferPointing(platepar, night['platepar'])
            except ValueError as e:
                log.warning("The platepar of {:s} is not used: {:s}".format(night['dir'], str(e)))
                night['platepar'] = None

    # Every chunk_step-th chunk of the nights of the fit, all chunks of the held-out nights, and every
    #   holdout_chunks-th chunk of the nights of the fit if asked for. The key of a chunk is (night index, chunk
    #   index)
    usable = [(n, c) for n, night in enumerate(nights) for c in range(len(night['chunks']))]
    held_chunk_keys = set()
    if holdout_chunks:
        if holdout_chunks < 2:
            raise ValueError("Every N-th chunk can be held out only with N >= 2 (holdout_chunks = {:d})".format(
                holdout_chunks))
        held_chunk_keys = set(key for key in usable if (key[0] not in held)
                              and (key[1]%holdout_chunks == holdout_chunks//2))
    train_keys = [key for key in usable if (key[0] not in held) and (key not in held_chunk_keys)][::max(1, chunk_step)]
    held_keys = [key for key in usable if (key[0] in held) or (key in held_chunk_keys)]
    if max_chunks:
        train_keys, held_keys = _spreadKeys(train_keys, max_chunks), _spreadKeys(held_keys, max_chunks)
    if not train_keys:
        raise ValueError("No chunks in the nights of the fit")

    log.info("Joint distortion fit: {:d} nights ({:d} held out), {:d} chunks in the fit, {:d} held out ({:d} of "
             "them from the nights of the fit)".format(len(nights), len(held), len(train_keys), len(held_keys),
                                                       len(held_chunk_keys)))

    def _chunk(key):
        """ The chunk of a key.

        Arguments:
            key: [tuple] (night index, chunk index).

        Return:
            [tuple] The chunk (see readChunks).
        """

        return nights[key[0]]['chunks'][key[1]]

    # The time of every chunk
    jds = {key: _chunk(key)[1] for key in usable}

    # The catalog of the recalibration (as in the processing) and of the matching, with the proper motion at the
    #   middle of the time span
    jd_mid = float(np.median([jds[key] for key in usable]))
    catalog_stars = loadCatalogStars(config, config.catalog_mag_limit, jd=jd_mid)
    match_catalog_stars = catalog_stars if lim_mag == config.catalog_mag_limit \
        else loadCatalogStars(config, lim_mag, jd=jd_mid)

    def _startOf(night, distortion):
        """ The start of the recalibration of the chunks of a night: the distortion with the pointing of the
            platepar of the night, or of the starting platepar (also if the night's can't be carried over).

        Arguments:
            night: [dict] Night (see readNights).
            distortion: [Platepar] Current distortion.

        Return:
            [Platepar] The start of the recalibration.
        """

        if night['platepar'] is not None:
            try:
                return transferPointing(distortion, night['platepar'])[0]
            except ValueError as e:
                log.warning("The pointing of the platepar of {:s} can't be carried over to the current distortion, "
                            "the starting platepar's is used: {:s}".format(night['dir'], str(e)))

        return transferPointing(distortion, platepar)[0]

    def _tasks(keys, starts):
        """ The recalibration tasks of chunks, each starting from the platepar of its night.

        Arguments:
            keys: [list] Keys of the chunks.
            starts: [dict] Platepar to start from for every night index.

        Return:
            [list] Tasks of recalibrateChunks.
        """

        return [(key, starts[key[0]], _chunk(key)[0], _chunk(key)[1], _chunk(key)[2], _chunk(key)[3])
                for key in keys]

    # The workers of the recalibration, which share the catalogs and the config. Without a pool the data is set
    #   here (the log is kept quiet only while recalibrating, see recalibrateChunks)
    init_args = (catalog_stars, match_catalog_stars, config, match_radius, lim_mag)
    pool = None
    if nproc > 1:
        pool = multiprocessing.Pool(nproc, initializer=_initPoolWorker, initargs=init_args)
    else:
        _setWorkerData(*init_args)

    report = {'nights': [night['dir'] for night in nights], 'held_out': sorted(nights[n]['dir'] for n in held),
              'distortion_type': distortion_type, 'decentering': decentering, 'match_radius': match_radius,
              'lim_mag': lim_mag,
              'snr_min': snr_min, 'chunks': len(train_keys), 'held_chunks': len(held_keys), 'iterations': []}
    width, height = platepar.X_res, platepar.Y_res

    try:

        # Recalibrate the chunks with the current distortion and fit the distortion on their stars, in turn. The
        #   last round only recalibrates, for the residuals of the final distortion
        distortion = copy.deepcopy(platepar)
        for iteration in range(n_iter + 1):

            # Every chunk recalibrated from the pointing of its night carried over to the current distortion
            starts = {n: _startOf(nights[n], distortion) for n in set(key[0] for key in train_keys)}
            results = recalibrateChunks(_tasks(train_keys, starts), pool=pool)

            # The residuals of the stars of the chunks of the fit
            matched, scale, n_recalibrated = _collect(results, train_keys, snr_min)
            entry = {'iteration': iteration, 'recalibrated': n_recalibrated,
                     'stats': residualStats(matched, scale, width, height)}
            report['iterations'].append(entry)
            log.info("Iteration {:d}: {:s}".format(iteration, _statsLine(entry['stats'], n_recalibrated,
                                                                            len(train_keys))))

            # The nights whose chunks give no stars for the fit are reported once
            if iteration == 0:
                _warnNightsWithoutStars(nights, train_keys, results, snr_min)

            if iteration == n_iter:
                break

            # The chunks and nights whose residuals are far larger than the others' are left out of this fit
            outliers = _outliers(results, train_keys, snr_min)
            entry['outlier_chunks'] = len(outliers)

            # The chunks with enough matched stars for the fit of the distortion
            fit_platepars, fit_stars = {}, {}
            for key in train_keys:
                if key in outliers:
                    continue
                pp_chunk, stars = results[key]
                if pp_chunk is None:
                    continue
                stars = _goodStars(stars, snr_min)
                if len(stars) >= MIN_CHUNK_STARS:
                    fit_platepars[key], fit_stars[key] = pp_chunk, stars
            if not fit_platepars:
                raise ValueError("No recalibrated chunk has {:d} matched stars of SNR >= {:.1f} for the fit".format(
                    MIN_CHUNK_STARS, snr_min))

            previous = distortion
            distortion, n_fit, reference_key = fitDistortion(fit_platepars, fit_stars, jds, distortion_type,
                                                             max_stars, decentering=decentering)
            entry['fit_chunks'], entry['fit_stars'] = len(fit_platepars), n_fit

            # How much the mapping of the image changed in this fit (px), which shows whether the fit converged
            try:
                entry['change_px'] = mappingDifference(distortion, previous)
            except ValueError:
                entry['change_px'] = (np.inf, np.inf)
            report['reference'] = (nights[reference_key[0]]['dir'], _chunk(reference_key)[0])

        # The final distortion is kept, but a warning is given if an earlier one fitted the stars clearly better
        _warnIfWorse(report['iterations'])

        # The chunks of the final distortion which are outliers
        report['iterations'][-1]['outlier_chunks'] = len(_outliers(results, train_keys, snr_min))

        # The residuals of every night of the fit with the final distortion
        report['night_stats'] = []
        for n in sorted(set(key[0] for key in train_keys)):
            keys = [key for key in train_keys if key[0] == n]
            matched, scale, n_recalibrated = _collect(results, keys, snr_min)
            report['night_stats'].append((nights[n]['dir'], len(keys), n_recalibrated,
                                          residualStats(matched, scale, width, height)))
        report['stars'] = {'fit': _collect(results, train_keys, snr_min)[0]}

        # The held-out nights recalibrated from their own platepars (as they were) and from the fitted distortion
        #   with their pointing carried over
        if held_keys:

            held_nights = sorted(set(key[0] for key in held_keys))
            own = dict((n, nights[n]['platepar'] if nights[n]['platepar'] is not None else platepar)
                       for n in held_nights)
            fitted = dict((n, _startOf(nights[n], distortion)) for n in held_nights)
            report['held_out_stats'] = {}
            for label, starts in (('start', own), ('fitted', fitted)):

                results = recalibrateChunks(_tasks(held_keys, starts), pool=pool)
                matched, scale, n_recalibrated = _collect(results, held_keys, snr_min)
                per_night = []
                for n in held_nights:
                    keys = [key for key in held_keys if key[0] == n]
                    m, s, r = _collect(results, keys, snr_min)
                    per_night.append((nights[n]['dir'], len(keys), r, residualStats(m, s, width, height)))
                report['held_out_stats'][label] = {'chunks': len(held_keys), 'recalibrated': n_recalibrated,
                                                   'stats': residualStats(matched, scale, width, height),
                                                   'nights': per_night}
                report['stars']['held_out_' + label] = matched
                log.info("Held out ({:s}): {:s}".format(label, _statsLine(
                    report['held_out_stats'][label]['stats'], n_recalibrated, len(held_keys))))

        # The quality control of the fit. The judgement of the automatic calibration is made on the held-out
        #   chunks of every night, from the fitted distortion with the pointing of the night
        if qc:
            options = dict(QC_DEFAULTS)
            options.update(qc_options or {})
            judge_groups = []
            if options['judge'] and held_keys:
                for n in sorted(set(key[0] for key in held_keys)):
                    keys = [key for key in held_keys if key[0] == n]
                    judge_groups.append((nights[n]['dir'], _startOf(nights[n], distortion),
                                         [_chunk(key) for key in keys]))
            chunk_hours = [len(night['chunks'])*night['chunks'][0][3]/config.fps/3600.0 for night in nights]
            spans = [(night['chunks'][-1][1] - night['chunks'][0][1])*24.0 for night in nights]
            report['qc'] = qualityControl(distortion, platepar, report, config, options, judge_groups=judge_groups,
                                          clear_hours=float(np.sum(chunk_hours)), longest_span_hours=max(spans))
            log.info("Quality control: {:s}".format(_qcLine(report['qc'])))

    # On an error the workers are stopped right away
    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
            pool = None
        raise

    # The workers are stopped and the shared data of the recalibration in this process is released
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        _worker_data.clear()

    # The photometry of the starting platepar is kept, it isn't fitted here
    for name in PHOTOMETRY_PARAMETERS:
        if hasattr(platepar, name):
            setattr(distortion, name, copy.deepcopy(getattr(platepar, name)))

    return distortion, report


def _outliers(results, keys, snr_min):
    """ The chunks whose residuals are far larger than those of the others, and the chunks of the nights whose
        residuals are far larger than those of the other nights (see OUTLIER_CHUNK_FACTOR, OUTLIER_NIGHT_FACTOR).

    Arguments:
        results: [dict] Results of recalibrateChunks.
        keys: [list] Keys of the chunks (night index, chunk index).
        snr_min: [float] Minimum SNR of the stars.

    Return:
        [set] Keys of the outlier chunks.
    """

    # The median distance (px) of the matched stars of every recalibrated chunk with enough of them
    medians = {}
    for key in keys:
        platepar, stars = results.get(key, (None, None))
        if platepar is None:
            continue
        stars = _goodStars(stars, snr_min)
        if len(stars) >= MIN_CHUNK_STARS:
            medians[key] = float(np.median(np.hypot(stars[:, 0] - stars[:, 7], stars[:, 1] - stars[:, 8])))

    if len(medians) < 3:
        return set()

    # The chunks far above the median of the chunks
    typical = np.median(list(medians.values()))
    outliers = set(key for key, value in medians.items() if value > OUTLIER_CHUNK_FACTOR*typical)

    # The nights far above the median of the nights
    night_medians = {}
    for n in set(key[0] for key in medians):
        night_medians[n] = np.median([value for key, value in medians.items() if key[0] == n])
    if len(night_medians) >= 3:
        typical_night = np.median(list(night_medians.values()))
        for n, value in night_medians.items():
            if value > OUTLIER_NIGHT_FACTOR*typical_night:
                outliers |= set(key for key in medians if key[0] == n)

    return outliers


def qualityControl(fitted, start, report, config, options, judge_groups=(), clear_hours=None,
                   longest_span_hours=None):
    """ The quality control of a joint fit: whether the fitted distortion can be installed (see QC_DEFAULTS for
        the checks).

    Arguments:
        fitted: [Platepar] Fitted platepar.
        start: [Platepar] Starting platepar.
        report: [dict] Report of jointDistortionFit (iterations, held-out statistics).
        config: [Config] Configuration.
        options: [dict] Limits of the checks (see QC_DEFAULTS).

    Keyword arguments:
        judge_groups: [list] Held-out chunks to judge the fitted distortion on, as (name, platepar to start from,
            chunks as in readChunks), e.g. one group per night. Empty by default (not judged).
        clear_hours: [float] Hours of the chunks of the nights. None by default.
        longest_span_hours: [float] Hours from the first to the last chunk of the longest night. None by default.

    Return:
        [dict] 'ok' (True if all checks pass), 'checks' (list of dictionaries with the 'name' of the check, whether
            it is 'ok' and the 'value' as text) and 'failed' (names of the failed checks).
    """

    checks = []

    def _check(name, ok, value):
        checks.append({'name': name, 'ok': bool(ok), 'value': value})

    last = report['iterations'][-1]
    stats = last['stats']
    scale = 3600.0/fitted.F_scale

    # Enough data: chunks of the fit and held out, stars in every zone and corner, clear sky
    _check('chunks of the fit', last['recalibrated'] >= options['min_chunks'],
           '{:d} recalibrated (min {:d})'.format(last['recalibrated'], options['min_chunks']))
    held = report.get('held_out_stats')
    n_held = held['fitted']['recalibrated'] if held else 0
    _check('held-out chunks', n_held >= options['min_held_chunks'],
           '{:d} recalibrated (min {:d})'.format(n_held, options['min_held_chunks']))
    if stats is None:
        _check('stars in every zone and corner', False, 'no stars')
    else:
        zone_counts = [zone[2] for row in stats['zones'] for zone in row]
        corner_counts = [stats['corners'][name][2] for name in CORNER_NAMES]
        fewest = min(zone_counts + corner_counts)
        _check('stars in every zone and corner', fewest >= options['min_zone_stars'],
               'fewest {:d} (min {:d})'.format(fewest, options['min_zone_stars']))
    if options['min_hours'] is not None:
        _check('hours of clear sky', (clear_hours or 0) >= options['min_hours'],
               '{:.1f} h (min {:.1f} h)'.format(clear_hours or 0, options['min_hours']))
    if options['min_span_hours'] is not None:
        _check('span of the night', (longest_span_hours or 0) >= options['min_span_hours'],
               '{:.1f} h (min {:.1f} h)'.format(longest_span_hours or 0, options['min_span_hours']))

    # Better on the held-out stars than the starting platepars: overall and in every zone and corner
    if (not held) or (held['start']['stats'] is None) or (held['fitted']['stats'] is None):
        _check('held-out stars not worse', False, 'no held-out stars')
    else:
        before, after = held['start']['stats'], held['fitted']['stats']
        limit = 1 + options['max_worse_share']
        _check('held-out stars not worse', (after['median'] <= limit*before['median'])
               and (after['rms'] <= limit*before['rms']),
               'median {:.2f}" -> {:.2f}", rms {:.2f}" -> {:.2f}"'.format(before['median'], after['median'],
                                                                         before['rms'], after['rms']))
        tolerance = options['max_zone_worse_px']*scale
        worst, worst_name = -np.inf, ''
        zones_before = [zone for row in before['zones'] for zone in row] \
            + [before['corners'][name] for name in CORNER_NAMES]
        zones_after = [zone for row in after['zones'] for zone in row] \
            + [after['corners'][name] for name in CORNER_NAMES]
        names = ['zone {:d},{:d}'.format(j, i) for j in range(ZONE_GRID) for i in range(ZONE_GRID)] \
            + list(CORNER_NAMES)
        for name, (bx, by, bn), (ax, ay, an) in zip(names, zones_before, zones_after):
            if min(bn, an) < options['min_zone_stars']:
                continue
            change = np.hypot(ax, ay) - np.hypot(bx, by)
            if change > worst:
                worst, worst_name = change, name
        _check('no held-out zone worse', worst <= tolerance,
               'largest change of a zone {:+.2f}" in {:s} (max {:+.2f}")'.format(worst, worst_name, tolerance)
               if np.isfinite(worst) else 'no zone with enough stars')

    # The judgement of the automatic calibration on the held-out chunks of every night
    if options['judge']:
        from RMS.Astrometry import AutoCalibration

        failed = []
        for name, platepar, group_chunks in judge_groups:
            chunks = AutoCalibration.spreadChunks([chunk[:3] for chunk in group_chunks], options['judge_chunks'])
            catalog_stars = AutoCalibration._loadCatalog(config, chunks[len(chunks)//2][1])
            score = AutoCalibration.scorePlatepar(platepar, chunks, catalog_stars, config,
                                                  ff_frames=group_chunks[0][3])
            ok, reasons = AutoCalibration.judgeScore(score)
            if not ok:
                failed.append('{:s}: {:s}'.format(os.path.basename(os.path.normpath(name)), '; '.join(reasons)))
        _check('judged on the held-out chunks', bool(judge_groups) and not failed,
               ('{:d} nights pass'.format(len(judge_groups)) if not failed else ' | '.join(failed))
               if judge_groups else 'no held-out chunks')

    # Converged, with few outlier chunks
    changes = [entry['change_px'][0] for entry in report['iterations'] if 'change_px' in entry]
    change = changes[-1] if changes else np.inf
    _check('converged', change <= options['max_change_px'],
           'last fit changed the mapping by {:.3f} px rms (max {:.3f})'.format(change, options['max_change_px']))
    outlier_share = last.get('outlier_chunks', 0)/max(report['chunks'], 1)
    _check('few outlier chunks', outlier_share <= options['max_outlier_share'],
           '{:.0f}% (max {:.0f}%)'.format(100*outlier_share, 100*options['max_outlier_share']))

    # A sane model over the whole image
    try:
        _, change_max = mappingDifference(fitted, start)
    except ValueError:
        change_max = np.inf
    limit = options['max_model_change_diag']*np.hypot(fitted.X_res, fitted.Y_res)
    _check('change from the starting platepar', change_max <= limit,
           'largest {:.2f} px (max {:.2f} px)'.format(change_max, limit))
    folds, roundtrip = modelSanity(fitted)
    _check('no fold-over', folds == 0, '{:d} folded cells'.format(folds))
    _check('forward and reverse agree', roundtrip <= options['max_roundtrip_px'],
           'largest difference {:.3f} px (max {:.3f})'.format(roundtrip, options['max_roundtrip_px']))

    failed_names = [check['name'] for check in checks if not check['ok']]

    return {'ok': not failed_names, 'checks': checks, 'failed': failed_names}


def _qcLine(qc):
    """ One line of the verdict of the quality control.

    Arguments:
        qc: [dict] Verdict (see qualityControl).

    Return:
        [str] The line.
    """

    if qc['ok']:
        return 'PASS'

    return 'FAIL ({:s})'.format(', '.join(qc['failed']))


def _warnNightsWithoutStars(nights, keys, results, snr_min):
    """ Warn about the nights whose recalibrated chunks give no matched stars of the minimum SNR.

    Arguments:
        nights: [list] Nights (see readNights).
        keys: [list] Keys of the chunks (night index, chunk index).
        results: [dict] Results of recalibrateChunks.
        snr_min: [float] Minimum SNR of the stars.

    Return:
        None
    """

    for n in sorted(set(key[0] for key in keys)):
        night_keys = [key for key in keys if key[0] == n]
        matched, _, n_recalibrated = _collect(results, night_keys, snr_min)
        if len(matched) > 0:
            continue

        # Why the night has no stars: no chunk was recalibrated, or no matched star has the SNR
        if n_recalibrated == 0:
            reason = "none of its {:d} chunks was recalibrated".format(len(night_keys))
        else:
            reason = "no matched star of its {:d} recalibrated chunks has SNR >= {:.1f} (older CALSTARS files " \
                "have the SNR on another scale)".format(n_recalibrated, snr_min)
        log.warning("No usable stars in {:s}: {:s}".format(nights[n]['dir'], reason))


def _warnIfWorse(iterations):
    """ Warn if the residuals of the final distortion are larger than those of an earlier one by more than
        WORSE_SHARE.

    Arguments:
        iterations: [list] Entries of the iterations of the report of jointDistortionFit.

    Return:
        [bool] True if the warning was given.
    """

    rms = [entry['stats']['rms'] if entry['stats'] is not None else np.inf for entry in iterations]
    best = int(np.argmin(rms))
    if rms[-1] > (1 + WORSE_SHARE)*rms[best]:
        log.warning("The residuals of the final distortion ({:.2f}\") are larger than those of iteration {:d} "
                    "({:.2f}\"), the fit may not converge".format(rms[-1], best, rms[best]))
        return True

    return False


def _statsLine(stats, n_recalibrated, n_chunks):
    """ One line of the residual statistics, for the log and the report.

    Arguments:
        stats: [dict] Residual statistics (see residualStats), None without stars.
        n_recalibrated: [int] Number of recalibrated chunks.
        n_chunks: [int] Number of chunks.

    Return:
        [str] The line.
    """

    if stats is None:
        return "recalibrated {:d}/{:d}, no usable stars".format(n_recalibrated, n_chunks)

    return ("recalibrated {:d}/{:d}, {:d} stars, median {:.2f}\", rms {:.2f}\" (x {:.2f}\", y {:.2f}\"), corners "
            "{:s}".format(n_recalibrated, n_chunks, stats['n'], stats['median'], stats['rms'], stats['std_x'],
                          stats['std_y'], _cornersLine(stats)))


def _cornersLine(stats):
    """ The size of the mean residual in every corner.

    Arguments:
        stats: [dict] Residual statistics (see residualStats).

    Return:
        [str] The size of the mean residual in every corner (arcsec).
    """

    return ' '.join('{:s} {:.2f}"'.format(name, np.hypot(*stats['corners'][name][:2])) for name in CORNER_NAMES)


def _statsBlock(stats, indent='    '):
    """ The radial profile, zones and corners of the residual statistics, as lines of text.

    Arguments:
        stats: [dict] Residual statistics (see residualStats), None without stars.

    Keyword arguments:
        indent: [str] Indentation of the lines. Four spaces by default.

    Return:
        [list] The lines.
    """

    if stats is None:
        return [indent + 'no usable stars']

    lines = [indent + 'radial (mean, outwards +): ' + '  '.join(
        '{:.0f}-{:.0f}px {:+.2f}"'.format(inner, outer, mean) for inner, outer, mean, _ in stats['radial'])]
    lines.append(indent + 'zones (mean dx, dy), rows from the top:')
    for row in stats['zones']:
        lines.append(indent + '  ' + '  '.join('({:+.2f}, {:+.2f}) n {:d}'.format(*zone) for zone in row))
    lines.append(indent + 'corners (mean dx, dy, size): ' + '  '.join(
        '{:s} ({:+.2f}, {:+.2f}) {:.2f}" n {:d}'.format(name, dx, dy, np.hypot(dx, dy), n)
        for name, (dx, dy, n) in ((name, stats['corners'][name]) for name in CORNER_NAMES)))

    return lines


def formatReport(report):
    """ The report of a joint distortion fit as text.

    Arguments:
        report: [dict] Report of jointDistortionFit.

    Return:
        [str] The report.
    """

    lines = ['Joint distortion fit, distortion type {:s}{:s}'.format(
        report['distortion_type'], ' with the decentering correction' if report.get('decentering') else '')]
    lines.append('{:d} nights, {:d} held out, {:d} chunks in the fit; match radius {:.2f} px, catalog to mag '
                 '{:.1f}, SNR >= {:.1f}'.format(len(report['nights']), len(report['held_out']), report['chunks'],
                                                report['match_radius'], report['lim_mag'], report['snr_min']))
    if 'reference' in report:
        lines.append('Reference chunk: {:s} {:s}'.format(*report['reference']))
    lines.append('All residuals in arcseconds, after the recalibration of the pointing of every chunk.')

    # The residuals of the chunks of the fit after every fit of the distortion
    lines.append('')
    lines.append('Residuals of the chunks of the fit (iteration 0: the starting distortion):')
    for entry in report['iterations']:
        fit = ''
        if 'fit_chunks' in entry:
            fit = ' -> fit on {:d} stars of {:d} chunks'.format(entry['fit_stars'], entry['fit_chunks'])
            if entry.get('outlier_chunks'):
                fit += ' ({:d} outlier chunks left out)'.format(entry['outlier_chunks'])
            if 'change_px' in entry:
                fit += ', mapping changed by {:.3f} px rms'.format(entry['change_px'][0])
        lines.append('  iteration {:d}: {:s}{:s}'.format(entry['iteration'], _statsLine(
            entry['stats'], entry['recalibrated'], report['chunks']), fit))
    lines.append('  final distortion:')
    lines += _statsBlock(report['iterations'][-1]['stats'])

    # Every night of the fit
    lines.append('')
    lines.append('Nights of the fit, final distortion:')
    for dir_path, n_chunks, n_recalibrated, stats in report.get('night_stats', []):
        lines.append('  {:s}: {:s}'.format(dir_path, _statsLine(stats, n_recalibrated, n_chunks)))

    # The held-out nights, with their own platepars and with the fitted distortion
    if report.get('held_out_stats'):
        for label, title in (('start', 'own platepars (or the starting platepar)'),
                             ('fitted', 'fitted distortion with their pointing')):
            held = report['held_out_stats'][label]
            lines.append('')
            lines.append('Held-out nights, {:s}:'.format(title))
            lines.append('  all: ' + _statsLine(held['stats'], held['recalibrated'], held['chunks']))
            lines += _statsBlock(held['stats'])
            for dir_path, n_chunks, n_recalibrated, stats in held['nights']:
                lines.append('  {:s}: {:s}'.format(dir_path, _statsLine(stats, n_recalibrated, n_chunks)))

    # The quality control
    if report.get('qc'):
        lines.append('')
        lines.append('Quality control: {:s}'.format(_qcLine(report['qc'])))
        for check in report['qc']['checks']:
            lines.append('  [{:s}] {:s}: {:s}'.format('ok' if check['ok'] else 'FAIL', check['name'],
                                                    check['value']))

    return '\n'.join(lines) + '\n'




def _parseHoldout(value):
    """ The held-out nights of the command line.

    Arguments:
        value: [str] N:K (every N-th night from the K-th), or none.

    Return:
        [tuple] (N, K), None for none.
    """

    if value.lower() == 'none':
        return None

    try:
        n, k = (int(v) for v in value.split(':'))
    except ValueError:
        raise argparse.ArgumentTypeError("The held-out nights must be N:K, e.g. 4:1, or none")

    if (n < 2) or (k < 0) or (k >= n):
        raise argparse.ArgumentTypeError("The held-out nights must be N:K with N >= 2 and 0 <= K < N")

    return (n, k)


def _positiveInt(value):
    """ A positive integer of the command line.

    Arguments:
        value: [str] The value.

    Return:
        [int] The integer.
    """

    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("{:s} is not an integer".format(value))

    if number < 1:
        raise argparse.ArgumentTypeError("{:s} must be at least 1".format(value))

    return number


def main(argv=None):
    """ The command line interface: fit the distortion and write the platepar and the report.

    Keyword arguments:
        argv: [list] Command line arguments. None by default, in which case those of the program are used.

    Return:
        [int] Exit code: 0 on success, 1 on an error (which is logged), 2 if the fit fails its quality control (the
            platepar and the report are written all the same).
    """

    import RMS.ConfigReader as cr

    # COMMAND LINE ARGUMENTS

    arg_parser = argparse.ArgumentParser(description="Fit one lens distortion on the stars of many nights of a "
                                         "camera, with the pointing of every chunk fitted separately.")

    arg_parser.add_argument('dir_paths', nargs='+', metavar='DIR_PATH', type=str,
                            help="Directories of the nights, with CALSTARS files and optionally a platepar, whose "
                            "pointing is the start of the chunks of the night.")

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str,
                            help="Path to a config file which will be used instead of the default one.")

    arg_parser.add_argument('-p', '--platepar', metavar='PLATEPAR', type=str, required=True,
                            help="Starting platepar: the start of the distortion, and the pointing of the nights "
                            "without a platepar.")

    arg_parser.add_argument('-o', '--out', metavar='PLATEPAR_OUT', type=str, required=True,
                            help="Path of the fitted platepar. The report is written next to it (_report.txt).")

    arg_parser.add_argument('-t', '--type', metavar='DISTORTION_TYPE', type=str,
                            help="Distortion type of the fit, e.g. poly3+radial5. The type of the starting "
                            "platepar by default; another type is fitted from no distortion.")

    holdout_group = arg_parser.add_mutually_exclusive_group()

    holdout_group.add_argument('--holdout', metavar='N:K', type=_parseHoldout, default=None,
                               help="Hold out every N-th night (in time order) from the K-th (from 0) to check the "
                               "fit, e.g. 4:1. None by default.")

    holdout_group.add_argument('--holdoutdirs', nargs='+', metavar='DIR_PATH', type=str,
                               help="Hold out the nights of these directories (some of the given ones) instead.")

    arg_parser.add_argument('--holdoutchunks', metavar='N', type=int,
                            help="Also hold out every N-th chunk of every night of the fit (N >= 2), e.g. to check a "
                            "fit of a single night. None by default.")

    arg_parser.add_argument('--maxchunks', metavar='N', type=_positiveInt,
                            help="At most N chunks in the fit and N held out, spread evenly in time. No limit by "
                            "default.")

    arg_parser.add_argument('--noqc', action='store_true',
                            help="Don't run the quality control of the fit.")

    arg_parser.add_argument('--iter', metavar='N_ITER', type=_positiveInt, default=4,
                            help="Number of fits of the distortion (at least 1). 4 by default.")

    arg_parser.add_argument('--snrmin', metavar='SNR', type=float, default=10.0,
                            help="Minimum SNR of the stars. 10 by default.")

    arg_parser.add_argument('--maxstars', metavar='N_STARS', type=_positiveInt, default=20000,
                            help="Maximum number of stars in a fit of the distortion. 20000 by default.")

    arg_parser.add_argument('--step', metavar='STEP', type=_positiveInt, default=2,
                            help="Use every STEP-th chunk of the nights of the fit. 2 by default.")

    arg_parser.add_argument('--radius', metavar='RADIUS_PX', type=float,
                            help="Match radius of the stars (px). 1 px for 1280x720, scaled with the image "
                            "diagonal, by default.")

    arg_parser.add_argument('--limmag', metavar='LIM_MAG', type=float,
                            help="Limiting magnitude of the matched catalog stars. catalog_mag_limit of the config "
                            "by default.")

    arg_parser.add_argument('--pointing', metavar='PLATEPAR', type=str,
                            help="Write the fitted distortion with the pointing of this platepar (e.g. the latest "
                            "one of the camera) instead of the pointing of the reference chunk.")

    arg_parser.add_argument('--decentering', choices=('on', 'off'),
                            help="Fit the decentering correction of the radial distortion types (on) or not (off). "
                            "The setting of the starting platepar by default (off for a type which isn't radial). "
                            "It applies to the fitted type, also with -t: e.g. -t radial7-odd --decentering on "
                            "fits radial7-odd with the correction from no distortion. On with a type which isn't "
                            "radial is an error.")

    arg_parser.add_argument('--nproc', metavar='N_PROC', type=_positiveInt,
                            help="Number of parallel workers. All cores by default.")

    cml_args = arg_parser.parse_args(argv)

    #########################

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, cml_args.dir_paths[0])

    # Print the progress
    getLogger("rmslogger", level="INFO")

    # The starting platepar and the platepar of the pointing, read as the processing reads them, are checked before
    #   the fit
    platepar = readPlatepar(cml_args.platepar, config.use_flat)
    if platepar is None:
        log.error("The starting platepar {:s} can't be read".format(cml_args.platepar))
        return 1

    pointing = None
    if cml_args.pointing:
        pointing = readPlatepar(cml_args.pointing, config.use_flat)
        if pointing is None:
            log.error("The platepar of the pointing {:s} can't be read".format(cml_args.pointing))
            return 1

    # The fit, whose errors (e.g. no usable stars) end the program with a message
    try:
        fitted, fit_report = jointDistortionFit(cml_args.dir_paths, platepar, config,
                                                distortion_type=cml_args.type,
                                                holdout=(cml_args.holdoutdirs or cml_args.holdout),
                                                n_iter=cml_args.iter, snr_min=cml_args.snrmin,
                                                max_stars=cml_args.maxstars, chunk_step=cml_args.step,
                                                match_radius=cml_args.radius, lim_mag=cml_args.limmag,
                                                nproc=cml_args.nproc,
                                                decentering=(None if cml_args.decentering is None
                                                             else cml_args.decentering == 'on'),
                                                holdout_chunks=cml_args.holdoutchunks,
                                                max_chunks=cml_args.maxchunks, qc=(not cml_args.noqc))

        # The fitted distortion with the pointing of the given platepar
        if pointing is not None:
            fitted, rms = transferPointing(fitted, pointing)
            log.info("Pointing of {:s} carried over, the two differ by {:.3f} px rms".format(cml_args.pointing,
                                                                                          rms))

    except ValueError as e:
        log.error("Joint distortion fit failed: {:s}".format(str(e)))
        return 1

    # Save the platepar and the report
    fitted.write(cml_args.out)
    report_path = os.path.splitext(cml_args.out)[0] + '_report.txt'
    with open(report_path, 'w') as f:
        f.write(formatReport(fit_report))

    print(formatReport(fit_report))
    print("Platepar written to {:s}, report to {:s}".format(cml_args.out, report_path))

    # A fit which fails its quality control ends with its own exit code
    if fit_report.get('qc') and (not fit_report['qc']['ok']):
        log.warning("The fit fails its quality control: {:s}".format(', '.join(fit_report['qc']['failed'])))
        return 2

    return 0


if __name__ == "__main__":

    sys.exit(main())
