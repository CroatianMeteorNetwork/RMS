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

The stars are those of the CALSTARS files, which RMS measures on dark-subtracted and flat-fielded images (if the
flat is used, see use_flat in the config). The photometry of the starting platepar (including its vignetting
coefficient) is kept as it is, it isn't fitted here.
"""

from __future__ import print_function, division, absolute_import

import argparse
import copy
import logging
import multiprocessing
import os

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
#   from the edges of the image
TRANSFER_GRID = 15
TRANSFER_MARGIN = 12

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



def transferPointing(platepar, pointing_platepar, n_grid=TRANSFER_GRID, margin=TRANSFER_MARGIN):
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

    Return:
        (platepar, rms): [tuple] The platepar with the pointing carried over, and the rms distance (px) between
            the image positions of the grid's sky positions with the two platepars, which shows how different the
            two distortions are.
    """

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
        """ Distances (px) between the image positions of the sky positions with the two platepars, with the
            given pointing parameters. """

        for key, value in zip(keys, params):
            setattr(transferred, key, value)
        x, y = raDecToXYPP(ra, dec, jd, transferred)

        return np.concatenate([x - x_pointing, y - y_pointing])

    # Fit the pointing, with steps of a similar size in all parameters
    p0 = [getattr(transferred, key) for key in keys]
    fit = scipy.optimize.least_squares(_residuals, p0, x_scale=[1e-3, 1e-3, 1e-3, 1e-3*transferred.F_scale])
    residuals = _residuals(fit.x)

    # The reference alt/az and the rotation from the horizon follow the fitted pointing
    transferred.updateRefAltAz()

    return transferred, float(np.sqrt(np.mean(residuals**2)))


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


def readNights(dir_paths, config, min_stars=0):
    """ Read the chunks and the platepar of every directory (a night) and sort the nights by time.

    Arguments:
        dir_paths: [list] Directories with CALSTARS files and optionally a platepar (config.platepar_name).
        config: [Config] Configuration.

    Keyword arguments:
        min_stars: [int] Only the chunks with at least this many stars are kept. 0 by default.

    Return:
        [list] Nights as dictionaries with the directory ('dir'), the chunks ('chunks', see readChunks) and the
            platepar of the directory ('platepar', None if there is none), sorted by the time of the first chunk.
            Directories without chunks are left out.
    """

    nights = []
    for dir_path in dir_paths:

        # The chunks with enough stars (a night without them doesn't count, also not among the held-out nights)
        chunks = [chunk for chunk in readChunks(dir_path, config) if len(chunk[2]) >= min_stars]
        if not chunks:
            log.warning("No chunks with at least {:d} stars in {:s}, the directory is not used".format(
                min_stars, dir_path))
            continue

        # The platepar of the night, read as the processing reads it (a flat-fielded camera gets no default
        #   vignetting coefficient)
        platepar = None
        platepar_path = os.path.join(dir_path, config.platepar_name)
        if os.path.isfile(platepar_path):
            platepar = Platepar()
            if platepar.read(platepar_path, use_flat=config.use_flat) is False:
                platepar = None

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

    if not holdout:
        return set()

    # Every N-th night from the K-th
    if (len(holdout) == 2) and all(isinstance(value, int) for value in holdout):
        n, k = holdout
        return set(range(k, len(nights), n))

    # The given directories
    held_dirs = set(os.path.normpath(os.path.abspath(dir_path)) for dir_path in holdout)
    return set(i for i, night in enumerate(nights)
               if os.path.normpath(os.path.abspath(night['dir'])) in held_dirs)


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


# The data the recalibration workers share, set once per worker (see _initWorker)
_worker_data = {}


def _initWorker(catalog_stars, match_catalog_stars, config, match_radius, lim_mag):
    """ Set the data of a recalibration worker, and keep the log of the many recalibrations quiet. """

    _worker_data.update(catalog_stars=catalog_stars, match_catalog_stars=match_catalog_stars, config=config,
                        match_radius=match_radius, lim_mag=lim_mag)
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
        pool: [multiprocessing.Pool] Pool of workers initialized by _initWorker. None by default, in which case the
            chunks are recalibrated here (_initWorker must have been called).

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
        """ Mean of the selected values, nan if none is selected. """

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
    """ The matched stars of at least the minimum SNR (all of them if snr_min is not positive). """

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


def fitDistortion(platepars, matched_stars, jds, distortion_type, max_stars, seed=0):
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

    # The distortion fitted with the pointing of the reference chunk as the start. A different distortion type
    #   starts from no distortion
    platepar = copy.deepcopy(pp_ref)
    if platepar.distortion_type != distortion_type:
        platepar.setDistortionType(distortion_type, reset_params=True)
    platepar.fitAstrometry(jd_ref, np.column_stack([x[chosen], y[chosen], intensity[chosen]]),
                           np.column_stack([ra[chosen], dec[chosen], mag[chosen]]))
    platepar.auto_recalibrated = False
    platepar.star_list = []

    return platepar, len(chosen), reference_key


def jointDistortionFit(dir_paths, platepar, config, distortion_type=None, holdout=None, n_iter=4, snr_min=10.0,
                       max_stars=20000, chunk_step=2, match_radius=None, lim_mag=None, nproc=None):
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
            which case config.platepar_distortion_type is used.
        holdout: [tuple or list] Nights held out of the fit (see heldOutNights). None by default.
        n_iter: [int] Number of fits of the distortion. 4 by default.
        snr_min: [float] Minimum SNR of the stars (not applied if not positive). 10 by default.
        max_stars: [int] Maximum number of stars in a fit of the distortion. 20000 by default.
        chunk_step: [int] Every chunk_step-th chunk of the nights of the fit is used. 2 by default.
        match_radius: [float] Match radius of the stars (px). None by default, in which case MATCH_RADIUS is
            scaled to the image size.
        lim_mag: [float] Limiting magnitude of the catalog stars which are matched (a deeper catalog leaves out
            more stars with a neighbour). None by default, in which case config.catalog_mag_limit is used.
        nproc: [int] Number of parallel workers. None by default, in which case all cores are used.

    Return:
        (platepar, report): [tuple] The fitted platepar (the distortion with the pointing of the reference chunk,
            the photometry of the starting platepar), and the report of the fit as a dictionary: the residual
            statistics (see residualStats) of the chunks of the fit after every recalibration ('iterations'), of
            every night of the fit with the final distortion ('night_stats') and of the held-out nights with their
            own platepars and with the fitted distortion ('held_out_stats'), and the matched stars behind them
            ('stars'). formatReport writes it as text.
    """

    # The distortion type, catalog depth and number of workers not given are those of the config and the machine
    if distortion_type is None:
        distortion_type = config.platepar_distortion_type

    if lim_mag is None:
        lim_mag = config.catalog_mag_limit

    if nproc is None:
        nproc = multiprocessing.cpu_count()

    # The match radius scaled to the image size as in the recalibration
    if match_radius is None:
        match_radius = MATCH_RADIUS*max(1.0, np.hypot(platepar.X_res, platepar.Y_res)/np.hypot(1280, 720))

    # The nights in time order, with the chunks with enough stars to be recalibrated, and the held-out nights
    min_stars = max(config.ff_min_stars, config.min_matched_stars)
    nights = readNights(dir_paths, config, min_stars=min_stars)
    if not nights:
        raise ValueError("No chunks with at least {:d} stars in the given directories".format(min_stars))
    held = heldOutNights(nights, holdout)

    # Every chunk_step-th chunk of the nights of the fit, all chunks of the held-out nights. The key of a chunk
    #   is (night index, chunk index)
    usable = [(n, c) for n, night in enumerate(nights) for c in range(len(night['chunks']))]
    train_keys = [key for key in usable if key[0] not in held][::max(1, chunk_step)]
    held_keys = [key for key in usable if key[0] in held]
    if not train_keys:
        raise ValueError("No chunks in the nights of the fit")

    log.info("Joint distortion fit: {:d} nights ({:d} held out), {:d} chunks in the fit, {:d} held out".format(
        len(nights), len(held), len(train_keys), len(held_keys)))

    def _chunk(key):
        """ The chunk of a key. """

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
            platepar of the night, or of the starting platepar. """

        return transferPointing(distortion, night['platepar'] if night['platepar'] is not None else platepar)[0]

    def _tasks(keys, starts):
        """ The recalibration tasks of chunks, each starting from the platepar of its night. """

        return [(key, starts[key[0]], _chunk(key)[0], _chunk(key)[1], _chunk(key)[2], _chunk(key)[3])
                for key in keys]

    # The workers of the recalibration, which share the catalogs and the config
    init_args = (catalog_stars, match_catalog_stars, config, match_radius, lim_mag)
    pool = None
    if nproc > 1:
        pool = multiprocessing.Pool(nproc, initializer=_initWorker, initargs=init_args)
    else:
        _initWorker(*init_args)

    report = {'nights': [night['dir'] for night in nights], 'held_out': sorted(nights[n]['dir'] for n in held),
              'distortion_type': distortion_type, 'match_radius': match_radius, 'lim_mag': lim_mag,
              'snr_min': snr_min, 'chunks': len(train_keys), 'iterations': []}
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
            if iteration == n_iter:
                break

            # The chunks with enough matched stars for the fit of the distortion
            fit_platepars, fit_stars = {}, {}
            for key in train_keys:
                pp_chunk, stars = results[key]
                if pp_chunk is None:
                    continue
                stars = _goodStars(stars, snr_min)
                if len(stars) >= MIN_CHUNK_STARS:
                    fit_platepars[key], fit_stars[key] = pp_chunk, stars
            if not fit_platepars:
                raise ValueError("No chunk has {:d} matched stars for the fit".format(MIN_CHUNK_STARS))

            distortion, n_fit, reference_key = fitDistortion(fit_platepars, fit_stars, jds, distortion_type,
                                                             max_stars)
            entry['fit_chunks'], entry['fit_stars'] = len(fit_platepars), n_fit
            report['reference'] = (nights[reference_key[0]]['dir'], _chunk(reference_key)[0])

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

            held_nights = sorted(held)
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

    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # The photometry of the starting platepar is kept, it isn't fitted here
    for name in PHOTOMETRY_PARAMETERS:
        if hasattr(platepar, name):
            setattr(distortion, name, copy.deepcopy(getattr(platepar, name)))

    return distortion, report


def _statsLine(stats, n_recalibrated, n_chunks):
    """ One line of the residual statistics, for the log and the report. """

    if stats is None:
        return "recalibrated {:d}/{:d}, no matched stars".format(n_recalibrated, n_chunks)

    return ("recalibrated {:d}/{:d}, {:d} stars, median {:.2f}\", rms {:.2f}\" (x {:.2f}\", y {:.2f}\"), corners "
            "{:s}".format(n_recalibrated, n_chunks, stats['n'], stats['median'], stats['rms'], stats['std_x'],
                          stats['std_y'], _cornersLine(stats)))


def _cornersLine(stats):
    """ The size of the mean residual in every corner (arcsec). """

    return ' '.join('{:s} {:.2f}"'.format(name, np.hypot(*stats['corners'][name][:2])) for name in CORNER_NAMES)


def _statsBlock(stats, indent='    '):
    """ The radial profile, zones and corners of the residual statistics, as lines of text. """

    if stats is None:
        return [indent + 'no matched stars']

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

    lines = ['Joint distortion fit, distortion type {:s}'.format(report['distortion_type'])]
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

    return '\n'.join(lines) + '\n'


def _parseHoldout(value):
    """ The held-out nights of the command line: N:K, or none. """

    if value.lower() == 'none':
        return None

    n, k = (int(v) for v in value.split(':'))
    if (n < 2) or (k < 0) or (k >= n):
        raise argparse.ArgumentTypeError("The held-out nights must be N:K with N >= 2 and 0 <= K < N")

    return (n, k)


if __name__ == "__main__":

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
                            help="Path of the fitted platepar. The report is written next to it (.txt).")

    arg_parser.add_argument('-t', '--type', metavar='DISTORTION_TYPE', type=str,
                            help="Distortion type of the fit, e.g. poly3+radial5. The platepar_distortion_type of "
                            "the config by default.")

    holdout_group = arg_parser.add_mutually_exclusive_group()

    holdout_group.add_argument('--holdout', metavar='N:K', type=_parseHoldout, default=None,
                               help="Hold out every N-th night (in time order) from the K-th (from 0) to check the "
                               "fit, e.g. 4:1. None by default.")

    holdout_group.add_argument('--holdoutdirs', nargs='+', metavar='DIR_PATH', type=str,
                               help="Hold out the nights of these directories (some of the given ones) instead.")

    arg_parser.add_argument('--iter', metavar='N_ITER', type=int, default=4,
                            help="Number of fits of the distortion. 4 by default.")

    arg_parser.add_argument('--snrmin', metavar='SNR', type=float, default=10.0,
                            help="Minimum SNR of the stars. 10 by default.")

    arg_parser.add_argument('--maxstars', metavar='N_STARS', type=int, default=20000,
                            help="Maximum number of stars in a fit of the distortion. 20000 by default.")

    arg_parser.add_argument('--step', metavar='STEP', type=int, default=2,
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

    arg_parser.add_argument('--nproc', metavar='N_PROC', type=int,
                            help="Number of parallel workers. All cores by default.")

    cml_args = arg_parser.parse_args()

    #########################

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, cml_args.dir_paths[0])

    # Print the progress
    log = getLogger("rmslogger", level="INFO")

    # The starting platepar, read as the processing reads it
    platepar = Platepar()
    platepar.read(cml_args.platepar, use_flat=config.use_flat)

    fitted, fit_report = jointDistortionFit(cml_args.dir_paths, platepar, config, distortion_type=cml_args.type,
                                            holdout=(cml_args.holdoutdirs or cml_args.holdout),
                                            n_iter=cml_args.iter,
                                            snr_min=cml_args.snrmin, max_stars=cml_args.maxstars,
                                            chunk_step=cml_args.step, match_radius=cml_args.radius,
                                            lim_mag=cml_args.limmag, nproc=cml_args.nproc)

    # The fitted distortion with the pointing of the given platepar
    if cml_args.pointing:
        pointing = Platepar()
        pointing.read(cml_args.pointing, use_flat=config.use_flat)
        fitted, rms = transferPointing(fitted, pointing)
        print("Pointing of {:s} carried over, the two differ by {:.3f} px rms".format(cml_args.pointing, rms))

    # Save the platepar and the report
    fitted.write(cml_args.out)
    report_path = os.path.splitext(cml_args.out)[0] + '_report.txt'
    with open(report_path, 'w') as f:
        f.write(formatReport(fit_report))

    print(formatReport(fit_report))
    print("Platepar written to {:s}, report to {:s}".format(cml_args.out, report_path))
