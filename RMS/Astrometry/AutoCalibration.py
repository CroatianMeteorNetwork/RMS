""" Automatic calibration of a camera: fitting a platepar from the stars and judging how well a platepar fits the
stars across the whole image.

A platepar is judged on chunks of frames recorded under a clear sky. Each chunk is recalibrated with the platepar
as in the processing (only the pointing is refitted), then the detected stars are matched to the catalog. The
share of the detected stars which are matched and their distance from the catalog positions are measured in zones
of the image: rings of the distance from the centre (as a share of the distance to the corners), and the four sides
of the outer frame of the image. A platepar which only fits a part of the image (e.g. the distortion is wrong near
the edges) matches the stars there poorly, even if the matched stars elsewhere fit well, which the overall match
statistics of the recalibration don't show. The photometric offset is fitted on the matched stars of every chunk,
and its residuals are measured in the same zones (e.g. a wrong vignetting coefficient makes the edges too bright).

A platepar is fitted from scratch (see AutoPlatepar) independently on several sections (chunks spread in time),
which have to agree with each other apart from the pointing, and the photometry (with the vignetting coefficient
for a new camera) is fitted on the chunks of the sections. The fitted platepar is judged on held-out chunks.

The controller (autoCalibrateFile) runs after every processed file of a camera. It judges the current platepar on
the clear chunks of the file and of the files shortly before it, fits a platepar when there is none, when the
current one fails, and periodically to check whether a clearly better one can be fitted. A new camera gets the
fitted platepar right away, a platepar is replaced only after the candidate is confirmed as clearly better on later
chunks.

The limits are in units of the measured width of the stars (FWHM) and in shares of the stars, so they work for
cameras with different resolutions and fields of view.
"""

from __future__ import print_function, division, absolute_import

import copy
import json
import logging
import os

import numpy as np

from scipy.spatial import cKDTree

from RMS.Astrometry.ApplyAstrometry import (extinctionCorrectionTrueToApparent, getFOVSelectionRadius, photomLine,
                                            photometryFitRobust, xyToRaDecPP)
from RMS.Astrometry.Conversions import jd2Date
from RMS.Astrometry.CyFunctions import subsetCatalog
from RMS.Formats.Platepar import getCatalogStarsImagePositions
from RMS.Logger import getLogger


log = getLogger("logger")


# Edges of the rings of the image, as a share of the distance from the centre to the corners
ZONE_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0)

# The outer frame of the image (beyond this share of the distance from the centre to the edge, along either axis)
#   is also split into its four sides, so a platepar which is wrong only on one side is seen. The sides are
#   measured along the axes, so they cover a similar area for any aspect ratio of the image
SIDE_MIN_DISTANCE = 0.5

# Minimum number of detected stars in a zone (over all chunks) for the zone to be judged
MIN_ZONE_STARS = 10

# With a mask, a zone is judged only if at least this share of its area is usable (not masked, nor next to the
#   mask), and the detections within this many FWHMs (at least MASK_EDGE_MIN_PX) of the mask are not counted: next
#   to a tree or a building there are half-hidden stars and lit branches which match no catalog star
MIN_ZONE_USABLE = 0.3
MASK_EDGE_MARGIN_FWHM = 2.0
MASK_EDGE_MIN_PX = 3.0

# A detected star is matched if a catalog star is within this many FWHMs of it (at least 1 px)
MATCH_RADIUS_FWHM = 1.0

# Columns of the stars of a chunk (CALSTARS)
STAR_Y, STAR_X, STAR_INTENSITY, STAR_AMPLITUDE, STAR_FWHM, STAR_BACKGROUND, STAR_SNR, STAR_SATURATED = range(8)


def zoneNames():
    """ Names of the zones of the image, in the order of the score: the rings, then the sides of the outer frame
        (right, bottom, left, top, as seen in the image with y down).

    Return:
        [list] Names of the zones.
    """

    rings = ['r{:.0f}-{:.0f}'.format(100*a, 100*b) for a, b in zip(ZONE_EDGES[:-1], ZONE_EDGES[1:])]
    sides = ['side_right', 'side_bottom', 'side_left', 'side_top']

    return rings + sides


def zoneMembership(x, y, width, height):
    """ The zones of the image which the positions are in. A position is in one ring, and in one side if it is in
        the outer frame.

    Arguments:
        x: [ndarray] X coordinates (px).
        y: [ndarray] Y coordinates (px).
        width: [int] Image width (px).
        height: [int] Image height (px).

    Return:
        [ndarray] Boolean array (n zones x n positions), see zoneNames for the order of the zones.
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Distance from the centre as a share of the distance to the corners
    dx = x - width/2.0
    dy = y - height/2.0
    r = np.hypot(dx, dy)/np.hypot(width/2.0, height/2.0)

    membership = []

    # Rings (the last one includes the corners exactly)
    for i, (a, b) in enumerate(zip(ZONE_EDGES[:-1], ZONE_EDGES[1:])):
        upper = (r <= b) if (i == len(ZONE_EDGES) - 2) else (r < b)
        membership.append((r >= a) & upper)

    # Sides of the outer frame: the distance from the centre along each axis as a share of the distance to that
    #   edge, the side is the one the position is closest to
    nx = dx/(width/2.0)
    ny = dy/(height/2.0)
    outer = np.maximum(np.abs(nx), np.abs(ny)) >= SIDE_MIN_DISTANCE
    horizontal = np.abs(nx) >= np.abs(ny)
    membership.append(outer & horizontal & (nx >= 0))
    membership.append(outer & (~horizontal) & (ny >= 0))
    membership.append(outer & horizontal & (nx < 0))
    membership.append(outer & (~horizontal) & (ny < 0))

    return np.array(membership)


def chunkCoverage(stars, width, height, mask=None, n_grid=4):
    """ How the stars of a chunk cover the image: the share of the cells of a grid (only the cells which are
        mostly not masked) which have a star. This needs no platepar, so it measures a clear sky also before
        the camera is calibrated.

    Arguments:
        stars: [ndarray] Stars of the chunk (CALSTARS rows: y, x, ...).
        width: [int] Image width (px).
        height: [int] Image height (px).

    Keyword arguments:
        mask: [ndarray] Mask image (0 is masked), None if there is none. None by default.
        n_grid: [int] Number of cells of the grid along each axis. 4 by default.

    Return:
        [float] Share of the usable cells which have a star.
    """

    stars = np.asarray(stars, dtype=np.float64)

    # The cells which are mostly not masked are usable
    usable = np.ones((n_grid, n_grid), dtype=bool)
    if mask is not None:
        mask = np.asarray(mask)
        for i in range(n_grid):
            for j in range(n_grid):
                cell = mask[i*mask.shape[0]//n_grid:(i + 1)*mask.shape[0]//n_grid,
                            j*mask.shape[1]//n_grid:(j + 1)*mask.shape[1]//n_grid]
                usable[i, j] = np.mean(cell > 0) >= 0.5

    if (len(stars) == 0) or (not np.any(usable)):
        return 0.0

    # Cells with a star
    occupied = np.zeros((n_grid, n_grid), dtype=bool)
    rows = np.clip((stars[:, STAR_Y]/height*n_grid).astype(int), 0, n_grid - 1)
    cols = np.clip((stars[:, STAR_X]/width*n_grid).astype(int), 0, n_grid - 1)
    occupied[rows, cols] = True

    return float(np.sum(occupied & usable))/np.sum(usable)


def _catalogInFOV(platepar, jd, catalog_stars, lim_mag):
    """ The catalog stars brighter than the limiting magnitude which are in the field of view.

    Arguments:
        platepar: [Platepar] Platepar.
        jd: [float] Julian date.
        catalog_stars: [ndarray] Catalog stars (ra, dec, mag).
        lim_mag: [float] Limiting magnitude.

    Return:
        [ndarray] Catalog stars (ra, dec, mag) in the field of view.
    """

    # The catalog stars within the radius of the field of view around the centre of the image
    fov_radius = getFOVSelectionRadius(platepar)
    _, ra_c, dec_c, _ = xyToRaDecPP([jd2Date(jd)], [platepar.X_res/2], [platepar.Y_res/2], [1], platepar,
                                    extinction_correction=False)
    _, catalog_fov = subsetCatalog(catalog_stars, ra_c[0], dec_c[0], jd, platepar.lat, platepar.lon, fov_radius,
                                   lim_mag)

    return catalog_fov


def matchChunk(platepar, jd, stars, catalog_stars, lim_mag, match_radius):
    """ Match the detected stars of a chunk to the catalog with the platepar.

    Arguments:
        platepar: [Platepar] Platepar, already recalibrated on the chunk if that is wanted.
        jd: [float] Julian date of the chunk.
        stars: [ndarray] Stars of the chunk (CALSTARS rows).
        catalog_stars: [ndarray] Catalog stars (ra, dec, mag).
        lim_mag: [float] Limiting magnitude of the catalog stars.
        match_radius: [float] A detected star is matched if a catalog star is within this distance (px).

    Return:
        (dist, cat_ra, cat_dec, cat_mag): [tuple of ndarrays] For every detected star the distance to the
            nearest catalog star (px, inf without catalog stars) and that catalog star.
    """

    # Without a catalog star nearby, a detected star is at an infinite distance
    stars = np.asarray(stars, dtype=np.float64)
    n = len(stars)
    dist = np.full(n, np.inf)
    cat_ra, cat_dec, cat_mag = np.zeros(n), np.zeros(n), np.zeros(n)

    # The catalog stars brighter than the limiting magnitude in the field of view
    catalog_fov = _catalogInFOV(platepar, jd, catalog_stars, lim_mag)
    if (n == 0) or (len(catalog_fov) == 0):
        return dist, cat_ra, cat_dec, cat_mag

    x_cat, y_cat, _ = getCatalogStarsImagePositions(catalog_fov, jd, platepar)

    # Only the catalog stars which project into (or near) the image
    inside = (x_cat > -match_radius) & (x_cat < platepar.X_res + match_radius) & (y_cat > -match_radius) \
        & (y_cat < platepar.Y_res + match_radius)
    if not np.any(inside):
        return dist, cat_ra, cat_dec, cat_mag

    x_cat, y_cat, catalog_fov = x_cat[inside], y_cat[inside], catalog_fov[inside]

    # The nearest catalog star of every detected star
    tree = cKDTree(np.column_stack([x_cat, y_cat]))
    dist, idx = tree.query(np.column_stack([stars[:, STAR_X], stars[:, STAR_Y]]))
    cat_ra, cat_dec, cat_mag = catalog_fov[idx, 0], catalog_fov[idx, 1], catalog_fov[idx, 2]

    return dist, cat_ra, cat_dec, cat_mag


def _usableArea(mask, platepar, fwhm):
    """ The pixels whose detections are counted in the score: not masked, and not next to the mask.

    Arguments:
        mask: [ndarray] Mask image (0 is masked), None if there is none.
        platepar: [Platepar] Platepar (its image size).
        fwhm: [float] Width of the stars (px).

    Return:
        [ndarray] Boolean image, None without a mask (or with a mask of another size, which is not used).
    """

    import cv2

    if mask is None:
        return None

    mask = np.asarray(mask)
    if mask.shape[:2] != (int(platepar.Y_res), int(platepar.X_res)):
        log.warning("Automatic calibration: the mask ({:d}x{:d}) doesn't match the image, it is not used".format(
            mask.shape[1], mask.shape[0]))
        return None

    # Distance of every pixel from the nearest masked pixel
    unmasked = (mask[:, :, 0] if mask.ndim == 3 else mask) > 0
    if np.all(unmasked):
        return unmasked
    distance = cv2.distanceTransform(unmasked.astype(np.uint8), cv2.DIST_L2, 3)

    return distance >= max(MASK_EDGE_MIN_PX, MASK_EDGE_MARGIN_FWHM*fwhm)


def _zoneUsable(usable_px, width, height, step=4):
    """ Whether enough of the area of every zone is usable to judge it (see MIN_ZONE_USABLE).

    Arguments:
        usable_px: [ndarray] Boolean image of the usable pixels.
        width: [int] Image width (px).
        height: [int] Image height (px).

    Keyword arguments:
        step: [int] Spacing of the sampled pixels. 4 by default.

    Return:
        [ndarray] Boolean array, one value per zone.
    """

    yy, xx = np.mgrid[0:height:step, 0:width:step]
    membership = zoneMembership(xx.ravel() + 0.5, yy.ravel() + 0.5, width, height)
    usable = usable_px[yy.ravel(), xx.ravel()]

    return np.array([(np.mean(usable[in_zone]) >= MIN_ZONE_USABLE) if np.any(in_zone) else False
                     for in_zone in membership])


def recalibrateChunk(platepar, chunk_name, jd, stars, catalog_stars, config, ff_frames):
    """ Recalibrate the platepar on a chunk as in the processing (the pointing is refitted).

    Arguments:
        platepar: [Platepar] Platepar.
        chunk_name: [str] Name of the chunk (FF name).
        jd: [float] Julian date of the chunk.
        stars: [ndarray] Stars of the chunk (CALSTARS rows).
        catalog_stars: [ndarray] Catalog stars (ra, dec, mag).
        config: [Config] Configuration.
        ff_frames: [int] Number of frames of the chunk.

    Return:
        (platepar, ok): [tuple] The recalibrated platepar (the given one if the recalibration failed) and
            whether the recalibration succeeded.
    """

    from RMS.Astrometry.ApplyRecalibrate import recalibratePlateparsForFF

    # The recalibration logs every step, which is too much for the many chunks which are judged
    previous_level = log.level
    rms_log = logging.getLogger("rmslogger")
    previous_rms_level = rms_log.level
    log.setLevel(logging.WARNING)
    rms_log.setLevel(logging.WARNING)
    try:
        result = recalibratePlateparsForFF(copy.deepcopy(platepar), [chunk_name],
                                           {chunk_name: [tuple(row) for row in stars]}, catalog_stars, config,
                                           ff_frames=ff_frames)

    finally:

        # The logging is restored also if the recalibration fails
        log.setLevel(previous_level)
        rms_log.setLevel(previous_rms_level)

    # If the recalibration failed, the given platepar is used as it is
    recalibrated = result.get(chunk_name)
    if (recalibrated is not None) and getattr(recalibrated, 'auto_recalibrated', False):
        return recalibrated, True

    return copy.deepcopy(platepar), False


class CalibrationScore(object):
    """ How well a platepar fits the stars of a set of chunks, over the whole image.

    Attributes:
        zones: [list] Names of the zones (see zoneNames).
        zone_stars: [ndarray] Number of detected stars in every zone.
        zone_share: [ndarray] Share of the detected stars of every zone which are matched (nan without stars).
        zone_dist: [ndarray] Median distance of the matched stars of every zone from the catalog (FWHM).
        zone_mag_residual: [ndarray] Median photometric residual of the matched stars of every zone (mag).
        n_stars: [int] Number of detected stars.
        n_matched: [int] Number of matched stars.
        share: [float] Share of the detected stars which are matched.
        dist: [float] Median distance of the matched stars from the catalog (FWHM).
        fwhm: [float] Median width of the stars (px).
        chunks: [int] Number of chunks.
        recalibrated: [int] Number of chunks which were recalibrated successfully.
        mag_lev: [ndarray] Photometric offset of every chunk with enough matched stars.
        mag_lev_stddev: [ndarray] Scatter of the photometric fit of those chunks (mag).
        vignetting_coeff: [float] Vignetting coefficient used for the photometry.
        zone_usable: [ndarray] Whether enough of the area of every zone is not masked to judge it.
    """

    def __init__(self):
        """ An empty score, see the class docstring for the attributes. """

        # The score of every zone, and over all zones and chunks (nan until it is measured)
        self.zones = zoneNames()
        n_zones = len(self.zones)
        self.zone_stars = np.zeros(n_zones, dtype=int)
        self.zone_share = np.full(n_zones, np.nan)
        self.zone_dist = np.full(n_zones, np.nan)
        self.zone_mag_residual = np.full(n_zones, np.nan)
        self.n_stars = 0
        self.n_matched = 0
        self.share = np.nan
        self.dist = np.nan
        self.fwhm = np.nan
        self.chunks = 0
        self.recalibrated = 0
        self.mag_lev = np.array([])
        self.mag_lev_stddev = np.array([])
        self.vignetting_coeff = None
        self.zone_usable = np.ones(n_zones, dtype=bool)


    def judged(self):
        """ Zones with enough stars, and enough area which is not masked, to be judged.

        Return:
            [ndarray] Boolean array, one value per zone.
        """

        return (self.zone_stars >= MIN_ZONE_STARS) & self.zone_usable


    def worstZoneShare(self):
        """ The lowest match share of the zones which can be judged.

        Return:
            [float] The share, nan if no zone can be judged.
        """

        # Only the zones with enough stars are compared
        judged = self.judged()
        if not np.any(judged):
            return np.nan

        return float(np.nanmin(self.zone_share[judged]))


    def magLevScatter(self):
        """ Scatter of the photometric offset between the chunks.

        Return:
            [float] Standard deviation of the offsets (mag), nan with fewer than 2 chunks.
        """

        if len(self.mag_lev) < 2:
            return np.nan

        return float(np.std(self.mag_lev))


    def toDict(self):
        """ The score as a dictionary which can be saved as JSON.

        Return:
            [dict] The score, with nan as None.
        """

        def _value(v):
            """ A value which can be saved as JSON. """
            # Non-finite numbers are saved as None, numpy numbers as Python numbers
            if isinstance(v, (float, np.floating)):
                return None if not np.isfinite(v) else round(float(v), 4)
            if isinstance(v, (int, np.integer)):
                return int(v)
            return v

        # The score of every zone, the totals and the photometry of the chunks
        return {
            'zones': {name: {'stars': _value(n), 'share': _value(s), 'dist_fwhm': _value(d),
                             'mag_residual': _value(m)}
                      for name, n, s, d, m in zip(self.zones, self.zone_stars, self.zone_share, self.zone_dist,
                                                  self.zone_mag_residual)},
            'n_stars': _value(self.n_stars), 'n_matched': _value(self.n_matched), 'share': _value(self.share),
            'dist_fwhm': _value(self.dist), 'fwhm_px': _value(self.fwhm), 'chunks': _value(self.chunks),
            'recalibrated': _value(self.recalibrated),
            'mag_lev': [_value(v) for v in self.mag_lev],
            'mag_lev_stddev': [_value(v) for v in self.mag_lev_stddev],
            'mag_lev_scatter': _value(self.magLevScatter()),
            'worst_zone_share': _value(self.worstZoneShare()),
            'vignetting_coeff': _value(self.vignetting_coeff),
        }


def scorePlatepar(platepar, chunks, catalog_stars, config, ff_frames=256, recalibrate=True, lim_mag=None,
                  fit_vignetting=False, mask=None):
    """ Score how well a platepar fits the stars of the chunks over the whole image, see the module docstring.

    Arguments:
        platepar: [Platepar] Platepar.
        chunks: [list] Chunks as (name, jd, stars), the stars as CALSTARS rows.
        catalog_stars: [ndarray] Catalog stars (ra, dec, mag).
        config: [Config] Configuration.

    Keyword arguments:
        ff_frames: [int] Number of frames of a chunk. 256 by default.
        recalibrate: [bool] Recalibrate the platepar on every chunk first, as in the processing. True by
            default.
        lim_mag: [float] Limiting magnitude of the catalog stars, config.catalog_mag_limit if None.
        fit_vignetting: [bool] Fit the vignetting coefficient in the photometry of every chunk, otherwise the
            coefficient of the platepar is used. False by default.
        mask: [ndarray] Mask image (0 is masked), None if there is none. The detections next to the mask are not
            counted, and the zones which are mostly masked are not judged. None by default.

    Return:
        [CalibrationScore] The score.
    """

    if lim_mag is None:
        lim_mag = config.catalog_mag_limit

    score = CalibrationScore()
    score.chunks = len(chunks)

    # The width of the stars sets the match radius and the unit of the distances
    all_fwhm = [row[STAR_FWHM] for _, _, stars in chunks for row in stars if len(row) > STAR_FWHM]
    fwhm = float(np.median(all_fwhm)) if all_fwhm else 2.0
    fwhm = fwhm if (np.isfinite(fwhm) and (fwhm > 0)) else 2.0
    score.fwhm = fwhm
    match_radius = max(1.0, MATCH_RADIUS_FWHM*fwhm)

    vignetting = platepar.vignetting_coeff if platepar.vignetting_coeff is not None else 0.0
    score.vignetting_coeff = vignetting

    # Positions, matches, distances (FWHM) and photometric residuals of the stars of all chunks, and the
    #   photometry of every chunk
    xs, ys, matched, dists, mag_residuals = [], [], [], [], []
    mag_levs, mag_lev_stddevs = [], []
    for name, jd, stars in chunks:

        # Chunks without stars are skipped
        stars = np.asarray(stars, dtype=np.float64)
        if len(stars) == 0:
            continue

        # The pointing is refitted on the chunk as in the processing
        pp = platepar
        if recalibrate:
            pp, ok = recalibrateChunk(platepar, name, jd, stars, catalog_stars, config, ff_frames)
            score.recalibrated += int(ok)

        dist, cat_ra, cat_dec, cat_mag = matchChunk(pp, jd, stars, catalog_stars, lim_mag, match_radius)
        is_matched = dist <= match_radius

        # The stars of the chunk, whether they are matched, and their distance from the catalog (FWHM)
        xs.append(stars[:, STAR_X])
        ys.append(stars[:, STAR_Y])
        matched.append(is_matched)
        dists.append(dist/fwhm)

        # Photometry on the matched, not saturated stars
        residual = np.full(len(stars), np.nan)
        saturated = stars[:, STAR_SATURATED] > 0 if stars.shape[1] > STAR_SATURATED else np.zeros(len(stars), bool)
        good = is_matched & (~saturated) & (stars[:, STAR_INTENSITY] > 0)
        if np.sum(good) >= config.min_matched_stars:

            apparent_mags = extinctionCorrectionTrueToApparent(cat_mag[good], cat_ra[good], cat_dec[good], jd, pp)
            radius = np.hypot(stars[good, STAR_X] - pp.X_res/2, stars[good, STAR_Y] - pp.Y_res/2)

            # Fit the photometric offset robustly (with the vignetting coefficient if asked)
            try:
                params, fit_stddev, _, _, _, _ = photometryFitRobust(
                    stars[good, STAR_INTENSITY], radius, apparent_mags,
                    fixed_vignetting=(None if fit_vignetting else vignetting))

                mag_lev, vig = params[0], (params[1] if fit_vignetting else vignetting)

                # Residual of every star: the magnitude from its intensity minus the catalog magnitude
                residual[good] = photomLine((stars[good, STAR_INTENSITY], radius), mag_lev, vig) - apparent_mags

                mag_levs.append(mag_lev)
                mag_lev_stddevs.append(fit_stddev)

            except Exception as e:
                log.debug("Photometry of chunk {:s} failed: {:s}".format(name, repr(e)))

        mag_residuals.append(residual)

    if not xs:
        return score

    # The stars of all chunks together
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    matched, dists = np.concatenate(matched), np.concatenate(dists)
    mag_residuals = np.concatenate(mag_residuals)

    # With a mask, the detections next to it are not counted, and the zones which are mostly masked not judged
    usable_px = _usableArea(mask, platepar, fwhm)
    if usable_px is not None:
        h, w = usable_px.shape
        keep = usable_px[np.clip(ys.astype(int), 0, h - 1), np.clip(xs.astype(int), 0, w - 1)]
        xs, ys, matched, dists, mag_residuals = xs[keep], ys[keep], matched[keep], dists[keep], mag_residuals[keep]
        score.zone_usable = _zoneUsable(usable_px, platepar.X_res, platepar.Y_res)
        if not len(xs):
            return score

    # Overall
    score.n_stars = len(xs)
    score.n_matched = int(np.sum(matched))
    score.share = float(np.mean(matched))
    if np.any(matched):
        score.dist = float(np.median(dists[matched]))

    # Per zone
    membership = zoneMembership(xs, ys, platepar.X_res, platepar.Y_res)
    for i, in_zone in enumerate(membership):
        score.zone_stars[i] = int(np.sum(in_zone))
        if score.zone_stars[i] == 0:
            continue

        # Share of the stars of the zone which are matched, and the distance of the matched ones
        score.zone_share[i] = float(np.mean(matched[in_zone]))
        if np.any(matched[in_zone]):
            score.zone_dist[i] = float(np.median(dists[in_zone & matched]))

        # Median photometric residual of the stars of the zone
        zone_residuals = mag_residuals[in_zone & np.isfinite(mag_residuals)]
        if len(zone_residuals):
            score.zone_mag_residual[i] = float(np.median(zone_residuals))

    score.mag_lev = np.array(mag_levs)
    score.mag_lev_stddev = np.array(mag_lev_stddevs)

    return score


### Judging a platepar ###

# A platepar fits the stars only if most chunks on a clear sky can be recalibrated with it
MIN_RECALIBRATED_SHARE = 0.9

# The match share of the worst zone relative to the median zone (a platepar which is wrong in a part of the image
#   matches the stars there poorly, while stars which are not in the catalog lower all zones alike)
MIN_WORST_ZONE_RATIO = 0.8

# Maximum median distance of the matched stars from the catalog (FWHM)
MAX_DIST_FWHM = 0.35

# Maximum scatter of the photometric fit of a chunk (median of the chunks, mag)
MAX_MAG_STDDEV = 0.15

# Maximum median photometric residual of a zone (mag), e.g. a wrong vignetting makes the edges too bright or
#   too faint. The vignetting is modelled radially about the centre of the image, so a camera without a flat whose
#   brightness falls off off-centre keeps a residual gradient across the image (about 0.1 mag was seen)
MAX_ZONE_MAG_RESIDUAL = 0.2

# A candidate which replaces a platepar may have a worst photometric residual of a zone larger than the current
#   platepar by at most this much (mag)
ZONE_MAG_TOLERANCE = 0.05

# A candidate replaces the current platepar only if its worst zone matches this share of the stars more (or the
#   current platepar fails the judgement), and no zone is worse by more than the tolerance
MIN_WORST_ZONE_GAIN = 0.1
ZONE_SHARE_TOLERANCE = 0.03

# A failing platepar is replaced right away, on trial, by a candidate which matches this many times the share of
#   the stars it matches, over the whole image or in the worst zone (a platepar which is far off would otherwise stay
#   in use until the candidate is confirmed)
TRIAL_MATCH_RATIO = 1.5


def judgeScore(score):
    """ Judge whether a platepar fits the stars well over the whole image, from its score on clear chunks.

    Arguments:
        score: [CalibrationScore] Score of the platepar.

    Return:
        (ok, reasons): [tuple] True if the platepar passes, and the reasons why it doesn't (list of str).
    """

    reasons = []

    if score.chunks == 0:
        return False, ['no chunks to judge on']

    # Most chunks have to be recalibrated (the recalibration fits the pointing, as in the processing)
    recalibrated_share = score.recalibrated/score.chunks
    if recalibrated_share < MIN_RECALIBRATED_SHARE:
        reasons.append('only {:d}/{:d} chunks recalibrated'.format(score.recalibrated, score.chunks))

    # Every zone has to match the stars about as well as the others
    judged = score.judged()
    if not np.any(judged):
        reasons.append('no zone has enough stars')

    else:
        # The share of the worst zone compared to the median zone
        median_share = float(np.nanmedian(score.zone_share[judged]))
        worst = score.worstZoneShare()
        if (median_share <= 0) or (worst < MIN_WORST_ZONE_RATIO*median_share):
            worst_zone = score.zones[int(np.nanargmin(np.where(judged, score.zone_share, np.nan)))]
            reasons.append('zone {:s} matches {:.2f} of its stars, the median zone {:.2f}'.format(worst_zone, worst,
                                                                                                median_share))

    if not (np.isfinite(score.dist) and (score.dist <= MAX_DIST_FWHM)):
        reasons.append('median distance from the catalog {:.2f} FWHM'.format(score.dist))

    # The photometry has to fit everywhere
    if len(score.mag_lev_stddev) == 0:
        reasons.append('no photometric fit')

    else:
        # The scatter of the photometric fits of the chunks
        mag_stddev = float(np.median(score.mag_lev_stddev))
        if mag_stddev > MAX_MAG_STDDEV:
            reasons.append('photometric scatter {:.2f} mag'.format(mag_stddev))

        # The photometric residual of every zone
        residuals = np.abs(score.zone_mag_residual[judged])
        if np.any(residuals > MAX_ZONE_MAG_RESIDUAL):
            worst_zone = score.zones[int(np.nanargmax(np.where(judged, np.abs(score.zone_mag_residual), np.nan)))]
            reasons.append('photometric residual {:+.2f} mag in zone {:s}'.format(
                score.zone_mag_residual[score.zones.index(worst_zone)], worst_zone))

    return (len(reasons) == 0), reasons


def candidateIsBetter(candidate, current):
    """ Whether a candidate platepar is clearly better than the current one, both scored on the same chunks. The
        candidate has to pass the judgement itself.

    Arguments:
        candidate: [CalibrationScore] Score of the candidate.
        current: [CalibrationScore] Score of the current platepar, None if there is none.

    Return:
        (better, reason): [tuple] True if the candidate is better, and why (str).
    """

    # The candidate has to pass the judgement itself
    candidate_ok, candidate_reasons = judgeScore(candidate)
    if not candidate_ok:
        return False, 'the candidate fails: ' + '; '.join(candidate_reasons)

    if current is None:
        return True, 'there is no platepar'

    # A failing current platepar is replaced by any candidate which passes
    current_ok, current_reasons = judgeScore(current)
    if not current_ok:
        return True, 'the current platepar fails: ' + '; '.join(current_reasons)

    # No zone which both can judge may be clearly worse, in the astrometry and in the photometry
    judged = candidate.judged() & current.judged()
    worse = (candidate.zone_share < current.zone_share - ZONE_SHARE_TOLERANCE) & judged
    if np.any(worse):
        return False, 'zone {:s} is worse'.format(candidate.zones[int(np.argmax(worse))])

    # The photometry of the worst zone may not be worse than now
    if np.any(judged):
        candidate_mag = np.nanmax(np.abs(candidate.zone_mag_residual[judged]))
        current_mag = np.nanmax(np.abs(current.zone_mag_residual[judged]))
        if candidate_mag > current_mag + ZONE_MAG_TOLERANCE:
            return False, 'the photometry is worse ({:.2f} mag in a zone, now {:.2f})'.format(candidate_mag,
                                                                                         current_mag)

    # The worst zone has to be clearly better
    gain = candidate.worstZoneShare() - current.worstZoneShare()
    if gain >= MIN_WORST_ZONE_GAIN:
        return True, 'the worst zone matches {:+.2f} of its stars more'.format(gain)

    return False, 'not clearly better (worst zone {:+.2f})'.format(gain)


def candidateIsFarBetter(candidate, current):
    """ Whether a candidate which is better than a failing platepar (see candidateIsBetter) is so much better that
        it is installed right away, on trial, instead of waiting for the confirmation on later chunks.

    Arguments:
        candidate: [CalibrationScore] Score of the candidate.
        current: [CalibrationScore] Score of the current platepar, None if there is none.

    Return:
        [bool] True if the current platepar fails and the candidate matches far more stars.
    """

    # Only a failing platepar is replaced on trial
    if (current is None) or judgeScore(current)[0]:
        return False

    def _farMore(candidate_share, current_share):

        # A platepar which matches nothing (or can't be judged) is far worse than one which matches stars
        if not np.isfinite(candidate_share):
            return False
        if (not np.isfinite(current_share)) or (current_share <= 0):
            return candidate_share > 0

        return candidate_share >= TRIAL_MATCH_RATIO*current_share

    # The candidate matches far more stars over the whole image or in the worst zone
    return bool(_farMore(candidate.share, current.share)
                or _farMore(candidate.worstZoneShare(), current.worstZoneShare()))



### Clear chunks ###

# A chunk is clear if it has at least this share of the stars of the clearest chunks of the camera, and its stars
#   cover this share of the image
CLEAR_STARS_SHARE = 0.6
CLEAR_COVERAGE = 0.75

# Minimum number of stars of a clear chunk
MIN_CLEAR_STARS = 30

# Percentile of the star counts of the chunks seen by the camera which is taken as the clearest sky
CLEAR_REFERENCE_PERCENTILE = 90


def clearChunks(chunks, width, height, mask=None, reference_stars=None, min_stars=MIN_CLEAR_STARS):
    """ The chunks recorded under a clear sky, judged without a platepar: enough stars compared to the clearest
        sky of the camera, which cover the image.

    Arguments:
        chunks: [list] Chunks as (name, jd, stars).
        width: [int] Image width (px).
        height: [int] Image height (px).

    Keyword arguments:
        mask: [ndarray] Mask image (0 is masked), None if there is none. None by default.
        reference_stars: [float] Number of stars of a chunk under the clearest sky of the camera. None to take it
            from these chunks (the CLEAR_REFERENCE_PERCENTILE of their star counts).
        min_stars: [int] Minimum number of stars of a clear chunk. MIN_CLEAR_STARS by default.

    Return:
        [list] The clear chunks, in the given order.
    """

    if not chunks:
        return []

    # Without a reference of the camera, the clearest of these chunks are the reference
    counts = np.array([len(stars) for _, _, stars in chunks])
    if reference_stars is None:
        reference_stars = np.percentile(counts, CLEAR_REFERENCE_PERCENTILE)

    # Enough stars, which cover the image
    clear = []
    for (name, jd, stars), n in zip(chunks, counts):
        if (n >= max(min_stars, CLEAR_STARS_SHARE*reference_stars)) \
                and (chunkCoverage(stars, width, height, mask=mask) >= CLEAR_COVERAGE):
            clear.append((name, jd, stars))

    return clear


def spreadChunks(chunks, n):
    """ Pick n chunks spread evenly over the given ones.

    Arguments:
        chunks: [list] Chunks as (name, jd, stars).
        n: [int] Number of chunks to pick.

    Return:
        [list] The picked chunks, sorted in time (all of them if there are at most n).
    """

    # Sorted in time, an even spread is picked only if there are more chunks than wanted
    chunks = sorted(chunks, key=lambda chunk: chunk[1])
    if len(chunks) <= n:
        return chunks

    return [chunks[i] for i in np.round(np.linspace(0, len(chunks) - 1, n)).astype(int)]



### Fitting a candidate platepar ###

# Number of sections (chunks spread in time) on which a platepar is fitted independently
N_SECTIONS = 4

# Minimum number of sections which have to pass on the held-out chunks and agree. A fit from scratch on one chunk
#   can converge to a wrong solution (a wrong scale and rotation) which still pairs many stars of its own chunk; it
#   fails on the held-out chunks and is dropped, so two independent sections which pass and agree are enough
MIN_SECTIONS = 2

# Minimum number of stars matched in the fit of a section
MIN_SECTION_STARS = 40

# The plate solver stops at the first solution with a log odds ratio above this (its default of 60 makes it search on
#   through all index files after a correct solution of 30-40 on a sparse chunk; a wrong solution is caught by the
#   held-out chunks and the agreement of the sections)
SOLVE_STOP_LOGODDS = 30.0

# Agreement of the sections: the difference of the image positions over the image, after removing the pointing
#   (shift, rotation and scale, which the recalibration refits on every chunk), in FWHM
MAX_SECTION_DIFF_P95 = 0.5
MAX_SECTION_DIFF_MAX = 1.0

# The vignetting coefficient is fitted only if the matched stars reach this share of the distance to the corners
#   (otherwise it is not constrained)
VIGNETTING_MIN_RADIUS = 0.7

# Upper limit of the fitted vignetting coefficient, as a multiple of the default coefficient for the resolution
VIGNETTING_MAX_FACTOR = 3.0


def sectionDifference(platepar_a, platepar_b, jd, fwhm):
    """ How differently two platepars map the image, apart from the pointing: a grid of image positions is mapped
        to the sky with the first platepar, the pointing of the second (RA, Dec, rotation and scale, which the
        recalibration refits on every chunk) is fitted to map those sky positions back onto the grid, and the
        remaining differences are measured.

    Arguments:
        platepar_a: [Platepar] First platepar.
        platepar_b: [Platepar] Second platepar.
        jd: [float] Julian date of the mapping.
        fwhm: [float] Width of the stars (px), the unit of the differences.

    Return:
        (p95, max): [tuple] 95th percentile and maximum of the differences (FWHM).
    """

    import scipy.optimize

    from RMS.Astrometry.ApplyAstrometry import raDecToXYPP

    # A grid of image positions (without the outermost edge) mapped to the sky with the first platepar
    w, h = platepar_a.X_res, platepar_a.Y_res
    gx, gy = np.meshgrid(np.linspace(0.03*w, 0.97*w, 25), np.linspace(0.03*h, 0.97*h, 15))
    gx, gy = gx.ravel(), gy.ravel()

    _, ra, dec, _ = xyToRaDecPP([jd2Date(jd)]*len(gx), gx, gy, [1]*len(gx), platepar_a,
                                extinction_correction=False)

    # Fit the pointing of the second platepar
    platepar = copy.deepcopy(platepar_b)
    p0 = np.array([platepar.RA_d, platepar.dec_d, platepar.pos_angle_ref, platepar.F_scale])

    def _residuals(params):
        """ Differences between the grid and the sky positions mapped back with the given pointing (px). """

        platepar.RA_d, platepar.dec_d, platepar.pos_angle_ref, platepar.F_scale = params
        x2, y2 = raDecToXYPP(ra, dec, jd, platepar)
        return np.concatenate([x2 - gx, y2 - gy])

    # Steps of about a pixel for the angles and the scale
    pixel_deg = 1.0/platepar.F_scale
    fit = scipy.optimize.least_squares(_residuals, p0, x_scale=[pixel_deg, pixel_deg, pixel_deg,
                                                                0.001*platepar.F_scale])
    residuals = _residuals(fit.x)
    diff = np.hypot(residuals[:len(gx)], residuals[len(gx):])/fwhm

    return float(np.percentile(diff, 95)), float(np.max(diff))


def fitPhotometry(platepar, chunks, catalog_stars, config, ff_frames=256, fit_vignetting=False):
    """ Fit the photometric offset of the platepar (and the vignetting coefficient if asked and constrained) on the
        matched stars of the chunks.

    Arguments:
        platepar: [Platepar] Platepar, changed in place.
        chunks: [list] Chunks as (name, jd, stars).
        catalog_stars: [ndarray] Catalog stars (ra, dec, mag).
        config: [Config] Configuration.

    Keyword arguments:
        ff_frames: [int] Number of frames of a chunk. 256 by default.
        fit_vignetting: [bool] Fit the vignetting coefficient. False by default (the coefficient of the platepar
            is kept).

    Return:
        [dict] What was fitted: 'mag_lev', 'vignetting_coeff', 'vignetting_fitted' and the reason if the
            vignetting fit was not used.
    """

    result = {'vignetting_fitted': False}

    # The vignetting is fitted jointly on the stars of all chunks, if they reach far enough from the centre
    if fit_vignetting:
        vignetting, reason = _jointVignetting(platepar, chunks, catalog_stars, config, ff_frames)
        if vignetting is not None:
            platepar.vignetting_coeff = vignetting
            result['vignetting_fitted'] = True
        else:
            result['vignetting_reason'] = reason

    # The photometric offset with the vignetting coefficient of the platepar
    score = scorePlatepar(platepar, chunks, catalog_stars, config, ff_frames=ff_frames)
    if len(score.mag_lev):
        platepar.mag_lev = float(np.median(score.mag_lev))
        platepar.mag_lev_stddev = float(np.median(score.mag_lev_stddev))

    result['mag_lev'] = platepar.mag_lev
    result['vignetting_coeff'] = platepar.vignetting_coeff

    return result


def _jointVignetting(platepar, chunks, catalog_stars, config, ff_frames):
    """ Fit the vignetting coefficient jointly on the matched stars of all chunks: one coefficient for all, and a
        photometric offset for every chunk (the transparency changes), minimizing the robust scatter of the
        magnitudes. The coefficient is constrained only if the stars reach far enough from the centre.

    Arguments:
        platepar: [Platepar] Platepar, recalibrated on every chunk first.
        chunks: [list] Chunks as (name, jd, stars).
        catalog_stars: [ndarray] Catalog stars (ra, dec, mag).
        config: [Config] Configuration.
        ff_frames: [int] Number of frames of a chunk.

    Return:
        (coefficient, reason): [tuple] The coefficient (rad/px), or None and the reason why it wasn't fitted.
    """

    import scipy.optimize

    # The matched, not saturated stars of every chunk
    intensities, radii, mags, chunk_index = [], [], [], []
    for k, (name, jd, stars) in enumerate(chunks):

        # Chunks without stars are skipped
        stars = np.asarray(stars, dtype=np.float64)
        if len(stars) == 0:
            continue

        pp, _ = recalibrateChunk(platepar, name, jd, stars, catalog_stars, config, ff_frames)

        # The matched stars which are not saturated, the chunk is used only with enough of them
        match_radius = max(1.0, MATCH_RADIUS_FWHM*np.median(stars[:, STAR_FWHM]))
        dist, cat_ra, cat_dec, cat_mag = matchChunk(pp, jd, stars, catalog_stars, config.catalog_mag_limit,
                                                    match_radius)
        saturated = stars[:, STAR_SATURATED] > 0 if stars.shape[1] > STAR_SATURATED else np.zeros(len(stars), bool)
        good = (dist <= match_radius) & (~saturated) & (stars[:, STAR_INTENSITY] > 0)
        if np.sum(good) < config.min_matched_stars:
            continue

        # Intensity, distance from the centre and apparent catalog magnitude of every star, and its chunk
        intensities.append(stars[good, STAR_INTENSITY])
        radii.append(np.hypot(stars[good, STAR_X] - pp.X_res/2, stars[good, STAR_Y] - pp.Y_res/2))
        mags.append(extinctionCorrectionTrueToApparent(cat_mag[good], cat_ra[good], cat_dec[good], jd, pp))
        chunk_index.append(np.full(np.sum(good), k))

    if not intensities:
        return None, 'no chunk has enough matched stars'

    intensities, radii = np.concatenate(intensities), np.concatenate(radii)
    mags, chunk_index = np.concatenate(mags), np.concatenate(chunk_index)

    # The stars have to reach far enough from the centre to constrain the vignetting
    corner = np.hypot(platepar.X_res/2, platepar.Y_res/2)
    if np.percentile(radii, 95) < VIGNETTING_MIN_RADIUS*corner:
        return None, "the stars don't reach far enough from the centre"

    chunk_ids = np.unique(chunk_index)

    def _scatter(coefficient):
        """ Robust scatter of the magnitude residuals, with the median offset of every chunk removed.

        Arguments:
            coefficient: [float] Vignetting coefficient (rad/px).

        Return:
            [float] Median absolute residual (mag).
        """

        # The residuals without the photometric offset, then the median of every chunk (its transparency) is
        #   removed
        residuals = photomLine((intensities, radii), 0.0, coefficient) - mags
        for k in chunk_ids:
            residuals[chunk_index == k] -= np.median(residuals[chunk_index == k])

        return np.median(np.abs(residuals))

    # The coefficient between 0 and a multiple of the default for the resolution
    default = 0.001*np.hypot(1280, 720)/corner/2.0
    result = scipy.optimize.minimize_scalar(_scatter, bounds=(0.0, VIGNETTING_MAX_FACTOR*default),
                                            method='bounded')

    return float(result.x), None


def _agreeingSubset(n, agree):
    """ The largest subset of the sections in which every pair agrees (brute force, there are few sections).

    Arguments:
        n: [int] Number of sections.
        agree: [list] n x n list, whether every pair of sections agrees.

    Return:
        [list] Indices of the sections of the subset.
    """

    import itertools

    # The largest subsets are tried first
    for size in range(n, 0, -1):
        for subset in itertools.combinations(range(n), size):
            if all(agree[i][j] for i, j in itertools.combinations(subset, 2)):
                return list(subset)

    return []


def fitCandidate(chunk_dirs, section_chunks, held_out, catalog_stars, config, mask=None, use_flat=False,
                 fit_vignetting=False, vignetting_coeff=None, ff_frames=256, position_hint=None):
    """ Fit a platepar from scratch (blind plate solve and distortion fit, see AutoPlatepar) independently on
        several sections (chunks spread in time), keep the sections which agree with each other apart from the
        pointing, take the one which scores best on the held-out chunks, and fit its photometry.

    Arguments:
        chunk_dirs: [dict] Directory with the CALSTARS file of every chunk, by the name of the chunk (the chunks
            can be from several files).
        section_chunks: [list] Clear chunks (name, jd, stars) the sections are picked from.
        held_out: [list] Clear chunks (name, jd, stars) which are not fitted on, the sections are scored on them.
        catalog_stars: [ndarray] Catalog stars (ra, dec, mag).
        config: [Config] Configuration.

    Keyword arguments:
        mask: [MaskStructure] Mask of the camera. None by default.
        use_flat: [bool] The stars were measured on flat fielded images. False by default.
        fit_vignetting: [bool] Fit the vignetting coefficient. False by default.
        vignetting_coeff: [float] Vignetting coefficient to use when it is not fitted (e.g. the one of the current
            platepar). None by default, in which case the default of AutoPlatepar is used.
        ff_frames: [int] Number of frames of a chunk. 256 by default.
        position_hint: [tuple] (ra, dec, radius) in degrees: the plate solver only searches within the radius of
            this pointing. None by default (the whole sky).

    Return:
        (platepar, info): [tuple] The candidate platepar (None if it could not be fitted) and a dictionary which
            describes the fit (sections, their agreement and scores, photometry, the reason of a failure).
    """

    import contextlib
    import io
    import itertools

    from RMS.Astrometry.AutoPlatepar import autoFitPlatepar

    info = {'sections': []}

    # Fit every section independently. The plate solver prints its progress, which is not logged
    fits = []
    for name, jd, stars in spreadChunks(section_chunks, N_SECTIONS):

        # Fit the section from scratch
        section = {'chunk': name, 'stars': len(stars)}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                platepar, paired_stars, _ = autoFitPlatepar(chunk_dirs[name], config, catalog_stars, ff_name=name,
                                                            mask=mask, use_flat=use_flat,
                                                            position_hint=position_hint,
                                                            stop_logodds=SOLVE_STOP_LOGODDS, verbose=False)

        except Exception as e:

            # A section which fails is recorded, the other sections are still fitted
            platepar, paired_stars = None, None
            section['error'] = repr(e)

        # The section is used only if its fit matched enough stars
        section['matched'] = len(paired_stars) if paired_stars else 0
        section['fitted'] = (platepar is not None) and (section['matched'] >= MIN_SECTION_STARS)
        info['sections'].append(section)

        if section['fitted']:
            fits.append((name, jd, platepar, section))

    # Enough sections are needed to compare them
    if len(fits) < MIN_SECTIONS:
        info['reason'] = 'only {:d} sections could be fitted'.format(len(fits))
        return None, info

    # Every section is judged on the held-out chunks: a section fitted to a wrong solution fails there and is
    #   dropped. Its photometry is fitted first on the chunks of the sections, as the candidate's is (below): the
    #   plate solver gives the default vignetting coefficient, which can be far from the camera's (e.g. a wide
    #   camera without a flat), and every section would fail on the photometry alone
    photometry_chunks = spreadChunks(section_chunks, 2*N_SECTIONS)
    scores = {}
    passing = []
    for name, jd, platepar, section in fits:
        if vignetting_coeff is not None:
            platepar.vignetting_coeff = vignetting_coeff
        section['photometry'] = fitPhotometry(platepar, photometry_chunks, catalog_stars, config,
                                              ff_frames=ff_frames, fit_vignetting=fit_vignetting)
        score = scorePlatepar(platepar, held_out, catalog_stars, config, ff_frames=ff_frames,
                              mask=getattr(mask, 'img', None))
        ok, reasons = judgeScore(score)
        section['held_out_ok'] = ok
        section['held_out_worst_zone'] = score.worstZoneShare()
        if not ok:
            section['held_out_reasons'] = reasons
            continue
        scores[name] = score
        passing.append((name, jd, platepar, section))

    if len(passing) < MIN_SECTIONS:
        info['reason'] = 'only {:d} sections pass on the held-out chunks'.format(len(passing))
        return None, info
    fits = passing

    # The width of the stars is the unit of the agreement
    fwhm = np.median([row[STAR_FWHM] for _, _, stars in section_chunks for row in stars])

    # Agreement of every pair of sections
    agree = [[True]*len(fits) for _ in fits]
    for i, j in itertools.combinations(range(len(fits)), 2):
        p95, dmax = sectionDifference(fits[i][2], fits[j][2], fits[i][1], fwhm)
        agree[i][j] = agree[j][i] = (p95 <= MAX_SECTION_DIFF_P95) and (dmax <= MAX_SECTION_DIFF_MAX)
        fits[i][3].setdefault('differences_fwhm', {})[fits[j][0]] = [round(p95, 3), round(dmax, 3)]

    # The largest group of sections which agree with each other
    subset = _agreeingSubset(len(fits), agree)
    info['agreeing_sections'] = [fits[i][0] for i in subset]
    if len(subset) < MIN_SECTIONS:
        info['reason'] = 'only {:d} sections agree'.format(len(subset))
        return None, info

    # The section which scores best on the held-out chunks (the worst zone, then the distance from the catalog)
    best = None
    for i in subset:
        score = scores[fits[i][0]]
        key = (np.nan_to_num(score.worstZoneShare(), nan=-1), -np.nan_to_num(score.dist, nan=99))
        if (best is None) or (key > best[0]):
            best = (key, i)

    name, jd, platepar, section = fits[best[1]]
    info['chosen_section'] = name

    # The photometry was fitted on the chunks of the sections (the held-out chunks are kept for the judgement)
    info['photometry'] = section['photometry']

    platepar.auto_recalibrated = False

    return platepar, info



### Controller: the calibration of a camera over the processed files ###

# Name of the state file of the automatic calibration in the output directory of the camera
STATE_NAME = 'auto_calibration_state.json'

# Name of the judgement of the calibration saved with the results of every file
FILE_RESULT_NAME = 'auto_calibration.json'

# The chunks of the files of this many minutes before the end of a file are used (a file can be short)
WINDOW_MINUTES = 30

# Minimum number of clear chunks in the window to judge or fit a platepar
MIN_CLEAR_CHUNKS = 8

# Maximum number of held-out chunks a platepar is judged on
MAX_HELD_OUT = 12

# Minutes of data between the attempts to fit a candidate: without a platepar or when it fails, and otherwise
#   (the periodic check whether a better platepar can be fitted)
RETRY_MINUTES = 10

# Without a platepar or when it fails, a candidate is also fitted again sooner if the clear chunks in the window grew
#   by this factor since the last attempt (e.g. a new camera with short files, whose first attempt had few chunks)
RETRY_EVIDENCE_GROWTH = 1.5

# A candidate which would replace a platepar has to be confirmed on the chunks of a later file, at least this many
#   minutes after the chunks it was fitted on. A platepar installed on trial (see TRIAL_MATCH_RATIO) is tested on
#   the chunks of as many minutes after its fit: it is kept if it passes, the previous platepar is put back if
#   that is clearly better on them, otherwise a better one is fitted
CONFIRM_MINUTES = 15

# A file older than the last attempt to fit by less than this many minutes was finished late by a worker and makes
#   no decision; an older one is older data processed again
LATE_MINUTES = 60

# The periodic fit of a camera whose platepar passes searches the sky within this many diagonals of the field of view
#   around its pointing
HINT_RADIUS_FOV = 2.0

# Number of star counts of chunks kept for the reference of the clear sky, and number of entries of the history
MAX_STAR_COUNTS = 5000
MAX_HISTORY = 200


def statePath(output_dir):
    """ Path of the state file of the automatic calibration of a camera.

    Arguments:
        output_dir: [str] Output directory of the camera.

    Return:
        [str] Path of the state file.
    """

    return os.path.join(output_dir, STATE_NAME)


class _StateLock(object):
    """ Exclusive lock of the state of the automatic calibration of a camera, so the workers which process its
        files in parallel judge and fit one at a time and see each other's results. """

    def __init__(self, output_dir, name=STATE_NAME + '.lock'):
        """ The lock of the camera with the given output directory (name: the lock file, e.g. of the backfill). """

        self.path = os.path.join(output_dir, name)
        self.fd = None

    def __enter__(self):
        """ Wait for the lock and take it. """

        import fcntl
        self.fd = open(self.path, 'w')
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *args):
        """ Release the lock. """

        import fcntl
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        self.fd.close()


def readState(output_dir):
    """ Read the state of the automatic calibration of a camera.

    Arguments:
        output_dir: [str] Output directory of the camera.

    Return:
        [dict] The state, a new state if there is none or it is damaged.
    """

    state = {'files': [], 'star_counts': [], 'last_attempt_jd': None, 'pending': None, 'trial': None,
             'installs': [], 'history': []}

    # The saved state replaces the defaults
    path = statePath(output_dir)
    if os.path.isfile(path):
        try:
            with open(path) as f:
                state.update(json.load(f))

        except Exception as e:

            # A damaged state is not fatal, the camera starts over
            log.warning("The state of the automatic calibration can't be read ({:s}), starting over".format(
                repr(e)))

    return state


def writeState(output_dir, state):
    """ Write the state of the automatic calibration atomically.

    Arguments:
        output_dir: [str] Output directory of the camera.
        state: [dict] The state.
    """

    # Written to a temporary file first, so a reader never sees a partial state
    path = statePath(output_dir)
    with open(path + '.tmp', 'w') as f:
        json.dump(state, f, indent=4, sort_keys=True, default=_jsonDefault)
    os.replace(path + '.tmp', path)


def _jsonDefault(value):
    """ JSON conversion of numpy values.

    Arguments:
        value: A value json can't convert itself.

    Return:
        A value json can convert.
    """

    # numpy numbers as Python numbers (non-finite as None), arrays as lists, anything else as text
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()

    return str(value)


def readChunks(results_dir, config):
    """ The chunks of a file from the CALSTARS file in its results directory.

    Arguments:
        results_dir: [str] Results directory of the file.
        config: [Config] Configuration.

    Return:
        (chunks, ff_frames): [tuple] Chunks as (name, jd, stars) sorted in time, and the frames of a chunk.
    """

    from RMS.Astrometry.Conversions import datetime2JD
    from RMS.Formats import CALSTARS
    from RMS.Formats.FFfile import filenameToDatetime

    # The CALSTARS file of the file
    names = [name for name in os.listdir(results_dir) if name.startswith('CALSTARS') and name.endswith('.txt')]
    if not names:
        return [], 256

    # The chunks with their times (a chunk whose name has no time is skipped)
    calstars, ff_frames = CALSTARS.readCALSTARS(results_dir, sorted(names)[0])
    chunks = []
    for name, stars in calstars:
        try:
            chunks.append((name, datetime2JD(filenameToDatetime(name)), np.array(stars, dtype=np.float64)))
        except Exception:
            continue

    return sorted(chunks, key=lambda chunk: chunk[1]), (ff_frames if ff_frames else 256)


def _platepar(path, use_flat):
    """ Read a platepar.

    Arguments:
        path: [str] Path of the platepar, None if there is none.
        use_flat: [bool] Whether the images are flat fielded.

    Return:
        [Platepar] The platepar, None if there is none or it can't be read.
    """

    from RMS.Formats.Platepar import Platepar

    if (path is None) or (not os.path.isfile(path)):
        return None

    # A platepar which can't be read (e.g. damaged) is none
    platepar = Platepar()
    try:
        if platepar.read(path, use_flat=use_flat) is False:
            return None

    except Exception as e:
        log.warning("Automatic calibration: the platepar {:s} can't be read: {:s}".format(path, repr(e)))
        return None

    return platepar


def installPlatepar(platepar, output_dir, config, reason, dry_run=False):
    """ Save a platepar fitted by the automatic calibration as the platepar of the camera (see
        MonitorNightReport.autoPlateparPath), keeping the previous one with the time it was replaced in its name.

    Arguments:
        platepar: [Platepar] The platepar.
        output_dir: [str] Output directory of the camera.
        config: [Config] Configuration.
        reason: [str] Why it is installed (logged).

    Keyword arguments:
        dry_run: [bool] Only save the platepar as a candidate (auto_candidate_<name>), which is not used. False by
            default.

    Return:
        [str] Path of the saved platepar.
    """

    import datetime

    from RMS.MonitorNightReport import _writePlatepar, autoPlateparPath

    # In a dry run the platepar is saved as a candidate, which is not used
    path = autoPlateparPath(output_dir, config)
    if dry_run:
        path = os.path.join(output_dir, 'auto_candidate_' + config.platepar_name)

    elif os.path.isfile(path):

        # The replaced platepar is kept with the time it was replaced, copied so the installed path never goes
        #   missing for the workers which read it
        import shutil
        stamp = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        shutil.copy2(path, os.path.join(output_dir, 'auto_replaced_{:s}_{:s}'.format(stamp, config.platepar_name)))

    # Written atomically over the old one, as the workers may be reading it
    _writePlatepar(platepar, path)
    log.info("Automatic calibration: {:s} platepar saved to {:s} ({:s})".format(
        'candidate' if dry_run else 'new', path, reason))

    return path


def autoCalibrateFile(results_dir, config, output_dir, platepar_path, mask=None, dry_run=False,
                      fit_vignetting=False):
    """ Judge the calibration of a camera on the clear chunks of a processed file (and of the files shortly
        before it), and fit a new platepar if there is none, if the platepar fails, or periodically to check
        whether a clearly better platepar can be fitted. A platepar is fitted from scratch on several sections,
        which have to agree, and judged on chunks which were not fitted on (see fitCandidate and judgeScore). A
        new platepar is installed right away if there is none; one which would replace a platepar has to be
        confirmed as better on a later file first, unless the platepar fails and is far worse: then the new one is
        installed right away on trial, and the previous one is put back if the new one fails on the following
        chunks while the previous one is clearly better there.

    Arguments:
        results_dir: [str] Results directory of the file (with its CALSTARS file).
        config: [Config] Configuration.
        output_dir: [str] Output directory of the camera.
        platepar_path: [str] Path of the platepar the file was processed with, None if there was none.

    Keyword arguments:
        mask: [MaskStructure] Mask of the camera. None by default.
        dry_run: [bool] Don't install the fitted platepars, only save them as candidates. False by default.
        fit_vignetting: [bool] Fit the vignetting coefficient also when a platepar exists (it is always fitted
            without one). False by default.

    Return:
        [dict] The judgement of the file, also saved to its results directory: whether the sky was clear, the
            score of the platepar, the candidate and what was done ('action'), and 'installed' with the path of a
            newly installed platepar.
    """

    # The chunks of the file, without them there is nothing to judge
    use_flat = config.use_flat
    chunks, ff_frames = readChunks(results_dir, config)
    result = {'results_dir': results_dir, 'platepar': platepar_path, 'action': None}
    if not chunks:
        result['action'] = 'no stars'
        return _saveFileResult(results_dir, result)

    width, height = int(config.width), int(config.height)
    mask_img = getattr(mask, 'img', None)

    with _StateLock(output_dir):

        state = readState(output_dir)

        # The reference of the clear sky: the clearest chunks seen by the camera
        state['star_counts'] = (state['star_counts'] + [len(stars) for _, _, stars in chunks])[-MAX_STAR_COUNTS:]
        reference = np.percentile(state['star_counts'], CLEAR_REFERENCE_PERCENTILE)

        # The clear chunks of this file, and of the files within the window before it
        clear_names = [name for name, _, _ in clearChunks(chunks, width, height, mask=mask_img,
                                                          reference_stars=reference)]
        end_jd = chunks[-1][1]
        state['files'] = [entry for entry in state['files'] if entry['results_dir'] != results_dir]
        state['files'].append({'results_dir': results_dir, 'start_jd': chunks[0][1], 'end_jd': end_jd,
                               'clear': clear_names})
        state['files'] = sorted(state['files'], key=lambda entry: entry['end_jd'])[-MAX_HISTORY:]

        # The clear chunks of the files in the window, with the directories of their CALSTARS files
        window, chunk_dirs = [], {}
        for entry in state['files']:
            if (entry['end_jd'] > end_jd) or (entry['end_jd'] < end_jd - WINDOW_MINUTES/1440.0) \
                    or (not entry['clear']) or (not os.path.isdir(entry['results_dir'])):
                continue

            # The chunks of this file are already read
            file_chunks = chunks if (entry['results_dir'] == results_dir) else readChunks(entry['results_dir'],
                                                                                         config)[0]
            for chunk in file_chunks:
                if chunk[0] in entry['clear']:
                    window.append(chunk)
                    chunk_dirs[chunk[0]] = entry['results_dir']

        result['clear_chunks'] = len(clear_names)
        result['window_clear_chunks'] = len(window)

        # Without enough clear sky, nothing is judged or fitted
        if len(window) < MIN_CLEAR_CHUNKS:
            result['action'] = 'not enough clear sky'
            writeState(output_dir, state)
            return _saveFileResult(results_dir, result)

        # Chunks which are fitted on, and held-out chunks the platepars are judged on. The held-out chunks
        #   prefer this file, so a pending candidate is confirmed on chunks it was not fitted on
        window = sorted(window, key=lambda chunk: chunk[1])
        held_out = spreadChunks(window[1::2], MAX_HELD_OUT)
        held_names = set(name for name, _, _ in held_out)
        pool = [chunk for chunk in window if chunk[0] not in held_names]

        catalog_stars = _loadCatalog(config, end_jd)

        # The platepar in use is judged: one installed while this file was processed (e.g. by another worker) is
        #   newer than the one it was processed with. In a dry run the saved candidate stands for the installed
        #   platepar, so the decisions are the ones which would be made
        platepar_path = _plateparInUse(output_dir, config, platepar_path, dry_run)
        result['judged_platepar'] = platepar_path

        # Judge the current platepar
        current = _platepar(platepar_path, use_flat)
        current_score = None
        if current is not None:
            current_score = scorePlatepar(current, held_out, catalog_stars, config,
                                          ff_frames=ff_frames, mask=mask_img)
            current_ok, current_reasons = judgeScore(current_score)
            result['current'] = {'ok': current_ok, 'reasons': current_reasons, 'score': current_score.toDict()}
            log.info("Automatic calibration: the platepar {:s} on {:d} clear chunks{:s}".format(
                'passes' if current_ok else 'FAILS', len(held_out),
                '' if current_ok else ': ' + '; '.join(current_reasons)))

        # A platepar on trial is tested on the chunks after the ones it was fitted on, once they span
        #   CONFIRM_MINUTES. No new candidate is fitted meanwhile. The trial ends if another platepar is in use
        trial = state.get('trial')
        if trial and ((current is None) or (os.path.abspath(platepar_path) != os.path.abspath(trial['path']))
                      or (not os.path.isfile(trial['previous']))):
            _removeFile(trial['previous'])
            state['trial'] = trial = None

        if trial:

            # The chunks later than the ones the platepar was fitted on, it is tested until they span
            #   CONFIRM_MINUTES
            later = [chunk for chunk in window if chunk[1] > trial['fitted_jd']]
            if (not later) or (later[-1][1] - trial['fitted_jd'] < CONFIRM_MINUTES/1440.0):
                result['action'] = 'the platepar on trial is tested'
                writeState(output_dir, state)
                return _saveFileResult(results_dir, result)

            # The platepar on trial and the previous one are scored on the same later chunks
            confirm_chunks = spreadChunks(later, MAX_HELD_OUT)
            previous = _platepar(trial['previous'], use_flat)
            trial_score = scorePlatepar(current, confirm_chunks, catalog_stars, config,
                                        ff_frames=ff_frames, mask=mask_img)
            trial_ok, trial_reasons = judgeScore(trial_score)
            previous_better, reason = False, ''
            if (not trial_ok) and (previous is not None):
                previous_score = scorePlatepar(previous, confirm_chunks, catalog_stars, config,
                                               ff_frames=ff_frames, mask=mask_img)
                previous_better, reason = candidateIsBetter(previous_score, trial_score)

            result['trial'] = {'kept': trial_ok, 'reasons': trial_reasons, 'rolled_back': previous_better,
                               'score': trial_score.toDict()}
            state['trial'] = None
            _removeFile(trial['previous'])

            # A platepar which passes is kept
            if trial_ok:
                log.info("Automatic calibration: the platepar on trial passes, it is kept")
                result['action'] = 'kept the platepar on trial'
                writeState(output_dir, state)
                return _saveFileResult(results_dir, result)

            # The previous platepar is put back if it is clearly better. The version of the rejected platepar is
            #   reported, so the files calibrated with it are calibrated again
            if previous_better:
                if os.path.isfile(trial['path']):
                    result['replaced_platepar_mtime'] = os.path.getmtime(trial['path'])
                path = installPlatepar(previous, output_dir, config, 'rolled back: ' + reason, dry_run=dry_run)
                state['installs'].append({'jd': end_jd, 'path': path, 'reason': 'rolled back: ' + reason,
                                          'dry_run': dry_run})
                result['action'] = _dryRunAction('rolled back the platepar on trial', dry_run)
                result['installed'] = None if dry_run else path
                writeState(output_dir, state)
                return _saveFileResult(results_dir, result)

            # Otherwise a better platepar is fitted below, as the platepar in use fails
            log.info("Automatic calibration: the platepar on trial fails ({:s}), a better one is fitted".format(
                '; '.join(trial_reasons)))

        # A pending candidate is confirmed (or dropped) on the chunks after the ones it was fitted on, once they
        #   span CONFIRM_MINUTES, comparing it with the current platepar on the same chunks. No new candidate is
        #   fitted meanwhile
        pending = state.get('pending')
        if pending and ((current is None) or (not os.path.isfile(pending.get('path', '')))):
            state['pending'] = pending = None

        if pending:

            # The chunks later than the ones the candidate was fitted on, it waits until they span
            #   CONFIRM_MINUTES
            later = [chunk for chunk in window if chunk[1] > pending['fitted_jd']]
            if (not later) or (later[-1][1] - pending['fitted_jd'] < CONFIRM_MINUTES/1440.0):
                result['action'] = 'candidate waits for the confirmation'
                writeState(output_dir, state)
                return _saveFileResult(results_dir, result)

            # Both platepars are scored on the same later chunks
            confirm_chunks = spreadChunks(later, MAX_HELD_OUT)
            candidate = _platepar(pending['path'], use_flat)
            candidate_score = scorePlatepar(candidate, confirm_chunks, catalog_stars, config,
                                            ff_frames=ff_frames, mask=mask_img)
            current_confirm_score = scorePlatepar(current, confirm_chunks, catalog_stars, config,
                                                  ff_frames=ff_frames, mask=mask_img)
            better, reason = candidateIsBetter(candidate_score, current_confirm_score)
            result['pending'] = {'better': better, 'reason': reason, 'score': candidate_score.toDict()}
            state['pending'] = None
            _removeFile(pending['path'])

            # A confirmed candidate is installed
            if better:
                result['replaced_platepar_mtime'] = _mtime(platepar_path)
                path = installPlatepar(candidate, output_dir, config, 'confirmed: ' + reason, dry_run=dry_run)
                state['installs'].append({'jd': end_jd, 'path': path, 'reason': 'confirmed: ' + reason,
                                          'dry_run': dry_run})
                result['action'] = _dryRunAction('installed the confirmed candidate', dry_run)
                result['installed'] = None if dry_run else path
                writeState(output_dir, state)
                return _saveFileResult(results_dir, result)

            log.info("Automatic calibration: the candidate was not confirmed ({:s})".format(reason))

        # Fit a candidate without a platepar or when it fails (retried after a while, or sooner with clearly more
        #   clear chunks), and periodically. A file which a worker finished late (older than the last attempt) makes
        #   no decision, so the decisions never go back to an older window
        current_fails = (current is None) or (not result['current']['ok'])
        last = state.get('last_attempt_jd')
        interval = RETRY_MINUTES if current_fails else config.monitor_auto_calibration_interval
        more_evidence = current_fails and (len(window) >= RETRY_EVIDENCE_GROWTH*state.get('last_attempt_clear', 0))

        # A file older than the last attempt by less than LATE_MINUTES is a late worker; a much older one is older
        #   data processed again, which is judged as new
        late = (last is not None) and (last - LATE_MINUTES/1440.0 < end_jd < last)
        if late or ((last is not None) and (end_jd >= last) and (end_jd - last < interval/1440.0)
                    and (not more_evidence)):
            result['action'] = 'judged'
            writeState(output_dir, state)
            return _saveFileResult(results_dir, result)

        # Record the attempt, then fit the candidate
        state['last_attempt_jd'] = end_jd
        state['last_attempt_clear'] = len(window)
        # A platepar which passes gives the pointing to search around (the blind search of the whole sky takes
        #   minutes per section); without one, or when it fails (e.g. the camera moved), the whole sky is searched
        position_hint = None if current_fails else _positionHint(current, end_jd)
        candidate, info = fitCandidate(chunk_dirs, pool, held_out, catalog_stars, config, mask=mask,
                                       use_flat=use_flat, fit_vignetting=(fit_vignetting or (current is None)),
                                       vignetting_coeff=(None if current is None else current.vignetting_coeff),
                                       ff_frames=ff_frames, position_hint=position_hint)
        result['candidate'] = info

        # Without a candidate (e.g. too few sections agree), the next attempt is made later
        if candidate is None:
            result['action'] = 'no candidate: ' + info.get('reason', '')
            log.info("Automatic calibration: no candidate platepar ({:s})".format(info.get('reason', '')))
            writeState(output_dir, state)
            return _saveFileResult(results_dir, result)

        # Judge the candidate on the held-out chunks, against the current platepar
        candidate_score = scorePlatepar(candidate, held_out, catalog_stars, config,
                                        ff_frames=ff_frames, mask=mask_img)
        better, reason = candidateIsBetter(candidate_score, current_score)
        result['candidate']['better'] = better
        result['candidate']['reason'] = reason
        result['candidate']['score'] = candidate_score.toDict()
        log.info("Automatic calibration: candidate platepar {:s} ({:s})".format(
            'is better' if better else 'is not used', reason))

        if better and (current is None):

            # Without a platepar the candidate is installed right away: its sections agreed and it passed on
            #   held-out chunks
            path = installPlatepar(candidate, output_dir, config, reason, dry_run=dry_run)
            state['installs'].append({'jd': end_jd, 'path': path, 'reason': reason, 'dry_run': dry_run})
            result['action'] = _dryRunAction('installed the first platepar', dry_run)
            result['installed'] = None if dry_run else path

        elif better and candidateIsFarBetter(candidate_score, current_score):

            # A candidate far better than a failing platepar is installed right away, on trial: it is tested on
            #   the following chunks, and the replaced platepar is kept to be put back
            import shutil

            previous_path = os.path.join(output_dir, 'auto_previous_' + config.platepar_name)
            shutil.copyfile(platepar_path, previous_path)
            result['replaced_platepar_mtime'] = _mtime(platepar_path)
            path = installPlatepar(candidate, output_dir, config, 'on trial: ' + reason, dry_run=dry_run)
            state['trial'] = {'path': path, 'previous': previous_path,
                              'fitted_jd': max(chunk[1] for chunk in window), 'reason': reason}
            state['installs'].append({'jd': end_jd, 'path': path, 'reason': 'on trial: ' + reason,
                                      'dry_run': dry_run})
            result['action'] = _dryRunAction('installed the platepar on trial', dry_run)
            result['installed'] = None if dry_run else path

        elif better:

            # A candidate which would replace a platepar waits for the confirmation on a later file
            path = os.path.join(output_dir, 'auto_pending_' + config.platepar_name)
            from RMS.MonitorNightReport import _writePlatepar
            _writePlatepar(candidate, path)
            state['pending'] = {'path': path, 'fitted_jd': max(chunk[1] for chunk in window), 'reason': reason}
            result['action'] = 'candidate waits for the confirmation'

        else:
            result['action'] = 'candidate not better'

        writeState(output_dir, state)

    return _saveFileResult(results_dir, result)


def _positionHint(platepar, jd):
    """ The pointing of a platepar at a time, as a hint for the plate solver: the RA and Dec of the centre of the
        image and a search radius of HINT_RADIUS_FOV diagonals of the field of view.

    Arguments:
        platepar: [Platepar] The platepar.
        jd: [float] Julian date.

    Return:
        [tuple] (ra, dec, radius) in degrees, None if it can't be computed.
    """

    from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP
    from RMS.Astrometry.Conversions import jd2Date

    try:
        _, ra, dec, _ = xyToRaDecPP([jd2Date(jd)], [platepar.X_res/2.0], [platepar.Y_res/2.0], [1], platepar,
                                    extinction_correction=False, jd_time=False)
        radius = HINT_RADIUS_FOV*float(np.hypot(platepar.fov_h, platepar.fov_v))

        return float(ra[0]), float(dec[0]), radius

    except Exception as e:
        log.warning("Automatic calibration: no pointing hint ({:s})".format(repr(e)))
        return None


def _plateparInUse(output_dir, config, platepar_path, dry_run):
    """ The platepar a camera uses now: the one installed by the automatic calibration if it is newer than the
        platepar a file was processed with (see MonitorNightReport.selectPlatepar), in a dry run the saved candidate.

    Arguments:
        output_dir: [str] Output directory of the camera.
        config: [Config] Configuration.
        platepar_path: [str] Platepar the file was processed with, None if there was none.
        dry_run: [bool] Whether the platepars are only saved as candidates.

    Return:
        [str] Path of the platepar in use, None if there is none.
    """

    from RMS.MonitorNightReport import autoPlateparPath

    # The platepar installed by the automatic calibration, or the candidate of a dry run
    installed_path = autoPlateparPath(output_dir, config)
    if dry_run:
        installed_path = os.path.join(output_dir, 'auto_candidate_' + config.platepar_name)

    if not os.path.isfile(installed_path):
        return platepar_path

    # The newer one is in use
    if (platepar_path is None) or (not os.path.isfile(platepar_path)) \
            or (os.path.getmtime(installed_path) >= os.path.getmtime(platepar_path)):
        return installed_path

    return platepar_path


def _mtime(path):
    """ Modification time of a file (the version of a platepar), None if it doesn't exist. """

    try:
        return os.path.getmtime(path)
    except (OSError, TypeError):
        return None


def _dryRunAction(action, dry_run):
    """ The action of the automatic calibration as it is reported, marked in a dry run (nothing is installed). """

    return ('dry run: ' + action) if dry_run else action


def _removeFile(path):
    """ Remove a file of the automatic calibration which is no longer needed, if it exists. """

    try:
        if path and os.path.isfile(path):
            os.remove(path)

    except OSError as e:
        log.warning("Automatic calibration: {:s} can't be removed: {:s}".format(path, repr(e)))


def _loadCatalog(config, jd):
    """ The catalog stars down to the limiting magnitude of the config, with the proper motion at the given time.

    Arguments:
        config: [Config] Configuration.
        jd: [float] Julian date.

    Return:
        [ndarray] Catalog stars (ra, dec, mag).
    """

    from RMS.Astrometry.AutoPlatepar import loadCatalogStars

    return loadCatalogStars(config, config.catalog_mag_limit, jd=jd)


def _saveFileResult(results_dir, result):
    """ Save the judgement of the calibration of a file to its results directory.

    Arguments:
        results_dir: [str] Results directory of the file.
        result: [dict] The judgement.

    Return:
        [dict] The judgement.
    """

    try:

        # Written to a temporary file first, so the status never reads a partial result
        with open(os.path.join(results_dir, FILE_RESULT_NAME + '.tmp'), 'w') as f:
            json.dump(result, f, indent=4, sort_keys=True, default=_jsonDefault)
        os.replace(os.path.join(results_dir, FILE_RESULT_NAME + '.tmp'), os.path.join(results_dir, FILE_RESULT_NAME))

    except Exception as e:
        log.warning("The result of the automatic calibration can't be saved: {:s}".format(repr(e)))

    return result
