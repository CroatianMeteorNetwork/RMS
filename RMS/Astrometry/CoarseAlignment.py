""" Coarse alignment of a platepar on the stars of an image, before the fits of the pointing polish it.

The fits of the pointing are local: they find the pointing within a few pixels of where they start. A platepar
whose pointing is a few degrees off, or turned by several degrees, has to be brought there first. With the
platepar, the catalog stars are projected onto the image, and every pair of a catalog star and an image star closer
than the largest shift gives a displacement. The displacements of the right pairs all agree, so they make a sharp
peak in the 2D histogram of the displacements, while the wrong pairs spread over it. The roll is found by repeating
this for a set of trial rolls of the camera and keeping the sharpest peak. The result is the pointing (turned with
Platepar.PointingOffsets) that brings the catalog stars onto the image stars, and the number of pairs that agree
with it.
"""

from __future__ import print_function, division, absolute_import

import copy

import numpy as np
from scipy.spatial import cKDTree

from RMS.Astrometry.ApplyAstrometry import raDecToXYPP
from RMS.Formats.Platepar import PointingOffsets, offsetsJacobian


# Largest shift of the pointing searched by default (deg on the sky) and largest roll (deg)
MAX_SHIFT_DEG = 5.0
MAX_ROLL_DEG = 20.0

# Step of the trial rolls of the first stage (deg), and the bin of its histogram, wide enough for the stars at a
#   quarter of the image diagonal to stay in the bin of their shift when the roll is off by half a step (px)
ROLL_STEP_DEG = 2.0
COARSE_BIN_PX = 8.0

# Step of the trial rolls of the second stage, about the best roll of the first one (deg), and its bin (px)
FINE_ROLL_STEP_DEG = 0.25
FINE_BIN_PX = 2.0

# Number of catalog stars used, as a multiple of the number of image stars (the brightest ones near the image),
#   and at least this many
CATALOG_FACTOR = 3
MIN_CATALOG_STARS = 100


def _displacements(img_tree, cat_x, cat_y, max_shift):
    """ Displacements from the catalog stars to all image stars closer than the largest shift.

    Arguments:
        img_tree: [cKDTree] Tree of the image star positions (x, y).
        cat_x: [ndarray] X positions of the catalog stars (px).
        cat_y: [ndarray] Y positions of the catalog stars (px).
        max_shift: [float] Largest shift (px).

    Return:
        (dx, dy, img_idx, cat_idx): [tuple of ndarrays] Displacements, image minus catalog (px), and the indices
            of the image and the catalog star of every pair.
    """

    empty = (np.array([]), np.array([]), np.array([], dtype=int), np.array([], dtype=int))
    finite = np.where(np.isfinite(cat_x) & np.isfinite(cat_y))[0]
    cat_xy = np.column_stack([cat_x[finite], cat_y[finite]])
    if len(cat_xy) == 0:
        return empty

    # All pairs of a catalog star and an image star within the largest shift
    neighbours = img_tree.query_ball_point(cat_xy, max_shift)
    counts = np.array([len(n) for n in neighbours])
    if counts.sum() == 0:
        return empty
    img_idx = np.concatenate([np.array(n, dtype=int) for n in neighbours])
    cat_idx = np.repeat(np.arange(len(cat_xy)), counts)
    img_xy = img_tree.data

    return img_xy[img_idx, 0] - cat_xy[cat_idx, 0], img_xy[img_idx, 1] - cat_xy[cat_idx, 1], img_idx, \
        finite[cat_idx]


def _peak(dx, dy, max_shift, bin_px):
    """ The peak of the 2D histogram of the displacements: the number of displacements in the 3x3 bins about it and
        its position.

    Arguments:
        dx: [ndarray] Displacements along x (px).
        dy: [ndarray] Displacements along y (px).
        max_shift: [float] Largest shift (px).
        bin_px: [float] Size of a bin (px).

    Return:
        (votes, x, y): [tuple] Number of displacements about the peak and the centre of the peak (px).
    """

    if len(dx) == 0:
        return 0, 0.0, 0.0

    # The histogram, and the sums of its 3x3 bins
    n_bins = int(np.ceil(2*max_shift/bin_px)) + 1
    edges = np.linspace(-n_bins*bin_px/2, n_bins*bin_px/2, n_bins + 1)
    hist, _, _ = np.histogram2d(dx, dy, bins=[edges, edges])
    padded = np.pad(hist, 1)
    window = sum(padded[i:i + n_bins, j:j + n_bins] for i in range(3) for j in range(3))

    # The bin with the most displacements about it
    i, j = np.unravel_index(np.argmax(window), window.shape)
    centres = (edges[:-1] + edges[1:])/2

    return int(window[i, j]), centres[i], centres[j]


def fitPairs(platepar, jd, img_xy, cat_stars, fit_scale=False, loss_scale=FINE_BIN_PX):
    """ Fit the shift and the roll of the pointing (and the scale) on pairs of image and catalog stars, by least
        squares with a robust loss (a wrong pair pulls less than its share).

    Arguments:
        platepar: [Platepar] Platepar with the starting pointing (not changed).
        jd: [float] Julian date of the image stars.
        img_xy: [ndarray] Image stars (x, y) of the pairs (px).
        cat_stars: [ndarray] Catalog stars (ra, dec, ...) of the pairs.

    Keyword arguments:
        fit_scale: [bool] Fit the scale too. False by default.
        loss_scale: [float] Distance (px) beyond which a pair pulls less than in plain least squares.
            FINE_BIN_PX by default.

    Return:
        [Platepar] A copy of the platepar with the fitted pointing.
    """

    import scipy.optimize

    fitted = copy.deepcopy(platepar)
    pointing = PointingOffsets(platepar)

    def _residuals(offsets):
        """ Distances of the image stars from their catalog stars (px) with the offsets of the pointing. """

        pointing.apply(fitted, offsets)
        x, y = raDecToXYPP(cat_stars[:, 0], cat_stars[:, 1], jd, fitted)
        res = np.concatenate([x - img_xy[:, 0], y - img_xy[:, 1]])
        res[~np.isfinite(res)] = 2*loss_scale

        return res

    # The derivatives step the offsets by a fixed fraction of a pixel (see Platepar.offsetsJacobian)
    fit = scipy.optimize.least_squares(_residuals, np.zeros(4 if fit_scale else 3), loss='soft_l1',
                                       f_scale=loss_scale, jac=offsetsJacobian(_residuals))
    pointing.apply(fitted, fit.x)

    return fitted


def _alignOnce(platepar, jd, img_x, img_y, catalog_stars, max_shift_deg, max_roll_deg):
    """ One round of coarseAlignment, with the given search range.

    Arguments:
        platepar: [Platepar] Platepar with the starting pointing (not changed).
        jd: [float] Julian date of the image stars.
        img_x: [ndarray] X positions of the image stars (px).
        img_y: [ndarray] Y positions of the image stars (px).
        catalog_stars: [ndarray] Catalog stars (ra, dec, mag), (N, 3) or more columns.

        max_shift_deg: [float] Largest shift of the pointing searched (deg on the sky).
        max_roll_deg: [float] Largest roll searched (deg).

    Return:
        (aligned, votes, votes_start): [tuple]
            aligned: [Platepar] A copy of the platepar with the pointing rolled and shifted.
            votes: [int] Number of pairs of catalog and image stars which agree with the aligned pointing (within
                1.5 FINE_BIN_PX).
            votes_start: [int] The same for the starting pointing.
    """

    pointing = PointingOffsets(platepar)
    max_shift = max_shift_deg*pointing.F_scale

    # The image stars with a position
    img_x, img_y = np.asarray(img_x, dtype=np.float64), np.asarray(img_y, dtype=np.float64)
    finite = np.isfinite(img_x) & np.isfinite(img_y)
    img_x, img_y = img_x[finite], img_y[finite]

    # Nothing to align without stars
    if (len(img_x) < 3) or (len(catalog_stars) < 3):
        return copy.deepcopy(platepar), 0, 0
    img_tree = cKDTree(np.column_stack([img_x, img_y]))

    # The brightest catalog stars which the starting pointing puts within the largest shift of the image
    catalog_stars = np.asarray(catalog_stars, dtype=np.float64)
    cat_x, cat_y = raDecToXYPP(catalog_stars[:, 0], catalog_stars[:, 1], jd, platepar)
    near = np.isfinite(cat_x) & np.isfinite(cat_y) & (cat_x > -max_shift) & (cat_x < platepar.X_res + max_shift) \
        & (cat_y > -max_shift) & (cat_y < platepar.Y_res + max_shift)
    candidates = catalog_stars[near]
    n_catalog = max(MIN_CATALOG_STARS, CATALOG_FACTOR*len(img_x))
    candidates = candidates[np.argsort(candidates[:, 2])[:n_catalog]]
    if len(candidates) < 3:
        return copy.deepcopy(platepar), 0, 0

    def _project(roll_deg):
        """ Image positions of the catalog stars with the camera rolled by the given angle. """

        pp = copy.copy(platepar)
        pointing.apply(pp, [0.0, 0.0, np.radians(roll_deg)/pointing.roll_unit])
        return raDecToXYPP(candidates[:, 0], candidates[:, 1], jd, pp)

    def _rollPeak(roll_deg, bin_px):
        """ The peak of the displacements with the camera rolled by the given angle. """

        dx, dy, img_idx, cat_idx = _displacements(img_tree, *_project(roll_deg), max_shift=max_shift)
        return _peak(dx, dy, max_shift, bin_px), dx, dy, img_idx, cat_idx

    # First stage: trial rolls over the whole range, with wide bins
    rolls = np.arange(-max_roll_deg, max_roll_deg + ROLL_STEP_DEG/2, ROLL_STEP_DEG) if max_roll_deg > 0 \
        else np.array([0.0])
    coarse = [(_rollPeak(roll, COARSE_BIN_PX)[0][0], roll) for roll in rolls]
    best_roll = max(coarse, key=lambda entry: (entry[0], -abs(entry[1])))[1]

    # Second stage: finer rolls about the best one, with narrow bins
    if max_roll_deg > 0:
        fine_rolls = best_roll + np.arange(-ROLL_STEP_DEG, ROLL_STEP_DEG + FINE_ROLL_STEP_DEG/2,
                                           FINE_ROLL_STEP_DEG)
    else:
        fine_rolls = np.array([0.0])
    best = None
    for roll in fine_rolls:
        (votes, x, y), dx, dy, img_idx, cat_idx = _rollPeak(roll, FINE_BIN_PX)
        if (best is None) or (votes > best[0]):
            best = (votes, x, y, roll, dx, dy, img_idx, cat_idx)
    votes, x, y, roll, dx, dy, img_idx, cat_idx = best

    # The shift is the mean of the displacements about the peak
    about = (np.abs(dx - x) <= 1.5*FINE_BIN_PX) & (np.abs(dy - y) <= 1.5*FINE_BIN_PX)
    if np.any(about):
        x, y = np.mean(dx[about]), np.mean(dy[about])

    # The pairs which agree with the starting pointing
    dx0, dy0, _, _ = _displacements(img_tree, *_project(0.0), max_shift=max_shift)
    votes_start = int(np.count_nonzero((np.abs(dx0) <= 1.5*FINE_BIN_PX) & (np.abs(dy0) <= 1.5*FINE_BIN_PX)))

    # Roll the camera, then shift it in its rolled axes: the stars have to move by the displacement, so the optical
    #   axis turns the other way
    aligned = copy.deepcopy(platepar)
    pointing.apply(aligned, [0.0, 0.0, np.radians(roll)/pointing.roll_unit])
    PointingOffsets(aligned).apply(aligned, [-x, -y, 0.0])

    # The bins leave the roll a fraction of a degree off, which far from the centre is a few pixels: the pointing
    #   is fitted on the pairs about the peak, by least squares with a robust loss
    if np.count_nonzero(about) >= 3:
        aligned = fitPairs(aligned, jd, img_tree.data[img_idx[about]], candidates[cat_idx[about]])
    aligned.updateRefAltAz()

    return aligned, int(votes), votes_start


# Search range of the second round, from the result of the first (deg): with the lens distortion a shift of the
#   pointing does not move all stars by the same amount, so the first round of a large shift leaves a few pixels
REFINE_SHIFT_DEG = 0.5
REFINE_ROLL_DEG = 2.0


def coarseAlignment(platepar, jd, img_x, img_y, catalog_stars, max_shift_deg=MAX_SHIFT_DEG,
                    max_roll_deg=MAX_ROLL_DEG):
    """ Find the shift and the roll of the pointing which bring the most catalog stars onto the image stars: a
        round over the whole search range, and a second round over a small range about its result.

    Arguments:
        platepar: [Platepar] Platepar with the starting pointing (not changed).
        jd: [float] Julian date of the image stars.
        img_x: [ndarray] X positions of the image stars (px).
        img_y: [ndarray] Y positions of the image stars (px).
        catalog_stars: [ndarray] Catalog stars (ra, dec, mag), (N, 3) or more columns.

    Keyword arguments:
        max_shift_deg: [float] Largest shift of the pointing searched (deg on the sky). MAX_SHIFT_DEG by default.
        max_roll_deg: [float] Largest roll searched (deg). MAX_ROLL_DEG by default.

    Return:
        (aligned, votes, votes_start): [tuple]
            aligned: [Platepar] A copy of the platepar with the pointing rolled and shifted.
            votes: [int] Number of pairs of catalog and image stars which agree with the aligned pointing (within
                1.5 FINE_BIN_PX).
            votes_start: [int] The same for the starting pointing.
    """

    # The whole range, then a small range about its result
    aligned, votes, votes_start = _alignOnce(platepar, jd, img_x, img_y, catalog_stars, max_shift_deg,
                                             max_roll_deg)
    refined, votes_refined, _ = _alignOnce(aligned, jd, img_x, img_y, catalog_stars,
                                           min(REFINE_SHIFT_DEG, max_shift_deg),
                                           min(REFINE_ROLL_DEG, max_roll_deg))

    # The second round is kept if more stars agree with it
    if votes_refined >= votes:
        return refined, votes_refined, votes_start

    return aligned, votes, votes_start


# The coarse alignment is used only if this many more pairs agree with it than with the starting pointing: twice as
#   many, at least MIN_EXTRA_VOTES or MIN_EXTRA_FRACTION of the image stars more, and at least MIN_VOTES in all
MIN_VOTES = 8
MIN_EXTRA_VOTES = 5
MIN_EXTRA_FRACTION = 0.1


def alignIfBetter(platepar, jd, img_x, img_y, catalog_stars, max_shift_deg=MAX_SHIFT_DEG,
                  max_roll_deg=MAX_ROLL_DEG):
    """ The coarse alignment of the platepar (coarseAlignment), if clearly more pairs of catalog and image stars
        agree with it than with the starting pointing. A pointing which already puts the stars near their catalog
        stars is left to the fits, which are more precise.

    Arguments:
        platepar: [Platepar] Platepar with the starting pointing (not changed).
        jd: [float] Julian date of the image stars.
        img_x: [ndarray] X positions of the image stars (px).
        img_y: [ndarray] Y positions of the image stars (px).
        catalog_stars: [ndarray] Catalog stars (ra, dec, mag), (N, 3) or more columns.

    Keyword arguments:
        max_shift_deg: [float] Largest shift of the pointing searched (deg on the sky). MAX_SHIFT_DEG by default.
        max_roll_deg: [float] Largest roll searched (deg). MAX_ROLL_DEG by default.

    Return:
        [Platepar] A copy of the platepar with the aligned pointing, or None if it is not clearly better.
    """

    aligned, votes, votes_start = coarseAlignment(platepar, jd, img_x, img_y, catalog_stars,
                                                  max_shift_deg=max_shift_deg, max_roll_deg=max_roll_deg)

    # Clearly more pairs agree with the alignment than with the starting pointing
    extra = max(MIN_EXTRA_VOTES, MIN_EXTRA_FRACTION*len(img_x))
    if (votes >= MIN_VOTES) and (votes >= 2*votes_start) and (votes >= votes_start + extra):
        return aligned

    return None
