""" Equivalence tests for the KD-tree rewrites of the astrometric fitting hot spots.

    Every rewritten routine is compared against a straightforward dense/loop reference implementation
    (the code it replaced) on random data.
"""

from __future__ import print_function, division, absolute_import

import numpy as np
import pytest
from scipy.spatial import cKDTree

from RMS.Formats.Platepar import _nearestCatalogStars, _raDecToUnitVectors
from RMS.Math import angularSeparation


### Reference implementations ###

def _randomSky(rng, n, ra_centre=120.0, dec_centre=30.0, half_width=25.0):
    """ Random RA/Dec (deg) in a box around a centre.

    Arguments:
        rng: [np.random.RandomState] Random generator.
        n: [int] Number of points to draw.

    Keyword arguments:
        ra_centre: [float] Right ascension of the box centre (deg). 120 by default.
        dec_centre: [float] Declination of the box centre (deg). 30 by default.
        half_width: [float] Half width of the box in both coordinates (deg). 25 by default.

    Return:
        ra: [ndarray] Right ascensions wrapped into [0, 360) deg.
        dec: [ndarray] Declinations clipped to +/- 89.9 deg, so no point sits exactly on a pole.
    """

    # Draw uniformly inside the box, keeping the declination off the poles
    ra = ra_centre + rng.uniform(-half_width, half_width, n)
    dec = np.clip(dec_centre + rng.uniform(-half_width, half_width, n), -89.9, 89.9)

    return ra % 360, dec


def _denseNearestReference(ra_det, dec_det, ra_cat, dec_cat):
    """ The dense N x M angularSeparation matrix with row-wise argmin/min that the KD-tree replaced.

    Arguments:
        ra_det: [ndarray] Right ascensions of the detections (deg).
        dec_det: [ndarray] Declinations of the detections (deg).
        ra_cat: [ndarray] Right ascensions of the catalog stars (deg).
        dec_cat: [ndarray] Declinations of the catalog stars (deg).

    Return:
        nearest_indices: [ndarray] Index of the closest catalog star for every detection.
        nearest_sep: [ndarray] Angular separation to that star (rad).
    """

    # Build the full detection x catalog separation matrix and reduce it row by row
    ra_det_rad = np.radians(ra_det)[:, np.newaxis]
    dec_det_rad = np.radians(dec_det)[:, np.newaxis]
    ra_cat_rad = np.radians(ra_cat)[np.newaxis, :]
    dec_cat_rad = np.radians(dec_cat)[np.newaxis, :]
    sep_matrix = angularSeparation(ra_det_rad, dec_det_rad, ra_cat_rad, dec_cat_rad)

    return np.argmin(sep_matrix, axis=1), np.min(sep_matrix, axis=1)


def _oneToOneReference(nearest_indices, nearest_sep):
    """ The greedy one-to-one resolution loop that the vectorised np.unique version replaced.

    Arguments:
        nearest_indices: [ndarray] Index of the closest catalog star for every detection.
        nearest_sep: [ndarray] Angular separation to that star.

    Return:
        keep_mask: [ndarray] Boolean mask of the detections that claim a catalog star first.
    """

    # Walk the detections from the closest to the furthest, the first one to claim a catalog star keeps it
    keep_mask = np.zeros(len(nearest_indices), dtype=bool)
    seen_catalog = set()
    for det_i in np.argsort(nearest_sep):
        cat_i = int(nearest_indices[det_i])
        if cat_i in seen_catalog:
            continue
        seen_catalog.add(cat_i)
        keep_mask[det_i] = True

    return keep_mask


def _oneToOneVectorised(nearest_indices, nearest_sep):
    """ The vectorised one-to-one rule used in Platepar.fitAstrometry after the RANSAC loop.

    Arguments:
        nearest_indices: [ndarray] Index of the closest catalog star for every detection.
        nearest_sep: [ndarray] Angular separation to that star.

    Return:
        keep_mask: [ndarray] Boolean mask of the detections that claim a catalog star first.
    """

    # np.unique returns the first occurrence in the separation-sorted order, which is the same rule
    sep_order = np.argsort(nearest_sep)
    _, first_claim = np.unique(nearest_indices[sep_order], return_index=True)
    keep_mask = np.zeros(len(nearest_indices), dtype=bool)
    keep_mask[sep_order[first_claim]] = True

    return keep_mask


### Tests ###

@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def testNearestCatalogStarsMatchesDenseMatrix(seed):
    """ KD-tree nearest neighbour gives the same indices and separations as the dense matrix. """

    # Random catalog on the sky
    rng = np.random.RandomState(seed)
    ra_cat, dec_cat = _randomSky(rng, 1500)

    # Detections: perturbed catalog stars plus some random positions
    pick = rng.choice(len(ra_cat), 400, replace=False)
    ra_det = np.concatenate([ra_cat[pick] + rng.normal(0, 0.02, 400), _randomSky(rng, 60)[0]])
    dec_det = np.concatenate([dec_cat[pick] + rng.normal(0, 0.02, 400), _randomSky(rng, 60)[1]])

    # The KD-tree must reproduce the dense matrix exactly
    idx_ref, sep_ref = _denseNearestReference(ra_det, dec_det, ra_cat, dec_cat)
    idx_new, sep_new = _nearestCatalogStars(ra_det, dec_det, ra_cat, dec_cat)

    assert np.array_equal(idx_ref, idx_new)
    np.testing.assert_allclose(sep_new, sep_ref, rtol=0, atol=1e-12)

    # Same result with a prebuilt tree
    tree = cKDTree(_raDecToUnitVectors(ra_cat, dec_cat))
    idx_tree, sep_tree = _nearestCatalogStars(ra_det, dec_det, ra_cat, dec_cat, cat_tree=tree)
    assert np.array_equal(idx_tree, idx_new)
    assert np.array_equal(sep_tree, sep_new)


def testNearestCatalogStarsHandlesNonFiniteDetections():
    """ Non-finite detections get index 0 and a NaN separation, like the dense matrix row did. """

    # Put a NaN right in the middle of the detection list
    rng = np.random.RandomState(5)
    ra_cat, dec_cat = _randomSky(rng, 50)
    ra_det = np.array([ra_cat[7], np.nan, ra_cat[3]])
    dec_det = np.array([dec_cat[7], 10.0, dec_cat[3]])

    # The non-finite row must not disturb the finite ones
    idx_ref, sep_ref = _denseNearestReference(ra_det, dec_det, ra_cat, dec_cat)
    idx_new, sep_new = _nearestCatalogStars(ra_det, dec_det, ra_cat, dec_cat)

    assert np.array_equal(idx_ref, idx_new)
    assert np.isnan(sep_new[1]) and np.isnan(sep_ref[1])
    np.testing.assert_allclose(sep_new[[0, 2]], sep_ref[[0, 2]], rtol=0, atol=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def testOneToOneMatchingRuleIsUnchanged(seed):
    """ The vectorised one-to-one resolution keeps exactly the same detections as the greedy loop,
        including when many detections claim the same catalog star. """

    rng = np.random.RandomState(seed)
    n_det = 600

    # Few catalog indices so there are many conflicts
    nearest_indices = rng.randint(0, 150, n_det)
    nearest_sep = rng.uniform(0, 1e-3, n_det)

    # The greedy loop and the vectorised rule must agree, and no catalog star may be claimed twice
    keep_ref = _oneToOneReference(nearest_indices, nearest_sep)
    keep_new = _oneToOneVectorised(nearest_indices, nearest_sep)

    assert np.array_equal(keep_ref, keep_new)
    assert len(np.unique(nearest_indices[keep_new])) == np.sum(keep_new)


@pytest.mark.parametrize("seed", [0, 1, 2])
def testSlidingMidpointTreeGivesExactNearestDistances(seed):
    """ The cheaper cKDTree build used in fitPointingNN (balanced_tree=False, compact_nodes=False) returns
        the same nearest-neighbour distances as the default build and as brute force. """

    # Random image-plane point clouds
    rng = np.random.RandomState(seed)
    cat_coords = rng.uniform(0, 1920, (3000, 2))
    img_coords = rng.uniform(0, 1920, (2000, 2))

    # Query the same points with the default tree, the cheap tree and brute force
    dist_default, _ = cKDTree(cat_coords).query(img_coords, k=1)
    dist_fast, _ = cKDTree(cat_coords, balanced_tree=False, compact_nodes=False).query(img_coords, k=1)
    dist_brute = np.sqrt(np.min(
        (img_coords[:, 0:1] - cat_coords[:, 0][np.newaxis, :])**2
        + (img_coords[:, 1:2] - cat_coords[:, 1][np.newaxis, :])**2, axis=1))

    assert np.array_equal(dist_default, dist_fast)
    np.testing.assert_allclose(dist_fast, dist_brute, rtol=1e-12, atol=0)


def testFitPointingNNRecoversSyntheticPointing():
    """ End-to-end: fitPointingNN on the synthetic benchmark field recovers a 0.5 deg pointing offset. """

    # Imported lazily so that collecting this module does not pull in the benchmark script
    import copy
    from Tests.BenchmarkAstrometryFit import (JD_OBS, buildSyntheticCatalog, buildSyntheticDetections,
                                              buildSyntheticPlatepar)

    # Build the synthetic field
    rng = np.random.RandomState(12345)
    pp = buildSyntheticPlatepar()
    catalog_stars = buildSyntheticCatalog(pp, rng, n_target=1500)
    img_stars, _ = buildSyntheticDetections(pp, catalog_stars, rng)

    # Offset the pointing by about 0.5 deg and let the NN fit pull it back
    pp_pert = copy.deepcopy(pp)
    pp_pert.RA_d = (pp_pert.RA_d + 0.4/np.cos(np.radians(pp.dec_d))) % 360
    pp_pert.dec_d = pp_pert.dec_d + 0.3

    success, rmsd, inlier_fraction, inlier_rmsd = pp_pert.fitPointingNN(JD_OBS, img_stars, catalog_stars)

    assert success
    assert inlier_fraction > 0.8
    assert abs((pp_pert.RA_d - pp.RA_d + 180) % 360 - 180)*np.cos(np.radians(pp.dec_d)) < 0.02
    assert abs(pp_pert.dec_d - pp.dec_d) < 0.02


def _blendedIndicesDenseReference(paired_stars, catalog_stars, platepar, jd, lim_mag, fwhm_mult, mag_margin):
    """ The dense (n_matched x n_catalog) distance-matrix blend test that filterBlendedStars replaced.

    Arguments:
        paired_stars: [PairedStars] Image/catalog pairs to test for blends.
        catalog_stars: [ndarray] (M, 3) array of (ra, dec, mag).
        platepar: [Platepar] Platepar used to project the catalog onto the image.
        jd: [float] Julian date of the observation.
        lim_mag: [float] Limiting magnitude of the catalog.
        fwhm_mult: [float] Blend radius of a star, as a multiple of its FWHM.
        mag_margin: [float] Catalog stars down to lim_mag + mag_margin are considered as blenders.

    Return:
        removed: [set] Indices into paired_stars.paired_stars that would be removed as blended.
    """

    # Imported lazily to keep the module import cheap
    from RMS.Astrometry.ApplyAstrometry import raDecToXYPP
    from RMS.Astrometry.StarFilters import catalogInFOVMask

    # Project the catalog stars that are bright enough to blend and that fall inside the FOV
    bright_mask = catalog_stars[:, 2] < (lim_mag + mag_margin)
    catalog_ra = catalog_stars[bright_mask, 0]
    catalog_dec = catalog_stars[bright_mask, 1]
    in_fov = catalogInFOVMask(catalog_ra, catalog_dec, platepar)
    catalog_x, catalog_y = raDecToXYPP(catalog_ra[in_fov], catalog_dec[in_fov], jd, platepar)

    # Collect the position and the blend radius of every matched star
    check_indices, ra_list, dec_list, radii = [], [], [], []
    for i, (x, y, fwhm, intens_acc, obj, snr, saturated) in enumerate(paired_stars.paired_stars):
        ra, dec, mag = obj.coords()
        check_indices.append(i)
        ra_list.append(ra)
        dec_list.append(dec)
        radii.append(fwhm_mult*fwhm)

    # Flag a star when any catalog star other than itself falls inside its blend radius
    mx, my = raDecToXYPP(np.array(ra_list), np.array(dec_list), jd, platepar)
    radii = np.array(radii)
    dist_matrix = np.sqrt((mx[:, np.newaxis] - catalog_x[np.newaxis, :])**2
                          + (my[:, np.newaxis] - catalog_y[np.newaxis, :])**2)
    has_neighbor = np.any((dist_matrix < radii[:, np.newaxis]) & (dist_matrix > 0.1), axis=1)

    return set(idx for k, idx in enumerate(check_indices) if has_neighbor[k])


@pytest.mark.parametrize("seed", [0, 1])
def testFilterBlendedStarsMatchesDenseReference(seed):
    """ The KD-tree blend filter removes exactly the stars the dense distance matrix flagged, including
        with zero, negative and NaN FWHM values (which can never flag a neighbour). """

    # Imported lazily so that collecting this module does not pull in the benchmark script
    from RMS.Astrometry.StarFilters import filterBlendedStars
    from Tests.BenchmarkAstrometryFit import (JD_OBS, buildPairedStars, buildSyntheticCatalog,
                                              buildSyntheticDetections, buildSyntheticPlatepar)

    # Build the synthetic field and pair the true detections
    rng = np.random.RandomState(seed)
    pp = buildSyntheticPlatepar()
    catalog_stars = buildSyntheticCatalog(pp, rng, n_target=2500)
    img_stars, truth = buildSyntheticDetections(pp, catalog_stars, rng)
    paired_stars = buildPairedStars(pp, catalog_stars, img_stars, truth, rng)

    # Inject degenerate FWHM values and a few very large ones
    paired_stars.paired_stars[0][2] = 0.0
    paired_stars.paired_stars[1][2] = -1.0
    paired_stars.paired_stars[2][2] = np.nan
    paired_stars.paired_stars[3][2] = 40.0
    paired_stars.paired_stars[4][2] = 40.0

    # The KD-tree filter must remove exactly the stars the dense matrix flagged
    lim_mag, fwhm_mult, mag_margin = 9.0, 2.0, 0.3
    removed_ref = _blendedIndicesDenseReference(paired_stars, catalog_stars, pp, JD_OBS, lim_mag, fwhm_mult,
                                                mag_margin)

    filtered, n_removed = filterBlendedStars(paired_stars, catalog_stars, pp, JD_OBS, lim_mag,
                                             fwhm_mult=fwhm_mult, mag_margin=mag_margin)

    # The surviving pairs must be the same ones, in the same order
    assert n_removed == len(removed_ref) > 0
    kept_ref = [p for i, p in enumerate(paired_stars.paired_stars) if i not in removed_ref]
    assert len(filtered.paired_stars) == len(kept_ref)
    for p_new, p_ref in zip(filtered.paired_stars, kept_ref):
        assert p_new[0] == p_ref[0] and p_new[1] == p_ref[1]
    assert 0 not in removed_ref and 1 not in removed_ref and 2 not in removed_ref


def _associateReference(star_x, star_y, x_data, y_data, radius):
    """ The per-star argmin association loop that AutoPlatepar.associateDetections replaced.

    Arguments:
        star_x: [ndarray] X image coordinates of the catalog stars.
        star_y: [ndarray] Y image coordinates of the catalog stars.
        x_data: [ndarray] X image coordinates of the detections.
        y_data: [ndarray] Y image coordinates of the detections.
        radius: [float] Maximum association distance (px).

    Return:
        out: [ndarray] Index of the associated detection for every star, -1 when there is none.
    """

    # Take the closest detection for every star, but only when it is inside the radius
    out = []
    for sx, sy in zip(star_x, star_y):
        closest = -1
        if len(x_data) > 0:
            distances = np.sqrt((x_data - sx)**2 + (y_data - sy)**2)
            closest_idx = np.argmin(distances)
            if distances[closest_idx] < radius:
                closest = closest_idx
        out.append(closest)

    return np.array(out, dtype=int)


@pytest.mark.parametrize("seed", [0, 1, 2])
def testAssociateDetectionsMatchesArgminLoop(seed):
    """ The KD-tree association picks the same detection (or none) as the per-star argmin loop. """

    # Imported lazily so that collecting this module does not pull in the astrometry.net dependencies
    from RMS.Astrometry.AutoPlatepar import NN_PAIR_ASSOC_RADIUS_PX, associateDetections

    # Random detections over the image
    rng = np.random.RandomState(seed)
    x_data = rng.uniform(0, 1920, 800)
    y_data = rng.uniform(0, 1080, 800)

    # Stars: exact detections, detections offset by less/more than the radius, and far-off positions
    pick = rng.choice(800, 300, replace=False)
    offsets = rng.uniform(0, 2*NN_PAIR_ASSOC_RADIUS_PX, 300)
    angles = rng.uniform(0, 2*np.pi, 300)
    star_x = np.concatenate([x_data[pick[:100]], x_data[pick[100:]] + offsets[100:]*np.cos(angles[100:]),
                             rng.uniform(-50, 2000, 50)])
    star_y = np.concatenate([y_data[pick[:100]], y_data[pick[100:]] + offsets[100:]*np.sin(angles[100:]),
                             rng.uniform(-50, 1200, 50)])

    # Both implementations must associate the same detections, with some stars left unassociated
    ref = _associateReference(star_x, star_y, x_data, y_data, NN_PAIR_ASSOC_RADIUS_PX)
    new = associateDetections(star_x, star_y, x_data, y_data, radius=NN_PAIR_ASSOC_RADIUS_PX)

    assert np.array_equal(ref, new)
    assert np.sum(new >= 0) > 100 and np.sum(new < 0) > 0

    # Degenerate inputs behave like the loop: no detections, or a NaN coordinate -> nothing associated
    assert np.array_equal(associateDetections(star_x, star_y, np.zeros(0), np.zeros(0)),
                          _associateReference(star_x, star_y, np.zeros(0), np.zeros(0), 3.0))
    x_nan = x_data.copy()
    x_nan[5] = np.nan
    assert np.array_equal(associateDetections(star_x, star_y, x_nan, y_data),
                          _associateReference(star_x, star_y, x_nan, y_data, 3.0))


def _duplicateKeepReference(x_arr, y_arr, intens_arr, radius):
    """ The inline duplicate-removal loop from ExtractStars.fitPSF before it was moved to a helper.

    Arguments:
        x_arr: [list or ndarray] X image coordinates of the detections.
        y_arr: [list or ndarray] Y image coordinates of the detections.
        intens_arr: [list or ndarray] Intensity of every detection.
        radius: [float] Detections closer than this (px) are treated as duplicates.

    Return:
        keep: [ndarray] Boolean mask of the detections that survive the duplicate removal.
    """

    # Find every pair of detections closer than the radius
    x_arr_f = np.array(x_arr)
    y_arr_f = np.array(y_arr)
    intens_arr_f = np.array(intens_arr)
    keep = np.ones(len(x_arr), dtype=bool)
    tree = cKDTree(np.column_stack([x_arr_f, y_arr_f]))
    pairs = tree.query_pairs(radius, output_type='ndarray')

    # Of each surviving pair, drop the fainter detection
    for i, j in pairs:
        if not keep[i] or not keep[j]:
            continue
        if intens_arr_f[j] > intens_arr_f[i]:
            keep[i] = False
        else:
            keep[j] = False

    return keep


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def testDuplicateDetectionKeepMaskMatchesInlineLoop(seed):
    """ The duplicate-removal helper keeps exactly the detections the original inline loop kept, on
        clustered data with chains of overlapping pairs and equal intensities. """

    # Imported lazily so that collecting this module does not pull in the detection dependencies
    from RMS.ExtractStars import duplicateDetectionKeepMask

    rng = np.random.RandomState(seed)

    # Clusters of detections a few px apart so that pairs form chains
    centres = rng.uniform(0, 500, (60, 2))
    members = np.repeat(centres, 5, axis=0) + rng.normal(0, 2.5, (300, 2))
    intens = np.round(rng.uniform(100, 110, 300))  # coarse values -> many exact ties
    order = rng.permutation(300)
    x_arr, y_arr, intens = members[order, 0], members[order, 1], intens[order]

    # The helper must keep exactly the same detections as the original inline loop
    keep_ref = _duplicateKeepReference(x_arr, y_arr, intens, 4)
    keep_new = duplicateDetectionKeepMask(list(x_arr), list(y_arr), list(intens), 4)

    assert np.array_equal(keep_ref, keep_new)
    assert 0 < np.sum(~keep_new) < 300
