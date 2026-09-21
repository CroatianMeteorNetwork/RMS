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


def _randomSky(rng, n, ra_centre=120.0, dec_centre=30.0, half_width=25.0):
    """ Random RA/Dec (deg) in a box around a centre. """

    ra = ra_centre + rng.uniform(-half_width, half_width, n)
    dec = np.clip(dec_centre + rng.uniform(-half_width, half_width, n), -89.9, 89.9)

    return ra % 360, dec


def _denseNearestReference(ra_det, dec_det, ra_cat, dec_cat):
    """ The dense N x M angularSeparation matrix with row-wise argmin/min that the KD-tree replaced. """

    ra_det_rad = np.radians(ra_det)[:, np.newaxis]
    dec_det_rad = np.radians(dec_det)[:, np.newaxis]
    ra_cat_rad = np.radians(ra_cat)[np.newaxis, :]
    dec_cat_rad = np.radians(dec_cat)[np.newaxis, :]
    sep_matrix = angularSeparation(ra_det_rad, dec_det_rad, ra_cat_rad, dec_cat_rad)

    return np.argmin(sep_matrix, axis=1), np.min(sep_matrix, axis=1)


def _oneToOneReference(nearest_indices, nearest_sep):
    """ The greedy one-to-one resolution loop that the vectorised np.unique version replaced. """

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
    """ The vectorised one-to-one rule used in Platepar.fitAstrometry after the RANSAC loop. """

    sep_order = np.argsort(nearest_sep)
    _, first_claim = np.unique(nearest_indices[sep_order], return_index=True)
    keep_mask = np.zeros(len(nearest_indices), dtype=bool)
    keep_mask[sep_order[first_claim]] = True

    return keep_mask


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def testNearestCatalogStarsMatchesDenseMatrix(seed):
    """ KD-tree nearest neighbour gives the same indices and separations as the dense matrix. """

    rng = np.random.RandomState(seed)
    ra_cat, dec_cat = _randomSky(rng, 1500)

    # Detections: perturbed catalog stars plus some random positions
    pick = rng.choice(len(ra_cat), 400, replace=False)
    ra_det = np.concatenate([ra_cat[pick] + rng.normal(0, 0.02, 400), _randomSky(rng, 60)[0]])
    dec_det = np.concatenate([dec_cat[pick] + rng.normal(0, 0.02, 400), _randomSky(rng, 60)[1]])

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

    rng = np.random.RandomState(5)
    ra_cat, dec_cat = _randomSky(rng, 50)
    ra_det = np.array([ra_cat[7], np.nan, ra_cat[3]])
    dec_det = np.array([dec_cat[7], 10.0, dec_cat[3]])

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

    keep_ref = _oneToOneReference(nearest_indices, nearest_sep)
    keep_new = _oneToOneVectorised(nearest_indices, nearest_sep)

    assert np.array_equal(keep_ref, keep_new)
    assert len(np.unique(nearest_indices[keep_new])) == np.sum(keep_new)


@pytest.mark.parametrize("seed", [0, 1, 2])
def testSlidingMidpointTreeGivesExactNearestDistances(seed):
    """ The cheaper cKDTree build used in fitPointingNN (balanced_tree=False, compact_nodes=False) returns
        the same nearest-neighbour distances as the default build and as brute force. """

    rng = np.random.RandomState(seed)
    cat_coords = rng.uniform(0, 1920, (3000, 2))
    img_coords = rng.uniform(0, 1920, (2000, 2))

    dist_default, _ = cKDTree(cat_coords).query(img_coords, k=1)
    dist_fast, _ = cKDTree(cat_coords, balanced_tree=False, compact_nodes=False).query(img_coords, k=1)
    dist_brute = np.sqrt(np.min(
        (img_coords[:, 0:1] - cat_coords[:, 0][np.newaxis, :])**2
        + (img_coords[:, 1:2] - cat_coords[:, 1][np.newaxis, :])**2, axis=1))

    assert np.array_equal(dist_default, dist_fast)
    np.testing.assert_allclose(dist_fast, dist_brute, rtol=1e-12, atol=0)


def testFitPointingNNRecoversSyntheticPointing():
    """ End-to-end: fitPointingNN on the synthetic benchmark field recovers a 0.5 deg pointing offset. """

    import copy
    from Tests.BenchmarkAstrometryFit import (JD_OBS, buildSyntheticCatalog, buildSyntheticDetections,
                                              buildSyntheticPlatepar)

    rng = np.random.RandomState(12345)
    pp = buildSyntheticPlatepar()
    catalog_stars = buildSyntheticCatalog(pp, rng, n_target=1500)
    img_stars, _ = buildSyntheticDetections(pp, catalog_stars, rng)

    pp_pert = copy.deepcopy(pp)
    pp_pert.RA_d = (pp_pert.RA_d + 0.4/np.cos(np.radians(pp.dec_d))) % 360
    pp_pert.dec_d = pp_pert.dec_d + 0.3

    success, rmsd, inlier_fraction, inlier_rmsd = pp_pert.fitPointingNN(JD_OBS, img_stars, catalog_stars)

    assert success
    assert inlier_fraction > 0.8
    assert abs((pp_pert.RA_d - pp.RA_d + 180) % 360 - 180)*np.cos(np.radians(pp.dec_d)) < 0.02
    assert abs(pp_pert.dec_d - pp.dec_d) < 0.02
