"""
Tests of the polishing of the fits of the pointing by least squares, where the cost is smooth.

The derivatives of the least squares fits step the offsets of the pointing (Platepar.PointingOffsets) by a fixed
fraction of a pixel (Platepar.offsetsJacobian). The fit of the pointing on star pairs (Platepar.fitPointing) is
polished by least squares after SLSQP, the NN fit (Platepar.fitPointingNN) on its inliers and their nearest catalog
stars, and the RANSAC fit (Platepar.fitAstrometry with use_nn_cost) gets the pairs it left out within 3 times their
RMS distance (Platepar.nearestPairs). The least squares refinement of a recalibration
(ApplyRecalibrate.refinePointingLSQ) starts from the stars within a wider radius, and also runs on the stars of
several images, with which the calibration check (CheckFit.autoCheckFit) polishes its result, also a platepar which
the coarse alignment brought onto the stars and which is good enough at once.
"""

import copy

import numpy as np

from RMS.Astrometry import ApplyRecalibrate, CheckFit
from RMS.Astrometry.ApplyAstrometry import raDecToXYPP
from RMS.Formats import Platepar as PlateparModule
from RMS.Formats.Platepar import PointingOffsets, nearestPairs, offsetsJacobian
from Tests import test_CheckFit
from Tests.test_CoarseAlignment import _scene
from Tests.test_RecalibrateLSQRefine import displacement, makeConfig, makeImage, makePlatepar


def _turned(pp, shift_px, roll_deg=0.0):
    """ A copy of the platepar with the pointing shifted along x and rolled. """

    pp = copy.deepcopy(pp)
    pointing = PointingOffsets(pp)
    pointing.apply(pp, [shift_px, 0.0, np.radians(roll_deg)/pointing.roll_unit])
    pp.updateRefAltAz()

    return pp


def test_jacobian_steps_by_a_fixed_fraction_of_a_pixel():

    # The Jacobian of a linear function is exact, and the offsets are stepped by the fixed step around any value,
    #   also near zero where a step relative to the value would vanish
    matrix = np.array([[1.0, 2.0, -1.0], [0.5, 0.0, 3.0]])
    steps = []

    def _residuals(offsets):
        """ A linear function, recording the offsets it is called with. """

        steps.append(np.array(offsets))
        return matrix.dot(offsets)

    jacobian = offsetsJacobian(_residuals)(np.array([1e-9, 0.0, 250.0]))

    assert np.allclose(jacobian, matrix, atol=1e-9)
    deltas = np.abs(np.array(steps) - np.array([1e-9, 0.0, 250.0]))
    assert np.allclose(np.max(deltas, axis=1), PlateparModule.OFFSET_JACOBIAN_STEP_PX)


def test_pointing_fit_near_the_pole_reaches_the_least_squares_optimum():

    # Noise-free star pairs 0.1 deg from the pole with the refraction: the fit of the pointing from 3 px and 0.5
    #   deg off ends on the true pointing (SLSQP with its derivatives alone stops about 1e-5 px off)
    pp, img_x, img_y, catalog_stars = _scene(89.9, n_detected=60, n_hot=0)
    x, y = raDecToXYPP(catalog_stars[:, 0], catalog_stars[:, 1], pp.JD, pp)
    inside = (x > 0) & (x < pp.X_res) & (y > 0) & (y < pp.Y_res)
    img_stars = np.column_stack([x[inside], y[inside], np.ones(np.count_nonzero(inside))])[:60]
    pairs = catalog_stars[inside][:60]

    fitted = _turned(pp, 3.0, 0.5)
    fitted.fitPointing(pp.JD, img_stars, pairs)

    assert displacement(pp, fitted, pairs) < 1e-6


def test_nn_fit_is_polished_on_its_inliers():

    # The NN fit of an image with 10 hot pixels from 5 px and 1 deg of roll off ends within the noise of the stars
    #   from the true pointing (the NN cost alone is pulled by the hot pixels by a part of a pixel)
    pp, img_x, img_y, catalog_stars = _scene(40.0, n_detected=60, n_hot=10)
    img_stars = np.column_stack([img_x, img_y, np.ones(len(img_x))])

    fitted = _turned(pp, 5.0, 1.0)
    success, _, inlier_fraction, inlier_rmsd = fitted.fitPointingNN(pp.JD, img_stars, catalog_stars)

    assert success
    assert inlier_fraction >= 60/70
    assert inlier_rmsd < 0.4
    assert displacement(pp, fitted, catalog_stars[:300]) < 0.15


def test_nearest_pairs_are_one_to_one():

    # Every catalog star is paired once, with its closest image star, and only within the distance
    pp = makePlatepar()
    catalog_stars, rows = makeImage(pp, n_stars=30)
    img_stars = np.column_stack([rows[:, 1], rows[:, 0], rows[:, 2]])

    # A second detection 0.5 px from the first star, a star without a catalog star and one without a position
    img_more = np.vstack([img_stars, img_stars[0] + [0.5, 0.0, 0.0], [600.0, 2.0, 1.0], [np.nan, 5.0, 1.0]])
    img_pairs, cat_pairs = nearestPairs(pp, pp.JD, img_more, catalog_stars, 1.0)

    assert len(img_pairs) == 30
    assert len({tuple(star) for star in cat_pairs}) == 30
    assert np.all(np.isfinite(img_pairs))
    assert np.any(np.all(img_pairs == img_stars[0], axis=1))

    # A tight distance keeps fewer pairs, and no image stars give none
    assert len(nearestPairs(pp, pp.JD, img_more, catalog_stars, 0.01)[0]) < 30
    assert len(nearestPairs(pp, pp.JD, img_more[:0], catalog_stars, 1.0)[0]) == 0


def test_ransac_pairs_are_completed():

    # The RANSAC fit of a platepar off by 0.3 deg, with 5 hot pixels, ends with all 60 stars of the image in its
    #   pairs and none of the hot pixels
    pp = makePlatepar()
    catalog_stars, rows = makeImage(pp, n_stars=60)
    img_stars = np.column_stack([rows[:, 1], rows[:, 0], rows[:, 2]])
    rng = np.random.default_rng(3)
    hot = np.column_stack([rng.uniform(0, pp.X_res, 5), rng.uniform(0, pp.Y_res, 5), np.ones(5)])

    fitted = _turned(pp, 0.3*pp.F_scale)
    img_pairs, _ = fitted.fitAstrometry(pp.JD, np.vstack([img_stars, hot]), catalog_stars, use_nn_cost=True)

    assert len(img_pairs) == 60
    assert not any(np.any(np.all(img_pairs[:, :2] == star[:2], axis=1)) for star in hot)
    assert displacement(pp, fitted, catalog_stars) < 0.1


def test_lsq_refinement_ends_on_the_same_pointing_from_starts_across_the_radius():

    # Stars matched within 0.5 px with a noise of 0.15 px: the refinements from the true pointing and from starts
    #   0.3 px off in different directions, which leave different stars near the edge of the radius, agree
    pp = makePlatepar()
    catalog_stars, rows = makeImage(pp, seed=5)
    rows = rows.copy()
    rows[:, :2] += np.random.default_rng(6).normal(0, 0.15, (len(rows), 2))
    config = makeConfig()
    star_dict = {pp.JD: rows}

    refined = []
    for shift_x, shift_y in [(0.0, 0.0), (0.3, 0.0), (0.0, -0.3), (-0.2, 0.2)]:
        start = copy.deepcopy(pp)
        pointing = PointingOffsets(start)
        pointing.apply(start, [shift_x, shift_y, 0.0])
        n_check, _, _, _ = CheckFit.matchStarsResiduals(config, start, catalog_stars, star_dict, 0.5,
                                                        ret_nmatch=True)
        result, _ = ApplyRecalibrate.refinePointingLSQ(config, start, pp.JD, star_dict, catalog_stars, 0.5, 0.5,
                                                       n_check)
        assert result is not None
        refined.append(result)

    assert max(displacement(refined[0], result, catalog_stars) for result in refined[1:]) < 0.005


def test_check_polishes_a_coarsely_aligned_platepar(monkeypatch):

    # A platepar 3 deg off and with the scale off by 0.05% (0.3 px in the corners), which the coarse alignment
    #   brings within the tolerance of the check, is polished by least squares instead of being handed over with
    #   the scale the alignment does not fit
    pp = test_CheckFit.makePlatepar()
    config = test_CheckFit.makeConfig()
    catalog_stars, calstars_data = test_CheckFit.makeNight(pp, config)
    test_CheckFit.useCatalog(monkeypatch, catalog_stars)

    start = _turned(pp, 3.0*pp.F_scale, 2.0)
    start.F_scale *= 1.0005
    result, fit_status = CheckFit.autoCheckFit(config, copy.deepcopy(start), calstars_data)

    assert fit_status
    assert displacement(pp, result, catalog_stars) < 0.01


def test_check_is_polished(monkeypatch):

    # The check of a platepar 2 px off ends within the noise of the stars from the true platepar (the search alone
    #   ends within its tolerance, a third of dist_check_threshold)
    pp = test_CheckFit.makePlatepar()
    config = test_CheckFit.makeConfig()
    catalog_stars, calstars_data = test_CheckFit.makeNight(pp, config)
    test_CheckFit.useCatalog(monkeypatch, catalog_stars)

    result, fit_status = CheckFit.autoCheckFit(config, test_CheckFit.offsetPlatepar(pp), calstars_data)

    assert fit_status
    assert displacement(pp, result, catalog_stars) < 0.01


def test_lsq_refinement_on_several_images():

    # The least squares refinement on the stars of several images (jd None, as the check uses it) finds the
    #   pointing of a platepar 2 px off
    pp = test_CheckFit.makePlatepar()
    config = test_CheckFit.makeConfig()
    catalog_stars, calstars_data = test_CheckFit.makeNight(pp, config)
    star_dict = CheckFit.starListToDict(config, calstars_data)
    assert len(star_dict) == test_CheckFit.N_FF

    start = test_CheckFit.offsetPlatepar(pp)
    n_check, _, _, _ = CheckFit.matchStarsResiduals(config, start, catalog_stars, star_dict, 3.0, ret_nmatch=True)
    refined, _ = ApplyRecalibrate.refinePointingLSQ(config, start, None, star_dict, catalog_stars, 3.0, 3.0,
                                                    n_check)

    assert refined is not None
    assert displacement(pp, refined, catalog_stars) < 0.01
