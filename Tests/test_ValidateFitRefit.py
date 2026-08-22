""" Tests for the cross-frame refit pair preparation (RMS.Astrometry.ValidateFit).

The drift compensation in buildRefitGroups must transfer the FULL pointing delta between
a frame's fitted pointing and the reference pointing - centre shift and roll - so that
the multi-image fit's single pointing is consistent with every frame's pairs.
"""

from __future__ import absolute_import, division, print_function

import copy

import numpy as np
import pytest

from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, xyToRaDecPP
from RMS.Astrometry.Conversions import JD2HourAngle
from RMS.Astrometry.ValidateFit import buildRefitGroups
from RMS.Formats.Platepar import Platepar


JD_TEST = 2460799.6  # 2025-05-04 02:24 UT


def makePlatepar():
    """ A distortion-free platepar pointed at alt ~45 deg from a mid-latitude site. """

    pp = Platepar()
    pp.lat, pp.lon, pp.elev = 44.0, 16.0, 100.0
    pp.X_res, pp.Y_res = 1280, 720
    pp.F_scale = 10.0  # px/deg, ~6 arcmin/px - typical meteor camera
    pp.JD = JD_TEST
    pp.Ho = JD2HourAngle(pp.JD)

    # Point somewhere sensible: RA near the local meridian, dec above the horizon
    pp.RA_d = pp.Ho%360
    pp.dec_d = 60.0
    pp.pos_angle_ref = 30.0

    return pp


def makeResults(pp_ref, pp_true, jd):
    """ Synthesize a validateFit results dict: detections on a grid of image points whose
        catalog counterparts are projected through the TRUE (drifted) pointing. """

    gx, gy = np.meshgrid(np.linspace(40, pp_ref.X_res - 40, 8),
                         np.linspace(40, pp_ref.Y_res - 40, 5))
    gx, gy = gx.ravel(), gy.ravel()

    _, ra, dec, _ = xyToRaDecPP(np.full(len(gx), jd), gx, gy, np.ones(len(gx)), pp_true,
        extinction_correction=False, jd_time=True, precompute_pointing_corr=True)

    return dict(
        star_x=np.array(gx), star_y=np.array(gy),
        star_frame=np.zeros(len(gx), dtype=int),
        star_ra=np.array(ra), star_dec=np.array(dec),
        star_mag=np.full(len(gx), 5.0), star_intens=np.full(len(gx), 1000.0),
        star_res=np.full(len(gx), 0.2),
        frames=[dict(ff_name="FF_TEST_20250504_022400_000_0000000.fits", jd=jd,
                     frame_index=0, drift_arcmin=1.0,
                     pointing_frame=(pp_true.RA_d, pp_true.dec_d, pp_true.pos_angle_ref))],
    )


def maxResidualPx(image_groups, pp_ref):
    """ Max distance (px) between each group's image stars and its catalog stars projected
        with the reference platepar. """

    worst = 0.0
    for _, jd, img_stars, cat_stars in image_groups:
        cat_x, cat_y = raDecToXYPP(cat_stars[:, 0], cat_stars[:, 1], jd, pp_ref)
        worst = max(worst, float(np.max(np.hypot(cat_x - img_stars[:, 0],
                                                 cat_y - img_stars[:, 1]))))
    return worst


@pytest.mark.parametrize("dra_deg,ddec_deg,droll_deg", [
    (0.1, -0.08, 0.0),      # pure centre drift (several arcmin)
    (0.0, 0.0, 0.1),        # pure roll - the case a centre-only compensation misses
    (0.1, -0.08, 0.1),      # combined
])
def test_drift_compensation_removes_pointing_delta(dra_deg, ddec_deg, droll_deg):
    pp_ref = makePlatepar()

    pp_true = copy.deepcopy(pp_ref)
    pp_true.RA_d = (pp_true.RA_d + dra_deg)%360
    pp_true.dec_d += ddec_deg
    pp_true.pos_angle_ref += droll_deg

    results = makeResults(pp_ref, pp_true, JD_TEST)

    # Uncompensated pairs disagree with the reference pointing by the injected drift
    raw = buildRefitGroups(results, pp_ref, max_per_cell=100, drift_correction=False)
    assert maxResidualPx(raw, pp_ref) > 0.5

    # Compensated pairs must be consistent with the reference pointing. The only inexact
    # ingredient is refraction evaluated at the compensated rather than the original
    # position (second order in the drift), so the tolerance is tight.
    corrected = buildRefitGroups(results, pp_ref, max_per_cell=100, drift_correction=True)
    assert maxResidualPx(corrected, pp_ref) < 0.05


def test_no_compensation_without_frame_pointing():
    pp_ref = makePlatepar()
    results = makeResults(pp_ref, pp_ref, JD_TEST)
    results["frames"][0]["pointing_frame"] = None

    groups = buildRefitGroups(results, pp_ref, max_per_cell=100, drift_correction=True)

    # Catalog passes through untouched (the spatial cap shuffles pair order, so compare
    # the sorted coordinate sets)
    assert len(groups) == 1
    np.testing.assert_allclose(np.sort(groups[0][3][:, 0]), np.sort(results["star_ra"]))
    np.testing.assert_allclose(np.sort(groups[0][3][:, 1]), np.sort(results["star_dec"]))


def test_gross_positional_outliers_are_dropped():
    """ validateFit matches within a wide radius so it can measure a platepar that is off.
        A pair that wide is a wrong catalog star, and it must not reach the fit. """

    pp_ref = makePlatepar()
    results = makeResults(pp_ref, pp_ref, JD_TEST)

    n = len(results["star_x"])
    results["star_res"] = np.full(n, 0.2)
    results["star_res"][:3] = [9.5, 7.1, 6.0]      # matched to the wrong catalog star

    info = {}
    groups = buildRefitGroups(results, pp_ref, max_per_cell=100, drift_correction=False,
                              info=info)

    assert info["n_residual_outliers"] == 3
    assert sum(len(img_stars) for _, _, img_stars, _ in groups) == n - 3


def test_a_clean_night_keeps_every_pair():
    """ The cut is floored, so a night whose residuals are all sub-pixel loses nothing -
        without the floor a robust sigma on a tight set would trim its own good tail. """

    pp_ref = makePlatepar()
    results = makeResults(pp_ref, pp_ref, JD_TEST)

    n = len(results["star_x"])
    rng = np.random.default_rng(3)
    results["star_res"] = np.abs(rng.normal(0.3, 0.15, n))

    info = {}
    groups = buildRefitGroups(results, pp_ref, max_per_cell=100, drift_correction=False,
                              info=info)

    assert info["n_residual_outliers"] == 0
    assert sum(len(img_stars) for _, _, img_stars, _ in groups) == n


def test_results_without_residuals_still_build():
    """ Hand-built inputs (older callers, tests) carry no residuals - the cut is skipped
        rather than failing. """

    pp_ref = makePlatepar()
    results = makeResults(pp_ref, pp_ref, JD_TEST)
    del results["star_res"]

    info = {}
    groups = buildRefitGroups(results, pp_ref, max_per_cell=100, drift_correction=False,
                              info=info)

    assert "n_residual_outliers" not in info
    assert sum(len(img_stars) for _, _, img_stars, _ in groups) == len(results["star_x"])


def test_a_corner_wide_error_is_not_mistaken_for_outliers():
    """ The platepar being replaced is usually worst in the corners - that is the reason to
        refit. Judged against the whole frame those pairs look like a fat outlier tail, and
        dropping them removes exactly the evidence the refit needs. """

    pp_ref = makePlatepar()
    results = makeResults(pp_ref, pp_ref, JD_TEST)

    x = np.asarray(results["star_x"], dtype=float)
    y = np.asarray(results["star_y"], dtype=float)
    radius = np.hypot(x - pp_ref.X_res/2, y - pp_ref.Y_res/2) \
        /np.hypot(pp_ref.X_res/2, pp_ref.Y_res/2)

    # Sub-pixel everywhere, several pixels in the outer ring
    outer = radius > 0.7
    assert outer.sum() >= 3, "fixture must have outer-field pairs"
    results["star_res"] = np.where(outer, 4.5, 0.3)

    info = {}
    # The fixture is small, so let each annulus speak for itself
    groups = buildRefitGroups(results, pp_ref, max_per_cell=100, drift_correction=False,
                              min_per_annulus=1, info=info)

    assert info["n_residual_outliers"] == 0
    assert sum(len(img_stars) for _, _, img_stars, _ in groups) == len(x)


def test_a_wrong_star_match_in_the_outer_field_is_still_dropped():
    """ ...but a pair that disagrees with its OWN annulus is a wrong catalog star, wherever
        in the frame it sits. """

    pp_ref = makePlatepar()
    results = makeResults(pp_ref, pp_ref, JD_TEST)

    x = np.asarray(results["star_x"], dtype=float)
    y = np.asarray(results["star_y"], dtype=float)
    radius = np.hypot(x - pp_ref.X_res/2, y - pp_ref.Y_res/2) \
        /np.hypot(pp_ref.X_res/2, pp_ref.Y_res/2)

    outer = np.where(radius > 0.7)[0]
    results["star_res"] = np.where(radius > 0.7, 4.5, 0.3)
    results["star_res"][outer[0]] = 9.5      # wrong catalog star, out in the corner

    info = {}
    groups = buildRefitGroups(results, pp_ref, max_per_cell=100, drift_correction=False,
                              min_per_annulus=1, info=info)

    assert info["n_residual_outliers"] == 1
    assert sum(len(img_stars) for _, _, img_stars, _ in groups) == len(x) - 1
