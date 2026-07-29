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
