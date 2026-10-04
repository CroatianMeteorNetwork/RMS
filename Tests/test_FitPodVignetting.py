""" Tests for the pod vignetting fit on cross-camera star pairs (Utils.FitPodVignetting).

The fit must recover a known shared coefficient and the per-pair zero points from synthetic pairs
with noise and outliers, the pairing must join the same star seen by two cameras at the same time,
and the photometric offset compensation must keep calibrated magnitudes unchanged when the
coefficient changes.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from Utils.FitPodVignetting import (fitPodVignetting, magLevShift, pairStations, vignettingLoss,
                                    _profileOffsets)

X_RES, Y_RES = 1920, 1080
R_CORNER = np.hypot(X_RES/2.0, Y_RES/2.0)


def _syntheticPairs(k_true, zps, n=4000, noise=0.15, outlier_frac=0.05, seed=1):
    """ Pairs for three cameras with a shared cos^4 profile, Gaussian noise and gross outliers. """

    rng = np.random.RandomState(seed)
    pairs = {}

    for key, zp in zps.items():

        r_a = rng.uniform(300, R_CORNER, n)
        r_b = rng.uniform(300, R_CORNER, n)
        dmag = vignettingLoss(r_a, k_true) - vignettingLoss(r_b, k_true) + zp + rng.normal(0, noise, n)

        n_out = int(outlier_frac*n)
        dmag[:n_out] += rng.choice([-1, 1], n_out)*rng.uniform(1.0, 3.0, n_out)

        pairs[key] = {
            "r_a": r_a, "r_b": r_b, "dmag": dmag,
            "fwhm_a": rng.uniform(3.0, 4.0, n), "fwhm_b": rng.uniform(3.0, 4.0, n),
            "snr_a": rng.uniform(10, 100, n), "snr_b": rng.uniform(10, 100, n),
            "jd": np.full(n, 2461000.5),
            }

    return pairs


def testFitRecoversSharedCoefficient():

    k_true = 0.0006
    zps = {("A", "B"): 0.3, ("B", "C"): -0.5, ("A", "C"): -0.2}
    pairs = _syntheticPairs(k_true, zps)
    resolutions = {s: (X_RES, Y_RES) for s in "ABC"}

    result = fitPodVignetting(pairs, resolutions, snr_min=20.0, fwhm_term=True)

    assert result is not None
    assert abs(result["k"] - k_true)/k_true < 0.03
    assert abs(result["k_fwhm"] - k_true)/k_true < 0.03
    assert abs(result["fwhm_coeff"]) < 0.01

    # Zero points per pair are the injected gain differences, and they close around the triangle
    for key, zp in zps.items():
        assert abs(result["per_pair"]["{:s}-{:s}".format(*key)]["zp"] - zp) < 0.03
    assert result["closure_max"] < 0.05

    # Outliers were clipped, so the robust scatter is the injected noise, not inflated by them
    assert 0.1 < result["scatter"] < 0.2

    # Every camera follows the shared profile: no camera is flagged, per-camera k agree
    assert result["flagged"] == []
    for k_cam in result["per_camera_k"].values():
        assert abs(k_cam - k_true)/k_true < 0.05


def testFlagsCameraWithDifferentProfile():

    k_true = 0.0006
    zps = {("A", "B"): 0.0, ("B", "C"): 0.0, ("A", "C"): 0.0}
    pairs = _syntheticPairs(k_true, zps, n=6000, outlier_frac=0.0)

    # Camera C is much more strongly vignetted than the others
    k_c = 0.0009
    for key in (("B", "C"), ("A", "C")):
        p = pairs[key]
        p["dmag"] += -(vignettingLoss(p["r_b"], k_c) - vignettingLoss(p["r_b"], k_true))

    resolutions = {s: (X_RES, Y_RES) for s in "ABC"}
    result = fitPodVignetting(pairs, resolutions, snr_min=20.0)

    assert "C" in result["flagged"]
    assert result["per_camera_k"]["C"] > result["per_camera_k"]["A"]
    assert result["per_camera_k"]["C"] > result["per_camera_k"]["B"]


def testSnrCutRemovesPairs():

    pairs = _syntheticPairs(0.0006, {("A", "B"): 0.0})
    resolutions = {s: (X_RES, Y_RES) for s in "AB"}

    loose = fitPodVignetting(pairs, resolutions, snr_min=0.0, per_camera=False)
    strict = fitPodVignetting(pairs, resolutions, snr_min=50.0, per_camera=False)

    assert strict["n_pairs_total"] < loose["n_pairs_total"]
    assert fitPodVignetting(pairs, resolutions, snr_min=1e6) is None


def testPairingJoinsSameStarAtSameTime():

    rng = np.random.RandomState(3)
    n = 50
    ra = np.radians(rng.uniform(0, 360, n))
    dec = np.radians(rng.uniform(-60, 60, n))
    vec = np.column_stack([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])

    jd = 2461000.5
    star_a = {"jd": jd, "vec": vec, "r": rng.uniform(0, 1000, n), "mag": rng.uniform(-10, -5, n),
              "fwhm": np.full(n, 3.0), "snr": np.full(n, 30.0)}

    # Camera B sees the same stars 3 s later, perturbed by 1 arcmin, 0.4 mag brighter instrumentally
    perturbed = vec + np.radians(1.0/60.0)*rng.normal(size=vec.shape)
    perturbed /= np.linalg.norm(perturbed, axis=1)[:, None]
    star_b = {"jd": jd + 3.0/86400.0, "vec": perturbed, "r": rng.uniform(0, 1000, n),
              "mag": star_a["mag"] - 0.4, "fwhm": np.full(n, 3.5), "snr": np.full(n, 25.0)}

    pairs = pairStations([star_a], [star_b])

    assert pairs is not None
    assert len(pairs["dmag"]) == n
    assert np.allclose(pairs["dmag"], 0.4)
    assert np.allclose(pairs["fwhm_a"], 3.0) and np.allclose(pairs["fwhm_b"], 3.5)

    # Too far apart in time: nothing is paired
    star_b_late = dict(star_b, jd=jd + 60.0/86400.0)
    assert pairStations([star_a], [star_b_late]) is None

    # An empty camera pairs with nothing
    assert pairStations([star_a], []) is None


def testMagLevShiftKeepsCalibratedMagnitudes():

    rng = np.random.RandomState(5)
    star_xy = np.column_stack([rng.uniform(0, X_RES, 300), rng.uniform(0, Y_RES, 300)])
    r = np.hypot(star_xy[:, 0] - X_RES/2.0, star_xy[:, 1] - Y_RES/2.0)

    k_old, k_new = 0.0008, 0.00059
    mag_lev_old = 11.0
    intens = 10**(-0.4*rng.uniform(-12, -8, 300))

    # Calibrated magnitude: mag = -2.5 log10(I) - V(r, k) + mag_lev
    mag_old = -2.5*np.log10(intens) - vignettingLoss(r, k_old) + mag_lev_old

    shift = magLevShift(star_xy, X_RES, Y_RES, k_old, k_new)
    mag_new = -2.5*np.log10(intens) - vignettingLoss(r, k_new) + mag_lev_old + shift

    # Less correction needs a smaller offset, and the sample mean is preserved exactly
    assert shift < 0
    assert abs(np.mean(mag_new - mag_old)) < 1e-9

    # Without a star sample, a uniform image-area average is used and has the same sign
    assert magLevShift(None, X_RES, Y_RES, k_old, k_new) < 0
    assert magLevShift(None, X_RES, Y_RES, k_old, k_old) == 0.0


def testProfileOffsetsUsesInlierMedian():

    resid = np.array([1.0, 1.2, 0.8, 50.0, -2.0, -2.2, -1.8])
    pid = np.array([0, 0, 0, 0, 1, 1, 1])
    mask = np.array([True, True, True, False, True, True, True])

    out, offsets = _profileOffsets(resid, pid, 2, mask)

    assert offsets[0] == pytest.approx(1.0) and offsets[1] == pytest.approx(-2.0)
    assert out[3] == pytest.approx(49.0)
    assert np.allclose(out[[0, 4]], 0.0)
