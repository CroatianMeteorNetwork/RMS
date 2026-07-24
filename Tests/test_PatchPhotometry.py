""" Tests for the forced patch photometry primitive, including the two field
failure modes the floors exist for. """

from __future__ import absolute_import, division, print_function

import numpy as np

from RMS.PatchPhotometry import (detectStars, measurePatchFluxes,
    nightNoiseFloor)


def _frameWithStars(xy_amp, noise=3.0, bg=30.0, shape=(720, 1280), seed=1):
    """ Synthetic frame: gaussian background noise + gaussian stars. """

    rng = np.random.RandomState(seed)
    img = bg + noise*rng.standard_normal(shape)

    yy, xx = np.mgrid[-3:4, -3:4]
    psf = np.exp(-(xx**2 + yy**2)/(2*1.2**2))
    for (x, y, amp) in xy_amp:
        img[y - 3:y + 4, x - 3:x + 4] += amp*psf

    return img


def test_bright_star_measured_and_detected():
    stars = [(100, 100, 120.0), (500, 300, 60.0)]
    img = _frameWithStars(stars)
    x = np.array([100, 500], dtype=float)
    y = np.array([100, 300], dtype=float)

    flux, fn = measurePatchFluxes(img, x, y)
    assert np.all(np.isfinite(flux))
    assert flux[0] > flux[1] > 0

    det, snr = detectStars(flux, fn, noise_floor=fn)
    assert det.all()
    assert np.all(snr > 5)


def test_empty_positions_not_detected():
    img = _frameWithStars([])
    rng = np.random.RandomState(3)
    x = rng.uniform(20, 1200, 200)
    y = rng.uniform(20, 700, 200)

    flux, fn = measurePatchFluxes(img, x, y)
    det, _ = detectStars(flux, fn, noise_floor=fn)

    # false-positive rate at 5 sigma on pure noise: essentially zero
    assert det.mean() < 0.02


def test_edge_positions_return_nan():
    img = _frameWithStars([])
    flux, _ = measurePatchFluxes(img, np.array([2.0, 640.0]), np.array([2.0, 360.0]))
    assert np.isnan(flux[0])
    assert np.isfinite(flux[1])


def test_collapsed_noise_floor_cannot_manufacture_detections():
    # The overcast failure mode: a perfectly smooth frame with one hot pixel.
    # Its own noise MAD is ~0, so without the floor the hot pixel gets an
    # astronomical SNR; with the nightly floor it must clear a real threshold.
    img = np.full((720, 1280), 40.0)
    img[400, 600] = 44.0     # 4 ADU bump - far below any real star

    flux, fn = measurePatchFluxes(img, np.array([600.0]), np.array([400.0]))
    assert fn < 0.5          # the frame's own floor did collapse

    night_floor = 12.0       # a realistic clear-night noise scale
    det, snr = detectStars(flux, fn, noise_floor=night_floor)
    assert not det.any()

    # And the same measurement WITHOUT the floor would have fired - the test
    # documents that the floor is what prevents it
    det_unfloored, _ = detectStars(flux, fn, noise_floor=0.0)
    assert det_unfloored.any()


def test_clear_flux_fraction_floor():
    # A 5-sigma blip at a tiny fraction of the star's normal flux is not the
    # star (JPEG bump under thin cloud): the fraction floor must reject it,
    # and pass a detection at normal brightness.
    stars = [(200, 200, 8.0)]         # weak blip
    img = _frameWithStars(stars, noise=1.0)
    flux, fn = measurePatchFluxes(img, np.array([200.0]), np.array([200.0]))

    clear_median = np.array([500.0])  # the star normally reads 500 ADU
    det, _ = detectStars(flux, fn, noise_floor=fn, clear_flux_median=clear_median)
    assert not det.any()

    det, _ = detectStars(flux, fn, noise_floor=fn,
        clear_flux_median=np.array([np.nan]))   # uncalibrated: SNR only
    assert det.any()


def test_night_noise_floor_ignores_collapsed_frames():
    assert nightNoiseFloor([12.0, 11.0, 0.0, 0.0, 13.0]) == 12.0
    assert nightNoiseFloor([]) == 1.0
