""" Tests for the noise-adaptive star-candidacy gate (RMS.ExtractStars). """

from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from RMS.ExtractStars import (adaptiveContrastThreshold, extractStars,
    NOISE_CONTRAST_FACTOR, MIN_CONTRAST_FLOOR)


def contrastField(median_value, n=10000):
    """ Synthetic contrast field with the given median. """
    rng = np.random.default_rng(1)
    return np.abs(rng.normal(median_value, median_value*0.3, n)) + 1e-6


def test_dark_site_gate_deepens():
    # Measured USC0G4 mid-night: contrast median 2.2 ADU against a configured 12
    c = contrastField(2.2)
    t = adaptiveContrastThreshold(c, 12.0)
    assert t < 12.0
    assert t == pytest.approx(max(NOISE_CONTRAST_FACTOR*np.median(c), MIN_CONTRAST_FLOOR), rel=0.01)


def test_twilight_gate_rises_with_noise():
    # Measured USC0G4 twilight: contrast median 4.2 - the working gate must be HIGHER
    # than the dark-sky one (this is the surge fix: admission tracks the noise)
    t_dark = adaptiveContrastThreshold(contrastField(2.2), 12.0)
    t_twil = adaptiveContrastThreshold(contrastField(4.2), 12.0)
    assert t_twil > t_dark


def test_configured_threshold_is_a_ceiling():
    # Noisy (light-polluted) camera: adaptive value exceeds the configured threshold,
    # so behavior must be exactly the current fixed-gate behavior
    t = adaptiveContrastThreshold(contrastField(10.0), 12.0)
    assert t == 12.0


def test_flat_image_floor():
    # A clipped/flat image must not open the gate entirely
    t = adaptiveContrastThreshold(np.full(1000, 0.1), 12.0)
    assert t == MIN_CONTRAST_FLOOR


def test_bit_depth_scales_floor_and_respects_ceiling():
    t = adaptiveContrastThreshold(np.full(1000, 0.1), 12.0*256, bit_depth=16)
    assert t == MIN_CONTRAST_FLOOR*256


def synthImage(sigma, star_amps, bg=38.0, size=400, seed=7):
    """ Synthetic avepixel: flat sky + noise + Gaussian stars on a grid. """
    rng = np.random.default_rng(seed)
    img = bg + rng.normal(0, sigma, (size, size))
    positions = []
    for i, amp in enumerate(star_amps):
        x0, y0 = 60 + 90*(i % 4), 60 + 90*(i//4)
        yy, xx = np.mgrid[0:size, 0:size]
        img += amp*np.exp(-((xx - x0)**2 + (yy - y0)**2)/(2*1.6**2))
        positions.append((x0, y0))
    return np.clip(img, 0, 255).astype(np.float32), positions


def found(status, positions, tol=3.0):
    if status is False or status is None:
        return [False]*len(positions)
    x_arr, y_arr = status[0], status[1]
    out = []
    for x0, y0 in positions:
        d = np.hypot(np.array(x_arr) - x0, np.array(y_arr) - y0)
        out.append(bool(len(d) and d.min() <= tol))
    return out


def test_faint_stars_recovered_on_quiet_image():
    # 6 ADU stars on a 0.6 ADU-noise image are ~10 sigma events: the fixed 12 ADU gate
    # misses them, the adaptive gate must find them
    img, pos = synthImage(0.6, [25.0, 6.0, 6.0, 6.0])
    fixed = found(extractStars(img, intensity_threshold=12, adaptive_threshold=False), pos)
    adapt = found(extractStars(img, intensity_threshold=12, adaptive_threshold=True), pos)
    assert fixed[0] and adapt[0]              # bright star always found
    assert sum(adapt[1:]) >= 2                # faint stars recovered
    assert sum(adapt[1:]) > sum(fixed[1:])    # strictly better than the fixed gate


def test_noisy_image_behavior_unchanged():
    # High-noise image: the adaptive gate hits the configured ceiling, so both modes
    # must produce identical candidates
    img, pos = synthImage(4.0, [60.0, 30.0])
    fixed = found(extractStars(img, intensity_threshold=12, adaptive_threshold=False), pos)
    adapt = found(extractStars(img, intensity_threshold=12, adaptive_threshold=True), pos)
    assert fixed == adapt
