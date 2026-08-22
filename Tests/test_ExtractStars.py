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
    # Measured USC0G4 mid-night: contrast median 2.2 ADU (old fixed gate was 12)
    c = contrastField(2.2)
    t = adaptiveContrastThreshold(c)
    assert t < 12.0
    assert t == pytest.approx(max(NOISE_CONTRAST_FACTOR*np.median(c), MIN_CONTRAST_FLOOR), rel=0.01)


def test_twilight_gate_rises_with_noise():
    # Measured USC0G4 twilight: contrast median 4.2 - the working gate must be HIGHER
    # than the dark-sky one AND free to exceed the old fixed value (no ceiling: capping
    # at the configured value reintroduced the twilight surge on station validation)
    t_dark = adaptiveContrastThreshold(contrastField(2.2))
    t_twil = adaptiveContrastThreshold(contrastField(4.2))
    assert t_twil > t_dark
    assert t_twil > 12.0


def test_noisy_camera_gate_scales_with_noise():
    # Noisy (light-polluted) camera: the gate tracks the noise with no ceiling
    c = contrastField(10.0)
    t = adaptiveContrastThreshold(c)
    assert t == pytest.approx(NOISE_CONTRAST_FACTOR*np.median(c), rel=0.01)


def test_flat_image_floor():
    # A clipped/flat image must not open the gate entirely
    t = adaptiveContrastThreshold(np.full(1000, 0.1))
    assert t == MIN_CONTRAST_FLOOR


def test_bit_depth_scales_floor():
    t = adaptiveContrastThreshold(np.full(1000, 0.1), bit_depth=16)
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
    # 6 ADU stars on a 0.6 ADU-noise image are ~10 sigma events the old fixed 12 ADU
    # gate missed: the adaptive gate must find them
    img, pos = synthImage(0.6, [25.0, 6.0, 6.0, 6.0])
    adapt = found(extractStars(img), pos)
    assert adapt[0]                           # bright star always found
    assert sum(adapt[1:]) >= 2                # faint stars recovered


def test_noisy_image_bright_stars_survive_raised_gate():
    # High-noise image: the adaptive gate rises above the old fixed value; bright
    # stars must still be found and the gate must not be noise-flooded
    img, pos = synthImage(4.0, [60.0, 30.0])
    adapt = found(extractStars(img), pos)
    assert all(adapt)


def test_extractStarsFF_end_to_end(tmp_path):
    # Regression guard: extractStarsFF must stay callable end-to-end (a merge once left
    # it passing a keyword extractStars had dropped, crashing every nightly extraction)
    from RMS.Formats import FFStruct, FFfits
    from RMS.ExtractStars import extractStarsFF

    rng = np.random.default_rng(2)
    h, w = 128, 128
    ave = (38 + rng.normal(0, 0.6, (h, w))).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for x0, y0, amp in [(40, 40, 30.0), (90, 70, 12.0)]:
        ave += amp*np.exp(-((xx - x0)**2 + (yy - y0)**2)/(2*1.6**2))
    ave = np.clip(ave, 0, 255)

    ff = FFStruct.FFStruct()
    ff.nrows, ff.ncols = h, w
    ff.nbits = 8
    ff.nframes = 256
    ff.first = 0
    ff.camno = "TEST01"
    ff.fps = 25.0
    ff.starttime = "2026-01-01T00:00:00"
    ff.avepixel = ave.astype(np.uint8)
    ff.maxpixel = np.clip(ave + 5, 0, 255).astype(np.uint8)
    ff.stdpixel = np.ones((h, w), dtype=np.uint8)
    ff.maxframe = np.zeros((h, w), dtype=np.uint8)

    name = "FF_TEST01_20260101_000000_000_0000000.fits"
    FFfits.write(ff, str(tmp_path), name)

    import RMS.ConfigReader as cr
    config = cr.Config()
    config.gamma = 1.0
    config.bit_depth = 8

    star_list = extractStarsFF(str(tmp_path), name, config=config)
    assert star_list and len(star_list[1]) >= 1


def test_gate_factor_knob():
    # The sensitivity knob: a lower factor deepens detection on the same image
    from RMS.ExtractStars import adaptiveContrastThreshold
    c = contrastField(2.2)
    assert adaptiveContrastThreshold(c, factor=2.0) < adaptiveContrastThreshold(c, factor=3.0)
    assert adaptiveContrastThreshold(c) == adaptiveContrastThreshold(c, factor=NOISE_CONTRAST_FACTOR)

    img, pos = synthImage(0.6, [25.0, 4.5, 4.5])
    deep = found(extractStars(img, gate_factor=2.0), pos)
    default = found(extractStars(img, gate_factor=3.0), pos)
    assert sum(deep) >= sum(default)


def test_gate_factor_from_config(tmp_path):
    # extractStarsFF must honor config.star_gate_factor
    import RMS.ConfigReader as cr
    config = cr.Config()
    assert config.star_gate_factor == pytest.approx(3.0)


def test_extra_info_reports_measured_gate():
    # The tuner and the station logs need the gate in ADU: the factor alone cannot be
    # turned back into a threshold without the frame's own contrast field
    img, _ = synthImage(0.6, [25.0, 4.5, 4.5])
    extra = {}
    extractStars(img, gate_factor=5.0, extra_info=extra)
    assert 'gate_adu' in extra

    deep = {}
    extractStars(img, gate_factor=2.0, extra_info=deep)
    assert deep['gate_adu'] < extra['gate_adu']
