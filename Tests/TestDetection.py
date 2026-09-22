""" Tests for the white ratio rejection warning cadence in RMS.Detection. """

from __future__ import print_function, division, absolute_import

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

import RMS.Detection as Detection


@pytest.fixture(autouse=True)
def reset_white_ratio_reject_count():
    """ Reset the process-local warning counter so tests remain independent. """

    original_count = Detection._white_ratio_reject_count
    Detection._white_ratio_reject_count = 0

    yield

    Detection._white_ratio_reject_count = original_count


def test_check_white_ratio_warning_cadence(monkeypatch):
    """ Rejections warn on the first and every 100th image, and otherwise log at debug level. """

    warning_counts = []
    debug_messages = []
    monkeypatch.setattr(Detection.log, 'warning',
        lambda message: warning_counts.append(Detection._white_ratio_reject_count))
    monkeypatch.setattr(Detection.log, 'debug', debug_messages.append)

    ff = SimpleNamespace(nrows=10, ncols=10)
    rejected_image = np.ones((ff.nrows, ff.ncols), dtype=np.uint8)

    results = [Detection.checkWhiteRatio(rejected_image, ff, 0.05) for _ in range(200)]

    assert not any(results)
    assert warning_counts == [1, 100, 200]
    assert len(debug_messages) == 197


def test_check_white_ratio_accepted_image_does_not_increment_counter(monkeypatch):
    """ Accepted images pass without affecting the rejection warning cadence. """

    warning_messages = []
    debug_messages = []
    monkeypatch.setattr(Detection.log, 'warning', warning_messages.append)
    monkeypatch.setattr(Detection.log, 'debug', debug_messages.append)

    ff = SimpleNamespace(nrows=10, ncols=10)
    accepted_image = np.zeros((ff.nrows, ff.ncols), dtype=np.uint8)

    assert Detection.checkWhiteRatio(accepted_image, ff, 0.05)
    assert Detection._white_ratio_reject_count == 0
    assert not warning_messages
    assert not debug_messages


def test_check_white_ratio_warning_preserves_threshold_precision(monkeypatch):
    """ The warning must distinguish a measured ratio just above the configured threshold. """

    warning_messages = []
    monkeypatch.setattr(Detection.log, 'warning', warning_messages.append)

    ff = SimpleNamespace(nrows=10, ncols=100)
    rejected_image = np.zeros((ff.nrows, ff.ncols), dtype=np.uint8)
    rejected_image.flat[:51] = 1

    assert not Detection.checkWhiteRatio(rejected_image, ff, 0.05)
    assert len(warning_messages) == 1
    assert "White ratio is 0.0510" in warning_messages[0]
    assert "max_white_ratio threshold: 0.0500" in warning_messages[0]


def test_gamma_white_point_uses_data_bit_depth(monkeypatch):
    """ An 8-bit FF with config.bit_depth = 12 must be gamma corrected with the 8-bit white point. """

    # The thresholding is memoized on the FF object and is not under test here
    monkeypatch.setattr(Detection, 'thresholdFF', lambda ff, k1, j1: None)

    rng = np.random.default_rng(0)
    maxpixel = rng.integers(50, 255, (16, 16)).astype(np.uint8)
    avepixel = rng.integers(10, 50, (16, 16)).astype(np.uint8)
    stdpixel = rng.integers(1, 10, (16, 16)).astype(np.uint8)
    ff = SimpleNamespace(maxpixel=maxpixel, avepixel=avepixel, stdpixel=stdpixel,
                         maxframe=np.zeros_like(maxpixel), nrows=16, ncols=16)
    img_handle = SimpleNamespace(ff=ff)

    results = []
    for bit_depth in (8, 12):
        config = SimpleNamespace(k1_det=1.5, j1_det=9, j1=5, gamma=0.8, bit_depth=bit_depth,
                                 min_patch_intensity_multiplier=2.0, detection_binning_method='avg',
                                 detection_binning_factor=1)
        results.append(Detection.thresholdAndCorrectGammaFF(img_handle, config, None))

    # The gamma corrected weights and patch intensity must not depend on the configured bit depth
    assert np.allclose(results[0][1], results[1][1])
    assert np.allclose(results[0][2], results[1][2])
    assert np.isclose(results[0][3], results[1][3])
