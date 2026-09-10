from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

import RMS.Detection as Detection
import RMS.DetectStarsAndMeteors as DSM


def makeFF(nrows=10, ncols=10, excess=26, std=3):
    """ FF stand-in whose maxpixel sits a constant excess above avepixel. """

    return SimpleNamespace(nrows=nrows, ncols=ncols,
        maxpixel=np.full((nrows, ncols), 50 + excess, dtype=np.uint8),
        avepixel=np.full((nrows, ncols), 50, dtype=np.uint8),
        stdpixel=np.full((nrows, ncols), std, dtype=np.uint8))


def test_check_white_ratio_rejection_is_debug_only(monkeypatch):
    """ A single skipped image is normal and never reaches warning level. """

    warning_messages = []
    debug_messages = []
    monkeypatch.setattr(Detection.log, 'warning', warning_messages.append)
    monkeypatch.setattr(Detection.log, 'debug', debug_messages.append)

    ff = makeFF(nrows=10, ncols=100)
    rejected_image = np.zeros((ff.nrows, ff.ncols), dtype=np.uint8)
    rejected_image.flat[:51] = 1

    assert not Detection.checkWhiteRatio(rejected_image, ff, 0.05)
    assert not warning_messages
    assert len(debug_messages) == 1
    assert "White ratio is 0.0510" in debug_messages[0]
    assert "max_white_ratio threshold: 0.0500" in debug_messages[0]


def test_check_white_ratio_records_diagnostics():
    """ Checks, rejections and the noise floor of the first rejected image are recorded. """

    ff = makeFF(excess=26, std=3)
    rejected_image = np.ones((ff.nrows, ff.ncols), dtype=np.uint8)
    accepted_image = np.zeros((ff.nrows, ff.ncols), dtype=np.uint8)

    diagnostics = {}
    assert Detection.checkWhiteRatio(accepted_image, ff, 0.05, diagnostics=diagnostics)
    assert diagnostics == {'white_ratio_checks': 1, 'white_ratio_max': 0.0}

    assert not Detection.checkWhiteRatio(rejected_image, ff, 0.05, diagnostics=diagnostics)
    assert diagnostics['white_ratio_checks'] == 2
    assert diagnostics['white_ratio_rejections'] == 1
    assert diagnostics['white_ratio_max'] == 1.0
    assert diagnostics['maxpixel_excess_median'] == 26.0
    assert diagnostics['stdpixel_median'] == 3.0

    # The noise floor describes the first rejected image and is not overwritten
    assert not Detection.checkWhiteRatio(rejected_image, makeFF(excess=5, std=2), 0.05,
        diagnostics=diagnostics)
    assert diagnostics['white_ratio_rejections'] == 2
    assert diagnostics['maxpixel_excess_median'] == 26.0


def makeResult(name, skipped=None, ran=True):
    """ Detection result tuple for one image: not run (no stars), run and passed, or run and skipped. """

    if not ran:
        return (name, [[], [], [], []], [], {})

    diagnostics = {'white_ratio_checks': 1, 'white_ratio_max': 0.01}

    if skipped is not None:
        diagnostics.update({'white_ratio_rejections': 1, 'white_ratio_max': skipped,
            'maxpixel_excess_median': 26.0, 'stdpixel_median': 3.0})

    return (name, [[], [], [], []], [], diagnostics)


def test_report_white_ratio_skips_warns_on_mostly_skipped_night(monkeypatch):
    """ A night on which most star-bearing images were skipped is reported once, with the noise floor. """

    warning_messages = []
    monkeypatch.setattr(DSM.log, 'warning', warning_messages.append)

    results = [makeResult('FF{:03d}'.format(i), skipped=0.68) for i in range(8)]
    results += [makeResult('FF{:03d}'.format(i)) for i in range(8, 12)]
    results += [makeResult('FF{:03d}'.format(i), ran=False) for i in range(12, 30)]

    assert DSM.reportWhiteRatioSkips(results) == (8, 12)
    assert len(warning_messages) == 1
    assert "skipped on 8 of 12 images" in warning_messages[0]
    assert "median white ratio 0.68" in warning_messages[0]
    assert "26.0 ADU against a median stdpixel of 3.0 ADU" in warning_messages[0]


def test_report_white_ratio_skips_silent_on_isolated_rejections(monkeypatch):
    """ A few bright images on an otherwise normal night are not a warning. """

    warning_messages = []
    monkeypatch.setattr(DSM.log, 'warning', warning_messages.append)

    results = [makeResult('FF{:03d}'.format(i), skipped=0.07) for i in range(2)]
    results += [makeResult('FF{:03d}'.format(i)) for i in range(2, 40)]

    assert DSM.reportWhiteRatioSkips(results) == (2, 40)
    assert not warning_messages


def test_report_white_ratio_skips_silent_below_minimum_images(monkeypatch):
    """ A handful of skipped twilight images with no real night behind them is not a warning. """

    warning_messages = []
    monkeypatch.setattr(DSM.log, 'warning', warning_messages.append)

    results = [makeResult('FF{:03d}'.format(i), skipped=0.3) for i in range(5)]

    assert DSM.reportWhiteRatioSkips(results) == (5, 5)
    assert not warning_messages


def test_save_detections_accepts_results_without_diagnostics(tmp_path, monkeypatch):
    """ Results restored from backups written before the diagnostics were added still save. """

    monkeypatch.setattr(DSM.log, 'warning', lambda message: None)
    config = SimpleNamespace(stationID='XX0001', height=10, width=10, fps=25)

    results = [('FF_XX0001_a.fits', [[], [], [], []], []), makeResult('FF_XX0001_b.fits')]

    calstars_name, ftpdetectinfo_name, ff_detected = DSM.saveDetections(results, str(tmp_path), config)

    assert ff_detected == []
    assert (tmp_path/calstars_name).exists()
    assert (tmp_path/ftpdetectinfo_name).exists()
