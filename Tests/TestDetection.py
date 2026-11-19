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
    config = SimpleNamespace(stationID='XX0001', height=10, width=10, fps=25, hot_pixels_filter=False)

    results = [('FF_XX0001_a.fits', [[], [], [], []], []), makeResult('FF_XX0001_b.fits')]

    calstars_name, ftpdetectinfo_name, ff_detected = DSM.saveDetections(results, str(tmp_path), config)

    assert ff_detected == []
    assert (tmp_path/calstars_name).exists()
    assert (tmp_path/ftpdetectinfo_name).exists()


def _segment(first_point, second_point, frame_min, frame_max):
    """ A line of the 3D detector: two points (x, y, frame) on it, the point count, the quality and the frame
    range.
    """

    return [first_point, second_point, 10, 10.0, frame_min, frame_max]


def test_merge3DLines_merges_segments_given_later_first():
    """ Two segments of one meteor must merge whatever their order in the list.

    The merge compares each segment with the vector from the start of the first segment to the end of the
    second one. With the later segment first that vector pointed back in time (about 180 deg from both
    segments), so the segments of one meteor stayed apart.
    """

    # A meteor moving 4 px right and 3 px down per frame, found as two segments overlapping in frames 15-20
    early = _segment((0, 0, 0), (40, 30, 10), 0, 20)
    late = _segment((80, 60, 20), (160, 120, 40), 15, 40)

    # Merged into one line covering both frame ranges, in either order
    for line_list in ([early, late], [late, early]):
        merged = Detection.merge3DLines(line_list, 20)
        assert len(merged) == 1
        assert merged[0][4:6] == [0, 40]


def test_merge3DLines_keeps_crossing_meteors_apart():
    """ Two meteors at the same time in different directions must not merge, whatever their order. """

    # One moving right, one moving down, on overlapping frames
    right = _segment((0, 100, 0), (100, 100, 20), 0, 20)
    down = _segment((300, 0, 5), (300, 100, 25), 5, 25)

    for line_list in ([right, down], [down, right]):
        assert len(Detection.merge3DLines(line_list, 20)) == 2
