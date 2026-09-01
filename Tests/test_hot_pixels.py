""" Tests for the persistent hot pixel blacklist (RMS/HotPixels.py). """

import datetime
import json
import os

import numpy as np
import pytest

from RMS import HotPixels


class HPConfig(object):
    """ Minimal config stub with the hot pixel and file-resolution attributes. """

    def __init__(self, config_dir=None):
        self.hot_pixels_filter = True
        self.hot_pixels_file = 'hotpixels.json'
        self.hot_pixels_radius = 2.0
        self.hot_pixels_min_count = 15
        self.hot_pixels_min_frac = 0.01
        self.hot_pixels_min_span_minutes = 45.0
        self.hot_pixels_max_age_days = 14
        self.config_file_path = config_dir


def makeFFName(dt):
    return "FF_US005A_{:s}_0000000.fits".format(dt.strftime("%Y%m%d_%H%M%S_%f")[:-3])


def makeNight(n_ff=240, start=None, hot_pixels=(), hot_jitter=0.3, drift_px_per_min=0.5,
        hot_ff_slice=None, seed=1):
    """ Build a synthetic CALSTARS-style star list: rows are (y, x, intensity, amplitude, fwhm,
        bg, snr, nsat). One FF per minute. Real stars drift; hot pixels recur with sub-pixel
        jitter.
    """

    rng = np.random.RandomState(seed)

    if start is None:
        start = datetime.datetime(2026, 8, 25, 20, 30, 0)

    star_list = []

    for i in range(n_ff):

        dt = start + datetime.timedelta(minutes=i)
        rows = []

        # Drifting "real" stars
        for s in range(5):
            x = 100.0 + 80*s + drift_px_per_min*i
            y = 150.0 + 60*s + 0.3*drift_px_per_min*i
            rows.append((y, x, 500, 50, 2.5, 30, 10, 0))

        # Hot pixels: same spot every FF (within jitter)
        in_slice = hot_ff_slice is None or (hot_ff_slice[0] <= i < hot_ff_slice[1])
        if in_slice:
            for hx, hy in hot_pixels:
                rows.append((hy + hot_jitter*rng.uniform(-1, 1),
                             hx + hot_jitter*rng.uniform(-1, 1), 800, 200, 1.0, 30, 20, 0))

        star_list.append([makeFFName(dt), rows])

    return star_list


def clusterPositions(clusters):
    return [(c["x"], c["y"]) for c in clusters]


def test_hot_pixel_found_drifting_stars_ignored():
    config = HPConfig()
    star_list = makeNight(hot_pixels=[(600.4, 400.7)])

    clusters = HotPixels.findStationaryDetections(star_list, config)

    assert len(clusters) == 1
    cx, cy = clusterPositions(clusters)[0]
    assert abs(cx - 600.4) < 1.0
    assert abs(cy - 400.7) < 1.0


def test_no_hot_pixels_no_clusters():
    config = HPConfig()
    star_list = makeNight(hot_pixels=[])

    assert HotPixels.findStationaryDetections(star_list, config) == []


def test_short_recurrence_not_flagged():
    config = HPConfig()

    # Hot pixel visible for only 20 minutes - below the 45 min span gate
    star_list = makeNight(hot_pixels=[(600.4, 400.7)], hot_ff_slice=(0, 20))

    assert HotPixels.findStationaryDetections(star_list, config) == []


def test_min_count_gate():
    config = HPConfig()

    # Recurs on only 12/240 FFs - below the absolute floor of 15, even though it spans hours
    star_list = makeNight(hot_pixels=[(600.4, 400.7)])
    thinned = []
    for i, (ff_name, rows) in enumerate(star_list):
        if i % 20 == 0:
            thinned.append([ff_name, rows])
        else:
            thinned.append([ff_name, [r for r in rows if abs(r[1] - 600.4) > 5]])

    assert HotPixels.findStationaryDetections(thinned, config) == []


def test_sparse_recurrence_across_night_is_flagged():
    config = HPConfig()

    # The PSF roundness filter hides hot pixels on most FFs: only every 4th FF sees it (25 > 15
    # hits), but the hits cover the whole night - still a hot pixel
    star_list = makeNight(hot_pixels=[(600.4, 400.7)])
    thinned = []
    for i, (ff_name, rows) in enumerate(star_list):
        if i % 4 == 0:
            thinned.append([ff_name, rows])
        else:
            thinned.append([ff_name, [r for r in rows if abs(r[1] - 600.4) > 5]])

    clusters = HotPixels.findStationaryDetections(thinned, config)
    assert len(clusters) == 1


def test_slow_near_pole_star_not_flagged():
    config = HPConfig()

    # A near-pole star drifting 0.04 px/min sits inside the 2 px match circle for ~100 min - more
    # hits than a typical hot pixel, above the span gate, but confined to one contiguous block of
    # the night (and with a net drift across the circle). Verified failure mode from a real
    # US005A night: 383 hits in one 65 min window.
    star_list = []
    start = datetime.datetime(2026, 8, 25, 20, 30, 0)
    for i in range(240):
        dt = start + datetime.timedelta(minutes=i)
        rows = [(400.0, 550.0 + 0.04*i, 500, 50, 2.5, 30, 10, 0)]
        star_list.append([makeFFName(dt), rows])

    assert HotPixels.findStationaryDetections(star_list, config) == []


def test_multiple_hot_pixels():
    config = HPConfig()
    star_list = makeNight(hot_pixels=[(600.4, 400.7), (50.1, 700.9)])

    clusters = HotPixels.findStationaryDetections(star_list, config)
    positions = sorted(clusterPositions(clusters))

    assert len(clusters) == 2
    assert abs(positions[0][0] - 50.1) < 1.0
    assert abs(positions[1][0] - 600.4) < 1.0


def test_match_hot_pixels_radius():
    hp_xy = np.array([[100.0, 200.0]])

    matched = HotPixels.matchHotPixels([100.5, 103.0], [200.5, 200.0], hp_xy, 2.0)

    assert matched.tolist() == [True, False]
    assert HotPixels.matchHotPixels([], [], hp_xy, 2.0).tolist() == []
    assert HotPixels.matchHotPixels([1.0], [1.0], np.empty((0, 2)), 2.0).tolist() == [False]


def test_filter_star_list():
    hp_xy = np.array([[600.0, 400.0]])
    star_list = [
        ["FF1", [(400.3, 600.2, 1, 1, 1, 1, 1, 0), (100.0, 100.0, 1, 1, 1, 1, 1, 0)]],
        ["FF2", [(400.1, 599.8, 1, 1, 1, 1, 1, 0)]],
    ]

    filtered, n_removed = HotPixels.filterStarList(star_list, hp_xy, 2.0)

    assert n_removed == 2
    # FF2 lost its only star and is dropped entirely
    assert len(filtered) == 1
    assert filtered[0][0] == "FF1"
    assert len(filtered[0][1]) == 1
    assert filtered[0][1][0][0] == 100.0


def test_update_add_refresh_age():
    config = HPConfig()
    night1 = datetime.date(2026, 8, 25)

    # New pixel added
    hp_data = {"version": 1, "updated": None, "pixels": []}
    clusters = [{"x": 600.4, "y": 400.7, "n_det": 200, "n_ff": 200, "span_minutes": 239.0}]
    hp_data = HotPixels.updateHotPixelList(hp_data, clusters, night1, config)

    assert len(hp_data["pixels"]) == 1
    assert hp_data["pixels"][0]["last_seen"] == "2026-08-25"
    assert hp_data["pixels"][0]["nights_seen"] == 1

    # Seen again the next night - refreshed
    night2 = datetime.date(2026, 8, 26)
    hp_data = HotPixels.updateHotPixelList(hp_data, clusters, night2, config)

    assert len(hp_data["pixels"]) == 1
    assert hp_data["pixels"][0]["last_seen"] == "2026-08-26"
    assert hp_data["pixels"][0]["nights_seen"] == 2

    # Reprocessing an older night must not roll last_seen back or age anything out
    hp_data = HotPixels.updateHotPixelList(hp_data, clusters, night1, config)
    assert hp_data["pixels"][0]["last_seen"] == "2026-08-26"

    # Not seen for longer than max_age_days - aged out
    night_late = night2 + datetime.timedelta(days=config.hot_pixels_max_age_days + 1)
    hp_data = HotPixels.updateHotPixelList(hp_data, [], night_late, config)

    assert hp_data["pixels"] == []


def test_apply_hot_pixels_end_to_end(tmp_path):
    config_dir = tmp_path/"station"
    night_dir = tmp_path/"night"
    config_dir.mkdir()
    night_dir.mkdir()

    config = HPConfig(config_dir=str(config_dir))
    star_list = makeNight(hot_pixels=[(600.4, 400.7)])
    n_rows_before = sum(len(rows) for _, rows in star_list)

    filtered = HotPixels.applyHotPixels(star_list, str(night_dir), config)

    # All ~240 hot pixel rows removed, drifting stars untouched
    n_rows_after = sum(len(rows) for _, rows in filtered)
    assert n_rows_before - n_rows_after == 240

    # Master list and the night-dir audit copy were written
    for d in [config_dir, night_dir]:
        with open(os.path.join(str(d), config.hot_pixels_file)) as f:
            hp_data = json.load(f)
        assert len(hp_data["pixels"]) == 1
        assert abs(hp_data["pixels"][0]["x"] - 600.4) < 1.0

    # The live gate now sees the blacklist via the config-dir fallback
    hp_xy = HotPixels.loadHotPixelCoords(str(tmp_path/"nonexistent"), config)
    assert len(hp_xy) == 1


def test_load_missing_and_corrupt(tmp_path):
    config = HPConfig(config_dir=str(tmp_path))

    assert HotPixels.loadHotPixels(str(tmp_path), config)["pixels"] == []

    with open(str(tmp_path/"hotpixels.json"), "w") as f:
        f.write("not json{")

    assert HotPixels.loadHotPixels(str(tmp_path), config)["pixels"] == []


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
