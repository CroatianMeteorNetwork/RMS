""" Tests of continuous capture during polar day and polar night.

    During polar day or night the Sun never crosses the day/night switching horizon, so the capture mode never
    switches and the capture session never ended: nothing was processed, no timelapses were made and nothing
    was uploaded until the polar day/night was over (months). The capture mode switcher now splits the session
    once a day during polar conditions. These tests simulate a station through whole years at several
    locations, including AQ000A at Arrival Heights, Antarctica.
"""

from __future__ import print_function, division, absolute_import

import os
import types
import ctypes
import multiprocessing
from datetime import datetime, timedelta

import ephem
import pytest

import logging

import RMS.ConfigReader as cr
import RMS.StartCapture as sc
import RMS.Reprocess as rp
from RMS.Misc import RmsDateTime
import Utils.GenerateTimelapse as gt
from RMS.CaptureModeSwitcher import captureModeAt, polarSessionBoundary, switcherStep, lastNightToDaySwitch, \
    POLAR_RECHECK_SECONDS, SWITCH_HORIZON_DEG


# StartCapture creates its logger only when run as a script
if not hasattr(sc, 'log'):
    sc.log = logging.getLogger("rmslogger")

REPO_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.config')

# Station locations: name -> (latitude, longitude, elevation)
SITES = {
    'AQ000A_ArrivalHeights': (-77.8476, 166.6710, 184),
    'AQ0001_TerraNovaBay': (-74.62035, 164.22646, 30),
    'CA_Elginfield': (43.1928, -81.3157, 323),
    'FI_Helsinki': (60.1699, 24.9384, 20),
    'NO_Tromso': (69.6496, 18.9560, 10),
    'NO_Svalbard': (78.2232, 15.6267, 30),
    'AQ_SouthPole': (-89.99, 0.0, 2835),
    'Equator': (0.0, 0.0, 0),
}

YEAR_START = datetime(2026, 1, 1)
YEAR_END = datetime(2027, 1, 1)


class Clock(object):
    """ Fake clock for simulations. """

    def __init__(self, t):
        self.t = t

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += timedelta(seconds=seconds)


def siteConfig(name, switch_camera_modes=True):
    lat, lon, elev = SITES[name]
    return types.SimpleNamespace(latitude=lat, longitude=lon, elevation=elev,
        switch_camera_modes=switch_camera_modes, capture_wait_seconds=30, continuous_capture=True)


def sunIsUp(config, t):
    """ Independent check: is the upper limb of the Sun above the switching horizon at the given time (the
        definition ephem uses for rising and setting). """

    o = ephem.Observer()
    o.lat, o.long, o.elevation = str(config.latitude), str(config.longitude), config.elevation
    o.date = ephem.Date(t)
    s = ephem.Sun()
    s.compute(o)
    return float(s.alt) + float(s.radius) > float(ephem.degrees(SWITCH_HORIZON_DEG))


def simulateSwitcher(config, start, end):
    """ Run the capture mode switcher with a fake clock from start to end.

    Return:
        steps: [list] (time, daytime) after every step.
        changes: [list] (time, daytime) of every mode change.
        triggers: [list] Times when the camera mode switch was triggered.
        cuts: [list] Session split times.
    """

    clock = Clock(start)
    daytime_mode = multiprocessing.Value(ctypes.c_bool, False, lock=False)
    trigger = multiprocessing.Value(ctypes.c_bool, False, lock=False)
    session_cut = multiprocessing.Value(ctypes.c_int, 0, lock=False)
    session_cut_time = multiprocessing.Value(ctypes.c_double, 0.0, lock=False)

    state = {}
    steps, changes, triggers, cuts = [], [], [], []
    prev_mode, prev_cut = None, 0

    while clock.t < end:

        wait = switcherStep(config, state, daytime_mode, trigger, session_cut=session_cut,
            session_cut_time=session_cut_time, now_func=clock.now, sleep_func=clock.sleep)

        if trigger.value:
            triggers.append(clock.t)
            trigger.value = False

        if daytime_mode.value != prev_mode:
            changes.append((clock.t, daytime_mode.value))
            prev_mode = daytime_mode.value

        if session_cut.value != prev_cut:
            assert session_cut.value == prev_cut + 1
            cuts.append(datetime(1970, 1, 1) + timedelta(seconds=session_cut_time.value))
            prev_cut = session_cut.value

        steps.append((clock.t, daytime_mode.value))

        assert 0 <= wait <= 86400 + 60
        clock.sleep(wait)

    return steps, changes, triggers, cuts


@pytest.fixture(scope='module')
def yearly_runs():
    """ One simulated year of the switcher for every site (computed once). """

    return {name: simulateSwitcher(siteConfig(name), YEAR_START, YEAR_END) for name in SITES}


### CAPTURE MODE SWITCHER ###

@pytest.mark.parametrize('name', sorted(SITES))
def test_mode_follows_the_sun(yearly_runs, name):
    """ The capture mode matches the Sun all year, except right after a change which may be detected late:
        by the camera switch stagger normally, and by up to the hourly re-check when a polar period ends. """

    config = siteConfig(name)
    steps, changes, _, _ = yearly_runs[name]

    max_lag = timedelta(seconds=POLAR_RECHECK_SECONDS + config.capture_wait_seconds + 60)

    # Sample the mode in effect every 20 minutes through the year
    t, i, mismatches = YEAR_START, 0, 0
    while t < YEAR_END - timedelta(days=1):

        while (i + 1 < len(steps)) and (steps[i + 1][0] <= t):
            i += 1

        mode = steps[i][1]
        if mode != sunIsUp(config, t):

            # The Sun must have crossed the horizon shortly before
            assert sunIsUp(config, t - max_lag) != sunIsUp(config, t), \
                "{}: wrong mode at {} (daytime={})".format(name, t, mode)
            mismatches += 1

        t += timedelta(minutes=20)

    # Mismatches only happen around changes
    assert mismatches <= 2*len(changes) + 2


@pytest.mark.parametrize('name', sorted(SITES))
def test_camera_switched_only_on_mode_changes(yearly_runs, name):
    """ The camera is switched (and rebooted) only when the mode changes, not on every hourly re-check during
        polar day or night. """

    _, changes, triggers, _ = yearly_runs[name]
    assert len(triggers) == len(changes)


@pytest.mark.parametrize('name', sorted(SITES))
def test_session_ends_at_least_daily(yearly_runs, name):
    """ In continuous mode the capture session ends (data is processed and uploaded) at the end of every
        night, or at the daily split during polar day/night. There is never a gap longer than ~24 h. """

    _, changes, _, cuts = yearly_runs[name]

    # Sessions end on night->day switches and on splits
    ends = sorted([t for t, daytime in changes if daytime] + cuts)
    ends = [YEAR_START] + ends + [YEAR_END]

    gaps = [(b - a) for a, b in zip(ends[:-1], ends[1:])]
    assert max(gaps) <= timedelta(hours=25, minutes=2), "{}: longest gap {}".format(name, max(gaps))


@pytest.mark.parametrize('name', sorted(SITES))
def test_splits_only_at_culmination_without_horizon_crossing(yearly_runs, name):
    """ Splits happen only at solar midnight with the Sun above the switching horizon (polar day and the
        days around it), or at solar noon with the Sun below it (polar night and the days around it). """

    config = siteConfig(name)
    _, _, _, cuts = yearly_runs[name]

    for cut in cuts:

        daytime = sunIsUp(config, cut)

        # At the pole the culmination is not defined, the split falls back to UTC midnight
        if abs(config.latitude) > 89:
            continue

        o = ephem.Observer()
        o.lat, o.long, o.elevation = str(config.latitude), str(config.longitude), config.elevation
        o.date = ephem.Date(cut - timedelta(hours=1))
        culmination = (o.next_antitransit(ephem.Sun()) if daytime else o.next_transit(ephem.Sun())).datetime()

        assert abs((culmination - cut).total_seconds()) < 5, \
            "{}: split at {} is not at the {} culmination {}".format(name, cut,
                "lower" if daytime else "upper", culmination)


def test_no_splits_at_mid_and_low_latitudes(yearly_runs):
    for name in ('CA_Elginfield', 'Equator'):
        assert yearly_runs[name][3] == []


def test_high_latitude_summer(yearly_runs):
    """ Above ~57.6 deg the Sun stays above the -9 deg switching horizon around the summer solstice, so the
        session is split daily there too (not only at polar stations). Only polar day, no polar night. """

    for name in ('FI_Helsinki', 'NO_Tromso'):

        config = siteConfig(name)
        _, _, _, cuts = yearly_runs[name]

        assert len(cuts) > 40, "{}: only {} splits".format(name, len(cuts))

        for c in cuts:
            assert sunIsUp(config, c), "{}: split at {} with the Sun below the horizon".format(name, c)
            assert 4 <= c.month <= 8, "{}: split outside of the summer at {}".format(name, c)

        # One split per day
        for a, b in zip(cuts[:-1], cuts[1:]):
            assert timedelta(hours=23, minutes=50) <= b - a <= timedelta(hours=24, minutes=10)


def test_aq000a_polar_periods(yearly_runs):
    """ AQ000A: polar day (Sun above -9 deg) from Sep 30 to mid March, polar night around the June
        solstice. One split a day during each. """

    _, _, _, cuts = yearly_runs['AQ000A_ArrivalHeights']
    config = siteConfig('AQ000A_ArrivalHeights')

    day_cuts = [c for c in cuts if sunIsUp(config, c)]
    night_cuts = [c for c in cuts if not sunIsUp(config, c)]

    assert day_cuts and night_cuts

    # Polar day of 2026 starts at the end of September, the first split is on Sep 30 or Oct 1
    oct_cuts = [c for c in day_cuts if c.month >= 9]
    assert datetime(2026, 9, 29) <= oct_cuts[0] <= datetime(2026, 10, 2)

    # Solar midnight at Arrival Heights is around 12:50 UTC
    for c in day_cuts:
        assert 12 <= c.hour <= 13

    # One split per day
    for a, b in zip(day_cuts[:-1], day_cuts[1:]):
        if b - a < timedelta(days=2):
            assert timedelta(hours=23, minutes=50) <= b - a <= timedelta(hours=24, minutes=10)

    # Polar night around the June solstice, splits at solar noon (~00:50 UTC)
    assert all(datetime(2026, 5, 1) < c < datetime(2026, 8, 15) for c in night_cuts)
    assert any(c.month == 6 for c in night_cuts)
    for c in night_cuts:
        assert c.hour in (0, 1)


def test_capture_mode_at_last_night_before_polar_day():
    """ The last short night before the polar day is a night, even though the following night does not
        exist (the old switcher treated this as polar day because it looked at both the next sunrise and
        sunset). """

    config = siteConfig('AQ000A_ArrivalHeights')

    # Find the middle of the last night before the polar day
    t = datetime(2026, 9, 29, 0, 0)
    nights = []
    while t < datetime(2026, 9, 30, 12):
        if not sunIsUp(config, t):
            nights.append(t)
        t += timedelta(minutes=5)

    assert nights
    mid = nights[len(nights)//2]

    daytime, next_change, polar = captureModeAt(config, mid)
    assert (daytime, polar) == (False, None)
    assert next_change > mid


def test_polar_session_boundary_within_a_day():

    for name in SITES:
        config = siteConfig(name)
        for polar in ('day', 'night'):
            t = datetime(2026, 3, 1)
            while t < datetime(2026, 3, 4):
                b = polarSessionBoundary(config, t, polar)
                assert t < b <= t + timedelta(hours=25)
                t += timedelta(hours=7)


### WAIT ###

def test_wait_returns_on_session_cut(monkeypatch):

    monkeypatch.setattr(sc.time, 'sleep', lambda s: None)
    monkeypatch.setattr(sc, 'STOP_CAPTURE', False)

    daytime_mode = multiprocessing.Value(ctypes.c_bool, True, lock=False)
    session_cut = multiprocessing.Value(ctypes.c_int, 3, lock=False)

    # Already handled - would wait forever, so stop it via a mode switch after a few calls
    calls = {'n': 0}

    def sleeper(s):
        calls['n'] += 1
        if calls['n'] == 3:
            session_cut.value += 1

    monkeypatch.setattr(sc.time, 'sleep', sleeper)
    assert sc.wait(None, None, None, None, daytime_mode, session_cut=session_cut, session_cut_handled=3) \
        == "session_cut"


def test_wait_mode_switch_takes_precedence(monkeypatch):

    monkeypatch.setattr(sc, 'STOP_CAPTURE', False)

    daytime_mode = multiprocessing.Value(ctypes.c_bool, True, lock=False)
    session_cut = multiprocessing.Value(ctypes.c_int, 0, lock=False)

    def sleeper(s):
        daytime_mode.value = False
        session_cut.value = 1

    monkeypatch.setattr(sc.time, 'sleep', sleeper)
    assert sc.wait(None, None, None, None, daytime_mode, session_cut=session_cut, session_cut_handled=0) \
        == "mode_switch"


def test_wait_without_session_cut_unchanged(monkeypatch):
    """ Standard mode passes no session_cut, wait() must behave as before. """

    monkeypatch.setattr(sc, 'STOP_CAPTURE', False)
    monkeypatch.setattr(sc.time, 'sleep', lambda s: None)
    assert sc.wait(0, None, None, None, None) == "normal"


### RUN CAPTURE CONTROL FLOW ###

class FakeProcess(object):
    def __init__(self, *args, **kwargs):
        self.exit = multiprocessing.Event()
        self.pid = 0

    def startCapture(self):
        pass

    def stopCapture(self):
        return 0

    def is_alive(self):
        return True

    def start(self):
        pass


class FakeQueue(object):
    def qsize(self):
        return 0


class FakeDetector(object):
    def __init__(self, *args, **kwargs):
        self.input_queue = FakeQueue()

    def startPool(self):
        pass

    def closePool(self):
        pass

    def getResults(self):
        return []

    def shutdownManager(self):
        pass

    def deleteBackupFiles(self):
        pass

    def updateCoreNumber(self, cores):
        pass


class FakeUploadManager(object):
    def __init__(self):
        self.files = []

    def addFiles(self, files):
        self.files.extend(files)

    def delayNextUpload(self, delay=0):
        pass

    def is_alive(self):
        return False


class CaptureHarness(object):
    """ Runs runCapture() in continuous mode with all capture, compression and processing mocked. The
        script is a list of actions, one per call to wait(), each changing the shared state the way the
        capture mode switcher would. """

    def __init__(self, monkeypatch, tmp_path, start_daytime, initial_cuts=0):

        self.config = cr.parse(REPO_CONFIG)
        self.config.data_dir = str(tmp_path)
        self.config.continuous_capture = True
        self.config.mask_download_permissive = False
        self.config.save_frames = True
        self.config.save_frame_times = False
        self.config.raw_video_save = False
        self.config.timelapse_generate_from_frames = True
        self.config.live_maxpixel_enable = False
        self.config.reboot_after_processing = False
        self.config.upload_delay = 0
        self.config.postprocess_delay = 0
        self.config.config_file_name = os.path.join(str(tmp_path), 'nonexistent.config')

        self.daytime_mode = multiprocessing.Value(ctypes.c_bool, start_daytime, lock=False)
        self.trigger = multiprocessing.Value(ctypes.c_bool, False, lock=False)
        self.session_cut = multiprocessing.Value(ctypes.c_int, initial_cuts, lock=False)
        self.session_cut_time = multiprocessing.Value(ctypes.c_double, 0.0, lock=False)

        self.processed_nights = []
        self.compressor_stops = 0

        # Called with the harness when a night is processed, to change the state in the meantime
        self.on_process = None
        self.timelapse_cutoffs = []
        self.deletes = 0
        self.compressors = 0

        # Order of the disk cleanups ('delete') and timelapse runs ('timelapse'), and the results the scripted
        #   cleanups return (True when the list is empty)
        self.events = []
        self.delete_results = []

        # Paths each disk cleanup was told to keep, and whether the night processing fails
        self.delete_keeps = []
        self.processing_fails = False
        self.upload_manager = FakeUploadManager()
        self.script = []
        self.clock = Clock(datetime(2026, 10, 10, 0, 0))

        harness = self

        class HarnessDetector(FakeDetector):
            def __init__(self, *args, **kwargs):
                FakeDetector.__init__(self)
                harness.events.append('detect')

        class FakeCompressor(FakeProcess):
            def __init__(self, night_data_dir, *args, **kwargs):
                FakeProcess.__init__(self)
                harness.compressors += 1
                harness.events.append('compress')
                self.detector = kwargs.get('detector')

            def stop(self):
                harness.compressor_stops += 1
                harness.events.append('stop')
                return self.detector

        def fakeProcessNight(night_data_dir, config, detection_results=None, nodetect=False):
            harness.processed_nights.append(night_data_dir)
            harness.events.append('process')
            if harness.on_process is not None:
                harness.on_process(harness)
            if harness.processing_fails:
                raise RuntimeError("Processing failed")
            name = os.path.basename(night_data_dir)
            return night_data_dir, name + '_detected.tar.bz2', None, None, None

        def fakeProcessFramesFiles(config, cutoff_utc=None):
            harness.timelapse_cutoffs.append(cutoff_utc)
            harness.events.append('timelapse')
            return ['timelapse_{:d}.tar'.format(len(harness.timelapse_cutoffs))]

        def fakeDelete(*args, **kwargs):
            harness.deletes += 1
            harness.events.append('delete')
            harness.delete_keeps.append(kwargs.get('keep_paths'))
            return harness.delete_results.pop(0) if harness.delete_results else True

        def fakeWait(duration, compressor, bc, video_file, daytime_mode=None, session_cut=None,
                session_cut_handled=None, mode_at_start=None):

            # Same as the real wait(): a mode which already differs from the session start ends the wait at once
            mode_at_entry = daytime_mode.value if mode_at_start is None else mode_at_start
            if daytime_mode.value != mode_at_entry:
                return "mode_switch"

            action = harness.script.pop(0)

            if action == 'stop':
                sc.STOP_CAPTURE = True
                return "normal"

            elif action == 'day':
                daytime_mode.value = True

            elif action == 'night':
                daytime_mode.value = False

            elif action.startswith('cut'):
                # 'cut' or 'cutN' - N splits before the main loop gets to handle them
                n = int(action[3:] or 1)
                for _ in range(n):
                    harness.clock.sleep(86400)
                    harness.session_cut_time.value = (harness.clock.t - datetime(1970, 1, 1)).total_seconds()
                    harness.session_cut.value += 1

            # Same checks as the real wait()
            if daytime_mode.value != mode_at_entry:
                return "mode_switch"
            if (session_cut is not None) and (session_cut.value != session_cut_handled):
                return "session_cut"

            raise AssertionError("Scripted action {} did not end the wait".format(action))

        for name, value in [
                ('BufferedCapture', FakeProcess), ('Compressor', FakeCompressor), ('QueuedPool', HarnessDetector),
                ('processNight', fakeProcessNight), ('processFramesFiles', fakeProcessFramesFiles),
                ('deleteOldObservations', fakeDelete), ('wait', fakeWait),
                ('runExternalScript', lambda *a, **k: None),
                ('startObservationSummaryReport', lambda *a, **k: ''),
                ('getObservationSummaryDict', lambda *a, **k: {}),
                ('addObsParam', lambda *a, **k: None),
                ('getPlatepar', lambda *a, **k: (None, None, None)),
                ('compareConfigs', lambda *a, **k: ''),
                ('frameBufferShape', lambda config: (1, 1, 1)),
                ('setSIGINT', lambda: None), ('resetSIGINT', lambda: None),
                ]:
            monkeypatch.setattr(sc, name, value)

        monkeypatch.setattr(sc, 'STOP_CAPTURE', False)

        # Unique directory names for every new night
        counter = {'n': 0}

        class FakeDateTime(object):
            @staticmethod
            def utcnow():
                counter['n'] += 1
                return datetime(2026, 10, 10) + timedelta(hours=counter['n'])

        monkeypatch.setattr(sc, 'RmsDateTime', FakeDateTime)

    def run(self, script):

        self.script = list(script)
        with pytest.raises(SystemExit):
            sc.runCapture(self.config, upload_manager=self.upload_manager, daytime_mode=self.daytime_mode,
                camera_mode_switch_trigger=self.trigger, session_cut=self.session_cut,
                session_cut_time=self.session_cut_time)

        assert self.script == [], "Not all scripted actions were used: {}".format(self.script)


def cutTime(harness, n):
    """ Time of the n-th scripted split (1-based). """
    return datetime(2026, 10, 10) + timedelta(days=n)


def test_runcapture_polar_day(monkeypatch, tmp_path):
    """ Polar day: every split makes and uploads the daytime timelapse up to the split and frees the disk.
        No night is processed. """

    h = CaptureHarness(monkeypatch, tmp_path, start_daytime=True)
    h.run(['cut', 'cut', 'cut', 'stop'])

    assert h.processed_nights == []
    assert h.timelapse_cutoffs == [cutTime(h, 1), cutTime(h, 2), cutTime(h, 3)]
    assert h.upload_manager.files == ['timelapse_1.tar', 'timelapse_2.tar', 'timelapse_3.tar']
    assert h.deletes == 3
    assert h.compressors == 0

    # The disk is cleaned up before the timelapses are made, so the new archives are not deleted before
    #   they are uploaded
    assert h.events == ['delete', 'timelapse']*3


def test_runcapture_polar_night(monkeypatch, tmp_path):
    """ Polar night: every split processes and uploads the night so far and starts a new night directory. """

    h = CaptureHarness(monkeypatch, tmp_path, start_daytime=False)
    h.run(['cut', 'cut', 'stop'])

    # Three night directories: two split off, and the last one processed on stop
    assert len(h.processed_nights) == 3
    assert len(set(h.processed_nights)) == 3
    assert h.compressors == 3

    # The timelapses of the split nights use the frames up to the split
    assert h.timelapse_cutoffs == [cutTime(h, 1), cutTime(h, 2), None]

    nights = [os.path.basename(d) + '_detected.tar.bz2' for d in h.processed_nights]
    assert h.upload_manager.files == [nights[0], 'timelapse_1.tar', nights[1], 'timelapse_2.tar',
        nights[2], 'timelapse_3.tar']

    # At every split the disk is freed for the next night while the split night is still compressed, then
    #   the compressors are swapped, the split night is processed and the disk is freed again before its
    #   timelapses are made
    assert h.events == ['detect', 'compress'] \
        + ['delete', 'detect', 'stop', 'compress', 'process', 'delete', 'timelapse']*2 \
        + ['stop', 'process', 'timelapse']


def test_runcapture_polar_night_keeps_split_night(monkeypatch, tmp_path):
    """ The night split off is still being uploaded and used by the external script when the cleanups at
        the split and at the start of the new night run, so they must not delete it or its archives. Later
        cleanups can. """

    h = CaptureHarness(monkeypatch, tmp_path, start_daytime=False)
    h.run(['cut', 'cut', 'stop'])

    # Night directory, archive directory (the same in the fake processing), the uploaded archive and the
    #   directory of the next night, which is already being captured
    kept = [[d, d, os.path.basename(d) + '_detected.tar.bz2', n] for d, n
        in zip(h.processed_nights, h.processed_nights[1:])]

    # Cleanups at the start of the second night (before the first is processed, only its directory exists,
    #   and the new night directory), at the first split, the start of the third night and the second split
    split = [[d, n] for d, n in zip(h.processed_nights, h.processed_nights[1:])]
    assert h.delete_keeps == [split[0], kept[0], split[1], kept[1]]


def test_runcapture_polar_night_keeps_unprocessed_night(monkeypatch, tmp_path):
    """ A night whose processing failed is kept too, its data is needed to reprocess it on the next
        startup. """

    h = CaptureHarness(monkeypatch, tmp_path, start_daytime=False)
    h.processing_fails = True
    h.run(['cut', 'stop'])

    assert h.delete_keeps == [h.processed_nights[:2], [h.processed_nights[0], None, h.processed_nights[1]]]


def test_runcapture_polar_night_capture_goes_on(monkeypatch, tmp_path):
    """ The next night is compressed while the split night is processed, so no capture is lost, and the
        night is processed once it ends. """

    h = CaptureHarness(monkeypatch, tmp_path, start_daytime=False)
    h.run(['cut', 'stop'])

    assert h.events.index('compress', 2) < h.events.index('process')

    # The cleanup and the detector for the next night run before the split night's compressor is stopped,
    #   and the next compressor is started right after it
    stop = h.events.index('stop')
    assert h.events[stop - 2:stop + 2] == ['delete', 'detect', 'stop', 'compress']
    assert h.compressors == 2
    assert h.compressor_stops == 2
    assert len(set(h.processed_nights)) == 2


def test_runcapture_polar_night_ends_with_sunrise_during_processing(monkeypatch, tmp_path):
    """ The Sun rose while the split night was processed: the night started at the split ends at once, is
        processed, and the daytime capture starts. """

    h = CaptureHarness(monkeypatch, tmp_path, start_daytime=False)

    def sunrise(harness):
        if len(harness.processed_nights) == 1:
            harness.daytime_mode.value = True

    h.on_process = sunrise
    h.run(['cut', 'stop'])

    assert len(set(h.processed_nights)) == 2
    assert h.compressor_stops == 2

    # The second night ended without waiting, the daytime capture took the 'stop' action
    assert h.timelapse_cutoffs == [cutTime(h, 1), None]


def test_runcapture_stop_during_split_processing(monkeypatch, tmp_path):
    """ Capture stopped while the split night was processed: the compression of the night started at the
        split is stopped too, so the program can exit (it is reprocessed on the next startup). """

    h = CaptureHarness(monkeypatch, tmp_path, start_daytime=False)

    def stop(harness):
        sc.STOP_CAPTURE = True

    h.on_process = stop
    h.run(['cut'])

    assert h.compressors == 2
    assert h.compressor_stops == 2
    assert len(h.processed_nights) == 1


def test_runcapture_reboot_after_split(monkeypatch, tmp_path):
    """ With a reboot after the processing, the next night is not started at the split: the capture stops for
        the reboot after the split night is processed. """

    h = CaptureHarness(monkeypatch, tmp_path, start_daytime=False)
    h.config.reboot_after_processing = True
    h.script = ['cut']
    sc.runCapture(h.config, upload_manager=h.upload_manager, daytime_mode=h.daytime_mode,
        camera_mode_switch_trigger=h.trigger, session_cut=h.session_cut, session_cut_time=h.session_cut_time)

    assert h.script == []
    assert h.compressors == 1
    assert h.compressor_stops == 1
    assert len(h.processed_nights) == 1


def test_runcapture_keeps_new_night_outside_polar_night(monkeypatch, tmp_path):
    """ Outside the polar night only the directory of the night being started is kept by the cleanup. """

    h = CaptureHarness(monkeypatch, tmp_path, start_daytime=False)
    h.run(['day', 'cut', 'night', 'day', 'stop'])

    # Only the polar day split cleans up without anything to keep, the start of the second night keeps just
    #   its own new directory
    assert h.delete_keeps == [None, [h.processed_nights[1]]]


def test_runcapture_split_low_disk(monkeypatch, tmp_path):
    """ If there is not enough disk space at a split, the cleanup is repeated after the timelapses are made
        (they delete the source frames). Capture stops only if there is still not enough space. """

    # Polar day: enough space after the timelapses
    h = CaptureHarness(monkeypatch, tmp_path, start_daytime=True)
    h.delete_results = [False, True]
    h.run(['cut', 'stop'])
    assert h.events == ['delete', 'timelapse', 'delete']

    # Polar day: still not enough space, capture stops at the split
    h = CaptureHarness(monkeypatch, tmp_path, start_daytime=True)
    h.delete_results = [False, False]
    h.run(['cut'])
    assert h.events == ['delete', 'timelapse', 'delete']

    # Polar night: not enough space for the next night at the split, so it is not started before the split
    #   night is processed. Still not enough space after the timelapses, capture stops without a new night
    h = CaptureHarness(monkeypatch, tmp_path, start_daytime=False)
    h.delete_results = [False, False, False]
    h.run(['cut'])
    assert h.events == ['detect', 'compress', 'delete', 'stop', 'process', 'delete', 'timelapse', 'delete']
    assert h.compressors == 1
    assert len(os.listdir(os.path.join(str(tmp_path), h.config.captured_dir))) == 1

    # Polar night: enough space once the timelapses are made, the next night starts after the processing
    h = CaptureHarness(monkeypatch, tmp_path, start_daytime=False)
    h.delete_results = [False, False, True]
    h.run(['cut', 'stop'])
    assert h.events == ['detect', 'compress', 'delete', 'stop', 'process', 'delete', 'timelapse', 'delete',
        'delete', 'detect', 'compress', 'stop', 'process', 'timelapse']


def test_runcapture_normal_and_transitions(monkeypatch, tmp_path):
    """ Normal nights are unchanged, and the transitions in and out of polar day work. """

    h = CaptureHarness(monkeypatch, tmp_path, start_daytime=False)
    h.run([
        'day',      # sunrise - night 1 processed, timelapse up to the end of the night
        'cut',      # polar day split
        'cut',      # polar day split
        'night',    # polar day over, first night
        'day',      # sunrise - night 2 processed
        'stop',
        ])

    assert len(h.processed_nights) == 2
    assert h.timelapse_cutoffs == [None, cutTime(h, 1), cutTime(h, 2), None]
    assert h.compressors == 2


def test_runcapture_missed_and_stale_splits(monkeypatch, tmp_path):
    """ A split from before the capture started is not handled, and several splits arriving at once are
        handled once (the timelapse covers all frames up to the latest split). """

    h = CaptureHarness(monkeypatch, tmp_path, start_daytime=True, initial_cuts=5)
    h.run(['cut3', 'cut', 'stop'])

    assert h.timelapse_cutoffs == [cutTime(h, 3), cutTime(h, 4)]


def test_runcapture_start_in_day_then_polar_night(monkeypatch, tmp_path):
    """ Started during the day (no night directory yet), the night starts and becomes a polar night. """

    h = CaptureHarness(monkeypatch, tmp_path, start_daytime=True)
    h.run(['night', 'cut', 'stop'])

    # Night processed on the split and on stop, the daytime part ended without processing (no split)
    assert len(h.processed_nights) == 2
    assert h.timelapse_cutoffs == [cutTime(h, 1), None]


### TIMELAPSES FROM FRAMES ###

def makeFrame(frames_dir, station, t, daytime):
    """ Create an empty frame file named and placed like RawFrameSave does. """

    d = os.path.join(frames_dir, t.strftime("%Y/%Y%m%d-%j/%Y%m%d-%j_%H"))
    if not os.path.isdir(d):
        os.makedirs(d)

    name = "{}_{}_{:03d}_{}.jpg".format(station, t.strftime("%Y%m%d_%H%M%S"), 0, 'd' if daytime else 'n')
    path = os.path.join(d, name)
    open(path, 'w').close()
    return path


@pytest.fixture
def frames_env(monkeypatch, tmp_path):
    """ Config with a frames directory and a mocked video encoder which records the frame blocks, writes
        dummy video files and deletes the used frames (frame_cleanup = delete). """

    config = cr.parse(REPO_CONFIG)
    config.data_dir = str(tmp_path)
    config.frame_cleanup = 'delete'
    config.stationID = 'AQ000A'
    config.latitude, config.longitude, config.elevation = SITES['AQ000A_ArrivalHeights']
    config.continuous_capture = True

    frames_dir = os.path.join(config.data_dir, config.frame_dir)
    os.makedirs(frames_dir)

    blocks = []

    def fakeEncode(image_files, frames_root, video_path, fps=30, base_crf=25, cleanup_mode='none',
            compression='bz2', use_color=True):

        blocks.append([gt._timestampFromName(f) for f in image_files])
        modes = set(gt._modeFromName(f) for f in image_files)
        assert len(modes) == 1, "Mixed day/night frames in one timelapse"

        json_path = video_path.replace('.mp4', '.json')
        for p in (video_path, json_path):
            with open(p, 'w') as f:
                f.write('x')

        if cleanup_mode == 'delete':
            for f in image_files:
                os.remove(f)

        return video_path, json_path

    monkeypatch.setattr(gt, 'generateTimelapseFromFrames', fakeEncode)

    return config, frames_dir, blocks


def test_polar_day_daily_timelapses(frames_env):
    """ Frames every 10 minutes for three polar days, a split every day: one timelapse per day with the frames
        up to the split, later frames are left for the next one. """

    config, frames_dir, blocks = frames_env

    start = datetime(2026, 10, 10, 12, 50)
    t = start
    while t < start + timedelta(days=3):
        makeFrame(frames_dir, 'AQ000A', t, True)
        t += timedelta(minutes=10)

    cut1 = start + timedelta(days=1)
    archives = rp.processFramesFiles(config, cutoff_utc=cut1)

    assert len(blocks) == 1
    assert blocks[0][0] == start and blocks[0][-1] < cut1
    assert len(archives) == 1 and archives[0].endswith('_frames_timelapse.tar')
    assert os.path.isfile(archives[0])

    archives = rp.processFramesFiles(config, cutoff_utc=cut1 + timedelta(days=1))
    assert len(blocks) == 2
    assert blocks[1][0] >= cut1 and blocks[1][-1] < cut1 + timedelta(days=1)
    assert len(archives) == 1


def test_polar_day_backlog(frames_env):
    """ A station which has not uploaded for 8 days (e.g. AQ000A after the update): the first split makes
        one timelapse per UTC day, none longer than 24 hours. """

    config, frames_dir, blocks = frames_env

    start = datetime(2026, 10, 5, 3, 41)
    t = start
    made = 0
    while t < datetime(2026, 10, 13, 13, 10):
        makeFrame(frames_dir, 'AQ000A', t, True)
        made += 1
        t += timedelta(minutes=10)

    cutoff = datetime(2026, 10, 13, 12, 52)
    rp.processFramesFiles(config, cutoff_utc=cutoff)

    assert len(blocks) >= 8
    for b in blocks:
        assert b[-1] - b[0] < timedelta(days=1)
        assert b[-1] < cutoff

    # All frames before the cutoff were used, the ones after (13:01) were left for the next split
    used = sum(len(b) for b in blocks)
    left = sum(1 for _, _, f in os.walk(frames_dir) for x in f if x.endswith('.jpg'))
    assert left == 1 and used + left == made


### WHOLE YEAR AT ARRIVAL HEIGHTS ###

def test_aq000a_full_year_frames(frames_env, monkeypatch):
    """ Simulate AQ000A in continuous mode through a whole year: frames are saved every 30 minutes with the
        day/night suffix of the current mode, and timelapses are made the way runCapture does it - at the
        end of every night, and at every daily split during polar day/night. Every frame must end up in
        exactly one timelapse, timelapses never mix day and night frames and are never longer than 24 h,
        and no frame waits more than ~25 hours to be uploaded. """

    config, frames_dir, blocks = frames_env

    clock = Clock(YEAR_START)
    monkeypatch.setattr(rp, 'lastNightToDaySwitch', lambda cfg: lastNightToDaySwitch(cfg, clock.t))

    daytime_mode = multiprocessing.Value(ctypes.c_bool, False, lock=False)
    trigger = multiprocessing.Value(ctypes.c_bool, False, lock=False)
    session_cut = multiprocessing.Value(ctypes.c_int, 0, lock=False)
    session_cut_time = multiprocessing.Value(ctypes.c_double, 0.0, lock=False)

    state = {}
    next_switcher = YEAR_START
    next_frame = YEAR_START
    frames_made = 0
    prev_mode = None
    handled_cut = 0
    processing_delay = timedelta(minutes=10)
    pending = []        # (time to process, cutoff)

    while clock.t < YEAR_END:

        # Next thing to happen
        t_next = min([next_switcher, next_frame] + [p[0] for p in pending])
        clock.t = t_next

        if clock.t >= next_switcher:
            wait = switcherStep(config, state, daytime_mode, trigger, session_cut=session_cut,
                session_cut_time=session_cut_time, now_func=clock.now, sleep_func=lambda s: None)
            next_switcher = clock.t + timedelta(seconds=wait)

            # Night -> day: the night is processed, timelapse up to the end of the night
            if (prev_mode is False) and daytime_mode.value:
                pending.append((clock.t + processing_delay, None))

            # Daily split during polar day/night
            if session_cut.value != handled_cut:
                handled_cut = session_cut.value
                cut = datetime(1970, 1, 1) + timedelta(seconds=session_cut_time.value)
                pending.append((clock.t + processing_delay, cut))

            prev_mode = daytime_mode.value

        if clock.t >= next_frame:
            makeFrame(frames_dir, 'AQ000A', clock.t, daytime_mode.value)
            frames_made += 1
            next_frame = clock.t + timedelta(minutes=30)

        for p in [p for p in pending if p[0] <= clock.t]:
            pending.remove(p)
            n_before = len(blocks)
            rp.processFramesFiles(config, cutoff_utc=p[1])

            # Upload latency: the oldest frame in the new timelapses was captured at most ~25 h ago
            for b in blocks[n_before:]:
                assert clock.t - b[0] <= timedelta(hours=25, minutes=30), \
                    "Frames from {} uploaded only at {}".format(b[0], clock.t)

    # Every frame in exactly one timelapse, except the ones captured after the last processing
    timestamps = [t for b in blocks for t in b]
    assert len(timestamps) == len(set(timestamps))
    left = sum(1 for _, _, f in os.walk(frames_dir) for x in f if x.endswith('.jpg'))
    assert len(timestamps) + left == frames_made
    assert left <= 50

    # No timelapse longer than 24 hours
    for b in blocks:
        assert b[-1] - b[0] < timedelta(days=1)

    # Daytime timelapses were made all through the polar day (Oct - Dec)
    months = set(b[0].month for b in blocks)
    assert {1, 2, 6, 10, 11, 12} <= months


def test_switcher_thread_runs_now_at_aq000a():
    """ Smoke test: the real switcher thread with the real clock at Arrival Heights. Check the mode it sets
        against the Sun now, and that it stops cleanly. """

    import threading
    from RMS.CaptureModeSwitcher import captureModeSwitcher

    config = siteConfig('AQ000A_ArrivalHeights', switch_camera_modes=False)
    daytime_mode = multiprocessing.Value(ctypes.c_bool, False, lock=False)
    trigger = multiprocessing.Value(ctypes.c_bool, False, lock=False)
    session_cut = multiprocessing.Value(ctypes.c_int, 0, lock=False)
    session_cut_time = multiprocessing.Value(ctypes.c_double, 0.0, lock=False)
    stop_event = threading.Event()

    thread = threading.Thread(target=captureModeSwitcher, args=(config, daytime_mode, trigger, stop_event),
        kwargs={'session_cut': session_cut, 'session_cut_time': session_cut_time})
    thread.daemon = True
    thread.start()

    import time
    time.sleep(2)
    assert thread.is_alive()
    assert daytime_mode.value == sunIsUp(config, RmsDateTime.utcnow())
    assert session_cut.value == 0

    stop_event.set()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_delete_old_observations_keeps_paths(monkeypatch, tmp_path):
    """ The disk cleanup deletes everything it can except the kept night directories and archives, and
        stops instead of waiting for more space when only kept objects are left. """

    import RMS.DeleteOldObservations as dob

    config = cr.parse(REPO_CONFIG)
    config.data_dir = str(tmp_path)
    config.quota_management_enabled = False

    captured = os.path.join(str(tmp_path), config.captured_dir)
    archived = os.path.join(str(tmp_path), config.archived_dir)

    # Two nights, each with a captured and an archived directory and an archive file
    nights = ['{:s}_2026101{:d}_000000_000000'.format(config.stationID, i) for i in (1, 2)]
    for night in nights:
        os.makedirs(os.path.join(captured, night))
        os.makedirs(os.path.join(archived, night))
        open(os.path.join(archived, night + '_detected.tar.bz2'), 'w').close()

    # Never enough space, so everything that can be deleted is
    monkeypatch.setattr(dob, 'availableSpace', lambda path: 0)
    monkeypatch.setattr(dob.time, 'sleep', lambda t: None)

    keep_paths = [os.path.join(captured, nights[1]), os.path.join(archived, nights[1]),
        os.path.join(archived, nights[1] + '_detected.tar.bz2'), None]

    space_ok = dob.deleteOldObservations(config.data_dir, config.captured_dir, config.archived_dir, config,
        duration=3600, keep_paths=keep_paths)

    assert not space_ok
    assert sorted(os.listdir(captured)) == [nights[1]]
    assert sorted(os.listdir(archived)) == sorted([nights[1], nights[1] + '_detected.tar.bz2'])


def test_retention_keeps_paths(monkeypatch, tmp_path):
    """ The count-based retention runs before the space check and must not delete the kept night either,
        e.g. an older split night which failed to process, kept to be reprocessed on the next startup. """

    import RMS.DeleteOldObservations as dob

    config = cr.parse(REPO_CONFIG)
    config.data_dir = str(tmp_path)
    config.quota_management_enabled = False
    config.arch_dirs_to_keep = 1
    config.capt_dirs_to_keep = 1
    config.bz2_files_to_keep = 1

    captured = os.path.join(str(tmp_path), config.captured_dir)
    archived = os.path.join(str(tmp_path), config.archived_dir)

    # Three nights, the middle one is kept
    nights = ['{:s}_2026101{:d}_000000_000000'.format(config.stationID, i) for i in (0, 1, 2)]
    for night in nights:
        os.makedirs(os.path.join(captured, night))
        os.makedirs(os.path.join(archived, night))
        open(os.path.join(archived, night + '_detected.tar.bz2'), 'w').close()

    # Plenty of space, so only the retention deletes anything
    monkeypatch.setattr(dob, 'availableSpace', lambda path: 1e15)

    keep_paths = [os.path.join(captured, nights[1]), os.path.join(archived, nights[1]),
        os.path.join(archived, nights[1] + '_detected.tar.bz2')]

    assert dob.deleteOldObservations(config.data_dir, config.captured_dir, config.archived_dir, config,
        duration=3600, keep_paths=keep_paths)

    # The newest night is within the counts to keep, the kept one is on top of them, the oldest is deleted
    assert sorted(os.listdir(captured)) == nights[1:]
    assert sorted(os.listdir(archived)) == sorted(nights[1:] + [n + '_detected.tar.bz2' for n in nights[1:]])
