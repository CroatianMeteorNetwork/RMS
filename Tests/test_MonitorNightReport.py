"""
Tests for the night reports of RMS.MonitorProcessFrameInterface (RMS.MonitorNightReport) and the saving of
the star extraction chunks as image pairs.
"""

import datetime
import json
import os
import shutil

import numpy as np
import pytest

import RMS.ConfigReader as cr
import RMS.MonitorNightReport as mnr
from RMS.DetectStarsAndMeteors import chunkForFrame, saveResultsFrameInterface
from RMS.Formats import CALSTARS, FFfile, FFpng, FTPdetectinfo
from RMS.Formats.FrameInterface import getCacheID
from RMS.MonitorProcessFrameInterface import ChunkImageSaver


REPO_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.config')

BEG_TIME = datetime.datetime(2025, 12, 25, 3, 0, 0, 12345)


def _config(lat=43.19, lon=-81.32, elev=300.0):
    config = cr.parse(REPO_CONFIG)
    config.stationID = 'XX0001'
    config.latitude = lat
    config.longitude = lon
    config.elevation = elev
    config.width = 64
    config.height = 48
    return config


@pytest.fixture
def config_path(tmp_path):
    path = str(tmp_path/'test.config')
    shutil.copy(REPO_CONFIG, path)
    return path


### Night naming ###

def test_night_contains_evening_and_morning():

    config = _config()

    # Local time is UTC-5: 21:00 local in the evening, and 05:00 local the next morning
    evening = datetime.datetime(2025, 12, 25, 2, 0, 0)
    morning = datetime.datetime(2025, 12, 25, 10, 0, 0)

    name_evening, start, end = mnr.nightInfo(config, evening)
    name_morning, _, _ = mnr.nightInfo(config, morning)

    assert name_evening == name_morning
    assert name_evening.startswith('XX0001_20251224_')
    assert start < evening < morning < end


def test_dusk_belongs_to_the_coming_night():

    config = _config()

    _, start, end = mnr.nightInfo(config, datetime.datetime(2025, 12, 25, 2, 0, 0))

    # Shortly before the night starts (the Sun is above the capture horizon at dusk)
    dusk = start - datetime.timedelta(minutes=10)
    name_dusk, start_dusk, _ = mnr.nightInfo(config, dusk)

    assert start_dusk == start
    assert name_dusk == mnr.nightInfo(config, start + datetime.timedelta(hours=2))[0]


def test_consecutive_nights_differ():

    config = _config()

    name1, _, end1 = mnr.nightInfo(config, datetime.datetime(2025, 12, 25, 2, 0, 0))
    name2, start2, _ = mnr.nightInfo(config, datetime.datetime(2025, 12, 26, 2, 0, 0))

    assert name1 != name2
    assert end1 < start2


@pytest.mark.parametrize('dt', [datetime.datetime(2025, 12, 21, 12, 0, 0),
                                datetime.datetime(2025, 6, 21, 12, 0, 0)])
def test_polar_night_and_day(dt):

    # 78 deg N has polar night in December and polar day in June
    config = _config(lat=78.2, lon=15.6, elev=10.0)

    name, start, end = mnr.nightInfo(config, dt)

    assert start <= dt < end
    assert (end - start) == datetime.timedelta(days=1)
    assert mnr.nightInfo(config, dt + datetime.timedelta(hours=6))[0] == name


def test_nightBoundsFromName():

    config = _config()
    name, start, end = mnr.nightInfo(config, datetime.datetime(2025, 12, 25, 2, 0, 0))

    start_name, end_name = mnr.nightBoundsFromName(config, name)

    assert abs((start_name - start).total_seconds()) < 1
    assert end_name == end


### Done flag ###

def test_done_flag_round_trip_and_backward_compatibility(tmp_path):

    mnr.writeDoneFlag(str(tmp_path), {'night': 'XX0001_20251224_220000_000000', 'begin': BEG_TIME,
                                      'n_images': 3})

    info = mnr.readDoneFlag(str(tmp_path))
    assert info['night'] == 'XX0001_20251224_220000_000000'
    assert info['begin'] == BEG_TIME.strftime(mnr.JSON_TIME_FORMAT)

    # Older results have an empty flag
    open(str(tmp_path/'done.flag'), 'w').close()
    assert mnr.readDoneFlag(str(tmp_path)) == {}

    assert mnr.readDoneFlag(str(tmp_path/'missing')) == {}


### Chunk images ###

class _FakeFF(object):
    def __init__(self, nframes, dtype=np.uint16, level=1000):
        self.nframes = nframes
        self.maxpixel = np.full((48, 64), level + 500, dtype=dtype)
        self.avepixel = np.full((48, 64), level, dtype=dtype)
        self.successful = True


class _FakeHandle(object):
    """ Image handle with 25 fps frames, 128 frames per chunk and 300 frames in total. """

    input_type = 'video'

    def __init__(self, fps=25.0, total_frames=300, chunk_frames=128):
        self.fps = fps
        self.total_frames = total_frames
        self.chunk_frames = chunk_frames
        self.total_fr_chunks = total_frames//chunk_frames
        self.current_frame_chunk = 0
        self.beginning_datetime = BEG_TIME
        self.dir_path = '.'
        self.cache = {}
        self.loaded = []

    def currentFrameTime(self, frame_no=None, dt_obj=False):
        return self.beginning_datetime + datetime.timedelta(seconds=frame_no/self.fps)

    def loadChunk(self, first_frame=None, read_nframes=None):
        self.loaded.append((first_frame, read_nframes))
        return _FakeFF(read_nframes)


def test_chunk_image_saver(tmp_path):

    config = _config()
    handle = _FakeHandle()
    saver = ChunkImageSaver(str(tmp_path), config, 'input.mkv')

    names = []
    for chunk in range(handle.total_fr_chunks):
        handle.current_frame_chunk = chunk
        names.append(saver(handle, _FakeFF(128)))

    # The trailing 44 frames (256-299) are saved as well, after removing a possibly calibrated cached chunk
    handle.cache[getCacheID(256, 44)] = 'calibrated'
    names.append(saver.saveTrailing(handle))

    assert getCacheID(256, 44) not in handle.cache
    assert handle.loaded == [(256, 44)]

    assert [c[:2] for c in saver.chunk_images] == [(0, 128), (128, 128), (256, 44)]
    assert names == [c[2] for c in saver.chunk_images]
    assert names[1] == FFfile.constructFFName('XX0001', handle.currentFrameTime(128), frame=128,
                                              suffix=FFpng.PAIR_MAX_SUFFIX)

    ff = FFfile.read(str(tmp_path), names[2])
    assert ff.nframes == 44 and ff.first == 256
    assert ff.maxpixel.dtype == np.uint16


def test_trailing_chunk_not_saved_if_too_short(tmp_path):

    handle = _FakeHandle(total_frames=257)
    saver = ChunkImageSaver(str(tmp_path), _config(), 'input.mkv')

    assert saver.saveTrailing(handle) is None
    assert handle.loaded == []


def test_chunkForFrame():

    chunks = [(0, 128, 'a'), (128, 128, 'b'), (256, 44, 'c')]

    assert chunkForFrame(chunks, 0) == 'a'
    assert chunkForFrame(chunks, 127.9) == 'a'
    assert chunkForFrame(chunks, 128) == 'b'
    assert chunkForFrame(chunks, 299) == 'c'

    # After the last chunk, the last preceding chunk is used
    assert chunkForFrame(chunks, 320) == 'c'

    # A gap (unsaved chunk) maps to the preceding chunk
    assert chunkForFrame([(0, 128, 'a'), (256, 128, 'c')], 200) == 'a'

    assert chunkForFrame([(128, 128, 'b')], 10) is None
    assert chunkForFrame([], 10) is None


def _meteor(frames):
    """ Meteor with picks on the given frames: (rho, theta, centroids). """

    # Columns: frame, x, y, level, background, SNR, saturated pixel count
    centroids = np.array([[f, 10.0 + i, 20.0 + i, 100, 5, 8.0, 0] for i, f in enumerate(frames)],
                         dtype=np.float64)

    return 10.0, 45.0, centroids


def test_meteor_times_round_trip_through_chunk_names(tmp_path):

    config = _config()
    handle = _FakeHandle(fps=25.0)
    chunk_images = [(0, 128, FFfile.constructFFName('XX0001', handle.currentFrameTime(0), frame=0,
                                                    suffix=FFpng.PAIR_MAX_SUFFIX)),
                    (128, 128, FFfile.constructFFName('XX0001', handle.currentFrameTime(128), frame=128,
                                                      suffix=FFpng.PAIR_MAX_SUFFIX))]

    # Two meteors in the second chunk (one with a rolling shutter fraction), one in the first
    meteors = [_meteor([130.0, 131.5, 133.0]), _meteor([140.0, 141.0]), _meteor([5.0, 6.0, 7.0])]
    pick_frames = [[130.0, 131.5, 133.0], [140.0, 141.0], [5.0, 6.0, 7.0]]

    _, _, ftp_name = saveResultsFrameInterface([], meteors, handle, config, chunk_frames=128,
                                               output_dir=str(tmp_path), chunk_images=chunk_images)

    entries = FTPdetectinfo.readFTPdetectinfo(str(tmp_path), ftp_name)

    assert [(e[0], e[2]) for e in entries] == [(chunk_images[1][2], 1), (chunk_images[1][2], 2),
                                               (chunk_images[0][2], 1)]

    for entry, frames in zip(entries, pick_frames):

        ff_name, meteor_fps, meas = entry[0], entry[4], entry[11]
        ref_time = FFfile.filenameToDatetime(ff_name)

        assert meteor_fps == pytest.approx(25.0)

        for line, frame in zip(meas, frames):
            pick_time = ref_time + datetime.timedelta(seconds=line[1]/meteor_fps)
            true_time = handle.currentFrameTime(frame)
            assert abs((pick_time - true_time).total_seconds()) < 0.001


def test_meteor_names_unchanged_without_chunk_images(tmp_path):

    config = _config()
    handle = _FakeHandle(fps=25.0)

    _, _, ftp_name = saveResultsFrameInterface([], [_meteor([130.0, 131.0])], handle, config,
                                               chunk_frames=128, output_dir=str(tmp_path))

    entry = FTPdetectinfo.readFTPdetectinfo(str(tmp_path), ftp_name)[0]

    assert entry[0] == FFfile.constructFFName('XX0001', handle.currentFrameTime(130))
    assert entry[11][0][1] == 0


### Merging and reporting ###

NIGHT = 'XX0001_20251224_222000_000000'


def _makeResults(output_dir, night_dir, config, file_index, n_chunks=2, with_meteor=True):
    """ Write the results of one input file and its chunk images, like processFile does. """

    handle = _FakeHandle(fps=25.0, total_frames=128*n_chunks)
    handle.beginning_datetime = BEG_TIME + datetime.timedelta(minutes=10*file_index)

    saver = ChunkImageSaver(night_dir, config, 'input_{:d}.mkv'.format(file_index))

    star_list = []
    for chunk in range(n_chunks):
        handle.current_frame_chunk = chunk
        name = saver(handle, _FakeFF(128))
        star_list.append([name, [(10.0, 20.0, 1000, 50, 2.0, 7, 9.0, 0)]])

    results_dir = os.path.join(output_dir, '2025', '202512', '20251225', 'input_{:d}'.format(file_index))
    os.makedirs(results_dir)

    meteors = [_meteor([130.0, 131.0, 132.0])] if with_meteor else []
    saveResultsFrameInterface(star_list, meteors, handle, config, chunk_frames=128, output_dir=results_dir,
                              chunk_images=saver.chunk_images)

    shutil.copy(REPO_CONFIG, os.path.join(results_dir, 'test.config'))

    with open(os.path.join(results_dir, config.platepars_recalibrated_name), 'w') as f:
        json.dump({name: {'file': file_index} for _, _, name in saver.chunk_images}, f)

    mnr.writeDoneFlag(results_dir, {'night': NIGHT, 'n_images': n_chunks})

    return results_dir, saver.chunk_images


def test_scan_and_merge_night(tmp_path):

    config = _config()
    output_dir = str(tmp_path)
    night_dir = mnr.nightDirPath(output_dir, NIGHT)
    os.makedirs(night_dir)

    results = [_makeResults(output_dir, night_dir, config, i) for i in range(2)]

    # Results without a night (older results) are ignored
    old_dir = tmp_path/'2025'/'202512'/'20251225'/'old'
    old_dir.mkdir()
    (old_dir/'done.flag').write_text('')

    nights = mnr.scanNights(output_dir)
    assert list(nights) == [NIGHT]
    assert nights[NIGHT] == sorted(r[0] for r in results)

    calstars_name, ftp_name, fps, chunk_frames, ff_detected = mnr.mergeNightResults(night_dir, nights[NIGHT],
                                                                                    config)

    # One CALSTARS with all chunks, named by the image pairs in the night directory
    calstars_files = [f for f in os.listdir(night_dir) if f.startswith('CALSTARS')]
    assert calstars_files == ['CALSTARS_{:s}.txt'.format(NIGHT)] == [calstars_name]

    star_list, chunk_frames, calstars_fps = CALSTARS.readCALSTARS(night_dir, calstars_name, return_fps=True)
    night_ffs = set(f for f in os.listdir(night_dir) if FFfile.validFFName(f))
    assert len(star_list) == 4
    assert set(entry[0] for entry in star_list) == night_ffs
    assert chunk_frames == 128 and calstars_fps == pytest.approx(25.0) and fps == pytest.approx(25.0)

    # One FTPdetectinfo with both meteors, keeping their fps
    entries = FTPdetectinfo.readFTPdetectinfo(night_dir, ftp_name)
    assert len(entries) == 2
    assert all(e[4] == pytest.approx(25.0) for e in entries)
    assert ff_detected == sorted(e[0] for e in entries)
    assert set(ff_detected) <= night_ffs

    # Recalibrated platepars of both files
    with open(os.path.join(night_dir, config.platepars_recalibrated_name)) as f:
        assert len(json.load(f)) == 4

    # Merging again doesn't duplicate the merged files
    mnr.mergeNightResults(night_dir, nights[NIGHT], config)
    assert len([f for f in os.listdir(night_dir) if f.startswith('FTPdetectinfo')]) == 1


def test_unreported_results(tmp_path):

    night_dir = str(tmp_path)
    results_dirs = ['/out/a', '/out/b']

    assert mnr.unreportedResults(night_dir, results_dirs) == results_dirs

    with open(os.path.join(night_dir, mnr.REPORT_STATE_FILE_NAME), 'w') as f:
        json.dump({'files': ['a']}, f)

    assert mnr.unreportedResults(night_dir, results_dirs) == ['/out/b']


def test_generate_night_report(tmp_path):

    config = _config()
    config.timelapse_generate_captured = False
    config.thumb_bin = 1

    output_dir = str(tmp_path)
    night_dir = mnr.nightDirPath(output_dir, NIGHT)
    os.makedirs(night_dir)

    for i in range(2):
        _makeResults(output_dir, night_dir, config, i, n_chunks=3)

    state = mnr.generateNightReport(output_dir, NIGHT, config, archive=False)

    assert 'merge' in state['ok_steps']
    assert 'thumbnails_and_stacks' in state['ok_steps']
    assert sorted(state['files']) == ['input_0', 'input_1']

    night_files = os.listdir(night_dir)
    assert NIGHT + '_captured_stack.jpg' in night_files
    assert any(f.endswith('_CAPTURED_thumbs.jpg') for f in night_files)

    assert mnr.readReportState(night_dir)['files'] == state['files']
    assert mnr.unreportedResults(night_dir, mnr.scanNights(output_dir)[NIGHT]) == []


### Scheduling ###

def _touchResults(output_dir, night, name, mtime):
    results_dir = os.path.join(output_dir, '2025', name)
    os.makedirs(results_dir)
    mnr.writeDoneFlag(results_dir, {'night': night})
    os.utime(os.path.join(results_dir, 'done.flag'), (mtime, mtime))
    return results_dir


def _ts(dt):
    return (dt - datetime.datetime(1970, 1, 1)).total_seconds()


@pytest.fixture
def night_setup(tmp_path, config_path):

    config = cr.parse(config_path)
    night_name, night_start, night_end = mnr.nightInfo(config, datetime.datetime(2025, 12, 25, 3, 0, 0))

    last_result = night_end - datetime.timedelta(minutes=30)
    results_dir = _touchResults(str(tmp_path), night_name, 'f1', _ts(last_result))

    return str(tmp_path), config_path, night_name, night_end, last_result, [results_dir]


def test_sunrise_mode(night_setup):

    output_dir, config_path, night, night_end, last_result, results = night_setup
    reporter = mnr.NightReporter(output_dir, config_path, report_mode='sunrise')

    # Before the end of the night
    assert not reporter.isDue(night, results, True, night_end - datetime.timedelta(minutes=1))

    # After the end of the night, but not idle
    assert not reporter.isDue(night, results, False, night_end + datetime.timedelta(minutes=1))

    # After the end of the night and idle
    assert reporter.isDue(night, results, True, night_end + datetime.timedelta(minutes=1))


def test_idle_mode_waits_for_quiet_period(night_setup):

    output_dir, config_path, night, _, last_result, results = night_setup
    reporter = mnr.NightReporter(output_dir, config_path, report_mode='idle')

    quiet = datetime.timedelta(minutes=reporter.config.monitor_report_quiet_min)

    assert not reporter.isDue(night, results, True, last_result + quiet - datetime.timedelta(seconds=10))
    assert reporter.isDue(night, results, True, last_result + quiet + datetime.timedelta(seconds=10))


@pytest.mark.parametrize('mode', ['external', 'none'])
def test_external_modes_only_report_on_trigger(night_setup, mode):

    output_dir, config_path, night, night_end, _, results = night_setup
    reporter = mnr.NightReporter(output_dir, config_path, report_mode=mode)

    later = night_end + datetime.timedelta(days=1)
    assert not reporter.isDue(night, results, True, later)

    # Create the trigger file
    open(os.path.join(output_dir, mnr.REPORT_TRIGGER_FILE_NAME), 'w').close()
    reporter._checkTrigger({night: results})

    assert not os.path.exists(os.path.join(output_dir, mnr.REPORT_TRIGGER_FILE_NAME))
    assert reporter.isDue(night, results, False, later)


def test_failed_report_retry(night_setup):

    output_dir, config_path, night, night_end, _, results = night_setup
    reporter = mnr.NightReporter(output_dir, config_path, report_mode='idle', fail_wait_time=300)

    now = night_end + datetime.timedelta(hours=1)

    # Wait before retrying a failed report
    reporter.failed[night] = {'count': 1, 'time': now, 'n_files': len(results)}
    assert not reporter.isDue(night, results, True, now + datetime.timedelta(seconds=100))
    assert reporter.isDue(night, results, True, now + datetime.timedelta(seconds=400))

    # Give up after the second failure
    reporter.failed[night] = {'count': 2, 'time': now, 'n_files': len(results)}
    assert not reporter.isDue(night, results, True, now + datetime.timedelta(days=1))

    # ... until new files arrive
    assert reporter.isDue(night, results + ['/new'], True, now + datetime.timedelta(days=1))


def test_delete_old_night_images(tmp_path):

    output_dir = str(tmp_path)
    night_dir = mnr.nightDirPath(output_dir, NIGHT)
    os.makedirs(night_dir)

    FFpng.writePair(night_dir, 'FF_XX0001_20251225_030000_000_0000000_maxpixel.png',
                    np.zeros((4, 4), np.uint8), np.zeros((4, 4), np.uint8))
    open(os.path.join(night_dir, 'stack.jpg'), 'w').close()

    reported_at = datetime.datetime(2025, 12, 25, 12, 0, 0)
    with open(os.path.join(night_dir, mnr.REPORT_STATE_FILE_NAME), 'w') as f:
        json.dump({'reported_at': reported_at.strftime(mnr.JSON_TIME_FORMAT), 'failed_steps': []}, f)

    # Not old enough, or deleting disabled
    assert mnr.deleteOldNightImages(output_dir, 2, now=reported_at + datetime.timedelta(days=1)) == []
    assert mnr.deleteOldNightImages(output_dir, 0, now=reported_at + datetime.timedelta(days=10)) == []

    assert mnr.deleteOldNightImages(output_dir, 2, now=reported_at + datetime.timedelta(days=3)) == [NIGHT]
    assert sorted(os.listdir(night_dir)) == sorted([mnr.REPORT_STATE_FILE_NAME, 'stack.jpg'])


def test_finished_report_uploads_archives(night_setup, monkeypatch):

    output_dir, config_path, night, _, _, results = night_setup

    calls = []

    class _FakeUploadManager(object):
        def __init__(self, config):
            calls.append(('init', config.data_dir))
        def start(self):
            calls.append(('start',))
        def addFiles(self, file_list):
            calls.append(('add', file_list))
        def delayNextUpload(self, delay=0):
            calls.append(('delay', delay))
        def stop(self):
            calls.append(('stop',))

    import RMS.UploadManager
    monkeypatch.setattr(RMS.UploadManager, 'UploadManager', _FakeUploadManager)

    class _FinishedProcess(object):
        exitcode = 0
        def join(self, timeout=None):
            pass
        def is_alive(self):
            return False

    reporter = mnr.NightReporter(output_dir, config_path, report_mode='idle')
    reporter.upload = True

    night_dir = mnr.nightDirPath(output_dir, night)
    os.makedirs(night_dir)
    with open(os.path.join(night_dir, mnr.REPORT_RESULT_FILE_NAME), 'w') as f:
        json.dump({'upload_files': ['/archives/night_detected.tar.bz2']}, f)

    reporter.active = (_FinishedProcess(), night, results)
    reporter._finishReport(datetime.datetime(2025, 12, 26))
    reporter.stop()

    # The upload queue is kept in the output directory
    assert calls[0] == ('init', output_dir)
    assert ('add', ['/archives/night_detected.tar.bz2']) in calls
    assert calls[-1] == ('stop',)
    assert night not in reporter.failed


### Review fixes ###

def test_chunk_saver_ignores_handles_without_chunks(tmp_path):

    class _FFHandle(object):
        input_type = 'ff'

    saver = ChunkImageSaver(str(tmp_path), _config(), 'FF_input.fits')

    assert saver(_FFHandle(), _FakeFF(256)) is None
    assert saver.chunk_images == []
    assert os.listdir(str(tmp_path)) == []


def test_failed_report_stays_pending(tmp_path, monkeypatch):

    import RMS.ArchiveDetections

    config = _config()
    config.timelapse_generate_captured = False

    output_dir = str(tmp_path)
    night_dir = mnr.nightDirPath(output_dir, NIGHT)
    os.makedirs(night_dir)
    _makeResults(output_dir, night_dir, config, 0)

    def _fail(*args, **kwargs):
        raise RuntimeError("stack failed")

    monkeypatch.setattr(RMS.ArchiveDetections, 'generateThumbsAndStacks', _fail)

    state = mnr.generateNightReport(output_dir, NIGHT, config, archive=False)

    assert state['failed_steps'] == ['thumbnails_and_stacks']
    assert state['attempted_files'] == ['input_0']
    assert state['files'] == []

    # The night is still pending, so the report is retried
    assert mnr.unreportedResults(night_dir, mnr.scanNights(output_dir)[NIGHT])


def test_cleanup_keeps_images_of_nights_with_late_files(tmp_path):

    output_dir = str(tmp_path)
    night_dir = mnr.nightDirPath(output_dir, NIGHT)
    os.makedirs(night_dir)

    FFpng.writePair(night_dir, 'FF_XX0001_20251225_030000_000_0000000_maxpixel.png',
                    np.zeros((4, 4), np.uint8), np.zeros((4, 4), np.uint8))

    reported_at = datetime.datetime(2025, 12, 25, 12, 0, 0)
    with open(os.path.join(night_dir, mnr.REPORT_STATE_FILE_NAME), 'w') as f:
        json.dump({'reported_at': reported_at.strftime(mnr.JSON_TIME_FORMAT), 'failed_steps': [],
                   'files': ['f1']}, f)

    # A late file for the night, which is not reported yet
    _touchResults(output_dir, NIGHT, 'f1', _ts(reported_at))
    _touchResults(output_dir, NIGHT, 'f2', _ts(reported_at + datetime.timedelta(days=5)))

    assert mnr.deleteOldNightImages(output_dir, 2, now=reported_at + datetime.timedelta(days=5)) == []
    assert len(os.listdir(night_dir)) == 3


def test_stop_finishes_completed_report(night_setup, monkeypatch):

    output_dir, config_path, night, _, _, results = night_setup

    reporter = mnr.NightReporter(output_dir, config_path, report_mode='idle')

    finished = []
    monkeypatch.setattr(reporter, '_finishReport', lambda now: finished.append(reporter.active[1]))

    class _FinishedProcess(object):
        exitcode = 0
        def join(self, timeout=None):
            pass
        def is_alive(self):
            return False

    reporter.active = (_FinishedProcess(), night, results)
    reporter.stop()

    assert finished == [night]


### Archive and cleanup ###

def test_night_is_archived_by_default(tmp_path):

    config = _config()
    config.timelapse_generate_captured = False
    config.thumb_bin = 1
    config.upload_split = False

    output_dir = str(tmp_path)
    night_dir = mnr.nightDirPath(output_dir, NIGHT)
    os.makedirs(night_dir)
    _makeResults(output_dir, night_dir, config, 0, n_chunks=3)

    state = mnr.generateNightReport(output_dir, NIGHT, config)

    assert 'archive' in state['ok_steps']

    archived_dir = os.path.join(output_dir, mnr.ARCHIVED_DIR_NAME)
    assert os.path.isdir(os.path.join(archived_dir, NIGHT))
    assert os.path.isfile(os.path.join(archived_dir, NIGHT + '_detected.tar.bz2'))
    assert state['upload_files'] == [os.path.join(archived_dir, NIGHT + '_detected.tar.bz2')]


@pytest.mark.parametrize('enabled', [True, False])
def test_cleanup_uses_rms_data_management(tmp_path, monkeypatch, enabled):

    import RMS.DeleteOldObservations

    calls = []

    def _deleteOldObservations(data_dir, captured_dir, archived_dir, config, duration=None):
        calls.append((data_dir, captured_dir, archived_dir, config.data_dir, config.log_dir))
        return True

    monkeypatch.setattr(RMS.DeleteOldObservations, 'deleteOldObservations', _deleteOldObservations)

    config = _config()
    config.monitor_delete_old_data = enabled

    assert mnr.cleanupOldData(str(tmp_path), config)

    if enabled:
        assert calls == [(str(tmp_path), 'CapturedFiles', 'ArchivedFiles', str(tmp_path), 'logs')]
    else:
        assert calls == []


def test_cleanup_scheduling(night_setup, monkeypatch):

    output_dir, config_path, night, night_end, _, results = night_setup

    reporter = mnr.NightReporter(output_dir, config_path, report_mode='idle')

    class _Process(object):
        def __init__(self, alive):
            self.alive = alive
            self.exitcode = 0
        def is_alive(self):
            return self.alive
        def join(self, timeout=None):
            pass

    started = []

    def _startCleanup():
        started.append('cleanup')
        reporter.cleanup_proc = _Process(alive=True)

    def _startReport(night_name, results_dirs):
        started.append('report')
        reporter.active = (_Process(alive=True), night_name, results_dirs)

    monkeypatch.setattr(reporter, '_startCleanup', _startCleanup)
    monkeypatch.setattr(reporter, '_startReport', _startReport)

    now = night_end + datetime.timedelta(hours=1)

    # A cleanup runs at startup, and no report is started while it runs
    reporter.poll(True, now=now)
    reporter.poll(True, now=now)
    assert started == ['cleanup']

    # After the cleanup the due report runs
    reporter.cleanup_proc.alive = False
    reporter.poll(True, now=now)
    assert started == ['cleanup', 'report']

    # After the report finishes, another cleanup runs
    reporter.active[0].alive = False
    monkeypatch.setattr(reporter, '_finishReport', lambda now: None)
    reporter.poll(True, now=now)
    assert started == ['cleanup', 'report', 'cleanup']


def test_saved_pairs_are_dark_and_flat_corrected(tmp_path):

    from RMS.Routines.Image import FlatStruct

    handle = _FakeHandle()

    # The chunk has 1500 counts in the max pixel and 1000 in the average pixel image
    dark = np.full((48, 64), 100, dtype=np.uint16)

    # Flat with the right side half as bright as the left side (vignetting)
    flat_img = np.full((48, 64), 200.0)
    flat_img[:, 32:] = 100.0
    flat_struct = FlatStruct(flat_img)

    saver = ChunkImageSaver(str(tmp_path), _config(), 'input.mkv', dark=dark, flat_struct=flat_struct)
    name = saver(handle, _FakeFF(128))

    ff = FFfile.read(str(tmp_path), name)

    # Dark subtracted, then flat applied: flat_avg*(img - dark)/flat
    flat_avg = flat_struct.flat_avg
    assert ff.avepixel[0, 0] == pytest.approx(flat_avg*900/flat_struct.flat_img[0, 0], abs=1)
    assert ff.avepixel[0, 63] == pytest.approx(flat_avg*900/flat_struct.flat_img[0, 63], abs=1)
    assert ff.maxpixel[0, 63] == pytest.approx(flat_avg*1400/flat_struct.flat_img[0, 63], abs=1)

    # The vignetted side is brightened relative to the other side
    assert ff.avepixel[0, 63] == pytest.approx(2*ff.avepixel[0, 0], abs=2)

    # The images are marked as corrected, so detection doesn't correct them again
    assert ff.dark_applied and ff.flat_applied and ff.calibrated


def test_uncorrected_pairs_are_not_marked_calibrated(tmp_path):

    saver = ChunkImageSaver(str(tmp_path), _config(), 'input.mkv')
    name = saver(_FakeHandle(), _FakeFF(128))

    ff = FFfile.read(str(tmp_path), name)

    assert not (ff.dark_applied or ff.flat_applied or ff.calibrated)
    assert ff.avepixel[0, 0] == 1000


def test_report_uses_calibration_of_processing(tmp_path):

    dirs = []
    for i, (dark, flat) in enumerate([(False, True), (False, False)]):
        d = tmp_path/'r{:d}'.format(i)
        d.mkdir()
        mnr.writeDoneFlag(str(d), {'night': NIGHT, 'dark_applied': dark, 'flat_applied': flat})
        dirs.append(str(d))

    assert mnr.calibrationApplied(dirs) == (False, True)
    assert mnr.calibrationApplied(dirs[1:]) == (False, False)

    # Older results without the info
    old = tmp_path/'old'
    old.mkdir()
    (old/'done.flag').write_text('')
    assert mnr.calibrationApplied([str(old)]) == (False, False)


### Platepar refinement and additional products ###

def test_latest_platepar_never_overwrites_the_given_one(tmp_path):

    config = _config()
    path = mnr.latestPlateparPath(str(tmp_path), config)

    assert os.path.dirname(path) == str(tmp_path)
    assert os.path.basename(path) == 'latest_' + config.platepar_name


def _recalibratedPlatepar(n_stars, tag=0, auto_recalibrated=True):
    """ Recalibrated platepar as stored in the JSON file. The tag is stored as the star intensity, so the
        platepar can be identified.
    """
    from RMS.Formats.Platepar import Platepar
    pp = Platepar()
    pp.auto_recalibrated = auto_recalibrated
    pp.star_list = [[2461034.8, 10.0 + i, 20.0, tag, 170.0, 50.0, 8.0] for i in range(n_stars)]
    return json.loads(pp.jsonStr())


@pytest.mark.parametrize('update', [True, False])
def test_select_best_night_platepar(tmp_path, monkeypatch, update):

    config = _config()
    config.monitor_update_platepar = update

    output_dir = str(tmp_path)
    night_dir = mnr.nightDirPath(output_dir, NIGHT)
    os.makedirs(night_dir)

    recalibrated = {
        'FF_a': _recalibratedPlatepar(150, tag=1),
        'FF_b': _recalibratedPlatepar(197, tag=2),
        'FF_c': _recalibratedPlatepar(197, tag=3),
        # Most stars, but not recalibrated successfully
        'FF_d': _recalibratedPlatepar(250, tag=4, auto_recalibrated=False),
        'FF_e': None,
    }
    with open(os.path.join(night_dir, config.platepars_recalibrated_name), 'w') as f:
        json.dump(recalibrated, f)

    # FF_c fits better than FF_b, which has the same number of stars, and FF_a fits best but has fewer stars
    residuals = {1: 0.03, 2: 0.05, 3: 0.04, 4: 0.01}
    monkeypatch.setattr(mnr, 'plateparResidual', lambda pp: residuals[pp.star_list[0][3]])

    ff_name, platepar, n_stars, residual = mnr.selectBestNightPlatepar(night_dir, config, output_dir=output_dir)

    assert (ff_name, n_stars, residual) == ('FF_c', 197, 0.04)
    assert os.path.isfile(mnr.latestPlateparPath(output_dir, config)) == update


def test_select_best_night_platepar_none_recalibrated(tmp_path):

    config = _config()

    with open(os.path.join(str(tmp_path), config.platepars_recalibrated_name), 'w') as f:
        json.dump({'FF_a': _recalibratedPlatepar(100, auto_recalibrated=False)}, f)

    assert mnr.selectBestNightPlatepar(str(tmp_path), config, output_dir=str(tmp_path)) \
        == (None, None, None, None)
    assert not os.path.exists(mnr.latestPlateparPath(str(tmp_path), config))


def test_observation_summary_counts_image_pairs(tmp_path):

    from RMS.Formats.ObservationSummary import nightSummaryData

    config = _config()
    config.fps = 25.0

    # Three consecutive 128-frame chunks (5.12 s each)
    for i in range(3):
        name = FFfile.constructFFName('XX0001', BEG_TIME + datetime.timedelta(seconds=5.12*i), frame=128*i,
                                      suffix=FFpng.PAIR_MAX_SUFFIX)
        FFpng.writePair(str(tmp_path), name, np.zeros((4, 4), np.uint8), np.zeros((4, 4), np.uint8))

    result = nightSummaryData(config, str(tmp_path), frames_per_file=128)
    capture_duration, fits_count, total_expected = result[0], result[4], result[11]

    assert fits_count == 3
    assert capture_duration == pytest.approx(3*5.12, abs=0.01)
    assert total_expected == 3


def test_additional_products_are_off_by_default():

    config = cr.Config()

    assert config.monitor_update_platepar
    assert not config.monitor_save_ecsv
    assert not config.monitor_shower_association
    assert not config.monitor_fov_kml
    assert not config.monitor_flux
    assert not config.monitor_observation_summary


def test_report_runs_only_enabled_products(tmp_path, monkeypatch):

    import Utils.ShowerAssociation

    config = _config()
    config.timelapse_generate_captured = False
    config.monitor_shower_association = True

    calls = []
    monkeypatch.setattr(Utils.ShowerAssociation, 'showerAssociation',
                        lambda config, ftp_list, **kwargs: calls.append(ftp_list))

    output_dir = str(tmp_path)
    night_dir = mnr.nightDirPath(output_dir, NIGHT)
    os.makedirs(night_dir)
    _makeResults(output_dir, night_dir, config, 0)

    state = mnr.generateNightReport(output_dir, NIGHT, config, archive=False)

    assert 'shower_association' in state['ok_steps']
    assert calls == [[os.path.join(night_dir, 'FTPdetectinfo_{:s}.txt'.format(NIGHT))]]

    # The other products are off
    for step in ['fov_kml', 'flux', 'observation_summary']:
        assert step not in state['ok_steps'] + state['failed_steps']
