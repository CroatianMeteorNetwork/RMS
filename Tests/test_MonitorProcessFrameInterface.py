"""
Tests for RMS.MonitorProcessFrameInterface.

Covers parsing of the start time, the skip gate in processFile, isolation of the config used for reading
the beginning time, the start time precedence in the multicam mode, the worker exit codes which the monitor
uses to tell successful, skipped and failed files apart, the setup of processFile (dark and flat, platepar),
saving the star extraction chunks as image pairs, and assigning meteors to them.
"""

import collections
import configparser
import datetime
import multiprocessing
import os
import shutil
import sys
import time

import numpy as np
import pytest

import RMS.ConfigReader as cr
import RMS.MatchedFilterDetection as mfd
import RMS.MonitorProcessFrameInterface as mon
from RMS.Formats import FFfile, FFpng, FTPdetectinfo


REPO_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.config')

START_TIME = datetime.datetime(2025, 12, 25, 10, 20, 0)


@pytest.fixture
def config_path(tmp_path):
    """ Copy the repository config into a temporary directory. """

    path = str(tmp_path/'.config')
    shutil.copy(REPO_CONFIG, path)

    return path


### parseStartTime ###

@pytest.mark.parametrize('start_time_str', [
    '20251225_102000',
    '20251225-102000',
    '2025-12-25 10:20:00',
    '2025-12-25T10:20:00',
    '  2025-12-25 10:20:00  ',
])
def test_parseStartTime_formats(start_time_str):
    assert mon.parseStartTime(start_time_str) == START_TIME


def test_parseStartTime_date_only():
    assert mon.parseStartTime('20251225') == datetime.datetime(2025, 12, 25)


@pytest.mark.parametrize('start_time_str', ['foo', '', '2025-12-25 25:00:00', '20251225_1020'])
def test_parseStartTime_invalid(start_time_str):
    with pytest.raises(ValueError):
        mon.parseStartTime(start_time_str)


### processFile skip gate ###

def test_processFile_skips_file_beginning_before_start_time(config_path, tmp_path, monkeypatch):

    # The file begins inside the 10 minutes before the start time, so it spans the start time
    monkeypatch.setattr(mon, 'readBeginningDatetime',
        lambda *args, **kwargs: START_TIME - datetime.timedelta(minutes=5))

    # The full open must never happen
    def _fail(*args, **kwargs):
        raise AssertionError("The file was opened for processing")

    monkeypatch.setattr(mon, 'detectInputType', _fail)

    with pytest.raises(SystemExit) as exc_info:
        mon.processFile(str(tmp_path/'dummy.vid'), config_path, None, str(tmp_path), 128,
                        start_time=START_TIME)

    assert exc_info.value.code == mon.SKIP_EXIT_CODE


@pytest.mark.parametrize('hours_after_cutoff, skipped', [(-1, False), (1, True)])
def test_processFile_skips_files_of_nights_past_the_cutoff(config_path, tmp_path, monkeypatch,
                                                           hours_after_cutoff, skipped):

    import RMS.MonitorNightReport as mnr

    config = cr.parse(config_path)
    config.monitor_night_cutoff_hours = 6
    monkeypatch.setattr(cr, 'parse', lambda path: config)

    beginning = datetime.datetime(2025, 12, 25, 3, 0, 0)
    night_name, _, night_end = mnr.nightInfo(config, beginning)
    now = night_end + datetime.timedelta(hours=6 + hours_after_cutoff)

    monkeypatch.setattr(mon, 'readBeginningDatetime', lambda *args, **kwargs: beginning)
    monkeypatch.setattr(mon.RmsDateTime, 'utcnow', staticmethod(lambda: now))

    # Files which are not skipped are opened for processing
    def _open(*args, **kwargs):
        raise IOError("opened")

    monkeypatch.setattr(mon, 'detectInputType', _open)

    output_dir = str(tmp_path/'out')
    if not skipped:
        assert mon.processFile(str(tmp_path/'dummy.vid'), config_path, None, output_dir, 128) is False
        return

    with pytest.raises(SystemExit) as exc_info:
        mon.processFile(str(tmp_path/'dummy.vid'), config_path, None, output_dir, 128)

    assert exc_info.value.code == mon.SKIP_EXIT_CODE

    # The file is marked as done, so it is not opened again after a restart, but it is not a result of the
    #   night
    results_dir = mon.resultsDirPath(output_dir, beginning, 'dummy')
    assert mnr.readDoneFlag(results_dir) == {'night': night_name, 'skipped': True,
                                             'input_file': str(tmp_path/'dummy.vid')}
    assert mnr.scanNights(output_dir, config) == {}

    # A file with results (processed again with --force) is processed, its results are not marked skipped
    mnr.writeDoneFlag(results_dir, {'night': night_name})
    assert mon.processFile(str(tmp_path/'dummy.vid'), config_path, None, output_dir, 128) is False
    assert mnr.readDoneFlag(results_dir) == {'night': night_name}


@pytest.mark.parametrize('offset_s', [0, 60])
def test_processFile_processes_file_beginning_at_or_after_start_time(config_path, tmp_path, monkeypatch,
                                                                     offset_s):

    monkeypatch.setattr(mon, 'readBeginningDatetime',
        lambda *args, **kwargs: START_TIME + datetime.timedelta(seconds=offset_s))

    # Record that the full open was reached, then stop processing
    opened = []

    def _fakeDetect(*args, **kwargs):
        opened.append(True)
        return None

    monkeypatch.setattr(mon, 'detectInputType', _fakeDetect)

    result = mon.processFile(str(tmp_path/'dummy.vid'), config_path, None, str(tmp_path), 128,
                             start_time=START_TIME)

    assert result is False
    assert opened == [True]


def test_processFile_without_start_time_does_not_read_beginning(config_path, tmp_path, monkeypatch):

    def _fail(*args, **kwargs):
        raise AssertionError("The beginning time was read without a start time")

    monkeypatch.setattr(mon, 'readBeginningDatetime', _fail)
    monkeypatch.setattr(mon, 'detectInputType', lambda *args, **kwargs: None)

    assert mon.processFile(str(tmp_path/'dummy.vid'), config_path, None, str(tmp_path), 128) is False


def test_processFile_beginning_read_does_not_modify_config(config_path, tmp_path, monkeypatch):
    """ Opening some input types (e.g. FITS directories) changes the config. The changes made when reading
        the beginning time must not leak into the config used for processing.
    """

    class _FakeHandle(object):
        beginning_datetime = START_TIME + datetime.timedelta(minutes=1)

    widths_seen = []

    def _fakeDetect(file_path, config, **kwargs):

        widths_seen.append(config.width)

        # First call is the light open which reads the beginning time - mimic InputTypeImages
        if len(widths_seen) == 1:
            config.width = 4912
            return _FakeHandle()

        # Second call is the full open
        return None

    monkeypatch.setattr(mon, 'detectInputType', _fakeDetect)

    mon.processFile(str(tmp_path/'dummy'), config_path, None, str(tmp_path), 128, start_time=START_TIME)

    assert len(widths_seen) == 2
    assert widths_seen[1] == widths_seen[0]
    assert widths_seen[1] != 4912


### Multicam start time precedence ###

def _multicamParser(global_start=None, cam_start=None):

    cp = configparser.ConfigParser()
    cp.add_section('Global')
    cp.add_section('CAM1')

    if global_start is not None:
        cp.set('Global', 'start_time', global_start)

    if cam_start is not None:
        cp.set('CAM1', 'start_time', cam_start)

    return cp


def test_resolveCameraStartTime_not_set():
    assert mon.resolveCameraStartTime(_multicamParser(), 'CAM1') is None


def test_resolveCameraStartTime_global():
    cp = _multicamParser(global_start='20251225_100000')
    assert mon.resolveCameraStartTime(cp, 'CAM1') == datetime.datetime(2025, 12, 25, 10, 0, 0)


def test_resolveCameraStartTime_camera_overrides_global():
    cp = _multicamParser(global_start='20251225_100000', cam_start='2025-12-25 10:30:00')
    assert mon.resolveCameraStartTime(cp, 'CAM1') == datetime.datetime(2025, 12, 25, 10, 30, 0)


def test_resolveCameraStartTime_cli_overrides_ini():
    cp = _multicamParser(global_start='20251225_100000', cam_start='2025-12-25 10:30:00')
    assert mon.resolveCameraStartTime(cp, 'CAM1', cli_start_time=START_TIME) == START_TIME


def test_resolveCameraStartTime_invalid_names_section():
    cp = _multicamParser(cam_start='foo')
    with pytest.raises(ValueError, match=r'\[CAM1\]'):
        mon.resolveCameraStartTime(cp, 'CAM1')


### Worker exit codes ###

def _runWorker(monkeypatch, fake_process_file):

    monkeypatch.setattr(mon, 'processFile', fake_process_file)

    with pytest.raises(SystemExit) as exc_info:
        mon.processFileWorker('dummy.vid', 'dummy.config', None, 'out', 128)

    return exc_info.value.code


def test_processFileWorker_success_exits_0(monkeypatch):
    assert _runWorker(monkeypatch, lambda *args, **kwargs: True) == 0


def test_processFileWorker_failure_exits_nonzero(monkeypatch):
    assert _runWorker(monkeypatch, lambda *args, **kwargs: False) == 1


def test_processFileWorker_keeps_skip_exit_code(monkeypatch):

    def _skip(*args, **kwargs):
        sys.exit(mon.SKIP_EXIT_CODE)

    assert _runWorker(monkeypatch, _skip) == mon.SKIP_EXIT_CODE


@pytest.mark.parametrize('exit_arg', [(), (0,)])
def test_processFileWorker_early_sys_exit_is_failure(monkeypatch, exit_arg):
    """ Some input types call sys.exit() when they can't open a file, which must count as a failure. """

    def _exit(*args, **kwargs):
        sys.exit(*exit_arg)

    assert _runWorker(monkeypatch, _exit) == 1


def test_processFileWorker_passes_arguments(monkeypatch):

    received = {}

    def _record(*args, **kwargs):
        received['args'] = args
        received['kwargs'] = kwargs
        return True

    monkeypatch.setattr(mon, 'processFile', _record)

    with pytest.raises(SystemExit):
        mon.processFileWorker('a.vid', 'b.config', 'c.cal', 'out', 64, unique_id='a', start_time=START_TIME)

    assert received['args'] == ('a.vid', 'b.config', 'c.cal', 'out', 64)
    assert received['kwargs'] == {'unique_id': 'a', 'start_time': START_TIME}


def test_processFile_error_before_logger_init_returns_false(config_path, tmp_path, monkeypatch):
    """ An exception before the per-file logger is set up must be logged and reported as a failure. """

    def _raise(*args, **kwargs):
        raise IOError("corrupt file")

    monkeypatch.setattr(mon, 'detectInputType', _raise)

    assert mon.processFile(str(tmp_path/'dummy.vid'), config_path, None, str(tmp_path), 128) is False


def test_failed_worker_process_exit_code(config_path, tmp_path):
    """ End to end: a worker process for a file which can't be opened exits with a non-zero code. """

    # Not a valid video, so the frame interface can't open it
    bad_file = tmp_path/'20251225_102000.mkv'
    bad_file.write_bytes(b'not a video')

    proc = multiprocessing.Process(target=mon.processFileWorker,
        args=(str(bad_file), config_path, None, str(tmp_path/'out'), 128))
    proc.start()
    proc.join(timeout=120)

    assert proc.exitcode not in (0, mon.SKIP_EXIT_CODE, None)


def test_stuck_worker_is_stopped():
    """ A worker which runs longer than the time limit is stopped with a failure exit code, so its file is
        retried. Workers within the limit, or without a limit, keep running.
    """

    proc = multiprocessing.Process(target=time.sleep, args=(60,))
    proc.start()
    proc.start_time = time.monotonic()

    mon.stopStuckWorker(proc, 'file', 0)
    mon.stopStuckWorker(proc, 'file', 100)
    assert proc.is_alive()

    proc.start_time -= 101
    mon.stopStuckWorker(proc, 'file', 100)
    assert (not proc.is_alive()) and (proc.exitcode < 0)


### Processing a file ###

class _ProcessHandle(object):
    """ Minimal image handle which lets processFile run up to the detection. """

    input_type = 'video'
    beginning_datetime = datetime.datetime(2025, 12, 25, 3, 0, 0)
    byteswap = False

    def __init__(self, dir_path):
        self.dir_path = dir_path

    class ff(object):
        dtype = np.uint8


@pytest.fixture
def process_file(config_path, tmp_path, monkeypatch):
    """ Run processFile up to the detection, which records the config it gets. Return the run function and the
        recorded config.
    """


    config = cr.parse(config_path)
    config.monitor_save_images = False
    monkeypatch.setattr(cr, 'parse', lambda path: config)

    monkeypatch.setattr(mon, 'detectInputType', lambda *args, **kwargs: _ProcessHandle(str(tmp_path)))

    seen = {}

    def _detect(img_handle, config, **kwargs):
        seen['config'] = config
        seen['chunk_callback'] = kwargs.get('chunk_callback')
        raise RuntimeError("stop after the setup")

    monkeypatch.setattr(mon, 'detectStarsAndMeteorsFrameInterface', _detect)

    given_platepar = tmp_path/'given.cal'
    given_platepar.write_text('given')

    def _run(calibration=(None, None, None), **kwargs):
        monkeypatch.setattr(mon, 'loadImageCalibration', lambda *args, **kw: calibration)
        return mon.processFile(str(tmp_path/'dummy.vid'), config_path, str(given_platepar),
                               str(tmp_path/'out'), 128, **kwargs)

    return _run, config, seen


@pytest.mark.parametrize('input_type, saved', [('video', True), ('ff', False)])
def test_chunk_images_are_saved_only_for_non_ff_inputs(process_file, monkeypatch, input_type, saved):

    run, config, seen = process_file
    config.monitor_save_images = True

    # FF inputs already are FF files, and their handles have no frame chunks
    monkeypatch.setattr(_ProcessHandle, 'input_type', input_type)
    run()

    assert isinstance(seen['chunk_callback'], mon.ChunkImageSaver) == saved


@pytest.mark.parametrize('dark_given, flat_given', [(False, False), (True, False), (False, True),
                                                    (True, True)])
def test_dark_and_flat_applied_only_if_given(process_file, dark_given, flat_given):

    run, config, seen = process_file

    # The config file enables both, but only what is given to the monitor is applied
    config.use_dark = True
    config.use_flat = True

    dark = np.zeros((4, 4), np.uint8) if dark_given else None
    flat = object() if flat_given else None

    run(calibration=(None, dark, flat), dark_path=('bias.png' if dark_given else None),
        flat_path=('flat.png' if flat_given else None))

    assert seen['config'].use_dark == dark_given
    assert seen['config'].use_flat == flat_given

    if dark_given:
        assert seen['config'].dark_file == os.path.abspath('bias.png')


def test_given_dark_which_cannot_be_loaded_fails(process_file):

    run, config, seen = process_file

    assert run(calibration=(None, None, None), dark_path='missing.png') is False
    assert 'config' not in seen


def test_processFile_uses_latest_platepar_unless_given_is_newer(process_file, tmp_path):

    from RMS.MonitorNightReport import latestPlateparPath

    run, config, seen = process_file

    output_dir = tmp_path/'out'
    output_dir.mkdir()
    latest_path = latestPlateparPath(str(output_dir), config)
    results_platepar = os.path.join(str(output_dir), '2025', '202512', '20251225', 'dummy',
                                    config.platepar_name)

    def _usedPlatepar():
        run()
        with open(results_platepar) as f:
            return f.read()

    with open(latest_path, 'w') as f:
        f.write('latest')

    # The best platepar of a previous night is newer than the given one
    os.utime(str(tmp_path/'given.cal'), (0, 0))
    assert _usedPlatepar() == 'latest'

    config.monitor_update_platepar = False
    assert _usedPlatepar() == 'given'

    # A given platepar which was fitted again is newer
    config.monitor_update_platepar = True
    os.utime(latest_path, (0, 0))
    os.utime(str(tmp_path/'given.cal'), None)
    assert _usedPlatepar() == 'given'


@pytest.mark.parametrize('mf_enable, mf_fails', [(False, False), (True, False), (True, True)])
def test_matched_filter_pass(config_path, tmp_path, monkeypatch, mf_enable, mf_fails):
    """ The matched filter runs after the normal detection when it is enabled, saves its results into their
        own directory, and its failure doesn't fail the file.
    """

    # The config of the test (with or without the matched filter), a synthetic input and no calibration
    config = cr.parse(config_path)
    config.monitor_save_images = False
    config.mf_enable = mf_enable
    monkeypatch.setattr(cr, 'parse', lambda path: config)
    monkeypatch.setattr(mon, 'detectInputType', lambda *args, **kwargs: _ProcessHandle(str(tmp_path)))
    monkeypatch.setattr(mon, 'loadImageCalibration', lambda *args, **kwargs: (None, None, None))

    # The normal detection is replaced by one which returns enough stars for the detection (the matched filter
    #   is skipped without them, like the normal detection) and no detections, and its results are not saved for the detection (the matched filter is skipped without them, like the normal detection)
    stars = [['FF_XX0001_20260101_000000_000_0000000.fits', [(10.0, 10.0, 100.0)]*config.ff_min_stars]]
    monkeypatch.setattr(mon, 'detectStarsAndMeteorsFrameInterface', lambda *args, **kwargs: (stars, []))
    monkeypatch.setattr(mon, 'saveResultsFrameInterface', lambda *args, **kwargs: None)

    # The matched filter is replaced by functions which record their calls (and the output directory of the
    #   results), and the detection fails if requested. The monitor imports them from the module when it runs
    #   them, so they are replaced in the module
    calls = []

    def _detect(img_handle, config, **kwargs):
        calls.append('detect')
        if mf_fails:
            raise RuntimeError("matched filter failure")
        return [], None

    def _save(detections, star_list, img_handle, config, output_dir, **kwargs):
        calls.append(output_dir)
        return os.path.join(output_dir, 'FTPdetectinfo_mf.txt')

    monkeypatch.setattr(mfd, 'detectMatchedFilter', _detect)
    monkeypatch.setattr(mfd, 'saveMatchedFilterResults', _save)

    given_platepar = tmp_path/'given.cal'
    given_platepar.write_text('given')
    output_dir = tmp_path/'out'

    # The file is processed successfully in every case, with the normal results
    assert mon.processFile(str(tmp_path/'dummy.vid'), config_path, str(given_platepar), str(output_dir), 128)

    results_dir = os.path.join(str(output_dir), '2025', '202512', '20251225', 'dummy')
    assert mon.hasResults(results_dir)

    # Disabled: not run. Failed: nothing saved. Otherwise: saved into the matched-filter directory of the
    #   results
    if not mf_enable:
        assert calls == []
    elif mf_fails:
        assert calls == ['detect']
    else:
        assert calls == ['detect', os.path.join(results_dir, mfd.MATCHED_FILTER_DIR)]


def test_walkInput_skips_monitor_output(tmp_path):

    config = cr.Config()
    config.captured_dir = 'Nights'
    for dir_name in ['2025', 'Nights', config.archived_dir, 'logs', 'CapturedFiles', 'cam/Nights']:
        (tmp_path/'videos'/dir_name).mkdir(parents=True)

    def _walked(input_dir, output_dir):
        return sorted(os.path.relpath(root, input_dir)
                      for root, _, _ in mon.walkInput(input_dir, output_dir, config))

    # The output is in the input directory: its night directory comes from the config, and only the
    #   directories of the output are skipped
    input_dir = str(tmp_path/'videos')
    assert _walked(input_dir, input_dir) == ['.', '2025', 'CapturedFiles', 'cam', 'cam/Nights']

    # A separate output directory: nothing in the input is skipped
    all_dirs = ['.', '2025', 'Nights', config.archived_dir, 'logs', 'CapturedFiles', 'cam', 'cam/Nights']
    assert _walked(input_dir, str(tmp_path/'output')) == sorted(all_dirs)


def test_input_file_is_staged_locally(tmp_path, monkeypatch):

    input_dir = tmp_path/'nfs'
    input_dir.mkdir()
    staging_dir = str(tmp_path/'staging')

    # Video files are timed by their names, which the copy keeps
    for name in ['dump_6AC5999C_02F.vid', 'CAWE01_20261008_010203_123456_video.mkv', '20261008_010203.mp4']:
        file_path = str(input_dir/name)
        with open(file_path, 'wb') as f:
            f.write(b'0'*1000)

        # Disabled: the file is processed where it is
        assert mon.stageInputFile(file_path, '') == file_path

        # Enabled: a copy with the same name, in a directory of the worker
        staged = mon.stageInputFile(file_path, staging_dir)
        assert staged == os.path.join(staging_dir, mon.STAGED_DIR_NAME.format(os.getpid()), name)
        assert open(staged, 'rb').read() == open(file_path, 'rb').read()

    # FF files and directories are processed where they are
    ff_path = str(input_dir/'FF_XX0001_20251225_031500_123_0000000.fits')
    open(ff_path, 'w').close()
    assert mon.stageInputFile(ff_path, staging_dir) == ff_path
    assert mon.stageInputFile(str(input_dir), staging_dir) == str(input_dir)

    # A copy which fails (e.g. the disk filled up meanwhile) is processed where it is, nothing is left behind
    def _fail(src, dst):
        open(dst, 'w').close()
        raise OSError(28, "No space left on device")

    shutil.rmtree(staging_dir)
    monkeypatch.setattr(mon.shutil, 'copyfile', _fail)
    assert mon.stageInputFile(file_path, staging_dir) == file_path
    assert os.listdir(staging_dir) == []


def test_input_file_is_not_staged_without_space(tmp_path, monkeypatch):

    file_path = str(tmp_path/'dump_6AC5999C_02F.vid')
    with open(file_path, 'wb') as f:
        f.write(b'0'*1000)

    monkeypatch.setattr(mon.shutil, 'disk_usage', lambda path: collections.namedtuple('usage', 'free')(10))
    assert mon.stageInputFile(file_path, str(tmp_path/'staging')) == file_path


def test_staged_copies_are_removed(tmp_path):

    staging_dir = tmp_path
    dead_pid = 2**22 + 12345
    names = [mon.STAGED_DIR_NAME.format(os.getpid()), mon.STAGED_DIR_NAME.format(dead_pid),
             mon.STAGED_DIR_NAME.format(1)]
    for name in names:
        (staging_dir/name).mkdir()
        (staging_dir/name/'a.vid').write_text('0')
    (staging_dir/'other.txt').write_text('0')

    # Without a PID, the copies of workers which are not running are removed; the copies of running processes
    #   of other users (PID 1, which can't be signalled) and other files are kept
    mon.removeStagedFiles(str(staging_dir))
    assert sorted(os.listdir(str(staging_dir))) == sorted([names[0], names[2], 'other.txt'])

    # The copies of a given worker
    mon.removeStagedFiles(str(staging_dir), pid=os.getpid())
    assert sorted(os.listdir(str(staging_dir))) == sorted([names[2], 'other.txt'])

    # Disabled staging
    mon.removeStagedFiles('')


def test_processFile_reads_the_staged_copy_and_deletes_it(config_path, tmp_path, monkeypatch):

    config = cr.parse(config_path)
    config.monitor_staging_dir = str(tmp_path/'staging')
    monkeypatch.setattr(cr, 'parse', lambda path: config)

    input_dir = tmp_path/'nfs'
    input_dir.mkdir()
    file_path = str(input_dir/'dump_6AC5999C_02F.vid')
    with open(file_path, 'wb') as f:
        f.write(b'0'*1000)

    # Record which file is opened, and the staging directory content at that time
    opened = []
    def _open(path, *args, **kwargs):
        opened.append((path, os.listdir(config.monitor_staging_dir)))
        raise IOError("stop")

    monkeypatch.setattr(mon, 'detectInputType', _open)

    assert mon.processFile(file_path, config_path, None, str(tmp_path/'out'), 128) is False

    path, staged = opened[0]
    assert path == os.path.join(config.monitor_staging_dir, mon.STAGED_DIR_NAME.format(os.getpid()),
                                'dump_6AC5999C_02F.vid')
    assert staged == [mon.STAGED_DIR_NAME.format(os.getpid())]

    # The copy is deleted, the input file is kept
    assert os.listdir(config.monitor_staging_dir) == []
    assert os.path.isfile(file_path)


def test_input_extension():

    assert mon.inputExtension('vid') == '.vid'
    assert mon.inputExtension('MKV') == '.mkv'
    assert mon.inputExtension('ff') is None
    assert mon.inputExtension('fitsdirs') is None

    # Other types are the extension itself, as in matchesFileType
    assert mon.inputExtension('H264') == '.h264'
    assert mon.matchesFileType('a.h264', 'H264')


def test_unique_ids_keep_files_in_subdirectories_apart():

    # Files in the input directory keep their name
    assert mon.uniqueId('22-00-00.mkv') == '22-00-00'

    paths = [os.path.join('2026-10-07', '22-00-00.mkv'), os.path.join('2026-10-08', '22-00-00.mkv'),
             '2026-10-07_22-00-00.mkv', os.path.join('a', 'b_c.mkv'), os.path.join('a_b', 'c.mkv')]
    ids = [mon.uniqueId(path) for path in paths]
    assert len(set(ids)) == len(ids)
    assert ids[0].startswith('22-00-00_')


def test_output_dir_inside_another_monitor_output_is_found(tmp_path):

    outer = tmp_path/'outer'
    (outer/'inner').mkdir(parents=True)
    assert mon.enclosingMonitorOutput(str(outer/'inner')) is None

    (outer/mon.MONITOR_LOCK_FILE_NAME).write_text('1\n')
    assert mon.enclosingMonitorOutput(str(outer/'inner')) == os.path.realpath(str(outer))
    assert mon.enclosingMonitorOutput(str(outer)) is None


def test_walkInput_skips_other_monitor_outputs(tmp_path):

    config = cr.Config()
    (tmp_path/'videos'/'other'/'2025').mkdir(parents=True)
    (tmp_path/'videos'/'other'/mon.MONITOR_LOCK_FILE_NAME).write_text('1\n')

    input_dir = str(tmp_path/'videos')
    walked = [os.path.relpath(root, input_dir) for root, _, _ in mon.walkInput(input_dir, input_dir, config)]
    assert walked == ['.']


def test_mask_is_found_in_the_input_dir(tmp_path):

    config = cr.Config()
    input_dir = str(tmp_path)

    assert mon.findInputMask(input_dir, config) is None

    open(os.path.join(input_dir, config.mask_file), 'w').close()
    assert mon.findInputMask(input_dir, config) == os.path.join(input_dir, config.mask_file)

    # A given mask is used instead
    assert mon.findInputMask(input_dir, config, 'other.bmp') == os.path.abspath('other.bmp')


### Saving the chunk images ###

class _FakeFF(object):
    def __init__(self, nframes, level=1000):
        self.nframes = nframes
        self.maxpixel = np.full((48, 64), level + 500, dtype=np.uint16)
        self.avepixel = np.full((48, 64), level, dtype=np.uint16)
        self.successful = True


class _ChunkHandle(object):
    """ Image handle with 25 fps frames, 128 frames per chunk and 330 frames in total. """

    input_type = 'video'

    def __init__(self, fps=25.0, total_frames=330, chunk_frames=128):
        self.fps = fps
        self.total_frames = total_frames
        self.chunk_frames = chunk_frames
        self.total_fr_chunks = total_frames//chunk_frames
        self.current_frame_chunk = 0
        self.beginning_datetime = datetime.datetime(2025, 12, 25, 3, 0, 0, 12345)
        self.cache = {}
        self.loaded = []

    def currentFrameTime(self, frame_no=None, dt_obj=False):
        return self.beginning_datetime + datetime.timedelta(seconds=frame_no/self.fps)

    def loadChunk(self, first_frame=None, read_nframes=None):
        self.loaded.append((first_frame, read_nframes))
        return _FakeFF(read_nframes)


@pytest.fixture
def config():
    config = cr.parse(REPO_CONFIG)
    config.stationID = 'XX0001'
    return config


def _chunkName(handle, frame):
    return FFpng.pairNames(FFfile.constructFFName('XX0001', handle.currentFrameTime(frame), frame=frame,
                                                  ext=None))[0]


def test_chunk_image_saver(tmp_path, config):

    handle = _ChunkHandle()
    saver = mon.ChunkImageSaver(str(tmp_path), config, 'input.mkv')

    names = []
    for chunk in range(handle.total_fr_chunks):
        handle.current_frame_chunk = chunk
        names.append(saver(handle, _FakeFF(128)))

    # The trailing 74 frames (256-329, more than half a chunk) are saved as well, after removing a possibly
    #   calibrated cached chunk
    handle.cache[mon.getCacheID(256, 74)] = 'calibrated'
    names.append(saver.saveTrailing(handle))

    assert mon.getCacheID(256, 74) not in handle.cache
    assert handle.loaded == [(256, 74)]

    assert [c[:2] for c in saver.chunk_images] == [(0, 128), (128, 128), (256, 74)]
    assert names == [c[2] for c in saver.chunk_images]
    assert names[1] == _chunkName(handle, 128)

    ff = FFfile.read(str(tmp_path), names[2])
    assert (ff.nframes, ff.first) == (74, 256)
    assert ff.maxpixel.dtype == np.uint16


def test_short_trailing_frames_are_not_saved(tmp_path, config):

    # 30 frames after the last chunk, less than half a chunk
    handle = _ChunkHandle(total_frames=286)
    saver = mon.ChunkImageSaver(str(tmp_path), config, 'input.mkv')

    assert saver.saveTrailing(handle) is None
    assert handle.loaded == []


def test_saved_pairs_are_dark_and_flat_corrected(tmp_path, config):

    from RMS.Routines.Image import FlatStruct

    # The chunk has 1500 counts in the max pixel and 1000 in the average pixel image
    dark = np.full((48, 64), 100, dtype=np.uint16)

    # Flat with the right side half as bright as the left side (vignetting)
    flat_img = np.full((48, 64), 200.0)
    flat_img[:, 32:] = 100.0
    flat_struct = FlatStruct(flat_img)

    saver = mon.ChunkImageSaver(str(tmp_path), config, 'input.mkv', dark=dark, flat_struct=flat_struct)
    name = saver(_ChunkHandle(), _FakeFF(128))

    ff = FFfile.read(str(tmp_path), name)

    # Dark subtracted, then flat applied, so the vignetted side is brightened relative to the other side
    assert ff.avepixel[0, 0] == pytest.approx(flat_struct.flat_avg*900/200, abs=1)
    assert ff.maxpixel[0, 63] == pytest.approx(flat_struct.flat_avg*1400/100, abs=1)
    assert ff.avepixel[0, 63] == pytest.approx(2*ff.avepixel[0, 0], abs=2)


def test_saver_bins_the_calibration_for_binned_chunks(tmp_path, config):

    from RMS.Routines.Image import FlatStruct

    config.detection_binning_factor = 2

    # Full resolution calibration, twice the size of the binned chunks
    dark = np.full((96, 128), 100, dtype=np.uint16)
    flat_struct = FlatStruct(np.full((96, 128), 200.0))

    saver = mon.ChunkImageSaver(str(tmp_path), config, 'input.mkv', dark=dark, flat_struct=flat_struct)
    name = saver(_ChunkHandle(), _FakeFF(128))

    # The binned dark was applied, and the given flat is unchanged
    assert FFpng.readMeta(os.path.join(str(tmp_path), name))['binning'] == '2'
    assert int(np.max(FFfile.read(str(tmp_path), name).avepixel)) == 900
    assert flat_struct.flat_img.shape == (96, 128)


def test_saver_scales_summed_binning_to_the_data_levels(tmp_path, config):

    # 8-bit data summed over 2x2 pixels: the 1500 and 1000 counts of the chunk are 4 times the levels
    config.detection_binning_factor = 2
    config.detection_binning_method = 'sum'
    config.bit_depth = 8

    saver = mon.ChunkImageSaver(str(tmp_path), config, 'input.mkv')
    ff = FFfile.read(str(tmp_path), saver(_ChunkHandle(), _FakeFF(128)))

    assert ff.maxpixel.dtype == np.uint8
    assert (int(np.max(ff.maxpixel)), int(np.max(ff.avepixel))) == (255, 250)


### Assigning meteors to the chunk images ###

def _meteor(frames):
    """ Meteor with picks on the given frames: (rho, theta, centroids). """

    # Columns: frame, x, y, level, background, SNR, saturated pixel count
    centroids = np.array([[f, 10.0 + i, 20.0 + i, 100, 5, 8.0, 0] for i, f in enumerate(frames)],
                         dtype=np.float64)

    return 10.0, 45.0, centroids


def test_chunkForFrame():

    from RMS.DetectStarsAndMeteors import chunkForFrame

    chunks = [(0, 128, 'a'), (128, 128, 'b'), (256, 44, 'c')]

    assert chunkForFrame(chunks, 0) == 'a'
    assert chunkForFrame(chunks, 127.9) == 'a'
    assert chunkForFrame(chunks, 128) == 'b'
    assert chunkForFrame(chunks, 299) == 'c'

    # After the last chunk, or in an unsaved chunk, the last preceding chunk is used
    assert chunkForFrame(chunks, 320) == 'c'
    assert chunkForFrame([(0, 128, 'a'), (256, 128, 'c')], 200) == 'a'

    assert chunkForFrame([(128, 128, 'b')], 10) is None


def test_meteor_times_round_trip_through_chunk_names(tmp_path, config):

    from RMS.DetectStarsAndMeteors import saveResultsFrameInterface

    handle = _ChunkHandle(fps=25.0)
    chunk_images = [(0, 128, _chunkName(handle, 0)), (128, 128, _chunkName(handle, 128))]

    # Two meteors in the second chunk (one with a rolling shutter fraction), one in the first
    pick_frames = [[130.0, 131.5, 133.0], [140.0, 141.0], [5.0, 6.0, 7.0]]
    meteors = [_meteor(frames) for frames in pick_frames]

    _, _, ftp_name = saveResultsFrameInterface([], meteors, handle, config, chunk_frames=128,
                                               output_dir=str(tmp_path), chunk_images=chunk_images)

    entries = FTPdetectinfo.readFTPdetectinfo(str(tmp_path), ftp_name)

    # Meteors are numbered per chunk
    assert [(e[0], e[2]) for e in entries] == [(chunk_images[1][2], 1), (chunk_images[1][2], 2),
                                               (chunk_images[0][2], 1)]

    for entry, frames in zip(entries, pick_frames):

        ff_name, meteor_fps, meas = entry[0], entry[4], entry[11]
        ref_time = FFfile.filenameToDatetime(ff_name)

        assert meteor_fps == pytest.approx(25.0)

        for line, frame in zip(meas, frames):
            pick_time = ref_time + datetime.timedelta(seconds=line[1]/meteor_fps)
            assert abs((pick_time - handle.currentFrameTime(frame)).total_seconds()) < 0.001


def test_meteor_names_unchanged_without_chunk_images(tmp_path, config):

    from RMS.DetectStarsAndMeteors import saveResultsFrameInterface

    handle = _ChunkHandle(fps=25.0)

    _, _, ftp_name = saveResultsFrameInterface([], [_meteor([130.0, 131.0])], handle, config,
                                               chunk_frames=128, output_dir=str(tmp_path))

    entry = FTPdetectinfo.readFTPdetectinfo(str(tmp_path), ftp_name)[0]

    assert entry[0] == FFfile.constructFFName('XX0001', handle.currentFrameTime(130))

    # The frames are relative to the time in the name (rounded down to the millisecond) at the measured
    #   frame rate, so the time of every pick is name time + frame/fps
    name_time = FFfile.filenameToDatetime(entry[0])
    for frame, pick_frame in zip([row[1] for row in entry[11]], [130, 131]):
        pick_time = name_time + datetime.timedelta(seconds=frame/25.0)
        assert abs((pick_time - handle.currentFrameTime(pick_frame)).total_seconds()) < 0.001


### Worker lifetime ###

def _workerWithMonitorWatch(pid_queue):
    mon.exitWithMonitor(interval=0.2)
    pid_queue.put(os.getpid())
    time.sleep(60)


def _monitorWithWorker(pid_queue):
    worker = multiprocessing.Process(target=_workerWithMonitorWatch, args=(pid_queue,))
    worker.start()
    time.sleep(60)


@pytest.mark.parametrize('start_method', ['fork', 'forkserver'])
def test_worker_ends_when_the_monitor_is_killed(start_method):

    ctx = multiprocessing.get_context(start_method)
    pid_queue = ctx.Queue()

    monitor = ctx.Process(target=_monitorWithWorker, args=(pid_queue,))
    monitor.start()
    worker_pid = pid_queue.get(timeout=60)

    monitor.kill()
    monitor.join()

    t0 = time.time()
    while time.time() - t0 < 10:
        try:
            os.kill(worker_pid, 0)
            with open('/proc/{:d}/stat'.format(worker_pid)) as f:
                if f.read().split()[2] == 'Z':
                    break
        except (OSError, IOError):
            break
        time.sleep(0.1)

    assert time.time() - t0 < 5


### Failed files ###

def test_killed_workers_are_given_up_later(tmp_path):

    output_dir = str(tmp_path)
    failed_files, _ = mon.loadFailedFiles(output_dir)

    # A worker killed from outside (e.g. out of memory) is retried until it was killed MAX_KILLS times
    for _ in range(mon.MAX_KILLS - 1):
        assert not mon.recordFailure(output_dir, failed_files, 'big', -9, 300, killed=True)
    assert mon.recordFailure(output_dir, failed_files, 'big', -9, 300, killed=True)
    assert mon.loadFailedFiles(output_dir)[1] == {'big'}


def test_killed_from_outside():

    class _Proc(object):
        exitcode = -9

    proc = _Proc()
    assert mon.killedFromOutside(proc)

    # A worker stopped by the monitor (time limit) counts as a normal failure
    proc.stopped_by_monitor = True
    assert not mon.killedFromOutside(proc)

    proc = _Proc()
    proc.exitcode = 1
    assert not mon.killedFromOutside(proc)


def test_calibration_size_is_checked():

    class _Handle(object):
        nrows, ncols = 540, 960

    # Detection binning by 2: the handle has the binned size, the calibration images the full size
    assert mon.calibrationMatchesFrames(np.zeros((1080, 1920)), _Handle(), 2)
    assert not mon.calibrationMatchesFrames(np.zeros((512, 512)), _Handle(), 2)
    assert mon.calibrationMatchesFrames(None, _Handle(), 2)


def test_failed_files_are_given_up_and_remembered(tmp_path):

    output_dir = str(tmp_path)
    failed_files, given_up = mon.loadFailedFiles(output_dir)
    assert (failed_files, given_up) == ({}, set())

    # The first failure is retried, the second one gives the file up
    assert not mon.recordFailure(output_dir, failed_files, 'bad', 1, 300)
    assert mon.loadFailedFiles(output_dir)[1] == set()

    assert mon.recordFailure(output_dir, failed_files, 'bad', 1, 300)
    mon.recordFailure(output_dir, failed_files, 'other', 1, 300)

    # After a restart, the given up file is still skipped and the other one keeps its failure count
    failed_files, given_up = mon.loadFailedFiles(output_dir)
    assert given_up == {'bad'}
    assert failed_files['other']['count'] == 1

    # Retrying the failed files clears the record
    assert mon.loadFailedFiles(output_dir, retry_failed=True) == ({}, set())
    assert mon.loadFailedFiles(output_dir) == ({}, set())


def test_processing_pauses_while_the_output_disk_is_full(monkeypatch, tmp_path):

    class _Reporter(object):
        cleanup_due = False

        class config(object):
            extra_space_gb = 5

    reporter = _Reporter()
    guard = mon.FreeSpaceGuard(str(tmp_path), reporter)

    # Full disk: paused, and a cleanup is requested once, not on every check
    monkeypatch.setattr(mon, 'availableSpace', lambda path: 1*1024**3)
    assert not guard.ok()
    assert reporter.cleanup_due
    reporter.cleanup_due = False
    assert not guard.ok()
    assert not reporter.cleanup_due

    # Space again: resumed
    monkeypatch.setattr(mon, 'availableSpace', lambda path: 10*1024**3)
    assert guard.ok() and not guard.paused


def test_processFileWorker_keeps_changed_exit_code(monkeypatch):

    # A file which was still being written is processed again, not counted as failed
    def _changed(*args, **kwargs):
        sys.exit(mon.CHANGED_EXIT_CODE)

    assert _runWorker(monkeypatch, _changed) == mon.CHANGED_EXIT_CODE


def test_missing_camera_files_are_found(tmp_path):

    config_path = str(tmp_path/'a.config')
    open(config_path, 'w').close()

    assert mon.missingCameraFiles(config_path, config_path) == []
    assert mon.missingCameraFiles(config_path, str(tmp_path/'pp.cal'), dark_path=str(tmp_path/'dark.png')) == \
        [str(tmp_path/'pp.cal'), str(tmp_path/'dark.png')]



@pytest.mark.parametrize('enough_stars', [True, False])
def test_matched_filter_replaces_detection(config_path, tmp_path, monkeypatch, enough_stars):
    """ With mf_replace_detection, the normal detection is not run, the stars are extracted, and the
        detections of the matched filter are the results of the file. Without enough stars (clouds), nothing is
        detected, as in the normal detection. Results of an earlier separate matched-filter pass are removed.
    """

    # The config of the test with the matched filter replacing the normal detection, a synthetic input and no
    #   calibration
    config = cr.parse(config_path)
    config.monitor_save_images = False
    config.mf_enable = True
    config.mf_replace_detection = True
    monkeypatch.setattr(cr, 'parse', lambda path: config)
    monkeypatch.setattr(mon, 'detectInputType', lambda *args, **kwargs: _ProcessHandle(str(tmp_path)))
    monkeypatch.setattr(mon, 'loadImageCalibration', lambda *args, **kwargs: (None, None, None))

    # The normal detection, the star extraction, the matched filter and the saving of the results are replaced
    #   by functions which record their calls. The star extraction gives enough stars, or one star too few
    calls = {}

    def _normal(*args, **kwargs):
        calls['normal'] = True
        return [], []

    def _stars(*args, **kwargs):
        calls['stars'] = True
        n_stars = config.ff_min_stars if enough_stars else config.ff_min_stars - 1
        return [['FF_XX0001_20251225_030000_000_0000000.fits', [(10.0, 20.0, 100, 50, 2.0, 7, 9.0, 0)]*n_stars]]

    # One detection of the matched filter: [rho, theta, centroids]
    detections = [[1.0, 2.0, np.array([[1.0, 10.0, 20.0, 100, 50, 5.0, 0]])]]

    def _mf(img_handle, config, **kwargs):
        calls['mf'] = True
        return detections, None

    def _save(star_list, meteor_list, img_handle, config, **kwargs):
        calls['saved'] = (star_list, meteor_list, kwargs.get('output_suffix', ''))

    monkeypatch.setattr(mon, 'detectStarsAndMeteorsFrameInterface', _normal)
    monkeypatch.setattr(mon, 'extractStarsFrameInterface', _stars)
    monkeypatch.setattr(mfd, 'detectMatchedFilter', _mf)
    monkeypatch.setattr(mon, 'saveResultsFrameInterface', _save)

    given_platepar = tmp_path/'given.cal'
    given_platepar.write_text('given')

    # Results of an earlier processing with the separate matched-filter pass
    results_dir = os.path.join(str(tmp_path/'out'), '2025', '202512', '20251225', 'dummy')
    os.makedirs(os.path.join(results_dir, mfd.MATCHED_FILTER_DIR))

    assert mon.processFile(str(tmp_path/'dummy.vid'), config_path, str(given_platepar), str(tmp_path/'out'), 128)

    # The normal detection is not run, the stars are extracted, and the matched filter runs only with enough
    #   stars
    assert 'normal' not in calls
    assert calls['stars'] and (calls.get('mf', False) == enough_stars)

    # The results of the file (no suffix) are the stars and the detections of the matched filter
    star_list, meteor_list, suffix = calls['saved']
    assert len(star_list) == 1 and suffix == ''
    if enough_stars:
        assert meteor_list is detections
    else:
        assert meteor_list == []

    # The summary of the matched filter is saved with the results, and there is no separate directory
    assert os.path.isfile(os.path.join(results_dir, mfd.DONE_NAME))
    assert not os.path.isdir(os.path.join(results_dir, mfd.MATCHED_FILTER_DIR))
