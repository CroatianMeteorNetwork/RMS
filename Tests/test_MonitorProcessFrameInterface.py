"""
Tests for RMS.MonitorProcessFrameInterface.

Covers parsing of the start time, the skip gate in processFile, isolation of the config used for reading
the beginning time, the start time precedence in the multicam mode, and the worker exit codes which the
monitor uses to tell successful, skipped and failed files apart.
"""

import configparser
import datetime
import multiprocessing
import os
import shutil
import sys

import pytest

import RMS.MonitorProcessFrameInterface as mon


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


### Dark and flat ###

@pytest.mark.parametrize('dark_path, flat_path', [(None, None), ('bias.png', None), (None, 'flat.png'),
                                                  ('bias.png', 'flat.png')])
def test_dark_and_flat_applied_only_if_given(config_path, tmp_path, monkeypatch, dark_path, flat_path):

    import RMS.ConfigReader as cr

    # The config file enables both, but only what is given to the monitor is applied
    config = cr.parse(config_path)
    config.use_dark = True
    config.use_flat = True
    monkeypatch.setattr(cr, 'parse', lambda path: config)

    seen = {}

    def _fakeDetect(file_path, config, **kwargs):
        seen['use_dark'] = config.use_dark
        seen['use_flat'] = config.use_flat
        seen['dark_file'] = config.dark_file
        seen['flat_file'] = config.flat_file
        return None

    monkeypatch.setattr(mon, 'detectInputType', _fakeDetect)

    mon.processFile(str(tmp_path/'dummy.vid'), config_path, None, str(tmp_path), 128,
                    dark_path=dark_path, flat_path=flat_path)

    assert seen['use_dark'] == (dark_path is not None)
    assert seen['use_flat'] == (flat_path is not None)

    if dark_path is not None:
        assert seen['dark_file'] == os.path.abspath(dark_path)

    if flat_path is not None:
        assert seen['flat_file'] == os.path.abspath(flat_path)
