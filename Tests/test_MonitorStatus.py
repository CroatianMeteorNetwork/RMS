"""
Tests of the live status of the monitor (RMS.MonitorStatus): the recording, the processing, the sky and the night,
the calibration and the faults in the status of synthetic results, the status file itself, and the status written
by the monitor loop.
"""

import datetime
import json
import os
import shutil
import time

import numpy as np
import pytest

import RMS.MonitorNightReport as mnr
import RMS.MonitorProcessFrameInterface as mon
import RMS.MonitorStatus as mst
from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, xyToRaDecPP
from RMS.Formats import CALSTARS, FFfile
from RMS.Formats.Platepar import Platepar
from RMS.Routines.SkyQuality import CONDITION_COLUMNS
from Tests.test_MonitorNightReport import BEG_TIME, NIGHT, REPO_CONFIG, TEMPLATE_PLATEPAR, _config, _makeResults


# The time of the status, in the night of the synthetic results
NOW = BEG_TIME + datetime.timedelta(minutes=30)


class _Reporter(object):
    """ Night reporter of the monitor, as far as the status uses it. """
    report_mode = 'sunrise'
    active = None
    last_cleanup = None


class _SpaceGuard(object):
    paused = False


@pytest.fixture
def now(monkeypatch):
    """ The status is made at NOW. """

    monkeypatch.setattr(mst.RmsDateTime, 'utcnow', staticmethod(lambda: NOW))
    return NOW


def _state(**kwargs):
    """ State of the monitor for the status, see MonitorStatus.build. """

    state = {'stopped': False, 'input_readable': True, 'paused_for_space': False, 'workers_busy': 0,
             'in_progress': [], 'queued': [], 'settling': [], 'failed_files': {}, 'max_failures': 2,
             'max_kills': 5, 'reporter': _Reporter()}
    state.update(kwargs)
    return state


def _matchedFilterConfig():
    """ Configuration of a camera whose detection is the matched filter. """

    config = _config()
    config.mf_enable = True
    config.mf_replace_detection = True
    return config


def _status(tmp_path, config=None):
    """ The status of a camera with its input in tmp_path/input and its output in tmp_path/output. """

    config = config or _matchedFilterConfig()
    input_dir = tmp_path/'input'
    output_dir = tmp_path/'output'
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    scanner = lambda: mon.scanInputs(str(input_dir), str(output_dir), config, 'vid', False)
    return mst.MonitorStatus('XX0001', str(output_dir), str(input_dir), config, scanner, 2), config


def _inputFile(tmp_path, name, minutes_ago):
    """ An input file last written the given number of minutes before NOW. """

    path = tmp_path/'input'/name
    path.write_bytes(b'0'*100)
    t = (NOW - datetime.timedelta(minutes=minutes_ago) - datetime.datetime(1970, 1, 1)).total_seconds()
    os.utime(str(path), (t, t))
    return str(path), t


def _searchedFile(output_dir, config, name, n_chunks=12, start_s=0, clear=1.0, detections=3, removed_utc=(),
                  recalibrated=True, shift_deg=0.0, fps=25.0):
    """ Results of a file which the matched filter searched, with its conditions in the log and its recalibrated
        platepars (fitted exactly on the stars, the pointing shifted by shift_deg in altitude).
    """

    results_dir = _makeResults(output_dir, config, name, n_chunks=n_chunks, start_s=start_s)
    results_path = os.path.join(output_dir, results_dir)

    # A different length of the chunks (the chunks still start every 128 frames at 25 fps)
    if fps != 25.0:
        star_list, _ = CALSTARS.readCALSTARS(results_path, 'CALSTARS_{:s}.txt'.format(name))
        CALSTARS.writeCALSTARS(star_list, results_path, 'CALSTARS_{:s}.txt'.format(name), 'XX0001', 48, 64,
                               chunk_frames=128, fps=fps)

    results_path = os.path.join(output_dir, results_dir)
    input_file = '/data/{:s}.vid'.format(name)

    # The done flag with the input file, and the summary of the matched filter
    info = mnr.readDoneFlag(results_path)
    info['input_file'] = input_file
    mnr.writeDoneFlag(results_path, info)
    with open(os.path.join(results_path, mnr.MF_DONE_NAME), 'w') as f:
        json.dump({'total_frames': 128*n_chunks, 'clear_sky_fraction': clear, 'detections': detections,
                   'cloud_detections_removed': len(removed_utc), 'gpu': True, 'tracks_dropped': 0,
                   'cloud_removed_utc': [t.strftime('%Y-%m-%dT%H:%M:%S.%f') for t in removed_utc]}, f)

    # The conditions of its chunks
    times = [BEG_TIME + datetime.timedelta(seconds=start_s + i*128/25.0) for i in range(n_chunks)]
    log_path = os.path.join(output_dir, 'sky_conditions_XX0001_{:s}.csv'.format(BEG_TIME.strftime('%Y%m%d')))
    new = not os.path.isfile(log_path)
    with open(log_path, 'a') as f:
        if new:
            f.write(','.join(CONDITION_COLUMNS) + ',processed_utc\n')
        for t in times:
            f.write('{:s},{:s},500,300,{:.3f},0.020,0.950,2025-12-25T04:00:00\n'.format(
                t.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3], input_file, clear))

    # The platepars recalibrated on every chunk, on stars which the platepar projects exactly
    reference = Platepar()
    reference.read(TEMPLATE_PLATEPAR)
    recalibrated_pps = {}
    for t in times:
        pp = Platepar()
        pp.read(TEMPLATE_PLATEPAR)
        time_data = [(t.year, t.month, t.day, t.hour, t.minute, t.second, 0)]*5
        jd, ra, dec, _ = xyToRaDecPP(time_data, [100.0, 640.0, 1000.0, 300.0, 800.0],
                                     [100.0, 360.0, 600.0, 500.0, 200.0], [1]*5, pp, extinction_correction=False)
        x, y = raDecToXYPP(ra, dec, np.mean(jd), pp)
        pp.star_list = [[j, yy, xx, 1000, r, d, 5.0] for j, yy, xx, r, d in zip(jd, y, x, ra, dec)]
        pp.auto_recalibrated = recalibrated
        pp.alt_centre = reference.alt_centre + shift_deg
        recalibrated_pps[FFfile.constructFFName('XX0001', t, ext='fits')] = json.loads(pp.jsonStr())
    with open(os.path.join(results_path, config.platepars_recalibrated_name), 'w') as f:
        json.dump(recalibrated_pps, f)

    return results_dir


### Sections ###

def test_recording_state(tmp_path, now):
    """ The input is being recorded while the newest file grows, not recorded at night otherwise, and the state
        is never a fault.
    """

    status, config = _status(tmp_path)

    # The newest file was written a minute ago
    _inputFile(tmp_path, 'a.vid', 20)
    _inputFile(tmp_path, 'b.vid', 1)
    recording = status.build(_state())['recording']
    assert recording['state'] == 'recording'
    assert recording['newest_file'] == 'b.vid' and recording['files_tonight'] == 2

    # Nothing written for 10 minutes at night: not recording, which is not a fault
    _inputFile(tmp_path, 'b.vid', 10)
    built = status.build(_state())
    assert built['recording']['state'] == 'not_recording'
    assert built['faults'] == [] and built['overall'] == 'ok'

    # The input can't be read
    built = status.build(_state(input_readable=False))
    assert built['recording']['state'] == 'unknown'
    assert [fault['code'] for fault in built['faults']] == ['INPUT_UNREADABLE']


def test_processing_and_keeping_up(tmp_path, now):
    """ The processing times of the files and the delay of the oldest waiting file, which doesn't keep up when it
        was recorded more than three file lengths ago.
    """

    status, config = _status(tmp_path)
    _searchedFile(str(tmp_path/'output'), config, 'file1', n_chunks=12)

    # Processed files: the length of the file comes from its results (12 chunks of 5.12 s)
    for minutes in (1.0, 2.0, 3.0):
        status.fileFinished('file1', 'ok', 60*minutes)
    status.fileFinished('file2', 'failed', 30)

    # A file waits which was recorded two minutes ago (about two file lengths)
    path, mtime = _inputFile(tmp_path, 'c.vid', 2)
    processing = status.build(_state(queued=[(path, mtime)]))['processing']
    assert processing['file_length_min'] == pytest.approx(12*5.12/60, abs=0.01)
    assert processing['queued'] == 1 and processing['delay_min'] == pytest.approx(2.0, abs=0.1)
    assert not processing['not_keeping_up']
    assert processing['processing_time_min_last_24h']['median'] == pytest.approx(2.0)
    assert processing['processing_time_min_last_24h']['n'] == 3
    assert processing['files_failed_last_24h'] == 1

    # It was recorded ten minutes ago: not keeping up
    path, mtime = _inputFile(tmp_path, 'c.vid', 10)
    built = status.build(_state(queued=[(path, mtime)]))
    assert built['processing']['not_keeping_up']
    assert 'NOT_KEEPING_UP' in [fault['code'] for fault in built['faults']]


def test_sky_and_night(tmp_path, now):
    """ The sky in the latest data and the night so far, from the conditions of the measured files, with the
        files which were not searched and the detections.
    """

    status, config = _status(tmp_path)
    output_dir = str(tmp_path/'output')
    _searchedFile(output_dir, config, 'file1', n_chunks=12, clear=1.0, detections=3,
                  removed_utc=[BEG_TIME + datetime.timedelta(seconds=10)])
    _searchedFile(output_dir, config, 'file2', n_chunks=12, start_s=62, clear=0.0, detections=1)
    _searchedFile(output_dir, config, 'file2b', n_chunks=60, start_s=124, clear=0.0, detections=0)

    built = status.build(_state())
    night, sky = built['night'], built['sky']
    assert night['name'] == mnr.nightInfo(config, NOW)[0]
    assert night['files_processed'] == 3 and night['removed_by_clouds'] == 1

    # The detections are counted in the FTPdetectinfo of every file (one each)
    assert night['detections'] == 3
    assert night['clear_percent'] == pytest.approx(100*12/84, abs=1.0)
    assert night['clouded_percent'] == pytest.approx(100*72/84, abs=1.0)

    # The latest data is clouded, and not observable
    assert sky['state'] == 'clouded' and sky['observable'] is False
    assert sky['clear_percent'] == 0.0 and sky['stars'] == 500

    # All clear in the latest data: observable
    _searchedFile(output_dir, config, 'file3', n_chunks=150, start_s=432, clear=1.0)
    sky = status.build(_state())['sky']
    assert sky['state'] == 'clear' and sky['observable'] is True


def test_calibration(tmp_path, now):
    """ The calibration is ok when the chunks on clear sky are recalibrated, failing when most of them are not,
        and the camera has moved when the pointing differs from the reference platepar by more than 2 px.
    """

    config = _matchedFilterConfig()

    # Recalibrated on the exact stars, the pointing as the reference
    status, _ = _status(tmp_path/'ok', config)
    _searchedFile(str(tmp_path/'ok'/'output'), config, 'file1')
    calibration = status.build(_state())['calibration']
    assert calibration['status'] == 'ok'
    assert calibration['last_file']['median_residual_px'] < 0.01
    assert calibration['last_file']['clear_recalibrated_share'] == 1.0

    # Not recalibrated on clear sky
    status, _ = _status(tmp_path/'failing', config)
    _searchedFile(str(tmp_path/'failing'/'output'), config, 'file1', recalibrated=False)
    built = status.build(_state())
    assert built['calibration']['status'] == 'failing'
    assert [fault['code'] for fault in built['faults']] == ['CALIBRATION_FAILING']
    assert built['overall'] == 'error'

    # Not recalibrated, but clouded: the calibration can't be judged, which is not a fault
    status, _ = _status(tmp_path/'clouded', config)
    _searchedFile(str(tmp_path/'clouded'/'output'), config, 'file1', recalibrated=False, clear=0.0)
    built = status.build(_state())
    assert built['calibration']['status'] == 'unknown' and built['faults'] == []

    # The pointing moved by much more than 2 px
    status, _ = _status(tmp_path/'moved', config)
    _searchedFile(str(tmp_path/'moved'/'output'), config, 'file1', shift_deg=0.5)
    built = status.build(_state())
    assert built['calibration']['status'] == 'degraded'
    assert built['calibration']['last_file']['median_pointing_shift_px'] > 2
    assert [fault['code'] for fault in built['faults']] == ['CAMERA_MOVED']


def test_faults(tmp_path, now):
    """ The faults of the monitor, the failed files, the disk, the night report and the detector, with the time
        they were first seen.
    """

    status, config = _status(tmp_path)
    output_dir = str(tmp_path/'output')
    results_dir = _searchedFile(output_dir, config, 'file1')

    # The last file ran on the CPU and dropped candidates
    summary_path = os.path.join(output_dir, results_dir, mnr.MF_DONE_NAME)
    with open(summary_path) as f:
        summary = json.load(f)
    summary.update({'gpu': False, 'tracks_dropped': 5})
    with open(summary_path, 'w') as f:
        json.dump(summary, f)
    config.mf_gpu = 'on'

    # The latest night report failed in an essential step
    mnr.updateReportState(output_dir, NIGHT, reported_at='2025-12-25T12:00:00.000000', failed_steps=['archive'])

    failed = {'f1': {'count': 1, 'kills': 0, 'last_fail_time': 0}, 'f2': {'count': 2, 'kills': 0,
                                                                           'last_fail_time': 0}}
    built = status.build(_state(failed_files=failed, paused_for_space=True, stopped=True))
    codes = [fault['code'] for fault in built['faults']]
    assert set(codes) == {'MONITOR_STOPPED', 'DISK_LOW', 'FILE_GIVEN_UP', 'FILE_FAILED', 'REPORT_FAILED',
                          'GPU_NOT_USED', 'CANDIDATES_DROPPED'}
    assert built['overall'] == 'error'
    assert built['monitor']['state'] == 'stopped'

    # The most severe first, every fault with the time it was first seen
    severities = [mst.SEVERITIES.index(fault['severity']) for fault in built['faults']]
    assert severities == sorted(severities, reverse=True)
    assert all(fault['since_utc'] == NOW.strftime(mnr.JSON_TIME_FORMAT) for fault in built['faults'])


### The status file ###

def test_status_file_is_written_and_continued(tmp_path, now, monkeypatch):
    """ The status file is written when it is due (every interval and right after a file was processed), the
        processed files continue after a restart, and an error writing it doesn't stop the monitor.
    """

    status, config = _status(tmp_path)
    assert status.update(_state())
    with open(status.path) as f:
        written = json.load(f)
    assert written['schema_version'] == mst.SCHEMA_VERSION and written['overall'] == 'ok'
    assert not os.path.isfile(status.path + '.tmp')

    # Not again before the interval, but right after a processed file
    assert not status.update(_state())
    status.fileFinished('file1', 'ok', 120)
    assert status.update(_state())

    # A new monitor continues the processed files
    restarted, _ = _status(tmp_path, config)
    assert [r['file'] for r in restarted.file_records] == ['file1']

    # An error is only logged
    monkeypatch.setattr(mst, '_writeJSON', lambda path, data: (_ for _ in ()).throw(OSError('disk full')))
    assert not restarted.update(_state(), force=True)


def test_matched_filter_names():
    """ The names of the files of the matched filter in the night report are the ones of the matched filter. """

    from RMS.MatchedFilterDetection import DONE_NAME, MATCHED_FILTER_DIR
    assert mnr.MF_DONE_NAME == DONE_NAME
    assert mnr.MF_DIR_NAME == MATCHED_FILTER_DIR


def _fakeWorker(*args, **kwargs):
    """ A worker which processes nothing. """

    raise SystemExit(0)


def test_monitor_loop_writes_the_status(tmp_path, monkeypatch):
    """ The monitor writes the status while it runs, with the processed files, and says it stopped when it
        stops.
    """

    input_dir = tmp_path/'input'
    output_dir = tmp_path/'output'
    input_dir.mkdir()
    output_dir.mkdir()
    old = time.time() - 3600
    vid_path = input_dir/'dump_6AC5999C_02F.vid'
    vid_path.write_bytes(b'0'*100)
    os.utime(str(vid_path), (old, old))

    # The worker does nothing, and the monitor stops when the status has the processed file (at most after 30 s)
    monkeypatch.setattr(mon, 'processFileWorker', _fakeWorker)
    status_path = str(output_dir/mst.STATUS_FILE_NAME)
    real_sleep = time.sleep
    started = time.time()
    def _sleep(seconds):
        if time.time() - started > 30:
            raise KeyboardInterrupt
        try:
            with open(status_path) as f:
                if json.load(f)['processing']['recent_files']:
                    raise KeyboardInterrupt
        except (OSError, ValueError):
            pass
        real_sleep(0.05)
    monkeypatch.setattr(mon.time, 'sleep', _sleep)

    mon.monitorDirectory(str(input_dir), 'vid', REPO_CONFIG, TEMPLATE_PLATEPAR, str(output_dir), nproc=1,
                         poll_interval=0.01, report_mode='none')

    with open(str(output_dir/mst.STATUS_FILE_NAME)) as f:
        status = json.load(f)
    assert status['monitor']['state'] == 'stopped'
    assert 'MONITOR_STOPPED' in [fault['code'] for fault in status['faults']]
    assert [r['file'] for r in status['processing']['recent_files']] == ['dump_6AC5999C_02F']
    assert status['processing']['recent_files'][0]['outcome'] == 'ok'


def test_multicamera_monitor_writes_the_status_of_every_camera(tmp_path, monkeypatch):
    """ In multi-camera mode every camera has its own status, with its own files. """

    cameras = []
    for cam_id in ('CAM1', 'CAM2'):
        input_dir = tmp_path/cam_id/'input'
        output_dir = tmp_path/cam_id/'output'
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        old = time.time() - 3600
        vid_path = input_dir/'dump_{:s}_02F.vid'.format(cam_id)
        vid_path.write_bytes(b'0'*100)
        os.utime(str(vid_path), (old, old))
        cameras.append((cam_id, input_dir, output_dir))

    ini_path = tmp_path/'multicam.ini'
    ini_path.write_text('[Global]\nnproc = 2\nfile_type = vid\npoll_interval = 0.01\n\n' + '\n'.join(
        '[{:s}]\ninput_dir = {:s}\noutput_dir = {:s}\nconfig = {:s}\nplatepar = {:s}\n'.format(
            cam_id, str(input_dir), str(output_dir), REPO_CONFIG, TEMPLATE_PLATEPAR)
        for cam_id, input_dir, output_dir in cameras))

    # The workers do nothing, and the monitor stops when both statuses have their processed file
    monkeypatch.setattr(mon, 'processFileWorker', _fakeWorker)
    real_sleep = time.sleep
    started = time.time()
    def _sleep(seconds):
        if time.time() - started > 30:
            raise KeyboardInterrupt
        try:
            done = 0
            for _, _, output_dir in cameras:
                with open(str(output_dir/mst.STATUS_FILE_NAME)) as f:
                    done += bool(json.load(f)['processing']['recent_files'])
            if done == len(cameras):
                raise KeyboardInterrupt
        except (OSError, ValueError):
            pass
        real_sleep(0.05)
    monkeypatch.setattr(mon.time, 'sleep', _sleep)

    mon.monitorMultipleCameras(str(ini_path), report_mode='none')

    for cam_id, _, output_dir in cameras:
        with open(str(output_dir/mst.STATUS_FILE_NAME)) as f:
            status = json.load(f)
        assert status['camera'] == cam_id and status['monitor']['state'] == 'stopped'
        assert [r['file'] for r in status['processing']['recent_files']] == ['dump_{:s}_02F'.format(cam_id)]


def test_a_failing_section_doesnt_stop_the_status(tmp_path, now, monkeypatch):
    """ A section of the status which fails is reported as a fault, and the rest of the status is written. """

    status, config = _status(tmp_path)
    monkeypatch.setattr(mst, 'availableSpace', lambda path: (_ for _ in ()).throw(OSError('stale handle')))

    assert status.update(_state())
    with open(status.path) as f:
        written = json.load(f)
    assert [fault['code'] for fault in written['faults']] == ['STATUS_INCOMPLETE']
    assert written['overall'] == 'warning' and written['recording']['state'] == 'not_recording'


### All input types ###

# An input of every type the monitor reads: (file_type, name, is a FITS directory)
INPUTS = [('vid', 'dump_6AC5999C_02F.vid', False), ('mkv', 'CAWE01_20261008_010203_123456_video.mkv', False),
          ('mp4', '20261008_010203.mp4', False), ('ff', 'FF_XX0001_20251225_031500_123_0000000.fits', False),
          ('ff', 'FF_XX0001_20251225_031500_123_0000000.bin', False), ('fitsdirs', '20261008_010203', True)]


def _makeInput(input_dir, name, fits_dir, minutes_ago=60):
    """ An input file, or a FITS directory with frames, last written the given number of minutes ago. """

    t = time.time() - 60*minutes_ago
    path = os.path.join(str(input_dir), name)
    if fits_dir:
        os.makedirs(path)
        for i in range(3):
            frame = os.path.join(path, 'frame_{:03d}.fits'.format(i))
            with open(frame, 'wb') as f:
                f.write(b'0'*100)
            os.utime(frame, (t, t))
    else:
        with open(path, 'wb') as f:
            f.write(b'0'*100)
    os.utime(path, (t, t))
    return path


@pytest.mark.parametrize('file_type, name, fits_dir', INPUTS)
def test_inputs_of_every_type(tmp_path, file_type, name, fits_dir):
    """ The inputs of every type are found as by the monitor, with the time of their last change and their size,
        and the other files and directories are not inputs.
    """

    path = _makeInput(tmp_path, name, fits_dir)

    # Not inputs: a file of another type, an empty directory, and for FITS directories a file
    _makeInput(tmp_path, 'notes.txt', False)
    os.makedirs(str(tmp_path/'empty_dir'))

    inputs = mon.scanInputs(str(tmp_path), str(tmp_path/'output'), _config(), file_type, False)
    assert [os.path.basename(p) for p, _, _ in inputs] == [name]
    assert mon.isInputPath(path, file_type)
    assert not mon.isInputPath(str(tmp_path/'empty_dir'), file_type)

    # The state of the input: the size of the file, or of the frames of a FITS directory
    mtime, size = mon.inputState(path, file_type)
    assert size == (300 if fits_dir else 100)
    assert mtime == pytest.approx(time.time() - 3600, abs=5)


def test_recursive_inputs(tmp_path):
    """ The inputs in the subdirectories are found when the monitor is recursive, the night directories of the
        monitor in the input directory are skipped (as by the monitor).
    """

    config = _config()
    os.makedirs(str(tmp_path/'night1'))
    _makeInput(tmp_path/'night1', 'a.vid', False)
    _makeInput(tmp_path, 'b.vid', False)
    os.makedirs(str(tmp_path/config.captured_dir))
    _makeInput(tmp_path/config.captured_dir, 'c.vid', False)

    names = lambda inputs: sorted(os.path.basename(p) for p, _, _ in inputs)
    assert names(mon.scanInputs(str(tmp_path), str(tmp_path), config, 'vid', True)) == ['a.vid', 'b.vid']
    assert names(mon.scanInputs(str(tmp_path), str(tmp_path), config, 'vid', False)) == ['b.vid']


def test_recording_of_fits_directories(tmp_path, now):
    """ A FITS directory which gets new frames is being recorded. """

    config = _matchedFilterConfig()
    (tmp_path/'output').mkdir()
    scanner = lambda: mon.scanInputs(str(tmp_path/'input'), str(tmp_path/'output'), config, 'fitsdirs', False)
    status = mst.MonitorStatus('XX0001', str(tmp_path/'output'), str(tmp_path/'input'), config, scanner, 2)

    # A directory written to a minute ago (by the time of the status)
    (tmp_path/'input').mkdir()
    path = _makeInput(tmp_path/'input', '20251225_030000', True)
    t = (NOW - datetime.timedelta(minutes=1) - datetime.datetime(1970, 1, 1)).total_seconds()
    os.utime(path, (t, t))

    recording = status.build(_state())['recording']
    assert recording['state'] == 'recording' and recording['newest_file'] == '20251225_030000'


@pytest.mark.parametrize('file_type, name, fits_dir', INPUTS)
def test_monitor_loop_with_every_input_type(tmp_path, monkeypatch, file_type, name, fits_dir):
    """ The monitor processes the inputs of every type and writes the status with them. """

    input_dir = tmp_path/'input'
    output_dir = tmp_path/'output'
    input_dir.mkdir()
    output_dir.mkdir()
    _makeInput(input_dir, name, fits_dir)

    monkeypatch.setattr(mon, 'processFileWorker', _fakeWorker)
    status_path = str(output_dir/mst.STATUS_FILE_NAME)
    real_sleep = time.sleep
    started = time.time()
    def _sleep(seconds):
        if time.time() - started > 30:
            raise KeyboardInterrupt
        try:
            with open(status_path) as f:
                if json.load(f)['processing']['recent_files']:
                    raise KeyboardInterrupt
        except (OSError, ValueError):
            pass
        real_sleep(0.05)
    monkeypatch.setattr(mon.time, 'sleep', _sleep)

    mon.monitorDirectory(str(input_dir), file_type, REPO_CONFIG, TEMPLATE_PLATEPAR, str(output_dir), nproc=1,
                         poll_interval=0.01, report_mode='none')

    with open(status_path) as f:
        status = json.load(f)
    assert [r['file'] for r in status['processing']['recent_files']] == [mon.uniqueId(name)]
    assert status['processing']['recent_files'][0]['input'] == name
    assert status['processing']['recent_files'][0]['size_gb'] == pytest.approx((300 if fits_dir else 100)/1024**3,
                                                                              abs=1e-4)
    assert status['recording']['newest_file'] == name and status['recording']['input_readable']


def test_multicamera_status_with_the_disk_full(tmp_path, monkeypatch):
    """ The files held back while the output disk is full are waiting in the status of the camera. """

    input_dir = tmp_path/'input'
    output_dir = tmp_path/'output'
    input_dir.mkdir()
    output_dir.mkdir()
    _makeInput(input_dir, 'dump_6AC5999C_02F.vid', False, minutes_ago=30)
    ini_path = tmp_path/'multicam.ini'
    ini_path.write_text('[Global]\nnproc = 1\nfile_type = vid\npoll_interval = 0.01\n\n[CAM1]\ninput_dir = {:s}\n'
                        'output_dir = {:s}\nconfig = {:s}\nplatepar = {:s}\n'.format(str(input_dir), str(output_dir),
                                                                                   REPO_CONFIG, TEMPLATE_PLATEPAR))

    # The disk is full, and the monitor stops after a few polls
    def _full(self):
        self.paused = True
        return False
    monkeypatch.setattr(mon.FreeSpaceGuard, 'ok', _full)
    polls = {'n': 0}
    real_sleep = time.sleep
    def _sleep(seconds):
        polls['n'] += 1
        if polls['n'] > 5:
            raise KeyboardInterrupt
        real_sleep(0.01)
    monkeypatch.setattr(mon.time, 'sleep', _sleep)

    mon.monitorMultipleCameras(str(ini_path), report_mode='none')

    with open(str(output_dir/mst.STATUS_FILE_NAME)) as f:
        status = json.load(f)
    assert status['processing']['queued'] == 1
    assert status['processing']['delay_min'] == pytest.approx(30, abs=1)
    assert 'DISK_LOW' in [fault['code'] for fault in status['faults']]


@pytest.mark.parametrize('mf_enable, replace', [(False, False), (True, False)])
def test_detections_without_the_matched_filter_replacing(tmp_path, now, mf_enable, replace):
    """ With the normal detection, the detections are counted in the FTPdetectinfo of the files, and the ones of a
        separate pass of the matched filter apart.
    """

    config = _config()
    config.mf_enable = mf_enable
    config.mf_replace_detection = replace
    status, _ = _status(tmp_path, config)
    output_dir = str(tmp_path/'output')

    for name in ('file1', 'file2'):
        results_dir = _makeResults(output_dir, config, name, n_chunks=3)
        if mf_enable:
            os.makedirs(os.path.join(output_dir, results_dir, mnr.MF_DIR_NAME))
            with open(os.path.join(output_dir, results_dir, mnr.MF_DIR_NAME, mnr.MF_DONE_NAME), 'w') as f:
                json.dump({'total_frames': 384, 'detections': 5}, f)

    built = status.build(_state())
    assert built['night']['detections'] == 2
    assert built['night'].get('matched_filter_detections') == (10 if mf_enable else None)
    assert built['detector']['name'] == ('normal_and_matched_filter' if mf_enable else 'normal')

    # Without the conditions, the sky has the stars of the latest chunk
    assert built['sky']['stars'] == 1 and built['sky']['state'] in ('no_data', 'too_few_stars')


### Progress and prediction ###

def _statusWithSizes(tmp_path, sizes_gb, nproc=2):
    """ The status of a camera whose inputs have the given sizes (GB) by their names. """

    config = _matchedFilterConfig()
    (tmp_path/'input').mkdir(parents=True, exist_ok=True)
    (tmp_path/'output').mkdir(parents=True, exist_ok=True)
    sizer = lambda path: int(sizes_gb[os.path.basename(path)]*1024**3)
    scanner = lambda: mon.scanInputs(str(tmp_path/'input'), str(tmp_path/'output'), config, 'vid', False)
    status = mst.MonitorStatus('XX0001', str(tmp_path/'output'), str(tmp_path/'input'), config, scanner, nproc,
                               input_sizer=sizer)
    return status, config


def test_queue_prediction(tmp_path, now):
    """ The time when the waiting files will be done is predicted from the processing time per GB of the recent
        files: the running files finish first, then every waiting file goes to the next free worker.
    """

    sizes = {'run.vid': 2.0, 'q1.vid': 1.0, 'q2.vid': 1.0, 'q3.vid': 4.0}
    status, config = _statusWithSizes(tmp_path, sizes, nproc=2)

    # Recent files took 1 min per GB
    for i in range(3):
        status.fileFinished('done{:d}'.format(i), 'ok', 60*2.0, input_name='done.vid', size_bytes=2*1024**3)

    # One file of 2 GB has been running for 0.5 min, three wait
    run_path, _ = _inputFile(tmp_path, 'run.vid', 5)
    queued = [_inputFile(tmp_path, name, 3) for name in ('q1.vid', 'q2.vid', 'q3.vid')]
    start = NOW - datetime.timedelta(minutes=0.5)
    processing = status.build(_state(in_progress=[('run', run_path, start)], queued=queued))['processing']

    current = processing['in_progress'][0]
    assert current['input'] == 'run.vid' and current['expected_min'] == 2.0 and current['remaining_min'] == 1.5
    assert current['expected_done_utc'] == (start + datetime.timedelta(minutes=2)).strftime(mnr.JSON_TIME_FORMAT)
    assert processing['queued_inputs'] == ['q1.vid', 'q2.vid', 'q3.vid']
    assert processing['prediction_basis'] == 'size'

    # Workers: [1.5, 0] -> q1 on the second (done at 1), q2 on it (done at 2), q3 on the first (done at 5.5)
    assert processing['queue_empty_in_min'] == 5.5
    assert processing['queue_empty_utc'] == (NOW + datetime.timedelta(minutes=5.5)).strftime(mnr.JSON_TIME_FORMAT)

    # The progress: three files wait and one runs, none is done tonight
    progress = processing['progress']
    assert (progress['done_tonight'], progress['in_progress'], progress['queued']) == (0, 1, 3)
    assert progress['percent_done'] == 0.0

    # In multi-camera mode the monitor gives the prediction for all cameras together
    processing = status.build(_state(in_progress=[('run', run_path, start)], queued=queued,
                                     queue_empty_min=12.0))['processing']
    assert processing['queue_empty_in_min'] == 12.0


def test_simulate_queues():
    """ The queues of the cameras are predicted as the monitor schedules them: the running files are done first,
        and a free worker takes a file of the camera with the fewest running files, then the next camera in turn.
    """

    # Two workers run long files of camera A: the file of camera B waits for one of them
    assert mst.simulateQueues(2, [('A', [60.0, 60.0], []), ('B', [], [5.0])]) == {'A': 60.0, 'B': 65.0}

    # Both cameras get a worker, then camera A gets both
    assert mst.simulateQueues(2, [('A', [], [10.0, 10.0, 10.0]), ('B', [], [10.0])]) == {'A': 20.0, 'B': 10.0}

    # A camera whose times are not known is not predicted, its files still take a worker (the median time)
    assert mst.simulateQueues(1, [('A', [], [None]), ('B', [], [4.0])]) == {'A': None, 'B': 8.0}

    # Nothing to do
    assert mst.simulateQueues(2, [('A', [], [])]) == {'A': 0.0}


def test_queue_prediction_without_sizes_or_history(tmp_path, now):
    """ Without the sizes of the inputs the prediction is per file, without processed files there is none, and
        nothing waiting is done now.
    """

    status, config = _status(tmp_path)
    path, mtime = _inputFile(tmp_path, 'q1.vid', 3)

    # Nothing processed yet: no prediction
    processing = status.build(_state(queued=[(path, mtime)]))['processing']
    assert processing['queue_empty_in_min'] is None and processing['prediction_basis'] is None

    # Processed files without their sizes: 3 min per file
    for i in range(3):
        status.fileFinished('done{:d}'.format(i), 'ok', 180)
    processing = status.build(_state(queued=[(path, mtime)]))['processing']
    assert processing['prediction_basis'] == 'file' and processing['queue_empty_in_min'] == 3.0

    # Nothing waits
    processing = status.build(_state())['processing']
    assert processing['queue_empty_in_min'] == 0.0
    assert processing['queue_empty_utc'] == NOW.strftime(mnr.JSON_TIME_FORMAT)


def test_progress_of_the_night(tmp_path, now):
    """ The files of the night which are done count towards the progress. """

    status, config = _status(tmp_path)
    _searchedFile(str(tmp_path/'output'), config, 'file1')
    _searchedFile(str(tmp_path/'output'), config, 'file2', start_s=62)
    path, mtime = _inputFile(tmp_path, 'q1.vid', 3)

    progress = status.build(_state(queued=[(path, mtime)]))['processing']['progress']
    assert (progress['done_tonight'], progress['queued']) == (2, 1)
    assert progress['percent_done'] == pytest.approx(100*2/3, abs=0.1)


def test_history_of_the_last_day(tmp_path, monkeypatch):
    """ All files of the last day are kept for the statistics (also more than the ones listed), also after a
        restart, and older ones are forgotten.
    """

    times = {'now': NOW}
    monkeypatch.setattr(mst.RmsDateTime, 'utcnow', staticmethod(lambda: times['now']))
    status, config = _status(tmp_path)

    # 100 files, one every 10 minutes (about 16 hours)
    for i in range(100):
        times['now'] = NOW + datetime.timedelta(minutes=10*i)
        status.fileFinished('f{:03d}'.format(i), 'ok', 60)
    processing = status.build(_state())['processing']
    assert processing['files_done_last_24h'] == 100 and len(processing['recent_files']) == mst.RECENT_FILES

    # A damaged line in the history is skipped after a restart
    with open(status.history_path, 'a') as f:
        f.write('{"broken\n')
    restarted, _ = _status(tmp_path, config)
    assert len(restarted.file_records) == 100

    # 20 hours after the last file only the files of the last 24 hours (f075 to f099) are kept, also in the
    #   history file
    times['now'] = NOW + datetime.timedelta(minutes=10*99) + datetime.timedelta(hours=20)
    restarted, _ = _status(tmp_path, config)
    assert len(restarted.file_records) == 25
    with open(restarted.history_path) as f:
        assert len(f.readlines()) == 25


def test_a_failing_results_scan_doesnt_stop_the_status(tmp_path, now, monkeypatch):
    """ The status is written when the results of the night can't be read. """

    status, config = _status(tmp_path)
    monkeypatch.setattr(mst, 'scanNights', lambda *args, **kwargs: (_ for _ in ()).throw(OSError('stale')))

    assert status.update(_state())
    with open(status.path) as f:
        written = json.load(f)
    assert [fault['code'] for fault in written['faults']] == ['STATUS_INCOMPLETE']
    assert 'results' in written['faults'][0]['message']


### Edge cases ###

def test_missing_recursive_input(tmp_path, now):
    """ A recursive input which can't be read (e.g. the network share is gone) is unreadable, not empty. """

    config = _matchedFilterConfig()
    with pytest.raises(OSError):
        mon.scanInputs(str(tmp_path/'gone'), str(tmp_path/'output'), config, 'vid', True)
    with pytest.raises(OSError):
        list(mon.walkInput(str(tmp_path/'gone'), str(tmp_path/'output'), config))

    (tmp_path/'output').mkdir()
    scanner = lambda: mon.scanInputs(str(tmp_path/'gone'), str(tmp_path/'output'), config, 'vid', True)
    status = mst.MonitorStatus('XX0001', str(tmp_path/'output'), str(tmp_path/'gone'), config, scanner, 2)
    built = status.build(_state(input_readable=None))
    assert built['recording']['state'] == 'unknown'
    assert [fault['code'] for fault in built['faults']] == ['INPUT_UNREADABLE']


def test_old_files_drop_out_without_new_ones(tmp_path, monkeypatch):
    """ The files processed more than a day ago drop out of the statistics also when no file was processed
        since.
    """

    times = {'now': NOW}
    monkeypatch.setattr(mst.RmsDateTime, 'utcnow', staticmethod(lambda: times['now']))
    status, config = _status(tmp_path)
    status.fileFinished('f1', 'ok', 60)
    assert status.build(_state())['processing']['files_done_last_24h'] == 1

    times['now'] = NOW + datetime.timedelta(hours=25)
    processing = status.build(_state())['processing']
    assert processing['files_done_last_24h'] == 0 and processing['last_done'] is None


def test_lengths_of_the_files_are_kept(tmp_path, now):
    """ The lengths of the processed files are kept in the history, so they are known after a restart. """

    status, config = _status(tmp_path)
    _searchedFile(str(tmp_path/'output'), config, 'file1', n_chunks=12)
    status.fileFinished('file1', 'ok', 60)
    status.build(_state())

    restarted, _ = _status(tmp_path, config)
    assert restarted.file_records[0]['length_min'] == pytest.approx(12*5.12/60, abs=0.01)


def _hangingWorker(*args, **kwargs):
    """ A worker which takes long. """

    time.sleep(60)


def test_stopped_workers_are_not_in_the_final_status(tmp_path, monkeypatch):
    """ The workers which are stopped with the monitor are not processing in its final status. """

    input_dir = tmp_path/'input'
    output_dir = tmp_path/'output'
    input_dir.mkdir()
    output_dir.mkdir()
    _makeInput(input_dir, 'dump_6AC5999C_02F.vid', False)

    # The worker takes long, and the monitor stops once the status shows it processing
    monkeypatch.setattr(mon, 'processFileWorker', _hangingWorker)
    status_path = str(output_dir/mst.STATUS_FILE_NAME)
    real_sleep = time.sleep
    started = time.time()
    def _sleep(seconds):
        if time.time() - started > 30:
            raise KeyboardInterrupt
        try:
            with open(status_path) as f:
                if json.load(f)['processing']['in_progress']:
                    raise KeyboardInterrupt
        except (OSError, ValueError):
            pass
        real_sleep(0.05)
    monkeypatch.setattr(mon.time, 'sleep', _sleep)

    # The status is written at every poll
    monkeypatch.setattr(mst.MonitorStatus, 'update', _alwaysUpdate(mst.MonitorStatus.update))

    mon.monitorDirectory(str(input_dir), 'vid', REPO_CONFIG, TEMPLATE_PLATEPAR, str(output_dir), nproc=1,
                         poll_interval=0.01, report_mode='none')

    with open(status_path) as f:
        status = json.load(f)
    assert status['monitor']['state'] == 'stopped'
    assert status['processing']['in_progress'] == [] and status['monitor']['workers_this_camera'] == 0


def _alwaysUpdate(update):
    """ Write the status at every update. """

    def _update(self, state, force=False):
        return update(self, state, force=True)

    return _update


def _runningUpdate(update):
    """ Write the status at every update while the monitor runs, and keep it when the monitor stops (the
        stopped status has no prediction of the queue).
    """

    def _update(self, state, force=False):
        return False if state.get('stopped') else update(self, state, force=True)

    return _update


def test_latest_sky_of_a_file_which_was_not_searched(tmp_path, now):
    """ When the latest file was not searched (too few stars), the sky has its state and stars, and not the
        values of the older measured file.
    """

    status, config = _status(tmp_path)
    output_dir = str(tmp_path/'output')
    _searchedFile(output_dir, config, 'file1', n_chunks=12, clear=1.0)

    # A later file with one star per chunk, which was not searched
    skipped = _makeResults(output_dir, config, 'file2', n_chunks=3, start_s=120)
    with open(os.path.join(output_dir, skipped, mnr.MF_DONE_NAME), 'w') as f:
        json.dump({'max_stars': 1}, f)

    sky = status.build(_state())['sky']
    assert sky['state'] == 'too_few_stars' and sky['stars'] == 1
    assert 'clear_percent' not in sky and 'matched_stars' not in sky


@pytest.mark.parametrize('update, latest_newer, expected', [(True, True, 'latest'), (True, False, 'given'),
                                                           (False, True, 'given')])
def test_platepar_in_use(tmp_path, now, update, latest_newer, expected):
    """ The platepar in use is selected as by the workers: the best platepar of the latest night if it is carried
        forward and newer than the given one.
    """

    config = _matchedFilterConfig()
    config.monitor_update_platepar = update
    (tmp_path/'input').mkdir()
    (tmp_path/'output').mkdir()
    given = str(tmp_path/'platepar.cal')
    latest = mnr.latestPlateparPath(str(tmp_path/'output'), config)
    for path, age in [(given, 100 if latest_newer else 10), (latest, 10 if latest_newer else 100)]:
        with open(TEMPLATE_PLATEPAR) as src, open(path, 'w') as dst:
            dst.write(src.read())
        t = time.time() - age
        os.utime(path, (t, t))

    scanner = lambda: mon.scanInputs(str(tmp_path/'input'), str(tmp_path/'output'), config, 'vid', False)
    status = mst.MonitorStatus('XX0001', str(tmp_path/'output'), str(tmp_path/'input'), config, scanner, 2,
                               platepar_path=given)
    calibration = status.build(_state())['calibration']
    assert calibration['platepar'] == (latest if expected == 'latest' else given)
    assert calibration['platepar_from_previous_night'] == (expected == 'latest')


def test_reports_which_didnt_finish(tmp_path, now):
    """ A report which was started but didn't finish is not the latest report. """

    status, config = _status(tmp_path)
    output_dir = str(tmp_path/'output')
    mnr.updateReportState(output_dir, 'XX0001_20251223_222000_000000', reported_at='2025-12-24T12:00:00.000000',
                          failed_steps=[])
    mnr.updateReportState(output_dir, NIGHT, trigger='sunrise', upload_files=[])

    reports = status.build(_state())['reports']
    assert reports['last_night'] == 'XX0001_20251223_222000_000000' and reports['last_report_ok']


def test_results_which_cant_be_read_are_read_again(tmp_path, now):
    """ A processed file whose results can't be read counts as processed, and is read again later. """

    status, config = _status(tmp_path)
    output_dir = str(tmp_path/'output')
    results_dir = _searchedFile(output_dir, config, 'file1')
    summary_path = os.path.join(output_dir, results_dir, mnr.MF_DONE_NAME)
    with open(summary_path) as f:
        summary = f.read()
    with open(summary_path, 'w') as f:
        f.write(summary[:20])

    night = status.build(_state())['night']
    assert (night['files_processed'], night['files_unreadable'], night['files_not_searched']) == (1, 1, 0)
    assert night['detections'] == 1

    # The summary is complete again (the done flag didn't change)
    with open(summary_path, 'w') as f:
        f.write(summary)
    night = status.build(_state())['night']
    assert (night['files_processed'], night['files_unreadable']) == (1, 0)
    assert night['observed_h'] > 0


def test_latest_sky_of_overlapping_files(tmp_path, now):
    """ With two files of the same time, one clouded and one clear, the sky is clouded, with the values of the
        clouded chunk.
    """

    status, config = _status(tmp_path)
    output_dir = str(tmp_path/'output')
    _searchedFile(output_dir, config, 'file1', n_chunks=12, clear=0.0)
    _searchedFile(output_dir, config, 'file2', n_chunks=12, clear=1.0)

    sky = status.build(_state())['sky']
    assert sky['state'] == 'clouded' and sky['clear_percent'] == 0.0


def test_calibration_of_files_which_were_not_measured(tmp_path, now):
    """ With the cloud filter, the chunks of a file whose sky was not measured are not on known clear sky, so its
        failed recalibrations don't make the calibration fail; without the cloud filter all chunks count.
    """

    for cloud_filter, expected in [(True, 'unknown'), (False, 'failing')]:
        config = _matchedFilterConfig()
        config.mf_cloud_filter = cloud_filter
        status, _ = _status(tmp_path/str(cloud_filter), config)
        output_dir = str(tmp_path/str(cloud_filter)/'output')
        results_dir = _searchedFile(output_dir, config, 'file1', recalibrated=False)

        # The sky was not measured (no sky quality in the summary)
        summary_path = os.path.join(output_dir, results_dir, mnr.MF_DONE_NAME)
        with open(summary_path) as f:
            summary = json.load(f)
        summary['clear_sky_fraction'] = None
        with open(summary_path, 'w') as f:
            json.dump(summary, f)

        assert status.build(_state())['calibration']['status'] == expected


def test_file_length_from_the_history(tmp_path, now):
    """ Before a file of the night was processed, the length of the files comes from the files of the last day. """

    status, config = _status(tmp_path)
    status.fileFinished('old', 'ok', 600)
    status.file_records[-1]['length_min'] = 10.0

    path, mtime = _inputFile(tmp_path, 'q1.vid', 40)
    processing = status.build(_state(queued=[(path, mtime)]))['processing']
    assert processing['file_length_min'] == 10.0 and processing['not_keeping_up']


def test_unreadable_camera_has_no_queue(tmp_path, monkeypatch):
    """ In multi-camera mode, a camera whose input can't be read any more has no waiting files. """

    input_dir = tmp_path/'input'
    output_dir = tmp_path/'output'
    input_dir.mkdir()
    output_dir.mkdir()
    _makeInput(input_dir, 'dump_6AC5999C_02F.vid', False, minutes_ago=30)
    ini_path = tmp_path/'multicam.ini'
    ini_path.write_text('[Global]\nnproc = 1\nfile_type = vid\npoll_interval = 0.01\n\n[CAM1]\ninput_dir = {:s}\n'
                        'output_dir = {:s}\nconfig = {:s}\nplatepar = {:s}\n'.format(str(input_dir), str(output_dir),
                                                                                   REPO_CONFIG, TEMPLATE_PLATEPAR))

    # The disk is full (the file waits), then the input disappears, and the monitor stops
    def _full(self):
        self.paused = True
        return False
    monkeypatch.setattr(mon.FreeSpaceGuard, 'ok', _full)
    polls = {'n': 0}
    real_sleep = time.sleep
    def _sleep(seconds):
        polls['n'] += 1
        if polls['n'] == 3:
            os.rename(str(input_dir), str(tmp_path/'gone'))
        if polls['n'] > 6:
            raise KeyboardInterrupt
        real_sleep(0.01)
    monkeypatch.setattr(mon.time, 'sleep', _sleep)
    monkeypatch.setattr(mst.MonitorStatus, 'update', _alwaysUpdate(mst.MonitorStatus.update))

    mon.monitorMultipleCameras(str(ini_path), report_mode='none')

    with open(str(output_dir/mst.STATUS_FILE_NAME)) as f:
        status = json.load(f)
    assert status['processing']['queued'] == 0
    assert 'INPUT_UNREADABLE' in [fault['code'] for fault in status['faults']]


def test_observable_with_overlapping_files(tmp_path, now):
    """ Overlapping files count once in the observable time, with the worse state: four mostly clear files and one
        clouded file of the same time are not observable.
    """

    status, config = _status(tmp_path)
    output_dir = str(tmp_path/'output')
    for i in range(4):
        _searchedFile(output_dir, config, 'clear{:d}'.format(i), n_chunks=60, clear=0.85)
    _searchedFile(output_dir, config, 'clouded', n_chunks=60, clear=0.0)

    assert status.build(_state())['sky']['observable'] is False


def test_pointing_shift_on_clear_sky_only(tmp_path, now):
    """ The pointing shift of the camera is only measured on the chunks on clear sky. """

    config = _matchedFilterConfig()
    output_dir = str(tmp_path/'output')
    os.makedirs(output_dir)
    results_dir = _searchedFile(output_dir, config, 'file1', shift_deg=0.5)
    results_path = os.path.join(output_dir, results_dir)

    assert len(mst.fileCalibration(results_path, config, clear_times=None)['shifts_px']) == 12
    assert mst.fileCalibration(results_path, config, clear_times=[])['shifts_px'] == []


def test_paused_camera_doesnt_share_the_workers(tmp_path, monkeypatch):
    """ The waiting files of a camera paused for disk space don't take workers from the prediction of another
        camera, which gets its file done in its expected time.
    """

    cameras = []
    for cam_id, n_files in (('CAM1', 1), ('CAM2', 3)):
        input_dir = tmp_path/cam_id/'input'
        output_dir = tmp_path/cam_id/'output'
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        for i in range(n_files):
            _makeInput(input_dir, 'dump_{:s}{:d}_02F.vid'.format(cam_id, i), False, minutes_ago=30)
        cameras.append((cam_id, input_dir, output_dir))

        # Processed files of the last day, so the processing can be predicted
        with open(str(output_dir/mst.HISTORY_FILE_NAME), 'w') as f:
            for i in range(3):
                f.write(json.dumps({'file': 'old{:d}'.format(i), 'outcome': 'ok', 'processing_min': 5.0,
                                    'finished_utc': mst.RmsDateTime.utcnow().strftime(mnr.JSON_TIME_FORMAT)})
                        + '\n')

    ini_path = tmp_path/'multicam.ini'
    ini_path.write_text('[Global]\nnproc = 4\nfile_type = vid\npoll_interval = 0.01\n\n' + '\n'.join(
        '[{:s}]\ninput_dir = {:s}\noutput_dir = {:s}\nconfig = {:s}\nplatepar = {:s}\n'.format(
            cam_id, str(input_dir), str(output_dir), REPO_CONFIG, TEMPLATE_PLATEPAR)
        for cam_id, input_dir, output_dir in cameras))

    # The disk of every camera is full (nothing is processed), CAM2 is paused
    paused_dir = str(cameras[1][2])
    def _ok(self):
        self.paused = (self.output_dir == paused_dir)
        return False
    monkeypatch.setattr(mon.FreeSpaceGuard, 'ok', _ok)
    polls = {'n': 0}
    real_sleep = time.sleep
    def _sleep(seconds):
        polls['n'] += 1
        if polls['n'] > 5:
            raise KeyboardInterrupt
        real_sleep(0.01)
    monkeypatch.setattr(mon.time, 'sleep', _sleep)
    monkeypatch.setattr(mst.MonitorStatus, 'update', _runningUpdate(mst.MonitorStatus.update))

    mon.monitorMultipleCameras(str(ini_path), report_mode='none')

    with open(str(cameras[0][2]/mst.STATUS_FILE_NAME)) as f:
        status = json.load(f)
    assert status['processing']['queued'] == 1 and status['processing']['queue_empty_in_min'] == 5.0

    # The paused camera can't tell when its files will be done
    with open(str(cameras[1][2]/mst.STATUS_FILE_NAME)) as f:
        status = json.load(f)
    assert status['processing']['queued'] == 3 and status['processing']['queue_empty_in_min'] is None


def test_progress_with_a_backlog_of_other_nights(tmp_path, now):
    """ The waiting files of other nights are counted apart from the progress of the night. """

    status, config = _status(tmp_path)
    tonight, t1 = _inputFile(tmp_path, 'tonight.vid', 10)
    old, t2 = _inputFile(tmp_path, 'old.vid', 60*24)

    progress = status.build(_state(queued=[(old, t2), (tonight, t1)]))['processing']['progress']
    assert (progress['queued'], progress['other_nights_open']) == (1, 1)


def test_simulate_queues_continues_the_turn():
    """ The prediction continues the turn of the cameras where the monitor is: with camera A given a worker last,
        camera B gets the next free worker when they have as many running files.
    """

    cameras = [('A', [], [10.0]), ('B', [], [10.0])]
    assert mst.simulateQueues(1, cameras, last_idx=0) == {'A': 20.0, 'B': 10.0}
    assert mst.simulateQueues(1, cameras, last_idx=1) == {'A': 10.0, 'B': 20.0}


def test_camera_without_the_status_shares_the_workers(tmp_path, monkeypatch):
    """ A camera without the status writes no status, but its files still take workers in the prediction of the
        other cameras.
    """

    cameras = []
    for cam_id, enabled in (('CAM1', True), ('CAM2', False)):
        input_dir = tmp_path/cam_id/'input'
        output_dir = tmp_path/cam_id/'output'
        input_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        _makeInput(input_dir, 'dump_{:s}_02F.vid'.format(cam_id), False, minutes_ago=30)
        config_path = tmp_path/'{:s}.config'.format(cam_id)
        with open(REPO_CONFIG) as f:
            config_text = f.read()
        config_path.write_text(config_text.replace('monitor_status: true', 'monitor_status: {}'.format(
            'true' if enabled else 'false')))
        cameras.append((cam_id, input_dir, output_dir, config_path))

        # The history of processed files of both cameras (5 min per file)
        with open(str(output_dir/mst.HISTORY_FILE_NAME), 'w') as f:
            for i in range(3):
                f.write(json.dumps({'file': 'old{:d}'.format(i), 'outcome': 'ok', 'processing_min': 5.0,
                                    'finished_utc': mst.RmsDateTime.utcnow().strftime(mnr.JSON_TIME_FORMAT)}) + '\n')

    ini_path = tmp_path/'multicam.ini'
    ini_path.write_text('[Global]\nnproc = 1\nfile_type = vid\npoll_interval = 0.01\n\n' + '\n'.join(
        '[{:s}]\ninput_dir = {:s}\noutput_dir = {:s}\nconfig = {:s}\nplatepar = {:s}\n'.format(
            cam_id, str(input_dir), str(output_dir), str(config_path), TEMPLATE_PLATEPAR)
        for cam_id, input_dir, output_dir, config_path in cameras))

    # The disk is full (nothing is processed, the files wait), and the monitor stops after a few polls
    monkeypatch.setattr(mon.FreeSpaceGuard, 'ok', lambda self: False)
    polls = {'n': 0}
    real_sleep = time.sleep
    def _sleep(seconds):
        polls['n'] += 1
        if polls['n'] > 5:
            raise KeyboardInterrupt
        real_sleep(0.01)
    monkeypatch.setattr(mon.time, 'sleep', _sleep)
    monkeypatch.setattr(mst.MonitorStatus, 'update', _runningUpdate(mst.MonitorStatus.update))

    mon.monitorMultipleCameras(str(ini_path), report_mode='none')

    # The one worker takes the file of CAM2 first (the monitor starts its turn after the first camera), then of
    #   CAM1
    with open(str(cameras[0][2]/mst.STATUS_FILE_NAME)) as f:
        status = json.load(f)
    assert status['processing']['queue_empty_in_min'] == 10.0
    assert not os.path.isfile(str(cameras[1][2]/mst.STATUS_FILE_NAME))


def test_files_searched_by_the_normal_detection(tmp_path, now):
    """ With the normal detection, the files with enough stars were searched (they have no summary of the matched
        filter), only the ones with too few stars were not.
    """

    config = _config()
    status, _ = _status(tmp_path, config)
    output_dir = str(tmp_path/'output')

    # A file with enough stars and one with too few (one star per chunk)
    searched = _makeResults(output_dir, config, 'file1', n_chunks=3)
    calstars = os.path.join(output_dir, searched, 'CALSTARS_file1.txt')
    star_list = CALSTARS.readCALSTARS(os.path.dirname(calstars), os.path.basename(calstars))[0]
    star_list = [[name, [stars[0]]*(config.ff_min_stars + 5)] for name, stars in star_list]
    CALSTARS.writeCALSTARS(star_list, os.path.dirname(calstars), os.path.basename(calstars), 'XX0001', 48, 64,
                           chunk_frames=128, fps=25.0)
    _makeResults(output_dir, config, 'file2', n_chunks=3, start_s=60)

    night = status.build(_state())['night']
    assert (night['files_processed'], night['files_not_searched']) == (2, 1)


def test_files_of_the_night_end_at_sunrise(tmp_path, monkeypatch):
    """ After sunrise (before noon, while it is still the same night), the inputs written after sunrise are not
        counted as written tonight.
    """

    config = _matchedFilterConfig()
    _, night_start, night_end = mnr.nightInfo(config, NOW)
    after_sunrise = night_end + datetime.timedelta(hours=2)
    assert mnr.nightInfo(config, after_sunrise)[0] == mnr.nightInfo(config, NOW)[0]
    monkeypatch.setattr(mst.RmsDateTime, 'utcnow', staticmethod(lambda: after_sunrise))

    status, _ = _status(tmp_path, config)
    for name, t in [('night.vid', night_end - datetime.timedelta(hours=1)),
                    ('day.vid', night_end + datetime.timedelta(hours=1))]:
        path = tmp_path/'input'/name
        path.write_bytes(b'0')
        ts = (t - datetime.datetime(1970, 1, 1)).total_seconds()
        os.utime(str(path), (ts, ts))

    assert status.build(_state())['recording']['files_tonight'] == 1


def test_writes_continue_when_the_clock_is_set_back(tmp_path, now, monkeypatch):
    """ The status is written every interval of the monotonic clock, also when the clock of the system is set
        back after a write.
    """

    status, config = _status(tmp_path)
    clock = {'wall': 1e9, 'mono': 1000.0}
    monkeypatch.setattr(mst.time, 'time', lambda: clock['wall'])
    monkeypatch.setattr(mst.time, 'monotonic', lambda: clock['mono'])
    assert status.update(_state())

    # The clock of the system is set back by an hour, and a minute passes
    clock['wall'] -= 3600 - status.interval
    clock['mono'] += status.interval
    assert status.due()


def test_damaged_history_doesnt_stop_the_monitor(tmp_path, monkeypatch):
    """ A damaged history of processed files (a record with values which aren't numbers) doesn't stop the
        multi-camera monitor or its status.
    """

    input_dir = tmp_path/'input'
    output_dir = tmp_path/'output'
    input_dir.mkdir()
    output_dir.mkdir()
    _makeInput(input_dir, 'dump_6AC5999C_02F.vid', False, minutes_ago=30)
    with open(str(output_dir/mst.HISTORY_FILE_NAME), 'w') as f:
        f.write(json.dumps({'file': 'old', 'outcome': 'ok', 'processing_min': 'abc', 'size_gb': 'x',
                            'finished_utc': mst.RmsDateTime.utcnow().strftime(mnr.JSON_TIME_FORMAT)}) + '\n')
    ini_path = tmp_path/'multicam.ini'
    ini_path.write_text('[Global]\nnproc = 1\nfile_type = vid\npoll_interval = 0.01\n\n[CAM1]\ninput_dir = {:s}\n'
                        'output_dir = {:s}\nconfig = {:s}\nplatepar = {:s}\n'.format(str(input_dir), str(output_dir),
                                                                                   REPO_CONFIG, TEMPLATE_PLATEPAR))

    # The disk is full (the file waits), and the monitor stops after a few polls
    monkeypatch.setattr(mon.FreeSpaceGuard, 'ok', lambda self: False)
    polls = {'n': 0}
    real_sleep = time.sleep
    def _sleep(seconds):
        polls['n'] += 1
        if polls['n'] > 5:
            raise KeyboardInterrupt
        real_sleep(0.01)
    monkeypatch.setattr(mon.time, 'sleep', _sleep)
    monkeypatch.setattr(mst.MonitorStatus, 'due', lambda self: True)

    mon.monitorMultipleCameras(str(ini_path), report_mode='none')

    # The monitor ran until it was stopped, and wrote its status
    assert polls['n'] > 5
    with open(str(output_dir/mst.STATUS_FILE_NAME)) as f:
        assert json.load(f)['monitor']['state'] == 'stopped'


def test_progress_of_files_being_written(tmp_path, now):
    """ The files still being written are counted in the night they are written in. """

    status, config = _status(tmp_path)
    tonight, t1 = _inputFile(tmp_path, 'tonight.vid', 1)
    old, t2 = _inputFile(tmp_path, 'old.vid', 60*24)

    progress = status.build(_state(settling=[(tonight, t1), (old, t2)]))['processing']['progress']
    assert (progress['being_written'], progress['other_nights_open']) == (1, 1)


def test_latest_stars_with_their_time(tmp_path, now):
    """ When the latest state of the sky comes from a chunk which is not the latest one (files with different
        lengths of the chunks), the stars are given with the time of that chunk.
    """

    status, config = _status(tmp_path)
    output_dir = str(tmp_path/'output')
    _searchedFile(output_dir, config, 'file1', n_chunks=12, clear=0.0, fps=12.5)
    _searchedFile(output_dir, config, 'file2', n_chunks=12, start_s=1, clear=1.0)

    sky = status.build(_state())['sky']
    assert sky['state'] == 'clouded'
    assert sky['stars_utc'] == mst._timeString(BEG_TIME + datetime.timedelta(seconds=11*128/25.0))


def test_calibration_after_a_failed_sky_scan(tmp_path, now, monkeypatch):
    """ When the conditions of the sky could not be read, the calibration is judged again once they can. """

    status, config = _status(tmp_path)
    _searchedFile(str(tmp_path/'output'), config, 'file1')

    # The conditions can't be read once
    read_rows = mst.nightConditionRows
    calls = {'n': 0}

    def _failOnce(*args, **kwargs):
        calls['n'] += 1
        if calls['n'] == 1:
            raise OSError("The log is being written")
        return read_rows(*args, **kwargs)

    monkeypatch.setattr(mst, 'nightConditionRows', _failOnce)

    assert status.build(_state())['calibration']['status'] == 'unknown'
    assert status.build(_state())['calibration']['status'] == 'ok'


def test_queue_of_a_camera_paused_for_space(tmp_path, now):
    """ While the camera is paused for disk space, its waiting files can't start, so it can't tell when they
        will be done.
    """

    status, config = _statusWithSizes(tmp_path, {'waiting.vid': 1.0})
    for i in range(3):
        status.fileFinished('done{:d}'.format(i), 'ok', 60.0, input_name='done.vid', size_bytes=1024**3)
    waiting, t1 = _inputFile(tmp_path, 'waiting.vid', 5)

    processing = status.build(_state(paused_for_space=True, queued=[(waiting, t1)]))['processing']
    assert processing['queued'] == 1 and processing['queue_empty_in_min'] is None


@pytest.mark.parametrize('dropped', ['many', 2.5, True, [3]])
def test_damaged_summary_of_the_matched_filter(tmp_path, now, dropped):
    """ A damaged number of the dropped candidates in the summary of the matched filter doesn't stop the status. """

    status, config = _status(tmp_path)
    results_dir = _searchedFile(str(tmp_path/'output'), config, 'file1')
    summary_path = os.path.join(str(tmp_path/'output'), results_dir, mnr.MF_DONE_NAME)
    with open(summary_path) as f:
        summary = json.load(f)
    summary['tracks_dropped'] = dropped
    with open(summary_path, 'w') as f:
        json.dump(summary, f)

    built = status.build(_state())
    assert built['detector']['candidates_dropped_last_file'] is None
    assert 'CANDIDATES_DROPPED' not in [fault['code'] for fault in built['faults']]


def test_caches_keep_only_the_open_files(tmp_path, now):
    """ The sizes and nights of the inputs are only kept while the inputs wait, are processed or are written. """

    status, config = _statusWithSizes(tmp_path, {'a.vid': 1.0, 'b.vid': 1.0, 'c.vid': 1.0})
    a, ta = _inputFile(tmp_path, 'a.vid', 5)
    b, tb = _inputFile(tmp_path, 'b.vid', 3)
    c, tc = _inputFile(tmp_path, 'c.vid', 1)

    status.build(_state(queued=[(a, ta), (b, tb)]))
    status.build(_state(in_progress=[('b', b, NOW)], settling=[(c, tc)]))

    assert {key[0] for key in status.size_cache} <= {b, c}
    assert {key[0] for key in status.night_cache} == {b, c}


def test_calibration_of_a_file_processed_again_after_a_failed_sky_scan(tmp_path, now, monkeypatch):
    """ When a file was processed again and the new conditions could not be read, the calibration is judged
        again once they can, not on the conditions of the earlier processing.
    """

    status, config = _status(tmp_path)
    output_dir = str(tmp_path/'output')
    results_dir = _searchedFile(output_dir, config, 'file1')
    assert status.build(_state())['calibration']['status'] == 'ok'

    # Processed again on a clouded sky, not recalibrated (the new conditions are processed later)
    shutil.rmtree(os.path.join(output_dir, results_dir))
    _searchedFile(output_dir, config, 'file1', clear=0.0, recalibrated=False)
    log_path = os.path.join(output_dir, 'sky_conditions_XX0001_{:s}.csv'.format(BEG_TIME.strftime('%Y%m%d')))
    with open(log_path) as f:
        lines = [line.replace('2025-12-25T04:00:00', '2025-12-25T05:00:00') if ',0.000,0.020,' in line else line
                 for line in f]
    with open(log_path, 'w') as f:
        f.writelines(lines)
    done_flag = os.path.join(output_dir, results_dir, 'done.flag')
    os.utime(done_flag, (os.path.getmtime(done_flag) + 10, os.path.getmtime(done_flag) + 10))

    # The new conditions can't be read once
    read_rows = mst.nightConditionRows
    calls = {'n': 0}

    def _failOnce(*args, **kwargs):
        calls['n'] += 1
        if calls['n'] == 1:
            raise OSError("The log is being written")
        return read_rows(*args, **kwargs)

    monkeypatch.setattr(mst, 'nightConditionRows', _failOnce)

    status.build(_state())
    assert status.build(_state())['calibration']['status'] == 'unknown'


@pytest.mark.parametrize('replace', [True, False])
def test_summary_of_the_matched_filter_when_it_is_off(tmp_path, replace):
    """ The summary of an earlier run of the matched filter is not used when the matched filter is off (e.g. the
        file was processed again with the normal detection only).
    """

    config = _matchedFilterConfig()
    config.mf_replace_detection = replace
    results_path = os.path.join(str(tmp_path), _searchedFile(str(tmp_path), config, 'file1'))
    if not replace:
        os.makedirs(os.path.join(results_path, mnr.MF_DIR_NAME))
        shutil.copy(os.path.join(results_path, mnr.MF_DONE_NAME), os.path.join(results_path, mnr.MF_DIR_NAME))
    assert mnr.readFileSkyResults(results_path, config)['searched']

    config.mf_enable = False
    file_result = mnr.readFileSkyResults(results_path, config)
    assert (file_result['searched'], file_result['measured'], file_result['detections']) == (False, False, 0)
    assert file_result['star_chunks']


def test_progress_of_files_processed_again(tmp_path, now):
    """ A done file which is processed again (e.g. with --force) counts as open, not also as done. """

    status, config = _status(tmp_path)
    _searchedFile(str(tmp_path/'output'), config, 'file1')
    _searchedFile(str(tmp_path/'output'), config, 'file2', start_s=62)
    _, mtime = _inputFile(tmp_path, 'file1.vid', 3)

    progress = status.build(_state(queued=[('/data/file1.vid', mtime)]))['processing']['progress']
    assert (progress['done_tonight'], progress['queued'], progress['percent_done']) == (1, 1, 50.0)


def test_damaged_bytes_in_the_history(tmp_path, now):
    """ A history with bytes which are not text or with numbers too large to use doesn't stop the monitor from
        starting, and its other lines are kept.
    """

    (tmp_path/'output').mkdir()
    good = {'file': 'good', 'outcome': 'ok', 'processing_min': 5.0,
            'finished_utc': NOW.strftime(mnr.JSON_TIME_FORMAT)}
    with open(str(tmp_path/'output'/mst.HISTORY_FILE_NAME), 'wb') as f:
        f.write(b'\xff\xfe{"damaged": \x80}\n')
        f.write(('{"file": "huge", "outcome": "ok", "processing_min": 1' + '0'*400 + ', "finished_utc": "'
                 + NOW.strftime(mnr.JSON_TIME_FORMAT) + '"}\n').encode())
        f.write((json.dumps(good) + '\n').encode())

    status, config = _status(tmp_path)
    assert [r['file'] for r in status.file_records] == ['huge', 'good']
    assert status.file_records[0]['processing_min'] is None


def test_failed_write_waits_for_the_interval(tmp_path, now, monkeypatch):
    """ When the status can't be written, it is tried again after the interval, not at every poll. """

    status, config = _status(tmp_path)
    builds = {'n': 0}
    real_build = status.build

    def _build(state):
        builds['n'] += 1
        return real_build(state)

    monkeypatch.setattr(status, 'build', _build)
    monkeypatch.setattr(mst, '_writeJSON', lambda *args: (_ for _ in ()).throw(OSError(28, 'No space left')))

    for _ in range(5):
        status.update(_state())
    assert builds['n'] == 1 and not status.due()


def test_stopped_monitor_has_no_queue_end(tmp_path, now):
    """ When the monitor has stopped, its waiting files are not processed, so no end of the queue is given. """

    status, config = _statusWithSizes(tmp_path, {'waiting.vid': 1.0})
    for i in range(3):
        status.fileFinished('done{:d}'.format(i), 'ok', 60.0, input_name='done.vid', size_bytes=1024**3)
    waiting, t1 = _inputFile(tmp_path, 'waiting.vid', 5)

    for state in (_state(stopped=True, queued=[(waiting, t1)]),
                  _state(stopped=True, queued=[(waiting, t1)], queue_empty_min=1.0)):
        processing = status.build(state)['processing']
        assert processing['queued'] == 1 and processing['queue_empty_in_min'] is None
        assert processing['queue_empty_utc'] is None


def test_sky_without_measurements_after_a_file_with_too_few_stars(tmp_path, now):
    """ Without the measured sky (e.g. with the normal detection), a file with too few stars early in the night
        doesn't give the state of the sky after later files with enough stars.
    """

    config = _config()
    config.mf_enable = False
    status, _ = _status(tmp_path, config)
    night_start, night_end = NOW - datetime.timedelta(hours=5), NOW + datetime.timedelta(hours=7)

    def _fileResult(start, stars):
        return {'star_chunks': [(start + datetime.timedelta(seconds=10*i), stars) for i in range(60)],
                'chunk_s': 10.0, 'summary': None, 'searched': False, 'measured': False, 'removed_times': None,
                'removed_count': 0, 'detections': 0, 'ftp_detections': 0}

    files = [('a', {'input_file': '/in/a.vid'}, _fileResult(NOW - datetime.timedelta(hours=4), 2), 10),
             ('b', {'input_file': '/in/b.vid'}, _fileResult(NOW - datetime.timedelta(minutes=15), 150), 10)]
    _, sky = status._sky(NOW, 'night', night_start, night_end, files)
    assert (sky['state'], sky['observable'], sky['stars']) == ('no_data', None, 150)


def test_camera_without_the_status_keeps_no_sizes(tmp_path, now):
    """ A camera without the status (whose status is never made) only keeps the sizes of its open files, which
        the prediction of the other cameras asks for.
    """

    status, config = _statusWithSizes(tmp_path, {'a.vid': 1.0, 'b.vid': 1.0})
    status.enabled = False
    a, ta = _inputFile(tmp_path, 'a.vid', 5)
    b, tb = _inputFile(tmp_path, 'b.vid', 3)

    status.openWork(_state(queued=[(a, ta)]))
    status.openWork(_state(queued=[(b, tb)]))
    assert {key[0] for key in status.size_cache} == {b}


@pytest.mark.skipif(hasattr(os, 'geteuid') and os.geteuid() == 0, reason="root can read any file")
def test_summary_of_the_matched_filter_which_cant_be_opened(tmp_path, now):
    """ A summary of the matched filter which can't be opened makes only its file unreadable, the other files of
        the night are still reported.
    """

    status, config = _status(tmp_path)
    output_dir = str(tmp_path/'output')
    results_dir = _searchedFile(output_dir, config, 'file1')
    _searchedFile(output_dir, config, 'file2', start_s=62)
    summary_path = os.path.join(output_dir, results_dir, mnr.MF_DONE_NAME)
    os.chmod(summary_path, 0)
    try:
        assert mnr.readFileSkyResults(os.path.join(output_dir, results_dir), config) is None
        built = status.build(_state())
        assert built['night']['files_processed'] == 2 and built['night']['files_unreadable'] == 1
        assert 'STATUS_INCOMPLETE' not in [fault['code'] for fault in built['faults']]
    finally:
        os.chmod(summary_path, 0o644)


def test_sky_with_a_newer_file_which_was_not_measured(tmp_path, now):
    """ A file which was not measured but had enough stars, which ends after the measured data (here it begins
        1 s later), makes the sky now unknown.
    """

    status, config = _status(tmp_path)
    output_dir = str(tmp_path/'output')
    _searchedFile(output_dir, config, 'file1', clear=1.0)

    # The newer file has enough stars, but its sky was not measured (e.g. the matched filter failed)
    results_path = os.path.join(output_dir, _makeResults(output_dir, config, 'file2', n_chunks=12, start_s=1))
    star_list, _ = CALSTARS.readCALSTARS(results_path, 'CALSTARS_file2.txt')
    star_list = [[ff_name, stars*(config.ff_min_stars + 5)] for ff_name, stars in star_list]
    CALSTARS.writeCALSTARS(star_list, results_path, 'CALSTARS_file2.txt', 'XX0001', 48, 64, chunk_frames=128,
                           fps=25.0)

    sky = status.build(_state())['sky']
    assert (sky['state'], sky['observable']) == ('no_data', None)


def test_sky_until_the_end_of_the_input(tmp_path, now):
    """ The time of the sky includes the frames after the last chunk of the input (fewer than a chunk), from the
        number of frames in the summary of the matched filter.
    """

    status, config = _status(tmp_path)
    results_dir = _searchedFile(str(tmp_path/'output'), config, 'file1', n_chunks=12)

    # 20 frames after the last chunk
    summary_path = os.path.join(str(tmp_path/'output'), results_dir, mnr.MF_DONE_NAME)
    with open(summary_path) as f:
        summary = json.load(f)
    summary['total_frames'] += 20
    with open(summary_path, 'w') as f:
        json.dump(summary, f)

    end = BEG_TIME + datetime.timedelta(seconds=(12*128 + 20)/25.0)
    assert mnr.readFileSkyResults(os.path.join(str(tmp_path/'output'), results_dir), config)['end_time'] == end
    assert status.build(_state())['sky']['observed_utc'] == mst._timeString(end)


def test_sky_with_a_newer_file_which_was_not_measured_until_its_end(tmp_path, now):
    """ A file which was searched but whose sky was not measured, which ends after the measured data only with
        the frames after its last chunk, makes the sky now unknown.
    """

    status, config = _status(tmp_path)
    output_dir = str(tmp_path/'output')

    def _summary(results_dir, **changes):
        path = os.path.join(output_dir, results_dir, mnr.MF_DONE_NAME)
        with open(path) as f:
            summary = json.load(f)
        summary.update(changes)
        with open(path, 'w') as f:
            json.dump(summary, f)

    # Both files have 20 frames after their last chunks, the second one begins 0.5 s later and was not measured
    measured = _searchedFile(output_dir, config, 'file1', n_chunks=12)
    _summary(measured, total_frames=12*128 + 20)
    unmeasured = _searchedFile(output_dir, config, 'file2', n_chunks=12, start_s=0.5)
    _summary(unmeasured, total_frames=12*128 + 20, clear_sky_fraction=None)

    assert status.build(_state())['sky']['state'] == 'no_data'


def test_latest_sky_values_until_the_end_of_the_input(tmp_path, now, monkeypatch):
    """ The values of the latest sky come from the chunk which gave its state also when that state is in the
        frames after the last chunk of its input (here a clear input until 7 s, and a clouded one from 1 to 5 s).
    """

    status, config = _status(tmp_path)
    t0 = NOW - datetime.timedelta(minutes=10)

    def _file(name, start, end_time, clear):
        result = {'star_chunks': [(start, 500)], 'chunk_s': 4.0, 'summary': {'total_frames': 1},
                  'searched': True, 'measured': True, 'removed_times': [], 'removed_count': 0, 'detections': 0,
                  'ftp_detections': 0, 'end_time': end_time}
        row = {'time': start, 'input': name, 'clear_fraction': clear, 'transparency_mag': 0.02 if clear else np.nan,
               'reliable_seen': 1.0, 'stars': 500, 'matched': 300}
        return (name, {'input_file': name}, result, 1.0), row

    clear_file, clear_row = _file('a.vid', t0, t0 + datetime.timedelta(seconds=7), 1.0)
    clouded_file, clouded_row = _file('b.vid', t0 + datetime.timedelta(seconds=1), None, 0.0)
    monkeypatch.setattr(status, '_conditionRows', lambda *args: [clear_row, clouded_row])

    _, sky = status._sky(NOW, 'night', NOW - datetime.timedelta(hours=5), NOW + datetime.timedelta(hours=5),
                         [clear_file, clouded_file])
    assert (sky['state'], sky['clear_percent']) == ('clear', 100.0)


def test_end_of_the_input_from_the_summary(tmp_path, now):
    """ The end of the input is taken from the summary of the matched filter. Without it, it comes from the frame
        of the last chunk in its name, which is not known if the chunks are not named by their frames (e.g. when
        the chunk images are not saved).
    """

    config = _matchedFilterConfig()
    output_dir = str(tmp_path)
    results_path = os.path.join(output_dir, _searchedFile(output_dir, config, 'file1', n_chunks=12))
    summary_path = os.path.join(results_path, mnr.MF_DONE_NAME)
    with open(summary_path) as f:
        summary = json.load(f)

    # The chunks named without their frames
    star_list, _ = CALSTARS.readCALSTARS(results_path, 'CALSTARS_file1.txt')
    star_list = [[FFfile.constructFFName('XX0001', FFfile.filenameToDatetime(name), ext='fits'), stars]
                 for name, stars in star_list]
    CALSTARS.writeCALSTARS(star_list, results_path, 'CALSTARS_file1.txt', 'XX0001', 48, 64, chunk_frames=128,
                           fps=25.0)
    assert mnr.readFileSkyResults(results_path, config)['end_time'] is None

    # The end in the summary
    end = BEG_TIME + datetime.timedelta(seconds=63.0)
    summary['input_end_utc'] = end.strftime('%Y-%m-%dT%H:%M:%S.%f')
    with open(summary_path, 'w') as f:
        json.dump(summary, f)
    assert mnr.readFileSkyResults(results_path, config)['end_time'] == end


def test_status_reports_the_automatic_calibration(tmp_path):
    """ With the automatic calibration, the status has its judgement of the latest file, the last installed platepar
        and whether a candidate waits for the confirmation. """

    import RMS.Astrometry.AutoCalibration as ac

    # A camera with the automatic calibration and two processed files
    config = _config()
    config.monitor_auto_calibration = True
    output_dir = tmp_path/'output'
    os.makedirs(str(output_dir/'2025'/'f1'))
    os.makedirs(str(output_dir/'2025'/'f2'))

    status = mst.MonitorStatus('XX0001', str(output_dir), str(tmp_path/'input'), config, lambda: [], 2)

    # One installed platepar and a candidate of a dry run, and the judgements of the files (the platepar failed on
    #   the later one)
    state = ac.readState(str(output_dir))
    state['installs'].append({'jd': 2461000.5, 'path': 'auto.cal', 'reason': 'there is no platepar'})
    state['installs'].append({'jd': 2461000.6, 'path': 'auto_candidate.cal', 'reason': 'better', 'dry_run': True})
    ac.writeState(str(output_dir), state)
    for name, ok in [('f1', True), ('f2', False)]:
        with open(str(output_dir/'2025'/name/ac.FILE_RESULT_NAME), 'w') as f:
            json.dump({'action': 'judged', 'current': {'ok': ok, 'reasons': [] if ok else ['zone r75-100']}}, f)

    # The summary of the processed files
    files = [(os.path.join('2025', name), {}, {}, None) for name in ['f1', 'f2']]
    summary = status._autoCalibration(files)

    # The installed platepar, the candidate of the dry run, no pending candidate or platepar on trial, and the
    #   judgement of the latest file
    assert summary['installs'] == 1
    assert summary['last_install'] == {'path': 'auto.cal', 'reason': 'there is no platepar'}
    assert summary['dry_run_candidates'] == 1
    assert summary['pending'] is False
    assert summary['trial'] is False
    assert summary['latest']['results_dir'] == os.path.join('2025', 'f2')
    assert summary['latest']['platepar_ok'] is False
