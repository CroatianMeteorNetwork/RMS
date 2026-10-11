"""
Tests of the live status of the monitor (RMS.MonitorStatus): the recording, the processing, the sky and the night,
the calibration and the faults in the status of synthetic results, the status file itself, and the status written
by the monitor loop.
"""

import datetime
import json
import os
import time

import numpy as np
import pytest

import RMS.MonitorNightReport as mnr
import RMS.MonitorProcessFrameInterface as mon
import RMS.MonitorStatus as mst
from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, xyToRaDecPP
from RMS.Formats import FFfile
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
                  recalibrated=True, shift_deg=0.0):
    """ Results of a file which the matched filter searched, with its conditions in the log and its recalibrated
        platepars (fitted exactly on the stars, the pointing shifted by shift_deg in altitude).
    """

    results_dir = _makeResults(output_dir, config, name, n_chunks=n_chunks, start_s=start_s)
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

    # With the files of another camera waiting as well, this camera gets one of the two workers
    processing = status.build(_state(in_progress=[('run', run_path, start)], queued=queued,
                                     other_cameras_files=4))['processing']
    assert processing['queue_empty_in_min'] == 1.5 + 1 + 1 + 4


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
