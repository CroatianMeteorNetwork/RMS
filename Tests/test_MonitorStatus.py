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
    return mst.MonitorStatus('XX0001', str(output_dir), str(input_dir), config,
                             lambda name: name.endswith('.vid'), 2), config


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
    assert night['files_processed'] == 3 and night['detections'] == 4 and night['removed_by_clouds'] == 1
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
