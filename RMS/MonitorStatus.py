""" Live status of the continuous processing (RMS.MonitorProcessFrameInterface) for external monitoring.

The monitor keeps a JSON file in the output directory of every camera (monitor_status.json) which tells what the
processing is doing right now: whether the input is being recorded, whether the processing keeps up, how the sky
is (from the stars of the latest processed data), how the night is going, how healthy the calibration is, and any
faults. It is rewritten every minute and after every processed file, and once more when the monitor stops, so an
external process (e.g. an online monitor of the station) only needs to read it. The fields and the fault codes
are described in Guides/MonitorProcessing.md.

The status is written by the main process of the monitor, so this module must not import the matched filter
(importing it initializes CUDA, which can't be used by the worker processes forked from the monitor).
"""

from __future__ import print_function, division, absolute_import

import datetime
import json
import os
import platform
import time

import numpy as np

from RMS.DeleteOldObservations import availableSpace
from RMS.Formats.FFfile import filenameToDatetime
from RMS.Formats.FTPdetectinfo import validDefaultFTPdetectinfo
from RMS.Formats.Platepar import Platepar
from RMS.Logger import getLogger
from RMS.Misc import RmsDateTime, UTCFromTimestamp
from RMS.MonitorNightReport import (ESSENTIAL_STEPS, JSON_TIME_FORMAT, _writeJSON, latestPlateparPath, nightInfo,
                                    nightConditionRows, nightSkyResults, plateparResidual, readFileSkyResults,
                                    readReportStates, scanNights)


log = getLogger("logger")


# Name of the status file in the output directory, and the version of its format (increased when fields are
#   changed or removed, not when they are added)
STATUS_FILE_NAME = 'monitor_status.json'
SCHEMA_VERSION = 1

# Default time between the writes of the status (s)
STATUS_INTERVAL = 60.0

# Time without a growing input file after which the input is not being recorded (min)
NOT_RECORDING_MIN = 5.0

# The processing doesn't keep up when the oldest waiting file was recorded this many file lengths ago
BEHIND_FILE_LENGTHS = 3.0

# The sky is observable when this share of the observed time in the last minutes of the data had at least this
#   share of the image clear
OBSERVABLE_WINDOW_MIN = 10.0
OBSERVABLE_SHARE = 0.8
OBSERVABLE_CLEAR_FRACTION = 0.8

# The calibration is failing when fewer than this share of the chunks on clear sky could be recalibrated (which
#   already requires the stars to match within config.dist_check_threshold), judged on at least this many chunks
CALIBRATION_MIN_SHARE = 0.5
CALIBRATION_MIN_CHUNKS = 10

# The camera has moved when its pointing differs from the reference platepar by more than this (px). The pointing
#   normally changes by a fraction of a pixel through the night (refraction, temperature)
CAMERA_MOVED_PX = 2.0

# Number of the latest processed files kept in the status (for the statistics after a restart)
RECENT_FILES = 50

# Severities of the faults, from the least to the most severe
SEVERITIES = ['info', 'warning', 'error']



def _timeString(t):
    """ Time as a string in the status (UTC, ISO format), None if not given. """

    return t.strftime(JSON_TIME_FORMAT) if t is not None else None


def _parseTime(text):
    """ Time from a string of the status, None if it can't be read. """

    try:
        return datetime.datetime.strptime(text, JSON_TIME_FORMAT)
    except (ValueError, TypeError):
        return None


def _round(value, digits=2):
    """ Round a number for the status, None if it is not finite. """

    if (value is None) or (not np.isfinite(value)):
        return None

    return round(float(value), digits)


def _percentiles(values):
    """ Statistics of the given values for the status: the number, the minimum, the 10th, 25th, 50th, 75th and
        90th percentile and the maximum. """

    values = np.array([v for v in values if (v is not None) and np.isfinite(v)], dtype=np.float64)
    if not len(values):
        return {'n': 0}

    p10, p25, p50, p75, p90 = np.percentile(values, [10, 25, 50, 75, 90])

    return {'n': int(len(values)), 'min': _round(values.min()), 'p10': _round(p10), 'p25': _round(p25),
            'median': _round(p50), 'p75': _round(p75), 'p90': _round(p90), 'max': _round(values.max())}


def _codeVersion():
    """ Commit of the code which runs, None if it is not a git repository. """

    try:
        import git
        repo = git.Repo(os.path.dirname(os.path.abspath(__file__)), search_parent_directories=True)
        return repo.head.object.hexsha

    except Exception:
        return None



def fileDetections(results_path):
    """ The number of the detections of a processed file, from the header of its FTPdetectinfo (the detections
        of the normal detection, or of the matched filter when it replaces it).

    Arguments:
        results_path: [str] Results directory of the file.

    Return:
        [int] Number of the detections, None if there is no FTPdetectinfo or it can't be read.
    """

    detections = None
    for file_name in os.listdir(results_path):
        if not validDefaultFTPdetectinfo(file_name):
            continue

        # The first line is "Meteor Count = N"
        try:
            with open(os.path.join(results_path, file_name)) as f:
                detections = (detections or 0) + int(f.readline().split('=')[1])
        except (OSError, ValueError, IndexError):
            continue

    return detections


def fileCalibration(results_path, config, clear_times=None):
    """ How well the chunks of a processed file could be calibrated, from the platepars which were recalibrated on
        the stars of every chunk (ApplyRecalibrate).

    Arguments:
        results_path: [str] Results directory of the file.
        config: [Config] Configuration.

    Keyword arguments:
        clear_times: [list] Times of the chunks on clear sky. None if the sky is not known, then all chunks are
            counted as clear.

    Return:
        calibration: [dict] chunks, recalibrated, clear_chunks, clear_recalibrated, residuals_px (median distance
            between the image and the catalog stars after the fit, of every recalibrated chunk), matched_stars,
            zero_points (photometric zero points, mag) and shifts_px (difference of the pointing from the
            reference platepar of the file). None if there are no recalibrated platepars.
    """

    recalibrated_path = os.path.join(results_path, config.platepars_recalibrated_name)
    if not os.path.isfile(recalibrated_path):
        return None

    try:
        with open(recalibrated_path) as f:
            recalibrated = json.load(f)
    except (OSError, ValueError) as e:
        log.warning("The recalibrated platepars {:s} could not be read: {:s}".format(recalibrated_path, repr(e)))
        return None

    # The reference platepar of the file (the one the processing started with)
    reference = None
    reference_path = os.path.join(results_path, config.platepar_name)
    if os.path.isfile(reference_path):
        try:
            reference = Platepar()
            reference.read(reference_path)
        except Exception:
            reference = None

    calibration = {'chunks': 0, 'recalibrated': 0, 'clear_chunks': 0, 'clear_recalibrated': 0, 'residuals_px': [],
                   'matched_stars': [], 'zero_points': [], 'shifts_px': []}

    for ff_name, pp_dict in recalibrated.items():

        # Skip the chunks which can't be read
        if not isinstance(pp_dict, dict):
            continue

        # The chunk is on clear sky if its time is among the clear chunks (all are clear if the sky is not known)
        try:
            chunk_time = filenameToDatetime(ff_name)
        except Exception:
            chunk_time = None
        clear = (clear_times is None) or ((chunk_time is not None) and any(abs((chunk_time - t).total_seconds())
                                                                             < 0.5 for t in clear_times))

        recalibrated_ok = bool(pp_dict.get('auto_recalibrated'))
        calibration['chunks'] += 1
        calibration['recalibrated'] += int(recalibrated_ok)
        calibration['clear_chunks'] += int(clear)
        calibration['clear_recalibrated'] += int(clear and recalibrated_ok)

        # The quality of the fit is only known for the recalibrated chunks
        if not recalibrated_ok:
            continue

        try:
            platepar = Platepar()
            platepar.loadFromDict(pp_dict)

            # The residual of the fit and the number of the stars it used
            if platepar.star_list:
                calibration['residuals_px'].append(plateparResidual(platepar))
                calibration['matched_stars'].append(len(platepar.star_list))

            calibration['zero_points'].append(float(platepar.mag_lev))

            # The difference of the pointing from the reference platepar, in pixels (the plate scale is in px/deg)
            if reference is not None:
                d_az = (platepar.az_centre - reference.az_centre + 180)%360 - 180
                d_alt = platepar.alt_centre - reference.alt_centre
                shift_deg = np.hypot(d_az*np.cos(np.radians(reference.alt_centre)), d_alt)
                calibration['shifts_px'].append(float(shift_deg*reference.F_scale))

        except Exception as e:
            log.debug("The calibration of {:s} could not be read: {:s}".format(ff_name, repr(e)))

    return calibration



class MonitorStatus(object):
    def __init__(self, camera_id, output_dir, input_dir, config, input_scanner, nproc):
        """ Keeps the status file of a camera of the monitor up to date.

        Arguments:
            camera_id: [str] Name of the camera (its station code in a single-camera monitor).
            output_dir: [str] Output directory of the camera, where the status file is written.
            input_dir: [str] Input directory of the camera.
            config: [Config] Configuration of the camera.
            input_scanner: [callable] Function which returns the inputs as the monitor finds them, as (path,
                mtime, size), see MonitorProcessFrameInterface.scanInputs. It raises OSError if the input
                directory can't be read.
            nproc: [int] Number of the worker processes of the monitor.
        """

        self.camera_id = camera_id
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.config = config
        self.input_scanner = input_scanner
        self.nproc = nproc

        self.path = os.path.join(output_dir, STATUS_FILE_NAME)
        self.interval = getattr(config, 'monitor_status_interval', STATUS_INTERVAL)

        # Information about the monitor which doesn't change
        self.started = RmsDateTime.utcnow()
        self.pid = os.getpid()
        self.host = platform.node()
        self.code_version = _codeVersion()

        # The processed files (processing time, length of the file), the faults with the time they were first
        #   seen, and the caches of the results of the files and of the conditions
        self.file_records = []
        self.fault_since = {}
        self.scan_cache = {}
        self.file_cache = {}
        self.conditions_cache = (None, [])

        self.last_write = 0.0
        self.dirty = True

        # Continue the statistics and the faults of a previous run of the monitor
        self._restore()


    def _restore(self):
        """ Take the processed files and the times of the faults from the status of a previous run. """

        if not os.path.isfile(self.path):
            return

        try:
            with open(self.path) as f:
                previous = json.load(f)

            self.file_records = list(previous.get('processing', {}).get('recent_files', []))[-RECENT_FILES:]

            for fault in previous.get('faults', []):
                since = _parseTime(fault.get('since_utc'))
                if since is not None:
                    self.fault_since[fault.get('code')] = since

        except Exception as e:
            log.debug("The previous status {:s} could not be read: {:s}".format(self.path, repr(e)))


    def fileFinished(self, unique_id, outcome, processing_s):
        """ Record a processed file. The status is written at the next update.

        Arguments:
            unique_id: [str] Unique ID of the file (the name of its results directory).
            outcome: [str] 'ok', 'failed', 'skipped' or 'changed' (still being written, processed again).
            processing_s: [float] Processing time (s).
        """

        self.file_records.append({'file': unique_id, 'outcome': outcome,
                                  'finished_utc': _timeString(RmsDateTime.utcnow()),
                                  'processing_min': _round(processing_s/60.0), 'length_min': None})
        self.file_records = self.file_records[-RECENT_FILES:]
        self.dirty = True


    def update(self, state, force=False):
        """ Write the status if it is due (every interval, right after a file was processed, or if forced). An
            error is only logged, as the status must never stop the processing.

        Arguments:
            state: [dict] State of the monitor for the camera, see build.

        Keyword arguments:
            force: [bool] Write the status now. False by default.

        Return:
            [bool] True if the status was written.
        """

        now = time.time()
        if not (force or self.dirty or (now - self.last_write >= self.interval)):
            return False

        try:
            _writeJSON(self.path, self.build(state))
            self.last_write = now
            self.dirty = False
            return True

        except Exception as e:
            log.warning("[{}] The status could not be written: {}".format(self.camera_id, repr(e)))
            return False


    ### Sections of the status ###

    def build(self, state):
        """ Put together the status of the camera.

        Arguments:
            state: [dict] State of the monitor for the camera:
                - stopped: [bool] The monitor has stopped.
                - input_readable: [bool] The input directory could be read at the last scan (None if not known).
                - paused_for_space: [bool] The processing is paused as the output disk is full.
                - workers_busy: [int] Busy worker processes of the whole monitor.
                - in_progress: [list] Files being processed, as (unique_id, file_path, start time (datetime)).
                - queued: [list] Files waiting to be processed, as (file_path, mtime).
                - settling: [list] Files which are still being written.
                - failed_files: [dict] The failed files of the camera (see loadFailedFiles).
                - max_failures, max_kills: [int] Failures and kills after which a file is given up.
                - reporter: [NightReporter] Night reporter of the camera, None if there is none.

        Return:
            status: [dict] The status.
        """

        now = RmsDateTime.utcnow()

        # The night which is current or comes next, and the results of its files
        night_name, night_start, night_end = nightInfo(self.config, now)
        files = self._nightFiles(night_name, night_start, night_end)

        status = {
            'schema_version': SCHEMA_VERSION,
            'camera': self.camera_id,
            'station': self.config.stationID,
            'written_utc': _timeString(now),
            'interval_s': self.interval,
        }

        # Put together the sections. A section which fails is reported as failed, so the rest of the status is
        #   still written
        failed_sections = []

        def _section(name, func, default):
            try:
                return func()
            except Exception as e:
                log.warning("[{}] The {} of the status could not be made: {}".format(self.camera_id, name,
                                                                                    repr(e)))
                failed_sections.append(name)
                return default

        status['monitor'] = self._monitor(state)
        status['recording'] = _section('recording', lambda: self._recording(state, now, night_start, night_end),
                                       {'state': 'unknown', 'input_readable': state.get('input_readable')})
        status['processing'] = _section('processing', lambda: self._processing(state, now, files),
                                        {'not_keeping_up': False, 'failed_files': [], 'given_up': []})

        night, sky = _section('sky', lambda: self._sky(now, night_name, night_start, night_end, files),
                              ({'name': night_name}, {'state': 'no_data', 'observable': None}))
        status['sky'] = sky
        status['night'] = night

        status['calibration'] = _section('calibration', lambda: self._calibration(files), {'status': 'unknown'})
        status['detector'] = _section('detector', lambda: self._detector(files),
                                      {'gpu_used_last_file': None, 'gpu_configured': None,
                                       'candidates_dropped_last_file': None})
        status['reports'] = _section('reports', lambda: self._reports(state), {'failed_steps': []})
        status['disk'] = _section('disk', lambda: self._disk(state),
                                  {'paused_for_space': bool(state.get('paused_for_space'))})

        # The faults, from the most severe, and the overall state
        status['faults'] = self._faults(status, state, now, failed_sections)
        severities = [SEVERITIES.index(fault['severity']) for fault in status['faults']]
        worst = max(severities) if severities else 0
        status['overall'] = {2: 'error', 1: 'warning'}.get(worst, 'ok')

        return status


    def _monitor(self, state):
        """ The monitor itself: running or stopped, and its workers. """

        if state.get('stopped'):
            monitor_state = 'stopped'
        elif state.get('paused_for_space'):
            monitor_state = 'paused_disk_full'
        else:
            monitor_state = 'running'

        return {'state': monitor_state, 'pid': self.pid, 'host': self.host, 'started_utc': _timeString(self.started),
                'code_version': self.code_version, 'nproc': self.nproc,
                'workers_busy': state.get('workers_busy', 0),
                'workers_this_camera': len(state.get('in_progress', []))}


    def _recording(self, state, now, night_start, night_end):
        """ Whether the input is being recorded, from the newest input file growing.

        The recording is done by other software, which RMS doesn't communicate with (e.g. it doesn't record on
            cloudy nights by its own decision), so the state of the recording is only reported, never as a fault.
            TODO: when RMS records the data itself, the recording can report its own state here.
        """

        recording = {'input_dir': self.input_dir, 'input_readable': state.get('input_readable')}

        # Find the newest input (as the monitor finds them, also FITS directories), and count the inputs which
        #   were written tonight
        newest, newest_mtime, newest_size, files_tonight = None, None, None, 0
        try:
            for path, mtime, size in self.input_scanner():
                mtime = UTCFromTimestamp.utcfromtimestamp(mtime)
                if night_start <= mtime <= now:
                    files_tonight += 1
                if (newest_mtime is None) or (mtime > newest_mtime):
                    newest, newest_mtime, newest_size = os.path.basename(path), mtime, size

        except OSError:
            recording['input_readable'] = False

        recording['newest_file'] = newest
        recording['newest_file_written_utc'] = _timeString(newest_mtime)
        recording['newest_file_size_gb'] = _round(newest_size/1024**3, 3) if newest_size is not None else None
        recording['files_tonight'] = files_tonight
        minutes_since = (now - newest_mtime).total_seconds()/60.0 if newest_mtime is not None else None
        recording['minutes_since_last_write'] = _round(minutes_since, 1)

        # Recording while the newest file grows, otherwise not recording at night and daytime outside of it
        if recording['input_readable'] is False:
            recording['state'] = 'unknown'
        elif (minutes_since is not None) and (minutes_since <= NOT_RECORDING_MIN):
            recording['state'] = 'recording'
        elif night_start <= now <= night_end:
            recording['state'] = 'not_recording'
        else:
            recording['state'] = 'daytime'

        return recording


    def _nightFiles(self, night_name, night_start, night_end):
        """ The results of the processed files of the night, read once per file (and again when it is processed
            again).

        Return:
            files: [list] (results_dir, done info, file_result, file length (min)) of every file, sorted by the
                time of the file.
        """

        # The results of the night, by its name (or its date, if the name was made with other coordinates)
        nights = scanNights(self.output_dir, self.config, cache=self.scan_cache)
        results = nights.get(night_name)
        if results is None:
            prefix = '_'.join(night_name.split('_')[:-2])
            results = {}
            for name, night_results in nights.items():
                if name.startswith(prefix):
                    results.update(night_results)

        files = []
        for results_dir in sorted(results):
            results_path = os.path.join(self.output_dir, results_dir)

            # Read the results of the file once, and again when it was processed again (a new done flag)
            try:
                stamp = os.path.getmtime(os.path.join(results_path, 'done.flag'))
            except OSError:
                stamp = None
            cached = self.file_cache.get(results_dir)
            if (cached is None) or (cached[0] != stamp):
                file_result = readFileSkyResults(results_path, self.config)
                if file_result is not None:
                    file_result['ftp_detections'] = fileDetections(results_path)
                cached = (stamp, file_result, {})
                self.file_cache[results_dir] = cached

            file_result = cached[1]
            if file_result is None:
                continue

            # The length of the file from the times of its chunks (min)
            length_min = None
            chunks = file_result['star_chunks']
            if chunks and file_result['chunk_s']:
                length_min = ((chunks[-1][0] - chunks[0][0]).total_seconds() + file_result['chunk_s'])/60.0

            files.append((results_dir, results[results_dir], file_result, length_min))

        # Forget the files of other nights
        night_dirs = set(results)
        self.file_cache = {k: v for k, v in self.file_cache.items() if k in night_dirs}

        files.sort(key=lambda entry: entry[2]['star_chunks'][0][0] if entry[2]['star_chunks']
                   else datetime.datetime.min)

        return files


    def _processing(self, state, now, files):
        """ Whether the processing keeps up: the waiting files, the delay, and the processing times. """

        # The length of the input files (median of the processed files of the night, min)
        lengths = {os.path.basename(results_dir): length for results_dir, _, _, length in files if length}
        for record in self.file_records:
            if record.get('length_min') is None:
                record['length_min'] = _round(lengths.get(record['file']))
        known_lengths = [length for length in lengths.values()]
        file_length_min = float(np.median(known_lengths)) if known_lengths else None

        # The files which are being processed and which wait for it
        in_progress = [{'file': uid, 'started_utc': _timeString(start),
                        'running_min': _round((now - start).total_seconds()/60.0, 1)}
                       for uid, _, start in state.get('in_progress', [])]
        queued = state.get('queued', [])

        # The delay: how long ago the oldest waiting or processed file was recorded (its last write), zero when
        #   nothing waits
        waiting_mtimes = [mtime for _, mtime in queued]
        for _, file_path, _ in state.get('in_progress', []):
            try:
                waiting_mtimes.append(os.path.getmtime(file_path))
            except (OSError, TypeError):
                pass
        delay_min = 0.0
        if waiting_mtimes:
            delay_min = max(0.0, (now - UTCFromTimestamp.utcfromtimestamp(min(waiting_mtimes))).total_seconds()/60.0)

        # The files processed tonight and in the last hour, and their processing times
        night_records = [r for r in self.file_records if (_parseTime(r.get('finished_utc')) is not None)
                         and (_parseTime(r['finished_utc']) >= now - datetime.timedelta(hours=24))]
        ok_records = [r for r in night_records if r.get('outcome') == 'ok']
        hour_records = [r for r in ok_records if _parseTime(r['finished_utc']) >= now - datetime.timedelta(hours=1)]
        ratios = [r['processing_min']/r['length_min'] for r in ok_records
                  if r.get('processing_min') is not None and r.get('length_min')]

        last_done = ok_records[-1] if ok_records else None

        # The files which failed and which are not processed any more
        failed = state.get('failed_files', {})
        given_up = sorted(uid for uid, info in failed.items()
                          if (info.get('count', 0) >= state.get('max_failures', 2))
                          or (info.get('kills', 0) >= state.get('max_kills', 5)))

        processing = {
            'queued': len(queued),
            'settling': len(state.get('settling', [])),
            'in_progress': in_progress,
            'file_length_min': _round(file_length_min),
            'waiting_recording_min': _round((len(queued) + len(in_progress))*file_length_min)
                if file_length_min else None,
            'delay_min': _round(delay_min, 1),
            'not_keeping_up': bool(file_length_min and (delay_min > BEHIND_FILE_LENGTHS*file_length_min)),
            'last_done': last_done,
            'files_done_last_24h': len(ok_records),
            'files_failed_last_24h': sum(1 for r in night_records if r.get('outcome') == 'failed'),
            'failed_files': sorted(failed),
            'given_up': given_up,
            'processing_time_min_last_24h': _percentiles([r.get('processing_min') for r in ok_records]),
            'processing_time_min_last_hour': _percentiles([r.get('processing_min') for r in hour_records]),
            'processing_to_file_length_ratio': _percentiles(ratios),
            'recent_files': self.file_records[-RECENT_FILES:],
        }

        return processing


    def _sky(self, now, night_name, night_start, night_end, files):
        """ The sky now (from the latest processed data) and through the night, from the conditions of the
            chunks measured by the matched filter.

        Return:
            (night, sky): [tuple of dict] The night so far, and the sky in the latest data.
        """

        from RMS.Routines.SkyQuality import CLEAR_FRACTION_CLEAR, nightTimeline

        # The files whose sky was measured, the files which were not searched, and the detections
        file_results = [(info.get('input_file'), file_result) for _, info, file_result, _ in files]
        not_searched, removed, measured, chunk_seconds = nightSkyResults(file_results, self.config, night_start,
                                                                        night_end)
        rows = self._conditionRows(measured, chunk_seconds, night_start, night_end)
        timeline = nightTimeline(rows, not_searched=not_searched, chunk_seconds=chunk_seconds,
                                 bounds=(night_start, night_end))

        # The night so far, the same numbers as on the plot of the sky conditions of the night report
        summary = timeline['summary']
        night = {
            'name': night_name,
            'start_utc': _timeString(night_start),
            'end_utc': _timeString(night_end),
            'files_processed': len(files),
            'files_not_searched': sum(1 for _, _, file_result, _ in files if not file_result['searched']),
            'observed_h': _round(summary['observed_h']),
            'clear_percent': _round(100*summary['clear'], 1),
            'partly_clouded_percent': _round(100*summary['partly'], 1),
            'clouded_percent': _round(100*summary['clouded'], 1),
            'too_few_stars_percent': _round(100*summary['not_searched'], 1),
            'detections': sum(file_result['ftp_detections'] or 0 for _, _, file_result, _ in files),
            'removed_by_clouds': removed,
        }

        # The detections of the matched filter when it runs as a separate pass next to the normal detection
        if self.config.mf_enable and not self.config.mf_replace_detection:
            night['matched_filter_detections'] = sum(file_result['detections'] for _, _, file_result, _ in files)

        sky = {'state': 'no_data', 'observed_utc': None, 'age_min': None, 'observable': None}

        # The stars of the latest chunk, also without the conditions (e.g. with the normal detection)
        latest_chunks = [file_result['star_chunks'][-1] for _, _, file_result, _ in files
                         if file_result['star_chunks']]
        if latest_chunks:
            latest_time, latest_stars = max(latest_chunks)
            sky['stars'] = int(latest_stars)
            sky['stars_utc'] = _timeString(latest_time)

        segments = timeline['segments']
        if not segments:
            return night, sky

        # The state at the end of the latest data, from the time line (the worst of overlapping files)
        latest_state, _, latest_end = segments[-1]
        sky['state'] = {'partly': 'partly_clouded', 'not_searched': 'too_few_stars'}.get(latest_state, latest_state)
        sky['observed_utc'] = _timeString(latest_end)
        sky['age_min'] = _round((now - latest_end).total_seconds()/60.0, 1)

        # The values of the latest chunk with conditions
        if rows:
            row = rows[-1]
            sky.update({'clear_percent': _round(100*row['clear_fraction'], 1),
                        'dimming_mag': _round(row['transparency_mag']),
                        'stars': int(row['stars']), 'matched_stars': int(row['matched']),
                        'reliable_seen_percent': _round(100*row['reliable_seen'], 1)})

        # The last hour of the data
        hour = nightTimeline(rows, not_searched=not_searched, chunk_seconds=chunk_seconds,
                             bounds=(latest_end - datetime.timedelta(hours=1), latest_end))['summary']
        hour_rows = [r for r in rows if r['time'] >= latest_end - datetime.timedelta(hours=1)]
        dimming = [r['transparency_mag'] for r in hour_rows
                   if (r['clear_fraction'] > 0) and np.isfinite(r['transparency_mag'])]
        sky['last_hour'] = {'observed_h': _round(hour['observed_h']),
                            'clear_percent': _round(100*hour['clear'], 1),
                            'partly_clouded_percent': _round(100*hour['partly'], 1),
                            'clouded_percent': _round(100*hour['clouded'], 1),
                            'too_few_stars_percent': _round(100*hour['not_searched'], 1),
                            'median_dimming_mag': _round(np.median(dimming)) if dimming else None}

        # Observable: most of the time in the last minutes of the data had most of the image clear. The time of
        #   every chunk is weighted (a clear chunk counts as good, the files which were not searched as bad)
        window_start = latest_end - datetime.timedelta(minutes=OBSERVABLE_WINDOW_MIN)
        good = total = 0.0
        steps = timeline['steps']
        for r in rows:
            if r['time'] >= window_start:
                length = steps[r.get('input')].total_seconds()
                total += length
                good += length*(r['clear_fraction'] >= min(OBSERVABLE_CLEAR_FRACTION, CLEAR_FRACTION_CLEAR))
        for start, end, _, _ in not_searched:
            total += max(0.0, (min(end, latest_end) - max(start, window_start)).total_seconds())
        if total >= 60*OBSERVABLE_WINDOW_MIN/3:
            sky['observable'] = bool(good/total >= OBSERVABLE_SHARE)

        return night, sky


    def _conditionRows(self, measured, chunk_seconds, night_start, night_end):
        """ The conditions of the chunks of the night, read again only when the log changed. """

        # The log changes when a file was processed, its files are named by the UTC date
        stamps = []
        for name in sorted(os.listdir(self.output_dir)):
            if name.startswith('sky_conditions_') and name.endswith('.csv'):
                try:
                    st = os.stat(os.path.join(self.output_dir, name))
                    stamps.append((name, st.st_mtime, st.st_size))
                except OSError:
                    pass
        key = (tuple(stamps), tuple(sorted(measured)), night_start)

        if self.conditions_cache[0] != key:
            rows = nightConditionRows(self.output_dir, self.config, measured, chunk_seconds, night_start, night_end)
            self.conditions_cache = (key, rows)

        return self.conditions_cache[1]


    def _calibration(self, files):
        """ How healthy the calibration is, from the recalibration of the chunks of the processed files. Only the
            chunks on clear sky are judged, as under clouds there are too few stars for any calibration. """

        calibration = {'dist_check_threshold_px': self.config.dist_check_threshold,
                       'camera_moved_threshold_px': CAMERA_MOVED_PX}

        # The reference platepar which the following data is processed with
        latest_path = latestPlateparPath(self.output_dir, self.config)
        if os.path.isfile(latest_path):
            calibration['latest_platepar'] = latest_path
            calibration['latest_platepar_utc'] = _timeString(
                UTCFromTimestamp.utcfromtimestamp(os.path.getmtime(latest_path)))

        # The calibration of every file (read once per file), with its clear chunks from its conditions
        per_file = []
        for results_dir, info, file_result, _ in files:
            cache = self.file_cache.get(results_dir, (None, None, {}))[2]
            if 'calibration' not in cache:
                clear_times = None
                if file_result['measured']:
                    input_file = info.get('input_file')
                    clear_times = [r['time'] for r in self.conditions_cache[1]
                                   if (r['input'] in (input_file, os.path.basename(input_file or '')))
                                   and (r['clear_fraction'] >= OBSERVABLE_CLEAR_FRACTION)]
                cache['calibration'] = fileCalibration(os.path.join(self.output_dir, results_dir), self.config,
                                                       clear_times=clear_times)
            if cache['calibration'] is not None:
                per_file.append(cache['calibration'])

        if not per_file:
            calibration['status'] = 'unknown'
            calibration['reason'] = 'No recalibrated files tonight'
            return calibration

        def _stats(cals):
            clear_chunks = sum(c['clear_chunks'] for c in cals)
            clear_ok = sum(c['clear_recalibrated'] for c in cals)
            return {'files': len(cals),
                    'chunks': sum(c['chunks'] for c in cals),
                    'recalibrated': sum(c['recalibrated'] for c in cals),
                    'clear_chunks': clear_chunks,
                    'clear_recalibrated_share': _round(clear_ok/clear_chunks, 3) if clear_chunks else None,
                    'median_residual_px': _round(np.median(sum((c['residuals_px'] for c in cals), [])), 3)
                        if any(c['residuals_px'] for c in cals) else None,
                    'median_matched_stars': _round(np.median(sum((c['matched_stars'] for c in cals), [])), 0)
                        if any(c['matched_stars'] for c in cals) else None,
                    'median_zero_point_mag': _round(np.median(sum((c['zero_points'] for c in cals), [])), 3)
                        if any(c['zero_points'] for c in cals) else None,
                    'median_pointing_shift_px': _round(np.median(sum((c['shifts_px'] for c in cals), [])), 2)
                        if any(c['shifts_px'] for c in cals) else None}

        calibration['last_file'] = _stats(per_file[-1:])
        calibration['tonight'] = _stats(per_file)

        # Judge the calibration on the latest files with enough chunks on clear sky
        recent, clear_chunks = [], 0
        for cal in reversed(per_file):
            recent.append(cal)
            clear_chunks += cal['clear_chunks']
            if clear_chunks >= CALIBRATION_MIN_CHUNKS:
                break
        recent_stats = _stats(recent)
        calibration['recent'] = recent_stats

        share = recent_stats['clear_recalibrated_share']
        shift = calibration['last_file']['median_pointing_shift_px']
        if (share is None) or (recent_stats['clear_chunks'] < CALIBRATION_MIN_CHUNKS):
            calibration['status'] = 'unknown'
            calibration['reason'] = 'Too few chunks on clear sky to judge the calibration'
        elif share < CALIBRATION_MIN_SHARE:
            calibration['status'] = 'failing'
            calibration['reason'] = ('Only {:.0f}% of the chunks on clear sky could be recalibrated (the stars match '
                                     'within {:.2f} px)'.format(100*share, self.config.dist_check_threshold))
        elif (shift is not None) and (shift > CAMERA_MOVED_PX):
            calibration['status'] = 'degraded'
            calibration['reason'] = ('The pointing differs from the reference platepar by {:.1f} px, the camera may '
                                     'have moved'.format(shift))
        else:
            calibration['status'] = 'ok'
            calibration['reason'] = None

        return calibration


    def _detector(self, files):
        """ Which detector runs, and how its latest file went. """

        if self.config.mf_enable and self.config.mf_replace_detection:
            name = 'matched_filter'
        elif self.config.mf_enable:
            name = 'normal_and_matched_filter'
        else:
            name = 'normal'

        detector = {'name': name, 'gpu_configured': self.config.mf_gpu if self.config.mf_enable else None,
                    'cloud_filter': bool(self.config.mf_enable and self.config.mf_cloud_filter),
                    'gpu_used_last_file': None, 'candidates_dropped_last_file': None}

        # The latest file which the matched filter searched
        for _, _, file_result, _ in reversed(files):
            summary = file_result['summary']
            if file_result['searched'] and summary:
                detector['gpu_used_last_file'] = summary.get('gpu')
                detector['candidates_dropped_last_file'] = summary.get('tracks_dropped')
                break

        return detector


    def _reports(self, state):
        """ The night reports: the latest one and whether one is running. """

        reporter = state.get('reporter')
        reports = {'mode': getattr(reporter, 'report_mode', None),
                   'report_running': bool(getattr(reporter, 'active', None)),
                   'last_night': None, 'last_report_utc': None, 'last_report_ok': None, 'failed_steps': []}

        try:
            nights = readReportStates(self.output_dir).get('nights', {})
        except Exception:
            nights = {}

        # The latest report
        reported = [(night_state.get('reported_at') or '', night) for night, night_state in nights.items()]
        if reported:
            reported_at, night = max(reported)
            night_state = nights[night]
            reports.update({'last_night': night, 'last_report_utc': reported_at or None,
                            'last_report_ok': not night_state.get('failed_steps'),
                            'failed_steps': list(night_state.get('failed_steps') or []),
                            'recalibration': night_state.get('recalibration')})

        return reports


    def _disk(self, state):
        """ The free space of the output and of the staging directory. """

        disk = {'output_free_gb': _round(availableSpace(self.output_dir)/1024**3, 1),
                'paused_for_space': bool(state.get('paused_for_space'))}

        staging_dir = getattr(self.config, 'monitor_staging_dir', '')
        if staging_dir and os.path.isdir(staging_dir):
            disk['staging_free_gb'] = _round(availableSpace(staging_dir)/1024**3, 1)

        reporter = state.get('reporter')
        last_cleanup = getattr(reporter, 'last_cleanup', None)
        disk['last_cleanup_utc'] = _timeString(last_cleanup) if isinstance(last_cleanup, datetime.datetime) \
            else None

        return disk


    def _faults(self, status, state, now, failed_sections=()):
        """ The faults which are found in the status, with the time they were first seen.

        Arguments:
            status: [dict] The sections of the status.
            state: [dict] State of the monitor, see build.
            now: [datetime] Time of the status (UTC).

        Keyword arguments:
            failed_sections: [list] Sections of the status which could not be made.

        Return:
            faults: [list] {code, severity, since_utc, message} of every fault, from the most severe.
        """

        faults = []

        def _add(code, severity, message):
            faults.append({'code': code, 'severity': severity, 'message': message})

        # The status itself
        if failed_sections:
            _add('STATUS_INCOMPLETE', 'warning', 'Parts of the status could not be made (see the log): {}'.format(
                ', '.join(failed_sections)))

        # The monitor and its input
        if status['monitor']['state'] == 'stopped':
            _add('MONITOR_STOPPED', 'error', 'The monitor has stopped')
        if status['recording']['input_readable'] is False:
            _add('INPUT_UNREADABLE', 'error', 'The input directory {:s} can\'t be read'.format(self.input_dir))
        if status['disk']['paused_for_space']:
            _add('DISK_LOW', 'error', 'The processing is paused as the output disk is full')

        # The processing
        processing = status['processing']
        if processing.get('not_keeping_up'):
            _add('NOT_KEEPING_UP', 'warning', 'The oldest waiting file was recorded {:.0f} min ago, more than {:.0f} '
                 'file lengths ({:.0f} min)'.format(processing['delay_min'], BEHIND_FILE_LENGTHS,
                                                    BEHIND_FILE_LENGTHS*processing['file_length_min']))
        if processing['given_up']:
            _add('FILE_GIVEN_UP', 'error', '{:d} file(s) failed too often and are not processed: {:s}'.format(
                len(processing['given_up']), ', '.join(processing['given_up'][:5])))
        failing = sorted(set(processing['failed_files']) - set(processing['given_up']))
        if failing:
            _add('FILE_FAILED', 'warning', '{:d} file(s) failed and will be processed again: {:s}'.format(
                len(failing), ', '.join(failing[:5])))

        # The calibration
        calibration = status['calibration']
        if calibration.get('status') == 'failing':
            _add('CALIBRATION_FAILING', 'error', calibration['reason'])
        elif calibration.get('status') == 'degraded':
            _add('CAMERA_MOVED', 'warning', calibration['reason'])

        # The night report
        reports = status['reports']
        if reports['failed_steps']:
            severity = 'error' if set(reports['failed_steps']) & ESSENTIAL_STEPS else 'warning'
            _add('REPORT_FAILED', severity, 'The report of night {} failed in: {}'.format(reports['last_night'],
                 ', '.join(reports['failed_steps'])))

        # The detector
        detector = status['detector']
        if (detector['gpu_used_last_file'] is False) and (detector['gpu_configured'] in ('on', 'auto')):
            _add('GPU_NOT_USED', 'warning' if detector['gpu_configured'] == 'on' else 'info',
                 'The matched filter ran on the CPU (mf_gpu: {})'.format(detector['gpu_configured']))
        if detector['candidates_dropped_last_file']:
            _add('CANDIDATES_DROPPED', 'info', '{:d} of the weakest candidates of the last file were not measured '
                 '(mf_max_tracks)'.format(detector['candidates_dropped_last_file']))

        # Keep the time every fault was first seen, and forget the ones which are gone
        codes = set(fault['code'] for fault in faults)
        self.fault_since = {code: since for code, since in self.fault_since.items() if code in codes}
        for fault in faults:
            fault['since_utc'] = _timeString(self.fault_since.setdefault(fault['code'], now))

        return sorted(faults, key=lambda fault: -SEVERITIES.index(fault['severity']))
