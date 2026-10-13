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
from RMS.MonitorNightReport import (ESSENTIAL_STEPS, JSON_TIME_FORMAT, _writeJSON, autoPlateparPath,
                                    latestPlateparPath, nightInfo, nightConditionRows, nightSkyResults, plateparResidual, readFileSkyResults,
                                    readReportStates, scanNights, selectPlatepar)


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

# A chunk is on clear sky for the calibration when at least this share of the image is clear. The fit uses the
#   stars of the whole image, so a little cloud doesn't spoil it, while requiring all of the image to be clear (as
#   for the clear state of the sky, CLEAR_FRACTION_CLEAR) would leave the calibration unjudged on most partly
#   clouded nights
CALIBRATION_CLEAR_FRACTION = 0.8

# The camera has moved when its pointing differs from the reference platepar by more than this (px). The pointing
#   normally changes by a fraction of a pixel through the night (refraction, temperature)
CAMERA_MOVED_PX = 2.0

# Number of the latest processed files listed in the status
RECENT_FILES = 50

# History of the processed files of the last day, for the statistics (also after a restart), one line per file
HISTORY_FILE_NAME = '.monitor_status_files.jsonl'
HISTORY_HOURS = 24.0

# Smallest number of processed files of the last hour for its processing times to predict the queue (otherwise
#   the files of the last day are used)
PREDICTION_MIN_FILES = 3

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


def _cleanRecord(record):
    """ A processed file of the history as it is used: its numbers as numbers (None if they aren't), None if
        the record can't be used (e.g. a damaged line). """

    if (not isinstance(record, dict)) or (_parseTime(record.get('finished_utc')) is None):
        return None

    record = dict(record)
    record['file'] = str(record.get('file'))
    record['outcome'] = str(record.get('outcome'))

    # The numbers, None where they are not numbers (a bool is not taken as a number)
    for name in ('processing_min', 'size_gb', 'length_min'):
        value = record.get(name)
        try:
            value = None if isinstance(value, bool) else float(value) if isinstance(value, (int, float)) else None
        except OverflowError:
            value = None
        record[name] = value if (value is not None) and np.isfinite(value) else None

    return record


def _codeVersion():
    """ Commit of the code which runs, None if it is not a git repository. """

    try:
        import git
        repo = git.Repo(os.path.dirname(os.path.abspath(__file__)), search_parent_directories=True)
        return repo.head.object.hexsha

    except Exception:
        return None



def _tooFewStars(file_result, config):
    """ Check if a processed file was not searched as it had too few stars in all its chunks (as for the time
        which is shaded as not searched, see MonitorNightReport.nightSkyResults). A file of the normal detection
        has no summary of the matched filter, but it was searched if it had enough stars.
    """

    chunks = file_result['star_chunks']

    return (not file_result['searched']) and bool(chunks) and (max(n for _, n in chunks) < config.ff_min_stars)


def simulateQueues(nproc, cameras, last_idx=None):
    """ Predict when the waiting files of every camera will be done, by running the scheduling of the monitor
        forward in time: every running file is done after its remaining time, and a free worker takes the next
        waiting file of the camera with the fewest running files (the next camera in turn when they have as many),
        as the monitor assigns them.

    Arguments:
        nproc: [int] Number of the worker processes of the monitor.
        cameras: [list] (camera_id, remaining times of its running files, expected times of its waiting files in
            their order) of every camera in the order of the monitor, all in minutes. A time which is not known
            is None.

    Keyword arguments:
        last_idx: [int] Index of the camera which the monitor gave a worker last, where the turn continues. The
            last camera if None (as at the start of the monitor). None by default.

    Return:
        [dict] Minutes from now until every camera has no waiting or running files, by camera. None for a camera
            whose times are not all known.
    """

    # Times which are not known (e.g. a camera without processed files yet) are taken as the median of the known
    #   ones, so they still take a worker for the other cameras, but their own camera is not predicted
    known = [t for _, running, waiting in cameras for t in list(running) + list(waiting) if t is not None]
    fill = float(np.median(known)) if known else 0.0
    unknown = set(cam_id for cam_id, running, waiting in cameras
                  if any(t is None for t in list(running) + list(waiting)))

    order = [cam_id for cam_id, _, _ in cameras]
    pending = {cam_id: [fill if t is None else t for t in waiting] for cam_id, _, waiting in cameras}

    # The running files, as (time when they are done, camera)
    active = [(fill if t is None else t, cam_id) for cam_id, running, _ in cameras for t in running]
    done = {cam_id: max([end for end, c in active if c == cam_id] + [0.0]) for cam_id in order}

    t = 0.0
    if last_idx is None:
        last_idx = len(order) - 1
    while any(pending.values()):

        # The next free worker: now if one is free, otherwise when the first running file is done
        if len(active) >= nproc:
            t = min(end for end, _ in active)
        active = [(end, c) for end, c in active if end > t]

        # The camera which gets it, as in the monitor: the fewest running files, then the next one in turn
        candidates = [i for i, cam_id in enumerate(order) if pending[cam_id]]
        candidates.sort(key=lambda i: (sum(1 for _, c in active if c == order[i]), (i - last_idx - 1)%len(order)))
        last_idx = candidates[0]
        cam_id = order[last_idx]

        # Its next waiting file runs from now
        end = t + pending[cam_id].pop(0)
        active.append((end, cam_id))
        done[cam_id] = max(done[cam_id], end)

    return {cam_id: (None if cam_id in unknown else done[cam_id]) for cam_id in order}


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

        # Only the main FTPdetectinfo counts (not its backups and other versions)
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
            reference platepar of the file, of the chunks on clear sky). None if there are no recalibrated
            platepars.
    """

    # Read the platepars which were recalibrated on every chunk, there is nothing to judge without them
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

        # Count the chunk, and whether it was recalibrated, also on clear sky
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

            # The photometric zero point of the chunk
            calibration['zero_points'].append(float(platepar.mag_lev))

            # The difference of the pointing from the reference platepar, in pixels (the plate scale is in px/deg),
            #   on clear sky only, as the rest of the judgement of the calibration
            if (reference is not None) and clear:
                d_az = (platepar.az_centre - reference.az_centre + 180)%360 - 180
                d_alt = platepar.alt_centre - reference.alt_centre
                shift_deg = np.hypot(d_az*np.cos(np.radians(reference.alt_centre)), d_alt)
                calibration['shifts_px'].append(float(shift_deg*reference.F_scale))

        except Exception as e:
            log.debug("The calibration of {:s} could not be read: {:s}".format(ff_name, repr(e)))

    return calibration



class MonitorStatus(object):
    def __init__(self, camera_id, output_dir, input_dir, config, input_scanner, nproc, input_sizer=None,
                 platepar_path=None, enabled=True):
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

        Keyword arguments:
            input_sizer: [callable] Function which returns the size of an input (bytes), None if it can't be read
                (see MonitorProcessFrameInterface.inputState). The size of the file by default.
            platepar_path: [str] Path of the platepar given to the monitor, None if not known. None by default.
            enabled: [bool] Write the status and the history of the camera. A camera without them is still
                followed in memory, as its files share the workers with the other cameras (see simulateQueues).
                True by default.
        """

        # The camera, its directories, how to find its inputs, and its platepar
        self.camera_id = camera_id
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.config = config
        self.input_scanner = input_scanner
        self.nproc = nproc
        self.input_sizer = input_sizer if input_sizer is not None else os.path.getsize
        self.platepar_path = platepar_path
        self.enabled = enabled

        # The status file, the history of the processed files, and how often the status is written
        self.path = os.path.join(output_dir, STATUS_FILE_NAME)
        self.history_path = os.path.join(output_dir, HISTORY_FILE_NAME)
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
        self.conditions_fresh = False
        self.size_cache = {}
        self.night_cache = {}

        # The status is written right away the first time (the time of the last write is of the monotonic clock)
        self.last_write = -float('inf')
        self.dirty = True

        # Continue the statistics and the faults of a previous run of the monitor (a damaged file of the previous
        #   run must not stop the monitor from starting)
        try:
            self._restore()
        except Exception as e:
            log.warning("[{}] The status of the previous run could not be read: {}".format(camera_id, repr(e)))


    def _restore(self):
        """ Take the processed files of the last day from the history, and the times of the faults from the status
            of a previous run. """

        # The processed files of the last day (lines which can't be read are skipped)
        if os.path.isfile(self.history_path):
            try:
                with open(self.history_path, errors='replace') as f:
                    for line in f:
                        try:
                            record = _cleanRecord(json.loads(line))
                        except Exception:
                            continue
                        if record is not None:
                            self.file_records.append(record)
            except OSError as e:
                log.debug("The history {:s} could not be read: {:s}".format(self.history_path, repr(e)))

            # Keep only the last day in the history file
            self._pruneRecords(RmsDateTime.utcnow())
            self._writeHistory()

        # The times when the faults of the previous run were first seen, so a fault which continues keeps its time
        if not os.path.isfile(self.path):
            return

        try:
            with open(self.path) as f:
                previous = json.load(f)

            for fault in previous.get('faults', []):
                since = _parseTime(fault.get('since_utc'))
                if since is not None:
                    self.fault_since[fault.get('code')] = since

        except Exception as e:
            log.debug("The previous status {:s} could not be read: {:s}".format(self.path, repr(e)))


    def fileFinished(self, unique_id, outcome, processing_s, input_name=None, size_bytes=None):
        """ Record a processed file in the history of the last day. The status is written at the next update.

        Arguments:
            unique_id: [str] Unique ID of the file (the name of its results directory).
            outcome: [str] 'ok', 'failed', 'skipped' or 'changed' (still being written, processed again).
            processing_s: [float] Processing time (s).

        Keyword arguments:
            input_name: [str] Name of the input (file or FITS directory). None by default.
            size_bytes: [int] Size of the input (bytes), None if not known. None by default.
        """

        # Record the file with its input and size, and keep only the files of the last day
        now = RmsDateTime.utcnow()
        record = {'file': unique_id, 'input': input_name, 'outcome': outcome, 'finished_utc': _timeString(now),
                  'processing_min': _round(processing_s/60.0),
                  'size_gb': _round(size_bytes/1024**3, 4) if size_bytes else None, 'length_min': None}
        self.file_records.append(record)
        self._pruneRecords(now)

        # Append it to the history file, so the statistics continue after a restart
        if not self.enabled:
            self.dirty = True
            return
        try:
            with open(self.history_path, 'a') as f:
                f.write(json.dumps(record) + '\n')
        except OSError as e:
            log.debug("The history {:s} could not be written: {:s}".format(self.history_path, repr(e)))

        self.dirty = True


    def _writeHistory(self):
        """ Write the history of the processed files of the last day again (e.g. to keep only the last day, or
            with the lengths of the files). It replaces the old file, so it is never partly written. """

        if not self.enabled:
            return

        try:
            with open(self.history_path + '.tmp', 'w') as f:
                for record in self.file_records:
                    f.write(json.dumps(record) + '\n')
            os.replace(self.history_path + '.tmp', self.history_path)
        except OSError as e:
            log.debug("The history {:s} could not be written: {:s}".format(self.history_path, repr(e)))


    def _pruneRecords(self, now):
        """ Keep only the processed files of the last day. """

        oldest = now - datetime.timedelta(hours=HISTORY_HOURS)
        self.file_records = [r for r in self.file_records if _parseTime(r.get('finished_utc')) is not None
                             and _parseTime(r['finished_utc']) >= oldest]


    def due(self):
        """ Check if the status is due to be written (every interval, or right after a file was processed). """

        # The interval is measured on the monotonic clock, so the writes go on when the clock of the system is set
        #   (only the times written in the status are of the clock of the system)
        return self.dirty or (time.monotonic() - self.last_write >= self.interval)


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

        # Write only when it is due (and never for a camera without the status)
        if not (self.enabled and (force or self.due())):
            return False

        try:
            _writeJSON(self.path, self.build(state))
            self.last_write = time.monotonic()
            self.dirty = False
            return True

        except Exception as e:
            log.warning("[{}] The status could not be written: {}".format(self.camera_id, repr(e)))

            # Try again after the interval, not at every poll
            self.last_write = time.monotonic()
            self.dirty = False
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
                - settling: [list] Files which are still being written, as (path, time of the last change).
                - failed_files: [dict] The failed files of the camera (see loadFailedFiles).
                - max_failures, max_kills: [int] Failures and kills after which a file is given up.
                - reporter: [NightReporter] Night reporter of the camera, None if there is none.
                - queue_empty_min: [float] Minutes until the files of the camera are done, predicted by the monitor
                    for all its cameras together (see simulateQueues). Predicted for this camera alone if not
                    given.

        Return:
            status: [dict] The status.
        """

        now = RmsDateTime.utcnow()

        # The night which is current or comes next
        night_name, night_start, night_end = nightInfo(self.config, now)

        # The format, the camera and the time of the status
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
            """ Make a section, or return its default if it fails. """

            try:
                return func()
            except Exception as e:
                log.warning("[{}] The {} of the status could not be made: {}".format(self.camera_id, name,
                                                                                    repr(e)))
                failed_sections.append(name)
                return default

        # The results of the files of the night, which the following sections use
        files = _section('results', lambda: self._nightFiles(night_name, night_start, night_end), [])

        status['monitor'] = self._monitor(state)
        status['recording'] = _section('recording', lambda: self._recording(state, now, night_start, night_end),
                                       {'state': 'unknown', 'input_readable': state.get('input_readable')})
        status['processing'] = _section('processing', lambda: self._processing(state, now, files),
                                        {'not_keeping_up': False, 'failed_files': [], 'given_up': []})

        # The sky reads the conditions of the night (see _conditionRows)
        self.conditions_fresh = False
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

        # The faults, from the most severe, and the overall state. If they can't be judged, the status says so
        try:
            status['faults'] = self._faults(status, state, now, failed_sections)
        except Exception as e:
            log.warning("[{}] The faults of the status could not be judged: {}".format(self.camera_id, repr(e)))
            status['faults'] = [{'code': 'STATUS_INCOMPLETE', 'severity': 'warning',
                                 'message': 'The faults could not be judged (see the log)',
                                 'since_utc': _timeString(now)}]
        severities = [SEVERITIES.index(fault['severity']) for fault in status['faults']]
        worst = max(severities) if severities else 0
        status['overall'] = {2: 'error', 1: 'warning'}.get(worst, 'ok')

        return status


    def _monitor(self, state):
        """ The monitor itself: running or stopped, and its workers. """

        # Stopped, paused while the output disk is full, or running
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
                if night_start <= mtime <= min(now, night_end):
                    files_tonight += 1
                if (newest_mtime is None) or (mtime > newest_mtime):
                    newest, newest_mtime, newest_size = os.path.basename(path), mtime, size

        except OSError:
            recording['input_readable'] = False

        # The newest input, and how long ago it was written
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

                # Results which can't be read are read again next time (e.g. they were being written), and the
                #   file still counts as processed, without its sky
                if file_result is None:
                    file_result = {'star_chunks': [], 'chunk_s': None, 'summary': None, 'searched': False,
                                   'measured': False, 'removed_times': None, 'removed_count': 0, 'detections': 0,
                                   'unreadable': True}
                    cached = (None, file_result, {})
                else:
                    cached = (stamp, file_result, {})
                    self.file_cache[results_dir] = cached

                file_result['ftp_detections'] = fileDetections(results_path)

            file_result = cached[1]

            # The length of the file from the times of its chunks (min)
            length_min = None
            chunks = file_result['star_chunks']
            if chunks and file_result['chunk_s']:
                length_min = ((chunks[-1][0] - chunks[0][0]).total_seconds() + file_result['chunk_s'])/60.0

            files.append((results_dir, results[results_dir], file_result, length_min))

        # Forget the files of other nights
        night_dirs = set(results)
        self.file_cache = {k: v for k, v in self.file_cache.items() if k in night_dirs}

        # Sort the files by the time of their first chunk
        files.sort(key=lambda entry: entry[2]['star_chunks'][0][0] if entry[2]['star_chunks']
                   else datetime.datetime.min)

        return files


    def _inputSize(self, file_path):
        """ The size of an input (GB), read once per version of the input. None if it can't be read. """

        # An input which changed is read again
        try:
            key = (file_path, os.path.getmtime(file_path))
        except (OSError, TypeError):
            return None

        if key not in self.size_cache:
            try:
                size = self.input_sizer(file_path)
            except Exception:
                size = None
            self.size_cache[key] = size/1024**3 if size else None

        return self.size_cache[key]


    def _expectedMinutes(self, now):
        """ How to predict the processing time of a file: from the processing time per GB of the input of the
            recent files (so short and long inputs are both predicted), or per file if the sizes are not known.
            The files of the last hour are used if there are enough of them, otherwise the files of the last day.

        Return:
            (predict, basis): [tuple] Function which returns the expected processing time (min) of an input of a
                given size (GB, or None), and the basis of the prediction ('size', 'file' or None if it can't be
                predicted).
        """

        # Learn from the processed files of the last hour if there are enough of them, otherwise of the last day.
        #   Nothing can be predicted before any file was processed
        ok_records = [r for r in self.file_records if (r.get('outcome') == 'ok')
                      and (r.get('processing_min') is not None)]
        hour = [r for r in ok_records if _parseTime(r['finished_utc']) >= now - datetime.timedelta(hours=1)]
        records = hour if len(hour) >= PREDICTION_MIN_FILES else ok_records
        if not records:
            return (lambda size_gb: None), None

        # The processing time per GB, and per file for the inputs whose size is not known
        per_file = float(np.median([r['processing_min'] for r in records]))
        rates = [r['processing_min']/r['size_gb'] for r in records if r.get('size_gb')]
        if not rates:
            return (lambda size_gb: per_file), 'file'

        rate = float(np.median(rates))
        return (lambda size_gb: rate*size_gb if size_gb else per_file), 'size'


    def _pruneCaches(self, state):
        """ Keep the sizes and nights of the inputs only while they are open (waiting, processed or written). """

        open_paths = {file_path for _, file_path, _ in state.get('in_progress', [])}
        open_paths |= {path for path, _ in list(state.get('queued', [])) + list(state.get('settling', []))}
        self.size_cache = {key: value for key, value in self.size_cache.items() if key[0] in open_paths}
        self.night_cache = {key: value for key, value in self.night_cache.items() if key[0] in open_paths}


    def openWork(self, state):
        """ The files of the camera which are being processed and which wait, as times for the prediction of the
            queue (see simulateQueues).

        Arguments:
            state: [dict] State of the monitor for the camera, see build.

        Return:
            (running, waiting): [tuple] Remaining times of the running files and expected times of the waiting
                files (min, None if not known). The waiting files of a camera which is paused for disk space can't
                get a worker, so they are left out.
        """

        now = RmsDateTime.utcnow()
        predict, _ = self._expectedMinutes(now)

        # Also a camera without the status keeps only the sizes of its open files
        self._pruneCaches(state)

        # The remaining time of every running file: its expected time minus how long it has been running
        running = []
        for _, file_path, start in state.get('in_progress', []):
            expected_min = predict(self._inputSize(file_path))
            running.append(None if expected_min is None
                           else max(expected_min - (now - start).total_seconds()/60.0, 0.0))

        # The expected time of every waiting file
        waiting = [] if state.get('paused_for_space') \
            else [predict(self._inputSize(path)) for path, _ in state.get('queued', [])]

        return running, waiting


    def _nightOf(self, file_path, mtime):
        """ The night of an input, from the time it was last written (read once per version of the input). """

        key = (file_path, mtime)
        if key not in self.night_cache:
            self.night_cache[key] = nightInfo(self.config, UTCFromTimestamp.utcfromtimestamp(mtime))[0]

        return self.night_cache[key]


    def _processing(self, state, now, files):
        """ Whether the processing keeps up: the files being processed and waiting, the progress of the night, when
            the waiting files will be done, the delay, and the processing times. """

        # Only the processed files of the last day count, also if no file was processed for a while
        self._pruneRecords(now)

        # The sizes and nights of the inputs are only kept while they are open
        self._pruneCaches(state)

        # The length of the input files (median of the processed files of the night, min). The lengths of the
        #   processed files are kept in the history, so they are known after a restart too
        lengths = {os.path.basename(results_dir): length for results_dir, _, _, length in files if length}
        new_lengths = False
        for record in self.file_records:
            if (record.get('length_min') is None) and lengths.get(record['file']):
                record['length_min'] = _round(lengths[record['file']])
                new_lengths = True
        if new_lengths:
            self._writeHistory()
        # Before a file of the night was processed (e.g. at its beginning), the lengths of the files of the last day
        #   are taken
        known_lengths = [length for length in lengths.values()]
        if not known_lengths:
            known_lengths = [r['length_min'] for r in self.file_records if r.get('length_min')]
        file_length_min = float(np.median(known_lengths)) if known_lengths else None

        # The expected processing time of a file, from the recent files
        predict, basis = self._expectedMinutes(now)

        # The files being processed: the input (file or FITS directory), when it started, and when it is expected
        #   to be done. The start time comes from the monotonic clock of the monitor (see statusState), so it
        #   doesn't jump with the clock of the system
        in_progress = []
        for uid, file_path, start in state.get('in_progress', []):

            # How long it has been running, and how long its input is expected to take
            running_min = (now - start).total_seconds()/60.0
            expected_min = predict(self._inputSize(file_path))
            in_progress.append({
                'file': uid,
                'input': os.path.basename(file_path) if file_path else None,
                'started_utc': _timeString(start),
                'running_min': _round(running_min, 1),
                'expected_min': _round(expected_min, 1),
                'expected_done_utc': _timeString(start + datetime.timedelta(minutes=expected_min))
                    if expected_min is not None else None,
                'remaining_min': _round(max(expected_min - running_min, 0.0), 1) if expected_min is not None
                    else None})

        # The files waiting to be processed, in the order they will be processed
        queued = state.get('queued', [])

        # Predict when the waiting files will be done. In multi-camera mode the monitor predicts it for all its
        #   cameras together, as they share the workers (see simulateQueues), otherwise it is predicted here
        if 'queue_empty_min' in state:
            queue_empty_min = state['queue_empty_min']
        elif basis is None:
            queue_empty_min = None if (in_progress or queued) else 0.0
        else:
            running, waiting = self.openWork(state)
            queue_empty_min = simulateQueues(self.nproc, [(self.camera_id, running, waiting)])[self.camera_id]

        # While the camera is paused for disk space its waiting files can't start, so it is not known when they
        #   will be done, and after the monitor stopped they are not processed
        if (state.get('paused_for_space') and queued) or state.get('stopped'):
            queue_empty_min = None

        # The progress of the night: the files of the night which are done, being processed, waiting, and still
        #   being written. The files of other nights (e.g. a backlog) are counted apart
        night_name = nightInfo(self.config, now)[0]
        open_nights = []
        for _, file_path, _ in state.get('in_progress', []):
            try:
                open_nights.append(('in_progress', self._nightOf(file_path, os.path.getmtime(file_path))))
            except (OSError, TypeError):
                open_nights.append(('in_progress', None))
        open_nights += [('queued', self._nightOf(path, mtime)) for path, mtime in queued]
        open_nights += [('settling', self._nightOf(path, mtime)) for path, mtime in state.get('settling', [])]

        # A done file which is processed again (e.g. with --force) counts as open only
        open_inputs = {os.path.abspath(file_path) for _, file_path, _ in state.get('in_progress', [])}
        open_inputs |= {os.path.abspath(path) for path, _ in queued}
        n_done = sum(1 for _, info, _, _ in files if info.get('input_file') not in open_inputs)
        n_running, n_waiting, n_writing = [sum(1 for kind, night in open_nights
                                               if (kind == wanted) and (night == night_name))
                                           for wanted in ('in_progress', 'queued', 'settling')]
        n_open = n_running + n_waiting
        progress = {'done_tonight': n_done, 'in_progress': n_running, 'queued': n_waiting,
                    'being_written': n_writing,
                    'percent_done': _round(100.0*n_done/(n_done + n_open), 1) if (n_done + n_open) else None,
                    'other_nights_open': len(open_nights) - n_open - n_writing}

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
            oldest = UTCFromTimestamp.utcfromtimestamp(min(waiting_mtimes))
            delay_min = max(0.0, (now - oldest).total_seconds()/60.0)

        # The files processed in the last day and in the last hour, and their processing times
        day_records = list(self.file_records)
        ok_records = [r for r in day_records if r.get('outcome') == 'ok']
        hour_records = [r for r in ok_records if _parseTime(r['finished_utc']) >= now - datetime.timedelta(hours=1)]
        ratios = [r['processing_min']/r['length_min'] for r in ok_records
                  if r.get('processing_min') is not None and r.get('length_min')]

        # The last processed file
        last_done = ok_records[-1] if ok_records else None

        # The files which failed and which are not processed any more
        failed = state.get('failed_files', {})
        given_up = sorted(uid for uid, info in failed.items()
                          if (info.get('count', 0) >= state.get('max_failures', 2))
                          or (info.get('kills', 0) >= state.get('max_kills', 5)))

        # Put the section together
        processing = {
            'progress': progress,
            'in_progress': in_progress,
            'queued': len(queued),
            'queued_inputs': [os.path.basename(path) for path, _ in queued[:RECENT_FILES]],
            'settling': len(state.get('settling', [])),
            'queue_empty_in_min': _round(queue_empty_min, 1),
            'queue_empty_utc': _timeString(now + datetime.timedelta(minutes=queue_empty_min))
                if queue_empty_min is not None else None,
            'prediction_basis': basis,
            'file_length_min': _round(file_length_min),
            'waiting_recording_min': _round((len(queued) + len(in_progress))*file_length_min)
                if file_length_min else None,
            'delay_min': _round(delay_min, 1),
            'not_keeping_up': bool(file_length_min and (delay_min > BEHIND_FILE_LENGTHS*file_length_min)),
            'last_done': last_done,
            'files_done_last_24h': len(ok_records),
            'files_failed_last_24h': sum(1 for r in day_records if r.get('outcome') == 'failed'),
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

        from RMS.Routines.SkyQuality import nightTimeline

        # The files whose sky was measured, the files which were not searched, and the detections
        file_results = [(info.get('input_file'), file_result) for _, info, file_result, _ in files]
        not_searched, removed, measured, chunk_seconds, input_ends = nightSkyResults(file_results, self.config,
                                                                                    night_start, night_end)
        # The conditions of the chunks of the measured files, and the time line of the states of the sky
        rows = self._conditionRows(measured, chunk_seconds, night_start, night_end, input_ends)
        timeline = nightTimeline(rows, not_searched=not_searched, chunk_seconds=chunk_seconds, input_ends=input_ends,
                                 bounds=(night_start, night_end))

        # The night so far, the same numbers as on the plot of the sky conditions of the night report
        summary = timeline['summary']
        night = {
            'name': night_name,
            'start_utc': _timeString(night_start),
            'end_utc': _timeString(night_end),
            'files_processed': len(files),
            'files_not_searched': sum(1 for _, _, file_result, _ in files if _tooFewStars(file_result, self.config)),
            'files_unreadable': sum(1 for _, _, file_result, _ in files if file_result.get('unreadable')),
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

        # The sky when nothing is known yet
        sky = {'state': 'no_data', 'observed_utc': None, 'age_min': None, 'observable': None}

        # The stars of the latest chunk, also without the conditions (e.g. with the normal detection)
        latest_chunks = [file_result['star_chunks'][-1] for _, _, file_result, _ in files
                         if file_result['star_chunks']]
        if latest_chunks:
            latest_time, latest_stars = max(latest_chunks)
            sky['stars'] = int(latest_stars)
            sky['stars_utc'] = _timeString(latest_time)

        # Without data in the night there is nothing more to say about the sky
        segments = timeline['segments']
        if not segments:
            return night, sky

        # The state at the end of the latest data, from the time line (the worst of overlapping files)
        latest_state, _, latest_end = segments[-1]

        # The sky now is not known if data which was not measured (e.g. with the normal detection) ends after the
        #   time line (the measured files and the ones with too few stars are on it), at the end of the input where
        #   it is known
        unmeasured_ends = [file_result.get('end_time') or (file_result['star_chunks'][-1][0]
                           + datetime.timedelta(seconds=file_result['chunk_s'] or 0))
                           for _, _, file_result, _ in files
                           if file_result['star_chunks'] and not file_result['measured']
                           and not _tooFewStars(file_result, self.config)]
        if unmeasured_ends and (max(unmeasured_ends) > latest_end):
            return night, sky

        sky['state'] = {'partly': 'partly_clouded', 'not_searched': 'too_few_stars'}.get(latest_state, latest_state)
        sky['observed_utc'] = _timeString(latest_end)
        sky['age_min'] = _round((now - latest_end).total_seconds()/60.0, 1)

        # The values of the latest chunk which gave the latest state (with overlapping files, the worse state is
        #   taken, so the values come from the chunk with that state). Without one, e.g. when the latest file was not
        #   searched, only the stars are known
        latest_start = segments[-1][1]
        steps, labels = timeline['steps'], timeline['labels']

        # A chunk lasts for its time, the last chunk of an input until the end of the input
        last_rows = {idx[-1]: name for name, idx in timeline['input_rows'].items()}
        row_ends = [timeline['ends'][last_rows[i]] if i in last_rows else r['time'] + steps[r.get('input')]
                    for i, r in enumerate(rows)]
        candidates = [i for i, r in enumerate(rows)
                      if (r['time'] < latest_end) and (row_ends[i] > latest_start) and (labels[i] == latest_state)]
        if candidates:
            row = rows[max(candidates, key=lambda i: (rows[i]['time'], -rows[i]['clear_fraction']))]
            sky.update({'clear_percent': _round(100*row['clear_fraction'], 1),
                        'dimming_mag': _round(row['transparency_mag']),
                        'stars': int(row['stars']), 'stars_utc': _timeString(row['time']),
                        'matched_stars': int(row['matched']),
                        'reliable_seen_percent': _round(100*row['reliable_seen'], 1)})

        # The last hour of the data
        hour = nightTimeline(rows, not_searched=not_searched, chunk_seconds=chunk_seconds, input_ends=input_ends,
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

        # Observable: most of the time in the last minutes of the data had most of the image clear. The time is
        #   taken from the time line, so overlapping files count once, with the worse state (the files which were
        #   not searched count as not observable)
        window = nightTimeline(rows, not_searched=not_searched, chunk_seconds=chunk_seconds, input_ends=input_ends,
                               bounds=(latest_end - datetime.timedelta(minutes=OBSERVABLE_WINDOW_MIN), latest_end),
                               clear_level=OBSERVABLE_CLEAR_FRACTION)['summary']
        if window['observed_h']*60 >= OBSERVABLE_WINDOW_MIN/3:
            sky['observable'] = bool(window['clear'] >= OBSERVABLE_SHARE)

        return night, sky


    def _conditionRows(self, measured, chunk_seconds, night_start, night_end, input_ends=None):
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
        key = (tuple(stamps), tuple(sorted(measured)), night_start, tuple(sorted((input_ends or {}).items())))

        # Read the conditions again only when the log, the measured files or their ends changed
        if self.conditions_cache[0] != key:
            rows = nightConditionRows(self.output_dir, self.config, measured, chunk_seconds, night_start, night_end,
                                      input_ends=input_ends)
            self.conditions_cache = (key, rows)

        # The conditions are up to date for this status
        self.conditions_fresh = True

        return self.conditions_cache[1]


    def _autoCalibration(self, files):
        """ The state of the automatic calibration of the camera, see RMS.Astrometry.AutoCalibration.

        Arguments:
            files: [list] Processed files of the night as (results_dir, info, file_result, _), results_dir
                relative to the output directory.

        Return:
            [dict] The number of installed platepars, the last one, the number of candidates of a dry run, whether
                a candidate waits for the confirmation or a platepar is on trial, and the judgement of the latest
                judged file.
        """

        from RMS.Astrometry.AutoCalibration import FILE_RESULT_NAME, readState

        # The installed platepars (apart from the candidates of a dry run, which are not used), a pending
        #   candidate and a platepar on trial, from the state of the camera
        state = readState(self.output_dir)
        installs = [entry for entry in state['installs'] if not entry.get('dry_run')]
        summary = {'installs': len(installs), 'dry_run_candidates': len(state['installs']) - len(installs),
                   'pending': state.get('pending') is not None, 'trial': state.get('trial') is not None}
        if installs:
            summary['last_install'] = {key: installs[-1].get(key) for key in ['path', 'reason']}

        # The judgement of the latest file which has one
        for results_dir, _, _, _ in sorted(files, key=lambda entry: entry[0], reverse=True):
            result_path = os.path.join(self.output_dir, results_dir, FILE_RESULT_NAME)
            if not os.path.isfile(result_path):
                continue

            # A damaged judgement is skipped
            try:
                with open(result_path) as f:
                    result = json.load(f)

            except ValueError:
                continue

            # What was done, and whether the platepar passed
            summary['latest'] = {'results_dir': results_dir, 'action': result.get('action'),
                                 'platepar_ok': result.get('current', {}).get('ok'),
                                 'reasons': result.get('current', {}).get('reasons', [])}
            break

        return summary


    def _calibration(self, files):
        """ How healthy the calibration is, from the recalibration of the chunks of the processed files. Only the
            chunks on clear sky are judged, as under clouds there are too few stars for any calibration. """

        # The limits of the judgement
        calibration = {'dist_check_threshold_px': self.config.dist_check_threshold,
                       'camera_moved_threshold_px': CAMERA_MOVED_PX}

        # The platepar which the following files are processed with, selected as by the workers: the newest of the
        #   given platepar, the best platepar of the latest reported night and the automatically fitted one. Without
        #   any, the files are processed without the calibration
        platepar_path = selectPlatepar(self.output_dir, self.config, self.platepar_path)
        calibration['calibrated'] = platepar_path is not None
        if platepar_path is not None:
            calibration['platepar'] = platepar_path
            calibration['platepar_utc'] = _timeString(
                UTCFromTimestamp.utcfromtimestamp(os.path.getmtime(platepar_path)))
            calibration['platepar_from_previous_night'] = (platepar_path == latestPlateparPath(self.output_dir,
                                                                                               self.config))
            calibration['platepar_auto'] = (platepar_path == autoPlateparPath(self.output_dir, self.config))

        # The automatic calibration: its judgement of the latest file judged, the last platepar it installed and a
        #   candidate which waits for the confirmation
        if self.config.monitor_auto_calibration:
            calibration['auto_calibration'] = self._autoCalibration(files)

        # The calibration of every file (read once per file), with its clear chunks from its conditions
        per_file = []
        for results_dir, info, file_result, _ in files:
            cache = self.file_cache.get(results_dir, (None, None, {}))[2]
            file_calibration = cache.get('calibration')
            if 'calibration' not in cache:

                # The times of the chunks of the file on clear sky. With the cloud filter, a file whose sky was
                #   not measured (e.g. too few stars) has no chunk known to be clear. Without the cloud filter the
                #   sky is never known, and all chunks count
                cloud_filter = self.config.mf_enable and self.config.mf_cloud_filter
                clear_times = [] if cloud_filter else None
                conditions_known = True
                if file_result['measured']:
                    input_file = info.get('input_file')
                    file_rows = [r for r in self.conditions_cache[1]
                                 if r['input'] in (input_file, os.path.basename(input_file or ''))]
                    clear_times = [r['time'] for r in file_rows if r['clear_fraction'] >= CALIBRATION_CLEAR_FRACTION]
                    conditions_known = self.conditions_fresh and bool(file_rows)
                file_calibration = fileCalibration(os.path.join(self.output_dir, results_dir), self.config,
                                                   clear_times=clear_times)

                # Kept only when the conditions of a measured file were read for this status (they may not be,
                #   e.g. when the log could not be read), otherwise it is judged again with them
                if conditions_known:
                    cache['calibration'] = file_calibration

            if file_calibration is not None:
                per_file.append(file_calibration)

        # Nothing to judge without recalibrated files
        if not per_file:
            calibration['status'] = 'unknown'
            calibration['reason'] = 'No recalibrated files tonight'
            return calibration

        def _stats(cals):
            """ The calibration of the given files together. """

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

        # The calibration of the last file and of the night
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

        # Judge it: too few chunks on clear sky to say, most of them could not be recalibrated (the platepar doesn't
        #   fit), the pointing moved (the camera may have moved), or fine
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

        # The detector from the config
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
                dropped = summary.get('tracks_dropped')
                if isinstance(dropped, int) and not isinstance(dropped, bool):
                    detector['candidates_dropped_last_file'] = dropped
                break

        return detector


    def _reports(self, state):
        """ The night reports: the latest one and whether one is running. """

        # The mode of the reports and whether one is running
        reporter = state.get('reporter')
        reports = {'mode': getattr(reporter, 'report_mode', None),
                   'report_running': bool(getattr(reporter, 'active', None)),
                   'last_night': None, 'last_report_utc': None, 'last_report_ok': None, 'failed_steps': []}

        # The states of the reports of the nights
        try:
            nights = readReportStates(self.output_dir).get('nights', {})
        except Exception:
            nights = {}

        # The latest report which was done (a report which is running or didn't finish has no time yet)
        reported = [(night_state['reported_at'], night) for night, night_state in nights.items()
                    if night_state.get('reported_at')]
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

        # The free space of the output directory, and whether the processing is paused for it
        disk = {'output_free_gb': _round(availableSpace(self.output_dir)/1024**3, 1),
                'paused_for_space': bool(state.get('paused_for_space'))}

        # The free space of the staging directory, if it is used
        staging_dir = getattr(self.config, 'monitor_staging_dir', '')
        if staging_dir and os.path.isdir(staging_dir):
            disk['staging_free_gb'] = _round(availableSpace(staging_dir)/1024**3, 1)

        # When old data was last cleaned up
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
            """ Add a fault. """

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
