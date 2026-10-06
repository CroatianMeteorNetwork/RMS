""" Night reports for data processed by RMS.MonitorProcessFrameInterface.

The monitor processes every input file into its own results directory (CALSTARS, FTPdetectinfo, recalibrated
platepars) and saves the max pixel and average pixel images of every star extraction chunk as FF-equivalent
image pairs into a night directory:

    <output_dir>/CapturedFiles/<STATION>_<YYYYMMDD>_<HHMMSS>_000000/

where the time in the name is the beginning of the night (sunset). The night report merges the results of all
files of the night into the night directory, like a normal RMS night directory, and generates the calibration
report, the thumbnails, the stacks of all images and of the detections, and the timelapse. Optionally, the night
is archived for upload.

The report can be run from the command line:

    python -m RMS.MonitorNightReport <output_dir> [--night NIGHT_NAME] [--config CONFIG_PATH] [--upload]
"""

from __future__ import print_function, division, absolute_import

import argparse
import datetime
import glob
import json
import os
import shutil
import sys
import traceback

import ephem

import RMS.ConfigReader as cr
from RMS.CaptureDuration import CAPTURE_HORIZON_DEG
from RMS.Formats import CALSTARS
from RMS.Formats import FTPdetectinfo
from RMS.Formats import FFpng
from RMS.Formats.FFfile import validFFName
from RMS.Logger import getLogger, LoggingManager
from RMS.Misc import RmsDateTime


log = getLogger("logger")


# Directory with the night directories (image pairs and merged results)
CAPTURED_DIR_NAME = 'CapturedFiles'

# Directory with the night archives
ARCHIVED_DIR_NAME = 'ArchivedFiles'

# Name of the flag file which marks a processed input file in its results directory
DONE_FLAG_NAME = 'done.flag'

# Creating this file in the output directory triggers the reports of all nights with new data
REPORT_TRIGGER_FILE_NAME = '.report_now'

# Report state, stored in the night directory
REPORT_STATE_FILE_NAME = '.report_state.json'

# Paths of the archives to upload, written by the report to the night directory
REPORT_RESULT_FILE_NAME = 'report_result.json'

# Time format used in the JSON files
JSON_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"


def _toDatetime(ephem_date):
    """ Convert an ephem date to a naive UTC datetime. """
    return ephem.Date(ephem_date).datetime()


def nightInfo(config, dt):
    """ Determine the night which the given time belongs to.

    A night spans from one local solar noon to the next, so a time always belongs to exactly one night,
    including at dusk and in polar regions. The night is named after the sunset at the capture horizon
    following the noon, and ends at the following sunrise. If the Sun doesn't set or rise (polar day or
    night), the night is named after the noon and ends 24 hours later.

    Arguments:
        config: [Config] Configuration with the station coordinates.
        dt: [datetime] Time (UTC).

    Return:
        (night_name, night_start, night_end): [tuple]
            - night_name: [str] Name of the night directory, e.g. XX0001_20251224_163212_000000.
            - night_start: [datetime] Beginning of the night (UTC).
            - night_end: [datetime] End of the night (UTC).
    """

    o = ephem.Observer()
    o.lat = str(config.latitude)
    o.long = str(config.longitude)
    o.elevation = config.elevation
    o.date = dt

    sun = ephem.Sun()

    # Local solar noon before the given time. Transits always exist, even in polar regions
    noon = _toDatetime(o.previous_transit(sun))

    o.date = noon
    o.horizon = CAPTURE_HORIZON_DEG

    try:
        night_start = _toDatetime(o.next_setting(sun))

        o.date = night_start
        night_end = _toDatetime(o.next_rising(sun))

    except (ephem.NeverUpError, ephem.AlwaysUpError):
        night_start = noon
        night_end = noon + datetime.timedelta(days=1)

    station_code = config.stationID.replace("_", "").replace(" ", "").replace(":", "")
    night_name = "{:s}_{:s}_000000".format(station_code, night_start.strftime("%Y%m%d_%H%M%S"))

    return night_name, night_start, night_end


def nightDirPath(output_dir, night_name):
    """ Path of the night directory with the image pairs and the merged results. """
    return os.path.join(output_dir, CAPTURED_DIR_NAME, night_name)


def writeDoneFlag(results_dir, info):
    """ Mark a results directory as processed. The flag contains the given info as JSON.

    Arguments:
        results_dir: [str] Results directory of an input file.
        info: [dict] Info about the processed file, e.g. night, n_images, begin, end, input_dir.
    """

    info = dict(info)
    for key, value in info.items():
        if isinstance(value, datetime.datetime):
            info[key] = value.strftime(JSON_TIME_FORMAT)

    tmp_path = os.path.join(results_dir, DONE_FLAG_NAME + '.tmp')
    with open(tmp_path, 'w') as f:
        json.dump(info, f, indent=4)

    os.replace(tmp_path, os.path.join(results_dir, DONE_FLAG_NAME))


def readDoneFlag(results_dir):
    """ Read the info from the done flag of a results directory.

    Arguments:
        results_dir: [str] Results directory of an input file.

    Return:
        [dict] Info stored in the flag. Empty if the flag is empty (older results) or can't be read.
    """

    try:
        with open(os.path.join(results_dir, DONE_FLAG_NAME)) as f:
            content = f.read().strip()

    except (IOError, OSError):
        return {}

    if not content:
        return {}

    try:
        info = json.loads(content)
    except ValueError:
        return {}

    return info if isinstance(info, dict) else {}


def scanNights(output_dir):
    """ Find the results directories of all processed files, grouped by night.

    Arguments:
        output_dir: [str] Output directory of the monitor.

    Return:
        [dict] {night_name: [results_dir, ...]}. Results without a night (older results) are not included.
    """

    nights = {}

    skip_dirs = {CAPTURED_DIR_NAME, ARCHIVED_DIR_NAME, 'logs'}

    for root, dirs, files in os.walk(output_dir):

        # Don't descend into the night, archive and log directories
        if os.path.abspath(root) == os.path.abspath(output_dir):
            dirs[:] = [d for d in dirs if d not in skip_dirs]

        if DONE_FLAG_NAME in files:

            night_name = readDoneFlag(root).get('night')
            if night_name:
                nights.setdefault(night_name, []).append(root)

            # Results directories don't contain other results
            dirs[:] = []

    for night_name in nights:
        nights[night_name] = sorted(nights[night_name])

    return nights


def readReportState(night_dir):
    """ Read the report state of the night. Empty if the night wasn't reported yet. """

    try:
        with open(os.path.join(night_dir, REPORT_STATE_FILE_NAME)) as f:
            state = json.load(f)

    except (IOError, OSError, ValueError):
        return {}

    return state if isinstance(state, dict) else {}


def unreportedResults(night_dir, results_dirs):
    """ Return the results directories which were not included in the last report of the night. """

    reported = set(readReportState(night_dir).get('files', []))

    return [d for d in results_dirs if os.path.basename(d) not in reported]


def _writeJSON(file_path, data):
    """ Write JSON atomically. """

    tmp_path = file_path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)

    os.replace(tmp_path, file_path)


def _defaultFTPdetectinfo(results_dir):
    """ Return the name of the FTPdetectinfo file in the results directory, or None. """

    names = sorted(f for f in os.listdir(results_dir)
                   if f.startswith('FTPdetectinfo_') and f.endswith('.txt')
                   and FTPdetectinfo.validDefaultFTPdetectinfo(f))

    return names[0] if names else None


def mergeNightResults(night_dir, results_dirs, config):
    """ Merge the CALSTARS, FTPdetectinfo and recalibrated platepars of all files of the night into one file
        each in the night directory, and copy the platepar.

    Arguments:
        night_dir: [str] Night directory.
        results_dirs: [list] Results directories of the files of the night.
        config: [Config] Configuration.

    Return:
        (calstars_name, ftpdetectinfo_name, fps, ff_detected): [tuple]
            - calstars_name: [str] Name of the merged CALSTARS file, None if there are no stars.
            - ftpdetectinfo_name: [str] Name of the merged FTPdetectinfo file.
            - fps: [float] Frame rate from the CALSTARS files, None if not available.
            - ff_detected: [list] FF names with detections which exist in the night directory.
    """

    night_name = os.path.basename(os.path.normpath(night_dir))

    # Remove merged files from a previous report, so the directory always contains exactly one of each
    for file_name in os.listdir(night_dir):
        if (file_name.startswith('CALSTARS_') or file_name.startswith('FTPdetectinfo_')) \
                and file_name.endswith('.txt'):
            os.remove(os.path.join(night_dir, file_name))


    ### Merge CALSTARS ###

    star_list = []
    chunk_frames = None
    fps = None

    for results_dir in results_dirs:
        for calstars_name in sorted(glob.glob(os.path.join(results_dir, 'CALSTARS_*.txt'))):

            calstars_data = CALSTARS.readCALSTARS(results_dir, os.path.basename(calstars_name),
                                                  return_fps=True)
            if calstars_data is False:
                continue

            stars, file_chunk_frames, file_fps = calstars_data

            if (chunk_frames is not None) and (file_chunk_frames != chunk_frames):
                log.warning("CALSTARS {:s} has {:d} frames per chunk, expected {:d}".format(calstars_name,
                    file_chunk_frames, chunk_frames))

            if chunk_frames is None:
                chunk_frames = file_chunk_frames

            if (fps is None) and file_fps:
                fps = file_fps

            star_list += [entry for entry in stars if len(entry) > 1]

    calstars_name = None
    if star_list:

        star_list = sorted(star_list, key=lambda entry: entry[0])

        calstars_name = 'CALSTARS_{:s}.txt'.format(night_name)
        CALSTARS.writeCALSTARS(star_list, night_dir, calstars_name, config.stationID, config.height,
                               config.width, chunk_frames=chunk_frames, fps=fps)


    ### Merge FTPdetectinfo ###

    meteor_list = []
    for results_dir in results_dirs:

        ftpdetectinfo_name = _defaultFTPdetectinfo(results_dir)
        if ftpdetectinfo_name is None:
            continue

        # Read the full format to keep the frame rate of every meteor
        for entry in FTPdetectinfo.readFTPdetectinfo(results_dir, ftpdetectinfo_name):

            ff_name, _, meteor_No, _, meteor_fps, _, _, _, _, rho, phi, meteor_meas = entry

            # Remove the calibration status from the measurements
            meteor_meas = [line[1:] for line in meteor_meas]

            meteor_list.append([ff_name, meteor_No, rho, phi, meteor_meas, meteor_fps])

    meteor_list = sorted(meteor_list, key=lambda entry: (entry[0], entry[1]))

    ftpdetectinfo_name = 'FTPdetectinfo_{:s}.txt'.format(night_name)
    FTPdetectinfo.writeFTPdetectinfo(meteor_list, night_dir, ftpdetectinfo_name, night_dir,
                                     config.stationID, fps if fps else config.fps,
                                     celestial_coords_given=True)


    ### Merge the recalibrated platepars ###

    recalibrated = {}
    for results_dir in results_dirs:

        recalibrated_path = os.path.join(results_dir, config.platepars_recalibrated_name)
        if not os.path.isfile(recalibrated_path):
            continue

        try:
            with open(recalibrated_path) as f:
                recalibrated.update(json.load(f))

        except (IOError, OSError, ValueError):
            log.warning("Could not read {:s}".format(recalibrated_path))

    if recalibrated:
        with open(os.path.join(night_dir, config.platepars_recalibrated_name), 'w') as f:
            json.dump(recalibrated, f, indent=4, sort_keys=True)


    ### Copy the platepar, the config and the mask ###

    for results_dir in results_dirs:
        platepar_path = os.path.join(results_dir, config.platepar_name)
        if os.path.isfile(platepar_path):
            shutil.copy2(platepar_path, os.path.join(night_dir, config.platepar_name))
            break

    for results_dir in results_dirs:
        config_files = sorted(glob.glob(os.path.join(results_dir, '*.config')))
        if config_files:
            shutil.copy2(config_files[0], os.path.join(night_dir, os.path.basename(config_files[0])))
            break

    for results_dir in results_dirs:
        input_dir = readDoneFlag(results_dir).get('input_dir')
        if input_dir and os.path.isfile(os.path.join(input_dir, config.mask_file)):
            shutil.copy2(os.path.join(input_dir, config.mask_file), os.path.join(night_dir, config.mask_file))
            break


    # FF names with detections which have images in the night directory
    night_ffs = set(f for f in os.listdir(night_dir) if validFFName(f))
    ff_detected = sorted(set(entry[0] for entry in meteor_list if entry[0] in night_ffs))

    return calstars_name, ftpdetectinfo_name, fps, ff_detected


def generateNightReport(output_dir, night_name, config, results_dirs=None, upload=False):
    """ Merge the results of a night and generate its report.

    Arguments:
        output_dir: [str] Output directory of the monitor.
        night_name: [str] Name of the night.
        config: [Config] Configuration of the camera.

    Keyword arguments:
        results_dirs: [list] Results directories of the night. Found in the output directory if not given.
        upload: [bool] Archive the night for upload. False by default.

    Return:
        [dict] Report state: reported_at, files (reported files, only updated if all steps succeeded),
            attempted_files, ok_steps, failed_steps, upload_files.
    """

    # Import the report tools here, they are not needed for processing
    from RMS.ArchiveDetections import archiveDetections, generateThumbsAndStacks
    from Utils.CalibrationReport import generateCalibrationReport
    from Utils.GenerateTimelapse import generateTimelapse

    night_dir = nightDirPath(output_dir, night_name)

    if results_dirs is None:
        results_dirs = scanNights(output_dir).get(night_name, [])

    if not os.path.isdir(night_dir):
        os.makedirs(night_dir)

    log.info("Generating the report for night {:s} from {:d} file(s)...".format(night_name, len(results_dirs)))

    # Keep the log archive and the archives inside the output directory
    config.data_dir = output_dir
    config.log_dir = 'logs'

    ok_steps = []
    failed_steps = []
    upload_files = []

    def _runStep(step_name, func):
        log.info("Night report step: {:s}".format(step_name))
        try:
            result = func()
            ok_steps.append(step_name)
            return result

        except Exception as e:
            log.error("Night report step {:s} failed: {:s}".format(step_name, repr(e)))
            log.error("".join(traceback.format_exception(*sys.exc_info())))
            failed_steps.append(step_name)
            return None


    merged = _runStep('merge', lambda: mergeNightResults(night_dir, results_dirs, config))

    if merged is not None:

        calstars_name, ftpdetectinfo_name, fps, ff_detected = merged

        # Use the frame rate of the data, the chunk times are computed from it
        if fps:
            config.fps = fps

        # Use the mask copied to the night directory for the stacks
        config.config_file_path = night_dir

        # Remove the detection stacks of a previous report, their names contain the number of meteors and
        #   would not be overwritten
        for stack_path in glob.glob(os.path.join(night_dir, night_name + '_stack_*_meteors.*')):
            os.remove(stack_path)

        if calstars_name is not None:
            _runStep('calibration_report', lambda: generateCalibrationReport(config, night_dir))

        # Generate the timelapse first, so it can be archived
        timelapse_name = night_name + "_timelapse.mp4"
        if config.timelapse_generate_captured:
            _runStep('timelapse', lambda: generateTimelapse(night_dir, output_file=timelapse_name))

        if upload:

            archived_dir = os.path.join(output_dir, ARCHIVED_DIR_NAME, night_name)

            # Add the files which are not selected from the night directory by default, like processNight
            extra_names = [config.platepar_name, config.platepars_recalibrated_name, config.mask_file,
                           timelapse_name]
            extra_names += [os.path.basename(f) for f in glob.glob(os.path.join(night_dir, '*.config'))]
            extra_files = [os.path.join(night_dir, name) for name in extra_names
                           if os.path.isfile(os.path.join(night_dir, name))]

            # Archiving also generates the thumbnails and the stacks
            archives = _runStep('archive', lambda: archiveDetections(night_dir, archived_dir, ff_detected,
                                                                     config, extra_files=extra_files))
            if archives is not None:
                upload_files = [os.path.abspath(path) for path in archives if path is not None]

        else:
            _runStep('thumbnails_and_stacks', lambda: generateThumbsAndStacks(night_dir, config, ff_detected))

    # Mark the files as reported only if all steps succeeded, so a failed report is retried
    attempted_files = [os.path.basename(d) for d in results_dirs]
    if failed_steps:
        reported_files = readReportState(night_dir).get('files', [])
    else:
        reported_files = attempted_files

    state = {
        'reported_at': RmsDateTime.utcnow().strftime(JSON_TIME_FORMAT),
        'files': reported_files,
        'attempted_files': attempted_files,
        'ok_steps': ok_steps,
        'failed_steps': failed_steps,
        'upload_files': upload_files,
    }

    _writeJSON(os.path.join(night_dir, REPORT_RESULT_FILE_NAME), {'upload_files': upload_files})
    _writeJSON(os.path.join(night_dir, REPORT_STATE_FILE_NAME), state)

    log.info("Night report for {:s} done, failed steps: {:s}".format(night_name,
        ", ".join(failed_steps) if failed_steps else "none"))

    return state


def nightReportWorker(output_dir, night_name, config_path, results_dirs, upload):
    """ Worker process target which generates a night report. Exits with 0 if all steps succeeded, and with 1
        otherwise.
    """

    config = cr.parse(config_path)

    # Log into the output directory
    orig_data_dir, orig_log_dir = config.data_dir, config.log_dir
    config.data_dir = output_dir
    config.log_dir = 'logs'
    LoggingManager().initLogging(config, 'report_{:s}_'.format(night_name))
    config.data_dir, config.log_dir = orig_data_dir, orig_log_dir

    try:
        state = generateNightReport(output_dir, night_name, config, results_dirs=results_dirs, upload=upload)

    except Exception:
        getLogger("logger").error("".join(traceback.format_exception(*sys.exc_info())))
        sys.exit(1)

    sys.exit(0 if not state['failed_steps'] else 1)


def deleteOldNightImages(output_dir, days, now=None):
    """ Delete the image pairs of nights which were reported more than the given number of days ago.

    Arguments:
        output_dir: [str] Output directory of the monitor.
        days: [float] Age of the report in days after which the images are deleted. Nothing is deleted if 0.

    Keyword arguments:
        now: [datetime] Current time, the current UTC time by default.

    Return:
        [list] Names of the nights whose images were deleted.
    """

    if days <= 0:
        return []

    if now is None:
        now = RmsDateTime.utcnow()

    captured_dir = os.path.join(output_dir, CAPTURED_DIR_NAME)
    if not os.path.isdir(captured_dir):
        return []

    nights = scanNights(output_dir)

    cleaned = []
    for night_name in sorted(os.listdir(captured_dir)):

        night_dir = os.path.join(captured_dir, night_name)
        state = readReportState(night_dir)

        if (not state.get('reported_at')) or state.get('failed_steps'):
            continue

        # Keep the images of nights with files which were not reported yet (e.g. files which came in late)
        if unreportedResults(night_dir, nights.get(night_name, [])):
            continue

        reported_at = datetime.datetime.strptime(state['reported_at'], JSON_TIME_FORMAT)
        if (now - reported_at).total_seconds() < days*86400:
            continue

        pair_files = [f for f in os.listdir(night_dir) if FFpng.isPairMaxName(f) or FFpng.isPairAveName(f)]
        if not pair_files:
            continue

        for file_name in pair_files:
            os.remove(os.path.join(night_dir, file_name))

        log.info("Deleted {:d} image files of night {:s}".format(len(pair_files), night_name))
        cleaned.append(night_name)

    return cleaned


def nightBoundsFromName(config, night_name):
    """ Return the (night_start, night_end) of the night with the given name. """

    # The time in the name is the beginning of the night, the night it belongs to is the one after it
    time_str = "_".join(night_name.split('_')[-3:-1])
    night_start = datetime.datetime.strptime(time_str, "%Y%m%d_%H%M%S")

    _, night_start, night_end = nightInfo(config, night_start + datetime.timedelta(minutes=1))

    return night_start, night_end


class NightReporter(object):
    """ Schedules the night reports of one camera from the monitor loop. Reports run in a separate process, one
        at a time.

    Arguments:
        output_dir: [str] Output directory of the monitor.
        config_path: [str] Path to the config file of the camera.

    Keyword arguments:
        report_mode: [str] 'sunrise', 'idle', 'external' or 'none'. The config value is used if None.
        fail_wait_time: [float] Seconds to wait before retrying a failed report. 300 by default.
        log_prefix: [str] Prefix of the log messages, e.g. the camera ID. Empty by default.
    """

    def __init__(self, output_dir, config_path, report_mode=None, fail_wait_time=300, log_prefix=''):

        self.output_dir = output_dir
        self.config_path = config_path
        self.config = cr.parse(config_path)
        self.fail_wait_time = fail_wait_time
        self.log_prefix = log_prefix

        self.report_mode = report_mode if report_mode is not None else self.config.monitor_report_mode

        # Archive after the report if enabled, upload only if uploading is enabled as well
        self.archive = self.config.monitor_archive_upload
        self.upload = self.archive and self.config.upload_enabled

        self.quiet_s = 60*self.config.monitor_report_quiet_min

        # Running report: (process, night_name, results_dirs)
        self.active = None

        # Failed reports: {night_name: {'count': int, 'time': datetime, 'n_files': int}}
        self.failed = {}

        # Nights requested by the trigger file
        self.triggered = set()

        self.upload_manager = None

        self.last_cleanup = None

        # Scan the output directory for nights to report at most this often (seconds)
        self.scan_interval = 30
        self.last_scan = None


    def _log(self, msg):
        return "{:s}{:s}".format(self.log_prefix, msg)


    def _pendingNights(self):
        """ Return {night_name: unreported results dirs} for all nights with unreported results. """

        pending = {}
        for night_name, results_dirs in scanNights(self.output_dir).items():

            new_results = unreportedResults(nightDirPath(self.output_dir, night_name), results_dirs)
            if new_results:
                pending[night_name] = results_dirs

        return pending


    def _lastResultAge(self, results_dirs, now):
        """ Seconds since the most recent result of the night was finished. """

        mtimes = []
        for results_dir in results_dirs:
            try:
                mtimes.append(os.path.getmtime(os.path.join(results_dir, DONE_FLAG_NAME)))
            except OSError:
                pass

        if not mtimes:
            return float('inf')

        last = datetime.datetime.fromtimestamp(max(mtimes), tz=datetime.timezone.utc).replace(tzinfo=None)

        return (now - last).total_seconds()


    def _checkTrigger(self, pending):
        """ Consume the trigger file and remember the nights to report. """

        trigger_path = os.path.join(self.output_dir, REPORT_TRIGGER_FILE_NAME)
        if not os.path.exists(trigger_path):
            return

        try:
            os.remove(trigger_path)
        except OSError:
            pass

        log.info(self._log("Report trigger file found, reporting {:d} night(s) with new data".format(
            len(pending))))

        self.triggered.update(pending.keys())


    def isDue(self, night_name, results_dirs, idle, now):
        """ Check if the report of the night is due.

        Arguments:
            night_name: [str] Name of the night.
            results_dirs: [list] Results directories of the night.
            idle: [bool] True if the camera has no files being processed or waiting to be processed.
            now: [datetime] Current time (UTC).
        """

        # Wait before retrying a failed report, and give up after the second failure until new files arrive
        if night_name in self.failed:
            fail = self.failed[night_name]

            if fail['n_files'] == len(results_dirs):
                if fail['count'] >= 2:
                    return False

                if (now - fail['time']).total_seconds() < self.fail_wait_time:
                    return False

            else:
                del self.failed[night_name]

        if night_name in self.triggered:
            return True

        if self.report_mode not in ('sunrise', 'idle'):
            return False

        # Only report once all data is processed and no new results came in for a while
        if (not idle) or (self._lastResultAge(results_dirs, now) < self.quiet_s):
            return False

        if self.report_mode == 'idle':
            return True

        # In the sunrise mode, the night also has to be over
        _, night_end = nightBoundsFromName(self.config, night_name)

        return now >= night_end


    def _startReport(self, night_name, results_dirs):

        import multiprocessing

        log.info(self._log("Starting the report of night {:s}".format(night_name)))

        proc = multiprocessing.Process(target=nightReportWorker,
            args=(self.output_dir, night_name, self.config_path, results_dirs, self.archive))
        proc.start()

        self.active = (proc, night_name, results_dirs)
        self.triggered.discard(night_name)


    def _finishReport(self, now):

        proc, night_name, results_dirs = self.active
        proc.join()
        self.active = None

        if proc.exitcode == 0:
            log.info(self._log("Report of night {:s} finished".format(night_name)))
            self.failed.pop(night_name, None)

        else:
            fail = self.failed.get(night_name, {'count': 0})
            fail.update({'count': fail['count'] + 1, 'time': now, 'n_files': len(results_dirs)})
            self.failed[night_name] = fail

            log.warning(self._log("Report of night {:s} failed (exit code {}), see the report log".format(
                night_name, proc.exitcode)))

        # Upload the archives
        if self.upload:

            try:
                with open(os.path.join(nightDirPath(self.output_dir, night_name),
                                       REPORT_RESULT_FILE_NAME)) as f:
                    upload_files = json.load(f).get('upload_files', [])

            except (IOError, OSError, ValueError):
                upload_files = []

            if upload_files:
                self._uploadFiles(upload_files)


    def _uploadFiles(self, upload_files):

        from RMS.UploadManager import UploadManager

        if self.upload_manager is None:

            # Keep the upload queue file in the output directory
            upload_config = cr.parse(self.config_path)
            upload_config.data_dir = self.output_dir

            log.info(self._log("Starting the upload manager..."))
            self.upload_manager = UploadManager(upload_config)
            self.upload_manager.start()

        log.info(self._log("Adding files to upload list: {}".format(upload_files)))
        self.upload_manager.addFiles(upload_files)
        self.upload_manager.delayNextUpload(delay=60*self.config.upload_delay)


    def poll(self, idle, now=None):
        """ Check the running report and start the next due one. Call this from the monitor loop.

        Arguments:
            idle: [bool] True if the camera has no files being processed or waiting to be processed.

        Keyword arguments:
            now: [datetime] Current time (UTC), the current time by default.
        """

        if now is None:
            now = RmsDateTime.utcnow()

        # Check the running report
        if self.active is not None:
            if self.active[0].is_alive():
                return

            self._finishReport(now)

        # Delete the images of old nights at most once per hour
        if (self.last_cleanup is None) or ((now - self.last_cleanup).total_seconds() > 3600):
            self.last_cleanup = now
            try:
                deleteOldNightImages(self.output_dir, self.config.monitor_delete_images_days, now=now)
            except Exception as e:
                log.warning(self._log("Deleting old night images failed: {:s}".format(repr(e))))

        trigger = os.path.exists(os.path.join(self.output_dir, REPORT_TRIGGER_FILE_NAME))

        # Without a trigger, nights are only reported automatically in the sunrise and idle modes, once the
        #   camera is idle
        if not (trigger or self.triggered):

            if (self.report_mode not in ('sunrise', 'idle')) or (not idle):
                return

            # Don't scan the output directory too often
            if (self.last_scan is not None) and ((now - self.last_scan).total_seconds() < self.scan_interval):
                return

        self.last_scan = now

        pending = self._pendingNights()

        # Forget triggered nights which have nothing left to report
        self.triggered &= set(pending)

        self._checkTrigger(pending)

        # Report the oldest due night
        for night_name in sorted(pending):
            if self.isDue(night_name, pending[night_name], idle, now):
                self._startReport(night_name, pending[night_name])
                break


    def stop(self, timeout=300):
        """ Wait for the running report and stop the upload manager. """

        if self.active is not None:
            proc, night_name, _ = self.active
            proc.join(timeout=timeout)

            if proc.is_alive():
                log.warning(self._log("Report of night {:s} did not finish in time, terminating".format(
                    night_name)))
                proc.terminate()
                self.active = None

            # Finish the report which completed (e.g. hand its archives over for upload)
            else:
                self._finishReport(RmsDateTime.utcnow())

        if self.upload_manager is not None:
            self.upload_manager.stop()
            self.upload_manager = None


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Generate the night reports for data processed by RMS.MonitorProcessFrameInterface.")

    arg_parser.add_argument('output_dir', type=str,
        help="Output directory of the monitor.")

    arg_parser.add_argument('--night', '-n', type=str, default=None,
        help="Name of the night to report (a directory in CapturedFiles). All nights with new data by default.")

    arg_parser.add_argument('--config', '-c', type=str, default=None,
        help="Path to the config file. By default, the config copied to the results of the night is used.")

    arg_parser.add_argument('--upload', '-u', action='store_true',
        help="Archive the night for upload (the archives are not uploaded by this script).")

    arg_parser.add_argument('--all', '-a', action='store_true',
        help="Report all nights, also the ones without new data.")

    cml_args = arg_parser.parse_args()

    output_dir = os.path.abspath(cml_args.output_dir)
    nights = scanNights(output_dir)

    if cml_args.night is not None:
        if cml_args.night not in nights:
            print("ERROR: No processed files found for night {:s}".format(cml_args.night))
            sys.exit(1)
        night_names = [cml_args.night]

    elif cml_args.all:
        night_names = sorted(nights)

    else:
        night_names = sorted(n for n in nights
                             if unreportedResults(nightDirPath(output_dir, n), nights[n]))

    if not night_names:
        print("No nights to report.")
        sys.exit(0)

    exit_code = 0
    for night_name in night_names:

        # Use the given config, otherwise the config copied to a results directory of the night
        config_path = cml_args.config
        if config_path is None:
            config_files = sorted(glob.glob(os.path.join(nights[night_name][0], '*.config')))
            if not config_files:
                print("ERROR: No config file found for night {:s}, use --config".format(night_name))
                exit_code = 1
                continue
            config_path = config_files[0]

        config = cr.parse(os.path.abspath(config_path))

        state = generateNightReport(output_dir, night_name, config, results_dirs=nights[night_name],
                                    upload=cml_args.upload)

        print("Night {:s}: ok steps: {:s}, failed steps: {:s}".format(night_name,
            ", ".join(state['ok_steps']), ", ".join(state['failed_steps']) or "none"))

        if state['failed_steps']:
            exit_code = 1

    sys.exit(exit_code)
