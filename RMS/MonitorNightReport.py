""" Night reports for data processed by RMS.MonitorProcessFrameInterface.

The monitor processes every input file into its own results directory (CALSTARS, FTPdetectinfo, recalibrated
platepars, and a done.flag with the night the file belongs to), and saves the max pixel and average pixel
images of every star extraction chunk as FF-equivalent image pairs into a night directory, laid out like the
data directory of a normal RMS station:

    <output_dir>/CapturedFiles/<STATION>_<YYYYMMDD>_<HHMMSS>_000000/

The time in the name is the beginning of the night (sunset). The night report merges the results of all files
of the night into the night directory and generates the products of a normal RMS night: the calibration report
and plots, thumbnails, stacks, timelapse, optional products (shower association, FOV KML, flux, observation
summary), and the archive in ArchivedFiles. The most confident platepar of the night is carried forward to the
following data. The report state of all nights is kept in the output directory (REPORT_STATE_FILE_NAME), so it
survives the deletion of old night directories.

The report can be run from the command line:

    python -m RMS.MonitorNightReport <output_dir> [--night NIGHT] [--all] [--config CONFIG] [--no_archive]
"""

from __future__ import print_function, division, absolute_import

import argparse
import bisect
import collections
import copy
import datetime
import glob
import json
import multiprocessing
import os
import platform
import shutil
import signal
import sys
import threading
import time
import traceback

import ephem
import numpy as np

try:
    import fcntl
except ImportError:
    fcntl = None

import RMS.ConfigReader as cr
from RMS.Astrometry.ApplyAstrometry import raDecToXYPP
from RMS.CaptureDuration import CAPTURE_HORIZON_DEG
from RMS.DeleteOldObservations import availableSpace, deleteOldLogfiles, deleteOldObservations, getNightDirs
from RMS.Formats import CALSTARS
from RMS.Formats import FFpng
from RMS.Formats import FTPdetectinfo
from RMS.Formats.FFfile import filenameToDatetime, validFFName
from RMS.Formats.Platepar import Platepar
from RMS.Logger import getLogger, LoggingManager
from RMS.Misc import RmsDateTime
from RMS.Routines.MaskImage import loadMask


log = getLogger("logger")


# Name of the flag file which marks a processed input file in its results directory
DONE_FLAG_NAME = 'done.flag'

# Report state of all nights, kept in the output directory
REPORT_STATE_FILE_NAME = '.night_reports.json'

# Lock file which lets only one monitor (or command line report) work on an output directory at a time
MONITOR_LOCK_FILE_NAME = '.monitor.lock'

# Creating this file in the output directory triggers the reports of all nights with new data
REPORT_TRIGGER_FILE_NAME = '.report_now'

# Directory in the night directory with the ECSV files of the detections
ECSV_DIR_NAME = 'ECSV'

# Time format used in the JSON files
JSON_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"

# Time limits (seconds) of a night report and of the cleanup, after which they are stopped (e.g. stuck on a
#   file system or in an external program), so they don't block the following reports and cleanups
REPORT_TIMEOUT = 4*3600
CLEANUP_TIMEOUT = 3600

# Report steps which decide if a night is reported. Failed optional products are only logged, as in a normal
#   RMS night, so they don't block the night from being reported and uploaded
ESSENTIAL_STEPS = {'merge', 'archive', 'thumbnails_and_stacks'}


# Merged results of a night
MergedNight = collections.namedtuple('MergedNight', ['calstars_name', 'ftpdetectinfo_name', 'fps',
    'chunk_frames', 'ff_detected', 'n_meteors', 'recalibrated'])



### Night naming ###

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
    noon = o.previous_transit(sun).datetime()

    o.date = noon
    o.horizon = CAPTURE_HORIZON_DEG

    try:
        night_start = o.next_setting(sun).datetime()

        o.date = night_start
        night_end = o.next_rising(sun).datetime()

    except (ephem.NeverUpError, ephem.AlwaysUpError):
        night_start = noon
        night_end = noon + datetime.timedelta(days=1)

    night_name = "{:s}_{:s}_000000".format(config.stationID, night_start.strftime("%Y%m%d_%H%M%S"))

    return night_name, night_start, night_end


def nightBoundsFromName(config, night_name):
    """ Return the beginning and the end of the night with the given name.

    Arguments:
        config: [Config] Configuration with the station coordinates.
        night_name: [str] Name of the night.

    Return:
        (night_start, night_end): [tuple of datetime]
    """

    # The time in the name is the beginning of the night, so a minute later is inside the night
    night_start = datetime.datetime.strptime("_".join(night_name.split('_')[-3:-1]), "%Y%m%d_%H%M%S")

    _, night_start, night_end = nightInfo(config, night_start + datetime.timedelta(minutes=1))

    return night_start, night_end


def lockOutputDir(output_dir):
    """ Lock the output directory, so a second monitor (e.g. started by mistake) or a command line report
        can't work on it at the same time. The lock is released when the process ends, also if it is killed,
        so it never goes stale. It is a POSIX record lock, which belongs to this process only, so child
        processes which outlive it (e.g. the upload manager) don't keep the directory locked.

    Arguments:
        output_dir: [str] Output directory of the monitor.

    Return:
        [file] The open lock file, which has to stay open while the directory is used. None if the directory
            is locked by another process.
    """

    lock_file = open(os.path.join(output_dir, MONITOR_LOCK_FILE_NAME), 'a+')

    # Locking is not available on all platforms
    if fcntl is None:
        return lock_file

    try:
        fcntl.lockf(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)

    except (IOError, OSError):
        lock_file.close()
        return None

    # Note the PID of the lock owner, for the error message of the next one
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write("{:d}\n".format(os.getpid()))
    lock_file.flush()

    return lock_file


def lockOwner(output_dir):
    """ Return the PID of the process which locked the output directory, as noted in the lock file. """

    with open(os.path.join(output_dir, MONITOR_LOCK_FILE_NAME)) as f:
        return f.read().strip()


def partialReportTime(config, night_name):
    """ Time of the partial report of the night, the first monitor_partial_report_time after the beginning
        of the night.

    Arguments:
        config: [Config] Configuration.
        night_name: [str] Name of the night.

    Return:
        [datetime] Time of the partial report (UTC), None if partial reports are disabled.
    """

    if not config.monitor_partial_report_time:
        return None

    night_start, _ = nightBoundsFromName(config, night_name)

    report_time = datetime.datetime.strptime(config.monitor_partial_report_time, "%H:%M")
    partial_time = night_start.replace(hour=report_time.hour, minute=report_time.minute, second=0,
                                       microsecond=0)
    if partial_time <= night_start:
        partial_time += datetime.timedelta(days=1)

    return partial_time


def nightDirPath(output_dir, night_name, config):
    """ Path of the night directory with the image pairs and the merged results.

    Arguments:
        output_dir: [str] Output directory of the monitor.
        night_name: [str] Name of the night.
        config: [Config] Configuration.

    Return:
        [str] Path of the night directory.
    """

    return os.path.join(output_dir, config.captured_dir, night_name)


def latestPlateparPath(output_dir, config):
    """ Path of the best platepar of the latest reported night, which is used for the following data. It has
        its own name, so the platepar given to the monitor is never overwritten.

    Arguments:
        output_dir: [str] Output directory of the monitor.
        config: [Config] Configuration.

    Return:
        [str] Path of the platepar.
    """

    return os.path.join(output_dir, 'latest_' + config.platepar_name)



### Done flags and report state ###

def _writeJSON(file_path, data):
    """ Write JSON to a temporary file and rename it, so a reader never sees a partially written file. """

    tmp_path = file_path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)

        # Make sure the data is on the disk before the rename, so a power loss doesn't leave a broken file
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, file_path)


def readStateFile(file_path, default):
    """ Read a JSON state file of the monitor. A damaged file (e.g. after a disk error) is moved aside to
        <name>.corrupt, and the monitor starts over with the default instead of failing at every start.

    Arguments:
        file_path: [str] Path of the state file.
        default: [object] State returned if the file doesn't exist or is damaged.

    Return:
        [object] The state.
    """

    if not os.path.isfile(file_path):
        return default

    try:
        with open(file_path) as f:
            return json.load(f)

    except ValueError:
        log.warning("The state file {:s} is damaged, it was moved to {:s}.corrupt and the monitor starts "
                    "over".format(file_path, file_path))
        os.replace(file_path, file_path + '.corrupt')

        return default


def writeDoneFlag(results_dir, info):
    """ Mark a results directory as processed. The flag contains the given info as JSON.

    Arguments:
        results_dir: [str] Results directory of an input file.
        info: [dict] Info about the processed file: night, mask_path, dark_applied, flat_applied. A file which
            was skipped by the cutoff of its night has only night and skipped.
    """

    _writeJSON(os.path.join(results_dir, DONE_FLAG_NAME), info)


def readDoneFlag(results_dir):
    """ Read the info from the done flag of a results directory.

    Arguments:
        results_dir: [str] Results directory of an input file.

    Return:
        [dict] Info stored in the flag. Empty if the flag is empty (results from before the night reports).
    """

    try:
        with open(os.path.join(results_dir, DONE_FLAG_NAME)) as f:
            return json.load(f)

    except ValueError:
        return {}


def scanNights(output_dir, config):
    """ Find the results of all processed files, grouped by night.

    Arguments:
        output_dir: [str] Output directory of the monitor.
        config: [Config] Configuration.

    Return:
        [dict] {night_name: {results_dir: info}}, where results_dir is relative to the output directory and
            info is the content of its done flag. Results from before the night reports and files skipped by
            the cutoff of their night are not included.
    """

    nights = {}

    for root, dirs, files in os.walk(output_dir):

        # Don't descend into the night, archive and log directories
        if root == output_dir:
            dirs[:] = [d for d in dirs if d not in (config.captured_dir, config.archived_dir, 'logs')]

        if DONE_FLAG_NAME in files:

            # Files skipped by the cutoff of their night have no results
            info = readDoneFlag(root)
            if ('night' in info) and (not info.get('skipped')):
                nights.setdefault(info['night'], {})[os.path.relpath(root, output_dir)] = info

            # Results directories don't contain other results
            dirs[:] = []

    return nights


def readReportStates(output_dir):
    """ Read the report state of all nights.

    Arguments:
        output_dir: [str] Output directory of the monitor.

    Return:
        [dict] {'nights': {night_name: {files, reported_at, failed_steps, upload_files}},
            'latest_platepar_night': night_name}
    """

    # A damaged state makes all nights be reported again
    return readStateFile(os.path.join(output_dir, REPORT_STATE_FILE_NAME),
                         {'nights': {}, 'latest_platepar_night': None})


def updateReportState(output_dir, night_name=None, latest_platepar_night=None, **night_state):
    """ Update the report state of a night and/or the night of the latest platepar.

    Arguments:
        output_dir: [str] Output directory of the monitor.

    Keyword arguments:
        night_name: [str] Night whose state is updated with the given night_state values. None by default.
        latest_platepar_night: [str] Night of the latest carried forward platepar. None by default.
        **night_state: Values of the night state to update, e.g. files, reported_at, failed_steps,
            upload_files.
    """

    states = readReportStates(output_dir)

    if night_name is not None:
        states['nights'].setdefault(night_name, {}).update(night_state)

    if latest_platepar_night is not None:
        states['latest_platepar_night'] = latest_platepar_night

    _writeJSON(os.path.join(output_dir, REPORT_STATE_FILE_NAME), states)


def unreportedResults(states, night_name, results):
    """ Return the results of the night which were not included in its last successful report.

    Arguments:
        states: [dict] Report states, see readReportStates.
        night_name: [str] Name of the night.
        results: [dict] {results_dir: info} of the night, see scanNights.

    Return:
        [list] Unreported results directories.
    """

    reported = set(states['nights'].get(night_name, {}).get('files', []))

    return [results_dir for results_dir in results if results_dir not in reported]



### Night products ###

def detectedImageNames(meteor_list, image_names):
    """ Return the names of all images which cover the duration of the detected meteors. A meteor longer than
        one chunk is spread over several images, so all images from the one with the first pick (the FF name
        of the meteor) to the one with the last pick are returned.

    Arguments:
        meteor_list: [list] Meteors as [ff_name, meteor_No, rho, phi, meteor_meas, meteor_fps], where the
            first element of every measurement is the frame relative to the beginning of the FF file.
        image_names: [list] FF names of the images in the night directory.

    Return:
        [list] Sorted FF names of the images covering the meteors.
    """

    # Every image covers the time from its beginning to the beginning of the next image
    image_names = sorted(image_names, key=filenameToDatetime)
    image_times = [filenameToDatetime(file_name) for file_name in image_names]
    image_index = {file_name: i for i, file_name in enumerate(image_names)}

    detected = set()
    for ff_name, _, _, _, meteor_meas, meteor_fps in meteor_list:

        if ff_name not in image_index:
            continue

        first = image_index[ff_name]

        # Find the image with the last pick
        last_frame = max(line[0] for line in meteor_meas)
        last_time = image_times[first] + datetime.timedelta(seconds=last_frame/meteor_fps)
        last = max(first, bisect.bisect_right(image_times, last_time) - 1)

        detected.update(image_names[first:last + 1])

    return sorted(detected)


def mergeNightResults(night_dir, output_dir, results, config):
    """ Merge the CALSTARS, FTPdetectinfo and recalibrated platepars of all files of the night into one file
        each in the night directory, collect the ECSV files into its ECSV directory, and copy the platepar,
        the config and the mask there.

    Arguments:
        night_dir: [str] Night directory.
        output_dir: [str] Output directory of the monitor.
        results: [dict] {results_dir: info} of the night, see scanNights.
        config: [Config] Configuration.

    Return:
        [MergedNight] Merged results.
    """

    night_name = os.path.basename(night_dir)
    results_paths = [os.path.join(output_dir, results_dir) for results_dir in sorted(results)]

    # Remove merged files from a previous report, so the directory always contains exactly one of each
    for file_name in os.listdir(night_dir):
        if file_name.startswith(('CALSTARS_', 'FTPdetectinfo_')) and file_name.endswith('.txt'):
            os.remove(os.path.join(night_dir, file_name))


    ### Merge CALSTARS ###

    star_list = []
    chunk_frames = None
    fps = None

    for results_path in results_paths:
        for calstars_path in glob.glob(os.path.join(results_path, 'CALSTARS_*.txt')):

            stars, chunk_frames, fps = CALSTARS.readCALSTARS(results_path, os.path.basename(calstars_path),
                                                             return_fps=True)
            star_list += stars

    calstars_name = None
    if star_list:
        calstars_name = 'CALSTARS_{:s}.txt'.format(night_name)
        CALSTARS.writeCALSTARS(sorted(star_list, key=lambda entry: entry[0]), night_dir, calstars_name,
                               config.stationID, config.height, config.width, chunk_frames=chunk_frames,
                               fps=fps)


    ### Merge FTPdetectinfo ###

    meteor_list = []
    for results_path in results_paths:

        ftpdetectinfo_names = [file_name for file_name in sorted(os.listdir(results_path))
                               if file_name.startswith('FTPdetectinfo_')
                               and FTPdetectinfo.validDefaultFTPdetectinfo(file_name)]

        for ftpdetectinfo_name in ftpdetectinfo_names:

            # Read the full format to keep the frame rate of every meteor, and remove the calibration status
            #   from the measurements to get the input format of the writer
            for entry in FTPdetectinfo.readFTPdetectinfo(results_path, ftpdetectinfo_name):
                ff_name, _, meteor_No, _, meteor_fps, _, _, _, _, rho, phi, meteor_meas = entry
                meteor_list.append([ff_name, meteor_No, rho, phi, [line[1:] for line in meteor_meas],
                                    meteor_fps])

    ftpdetectinfo_name = 'FTPdetectinfo_{:s}.txt'.format(night_name)
    FTPdetectinfo.writeFTPdetectinfo(sorted(meteor_list, key=lambda entry: (entry[0], entry[1])), night_dir,
        ftpdetectinfo_name, night_dir, config.stationID, fps if fps else config.fps,
        calibration="Recalibrated with RMS on: " + RmsDateTime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f UTC"),
        celestial_coords_given=True)


    ### Merge the recalibrated platepars ###

    recalibrated = {}
    for results_path in results_paths:

        recalibrated_path = os.path.join(results_path, config.platepars_recalibrated_name)
        if os.path.isfile(recalibrated_path):
            with open(recalibrated_path) as f:
                recalibrated.update(json.load(f))

    if recalibrated:
        _writeJSON(os.path.join(night_dir, config.platepars_recalibrated_name), recalibrated)


    ### Collect the ECSV files ###

    # Detections of different files which begin in the same second have ECSV files of the same name, so the
    #   names are made unique when they are collected into the night directory
    ecsv_dir = os.path.join(night_dir, ECSV_DIR_NAME)
    if os.path.isdir(ecsv_dir):
        shutil.rmtree(ecsv_dir)

    for results_path in results_paths:
        for ecsv_path in sorted(glob.glob(os.path.join(results_path, '*.ecsv'))):

            os.makedirs(ecsv_dir, exist_ok=True)

            base_name, ext = os.path.splitext(os.path.basename(ecsv_path))
            file_name = base_name + ext
            collision = 1
            while os.path.isfile(os.path.join(ecsv_dir, file_name)):
                collision += 1
                file_name = '{:s}_{:d}{:s}'.format(base_name, collision, ext)

            shutil.copy2(ecsv_path, os.path.join(ecsv_dir, file_name))


    ### Copy the platepar, the config and the mask ###

    first_results = results_paths[0]

    shutil.copy2(os.path.join(first_results, config.platepar_name), night_dir)

    for config_path in glob.glob(os.path.join(first_results, '*.config')):
        shutil.copy2(config_path, night_dir)

    mask_path = results[sorted(results)[0]].get('mask_path')
    if mask_path and os.path.isfile(mask_path):
        shutil.copy2(mask_path, os.path.join(night_dir, config.mask_file))


    # Images in the night directory which cover the detections
    night_ffs = [file_name for file_name in os.listdir(night_dir) if validFFName(file_name)]
    ff_detected = detectedImageNames(meteor_list, night_ffs)

    return MergedNight(calstars_name, ftpdetectinfo_name, fps, chunk_frames, ff_detected, len(meteor_list),
                       recalibrated)


def plateparResidual(platepar):
    """ Median distance (px) between the image positions of the stars the platepar was fitted on and their
        catalog positions projected with the platepar.

    Arguments:
        platepar: [Platepar instance] Recalibrated platepar (ApplyRecalibrate) with a star list. Its entries
            are [jd, y, x, intensity, ra, dec, mag], as the image coordinates come from CALSTARS (Y X).

    Return:
        [float] Median residual in pixels.
    """

    star_list = np.array(platepar.star_list, dtype=np.float64)

    jd, y, x, ra, dec = star_list[:, 0], star_list[:, 1], star_list[:, 2], star_list[:, 4], star_list[:, 5]

    x_cat, y_cat = raDecToXYPP(ra, dec, np.mean(jd), platepar)

    return float(np.median(np.hypot(x - x_cat, y - y_cat)))


def selectBestNightPlatepar(recalibrated, use_flat=False):
    """ Select the most confident platepar of the night: the successfully recalibrated platepar with the most
        matched stars, and the lowest residual among those.

    Arguments:
        recalibrated: [dict] Recalibrated platepars as stored in the JSON file, keyed by the FF name.

    Keyword arguments:
        use_flat: [bool] Whether the data was flat fielded. False by default.

    Return:
        (ff_name, platepar, n_stars, residual): [tuple] The selected platepar, None if no platepar was
            recalibrated successfully.
    """

    # Number of matched stars of the successfully recalibrated platepars
    candidates = {ff_name: len(pp_dict['star_list']) for ff_name, pp_dict in recalibrated.items()
                  if pp_dict.get('auto_recalibrated') and pp_dict.get('star_list')}

    if not candidates:
        return None

    # Compare the residuals of the platepars with the most stars
    n_stars = max(candidates.values())

    best = None
    for ff_name in sorted(ff for ff, n in candidates.items() if n == n_stars):

        platepar = Platepar()
        platepar.loadFromDict(recalibrated[ff_name], use_flat=use_flat)
        residual = plateparResidual(platepar)

        if (best is None) or (residual < best[3]):
            best = (ff_name, platepar, n_stars, residual)

    return best


def _writePlatepar(platepar, file_path):
    """ Write the platepar to a temporary file and rename it, as other processes may be reading it. """

    platepar.write(file_path + '.tmp', fmt='json')
    os.replace(file_path + '.tmp', file_path)


def saveNightPlatepar(platepar, night_dir, night_name, output_dir, config):
    """ Save the best platepar of the night to the night directory (as at the end of a normal RMS night), and
        carry it forward to the following data if monitor_update_platepar is set and no newer night has been
        reported (e.g. when an old night is reported again).

    Arguments:
        platepar: [Platepar instance] Best platepar of the night.
        night_dir: [str] Night directory.
        night_name: [str] Name of the night.
        output_dir: [str] Output directory of the monitor.
        config: [Config] Configuration.
    """

    _writePlatepar(platepar, os.path.join(night_dir, config.platepar_name))

    latest_night = readReportStates(output_dir)['latest_platepar_night']

    if config.monitor_update_platepar and ((latest_night is None) or (night_name >= latest_night)):
        _writePlatepar(platepar, latestPlateparPath(output_dir, config))
        updateReportState(output_dir, latest_platepar_night=night_name)
        log.info("Best platepar saved for the following data: {:s}".format(
            latestPlateparPath(output_dir, config)))


def plotNightCalibrationVariation(night_dir, night_name, recalibrated, config, ff_frames):
    """ Plot the variation of the pointing and the photometric offset through the night from the recalibrated
        platepars, relative to the platepar of the night.

    Arguments:
        night_dir: [str] Night directory.
        night_name: [str] Name of the night, used as the prefix of the plot names.
        recalibrated: [dict] Recalibrated platepars as stored in the JSON file, keyed by the FF name.
        config: [Config] Configuration.
        ff_frames: [int] Number of frames per chunk.

    Return:
        [bool] True if the plots were saved.
    """

    from RMS.Astrometry.ApplyRecalibrate import plotCalibrationVariation

    platepar = Platepar()
    platepar.read(os.path.join(night_dir, config.platepar_name))

    recalibrated_platepars = {}
    for ff_name, pp_dict in recalibrated.items():
        recalibrated_platepars[ff_name] = Platepar()
        recalibrated_platepars[ff_name].loadFromDict(pp_dict)

    return plotCalibrationVariation(recalibrated_platepars, platepar, config, night_dir, night_name,
                                    ff_frames=ff_frames)


def writeObservationSummary(night_dir, night_name, config, frames_per_file, n_detections, photometry_good):
    """ Write the observation summary of the night (observation_summary.txt and .json), as at the end of a
        normal RMS night. The parameters which RMS records at the start of the capture are filled in from the
        data, as there is no live capture (e.g. the camera is not queried).

    Arguments:
        night_dir: [str] Night directory.
        night_name: [str] Name of the night.
        config: [Config] Configuration, with data_dir set to the output directory of the monitor.
        frames_per_file: [int] Number of frames per FF-equivalent image pair.
        n_detections: [int] Number of detections in the night.
        photometry_good: [bool] True if a platepar was recalibrated successfully during the night.

    Return:
        [tuple] Paths of the text and the JSON summary.
    """

    from RMS.Formats import ObservationSummary as obs

    # The observation spans from the first to the end of the last image of the night, or the whole night if
    #   no images were saved
    ff_names = sorted(file_name for file_name in os.listdir(night_dir) if FFpng.isPairMaxName(file_name))
    if ff_names:
        obs_start = filenameToDatetime(ff_names[0])
        obs_end = filenameToDatetime(ff_names[-1]) + datetime.timedelta(seconds=frames_per_file/config.fps)
    else:
        obs_start, obs_end = nightBoundsFromName(config, night_name)

    # Start a new summary for every night
    db_path = os.path.join(config.data_dir, "observation.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = obs.getObsDBConn(config)
    obs.addObsParam(conn, "start_time", obs_start.replace(tzinfo=datetime.timezone.utc))
    obs.addObsParam(conn, "duration_from_start_of_observation", round((obs_end - obs_start).total_seconds()))
    obs.addObsParam(conn, "stationID", config.stationID)
    obs.addObsParam(conn, "hardware_version", platform.machine())
    obs.addObsParam(conn, "camera_information", "Processed from recorded data")
    obs.addObsParam(conn, "detections_after_ml", n_detections)
    obs.addObsParam(conn, "photometry_good", str(photometry_good))
    conn.close()

    return obs.finalizeObservationSummary(config, night_dir, frames_per_file=frames_per_file)


def scaledThumbStack(thumb_stack, chunk_frames, ff_frames=256):
    """ Scale the number of images stacked in one thumbnail from FF files to frame chunks of a different
        length, so a thumbnail covers the same number of frames.

    Arguments:
        thumb_stack: [int] Number of FF files stacked in one thumbnail (config.thumb_stack).
        chunk_frames: [int] Number of frames per chunk, None if unknown.

    Keyword arguments:
        ff_frames: [int] Number of frames in an FF file. 256 by default.

    Return:
        [int] Number of chunks stacked in one thumbnail, at least 1.
    """

    if not chunk_frames:
        return thumb_stack

    return max(1, int(round(thumb_stack*ff_frames/chunk_frames)))


def generateNightReport(output_dir, night_name, config, results=None, archive=True):
    """ Merge the results of a night and generate its report.

    Arguments:
        output_dir: [str] Output directory of the monitor.
        night_name: [str] Name of the night.
        config: [Config] Configuration of the camera.

    Keyword arguments:
        results: [dict] {results_dir: info} of the night, see scanNights. Found in the output directory if not
            given.
        archive: [bool] Archive the night into ArchivedFiles, like a normal RMS night. True by default. If
            False, only the thumbnails and the stacks are generated.

    Return:
        [dict] Report state of the night: files (only updated if all essential steps succeeded), reported_at,
            ok_steps, failed_steps, upload_files.
    """

    # The report tools are imported here, as they are not needed for the processing
    from RMS.ArchiveDetections import archiveDetections, generateThumbsAndStacks
    from Utils.CalibrationReport import generateCalibrationReport
    from Utils.GenerateTimelapse import generateTimelapse

    if results is None:
        results = scanNights(output_dir, config).get(night_name, {})

    night_dir = nightDirPath(output_dir, night_name, config)
    os.makedirs(night_dir, exist_ok=True)

    log.info("Generating the report for night {:s} from {:d} file(s)...".format(night_name, len(results)))

    ok_steps = []
    failed_steps = []

    def _runStep(step_name, func):
        """ Run a report step, logging its errors. Return its result, None if it failed. """

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


    merged = _runStep('merge', lambda: mergeNightResults(night_dir, output_dir, results, config))

    upload_files = []
    if merged is not None:

        ### Configuration of the night ###
        config = copy.deepcopy(config)

        # Keep the logs, archives and observation database inside the output directory, and use the mask and
        #   the platepar copied to the night directory
        config.data_dir = output_dir
        config.log_dir = 'logs'
        config.config_file_path = night_dir

        # The chunk times are computed from the frame rate of the data
        if merged.fps:
            config.fps = merged.fps

        # Use the same dark and flat settings as the processing, so e.g. the photometry doesn't correct the
        #   vignetting of flat fielded images again
        config.use_dark = any(info.get('dark_applied', False) for info in results.values())
        config.use_flat = any(info.get('flat_applied', False) for info in results.values())

        # The thumbnail stacking is set up for FF files with 256 frames. Scale it to the chunk length, so
        #   every thumbnail covers the same number of frames as with normal FF files
        config.thumb_stack = scaledThumbStack(config.thumb_stack, merged.chunk_frames)

        chunk_frames = merged.chunk_frames if merged.chunk_frames else 256
        ftpdetectinfo_path = os.path.join(night_dir, merged.ftpdetectinfo_name)

        ### ###


        # Save the most confident platepar of the night and carry it forward to the following data
        best_platepar = None
        if merged.recalibrated:
            best_platepar = _runStep('platepar_selection',
                lambda: selectBestNightPlatepar(merged.recalibrated, use_flat=config.use_flat))

        if best_platepar is not None:
            ff_name, platepar, n_stars, residual = best_platepar
            log.info("Best platepar of the night: {:s}, {:d} stars, median residual {:.3f} px".format(ff_name,
                n_stars, residual))
            saveNightPlatepar(platepar, night_dir, night_name, output_dir, config)

        if merged.calstars_name is not None:
            _runStep('calibration_report',
                lambda: generateCalibrationReport(config, night_dir, min_size=1000))

        if merged.recalibrated:
            _runStep('calibration_variation', lambda: plotNightCalibrationVariation(night_dir, night_name,
                merged.recalibrated, config, chunk_frames))


        ### Optional products (off by default) ###

        platepar = Platepar()
        platepar.read(os.path.join(night_dir, config.platepar_name), use_flat=config.use_flat)

        mask = None
        if os.path.isfile(os.path.join(night_dir, config.mask_file)):
            mask = loadMask(os.path.join(night_dir, config.mask_file))

        if config.monitor_shower_association:
            from Utils.ShowerAssociation import showerAssociation
            _runStep('shower_association', lambda: showerAssociation(config, [ftpdetectinfo_path],
                save_plot=True, plot_activity=True, color_map=config.shower_color_map,
                sporadic_color=config.sporadic_color))

        if config.monitor_fov_kml:
            from Utils.FOVKML import fovKML
            _runStep('fov_kml', lambda: [fovKML(night_dir, platepar, mask=mask, plot_station=False,
                area_ht=area_ht) for area_ht in [100000, 70000, 25000]])

        if config.monitor_flux:
            from Utils.Flux import prepareFluxFiles
            _runStep('flux', lambda: prepareFluxFiles(config, night_dir, ftpdetectinfo_path, mask=mask,
                                                      platepar=platepar))

        if config.monitor_observation_summary:
            _runStep('observation_summary', lambda: writeObservationSummary(night_dir, night_name, config,
                chunk_frames, merged.n_meteors, best_platepar is not None))

        ### ###


        # Generate the timelapse before archiving, so it is archived
        timelapse_name = night_name + "_timelapse.mp4"
        if config.timelapse_generate_captured:
            _runStep('timelapse', lambda: generateTimelapse(night_dir, output_file=timelapse_name))

        # Remove the detection stacks of a previous report, their names contain the number of meteors
        for stack_path in glob.glob(os.path.join(night_dir, night_name + '_stack_*_meteors.*')):
            os.remove(stack_path)

        if archive:

            # Archive the night from scratch, so no files of a previous report remain in the archive
            archived_dir = os.path.join(output_dir, config.archived_dir, night_name)
            shutil.rmtree(archived_dir, ignore_errors=True)

            # Add the files which are not selected from the night directory, as processNight does
            extra_files = [os.path.join(night_dir, file_name) for file_name in os.listdir(night_dir)
                           if file_name.endswith(('.config', '.kml', '.mp4', '.json', '.ecsv', '.cal'))
                           or (file_name == config.mask_file)]

            ecsv_dir = os.path.join(night_dir, ECSV_DIR_NAME)
            if os.path.isdir(ecsv_dir):
                extra_files += sorted(glob.glob(os.path.join(ecsv_dir, '*.ecsv')))

            # Archiving also generates the thumbnails and the stacks
            archives = _runStep('archive', lambda: archiveDetections(night_dir, archived_dir,
                merged.ff_detected, config, extra_files=extra_files))

            if archives is not None:
                upload_files = [os.path.abspath(path) for path in archives if path is not None]

        else:
            _runStep('thumbnails_and_stacks', lambda: generateThumbsAndStacks(night_dir, config,
                                                                              merged.ff_detected))


    # Mark the files as reported only if all essential steps succeeded, otherwise the report is retried
    night_state = {
        'reported_at': RmsDateTime.utcnow().strftime(JSON_TIME_FORMAT),
        'ok_steps': ok_steps,
        'failed_steps': failed_steps,
        'upload_files': upload_files,
    }

    if not (ESSENTIAL_STEPS & set(failed_steps)):
        night_state['files'] = sorted(results)

    updateReportState(output_dir, night_name, **night_state)

    log.info("Night report for {:s} done, failed steps: {:s}".format(night_name,
        ", ".join(failed_steps) if failed_steps else "none"))

    return readReportStates(output_dir)['nights'][night_name]



### Cleanup ###

def deleteOldNightImages(output_dir, config, states, nights, now=None):
    """ Delete the image pairs of nights which were reported more than monitor_delete_images_days ago and have
        no unreported files. The reports and the archives are kept.

    Arguments:
        output_dir: [str] Output directory of the monitor.
        config: [Config] Configuration.
        states: [dict] Report states, see readReportStates.
        nights: [dict] Results of all nights, see scanNights.

    Keyword arguments:
        now: [datetime] Current time, the current UTC time by default.

    Return:
        [list] Names of the nights whose images were deleted.
    """

    if config.monitor_delete_images_days <= 0:
        return []

    if now is None:
        now = RmsDateTime.utcnow()

    cleaned = []
    for night_name, night_state in sorted(states['nights'].items()):

        night_dir = nightDirPath(output_dir, night_name, config)
        if (not os.path.isdir(night_dir)) or ('files' not in night_state) \
                or unreportedResults(states, night_name, nights.get(night_name, {})):
            continue

        reported_at = datetime.datetime.strptime(night_state['reported_at'], JSON_TIME_FORMAT)
        if (now - reported_at).total_seconds() < config.monitor_delete_images_days*86400:
            continue

        pair_files = [file_name for file_name in os.listdir(night_dir) if FFpng.isPairName(file_name)]
        for file_name in pair_files:
            os.remove(os.path.join(night_dir, file_name))

        if pair_files:
            log.info("Deleted {:d} image files of night {:s}".format(len(pair_files), night_name))
            cleaned.append(night_name)

    return cleaned


def directorySize(dir_path):
    """ Total size of the files in a directory tree, in bytes. """

    return sum(os.path.getsize(os.path.join(root, file_name))
               for root, _, files in os.walk(dir_path) for file_name in files)


def pruneResults(output_dir, config, nights):
    """ Delete the results and the report states of the nights whose directory was deleted, except the done
        flags, which mark the input files as processed. Such nights are not reported again (see
        NightReporter), so their results (CALSTARS, FTPdetectinfo, recalibrated platepars, config copies)
        and report states are not needed anymore.

    Arguments:
        output_dir: [str] Output directory of the monitor.
        config: [Config] Configuration.
        nights: [dict] {night_name: {results_dir: info}}, see scanNights.

    Return:
        [int] Number of pruned results directories.
    """

    n_pruned = 0
    for night_name, results in nights.items():

        if os.path.isdir(nightDirPath(output_dir, night_name, config)):
            continue

        for results_dir in results:
            results_path = os.path.join(output_dir, results_dir)

            file_names = [file_name for file_name in os.listdir(results_path) if file_name != DONE_FLAG_NAME]
            for file_name in file_names:
                path = os.path.join(results_path, file_name)
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)

            n_pruned += bool(file_names)

    if n_pruned:
        log.info("Deleted the results of {:d} processed files of deleted nights, keeping their done "
                 "flags".format(n_pruned))

    # Forget the report states of the deleted nights, which are not reported again, so the state file doesn't
    #   grow. The cleanup never runs at the same time as a report, which also writes the states
    states = readReportStates(output_dir)
    deleted = [night_name for night_name in states['nights']
               if not os.path.isdir(nightDirPath(output_dir, night_name, config))]
    if deleted:
        for night_name in deleted:
            del states['nights'][night_name]

        _writeJSON(os.path.join(output_dir, REPORT_STATE_FILE_NAME), states)

    return n_pruned


def freeSpace(output_dir, config, needed_bytes):
    """ Delete the oldest nights (the night directory and its archives) until the output directory has the
        given free space. The latest night is never deleted. If other data takes the disk (e.g. the recordings
        on a shared disk), only the older nights of the monitor are deleted.

    Arguments:
        output_dir: [str] Output directory of the monitor.
        config: [Config] Configuration.
        needed_bytes: [float] Free space needed, in bytes.

    Return:
        [bool] True if there is enough free space.
    """

    captured_path = os.path.join(output_dir, config.captured_dir)
    archived_path = os.path.join(output_dir, config.archived_dir)

    night_names = sorted(set(getNightDirs(captured_path, config.stationID))
                         | set(getNightDirs(archived_path, config.stationID)))

    for night_name in night_names[:-1]:

        if availableSpace(output_dir) >= needed_bytes:
            return True

        log.info("Deleting night {:s} to free space for the next night".format(night_name))
        shutil.rmtree(os.path.join(captured_path, night_name), ignore_errors=True)

        # The archive directory and the archives of the night
        if os.path.isdir(archived_path):
            for file_name in os.listdir(archived_path):
                if file_name.startswith(night_name):
                    path = os.path.join(archived_path, file_name)
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        os.remove(path)

    if availableSpace(output_dir) >= needed_bytes:
        return True

    log.warning("{:.1f} GB are needed in {:s} for the next night, but only {:.1f} GB are free after deleting "
                "the old nights, the rest of the disk is used by other data".format(needed_bytes/1024**3,
                    output_dir, availableSpace(output_dir)/1024**3))

    return False


def cleanupOldData(output_dir, config):
    """ Delete old data from the output directory with the data management of normal RMS
        (DeleteOldObservations): old night directories in CapturedFiles and ArchivedFiles (by the number of
        directories to keep, the quotas, and the free space needed for the next night), old archives and log
        files. Night directories with unreported files are kept unless the space is needed. The image pairs of
        reported nights are deleted after monitor_delete_images_days.

    Arguments:
        output_dir: [str] Output directory of the monitor.
        config: [Config] Configuration of the camera.

    Return:
        [bool] True if there's enough free space for the next night.
    """

    states = readReportStates(output_dir)
    nights = scanNights(output_dir, config)

    deleteOldNightImages(output_dir, config, states, nights)

    if not config.monitor_delete_old_data:
        return True

    # Manage the output directory like the RMS data directory
    config = copy.deepcopy(config)
    config.data_dir = output_dir
    config.log_dir = 'logs'

    # Keep the night directories back to the oldest night with unreported files
    pending = [night_name for night_name, results in nights.items()
               if unreportedResults(states, night_name, results)]

    captured_path = os.path.join(output_dir, config.captured_dir)
    night_dirs = getNightDirs(captured_path, config.stationID)

    if pending and (config.capt_dirs_to_keep > 0):
        config.capt_dirs_to_keep = max(config.capt_dirs_to_keep,
                                       len([d for d in night_dirs if d >= min(pending)]))

    # Delete by the numbers of directories to keep and the quotas. The space is freed below, as the RMS
    #   estimate of the space needed for the next night comes from the capture settings, which don't describe
    #   the data of the monitor, and its deletion loop can delete the latest night
    deleteOldObservations(output_dir, config.captured_dir, config.archived_dir, config, needed_bytes=0)

    # The space needed for the next night is the size of the largest of the last nights
    night_dirs = getNightDirs(captured_path, config.stationID)
    needed_bytes = config.extra_space_gb*1024**3
    if night_dirs:
        needed_bytes += max(directorySize(os.path.join(captured_path, night_dir))
                            for night_dir in night_dirs[-3:])

    enough_space = freeSpace(output_dir, config, needed_bytes)

    pruneResults(output_dir, config, nights)

    # Delete the old logs of the monitor processes, which have a prefix
    deleteOldLogfiles(output_dir, config, pattern='*log_*.log*')

    if not enough_space:
        log.warning("Not enough free space in {:s} for the next night!".format(output_dir))

    return enough_space



### Worker processes ###

def stopProcess(proc, name, timeout=10):
    """ Stop a process with SIGTERM, and with SIGKILL if it doesn't end. The waits are bounded, so a process
        stuck in an uninterruptible call (e.g. on a hung file system) can't block the caller.

    Arguments:
        proc: [multiprocessing.Process] The process.
        name: [str] Name of the process for the log.

    Keyword arguments:
        timeout: [float] Seconds to wait after each signal. 10 by default.

    Return:
        [bool] True if the process ended.
    """

    proc.terminate()
    proc.join(timeout)

    if proc.is_alive():
        proc.kill()
        proc.join(timeout)

    if proc.is_alive():
        log.error("{:s} (PID {}) could not be stopped, it is left behind".format(name, proc.pid))
        return False

    return True


def exitWithMonitor(interval=2.0):
    """ End this worker process when the monitor process which started it is gone, e.g. when it was killed,
        so the worker doesn't keep running next to a restarted monitor. Call it at the start of the worker.

    Keyword arguments:
        interval: [float] Seconds between the checks. 2 by default.
    """

    # Nothing to watch outside of a worker process (e.g. a report from the command line)
    parent = multiprocessing.parent_process()
    if parent is None:
        return

    monitor_pid = parent.pid

    # With the fork start method the worker is orphaned when the monitor dies, with other start methods (e.g.
    #   forkserver) the monitor process doesn't exist anymore
    ppid = os.getppid()

    def _watch():
        while True:
            time.sleep(interval)
            try:
                os.kill(monitor_pid, 0)
                if os.getppid() == ppid:
                    continue
            except OSError:
                pass

            os._exit(1)

    threading.Thread(target=_watch, daemon=True).start()


def _initWorkerLogging(config, output_dir, prefix):
    """ Log into the logs directory of the output directory, let SIGTERM end the worker right away (the
        monitor stops its workers with SIGTERM), and end it if the monitor is gone.
    """

    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    exitWithMonitor()

    config.data_dir = output_dir
    config.log_dir = 'logs'
    LoggingManager().initLogging(config, prefix)


def nightReportWorker(output_dir, night_name, config_path, results):
    """ Worker process target which generates a night report. Exits with 1 if an essential step failed. """

    config = cr.parse(config_path)
    _initWorkerLogging(config, output_dir, 'report_{:s}_'.format(night_name))

    try:
        night_state = generateNightReport(output_dir, night_name, config, results=results)

    except Exception:
        log.error("".join(traceback.format_exception(*sys.exc_info())))
        sys.exit(1)

    sys.exit(1 if (ESSENTIAL_STEPS & set(night_state['failed_steps'])) else 0)


def cleanupWorker(output_dir, config_path):
    """ Worker process target which deletes old data. """

    config = cr.parse(config_path)
    _initWorkerLogging(config, output_dir, 'cleanup_')

    cleanupOldData(output_dir, config)



### Scheduling ###

class ReportLock(object):
    """ Lets only one of several night reporters (e.g. the cameras of the multi-camera monitor) run a report
        at a time, so the reports don't compete with the processing for the CPU.
    """

    def __init__(self):
        self.owner = None


class NightReporter(object):
    """ Schedules the night reports and the cleanup of old data of one camera from the monitor loop. Reports
        and cleanups run in separate processes, one at a time.

    Arguments:
        output_dir: [str] Output directory of the monitor.
        config_path: [str] Path to the config file of the camera.

    Keyword arguments:
        report_mode: [str] One of ConfigReader.MONITOR_REPORT_MODES. The config value is used if None.
        fail_wait_time: [float] Seconds to wait before retrying a failed report. 300 by default.
        camera_id: [str] Camera ID used in the log messages. None by default.
        report_lock: [ReportLock] Lock shared with the reporters of other cameras. None by default.
    """

    def __init__(self, output_dir, config_path, report_mode=None, fail_wait_time=300, camera_id=None,
                 report_lock=None):

        self.output_dir = output_dir
        self.config_path = config_path
        self.config = cr.parse(config_path)
        self.fail_wait_time = fail_wait_time
        self.report_lock = report_lock if report_lock is not None else ReportLock()

        self.log_prefix = "[{:s}] ".format(camera_id) if camera_id else ""

        self.report_mode = report_mode if report_mode is not None else self.config.monitor_report_mode
        self.quiet_s = 60*self.config.monitor_report_quiet_min

        # Running report: (process, night_name, results)
        self.active = None

        # Failed reports: {night_name: {'count': int, 'time': datetime, 'n_files': int}}
        self.failed = {}

        # Nights requested by the trigger file
        self.triggered = set()

        # Nights with unreported files, {night_name: results}, and the report states. The output directory is
        #   scanned again after every report, when new results were added (resultsChanged, at most every
        #   rescan_interval, as the scan takes long with months of results), and periodically
        self.pending = None
        self.states = None
        self.results_changed = False
        self.last_scan = None
        self.scan_interval = 600
        self.rescan_interval = 60

        # Old nights which are not reported because their directory was deleted
        self.ignored_nights = set()

        # Cleanup of old data, at startup, after every report, and periodically
        self.cleanup_proc = None
        self.cleanup_due = True
        self.last_cleanup = None
        self.cleanup_interval = 12*3600

        # Upload the night archives only if enabled for the monitor and uploading is enabled in general. The
        #   upload manager is started right away, so files left in its queue are uploaded after a restart
        self.upload_manager = None
        if self.config.monitor_upload and self.config.upload_enabled:

            from RMS.UploadManager import UploadManager

            upload_config = copy.deepcopy(self.config)
            upload_config.data_dir = output_dir

            self.upload_manager = UploadManager(upload_config)
            self.upload_manager.start()


    def resultsChanged(self):
        """ Tell the reporter that a file was processed, so the output directory is scanned again. """

        self.results_changed = True


    def _lastResultAge(self, results, now):
        """ Seconds since the latest result of the night was finished. """

        last = max(os.path.getmtime(os.path.join(self.output_dir, results_dir, DONE_FLAG_NAME))
                   for results_dir in results)

        last = datetime.datetime.fromtimestamp(last, tz=datetime.timezone.utc).replace(tzinfo=None)

        return (now - last).total_seconds()


    def _gaveUp(self, night_name, results):
        """ True if the report of the night failed twice and no new files arrived since. """

        fail = self.failed.get(night_name)

        return (fail is not None) and (fail['count'] >= 2) and (fail['n_files'] == len(results))


    def isDue(self, night_name, results, idle, now):
        """ Check if the report of the night is due.

        Arguments:
            night_name: [str] Name of the night.
            results: [dict] {results_dir: info} of the night.
            idle: [bool] True if the camera has no files being processed or waiting to be processed.
            now: [datetime] Current time (UTC).

        Return:
            [str] Why the report is due ('trigger', 'partial', 'idle' or 'sunrise'), None if it isn't due.
        """

        # Wait before retrying a failed report, and give up after the second failure until new files arrive
        fail = self.failed.get(night_name)
        if (fail is not None) and (fail['n_files'] == len(results)):
            if (fail['count'] >= 2) or ((now - fail['time']).total_seconds() < self.fail_wait_time):
                return False

        if night_name in self.triggered:
            return 'trigger'

        if self.report_mode not in ('sunrise', 'idle'):
            return None

        # Make the partial report at its time with the data processed until then, once per night. It is not
        #   made for old nights, e.g. when a backlog is processed after a downtime
        partial_time = partialReportTime(self.config, night_name)
        if (partial_time is not None) and (partial_time <= now < partial_time + datetime.timedelta(days=1)):

            reported_at = self.states['nights'].get(night_name, {}).get('reported_at')
            if (reported_at is None) or (datetime.datetime.strptime(reported_at, JSON_TIME_FORMAT)
                                         < partial_time):
                return 'partial'

        # Wait until no new results of the night came in for a while
        if self._lastResultAge(results, now) < self.quiet_s:
            return None

        # In the idle mode, report once all data is processed
        if self.report_mode == 'idle':
            return 'idle' if idle else None

        # In the sunrise mode, report once the night is over. The camera doesn't have to be idle, as it may be
        #   processing the following night already
        if now >= nightBoundsFromName(self.config, night_name)[1]:
            return 'sunrise'

        return None


    def _overTime(self, proc, timeout, name):
        """ Stop the process if it runs longer than the timeout (seconds). Return True if it was stopped. """

        if time.monotonic() - proc.start_time < timeout:
            return False

        log.error(self.log_prefix + "{:s} did not finish in {:.1f} h, stopping it".format(name, timeout/3600))
        stopProcess(proc, self.log_prefix + name)

        return True


    def _startCleanup(self):
        """ Start the cleanup of old data in a separate process. """

        self.cleanup_proc = multiprocessing.Process(target=cleanupWorker,
                                                    args=(self.output_dir, self.config_path))
        self.cleanup_proc.start()
        self.cleanup_proc.start_time = time.monotonic()


    def _startReport(self, night_name, results, reason):
        """ Start the report of the night in a separate process. """

        log.info(self.log_prefix + "Starting the {:s} report of night {:s} ({:d} files)".format(
            reason, night_name, len(results)))

        # Clear the archives of a previous report, so they are not uploaded again if this report fails, and
        #   record why the report was made
        updateReportState(self.output_dir, night_name, upload_files=[], trigger=reason)

        proc = multiprocessing.Process(target=nightReportWorker,
                                       args=(self.output_dir, night_name, self.config_path, results))
        proc.start()
        proc.start_time = time.monotonic()

        self.active = (proc, night_name, results)
        self.report_lock.owner = self
        self.triggered.discard(night_name)


    def _finishReport(self, now):
        """ Record the result of the finished report and hand its archives over for upload. """

        proc, night_name, results = self.active
        proc.join()

        self.active = None
        self.report_lock.owner = None
        self.pending = None

        if proc.exitcode == 0:
            log.info(self.log_prefix + "Report of night {:s} finished".format(night_name))
            self.failed.pop(night_name, None)

        else:
            fail = self.failed.get(night_name, {'count': 0})
            self.failed[night_name] = {'count': fail['count'] + 1, 'time': now, 'n_files': len(results)}
            log.warning(self.log_prefix + "Report of night {:s} failed (exit code {}), see its log".format(
                night_name, proc.exitcode))

        if self.upload_manager is not None:

            upload_files = readReportStates(self.output_dir)['nights'][night_name].get('upload_files', [])
            if upload_files:
                log.info(self.log_prefix + "Adding files to upload list: {}".format(upload_files))
                self.upload_manager.addFiles(upload_files)
                self.upload_manager.delayNextUpload(delay=60*self.config.upload_delay)


    def poll(self, idle, now=None):
        """ Check the running report and cleanup, and start the next due one. Call this from the monitor loop.

        Arguments:
            idle: [bool] True if the camera has no files being processed or waiting to be processed.

        Keyword arguments:
            now: [datetime] Current time (UTC), the current time by default.
        """

        if now is None:
            now = RmsDateTime.utcnow()

        # Check the running report, and clean up after it. A report which takes too long is stopped and
        #   counts as failed
        if self.active is not None:
            if self.active[0].is_alive():
                if not self._overTime(self.active[0], REPORT_TIMEOUT, "Report of night {:s}".format(
                        self.active[1])):
                    return

            self._finishReport(now)
            self.cleanup_due = True

        # Reports and cleanups don't run at the same time
        if self.cleanup_proc is not None:
            if self.cleanup_proc.is_alive():
                if not self._overTime(self.cleanup_proc, CLEANUP_TIMEOUT, "Cleanup"):
                    return

            self.cleanup_proc.join()
            self.cleanup_proc = None

        if self.cleanup_due or ((now - self.last_cleanup).total_seconds() > self.cleanup_interval):
            self.cleanup_due = False
            self.last_cleanup = now
            self._startCleanup()
            return

        if self.report_mode == 'none':
            return

        # The trigger file requests the reports of all nights with new data
        trigger_path = os.path.join(self.output_dir, REPORT_TRIGGER_FILE_NAME)
        trigger = os.path.exists(trigger_path)
        if trigger:
            os.remove(trigger_path)
            self.pending = None

        # Find the nights with unreported files
        scan_age = (now - self.last_scan).total_seconds() if (self.last_scan is not None) else None
        if (self.pending is None) or (scan_age > self.scan_interval) \
                or (self.results_changed and (scan_age > self.rescan_interval)):

            self.results_changed = False

            self.states = readReportStates(self.output_dir)
            self.pending = {}
            for night_name, results in scanNights(self.output_dir, self.config).items():

                if not unreportedResults(self.states, night_name, results):
                    continue

                # Nights whose directory was deleted by the cleanup are not reported again (e.g. a late file
                #   of an old night, or a lost report state), as their archives would have no images. They
                #   can still be reported from the command line
                if not os.path.isdir(nightDirPath(self.output_dir, night_name, self.config)):
                    if night_name not in self.ignored_nights:
                        log.warning(self.log_prefix + "Night {:s} has unreported files, but its directory "
                                    "was deleted by the cleanup, so it is not reported".format(night_name))
                        self.ignored_nights.add(night_name)
                    continue

                self.pending[night_name] = results

            self.last_scan = now

        if trigger:
            log.info(self.log_prefix + "Report trigger file found, reporting {:d} night(s) with new "
                     "data".format(len(self.pending)))
            self.triggered |= set(self.pending)

        # Forget triggered nights which were reported or given up
        self.triggered = set(night_name for night_name in self.triggered if (night_name in self.pending)
                             and not self._gaveUp(night_name, self.pending[night_name]))

        # Report the oldest due night, if no other camera is reporting. Wait for the scan of new results, so
        #   the report includes them
        if (self.report_lock.owner is not None) or self.results_changed:
            return

        for night_name in sorted(self.pending):
            reason = self.isDue(night_name, self.pending[night_name], idle, now)
            if reason:
                self._startReport(night_name, self.pending[night_name], reason)
                break


    def stop(self, timeout=300):
        """ Wait for the running report and cleanup, and stop the upload manager.

        Keyword arguments:
            timeout: [float] Seconds to wait for each process before terminating it. 300 by default.
        """

        if self.cleanup_proc is not None:
            self.cleanup_proc.join(timeout=timeout)

            if self.cleanup_proc.is_alive():
                log.warning(self.log_prefix + "Cleanup did not finish in time, terminating")
                self.cleanup_proc.terminate()

            self.cleanup_proc = None

        if self.active is not None:
            proc, night_name, _ = self.active
            proc.join(timeout=timeout)

            if proc.is_alive():
                log.warning(self.log_prefix + "Report of night {:s} did not finish in time, "
                            "terminating".format(night_name))
                proc.terminate()
                self.active = None
                self.report_lock.owner = None

            # Finish the report which completed, so its archives are handed over for upload
            else:
                self._finishReport(RmsDateTime.utcnow())

        # The upload queue is kept on disk, so an interrupted upload continues after a restart. The upload
        #   manager is stopped here instead of with its stop method, whose final wait is not bounded
        if self.upload_manager is not None:
            self.upload_manager.exit.set()
            self.upload_manager.join(timeout)
            if self.upload_manager.is_alive():
                log.warning(self.log_prefix + "The upload manager did not stop in time, terminating it")
                stopProcess(self.upload_manager, self.log_prefix + "Upload manager")

            self.upload_manager = None



if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Generate the night reports for data processed by "
        "RMS.MonitorProcessFrameInterface. Don't run it while the monitor is reporting the same output "
        "directory, create the trigger file " + REPORT_TRIGGER_FILE_NAME + " instead.")

    arg_parser.add_argument('output_dir', type=str,
        help="Output directory of the monitor.")

    arg_parser.add_argument('--night', '-n', type=str, default=None,
        help="Name of the night to report (a directory in CapturedFiles). All nights with new data by "
             "default.")

    arg_parser.add_argument('--all', '-a', action='store_true',
        help="Report all nights, also the ones without new data.")

    arg_parser.add_argument('--config', '-c', type=str, default=None,
        help="Path to the config file. By default, the config copied to the results of the night is used.")

    arg_parser.add_argument('--no_archive', action='store_true',
        help="Don't archive the night into ArchivedFiles. The archives are never uploaded by this script.")

    cml_args = arg_parser.parse_args()

    output_dir = os.path.abspath(cml_args.output_dir)

    # Don't report next to a running monitor
    output_lock = lockOutputDir(output_dir)
    if output_lock is None:
        print("A monitor (PID {:s}) is using {:s}, create the {:s} file in it to report the nights with new "
              "data instead.".format(lockOwner(output_dir), output_dir, REPORT_TRIGGER_FILE_NAME))
        sys.exit(1)

    # The layout of the output directory only depends on the config, so any config of the camera can be used
    config = cr.parse(os.path.abspath(cml_args.config)) if cml_args.config else cr.Config()
    nights = scanNights(output_dir, config)
    states = readReportStates(output_dir)

    if cml_args.night is not None:
        night_names = [cml_args.night] if cml_args.night in nights else []

    elif cml_args.all:
        night_names = sorted(nights)

    else:
        night_names = sorted(night_name for night_name, results in nights.items()
                             if unreportedResults(states, night_name, results))

    if not night_names:
        print("No nights to report.")
        sys.exit(0)

    exit_code = 0
    for night_name in night_names:

        # Use the given config, otherwise the config copied to the results of the night
        if not cml_args.config:
            first_results = os.path.join(output_dir, sorted(nights[night_name])[0])
            config = cr.parse(sorted(glob.glob(os.path.join(first_results, '*.config')))[0])

        _initWorkerLogging(config, output_dir, 'report_{:s}_'.format(night_name))

        night_state = generateNightReport(output_dir, night_name, config, results=nights[night_name],
                                          archive=(not cml_args.no_archive))

        print("Night {:s}: ok steps: {:s}, failed steps: {:s}".format(night_name,
            ", ".join(night_state['ok_steps']), ", ".join(night_state['failed_steps']) or "none"))

        if ESSENTIAL_STEPS & set(night_state['failed_steps']):
            exit_code = 1

    sys.exit(exit_code)
