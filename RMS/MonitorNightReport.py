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

    python -m RMS.MonitorNightReport <output_dir> [--night NIGHT_NAME] [--config CONFIG_PATH] [--all] [--no_archive]
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
import numpy as np

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

# Directory in the night directory with the ECSV files of the detections
ECSV_DIR_NAME = 'ECSV'

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


def latestPlateparPath(output_dir, config):
    """ Path of the platepar refined at the end of the latest reported night, which is used for the following
        data. It has its own name, so the platepar given to the monitor is never overwritten.
    """
    return os.path.join(output_dir, 'latest_' + config.platepar_name)


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


def calibrationApplied(results_dirs):
    """ Check if the dark and the flat were applied during the processing of the files of the night.

    Arguments:
        results_dirs: [list] Results directories of the files of the night.

    Return:
        (dark_applied, flat_applied): [tuple of bool]
    """

    infos = [readDoneFlag(results_dir) for results_dir in results_dirs]

    dark_applied = any(info.get('dark_applied', False) for info in infos)
    flat_applied = any(info.get('flat_applied', False) for info in infos)

    return dark_applied, flat_applied


def mergeNightResults(night_dir, results_dirs, config):
    """ Merge the CALSTARS, FTPdetectinfo and recalibrated platepars of all files of the night into one file
        each in the night directory, and copy the platepar.

    Arguments:
        night_dir: [str] Night directory.
        results_dirs: [list] Results directories of the files of the night.
        config: [Config] Configuration.

    Return:
        (calstars_name, ftpdetectinfo_name, fps, chunk_frames, ff_detected): [tuple]
            - calstars_name: [str] Name of the merged CALSTARS file, None if there are no stars.
            - ftpdetectinfo_name: [str] Name of the merged FTPdetectinfo file.
            - fps: [float] Frame rate from the CALSTARS files, None if not available.
            - chunk_frames: [int] Number of frames per chunk from the CALSTARS files, None if not available.
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

    return calstars_name, ftpdetectinfo_name, fps, chunk_frames, ff_detected


def plateparResidual(platepar):
    """ Median distance (px) between the image positions of the stars the platepar was fitted on and their
        catalog positions projected with the platepar.

    Arguments:
        platepar: [Platepar instance] Recalibrated platepar (ApplyRecalibrate) with a star list. Its entries
            are [jd, y, x, intensity, ra, dec, mag], as the image coordinates come from CALSTARS (Y X).

    Return:
        [float] Median residual in pixels, inf if there are no stars.
    """

    from RMS.Astrometry.ApplyAstrometry import raDecToXYPP

    star_list = np.array(platepar.star_list if platepar.star_list else [], dtype=np.float64)

    if len(star_list) == 0:
        return np.inf

    jd, y, x, ra, dec = star_list[:, 0], star_list[:, 1], star_list[:, 2], star_list[:, 4], star_list[:, 5]

    x_cat, y_cat = raDecToXYPP(ra, dec, np.mean(jd), platepar)

    return float(np.median(np.hypot(x - x_cat, y - y_cat)))


def selectBestNightPlatepar(night_dir, config, output_dir=None):
    """ Select the most confident platepar of the night from the recalibrated platepars: the successfully
        recalibrated one with the most matched stars, and the lowest residual among those. If output_dir is
        given and monitor_update_platepar is set, it is saved for the following data (latestPlateparPath).

    Arguments:
        night_dir: [str] Night directory with the merged recalibrated platepars.
        config: [Config] Configuration.

    Keyword arguments:
        output_dir: [str] Output directory of the monitor. None by default.

    Return:
        (ff_name, platepar, n_stars, residual): [tuple] The selected platepar, None for all if there is no
            successfully recalibrated platepar.
    """

    from RMS.Formats.Platepar import Platepar

    with open(os.path.join(night_dir, config.platepars_recalibrated_name)) as f:
        recalibrated_dicts = json.load(f)

    candidates = []
    for ff_name, pp_dict in recalibrated_dicts.items():

        if (pp_dict is None) or (not pp_dict.get('auto_recalibrated', False)):
            continue

        platepar = Platepar()
        platepar.loadFromDict(pp_dict, use_flat=config.use_flat)

        n_stars = len(platepar.star_list) if platepar.star_list else 0
        if n_stars == 0:
            continue

        candidates.append((n_stars, -plateparResidual(platepar), ff_name, platepar))

    if not candidates:
        log.info("No successfully recalibrated platepar in the night, keeping the old platepar")
        return None, None, None, None

    # Most matched stars first, then the lowest residual
    n_stars, neg_residual, ff_name, platepar = max(candidates, key=lambda c: (c[0], c[1]))
    residual = -neg_residual

    log.info("Best platepar of the night: {:s}, {:d} stars, median residual {:.3f} px".format(ff_name, n_stars,
        residual))

    if (output_dir is not None) and config.monitor_update_platepar:
        latest_path = latestPlateparPath(output_dir, config)
        platepar.write(latest_path)
        log.info("Best platepar saved for the following data: {:s}".format(latest_path))

    return ff_name, platepar, n_stars, residual


def writeObservationSummary(night_dir, night_name, config, frames_per_file, n_detections,
                            photometry_good=None):
    """ Write the observation summary of the night (observation_summary.txt and .json), as at the end of a
        normal RMS night. The parameters which RMS records at the start of the capture are filled in from the
        night, as there is no live capture (e.g. the camera is not queried).

    Arguments:
        night_dir: [str] Night directory.
        night_name: [str] Name of the night.
        config: [Config] Configuration, with data_dir set to the output directory of the monitor.
        frames_per_file: [int] Number of frames per FF-equivalent image pair.
        n_detections: [int] Number of detections in the night.

    Keyword arguments:
        photometry_good: [bool] Result of the platepar refinement, None if it wasn't run.

    Return:
        [tuple] Paths of the text and the JSON summary.
    """

    import platform

    from RMS.Formats import ObservationSummary as obs

    # Start a new summary for every night
    db_path = os.path.join(config.data_dir, "observation.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    night_start, night_end = nightBoundsFromName(config, night_name)

    conn = obs.getObsDBConn(config)
    obs.addObsParam(conn, "start_time", night_start.replace(tzinfo=datetime.timezone.utc))
    obs.addObsParam(conn, "duration_from_start_of_observation", round((night_end - night_start).total_seconds()))
    obs.addObsParam(conn, "stationID", config.stationID)
    obs.addObsParam(conn, "hardware_version", platform.machine())
    obs.addObsParam(conn, "camera_information", "Processed from recorded data")
    obs.addObsParam(conn, "detections_after_ml", n_detections)

    if photometry_good is not None:
        obs.addObsParam(conn, "photometry_good", str(bool(photometry_good)))

    conn.close()

    return obs.finalizeObservationSummary(config, night_dir, frames_per_file=frames_per_file)


def plotNightCalibrationVariation(night_dir, night_name, config, ff_frames=256):
    """ Plot the variation of the pointing and the photometric offset through the night from the merged
        recalibrated platepars in the night directory.

    Arguments:
        night_dir: [str] Night directory.
        night_name: [str] Name of the night, used as the prefix of the plot names.
        config: [Config] Configuration.

    Keyword arguments:
        ff_frames: [int] Number of frames per chunk. 256 by default.

    Return:
        [bool] True if the plots were saved.
    """

    from RMS.Astrometry.ApplyRecalibrate import plotCalibrationVariation
    from RMS.Formats.Platepar import Platepar

    platepar = Platepar()
    platepar.read(os.path.join(night_dir, config.platepar_name))

    with open(os.path.join(night_dir, config.platepars_recalibrated_name)) as f:
        recalibrated_dicts = json.load(f)

    recalibrated_platepars = {}
    for ff_name, pp_dict in recalibrated_dicts.items():

        if pp_dict is None:
            recalibrated_platepars[ff_name] = None
            continue

        pp = Platepar()
        pp.loadFromDict(pp_dict)
        recalibrated_platepars[ff_name] = pp

    return plotCalibrationVariation(recalibrated_platepars, platepar, config, night_dir, night_name,
                                    ff_frames=ff_frames)


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


def generateNightReport(output_dir, night_name, config, results_dirs=None, archive=True):
    """ Merge the results of a night and generate its report.

    Arguments:
        output_dir: [str] Output directory of the monitor.
        night_name: [str] Name of the night.
        config: [Config] Configuration of the camera.

    Keyword arguments:
        results_dirs: [list] Results directories of the night. Found in the output directory if not given.
        archive: [bool] Archive the night into ARCHIVED_DIR_NAME/<night>, like a normal RMS night. True by
            default. If False, only the thumbnails and the stacks are generated.

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

        calstars_name, ftpdetectinfo_name, fps, chunk_frames, ff_detected = merged

        # Use the frame rate of the data, the chunk times are computed from it
        if fps:
            config.fps = fps

        # The thumbnail stacking is set up for FF files with 256 frames. Scale it to the chunk length, so every
        #   thumbnail covers the same number of frames as with normal FF files
        config.thumb_stack = scaledThumbStack(config.thumb_stack, chunk_frames)

        # Use the same dark and flat settings as the processing, so e.g. the photometry doesn't correct the
        #   vignetting of flat fielded images again. The images were corrected if any file of the night was
        config.use_dark, config.use_flat = calibrationApplied(results_dirs)

        # Use the mask copied to the night directory for the stacks
        config.config_file_path = night_dir

        # Remove the detection stacks of a previous report, their names contain the number of meteors and
        #   would not be overwritten
        for stack_path in glob.glob(os.path.join(night_dir, night_name + '_stack_*_meteors.*')):
            os.remove(stack_path)

        platepar_available = os.path.isfile(os.path.join(night_dir, config.platepar_name))

        # Select the most confident recalibrated platepar of the night and keep it for the following data
        fit_status = None
        if os.path.isfile(os.path.join(night_dir, config.platepars_recalibrated_name)):
            best = _runStep('platepar_selection', lambda: selectBestNightPlatepar(night_dir, config,
                                                                                 output_dir=output_dir))
            if best is not None:
                fit_status = best[0] is not None

        if calstars_name is not None:
            _runStep('calibration_report', lambda: generateCalibrationReport(config, night_dir))

        # Plot the variation of the calibration through the night
        if os.path.isfile(os.path.join(night_dir, config.platepars_recalibrated_name)) \
                and os.path.isfile(os.path.join(night_dir, config.platepar_name)):
            _runStep('calibration_variation', lambda: plotNightCalibrationVariation(night_dir, night_name, config,
                ff_frames=(chunk_frames if chunk_frames else 256)))

        ### Additional night products (off by default) ###

        ftpdetectinfo_path = os.path.join(night_dir, ftpdetectinfo_name)
        product_files = []

        mask = None
        if os.path.isfile(os.path.join(night_dir, config.mask_file)):
            from RMS.Routines.MaskImage import loadMask
            mask = loadMask(os.path.join(night_dir, config.mask_file))

        def _loadPlatepar():
            from RMS.Formats.Platepar import Platepar
            platepar = Platepar()
            platepar.read(os.path.join(night_dir, config.platepar_name), use_flat=config.use_flat)
            return platepar

        if config.monitor_shower_association:
            from Utils.ShowerAssociation import showerAssociation
            _runStep('shower_association', lambda: showerAssociation(config, [ftpdetectinfo_path],
                save_plot=True, plot_activity=True, color_map=config.shower_color_map,
                sporadic_color=config.sporadic_color))

        if config.monitor_fov_kml and platepar_available:
            from Utils.FOVKML import fovKML

            def _fovKML():
                platepar = _loadPlatepar()
                return [fovKML(night_dir, platepar, mask=mask, plot_station=False, area_ht=area_ht)
                        for area_ht in [100000, 70000, 25000]]

            kml_files = _runStep('fov_kml', _fovKML)
            if kml_files:
                product_files += [f for f in kml_files if f]

        if config.monitor_flux and platepar_available:
            from Utils.Flux import prepareFluxFiles
            _runStep('flux', lambda: prepareFluxFiles(config, night_dir, ftpdetectinfo_path, mask=mask,
                                                      platepar=_loadPlatepar()))

            for file_name in sorted(os.listdir(night_dir)):
                if ("flux" in file_name) and (file_name.endswith(".json") or file_name.endswith(".ecsv")):
                    product_files.append(os.path.join(night_dir, file_name))

        if config.monitor_observation_summary:
            n_detections = len(set((entry[0], entry[1]) for entry in
                                   FTPdetectinfo.readFTPdetectinfo(night_dir, ftpdetectinfo_name,
                                                                   ret_input_format=True)[2]))
            summary_files = _runStep('observation_summary', lambda: writeObservationSummary(night_dir,
                night_name, config, chunk_frames if chunk_frames else 256, n_detections,
                photometry_good=fit_status))
            if summary_files:
                product_files += [f for f in summary_files if f and f.endswith('.json')]

        # ECSV files of the detections
        ecsv_dir = os.path.join(night_dir, ECSV_DIR_NAME)
        if os.path.isdir(ecsv_dir):
            product_files += sorted(glob.glob(os.path.join(ecsv_dir, '*.ecsv')))

        ### ###

        # Generate the timelapse first, so it can be archived
        timelapse_name = night_name + "_timelapse.mp4"
        if config.timelapse_generate_captured:
            _runStep('timelapse', lambda: generateTimelapse(night_dir, output_file=timelapse_name))

        if archive:

            archived_dir = os.path.join(output_dir, ARCHIVED_DIR_NAME, night_name)

            # Add the files which are not selected from the night directory by default, like processNight
            extra_names = [config.platepar_name, config.platepars_recalibrated_name, config.mask_file,
                           timelapse_name]
            extra_names += [os.path.basename(f) for f in glob.glob(os.path.join(night_dir, '*.config'))]
            extra_files = [os.path.join(night_dir, name) for name in extra_names
                           if os.path.isfile(os.path.join(night_dir, name))]
            extra_files += [path for path in product_files if os.path.isfile(path)]

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


def nightReportWorker(output_dir, night_name, config_path, results_dirs, archive=True):
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
        state = generateNightReport(output_dir, night_name, config, results_dirs=results_dirs, archive=archive)

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


def cleanupOldData(output_dir, config):
    """ Delete old data from the output directory with the same mechanism as normal RMS
        (DeleteOldObservations): old night directories in CapturedFiles and ArchivedFiles (by the number of
        directories to keep, the quotas, and the free space needed for the next night), old archives and log
        files. The image pairs of reported nights are also deleted after monitor_delete_images_days.

    Arguments:
        output_dir: [str] Output directory of the monitor.
        config: [Config] Configuration of the camera.

    Return:
        [bool] True if there's enough free space for the next night.
    """

    from RMS.DeleteOldObservations import deleteOldObservations

    deleteOldNightImages(output_dir, config.monitor_delete_images_days)

    if not config.monitor_delete_old_data:
        return True

    # Manage the output directory like the RMS data directory
    config.data_dir = output_dir
    config.log_dir = 'logs'

    enough_space = deleteOldObservations(output_dir, CAPTURED_DIR_NAME, ARCHIVED_DIR_NAME, config)

    if not enough_space:
        log.warning("Not enough free space in {:s} for the next night!".format(output_dir))

    return enough_space


def cleanupWorker(output_dir, config_path):
    """ Worker process target which cleans up old data. """

    config = cr.parse(config_path)

    # Log into the output directory
    orig_data_dir, orig_log_dir = config.data_dir, config.log_dir
    config.data_dir = output_dir
    config.log_dir = 'logs'
    LoggingManager().initLogging(config, 'cleanup_')
    config.data_dir, config.log_dir = orig_data_dir, orig_log_dir

    try:
        cleanupOldData(output_dir, config)

    except Exception:
        getLogger("logger").error("".join(traceback.format_exception(*sys.exc_info())))
        sys.exit(1)


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

        # Upload the night archives only if enabled for the monitor and uploading is enabled in general
        self.upload = self.config.monitor_upload and self.config.upload_enabled

        self.quiet_s = 60*self.config.monitor_report_quiet_min

        # Running report: (process, night_name, results_dirs)
        self.active = None

        # Failed reports: {night_name: {'count': int, 'time': datetime, 'n_files': int}}
        self.failed = {}

        # Nights requested by the trigger file
        self.triggered = set()

        self.upload_manager = None

        # Old data cleanup process, which runs at startup, after every report and periodically
        self.cleanup_proc = None
        self.cleanup_due = False
        self.last_cleanup = None
        self.cleanup_interval_h = 12

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


    def _startCleanup(self):

        import multiprocessing

        self.cleanup_proc = multiprocessing.Process(target=cleanupWorker,
            args=(self.output_dir, self.config_path))
        self.cleanup_proc.start()


    def _startReport(self, night_name, results_dirs):

        import multiprocessing

        log.info(self._log("Starting the report of night {:s}".format(night_name)))

        proc = multiprocessing.Process(target=nightReportWorker,
            args=(self.output_dir, night_name, self.config_path, results_dirs))
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

            # Clean up old data after every report
            self.cleanup_due = True

        # Check the running cleanup. Reports and cleanups don't run at the same time
        if self.cleanup_proc is not None:
            if self.cleanup_proc.is_alive():
                return

            self.cleanup_proc.join()
            self.cleanup_proc = None

        # Clean up old data at startup, after every report, and periodically
        if self.cleanup_due or (self.last_cleanup is None) \
                or ((now - self.last_cleanup).total_seconds() > 3600*self.cleanup_interval_h):

            self.cleanup_due = False
            self.last_cleanup = now
            self._startCleanup()
            return

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
        """ Wait for the running report and cleanup, and stop the upload manager. """

        if self.cleanup_proc is not None:
            self.cleanup_proc.join(timeout=timeout)

            if self.cleanup_proc.is_alive():
                log.warning(self._log("Cleanup did not finish in time, terminating"))
                self.cleanup_proc.terminate()

            self.cleanup_proc = None

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

    arg_parser.add_argument('--no_archive', action='store_true',
        help="Don't archive the night into ArchivedFiles, only generate the reports in CapturedFiles. The "
             "archives are never uploaded by this script.")

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
                                    archive=(not cml_args.no_archive))

        print("Night {:s}: ok steps: {:s}, failed steps: {:s}".format(night_name,
            ", ".join(state['ok_steps']), ", ".join(state['failed_steps']) or "none"))

        if state['failed_steps']:
            exit_code = 1

    sys.exit(exit_code)
