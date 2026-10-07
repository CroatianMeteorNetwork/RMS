""" Monitor a directory for new files and process them through star extraction, meteor detection,
    and astrometric recalibration.

    The max pixel and average pixel images of every star extraction chunk are saved as FF-equivalent image
    pairs into a night directory (OUTPUT_DIR/CapturedFiles/<night>/), and a night report (calibration
    report, thumbnails, stacks, timelapse, optional archive and upload) is generated from them, see
    RMS.MonitorNightReport and the [MonitorProcessing] section of the config file.

    Usage:
        python -m RMS.MonitorProcessFrameInterface <file_type> <input_dir> \
            [--output OUTPUT_DIR] [--config CONFIG_PATH] [--platepar PLATEPAR_PATH] \
            [--nproc N] [--chunk_frames N] [--start_time START_TIME] \
            [--report_mode {sunrise,idle,external,none}]

    Example:
        python -m RMS.MonitorProcessFrameInterface vid /path/to/input \
            --output /path/to/output --platepar ~/source/Stations/02F/platepar_cmn2010.cal
"""

from __future__ import print_function, division, absolute_import

import argparse
import copy
import datetime
import gc
import glob
import json
import os
import shutil
import sys
import time
import logging
import traceback
import multiprocessing
import configparser
import signal

import RMS.ConfigReader as cr
from RMS.Formats.FrameInterface import detectInputType, getCacheID
from RMS.Formats.FFfile import validFFName, constructFFName
from RMS.Formats import FFpng
from RMS.Routines import Image
from RMS.MonitorNightReport import exitWithMonitor, latestPlateparPath, lockOutputDir, lockOwner, \
    nightDirPath, nightInfo, NightReporter, readStateFile, ReportLock, writeDoneFlag
from RMS.DetectStarsAndMeteors import (
    detectStarsAndMeteorsFrameInterface,
    saveResultsFrameInterface,
)
from RMS.DeleteOldObservations import availableSpace
from RMS.DetectionTools import binImageCalibration, findMaskPath, loadImageCalibration
from RMS.Astrometry.ApplyRecalibrate import applyRecalibrate
from RMS.Logger import LoggingManager, getLogger
from RMS.Misc import RmsDateTime


# Get the logger from the main module
log = getLogger("logger")


# File type to extension mapping
FILE_TYPE_MAP = {
    'vid': ['.vid'],
    'ff':  None,  # Special handling via validFFName()
    'mkv': ['.mkv'],
    'mp4': ['.mp4'],
    'avi': ['.avi'],
    'mov': ['.mov'],
    'wmv': ['.wmv'],
    'fitsdirs': None,
}

# Directories of the monitor output which are not searched for input files
MONITOR_OUTPUT_DIRS = {'CapturedFiles', 'ArchivedFiles', 'logs'}

# Exit code of a worker process which skipped a file because it begins before the start time, or because the
#   processing of its night was cut off (monitor_night_cutoff_hours)
SKIP_EXIT_CODE = 3

# Default time limit (seconds) for processing one file, after which the worker is stopped and the file retried
WORKER_TIMEOUT = 3600

# Seconds the night report, cleanup and upload are given to finish when the monitor is stopped, before they
#   are terminated (an unfinished report is made again after a restart)
SHUTDOWN_TIMEOUT = 30

# Record of the input files which failed, kept in the output directory: {unique_id: {'count': int,
#   'last_fail_time': float}}. A file is retried once after fail_wait_time, and after MAX_FAILURES failures it
#   is skipped, also after a restart, unless the monitor is started with --retry_failed
FAILED_FILES_NAME = '.failed_files.json'
MAX_FAILURES = 2

# Accepted formats of the start time given on the command line or in the multicam INI file
START_TIME_FORMATS = ["%Y%m%d_%H%M%S", "%Y%m%d-%H%M%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                      "%Y%m%d"]


def parseStartTime(start_time_str):
    """ Parse the start time string into a datetime object.

    Arguments:
        start_time_str: [str] Start time in UTC, in one of the START_TIME_FORMATS formats.

    Return:
        [datetime] Parsed start time (naive, UTC).
    """

    start_time_str = start_time_str.strip()

    for fmt in START_TIME_FORMATS:
        try:
            start_time = datetime.datetime.strptime(start_time_str, fmt)
        except ValueError:
            continue

        # strptime accepts fields which are not zero-padded (e.g. 20260101_1020 is read as 10:02:00),
        # so require an exact round trip
        if start_time.strftime(fmt) == start_time_str:
            return start_time

    raise ValueError("Could not parse the start time '{}'. Accepted formats: {}".format(
        start_time_str, ", ".join(START_TIME_FORMATS)))


def resolveCameraStartTime(cp, section, cli_start_time=None):
    """ Determine the start time for one camera in the multicam mode.

    Precedence: the command line start time, then start_time in the camera section, then start_time in
    the [Global] section.

    Arguments:
        cp: [ConfigParser] Parsed multicam INI file.
        section: [str] Name of the camera section.

    Keyword arguments:
        cli_start_time: [datetime] Start time given on the command line. None by default.

    Return:
        [datetime] Start time for the camera, None if not set anywhere.
    """

    if cli_start_time is not None:
        return cli_start_time

    for sec in [section, 'Global']:
        if cp.has_option(sec, 'start_time'):
            try:
                return parseStartTime(cp.get(sec, 'start_time'))
            except ValueError as e:
                raise ValueError("[{}] start_time: {}".format(sec, e))

    return None


def readBeginningDatetime(file_path, config, chunk_frames):
    """ Read the time of the beginning of the recording using the frame interface, without loading the
        whole file.

    Arguments:
        file_path: [str] Path to the input file or directory.
        config: [Config instance] Loaded config object.
        chunk_frames: [int] Number of frames per chunk.

    Return:
        [datetime] Time of the beginning of the recording, None if the file could not be opened.
    """

    img_handle = detectInputType(file_path, config, detection=True, preload_video=False,
        chunk_frames=chunk_frames)

    if img_handle is None:
        return None

    beginning_datetime = img_handle.beginning_datetime

    # Release the open file handles
    if getattr(img_handle, 'cap', None) is not None:
        img_handle.cap.release()

    if getattr(img_handle, 'vid_file', None) is not None:
        img_handle.vid_file.close()

    return beginning_datetime


def resultsDirPath(output_dir, beginning_datetime, file_base):
    """ Path of the results directory of an input file, output_dir/YYYY/YYYYMM/YYYYMMDD/<file_base>/.

    Arguments:
        output_dir: [str] Output directory of the monitor.
        beginning_datetime: [datetime] Beginning of the recording (UTC).
        file_base: [str] Unique name of the input file.

    Return:
        [str] Path of the results directory.
    """

    dt = beginning_datetime

    return os.path.join(output_dir, "{:04d}".format(dt.year), "{:04d}{:02d}".format(dt.year, dt.month),
                        "{:04d}{:02d}{:02d}".format(dt.year, dt.month, dt.day), file_base)


def walkInput(top_dir):
    """ Walk the directory tree like os.walk, without descending into the night, archive and log directories
        of the monitor, which are in the input directory if it is also the output directory (the default).

    Arguments:
        top_dir: [str] Directory to walk.

    Return:
        Yields (root, dirs, files) like os.walk.
    """

    for root, dirs, files in os.walk(top_dir):
        dirs[:] = [d for d in dirs if d not in MONITOR_OUTPUT_DIRS]
        yield root, dirs, files


def matchesFileType(file_name, file_type):
    """ Check if the given file matches the specified file type.

    Arguments:
        file_name: [str] File name to check.
        file_type: [str] File type identifier (e.g. 'vid', 'ff', 'mkv').

    Return:
        [bool] True if the file matches the type.
    """

    file_type = file_type.lower()

    # Special handling for FF files
    if file_type == 'ff':
        return validFFName(file_name)

    # Special handling for fitsdirs (matches everything, checked in monitorDirectory)
    if file_type == 'fitsdirs':
        return True

    # Check by extension
    if file_type in FILE_TYPE_MAP:
        extensions = FILE_TYPE_MAP[file_type]
        if extensions is not None:
            return any(file_name.lower().endswith(ext) for ext in extensions)

    # If the type is not in the map, try matching directly as an extension
    return file_name.lower().endswith('.' + file_type)




class ChunkImageSaver(object):
    """ Saves the max pixel and average pixel images of every star extraction chunk as an FF-equivalent image
        pair, corrected with the dark and the flat, and keeps the list of saved chunks. Used as the chunk
        callback of the star extraction.

    Arguments:
        night_dir: [str] Directory where the image pairs are saved.
        config: [Config] Configuration.
        source_file: [str] Name of the input file.

    Keyword arguments:
        dark: [ndarray] Dark frame at the full image size. None by default.
        flat_struct: [Flat struct] Flat field at the full image size. None by default.
    """

    def __init__(self, night_dir, config, source_file, dark=None, flat_struct=None):

        self.night_dir = night_dir
        self.config = config
        self.source_file = source_file

        # The chunks are binned for detection, so bin copies of the dark and the flat to match them
        self.dark, self.flat_struct = dark, flat_struct
        if config.detection_binning_factor > 1:
            _, self.dark, self.flat_struct = binImageCalibration(config, None, dark,
                                                                 copy.deepcopy(flat_struct))

        # List of (first_frame, nframes, ff_name) of the saved chunks
        self.chunk_images = []


    def savePair(self, img_handle, ff, first_frame):
        """ Save the images of a chunk.

        Arguments:
            img_handle: [FrameInterface instance] Image handle.
            ff: [FFMimickInterface instance] Uncalibrated chunk.
            first_frame: [int] First frame of the chunk.

        Return:
            [str] FF name of the saved pair.
        """

        begin_time = img_handle.currentFrameTime(frame_no=first_frame, dt_obj=True)

        ff_name = FFpng.pairNames(constructFFName(self.config.stationID, begin_time, frame=first_frame,
                                                  ext=None))[0]

        # Apply the dark and the flat, like for the star extraction (the cached chunk is not modified)
        maxpixel, avepixel = ff.maxpixel, ff.avepixel

        if self.dark is not None:
            maxpixel = Image.applyDark(maxpixel, self.dark)
            avepixel = Image.applyDark(avepixel, self.dark)

        if self.flat_struct is not None:
            maxpixel = Image.applyFlat(maxpixel, self.flat_struct)
            avepixel = Image.applyFlat(avepixel, self.flat_struct)

        meta = {
            'nframes': ff.nframes,
            'fps': img_handle.fps,
            'first_frame': first_frame,
            'binning': self.config.detection_binning_factor,
            'station': self.config.stationID,
            'begin_utc': begin_time,
            'source_file': self.source_file,
            'dark_applied': self.dark is not None,
            'flat_applied': self.flat_struct is not None,
        }

        FFpng.writePair(self.night_dir, ff_name, maxpixel, avepixel, meta=meta)

        self.chunk_images.append((first_frame, ff.nframes, ff_name))

        return ff_name


    def __call__(self, img_handle, ff):
        """ Chunk callback of the star extraction, called with the uncalibrated chunk. """

        return self.savePair(img_handle, ff, img_handle.current_frame_chunk*img_handle.chunk_frames)


    def saveTrailing(self, img_handle):
        """ Save the frames after the last full chunk, which are not used for star extraction, if there are at
            least half a chunk of them. Meteors in shorter remainders are assigned to the last full chunk.

        Arguments:
            img_handle: [FrameInterface instance] Image handle.

        Return:
            [str] FF name of the saved pair, None if nothing was saved.
        """

        first_frame = img_handle.total_fr_chunks*img_handle.chunk_frames
        n_frames = img_handle.total_frames - first_frame

        if n_frames < img_handle.chunk_frames//2:
            return None

        # Remove the chunk from the cache, as meteor detection may have cached a calibrated chunk with the
        #   same frames
        img_handle.cache.pop(getCacheID(first_frame, n_frames), None)

        ff = img_handle.loadChunk(first_frame=first_frame, read_nframes=n_frames)
        if not ff.successful:
            return None

        return self.savePair(img_handle, ff, first_frame)



def processFile(file_path, config_path, platepar_path, output_dir, chunk_frames,
                flat_path=None, dark_path=None, unique_id=None, start_time=None):
    """ Process a single file through the detection and recalibration pipeline.

    Arguments:
        file_path: [str] Path to the input file.
        config_path: [str] Path to the config file.
        platepar_path: [str] Path to the platepar file.
        output_dir: [str] Path to the output directory.
        chunk_frames: [int] Number of frames per chunk for star extraction.

    Keyword arguments:
        flat_path: [str] Path to a flat field file. None by default.
        dark_path: [str] Path to a dark frame file. None by default.
        unique_id: [str] Safely flattened string to uniquely identify output directories. None by default.
        start_time: [datetime] If given, files whose recording begins before this time (UTC) are skipped
            and the process exits with SKIP_EXIT_CODE. None by default. Files of nights past the cutoff
            (monitor_night_cutoff_hours) are skipped the same way, and marked as done.

    Return:
        [bool] True if processing succeeded, False otherwise.
    """

    file_name = os.path.basename(file_path)
    # If a unique_id is provided, use it for the output folder structure. Otherwise use file_base.
    file_base = unique_id if unique_id else os.path.splitext(file_name)[0]

    # Load the config file
    config = cr.parse(config_path)

    # The dark and the flat are applied only if they are explicitly given to the monitor, regardless of the
    #   use_dark and use_flat options in the config file. The config flags are set accordingly, so the whole
    #   processing chain (star extraction, detection, saved images, recalibration and photometry) agrees on
    #   whether the data is corrected, and nothing is corrected twice
    config.use_dark = dark_path is not None
    if dark_path is not None:
        config.dark_file = os.path.abspath(dark_path)

    config.use_flat = flat_path is not None
    if flat_path is not None:
        config.flat_file = os.path.abspath(flat_path)

    # Skip files which begin before the start time or whose night is past its cutoff, before loading the
    #   whole file
    if (start_time is not None) or (config.monitor_night_cutoff_hours > 0):

        # Use a copy of the config, as opening some input types modifies it (e.g. the image size)
        beginning_datetime = readBeginningDatetime(file_path, copy.deepcopy(config), chunk_frames)

        if (start_time is not None) and (beginning_datetime is not None) \
                and (beginning_datetime < start_time):
            print("Skipping {}: begins at {} UTC, before the start time {} UTC".format(
                file_name, beginning_datetime, start_time))
            sys.exit(SKIP_EXIT_CODE)

        if (config.monitor_night_cutoff_hours > 0) and (beginning_datetime is not None):

            night_name, _, night_end = nightInfo(config, beginning_datetime)
            cutoff = night_end + datetime.timedelta(hours=config.monitor_night_cutoff_hours)

            # Mark the file as done, so it is not processed again after a restart
            if RmsDateTime.utcnow() > cutoff:
                print("Skipping {}: the processing of night {} was cut off at {} UTC".format(file_name,
                    night_name, cutoff))
                results_dir = resultsDirPath(output_dir, beginning_datetime, file_base)
                os.makedirs(results_dir, exist_ok=True)
                writeDoneFlag(results_dir, {'night': night_name, 'skipped': True})
                sys.exit(SKIP_EXIT_CODE)

    # Use the module logger until the per-file logger is initialized, so errors before that are logged
    proc_log = log

    try:

        # Open the file as an image handle first to get the timestamp
        img_handle = detectInputType(
            file_path, config, detection=True, preload_video=True, chunk_frames=chunk_frames
        )

        if img_handle is None:
            print("ERROR: Could not open file: {}".format(file_path))
            return False

        # Create the results directory, sorted by the date of the first frame
        results_dir = resultsDirPath(output_dir, img_handle.beginning_datetime, file_base)
        os.makedirs(results_dir, exist_ok=True)

        # Initialize the logger for this process
        orig_data_dir = config.data_dir
        orig_log_dir = config.log_dir
        
        config.data_dir = output_dir
        config.log_dir = 'logs'

        log_manager = LoggingManager()
        log_prefix = 'monitor_{}_'.format(unique_id) if unique_id else 'monitor_{}_'.format(file_base)
        log_manager.initLogging(config, log_prefix)
        proc_log = getLogger("logger")

        config.data_dir = orig_data_dir
        config.log_dir = orig_log_dir

        proc_log.info("Processing file: {}".format(file_path))

        # Use the best platepar of the latest reported night, unless the given platepar is newer (e.g. it was
        #   fitted again manually)
        latest_platepar_path = latestPlateparPath(output_dir, config)
        if config.monitor_update_platepar and os.path.isfile(latest_platepar_path) \
                and (os.path.getmtime(latest_platepar_path) > os.path.getmtime(platepar_path)):
            platepar_path = latest_platepar_path
            proc_log.info("Using the best platepar of a previous night: {}".format(platepar_path))

        # Copy the platepar into the results directory so ApplyRecalibrate can find it
        results_platepar_path = os.path.join(results_dir, config.platepar_name)
        shutil.copy2(platepar_path, results_platepar_path)

        # Copy the config file into the results directory
        results_config_path = os.path.join(results_dir, os.path.basename(config_path))
        if not os.path.exists(results_config_path):
            shutil.copy2(config_path, results_config_path)

        # Load calibration files (mask, dark, flat) from the input directory
        mask, dark, flat_struct = loadImageCalibration(
            img_handle.dir_path, config, dtype=img_handle.ff.dtype, byteswap=img_handle.byteswap
        )

        # The mask which was found, it is copied to the night directory for the night report
        mask_path = findMaskPath(img_handle.dir_path, config)

        # The given dark and flat have to be applied
        if config.use_dark and (dark is None):
            raise IOError("The dark could not be loaded: {}".format(config.dark_file))

        if config.use_flat and (flat_struct is None):
            raise IOError("The flat could not be loaded: {}".format(config.flat_file))

        # Determine the night the file belongs to. The chunk images are saved to the night directory
        night_name, _, _ = nightInfo(config, img_handle.beginning_datetime)
        night_dir = nightDirPath(output_dir, night_name, config)
        os.makedirs(night_dir, exist_ok=True)

        # FF inputs are not saved again, they already are FF files
        image_saver = None
        if config.monitor_save_images and (img_handle.input_type != 'ff'):
            image_saver = ChunkImageSaver(night_dir, config, file_name, dark=dark, flat_struct=flat_struct)

        # Run star extraction and meteor detection
        star_list, meteor_list = detectStarsAndMeteorsFrameInterface(
            img_handle, config, flat_struct=flat_struct, dark=dark, mask=mask,
            chunk_frames=chunk_frames, chunk_callback=image_saver
        )

        # Save the images of the frames after the last full chunk
        if image_saver is not None:
            image_saver.saveTrailing(img_handle)
            proc_log.info("Saved {:d} chunk image pairs to: {}".format(len(image_saver.chunk_images),
                image_saver.night_dir))

        # Save results (CALSTARS + FTPdetectinfo) to the results directory. Meteors are assigned to the saved
        #   chunk images
        saveResultsFrameInterface(
            star_list, meteor_list, img_handle, config,
            chunk_frames=chunk_frames, output_dir=results_dir,
            chunk_images=(image_saver.chunk_images if image_saver is not None else None)
        )

        proc_log.info("Detection results saved to: {}".format(results_dir))

        # Release the video handle if applicable
        if hasattr(img_handle, 'cap') and img_handle.cap is not None:
            img_handle.cap.release()

        del img_handle
        gc.collect()

        # Find the FTPdetectinfo file in the results directory
        ftpdetect_files = glob.glob(os.path.join(results_dir, 'FTPdetectinfo_*.txt'))

        if ftpdetect_files:
            ftpdetectinfo_path = ftpdetect_files[0]

            proc_log.info("Running ApplyRecalibrate on: {}".format(ftpdetectinfo_path))

            # Run recalibration with load_all=True. The ECSV files of the detections are saved to the results
            #   directory and collected into the night directory by the night report
            applyRecalibrate(
                ftpdetectinfo_path, config,
                # The calibration variation plots are made for the whole night in the night report
                generate_plot=False,
                load_all=True,
                generate_ufoorbit=False,
                ecsv_out=config.monitor_save_ecsv,
            )

            proc_log.info("Recalibration complete for: {}".format(file_name))

        else:
            proc_log.info("No FTPdetectinfo file found, skipping recalibration for: {}".format(
                file_name))

        # Create the done.flag file, with the info needed for the night report
        writeDoneFlag(results_dir, {
            'night': night_name,
            'mask_path': mask_path,
            'dark_applied': dark is not None,
            'flat_applied': flat_struct is not None,
        })

        proc_log.info("Done processing: {} -> {}".format(file_name, results_dir))
        return True

    except Exception as e:
        proc_log.error(traceback.format_exc())
        proc_log.error("Error processing {}: {}".format(file_name, str(e)))
        return False


def processFileWorker(*args, **kwargs):
    """ Worker process target which runs processFile and turns its result into the process exit code, so
        the monitor can tell failed files apart and retry them. Takes the same arguments as processFile.

    Exit codes:
        0 - processing succeeded
        SKIP_EXIT_CODE - the file begins before the start time, or its night is past the cutoff
        1 - processing failed
    """

    # The monitor stops the worker with SIGTERM, which ends it right away, and the worker ends if the monitor is
    #   gone
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    exitWithMonitor()

    try:
        success = processFile(*args, **kwargs)

    except SystemExit as e:

        if e.code == SKIP_EXIT_CODE:
            raise

        # Some input types call sys.exit() when they can't open the file (e.g. no time in the file name),
        # which would otherwise end the process with exit code 0
        print("ERROR: Processing exited early with code {}".format(e.code))
        success = False

    sys.exit(0 if success else 1)


def loadFailedFiles(output_dir, retry_failed=False):
    """ Load the record of the failed files from the output directory.

    Arguments:
        output_dir: [str] Output directory of the monitor.

    Keyword arguments:
        retry_failed: [bool] Clear the record, so all failed files are processed again. False by default.

    Return:
        (failed_files, given_up): [tuple]
            - failed_files: [dict] {unique_id: {'count': int, 'last_fail_time': float}}
            - given_up: [set] Unique IDs of the files which failed MAX_FAILURES times.
    """

    failed_files = {}
    if retry_failed:
        saveFailedFiles(output_dir, failed_files)

    # A damaged record makes the failed files be retried
    else:
        failed_files = readStateFile(os.path.join(output_dir, FAILED_FILES_NAME), failed_files)

    given_up = set(uid for uid, fail in failed_files.items() if fail['count'] >= MAX_FAILURES)

    return failed_files, given_up


def saveFailedFiles(output_dir, failed_files):
    """ Save the record of the failed files into the output directory (see loadFailedFiles). """

    failed_files_path = os.path.join(output_dir, FAILED_FILES_NAME)

    with open(failed_files_path + '.tmp', 'w') as f:
        json.dump(failed_files, f, indent=4, sort_keys=True)

    os.replace(failed_files_path + '.tmp', failed_files_path)


def recordFailure(output_dir, failed_files, unique_id, exitcode, fail_wait_time, log_prefix=''):
    """ Record a failure of a file, and save the record.

    Arguments:
        output_dir: [str] Output directory of the monitor.
        failed_files: [dict] Record of the failed files, see loadFailedFiles. It is updated.
        unique_id: [str] Unique ID of the file.
        exitcode: [int] Exit code of the worker.
        fail_wait_time: [float] Seconds before the file is retried.

    Keyword arguments:
        log_prefix: [str] Prefix of the log messages, e.g. the camera ID. Empty by default.

    Return:
        [bool] True if the file failed MAX_FAILURES times and is not retried anymore.
    """

    fail = failed_files.setdefault(unique_id, {'count': 0})
    fail['count'] += 1
    fail['last_fail_time'] = time.time()

    saveFailedFiles(output_dir, failed_files)

    if fail['count'] >= MAX_FAILURES:
        log.error("{}Processing failed for: {} (exit code {:d}) {:d} times, giving up. It is not processed "
                  "again unless the monitor is started with --retry_failed.".format(log_prefix, unique_id,
                                                                                    exitcode, fail['count']))
        return True

    log.warning("{}Processing failed for: {} (exit code {:d}), will retry in {:.1f} mins...".format(
        log_prefix, unique_id, exitcode, fail_wait_time/60.0))

    return False


class FreeSpaceGuard(object):
    """ Pauses the processing while the disk of the output directory is almost full (less than extra_space_gb
        free), so the files don't fail because their results can't be written, which would give them up. The
        cleanup of old data is requested to free space, and the processing continues once there is space.

    Arguments:
        output_dir: [str] Output directory of the camera.
        reporter: [NightReporter] Night reporter of the camera, with its config, which runs the cleanup.

    Keyword arguments:
        log_prefix: [str] Prefix of the log messages, e.g. the camera ID. Empty by default.
    """

    # Seconds between the cleanup requests while the processing is paused
    CLEANUP_REQUEST_INTERVAL = 1800

    def __init__(self, output_dir, reporter, log_prefix=''):

        self.output_dir = output_dir
        self.reporter = reporter
        self.log_prefix = log_prefix

        self.paused = False
        self.last_cleanup_request = None


    def ok(self):
        """ Check the free space. Return True if new files can be processed. """

        free_gb = availableSpace(self.output_dir)/1024**3

        if free_gb >= self.reporter.config.extra_space_gb:

            if self.paused:
                log.info("{}{:.1f} GB free in {} again, resuming the processing".format(self.log_prefix,
                    free_gb, self.output_dir))
            self.paused = False

            return True

        if not self.paused:
            log.error("{}Only {:.1f} GB free in {} (extra_space_gb: {:.1f}), pausing the processing until "
                      "there is space".format(self.log_prefix, free_gb, self.output_dir,
                                              self.reporter.config.extra_space_gb))
        self.paused = True

        # Ask for a cleanup of old data
        if (self.last_cleanup_request is None) \
                or (time.time() - self.last_cleanup_request > self.CLEANUP_REQUEST_INTERVAL):
            self.reporter.cleanup_due = True
            self.last_cleanup_request = time.time()

        return False


def stopOnSigterm(signum, frame):
    """ Signal handler which stops the monitor on SIGTERM (e.g. from systemd) like Ctrl+C does, so the workers
        and the night report are stopped cleanly instead of being left running.
    """

    raise KeyboardInterrupt


def stopStuckWorker(proc, unique_id, worker_timeout):
    """ Stop a worker process which runs longer than the time limit, e.g. one stuck in the video decoder. The
        stopped worker has a negative exit code, so its file is handled as failed and retried.

    Arguments:
        proc: [multiprocessing.Process] Worker process, with the time it was started in proc.start_time.
        unique_id: [str] ID of the processed file.
        worker_timeout: [float] Time limit in seconds, no limit if 0.
    """

    if (worker_timeout <= 0) or (time.time() - proc.start_time < worker_timeout) or (not proc.is_alive()):
        return

    log.error("Processing {} did not finish in {:.1f} min, stopping the worker...".format(unique_id,
        worker_timeout/60))

    stopWorker(proc)


def stopWorker(proc):
    """ Stop a worker process, killing it if it doesn't end after SIGTERM. """

    proc.terminate()
    proc.join(10)
    if proc.is_alive():
        proc.kill()
        proc.join()


def monitorDirectory(input_dir, file_type, config_path, platepar_path, output_dir, nproc=2,
                     chunk_frames=128, poll_interval=2, force=False, recursive=False, flat_path=None,
                     dark_path=None, fail_wait_time=300, start_time=None, report_mode=None,
                     worker_timeout=WORKER_TIMEOUT, retry_failed=False):
    """ Monitor a directory for new files of the given type and process them.

    Arguments:
        input_dir: [str] Directory to monitor.
        file_type: [str] File type to watch for (e.g. 'vid', 'ff', 'mkv').
        config_path: [str] Path to the config file.
        platepar_path: [str] Path to the platepar file.
        output_dir: [str] Directory for output results.

    Keyword arguments:
        nproc: [int] Number of parallel worker processes. Default is 2.
        chunk_frames: [int] Frames per chunk for star extraction. Default is 128.
        poll_interval: [float] Seconds between directory scans. Default is 2.
        force: [bool] If True, re-process files even if done.flag exists. Default is False.
        flat_path: [str] Path to a flat field file. None by default.
        dark_path: [str] Path to a dark frame file. None by default.
        fail_wait_time: [float] Seconds to wait before retrying a failed file. Default is 300.
        start_time: [datetime] Only process files whose recording begins at or after this time (UTC).
            None by default, in which case all files are processed.
        report_mode: [str] When to generate the night reports ('sunrise', 'idle', 'external', 'none'). The
            config value is used if None.
        worker_timeout: [float] Seconds after which a worker which is still processing a file is stopped
            and the file is retried. 0 disables the limit. WORKER_TIMEOUT by default.
        retry_failed: [bool] Process the files which failed in previous runs again. False by default.
    """

    log.info("Monitoring directory: {}".format(input_dir))
    log.info("File type: {}".format(file_type))
    log.info("Output directory: {}".format(output_dir))
    log.info("Parallel processes: {:d}".format(nproc))
    if start_time is not None:
        log.info("Only processing files beginning at or after: {} UTC".format(start_time))

    # Only one monitor can work on the output directory
    output_lock = lockOutputDir(output_dir)
    if output_lock is None:
        log.error("Another monitor (PID {:s}) is already working on {:s}, exiting.".format(
            lockOwner(output_dir), output_dir))
        sys.exit(1)

    # Track files that have been processed or are being processed
    processed_files = set()

    # Failed files, also from previous runs. Files which failed too often are not processed again
    failed_files, given_up = loadFailedFiles(output_dir, retry_failed=(retry_failed or force))
    if given_up:
        log.info("Skipping {:d} file(s) which failed in previous runs (use --retry_failed to process them "
                 "again): {}".format(len(given_up), ", ".join(sorted(given_up))))
        processed_files |= given_up

    # Scan the output directory for previously completed results (done.flag)
    if not force:
        for root, dirs, files in walkInput(output_dir):
            if 'done.flag' in files:
                # The parent dir name is the file base name
                completed_base = os.path.basename(root)
                processed_files.add(completed_base)

        if processed_files:
            log.info("Found {:d} previously processed file(s), skipping.".format(
                len(processed_files)))

    # Active worker processes: {unique_id: Process}
    active_workers = {}

    stability_tracker = {}
    stable_wait_time = 5
    stable_max_age = 30

    # Flag to avoid repeating the "waiting for data" message
    waiting_for_data = False

    # Schedules the night reports
    reporter = NightReporter(output_dir, config_path, report_mode=report_mode, fail_wait_time=fail_wait_time)
    log.info("Night report mode: {:s}".format(reporter.report_mode))

    # Pauses the processing while the output disk is full
    space_guard = FreeSpaceGuard(output_dir, reporter)

    try:
        while True:

            # Clean up finished workers
            finished = []
            for uid, proc in active_workers.items():
                stopStuckWorker(proc, uid, worker_timeout)
                if not proc.is_alive():
                    proc.join()
                    if proc.exitcode == 0:
                        log.info("Successfully processed: {}".format(uid))
                        processed_files.add(uid)
                        reporter.resultsChanged()

                        # Forget an earlier failure of the file
                        if failed_files.pop(uid, None) is not None:
                            saveFailedFiles(output_dir, failed_files)

                    elif proc.exitcode == SKIP_EXIT_CODE:
                        log.info("Skipped: {} (begins before the start time or after the cutoff of its "
                                 "night)".format(uid))
                        processed_files.add(uid)
                    elif recordFailure(output_dir, failed_files, uid, proc.exitcode, fail_wait_time):
                        processed_files.add(uid)

                    finished.append(uid)

            for uid in finished:
                del active_workers[uid]

            # 1. Get ALL paths and their metadata first
            candidate_paths = []
            try:
                if recursive:
                    for root, dirs, files in walkInput(input_dir):
                        if file_type == 'fitsdirs':
                            for d in dirs:
                                candidate_paths.append(os.path.join(root, d))
                        else:
                            for f in files:
                                candidate_paths.append(os.path.join(root, f))
                else:
                    candidate_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
            except OSError:
                log.error("Cannot read directory: {}".format(input_dir))

            # Forget the files which disappeared while waiting for them to be written
            stability_tracker = {path: trk for path, trk in stability_tracker.items()
                                 if path in candidate_paths}

            # 2. Identify which paths are "Stable"
            stable_files = []
            for file_path in candidate_paths:
                file_name = os.path.basename(file_path)
                file_rel_path = os.path.relpath(file_path, input_dir)

                # Check if the file matches the requested type
                if not matchesFileType(file_name, file_type):
                    continue

                if file_type == 'fitsdirs':
                    if not os.path.isdir(file_path):
                        continue
                    
                    # Ensure it contains .fits files
                    try:
                        fits_files = [f for f in os.listdir(file_path) if f.lower().endswith('.fits')]
                        if not fits_files:
                            continue
                    except OSError:
                        continue

                else:
                    if os.path.isdir(file_path):
                        continue

                # Generate a unique ID based on the path name.
                unique_id = os.path.splitext(file_name)[0]

                # Skip already processed or currently processing files
                is_processed = unique_id in processed_files or any(pb.endswith(unique_id) for pb in processed_files)
                if is_processed or unique_id in active_workers:
                    continue

                # Skip files that failed recently (wait fail_wait_time)
                if unique_id in failed_files:
                    if (time.time() - failed_files[unique_id]['last_fail_time']) < fail_wait_time:
                        continue

                # Non-blocking stability check
                try:
                    if file_type == 'fitsdirs':
                        # For directories, check the mtime and total size of all .fits files
                        full_fits_paths = [os.path.join(file_path, f) for f in fits_files]
                        curr_mtime = max(os.path.getmtime(f) for f in full_fits_paths)
                        curr_size = sum(os.path.getsize(f) for f in full_fits_paths)
                    else:
                        curr_mtime = os.path.getmtime(file_path)
                        curr_size = os.path.getsize(file_path)
                except OSError:
                    continue  # File disappeared or momentarily locked, try next pass

                # If file is older than max age, consider it stable immediately
                if (time.time() - curr_mtime) > stable_max_age:
                    stable = True
                else:
                    if file_path not in stability_tracker:
                        log.info("New {} detected: {} - waiting for write completion...".format(
                            "directory" if file_type == 'fitsdirs' else "file", file_rel_path))
                        stability_tracker[file_path] = {'mtime': curr_mtime, 'size': curr_size, 'since': time.time()}
                        stable = False
                    else:
                        trk = stability_tracker[file_path]
                        if trk['mtime'] != curr_mtime or trk['size'] != curr_size:
                            stability_tracker[file_path] = {'mtime': curr_mtime, 'size': curr_size, 'since': time.time()}
                            stable = False
                        else:
                            if (time.time() - trk['since']) >= stable_wait_time:
                                stable = True
                                del stability_tracker[file_path]
                            else:
                                stable = False

                if stable:
                    stable_files.append((file_path, curr_mtime, unique_id, file_rel_path))

            # 3. SORT STABLE FILES BY AGE (Oldest First)
            # If a file has failed before, use its last fail time for sorting to put it at the end of the queue
            def getSortTime(file_info):
                mtime = file_info[1]
                uid = file_info[2]
                if uid in failed_files:
                    return max(mtime, failed_files[uid]['last_fail_time'])
                return mtime

            stable_files.sort(key=getSortTime)

            # 4. Start processes for the oldest stable files until nproc is full, unless the output disk is
            #   full
            new_files_queued = False
            space_ok = space_guard.ok() if stable_files else True
            for file_path, mtime, unique_id, file_rel_path in stable_files:
                if (len(active_workers) >= nproc) or (not space_ok):
                    break

                if file_path in stability_tracker:
                    del stability_tracker[file_path]

                log.info("Putting file {} on the processing queue...".format(file_rel_path))

                proc = multiprocessing.Process(
                    target=processFileWorker,
                    args=(file_path, config_path, platepar_path, output_dir, chunk_frames),
                    kwargs={'flat_path': flat_path, 'dark_path': dark_path, 'unique_id': unique_id,
                            'start_time': start_time}
                )
                proc.start()
                proc.start_time = time.time()
                active_workers[unique_id] = proc
                new_files_queued = True

            # If no workers are active and no new files were queued, we're idle
            if not active_workers and not new_files_queued and not waiting_for_data:
                log.info("Waiting for more data...")
                waiting_for_data = True
            elif active_workers or new_files_queued:
                waiting_for_data = False

            # Generate the night reports which are due. The camera is idle when no files are being
            #   processed, waiting to be processed, or being written
            idle = (not active_workers) and (not stable_files) and (not stability_tracker)
            try:
                reporter.poll(idle)
            except Exception as e:
                log.error("Night report scheduling failed: {:s}".format(repr(e)))
                log.error(traceback.format_exc())

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        log.info("Monitoring stopped by user.")

    finally:

        # Stop the workers right away. Their files have no done flag, so they are processed again after a
        #   restart
        if active_workers:
            log.info("Stopping {:d} worker(s), their files will be processed again after a restart: "
                     "{}".format(len(active_workers), ", ".join(sorted(active_workers))))
            for proc in active_workers.values():
                stopWorker(proc)

        reporter.stop(timeout=SHUTDOWN_TIMEOUT)

        log.info("All workers stopped.")


def monitorMultipleCameras(multicam_ini_path, start_time=None, report_mode=None, retry_failed=False):
    """ Monitor multiple directories for multiple cameras based on an INI config file.

    Arguments:
        multicam_ini_path: Path to the INI config file.

    Keyword arguments:
        start_time: [datetime] Only process files whose recording begins at or after this time (UTC).
            Overrides start_time from the INI file for all cameras.
            None by default, in which case all files are processed.
        report_mode: [str] When to generate the night reports ('sunrise', 'idle', 'external', 'none'), for
            all cameras. The value from each camera's config is used if None.
        retry_failed: [bool] Process the files which failed in previous runs again, for all cameras. False by
            default, in which case retry_failed from the INI file is used.

    Returns:
        None

    Example config file:
        
        [Global]
        
        # Maximum number of parallel worker processes to run across all cameras globally
        nproc = 8
        
        # File extension/type to monitor for (e.g., mkv, mp4, avi, ff, vid)
        file_type = mkv

        # Whether to recursively search subdirectories for video files
        recursive = True

        # Number of frames to process per chunk for star extraction
        chunk_frames = 128

        # Time in seconds to wait before rescanning directories for new files
        poll_interval = 2

        # Set to True to re-process files even if they already have a done.flag
        force = False

        # Set to True to process the files which failed in previous runs again (they are skipped otherwise)
        retry_failed = False

        # Seconds after which a worker which is still processing a file (e.g. stuck in the video decoder) is
        # stopped and the file is retried. 0 disables the limit.
        worker_timeout = 3600

        # Optional. Only process files whose recording begins at or after this time (UTC). Files which
        # begin earlier are skipped, even if they span this time. Can be overridden per camera, and
        # --start_time on the command line overrides both.
        # start_time = 20260101_220000

        [CA0001]
        input_dir = /path/to/CA0001/video
        output_dir = /path/to/CA0001/output
        config = /path/to/CA0001/CA0001.config
        platepar = /path/to/CA0001/platepar.cal
        flat = /path/to/CA0001/flat.bmp
        dark = /path/to/CA0001/dark.bmp
        # start_time = 2026-01-02 01:30:00

        [CA0002]
        input_dir = /path/to/CA0002/video
        output_dir = /path/to/CA0002/output
        config = /path/to/CA0002/CA0002.config
        platepar = /path/to/CA0002/platepar.cal
        
        [CA0003]
        input_dir = /path/to/CA0003/video
        output_dir = /path/to/CA0003/output
        config = /path/to/CA0003/CA0003.config
        platepar = /path/to/CA0003/platepar.cal
    """

    # Initialize the ConfigParser to read the INI configuration
    cp = configparser.ConfigParser()
    cp.read(multicam_ini_path)

    # The [Global] section is mandatory as it contains settings applied to all cameras
    if not cp.has_section('Global'):
        print("ERROR: Multi-camera config must have a [Global] section.")
        sys.exit(1)

    # Extract global parameters, falling back to sensible defaults if not provided
    nproc = cp.getint('Global', 'nproc', fallback=2)
    file_type = cp.get('Global', 'file_type', fallback='mkv')
    recursive = cp.getboolean('Global', 'recursive', fallback=False)
    chunk_frames = cp.getint('Global', 'chunk_frames', fallback=128)
    poll_interval = cp.getfloat('Global', 'poll_interval', fallback=2.0)
    force = cp.getboolean('Global', 'force', fallback=False)
    fail_wait_time = cp.getfloat('Global', 'fail_wait_time', fallback=300.0)
    retry_failed = retry_failed or cp.getboolean('Global', 'retry_failed', fallback=False)
    worker_timeout = cp.getfloat('Global', 'worker_timeout', fallback=WORKER_TIMEOUT)

    # Parse individual camera sections. Each section other than 'Global' defines a single camera.
    cameras = []
    for section in cp.sections():
        if section == 'Global':
            continue
        
        # Extract configuration paths and directories for this specific camera
        cam = {
            'id': section,
            'input_dir': os.path.abspath(cp.get(section, 'input_dir')),
            'output_dir': os.path.abspath(cp.get(section, 'output_dir')),
            'config_path': os.path.abspath(cp.get(section, 'config')),
            'platepar_path': os.path.abspath(cp.get(section, 'platepar')),
            'flat_path': cp.get(section, 'flat', fallback=None),
            'dark_path': cp.get(section, 'dark', fallback=None),
        }

        # Determine the start time for this camera (command line > camera section > [Global])
        try:
            cam['start_time'] = resolveCameraStartTime(cp, section, cli_start_time=start_time)
        except ValueError as e:
            print("ERROR: {}".format(e))
            sys.exit(1)

        # Convert optional calibration paths to absolute paths if they exist
        if cam['flat_path']:
            cam['flat_path'] = os.path.abspath(cam['flat_path'])
        if cam['dark_path']:
            cam['dark_path'] = os.path.abspath(cam['dark_path'])
        
        # Ensure the output directory for this camera exists before we start dumping data there
        os.makedirs(cam['output_dir'], exist_ok=True)
        cameras.append(cam)

    if not cameras:
        print("ERROR: No cameras defined in config.")
        sys.exit(1)

    # To initialize the global logger, we need a directory. We will use the output directory 
    # of the first camera in the list as the primary logging location, or the current directory if missing.
    log_dir = cameras[0]['output_dir'] if cameras else os.getcwd()
    
    # We parse the config of the first camera just to hijack its logging settings for our global logger
    first_config = cr.parse(cameras[0]['config_path'])
    orig_data_dir = first_config.data_dir
    orig_log_dir = first_config.log_dir
    first_config.data_dir = log_dir
    first_config.log_dir = 'logs'

    log_manager = LoggingManager()
    log_manager.initLogging(first_config, 'monitor_multicam_')
    
    # Restore the original config parameters so we don't accidentally mutate the configuration
    first_config.data_dir = orig_data_dir
    first_config.log_dir = orig_log_dir

    log = getLogger("logger")
    log.info("Started multi-camera monitoring with {:d} cameras, nproc={:d}".format(len(cameras), nproc))
    for cam in cameras:
        if cam['start_time'] is not None:
            log.info("Camera {}: only processing files beginning at or after: {} UTC".format(
                cam['id'], cam['start_time']))

    # Only one monitor can work on an output directory, which also catches cameras sharing one
    output_locks = []
    for cam in cameras:
        output_locks.append(lockOutputDir(cam['output_dir']))
        if output_locks[-1] is None:
            log.error("Camera {}: another monitor (PID {:s}) is already working on {:s}, exiting.".format(
                cam['id'], lockOwner(cam['output_dir']), cam['output_dir']))
            sys.exit(1)

    # Schedule the night reports of every camera, only one camera reports at a time
    report_lock = ReportLock()
    reporters = {}
    for cam in cameras:
        reporters[cam['id']] = NightReporter(cam['output_dir'], cam['config_path'], report_mode=report_mode,
                                             fail_wait_time=fail_wait_time, camera_id=cam['id'],
                                             report_lock=report_lock)
        log.info("Camera {}: night report mode: {}".format(cam['id'], reporters[cam['id']].report_mode))

    # Pause the processing of a camera while its output disk is full
    space_guards = {cam['id']: FreeSpaceGuard(cam['output_dir'], reporters[cam['id']],
                                              log_prefix="[{}] ".format(cam['id'])) for cam in cameras}

    # Initialize state tracking dictionaries. Since files from different cameras might have the same name,
    # we track these metrics per camera using nested dictionaries or sets.
    processed_files = {cam['id']: set() for cam in cameras}
    cam_output_dirs = {cam['id']: cam['output_dir'] for cam in cameras}

    # Failed files of every camera, also from previous runs. Files which failed too often are not processed
    #   again
    failed_files = {}
    for cam in cameras:
        failed_files[cam['id']], given_up = loadFailedFiles(cam['output_dir'],
                                                             retry_failed=(retry_failed or force))
        if given_up:
            log.info("Camera {}: skipping {:d} file(s) which failed in previous runs (use --retry_failed to "
                     "process them again): {}".format(cam['id'], len(given_up), ", ".join(sorted(given_up))))
            processed_files[cam['id']] |= given_up

    stability_tracker = {cam['id']: {} for cam in cameras}
    
    # Files are considered "stable" (i.e., finished writing to disk) if their size/mtime hasn't 
    # changed for `stable_wait_time` seconds, or if they are older than `stable_max_age` seconds.
    stable_wait_time = 5
    stable_max_age = 30

    # Dictionary to keep track of currently active processing jobs
    # Format: { unique_id : (multiprocessing.Process, camera_id) }
    active_workers = {} 
    
    # Counter for how many processes each camera is currently running (used for load balancing)
    active_count_per_cam = {cam['id']: 0 for cam in cameras}

    # Pre-scan the output directories for files that have already been processed 
    # (indicated by the presence of a 'done.flag'). We do this to avoid reprocessing old files.
    if not force:
        for cam in cameras:
            for root, dirs, files in walkInput(cam['output_dir']):
                if 'done.flag' in files:
                    # The parent directory name is typically the original base filename
                    completed_base = os.path.basename(root)
                    processed_files[cam['id']].add(completed_base)
            
            if processed_files[cam['id']]:
                log.info("Camera {}: Found {:d} previously processed file(s).".format(
                    cam['id'], len(processed_files[cam['id']])))

    # Keep track of the last camera that received a process slot to facilitate round-robin tiebreaking
    last_assigned_idx = 0

    try:
        # Main monitoring loop
        while True:

            # 1. Clean up finished workers and log their completion status
            finished = []

            for uid, (proc, cam_id) in active_workers.items():
                stopStuckWorker(proc, uid, worker_timeout)
                if not proc.is_alive():
                    proc.join()
                    
                    # A worker has finished, so we free up a slot for this camera
                    active_count_per_cam[cam_id] -= 1
                    
                    if proc.exitcode == 0:

                        # Process exited normally
                        log.info("Successfully processed [{}]: {}".format(cam_id, uid))
                        processed_files[cam_id].add(uid)
                        reporters[cam_id].resultsChanged()

                        # Forget an earlier failure of the file
                        if failed_files[cam_id].pop(uid, None) is not None:
                            saveFailedFiles(cam_output_dirs[cam_id], failed_files[cam_id])

                    elif proc.exitcode == SKIP_EXIT_CODE:

                        # The file begins before the start time, don't queue it again
                        log.info("Skipped [{}]: {} (begins before the start time or after the cutoff of its "
                                 "night)".format(cam_id, uid))
                        processed_files[cam_id].add(uid)

                    # Process failed, it is retried once after fail_wait_time and then given up
                    elif recordFailure(cam_output_dirs[cam_id], failed_files[cam_id], uid, proc.exitcode,
                                       fail_wait_time, log_prefix="[{}] ".format(cam_id)):
                        processed_files[cam_id].add(uid)
                                    
                    finished.append(uid)

            # Remove finished processes from our tracking dictionary
            for uid in finished:
                del active_workers[uid]

            # 2. Collect all new, stable files ready to be processed for each camera
            stable_per_cam = {}
            for cam in cameras:

                cam_id = cam['id']
                candidate_paths = []
                
                # Fetch all paths from the input directory
                try:
                    if recursive:
                        for root, dirs, files in walkInput(cam['input_dir']):
                            if file_type == 'fitsdirs':
                                for d in dirs:
                                    candidate_paths.append(os.path.join(root, d))
                            else:
                                for f in files:
                                    candidate_paths.append(os.path.join(root, f))
                    else:
                        candidate_paths = [os.path.join(cam['input_dir'], f) for f in os.listdir(cam['input_dir'])]
                
                except OSError:
                    log.error("Cannot read directory for camera {}: {}".format(cam_id, cam['input_dir']))
                    continue

                # Forget the files which disappeared while waiting for them to be written
                stability_tracker[cam_id] = {path: trk for path, trk in stability_tracker[cam_id].items()
                                             if path in candidate_paths}

                cam_stable = []
                for file_path in candidate_paths:
                    file_name = os.path.basename(file_path)
                    file_rel_path = os.path.relpath(file_path, cam['input_dir'])
                    
                    # Filter out non-target files and directories
                    if not matchesFileType(file_name, file_type):
                        continue

                    if file_type == 'fitsdirs':
                        if not os.path.isdir(file_path):
                            continue
                        
                        # Ensure it contains .fits files
                        try:
                            fits_files = [f for f in os.listdir(file_path) if f.lower().endswith('.fits')]
                            if not fits_files:
                                continue
                        except OSError:
                            continue

                    else:
                        if os.path.isdir(file_path):
                            continue
                    
                    # Generate a unique ID based on the path name.
                    unique_id = os.path.splitext(file_name)[0]

                    # Ignore files we've already processed or are currently working on
                    is_processed = unique_id in processed_files[cam_id] or any(pb.endswith(unique_id) for pb in processed_files[cam_id])
                    if is_processed or unique_id in active_workers:
                        continue
                    
                    # If the file recently failed, wait before retrying it
                    if unique_id in failed_files[cam_id]:
                        if (time.time() - failed_files[cam_id][unique_id]['last_fail_time']) < fail_wait_time:
                            continue

                    # Attempt to read path metadata for stability checking
                    try:
                        if file_type == 'fitsdirs':
                            # For directories, check the mtime and total size of all .fits files
                            full_fits_paths = [os.path.join(file_path, f) for f in fits_files]
                            curr_mtime = max(os.path.getmtime(f) for f in full_fits_paths)
                            curr_size = sum(os.path.getsize(f) for f in full_fits_paths)
                        else:
                            curr_mtime = os.path.getmtime(file_path)
                            curr_size = os.path.getsize(file_path)
                    except OSError:
                        # File might have been deleted or locked, we'll try again next poll
                        continue

                    # Check if the file is "stable" (fully written to disk)
                    if (time.time() - curr_mtime) > stable_max_age:
                        
                        # Old files are assumed to be fully written
                        stable = True

                    else:

                        cam_trk = stability_tracker[cam_id]
                        if file_path not in cam_trk:
                            # Start tracking a new file
                            log.info("[{}] New {} detected: {} - waiting...".format(
                                cam_id, "directory" if file_type == 'fitsdirs' else "file", file_rel_path))
                            cam_trk[file_path] = {'mtime': curr_mtime, 'size': curr_size, 'since': time.time()}
                            stable = False

                        else:

                            # Verify if the file has changed since the last check
                            trk = cam_trk[file_path]

                            if trk['mtime'] != curr_mtime or trk['size'] != curr_size:
                                # File is still being written
                                cam_trk[file_path] = {'mtime': curr_mtime, 'size': curr_size, 'since': time.time()}
                                stable = False

                            else:

                                # File hasn't changed; check if it has been unchanged for long enough
                                if (time.time() - trk['since']) >= stable_wait_time:
                                    stable = True
                                    del cam_trk[file_path] # We no longer need to track its stability

                                else:
                                    stable = False
                    
                    if stable:
                        cam_stable.append((file_path, curr_mtime, unique_id, file_rel_path))

                # Sort stable files chronologically (oldest files are processed first)
                def getSortTime(file_info):
                    mtime = file_info[1]
                    uid = file_info[2]
                    # Failed files use their last failure time to push them to the back of the queue
                    if uid in failed_files[cam_id]:
                        return max(mtime, failed_files[cam_id][uid]['last_fail_time'])
                    return mtime

                cam_stable.sort(key=getSortTime)
                if cam_stable and space_guards[cam_id].ok():
                    stable_per_cam[cam_id] = cam_stable

            # 3. Load Balancing Assignment
            # We assign available processing slots based on which camera has the fewest active workers
            new_files_queued = False
            
            while len(active_workers) < nproc and stable_per_cam:
                
                # Identify which cameras have stable files waiting to be processed
                available_indices = []
                for i, cam in enumerate(cameras):
                    if cam['id'] in stable_per_cam and len(stable_per_cam[cam['id']]) > 0:
                        available_indices.append(i)
                
                # If no cameras have pending files, break out of the queuing loop
                if not available_indices:
                    break

                # Sort available cameras based on load balancing criteria:
                # 1. Primary sort: The current number of active workers for the camera (fewer is better).
                # 2. Secondary sort: A round-robin distance from the last assigned camera to act as a fair tiebreaker.
                num_cameras = len(cameras)
                available_indices.sort(key=lambda i: (
                    active_count_per_cam[cameras[i]['id']],
                    (i - last_assigned_idx - 1) % num_cameras
                ))

                # Pick the most eligible camera and extract its oldest stable file
                chosen_idx = available_indices[0]
                chosen_cam = cameras[chosen_idx]
                cam_id = chosen_cam['id']
                
                file_info = stable_per_cam[cam_id].pop(0)
                file_path, mtime, unique_id, file_rel_path = file_info

                # Stop tracking stability for this file since it's about to be processed
                if file_path in stability_tracker[cam_id]:
                    del stability_tracker[cam_id][file_path]

                log.info("[{}] Putting file {} on queue (active processes for this cam: {})...".format(
                    cam_id, file_rel_path, active_count_per_cam[cam_id]))

                # Spawn the worker process
                proc = multiprocessing.Process(
                    target=processFileWorker,
                    args=(file_path, chosen_cam['config_path'], chosen_cam['platepar_path'], chosen_cam['output_dir'], chunk_frames),
                    kwargs={'flat_path': chosen_cam['flat_path'], 'dark_path': chosen_cam['dark_path'], 'unique_id': unique_id,
                            'start_time': chosen_cam['start_time']}
                )
                proc.start()
                proc.start_time = time.time()
                
                # Update load balancing metrics and tracking dictionaries
                active_workers[unique_id] = (proc, cam_id)
                active_count_per_cam[cam_id] += 1
                new_files_queued = True
                last_assigned_idx = chosen_idx

                # If this camera has no more stable files, remove it from the pool of candidates for this polling cycle
                if len(stable_per_cam[cam_id]) == 0:
                    del stable_per_cam[cam_id]
            
            # Generate the night reports which are due. A camera is idle when none of its files are being
            #   processed, waiting to be processed, or being written
            for cam in cameras:
                cam_id = cam['id']
                idle = (active_count_per_cam[cam_id] == 0) and (not stable_per_cam.get(cam_id)) \
                    and (not stability_tracker[cam_id])

                try:
                    reporters[cam_id].poll(idle)
                except Exception as e:
                    log.error("[{}] Night report scheduling failed: {}".format(cam_id, repr(e)))
                    log.error(traceback.format_exc())

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        log.info("Multi-camera monitoring stopped by user.")

    finally:

        # Stop the workers right away. Their files have no done flag, so they are processed again after a
        #   restart
        if active_workers:
            log.info("Stopping {:d} worker(s), their files will be processed again after a restart: "
                     "{}".format(len(active_workers), ", ".join(sorted(active_workers))))
            for proc, cam_id in active_workers.values():
                stopWorker(proc)

        for reporter in reporters.values():
            reporter.stop(timeout=SHUTDOWN_TIMEOUT)

        log.info("All workers stopped.")


def findConfigFile(input_dir, cml_config=None):
    """ Find the config file in the input directory or use the one provided.

    Arguments:
        input_dir: [str] Input directory to search for config files.

    Keyword arguments:
        cml_config: [str] Path to a config file provided on the command line.

    Return:
        [str] Absolute path to the config file.
    """

    if cml_config is not None:
        config_path = os.path.abspath(cml_config)
        if not os.path.isfile(config_path):
            print("ERROR: Config file not found: {}".format(config_path))
            sys.exit(1)
        return config_path

    # Look for a .config file in the input directory
    config_files = [f for f in os.listdir(input_dir) 
                    if f.endswith('.config') and f != 'bak.config']

    if len(config_files) == 1:
        return os.path.join(os.path.abspath(input_dir), config_files[0])
    elif len(config_files) > 1:
        print("ERROR: Multiple .config files found in {}:".format(input_dir))
        for cf in config_files:
            print("    {}".format(cf))
        print("Use --config to specify which one to use.")
        sys.exit(1)
    else:
        print("ERROR: No .config file found in {}. Use --config to specify one.".format(input_dir))
        sys.exit(1)


def findPlatepar(input_dir, config, cml_platepar=None):
    """ Find the platepar file in the input directory or use the one provided.

    Arguments:
        input_dir: [str] Input directory to search for platepar files.
        config: [Config instance] Loaded config object.

    Keyword arguments:
        cml_platepar: [str] Path to a platepar file provided on the command line.

    Return:
        [str] Absolute path to the platepar file.
    """

    if cml_platepar is not None:
        platepar_path = os.path.abspath(cml_platepar)
        if not os.path.isfile(platepar_path):
            print("ERROR: Platepar file not found: {}".format(platepar_path))
            sys.exit(1)
        return platepar_path

    # Look for the default platepar in the input directory
    default_path = os.path.join(os.path.abspath(input_dir), config.platepar_name)
    if os.path.isfile(default_path):
        return default_path

    print("ERROR: No platepar file '{}' found in {}. Use --platepar to specify one.".format(
        config.platepar_name, input_dir))
    sys.exit(1)



if __name__ == "__main__":

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(
        description="Monitor a directory for new files and process them through star extraction, "
                    "meteor detection, and astrometric recalibration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m RMS.MonitorProcessFrameInterface vid /path/to/input --output /path/to/output
  python -m RMS.MonitorProcessFrameInterface mkv /path/to/input -p ~/platepar_cmn2010.cal
  python -m RMS.MonitorProcessFrameInterface ff /path/to/input --nproc 4
  python -m RMS.MonitorProcessFrameInterface vid /path/to/input --start_time "2026-01-01 22:00:00"
        """
    )

    arg_parser.add_argument('file_type', type=str, nargs='?', default=None,
        help="File type to monitor for. Supported: vid, ff, mkv, mp4, avi, mov, wmv. "
             "Any other value will be treated as a file extension."
    )

    arg_parser.add_argument('input_dir', type=str, nargs='?', default=None,
        help="Path to the directory to monitor for new files."
    )

    arg_parser.add_argument('--output', '-o', type=str, default=None,
        help="Output directory for results. Default: same as input directory."
    )

    arg_parser.add_argument('--config', '-c', type=str, default=None,
        help="Path to a .config file. Default: look for one in the input directory."
    )

    arg_parser.add_argument('--platepar', '-p', type=str, default=None,
        help="Path to a platepar file. Default: look for one in the input directory."
    )

    arg_parser.add_argument('--nproc', '-n', type=int, default=2,
        help="Number of parallel worker processes. Default: 2."
    )

    arg_parser.add_argument('--chunk_frames', type=int, default=128,
        help="Number of frames per chunk for star extraction. Default: 128."
    )

    arg_parser.add_argument('--force', '-f', action='store_true', default=False,
        help="Re-process files even if they have already been processed (done.flag exists)."
    )

    arg_parser.add_argument('--recursive', '-r', action='store_true', default=False,
        help="Recursively monitor subdirectories for files."
    )

    arg_parser.add_argument('--flat', type=str, default=None,
        help="Path to a flat field image file. The flat is applied only if given (use_flat in the config "
             "file is ignored)."
    )

    arg_parser.add_argument('--dark', type=str, default=None,
        help="Path to a dark frame (or bias) image file. The dark is applied only if given (use_dark in the "
             "config file is ignored)."
    )

    arg_parser.add_argument('--multicam', '-m', type=str, default=None,
        help="Path to an INI file containing multiple camera configurations. If provided, other arguments like input_dir are ignored."
    )

    arg_parser.add_argument('--start_time', '-s', type=str, default=None,
        help="Only process files whose recording begins at or after this UTC time, e.g. 20260101_220000 "
             "or '2026-01-01 22:00:00'. Files which begin earlier are skipped, even if they span this "
             "time. In --multicam mode, overrides start_time in the INI file."
    )

    arg_parser.add_argument('--report_mode', type=str, default=None,
        choices=cr.MONITOR_REPORT_MODES,
        help="When to generate the night reports: 'sunrise' (after the night is over and no new data came in "
             "for monitor_report_quiet_min), 'idle' (whenever all data is processed), 'external' (only when "
             "the file .report_now is created in the output directory), or 'none'. Overrides "
             "monitor_report_mode in the config file."
    )

    arg_parser.add_argument('--worker_timeout', type=float, default=WORKER_TIMEOUT,
        help="Seconds after which a worker which is still processing a file (e.g. stuck in the video "
             "decoder) is stopped and the file is retried. 0 disables the limit. Default: {:d}. In --multicam "
             "mode, worker_timeout in the [Global] section of the INI file is used.".format(WORKER_TIMEOUT)
    )

    arg_parser.add_argument('--retry_failed', action='store_true',
        help="Process the files which failed twice in previous runs again. They are skipped by default "
             "(recorded in " + FAILED_FILES_NAME + " in the output directory)."
    )

    # Parse
    cml_args = arg_parser.parse_args()

    signal.signal(signal.SIGTERM, stopOnSigterm)

    # Parse the start time
    start_time = None
    if cml_args.start_time is not None:
        try:
            start_time = parseStartTime(cml_args.start_time)
        except ValueError as e:
            print("ERROR: {}".format(e))
            sys.exit(1)

    if cml_args.multicam is not None:
        multicam_path = os.path.abspath(cml_args.multicam)
        if not os.path.isfile(multicam_path):
            print("ERROR: Multicam config file does not exist: {}".format(multicam_path))
            sys.exit(1)
        monitorMultipleCameras(multicam_path, start_time=start_time, report_mode=cml_args.report_mode,
                               retry_failed=cml_args.retry_failed)
        sys.exit(0)

    if cml_args.file_type is None or cml_args.input_dir is None:
        print("ERROR: file_type and input_dir are required unless --multicam is used.")
        arg_parser.print_help()
        sys.exit(1)


    # Validate input directory
    input_dir = os.path.abspath(cml_args.input_dir)
    if not os.path.isdir(input_dir):
        print("ERROR: Input directory does not exist: {}".format(input_dir))
        sys.exit(1)

    # Set output directory
    output_dir = os.path.abspath(cml_args.output) if cml_args.output else input_dir

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Find and load the config file
    config_path = findConfigFile(input_dir, cml_args.config)
    config = cr.parse(config_path)
    print("Using config: {}".format(config_path))

    # Initialize the logger (set the log dir to output_dir)
    orig_data_dir = config.data_dir
    orig_log_dir = config.log_dir
    
    config.data_dir = output_dir
    config.log_dir = 'logs'

    log_manager = LoggingManager()
    log_manager.initLogging(config, 'monitor_')
    log = getLogger("logger")

    config.data_dir = orig_data_dir
    config.log_dir = orig_log_dir

    # Find the platepar file  
    platepar_path = findPlatepar(input_dir, config, cml_args.platepar)
    log.info("Using platepar: {}".format(platepar_path))

    # Start monitoring
    monitorDirectory(
        input_dir,
        cml_args.file_type,
        config_path,
        platepar_path,
        output_dir,
        nproc=cml_args.nproc,
        chunk_frames=cml_args.chunk_frames,
        recursive=cml_args.recursive,
        force=cml_args.force,
        flat_path=cml_args.flat,
        dark_path=cml_args.dark,
        start_time=start_time,
        report_mode=cml_args.report_mode,
        worker_timeout=cml_args.worker_timeout,
        retry_failed=cml_args.retry_failed
    )
