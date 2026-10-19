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
import hashlib
import json
import os
import re
import shutil
import sys
import time
import logging
import traceback
import multiprocessing
import configparser
import signal

import numpy as np

import RMS.ConfigReader as cr
from RMS.Formats.FrameInterface import detectInputType, getCacheID
from RMS.Formats.FFfile import validFFName, constructFFName
from RMS.Formats import FFpng
from RMS.Formats.Platepar import Platepar
from RMS.Routines import Image
from RMS.MonitorNightReport import exitWithMonitor, lockOutputDir, lockOwner, \
    MONITOR_LOCK_FILE_NAME, nightDirPath, nightInfo, NightReporter, readDoneFlag, readStateFile, ReportLock, \
    selectPlatepar, stopProcess, writeDoneFlag
from RMS.MonitorStatus import MonitorStatus, simulateQueues
from RMS.ExtractStarsFrameInterface import extractStarsFrameInterface
from RMS.DetectStarsAndMeteors import (
    detectStarsAndMeteorsFrameInterface,
    saveResultsFrameInterface,
)
from RMS.DeleteOldObservations import availableSpace
from RMS.DetectionTools import binImageCalibration, findMaskPath, loadImageCalibration
from RMS.Astrometry.ApplyRecalibrate import applyRecalibrate
from RMS.Logger import LoggingManager, getLogger
from RMS.ContinuousResults import JOIN_RECORD, processPendingJoins
from RMS.Misc import RmsDateTime


# Get the logger from the main module
log = getLogger("logger")


# File type to extension mapping
FILE_TYPE_MAP = {
    'vid': ['.vid'],
    'fblk': ['.fblk'],
    'ff':  None,  # Special handling via validFFName()
    'mkv': ['.mkv'],
    'mp4': ['.mp4'],
    'avi': ['.avi'],
    'mov': ['.mov'],
    'wmv': ['.wmv'],
    'fitsdirs': None,
}

# Exit code of a worker process which skipped a file because it begins before the start time, or because the
#   processing of its night was cut off (monitor_night_cutoff_hours)
SKIP_EXIT_CODE = 3

# Exit code of a worker whose file changed while it was processed (the recording was still being written), so
#   the file is processed again
CHANGED_EXIT_CODE = 4

# Exit code of a worker whose input disappeared while it was processed (e.g. it was deleted or moved away), so it
#   is not processed again
VANISHED_EXIT_CODE = 5

# Default time limit (seconds) for processing one file, after which the worker is stopped and the file retried
WORKER_TIMEOUT = 3600

# Seconds the night report, cleanup and upload are given to finish when the monitor is stopped, before they
#   are terminated (an unfinished report is made again after a restart)
SHUTDOWN_TIMEOUT = 30

# Record of the input files which failed, kept in the output directory: {unique_id: {'count': int,
#   'kills': int, 'last_fail_time': float}}. A file is retried once after fail_wait_time, and after
#   MAX_FAILURES failures it is skipped, also after a restart, unless the monitor is started with
#   --retry_failed. A worker killed from outside (e.g. by the system when the memory runs out) doesn't count
#   as a failure of the file, it is retried until it was killed MAX_KILLS times
FAILED_FILES_NAME = '.failed_files.json'
MAX_FAILURES = 2
MAX_KILLS = 5

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


def isHidden(name):
    """ Whether a file or directory name is hidden (starts with a dot), e.g. a trash directory or the temporary
        file of a copy in progress, which are never inputs.

    Arguments:
        name: [str] Name of the file or directory.

    Return:
        [bool] True if the name is hidden.
    """

    return name.startswith('.')


def walkInput(top_dir, output_dir, config):
    """ Walk the directory tree like os.walk, without descending into the night, archive and log directories
        of the monitor, which are in the input directory if it is also the output directory (the default).
        Other directories with the same names are walked.

    Arguments:
        top_dir: [str] Directory to walk.
        output_dir: [str] Output directory of the monitor.
        config: [Config] Configuration of the camera, which names the night and archive directories.

    Return:
        Yields (root, dirs, files) like os.walk. Raises OSError if the directory can't be read (os.walk alone
            yields nothing then, e.g. when the network share of the input is gone).
    """

    # Fail if the directory can't be read
    os.listdir(top_dir)

    output_dirs = {os.path.realpath(os.path.join(output_dir, dir_name))
                   for dir_name in (config.captured_dir, config.archived_dir, 'logs')}

    for root, dirs, files in os.walk(top_dir):

        # Don't descend into the output directory of another monitor
        if (root != top_dir) and (MONITOR_LOCK_FILE_NAME in files):
            dirs[:] = []
            continue

        # Hidden directories (e.g. .Trash-1000 of a desktop, whose content is deleted at any time) are not inputs
        dirs[:] = [d for d in dirs if (not isHidden(d))
                   and (os.path.realpath(os.path.join(root, d)) not in output_dirs)]
        yield root, dirs, files


# Input directories which can't be read: {dir_path: monotonic time of the last error message}
_UNREADABLE_DIRS = {}

# Seconds between the error messages about an input directory which still can't be read
UNREADABLE_LOG_INTERVAL = 3600


def logReadable(dir_path, readable, log_prefix=''):
    """ Log that the input directory can't be read, but only once per UNREADABLE_LOG_INTERVAL while it lasts
        (e.g. an unmounted disk), as the directory is read every few seconds. Log when it can be read again.

    Arguments:
        dir_path: [str] Input directory.
        readable: [bool] True if it could be read.

    Keyword arguments:
        log_prefix: [str] Prefix of the log messages, e.g. the camera ID. Empty by default.
    """

    if readable:
        if _UNREADABLE_DIRS.pop(dir_path, None) is not None:
            log.info("{}The input directory {} can be read again".format(log_prefix, dir_path))
        return

    last = _UNREADABLE_DIRS.get(dir_path)
    if (last is None) or (time.monotonic() - last >= UNREADABLE_LOG_INTERVAL):
        log.error("{}Cannot read the input directory {} (repeated every {:.0f} min while it lasts)".format(
            log_prefix, dir_path, UNREADABLE_LOG_INTERVAL/60))
        _UNREADABLE_DIRS[dir_path] = time.monotonic()


def inputExtension(file_type):
    """ Return the extension of the input files of the given type, None if they don't have a single one
        (FF files, FITS directories). Types which are not in FILE_TYPE_MAP are the extension itself, as in
        matchesFileType.
    """

    file_type = file_type.lower()
    if file_type not in FILE_TYPE_MAP:
        return '.' + file_type

    extensions = FILE_TYPE_MAP[file_type]

    return extensions[0] if extensions else None


def uniqueId(file_rel_path):
    """ Return the ID of an input file, which names its results directory. Files directly in the input
        directory are identified by their name without the extension. Files in subdirectories (--recursive)
        also get a short hash of the subdirectory, so files with the same name in different subdirectories
        (e.g. HH-MM-SS.mkv in dated directories) are kept apart.

    Arguments:
        file_rel_path: [str] Path of the file relative to the input directory.

    Return:
        [str] Unique ID.
    """

    sub_dir, file_name = os.path.split(os.path.normpath(file_rel_path))
    file_base = os.path.splitext(file_name)[0]

    if not sub_dir:
        return file_base

    return "{:s}_{:s}".format(file_base, hashlib.sha1(sub_dir.encode('utf-8')).hexdigest()[:10])


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




def isInputPath(file_path, file_type):
    """ Check if a path is an input of the monitor: a file of the given type, or for FITS directories a directory
        with FITS frames.

    Arguments:
        file_path: [str] Path to check.
        file_type: [str] File type identifier (e.g. 'vid', 'ff', 'mkv', 'fitsdirs').

    Return:
        [bool] True if it is an input.
    """

    if not matchesFileType(os.path.basename(file_path), file_type):
        return False

    # A FITS directory is a directory with FITS frames, any other input is a file
    if file_type == 'fitsdirs':
        try:
            return os.path.isdir(file_path) and any(f.lower().endswith('.fits') for f in os.listdir(file_path))
        except OSError:
            return False

    return not os.path.isdir(file_path)


def inputState(file_path, file_type):
    """ The time of the last change and the size of an input, which tell whether it is still being written. A
        FITS directory changes with its frames, so it has the newest time and the total size of its frames.

    Arguments:
        file_path: [str] Path of the input (see isInputPath).
        file_type: [str] File type identifier.

    Return:
        (mtime, size): [tuple] Time of the last change (Unix time) and size (bytes). None if it can't be read
            (e.g. it was deleted meanwhile).
    """

    try:

        # A FITS directory: the newest of its frames and their total size
        if file_type == 'fitsdirs':
            fits_paths = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.lower().endswith('.fits')]
            if not fits_paths:
                return None
            return max(os.path.getmtime(f) for f in fits_paths), sum(os.path.getsize(f) for f in fits_paths)

        # A file: its own time and size
        return os.path.getmtime(file_path), os.path.getsize(file_path)

    except OSError:
        return None


def inputSize(file_path, file_type):
    """ The size of an input (bytes, the total of the frames of a FITS directory), None if it can't be read. """

    input_state = inputState(file_path, file_type)

    return input_state[1] if input_state is not None else None


def scanInputs(input_dir, output_dir, config, file_type, recursive):
    """ The inputs in the input directory with the time of their last change and their size, for the status of
        the monitor (MonitorStatus). The inputs are found as by the monitor. For FITS directories only the time
        of the directory is read (it changes when frames are added), as reading all frames would take long.

    Arguments:
        input_dir: [str] Input directory.
        output_dir: [str] Output directory, which is skipped when it is inside the input directory.
        config: [Config] Configuration.
        file_type: [str] File type identifier.
        recursive: [bool] The inputs are also in the subdirectories.

    Return:
        [list] (path, mtime, size) of every input, size None for FITS directories. Raises OSError if the input
            directory can't be read.
    """

    # Collect the paths as the monitor does
    if recursive:
        paths = []
        for root, dirs, files in walkInput(input_dir, output_dir, config):
            paths += [os.path.join(root, name) for name in (dirs if file_type == 'fitsdirs' else files)
                      if not isHidden(name)]
    else:
        paths = [os.path.join(input_dir, name) for name in os.listdir(input_dir) if not isHidden(name)]

    # Keep the inputs, with the time and the size of their file or directory
    inputs = []
    for path in paths:
        if not isInputPath(path, file_type):
            continue
        try:
            st = os.stat(path)
        except OSError:
            continue
        inputs.append((path, st.st_mtime, None if file_type == 'fitsdirs' else st.st_size))

    return inputs


def chunkImageName(config, img_handle, first_frame):
    """ The FF name of the image pair of a chunk of frames (see ChunkImageSaver), after the time of its first
        frame.

    Arguments:
        config: [Config] Configuration.
        img_handle: [FrameInterface instance] Image handle.
        first_frame: [int] First frame of the chunk.

    Return:
        [str] FF name of the pair.
    """

    begin_time = img_handle.currentFrameTime(frame_no=first_frame, dt_obj=True)

    return FFpng.pairNames(constructFFName(config.stationID, begin_time, frame=first_frame, ext=None))[0]


def savedChunkImages(config, img_handle):
    """ The image pairs which ChunkImageSaver saves for an input: one for every whole chunk of frames, and one
        for the frames after them if they make at least half a chunk (see ChunkImageSaver.saveTrailing).

    Arguments:
        config: [Config] Configuration.
        img_handle: [FrameInterface instance] Image handle.

    Return:
        [list] (time of the first frame of the chunk, FF name of its pair), sorted by time.
    """

    # The first frames of the whole chunks, and of the frames after them if they make at least half a chunk
    chunk_frames = img_handle.chunk_frames
    firsts = [k*chunk_frames for k in range(img_handle.total_fr_chunks)]
    if img_handle.total_frames - img_handle.total_fr_chunks*chunk_frames >= chunk_frames//2:
        firsts.append(img_handle.total_fr_chunks*chunk_frames)

    return [(img_handle.currentFrameTime(frame_no=f, dt_obj=True), chunkImageName(config, img_handle, f))
            for f in firsts]


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

        ff_name = chunkImageName(self.config, img_handle, first_frame)
        begin_time = img_handle.currentFrameTime(frame_no=first_frame, dt_obj=True)

        # Apply the dark and the flat, like for the star extraction (the cached chunk is not modified)
        maxpixel, avepixel = ff.maxpixel, ff.avepixel

        if self.dark is not None:
            maxpixel = Image.applyDark(maxpixel, self.dark)
            avepixel = Image.applyDark(avepixel, self.dark)

        if self.flat_struct is not None:
            maxpixel = Image.applyFlat(maxpixel, self.flat_struct)
            avepixel = Image.applyFlat(avepixel, self.flat_struct)

        # Summed binned pixels are scaled to the levels of the full size pixels, like averaged ones, so the
        #   images have the levels and the bit depth of the data
        if (self.config.detection_binning_factor > 1) and (self.config.detection_binning_method == 'sum'):
            n_binned = self.config.detection_binning_factor**2
            dtype = np.uint8 if self.config.bit_depth <= 8 else maxpixel.dtype
            maxpixel, avepixel = [np.clip(np.round(img.astype(np.float64)/n_binned), 0,
                                          np.iinfo(dtype).max).astype(dtype) for img in (maxpixel, avepixel)]

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



# Directory of the local copies of the input files of a worker in the staging directory, with its PID. The
#   copies keep the name of the input file, as video files are timed by their names
STAGED_DIR_NAME = 'rms_monitor_{:d}'

# Free space which is left in the staging directory when a file is copied there, in bytes
STAGING_FREE_MARGIN = 1024**3


def stageInputFile(file_path, staging_dir):
    """ Copy an input file into the local staging directory (monitor_staging_dir), so it is read only once
        over the network: the processing reads every file twice. The copy is in a directory of the worker
        and keeps the file name. FF files and directories (FITS directories) are processed where they are, as
        are files for which there is not enough space or which can't be copied.

    Arguments:
        file_path: [str] Path to the input file.
        staging_dir: [str] Staging directory, empty or None if staging is disabled.

    Return:
        [str] Path to the local copy, or file_path if it is processed where it is.
    """

    if (not staging_dir) or (not os.path.isfile(file_path)) or validFFName(os.path.basename(file_path)):
        return file_path

    os.makedirs(staging_dir, exist_ok=True)

    file_size = os.path.getsize(file_path)
    free_space = shutil.disk_usage(staging_dir).free
    if free_space < file_size + STAGING_FREE_MARGIN:
        log.warning("Not enough space in the staging directory {} ({:.1f} GB free) for {} ({:.1f} GB), "
                    "processing it where it is".format(staging_dir, free_space/1024**3, file_path,
                                                       file_size/1024**3))
        return file_path

    worker_dir = os.path.join(staging_dir, STAGED_DIR_NAME.format(os.getpid()))
    staged_path = os.path.join(worker_dir, os.path.basename(file_path))

    # Several workers can copy at the same time, so the space can still run out. The file is then processed
    #   where it is, instead of failing
    try:
        os.makedirs(worker_dir, exist_ok=True)
        shutil.copyfile(file_path, staged_path)

    except OSError as e:
        log.warning("{} could not be copied to the staging directory ({}), processing it where it "
                    "is".format(file_path, repr(e)))
        shutil.rmtree(worker_dir, ignore_errors=True)
        return file_path

    # The copy left the input file in the page cache too, which takes as much memory as the local copy. Only
    #   the local copy is read again, so the input is dropped from the cache, which leaves the memory to the
    #   local copies of the files processed at the same time
    dropFromPageCache(file_path)

    return staged_path


def dropFromPageCache(file_path):
    """ Tell the kernel that the cached pages of a file are not needed anymore, so they are freed. Only a
        hint, which does nothing where it is not supported.

    Arguments:
        file_path: [str] Path to the file.
    """

    if not hasattr(os, 'posix_fadvise'):
        return

    try:
        fd = os.open(file_path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)

    except OSError as e:
        log.debug("Could not drop {} from the page cache: {}".format(file_path, repr(e)))


def removeStagedFiles(staging_dir, pid=None):
    """ Delete the local copies of the input files in the staging directory: those of the given worker (e.g.
        when it ended), or, without a PID, those of the workers which are not running anymore (e.g. left
        behind by a monitor which was killed).

    Arguments:
        staging_dir: [str] Staging directory, empty or None if staging is disabled.

    Keyword arguments:
        pid: [int] PID of the worker whose copies are deleted. None by default.
    """

    if (not staging_dir) or (not os.path.isdir(staging_dir)):
        return

    for file_name in os.listdir(staging_dir):

        match = re.match(r'rms_monitor_(\d+)$', file_name)
        if match is None:
            continue

        file_pid = int(match.group(1))
        if pid is not None:
            if file_pid != pid:
                continue

        # Keep the copies of running workers (also of other users, which can't be signalled)
        else:
            try:
                os.kill(file_pid, 0)
                continue
            except ProcessLookupError:
                pass
            except OSError:
                continue

        shutil.rmtree(os.path.join(staging_dir, file_name), ignore_errors=True)


def calibrationMatchesFrames(img, img_handle, binning_factor):
    """ Check if a calibration image (dark, flat) has the size of the frames. The frame size of the image
        handle is binned with detection binning, the calibration images have the full resolution.

    Arguments:
        img: [ndarray] Calibration image, None if not used.
        img_handle: [FrameInterface] Image handle of the input file.
        binning_factor: [int] Detection binning factor.

    Return:
        [bool] True if the size matches, or if it can't be checked.
    """

    if (img is None) or (getattr(img_handle, 'nrows', None) is None):
        return True

    binned_shape = (img.shape[0]//binning_factor, img.shape[1]//binning_factor)

    return binned_shape == (img_handle.nrows, img_handle.ncols)


def hasResults(results_dir):
    """ Check if the results directory has the results of a processed file: a done flag which doesn't mark
        the file as skipped.
    """

    if not os.path.isfile(os.path.join(results_dir, 'done.flag')):
        return False

    return not readDoneFlag(results_dir).get('skipped')


def readPlatepar(platepar_path):
    """ Read a platepar.

    Arguments:
        platepar_path: [str] Path to the platepar file, None if there is no platepar.

    Return:
        [Platepar] The platepar, or None if there is none or it can't be read.
    """

    if platepar_path is None:
        return None

    # Platepar.read returns False for a missing file, and a damaged file raises an exception
    try:
        platepar = Platepar()
        if platepar.read(platepar_path) is not False:
            return platepar

    except Exception:
        pass

    log.warning("The platepar can't be read: {}".format(platepar_path))
    return None



def recalibrateResults(results_dir, config, platepar_path=None, proc_log=None, matched_filter=False):
    """ Recalibrate the detections in the FTPdetectinfo of the results of a file (ApplyRecalibrate), which adds
        their astrometry and photometry, and if asked the completed results of the separate matched-filter pass.

    Arguments:
        results_dir: [str] Results directory of the file.
        config: [Config] Configuration.

    Keyword arguments:
        platepar_path: [str] Platepar which is copied into the results directory first. None by default (the
            platepar in the results directory is used).
        proc_log: [Logger] Logger, the module logger if None.
        matched_filter: [bool] Also recalibrate the results of the separate matched-filter pass (when a new
            platepar is installed; in the processing of the file the pass already used its platepar). A failure
            there is logged and doesn't affect the normal detections. False by default.

    Return:
        [bool] True if there was an FTPdetectinfo file to recalibrate.
    """

    proc_log = log if proc_log is None else proc_log

    if platepar_path is not None:
        shutil.copy2(platepar_path, os.path.join(results_dir, config.platepar_name))

    # The detections of the file (not the backups which earlier recalibrations left)
    from RMS.Formats.FTPdetectinfo import validDefaultFTPdetectinfo
    ftpdetect_files = sorted(path for path in glob.glob(os.path.join(results_dir, 'FTPdetectinfo_*.txt'))
                             if validDefaultFTPdetectinfo(os.path.basename(path)))
    if ftpdetect_files:

        proc_log.info("Running ApplyRecalibrate on: {}".format(ftpdetect_files[0]))

        # Run recalibration with load_all=True. The ECSV files of the detections are saved to the results directory
        #   and collected into the night directory by the night report
        applyRecalibrate(
            ftpdetect_files[0], config,
            # The calibration variation plots are made for the whole night in the night report
            generate_plot=False,
            load_all=True,
            generate_ufoorbit=False,
            ecsv_out=config.monitor_save_ecsv,
        )

    # The detections of the separate matched-filter pass get the same platepar; a failure there doesn't fail the
    #   file
    if matched_filter:
        try:
            _recalibrateMatchedFilter(results_dir, config, os.path.join(results_dir, config.platepar_name),
                                      proc_log)
        except Exception:
            proc_log.error("Recalibrating the matched-filter results failed:\n" + traceback.format_exc())

    return bool(ftpdetect_files)


def _recalibrateMatchedFilter(results_dir, config, platepar_path, proc_log):
    """ Recalibrate the detections of the matched filter in the subdirectory of the results of a file, if its
        results are complete and it has detections (see MatchedFilterDetection.saveMatchedFilterResults).

    Arguments:
        results_dir: [str] Results directory of the file.
        config: [Config] Configuration.
        platepar_path: [str] Platepar the detections are recalibrated with.
        proc_log: [Logger] Logger.
    """

    from RMS.Formats import FTPdetectinfo
    from RMS.MatchedFilterDetection import DONE_NAME, MATCHED_FILTER_DIR

    # Only complete results with detections, and only with a platepar
    mf_dir = os.path.join(results_dir, MATCHED_FILTER_DIR)
    if (not os.path.isfile(os.path.join(mf_dir, DONE_NAME))) or (not os.path.isfile(platepar_path)):
        return

    for ftp_path in sorted(glob.glob(os.path.join(mf_dir, 'FTPdetectinfo_*.txt'))):

        # The backups which earlier recalibrations left are not recalibrated (every recalibration makes one)
        if (not FTPdetectinfo.validDefaultFTPdetectinfo(os.path.basename(ftp_path))) \
                or (not FTPdetectinfo.readFTPdetectinfo(mf_dir, os.path.basename(ftp_path))):
            continue

        # As for the normal detections, the platepar is copied next to them first
        proc_log.info("Running ApplyRecalibrate on: {}".format(ftp_path))
        shutil.copy2(platepar_path, os.path.join(mf_dir, config.platepar_name))
        applyRecalibrate(ftp_path, config, generate_plot=False, generate_ufoorbit=False,
                         ecsv_out=config.monitor_save_ecsv)


def backfillCalibration(output_dir, config, night_name, proc_log=None, replaced_mtimes=None):
    """ Add the astrometry and the photometry of the installed automatic platepar to the files which need it (the
        detections themselves are not repeated):
          - the files of the night processed without a platepar,
          - the files of the night whose platepar failed the judgement of the automatic calibration,
          - with replaced_mtimes, the files of every night not reported yet which were calibrated with a replaced
            version, unless their own judgement passed it.
        One backfill runs at a time per camera, and it stops when another platepar is installed meanwhile (the
        backfill of that install continues).

    Arguments:
        output_dir: [str] Output directory of the monitor.
        config: [Config] Configuration.
        night_name: [str] Name of the night.

    Keyword arguments:
        proc_log: [Logger] Logger, the module logger if None.
        replaced_mtimes: [list] Versions (modification times, see plateparTime) of the platepars which the
            installed one replaced (e.g. a failing platepar, or a platepar on trial which was rolled back). None by
            default.

    Return:
        [list] Results directories which were calibrated.
    """

    from RMS.Astrometry.AutoCalibration import FILE_RESULT_NAME, _StateLock
    from RMS.MonitorNightReport import autoPlateparPath, readDoneFlag, readReportStates, scanNights

    proc_log = log if proc_log is None else proc_log
    replaced_mtimes = [t for t in (replaced_mtimes or []) if t is not None]

    # Only the automatically fitted platepar is added (a given platepar was used from the start)
    platepar_path = autoPlateparPath(output_dir, config)
    if not os.path.isfile(platepar_path):
        return []

    calibrated = []
    with _StateLock(output_dir, name='auto_calibration_backfill.lock'):

        # The version which is added; the nights which are searched: this one, and with replaced versions every
        #   night which is not reported yet
        platepar_mtime = plateparTime(platepar_path)
        nights = scanNights(output_dir, config)
        night_names = [night_name]
        if replaced_mtimes:
            reported = readReportStates(output_dir).get('nights', {})
            night_names += sorted(n for n in nights if (n != night_name)
                                  and (not (reported.get(n) or {}).get('reported_at')))

        for name in night_names:
            for results_dir, info in sorted(nights.get(name, {}).items()):
                results_path = os.path.join(output_dir, results_dir)
                if not _needsCalibration(info, results_path, platepar_mtime, replaced_mtimes, FILE_RESULT_NAME):
                    continue

                # Another platepar installed meanwhile: its own backfill continues
                if plateparTime(platepar_path) != platepar_mtime:
                    proc_log.info("A new platepar was installed during the backfill, it is stopped")
                    break

                # Recalibrate the detections of the file with the fitted platepar, and mark it as calibrated with
                #   the version it got (the copy in the results keeps its modification time)
                try:
                    recalibrateResults(results_path, config, platepar_path=platepar_path, proc_log=proc_log,
                                       matched_filter=True)

                    # Read the flag again, it may have changed since the scan
                    info = readDoneFlag(results_path)
                    info['calibrated'] = True
                    info['calibrated_later'] = True
                    info['platepar_mtime'] = plateparTime(os.path.join(results_path, config.platepar_name))
                    writeDoneFlag(results_path, info)
                    calibrated.append(results_dir)

                except Exception:

                    # A file which fails is left as it is, the others are still calibrated
                    proc_log.error("Adding the calibration to {:s} failed:\n{:s}".format(results_dir,
                                                                                        traceback.format_exc()))
            else:
                continue
            break

    # Report how many files got the calibration
    if calibrated:
        proc_log.info("Added the calibration to {:d} files processed without it or with a failing or replaced "
                      "platepar".format(len(calibrated)))

    return calibrated


def _needsCalibration(info, results_path, platepar_mtime, replaced_mtimes, result_name):
    """ Whether the backfill calibrates a file with the installed platepar (see backfillCalibration).

    Arguments:
        info: [dict] Done flag of the file.
        results_path: [str] Results directory of the file.
        platepar_mtime: [float] Version of the installed platepar.
        replaced_mtimes: [list] Versions of the platepars it replaced.
        result_name: [str] Name of the file with the judgement (see AutoCalibration.FILE_RESULT_NAME).

    Return:
        [bool] True if the file is calibrated with the installed platepar.
    """

    # Without a platepar
    if not info.get('calibrated', True):
        return True

    # Already with the installed version
    version = info.get('platepar_mtime')
    if version == platepar_mtime:
        return False

    # With a replaced version, unless its own judgement passed it
    if version in replaced_mtimes:
        return not _plateparPassed(results_path, result_name)

    # With a platepar which failed its judgement
    return _plateparFailed(results_path, result_name)


def _plateparPassed(results_path, result_name):
    """ Whether the platepar a file was judged with passed the judgement of the automatic calibration.

    Return:
        [bool] True if it passed, False if it failed or the file was not judged.
    """

    try:
        with open(os.path.join(results_path, result_name)) as f:
            return (json.load(f).get('current') or {}).get('ok') is True

    except (OSError, ValueError):
        return False


def _plateparFailed(results_path, result_name):
    """ Whether the platepar a file was judged with failed the judgement of the automatic calibration.

    Arguments:
        results_path: [str] Results directory of the file.
        result_name: [str] Name of the file with the judgement (see AutoCalibration.FILE_RESULT_NAME).

    Return:
        [bool] True if the platepar failed, False if it passed or the file was not judged.
    """

    try:
        with open(os.path.join(results_path, result_name)) as f:
            return (json.load(f).get('current') or {}).get('ok') is False

    except (OSError, ValueError):
        return False



def continuousInput(file_path, input_dir, img_handle, config, chunk_frames, output_dir, proc_log=None):
    """ The input of the matched filter with continuous detection: the file preceded by the last chunks of frames of
        the previous file of the recording, if that file follows it without missing frames, and the pending join of
        their results left for the monitor (see RMS.ContinuousResults).

    Arguments:
        file_path: [str] The input file.
        input_dir: [str] The input directory of the monitor (the IDs of the files are relative to it).
        img_handle: [FrameInterface] The opened input file.
        config: [Config] Configuration.
        chunk_frames: [int] Number of frames of a chunk.
        output_dir: [str] Output directory of the camera.

    Keyword arguments:
        proc_log: [Logger] Logger, the module logger if None.

    Return:
        (chain, pending): [tuple] The chained input (RMS.Formats.FrameChain.ChainedInput) and the pending join, or
            (None, None) without a contiguous previous file.
    """

    from RMS.ContinuousResults import previousInput, writePending
    from RMS.Formats.FrameChain import ChainedInput, contiguous

    proc_log = log if proc_log is None else proc_log

    # The previous file of the recording, opened like this one
    prev_path = previousInput(file_path, root=input_dir)
    if prev_path is None:
        proc_log.info("Continuous detection: no previous file")
        return None, None

    # Opened with a copy of the config, as opening an input can change it (e.g. the size and the frame rate of FITS
    #   frames)
    try:
        prev = detectInputType(prev_path, copy.deepcopy(config), detection=True, preload_video=True,
                               chunk_frames=chunk_frames)
    except Exception as e:
        proc_log.warning("Continuous detection: the previous file {} can't be read: {}".format(prev_path, repr(e)))
        return None, None

    # Only a previous file which this one follows without missing frames is put in front
    if (prev is None) or (not contiguous(prev, img_handle)):
        proc_log.info("Continuous detection: {} doesn't follow {} without missing frames".format(
            os.path.basename(file_path), os.path.basename(prev_path)))
        return None, None

    # The last monitor_continuous_chunks chunks of the previous file in front of this one
    chain = ChainedInput(prev, config.monitor_continuous_chunks*chunk_frames, img_handle)

    # The results directories of both files (as the monitor names them), and the boundary between them
    def unixTime(dt):
        """ Unix time of a datetime (UTC if it has no time zone). """
        return dt.replace(tzinfo=datetime.timezone.utc).timestamp() if dt.tzinfo is None else dt.timestamp()

    prev_uid = uniqueId(os.path.relpath(prev_path, input_dir))
    cur_uid = uniqueId(os.path.relpath(file_path, input_dir))
    boundary = unixTime(chain.currentFrameTime(frame_no=chain.n_context, dt_obj=True))
    pending = {
        'results_dir': resultsDirPath(output_dir, img_handle.beginning_datetime, cur_uid),
        'previous_results_dir': resultsDirPath(output_dir, prev.beginning_datetime, prev_uid),
        'previous_input': os.path.abspath(prev_path),
        'boundary_unix': boundary,
        'context_s': boundary - unixTime(chain.currentFrameTime(frame_no=0, dt_obj=True)),
    }

    # The image pairs of the chunks of the previous file (when they are saved), for the detections moved to it
    if config.monitor_save_images and (prev.input_type != 'ff'):
        pending['previous_chunk_images'] = [[unixTime(t), name] for t, name in savedChunkImages(config, prev)]

    # Left for the monitor, which joins the results once both files are processed
    writePending(output_dir, cur_uid, pending)
    proc_log.info("Continuous detection: the last {:d} frames of {} are searched in front".format(chain.n_context,
                                                                                               prev_path))

    return chain, pending


def recalibrateJoined(results_dir, config, proc_log=None):
    """ Recalibrate results changed by a join of the continuous detection, with new ECSV files of their detections.

    Arguments:
        results_dir: [str] Results directory.
        config: [Config] Configuration.

    Keyword arguments:
        proc_log: [Logger] Logger, the module logger if None.
    """

    # The ECSV files of the detections before the join are written again by the recalibration
    for path in glob.glob(os.path.join(results_dir, '*.ecsv')):
        os.remove(path)
    if os.path.isfile(os.path.join(results_dir, config.platepar_name)):
        recalibrateResults(results_dir, config, proc_log=proc_log)


def processFile(file_path, config_path, platepar_path, output_dir, chunk_frames,
                flat_path=None, dark_path=None, unique_id=None, start_time=None, mask_path=None, input_dir=None):
    """ Process a single file through the detection and recalibration pipeline.

    Arguments:
        file_path: [str] Path to the input file.
        config_path: [str] Path to the config file.
        platepar_path: [str] Path to the platepar file. With the automatic calibration it may not exist (or be
            None), the file is then processed without the astrometry and the photometry.
        output_dir: [str] Path to the output directory.
        chunk_frames: [int] Number of frames per chunk for star extraction.

    Keyword arguments:
        flat_path: [str] Path to a flat field file. None by default.
        dark_path: [str] Path to a dark frame file. None by default.
        unique_id: [str] Safely flattened string to uniquely identify output directories. None by default.
        start_time: [datetime] If given, files whose recording begins before this time (UTC) are skipped
            and the process exits with SKIP_EXIT_CODE. None by default. Files of nights past the cutoff
            (monitor_night_cutoff_hours) are skipped the same way, and marked as done.
        mask_path: [str] Path to the mask. None by default, in which case the mask named in the config next
            to the file or next to the config file is used, if there is one.
        input_dir: [str] The input directory of the monitor, which the IDs of the files are relative to. With
            monitor_continuous, the previous file of the recording is found and its results named with it. None by
            default (no continuous detection).

    Return:
        [bool] True if processing succeeded, False otherwise.
    """

    file_name = os.path.basename(file_path)
    # If a unique_id is provided, use it for the output folder structure. Otherwise use file_base.
    file_base = unique_id if unique_id else os.path.splitext(file_name)[0]

    # Size and time of the file, to check at the end that it was not written meanwhile
    file_state = fileState(file_path)

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

    # A given mask is used wherever the file is (the mask is looked up by its name, and an absolute path
    #   stays absolute when joined to a directory)
    if mask_path is not None:
        config.mask_file = os.path.abspath(mask_path)

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

            # Mark the file as done, so it is not processed again after a restart. A file which already has
            #   results (processed again with --force or --retry_failed) is processed, so its results are kept
            results_dir = resultsDirPath(output_dir, beginning_datetime, file_base)
            if (RmsDateTime.utcnow() > cutoff) and (not hasResults(results_dir)):
                print("Skipping {}: the processing of night {} was cut off at {} UTC".format(file_name,
                    night_name, cutoff))
                os.makedirs(results_dir, exist_ok=True)
                writeDoneFlag(results_dir, {'night': night_name, 'skipped': True,
                                            'input_file': os.path.abspath(file_path)})
                sys.exit(SKIP_EXIT_CODE)

    # Use the module logger until the per-file logger is initialized, so errors before that are logged
    proc_log = log

    # The file which is read, a local copy of it with monitor_staging_dir
    processing_path = file_path

    try:

        processing_path = stageInputFile(file_path, config.monitor_staging_dir)

        # Open the file as an image handle first to get the timestamp
        img_handle = detectInputType(
            processing_path, config, detection=True, preload_video=True, chunk_frames=chunk_frames
        )

        if img_handle is None:
            raise IOError("Could not open file: {}".format(file_path))

        # Create the results directory, sorted by the date of the first frame
        results_dir = resultsDirPath(output_dir, img_handle.beginning_datetime, file_base)
        os.makedirs(results_dir, exist_ok=True)

        # With continuous detection, a file processed again (--force) has no done flag and no record of an earlier
        #   join until it is done: the monitor joins the results of two files once both have a done flag, and
        #   would otherwise join the old results while they are being written again
        if config.monitor_continuous:
            for name in ('done.flag', JOIN_RECORD):
                if os.path.isfile(os.path.join(results_dir, name)):
                    os.remove(os.path.join(results_dir, name))

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
        if processing_path != file_path:
            proc_log.info("Processing the local copy: {}".format(processing_path))

        # Use the newest of the given platepar, the best platepar of the latest reported night and the
        #   automatically fitted platepar
        given_platepar_path = platepar_path
        selected_platepar_path = selectPlatepar(output_dir, config, platepar_path)
        if selected_platepar_path is None:
            proc_log.info("There is no platepar yet, the file is processed without the astrometry and the "
                          "photometry")
        elif selected_platepar_path != platepar_path:
            proc_log.info("Using the platepar: {}".format(selected_platepar_path))
        platepar_path = selected_platepar_path

        # Copy the platepar into the results directory so ApplyRecalibrate can find it. Without a platepar, the
        #   platepar of an earlier processing of the file is removed, so the results don't refer to it
        results_platepar_path = os.path.join(results_dir, config.platepar_name)
        if platepar_path is not None:
            shutil.copy2(platepar_path, results_platepar_path)

        else:

            # Without a platepar, the platepar and the recalibrated platepars of an earlier processing are removed
            for stale_path in [results_platepar_path,
                               os.path.join(results_dir, config.platepars_recalibrated_name)]:
                if os.path.isfile(stale_path):
                    os.remove(stale_path)
            results_platepar_path = None

        # The version of the platepar the file is processed with, from its copy (the automatic calibration may
        #   replace the platepar at the same path meanwhile)
        platepar_mtime = plateparTime(results_platepar_path)

        # Copy the config file into the results directory
        results_config_path = os.path.join(results_dir, os.path.basename(config_path))
        if not os.path.exists(results_config_path):
            shutil.copy2(config_path, results_config_path)

        # Load calibration files (mask, dark, flat) from the input directory (not the staging directory)
        calibration_dir = img_handle.dir_path if (processing_path == file_path) \
            else os.path.dirname(os.path.abspath(file_path))
        mask, dark, flat_struct = loadImageCalibration(
            calibration_dir, config, dtype=img_handle.ff.dtype, byteswap=img_handle.byteswap
        )

        # The mask which was found, it is copied to the night directory for the night report
        mask_path = findMaskPath(calibration_dir, config)

        # The given dark and flat have to be applied
        if config.use_dark and (dark is None):
            raise IOError("The dark could not be loaded: {}".format(config.dark_file))

        if config.use_flat and (flat_struct is None):
            raise IOError("The flat could not be loaded: {}".format(config.flat_file))

        # The dark and the flat are silently not applied if their size doesn't match the frames, so a wrong
        #   file fails the processing instead of the data being marked as corrected
        flat_img = getattr(flat_struct, 'flat_img', None)
        for name, img in [('dark', dark), ('flat', flat_img)]:
            if not calibrationMatchesFrames(img, img_handle, config.detection_binning_factor):
                raise ValueError("The {:s} is {:d}x{:d} px, but the frames are {:d}x{:d} px".format(name,
                    img.shape[1], img.shape[0], img_handle.ncols*config.detection_binning_factor,
                    img_handle.nrows*config.detection_binning_factor))

        # Determine the night the file belongs to. The chunk images are saved to the night directory
        night_name, _, _ = nightInfo(config, img_handle.beginning_datetime)
        night_dir = nightDirPath(output_dir, night_name, config)
        os.makedirs(night_dir, exist_ok=True)

        # FF inputs are not saved again, they already are FF files
        image_saver = None
        if config.monitor_save_images and (img_handle.input_type != 'ff'):
            image_saver = ChunkImageSaver(night_dir, config, file_name, dark=dark, flat_struct=flat_struct)

        # The matched filter replaces the normal detection: its detections are the results of the file. It
        #   needs the frames, so FF files always use the normal detection
        mf_only = config.mf_enable and config.mf_replace_detection and (img_handle.input_type != 'ff')

        # The frames are read from the file, or from memory when it is read in segments (see below)
        frame_source = img_handle

        # The input of the matched filter with the frames of the previous file in front (continuous detection), and
        #   the join of the results it leaves for the monitor
        mf_handle = img_handle
        continuous_pending = None

        if mf_only:

            # The matched filter is imported here, in the worker, and not at the top of the module: importing its
            #   kernels checks for a CUDA GPU, which initializes CUDA. Imported at the top, CUDA would be
            #   initialized in the main monitor process, and a CUDA context can't be used in the processes forked
            #   from it (the workers)
            from RMS.MatchedFilterDetection import (DONE_NAME, MATCHED_FILTER_DIR, detectMatchedFilter,
                                                    detectMatchedFilterSegments, saveSummary)
            from RMS.Formats.FrameBuffer import segmentFrames

            # Results of an earlier processing with the separate matched-filter pass would be merged again, and the
            #   summary of an earlier processing is removed, so a processing which doesn't finish leaves none
            shutil.rmtree(os.path.join(results_dir, MATCHED_FILTER_DIR), ignore_errors=True)
            if os.path.isfile(os.path.join(results_dir, DONE_NAME)):
                os.remove(os.path.join(results_dir, DONE_NAME))

            # With mf_buffer_memory, the file is read only once, in segments held in memory, and the stars
            #   (and the image pairs) and the matched filter take the frames from there. A segment without enough
            #   stars is not searched
            segment_frames = segmentFrames(img_handle, config, config.mf_buffer_memory)
            if segment_frames:
                mf_t0 = time.time()
                star_list, meteor_list, mf_detector, frame_source = detectMatchedFilterSegments(
                    img_handle, config, segment_frames, mask=mask, dark=dark, flat_struct=flat_struct,
                    platepar=readPlatepar(results_platepar_path), reference_dir=output_dir,
                    input_name=os.path.abspath(file_path), chunk_callback=image_saver)
                max_stars = max([len(entry[1]) for entry in star_list] + [0])

            # Otherwise the star extraction (and the image pairs) reads the file, then the matched filter
            else:
                star_list = extractStarsFrameInterface(img_handle, config, chunk_frames=chunk_frames,
                    flat_struct=flat_struct, dark=dark, mask=mask, save_calstars=False, chunk_callback=image_saver)

                # Like the normal detection, nothing is detected without enough stars (clouds, twilight). The
                #   number of stars is the largest number in a chunk of frames
                max_stars = max([len(entry[1]) for entry in star_list] + [0])
                mf_t0 = time.time()

                # With continuous detection, the last chunks of the previous file of the recording are searched in
                #   front of the file, so the search continues across the boundary (see RMS.ContinuousResults)
                mf_handle = img_handle
                if config.monitor_continuous and (input_dir is not None):
                    chain, continuous_pending = continuousInput(file_path, input_dir, img_handle, config,
                                                                chunk_frames, output_dir, proc_log)
                    if chain is not None:
                        mf_handle = chain

                if max_stars >= config.ff_min_stars:
                    meteor_list, mf_detector = detectMatchedFilter(mf_handle, config, mask=mask, dark=dark,
                                                                   flat_struct=flat_struct, star_list=star_list,
                                                                   return_detector=True,
                                                                   platepar=readPlatepar(results_platepar_path),
                                                                   reference_dir=output_dir,
                                                                   input_name=os.path.abspath(file_path))

                    # The detections only in the borrowed frames are those of the previous file, which searched them
                    #   itself
                    if mf_handle is not img_handle:
                        n_all = len(meteor_list)
                        meteor_list = [det for det in meteor_list
                                       if np.max(np.asarray(det[2])[:, 0]) >= mf_handle.n_context]
                        proc_log.info("Continuous detection: {:d} detections, {:d} only in the frames of the "
                                      "previous file dropped".format(n_all, n_all - len(meteor_list)))
                else:
                    proc_log.info("Not enough stars for the matched filter: {:d} < {:d}".format(max_stars,
                                                                                              config.ff_min_stars))
                    meteor_list, mf_detector = [], None

            # The summary of the matched filter is saved as its done file in the results directory. The
            #   detections themselves are saved below with the stars, as the results of the normal detection
            saveSummary(results_dir, mf_detector, input_file=os.path.abspath(file_path),
                        detections=len(meteor_list), max_stars=max_stars,
                        processing_time_s=round(time.time() - mf_t0, 1))
            proc_log.info("Matched filter (replacing the normal detection): {:d} detections".format(
                len(meteor_list)))

        else:

            # Run star extraction and meteor detection
            star_list, meteor_list = detectStarsAndMeteorsFrameInterface(
                img_handle, config, flat_struct=flat_struct, dark=dark, mask=mask,
                chunk_frames=chunk_frames, chunk_callback=image_saver
            )

        # Save the images of the frames after the last full chunk
        if image_saver is not None:
            image_saver.saveTrailing(frame_source)
            proc_log.info("Saved {:d} chunk image pairs to: {}".format(len(image_saver.chunk_images),
                image_saver.night_dir))

        # Save results (CALSTARS + FTPdetectinfo) to the results directory. Meteors are assigned to the saved
        #   chunk images
        # With continuous detection, the frames of the detections are those of the chained input (the chunk images
        #   are moved to their frames there)
        chunk_images = image_saver.chunk_images if image_saver is not None else None
        if (mf_handle is not img_handle) and (chunk_images is not None):
            chunk_images = [(c[0] + mf_handle.n_context,) + tuple(c[1:]) for c in chunk_images]
        saveResultsFrameInterface(
            star_list, meteor_list, mf_handle, config,
            chunk_frames=chunk_frames, output_dir=results_dir,
            chunk_images=chunk_images
        )

        proc_log.info("Detection results saved to: {}".format(results_dir))

        # Optional detection of faint moving objects with the matched filter, saved apart from the normal
        #   results. It needs the frames, so FF files are skipped. A failure doesn't fail the file, the normal
        #   results are complete. The results of an earlier processing of the file are not kept (if this pass
        #   fails or is skipped, there are no matched-filter results rather than old ones)
        mf_pass = config.mf_enable and (img_handle.input_type != 'ff') and (not mf_only)
        if mf_pass:

            # Imported only when used (see above), and the old results removed
            from RMS.MatchedFilterDetection import MATCHED_FILTER_DIR
            shutil.rmtree(os.path.join(results_dir, MATCHED_FILTER_DIR), ignore_errors=True)

            # Like the normal detection, it needs enough stars (not in clouds or twilight)
            max_stars = max([len(entry[1]) for entry in (star_list or [])] + [0])
            if max_stars < config.ff_min_stars:
                proc_log.info("Not enough stars for the matched filter: {:d} < {:d}".format(max_stars,
                                                                                          config.ff_min_stars))
                mf_pass = False

        if mf_pass:
            try:
                from RMS.MatchedFilterDetection import (MATCHED_FILTER_DIR, detectMatchedFilter,
                                                        saveMatchedFilterResults, saveSummary)

                # Detect on the same frames and calibration as the normal detection, and use its stars for the
                #   aperture correction of the intensities
                mf_t0 = time.time()
                mf_dir = os.path.join(results_dir, MATCHED_FILTER_DIR)
                mf_detections, mf_detector = detectMatchedFilter(img_handle, config, mask=mask, dark=dark,
                                                                 flat_struct=flat_struct, star_list=star_list,
                                                                 return_detector=True,
                                                                 platepar=readPlatepar(results_platepar_path),
                                                                 reference_dir=output_dir,
                                                                 input_name=os.path.abspath(file_path))
                # Save the FTPdetectinfo and CALSTARS of the matched filter in its subdirectory, assigned to the
                #   same chunk images as the normal detections, and recalibrate them with the platepar of the
                #   results
                mf_ftp_path = saveMatchedFilterResults(
                    mf_detections, star_list, img_handle, config, mf_dir,
                    platepar_path=results_platepar_path, chunk_frames=chunk_frames,
                    chunk_images=(image_saver.chunk_images if image_saver is not None else None),
                    ecsv_out=config.monitor_save_ecsv)
                # The done file is written last: the night report only merges the results which have one
                saveSummary(mf_dir, mf_detector, input_file=os.path.abspath(file_path),
                            ftpdetectinfo=os.path.basename(mf_ftp_path), detections=len(mf_detections),
                            processing_time_s=round(time.time() - mf_t0, 1))
                proc_log.info("Matched filter: {:d} detections saved to: {}".format(len(mf_detections),
                                                                                    mf_ftp_path))

            except Exception:
                proc_log.error("The matched-filter detection failed:\n" + traceback.format_exc())

        # Release the video handle if applicable
        if hasattr(img_handle, 'cap') and img_handle.cap is not None:
            img_handle.cap.release()

        del img_handle
        gc.collect()

        # Find the FTPdetectinfo file in the results directory
        ftpdetect_files = glob.glob(os.path.join(results_dir, 'FTPdetectinfo_*.txt'))

        if ftpdetect_files and (results_platepar_path is None):
            proc_log.info("No platepar, skipping recalibration for: {}".format(file_name))

        elif ftpdetect_files:
            recalibrateResults(results_dir, config, proc_log=proc_log)
            proc_log.info("Recalibration complete for: {}".format(file_name))

        else:
            proc_log.info("No FTPdetectinfo file found, skipping recalibration for: {}".format(
                file_name))

        # Judge the calibration on the clear chunks of the file, and fit a platepar if there is none, if it fails,
        #   or periodically to check whether a better one can be fitted. A failure doesn't fail the file
        calibrated = results_platepar_path is not None
        if config.monitor_auto_calibration:
            calibrated, platepar_path, platepar_mtime = autoCalibrationStep(
                results_dir, config, output_dir, given_platepar_path, platepar_path, platepar_mtime, calibrated,
                night_name, mask, proc_log)

        # A file which changed while it was processed was still being written (e.g. the recording paused for a
        #   while in the middle of the file), so only a part of it was processed. It gets no done flag, so the
        #   partial results are never taken as final, and it is processed again. A file deleted meanwhile
        #   (e.g. by the recorder) is done
        final_state = fileState(file_path)
        if (final_state is not None) and (final_state != file_state):
            proc_log.warning("{} changed while it was processed (it was still being written), it will be "
                             "processed again".format(file_name))
            sys.exit(CHANGED_EXIT_CODE)

        # Create the done.flag file, with the info needed for the night report
        writeDoneFlag(results_dir, {
            'night': night_name,
            'input_file': os.path.abspath(file_path),
            'mask_path': mask_path,
            'dark_applied': dark is not None,
            'flat_applied': flat_struct is not None,
            'calibrated': calibrated,
            'platepar_mtime': platepar_mtime,
        })

        proc_log.info("Done processing: {} -> {}".format(file_name, results_dir))
        return True

    except Exception as e:

        # A file which is still being written can fail to open or to read (e.g. an MP4 file without its index
        #   yet), which is not a failure of the file
        final_state = fileState(file_path)
        if (final_state is not None) and (final_state != file_state):
            proc_log.warning("{} failed while it was still being written ({}), it will be processed "
                             "again".format(file_name, repr(e)))
            sys.exit(CHANGED_EXIT_CODE)

        proc_log.error(traceback.format_exc())
        proc_log.error("Error processing {}: {}".format(file_name, str(e)))
        return False

    finally:

        # Delete the local copy (a worker which is killed leaves it behind, the monitor deletes it then)
        if processing_path != file_path:
            shutil.rmtree(os.path.dirname(processing_path), ignore_errors=True)


def inputVanished(file_path):
    """ Whether an input was removed: the path is not found while its directory can still be read. Any other error
        (e.g. a network share which is unavailable for a while, a permission or I/O error) is not a removal, as the
        input can come back.

    Arguments:
        file_path: [str] Path of the input file or directory.

    Return:
        [bool] True if the input was removed.
    """

    try:
        os.stat(file_path)
        return False

    # Not found: removed, if the directory it was in can still be read
    except FileNotFoundError:
        try:
            os.listdir(os.path.dirname(os.path.abspath(file_path)))
            return True
        except OSError:
            return False

    # Another error of the lookup, e.g. an unavailable network share
    except OSError:
        return False


def processFileWorker(*args, **kwargs):
    """ Worker process target which runs processFile and turns its result into the process exit code, so
        the monitor can tell failed files apart and retry them. Takes the same arguments as processFile.

    Exit codes:
        0 - processing succeeded
        SKIP_EXIT_CODE - the file begins before the start time, or its night is past the cutoff
        CHANGED_EXIT_CODE - the file changed while it was processed, it has to be processed again
        VANISHED_EXIT_CODE - the file existed when the worker started and disappeared while it was processed
        1 - processing failed
    """

    # The monitor stops the worker with SIGTERM, which ends it right away, and the worker ends if the monitor is
    #   gone
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    exitWithMonitor()

    # Whether the input exists now, to tell a failure from an input which disappeared while it was processed
    file_path = args[0] if args else kwargs.get('file_path')
    existed = (file_path is not None) and os.path.exists(file_path)

    try:
        success = processFile(*args, **kwargs)

    except SystemExit as e:

        if e.code in (SKIP_EXIT_CODE, CHANGED_EXIT_CODE):
            raise

        # Some input types call sys.exit() when they can't open the file (e.g. no time in the file name),
        # which would otherwise end the process with exit code 0
        print("ERROR: Processing exited early with code {}".format(e.code))
        success = False

    except Exception:

        # An error before processFile handles its own errors (e.g. reading the time of a file which is being
        #   deleted) is a failure too, with its traceback
        traceback.print_exc()
        success = False

    # A failed input which existed at the start and is gone now (e.g. deleted or moved away while it was read)
    #   is not processed again
    if (not success) and existed and inputVanished(file_path):
        print("WARNING: {} disappeared while it was processed".format(file_path))
        sys.exit(VANISHED_EXIT_CODE)

    sys.exit(0 if success else 1)


def autoCalibrationStep(results_dir, config, output_dir, given_platepar_path, platepar_path, platepar_mtime,
                        calibrated, night_name, mask, proc_log):
    """ The automatic calibration of a processed file (see RMS.Astrometry.AutoCalibration.autoCalibrateFile). If a
        platepar was installed meanwhile (by this or another worker, also a new version at the same path) and the
        file was processed without a platepar, its platepar fails, or the new one was fitted automatically, the file
        is recalibrated with the new platepar. After a new platepar is installed, the files of the night which were
        processed without a platepar or with a failing one are calibrated too.

    Arguments:
        results_dir: [str] Results directory of the file.
        config: [Config] Configuration.
        output_dir: [str] Output directory of the monitor.
        given_platepar_path: [str] Platepar given to the monitor (it may not exist).
        platepar_path: [str] Platepar the file was processed with, None if there was none.
        platepar_mtime: [float] Modification time of the version of the platepar the file was processed with (see
            plateparTime), None if there was none.
        calibrated: [bool] Whether the file was calibrated.
        night_name: [str] Name of the night of the file.
        mask: [MaskStructure] Mask of the camera.
        proc_log: [Logger] Logger.

    Return:
        (calibrated, platepar_path, platepar_mtime): [tuple] Whether the file is calibrated now, the platepar it
            is calibrated with and the modification time of its version.
    """

    try:
        from RMS.Astrometry.AutoCalibration import autoCalibrateFile
        from RMS.MonitorNightReport import autoPlateparPath

        # Judge the calibration on the clear chunks of the file, and fit a new platepar if needed
        judgement = autoCalibrateFile(results_dir, config, output_dir, platepar_path, mask=mask,
                                      dry_run=config.monitor_auto_calibration_dry_run,
                                      fit_vignetting=config.monitor_auto_calibration_fit_vignetting)

        # The platepar the following files are processed with, and whether it differs from the one this file was
        #   processed with (another platepar, or a new version at the same path)
        new_platepar_path = selectPlatepar(output_dir, config, given_platepar_path)
        new_platepar_mtime = plateparTime(new_platepar_path)
        changed = (new_platepar_path is not None) and ((new_platepar_path != platepar_path)
                                                       or (new_platepar_mtime != platepar_mtime))
        platepar_failed = judgement.get('current', {}).get('ok') is False
        fitted = (new_platepar_path == autoPlateparPath(output_dir, config))
        if changed and ((not calibrated) or platepar_failed or fitted):

            # The detections of the file get the astrometry and the photometry of the new platepar
            proc_log.info("Recalibrating the file with the new platepar: {}".format(new_platepar_path))
            recalibrateResults(results_dir, config, platepar_path=new_platepar_path, proc_log=proc_log,
                               matched_filter=True)
            calibrated = True
            platepar_path = new_platepar_path
            platepar_mtime = new_platepar_mtime

        # The files which were processed without a platepar, with a failing one or with the one which was
        #   replaced
        if judgement.get('installed'):
            backfillCalibration(output_dir, config, night_name, proc_log=proc_log,
                                replaced_mtimes=[judgement.get('replaced_platepar_mtime')])

    except Exception:
        proc_log.error("The automatic calibration failed:\n" + traceback.format_exc())

    return calibrated, platepar_path, platepar_mtime


def plateparTime(platepar_path):
    """ Modification time of a platepar, which identifies the version of a platepar a file was calibrated with.

    Arguments:
        platepar_path: [str] Path of the platepar, None if there is none.

    Return:
        [float] Modification time (Unix time), None if there is no platepar.
    """

    if (platepar_path is None) or (not os.path.isfile(platepar_path)):
        return None

    return os.path.getmtime(platepar_path)


def missingCameraFiles(config_path, platepar_path, dark_path=None, flat_path=None, mask_path=None):
    """ Return the files given for a camera which don't exist. They are checked at the start, as with a wrong
        path every file would fail and be given up. The platepar may be missing if the camera is calibrated
        automatically (monitor_auto_calibration in the config file), until a platepar is fitted.

    Arguments:
        config_path: [str] Path to the config file.
        platepar_path: [str] Path to the platepar file, None if not given.

    Keyword arguments:
        dark_path: [str] Path to the dark frame, None if not given.
        flat_path: [str] Path to the flat field, None if not given.
        mask_path: [str] Path to the mask, None if not given.

    Return:
        [list] Paths which don't exist (and 'platepar (not given)' if the platepar is required but not given).
    """

    # The platepar is not required with the automatic calibration. A config file which can't be read requires it
    #   (the file fails anyway when it is processed)
    platepar_required = True
    if os.path.isfile(config_path):
        try:
            platepar_required = not cr.parse(config_path).monitor_auto_calibration

        except Exception:
            pass

    # The given files which don't exist, the platepar only if it is required
    missing = []
    for name, path in [('config', config_path), ('platepar', platepar_path), ('dark', dark_path),
                       ('flat', flat_path), ('mask', mask_path)]:

        # A platepar which is not required is not checked, and one which is required has to be given
        if name == 'platepar':
            if not platepar_required:
                continue

            # The platepar was not given at all
            if path is None:
                missing.append('platepar (not given)')
                continue

        if (path is not None) and (not os.path.isfile(path)):
            missing.append(path)

    return missing


def findInputMask(input_dir, config, mask_path=None):
    """ Return the mask of the camera: the given one, otherwise the mask file named in the config
        (mask_file) in the input directory. If neither, None is returned, and each file uses the mask next to
        it or next to the config file, if there is one.

    Arguments:
        input_dir: [str] Input directory of the camera.
        config: [Config] Configuration of the camera.

    Keyword arguments:
        mask_path: [str] Path to the mask given to the monitor. None by default.

    Return:
        [str] Absolute path to the mask, or None.
    """

    if mask_path is not None:
        return os.path.abspath(mask_path)

    input_mask_path = os.path.join(input_dir, config.mask_file)
    if os.path.isfile(input_mask_path):
        return os.path.abspath(input_mask_path)

    return None


def enclosingMonitorOutput(output_dir):
    """ Return the output directory of another monitor which contains the given output directory, found by its
        lock file. The scans of that monitor would take the results in this one as its own.

    Arguments:
        output_dir: [str] Output directory.

    Return:
        [str] The enclosing output directory, None if there is none.
    """

    parent = os.path.dirname(os.path.realpath(output_dir))
    while True:
        if os.path.isfile(os.path.join(parent, MONITOR_LOCK_FILE_NAME)):
            return parent

        if os.path.dirname(parent) == parent:
            return None

        parent = os.path.dirname(parent)


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

    given_up = set(uid for uid, fail in failed_files.items()
                   if (fail.get('count', 0) >= MAX_FAILURES) or (fail.get('kills', 0) >= MAX_KILLS))

    return failed_files, given_up


def saveFailedFiles(output_dir, failed_files):
    """ Save the record of the failed files into the output directory (see loadFailedFiles). If it can't be
        written (e.g. the disk is full), the error is logged and the record is saved with the next change.
    """

    failed_files_path = os.path.join(output_dir, FAILED_FILES_NAME)

    try:
        with open(failed_files_path + '.tmp', 'w') as f:
            json.dump(failed_files, f, indent=4, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())

        os.replace(failed_files_path + '.tmp', failed_files_path)

    except OSError as e:
        log.error("Could not save the record of the failed files: {:s}".format(repr(e)))


def recordFailure(output_dir, failed_files, unique_id, exitcode, fail_wait_time, log_prefix='', killed=False):
    """ Record a failure of a file, and save the record.

    Arguments:
        output_dir: [str] Output directory of the monitor.
        failed_files: [dict] Record of the failed files, see loadFailedFiles. It is updated.
        unique_id: [str] Unique ID of the file.
        exitcode: [int] Exit code of the worker.
        fail_wait_time: [float] Seconds before the file is retried.

    Keyword arguments:
        log_prefix: [str] Prefix of the log messages, e.g. the camera ID. Empty by default.
        killed: [bool] The worker was killed from outside the monitor (e.g. by the system when the memory ran
            out), which is counted separately (MAX_KILLS). False by default.

    Return:
        [bool] True if the file failed MAX_FAILURES times (or was killed MAX_KILLS times) and is not retried
            anymore.
    """

    fail = failed_files.setdefault(unique_id, {'count': 0})
    fail['last_fail_time'] = time.time()

    if killed:
        fail['kills'] = fail.get('kills', 0) + 1
    else:
        fail['count'] = fail.get('count', 0) + 1

    saveFailedFiles(output_dir, failed_files)

    if killed:
        message = "{}The worker processing {} was killed (signal {:d}), most likely by the system because " \
            "the memory ran out. Check that --nproc workers fit into the RAM."
        message = message.format(log_prefix, unique_id, -exitcode)
        if fail['kills'] >= MAX_KILLS:
            log.error(message + " Killed {:d} times, giving up. It is not processed again unless the monitor "
                      "is started with --retry_failed.".format(fail['kills']))
            return True

        log.warning(message + " Retrying in {:.1f} mins...".format(fail_wait_time/60.0))
        return False

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

        # The output directory may be gone, e.g. if its disk was unmounted
        try:
            free_gb = availableSpace(self.output_dir)/1024**3
        except OSError:
            free_gb = 0

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


def fileState(file_path):
    """ Size and modification time of a file, None if it doesn't exist. """

    try:
        stat = os.stat(file_path)
        return (stat.st_size, stat.st_mtime)

    except OSError:
        return None


# Configs whose star catalog was loaded by the monitor (see preloadStarCatalog)
_PRELOADED_CATALOGS = set()


def preloadStarCatalog(config_path, platepar_path=None):
    """ Load the star catalog of a config once in the monitor. The workers are forked from the monitor for every
        file, so they find the catalog already decompressed (and share its memory) instead of decompressing it
        for every file (1.6 s for the full GMN catalog). Only with forked workers which use the catalog: with a
        platepar to recalibrate, or with the automatic calibration, which fits one. A failure is left to the
        workers, which report it.

    Arguments:
        config_path: [str] Path to the config file.

    Keyword arguments:
        platepar_path: [str] Path to the platepar the workers are given. None by default.
    """

    # Once per config, and only for forked workers (spawned workers start from nothing and load it themselves)
    if (config_path in _PRELOADED_CATALOGS) or (multiprocessing.get_start_method(allow_none=True) != 'fork'):
        return

    # The catalog of the config, read as the workers read it (the decompressed catalog stays in the memory of
    #   StarCatalog, whatever the epoch and the limiting magnitude of a reading)
    try:
        from RMS.Formats import StarCatalog
        config = cr.parse(config_path)

        # Workers without a platepar and without the automatic calibration don't use the catalog; this is checked
        #   again for the next worker, e.g. once a platepar is there
        has_platepar = (platepar_path is not None) and os.path.isfile(platepar_path)
        if not (has_platepar or getattr(config, 'monitor_auto_calibration', False)):
            return

        _PRELOADED_CATALOGS.add(config_path)
        StarCatalog.readStarCatalog(config.star_catalog_path, config.star_catalog_file,
                                    lim_mag=config.catalog_mag_limit + 1,
                                    mag_band_ratios=config.star_catalog_band_ratios)
    except Exception:
        log.warning('The star catalog could not be loaded in advance:\n' + traceback.format_exc())


def startWorker(file_path, args, kwargs):
    """ Start a worker process which processes the given file (processFileWorker).

    Arguments:
        file_path: [str] Path of the input file.
        args: [tuple] Positional arguments of processFile.
        kwargs: [dict] Keyword arguments of processFile.

    Return:
        [multiprocessing.Process] The started worker, with the time it was started (time.monotonic) in
            start_time. None if the process could not be started (e.g. out of memory), the file is then tried
            again later.
    """

    # The star catalog loaded before the fork, for this worker and the next ones (args[1] is the config path,
    #   args[2] the platepar path)
    preloadStarCatalog(args[1], args[2])

    proc = multiprocessing.Process(target=processFileWorker, args=args, kwargs=kwargs)

    try:
        proc.start()

    except OSError as e:
        log.error("Could not start a worker for {:s}: {:s}".format(file_path, repr(e)))
        return None

    proc.start_time = time.monotonic()

    return proc


def killedFromOutside(proc):
    """ Check if a finished worker was killed by a signal which didn't come from the monitor (e.g. the system
        killed it when the memory ran out).
    """

    return (proc.exitcode is not None) and (proc.exitcode < 0) and (not getattr(proc, 'stopped_by_monitor',
                                                                                  False))


def stopStuckWorker(proc, unique_id, worker_timeout):
    """ Stop a worker process which runs longer than the time limit, e.g. one stuck in the video decoder. The
        stopped worker has a negative exit code, so its file is handled as failed and retried.

    Arguments:
        proc: [multiprocessing.Process] Worker process started by startWorker.
        unique_id: [str] ID of the processed file.
        worker_timeout: [float] Time limit in seconds, no limit if 0.
    """

    if (worker_timeout <= 0) or (time.monotonic() - proc.start_time < worker_timeout) \
            or (not proc.is_alive()):
        return

    log.error("Processing {} did not finish in {:.1f} min, stopping the worker...".format(unique_id,
        worker_timeout/60))

    # A worker stopped by the monitor counts as a failure of its file, unlike one killed from outside
    proc.stopped_by_monitor = True
    stopWorker(proc)


def stopWorker(proc):
    """ Stop a worker process, killing it if it doesn't end after SIGTERM. """

    stopProcess(proc, "Worker")


def recordFinished(status, unique_id, proc, file_path, file_type):
    """ Record a finished worker in the status of the monitor (see MonitorStatus.fileFinished): its outcome, how
        long it ran, and its input with its size.

    Arguments:
        status: [MonitorStatus] Status of the camera of the worker.
        unique_id: [str] ID of the processed file.
        proc: [multiprocessing.Process] The finished worker.
        file_path: [str] Path of its input, None if not known.
        file_type: [str] File type identifier.
    """

    # The outcome from the exit code of the worker (see processFileWorker)
    outcome = {0: 'ok', SKIP_EXIT_CODE: 'skipped', CHANGED_EXIT_CODE: 'changed',
               VANISHED_EXIT_CODE: 'skipped'}.get(proc.exitcode, 'failed')

    # Record it with how long it ran (on the monotonic clock) and its input
    status.fileFinished(unique_id, outcome, time.monotonic() - proc.start_time,
                        input_name=os.path.basename(file_path) if file_path else None,
                        size_bytes=inputSize(file_path, file_type) if file_path else None)


def statusState(workers, queued, settling, failed_files, space_guard, reporter, input_readable, workers_busy,
                stopped=False):
    """ The state of the monitor for the status of a camera (see MonitorStatus.build).

    Arguments:
        workers: [dict] {unique_id: (file_path, process)} of the workers of the camera.
        queued: [list] Files waiting to be processed, as (file_path, mtime, ...).
        settling: [dict] Files which are still being written, with their last size and change (the stability
            tracker of the monitor).
        failed_files: [dict] The failed files of the camera.
        space_guard: [FreeSpaceGuard] Guard of the free space of the camera.
        reporter: [NightReporter] Night reporter of the camera.
        input_readable: [bool] The input directory could be read at the last scan, None if not known.
        workers_busy: [int] Busy workers of the whole monitor.

    Keyword arguments:
        stopped: [bool] The monitor has stopped. False by default.

    Return:
        [dict] The state.
    """

    # The wall time when every worker started (they keep the monotonic time)
    now_wall, now_mono = RmsDateTime.utcnow(), time.monotonic()
    in_progress = [(uid, file_path, now_wall - datetime.timedelta(seconds=now_mono - proc.start_time))
                   for uid, (file_path, proc) in sorted(workers.items())]

    # The state, with the waiting files as (path, time of the last change)
    return {'stopped': stopped, 'input_readable': input_readable, 'paused_for_space': space_guard.paused,
            'workers_busy': workers_busy, 'in_progress': in_progress,
            'queued': [(entry[0], entry[1]) for entry in queued], 'settling': [(path, trk['mtime']) for path, trk in settling.items()],
            'failed_files': failed_files, 'max_failures': MAX_FAILURES, 'max_kills': MAX_KILLS,
            'reporter': reporter}


def monitorDirectory(input_dir, file_type, config_path, platepar_path, output_dir, nproc=2,
                     chunk_frames=128, poll_interval=2, force=False, recursive=False, flat_path=None,
                     dark_path=None, fail_wait_time=300, start_time=None, report_mode=None,
                     worker_timeout=WORKER_TIMEOUT, retry_failed=False, mask_path=None):
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
        mask_path: [str] Path to the mask. None by default, in which case the mask named in the config is
            looked up in the input directory, then next to each file and next to the config file.
    """

    log.info("Monitoring directory: {}".format(input_dir))
    log.info("File type: {}".format(file_type))
    log.info("Output directory: {}".format(output_dir))
    log.info("Parallel processes: {:d}".format(nproc))
    if start_time is not None:
        log.info("Only processing files beginning at or after: {} UTC".format(start_time))

    missing = missingCameraFiles(config_path, platepar_path, dark_path, flat_path, mask_path)
    if missing:
        log.error("The given files don't exist: {}".format(", ".join(missing)))
        sys.exit(1)

    if (platepar_path is None) or (not os.path.isfile(platepar_path)):
        log.info("No platepar is given, the camera is calibrated automatically from the stars")

    # Output directories of monitors can't be nested
    enclosing = enclosingMonitorOutput(output_dir)
    if enclosing is not None:
        log.error("The output directory {} is inside the output directory of another monitor, {} (it has a "
                  "{} file). Use a separate directory, or delete the lock file if {} is no longer used by a "
                  "monitor.".format(output_dir, enclosing, MONITOR_LOCK_FILE_NAME, enclosing))
        sys.exit(1)

    # Only one monitor can work on the output directory
    output_lock = lockOutputDir(output_dir)
    if output_lock is None:
        log.error("Another monitor (PID {:s}) is already working on {:s}, exiting.".format(
            lockOwner(output_dir), output_dir))
        sys.exit(1)

    # Track files that have been processed or are being processed
    processed_files = set()

    # With --force, the results of the files processed before this start are old: the joins of the continuous
    #   detection wait for the new ones
    joins_done_after = time.time() if force else None

    # Failed files, also from previous runs. Files which failed too often are not processed again
    failed_files, given_up = loadFailedFiles(output_dir, retry_failed=(retry_failed or force))
    if given_up:
        log.info("Skipping {:d} file(s) which failed in previous runs (use --retry_failed to process them "
                 "again): {}".format(len(given_up), ", ".join(sorted(given_up))))
        processed_files |= given_up

    # The reporter (and its upload manager) is created in the try block, so it's stopped if the startup
    #   fails or is interrupted, as the monitor would otherwise not exit
    reporter = None

    # Active worker processes: {unique_id: Process}, and their input files
    active_workers = {}
    worker_paths = {}

    # The live status of the monitor (see below), the files waiting to be processed and being written, and
    #   whether the input could be read at the last scan
    status = None
    stable_files = []
    stability_tracker = {}
    input_readable = None

    def monitorState(stopped=False):
        """ The state of the monitor for its status. """

        workers = {uid: (worker_paths.get(uid), proc) for uid, proc in active_workers.items()}
        queued = [entry for entry in stable_files if entry[2] not in active_workers]

        return statusState(workers, queued, stability_tracker, failed_files, space_guard, reporter, input_readable,
                           len(active_workers), stopped=stopped)

    try:

        # Schedules the night reports
        reporter = NightReporter(output_dir, config_path, report_mode=report_mode,
                                 fail_wait_time=fail_wait_time, input_dir=input_dir,
                                 input_ext=inputExtension(file_type), recursive=recursive)
        log.info("Night report mode: {:s}".format(reporter.report_mode))

        # Pauses the processing while the output disk is full
        space_guard = FreeSpaceGuard(output_dir, reporter)

        # Keep the live status of the monitor in the output directory, for external monitoring
        if reporter.config.monitor_status:
            status = MonitorStatus(reporter.config.stationID, output_dir, input_dir, reporter.config,
                                   lambda: scanInputs(input_dir, output_dir, reporter.config, file_type, recursive),
                                   nproc, input_sizer=lambda path: inputSize(path, file_type),
                                   platepar_path=platepar_path)

        # Delete the local copies of input files left behind by workers which were killed
        removeStagedFiles(reporter.config.monitor_staging_dir)

        mask_path = findInputMask(input_dir, reporter.config, mask_path)
        if mask_path is not None:
            log.info("Using mask: {}".format(mask_path))

        # Scan the output directory for previously completed results (done.flag)
        if not force:
            for root, dirs, files in walkInput(output_dir, output_dir, reporter.config):
                if 'done.flag' in files:
                    # The parent dir name is the file base name
                    completed_base = os.path.basename(root)
                    processed_files.add(completed_base)

            if processed_files:
                log.info("Found {:d} previously processed file(s), skipping.".format(
                    len(processed_files)))

        stability_tracker = {}
        stable_wait_time = 5
        stable_max_age = 30

        # Flag to avoid repeating the "waiting for data" message
        waiting_for_data = False

        while True:

            # Clean up finished workers
            finished = []
            for uid, proc in active_workers.items():
                stopStuckWorker(proc, uid, worker_timeout)
                if not proc.is_alive():
                    proc.join()

                    # Delete the local copy of a worker which was killed
                    removeStagedFiles(reporter.config.monitor_staging_dir, pid=proc.pid)

                    # Record the file in the status
                    if status is not None:
                        recordFinished(status, uid, proc, worker_paths.get(uid), file_type)

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
                    elif proc.exitcode == CHANGED_EXIT_CODE:
                        log.warning("{} was still being written while it was processed, processing it "
                                    "again".format(uid))

                    # The input disappeared while it was processed (deleted or moved away), it is not retried
                    elif proc.exitcode == VANISHED_EXIT_CODE:
                        log.warning("{} disappeared while it was processed, skipped".format(uid))
                        processed_files.add(uid)
                    elif recordFailure(output_dir, failed_files, uid, proc.exitcode, fail_wait_time,
                                       killed=killedFromOutside(proc)):
                        processed_files.add(uid)

                    finished.append(uid)

            for uid in finished:
                del active_workers[uid]
                worker_paths.pop(uid, None)

            # The joins of the continuous detection whose files are both processed
            if reporter.config.monitor_continuous:
                if processPendingJoins(output_dir, reporter.config,
                                       recalibrate=lambda d: recalibrateJoined(d, reporter.config),
                                       done_after=joins_done_after):
                    reporter.resultsChanged()

            # 1. Get ALL paths and their metadata first
            candidate_paths = []
            try:
                if recursive:
                    for root, dirs, files in walkInput(input_dir, output_dir, reporter.config):
                        if file_type == 'fitsdirs':
                            for d in dirs:
                                candidate_paths.append(os.path.join(root, d))
                        else:
                            for f in files:
                                if not isHidden(f):
                                    candidate_paths.append(os.path.join(root, f))
                else:
                    candidate_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                                       if not isHidden(f)]
                logReadable(input_dir, True)
                input_readable = True
            except OSError:
                logReadable(input_dir, False)
                input_readable = False

            # Forget the files which disappeared while waiting for them to be written
            stability_tracker = {path: trk for path, trk in stability_tracker.items()
                                 if path in candidate_paths}

            # 2. Identify which paths are "Stable"
            stable_files = []
            for file_path in candidate_paths:
                file_rel_path = os.path.relpath(file_path, input_dir)

                # Skip the paths which are not inputs of the requested type
                if not isInputPath(file_path, file_type):
                    continue

                unique_id = uniqueId(file_rel_path)

                # Skip already processed or currently processing files
                if (unique_id in processed_files) or (unique_id in active_workers):
                    continue

                # Skip files that failed recently (wait fail_wait_time)
                if unique_id in failed_files:
                    if (time.time() - failed_files[unique_id]['last_fail_time']) < fail_wait_time:
                        continue

                # Non-blocking stability check (skip the file if it disappeared or is momentarily locked)
                input_state = inputState(file_path, file_type)
                if input_state is None:
                    continue
                curr_mtime, curr_size = input_state

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

                # Two files with the same ID (e.g. the same name with another extension) are never processed
                #   at the same time
                if unique_id in active_workers:
                    continue

                if file_path in stability_tracker:
                    del stability_tracker[file_path]

                log.info("Putting file {} on the processing queue...".format(file_rel_path))

                proc = startWorker(file_path,
                    (file_path, config_path, platepar_path, output_dir, chunk_frames),
                    {'flat_path': flat_path, 'dark_path': dark_path, 'unique_id': unique_id,
                     'start_time': start_time, 'mask_path': mask_path, 'input_dir': input_dir})
                if proc is None:
                    break

                active_workers[unique_id] = proc
                worker_paths[unique_id] = file_path
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

            # Update the status (every interval, and right after a file was processed)
            if status is not None:
                status.update(monitorState())

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

            # The stopped workers are not processing any more (their files are processed again after a restart)
            active_workers.clear()

        if reporter is not None:

            # The stopped workers don't delete their local copies
            removeStagedFiles(reporter.config.monitor_staging_dir)

            reporter.stop(timeout=SHUTDOWN_TIMEOUT)

        # The status says the monitor has stopped
        if status is not None:
            try:
                status.update(monitorState(stopped=True), force=True)
            except Exception as e:
                log.warning("The final status could not be written: {}".format(repr(e)))

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
            'platepar_path': (os.path.abspath(cp.get(section, 'platepar')) if cp.has_option(section, 'platepar')
                              else None),
            'flat_path': cp.get(section, 'flat', fallback=None),
            'dark_path': cp.get(section, 'dark', fallback=None),
            'mask_path': cp.get(section, 'mask', fallback=None),
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
        if cam['mask_path']:
            cam['mask_path'] = os.path.abspath(cam['mask_path'])
        
        # Ensure the output directory for this camera exists before we start dumping data there
        os.makedirs(cam['output_dir'], exist_ok=True)
        cameras.append(cam)

    if not cameras:
        print("ERROR: No cameras defined in config.")
        sys.exit(1)

    # The files of every camera have to exist
    for cam in cameras:
        missing = missingCameraFiles(cam['config_path'], cam['platepar_path'], cam['dark_path'],
                                     cam['flat_path'], cam['mask_path'])
        if missing:
            print("ERROR: Camera {}: the given files don't exist: {}".format(cam['id'], ", ".join(missing)))
            sys.exit(1)

        # Without a platepar the camera is calibrated automatically (missingCameraFiles checked that it may be
        #   missing)
        if (cam['platepar_path'] is None) or (not os.path.isfile(cam['platepar_path'])):
            print("Camera {}: no platepar is given, the camera is calibrated automatically from the "
                  "stars".format(cam['id']))

    # Every camera needs its own output directory, and they can't be nested
    output_dirs = [os.path.realpath(cam['output_dir']) for cam in cameras]
    if len(set(output_dirs)) < len(output_dirs):
        print("ERROR: Cameras in the multicam config share an output directory: {}".format(", ".join(
            sorted(set(path for path in output_dirs if output_dirs.count(path) > 1)))))
        sys.exit(1)

    for path in output_dirs:
        for other in output_dirs:
            if (path != other) and other.startswith(path + os.sep):
                print("ERROR: The output directory {} of a camera is inside the output directory {} of "
                      "another camera, use separate directories".format(other, path))
                sys.exit(1)

    for cam in cameras:
        enclosing = enclosingMonitorOutput(cam['output_dir'])
        if enclosing is not None:
            print("ERROR: Camera {}: the output directory {} is inside the output directory of another "
                  "monitor, {} (it has a {} file). Use a separate directory, or delete the lock file if {} "
                  "is no longer used by a monitor.".format(cam['id'], cam['output_dir'], enclosing,
                                                        MONITOR_LOCK_FILE_NAME, enclosing))
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

    # Dictionary to keep track of currently active processing jobs
    # Format: { (camera_id, unique_id) : multiprocessing.Process }
    active_workers = {}

    # The input files of the active jobs, and the status of every camera (see below)
    worker_paths = {}
    statuses = {}

    # The reporters (and their upload managers) are created in the try block, so they're stopped if the
    #   startup fails or is interrupted, as the monitor would otherwise not exit
    try:

        for cam in cameras:
            reporters[cam['id']] = NightReporter(cam['output_dir'], cam['config_path'],
                                                 report_mode=report_mode, fail_wait_time=fail_wait_time,
                                                 camera_id=cam['id'], report_lock=report_lock,
                                                 input_dir=cam['input_dir'],
                                                 input_ext=inputExtension(file_type), recursive=recursive)
            log.info("Camera {}: night report mode: {}".format(cam['id'], reporters[cam['id']].report_mode))

        # The mask of every camera
        for cam in cameras:
            cam['mask_path'] = findInputMask(cam['input_dir'], reporters[cam['id']].config, cam['mask_path'])
            if cam['mask_path'] is not None:
                log.info("Camera {}: using mask: {}".format(cam['id'], cam['mask_path']))

        # Delete the local copies of input files left behind by workers which were killed
        for cam in cameras:
            removeStagedFiles(reporters[cam['id']].config.monitor_staging_dir)

        # Pause the processing of a camera while its output disk is full
        space_guards = {cam['id']: FreeSpaceGuard(cam['output_dir'], reporters[cam['id']],
                                                  log_prefix="[{}] ".format(cam['id'])) for cam in cameras}

        # Keep the live status of every camera in its output directory, for external monitoring. A camera without
        #   the status is followed in memory, as its files share the workers with the other cameras
        for cam in cameras:
            cam_config = reporters[cam['id']].config

            # The inputs of the camera as the monitor finds them (bound to this camera)
            def _scanner(cam=cam, cam_config=cam_config):
                return scanInputs(cam['input_dir'], cam['output_dir'], cam_config, file_type, recursive)

            statuses[cam['id']] = MonitorStatus(cam['id'], cam['output_dir'], cam['input_dir'], cam_config,
                                                _scanner, nproc, input_sizer=lambda path: inputSize(path, file_type),
                                                platepar_path=cam['platepar_path'],
                                                enabled=cam_config.monitor_status)
        input_readable = {cam['id']: None for cam in cameras}

        # Initialize state tracking dictionaries. Since files from different cameras might have the same name,
        # we track these metrics per camera using nested dictionaries or sets.
        processed_files = {cam['id']: set() for cam in cameras}

        # With --force, the results of the files processed before this start are old: the joins of the continuous
        #   detection wait for the new ones
        joins_done_after = time.time() if force else None
        cam_output_dirs = {cam['id']: cam['output_dir'] for cam in cameras}

        # Failed files of every camera, also from previous runs. Files which failed too often are not
        #   processed again
        failed_files = {}
        for cam in cameras:
            failed_files[cam['id']], given_up = loadFailedFiles(cam['output_dir'],
                                                                 retry_failed=(retry_failed or force))
            if given_up:
                log.info("Camera {}: skipping {:d} file(s) which failed in previous runs (use --retry_failed "
                         "to process them again): {}".format(cam['id'], len(given_up),
                                                             ", ".join(sorted(given_up))))
                processed_files[cam['id']] |= given_up

        stability_tracker = {cam['id']: {} for cam in cameras}
    
        # Files are considered "stable" (i.e., finished writing to disk) if their size/mtime hasn't 
        # changed for `stable_wait_time` seconds, or if they are older than `stable_max_age` seconds.
        stable_wait_time = 5
        stable_max_age = 30

    
        # Counter for how many processes each camera is currently running (used for load balancing)
        active_count_per_cam = {cam['id']: 0 for cam in cameras}

        # Pre-scan the output directories for files that have already been processed 
        # (indicated by the presence of a 'done.flag'). We do this to avoid reprocessing old files.
        if not force:
            for cam in cameras:
                for root, dirs, files in walkInput(cam['output_dir'], cam['output_dir'],
                                                   reporters[cam['id']].config):
                    if 'done.flag' in files:
                        # The parent directory name is typically the original base filename
                        completed_base = os.path.basename(root)
                        processed_files[cam['id']].add(completed_base)
            
                if processed_files[cam['id']]:
                    log.info("Camera {}: Found {:d} previously processed file(s).".format(
                        cam['id'], len(processed_files[cam['id']])))

        # Keep track of the last camera that received a process slot to facilitate round-robin tiebreaking
        last_assigned_idx = 0

        # The files waiting to be processed by every camera, filled at every poll, and all stable files of every
        #   camera (also the ones held back while its output disk is full), for the status
        stable_per_cam = {}
        waiting_per_cam = {cam['id']: [] for cam in cameras}

        def cameraState(cam_id, stopped=False):
            """ The state of a camera for its status. """

            workers = {uid: (worker_paths.get((c_id, uid)), proc) for (c_id, uid), proc in active_workers.items()
                       if c_id == cam_id}

            queued = [entry for entry in waiting_per_cam[cam_id] if (cam_id, entry[2]) not in active_workers]

            return statusState(workers, queued, stability_tracker[cam_id],
                               failed_files[cam_id], space_guards[cam_id], reporters[cam_id], input_readable[cam_id],
                               len(active_workers), stopped=stopped)

        def updateStatuses(stopped=False):
            """ Write the status of the cameras which are due (all of them when the monitor stops). The queue is
                predicted for all cameras together, as they share the workers. """

            if not (stopped or any(status.enabled and status.due() for status in statuses.values())):
                return

            states = {cam_id: cameraState(cam_id, stopped=stopped) for cam_id in statuses}

            # The files of every camera for the prediction. A camera whose files can't be predicted (e.g. a damaged
            #   history) takes part without files (so the turn of the cameras stays), and isn't predicted
            work, failed = [], set()
            for cam in cameras:
                try:
                    work.append((cam['id'],) + statuses[cam['id']].openWork(states[cam['id']]))
                except Exception as e:
                    log.warning("[{}] The queue could not be predicted: {}".format(cam['id'], repr(e)))
                    work.append((cam['id'], [], []))
                    failed.add(cam['id'])

            # The prediction continues the turn of the cameras where the monitor is
            try:
                queue_empty = simulateQueues(nproc, work, last_idx=last_assigned_idx)
            except Exception as e:
                log.warning("The queues could not be predicted: {}".format(repr(e)))
                queue_empty = {}

            for cam_id, status in statuses.items():
                states[cam_id]['queue_empty_min'] = None if cam_id in failed else queue_empty.get(cam_id)
                status.update(states[cam_id], force=stopped)

        # Main monitoring loop
        while True:

            # 1. Clean up finished workers and log their completion status
            finished = []

            for (cam_id, uid), proc in active_workers.items():
                stopStuckWorker(proc, uid, worker_timeout)
                if not proc.is_alive():
                    proc.join()

                    # Delete the local copy of a worker which was killed
                    removeStagedFiles(reporters[cam_id].config.monitor_staging_dir, pid=proc.pid)

                    # A worker has finished, so we free up a slot for this camera
                    active_count_per_cam[cam_id] -= 1

                    # Record the file in the status of the camera
                    if cam_id in statuses:
                        recordFinished(statuses[cam_id], uid, proc, worker_paths.get((cam_id, uid)), file_type)
                    
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

                    # The file was still being written, it is processed again
                    elif proc.exitcode == CHANGED_EXIT_CODE:
                        log.warning("[{}] {} was still being written while it was processed, processing it "
                                    "again".format(cam_id, uid))

                    # The input disappeared while it was processed (deleted or moved away), it is not retried
                    elif proc.exitcode == VANISHED_EXIT_CODE:
                        log.warning("[{}] {} disappeared while it was processed, skipped".format(cam_id, uid))
                        processed_files[cam_id].add(uid)

                    # Process failed, it is retried once after fail_wait_time and then given up
                    elif recordFailure(cam_output_dirs[cam_id], failed_files[cam_id], uid, proc.exitcode,
                                       fail_wait_time, log_prefix="[{}] ".format(cam_id),
                                       killed=killedFromOutside(proc)):
                        processed_files[cam_id].add(uid)
                                    
                    finished.append((cam_id, uid))

            # Remove finished processes from our tracking dictionary
            for key in finished:
                del active_workers[key]
                worker_paths.pop(key, None)

            # The joins of the continuous detection of every camera whose files are both processed
            for cam in cameras:
                cam_config = reporters[cam['id']].config
                if cam_config.monitor_continuous:
                    if processPendingJoins(cam['output_dir'], cam_config,
                                           recalibrate=lambda d, c=cam_config: recalibrateJoined(d, c),
                                           done_after=joins_done_after):
                        reporters[cam['id']].resultsChanged()

            # 2. Collect all new, stable files ready to be processed for each camera
            stable_per_cam = {}
            for cam in cameras:

                cam_id = cam['id']
                candidate_paths = []
                
                # Fetch all paths from the input directory
                try:
                    if recursive:
                        for root, dirs, files in walkInput(cam['input_dir'], cam['output_dir'],
                                                         reporters[cam['id']].config):
                            if file_type == 'fitsdirs':
                                for d in dirs:
                                    candidate_paths.append(os.path.join(root, d))
                            else:
                                for f in files:
                                    if not isHidden(f):
                                        candidate_paths.append(os.path.join(root, f))
                    else:
                        candidate_paths = [os.path.join(cam['input_dir'], f)
                                           for f in os.listdir(cam['input_dir']) if not isHidden(f)]
                    logReadable(cam['input_dir'], True, log_prefix="[{}] ".format(cam_id))
                    input_readable[cam_id] = True

                except OSError:
                    logReadable(cam['input_dir'], False, log_prefix="[{}] ".format(cam_id))
                    input_readable[cam_id] = False

                    # Its files can't be found now, so none waits or is being written
                    waiting_per_cam[cam_id] = []
                    stability_tracker[cam_id] = {}
                    continue

                # Forget the files which disappeared while waiting for them to be written
                stability_tracker[cam_id] = {path: trk for path, trk in stability_tracker[cam_id].items()
                                             if path in candidate_paths}

                cam_stable = []
                for file_path in candidate_paths:
                    file_rel_path = os.path.relpath(file_path, cam['input_dir'])
                    
                    # Filter out non-target files and directories
                    if not isInputPath(file_path, file_type):
                        continue
                    
                    # Generate a unique ID based on the path name.
                    unique_id = uniqueId(file_rel_path)

                    # Ignore files we've already processed or are currently working on
                    # Workers are tracked per camera, as cameras can have files of the same name
                    if (unique_id in processed_files[cam_id]) or ((cam_id, unique_id) in active_workers):
                        continue
                    
                    # If the file recently failed, wait before retrying it
                    if unique_id in failed_files[cam_id]:
                        if (time.time() - failed_files[cam_id][unique_id]['last_fail_time']) < fail_wait_time:
                            continue

                    # Attempt to read path metadata for stability checking (the file might have been deleted or
                    #   locked, we'll try again next poll)
                    input_state = inputState(file_path, file_type)
                    if input_state is None:
                        continue
                    curr_mtime, curr_size = input_state

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
                waiting_per_cam[cam_id] = list(cam_stable)
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

                # Two files with the same ID (e.g. the same name with another extension) are never processed
                #   at the same time
                if (cam_id, unique_id) in active_workers:
                    continue

                # Stop tracking stability for this file since it's about to be processed
                if file_path in stability_tracker[cam_id]:
                    del stability_tracker[cam_id][file_path]

                log.info("[{}] Putting file {} on queue (active processes for this cam: {})...".format(
                    cam_id, file_rel_path, active_count_per_cam[cam_id]))

                # Spawn the worker process
                proc = startWorker(file_path,
                    (file_path, chosen_cam['config_path'], chosen_cam['platepar_path'],
                     chosen_cam['output_dir'], chunk_frames),
                    {'flat_path': chosen_cam['flat_path'], 'dark_path': chosen_cam['dark_path'],
                     'unique_id': unique_id, 'start_time': chosen_cam['start_time'],
                     'mask_path': chosen_cam['mask_path'], 'input_dir': chosen_cam['input_dir']})
                if proc is None:
                    break

                # Update load balancing metrics and tracking dictionaries
                active_workers[(cam_id, unique_id)] = proc
                worker_paths[(cam_id, unique_id)] = file_path
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

            # Update the status of every camera (every interval, and right after a file was processed). The status
            #   never stops the processing
            try:
                updateStatuses()
            except Exception as e:
                log.warning("The status could not be updated: {}".format(repr(e)))

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        log.info("Multi-camera monitoring stopped by user.")

    finally:

        # Stop the workers right away. Their files have no done flag, so they are processed again after a
        #   restart
        if active_workers:
            log.info("Stopping {:d} worker(s), their files will be processed again after a restart: "
                     "{}".format(len(active_workers), ", ".join("[{}] {}".format(*key)
                                                                for key in sorted(active_workers))))
            for proc in active_workers.values():
                stopWorker(proc)

            # The stopped workers are not processing any more (their files are processed again after a restart)
            active_workers.clear()

        for reporter in reporters.values():

            # The stopped workers don't delete their local copies
            removeStagedFiles(reporter.config.monitor_staging_dir)

            reporter.stop(timeout=SHUTDOWN_TIMEOUT)

        # The status of every camera says the monitor has stopped
        try:
            updateStatuses(stopped=True)
        except Exception as e:
            log.warning("The final status could not be written: {}".format(repr(e)))

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
        [str] Absolute path to the platepar file. With the automatic calibration (monitor_auto_calibration) the
            file may not exist, the camera is then calibrated from the stars.
    """

    if cml_platepar is not None:
        platepar_path = os.path.abspath(cml_platepar)
        if (not os.path.isfile(platepar_path)) and (not config.monitor_auto_calibration):
            print("ERROR: Platepar file not found: {}".format(platepar_path))
            sys.exit(1)
        return platepar_path

    # Look for the default platepar in the input directory
    default_path = os.path.join(os.path.abspath(input_dir), config.platepar_name)
    if os.path.isfile(default_path) or config.monitor_auto_calibration:
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
        help="File type to monitor for. Supported: vid, ff, mkv, mp4, avi, mov, wmv, and fitsdirs "
             "(directories of FITS frames). Any other value will be treated as a file extension. See "
             "Guides/MonitorProcessing.md."
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

    arg_parser.add_argument('--mask', type=str, default=None,
        help="Path to the mask image. If not given, the mask named in the config (mask_file) is used from "
             "the input directory, or next to each file or the config file."
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
        mask_path=cml_args.mask,
        start_time=start_time,
        report_mode=cml_args.report_mode,
        worker_timeout=cml_args.worker_timeout,
        retry_failed=cml_args.retry_failed
    )
