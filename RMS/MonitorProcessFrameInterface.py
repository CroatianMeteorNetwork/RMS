""" Monitor a directory for new files and process them through star extraction, meteor detection,
    and astrometric recalibration.

    Usage:
        python -m RMS.MonitorProcessFrameInterface <file_type> <input_dir> \
            [--output OUTPUT_DIR] [--config CONFIG_PATH] [--platepar PLATEPAR_PATH] \
            [--nproc N] [--chunk_frames N]

    Example:
        python -m RMS.MonitorProcessFrameInterface vid /path/to/input \
            --output /path/to/output --platepar ~/source/Stations/02F/platepar_cmn2010.cal
"""

from __future__ import print_function, division, absolute_import

import argparse
import gc
import glob
import os
import shutil
import sys
import time
import logging
import multiprocessing
import configparser

import RMS.ConfigReader as cr
from RMS.Formats.FrameInterface import detectInputTypeFile, checkIfVideoFile
from RMS.Formats.FFfile import validFFName
from RMS.DetectStarsAndMeteors import (
    detectStarsAndMeteorsFrameInterface,
    saveResultsFrameInterface,
)
from RMS.DetectionTools import loadImageCalibration
from RMS.Astrometry.ApplyRecalibrate import applyRecalibrate
from RMS.Logger import LoggingManager, getLogger


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
}


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

    # Check by extension
    if file_type in FILE_TYPE_MAP:
        extensions = FILE_TYPE_MAP[file_type]
        if extensions is not None:
            return any(file_name.lower().endswith(ext) for ext in extensions)

    # If the type is not in the map, try matching directly as an extension
    return file_name.lower().endswith('.' + file_type)




def processFile(file_path, config_path, platepar_path, output_dir, chunk_frames,
                flat_path=None, dark_path=None, unique_id=None):
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

    Return:
        [bool] True if processing succeeded, False otherwise.
    """
    
    file_name = os.path.basename(file_path)
    # If a unique_id is provided, use it for the output folder structure. Otherwise use file_base.
    file_base = unique_id if unique_id else os.path.splitext(file_name)[0]

    # Load the config file
    config = cr.parse(config_path)

    # Override dark/flat paths in config if explicitly given
    if dark_path is not None:
        config.dark_file = os.path.abspath(dark_path)
        config.use_dark = True
    if flat_path is not None:
        config.flat_file = os.path.abspath(flat_path)
        config.use_flat = True

    try:

        # Open the file as an image handle first to get the timestamp
        img_handle = detectInputTypeFile(
            file_path, config, detection=True, preload_video=True, chunk_frames=chunk_frames
        )

        if img_handle is None:
            print("ERROR: Could not open file: {}".format(file_path))
            return False

        # Get the first frame time to build the date-sorted directory structure
        dt = img_handle.beginning_datetime
        date_path = os.path.join(
            "{:04d}".format(dt.year),
            "{:04d}{:02d}".format(dt.year, dt.month),
            "{:04d}{:02d}{:02d}".format(dt.year, dt.month, dt.day),
        )

        # Create a results directory: output_dir/YYYY/YYYYMM/YYYYMMDD/<file_base>/
        results_dir = os.path.join(output_dir, date_path, file_base)
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

        # Copy the platepar into the results directory so ApplyRecalibrate can find it
        results_platepar_path = os.path.join(results_dir, config.platepar_name)
        if not os.path.exists(results_platepar_path):
            shutil.copy2(platepar_path, results_platepar_path)

        # Copy the config file into the results directory
        results_config_path = os.path.join(results_dir, os.path.basename(config_path))
        if not os.path.exists(results_config_path):
            shutil.copy2(config_path, results_config_path)

        # Load calibration files (mask, dark, flat) from the input directory
        mask, dark, flat_struct = loadImageCalibration(
            img_handle.dir_path, config, dtype=img_handle.ff.dtype, byteswap=img_handle.byteswap
        )

        # Run star extraction and meteor detection
        star_list, meteor_list = detectStarsAndMeteorsFrameInterface(
            img_handle, config, flat_struct=flat_struct, dark=dark, mask=mask,
            chunk_frames=chunk_frames
        )

        # Save results (CALSTARS + FTPdetectinfo) to the results directory
        saveResultsFrameInterface(
            star_list, meteor_list, img_handle, config,
            chunk_frames=chunk_frames, output_dir=results_dir
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

            # Run recalibration with load_all=True
            applyRecalibrate(
                ftpdetectinfo_path, config,
                generate_plot=True,
                load_all=True,
                generate_ufoorbit=False,
            )

            proc_log.info("Recalibration complete for: {}".format(file_name))

        else:
            proc_log.info("No FTPdetectinfo file found, skipping recalibration for: {}".format(
                file_name))

        # Create the done.flag file
        done_flag_path = os.path.join(results_dir, 'done.flag')
        with open(done_flag_path, 'w') as _:
            pass

        proc_log.info("Done processing: {} -> {}".format(file_name, results_dir))
        return True

    except Exception as e:
        proc_log.error("Error processing {}: {}".format(file_name, str(e)))
        import traceback
        proc_log.error(traceback.format_exc())
        return False


def monitorDirectory(input_dir, file_type, config_path, platepar_path, output_dir, nproc=2,
                     chunk_frames=128, poll_interval=2, force=False, recursive=False, flat_path=None,
                     dark_path=None, fail_wait_time=300):
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
    """

    log.info("Monitoring directory: {}".format(input_dir))
    log.info("File type: {}".format(file_type))
    log.info("Output directory: {}".format(output_dir))
    log.info("Parallel processes: {:d}".format(nproc))

    # Track files that have been processed or are being processed
    processed_files = set()
    
    # Track failed files: {unique_id: {'count': int, 'last_fail_time': float}}
    failed_files = {}

    # Scan the output directory for previously completed results (done.flag)
    if not force:
        for root, dirs, files in os.walk(output_dir):
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

    try:
        while True:

            # Clean up finished workers
            finished = []
            for uid, proc in active_workers.items():
                if not proc.is_alive():
                    proc.join()
                    if proc.exitcode == 0:
                        log.info("Successfully processed: {}".format(uid))
                        processed_files.add(uid)
                    else:
                        if uid not in failed_files:
                            failed_files[uid] = {'count': 1, 'last_fail_time': time.time()}
                            log.warning("Processing failed for: {} (exit code {:d}), will retry in {:.1f} mins...".format(
                                uid, proc.exitcode, fail_wait_time/60.0))
                        else:
                            failed_files[uid]['count'] += 1
                            if failed_files[uid]['count'] >= 2:
                                log.error("Processing failed for: {} (exit code {:d}) twice, giving up.".format(
                                    uid, proc.exitcode))
                                processed_files.add(uid)
                            else:
                                failed_files[uid]['last_fail_time'] = time.time()
                                log.warning("Processing failed for: {} (exit code {:d}), will retry in {:.1f} mins...".format(
                                    uid, proc.exitcode, fail_wait_time/60.0))

                    finished.append(uid)

            for uid in finished:
                del active_workers[uid]

            # 1. Get ALL files and their metadata first
            candidate_files = []
            try:
                if recursive:
                    for root, _, files in os.walk(input_dir):
                        for f in files:
                            candidate_files.append(os.path.join(root, f))
                else:
                    candidate_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
            except OSError:
                log.error("Cannot read directory: {}".format(input_dir))
                time.sleep(poll_interval)
                continue

            # 2. Identify which files are "Stable"
            stable_files = []
            for file_path in candidate_files:
                file_name = os.path.basename(file_path)

                # Check if the file matches the requested type
                if not matchesFileType(file_name, file_type):
                    continue

                if os.path.isdir(file_path):
                    continue

                # Generate a unique ID based on the file name. RMS file names are already
                # globally unique by design (they include the camera ID, date, and microsecond timestamp).
                unique_id = os.path.splitext(file_name)[0]

                # Skip already processed or currently processing files
                if unique_id in processed_files or unique_id in active_workers:
                    continue

                # Skip files that failed recently (wait fail_wait_time)
                if unique_id in failed_files:
                    if (time.time() - failed_files[unique_id]['last_fail_time']) < fail_wait_time:
                        continue

                # Non-blocking stability check
                try:
                    curr_mtime = os.path.getmtime(file_path)
                    curr_size = os.path.getsize(file_path)
                except OSError:
                    continue  # File disappeared or momentarily locked, try next pass

                # If file is older than max age, consider it stable immediately
                if (time.time() - curr_mtime) > stable_max_age:
                    stable = True
                else:
                    if file_path not in stability_tracker:
                        log.info("New file detected: {} - waiting for write completion...".format(file_rel_path))
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

            # 4. Start processes for the oldest stable files until nproc is full
            new_files_queued = False
            for file_path, mtime, unique_id, file_rel_path in stable_files:
                if len(active_workers) >= nproc:
                    break

                if file_path in stability_tracker:
                    del stability_tracker[file_path]

                log.info("Putting file {} on the processing queue...".format(file_rel_path))

                proc = multiprocessing.Process(
                    target=processFile,
                    args=(file_path, config_path, platepar_path, output_dir, chunk_frames),
                    kwargs={'flat_path': flat_path, 'dark_path': dark_path, 'unique_id': unique_id}
                )
                proc.start()
                active_workers[unique_id] = proc
                new_files_queued = True

            # If no workers are active and no new files were queued, we're idle
            if not active_workers and not new_files_queued and not waiting_for_data:
                log.info("Waiting for more data...")
                waiting_for_data = True
            elif active_workers or new_files_queued:
                waiting_for_data = False

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        log.info("Monitoring stopped by user.")

    finally:
        # Wait for any remaining workers
        if active_workers:
            log.info("Waiting for {:d} remaining task(s) to finish...".format(len(active_workers)))
            for fname, proc in active_workers.items():
                proc.join(timeout=300)
                if proc.is_alive():
                    log.warning("Worker for {} did not finish in time, terminating.".format(fname))
                    proc.terminate()

        log.info("All workers stopped.")


def monitorMultipleCameras(multicam_ini_path):
    """ Monitor multiple directories for multiple cameras based on an INI config file. 
    
    Arguments:
        multicam_ini_path: Path to the INI config file.
        
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
        
        [CA0001]
        input_dir = /path/to/CA0001/video
        output_dir = /path/to/CA0001/output
        config = /path/to/CA0001/CA0001.config
        platepar = /path/to/CA0001/platepar.cal
        flat = /path/to/CA0001/flat.bmp
        dark = /path/to/CA0001/dark.bmp
        
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

    # Initialize state tracking dictionaries. Since files from different cameras might have the same name,
    # we track these metrics per camera using nested dictionaries or sets.
    processed_files = {cam['id']: set() for cam in cameras}
    failed_files = {cam['id']: {} for cam in cameras}
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
            for root, dirs, files in os.walk(cam['output_dir']):
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
                if not proc.is_alive():
                    proc.join()
                    
                    # A worker has finished, so we free up a slot for this camera
                    active_count_per_cam[cam_id] -= 1
                    
                    if proc.exitcode == 0:

                        # Process exited normally
                        log.info("Successfully processed [{}]: {}".format(cam_id, uid))
                        processed_files[cam_id].add(uid)

                    else:

                        # Process failed. We implement a retry mechanism.
                        cam_fails = failed_files[cam_id]
                        if uid not in cam_fails:
                            # First failure: log warning and set up for a retry
                            cam_fails[uid] = {'count': 1, 'last_fail_time': time.time()}
                            log.warning("Processing failed for [{}] {} (exit code {:d}), retry in {:.1f} mins...".format(
                                cam_id, uid, proc.exitcode, fail_wait_time/60.0))

                        else:

                            cam_fails[uid]['count'] += 1

                            if cam_fails[uid]['count'] >= 2:

                                # Failed twice: give up and mark as "processed" to prevent infinite loops
                                log.error("Processing failed for [{}] {} (exit code {:d}) twice, giving up.".format(
                                    cam_id, uid, proc.exitcode))
                                processed_files[cam_id].add(uid)

                            else:

                                # Subsequent failure tracking (should not occur with current max of 2 retries)
                                cam_fails[uid]['last_fail_time'] = time.time()
                                log.warning("Processing failed for [{}] {} (exit code {:d}), retry in {:.1f} mins...".format(
                                    cam_id, uid, proc.exitcode, fail_wait_time/60.0))
                                    
                    finished.append(uid)

            # Remove finished processes from our tracking dictionary
            for uid in finished:
                del active_workers[uid]

            # 2. Collect all new, stable files ready to be processed for each camera
            stable_per_cam = {}
            for cam in cameras:

                cam_id = cam['id']
                candidate_files = []
                
                # Fetch all files from the input directory
                try:
                    if recursive:
                        for root, _, files in os.walk(cam['input_dir']):
                            for f in files:
                                candidate_files.append(os.path.join(root, f))
                    else:
                        candidate_files = [os.path.join(cam['input_dir'], f) for f in os.listdir(cam['input_dir'])]
                
                except OSError:
                    log.error("Cannot read directory for camera {}: {}".format(cam_id, cam['input_dir']))
                    continue

                cam_stable = []
                for file_path in candidate_files:
                    file_name = os.path.basename(file_path)
                    
                    # Filter out non-target files and directories
                    if not matchesFileType(file_name, file_type):
                        continue
                    if os.path.isdir(file_path):
                        continue
                    
                    # Generate a unique ID based on the file name. RMS file names are already
                    # globally unique by design (they include the camera ID, date, and microsecond timestamp).
                    unique_id = os.path.splitext(file_name)[0]

                    # Ignore files we've already processed or are currently working on
                    if unique_id in processed_files[cam_id] or unique_id in active_workers:
                        continue
                    
                    # If the file recently failed, wait before retrying it
                    if unique_id in failed_files[cam_id]:
                        if (time.time() - failed_files[cam_id][unique_id]['last_fail_time']) < fail_wait_time:
                            continue

                    # Attempt to read file metadata for stability checking
                    try:
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
                            log.info("[{}] New file detected: {} - waiting...".format(cam_id, file_rel_path))
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
                if cam_stable:
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
                    target=processFile,
                    args=(file_path, chosen_cam['config_path'], chosen_cam['platepar_path'], chosen_cam['output_dir'], chunk_frames),
                    kwargs={'flat_path': chosen_cam['flat_path'], 'dark_path': chosen_cam['dark_path'], 'unique_id': unique_id}
                )
                proc.start()
                
                # Update load balancing metrics and tracking dictionaries
                active_workers[unique_id] = (proc, cam_id)
                active_count_per_cam[cam_id] += 1
                new_files_queued = True
                last_assigned_idx = chosen_idx

                # If this camera has no more stable files, remove it from the pool of candidates for this polling cycle
                if len(stable_per_cam[cam_id]) == 0:
                    del stable_per_cam[cam_id]
            
            # Idle wait: If we did nothing this iteration, wait `poll_interval` before scanning the disk again
            if not active_workers and not new_files_queued:
                pass # Optionally log "Waiting for more data..."
                
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        log.info("Multi-camera monitoring stopped by user.")

    finally:

        # Gracefully handle active workers during script termination
        if active_workers:

            log.info("Waiting for {:d} remaining task(s) to finish...".format(len(active_workers)))

            for fname, (proc, cam_id) in active_workers.items():
                
                # Allow processes up to 300 seconds to finish what they were doing
                proc.join(timeout=300)

                if proc.is_alive():

                    # Terminate processes that refuse to close or get stuck
                    log.warning("Worker for [{}] {} did not finish in time, terminating.".format(cam_id, fname))
                    proc.terminate()
                    
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
        help="Path to a flat field image file."
    )

    arg_parser.add_argument('--dark', type=str, default=None,
        help="Path to a dark frame image file."
    )

    arg_parser.add_argument('--multicam', '-m', type=str, default=None,
        help="Path to an INI file containing multiple camera configurations. If provided, other arguments like input_dir are ignored."
    )

    # Parse
    cml_args = arg_parser.parse_args()

    if cml_args.multicam is not None:
        multicam_path = os.path.abspath(cml_args.multicam)
        if not os.path.isfile(multicam_path):
            print("ERROR: Multicam config file does not exist: {}".format(multicam_path))
            sys.exit(1)
        monitorMultipleCameras(multicam_path)
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
        dark_path=cml_args.dark
    )
