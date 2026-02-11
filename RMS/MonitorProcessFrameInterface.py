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


def isFileStable(file_path, stable_time=5, poll_interval=1, max_age=30):
    """ Check if a file is stable (finished being written).

    If the file's modification time is older than max_age seconds, it is considered already stable
    and no waiting is performed. Otherwise, poll every poll_interval seconds until the modification
    time hasn't changed for stable_time seconds, then confirm the size is unchanged.

    Arguments:
        file_path: [str] Path to the file.

    Keyword arguments:
        stable_time: [float] Time in seconds the mtime must be unchanged. Default is 5.
        poll_interval: [float] Time between checks in seconds. Default is 1.
        max_age: [float] If the file's mtime is older than this many seconds, skip waiting. Default is 30.

    Return:
        [bool] True if the file is stable, False if it disappeared.
    """

    try:
        prev_size = os.path.getsize(file_path)
        prev_mtime = os.path.getmtime(file_path)
    except OSError:
        return False

    # If the file hasn't been modified recently, it's already stable
    if (time.time() - prev_mtime) > max_age:
        return True

    stable_since = time.time()

    while (time.time() - stable_since) < stable_time:
        time.sleep(poll_interval)

        try:
            curr_mtime = os.path.getmtime(file_path)
        except OSError:
            return False

        # If the mtime changed, reset the stability timer
        if curr_mtime != prev_mtime:
            prev_mtime = curr_mtime
            stable_since = time.time()

    # Final size confirmation
    try:
        final_size = os.path.getsize(file_path)
    except OSError:
        return False

    if final_size != prev_size and final_size != os.path.getsize(file_path):
        return False

    return True


def processFile(file_path, config_path, platepar_path, output_dir, chunk_frames):
    """ Process a single file through the detection and recalibration pipeline.

    Arguments:
        file_path: [str] Path to the input file.
        config_path: [str] Path to the config file.
        platepar_path: [str] Path to the platepar file.
        output_dir: [str] Path to the output directory.
        chunk_frames: [int] Number of frames per chunk for star extraction.

    Return:
        [bool] True if processing succeeded, False otherwise.
    """
    
    file_name = os.path.basename(file_path)
    file_base = os.path.splitext(file_name)[0]

    # Load the config file
    config = cr.parse(config_path)

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
        log_manager = LoggingManager()
        log_manager.initLogging(config, 'monitor_', safedir=results_dir)
        proc_log = getLogger("logger")

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
                     chunk_frames=128, poll_interval=2, force=False):
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
    """

    log.info("Monitoring directory: {}".format(input_dir))
    log.info("File type: {}".format(file_type))
    log.info("Output directory: {}".format(output_dir))
    log.info("Parallel processes: {:d}".format(nproc))

    # Track files that have been processed or are being processed
    processed_files = set()

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

    # Active worker processes: {file_name: Process}
    active_workers = {}

    # Flag to avoid repeating the "waiting for data" message
    waiting_for_data = False

    try:
        while True:

            # Clean up finished workers
            finished = []
            for fname, proc in active_workers.items():
                if not proc.is_alive():
                    proc.join()
                    if proc.exitcode == 0:
                        log.info("Successfully processed: {}".format(fname))
                    else:
                        log.warning("Processing failed for: {} (exit code {:d})".format(
                            fname, proc.exitcode))
                    finished.append(fname)

            for fname in finished:
                del active_workers[fname]

            # Scan the directory for matching files
            try:
                all_files = sorted(os.listdir(input_dir))
            except OSError:
                log.error("Cannot read directory: {}".format(input_dir))
                time.sleep(poll_interval)
                continue

            for file_name in all_files:

                # Skip already processed or in-progress files (match by base name)
                file_base = os.path.splitext(file_name)[0]
                if file_name in processed_files or file_base in processed_files:
                    continue

                # Check if the file matches the requested type
                if not matchesFileType(file_name, file_type):
                    continue

                file_path = os.path.join(input_dir, file_name)

                # Skip directories
                if os.path.isdir(file_path):
                    continue

                # Don't exceed the worker limit
                if len(active_workers) >= nproc:
                    break

                # Wait for the file to be stable (finished writing)
                log.info("New file detected: {} - waiting for write completion...".format(file_name))

                if not isFileStable(file_path):
                    log.warning("File disappeared while waiting: {}".format(file_name))
                    continue

                # Mark as processed and start a worker process
                processed_files.add(file_name)
                log.info("Putting file {} on the processing queue...".format(file_name))

                proc = multiprocessing.Process(
                    target=processFile,
                    args=(file_path, config_path, platepar_path, output_dir, chunk_frames)
                )
                proc.start()
                active_workers[file_name] = proc

            # If no workers are active and no new files were queued, we're idle
            if not active_workers and not waiting_for_data:
                log.info("Waiting for more data...")
                waiting_for_data = True
            elif active_workers:
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

    arg_parser.add_argument('file_type', type=str,
        help="File type to monitor for. Supported: vid, ff, mkv, mp4, avi, mov, wmv. "
             "Any other value will be treated as a file extension."
    )

    arg_parser.add_argument('input_dir', type=str,
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

    # Parse
    cml_args = arg_parser.parse_args()


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

    # Initialize the logger
    log_manager = LoggingManager()
    log_manager.initLogging(config, 'monitor_', safedir=output_dir)
    log = getLogger("logger")

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
        force=cml_args.force
    )
