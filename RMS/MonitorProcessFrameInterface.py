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
                     dark_path=None):
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

    # Active worker processes: {unique_id: Process}
    active_workers = {}

    failed_files = set()
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
                        log.warning("Processing failed for: {} (exit code {:d})".format(
                            uid, proc.exitcode))
                        failed_files.add(uid)
                    finished.append(uid)

            for uid in finished:
                del active_workers[uid]

            # Scan the directory for matching files
            try:
                if recursive:
                    all_files = []
                    for root, _, files in os.walk(input_dir):
                        for f in files:
                            all_files.append(os.path.relpath(os.path.join(root, f), input_dir))
                    all_files = sorted(all_files)
                else:
                    all_files = sorted(os.listdir(input_dir))
            except OSError:
                log.error("Cannot read directory: {}".format(input_dir))
                time.sleep(poll_interval)
                continue

            for file_name in all_files:

                # Ensure uniqueness for recursive files by including subfolder path
                # Replace slashes to make a flat unique identifier
                file_rel_path = os.path.relpath(os.path.join(input_dir, file_name), input_dir)
                unique_id = os.path.splitext(file_rel_path)[0].replace(os.path.sep, '_')

                # Skip already processed, failed, or currently processing files
                if unique_id in processed_files or unique_id in failed_files or unique_id in active_workers:
                    continue

                base_name = os.path.basename(file_name)
                # Check if the file matches the requested type
                if not matchesFileType(base_name, file_type):
                    continue

                file_path = os.path.join(input_dir, file_name)

                # Skip directories
                if os.path.isdir(file_path):
                    continue

                # Don't exceed the worker limit
                if len(active_workers) >= nproc:
                    break

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

                if not stable:
                    continue

                log.info("Putting file {} on the processing queue...".format(file_rel_path))

                proc = multiprocessing.Process(
                    target=processFile,
                    args=(file_path, config_path, platepar_path, output_dir, chunk_frames),
                    kwargs={'flat_path': flat_path, 'dark_path': dark_path, 'unique_id': unique_id}
                )
                proc.start()
                active_workers[unique_id] = proc

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

    arg_parser.add_argument('--recursive', '-r', action='store_true', default=False,
        help="Recursively monitor subdirectories for files."
    )

    arg_parser.add_argument('--flat', type=str, default=None,
        help="Path to a flat field image file."
    )

    arg_parser.add_argument('--dark', type=str, default=None,
        help="Path to a dark frame image file."
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
