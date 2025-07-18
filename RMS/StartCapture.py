# RPi Meteor Station
# Copyright (C) 2017 Dario Zubovic, Denis Vida
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function, absolute_import

import os
import sys
import glob
import argparse
import time
import datetime
import random
import signal
import shutil
import ctypes
import threading
import multiprocessing
import traceback
import git
from RMS.Formats.ObservationSummary import getObsDBConn, addObsParam

import numpy as np

# This needs to be first to set the proper matplotlib backend it needs
from Utils.LiveViewer import LiveViewer

import RMS.ConfigReader as cr
from RMS.Logger import initLogging, getLogger
from RMS.BufferedCapture import BufferedCapture
from RMS.CaptureDuration import captureDuration
from RMS.CaptureModeSwitcher import captureModeSwitcher
from RMS.Compression import Compressor
from RMS.DeleteOldObservations import deleteOldObservations
from RMS.DetectStarsAndMeteors import detectStarsAndMeteors
from RMS.Formats.FFfile import validFFName
from RMS.Misc import mkdirP, RmsDateTime, UTCFromTimestamp
from RMS.QueuedPool import QueuedPool
from RMS.Reprocess import getPlatepar, processNight, processFramesFiles
from RMS.RunExternalScript import runExternalScript
from RMS.UploadManager import UploadManager
from RMS.EventMonitor import EventMonitor
from RMS.DownloadMask import downloadNewMask
from RMS.Formats.ObservationSummary import startObservationSummaryReport
from Utils.AuditConfig import compareConfigs

# Flag indicating that capturing should be stopped
STOP_CAPTURE = False

def breakHandler(signum, frame):
    """ Handles what happens when Ctrl+C is pressed. """
        
    global STOP_CAPTURE

    # Set the flag to stop capturing video
    STOP_CAPTURE = True

    # This log entry is an adhoc fix to prevents Ctrl+C failure until the root cause is identified
    log.info("Ctrl+C pressed. Setting STOP_CAPTURE to True")

# Save the original event for the Ctrl+C
ORIGINAL_BREAK_HANDLE = signal.getsignal(signal.SIGINT)


def setSIGINT():
    """ Set the breakHandler function for the SIGINT signal, will be called when Ctrl+C is pressed. """

    signal.signal(signal.SIGINT, breakHandler)



def resetSIGINT():
    """ Restore the original Ctrl+C action. """

    signal.signal(signal.SIGINT, ORIGINAL_BREAK_HANDLE)





def wait(duration, compressor, buffered_capture, video_file, daytime_mode=None):
    """ The function will wait for the specified time, or it will stop when Enter is pressed. If no time was
        given (in seconds), it will wait until Enter is pressed. Additionally, it will also stop when the camera mode 
        (day/night) changes, in case of continuous capture mode.

    Arguments:
        duration: [float] Time in seconds to wait
        compressor: [Compressor] compressor process object
        buffered_capture: [BufferedCapture] buffered capture process object
        video_file: [str] Path to the video file, if it was given as the video source.
        daytime_mode: [multiprocessing.Value] shared boolean variable to keep track of camera day/night mode switching. None by default.
    """

    global STOP_CAPTURE


    log.info('Press Ctrl+C to stop capturing...')

    # Get the time of capture start
    time_start = RmsDateTime.utcnow()

    # Remember the initial camera mode value
    if daytime_mode is not None:
        daytime_mode_prev = daytime_mode.value
    else:
        daytime_mode_prev = False

    while True:

        # Sleep for a short interval
        time.sleep(1)


        # Break in case camera modes switched
        if (daytime_mode is not None) and (daytime_mode_prev != daytime_mode.value):
            break


        # If the compressor has died, restart capture
        # This will not be checked during daytime
        if (not daytime_mode_prev) and (not compressor.is_alive()):
            log.info('The compressor has died, restarting the capture!')
            break


        # If some wait time was given, check if it passed
        if duration is not None:

            time_elapsed = (RmsDateTime.utcnow() - time_start).total_seconds()

            # If the total time is elapsed, break the wait
            if time_elapsed >= duration:
                break


        if STOP_CAPTURE:
            break


        # If a video is given, quit when the video is done
        if video_file is not None:
            if buffered_capture.exit.is_set():
                break



def runCapture(config, duration=None, video_file=None, nodetect=False, detect_end=False, \
    upload_manager=None, eventmonitor=None, resume_capture=False, daytime_mode=None, camera_mode_switch_trigger=None):
    """ Run capture and compression for the given time.given
    
    Arguments:
        config: [config object] Configuration read from the .config file.

    Keyword arguments:
        duration: [float] Time in seconds to capture. None by default.
        video_file: [str] Path to the video file, if it was given as the video source. None by default.
        nodetect: [bool] If True, detection will not be performed. False by default.
        detect_end: [bool] If True, detection will be performed at the end of the night, when capture
            finishes. False by default.
        upload_manager: [UploadManager object] A handle to the UploadManager, which handles uploading files to
            the central server. None by default.
        eventmonitor: [EventMonitor object]. Event monitor object. None by default.
        resume_capture: [bool] Resume capture in the last data directory in CapturedFiles. False by default.
        daytime_mode: [multiprocessing.Value] shared boolean variable to keep track of camera day/night mode switching. None by default.

    Return:
        night_archive_dir: [str] Path to the archive folder of the processed night.
    """

    global STOP_CAPTURE


    # Check if resuming capture to the last capture directory
    night_data_dir_name = None
    if resume_capture:

        log.info("Resuming capture in the last capture directory...")

        # Find the latest capture directory
        capturedfiles_path = os.path.join(os.path.abspath(config.data_dir), config.captured_dir)
        most_recent_dir_time = 0
        for dir_name in sorted(os.listdir(capturedfiles_path)):

            dir_path_check = os.path.join(capturedfiles_path, dir_name)

            # Check it's a directory
            if os.path.isdir(dir_path_check):

                # Check if it starts with the correct station code
                if dir_name.startswith(str(config.stationID)):

                    dir_mod_time = os.path.getmtime(dir_path_check)

                    # Check that it is the most recent directory
                    if (night_data_dir_name is None) or (dir_mod_time > most_recent_dir_time):
                        night_data_dir_name = dir_name
                        night_data_dir = dir_path_check
                        most_recent_dir_time = dir_mod_time


        if night_data_dir_name is None:
            log.info("Previous capture directory could not be found! Creating a new one...")

        else:
            log.info("Previous capture directory found: {:s}".format(night_data_dir))


    # Make a name for the capture data directory
    if night_data_dir_name is None:

        # Create a directory for captured files based on the current time
        if video_file is None:
            night_data_dir_name = str(config.stationID) + '_' \
                + RmsDateTime.utcnow().strftime('%Y%m%d_%H%M%S_%f')

        # If a video file is given, take the folder name from the video file
        else:
            night_data_dir_name = os.path.basename(video_file[:-4])

        # Full path to the data directory
        night_data_dir = os.path.join(os.path.abspath(config.data_dir), config.captured_dir, \
            night_data_dir_name)

    # Full path to the time files directories
    if config.save_frame_times:
        ft_file_dir = os.path.join(os.path.abspath(config.data_dir), config.times_dir)

    # Full path to the saved frames directory
    if config.save_frames:
        saved_frames_dir = os.path.join(os.path.abspath(config.data_dir), config.frame_dir)
    else:
        saved_frames_dir = None

    # Full path to the video files
    if config.raw_video_save:
        saved_video_dir = os.path.join(os.path.abspath(config.data_dir), config.video_dir)
    else:
        saved_video_dir = None


    # Add a note about Patreon supporters
    print("################################################################")
    print("Thanks to our Patreon supporters in the 'Dinosaur Killer' class:")
    print("- Myron Valenta")
    print("And thanks to our Patreon supporters in the 'Bolide' class:")
    print("- David Attreed")
    print("https://www.patreon.com/globalmeteornetwork")
    print("\n\n\n" \
        + "       .:'       .:'        .:'       .:'  \n"\
        + "   _.::'     _.::'      _.::'     _.::'    \n"\
        + "  (_.'      (_.'       (_.'      (_.'      \n"\
        + "                         __                \n"\
        + "                        / _)               \n"\
        + "_\\/_          _/\\/\\/\\_/ /             _\\/_ \n"\
        + "/o\\         _|         /              //o\\ \n"\
        + " |         _|  (  | (  |                |  \n"\
        + "_|____    /__.-'|_|--|_|          ______|__\n")
    print("################################################################")

    # Add a note about deceased members
    print()
    print("In memory of Global Meteor Network members:")
    print("- Dr. Daniel A. Klinglesmith III (d. 2019)")
    print("- Martin Richmond-Hardy (d. 2023)")
    print("- Rajko Susanj (d. 2023)")
    print("- Zoran Dragic (d. 2025)")
    print("- Romke Schievink (d. 2025)")
    print()
    print("Memento mori")
    print("Each of us, a fleeting flame")
    print("Yet our paths remain.")
    print()
    print("################################################################")

    # Make a directory for the night - if currently in night capture mode
    in_night_capture = (daytime_mode is None) or (not daytime_mode.value)
    if (not config.continuous_capture) or in_night_capture:
        mkdirP(night_data_dir)
        log.info('Data directory: {}'.format(night_data_dir))

    # Make a directory for the time files if configured
    if config.save_frame_times:
        mkdirP(ft_file_dir)
        log.info('Saved FT files directory: {}'.format(ft_file_dir))

    # Make a directory for the saved frames
    if saved_frames_dir is not None:
        mkdirP(saved_frames_dir)
        log.info('Saved frames directory: {}'.format(saved_frames_dir))

    # Make a directory for the saved videos
    if saved_video_dir is not None:
        mkdirP(saved_video_dir)
        log.info('Saved videos directory: {}'.format(saved_video_dir))

    # Copy the used config file to the capture directory
    if os.path.isfile(config.config_file_name) and os.path.isdir(night_data_dir):
        try:
            # Get the name of the originating config file
            config_file_name = os.path.basename(config.config_file_name)

            # Copy the config file to the capture directory
            shutil.copy2(config.config_file_name, os.path.join(night_data_dir, config_file_name))

        except:
            log.error("Cannot copy the config file to the capture directory!")

    # Audit config file
    try:
        log.info(compareConfigs(config.config_file_name,
                                os.path.join(config.rms_root_dir, ".configTemplate"),
                                os.path.join(config.rms_root_dir, "RMS/ConfigReader.py")))
    except Exception as e:
        log.debug('Could not generate config audit report:' + repr(e))

    # Check for and get an updated mask
    if config.mask_download_permissive:
        downloadNewMask(config)

    # Get the platepar file
    platepar, platepar_path, platepar_fmt = getPlatepar(config, night_data_dir)

    # If the platepar is not none, set the FOV from it
    if platepar is not None:
        config.fov_w = platepar.fov_h
        config.fov_h = platepar.fov_v
        

    log.info('Initializing frame buffers...')
    ### For some reason, the RPi 3 does not like memory chunks which size is the multiplier of its L2
    ### cache size (512 kB). When such a memory chunk is provided, the compression becomes 10x slower
    ### then usual. We are applying a dirty fix here where we just add an extra image row and column
    ### if such a memory chunk will be created. The compression is performed, and the image is cropped
    ### back to its original dimensions.
    array_pad = 0

    # Check if the image dimensions are divisible by RPi3 L2 cache size and add padding
    if (256*config.width*config.height)%(512*1024) == 0:
        array_pad = 1


    # Init arrays for parallel compression on 2 cores
    sharedArrayBase = multiprocessing.Array(ctypes.c_uint8, 256*(config.width + array_pad)*(config.height + array_pad))
    sharedArray = np.ctypeslib.as_array(sharedArrayBase.get_obj())
    sharedArray = sharedArray.reshape(256, (config.height + array_pad), (config.width + array_pad))
    startTime = multiprocessing.Value('d', 0.0)

    sharedArrayBase2 = multiprocessing.Array(ctypes.c_uint8, 256*(config.width + array_pad)*(config.height + array_pad))
    sharedArray2 = np.ctypeslib.as_array(sharedArrayBase2.get_obj())
    sharedArray2 = sharedArray2.reshape(256, (config.height + array_pad), (config.width + array_pad))
    start_time2 = multiprocessing.Value('d', 0.0)

    log.info('Initializing frame buffers done!')


    # Initialize buffered capture
    bc = BufferedCapture(sharedArray, startTime, sharedArray2, start_time2, config, video_file=video_file,
                         night_data_dir=night_data_dir, saved_frames_dir=saved_frames_dir, 
                         daytime_mode=daytime_mode, camera_mode_switch_trigger=camera_mode_switch_trigger)
    bc.startCapture()

    # To track and make new directories every iteration
    ran_once = False

    # To handle control flow in case of disk space issues
    disk_full = False

    # Keep track of which state we started at
    daytime_mode_prev = False

    # Initialize the detector
    detector = None

    # Loop to handle both continuous and standard capture modes
    while True:

        # Continuous mode only: Setup new directories for a new night capture
        if config.continuous_capture and (not daytime_mode.value) and ran_once:

            # Create a new directory for captured files based on the current time
            night_data_dir_name = str(config.stationID) + '_' + RmsDateTime.utcnow().strftime('%Y%m%d_%H%M%S_%f')

            # Full path to the new data directory
            night_data_dir = os.path.join(os.path.abspath(config.data_dir), config.captured_dir, \
                night_data_dir_name)

            # Make a directory for the next capture
            mkdirP(night_data_dir)

            log.info('New data directory: {}'.format(night_data_dir))

            # Copy the used config file to the capture directory
            if os.path.isfile(config.config_file_name):
                try:
                    shutil.copy2(config.config_file_name, os.path.join(night_data_dir, ".config"))
                except:
                    log.error("Cannot copy the config file to the capture directory!")


            # Free up disk space for new capture
            log.info('Freeing up disk space...')

            if not deleteOldObservations(config.data_dir, config.captured_dir, config.archived_dir, config, duration=duration):

                log.error('No more disk space can be freed up! Stopping capture...')
                disk_full = True


        # Continuous mode only: Daytime capture
        if config.continuous_capture and daytime_mode.value:
                        
            log.info('Capturing in daytime mode...')

            # Capture until Ctrl+C is pressed / camera switches modes
            daytime_mode_prev = daytime_mode.value
            wait(duration, None, bc, video_file, daytime_mode)


        # Continuous OR standard mode: Nighttime capture
        elif (not disk_full):
        
            log.info('Capturing in nighttime mode...')

            # Check if the detection should be performed or not
            if nodetect:
                detector = None

            else:

                if detect_end and (duration is not None):

                    # Delay detection until the end of the night
                    delay_detection = duration

                else:
                    # Delay the detection for 2 minutes after capture start (helps stability)
                    delay_detection = 120


                # Add an additional postprocessing delay
                delay_detection += config.postprocess_delay


                # Set a flag file to indicate that previous files are being loaded (if any)
                capture_resume_file_path = os.path.join(config.data_dir, config.capture_resume_flag_file)
                with open(capture_resume_file_path, 'w') as f:
                    pass

                # Initialize the detector
                detector = QueuedPool(detectStarsAndMeteors, cores=1, log=log, delay_start=delay_detection, \
                    backup_dir=night_data_dir, input_queue_maxsize=None)
                detector.startPool()


                # If the capture is being resumed into the directory, load all previously saved FF files
                if resume_capture:

                    # Load all processed FF files
                    for i, ff_name in enumerate(sorted(os.listdir(night_data_dir))):

                        # Every 50 files loaded, update the flag file
                        if i%50 == 0:
                            with open(capture_resume_file_path, 'a') as f:
                                f.write("{:d}\n".format(i))
                                

                        # Check if the file is a valid FF files
                        ff_path = os.path.join(night_data_dir, ff_name)
                        if os.path.isfile(ff_path) and (str(config.stationID) in ff_name) and validFFName(ff_name):

                            # Add the FF file to the detector
                            detector.addJob([night_data_dir, ff_name, config], wait_time=0.005)
                            log.info("Added existing FF files for detection: {:s}".format(ff_name))


                # Remove the flag file
                if os.path.isfile(capture_resume_file_path):
                    try:
                        os.remove(capture_resume_file_path)
                    except:
                        log.error("There was an error during removing the capture resume flag file: " \
                            + capture_resume_file_path)


            # Initialize the live image viewer
            if config.live_maxpixel_enable:

                # Enable showing the live JPG
                config.live_jpg = True

                live_jpg_path = os.path.join(config.data_dir, 'live.jpg')

                live_view = LiveViewer(live_jpg_path, image=True, slideshow=False, banner_text="Live")
                live_view.start()

            else:
                live_view = None


            # Initialize compression
            compressor = Compressor(night_data_dir, sharedArray, startTime, sharedArray2, start_time2, config,
                detector=detector)

            # Open the observation summary report
            if video_file is None:
                log.info(startObservationSummaryReport(config, duration, force_delete=False))

            # Start the compressor
            compressor.start()

            # Capture until Ctrl+C is pressed / camera switches modes
            if (daytime_mode is not None):
                daytime_mode_prev = daytime_mode.value
                
            wait(duration, compressor, bc, video_file, daytime_mode)

            # Stop the compressor
            log.debug('Stopping compression...')
            detector = compressor.stop()
            log.debug('Compression stopped')

            if live_view is not None:

                # Stop the live viewer
                log.debug('Stopping live viewer...')

                live_view.stop()
                live_view.join()
                del live_view
                live_view = None

                log.debug('Live view stopped')


        # Manage capture termination:
        # stops BufferedCapture when capture is terminated manually (Continuous Mode)
        # stops BufferedCapture when night is done (Standard mode)
        if STOP_CAPTURE or (not config.continuous_capture):

            log.info('Ending capture...')

            # Stop the capture
            log.debug('Stopping capture...')
            dropped_frames = bc.stopCapture()
            log.debug('Capture stopped')

            log.info('Total number of late or dropped frames: ' + str(dropped_frames))
            obs_db_conn = getObsDBConn(config)
            addObsParam(obs_db_conn, "dropped_frames", dropped_frames)
            obs_db_conn.close()

            # Free shared memory after the compressor is done
            try:
                log.debug('Freeing frame buffers in StartCapture...')
                del sharedArrayBase
                del sharedArray
                del sharedArrayBase2
                del sharedArray2

            except Exception as e:
                log.debug('Freeing frame buffers failed with error:' + repr(e))
                log.debug(repr(traceback.format_exception(*sys.exc_info())))

            log.debug('Compression buffers freed')


        # Continuous OR standard mode: uploading and post-processing after night capture
        if (not config.continuous_capture) or (not daytime_mode_prev) and (not disk_full):

            # If detection should be performed
            if not nodetect:

                try:
                    if detector is None:
                        log.info('No detection queued')
                    else:
                        log.info('Finishing up the detection, ' + str(detector.input_queue.qsize()) \
                        + ' files to process...')
                except Exception:
                    log.exception('Finishing up the detection... error when getting input queue size!')


                # Reset the Ctrl+C to KeyboardInterrupt
                resetSIGINT()


                try:

                    # If there are some more files to process, process them on more cores
                    if detector.input_queue.qsize() > 0:

                        # If a fixed number of cores is not set, use all but 2 cores
                        if config.num_cores <= 0:
                            available_cores = multiprocessing.cpu_count() - 2

                        else:
                            available_cores = config.num_cores


                        if available_cores > 1:

                            log.info('Running the detection on {:d} cores...'.format(available_cores))

                            # Start the detector
                            detector.updateCoreNumber(cores=available_cores)


                    log.info('Waiting for the detection to finish...')

                    # Wait for the detector to finish and close it
                    try:
                        detector.closePool()
                    except Exception:
                        log.exception('Detector closePool() raised; continuing with shutdown')

                    log.info('Detection finished!')


                except KeyboardInterrupt:

                    log.info('Ctrl + C pressed, exiting...')

                    if upload_manager is not None:

                        # Stop the upload manager
                        if upload_manager.is_alive():
                            log.debug('Closing upload manager...')
                            upload_manager.stop()
                            del upload_manager

                    if eventmonitor is not None:

                        # Stop the eventmonitor manager
                        if eventmonitor.is_alive():
                            log.debug('Closing eventmonitor...')
                            eventmonitor.stop()
                            del eventmonitor


                    # Terminate the detector
                    if detector is not None:
                        del detector

                    sys.exit()


                # Set the Ctrl+C back to 'soft' program kill
                setSIGINT()

                ### SAVE DETECTIONS TO FILE


                log.info('Collecting results...')

                # Get the detection results from the queue
                detection_results = detector.getResults()

            else:

                detection_results = []

            # Save detection to disk and archive detection
            night_archive_dir, archive_name, _ = processNight(night_data_dir, config, \
                detection_results=detection_results, nodetect=nodetect)


            # Put the archive up for upload
            if upload_manager is not None:
                log.info("Adding file to upload list: %s", archive_name)
                upload_manager.addFiles([archive_name])
                log.info("File added.")

                # optional delay (minutes in .config, converted to seconds)
                upload_manager.delayNextUpload(delay=60*config.upload_delay)

            # Delete detector backup files
            if detector is not None:
                detector.deleteBackupFiles()


            # frames -> timelapse(s) -> archive(s) -> upload
            if config.timelapse_generate_from_frames:
                try:
                    log.info("Processing frame files...")
                    archive_paths = processFramesFiles(config)          # may return None
                    log.info("Processing frame files done.")

                except Exception:
                    log.exception("An error occurred when processing frame files!")
                    archive_paths = None

                # -- enqueue & upload -----------------------------------------
                if archive_paths and upload_manager:
                    try:
                        log.info("Adding file to upload list: %s", archive_paths)
                        upload_manager.addFiles(archive_paths)
                        log.info("File added.")

                    except Exception:
                        log.exception("Frames upload failed")


            # Run the external script
            runExternalScript(night_data_dir, night_archive_dir, config)


        # If capture is terminated manually, or the disk is full, exit program
        if STOP_CAPTURE or (disk_full):

            # Stop the upload manager
            if upload_manager is not None:
                if upload_manager.is_alive():
                    upload_manager.stop()
                    log.info('Closing upload manager...')

            if eventmonitor is not None:
                # Stop the eventmonitor
                if eventmonitor.is_alive():
                    log.debug('Closing eventmonitor...')
                    eventmonitor.stop()
                    del eventmonitor

            sys.exit()

        # Standard mode: need to run it all just once
        # Continuous mode: if the program just got done with nighttime processing and needs to reboot
        elif (not config.continuous_capture) or (not daytime_mode_prev and config.reboot_after_processing):
            break 

        ran_once = True


    return night_archive_dir



def processIncompleteCaptures(config, upload_manager):
    """ Reprocess broken capture folders.
    Arguments:
        config: [config object] Configuration read from the .config file.
        upload_manager: [UploadManager object] A handle to the UploadManager, which handles uploading files to
            the central server.
    """

    log.debug('Checking for folders containing partially-processed data')

    # Create a list of capture directories
    captured_dir_list = []
    captured_data_path = os.path.join(config.data_dir, config.captured_dir)
    for captured_dir_name in sorted(os.listdir(captured_data_path)):

        captured_dir_path = os.path.join(captured_data_path, captured_dir_name)

        # Check that the dir starts with the correct station code, that it really is a directory, and that
        #   there are some FF files inside
        if captured_dir_name.startswith(config.stationID):

            if os.path.isdir(captured_dir_path):

                if any([file_name.startswith("FF_{:s}".format(config.stationID)) \
                    for file_name in os.listdir(captured_dir_path)]):

                        captured_dir_list.append(captured_dir_name)


    # Check if there are any unprocessed or incompletely processed captured dirs
    for captured_subdir in captured_dir_list:

        captured_dir_path = os.path.join(config.data_dir, config.captured_dir, captured_subdir)
        log.debug("Checking folder: {:s}".format(captured_subdir))

        # Check if there are any backup pickle files in the capture directory
        pickle_files = glob.glob("{:s}/rms_queue_bkup_*.pickle".format(captured_dir_path))
        any_pickle_files = False
        if len(pickle_files) > 0:
            any_pickle_files = True

        # Check if there is an FTPdetectinfo file in the directory, indicating the folder was fully
        #   processed
        FTPdetectinfo_files = glob.glob('{:s}/FTPdetectinfo_*.txt'.format(captured_dir_path))
        any_ftpdetectinfo_files = False
        newest_FTPfile_older_than_platepar = False
        if len(FTPdetectinfo_files) > 0:
            any_ftpdetectinfo_files = True

            # Is the platepar in the captured_dir_path newer than latest FTP file?
            # i.e. has the operator replaced the platepar because of bad calibration?
            newest_FTPfile_older_than_platepar = True
            for FTPfile in FTPdetectinfo_files:
                capture_platepar = os.path.join(captured_dir_path,config.platepar_name)
                if os.path.exists(capture_platepar):
                    # Any FTPfile newer than platepar - no need to reprocess
                    if os.path.getmtime(FTPfile) > os.path.getmtime(capture_platepar):
                        newest_FTPfile_older_than_platepar = False
                else:
                    # if there is no platepar in the captured_dir_path
                    newest_FTPfile_older_than_platepar = False

        # Auto reprocess criteria:
        #   - Any backup pickle files
        #   - No pickle and no FTPdetectinfo files
        #   - Newest FTP file older than platepar in capture directory

        run_reprocess = False
        if any_pickle_files:
            run_reprocess = True
        else:
            if not any_ftpdetectinfo_files:
                run_reprocess = True
        if newest_FTPfile_older_than_platepar:
                run_reprocess = True
                log.info("Reprocessing because newest FTPDetect file older than platepar file")

        # Skip the folder if it doesn't need to be reprocessed
        if not run_reprocess:
            log.debug("    ... fully processed!")
            continue


        log.info("Found partially-processed data in {:s}".format(captured_dir_path))
        try:

            # Reprocess the night
            night_archive_dir, archive_name, detector = processNight(captured_dir_path, config)

            # Upload the archive, if upload is enabled
            if upload_manager is not None:
                log.info("Adding file to upload list: {:s}".format(archive_name))
                upload_manager.addFiles([archive_name])
                log.info("File added...")

            # Delete detection backup files
            if detector is not None:
                detector.deleteBackupFiles()


            # Run the external script if running after autoreprocess is enabled
            if config.external_script_run and config.auto_reprocess_external_script_run:
                runExternalScript(captured_dir_path, night_archive_dir, config)

            log.info("Folder {:s} reprocessed with success!".format(captured_dir_path))


        except Exception as e:
            log.error("An error occurred when trying to reprocess partially processed data!")
            log.error(repr(e))
            log.error(repr(traceback.format_exception(*sys.exc_info())))

        # If capture should have started do not process any more incomplete directories
        # Ignore this for continuous capture
        if (not config.continuous_capture):
            
            start_time, duration = captureDuration(config.latitude, config.longitude, config.elevation)
            if isinstance(start_time, bool):
                log.info("Capture should have started, do not start reprocessing another directory")
                break




if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description=""" Starting capture and compression.
        """)

    # Add a mutually exclusive for the parser (the arguments in the group can't be given at the same)
    arg_group = arg_parser.add_mutually_exclusive_group()

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one.")

    arg_group.add_argument('-d', '--duration', metavar='DURATION_HOURS', help="""Start capturing right away, 
        with the given duration in hours. """)

    arg_group.add_argument('-i', '--input', metavar='FILE_PATH', help="""Use video from the given file, 
        not from a video device. The name of the file needs to be in the following format: STATIONID_YYYYMMDD_HHMMSS_US.mp4, where the time is the UTC time of the first frame. Example: CZ0002_20210317_193338_404889.mp4.""")

    arg_parser.add_argument('-n', '--nodetect', action="store_true", help="""Do not perform star extraction 
        nor meteor detection. """)

    arg_parser.add_argument('-e', '--detectend', action="store_true", help="""Detect stars and meteors at the
        end of the night, after capture finishes. """)

    arg_parser.add_argument('-r', '--resume', action="store_true", \
        help="""Resume capture into the last night directory in CapturedFiles. """)
    
    arg_parser.add_argument('--num_cores', metavar='NUM_CORES', type=int, default=None, \
        help="Number of cores to use for detection. Default is what is specific in the config file. " 
        "If not given in the config file, all available cores will be used."
        )


    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    ######
    video_file = cml_args.input

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('.'))


    # Initialize the logger
    initLogging(config)

    # Get the logger handle
    log = getLogger("logger")


    log.info("Program start")
    log.info("Station code: {:s}".format(str(config.stationID)))

    # Get the program version
    try:
        # Get latest version's commit hash and time of commit
        repo = git.Repo(search_parent_directories=True)
        commit_unix_time = repo.head.object.committed_date
        sha = repo.head.object.hexsha
        commit_time = UTCFromTimestamp.utcfromtimestamp(commit_unix_time).strftime('%Y%m%d_%H%M%S')

    except:
        commit_time = ""
        sha = ""

    log.info("Program version: {:s}, {:s}".format(commit_time, sha))


    # Set the number of cores to use if given
    if cml_args.num_cores is not None:
        config.num_cores = cml_args.num_cores

        if config.num_cores <= 0:
            config.num_cores = -1

            log.info("Using all available cores for detection.")


    # Change the Ctrl+C action to the special handle
    setSIGINT()


    # Make the data directories
    root_dir = os.path.abspath(config.data_dir)
    mkdirP(root_dir)
    mkdirP(os.path.join(root_dir, config.captured_dir))
    mkdirP(os.path.join(root_dir, config.archived_dir))

    # Check for and get an updated mask
    if config.mask_download_permissive:
        downloadNewMask(config)

    # If the duration of capture was given, capture right away for a specified time
    if cml_args.duration:

        try:
            # Get the duration in seconds
            duration = float(cml_args.duration)*60*60

        except:
            log.error('Given duration is not a proper number of hours!')
            sys.exit()



        log.info('Freeing up disk space...')

        # Free up disk space by deleting old files, if necessary
        if not deleteOldObservations(config.data_dir, config.captured_dir, config.archived_dir, config,
            duration=duration):

            log.error('No more disk space can be freed up! Stopping capture...')
            sys.exit()


        upload_manager = None
        if config.upload_enabled:

            # Init the upload manager
            log.info('Starting the upload manager...')
            upload_manager = UploadManager(config)
            upload_manager.start()


        log.info("Running for " + str(duration/60/60) + ' hours...')

        # Disable continuous capture for fixed duration capture
        config.continuous_capture = False

        # Run the capture for the given number of hours
        runCapture(config, duration=duration, nodetect=cml_args.nodetect, upload_manager=upload_manager, \
            detect_end=cml_args.detectend, resume_capture=cml_args.resume)
        cml_args.resume = False

        if upload_manager is not None:
            # Stop the upload manager
            if upload_manager.is_alive():
                log.info('Closing upload manager...')
                upload_manager.stop()


        sys.exit()



    # If a file with video input was give, use it as a video source. These files fill not the uploaded to the
    # server, because the video was recorded beforehand!
    if cml_args.input:

        log.info('Video source: ' + cml_args.input)

        # Disable continuous capture for video file capture
        config.continuous_capture = False
        
        # Capture the video frames from the video file
        runCapture(config, duration=None, video_file=video_file, nodetect=cml_args.nodetect,
            resume_capture=cml_args.resume)
        cml_args.resume = False

        sys.exit()

    upload_manager = None
    if config.upload_enabled:

        # Init the upload manager
        log.info('Starting the upload manager...')
        upload_manager = UploadManager(config)
        upload_manager.start()

    eventmonitor = None

    if config.event_monitor_enabled:
        # Init the event monitor
        log.info('Starting the event monitor...')
        eventmonitor = EventMonitor(config)
        eventmonitor.start()

    # Automatic running and stopping the capture at sunrise and sunset
    ran_once = False
    slideshow_view = None
    while True:

        if config.continuous_capture:

            # Start immediately in case of continuous capture mode
            start_time = True
            duration = None
            log.info('Starting continuous capture now...')
        
        else:

            # Calculate when and how should the capture run
            start_time, duration = captureDuration(config.latitude, config.longitude, config.elevation)
            log.info('Next start time: ' + str(start_time) + ' UTC')


        # Reboot the computer after processing is done for the previous night
        if ran_once and config.reboot_after_processing:

            log.info("Trying to reboot after processing in 30 seconds...")
            time.sleep(30)

            # Try rebooting for 4 hours, stop if capture should run
            for reboot_try in range(4*60):

                reboot_go = True

                # Check if the upload manager is uploading
                if upload_manager is not None:

                    # Prevent rebooting if the upload manager is uploading
                    if upload_manager.upload_in_progress.value:
                        log.info("Reboot delayed for 1 minute due to upload...")
                        reboot_go = False

                # Check if the reboot lock file exists
                reboot_lock_file_path = os.path.join(config.data_dir, config.reboot_lock_file)
                if os.path.exists(reboot_lock_file_path):
                    log.info("Reboot delayed for 1 minute because the lock file exists: {:s}".format(reboot_lock_file_path))
                    reboot_go = False


                # Reboot the computer
                if reboot_go:

                    log.info('Rebooting now!')

                    # Reboot the computer (script needs sudo privileges, works only on Linux)
                    try:
                        os.system('sudo shutdown -r now')

                    except Exception as e:
                        log.debug('Rebooting failed with message:\n' + repr(e))
                        log.debug(repr(traceback.format_exception(*sys.exc_info())))

                else:

                    # Wait one more minute and try again to reboot
                    time.sleep(60)


                ### Stop reboot tries if it's time to capture ###
                if (not config.continuous_capture):
                    
                    if isinstance(start_time, bool):
                        if start_time:
                            break

                    time_now = RmsDateTime.utcnow()
                    waiting_time = start_time - time_now
                    if waiting_time.total_seconds() <= 0:
                        break

                ### ###



        # Don't start the capture if there's less than 15 minutes left
        if (not config.continuous_capture) and duration < 15*60:

            log.debug('Less than 15 minutes left to record, waiting for a new recording session tonight...')

            # Reset the Ctrl+C to KeyboardInterrupt
            resetSIGINT()

            try:
                # Wait for 30 mins before checking again
                time.sleep(30*60)

            except KeyboardInterrupt:

                log.info('Ctrl + C pressed, exiting...')

                if upload_manager is not None:

                    # Stop the upload manager
                    if upload_manager.is_alive():
                        log.debug('Closing upload manager...')
                        upload_manager.stop()
                        del upload_manager

                if eventmonitor is not None:

                    # Stop the eventmonitor
                    if eventmonitor.is_alive():
                        log.debug('Closing eventmonitor...')
                        eventmonitor.stop()
                        del eventmonitor



                sys.exit()

            # Change the Ctrl+C action to the special handle
            setSIGINT()

            continue


        # Run auto-reprocessing only if the config option is set
        if config.auto_reprocess:

            # In case of continuous capture, start processing incomplete captures
            if not isinstance(start_time, bool) or config.continuous_capture:

                # Check if there's a folder containing unprocessed data.
                # This may happen if the system crashed during processing.
                processIncompleteCaptures(config, upload_manager)


        # Wait to start capturing and initialize last night's slideshow
        if not isinstance(start_time, bool):

            # Initialize the slideshow of last night's detections
            if config.slideshow_enable:

                # Make a list of all archived directories previously generated
                archive_dir_list = []
                for archive_dir_name in sorted(os.listdir(os.path.join(config.data_dir,
                    config.archived_dir))):

                    if archive_dir_name.startswith(config.stationID):
                        if os.path.isdir(os.path.join(config.data_dir, config.archived_dir, \
                            archive_dir_name)):

                            archive_dir_list.append(archive_dir_name)



                # If there are any archived dirs, choose the last one
                if archive_dir_list:

                    latest_night_archive_dir = os.path.join(config.data_dir, config.archived_dir, \
                        archive_dir_list[-1])

                    # Make sure that there are any FF files in the chosen archived dir
                    ffs_latest_night_archive = [ff_name for ff_name \
                        in os.listdir(latest_night_archive_dir) if validFFName(ff_name)]

                    if len(ffs_latest_night_archive):

                        log.info("Starting a slideshow of {:d} detections from the previous night.".format(len(ffs_latest_night_archive)))

                        # Start the slide show
                        slideshow_view = LiveViewer(latest_night_archive_dir, slideshow=True, \
                            banner_text="Last night's detections")
                        slideshow_view.start()

                    else:
                        log.info("No detections from the previous night to show as a slideshow!")



            # Update start time and duration
            start_time, duration = captureDuration(config.latitude, config.longitude, config.elevation)

            # Check if waiting is needed to start capture
            if not isinstance(start_time, bool):

                # Calculate how many seconds to wait until capture starts, and with for that time
                time_now = RmsDateTime.utcnow()
                waiting_time = start_time - time_now

                log.info('Waiting {:s} to start recording for {:.3f} hrs'.format(str(waiting_time), \
                    duration/60/60))

                # Reset the Ctrl+C to KeyboardInterrupt
                resetSIGINT()

                try:

                    # Wait until sunset
                    waiting_time_seconds = int(waiting_time.total_seconds())
                    if waiting_time_seconds > 0:
                        time.sleep(waiting_time_seconds)

                except KeyboardInterrupt:

                    log.info('Ctrl + C pressed, exiting...')

                    if upload_manager is not None:

                        # Stop the upload manager
                        if upload_manager.is_alive():
                            log.debug('Closing upload manager...')
                            upload_manager.stop()
                            del upload_manager

                    if eventmonitor is not None:

                        # Stop the eventmonitor
                        if eventmonitor.is_alive():
                             log.debug('Closing eventmonitor...')
                             eventmonitor.stop()
                             del eventmonitor

                        # Stop the slideshow if it was on
                        if slideshow_view is not None:

                            log.info("Stopping slideshow...")
                            slideshow_view.stop()
                            slideshow_view.join()
                            del slideshow_view
                            slideshow_view = None

                    sys.exit()

                # Change the Ctrl+C action to the special handle
                setSIGINT()


        # Break the loop if capturing was stopped
        if STOP_CAPTURE:
            break


        # Stop the slideshow if it was on
        if slideshow_view is not None:
            log.info("Stopping slideshow...")
            slideshow_view.stop()
            slideshow_view.join()
            del slideshow_view
            slideshow_view = None


        # Determine how long to wait before the capture starts (include randomization if set)
        capture_wait_time = config.capture_wait_seconds
        if config.capture_wait_randomize and (config.capture_wait_seconds > 0):
            capture_wait_time = random.randint(0, config.capture_wait_seconds)

        # Wait before the capture starts if a time has been given
        if (not cml_args.resume) and ((capture_wait_time > 0) or config.capture_wait_randomize):

            rand_str = ""
            if config.capture_wait_randomize:
                rand_str = " (randomized between 0 and {:d})".format(config.capture_wait_seconds)

            log.info("Waiting {:d} seconds{:s} before capture start...".format(int(capture_wait_time), rand_str))
            time.sleep(capture_wait_time)



        log.info('Freeing up disk space...')

        # Free up disk space by deleting old files, if necessary
        if not deleteOldObservations(config.data_dir, config.captured_dir, config.archived_dir, config,
            duration=duration):

            log.error('No more disk space can be freed up! Stopping capture...')
            break


        if config.continuous_capture:
            
            # Setup shared value to communicate day/night switch between processes.
            daytime_mode = multiprocessing.Value(ctypes.c_bool, False)
            camera_mode_switch_trigger = multiprocessing.Value(ctypes.c_bool, True)

            # Setup the capture mode switcher on another thread
            capture_switcher = threading.Thread(target=captureModeSwitcher, args=(config, daytime_mode, camera_mode_switch_trigger))
            
            # To make sure the capture switcher thread exits automatically at the end
            capture_switcher.daemon = True
            
            capture_switcher.start()

            # Wait for the switcher to complete calculation and switch to correct camera mode
            time.sleep(3)

            # Capture the health of the thread. If dead, then restart capture with
            # continuous_capture disabled
            if capture_switcher.is_alive():
                log.info('Started capture mode switcher on a separate thread')

            else:
                log.error('Capture mode switcher thread failed. Restarting capture with continuous capture off')
                config.continuous_capture = False
                continue

        else:
            daytime_mode = None
            camera_mode_switch_trigger = None
            log.info('Starting capture for {:.2f} hours'.format(duration/60/60))


        # Run capture and compression
        night_archive_dir = runCapture(config, duration=duration, nodetect=cml_args.nodetect, \
            upload_manager=upload_manager, eventmonitor=eventmonitor, detect_end=(cml_args.detectend or config.postprocess_at_end), \
            resume_capture=cml_args.resume, daytime_mode=daytime_mode, camera_mode_switch_trigger=camera_mode_switch_trigger)
        cml_args.resume = False

        # Indicate that the capture was done once
        ran_once = True



    if upload_manager is not None:

        # Stop the upload manager
        if upload_manager.is_alive():
            log.debug('Closing upload manager...')
            upload_manager.stop()
            del upload_manager

    if eventmonitor is not None:

    # Stop the event monitor
        if eventmonitor.is_alive():
             log.debug('Closing eventmonitor...')
             eventmonitor.stop()

             del eventmonitor

