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
import signal
import ctypes
import logging
import multiprocessing
import traceback

import numpy as np

# This needs to be first to set the proper matplotlib backend it needs
from Utils.LiveViewer import LiveViewer

import RMS.ConfigReader as cr
from RMS.Logger import initLogging
from RMS.BufferedCapture import BufferedCapture
from RMS.CaptureDuration import captureDuration
from RMS.Compression import Compressor
from RMS.DeleteOldObservations import deleteOldObservations
from RMS.DetectStarsAndMeteors import detectStarsAndMeteors
from RMS.Formats.FFfile import validFFName
from RMS.Misc import mkdirP
from RMS.QueuedPool import QueuedPool
from RMS.Reprocess import getPlatepar, processNight
from RMS.RunExternalScript import runExternalScript
from RMS.UploadManager import UploadManager

# Flag indicating that capturing should be stopped
STOP_CAPTURE = False

def breakHandler(signum, frame):
    """ Handles what happens when Ctrl+C is pressed. """
        
    global STOP_CAPTURE

    # Set the flag to stop capturing video
    STOP_CAPTURE = True


# Save the original event for the Ctrl+C
ORIGINAL_BREAK_HANDLE = signal.getsignal(signal.SIGINT)


def setSIGINT():
    """ Set the breakHandler function for the SIGINT signal, will be called when Ctrl+C is pressed. """

    signal.signal(signal.SIGINT, breakHandler)



def resetSIGINT():
    """ Restore the original Ctrl+C action. """

    signal.signal(signal.SIGINT, ORIGINAL_BREAK_HANDLE)





def wait(duration, compressor):
    """ The function will wait for the specified time, or it will stop when Enter is pressed. If no time was
        given (in seconds), it will wait until Enter is pressed. 
    Arguments:
        duration: [float] Time in seconds to wait
    """

    global STOP_CAPTURE


    log.info('Press Ctrl+C to stop capturing...')

    # Get the time of capture start
    time_start = datetime.datetime.utcnow()


    while True:

        # Sleep for a short interval
        time.sleep(1)


        # If the compressor has died, restart capture
        if not compressor.is_alive():
            log.info('The compressor has died, restarting the capture!')
            break


        # If some wait time was given, check if it passed
        if duration is not None:

            time_elapsed = (datetime.datetime.utcnow() - time_start).total_seconds()

            # If the total time is elapsed, break the wait
            if time_elapsed >= duration:
                break


        if STOP_CAPTURE:
            break




def runCapture(config, duration=None, video_file=None, nodetect=False, detect_end=False, \
    upload_manager=None, resume_capture=False):
    """ Run capture and compression for the given time.given
    Arguments:
        config: [config object] Configuration read from the .config file.
    Keyword arguments:
        duration: [float] Time in seconds to capture. None by default.
        video_file: [str] Path to the video file, if it was given as the video source. None by default.
        nodetect: [bool] If True, detection will not be performed. False by defualt.
        detect_end: [bool] If True, detection will be performed at the end of the night, when capture
            finishes. False by default.
        upload_manager: [UploadManager object] A handle to the UploadManager, which handles uploading files to
            the central server. None by default.
        resume_capture: [bool] Resume capture in the last data directory in CapturedFiles.
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

        # Resume run is finished now, reset resume flag
        cml_args.resume = False

    # Make a name for the capture data directory
    if night_data_dir_name is None:

        # Create a directory for captured files
        night_data_dir_name = str(config.stationID) + '_' \
            + datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')

        # Full path to the data directory
        night_data_dir = os.path.join(os.path.abspath(config.data_dir), config.captured_dir, \
            night_data_dir_name)



    # Make a directory for the night
    mkdirP(night_data_dir)

    log.info('Data directory: ' + night_data_dir)


    # Get the platepar file
    platepar, platepar_path, platepar_fmt = getPlatepar(config, night_data_dir)


    log.info('Initializing frame buffers...')
    ### For some reason, the RPi 3 does not like memory chunks which size is the multipier of its L2
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
    startTime2 = multiprocessing.Value('d', 0.0)

    log.info('Initializing frame buffers done!')


    # Check if the detection should be performed or not
    if nodetect:
        detector = None

    else:

        if detect_end:

            # Delay detection until the end of the night
            delay_detection = duration

        else:
            # Delay the detection for 2 minutes after capture start
            delay_detection = 120

        # Initialize the detector
        detector = QueuedPool(detectStarsAndMeteors, cores=1, log=log, delay_start=delay_detection, \
            backup_dir=night_data_dir)
        detector.startPool()


        # If the capture is being resumed into the directory, load all previously saved FF files
        if resume_capture:

            for ff_name in sorted(os.listdir(night_data_dir)):

                # Check if the file is a valid FF files
                ff_path = os.path.join(night_data_dir, ff_name)
                if os.path.isfile(ff_path) and (str(config.stationID) in ff_name) and validFFName(ff_name):

                    # Add the FF file to the detector
                    detector.addJob([night_data_dir, ff_name, config])
                    log.info("Added existing FF files for detection: {:s}".format(ff_name))


    # Initialize buffered capture
    bc = BufferedCapture(sharedArray, startTime, sharedArray2, startTime2, config, video_file=video_file)


    # Initialize the live image viewer
    if config.live_maxpixel_enable:
        live_view = LiveViewer(night_data_dir, slideshow=False, banner_text="Live")
        live_view.start()

    else:
        live_view = None


    # Initialize compression
    compressor = Compressor(night_data_dir, sharedArray, startTime, sharedArray2, startTime2, config,
        detector=detector)


    # Start buffered capture
    bc.startCapture()

    # Init and start the compression
    compressor.start()


    # Capture until Ctrl+C is pressed
    wait(duration, compressor)

    # If capture was manually stopped, end capture
    if STOP_CAPTURE:
        log.info('Ending capture...')


    # Stop the capture
    log.debug('Stopping capture...')
    bc.stopCapture()
    log.debug('Capture stopped')

    dropped_frames = bc.dropped_frames
    log.info('Total number of late or dropped frames: ' + str(dropped_frames))


    # Stop the compressor
    log.debug('Stopping compression...')
    detector = compressor.stop()

    # Free shared memory after the compressor is done
    try:
        log.debug('Freeing frame buffers...')
        del sharedArrayBase
        del sharedArray
        del sharedArrayBase2
        del sharedArray2

    except Exception as e:
        log.debug('Freeing frame buffers failed with error:' + repr(e))
        log.debug(repr(traceback.format_exception(*sys.exc_info())))

    log.debug('Compression stopped')


    if live_view is not None:

        # Stop the live viewer
        log.debug('Stopping live viewer...')

        live_view.stop()
        live_view.join()
        del live_view
        live_view = None

        log.debug('Live view stopped')



    # If detection should be performed
    if not nodetect:

        try:
            log.info('Finishing up the detection, ' + str(detector.input_queue.qsize()) \
                + ' files to process...')
        except:
            print('Finishing up the detection... error when getting input queue size!')


        # Reset the Ctrl+C to KeyboardInterrupt
        resetSIGINT()


        try:

            # If there are some more files to process, process them on more cores
            if detector.input_queue.qsize() > 0:

                # Let the detector use all cores, but leave 2 free
                available_cores = multiprocessing.cpu_count() - 2


                if available_cores > 1:

                    log.info('Running the detection on {:d} cores...'.format(available_cores))

                    # Start the detector
                    detector.updateCoreNumber(cores=available_cores)


            log.info('Waiting for the detection to finish...')

            # Wait for the detector to finish and close it
            detector.closePool()

            log.info('Detection finished!')


        except KeyboardInterrupt:

            log.info('Ctrl + C pressed, exiting...')

            if upload_manager is not None:

                # Stop the upload manager
                if upload_manager.is_alive():
                    log.debug('Closing upload manager...')
                    upload_manager.stop()
                    del upload_manager


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
        log.info('Adding file to upload list: ' + archive_name)
        upload_manager.addFiles([archive_name])
        log.info('File added...')


    # Delete detector backup files
    if detector is not None:
        detector.deleteBackupFiles()


    # If the capture was run for a limited time, run the upload right away
    if (duration is not None) and (upload_manager is not None):
        log.info('Uploading data before exiting...')
        upload_manager.uploadData()


    # Run the external script
    runExternalScript(night_data_dir, night_archive_dir, config)


    # If capture was manually stopped, end program
    if STOP_CAPTURE:

        log.info('Ending program')

        # Stop the upload manager
        if upload_manager is not None:
            if upload_manager.is_alive():
                upload_manager.stop()
                log.info('Closing upload manager...')

        sys.exit()


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

        # Check that the dir stars with the correct station code, that it really is a directory, and that
        #   there are some FF files inside
        if captured_dir_name.startswith(config.stationID):

            if os.path.isdir(captured_dir_path):

                if any([file_name.startswith("FF_{:s}".format(config.stationID)) \
                    for file_name in os.listdir(captured_dir_path)]):

                        captured_dir_list.append(captured_dir_name)


    # Check if there is a processed archived dir for every captured dir
    for captured_subdir in captured_dir_list:

        captured_dir_path = os.path.join(config.data_dir, config.captured_dir, captured_subdir)
        log.debug("Checking folder: {:s}".format(captured_subdir))

        # Check if there are any backup pickle files in the capture directory
        pickle_files = glob.glob("{:s}/rms_queue_bkup_*.pickle".format(captured_dir_path))
        any_pickle_files = False
        if len(pickle_files) > 0:
            any_pickle_files = True

        # Check if there is an FTPdetectinfo file in the directory, indicating the the folder was fully
        #   processed
        FTPdetectinfo_files = glob.glob('{:s}/FTPdetectinfo_*.txt'.format(captured_dir_path))
        any_ftpdetectinfo_files = False
        if len(FTPdetectinfo_files) > 0:
            any_ftpdetectinfo_files = True

        # Auto reprocess criteria:
        #   - Any backup pickle files
        #   - No pickle and no FTPdetectinfo files
        run_reprocess = False
        if any_pickle_files:
            run_reprocess = True
        else:
            if not any_ftpdetectinfo_files:
                run_reprocess = True

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

        except:
            log.error("An error occured when trying to reprocess partially-processed data!")


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
        not from a video device. """)

    arg_parser.add_argument('-n', '--nodetect', action="store_true", help="""Do not perform star extraction 
        nor meteor detection. """)

    arg_parser.add_argument('-e', '--detectend', action="store_true", help="""Detect stars and meteors at the
        end of the night, after capture finishes. """)

    arg_parser.add_argument('-r', '--resume', action="store_true", \
        help="""Resume capture into the last night directory in CapturedFiles. """)


    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    ######


    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('.'))

    # Initialize the logger
    initLogging(config)

    # Get the logger handle
    log = logging.getLogger("logger")


    log.info('Program start')

    # Change the Ctrl+C action to the special handle
    setSIGINT()


    # Make the data directories
    root_dir = os.path.abspath(config.data_dir)
    mkdirP(root_dir)
    mkdirP(os.path.join(root_dir, config.captured_dir))
    mkdirP(os.path.join(root_dir, config.archived_dir))


    # If the duration of capture was given, capture right away for a specified time
    if cml_args.duration:

        try:
            # Get the duration in seconds
            duration = float(cml_args.duration)*60*60

        except:
            log.error('Given duration is not a proper number of hours!')



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

        # Run the capture for the given number of hours
        runCapture(config, duration=duration, nodetect=cml_args.nodetect, upload_manager=upload_manager, \
            detect_end=cml_args.detectend, resume_capture=cml_args.resume)

        if upload_manager is not None:
            # Stop the upload manager
            if upload_manager.is_alive():
                log.info('Closing upload manager...')
                upload_manager.stop()
                del upload_manager


        sys.exit()



    # If a file with video input was give, use it as a video source. These files fill not the uploaded to the
    # server, because the video was recorded beforehand!
    if cml_args.input:

        log.info('Video source: ' + cml_args.input)

        # Capture the video frames from the video file
        runCapture(config, video_file=cml_args.input, nodetect=cml_args.nodetect,
            resume_capture=cml_args.resume)


    upload_manager = None
    if config.upload_enabled:

        # Init the upload manager
        log.info('Starting the upload manager...')
        upload_manager = UploadManager(config)
        upload_manager.start()


    # Automatic running and stopping the capture at sunrise and sunset
    ran_once = False
    slideshow_view = None
    while True:

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
                    log.info("Reboot delayed for 1 minute becase the lock file exists: {:s}".format(reboot_lock_file_path))
                    reboot_go = False


                # Reboot the computer
                if reboot_go:

                    log.info('Rebooting now!')

                    # Reboot the computer (script needs sudo priviledges, works only on Linux)
                    try:
                        os.system('sudo shutdown -r now')

                    except Exception as e:
                        log.debug('Rebooting failed with message:\n' + repr(e))
                        log.debug(repr(traceback.format_exception(*sys.exc_info())))

                else:

                    # Wait one more minute and try again to reboot
                    time.sleep(60)


                ### Stop reboot tries if it's time to capture ###
                if isinstance(start_time, bool):
                    if start_time:
                        break

                time_now = datetime.datetime.utcnow()
                waiting_time = start_time - time_now
                if waiting_time.total_seconds() <= 0:
                    break

                ### ###



        # Don't start the capture if there's less than 15 minutes left
        if duration < 15*60:

            log.debug('Less than 15 minues left to record, waiting for a new recording session tonight...')

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

                sys.exit()

            # Change the Ctrl+C action to the special handle
            setSIGINT()

            continue


        # Wait to start capturing
        if start_time != True:

            # Run auto-reprocessing
            if config.auto_reprocess:

                # Check if there's a folder containing unprocessed data.
                # This may happen if the system crashed during processing.
                processIncompleteCaptures(config, upload_manager)


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


            # Calculate how many seconds to wait until capture starts, and with for that time
            time_now = datetime.datetime.utcnow()
            waiting_time = start_time - time_now

            log.info('Waiting {:s} to start recording for {:.2f} hrs'.format(str(waiting_time), \
                duration/60/60))

            # Reset the Ctrl+C to KeyboardInterrupt
            resetSIGINT()

            try:
                # Wait until sunset
                time.sleep(int(waiting_time.total_seconds()))

            except KeyboardInterrupt:

                log.info('Ctrl + C pressed, exiting...')

                if upload_manager is not None:

                    # Stop the upload manager
                    if upload_manager.is_alive():
                        log.debug('Closing upload manager...')
                        upload_manager.stop()
                        del upload_manager


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


        log.info('Freeing up disk space...')

        # Free up disk space by deleting old files, if necessary
        if not deleteOldObservations(config.data_dir, config.captured_dir, config.archived_dir, config,
            duration=duration):

            log.error('No more disk space can be freed up! Stopping capture...')
            break


        log.info('Starting capture for {:.2f} hours'.format(duration/60/60))

        # Run capture and compression
        night_archive_dir = runCapture(config, duration=duration, nodetect=cml_args.nodetect,
            upload_manager=upload_manager, detect_end=cml_args.detectend, resume_capture=cml_args.resume)

        # Indicate that the capture was done once
        ran_once = True



    if upload_manager is not None:

        # Stop the upload manager
        if upload_manager.is_alive():
            log.debug('Closing upload manager...')
            upload_manager.stop()
            del upload_manager