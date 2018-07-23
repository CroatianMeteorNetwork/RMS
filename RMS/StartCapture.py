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
import argparse
import time
import datetime
import signal
import ctypes
import logging
import multiprocessing

import numpy as np
import scipy.misc

import RMS.ConfigReader as cr
from RMS.Logger import initLogging

from RMS.Formats import FTPdetectinfo
from RMS.Formats import CALSTARS
from RMS.Formats.Platepar import Platepar
from RMS.Routines import Image


from RMS.DeleteOldObservations import deleteOldObservations
from RMS.BufferedCapture import BufferedCapture
from RMS.Compression import Compressor
from RMS.CaptureDuration import captureDuration
from RMS.DetectStarsAndMeteors import detectStarsAndMeteors
from RMS.Astrometry.CheckFit import autoCheckFit
from RMS.Astrometry.ApplyAstrometry import applyAstrometryFTPdetectinfo
from RMS.ArchiveDetections import archiveDetections, archiveFieldsums
from RMS.UploadManager import UploadManager

from RMS.LiveViewer import LiveViewer
from RMS.QueuedPool import QueuedPool
from RMS.Misc import mkdirP

from Utils.PlotFieldsums import plotFieldsums
from Utils.MakeFlat import makeFlat



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





def wait(duration=None):
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
        time.sleep(0.1)

        # If some wait time was given, check if it passed
        if duration is not None:

            time_elapsed = (datetime.datetime.utcnow() - time_start).total_seconds()

            # If the total time is elapsed, break the wait
            if time_elapsed >= duration:
                break


        if STOP_CAPTURE:
            break




def runCapture(config, duration=None, video_file=None, nodetect=False, detect_end=False, upload_manager=None):
    """ Run capture and compression for the given time.given

    Arguments:
        config: [config object] Configuration read from the .config file

    Keyword arguments:
        duration: [float] Time in seconds to capture. None by default.
        video_file: [str] Path to the video file, if it was given as the video source. None by default.
        nodetect: [bool] If True, detection will not be performed. False by defualt.
        detect_end: [bool] If True, detection will be performed at the end of the night, when capture 
            finishes. False by default.
        upload_manager: [UploadManager object] A handle to the UploadManager, which handles uploading files to
            the central server. None by default.

    """

    global STOP_CAPTURE


    # Create a directory for captured files
    night_data_dir_name = str(config.stationID) + '_' + datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')

    # Full path to the data directory
    night_data_dir = os.path.join(os.path.abspath(config.data_dir), config.captured_dir, night_data_dir_name)


    # Make a directory for the night
    mkdirP(night_data_dir)

    log.info('Data directory: ' + night_data_dir)



    # Load the default flat field image if it is available
    flat_struct = None

    if config.use_flat:

        # Check if the flat exists
        if os.path.exists(os.path.join(os.getcwd(), config.flat_file)):
            flat_struct = Image.loadFlat(os.getcwd(), config.flat_file)

            log.info('Loaded flat field image: ' + os.path.join(os.getcwd(), config.flat_file))



    # Load the default platepar if it is available
    platepar = None
    platepar_path = os.path.join(os.getcwd(), config.platepar_name)
    if os.path.exists(platepar_path):
        platepar = Platepar()
        platepar_fmt = platepar.read(platepar_path)

        log.info('Loaded platepar: ' + platepar_path)

    else:

        log.info('No platepar file found!')



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
        detector = QueuedPool(detectStarsAndMeteors, cores=1, log=log, delay_start=delay_detection)
        detector.startPool()

    
    # Initialize buffered capture
    bc = BufferedCapture(sharedArray, startTime, sharedArray2, startTime2, config, video_file=video_file)

    # Initialize the live image viewer
    live_view = LiveViewer(window_name='Maxpixel')
    
    # Initialize compression
    compressor = Compressor(night_data_dir, sharedArray, startTime, sharedArray2, startTime2, config, 
        detector=detector, live_view=live_view, flat_struct=flat_struct)

    
    # Start buffered capture
    bc.startCapture()

    # Start the compression
    compressor.start()

    
    # Capture until Ctrl+C is pressed
    wait(duration)
        
    # If capture was manually stopped, end capture
    if STOP_CAPTURE:
        log.info('Ending capture...')


    # Stop the capture
    log.debug('Stopping capture...')
    bc.stopCapture()
    log.debug('Capture stopped')

    dropped_frames = bc.dropped_frames
    log.info('Total number of dropped frames: ' + str(dropped_frames))


    # Stop the compressor
    log.debug('Stopping compression...')
    detector, live_view = compressor.stop()
    log.debug('Compression stopped')

    # Stop the live viewer
    log.debug('Stopping live viewer...')
    live_view.stop()
    del live_view
    log.debug('Live view stopped')


    # Init data lists
    star_list = []
    meteor_list = []
    ff_detected = []


    # If detection should be performed
    if not nodetect:

        log.info('Finishing up the detection, ' + str(detector.input_queue.qsize()) + ' files to process...')


        # Reset the Ctrl+C to KeyboardInterrupt
        resetSIGINT()


        try:

            # If there are some more files to process, process them on more cores
            if detector.input_queue.qsize() > 0:

                # Let the detector use all cores, but leave 1 free
                available_cores = multiprocessing.cpu_count() - 1


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

        # Remove all 'None' results, which were errors
        detection_results = [res for res in detection_results if res is not None]

        # Count the number of detected meteors
        meteors_num = 0
        for _, _, meteor_data in detection_results:
            for meteor in meteor_data:
                meteors_num += 1

        log.info('TOTAL: ' + str(meteors_num) + ' detected meteors.')


        # Save the detections to a file
        for ff_name, star_data, meteor_data in detection_results:

            x2, y2, background, intensity = star_data

            # Skip if no stars were found
            if not x2:
                continue

            # Construct the table of the star parameters
            star_data = zip(x2, y2, background, intensity)

            # Add star info to the star list
            star_list.append([ff_name, star_data])

            # Handle the detected meteors
            meteor_No = 1
            for meteor in meteor_data:

                rho, theta, centroids = meteor

                # Append to the results list
                meteor_list.append([ff_name, meteor_No, rho, theta, centroids])
                meteor_No += 1


            # Add the FF file to the archive list if a meteor was detected on it
            if meteor_data:
                ff_detected.append(ff_name)


        # Generate the name for the CALSTARS file
        calstars_name = 'CALSTARS_' + "{:s}".format(str(config.stationID)) + '_' \
            + os.path.basename(night_data_dir) + '.txt'

        # Write detected stars to the CALSTARS file
        CALSTARS.writeCALSTARS(star_list, night_data_dir, calstars_name, config.stationID, config.height, \
            config.width)

        # Generate FTPdetectinfo file name
        ftpdetectinfo_name = 'FTPdetectinfo_' + os.path.basename(night_data_dir) + '.txt'

        # Write FTPdetectinfo file
        FTPdetectinfo.writeFTPdetectinfo(meteor_list, night_data_dir, ftpdetectinfo_name, night_data_dir, \
            config.stationID, config.fps)


        # Run calibration check and auto astrometry refinement
        if platepar is not None:

            # Read in the CALSTARS file
            calstars_list = CALSTARS.readCALSTARS(night_data_dir, calstars_name)

            # Run astrometry check and refinement
            platepar, fit_status = autoCheckFit(config, platepar, calstars_list)

            # If the fit was sucessful, apply the astrometry to detected meteors
            if fit_status:

                log.info('Astrometric calibration SUCCESSFUL!')

                # Save the refined platepar to the night directory and as default
                platepar.write(os.path.join(night_data_dir, config.platepar_name), fmt=platepar_fmt)
                platepar.write(platepar_path, fmt=platepar_fmt)

            else:
                log.info('Astrometric calibration FAILED!, Using old platepar for calibration...')    


            # Calculate astrometry for meteor detections
            applyAstrometryFTPdetectinfo(night_data_dir, ftpdetectinfo_name, platepar_path)



    log.info('Plotting field sums...')

    # Plot field sums to a graph
    plotFieldsums(night_data_dir, config)

    # Archive all fieldsums to one archive
    archiveFieldsums(night_data_dir)


    # List for any extra files which will be copied to the night archive directory. Full paths have to be 
    #   given
    extra_files = []

    log.info('Making a flat...')

    # Make a new flat field
    flat_img = makeFlat(night_data_dir, config)

    # If making flat was sucessfull, save it
    if flat_img is not None:

        # Save the flat in the root directory, to keep the operational flat updated
        scipy.misc.imsave(config.flat_file, flat_img)
        flat_path = os.path.join(os.getcwd(), config.flat_file)
        log.info('Flat saved to: ' + flat_path)

        # Copy the flat to the night's directory as well
        extra_files.append(flat_path)

    else:
        log.info('Making flat image FAILED!')


    # Add the platepar to the archive if it exists
    if os.path.exists(platepar_path):
        extra_files.append(platepar_path)


    night_archive_dir = os.path.join(os.path.abspath(config.data_dir), config.archived_dir, 
        night_data_dir_name)


    log.info('Archiving detections to ' + night_archive_dir)
    
    # Archive the detections
    archive_name = archiveDetections(night_data_dir, night_archive_dir, ff_detected, config, \
        extra_files=extra_files)


    # Put the archive up for upload
    if upload_manager is not None:
        log.info('Adding file on upload list: ' + archive_name)
        upload_manager.addFiles([archive_name])


    # If capture was manually stopped, end program
    if STOP_CAPTURE:

        log.info('Ending program')

        # Stop the upload manager
        if upload_manager is not None:
            if upload_manager.is_alive():
                upload_manager.stop()
                log.info('Closing upload manager...')

        sys.exit()





if __name__ == "__main__":

    # Load the configuration file
    config = cr.parse(".config")


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description=""" Starting capture and compression.
        """)

    # Add a mutually exclusive for the parser (the arguments in the group can't be given at the same)
    arg_group = arg_parser.add_mutually_exclusive_group()

    arg_group.add_argument('-d', '--duration', metavar='DURATION_HOURS', help="""Start capturing right away, 
        with the given duration in hours. """)
    arg_group.add_argument('-i', '--input', metavar='FILE_PATH', help="""Use video from the given file, 
        not from a video device. """)

    arg_parser.add_argument('-n', '--nodetect', action="store_true", help="""Do not perform star extraction 
        nor meteor detection. """)

    arg_parser.add_argument('-e', '--detectend', action="store_true", help="""Detect stars and meteors at the
        end of the night, after capture finishes. """)

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    ######


    # Initialize the logger
    initLogging()

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
            detect_end=cml_args.detectend)

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
        runCapture(config, video_file=cml_args.input, nodetect=cml_args.nodetect)


    upload_manager = None
    if config.upload_enabled:

        # Init the upload manager
        log.info('Starting the upload manager...')
        upload_manager = UploadManager(config)
        upload_manager.start()


    # Automatic running and stopping the capture at sunrise and sunset
    while True:
            
        # Calculate when and how should the capture run
        start_time, duration = captureDuration(config.latitude, config.longitude, config.elevation)

        log.info('Next start time: ' + str(start_time) + ' UTC')

        # Don't start the capture if there's less than 15 minutes left
        if duration < 15*60:
            
            log.debug('Less than 15 minues left to record, waiting new recording session...')
            
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
            
            # Calculate how many seconds to wait until capture starts, and with for that time
            time_now = datetime.datetime.utcnow()
            waiting_time = start_time - time_now

            log.info('Waiting ' + str(waiting_time) + ' to start recording for ' + str(duration/60/60) \
                + ' hours')

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

                sys.exit()

            # Change the Ctrl+C action to the special handle
            setSIGINT()


        # Break the loop if capturing was stopped
        if STOP_CAPTURE:
            break



        log.info('Freeing up disk space...')
        
        # Free up disk space by deleting old files, if necessary
        if not deleteOldObservations(config.data_dir, config.captured_dir, config.archived_dir, config, 
            duration=duration):

            log.error('No more disk space can be freed up! Stopping capture...')
            break


        log.info('Starting capturing for ' + str(duration/60/60) + ' hours')

        # Run capture and compression
        runCapture(config, duration=duration, nodetect=cml_args.nodetect, upload_manager=upload_manager, 
            detect_end=cml_args.detectend)


    if upload_manager is not None:

        # Stop the upload manager
        if upload_manager.is_alive():
            log.debug('Closing upload manager...')
            upload_manager.stop()
            del upload_manager