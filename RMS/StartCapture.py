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
import logging.handlers
import numpy as np
import multiprocessing

import RMS.ConfigReader as cr

from RMS.Formats import FTPdetectinfo
from RMS.Formats import CALSTARS

from RMS.DeleteOldObservations import deleteOldObservations
from RMS.BufferedCapture import BufferedCapture
from RMS.Compression import Compressor
from RMS.CaptureDuration import captureDuration
from RMS.DetectStarsAndMeteors import detectStarsAndMeteors
from RMS.ArchiveDetections import archiveDetections

from RMS.LiveViewer import LiveViewer
from RMS.QueuedPool import QueuedPool
from RMS.Misc import mkdirP


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
    time_start = datetime.datetime.now()

    
    while True:

        # Sleep for a short interval
        time.sleep(0.1)

        # If some wait time was given, check if it passed
        if duration is not None:

            time_elapsed = (datetime.datetime.now() - time_start).total_seconds()

            # If the total time is elapsed, break the wait
            if time_elapsed >= duration:
                break


        if STOP_CAPTURE:
            break




def runCapture(config, duration=None, video_file=None):
    """ Run capture and compression for the given time.given

    Arguments:
        config: [config object] Configuration read from the .config file
        duration: [float] Time in seconds to capture. None by default.
        video_file: [str] Path to the video file, if it was given as the video source. None by default.

    """

    global STOP_CAPTURE


    # Create a directory for captured files
    night_data_dir_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    # Full path to the data directory
    night_data_dir = os.path.join(os.path.abspath(config.data_dir), config.captured_dir, night_data_dir_name)

    # Make a directory for the night
    mkdirP(night_data_dir)

    log.info('Data directory: ' + night_data_dir)


    # Init arrays for parallel compression on 2 cores
    sharedArrayBase = multiprocessing.Array(ctypes.c_uint8, 256*config.width*config.height)
    sharedArray = np.ctypeslib.as_array(sharedArrayBase.get_obj())
    sharedArray = sharedArray.reshape(256, config.height, config.width)
    startTime = multiprocessing.Value('d', 0.0)
    
    sharedArrayBase2 = multiprocessing.Array(ctypes.c_uint8, 256*config.width*config.height)
    sharedArray2 = np.ctypeslib.as_array(sharedArrayBase2.get_obj())
    sharedArray2 = sharedArray2.reshape(256, config.height, config.width)
    startTime2 = multiprocessing.Value('d', 0.0)


    # Initialize the detector
    detector = QueuedPool(detectStarsAndMeteors, cores=1)
    
    # Initialize buffered capture
    bc = BufferedCapture(sharedArray, startTime, sharedArray2, startTime2, config, video_file=video_file)

    # Initialize the live image viewer
    live_view = LiveViewer(window_name='Maxpixel')
    
    # Initialize compression
    c = Compressor(night_data_dir, sharedArray, startTime, sharedArray2, startTime2, config, 
        detector=detector, live_view=live_view)

    
    # Start buffered capture
    bc.startCapture()

    # Start the compression
    c.start()

    
    # Capture until Ctrl+C is pressed
    wait(duration)
        
    # If capture was manually stopped, end capture
    if STOP_CAPTURE:
        log.info('Ending capture...')


    # Stop the capture
    log.debug('Stopping capture...')
    bc.stopCapture()
    log.debug('Capture stopped')

    # Stop the compressor
    log.debug('Stopping compression...')
    detector, live_view = c.stop()
    log.debug('Compression stopped')

    # Stop the live viewer
    log.debug('Stopping live viewer...')
    live_view.stop()
    log.debug('Live view stopped')


    log.info('Finishing up the detection, ' + str(detector.input_queue.qsize()) + ' files to process...')

    # If there are some more files to process, preocess them on more cores
    if not detector.allDone():

        # Let the detector use all cores, but leave 1 free
        available_cores = multiprocessing.cpu_count()
        if available_cores > 1:
            detector.updateCoreNumber(available_cores - 1)


    # Reset the Ctrl+C to KeyboardInterrupt
    resetSIGINT()

    log.info('Closing the detection thread...')

    # Wait for the detector to finish and close it
    detector.closePool()

    log.info('Detection finished!')

    # Set the Ctrl+C back to 'soft' program kill
    setSIGINT()

    ### SAVE DETECTIONS TO FILE

    # Init data lists
    star_list = []
    meteor_list = []
    ff_detected = []


    log.info('Collecting results...')

    # Get the detection results from the queue
    detection_results = detector.getResults()

    # Count the number of detected meteors
    meteors_num = 0
    for _, _, meteor_data in detection_results:
        for meteor in meteor_data:
            meteors_num += 1

    log.info(str(meteors_num) + ' detected meteors.')


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


        ff_detected.append(ff_name)


    # Generate the name for the CALSTARS file
    calstars_name = 'CALSTARS' + "{:04d}".format(config.stationID) + os.path.basename(night_data_dir) + '.txt'

    # Write detected stars to the CALSTARS file
    CALSTARS.writeCALSTARS(star_list, night_data_dir, calstars_name, config.stationID, config.height, 
        config.width)

    # Generate FTPdetectinfo file name
    ftpdetectinfo_name = os.path.join(night_data_dir, 
        'FTPdetectinfo_' + os.path.basename(night_data_dir) + '.txt')

    # Write FTPdetectinfo file
    FTPdetectinfo.writeFTPdetectinfo(meteor_list, night_data_dir, ftpdetectinfo_name, night_data_dir, 
        config.stationID, config.fps)



    night_archive_dir = os.path.join(os.path.abspath(config.data_dir), config.archived_dir, 
        night_data_dir_name)


    log.info('Archiving detections to ' + night_archive_dir)
    
    # Archive the detections
    archiveDetections(night_data_dir, night_archive_dir, ff_detected)

    ######


    # If capture was manually stopped, end program
    if STOP_CAPTURE:

        log.info('Ending program')
        sys.exit()





if __name__ == "__main__":

    # Load the configuration file
    config = cr.parse(".config")


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description=""" Starting the capture and compression.
        """)

    arg_parser.add_argument('-d', '--duration', metavar='DURATION_HOURS', help="""Start capturing right away, 
        with the given duration in hours. """)

    arg_parser.add_argument('-i', '--input', metavar='FILE_PATH', help="""Use video from the given file, 
        not from a video device. """)

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    ######

    ### LOGGING SETUP

    log_file_name = "log_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S.%f') + ".log"
        
    # Init logging
    log = logging.getLogger('logger')
    log.setLevel(logging.INFO)
    log.setLevel(logging.DEBUG)

    # Make a new log file each day
    handler = logging.handlers.TimedRotatingFileHandler(log_file_name, when='D', interval=1) 
    handler.setLevel(logging.INFO)
    handler.setLevel(logging.DEBUG)

    # Set the log formatting
    formatter = logging.Formatter(fmt='%(asctime)s-%(levelname)s-%(module)s-line:%(lineno)d - %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    handler.setFormatter(formatter)
    log.addHandler(handler)

    # Stream all logs to stdout as wll
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s-%(levelname)s-%(module)s-line:%(lineno)d - %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    ch.setFormatter(formatter)
    log.addHandler(ch)

    ######


    log.info('Program start')

    # Change the Ctrl+C action to the special handle
    setSIGINT()


    # If the duration of capture was given, capture right away
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



        log.info("Running for " + str(duration/60/60) + ' hours...')

        # Run the capture for the given number of hours
        runCapture(config, duration=duration)

        sys.exit()



    # If a file with video input was give, use it as a video source
    if cml_args.input:

        log.info('Video source: ' + cml_args.input)

        # Capture the video frames from the video file
        runCapture(config, video_file=cml_args.input)



    # Automatic running and stopping the capture at sunrise and sunset
    while True:
            
        # Calculate when and how should the capture run
        start_time, duration = captureDuration(config.latitude, config.longitude, config.elevation)

        log.info('Next start time: ' + str(start_time))

        # Don't start the capture if there's less than 15 minutes left
        if duration < 15*60:
            
            log.debug('Less than 15 minues left to record, waiting new recording session...')
            
            # Reset the Ctrl+C to KeyboardInterrupt
            resetSIGINT()

            # Wait for 30 mins before checking again
            time.sleep(30*60)

            # Change the Ctrl+C action to the special handle
            setSIGINT()

            continue


        # Wait to start capturing
        if start_time != True:
            
            # Calculate how many seconds to wait until capture starts, and with for that time
            time_now = datetime.datetime.now()
            waiting_time = start_time - time_now

            log.info('Waiting ' + str(waiting_time) + ' to start recording for ' + str(duration/60/60) \
                + ' hours')

            # Reset the Ctrl+C to KeyboardInterrupt
            resetSIGINT()

            # Wait until sunset
            time.sleep(int(waiting_time.total_seconds()))

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
            sys.exit()


        log.info('Starting capturing for ' + str(duration/60/60) + ' hours')

        # Run capture and compression
        runCapture(config, duration=duration)



    
