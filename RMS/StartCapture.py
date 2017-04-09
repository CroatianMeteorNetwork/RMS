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

import sys
import argparse
import time
import datetime
import signal
import ctypes
import logging
import logging.handlers
import numpy as np
from multiprocessing import Array, Value

import RMS.ConfigReader as cr
from RMS.BufferedCapture import BufferedCapture
from RMS.Compression import Compressor
from RMS.CaptureDuration import captureDuration


# Flag indicating that capturing should be stopped
STOP_CAPTURE = False

def breakHandler(signum, frame):
    """ Handles what happens when Ctrl+C is pressed. """
        
    global STOP_CAPTURE

    # Set the flag to step capturing video
    STOP_CAPTURE = True


# The breakHandler function will be called when Ctrl+C is pressed
signal.signal(signal.SIGINT, breakHandler)



def wait(time_sec=None):
    """ The function will wait for the specified time, or it will stop when Enter is pressed. If no time was
        given (in seconds), it will wait until Enter is pressed. 

    Arguments:
        time_sec: [float] time in seconds to wait

    """

    global STOP_CAPTURE

    
    print('Press Ctrl+C to stop capturing...')

    # Get the time of capture start
    time_start = datetime.datetime.now()

    
    while True:

        # Sleep for a short interval
        time.sleep(0.1)

        # If some wait time was given, check if it passed
        if time_sec is not None:

            time_elapsed = (datetime.datetime.now() - time_start).total_seconds()

            # If the total time is elapsed, break the wait
            if time_elapsed >= time_sec:
                break


        if STOP_CAPTURE:
            break

    STOP_CAPTURE = False




def runCapture(duration=None):
    """ Run capture and compression for the given time.given

    Arguments:
        duration: [float] Time in seconds to capture. None by default.

    """

    # Init arrays for parallel compression on 2 cores
    sharedArrayBase = Array(ctypes.c_uint8, 256*config.width*config.height)
    sharedArray = np.ctypeslib.as_array(sharedArrayBase.get_obj())
    sharedArray = sharedArray.reshape(256, config.height, config.width)
    startTime = Value('d', 0.0)
    
    sharedArrayBase2 = Array(ctypes.c_uint8, 256*config.width*config.height)
    sharedArray2 = np.ctypeslib.as_array(sharedArrayBase2.get_obj())
    sharedArray2 = sharedArray2.reshape(256, config.height, config.width)
    startTime2 = Value('d', 0.0)
    
    # Initialize buffered capture
    bc = BufferedCapture(sharedArray, startTime, sharedArray2, startTime2, config)
    
    # Initialize compression
    c = Compressor(sharedArray, startTime, sharedArray2, startTime2, config)
    
    # Start buffered capture
    bc.startCapture()

    # Start the compression
    c.start()
    
    # Capture until Ctrl+C is pressed
    wait(duration)
    
    # Stop the capture
    bc.stopCapture()
    c.stop()




if __name__ == "__main__":

    # Load the configuration file
    config = cr.parse(".config")


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description=""" Starting the capture and compression.
        """)

    arg_parser.add_argument('-d', '--duration', metavar='DURATION_HOURS', help="""Start capturing right away, 
        with the given duration in hours. """)

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    ######


    # log_file_name = "log_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S.%f') + ".log"

    # # Initialize logging
    # logging.basicConfig(filename=log_file_name, level=logging.DEBUG)

    # logging.debug("########## START #########")
    # logging.debug("Start time:" + str(datetime.datetime.now()))

    log_file_name = "log.log"
        
    # Init logging
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    # Make a new log file each day
    handler = logging.handlers.TimedRotatingFileHandler(log_file_name, when='D', interval=1) #Log to a different file each day
    handler.setLevel(logging.INFO)

    # Set the log formatting
    formatter = logging.Formatter(fmt='%(asctime)s-%(levelname)s-%(module)s-line:%(lineno)d - %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    handler.setFormatter(formatter)
    log.addHandler(handler)

    # Stream all logs to stdout as wll
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s-%(levelname)s: %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    ch.setFormatter(formatter)
    log.addHandler(ch)

    log.info("Program start")


    # If the duration of capture was given, capture right away
    if cml_args.duration:

        try:
            duration = float(cml_args.duration)

        except:
            print('Given duration is not a proper number of hours!')


        log.info("Running for " + str(duration) + ' hours...')
        
        # Run the capture for the given number of hours
        runCapture(duration*60*60)

        sys.exit()


    # Automatic running and stopping the capture at sunrise and sunset
    while True:
            
        # Calculate when and how should the capture run
        start_time, duration = captureDuration(config.latitude, config.longitude, config.elevation)

        log.info('Next start time: ' + str(start_time))

        # Don't start the capture if there's less than 15 minutes left
        if duration < 15*60:
            
            log.debug('Less than 15 minues left to record, waiting new recording session...')
            
            # Wait for 30 mins before checking again
            time.sleep(30*60)

            continue


        # Wait to start capturing
        if start_time != True:
            
            # Calculate how many seconds to wait until capture starts, and with for that time
            time_now = datetime.datetime.now()
            waiting_time = start_time - time_now

            log.info('Waiting ' + str(waiting_time) + ' to start recording')

            # Wait until sunset
            time.sleep(int(waiting_time.total_seconds()))


        # Run capture and compression
        runCapture(duration)



    
