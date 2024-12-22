# RPi Meteor Station
# Copyright (C) 2015  Dario Zubovic
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

from __future__ import print_function, division, absolute_import

import os
import sys
import traceback
import time
import logging
import multiprocessing
from math import floor
import json

import cv2

from RMS.Misc import mkdirP

# Get the logger from the main module
log = logging.getLogger("logger")


class RawFrameSaver(multiprocessing.Process):
    """Save list of numpy arrays (raw video frames).
    """

    running = False
    
    def __init__(self, saved_frames_dir, array1, startTime1, array2, startTime2, tsArray1, tsArray2, config):
        """

        Arguments:
            saved_frames_dir: directory to save raw frames to
            array1: first numpy array in shared memory of raw video frames
            startTime1: float in shared memory that holds time of first raw frame in array1
            array2: second numpy array in shared memory
            startTime1: float in shared memory that holds time of first raw frame in array2
            tsArray1: first numpy array in shared memory for timestamps
            tsArray2: second numpy array in shared memory for timestamps
            config: configuration class

        """
        
        super(RawFrameSaver, self).__init__()
        
        self.saved_frames_dir = saved_frames_dir
        self.array1 = array1
        self.startTime1 = startTime1
        self.array2 = array2
        self.startTime2 = startTime2
        self.timeStamps1 = tsArray1
        self.timeStamps2 = tsArray2
        self.config = config

        self.total_saved_frames = 0
        self.day_of_year = time.strftime("%j", time.gmtime())

        self.exit = multiprocessing.Event()
        self.run_exited = multiprocessing.Event()


    def initRawFrameCount(self):        
        """Initiates frame counter for the JSON file with (frame, timestamp) pairs, so that the frame count is
        consistent after a reboot
        """

        # Find sidecar json file for current day in year's directory
        json_subpath = time.strftime("%Y/{}_%Y%m%d-%j".format(self.config.stationID)) + "_frametimes.json"
        json_file_path = os.path.join(self.saved_frames_dir, json_subpath)


        # Get latest frame count in case file exists
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as json_file:
                json_data = json.load(json_file)
            self.total_saved_frames = max(map(int, json_data.keys())) + 1
            log.info("Continuing raw frame saving for today from frame #{}".format(self.total_saved_frames))


    def saveJsonTimestamps(self, block_timestamp, block_json_data) :
        """Produces/Updates a JSON file with (frame, timestamp) pairs

        Arguments
        ---------
            block_timestamp : [float] time of first frame in block
            block_json_data: [Dict] (frame, timestamp) mappings

        """

        # Sidecar json file to write to (frame, timestamp) pairs to year's directory
        json_subpath = time.strftime("%Y/{}_%Y%m%d-%j".format(self.config.stationID), time.gmtime(block_timestamp)) + "_frametimes.json"
        json_file_path = os.path.join(self.saved_frames_dir, json_subpath)

        # Create JSON on path if not present, else update file
        if not os.path.exists(json_file_path):
            present_json_data = block_json_data
            log.info("Initialized new json at {}.".format(json_file_path)) 

        else:
            with open(json_file_path, 'r') as json_file:
                present_json_data = json.load(json_file)
            present_json_data.update(block_json_data)

        with open(json_file_path, 'w') as json_file:
            json.dump(present_json_data, json_file, indent=4)

        log.info("Updated {} with {} new frame timestamps".format(json_file_path, len(block_json_data))) 



    def saveFramesToDisk(self, frametimes, block_start_time):
        """Saves a block of raw image frames to disk with timestamp-based filenames.

        This method calculates each filename using station ID, the UTC date
        and time from the timestamp, and the milliseconds part of the timestamp
        to ensure uniqueness, and then saves the frames in the specified format
        to the saved_frames_dir directory.

        Each file is stored in a path based on its time:
        saved_frames_dir/YYYY/YYYYMMDD-DoY/YYYYMMDD-DoY_HH/stationID_YYYYMMDD_HHMMSS_MMM.ttt

        Where 'DoY' is day of year and 'ttt' is either the jpg or png file type.

        Arguments
        ---------
            frametimes : [List] list of (frame, timestamp) pairs of corresponding frames and timestamps
            block_start_time: [float] time of first frame in frame_array
        """

        block_json_data = {}

        for (frame, timestamp) in frametimes:

            # If timestamp is 0, then we've reached the end and this is the last block 
            if timestamp == 0:
                break

            # In case the timestamp day changes mid-block
            if self.day_of_year != time.strftime("%j", time.gmtime(timestamp)):
                
                # Save block data from previous day
                self.saveJsonTimestamps(block_start_time, block_json_data)

                # Adjust values for the day change
                block_json_data = {}
                block_start_time = timestamp
                self.total_saved_frames = 0
                self.day_of_year = time.strftime("%j", time.gmtime(timestamp))


            # Generate names for the file and path
            date_string = time.strftime("%Y%m%d_%H%M%S", time.gmtime(timestamp))
            timed_dir_string = time.strftime("%Y/%Y%m%d-%j/%Y%m%d-%j_%H", time.gmtime(timestamp))

            # Calculate milliseconds
            millis = int((timestamp - floor(timestamp))*1000)

            # Create the filename
            if self.config.frame_file_type == 'png':
                file_extension = '.png'
            else:
                file_extension = '.jpg'

            filename = "{0}_{1}_{2:03d}{3}".format(
                str(self.config.stationID).zfill(3),
                date_string,
                millis,
                file_extension
            )

            # Full path for saving the file
            frame_dir_path = os.path.join(self.saved_frames_dir, timed_dir_string)
            mkdirP(frame_dir_path)
            frame_path = os.path.join(frame_dir_path, filename)

            # Write the image file
            try:
                if file_extension == '.png':
                    cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_PNG_COMPRESSION), self.config.png_compression])

                else:
                    cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpgs_quality])

                log.info("Frame saved: {0}".format(filename))

            except Exception as e:
                log.error("Could not save frame to disk: {0}".format(e))


            # Write timestamps to temporary dictionary, to be then saved to json
            block_json_data[self.total_saved_frames] = "{0}_{1:03d}".format(date_string, millis)
            self.total_saved_frames += 1


        self.saveJsonTimestamps(block_start_time, block_json_data)



    def stop(self):
        """ Stop saving frames.
        """

        self.exit.set()
        log.debug('Raw frame saver exit flag set')

        # Free shared memory after the raw frame saver is done
        try:
            log.debug('Freeing frame buffers in raw frame saver...')
            del self.array1
            del self.array2
            del self.timeStamps1
            del self.timeStamps2

        except Exception as e:
            log.debug('Freeing raw frame buffers failed with error:' + repr(e))
            log.debug(repr(traceback.format_exception(*sys.exc_info())))
    

    def start(self):
        """ Start raw frame saving.
        """
        
        super(RawFrameSaver, self).start()
    

    def run(self):
        """ Retrieve raw frames from shared array and save them.
        """

        self.initRawFrameCount()

        # Repeat until the raw frame saver is killed from the outside
        while not self.exit.is_set():

            # Block until the raw frames are available
            while (self.startTime1.value == 0) and (self.startTime2.value == 0):

                # Exit function if process was stopped from the outside
                if self.exit.is_set():

                    log.debug('Raw frame saver run exit')
                    self.run_exited.set()

                    return None

                time.sleep(0.1)

            raw_buffer_one = True

            if self.startTime1.value > 0:

                # Retrieve time of first frame
                startTime = float(self.startTime1.value)

                # Copy raw (frames, timestamps)
                # Clear out the timestamp array so it can be used by 
                # saveFramesToDisk to halt
                frametimes = list(zip(self.array1, self.timeStamps1))
                self.timeStamps1.fill(0)
                raw_buffer_one = True

            elif self.startTime2.value > 0:

                # Retrieve time of first frame
                startTime = float(self.startTime2.value)

                # Copy raw (frames, timestamps)
                # Clear out the timestamp array so it can be used by 
                # saveFramesToDisk to halt
                frametimes = list(zip(self.array2, self.timeStamps2))
                self.timeStamps2.fill(0)
                raw_buffer_one = False

            else:

                # Wait until data is available
                log.debug("Raw frame saver waiting for frames...")
                time.sleep(0.1)
                continue
            
            log.debug("Saving raw frame block with start time at: {:s}".format(str(startTime)))

            t = time.time()

            # Run the frame block save
            self.saveFramesToDisk(frametimes, startTime)

            # Once the frame saving is done, tell the capture thread to keep filling the buffer
            if raw_buffer_one:
                self.startTime1.value = 0
            else:
                self.startTime2.value = 0

            log.debug("Raw frame block saving time: {:.3f} s".format(time.time() - t))

        log.debug('Raw frame saver run exit')
        time.sleep(1.0)
        self.run_exited.set()


