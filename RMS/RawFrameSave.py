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


import os
import sys
import traceback
import time
import logging
import multiprocessing
from math import floor
import json

import cv2

# Get the logger from the main module
log = logging.getLogger("logger")


class RawFrameSaver(multiprocessing.Process):
    """Save list of numpy arrays (raw video frames).
    """

    running = False
    
    def __init__(self, saved_frames_dir, array1, startTime1, array2, startTime2, config):
        """

        Arguments:
            saved_frames_dir: directory to save raw frames to
            array1: first numpy array in shared memory of raw video frames
            startTime1: float in shared memory that holds time of first raw frame in array1
            array2: second numpy array in shared memory
            startTime1: float in shared memory that holds time of first raw frame in array2
            config: configuration class

        """
        
        super(RawFrameSaver, self).__init__()
        
        self.saved_frames_dir = saved_frames_dir
        self.array1 = array1
        self.startTime1 = startTime1
        self.array2 = array2
        self.startTime2 = startTime2
        self.config = config

        self.exit = multiprocessing.Event()
        self.run_exited = multiprocessing.Event()


    def saveFrameToDisk(self, frame_array, frames_start_time, block_iteration, json_file_path):
        """Saves a block of raw image frames to disk with timestamp-based filenames.

        This method calculates each filename using station ID, the UTC date
        and time from the timestamp, and the milliseconds part of the timestamp
        to ensure uniqueness, and then saves the frames in the specified format
        to the saved_frames_dir/frame_subdir directory.

        The filename format is 'stationID_YYYYMMDD_HHMMSS_MMM.ttt', where 'ttt' is
        either the jpg or png file type.

        Arguments
        ---------
            frame_array : [ndarray] numpy array of saved frames block
            frames_start_time: [float] time of first frame in frame_array
            block_iteration: [int] number of iteration of calling this function, used to track total frame number
            json_file_path: [str] json file path to write (frame number, timestamp) pairs to
        """

        block_frame_index = 0
        block_json_data = {}

        for frame in frame_array:

            # Generate the name for the file
            date_string = time.strftime(
                            "%Y%m%d_%H%M%S", 
                            time.gmtime(frames_start_time + (block_frame_index * self.config.frame_save_interval * (1 / self.config.fps)))
                        )

            # Calculate milliseconds (TEST)
            millis = int((frames_start_time - floor(frames_start_time))*1000)

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

            # write timestamps to temporary dictionary, to be then flushed to json
            block_json_data[block_iteration * len(frame_array) + block_frame_index] = "{0}_{1:03d}".format(date_string, millis)

            # Define the full path for saving the file
            frame_path = os.path.join(self.saved_frames_dir, self.config.frame_subdir, filename)

            # Write the image file
            try:
                if file_extension == '.png':
                    cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_PNG_COMPRESSION), self.config.png_compression])

                else:
                    cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpgs_quality])

                log.info("Frame saved: {0}".format(filename))

            except Exception as e:
                log.error("Could not save frame to disk: {0}".format(e))
            
            block_frame_index += 1


        # flush json file
        with open(json_file_path, 'r') as json_file:
            present_json_data = json.load(json_file)

        present_json_data.update(block_json_data)

        with open(json_file_path, 'w') as json_file:
            json.dump(present_json_data, json_file, indent=4)

        log.info(f"Updated {json_file_path} with current block's frame timestamps")



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

        # iterator to keep track of total saved frames
        block_iteration = 0

        # setup sidecar json file to write to (frame, timestamp) pairs to
        json_file_path = os.path.join(self.saved_frames_dir, "framestamps.json")
        with open(json_file_path, 'w') as json_file:
            json.dump({}, json_file, indent=4)

        log.info(f"Initialized {json_file_path} with an empty dictionary.") 

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

                # Copy raw frames
                raw_frames = self.array1
                raw_buffer_one = True

            elif self.startTime2.value > 0:

                # Retrieve time of first frame
                startTime = float(self.startTime2.value)

                # Copy frames
                raw_frames = self.array2
                raw_buffer_one = False

            else:

                # Wait until data is available
                log.debug("Raw frame saver waiting for frames...")
                time.sleep(0.1)
                continue
            
            log.debug("Saving raw frame block with start time at: {:s}".format(str(startTime)))

            t = time.time()

            # Run the frame block save
            self.saveFrameToDisk(raw_frames, startTime, block_iteration, json_file_path)

            # Once the frame saving is done, tell the capture thread to keep filling the buffer
            if raw_buffer_one:
                self.startTime1.value = 0
            else:
                self.startTime2.value = 0

            block_iteration += 1

            log.debug("Raw frame block saving time: {:.3f} s".format(time.time() - t))

        log.debug('Raw frame saver run exit')
        time.sleep(1.0)
        self.run_exited.set()


