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
import time
import logging
import multiprocessing
from math import floor

import numpy as np
import cv2


from RMS.VideoExtraction import Extractor
from RMS.Formats import FFfile, FFStruct
from RMS.Formats import FieldIntensities
from RMS.Routines.Image import saveImage

# Import Cython functions
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from RMS.CompressionCy import compressFrames


# Get the logger from the main module
log = logging.getLogger("logger")


class Compressor(multiprocessing.Process):
    """Compress list of numpy arrays (video frames).

        Output is in Four-frame Temporal Pixel (FTP) format. See the Jenniskens et al., 2011 paper about the
        CAMS project for more info.

    """

    running = False
    
    def __init__(self, data_dir, array1, startTime1, array2, startTime2, config, detector=None):
        """

        Arguments:
            array1: first numpy array in shared memory of grayscale video frames
            startTime1: float in shared memory that holds time of first frame in array1
            array2: second numpy array in shared memory
            startTime1: float in shared memory that holds time of first frame in array2
            config: configuration class

        Keyword arguments:
            detector: [Detector object] Handle to Detector object used for running star extraction and
                meteor detection.

        """
        
        super(Compressor, self).__init__()
        
        self.data_dir = data_dir
        self.array1 = array1
        self.startTime1 = startTime1
        self.array2 = array2
        self.startTime2 = startTime2
        self.config = config

        self.detector = detector

        self.exit = multiprocessing.Event()

        self.run_exited = multiprocessing.Event()
    


    def compress(self, frames):
        """ Compress frames to the FTP-compatible array and extract sums of intensities per every field.

        NOTE: The standard deviation calculation is performed in a non-standard way due to performance 
            concerns. The end result is the same as a proper calculation due to the usage of low-precision
            8-bit unsigned integers, so the difference does not matter.
        
        Arguments:
            frames: [3D ndarray] grayscale frames stored as 3d numpy array
        
        Return:
            [3D ndarray]: in format: (N, y, x) where N is a member of [0, 1, 2, 3]

        """
        
        # Run cythonized compression
        ftp_array, fieldsum = compressFrames(frames, self.config.deinterlace_order)

        return ftp_array, fieldsum
    


    def saveFF(self, arr, startTime, N):
        """ Write metadata and data array to FF file and return filenames for FF and FS files
        
        Arguments:
            arr: [3D ndarray] 3D numpy array in format: (N, y, x) where N is [0, 4)
            startTime: [float] seconds and fractions of a second from epoch to first frame
            N: [int] frame counter (ie. 0000512)
        """
        
        # Generate the name for the file
        date_string = time.strftime("%Y%m%d_%H%M%S", time.gmtime(startTime))

        # Calculate microseconds and milliseconds
        micros = int((startTime - floor(startTime))*1000000)
        millis = int((startTime - floor(startTime))*1000)
        

        filename_millis = str(self.config.stationID).zfill(3) +  "_" + date_string + "_" + str(millis).zfill(3) \
            + "_" + str(N).zfill(7)
        
        filename_micros = str(self.config.stationID).zfill(3) +  "_" + date_string + "_" + str(micros).zfill(6) \
            + "_" + str(N).zfill(7)

        ff = FFStruct.FFStruct()
        ff.array = arr
        ff.nrows = arr.shape[1]
        ff.ncols = arr.shape[2]
        ff.nbits = self.config.bit_depth
        ff.nframes = 256
        ff.first = N + 256
        ff.camno = self.config.stationID
        ff.fps = self.config.fps
        ff.starttime = date_string + "_" + str(micros).zfill(6)
        
        # Write the FF file
        FFfile.write(ff, self.data_dir, filename_millis, fmt=self.config.ff_format)
        
        return filename_millis, filename_micros


    def saveLiveJPG(self, array, startTime):
        """ Save a live.jpg file to the data directory with the latest compressed image. """


        # Name of the file
        live_name = 'live.jpg'

        # Generate the name for the file
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(startTime))

        maxpixel, _, _, _ = np.split(array, 4, axis=0)
        maxpixel = np.array(maxpixel[0])

        # Draw text to image
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = self.config.stationID + " " + timestamp + " UTC"
        cv2.putText(maxpixel, text, (10, maxpixel.shape[0] - 6), font, 0.4, (255, 255, 255), 1, \
            cv2.LINE_AA)

        # Save the labelled image to disk
        try:
            # Save the file to disk
            saveImage(os.path.join(self.config.data_dir, live_name), maxpixel)
        except:
            log.error("Could not save {:s} to disk!".format(live_name))
    


    def stop(self):
        """ Stop compression.
        """
        

        self.exit.set()
        log.debug('Compression exit flag set')

            
        log.debug('Joining compression...')


        t_beg = time.time()

        # Wait until everything is done
        while not self.run_exited.is_set():
            
            time.sleep(0.01)

            # Do not wait more than a minute, just terminate the compression thread then
            if (time.time() - t_beg) > 60:
                log.debug('Waited more than 60 seconds for compression to end, killing it...')
                break


        log.debug('Compression joined!')

        self.terminate()
        self.join()

        # Return the detector and live viewer objects because they were updated in this namespace
        return self.detector
    


    def start(self):
        """ Start compression.
        """
        
        super(Compressor, self).start()
    


    def run(self):
        """ Retrieve frames from list, convert, compress and save them.
        """
        
        n = 0
        
        # Repeat until the compressor is killed from the outside
        while not self.exit.is_set():

            # Block until frames are available
            while (self.startTime1.value == 0) and (self.startTime2.value == 0):

                # Exit function if process was stopped from the outside
                if self.exit.is_set():

                    log.debug('Compression run exit')
                    self.run_exited.set()

                    return None

                time.sleep(0.1)

                

            t = time.time()

            
            buffer_one = True
            if self.startTime1.value > 0:

                # Retrieve time of first frame
                startTime = float(self.startTime1.value)

                # Copy frames
                frames = self.array1

                # Tell the capture thread to wait until the compression is completed by setting this to -1
                self.startTime1.value = -1
                buffer_one = True

            elif self.startTime2.value > 0:

                # Retrieve time of first frame
                startTime = float(self.startTime2.value)

                # Copy frames
                frames = self.array2

                # Tell the capture thread to wait until the compression is completed
                self.startTime2.value = -1
                buffer_one = False

            else:

                # Wait until data is available
                log.debug("Compression waiting for frames...")
                time.sleep(0.1)
                continue

            
            log.debug("Compressing frame block with start time at: {:s}".format(str(startTime)))

            #log.debug("memory copy: " + str(time.time() - t) + "s")
            t = time.time()
            
            
            # Run the compression
            compressed, field_intensities = self.compress(frames)

            
            # Once the compression is done, tell the capture thread to keep filling the buffer
            if buffer_one:
                self.startTime1.value = 0

            else:
                self.startTime2.value = 0


            # Cut out the compressed frames to the proper size
            compressed = compressed[:, :self.config.height, :self.config.width]
            
            log.debug("Compression time: {:.3f} s".format(time.time() - t))
            t = time.time()
            
            # Save the compressed image
            filename_millis, filename_micros = self.saveFF(compressed, startTime, n*256)
            n += 1
            
            log.debug("Saving time: {:.3f} s".format(time.time() - t))


            # Save a live.jpg file to the data directory
            if self.config.live_jpg:
                log.debug("Saving live jpg")
                self.saveLiveJPG(compressed, startTime)


            # Save the extracted intensities per every field
            FieldIntensities.saveFieldIntensitiesBin(field_intensities, self.data_dir, filename_micros)

            # Run the extractor
            if self.config.enable_fireball_detection:
                extractor = Extractor(self.config, self.data_dir)
                extractor.start(frames, compressed, filename_millis)

                log.debug('Extractor started for: ' + filename_millis)


            # Fully format the filename (this could not have been done before as the extractor has to add
            # the FR prefix to the given file name)
            filename = "FF_" + filename_millis + "." + self.config.ff_format


            # Run the detection on the file, if the detector handle was given
            if self.detector is not None:

                # Add the file to the detector queue
                self.detector.addJob([self.data_dir, filename, self.config])
                log.info('Added file for detection: {:s}'.format(filename))



        log.debug('Compression run exit')
        time.sleep(1.0)
        self.run_exited.set()


