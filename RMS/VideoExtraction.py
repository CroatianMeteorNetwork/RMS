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



import math
import time
import logging
from multiprocessing import Process, Event

import numpy as np

from RMS.DetectionTools import loadImageCalibration, binImageCalibration
from RMS.Routines import Grouping3D
from RMS.Routines.MaskImage import maskImage
from RMS.Formats import FRbin


# Get the logger from the main module
log = logging.getLogger("logger")


class Extractor(Process):
    """ Detects fireballs and brighter meteors on the FF files, and extracts raw frames while they are still
        in memory. 

    """

    def __init__(self, config, data_dir):
        """
        Arguments:
            config: [Configuration object] config obj
            data_dir: [str] path to the directory where FF files are located

        """

        super(Extractor, self).__init__()
        
        self.config = config
        self.data_dir = data_dir

        # Load the calibration files (only the mask is used currently)
        self.mask, self.dark, self.flat_struct = loadImageCalibration(self.data_dir, self.config)

        # Bin the calibration images
        self.mask, self.dark, self.flat_struct = binImageCalibration(self.config, self.mask, self.dark, \
            self.flat_struct)


    
    def findPoints(self):
        """Threshold and subsample frames and return as list of points.
        
        Return:
            (y, x, z): [tuple] Coordinates of points that form the fireball, where Z is the frame number.
        """

        # Threshold and subsample frames
        length, x, y, z = Grouping3D.thresholdAndSubsample(self.frames, self.compressed, \
            self.config.min_level, self.config.min_pixels, self.config.k1, self.config.j1, self.config.f)


        # Return empty list if there is no points
        if length == 0:
            return []

        
        # Return list with the number of occurence of each frame number (Z axis)
        freq = np.array(np.unique(z, return_counts=True)).T
        

        # Reject the image if there are too little event frames
        if len(freq) <= self.config.min_frames:
            return []
        

        # Calculate a threshold based on factors and median number of points on the images per frame
        outlier_threshold = self.config.max_per_frame_factor * np.median(freq[:, 1])
        if outlier_threshold > self.config.max_points_per_frame:
            outlier_threshold = self.config.max_points_per_frame
        

        # Remove all outliers (aka. frames with a strong flare)
        for frameNum, count in freq:
            if count >= outlier_threshold:
                indices = np.where(z != frameNum)
                y = y[indices]
                x = x[indices]
                z = z[indices]
                

        # Randomize points if there are too many in total
        if len(z) > self.config.max_points:
            indices = np.random.choice(len(z), self.config.max_points, replace=False)
            y = y[indices]
            x = x[indices]
            z = z[indices]
        

        # Sort by frame number
        indices = np.argsort(z)
        y = y[indices]
        x = x[indices]
        z = z[indices]
        
        # Do a quick test to check if the points form a viable solution
        if not self.testPoints(y, x, z):
            return []
        
        # Convert to python list
        event_points = np.squeeze(np.dstack((y, x, z))).tolist()
        
        return event_points
    


    def testPoints(self, pointsy, pointsx, pointsz):
        """ Quick test if points are interesting (ie. something is detected).

        Arguments:
            pointsy: 1D numpy array with Y coords of points
            pointsx: 1D numpy array with X coords of points
            pointsz: 1D numpy array with Z coords of points (aka. frame number)
        
        Return:
            [bool] True if video should be further checked for fireballs, False otherwise
        """
        
        # Check if there are enough points
        if(len(pointsy) < self.config.min_points):
            return False

        # Check how many points are close to each other (along the time line)
        count = Grouping3D.testPoints(self.config.gap_threshold, pointsy, pointsx, pointsz)
        
        return count >= self.config.min_points
    


    def extract(self, coefficients):
        """ Determinate window size and crop out frames.
        
        Arguments:
            coefficients: [list] linear coefficients for each detected meteor
        
        Return:
            clips: [list] Cropped frames in format [frames, size and position]
        """
        
        clips = []
        
        # [first point, slope of XZ, slope of YZ, first frame, last frame]
        for point, slopeXZ, slopeYZ, firstFrame, lastFrame in coefficients:

            slopeXZ = float(slopeXZ)
            slopeYZ = float(slopeYZ)

            firstFrame = int(firstFrame)
            lastFrame = int(lastFrame)
            
            diff = lastFrame - firstFrame

            
            # Extrapolate before first detected point
            before_frames = math.ceil(diff*self.config.before)

            # Make sure at least 4 frames before are taken
            if before_frames < 4:
                before_frames = 4

            firstFrame = firstFrame - before_frames

            if firstFrame < 0:
                firstFrame = 0


            # Extrapolate after last detected point
            after_frames = math.ceil(diff*self.config.after) 

            # Make sure at least 4 frames after are taken
            if after_frames < 4:
                after_frames = 4

            lastFrame = lastFrame + after_frames

            if lastFrame >= self.frames.shape[0]:
                lastFrame = self.frames.shape[0] - 1
                

            # Cut of the fireball from raw video frames
            length, cropouts, sizepos = Grouping3D.detectionCutOut(self.frames, self.compressed, 
                point.astype(np.uint16), slopeXZ, slopeYZ, firstFrame, lastFrame, self.config.f, \
                self.config.limitForSize, self.config.minSize, self.config.maxSize)
            
            # Shorten the output arrays to their maximum sizes
            cropouts = cropouts[:length]
            sizepos = sizepos[:length]
            
            clips.append([cropouts, sizepos])

        
        return clips
    


    def save(self, clips):
        """ Save extracted clips to FR*.bin file.

        Arguments:
            clips: [list] a list of extracted clips of the fireball
        """
        
        FRbin.writeArray(clips, self.data_dir, self.filename)
    


    def stop(self):
        """ Stop the extractor.
        """
        
        self.exit.set()
        self.join()
        


    def start(self, frames, compressed, filename):
        """ Start the extractor.

        Arguments:
            frame: [int] frame number
            compressed: [ndarray] array with FTP compressed frames
            filename: [str] name of the FF file which is being processed

        """
        
        self.exit = Event()
        
        self.frames = frames
        self.compressed = compressed
        self.filename = filename
        
        super(Extractor, self).start()
    


    def run(self):
        """ Retrieve frames from list, convert, compress and save them.
        """
        
        self.executeAll()
    


    def executeAll(self):
        """ Run the complete extraction procedure. """

        # Apply the mask to the compressed frames (maxpixel, avepixel)
        if self.mask is not None:
            self.compressed[0] = maskImage(self.compressed[0], self.mask)
            self.compressed[2] = maskImage(self.compressed[2], self.mask)
        

        # Check if the average image is too white and skip it
        if np.average(self.compressed[2]) > self.config.white_avg_level:
            log.debug("[" + self.filename + "] frames are all white")
            return
        
        t = time.time()

        # Extract points of the fireball
        event_points = self.findPoints()
        log.debug("[" + self.filename + "] time for thresholding and subsampling: " + str(time.time() - t) + "s")
        
        # Skip the event if not points where found
        if len(event_points) == 0:
            log.debug("[" + self.filename + "] nothing found, not extracting anything 1")
            return
        
        t = time.time()

        # Find lines in 3D space and store them to line_list
        line_list = Grouping3D.find3DLines(event_points, time.time(), self.config)
        log.debug("[" + self.filename + "] Time for finding lines: " + str(time.time() - t) + "s")
        
        if line_list is None:
            log.debug("[" + self.filename + "] no lines found, not extracting anything")
            return
        
        t = time.time()

        # Find the parameters of the line in the 3D point cloud
        coeff = Grouping3D.findCoefficients(line_list)
        log.debug("[" + self.filename + "] Time for finding coefficients: " + str(time.time() - t) + "s")
        
        if len(coeff) == 0:
            log.debug("[" + self.filename + "] nothing found, not extracting anything 2")
            return
        
        t = time.time()

        # Extract video clips from the raw data
        clips = self.extract(coeff)
        log.debug("[" + self.filename + "] Time for extracting: " + str(time.time() - t) + "s")

         
        t = time.time()

        # Save the extracted clips
        self.save(clips)

        log.debug("[" + self.filename + "] Time for saving: " + str(time.time() - t) + "s")