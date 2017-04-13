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

from multiprocessing import Process, Event
import numpy as np
from scipy import weave, stats
from RMS.Routines import Grouping3D
from math import floor, sqrt, ceil
import time
import logging
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

    
    def findPoints(self):
        """Threshold and subsample frames and return as list of points.
        
        Return:
            (y, x, z): [tuple] Coordinates of points that form the fireball, where Z is the frame number.
        """
     
        count = np.zeros((self.frames.shape[0], floor(self.frames.shape[1]//self.config.f), floor(self.frames.shape[2]//self.config.f)), np.int)
        pointsy = np.empty((self.frames.shape[0]*floor(self.frames.shape[1]//self.config.f)*floor(self.frames.shape[2]//self.config.f)), np.uint16)
        pointsx = np.empty((self.frames.shape[0]*floor(self.frames.shape[1]//self.config.f)*floor(self.frames.shape[2]//self.config.f)), np.uint16)
        pointsz = np.empty((self.frames.shape[0]*floor(self.frames.shape[1]//self.config.f)*floor(self.frames.shape[2]//self.config.f)), np.uint16)
        
        code = """
        unsigned int x, y, x2, y2, n, i, max;
        unsigned int num = 0, acc = 0;
        unsigned int avg_std;
        
        for(y=0; y<Nframes[1]; y++) {
            for(x=0; x<Nframes[2]; x++) {
                max = COMPRESSED3(0, y, x);
                avg_std = COMPRESSED3(2, y, x) + k1*COMPRESSED3(3, y, x);
                
                //if((max > min_level) && (avg_std > 255 || max >= avg_std)) {
                if((max > min_level) && (max >= avg_std)) {
                    n = COMPRESSED3(1, y, x);
                    
                    // Subsample frame in f*f squares
                    y2 = y/f; 
                    x2 = x/f;
                    
                    // Check if there are enough of threshold passers inside of this square
                    if(COUNT3(n, y2, x2) >= min_points) { 
                        POINTSY1(num) = y2;
                        POINTSX1(num) = x2;
                        POINTSZ1(num) = n;
                        num++;

                        // Don't repeat this number
                        COUNT3(n, y2, x2) = -1;

                    // Increase counter if not enough threshold passers and this number isn't written already
                    } else if(COUNT3(n, y2, x2) != -1) { 
                        COUNT3(n, y2, x2) += 1;
                    }
                }
            }     
        }
        
        // Output the length of POINTS arrays
        return_val = num; 
        """
        
        dictionary = {'frames': self.frames, 'compressed': self.compressed, 'min_level': self.config.min_level,
                      'min_points': self.config.min_pixels, 'k1': self.config.k1, 'f': self.config.f,
                      'count': count, 'pointsy': pointsy, 'pointsx': pointsx, 'pointsz': pointsz}

        length = weave.inline(code, dictionary.keys(), dictionary, verbose=2, extra_compile_args=self.config.weaveArgs, extra_link_args=self.config.weaveArgs)
        

        # Return empty list if there is no points
        if length == 0:
            return []
        

        # Cut away extra long array
        y = pointsy[0 : length]
        x = pointsx[0 : length]
        z = pointsz[0 : length]
        
        # Return list with the number of occurence of each frame number (Z axis)
        freq = stats.itemfreq(z) 
        

        # Reject the image if there are too little event frames
        if len(freq) <= self.config.min_frames:
            return []
        

        # Calculate a threshold based on factors and median number of points on the images per frame
        outlier_treshold = self.config.max_per_frame_factor * np.median(freq[:, 1])
        if outlier_treshold > self.config.max_points_per_frame:
            outlier_treshold = self.config.max_points_per_frame
        

        # Remove all outliers (aka. frames with a strong flare)
        for frameNum, count in freq:
            if count >= outlier_treshold:
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
        

        # Sort by frame number and convert to float
        indices = np.argsort(z)
        y = y[indices].astype(np.float)
        x = x[indices].astype(np.float)
        z = z[indices].astype(np.float)
        
        # Do a quick test to check if the points form a viable solution
        if not self.testPoints(y, x, z):
            return []
        
        # Convert to python list
        event_points = np.squeeze(np.dstack((y, x, z))).tolist()
        
        return event_points
    


    def testPoints(self, y, x, z):
        """ Quick test if points are interesting (ie. something is detected).

        Arguments:
            y: 1D numpy array with Y coords of points
            x: 1D numpy array with X coords of points
            z: 1D numpy array with Z coords of points (aka. frame number)
        
        Return:
            [bool] True if video should be further checked for fireballs, False otherwise
        """
        
        # Check if there are enough points
        if(len(y) < self.config.min_points):
            return False
        
        # Check how many points are close to each other (along the time line)
        code = """
        unsigned int distance, i, count = 0,
        y_dist, x_dist, z_dist,
        y_prev = 0, x_prev = 0, z_prev = 0;
        
        for(i=1; i<Ny[0]; i++) {
            y_dist = Y1(i) - y_prev;
            x_dist = X1(i) - x_prev;
            z_dist = Z1(i) - z_prev;
            
            distance = sqrt(y_dist*y_dist + z_dist*z_dist + z_dist*z_dist);
            
            if(distance < gap_treshold) {
                count++;
            }
            
            y_prev = Y1(i);
            x_prev = X1(i);
            z_prev = Z1(i);
        }
        
        return_val = count;
        """
        
        dictionary = {'gap_treshold': sqrt(self.config.gap_treshold), 'y': y, 'x': x, 'z': z}
        count = weave.inline(code, dictionary.keys(), dictionary, verbose=2, extra_compile_args=self.config.weaveArgs, extra_link_args=self.config.weaveArgs)
        
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
            firstFrame = firstFrame - ceil(diff*self.config.before) 
            if firstFrame < 0:
                firstFrame = 0

            # Extrapolate after last detected point
            lastFrame = lastFrame + ceil(diff*self.config.after) 
            if lastFrame >= self.frames.shape[0]:
                lastFrame = self.frames.shape[0] - 1
            
            out = np.zeros((self.frames.shape[0], self.config.maxSize, self.config.maxSize), np.uint8)

            # y, x, size
            sizepos = np.empty((self.frames.shape[0], 4), np.uint16)
            
            code = """
                int x_m, x_p, x_t, y_m, y_p, y_t, k,
                half_max_size = maxSize / 2,
                half_f = f / 2;
                unsigned int x, y, i, x2, y2, num = 0,
                max, pixel, limit, max_width, max_height, size, half_size, num_equal;
                
                for(i = firstFrame; i < lastFrame; i++) {
                    // calculate point at current time
                    k = i - POINT1(2);
                    y_t = (POINT1(0) + slopeYZ * k) * f + half_f;
                    x_t = (POINT1(1) + slopeXZ * k) * f + half_f;
                    
                    if(y_t < 0 || x_t < 0 || y_t >= Nframes[1] || x_t >= Nframes[2]) {
                        // skip if out of bounds
                        continue;
                    }
                    
                    // calculate boundaries for finding max value
                    y_m = y_t - half_f, y_p = y_t + half_f, 
                    x_m = x_t - half_f, x_p = x_t + half_f;
                    if(y_m < 0) {
                        y_m = 0;
                    }
                    if(x_m < 0) {
                        x_m = 0;
                    }
                    if(y_p >= Nframes[1]) {
                        y_p = Nframes[1] - 1;
                    }
                    if(x_p >= Nframes[2]) {
                        x_p = Nframes[2] - 1;
                    }
                    
                    // find max value
                    max = 0;
                    for(y=y_m; y<y_p; y++) {
                        for(x=x_m; x<x_p; x++) {
                            pixel = FRAMES3(i, y, x);
                            if(pixel > max) {
                                max = pixel;
                            }
                        }
                    }
                    
                    // calculate boundaries for finding size
                    y_m = y_t - half_max_size, y_p = y_t + half_max_size, 
                    x_m = x_t - half_max_size, x_p = x_t + half_max_size;
                    if(y_m < 0) {
                        y_m = 0;
                    }
                    if(x_m < 0) {
                        x_m = 0;
                    }
                    if(y_p >= Nframes[1]) {
                        y_p = Nframes[1] - 1;
                    }
                    if(x_p >= Nframes[2]) {
                        x_p = Nframes[2] - 1;
                    }
                    
                    // calculate mean distance from center
                    max_width = 0, max_height = 0, num_equal = 0,
                    limit = limitForSize * max;
                    for(y=y_m; y<y_p; y++) {
                        for(x=x_m; x<x_p; x++) {
                            if(FRAMES3(i, y, x) - COMPRESSED3(2, y, x) >= limit) {
                                max_height += abs(y_t - y);
                                max_width += abs(x_t - x);
                                num_equal++;
                            }
                        }
                    }
                    
                    // calculate size
                    if(max_height > max_width) {
                        size = max_height / num_equal;
                    } else {
                        size = max_width / num_equal;
                    }
                    if(size < minSize) {
                        size = minSize;
                    } else if(size > half_max_size) {
                        size = half_max_size;
                    }
                    
                    // save size
                    SIZEPOS2(num, 3) = size;
                    half_size = size / 2;
                    
                    // adjust position for frame extraction if out of borders
                    if(y_t < half_size) {
                        y_t = half_size;
                    }
                    if(x_t < half_size) {
                        x_t = half_size;
                    }
                    if(y_t >= Nframes[1] - half_size) {
                        y_t = Nframes[1] - 1 - half_size;
                    }
                    if(x_t >= Nframes[2] - half_size) {
                        x_t = Nframes[2] - 1 - half_size;
                    }
                    
                    // save location
                    SIZEPOS2(num, 0) = y_t; 
                    SIZEPOS2(num, 1) = x_t; 
                    SIZEPOS2(num, 2) = i;
                    
                    // calculate bounds for frame extraction
                    y_m = y_t - half_size, y_p = y_t + half_size, 
                    x_m = x_t - half_size, x_p = x_t + half_size;
                    
                    // crop part of frame
                    y2 = 0, x2 = 0;
                    for(y=y_m; y<y_p; y++) {
                        x2 = 0;
                        for(x=x_m; x<x_p; x++) {
                            OUT3(num, y2, x2) = FRAMES3(i, y, x);
                            x2++;
                        }
                        y2++;
                    }
                    
                    num++;
                }
                
                return_val = num;                
            """
            
            dict = {'frames': self.frames, 'compressed': self.compressed, 'point': point, 'slopeXZ': slopeXZ, 'slopeYZ': slopeYZ,
                    'firstFrame': firstFrame, 'lastFrame': lastFrame, 'f': self.config.f, 'limitForSize': self.config.limitForSize,
                    'minSize': self.config.minSize, 'maxSize': self.config.maxSize, 'sizepos': sizepos, 'out': out}

            length = weave.inline(code, dict.keys(), dict, verbose=2, extra_compile_args=self.config.weaveArgs, extra_link_args=self.config.weaveArgs)
            
            out = out[:length]
            sizepos = sizepos[:length]
            
            clips.append([out, sizepos])
        
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

        # Save the clips
        self.save(clips)

        log.debug("[" + self.filename + "] Time for saving: " + str(time.time() - t) + "s")