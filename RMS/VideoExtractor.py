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
from RMS import Grouping3D
from math import floor, sqrt, ceil
import time
import struct
from os import uname
import logging

class Extractor(Process):
    f = 16                    # subsampling factor
    
    # params for findPoints()
    min_level = 40            # ignore pixel if below this level
    min_points = 8            # minimum number of points required to add event point
    k1 = 4                    # k1 factor for thresholding
    max_points_per_frame = 30 # ignore frame if there are too many points in it (ie. flare)
    max_per_frame_factor = 10 # multiplied with median number of points for flare detection
    max_points = 190          # if there are too many points in total after flare removal, randomize them
    min_frames = 4            # minimum number of frames
    
    def __init__(self):
        super(Extractor, self).__init__()
    
    def findPoints(self):
        """Threshold and subsample frames and return as list of points.
        
        @return: (y, x, z) of found points
        """
     
        count = np.zeros((self.frames.shape[0], floor(self.frames.shape[1]//self.f), floor(self.frames.shape[2]//self.f)), np.int16)
        pointsy = np.empty((self.frames.shape[0]*floor(self.frames.shape[1]//self.f)*floor(self.frames.shape[2]//self.f)), np.uint16)
        pointsx = np.empty((self.frames.shape[0]*floor(self.frames.shape[1]//self.f)*floor(self.frames.shape[2]//self.f)), np.uint16)
        pointsz = np.empty((self.frames.shape[0]*floor(self.frames.shape[1]//self.f)*floor(self.frames.shape[2]//self.f)), np.uint16)
        
        code = """
        unsigned int x, y, x2, y2, n, i, max;
        unsigned int num = 0, acc = 0;
        
        for(y=0; y<Nframes[1]; y++) {
            for(x=0; x<Nframes[2]; x++) {
                max = COMPRESSED3(0, y, x);
                if((max > min_level) && (max >= COMPRESSED3(2, y, x) + k1 * COMPRESSED3(3, y, x))) {
                    n = COMPRESSED3(1, y, x);
                    
                    y2 = y/f; // subsample frame in f*f squares
                    x2 = x/f;
                    
                    if(COUNT3(n, y2, x2) >= min_points) { // check if there is enough of threshold passers inside of this square
                        POINTSY1(num) = y2;
                        POINTSX1(num) = x2;
                        POINTSZ1(num) = n;
                        num++;
                        COUNT3(n, y2, x2) = -1; //don't repeat this number
                    } else if(COUNT3(n, y2, x2) != -1) { // increase counter if not enough threshold passers and this number isn't written already
                        COUNT3(n, y2, x2) += 1;
                    }
                }
            }    
        }
        
        return_val = num; // output length of POINTS arrays
        """
        
        args = []
        if uname()[4] == "armv7l":
            args = ["-O3", "-mfpu=neon", "-mfloat-abi=hard", "-fdump-tree-vect-details", "-funsafe-loop-optimizations", "-ftree-loop-if-convert-stores"]
        
        dictionary = {'frames': self.frames, 'compressed': self.compressed, 'min_level': self.min_level,
                      'min_points': self.min_points, 'k1': self.k1, 'f': self.f,
                      'count': count, 'pointsy': pointsy, 'pointsx': pointsx, 'pointsz': pointsz}
        length = weave.inline(code, dictionary.keys(), dictionary, verbose=2, extra_compile_args=args, extra_link_args=args)
        
        # return empty list if there is no points
        if length == 0:
            return []
        
        # cut away extra long array
        y = pointsy[0 : length]
        x = pointsx[0 : length]
        z = pointsz[0 : length]
        
        freq = stats.itemfreq(z) # return list with number of occurance of each frame num (Z axis)
        
        # reject the image if there are too little event frames
        if len(freq) <= self.min_frames:
            return []
        
        # calculate a threshold based on factors and median number of points on the images per frame
        outlier_treshold = self.max_per_frame_factor * np.median(freq[:, 1])
        if outlier_treshold > self.max_points_per_frame:
            outlier_treshold = self.max_points_per_frame
        
        # remove all outliers (aka frames with a strong flare
        for i, item in enumerate(freq):
            frameNum, count = item
            if count >= outlier_treshold:
                indices = np.where(z != frameNum)
                y = y[indices]
                x = x[indices]
                z = z[indices]
                
        # randomize points if there are too many in total
        if len(z) > self.max_points:
            indices = np.random.randint(0, len(z), self.max_points)
            y = y[indices]
            x = x[indices]
            z = z[indices]
        
        # sort by frame number
        indices = np.argsort(z) # quicksort
        y = y[indices].astype(np.float)
        x = x[indices].astype(np.float)
        z = z[indices].astype(np.float)
        
        # test points
        if not self.testPoints(y, x, z):
            return []
        
        # convert to python list
        event_points = np.squeeze(np.dstack((y, x, z))).tolist()
        
        return event_points
    
    def testPoints(self, y, x, z, min_points=5, gap_treshold=70):
        """ Test if points are interesting (ie. something is detected).
        
        @param y: 1D numpy array with Y coords of points
        @param x: 1D numpy array with X coords of points
        @param z: 1D numpy array with Z coords of points
        @return: true if video should be further checked for meteors, false otherwise
        """
        
        # check if there is enough points
        if(len(y) < min_points):
            return False
        
        # check how many points are close to each other (along the time line)
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
        
        args = []
        if uname()[4] == "armv7l":
            args = ["-O3", "-mfpu=neon", "-mfloat-abi=hard", "-fdump-tree-vect-details", "-funsafe-loop-optimizations", "-ftree-loop-if-convert-stores"]
        count = weave.inline(code, ['gap_treshold', 'y', 'x', 'z'], verbose=2, extra_compile_args=args, extra_link_args=args)
        
        return count >= min_points
    
    def extract(self, frames, compressed, coefficients, before=0.15, after=0.25, f=16, limitForSize=0.90, minSize=8, maxSize=192):
        """ Determinate window size and crop out frames.
        
        @param frames: raw video frames
        @param compressed: compressed frames
        @param coefficients: linear coefficients for each detected meteor
        @param before: number of frames to extract before detected meteor
        @param after: number of frames to extract after detected meteor 
        @param f: subsampling size
        """
        
        clips = []
        
        for point, slopeXZ, slopeYZ, firstFrame, lastFrame in coefficients:
            slopeXZ = float(slopeXZ)
            slopeYZ = float(slopeYZ)
            firstFrame = int(firstFrame)
            lastFrame = int(lastFrame)
            
            diff = lastFrame - firstFrame
            firstFrame = firstFrame - ceil(diff*before) # extrapolate before first detected point
            if firstFrame < 0:
                firstFrame = 0
            lastFrame = lastFrame + ceil(diff*after) # extrapolate after last detected point
            if lastFrame >= frames.shape[0]:
                lastFrame = frames.shape[0] - 1
            
            out = np.zeros((frames.shape[0], maxSize, maxSize), np.uint8)
            sizepos = np.empty((frames.shape[0], 4), np.uint16) # y, x, size
            
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
            
            args = []
            if uname()[4] == "armv7l":
                args = ["-O3", "-mfpu=neon", "-mfloat-abi=hard", "-fdump-tree-vect-details", "-funsafe-loop-optimizations", "-ftree-loop-if-convert-stores"]
            length = weave.inline(code, ['frames', 'compressed', 'point', 'slopeXZ', 'slopeYZ', 'firstFrame', 'lastFrame', 'f', 'limitForSize', 'minSize', 'maxSize', 'sizepos', 'out'], verbose=2, extra_compile_args=args, extra_link_args=args)
            
            out = out[:length]
            sizepos = sizepos[:length]
            
            clips.append([out, sizepos])
        
        return clips
    
    def save(self, clips, fileName):
        """ Save extracted clips to FR*.bin file
        """
        
        file = "FR" + fileName + ".bin"
        
        with open(file, "wb") as f:
            f.write(struct.pack('I', len(clips)))             # number of extracted lines
            
            for clip in clips:
                frames, sizepos = clip
                
                f.write(struct.pack('I', len(frames)))        # number of extracted frames
                
                for i, frame in enumerate(frames):
                    f.write(struct.pack('I', sizepos[i, 0]))  # y of center
                    f.write(struct.pack('I', sizepos[i, 1]))  # x of center
                    f.write(struct.pack('I', sizepos[i, 2]))  # time
                    size = sizepos[i, 3]
                    f.write(struct.pack('I', size))           # cropped frame size
                    frame[:size, :size].tofile(f)             # cropped frame
    
    def stop(self):
        """Stop the process.
        """
        
        self.exit.set()
        self.join()
        
    def start(self, frames, compressed, filename):
        """Start the process.
        """
        
        self.frames = frames
        self.compressed = compressed
        self.filename = filename
        self.exit = Event()
        
        super(Extractor, self).start()
    
    def run(self):
        """Retrieve frames from list, convert, compress and save them.
        """
        
        self.executeAll(self.frames, self.compressed, self.filename)
    
    def executeAll(self, frames, compressed, filename):
        # Check if the average is all white (or close to it) and skip it
        if np.average(compressed[2]) > 220:
            logging.debug("[" + filename + "] frames are all white")
            return
        
        t = time.time()
        event_points = self.findPoints()
        logging.debug("[" + filename + "] time for thresholding and subsampling: " + str(time.time() - t) + "s")
        
        t = time.time()
        
        if len(event_points) == 0:
            logging.debug("[" + filename + "] nothing found, not extracting anything")
            return
        
        y_dim = frames.shape[1]/16
        x_dim = frames.shape[2]/16
    
        ############################
        # Define parameters
        distance_treshold = 70
        distance_treshold = Grouping3D.normalizeParameter(distance_treshold, y_dim, x_dim)
        line_distance_const = 4
        gap_treshold = 130
        gap_treshold = Grouping3D.normalizeParameter(gap_treshold, y_dim, x_dim)
        minimum_points = 3
        point_ratio_treshold = 0.7
        ###########################
        
        logging.debug("[" + filename + "] time for defining parameters: " + str(time.time() - t) + "s")
        
        t = time.time()
        # Find lines in 3D space and store them to line_list
        line_list = Grouping3D.find3DLines(event_points, [], distance_treshold, line_distance_const, gap_treshold, minimum_points, point_ratio_treshold)
        logging.debug("[" + filename + "] Time for finding lines: " + str(time.time() - t) + "s")
        
        if line_list == None:
            logging.debug("[" + filename + "] no lines found, not extracting anything")
            return
        
        t = time.time()
        coeff = Grouping3D.findCoefficients(event_points, line_list)
        logging.debug("[" + filename + "] Time for finding coefficients: " + str(time.time() - t) + "s")
        
        t = time.time()
        clips = self.extract(frames, compressed, coeff)
        logging.debug("[" + filename + "] Time for extracting: " + str(time.time() - t) + "s")
        t = time.time()
         
        t = time.time()
        self.save(clips, filename)
        logging.debug("[" + filename + "] Time for saving: " + str(time.time() - t) + "s")
        t = time.time()