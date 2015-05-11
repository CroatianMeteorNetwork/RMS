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

from multiprocessing import Process
from math import floor
from scipy import weave
import numpy as np
import time
import struct

class Compression(Process):
    """Compress list of numpy arrays (video frames).
    Output is in Flat-field Temporal Pixel (FTP) format like the one used by CAMS project (P. Jenniskens et al., 2011).
    """

    running = False
    
    def __init__(self, framesList, camNum):
        """
        
        @param framesList: list in shared memory of 2d numpy arrays (grayscale video frames)
        @param camNum: camera ID (ie. 459)
        """
        
        super(Compression, self).__init__()
        self.framesList = framesList
        self.camNum = camNum
    
    def convert(self, frames):
        """Convert from (256, 576, 720) to (576, 720, 256) and handle dropped frames.
        
        @param frames: video frames as 3d numpy array
        @return: video frames as 3d numpy array ready for Compression.compress()
        """
        
        for i, frame in enumerate(frames):
            if(frame == None):
                frames[i] = np.zeros((576, 720), np.uint8) #TODO: increase dropped frames counter
        
        frames = np.swapaxes(frames, 0, 2)
        frames = np.swapaxes(frames, 0, 1)
        
        return frames
    
    def compress(self, frames):
        """Compress frames to the FTP-compatible array.
        
        @param frames: grayscale frames stored as 3d numpy array
        @return: 3d numpy array in format: (N, y, x) where N is [0, 4)
        """
        
        out = np.empty((4, 576, 720), np.uint8)
        rands = np.random.uniform(low = 0.0, high = 1.0, size = 1048576)
        
        code = """
        unsigned int x, y, n, acc, var, max, max_frame, pixel;
        float num_equal;
        unsigned int rand_count = 0;
    
        for(y=0; y<576; y++) {
            for(x=0; x<720; x++) {
                acc = 0;
                var = 0;
                max = 0;
                max_frame = 0;
                num_equal = 0;
                
                for(n=0; n<256; n++) {
                    pixel = FRAMES3(y, x, n);
                    acc += pixel;
                    if(pixel > max) {
                        max = pixel;
                        max_frame = n;
                        num_equal = 1;
                    } else if(pixel == max) {
                        num_equal++;
                        
                        rand_count = (rand_count + 1) % 1048576L;
                        if(RANDS1(rand_count) <= 1/num_equal) {
                            max_frame = n;
                        }
                    }
                }
                acc -= max;
                acc = acc >> 8;
    
                for(n=0; n<256&&n!=max_frame; n++) {
                    pixel = FRAMES3(y, x, n) - acc;
                    var += pixel*pixel;
                }
                var = sqrt(var >> 8);
                
                OUT3(0, y, x) = max;
                OUT3(1, y, x) = max_frame;
                OUT3(2, y, x) = acc;
                OUT3(3, y, x) = var;
            }
        }
        """
        
        weave.inline(code, ['frames', 'rands', 'out'])
        return out
    
    def save(self, arr, startTime, N, camNum):
        """Writes metadata and data array to FTP .bin file.
        
        @param arr: 3d numpy array in format: (N, y, x) where N is [0, 4)
        @param startTime: seconds and fractions of a second from epoch to first frame
        @param N: frame counter (ie. 0000353)
        @param camNum: camera ID (ie. 459)
        """
        
        dateTime = time.strftime("%Y%m%d_%H%M%S", time.localtime(startTime))
        millis = int((startTime - floor(startTime))*1000)
        
        image = "FF" + str(camNum).zfill(3) +  "_" + dateTime + "_" + str(millis).zfill(3) + "_" + str(N).zfill(7) + ".bin"
        
        with open(image, "wb") as f:
            f.write(struct.pack('I', arr.shape[1]))  # nrows
            f.write(struct.pack('I', arr.shape[2]))  # ncols
            f.write(struct.pack('I', 8))             # nbits
            f.write(struct.pack('I', N+256))         # first
            f.write(struct.pack('I', camNum))        # camera number
        
            arr.tofile(f)                            # image array
    
    def stop(self):
        """Stop process.
        """
        
        self.running = False
        self.join()
        
    def start(self):
        """Start process.
        """
        self.running = True
        super(Compression, self).start()
        
    def run(self):
        """Retrieve frames from list, convert, compress and save them.
        """
        
        n = 0
        
        while self.running:
            while len(self.framesList)<1: #block until frames are available
                if not self.running:      #exit function if process was stopped
                    return
                
            startTime = self.framesList[0][0] #retrieve time of first frame
            
            frames = self.convert(self.framesList[0][1]) #convert frames
            del self.framesList[0]                       #and clean buffer
            
            frames = self.compress(frames)
            
            self.save(frames, startTime, n, self.camNum)
            n += 1
    