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

from multiprocessing import Process, Queue
import numpy as np
import cv2
import time


class BufferedCapture(Process):
    """Capture from device to buffer in memory.
    """
    
    TIME_FOR_DROP = 0.05
    
    running = False
    cameraID = 0
    
    def __init__(self, framesList):
        """Populate list with (startTime, frames) after startCapture is called.
        
        @param framesList: list in shared memory that is going to be filled with frames and start times
        """
        
        super(BufferedCapture, self).__init__()
        self.framesList = framesList
    
    def startCapture(self, cameraID=0):
        """Start capture using specified camera.
        
        @param cameraID: ID of video capturing device (ie. ID for /dev/video3 is 3). Default is 0.
        """
        
        self.cameraID = cameraID
        self.running = True
        self.start()
    
    def stopCapture(self):
        """Stop capture.
        """
        
        self.running = False
        self.join()

    def run(self):
        """Capture frames.
        """
        
        device = cv2.VideoCapture(self.cameraID)
        
        while self.running:
            frames = np.empty((256, 576, 720), np.uint8)
            t = 0
            last_time = time.time()
            
            for i in range(256):
                ret, frame = device.read()
                
                t = time.time()
                if i == 0: 
                    startTime = t
                elif last_time - t > self.TIME_FOR_DROP: #check if frame is dropped TODO: better frame dropping detection
                    frames[i] = None
                    i += 1
                    print "frame dropped!"
                last_time = t
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames[i] = gray
            
            self.framesList.append((startTime, frames))
            frames = None
        
        device.release()
    