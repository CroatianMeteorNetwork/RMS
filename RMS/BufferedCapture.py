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
import cv2
import time
import logging

# Get the logger from the main module
log = logging.getLogger("logger")


class BufferedCapture(Process):
    """Capture from device to buffer in memory.
    """
    
    TIME_FOR_DROP = 0.05
    
    running = False
    
    def __init__(self, array1, startTime1, array2, startTime2, config):
        """Populate arrays with (startTime, frames) after startCapture is called.
        
        @param array1: numpy array in shared memory that is going to be filled with frames
        @param startTime1: float in shared memory that holds time of first frame in array1
        @param array2: second numpy array in shared memory
        @param startTime2: float in shared memory that holds time of first frame in array2
        """
        
        super(BufferedCapture, self).__init__()
        self.array1 = array1
        self.startTime1 = startTime1
        self.array2 = array2
        self.startTime2 = startTime2
        
        self.startTime1.value = 0
        self.startTime2.value = 0
        
        self.config = config
    
    def startCapture(self, cameraID=0):
        """Start capture using specified camera.
        
        @param cameraID: ID of video capturing device (ie. ID for /dev/video3 is 3). Default is 0.
        """
        
        self.cameraID = cameraID
        self.exit = Event()
        self.start()
    
    def stopCapture(self):
        """Stop capture.
        """
        
        self.exit.set()
        self.join()

    def run(self):
        """Capture frames.
        """
        
        device = cv2.VideoCapture(self.config.deviceID)
        device.read() # throw away first frame
        first = True
        
        while not self.exit.is_set():
            lastTime = 0
            
            if first:
                self.startTime1.value = 0
            else:
                self.startTime2.value = 0
            
            for i in range(256):
                ret, frame = device.read()
                
                t = time.time()
                if i == 0: 
                    startTime = t
                elif lastTime - t > self.TIME_FOR_DROP: #check if frame is dropped TODO: better frame dropping detection
                    #TODO: increase dropped frames counter
                    log.warn("frame dropped!")
                lastTime = t
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if first:
                    self.array1[i] = gray
                else:
                    self.array2[i] = gray
            
            if first:
                self.startTime1.value = startTime
            else:
                self.startTime2.value = startTime
            first = not first
        
        device.release()
    