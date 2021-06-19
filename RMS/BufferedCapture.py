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

import re
import time
import logging
from multiprocessing import Process, Event

import cv2

from RMS.Misc import ping

# Get the logger from the main module
log = logging.getLogger("logger")


class BufferedCapture(Process):
    """ Capture from device to buffer in memory.
    """
    
    running = False
    
    def __init__(self, array1, startTime1, array2, startTime2, config, video_file=None):
        """ Populate arrays with (startTime, frames) after startCapture is called.
        
        Arguments:
            array1: numpy array in shared memory that is going to be filled with frames
            startTime1: float in shared memory that holds time of first frame in array1
            array2: second numpy array in shared memory
            startTime2: float in shared memory that holds time of first frame in array2

        Keyword arguments:
            video_file: [str] Path to the video file, if it was given as the video source. None by default.

        """
        
        super(BufferedCapture, self).__init__()
        self.array1 = array1
        self.startTime1 = startTime1
        self.array2 = array2
        self.startTime2 = startTime2
        
        self.startTime1.value = 0
        self.startTime2.value = 0
        
        self.config = config

        self.video_file = video_file

        # A frame will be considered dropped if it was late more then half a frame
        self.time_for_drop = 1.5*(1.0/config.fps)

        self.dropped_frames = 0
    


    def startCapture(self, cameraID=0):
        """ Start capture using specified camera.
        
        Arguments:
            cameraID: ID of video capturing device (ie. ID for /dev/video3 is 3). Default is 0.
            
        """
        
        self.cameraID = cameraID
        self.exit = Event()
        self.start()
    


    def stopCapture(self):
        """ Stop capture.
        """
        
        self.exit.set()

        time.sleep(1)

        log.info("Joining capture...")

        # Wait for the capture to join for 60 seconds, then terminate
        for i in range(60):
            if self.is_alive():
                time.sleep(1)
            else:
                break

        if self.is_alive():
            log.info('Terminating capture...')
            self.terminate()


    def initVideoDevice(self):
        """ Initialize the video device. """

        device = None

        # use a file as the video source
        if self.video_file is not None:
            device = cv2.VideoCapture(self.video_file)

        # Use a device as the video source
        else:

            # If an analog camera is used, skip the ping
            ip_cam = False
            if "rtsp" in str(self.config.deviceID):
                ip_cam = True


            if ip_cam:

                ### If the IP camera is used, check first if it can be pinged

                # Extract the IP address
                ip = re.findall(r"[0-9]+(?:\.[0-9]+){3}", self.config.deviceID)

                # Check if the IP address was found
                if ip:
                    ip = ip[0]

                    # Try pinging 5 times
                    ping_success = False

                    for i in range(500):

                        print('Trying to ping the IP camera...')
                        ping_success = ping(ip)

                        if ping_success:
                            log.info("Camera IP ping successful!")
                            break

                        time.sleep(5)

                    if not ping_success:
                        log.error("Can't ping the camera IP!")
                        return None

                else:
                    return None



            # Init the video device
            log.info("Initializing the video device...")
            if self.config.force_v4l2:
                device = cv2.VideoCapture(self.config.deviceID, cv2.CAP_V4L2)
                device.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            else:
                device = cv2.VideoCapture(self.config.deviceID)

            # Try setting the resultion if using a video device, not gstreamer
            try:

                # This will fail if the video device is a gstreamer pipe
                int(self.config.deviceID)
                
                # Set the resolution (crashes if using an IP camera and gstreamer!)
                device.set(3, self.config.width_device)
                device.set(4, self.config.height_device)

            except:
                pass


        return device


    def run(self):
        """ Capture frames.
        """
        
        # Init the video device
        device = self.initVideoDevice()


        if device is None:

            log.info('The video source could not be opened!')
            self.exit.set()
            return False


        # Wait until the device is opened
        device_opened = False
        for i in range(20):
            time.sleep(1)
            if device.isOpened():
                device_opened = True
                break


        # If the device could not be opened, stop capturing
        if not device_opened:
            log.info('The video source could not be opened!')
            self.exit.set()
            return False

        else:
            log.info('Video device opened!')



        # Throw away first 10 frame
        for i in range(10):
            device.read()


        first = True

        wait_for_reconnect = False
        
        # Run until stopped from the outside
        while not self.exit.is_set():


            lastTime = 0
            
            if first:
                self.startTime1.value = 0
            else:
                self.startTime2.value = 0
            

            # If the video device was disconnected, wait 5s for reconnection
            if wait_for_reconnect:

                print('Reconnecting...')

                while not self.exit.is_set():

                    log.info('Waiting for the video device to be reconnected...')

                    time.sleep(5)

                    # Reinit the video device
                    device = self.initVideoDevice()


                    if device is None:
                        print("The video device couldn't be connected! Retrying...")
                        continue


                    if self.exit.is_set():
                        break

                    # Read the frame
                    log.info("Reading frame...")
                    ret, frame = device.read()
                    log.info("Frame read!")

                    # If the connection was made and the frame was retrieved, continue with the capture
                    if ret:
                        log.info('Video device reconnected successfully!')
                        wait_for_reconnect = False
                        break


                wait_for_reconnect = False


            t_frame = 0
            t_assignment = 0
            t_convert = 0
            t_block = time.time()
            block_frames = 256

            log.info('Grabbing a new block of {} frames...'.format(block_frames))
            for i in range(block_frames):

                # Read the frame
                t1_frame = time.time()
                ret, frame = device.read()
                t_frame = time.time() - t1_frame


                # If the video device was disconnected, wait for reconnection
                if not ret:

                    log.info('Frame grabbing failed, video device is probably disconnected!')

                    wait_for_reconnect = True
                    break



                # If the end of the file was reached, stop the capture
                if (self.video_file is not None) and (frame is None):

                    log.info('End of video file! Press Ctrl+C to finish.')

                    self.exit.set()
                    
                    time.sleep(0.1)

                    break

                
                t = time.time()

                if i == 0: 
                    startTime = t

                # Check if frame is dropped if it has been more than 1.5 frames than the last frame
                elif (t - lastTime) >= self.time_for_drop:
                    
                    # Calculate the number of dropped frames
                    n_dropped = int((t - lastTime)*self.config.fps)
                    
                    if self.config.report_dropped_frames:
                        log.info(str(n_dropped) + " frames dropped! Time for frame: {:.3f}, convert: {:.3f}, assignment: {:.3f}".format(t_frame, t_convert, t_assignment))

                    self.dropped_frames += n_dropped

                    

                lastTime = t
                
                t1_convert = time.time()

                # Convert the frame to grayscale
                #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Convert the frame to grayscale
                if len(frame.shape) == 3:

                    # If a color image is given, take the green channel
                    if frame.shape[2] == 3:

                        gray = frame[:, :, 1]
                    # if UYVY image given, take luma (Y) channel
                    elif frame.shape[2] == 2 and self.config.uyvy_pixelformat:
                        gray = frame[:, :, 1]
                    else:
                        gray = frame[:, :, 0]

                else:
                    gray = frame


                # Cut the frame to the region of interest (ROI)
                gray = gray[self.config.roi_up:self.config.roi_down, \
                    self.config.roi_left:self.config.roi_right]


                t_convert = time.time() - t1_convert


                # Assign the frame to shared memory
                t1_assign = time.time()
                if first:
                    self.array1[i, :gray.shape[0], :gray.shape[1]] = gray
                else:
                    self.array2[i, :gray.shape[0], :gray.shape[1]] = gray

                t_assignment = time.time() - t1_assign

                # If video is loaded from a file, simulate real FPS
                if self.video_file is not None:

                    time.sleep(1.0/self.config.fps)

                    # If the video finished, stop the capture
                    if not device.isOpened():
                        self.exit.set()



            if self.exit.is_set():
                wait_for_reconnect = False
                log.info('Capture exited!')
                break


            if not wait_for_reconnect:

                # Set the starting value of the frame block, which indicates to the compression that the
                # block is ready for processing
                if first:
                    self.startTime1.value = startTime

                else:
                    self.startTime2.value = startTime

                log.info('New block of raw frames available for compression with starting time: {:s}'.format(str(startTime)))

            
            # Switch the frame block buffer flags
            first = not first
            if self.config.report_dropped_frames:
                log.info('Estimated FPS: {:.3f}'.format(block_frames/(time.time() - t_block)))
        

        log.info('Releasing video device...')
        device.release()
        log.info('Video device released!')
    
