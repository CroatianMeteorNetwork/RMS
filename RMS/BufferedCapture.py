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
# Set GStreamer debug level. Use '2' for warnings in production environments.
os.environ['GST_DEBUG'] = '3'

import re
import time
import logging
import datetime
import os.path
from multiprocessing import Process, Event, Value

import cv2
import numpy as np

from RMS.Misc import ping

# Get the logger from the main module
log = logging.getLogger("logger")

GST_IMPORTED = False
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    GST_IMPORTED = True

except ImportError as e:
    log.info('Could not import gi: {}. Using OpenCV.'.format(e))


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
        self.media_backend_override = False
        self.video_device_type = "cv2"

        self.video_file = video_file

        # A frame will be considered dropped if it was late more then half a frame
        self.time_for_drop = 1.5*(1.0/config.fps)

        # Initialize Smoothing variables
        self.startup_flag = True
        self.last_calculated_fps = 0
        self.last_calculated_fps_n = 0
        self.expected_m = 1e9/self.config.fps
        self.reset_count = -1

        self.dropped_frames = Value('i', 0)
        self.device = None
        self.pipeline = None
        self.start_timestamp = 0
        self.frame_shape = None
        self.convert_to_gray = False
                    

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

        return self.dropped_frames.value


    def deviceIsOpened(self):
        """ Return True if media backend is opened.
        """

        if self.device is None:
            return False
        
        try:
            # OpenCV
            if self.video_device_type == "cv2":

                return self.device.isOpened()
            
            # GStreamer
            else:

                if GST_IMPORTED:

                    state = self.device.get_state(Gst.CLOCK_TIME_NONE).state
                    if state == Gst.State.PLAYING:
                        return True
                    else:
                        return False
                    
                else:
                    return False
                
        except Exception as e:
            log.error('Error checking device status: {}'.format(e))
            return False


    def calculatePTSRegressionParams(self, y):
        """ Add pts and perform an online linear regression on pts.
            smoothed_pts = m*frame_count + b
            m is the slope in ns per frame (1e9/fps)
            Adjust b so that the line passes through the earliest frames.

        Arguments:
            y: [float] pts of the frame

        Return:
            m: [float] slope in ns per frame
            b: [float] y-intercept in ns

        """
        self.n += 1
        x = self.n
        self.sum_x += x
        self.sum_y += y
        self.sum_xx += x*x
        self.sum_xy += x*y

        # Update regression parameters
        if x > 1:
            m = (self.n*self.sum_xy - self.sum_x*self.sum_y)/(self.n*self.sum_xx - self.sum_x**2)
        
        # First frame
        else:
            m = self.expected_m
            self.b = y - m*x
        
        ## STARTUP ##
        # On startup, use expected fps until calculate fps stabilizes
        if (self.n <= self.startup_frames) and self.startup_flag:

            # Exit startup if calculated m doesn't converge with expected m

            # Check error at increasingly longer intervals
            if x < self.startup_frames/32:
                sample_interval = 128
            elif x < self.startup_frames/16:
                sample_interval = 512
            elif x < self.startup_frames/8:
                sample_interval = 1024
            elif x < self.startup_frames/4:
                sample_interval = 2048
            else:
                sample_interval = 4096

            # Determine if the values converge. Skipping the first few noisy frames
            if ((x - 25)%sample_interval == 0) or (x == self.startup_frames):

                m_err = abs(m - self.expected_m)
                delta_m_err = (m_err - self.last_m_err)/(x - self.last_m_err_n)
                startup_remaining = self.startup_frames - x
                final_m_err = m_err + startup_remaining*delta_m_err
                self.last_m_err = m_err
                self.last_m_err_n = x

                # If end is reached, or error does not converge to zero, exit startup
                if (final_m_err > 0) or (x == self.startup_frames):

                    # If residual error on exit is too large, the expected m is probably wrong.
                    if m_err > 2000:

                        # Reset debt and b as they were probably wrong, and permanently disable startup
                        self.startup_flag = False
                        self.b_error_debt = 0
                        self.b = y - m*x
                        self.m_jump_error = 0

                        log.info("Check config FPS! Startup sequence exited early probably due to inaccurate FPS value. "
                                 "Startup is disabled for the remainder of the run")

                    # On normal exit, calculate residual error for smooth transition to calculate m
                    else:

                        # calculate the jump error
                        self.m_jump_error = x*(m - self.expected_m) # ns

                    log.info("Exiting startup logic at {:.1f}% of startup sequence, Expected fps: {:.6f}, "
                             "calculated fps at this point: {:.6f}, residual m error: {:.1f} ns, sample interval: {}"
                             .format(100*x/self.startup_frames, 1e9/self.expected_m, 1e9/m, m_err, sample_interval))

                    # This will temporarily exit startup
                    self.startup_frames = 0

            # Use expected value during startup
            if self.startup_frames > 0:
                m = self.expected_m

        ### LEAST DELAYED FRAME LOGIC ###
                
        # The code attempts to smoothly distribute presentation timestamps (pts) on a line that passes
        # through the least-delayed frame. The idea is that the least-delayed frames are thought to
        # be the least affected by network and other delays, and should therefore offer the most
        # consistent points of reference.
        # When a new least-delayed frame is detected, the time delta is smoothly distributed over
        # time.
        # The line has a slope m (ns per frame) that passes through the least delayed frame by
        # adjusting b in: y = m*x + b
        # where y is the pts, and x is the frame number.
        # A slow positive bias is introduce to keep the line in contact with a slowly accelerating
        # frame rate.
        # Finally, the small jump error at the completion of the startup sequence, when
        # transitioning from expected fps to calculated fps (linear regression), is smoothly
        # distributed over time.
                
        # Calculate the delta between the lowest point and current point
        delta_b = self.b - (y - m*x)

        # Adjust b error debt to the max of current debt or new delta b
        self.b_error_debt = max(self.b_error_debt, delta_b)
        
        # Skew b, if due
        if self.b_error_debt > 0 or self.m_jump_error != 0:

            # Don't limit changes to b for the first few blocks of frames
            if x <= 256*3:
                max_adjust = float('inf')

            # Then adjust b aggressively for the first few minutes
            elif x <= 256*6*10: # first ~10 min
                max_adjust = 100*1000/256 # 0.1 ms per block

            # Then only allow small changes for the remainder of the run
            else:
                max_adjust = 25*1000/256 # 0.025 ms per block
            
            # Determine the correction factor
            b_corr = min(self.b_error_debt, max_adjust) # ns

            # Update the lowest b and adjust the debt
            self.b -= b_corr
            self.b_error_debt -= b_corr

            # Update m jump error debt
            if self.m_jump_error > 0:
                self.m_jump_error = max(self.m_jump_error - max_adjust, 0)
            else:
                self.m_jump_error = min(self.m_jump_error + max_adjust, 0)

        else:
            # Introduce a very small positive bias
            self.b += 25 # ns
        
        return m, self.b - self.m_jump_error


    def smoothPTS(self, new_pts):
        """ Smooth pts using linear regression.

        Arguments:
            new_pts: [float] pts of the frame

        Return:
            smoothed_pts: [float] smoothed pts

        """

        # Disable smoothing if too many resets are detected
        if self.reset_count >= 50:
            if self.reset_count == 50:
                log.info("Too many resets. Disabling smoothing function!")
                self.reset_count += 1
            return new_pts

        # Calculate linear regression params
        m, b = self.calculatePTSRegressionParams(new_pts)

        # Store last calculated fps for the longest run so far
        if self.n > self.last_calculated_fps_n:
            self.last_calculated_fps = 1e9/m
            self.last_calculated_fps_n = self.n

        # On initial run or after a reset
        if self.n == 1:
            smoothed_pts = new_pts

        # Calculate smoothed pts from regression parameters
        else:
            smoothed_pts = m*self.n + b

            # Reset regression on dropped frame (raw pts is more than 1 frame late)
            if new_pts - smoothed_pts > self.expected_m:

                self.reset_count += 1
                self.n = 0
                self.sum_x = 0
                self.sum_y = 0
                self.sum_xx = 0
                self.sum_xy = 0
                self.startup_frames = 25*60*10 # 10 minutes
                self.m_jump_error = 0
                self.b_error_debt = 0
                self.last_m_err = float('inf')
                self.last_m_err_n = 0
                log.info('smooth_pts detected dropped frame. Resetting regression parameters.')

                return new_pts
        
        return smoothed_pts


    def read(self):
        """ Retrieve frames and timestamp.

        Return:
        (tuple): (ret, frame, timestamp) where ret is a boolean indicating success,
                 frame is the captured frame, and timestamp is the frame timestamp.
        """
        ret, frame, timestamp = False, None, None

        # Read Video file frame
        if self.video_file is not None:
            ret, frame = self.device.read()
            if ret:
                timestamp = None # assigned later
        
        # Read capture device frame
        else:

            # GStreamer
            if GST_IMPORTED and (self.config.media_backend == 'gst') and (not self.media_backend_override):

                # Pull a sample from the appsink
                sample = self.device.emit("pull-sample")
                if not sample:
                    log.info("GStreamer pipeline did not emit a sample.")
                    return False, None, None

                # Try to get the buffer from the sample
                buffer = sample.get_buffer()
                if not buffer:
                    log.error("Failed to get buffer from sample.")
                    return False, None, None

                gst_timestamp_ns = buffer.pts  # GStreamer timestamp in nanoseconds

                # Sanity check for pts value
                max_expected_ns = 24*60*60*1e9  # 24 hours in nanoseconds
                if not (0 < gst_timestamp_ns <= max_expected_ns):
                    log.info("Unexpected PTS value: {}.".format(gst_timestamp_ns))
                    return False, None, None

                ret, map_info = buffer.map(Gst.MapFlags.READ)
                if not ret:
                    log.info("GStreamer Buffer did not contain a frame.")
                    return False, None, None

                # Handling for grayscale conversion
                frame = self.handleGrayscaleConversion(map_info)

                # Smooth raw pts and calculate actual timestamp
                smoothed_pts = self.smoothPTS(gst_timestamp_ns)
                timestamp = self.start_timestamp + (smoothed_pts/1e9)

                buffer.unmap(map_info)

            # OpenCV
            else:
                ret, frame = self.device.read()
                if ret:
                    timestamp = time.time()

        return ret, frame, timestamp


    def extractRtspUrl(self, input_string):
        """
        Return validated camera url
        """

        # Define a regular expression pattern for RTSP URLs
        pattern = r'rtsp://[^\s]+'

        # Search for the pattern in the input string
        match = re.search(pattern, input_string)

        # Extract, format, and return the RTSP URL
        if match:

            rtsp_url = match.group(0)

            # Add '/' if it's missing from '.sdp' URL
            if rtsp_url.endswith('.sdp'):
                rtsp_url += '/'

            return rtsp_url

        # If no match is found, return None or handle as appropriate        
        else:
            log.error("No RTSP URL found in the input string: {}".format(input_string))
            raise ValueError("No RTSP URL found in the input string: {}".format(input_string))
            

    def isGrayscale(self, frame):
        """
        Return True if all color channels contain identical data.
        """

        # Check if the R, G, and B channels are equal
        b, g, r = cv2.split(frame)
        if np.array_equal(r, g) and np.array_equal(g, b):
            return True
        
        return False

    
    def handleGrayscaleConversion(self, map_info):
        """Handle conversion of frame to grayscale if necessary."""
        if not self.convert_to_gray:
            return np.ndarray(shape=self.frame_shape, buffer=map_info.data, dtype=np.uint8)

        # Convert to grayscale by selecting a specific channel
        bgr_frame = np.ndarray(shape=self.frame_shape, buffer=map_info.data, dtype=np.uint8)
        gray_frame = bgr_frame[:, :, 0]  # Assuming the blue channel for grayscale
        return gray_frame


    def createGstreamDevice(self, video_format, max_retries=5, retry_interval=1):
        """
        Creates a GStreamer pipeline for capturing video from an RTSP source and 
        initializes playback with specific configurations.

        The method also sets an initial timestamp for the pipeline's operation.

        Arguments:
            video_format: [str] The desired video format for the conversion, 
                e.g., 'BGR', 'GRAY8', etc.
            max_retries: [int] The maximum number of retry attempts
            retry_interval: [float] The number of seconds to wait between retries


        Returns:
            Gst.Element: The appsink element of the created GStreamer pipeline, 
                which can be used for further processing of the captured video frames.
        """

        device_url = self.extractRtspUrl(self.config.deviceID)

        device_str = ("rtspsrc  buffer-mode=1 protocols=tcp tcp-timeout=5000000 retry=5 "
                      "location=\"{}\" ! "
                      "rtph264depay ! h264parse ! avdec_h264").format(device_url)

        conversion = "videoconvert ! video/x-raw,format={}".format(video_format)
        pipeline_str = ("{} ! queue leaky=downstream max-size-buffers=100 max-size-bytes=0 max-size-time=0 ! "
                        "{} ! queue max-size-buffers=100 max-size-bytes=0 max-size-time=0 ! "
                        "appsink max-buffers=100 drop=true sync=0 name=appsink").format(device_str, conversion)

        log.debug("GStreamer pipeline string: {}".format(pipeline_str))

        # Set the pipeline to PLAYING state with retries
        for attempt in range(max_retries):
            
            # Parse and create the pipeline
            self.pipeline = Gst.parse_launch(pipeline_str)

            # Set the pipeline to PLAYING state
            self.pipeline.set_state(Gst.State.PLAYING)

            # Capture time
            start_time = time.time()

            # Wait for the state change to complete
            state_change_return, current_state, pending_state = self.pipeline.get_state(Gst.CLOCK_TIME_NONE)

            # Check if the state change was successful
            if state_change_return != Gst.StateChangeReturn.FAILURE and current_state == Gst.State.PLAYING:
                log.info("Pipeline is in PLAYING state.")

                # Calculate camera latency from config parameters
                total_latency = self.config.camera_buffer/self.config.fps + self.config.camera_latency

                # Calculate stream start time
                self.start_timestamp = start_time - total_latency

                # Log start time
                start_time_str = (datetime.datetime.fromtimestamp(self.start_timestamp)
                                  .strftime('%Y-%m-%d %H:%M:%S.%f'))

                log.info("Start time is {}".format(start_time_str))

                return self.pipeline.get_by_name("appsink")

            # Log the failure and retry if attempts are left
            log.error("Attempt {}: Pipeline did not transition to PLAYING state, current state is {}. \
                      Retrying in {} seconds."
                      .format(attempt + 1, current_state, retry_interval))

            time.sleep(retry_interval)

        log.error("Failed to set pipeline to PLAYING state after {} attempts.".format(max_retries))
        return False


    def initVideoDevice(self):
        """ Initialize the video device. """

        # Assume OpenCV as the default video device type, which will be overridden if GStreamer is used
        self.video_device_type = "cv2"

        # Use a file as the video source
        if self.video_file is not None:
            self.device = cv2.VideoCapture(self.video_file)

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

                    # Try pinging 500 times
                    ping_success = False

                    for i in range(500):

                        print('Trying to ping the IP camera...')
                        ping_success = ping(ip)

                        if ping_success:
                            log.info("Camera IP ping successful! Waiting  10 seconds. ")

                            # Wait for camera to finish booting up
                            time.sleep(10)
                            break

                        time.sleep(5)

                    if not ping_success:
                        log.error("Can't ping the camera IP!")
                        return False

                else:
                    log.error("Can't find the camera IP!")
                    return False


            # Init the video device
            log.info("Initializing the video device...")
            log.info("Device: " + str(self.config.deviceID))

            # If media backend is set to gst, but GStreamer is not available, switch to openCV
            if (self.config.media_backend == 'gst') and (not GST_IMPORTED):
                log.info("GStreamer is not available. Switching to alternative.")
                self.media_backend_override = True

            if (self.config.media_backend == 'gst') and GST_IMPORTED:
                
                log.info("Initialize GStreamer Standalone Device.")
                
                # Initialize Smoothing parameters
                self.reset_count += 1
                self.n = 0
                self.sum_x = 0
                self.sum_y = 0
                self.sum_xx = 0
                self.sum_xy = 0
                self.startup_frames = 25*60*10 # 10 minutes
                self.b = 0
                self.b_error_debt = 0
                self.m_jump_error = 0
                self.last_m_err = float('inf')
                self.last_m_err_n = 0

                try:
                    # Initialize GStreamer
                    Gst.init(None)

                    # Create and start a GStreamer pipeline
                    self.device = self.createGstreamDevice('BGR', max_retries=5, retry_interval=1)
                    self.pts_buffer = []  # Reset pts buffer

                    # Attempt to get a sample and determine the frame shape
                    sample = self.device.emit("pull-sample")
                    if not sample:
                        raise ValueError("Could not obtain sample.")

                    buffer = sample.get_buffer()
                    ret, map_info = buffer.map(Gst.MapFlags.READ)
                    if not ret:
                        raise ValueError("Could not obtain frame.")

                    # Extract video information from caps
                    caps = sample.get_caps()
                    if not caps:
                        raise ValueError("Sample caps are None.")
                        
                    structure = caps.get_structure(0)
                    if not structure:
                        raise ValueError("Could not determine frame shape.")
                    
                    # Extract width, height, and format, and create frame
                    width = structure.get_value('width')
                    height = structure.get_value('height')
                    self.frame_shape = (height, width, 3)
                    frame = np.ndarray(shape=self.frame_shape, buffer=map_info.data, dtype=np.uint8)
                    
                    # Check if frame is grayscale and set flag
                    self.convert_to_gray = self.isGrayscale(frame)
                    log.info("Video format: BGR, {}P, color: {}".format(height, not self.convert_to_gray))

                    # Set the video device type
                    self.video_device_type = "gst"

                    return True

                except Exception as e:
                    log.info("Error initializing GStreamer, switching to alternative. Error: {}".format(e))
                    self.media_backend_override = True
                    self.releaseResources()


            if self.config.media_backend == 'v4l2':
                try:
                    log.info("Initialize OpenCV Device with v4l2.")
                    self.device = cv2.VideoCapture(self.config.deviceID, cv2.CAP_V4L2)
                    self.device.set(cv2.CAP_PROP_CONVERT_RGB, 0)

                    return True
                
                except Exception as e:
                    log.info("Could not initialize OpenCV with v4l2. Initialize "
                             "OpenCV Device without v4l2 instead. Error: {}".format(e))
                    self.media_backend_override = True
                    self.releaseResources()


            elif (self.config.media_backend == 'cv2') or self.media_backend_override:
                log.info("Initialize OpenCV Device.")
                self.device = cv2.VideoCapture(self.config.deviceID)

                return True

            else:
                error_msg  = "Invalid media backend: {}\n".format(self.config.media_backend)
                error_msg += "Or GStreamer is not available but is set as the media_backend."
                raise ValueError(error_msg)

        return False


    def releaseResources(self):
        """Releases resources for GStreamer and OpenCV devices."""

        if self.pipeline:

            try:
                self.pipeline.set_state(Gst.State.NULL)

                if abs(self.last_calculated_fps - self.config.fps) > 0.0005 and self.last_calculated_fps_n > 25*60*60:
                    log.info('Config file fps appears to be inaccurate. Consider updating the config file!')
                log.info("Last calculated FPS: {:.6f} at frame {}, config FPS: {}, resets: {}, startup status: {}"
                         .format(self.last_calculated_fps, self.last_calculated_fps_n, self.config.fps, self.reset_count, self.startup_flag))

                time.sleep(5)
                log.info('GStreamer Video device released!')

            except Exception as e:
                log.error('Error releasing GStreamer pipeline: {}'.format(e))
                
        if self.device:

            try:

                if self.video_device_type == "cv2":
                    self.device.release()
                    log.info('OpenCV Video device released!')

            except Exception as e:
                log.error('Error releasing OpenCV device: {}'.format(e))

            finally:
                self.device = None  # Reset device to None after releasing


    def run(self):
        """ Capture frames.
        """
        
        # Init the video device
        while not self.exit.is_set() and not self.initVideoDevice():
            log.info('Waiting for the video device to be connect...')
            time.sleep(5)

        if self.device is None:

            log.info('The video source could not be opened!')
            self.exit.set()
            return False

        # Wait until the device is opened
        device_opened = False
        for i in range(20):
            time.sleep(1)
            if self.deviceIsOpened():
                device_opened = True
                break

        # If the device could not be opened, stop capturing
        if not device_opened:
            log.info('The video source could not be opened!')
            self.exit.set()
            return False

        else:
            log.info('Video device opened!')


        # Keep track of the total number of frames
        total_frames = 0

        # For video devices only (not files), throw away the first 10 frames
        if (self.video_file is None) and (self.video_device_type == "cv2"):

            first_skipped_frames = 10
            for i in range(first_skipped_frames):
                self.read()

            total_frames = first_skipped_frames

        # If a video file was used, set the time of the first frame to the time read from the file name
        if self.video_file is not None:
            time_stamp = "_".join(os.path.basename(self.video_file).split("_")[1:4])
            time_stamp = time_stamp.split(".")[0]
            video_first_time = datetime.datetime.strptime(time_stamp, "%Y%m%d_%H%M%S_%f")
            log.info("Using a video file: " + self.video_file)
            log.info("Setting the time of the first frame to: " + str(video_first_time))

            # Convert the first time to a UNIX timestamp
            video_first_timestamp = (video_first_time - datetime.datetime(1970, 1, 1)).total_seconds()

        # Use the first frame buffer to start - it will be flip-flopped between the first and the second
        #   buffer during capture, to prevent any data loss
        buffer_one = True

        wait_for_reconnect = False

        last_frame_timestamp = False
        
        # Run until stopped from the outside
        while not self.exit.is_set():

            # Wait until the compression is done (only when a video file is used)
            if self.video_file is not None:
                
                wait_for_compression = False

                if buffer_one:
                    if self.startTime1.value == -1:
                        wait_for_compression = True
                else:
                    if self.startTime2.value == -1:
                        wait_for_compression = True

                if wait_for_compression:
                    log.debug("Waiting for the {:d}. compression thread to finish...".format(int(not buffer_one) + 1))
                    time.sleep(0.1)
                    continue

            
            if buffer_one:
                self.startTime1.value = 0
            else:
                self.startTime2.value = 0
            

            # If the video device was disconnected, wait 5s for reconnection
            if wait_for_reconnect:

                print('Reconnecting...')

                while not self.exit.is_set() and not self.initVideoDevice():

                    log.info('Waiting for the video device to be reconnected...')

                    time.sleep(5)

                    if self.device is None:
                        print("The video device couldn't be connected! Retrying...")
                        continue


                    if self.exit.is_set():
                        break

                    # Read the frame
                    log.info("Reading frame...")
                    ret, _, _ = self.read()
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
            max_frame_interval_normalized = 0.0
            max_frame_age_seconds = 0.0


            # Capture a block of 256 frames
            block_frames = 256

            log.info('Grabbing a new block of {:d} frames...'.format(block_frames))
            for i in range(block_frames):


                # Read the frame (keep track how long it took to grab it)
                t1_frame = time.time()
                ret, frame, frame_timestamp = self.read()
                t_frame = time.time() - t1_frame


                # If the video device was disconnected, wait for reconnection
                if (self.video_file is None) and (not ret):

                    log.info('Frame grabbing failed, video device is probably disconnected!')
                    self.releaseResources()
                    wait_for_reconnect = True
                    break


                # If a video file is used, compute the time using the time from the file timestamp
                if self.video_file is not None:
                
                    frame_timestamp = video_first_timestamp + total_frames/self.config.fps

                    # print("tot={:6d}, i={:3d}, fps={:.2f}, t={:.8f}".format(total_frames, i, self.config.fps, frame_timestamp))

                    
                # Set the time of the first frame
                if i == 0:

                    # Initialize last frame timestamp if it's not set
                    if not last_frame_timestamp:
                        last_frame_timestamp = frame_timestamp
                    
                    # Always set first frame timestamp in the beginning of the block
                    first_frame_timestamp = frame_timestamp


                # If the end of the video file was reached, stop the capture
                if self.video_file is not None: 
                    if (frame is None) or (not self.deviceIsOpened()):

                        log.info("End of video file!")
                        log.debug("Video end status:")
                        log.debug("Frame:" + str(frame))
                        log.debug("Device open:" + str(self.deviceIsOpened()))

                        self.exit.set()
                        time.sleep(0.1)
                        break


                # Check if frame is dropped if it has been more than 1.5 frames than the last frame
                elif (frame_timestamp - last_frame_timestamp) >= self.time_for_drop:
                    
                    # Calculate the number of dropped frames
                    n_dropped = int((frame_timestamp - last_frame_timestamp)*self.config.fps)

                    self.dropped_frames.value += n_dropped

                    if self.config.report_dropped_frames:
                        log.info("{}/{} frames dropped or late! Time for frame: {:.3f}, convert: {:.3f}, assignment: {:.3f}".format(
                            str(n_dropped), str(self.dropped_frames.value), t_frame, t_convert, t_assignment))


                # If cv2:
                if (self.config.media_backend != 'gst') and not self.media_backend_override:
                    # Calculate the normalized frame interval between the current and last frame read, normalized by frames per second (fps)
                    frame_interval_normalized = (frame_timestamp - last_frame_timestamp)/(1/self.config.fps)
                    # Update max_frame_interval_normalized for this cycle
                    max_frame_interval_normalized = max(max_frame_interval_normalized, frame_interval_normalized)

                # If GStreamer:
                else:
                    # Calculate the time difference between the current time and the frame's timestamp
                    frame_age_seconds = time.time() - frame_timestamp
                    # Update max_frame_age_seconds for this cycles
                    max_frame_age_seconds = max(max_frame_age_seconds, frame_age_seconds)

                # On the last loop, report late or dropped frames
                if i == block_frames - 1:

                    # For cv2, show elapsed time since frame read to assess loop performance
                    if self.config.media_backend != 'gst' and not self.media_backend_override:
                        log.info("Block's max frame interval: {:.3f} (normalized). Run's late frames: {}"
                                 .format(max_frame_interval_normalized, self.dropped_frames.value))
                    
                    # For GStreamer, show elapsed time since frame capture to assess sink fill level
                    else:
                        log.info("Block's max frame age: {:.3f} seconds. Run's dropped frames: {}"
                                 .format(max_frame_age_seconds, self.dropped_frames.value))

                last_frame_timestamp = frame_timestamp
                

                ### Convert the frame to grayscale ###

                t1_convert = time.time()

                # Convert the frame to grayscale
                #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Convert the frame to grayscale
                if len(frame.shape) == 3:

                    # If a color image is given, take the green channel
                    if frame.shape[2] == 3:

                        gray = frame[:, :, 1]

                    # If UYVY image given, take luma (Y) channel
                    elif self.config.uyvy_pixelformat and (frame.shape[2] == 2):
                        gray = frame[:, :, 1]

                    # Otherwise, take the first available channel
                    else:
                        gray = frame[:, :, 0]

                else:
                    gray = frame


                # Cut the frame to the region of interest (ROI)
                gray = gray[self.config.roi_up:self.config.roi_down, \
                    self.config.roi_left:self.config.roi_right]

                # Track time for frame conversion
                t_convert = time.time() - t1_convert


                ### ###




                # Assign the frame to shared memory (track time to do so)
                t1_assign = time.time()
                if buffer_one:
                    self.array1[i, :gray.shape[0], :gray.shape[1]] = gray
                else:
                    self.array2[i, :gray.shape[0], :gray.shape[1]] = gray

                t_assignment = time.time() - t1_assign



                # Keep track of all captured frames
                total_frames += 1


            if self.exit.is_set():
                wait_for_reconnect = False
                log.info('Capture exited!')
                break


            if not wait_for_reconnect:

                # Set the starting value of the frame block, which indicates to the compression that the
                # block is ready for processing
                if buffer_one:
                    self.startTime1.value = first_frame_timestamp

                else:
                    self.startTime2.value = first_frame_timestamp

                log.info('New block of raw frames available for compression with starting time: {:s}'
                         .format(str(first_frame_timestamp)))

            
            # Switch the frame block buffer flags
            buffer_one = not buffer_one
            if self.config.report_dropped_frames:
                log.info('Estimated FPS: {:.3f}'.format(block_frames/(time.time() - t_block)))
        

        log.info('Releasing video device...')
        self.releaseResources()