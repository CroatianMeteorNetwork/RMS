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
import sys
import ctypes
import traceback

import re
import time
import datetime
import copy
import os.path
from multiprocessing import Process, Event, Value, Array
import threading

import cv2
import numpy as np
import socket
import errno
import json


from RMS.Misc import obfuscatePassword
from RMS.Routines.GstreamerCapture import GstVideoFile
from RMS.Formats.ObservationSummary import getObsDBConn, addObsParam
from RMS.RawFrameSave import RawFrameSaver
from RMS.Misc import RmsDateTime, mkdirP, UTCFromTimestamp
from RMS.Formats import FTfile, FTStruct
from RMS.Logger import getLogger, gstDebugLogger
from RMS.CaptureModeSwitcher import switchCameraMode
import Utils.CameraControl as cc

# Get the logger from the main module
log = getLogger("logger")

try:
    # py3
    from urllib.parse import urlparse
except ImportError:
    # py2
    from urlparse import urlparse


GST_IMPORTED = False
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    GST_IMPORTED = True

except ImportError as e:
    log.info('Could not import gi: {}. Using OpenCV.'.format(e))

except ValueError as e:
    log.info('Could not import Gst: {}. Using OpenCV.'.format(e))


# Define probe result constants
class RtspProbeResult:
    """
    Constants representing possible RTSP probe results.
    
    SUCCESS: Connection successful
    NETWORK_DOWN: Local network interface is down or unreachable
    HOST_UNREACHABLE: Network up but target host cannot be reached
    CONNECTION_REFUSED: Host is up but actively refusing RTSP connections
    TIMEOUT: Connection attempt exceeded specified timeout
    DNS_ERROR: Unable to resolve hostname to IP address
    UNKNOWN_ERROR: Other unspecified connection errors
    """
    SUCCESS = "SUCCESS"
    NETWORK_DOWN = "NETWORK_DOWN"          # No network connectivity
    HOST_UNREACHABLE = "HOST_UNREACHABLE"  # Can't reach the host  
    CONNECTION_REFUSED = "CONNECTION_REFUSED" # Host reachable but RTSP port closed
    TIMEOUT = "TIMEOUT"                    # Connection attempt timed out
    DNS_ERROR = "DNS_ERROR"                # Can't resolve hostname
    UNKNOWN_ERROR = "UNKNOWN_ERROR"        # Other connection errors


class BufferedCapture(Process):
    """ Capture from device to buffer in memory.
    """
    
    running = False
    
    def __init__(self, array1, start_time1, array2, start_time2, config, video_file=None, night_data_dir=None,
                 saved_frames_dir=None, daytime_mode=None, camera_mode_switch_trigger=None):
        """ Populate arrays with (startTime, frames) after startCapture is called.
        
        Arguments:
            array1: numpy array in shared memory that is going to be filled with frames
            start_time1: float in shared memory that holds time of first frame in array1
            array2: second numpy array in shared memory
            start_time2: float in shared memory that holds time of first frame in array2

        Keyword arguments:
            video_file: [str] Path to the video file, if it was given as the video source. None by default.
            night_data_dir: [str] Path to the directory where night data is stored. None by default.
            saved_frames_dir: [str] Path to the directory where saved frames are stored. None by default.
            daytime_mode: [multiprocessing.Value] Shared boolean variable to communicate camera mode switch
                direction (daytime or nighttime).
            camera_mode_switch_trigger: [multiprocessing.Value] Shared boolean variable to trigger camera 
                mode switch at the right time.
        """
        
        super(BufferedCapture, self).__init__()
        
        # Store configuration and paths (immutable data is safe to pass to child process)
        self.config = config
        self.video_file = video_file
        self.night_data_dir = night_data_dir
        self.saved_frames_dir = saved_frames_dir

        # make sure the flags are always real shared Values
        if daytime_mode is None:
            self.daytime_mode = Value(ctypes.c_bool, False)       # default: "night"
        else:
            self.daytime_mode = daytime_mode

        if camera_mode_switch_trigger is None:
            self.camera_mode_switch_trigger = Value(ctypes.c_bool, False)
        else:
            self.camera_mode_switch_trigger = camera_mode_switch_trigger

        # Store shared memory arrays and values for compressor (these are designed for multiprocessing)
        self.array1 = array1
        self.start_time1 = start_time1
        self.array2 = array2
        self.start_time2 = start_time2
        self.start_time1.value = 0
        self.start_time2.value = 0

        # Initialize shared values for raw frame saving (these are designed for multiprocessing)
        if self.config.save_frames:

            # Frame saving block size - these many raw frames are written to buffer before saving to disk
            self.num_raw_frames = 10

            self.start_raw_time1 = Value('d', 0.0)
            self.start_raw_time2 = Value('d', 0.0)
            self.shared_timestamps_base = Array(ctypes.c_double, self.num_raw_frames)
            self.shared_timestamps_base2 = Array(ctypes.c_double, self.num_raw_frames)

        # Initialize shared counter for dropped frames
        self.dropped_frames = Value('i', 0)

        # Flag for process control
        self.exit = Event()

        # handle for the Gst bus-poller thread
        self._bus_thread = None


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
        waiting_to_join = 60
        log.info("Waiting up to {} seconds for capture to join...".format(waiting_to_join))
        
        # Track how many seconds we actually waited
        seconds_waited = 0
        
        for i in range(waiting_to_join):
            seconds_waited = i + 1
            if self.is_alive():
                time.sleep(1)
            else:
                break
        
        # Log the outcome based on final state
        if not self.is_alive():
            log.info("Capture joined successfully after {} seconds".format(seconds_waited))
        else:
            log.info("Timed out after waiting {} seconds, capture thread still alive".format(seconds_waited))
            log.info("This is a known issue with GStreamer pipelines not releasing all threads")
            log.info('Terminating capture...')
            self.terminate()

        # Clean up shared memory resources
        try:
            log.debug('Freeing shared memory resources...')
            # Frame buffers
            del self.array1
            del self.array2

            # Raw frame and timestamp buffers if they exist
            if self.config.save_frames:
                del self.shared_timestamps_base
                del self.shared_timestamps_base2

                if hasattr(self, 'shared_raw_array_base'):
                    del self.shared_raw_array_base
                    del self.shared_raw_array_base2
                    del self.shared_raw_array
                    del self.shared_raw_array2

            log.debug('Shared memory resources freed successfully')

        except Exception as e:
            log.error('Error freeing shared memory: {}'.format(e))
            log.debug(repr(traceback.format_exception(*sys.exc_info())))

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

                    # Use a 10-second timeout to avoid indefinite blocking while checking if the device is in the PLAYING state
                    state = self.device.get_state(Gst.SECOND * 10).state

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


    def read(self, check_color=False):
        """ Retrieve frames and timestamp.

        Arguments:
            check_color: [bool] whether to check if frame contains color information

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

                # Pull a frame from the GStreamer pipeline with a .5 sec timeout
                sample = self.device.emit("try-pull-sample", 500 * Gst.MSECOND)
                if not sample:
                    log.info("GStreamer pipeline did not emit a sample.")
                    return False, None, None
                
                # Extract the frame buffer and timestamp
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

                # Convert to np.ndarray
                frame = np.ndarray(shape=self.frame_shape, buffer=map_info.data, dtype=np.uint8)

                # Smooth raw pts and calculate actual timestamp
                smoothed_pts = self.smoothPTS(gst_timestamp_ns)
                timestamp = self.start_timestamp + (smoothed_pts/1e9)

                buffer.unmap(map_info)

            # OpenCV
            else:
                ret, frame = self.device.read()
                if ret:
                    timestamp = time.time()

            # Check if frame contains color information
            if check_color:
                self.convert_to_gray = self.isGrayscale(frame)

            # Handling for grayscale conversion
            frame = self.handleGrayscaleConversion(frame)

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
    

    def probeRtspService(self, max_attempts=720, probe_interval=10, timeout=1):
        """
        Test RTSP service availability by attempting TCP connection to the service port.
        Uses TCP connection only - does not validate RTSP protocol
        
        Performs a thorough connection test by:
        1. Resolving hostname via DNS
        2. Creating a TCP socket connection to the RTSP port
        3. Analyzing any connection failures
        4. Retrying with backoff if connection fails
        
        Args:
            max_attempts (int, optional): Maximum number of connection attempts before giving up.
                Defaults to 720.
            probe_interval (int, optional): Time in seconds between connection attempts.
                Defaults to 10 seconds.
            timeout (int, optional): Socket connection timeout in seconds.
                Defaults to 1 second.
        
        Returns:
            tuple: A pair (success, status) where:
                - success (bool): True if connection was successful, False otherwise
                - status (str): One of the RtspProbeResult status strings:
                    - SUCCESS: Connection successful
                    - NETWORK_DOWN: Local network interface is down
                    - HOST_UNREACHABLE: Cannot reach the target host
                    - CONNECTION_REFUSED: Host up but RTSP port is closed
                    - TIMEOUT: Connection attempt timed out
                    - DNS_ERROR: Cannot resolve hostname
                    - UNKNOWN_ERROR: Other connection failures
                
        """
        try:
            # Parse RTSP URL to get host and port
            device_url = self.extractRtspUrl(self.config.deviceID)
            parsed = urlparse(device_url)
            host = parsed.hostname
            port = parsed.port or 554

            last_error = None
            
            for attempt in range(max_attempts):
                try:
                    # Try to resolve hostname first
                    try:
                        socket.gethostbyname(host)
                    except socket.gaierror:
                        last_error = RtspProbeResult.DNS_ERROR
                        raise

                    # Create socket with timeout
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(timeout)
                    
                    # Try to connect
                    result = sock.connect_ex((host, port))
                    sock.close()
                    
                    if result == 0:
                        log.info("RTSP service ready after {} attempts".format(attempt + 1))
                        return True, RtspProbeResult.SUCCESS
                    
                    # Analyze specific connection errors
                    if result in (errno.ENETUNREACH, errno.ENETDOWN):
                        last_error = RtspProbeResult.NETWORK_DOWN
                    elif result in (errno.EHOSTUNREACH, errno.EHOSTDOWN):
                        last_error = RtspProbeResult.HOST_UNREACHABLE
                    elif result == errno.ECONNREFUSED:
                        last_error = RtspProbeResult.CONNECTION_REFUSED
                    elif result == errno.ETIMEDOUT:
                        last_error = RtspProbeResult.TIMEOUT
                    else:
                        last_error = RtspProbeResult.UNKNOWN_ERROR
                        
                except socket.gaierror:
                    last_error = RtspProbeResult.DNS_ERROR
                except socket.timeout:
                    last_error = RtspProbeResult.TIMEOUT
                except socket.error as e:
                    if e.errno in (errno.ENETUNREACH, errno.ENETDOWN):
                        last_error = RtspProbeResult.NETWORK_DOWN
                    elif e.errno in (errno.EHOSTUNREACH, errno.EHOSTDOWN):
                        last_error = RtspProbeResult.HOST_UNREACHABLE
                    else:
                        last_error = RtspProbeResult.UNKNOWN_ERROR
                    log.debug("RTSP probe attempt {} failed: {}".format(attempt + 1, e))
                
                error_messages = {
                    RtspProbeResult.NETWORK_DOWN: "Network appears to be down",
                    RtspProbeResult.HOST_UNREACHABLE: "Camera is unreachable",
                    RtspProbeResult.CONNECTION_REFUSED: "RTSP service not accepting connections",
                    RtspProbeResult.TIMEOUT: "Connection attempt timed out",
                    RtspProbeResult.DNS_ERROR: "Cannot resolve camera hostname",
                    RtspProbeResult.UNKNOWN_ERROR: "Unknown connection error"
                }
                
                print('Trying to connect to camera RTSP service... (attempt {}) - {}'.format(
                    attempt + 1, error_messages[last_error]))
                time.sleep(probe_interval)
                
            log.error("RTSP service not responding after all attempts. Last error: {}".format(
                error_messages[last_error]))
            return False, last_error
            
        except Exception as e:
            log.error("Error probing RTSP service: {}".format(e))
            return False, RtspProbeResult.UNKNOWN_ERROR
                

    def isGrayscale(self, frame, stride=64):
        """
        Quickly check if a frame is grayscale by sampling pixels along the diagonal.
        If all three channels match on those diagonal samples, return True.
        If an IndexError is raised (i.e., frame is single-channel), also return True.
        This trades completeness for speed, as only the diagonal is checked.

        Args:
            frame (numpy.ndarray): The image frame to check (usually BGR or GRAY).
            stride (int): Spacing for diagonal sampling, skipping many pixels for efficiency.

        Returns:
            bool: True if all sampled channels match (or frame is single-channel), otherwise False.
        """

        # We don't explicitly check frame.shape first; instead we rely on an IndexError
        # if 'frame' is single-channel (which is inherently grayscale).
        # This is faster than an extra dimension check for most BGR GMN stations 

        try:
            # If diagonal samples are not identical, frame is color
            sampled = frame[::stride, ::stride]
            is_gray = np.all(sampled[..., 0] == sampled[..., 1]) and \
                    np.all(sampled[..., 1] == sampled[..., 2])
            
        except IndexError:
             # If IndexError, frame is grayscale
            is_gray = True

        return is_gray


    def handleGrayscaleConversion(self, frame):
        """Handle conversion of frame to grayscale if necessary.

            Camera outputs BGR (3 channels) even in night mode. For efficiency, we save raw frames in:
            - Grayscale (1 channel) when all channels are identical
            - Full BGR (3 channels) when they differ

            Note: While raw frames are saved in color when available, frames are converted
            to grayscale before compression in the processing pipeline

        Args:
            frame: a numpy.ndarray frame

        Returns:
            numpy.ndarray: Frame data either as grayscale (2D) or BGR (3D) array
        """

        # First check if frame is None to prevent NoneType subscript error
        if frame is None:
            return None

        # We don't explicitly check frame.shape first; instead we rely on an IndexError
        # if 'frame' is single-channel (which is inherently grayscale).
        # This is faster than an extra dimension check for most BGR GMN stations 

        try:
            # If frame channels are not identical (color), return all 3 channels
            if not self.convert_to_gray:
                return frame

            try:
                # If frame channels are identical (gray), extract green channel for grayscale
                return frame[:, :, 1]
        
            except IndexError:
                # If IndexError occurs, frame is already grayscale (single-channel)
                return frame
            
        except Exception as e:
            log.error('Error in grayscale conversion: {}'.format(e))
            log.debug('Frame shape: {}'.format(frame.shape if frame is not None else None))
            return None


    def moveSegment(self, splitmuxsink, fragment_id):
        """
        Custom callback for splitmuxsink's format-location signal to name and move each segment as its
        created. Generates a timestamp-based folder structure: Year/Day-Of-Year/Hour/ per video segment.

        Arguments:
          splitmuxsink [GstElement]: The splitmuxsink object itself, included in arguments as GStreamer expects it.
          fragment_id [int]: Fragment / segment number of the new clip

        Returns:
          full_path [str]: Full path to save this new video segment to
        """

        # Segment name is based on timestamp recorded during last segment save
        segment_time = UTCFromTimestamp.utcfromtimestamp(self.last_segment_savetime)
        self.last_segment_savetime = time.time()
        segment_filename = segment_time.strftime("{}_%Y%m%d_%H%M%S_video.mkv".format(self.config.stationID))
        segment_subpath = os.path.join(self.config.data_dir, self.config.video_dir, segment_time.strftime("%Y/%Y%m%d-%j/%Y%m%d-%j_%H"))

        # Create full path for the segment
        mkdirP(segment_subpath)
        segment_full_path = os.path.join(segment_subpath, segment_filename)
        log.info("Created new video segment #{} at: {}".format(fragment_id, segment_full_path))

        # Return full path to splitmux's callback
        return segment_full_path

      
    def handleStateChange(self, pipeline, target_state, timeout=60):
        """Handle GStreamer pipeline state changes with proper synchronization.
        
        Transitions pipeline through state sequence (NULL->READY->PAUSED->PLAYING),
        ensuring each state change is complete before proceeding. Uses explicit synchronization
        to prevent race conditions.

        For live sources like RTSP, accepts both SUCCESS and NO_PREROLL as valid state changes.

        Args:
            pipeline: The GStreamer pipeline to change state
            target_state: The target state to reach (usually Gst.State.PLAYING)
            timeout: Maximum seconds to wait for each state change (default 60)

        tuple: (success, start_time) where:
            - success (bool): True if state change succeeded, False if any step failed
            - start_time (float or None): Timestamp when PAUSED state was initiated, or None if not reached
        """

        try:
            # Initialize start time
            start_time = None

            # Get current pipeline state
            ret, current, pending = pipeline.get_state(0)
            log.debug("Current pipeline state: {}, pending: {}".format(current.value_nick, pending.value_nick))

            # Define the sequence of states we need to go through
            target_sequence = [Gst.State.READY, Gst.State.PAUSED, Gst.State.PLAYING]

            # Find where we are in the sequence (-1 if current state isn't in sequence)
            current_index = target_sequence.index(current) if current in target_sequence else -1
            target_index = target_sequence.index(target_state)
            
            # Step through each state change needed to reach target
            for state in target_sequence[current_index + 1:target_index + 1]:
                log.debug("Transitioning to {} state...".format(state.value_nick))
                
                # Force synchronization before state change to prevent race conditions
                if not pipeline.sync_children_states():
                    log.warning("Sync failed before {}".format(state.value_nick))

                # Capture time just before camera starts capture
                if state == Gst.State.PAUSED:
                    start_time = time.time()

                # Request state change and wait for completion
                ret = pipeline.set_state(state)
                ret, new_state, pending = pipeline.get_state(Gst.SECOND * timeout)
                
                # Both SUCCESS and NO_PREROLL are valid (NO_PREROLL happens with live sources)
                if ret not in (Gst.StateChangeReturn.SUCCESS, Gst.StateChangeReturn.NO_PREROLL):
                    log.error("Failed to change to state {}".format(state.value_nick))
                    return False, None
                
                # Force synchronization after state change
                if not pipeline.sync_children_states():
                    log.warning("Sync failed after {}".format(state.value_nick))
                
                log.debug("Successfully transitioned to {} state".format(state.value_nick))
                    
            return True, start_time
            
        except Exception as e:
            log.error("State change error: {}".format(str(e)))
            import traceback
            log.debug(traceback.format_exc())
            return False, None


    def _busPoller(self):
        """Poll the GStreamer bus and drain queued messages.

        Runs in a background daemon thread:
                - Wakes every 5 s via ``bus.timed_pop_filtered``.
                - Logs any ``ERROR`` or ``WARNING`` message for visibility.
                - Silently discards all other message types to keep the queue small.

        The loop exits when ``self.pipeline`` becomes ``None`` inside ``releaseResources``.

        Arguments:
            None

        Return:
            None

        """
        if not GST_IMPORTED:
            return

        bus = self.pipeline.get_bus()
        mask = Gst.MessageType.ANY

        while True:
            # exit as soon as pipeline ref is cleared
            if self.pipeline is None:
                break
            
            # pop messages from bus every 5 seconds
            msg = bus.timed_pop_filtered(5*Gst.SECOND, mask)
            if not msg:
                continue

            if msg.type == Gst.MessageType.ERROR:
                err, _dbg = msg.parse_error()
                log.error("GST ERROR from %s: %s", msg.src.get_name(), err)
            elif msg.type == Gst.MessageType.WARNING:
                warn, _dbg = msg.parse_warning()
                log.warning("GST WARN  from %s: %s", msg.src.get_name(), warn)
            # all other msgs just pop & vanish


    def createGstreamDevice(self, video_format, gst_decoder='decodebin', 
                            video_file_dir=None, segment_duration_sec=30, max_retries=5, retry_interval=1):
        """
        Creates a GStreamer pipeline for capturing video from an RTSP source and 
        initializes playback with specific configurations.

        The method also sets an initial timestamp for the pipeline's operation.

        Arguments:
            video_format: [str] The desired video format for the conversion, 
                e.g., 'BGR', 'GRAY8', etc.
            
        Keyword arguments:
            gst_decoder: [str] The gst_decoder to use for the Gstreamer video stream. Default is 'decodebin'.
            video_file_dir: [str] The directory where the raw video stream should be saved. 
                If None, the raw stream will not be saved to disk. Default is None.
            segment_duration_sec: [int] The duration of each video segment in seconds. 
                Default is 30.
            max_retries: [int] The maximum number of retry attempts
            retry_interval: [float] The number of seconds to wait between retries

        Returns:
            Gst.Element: The appsink element of the created GStreamer pipeline, 
                which can be used for further processing of the captured video frames.
        """

        device_url = self.extractRtspUrl(self.config.deviceID)

        if self.config.protocol == 'udp':
            protocol_str = "protocols=udp retry=5"
            # rtspsrc_params = ("rtspsrc buffer-mode=1 protocols=udp retry=5")

        else:
            # Default to TCP
            protocol_str = "protocols=tcp tcp-timeout=5000000 retry=5"

        # Define the source up to the point where we want to branch off
        source_to_tee = (
            "rtspsrc name=src buffer-mode=1 {:s} "
            "location=\"{:s}\" ! "
            "rtph264depay ! h264parse ! tee name=t"
            ).format(protocol_str, device_url)

        # Branch for processing
        processing_branch = (
            "t. ! queue ! {:s} ! "
            "queue leaky=downstream max-size-buffers=100 max-size-bytes=0 max-size-time=0 ! "
            "videoconvert ! video/x-raw,format={:s} ! "
            "queue max-size-buffers=100 max-size-bytes=0 max-size-time=0 ! "
            "appsink max-buffers=100 drop=true sync=0 name=appsink"
            ).format(gst_decoder, video_format)
        
         # Branch for storage - if video_file_dir is not None, save the raw stream to a file
        if video_file_dir is not None:
            
            # The video will be split into segments of segment_duration_sec seconds
            # The splitmuxsink will save the segments to video_file_dir
            # The splitmuxsink will use the matroskamux muxer
            # The splitmuxsink will use the format-location signal to name and move each segment
            # queue2 smooths out the writes, but doesn't wait until the buffers fill up for writing
            storage_branch = (
                "t. ! queue2 max-size-buffers=150 max-size-bytes=2097152 max-size-time=5000000000 ! "
                "splitmuxsink name=splitmuxsink0 max-size-time={:d} muxer-factory=matroskamux"
                ).format(int(segment_duration_sec*1e9))

        # Otherwise, skip saving the raw stream to disk
        else:
            storage_branch = ""

         # Combine all parts of the pipeline
        pipeline_str = "{:s} {:s} {:s}".format(source_to_tee, processing_branch, storage_branch)

        # Obfuscate the password in the pipeline string before logging
        obfuscated_pipeline_str = obfuscatePassword(pipeline_str)

        log.debug("GStreamer pipeline string: {:s}".format(obfuscated_pipeline_str))

        # Set the pipeline to PLAYING state with retries
        for attempt in range(max_retries):
            try:
                log.info("Attempt {}: transitioning Pipeline to PLAYING state.".format(attempt + 1))
                
                # Make sure any previous pipeline is cleaned up
                if hasattr(self, 'pipeline') and self.pipeline:
                    self.releaseResources()

                # Parse and create the pipeline
                self.pipeline = Gst.parse_launch(pipeline_str)
                if not self.pipeline:
                    raise ValueError("Could not create pipeline")
                
                # Start a daemon thread that drains the GstBus so it never fills
                self._bus_thread = threading.Thread(target=self._busPoller, daemon=True)
                self._bus_thread.start()
                
                # If raw video saving is enabled, Connect the "format-location" signal to the 
                # moveSegment function
                if video_file_dir is not None:
                    
                    splitmuxsink = self.pipeline.get_by_name("splitmuxsink0")
                    splitmuxsink.connect("format-location", self.moveSegment)

                # Transition through states
                log.info("Starting pipeline state transitions...")

                success, start_time = self.handleStateChange(self.pipeline, Gst.State.PLAYING)
                if not success:
                    raise ValueError("Failed to transition pipeline to PLAYING state")

                # Calculate start timestamp
                if start_time is not None:
                    self.start_timestamp = start_time - (self.config.camera_buffer/self.config.fps + self.config.camera_latency)

                # Log start time
                start_time_str = (UTCFromTimestamp.utcfromtimestamp(self.start_timestamp)
                                    .strftime('%Y-%m-%d %H:%M:%S.%f'))

                log.info("Start time is {:s}".format(start_time_str))

                # Get appsink for frame retrieval
                appsink = self.pipeline.get_by_name("appsink")
                if not appsink:
                    raise ValueError("Could not get appsink from pipeline")
                
                log.info("Pipeline successfully created and started")
                return appsink
            
            except Exception as e:
                log.error("Attempt {} failed: {}".format(attempt + 1, str(e)))
                if attempt < max_retries - 1:
                    log.info("Waiting {} seconds before next attempt...".format(retry_interval))
                    time.sleep(retry_interval)
                    continue
                else:
                    log.error("All attempts to create pipeline failed")
                    self.releaseResources()
                    return False
        return False


    def initVideoDevice(self):
        """ Initialize the video device. """

        # Assume OpenCV as the default video device type, which will be overridden if GStreamer is used
        self.video_device_type = "cv2"

        # Use a file as the video source
        if self.video_file is not None:

            # If the video file is a GStreamer file, use the GstVideoFile class
            if GST_IMPORTED and (self.config.media_backend == 'gst'):

                self.device = GstVideoFile(self.video_file, decoder=self.config.gst_decoder,
                                           video_format=self.config.gst_colorspace)

            # Fall back to OpenCV if GStreamer is not available
            else:
                self.device = cv2.VideoCapture(self.video_file)

        # Use a device as the video source
        else:

            reprobe = False

            # If an analog camera is used, skip the probe
            if "rtsp" in str(self.config.deviceID):
                success, probe_result = self.probeRtspService()
                if not success:
                    error_messages = {
                        RtspProbeResult.NETWORK_DOWN: 
                            "Cannot connect to camera - Please check your network connection",
                        RtspProbeResult.HOST_UNREACHABLE: 
                            "Cannot reach camera - Please check if camera is powered on and connected to network",
                        RtspProbeResult.CONNECTION_REFUSED: 
                            "Camera is reachable but RTSP service is not responding - Camera might still be booting",
                        RtspProbeResult.TIMEOUT: 
                            "Connection timeout - Network might be slow or unstable",
                        RtspProbeResult.DNS_ERROR: 
                            "Cannot resolve camera hostname - Please check network DNS settings",
                        RtspProbeResult.UNKNOWN_ERROR: 
                            "Unknown connection error - Please check logs for details"
                    }
                    log.error("Camera connection failed: {}".format(error_messages[probe_result]))
                    return False
                else:
                    # After camera connection is established, if necessary inititliaze camera settings
                    # and/or perform camera mode change

                    # initialize flag to indicate if camera should be reprobed after mode change
                    reprobe = False

                    # ------------------------------------------------------------------
                    # One-time camera initialization with flag file in rms_root_dir
                    # ------------------------------------------------------------------
                    root_dir  = self.config.rms_root_dir

                    # e.g.  "XX0001.camera_init.done"
                    flag_file = os.path.join(root_dir, "{}.camera_init.done".format(self.config.stationID))

                    if self.config.initialize_camera and not os.path.exists(flag_file):
                        log.info("Running camera init sequence ...")
                        reprobe = True
                        try:
                            mode_name = "init"
                            mode_path = self.config.camera_settings_path

                            if not os.path.exists(mode_path):
                                raise FileNotFoundError("Mode file {} not found.".format(mode_path))

                            with open(mode_path, 'r') as f:
                                modes = json.load(f)

                            if mode_name not in modes:
                                raise KeyError("Mode '{}' not defined in {}.".format(mode_name, mode_path))

                            try:
                                cc.cameraControlV2(self.config, "SwitchMode", mode_name)

                                # create empty sentinel file
                                open(flag_file, "a").close()
                                log.info("Init complete - flag written to %s", flag_file)

                            except Exception as e:
                                raise RuntimeError("Failed to switch camera mode: {}".format(e))

                        except Exception as e:
                            log.warning("Camera switch to %s mode failed: %s. Will retry later.", mode_name, e)

                    # -------------------------------------------
                    # Day/night switching
                    # -------------------------------------------
                    if self.config.continuous_capture and self.config.switch_camera_modes:
                        if self.camera_mode_switch_trigger.value:
                            reprobe = True
                            switchCameraMode(self.config, self.daytime_mode, self.camera_mode_switch_trigger)

            if reprobe:
                # Wait 5 seconds for the camera to register all commands after mode switching / reboot
                log.info("Waiting for camera to register all commands...")
                time.sleep(5)
                success, probe_result = self.probeRtspService()
                if not success:
                    log.error("Camera connection failed after switching modes: {}".format(probe_result))
                    return False

            # Init the video device
            log.info("Initializing the video device...")
            log.info("Device: " + str(self.config.deviceID))

            # If media backend is set to gst, but GStreamer is not available, switch to openCV
            if (self.config.media_backend == 'gst') and (not GST_IMPORTED):
                log.info("GStreamer is not available. Switching to alternative.")
                self.media_backend_override = True

            if (self.config.media_backend == 'gst') and GST_IMPORTED and (self.media_backend_override == False):
                
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

                    # Determine if which directory to save the raw video, if any
                    if self.config.raw_video_save:
                        raw_video_dir = os.path.join(self.config.data_dir, self.config.video_dir)
                    else:
                        raw_video_dir = None

                    # Create and start a GStreamer pipeline
                    log.info("Creating GStreamer pipeline...")
                    self.device = self.createGstreamDevice(
                        self.config.gst_colorspace, gst_decoder=self.config.gst_decoder,
                        video_file_dir=raw_video_dir, segment_duration_sec=self.config.raw_video_duration,
                        max_retries=5, retry_interval=1
                        )

                    if not self.device:
                        raise ValueError("Could not create GStreamer pipeline.")
                    
                    log.info("GStreamer pipeline created!")   
                    
                    # Reset presentation time stamp buffer
                    self.pts_buffer = []

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

                    if self.config.gst_colorspace == 'GRAY8':
                        self.frame_shape = (height, width)
                    else:
                        self.frame_shape = (height, width, 3)

                    frame = np.ndarray(shape=self.frame_shape, buffer=map_info.data, dtype=np.uint8)

                    # Unmap the buffer
                    buffer.unmap(map_info)
                    
                    # Check if frame is grayscale and set flag
                    self.convert_to_gray = self.isGrayscale(frame)
                    log.info("Video format: {}, {}P, color: {}".format(self.config.gst_colorspace, height, 
                                                                       not self.convert_to_gray))

                    # Set the video device type
                    self.video_device_type = "gst"

                    conn = getObsDBConn(self.config)
                    addObsParam(conn, "media_backend", self.video_device_type)
                    conn.close()

                    return True

                except Exception as e:
                    log.info("Error initializing GStreamer, switching to alternative. Error: {}".format(e))
                    self.media_backend_override = True
                    self.releaseResources()

                    conn = getObsDBConn(self.config)
                    addObsParam(conn, "media_backend", self.video_device_type)
                    conn.close()

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

        # Stop and release the GStreamer pipeline
        if self.pipeline:

            try:
                                
                # 1. Post EOS and monitor

                # Force sync all children states before EOS
                if not self.pipeline.sync_children_states():
                    log.warning("Initial children sync failed")

                # Send EOS (End of Stream) to initiate graceful shutdown
                self.pipeline.send_event(Gst.Event.new_eos())
                time.sleep(0.1)
                
                # 2. Stop RTSP source

                # Sync again before accessing source - pipeline state might have changed after EOS
                if not self.pipeline.sync_children_states():
                    log.warning("Pre-source children sync failed")

                # Get RTSP source element by name - 'src' is the name given to RTSP source when pipeline was created
                # We need direct access to source element for proper RTSP cleanup (network disconnect)
                src = self.pipeline.get_by_name('src')

                if src:
                    log.debug("Stopping RTSP source...")

                    # Force sync before changing source state
                    if not src.sync_state_with_parent():
                        log.warning("Source sync with parent failed")

                    # Flush any pending data to prevent hanging on network operations
                    src.send_event(Gst.Event.new_flush_start())

                    ret = src.set_state(Gst.State.NULL)
                    if ret == Gst.StateChangeReturn.ASYNC:
                        # If state change is async, wait up to 1 second for it to complete
                        ret, state, pending = src.get_state(Gst.SECOND)
                        log.debug("RTSP source final state: {}, pending: {}".format(state, pending))

                else:
                    log.debug("NO RTSP source found.")

                # 3. Stop pipeline
                log.debug("Stopping pipeline...")
                # Set entire pipeline to NULL - this stops everything
                ret = self.pipeline.set_state(Gst.State.NULL)

                # Wait up to 1 second for pipeline to stop completely
                ret, final_state, pending = self.pipeline.get_state(Gst.SECOND)
                log.debug("Pipeline final state: {}".format(final_state))

                 # Clear pipeline reference to allow proper cleanup   
                self.pipeline = None

                # make sure poller thread exits
                if self._bus_thread and self._bus_thread.is_alive():
                    self._bus_thread.join(timeout=2)
                    self._bus_thread = None
                    
            except Exception as e:
                log.error("Error releasing GStreamer pipeline: {}".format(str(e)))
                # Emergency cleanup
                if self.pipeline:
                    self.pipeline.set_state(Gst.State.NULL)
                    self.pipeline = None
                    
                # Log to database
                conn = getObsDBConn(self.config)
                addObsParam(conn, "media_backend", "gst not successfully released")
                conn.close()

            finally:
                # If failed to stop the source first, use brute force to stop the pipeline
                if self.pipeline:
                    log.error("Attempting to brute force the GStreamer pipeline to NULL")
                    self.pipeline.set_state(Gst.State.NULL)
                    self.pipeline = None

                if abs(self.last_calculated_fps - self.config.fps) > 0.0005 and self.last_calculated_fps_n > 25*60*60:
                    log.info('Config file fps appears to be inaccurate. Consider updating the config file!')

                log.info("Last calculated FPS: {:.6f} at frame {}, config FPS: {}, resets: {}, startup status: {}"
                         .format(self.last_calculated_fps, self.last_calculated_fps_n, self.config.fps, self.reset_count, self.startup_flag))

                log.info('GStreamer Video device released!')

                conn = getObsDBConn(self.config)
                addObsParam(conn, "media_backend", self.video_device_type)
                conn.close()

        # Release the CV2 device (stream or video file)
        if self.device:

            try:

                if self.video_device_type == "cv2":
                    self.device.release()
                    log.info('OpenCV Video device released!')

            except Exception as e:
                log.error('Error releasing OpenCV device: {}'.format(e))
                conn = getObsDBConn(self.config)
                addObsParam(conn, "media_backend", "OpenCV not successfully released")
                conn.close()
            finally:
                self.device = None  # Reset device to None after releasing


        # Release the video device if running Gstreamer
        if self.video_file is not None:

            if GST_IMPORTED and (self.config.media_backend == 'gst'):

                try:
                    self.device.release()
                    log.info('GStreamer Video device released!')

                except Exception as e:
                    log.error('Error releasing GStreamer device: {}'.format(e))

                finally:
                    self.device = None


    def releaseRawArrays(self):
        """Clean up raw frame arrays and saver."""
        if self.raw_frame_saver:
            try:
                self.raw_frame_saver.stop()
                self.raw_frame_saver.join(5)
                if self.raw_frame_saver.is_alive():
                    log.warning("RawFrameSaver still busy. Terminating")
                    self.raw_frame_saver.terminate()
                    self.raw_frame_saver.join()
            finally:
                self.raw_frame_saver = None

        # Clean up array resources
        self.current_raw_frame_shape = None
        self.shared_raw_array = None
        del self.shared_raw_array_base
        del self.shared_raw_array
        del self.shared_raw_array_base2
        del self.shared_raw_array2


    def initRawFrameArrays(self, frame_shape):
        """Initialize raw frame arrays based on current frame shape.
        
        Arguments:
            frame_shape: tuple of frame dimensions
        """
        try:
            # Clean up any existing arrays first
            self.releaseRawArrays()

            # Calculate buffer size based on actual dimensions
            if len(frame_shape) == 3:
                buffer_size = self.num_raw_frames * frame_shape[0] * frame_shape[1] * frame_shape[2]
                array_shape = (self.num_raw_frames, frame_shape[0], frame_shape[1], frame_shape[2])
            else:
                buffer_size = self.num_raw_frames * frame_shape[0] * frame_shape[1]
                array_shape = (self.num_raw_frames, frame_shape[0], frame_shape[1])

            log.debug("Creating shared arrays with shape: {}".format(array_shape))

            # Initialize shared memory arrays
            self.shared_raw_array_base = Array(ctypes.c_uint8, buffer_size)
            self.shared_raw_array = np.ctypeslib.as_array(self.shared_raw_array_base.get_obj())
            self.shared_raw_array = self.shared_raw_array.reshape(array_shape)

            self.shared_raw_array_base2 = Array(ctypes.c_uint8, buffer_size)
            self.shared_raw_array2 = np.ctypeslib.as_array(self.shared_raw_array_base2.get_obj())
            self.shared_raw_array2 = self.shared_raw_array2.reshape(array_shape)

            # Store current array configuration
            self.current_raw_frame_shape = frame_shape
            self.current_mode = self.daytime_mode.value if self.daytime_mode is not None else False
            
            return True

        except Exception as e:
            log.error("Failed to initialize raw frame arrays: {}".format(e))
            log.debug(repr(traceback.format_exception(*sys.exc_info())))
            return False


    def run(self):
        """ Main process function - initializes all process-specific resources and runs capture loop.
        """
        try:
            log.debug("Initializing process-specific resources...")

            # GStreamer debug setup
            if GST_IMPORTED:
                try:
                    # Activate debug system
                    Gst.debug_set_active(True)

                    # Set debug level from environment or default given value
                    # The Gst debug level is set in Logger.py
                    debug_env = os.environ.get("GST_DEBUG", "2")
                    Gst.debug_set_default_threshold(int(debug_env))

                    # Comment out if higher than logging level 3 is needed
                    Gst.debug_add_log_function(gstDebugLogger, None)

                    log.info("GStreamer logging initialized at level: {}".format(debug_env))

                except Exception as e:
                    log.error("Failed to initialize GStreamer logging: {}".format(e))

            # Initialize process-specific variables
            self.media_backend_override = False
            self.video_device_type = "cv2"
            self.time_for_drop = 1.5*(1.0/self.config.fps)
            self.device = None
            self.pipeline = None
            self.start_timestamp = 0
            self.frame_shape = None
            self.convert_to_gray = False

            # Initialize smoothing variables
            self.startup_flag = True
            self.last_calculated_fps = 0
            self.last_calculated_fps_n = 0
            self.expected_m = 1e9/self.config.fps
            self.reset_count = -1
            self.n = 0
            self.sum_x = 0
            self.sum_y = 0
            self.sum_xx = 0
            self.sum_xy = 0
            self.startup_frames = 25*60*10  # 10 minutes
            self.b = 0
            self.b_error_debt = 0
            self.m_jump_error = 0
            self.last_m_err = float('inf')
            self.last_m_err_n = 0
            self.current_raw_frame_shape = None
            self.current_mode = None

            # Initialize raw frame handling if enabled
            if self.config.save_frames:
                self.raw_frame_count = 0
                
                # Convert shared timestamp arrays to numpy arrays
                self.sharedTimestamps = np.ctypeslib.as_array(self.shared_timestamps_base.get_obj())
                self.sharedTimestamps2 = np.ctypeslib.as_array(self.shared_timestamps_base2.get_obj())

                # Raw frame arrays will be initialized after we know the frame shape
                self.shared_raw_array_base = None
                self.shared_raw_array = None
                self.shared_raw_array_base2 = None
                self.shared_raw_array2 = None
                self.raw_frame_saver = None

            # Initialize timestamp array for ft file buffer
            if self.config.save_frame_times:
                self.timestamp_buffer = []
                # For testing ft files
                # self.ft_test_time = time.time()

            # Initialize segment saving time for raw video saving
            if self.config.raw_video_save:
                self.last_segment_savetime = time.time()

            log.debug("Process-specific initialization complete")

            # Main capture loop
            while not self.exit.is_set() and not self.initVideoDevice():
                log.info('Waiting for the video device to be connected...')
                time.sleep(5)

            if self.device is None:
                log.info('The video source could not be opened!')
                self.exit.set()
                return False

            # Continue with main capture loop
            self.captureFrames()

        except Exception as e:
            log.error("Error in capture process: {}".format(e))
            log.debug(repr(traceback.format_exception(*sys.exc_info())))
            self.exit.set()
        finally:
            self.releaseResources()



    def captureFrames(self):
        """ Main frame capture loop - moved from run() for clarity """

        # Keep track of the total number of frames
        total_frames = 0

        # Timestamp of the very first good frame - becomes the run's origin
        run_start_ts = None

        # For video devices only (not files), throw away the first 10 frames
        if (self.video_file is None) and (self.video_device_type == "cv2"):

            first_skipped_frames = 10
            for i in range(first_skipped_frames):
                _, _, ts = self.read()
                if run_start_ts is None:
                    run_start_ts = ts  

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

        # Setup additional timing variables for memory share with RawFrameSaver
        if self.config.save_frames:
            raw_buffer_one = True
            first_raw_frame_timestamp = False


        # Run until stopped from the outside
        while not self.exit.is_set():

            # Wait until the compression is done (only when a video file is used)
            if self.video_file is not None:
                
                wait_for_compression = False

                if buffer_one:
                    if self.start_time1.value == -1:
                        wait_for_compression = True
                else:
                    if self.start_time2.value == -1:
                        wait_for_compression = True

                if wait_for_compression:
                    log.debug("Waiting for the {:d}. compression thread to finish...".format(int(not buffer_one) + 1))
                    time.sleep(0.1)
                    continue

            
            if buffer_one:
                self.start_time1.value = 0
            else:
                self.start_time2.value = 0
            

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
                    ret, _, _ = self.read(check_color=True)
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
            first_frame_timestamp = None

            # running totals for mean calculations
            sum_frame_interval_norm = 0.0
            sum_frame_age_seconds   = 0.0

            # Capture a block of 256 frames
            block_frames = 256

            # Check if camera needs switching
            if self.config.continuous_capture and self.config.switch_camera_modes:

                # Check that the camera mode switch is triggered
                if self.camera_mode_switch_trigger.value:
                    
                    # If the camera mode switch trigger is set, switch the camera mode
                    switchCameraMode(self.config, self.daytime_mode, self.camera_mode_switch_trigger)


            log.info('Grabbing a new block of {:d} frames...'.format(block_frames))
            for i in range(block_frames):

                # Set flag to save a raw frame
                save_this_frame = (
                self.config.save_frames and
                self.video_file is None and
                total_frames % self.config.frame_save_interval_count == 0
                )

                # Read the frame (keep track how long it took to grab it), and check for color if saving raw frame
                t1_frame = time.time()
                ret, frame, frame_timestamp = self.read(check_color=save_this_frame)
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

                # Append current timestamp to ft file buffer
                if self.config.save_frame_times:
                    self.timestamp_buffer.append((total_frames, frame_timestamp))

                # If save_frames is set and a video device is used, save a frame every nth frames
                if save_this_frame:

                    # Check if frame shape (color or grayscale) or capture mode changed (day or night)
                    if (frame.shape != self.current_raw_frame_shape) or \
                        (self.current_mode != (self.daytime_mode.value if self.daytime_mode is not None else False)) or \
                        (self.shared_raw_array is None):

                        log.info("Frame shape/mode changed, reinitializing arrays...")

                        # First signal the raw frame saver to finish saving current block
                        if raw_buffer_one:
                            self.start_raw_time1.value = first_raw_frame_timestamp
                        else:
                            self.start_raw_time2.value = first_raw_frame_timestamp

                        if not self.initRawFrameArrays(frame.shape):
                            log.error("Failed to reinitialize arrays after mode change")

                        else:
                            # Initialize new frame saver
                            self.raw_frame_saver = RawFrameSaver(
                                self.saved_frames_dir,
                                self.shared_raw_array, self.start_raw_time1,
                                self.shared_raw_array2, self.start_raw_time2,
                                self.sharedTimestamps, self.sharedTimestamps2,
                                self.daytime_mode.value,
                                self.config
                            )
                            self.raw_frame_saver.start()
                            self.raw_frame_count = 0
                            log.info("Successfully reinitialized raw frame handling")


                    # reset start time values everytime the buffers are switched
                    if self.raw_frame_count == 0:

                        if raw_buffer_one:
                            self.start_raw_time1.value = 0
                        else:
                            self.start_raw_time2.value = 0

                        # Always set first raw frame timestamp in the beginning of the block
                        first_raw_frame_timestamp = frame_timestamp 


                    # Write raw frame and timestamp to one of the two corresponding buffers
                    # Use appropriate indexing based on frame dimensions
                    if len(frame.shape) == 3:
                        # Color frame - use 4D indexing
                        if raw_buffer_one:
                            self.shared_raw_array[self.raw_frame_count, :, :, :] = frame
                            self.sharedTimestamps[self.raw_frame_count] = frame_timestamp
                        else:
                            self.shared_raw_array2[self.raw_frame_count, :, :, :] = frame
                            self.sharedTimestamps2[self.raw_frame_count] = frame_timestamp
                    else:
                        # Grayscale frame - use 3D indexing
                        if raw_buffer_one:
                            self.shared_raw_array[self.raw_frame_count, :, :] = frame
                            self.sharedTimestamps[self.raw_frame_count] = frame_timestamp
                        else:
                            self.shared_raw_array2[self.raw_frame_count, :, :] = frame
                            self.sharedTimestamps2[self.raw_frame_count] = frame_timestamp

                    self.raw_frame_count += 1

                    # switch buffers arrays every (self.num_raw_frames) frames
                    if self.raw_frame_count == self.num_raw_frames:

                        if raw_buffer_one:
                            self.start_raw_time1.value = first_raw_frame_timestamp
                        else:
                            self.start_raw_time2.value = first_raw_frame_timestamp
                        
                        self.raw_frame_count = 0
                        raw_buffer_one = not raw_buffer_one


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
                if (self.config.media_backend != 'gst') and not self.media_backend_override and last_frame_timestamp is not False:
                    # Calculate the normalized frame interval between the current and last frame read, normalized by frames per second (fps)
                    frame_interval_normalized = (frame_timestamp - last_frame_timestamp)*self.config.fps
                    # Update max_frame_interval_normalized for this cycle
                    max_frame_interval_normalized = max(max_frame_interval_normalized, frame_interval_normalized)
                    sum_frame_interval_norm += frame_interval_normalized

                # If GStreamer:
                else:
                    # Calculate the time difference between the current time and the frame's timestamp
                    frame_age_seconds = time.time() - frame_timestamp
                    # Update max_frame_age_seconds for this cycles
                    max_frame_age_seconds = max(max_frame_age_seconds, frame_age_seconds)
                    sum_frame_age_seconds += frame_age_seconds

                # On the last loop, report late or dropped frames
                if i == block_frames - 1:

                    # For cv2, show elapsed time since frame read to assess loop performance
                    if self.config.media_backend != 'gst' and not self.media_backend_override:
                        mean_interval_norm = sum_frame_interval_norm/block_frames

                        # running late-frame total since the start of capture
                        if run_start_ts is not None:
                            elapsed_run = last_frame_timestamp - run_start_ts
                            expected_run = int(round(elapsed_run*self.config.fps))
                            run_late_frames = max(0, expected_run - total_frames)
                        else:
                            run_start_ts = last_frame_timestamp
                            run_late_frames = 0

                        log.info("Block interval: mean %.3f, max %.3f (normalized). Dropped frames: %d",
                                 mean_interval_norm, max_frame_interval_normalized, run_late_frames)
                    
                    # For GStreamer, show elapsed time since frame capture to assess sink fill level
                    else:
                        log.info("Block's max frame age: {:.3f} seconds. Run's dropped frames: {}"
                                 .format(max_frame_age_seconds, self.dropped_frames.value))

                last_frame_timestamp = frame_timestamp
                

                ### Convert the frame to grayscale ###  (Not to be done in case of daytime mode)
                if not self.daytime_mode.value:

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

                # Save leftover raw frames from last used buffer
                if self.config.save_frames:
                    if raw_buffer_one:
                        self.start_raw_time1.value = first_raw_frame_timestamp
                    else:
                        self.start_raw_time2.value = first_raw_frame_timestamp

                break


            if (not wait_for_reconnect
                and not self.daytime_mode.value
                and first_frame_timestamp is not None):

                # Set the starting value of the frame block, which indicates to the compression that the
                # block is ready for processing
                if buffer_one:
                    self.start_time1.value = first_frame_timestamp

                else:
                    self.start_time2.value = first_frame_timestamp

                log.info('New block of raw frames available for compression with starting time: {:s}'
                         .format(str(first_frame_timestamp)))

            
            # Switch the frame block buffer flags
            buffer_one = not buffer_one
            if self.config.report_dropped_frames:
                log.info('Estimated FPS: {:.3f}'.format(block_frames/(time.time() - t_block)))
        

            # Save current timestamp buffer to ft file
            # Construct FTStruct, record timestamps, and reset the timestamp array in memory
            if (self.config.save_frame_times and first_frame_timestamp is not None):
                ft = FTStruct.FTStruct()
                ft.timestamps = copy.copy(self.timestamp_buffer)

                # Clear the timestamp buffer list                
                del self.timestamp_buffer[:]

                base_time = UTCFromTimestamp.utcfromtimestamp(first_frame_timestamp)
                ft_filename = base_time.strftime("FT_{}_%Y%m%d_%H%M%S.bin".format(self.config.stationID))
                ft_subpath = os.path.join(self.config.data_dir, self.config.times_dir, base_time.strftime("%Y/%Y%m%d-%j/%Y%m%d-%j_%H"))

                mkdirP(ft_subpath)
                FTfile.write(ft, ft_subpath, ft_filename)
                log.info("Created FT file {} for block starting at {}".format(os.path.join(ft_subpath, ft_filename), first_frame_timestamp))

                # For Testing: 
                # Print first and last 10 timestamps, array length, average time difference and time difference from last block
                # Enable self.ft_test_time in __init__
                
                # print("\n\n --- FT file data --- \nFirst 10 timestamps: {}\n\nLast 10 timestamps: {}\n\nArray length: {}\n\n".format(
                #       ft.timestamps[:11], 
                #       ft.timestamps[-10:],
                #       len(ft.timestamps),
                # ),
                #       "Average per-frame time difference: {}\n\nLast segment time difference: {}\n\n ---------------- \n\n".format(
                #       sum(ft.timestamps[i+1][1] - ft.timestamps[i][1] for i in range(len(ft.timestamps) - 1)) / (len(ft.timestamps) - 1),
                #       ft.timestamps[0][1] - self.ft_test_time
                # ), end='')
                # self.ft_test_time = ft.timestamps[-1][1]


        log.info('Releasing video device...')
        self.releaseResources()


if __name__ == "__main__":

    import argparse
    import ctypes

    import multiprocessing

    import RMS.ConfigReader as cr
    from RMS.Logger import initLogging

    ###

    arg_parser = argparse.ArgumentParser(description='Test capturing frames from a video source defined in the config file. ')

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one.")
    
    arg_parser.add_argument('--video_file', metavar='VIDEO_FILE', type=str, \
        help="Path to a video file to be used as a video source instead of a camera.")
    

     # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    ###
    
    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('.'))

    # Initialize the logger
    initLogging(config)

    # Get the logger handle
    log = getLogger("logger")

    # Print the kind of media backend
    print("Station code: {}".format(config.stationID))
    print('Media backend: {}'.format(config.media_backend))


    # Init dummy shared memory
    sharedArrayBase = multiprocessing.Array(ctypes.c_uint8, 256*(config.width)*(config.height))
    sharedArray = np.ctypeslib.as_array(sharedArrayBase.get_obj())
    sharedArray = sharedArray.reshape(256, (config.height), (config.width))
    startTime = multiprocessing.Value('d', 0.0)


    # If a video is given, use it as the video source
    if cml_args.video_file:

        print("Using video file: {}".format(cml_args.video_file))

        bc = BufferedCapture(sharedArray, startTime, sharedArray, startTime, config, 
                             video_file=cml_args.video_file)
        
        bc.initVideoDevice()
        

        # Read at least 256 frames from the video file
        for i in range(256):
            ret, frame = bc.device.read()

            print('Frame read: {}'.format(i))
            if not ret:
                print("End of video file!")
                break
                
        # Close the device
        bc.releaseResources()

        
    
    # Capture from a camera
    else:

        # Init the BufferedCapture object
        bc = BufferedCapture(sharedArray, startTime, sharedArray, startTime, config)

        device = bc.createGstreamDevice('BGR', video_file_dir=None, segment_duration_sec=config.raw_video_duration)

        print('GStreamer device created!')

        ### TEST
        print("Pulling a sample...", end=' ')
        sample = device.emit("pull-sample")
        print('Sample pulled!')

        print('Mapping buffer...', end=' ')
        buffer = sample.get_buffer()
        ret, map_info = buffer.map(Gst.MapFlags.READ)
        print('Buffer mapped!')

        print('Getting caps...', end=' ')
        caps = sample.get_caps()
        print('Caps obtained!')

        print('Getting structure...', end=' ')
        structure = caps.get_structure(0)
        print('Structure obtained!')

        print('Extracting width and height...', end=' ')
        width = structure.get_value('width')
        height = structure.get_value('height')
        print('Width and height extracted!')

        print('Creating frame...', end=' ')
        frame_shape = (height, width, 3)
        frame = np.ndarray(shape=frame_shape, buffer=map_info.data, dtype=np.uint8)
        print('Frame created!')

        print('Unmapping buffer...', end=' ')
        buffer.unmap(map_info)
        print('Buffer unmapped!')
        ###

        # Close the device
        bc.releaseResources()
