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
import traceback
import time
import multiprocessing
from math import floor

import cv2
import numpy as np

from RMS.Logger import getLogger, getLoggingQueue, initChildProcess
from RMS.Misc import mkdirP, setParentDeathSignal

# Get the logger from the main module
log = getLogger("rmslogger")


class RawFrameSaver(multiprocessing.Process):
    """Save list of numpy arrays (raw video frames).
    """

    running = False
    
    def __init__(self, saved_frames_dir, array1, start_time1, array2, start_time2, tsArray1, tsArray2, daytime_mode, config, raw_array_shape):
        """

        Arguments:
            saved_frames_dir: directory to save raw frames to
            array1: multiprocessing.Array base for the first raw-frame buffer (shared memory)
            start_time1: float in shared memory that holds time of first raw frame in array1
            array2: multiprocessing.Array base for the second raw-frame buffer
            start_time1: float in shared memory that holds time of first raw frame in array2
            tsArray1: multiprocessing.Array base for the first timestamp buffer
            tsArray2: multiprocessing.Array base for the second timestamp buffer
            config: configuration class
            daytime_mode: [bool] True if the camera is in daytime mode, False if in nightime mode
            raw_array_shape: [tuple] Shape of the raw-frame buffer, used to rebuild the numpy view.

        """

        super(RawFrameSaver, self).__init__()

        self.saved_frames_dir = saved_frames_dir
        # array1/array2 and the timestamp arrays are multiprocessing.Array BASE objects (picklable
        # across forkserver/spawn). The numpy views over them are rebuilt in run() so they stay
        # backed by the same shared memory the capture process writes into.
        self.array1_base = array1
        self.array2_base = array2
        self.timeStamps1_base = tsArray1
        self.timeStamps2_base = tsArray2
        self.raw_array_shape = raw_array_shape
        self.array1 = None
        self.array2 = None
        self.timeStamps1 = None
        self.timeStamps2 = None
        self.start_time1 = start_time1
        self.start_time2 = start_time2
        self.daytime_mode = daytime_mode
        self.config = config

        self.total_saved_frames = 0
        self.day_of_year = time.strftime("%j", time.gmtime())

        self.exit = multiprocessing.Event()
        self.run_exited = multiprocessing.Event()

        # Grab the logging queue on the parent side so the child can re-attach logging
        # under the 'forkserver'/'spawn' start methods (handlers are not inherited there)
        self.logging_queue = getLoggingQueue()

        # PID of the logical parent (BufferedCapture). __init__ runs in the parent, so this
        # is captured correctly under fork, spawn AND forkserver - unlike os.getppid(),
        # which under forkserver returns the fork-server, not BufferedCapture. Used in run()
        # to self-terminate if the parent dies without setting our exit Event (watchdog
        # force-kill), so an orphan can never linger and leak its buffer. Start-method- and
        # platform-agnostic, so it works across Python 3.6-3.14.
        self.parent_pid = os.getpid()


    def saveFramesToDisk(self, frametimes, daytime_mode=False):
        """Saves a block of raw image frames to disk with timestamp-based filenames.

        This method calculates each filename using station ID, the UTC date
        and time from the timestamp, and the milliseconds part of the timestamp
        to ensure uniqueness, and then saves the frames in the specified format
        to the saved_frames_dir directory.

        Each file is stored in a path based on its time:
        saved_frames_dir/YYYY/YYYYMMDD-DoY/YYYYMMDD-DoY_HH/stationID_YYYYMMDD_HHMMSS_MMM.ttt

        Where 'DoY' is day of year and 'ttt' is either the jpg or png file type.

        Arguments
        ---------
            frametimes : [List] list of (frame, timestamp) pairs of corresponding frames and timestamps
        """

        # Log block-level summary with day/night mode
        frame_count = sum(1 for _, ts in frametimes if ts != 0)
        if frame_count > 0:
            mode_str = "day" if daytime_mode else "night"
            log.info("Saving block of %d raw frames to disk (%s mode)", frame_count, mode_str)

        for (frame, timestamp) in frametimes:

            # If timestamp is 0, then we've reached the end and this is the last block 
            if timestamp == 0:
                break

            # Handle when frame has only two channels (yuyv/uyvy)
            # OpenCV only supports 1, 3, or 4 color channels
            if (len(frame.shape) == 3) and (frame.shape[2] == 2):
                # If UYVY image given, luma (Y) channel is channel 1
                if self.config.uyvy_pixelformat:
                    frame = frame[:, :, 1]
                
                # Otherwise, take the first available channel
                else:
                    frame = frame[:, :, 0]

            # In case the timestamp day changes mid-block
            if self.day_of_year != time.strftime("%j", time.gmtime(timestamp)):
                
                # Adjust values for the day change
                self.total_saved_frames = 0
                self.day_of_year = time.strftime("%j", time.gmtime(timestamp))


            # Generate names for the file and path
            date_string = time.strftime("%Y%m%d_%H%M%S", time.gmtime(timestamp))
            timed_dir_string = time.strftime("%Y/%Y%m%d-%j/%Y%m%d-%j_%H", time.gmtime(timestamp))

            # Calculate milliseconds
            millis = int((timestamp - floor(timestamp))*1000)

            # Suffix for indicating if the camera is in daytime or nighttime mode
            mode_suffix = ""
            if daytime_mode:
                mode_suffix = "_d"
            else:
                mode_suffix = "_n"

            # Create the filename
            if self.config.frame_file_type == 'png':
                file_extension = '.png'
            else:
                file_extension = '.jpg'

            filename = "{0}_{1}_{2:03d}{3}{4}".format(
                str(self.config.stationID).zfill(3),
                date_string,
                millis,
                mode_suffix,
                file_extension
            )

            # Full path for saving the file
            frame_dir_path = os.path.join(self.saved_frames_dir, timed_dir_string)
            mkdirP(frame_dir_path)
            frame_path = os.path.join(frame_dir_path, filename)

            # Write the image file
            try:
                if file_extension == '.png':
                    cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_PNG_COMPRESSION), self.config.png_compression])

                else:
                    cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpgs_quality])

                log.debug("Frame saved: {0}".format(filename))

            except Exception as e:
                log.error("Could not save frame to disk: {0}".format(e))

            self.total_saved_frames += 1


    def ensureViews(self):
        """ Build numpy views over the shared multiprocessing.Array bases if not already built.

        Both the capture process (which calls stop() to flush tail-end frames) and this saver
        process (run()) need a view backed by the same shared memory. Under forkserver/spawn a
        numpy view cannot be inherited or pickled, so each process builds its own view from the
        shared base here. Idempotent.
        """
        if self.array1 is None:
            self.array1 = np.ctypeslib.as_array(self.array1_base.get_obj()).reshape(self.raw_array_shape)
            self.array2 = np.ctypeslib.as_array(self.array2_base.get_obj()).reshape(self.raw_array_shape)
            self.timeStamps1 = np.ctypeslib.as_array(self.timeStamps1_base.get_obj())
            self.timeStamps2 = np.ctypeslib.as_array(self.timeStamps2_base.get_obj())


    def stop(self):
        """ Stop saving frames.
        """

        self.exit.set()
        log.debug('Raw frame saver exit flag set')

        # Build views over the shared buffers in this (capture) process so the
        # tail-end flush reads the same shared memory the saver process wrote
        # into (under forkserver/spawn views are not inherited - see
        # ensureViews). The bases may already be gone by the time we get here:
        # stop() frees them itself (see the `del`s below), and
        # BufferedCapture.releaseRawArrays() calls stop() both on shutdown AND
        # on every day/night mode switch, where the saver is torn down before
        # the arrays are re-created for the new frame shape. Zipping None then
        # raised "TypeError: 'NoneType' object is not iterable" out of a
        # teardown path, logged as a traceback on a perfectly healthy station
        # (twice a day, per camera). There is nothing to flush in that case.
        try:
            self.ensureViews()
        except Exception:
            pass
        array1 = getattr(self, 'array1', None)
        array2 = getattr(self, 'array2', None)
        timestamps1 = getattr(self, 'timeStamps1', None)
        timestamps2 = getattr(self, 'timeStamps2', None)

        if array1 is None and array2 is None:
            log.debug('Raw frame saver buffers already released - '
                      'nothing to flush')

        leftovers = []
        if (array1 is not None) and (timestamps1 is not None):
            for frame, ts in zip(array1, timestamps1):
                if ts: leftovers.append((frame.copy(), float(ts)))
        if (array2 is not None) and (timestamps2 is not None):
            for frame, ts in zip(array2, timestamps2):
                if ts: leftovers.append((frame.copy(), float(ts)))
        if leftovers:
            log.info("Flushing %d tail-end raw frames before shutdown", len(leftovers))
            self.saveFramesToDisk(leftovers, self.daytime_mode)

            # mark buffers consumed so run() won’t resave them
            if timestamps1 is not None:
                timestamps1.fill(0)
            if timestamps2 is not None:
                timestamps2.fill(0)
            self.start_time1.value = 0
            self.start_time2.value = 0

        # Free shared memory after the raw frame saver is done
        try:
            log.debug('Freeing frame buffers in raw frame saver...')
            self.array1 = None
            self.array2 = None
            self.timeStamps1 = None
            self.timeStamps2 = None

        except Exception as e:
            log.debug('Freeing raw frame buffers failed with error:' + repr(e))
            log.debug(repr(traceback.format_exception(*sys.exc_info())))
    

    def start(self):
        """ Start raw frame saving.
        """
        
        super(RawFrameSaver, self).start()
    

    def run(self):
        """ Retrieve raw frames from shared array and save them.
        """

        # Die if our parent BufferedCapture dies (e.g. watchdog force-kill). Without this
        # the orphaned saver loops forever on a shared exit Event that is never set,
        # leaking its inherited ~450 MB buffer; hundreds accumulate and OOM the box.
        # Set as early as possible. Note that under forkserver this fires on the wrong
        # parent's death, which is why the os.kill() liveness probe in the wait loop below
        # complements it.
        setParentDeathSignal()

        # Re-establish logging and signal handling in the child (no-op under 'fork')
        initChildProcess(self.logging_queue, self.config)

        # Rebuild numpy views over the shared raw-frame and timestamp buffers in this process.
        # Under forkserver/spawn the views cannot be inherited, so build them here from the
        # shared multiprocessing.Array base objects (same memory the capture process writes).
        self.ensureViews()

        try:
            # Repeat until the raw frame saver is killed from the outside
            while not self.exit.is_set():

                # Block until the raw frames are available
                while (self.start_time1.value == 0) and (self.start_time2.value == 0):

                    # Exit function if process was stopped from the outside
                    if self.exit.is_set():

                        log.debug('Raw frame saver run exit')
                        self.run_exited.set()

                        return None

                    # Forkserver-safe orphan guard: if the logical parent (BufferedCapture)
                    # is gone, our exit Event will never be set, so self-terminate instead
                    # of spinning here forever holding the frame buffer. Complements
                    # PR_SET_PDEATHSIG (which doesn't fire correctly under forkserver).
                    # POSIX only: on Windows signal 0 is CTRL_C_EVENT, so os.kill(pid, 0)
                    # is not a liveness probe there. ProcessLookupError (ESRCH) rather than
                    # OSError, so an EPERM on a live parent isn't mistaken for death.
                    if os.name == 'posix':
                        try:
                            os.kill(self.parent_pid, 0)
                        except ProcessLookupError:
                            log.warning('RawFrameSaver: parent process %d gone, exiting orphan',
                                        self.parent_pid)
                            self.run_exited.set()
                            return None

                    time.sleep(0.1)

                raw_buffer_one = True

                if self.start_time1.value > 0:

                    # Retrieve time of first frame
                    startTime = float(self.start_time1.value)

                    # Copy raw (frames, timestamps)
                    # Clear out the timestamp array so it can be used by 
                    # saveFramesToDisk to halt
                    frametimes = list(zip(self.array1, self.timeStamps1))
                    self.timeStamps1.fill(0)
                    raw_buffer_one = True

                elif self.start_time2.value > 0:

                    # Retrieve time of first frame
                    startTime = float(self.start_time2.value)

                    # Copy raw (frames, timestamps)
                    # Clear out the timestamp array so it can be used by 
                    # saveFramesToDisk to halt
                    frametimes = list(zip(self.array2, self.timeStamps2))
                    self.timeStamps2.fill(0)
                    raw_buffer_one = False

                else:

                    # Wait until data is available
                    log.debug("Raw frame saver waiting for frames...")
                    time.sleep(0.1)
                    continue
                
                log.debug("Saving raw frame block with start time at: {:s}".format(str(startTime)))

                t = time.time()

                # Run the frame block save
                self.saveFramesToDisk(frametimes, self.daytime_mode)

                # Once the frame saving is done, tell the capture thread to keep filling the buffer
                if raw_buffer_one:
                    self.start_time1.value = 0
                else:
                    self.start_time2.value = 0

                log.debug("Raw frame block saving time: {:.3f} s".format(time.time() - t))

            log.debug('Raw frame saver run exit')
            time.sleep(1.0)
            self.run_exited.set()

        except KeyboardInterrupt:
            log.info("RawFrameSaver process received interrupt signal. Shutting down gracefully...")
            self.exit.set()
            self.run_exited.set()
        except Exception as e:
            log.error("Error in RawFrameSaver process: {}".format(e))
            log.debug(repr(traceback.format_exception(*sys.exc_info())))
            self.exit.set()
            self.run_exited.set()
