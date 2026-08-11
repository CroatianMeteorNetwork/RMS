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


import os
import sys
import traceback
import time
import datetime
import multiprocessing
import ctypes
import signal
from math import floor
import numpy as np
import cv2


from RMS.VideoExtraction import Extractor
from RMS.Formats import FFfile, FFStruct
from RMS.Formats import FieldIntensities
from RMS.Logger import getLogger, getLoggingQueue, initChildProcess
from RMS.Misc import UTCFromTimestamp, frameBufferShape, AtomicFlag, stableDoubleRead
from RMS.Routines.Image import saveImage

# Import Cython functions
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from RMS.CompressionCy import compressFrames


# Get the logger from the main module
log = getLogger("rmslogger")


class Compressor(multiprocessing.Process):
    """Compress list of numpy arrays (video frames).

        Output is in Four-frame Temporal Pixel (FTP) format. See the Jenniskens et al., 2011 paper about the
        CAMS project for more info.

    """

    running = False
    
    def __init__(self, data_dir, array1, start_time1, array2, start_time2, config, detector=None):
        """

        Arguments:
            array1: multiprocessing.Array base for the first frame buffer (the numpy
                view is rebuilt per-process via frameBufferShape - see RMS.Misc)
            start_time1: float in shared memory that holds time of first frame in array1
            array2: multiprocessing.Array base for the second frame buffer
            start_time2: float in shared memory that holds time of first frame in array2
            config: configuration class

        Keyword arguments:
            detector: [Detector object] Handle to Detector object used for running star extraction and
                meteor detection.

        """
        
        super(Compressor, self).__init__()
        
        self.data_dir = data_dir
        # array1/array2 are multiprocessing.Array BASE objects (picklable across forkserver/spawn).
        # The numpy views over them are rebuilt in run() so they stay backed by the same shared
        # memory the capture process writes into; a numpy view passed here would pickle by value.
        self.array1_base = array1
        self.array2_base = array2
        self.array1 = None
        self.array2 = None
        self.start_time1 = start_time1
        self.start_time2 = start_time2
        self.config = config

        self.detector = detector

        # Lock-free flags: these are set/polled across processes and must never be able to
        # deadlock, even if a process sharing them is killed (see AtomicFlag)
        self.exit = AtomicFlag()

        # Lock-free flag: an mp.Event deadlocks if a process sharing it is OOM-killed while
        # holding its internal semaphore (see AtomicFlag). Only .set()/.is_set() are used here.
        self.run_exited = AtomicFlag()

        # Grab the logging queue on the parent side so the child can re-attach logging
        # under the 'forkserver'/'spawn' start methods (handlers are not inherited there)
        self.logging_queue = getLoggingQueue()


    def compress(self, frames):
        """ Compress frames to the FTP-compatible array and extract sums of intensities per every field.

        Arguments:
            frames: [3D ndarray] grayscale frames stored as 3d numpy array

        Return:
            (ftp_array, ave16, fieldsum):
                - ftp_array: [3D ndarray] in format: (N, y, x) where N is a member of [0, 1, 2, 3]
                - ave16: [2D ndarray] average pixel image in 8.8 fixed point (uint16, 1/256 ADU units)
                - fieldsum: [ndarray] sums of intensities per every field

        """

        # Run cythonized compression. The camera gamma is passed so the average is computed in
        # the linear domain (and re-encoded), removing the Jensen bias of averaging
        # gamma-encoded samples
        ftp_array, ave16, fieldsum = compressFrames(frames, self.config.deinterlace_order,
            self.config.gamma)

        return ftp_array, ave16, fieldsum
    


    def saveFF(self, arr, startTime, N, ave16=None):
        """ Write metadata and data array to FF file and return filenames for FF and FS files

        Arguments:
            arr: [3D ndarray] 3D numpy array in format: (N, y, x) where N is [0, 4)
            startTime: [float] seconds and fractions of a second from epoch to first frame
            N: [int] frame counter (ie. 0000512)

        Keyword arguments:
            ave16: [2D ndarray] average pixel image in 8.8 fixed point (uint16). Written to the FF
                file as a full-precision average plane if ff_avepixel16 is enabled in the config.
        """
        
        # Generate the name for the file
        date_string = time.strftime("%Y%m%d_%H%M%S", time.gmtime(startTime))

        # Calculate microseconds and milliseconds
        micros = int((startTime - floor(startTime))*1000000)
        millis = int((startTime - floor(startTime))*1000)
        

        filename_millis = str(self.config.stationID).zfill(3) +  "_" + date_string + "_" + str(millis).zfill(3) \
            + "_" + str(N).zfill(7)
        
        filename_micros = str(self.config.stationID).zfill(3) +  "_" + date_string + "_" + str(micros).zfill(6) \
            + "_" + str(N).zfill(7)

        ff = FFStruct.FFStruct()
        ff.array = arr

        # Attach the full-precision average so it gets written as a 16-bit plane. Only the FITS
        # format can carry it; the legacy bin writer ignores it
        if (ave16 is not None) and self.config.ff_avepixel16:
            ff.avepixel16 = ave16

            # Record the gamma used for the linear-domain averaging (provenance)
            ff.avegamma = self.config.gamma

        ff.nrows = arr.shape[1]
        ff.ncols = arr.shape[2]
        ff.nbits = self.config.bit_depth
        ff.nframes = 256
        ff.first = N + 256
        ff.camno = self.config.stationID
        ff.fps = self.config.fps

        if sys.version_info[0] == 2:
            # Python 2 code
            dt = UTCFromTimestamp.utcfromtimestamp(startTime)
            ff.starttime = dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

        else:
            # Python 3 code
            dt = UTCFromTimestamp.utcfromtimestamp(startTime)
            ff.starttime = dt.isoformat(timespec='microseconds')
        
        # Write the FF file
        FFfile.write(ff, self.data_dir, filename_millis, fmt=self.config.ff_format)
        
        return filename_millis, filename_micros


    def saveLiveJPG(self, array, startTime):
        """ Save a live.jpg file to the data directory with the latest compressed image. """


        # Name of the file
        live_name = 'live.jpg'

        # Generate the name for the file
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(startTime))

        maxpixel, _, _, _ = np.split(array, 4, axis=0)
        maxpixel = np.array(maxpixel[0])

        # Draw text to image
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = self.config.stationID + " " + timestamp + " UTC"
        cv2.putText(maxpixel, text, (10, maxpixel.shape[0] - 6), font, 0.4, (255, 255, 255), 1, \
            cv2.LINE_AA)

        # Save the labelled image to disk
        try:
            # Save the file to disk
            saveImage(os.path.join(self.config.data_dir, live_name), maxpixel)
        except:
            log.error("Could not save {:s} to disk!".format(live_name))
    


    def stop(self):
        """ Stop compression.
        """
        

        self.exit.set()
        log.debug('Compression exit flag set')

            
        log.debug('Joining compression...')


        t_beg = time.time()

        # Wait until everything is done
        while not self.run_exited.is_set():
            
            time.sleep(0.01)

            # Do not wait more than a minute, just terminate the compression thread then
            if (time.time() - t_beg) > 60:
                log.debug('Waited more than 60 seconds for compression to end, killing it...')
                break

        log.debug('Compression joined!')

        # If process didn't exit cleanly, send graceful interrupt
        if self.is_alive():
            log.info("Compression process still alive, sending interrupt signal...")
            try:
                if self.pid:
                    os.kill(self.pid, signal.SIGINT)
                
                # Wait for graceful shutdown
                self.join(5)
                
                if self.is_alive():
                    log.warning("Compression process still alive after interrupt, forcing termination")
                    self.terminate()
                else:
                    log.info("Compression process exited gracefully after interrupt")

            except ProcessLookupError:
                log.info("Compression process already terminated")
            except Exception as e:
                log.error("Error during graceful compression shutdown: {}".format(e))
                log.info("Falling back to terminate()")
                self.terminate()

            # A bare join() would hang forever on a process that ignores SIGTERM -
            # bound the wait and escalate to SIGKILL (review finding)
            self.join(5)

            if self.is_alive():
                log.warning("Compression process survived terminate, sending SIGKILL...")
                try:
                    os.kill(self.pid, signal.SIGKILL)
                except (OSError, AttributeError):
                    pass

            # Always join to reap zombie (returns instantly if already dead)
            self.join()

        else:
            # Process is not alive but may not have been joined yet - reap it
            log.debug("Compression process not alive, joining to reap resources")
            self.join(timeout=5)

        # Return the detector and live viewer objects because they were updated in this namespace
        return self.detector
    


    def start(self):
        """ Start compression.
        """
        
        super(Compressor, self).start()
    


    def run(self):
        """ Retrieve frames from list, convert, compress and save them.
        """

        # Re-establish logging and signal handling in the child (no-op under 'fork')
        initChildProcess(self.logging_queue, self.config)

        # Rebuild numpy views over the shared frame buffers in this process. Under forkserver/spawn
        # the views cannot be inherited, so build them here from the shared multiprocessing.Array
        # base objects - this is the same shared memory the capture process writes frames into.
        frame_buffer_shape = frameBufferShape(self.config)
        self.array1 = np.ctypeslib.as_array(self.array1_base.get_obj()).reshape(frame_buffer_shape)
        self.array2 = np.ctypeslib.as_array(self.array2_base.get_obj()).reshape(frame_buffer_shape)

        n = 0
        exit_wait_start = None
        
        # Repeat until the compressor is killed from the outside
        while True:
            # graceful-exit check
            if self.exit.is_set():
                if self.start_time1.value == 0 and self.start_time2.value == 0:
                    break
                # Start timeout counter on first exit request
                if exit_wait_start is None:
                    exit_wait_start = time.time()
                    log.info("Waiting for compression to finish before exit...")
                # Force exit after 30 seconds
                elif time.time() - exit_wait_start > 30:
                    log.warning("Forced exit after 30s timeout - frames may be lost")
                    break
                # Sleep briefly to avoid busy-waiting
                time.sleep(0.1)
                continue

            # Block until frames are available
            while (self.start_time1.value == 0) and (self.start_time2.value == 0):

                # Exit function if process was stopped from the outside
                if self.exit.is_set():

                    log.debug('Compression run exit')

                    self.run_exited.set()
                    os._exit(0)

                time.sleep(0.1)

                

            t = time.time()

            
            buffer_one = True

            # Stable reads: the 0 -> t transition of these lock-free doubles can
            # tear on 32-bit ARM and a torn value passes the > 0 gate with a
            # timestamp wrong by up to ~1024 s (review finding)
            start_time1_val = stableDoubleRead(self.start_time1)
            start_time2_val = stableDoubleRead(self.start_time2)

            if start_time1_val > 0:

                # Retrieve time of first frame
                startTime = float(start_time1_val)

                # Copy frames
                frames = self.array1

                # Tell the capture thread to wait until the compression is completed by setting this to -1
                self.start_time1.value = -1
                buffer_one = True

            elif start_time2_val > 0:

                # Retrieve time of first frame
                startTime = float(start_time2_val)

                # Copy frames
                frames = self.array2

                # Tell the capture thread to wait until the compression is completed
                self.start_time2.value = -1
                buffer_one = False

            else:

                # Wait until data is available
                log.debug("Compression waiting for frames...")
                time.sleep(0.1)
                continue

            
            log.debug("Compressing frame block with start time at: {:s}".format(str(startTime)))

            #log.debug("memory copy: " + str(time.time() - t) + "s")
            t = time.time()
            
            
            # Run the compression
            compressed, ave16, field_intensities = self.compress(frames)


            # Snapshot the raw block for the extractor BEFORE releasing the capture
            # handshake: once start_time returns to 0 the buffer is eligible for refill,
            # and the extractor reads it seconds later - the old code raced the refill
            # under 'fork' and pickled the whole ~236 MB block by value under
            # 'forkserver'/'spawn' (stalling compression and spiking RSS). A shared
            # mp.Array snapshot costs one memcpy while the handshake is held and crosses
            # the process boundary without pickling.
            frames_snapshot_base = None
            if self.config.enable_fireball_detection:
                frames_snapshot_base = multiprocessing.Array(
                    ctypes.c_uint8, int(frames.size), lock=False)
                snapshot_view = np.frombuffer(frames_snapshot_base,
                    dtype=np.uint8).reshape(frames.shape)
                np.copyto(snapshot_view, frames)

            # Once the compression is done, tell the capture thread to keep filling the buffer
            if buffer_one:
                self.start_time1.value = 0

            else:
                self.start_time2.value = 0


            # Cut out the compressed frames to the proper size
            compressed = compressed[:, :self.config.height, :self.config.width]
            ave16 = ave16[:self.config.height, :self.config.width]

            log.info("Compression time: {:.3f} s".format(time.time() - t))
            t = time.time()

            # Save the compressed image
            filename_millis, filename_micros = self.saveFF(compressed, startTime, n*256, ave16=ave16)
            n += 1
            
            log.info("Saving time: {:.3f} s".format(time.time() - t))


            # Save a live.jpg file to the data directory
            if self.config.live_jpg:
                log.debug("Saving live jpg")
                self.saveLiveJPG(compressed, startTime)


            # Save the extracted intensities per every field
            FieldIntensities.saveFieldIntensitiesBin(field_intensities, self.data_dir, filename_micros)

            # Run the extractor (on the pre-handshake snapshot, never the live buffer)
            if self.config.enable_fireball_detection:
                extractor = Extractor(self.config, self.data_dir)
                extractor.start(frames_snapshot_base, frames.shape, compressed,
                    filename_millis)

                log.debug('Extractor started for: ' + filename_millis)


            # Fully format the filename (this could not have been done before as the extractor has to add
            # the FR prefix to the given file name)
            filename = "FF_" + filename_millis + "." + self.config.ff_format


            # Run the detection on the file, if the detector handle was given
            if self.detector is not None:

                # Add the file to the detector queue
                self.detector.addJob([self.data_dir, filename, self.config])
                log.debug('Added file for detection: {:s}'.format(filename))



        log.debug('Compression run exit')

        time.sleep(1.0)
        self.run_exited.set()

        # Force-exit the process. The forked QueuedPool Manager proxy threads
        # hold open socket connections that survive even after dropping all
        # Python references. os._exit() is the only reliable way to terminate
        # the process without waiting for those threads.
        os._exit(0)


