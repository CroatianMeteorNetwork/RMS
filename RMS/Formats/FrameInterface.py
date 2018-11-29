""" Module which provides a common functional interface for loading video frames/images from different
    input data formats. """

from __future__ import print_function, division, absolute_import

import os
import sys
import copy
import time
import datetime

# tkinter import that works on both Python 2 and 3
try:
    from tkinter import messagebox
except:
    import tkMessageBox as messagebox


import cv2
import numpy as np

from RMS.Astrometry.Conversions import unixTime2Date, datetime2UnixTime
from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import reconstructFrame as reconstructFrameFF
from RMS.Formats.FFfile import validFFName, filenameToDatetime
from RMS.Formats.FFfile import getMiddleTimeFF, selectFFFrames
from RMS.Formats.Vid import readFrame as readVidFrame
from RMS.Formats.Vid import VidStruct


# Morphology - Cython init
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from RMS.Routines.DynamicFTPCompressionCy import FFMimickInterface


def getCacheID(first_frame, size):
    """ Get the frame chunk ID. """

    return "first:{:d},size:{:d}".format(int(first_frame), int(size))



def computeFramesToRead(read_nframes, total_frames, fr_chunk_no, current_frame_chunk, first_frame):

    ### Compute the number of frames to read

    if read_nframes == -1:
        frames_to_read = total_frames

    else:

        # If the number of frames to read was not given, use the default value
        if read_nframes is None:
            frames_to_read = fr_chunk_no

        else:
            frames_to_read = read_nframes

        # Make sure not to try to read more frames than there's available
        if first_frame + fr_chunk_no > total_frames:
            frames_to_read = total_frames - first_frame


    return int(frames_to_read)



# class FFMimickInterface(object):
#     def __init__(self, nrows, ncols, dtype):
#         """ Structure which is used to make FF file format data. It mimicks the interface of an FF structure. """

#         self.nrows = nrows
#         self.ncols = ncols
#         self.dtype = dtype

#         # Init the empty structures
#         self.maxpixel = np.zeros(shape=(self.nrows, self.ncols), dtype=self.dtype)
#         self.acc = np.zeros(shape=(self.nrows, self.ncols), dtype=np.uint64)
#         self.stdpixel = np.zeros(shape=(self.nrows, self.ncols), dtype=np.uint64)

#         self.nframes = 0

#         # False if dark and flat weren't applied, True otherwise (False be default)
#         self.calibrated = False


#     def addFrame(self, frame):
#         """ Add raw frame for computation of FF data. """

#         # Get the maximum values
#         self.maxpixel = np.fmax(self.maxpixel, frame)

#         frame_conv = frame.astype(np.uint64)

#         self.acc += frame_conv
#         self.stdpixel += frame_conv**2

#         self.nframes += 1


#     def finish(self):
#         """ Finish making an FF structure. """

#         # Remove the contribution of the maxpixel to the avepixel
#         self.acc -= self.maxpixel

#         self.avepixel = self.acc//(self.nframes - 1)
#         #self.avepixel = self.acc//self.nframes


#         # Compute the standard deviation
#         self.stdpixel -= (self.maxpixel.astype(np.uint64))**2
#         self.stdpixel -= self.acc*self.avepixel
        
#         self.stdpixel  = np.sqrt(self.stdpixel/(self.nframes - 2))
#         #self.stdpixel  = np.sqrt(self.stdpixel//(self.nframes - 1))

#         # Make sure there are no zeros in standard deviation
#         self.stdpixel[self.stdpixel == 0] = 1

#         # Convert stddev and avepixel to appropriate format
#         self.avepixel = self.avepixel.astype(self.dtype)
#         self.stdpixel = self.stdpixel.astype(self.dtype)

#         # print('---')
#         # print('nframes', self.nframes)
#         # print('mean stddev2:', np.mean(self.stdpixel))
#         # print('mean avepixel2', np.mean(self.avepixel))
#         # print('mean maxpixel', np.mean(self.maxpixel))




class InputTypeFF(object):
    def __init__(self, dir_path, config, single_ff=False):
        """ Input file type handle for FF files.
        
        Arguments:
            dir_path: [str] Path to directory with FF files. 
            config: [ConfigStruct object]

        Keyword arguments:
            single_ff: [bool] If True, a single FF file should be given as input, and not a directory with FF
                files. False by default.

        """

        self.input_type = 'ff'

        self.dir_path = dir_path
        self.config = config

        self.single_ff = single_ff

        # This type of input should have the calstars file
        self.require_calstars = True

        # Don't byteswap the images
        self.byteswap = False

        if self.single_ff:
            print('Using FF file:', self.dir_path)
        else:
            print('Using FF files from:', self.dir_path)


        self.ff_list = []
        self.ff = None
        self.ff_frame = None
        self.frame_ff_name = None


        # Add the single FF file to the list
        if self.single_ff:

            self.dir_path, file_name = os.path.split(self.dir_path)

            self.ff_list.append(file_name)

        else:

            # Get a list of FF files in the folder
            for file_name in os.listdir(dir_path):
                if validFFName(file_name):
                    self.ff_list.append(file_name)


        # Check that there are any FF files in the folder
        if not self.ff_list:
            messagebox.showinfo(title='File list warning', message='No FF files in the selected folder!')

            sys.exit()


        # Sort the FF list
        self.ff_list = sorted(self.ff_list)

        # Init the first file
        self.current_ff_index = 0
        self.current_ff_file = self.ff_list[self.current_ff_index]

        # Update the beginning time
        self.beginning_datetime = filenameToDatetime(self.current_ff_file)

        # Init the frame number
        self.current_frame = 0


        # Number for frames to read by default
        self.fr_chunk_no = 256

        # Initially assume this to be true, but this will change after the first load
        self.total_frames = self.fr_chunk_no


        self.cache = {}
        self.cache_frames = {}

        # Load the first chunk for initing parameters
        self.loadChunk()

        # Read FPS from FF file if available, otherwise use from config
        if hasattr(self.ff, 'fps'):
            self.fps = self.ff.fps

        else:
            self.fps = self.config.fps

        # Get the image size
        self.nrows = self.ff.nrows
        self.ncols = self.ff.ncols

        # Compute the total number of frames in all video files
        self.total_frames = len(self.ff_list)*self.ff.nframes



    def nextChunk(self):
        """ Go to the next FF file. """

        self.current_ff_index = (self.current_ff_index + 1)%len(self.ff_list)
        self.current_ff_file = self.ff_list[self.current_ff_index]

        # Update the beginning time
        self.beginning_datetime = filenameToDatetime(self.current_ff_file)



    def prevChunk(self):
        """ Go to the previous FF file. """

        self.current_ff_index = (self.current_ff_index - 1)%len(self.ff_list)
        self.current_ff_file = self.ff_list[self.current_ff_index]

        # Update the beginning time
        self.beginning_datetime = filenameToDatetime(self.current_ff_file)



    def loadChunk(self, first_frame=None, read_nframes=None):
        """ Load the frame chunk file. 
    
        Keyword arguments:
            first_frame: [int] First frame to read.
            read_nframes: [int] Number of frames to read. If not given (None), self.fr_chunk_no frames will be
                read. If -1, all frames will be read in.
        """

        # If no extra arguments were given, assume it's a normal read and read the current FF file
        if (first_frame is None) and (read_nframes is None):
            ff_file = self.current_ff_file

        else:
            ff_file = None


        # If all frames should be taken, set the frame to 0
        if read_nframes == -1:

            first_frame = 0

        # Otherwise, set it to the appropriate chunk or given first frame
        else:

            # Compute the first frame if not given
            if first_frame is None:
                first_frame = self.current_ff_index*self.fr_chunk_no

            # Make sure the first frame is within the limits
            first_frame = first_frame%self.total_frames

        # Compute the number of frames to read
        frames_to_read = computeFramesToRead(read_nframes, self.total_frames, self.fr_chunk_no, \
            self.current_ff_index, first_frame)


        # If it's a normal read, get the current FF file
        if ff_file is not None:
            cache_id = self.current_ff_file
            whole_ff = True

        # If the number of frames is exactly one FF file from beginning to end
        #   just return the whole FF file
        elif (first_frame%self.fr_chunk_no == 0) and (frames_to_read == self.fr_chunk_no):

            # Find the FF file to read
            ff_file = self.ff_list[first_frame//self.fr_chunk_no]

            cache_id = ff_file
            whole_ff = True

        else:

            # Get the cache ID
            cache_id = getCacheID(first_frame, frames_to_read)
            whole_ff = False


        # Check if this chunk has been cached
        if cache_id in self.cache:
            self.ff = self.cache[cache_id]
            return self.ff


        # If the whole file has to be returned
        if whole_ff:

            # Load the FF file from disk
            self.ff = readFF(self.dir_path, ff_file)

        # If a selection of frames has to be reconstructed, go through all FF files and create new FF
        else:

            # Determine which FF files are to be read and which frame ranges from each
            frame_ranges = []
            ffs_to_read = []
            for i in range(frames_to_read):

                # Compute the frame index
                fr_index = first_frame + i

                # Get the file name of the file that has to be read
                file_index = fr_index//self.fr_chunk_no
                file_name = self.ff_list[file_index]

                # Get the frame index on the FF
                ff_local_index = fr_index%self.fr_chunk_no

                # Add the file to the list
                if file_name not in ffs_to_read:
                    
                    # Add the FF to the list of frames to read
                    ffs_to_read.append(file_name)

                    # Add the frame index to the list
                    frame_ranges.append([ff_local_index])

                # Store the local frame number to the list if on the same FF file
                else:
                    frame_ranges[len(ffs_to_read) - 1].append(ff_local_index)

            # If there is only one FF file to read, make a selection of frames, but preserve everything else
            if len(ffs_to_read) == 1:

                file_name = ffs_to_read[0]

                frame_range = frame_ranges[0]

                # Compute the range of frames to read
                min_frame = np.min(frame_range)
                max_frame = np.max(frame_range)

                # Read the FF file
                self.ff = readFF(self.dir_path, file_name)

                # Select the frames
                self.ff.maxpixel = selectFFFrames(self.ff.maxpixel, self.ff, min_frame, max_frame)


            else:



                # Init an empty FF structure
                self.ff = FFMimickInterface(self.nrows, self.ncols, self.fr_chunk_no, np.uint8)

                # Store maxpixel selections, avepixels, stdpixels
                maxpixel_list = []
                avepixel_list = []
                stdpixel_list = []

                # Read the FF files that have to read and reconstruct the frames
                for file_name, frame_range in zip(ffs_to_read, frame_ranges):

                    # Compute the range of frames to read
                    min_frame = np.min(frame_range)
                    max_frame = np.max(frame_range)

                    # Read the FF file
                    ff = readFF(self.dir_path, file_name)

                    # Reconstruct the maxpixel in the given frame range
                    maxpixel = selectFFFrames(ff.maxpixel, ff, min_frame, max_frame)

                    # Reconstruct the avepixel in the given frame range
                    avepixel = selectFFFrames(ff.avepixel, ff, min_frame, max_frame)

                    # Store the computed frames
                    maxpixel_list.append(maxpixel)
                    avepixel_list.append(avepixel)
                    stdpixel_list.append(ff.stdpixel)


                # Immidiately extract the appropriate frames
                if len(maxpixel_list) == 1:

                    self.ff.maxpixel = maxpixel_list[0]
                    self.ff.avepixel = avepixel_list[0]
                    self.ff.stdpixel = stdpixel_list[0]

                # Otherwise, compute the combined FF
                else:
                    maxpixel_list = np.array(maxpixel_list)
                    avepixel_list = np.array(avepixel_list)
                    stdpixel_list = np.array(stdpixel_list)

                    self.ff.maxpixel = np.max(maxpixel_list, axis=0)

                    # The maximum of the avepixel is taken because only the frame range of avepixel is taken
                    self.ff.avepixel = np.max(avepixel_list, axis=0)

                    self.ff.stdpixel = np.max(stdpixel_list, axis=0)

                
        # Store the loaded file to cache for faster loading
        self.cache = {}
        self.cache[cache_id] = self.ff

        return self.ff


    def name(self, beginning=None):
        """ Return the name of the FF file. """

        return self.current_ff_file


    def currentTime(self, dt_obj=False):
        """ Return the middle time of the current image. """

        if dt_obj:
            return datetime.datetime(*getMiddleTimeFF(self.current_ff_file, self.fps, \
                ret_milliseconds=False))

        else:
            return getMiddleTimeFF(self.current_ff_file, self.fps, ret_milliseconds=True)



    def nextFrame(self):
        """ Increment the current frame. """

        self.current_frame = (self.current_frame + 1)%self.total_frames


    def prevFrame(self):
        """ Decrement the current frame. """
        
        self.current_frame = (self.current_frame - 1)%self.total_frames


    def setFrame(self, fr_num):
        """ Set the current frame. 
    
        Arguments:
            fr_num: [float] Frame number to set.
        """

        self.current_frame = fr_num%self.total_frames


    def loadFrame(self, avepixel=False):
        """ Load the current frame. """

        # Compute which file read
        file_index = self.current_frame//self.fr_chunk_no
        file_name = self.ff_list[file_index]

        # Try loading the FF file from cache
        if file_name in self.cache_frames:
            self.ff_frame = self.cache_frames[file_name]
        else:
            # Load the FF file from disk
            self.ff_frame = readFF(self.dir_path, file_name)

            # Put the FF into separate cache
            self.cache_frames = {}
            self.cache_frames[file_name] = self.ff_frame

        # Store the name of the current FF file from which the frame was read
        self.frame_ff_name = file_name

        # Reconstruct the frame from an FF file
        frame = reconstructFrameFF(self.ff_frame, self.current_frame%self.fr_chunk_no, avepixel=avepixel)

        return frame


    def currentFrameTime(self, frame_no=None, dt_obj=False):
        """ Return the time of the frame. """

        if frame_no is None:
            frame_no = self.current_frame

        # Compute the datetime of the current frame
        dt = self.beginning_datetime + datetime.timedelta(seconds=frame_no/self.fps)
        
        if dt_obj:
            return dt

        else:
            return (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond/1000)




class InputTypeVideo(object):
    def __init__(self, dir_path, config, beginning_time=None):
        """ Input file type handle for video files.
        
        Arguments:
            dir_path: [str] Path to the video file.
            config: [ConfigStruct object]

        Keyword arguments:
            beginning_time: [datetime] datetime of the beginning of the video. Optional, None by default.

        """

        self.input_type = 'video'

        # Separate dir path and file name
        self.file_path = dir_path
        self.dir_path, self.file_name = os.path.split(dir_path)
        
        self.config = config

        self.ff = None

        # This type of input probably won't have any calstars files
        self.require_calstars = False

        # Don't byteswap the images
        self.byteswap = False

        # Remove the file extension
        file_name_noext = ".".join(self.file_name.split('.')[:-1])

        if beginning_time is None:
            
            try:
                # Try reading the beginning time of the video from the name if time is not given
                self.beginning_datetime = datetime.datetime.strptime(file_name_noext, "%Y%m%d_%H%M%S.%f")

            except:
                messagebox.showerror('Input error', 'The time of the beginning cannot be read from the file name! Either change the name of the file to be in the YYYYMMDD_hhmmss format, or specify the beginning time using command line options.')
                sys.exit()

        else:
            self.beginning_datetime = beginning_time



        print('Using video file:', self.file_path)

        # Open the video file
        self.cap = cv2.VideoCapture(self.file_path)

        self.current_frame_chunk = 0

        # Prop values: https://stackoverflow.com/questions/11420748/setting-camera-parameters-in-opencv-python

        # Get the FPS
        self.fps = self.cap.get(5)

        # Get the total time number of video frames in the file
        self.total_frames = int(self.cap.get(7))

        # Get the image size
        self.nrows = int(self.cap.get(4))
        self.ncols = int(self.cap.get(3))


        # Set the number of frames to be used for averaging and maxpixels
        self.fr_chunk_no = 256

        # Compute the number of frame chunks
        self.total_fr_chunks = self.total_frames//self.fr_chunk_no
        if self.total_fr_chunks == 0:
            self.total_fr_chunks = 1

        self.current_fr_chunk_size = self.fr_chunk_no

        self.current_frame = 0


        self.cache = {}


    def nextChunk(self):
        """ Go to the next frame chunk. """

        self.current_frame_chunk += 1
        self.current_frame_chunk = self.current_frame_chunk%self.total_fr_chunks

        # Update the current frame
        self.current_frame = self.current_frame_chunk*self.fr_chunk_no


    def prevChunk(self):
        """ Go to the previous frame chunk. """

        self.current_frame_chunk -= 1
        self.current_frame_chunk = self.current_frame_chunk%self.total_fr_chunks

        # Update the current frame
        self.current_frame = self.current_frame_chunk*self.fr_chunk_no


    def loadChunk(self, first_frame=None, read_nframes=None):
        """ Load the frame chunk file. 
    
        Keyword arguments:
            first_frame: [int] First frame to read.
            read_nframes: [int] Number of frames to read. If not given (None), self.fr_chunk_no frames will be
                read. If -1, all frames will be read in.
        """


        # If all frames should be taken, set the frame to 0
        if read_nframes == -1:

            first_frame = 0

            self.cap.set(1, 0)

        # Otherwise, set it to the appropriate chunk
        else:

            # Compute the first frame if it wasn't given
            if first_frame is None:
                first_frame = self.current_frame_chunk*self.fr_chunk_no

            # Make sure the first frame is within the limits
            first_frame = first_frame%self.total_frames


        # Set the first frame location
        self.cap.set(1, first_frame)


        # Compute the number of frames to read
        frames_to_read = computeFramesToRead(read_nframes, self.total_frames, self.fr_chunk_no, \
            self.current_frame_chunk, first_frame)


        # Get the cache ID
        cache_id = getCacheID(first_frame, frames_to_read)


        # Check if this chunk has been cached
        if cache_id in self.cache:
            frame, self.current_fr_chunk_size = self.cache[cache_id]
            return frame


        # Init making the FF structure
        ff_struct_fake = FFMimickInterface(self.nrows, self.ncols, np.uint8)

        # Load the chunk of frames
        for i in range(frames_to_read):

            ret, frame = self.cap.read()

            # If the end of the video files was reached, stop the loop
            if frame is None:
                break

            # Convert frame to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Add frame for FF processing
            ff_struct_fake.addFrame(frame)


        self.current_fr_chunk_size = i + 1

        # Finish making the fake FF file
        ff_struct_fake.finish()


        # Store the FF struct to cache to avoid recomputing
        self.cache = {}

        self.cache[cache_id] = [ff_struct_fake, self.current_fr_chunk_size]

        # Set the computed chunk as the current FF
        self.ff = ff_struct_fake

        return ff_struct_fake
        


    def name(self, beginning=False):
        """ Return the name of the chunk, which is just the time of the middle of the current frame chunk. 
            Alternatively, the beginning of the whole file can be returned.

        Keyword arguments:
            beginning: [bool] If True, the beginning time of the file will be retunred instead of the middle
                time of the chunk.
        """

        if beginning:
            return str(self.beginning_datetime)
        else:
            return str(self.currentTime(dt_obj=True))


    def currentTime(self, dt_obj=False):
        """ Return the mean time of the current image. """

        # Compute number of seconds since the beginning of the video file to the mean time of the frame chunk
        seconds_since_beginning = (self.current_frame_chunk*self.fr_chunk_no \
            + self.current_fr_chunk_size/2)/self.fps

        # Compute the absolute time
        dt = self.beginning_datetime + datetime.timedelta(seconds=seconds_since_beginning)

        if dt_obj:
            return dt

        else:
            return (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond/1000)


    def nextFrame(self):
        """ Increment the current frame. """

        self.current_frame = (self.current_frame + 1)%self.total_frames


    def prevFrame(self):
        """ Decrement the current frame. """
        
        self.current_frame = (self.current_frame - 1)%self.total_frames


    def setFrame(self, fr_num):
        """ Set the current frame. 
    
        Arguments:
            fr_num: [float] Frame number to set.
        """

        self.current_frame = fr_num%self.total_frames


    def loadFrame(self, avepixel=False):
        """ Load the current frame. """


        # Set the frame location
        self.cap.set(1, self.current_frame)

        # Read the frame
        ret, frame = self.cap.read()

        # Convert frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame


    def currentFrameTime(self, frame_no=None, dt_obj=False):
        """ Return the time of the frame. """

        if frame_no is None:
            frame_no = self.current_frame

        # Compute the datetime of the current frame
        dt = self.beginning_datetime + datetime.timedelta(seconds=frame_no/self.fps)
        
        if dt_obj:
            return dt

        else:
            return (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond/1000)






class InputTypeUWOVid(object):
    def __init__(self, dir_path, config):
        """ Input file type handle for UWO .vid files.
        
        Arguments:
            dir_path: [str] Path to the vid file.
            config: [ConfigStruct object]

        """

        self.input_type = 'vid'



        # Separate directory path and file name
        self.vid_path = dir_path
        self.dir_path, vid_file = os.path.split(dir_path)

        self.config = config

        self.ff = None

        # This type of input probably won't have any calstars files
        self.require_calstars = False

        # Byteswap the images
        self.byteswap = True


        print('Using vid file:', self.vid_path)

        # Open the vid file
        self.vid = VidStruct()
        self.vid_file = open(self.vid_path, 'rb')

        # Read one video frame and rewind to beginning
        readVidFrame(self.vid, self.vid_file)
        self.vidinfo = copy.deepcopy(self.vid)
        self.vid_file.seek(0)
        
        # Try reading the beginning time of the video from the name
        self.beginning_datetime = unixTime2Date(self.vidinfo.ts, self.vidinfo.tu, dt_obj=True)


        self.current_frame_chunk = 0
        self.current_frame = 0
        self.current_fr_chunk_size = 0

        # Get the total time number of video frames in the file
        self.total_frames = os.path.getsize(self.vid_path)//self.vidinfo.seqlen

        # Get the image size
        self.nrows = self.vidinfo.ht
        self.ncols = self.vidinfo.wid


        # Set the number of frames to be used for averaging and maxpixels
        self.fr_chunk_no = 128

        # Compute the number of frame chunks
        self.total_fr_chunks = self.total_frames//self.fr_chunk_no
        if self.total_fr_chunks == 0:
            self.total_fr_chunks = 1

        self.frame_chunk_unix_times = []


        self.cache = {}

        # Init the dictionary for storing unix times of corresponding frames that were already loaded
        self.utime_frame_dict = {}

        # Do the initial load
        self.loadChunk()


        # Estimate the FPS
        self.fps = 1/((self.frame_chunk_unix_times[-1] - self.frame_chunk_unix_times[0])/self.current_fr_chunk_size)


    def nextChunk(self):
        """ Go to the next frame chunk. """

        self.current_frame_chunk += 1
        self.current_frame_chunk = self.current_frame_chunk%self.total_fr_chunks

        # Update the current frame
        self.current_frame = self.current_frame_chunk*self.fr_chunk_no


    def prevChunk(self):
        """ Go to the previous frame chunk. """

        self.current_frame_chunk -= 1
        self.current_frame_chunk = self.current_frame_chunk%self.total_fr_chunks

        # Update the current frame
        self.current_frame = self.current_frame_chunk*self.fr_chunk_no


    def loadChunk(self, first_frame=None, read_nframes=None):
        """ Load the frame chunk file. 
    
        Keyword arguments:
            first_frame: [int] First frame to read.
            read_nframes: [int] Number of frames to read. If not given (None), self.fr_chunk_no frames will be
                read. If -1, all frames will be read in.
        """


        # If all frames should be taken, set the frame to 0
        if read_nframes == -1:

            first_frame = 0

        # Otherwise, set it to the appropriate chunk or given first frame
        else:

            # Compute the first frame if not given
            if first_frame is None:
                first_frame = self.current_frame_chunk*self.fr_chunk_no

            # Make sure the first frame is within the limits
            first_frame = first_frame%self.total_frames

        # Compute the number of frames to read
        frames_to_read = computeFramesToRead(read_nframes, self.total_frames, self.fr_chunk_no, \
            self.current_frame_chunk, first_frame)


        # Get the cache ID
        cache_id = getCacheID(first_frame, frames_to_read)


        # Check if this chunk has been cached
        if cache_id in self.cache:
            frame, self.frame_chunk_unix_times, self.current_fr_chunk_size = self.cache[cache_id]
            return frame




        # Set the vid file pointer to the right byte
        self.vid_file.seek(first_frame*self.vidinfo.seqlen)

        # Init making the FF structure
        ff_struct_fake = FFMimickInterface(self.nrows, self.ncols, np.uint16)

        self.frame_chunk_unix_times = []

        # Load the chunk of frames
        for i in range(frames_to_read):

            frame = readVidFrame(self.vid, self.vid_file)

            # If the end of the vid file was reached, stop the loop
            if frame is None:
                break

            frame = frame.astype(np.uint16)


            unix_time = self.vid.ts + self.vid.tu/1000000.0

            # Add the unix time to list
            self.frame_chunk_unix_times.append(unix_time)
            
            # Add frame for FF processing
            ff_struct_fake.addFrame(frame)

            unix_time_lst = (self.vid.ts, self.vid.tu)
            if unix_time_lst not in self.utime_frame_dict:
                self.utime_frame_dict[first_frame + i] = unix_time_lst


        self.current_fr_chunk_size = i + 1

        # Finish making the fake FF file
        ff_struct_fake.finish()

        # Store the FF struct to cache to avoid recomputing
        self.cache = {}

        # Save the computed FF to cache
        self.cache[cache_id] = [ff_struct_fake, self.frame_chunk_unix_times, self.current_fr_chunk_size]

        # Set the computed chunk as the current FF
        self.ff = ff_struct_fake

        return ff_struct_fake
        


    def name(self, beginning=False):
        """ Return the name of the chunk, which is just the time of the middle of the current frame chunk. 
            Alternatively, the beginning of the whole file can be returned.

        Keyword arguments:
            beginning: [bool] If True, the beginning time of the file will be retunred instead of the middle
                time of the chunk.
        """

        if beginning:
            return str(self.beginning_datetime)
        else:
            return str(self.currentTime(dt_obj=True))


    def currentTime(self, dt_obj=False):
        """ Return the mean time of the current image. """

        # Compute the mean UNIX time
        mean_utime = np.mean(self.frame_chunk_unix_times)

        mean_ts = int(mean_utime)
        mean_tu = int((mean_utime - mean_ts)*1000000)

        return unixTime2Date(mean_ts, mean_tu, dt_obj=dt_obj)


    def nextFrame(self):
        """ Increment the current frame. """

        self.current_frame = (self.current_frame + 1)%self.total_frames


    def prevFrame(self):
        """ Decrement the current frame. """
        
        self.current_frame = (self.current_frame - 1)%self.total_frames


    def setFrame(self, fr_num):
        """ Set the current frame. 
    
        Arguments:
            fr_num: [float] Frame number to set.
        """

        self.current_frame = fr_num%self.total_frames


    def loadFrame(self, avepixel=False):
        """ Load the current frame. """

        # Set the vid file pointer to the right byte
        self.vid_file.seek(self.current_frame*self.vidinfo.seqlen)

        # Load a frame
        frame = readVidFrame(self.vid, self.vid_file)

        # Save the frame time
        self.current_frame_time = unixTime2Date(self.vid.ts, self.vid.tu, dt_obj=True) 

        unix_time_lst = (self.vid.ts, self.vid.tu)
        if unix_time_lst not in self.utime_frame_dict:
            self.utime_frame_dict[self.current_frame] = unix_time_lst

        return frame



    def currentFrameTime(self, frame_no=None, dt_obj=False):
        """ Return the time of the frame. """
        
        if frame_no is None:
            dt = self.current_frame_time


        else:

            # If the frame number was given, read it from the dictionary or from the file
            if frame_no in self.utime_frame_dict:
                dt = unixTime2Date(*self.utime_frame_dict[frame_no], dt_obj=True)


            else:

                # Set the vid file to the right frame
                self.vid_file.seek(frame_no*self.vidinfo.seqlen)

                # Read the vid file metadata
                readVidFrame(self.vid, self.vid_file)

                # Store the current time to the dictionary
                unix_time_lst = (self.vid.ts, self.vid.tu)
                if unix_time_lst not in self.utime_frame_dict:
                    self.utime_frame_dict[frame_no] = unix_time_lst


                # Revert the vid file pointer to the current frame in the image handle
                self.vid_file.seek((self.current_frame + 1)*self.vidinfo.seqlen)

                dt = unixTime2Date(*unix_time_lst, dt_obj=True)


        if dt_obj:
            return dt

        else:
            return (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond/1000)





class InputTypeImages(object):
    def __init__(self, dir_path, config, beginning_time=None, fps=None):
        """ Input file type handle for a folder with images.
        
        Arguments:
            dir_path: [str] Path to the vid file.
            config: [ConfigStruct object]

        """

        self.input_type = 'images'

        self.dir_path = dir_path
        self.config = config

        self.ff = None

        # This type of input probably won't have any calstars files
        self.require_calstars = False


        ### Find images in the given folder ###
        img_types = ['.png', '.jpg', '.bmp']

        self.img_list = []

        for file_name in sorted(os.listdir(self.dir_path)):

            # Check if the file ends with support file extensions
            for fextens in img_types:

                if file_name.lower().endswith(fextens):

                    # Don't take flats, biases, darks, etc.
                    if ('flat' in file_name.lower()) or ('dark' in file_name.lower()) \
                        or ('bias' in file_name.lower()) or ('grid' in file_name.lower()):
                            continue

                    self.img_list.append(file_name)
                    break


        if len(self.img_list) == 0:
            messagebox.showerror('Input error', "Can't find any images in the given directory! Only PNG, JPG and BMP are supported!")
            sys.exit()

        ### ###



        ### Try to detect if the given images are UWO-style PNGs ###
        
        self.uwo_png_mode = False

        # Load the first image
        img = self.loadFrame(fr_no=0)

        # Check the magick number
        if (img[0][0] == 22121) and (img[0][1] == 17410):
            
            self.uwo_png_mode = True

            # Get the beginning time
            self.loadFrame(fr_no=0)
            beginning_time = self.uwo_png_frame_time
            
            print('UWO PNG mode')

        ###

        # Decide if images need to be byteswapped
        if self.uwo_png_mode:
            self.byteswap = True

        else:
            self.byteswap = False


        self.uwo_png_frame_time = None
        self.uwo_png_dt_list = None


        # Check if the beginning time was given
        if beginning_time is None:
            
            try:
                # Try reading the beginning time of the video from the name if time is not given
                self.beginning_datetime = datetime.datetime.strptime(os.path.basename(self.dir_path), \
                    "%Y%m%d_%H%M%S.%f")

            except:
                messagebox.showerror('Input error', 'The time of the beginning cannot be read from the file name! Either change the name of the file to be in the YYYYMMDD_hhmmss format, or specify the beginning time using command line options.')
                sys.exit()

        else:
            self.beginning_datetime = beginning_time



        print('Using folder:', self.dir_path)


        self.current_frame_chunk = 0

        # Compute the total number of used frames
        self.total_frames = len(self.img_list)

        self.current_frame = 0
        self.current_img_file = self.img_list[self.current_frame]

        # Load the first image
        img = self.loadFrame()

        # Get the image size
        self.nrows = img.shape[0]
        self.ncols = img.shape[1]

        # Get the image dtype
        self.img_dtype = img.dtype


        # Set the number of frames to be used for averaging and maxpixels
        self.fr_chunk_no = 64

        self.current_fr_chunk_size = self.fr_chunk_no

        # Compute the number of frame chunks
        self.total_fr_chunks = self.total_frames//self.fr_chunk_no
        if self.total_fr_chunks == 0:
            self.total_fr_chunks = 1



        self.cache = {}

        # Do the initial load
        self.loadChunk()


        # Estimate the FPS if UWO pngs are given
        if self.uwo_png_mode:

            # Convert datetimes to Unix times
            unix_times = [datetime2UnixTime(dt) for dt in self.uwo_png_dt_list]

            fps = 1/((unix_times[-1] - unix_times[0])/self.current_fr_chunk_size)


        # If FPS is not given, use one from the config file
        if fps is None:

            self.fps = self.config.fps
            print('Using FPS from config file: ', self.fps)

        else:

            self.fps = fps
            print('Using FPS:', self.fps)



    def nextChunk(self):
        """ Go to the next frame chunk. """

        self.current_frame_chunk += 1
        self.current_frame_chunk = self.current_frame_chunk%self.total_fr_chunks

        self.current_frame = self.current_frame_chunk*self.fr_chunk_no


    def prevChunk(self):
        """ Go to the previous frame chunk. """

        self.current_frame_chunk -= 1
        self.current_frame_chunk = self.current_frame_chunk%self.total_fr_chunks

        self.current_frame = self.current_frame_chunk*self.fr_chunk_no


    def loadChunk(self, first_frame=None, read_nframes=None):
        """ Load the frame chunk file. 
    
        Keyword arguments:
            first_frame: [int] First frame to read.
            read_nframes: [int] Number of frames to read. If not given (None), self.fr_chunk_no frames will be
                read. If -1, all frames will be read in.
        """

            
        # Compute the first index of the chunk
        if read_nframes == -1:
            first_frame = 0

        else:

            # Compute the first frame if it wasn't given
            if first_frame is None:
                first_frame = self.current_frame_chunk*self.fr_chunk_no

            # Make sure the first frame is within the limits
            first_frame = first_frame%self.total_frames


        # Compute the number of frames to read
        frames_to_read = computeFramesToRead(read_nframes, self.total_frames, self.fr_chunk_no, \
            self.current_frame_chunk, first_frame)

        # Get the cache ID
        cache_id = getCacheID(first_frame, frames_to_read)


        # Check if this chunk has been cached
        if cache_id in self.cache:
            frame, self.uwo_png_dt_list, self.current_fr_chunk_size = self.cache[cache_id]
            return frame


        # Init making the FF structure
        ff_struct_fake = FFMimickInterface(self.nrows, self.ncols, self.img_dtype)

        self.uwo_png_dt_list = []

        # Load the chunk of frames
        for i in range(frames_to_read):

            # Compute the image index
            img_indx = first_frame + i

            # Stop the loop if the ends of images has been reached
            if img_indx >= self.total_frames - 1:
                break

            # Load the image
            frame = self.loadFrame(fr_no=img_indx)

            # Add frame for FF processing
            ff_struct_fake.addFrame(frame)

            # Add the datetime of the frame to list of the UWO png is used
            if self.uwo_png_mode:
                self.uwo_png_dt_list.append(self.currentFrameTime(dt_obj=True))


        self.current_fr_chunk_size = i

        # Finish making the fake FF file
        ff_struct_fake.finish()


        # Store the FF struct to cache to avoid recomputing
        self.cache = {}

        self.cache[cache_id] = [ff_struct_fake, self.uwo_png_dt_list, self.current_fr_chunk_size]

        # Set the computed chunk as the current FF
        self.ff = ff_struct_fake

        return ff_struct_fake
    

    def nextFrame(self):
        """ Increment current frame. """

        self.current_frame = (self.current_frame + 1)%self.total_frames
        self.current_img_file = self.img_list[self.current_frame]


    def prevFrame(self):
        """ Increment current frame. """

        self.current_frame = (self.current_frame - 1)%self.total_frames
        self.current_img_file = self.img_list[self.current_frame]


    def setFrame(self, fr_num):
        """ Set the current frame. 
    
        Arguments:
            fr_num: [float] Frame number to set.
        """

        self.current_frame = fr_num%self.total_frames
        self.current_img_file = self.img_list[self.current_frame]


    def loadFrame(self, avepixel=None, fr_no=None):
        """ Loads the current frame. 
    
        Keyword arguments:
            avepixel: [bool] Does nothing, just for function interface consistency with other input types.
            fr_no: [int] Load a specific frame. None by defualt, then the current frame will be loaded.
        """


        # If a special frame number was given, use that one
        if fr_no is not None:
            current_img_file = self.img_list[fr_no]

        else:
            current_img_file = self.current_img_file

        # Get the current image
        img = cv2.imread(os.path.join(self.dir_path, current_img_file), -1)

        # Convert the image to black and white if it's 8 bit
        if 8*img.itemsize == 8:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        
        if self.uwo_png_mode:

            # Byteswap if it's the UWO style png
            img = img.byteswap()

            # Read the time from the image
            ts = img[0][6] + (img[0][7] << 16)
            tu = img[0][8] + (img[0][9] << 16)

            self.uwo_png_frame_time = unixTime2Date(ts, tu, dt_obj=True)


        return img




    def name(self, beginning=False):
        """ Return the name of the chunk, which is just the time of the middle of the current frame chunk. 
            Alternatively, the beginning of the whole file can be returned.

        Keyword arguments:
            beginning: [bool] If True, the beginning time of the file will be retunred instead of the middle
                time of the chunk.
        """

        if beginning:
            return str(self.beginning_datetime)

        else:
            return str(self.currentTime(dt_obj=True))


    def currentTime(self, dt_obj=False):
        """ Return the mean time of the current image. """


        if self.uwo_png_mode:

            # Convert datetimes to Unix times
            unix_times = [datetime2UnixTime(dt) for dt in self.uwo_png_dt_list]

            # Compute the mean of unix times
            unix_mean = np.mean(unix_times)

            ts = int(unix_mean)
            tu = (unix_mean - ts)*1000000

            dt = unixTime2Date(ts, tu, dt_obj=True)

        else:

            # Compute number of seconds since the beginning of the video file to the mean time of the frame chunk
            seconds_since_beginning = (self.current_frame_chunk*self.fr_chunk_no \
                + self.current_fr_chunk_size/2)/self.fps

            # Compute the absolute time
            dt = self.beginning_datetime + datetime.timedelta(seconds=seconds_since_beginning)

        if dt_obj:
            return dt

        else:
            return (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond/1000)



    def currentFrameTime(self, dt_obj=False):
        """ Return the time of the frame. """

        # If the UWO png is used, return the time read from the PNG
        if self.uwo_png_mode:
            
            dt = self.uwo_png_frame_time

        else:

            # Compute the datetime of the current frame
            dt = self.beginning_datetime + datetime.timedelta(seconds=self.current_frame/self.fps)
            


        if dt_obj:
            return dt

        else:
            return (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond/1000)




def detectInputType(input_path, config, beginning_time=None, fps=None, skip_ff_dir=False):
    """ Given the folder of a file, detect the input format.

    Arguments:
        input_path: [str] Input directory path or file name (e.g. dir with FF files, or path to video file).
        config: [Config Struct]

    Keyword arguments:
        beginning_time: [datetime] Datetime of the video beginning. Optional, only can be given for
            video input formats.
        fps: [float] Frames per second, used only when images in a folder are used.
        skip_ff_dir: [bool] Skip the input type where there are multiple FFs in the same directory. False
            by default. This is only used for ManualReduction.

    """


    # If the given dir path is a directory, search for FF files or individual images
    if os.path.isdir(input_path):

        # Check if there are valid FF names in the directory
        if any([validFFName(ff_file) for ff_file in os.listdir(input_path)]):

            if skip_ff_dir:
                messagebox.showinfo('FF directory', 'ManualReduction only works on individual FF files, and not directories with FF files!')
                return None
            else:
                # Init the image handle for FF files in a directory
                img_handle = InputTypeFF(input_path, config)

        # If not, check if there any image files in the folder
        else:
            img_handle = InputTypeImages(input_path, config, beginning_time=beginning_time, fps=fps)


    # If the given path is a file, look for a single FF file, video files, or vid files
    else:

        dir_path, file_name = os.path.split(input_path)

        # Check if a single FF file was given
        if validFFName(file_name):

            # Init the image handle for FF a single FF file
            img_handle = InputTypeFF(input_path, config, single_ff=True)


        # Check if the given file is a video file
        elif file_name.endswith('.mp4') or file_name.endswith('.avi') or file_name.endswith('.mkv'):

            # Init the image hadle for video files
            img_handle = InputTypeVideo(input_path, config, beginning_time=beginning_time)


        # Check if the given files is the UWO .vid format
        elif file_name.endswith('.vid'):
            
            # Init the image handle for UWO-type .vid files
            img_handle = InputTypeUWOVid(input_path, config)


        else:
            messagebox.showerror(title='Input format error', message='Only these video formats are supported: .mp4, .avi, .mkv, .vid!')
            return None


    return img_handle




if __name__ == "__main__":

    import argparse

    import matplotlib.pyplot as plt

    import RMS.ConfigReader as cr

    ### Functions for testing

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="""Test.""", formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('dir_path', metavar='DIRPATH', type=str, nargs=1, \
                    help='Path to data.')

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    # Load the configuration file
    config = cr.parse(".config")


    # Test creating a fake FF
    nframes = 64
    img_h = 20
    img_w = 20

    ff = FFMimickInterface(img_h, img_w, np.uint16)

    frames = np.random.normal(10000, 500, size=(nframes, img_h, img_w)).astype(np.uint16)

    for frame in frames:

        ff.addFrame(frame)


    ff.finish()


    # Compute real values
    avepixel = np.mean(frames, axis=0)
    stdpixel = np.std(frames, axis=0)


    print('Std mean ff:', np.mean(ff.stdpixel))
    print('Std mean:', np.mean(stdpixel))
    print('Mean diff:', np.mean(stdpixel - ff.stdpixel))
    plt.imshow(stdpixel - ff.stdpixel)
    plt.show()


    print('ave mean ff:', np.mean(ff.avepixel))
    print('ave mean:', np.mean(avepixel))
    print('Mean diff:', np.mean(avepixel - ff.avepixel))
    plt.imshow(avepixel - ff.avepixel)
    plt.show()

        



    # # Load the appropriate files
    # img_handle = detectInputType(cml_args.dir_path[0], config)

    # chunk_size = 64

    # for i in range(img_handle.total_frames//chunk_size + 1):
        
    #     first_frame = i*chunk_size

    #     # Load a chunk of frames
    #     ff = img_handle.loadChunk(first_frame=first_frame, read_nframes=chunk_size)

    #     print(first_frame, first_frame + chunk_size)
    #     plt.imshow(ff.maxpixel - ff.avepixel, cmap='gray')
    #     plt.show()


    #     # Show stdpixel
    #     plt.imshow(ff.stdpixel, cmap='gray')
    #     plt.show()

    #     # Show thresholded image
    #     thresh_img = (ff.maxpixel - ff.avepixel) > (1.0*ff.stdpixel + 30)
    #     plt.imshow(thresh_img, cmap='gray')
    #     plt.show()