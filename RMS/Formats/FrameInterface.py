""" Module which provides a common functional interface for loading video frames/images from different
    input data formats. """

from __future__ import print_function, division, absolute_import

import os
import sys
import copy
import datetime

# tkinter import that works on both Python 2 and 3
try:
    from tkinter import messagebox
except:
    import tkMessageBox as messagebox


import cv2
import numpy as np

from RMS.Astrometry.Conversions import unixTime2Date
from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import validFFName
from RMS.Formats.FFfile import getMiddleTimeFF
from RMS.Formats.Vid import readFrame as readVidFrame
from RMS.Formats.Vid import VidStruct




class InputTypeFF(object):
    def __init__(self, dir_path, config):
        """ Input file type handle for FF files.
        
        Arguments:
            dir_path: [str] Path to directory with FF files. 
            config: [ConfigStruct object]

        """

        self.input_type = 'ff'

        self.dir_path = dir_path
        self.config = config

        # This type of input should have the calstars file
        self.require_calstars = True

        print('Using FF files from:', self.dir_path)


        self.ff_list = []

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


        self.cache = {}



    def nextChunk(self):
        """ Go to the next FF file. """

        self.current_ff_index = (self.current_ff_index + 1)%len(self.ff_list)
        self.current_ff_file = self.ff_list[self.current_ff_index]



    def prevChunk(self):
        """ Go to the previous FF file. """

        self.current_ff_index = (self.current_ff_index - 1)%len(self.ff_list)
        self.current_ff_file = self.ff_list[self.current_ff_index]



    def load(self):
        """ Load the FF file. """

        # Load from cache to avoid recomputing
        if self.current_ff_file in self.cache:
            return self.cache[self.current_ff_file]

        # Load the FF file from disk
        ff = readFF(self.dir_path, self.current_ff_file)
                
        # Store the loaded file to cache for faster loading
        self.cache = {}
        self.cache[self.current_ff_file] = ff

        return ff


    def name(self):
        """ Return the name of the FF file. """

        return self.current_ff_file


    def currentTime(self):
        """ Return the time of the current image. """

        return getMiddleTimeFF(self.current_ff_file, self.config.fps, ret_milliseconds=True)



class FFMimickInterface(object):
    def __init__(self, maxpixel, avepixel, nrows, ncols):
        """ Object which mimicks the interface of an FF structure. """

        self.maxpixel = maxpixel
        self.avepixel = avepixel
        self.nrows = nrows
        self.ncols = ncols



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

        self.dir_path = dir_path
        self.config = config

        # This type of input probably won't have any calstars files
        self.require_calstars = False


        _, file_name = os.path.split(self.dir_path)

        # Remove the file extension
        file_name = ".".join(file_name.split('.')[:-1])

        if beginning_time is None:
            
            try:
                # Try reading the beginning time of the video from the name if time is not given
                self.beginning_datetime = datetime.datetime.strptime(file_name, "%Y%m%d_%H%M%S.%f")

            except:
                messagebox.showerror('Input error', 'The time of the beginning cannot be read from the file name! Either change the name of the file to be in the YYYYMMDD_hhmmss format, or specify the beginning time using the -t option.')
                sys.exit()

        else:
            self.beginning_datetime = beginning_time



        print('Using video file:', self.dir_path)

        # Open the video file
        self.cap = cv2.VideoCapture(self.dir_path)

        self.current_frame_chunk = 0

        # Prop values: https://stackoverflow.com/questions/11420748/setting-camera-parameters-in-opencv-python

        # Get the FPS
        self.fps = self.cap.get(5)

        # Get the total time number of video frames in the file
        self.total_frames = self.cap.get(7)

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


        self.cache = {}


    def nextChunk(self):
        """ Go to the next frame chunk. """

        self.current_frame_chunk += 1
        self.current_frame_chunk = self.current_frame_chunk%self.total_fr_chunks


    def prevChunk(self):
        """ Go to the previous frame chunk. """

        self.current_frame_chunk -= 1
        self.current_frame_chunk = self.current_frame_chunk%self.total_fr_chunks


    def load(self):
        """ Load the frame chunk file. """

        # First try to load the frame form cache, if available
        if self.current_frame_chunk in self.cache:
            return self.cache[self.current_frame_chunk]

        frames = np.empty(shape=(self.fr_chunk_no, self.nrows, self.ncols), dtype=np.uint8)

        # Set the first frame location
        self.cap.set(1, self.current_frame_chunk*self.fr_chunk_no)

        # Load the chunk of frames
        for i in range(self.fr_chunk_no):

            ret, frame = self.cap.read()

            # If the end of the video files was reached, stop the loop
            if frame is None:
                break

            # Convert frame to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frames[i] = frame.astype(np.uint8)

        # Crop the frame number to total size
        frames = frames[:i]

        self.current_fr_chunk_size = i + 1


        # Compute the maxpixel and avepixel
        maxpixel = np.max(frames, axis=0).astype(np.uint8)
        avepixel = np.mean(frames, axis=0).astype(np.uint8)


        # Init the structure that mimicks the FF file structure
        ff_struct_fake = FFMimickInterface(maxpixel, avepixel, self.nrows, self.ncols)

        # Store the FF struct to cache to avoid recomputing
        self.cache = {}
        self.cache[self.current_frame_chunk] = ff_struct_fake

        return ff_struct_fake
        


    def name(self):
        """ Return the name of the chunk, which is just the time range. """

        year, month, day, hours, minutes, seconds, milliseconds = self.currentTime()
        microseconds = int(1000*milliseconds)

        return str(datetime.datetime(year, month, day, hours, minutes, seconds, microseconds))


    def currentTime(self):
        """ Return the mean time of the current image. """

        # Compute number of seconds since the beginning of the video file to the mean time of the frame chunk
        seconds_since_beginning = (self.current_frame_chunk*self.fr_chunk_no \
            + self.current_fr_chunk_size/2)/self.fps

        # Compute the absolute time
        dt = self.beginning_datetime + datetime.timedelta(seconds=seconds_since_beginning)

        return (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond/1000)




class InputTypeUWOVid(object):
    def __init__(self, dir_path, config):
        """ Input file type handle for UWO .vid files.
        
        Arguments:
            dir_path: [str] Path to the vid file.
            config: [ConfigStruct object]

        """

        self.input_type = 'vid'

        self.dir_path = dir_path
        self.config = config

        # This type of input probably won't have any calstars files
        self.require_calstars = False


        print('Using vid file:', self.dir_path)

        # Open the vid file
        self.vid = VidStruct()
        self.vid_file = open(self.dir_path, 'rb')

        # Read one video frame and rewind to beginning
        readVidFrame(self.vid, self.vid_file)
        self.vidinfo = copy.deepcopy(self.vid)
        self.vid_file.seek(0)
        
        # Try reading the beginning time of the video from the name
        self.beginning_datetime = unixTime2Date(self.vidinfo.ts, self.vidinfo.tu, dt_obj=True)


        self.current_frame_chunk = 0

        # Get the total time number of video frames in the file
        self.total_frames = os.path.getsize(self.dir_path)//self.vidinfo.seqlen

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

        # Do the initial load
        self.load()


    def nextChunk(self):
        """ Go to the next frame chunk. """

        self.current_frame_chunk += 1
        self.current_frame_chunk = self.current_frame_chunk%self.total_fr_chunks


    def prevChunk(self):
        """ Go to the previous frame chunk. """

        self.current_frame_chunk -= 1
        self.current_frame_chunk = self.current_frame_chunk%self.total_fr_chunks


    def load(self):
        """ Load the frame chunk file. """

        # First try to load the frame from cache, if available
        if self.current_frame_chunk in self.cache:
            frame, self.frame_chunk_unix_times = self.cache[self.current_frame_chunk]
            return frame


        # Set the vid file pointer to the right byte
        self.vid_file.seek(self.current_frame_chunk*self.fr_chunk_no*self.vidinfo.seqlen)

        # Init frame container
        frames = np.empty(shape=(self.fr_chunk_no, self.nrows, self.ncols), dtype=np.uint16)

        self.frame_chunk_unix_times = []

        # Load the chunk of frames
        for i in range(self.fr_chunk_no):

            frame = readVidFrame(self.vid, self.vid_file)

            # If the end of the vid file was reached, stop the loop
            if frame is None:
                break

            frames[i] = frame.astype(np.uint16)

            # Add the unix time to list
            self.frame_chunk_unix_times.append(self.vid.ts + self.vid.tu/1000000.0)


        # Crop the frame number to total size
        frames = frames[:i]

        # Compute the maxpixel and avepixel
        maxpixel = np.max(frames, axis=0).astype(np.uint16)
        avepixel = np.mean(frames, axis=0).astype(np.uint16)


        # Init the structure that mimicks the FF file structure
        ff_struct_fake = FFMimickInterface(maxpixel, avepixel, self.nrows, self.ncols)

        # Store the FF struct to cache to avoid recomputing
        self.cache = {}
        self.cache[self.current_frame_chunk] = [ff_struct_fake, self.frame_chunk_unix_times]

        return ff_struct_fake
        


    def name(self):
        """ Return the name of the chunk, which is just the time range. """

        year, month, day, hours, minutes, seconds, milliseconds = self.currentTime()
        microseconds = int(1000*milliseconds)

        return str(datetime.datetime(year, month, day, hours, minutes, seconds, microseconds))


    def currentTime(self):
        """ Return the mean time of the current image. """

        # Compute the mean UNIX time
        mean_utime = np.mean(self.frame_chunk_unix_times)

        mean_ts = int(mean_utime)
        mean_tu = int((mean_utime - mean_ts)*1000000)

        return unixTime2Date(mean_ts, mean_tu)



def detectInputType(dir_path, config, beginning_time=None):
    """ Given the folder of a file, detect the input format.

    Arguments:
        dir_path: [str] Input directory path or file name (e.g. dir with FF files, or path to video file).
        config: [Config Struct]

    Keyword arguments:
        beginning_time: [datetime] Datetime of the video beginning. Optional, only can be given for
            video input formats.

    """
    

    # If the given dir path is a directory
    if os.path.isdir(dir_path):

        # Check if there are valid FF names in the directory
        if any([validFFName(ff_file) for ff_file in os.listdir(dir_path)]):

            # Init the image handle for FF files
            img_handle = InputTypeFF(dir_path, config)

        # If not, check if there any image files in the folder
        else:
            ### PLACEHOLDER !!!
            return None


    # Use the given video file
    else:

        # Check if the given file is a video file
        if dir_path.endswith('.mp4') or dir_path.endswith('.avi') or dir_path.endswith('.mkv'):

            # Init the image hadle for video files
            img_handle = InputTypeVideo(dir_path, config, beginning_time=beginning_time)


        # Check if the given files is the UWO .vid format
        elif dir_path.endswith('.vid'):
            
            # Init the image handle for UWO-type .vid files
            img_handle = InputTypeUWOVid(dir_path, config)


        else:
            messagebox.showerror(title='Input format error', message='Only these video formats are supported: .mp4, .avi, .mkv, .vid!')
            return None


    return img_handle