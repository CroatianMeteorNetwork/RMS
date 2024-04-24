""" Module which provides a common functional interface for loading video frames/images from different
    input data formats. """

from __future__ import print_function, division, absolute_import

import os
import sys
import copy
import datetime


# Rawpy for DFN images
try:
    import rawpy
except ImportError:
    pass


import cv2
import numpy as np
from astropy.io import fits

from RMS.Astrometry.Conversions import unixTime2Date, datetime2UnixTime
from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import reconstructFrame as reconstructFrameFF
from RMS.Formats.FFfile import validFFName, filenameToDatetime
from RMS.Formats.FFfile import getMiddleTimeFF, selectFFFrames
from RMS.Formats.FRbin import read as readFR, validFRName
from RMS.Formats.Vid import readFrame as readVidFrame
from RMS.Formats.Vid import VidStruct
from RMS.GeoidHeightEGM96 import wgs84toMSLHeight
from RMS.Routines import Image


# Try importaing a Qt message box if available
try:
    from RMS.Routines.CustomPyqtgraphClasses import qmessagebox as messagebox
except:

    # Otherwise import a tk message box
    # tkinter import that works on both Python 2 and 3
    try:
        from tkinter import messagebox
    except:
        import tkMessageBox as messagebox


# Import cython functions
import pyximport
pyximport.install(setup_args={'include_dirs': [np.get_include()]})
from RMS.Routines.DynamicFTPCompressionCy import FFMimickInterface


# ConstantsO
UWO_MAGICK_CAMO = 1144018537
UWO_MAGICK_EMCCD = 1141003881
UWO_MAGICK_ASGARD = 38037846


def getCacheID(first_frame, size):
    """ Get the frame chunk ID. """

    return "first:{:d},size:{:d}".format(int(first_frame), int(size))


def computeFramesToRead(read_nframes, total_frames, fr_chunk_no, first_frame):
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
        if first_frame + frames_to_read > total_frames:
            frames_to_read = total_frames - first_frame

    return int(frames_to_read)


class InputType(object):
    def __init__(self):
        """ Template class for all input types. """

        self.current_frame = 0
        self.total_frames = 1

        # Only used for image mode
        self.single_image_mode = False

    def nextChunk(self):
        pass

    def prevChunk(self):
        pass

    def loadChunk(self, first_frame=None, read_nframes=None):
        pass

    def name(self, beginning=False):
        pass

    def currentTime(self, dt_obj=False):
        pass

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
        pass

    def getSequenceNumber(self):
        """ Returns the frame sequence number for the current frame.

        Return:
            [int] Frame sequence number.
        """

        return self.current_frame

    def currentFrameTime(self, frame_no=None, dt_obj=False):
        pass



class InputTypeFRFF(InputType):
    def __init__(self, dir_path, config, single_ff=False, use_fr_files=False):
        """ Input file type handle for FF files.

        Arguments:
            dir_path: [str] Path to directory with FF files.
            config: [ConfigStruct object]

        Keyword arguments:
            single_ff: [bool] If True, a single FF file should be given as input, and not a directory with FF
                files. False by default.
            detection: [bool] Indicates that the input is used for detection. False by default. Has no effect
                for FF files.
            use_fr_files: [bool] Include FR files together with FF files. False by default, only used in SkyFit.

        """

        self.input_type = 'ff'

        self.dir_path = dir_path
        self.config = config

        self.use_fr_files = use_fr_files

        self.__nrows = None
        self.__ncols = None

        self.single_ff = single_ff

        # This type of input should have the calstars file
        self.require_calstars = False

        # Don't byteswap the images
        self.byteswap = False

        if self.single_ff:
            print('Using file:', self.dir_path)
        else:
            if use_fr_files:
                print('Using FF and/or FR files from:', self.dir_path)
            else:
                print('Using FF files from:', self.dir_path)


        # List of FF and FR file names
        self.ff_list = []  
        self.current_ff_index = 0
        self.ff = None

        # Add the single FF file to the list
        if self.single_ff:
            self.dir_path, file_name = os.path.split(self.dir_path)
            self.ff_list.append(file_name)

        else:

            # Get a list of FF files in the folder
            for file_name in os.listdir(dir_path):
                if validFFName(file_name):
                    self.ff_list.append(file_name)

                # Add FR files to the list only if they are used
                if self.use_fr_files:
                    if validFRName(file_name):
                        self.ff_list.append(file_name)


        # Check that there are any FF files in the folder
        if not self.ff_list:
            messagebox(title='File list warning', message='No FF files in the selected folder!')

            sys.exit()

        # Sort the FF list
        self.ff_list = sorted(self.ff_list, key=lambda x: x[2:] + x[:2])

        self.current_frame_list = [None]*len(self.ff_list)  # each file has their own current frame
        self.line_list = [0]*len(self.ff_list)  # the line stored for each file (0 to self.line_numer[i])
        self.line_number = [1]*len(self.ff_list)  # number of lines for each file

        # Number for frames to read by default
        self.total_frames = 256

        # Cahcne for whole FF files
        self.cache = {}

        # Cache for individual frames
        self.cache_frames = {}

        # Load the first chunk for initing parameters
        self.loadChunk()

        self.fps = self.config.fps

    @property
    def current_line(self):
        return self.line_list[self.current_ff_index]

    @current_line.setter
    def current_line(self, line):
        self.line_list[self.current_ff_index] = line

    @property
    def nrows(self):
        return self.__nrows

    @nrows.setter
    def nrows(self, nrows):
        self.__nrows = nrows
        for file in self.cache.keys():
            if self.cache[file].nrows is None:
                self.cache[file].nrows = nrows

    @property
    def ncols(self):
        return self.__ncols

    @ncols.setter
    def ncols(self, ncols):
        self.__ncols = ncols
        for file in self.cache.keys():
            if self.cache[file].ncols is None:
                self.cache[file].ncols = ncols

    @property
    def beginning_datetime(self):
        return filenameToDatetime(self.name())

    def nextChunk(self):
        """ Go to the next FF file. """

        self.current_ff_index = (self.current_ff_index + 1)%len(self.ff_list)

    def prevChunk(self):
        """ Go to the previous FF file. """

        self.current_ff_index = (self.current_ff_index - 1)%len(self.ff_list)


    def loadChunk(self, first_frame=None, read_nframes=None):
        """ Load the frame chunk file.

        Keyword arguments:
            first_frame: [int] First frame to read.
            read_nframes: [int] Number of frames to read. If not given (None), self.fr_chunk_no frames will be
                read. If -1, all frames will be read in.
        """

        # Load pure FF or FR files
        if (first_frame is None) and (read_nframes is None):
            
            # Find which file read
            file_name = self.name()

            # Save and load file from cache
            if file_name in self.cache:
                ff = self.cache[file_name]

            elif validFFName(file_name):
                
                # Load the FF file from disk
                ff = readFF(self.dir_path, file_name)

                # Put the FF into separate cache
                self.cache[file_name] = ff

            else:
                # Load the FR files from disk
                ff = readFR(self.dir_path, file_name)
                ff.nrows = self.nrows
                ff.ncols = self.ncols
                self.cache[file_name] = ff

                self.line_number[self.current_ff_index] = ff.lines

            # when calling loadChunk on an image never called before, set the current frame to the start
            # if it's an FR file, otherwise set it to 0
            if self.current_frame is None:
                if validFFName(file_name):
                    self.current_frame = 0
                else:
                    self.current_frame = ff.t[self.current_line][0]


        # If it contains at least one FF file
        elif any([validFFName(ff_file) for ff_file in self.ff_list]):

            if first_frame == -1:
                first_frame = 0

            total_ff_frames = len([x for x in self.ff_list if validFFName(x)])*256
            frames_to_read = computeFramesToRead(read_nframes, total_ff_frames, 256, first_frame)
            ffs_to_read = self.ff_list[first_frame//256:(first_frame + frames_to_read)//256 + 1]

            # If there is only one FF to read, reconstruct given frames
            if len(ffs_to_read) == 1:
                file_name = ffs_to_read[0]

                # Compute the range of frames to read
                min_frame = first_frame%256
                max_frame = (first_frame + frames_to_read)%256

                # Read the FF file
                ff = readFF(self.dir_path, file_name)

                # Select the frames
                ff.maxpixel = selectFFFrames(ff.maxpixel, ff, min_frame, max_frame)

            # If there are more FFs to read, make a fake FF
            else:

                # Init an empty FF structure
                ff = FFMimickInterface(self.nrows, self.ncols, np.uint8)

                # Store maxpixel selections, avepixels, stdpixels
                maxpixel_list = []
                avepixel_list = []
                stdpixel_list = []

                # Read the FF files that have to read and reconstruct the frames
                for i, file_name in enumerate(ffs_to_read):
                    # Compute the range of frames to read
                    min_frame = 0
                    max_frame = 255

                    if i == 0:
                        min_frame = first_frame%256

                    elif i == len(ffs_to_read) - 1:
                        max_frame = (first_frame + frames_to_read)%256

                    # Read the FF file
                    ff_temp = readFF(self.dir_path, file_name)

                    # Reconstruct the maxpixel in the given frame range
                    maxpixel = selectFFFrames(ff_temp.maxpixel, ff_temp, min_frame, max_frame)

                    # Reconstruct the avepixel in the given frame range
                    avepixel = selectFFFrames(ff_temp.avepixel, ff_temp, min_frame, max_frame)

                    # Store the computed frames
                    maxpixel_list.append(maxpixel)
                    avepixel_list.append(avepixel)
                    stdpixel_list.append(ff_temp.stdpixel)

                # Immidiately extract the appropriate frames
                if len(maxpixel_list) == 1:

                    ff.maxpixel = maxpixel_list[0]
                    ff.avepixel = avepixel_list[0]
                    ff.stdpixel = stdpixel_list[0]

                # Otherwise, compute the combined FF
                else:
                    maxpixel_list = np.array(maxpixel_list)
                    avepixel_list = np.array(avepixel_list)
                    stdpixel_list = np.array(stdpixel_list)

                    ff.maxpixel = np.max(maxpixel_list, axis=0)

                    # The maximum of the avepixel is taken because only the frame range of avepixel is taken
                    ff.avepixel = np.max(avepixel_list, axis=0)

                    ff.stdpixel = np.max(stdpixel_list, axis=0)

        # If there are only FR files
        else:  

            if first_frame == -1:
                first_frame = 0


            # Load the given FR file
            ff = readFR(self.dir_path, self.ff_list[self.current_ff_index]) 
            ff.nrows = self.nrows
            ff.ncols = self.ncols

            self.line_number[self.current_ff_index] = ff.lines

            fr_files = [ff]
            fr_file_frames = [fr.frameNum for fr in fr_files]  # number of frames in each fr file
            total_frames = sum(sum(x) for x in fr_file_frames)
            frames_to_read = computeFramesToRead(read_nframes, total_frames, 256, first_frame)

            frame_list = []
            img_count = np.full((self.ncols, self.nrows), -1, dtype=np.float64)
            stop = False

            # Go through every line in every FR file
            for fr, line_list in enumerate(fr_file_frames):
                for line, frame_count_line in enumerate(line_list):
                    # don't do anything until you get to the first frame
                    if first_frame > 0:
                        first_frame -= frame_count_line

                    if first_frame <= 0 < frames_to_read:
                        min_frame = (frame_count_line - abs(first_frame))%frame_count_line
                        max_frame = min(frame_count_line, min_frame + frames_to_read)

                        # Create a fake FF using FR frames
                        for i in range(min_frame, max_frame + 1):

                            # Init an empty image
                            img = np.zeros((self.ncols, self.nrows), float)

                            # Compute indices on the image where the FR file will be pasted
                            x_img = np.arange(int(fr_files[fr].xc[line][i] - fr_files[fr].size[line][i]//2),
                                              int(fr_files[fr].xc[line][i] + fr_files[fr].size[line][i]//2))
                            y_img = np.arange(int(fr_files[fr].yc[line][i] - fr_files[fr].size[line][i]//2),
                                              int(fr_files[fr].yc[line][i] + fr_files[fr].size[line][i]//2))
                            X_img, Y_img = np.meshgrid(x_img, y_img)

                            # Compute FR frame coordiantes
                            y_frame = np.arange(len(y_img))
                            x_frame = np.arange(len(x_img))
                            Y_frame, X_frame = np.meshgrid(y_frame, x_frame)

                            # Paste frame onto the image
                            img[X_img, Y_img] = fr_files[fr].frames[line][i][Y_frame, X_frame]
                            img_count[X_img, Y_img] += 1

                            frame_list.append(img)

                        first_frame = 0
                        frames_to_read -= frame_count_line

                    if frames_to_read <= 0:
                        stop = True
                        break
                if stop:
                    break

            frame_list = np.array(frame_list)

            # calculate maxpixel
            ff.maxpixel = np.swapaxes(np.max(frame_list, axis=0), 0, 1).astype(np.uint8)

            # calculate avepixel
            img_count[img_count <= 0] = 1
            img = np.sum(frame_list, axis=0)
            ff.avepixel = np.swapaxes(img/img_count, 0, 1).astype(np.uint8)

            ff.stdpixel = np.zeros_like(ff.avepixel)

        
        self.ff = ff

        # Set the fixed dtype of uint8 to the FF
        if self.ff is not None:
            self.ff.dtype = np.uint8
                
        # Store the loaded file to cache for faster loading (always just have a cache of 1)
        self.cache = {}
        self.cache[file_name] = self.ff

        return ff
    

    def setCurrentFF(self, ff_name):
        """ Set the current FF file. """

        if ff_name in self.ff_list:
            self.current_ff_index = self.ff_list.index(ff_name)

            # Load the chunk
            self.loadChunk()

    @property
    def current_ff_file(self):
        return self.name()

    @property
    def current_frame(self):
        return self.current_frame_list[self.current_ff_index]

    @current_frame.setter
    def current_frame(self, frame):
        self.current_frame_list[self.current_ff_index] = frame

    def name(self, beginning=None):
        """ Return the name of the FF file. """

        return self.ff_list[self.current_ff_index]

    def currentTime(self, dt_obj=False):
        """ Return the middle time of the current image. """

        if dt_obj:
            return datetime.datetime(*getMiddleTimeFF(self.name(), self.fps, ret_milliseconds=False))

        else:
            return getMiddleTimeFF(self.name(), self.fps, ret_milliseconds=True)

    def nextLine(self):
        self.current_line = (self.current_line + 1)%self.line_number[self.current_ff_index]

    def prevLine(self):
        self.current_line = (self.current_line - 1)%self.line_number[self.current_ff_index]

    def nextFrame(self):
        self.current_frame = self.current_frame + 1

        # Increment FR line index
        if self.current_frame >= self.total_frames:
            self.nextLine()

        self.current_frame %= self.total_frames

    def prevFrame(self):
        self.current_frame = self.current_frame - 1

        # Decrement FR line index
        if self.current_frame < 0:
            self.prevLine()

        self.current_frame %= self.total_frames

    def setFrame(self, fr_num):

        # Increment/decrement line number in the FR file
        if fr_num > self.total_frames:
            self.nextLine()
        elif fr_num < 0:
            self.prevLine()

        self.current_frame = fr_num%self.total_frames

    def loadFrame(self, avepixel=False):
        """ Load the current frame. """

        # Find which file read
        file_name = self.name()

        # Save and load file from cache
        if file_name in self.cache:
            ff_frame = self.cache[file_name]

        elif validFFName(file_name):

            # Load the FF file from disk
            ff_frame = readFF(self.dir_path, file_name)

            # Put the FF into separate cache
            self.cache[file_name] = ff_frame

        # Read the FR file from disk
        else:
            ff_frame = readFR(self.dir_path, file_name)
            ff_frame.nrows = self.nrows
            ff_frame.ncols = self.ncols
            self.cache[file_name] = ff_frame

            # If there is a corresponding FF file in the folder, reconstruct the frame from it



        if self.current_frame is None:
            if validFFName(file_name):
                self.current_frame = 0
            else:
                self.current_frame = ff_frame.t[self.current_line][0]


        # Get frame from file
        if validFFName(file_name):
            
            # Reconstruct the frame from an FF file
            frame = reconstructFrameFF(ff_frame, self.current_frame%self.total_frames, \
                                       avepixel=avepixel)
        else:

            # For FR files

            # If there is no underlying FF file, make a blank background
            if self.nrows is None or self.nrows is None:
                
                # Get the maximum extent of the meteor frames
                y_size = max(
                    max(np.array(ff_frame.yc[i]) + np.array(ff_frame.size[i])//2) for i in range(ff_frame.lines))
                x_size = max(
                    max(np.array(ff_frame.xc[i]) + np.array(ff_frame.size[i])//2) for i in range(ff_frame.lines))

                # Make the image square
                img_size = max(y_size, x_size)

                frame = np.zeros((img_size, img_size), np.uint8)
            else:
                frame = np.zeros((self.nrows, self.ncols), np.uint8)

            # Compute the index of the frame in the FR bin structure
            frame_indx = int(self.current_frame) - ff_frame.t[self.current_line][0]

            # Reconstruct the frame if it is within the bounds
            if (frame_indx < ff_frame.frameNum[self.current_line]) and (frame_indx >= 0):

                # Get the center position of the detection on the current frame
                yc = ff_frame.yc[self.current_line][frame_indx]
                xc = ff_frame.xc[self.current_line][frame_indx]

                # Get the size of the window
                size = ff_frame.size[self.current_line][frame_indx]

                # Paste the frames onto the big image
                y_img = np.arange(int(yc - size//2), int(yc + size//2))
                x_img = np.arange(int(xc - size//2), int(xc + size//2))

                Y_img, X_img = np.meshgrid(y_img, x_img)


                y_frame = np.arange(len(y_img))
                x_frame = np.arange(len(x_img))

                Y_frame, X_frame = np.meshgrid(y_frame, x_frame)

                frame[Y_img, X_img] = ff_frame.frames[self.current_line][frame_indx][Y_frame, X_frame]

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


class InputTypeVideo(InputType):
    def __init__(self, dir_path, config, beginning_time=None, detection=False):
        """ Input file type handle for video files.
        
        Arguments:
            dir_path: [str] Path to the video file.
            config: [ConfigStruct object]

        Keyword arguments:
            beginning_time: [datetime] datetime of the beginning of the video. Optional, None by default.
            detection: [bool] Indicates that the input is used for detection. False by default. This will
                control whether the binning is applied or not.

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

            # Try reading the beginning time of the video from the name if time is not given
            try:
                self.beginning_datetime = datetime.datetime.strptime(file_name_noext, "%Y%m%d_%H%M%S.%f")
            
            except ValueError:

                try:
                    self.beginning_datetime = datetime.datetime.strptime(file_name_noext, "%Y%m%d_%H%M%S")

                except:
                    messagebox(title="Input error", \
                    message="The time of the beginning cannot be read from the file name! Either change the name of the file to be in the YYYYMMDD_hhmmss format, or specify the beginning time using command line options.")

                    sys.exit()

        else:
            self.beginning_datetime = beginning_time

        self.detection = detection

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

        # Apply the binning if the detection is used
        if self.detection:
            self.nrows = self.nrows//self.config.detection_binning_factor
            self.ncols = self.ncols//self.config.detection_binning_factor

        print('FPS from video:', self.fps)
        print('Total frames:', self.total_frames)

        # Set the number of frames to be used for averaging and maxpixels
        self.fr_chunk_no = 256

        # Compute the number of frame chunks
        self.total_fr_chunks = self.total_frames//self.fr_chunk_no
        if self.total_fr_chunks == 0:
            self.total_fr_chunks = 1

        self.current_fr_chunk_size = self.fr_chunk_no

        self.current_frame = 0

        self.cache = {}

        # Load the initial chunk
        self.loadChunk()

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
        frames_to_read = computeFramesToRead(read_nframes, self.total_frames, self.fr_chunk_no, first_frame)

        # Get the cache ID
        cache_id = getCacheID(first_frame, frames_to_read)

        # Check if this chunk has been cached
        if cache_id in self.cache:
            frame, self.current_fr_chunk_size = self.cache[cache_id]
            return frame

        # Init making the FF structure
        ff_struct_fake = FFMimickInterface(self.nrows, self.ncols, np.uint8)

        # If there are no frames to read, return an empty array
        if frames_to_read == 0 or frames_to_read == -1:
            print('There are no frames to read!')
            return ff_struct_fake

        print('Frames to read: ' + str(frames_to_read))

        # Load the chunk of frames
        for i in range(frames_to_read):

            ret, frame = self.cap.read()

            # If the end of the video files was reached, stop the loop
            if frame is None:
                break

            # Convert frame to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Bin the frame
            if self.detection and (self.config.detection_binning_factor > 1):
                frame = Image.binImage(frame, self.config.detection_binning_factor,
                                       self.config.detection_binning_method)

            # Add frame for FF processing
            ff_struct_fake.addFrame(frame.astype(np.uint16))

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
        seconds_since_beginning = (self.current_frame_chunk*self.fr_chunk_no
                                   + self.current_fr_chunk_size/2)/self.fps

        # Compute the absolute time
        dt = self.beginning_datetime + datetime.timedelta(seconds=seconds_since_beginning)

        if dt_obj:
            return dt

        else:
            return dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond/1000

    def loadFrame(self, avepixel=False):
        """ Load the current frame. """

        # Set the frame location
        self.cap.set(1, self.current_frame)

        # Read the frame
        ret, frame = self.cap.read()

        # Convert frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Bin the frame
        if self.detection and (self.config.detection_binning_factor > 1):
            frame = Image.binImage(frame, self.config.detection_binning_factor,
                                   self.config.detection_binning_method)

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
            return dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond/1000


class InputTypeUWOVid(InputType):
    def __init__(self, file_path, config, detection=False):
        """ Input file type handle for UWO .vid files.
        
        Arguments:
            file_path: [str] Path to the vid file.
            config: [ConfigStruct object]

        Keyword arguments:
            detection: [bool] Indicates that the input is used for detection. False by default. This will
                control whether the binning is applied or not.

        """
        self.input_type = 'vid'

        # Separate directory path and file name
        self.vid_path = file_path
        self.dir_path, vid_file = os.path.split(file_path)

        self.config = config

        self.detection = detection

        self.ff = None

        # This type of input probably won't have any calstars files
        self.require_calstars = False

        # Byteswap the images
        self.byteswap = True

        print('Using vid file:', self.vid_path)

        # Open the vid file
        self.vid = VidStruct()
        self.vid_file = open(self.vid_path, 'rb', buffering=65536)

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

        # Apply the binning if the detection is used
        if self.detection:
            self.nrows = self.nrows//self.config.detection_binning_factor
            self.ncols = self.ncols//self.config.detection_binning_factor

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
        frames_to_read = computeFramesToRead(read_nframes, self.total_frames, self.fr_chunk_no, first_frame)

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

            try:
                frame = readVidFrame(self.vid, self.vid_file)
            except:
                frame = None

            # If the end of the vid file was reached, stop the loop
            if frame is None:
                break

            frame = frame.astype(np.uint16)

            # Bin the frame
            if self.detection and (self.config.detection_binning_factor > 1):
                frame = Image.binImage(frame, self.config.detection_binning_factor,
                                       self.config.detection_binning_method)

            unix_time = self.vid.ts + self.vid.tu/1000000.0

            # Add the unix time to list
            self.frame_chunk_unix_times.append(unix_time)

            # Add frame for FF processing (the frame should already be uint16)
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

    def loadFrame(self, avepixel=False):
        """ Load the current frame. """

        # Set the vid file pointer to the right byte
        self.vid_file.seek(self.current_frame*self.vidinfo.seqlen)

        # Load a frame
        frame = readVidFrame(self.vid, self.vid_file)

        # Bin the frame
        if self.detection and (self.config.detection_binning_factor > 1):
            frame = Image.binImage(frame, self.config.detection_binning_factor, \
                                   self.config.detection_binning_method)

        # Save the frame time
        self.current_frame_time = unixTime2Date(self.vid.ts, self.vid.tu, dt_obj=True)

        unix_time_lst = (self.vid.ts, self.vid.tu)
        if unix_time_lst not in self.utime_frame_dict:
            self.utime_frame_dict[self.current_frame] = unix_time_lst

        return frame

    def getSequenceNumber(self):
        """ Returns the frame sequence number for the current frame. For vid files this is the frame number
            since the beginning of the recording.

        Return:
            [int] Frame sequence number.
        """

        return self.vid.seq

    def currentFrameTime(self, frame_no=None, dt_obj=False):
        """ Return the time of the frame. """

        if frame_no is None:

            # Load the frame time if it wasn't loaded yet
            if not hasattr(self, 'current_frame_time'):
                self.loadFrame()
            
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
    def __init__(self, dir_path, config, beginning_time=None, fps=None, detection=False):
        """ Input file type handle for a folder with images.

        Arguments:
            dir_path: [str] Path to the vid file.
            config: [ConfigStruct object]
        Keyword arguments:
            beginning_time: [datetime] datetime of the beginning of the video. Optional, None by default.
            fps: [float] Known FPS of the images. None by default, in which case it will be read from the
                config file.
            detection: [bool] Indicates that the input is used for detection. False by default. This will
                control whether the binning is applied or not.
        """

        self.input_type = 'images'

        self.dir_path = dir_path
        self.config = config

        self.detection = detection

        self.ff = None
        self.cache = {}

        self.dt_frame_time = None
        self.frame_dt_list = None

        # This type of input probably won't have any calstars files
        self.require_calstars = False

        # Disctionary which holds the time of every frame, used for fast frame time lookup
        self.frame_dt_dict = {}

        self.fripon_mode = False
        self.fripon_header = None
        self.cabernet_status = False

        img_types = ['.png', '.jpg', '.bmp', '.fit', '.fits', '.tif']

        # Add raw formats if rawpy is installed
        if 'rawpy' in sys.modules:
            img_types += ['.nef', '.cr2']

        self.img_list = []

        ### Find images in the given folder ###
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
            messagebox(title='Input error',
            message="Can't find any images in the given directory! Only PNG, JPG and BMP are supported!")
            sys.exit()

        ### ###

        ### Filter the file list so only the images with most extension types are used ###

        extensions = [entry.split('.')[-1] for entry in self.img_list]
        unique = np.unique(extensions, return_counts=True)
        most_common_extension = unique[0][np.argmax(unique[1])].lower()

        self.img_list = [entry for entry in self.img_list if entry.lower().endswith(most_common_extension)]

        ###


        # Compute the total number of used frames
        self.total_frames = len(self.img_list)



        self.single_image_mode = False

        # If there is only one image, enable the single image mode
        if self.total_frames == 1:

            self.single_image_mode = True

            print()
            print("Single image mode")



        ### Try to detect if the given images are UWO-style PNGs ###

        self.uwo_png_mode = False

        # Load the first image
        img = self.loadFrame(fr_no=0)

        # Get the magick type
        self.uwo_magick_type = self.getUWOMagickType(img)

        if self.uwo_magick_type is not None:

            self.uwo_png_mode = True

            # Get the beginning time
            self.loadFrame(fr_no=0)
            beginning_time = self.dt_frame_time

            print('UWO PNG mode')

        ###

        # Decide if images need to be byteswapped
        if self.uwo_png_mode:
            self.byteswap = True

        else:
            self.byteswap = False


        # If the resolution differs from the one in the config file, change it and write out a warning
        if (img.shape[0] != self.config.height) or (img.shape[1] != self.config.width):
            self.config.height = img.shape[0]
            self.config.width = img.shape[1]
            print()
            print("WARNING! The image resolution differs from the resolution set in the config file.")
            print("Image resolution set to {:d} x {:d} px".format(self.config.width, self.config.height))
            


        # If during the frame loading it was deterined that the images are in the FRIPON format
        if self.fripon_mode:

            ### Sort the frames according to the fits header time ###
            
            frame_time_list = []
            for i in range(self.total_frames):
                self.loadFrame(fr_no=i)
                frame_time_list.append(self.dt_frame_time)

            # Sort the image list according to the frame time
            self.img_list = [x for _, x in sorted(zip(frame_time_list, self.img_list))]

            # Load the first frame to get the beginning time
            self.loadFrame(fr_no=0)

            ### ###

            # Set the begin time if in the FRIPON mode
            beginning_time = self.dt_frame_time

            # Load info for CABERNET
            if self.cabernet_status:

                # Try to get the station ID if present
                if "SITE" in self.fripon_header:
                    self.config.stationID = self.fripon_header["SITE"].strip("'").strip()

                else:

                    # Find comment line with station name
                    station_comment = [line for line in self.fripon_header["COMMENT"] if "CABERNET at " in line]

                    if len(station_comment):
                        station_id = " ".join(station_comment[0].split()[2:]).strip("'").strip()
                    else:
                        station_id = "CABERNET-STAT"

                    self.config.stationID = station_id

                self.config.latitude = np.degrees(self.fripon_header["LATITUDE"])
                self.config.longitude = np.degrees(self.fripon_header["LONGITUD"])
                self.config.elevation = wgs84toMSLHeight(np.radians(self.config.latitude), 
                    np.radians(self.config.longitude), self.fripon_header["ALTITUDE"], self.config) # WGS84 in fits

                # Set approximate FOV
                self.config.fov_w = 40
                self.config.fov_h = 27

                # Set magnitude limit
                self.config.catalog_mag_limit = 6.0


            # Load info for FRIPON all-sky cameras
            else:

                # Set station parameters if in the FRIPON mode
                self.config.stationID = self.fripon_header["TELESCOP"].strip()
                self.config.latitude = self.fripon_header["SITELAT"]
                self.config.longitude = self.fripon_header["SITELONG"]
                self.config.elevation = self.fripon_header["SITEELEV"] # MSL

                # Set the catalog to BSC5
                self.config.star_catalog_path = os.path.join(self.config.rms_root_dir, "Catalogs")
                self.config.star_catalog_file = "BSC5"

                # Set approximate FOV
                self.config.fov_h = 180
                self.config.fov_w = 200

                # Set magnitude limit
                self.config.catalog_mag_limit = 3.5



            self.config.width = self.fripon_header["NAXIS1"]
            self.config.height = self.fripon_header["NAXIS2"]
            self.config.fps = self.fps

            # Global shutter
            self.config.deinterlace_order = -2

            



        # Check if the beginning time was given (it will be read from the PNG if the UWO format is given)
        if beginning_time is None:

            try:
                # Try reading the beginning time of the video from the name if time is not given
                self.beginning_datetime = datetime.datetime.strptime(os.path.basename(self.dir_path), \
                                                                     "%Y%m%d_%H%M%S.%f")

            except:
                messagebox(title='Input error',
                message='The time of the beginning cannot be read from the file name! Either change the name of the file to be in the YYYYMMDD_hhmmss format, or specify the beginning time using command line options.')
                sys.exit()

        else:
            self.beginning_datetime = beginning_time



        print('Using folder:', self.dir_path)

        self.current_frame_chunk = 0


        self.current_frame = 0
        self.current_img_file = self.img_list[self.current_frame]



        if self.single_image_mode:

            # Start at frame 100 to accomodate reversing picks, set the max number of frames to 1024
            self.current_frame = 100
            self.total_frames = 1024


        # Load the first image
        img = self.loadFrame()

        # Get the image size (the binning correction doesn't have to be applied because the image is already
        #   binned)
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
            self.current_fr_chunk_size = self.fr_chunk_no = self.total_frames


        # Do the initial load
        self.loadChunk()



        ### SET THE FPS ###

        # Estimate the FPS if UWO pngs are given
        if self.uwo_png_mode and not self.single_image_mode:

            # Convert datetimes to Unix times
            unix_times = [datetime2UnixTime(dt) for dt in self.frame_dt_list]

            fps = 1/((unix_times[-1] - unix_times[0])/self.current_fr_chunk_size)


        # If FPS is not given, use one from the config file
        if fps is None:

            # Don't use the config file if FRIPON files are used
            if not self.fripon_mode:
                self.fps = self.config.fps
                print('Using FPS from config file: ', self.fps)

        else:

            self.fps = fps
            print('Using FPS:', self.fps)

        ### ###


        if self.single_image_mode:

            # Correct the time so that the given time starts on frame 100
            self.beginning_datetime -= datetime.timedelta(seconds=self.current_frame/self.fps)

            print("Beginning time is now relative to frame 100!")
            print()


        print('Total frames:', self.total_frames)


    def nextChunk(self):
        """ Go to the next frame chunk. """

        if not self.single_image_mode:

            self.current_frame_chunk += 1
            self.current_frame_chunk = self.current_frame_chunk%self.total_fr_chunks

            self.current_frame = self.current_frame_chunk*self.fr_chunk_no


    def prevChunk(self):
        """ Go to the previous frame chunk. """

        if not self.single_image_mode:

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



        # In the single image mode, only read one frame
        if self.single_image_mode:

            frames_to_read = 1

        else:

            # Compute the number of frames to read
            frames_to_read = computeFramesToRead(read_nframes, self.total_frames, self.fr_chunk_no, \
                first_frame)


        # Get the cache ID
        cache_id = getCacheID(first_frame, frames_to_read)

        # Check if this chunk has been cached
        if cache_id in self.cache:
            frame, self.frame_dt_list, self.current_fr_chunk_size = self.cache[cache_id]
            return frame

        # Init making the FF structure
        ff_struct_fake = FFMimickInterface(self.nrows, self.ncols, self.img_dtype)

        self.frame_dt_list = []

        # Load the chunk of frames
        for i in range(frames_to_read):

            # Compute the image index
            img_indx = first_frame + i

            # Stop the loop if the ends of images has been reached
            if img_indx >= self.total_frames:
                break

            # Load the image
            frame = self.loadFrame(fr_no=img_indx)

            # Add frame for FF processing
            ff_struct_fake.addFrame(frame.astype(np.uint16))

            # Add the datetime of the frame to list of the UWO png is used
            if self.uwo_png_mode or self.fripon_mode:
                self.frame_dt_list.append(self.currentFrameTime(frame_no=img_indx, dt_obj=True))

        self.current_fr_chunk_size = i

        # Finish making the fake FF file
        ff_struct_fake.finish()

        # Store the FF struct to cache to avoid recomputing
        self.cache = {}

        self.cache[cache_id] = [ff_struct_fake, self.frame_dt_list, self.current_fr_chunk_size]

        # Set the computed chunk as the current FF
        self.ff = ff_struct_fake

        return ff_struct_fake


    def nextFrame(self):
        """ Increment current frame. """

        self.current_frame = (self.current_frame + 1)%self.total_frames


        # In the single image mode, continously cycle through the same frame
        if self.single_image_mode:
            pass
        else:
            self.current_img_file = self.img_list[self.current_frame]


    def prevFrame(self):
        """ Increment current frame. """

        self.current_frame = (self.current_frame - 1)%self.total_frames

        # In the single image mode, continously cycle through the same frame
        if self.single_image_mode:
            pass
        else:
            self.current_img_file = self.img_list[self.current_frame]


    def setFrame(self, fr_num):
        """ Set the current frame.

        Arguments:
            fr_num: [float] Frame number to set.
        """

        self.current_frame = fr_num%self.total_frames


        # In the single image mode, don't change the image
        if self.single_image_mode:
            pass

        else:
            self.current_img_file = self.img_list[self.current_frame]


    def loadFrame(self, avepixel=None, fr_no=None):
        """ Loads the current frame.

        Keyword arguments:
            avepixel: [bool] Does nothing, just for function interface consistency with other input types.
            fr_no: [int] Load a specific frame. None by default, then the current frame will be loaded.
        """

        # If a special frame number was given, use that one
        if fr_no is not None:
            current_img_file = self.img_list[fr_no]

        else:
            current_img_file = self.current_img_file
            fr_no = self.current_frame


        # In the single image mode (but not for UWO), the frame will not change, so load it from the cache 
        #   if available
        single_image_key = "single_image"
        if self.single_image_mode and (not self.uwo_png_mode) and (not self.fripon_mode):
            if single_image_key in self.cache:
                
                # Load the frame from cache
                frame = self.cache[single_image_key]

                return frame



        # Load an .NEF file
        if current_img_file.lower().endswith('.nef') or current_img_file.lower().endswith('.cr2'):
            
            # .nef files will not be brought here if rawpy is not installed

            # Load the raw image
            frame = Image.loadRaw(os.path.join(self.dir_path, current_img_file))


        # Load a FRIPON fit file
        if current_img_file.lower().endswith('.fit'):

            # Load the data from a fits file
            with open(os.path.join(self.dir_path, current_img_file), 'rb') as f:

                # Open the image data
                fits_file = fits.open(f)
                frame = fits_file[0].data

                # Flip image vertically for FRIPON (not CABERNET)
                if not self.cabernet_status:
                    frame = np.flipud(frame)

                # Load the header
                head = fits_file[0].header

                # Save the FRIPON header
                self.fripon_header = head

                # Load the frame time
                timestamp_stripped = head["DATE-OBS"].strip("=").strip("'").strip()
                self.dt_frame_time = datetime.datetime.strptime(timestamp_stripped, "%Y-%m-%dT%H:%M:%S.%f")

                # If CABERNET is used, set a fixed FPS
                if "COMMENT" in head:
                    self.cabernet_status = bool(len([line for line in head["COMMENT"] if "CABERNET" in line]))
                else:
                    self.cabernet_status = False

                if self.cabernet_status:
                    self.fps = 95.129375951

                # If not, read the FPS from the header
                else:
                    self.fps = 1.0/head["EXPOSURE"]


                # Indicate that a FRIPON fit file is read
                self.fripon_mode = True

        # Loads a non-FRIPON FITS image
        if current_img_file.lower().endswith('.fits'):
            
            # Load the data from a fits file
            with open(os.path.join(self.dir_path, current_img_file), 'rb') as f:

                # Open the image data
                fits_file = fits.open(f)
                frame = fits_file[0].data

                # # Flip image vertically
                # frame = np.flipud(frame)

        # Load a normal image
        else:

            # Get the current image if it's not an NEF file (e.g. png, jpg...)
            frame = cv2.imread(os.path.join(self.dir_path, current_img_file), -1)

        
        # Convert the image to black and white if it's 8 bit and has colors
        if (8*frame.itemsize == 8) and (frame.ndim == 3):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        # If UWO PNG's are used, byteswap the image and read the image time
        if self.uwo_png_mode:

            # Read time from an 8-bit image
            if (8*frame.itemsize == 8):


                # Read the time from an ASGARD image
                if frame[0][3] == 2:
                    ts = (frame[0][15] << 24) + (frame[0][14] << 16) + (frame[0][13] << 8) + frame[0][12]
                    tu = (frame[0][19] << 24) + (frame[0][18] << 16) + (frame[0][17] << 8) + frame[0][16]

                # Read the time from an ASGARD mancut image
                else:
                    ts = (frame[0][23] << 24) + (frame[0][22] << 16) + (frame[0][21] << 8) + frame[0][20]
                    tu = (frame[0][27] << 24) + (frame[0][26] << 16) + (frame[0][25] << 8) + frame[0][24]

            else:
                
                # Byteswap if it's the UWO style 16-bit png
                frame = frame.byteswap()

                if self.uwo_magick_type == "emccd":
                    # Read the time from the image for 16 bit images
                    ts = frame[0][6] + (frame[0][7] << 16)
                    tu = frame[0][8] + (frame[0][9] << 16)

                else:

                    # Read the time from the image for 16 bit images
                    ts = frame[0][10] + (frame[0][11] << 16)
                    tu = frame[0][12] + (frame[0][13] << 16)


            frame_dt = unixTime2Date(ts, tu, dt_obj=True)

            self.dt_frame_time = frame_dt


        if self.uwo_png_mode or self.fripon_mode:

            # Save the frame time of the current frame
            if fr_no not in self.frame_dt_dict:
                self.frame_dt_dict[fr_no] = self.dt_frame_time

        # Bin the frame
        if self.detection and (self.config.detection_binning_factor > 1):
            frame = Image.binImage(frame, self.config.detection_binning_factor,
                                   self.config.detection_binning_method)


        # In the single image mode, store the frame to memory so it doesn't have to be reloaded
        if self.single_image_mode:
            self.cache[single_image_key] = frame


        return frame


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

        if self.uwo_png_mode or self.fripon_mode:

            # Convert datetimes to Unix times
            unix_times = [datetime2UnixTime(dt) for dt in self.frame_dt_list]

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

    def currentFrameTime(self, frame_no=None, dt_obj=False):
        """ Return the time of the frame. """

        if frame_no is None:
            frame_no = self.current_frame

        # If the UWO png or FRIPON fit is used, return the time read from the file
        if (self.uwo_png_mode or self.fripon_mode) and (not self.cabernet_status):

            # If the frame number is not given, return the time of the current frame
            if frame_no is None:

                dt = self.dt_frame_time


            # Otherwise, load the frame time
            else:

                # If the frame number is not in the dictionary, load the frame and read the time from it
                if frame_no not in self.frame_dt_dict:
                    current_frame_backup = self.current_frame

                    # Load the time from the given frame
                    self.loadFrame(fr_no=frame_no)

                    # Load back the current frame
                    self.loadFrame(fr_no=current_frame_backup)

                # Read the frame time from the dictionary
                dt = self.frame_dt_dict[frame_no]


        else:

            # Compute the datetime of the current frame
            dt = self.beginning_datetime + datetime.timedelta(seconds=frame_no/self.fps)

        if dt_obj:
            return dt

        else:
            return (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond/1000)
        

    def getUWOMagickType(self, img):
        """ Return the type of the UWO PNG image. """

        # Read in the magick number as a uint32
        magicknum = np.frombuffer(img[0], dtype=np.uint32)[0]

        # Define the magick numbers for different UWO PNGs
        uwo_magick_num_list = [UWO_MAGICK_CAMO, UWO_MAGICK_EMCCD, UWO_MAGICK_ASGARD]

        # Check the magick number for UWO PNGs
        if magicknum in uwo_magick_num_list:

            # Set the magick type
            if magicknum == UWO_MAGICK_CAMO:
                uwo_magick_type = "camo"
            elif magicknum == UWO_MAGICK_EMCCD:
                uwo_magick_type = "emccd"
            elif magicknum == UWO_MAGICK_ASGARD:
                uwo_magick_type = "asgard"

            return uwo_magick_type

        else:
            return None


class InputTypeDFN(InputType):
    def __init__(self, file_path, config, beginning_time=None, fps=None):
        """ Input file type handle for a folder with images.

        Arguments:
            dir_path: [str] Path to the vid file.
            config: [ConfigStruct object]

        Keyword arguments:
            beginning_time: [datetime] datetime of the beginning of the video. Optional, None by default.
            fps: [float] Known FPS of the images. None by default, in which case it will be read from the
                config file.
            detection: [bool] Indicates that the input is used for detection. False by default. This will
                control whether the binning is applied or not.

        """
        self.input_type = 'dfn'

        self.dir_path, self.image_file = os.path.split(file_path)
        self.config = config

        self.byteswap = False

        # Set the frames to a global shutter, so no correction is applied
        self.config.deinterlace_order = -2

        # This type of input probably won't have any calstars files
        self.require_calstars = False

        if 'rawpy' in sys.modules:
            ### Find images in the given folder ###
            img_types = ['.png', '.jpg', '.bmp', '.tif', '.fits', '.nef', '.cr2']
        else:
            img_types = ['.png', '.jpg', '.bmp', '.tif', '.fits']

        self.beginning_datetime = beginning_time

        # Check if the file ends with support file extensions
        if self.beginning_datetime is None and \
                any([self.image_file.lower().endswith(fextens) for fextens in img_types]):
            try:
                
                # Extract the DFN timestamp from the file name
                image_filename_split = self.image_file.split("_")
                date_str = image_filename_split[1]
                time_str = image_filename_split[2]
                datetime_str = date_str + "_" + time_str
                
                beginning_datetime = datetime.datetime.strptime(
                    datetime_str,
                    "%Y-%m-%d_%H%M%S")

                self.beginning_datetime = beginning_datetime

            except:
                messagebox(title='Input error', \
                    message="Can't parse given DFN file name!")
                sys.exit()

        print('Using folder:', self.dir_path)


        # DFN frames start at 100 to accomodate picking previous frames, and 1024 picks total are allowed
        self.current_frame = 100
        self.total_frames = 1024

        self.ff = None

        # Load the first image
        img = self.loadImage()

        # Get the image size (the binning correction doesn't have to be applied because the image is already
        #   binned)
        self.nrows = img.shape[0]
        self.ncols = img.shape[1]
        self.img_dtype = img.dtype

        if self.nrows > self.ncols:
            temp = self.nrows
            self.nrows = self.ncols
            self.ncols = temp
            img = np.rot90(img)

        self.ff = FFMimickInterface(self.nrows, self.ncols, self.img_dtype)
        self.ff.addFrame(img.astype(np.uint16))
        self.ff.finish()

        # If FPS is not given, use one from the config file
        if fps is None:
            self.fps = self.config.fps
            print('Using FPS from config file: ', self.fps)

        else:
            self.fps = fps
            print('Using FPS:', self.fps)


    def loadImage(self):

        # Load the NEF file
        if self.image_file.lower().endswith('.nef') or self.image_file.lower().endswith('.cr2'):
            
            # .nef files will not be brought here if rawpy is not installed
            # get raw data from .nef file and get image from it
            raw = rawpy.imread(os.path.join(self.dir_path, self.image_file))
            frame = raw.postprocess(gamma=(1,1), output_bps=16, no_auto_bright=True, no_auto_scale=True, \
                output_color=rawpy.ColorSpace.sRGB)

            # Convert the image to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            
        else:
            # Get the current image
            frame = cv2.imread(os.path.join(self.dir_path, self.image_file), -1)

        # Convert the image to black and white if it's 8 bit
        if 8*frame.itemsize == 8:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame

    def loadChunk(self, first_frame=None, read_nframes=None):
        """ Load the frame chunk file.

        Keyword arguments:
            first_frame: [int] First frame to read.
            read_nframes: [int] Number of frames to read. If not given (None), self.fr_chunk_no frames will be
                read. If -1, all frames will be read in.
        """
        return self.ff

    def name(self, beginning=False):
        return self.image_file

    def currentTime(self, dt_obj=False):
        # Compute the datetime of the current frame
        dt = self.beginning_datetime + datetime.timedelta(seconds=self.total_frames/self.fps/2)

        if dt_obj:
            return dt

        else:
            return dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond/1000

    def currentFrameTime(self, frame_no=None, dt_obj=False):
        if frame_no is None:
            frame_no = self.current_frame

        # Compute the datetime of the current frame
        dt = self.beginning_datetime + datetime.timedelta(seconds=frame_no/self.fps)

        if dt_obj:
            return dt

        else:
            return dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond/1000


def detectInputType(input_path, config, beginning_time=None, fps=None, skip_ff_dir=False, detection=False,
    use_fr_files=False):
    """ Given the folder of a file, detect the input format.

    Arguments:
        input_path: [str] Input directory path (e.g. dir with FF files or path to a video file).
        config: [Config Struct]

    Keyword arguments:
        beginning_time: [datetime] Datetime of the video beginning. Optional, only can be given for
            video input formats.
        fps: [float] Frames per second, used only when images in a folder are used. If it's not given,
            it will be read from the config file.
        skip_ff_dir: [bool] Skip the input type where there are multiple FFs in the same directory. False
            by default. This is only used for ManualReduction.
        detection: [bool] Indicates that the input is used for detection. False by default. This will
                control whether the binning is applied or not. No effect on FF image handle.
        use_fr_files: [bool] Include FR files together with FF files. False by default, only used in SkyFit.

    """
    

    if os.path.isdir(input_path):

        # Detect input type if a directory is given
        img_handle = detectInputTypeFolder(input_path, config, beginning_time=beginning_time, fps=fps, \
            skip_ff_dir=skip_ff_dir, detection=detection, use_fr_files=use_fr_files)
        
    else:
        # Detect input type if a path to a file is given
        img_handle = detectInputTypeFile(input_path, config, beginning_time=beginning_time, fps=fps, \
            detection=fps)

    return img_handle


def detectInputTypeFolder(input_dir, config, beginning_time=None, fps=None, skip_ff_dir=False, \
    detection=False, use_fr_files=False):
    """ Given the folder of a file, detect the input format.

    Arguments:
        input_path: [str] Input directory path (e.g. dir with FF files).
        config: [Config Struct]

    Keyword arguments:
        beginning_time: [datetime] Datetime of the video beginning. Optional, only can be given for
            when images in a directory as used.
        fps: [float] Frames per second, used only when images in a folder are used. If it's not given,
            it will be read from the config file.
        skip_ff_dir: [bool] Skip the input type where there are multiple FFs in the same directory. False
            by default. This is only used for ManualReduction.
        detection: [bool] Indicates that the input is used for detection. False by default. This will
                control whether the binning is applied or not. No effect on FF image handle.
        use_fr_files: [bool] Include FR files together with FF files. False by default, only used in SkyFit.

    """

    ### Find images in the given folder ###
    img_types = ['.png', '.jpg', '.bmp', '.fit', '.tif', '.fits']

    if 'rawpy' in sys.modules:
        img_types += ['.nef', '.cr2']
        

    img_handle = None
    
    # If the given dir path is a directory, search for FF files or individual images
    if not os.path.isdir(input_dir):
        return None


    # Check if there are valid FF names in the directory
    if any([validFFName(ff_file) or validFRName(ff_file) for ff_file in os.listdir(input_dir)]):


        # If FR files are not used, only check for FF files
        if not use_fr_files:
            if not any([validFFName(ff_file) for ff_file in os.listdir(input_dir)]):
                print("No FF files found in directory!")
                return None


        if skip_ff_dir:
            messagebox(title='FF directory',
            message='ManualReduction only works on individual FF files, and not directories with FF files!')
            return None

        else:
            # Init the image handle for FF files in a directory
            img_handle = InputTypeFRFF(input_dir, config, use_fr_files=use_fr_files)
            img_handle.ncols = config.width
            img_handle.nrows = config.height

    elif any([any(file.lower().endswith(x) for x in img_types) for file in os.listdir(input_dir)]) and \
            config.width != 4912 and config.width != 7360:
        img_handle = InputTypeImages(input_dir, config, beginning_time=beginning_time, fps=fps,
                                     detection=detection)

    return img_handle



def detectInputTypeFile(input_file, config, beginning_time=None, fps=None, detection=False):
    """ Given a file, detect the input format.

    Arguments:
        input_path: [str] Input file path (e.g. path to a video file).
        config: [Config Struct]

    Keyword arguments:
        beginning_time: [datetime] Datetime of the video beginning. Optional, only can be given for
            video input formats.
        fps: [float] Frames per second, used only for a DFN image. If it's not given, it will be read from 
            the config file.
        detection: [bool] Indicates that the input is used for detection. False by default. This will
                control whether the binning is applied or not. No effect on FF image handle.

    """

    # If the given path is a file, look for a single FF file, video files, or vid files
    dir_path, file_name = os.path.split(input_file)

    # Check if a single FF file was given
    if validFFName(file_name) or validFRName(file_name):

        # Init the image handle for FF a single FF file
        img_handle = InputTypeFRFF(input_file, config, single_ff=True)
        img_handle.ncols = config.width
        img_handle.nrows = config.height

    # Check if the given file is a video file
    elif file_name.lower().endswith('.mp4') or file_name.lower().endswith('.avi') \
            or file_name.lower().endswith('.mkv') or file_name.lower().endswith('.wmv') \
            or file_name.lower().endswith('.mov'):

        # Init the image hadle for video files
        img_handle = InputTypeVideo(input_file, config, beginning_time=beginning_time,
                                    detection=detection)

    # Check if the given files is the UWO .vid format
    elif file_name.endswith('.vid'):

        # Init the image handle for UWO-type .vid files
        img_handle = InputTypeUWOVid(input_file, config, detection=detection)

    elif config.width == 4912 or config.width == 7360:
        img_handle = InputTypeDFN(input_file, config, beginning_time=beginning_time, fps=fps)

    else:
        messagebox(title="Input format error",
                             message="Couldn\'t find the file type given")
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
        ff.addFrame(frame.astype(np.uint16))

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
