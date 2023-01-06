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

from __future__ import print_function, division, absolute_import, unicode_literals

import os
import numpy as np
import struct


class fr_struct:
    def __init__(self):
        """ Default structure for a FR*.bin file. This file holds raw frame cutouts of fireball detections,
            with metadata necessary to reconstruct the original fireball video.
        """

        # Number of lines (i.e. detections) in the file
        self.lines = 0

        ### The lists below hold values for every line.
        ### E.g. a list of frame numbers for line (with index 0) can be retrieved as self.t[0]

        # Total number of frames for every line. E.g. for line 0, the total number of frames in the line
        #   can be retriever with self.frameNum[0]
        self.frameNum = []

        # Y coordinates of the centre of the cutout on the full image
        #   This is used to correctly place the cutout on the FF file
        self.yc = []

        # X coordinates of the centre of the cutout on the full image
        self.xc = []

        # Frame indices (as referece to the FF file) of cutouts for every line
        self.t = []

        # The width and height of every cutout (the cutouts are square, so only one size is saved)
        self.size = []

        # Image data for the cutouts. If you want to retrieve the cutout image for the first line and the 
        #   second frame, call: self.frames[0][1]
        self.frames = []

        ### ###

        self.nrows = None
        self.ncols = None
        self.__maxpixel = None
        self.__avepixel = None

        self.dtype = np.uint8


    @property
    def nframes(self):
        return 256

    @property
    def maxpixel(self):
        """ Construct a maxpixel from an FR file. """

        assert self.nrows is not None and self.ncols is not None

        if self.__maxpixel is None:

            # Init an empty image
            img = np.zeros((self.ncols, self.nrows), float)

            # Crease a maxpixel using all lines
            for line in range(self.lines):

                for i in range(self.frameNum[line]):

                    # Compute indices on the image where the FR file will be pasted
                    x_img = np.arange(int(self.xc[line][i] - self.size[line][i]//2),
                                      int(self.xc[line][i] + self.size[line][i]//2))
                    y_img = np.arange(int(self.yc[line][i] - self.size[line][i]//2),
                                      int(self.yc[line][i] + self.size[line][i]//2))
                    X_img, Y_img = np.meshgrid(x_img, y_img)

                    # Compute FR frame coordiantes
                    y_frame = np.arange(len(y_img))
                    x_frame = np.arange(len(x_img))
                    Y_frame, X_frame = np.meshgrid(y_frame, x_frame)

                    # Paste frame onto the image (take only max values)
                    img[X_img, Y_img] = np.maximum(img[X_img, Y_img], self.frames[line][i][X_frame, Y_frame])


            self.__maxpixel = np.swapaxes(img, 0, 1).astype(self.dtype)

        return self.__maxpixel

    @maxpixel.setter
    def maxpixel(self, maxpixel):
        self.__maxpixel = maxpixel


    @property
    def avepixel(self):
        """ Construct an avepixel from an FR file. """

        assert self.nrows is not None and self.ncols is not None

        if self.__avepixel is None:

            # Init an empty image
            img = np.zeros((self.ncols, self.nrows), np.float64)
            img_count = np.full((self.ncols, self.nrows), -1, dtype=np.float64)

            for line in range(self.lines):
                for i in range(self.frameNum[line]):

                    # Compute indices on the image where the FR file will be pasted
                    x_img = np.arange(int(self.xc[line][i] - self.size[line][i]//2),
                                      int(self.xc[line][i] + self.size[line][i]//2))
                    y_img = np.arange(int(self.yc[line][i] - self.size[line][i]//2),
                                      int(self.yc[line][i] + self.size[line][i]//2))
                    X_img, Y_img = np.meshgrid(x_img, y_img)

                    # Compute FR frame coordiantes
                    y_frame = np.arange(len(y_img))
                    x_frame = np.arange(len(x_img))
                    Y_frame, X_frame = np.meshgrid(y_frame, x_frame)

                    # Add values to correct positions on the image
                    img[X_img, Y_img] += self.frames[line][i][X_frame, Y_frame]
                    img_count[X_img, Y_img] += 1

            img_count[img_count <= 0] = 1
            img_count = np.swapaxes(img_count, 0, 1)
            img = np.swapaxes(img, 0, 1)

            self.__avepixel = ((img - self.maxpixel)/img_count).astype(self.dtype)

        return self.__avepixel


    @avepixel.setter
    def avepixel(self, avepixel):
        self.__avepixel = avepixel



def read(dir_path, filename):
    """ Read an FR*.bin file.
    
    Arguments:
        dir_path: [str] Path to directory containing file.
        filename: [str] Name of FR*.bin file (either with the FR prefix and the .bin suffix, or without).
    
    Return:
        fr: [fr_struct instance] 

    """
    if filename[:2] == "FR":
        fid = open(os.path.join(dir_path, filename), "rb")
    else:
        fid = open(os.path.join(dir_path, "FR_" + filename + ".bin"), "rb")

    fr = fr_struct()

    fr.lines = np.fromfile(fid, dtype=np.uint32, count=1)[0]

    for i in range(fr.lines):
        frameNum = np.fromfile(fid, dtype=np.uint32, count=1)[0]
        yc = []
        xc = []
        t = []
        size = []
        frames = []

        for z in range(frameNum):
            yc.append(int(np.fromfile(fid, dtype=np.uint32, count=1)))
            xc.append(int(np.fromfile(fid, dtype=np.uint32, count=1)))
            t.append(int(np.fromfile(fid, dtype=np.uint32, count=1)))
            size.append(int(np.fromfile(fid, dtype=np.uint32, count=1)))
            frames.append(np.reshape(np.fromfile(fid, dtype=np.uint8, count=size[-1]**2), (size[-1], size[-1])))

        fr.frameNum.append(frameNum)
        fr.yc.append(yc)
        fr.xc.append(xc)
        fr.t.append(t)
        fr.size.append(size)
        fr.frames.append(frames)

    return fr


def write(fr, dir_path, filename):
    """ Write FR*.bin structure to a file in specified directory.
    """

    if filename[:2] == "FR":
        file = os.path.join(dir_path, filename)
    else:
        file = os.path.join(dir_path, "FR_" + filename + ".bin")

    file = os.path.join()
    with open(file, "wb") as fid:
        fid.write(struct.pack('I', fr.lines))

        for i in range(fr.lines):
            fid.write(struct.pack('I', fr.frameNum[i]))

            for z in range(fr.frameNum[i]):
                fid.write(struct.pack('I', fr.yc[i, z]))
                fid.write(struct.pack('I', fr.xc[i, z]))
                fid.write(struct.pack('I', fr.t[i, z]))
                fid.write(struct.pack('I', fr.size[i, z]))
                fr.frames[i, z].tofile(fid)


def writeArray(arr, dir_path, filename):
    """ Write array with extracted clips to a file in specified directory.
    """

    if filename[:2] == "FR":
        file = os.path.join(dir_path, filename)
    else:
        file = os.path.join(dir_path, "FR_" + filename + ".bin")

    with open(file, "wb") as f:
        f.write(struct.pack('I', len(arr)))  # number of extracted lines

        for frames, sizepos in arr:
            f.write(struct.pack('I', len(frames)))  # number of extracted frames

            for i, frame in enumerate(frames):
                f.write(struct.pack('I', sizepos[i, 0]))  # y of center
                f.write(struct.pack('I', sizepos[i, 1]))  # x of center
                f.write(struct.pack('I', sizepos[i, 2]))  # time
                size = sizepos[i, 3]
                f.write(struct.pack('I', size))  # cropped frame size
                frame[:size, :size].tofile(f)  # cropped frame


def validFRName(fr_name):
    """ Checks if the given file is an FR file. 
    
    Arguments:
        fr_name: [str] Name of the FR file
    """

    if fr_name.startswith('FR') and fr_name.endswith('.bin'):
        return True

    else:
        return False
