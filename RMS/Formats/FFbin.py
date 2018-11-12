# RPi Meteor Station
# Copyright (C) 2016  Dario Zubovic, Denis Vida
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
import re
import struct

import numpy as np

from RMS.Formats.FFStruct import FFStruct


# FFbin handling stolen from FF_bin_suite.py from CMN_binViewer written by Denis Vida


def read(directory, filename, array=False, full_filename=False):
    """ Read FF*.bin file from the specified directory.
    
    Arguments:
        directory: [str] Path to directory containing file
        filename: [str] Name of FF*.bin file (either with FF and extension or without)

    Keyword arguments:
        array: [ndarray] True in order to populate structure's array element (default is False)
        full_filename: [bool] True if full file name is given explicitly, a name which may differ from the
            usual FF*.fits format. False by default.
    
    Return:
        [ff structure]

    """
    
    if (filename.startswith("FF") and ('.bin' in filename)) or full_filename:
        fid = open(os.path.join(directory, filename), "rb")
    else:
        fid = open(os.path.join(directory, "FF" + filename + ".bin"), "rb")

    ff = FFStruct()
    
    # Check if it is the new of the old CAMS data format
    version_flag = int(np.fromfile(fid, dtype=np.int32, count = 1))

    # Old format
    if version_flag > 0:

        ff.nrows = version_flag
        ff.ncols = int(np.fromfile(fid, dtype=np.uint32, count = 1))
        ff.nbits = int(np.fromfile(fid, dtype=np.uint32, count = 1))
        ff.nframes = 2**ff.nbits
        ff.first = int(np.fromfile(fid, dtype=np.uint32, count = 1))
        ff.camno = int(np.fromfile(fid, dtype=np.uint32, count = 1))

        ff.decimation_fact = 1

        

    # New format
    elif version_flag == -1:

        ff.nrows = int(np.fromfile(fid, dtype=np.uint32, count = 1))
        ff.ncols = int(np.fromfile(fid, dtype=np.uint32, count = 1))

        ff.nframes = int(np.fromfile(fid, dtype=np.uint32, count = 1))
        ff.first = int(np.fromfile(fid, dtype=np.uint32, count = 1))
        ff.camno = int(np.fromfile(fid, dtype=np.uint32, count = 1))

        ff.decimation_fact = int(np.fromfile(fid, dtype=np.uint32, count = 1))

        ff.interleave_flag = int(np.fromfile(fid, dtype=np.uint32, count = 1))

        ff.fps = float(np.fromfile(fid, dtype=np.uint32, count = 1))/1000

    
    if array:
        N = 4*ff.nrows*ff.ncols
    
        ff.array = np.reshape(np.fromfile(fid, dtype=np.uint8, count=N), (4, ff.nrows, ff.ncols))
        
    else:
        N = ff.nrows*ff.ncols
    
        ff.maxpixel = np.reshape(np.fromfile(fid, dtype=np.uint8, count=N), (ff.nrows, ff.ncols))
        ff.maxframe = np.reshape(np.fromfile(fid, dtype=np.uint8, count=N), (ff.nrows, ff.ncols))
        ff.avepixel = np.reshape(np.fromfile(fid, dtype=np.uint8, count=N), (ff.nrows, ff.ncols))
        ff.stdpixel = np.reshape(np.fromfile(fid, dtype=np.uint8, count=N), (ff.nrows, ff.ncols))

    return ff



def write(ff, directory, filename, version=2):
    """ Write FF structure to a .bin file in the specified directory.
    
    Arguments:
        ff: [ff bin struct] FF bin file loaded in the FF bin structure
        directory: [str] path to the directory where the file will be written
        filename: [str] name of the file which will be written

    Keyword arguments:
        version: [int] CAMS bin format version. Old version is 1, the new version is 2 (defualt).
    
    Return:
        None
    """
    
    if filename[:2] == "FF":
        image = os.path.join(directory, filename)
    else:
        image = os.path.join(directory, "FF" + filename + ".bin")
        
    
    with open(image, "wb") as fid:
        if ff.array is not None:
            arr = ff.array
        else:
            arr = np.empty((4, ff.nrows, ff.ncols), np.uint8)
            arr[0] = ff.maxpixel
            arr[1] = ff.maxframe
            arr[2] = ff.avepixel
            arr[3] = ff.stdpixel
        
        # Extract only the number from the camera code
        camno_num = int(re.findall('\d+', str(ff.camno))[0])


        # Write the old version
        if version == 1:

            fid.write(struct.pack('I', ff.nrows))
            fid.write(struct.pack('I', ff.ncols))
            fid.write(struct.pack('I', ff.nbits))
            fid.write(struct.pack('I', ff.first))
            fid.write(struct.pack('I', camno_num))


        # Write the new version
        else:

            # Write new version indicator
            fid.write(struct.pack('i', -1))

            fid.write(struct.pack('I', ff.nrows))
            fid.write(struct.pack('I', ff.ncols))

            fid.write(struct.pack('I', ff.nframes))
            fid.write(struct.pack('I', ff.first))
            fid.write(struct.pack('I', camno_num))

            fid.write(struct.pack('I', ff.decimation_fact))

            fid.write(struct.pack('I', ff.interleave_flag))

            if ff.fps < 0:
                ff.fps = 25

            # Store FPS as unsigned integer, 1000*fps
            fid.write(struct.pack('I', int(1000*ff.fps)))


        # Write image arrays
        arr.tofile(fid)
        