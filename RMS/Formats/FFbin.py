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

import numpy as np
import struct

# FFbin handling stolen from FF_bin_suite.py from CMN_binViewer written by Denis Vida

class ff_struct:
    """ Default structure for a FF*.bin file.
    """
    
    def __init__(self):
        self.nrows = 0
        self.ncols = 0
        self.nbits = 0
        self.first = 0
        self.camno = 0
        
        self.maxpixel = None
        self.maxframe = None
        self.avepixel = None
        self.stdpixel = None
        
        self.array = None
        
def read(dir, filename, array=False):
    """ Read FF*.bin file from specified directory.
    
    @param dir: path to directory containing file
    @param filename: name of FF*.bin file (either with FF and extension or without)
    @param array: True in order to populate structure's array element (default is False)
    @return: ff structure
    """
    
    if filename[:2] == "FF":
        fid = open(dir + filename, "rb")
    else:
        fid = open(dir + "FF" + filename + ".bin", "rb")

    ff = ff_struct()
    
    ff.nrows = np.fromfile(fid, dtype=np.uint32, count = 1)
    ff.ncols = np.fromfile(fid, dtype=np.uint32, count = 1)
    ff.nbits = np.fromfile(fid, dtype=np.uint32, count = 1)
    ff.first = np.fromfile(fid, dtype=np.uint32, count = 1)
    ff.camno = np.fromfile(fid, dtype=np.uint32, count = 1)
    
    if array:
        N = 4 * ff.nrows * ff.ncols
    
        ff.array = np.reshape(np.fromfile(fid, dtype=np.uint8, count = N), (4, ff.nrows, ff.ncols))
        
    else:
        N = ff.nrows * ff.ncols
    
        ff.maxpixel = np.reshape(np.fromfile(fid, dtype=np.uint8, count = N), (ff.nrows, ff.ncols))
        ff.maxframe = np.reshape(np.fromfile(fid, dtype=np.uint8, count = N), (ff.nrows, ff.ncols))
        ff.avepixel = np.reshape(np.fromfile(fid, dtype=np.uint8, count = N), (ff.nrows, ff.ncols))
        ff.stdpixel = np.reshape(np.fromfile(fid, dtype=np.uint8, count = N), (ff.nrows, ff.ncols))

    return ff

def write(ff, dir, filename):
    """ Write FF*.bin structure to a file in specified directory.
    """
    
    if filename[:2] == "FF":
        image = dir + filename, 'rb'
    else:
        image = dir + "FF" + filename + ".bin"
        
    print image
    
    with open(image, "wb") as fid:
        if ff.array is not None:
            arr = ff.array
        else:
            arr = np.empty((4, ff.nrows, ff.ncols), np.uint8)
            arr[0] = ff.maxpixel
            arr[1] = ff.maxframe
            arr[2] = ff.avepixel
            arr[3] = ff.stdpixel
        
        fid.write(struct.pack('I', ff.nrows))
        fid.write(struct.pack('I', ff.ncols))
        fid.write(struct.pack('I', ff.nbits))
        fid.write(struct.pack('I', ff.first))
        fid.write(struct.pack('I', ff.camno))
    
        arr.tofile(fid)