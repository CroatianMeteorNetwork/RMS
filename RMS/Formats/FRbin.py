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
import numpy as np
import struct


class fr_struct:
    """ Default structure for a FR*.bin file.
    """
    
    def __init__(self):
        self.lines = 0
        self.frameNum = []
        self.yc = []
        self.xc = []
        self.t = []
        self.size = []
        self.frames = []
        


def read(dir, filename):
    """ Read FRF*.bin file from specified directory.
    
    @param dir: path to directory containing file
    @param filename: name of FR*.bin file (either with FR and extension or without)
    
    @return: fr structure
    """
    
    if filename[:2] == "FR":
        fid = open(dir + filename, "rb")
    else:
        fid = open(dir + "FRF" + filename + ".bin", "rb")
    
    fr = fr_struct()
    
    fr.lines = np.fromfile(fid, dtype=np.uint32, count = 1)
    
    for i in range(fr.lines):
        frameNum = np.fromfile(fid, dtype=np.uint32, count = 1)
        yc = []
        xc = []
        t = []
        size = []
        frames = []
        
        for z in range(frameNum):
            yc.append(int(np.fromfile(fid, dtype=np.uint32, count = 1)))
            xc.append(int(np.fromfile(fid, dtype=np.uint32, count = 1)))
            t.append(int(np.fromfile(fid, dtype=np.uint32, count = 1)))
            size.append(int(np.fromfile(fid, dtype=np.uint32, count = 1)))
            frames.append(np.reshape(np.fromfile(fid, dtype=np.uint8, count = size[-1]**2), (size[-1], size[-1])))
        
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
        file = os.path.join(dir_path, "FR" + filename + ".bin")
    
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
        file = os.path.join(dir_path, "FR" + filename + ".bin")
        
            
    with open(file, "wb") as f:
        f.write(struct.pack('I', len(arr)))               # number of extracted lines
        
        for frames, sizepos in arr:
            f.write(struct.pack('I', len(frames)))        # number of extracted frames
            
            for i, frame in enumerate(frames):
                f.write(struct.pack('I', sizepos[i, 0]))  # y of center
                f.write(struct.pack('I', sizepos[i, 1]))  # x of center
                f.write(struct.pack('I', sizepos[i, 2]))  # time
                size = sizepos[i, 3]
                f.write(struct.pack('I', size))           # cropped frame size
                frame[:size, :size].tofile(f)             # cropped frame
                