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
iamport cv2
from RMS import MorphologicalOperations as morph

class ff_struct:
    """ Default structure for a FF*.bin file.
    """
    def __init__(self):
        self.nrows = 0
        self.ncols = 0
        self.nbits = 0
        self.first = 0
        self.camno = 0
        self.maxpixel = 0
        self.maxframe = 0
        self.avepixel = 0
        self.stdpixel = 0
        
def readFF(filename):
    """Function for reading FF bin files.
    Returns a structure that allows access to individual parameters of the image
    e.g. print readFF("FF300_20140802_205545_600_0090624.bin").nrows to print out the number of rows
    e.g. print readFF("FF300_20140802_205545_600_0090624.bin").maxpixel to print out the array of nrows*ncols numbers which represent the image
    INPUTS:
        filename: file name from the file to be read
    """

    fid = open(filename, 'rb')
    ff = ff_struct()
    ff.nrows = np.fromfile(fid, dtype=np.uint32, count = 1)
    ff.ncols = np.fromfile(fid, dtype=np.uint32, count = 1)
    ff.nbits = np.fromfile(fid, dtype=np.uint32, count = 1)
    ff.first = np.fromfile(fid, dtype=np.uint32, count = 1)
    ff.camno = np.fromfile(fid, dtype=np.uint32, count = 1)

    N = ff.nrows * ff.ncols

    ff.maxpixel = np.reshape (np.fromfile(fid, dtype=np.uint8, count = N), (ff.nrows, ff.ncols))
    ff.maxframe = np.reshape (np.fromfile(fid, dtype=np.uint8, count = N), (ff.nrows, ff.ncols))
    ff.avepixel = np.reshape (np.fromfile(fid, dtype=np.uint8, count = N), (ff.nrows, ff.ncols))
    ff.stdpixel = np.reshape (np.fromfile(fid, dtype=np.uint8, count = N), (ff.nrows, ff.ncols))

    return ff

def treshold(window, ff):
    return window >= (ff.avepixel + k1 * ff.stdpixel + j1)

def reconstructWindows(filename):    
    ff = readFF(filename)
    
    for i in range(0, 256/time_slide-1):
        indices = np.where((ff.maxframe >= i*time_slide) & (ff.maxframe < i*time_slide+time_window_size))
    
        img = np.zeros((ff.nrows, ff.ncols))
        img[indices] = ff.maxpixel[indices]
        
        cv2.imshow(str(i*time_slide) + "-" + str(i*time_slide+time_window_size) + " raw", img.astype(np.uint8)*255)
        cv2.waitKey(0)
        
        img = treshold(img, ff)
        
        cv2.imshow(str(i*time_slide) + "-" + str(i*time_slide+time_window_size) + " treshold", img.astype(np.uint8)*255)
        cv2.waitKey(0)
        
        img = morph.clean(img)
        
        cv2.imshow(str(i*time_slide) + "-" + str(i*time_slide+time_window_size) + " clean", img.astype(np.uint8)*255)
        cv2.waitKey(0)
        
        img = morph.bridge(img)
        
        cv2.imshow(str(i*time_slide) + "-" + str(i*time_slide+time_window_size) + " bridge", img.astype(np.uint8)*255)
        cv2.waitKey(0)
        
        img = morph.close(img)
        
        cv2.imshow(str(i*time_slide) + "-" + str(i*time_slide+time_window_size) + " close", img.astype(np.uint8)*255)
        cv2.waitKey(0)
    
        img = morph.skeleton(img)
        
        cv2.imshow(str(i*time_slide) + "-" + str(i*time_slide+time_window_size) + " thin", img.astype(np.uint8)*255)
        cv2.waitKey(0)
        
        img = morph.repeat(morph.spur, img, 2)
        
        cv2.imshow(str(i*time_slide) + "-" + str(i*time_slide+time_window_size) + " spur", img.astype(np.uint8)*255)
        cv2.waitKey(0)
        
        img = morph.clean(img)
        
        cv2.imshow(str(i*time_slide) + "-" + str(i*time_slide+time_window_size) + " clean", img.astype(np.uint8)*255)
        cv2.waitKey(0)

if __name__ == "__main__":
    time_window_size = 64
    time_slide = 32
    k1 = 1.5
    j1 = 9
    
    reconstructWindows("FF453_20150106_031308_605_0989952.bin")