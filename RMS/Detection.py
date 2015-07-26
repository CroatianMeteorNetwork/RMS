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
import cv2
from RMS.Routines import MorphologicalOperations as morph
from time import time
import sys, os
from RMS.Formats import FFbin

def treshold(window, ff):
    return window > (ff.avepixel + k1 * ff.stdpixel + j1)

def show(name, img):
    cv2.imshow(name, img.astype(np.uint8)*255)
    cv2.moveWindow(name, 0, 0)
    cv2.waitKey(0)

def reconstructWindows(path, filename):    
    ff = FFbin.read(path, filename)
    
    for i in range(0, 256/time_slide-1):
        indices = np.where((ff.maxframe >= i*time_slide) & (ff.maxframe < i*time_slide+time_window_size))
    
        img = np.zeros((ff.nrows, ff.ncols))
        img[indices] = ff.maxpixel[indices]
        
        img = treshold(img, ff)
        
        show(filename + " " + str(i*time_slide) + "-" + str(i*time_slide+time_window_size) + " treshold", img)
        
        t = time()
        
        img = morph.clean(img)
        
        img = morph.bridge(img)
        
        img = morph.close(img)
        
        img = morph.repeat(morph.thin, img, None)
        
        img = morph.clean(img)
        
        print "time for morph:", time() - t
        
        show(filename + " " + str(i*time_slide) + "-" + str(i*time_slide+time_window_size) + " morph", img)

if __name__ == "__main__":
    time_window_size = 64
    time_slide = 32
    k1 = 1.5
    j1 = 9
    
    if len(sys.argv) == 1:
        print "Usage: python -m RMS.Detection /path/to/bin/files/"
        sys.exit()
    
    ff_list = [ff for ff in os.listdir(sys.argv[1]) if ff[0:2]=="FF" and ff[-3:]=="bin"]
    
    if(len(ff_list) == None):
        print "No files found!"
        sys.exit()
    
    for ff in ff_list:
        reconstructWindows(sys.argv[1], ff)