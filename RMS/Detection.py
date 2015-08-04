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
import numpy.ctypeslib as npct
import cv2
from RMS.Routines import MorphologicalOperations as morph
from time import time
import sys, os
from RMS.Formats import FFbin
import ctypes
import matplotlib.pyplot as plt

def treshold(window, ff):
    return window > (ff.avepixel + k1 * ff.stdpixel + j1)

def show(name, img):
    cv2.imshow(name, img.astype(np.uint8)*255)
    cv2.moveWindow(name, 0, 0)
    cv2.waitKey(0)
    
def reconstructFrame(ff, frameNum):
    img = np.zeros((ff.nrows, ff.ncols))
    indices = np.where(ff.maxframe == frameNum)
    img[indices] = ff.maxpixel[indices]
    return img

def getLightcurve(ff, stripe_indices, start, stop):
    xaxis = []
    lightcurve = []
    
    print stripe_indices
    
    stripe = np.zeros((ff.nrows, ff.ncols), np.uint8)
    stripe[stripe_indices] = ff.maxpixel[stripe_indices]
    
    stripe = treshold(stripe, ff).astype(np.uint8)*255
    
    cv2.imshow("stripe", stripe)
    cv2.moveWindow("stripe", 0, 0)
    cv2.waitKey(0)
    
    for i in range(start, stop):
        xaxis.append(i)
        
        single_frame = np.zeros((ff.nrows, ff.ncols))
        
        single_frame_indices = np.where(ff.maxframe == i)
        single_frame[single_frame_indices] = stripe[single_frame_indices]
                
        sum = np.sum(single_frame)
        
        lightcurve.append(sum)
        
    return lightcurve, xaxis

def reconstructWindows(path, filename):    
    ff = FFbin.read(path, filename)
    
    kht = ctypes.cdll.LoadLibrary("/home/dario/git/RMS/build/lib.linux-x86_64-2.7/kht_module.so")
    kht.kht_wrapper.argtypes = [npct.ndpointer(dtype=np.double, ndim=2),
                                npct.ndpointer(dtype=np.byte, ndim=1),
                                ctypes.c_size_t,
                                ctypes.c_size_t,
                                ctypes.c_size_t,
                                ctypes.c_double,
                                ctypes.c_double,
                                ctypes.c_double,
                                ctypes.c_double]
    kht.kht_wrapper.restype = ctypes.c_size_t
    
    for i in range(0, 256/time_slide-1):
        indices = np.where((ff.maxframe >= i*time_slide) & (ff.maxframe < i*time_slide+time_window_size))
        print indices
        img = np.zeros((ff.nrows, ff.ncols))
        img[indices] = ff.maxpixel[indices]
        
        raw_img = img.copy()
        
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
        
        img_cpy = img.copy()
        w = img.shape[1]
        h = img.shape[0]
        img = (img.flatten().astype(np.byte)*255).astype(np.byte)
        
        lines = np.empty((50, 2), np.double)
        
        t = time()
        
        length = kht.kht_wrapper(lines, img, w, h,
                                 9, 2, 0.1, 0.002, 1)
        
        lines = lines[:length]
        
        print "Time for KHT:", time()-t
        
        print lines
        
        if len(lines) > 0:
            img = img_cpy.astype(np.uint8)*255
            
            hh = img.shape[0] / 2.0
            hw = img.shape[1] / 2.0
            
            mul = img.shape[0] if hh>hw else img.shape[1]
                  
            strip_width = 15
            
            print "line_num:", len(lines)
                
            for rho, theta in lines:
                
                indicesy = []
                indicesx = []
                
                if theta < 45 or (theta > 90 and theta < 135):
                    theta = np.deg2rad(theta)
                    half_limit = strip_width/2 / np.cos(theta)
                    a = -np.tan(theta)
                    b = rho/np.cos(theta)
                    
                    print int(-hh), int(hh)
                    
                    for y in range(int(-hh), int(hh)):
                        x0 = a*y + b
                        
                        x1 = int(x0 - half_limit + hw)
                        x2 = int(x0 + half_limit + hw)
                        
                        if x1 > x2:
                            x1, x2 = x2, x1
                        
                        if x2 < 0 or x1 >= img.shape[1]:
                            continue
                        
                        for x in range(x1, x2):
                            if x < 0 or x >= img.shape[1]:
                                continue
                            
                            indicesy.append(y+hh)
                            indicesx.append(x)
                            
#                     print indices
                            
                else:
                    theta = np.deg2rad(theta)
                    half_limit = strip_width/2 / np.sin(theta)
                    a = -1/np.tan(theta)
                    b = rho/np.sin(theta)
                    
                    for x in range(int(-hw), int(hw)):
                        y0 = a*x + b
                        
                        y1 = int(y0 - half_limit + hh)                        
                        y2 = int(y0 + half_limit + hh)
                        
                        if y1 > y2:
                            y1, y2 = y2, y1
                        
                        if y2 < 0 or y1 >= img.shape[0]:
                            continue
                           
                        for y in range(y1, y2):
                            if y < 0 or y >= img.shape[0]:
                                continue
                            
                            indicesy.append(y)
                            indicesx.append(x+hw)
                            
#                 lightcurve, xaxis = getLightcurve(ff, np.swapaxes(np.array(indices, np.uint8), 0, 1))
                lightcurve, xaxis = getLightcurve(ff, (np.array(indicesy, np.uint32), np.array(indicesx, np.uint32)),
                                                  i*time_slide, i*time_slide+time_window_size)
#                 lightcurve, xaxis = getLightcurve(ff, ( np.array(indicesx, np.uint8), np.array(indicesy, np.uint8)))
                
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(np.array(xaxis), np.array(lightcurve))
                plt.xlim([0, 256])
                plt.show()
                
                break
#  *
#  *      'cluster_min_size' : Minimum number of pixels in the clusters of approximately
#  *                           collinear feature pixels. The default value is 10.
#  *
#  * 'cluster_min_deviation' : Minimum accepted distance between a feature pixel and
#  *                           the line segment defined by the end points of its cluster.
#  *                           The default value is 2.
#  *
#  *                 'delta' : Discretization step for the parameter space. The default
#  *                           value is 0.5.
#  *
#  *     'kernel_min_height' : Minimum height for a kernel pass the culling operation.
#  *                           This property is restricted to the [0,1] range. The
#  *                           default value is 0.002.
#  *
#  *              'n_sigmas' : Number of standard deviations used by the Gaussian kernel
#  *                           The default value is 2.

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