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

""" Timings of compression algorithm with various cases.

Averaged results (20 runs) for Raspberry Pi 2 at 1000MHz ARM, 500MHz core, 500MHz SDRAM, 2 overvolt:
black: 10.0018053055
white: 9.9432981679
uniform noise: 5.16999864578
Gaussian noise: 5.31156075001
"""

from RMS.Compression import Compression
import numpy as np
import time
import sys

comp = Compression(None, None, None, None, 000)

def timing(img):
    t = time.time()
    comp.compress(img)
    return time.time() - t
   
def create(f):
    arr = np.empty((256, 576, 720), np.uint8)
    for i in range(256):
        arr[i] = f()
    return arr

def black():
    return np.zeros((576, 720), np.uint8)

def white():
    return np.full((576, 720), 255, np.uint8)

def uniform():
    return np.random.uniform(0, 256, (576, 720))

def gauss():
    return np.random.normal(128, 2, (576, 720))

def test(filename):
    npzFile = np.load(filename)
    
    t = [0, 0, 0, 0]
    
    for i in range(4):
        arr = npzFile["arr_"+str(i)]
        for n in range(2):
            t[i] += timing(arr)
    
    print "Black:", t[0]/2
    print "White:", t[1]/2
    print "Uniform noise:", t[2]/2
    print "Gaussian noise:", t[3]/2
       
def generate(filename):    
    blackArr = create(black)
    whiteArr = create(white)
    uniformArr = create(uniform)
    gaussArr = create(gauss)
    
    np.savez(filename, blackArr, whiteArr, uniformArr, gaussArr)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        test(sys.argv[1])
    elif len(sys.argv) == 3:
        generate(sys.argv[1])
    else:
        print "Usage: python -m Tests.CompressionTimings filename.npz"
        print "or:    python -m Tests.CompressionTimings filename.npz --generate"