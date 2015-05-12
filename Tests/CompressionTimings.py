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

from RMS.Compression import Compression
import numpy as np
import time

def timing(img, s):
    t = time.time()
    Compression.compress(img)
    print s + ": " + str(time.time() - t) + "s"

if __name__ == "__main__":
    shape = (576, 720, 256)
    
    black = np.zeros(shape, np.uint8)
    timing(black, "black")
    
    white = np.full(shape, 255, np.uint8)
    timing(white, "white")
    
    noise = np.random.uniform(0, 256, shape)
    timing(noise, "uniform noise")
    
    gauss = np.random.normal(128, 2, shape)
    timing(gauss, "Gaussian noise")