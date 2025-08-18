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

from RMS.Compression import Compressor
import RMS.ConfigReader as cr

import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":

    config = cr.parse(".config")

    frames = np.empty((256, 576, 720), np.uint8)
    for i in range(256):
        frames[i] = np.random.normal(128, 2, (576, 720))
    
    comp = Compressor(None, None, None, None, None, config)
    compressed, field_intensities = comp.compress(frames)
    
    plt.hist(compressed[1].ravel(), 256, [0,256])
    plt.xlim((0, 255))
    plt.title('Randomness histogram')
    plt.xlabel('Frame')
    plt.ylabel('Random value count')
    plt.show()