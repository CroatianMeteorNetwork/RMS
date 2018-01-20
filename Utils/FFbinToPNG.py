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

import cv2
import sys, os
from RMS.Formats import FFfile

if __name__ == "__main__":
    ff = FFfile.read(sys.argv[1], sys.argv[2])
    
    cv2.imwrite(sys.argv[2] + "_max.png", ff.maxpixel)
    cv2.imwrite(sys.argv[2] + "_frame.png", ff.maxframe) 
    cv2.imwrite(sys.argv[2] + "_avg.png", ff.avepixel)
    cv2.imwrite(sys.argv[2] + "_stddev.png", ff.stdpixel)