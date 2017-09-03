""" Showing fireball detections from FR bin files. """

# RPi Meteor Station
# Copyright (C) 2017  Dario Zubovic, Denis Vida
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

from __future__ import print_function, absolute_import, division

import cv2
import numpy as np
import os
import sys

import RMS.ConfigReader as cr
from RMS.Formats import FFfile, FRbin


def view(dir_path, ff, fr, config):
    """ Shows the detected fireball stored in the FR file. 
    
    Arguments:
        dir_path: [str] Current directory.
        ff: [str] path to the FF bin file
        fr: [str] path to the FR bin file
        config: [conf object] configuration structure

    """

    if ff is None:
        background = np.zeros((config.height, config.width), np.uint8)

    else:
        background = FFfile.read(dir_path, ff).maxpixel
    
    name = fr
    fr = FRbin.read(dir_path, fr)
    
    print("Number of lines:", fr.lines[0])
    
    first_image = True

    for i in range(fr.lines):

        print('Frame,  Y ,  X , size')

        for z in range(fr.frameNum[i]):

            # Get the center position of the detection on the current frame
            yc = fr.yc[i][z]
            xc = fr.xc[i][z]

            # Get the frame number
            t = fr.t[i][z]

            # Get the size of the window
            size = fr.size[i][z]
            
            print("  {:3d}, {:3d}, {:3d}, {:d}".format(t, yc, xc, size))
            
            
            y2 = 0

            # Assign the detection pixels to the background image
            for y in range(yc - size//2, yc + size//2):

                x2 = 0

                for x in range(xc - size//2,  xc + size//2):

                    background[y, x] = fr.frames[i][z][y2, x2]
                    x2 += 1

                y2 += 1
            
            cv2.imshow(name, background)

            # If this is the first image, move it to the upper left corner
            if first_image:
                cv2.moveWindow(name, 0, 0)
                first_image = False

            cv2.waitKey(2*int(1000.0/config.fps))
    
    cv2.destroyWindow(name)
            

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print('Usage: python -m Utils.FRbinViewer /path/to/FRbin/dir/')
        sys.exit()

    dir_path = sys.argv[1].replace('"', '')

    # Load the configuration file
    config = cr.parse(".config")

    

    # Get the list of FR bin files (fireball detections) in the given directory
    fr_list = [fr for fr in os.listdir(dir_path) if fr[0:2]=="FR" and fr[-3:]=="bin"]
    fr_list = sorted(fr_list)

    if not fr_list:

        print("No files found!")
        sys.exit()

    # Get the list of FF bin files (compressed video frames)
    ff_list = [ff for ff in os.listdir(dir_path) if ff[0:2]=="FF" and ff[-3:]=="bin"]
    ff_list = sorted(ff_list)

    for fr in fr_list:
        ff_match = None

        # Find the matching FF bin to the given FR bin
        for ff in ff_list:

            if ff[2:] == fr[2:]:
                ff_match = ff
                break
        
        # View the fireball detection
        view(dir_path, ff_match, fr, config)