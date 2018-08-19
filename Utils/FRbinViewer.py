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

import os
import sys
import argparse

import cv2
import numpy as np

import RMS.ConfigReader as cr
from RMS.Formats import FFfile, FRbin


def view(dir_path, ff_path, fr_path, config, save_frames=False):
    """ Shows the detected fireball stored in the FR file. 
    
    Arguments:
        dir_path: [str] Current directory.
        ff: [str] path to the FF bin file
        fr: [str] path to the FR bin file
        config: [conf object] configuration structure

    """
    
    name = fr_path
    fr = FRbin.read(dir_path, fr_path)


    if ff_path is None:
        #background = np.zeros((config.height, config.width), np.uint8)

        # Get the maximum extent of the meteor frames
        y_size = max(max(np.array(fr.yc[0]) + np.array(fr.size[0])//2) for i in range(fr.lines))
        x_size = max(max(np.array(fr.xc[0]) + np.array(fr.size[0])//2) for i in range(fr.lines))

        # Make the image square
        img_size = max(y_size, x_size)

        background = np.zeros((img_size, img_size), np.uint8)

    else:
        background = FFfile.read(dir_path, ff_path).maxpixel
    
    print("Number of lines:", fr.lines)
    
    first_image = True

    for current_line in range(fr.lines):

        print('Frame,  Y ,  X , size')

        for z in range(fr.frameNum[current_line]):

            # Get the center position of the detection on the current frame
            yc = fr.yc[current_line][z]
            xc = fr.xc[current_line][z]

            # Get the frame number
            t = fr.t[current_line][z]

            # Get the size of the window
            size = fr.size[current_line][z]
            
            print("  {:3d}, {:3d}, {:3d}, {:d}".format(t, yc, xc, size))

            img = np.copy(background)
            
            # Paste the frames onto the big image
            y_img = np.arange(yc - size//2, yc + size//2)
            x_img = np.arange(xc - size//2,  xc + size//2)

            Y_img, X_img = np.meshgrid(y_img, x_img)

            y_frame = np.arange(len(y_img))
            x_frame = np.arange(len(x_img))

            Y_frame, X_frame = np.meshgrid(y_frame, x_frame)                

            img[Y_img, X_img] = fr.frames[current_line][z][Y_frame, X_frame]


            # Save frame to disk
            if save_frames:
                frame_file_name = fr_path.replace('.bin') + "_frame_{:03d}.png".format(t)
                cv2.imwrite(os.path.join(dir_path, frame_file_name), img)


            # Show the frame
            cv2.imshow(name, img)

            # If this is the first image, move it to the upper left corner
            if first_image:
                cv2.moveWindow(name, 0, 0)
                first_image = False

            cv2.waitKey(2*int(1000.0/config.fps))
    
    cv2.destroyWindow(name)
            

if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Show reconstructed fireball detections from FR files.")

    arg_parser.add_argument('dir_path', nargs=1, metavar='DIR_PATH', type=str, \
        help='Path to the directory which contains FR bin files.')

    arg_parser.add_argument('-e', '--extract', metavar='EXTRACT_FR_FILE', type=str, \
        help="Save frames from a given FR file to disk.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    dir_path = cml_args.dir_path[0]

    # Load the configuration file
    config = cr.parse(".config")

    

    # Get the list of FR bin files (fireball detections) in the given directory
    fr_list = [fr for fr in os.listdir(dir_path) if fr[0:2]=="FR" and fr.endswith('bin')]
    fr_list = sorted(fr_list)

    if not fr_list:

        print("No files found!")
        sys.exit()

    # Get the list of FF bin files (compressed video frames)
    ff_list = [ff for ff in os.listdir(dir_path) if FFfile.validFFName(ff)]
    ff_list = sorted(ff_list)

    for fr in fr_list:
        ff_match = None

        # Strip extensions
        fr_name = ".".join(fr.split('.')[:-1]).replace('FR', '').strip("_")

        # Find the matching FF bin to the given FR bin
        for ff in ff_list:

            # Strip extensions
            ff_name = ".".join(ff.split('.')[:-1]).replace('FF', "").strip("_")


            if ff_name[2:] == fr_name[2:]:
                ff_match = ff
                break
        
        # View the fireball detection
        view(dir_path, ff_match, fr, config)