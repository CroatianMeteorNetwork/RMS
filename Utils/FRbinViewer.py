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
import platform
import subprocess

import cv2
import numpy as np

import RMS.ConfigReader as cr
from RMS.Formats import FFfile, FRbin


def view(dir_path, ff_path, fr_path, config, save_frames=False, extract_format=None, hide=False,
        avg_background=False):
    """ Shows the detected fireball stored in the FR file. 
    
    Arguments:
        dir_path: [str] Current directory.
        ff: [str] path to the FF bin file
        fr: [str] path to the FR bin file
        config: [conf object] configuration structure

    Keyword arguments:
        save_frames: [bool] Save FR frames to disk. False by defualt.
        extract_format: [str] Format of saved images. E.g. png, jpg, mp4.
        hide: [bool] Don't show frames on the screen.
        avg_background: [bool] Avepixel as background. False by default, in which case the maxpixel will be
            used.

    """

    if extract_format is None:
        extract_format = 'png'

    else:
        save_frames = True
    
    name = fr_path
    fr = FRbin.read(dir_path, fr_path)

    print('------------------------')
    print('Showing file:', fr_path)


    if ff_path is None:
        #background = np.zeros((config.height, config.width), np.uint8)

        # Get the maximum extent of meteor frames
        y_size = max([max(np.array(fr.yc[i]) + np.array(fr.size[i])//2) for i in range(fr.lines)])
        x_size = max([max(np.array(fr.xc[i]) + np.array(fr.size[i])//2) for i in range(fr.lines)])

        # Make the image square
        img_size = max(y_size, x_size)

        background = np.zeros((img_size, img_size), np.uint8)

    else:
        if avg_background:
            background = FFfile.read(dir_path, ff_path).avepixel
        else:
            background = FFfile.read(dir_path, ff_path).maxpixel
    
    print("Number of lines:", fr.lines)
    
    first_image = True
    wait_time = 2*int(1000.0/config.fps)

    pause_flag = False

    # if the file format was mp4, lets make a video from the data
    makevideo = False
    if extract_format == 'mp4':
        makevideo = True
        extract_format = 'png'

    for current_line in range(fr.lines):

        print('Frame,  Y ,  X , size')
        framefiles=[] # array to hold names of frames for later deletion 

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
            x_img = np.arange(xc - size//2, xc + size//2)

            Y_img, X_img = np.meshgrid(y_img, x_img)

            y_frame = np.arange(len(y_img))
            x_frame = np.arange(len(x_img))

            Y_frame, X_frame = np.meshgrid(y_frame, x_frame)                

            img[Y_img, X_img] = fr.frames[current_line][z][Y_frame, X_frame]


            # Save frame to disk
            if save_frames:
                frame_file_name = fr_path.replace('.bin', '') \
                    + "_line_{:02d}_frame_{:03d}.{:s}".format(current_line, t, extract_format)
                cv2.imwrite(os.path.join(dir_path, frame_file_name), img)
                framefiles.append(frame_file_name)
                img_patt = os.path.join(dir_path, fr_path.replace('.bin', '')
                    + "_line_{:02d}_frame_%03d.{:s}".format(current_line, extract_format))

            if not hide:
            
                # Show the frame
                try:
                    cv2.imshow(name, img)
                except:
                    print("imshow not available in OpenCV, Rebuild the library with Windows, GTK+ 2.x or Cocoa support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function 'cvShowImage'")
                    hide = True
                    first_image = False
                    continue

                # If this is the first image, move it to the upper left corner
                if first_image:
                    cv2.moveWindow(name, 0, 0)
                    first_image = False


                if pause_flag:
                    wait_time = 0
                else:
                    wait_time = 2*int(1000.0/config.fps)

                # Space key: pause display. 
                # 1: previous file. 
                # 2: next line. 
                # q: Quit.
                key = cv2.waitKey(wait_time) & 0xFF

                if key == ord("1"): 
                    cv2.destroyWindow(name)
                    return -1

                elif key == ord("2"): 
                    break

                elif key == ord(" "): 
                    
                    # Pause/unpause video
                    pause_flag = not pause_flag

                elif key == ord("q"): 
                    os._exit(0)

        if makevideo is True:
            root = os.path.dirname(__file__)
            ffmpeg_path = os.path.join(root, "ffmpeg.exe")
            first_frame = min(fr.t[current_line])
            mp4_path = os.path.join(dir_path, fr_path.replace('.bin', '') + '_line_{:02d}.mp4'.format(current_line))

            # If running on Windows, use ffmpeg.exe
            if platform.system() == 'Windows':
                com = ffmpeg_path + " -y -f image2 -pattern_type sequence -start_number " + str(first_frame) + " -i " + img_patt +" " + mp4_path
                subprocess.call(com, shell=True, cwd=dir_path)
            else:
                software_name = "avconv"
                if os.system(software_name + " --help > /dev/null"):
                    software_name = "ffmpeg"
                    # Construct the ecommand for ffmpeg           
                    com = software_name + " -y -f image2 -pattern_type sequence -start_number " + str(first_frame) + " -i " + img_patt +" " + mp4_path
                else:
                    com = "cd " + dir_path + ";" \
                        + software_name + " -v quiet -r 30 -y -start_number " + str(first_frame) + " -i " + img_patt \
                        + " -vcodec libx264 -pix_fmt yuv420p -crf 25 -movflags faststart -g 15 -vf \"hqdn3d=4:3:6:4.5,lutyuv=y=gammaval(0.97)\" " \
                        + mp4_path
                subprocess.call(com, shell=True, cwd=dir_path)
            for frame in framefiles:
                os.remove(os.path.join(dir_path, frame))

    if not hide:
        cv2.destroyWindow(name)
            

if __name__ == "__main__":

    # COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="""Show reconstructed fireball detections from FR files.
        Key mapping:
            Space: pause display.
            1: previous file.
            2: next line.
            q: Quit.
            """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('dir_path', nargs=1, metavar='DIR_PATH', type=str,
        help='Path to the directory which contains FR bin files.')

    arg_parser.add_argument('-e', '--extract', action="store_true",
        help="Save frames from FR files to disk.")

    arg_parser.add_argument('-a', '--avg', action="store_true",
        help="Average pixel as the background instead of maxpixel.")

    arg_parser.add_argument('-x', '--hide', action="store_true",
        help="Do not show frames on the screen.")
    
    arg_parser.add_argument('-f', '--extractformat', metavar='EXTRACT_FORMAT', help="""Image format for extracted files. png by default. """)

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str,
        help="Path to a config file which will be used instead of the default one.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    dir_path = cml_args.dir_path[0]

    # Load the configuration file
    config = cr.loadConfigFromDirectory(cml_args.config, 'notused')

    

    # Get the list of FR bin files (fireball detections) in the given directory
    fr_list = [fr for fr in os.listdir(dir_path) if fr[0:2]=="FR" and fr.endswith('bin')]
    fr_list = sorted(fr_list)

    if not fr_list:

        print("No files found!")
        sys.exit()

    # Get the list of FF bin files (compressed video frames)
    ff_list = [ff for ff in os.listdir(dir_path) if FFfile.validFFName(ff)]
    ff_list = sorted(ff_list)


    i = 0

    while True:

        # Break the loop if at the end
        if i >= len(fr_list):
            break

        fr = fr_list[i]

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
        retval = view(dir_path, ff_match, fr, config, save_frames=cml_args.extract,
            extract_format=cml_args.extractformat, hide=cml_args.hide, avg_background=cml_args.avg)

        # Return to previous file
        if retval == -1:
            i -= 2

        if i < 0:
            i = 0

        i += 1
