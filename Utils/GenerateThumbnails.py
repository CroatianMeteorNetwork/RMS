# RPi Meteor Station
# Copyright (C) 2017  Denis Vida
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


from __future__ import print_function, division, absolute_import


""" Generates a thumbnail image of all FF files in the given directory. """

import os
import argparse

import numpy as np
import cv2

try:
    import imageio
    imwrite = imageio.imwrite
    USING_IMAGEIO = True
except ImportError:
    imwrite = cv2.imwrite
    USING_IMAGEIO = False


import RMS.ConfigReader as cr
import RMS.Formats.FFfile as FFfile


def stackIfLighter(arr1, arr2):
    """ Blends two image array with lighen method (only takes the lighter pixel on each spot).
    """

    arr1 = arr1.astype(np.int16)

    temp = arr1 - arr2
    temp[temp > 0] = 0
    new_arr = arr1 - temp
    
    new_arr = new_arr.astype(np.uint8)
    
    return new_arr




def generateThumbnails(dir_path, config, mosaic_type, file_list=None, no_stack=False):
    """ Generates a mosaic of thumbnails from all FF files in the given folder and saves it as a JPG image.
    
    Arguments:
        dir_path: [str] Path of the night directory.
        config: [Conf object] Configuration.
        mosaic_type: [str] Type of the mosaic (e.g. "Captured" or "Detected")

    Keyword arguments:
        file_list: [list] A list of file names (without full path) which will be searched for FF files. This
            is used when generating separate thumbnails for captured and detected files.

    Return:
        file_name: [str] Name of the thumbnail file.
        no_stack: [bool] Don't stack the images using the config.thumb_stack option. A max of 1000 images
            are supported with this option. If there are more, stacks will be done according to the 
            config.thumb_stack option.

    """

    if file_list is None:
        file_list = sorted(os.listdir(dir_path))


    # Make a list of all FF files in the night directory
    ff_list = []

    for file_name in file_list:
        if FFfile.validFFName(file_name):
            ff_list.append(file_name)


    # Calculate the dimensions of the binned image
    bin_w = int(config.width/config.thumb_bin)
    bin_h = int(config.height/config.thumb_bin)


    ### RESIZE AND STACK THUMBNAILS ###
    ##########################################################################################################

    timestamps = []
    stacked_imgs = []


    thumb_stack = config.thumb_stack
    
    # Check if no stacks should be done (max 1000 images for no stack)
    if no_stack and (len(ff_list) < 1000):
        thumb_stack = 1


    for i in range(0, len(ff_list), thumb_stack):

        img_stack = np.zeros((bin_h, bin_w))

        # Stack thumb_stack images using the 'if lighter' method
        for j in range(thumb_stack):

            if (i + j) < len(ff_list):

                tmp_file_name = ff_list[i + j]

                    
                # Read the FF file
                ff = FFfile.read(dir_path, tmp_file_name)

                # Skip the FF if it is corruped
                if ff is None:
                    continue

                img = ff.maxpixel

                # Resize the image
                img = cv2.resize(img, (bin_w, bin_h))

                # Stack the image
                img_stack = stackIfLighter(img_stack, img)

            else:
                break


        # Save the timestamp of the first image in the stack
        timestamps.append(FFfile.filenameToDatetime(ff_list[i]))

        # Save the stacked image
        stacked_imgs.append(img_stack)

        # cv2.imshow('test', img_stack)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


    ##########################################################################################################

    ### ADD THUMBS TO ONE MOSAIC IMAGE ###
    ##########################################################################################################

    header_height = 20
    timestamp_height = 10

    # Calculate the number of rows for the thumbnail image
    n_rows = int(np.ceil(float(len(ff_list))/thumb_stack/config.thumb_n_width))

    # Calculate the size of the mosaic
    mosaic_w = int(config.thumb_n_width*bin_w)
    mosaic_h = int((bin_h + timestamp_height)*n_rows + header_height)

    mosaic_img = np.zeros((mosaic_h, mosaic_w), dtype=np.uint8)

    # Write header text
    header_text = 'Station: ' + str(config.stationID) + ' Night: ' + os.path.basename(dir_path) \
        + ' Type: ' + mosaic_type
    cv2.putText(mosaic_img, header_text, (0, header_height//2), \
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    for row in range(n_rows):

        for col in range(config.thumb_n_width):

            # Calculate image index
            indx = row*config.thumb_n_width + col

            if indx < len(stacked_imgs):

                # Calculate position of the text
                text_x = col*bin_w
                text_y = row*bin_h + (row + 1)*timestamp_height - 1 + header_height

                # Add timestamp text
                cv2.putText(mosaic_img, timestamps[indx].strftime('%H:%M:%S'), (text_x, text_y), \
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Add the image to the mosaic
                img_pos_x = col*bin_w
                img_pos_y = row*bin_h + (row + 1)*timestamp_height + header_height

                mosaic_img[img_pos_y : img_pos_y + bin_h, img_pos_x : img_pos_x + bin_w] = stacked_imgs[indx]


            else:
                break

    ##########################################################################################################

    # Only add the station ID if the dir name already doesn't start with it
    dir_name = os.path.basename(os.path.abspath(dir_path))
    if dir_name.startswith(config.stationID):
        prefix = dir_name
    else:
        prefix = "{:s}_{:s}".format(config.stationID, dir_name)

    thumb_name = "{:s}_{:s}_thumbs.jpg".format(prefix, mosaic_type)

    # Save the mosaic
    if USING_IMAGEIO:
        # Use imageio to write the image
        imwrite(os.path.join(dir_path, thumb_name), mosaic_img, quality=80)
    else:
        # Use OpenCV to save the image
        imwrite(os.path.join(dir_path, thumb_name), mosaic_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

    return thumb_name
    


    




if __name__ == "__main__":


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Generates a thumbnail image of all FF files in the given directory.")

    arg_parser.add_argument('dir_path', nargs=1, metavar='DIR_PATH', type=str, \
        help='Path to directory with FF files.')

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one.")

    arg_parser.add_argument('-n', '--nostack', action="store_true", \
        help="""Don't stack images.""")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, cml_args.dir_path)


    # Read the argument as a path to the night directory
    dir_path = cml_args.dir_path[0]

    generateThumbnails(dir_path, config, 'mosaic', no_stack=cml_args.nostack)
