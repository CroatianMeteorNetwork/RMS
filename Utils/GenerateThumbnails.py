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
    """ Blends two image arrays with the lighten method (only takes the lighter pixel on each spot).
        The result is uint8, as the thumbnails are.
    """

    return np.maximum(arr1, arr2).astype(np.uint8, copy=False)



class ThumbnailMosaic(object):
    def __init__(self, config, mosaic_type, n_files, no_stack=False):
        """ Builds the mosaic of thumbnails one FF file at a time, so the images can come from any reader
            loop. Every thumb_stack consecutive files are stacked 'if lighter' into one thumbnail.

        Arguments:
            config: [Conf object] Configuration.
            mosaic_type: [str] Type of the mosaic (e.g. "CAPTURED" or "DETECTED")
            n_files: [int] Number of FF files that will be added.

        Keyword arguments:
            no_stack: [bool] Don't stack the images using the config.thumb_stack option. A max of 1000
                images are supported with this option. If there are more, stacks will be done according
                to the config.thumb_stack option.
        """

        self.config = config
        self.mosaic_type = mosaic_type

        # Calculate the dimensions of the binned image
        self.bin_w = int(config.width/config.thumb_bin)
        self.bin_h = int(config.height/config.thumb_bin)

        self.thumb_stack = config.thumb_stack

        # Check if no stacks should be done (max 1000 images for no stack)
        if no_stack and (n_files < 1000):
            self.thumb_stack = 1

        self.timestamps = []
        self.stacked_imgs = []
        self.n_added = 0


    def add(self, ff_name, maxpixel):
        """ Add one FF file to the mosaic. The input array is not modified.

        Arguments:
            ff_name: [str] Name of the FF file, used for the timestamp of the first file in each stack.
            maxpixel: [ndarray] Maxpixel image, or None if the file is corrupted (it still takes its slot
                in the stack, as before).
        """

        # Start a new stack every thumb_stack files, stamped with the time of its first file
        if self.n_added % self.thumb_stack == 0:
            self.stacked_imgs.append(np.zeros((self.bin_h, self.bin_w), dtype=np.uint8))
            self.timestamps.append(FFfile.filenameToDatetime(ff_name))

        self.n_added += 1

        if maxpixel is None:
            return

        # Resize the image
        img = cv2.resize(maxpixel, (self.bin_w, self.bin_h))

        # Stack the image
        self.stacked_imgs[-1] = stackIfLighter(self.stacked_imgs[-1], img)


    def save(self, dir_path):
        """ Assemble the mosaic and save it to the night directory as a JPG.

        Arguments:
            dir_path: [str] Path of the night directory.

        Return:
            file_name: [str] Name of the thumbnail file.
        """

        config = self.config
        bin_w, bin_h = self.bin_w, self.bin_h

        header_height = 20
        timestamp_height = 10

        # Calculate the number of rows for the thumbnail image
        n_rows = int(np.ceil(float(self.n_added)/self.thumb_stack/config.thumb_n_width))

        # Calculate the size of the mosaic
        mosaic_w = int(config.thumb_n_width*bin_w)
        mosaic_h = int((bin_h + timestamp_height)*n_rows + header_height)

        mosaic_img = np.zeros((mosaic_h, mosaic_w), dtype=np.uint8)

        # Write header text
        header_text = 'Station: ' + str(config.stationID) + ' Night: ' + os.path.basename(dir_path) \
            + ' Type: ' + self.mosaic_type

        cv2.putText(mosaic_img, header_text, (0, header_height//2), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        for row in range(n_rows):

            for col in range(config.thumb_n_width):

                # Calculate image index
                indx = row*config.thumb_n_width + col

                if indx < len(self.stacked_imgs):

                    # Calculate position of the text
                    text_x = col*bin_w
                    text_y = row*bin_h + (row + 1)*timestamp_height - 1 + header_height

                    # Add timestamp text
                    cv2.putText(mosaic_img, self.timestamps[indx].strftime('%H:%M:%S'), (text_x, text_y), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                    # Add the image to the mosaic
                    img_pos_x = col*bin_w
                    img_pos_y = row*bin_h + (row + 1)*timestamp_height + header_height

                    mosaic_img[img_pos_y : img_pos_y + bin_h, img_pos_x : img_pos_x + bin_w] = \
                        self.stacked_imgs[indx]


                else:
                    break


        # Only add the station ID if the dir name already doesn't start with it
        dir_name = os.path.basename(os.path.abspath(dir_path))
        if dir_name.startswith(config.stationID):
            prefix = dir_name
        else:
            prefix = "{:s}_{:s}".format(config.stationID, dir_name)

        thumb_name = "{:s}_{:s}_thumbs.jpg".format(prefix, self.mosaic_type)

        # Save the mosaic
        if USING_IMAGEIO:
            # Use imageio to write the image
            imwrite(os.path.join(dir_path, thumb_name), mosaic_img, quality=80)
        else:
            # Use OpenCV to save the image
            imwrite(os.path.join(dir_path, thumb_name), mosaic_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

        return thumb_name



def generateThumbnails(dir_path, config, mosaic_type, file_list=None, no_stack=False):
    """ Generates a mosaic of thumbnails from all FF files in the given folder and saves it as a JPG image.
    
    Arguments:
        dir_path: [str] Path of the night directory.
        config: [Conf object] Configuration.
        mosaic_type: [str] Type of the mosaic (e.g. "Captured" or "Detected")

    Keyword arguments:
        file_list: [list] A list of file names (without full path) which will be searched for FF files. This
            is used when generating separate thumbnails for captured and detected files.
        no_stack: [bool] Don't stack the images using the config.thumb_stack option. A max of 1000 images
            are supported with this option. If there are more, stacks will be done according to the 
            config.thumb_stack option.

    Return:
        file_name: [str] Name of the thumbnail file.

    """

    if file_list is None:
        file_list = sorted(os.listdir(dir_path))


    # Make a list of all FF files in the night directory
    ff_list = [file_name for file_name in file_list if FFfile.validFFName(file_name)]

    mosaic = ThumbnailMosaic(config, mosaic_type, len(ff_list), no_stack=no_stack)

    for ff_name in ff_list:

        # Read the FF file, only the maxpixel is used
        ff = FFfile.read(dir_path, ff_name, planes=('maxpixel',))

        # A corrupted FF still takes its slot in the stack
        mosaic.add(ff_name, None if ff is None else ff.maxpixel)

    return mosaic.save(dir_path)



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
