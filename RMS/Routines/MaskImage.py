# RPi Meteor Station
# Copyright (C) 2016  Dario Zubovic, Denis Vida
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

from __future__ import absolute_import, division, print_function

import os
import zipfile

import numpy as np
import cv2

from RMS.Logger import getLogger
from RMS.Routines.Image import loadImage

# Get the logger from the main module
log = getLogger("logger")



class MaskStructure(object):
    def __init__(self, img):
        """ Structure for holding the mask. This is used so the mask can be hashed. """
        
        self.img = img

        if img is not None:
            
            # Get the mask resolution
            self.height = img.shape[0]
            self.width = img.shape[1]

        else:
            self.height = None
            self.width = None

    def resetEmpty(self, x_res, y_res):
        """ Reset the mask to an empty array. """

        self.img = np.full((y_res, x_res), 255, dtype=np.uint8)


    def checkResolution(self, x_res, y_res):
        """ Check if the mask has the given resolution. """

        if self.img is not None:

            if (self.img.shape[0] == y_res) and (self.img.shape[1] == x_res):

                return True

            else:

                return False

        return None


    def checkMask(self, x_res, y_res):
        """ Check the if the mask resolution matches and reset it if it doesn't. """

        if self.img is None:
            self.resetEmpty(x_res, y_res)

        elif not self.checkResolution(x_res, y_res):
            print("MASK RESET because the resolution didn't match!")
            self.resetEmpty(x_res, y_res)



def getMaskFile(dir_path, config, file_list=None, default_as_backup=False):
    """ Get the mask file from the given directory. If the mask file is not found, return None.
        It will also check if the mask is a zip file and load it if it is.

    Arguments:
        dir_path: [str] Path to the directory where the mask file is located.
        config: [Config] Configuration object.

    Keyword arguments:
        file_list: [list] List of files in the directory. If None, the files will be listed.
        default_as_backup: [bool] If True, the default mask file will be used as a backup, from the RMS 
            directory.

    Returns:
        mask: [MaskStructure] Mask structure object. If the mask file is not found, None is returned.
    """

    mask = None

    if file_list is None:
        file_list = os.listdir(dir_path)

    # Look through files and if there is mask.bmp or mask.zip, keep track of that then load it
    mask_status = max(
        2*(os.path.splitext(os.path.basename(config.mask_file))[0] == os.path.splitext(os.path.basename(filename))[0]) 
            - filename.endswith('.zip')
               for filename in file_list
    )
    
    # If a mask file is found, load it
    if mask_status > 0:
        
        mask_path = os.path.join(
            dir_path, 
            config.mask_file if mask_status == 2 else os.path.splitext(
                os.path.basename(config.mask_file))[0] + '.zip'
        )
        
        print("Loading mask:", mask_path)
        mask = loadMask(mask_path)

        if mask is None:
            print("  Mask file could not be loaded!")
        else:
            print("  Mask file loaded!")
            

    # If the mask file is not found, use the default mask file as a backup
    if (mask is None) and default_as_backup:

        # Path to the mask file in the config directory
        default_mask_path = os.path.join(config.config_file_path, config.mask_file)

        # Check if the mask file is in the config directory, if not, use the default mask file
        if not os.path.isfile(default_mask_path):

            # Path to the mask in RMS source directory
            default_mask_path = os.path.join(config.rms_root_dir, config.mask_file)

        print("Mask file not found! Using default mask file as a backup:", default_mask_path)

        mask = loadMask(default_mask_path)

        if mask is None:
            print("  Default mask file could not be loaded!")
        else:
            print("  Default mask file loaded!")


    if mask is None:
        print("No mask used!")

    return mask
    

def loadMask(mask_file):
    """ Load the mask image. """

    # If there is no mask file
    if not os.path.isfile(mask_file):
        return None

    # Load the mask file
    try:

        # Load a mask from zip
        if mask_file.endswith('.zip'):

            with zipfile.ZipFile(mask_file, 'r') as archive:
                
                data = archive.read('mask.bmp')
                mask = cv2.imdecode(np.frombuffer(data, np.uint8), 1)

        else:
                
            mask = loadImage(mask_file, flatten=0)
        
    except:
        print("WARNING! The mask file could not be loaded! File path: {:s}".format(mask_file))
        return None

    # Convert the RGB image to one channel image (if not already one channel)
    try:
        mask = mask[:,:,0]
    except:
        pass


    mask_struct = MaskStructure(mask)

    return mask_struct



def maskImage(input_image, mask, image=False):
    """ Apply masking to the given image. 

    Keyword arguments:
        image: [bool] If True, the image for the mask was given, and no the MaskStructure instance.
    """


    if not image:
        mask = mask.img

    # If the image dimensions don't agree, dont apply the mask
    if input_image.shape != mask.shape:
        # log.warning('Image and mask dimensions do not agree! Skipping masking...')
        return input_image

    # Set all image pixels where the mask is black to the mean value of the image
    input_image[mask == 0] = np.mean(input_image[mask > 0])

    return input_image



def applyMask(input_image, mask, ff_flag=False, image=False):
    """ Apply a mask to the given image array or FF file. 
    
    Keyword arguments:
        image: [bool] If True, the image for the mask was given, and not the MaskStructure instance.
    """

    if mask is None:
        return input_image

    # Apply masking to an FF file
    if ff_flag:
        input_image.maxpixel = maskImage(input_image.maxpixel, mask, image=image)
        input_image.avepixel = maskImage(input_image.avepixel, mask, image=image)
        input_image.stdpixel = maskImage(input_image.stdpixel, mask, image=image)
        #input_image.maxframe = maskImage(input_image.maxframe, mask)

        return input_image

    # Apply the mask to a regular image array
    else:
        return maskImage(input_image, mask, image=image)




if __name__ == '__main__':

    mask_file = '../../mask.bmp'

    print(loadMask(mask_file))
