""" Make a flat field image from the images in the given folder. Images throughout the night will be used
    to estimate the background, but only those with enough stars so the clouds do not spoil the flat.
"""

from __future__ import print_function, division, absolute_import

import os
import random
import argparse

import numpy as np

from RMS.Astrometry.Conversions import date2JD
import RMS.ConfigReader as cr
import RMS.Formats.CALSTARS as CALSTARS
from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import validFFName
from RMS.Formats.FFfile import getMiddleTimeFF
from RMS.Routines.Image import loadImage, saveImage


def makeFlat(dir_path, config, nostars=False, use_images=False, make_dark=False):
    """ Makes a flat field from the files in the given folder. CALSTARS file is needed to estimate the
        quality of every image by counting the number of detected stars.

    Arguments:
        dir_path: [str] Path to the directory which contains the FF files and a CALSTARS file.
        config: [config object]

    Keyword arguments:
        nostars: [bool] If True, all files will be taken regardless of if they have stars on them or not.
        use_images: [bool] Use image files instead of FF files. False by default.
        make_dark: [bool] If True, a dark frame will be made instead of a flat field. False by default.

    Return:
        [2d ndarray] Flat field image as a numpy array. If the flat generation failed, None will be returned.
        
    """

    # If only images are used, then don't look for a CALSTARS file
    if use_images:
        nostars = True

    # Load the calstars file if it should be used
    if not nostars:

        # Find the CALSTARS file in the given folder
        calstars_file = None
        for calstars_file in os.listdir(dir_path):
            if ('CALSTARS' in calstars_file) and ('.txt' in calstars_file):
                break

        if calstars_file is None:
            print('CALSTARS file could not be found in the given directory!')
            return None

        # Load the calstars file
        calstars_list = CALSTARS.readCALSTARS(dir_path, calstars_file)

        # Convert the list to a dictionary
        calstars = {ff_file: star_data for ff_file, star_data in calstars_list}

        print('CALSTARS file: ' + calstars_file + ' loaded!')

        # A list of FF files which have any stars on them
        calstars_ff_files = [line[0] for line in calstars_list]

    else:
        calstars = {}
        calstars_ff_files = []



    # Use image files
    if use_images:

        # Find the file type with the highest file frequency in the given folder
        file_extensions = []
        for file_name in os.listdir(dir_path):
            file_ext = file_name.split('.')[-1]
            if file_ext.lower() in ['jpg', 'png', 'bmp']:
                file_extensions.append(file_ext)
            
        # Get only the most frequent file type
        file_freqs = np.unique(file_extensions, return_counts=True)
        most_freq_type = file_freqs[0][0]

        print('Using image type:', most_freq_type)

        # Take only files of that file type
        ff_list = [file_name for file_name in sorted(os.listdir(dir_path))
            if file_name.lower().endswith(most_freq_type)]


    # Use FF files
    else:
        ff_list = []

        # Get a list of FF files in the folder
        for file_name in os.listdir(dir_path):
            if validFFName(file_name) and ((file_name in calstars_ff_files) or nostars):
                ff_list.append(file_name)
                

        # Check that there are any FF files in the folder
        if not ff_list:
            print('No valid FF files in the selected folder!')
            return None



    ff_list_good = []
    ff_times = []

    # Take only those FF files with enough stars on them
    for ff_name in ff_list:

        if (ff_name in calstars) or nostars:

            # Disable requiring minimum number of stars if specified
            if not nostars:
                
                # Get the number of stars detected on the FF image
                ff_nstars = len(calstars[ff_name])

            else:
                ff_nstars = 0

            
            # Check if the number of stars on the image is over the detection threshold
            if (ff_nstars > config.ff_min_stars) or nostars:

                # Add the FF file to the list of FF files to be used to make a flat
                ff_list_good.append(ff_name)


                # If images are used, don't compute the time
                if use_images:
                    ff_time = 0

                else:
                    # Calculate the time of the FF files
                    ff_time = date2JD(*getMiddleTimeFF(ff_name, config.fps, ret_milliseconds=True))


                ff_times.append(ff_time)


    # Check that there are enough good FF files in the folder
    if (len(ff_times) < config.flat_min_imgs) and (not nostars):
        print('Not enough FF files have enough stars on them!')
        return None
        
    
    # Make sure the files cover at least 2 hours
    if (not (max(ff_times) - min(ff_times))*24 > 2) and (not nostars):
        print('Good FF files cover less than 2 hours!')
        return None


    # Sample FF files if there are more than 200
    max_ff_flat = 200
    if len(ff_list_good) > max_ff_flat:
        ff_list_good = sorted(random.sample(ff_list_good, max_ff_flat))



    if make_dark:
        print('Making a dark frame...')

        combine_function = np.min

    else:

        print('Using {:d} files for flat...'.format(len(ff_list_good)))

        combine_function = np.median



    c = 0
    img_list = []
    median_list = []

    # Median combine all good FF files
    for i in range(len(ff_list_good)):

        # Load 10 files at the time and median combine them, which conserves memory
        if c < 10:

            # Use images
            if use_images:
                img = loadImage(os.path.join(dir_path, ff_list_good[i]), -1)


            # Use FF files
            else:
                ff = readFF(dir_path, ff_list_good[i])

                # Skip the file if it is corruped
                if ff is None:
                    continue

                img = ff.avepixel

            
            img_list.append(img)

            c += 1


        else:

            img_list = np.array(img_list)

            # Median combine the loaded 10 (or less) images
            ff_median = combine_function(img_list, axis=0)
            median_list.append(ff_median)

            img_list = []
            c = 0


    # If there are more than 1 calculated median image, combine them
    if len(median_list) > 1:

        # Median combine all median images
        median_list = np.array(median_list)
        ff_median = combine_function(median_list, axis=0)

    else:
        if len(median_list) > 0:
            ff_median = median_list[0]
        else:
            ff_median = combine_function(np.array(img_list), axis=0)


    if not make_dark:
        
        # Stretch flat to 0-255 
        ff_median = ff_median/np.max(ff_median)*255

    # Convert the flat to 8 bits
    ff_median = ff_median.astype(np.uint8)

    return ff_median




if __name__ == "__main__":

    # COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Makes a flat from FF files in the given folder. Only those files with star detection are taken, but this can be disabled.")

    arg_parser.add_argument('dir_path', nargs=1, metavar='DIR_PATH', type=str,
        help='Path to directory with FF files.')

    arg_parser.add_argument('-n', '--nostars', action="store_true",
        help="""Disable requiring stars on images for generating the flat field.""")

    arg_parser.add_argument('-i', '--images', action="store_true",
        help="""Use image files (bmp, png, jpg) for flat instead of FF files. Images of the file type with the higest frequency in the directory will be taken.""")

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str,
        help="Path to a config file which will be used instead of the default one.")

    arg_parser.add_argument('-d', '--dark', action="store_true",
        help="Make a dark frame instead by taking the minimum value of the images insted of the median.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    dir_path = cml_args.dir_path[0]


    # Load the configuration file
    config = cr.loadConfigFromDirectory(cml_args.config, 'notused')

    # Make the flat
    img = makeFlat(dir_path, config, nostars=cml_args.nostars, use_images=cml_args.images, 
                         make_dark=cml_args.dark)

    if img is not None:


        # Save the flat in the input directory
        if cml_args.dark:
            img_save_path = os.path.join(dir_path, "dark.png")

        else:
            img_save_path = os.path.join(dir_path, config.flat_file)

        saveImage(img_save_path, img)
        print('Image saved to:', img_save_path)

        import matplotlib.pyplot as plt
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.show()

    else:
        print('Flat file could not be made!')
