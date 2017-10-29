""" Make a flat field image from the images in the given folder. Images throughout the night will be used
    to estimate the background, but only those with enough stars so the clouds do not spoil the flat.
"""

from __future__ import print_function, division, absolute_import

import os
import sys
import random

import numpy as np

import RMS.ConfigReader as cr
import RMS.Formats.CALSTARS as CALSTARS
from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import validFFName
from RMS.Formats.FFfile import getMiddleTimeFF
from RMS.Astrometry.Conversions import date2JD


def makeFlat(dir_path, config):
    """ Makes a flat field from the files in the given folder. CALSTARS file is needed to estimate the
        quality of every image by counting the number of detected stars.

    Arguments:
        dir_path: [str] Path to the directory which contains the FF files and a CALSTARS file.
        config: [config object]

    Return:
        [2d ndarray] Flat field image as a numpy array. If the flat generation failed, None will be returned.
        
    """


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

    ff_list = []

    # Get a list of FF files in the folder
    for file_name in os.listdir(dir_path):
        if validFFName(file_name) and (file_name in calstars_ff_files):
            ff_list.append(file_name)


    # Check that there are any FF files in the folder
    if not ff_list:
        print('No FF files in the selected folder!')
        return None



    ff_list_good = []
    ff_times = []

    # Take only those FF files with enough stars on them
    for ff_name in ff_list:

        if not validFFName(ff_name):
            continue

        if ff_name in calstars:

            # Get the number of stars detected on the FF image
            ff_nstars = len(calstars[ff_name])
            
            # Check if the number of stars on the image is over the detection threshold
            if ff_nstars > config.ff_min_stars:

                # Add the FF file to the list of FF files to be used to make a flat
                ff_list_good.append(ff_name)

                # Calculate the time of the FF files
                ff_time = date2JD(*getMiddleTimeFF(ff_name, config.fps, ret_milliseconds=True))
                ff_times.append(ff_time)


    
    # Make sure the files cover at least 2 hours
    if not (max(ff_times) - min(ff_times))*24 > 2:
        print('Good FF files cover less than 2 hours!')
        return None


    # Sample FF files if there are more than 200
    max_ff_flat = 200
    if len(ff_list_good) > max_ff_flat:
        ff_list_good = [x[1] for x in sorted(random.sample(enumerate(ff_list_good), max_ff_flat))]

    c = 0
    ff_avg_list = []
    median_list = []

    # Median combine all good FF files
    for i in range(len(ff_list_good)):

        # Load 10 files at the time and median combine them, which conserves memory
        if c < 10:

            ff = readFF(dir_path, ff_list_good[i])
            ff_avg_list.append(ff.avepixel)

            c += 1 


        else:

            ff_avg_list = np.array(ff_avg_list)

            # Median combine the loaded 10 (or less) images
            ff_median = np.median(ff_avg_list, axis=0)
            median_list.append(ff_median)

            ff_avg_list = []
            c = 0


    # If there are more than 1 calculated median image, combine them
    if len(median_list) > 1:

        # Median combine all median images
        median_list = np.array(median_list)
        ff_median = np.median(median_list, axis=0)

    else:
        ff_median = median_list[0]


    return ff_median




if __name__ == "__main__":

    import scipy.misc

    if len(sys.argv) < 2:
        print('Usage: python -m RMS.Utils.MakeFlat /path/to/FFbin/dir/')
        sys.exit()

    dir_path = sys.argv[1].replace('"', '')


    # Load the configuration file
    config = cr.parse(".config")

    # Make the flat
    ff_median = makeFlat(dir_path, config)


    if ff_median is not None:

        # Save the flat in the root directory
        scipy.misc.imsave(config.flat_file, ff_median)

        print('Flat saved to:', os.path.join(os.getcwd(), config.flat_file))

        import matplotlib.pyplot as plt
        plt.imshow(ff_median, cmap='gray', vmin=0, vmax=255)
        plt.show()

    else:
        print('Flat filed could not be made!')