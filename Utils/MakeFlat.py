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
        for file_name in os.listdir(dir_path):
            if ('CALSTARS' in file_name) and ('.txt' in file_name):
                calstars_file = file_name
                break

        if calstars_file is None:
            print('CALSTARS file could not be found in the given directory!')
            return None

        # Load the calstars file
        calstars_data = CALSTARS.readCALSTARS(dir_path, calstars_file)
        calstars_list, ff_frames = calstars_data

        # Convert the list to a dictionary
        calstars = {ff_file: star_data for ff_file, star_data in calstars_list}

        print('CALSTARS file: ' + calstars_file + ' loaded!')

        # A list of FF files which have any stars on them
        calstars_ff_files = [line[0] for line in calstars_list]

    else:
        calstars = {}
        calstars_ff_files = []

        # Without CALSTARS the frame count per FF is not known, use the standard block
        ff_frames = 256



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
                    ff_time = date2JD(*getMiddleTimeFF(ff_name, config.fps, ret_milliseconds=True, 
                                                       ff_frames=ff_frames))


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



    # Combine the images in chunks, then combine the chunk results. The chunk results are kept in
    # float32, exact for the half-integer medians of 8 and 16-bit input, in one preallocated block, so
    # the peak is that block plus one chunk of images rather than a float64 copy of everything
    chunk_size = 10
    n_files = len(ff_list_good)
    n_chunks = int(np.ceil(n_files/float(chunk_size)))

    chunk_results = None
    n_stored = 0
    img_list = []

    for i, ff_name in enumerate(ff_list_good):

        if use_images:
            img = loadImage(os.path.join(dir_path, ff_name), -1)

        else:
            # Only the average pixel is used for the flat
            ff = readFF(dir_path, ff_name, planes=('avepixel',))

            if ff is None:
                continue

            img = ff.avepixel

        img_list.append(img)

        # Combine a full chunk, or whatever is left at the end
        if (len(img_list) == chunk_size) or (i == n_files - 1):

            chunk = combine_function(np.array(img_list), axis=0)
            img_list = []

            if chunk_results is None:
                chunk_results = np.empty((n_chunks,) + chunk.shape, dtype=np.float32)

            chunk_results[n_stored] = chunk
            n_stored += 1


    if n_stored == 0:
        print('No readable images to make a flat from!')
        return None

    if n_stored == 1:
        ff_median = chunk_results[0].astype(np.float64)

    else:
        # Combine the chunk results in row strips, so the working copy numpy makes for the reduction
        # never holds more than a strip
        ff_median = np.empty(chunk_results.shape[1:], dtype=np.float64)
        strip = 64
        for r0 in range(0, ff_median.shape[0], strip):
            ff_median[r0:r0 + strip] = combine_function(chunk_results[:n_stored, r0:r0 + strip], axis=0)


    if not make_dark:

        # Stretch flat to 0-255
        ff_median = ff_median/np.max(ff_median)*255

        # Convert the flat to 8 bits, rounding instead of truncating
        ff_median = np.rint(ff_median).astype(np.uint8)

    else:

        # Keep darks in a type wide enough for the source data (a 16-bit dark does not fit in
        # 8 bits - the old unconditional uint8 cast wrapped such values modulo 256), rounding
        # instead of truncating
        out_type = np.uint8 if np.max(ff_median) <= 255 else np.uint16
        ff_median = np.clip(np.rint(ff_median), 0, np.iinfo(out_type).max).astype(out_type)

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
        help="""Use image files (bmp, png, jpg) for flat instead of FF files. Images of the file type with the highest frequency in the directory will be taken.""")

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str,
        help="Path to a config file which will be used instead of the default one.")

    arg_parser.add_argument('-d', '--dark', action="store_true",
        help="Make a dark frame instead by taking the minimum value of the images instead of the median.")

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
