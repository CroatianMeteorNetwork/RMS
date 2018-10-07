""" Stacks all maxpixles in the given folder to one image. """

from __future__ import print_function, division, absolute_import

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import validFFName
from RMS.Routines.Image import deinterlaceBlend, blendLighten, loadFlat, applyFlat




if __name__ == '__main__':

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Stacks all maxpixles in the given folder to one image.")

    arg_parser.add_argument('dir_path', nargs=1, metavar='DIR_PATH', type=str, \
        help='Path to directory with FF files.')

    arg_parser.add_argument('file_format', nargs=1, metavar='FILE_FORMAT', type=str, \
        help='File format of the image, e.g. jpg or png.')

    arg_parser.add_argument('-d', '--deinterlace', action="store_true", \
        help="""Deinterlace the image before stacking. """)

    arg_parser.add_argument('-s', '--subavg', action="store_true", \
        help="""Subtract the average image from maxpixel before stacking. """)

    arg_parser.add_argument('-f', '--flat', nargs='?', metavar='FLAT_PATH', type=str, default='', 
        help="Apply a given flat frame. If no path to the flat is given, flat.bmp from the folder will be taken.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    dir_path = cml_args.dir_path[0]

    first_img = True


    # Check if a flat was given
    flat_path = cml_args.flat

    print(flat_path)

    # Load the flat if it was given
    flat = None
    if flat_path != '':

        # Try finding the default flat
        if flat_path is None:
            flat_path = dir_path
            flat_file = 'flat.bmp'

        else:
            flat_path, flat_file = os.path.split(flat_path)

        flat_full_path = os.path.join(flat_path, flat_file)
        if os.path.isfile(flat_full_path):

            # Load the flat
            flat = loadFlat(flat_path, flat_file)

            print('Loaded flat:', flat_full_path)


    # List all FF files in the current dir
    for ff_name in os.listdir(dir_path):
        if validFFName(ff_name):

            print('Stacking: ', ff_name)

            # Load FF file
            ff = readFF(dir_path, ff_name)

            maxpixel = ff.maxpixel
            avepixel = ff.avepixel

            # Dinterlace the images
            if cml_args.deinterlace:
                maxpixel = deinterlaceBlend(maxpixel)
                avepixel = deinterlaceBlend(avepixel)

            # If the flat was given, apply it to the image, only if no subtraction is done
            if (flat is not None) and not cml_args.subavg:
                maxpixel = applyFlat(maxpixel, flat)
                avepixel = applyFlat(avepixel, flat)

            # Subtract the average from maxpixel
            if cml_args.subavg:
                img = maxpixel - avepixel

            else:
                img = maxpixel


            if first_img:
                merge_img = np.copy(img)
                first_img = False
                continue

            # Blend images 'if lighter'
            merge_img = blendLighten(merge_img, img)


    stack_path = os.path.join(dir_path, 'stacked.' + cml_args.file_format[0])

    print("Saving to:", stack_path)
    
    # Save the blended image
    scipy.misc.imsave(stack_path, merge_img)

    # Plot the blended image
    plt.imshow(merge_img, cmap='gray', vmin=0, vmax=255)

    plt.show()




