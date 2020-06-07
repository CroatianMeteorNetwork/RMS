""" Stacks all images in the given folder to one image. """

from __future__ import print_function, division, absolute_import

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from RMS.Routines.Image import loadImage, saveImage
from Utils.StackFFs import deinterlaceBlend, blendLighten





if __name__ == '__main__':

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Stacks all images of the given type in the given folder.")

    arg_parser.add_argument('dir_path', nargs=1, metavar='DIR_PATH', type=str, \
        help='Path to directory with FF files.')

    arg_parser.add_argument('input_file_format', nargs=1, metavar='INPUT_FILE_FORMAT', type=str, \
        help='File format of input images, e.g. jpg or png.')

    arg_parser.add_argument('output_file_format', nargs=1, metavar='OUTPUT_FILE_FORMAT', type=str, \
        help='File format of the output stacked image, e.g. jpg or png.')

    arg_parser.add_argument('-d', '--deinterlace', action="store_true", help="""Deinterlace the image before stacking. """)


    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    dir_path = cml_args.dir_path[0]

    first_img = True

    # List all FF files in the current dir
    for ff_name in os.listdir(dir_path):
        if ff_name.endswith('.' + cml_args.input_file_format[0]):

            print('Stacking: ', ff_name)

            # Load the image
            img = loadImage(os.path.join(dir_path, ff_name), -1)


            # Deinterlace the image
            if cml_args.deinterlace:
                img = deinterlaceBlend(img)


            if first_img:
                merge_img = np.copy(img)
                first_img = False
                continue

            # Blend images 'if lighter'
            merge_img = blendLighten(merge_img, img)


    stack_path = os.path.join(dir_path, 'stacked.' + cml_args.output_file_format[0])

    print("Saving to:", stack_path)
    
    # Save the blended image
    saveImage(stack_path, merge_img)

    # Plot the blended image
    plt.imshow(merge_img, cmap='gray')

    plt.show()




