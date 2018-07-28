""" Batch convert FF files to image files, jpg, png, etc. """

import sys
import os
import argparse

import scipy.misc

from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import validFFName




if __name__ == "__main__":

	### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Convert all FF files in a folder to images.")

    arg_parser.add_argument('dir_path', nargs=1, metavar='DIR PATH', type=str, \
        help='Path to the FR bin file.')

    arg_parser.add_argument('file_format', nargs=1, metavar='FILE FORMAT', type=str, \
        help='File format of the image, e.g. jpg or png.')

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    dir_path = cml_args.dir_path[0]

    # Go through all files in the given folder
    for file_name in os.listdir(dir_path):

    	# Check if the file is an FF file
    	if validFFName(file_name):

    		# Read the FF file
    		ff = readFF(dir_path, file_name)

    		# Make a filename for the image
    		img_file_name = file_name.replace('fits', '') + cml_args.file_format[0]

    		print('Saving: ', img_file_name)

    		# Save the maxpixel to disk
    		scipy.misc.imsave(os.path.join(dir_path, img_file_name), ff.maxpixel)

