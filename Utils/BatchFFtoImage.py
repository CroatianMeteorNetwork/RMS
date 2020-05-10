""" Batch convert FF files to image files, jpg, png, etc. """

import os
import argparse

from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import validFFName
from RMS.Routines.Image import saveImage

def BatchFFtoImage(dir_path):
    # Go through all files in the given folder
    for file_name in os.listdir(dir_path):

        # Check if the file is an FF file
        if validFFName(file_name):

            # Read the FF file
            ff = readFF(dir_path, file_name)

            # Skip the file if it could not be read
            if ff is None:
                continue

            # Make a filename for the image
            img_file_name = file_name.replace('fits', '') + cml_args.file_format[0]

            print('Saving: ', img_file_name)

            # Save the maxpixel to disk
            saveImage(os.path.join(dir_path, img_file_name), ff.maxpixel)


if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Convert all FF files in a folder to images.")

    arg_parser.add_argument('dir_path', nargs=1, metavar='DIR_PATH', type=str, \
        help='Path to directory with FF files.')

    arg_parser.add_argument('file_format', nargs=1, metavar='FILE_FORMAT', type=str, \
        help='File format of the image, e.g. jpg or png.')

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    dir_path = cml_args.dir_path[0]

    BatchFFtoImage(dir_path)

