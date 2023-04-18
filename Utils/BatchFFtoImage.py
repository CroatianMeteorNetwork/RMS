""" Batch convert FF files to image files, jpg, png, etc. """

import os
import argparse

from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import validFFName
from RMS.Routines.Image import saveImage


def batchFFtoImage(dir_path, fmt, add_timestamp=False, ff_component='maxpixel'):
    """ Batch convert FF files to image files, jpg, png, etc.

    Arguments:
        dir_path: [str] Path to the directory with FF files.
        fmt: [str] File format of the image, e.g. jpg or png.

    Keyword arguments:
        add_timestamp: [bool] Add a timestamp to the image.
        ff_component: [str] FF component to save, e.g. maxpixel, avepixel, stdpixel, maxframe.

    """

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
            img_file_name = file_name.replace('fits', '') + fmt

            print('Saving: ', img_file_name)

            # Save the maxpixel to disk
            saveImage(os.path.join(dir_path, img_file_name), getattr(ff, ff_component), add_timestamp)


if __name__ == "__main__":

    # COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Convert all FF files in a folder to images.")

    arg_parser.add_argument('dir_path', nargs=1, metavar='DIR_PATH', type=str,
        help='Path to directory with FF files.')

    arg_parser.add_argument('file_format', nargs=1, metavar='FILE_FORMAT', type=str,
        help='File format of the image, e.g. jpg or png.')

    arg_parser.add_argument('-t', '--add_timestamp', action="store_true",
        help="""Add a title to the image. """)


    # Set up an exclusive group for the image type
    image_type_group = arg_parser.add_mutually_exclusive_group()

    image_type_group.add_argument('-a', '--avepixel', action="store_true",
        help="""Save the average pixel image instead of the maxpixel.""")

    image_type_group.add_argument('-s', '--stdpixel', action="store_true",
        help="""Save the standard deviation pixel image instead of the maxpixel.""")

    image_type_group.add_argument('-f', '--maxframe', action="store_true",
        help="""Save the maxfrmae pixel image instead of the maxpixel.""")
        
    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    # Get the FF component to save
    ff_component = 'maxpixel'
    if cml_args.avepixel:
        ff_component = 'avepixel'
    
    elif cml_args.stdpixel:
        ff_component = 'stdpixel'

    elif cml_args.maxframe:
        ff_component = 'maxframe'

    dir_path = cml_args.dir_path[0]
    add_timestamp = cml_args.add_timestamp

    batchFFtoImage(dir_path, cml_args.file_format[0], add_timestamp, ff_component=ff_component)
