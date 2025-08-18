""" Converts an FS bin file to CSV. """

from __future__ import print_function, division, absolute_import

import os
import argparse

from RMS.Formats.FieldIntensities import convertFieldIntensityBinToTxt


if __name__ == "__main__":

	### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Convert a field sum FS file into a txt file.")

    arg_parser.add_argument('file_path', metavar='FILE', type=str, help='Path to an FS file.')

    arg_parser.add_argument('-d', '--deinterlace', action='store_true', help='Deinterlaced half-frames will be returned.')

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    dir_path, file_name = os.path.split(cml_args.file_path)

    convertFieldIntensityBinToTxt(dir_path, file_name, deinterlace=cml_args.deinterlace)