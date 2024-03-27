"""
This script automates the conversion of compressed FITS files to uncompressed FITS files. 
Designed to work recursively, it searches for all .fits files within a specified input directory 
and its subdirectories, then converts each file to an uncompressed format, preserving the original 
directory structure in a separate output directory.

Usage:
    python ConvertCompressedFits.py <input_directory> <output_directory>

Where:
    <input_directory> is the directory containing the compressed .fits files to convert.
    <output_directory> is the directory where the uncompressed .fits files will be saved.

Example:
    python ConvertCompressedFits.py /path/to/compressed /path/to/uncompressed

"""

import argparse
import os
import sys
import glob

from RMS.Formats.FFfits import read as readFFfits
from RMS.Formats.FFfits import write as writeFFfits


def findFitsFiles(directory):
    """Recursively find all FITS files in a directory and its subdirectories.

    Arguments:
        directory: [str] The path to the folder containing the files to convert.

    Return:
        None

    """
    fits_files = []
    for root, dirs, files in os.walk(directory):
        fits_files.extend(glob.glob(os.path.join(root, '*.fits')))
    return fits_files


def convertDirectoryFits(input_directory, output_directory):
    """Convert all FITS files found in the input directory to uncompressed
    files in the output directory.

    Arguments:
        input_directory: [str] The path to the folder containing the files to convert.
        output_directory: [str] The path to the folder where the converted files will be saved.

    Return:
        None

    """
    fits_files = findFitsFiles(input_directory)
    for fits_file in fits_files:
        try:
            # Assuming the path structure should be preserved in the output directory
            relative_path = os.path.relpath(fits_file, input_directory)
            output_path = os.path.join(output_directory, relative_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Read the compressed FITS file
            ff = readFFfits('', fits_file, array=True, full_filename=True)

            # Write it uncompressed
            writeFFfits(ff, os.path.dirname(output_path), os.path.basename(output_path), compress=False)
            print(f"Converted {fits_file} to {output_path}")
        except Exception as e:
            print(f"Failed to convert {fits_file}: {e}", file=sys.stderr)


if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser

    arg_parser = argparse.ArgumentParser(description="Convert all compressed FITS files in a directory (and "
                                         "subdirectories) to uncompressed FITS files in a separate directory.")
    
    arg_parser.add_argument("input_directory",
                            help="The directory to search for FITS files.")
    
    arg_parser.add_argument("output_directory",
                            help="The directory where the uncompressed FITS files will be saved.")

    cml_args = arg_parser.parse_args()

    #########################

    if not os.path.isdir(cml_args.input_directory):
        print("The specified input directory does not exist.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(cml_args.output_directory, exist_ok=True)

    convertDirectoryFits(cml_args.input_directory, cml_args.output_directory)
