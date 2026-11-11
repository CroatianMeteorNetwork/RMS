"""
This script converts FITS files between compressed and uncompressed formats.
Designed to work recursively, it searches for all .fits files within a specified input directory
and its subdirectories, then converts each file preserving the original directory structure
in a separate output directory.

Usage:
    python ConvertCompressedFits.py <input_directory> <output_directory> [--compress | --decompress]

Where:
    <input_directory> is the directory containing the .fits files to convert.
    <output_directory> is the directory where the converted .fits files will be saved.
    --compress: Convert uncompressed FITS to compressed (RICE_1) format.
    --decompress: Convert compressed FITS to uncompressed format (default).

Examples:
    python ConvertCompressedFits.py /path/to/uncompressed /path/to/compressed --compress
    python ConvertCompressedFits.py /path/to/compressed /path/to/uncompressed --decompress

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
        directory: [str] The path to the directory to search.

    Return:
        [list] List of paths to FITS files found.

    """
    fits_files = []
    for root, dirs, files in os.walk(directory):
        fits_files.extend(glob.glob(os.path.join(root, '*.fits')))
    return fits_files


def convertDirectoryFits(input_directory, output_directory, compress=False):
    """Convert all FITS files found in the input directory.

    Arguments:
        input_directory: [str] The path to the folder containing the files to convert.
        output_directory: [str] The path to the folder where the converted files will be saved.
        compress: [bool] If True, compress the output files. If False, decompress them.

    Return:
        None

    """
    fits_files = findFitsFiles(input_directory)
    action = "Compressed" if compress else "Decompressed"

    for fits_file in fits_files:
        try:
            # Preserve the path structure in the output directory
            relative_path = os.path.relpath(fits_file, input_directory)
            output_path = os.path.join(output_directory, relative_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Read the FITS file
            ff = readFFfits('', fits_file, array=True, full_filename=True)

            # Write with specified compression setting
            writeFFfits(ff, os.path.dirname(output_path), os.path.basename(output_path), compress=compress)
            print(f"{action} {fits_file} to {output_path}")
        except Exception as e:
            print(f"Failed to convert {fits_file}: {e}", file=sys.stderr)


if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser

    arg_parser = argparse.ArgumentParser(description="Convert FITS files between compressed and uncompressed "
                                         "formats in a directory (and subdirectories).")

    arg_parser.add_argument("input_directory",
                            help="The directory to search for FITS files.")

    arg_parser.add_argument("output_directory",
                            help="The directory where the converted FITS files will be saved.")

    mode_group = arg_parser.add_mutually_exclusive_group()
    mode_group.add_argument("--compress", action="store_true",
                            help="Compress FITS files using RICE_1 algorithm.")
    mode_group.add_argument("--decompress", action="store_true", default=True,
                            help="Decompress FITS files (default).")

    cml_args = arg_parser.parse_args()

    #########################

    if not os.path.isdir(cml_args.input_directory):
        print("The specified input directory does not exist.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(cml_args.output_directory, exist_ok=True)

    convertDirectoryFits(cml_args.input_directory, cml_args.output_directory, compress=cml_args.compress)
