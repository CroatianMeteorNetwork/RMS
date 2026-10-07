""" Convert FF FITS files between the RICE_1-compressed and the uncompressed layout.

Searches the input directory and its subdirectories for FF*.fits files and writes each one, converted,
to the same relative path under the output directory. The conversion works on the HDUs directly: the
primary header and every plane (including AVERESID/STDRESID) are carried over unchanged, only the
storage of the planes in FFfits.COMPRESSED_PLANES changes. Both layouts hold identical data.

Usage:
    python -m Utils.ConvertCompressedFits <input_directory> <output_directory> [--compress | --decompress]

Examples:
    python -m Utils.ConvertCompressedFits ~/RMS_data/CapturedFiles/XX0001_... /tmp/plain --decompress
    python -m Utils.ConvertCompressedFits ~/RMS_data/CapturedFiles/XX0001_... /tmp/rice --compress
"""

from __future__ import print_function, division, absolute_import

import argparse
import fnmatch
import os
import sys

from astropy.io import fits

from RMS.Formats.FFfits import imageHDU


def findFFFiles(directory):
    """ Recursively find all FF FITS files in a directory and its subdirectories.

    Arguments:
        directory: [str] The path to the directory to search.

    Return:
        [list] Paths of the FF FITS files found, sorted.
    """

    ff_files = []
    for root, _, files in os.walk(directory):
        for file_name in fnmatch.filter(files, 'FF*.fits'):
            ff_files.append(os.path.join(root, file_name))

    return sorted(ff_files)


def convertFF(input_path, output_path, compress):
    """ Write a copy of one FF file with its planes compressed or uncompressed.

    Arguments:
        input_path: [str] FF file to convert.
        output_path: [str] Where to write the converted file.
        compress: [bool] True to RICE_1-compress the planes that compress, False to store all planes
            uncompressed.
    """

    with fits.open(input_path, memmap=False) as hdulist:

        out = fits.HDUList([fits.PrimaryHDU(header=hdulist[0].header)])

        for hdu in hdulist[1:]:
            out.append(imageHDU(hdu.data, hdu.name, compress))

        out.writeto(output_path, overwrite=True)


def convertDirectory(input_directory, output_directory, compress=False):
    """ Convert all FF FITS files found in the input directory.

    Arguments:
        input_directory: [str] The path to the folder containing the files to convert.
        output_directory: [str] The path to the folder where the converted files will be saved.

    Keyword arguments:
        compress: [bool] If True, compress the output files. If False, decompress them.

    Return:
        [int] Number of files that failed to convert.
    """

    action = "Compressed" if compress else "Decompressed"
    failed = 0

    for ff_path in findFFFiles(input_directory):

        # Preserve the path structure in the output directory
        output_path = os.path.join(output_directory, os.path.relpath(ff_path, input_directory))

        try:
            if not os.path.isdir(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))

            convertFF(ff_path, output_path, compress)
            print("{:s} {:s} to {:s}".format(action, ff_path, output_path))

        except Exception as e:
            failed += 1
            print("Failed to convert {:s}: {}".format(ff_path, e), file=sys.stderr)

    return failed


if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Convert FF FITS files between the RICE_1-compressed "
        "and the uncompressed layout, in a directory and its subdirectories.")

    arg_parser.add_argument("input_directory", help="The directory to search for FF FITS files.")

    arg_parser.add_argument("output_directory", help="The directory where the converted files will be saved.")

    mode_group = arg_parser.add_mutually_exclusive_group()
    mode_group.add_argument("--compress", action="store_true",
        help="RICE_1-compress the planes that compress.")
    mode_group.add_argument("--decompress", action="store_true",
        help="Store all planes uncompressed (default).")

    cml_args = arg_parser.parse_args()

    #########################

    if not os.path.isdir(cml_args.input_directory):
        print("The specified input directory does not exist.", file=sys.stderr)
        sys.exit(1)

    if os.path.abspath(cml_args.input_directory) == os.path.abspath(cml_args.output_directory):
        print("The output directory must differ from the input directory.", file=sys.stderr)
        sys.exit(1)

    sys.exit(1 if convertDirectory(cml_args.input_directory, cml_args.output_directory,
        compress=cml_args.compress) else 0)
