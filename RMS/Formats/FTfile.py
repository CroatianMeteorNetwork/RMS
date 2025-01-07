""" Functions for reading/writing FT files in .bin format """


from __future__ import print_function, division, absolute_import

import os
import struct

import numpy as np

from RMS.Formats.FTStruct import FTStruct


def read(directory, filename):
    """ Read FT*.bin file from the specified directory.

    Args:
        directory: [str] Path to directory containing file.
        filename: [str] Name of FT*.bin file.

    Returns:
        ft: FTStruct object populated with data from the file.
    """

    filepath = os.path.join(directory, filename)

    with open(filepath, "rb") as ft_file:
        ft = FTStruct()
        
        # Read number of frames
        n_frames = int(np.fromfile(ft_file, dtype=np.uint32, count=1))

        # Read timestamps
        for _ in range(n_frames):
            frame_number = int(np.fromfile(ft_file, dtype=np.uint32, count=1))
            timestamp = float(np.fromfile(ft_file, dtype=np.float64, count=1))
            ft.timestamps.append((frame_number, timestamp))

    return ft


def write(ft, directory, filename):
    """ Write FT structure to a .bin file in the specified directory.

    Args:
        ft: FTStruct object containing data to write.
        directory: [str] Path to the directory where the file will be written.
        filename: [str] Name of the file which will be written.
    """

    ft_full_path = os.path.join(directory, filename)

    with open(ft_full_path, "wb") as ft_file:

        # Write number of frames
        n_frames = len(ft.timestamps)
        ft_file.write(struct.pack("I", n_frames))

        # Write each frame number and timestamp
        for frame_number, timestamp in ft.timestamps:
            ft_file.write(struct.pack("I", frame_number))
            ft_file.write(struct.pack("d", timestamp))     


if __name__ == '__main__':
    pass

    # ### TEST ###

    # # Load a .bin file
    # dir_path = "D:/Dropbox/RPi_Meteor_Station/samples/sample_bins"

    # file_name = 'FF453_20150620_201239_920_0058880.bin'

    # # Read the FF file
    # ff = read(dir_path, file_name)

    # print(ff)

    # # file_name_fits = file_name.replace('.bin', '.fits')

    # # # Write the read FF file as FITS
    # # write(ff, dir_path, file_name_fits, fmt='fits')

    # # # Read the FFfits
    # # ff = read(dir_path, file_name_fits)


    # # Save as new CAMS format
    # file_name_new_bin = file_name[:-5] + '_new.bin'

    # write(ff, dir_path, file_name_new_bin, fmt='bin')

    # # Read the newly written file
    # ff = read(dir_path, file_name_new_bin)


    # import matplotlib.pyplot as plt

    # plt.imshow(ff.maxpixel, cmap='gray', vmin=0, vmax=255)
    # plt.show()

    # print(ff)
