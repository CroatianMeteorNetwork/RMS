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

    import tempfile
    
    # Temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        test_filename = "FT_test.bin"
        test_filepath = os.path.join(temp_dir, test_filename)

        # Create a sample FTStruct object
        original_ft = FTStruct()
        original_ft.timestamps = [
            (1, 0.033),
            (2, 0.066),
            (3, 0.099),
            (4, 0.132),
        ]

        # Write the FTStruct to a file
        print("Writing FT file to {}".format(test_filepath))
        write(original_ft, temp_dir, test_filename)

        # Read the FTStruct back from the file
        print("Reading FT file from {}".format(test_filepath))
        loaded_ft = read(temp_dir, test_filename)

        # Verify the content matches
        assert original_ft.timestamps == loaded_ft.timestamps, "Timestamps do not match!"
        print("Test passed: Written and read timestamps match.")