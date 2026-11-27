""" Functions for reading/writing FT files in .bin format """


from __future__ import print_function, division, absolute_import

import os
import struct

import numpy as np

from RMS.Formats.FTStruct import FTStruct


# One entry of an FT file: the frame counter and the Unix time of the frame, packed as write() writes them
FT_ENTRY_DTYPE = np.dtype([('frame', np.uint32), ('time', np.float64)])


def readBuffer(data):
    """ Read the contents of an FT*.bin file (e.g. a member of an archive read without extracting it).

    Arguments:
        data: [bytes] Contents of the file.

    Return:
        (frame_numbers, timestamps): [tuple of ndarray] The frame counters and the Unix times of the frames.
    """

    # The number of frames, then the entries
    n_frames = int(np.frombuffer(data, dtype=np.uint32, count=1)[0])
    if len(data) < 4 + n_frames*FT_ENTRY_DTYPE.itemsize:
        raise ValueError("The FT file is shorter than its {:d} frames".format(n_frames))

    entries = np.frombuffer(data, dtype=FT_ENTRY_DTYPE, count=n_frames, offset=4)

    return entries['frame'].astype(np.int64), entries['time'].astype(np.float64)


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
        frame_numbers, timestamps = readBuffer(ft_file.read())

    ft = FTStruct()
    ft.timestamps = list(zip(frame_numbers.tolist(), timestamps.tolist()))

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