""" Functions for reading/writing FT files in .bin format """


from __future__ import print_function, division, absolute_import

import os
import struct

import numpy as np

from RMS.Formats.FTStruct import FTStruct


def read(directory, filename):
    """ Read FT*.bin file from the specified directory.

    Supports three formats:
        v1 (legacy): [n_frames:u32] [frame_num:u32, utc:f64] ...
        v2:          [0xFE020000:u32] [n_frames:u32]
                     [frame_num:u32, utc:f64, raw_pts_us:f64] ...
        v3:          [0xFE030000:u32] [n_frames:u32]
                     [frame_num:u32, utc:f64, raw_pts_us:f64, gst_pts_ns:u64] ...

    v2 adds raw_pts_us: the VENC hardware PTS in microseconds.
    v3 adds gst_pts_ns: GStreamer buffer PTS in nanoseconds (post-jitter-buffer).
    gst_pts_ns is the shared key between MKV frames and FT entries.

    Args:
        directory: [str] Path to directory containing file.
        filename: [str] Name of FT*.bin file.

    Returns:
        ft: FTStruct object populated with data from the file.
    """

    filepath = os.path.join(directory, filename)

    with open(filepath, "rb") as ft_file:
        ft = FTStruct()

        # Read first word — magic (v2/v3) or n_frames (v1)
        first_word = int(np.fromfile(ft_file, dtype=np.uint32, count=1))

        if first_word == 0xFE030000:
            # v3 format: magic, n_frames, then (frame_num, utc, raw_pts_us, gst_pts_ns)
            n_frames = int(np.fromfile(ft_file, dtype=np.uint32, count=1))
            for _ in range(n_frames):
                frame_number = int(np.fromfile(ft_file, dtype=np.uint32, count=1))
                timestamp = float(np.fromfile(ft_file, dtype=np.float64, count=1))
                raw_pts_us = float(np.fromfile(ft_file, dtype=np.float64, count=1))
                gst_pts = int(np.fromfile(ft_file, dtype=np.uint64, count=1))
                ft.timestamps.append((frame_number, timestamp))
                ft.raw_pts_us.append(raw_pts_us)
                ft.gst_pts_ns.append(gst_pts)
        elif first_word == 0xFE020000:
            # v2 format: magic, n_frames, then (frame_num, utc, raw_pts_us)
            n_frames = int(np.fromfile(ft_file, dtype=np.uint32, count=1))
            for _ in range(n_frames):
                frame_number = int(np.fromfile(ft_file, dtype=np.uint32, count=1))
                timestamp = float(np.fromfile(ft_file, dtype=np.float64, count=1))
                raw_pts_us = float(np.fromfile(ft_file, dtype=np.float64, count=1))
                ft.timestamps.append((frame_number, timestamp))
                ft.raw_pts_us.append(raw_pts_us)
        else:
            # v1 format: first_word is n_frames
            n_frames = first_word
            for _ in range(n_frames):
                frame_number = int(np.fromfile(ft_file, dtype=np.uint32, count=1))
                timestamp = float(np.fromfile(ft_file, dtype=np.float64, count=1))
                ft.timestamps.append((frame_number, timestamp))

    return ft


def write(ft, directory, filename):
    """ Write FT structure to a .bin file in the specified directory.

    Uses v3 if gst_pts_ns is available, v2 if only raw_pts_us, v1 otherwise.

    Args:
        ft: FTStruct object containing data to write.
        directory: [str] Path to the directory where the file will be written.
        filename: [str] Name of the file which will be written.
    """

    ft_full_path = os.path.join(directory, filename)

    with open(ft_full_path, "wb") as ft_file:

        n_frames = len(ft.timestamps)
        has_raw_pts = hasattr(ft, 'raw_pts_us') and len(ft.raw_pts_us) == n_frames
        has_gst_pts = hasattr(ft, 'gst_pts_ns') and len(ft.gst_pts_ns) == n_frames

        if has_gst_pts:
            # v3: magic + n_frames + (frame_num, utc, raw_pts_us, gst_pts_ns)
            ft_file.write(struct.pack("I", 0xFE030000))
            ft_file.write(struct.pack("I", n_frames))
            for i, (frame_number, timestamp) in enumerate(ft.timestamps):
                ft_file.write(struct.pack("I", frame_number))
                ft_file.write(struct.pack("d", timestamp))
                raw = ft.raw_pts_us[i] if has_raw_pts else 0.0
                ft_file.write(struct.pack("d", raw))
                ft_file.write(struct.pack("Q", ft.gst_pts_ns[i]))
        elif has_raw_pts:
            # v2: magic + n_frames + (frame_num, utc, raw_pts_us) per frame
            ft_file.write(struct.pack("I", 0xFE020000))
            ft_file.write(struct.pack("I", n_frames))
            for i, (frame_number, timestamp) in enumerate(ft.timestamps):
                ft_file.write(struct.pack("I", frame_number))
                ft_file.write(struct.pack("d", timestamp))
                ft_file.write(struct.pack("d", ft.raw_pts_us[i]))
        else:
            # v1: n_frames + (frame_num, utc) per frame
            ft_file.write(struct.pack("I", n_frames))
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
        print("Test passed: v1 round-trip OK.")

        # v2 round-trip
        v2_ft = FTStruct()
        v2_ft.timestamps = [(1, 0.033), (2, 0.066)]
        v2_ft.raw_pts_us = [1000.0, 41030.0]
        write(v2_ft, temp_dir, "FT_v2.bin")
        v2_loaded = read(temp_dir, "FT_v2.bin")
        assert v2_loaded.timestamps == v2_ft.timestamps
        assert v2_loaded.raw_pts_us == v2_ft.raw_pts_us
        assert v2_loaded.gst_pts_ns == []
        print("Test passed: v2 round-trip OK.")

        # v3 round-trip
        v3_ft = FTStruct()
        v3_ft.timestamps = [(10, 1000.033), (11, 1000.073)]
        v3_ft.raw_pts_us = [5000000.0, 5040030.0]
        v3_ft.gst_pts_ns = [200000000000, 200040030000]
        write(v3_ft, temp_dir, "FT_v3.bin")
        v3_loaded = read(temp_dir, "FT_v3.bin")
        assert v3_loaded.timestamps == v3_ft.timestamps
        assert v3_loaded.raw_pts_us == v3_ft.raw_pts_us
        assert v3_loaded.gst_pts_ns == v3_ft.gst_pts_ns
        print("Test passed: v3 round-trip OK.")