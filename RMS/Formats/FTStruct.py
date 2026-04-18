""" Definition of an FT file structure. Helps store timestamps of individual frames for raw mkv segment capture """

from __future__ import print_function, division, absolute_import

class FTStruct:
    """ Default structure for an FT file to store frame timestamps.
    """

    def __init__(self):
        # List of tuples: [(frame_number, timestamp), ...]
        self.timestamps = []
        # Raw VENC PTS in microseconds (from pts_stream side-door).
        # Intervals between entries = true sensor frame period (crystal
        # jitter only, no clock model).  Empty for non-VENC captures.
        self.raw_pts_us = []
        # GStreamer buffer PTS in nanoseconds (post-jitter-buffer).
        # Shared key between MKV frames and FT entries — same value
        # seen by both appsink and splitmuxsink before the tee.
        # Empty for non-GStreamer captures.
        self.gst_pts_ns = []

    def __repr__(self):
        return "FTStruct with timestamp for {:d} frames.\n".format(len(self.timestamps))
