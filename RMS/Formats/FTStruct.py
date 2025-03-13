""" Definition of an FT file structure. Helps store timestamps of individual frames for raw mkv segment capture """

from __future__ import print_function, division, absolute_import

class FTStruct:
    """ Default structure for an FT file to store frame timestamps.
    """

    def __init__(self):
        # List of tuples: [(frame_number, timestamp), ...]
        self.timestamps = []

    def __repr__(self):
        return "FTStruct with timestamp for {:d} frames.\n".format(len(self.timestamps))
