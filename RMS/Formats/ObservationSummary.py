""" Summary text file for station and observation session
"""
from datetime import datetime

# The MIT License

# Copyright (c) 2024

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
Creates a text file summarising key station information and observation session
"""


import datetime
import git
import shutil
import platform
import os
from RMS.Misc import niceFormat, isRaspberryPi


class ObservationSummary:
    """
    Class to retrieve and hold salient data summarising information about a station and the observation session
    """
    def __init__(self):

        self.repo = git.Repo(search_parent_directories=True)
        self.date = self.repo.head.object.committed_date
        self.commit = self.repo.head.object.hexsha
        self.merge = git.Commit(self.repo, self.repo.head.object.binsha).parents[0]
        if isRaspberryPi():
            with open('/sys/firmware/devicetree/base/model', 'r') as m:
                self.hardware_version = m.read.lower()
        else:
            self.hardware_version = platform.machine()
        self.os_version = platform.platform()
        # todo: handle older versions of python
        self.storage_total, self.storage_used, self.storage_free = shutil.disk_usage("/")
        self.fits_files = None
        self.fits_missing = None
        self.minutes_missing = None
        self.capture_directories = None
        self.detections_before_ml = None
        self.detections_after_ml = None
        self.number_fits_detected = None
        self.photometry_good = None
        self.dropped_frames = None
        self.jitter_quality = None
        self.dropped_frame_rate = None
        self.last_calculated_fps = None
        self.sensor_type = None
        self.lens = None
        self.protocol = None
        self.media_backend = None


    def serialize(self, format_nicely = True):

        """

        Args:
            self:

        Returns:
            output:class represented as a string, formatted for human readability

        """

        output = ""
        output += "date: {}\n".format(datetime.datetime.fromtimestamp(self.date).strftime('%Y%m%d_%H%M%S'))
        output += "commit: {}\n".format(self.commit)
        for parent in self.merge.parents:
            output += "merge_parents: {}\n".format(parent.hexsha)
        output += "hardware_version: {}\n".format(self.hardware_version)
        output += "os_version: {}\n".format(self.os_version)
        output += "storage_total: {:.2f}GB\n".format(self.storage_total / 1024 ** 3)
        output += "storage_used: {:.2f}GB\n".format(self.storage_used / 1024 ** 3)
        output += "storage_free: {:.2f}GB\n".format(self.storage_free / 1024 ** 3)
        output += "fits_files: {}\n".format(self.fits_files)
        output += "fits_missing: {}\n".format(self.fits_missing)
        output += "minutes_missing: {}\n".format(self.minutes_missing)
        output += "capture_directories: {}\n".format(self.capture_directories)
        output += "detections_before_ml: {}\n".format(self.detections_before_ml)
        output += "detections_after_ml: {}\n".format(self.detections_after_ml)
        output += "number_fits_detected: {}\n".format(self.number_fits_detected)
        output += "number_detections_before_ml: {}\n".format(self.detections_before_ml)
        output += "number_detections_after_ml: {}\n".format(self.detections_before_ml)
        output += "number_fits_detected: {}\n".format(self.number_fits_detected)
        output += "photometry_good: {}\n".format(self.photometry_good)
        output += "dropped_frames: {}\n".format(self.dropped_frames)
        output += "jitter_quality: {}\n".format(self.jitter_quality)
        output += "dropped_frame_rate: {}%\n".format(self.jitter_quality)
        output += "last_calculated_fps: {}\n".format(self.last_calculated_fps)
        output += "sensor_type: {}\n".format(self.sensor_type)
        output += "lens: {}\n".format(self.lens)
        output += "protocol: {}\n".format(self.protocol)
        output += "media_backend: {}\n".format(self.media_backend)


        if format_nicely:
            return niceFormat(output)

        return output


    def writeToFile(self, file_path_and_name):
        with open(file_path_and_name, "w") as summary_file_handle:
            summary_file_handle.write(self.serialize())





if __name__ == "__main__":

    summary_file = ObservationSummary()
    print(summary_file.serialize())
    summary_file.writeToFile(os.path.expanduser("~/tmp/summary.txt"))