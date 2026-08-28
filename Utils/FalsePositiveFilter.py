# RPi Meteor Station
# Copyright (C) 2025  Dino Grzinic
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function, division, absolute_import

from collections import deque
from dataclasses import dataclass, field

from RMS.Logger import getLogger

# Get the logger from the main module
log = getLogger("rmslogger")


@dataclass
class Detection:
    """ Single buffered detection with taint tracking.

    Attributes:
        timestamp: [float] UTC timestamp of the detection.
        data: [object] Detection payload (list of dicts, or any data).
        filename: [str] FF filename associated with this detection.
        tainted: [bool] True if the sliding window exceeded the threshold while this detection was buffered.
    """
    timestamp: float
    data: object
    filename: str
    tainted: bool = field(default=False)


class FalsePositiveFilter:
    """ Sliding-window false positive filter for sprite detections.

    Buffers detections in a time window. If the number of detections in the window exceeds
    max_detections at any point, ALL currently buffered detections are tainted. When a detection
    ages out of the window, it is confirmed only if it was never tainted.

    This prevents bursts of false positives (e.g. from lightning storms) from being reported,
    while allowing isolated real detections through.

    Arguments:
        window_seconds: [float] Duration of the sliding window in seconds.
        max_detections: [int] Maximum allowed detections in the window before tainting.
        on_confirmed: [callable] Callback invoked for each confirmed detection.
    """

    def __init__(self, window_seconds, max_detections, on_confirmed):

        self.window_seconds = window_seconds
        self.max_detections = max_detections
        self.on_confirmed = on_confirmed
        self._buffer = deque()


    def addDetection(self, timestamp, data, filename):
        """ Add a new detection to the sliding window buffer.

        Arguments:
            timestamp: [float] UTC timestamp of the detection.
            data: [object] Detection payload.
            filename: [str] FF filename.
        """

        det = Detection(timestamp=timestamp, data=data, filename=filename)
        self._buffer.append(det)

        # If window is now saturated, taint ALL buffered detections
        if len(self._buffer) > self.max_detections:
            for d in self._buffer:
                d.tainted = True

        self._flush(timestamp)


    def tick(self, now):
        """ Advance the window without adding a detection.

        Call periodically so aged-out detections get resolved even when no new detection arrives.

        Arguments:
            now: [float] Current UTC timestamp.
        """

        self._flush(now)


    def flush(self, now):
        """ Force-flush all buffered detections by advancing the clock past the window.

        Arguments:
            now: [float] Timestamp far enough in the future to expire all buffered detections.
        """

        self._flush(now)


    def _flush(self, now):
        """ Expire detections older than the window and confirm or reject them.

        Arguments:
            now: [float] Current UTC timestamp.
        """

        cutoff = now - self.window_seconds

        while self._buffer and self._buffer[0].timestamp < cutoff:
            expired = self._buffer.popleft()

            if not expired.tainted:
                log.info("{:s} passed the filter.".format(expired.filename))
                self.on_confirmed(expired)
            else:
                log.info("{:s} rejected by filter (window saturated).".format(expired.filename))
