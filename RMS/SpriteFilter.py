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

""" Burst filter for sprite detections.

    Real sprites are rare on any one camera, while lightning, clouds lit by lightning and passing lights
    produce many detections within seconds. Every FF with detections is buffered in a sliding window of
    window_sec (60 s by default). If more than max_detections (3) buffered FFs ever fall in the window at
    once, every entry buffered at that moment is tainted. An entry is confirmed when it leaves the window
    untainted (on_confirmed) and rejected (on_rejected) when it leaves tainted; every entry reaches exactly
    one of the two callbacks, exactly once.

    The filter is driven only by FF start times, never by the wall clock, so the result does not depend on
    the machine speed or on the processing delay. The event clock is the largest FF time seen and never
    goes back.

    Latency: an entry is confirmed only when the clock passes its time plus window_sec. With tick() called
    for every FF (one per 10.24 s at 25 fps), a detection is confirmed 60 to 70 s of FF time after its FF
    started. Call flush() at the end of the night to confirm what is left.

    filterCandidatesDeterministic() gives the same decisions for a whole set of timestamps at once, in any
    order, e.g. for reprocessing.
"""

from __future__ import print_function, division, absolute_import

import bisect

import numpy as np

from RMS.Logger import getLogger


# Get the logger from the main module
log = getLogger("rmslogger")


# Sliding window length (seconds) and the largest number of FFs with detections allowed in it
FP_WINDOW_SEC = 60.0
FP_MAX_DETECTIONS = 3



class SpriteCandidate(object):
    """ One FF with detections, buffered in the false-positive filter.

    Arguments:
        timestamp: [float] FF start time, unix seconds.
        ff_name: [str] FF file name.
        detections: [object] Detection records of this FF, passed through untouched.

    Keyword arguments:
        tainted: [bool] True once the window was saturated while this entry was buffered. False by default.
    """

    __slots__ = ("timestamp", "ff_name", "detections", "tainted")

    def __init__(self, timestamp, ff_name, detections, tainted=False):

        self.timestamp = float(timestamp)
        self.ff_name = ff_name
        self.detections = detections
        self.tainted = tainted


    def __repr__(self):

        return "SpriteCandidate(timestamp={:.3f}, ff_name={:s}, tainted={:s})".format(
            self.timestamp, str(self.ff_name), str(self.tainted))



class SpriteFalsePositiveFilter(object):
    """ Sliding-window burst filter, driven by FF start times.

    Arguments:
        on_confirmed: [callable] Called with each confirmed SpriteCandidate, exactly once per entry.

    Keyword arguments:
        on_rejected: [callable or None] Called with each burst-rejected SpriteCandidate, exactly once per
            entry. Every entry ends up in exactly one of the two callbacks. None by default.
        window_sec: [float] Window length in seconds. FP_WINDOW_SEC by default.
        max_detections: [int] Largest number of buffered FFs allowed in the window. FP_MAX_DETECTIONS by
            default.
    """

    def __init__(self, on_confirmed, on_rejected=None, window_sec=FP_WINDOW_SEC,
                 max_detections=FP_MAX_DETECTIONS):

        self.on_confirmed = on_confirmed
        self.on_rejected = on_rejected
        self.window_sec = float(window_sec)
        self.max_detections = int(max_detections)

        # Buffered candidates sorted by timestamp, and the event clock
        self._buffer = []
        self._clock = float("-inf")


    def addCandidate(self, timestamp, ff_name, detections):
        """ Buffer an FF with detections and advance the clock to its start time.

        Arguments:
            timestamp: [float] FF start time, unix seconds.
            ff_name: [str] FF file name.
            detections: [object] Detection records of this FF.
        """

        candidate = SpriteCandidate(timestamp, ff_name, detections)

        # Expire the old entries first, so only entries really inside the window are counted
        self._advance(candidate.timestamp)

        # Keep the buffer sorted even if an FF arrives out of order
        keys = [c.timestamp for c in self._buffer]
        self._buffer.insert(bisect.bisect_right(keys, candidate.timestamp), candidate)

        # An out-of-order entry may already be older than the window, it must not count towards a burst
        self._expire(self._clock - self.window_sec)

        # A burst taints everything buffered at this moment
        if len(self._buffer) > self.max_detections:
            for c in self._buffer:
                c.tainted = True


    def tick(self, timestamp):
        """ Advance the event clock without adding a candidate, e.g. for every FF without detections.

        Arguments:
            timestamp: [float] FF start time, unix seconds.
        """

        self._advance(timestamp)


    def flush(self):
        """ Resolve every buffered entry now, as if the clock were infinitely far ahead.

            The clock itself is not moved, so the filter can keep being used afterwards.
        """

        self._expire(float("inf"))


    def pending(self):
        """ Return the number of buffered, unresolved entries.

        Return:
            [int] Number of entries in the buffer.
        """

        return len(self._buffer)


    def _advance(self, timestamp):
        """ Move the clock forward (never back) and resolve the entries that left the window.

        Arguments:
            timestamp: [float] New clock value, unix seconds.
        """

        self._clock = max(self._clock, float(timestamp))
        self._expire(self._clock - self.window_sec)


    def _expire(self, cutoff):
        """ Confirm or reject the entries older than the cutoff.

        Arguments:
            cutoff: [float] Entries with a timestamp strictly below this leave the window.
        """

        while self._buffer and (self._buffer[0].timestamp < cutoff):

            expired = self._buffer.pop(0)

            if expired.tainted:
                log.info("{:s} rejected by the sprite burst filter (window saturated)".format(
                    str(expired.ff_name)))
                callback = self.on_rejected
            else:
                log.info("{:s} passed the sprite burst filter".format(str(expired.ff_name)))
                callback = self.on_confirmed

            if callback is None:
                continue

            # The entry is already out of the buffer, a failing callback cannot see it twice
            try:
                callback(expired)
            except Exception as e:
                log.error("Sprite burst filter callback failed for {:s}: {:s}".format(
                    str(expired.ff_name), repr(e)))



def filterCandidatesDeterministic(timestamps, window_sec=FP_WINDOW_SEC, max_detections=FP_MAX_DETECTIONS):
    """ Decide the whole set of candidates at once, independently of their order.

        This reproduces the streaming filter fed with the timestamps in sorted order. The streaming filter
        taints entry i when, at some FF time c with t_i <= c <= t_i + window_sec, more than max_detections
        entries have a time in [c - window_sec, c]. Here the count for every candidate time c is found with
        np.searchsorted on both window edges, and entry i is rejected if any candidate time in
        [t_i, t_i + window_sec] saturates its window.

    Arguments:
        timestamps: [array like] FF start times of the FFs with detections, unix seconds, in any order.

    Keyword arguments:
        window_sec: [float] Window length in seconds. FP_WINDOW_SEC by default.
        max_detections: [int] Largest number of FFs allowed in the window. FP_MAX_DETECTIONS by default.

    Return:
        [ndarray of bool] accepted, in the order of the input.
    """

    t = np.asarray(timestamps, dtype=np.float64).ravel()
    if t.size == 0:
        return np.zeros(0, dtype=bool)

    t_sorted = np.sort(t)

    # Number of entries in the window [c - W, c] ending at each candidate time c
    count = np.searchsorted(t_sorted, t_sorted, side="right") \
        - np.searchsorted(t_sorted, t_sorted - window_sec, side="left")
    saturated = count > max_detections

    # Prefix sum of the saturated window ends, to query any index range in O(1)
    sat_cum = np.concatenate([[0], np.cumsum(saturated)])

    # Window ends that can taint entry i lie in [t_i, t_i + W]
    lo = np.searchsorted(t_sorted, t, side="left")
    hi = np.searchsorted(t_sorted, t + window_sec, side="right")
    tainted = (sat_cum[hi] - sat_cum[lo]) > 0

    return ~tainted
