""" Tests of captureDuration, the start time and the duration of the capture.

    On the last night before the polar day the Sun rises and then does not set for weeks, so there is a next
    sunrise but no next sunset. Next to the poles ephem can also report that the Sun never rises while it
    skims the horizon. captureDuration looked up the next sunset without guarding against either case, so
    it raised and StartCapture exited at high-latitude stations.
"""

from __future__ import print_function, division, absolute_import

import datetime

import pytest

from RMS.CaptureDuration import captureDuration


# Windows in which there is a next sunrise but ephem cannot find the next sunset:
# (latitude, longitude, first time, last time, step in minutes, continuous_capture)
POLAR_WINDOWS = [
    pytest.param(72.0, 15.0, (2026, 4, 20, 22, 0), (2026, 4, 21, 0, 0), 2, None, id='72N'),
    pytest.param(74.0, 15.0, (2026, 4, 15, 22, 30), (2026, 4, 15, 23, 30), 2, None, id='74N'),
    pytest.param(76.0, 15.0, (2026, 4, 9, 22, 10), (2026, 4, 9, 23, 50), 2, None, id='76N'),
    pytest.param(80.0, 15.0, (2026, 3, 30, 22, 30), (2026, 3, 30, 23, 30), 2, None, id='80N'),
    pytest.param(80.0, 15.0, (2026, 3, 21, 22, 10), (2026, 3, 21, 23, 58), 2, True, id='80N-continuous'),
    pytest.param(-77.85, 166.67, (2026, 10, 8, 11, 40), (2026, 10, 8, 13, 30), 2, None, id='McMurdo'),
    pytest.param(89.9, 0.0, (2026, 3, 5, 9, 0), (2026, 3, 5, 10, 0), 2, None, id='89.9N-March'),
    pytest.param(89.9, 0.0, (2026, 10, 6, 23, 30), (2026, 10, 8, 6, 30), 10, None, id='89.9N-October'),
    ]


@pytest.mark.parametrize('lat, lon, t_beg, t_end, step, continuous', POLAR_WINDOWS)
def testCaptureDurationLastNightBeforePolarDay(lat, lon, t_beg, t_end, step, continuous):
    """ captureDuration must not raise when the next sunset does not exist, and must return a sane start
        time and duration throughout the window.
    """

    t = datetime.datetime(*t_beg)
    t_end = datetime.datetime(*t_end)

    # Step through the window, every time must give a capture
    while t <= t_end:

        start_time, duration = captureDuration(lat, lon, 0, current_time=t, continuous_capture=continuous)

        # The capture lasts a positive time, capped to the default maximum of 23 hours
        assert 0 < duration <= 23*3600, (t, start_time, duration)

        # The capture starts either right away or at a time which is not in the past
        if isinstance(start_time, bool):
            assert start_time is True, (t, start_time)

        else:
            assert start_time >= t - datetime.timedelta(minutes=1), (t, start_time)

        t += datetime.timedelta(minutes=step)
