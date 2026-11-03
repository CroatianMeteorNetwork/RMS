"""
Tests of the correction of the timestamps of UWO .vid frames from their frame counter: the timestamps are the time of
the first frame plus the counter times the frame period, and a frame whose seconds field is off by whole seconds
(the microseconds are right) gets the time given by its counter. A lost frame (a step of a frame period with the
counter going up by one) is kept as it is, and a file whose counter doesn't follow the timestamps, or whose timestamps
have a real gap or step (many lost frames), is not changed.
"""

import datetime

import numpy as np
import pytest

import RMS.ConfigReader as cr
from RMS.Astrometry.Conversions import datetime2UnixTime
from RMS.Formats.FrameInterface import InputTypeUWOVid


# Time of the first frame, and the frame period of the test files
T0 = 1792397709.013639
PERIOD = 0.031119


def writeVid(path, times, counters, width=32, height=16):
    """ A vid file of blank frames with the given timestamps (Unix s) and frame counters (UWO vid format: the
        header of every frame is in its first bytes).
    """

    with open(path, 'wb') as f:
        for t, c in zip(times, counters):

            # The timestamp as seconds and microseconds, as written by the recording
            ts = int(np.floor(t))
            tu = int(round((t - ts)*1e6))
            if tu == 1000000:
                ts, tu = ts + 1, 0

            frame = np.zeros(height*width, dtype=np.uint16)
            header = b''.join([
                np.array([0x10000, width*height*2, 116, 0, c], dtype=np.uint32).tobytes(),
                np.array([ts, tu], dtype=np.int32).tobytes(),
                np.array([1, width, height, 16], dtype=np.int16).tobytes(),
                np.array([0, 0, 0, 0], dtype=np.uint16).tobytes(),
                np.array([30000, 0], dtype=np.uint32).tobytes(),
                bytes(64)])
            data = bytearray(frame.tobytes())
            data[:len(header)] = header
            f.write(bytes(data))


def regular(n, lost=(), n_lost=1):
    """ Timestamps and counters of n frames, with n_lost frames lost after each of the given frames (n_lost more
        periods between the timestamps, the counter going up by one).
    """

    steps = np.ones(n)
    for k in lost:
        steps[k + 1] += n_lost
    steps[0] = 0
    times = T0 + PERIOD*np.cumsum(steps)
    counters = 636000 + np.arange(n)

    return times, counters


def readTimes(path, frames):
    """ The times (Unix s) of the given frames as read by the vid input (no frame_time_offset). """

    handle = InputTypeUWOVid(path, cr.Config(), chunk_frames=16)
    times = [datetime2UnixTime(handle.currentFrameTime(frame_no=i, dt_obj=True)) for i in frames]

    return handle, np.array(times)


def test_seconds_off_corrected(tmp_path):

    # Runs of frames with the seconds field 1 s low, in the middle of the file
    n = 600
    times, counters = regular(n)
    stamped = times.copy()
    bad = list(range(100, 107)) + list(range(331, 340))
    stamped[bad] -= 1.0
    path = str(tmp_path/'glitch.vid')
    writeVid(path, stamped, counters)

    # Every frame gets the time of its counter, and the corrected frames are counted
    handle, read = readTimes(path, range(n))
    assert np.max(np.abs(read - times)) < 2e-6
    assert len(handle.vid_repaired_frames) == len(bad)


def test_seconds_off_at_beginning_corrected(tmp_path):

    # The first frames of the file with the seconds field 1 s low: the beginning of the file is corrected too
    n = 400
    times, counters = regular(n)
    stamped = times.copy()
    stamped[:9] -= 1.0
    path = str(tmp_path/'start.vid')
    writeVid(path, stamped, counters)

    handle, read = readTimes(path, range(n))
    assert np.max(np.abs(read - times)) < 2e-6
    assert datetime2UnixTime(handle.beginning_datetime) == pytest.approx(times[0], abs=2e-6)


def test_lost_frame_kept(tmp_path):

    # A lost frame: the timestamps step by two periods there, the counter by one, and nothing is changed
    n = 400
    times, counters = regular(n, lost=(250,))
    path = str(tmp_path/'lost.vid')
    writeVid(path, times, counters)

    handle, read = readTimes(path, range(n))
    assert np.max(np.abs(read - times)) < 2e-6
    assert not handle.vid_repaired_frames


def test_counter_not_following_times(tmp_path):

    # A counter which doesn't follow the timestamps: no model, and the timestamps (one 1 s off) are kept
    n = 300
    times, _ = regular(n)
    counters = 1000 + np.random.default_rng(1).permutation(n)
    stamped = times.copy()
    stamped[150] -= 1.0
    path = str(tmp_path/'counter.vid')
    writeVid(path, stamped, counters)

    handle, read = readTimes(path, range(n))
    assert handle.vid_time_model is None
    assert np.max(np.abs(read - stamped)) < 2e-6


@pytest.mark.parametrize('n_lost', [30, 32])
def test_many_lost_frames_kept(tmp_path, n_lost):

    # Many lost frames: the timestamps after the gap are later by close to a second (30 frames, 0.93 s) or by
    #   almost exactly one (32 frames), and are right; the repair is off and nothing is changed
    n = 600
    times, counters = regular(n, lost=(299,), n_lost=n_lost)
    path = str(tmp_path/'gap.vid')
    writeVid(path, times, counters)

    handle, read = readTimes(path, range(n))
    assert handle.vid_time_model is None
    assert np.max(np.abs(read - times)) < 2e-6
    assert not handle.vid_repaired_frames


def test_seconds_off_with_lost_frame_corrected(tmp_path):

    # A lost frame and runs of frames with the seconds field 1 s low before and after it: all corrected
    n = 600
    times, counters = regular(n, lost=(250,))
    stamped = times.copy()
    bad = list(range(120, 127)) + list(range(410, 418))
    stamped[bad] -= 1.0
    path = str(tmp_path/'lostglitch.vid')
    writeVid(path, stamped, counters)

    handle, read = readTimes(path, range(n))
    assert np.max(np.abs(read - times)) < 2e-6
    assert len(handle.vid_repaired_frames) == len(bad)
