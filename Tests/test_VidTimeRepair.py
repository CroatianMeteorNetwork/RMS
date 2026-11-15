"""
Tests of the correction of the timestamps of UWO .vid frames from their frame counter: the timestamps are the time of
the first frame plus the counter times the frame period, and a frame whose seconds field is off by whole seconds
(the microseconds are right) gets the time given by its counter. A lost frame (a step of a frame period with the
counter going up by one) is kept as it is, and a file whose counter doesn't follow the timestamps, or whose timestamps
have a real gap or step (many lost frames), is not changed. Short inputs, whose sampled frames are dense, are
corrected as the long ones, and an input whose timestamps can't be checked gives a warning.
"""

import datetime
import logging

import numpy as np
import pytest

import RMS.ConfigReader as cr
from RMS.Astrometry.Conversions import datetime2UnixTime
from RMS.Formats import Vid
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


@pytest.mark.parametrize('after', [10, 389])
def test_gap_near_an_end_kept(tmp_path, after):

    # About a second of lost frames (32 frames) near the start or the end of the file, where few sampled frames
    #   are on one side of the gap: the timestamps are right and nothing is changed
    n = 400
    times, counters = regular(n, lost=(after,), n_lost=32)
    path = str(tmp_path/'endgap.vid')
    writeVid(path, times, counters)

    handle, read = readTimes(path, range(n))
    assert np.max(np.abs(read - times)) < 2e-6
    assert not handle.vid_repaired_frames


def test_seconds_off_next_to_lost_frame_corrected(tmp_path):

    # A frame with the seconds field off right before a lost frame (the step of the lost frame between the same
    #   two sampled frames): corrected
    n = 600
    times, counters = regular(n, lost=(250,))
    stamped = times.copy()
    stamped[250] -= 1.0
    path = str(tmp_path/'next.vid')
    writeVid(path, stamped, counters)

    handle, read = readTimes(path, range(n))
    assert np.max(np.abs(read - times)) < 2e-6
    assert len(handle.vid_repaired_frames) == 1



def stampFields(n):
    """ The seconds and microseconds fields of the timestamps of n regular frames, and their counters.

    Return:
        (ts, tu, counters): [tuple of ndarrays]
    """

    times, counters = regular(n)
    ts = np.floor(times).astype(np.int64)
    tu = np.round((times - ts)*1e6).astype(np.int64)
    carry = tu >= 1000000
    ts[carry] += 1
    tu[carry] -= 1000000

    return ts, tu, counters


def writeHeaders(path, ts, tu, counters, width=64, height=2):
    """ A vid file of blank frames with the given timestamp fields and frame counters, written at once (for
        long files): only the header of every frame is filled.
    """

    words = np.zeros((len(ts), width*height//2), dtype=np.uint32)
    words[:, :4] = [0x10000, width*height*2, 116, 0]
    words[:, 4] = counters
    words[:, 5] = ts.astype(np.int32).view(np.uint32)
    words[:, 6] = tu.astype(np.int32).view(np.uint32)
    words.view(np.int16)[:, 14:18] = [1, width, height, 16]
    words.tofile(path)


def repairAll(path, ts, tu, counters):
    """ The model of the timestamps of a vid file, and the seconds fields of all its frames after the repair.

    Return:
        (model, ts_repaired): [tuple] The model (see Vid.frameTimeModel) and the repaired seconds fields.
    """

    with open(path, 'rb') as fid:
        model = Vid.frameTimeModel(fid)

    # Every frame repaired as the vid input repairs it when the frame is read
    st = Vid.VidStruct()
    ts_repaired = ts.copy()
    for i in range(len(ts)):
        st.seq, st.ts, st.tu = int(counters[i]), int(ts[i]), int(tu[i])
        Vid.repairFrameTime(st, model)
        ts_repaired[i] = st.ts

    return model, ts_repaired


def sameModel(a, b):
    """ Whether two models of the timestamps (or None) are identical. """

    if (a is None) or (b is None):
        return (a is None) and (b is None)

    return all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))


# Lengths of the inputs and of the runs of frames with the seconds field off: the runs in the recordings are
#   mostly under 12 frames, the longest seen 75 frames. In inputs under about 1700 frames the 64 sampled
#   frames are dense, so a long run covered more than 3 of them and the repair was skipped
RUN_CASES = [(n, k) for n in (128, 300, 578, 1125, 1700, 5760, 20000) for k in (1, 10, 41, 75) if k <= n//4]


@pytest.mark.parametrize('where', ['start', 'middle', 'end'])
@pytest.mark.parametrize('n, k', RUN_CASES)
def test_run_corrected_in_short_and_long_inputs(tmp_path, n, k, where):

    # A run of k frames with the seconds field 1 s low at the start, in the middle or at the end of the input
    ts, tu, counters = stampFields(n)
    first = {'start': 0, 'middle': n//2 - k//2, 'end': n - k}[where]
    stamped = ts.copy()
    stamped[first:first + k] -= 1
    path = str(tmp_path/'run.vid')
    writeHeaders(path, stamped, tu, counters)

    # Every frame gets the seconds of its counter
    model, repaired = repairAll(path, stamped, tu, counters)
    assert model is not None
    assert np.array_equal(repaired, ts)


@pytest.mark.parametrize('n', [578, 1125])
def test_burst_of_runs_corrected_in_short_input(tmp_path, n):

    # Runs of 41 frames with the seconds field off every 64 frames (as they come in the recordings), over 300
    #   frames of a short input: every run covers several sampled frames and all are corrected
    ts, tu, counters = stampFields(n)
    stamped = ts.copy()
    for first in range(100, 400, 64):
        stamped[first:first + 41] -= 1
    path = str(tmp_path/'burst.vid')
    writeHeaders(path, stamped, tu, counters)

    model, repaired = repairAll(path, stamped, tu, counters)
    assert model is not None
    assert np.array_equal(repaired, ts)


@pytest.mark.parametrize('n, first, length', [(578, 300, 278), (1125, 400, 300), (1700, 1100, 600),
                                              (19281, 17900, 1381)])
def test_long_run_of_whole_seconds_kept(tmp_path, n, first, length):

    # A run of frames 1 s low much longer than a run of the seconds field off (here to the end of the input or
    #   in the middle of it) is a step in the timestamps: no model, and nothing is changed
    ts, tu, counters = stampFields(n)
    stamped = ts.copy()
    stamped[first:first + length] -= 1
    path = str(tmp_path/'step.vid')
    writeHeaders(path, stamped, tu, counters)

    model, repaired = repairAll(path, stamped, tu, counters)
    assert model is None
    assert np.array_equal(repaired, stamped)


def test_step_not_whole_seconds_kept(tmp_path):

    # Frames half a second early in a short input are not the seconds field off: no model, nothing is changed
    n = 578
    ts, tu, counters = stampFields(n)
    t_us = ts*1000000 + tu
    t_us[200:241] -= 500000
    stamped, stamped_tu = t_us//1000000, t_us%1000000
    path = str(tmp_path/'half.vid')
    writeHeaders(path, stamped, stamped_tu, counters)

    model, repaired = repairAll(path, stamped, stamped_tu, counters)
    assert model is None
    assert np.array_equal(repaired, stamped)


@pytest.mark.parametrize('n', [5760, 19281])
@pytest.mark.parametrize('case', ['run', 'burst', 'step'])
def test_long_input_model_unchanged(tmp_path, n, case):

    # In a long input the sampled frames are sparse and the model is the one of at most 3 consecutive sampled
    #   frames off it (the rule without the allowance for dense samples), for runs, bursts of runs and a step
    ts, tu, counters = stampFields(n)
    stamped = ts.copy()
    if case == 'run':
        stamped[3000:3075] -= 1
    elif case == 'burst':
        for first in range(2000, 3000, 64):
            stamped[first:first + 41] -= 1
    else:
        stamped[n - 700:] -= 1
    path = str(tmp_path/'long.vid')
    writeHeaders(path, stamped, tu, counters)

    with open(path, 'rb') as fid:
        model = Vid.frameTimeModel(fid)
        model_sparse = Vid.frameTimeModel(fid, max_run_frames=1)

    assert sameModel(model, model_sparse)


def test_short_file_long_run_corrected(tmp_path):

    # A run of 41 frames with the seconds field off in a short file, read by the vid input: all corrected
    n = 300
    times, counters = regular(n)
    stamped = times.copy()
    bad = list(range(130, 171))
    stamped[bad] -= 1.0
    path = str(tmp_path/'short.vid')
    writeVid(path, stamped, counters)

    handle, read = readTimes(path, range(n))
    assert np.max(np.abs(read - times)) < 2e-6
    assert len(handle.vid_repaired_frames) == len(bad)


def test_warning_without_model(tmp_path, caplog):

    # A file whose timestamps can't be checked against the counter: a warning names it
    n = 300
    times, _ = regular(n)
    counters = 1000 + np.random.default_rng(1).permutation(n)
    path = str(tmp_path/'unchecked.vid')
    writeVid(path, times, counters)

    with caplog.at_level(logging.WARNING, logger='rmslogger'):
        handle, _ = readTimes(path, range(1))

    assert handle.vid_time_model is None
    assert any((record.levelno == logging.WARNING) and ('could not be checked' in record.getMessage())
               and ('unchecked.vid' in record.getMessage()) for record in caplog.records)


def test_no_warning_with_model(tmp_path, caplog):

    # A file with a model of its timestamps gives no warning about them
    n = 300
    times, counters = regular(n)
    path = str(tmp_path/'checked.vid')
    writeVid(path, times, counters)

    with caplog.at_level(logging.WARNING, logger='rmslogger'):
        handle, _ = readTimes(path, range(1))

    assert handle.vid_time_model is not None
    assert not any('could not be checked' in record.getMessage() for record in caplog.records)
