"""
Tests of the correction of the timestamps of UWO PNG frame dumps from their frame counter: as in the vid files, the
header of every frame (in its first row) carries its timestamp and a frame counter, and a frame whose seconds field is
off by whole seconds (the microseconds are right) gets the time given by its counter. A lost frame is kept as it is,
and dumps whose timestamps have a real gap (many lost frames) or whose counter doesn't follow the timestamps, and
PNGs which are not frame dumps, are not changed.
Dense bursts of runs in long dumps are corrected from the headers between the sampled frames, with few more
images read.
"""

import datetime
import logging
import os

import cv2
import numpy as np
import pytest

import RMS.ConfigReader as cr
from RMS.Astrometry.Conversions import datetime2UnixTime
import RMS.Formats.FrameInterface as FrameInterface
from RMS.Formats.FrameInterface import InputTypeImages, UWO_MAGICK_EMCCD
from RMS.Formats.FrameInterface import MAX_DENSE_PNG_HEADERS


# Time of the first frame, and the frame period of the test frames
T0 = 1792397709.013639
PERIOD = 0.031119


def writeDumps(path, times, counters, width=32, height=16):
    """ A directory of UWO PNG frame dumps (dump_NNNNN.png, 16-bit) with the given timestamps (Unix s) and frame
        counters in the header in the first row, laid out as the frames are read (words of the byteswapped row:
        timestamp seconds and microseconds in 32-bit words 3 and 4, the frame counter in word 7).
    """

    os.makedirs(path, exist_ok=True)
    for i, (t, c) in enumerate(zip(times, counters)):

        # The timestamp as seconds and microseconds, as written by the recording
        ts = int(np.floor(t))
        tu = int(round((t - ts)*1e6))
        if tu == 1000000:
            ts, tu = ts + 1, 0

        # The header words in the byteswapped first row, then the row as stored with the magick number first
        row = np.zeros(width, dtype=np.uint16)
        for word, value in ((3, ts), (4, tu), (7, c)):
            row[2*word] = value & 0xFFFF
            row[2*word + 1] = (value >> 16) & 0xFFFF
        frame = np.zeros((height, width), dtype=np.uint16)
        frame[0] = row.byteswap()
        frame[0, :2] = np.frombuffer(np.uint32(UWO_MAGICK_EMCCD).tobytes(), dtype=np.uint16)
        cv2.imwrite(os.path.join(path, 'dump_{:05d}.png'.format(i)), frame)


def regular(n, lost=(), n_lost=1):
    """ Timestamps and counters of n frames, with n_lost frames lost after each of the given frames (n_lost more
        periods between the timestamps, the counter going up by one).
    """

    steps = np.ones(n)
    for k in lost:
        steps[k + 1] += n_lost
    steps[0] = 0
    times = T0 + PERIOD*np.cumsum(steps)
    counters = 428900 + np.arange(n)

    return times, counters


def readTimes(path, n):
    """ The times (Unix s) of the frames as read by the image input (no frame_time_offset). """

    handle = InputTypeImages(path, cr.Config(), beginning_time=None)
    times = [datetime2UnixTime(handle.currentFrameTime(frame_no=i, dt_obj=True)) for i in range(n)]

    return handle, np.array(times)


def test_dumps_read_as_written(tmp_path):

    # The synthetic dumps are read as frame dumps with the written times
    n = 200
    times, counters = regular(n)
    path = str(tmp_path/'clean')
    writeDumps(path, times, counters)

    handle, read = readTimes(path, n)
    assert handle.uwo_png_mode and handle.uwo_magick_type == 'emccd'
    assert handle.png_time_model is not None
    assert np.max(np.abs(read - times)) < 2e-6
    assert not handle.png_repaired_frames


def test_seconds_off_corrected(tmp_path):

    # Runs of frames with the seconds field 1 s low, at the start and in the middle: every frame gets the time of
    #   its counter, the beginning too, and the corrected frames are counted
    n = 400
    times, counters = regular(n)
    stamped = times.copy()
    bad = list(range(0, 7)) + list(range(200, 210))
    stamped[bad] -= 1.0
    path = str(tmp_path/'glitch')
    writeDumps(path, stamped, counters)

    handle, read = readTimes(path, n)
    assert np.max(np.abs(read - times)) < 2e-6
    assert datetime2UnixTime(handle.beginning_datetime) == pytest.approx(times[0], abs=2e-6)
    assert len(handle.png_repaired_frames) == len(bad)


def test_lost_frame_kept(tmp_path):

    # A lost frame: the timestamps step by two periods there, the counter by one, and nothing is changed
    n = 300
    times, counters = regular(n, lost=(150,))
    path = str(tmp_path/'lost')
    writeDumps(path, times, counters)

    handle, read = readTimes(path, n)
    assert np.max(np.abs(read - times)) < 2e-6
    assert not handle.png_repaired_frames


@pytest.mark.parametrize('n_lost', [30, 32])
def test_many_lost_frames_kept(tmp_path, n_lost):

    # Many lost frames: the later timestamps are right, the repair is off and nothing is changed
    n = 400
    times, counters = regular(n, lost=(199,), n_lost=n_lost)
    path = str(tmp_path/'gap')
    writeDumps(path, times, counters)

    handle, read = readTimes(path, n)
    assert handle.png_time_model is None
    assert np.max(np.abs(read - times)) < 2e-6


def test_counter_not_following_times(tmp_path):

    # A counter which doesn't follow the timestamps: no model, and the timestamps (one 1 s off) are kept
    n = 200
    times, _ = regular(n)
    counters = 1000 + np.random.default_rng(1).permutation(n)
    stamped = times.copy()
    stamped[100] -= 1.0
    path = str(tmp_path/'counter')
    writeDumps(path, stamped, counters)

    handle, read = readTimes(path, n)
    assert handle.png_time_model is None
    assert np.max(np.abs(read - stamped)) < 2e-6


def test_plain_images_untouched(tmp_path):

    # PNGs which are not frame dumps get no timestamp model
    d = tmp_path/'plain'
    d.mkdir()
    for i in range(5):
        cv2.imwrite(str(d/'frame_{:03d}.png'.format(i)), np.full((16, 32), 1000, dtype=np.uint16))

    handle = InputTypeImages(str(d), cr.Config(), beginning_time=datetime.datetime(2026, 10, 20, 1, 2, 3), fps=25.0)
    assert not handle.uwo_png_mode
    assert handle.png_time_model is None


@pytest.mark.parametrize('after', [10, 389])
def test_gap_near_an_end_kept(tmp_path, after):

    # About a second of lost frames (32 frames) near the start or the end of the input, where few sampled frames
    #   are on one side of the gap: the timestamps are right and nothing is changed
    n = 400
    times, counters = regular(n, lost=(after,), n_lost=32)
    path = str(tmp_path/'endgap')
    writeDumps(path, times, counters)

    handle, read = readTimes(path, n)
    assert np.max(np.abs(read - times)) < 2e-6
    assert not handle.png_repaired_frames


def test_seconds_late_kept(tmp_path):

    # Frames reading a second late are not the seconds field off (it only reads low): kept as they are
    n = 400
    times, counters = regular(n)
    stamped = times.copy()
    stamped[100:107] += 1.0
    path = str(tmp_path/'late')
    writeDumps(path, stamped, counters)

    handle, read = readTimes(path, n)
    assert np.max(np.abs(read - stamped)) < 2e-6
    assert not handle.png_repaired_frames



def test_short_dump_long_run_corrected(tmp_path):

    # A run of 20 frames with the seconds field off in a short recording (the sampled frames are dense, so the
    #   run covers several of them): all corrected
    n = 200
    times, counters = regular(n)
    stamped = times.copy()
    bad = list(range(90, 110))
    stamped[bad] -= 1.0
    path = str(tmp_path/'short')
    writeDumps(path, stamped, counters)

    handle, read = readTimes(path, n)
    assert np.max(np.abs(read - times)) < 2e-6
    assert len(handle.png_repaired_frames) == len(bad)


def test_warning_without_model(tmp_path, caplog):

    # Frame dumps whose timestamps can't be checked against the counter: a warning names the directory
    n = 100
    times, _ = regular(n)
    counters = 1000 + np.random.default_rng(2).permutation(n)
    path = str(tmp_path/'unchecked')
    writeDumps(path, times, counters)

    with caplog.at_level(logging.WARNING, logger='rmslogger'):
        handle = InputTypeImages(path, cr.Config(), beginning_time=None)

    assert handle.png_time_model is None
    assert any((record.levelno == logging.WARNING) and ('could not be checked' in record.getMessage())
               and ('unchecked' in record.getMessage()) for record in caplog.records)


def test_dense_burst_corrected_in_3_minute_dump(tmp_path, monkeypatch):

    # A burst of runs as in the recordings in a 3-minute dump (5760 frames): runs of 7 frames every 64 frames
    #   over 2000 frames, with 5 runs of 41 frames every 96 frames in it. The sampled frames alone give no
    #   model (more than 3 in a row are off it), the headers between them do, and every frame is corrected
    n = 5760
    times, counters = regular(n)
    stamped = times.copy()
    for first in range(1000, 3000, 64):
        stamped[first:first + 7] -= 1.0
    for first in range(1700, 1700 + 5*96, 96):
        stamped[first:first + 41] -= 1.0
    path = str(tmp_path/'burst')
    writeDumps(path, stamped, counters)

    handle, read = readTimes(path, n)
    assert handle.png_time_model is not None
    assert np.max(np.abs(read - times)) < 2e-6
    assert len(handle.png_repaired_frames) == np.sum(np.abs(stamped - times) > 0.5)

    # The images read for the model: the 64 sampled frames and fewer than MAX_DENSE_PNG_HEADERS more, and with
    #   no more allowed than the samples there is no model
    reads = []
    imread = FrameInterface.cv2.imread

    def countingImread(*args):
        reads.append(args[0])
        return imread(*args)

    monkeypatch.setattr(FrameInterface.cv2, 'imread', countingImread)
    assert handle.uwoTimeModel() is not None
    assert 64 < len(reads) < 64 + MAX_DENSE_PNG_HEADERS
    reads.clear()
    assert handle.uwoTimeModel(max_dense=0) is None
    assert len(reads) <= 64
