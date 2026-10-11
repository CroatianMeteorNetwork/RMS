""" Tests of the segments of an input held in memory (RMS.Formats.FrameBuffer) and of the joining of the
    detections of neighbouring segments (RMS.MatchedFilterDetection.stitchSegments).
"""

from __future__ import print_function, division, absolute_import

import datetime

import numpy as np
import pytest

import RMS.ConfigReader as cr
from RMS.Formats.FrameBuffer import SegmentBuffer, segmentFrames, segmentRanges, segmentUnit
from RMS.MatchedFilterDetection import stitchSegments
from RMS.Routines.DynamicFTPCompressionCy import FFMimickInterface


BEGIN = datetime.datetime(2025, 12, 25, 3, 0, 0)
FPS = 32.0


class _Input(object):
    """ An input of frames whose pixels are the frame number, with a frame time from the frame rate. It counts
        the frames read from it. """

    input_type = 'vid'

    def __init__(self, total_frames=2000, size=(16, 24), chunk_frames=128):
        self.total_frames = total_frames
        self.nrows, self.ncols = size
        self.chunk_frames = chunk_frames
        self.total_fr_chunks = total_frames//chunk_frames
        self.fps = FPS
        self.beginning_datetime = BEGIN
        self.config = cr.Config()
        self.current_frame = 0
        self.reads = 0

    def setFrame(self, fr_num):
        self.current_frame = fr_num%self.total_frames

    def loadFrame(self, avepixel=False):
        self.reads += 1
        rng = np.random.default_rng(self.current_frame)
        return (100 + rng.integers(0, 50, (self.nrows, self.ncols))).astype(np.uint16)

    def currentFrameTime(self, frame_no=None, dt_obj=False):
        return BEGIN + datetime.timedelta(seconds=(self.current_frame if frame_no is None else frame_no)/FPS)

    def getTargetDtype(self, first_frame_data=None):
        return np.uint16


def test_segment_ranges():
    """ The segments cover the input, neighbouring ones share the overlap, and a short last one is taken into the
        one before it. """

    ranges = segmentRanges(5760, 2048, 512, 256)
    assert ranges == [(0, 2048), (1536, 3584), (3072, 5120), (4608, 5760)]

    # Every frame is in a segment, and only the overlaps are in two
    counts = np.zeros(5760, dtype=int)
    for first, end in ranges:
        counts[first:end] += 1
    assert counts.min() == 1 and counts.max() == 2 and (counts == 2).sum() == 3*512

    # A last segment of little more than the overlap is taken into the one before it if that still fits in the
    #   memory, otherwise it is a short segment of its own
    assert segmentRanges(5200, 2048, 512, 256) == [(0, 2048), (1536, 3584), (3072, 5120), (4608, 5200)]
    assert segmentRanges(5100, 2048, 512, 256) == [(0, 2048), (1536, 3584), (3072, 5100)]
    assert max(end - first for first, end in segmentRanges(5320, 2048, 512, 256)) <= 2048

    # An input which fits in one segment, and no segments
    assert segmentRanges(1000, 2048, 512, 256) == [(0, 1000)]
    assert segmentRanges(5000, 0, 512, 256) == [(0, 5000)]


def test_segment_frames():
    """ The frames of a segment fit in the memory, in whole blocks of the matched filter, and only .vid inputs are
        read in segments. """

    config = cr.Config()
    handle = _Input(size=(512, 512))
    assert segmentUnit(config, handle) == 256

    # 4 GB of 512x512 frames of 2 bytes: 8192 frames
    assert segmentFrames(handle, config, 4.0) == 8192
    assert segmentFrames(handle, config, 0) == 0

    handle.input_type = 'images'
    assert segmentFrames(handle, config, 4.0) == 0


def test_segment_buffer():
    """ The frames, their times and the chunks of a segment are those of the input, each frame is read once as the
        segment slides, and frames outside of it are read from the input. """

    handle = _Input()
    reference = _Input()
    buffer = SegmentBuffer(handle, 1024)

    buffer.load(0, 1024)
    buffer.load(512, 1536)
    assert handle.reads == 1536 and buffer.frames_read == 1536

    for f in (512, 1000, 1535):
        buffer.setFrame(f)
        reference.setFrame(f)
        assert np.array_equal(buffer.loadFrame(), reference.loadFrame())
        assert buffer.currentFrameTime(frame_no=f, dt_obj=True) == reference.currentFrameTime(frame_no=f, dt_obj=True)

    # A frame outside of the segment is read from the input
    buffer.setFrame(100)
    reference.setFrame(100)
    assert np.array_equal(buffer.loadFrame(), reference.loadFrame()) and buffer.frames_outside == 1

    # A chunk is made of the frames as the input makes it
    chunk = buffer.loadChunk(first_frame=640)
    ff = FFMimickInterface(handle.nrows, handle.ncols, np.uint16, res_size=handle.config.background_reservoir_size,
                           median_threads=handle.config.median_threads)
    for f in range(640, 768):
        reference.setFrame(f)
        ff.addFrame(reference.loadFrame())
    ff.finish()
    assert np.array_equal(np.asarray(chunk.maxpixel), np.asarray(ff.maxpixel))
    assert np.array_equal(np.asarray(chunk.avepixel), np.asarray(ff.avepixel))
    assert len(buffer.frame_chunk_unix_times) == 128

    # A segment which doesn't fit is refused
    with pytest.raises(ValueError):
        buffer.load(0, 2000)


class _Detector(object):
    """ The polar line of the joined detections, as MatchedFilterDetector.polarLine. """

    def polarLine(self, cent):
        return 0.0, 0.0


def _track(frames, x0=100.0, y0=200.0, vx=0.5, vy=-0.2):
    """ Centroids [frame, x, y, intensity] of an object moving in a straight line. """

    frames = np.asarray(frames, dtype=np.float64)
    return np.column_stack([frames, x0 + vx*frames/10, y0 + vy*frames/10, np.full(len(frames), 1000.0)])


def test_stitch_segments():
    """ An object found in both segments of an overlap is joined at its middle, the detections found in one
        segment are kept whole, and different objects in the overlap are not joined. """

    # Segments 0-2048 and 1536-3584, overlap 1536-2048, cut at 1792
    cuts, overlaps = [1792], [(1536, 2048)]

    crossing_a = _track(np.arange(1000, 2048, 4))
    crossing_b = _track(np.arange(1536, 3000, 4))
    only_a = _track(np.arange(100, 600, 4), x0=300.0)
    only_b = _track(np.arange(2500, 3500, 4), x0=400.0)
    other_in_overlap = _track(np.arange(1600, 2000, 4), x0=500.0)

    detections = stitchSegments([[[0, 0, crossing_a], [0, 0, only_a]],
                                 [[0, 0, crossing_b], [0, 0, only_b], [0, 0, other_in_overlap]]],
                                cuts, overlaps, _Detector(), tolerance=5.0)

    assert len(detections) == 4
    joined = [det[2] for det in detections if det[2][0, 0] == 1000][0]
    assert joined[0, 0] == 1000 and joined[-1, 0] == 2996
    assert np.all(np.diff(joined[:, 0]) > 0)
    assert any(det[2][0, 0] == 1600 for det in detections)


def test_block_indices_within_the_blocks():
    """ The frames outside of the blocks (before the first one and after the end of the last one, e.g. of a
        segment) are in no block. """

    from RMS.MatchedFilterDetection import blockIndices

    frames = np.array([100, 300, 600, 800, 1100])
    idx = blockIndices(frames, np.array([256, 512, 768]), end=1024)
    assert {k: list(v) for k, v in idx.items()} == {0: [1], 1: [2], 2: [3]}


def test_stitch_segments_ownership():
    """ The detections of a segment which lie entirely in the half of an overlap which belongs to the neighbouring
        segment are left to that segment (e.g. a fragment of an object at the edge of a segment). """

    cuts, overlaps = [1792], [(1536, 2048)]

    # The previous segment has a fragment after the cut, the next one before it
    fragment_a = _track(np.arange(1850, 1900, 4), x0=600.0)
    fragment_b = _track(np.arange(1600, 1700, 4), x0=700.0)
    crossing_a = _track(np.arange(1000, 2048, 4))
    crossing_b = _track(np.arange(1536, 3000, 4))

    detections = stitchSegments([[[0, 0, crossing_a], [0, 0, fragment_a]], [[0, 0, crossing_b], [0, 0, fragment_b]]],
                                cuts, overlaps, _Detector(), tolerance=5.0)

    assert len(detections) == 1 and detections[0][2][0, 0] == 1000 and detections[0][2][-1, 0] == 2996


@pytest.mark.parametrize('segment_frames', [256, 512, 768])
def test_segment_ranges_with_a_large_overlap(segment_frames):
    """ An overlap which is not smaller than a segment is reduced, so every segment adds at least a unit of new
        frames (it would never advance otherwise). """

    ranges = segmentRanges(10000, segment_frames, 512, 256)
    assert ranges[0][0] == 0 and ranges[-1][1] == 10000
    assert all(b[0] - a[0] >= 256 for a, b in zip(ranges[:-1], ranges[1:]))
    assert all(end - first <= segment_frames for first, end in ranges)


def test_stitch_segments_next_to_a_segment_not_searched():
    """ The half of an overlap which belongs to a segment that was not searched stays with the searched segment. """

    cuts, overlaps = [1792], [(1536, 2048)]
    after_cut = _track(np.arange(1900, 1950, 4))
    before_cut = _track(np.arange(1600, 1650, 4), x0=300.0)

    detections = stitchSegments([[[0, 0, after_cut]], []], cuts, overlaps, _Detector(), tolerance=5.0,
                                searched=[True, False])
    assert len(detections) == 1

    detections = stitchSegments([[], [[0, 0, before_cut]]], cuts, overlaps, _Detector(), tolerance=5.0,
                                searched=[False, True])
    assert len(detections) == 1


def test_stitch_segments_without_a_gap():
    """ The two detections of an object are joined where both have measurements, so no frames are lost when one
        of them doesn't reach the middle of the overlap. """

    cuts, overlaps = [1792], [(1536, 2048)]
    first = _track(np.arange(1600, 2048, 1))
    second = _track(np.arange(1900, 2400, 1))

    joined = stitchSegments([[[0, 0, first]], [[0, 0, second]]], cuts, overlaps, _Detector(), tolerance=5.0)[0][2]
    assert joined[0, 0] == 1600 and joined[-1, 0] == 2399
    assert np.all(np.diff(joined[:, 0]) == 1)


def test_summary_without_a_searched_segment():
    """ A file of which no segment was searched has no summary of the matched filter (it is not searched, as a
        file without enough stars). """

    from RMS.MatchedFilterDetection import SegmentedRun

    run = SegmentedRun(_Input(), [(0, 1024), (512, 2000)], [None, None], [768], None, None, {})
    assert run.summary() == {}
