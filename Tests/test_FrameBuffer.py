""" Tests of the segments of an input held in memory (RMS.Formats.FrameBuffer) and of the joining of the
    detections of neighbouring segments (RMS.MatchedFilterDetection.stitchSegments).
"""

from __future__ import print_function, division, absolute_import

import datetime

import numpy as np
import pytest

import RMS.ConfigReader as cr
from RMS.Formats.FrameBuffer import SegmentBuffer, segmentFrames, segmentMinimum, segmentRanges, segmentUnit
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
    """ The frames of a segment fit in the memory, in whole blocks of the matched filter, and only .vid inputs and
        frame blocks are read in segments. """

    config = cr.Config()
    handle = _Input(size=(512, 512))
    assert segmentUnit(config, handle) == 256

    # 4 GB of 512x512 frames of 2 bytes: 8192 frames
    assert segmentFrames(handle, config, 4.0) == 8192
    assert segmentFrames(handle, config, 0) == 0

    handle.input_type = 'images'
    assert segmentFrames(handle, config, 4.0) == 0


def test_whole_short_input_is_one_segment():
    """ A frame block which fits in the memory as a whole is one segment of all its frames, also when the memory
        is less than the fewest frames of a segment of a long input; one which doesn't fit isn't read in segments.
    """

    # A block of 450 1080p frames; 2 GB is less than the fewest frames of two segments of a long input
    config = cr.Config()
    handle = _Input(total_frames=450, size=(1080, 1920))
    handle.input_type = 'fblk'
    assert 2*segmentMinimum(config, handle)*1080*1920*2 > 2*1024**3

    # 2 GB holds the whole block, which is one segment; 1 GB doesn't, so it isn't read in segments
    assert segmentFrames(handle, config, 2.0) == 450
    assert segmentRanges(450, 450, 0, segmentUnit(config, handle)) == [(0, 450)]
    assert segmentFrames(handle, config, 1.0) == 0


def test_segment_of_a_frame_block(tmp_path):
    """ A frame block read through a segment gives the frames, their times and the chunks of the block. """

    import os
    import RMS.Formats.FrameBlock as fb
    from RMS.Formats.FrameInterface import detectInputTypeFile

    # A block of 300 random frames at 15 fps
    config = cr.parse(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.config'))
    config.width, config.height = 24, 16
    path = str(tmp_path/('XX0001_20251225_010203_000000' + fb.EXTENSION))
    rng = np.random.default_rng(3)
    with fb.FrameBlockWriter(path, 24, 16, max_frames=300) as w:
        for i in range(300):
            w.write(rng.integers(0, 4096, (16, 24)), 1_766_624_523_000_000_000 + i*66_666_667)

    # The whole block in a segment, and the block read directly for reference
    handle = detectInputTypeFile(path, config, detection=True, chunk_frames=128)
    reference = detectInputTypeFile(path, config, detection=True, chunk_frames=128)
    buffer = SegmentBuffer(handle, 300)
    buffer.load(0, 300)

    # The same frames and times

    for f in (0, 150, 299):
        buffer.setFrame(f)
        reference.setFrame(f)
        assert np.array_equal(buffer.loadFrame(), reference.loadFrame())
        assert buffer.currentFrameTime(frame_no=f, dt_obj=True) == reference.currentFrameTime(frame_no=f, dt_obj=True)

    # The same chunk images
    chunk = buffer.loadChunk(first_frame=128)
    expected = reference.loadChunk(first_frame=128)
    assert np.array_equal(np.asarray(chunk.maxpixel), np.asarray(expected.maxpixel))
    assert np.array_equal(np.asarray(chunk.avepixel), np.asarray(expected.avepixel))


def test_segment_minimum():
    """ A segment holds enough blocks for the slow search of the matched filter to search consecutive blocks, a
        memory which can't hold twice as many is not used, and the last segment is never shorter. """

    from RMS.MatchedFilterDetection import SLOW_MIN_BLOCKS, slowReferenceBlocks, slowSearchMinBlocks

    # The fewest blocks in which the slow search has SLOW_MIN_BLOCKS consecutive blocks with references
    def longest_run(n):
        best = run = 0
        for k in range(n):
            run = run + 1 if len(slowReferenceBlocks(k, n)) >= 2 else 0
            best = max(best, run)
        return best

    n_min = slowSearchMinBlocks()
    assert longest_run(n_min) >= SLOW_MIN_BLOCKS
    assert longest_run(n_min - 1) < SLOW_MIN_BLOCKS

    config = cr.Config()
    handle = _Input(size=(512, 512))
    unit = segmentUnit(config, handle)
    minimum = segmentMinimum(config, handle)
    assert minimum == n_min*256 == 7*256

    # 1.5 GB holds 3072 frames, less than twice the minimum, so the input is read whole; 2 GB holds 4096
    assert segmentFrames(handle, config, 1.5) == 0
    assert segmentFrames(handle, config, 2.0) == 4096

    # A last segment of 812 frames begins earlier, so it holds the minimum, and still shares frames only with the
    #   segment before it, at most half a segment
    assert segmentRanges(4396, 4096, 512, unit)[-1] == (3584, 4396)
    assert segmentRanges(4396, 4096, 512, unit, minimum=minimum) == [(0, 4096), (2560, 4396)]

    # For any input: every segment holds the minimum (or the whole input), begins on a whole unit, shares at most
    #   half a segment with the one before it and nothing with the others, and all frames are covered
    for total in range(100, 25000, 37):
        ranges = segmentRanges(total, 4096, 512, unit, minimum=minimum)
        assert ranges[0][0] == 0 and ranges[-1][1] == total
        for k, (first, end) in enumerate(ranges):
            assert (end - first >= min(minimum, total)) and (end - first <= 4096) and (first % unit == 0)
            if k > 0:
                assert first <= ranges[k - 1][1]
                assert ranges[k - 1][1] - first <= 4096//2
            if k > 1:
                assert first >= ranges[k - 2][1]


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
    """ The polar line of the joined detections, as MatchedFilterDetector.polarLine, and the checks of a
        detection. """

    det_bin = 1

    class opts(object):
        min_frames = 20
        min_centroids = 6

    def polarLine(self, cent):
        return 0.0, 0.0

    def minDisplacement(self):
        return 4.0


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
    # The object found only in the second segment is kept whole (the first one found nothing near it)
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
        segment are left to that segment if it found something near them (e.g. a fragment of an object at the edge
        of a segment, next to its track measured by the other segment). """

    cuts, overlaps = [1792], [(1536, 2048)]

    # The previous segment has a fragment after the cut, the next one before it, both 8 px from the object
    fragment_a = _track(np.arange(1850, 1900, 4), x0=108.0)
    fragment_b = _track(np.arange(1600, 1700, 4), x0=108.0)
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


def test_stitch_segments_each_frame_once():
    """ The detections which cross the middle of an overlap and are not the same object as one of the other segment
        are cut at the middle, so the frames of the overlap are covered by the detections of one segment only (an
        object measured differently in the two segments isn't kept twice). """

    cuts, overlaps = [1792], [(1536, 2048)]

    # The same faint object, found 8 px apart in the two segments (more than the tolerance)
    first = _track(np.arange(1600, 2048, 4))
    second = _track(np.arange(1650, 2200, 4), x0=108.0)

    detections = stitchSegments([[[0, 0, first]], [[0, 0, second]]], cuts, overlaps, _Detector(), tolerance=5.0)
    assert len(detections) == 2
    spans = sorted((det[2][0, 0], det[2][-1, 0]) for det in detections)
    assert spans[0][1] < 1792 <= spans[1][0]


def test_stitch_segments_keeps_one_sided_detections():
    """ A detection in the half of an overlap which belongs to the other segment, or crossing its middle, is kept
        whole if the other segment found nothing near it (the segments are searched with their own statistics, so
        an object can be found in one of them only). """

    cuts, overlaps = [1792], [(1536, 2048)]
    only_first = _track(np.arange(1850, 1950, 4), x0=600.0)
    only_second = _track(np.arange(1600, 1700, 4), x0=700.0)
    crossing_first = _track(np.arange(1700, 2000, 4), x0=800.0)

    detections = stitchSegments([[[0, 0, only_first], [0, 0, crossing_first]], [[0, 0, only_second]]], cuts,
                                overlaps, _Detector(), tolerance=5.0)
    spans = sorted((det[2][0, 0], det[2][-1, 0]) for det in detections)
    assert spans == [(1600, 1696), (1700, 1996), (1850, 1946)]


class _SegmentDetector(object):
    """ The parts of a MatchedFilterDetector which the summary of the segments uses. """

    class opts(object):
        use_gpu = False

    def __init__(self, tracks):
        self.tracks = tracks
        self.cloud_removed_frames, self.timing, self.aperture_correction = [], {}, None
        self.cloud_rows_removed, self.tracks_dropped, self.psf_sigma = 0, 0, 0.7
        self.cloud_removed_row_frames = []
        self.faint_measured = 0

    def frameTimes(self, frames):
        return []

    def inputEnd(self):
        return None

    def summary(self):
        return {'tracks': [{'first_frame': f0, 'last_frame': f1, 'significance': 10.0, 'detected': True}
                           for f0, f1 in self.tracks]}


def test_summary_tracks_of_the_segments():
    """ The tracks of the summary of the segments are those of the frames which belong to every segment (an object
        in an overlap is in the tracks of both segments, and is counted once). """

    from RMS.MatchedFilterDetection import SegmentedRun

    first = _SegmentDetector([(100, 400), (1700, 1900)])
    second = _SegmentDetector([(1700, 1900), (2500, 2800)])
    run = SegmentedRun(_Input(), [(0, 2048), (1536, 3584)], [first, second], [1792], None, SegmentBuffer(_Input(), 1),
                       {})
    tracks = sorted((t['first_frame'], t['last_frame']) for t in run.summary()['tracks'])
    assert tracks == [(100, 400), (1700, 1900), (2500, 2800)]


def test_summary_cloud_rows_of_the_segments():
    """ The measurements removed on clouded sky are counted in the segment which their frame belongs to (those in
        an overlap are removed in both segments). """

    from RMS.MatchedFilterDetection import SegmentedRun

    first, second = _SegmentDetector([]), _SegmentDetector([])
    first.cloud_removed_row_frames = [100, 200, 1700, 1750, 1800]
    second.cloud_removed_row_frames = [1700, 1750, 1800, 2500]
    first.cloud_rows_removed, second.cloud_rows_removed = 5, 4
    run = SegmentedRun(_Input(), [(0, 2048), (1536, 3584)], [first, second], [1792], None, SegmentBuffer(_Input(), 1),
                       {})
    assert run.summary()['cloud_frames_removed'] == 6


def test_segments_sky_from_the_stars_so_far(monkeypatch):
    """ The sky quality of every segment is made from the stars of all chunks read so far (a short segment alone may
        have too few chunks for the reference of the stars), while the segment itself is searched. """

    import RMS.MatchedFilterDetection as mfd

    config = cr.Config()
    config.mf_block_frames = 256
    config.mf_segment_overlap = 256
    config.ff_min_stars = 1
    handle = _Input(total_frames=2048)

    # Every chunk has stars, named by the time of its first frame
    def stars(img_handle, config, chunk_range=None, **kwargs):
        from RMS.Formats.FFfile import constructFFName
        return [[constructFFName('XX0001', handle.currentFrameTime(frame_no=c*128, dt_obj=True)), [(1.0,)*8]*5]
                for c in range(*chunk_range)]

    calls = []
    def detect(img_handle, config, star_list=None, frame_range=None, sky_range=None, **kwargs):
        calls.append((frame_range, sky_range, len(star_list)))
        return [], None

    monkeypatch.setattr(mfd, 'extractStarsFrameInterface', stars)
    monkeypatch.setattr(mfd, 'detectMatchedFilter', detect)
    monkeypatch.setattr(mfd, 'skyQuality', lambda *args, **kwargs: None)

    mfd.detectMatchedFilterSegments(handle, config, 768)
    assert [c[0] for c in calls] == [(0, 768), (512, 1280), (1024, 1792), (1536, 2048)]
    assert [c[1] for c in calls] == [(0, 768), (0, 1280), (0, 1792), (0, 2048)]
    assert [c[2] for c in calls] == [6, 10, 14, 16]


def test_persistence_carried_without_the_overlap():
    """ The persistence counts carried to the next segment are those before its first frame, so the runs of the
        overlap (searched again in the next segment) are counted once. """

    from RMS.MatchedFilterDetection import MatchedFilterDetector

    detector = MatchedFilterDetector(_Input(), cr.Config())
    detector.persistence_until = 512
    detector.persistence_counts = {(8, 2): np.ones((4, 4), dtype=np.int32)}
    detector.persistence_runs = {(8, 2): 10}

    detector.persistenceCheckpoint(256)
    assert detector.persistence_carry is None
    detector.persistenceCheckpoint(512)
    detector.persistence_counts[(8, 2)] += 5
    detector.persistence_runs[(8, 2)] += 7
    counts, runs = detector.persistence_carry[(8, 2)]
    assert runs == 10 and counts.max() == 1


def test_segments_each_have_the_track_cap_and_carry_persistence(monkeypatch):
    """ Every segment measures at most mf_max_tracks candidate tracks of its own, whatever the other segments
        measured (a busy part of a file doesn't take the measurements of the rest, nor the rest those of a busy
        part). The persistence is carried from before the next segment, and the smooth path of the detections is
        judged only after the segments are joined. """

    import RMS.MatchedFilterDetection as mfd

    config = cr.Config()
    config.mf_block_frames = 256
    config.mf_segment_overlap = 256
    config.mf_max_tracks = 300
    config.ff_min_stars = 1
    handle = _Input(total_frames=2048)

    def stars(img_handle, config, chunk_range=None, **kwargs):
        from RMS.Formats.FFfile import constructFFName
        return [[constructFFName('XX0001', handle.currentFrameTime(frame_no=c*128, dt_obj=True)), [(1.0,)*8]*5]
                for c in range(*chunk_range)]

    # Every segment is detected with the cap of the config (max_tracks not lowered)
    calls = []
    class _Detector(object):
        persistence_carry = {}
        persistence_carry_frames = 0
        psf_sigma = None
    def detect(img_handle, config, max_tracks=None, persistence_until=None, track_deviation=True, **kwargs):
        """ The search of a segment: records its options, finds nothing. """
        calls.append((max_tracks, persistence_until, config.mf_max_tracks, track_deviation))
        det = _Detector()
        det.tracks_measured = 300
        return [], det

    monkeypatch.setattr(mfd, 'extractStarsFrameInterface', stars)
    monkeypatch.setattr(mfd, 'detectMatchedFilter', detect)
    monkeypatch.setattr(mfd, 'skyQuality', lambda *args, **kwargs: None)
    monkeypatch.setattr(mfd, 'SegmentedRun', lambda *args, **kwargs: None)

    mfd.detectMatchedFilterSegments(handle, config, 768)
    assert [c[0] for c in calls] == [None]*4
    assert [c[2] for c in calls] == [300]*4
    assert [c[1] for c in calls] == [512, 1024, 1536, None]

    # The smooth path is judged only on the whole tracks after the segments are joined, not on their pieces
    assert [c[3] for c in calls] == [False]*4


def test_persistence_of_a_segment_judged_over_the_whole_input():
    """ In a segment, a pixel is persistent only if it is above the threshold in more than the fraction of the runs
        of the whole input (the counts only grow, so it is then persistent over the whole input too). """

    from RMS.MatchedFilterDetection import Hit, MatchedFilterDetector

    detector = MatchedFilterDetector(_Input(total_frames=2048), cr.Config())
    counts = np.zeros((8, 8), dtype=np.int32)
    counts[4, 4] = 5
    detector.persistence_counts, detector.persistence_runs = {(8, 2): counts}, {(8, 2): 10}
    hit = Hit(100.0, 8.5, 8.5, 0.0, 0.0, 6.0, 8, 2)

    # 5 of 10 runs, over all frames of the input: persistent
    detector.persistence_frames = 2048
    assert detector.removePersistent([hit]) == []

    # 5 of 10 runs of a quarter of the input: not known to be persistent (5 < 0.2*40)
    detector.persistence_frames = 512
    assert detector.removePersistent([hit]) == [hit]


def test_detections_on_persistent_pixels_of_the_whole_input():
    """ After the segments, the detections mostly on pixels which are persistent over the whole input are removed. """

    from RMS.MatchedFilterDetection import removePersistentDetections

    counts = np.zeros((32, 32), dtype=np.int32)
    counts[10, 5:15] = 50
    persistence = {(8, 2): (counts, 100)}
    on = np.array([[f, 2*x + 0.5, 20.5, 1.0] for f, x in zip(range(10), range(5, 15))], dtype=np.float64)
    off = on.copy()
    off[:, 2] = 40.5

    kept = removePersistentDetections([[0, 0, on], [0, 0, off]], persistence, 0.2, 1)
    assert len(kept) == 1 and kept[0][2][0, 2] == 40.5


def test_clouded_rows_removed_with_the_sky_of_the_whole_input():
    """ After the segments, the rows of the detections on sky which is clouded in the sky quality of the whole input
        are removed, and the detections left with fewer than 2 rows. """

    from RMS.MatchedFilterDetection import removeCloudedRows

    class _Sky(object):
        def isClear(self, x, y, frames):
            return np.asarray(frames) < 150

    partly = _track(np.arange(100, 200, 1), vx=5.0)
    clouded = _track(np.arange(300, 400, 1), vx=5.0)
    detections, rows, removed = removeCloudedRows([[0, 0, partly], [0, 0, clouded]], _Sky(), _Detector())
    assert len(detections) == 1 and detections[0][2][-1, 0] == 149
    assert rows == [float(f) for f in range(150, 200)] + [float(f) for f in range(300, 400)]
    assert removed == [300]


def test_persistence_carried_past_a_segment_not_searched(monkeypatch):
    """ When the next segment is not searched, the persistence counts of a segment are carried in full (also its
        overlap with the segment not searched, which counts them nowhere else). """

    import RMS.MatchedFilterDetection as mfd

    config = cr.Config()
    config.mf_block_frames = 256
    config.mf_segment_overlap = 256
    config.ff_min_stars = 3
    handle = _Input(total_frames=2048)

    # The chunks of the second segment (512-1280) have too few stars
    def stars(img_handle, config, chunk_range=None, **kwargs):
        from RMS.Formats.FFfile import constructFFName
        return [[constructFFName('XX0001', handle.currentFrameTime(frame_no=c*128, dt_obj=True)),
                 [(1.0,)*8]*(1 if 512 <= c*128 < 1280 else 5)] for c in range(*chunk_range)]

    calls = []
    class _Detector(object):
        psf_sigma = None
        tracks_measured = 0
    def detect(img_handle, config, frame_range=None, persistence_prior=None, persistence_prior_frames=0,
               **kwargs):
        calls.append((frame_range, persistence_prior, persistence_prior_frames))
        det = _Detector()
        det.persistence_carry, det.persistence_carry_frames = 'before {}'.format(frame_range), 1
        det.persistence_full, det.persistence_full_frames = 'all {}'.format(frame_range), 2
        return [], det

    monkeypatch.setattr(mfd, 'extractStarsFrameInterface', stars)
    monkeypatch.setattr(mfd, 'detectMatchedFilter', detect)
    monkeypatch.setattr(mfd, 'skyQuality', lambda *args, **kwargs: None)
    monkeypatch.setattr(mfd, 'SegmentedRun', lambda *args, **kwargs: None)

    mfd.detectMatchedFilterSegments(handle, config, 768)

    # The second segment is not searched, and the third one continues from all counts of the first one
    searched = [c[0] for c in calls]
    assert (512, 1280) not in searched
    following = calls[searched.index((1024, 1792))]
    assert following[1:] == ('all (0, 768)', 2)


@pytest.mark.parametrize('segment_frames, overlap', [(768, 512), (1024, 768), (2048, 2000), (8192, 512)])
def test_only_neighbouring_segments_overlap(segment_frames, overlap):
    """ The overlap is at most half a segment, so only neighbouring segments share frames (the joining of the
        detections and the persistence counts assume it). """

    ranges = segmentRanges(20000, segment_frames, overlap, 256)
    assert all(ranges[k + 2][0] >= ranges[k][1] for k in range(len(ranges) - 2))
    assert ranges[0][0] == 0 and ranges[-1][1] == 20000


def test_clouded_rows_removed_with_the_checks_of_a_detection():
    """ A detection which is left too short (in frames or in motion) after its clouded rows are removed is removed. """

    from RMS.MatchedFilterDetection import removeCloudedRows

    class _Sky(object):
        def isClear(self, x, y, frames):
            return np.asarray(frames) < 110

    # 10 clear frames left (fewer than 20), moving 0.5 px
    short = _track(np.arange(100, 200, 1))
    detections, rows, removed = removeCloudedRows([[0, 0, short]], _Sky(), _Detector())
    assert detections == [] and removed == [100]


def test_segments_on_the_blocks_of_the_detector():
    """ The segments begin and end on the blocks which the matched filter uses, also when it corrects the block
        length of the config. """

    from RMS.MatchedFilterDetection import matchedFilterBlockFrames

    config = cr.Config()
    config.mf_block_frames = 300
    block = matchedFilterBlockFrames(config)
    assert block != 300
    assert segmentUnit(config, _Input()) % block == 0


def test_invalid_detections_removed():
    """ Detections which don't pass the checks of a detection (e.g. a short fragment left by the cut at the middle of
        an overlap) are removed after the segments are joined. """

    from RMS.MatchedFilterDetection import removeInvalidDetections

    fragment = _track(np.arange(1792, 1800, 1), vx=50.0)
    good = _track(np.arange(100, 300, 1), vx=5.0)
    kept = removeInvalidDetections([[0, 0, fragment], [0, 0, good]], _Detector())
    assert len(kept) == 1 and kept[0][2][0, 0] == 100


def test_tail_segment_without_a_chunk_of_its_own(monkeypatch):
    """ A last segment shorter than a chunk of the star extraction is judged by the stars of the chunk which its
        frames belong to (the frames after the last whole chunk belong to it), and is searched. """

    import RMS.MatchedFilterDetection as mfd

    config = cr.Config()
    config.mf_block_frames = 256
    config.mf_segment_overlap = 0
    config.ff_min_stars = 3
    handle = _Input(total_frames=600)

    def stars(img_handle, config, chunk_range=None, **kwargs):
        from RMS.Formats.FFfile import constructFFName
        return [[constructFFName('XX0001', handle.currentFrameTime(frame_no=c*128, dt_obj=True)), [(1.0,)*8]*5]
                for c in range(*chunk_range)]

    calls = []
    class _Det(object):
        psf_sigma = None
        tracks_measured = 0
        persistence_carry, persistence_carry_frames, persistence_full, persistence_full_frames = {}, 0, {}, 0
    def detect(img_handle, config, frame_range=None, **kwargs):
        calls.append(frame_range)
        return [], _Det()

    monkeypatch.setattr(mfd, 'extractStarsFrameInterface', stars)
    monkeypatch.setattr(mfd, 'detectMatchedFilter', detect)
    monkeypatch.setattr(mfd, 'skyQuality', lambda *args, **kwargs: None)
    monkeypatch.setattr(mfd, 'SegmentedRun', lambda *args, **kwargs: None)

    mfd.detectMatchedFilterSegments(handle, config, 256)
    assert calls[-1] == (512, 600)


def test_persistence_kept_past_a_segment_too_short_to_search(monkeypatch):
    """ A segment too short to be searched (shorter than the longest run) has no persistence counts, and the counts
        of the segments before it are used for the detections of the whole input. """

    import RMS.MatchedFilterDetection as mfd

    config = cr.Config()
    config.mf_block_frames = 256
    config.mf_segment_overlap = 0
    config.ff_min_stars = 1
    handle = _Input(total_frames=600)

    def stars(img_handle, config, chunk_range=None, **kwargs):
        from RMS.Formats.FFfile import constructFFName
        return [[constructFFName('XX0001', handle.currentFrameTime(frame_no=c*128, dt_obj=True)), [(1.0,)*8]*5]
                for c in range(*chunk_range)]

    class _Det(_Detector):
        psf_sigma = 0.7
        tracks_measured = 0

        def joinDetections(self, detections, connected=None, cut=None):
            """ No joining: the detections as they are. """
            return detections

    def detect(img_handle, config, frame_range=None, **kwargs):
        det = _Det()

        # The last segment is not searched (too short): no counts
        if frame_range[0] >= 512:
            det.persistence_carry = det.persistence_full = None
        else:
            det.persistence_carry = det.persistence_full = 'counts to {}'.format(frame_range[1])
        det.persistence_carry_frames = det.persistence_full_frames = frame_range[1]
        return [], det

    used = {}
    def persistent(detections, persistence, fraction, det_bin):
        used['persistence'] = persistence
        return detections

    monkeypatch.setattr(mfd, 'extractStarsFrameInterface', stars)
    monkeypatch.setattr(mfd, 'detectMatchedFilter', detect)
    monkeypatch.setattr(mfd, 'skyQuality', lambda *args, **kwargs: None)
    monkeypatch.setattr(mfd, 'SegmentedRun', lambda *args, **kwargs: None)
    monkeypatch.setattr(mfd, 'removePersistentDetections', persistent)

    mfd.detectMatchedFilterSegments(handle, config, 256)
    assert used['persistence'] == 'counts to 512'


def test_segments_corrected_for_the_rolling_shutter_once(monkeypatch):
    """ The detections of the segments of an input are corrected for a rolling shutter once, together, after they
        are joined (the segments themselves are searched without the correction, see detectMatchedFilter). """

    import RMS.MatchedFilterDetection as mfd

    # An input of 2048 frames, searched in segments
    config = cr.Config()
    config.mf_block_frames = 256
    config.mf_segment_overlap = 256
    config.ff_min_stars = 1
    handle = _Input(total_frames=2048)

    # Stars in every chunk, and the frame range of every segment searched, without detections
    def stars(img_handle, config, chunk_range=None, **kwargs):
        from RMS.Formats.FFfile import constructFFName
        return [[constructFFName('XX0001', handle.currentFrameTime(frame_no=c*128, dt_obj=True)), [(1.0,)*8]*5]
                for c in range(*chunk_range)]

    segments, corrected = [], []
    def detect(img_handle, config, frame_range=None, **kwargs):
        segments.append(frame_range)
        return [], None

    # The calls of the rolling-shutter correction are counted
    monkeypatch.setattr(mfd, 'extractStarsFrameInterface', stars)
    monkeypatch.setattr(mfd, 'detectMatchedFilter', detect)
    monkeypatch.setattr(mfd, 'skyQuality', lambda *args, **kwargs: None)
    monkeypatch.setattr(mfd, 'correctRollingShutter', lambda detections, *args: corrected.append(detections) or detections)

    # The segments are searched as parts of the input (no correction there), the joined detections corrected once
    mfd.detectMatchedFilterSegments(handle, config, 768)
    assert all(frame_range is not None for frame_range in segments)
    assert len(corrected) == 1
