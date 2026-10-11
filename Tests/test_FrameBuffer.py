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


def test_segments_share_the_track_cap_and_carry_persistence(monkeypatch):
    """ The segments share mf_max_tracks: every segment has a share by the frames which belong to it, which is kept
        for it, and measures at most what the segments before it left minus the shares of the segments after it. A
        file measures at most mf_max_tracks candidate tracks, all of them while it has fewer, and a later segment
        always has its share. The persistence is carried from before the next segment. """

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

    # The segments measure 0, 200, 80 and all the rest of the tracks they may
    measured = iter([0, 200, 80, 999])
    calls = []
    class _Detector(object):
        persistence_carry = {}
        persistence_carry_frames = 0
        psf_sigma = None
    def detect(img_handle, config, max_tracks=None, persistence_until=None, **kwargs):
        calls.append((max_tracks, persistence_until))
        det = _Detector()
        det.tracks_measured = min(next(measured), max_tracks)
        return [], det

    monkeypatch.setattr(mfd, 'extractStarsFrameInterface', stars)
    monkeypatch.setattr(mfd, 'detectMatchedFilter', detect)
    monkeypatch.setattr(mfd, 'skyQuality', lambda *args, **kwargs: None)
    monkeypatch.setattr(mfd, 'SegmentedRun', lambda *args, **kwargs: None)

    mfd.detectMatchedFilterSegments(handle, config, 768)
    assert [c[0] for c in calls] == [94, 169, 75, 56]
    assert [c[1] for c in calls] == [512, 1024, 1536, None]


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

    partly = _track(np.arange(100, 200, 1))
    clouded = _track(np.arange(300, 400, 1))
    detections, rows, removed = removeCloudedRows([[0, 0, partly], [0, 0, clouded]], _Sky(), _Detector())
    assert len(detections) == 1 and detections[0][2][-1, 0] == 149
    assert rows == [float(f) for f in range(150, 200)] + [float(f) for f in range(300, 400)]
    assert removed == [300]


def test_track_shares_sum_to_the_cap():
    """ The shares of the track cap add up to the cap exactly, so no segment is left without one by the rounding
        (e.g. a cap of 3 for 4 equal segments). """

    from RMS.MatchedFilterDetection import apportion

    assert sum(apportion(3, [1, 1, 1, 1])) == 3
    assert apportion(300, [640, 512, 512, 384]) == [94, 75, 75, 56]
    assert apportion(10, [0, 5]) == [0, 10]
