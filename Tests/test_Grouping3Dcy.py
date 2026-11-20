"""Regression tests for the Grouping3D Cython routines."""

import pytest

np = pytest.importorskip("numpy")

from RMS.Routines.Grouping3Dcy import thresholdAndSubsample


def _thresholdCount(max_value, average, stddev, k1, j1):
    frames = np.zeros((1, 1, 1), dtype=np.uint8)
    compressed = np.array(
        [max_value, 0, average, stddev], dtype=np.uint8
    ).reshape(4, 1, 1)

    count, _, _, _ = thresholdAndSubsample(
        frames, compressed, min_level=0, min_points=0, k1=k1, j1=j1, f=1
    )
    return count


@pytest.mark.parametrize(
    "max_value, average, stddev, k1, j1, expected",
    [
        # Production configs use integer j1 values, but the fractional value is intentional here to pin
        # the truncation order for this function's float parameter.
        (10, 10, 1, 0.6, 0.6, 1),
        # Preserve the normal integer-offset behavior.
        (15, 10, 1, 0.6, 5.0, 1),
        # Clamp thresholds above the uint8 range to 255.
        (255, 250, 10, 1.0, 0.0, 1),
        # Reject a pixel below the threshold: avg_std = 10, max_value = 5.
        (5, 10, 1, 0.6, 0.0, 0),
        # A negative j1 gives a negative threshold, which must clamp to 0 (accept) instead of wrapping to a
        # huge unsigned value that then clamps to 255 (reject).
        (5, 10, 1, 0.6, -20.0, 1),
    ],
)
def testThresholdAndSubsampleThresholdConversion(max_value, average, stddev, k1, j1, expected):
    assert _thresholdCount(max_value, average, stddev, k1, j1) == expected


def testThresholdAndSubsampleSkipsPartialEdgeBlocks():
    """ Pixels in the partial edge block (image size not a multiple of f) must not be counted.

    With 1080 rows and f = 16 there are 67 full block rows, so rows 1072-1079 would map to block row 67,
    which is outside the count array and used to alias into the next frame (or past the buffer).
    """

    n_frames, height, width, f = 8, 1080, 1920, 16

    # Empty frames, with a standard deviation of 1 everywhere
    frames = np.zeros((n_frames, height, width), dtype=np.uint8)
    compressed = np.zeros((4, height, width), dtype=np.uint8)
    compressed[3] = 1

    # Bright pixels only in the last 8 rows (the partial block row), on frame 5
    compressed[0, 1072:1080, 0:64] = 200
    compressed[1, 1072:1080, 0:64] = 5

    # Nothing is found in the partial block row
    num, pointsx, pointsy, pointsz = thresholdAndSubsample(
        frames, compressed, min_level=40, min_points=8, k1=1.5, j1=9.0, f=f
    )

    assert num == 0
    assert len(pointsy) == 0

    # The same bright block moved into the last full block row must still be detected
    compressed[0] = 0
    compressed[1] = 0
    compressed[0, 1056:1072, 0:64] = 200
    compressed[1, 1056:1072, 0:64] = 5

    # One point per block in the last full block row, on frame 5
    num, pointsx, pointsy, pointsz = thresholdAndSubsample(
        frames, compressed, min_level=40, min_points=8, k1=1.5, j1=9.0, f=f
    )

    assert num == 4
    assert np.all(pointsy == height//f - 1)
    assert np.all(pointsz == 5)


def testThresholdAndSubsampleSkipsPartialEdgeColumns():
    """ Pixels in a partial edge block column must not be counted either. """

    n_frames, height, width, f = 4, 64, 72, 16

    # Empty frames, with a standard deviation of 1 everywhere
    frames = np.zeros((n_frames, height, width), dtype=np.uint8)
    compressed = np.zeros((4, height, width), dtype=np.uint8)
    compressed[3] = 1

    # Bright pixels only in the last 8 columns (the partial block column), on frame 1
    compressed[0, 0:16, 64:72] = 200
    compressed[1, 0:16, 64:72] = 1

    # Nothing is found in the partial block column
    num, pointsx, pointsy, pointsz = thresholdAndSubsample(
        frames, compressed, min_level=40, min_points=8, k1=1.5, j1=9.0, f=f
    )

    assert num == 0


@pytest.mark.parametrize("contiguous", [True, False])
def testThresholdAndSubsamplePaddedFrameBuffer(contiguous):
    """ A frame buffer padded by a row and a column must give the same points as an unpadded one.

    At 1280x720 the capture buffer is padded to 721x1281 (RMS.Misc.frameBufferShape), while the compressed
    frames are cropped to 720x1280. Under the forkserver/spawn start methods the extractor gets the compressed
    frames as a contiguous copy, and sizing the loops from the padded buffer made them read the wrong pixels
    (column 1280 of a row is the first pixel of the next row) and past the end of the array. Under fork the
    compressed frames are a view into the padded array; its padding is filled with bright pixels here, so
    reading it would also show up as extra points.
    """

    from types import SimpleNamespace

    from RMS.Misc import frameBufferShape

    n_frames, height, width, f = 8, 720, 1280, 16

    # The padded size of the 720p capture buffer
    _, buffer_height, buffer_width = frameBufferShape(SimpleNamespace(width=width, height=height))
    assert (buffer_height, buffer_width) == (height + 1, width + 1)

    # A short bright streak in the first column, on frame 3. It fills the first block, and reading column
    #   1280 of a contiguous copy would see it again as the first pixel of the next row
    compressed_padded = np.zeros((4, buffer_height, buffer_width), dtype=np.uint8)
    compressed_padded[3] = 1
    compressed_padded[0, 1:17, 0] = 200
    compressed_padded[1, 1:17, 0] = 3

    # Fill the padding row and column with bright pixels on the same frame, which must never be read
    compressed_padded[0, height, :] = 255
    compressed_padded[0, :, width] = 255
    compressed_padded[1, height, :] = 3
    compressed_padded[1, :, width] = 3

    # The compressed frames cropped to the image size, as a view (fork) or as a contiguous copy
    #   (forkserver/spawn)
    compressed = compressed_padded[:, :height, :width]
    if contiguous:
        compressed = np.ascontiguousarray(compressed)

    # Reference points from an unpadded frame buffer
    reference = thresholdAndSubsample(
        np.zeros((n_frames, height, width), dtype=np.uint8), np.ascontiguousarray(compressed),
        min_level=40, min_points=8, k1=1.5, j1=9.0, f=f
    )

    # Points from the padded frame buffer
    padded = thresholdAndSubsample(
        np.zeros((n_frames, buffer_height, buffer_width), dtype=np.uint8), compressed,
        min_level=40, min_points=8, k1=1.5, j1=9.0, f=f
    )

    # Only the streak's block is found, with or without the padding
    assert reference[0] == 1
    assert padded[0] == reference[0]
    for padded_coords, reference_coords in zip(padded[1:], reference[1:]):
        assert np.array_equal(padded_coords, reference_coords)



def _syntheticMeteorPoints(first_frame, last_frame, x0=100, y0=200):
    """ Points of a straight meteor track, a few pixels per frame, with three pixels on every frame. """

    # Move the head of the track a few pixels per frame and add two neighbouring pixels on each frame
    points = []
    for z in range(first_frame, last_frame + 1):
        x = x0 + 4*(z - first_frame)
        y = y0 + 3*(z - first_frame)
        points.extend([(x, y, z), (x + 1, y, z), (x, y + 1, z)])

    return np.array(points, dtype=np.int64)


def _faintDetectionGroupingConfig():
    """ Grouping parameters of the faint meteor detection (as set by Grouping3D.find3DLines), but allowing
    several lines, so a line list shared between calls would show up as extra lines.
    """

    from types import SimpleNamespace

    from RMS.ConfigReader import Config

    config = Config()

    # Faint detection thresholds, with room for more than one line
    return SimpleNamespace(
        max_lines=10,
        point_ratio_threshold=1,
        max_time=config.max_time_det,
        distance_threshold=config.distance_threshold_det,
        gap_threshold=config.gap_threshold_det,
        min_points=config.min_pixels_det,
        min_frames=config.min_frames,
        line_minimum_frame_range=config.line_minimum_frame_range_det,
        line_distance_const=config.line_distance_const_det,
    )


def testFind3DLinesFrameRangeIsSignedInteger():
    """ The frame range of a found line must be a plain integer.

    With NumPy 2 (NEP 50), uint16 scalars no longer promote to a signed type when an int is subtracted,
    so a line starting in the first few frames wrapped around when the detection extended its frame range
    (frame_min - frame_extension -> ~65535), and the meteor was dropped.
    """

    from time import time

    from RMS.ConfigReader import Config
    from RMS.Detection import extendFrameRange
    from RMS.Routines.Grouping3D import find3DLines

    config = Config()

    # Find the line of a meteor that starts in the second frame
    lines = find3DLines(_syntheticMeteorPoints(1, 12), time(), config, fireball_detection=False)

    assert lines
    first_frame, last_frame = lines[0][4], lines[0][5]

    # The frame range must come back as plain ints with the right values
    assert type(first_frame) is int
    assert type(last_frame) is int
    assert (first_frame, last_frame) == (1, 12)

    # Extending the range by 3 frames on a 256-frame FF caps it at the first frame instead of wrapping around
    assert extendFrameRange(first_frame, last_frame, 3, 256) == (0, 15)


@pytest.mark.parametrize(
    "frame_min, frame_max, frame_extension, total_frames, expected",
    [
        # A line starting in the first frames is capped at frame 0
        (1, 12, 3, 256, (0, 15)),
        # A line ending in the last frames is capped at the last frame
        (240, 254, 3, 256, (237, 255)),
        # A line in the middle is extended on both sides
        (100, 120, 3, 256, (97, 123)),
    ],
)
def testExtendFrameRangeWithUnsignedScalars(frame_min, frame_max, frame_extension, total_frames, expected):
    """ The frame range extension in the detection must not wrap around when given unsigned NumPy scalars
    (as the grouping returned them before), and must return plain ints.
    """

    from RMS.Detection import extendFrameRange

    # Extend a range given as uint16 scalars, then check the values and that they are plain ints
    extended = extendFrameRange(np.uint16(frame_min), np.uint16(frame_max), frame_extension, total_frames)

    assert extended == expected
    assert all(type(frame) is int for frame in extended)


def testFind3DLinesFrameNumbersAbove65535():
    """ Frame numbers of long videos (above the uint16 range) must not wrap around in the grouping. """

    from time import time

    from RMS.ConfigReader import Config
    from RMS.Routines.Grouping3D import find3DLines, getAllPoints

    config = Config()

    # Find the line of a meteor about 39 minutes into a 30 fps video
    points = _syntheticMeteorPoints(70000, 70011)

    lines = find3DLines(points, time(), config, fireball_detection=False)

    assert lines
    assert (lines[0][4], lines[0][5]) == (70000, 70011)

    # The points of the line keep their frame numbers too
    (x1, y1, z1), (x2, y2, z2) = lines[0][0], lines[0][1]
    line_points = getAllPoints(points, x1, y1, z1, x2, y2, z2, config, fireball_detection=False)

    assert line_points[:, 2].min() == 70000
    assert line_points[:, 2].max() == 70011


def testFind3DLinesDoesNotShareLinesBetweenCalls():
    """ Calling the grouping without a line list must start from an empty list every time, and not return
    the lines of an earlier call.
    """

    from time import time

    from RMS.Routines.Grouping3Dcy import find3DLines as find3DLinesCy

    grouping_config = _faintDetectionGroupingConfig()

    # Find the lines of two different meteors in two calls, neither given a line list
    first = find3DLinesCy(_syntheticMeteorPoints(1, 12).astype(np.int32), time(), grouping_config,
                          get_single=True)
    second = find3DLinesCy(_syntheticMeteorPoints(50, 61, x0=400, y0=100).astype(np.int32), time(),
                           grouping_config, get_single=True)

    # Each call must return only its own line
    assert len(first) == 1
    assert len(second) == 1
    assert (second[0][4], second[0][5]) == (50, 61)


def _fireballTrackPoints(first_frame, last_frame, y0, x0, dy, dx, per_frame):
    """ Event points (y, x, frame) of a straight track in the subsampled image of the fireball detection, with
    per_frame neighbouring points on every frame.
    """

    points = []
    for z in range(first_frame, last_frame + 1):
        y = int(round(y0 + dy*(z - first_frame)))
        x = int(round(x0 + dx*(z - first_frame)))
        for k in range(per_frame):
            points.append((y + k%2, x + k//2, z))

    return points


def testFireballSearchFindsLineNextToLineWithMostPoints():
    """ The fireball search must go on after a line holding most of the points (e.g. the glare of a bright
    source), so a meteor on the same frames keeps its line.

    point_ratio_threshold is compared with the points on the best line over the number of candidate lines,
    which stays far below it here. Comparing the share of all points on the best line (90% here) instead
    stopped the search after the first line, and the meteors next to such lines lost their clips.
    """

    from time import time

    from RMS.ConfigReader import Config
    from RMS.Routines.Grouping3D import find3DLines

    config = Config()

    # A long track with three points on every frame and a shorter one with one point per frame, later and
    #   elsewhere in the image
    long_track = _fireballTrackPoints(0, 59, 5, 5, 0.5, 1.0, 3)
    short_track = _fireballTrackPoints(150, 169, 40, 70, -0.5, -1.0, 1)
    points = np.array(sorted(long_track + short_track, key=lambda p: p[2]))
    assert len(long_track)/float(len(points)) > config.point_ratio_threshold

    lines = find3DLines(points, time(), config)

    # Both tracks are found, each with its own frame range
    assert lines is not None
    assert sorted((line[4], line[5]) for line in lines) == [(0, 59), (150, 169)]
