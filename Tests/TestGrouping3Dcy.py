""" Regression tests for the Grouping3D Cython routines. """

from __future__ import print_function, division, absolute_import

import pytest

np = pytest.importorskip("numpy")

from RMS.Routines.Grouping3Dcy import thresholdAndSubsample


def _thresholdCount(max_value, average, stddev, k1, j1):
    """ Run thresholdAndSubsample on a single pixel and return the number of threshold passers. """

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
    """ The float threshold is truncated in the original order and clamped to the uint8 range. """

    assert _thresholdCount(max_value, average, stddev, k1, j1) == expected


def testThresholdAndSubsampleSkipsPartialEdgeBlocks():
    """ Pixels in the partial edge block (image size not a multiple of f) must not be counted.

    With 1080 rows and f = 16 there are 67 full block rows, so rows 1072-1079 would map to block row 67,
    which is outside the count array and used to alias into the next frame (or past the buffer).
    """

    n_frames, height, width, f = 8, 1080, 1920, 16

    frames = np.zeros((n_frames, height, width), dtype=np.uint8)
    compressed = np.zeros((4, height, width), dtype=np.uint8)
    compressed[3] = 1

    # Bright pixels only in the last 8 rows (the partial block row), on frame 5
    compressed[0, 1072:1080, 0:64] = 200
    compressed[1, 1072:1080, 0:64] = 5

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

    num, pointsx, pointsy, pointsz = thresholdAndSubsample(
        frames, compressed, min_level=40, min_points=8, k1=1.5, j1=9.0, f=f
    )

    assert num == 4
    assert np.all(pointsy == height//f - 1)
    assert np.all(pointsz == 5)


def testThresholdAndSubsampleSkipsPartialEdgeColumns():
    """ Pixels in a partial edge block column must not be counted either. """

    n_frames, height, width, f = 4, 64, 72, 16

    frames = np.zeros((n_frames, height, width), dtype=np.uint8)
    compressed = np.zeros((4, height, width), dtype=np.uint8)
    compressed[3] = 1
    compressed[0, 0:16, 64:72] = 200
    compressed[1, 0:16, 64:72] = 1

    num, pointsx, pointsy, pointsz = thresholdAndSubsample(
        frames, compressed, min_level=40, min_points=8, k1=1.5, j1=9.0, f=f
    )

    assert num == 0
