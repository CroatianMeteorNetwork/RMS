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
    ],
)
def testThresholdAndSubsampleThresholdConversion(max_value, average, stddev, k1, j1, expected):
    assert _thresholdCount(max_value, average, stddev, k1, j1) == expected


def _syntheticMeteorPoints(first_frame, last_frame):
    """ Points of a straight meteor track, a few pixels per frame, as uint16 like the detection stripes. """
    points = []
    for z in range(first_frame, last_frame + 1):
        x = 100 + 4*(z - first_frame)
        y = 200 + 3*(z - first_frame)
        points.extend([(x, y, z), (x + 1, y, z), (x, y + 1, z)])

    return np.array(points, dtype=np.uint16)


def testFind3DLinesFrameRangeIsSignedInteger():
    """ The frame range of a found line must be a plain integer.

    With NumPy 2 (NEP 50), uint16 scalars no longer promote to a signed type when an int is subtracted,
    so a line starting in the first few frames wrapped around when the detection extended its frame range
    (frame_min -= frame_extension -> ~65535), and the meteor was dropped.
    """

    from time import time

    from RMS.ConfigReader import Config
    from RMS.Routines.Grouping3D import find3DLines

    config = Config()

    lines = find3DLines(_syntheticMeteorPoints(1, 12), time(), config, fireball_detection=False)

    assert lines
    first_frame, last_frame = lines[0][4], lines[0][5]

    assert type(first_frame) is int
    assert type(last_frame) is int
    assert (first_frame, last_frame) == (1, 12)

    # The extended frame range must go below zero instead of wrapping around
    assert first_frame - config.frame_extension < 0
