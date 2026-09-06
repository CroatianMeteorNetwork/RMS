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
