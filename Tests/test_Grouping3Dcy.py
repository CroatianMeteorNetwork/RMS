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
    "max_value, average, stddev, k1, j1",
    [
        # Truncate average + k1*stddev before adding the fractional offset.
        (10, 10, 1, 0.6, 0.6),
        # Preserve the normal integer-offset behavior.
        (15, 10, 1, 0.6, 5.0),
        # Clamp thresholds above the uint8 range to 255.
        (255, 250, 10, 1.0, 0.0),
    ],
)
def testThresholdAndSubsampleThresholdConversion(max_value, average, stddev, k1, j1):
    assert _thresholdCount(max_value, average, stddev, k1, j1) == 1
