""" The blocked median and MAD of the sampled frames give the same result as np.median along the first axis. """

import numpy as np
import pytest

from RMS.Routines.DynamicFTPCompressionCy import FFMimickInterface, sampleMedianMAD


def _reference(samples):
    median = np.median(samples, axis=0).astype(np.float32)
    mad = np.median(np.abs(samples.astype(np.float32) - median), axis=0)
    return median, mad


@pytest.mark.parametrize('n_samples', [1, 2, 63, 64])
@pytest.mark.parametrize('shape', [(512, 512), (13, 37)])
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_blocked_median_and_mad_are_identical_to_numpy(n_samples, shape, dtype):

    rng = np.random.default_rng(1)
    samples = rng.normal(100, 20, (n_samples,) + shape).clip(0, 255).astype(dtype)

    median, mad = sampleMedianMAD(samples)
    median_ref, mad_ref = _reference(samples)

    assert (median.dtype, mad.dtype) == (np.float32, np.float32)
    assert np.array_equal(median, median_ref)
    assert np.array_equal(mad, mad_ref)


def test_finished_chunk_uses_the_median_and_mad():

    # Fewer frames than the 64 samples of the reservoir, so all of them are sampled
    rng = np.random.default_rng(2)
    frames = rng.normal(1000, 30, (50, 24, 40)).astype(np.uint16)

    ff = FFMimickInterface(24, 40, np.uint16)
    for frame in frames:
        ff.addFrame(frame)
    ff.finish()

    median_ref, mad_ref = _reference(frames)
    assert np.array_equal(ff.avepixel, np.clip(median_ref, 0, 65535).astype(np.uint16))
    assert np.array_equal(ff.stdpixel, np.clip(np.maximum(mad_ref*1.4826, 1), 0, 65535).astype(np.uint16))
