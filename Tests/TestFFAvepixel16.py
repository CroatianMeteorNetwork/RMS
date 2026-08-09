""" Tests for the full-precision (16-bit fixed point) FF average plane.

Covers:
    - the compressor's ave16 output matches an exact numpy reference and stays
      consistent with the legacy 8-bit average (ave16 >> 8 == avepixel)
    - FITS round trip of the 16-bit plane, incl. the derived legacy 8-bit view
    - files without the plane and native 16-bit camera files are unaffected
    - extractStarsFF uses the full-precision plane when present

Run directly (python -m Tests.TestFFAvepixel16) or via pytest.
"""

from __future__ import print_function, division, absolute_import

import os
import shutil
import tempfile

import numpy as np

from RMS.CompressionCy import compressFrames
from RMS.Formats import FFfile, FFfits
from RMS.Formats.FFStruct import FFStruct


def referenceAverages(frames):
    """ Exact numpy reference of the compressor's trimmed averages. """

    frames_sorted = np.sort(frames.astype(np.uint64), axis=0)

    # Sum without the top 4 max values
    acc = frames_sorted[:-4].sum(axis=0)
    n = frames.shape[0] - 4

    ave8 = acc//n
    ave16 = (256*acc + n//2)//n

    return ave8.astype(np.uint8), ave16.astype(np.uint16)


def makeFF(ave16=None, dtype=np.uint8):
    """ Construct a minimal FF structure with random planes. """

    rng = np.random.default_rng(42)

    ff = FFStruct()
    ff.nrows, ff.ncols = 32, 48
    ff.nbits = 8
    ff.nframes = 256
    ff.first = 0
    ff.camno = 'XX0001'
    ff.fps = 25.0
    ff.starttime = '2026-01-01T00:00:00.000000'

    ff.maxpixel = rng.integers(0, 255, (32, 48)).astype(dtype)
    ff.maxframe = rng.integers(0, 255, (32, 48)).astype(dtype)
    ff.avepixel = rng.integers(0, 255, (32, 48)).astype(dtype)
    ff.stdpixel = rng.integers(1, 20, (32, 48)).astype(dtype)
    ff.avepixel16 = ave16

    return ff


def testCompressorConsistency():
    """ ave16 matches the reference and ave16 >> 8 equals the 8-bit average. """

    rng = np.random.default_rng(7)

    test_stacks = [
        rng.integers(0, 256, (256, 24, 24)).astype(np.uint8),
        np.clip(np.rint(40.4 + rng.normal(0, 4.5, (256, 24, 24))), 0, 255).astype(np.uint8),
        np.zeros((256, 8, 8), np.uint8),
        np.full((256, 8, 8), 255, np.uint8),
    ]

    for frames in test_stacks:

        ftp, ave16, _ = compressFrames(frames, -1)
        ref8, ref16 = referenceAverages(frames)

        assert np.array_equal(ftp[2], ref8)
        assert np.array_equal(ave16, ref16)
        assert np.array_equal(ave16 >> 8, ftp[2].astype(np.uint16))


def testFitsRoundTrip():
    """ The 16-bit plane survives a write/read cycle and yields the 8-bit view. """

    tmp_dir = tempfile.mkdtemp()

    try:

        ave16 = (np.random.default_rng(1).integers(0, 255, (32, 48)).astype(np.uint16)*256
            + np.random.default_rng(2).integers(0, 256, (32, 48)).astype(np.uint16))

        ff = makeFF(ave16=ave16)
        FFfits.write(ff, tmp_dir, 'FF_XX0001_roundtrip.fits')

        ff_read = FFfits.read(tmp_dir, 'FF_XX0001_roundtrip.fits', full_filename=True)

        assert ff_read.avepixel16 is not None
        assert ff_read.avepixel16.dtype == np.uint16
        assert np.array_equal(ff_read.avepixel16, ave16)

        # The legacy view must be the fixed-point plane shifted down
        assert ff_read.avepixel.dtype == np.uint8
        assert np.array_equal(ff_read.avepixel, (ave16 >> 8).astype(np.uint8))


        # The array interface stays uint8
        ff_read_arr = FFfile.read(tmp_dir, 'FF_XX0001_roundtrip.fits', array=True,
            full_filename=True)
        assert ff_read_arr.array.dtype == np.uint8

    finally:
        shutil.rmtree(tmp_dir)


def testLegacyFileUnchanged():
    """ Files written without the plane read exactly as before. """

    tmp_dir = tempfile.mkdtemp()

    try:

        ff = makeFF(ave16=None)
        FFfits.write(ff, tmp_dir, 'FF_XX0001_legacy.fits')

        ff_read = FFfits.read(tmp_dir, 'FF_XX0001_legacy.fits', full_filename=True)

        assert ff_read.avepixel16 is None
        assert ff_read.avepixel.dtype == np.uint8
        assert np.array_equal(ff_read.avepixel, ff.avepixel)

    finally:
        shutil.rmtree(tmp_dir)


def testNative16BitFileNotMisread():
    """ A native 16-bit camera FF (no AVEFRAC keyword) must not be shifted. """

    tmp_dir = tempfile.mkdtemp()

    try:

        ff = makeFF(ave16=None, dtype=np.uint16)
        ff.nbits = 16
        FFfits.write(ff, tmp_dir, 'FF_XX0001_16bit.fits')

        ff_read = FFfits.read(tmp_dir, 'FF_XX0001_16bit.fits', full_filename=True)

        assert ff_read.avepixel16 is None
        assert ff_read.avepixel.dtype == np.uint16
        assert np.array_equal(ff_read.avepixel, ff.avepixel)

    finally:
        shutil.rmtree(tmp_dir)


def testExtractStarsPrecisePath():
    """ extractStarsFF runs on both file variants and finds the injected stars. """

    import RMS.ConfigReader as cr
    from RMS.ExtractStars import extractStarsFF

    rng = np.random.default_rng(11)

    # Synthetic star field: sky + Gaussian PSF stars + per-frame noise
    height, width = 128, 128
    sky, sigma_noise, psf_sigma = 40.0, 4.5, 1.2

    yy, xx = np.mgrid[0:height, 0:width]
    star_img = np.zeros((height, width))
    positions = []

    for gy in range(3):
        for gx in range(3):
            cy, cx = 24 + 40*gy, 24 + 40*gx
            flux = 600.0
            star_img += flux/(2*np.pi*psf_sigma**2)*np.exp(
                -((yy - cy)**2 + (xx - cx)**2)/(2*psf_sigma**2))
            positions.append((cy, cx))

    frames = np.clip(np.rint(
        (sky + star_img)[None, :, :] + rng.normal(0, sigma_noise, (256, height, width))),
        0, 255).astype(np.uint8)

    ftp, ave16, _ = compressFrames(frames, -1)

    config = cr.Config()
    config.width, config.height = width, height

    tmp_dir = tempfile.mkdtemp()

    try:

        for name, use16 in [('FF_XX0001_stars8.fits', False), ('FF_XX0001_stars16.fits', True)]:

            ff = makeFF()
            ff.nrows, ff.ncols = height, width
            ff.maxpixel, ff.maxframe, ff.avepixel, ff.stdpixel = ftp
            ff.avepixel16 = ave16 if use16 else None
            FFfits.write(ff, tmp_dir, name)

            result = extractStarsFF(tmp_dir, name, config=config)

            x_arr, y_arr = result[1], result[2]
            assert len(x_arr) >= 8, 'too few stars found in {:s}'.format(name)

            # Every found star must be at an injected position
            for x, y in zip(x_arr, y_arr):
                dists = [np.hypot(y - cy, x - cx) for cy, cx in positions]
                assert min(dists) < 1.5

    finally:
        shutil.rmtree(tmp_dir)


if __name__ == '__main__':

    testCompressorConsistency()
    print('testCompressorConsistency OK')

    testFitsRoundTrip()
    print('testFitsRoundTrip OK')

    testLegacyFileUnchanged()
    print('testLegacyFileUnchanged OK')

    testNative16BitFileNotMisread()
    print('testNative16BitFileNotMisread OK')

    testExtractStarsPrecisePath()
    print('testExtractStarsPrecisePath OK')

    print('\nAll FF avepixel16 tests passed.')
