""" Tests for the RICE_1-compressed FF FITS layout (FFfits.write(compress=True)) and the
    Utils.ConvertCompressedFits converter.

    Run with:
        python -m pytest Tests/TestFFCompression.py
"""

from __future__ import print_function, division, absolute_import

import os
import shutil
import tempfile

import numpy as np
from astropy.io import fits

from RMS.Formats import FFfits
from RMS.Formats.FFStruct import FFStruct
from Utils.ConvertCompressedFits import convertDirectory


def makeFF(dtype=np.uint8, full_precision=True):
    """ Construct an FF structure with sky-like planes, optionally with the 16-bit average and sigma. """

    rng = np.random.default_rng(3)
    shape = (36, 64)

    ff = FFStruct()
    ff.nrows, ff.ncols = shape
    ff.nbits = 8 if dtype == np.uint8 else 16
    ff.nframes = 256
    ff.first = 0
    ff.camno = 'XX0001'
    ff.fps = 25.0
    ff.starttime = '2026-01-01T00:00:00.000000'

    top = 255 if dtype == np.uint8 else 4095
    sky = np.clip(np.rint(40 + rng.normal(0, 3, shape)), 0, top)
    ff.maxpixel = np.clip(sky + rng.integers(8, 20, shape), 0, top).astype(dtype)
    ff.maxframe = rng.integers(0, 256, shape).astype(dtype)
    ff.avepixel = sky.astype(dtype)
    ff.stdpixel = rng.integers(2, 5, shape).astype(dtype)

    if full_precision:
        ff.avepixel16 = (256*sky + rng.integers(-128, 128, shape)).astype(np.uint16)
        ff.stdpixel16 = rng.integers(128, 5*256, shape).astype(np.uint16)
        ff.avegamma = 1.0

    return ff


def readPlanes(file_path):
    """ All image HDUs of a file as {name: (hdu type, data)}. """

    with fits.open(file_path, memmap=False) as hdulist:
        return {hdu.name: (type(hdu).__name__, hdu.data.copy()) for hdu in hdulist[1:]}


def testCompressedRoundTrip():
    """ A compressed file reads back to the same planes as an uncompressed one, only the planes that
        compress are compressed, and readers that index the four legacy planes by position (as old
        RMS versions do) get the same uint8 arrays.
    """

    tmp_dir = tempfile.mkdtemp()

    try:
        ff = makeFF()
        FFfits.write(ff, tmp_dir, 'FF_XX0001_plain.fits')
        FFfits.write(ff, tmp_dir, 'FF_XX0001_rice.fits', compress=True)

        plain = readPlanes(os.path.join(tmp_dir, 'FF_XX0001_plain.fits'))
        rice = readPlanes(os.path.join(tmp_dir, 'FF_XX0001_rice.fits'))

        assert list(rice) == ['MAXPIXEL', 'MAXFRAME', 'AVEPIXEL', 'STDPIXEL', 'AVERESID', 'STDRESID']
        for name, (hdu_type, data) in rice.items():
            expected_type = 'CompImageHDU' if name in FFfits.COMPRESSED_PLANES else 'ImageHDU'
            assert hdu_type == expected_type, name
            assert data.dtype == plain[name][1].dtype
            assert np.array_equal(data, plain[name][1]), name

        with fits.open(os.path.join(tmp_dir, 'FF_XX0001_rice.fits')) as hdulist:
            for i, name in enumerate(['MAXPIXEL', 'MAXFRAME', 'AVEPIXEL', 'STDPIXEL'], start=1):
                assert hdulist[i].data.dtype == np.uint8
                assert np.array_equal(hdulist[i].data, plain[name][1])

        for name in ('FF_XX0001_plain.fits', 'FF_XX0001_rice.fits'):
            ff_read = FFfits.read(tmp_dir, name, full_filename=True)
            assert np.array_equal(ff_read.avepixel16, ff.avepixel16)
            assert np.array_equal(ff_read.stdpixel16, ff.stdpixel16)
            assert np.array_equal(ff_read.maxpixel, ff.maxpixel)

    finally:
        shutil.rmtree(tmp_dir)


def testCompressedNative16Bit():
    """ RICE_1 on a native 16-bit camera FF (uint16 planes, BZERO-scaled) is lossless. """

    tmp_dir = tempfile.mkdtemp()

    try:
        ff = makeFF(dtype=np.uint16, full_precision=False)
        FFfits.write(ff, tmp_dir, 'FF_XX0001_rice16.fits', compress=True)

        ff_read = FFfits.read(tmp_dir, 'FF_XX0001_rice16.fits', full_filename=True)
        for plane in ('maxpixel', 'maxframe', 'avepixel', 'stdpixel'):
            assert getattr(ff_read, plane).dtype == np.uint16
            assert np.array_equal(getattr(ff_read, plane), getattr(ff, plane)), plane

    finally:
        shutil.rmtree(tmp_dir)


def testConverterBothWays():
    """ The converter changes only the storage: data and the primary header survive both ways. """

    tmp_dir = tempfile.mkdtemp()

    try:
        night = os.path.join(tmp_dir, 'night', 'sub')
        os.makedirs(night)
        ff = makeFF()
        ff.soctemp = 41.5
        FFfits.write(ff, night, 'FF_XX0001_a.fits')

        rice_dir = os.path.join(tmp_dir, 'rice')
        plain_dir = os.path.join(tmp_dir, 'plain')
        assert convertDirectory(os.path.join(tmp_dir, 'night'), rice_dir, compress=True) == 0
        assert convertDirectory(rice_dir, plain_dir, compress=False) == 0

        original = os.path.join(night, 'FF_XX0001_a.fits')
        compressed = os.path.join(rice_dir, 'sub', 'FF_XX0001_a.fits')
        restored = os.path.join(plain_dir, 'sub', 'FF_XX0001_a.fits')

        planes = readPlanes(original)
        for path, types in ((compressed, None), (restored, 'ImageHDU')):
            converted = readPlanes(path)
            assert list(converted) == list(planes)
            for name in planes:
                assert np.array_equal(converted[name][1], planes[name][1]), name
                if types is not None:
                    assert converted[name][0] == types

        assert readPlanes(compressed)['MAXPIXEL'][0] == 'CompImageHDU'

        with fits.open(original) as a, fits.open(restored) as b:
            assert a[0].header.tostring() == b[0].header.tostring()

        # Decompressing restores the original bytes exactly
        with open(original, 'rb') as a, open(restored, 'rb') as b:
            assert a.read() == b.read()

    finally:
        shutil.rmtree(tmp_dir)


if __name__ == '__main__':

    testCompressedRoundTrip()
    print('testCompressedRoundTrip OK')

    testCompressedNative16Bit()
    print('testCompressedNative16Bit OK')

    testConverterBothWays()
    print('testConverterBothWays OK')

    print('\nAll FF compression tests passed.')
