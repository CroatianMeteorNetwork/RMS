""" Tests for InputTypeImages FITS header logging and consistent float-FITS uint16 mapping. """

from __future__ import print_function, division, absolute_import

import numpy as np
import pytest
from astropy.io import fits

from RMS.Formats.FrameInterface import InputTypeImages


def _makeImagesHandle(tmp_path, file_names):
    """ Create a minimal InputTypeImages stand-in without running __init__ GUI dialogs. """

    handle = InputTypeImages.__new__(InputTypeImages)
    handle.dir_path = str(tmp_path)
    handle.img_list = list(file_names)
    handle.current_frame = 0
    handle.current_img_file = file_names[0]
    handle.single_image_mode = False
    handle.uwo_png_mode = False
    handle.fripon_mode = False
    handle.cabernet_status = False
    handle.flipud = False
    handle.detection = False
    handle.cache = {}
    return handle


def _writeFloatFits(path, arr, extra_card=("EXPTIME", 1.5)):
    hdu = fits.PrimaryHDU(arr.astype(np.float32))
    hdu.header[extra_card[0]] = extra_card[1]
    hdu.writeto(str(path), overwrite=True)


def testFitsHeaderPrintedOncePerDirectory(tmp_path, capsys):
    """ Loading multiple .fits frames from one folder must print the FITS header only once. """

    arr1 = np.linspace(100.0, 500.0, 16, dtype=np.float32).reshape(4, 4)
    arr2 = np.linspace(120.0, 520.0, 16, dtype=np.float32).reshape(4, 4)
    _writeFloatFits(tmp_path/"img_001.fits", arr1)
    _writeFloatFits(tmp_path/"img_002.fits", arr2)

    handle = _makeImagesHandle(tmp_path, ["img_001.fits", "img_002.fits"])
    handle.loadFrame(fr_no=0)
    handle.loadFrame(fr_no=1)

    out = capsys.readouterr().out
    assert out.count("FITS Header:") == 1
    assert out.count("FITS mode") == 1


def testFloatFitsSequenceConsistentMapping(tmp_path):
    """ Two float FITS frames with different minima must map identical values to identical uint16 levels. """

    arr1 = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32)
    arr2 = np.array([[150.0, 200.0], [300.0, 500.0]], dtype=np.float32)
    _writeFloatFits(tmp_path/"img_001.fits", arr1)
    _writeFloatFits(tmp_path/"img_002.fits", arr2)

    handle = _makeImagesHandle(tmp_path, ["img_001.fits", "img_002.fits"])
    out1 = handle.loadFrame(fr_no=0)
    out2 = handle.loadFrame(fr_no=1)

    assert out1.dtype == np.uint16
    assert out2.dtype == np.uint16
    # Value 200.0 and 300.0 appear in both frames and must keep the exact same uint16 level
    assert out1[0, 1] == out2[0, 1] == 100
    assert out1[1, 0] == out2[1, 0] == 200


def testFloatFitsNormalizedWideRangeAndNaN(tmp_path):
    """ Normalised [0, 1] images stretch across uint16, wide >65535 ranges scale without clipping, NaN -> 0. """

    # Normalised [0, 1] with a NaN pixel
    norm = np.array([[0.0, 0.5], [1.0, np.nan]], dtype=np.float32)
    _writeFloatFits(tmp_path/"norm.fits", norm)
    h_norm = _makeImagesHandle(tmp_path, ["norm.fits"])
    out_norm = h_norm.loadFrame(fr_no=0)

    assert out_norm.dtype == np.uint16
    assert out_norm[0, 0] == 0
    assert out_norm[0, 1] == int(0.5*65535.0)
    assert out_norm[1, 0] == 65535
    assert out_norm[1, 1] == 0

    # Wide range > 65535 must scale down instead of clipping to 65535
    wide = np.array([[0.0, 100000.0], [200000.0, 50000.0]], dtype=np.float32)
    _writeFloatFits(tmp_path/"wide.fits", wide)
    h_wide = _makeImagesHandle(tmp_path, ["wide.fits"])
    out_wide = h_wide.loadFrame(fr_no=0)

    assert out_wide[0, 0] == 0
    assert out_wide[1, 0] == 65535
    assert 30000 < out_wide[0, 1] < 35000
    assert out_wide[1, 1] < out_wide[0, 1] < out_wide[1, 0]


@pytest.mark.parametrize("first_kind", ["constant", "all_nan"])
def testFloatFitsConstantOrAllNaNFirstFrameDoesNotFreezeScale(tmp_path, first_kind):
    """ A constant or all-NaN first frame must establish the sequence mapping from later frames
        so frame 0 and reloaded frame 0 match and later [0, 1] frames keep full 16-bit range.
    """

    if first_kind == "constant":
        first = np.full((4, 4), 0.25, dtype=np.float32)
    else:
        first = np.full((4, 4), np.nan, dtype=np.float32)

    second = np.array([[0.0, 0.25, 0.5, 1.0]]*4, dtype=np.float32)
    _writeFloatFits(tmp_path/"img_001.fits", first)
    _writeFloatFits(tmp_path/"img_002.fits", second)

    handle = _makeImagesHandle(tmp_path, ["img_001.fits", "img_002.fits"])
    out1 = handle.loadFrame(fr_no=0)
    out2 = handle.loadFrame(fr_no=1)
    out1_reloaded = handle.loadFrame(fr_no=0)

    # First load and reload of frame 0 must agree identically
    assert np.array_equal(out1, out1_reloaded)
    if first_kind == "constant":
        assert np.all(out1 == int(0.25*65535.0))
    else:
        assert np.all(out1 == 0)

    assert out2[0, 0] == 0
    assert out2[0, 2] == int(0.5*65535.0)
    assert out2[0, 3] == 65535
