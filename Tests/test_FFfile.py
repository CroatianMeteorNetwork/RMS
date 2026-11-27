"""
Tests of the FF file names: only the FF formats recorded by RMS (FITS, bin) are taken for FF files by the
directory scans (detection, star extraction, archiving, calibration reports...). PNG images with an FF name, made
e.g. by BatchFFtoImage, are not FF files. The FF image handle only opens a PNG pseudo-FF when it is given one
explicitly, like the in-memory placeholder of SkyFit2, which is never written to the data directory.
"""

import os

import numpy as np

import RMS.ConfigReader as cr
from RMS.Formats import FFfile
from RMS.Formats.FFStruct import FFStruct
from RMS.Formats.FrameInterface import detectInputTypeFolder
from Utils.BatchFFtoImage import batchFFtoImage


# Image size (px) of the FF files of the tests
IMG_WIDTH = 64
IMG_HEIGHT = 48

FF_NAME = "FF_XX0001_20261021_201500_123_0000512.fits"


def writeFF(dir_path, file_name, level):
    """ Write an FF file with flat images.

    Arguments:
        dir_path: [str] Directory of the file.
        file_name: [str] Name of the FF file.
        level: [int] Pixel level of the maxpixel and the avepixel.
    """

    ff = FFStruct()
    ff.nrows = IMG_HEIGHT
    ff.ncols = IMG_WIDTH
    ff.nbits = 8
    ff.nframes = 256
    ff.fps = 25.0
    ff.starttime = "2026-10-21T20:15:00.123"
    ff.maxpixel = np.full((IMG_HEIGHT, IMG_WIDTH), level, dtype=np.uint8)
    ff.avepixel = np.full((IMG_HEIGHT, IMG_WIDTH), level, dtype=np.uint8)
    ff.stdpixel = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    ff.maxframe = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

    FFfile.write(ff, dir_path, file_name, fmt='fits')


def testValidFFNameRejectsPngWithoutFormat():
    """ The directory scans (validFFName without fmt) never take a PNG image for an FF file. """

    png_name = "FF_XX0001_20261021_201500_123_0000512.png"
    placeholder_name = "FF_XX0001_20261021_201500_123_0000512_placeholder.png"

    # Without a format, only the recorded FF formats are FF files
    assert not FFfile.validFFName(png_name)
    assert not FFfile.validFFName(placeholder_name)
    assert FFfile.validFFName(FF_NAME)
    assert FFfile.validFFName("FF_XX0001_20261021_201500_123_0000512.bin")

    # A PNG pseudo-FF is accepted when asked for explicitly
    assert FFfile.validFFName(png_name, fmt='png')


def testFFHandleListsNoPngAndOpensMemoryFF(tmp_path):
    """ The FF image handle of a night folder with BatchFFtoImage PNGs lists only the FF files, and opens an FF
        structure added in memory without writing anything to the folder. """

    dir_path = str(tmp_path)

    # An FF file and its PNG image, made by BatchFFtoImage
    writeFF(dir_path, FF_NAME, 100)
    batchFFtoImage(dir_path, 'png')
    files_before = sorted(os.listdir(dir_path))
    assert any(file_name.endswith('.png') for file_name in files_before)

    config = cr.Config()
    config.width = IMG_WIDTH
    config.height = IMG_HEIGHT

    # Only the FF file is listed
    img_handle = detectInputTypeFolder(dir_path, config)
    assert img_handle.ff_list == [FF_NAME]

    # An in-memory placeholder is added to the end of the list
    placeholder_name = FF_NAME.replace('.fits', '_placeholder.png')
    placeholder_img = np.full((IMG_HEIGHT, IMG_WIDTH), 24, dtype=np.uint8)
    index = img_handle.addMemoryFF(placeholder_name, FFfile.ffStructFromImage(placeholder_img))
    assert index == 1
    assert img_handle.ff_list == [FF_NAME, placeholder_name]

    # The placeholder is read from memory, and its time is taken from its name
    img_handle.nextChunk()
    assert img_handle.name() == placeholder_name
    assert np.all(img_handle.loadChunk().avepixel == 24)
    assert img_handle.beginning_datetime == FFfile.filenameToDatetime(FF_NAME)

    # The FF file is still read from disk
    img_handle.nextChunk()
    assert img_handle.name() == FF_NAME
    assert np.all(img_handle.loadChunk().avepixel == 100)

    # Nothing was written to the folder
    assert sorted(os.listdir(dir_path)) == files_before
