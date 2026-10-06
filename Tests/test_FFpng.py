"""
Tests for FF-equivalent image pairs stored as PNG files (RMS.Formats.FFpng) and for reading them through the
FF file layer, the archive file selection, and the report utilities (stacks, thumbnails, timelapse).
"""

import datetime
import os
import shutil

import numpy as np
import pytest

import RMS.ConfigReader as cr
from RMS.Formats import FFfile, FFpng
from RMS.ArchiveDetections import selectFiles
from RMS.Routines.Image import to8bitDisplay


STATION = 'XX0001'
BEG_TIME = datetime.datetime(2025, 12, 25, 10, 0, 0, 69539)


def _pairName(seconds=0.0, frame=0):
    dt = BEG_TIME + datetime.timedelta(seconds=seconds)
    return FFfile.constructFFName(STATION, dt, frame=frame, suffix=FFpng.PAIR_MAX_SUFFIX)


def _images(dtype, shape=(48, 64), seed=0):
    """ Return a max pixel and an average pixel image of the given dtype, with a bright 'meteor' line. """

    rng = np.random.default_rng(seed)
    scale = 1 if dtype == np.uint8 else 256

    avepixel = (rng.normal(40, 3, shape)*scale).clip(0, np.iinfo(dtype).max).astype(dtype)
    maxpixel = avepixel.copy()
    maxpixel[10, 5:40] = 200*scale

    return maxpixel, avepixel


def _writeNight(dir_path, dtype, n=4):
    """ Write n image pairs, 1.28 s apart, and return their canonical names. """

    names = []
    for i in range(n):
        name = _pairName(seconds=1.28*i, frame=128*i)
        maxpixel, avepixel = _images(dtype, seed=i)
        FFpng.writePair(dir_path, name, maxpixel, avepixel,
                        meta={'nframes': 128, 'fps': 100.0, 'first_frame': 128*i, 'binning': 1})
        names.append(name)

    return names


### Names ###

def test_constructFFName_pair():

    name = _pairName(frame=128)

    assert name == 'FF_XX0001_20251225_100000_069_0000128_maxpixel.png'
    assert FFpng.pairNames(name) == (name, name.replace('_maxpixel.png', '_avepixel.png'))
    assert FFfile.filenameToDatetime(name) == datetime.datetime(2025, 12, 25, 10, 0, 0, 69000)


def test_constructFFName_default_unchanged():
    assert FFfile.constructFFName(STATION, BEG_TIME) == 'FF_XX0001_20251225_100000_069_0000000.fits'


def test_validFFName_lists_each_pair_once(tmp_path):

    names = _writeNight(str(tmp_path), np.uint8, n=3)

    listed = sorted(f for f in os.listdir(str(tmp_path)) if FFfile.validFFName(f))

    assert listed == sorted(names)


def test_validFFName_other_formats_unchanged():

    assert FFfile.validFFName('FF_XX0001_20251225_100000_069_0000000.fits')
    assert FFfile.validFFName('FF_XX0001_20251225_100000_069_0000000.bin')
    assert FFfile.validFFName('FF_XX0001_20251225_100000_069_0000000.png')
    assert not FFfile.validFFName('XX_XX0001_20251225_100000_069_0000000_maxpixel.png')


### Read/write ###

@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_pair_round_trip(tmp_path, dtype):

    maxpixel, avepixel = _images(dtype)
    name = _pairName()

    FFpng.writePair(str(tmp_path), name, maxpixel, avepixel,
                    meta={'nframes': 128, 'fps': 32.13, 'first_frame': 256, 'binning': 1,
                          'begin_utc': BEG_TIME, 'station': STATION})

    ff = FFfile.read(str(tmp_path), name)

    assert ff.maxpixel.dtype == dtype
    assert np.array_equal(ff.maxpixel, maxpixel)
    assert np.array_equal(ff.avepixel, avepixel)
    assert ff.stdpixel.shape == maxpixel.shape and not ff.stdpixel.any()
    assert ff.maxframe.shape == maxpixel.shape and not ff.maxframe.any()
    assert (ff.nrows, ff.ncols) == maxpixel.shape
    assert ff.nbits == 8*np.dtype(dtype).itemsize
    assert ff.nframes == 128
    assert ff.fps == pytest.approx(32.13)
    assert ff.first == 256
    assert ff.starttime == '2025-12-25T10:00:00.069539'

    # No temporary files are left behind
    assert sorted(os.listdir(str(tmp_path))) == sorted(FFpng.pairNames(name))


def test_read_with_full_filename(tmp_path):

    name = _pairName()
    maxpixel, avepixel = _images(np.uint8)
    FFpng.writePair(str(tmp_path), name, maxpixel, avepixel)

    ff = FFfile.read(None, os.path.join(str(tmp_path), name), full_filename=True)

    assert np.array_equal(ff.avepixel, avepixel)


def test_binned_pair_is_upscaled(tmp_path):

    name = _pairName()
    maxpixel, avepixel = _images(np.uint16, shape=(24, 32))
    FFpng.writePair(str(tmp_path), name, maxpixel, avepixel, meta={'binning': 2})

    ff = FFfile.read(str(tmp_path), name)

    assert ff.maxpixel.shape == (48, 64)
    assert ff.avepixel.shape == (48, 64)
    assert np.array_equal(ff.maxpixel[::2, ::2], maxpixel)
    assert (ff.nrows, ff.ncols) == (48, 64)


def test_missing_avepixel_falls_back_to_maxpixel(tmp_path):

    name = _pairName()
    maxpixel, avepixel = _images(np.uint8)
    FFpng.writePair(str(tmp_path), name, maxpixel, avepixel)
    os.remove(os.path.join(str(tmp_path), FFpng.pairNames(name)[1]))

    ff = FFfile.read(str(tmp_path), name)

    assert np.array_equal(ff.avepixel, maxpixel)


def test_missing_metadata_uses_defaults(tmp_path):

    name = _pairName()
    maxpixel, avepixel = _images(np.uint8)
    FFpng.writePair(str(tmp_path), name, maxpixel, avepixel)

    ff = FFfile.read(str(tmp_path), name)

    assert (ff.nframes, ff.fps, ff.first) == (-1, -1, 0)


def test_missing_pair_returns_none(tmp_path):
    assert FFfile.read(str(tmp_path), _pairName()) is None


def test_single_png_still_read_as_single_image(tmp_path):

    maxpixel, _ = _images(np.uint8)
    FFpng._writePNG(os.path.join(str(tmp_path), 'FF_single.png'), maxpixel)

    ff = FFfile.read(str(tmp_path), 'FF_single.png')

    assert np.array_equal(ff.maxpixel, maxpixel)
    assert np.array_equal(ff.avepixel, maxpixel)


### 8-bit display conversion ###

def test_to8bitDisplay():

    img8 = np.arange(256, dtype=np.uint8).reshape(16, 16)
    assert to8bitDisplay(img8) is img8

    img16 = (np.arange(256, dtype=np.uint16)*200).reshape(16, 16)
    out = to8bitDisplay(img16)
    assert out.dtype == np.uint8
    assert out.min() == 0 and out.max() == 255

    flat = np.full((8, 8), 1000, dtype=np.uint16)
    assert to8bitDisplay(flat).dtype == np.uint8


### Archive file selection ###

def _selectConfig(upload_mode):
    config = cr.Config()
    config.upload_mode = upload_mode
    return config


def test_selectFiles_includes_only_detected_pairs(tmp_path):

    names = _writeNight(str(tmp_path), np.uint8, n=3)
    open(os.path.join(str(tmp_path), 'CALSTARS_test.txt'), 'w').close()
    open(os.path.join(str(tmp_path), 'test_stack.jpg'), 'w').close()

    selected = selectFiles(_selectConfig(1), str(tmp_path), [names[1]])

    assert set(selected) == {'CALSTARS_test.txt', 'test_stack.jpg'} | set(FFpng.pairNames(names[1]))


def test_selectFiles_no_ffs_mode_excludes_pairs(tmp_path):

    names = _writeNight(str(tmp_path), np.uint8, n=3)

    selected = selectFiles(_selectConfig(2), str(tmp_path), [names[1]])

    assert not [f for f in selected if FFpng.isPairMaxName(f) or FFpng.isPairAveName(f)]


### Report utilities on a directory of pairs ###

def _reportConfig(dtype):
    config = cr.Config()
    config.stationID = STATION
    config.width = 64
    config.height = 48
    config.thumb_bin = 1
    config.thumb_stack = 2
    config.thumb_n_width = 4
    config.bit_depth = 8*np.dtype(dtype).itemsize
    return config


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_stack_of_pairs(tmp_path, dtype):

    from Utils.StackFFs import stackFFs

    names = _writeNight(str(tmp_path), dtype)

    stack_path, _ = stackFFs(str(tmp_path), 'jpg', subavg=True, captured_stack=True, print_progress=False)

    assert stack_path is not None and os.path.isfile(stack_path)

    # The detected stack with the brightness filter must accept the 16-bit images
    stack_path, _ = stackFFs(str(tmp_path), 'jpg', subavg=True, filter_bright=True, file_list=names[:2],
                             print_progress=False)

    assert stack_path.endswith('_stack_2_meteors.jpg')


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_thumbnails_of_pairs(tmp_path, dtype):

    import cv2
    from Utils.GenerateThumbnails import generateThumbnails

    _writeNight(str(tmp_path), dtype)

    thumb_name = generateThumbnails(str(tmp_path), _reportConfig(dtype), 'CAPTURED')

    thumbs = cv2.imread(os.path.join(str(tmp_path), thumb_name), cv2.IMREAD_GRAYSCALE)

    # The 'meteor' line must be visible, and the background must stay dark (16-bit values must not wrap
    #   around or saturate when converted to 8 bits)
    assert thumbs is not None
    assert thumbs.max() > 200
    assert np.percentile(thumbs, 90) < 128


@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason="ffmpeg is not available")
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_timelapse_of_pairs(tmp_path, dtype):

    from Utils.GenerateTimelapse import generateTimelapse

    night_dir = tmp_path/'XX0001_20251225_100000_000000'
    night_dir.mkdir()
    _writeNight(str(night_dir), dtype, n=6)

    # The timelapse is written into the night directory
    generateTimelapse(str(night_dir), keep_images=True, output_file='timelapse.mp4')

    output_path = night_dir/'timelapse.mp4'
    assert output_path.is_file() and output_path.stat().st_size > 0

    # The frames must show the 'meteor' on a dark background, not a saturated image
    import cv2
    frame = cv2.imread(str(night_dir/'temp_img_dir'/'temp_0000.jpg'), cv2.IMREAD_GRAYSCALE)
    assert frame.max() > 200
    assert np.percentile(frame, 90) < 128
