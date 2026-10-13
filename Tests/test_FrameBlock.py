""" Tests of the frame block files (RMS.Formats.FrameBlock): packing, writing, reading, a block whose writing
    stopped, and the conversion of FITS frames. """

from __future__ import print_function, division, absolute_import

import os

import numpy as np
import pytest

import RMS.Formats.FrameBlock as fb


@pytest.mark.parametrize('use_numba', [True, False])
def test_pack12_round_trip(monkeypatch, use_numba):
    """ 12-bit values survive packing and unpacking, with the compiled and the numpy code, and the layout is the
        PFNC Mono12p one. """

    if not use_numba:
        monkeypatch.setattr(fb, 'numba', None)

    rng = np.random.default_rng(1)
    frame = rng.integers(0, 4096, (6, 8), dtype=np.uint16)
    packed = fb.pack12(frame)
    assert packed.size == 3*frame.size//2
    assert np.array_equal(fb.unpack12(packed, 8, 6), frame)
    assert np.array_equal(fb.unpack12(packed, 8, 6, shift=4), frame << 4)

    # Mono12p: two pixels 0xABC and 0x123 are the bytes 0xBC, 0x3A, 0x12
    assert fb.pack12(np.array([[0xABC, 0x123]], dtype=np.uint16)).tolist() == [0xBC, 0x3A, 0x12]


@pytest.mark.parametrize('packing', fb.PACKINGS)
def test_write_and_read(tmp_path, packing):
    """ Frames, their times and the metadata are read back as written; the file is truncated after the last frame
        and has its final name only when finished. """

    rng = np.random.default_rng(2)
    frames = [rng.integers(0, 4096, (10, 16), dtype=np.uint16) for _ in range(5)]
    path = str(tmp_path/('XX0001_20251225_010203_000000' + fb.EXTENSION))

    with fb.FrameBlockWriter(path, 16, 10, max_frames=8, packing=packing, metadata={'station_id': 'XX0001'}) as w:
        for i, frame in enumerate(frames):
            w.write(frame, 1_766_624_523_000_000_000 + i*40_000_000, exposure_us=39940, gain=15.9, counter=100 + i)
        assert not os.path.exists(path)
    assert os.path.exists(path) and not os.path.exists(path + '.part')

    reader = fb.FrameBlockReader(path)
    assert reader.n_frames == 5
    assert reader.metadata['station_id'] == 'XX0001'
    for i, frame in enumerate(frames):
        assert np.array_equal(reader.frame(i), frame)
        assert np.array_equal(reader.frame(i, shift_to_16bit=True), frame << 4)
    assert np.allclose(np.diff(reader.times), 0.04)
    assert reader.table['counter'].tolist() == [100, 101, 102, 103, 104]
    assert os.path.getsize(path) == reader.header['data_offset'] + 5*reader.header['frame_stride']


def test_stopped_writing_keeps_complete_frames(tmp_path):
    """ A block whose writing stopped (e.g. a crash) is read up to its last complete frame. """

    path = str(tmp_path/('XX0001_20251225_010203_000000' + fb.EXTENSION))
    w = fb.FrameBlockWriter(path, 16, 10, max_frames=8)
    for i in range(3):
        w.write(np.full((10, 16), i, np.uint16), i)

    # Not closed: the partial file has room for 8 frames, 3 are valid
    reader = fb.FrameBlockReader(path + '.part')
    assert reader.n_frames == 3
    assert reader.frame(2).max() == 2


def test_camera_buffer_written_as_it_is(tmp_path):
    """ A camera's Mono12p buffer is written without conversion. """

    rng = np.random.default_rng(3)
    frame = rng.integers(0, 4096, (10, 16), dtype=np.uint16)
    buffer = bytes(fb.pack12(frame))

    path = str(tmp_path/('XX0001_20251225_010203_000000' + fb.EXTENSION))
    with fb.FrameBlockWriter(path, 16, 10, max_frames=1) as w:
        w.write(buffer, 0, packed=True)

    assert np.array_equal(fb.FrameBlockReader(path).frame(0), frame)


def test_convert_fits_directory(tmp_path):
    """ FITS frames of a 12-bit camera (16-bit values, times 16, with BZERO) are converted and read back exactly,
        with the times of DATE-OBS. """

    from astropy.io import fits

    rng = np.random.default_rng(4)
    fits_dir = tmp_path/'fits'
    os.makedirs(str(fits_dir))
    originals = []
    for i in range(4):
        values = (rng.integers(0, 4096, (10, 16)).astype(np.uint16) << 4)
        originals.append(values)
        hdu = fits.PrimaryHDU(values)
        hdu.header['DATE-OBS'] = '2026-04-24T05:40:22.{:06d}+00:00'.format(66667*i)
        hdu.header['EXPTIME'] = 0.065
        hdu.header['TELESCOP'] = 'TEST01'
        hdu.writeto(str(fits_dir/'TEST01_{:d}.fits'.format(i)))

    paths = fb.convertFitsDirectory(str(fits_dir), str(tmp_path/'out'), frames_per_block=3)
    assert len(paths) == 2

    reader = fb.FrameBlockReader(paths[0])
    assert reader.n_frames == 3
    assert np.array_equal(reader.frame(1, shift_to_16bit=True), originals[1])
    assert reader.times[1] - reader.times[0] == pytest.approx(0.066667, abs=1e-6)
    assert reader.table['exposure_us'][0] == 65000


def test_rms_reads_a_block_like_its_fits_frames(tmp_path):
    """ Through the frame interface of RMS, a block file gives the same frames and frame times as the FITS frames it
        was converted from. """

    from astropy.io import fits

    import RMS.ConfigReader as cr
    from RMS.Formats.FrameInterface import detectInputTypeFile, detectInputTypeFolder

    config = cr.parse(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.config'))
    config.width, config.height = 16, 10
    config.detection_binning_factor = 1

    rng = np.random.default_rng(5)
    fits_dir = tmp_path/'fits'
    os.makedirs(str(fits_dir))
    for i in range(6):
        hdu = fits.PrimaryHDU((rng.integers(0, 4096, (10, 16)).astype(np.uint16) << 4))
        hdu.header['DATE-OBS'] = '2026-04-24T05:40:22.{:06d}+00:00'.format(66667*i)
        hdu.header['EXPTIME'] = 0.065
        hdu.writeto(str(fits_dir/'TEST01_20260424_054022_{:06d}.fits'.format(66667*i)))

    block = fb.convertFitsDirectory(str(fits_dir), str(tmp_path/'out'), station_id='TEST01')[0]

    fits_handle = detectInputTypeFolder(str(fits_dir), config, detection=True)
    block_handle = detectInputTypeFile(block, config, detection=True)
    assert block_handle.total_frames == fits_handle.total_frames == 6
    for i in range(6):
        fits_handle.setFrame(i)
        block_handle.setFrame(i)
        assert np.array_equal(block_handle.loadFrame(), fits_handle.loadFrame())
        assert abs((block_handle.currentFrameTime(dt_obj=True)
                    - fits_handle.currentFrameTime(dt_obj=True)).total_seconds()) < 1e-6
    assert block_handle.fps == pytest.approx(15.0, rel=1e-3)
