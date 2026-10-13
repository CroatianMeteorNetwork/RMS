""" Tests of the frame block files (RMS.Formats.FrameBlock): packing, writing, reading, a block whose writing
    stopped, and the conversion of FITS frames. """

from __future__ import print_function, division, absolute_import

import os
import time

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


def _writeFitsFrames(fits_dir, frames, date_obs, names=None):
    """ Write FITS frames (one per file) with the given DATE-OBS values. """

    from astropy.io import fits

    os.makedirs(str(fits_dir))
    for i, (values, t) in enumerate(zip(frames, date_obs)):
        hdu = fits.PrimaryHDU(values)
        hdu.header['DATE-OBS'] = t
        hdu.header['EXPTIME'] = 0.065
        hdu.header['TELESCOP'] = 'TEST01'
        hdu.writeto(str(fits_dir/(names[i] if names else 'TEST01_{:d}.fits'.format(i))))


@pytest.mark.parametrize('dark_first', [True, False])
def test_convert_scales_dark_frames_like_the_others(tmp_path, dark_first):
    """ A dark frame of a 12-bit camera stored times 16 (all values at most 4095 once stored) is shifted down as
        the other frames of the directory, not kept 16 times brighter. """

    rng = np.random.default_rng(6)
    bright = rng.integers(0, 4096, (10, 16)).astype(np.uint16)
    dark = rng.integers(0, 256, (10, 16)).astype(np.uint16)
    originals = [dark, bright] if dark_first else [bright, dark]
    _writeFitsFrames(tmp_path/'fits', [values << 4 for values in originals],
                     ['2026-04-24T05:40:22.{:06d}+00:00'.format(66667*i) for i in range(2)])

    reader = fb.FrameBlockReader(fb.convertFitsDirectory(str(tmp_path/'fits'), str(tmp_path/'out'))[0])
    for i, values in enumerate(originals):
        assert np.array_equal(reader.frame(i), values)


def test_convert_takes_date_obs_without_offset_as_utc(tmp_path, monkeypatch):
    """ A DATE-OBS without a time zone is UTC, whatever the time zone of the computer. """

    import calendar

    monkeypatch.setenv('TZ', 'America/Toronto')
    if hasattr(time, 'tzset'):
        time.tzset()
    try:
        _writeFitsFrames(tmp_path/'fits', [np.zeros((10, 16), np.uint16)], ['2026-04-24T05:40:22.500000'])
        reader = fb.FrameBlockReader(fb.convertFitsDirectory(str(tmp_path/'fits'), str(tmp_path/'out'),
                                                             shifted=False)[0])
    finally:
        monkeypatch.delenv('TZ')
        if hasattr(time, 'tzset'):
            time.tzset()

    assert reader.times[0] == pytest.approx(calendar.timegm((2026, 4, 24, 5, 40, 22)) + 0.5, abs=1e-6)


def test_convert_orders_frames_by_date_obs(tmp_path):
    """ The frames are written in the order of their DATE-OBS, not of their file names. """

    frames = [np.full((10, 16), 4093 + i, np.uint16) for i in range(3)]
    _writeFitsFrames(tmp_path/'fits', frames, ['2026-04-24T05:40:22.{:06d}+00:00'.format(66667*i) for i in range(3)],
                     names=['c.fits', 'b.fits', 'a.fits'])

    reader = fb.FrameBlockReader(fb.convertFitsDirectory(str(tmp_path/'fits'), str(tmp_path/'out'),
                                                         station_id='TEST01')[0])
    assert [int(reader.frame(i).max()) for i in range(3)] == [4093, 4094, 4095]
    assert np.all(np.diff(reader.times) > 0)


def test_failed_writing_does_not_publish_the_block(tmp_path):
    """ An error while a block is written leaves it as <name>.part, readable up to its last frame, and never
        under its final name. """

    path = str(tmp_path/('XX0001_20251225_010203_000000' + fb.EXTENSION))
    with pytest.raises(RuntimeError):
        with fb.FrameBlockWriter(path, 16, 10, max_frames=8) as w:
            w.write(np.full((10, 16), 7, np.uint16), 0)
            raise RuntimeError('camera lost')

    assert not os.path.exists(path)
    assert fb.FrameBlockReader(path + '.part').n_frames == 1


@pytest.mark.parametrize('packing', fb.PACKINGS)
def test_frame_of_the_wrong_size_is_rejected(tmp_path, packing):
    """ A frame with fewer or more pixels than the block's frames is refused, and nothing is recorded for it. """

    path = str(tmp_path/('XX0001_20251225_010203_000000' + fb.EXTENSION))
    with fb.FrameBlockWriter(path, 16, 10, max_frames=4, packing=packing) as w:
        for shape in [(10, 15), (11, 16)]:
            with pytest.raises(ValueError):
                w.write(np.zeros(shape, np.uint16), 0)
        assert w.n_frames == 0


@pytest.mark.parametrize('packing', fb.PACKINGS)
def test_values_above_the_bit_depth_are_rejected(tmp_path, packing):
    """ A value above the bit depth (e.g. a 16-bit frame given to a 12-bit block) is refused instead of being
        stored wrapped around. """

    path = str(tmp_path/('XX0001_20251225_010203_000000' + fb.EXTENSION))
    with fb.FrameBlockWriter(path, 16, 10, max_frames=2, packing=packing) as w:
        frame = np.zeros((10, 16), np.uint16)
        frame[3, 4] = 4096
        with pytest.raises(ValueError):
            w.write(frame, 0)
        assert w.n_frames == 0


def test_empty_block_is_not_published(tmp_path):
    """ A block closed before its first frame is not given its final name. """

    path = str(tmp_path/('XX0001_20251225_010203_000000' + fb.EXTENSION))
    fb.FrameBlockWriter(path, 16, 10, max_frames=2).close()
    assert not os.path.exists(path)


def test_inconsistent_header_is_rejected(tmp_path):
    """ A header whose layout does not match the frame size is refused. """

    import json
    import struct

    path = str(tmp_path/('XX0001_20251225_010203_000000' + fb.EXTENSION))
    with fb.FrameBlockWriter(path, 16, 10, max_frames=2) as w:
        w.write(np.zeros((10, 16), np.uint16), 0)

    header = fb.readHeader(path)
    header['frame_stride'] = 0
    text = json.dumps({k: v for k, v in header.items() if k != 'version'}).encode()
    with open(path, 'r+b') as f:
        f.write(fb.MAGIC + struct.pack('<II', fb.VERSION, fb.HEADER_SIZE) + text + bytes(fb.HEADER_SIZE - 16 - len(text)))

    with pytest.raises(ValueError):
        fb.FrameBlockReader(path)


@pytest.mark.parametrize('first', ['zeros', 'multiples_of_16'])
def test_convert_does_not_guess_from_an_ambiguous_first_frame(tmp_path, first):
    """ Unshifted 12-bit frames whose first frame tells nothing about the scaling (all zeros, or only multiples
        of 16) are not shifted down: the scaling is taken from the first frame which shows it. """

    rng = np.random.default_rng(7)
    first_frame = np.zeros((10, 16), np.uint16) if first == 'zeros' else \
        (rng.integers(0, 256, (10, 16)).astype(np.uint16) << 4)
    later = rng.integers(0, 4096, (10, 16)).astype(np.uint16)
    later[0, 0] = 4095
    _writeFitsFrames(tmp_path/'fits', [first_frame, later],
                     ['2026-04-24T05:40:22.{:06d}+00:00'.format(66667*i) for i in range(2)])

    reader = fb.FrameBlockReader(fb.convertFitsDirectory(str(tmp_path/'fits'), str(tmp_path/'out'))[0])
    assert np.array_equal(reader.frame(0), first_frame)
    assert np.array_equal(reader.frame(1), later)


def test_convert_refuses_to_guess_when_no_frame_shows_the_scaling(tmp_path):
    """ When no frame shows whether the values are stored shifted, the scaling must be given. """

    frames = [np.full((10, 16), 16*i, np.uint16) for i in range(2)]
    _writeFitsFrames(tmp_path/'fits', frames, ['2026-04-24T05:40:22.{:06d}+00:00'.format(66667*i) for i in range(2)])

    with pytest.raises(ValueError):
        fb.convertFitsDirectory(str(tmp_path/'fits'), str(tmp_path/'out'))

    reader = fb.FrameBlockReader(fb.convertFitsDirectory(str(tmp_path/'fits'), str(tmp_path/'out'), shifted=False)[0])
    assert int(reader.frame(1).max()) == 16


@pytest.mark.parametrize('packing', fb.PACKINGS)
def test_negative_values_are_rejected(tmp_path, packing):
    """ A negative value is refused instead of being stored wrapped around. """

    path = str(tmp_path/('XX0001_20251225_010203_000000' + fb.EXTENSION))
    with fb.FrameBlockWriter(path, 16, 10, max_frames=2, packing=packing) as w:
        frame = np.zeros((10, 16), np.int32)
        frame[2, 2] = -1
        with pytest.raises(ValueError):
            w.write(frame, 0)
        assert w.n_frames == 0


def test_convert_names_blocks_uniquely_when_times_repeat(tmp_path):
    """ Blocks whose first frames have the same DATE-OBS (a coarse clock) get different names, so none is
        overwritten. """

    frames = [np.full((10, 16), 4093 - i, np.uint16) for i in range(4)]
    _writeFitsFrames(tmp_path/'fits', frames, ['2026-04-24T05:40:22+00:00']*4)

    paths = fb.convertFitsDirectory(str(tmp_path/'fits'), str(tmp_path/'out'), frames_per_block=2,
                                    station_id='TEST01')
    assert len(set(paths)) == 2
    assert sum(fb.FrameBlockReader(p).n_frames for p in paths) == 4


@pytest.mark.parametrize('packing, bit_depth', [('uint16', 17), ('uint16', 0), ('mono12p', 13)])
def test_bit_depth_beyond_the_storage_is_rejected(tmp_path, packing, bit_depth):
    """ A bit depth the packing can't hold (or not positive) is refused. """

    path = str(tmp_path/('XX0001_20251225_010203_000000' + fb.EXTENSION))
    with pytest.raises(ValueError):
        fb.FrameBlockWriter(path, 16, 10, max_frames=2, packing=packing, bit_depth=bit_depth)
    assert not os.path.exists(path + '.part')


def test_rms_opens_an_upper_case_extension(tmp_path):
    """ A block named with an upper-case extension is opened as a block, as the monitor accepts it. """

    import RMS.ConfigReader as cr
    from RMS.Formats.FrameInterface import detectInputTypeFile

    config = cr.parse(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.config'))
    config.width, config.height = 16, 10
    path = str(tmp_path/'XX0001_20251225_010203_000000.FBLK')
    with fb.FrameBlockWriter(path, 16, 10, max_frames=2) as w:
        w.write(np.zeros((10, 16), np.uint16), 1_766_624_523_000_000_000)
        w.write(np.zeros((10, 16), np.uint16), 1_766_624_523_040_000_000)

    assert detectInputTypeFile(path, config, detection=True).input_type == 'fblk'


def test_convert_16_bit_values_needs_no_scaling(tmp_path):
    """ 16-bit values are converted as they are, without asking whether they are shifted. """

    frames = [np.full((10, 16), 16*i, np.uint16) for i in range(2)]
    _writeFitsFrames(tmp_path/'fits', frames, ['2026-04-24T05:40:22.{:06d}+00:00'.format(66667*i) for i in range(2)])

    reader = fb.FrameBlockReader(fb.convertFitsDirectory(str(tmp_path/'fits'), str(tmp_path/'out'), packing='uint16',
                                                         bit_depth=16)[0])
    assert int(reader.frame(1).max()) == 16


def test_convert_takes_fits_of_any_case(tmp_path):
    """ Frames named .FITS, .fits and .fit are all converted. """

    frames = [np.full((10, 16), 4093 - i, np.uint16) for i in range(3)]
    _writeFitsFrames(tmp_path/'fits', frames, ['2026-04-24T05:40:22.{:06d}+00:00'.format(66667*i) for i in range(3)],
                     names=['a.FITS', 'b.fits', 'c.fit'])

    reader = fb.FrameBlockReader(fb.convertFitsDirectory(str(tmp_path/'fits'), str(tmp_path/'out'),
                                                         station_id='TEST01')[0])
    assert reader.n_frames == 3
