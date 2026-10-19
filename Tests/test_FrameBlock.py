""" Tests of the frame block files (RMS.Formats.FrameBlock): packing, writing, reading, a block whose writing
    stopped, and the conversion of FITS frames. """

from __future__ import print_function, division, absolute_import

import os
import datetime
import time

import numpy as np
import pytest

import RMS.Formats.FrameBlock as fb


@pytest.mark.parametrize('use_numba', [True, False])
def test_pack12_round_trip(monkeypatch, use_numba):
    """ 12-bit values survive packing and unpacking, with the compiled and the numpy code, and the layout is the
        PFNC Mono12p one. """

    # Without numba, the numpy code is used
    if not use_numba:
        monkeypatch.setattr(fb, 'numba', None)

    # Random 12-bit values packed into 3 bytes per pair, and unpacked as they were (also shifted to 16 bits)
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

    # 5 frames of a block laid out for 8, at 25 fps
    rng = np.random.default_rng(2)
    frames = [rng.integers(0, 4096, (10, 16), dtype=np.uint16) for _ in range(5)]
    path = str(tmp_path/('XX0001_20251225_010203_000000' + fb.EXTENSION))

    # The file has its final name only after the block is closed
    with fb.FrameBlockWriter(path, 16, 10, max_frames=8, packing=packing, metadata={'station_id': 'XX0001'}) as w:
        for i, frame in enumerate(frames):
            w.write(frame, 1_766_624_523_000_000_000 + i*40_000_000, exposure_us=39940, gain=15.9, counter=100 + i)
        assert not os.path.exists(path)
    assert os.path.exists(path) and not os.path.exists(path + '.part')

    # The frames, the metadata, the times and the counters are read back, and the file ends after the last frame
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

    # 3 frames written, and the writer never closed (as after a crash)
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

    # A frame as the camera sends it in the Mono12p pixel format
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 4096, (10, 16), dtype=np.uint16)
    buffer = bytes(fb.pack12(frame))

    # Written as given, and read back as the frame
    path = str(tmp_path/('XX0001_20251225_010203_000000' + fb.EXTENSION))
    with fb.FrameBlockWriter(path, 16, 10, max_frames=1) as w:
        w.write(buffer, 0, packed=True)

    assert np.array_equal(fb.FrameBlockReader(path).frame(0), frame)


def test_convert_fits_directory(tmp_path):
    """ FITS frames of a 12-bit camera (16-bit values, times 16, with BZERO) are converted and read back exactly,
        with the times of DATE-OBS. """

    from astropy.io import fits

    # 4 FITS frames of 12-bit values times 16, at 15 fps
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

    # Blocks of 3 frames: 2 blocks
    paths = fb.convertFitsDirectory(str(fits_dir), str(tmp_path/'out'), frames_per_block=3)
    assert len(paths) == 2

    # The values (shifted back to 16 bits), the frame times and the exposure as in the FITS files
    reader = fb.FrameBlockReader(paths[0])
    assert reader.n_frames == 3
    assert np.array_equal(reader.frame(1, shift_to_16bit=True), originals[1])
    assert reader.times[1] - reader.times[0] == pytest.approx(0.066667, abs=1e-6)
    assert reader.table['exposure_us'][0] == 65000


def test_rms_reads_a_block_like_its_fits_frames(tmp_path):
    """ Through the frame interface of RMS, a block file gives the same frames as the FITS frames it was converted
        from, and their times at the middle of the exposure: the FITS frames (DATE-OBS) and the block keep the start
        of the integration. """

    from astropy.io import fits

    import RMS.ConfigReader as cr
    from RMS.Formats.FrameInterface import detectInputTypeFile, detectInputTypeFolder

    # The default config, for frames of 16x10 px without binning
    config = cr.parse(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.config'))
    config.width, config.height = 16, 10
    config.detection_binning_factor = 1

    # 6 FITS frames at 15 fps, named as the recorder names them, and the block converted from them
    rng = np.random.default_rng(5)
    fits_dir = tmp_path/'fits'
    os.makedirs(str(fits_dir))
    for i in range(6):
        hdu = fits.PrimaryHDU((rng.integers(0, 4096, (10, 16)).astype(np.uint16) << 4))
        hdu.header['DATE-OBS'] = '2026-04-24T05:40:22.{:06d}+00:00'.format(66667*i)
        hdu.header['EXPTIME'] = 0.065
        hdu.writeto(str(fits_dir/'TEST01_20260424_054022_{:06d}.fits'.format(66667*i)))

    block = fb.convertFitsDirectory(str(fits_dir), str(tmp_path/'out'), station_id='TEST01')[0]

    # Every frame is the same through both inputs, half the exposure later in the block, and the frame rate comes
    #   from the frame times
    fits_handle = detectInputTypeFolder(str(fits_dir), config, detection=True)
    block_handle = detectInputTypeFile(block, config, detection=True)
    assert block_handle.total_frames == fits_handle.total_frames == 6
    for i in range(6):
        fits_handle.setFrame(i)
        block_handle.setFrame(i)
        assert np.array_equal(block_handle.loadFrame(), fits_handle.loadFrame())
        assert abs((block_handle.currentFrameTime(dt_obj=True)
                    - fits_handle.currentFrameTime(dt_obj=True)).total_seconds() - 0.065/2) < 1e-6
    assert block_handle.fps == pytest.approx(15.0, rel=1e-3)


def _writeFitsFrames(fits_dir, frames, date_obs, names=None):
    """ Write FITS frames (one per file) with the given DATE-OBS values. """

    from astropy.io import fits

    # One file per frame, with the time, the exposure and the station in the header
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

    # A frame with values up to 4095 and a dark one with values up to 255, both stored times 16, in either order
    rng = np.random.default_rng(6)
    bright = rng.integers(0, 4096, (10, 16)).astype(np.uint16)
    dark = rng.integers(0, 256, (10, 16)).astype(np.uint16)
    originals = [dark, bright] if dark_first else [bright, dark]
    _writeFitsFrames(tmp_path/'fits', [values << 4 for values in originals],
                     ['2026-04-24T05:40:22.{:06d}+00:00'.format(66667*i) for i in range(2)])

    # Both frames come back as their 12-bit values
    reader = fb.FrameBlockReader(fb.convertFitsDirectory(str(tmp_path/'fits'), str(tmp_path/'out'))[0])
    for i, values in enumerate(originals):
        assert np.array_equal(reader.frame(i), values)


def test_convert_takes_date_obs_without_offset_as_utc(tmp_path, monkeypatch):
    """ A DATE-OBS without a time zone is UTC, whatever the time zone of the computer. """

    import calendar

    # The conversion runs in a time zone which is not UTC
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

    # The time of the frame is the DATE-OBS taken as UTC
    assert reader.times[0] == pytest.approx(calendar.timegm((2026, 4, 24, 5, 40, 22)) + 0.5, abs=1e-6)


def test_convert_orders_frames_by_date_obs(tmp_path):
    """ The frames are written in the order of their DATE-OBS, not of their file names. """

    # 3 frames whose file names are in the reverse order of their times
    frames = [np.full((10, 16), 4093 + i, np.uint16) for i in range(3)]
    _writeFitsFrames(tmp_path/'fits', frames, ['2026-04-24T05:40:22.{:06d}+00:00'.format(66667*i) for i in range(3)],
                     names=['c.fits', 'b.fits', 'a.fits'])

    # The block has them in the order of their times

    reader = fb.FrameBlockReader(fb.convertFitsDirectory(str(tmp_path/'fits'), str(tmp_path/'out'),
                                                         station_id='TEST01')[0])
    assert [int(reader.frame(i).max()) for i in range(3)] == [4093, 4094, 4095]
    assert np.all(np.diff(reader.times) > 0)


def test_failed_writing_does_not_publish_the_block(tmp_path):
    """ An error while a block is written leaves it as <name>.part, readable up to its last frame, and never
        under its final name. """

    # The recording stops with an error after the first frame
    path = str(tmp_path/('XX0001_20251225_010203_000000' + fb.EXTENSION))
    with pytest.raises(RuntimeError):
        with fb.FrameBlockWriter(path, 16, 10, max_frames=8) as w:
            w.write(np.full((10, 16), 7, np.uint16), 0)
            raise RuntimeError('camera lost')

    # Only the partial file exists, with its one frame
    assert not os.path.exists(path)
    assert fb.FrameBlockReader(path + '.part').n_frames == 1


@pytest.mark.parametrize('packing', fb.PACKINGS)
def test_frame_of_the_wrong_size_is_rejected(tmp_path, packing):
    """ A frame with fewer or more pixels than the block's frames is refused, and nothing is recorded for it. """

    path = str(tmp_path/('XX0001_20251225_010203_000000' + fb.EXTENSION))
    # A frame one column short and one a row too long
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
    # One pixel above 12 bits
    with fb.FrameBlockWriter(path, 16, 10, max_frames=2, packing=packing) as w:
        frame = np.zeros((10, 16), np.uint16)
        frame[3, 4] = 4096
        with pytest.raises(ValueError):
            w.write(frame, 0)
        assert w.n_frames == 0


def test_empty_block_is_not_published(tmp_path):
    """ A block closed before its first frame is not given its final name. """

    # Opened and closed without a frame
    path = str(tmp_path/('XX0001_20251225_010203_000000' + fb.EXTENSION))
    fb.FrameBlockWriter(path, 16, 10, max_frames=2).close()
    assert not os.path.exists(path)


def test_inconsistent_header_is_rejected(tmp_path):
    """ A header whose layout does not match the frame size is refused. """

    import json
    import struct

    # A valid block
    path = str(tmp_path/('XX0001_20251225_010203_000000' + fb.EXTENSION))
    with fb.FrameBlockWriter(path, 16, 10, max_frames=2) as w:
        w.write(np.zeros((10, 16), np.uint16), 0)

    # Its header rewritten with frames 0 bytes apart
    header = fb.readHeader(path)
    header['frame_stride'] = 0
    text = json.dumps({k: v for k, v in header.items() if k != 'version'}).encode()
    with open(path, 'r+b') as f:
        f.write(fb.MAGIC + struct.pack('<II', fb.VERSION, fb.HEADER_SIZE) + text
                + bytes(fb.HEADER_SIZE - 16 - len(text)))

    with pytest.raises(ValueError):
        fb.FrameBlockReader(path)


@pytest.mark.parametrize('first', ['zeros', 'multiples_of_16'])
def test_convert_does_not_guess_from_an_ambiguous_first_frame(tmp_path, first):
    """ Unshifted 12-bit frames whose first frame tells nothing about the scaling (all zeros, or only multiples
        of 16) are not shifted down: the scaling is taken from the first frame which shows it. """

    # A first frame which could be either way, and a frame with values with their low bits set (not shifted)
    rng = np.random.default_rng(7)
    first_frame = np.zeros((10, 16), np.uint16) if first == 'zeros' else \
        (rng.integers(0, 256, (10, 16)).astype(np.uint16) << 4)
    later = rng.integers(0, 4096, (10, 16)).astype(np.uint16)
    later[0, 0] = 4095
    _writeFitsFrames(tmp_path/'fits', [first_frame, later],
                     ['2026-04-24T05:40:22.{:06d}+00:00'.format(66667*i) for i in range(2)])

    # Both frames keep their values
    reader = fb.FrameBlockReader(fb.convertFitsDirectory(str(tmp_path/'fits'), str(tmp_path/'out'))[0])
    assert np.array_equal(reader.frame(0), first_frame)
    assert np.array_equal(reader.frame(1), later)


def test_convert_refuses_to_guess_when_no_frame_shows_the_scaling(tmp_path):
    """ When no frame shows whether the values are stored shifted, the scaling must be given. """

    # Frames of zeros and of 16 only (multiples of 16 either way)
    frames = [np.full((10, 16), 16*i, np.uint16) for i in range(2)]
    _writeFitsFrames(tmp_path/'fits', frames, ['2026-04-24T05:40:22.{:06d}+00:00'.format(66667*i) for i in range(2)])

    # Refused without the scaling, converted with it
    with pytest.raises(ValueError):
        fb.convertFitsDirectory(str(tmp_path/'fits'), str(tmp_path/'out'))

    reader = fb.FrameBlockReader(fb.convertFitsDirectory(str(tmp_path/'fits'), str(tmp_path/'out'),
                                                         shifted=False)[0])
    assert int(reader.frame(1).max()) == 16


@pytest.mark.parametrize('packing', fb.PACKINGS)
def test_negative_values_are_rejected(tmp_path, packing):
    """ A negative value is refused instead of being stored wrapped around. """

    path = str(tmp_path/('XX0001_20251225_010203_000000' + fb.EXTENSION))
    # One negative pixel
    with fb.FrameBlockWriter(path, 16, 10, max_frames=2, packing=packing) as w:
        frame = np.zeros((10, 16), np.int32)
        frame[2, 2] = -1
        with pytest.raises(ValueError):
            w.write(frame, 0)
        assert w.n_frames == 0


def test_convert_names_blocks_uniquely_when_times_repeat(tmp_path):
    """ Blocks whose first frames have the same DATE-OBS (a coarse clock) get different names, so none is
        overwritten. """

    # 4 frames with the same time, in blocks of 2
    frames = [np.full((10, 16), 4093 - i, np.uint16) for i in range(4)]
    _writeFitsFrames(tmp_path/'fits', frames, ['2026-04-24T05:40:22+00:00']*4)

    # Two blocks of different names, with all 4 frames
    paths = fb.convertFitsDirectory(str(tmp_path/'fits'), str(tmp_path/'out'), frames_per_block=2,
                                    station_id='TEST01')
    assert len(set(paths)) == 2
    assert sum(fb.FrameBlockReader(p).n_frames for p in paths) == 4


@pytest.mark.parametrize('packing, bit_depth', [('uint16', 17), ('uint16', 0), ('mono12p', 13)])
def test_bit_depth_beyond_the_storage_is_rejected(tmp_path, packing, bit_depth):
    """ A bit depth the packing can't hold (or not positive) is refused. """

    # Refused before any file is created
    path = str(tmp_path/('XX0001_20251225_010203_000000' + fb.EXTENSION))
    with pytest.raises(ValueError):
        fb.FrameBlockWriter(path, 16, 10, max_frames=2, packing=packing, bit_depth=bit_depth)
    assert not os.path.exists(path + '.part')


def test_rms_opens_an_upper_case_extension(tmp_path):
    """ A block named with an upper-case extension is opened as a block, as the monitor accepts it. """

    import RMS.ConfigReader as cr
    from RMS.Formats.FrameInterface import detectInputTypeFile

    # A block of two frames named .FBLK
    config = cr.parse(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.config'))
    config.width, config.height = 16, 10
    path = str(tmp_path/'XX0001_20251225_010203_000000.FBLK')
    with fb.FrameBlockWriter(path, 16, 10, max_frames=2) as w:
        w.write(np.zeros((10, 16), np.uint16), 1_766_624_523_000_000_000)
        w.write(np.zeros((10, 16), np.uint16), 1_766_624_523_040_000_000)

    # Opened by the block reader of RMS
    assert detectInputTypeFile(path, config, detection=True).input_type == 'fblk'


def test_convert_16_bit_values_needs_no_scaling(tmp_path):
    """ 16-bit values are converted as they are, without asking whether they are shifted. """

    # Frames which could be shifted 12-bit values, converted as 16-bit values
    frames = [np.full((10, 16), 16*i, np.uint16) for i in range(2)]
    _writeFitsFrames(tmp_path/'fits', frames, ['2026-04-24T05:40:22.{:06d}+00:00'.format(66667*i) for i in range(2)])

    # Converted without the scaling, the values as they were
    reader = fb.FrameBlockReader(fb.convertFitsDirectory(str(tmp_path/'fits'), str(tmp_path/'out'), packing='uint16',
                                                         bit_depth=16)[0])
    assert int(reader.frame(1).max()) == 16


def test_convert_takes_fits_of_any_case(tmp_path):
    """ Frames named .FITS, .fits and .fit are all converted. """

    # 3 frames with the three extensions
    frames = [np.full((10, 16), 4093 - i, np.uint16) for i in range(3)]
    _writeFitsFrames(tmp_path/'fits', frames, ['2026-04-24T05:40:22.{:06d}+00:00'.format(66667*i) for i in range(3)],
                     names=['a.FITS', 'b.fits', 'c.fit'])

    # All in the block
    reader = fb.FrameBlockReader(fb.convertFitsDirectory(str(tmp_path/'fits'), str(tmp_path/'out'),
                                                         station_id='TEST01')[0])
    assert reader.n_frames == 3


def test_convert_keeps_existing_blocks(tmp_path):
    """ A block already in the output directory is not replaced, unless asked. """

    # A first conversion
    frames = [np.full((10, 16), 4093, np.uint16)]
    _writeFitsFrames(tmp_path/'fits', frames, ['2026-04-24T05:40:22+00:00'])
    path = fb.convertFitsDirectory(str(tmp_path/'fits'), str(tmp_path/'out'), station_id='TEST01')[0]

    # A second one is refused and leaves the block as it was
    with pytest.raises(FileExistsError):
        fb.convertFitsDirectory(str(tmp_path/'fits'), str(tmp_path/'out'), station_id='TEST01')
    assert fb.FrameBlockReader(path).n_frames == 1

    # With overwrite, the block is replaced
    assert fb.convertFitsDirectory(str(tmp_path/'fits'), str(tmp_path/'out'), station_id='TEST01',
                                   overwrite=True) == [path]


def test_short_writes_are_completed(tmp_path, monkeypatch):
    """ A write which stores only part of the bytes (as the system may do) is continued until all are written. """

    # Every write stores only a third of its bytes
    write = os.pwrite

    def shortWrite(fd, data, offset):
        data = memoryview(data).cast('B')
        return write(fd, data[:max(1, len(data)//3)], offset)

    # Two frames written that way
    monkeypatch.setattr(os, 'pwrite', shortWrite)
    rng = np.random.default_rng(8)
    frames = [rng.integers(0, 4096, (10, 16), dtype=np.uint16) for _ in range(2)]
    path = str(tmp_path/('XX0001_20251225_010203_000000' + fb.EXTENSION))
    with fb.FrameBlockWriter(path, 16, 10, max_frames=2) as w:
        for i, frame in enumerate(frames):
            w.write(frame, i, counter=7 + i)
    monkeypatch.setattr(os, 'pwrite', write)

    # The frames and their records are complete
    reader = fb.FrameBlockReader(path)
    assert reader.n_frames == 2
    assert all(np.array_equal(reader.frame(i), f) for i, f in enumerate(frames))
    assert reader.table['counter'].tolist() == [7, 8]


def test_written_without_positional_writes(tmp_path, monkeypatch):
    """ Where the system has no positional write (os.pwrite, e.g. Windows), the block is written by seeking. """

    # Two frames written without os.pwrite
    monkeypatch.delattr(os, 'pwrite')
    rng = np.random.default_rng(9)
    frames = [rng.integers(0, 4096, (10, 16), dtype=np.uint16) for _ in range(2)]
    path = str(tmp_path/('XX0001_20251225_010203_000000' + fb.EXTENSION))
    with fb.FrameBlockWriter(path, 16, 10, max_frames=3) as w:
        for i, frame in enumerate(frames):
            w.write(frame, i)

    # Read back as written
    reader = fb.FrameBlockReader(path)
    assert reader.n_frames == 2
    assert all(np.array_equal(reader.frame(i), f) for i, f in enumerate(frames))


@pytest.mark.skipif(os.name != 'posix', reason='Directories are synchronized on POSIX systems only')
def test_fsync_also_syncs_the_directory(tmp_path, monkeypatch):
    """ With fsync, the file and then its directory (the new name) are flushed to the disk. """

    # Every flush is recorded with the kind of what was flushed (file or directory)
    synced = []
    fsync = os.fsync
    monkeypatch.setattr(os, 'fsync', lambda fd: (synced.append(os.fstat(fd).st_mode), fsync(fd)))

    # A block closed with fsync
    path = str(tmp_path/('XX0001_20251225_010203_000000' + fb.EXTENSION))
    w = fb.FrameBlockWriter(path, 16, 10, max_frames=1)
    w.write(np.zeros((10, 16), np.uint16), 0)
    w.close(fsync=True)

    # The file first, then the directory
    import stat
    assert [stat.S_ISDIR(mode) for mode in synced] == [False, True]


@pytest.mark.skipif(not hasattr(os, 'posix_fallocate'), reason='No preallocation on this system')
@pytest.mark.parametrize('code, fails', [('ENOSPC', True), ('EOPNOTSUPP', False)])
def test_preallocation_errors(tmp_path, monkeypatch, code, fails):
    """ A full disk when the block is reserved is an error (with the file descriptor closed and no file left);
        a file system without preallocation falls back to setting the size. """

    import errno

    # Preallocation fails with the given error; the open files are counted before
    def fallocate(fd, offset, length):
        raise OSError(getattr(errno, code), code)

    monkeypatch.setattr(os, 'posix_fallocate', fallocate)
    fds = len(os.listdir('/proc/self/fd')) if os.path.isdir('/proc/self/fd') else None
    path = str(tmp_path/('XX0001_20251225_010203_000000' + fb.EXTENSION))

    # A full disk: no writer, no file, no open descriptor
    if fails:
        with pytest.raises(OSError):
            fb.FrameBlockWriter(path, 16, 10, max_frames=2)
        assert not os.path.exists(path + '.part')
        if fds is not None:
            assert len(os.listdir('/proc/self/fd')) == fds
    # No preallocation on the file system: the block is written anyway
    else:
        with fb.FrameBlockWriter(path, 16, 10, max_frames=2) as w:
            w.write(np.zeros((10, 16), np.uint16), 0)
        assert fb.FrameBlockReader(path).n_frames == 1


@pytest.mark.parametrize('frames', [0, -1])
def test_convert_refuses_non_positive_block_lengths(tmp_path, frames):
    """ A block length which is not positive is refused instead of converting nothing or everything. """

    # One frame, converted with blocks of 0 or -1 frames
    _writeFitsFrames(tmp_path/'fits', [np.full((10, 16), 4093, np.uint16)], ['2026-04-24T05:40:22+00:00'])
    with pytest.raises(ValueError):
        fb.convertFitsDirectory(str(tmp_path/'fits'), str(tmp_path/'out'), frames_per_block=frames)


def test_matched_filter_finds_blocks_in_a_directory(tmp_path):
    """ The matched filter takes the frame blocks of a directory as its inputs. """

    from RMS.MatchedFilterDetection import findInputFiles

    # A block and another file in a directory: only the block is an input
    open(str(tmp_path/'XX0001_20251225_010203_000000.fblk'), 'wb').close()
    open(str(tmp_path/'notes.txt'), 'w').close()
    assert [os.path.basename(p) for p in findInputFiles([str(tmp_path)])] == ['XX0001_20251225_010203_000000.fblk']


@pytest.mark.parametrize('failing', ['ftruncate', 'fsync'])
def test_failed_finishing_closes_the_file(tmp_path, monkeypatch, failing):
    """ An error while a block is finished closes its file descriptor and leaves the block unpublished. """

    def openFiles():
        """ The number of open files of the process (None where it can't be listed). """

        return len(os.listdir('/proc/self/fd')) if os.path.isdir('/proc/self/fd') else None

    # A block with one frame, and the open files before it
    path = str(tmp_path/('XX0001_20251225_010203_000000' + fb.EXTENSION))
    n_open = openFiles()
    w = fb.FrameBlockWriter(path, 16, 10, max_frames=2)
    w.write(np.zeros((10, 16), np.uint16), 0)

    # The truncation or the flush fails when the block is closed
    def fail(*args):
        raise OSError('disk error')

    monkeypatch.setattr(os, failing, fail)
    with pytest.raises(OSError):
        w.close(fsync=True)
    monkeypatch.undo()

    # The file is closed and stays unpublished
    assert w.fd is None
    assert openFiles() == n_open
    assert not os.path.exists(path) and os.path.exists(path + '.part')


def test_failed_close_never_touches_the_descriptor_again(tmp_path, monkeypatch):
    """ When closing the file itself fails, the writer forgets its descriptor (the system has released it, and the
        number can belong to another file by now): a later abort doesn't truncate or close anything. """

    # A block with one frame
    path = str(tmp_path/('XX0001_20251225_010203_000000' + fb.EXTENSION))
    w = fb.FrameBlockWriter(path, 16, 10, max_frames=2)
    w.write(np.zeros((10, 16), np.uint16), 0)

    # The descriptor is closed, but the close reports an error (as it can on network file systems)
    close = os.close

    def failingClose(fd):
        close(fd)
        raise OSError('close failed')

    monkeypatch.setattr(os, 'close', failingClose)
    with pytest.raises(OSError):
        w.close()
    monkeypatch.undo()
    assert w.fd is None

    # Nothing is done with the old descriptor afterwards
    touched = []
    monkeypatch.setattr(os, 'ftruncate', lambda fd, size: touched.append(fd))
    monkeypatch.setattr(os, 'close', lambda fd: touched.append(fd))
    w.abort()
    assert touched == []


@pytest.mark.skipif(os.name != 'posix', reason='Directories are synchronized on POSIX systems only')
def test_failed_directory_sync_after_publishing_warns(tmp_path, monkeypatch):
    """ If the directory can't be flushed after the block got its final name, the block is published: a warning,
        not an error which would look like the block was not written. """

    # A block with one frame
    path = str(tmp_path/('XX0001_20251225_010203_000000' + fb.EXTENSION))
    w = fb.FrameBlockWriter(path, 16, 10, max_frames=1)
    w.write(np.zeros((10, 16), np.uint16), 0)

    # The file is flushed, the directory not
    import stat
    fsync = os.fsync

    def failingDirectorySync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError('directory sync failed')
        fsync(fd)

    # A warning, and the block is there under its final name
    monkeypatch.setattr(os, 'fsync', failingDirectorySync)
    with pytest.warns(RuntimeWarning):
        w.close(fsync=True)
    assert fb.FrameBlockReader(path).n_frames == 1



@pytest.mark.parametrize('block_time, offset', [(None, 0.0475), ('start', 0.0), ('middle', 0.0475),
                                                ('end', 0.095)])
def test_block_frame_times_at_the_configured_point_of_the_exposure(tmp_path, block_time, offset):
    """ The frame times of a block file are those of the point of the exposures given by frame_block_time, by
        default the middle (where a moving object is measured), from the start of the integration and the exposure
        of every frame. """

    import RMS.ConfigReader as cr
    from RMS.Formats.FrameInterface import detectInputTypeFile

    config = cr.parse(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.config'))
    config.width, config.height = 16, 10
    config.detection_binning_factor = 1
    if block_time is not None:
        config.frame_block_time = block_time

    # 4 frames at 10 fps, started every 0.1 s, exposed 95 ms
    path = str(tmp_path/'TEST01_20261018_011528_000000.fblk')
    t0 = 1792199728000000000
    with fb.FrameBlockWriter(path, 16, 10, 4) as writer:
        for i in range(4):
            writer.write(np.full((10, 16), 100*i, np.uint16), t0 + i*100000000, exposure_us=95000)

    handle = detectInputTypeFile(path, config, detection=True)
    for i in range(4):
        t = handle.currentFrameTime(frame_no=i, dt_obj=True)
        unix = (t - datetime.datetime(1970, 1, 1)).total_seconds() if t.tzinfo is None else t.timestamp()
        assert unix == pytest.approx(t0/1e9 + i*0.1 + offset, abs=2e-6)
    assert handle.fps == pytest.approx(10.0, rel=1e-6)



@pytest.mark.parametrize('value, parsed', [('start', 'start'), (' End ', 'end'), ('middle', 'middle'),
                                           ('centre', 'middle')])
def test_frame_block_time_option(tmp_path, value, parsed):
    """ frame_block_time is read from the Capture section; a value other than start, middle or end is the middle. """

    import RMS.ConfigReader as cr

    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.config')
    with open(src) as f:
        text = f.read()
    assert 'frame_block_time: middle' in text
    path = tmp_path/'.config'
    path.write_text(text.replace('frame_block_time: middle', 'frame_block_time: ' + value))

    assert cr.parse(str(path)).frame_block_time == parsed



def test_block_fps_from_the_starts_of_the_frames_with_changing_exposures(tmp_path):
    """ The frame rate of a block comes from the starts of the frames: with exposures that change from frame to
        frame the middles of the exposures are not evenly spaced, the starts are. """

    import RMS.ConfigReader as cr
    from RMS.Formats.FrameInterface import detectInputTypeFile

    config = cr.parse(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.config'))
    config.width, config.height = 16, 10
    config.detection_binning_factor = 1

    # Frames started every 0.1 s, exposed 0 and 80 ms in turn (middles 140/60 ms apart)
    path = str(tmp_path/'TEST01_20261018_011528_000000.fblk')
    t0 = 1792199728000000000
    with fb.FrameBlockWriter(path, 16, 10, 20) as writer:
        for i in range(20):
            writer.write(np.zeros((10, 16), np.uint16), t0 + i*100000000, exposure_us=80000*(i%2))

    handle = detectInputTypeFile(path, config, detection=True)
    assert handle.fps == pytest.approx(10.0, rel=1e-6)
