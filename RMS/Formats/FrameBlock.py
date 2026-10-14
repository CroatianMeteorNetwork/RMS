""" Frame block files (.fblk): blocks of raw video frames, one file per block (e.g. 30 s), for fast recording and
reading of high frame rate cameras (e.g. 1920x1080 at 15-25 fps).

Why: one FITS file per frame costs a file open and a big-endian, offset conversion for every frame (2.1 s for 30 s of
1080p video in one thread). A block file is read with a memory map in one go, as the 16-bit or packed 12-bit values
of the camera (0.16-0.3 s for the same 30 s).

This module only needs numpy (numba, if available, makes the 12-bit packing several times faster), so it can be
copied into the recording software as it is. See Guides/FrameBlock.md for the format and the recorder example.

Layout (little-endian, every section starts at a multiple of 4096 bytes):

    0           header, HEADER_SIZE bytes: MAGIC (8 bytes), version (uint32), header size (uint32), then the
                metadata as UTF-8 JSON, padded with zeros
    table_offset
                the frame table: max_frames records of FRAME_RECORD (32 bytes each): the start of the integration
                (uint64, ns since 1970-01-01 UTC), the exposure (uint32, us), the gain (float32), the frame counter
                of the camera (uint32), flags (uint32, FLAG_VALID once the frame is written), reserved (8 bytes)
    data_offset
                max_frames frames, frame_stride bytes apart: the pixels row by row, either 16-bit unsigned
                ('uint16') or 12-bit packed two pixels in three bytes ('mono12p', the GigE Vision / PFNC Mono12p
                layout which cameras send: p0 = b0 | (b1 & 0x0F) << 8, p1 = (b1 >> 4) | b2 << 4)

A frame is written before its record, and the record is marked valid last, so a file whose writing stopped (e.g. a
crash) is read up to the last complete frame. A finished file is truncated after its last frame.

Usage:
    python -m RMS.Formats.FrameBlock info FILE
    python -m RMS.Formats.FrameBlock convert FITS_DIR OUT_DIR [--packing mono12p|uint16] [--bits 12] [--frames N]
        [--shifted auto|yes|no] [--overwrite]
    python -m RMS.Formats.FrameBlock demo OUT_DIR       (a simulated camera recording to block files)
"""

from __future__ import print_function, division, absolute_import

import errno
import json
import os
import struct
import time
import warnings

import numpy as np

try:
    import numba
except ImportError:
    numba = None


# File signature and version
MAGIC = b'RMSFBLK\x00'
VERSION = 1

# Alignment of the sections and of the frames, and the size of the header
ALIGNMENT = 4096
HEADER_SIZE = 4096

# File extension
EXTENSION = '.fblk'

# Record of a frame in the table
FRAME_RECORD = np.dtype([('time_ns', '<u8'), ('exposure_us', '<u4'), ('gain', '<f4'), ('counter', '<u4'),
                         ('flags', '<u4'), ('reserved', '<u8')])
FLAG_VALID = 1

# Packings of the pixels
PACKINGS = ('uint16', 'mono12p')


def _align(n):
    """ n rounded up to a multiple of ALIGNMENT. """

    return (n + ALIGNMENT - 1)//ALIGNMENT*ALIGNMENT


def frameBytes(width, height, packing):
    """ Number of bytes of the pixels of a frame.

    Arguments:
        width: [int] Frame width (px).
        height: [int] Frame height (px).
        packing: [str] 'uint16' or 'mono12p'.

    Return:
        [int] Bytes per frame (without the alignment padding).
    """

    # 16-bit values take 2 bytes per pixel
    if packing == 'uint16':
        return 2*width*height

    # Mono12p packs two pixels in three bytes, so a frame needs an even number of pixels
    if packing == 'mono12p':
        if (width*height) % 2:
            raise ValueError('mono12p needs an even number of pixels')
        return 3*width*height//2

    raise ValueError('Unknown packing: {}'.format(packing))


### 12-bit packing (PFNC Mono12p) ###

def _pack12Numpy(frame):
    """ Pack 12-bit values (uint16, 0-4095) into Mono12p bytes. """

    # The pixels in pairs, every pair packed into three bytes: the low 8 bits of the first pixel, its high 4 bits
    #   with the low 4 bits of the second pixel, and the high 8 bits of the second pixel
    p = frame.reshape(-1, 2).astype(np.uint16)
    out = np.empty((p.shape[0], 3), dtype=np.uint8)
    out[:, 0] = p[:, 0] & 0xFF
    out[:, 1] = ((p[:, 0] >> 8) & 0x0F) | ((p[:, 1] & 0x0F) << 4)
    out[:, 2] = (p[:, 1] >> 4) & 0xFF

    return out.ravel()


def _unpack12Numpy(buf, n_pixels, shift):
    """ Unpack Mono12p bytes to uint16 values, shifted left by shift bits. """

    # Every three bytes are a pair of pixels (the reverse of _pack12Numpy)
    b = np.frombuffer(buf, dtype=np.uint8, count=3*n_pixels//2).reshape(-1, 3).astype(np.uint16)
    out = np.empty((b.shape[0], 2), dtype=np.uint16)
    out[:, 0] = b[:, 0] | ((b[:, 1] & 0x0F) << 8)
    out[:, 1] = (b[:, 1] >> 4) | (b[:, 2] << 4)

    # The values shifted up, e.g. to span 16 bits
    if shift:
        out <<= shift

    return out.ravel()


# The same packing and unpacking compiled, every pair of pixels in parallel (several times faster than numpy)
if numba is not None:

    @numba.njit(parallel=True, cache=True)
    def _pack12Numba(values, out):
        """ Pack 12-bit values into Mono12p bytes (see _pack12Numpy). """

        for i in numba.prange(values.shape[0]//2):
            a = values[2*i]
            c = values[2*i + 1]
            out[3*i] = a & 0xFF
            out[3*i + 1] = ((a >> 8) & 0x0F) | ((c & 0x0F) << 4)
            out[3*i + 2] = (c >> 4) & 0xFF

    @numba.njit(parallel=True, cache=True)
    def _unpack12Numba(buf, out, shift):
        """ Unpack Mono12p bytes to uint16 values shifted left by shift bits (see _unpack12Numpy). """

        for i in numba.prange(out.shape[0]//2):
            b0 = np.uint16(buf[3*i])
            b1 = np.uint16(buf[3*i + 1])
            b2 = np.uint16(buf[3*i + 2])
            out[2*i] = (b0 | ((b1 & 0x0F) << 8)) << shift
            out[2*i + 1] = ((b1 >> 4) | (b2 << 4)) << shift


def pack12(frame):
    """ Pack a frame of 12-bit values (0-4095, any integer type) into Mono12p bytes.

    Arguments:
        frame: [ndarray] Frame (height x width).

    Return:
        [ndarray] uint8 array of 3*width*height/2 bytes.
    """

    # The values in one row, packed by numpy if numba is not available
    values = np.ascontiguousarray(frame, dtype=np.uint16).ravel()
    if numba is None:
        return _pack12Numpy(values)

    # Packed by the compiled kernel
    out = np.empty(3*values.size//2, dtype=np.uint8)
    _pack12Numba(values, out)

    return out


def unpack12(buf, width, height, shift=0):
    """ Unpack Mono12p bytes into a uint16 frame.

    Arguments:
        buf: [bytes-like] The packed bytes of the frame.
        width: [int] Frame width (px).
        height: [int] Frame height (px).

    Keyword arguments:
        shift: [int] Shift the values left by this many bits (4 makes them span 16 bits). 0 by default.

    Return:
        [ndarray] uint16 frame (height x width).
    """

    # Unpacked by numpy if numba is not available
    n = width*height
    if numba is None:
        return _unpack12Numpy(buf, n, shift).reshape(height, width)

    # Unpacked by the compiled kernel
    out = np.empty(n, dtype=np.uint16)
    _unpack12Numba(np.frombuffer(buf, dtype=np.uint8, count=3*n//2), out, shift)

    return out.reshape(height, width)


### Writing ###

# Errors of posix_fallocate meaning that the file system can't preallocate (the size is then only set)
PREALLOCATION_UNSUPPORTED = tuple(getattr(errno, name) for name in ('EINVAL', 'EOPNOTSUPP', 'ENOTSUP', 'ENOSYS')
                                  if hasattr(errno, name))


def _pwriteAll(fd, data, offset):
    """ Write all bytes at the offset: the system may write only a part of them at once (e.g. a full disk
        writes none, which is an error), so the writing continues until all are written.
    """

    view = memoryview(data).cast('B')
    while len(view):

        # Without positional writes (e.g. on Windows) the file is written at the offset after seeking to it (one
        #   writer per file, so nothing moves the position in between)
        if hasattr(os, 'pwrite'):
            n = os.pwrite(fd, view, offset)
        else:
            os.lseek(fd, offset, os.SEEK_SET)
            n = os.write(fd, view)
        # Nothing written is an error (e.g. a full disk), otherwise the rest is written next
        if n <= 0:
            raise OSError('Nothing could be written at byte {}'.format(offset))
        view = view[n:]
        offset += n


class FrameBlockWriter(object):
    def __init__(self, path, width, height, max_frames, packing='mono12p', bit_depth=12, metadata=None):
        """ Write frames to a block file as they arrive (see the module docstring for the layout).

        Use as a context manager, or call close() at the end:

            with FrameBlockWriter(path, 1920, 1080, 750) as writer:
                for frame, t_ns, exposure_us, gain, counter in camera:
                    writer.write(frame, t_ns, exposure_us, gain, counter)

        Arguments:
            path: [str] Path of the file (written to path + '.part' and renamed at close, so a reader never takes
                a file which is being written).
            width: [int] Frame width (px).
            height: [int] Frame height (px).
            max_frames: [int] Largest number of frames of the block (the file is truncated after the last frame).

        Keyword arguments:
            packing: [str] 'mono12p' (12-bit packed, 25% smaller) or 'uint16'. 'mono12p' by default.
            bit_depth: [int] Significant bits of the values (12 for a 12-bit sensor). 12 by default.
            metadata: [dict] Anything to keep with the block (station, camera, binning, sensor, lens, ...), saved
                in the header. None by default.
        """

        # The packing has to hold the values: at most 12 bits for Mono12p, 16 bits for uint16
        if packing not in PACKINGS:
            raise ValueError('Unknown packing: {}'.format(packing))
        if not (1 <= bit_depth <= (12 if packing == 'mono12p' else 16)):
            raise ValueError('{} can not hold {}-bit values'.format(packing, bit_depth))

        # The layout: the header, the frame table after it, and the frames after the table, every section and every
        #   frame starting at a multiple of ALIGNMENT bytes
        self.path = path
        self.part_path = path + '.part'
        self.width, self.height = int(width), int(height)
        self.max_frames = int(max_frames)
        self.packing = packing
        self.bit_depth = int(bit_depth)
        self.frame_bytes = frameBytes(self.width, self.height, packing)
        self.frame_stride = _align(self.frame_bytes)
        self.table_offset = HEADER_SIZE
        self.data_offset = self.table_offset + _align(self.max_frames*FRAME_RECORD.itemsize)
        self.n_frames = 0

        # The header describes the layout and keeps the metadata, as JSON in the first HEADER_SIZE bytes
        header = {'width': self.width, 'height': self.height, 'packing': packing, 'bit_depth': self.bit_depth,
                  'max_frames': self.max_frames, 'frame_bytes': self.frame_bytes, 'frame_stride': self.frame_stride,
                  'table_offset': self.table_offset, 'data_offset': self.data_offset,
                  'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  'metadata': metadata or {}}
        text = json.dumps(header, sort_keys=True).encode('utf-8')
        if 16 + len(text) > HEADER_SIZE:
            raise ValueError('The metadata is too large for the header ({} bytes)'.format(HEADER_SIZE - 16))

        # The file with its full size reserved (no data is written for the reserved space), the header and an
        #   empty frame table
        self.fd = os.open(self.part_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC | getattr(os, 'O_BINARY', 0), 0o644)
        try:
            total = self.data_offset + self.max_frames*self.frame_stride

            # Without preallocation (the system or the file system doesn't have it) the size is only set; a full
            #   disk is an error now rather than in the middle of the recording
            try:
                os.posix_fallocate(self.fd, 0, total)
            except AttributeError:
                os.ftruncate(self.fd, total)
            except OSError as e:
                if e.errno not in PREALLOCATION_UNSUPPORTED:
                    raise
                os.ftruncate(self.fd, total)

            # The header, and the frame table cleared (no frame is valid yet)
            _pwriteAll(self.fd, MAGIC + struct.pack('<II', VERSION, HEADER_SIZE) + text, 0)
            _pwriteAll(self.fd, bytes(self.data_offset - self.table_offset), self.table_offset)

        # A block which could not be started leaves no file and no open descriptor (forgotten before it is closed,
        #   see close)
        except Exception:
            fd, self.fd = self.fd, None
            os.close(fd)
            os.remove(self.part_path)
            raise

    def write(self, frame, time_ns, exposure_us=0, gain=0.0, counter=0, packed=False):
        """ Append a frame.

        Arguments:
            frame: [ndarray or bytes] The frame (height x width integers), or with packed=True the bytes of the
                frame exactly as they are stored (e.g. the Mono12p buffer of the camera, written without a copy).
            time_ns: [int] Start of the integration, ns since 1970-01-01 UTC (e.g. time.time_ns() or the camera's
                timestamp converted to UTC).

        Keyword arguments:
            exposure_us: [int] Exposure (us). 0 by default.
            gain: [float] Gain. 0 by default.
            counter: [int] Frame counter of the camera (to see dropped frames). 0 by default.
            packed: [bool] The frame is given as the stored bytes. False by default.

        Return:
            [int] Index of the frame in the block.
        """

        # The block was laid out for max_frames frames
        if self.n_frames >= self.max_frames:
            raise IndexError('The block is full ({} frames)'.format(self.max_frames))

        # The pixels as stored: the bytes given (e.g. the camera's Mono12p buffer, which has to be a whole frame),
        #   or a frame of values of the right size, packed here
        if packed:
            data = memoryview(frame).cast('B')
            if data.nbytes != self.frame_bytes:
                raise ValueError('{} bytes given, a frame is {}'.format(data.nbytes, self.frame_bytes))
        else:
            if np.shape(frame) != (self.height, self.width):
                raise ValueError('The frame is {}, expected {}x{}'.format(np.shape(frame), self.height, self.width))

            # Values outside the bit depth would be stored wrapped around
            if (np.min(frame) < 0) or (np.max(frame) > 2**self.bit_depth - 1):
                raise ValueError('The frame has values outside 0-{}'.format(2**self.bit_depth - 1))
            if self.packing == 'mono12p':
                data = pack12(frame)
            else:
                data = np.ascontiguousarray(frame, dtype='<u2')

        # The frame first, then its record, marked valid last: a reader never takes a frame which isn't complete
        i = self.n_frames
        _pwriteAll(self.fd, data, self.data_offset + i*self.frame_stride)
        record = np.zeros(1, dtype=FRAME_RECORD)
        record['time_ns'] = int(time_ns)
        record['exposure_us'] = int(exposure_us)
        record['gain'] = float(gain)
        record['counter'] = int(counter)
        record['flags'] = FLAG_VALID
        _pwriteAll(self.fd, record.tobytes(), self.table_offset + i*FRAME_RECORD.itemsize)
        self.n_frames += 1

        return i

    def close(self, fsync=False):
        """ Finish the file: truncate it after the last frame and give it its final name.

        Keyword arguments:
            fsync: [bool] Flush the file to the disk first (takes longer; only needed to survive a power loss).
                False by default.
        """

        if self.fd is None:
            return

        # A block without frames is not published
        if self.n_frames == 0:
            self.abort()
            return

        # The file is closed also when finishing it fails; it then stays unpublished as <name>.part. The descriptor
        #   is forgotten before it is closed: after a failed close the system may give its number to another file,
        #   which must never be truncated or closed by this writer
        fd, self.fd = self.fd, None
        try:
            os.ftruncate(fd, self.data_offset + self.n_frames*self.frame_stride)
            if fsync:
                os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(self.part_path, self.path)

        # The new name is in the directory, which is flushed too (on POSIX systems, where a directory can be
        #   opened), so the block survives a power loss under its final name. The block is published by now, so a
        #   failure here is a warning: an error would look like the block was not written
        if fsync and (os.name == 'posix'):
            try:
                dir_fd = os.open(os.path.dirname(os.path.abspath(self.path)), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError as e:
                warnings.warn('{} is written, but its directory could not be flushed to the disk: {}'.format(
                    self.path, e), RuntimeWarning)

    def abort(self):
        """ Stop writing without publishing the file: it stays as <name>.part, readable up to its last frame, and
            a complete file of the same name is not replaced.
        """

        if self.fd is None:
            return

        # The descriptor is forgotten before it is closed (see close)
        fd, self.fd = self.fd, None
        try:
            os.ftruncate(fd, self.data_offset + self.n_frames*self.frame_stride)
        finally:
            os.close(fd)

    def __enter__(self):
        """ Use the writer as a context manager (see __exit__). """

        return self

    def __exit__(self, exc_type, *args):
        """ Close the block at the end of a with block, or abort it if the with block raised an error. """

        # Only a block written without an error gets its final name
        if exc_type is None:
            self.close()
        else:
            self.abort()


### Reading ###

def readHeader(path):
    """ Read the header of a block file.

    Arguments:
        path: [str] Path of the file.

    Return:
        [dict] The header (see FrameBlockWriter), with 'version'.
    """

    with open(path, 'rb') as f:
        head = f.read(HEADER_SIZE)

    # The signature, the version and the size of the header, then the JSON description (padded with zeros)
    if head[:8] != MAGIC:
        raise ValueError('{} is not a frame block file'.format(path))
    version, header_size = struct.unpack('<II', head[8:16])
    header = json.loads(head[16:header_size].rstrip(b'\x00').decode('utf-8'))
    header['version'] = version

    # The layout must agree with the frame size, so frames are never read across each other
    try:
        expected_bytes = frameBytes(header['width'], header['height'], header['packing'])
        table_bytes = header['max_frames']*FRAME_RECORD.itemsize
        consistent = ((header['frame_bytes'] == expected_bytes)
                      and (header['frame_stride'] >= expected_bytes > 0)
                      and (header['table_offset'] >= header_size)
                      and (header['data_offset'] >= header['table_offset'] + table_bytes))
    except (KeyError, TypeError, ValueError):
        consistent = False
    if not consistent:
        raise ValueError('{} has an inconsistent header'.format(path))

    return header


class FrameBlockReader(object):
    def __init__(self, path):
        """ Read the frames of a block file through a memory map.

        Arguments:
            path: [str] Path of the file.
        """

        # The header, with the size of the frames, their packing and the metadata
        self.path = path
        self.header = readHeader(path)
        h = self.header
        self.width, self.height = h['width'], h['height']
        self.packing, self.bit_depth = h['packing'], h['bit_depth']
        self.metadata = h.get('metadata', {})

        # The frames which were written completely: the valid records, and only as many as the file holds
        self.mm = np.memmap(path, dtype=np.uint8, mode='r')
        table = np.frombuffer(self.mm, dtype=FRAME_RECORD, count=h['max_frames'], offset=h['table_offset'])

        # The frames are written in order, so the valid ones are those before the first invalid record
        valid = (table['flags'] & FLAG_VALID) > 0
        n_valid = int(np.argmin(valid)) if not np.all(valid) else len(valid)

        # A file cut short holds only the frames whose bytes are all in it
        n_in_file = max(0, (len(self.mm) - h['data_offset'])//h['frame_stride'])
        if (len(self.mm) - h['data_offset']) % h['frame_stride'] >= h['frame_bytes']:
            n_in_file += 1
        self.n_frames = min(n_valid, n_in_file)
        self.table = np.array(table[:self.n_frames])

    @property
    def times(self):
        """ Start of the integration of every frame, s since 1970-01-01 UTC (float64). """

        return self.table['time_ns']/1e9

    def frameBuffer(self, i):
        """ The stored bytes of frame i (a view into the file). """

        if not (0 <= i < self.n_frames):
            raise IndexError('Frame {} is not in the block ({} frames)'.format(i, self.n_frames))

        # The frames are frame_stride bytes apart after the frame table
        start = self.header['data_offset'] + i*self.header['frame_stride']

        return self.mm[start:start + self.header['frame_bytes']]

    def frame(self, i, shift_to_16bit=False):
        """ Frame i as uint16.

        Arguments:
            i: [int] Frame index.

        Keyword arguments:
            shift_to_16bit: [bool] Shift the values to span 16 bits (12-bit values times 16), as the 16-bit
                containers of 12-bit cameras hold them (e.g. FITS files). False by default.

        Return:
            [ndarray] uint16 frame (height x width).
        """

        # The values are shifted up by the bits which the bit depth doesn't use, if asked
        shift = (16 - self.bit_depth) if shift_to_16bit else 0
        buf = self.frameBuffer(i)

        # Packed frames are unpacked (and shifted at the same time)
        if self.packing == 'mono12p':
            return unpack12(buf, self.width, self.height, shift=shift)

        # 16-bit frames are read as they are stored, as a copy (not a view into the file)
        frame = np.frombuffer(buf, dtype='<u2').reshape(self.height, self.width)
        if shift:
            return frame << shift

        return frame.copy()

    def close(self):
        """ Release the memory map. """

        self.mm = None


### Conversion and example ###

def convertFitsDirectory(fits_dir, out_dir, packing='mono12p', bit_depth=12, frames_per_block=None,
                         station_id=None, shifted=None, overwrite=False):
    """ Convert a directory of FITS frames (one frame per file, DATE-OBS the start of the exposure) into block
        files.

    Arguments:
        fits_dir: [str] Directory of the FITS frames.
        out_dir: [str] Directory of the block files.

    Keyword arguments:
        packing: [str] 'mono12p' or 'uint16'. 'mono12p' by default.
        bit_depth: [int] Significant bits of the values (the FITS values are shifted down to them). 12 by default.
        frames_per_block: [int] Frames per block file, all frames of the directory in one block if None.
        station_id: [str] Station ID for the names of the files, from the FITS header (TELESCOP) if None.
        shifted: [bool] The values are stored shifted up to span 16 bits (a 12-bit camera in a 16-bit FITS holds
            them times 16). If None, taken from the first frame which shows it (a value above bit_depth bits, or
            a value with its low bits set); an error if no frame shows it.
        overwrite: [bool] Replace the blocks of the same names in out_dir (an error otherwise). False by default.

    Return:
        [list] Paths of the written block files.
    """

    import datetime

    from astropy.io import fits

    def _utc(date_obs):
        """ The time of a DATE-OBS value as a datetime in UTC. """

        # DATE-OBS is UTC, also when it has no time zone
        t = datetime.datetime.fromisoformat(date_obs.strip().replace('Z', '+00:00'))
        if t.tzinfo is None:
            t = t.replace(tzinfo=datetime.timezone.utc)
        return t

    # The frames in the order of their times
    paths = [os.path.join(fits_dir, name) for name in os.listdir(fits_dir)
             if name.lower().endswith(('.fits', '.fit'))]
    if not paths:
        raise ValueError('No FITS files in {}'.format(fits_dir))
    times = {path: _utc(fits.getheader(path)['DATE-OBS']) for path in paths}
    paths.sort(key=lambda path: times[path])

    # Whether the values are stored shifted up, decided once so all frames are scaled the same way: a value above
    #   bit_depth bits shows they are, a value with its low bits set shows they are not; a frame of zeros or of
    #   multiples of 2**shift alone shows neither
    shift = 16 - bit_depth
    if shift == 0:
        shifted = False
    if shifted is None:
        for path in paths:
            values = np.asarray(fits.getdata(path)).astype(np.int64)
            if values.max() > (2**bit_depth - 1):
                shifted = True
            elif np.any(values & (2**shift - 1)):
                shifted = False
            if shifted is not None:
                break
        if shifted is None:
            raise ValueError('The frames do not show whether their values are shifted to 16 bits, give shifted')
    # The number of frames of a block, all frames in one block if it is not given
    if (frames_per_block is not None) and (frames_per_block < 1):
        raise ValueError('A block holds at least one frame, {} given'.format(frames_per_block))
    frames_per_block = frames_per_block or len(paths)
    os.makedirs(out_dir, exist_ok=True)

    # The frames block by block. A block is named by the station and the time of its first frame, as the
    #   recorder names it
    written = []
    for start in range(0, len(paths), frames_per_block):
        chunk = paths[start:start + frames_per_block]
        header0 = fits.getheader(chunk[0])
        station = station_id or str(header0.get('TELESCOP', 'XX0000')).strip()
        t0 = times[chunk[0]]
        name = '{}_{}{}'.format(station, t0.strftime('%Y%m%d_%H%M%S_%f'), EXTENSION)

        # Blocks starting at the same time (a clock coarser than the frames) get names a microsecond apart, so
        #   none replaces another; the times of the frames are in the frame table
        while os.path.join(out_dir, name) in written:
            t0 += datetime.timedelta(microseconds=1)
            name = '{}_{}{}'.format(station, t0.strftime('%Y%m%d_%H%M%S_%f'), EXTENSION)

        # A block of an earlier conversion is kept, unless it is to be replaced
        if os.path.exists(os.path.join(out_dir, name)) and not overwrite:
            raise FileExistsError('{} exists (convert with overwrite to replace it)'.format(
                os.path.join(out_dir, name)))
        # The FITS header values which describe the camera and the site are kept as the metadata of the block
        metadata = {k: header0[k] for k in ('TELESCOP', 'SITELAT', 'SITELONG', 'SITEELEV', 'XBINNING',
                                            'YBINNING', 'GAIN', 'FRAMERAT', 'PIXFMT') if k in header0}
        metadata['station_id'] = station
        metadata['source'] = 'converted from FITS frames'

        # Every frame of the block with its values, time, exposure and gain from its FITS file
        width, height = header0['NAXIS1'], header0['NAXIS2']
        with FrameBlockWriter(os.path.join(out_dir, name), width, height, len(chunk), packing=packing,
                              bit_depth=bit_depth, metadata=metadata) as writer:
            for i, path in enumerate(chunk):
                with fits.open(path, memmap=False) as hdul:
                    hdu = hdul[0]
                    values = np.asarray(hdu.data).astype(np.int64)
                    exposure = float(hdu.header.get('EXPTIME', 0.0))
                    gain = float(str(hdu.header.get('GAIN', 0)).strip() or 0)

                # The values down to bit_depth bits, the same way for every frame
                if shifted:
                    if np.any(values & (2**shift - 1)):
                        raise ValueError('{} has values which are not shifted to 16 bits'.format(path))
                    values = values >> shift
                if (values.min() < 0) or (values.max() > (2**bit_depth - 1)):
                    raise ValueError('{} has values outside {} bits'.format(path, bit_depth))

                # The time of the frame in ns UTC; the frame number in the directory stands in for the counter
                time_ns = int(round(times[path].timestamp()*1e9))
                writer.write(values, time_ns, exposure_us=round(exposure*1e6), gain=gain, counter=start + i)
        written.append(os.path.join(out_dir, name))

    return written


def demoRecorder(out_dir, width=1920, height=1080, fps=25.0, block_seconds=30.0, n_blocks=2, packing='mono12p'):
    """ Example of a recorder: frames arrive from a camera (simulated here) and are streamed to block files, a new
        block every block_seconds. In a real recorder, replace the simulated frame by the camera's buffer (for a
        camera sending Mono12p, pass its bytes with packed=True, no copy or conversion is needed) and its
        timestamp, exposure, gain and frame counter.

    Arguments:
        out_dir: [str] Directory of the block files.

    Keyword arguments:
        width, height: [int] Frame size. 1920x1080 by default.
        fps: [float] Frame rate. 25 by default.
        block_seconds: [float] Length of a block file (s). 30 by default.
        n_blocks: [int] Number of blocks to record. 2 by default.
        packing: [str] 'mono12p' or 'uint16'. 'mono12p' by default.

    Return:
        [list] Paths of the written block files.
    """

    # The simulated camera starts now, at the given frame rate
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    frames_per_block = int(round(fps*block_seconds))
    t_ns = time.time_ns()
    written = []
    counter = 0

    # A new block file every block_seconds, named by the time of its first frame
    for b in range(n_blocks):
        name = 'XX0001_{}{}'.format(time.strftime('%Y%m%d_%H%M%S', time.gmtime(t_ns/1e9)) + '_{:06d}'.format(
            (t_ns//1000) % 1000000), EXTENSION)
        path = os.path.join(out_dir, name)
        with FrameBlockWriter(path, width, height, frames_per_block, packing=packing, bit_depth=12,
                              metadata={'station_id': 'XX0001', 'source': 'demoRecorder'}) as writer:
            for _ in range(frames_per_block):

                # A frame from the camera: here sky noise around 200 ADU of 4095
                frame = np.clip(rng.normal(200, 10, (height, width)), 0, 4095).astype(np.uint16)
                writer.write(frame, t_ns, exposure_us=int(1e6/fps), gain=1.0, counter=counter)

                # The next frame one frame period later
                t_ns += int(1e9/fps)
                counter += 1
        written.append(path)

    return written


if __name__ == '__main__':

    import argparse

    # Three commands: info, convert and demo
    parser = argparse.ArgumentParser(description='Frame block files (.fblk).')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('info', help='Print the header and the frames of a block file.')
    p.add_argument('file')

    p = sub.add_parser('convert', help='Convert a directory of FITS frames into block files.')
    p.add_argument('fits_dir')
    p.add_argument('out_dir')
    p.add_argument('--packing', default='mono12p', choices=PACKINGS)
    p.add_argument('--bits', type=int, default=12)
    p.add_argument('--frames', type=int, default=None, help='Frames per block (all by default).')
    p.add_argument('--shifted', choices=['auto', 'yes', 'no'], default='auto',
                   help='The FITS values are shifted to span 16 bits (taken from the frames by default).')
    p.add_argument('--overwrite', action='store_true', help='Replace existing blocks of the same names.')

    p = sub.add_parser('demo', help='Record simulated frames to block files.')
    p.add_argument('out_dir')
    p.add_argument('--packing', default='mono12p', choices=PACKINGS)

    args = parser.parse_args()

    # The header, the number of frames and the frame rate from the frame times
    if args.command == 'info':
        reader = FrameBlockReader(args.file)
        print(json.dumps(reader.header, indent=2, sort_keys=True))
        t = reader.times
        print('frames: {}'.format(reader.n_frames))
        if reader.n_frames > 1:
            print('first frame: {:.6f}, last: {:.6f}, fps: {:.4f}'.format(t[0], t[-1],
                                                                          (reader.n_frames - 1)/(t[-1] - t[0])))

    # A directory of FITS frames to block files
    elif args.command == 'convert':
        for path in convertFitsDirectory(args.fits_dir, args.out_dir, packing=args.packing, bit_depth=args.bits,
                                         frames_per_block=args.frames,
                                         shifted={'auto': None, 'yes': True, 'no': False}[args.shifted],
                                         overwrite=args.overwrite):
            print(path)

    # Simulated frames recorded to block files
    elif args.command == 'demo':
        for path in demoRecorder(args.out_dir, packing=args.packing):
            print(path)
