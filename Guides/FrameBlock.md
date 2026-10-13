# Frame block files (.fblk)

A file format for recording the frames of high frame rate cameras (e.g. 1920x1080 at 15-25 fps) for RMS: one
file per block of frames (e.g. 30 s), the raw 12-bit or 16-bit values of the camera, and the time, exposure, gain
and frame counter of every frame.

The reference implementation is `RMS/Formats/FrameBlock.py`. It needs only numpy (numba makes the 12-bit packing
faster if it is installed), so it can be copied into the recording software as it is.

## Why

Recording one FITS file per frame costs, for 30 s of 1920x1080 video at 15 fps:
- 450 files to create, open and close;
- every value written big-endian with an offset (BZERO 32768), and converted back when it is read;
- 16 bits per pixel for 12-bit data.

Measured on recorded 1080p frames (30 s, 449 frames), processed by RMS (matched filter, 24-core workstation, one
GPU):

| | Data per 30 s | 1 camera | 5 cameras at once | Matched-filter reading, 5 cameras |
|---|---|---|---|---|
| FITS, one file per frame | 1863 MB | 39-46 s | 98 s | 22-26 s per camera |
| Block, 16-bit | 1863 MB | 37-38 s | | |
| Block, packed 12-bit | **1398 MB** | **37-38 s** | **84 s** | **10-15 s** per camera |

The detections are identical (every measured point) from the FITS frames and from both block packings. The
matched-filter reading includes the conversion of every frame to floating point and the frames read in several
passes.

## Layout

All integers little-endian. Every section starts at a multiple of 4096 bytes, and so does every frame, so the file
can be read with a memory map or with direct I/O.

| Offset | Size | Content |
|---|---|---|
| 0 | 4096 | Header: `RMSFBLK\0` (8 bytes), version (uint32, 1), header size (uint32, 4096), then the metadata as UTF-8 JSON, padded with zeros |
| `table_offset` (4096) | `max_frames` x 32, padded to 4096 | Frame table, one record per frame |
| `data_offset` | `max_frames` x `frame_stride` | The frames, row by row |

Header JSON:

| Key | Meaning |
|---|---|
| `width`, `height` | Frame size (px) |
| `packing` | `mono12p` (12-bit packed) or `uint16` |
| `bit_depth` | Significant bits of the values (12 for a 12-bit sensor) |
| `max_frames` | Frames the file was laid out for |
| `frame_bytes`, `frame_stride` | Bytes of a frame, and the distance between frames (frame_bytes rounded up to 4096) |
| `table_offset`, `data_offset` | Offsets of the sections |
| `created_utc` | When the file was created |
| `metadata` | Free: station ID, camera, binning, sensor, lens, pixel format, site, ... |

Frame record (32 bytes):

| Field | Type | Meaning |
|---|---|---|
| `time_ns` | uint64 | Start of the integration, ns since 1970-01-01 UTC |
| `exposure_us` | uint32 | Exposure (us) |
| `gain` | float32 | Gain |
| `counter` | uint32 | Frame counter of the camera (shows dropped frames) |
| `flags` | uint32 | 1 = the frame is complete |
| `reserved` | 8 bytes | Zero |

Pixels:
- `uint16`: 2 bytes per pixel.
- `mono12p`: two pixels in three bytes, the GigE Vision / PFNC **Mono12p** layout:
  `p0 = b0 | (b1 & 0x0F) << 8`, `p1 = (b1 >> 4) | b2 << 4`. A camera set to the Mono12p pixel format sends exactly
  these bytes, so the recorder writes the camera's buffer as it is, with no conversion.
- The values are the camera's own (0-4095 for 12 bits). RMS shifts them to span 16 bits when it reads them (times 16
  for 12 bits), as the 16-bit FITS files of 12-bit cameras hold them.

Writing a frame: its pixels first, then its record with `flags = 1`. The writer reserves the full size at the start
(`posix_fallocate`, no data is written), writes to `<name>.part`, truncates after the last frame and renames at the
end, so a reader never takes a file which is still being written. A block without frames, or whose writing raised
an error, is not renamed.

A file whose writing stopped (e.g. a crash) stays as `<name>.part` and can be read up to its last complete frame.
RMS does not process `.part` files: rename a stopped block to its `.fblk` name to process it.

File names: `<station>_<YYYYMMDD>_<HHMMSS>_<microseconds>.fblk`, the time of the first frame (UTC).

## Recording

```python
from RMS.Formats.FrameBlock import FrameBlockWriter

frames_per_block = 750          # 30 s at 25 fps

with FrameBlockWriter(path, 1920, 1080, frames_per_block, packing='mono12p', bit_depth=12,
                      metadata={'station_id': 'CA0001', 'binning': 2, 'pixel_format': 'Mono12p'}) as writer:
    for i in range(frames_per_block):
        buf, t_ns, exposure_us, gain, counter = camera.nextFrame()   # the camera's Mono12p buffer
        writer.write(buf, t_ns, exposure_us=exposure_us, gain=gain, counter=counter, packed=True)
```

- With `packed=True` the camera's Mono12p buffer is written as it is. A frame given as a numpy array (12-bit
  values) is packed by the writer.
- `time_ns` is the start of the integration in UTC, e.g. the camera's timestamp converted with the offset of its
  clock, or `time.time_ns()` taken as the frame arrives, corrected for the transfer delay.
- A new block every 30 s (or any length). Writing is sequential and needs no fsync; `close(fsync=True)` flushes the
  file to the disk if it must survive a power loss.
- `python -m RMS.Formats.FrameBlock demo OUT_DIR` records two blocks of simulated frames as an example.

## Reading and converting

```
python -m RMS.Formats.FrameBlock info FILE
python -m RMS.Formats.FrameBlock convert FITS_DIR OUT_DIR [--packing mono12p|uint16] [--frames N] [--shifted auto|yes|no]
    [--overwrite]
```

The converter takes from the frames whether their values are stored shifted to span 16 bits (as 12-bit cameras
write them in 16-bit FITS files): a value above 12 bits shows they are, a value with its low 4 bits set shows they
are not. If no frame shows it (e.g. only dark frames), give it with `--shifted yes|no`.

In Python, `FrameBlockReader(path)` gives `n_frames`, `times` (s, UTC), `table` (the records), `metadata`, and
`frame(i, shift_to_16bit=False)`.

RMS reads `.fblk` files as any other input: the monitor with the file type `fblk` (`python -m
RMS.MonitorProcessFrameInterface fblk INPUT_DIR ...`, or `file_type = fblk` in a multi-camera INI), one block per
input file.
