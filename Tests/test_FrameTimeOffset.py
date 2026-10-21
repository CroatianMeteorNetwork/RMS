"""
Tests of the correction of the timestamps of recorded frames (frame_time_offset in the config): the true time of a
frame is its timestamp plus the correction, applied where an input gets its times, so the time of the beginning,
the times of the frames and the names of the frames all follow it. A correction of 0 changes nothing.
"""

import datetime
import os

import cv2
import numpy as np
import pytest

import RMS.ConfigReader as cr
from RMS.Formats.FrameInterface import InputTypeDFN, InputTypeImages, InputTypeUWOVid


OFFSET = 0.084

# Time of the beginning of the recordings of the tests
T0 = datetime.datetime(2026, 10, 20, 1, 2, 3, 500000)


def makeConfig(offset):
    """ A default config with the given correction of the timestamps. """

    config = cr.Config()
    config.frame_time_offset = offset

    return config


def writeVid(path, n_frames=32, width=64, height=64, fps=32.0):
    """ A vid file of blank frames with their timestamps (UWO vid format: the header of every frame is in its
        first bytes).
    """

    with open(path, 'wb') as f:
        for i in range(n_frames):

            # The timestamp of the frame
            t = (T0 - datetime.datetime(1970, 1, 1)).total_seconds() + i/fps
            ts = int(t)
            tu = int(round((t - ts)*1e6))

            frame = np.zeros(height*width, dtype=np.uint16)
            header = b''.join([
                np.array([0x10000, width*height*2, 116, 0, i], dtype=np.uint32).tobytes(),
                np.array([ts, tu], dtype=np.int32).tobytes(),
                np.array([1, width, height, 16], dtype=np.int16).tobytes(),
                np.array([0, 0, 0, 0], dtype=np.uint16).tobytes(),
                np.array([30000, 0], dtype=np.uint32).tobytes(),
                bytes(64)])
            data = bytearray(frame.tobytes())
            data[:len(header)] = header
            f.write(bytes(data))


def frameTimes(handle, frames):
    """ The times of some frames of an input. """

    return [handle.currentFrameTime(frame_no=i, dt_obj=True) for i in frames]


def test_vid_timestamps_corrected(tmp_path):

    # The same vid file read without and with the correction
    path = str(tmp_path/'test.vid')
    writeVid(path)
    plain = InputTypeUWOVid(path, makeConfig(0.0), chunk_frames=16)
    corrected = InputTypeUWOVid(path, makeConfig(OFFSET), chunk_frames=16)

    # The beginning and every frame are later by the correction
    assert (corrected.beginning_datetime - plain.beginning_datetime).total_seconds() == pytest.approx(OFFSET)
    for a, b in zip(frameTimes(plain, [0, 5, 20]), frameTimes(corrected, [0, 5, 20])):
        assert (b - a).total_seconds() == pytest.approx(OFFSET, abs=1e-6)

    # The names of the frames (which the results refer to) follow the correction
    assert plain.name(beginning=True) != corrected.name(beginning=True)


def test_image_directory_corrected(tmp_path):

    # A directory of images timed by its name
    d = tmp_path/T0.strftime('%Y%m%d_%H%M%S.%f')
    d.mkdir()
    for i in range(4):
        cv2.imwrite(str(d/'frame_{:03d}.png'.format(i)), np.zeros((32, 48), dtype=np.uint8))

    plain = InputTypeImages(str(d), makeConfig(0.0), fps=25.0)
    corrected = InputTypeImages(str(d), makeConfig(OFFSET), fps=25.0)

    assert plain.beginning_datetime == T0
    assert (corrected.beginning_datetime - T0).total_seconds() == pytest.approx(OFFSET)
    for a, b in zip(frameTimes(plain, [0, 3]), frameTimes(corrected, [0, 3])):
        assert (b - a).total_seconds() == pytest.approx(OFFSET)


def test_fits_frames_corrected_once(tmp_path):

    from astropy.io import fits

    # A directory of FITS frames with the time of every frame in its header, the beginning taken from the first one
    d = tmp_path/'fits'
    d.mkdir()
    for i in range(4):
        hdu = fits.PrimaryHDU(np.zeros((32, 48), dtype=np.uint16))
        hdu.header['DATE-OBS'] = (T0 + datetime.timedelta(seconds=0.04*i)).strftime('%Y-%m-%dT%H:%M:%S.%f')
        hdu.header['EXPOSURE'] = 0.04
        hdu.writeto(str(d/'frame_{:03d}.fit'.format(i)))

    corrected = InputTypeImages(str(d), makeConfig(OFFSET))

    # The beginning and the frames are corrected once
    assert (corrected.beginning_datetime - T0).total_seconds() == pytest.approx(OFFSET)
    for i, t in zip([0, 3], frameTimes(corrected, [0, 3])):
        assert (t - T0).total_seconds() == pytest.approx(0.04*i + OFFSET)


def test_video_file_corrected(tmp_path):

    import json
    import subprocess
    import sys

    # A video file timed by its name
    path = str(tmp_path/(T0.strftime('%Y%m%d_%H%M%S.%f') + '.avi'))
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 25.0, (64, 48), False)
    for _ in range(10):
        writer.write(np.zeros((48, 64), dtype=np.uint8))
    writer.release()

    # The video is read in another process: the decoders keep threads running after the file is closed, and a
    #   process forked from this one afterwards (e.g. by the tests of the monitor) can hang on them
    script = """
import json, sys
import RMS.ConfigReader as cr
from RMS.Formats.FrameInterface import InputTypeVideo
out = {}
for offset in (0.0, %r):
    config = cr.Config()
    config.frame_time_offset = offset
    video = InputTypeVideo(sys.argv[1], config)
    out[str(offset)] = [video.beginning_datetime.isoformat()] + \\
        [video.currentFrameTime(frame_no=i, dt_obj=True).isoformat() for i in (0, 7)]
print(json.dumps(out))
""" % OFFSET
    result = subprocess.run([sys.executable, '-c', script, path], capture_output=True, text=True, timeout=300,
                            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    times = json.loads(result.stdout.strip().splitlines()[-1])
    plain = [datetime.datetime.fromisoformat(t) for t in times['0.0']]
    corrected = [datetime.datetime.fromisoformat(t) for t in times[str(OFFSET)]]

    assert plain[0] == T0
    assert (corrected[0] - T0).total_seconds() == pytest.approx(OFFSET)
    for a, b in zip(plain[1:], corrected[1:]):
        assert (b - a).total_seconds() == pytest.approx(OFFSET)


def test_dfn_image_corrected(tmp_path):

    # A DFN image timed by its name
    path = str(tmp_path/'DFN_2026-10-20_010203_test.jpg')
    cv2.imwrite(path, np.zeros((32, 48, 3), dtype=np.uint8))

    plain = InputTypeDFN(path, makeConfig(0.0))
    corrected = InputTypeDFN(path, makeConfig(OFFSET))

    assert (corrected.beginning_datetime - plain.beginning_datetime).total_seconds() == pytest.approx(OFFSET)


def test_offset_of_unix_times():

    # Corrections which carry over to the next second or back to the previous one stay exact
    handle = InputTypeImages.__new__(InputTypeImages)
    handle.config = makeConfig(0.25)
    assert handle.offsetUnixTime(100, 900000) == (101, 150000)
    handle.config = makeConfig(-0.25)
    assert handle.offsetUnixTime(100, 100000) == (99, 850000)

    # Without a correction, or without a config, the times are kept
    handle.config = makeConfig(0.0)
    assert handle.offsetUnixTime(100, 123456) == (100, 123456)
    assert handle.offsetDatetime(T0) == T0
    del handle.config
    assert handle.frameTimeOffset() == 0.0


def test_frame_block_corrected(tmp_path):

    from RMS.Formats.FrameBlock import FrameBlockWriter
    from RMS.Formats.FrameInterface import InputTypeFrameBlock

    # A frame block of 10 frames at 10 fps, timed by the starts of their exposures
    path = str(tmp_path/'test.fblk')
    t0_ns = int(round((T0 - datetime.datetime(1970, 1, 1)).total_seconds()*1e9))
    with FrameBlockWriter(path, 64, 48, 10) as writer:
        for i in range(10):
            writer.write(np.zeros((48, 64), dtype=np.uint16), t0_ns + i*100000000, exposure_us=95000)

    plain = InputTypeFrameBlock(path, makeConfig(0.0))
    corrected = InputTypeFrameBlock(path, makeConfig(OFFSET))

    assert (corrected.beginning_datetime - plain.beginning_datetime).total_seconds() == pytest.approx(OFFSET, abs=1e-6)
    for a, b in zip(frameTimes(plain, [0, 9]), frameTimes(corrected, [0, 9])):
        assert (b - a).total_seconds() == pytest.approx(OFFSET, abs=1e-6)


def test_fits_directory_corrected_once(tmp_path):

    from astropy.io import fits

    # A directory of .fits frames (one frame per file, DATE-OBS of every frame), the beginning from the first frame
    d = tmp_path/'fitsdir'
    d.mkdir()
    for i in range(4):
        hdu = fits.PrimaryHDU(np.zeros((32, 48), dtype=np.uint16))
        hdu.header['DATE-OBS'] = (T0 + datetime.timedelta(seconds=0.1*i)).isoformat()
        hdu.writeto(str(d/'frame_{:03d}.fits'.format(i)))

    corrected = InputTypeImages(str(d), makeConfig(OFFSET))

    assert (corrected.beginning_datetime - T0).total_seconds() == pytest.approx(OFFSET)
    for i, t in zip([0, 3], frameTimes(corrected, [0, 3])):
        assert (t - T0).total_seconds() == pytest.approx(0.1*i + OFFSET)
