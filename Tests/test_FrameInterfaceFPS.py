"""
Tests for estimating the frame rate from the frame times in RMS.Formats.FrameInterface, on synthetic UWO .vid
files and directories of FITS frames.
"""

import datetime

import numpy as np
import pytest
from astropy.io import fits

import RMS.ConfigReader as cr
from RMS.Formats.FrameInterface import estimateFPS, InputTypeImages, InputTypeUWOVid


# Beginning Unix time of the synthetic videos
BEG_UNIX_TIME = 1735120800


def _writeVid(file_path, frame_times, wid=64, ht=32):
    """ Write a synthetic .vid file with blank frames taken at the given times.

    Arguments:
        file_path: [str] Path to the .vid file.
        frame_times: [list] Times of frames in seconds since BEG_UNIX_TIME.

    Keyword arguments:
        wid: [int] Image width in pixels.
        ht: [int] Image height in pixels.
    """

    # Every frame has seqlen bytes, starting with the header
    seqlen = 2*wid*ht

    with open(file_path, 'wb') as f:
        for i, t in enumerate(frame_times):

            # Split the time into seconds and microseconds
            ts = BEG_UNIX_TIME + int(t)
            tu = int(round((t - int(t))*1e6))

            header = b''.join([
                np.array([0, seqlen, 108, 0, i], dtype=np.uint32).tobytes(),
                np.array([ts, tu], dtype=np.int32).tobytes(),
                np.array([1, wid, ht, 16], dtype=np.int16).tobytes(),
                np.array([0, 0, 0, 0], dtype=np.uint16).tobytes(),
                np.array([0, 0], dtype=np.uint32).tobytes(),
                np.zeros(64, dtype=np.uint8).tobytes(),
            ])

            f.write(header + bytes(seqlen - len(header)))


@pytest.fixture
def config():

    config = cr.Config()
    config.fps = 25.0

    return config


def test_estimate_fps_counts_intervals():

    # 10 frames at 32 fps span 9 frame intervals
    assert estimateFPS([k/32.0 for k in range(10)]) == pytest.approx(32.0)


def test_estimate_fps_ignores_time_jump():

    # A jump of 1 s in the time stamps after the third frame (seen at the beginning of a .vid file)
    unix_times = [k/32.0 for k in range(128)]
    unix_times = unix_times[:3] + [t + 1.0 for t in unix_times[3:]]
    assert estimateFPS(unix_times) == pytest.approx(32.0)


@pytest.mark.parametrize('true_fps, step', [(30.0, 0.001), (32.0, 0.01), (25.0, 0.001)])
def test_estimate_fps_rounded_time_stamps(true_fps, step):

    # Time stamps rounded to more than the precision of the frame interval
    unix_times = [round(k/true_fps/step)*step for k in range(256)]
    assert estimateFPS(unix_times) == pytest.approx(true_fps, rel=2e-3)


def test_estimate_fps_repeated_time_stamps():

    # Every time stamp repeated for two frames: the rate from the whole span
    unix_times = [(k//2)*2/25.0 for k in range(100)]
    assert estimateFPS(unix_times) == pytest.approx(25.0, rel=0.05)


@pytest.mark.parametrize('unix_times', [[], [100.0], [100.0, 100.0]])
def test_estimate_fps_undefined(unix_times):
    assert estimateFPS(unix_times) is None


@pytest.mark.parametrize('true_fps, n_frames', [(32.0, 10), (80.0, 200)])
def test_vid_fps_estimate(tmp_path, config, true_fps, n_frames):

    vid_path = str(tmp_path/'ev_20251225_100000A_01T.vid')
    _writeVid(vid_path, [k/true_fps for k in range(n_frames)])

    handle = InputTypeUWOVid(vid_path, config)
    handle.vid_file.close()

    assert handle.fps == pytest.approx(true_fps, rel=1e-4)


def test_vid_single_frame_uses_config_fps(tmp_path, config):

    vid_path = str(tmp_path/'ev_20251225_100000A_01T.vid')
    _writeVid(vid_path, [0.0])

    handle = InputTypeUWOVid(vid_path, config)
    handle.vid_file.close()

    assert handle.fps == config.fps



def _writeFits(dir_path, frame_times, wid=64, ht=32):
    """ Write a directory of synthetic FITS frames (one frame per file) taken at the given times, with the time of
        every frame in DATE-OBS.

    Arguments:
        dir_path: [pathlib.Path] Directory of the frames (created).
        frame_times: [list] Times of frames in seconds since BEG_UNIX_TIME.

    Keyword arguments:
        wid: [int] Image width in pixels.
        ht: [int] Image height in pixels.
    """

    dir_path.mkdir()
    begin = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=BEG_UNIX_TIME)
    for i, t in enumerate(frame_times):
        hdu = fits.PrimaryHDU(np.zeros((ht, wid), dtype=np.uint16))
        hdu.header['DATE-OBS'] = (begin + datetime.timedelta(seconds=t)).isoformat() + '+00:00'
        hdu.writeto(str(dir_path/'XX0001_{:04d}.fits'.format(i)))


def test_fits_fps_estimate(tmp_path, config):
    """ The frame rate of a directory of FITS frames is measured from their times, not taken from the config
        (25 FPS here).
    """

    true_fps = 14.9993
    _writeFits(tmp_path/'frames', [k/true_fps for k in range(40)])

    handle = InputTypeImages(str(tmp_path/'frames'), config)

    assert handle.fps == pytest.approx(true_fps, rel=1e-4)


def test_fits_single_frame_uses_config_fps(tmp_path, config):

    _writeFits(tmp_path/'frames', [0.0])

    handle = InputTypeImages(str(tmp_path/'frames'), config)

    assert handle.fps == config.fps
