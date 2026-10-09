"""
Tests of reading UWO .vid files into memory (preload_video in RMS.Formats.FrameInterface): the frames, their
times and the frame chunks read from memory are the same as the ones read from the file, and files larger
than vid_preload_max_mb are read from the file.
"""

import os

import numpy as np
import pytest

import RMS.ConfigReader as cr
from RMS.Formats.FrameInterface import detectInputType, InputTypeUWOVid


# Beginning Unix time of the synthetic videos
BEG_UNIX_TIME = 1735120800


def _writeVid(file_path, n_frames, fps=32.0, wid=64, ht=32, seed=0, partial_last=False):
    """ Write a synthetic .vid file with frames of random pixel values.

    Arguments:
        file_path: [str] Path to the .vid file.
        n_frames: [int] Number of frames.

    Keyword arguments:
        fps: [float] Frame rate of the frame times.
        wid, ht: [int] Image size in pixels.
        seed: [int] Seed of the pixel values.
        partial_last: [bool] Write half of an extra frame at the end (an interrupted recording).
    """

    # Every frame has seqlen bytes: the header, followed by the pixels (the header takes the place of the first
    #   pixels of the first row)
    seqlen = 2*wid*ht
    rng = np.random.default_rng(seed)

    with open(file_path, 'wb') as f:
        for i in range(n_frames + int(partial_last)):

            # Split the time into seconds and microseconds
            t = i/fps
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

            pixels = rng.integers(0, 4000, wid*ht, dtype=np.uint16).tobytes()
            frame = header + pixels[len(header):]

            # The extra frame is cut in the middle
            if i == n_frames:
                frame = frame[:seqlen//2]

            f.write(frame)


@pytest.fixture
def config():

    config = cr.Config()
    config.fps = 25.0

    return config


def _handles(vid_path, config, **kwargs):
    """ The same file opened twice: read from the file, and read from memory. """

    from_file = InputTypeUWOVid(vid_path, config, **kwargs)
    from_memory = InputTypeUWOVid(vid_path, config, preload_video=True, **kwargs)

    return from_file, from_memory


@pytest.mark.parametrize('binning', [1, 2])
def test_preloaded_frames_are_the_frames_of_the_file(tmp_path, config, binning):
    """ The frames, the frame times and the frame chunks read from memory are the same as from the file, also
        when read out of order and with the binning of the detection.
    """

    vid_path = str(tmp_path/'ev_20251225_100000A_01T.vid')
    _writeVid(vid_path, 300)
    config.detection_binning_factor = binning

    from_file, from_memory = _handles(vid_path, config, detection=True, chunk_frames=64)
    assert from_file.vid_data is None
    assert from_memory.vid_data is not None
    assert from_memory.total_frames == from_file.total_frames == 300
    assert from_memory.fps == from_file.fps

    # Single frames, out of order (the frame time is read with the frame)
    for frame_no in (5, 299, 0, 150, 151, 17):
        for handle in (from_file, from_memory):
            handle.setFrame(frame_no)
        assert np.array_equal(from_file.loadFrame(), from_memory.loadFrame())
        assert from_file.currentFrameTime(dt_obj=True) == from_memory.currentFrameTime(dt_obj=True)

    # The times of frames which were not read yet
    for frame_no in (42, 250):
        assert from_file.currentFrameTime(frame_no=frame_no) == from_memory.currentFrameTime(frame_no=frame_no)

    # Frame chunks: the max and average images and the times of their frames
    for first_frame, n in ((0, 64), (200, 64), (256, 64), (100, -1)):
        a = from_file.loadChunk(first_frame=first_frame, read_nframes=n)
        b = from_memory.loadChunk(first_frame=first_frame, read_nframes=n)
        assert np.array_equal(a.maxpixel, b.maxpixel)
        assert np.array_equal(a.avepixel, b.avepixel)
        assert from_file.frame_chunk_unix_times == from_memory.frame_chunk_unix_times

    from_file.vid_file.close()
    from_memory.vid_file.close()


def test_partial_last_frame_is_not_read(tmp_path, config):
    """ An incomplete frame at the end of the file (an interrupted recording) is not a frame, in memory as in
        the file.
    """

    vid_path = str(tmp_path/'ev_20251225_100000A_01T.vid')
    _writeVid(vid_path, 100, partial_last=True)

    from_file, from_memory = _handles(vid_path, config)
    assert from_memory.total_frames == from_file.total_frames == 100

    # A chunk up to the end has the same frames
    a = from_file.loadChunk(first_frame=60, read_nframes=40)
    b = from_memory.loadChunk(first_frame=60, read_nframes=40)
    assert a.nframes == b.nframes == 40
    assert np.array_equal(a.maxpixel, b.maxpixel)

    from_file.vid_file.close()
    from_memory.vid_file.close()


def test_large_file_is_read_from_the_file(tmp_path, config):
    """ A file larger than vid_preload_max_mb is not loaded into memory, and is read from the file. """

    vid_path = str(tmp_path/'ev_20251225_100000A_01T.vid')
    _writeVid(vid_path, 50)
    config.vid_preload_max_mb = 0.5*os.path.getsize(vid_path)/1024/1024

    handle = InputTypeUWOVid(vid_path, config, preload_video=True)
    assert handle.vid_data is None

    handle.setFrame(10)
    assert handle.loadFrame().shape == (32, 64)

    handle.vid_file.close()


def test_detect_input_type_preloads_vid(tmp_path, config):
    """ detectInputType passes preload_video to the .vid input. """

    vid_path = str(tmp_path/'ev_20251225_100000A_01T.vid')
    _writeVid(vid_path, 20)

    handle = detectInputType(vid_path, config, preload_video=True)
    assert handle.vid_data is not None
    handle.vid_file.close()

    handle = detectInputType(vid_path, config)
    assert handle.vid_data is None
    handle.vid_file.close()


def test_config_option(tmp_path):
    """ vid_preload_max_mb is read from the config. """

    path = str(tmp_path/'.config')
    with open(path, 'w') as f:
        f.write("[System]\nstationID: XX0001\n\n[MeteorDetection]\nvid_preload_max_mb: 512\n")

    assert cr.parse(path).vid_preload_max_mb == 512
