""" Tests of the shape of moving objects measured on raw frames (RMS.Routines.TrackShape): a point source gives a
    spread of about 1, an extended structure a large one, the motion direction is taken into account, and the raw
    inputs read through the frame interface (.vid files, frame blocks, UWO PNG frame dumps) give the same frames,
    times and measurement as the frames written. """

from __future__ import print_function, division, absolute_import

import datetime
import os

import numpy as np
import pytest

import RMS.Formats.FrameBlock as fb
import RMS.Routines.TrackShape as ts


# Frames of the synthetic recordings: size, frame rate, beginning, the background level and its noise
WIDTH, HEIGHT = 128, 96
FPS = 10.0
T_BEGIN = 1793000000.0
SKY, NOISE = 100.0, 2.0

# The track: from (30, 40) to (90, 60) px between 5 and 9 s after the beginning (12 s of frames in all)
N_FRAMES = 120
TRACK_T = (T_BEGIN + 5.0, T_BEGIN + 9.0)
TRACK_XY = ((30.0, 40.0), (90.0, 60.0))


def trackPoints(n=20):
    """ Times and positions of the measurements of the track. """

    t = np.linspace(TRACK_T[0], TRACK_T[1], n)
    f = (t - t[0])/(t[-1] - t[0])

    return t, TRACK_XY[0][0] + f*(TRACK_XY[1][0] - TRACK_XY[0][0]), \
        TRACK_XY[0][1] + f*(TRACK_XY[1][1] - TRACK_XY[0][1])


def syntheticFrames(sigma_cross, sigma_along, amplitude=200.0, seed=0):
    """ Frames of an object moving along the track: an elliptical Gaussian with the given sigmas across and
        along the motion, on a flat noisy background (uint16), and the times of the frames.
    """

    rng = np.random.default_rng(seed)
    times = T_BEGIN + np.arange(N_FRAMES)/FPS
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
    (x0, y0), (x1, y1) = TRACK_XY
    angle = np.arctan2(y1 - y0, x1 - x0)

    frames = []
    for t in times:

        # The object only while it is on the track
        img = SKY + rng.normal(0, NOISE, (HEIGHT, WIDTH))
        if TRACK_T[0] <= t <= TRACK_T[1]:
            f = (t - TRACK_T[0])/(TRACK_T[1] - TRACK_T[0])
            cx, cy = x0 + f*(x1 - x0), y0 + f*(y1 - y0)
            u = (xx - cx)*np.cos(angle) + (yy - cy)*np.sin(angle)
            v = -(xx - cx)*np.sin(angle) + (yy - cy)*np.cos(angle)
            img += amplitude*np.exp(-u**2/(2*sigma_along**2) - v**2/(2*sigma_cross**2))
        frames.append(np.clip(np.round(img), 0, 4095).astype(np.uint16))

    return frames, times


class MemorySource(object):
    """ A raw frame source of frames held in memory, with the methods which the measurement uses. """

    def __init__(self, frames, times):
        """ The frames (2D arrays) and their times (Unix s). """
        self.frames, self.times = frames, np.asarray(times)
        self.height, self.width = frames[0].shape
        self.n_frames = len(frames)

    def frameTime(self, i):
        """ Time of frame i. """
        return float(self.times[i])

    def framesBetween(self, t0, t1):
        """ The frames whose times are between t0 and t1, and their times. """
        sel = np.where((self.times >= t0) & (self.times <= t1))[0]
        return sel, self.times[sel]

    def frame(self, i):
        """ Frame i as float32. """
        return self.frames[i].astype(np.float32)

    def close(self):
        """ Nothing to release. """
        return None


def writeVid(path, frames, times, seconds_off=()):
    """ Write frames to a .vid file: every frame is a record of the size of the image, with the header in the
        first bytes of the first row. The seconds field of the frames in seconds_off is written one second low. """

    header = np.dtype([('magic', '<u4'), ('seqlen', '<u4'), ('headlen', '<u4'), ('flags', '<u4'), ('seq', '<u4'),
                       ('ts', '<i4'), ('tu', '<i4'), ('station_id', '<i2'), ('wid', '<i2'), ('ht', '<i2'),
                       ('depth', '<i2'), ('hx', '<u2'), ('hy', '<u2'), ('str_num', '<u2'), ('reserved0', '<u2'),
                       ('exposure', '<u4'), ('reserved2', '<u4'), ('text', 'S64')])
    with open(path, 'wb') as f:
        for k, (frame, t) in enumerate(zip(frames, times)):
            record = frame.astype('<u2').copy()
            head = np.zeros(1, dtype=header)
            head['seqlen'], head['headlen'], head['seq'] = frame.nbytes, header.itemsize, k
            head['ts'], head['tu'] = int(t) - (1 if k in seconds_off else 0), int(round((t - int(t))*1e6))
            head['wid'], head['ht'], head['depth'], head['exposure'] = WIDTH, HEIGHT, 16, 100
            buf = record.view(np.uint8).reshape(-1)
            buf[:header.itemsize] = np.frombuffer(head.tobytes(), dtype=np.uint8)
            f.write(buf.tobytes())


def writeBlocks(dir_path, frames, times, frames_per_block=50):
    """ Write frames to frame block files of frames_per_block frames each, named by the time of their first
        frame. The frames are timed at the start of the exposure, so they are written half an exposure early
        (the sources time the frames at the middle of the exposure by default). """

    exposure_us = 20000
    paths = []
    for first in range(0, len(frames), frames_per_block):
        stamp = datetime.datetime.fromtimestamp(times[first], datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        path = os.path.join(dir_path, 'XX0001_{:s}{:s}'.format(stamp, fb.EXTENSION))
        with fb.FrameBlockWriter(path, WIDTH, HEIGHT, frames_per_block, packing='mono12p') as w:
            for frame, t in zip(frames[first:first + frames_per_block], times[first:first + frames_per_block]):
                w.write(frame, int(round((t - exposure_us/2e6)*1e9)), exposure_us=exposure_us)
        paths.append(path)

    return paths


def test_point_source_and_extended_structure():
    """ A moving point source has a spread of about 1 (its width across the motion is the PSF), an extended
        structure a spread of several PSF sigmas. """

    t, x, y = trackPoints()

    # A point source of PSF sigma 1 px
    frames, times = syntheticFrames(1.0, 1.0)
    res = ts.measureTrackShape(MemorySource(frames, times), t, x, y, psf_sigma=1.0)
    assert res['ok'] and (res['flag'] == 'ok')
    assert 0.9 < res['spread'] < 1.3
    assert 0.9 < res['sigma_along'] < 1.4
    assert res['snr'] > 20
    assert res['n_frames'] == 41

    # A structure 5 px wide
    frames, times = syntheticFrames(5.0, 5.0, amplitude=60.0)
    res = ts.measureTrackShape(MemorySource(frames, times), t, x, y, psf_sigma=1.0)
    assert res['ok'] and (res['spread'] > 4.0)


def test_width_across_the_motion():
    """ The spread is the width across the motion: an object elongated along its motion (diagonal in the image)
        is narrow across it, and one elongated across its motion is wide. """

    t, x, y = trackPoints()

    # Elongated along the motion
    frames, times = syntheticFrames(1.0, 4.0)
    res = ts.measureTrackShape(MemorySource(frames, times), t, x, y, psf_sigma=1.0)
    assert res['spread'] < 1.4
    assert res['sigma_along'] > 3.5

    # Elongated across the motion
    frames, times = syntheticFrames(4.0, 1.0)
    res = ts.measureTrackShape(MemorySource(frames, times), t, x, y, psf_sigma=1.0)
    assert res['spread'] > 3.5
    assert res['sigma_along'] < 1.5


def writeDumps(dir_path, frames, times):
    """ Write frames to a directory of UWO PNG frame dumps (dump_NNNNN.png, 16-bit): the values are stored
        byteswapped, and the first row is the header with the magick number, the timestamp (seconds and
        microseconds in 32-bit words 3 and 4) and the frame counter (word 7). """

    import cv2
    from RMS.Formats.FrameInterface import UWO_MAGICK_EMCCD

    os.makedirs(dir_path, exist_ok=True)
    for k, (frame, t) in enumerate(zip(frames, times)):

        # The timestamp as seconds and microseconds
        ts = int(np.floor(t))
        tu = int(round((t - ts)*1e6))

        # The header words of the first row (as read, then stored byteswapped with the magick number first)
        row = np.zeros(WIDTH, dtype=np.uint16)
        for word, value in ((3, ts), (4, tu), (7, 500000 + k)):
            row[2*word] = value & 0xFFFF
            row[2*word + 1] = (value >> 16) & 0xFFFF
        stored = frame.astype(np.uint16).byteswap()
        stored[0] = row.byteswap()
        stored[0, :2] = np.frombuffer(np.uint32(UWO_MAGICK_EMCCD).tobytes(), dtype=np.uint16)
        cv2.imwrite(os.path.join(dir_path, 'dump_{:05d}.png'.format(k)), stored)


def test_vid_input(tmp_path):
    """ A .vid file is read through the frame interface: the frames as written (the header row is set to 0), their
        times from the headers, frame_time_offset of the config added, and the same measurement as the frames in
        memory. """

    from RMS.Formats.FrameInterface import InputTypeUWOVid

    t, x, y = trackPoints()
    frames, times = syntheticFrames(1.0, 1.0)
    reference = ts.measureTrackShape(MemorySource(frames, times), t, x, y, psf_sigma=1.0)

    # The frames and their times
    vid_path = str(tmp_path/'test.vid')
    writeVid(vid_path, frames, times)
    source = ts.openRawSource(vid_path, TRACK_T[0], TRACK_T[1])
    assert isinstance(source, ts.RawFrames) and isinstance(source.img_handles[0], InputTypeUWOVid)
    assert source.n_frames == N_FRAMES
    assert source.names == ['test.vid']
    assert np.array_equal(source.frame(7)[1:], frames[7][1:].astype(np.float32))
    idx, tf = source.framesBetween(times[10] - 1e-3, times[20] + 1e-3)
    assert list(idx) == list(range(10, 21))
    assert np.allclose(tf, times[10:21], rtol=0, atol=1e-5)

    # The measurement
    res = ts.measureTrackShape(source, t, x, y, psf_sigma=1.0)
    source.close()
    assert abs(res['spread'] - reference['spread']) < 0.02
    assert res['n_stack'] == reference['n_stack']

    # The correction of the times of the config
    source = ts.openRawSource(vid_path, TRACK_T[0], TRACK_T[1], frame_time_offset=0.084)
    assert np.allclose([source.frameTime(i) for i in (0, 50, 119)], times[[0, 50, 119]] + 0.084, rtol=0,
                       atol=1e-5)
    source.close()


def test_frame_block_input(tmp_path):
    """ Frame blocks are read through the frame interface: the frames of all blocks around the interval numbered
        continuously, timed at the middle of the exposure, and the same measurement as the frames in memory (the
        12-bit values are read spanning 16 bits, which doesn't change the shape). """

    from RMS.Formats.FrameInterface import InputTypeFrameBlock

    t, x, y = trackPoints()
    frames, times = syntheticFrames(1.0, 1.0)
    reference = ts.measureTrackShape(MemorySource(frames, times), t, x, y, psf_sigma=1.0)

    block_dir = tmp_path/'blocks'
    block_dir.mkdir()
    paths = writeBlocks(str(block_dir), frames, times)
    source = ts.openRawSource(str(block_dir), TRACK_T[0] - 4, TRACK_T[1] + 4)
    assert all(isinstance(h, InputTypeFrameBlock) for h in source.img_handles)
    assert source.n_frames == N_FRAMES
    assert np.allclose([source.frameTime(i) for i in range(N_FRAMES)], times, rtol=0, atol=1e-6)
    assert np.array_equal(source.frame(77), 16*frames[77].astype(np.float32))
    res = ts.measureTrackShape(source, t, x, y, psf_sigma=1.0)
    source.close()
    assert abs(res['spread'] - reference['spread']) < 1e-3
    assert res['n_stack'] == reference['n_stack']

    # Only the blocks which can hold frames of the interval are opened
    assert ts.rawFilesAround(str(block_dir), times[60], times[70]) == [paths[1]]
    assert ts.rawFilesAround(str(block_dir), times[45], times[55]) == paths[:2]
    assert ts.rawFileStart(paths[0]) == pytest.approx(times[0], abs=1e-5)


def test_vid_files_of_a_directory(tmp_path):
    """ The .vid files of a directory named by the time of their beginning (dump_<Unix time in hex>_<camera>.vid)
        which hold the interval are joined into one sequence of frames. """

    t, x, y = trackPoints()
    frames, times = syntheticFrames(1.0, 1.0)
    reference = ts.measureTrackShape(MemorySource(frames, times), t, x, y, psf_sigma=1.0)

    # Two files of 60 frames, and one far later which is not opened
    vid_dir = tmp_path/'vids'
    vid_dir.mkdir()
    for first in (0, 60):
        writeVid(str(vid_dir/'dump_{:08X}_99X.vid'.format(int(times[first]))), frames[first:first + 60],
                 times[first:first + 60])
    writeVid(str(vid_dir/'dump_{:08X}_99X.vid'.format(int(times[0]) + 3600)), frames[:10], times[:10] + 3600)

    source = ts.openRawSource(str(vid_dir), TRACK_T[0] - 4, TRACK_T[1] + 4)
    assert len(source.img_handles) == 2 and source.n_frames == N_FRAMES
    assert np.allclose([source.frameTime(i) for i in range(N_FRAMES)], times, rtol=0, atol=1e-5)
    res = ts.measureTrackShape(source, t, x, y, psf_sigma=1.0)
    source.close()
    assert abs(res['spread'] - reference['spread']) < 0.02
    assert res['n_stack'] == reference['n_stack']


def test_png_frame_dumps(tmp_path):
    """ A directory of UWO PNG frame dumps is read through the frame interface (image input): the frames as written,
        their times from the headers, and the same measurement as the frames in memory. """

    from RMS.Formats.FrameInterface import InputTypeImages

    t, x, y = trackPoints()
    frames, times = syntheticFrames(1.0, 1.0)
    reference = ts.measureTrackShape(MemorySource(frames, times), t, x, y, psf_sigma=1.0)

    dump_dir = str(tmp_path/'dumps')
    writeDumps(dump_dir, frames, times)
    source = ts.openRawSource(dump_dir, TRACK_T[0], TRACK_T[1])
    assert isinstance(source.img_handles[0], InputTypeImages) and source.img_handles[0].uwo_png_mode
    assert source.n_frames == N_FRAMES
    assert np.allclose([source.frameTime(i) for i in range(0, N_FRAMES, 7)], times[::7], rtol=0, atol=2e-6)
    assert np.array_equal(source.frame(33)[1:], frames[33][1:].astype(np.float32))
    res = ts.measureTrackShape(source, t, x, y, psf_sigma=1.0)
    source.close()
    assert abs(res['spread'] - reference['spread']) < 0.02
    assert res['n_stack'] == reference['n_stack']


def test_compressed_video_and_ff_files_refused(tmp_path):
    """ Only raw frames are measured: compressed video and FF files are refused. """

    clip = tmp_path/'clip.mkv'
    clip.write_bytes(b'')
    with pytest.raises(ValueError, match='compressed video'):
        ts.openRawSource(str(clip), T_BEGIN, T_BEGIN + 1)

    ff = tmp_path/'FF_XX0001_20261027_101258_485_0000000.fits'
    ff.write_bytes(b'')
    with pytest.raises(ValueError, match='FF file'):
        ts.openRawSource(str(ff), T_BEGIN, T_BEGIN + 1)


def test_vid_seconds_field_repaired(tmp_path):
    """ Frames of a .vid file whose seconds field is one second low (a short run within the track; the sampled
        model of the file sees a longer run as a step) are timed from the frame counter, as the processing reads
        them, so the frames stay in time order and the measurement is the same as without the error. """

    t, x, y = trackPoints()
    frames, times = syntheticFrames(1.0, 1.0)
    reference = ts.measureTrackShape(MemorySource(frames, times), t, x, y, psf_sigma=1.0)

    vid_path = str(tmp_path/'glitch.vid')
    writeVid(vid_path, frames, times, seconds_off=range(62, 66))
    source = ts.openRawSource(vid_path, TRACK_T[0], TRACK_T[1])
    assert np.allclose([source.frameTime(i) for i in range(N_FRAMES)], times, rtol=0, atol=1e-5)
    idx, _ = source.framesBetween(times[55] - 1e-3, times[75] + 1e-3)
    assert list(idx) == list(range(55, 76))
    res = ts.measureTrackShape(source, t, x, y, psf_sigma=1.0)
    source.close()
    assert abs(res['spread'] - reference['spread']) < 0.02
    assert res['n_stack'] == reference['n_stack']


def test_psf_sigma_in_raw_pixels(tmp_path):
    """ The PSF sigma is in the pixels of the raw frames: the given one as is; the matched filter's (done file) and
        the configured one times the detection binning. """

    import json
    import types

    ftp_path = str(tmp_path/'FTPdetectinfo_XX0001_20261027_101258_485479.txt')
    config = types.SimpleNamespace(mf_psf_sigma=0.7, detection_binning_factor=2)

    # Without a done file: the configured one, binned
    assert ts.psfSigmaForRawFrames(ftp_path, config) == pytest.approx(1.4)
    assert ts.psfSigmaForRawFrames(ftp_path, config, psf_sigma=0.9) == pytest.approx(0.9)
    assert ts.psfSigmaForRawFrames(ftp_path, None) is None

    # With the matched filter's measurement
    with open(str(tmp_path/'matched_filter_done.json'), 'w') as f:
        json.dump({'psf_sigma': 0.6}, f)
    assert ts.psfSigmaForRawFrames(ftp_path, config) == pytest.approx(1.2)


def test_frame_numbers_and_missing_frames():
    """ A track given in frame numbers is converted to times; a track without frames is flagged. """

    frames, times = syntheticFrames(1.0, 1.0)
    source = MemorySource(frames, times)

    # Fractional frame numbers to times
    assert np.allclose(ts.timesOfFrames(source, [10, 10.5, 30]), [times[10], times[10] + 0.05, times[30]], rtol=0,
                       atol=1e-6)

    # A track outside the frames, and a track of a single point
    res = ts.measureTrackShape(source, [times[-1] + 100, times[-1] + 104], [30, 40], [40, 45], psf_sigma=1.0)
    assert (not res['ok']) and (res['flag'] == 'no frames')
    res = ts.measureTrackShape(source, [times[50]], [30], [40], psf_sigma=1.0)
    assert (not res['ok']) and (res['flag'] == 'few points')

    # A track which leaves the frame after its first frames: too few frames averaged
    t = np.linspace(TRACK_T[0], TRACK_T[1], 10)
    res = ts.measureTrackShape(source, t, np.linspace(25, -200, 10), np.full(10, 40.0), psf_sigma=1.0)
    assert (not res['ok']) and (res['flag'] == 'few frames') and (0 < res['n_stack'] < ts.MIN_STACK_FRAMES)

    # A track along the edge of the frame
    t = np.linspace(TRACK_T[0], TRACK_T[1], 10)
    res = ts.measureTrackShape(source, t, np.linspace(30, 90, 10), np.full(10, 2.0), psf_sigma=1.0)
    assert (not res['ok']) and (res['flag'] == 'edge')


def test_detections_of_an_ftpdetectinfo(tmp_path):
    """ The detections of an FTPdetectinfo file are found in the frame blocks by their times (the time in the FF
        name plus the frame over the frame rate), with the PSF sigma of the done file of the matched filter. """

    from RMS.Formats.FFfile import constructFFName
    from RMS.Formats.FTPdetectinfo import writeFTPdetectinfo

    frames, times = syntheticFrames(1.0, 1.0)
    block_dir = tmp_path/'blocks'
    block_dir.mkdir()
    writeBlocks(str(block_dir), frames, times)

    # A detection named by the time of a frame 2 s before the track, its measurements in frames from that time,
    #   one of them to every other frame
    ref_time = datetime.datetime.fromtimestamp(times[30], datetime.timezone.utc).replace(tzinfo=None)
    ff_name = constructFFName('XX0001', ref_time)
    t, x, y = trackPoints(21)
    centroids = [[(tk - times[30])*FPS, xk, yk, 0, 0, 0, 0, 100, 5.0, 100, 10.0, 0] for tk, xk, yk in zip(t, x, y)]
    out_dir = tmp_path/'detections'
    out_dir.mkdir()
    ftp_name = 'FTPdetectinfo_XX0001_test.txt'
    writeFTPdetectinfo([[ff_name, 1, 0, 0, centroids]], str(out_dir), ftp_name, str(out_dir), 'XX0001', FPS,
                       celestial_coords_given=True)
    with open(str(out_dir/'matched_filter_done.json'), 'w') as f:
        f.write('{"psf_sigma": 1.0, "input_file": "/elsewhere/missing.fblk"}')

    # The PSF sigma of the done file, the processed file which isn't there any more, and the measurement
    ftp_path = str(out_dir/ftp_name)
    assert ts.psfSigmaFromDone(ftp_path) == 1.0
    assert ts.rawInputFromDone(ftp_path, str(block_dir)) is None
    rows = ts.measureFTPdetectinfo(ftp_path, str(block_dir), 1.0)
    assert len(rows) == 1
    assert rows[0]['ok'] and (rows[0]['n_frames'] == 41)
    assert 0.9 < rows[0]['spread'] < 1.3
    assert rows[0]['raw_file'].count('.fblk') == 3
