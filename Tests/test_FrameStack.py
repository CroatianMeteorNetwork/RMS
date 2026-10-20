"""Tests for the burst stacking of saved frames (RMS.FrameStack) and its 16-bit save (RMS.RawFrameSave)."""

import os
import time

import cv2
import numpy as np
import pytest

from RMS.FrameStack import FrameStacker, stackWindow, stackToImage, STACK_SCALE
from RMS.FrameMetadata import readImageMeta
from RMS.RawFrameSave import RawFrameSaver

FPS, DT = 25.0, 5.0
SHAPE = (6, 8, 3)


def _frame(i, shape=SHAPE):
    """ A frame whose every pixel is its index (mod 200), so sums identify the frames. """
    return np.full(shape, i % 200, np.uint8)


def _run(stacker, frames):
    """ Feed (index, time) pairs; return the finished stacks. """
    out = []
    for i, t in frames:
        s = stacker.add(_frame(i), t, meta=("meta", i))
        if s is not None:
            out.append(s)
    return out


def testWindow():
    assert stackWindow(8) == (-4, 3)
    assert stackWindow(7) == (-3, 3)
    assert stackWindow(2) == (-1, 0)


def testEightFramesCentredOnEachTick():
    rng = np.random.default_rng(1)
    frames = [(i, 1000.0 + i/FPS + rng.uniform(-0.004, 0.004)) for i in range(int(16*FPS))]
    frames = frames[20:]                                      # start between ticks

    stacks = _run(FrameStacker(8, DT, FPS), frames)

    assert len(stacks) == 3                                   # ticks at 1005, 1010, 1015 s
    t = dict(frames)
    for s, tick_frame in zip(stacks, (125, 250, 375)):
        idx = range(tick_frame - 4, tick_frame + 4)
        assert s.count == 8
        assert np.all(s.frame_sum == sum(i % 200 for i in idx))
        assert s.timestamp == t[tick_frame]                   # named after the tick frame
        assert s.meta == ("meta", tick_frame)                 # with its exposure record
        assert (s.t_first, s.t_last) == (t[tick_frame - 4], t[tick_frame + 3])


def testDroppedFrames():
    frames = [(i, 1000.0 + i/FPS) for i in range(20, 200) if i not in (125, 128)]   # tick frame and last frame dropped

    s, = _run(FrameStacker(8, DT, FPS), frames)

    assert s.count == 6
    assert np.all(s.frame_sum == sum(i % 200 for i in (121, 122, 123, 124, 126, 127)))
    assert s.timestamp == pytest.approx(1005.0)               # no tick frame: the tick time
    assert s.meta == ("meta", 121)                            # and the first frame's record


def testShapeChangeDiscardsPartialStack():
    stacker = FrameStacker(8, DT, FPS)
    for i in range(121, 124):
        assert stacker.add(_frame(i), 1000.0 + i/FPS) is None
    out = [stacker.add(_frame(i, (6, 8)), 1000.0 + i/FPS) for i in range(124, 129)]

    s = [x for x in out if x is not None][0]
    assert s.frame_sum.shape == (6, 8)
    assert s.count == 5 and stacker.discarded == 1           # frames 124-128


def testStartsStackOnlyOnFirstFrame():
    stacker = FrameStacker(8, DT, FPS)
    starts = []
    for i in range(115, 135):
        t = 1000.0 + i/FPS
        starts.append(stacker.startsStack(t))
        stacker.add(_frame(i), t)
    assert [115 + k for k, x in enumerate(starts) if x] == [121]


def testStuckTimestampsCannotOverflow():
    stacker = FrameStacker(8, DT, FPS)
    for _ in range(300):
        stacker.add(np.full(SHAPE, 255, np.uint8), 1005.0)
    s = stacker.add(np.full(SHAPE, 255, np.uint8), 1006.0)

    assert s.count == 10 and s.frame_sum.max() == 2550


def testStackToImageIsExactFixedPoint():
    rng = np.random.default_rng(2)
    frames = rng.integers(0, 256, (8, 20, 30, 3)).astype(np.uint8)
    frame_sum = frames.sum(axis=0).astype(np.uint16)

    img = stackToImage(frame_sum, 8)

    assert img.dtype == np.uint16
    assert np.array_equal(img, frame_sum*32)                  # 256*mean, exact for 8 frames
    assert np.array_equal(stackToImage(np.full((2, 2), 7*255, np.uint16), 7), np.full((2, 2), 255*256))


class _Config(object):
    stationID = "XX0001"
    frame_file_type = "jpg"
    png_compression = 3
    jpgs_quality = 90
    uyvy_pixelformat = False
    save_frame_metadata = False
    day_highlight_rebuild = False
    day_highlight_method = "invert"


def _saver(tmp_path, **cfg):
    saver = RawFrameSaver.__new__(RawFrameSaver)
    saver.saved_frames_dir = str(tmp_path)
    saver.config = _Config()
    for k, v in cfg.items():
        setattr(saver.config, k, v)
    saver.total_saved_frames = 0
    saver.day_of_year = time.strftime("%j", time.gmtime(1.7e9))
    return saver


def _saved(tmp_path):
    files = [os.path.join(r, f) for r, _, fs in os.walk(str(tmp_path)) for f in fs]
    assert len(files) == 1
    return files[0]


def testSaverWritesSixteenBitPngWithStackRecord(tmp_path):
    rng = np.random.default_rng(3)
    frame_sum = rng.integers(0, 8*256, (12, 16, 3)).astype(np.uint16)

    _saver(tmp_path).saveFramesToDisk([(frame_sum, 1.7e9, None, np.array([8, 1.7e9 - 0.16, 1.7e9 + 0.12]))], True)

    path = _saved(tmp_path)
    assert path.endswith("_d.png")                             # PNG even though frame_file_type is jpg
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    assert img.dtype == np.uint16 and np.array_equal(img, frame_sum*32)
    meta = readImageMeta(path)
    assert meta["stack_n"] == 8 and meta["stack_scale"] == STACK_SCALE
    assert meta["stack_t1"] - meta["stack_t0"] == pytest.approx(0.28)


def testSaverSingleFramesUnchanged(tmp_path):
    frame = np.full((12, 16), 77, np.uint8)

    _saver(tmp_path).saveFramesToDisk([(frame, 1.7e9, None)], False)

    path = _saved(tmp_path)
    assert path.endswith("_n.jpg")
    assert cv2.imread(path, cv2.IMREAD_UNCHANGED).dtype == np.uint8


def testSaverRebuildsSixteenBitHighlightsThroughEightBit(tmp_path):
    from Tests.test_HighlightRebuild import _camera, _scene, CG
    from RMS.HighlightRebuild import rebuildGreen
    img8 = _camera(_scene(), CG)
    frame_sum = img8.astype(np.uint16)*8

    _saver(tmp_path, day_highlight_rebuild=True).saveFramesToDisk(
        [(frame_sum, 1.7e9, None, np.array([8, 1.7e9, 1.7e9]))], True)

    out = cv2.imread(_saved(tmp_path), cv2.IMREAD_UNCHANGED)
    ref8, info = rebuildGreen(img8)                            # no exposure record: the chroma method
    changed = np.any(ref8 != img8, axis=-1)
    assert info is not None and changed.sum() > 100
    assert np.array_equal(out[changed], ref8[changed].astype(np.uint16)*256)
    assert np.array_equal(out[~changed], frame_sum[~changed]*32)


def testSaverProcessOnSharedStackBuffers(tmp_path):
    """ The capture -> saver path over shared memory, laid out as BufferedCapture lays it out for stacks. """
    import ctypes
    from multiprocessing import Array, Value
    from RMS.ConfigReader import Config
    from RMS.FrameMetadata import NFIELDS, fillRow

    n, shape = 5, (12, 16, 3)
    config = Config()
    config.stationID, config.frame_file_type, config.save_frame_metadata = "XX0001", "jpg", False
    bases = [Array(ctypes.c_uint16, n*int(np.prod(shape))) for _ in range(2)]
    stamps = [Array(ctypes.c_double, n) for _ in range(2)]
    metas = [Array(ctypes.c_double, n*NFIELDS) for _ in range(2)]
    stacks = [Array(ctypes.c_double, n*3) for _ in range(2)]
    starts = [Value('d', 0.0, lock=False) for _ in range(2)]
    saver = RawFrameSaver(str(tmp_path), bases[0], starts[0], bases[1], starts[1], stamps[0], stamps[1], False, config,
                          (n,) + shape, metaArray1=metas[0], metaArray2=metas[1], stackArray1=stacks[0], stackArray2=stacks[1])
    saver.start()
    try:
        frames = np.ctypeslib.as_array(bases[0].get_obj()).reshape((n,) + shape)
        ts = np.ctypeslib.as_array(stamps[0].get_obj())
        meta = np.ctypeslib.as_array(metas[0].get_obj()).reshape(n, NFIELDS)
        stack = np.ctypeslib.as_array(stacks[0].get_obj()).reshape(n, 3)
        for k in range(n):
            frames[k] = 8*(10 + k)
            ts[k] = 1.7e9 + 5*k
            fillRow(meta[k], None)
            stack[k] = (8, ts[k] - 0.16, ts[k] + 0.12)
        starts[0].value = ts[0]
        t0 = time.time()
        while starts[0].value != 0 and time.time() - t0 < 30:
            time.sleep(0.05)
        assert starts[0].value == 0
    finally:
        saver.stop()
        saver.join(10)

    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(str(tmp_path)) for f in fs)
    assert len(files) == n and all(f.endswith("_n.png") for f in files)
    for k, f in enumerate(files):
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        assert img.dtype == np.uint16 and np.all(img == (10 + k)*256)
        assert readImageMeta(f)["stack_n"] == 8
