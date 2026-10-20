"""Tests for the 10-bit frames timelapse (Utils.GenerateTimelapse: high_bit, planTimelapseCrfs)."""

import json
import logging
import os
import subprocess

import cv2
import numpy as np
import pytest

from Utils import GenerateTimelapse as gt

FFMPEG = gt.isFfmpegWorking()
needs10 = pytest.mark.skipif(not (FFMPEG and gt.x264HighBitWorks(FFMPEG)), reason="no 10-bit libx264")
W, H = 160, 96


def _frames(tmp_path, n=24, mode="d", stacked=True, seed=0):
    """ n frames of a noisy gradient sky, 16-bit 8.8 (stacked) or 8-bit, colour by day, gray at night. """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    paths = []
    for k in range(n):
        sky = 60 + 40*x/W + 20*y/H + rng.normal(0, 3 if stacked else 8, (H, W))
        img = np.stack([sky*0.9, sky, sky*1.1], -1) if mode == "d" else sky
        img = np.clip(img, 0, 255)
        img = np.round(img*256).astype(np.uint16) if stacked else np.round(img).astype(np.uint8)
        p = os.path.join(str(tmp_path), "XX0001_20261020_{:02d}{:02d}{:02d}_000_{}.png".format(12, k//12, (k%12)*5, mode))
        cv2.imwrite(p, img)
        paths.append(p)
    return paths


def _probe(mp4):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=pix_fmt,profile",
                        "-of", "json", mp4], stdout=subprocess.PIPE, check=True)
    return json.loads(r.stdout)["streams"][0]


def _y10(mp4):
    """ Decoded luma, 10-bit codes. """
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", mp4, "-vf", "format=yuv420p10le,extractplanes=y", "-f", "rawvideo",
                        "-pix_fmt", "gray10le", "-"], stdout=subprocess.PIPE, check=True)
    return np.frombuffer(r.stdout, "<u2").reshape(-1, H, W).astype(np.float32)


def testDepthConversion():
    img8 = np.arange(256, dtype=np.uint8).reshape(16, 16)
    img16 = gt.toEncoderDepth(img8, True)
    assert img16.dtype == np.uint16 and np.array_equal(img16, img8.astype(np.uint16)*256)
    assert np.array_equal(gt.toEncoderDepth(img16, False), img8)
    assert np.array_equal(gt.toEncoderDepth(np.array([[383, 384]], np.uint16), False), [[1, 2]])   # rounds


def testBlockIsNight():
    assert gt.blockIsNight(["A_20261020_120000_000_n.png"]*3 + ["A_20261020_120000_000_d.png"], True)
    assert not gt.blockIsNight(["A_20261020_120000_000_d.png"]*2, False)
    assert gt.blockIsNight(["A_20261020_120000_000.png"], False)
    assert not gt.blockIsNight(["A_20261020_120000_000.png"], True)


def testEncoderArgs():
    legacy = gt.x264Args(25, 2)
    assert legacy[legacy.index("-x264-params") + 1] == "psy-rd=0.0,0.0:aq-strength=0.5"
    assert legacy[legacy.index("-pix_fmt") + 1] == "yuv420p" and "-crf" in legacy and "25" in legacy
    day, night = gt.x264Args(16, 2, high_bit=True), gt.x264Args(22, 2, high_bit=True, night=True)
    assert day[day.index("-profile:v") + 1] == "high10" and day[day.index("-pix_fmt") + 1] == "yuv420p10le"
    assert day[day.index("-x264-params") + 1].endswith("qcomp=1:trellis=0")
    assert night[night.index("-x264-params") + 1].endswith("qcomp=1:trellis=0:8x8dct=0")


@needs10
@pytest.mark.parametrize("mode", ["d", "n"])
def testTenBitTimelapse(tmp_path, mode):
    paths = _frames(tmp_path, mode=mode)
    out = os.path.join(str(tmp_path), "out_frames_timelapse.mp4")

    video, _ = gt.generateTimelapseFromFrames(paths, str(tmp_path), out, high_bit=True, dejump_sigma=0)

    assert video == out
    info = _probe(out)
    assert info["pix_fmt"] == "yuv420p10le" and info["profile"] == "High 10 Intra"   # all-intra High 10
    data = open(out, "rb").read()
    assert (b"8x8dct=0" in data) == (mode == "n")              # x264 writes its settings into the stream
    assert (b"crf=16.0" in data) == (mode == "d") and (b"crf=22.0" in data) == (mode == "n")
    assert b"qcomp=1.00" in data and b"trellis=0" in data

    # The 16-bit frames' fractions survive: decoded luma tracks the source to well under one 8-bit code
    y = _y10(out)
    src = cv2.imread(paths[3], cv2.IMREAD_UNCHANGED).astype(np.float32)
    src_y = src if src.ndim == 2 else cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    expected = 64 + 876*(src_y/256)/255                        # ffmpeg: 8.8 value 256c -> the 10-bit Y of 8-bit c
    core = (slice(8, H - 20), slice(8, W - 8))                 # away from the time stamp
    err = cv2.GaussianBlur(y[3], (0, 0), 3)[core] - cv2.GaussianBlur(expected, (0, 0), 3)[core]
    assert np.abs(err).mean() < 1.0                            # 10-bit codes: 1/4 of an 8-bit code


@needs10
def testEightBitEncoderTakesSixteenBitFrames(tmp_path):
    paths = _frames(tmp_path, mode="d")
    out = os.path.join(str(tmp_path), "out_frames_timelapse.mp4")

    video, _ = gt.generateTimelapseFromFrames(paths, str(tmp_path), out, dejump_sigma=0)

    assert video == out and _probe(out)["pix_fmt"] == "yuv420p"


@needs10
def testTenBitTakesEightBitFrames(tmp_path):
    paths = _frames(tmp_path, mode="n", stacked=False)
    out = os.path.join(str(tmp_path), "out_frames_timelapse.mp4")

    video, _ = gt.generateTimelapseFromFrames(paths, str(tmp_path), out, high_bit=True, dejump_sigma=0)

    assert video == out and _probe(out)["pix_fmt"] == "yuv420p10le"


@needs10
def testBudgetPlan(tmp_path, caplog):
    day = _frames(tmp_path, n=40, mode="d", seed=1)
    night = _frames(tmp_path, n=40, mode="n", seed=2)
    blocks = [day, night]
    kw = dict(crf_day=16.0, crf_night=22.0, crf_max_day=20.0, crf_max_night=25.0, frame_interval=5.0)

    # No budget pressure: the targets
    assert gt.planTimelapseCrfs(blocks, FFMPEG, budget_mb=1e6, **kw) == [(16.0, False), (22.0, True)]
    assert gt.planTimelapseCrfs(blocks, FFMPEG, budget_mb=0, **kw) == [(16.0, False), (22.0, True)]

    # A budget below even the maximum CRFs: the maximums, and a warning (quality wins)
    with caplog.at_level(logging.WARNING, logger="rmslogger"):
        assert gt.planTimelapseCrfs(blocks, FFMPEG, budget_mb=1e-3, **kw) == [(20.0, False), (25.0, True)]
    assert "over budget" in caplog.text

    # In between: both raised by the same step, and the plan fits
    sizes = {}
    for b, (t, m), night in ((day, (16.0, 20.0), False), (night, (22.0, 25.0), True)):
        sizes[night] = gt.sampleBytesPerFrame(FFMPEG, b, W, H, not night, True,
                                              [gt.x264Args(t, 2, True, night), gt.x264Args(m, 2, True, night)])
    at_target = 40*(sizes[False][0] + sizes[True][0])
    at_max = 40*(sizes[False][1] + sizes[True][1])
    assert at_max < at_target
    frames_per_day = 86400/5.0
    budget_mb = np.sqrt(at_target*at_max)/1e6*frames_per_day/80
    (c_day, _), (c_night, _) = gt.planTimelapseCrfs(blocks, FFMPEG, budget_mb=budget_mb, **kw)
    assert 16 < c_day < 20 and 22 < c_night <= 25
    assert c_day - 16 == pytest.approx(min(c_night - 22, 3), abs=0.02)


@needs10
def testBlocksUseThePlan(tmp_path, monkeypatch):
    blocks = [_frames(tmp_path, n=12, mode="d")]
    monkeypatch.setattr(gt, "planTimelapseCrfs", lambda *a, **k: [(19.5, False)])

    (video, _), = gt.generateTimelapseFromFrameBlocks(blocks, str(tmp_path), high_bit=True, dejump_sigma=0)

    assert b"crf=19.5" in open(video, "rb").read()
