"""Tests for the day-frame highlight rebuild (RMS.HighlightRebuild), on a synthetic satu 255 camera."""

import numpy as np
import pytest

from RMS import HighlightRebuild as hr
from RMS.FrameMetadata import BLC_FRACTION


def _camera(truth, cG):
    """ 8-bit BGR a satu 255 camera makes of a pre-saturation linear scene (H x W x 3, B, G, R): green
        clips at cG, red and blue at full scale, then the x1.99 saturation, clip, gamma 0.5. """
    raw = np.minimum(truth, np.array([1.0, cG, 1.0], np.float32))
    out = np.clip(np.stack(hr._saturate(raw[..., 0], raw[..., 1], raw[..., 2], hr.SATU), -1), 0, 1)
    return np.round(255*np.sqrt(out)).astype(np.uint8)


def _scene(h=240, w=320):
    """ Blue sky, a grey cloud and a sun with a halo (pre-saturation linear light). """
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    sky = np.array([0.30, 0.20, 0.11], np.float32)*(0.8 + 0.2*y[..., None]/h)
    cloud = (np.hypot(y - 170, x - 70) < 40)[..., None]*np.array([0.30, 0.32, 0.30], np.float32)
    sun = 6.0*np.exp(-np.hypot(y - 90, x - 220)/14.0)[..., None]*np.array([1.0, 1.0, 0.95], np.float32)
    return sky + cloud + sun


CG = 0.55
META = {"ispdgain": 1.0, "wb_g": CG/BLC_FRACTION}


def _magenta(img):
    o = img.astype(int)
    return (np.minimum(o[..., 0], o[..., 2]) - o[..., 1] > 12) & (o.max(-1) > 150)


def testInvertRemovesMagenta():
    img = _camera(_scene(), CG)
    assert _magenta(img).sum() > 500

    out, info = hr.rebuildGreenInvert(img, META)

    assert info["method"] == "invert" and info["clipped"] > 0
    assert _magenta(out).sum() == 0


def testInvertLeavesPixelsBelowTheClipAlone():
    truth = _scene()
    img = _camera(truth, CG)

    out, _ = hr.rebuildGreenInvert(img, META)

    dim = truth[..., 1] < 0.7*CG
    assert np.array_equal(out[dim], img[dim])


def testInvertRegionsEqualTheWholeFrame(monkeypatch):
    img = _camera(_scene(), CG)
    out, info = hr.rebuildGreenInvert(img, META)

    monkeypatch.setattr(hr, "_TILE", 1024)                 # one region: the whole frame
    whole, info_whole = hr.rebuildGreenInvert(img, META)

    assert np.array_equal(out, whole)
    assert info == info_whole


def testInvertNormalGainsUnchanged():
    img = _camera(_scene(), 1.0)

    out, info = hr.rebuildGreenInvert(img, {"ispdgain": 1.0, "wb_g": 1.07})

    assert out is img and info is None


@pytest.mark.parametrize("meta", [None, {}, {"ispdgain": float("nan"), "wb_g": 0.6}])
def testInvertWithoutRecordUsesChroma(meta):
    img = _camera(_scene(), CG)

    out, info = hr.rebuildGreenInvert(img, meta)
    chroma, _ = hr.rebuildGreen(img)

    assert info["method"] == "chroma"
    assert np.array_equal(out, chroma)


def testInvertGreyScene():
    """ A neutral scene (R = B everywhere: a singular plane fit) still gets rebuilt. """
    truth = np.repeat(_scene()[..., 1:2], 3, axis=-1)
    img = _camera(truth, CG)

    out, info = hr.rebuildGreenInvert(img, META)

    assert info["method"] == "invert"
    assert _magenta(out).sum() == 0
