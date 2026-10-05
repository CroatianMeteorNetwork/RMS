""" Tests for the camera response table (RMS.Routines.Response).

Covers:
    - the power-law ResponseCurve reproduces the old float-gamma arithmetic exactly, in the
      gamma helpers and in both compressors, so cameras without a table are unchanged
    - a camera node table (built here like the science firmware builds it: 257 nodes, straight
      lines between them) round-trips, parses from a `gamma decode` reply and survives json
    - the compressors' linear-domain average with a table matches an exact numpy reference
    - the night sidecar decides the response a night is reduced with
    - the table id travels in the FF header (AVERESP)

Run directly (python -m Tests.test_Response) or via pytest.
"""

from __future__ import print_function, division, absolute_import

import os
import shutil
import tempfile

import numpy as np

from RMS.CompressionCy import compressFrames
from RMS.Routines.DynamicFTPCompressionCy import FFMimickInterface
from RMS.Routines import Image
from RMS.Routines.Response import ResponseCurve, RESPONSE_FILE, useNightResponse


def nodeTableReply(nodes=257):
    """ A `gamma decode` reply for a gamma-0.5 node table, as the camera computes it: node i of
    the table maps input i/(nodes - 1) to round(4095*sqrt(i/(nodes - 1))), and an 8-bit code c
    (12-bit output c*4095/255) decodes by inverting the straight lines between nodes. """
    last = nodes - 1
    xs = np.arange(nodes)*4096.0/last
    ys = np.floor(4095.0*np.sqrt(np.arange(nodes)/last) + 0.5)
    lin = np.interp(np.arange(256)*4095.0/255.0, ys, xs)/4095.0*255.0
    return "gamma_decode decode=table nodes=%d " % nodes + " ".join("%.3f" % v for v in lin)


def table():
    return ResponseCurve.fromCameraReply(nodeTableReply())


def frames(seed=1, n=256, h=40, w=50):
    rng = np.random.default_rng(seed)
    lam = rng.uniform(0, 3000, (h, w))
    lin = rng.poisson(np.broadcast_to(lam, (n, h, w)))
    return np.clip(np.floor(255*np.sqrt(lin/4000.0) + 0.5), 0, 255).astype(np.uint8)


def testPowerCurveIsTheOldArithmetic():
    img = np.linspace(0, 255, 1000).astype(np.float32)
    for g in (0.45, 0.5, 1.0):
        old = Image.gammaCorrectionImage(img.copy(), g, wp=255, out_type=np.float32)
        new = Image.gammaCorrectionImage(img.copy(), ResponseCurve.power(g), wp=255, out_type=np.float32)
        assert np.array_equal(old, new)
        for v in (0, 3, 17.5, 200):
            assert Image.gammaCorrectionScalar(v, g) == Image.gammaCorrectionScalar(v, ResponseCurve.power(g))


def testPowerCurveCompressorsUnchanged():
    f = frames()
    _, a, _ = compressFrames(f, -1, 0.5)
    _, b, _ = compressFrames(f, -1, 0.5, response=ResponseCurve.power(0.5))
    assert np.array_equal(a, b)

    def mimick(**kw):
        ff = FFMimickInterface(f.shape[1], f.shape[2], np.uint8, gamma=0.5, bit_depth=8, **kw)
        for fr in f:
            ff.addFrame(fr.astype(np.uint16))
        ff.finish()
        return ff.avepixel16
    assert np.array_equal(mimick(), mimick(response=ResponseCurve.power(0.5)))


def testTableParseRoundTripAndJson():
    t = table()
    assert t.info["nodes"] == 257 and not t.isPower
    codes = np.arange(256)
    assert np.all(np.diff(t.linear) > 0)
    assert np.allclose(t.encode(t.decode(codes)), codes, atol=1e-6)

    # straight line below the first node (codes 0..16 on a 257-node table), not code**2
    assert np.allclose(np.diff(t.linear[:16]), t.linear[1], rtol=0.05)

    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, RESPONSE_FILE)
        t.save(p)
        assert ResponseCurve.load(p).ident() == t.ident()
    finally:
        shutil.rmtree(d)


def testTableScalarMatchesImage():
    t = table()
    vals = np.array([0, 1, 5, 16, 17, 40.5, 128, 254, 255], dtype=np.float32)
    img = Image.gammaCorrectionImage(vals.copy(), t, wp=255, out_type=np.float64)
    for v, o in zip(vals, img):
        assert abs(Image.gammaCorrectionScalar(float(v), t) - o) < 1e-4
    # black/white points are fixed points of the correction
    assert img[0] == 0 and abs(img[-1] - 255) < 1e-9


def testTableCompressorsMatchReference():
    t = table()
    f = frames(seed=2)
    s = np.sort(f.astype(np.float64), axis=0)
    lin = 255*t.decode(s)

    _, a16, _ = compressFrames(f, -1, 0.5, response=t)
    ref = np.floor(256*t.encode(lin[4:-4].mean(axis=0)/255) + 0.5)
    assert np.array_equal(a16, ref)

    ff = FFMimickInterface(f.shape[1], f.shape[2], np.uint8, gamma=0.5, bit_depth=8, response=t)
    for fr in f:
        ff.addFrame(fr.astype(np.uint16))
    ff.finish()
    ref = np.floor(256*t.encode(lin[1:-1].mean(axis=0)/255) + 0.5)
    assert np.max(np.abs(ff.avepixel16.astype(np.float64) - ref)) <= 1


def testNightSidecarDecides():
    class C(object):
        response_table = "auto"
        gamma = 0.5
        response = None
    d = tempfile.mkdtemp()
    try:
        c = C()
        assert useNightResponse(c, d) is None
        table().save(os.path.join(d, RESPONSE_FILE))
        assert useNightResponse(c, d).ident() == table().ident()
        c.response_table = "off"
        assert useNightResponse(c, d) is None
    finally:
        shutil.rmtree(d)


def testFitsCarriesTableId():
    from RMS.Formats import FFfits
    from Tests.TestFFAvepixel16 import makeFF
    ff = makeFF(ave16=(np.arange(32*48).reshape(32, 48) % 65536).astype(np.uint16))
    ff.avegamma = 0.5
    ff.averesp = table().ident()
    d = tempfile.mkdtemp()
    try:
        FFfits.write(ff, d, "FF_XX0001_resp.fits")
        back = FFfits.read(d, "FF_XX0001_resp.fits", full_filename=True)
        assert back.averesp == table().ident()
    finally:
        shutil.rmtree(d)


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test") and callable(v):
            v()
            print("ok", k)
