""" Tests for the FF meteor aperture photometry and its frame-split correction
(RMS/Routines/FFMeteorPhotometry.py).

A synthetic meteor (Gaussian PSF moving at constant speed, integrated over each frame's exposure,
noise, 8-bit) is compressed into FF planes with the real compressor. The true light of every frame is
known, so the light the FF drops at frame boundaries can be measured and compared with the model, and
the corrected photometry compared with the truth. """

from __future__ import absolute_import, division, print_function

import math

import numpy as np

from RMS.CompressionCy import compressFrames
from RMS.Routines.FFMeteorPhotometry import FFAperturePhotometry, splitLossFraction, SPLIT_K


def _meteorFF(v, angle_deg=0.0, sigma=1.2, peak=60.0, nframes=12, f_start=100, noise=1.5, bg=30.0,
    shape=(160, 320), seed=3):
    """ Compress a synthetic meteor into FF planes.

    Return:
        (maxpixel, maxframe, avepixel, truth, frames): truth[k] is the noise-free meteor light of
            FF frame k (sum over the image, codes), for the meteor's frames.
    """

    rng = np.random.RandomState(seed)
    h, w = shape
    # Brightness per unit time such that a pixel on the track peaks at ~peak codes in its frame
    amp = peak*max(v, 1.0)/(sigma*math.sqrt(2*math.pi))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ux, uy = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    x0, y0 = 20.0, h/2.0 - uy*v*nframes/2.0

    frames = bg + noise*rng.standard_normal((256, h, w))
    truth = {}
    sub = 24
    for j in range(nframes):
        k = f_start + j
        img = np.zeros(shape)
        # Integrate the moving PSF over the frame's exposure (the whole frame period)
        for t in (np.arange(sub) + 0.5)/sub:
            px, py = x0 + ux*v*(j + t), y0 + uy*v*(j + t)
            img += amp/sub*np.exp(-((xx - px)**2 + (yy - py)**2)/(2*sigma**2))
        frames[k] += img
        truth[k] = img.sum()

    frames = np.clip(np.round(frames), 0, 255).astype(np.uint8)
    ftp = compressFrames(frames, -1)[0]
    maxpixel, maxframe, avepixel = ftp[0], ftp[1], ftp[2]

    return maxpixel, maxframe, avepixel, truth


def _linePoints(maxpixel, maxframe, avepixel, frames, thresh=12):
    """ What the detector hands over: (x, y, frame) of the meteor's threshold passers. """

    diff = maxpixel.astype(np.float64) - avepixel
    ys, xs = np.nonzero((diff > thresh) & np.isin(maxframe, frames))

    return np.c_[xs, ys, maxframe[ys, xs]].astype(np.float64)


def _measure(v, angle_deg=0.0, **kw):
    maxpixel, maxframe, avepixel, truth = _meteorFF(v, angle_deg=angle_deg, **kw)
    frames = sorted(truth)
    lp = _linePoints(maxpixel, maxframe, avepixel, frames)
    phot = FFAperturePhotometry(maxframe, maxpixel.astype(np.float64) - avepixel, lp)

    # Interior frames: the first and last frames have one boundary only
    inner = frames[1:-1]
    measured = sum(phot.intensity(k) for k in inner)
    true = sum(truth[k] for k in inner)

    return phot, measured, true


def test_split_loss_fraction():
    assert abs(SPLIT_K - 0.7979) < 1e-4
    assert abs(splitLossFraction(1.2, 20.0) - SPLIT_K*1.2/20.0) < 1e-12
    assert math.isnan(splitLossFraction(1.2, 0.0))


def test_speed_and_sigma_recovered():
    phot, _, _ = _measure(15.0, angle_deg=30.0)
    assert abs(phot.v - 15.0) < 0.5
    assert 1.0 < phot.sigma() < 1.6


def test_split_loss_matches_model():
    # The aperture recovers everything but the split (and a little wing) light: the measured loss
    # agrees with sqrt(2/pi)*sigma/v at both speeds and on a diagonal track
    for v, ang in ((10.0, 0.0), (20.0, 0.0), (12.0, 45.0)):
        phot, measured, true = _measure(v, angle_deg=ang)
        loss = 1.0 - measured/true
        model = splitLossFraction(1.2, v)
        assert abs(loss - model) < 0.03, (v, ang, loss, model)


def test_corrected_photometry_recovers_truth():
    for v, ang in ((10.0, 0.0), (20.0, 0.0), (12.0, 45.0)):
        phot, measured, true = _measure(v, angle_deg=ang)
        factor, loss = phot.splitCorrection()
        assert factor > 1.0
        assert abs(measured*factor/true - 1.0) < 0.04, (v, ang, measured*factor/true)


def test_no_correction_outside_validated_range():
    # Too slow to have segment interiors (no sigma): uncorrected
    phot, _, _ = _measure(3.0, nframes=30)
    factor, loss = phot.splitCorrection(max_loss=0.25)
    assert factor == 1.0 and math.isnan(loss)

    # Loss above max_loss: uncorrected, the loss is still reported
    phot, _, _ = _measure(6.0, nframes=20)
    factor, loss = phot.splitCorrection(max_loss=0.10)
    assert factor == 1.0 and loss > 0.10


def test_bright_wide_object_not_clipped():
    # A bloomed object (sigma 3 px) is wider than the default 4.5 px half width: the aperture widens
    # to the frame's own line points, so only the frame-split light is missing (1 - 0.24); a fixed
    # +-4.5 px aperture would also clip ~13 % of the profile (ratio ~0.66)
    maxpixel, maxframe, avepixel, truth = _meteorFF(10.0, sigma=3.0, peak=120.0)
    frames = sorted(truth)
    lp = _linePoints(maxpixel, maxframe, avepixel, frames)
    phot = FFAperturePhotometry(maxframe, maxpixel.astype(np.float64) - avepixel, lp)
    inner = frames[1:-1]
    ratio = sum(phot.intensity(k) for k in inner)/sum(truth[k] for k in inner)
    assert ratio > 1.0 - splitLossFraction(3.0, 10.0) - 0.04, ratio


def test_aperture_follows_frame_centroids():
    # Centring each frame on its centroid: a track that bends away from the straight-line fit is
    # still measured (fit positions alone would miss it)
    maxpixel, maxframe, avepixel, truth = _meteorFF(12.0, nframes=12)
    frames = sorted(truth)
    lp = _linePoints(maxpixel, maxframe, avepixel, frames)
    # Fake a bend: shift the line points of the second half 8 px across, as a curved path would
    bent = lp.copy()
    bent[bent[:, 2] >= frames[6], 1] -= 8.0
    phot = FFAperturePhotometry(maxframe, maxpixel.astype(np.float64) - avepixel, bent)
    inner = frames[1:-1]
    sig = maxpixel.astype(np.float64) - avepixel
    measured = 0.0
    for k in inner:
        own = lp[lp[:, 2] == k]
        w = sig[own[:, 1].astype(int), own[:, 0].astype(int)]
        center = (np.sum(own[:, 0]*w)/np.sum(w), np.sum(own[:, 1]*w)/np.sum(w))
        measured += phot.intensity(k, center=center)
    assert measured/sum(truth[k] for k in inner) > 0.85
