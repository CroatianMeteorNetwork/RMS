"""
Tests of the gamma white point of the meteor detection at the bit depth of the image data (Detection).

With a gamma other than 1, the detection linearizes the images with a white point of 2**bit_depth - 1. FF
files from live capture are always 8-bit, so the white point has to follow the data: an 8-bit FF file must
give the same weights, intensities and backgrounds whatever bit depth is configured. The tests use small
synthetic FF files.
"""

import os
from types import SimpleNamespace

import numpy as np

import RMS.ConfigReader as cr
import RMS.Detection as Detection
from RMS.Formats import FFfile
from RMS.Formats.FFStruct import FFStruct
from RMS.Formats.FrameInterface import detectInputType


# Size of the synthetic FF files
FF_HEIGHT = 240
FF_WIDTH = 320


def writeMeteorFF(dir_path, seed=1):
    """ Write an 8-bit FF file with one meteor moving across the image between frames 100 and 140.

    Arguments:
        dir_path: [str] Directory to write the FF file into.

    Keyword arguments:
        seed: [int] Seed of the random numbers. 1 by default.

    Return:
        [str] Name of the FF file.
    """

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:FF_HEIGHT, 0:FF_WIDTH]

    # The background: average, standard deviation, maximum above the average and frames of the maxima
    avepixel = np.full((FF_HEIGHT, FF_WIDTH), 30.0) + rng.normal(0, 1, (FF_HEIGHT, FF_WIDTH))
    stdpixel = np.full((FF_HEIGHT, FF_WIDTH), 3.0)
    maxpixel = avepixel + 2*np.abs(rng.normal(0, 3, (FF_HEIGHT, FF_WIDTH)))
    maxframe = rng.integers(0, 256, (FF_HEIGHT, FF_WIDTH)).astype(np.float64)

    # Move a Gaussian meteor along a line, keeping the maximum of every pixel and its frame
    x_beg, y_beg, x_end, y_end = 60.0, 60.0, 260.0, 180.0
    frame_beg, frame_end = 100, 140
    for frame in range(frame_beg, frame_end + 1):
        t = (frame - frame_beg)/(frame_end - frame_beg)
        x_cent = x_beg + t*(x_end - x_beg)
        y_cent = y_beg + t*(y_end - y_beg)
        level = avepixel + 150.0*np.exp(-((xx - x_cent)**2 + (yy - y_cent)**2)/(2*1.2**2))

        brighter = level > maxpixel
        maxpixel[brighter] = level[brighter]
        maxframe[brighter] = frame

    # Pack the images into an 8-bit FF file
    ff = FFStruct()
    ff.nrows, ff.ncols = FF_HEIGHT, FF_WIDTH
    ff.nbits, ff.nframes, ff.first, ff.camno, ff.fps = 8, 256, 256, 'XX0001', 25.0
    ff.starttime = '2026-11-23T00:00:00.000000'
    ff.array = np.zeros((4, FF_HEIGHT, FF_WIDTH), np.uint8)
    for i, img in enumerate((maxpixel, maxframe, avepixel, stdpixel)):
        ff.array[i] = np.clip(np.round(img), 0, 255)

    ff_name = 'FF_XX0001_20261123_000000_000_0000256.fits'
    FFfile.write(ff, dir_path, ff_name, fmt='fits')

    return ff_name


def test_gamma_white_point_uses_data_bit_depth(monkeypatch):
    """ An 8-bit FF file with config.bit_depth = 12 is gamma corrected with the 8-bit white point. """

    # The thresholding is memoized on the FF object and is not under test here
    monkeypatch.setattr(Detection, 'thresholdFF', lambda ff, k1, j1: None)

    # A random 8-bit FF file
    rng = np.random.default_rng(0)
    maxpixel = rng.integers(50, 255, (16, 16)).astype(np.uint8)
    avepixel = rng.integers(10, 50, (16, 16)).astype(np.uint8)
    stdpixel = rng.integers(1, 10, (16, 16)).astype(np.uint8)
    ff = SimpleNamespace(maxpixel=maxpixel, avepixel=avepixel, stdpixel=stdpixel,
                         maxframe=np.zeros_like(maxpixel), nrows=16, ncols=16)
    img_handle = SimpleNamespace(ff=ff)

    # Prepare the FF file with both configured bit depths
    results = []
    for bit_depth in (8, 12):
        config = SimpleNamespace(k1_det=1.5, j1_det=9, j1=5, gamma=0.8, bit_depth=bit_depth,
                                 min_patch_intensity_multiplier=2.0, detection_binning_method='avg',
                                 detection_binning_factor=1)
        results.append(Detection.thresholdAndCorrectGammaFF(img_handle, config, None))

    # The gamma corrected image, weights and patch intensity don't depend on the configured bit depth
    assert np.allclose(results[0][1], results[1][1])
    assert np.allclose(results[0][2], results[1][2])
    assert np.isclose(results[0][3], results[1][3])


def test_detected_meteor_does_not_depend_on_config_bit_depth(tmp_path):
    """ With gamma 0.45, a meteor on an 8-bit FF file is detected with the same centroids, intensities,
        backgrounds and SNRs whether config.bit_depth is 8 or 12. """

    # An 8-bit FF file with a meteor
    ff_name = writeMeteorFF(str(tmp_path))

    # Detect the meteor with both configured bit depths
    detections = []
    for bit_depth in (8, 12):

        # The default config for the size of the FF file, with a gamma other than 1
        config = cr.Config()
        config.width, config.height = FF_WIDTH, FF_HEIGHT
        config.gamma = 0.45
        config.bit_depth = bit_depth

        # Load the FF file for the detection and detect the meteor
        img_handle = detectInputType(os.path.join(str(tmp_path), ff_name), config, skip_ff_dir=True,
                                     detection=True)
        detections.append(Detection.detectMeteors(img_handle, config))

    # One meteor, with the same centroids (frame, X, Y, intensity, background, SNR, saturated pixels)
    assert len(detections[0]) == 1
    assert len(detections[1]) == 1
    centroids_8 = np.array(detections[0][0][2])
    centroids_12 = np.array(detections[1][0][2])
    assert len(centroids_8) > 20
    assert centroids_12.shape == centroids_8.shape
    assert np.allclose(centroids_12, centroids_8)
