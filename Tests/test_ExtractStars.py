"""
Tests of the star extraction at the bit depth of the image data (ExtractStars, Image.effectiveBitDepth), and
of the table of stars printed by the ExtractStars command line.

The intensity threshold, the too-bright cutoff and the saturation level of the star extraction are given for
8-bit data and scaled to the bit depth. The configured bit depth describes the camera, but FF files from live
capture are always 8-bit, so the scaling has to follow the data: 8-bit images must give the same stars
whatever bit depth is configured. The tests use synthetic star fields of Gaussian stars on a flat background
with noise.
"""

import datetime
from types import SimpleNamespace

import numpy as np

import RMS.ExtractStars as ExtractStars
from RMS.Routines import Image


def starField(n_stars=40, height=200, width=300, scale=1, dtype=np.uint8, seed=3):
    """ A synthetic star field and the true star positions.

    Keyword arguments:
        n_stars: [int] Number of stars. 40 by default.
        height: [int] Image height in pixels. 200 by default.
        width: [int] Image width in pixels. 300 by default.
        scale: [float] Factor which scales the 8-bit levels to the bit depth of the image. 1 by default.
        dtype: [data-type] Data type of the image. np.uint8 by default.
        seed: [int] Seed of the random numbers. 3 by default.

    Return:
        (img, stars):
            - img: [ndarray] The image.
            - stars: [list] (y, x, amplitude) of every star, the amplitude in 8-bit units.
    """

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]

    # Place Gaussian stars on a flat background (8-bit levels)
    stars = [(rng.uniform(20, height - 20), rng.uniform(20, width - 20), rng.uniform(40, 200))
             for _ in range(n_stars)]
    img = np.zeros((height, width)) + 30.0
    for y, x, amp in stars:
        img += amp*np.exp(-((yy - y)**2 + (xx - x)**2)/(2*1.3**2))

    # Add noise, scale to the bit depth of the image and quantize
    img = scale*(img + rng.normal(0, 2, (height, width)))
    img = np.clip(img, 0, np.iinfo(dtype).max).astype(dtype)

    return img, stars


def starConfig(bit_depth):
    """ The star extraction settings of the default config, with the given bit depth.

    Arguments:
        bit_depth: [int] Configured bit depth.

    Return:
        [SimpleNamespace] The config.
    """

    return SimpleNamespace(
        max_global_intensity=150, border=10, neighborhood_size=10, intensity_threshold=18,
        segment_radius=4, roundness_threshold=0.5, max_feature_ratio=0.8, bit_depth=bit_depth,
        gamma=1.0, max_stars=1000, stationID='XX0001'
    )


def fakeFFRead(monkeypatch, img):
    """ Make FFfile.read return an FF file whose average pixel image is the given image.

    Arguments:
        monkeypatch: [MonkeyPatch] The pytest monkeypatch fixture.
        img: [ndarray] The average pixel image.
    """

    monkeypatch.setattr(ExtractStars.FFfile, 'read',
                        lambda ff_dir, ff_name: SimpleNamespace(avepixel=img.copy()))


def test_effective_bit_depth_caps_integer_images():
    """ The bit depth is capped at the size of an integer data type, never raised, and kept for
        floating-point images. """

    # 8-bit data with a larger configured bit depth
    assert Image.effectiveBitDepth(np.zeros(2, dtype=np.uint8), 12) == 8
    assert Image.effectiveBitDepth(np.zeros(2, dtype=np.uint8), 8) == 8

    # 16-bit containers keep the configured bit depth, also when it is lower
    assert Image.effectiveBitDepth(np.zeros(2, dtype=np.uint16), 12) == 12
    assert Image.effectiveBitDepth(np.zeros(2, dtype=np.uint16), 16) == 16
    assert Image.effectiveBitDepth(np.zeros(2, dtype=np.uint16), 8) == 8

    # Floating-point images don't tell the bit depth
    assert Image.effectiveBitDepth(np.zeros(2, dtype=np.float32), 12) == 12


def test_extract_stars_ignores_config_bit_depth_above_data_depth():
    """ 8-bit data with a larger configured bit depth gives the same stars as with bit_depth = 8. """

    img, _ = starField()
    img_median = np.median(img)

    # The reference extraction at the bit depth of the data
    ref = ExtractStars.extractStars(img, img_median=img_median, bit_depth=8)
    assert len(ref[0]) > 20

    # Larger configured bit depths must not change the stars
    for bit_depth in (10, 12, 16):
        res = ExtractStars.extractStars(img, img_median=img_median, bit_depth=bit_depth)
        assert len(res[0]) == len(ref[0])
        assert np.allclose(res[0], ref[0])
        assert np.allclose(res[1], ref[1])
        assert np.allclose(res[3], ref[3])


def test_extract_stars_ff_uses_data_bit_depth(monkeypatch):
    """ extractStarsFF on an 8-bit FF file with config.bit_depth = 12 finds the same stars as with 8. """

    # An 8-bit FF file with the star field
    img, _ = starField()
    fakeFFRead(monkeypatch, img)

    # Extract the stars with both configured bit depths
    results = []
    for bit_depth in (8, 12):
        results.append(ExtractStars.extractStarsFF('.', 'FF_test.fits', config=starConfig(bit_depth)))

    # The same stars at the same positions
    assert len(results[0][1]) > 20
    assert len(results[1][1]) == len(results[0][1])
    assert np.allclose(results[1][1], results[0][1])
    assert np.allclose(results[1][2], results[0][2])


def test_extract_stars_ff_skips_bright_8bit_image(monkeypatch):
    """ The too-bright cutoff is applied at the bit depth of the data: an 8-bit FF file brighter than
        max_global_intensity is skipped also when config.bit_depth = 12. """

    # An 8-bit image whose median (200) is above the cutoff (150)
    img, _ = starField()
    img = np.clip(img.astype(np.int32) + 170, 0, 255).astype(np.uint8)
    fakeFFRead(monkeypatch, img)

    # Record the calls of the star extraction
    calls = []

    def recordExtraction(*args, **kwargs):
        """ Record the call and report a failed extraction. """

        calls.append(kwargs)

        return False

    monkeypatch.setattr(ExtractStars, 'extractStars', recordExtraction)

    # The image is skipped before the extraction with both configured bit depths
    for bit_depth in (8, 12):
        ExtractStars.extractStarsFF('.', 'FF_test.fits', config=starConfig(bit_depth))

    assert calls == []


def test_extract_stars_img_handle_uses_data_bit_depth():
    """ extractStarsImgHandle on 8-bit frame chunks with config.bit_depth = 12 finds the same stars as with
        8. """

    # An 8-bit star field
    img, _ = starField()

    class ChunkHandle(object):
        """ An image handle with a single chunk whose average pixel image is the star field. """

        total_fr_chunks = 1
        chunk_frames = 256

        def setFrame(self, frame):
            """ Go to the given frame (there is only one chunk). """

        def loadChunk(self):
            """ Return the chunk, with the star field as its average pixel image. """

            return SimpleNamespace(avepixel=img.copy())

        def currentTime(self, dt_obj=False, beginning=False):
            """ Return the time of the beginning of the chunk. """

            return datetime.datetime(2026, 11, 23, 0, 0, 0)

        def nextChunk(self):
            """ Go to the next chunk (there is none). """

    # Extract the stars with both configured bit depths
    star_lists = []
    for bit_depth in (8, 12):
        star_lists.append(ExtractStars.extractStarsImgHandle(ChunkHandle(), config=starConfig(bit_depth)))

    # The same rows of the same chunk
    _, rows_8 = star_lists[0][0]
    _, rows_12 = star_lists[1][0]
    assert len(rows_8) > 20
    assert len(rows_12) == len(rows_8)
    assert np.allclose(np.array(rows_12), np.array(rows_8))


def test_extract_stars_auto_finds_stars_on_16bit_image():
    """ extractStarsAuto scales its 8-bit thresholds to the bit depth once: a 16-bit star field gives the
        stars. """

    # The star field at 16 bits, and the stars extracted from it
    img, stars = starField(scale=256, dtype=np.uint16)
    x_data, y_data = ExtractStars.extractStarsAuto(img, bit_depth=16)[:2]

    # Most of the true stars are found at their positions (blended pairs and stars near the border may
    #   be missed)
    found = 0
    for y, x, _ in stars:
        if len(x_data) and (np.min(np.hypot(np.array(x_data) - x, np.array(y_data) - y)) < 0.5):
            found += 1

    assert found >= 0.75*len(stars)


def test_extract_stars_auto_on_8bit_ff_with_larger_header_bit_depth():
    """ An 8-bit FF file whose header gives a larger bit depth (written with config.bit_depth = 12, which
        AstrometryNet hands to extractStarsAuto) gives the same stars as with bit_depth = 8. """

    # The stars of an 8-bit field with the bit depth of the data and with the one of the FF header
    img, _ = starField()
    ref = ExtractStars.extractStarsAuto(img, bit_depth=8)
    res = ExtractStars.extractStarsAuto(img, bit_depth=12)

    # The same stars at the same positions
    assert len(ref[0]) > 20
    assert len(res[0]) == len(ref[0])
    assert np.allclose(res[0], ref[0])
    assert np.allclose(res[1], ref[1])


def test_print_star_table_uses_calstars_column_order(capsys):
    """ The table of the command line prints the CALSTARS columns (Y X IntensSum Ampltd ...) under the right
        headers. """

    # Y = 10.5, X = 20.25, IntensSum = 500, Ampltd = 50, FWHM = 2.5, BgLvl = 30, SNR = 12.5, NSatPx = 0
    ExtractStars.printStarTable([(10.5, 20.25, 500.0, 50.0, 2.5, 30.0, 12.5, 0)])

    # The headers, and the values of the row under them
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].split()[:4] == ['ROW', 'COL', 'amp', 'intens']
    assert lines[1].split()[:4] == ['10.50', '20.25', '50', '500']
