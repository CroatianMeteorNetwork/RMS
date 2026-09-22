""" Tests for the star candidate display and plotting helpers in RMS.ExtractStars. """

from __future__ import print_function, division, absolute_import

from types import SimpleNamespace

import numpy as np

import RMS.ExtractStars as ExtractStars


def _singleCandidateImage():
    """ Return a small image with one isolated local maximum. """

    img = np.full((32, 32), 10, dtype=np.float32)
    img[16, 16] = 255

    return img


def _successfulFit(img, img_median, x_init, y_init, **kwargs):
    """ Return valid PSF-fit data for every supplied candidate. """

    count = len(x_init)

    return (
        list(x_init), list(y_init), [100]*count, [200]*count,
        [1]*count, [1]*count, [0]*count, [10]*count, [0]*count
    )


def testCandidatesAreShownBeforePsfFit(monkeypatch):
    """ With show_candidates the raw candidates are plotted, then fitted, then plotted with the fits. """

    events = []
    extra_info = {}

    # Record every plot and fit call, in the order in which extractStars makes them
    def recordCandidates(img, x_data, y_data, **kwargs):
        fitted_count = len(kwargs['x_fitted']) if kwargs.get('x_fitted') is not None else 0
        events.append(('plot', len(x_data), len(y_data), fitted_count))

    def recordFit(*args, **kwargs):
        events.append(('fit', len(args[2]), len(args[3])))
        return _successfulFit(*args, **kwargs)

    monkeypatch.setattr(ExtractStars, 'plotStars', recordCandidates)
    monkeypatch.setattr(ExtractStars, 'fitPSF', recordFit)

    # Run the extraction with the candidate display turned on
    status = ExtractStars.extractStars(
        _singleCandidateImage(), border=1, neighborhood_size=3,
        extra_info=extra_info, show_candidates=True
    )

    # Candidates are plotted first without fits, then fitted, then plotted again with the fits
    assert status is not False
    assert events == [
        ('plot', extra_info['num_candidates'], extra_info['num_candidates'], 0),
        ('fit', extra_info['num_candidates'], extra_info['num_candidates']),
        ('plot', extra_info['num_candidates'], extra_info['num_candidates'],
         extra_info['num_candidates'])
    ]


def testOverLimitCandidatesAreShownThenRejected(monkeypatch):
    """ An over-limit candidate set is still plotted once, but PSF fitting is skipped. """

    candidate_counts = []
    extra_info = {}

    # Count the plot calls, and fail outright if the PSF fit is reached
    def recordCandidates(img, x_data, y_data, **kwargs):
        candidate_counts.append(len(x_data))

    def unexpectedFit(*args, **kwargs):
        raise AssertionError('PSF fitting must not run for an over-limit candidate set')

    monkeypatch.setattr(ExtractStars, 'plotStars', recordCandidates)
    monkeypatch.setattr(ExtractStars, 'fitPSF', unexpectedFit)

    # A max_star_candidates of 0 rejects the frame, but the candidates are still shown once
    status = ExtractStars.extractStars(
        _singleCandidateImage(), border=1, neighborhood_size=3,
        max_star_candidates=0, extra_info=extra_info, show_candidates=True
    )

    # The frame is rejected, but the single plot call did happen
    assert status is False
    assert candidate_counts == [extra_info['num_candidates']]


def testDefaultOverLimitRejectionDoesNotPlot(monkeypatch):
    """ Without show_candidates the over-limit early rejection neither plots nor fits. """

    # Neither the plotting nor the fitting may be reached in the default (non-display) path
    def unexpectedCall(*args, **kwargs):
        raise AssertionError('Plotting and fitting must not run after the default early rejection')

    monkeypatch.setattr(ExtractStars, 'plotStars', unexpectedCall)
    monkeypatch.setattr(ExtractStars, 'fitPSF', unexpectedCall)

    status = ExtractStars.extractStars(
        _singleCandidateImage(), border=1, neighborhood_size=3, max_star_candidates=0
    )

    assert status is False


def testCandidateDisplayProcessesFfFilesSequentially(monkeypatch, tmp_path):
    """ The candidate display runs the extraction in the main process, in sorted FF order. """

    extraction_calls = []

    config = SimpleNamespace(stationID='XX0001', height=32, width=32)

    # Stub out the calibration, the directory listing and the CALSTARS writing
    monkeypatch.setattr(ExtractStars, 'loadImageCalibration', lambda *args: (None, None, None))
    monkeypatch.setattr(ExtractStars.os, 'listdir', lambda path: ['FF_b.bin', 'FF_a.bin'])
    monkeypatch.setattr(ExtractStars.FFfile, 'validFFName', lambda name: True)
    monkeypatch.setattr(ExtractStars.CALSTARS, 'writeCALSTARS', lambda *args: None)

    # Creating a worker pool would mean the extraction did not stay in the main process
    class UnexpectedPool(object):
        def __init__(self, *args, **kwargs):
            raise AssertionError('Candidate display must not create a worker pool')

    # Record the FF files in the order in which they are handed to the extraction
    def extractFf(ff_dir, ff_name, **kwargs):
        extraction_calls.append((ff_name, kwargs['show_candidates']))
        return ff_name, [1], [2], [3], [4], [5], [6], [7], [8]

    monkeypatch.setattr(ExtractStars, 'QueuedPool', UnexpectedPool)
    monkeypatch.setattr(ExtractStars, 'extractStarsFF', extractFf)

    star_list = ExtractStars.extractStarsAndSave(config, str(tmp_path), show_candidates=True)

    # The unsorted listing must be processed in sorted order, sequentially
    assert extraction_calls == [('FF_a.bin', True), ('FF_b.bin', True)]
    assert [entry[0] for entry in star_list] == ['FF_a.bin', 'FF_b.bin']


def testPlotStarsSupportsFfStructuresAndBitDepth(monkeypatch):
    """ An FF structure is unwrapped and the bit depth is inferred from the integer image type. """

    adjust_call = {}
    img = np.zeros((8, 8), dtype=np.uint16)
    ff = SimpleNamespace(avepixel=img)

    # Capture the levels that plotStars asks for instead of actually stretching the image
    def adjustLevels(input_img, minv, gamma, maxv, nbits=None):
        adjust_call['img'] = input_img
        adjust_call['maxv'] = maxv
        adjust_call['nbits'] = nbits
        return input_img

    monkeypatch.setattr(ExtractStars.Image, 'adjustLevels', adjustLevels)
    monkeypatch.setattr(ExtractStars.plt, 'show', lambda **kwargs: None)

    ExtractStars.plotStars(ff, [3], [4], title='Candidates')

    # The FF structure must be unwrapped to its avepixel, and the 16 bit depth inferred from the dtype
    assert adjust_call['img'] is img
    assert adjust_call['maxv'] == 2**16 - 1
    assert adjust_call['nbits'] == 16


def testPlotStarsDefaultsFloatingImagesToEightBits(monkeypatch):
    """ Floating-point images have no inferable bit depth and default to 8 bits. """

    adjust_call = {}
    img = np.zeros((8, 8), dtype=np.float32)

    # Capture the levels that plotStars asks for
    def adjustLevels(input_img, minv, gamma, maxv, nbits=None):
        adjust_call['maxv'] = maxv
        adjust_call['nbits'] = nbits
        return input_img

    monkeypatch.setattr(ExtractStars.Image, 'adjustLevels', adjustLevels)
    monkeypatch.setattr(ExtractStars.plt, 'show', lambda **kwargs: None)

    ExtractStars.plotStars(img, [3], [4], title='Candidates')

    # A float image has no integer bit depth, so the 8 bit default is used
    assert adjust_call == {'maxv': 2**8 - 1, 'nbits': 8}


def testPlotStarsAutomaticallyAdjustsBackgroundLevels(monkeypatch):
    """ The display levels are stretched between the 1st and 99.99th percentile of the image. """

    adjust_call = {}
    img = np.arange(10000, dtype=np.uint16).reshape((100, 100))

    # Capture the levels that plotStars asks for
    def adjustLevels(input_img, minv, gamma, maxv, nbits=None):
        adjust_call['minv'] = minv
        adjust_call['gamma'] = gamma
        adjust_call['maxv'] = maxv
        return input_img

    monkeypatch.setattr(ExtractStars.Image, 'adjustLevels', adjustLevels)
    monkeypatch.setattr(ExtractStars.plt, 'show', lambda **kwargs: None)

    ExtractStars.plotStars(img, [3], [4])

    # The black and white points are the percentiles of the image, with a fixed gamma
    assert adjust_call['minv'] == np.percentile(img, 1.0)
    assert adjust_call['gamma'] == 1.3
    assert adjust_call['maxv'] == np.percentile(img, 99.99)


def testPlotStarsMarksFittedPositions(monkeypatch):
    """ Raw candidates are drawn as circles and the fitted positions as a separate marker series. """

    plot_data = {}
    original_subplots = ExtractStars.plt.subplots

    # Keep a handle on the axes that plotStars draws into
    def recordSubplots():
        fig, ax = original_subplots()
        plot_data['ax'] = ax
        return fig, ax

    monkeypatch.setattr(ExtractStars.plt, 'subplots', recordSubplots)
    monkeypatch.setattr(ExtractStars.plt, 'show', lambda **kwargs: None)

    ExtractStars.plotStars(
        np.zeros((8, 8), dtype=np.uint8), [3, 5], [4, 6],
        x_fitted=[3.25], y_fitted=[4.25]
    )

    # Two raw candidates become two circle patches, the one fitted position becomes one line series
    ax = plot_data['ax']
    assert len(ax.patches) == 2
    assert len(ax.lines) == 1
    assert list(ax.lines[0].get_xdata()) == [3.25]
    assert list(ax.lines[0].get_ydata()) == [4.25]


def _syntheticStarField(n_stars=40, height=200, width=300, seed=3):
    """ Return an 8-bit synthetic star field and the true star positions (y, x). """

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]

    # Place Gaussian stars on a flat background
    stars = [(rng.uniform(20, height - 20), rng.uniform(20, width - 20), rng.uniform(40, 200))
             for _ in range(n_stars)]
    img = np.zeros((height, width)) + 30.0
    for y, x, amp in stars:
        img += amp*np.exp(-((yy - y)**2 + (xx - x)**2)/(2*1.3**2))

    # Add noise and quantize to 8 bits
    img = np.clip(img + rng.normal(0, 2, (height, width)), 0, 255).astype(np.uint8)

    return img, stars


def testEffectiveBitDepthCapsIntegerImages():
    """ The bit depth is capped at the integer data type size and kept for floating-point images. """

    assert ExtractStars.Image.effectiveBitDepth(np.zeros(2, dtype=np.uint8), 12) == 8
    assert ExtractStars.Image.effectiveBitDepth(np.zeros(2, dtype=np.uint8), 8) == 8
    assert ExtractStars.Image.effectiveBitDepth(np.zeros(2, dtype=np.uint16), 12) == 12
    assert ExtractStars.Image.effectiveBitDepth(np.zeros(2, dtype=np.uint16), 16) == 16
    assert ExtractStars.Image.effectiveBitDepth(np.zeros(2, dtype=np.float32), 12) == 12


def testExtractStarsIgnoresConfigBitDepthAboveDataDepth():
    """ 8-bit data with a larger configured bit depth must give the same stars as bit_depth = 8. """

    img, _ = _syntheticStarField()
    img_median = np.median(img)

    ref = ExtractStars.extractStars(img, img_median=img_median, bit_depth=8)
    assert len(ref[0]) > 20

    for bit_depth in (10, 12, 16):
        res = ExtractStars.extractStars(img, img_median=img_median, bit_depth=bit_depth)
        assert len(res[0]) == len(ref[0])
        assert np.allclose(res[0], ref[0])
        assert np.allclose(res[3], ref[3])


def testExtractStarsFFUsesDataBitDepth(monkeypatch):
    """ extractStarsFF on an 8-bit FF with config.bit_depth = 12 must find the same stars as with 8. """

    img, _ = _syntheticStarField()

    # Fake an 8-bit FF file whose average pixel is the star field
    monkeypatch.setattr(ExtractStars.FFfile, 'read',
                        lambda ff_dir, ff_name: SimpleNamespace(avepixel=img.copy()))

    counts = []
    for bit_depth in (8, 12):
        config = SimpleNamespace(
            max_global_intensity=150, border=10, neighborhood_size=10, intensity_threshold=18,
            segment_radius=4, roundness_threshold=0.5, max_feature_ratio=0.8, bit_depth=bit_depth,
            gamma=1.0, max_stars=1000
        )
        res = ExtractStars.extractStarsFF('.', 'FF_test.fits', config=config)
        counts.append(len(res[1]))

    assert counts[0] > 20
    assert counts[0] == counts[1]


def testPrintStarTableUsesCalstarsColumnOrder(capsys):
    """ The CLI table must print the CALSTARS columns (Y X IntensSum Ampltd ...) under the right headers. """

    # Y = 10.5, X = 20.25, IntensSum = 500, Ampltd = 50, FWHM = 2.5, BgLvl = 30, SNR = 12.5, NSatPx = 0
    ExtractStars.printStarTable([(10.5, 20.25, 500.0, 50.0, 2.5, 30.0, 12.5, 0)])

    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].split()[:4] == ['ROW', 'COL', 'amp', 'intens']
    assert lines[1].split()[:4] == ['10.50', '20.25', '50', '500']
