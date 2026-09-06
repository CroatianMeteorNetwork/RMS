from types import SimpleNamespace

import numpy as np

import RMS.ExtractStars as ExtractStars


def _singleCandidateImage():
    """Return a small image with one isolated local maximum."""

    img = np.full((32, 32), 10, dtype=np.float32)
    img[16, 16] = 255

    return img


def _successfulFit(img, img_median, x_init, y_init, **kwargs):
    """Return valid PSF-fit data for every supplied candidate."""

    count = len(x_init)

    return (
        list(x_init), list(y_init), [100]*count, [200]*count,
        [1]*count, [1]*count, [0]*count, [10]*count, [0]*count
    )


def testCandidatesAreShownBeforePsfFit(monkeypatch):
    events = []
    extra_info = {}

    def recordCandidates(img, x_data, y_data, **kwargs):
        fitted_count = len(kwargs['x_fitted']) if kwargs.get('x_fitted') is not None else 0
        events.append(('plot', len(x_data), len(y_data), fitted_count))

    def recordFit(*args, **kwargs):
        events.append(('fit', len(args[2]), len(args[3])))
        return _successfulFit(*args, **kwargs)

    monkeypatch.setattr(ExtractStars, 'plotStars', recordCandidates)
    monkeypatch.setattr(ExtractStars, 'fitPSF', recordFit)

    status = ExtractStars.extractStars(
        _singleCandidateImage(), border=1, neighborhood_size=3,
        extra_info=extra_info, show_candidates=True
    )

    assert status is not False
    assert events == [
        ('plot', extra_info['num_candidates'], extra_info['num_candidates'], 0),
        ('fit', extra_info['num_candidates'], extra_info['num_candidates']),
        ('plot', extra_info['num_candidates'], extra_info['num_candidates'],
         extra_info['num_candidates'])
    ]


def _twoCandidateImage():
    """Return a small image with two isolated local maxima of different prominence."""

    img = _singleCandidateImage()
    img[8, 8] = 200

    return img


def testOverLimitCandidatesAreSubsampledThenShownAndFitted(monkeypatch):
    """ Over the candidate limit, the most prominent candidates are kept (never the whole set
        rejected), and with show_candidates both the kept set and the fit result are plotted.
    """
    events = []
    extra_info = {}

    def recordCandidates(img, x_data, y_data, **kwargs):
        fitted_count = len(kwargs['x_fitted']) if kwargs.get('x_fitted') is not None else 0
        events.append(('plot', len(x_data), fitted_count))

    def recordFit(*args, **kwargs):
        events.append(('fit', len(args[2])))
        return _successfulFit(*args, **kwargs)

    monkeypatch.setattr(ExtractStars, 'plotStars', recordCandidates)
    monkeypatch.setattr(ExtractStars, 'fitPSF', recordFit)

    status = ExtractStars.extractStars(
        _twoCandidateImage(), border=1, neighborhood_size=3,
        max_star_candidates=1, extra_info=extra_info, show_candidates=True
    )

    assert status is not False
    assert extra_info['num_candidates'] > 1
    assert events == [('plot', 1, 0), ('fit', 1), ('plot', 1, 1)]


def testDefaultOverLimitSubsampleDoesNotPlot(monkeypatch):
    """ Without show_candidates, an over-limit candidate set is subsampled and fitted with no
        plotting at all.
    """
    fit_sizes = []

    def unexpectedPlot(*args, **kwargs):
        raise AssertionError('Plotting must not run unless show_candidates is requested')

    def recordFit(*args, **kwargs):
        fit_sizes.append(len(args[2]))
        return _successfulFit(*args, **kwargs)

    monkeypatch.setattr(ExtractStars, 'plotStars', unexpectedPlot)
    monkeypatch.setattr(ExtractStars, 'fitPSF', recordFit)

    status = ExtractStars.extractStars(
        _twoCandidateImage(), border=1, neighborhood_size=3, max_star_candidates=1
    )

    assert status is not False
    assert fit_sizes == [1]


def testCandidateDisplayProcessesFfFilesSequentially(monkeypatch, tmp_path):
    extraction_calls = []

    config = SimpleNamespace(stationID='XX0001', height=32, width=32)

    monkeypatch.setattr(ExtractStars, 'loadImageCalibration', lambda *args: (None, None, None))
    monkeypatch.setattr(ExtractStars.os, 'listdir', lambda path: ['FF_b.bin', 'FF_a.bin'])
    monkeypatch.setattr(ExtractStars.FFfile, 'validFFName', lambda name: True)
    monkeypatch.setattr(ExtractStars.CALSTARS, 'writeCALSTARS', lambda *args: None)

    class UnexpectedPool(object):
        def __init__(self, *args, **kwargs):
            raise AssertionError('Candidate display must not create a worker pool')

    def extractFf(ff_dir, ff_name, **kwargs):
        extraction_calls.append((ff_name, kwargs['show_candidates']))
        return ff_name, [1], [2], [3], [4], [5], [6], [7], [8]

    monkeypatch.setattr(ExtractStars, 'QueuedPool', UnexpectedPool)
    monkeypatch.setattr(ExtractStars, 'extractStarsFF', extractFf)

    star_list = ExtractStars.extractStarsAndSave(config, str(tmp_path), show_candidates=True)

    assert extraction_calls == [('FF_a.bin', True), ('FF_b.bin', True)]
    assert [entry[0] for entry in star_list] == ['FF_a.bin', 'FF_b.bin']


def testPlotStarsSupportsFfStructuresAndBitDepth(monkeypatch):
    adjust_call = {}
    img = np.zeros((8, 8), dtype=np.uint16)
    ff = SimpleNamespace(avepixel=img)

    def adjustLevels(input_img, minv, gamma, maxv, nbits=None):
        adjust_call['img'] = input_img
        adjust_call['maxv'] = maxv
        adjust_call['nbits'] = nbits
        return input_img

    monkeypatch.setattr(ExtractStars.Image, 'adjustLevels', adjustLevels)
    monkeypatch.setattr(ExtractStars.plt, 'show', lambda **kwargs: None)

    ExtractStars.plotStars(ff, [3], [4], title='Candidates')

    assert adjust_call['img'] is img
    assert adjust_call['maxv'] == 2**16 - 1
    assert adjust_call['nbits'] == 16


def testPlotStarsDefaultsFloatingImagesToEightBits(monkeypatch):
    adjust_call = {}
    img = np.zeros((8, 8), dtype=np.float32)

    def adjustLevels(input_img, minv, gamma, maxv, nbits=None):
        adjust_call['maxv'] = maxv
        adjust_call['nbits'] = nbits
        return input_img

    monkeypatch.setattr(ExtractStars.Image, 'adjustLevels', adjustLevels)
    monkeypatch.setattr(ExtractStars.plt, 'show', lambda **kwargs: None)

    ExtractStars.plotStars(img, [3], [4], title='Candidates')

    assert adjust_call == {'maxv': 2**8 - 1, 'nbits': 8}


def testPlotStarsAutomaticallyAdjustsBackgroundLevels(monkeypatch):
    adjust_call = {}
    img = np.arange(10000, dtype=np.uint16).reshape((100, 100))

    def adjustLevels(input_img, minv, gamma, maxv, nbits=None):
        adjust_call['minv'] = minv
        adjust_call['gamma'] = gamma
        adjust_call['maxv'] = maxv
        return input_img

    monkeypatch.setattr(ExtractStars.Image, 'adjustLevels', adjustLevels)
    monkeypatch.setattr(ExtractStars.plt, 'show', lambda **kwargs: None)

    ExtractStars.plotStars(img, [3], [4])

    assert adjust_call['minv'] == np.percentile(img, 1.0)
    assert adjust_call['gamma'] == 1.3
    assert adjust_call['maxv'] == np.percentile(img, 99.99)


def testPlotStarsMarksFittedPositions(monkeypatch):
    plot_data = {}
    original_subplots = ExtractStars.plt.subplots

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

    ax = plot_data['ax']
    assert len(ax.patches) == 2
    assert len(ax.lines) == 1
    assert list(ax.lines[0].get_xdata()) == [3.25]
    assert list(ax.lines[0].get_ydata()) == [4.25]
