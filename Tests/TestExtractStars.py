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
        events.append(('plot', len(x_data), len(y_data)))

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
        ('plot', extra_info['num_candidates'], extra_info['num_candidates']),
        ('fit', extra_info['num_candidates'], extra_info['num_candidates'])
    ]


def testOverLimitCandidatesAreShownThenRejected(monkeypatch):
    candidate_counts = []
    extra_info = {}

    def recordCandidates(img, x_data, y_data, **kwargs):
        candidate_counts.append(len(x_data))

    def unexpectedFit(*args, **kwargs):
        raise AssertionError('PSF fitting must not run for an over-limit candidate set')

    monkeypatch.setattr(ExtractStars, 'plotStars', recordCandidates)
    monkeypatch.setattr(ExtractStars, 'fitPSF', unexpectedFit)

    status = ExtractStars.extractStars(
        _singleCandidateImage(), border=1, neighborhood_size=3,
        max_star_candidates=0, extra_info=extra_info, show_candidates=True
    )

    assert status is False
    assert candidate_counts == [extra_info['num_candidates']]


def testDefaultOverLimitRejectionDoesNotPlot(monkeypatch):
    def unexpectedCall(*args, **kwargs):
        raise AssertionError('Plotting and fitting must not run after the default early rejection')

    monkeypatch.setattr(ExtractStars, 'plotStars', unexpectedCall)
    monkeypatch.setattr(ExtractStars, 'fitPSF', unexpectedCall)

    status = ExtractStars.extractStars(
        _singleCandidateImage(), border=1, neighborhood_size=3, max_star_candidates=0
    )

    assert status is False


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
