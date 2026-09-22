""" Tests for the frame time and frame selection in RMS.Astrometry.AutoPlatepar. """

from __future__ import print_function, division, absolute_import

import datetime

import numpy as np
import pytest

from RMS.Astrometry.AutoPlatepar import (directoryReferenceJD, frameReferenceTime, scoreFrameQuality,
    selectBestFrame)
from RMS.Astrometry.Conversions import date2JD


FF_NAME = "FF_XX0001_20260101_000000_000_0000000.fits"


def testFrameReferenceTimeIsMiddleOfFF():
    """ The reference time is the middle of the FF (half of 256 frames at 25 fps = 5.12 s after start). """

    ff_dt, jd = frameReferenceTime(FF_NAME, 25.0, ff_frames=256)

    assert ff_dt == datetime.datetime(2026, 1, 1, 0, 0, 5, 120000)
    assert np.isclose(jd, date2JD(2026, 1, 1, 0, 0, 5, 120.0), rtol=0, atol=1e-9)


def testDirectoryReferenceJDFromFFFiles(tmp_path):
    """ Without a CALSTARS file the first FF file in the directory gives the reference time. """

    for name in [FF_NAME.replace("000000_000", "010000_000"), FF_NAME, "notes.txt"]:
        (tmp_path/name).write_text("")

    jd = directoryReferenceJD(str(tmp_path), 25.0)

    assert np.isclose(jd, frameReferenceTime(FF_NAME, 25.0)[1], rtol=0, atol=1e-9)


def testDirectoryReferenceJDEmpty(tmp_path):
    """ No FF file names gives None (J2000 catalog positions). """

    assert directoryReferenceJD(str(tmp_path), 25.0) is None


def _frame(n_stars, seed, width=1280, height=720):
    """ Star data (y, x, IntensSum, Ampltd, FWHM, BgLvl, SNR, NSatPx) spread over the image. """

    rng = np.random.default_rng(seed)
    data = np.zeros((n_stars, 8))
    data[:, 0] = rng.uniform(0, height, n_stars)
    data[:, 1] = rng.uniform(0, width, n_stars)
    data[:, 2] = 1000.0
    data[:, 6] = 20.0

    return data


def _oldCountScore(n_stars, min_stars=10, max_stars=200):
    """ The count score before the penalty scaled with the night median. """

    if n_stars <= max_stars:
        return max(0, min(1, (n_stars - min_stars)/(max_stars - min_stars)))

    return max(0, 1.0 - (n_stars - max_stars)/max_stars)


@pytest.mark.parametrize("n_stars", [5, 50, 150, 200, 250, 350, 500])
def testScoreFrameQualityDefaultUnchanged(n_stars):
    """ Without a penalty knee above max_stars the count score is the original one. """

    _, details = scoreFrameQuality(_frame(n_stars, 0))

    if n_stars >= 10:
        assert np.isclose(details['count_score'], _oldCountScore(n_stars))


def testSelectBestFrameNormalNightUnchanged():
    """ On a night with a median below max_stars/2 the penalty is the same as before. """

    calstars = {"FF_{:d}".format(i): _frame(n, i) for i, n in enumerate([60, 80, 90, 100, 250, 400])}

    _, _, all_scores = selectBestFrame(calstars, 1280, 720)

    for ff_name, star_data in calstars.items():
        count_score = all_scores[ff_name]['quality_details']['count_score']
        assert np.isclose(count_score, _oldCountScore(len(star_data)))


def testSelectBestFrameDeepCameraNotPenalized():
    """ A camera routinely detecting ~400 stars must not prefer a sparse (e.g. cloudy) frame. """

    calstars = {"FF_{:d}".format(i): _frame(n, i) for i, n in enumerate([380, 400, 420, 410, 390, 120])}

    best_ff, _, all_scores = selectBestFrame(calstars, 1280, 720)

    assert len(calstars[best_ff]) > 300
    assert all_scores[best_ff]['quality_details']['count_score'] == 1.0
