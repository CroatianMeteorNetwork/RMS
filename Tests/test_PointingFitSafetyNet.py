"""
Tests of the safety net of the fits of the pointing, and of the scale bound of the calibration check.

Every search of the pointing runs first with the offsets of the pointing (Platepar.PointingOffsets). If its result
is not clearly good by the fit's own measures, the search of earlier versions (RA, Dec and position angle) runs as
well, from the same start, and its result is used if it is better: the recalibration of an image
(ApplyRecalibrate.recalibrateFF) and the calibration check (CheckFit.autoCheckFit) when they fail, the NN fit
(Platepar.fitPointingNN) when it is not accepted, and the RANSAC fit (Platepar.fitAstrometry with use_nn_cost) when
it matches too few stars or leaves large residuals. A fit of the check that changes the scale by more than 10% or
matches too few stars is not used, so the check of a platepar far off fails cleanly.
"""

import copy
import random

import numpy as np

from RMS.Astrometry import ApplyRecalibrate, CheckFit
from RMS.Formats import Platepar as PlateparModule
from RMS.Formats.Platepar import PointingOffsets
from Tests import test_CheckFit
from Tests.test_RecalibrateLSQRefine import displacement, makeConfig, makeImage, makePlatepar, perturbed


def _turned(pp, shift_px, roll_deg=0.0):
    """ A copy of the platepar with the pointing shifted along x and rolled. """

    pp = copy.deepcopy(pp)
    pointing = PointingOffsets(pp)
    pointing.apply(pp, [shift_px, 0.0, np.radians(roll_deg)/pointing.roll_unit])
    pp.updateRefAltAz()

    return pp


### Recalibration of an image ###

def _recordSearches(monkeypatch, fail_offsets):
    """ Record the searches run by recalibrateFF, with the search of the offsets failing if asked.

    Return:
        [list] The search classes, in the order they are run.
    """

    searches = []
    search = ApplyRecalibrate._recalibrateSearch

    def _search(*args, **kwargs):
        """ The recalibration with the given search, recorded. """

        searches.append(kwargs['search_class'])
        if fail_offsets and (kwargs['search_class'] is CheckFit.SearchOffsets):
            return None, None

        return search(*args, **kwargs)

    monkeypatch.setattr(ApplyRecalibrate, '_recalibrateSearch', _search)

    return searches


def test_recalibration_falls_back_to_the_search_of_earlier_versions(monkeypatch):

    # The recalibration with the offsets fails, so the search of RA, Dec and rotation runs and its result is used
    pp = makePlatepar()
    catalog_stars, rows = makeImage(pp)
    searches = _recordSearches(monkeypatch, fail_offsets=True)

    result, _ = ApplyRecalibrate.recalibrateFF(makeConfig(), perturbed(pp), pp.JD, {pp.JD: rows}, catalog_stars)

    assert searches == [CheckFit.SearchOffsets, CheckFit.SearchRaDec]
    assert result is not None
    assert displacement(pp, result, catalog_stars) < 0.05


def test_recalibration_with_the_offsets_needs_no_fallback(monkeypatch):

    # A successful recalibration with the offsets is used as it is
    pp = makePlatepar()
    catalog_stars, rows = makeImage(pp)
    searches = _recordSearches(monkeypatch, fail_offsets=False)

    result, _ = ApplyRecalibrate.recalibrateFF(makeConfig(), perturbed(pp), pp.JD, {pp.JD: rows}, catalog_stars)

    assert searches == [CheckFit.SearchOffsets]
    assert result is not None


def test_search_of_earlier_versions():

    # Guard: the search of the safety net has the parameters, the first simplex and the tolerance of the search of
    #   earlier versions (RA, Dec, position angle and scale, 5% of each value but at least 0.5 deg on the sky)
    pp = makePlatepar()
    pp.RA_d, pp.dec_d, pp.pos_angle_ref = 340.0, 60.0, 5.0
    config = makeConfig()
    search = CheckFit.SearchRaDec(config, pp)

    assert search.p0 == [340.0, 60.0, 5.0, pp.F_scale]
    assert np.allclose(search.steps, [17.0, 3.0, 0.5, 0.05*pp.F_scale])
    assert search.xatol == config.dist_check_threshold/pp.F_scale

    fitted = copy.deepcopy(pp)
    search.apply(fitted, [341.0, 61.0, 6.0, 15.0])
    assert (fitted.RA_d, fitted.dec_d, fitted.pos_angle_ref, fitted.F_scale) == (341.0, 61.0, 6.0, 15.0)


### Calibration check ###

def test_check_falls_back_to_the_search_of_earlier_versions(monkeypatch):

    # The check with the offsets fails, so the check with the search of RA, Dec and rotation runs from the initial
    #   platepar, with the same random draws of the images, and its result is used
    pp = test_CheckFit.makePlatepar()
    config = test_CheckFit.makeConfig()
    catalog_stars, calstars_data = test_CheckFit.makeNight(pp, config)
    test_CheckFit.useCatalog(monkeypatch, catalog_stars)

    runs = []
    check = CheckFit._autoCheckFit

    def _check(config, platepar, calstars_data, _nn_refinement, search_class):
        """ The check with the given search, recorded; the one with the offsets fails after changing the platepar.
        """

        runs.append((search_class, test_CheckFit.pointing(platepar), random.random()))
        if search_class is CheckFit.SearchOffsets:
            platepar.RA_d += 1.0
            return platepar, False

        return check(config, platepar, calstars_data, _nn_refinement, search_class)

    monkeypatch.setattr(CheckFit, '_autoCheckFit', _check)

    start = test_CheckFit.offsetPlatepar(pp)
    result, fit_status = CheckFit.autoCheckFit(config, copy.deepcopy(start), calstars_data)

    assert [run[0] for run in runs] == [CheckFit.SearchOffsets, CheckFit.SearchRaDec]
    assert runs[0][1] == runs[1][1] == test_CheckFit.pointing(start)
    assert runs[0][2] == runs[1][2]
    assert fit_status
    assert displacement(pp, result, catalog_stars) < 0.5


def test_check_of_a_platepar_far_off_fails_cleanly(monkeypatch):

    # A platepar pointing 90 deg away matches no stars: the check fails and hands over a platepar with the scale it
    #   was given (a fit that changes the scale by more than 10% or matches too few stars is not used)
    pp = test_CheckFit.makePlatepar()
    config = test_CheckFit.makeConfig()
    catalog_stars, calstars_data = test_CheckFit.makeNight(pp, config)
    test_CheckFit.useCatalog(monkeypatch, catalog_stars)

    start = _turned(pp, 90.0*pp.F_scale)
    result, fit_status = CheckFit.autoCheckFit(config, copy.deepcopy(start), calstars_data)

    assert not fit_status
    assert abs(result.F_scale/start.F_scale - 1) <= CheckFit.SEARCH_MAX_SCALE_CHANGE


### NN fit and RANSAC fit ###

def test_nn_fit_falls_back_to_the_fit_of_earlier_versions(monkeypatch):

    # The NN fit with the offsets is not accepted, so the NN fit of RA, Dec and position angle runs and its result
    #   is used
    pp = makePlatepar()
    catalog_stars, rows = makeImage(pp)
    img_stars = np.column_stack([rows[:, 1], rows[:, 0], rows[:, 2]])

    calls = []
    fit = PlateparModule.Platepar._fitPointingNNOnce

    def _fit(self, jd, img_stars, catalog_stars, fixed_scale, legacy=False):
        """ The NN fit, recorded; the one with the offsets is not accepted. """

        calls.append(legacy)
        result = fit(self, jd, img_stars, catalog_stars, fixed_scale, legacy=legacy)
        if not legacy:
            return (False,) + result[1:]

        return result

    monkeypatch.setattr(PlateparModule.Platepar, '_fitPointingNNOnce', _fit)

    fitted = _turned(pp, 5.0)
    success, _, inlier_fraction, _ = fitted.fitPointingNN(pp.JD, img_stars, catalog_stars)

    assert calls == [False, True]
    assert success and (inlier_fraction == 1.0)
    assert displacement(pp, fitted, catalog_stars) < 0.5


def test_ransac_falls_back_and_keeps_the_better_result(monkeypatch):

    # The RANSAC fit with the offsets returns no pairs, so the RANSAC fit of earlier versions runs from the
    #   platepar as it was given, and its pairs are used
    pp = makePlatepar()
    catalog_stars, rows = makeImage(pp, n_stars=60)
    img_stars = np.column_stack([rows[:, 1], rows[:, 0], rows[:, 2]])

    calls = []
    fit = PlateparModule.Platepar._fitAstrometry

    def _fit(self, *args, **kwargs):
        """ The fit, recorded; the RANSAC fit with the offsets moves the platepar and returns no pairs. """

        legacy = kwargs.get('legacy', False)
        if kwargs.get('use_nn_cost'):
            calls.append((legacy, self.RA_d))
            if not legacy:
                self.RA_d += 3.0
                return None

        return fit(self, *args, **kwargs)

    monkeypatch.setattr(PlateparModule.Platepar, '_fitAstrometry', _fit)

    start = perturbed(pp)
    fitted = copy.deepcopy(start)
    pairs = fitted.fitAstrometry(pp.JD, img_stars, catalog_stars, use_nn_cost=True)

    assert [call[0] for call in calls] == [False, True]
    assert calls[1][1] == start.RA_d
    assert (pairs is not None) and (len(pairs[0]) >= 58)
    assert displacement(pp, fitted, catalog_stars) < 0.1


def test_good_ransac_needs_no_fallback(monkeypatch):

    # A RANSAC fit with the offsets which matches nearly all stars with small residuals is used as it is
    pp = makePlatepar()
    catalog_stars, rows = makeImage(pp, n_stars=60)
    img_stars = np.column_stack([rows[:, 1], rows[:, 0], rows[:, 2]])

    calls = []
    fit = PlateparModule.Platepar._fitAstrometry

    def _fit(self, *args, **kwargs):
        """ The fit, recorded. """

        if kwargs.get('use_nn_cost'):
            calls.append(kwargs.get('legacy', False))

        return fit(self, *args, **kwargs)

    monkeypatch.setattr(PlateparModule.Platepar, '_fitAstrometry', _fit)

    fitted = perturbed(pp)
    pairs = fitted.fitAstrometry(pp.JD, img_stars, catalog_stars, use_nn_cost=True)

    assert calls == [False]
    assert PlateparModule.ransacResultGood(fitted, pp.JD, img_stars, pairs)
