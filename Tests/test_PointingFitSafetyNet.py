"""
Tests of the order of the searches of the pointing, and of the scale bound of the calibration check.

Every search of the pointing runs first as earlier versions did (RA, Dec and position angle), so a fit which
succeeded before gives the same result at the same cost. If its result is not clearly good by the fit's own
measures, the search of the offsets of the pointing (Platepar.PointingOffsets, from the coarse alignment of the
stars) runs as well, from the same start, and its result is used if it is better: the recalibration of an image
(ApplyRecalibrate.recalibrateFF) and the calibration check (CheckFit.autoCheckFit) when they fail, the NN fit
(Platepar.fitPointingNN) when it is not accepted (the fit of the offsets only if its inliers are within
NN_OFFSETS_MAX_RMSD_PX), and the RANSAC fit (Platepar.fitAstrometry with use_nn_cost) when it matches too few stars
or leaves large residuals. A fit of the check that changes the scale by more than 10% or matches too few stars is
not used, so the check of a platepar far off fails cleanly.
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

def _recordSearches(monkeypatch, fail_earlier):
    """ Record the searches run by recalibrateFF, with the search of earlier versions failing if asked.

    Return:
        [list] The search classes, in the order they are run.
    """

    searches = []
    search = ApplyRecalibrate._recalibrateSearch

    def _search(*args, **kwargs):
        """ The recalibration with the given search, recorded. """

        searches.append(kwargs['search_class'])
        if fail_earlier and (kwargs['search_class'] is CheckFit.SearchRaDec):
            return None, None

        return search(*args, **kwargs)

    monkeypatch.setattr(ApplyRecalibrate, '_recalibrateSearch', _search)

    return searches


def test_recalibration_falls_back_to_the_search_of_the_offsets(monkeypatch):

    # The recalibration with the search of RA, Dec and rotation fails, so the search of the offsets runs and its
    #   result is used
    pp = makePlatepar()
    catalog_stars, rows = makeImage(pp)
    searches = _recordSearches(monkeypatch, fail_earlier=True)

    result, _ = ApplyRecalibrate.recalibrateFF(makeConfig(), perturbed(pp), pp.JD, {pp.JD: rows}, catalog_stars)

    assert searches == [CheckFit.SearchRaDec, CheckFit.SearchOffsets]
    assert result is not None
    assert displacement(pp, result, catalog_stars) < 0.05


def test_recalibration_of_earlier_versions_needs_no_fallback(monkeypatch):

    # A successful recalibration with the search of earlier versions is used as it is
    pp = makePlatepar()
    catalog_stars, rows = makeImage(pp)
    searches = _recordSearches(monkeypatch, fail_earlier=False)

    result, _ = ApplyRecalibrate.recalibrateFF(makeConfig(), perturbed(pp), pp.JD, {pp.JD: rows}, catalog_stars)

    assert searches == [CheckFit.SearchRaDec]
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

    # A declination stepped past the pole is reflected back onto the sphere (prerelease stored it as it was)
    search.apply(fitted, [341.0, 90.5, 6.0, 15.0])
    assert -90.0 <= fitted.dec_d <= 90.0


### Calibration check ###

def test_check_falls_back_to_the_search_of_the_offsets(monkeypatch):

    # The check with the search of RA, Dec and rotation of a platepar 1 deg off fails, so the check with the offsets
    #   runs from the initial platepar (the coarse alignment finds a better pointing), with the same random draws of
    #   the images, and its result is used
    pp = test_CheckFit.makePlatepar()
    config = test_CheckFit.makeConfig()
    catalog_stars, calstars_data = test_CheckFit.makeNight(pp, config)
    test_CheckFit.useCatalog(monkeypatch, catalog_stars)

    runs = []
    check = CheckFit._autoCheckFit

    def _check(config, platepar, calstars_data, _nn_refinement, search_class):
        """ The check with the given search, recorded; the one of earlier versions fails after changing the
            platepar. """

        runs.append((search_class, test_CheckFit.pointing(platepar), random.random()))
        if search_class is CheckFit.SearchRaDec:
            platepar.RA_d += 1.0
            return platepar, False

        return check(config, platepar, calstars_data, _nn_refinement, search_class)

    monkeypatch.setattr(CheckFit, '_autoCheckFit', _check)

    start = _turned(pp, 1.0*pp.F_scale, roll_deg=3.0)
    result, fit_status = CheckFit.autoCheckFit(config, copy.deepcopy(start), calstars_data)

    assert [run[0] for run in runs] == [CheckFit.SearchRaDec, CheckFit.SearchOffsets]
    assert runs[0][1] == runs[1][1] == test_CheckFit.pointing(start)
    assert runs[0][2] == runs[1][2]
    assert fit_status
    assert displacement(pp, result, catalog_stars) < 0.5


def test_check_with_the_offsets_needs_a_better_pointing(monkeypatch):

    # The check with the search of RA, Dec and rotation of a platepar 2 px off fails: the coarse alignment finds no
    #   better pointing, so the check with the offsets is not run and the failed check of earlier versions is handed
    #   over
    pp = test_CheckFit.makePlatepar()
    config = test_CheckFit.makeConfig()
    catalog_stars, calstars_data = test_CheckFit.makeNight(pp, config)
    test_CheckFit.useCatalog(monkeypatch, catalog_stars)

    runs = []
    check = CheckFit._autoCheckFit

    def _check(config, platepar, calstars_data, _nn_refinement, search_class):
        """ The check with the given search, recorded; the one of earlier versions fails. """

        if search_class is CheckFit.SearchRaDec:
            runs.append('earlier')
            return platepar, False

        result = check(config, platepar, calstars_data, _nn_refinement, search_class)
        runs.append(result[0] is None)

        return result

    monkeypatch.setattr(CheckFit, '_autoCheckFit', _check)

    start = test_CheckFit.offsetPlatepar(pp)
    result, fit_status = CheckFit.autoCheckFit(config, copy.deepcopy(start), calstars_data)

    assert runs == ['earlier', True]
    assert not fit_status
    assert test_CheckFit.pointing(result) == test_CheckFit.pointing(start)


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

def _recordNNFits(monkeypatch, fail_earlier, rmsd_offsets=None):
    """ Record the NN fits, with the fit of earlier versions not accepted if asked, and the inlier RMSD of the fit
        of the offsets replaced if given.

    Return:
        [list] The legacy flags of the fits, in the order they are run.
    """

    calls = []
    fit = PlateparModule.Platepar._fitPointingNNOnce

    def _fit(self, jd, img_stars, catalog_stars, fixed_scale, legacy=False):
        """ The NN fit, recorded. """

        calls.append(legacy)
        result = fit(self, jd, img_stars, catalog_stars, fixed_scale, legacy=legacy)
        if legacy and fail_earlier:
            return (False,) + result[1:]
        if (not legacy) and (rmsd_offsets is not None):
            return result[:4] + (rmsd_offsets,) + result[5:]

        return result

    monkeypatch.setattr(PlateparModule.Platepar, '_fitPointingNNOnce', _fit)

    return calls


def test_nn_fit_falls_back_to_the_fit_of_the_offsets(monkeypatch):

    # The NN fit of RA, Dec and position angle is not accepted, so the NN fit of the offsets runs and its result is
    #   used
    pp = makePlatepar()
    catalog_stars, rows = makeImage(pp)
    img_stars = np.column_stack([rows[:, 1], rows[:, 0], rows[:, 2]])
    calls = _recordNNFits(monkeypatch, fail_earlier=True)

    fitted = _turned(pp, 5.0)
    success, _, inlier_fraction, _ = fitted.fitPointingNN(pp.JD, img_stars, catalog_stars)

    assert calls == [True, False]
    assert success and (inlier_fraction == 1.0)
    assert displacement(pp, fitted, catalog_stars) < 0.05


def test_nn_fit_of_the_offsets_far_from_the_stars_is_not_used(monkeypatch):

    # A fit of the offsets whose inliers are further than NN_OFFSETS_MAX_RMSD_PX from their catalog stars (a
    #   pointing which puts the stars near other catalog stars) is not used: the fit fails as it did before
    pp = makePlatepar()
    catalog_stars, rows = makeImage(pp)
    img_stars = np.column_stack([rows[:, 1], rows[:, 0], rows[:, 2]])
    calls = _recordNNFits(monkeypatch, fail_earlier=True, rmsd_offsets=2.5)

    fitted = _turned(pp, 5.0)
    ra_start = fitted.RA_d
    success, _, _, _ = fitted.fitPointingNN(pp.JD, img_stars, catalog_stars)

    assert calls == [True, False]
    assert not success
    assert fitted.RA_d == ra_start


def test_nn_fit_of_earlier_versions_needs_no_fallback(monkeypatch):

    # An accepted NN fit of earlier versions is used as it is
    pp = makePlatepar()
    catalog_stars, rows = makeImage(pp)
    img_stars = np.column_stack([rows[:, 1], rows[:, 0], rows[:, 2]])
    calls = _recordNNFits(monkeypatch, fail_earlier=False)

    fitted = _turned(pp, 5.0)
    success, _, _, _ = fitted.fitPointingNN(pp.JD, img_stars, catalog_stars)

    assert calls == [True]
    assert success


def test_ransac_falls_back_and_keeps_the_better_result(monkeypatch):

    # The RANSAC fit of earlier versions returns no pairs, so the RANSAC fit of the offsets runs from the platepar
    #   as it was given, and its pairs are used
    pp = makePlatepar()
    catalog_stars, rows = makeImage(pp, n_stars=60)
    img_stars = np.column_stack([rows[:, 1], rows[:, 0], rows[:, 2]])

    calls = []
    fit = PlateparModule.Platepar._fitAstrometry

    def _fit(self, *args, **kwargs):
        """ The fit, recorded; the RANSAC fit of earlier versions moves the platepar and returns no pairs. """

        legacy = kwargs.get('legacy', False)
        if kwargs.get('use_nn_cost'):
            calls.append((legacy, self.RA_d))
            if legacy:
                self.RA_d += 3.0
                return None

        return fit(self, *args, **kwargs)

    monkeypatch.setattr(PlateparModule.Platepar, '_fitAstrometry', _fit)

    start = perturbed(pp)
    fitted = copy.deepcopy(start)
    pairs = fitted.fitAstrometry(pp.JD, img_stars, catalog_stars, use_nn_cost=True)

    assert [call[0] for call in calls] == [True, False]
    assert calls[1][1] == start.RA_d
    assert (pairs is not None) and (len(pairs[0]) >= 58)
    assert displacement(pp, fitted, catalog_stars) < 0.1


def test_good_ransac_needs_no_fallback(monkeypatch):

    # A RANSAC fit of earlier versions which matches nearly all stars with small residuals is used as it is
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

    assert calls == [True]
    assert PlateparModule.ransacResultGood(fitted, pp.JD, img_stars, pairs)


def test_nn_fit_of_the_offsets_needs_a_better_pointing(monkeypatch):

    # The NN fit of RA, Dec and position angle of a platepar 2 px off is not accepted: the coarse alignment finds no
    #   better pointing, so the fit of the offsets is not run and the fit fails as before, with the platepar as it was
    pp = makePlatepar()
    catalog_stars, rows = makeImage(pp)
    img_stars = np.column_stack([rows[:, 1], rows[:, 0], rows[:, 2]])
    _recordNNFits(monkeypatch, fail_earlier=True)

    start = _turned(pp, 2.0)
    accepted, success, _, _, _, _ = start._fitPointingNNOnce(pp.JD, img_stars, catalog_stars, True)
    fitted = copy.deepcopy(start)
    fit_success, _, _, _ = fitted.fitPointingNN(pp.JD, img_stars, catalog_stars)

    assert not (accepted or success)
    assert not fit_success
    assert test_CheckFit.pointing(fitted) == test_CheckFit.pointing(start)


### Retries of the recalibration of an FF ###

def _recordRecalibrations(monkeypatch, aligned_differs):
    """ Make every recalibration of an FF fail and record the platepars they start from; the NN alignment returns
        the platepar it is given, or a moved copy if asked.

    Return:
        [list] The JSON of the starting platepars, in the order of the recalibrations.
    """

    starts = []

    def _recalibrate(config, platepar, *args, **kwargs):
        """ A failed recalibration, recorded. """

        starts.append(platepar.jsonStr())
        return None, None

    def _align(config, platepar, *args, **kwargs):
        """ The NN alignment, which fails (or moves the platepar if asked). """

        if aligned_differs:
            platepar = copy.deepcopy(platepar)
            platepar.RA_d += 0.1
        return platepar, None

    monkeypatch.setattr(ApplyRecalibrate, 'recalibrateFF', _recalibrate)
    monkeypatch.setattr(ApplyRecalibrate, 'alignPlatepar', _align)

    return starts


def test_failed_recalibration_is_not_repeated_from_the_same_platepar(monkeypatch):

    # On an FF which no recalibration matches, the retry from the original platepar is the recalibration just run
    #   after the failed NN alignment (same platepar, same arguments), so it is not run again; the result is the same
    pp = makePlatepar()
    _, rows = makeImage(pp)
    ff_name = 'FF_XX0001_20251214_014211_000_0000000.fits'
    starts = _recordRecalibrations(monkeypatch, aligned_differs=False)

    platepars = ApplyRecalibrate.recalibratePlateparsForFF(pp, [ff_name], {ff_name: rows}, np.zeros((10, 3)),
                                                          makeConfig())

    assert len(starts) == 2
    assert starts[0] == starts[1] == pp.jsonStr()
    assert not platepars[ff_name].auto_recalibrated


def test_failed_recalibration_is_retried_from_another_platepar(monkeypatch):

    # If the NN alignment moved the platepar, the retry from the original platepar is run as before
    pp = makePlatepar()
    _, rows = makeImage(pp)
    ff_name = 'FF_XX0001_20251214_014211_000_0000000.fits'
    starts = _recordRecalibrations(monkeypatch, aligned_differs=True)

    ApplyRecalibrate.recalibratePlateparsForFF(pp, [ff_name], {ff_name: rows}, np.zeros((10, 3)), makeConfig())

    assert len(starts) == 3
    assert starts[0] == starts[2] == pp.jsonStr()
    assert starts[1] != pp.jsonStr()
