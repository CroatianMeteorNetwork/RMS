"""
Tests of the platepar fit on the nearest catalog neighbours of the stars (Platepar.fitAstrometry with use_nn_cost,
the RANSAC fit of AutoPlatepar and of the automatic fit of SkyFit2).

A NN fit that stops without a result must leave the platepar as it was, with no star pairs, and stars or
intensities that cannot be used must not stop the fit with an exception. The tests use a synthetic image: stars of
a 1280x720 calibration at known sky positions (see test_RecalibrateLSQRefine), and a starting platepar with the
pointing off by 0.1 deg in RA and Dec.
"""

import copy

import numpy as np
import pytest

import RMS.Astrometry.ApplyAstrometry
from RMS.Formats import Platepar as PlateparModule
from Tests.test_RecalibrateLSQRefine import displacement, makeImage, makePlatepar


# Number of stars of the synthetic image
N_STARS = 80


def _scene():
    """ The synthetic image and the platepar the NN fit starts from.

    Return:
        (pp, pp_start, catalog_stars, img_stars): [tuple] The true platepar, the starting one, the catalog stars
            (ra, dec, mag) and the image stars (x, y, intensity).
    """

    # The true calibration, with the distortion model of the NN fit of AutoPlatepar
    pp = makePlatepar()
    pp.asymmetry_corr = False
    pp.setDistortionType("radial5-odd", reset_params=False)

    catalog_stars, rows = makeImage(pp, n_stars=N_STARS)
    img_stars = np.column_stack([rows[:, 1], rows[:, 0], rows[:, 2]])

    # The fit starts from a pointing 0.1 deg off, with the star list of an earlier fit
    pp_start = copy.deepcopy(pp)
    pp_start.RA_d += 0.1
    pp_start.dec_d += 0.1
    pp_start.star_list = [[pp.JD, 1.0, 2.0, 3.0, 100.0, 30.0, 4.0]]

    return pp, pp_start, catalog_stars, img_stars


def _fitNN(pp, img_stars, catalog_stars, **kwargs):
    """ The NN fit of the platepar, as AutoPlatepar runs it. """

    return pp.fitAstrometry(pp.JD, img_stars, catalog_stars, first_platepar_fit=True, use_nn_cost=True, **kwargs)


def _assertSameState(pp, state):
    """ The platepar has the given state (__dict__) and an empty star list. """

    assert set(pp.__dict__) == set(state)

    # Every value as it was, but the star list
    for key, value in state.items():
        if key == 'star_list':
            continue

        if isinstance(value, np.ndarray):
            assert np.array_equal(pp.__dict__[key], value), key
        else:
            assert pp.__dict__[key] == value, key

    # No star pairs came out of the fit
    assert pp.star_list == []


def test_accepted_fit_gives_the_pairs():

    pp, pp_start, catalog_stars, img_stars = _scene()

    # A guard of the fit that works: nearly every star matched, the pairs are the star list of the platepar, and
    #   the fitted platepar puts the catalog stars where the true one does
    result = _fitNN(pp_start, img_stars, catalog_stars)

    assert result is not None
    img_matched, catalog_matched = result
    assert len(img_matched) >= N_STARS - 3
    assert len(pp_start.star_list) == len(img_matched)
    assert pp_start.distortion_type == "radial5-odd"
    assert displacement(pp, pp_start, catalog_stars) < 0.5


def test_too_few_stars_leave_the_platepar_as_it_was():

    pp, pp_start, catalog_stars, img_stars = _scene()

    # Three stars are too few for the distortion fit. The star list must not get the stars paired with the catalog
    #   stars of the same index
    state = copy.deepcopy(pp_start.__dict__)
    result = _fitNN(pp_start, img_stars[:3], catalog_stars)

    assert result is None
    _assertSameState(pp_start, state)


def test_rejected_fit_leaves_the_platepar_as_it_was():

    pp, pp_start, catalog_stars, img_stars = _scene()

    # Stars unrelated to the catalog: no pointing fits them, so the RMSD is too large. The scale is held fixed so
    #   the fit cannot shrink the field onto a single catalog star
    rng = np.random.default_rng(1)
    n_det = 60
    img_stars = np.column_stack([rng.uniform(20, pp.X_res - 20, n_det), rng.uniform(20, pp.Y_res - 20, n_det),
                                 np.ones(n_det)])

    state = copy.deepcopy(pp_start.__dict__)
    result = _fitNN(pp_start, img_stars, catalog_stars, fixed_scale=True)

    # The RANSAC switched the distortion type and fitted the pointing on its way, all of it is undone
    assert result is None
    _assertSameState(pp_start, state)


@pytest.mark.parametrize("distortion_type, n_stars, fit_only_pointing", [
    ("poly3+radial", 30, False),
    ("poly3+radial", 60, False),
    ("radial5-odd", 60, True),
])
def test_nn_fit_not_run_without_a_radial_distortion_fit(monkeypatch, distortion_type, n_stars, fit_only_pointing):

    pp, pp_start, catalog_stars, img_stars = _scene()
    pp_start.setDistortionType(distortion_type, reset_params=True)

    # The NN fit only exists for the radial distortion fit. With a polynomial distortion or a fit of the pointing
    #   alone, no fit may run (the stars and the catalog stars are not pairs, and have different numbers)
    def _noFit(*args, **kwargs):
        raise AssertionError("A fit was run in the NN mode without a radial distortion fit")

    monkeypatch.setattr(PlateparModule.scipy.optimize, 'least_squares', _noFit)
    monkeypatch.setattr(PlateparModule.scipy.optimize, 'minimize', _noFit)

    state = copy.deepcopy(pp_start.__dict__)
    result = _fitNN(pp_start, img_stars[:n_stars], catalog_stars, fit_only_pointing=fit_only_pointing)

    assert result is None
    _assertSameState(pp_start, state)


def test_failed_fit_leaves_the_platepar_as_it_was(monkeypatch):

    pp, pp_start, catalog_stars, img_stars = _scene()

    # The optimizer fails in the third RANSAC iteration, after the fit changed the distortion type and the pointing
    minimize = PlateparModule.scipy.optimize.minimize
    calls = []

    def _failingMinimize(*args, **kwargs):

        # The real optimizer, but for the third call
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("optimizer failure")

        return minimize(*args, **kwargs)

    monkeypatch.setattr(PlateparModule.scipy.optimize, 'minimize', _failingMinimize)

    state = copy.deepcopy(pp_start.__dict__)
    with pytest.raises(RuntimeError, match="optimizer failure"):
        _fitNN(pp_start, img_stars, catalog_stars)

    _assertSameState(pp_start, state)


@pytest.mark.parametrize("where", ["iteration", "final"])
def test_no_catalog_star_in_the_field_rejects_the_fit(monkeypatch, where):

    pp, pp_start, catalog_stars, img_stars = _scene()

    # The catalog stars leave the image: after the first fit of the RANSAC (the scoring of its iterations), or with
    #   the final parameters of the RANSAC (the final matching). The platepar itself is only projected at the start
    #   and at the end, the iterations use copies of it
    raDecToXYPP = RMS.Astrometry.ApplyAstrometry.raDecToXYPP
    calls = {'own': 0, 'copy': 0}

    def _hiddenCatalog(ra_data, dec_data, jd, platepar, *args, **kwargs):
        x, y = raDecToXYPP(ra_data, dec_data, jd, platepar, *args, **kwargs)

        # Count the projections with the platepar and with its copies, and move the stars out of the image from
        #   the given one on
        kind = 'own' if platepar is pp_start else 'copy'
        calls[kind] += 1

        if ((where == "iteration") and (kind == 'copy') and (calls['copy'] > 1)) \
            or ((where == "final") and (kind == 'own') and (calls['own'] > 1)):

            x, y = np.full_like(x, -100.0), np.full_like(y, -100.0)

        return x, y

    monkeypatch.setattr(RMS.Astrometry.ApplyAstrometry, 'raDecToXYPP', _hiddenCatalog)

    state = copy.deepcopy(pp_start.__dict__)
    result = _fitNN(pp_start, img_stars, catalog_stars)

    assert result is None
    _assertSameState(pp_start, state)


def test_star_without_a_sky_position_is_not_matched():

    pp, pp_start, catalog_stars, img_stars = _scene()

    # One star with a non-finite position has no sky position with any parameters
    img_stars[5, 0] = np.nan

    result = _fitNN(pp_start, img_stars, catalog_stars)

    # The fit goes through, and the star is not in the pairs
    assert result is not None
    img_matched, catalog_matched = result
    assert np.all(np.isfinite(img_matched))
    assert len(img_matched) >= N_STARS - 4
    assert displacement(pp, pp_start, catalog_stars) < 0.5


@pytest.mark.parametrize("intensity", [0.0, np.nan])
def test_unusable_intensities_select_the_stars_uniformly(intensity):

    pp, pp_start, catalog_stars, img_stars = _scene()

    # Intensities which cannot weigh the selection of the stars of the RANSAC (a median <= 0, or NaN)
    img_stars[:, 2] = intensity

    result = _fitNN(pp_start, img_stars, catalog_stars)

    assert result is not None
    assert len(result[0]) >= N_STARS - 3
    assert displacement(pp, pp_start, catalog_stars) < 0.5
