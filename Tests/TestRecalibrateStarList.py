""" Regression tests for the column order of platepar.star_list written by the automated recalibration.

star_list entries must be [jd, x, y, intensity, ra, dec, mag] regardless of which code path wrote them.
Platepar.fitAstrometry (SkyFit2, AutoPlatepar) takes (x, y, intensity) image stars, while recalibrateFF
works on CALSTARS rows which are ordered (y, x, intensity, ...), so the latter has to reorder them.
"""

import copy
import json

import pytest

np = pytest.importorskip("numpy")

import RMS.ConfigReader as cr
from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, xyToRaDecPP
from RMS.Astrometry.ApplyRecalibrate import recalibrateFF
from RMS.Formats.Platepar import Platepar


# Julian date of the synthetic image (2023-02-28 ~18 UT)
JD_IMG = 2460004.25

N_STARS = 40


def _makePlatepar():
    """ Build a synthetic 1280x720 platepar pointed about 60 deg up, with no distortion. """

    pp = Platepar()
    pp.lat = 45.0
    pp.lon = 15.0
    pp.elev = 100.0
    pp.JD = JD_IMG
    pp.F_scale = 15.7
    pp.az_centre = 180.0
    pp.alt_centre = 60.0
    pp.updateRefRADec(skip_rot_update=True)
    pp.updateRefAltAz()

    return pp


def _makeStars(pp):
    """ Generate synthetic image stars and the matching catalog stars.

    Arguments:
        pp: [Platepar] Platepar used to project the image stars to the sky.

    Return:
        (img_xy, calstars_rows, catalog_stars):
            - img_xy: [ndarray] (N, 2) true (x, y) image positions.
            - calstars_rows: [ndarray] (N, 8) CALSTARS-style rows
                (y, x, intensity, amplitude, fwhm, background, snr, saturated).
            - catalog_stars: [ndarray] (N, 3) catalog (ra, dec, mag) in degrees.
    """

    rng = np.random.RandomState(0)

    # Draw positions away from the image edges, and reject stars too close to the x = y line so a swap
    #   of the coordinates is always detectable
    img_xy = []
    while len(img_xy) < N_STARS:
        x = rng.uniform(50, pp.X_res - 50)
        y = rng.uniform(50, pp.Y_res - 50)
        if abs(x - y) > 20:
            img_xy.append((x, y))

    img_xy = np.array(img_xy)
    x_arr, y_arr = img_xy.T

    _, ra_arr, dec_arr, _ = xyToRaDecPP(np.full(N_STARS, JD_IMG), x_arr, y_arr, np.ones(N_STARS), pp,
        extinction_correction=False, jd_time=True)

    mag_arr = rng.uniform(2.0, 5.0, N_STARS)
    catalog_stars = np.column_stack([ra_arr, dec_arr, mag_arr]).astype(np.float64)

    # Brighter stars get a larger summed intensity
    intens_arr = 10**(-0.4*(mag_arr - 12.0))

    calstars_rows = np.column_stack([
        y_arr, x_arr, intens_arr,
        np.full(N_STARS, 100.0),  # amplitude
        np.full(N_STARS, 2.5),    # fwhm
        np.full(N_STARS, 20.0),   # background
        np.full(N_STARS, 30.0),   # snr
        np.zeros(N_STARS),        # saturated count
    ]).astype(np.float64)

    return img_xy, calstars_rows, catalog_stars


@pytest.fixture(scope="module")
def recalibrated():
    """ Run recalibrateFF on synthetic data, starting from a slightly wrong pointing. """

    pp_true = _makePlatepar()
    img_xy, calstars_rows, catalog_stars = _makeStars(pp_true)

    config = cr.Config()
    config.width = pp_true.X_res
    config.height = pp_true.Y_res

    # Offset the pointing by ~0.5 px so the recalibration has to run a fit
    pp_start = copy.deepcopy(pp_true)
    pp_start.RA_d += 0.03
    pp_start.updateRefAltAz()

    result, _ = recalibrateFF(config, pp_start, JD_IMG, {JD_IMG: calstars_rows}, catalog_stars)

    assert result is not None, "recalibrateFF failed on noise-free synthetic data"
    assert result.auto_recalibrated

    return result, img_xy, calstars_rows, catalog_stars


def test_recalibrated_star_list_is_x_first(recalibrated):
    """ Columns 1 and 2 of a recalibrated star_list are the image x and y of each star. """

    result, img_xy, _, _ = recalibrated

    star_list = np.array(result.star_list)
    assert star_list.shape[1] == 7
    assert len(star_list) >= cr.Config().min_matched_stars

    # Every stored (x, y) pair must be one of the input (x, y) positions
    for entry in star_list:
        dist = np.hypot(img_xy[:, 0] - entry[1], img_xy[:, 1] - entry[2])
        assert np.min(dist) < 1e-6, "star_list entry {} is not an input (x, y) pair".format(entry[1:3])

    # Projecting the stored catalog RA/Dec with the recalibrated platepar must land on the stored x, y
    x_cat, y_cat = raDecToXYPP(star_list[:, 4], star_list[:, 5], JD_IMG, result)
    assert np.all(np.hypot(x_cat - star_list[:, 1], y_cat - star_list[:, 2]) < 1.0)


def test_recalibrated_matches_fitastrometry_format(recalibrated):
    """ recalibrateFF and Platepar.fitAstrometry store the same star with the same columns. """

    result, img_xy, calstars_rows, catalog_stars = recalibrated

    # Image stars in the (x, y, intensity) form SkyFit2 passes to fitAstrometry
    img_stars = np.column_stack([calstars_rows[:, 1], calstars_rows[:, 0], calstars_rows[:, 2]])

    pp_fit = _makePlatepar()
    pp_fit.fitAstrometry(JD_IMG, img_stars, catalog_stars, fit_only_pointing=True)

    fit_list = np.array(pp_fit.star_list)
    recal_list = np.array(result.star_list)

    # Pair the entries of the two lists through the catalog RA/Dec of each star
    for entry in recal_list:
        idx = np.argmin(np.hypot(fit_list[:, 4] - entry[4], fit_list[:, 5] - entry[5]))
        assert np.allclose(fit_list[idx, 1:4], entry[1:4])


def test_star_list_survives_json(recalibrated):
    """ The JSON written to platepars_all_recalibrated.json keeps the x, y order. """

    result, img_xy, _, _ = recalibrated

    star_list = np.array(json.loads(result.jsonStr())['star_list'])

    assert np.allclose(star_list, np.array(result.star_list))

    for entry in star_list:
        dist = np.hypot(img_xy[:, 0] - entry[1], img_xy[:, 1] - entry[2])
        assert np.min(dist) < 1e-6
