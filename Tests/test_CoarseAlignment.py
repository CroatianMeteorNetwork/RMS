"""
Tests of the coarse alignment of a platepar on the stars of an image (RMS.Astrometry.CoarseAlignment), which brings
a pointing that is degrees off or turned by degrees close enough for the fits of the pointing to polish it.

The tests use a synthetic image of a real 1280x720 calibration: catalog stars over the image and around it, the 80
brightest detected with a noise of 0.2 px, and image stars without a catalog star (hot pixels). The starting
platepar is the true one turned by shifts up to 5 deg and rolls up to 20 deg, at the equator, near the north pole
and at the zenith, with and without the refraction. The recalibration of an image and the NN fit start from the
coarse alignment, so they find a pointing degrees off.
"""

import copy

import numpy as np
import pytest

from RMS.Astrometry import ApplyRecalibrate
from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, xyToRaDecPP
from RMS.Astrometry.CoarseAlignment import alignIfBetter, coarseAlignment
from RMS.Formats.Platepar import PointingOffsets
from Tests.test_RecalibrateLSQRefine import displacement, makeConfig, makeImage, makePlatepar


def _scene(dec, refraction=True, zenith=False, n_detected=80, n_hot=10, seed=0):
    """ The true platepar and a synthetic image.

    Arguments:
        dec: [float] Declination of the pointing (deg), ignored with zenith.

    Keyword arguments:
        refraction: [bool] Correct the refraction. True by default.
        zenith: [bool] Point to the zenith. False by default.
        n_detected: [int] Number of detected catalog stars. 80 by default.
        n_hot: [int] Number of image stars without a catalog star. 10 by default.
        seed: [int] Seed of the random numbers. 0 by default.

    Return:
        (pp, img_x, img_y, catalog_stars): [tuple] The true platepar, the image stars and the catalog stars.
    """

    rng = np.random.default_rng(seed)
    pp = makePlatepar()
    pp.refraction = refraction
    if zenith:
        pp.az_centre, pp.alt_centre = 180.0, 90.0
        pp.updateRefRADec(skip_rot_update=True)
    else:
        pp.dec_d = dec
    pp.updateRefAltAz()

    # Catalog stars over the image and around it, the brightest of them detected
    x, y = rng.uniform(-150, pp.X_res + 150, 2000), rng.uniform(-150, pp.Y_res + 150, 2000)
    _, ra, dec, _ = xyToRaDecPP(np.full(len(x), pp.JD), x, y, np.ones(len(x)), pp, extinction_correction=False,
                                jd_time=True)
    catalog_stars = np.column_stack([ra, dec, rng.uniform(1.0, 7.0, len(x))])
    bright = catalog_stars[np.argsort(catalog_stars[:, 2])]
    x_img, y_img = raDecToXYPP(bright[:, 0], bright[:, 1], pp.JD, pp)
    inside = (x_img > 0) & (x_img < pp.X_res) & (y_img > 0) & (y_img < pp.Y_res)
    x_img, y_img = x_img[inside][:n_detected], y_img[inside][:n_detected]
    x_img = x_img + rng.normal(0, 0.2, len(x_img))
    y_img = y_img + rng.normal(0, 0.2, len(y_img))

    # Hot pixels
    x_img = np.append(x_img, rng.uniform(0, pp.X_res, n_hot))
    y_img = np.append(y_img, rng.uniform(0, pp.Y_res, n_hot))

    return pp, x_img, y_img, catalog_stars


def _turned(pp, shift_deg, angle_deg, roll_deg):
    """ A copy of the platepar turned by a shift in the given direction on the image and a roll. """

    pp = copy.deepcopy(pp)
    pointing = PointingOffsets(pp)
    shift = shift_deg*pp.F_scale
    pointing.apply(pp, [shift*np.cos(np.radians(angle_deg)), shift*np.sin(np.radians(angle_deg)),
                        np.radians(roll_deg)/pointing.roll_unit])
    pp.updateRefAltAz()

    return pp


@pytest.mark.parametrize("dec, zenith", [(0.0, False), (89.9, False), (0.0, True)])
@pytest.mark.parametrize("refraction", [True, False])
@pytest.mark.parametrize("shift_deg, angle_deg, roll_deg", [(1.0, 30.0, 0.0), (3.0, 200.0, 10.0),
                                                            (5.0, 120.0, -20.0)])
def test_far_pointing_is_aligned(dec, zenith, refraction, shift_deg, angle_deg, roll_deg):

    # A pointing up to 5 deg off and turned by up to 20 deg is brought within a pixel of the stars, at the equator,
    #   0.1 deg from the pole and at the zenith, and nearly all detected stars agree with it
    pp, img_x, img_y, catalog_stars = _scene(dec, refraction=refraction, zenith=zenith)
    start = _turned(pp, shift_deg, angle_deg, roll_deg)

    aligned = alignIfBetter(start, pp.JD, img_x, img_y, catalog_stars)

    assert aligned is not None
    assert displacement(pp, aligned, catalog_stars[:300]) < 1.0
    _, votes, votes_start = coarseAlignment(start, pp.JD, img_x, img_y, catalog_stars)
    assert votes >= 75
    assert votes_start < 10


def test_good_pointing_is_left_to_the_fits():

    # A pointing that already puts the stars near their catalog stars is not replaced: the fits are more precise
    pp, img_x, img_y, catalog_stars = _scene(30.0)

    assert alignIfBetter(pp, pp.JD, img_x, img_y, catalog_stars) is None
    assert alignIfBetter(_turned(pp, 0.05, 0.0, 0.1), pp.JD, img_x, img_y, catalog_stars) is None


@pytest.mark.parametrize("n_detected, n_hot", [(25, 5), (40, 40)])
def test_sparse_and_outlier_rich_fields(n_detected, n_hot):

    # 25 detected stars, and 40 detected stars with as many hot pixels, from 3 deg and 10 deg of roll off
    pp, img_x, img_y, catalog_stars = _scene(60.0, n_detected=n_detected, n_hot=n_hot)
    start = _turned(pp, 3.0, 300.0, 10.0)

    aligned = alignIfBetter(start, pp.JD, img_x, img_y, catalog_stars)

    assert aligned is not None
    assert displacement(pp, aligned, catalog_stars[:300]) < 1.0


def test_half_clouded_field():

    # Only the stars of the left half of the image are detected
    pp, img_x, img_y, catalog_stars = _scene(45.0)
    left = img_x < pp.X_res/2
    start = _turned(pp, 2.0, 45.0, -10.0)

    aligned = alignIfBetter(start, pp.JD, img_x[left], img_y[left], catalog_stars)

    assert aligned is not None
    assert displacement(pp, aligned, catalog_stars[:300]) < 1.0


def test_no_alignment_without_stars():

    # Too few stars, or stars without a position, give no alignment and leave the platepar as it was
    pp, img_x, img_y, catalog_stars = _scene(30.0)
    start = _turned(pp, 2.0, 0.0, 5.0)
    ra_d = start.RA_d

    assert alignIfBetter(start, pp.JD, img_x[:2], img_y[:2], catalog_stars) is None
    assert alignIfBetter(start, pp.JD, np.full(10, np.nan), img_y[:10], catalog_stars) is None
    assert start.RA_d == ra_d

    # A star without a position among others is skipped
    img_x = img_x.copy()
    img_x[3] = np.nan
    assert alignIfBetter(start, pp.JD, img_x, img_y, catalog_stars) is not None


### The fits from far off ###

def test_recalibration_from_far_off():

    # A platepar 3 deg off and rolled by 10 deg is recalibrated, after the coarse alignment on the stars (the
    #   search alone only reaches its largest match radius)
    pp = makePlatepar()
    catalog_stars, rows = makeImage(pp)
    start = _turned(pp, 3.0, 0.0, 10.0)

    result, _ = ApplyRecalibrate.recalibrateFF(makeConfig(), start, pp.JD, {pp.JD: rows}, catalog_stars)

    assert result is not None
    assert displacement(pp, result, catalog_stars) < 0.05


def test_nn_fit_from_far_off():

    # The NN fit of a pointing 4 deg off and rolled by 15 deg, with 10 hot pixels, finds the true pointing (the
    #   simplex of the fit alone reaches about a degree)
    pp, img_x, img_y, catalog_stars = _scene(50.0)
    fitted = _turned(pp, 4.0, 250.0, 15.0)

    img_stars = np.column_stack([img_x, img_y, np.ones(len(img_x))])

    success, _, inlier_fraction, _ = fitted.fitPointingNN(pp.JD, img_stars, catalog_stars)

    assert success and (inlier_fraction >= 80/90)
    assert displacement(pp, fitted, catalog_stars[:300]) < 0.15


def test_brightest_stars_are_used():

    # Of 300 image stars (80 detected catalog stars and 220 faint hot pixels), the MAX_IMAGE_STARS brightest by
    #   their intensities are used, which are the catalog stars: the platepar is aligned. Without the intensities
    #   the first MAX_IMAGE_STARS (here the hot pixels) are used, and there is no alignment
    pp, img_x, img_y, catalog_stars = _scene(30.0, n_detected=80, n_hot=0)
    rng = np.random.default_rng(7)
    hot_x, hot_y = rng.uniform(0, pp.X_res, 220), rng.uniform(0, pp.Y_res, 220)
    all_x, all_y = np.concatenate([hot_x, img_x]), np.concatenate([hot_y, img_y])
    intensity = np.concatenate([np.ones(220), np.full(len(img_x), 100.0)])
    start = _turned(pp, 2.0, 60.0, 5.0)

    aligned = alignIfBetter(start, pp.JD, all_x, all_y, catalog_stars, img_intensity=intensity)

    assert aligned is not None
    assert displacement(pp, aligned, catalog_stars[:300]) < 1.0
    assert alignIfBetter(start, pp.JD, all_x, all_y, catalog_stars) is None
