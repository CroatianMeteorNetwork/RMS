"""
Tests of the platepar fits of a camera pointing near a celestial pole, and of the random draws of the RANSAC
fit.

The fits step the declination of the pointing freely, so near a pole it goes past 90 deg.
Platepar.normalizeRaDec reflects such a pointing back onto the sphere (the right ascension turned by 180 deg).
Without the refraction the position angle is turned by 180 deg with it; with the refraction the position
angle is measured from the meridian at the refracted pointing, which goes beside the pole, and is kept. Either
way the projected field is continuous over the pole. The fit tests use a synthetic image of a real 1280x720
calibration turned to a pointing 0.3 deg from the north celestial pole, and start each fit from a pointing on
the other side of the pole, so the fit has to step the declination past 90 deg to reach the stars.
"""

import copy
import io
import contextlib

import numpy as np
import pytest

from RMS.Astrometry.ApplyAstrometry import raDecToXYPP
from RMS.Formats.Platepar import normalizeRaDec
from RMS.Math import angularSeparationDeg
from Tests.test_RecalibrateLSQRefine import makeImage, makePlatepar


# Image stars of the synthetic image of the (slower) RANSAC fit, the other fits use the 150 stars of makeImage
N_STARS_RANSAC = 60

# Largest RMS (px) of a fit which found the pointing (the stars have a noise of 0.05 px; a fit stuck at the
#   pole leaves more than 0.3 px)
FIT_RMS_PX = 0.15

# Largest distance (deg) between the fitted and the true pointing (a fit stuck at the pole is 0.3 deg off)
FIT_POINTING_DEG = 0.05


def _unitVector(ra_d, dec_d):
    """ Unit vector of a direction on the sphere.

    Arguments:
        ra_d: [float] Right ascension (deg).
        dec_d: [float] Declination (deg).

    Return:
        [ndarray] The (x, y, z) unit vector.
    """

    ra, dec = np.radians(ra_d), np.radians(dec_d)

    return np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])


def _projectedStars(pp, ra_d, dec_d, pos_angle_ref):
    """ Image positions of a few stars near the north celestial pole, with the given pointing of the platepar.

    Arguments:
        pp: [Platepar] Platepar of the camera (not changed).
        ra_d: [float] Right ascension of the pointing (deg).
        dec_d: [float] Declination of the pointing (deg).
        pos_angle_ref: [float] Position angle (deg).

    Return:
        [ndarray] The (N, 2) image positions (px).
    """

    pp = copy.deepcopy(pp)
    pp.RA_d, pp.dec_d, pp.pos_angle_ref = ra_d, dec_d, pos_angle_ref

    # Stars a couple of degrees from the pole, in the field for any pointing within a degree of it
    star_ra = np.array([30.0, 120.0, 210.0, 300.0])
    star_dec = np.array([88.0, 88.5, 88.2, 89.5])
    x, y = raDecToXYPP(star_ra, star_dec, pp.JD, pp)

    return np.column_stack([x, y])


def _polarPlatepar(refraction=True):
    """ The calibration of makePlatepar turned to a pointing 0.3 deg from the north celestial pole.

    Keyword arguments:
        refraction: [bool] Correct the refraction. True by default.

    Return:
        [Platepar] The platepar.
    """

    pp = makePlatepar()
    pp.refraction = refraction
    pp.RA_d, pp.dec_d, pp.pos_angle_ref = 210.0, 89.7, 220.0
    pp.updateRefAltAz()

    return pp


def _startAcrossThePole(pp):
    """ The platepar with the pointing on the other side of the pole. The true pointing (210, 89.7) with the
        position angle 220 is the pointing (30, 90.3) with the position angle 40, so from (30, 89.9) with 40
        the fit has to step the declination by 0.4 deg, past the pole.

    Arguments:
        pp: [Platepar] Platepar with the true pointing (not changed).

    Return:
        [Platepar] The platepar to start the fit from.
    """

    pp = copy.deepcopy(pp)
    pp.RA_d, pp.dec_d, pp.pos_angle_ref = 30.0, 89.9, 40.0
    pp.updateRefAltAz()

    return pp


def _imageStars(rows):
    """ Image stars of the fits from CALSTARS rows.

    Arguments:
        rows: [ndarray] CALSTARS rows (y, x, intensity, ...).

    Return:
        [ndarray] The (N, 3) image stars (x, y, intensity).
    """

    return np.column_stack([rows[:, 1], rows[:, 0], rows[:, 2]])


def _rmsPx(pp, img_stars, catalog_stars):
    """ RMS distance between the image stars and their catalog stars projected with the platepar.

    Arguments:
        pp: [Platepar] Platepar.
        img_stars: [ndarray] The (N, 3) image stars (x, y, intensity).
        catalog_stars: [ndarray] The (N, 3) catalog stars (ra, dec, mag) of the image stars.

    Return:
        [float] The RMS distance (px).
    """

    x, y = raDecToXYPP(catalog_stars[:, 0], catalog_stars[:, 1], pp.JD, pp)

    return np.sqrt(np.mean((x - img_stars[:, 0])**2 + (y - img_stars[:, 1])**2))


def _checkFit(fitted, truth, img_stars, catalog_stars):
    """ Check that the fit found the true pointing and stored it on the sphere.

    Arguments:
        fitted: [Platepar] The fitted platepar.
        truth: [Platepar] The platepar of the synthetic image.
        img_stars: [ndarray] The (N, 3) image stars (x, y, intensity).
        catalog_stars: [ndarray] The (N, 3) catalog stars (ra, dec, mag) of the image stars.

    Return:
        None
    """

    # The pointing is stored on the sphere
    assert 0.0 <= fitted.RA_d < 360.0
    assert -90.0 <= fitted.dec_d <= 90.0
    assert 0.0 <= fitted.pos_angle_ref < 360.0

    # The stars are fitted down to their noise, with the true pointing
    assert _rmsPx(fitted, img_stars, catalog_stars) < FIT_RMS_PX
    assert angularSeparationDeg(fitted.RA_d, fitted.dec_d, truth.RA_d, truth.dec_d) < FIT_POINTING_DEG


### normalizeRaDec ###

@pytest.mark.parametrize("ra, dec", [(0.0, 0.0), (123.4, 45.6), (359.9, -89.9), (10.0, 89.999), (50.0, 90.0),
    (50.0, -90.0)])
def test_pointing_on_the_sphere_is_kept(ra, dec):

    # A pointing on the sphere (the poles included) is not reflected
    ra_out, dec_out, pos_angle_offset = normalizeRaDec(ra, dec)

    assert ra_out == pytest.approx(ra, abs=1e-12)
    assert dec_out == pytest.approx(dec, abs=1e-12)
    assert pos_angle_offset == 0.0


def test_fit_parameters_away_from_the_poles_decode_as_before():

    # Guard: the fit parameters (RA/360, dec/90, position angle/360) of a pointing on the sphere decode to the
    #   same numbers, bit for bit, as with the modulo wrap used before, so fits that stay away from the poles
    #   do not change
    rng = np.random.default_rng(1)
    params = zip(rng.uniform(-0.5, 1.5, 1000), rng.uniform(-1.0, 1.0, 1000), rng.uniform(-0.5, 1.5, 1000))
    for ra_param, dec_param, pos_angle_param in params:

        ra_d, dec_d, pos_angle_offset = normalizeRaDec(360*ra_param, 90*dec_param)

        assert ra_d == (360*ra_param)%360
        assert dec_d == -90 + (90*dec_param + 90)%180.000001
        assert (360*pos_angle_param + pos_angle_offset)%360 == (360*pos_angle_param)%360


@pytest.mark.parametrize("ra, dec, ra_expected, dec_expected", [
    # Past the north pole, and past the south pole
    (50.0, 95.0, 230.0, 85.0),
    (50.0, -95.0, 230.0, -85.0),
    # Just past the north pole stays next to it (a modulo wrap jumped to -90)
    (50.0, 90.000001, 230.0, 89.999999),
    # The right ascension wraps after the turn by 180 deg
    (300.0, 95.0, 120.0, 85.0),
    # A declination of 180 deg is the opposite point of the meridian
    (0.0, 180.0, 180.0, 0.0),
    # Full turns of the declination
    (0.0, 270.0, 0.0, -90.0),
    (20.0, 400.0, 20.0, 40.0),
])
def test_pointing_past_a_pole_is_reflected(ra, dec, ra_expected, dec_expected):

    # The declination is reflected across the pole and the right ascension turned by 180 deg; without the
    #   refraction the position angle is turned by 180 deg too, with it the position angle is kept
    ra_out, dec_out, pos_angle_offset = normalizeRaDec(ra, dec)

    assert ra_out == pytest.approx(ra_expected, abs=1e-9)
    assert dec_out == pytest.approx(dec_expected, abs=1e-9)

    reflected = abs((ra_out - ra + 180)%360 - 180) > 90
    assert pos_angle_offset == (180.0 if reflected else 0.0)

    assert normalizeRaDec(ra, dec, refraction=True) == (ra_out, dec_out, 0.0)


@pytest.mark.parametrize("ra, dec", [(37.0, 97.5), (-10.0, -91.0), (200.0, 260.0), (5.0, 90.3)])
def test_reflection_keeps_the_direction(ra, dec):

    # The reflected pointing is the same direction on the sky, in the usual ranges
    ra_out, dec_out, _ = normalizeRaDec(ra, dec)

    assert np.allclose(_unitVector(ra, dec), _unitVector(ra_out, dec_out), atol=1e-12)
    assert 0.0 <= ra_out < 360.0
    assert -90.0 <= dec_out <= 90.0


@pytest.mark.parametrize("dec_param", [90.001, 90.3, 92.0, -90.3, -92.0])
def test_reflection_keeps_the_field(dec_param):

    # Without the refraction the projection also takes the unreflected parameters, which gives the field to
    #   compare with: the reflected pointing with the turned position angle puts every star where the
    #   unreflected one does, and without the turn the field is turned by 180 deg
    pp = _polarPlatepar(refraction=False)
    ra_param, pos_angle_param = 30.0, 40.0

    ra_d, dec_d, pos_angle_offset = normalizeRaDec(ra_param, dec_param)

    xy_raw = _projectedStars(pp, ra_param, dec_param, pos_angle_param)
    xy_reflected = _projectedStars(pp, ra_d, dec_d, (pos_angle_param + pos_angle_offset)%360)
    xy_not_turned = _projectedStars(pp, ra_d, dec_d, pos_angle_param)

    assert np.max(np.abs(xy_reflected - xy_raw)) < 1e-6
    assert np.max(np.abs(xy_not_turned - xy_raw)) > 10.0


def test_projection_is_continuous_across_the_pole():

    # Stepping the declination parameter past the pole in even steps moves the stars by even amounts, the step
    #   over the pole included
    pp = _polarPlatepar(refraction=False)

    positions = []
    for dec_param in [89.90, 89.96, 89.99, 90.01, 90.04, 90.10]:
        ra_d, dec_d, pos_angle_offset = normalizeRaDec(30.0, dec_param)
        positions.append(_projectedStars(pp, ra_d, dec_d, (40.0 + pos_angle_offset)%360))

    steps = [np.max(np.abs(positions[i + 1] - positions[i])) for i in range(len(positions) - 1)]
    assert max(steps) < 2.0


@pytest.mark.parametrize("ra_param", [30.0, 120.0, 300.0])
@pytest.mark.parametrize("refraction", [True, False])
def test_step_over_the_pole_is_small(refraction, ra_param):

    # A step of the declination parameter of 2e-5 deg over the pole moves the stars by a small amount, with
    #   and without the refraction. With the refraction the position angle is measured from the meridian at
    #   the refracted pointing, which goes beside the pole: turning the position angle by 180 deg there would
    #   turn the field upside down in the step
    pp = _polarPlatepar(refraction=refraction)
    pos_angle_param, half_step = 40.0, 1e-5

    positions = []
    for dec_param in [90.0 - half_step, 90.0 + half_step]:
        ra_d, dec_d, pos_angle_offset = normalizeRaDec(ra_param, dec_param, refraction=refraction)
        positions.append(_projectedStars(pp, ra_d, dec_d, (pos_angle_param + pos_angle_offset)%360))

    assert np.max(np.abs(positions[1] - positions[0])) < 0.1

    # The other position angle offset would turn the field upside down
    ra_d, dec_d, pos_angle_offset = normalizeRaDec(ra_param, 90.0 + half_step, refraction=refraction)
    other = _projectedStars(pp, ra_d, dec_d, (pos_angle_param + 180.0 - pos_angle_offset)%360)
    assert np.max(np.abs(other - positions[0])) > 10.0


### Fits across the pole ###

def test_radial_fit_crosses_the_pole():

    # The fit of the pointing and the radial distortion on paired stars. Without the refraction: with it, the
    #   position angle is measured from the meridian at the refracted pointing, which turns by about 180 deg
    #   where the refracted pointing goes beside the pole, so from this start the true field is also about
    #   180 deg away in the position angle
    truth = _polarPlatepar(refraction=False)
    catalog_stars, rows = makeImage(truth)
    img_stars = _imageStars(rows)

    fitted = _startAcrossThePole(truth)
    with contextlib.redirect_stdout(io.StringIO()):
        fitted.fitAstrometry(truth.JD, img_stars, catalog_stars)

    _checkFit(fitted, truth, img_stars, catalog_stars)


@pytest.mark.parametrize("refraction", [True, False])
def test_pointing_fit_crosses_the_pole(refraction):

    # The fit of the pointing alone (fitPointing)
    truth = _polarPlatepar(refraction=refraction)
    catalog_stars, rows = makeImage(truth)
    img_stars = _imageStars(rows)

    fitted = _startAcrossThePole(truth)
    fitted.fitAstrometry(truth.JD, img_stars, catalog_stars, fit_only_pointing=True)

    _checkFit(fitted, truth, img_stars, catalog_stars)


def test_nn_pointing_fit_crosses_the_pole():

    # The fit of the pointing on the nearest catalog stars (fitPointingNN, used by the NN alignment), without
    #   the refraction (see test_radial_fit_crosses_the_pole)
    truth = _polarPlatepar(refraction=False)
    catalog_stars, rows = makeImage(truth)
    img_stars = _imageStars(rows)

    fitted = _startAcrossThePole(truth)
    success, _, inlier_fraction, _ = fitted.fitPointingNN(truth.JD, img_stars, catalog_stars)

    assert success
    assert inlier_fraction == 1.0
    _checkFit(fitted, truth, img_stars, catalog_stars)


def test_nn_pointing_fit_goes_around_the_pole_with_the_refraction():

    # The NN pointing fit with the refraction, for a camera 1 deg from the pole, started from a pointing 1.7
    #   deg away around the pole (112 deg of right ascension, with the position angle turned the same, so the
    #   field is only shifted): the fit steps over the pole to reach it
    truth = _polarPlatepar()
    truth.dec_d = 89.0
    truth.updateRefAltAz()
    catalog_stars, rows = makeImage(truth)
    img_stars = _imageStars(rows)

    fitted = copy.deepcopy(truth)
    fitted.RA_d, fitted.pos_angle_ref = truth.RA_d - 112.0, truth.pos_angle_ref - 112.0
    fitted.updateRefAltAz()
    success, _, inlier_fraction, _ = fitted.fitPointingNN(truth.JD, img_stars, catalog_stars)

    assert success
    assert inlier_fraction == 1.0
    _checkFit(fitted, truth, img_stars, catalog_stars)


@pytest.fixture(scope="module")
def ransac_fit():
    """ The RANSAC fit (fitAstrometry with use_nn_cost) of the polar camera without the refraction, started
        across the pole, with the global NumPy random state before and after it.

    Return:
        (truth, fitted, pairs, state_before, state_after): [tuple] The true and the fitted platepar, the
            matched pairs (img_stars, catalog_stars) the fit returned, and the global random state before and
            after it.
    """

    # Without the refraction (see test_radial_fit_crosses_the_pole)
    truth = _polarPlatepar(refraction=False)
    catalog_stars, rows = makeImage(truth, n_stars=N_STARS_RANSAC)
    img_stars = _imageStars(rows)

    # Give the global generator a known state
    np.random.seed(12345)
    state_before = np.random.get_state()

    fitted = _startAcrossThePole(truth)
    with contextlib.redirect_stdout(io.StringIO()):
        pairs = fitted.fitAstrometry(truth.JD, img_stars, catalog_stars, use_nn_cost=True)

    state_after = np.random.get_state()

    return truth, fitted, pairs, state_before, state_after


def test_ransac_fit_crosses_the_pole(ransac_fit):

    # The RANSAC fit finds the stars across the pole, and the final fit on its pairs the pointing
    truth, fitted, pairs, _, _ = ransac_fit

    assert pairs is not None
    img_stars, catalog_stars = pairs
    assert len(img_stars) >= N_STARS_RANSAC - 2

    _checkFit(fitted, truth, img_stars, catalog_stars)


def test_ransac_fit_leaves_the_global_random_state_alone(ransac_fit):

    # The stars of each RANSAC iteration are drawn at random with a fixed seed, which must not reseed the
    #   global generator that every other user of NumPy shares
    _, _, _, state_before, state_after = ransac_fit

    assert state_after[0] == state_before[0]
    assert np.array_equal(state_after[1], state_before[1])
    assert state_after[2:] == state_before[2:]
