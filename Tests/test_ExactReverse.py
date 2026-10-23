""" Tests that raDecToXYPP with exact_reverse is the exact inverse of xyToRaDecPP.

For every distortion type and every combination of its flags (equal aspect, asymmetry, forced centre and, for the
radial models, decentering), a platepar with a few pixels of distortion takes a grid of image points to the sky and
back. Without refraction the round trip has to close to 1e-9 px. With refraction it can only close if the
sky -> image refraction is the exact inverse of the image -> sky one; where it is not, that test is skipped.

Also tested: a sky point the forward distortion cannot reach comes back as NaN, and fitAstrometry with the option
returns the platepar that generated noise-free stars, through its forward mapping.

Run with pytest:
    python -m pytest Tests/test_ExactReverse.py -v
"""

from __future__ import print_function, division, absolute_import

import copy
import itertools

import numpy as np
import pytest

import RMS.Astrometry.ApplyAstrometry as astrometry
from RMS.Astrometry.Conversions import JD2HourAngle, JD2LST
from RMS.Astrometry.CyFunctions import pyRefractionApparentToTrue, pyRefractionTrueToApparent
from RMS.Formats.Platepar import Platepar


JD = 2461070.9
TOLERANCE_PX = 1e-9

# A few pixels of distortion in each term, all on the rising branch of the radial models over the whole frame
RADIAL_K = {
    "odd": [-0.08, 0.01, -0.002, 0.0004],
    "all": [-0.02, -0.06, 0.008, -0.001],
}
POLY_X = [0.5, 1e-3, 5e-4, 2e-6, 1e-6, -1e-6, 1e-9, 2e-9, -1e-9, 1e-9, -1e-5, 2e-6, -1e-11, 1e-17]
POLY_Y = [-0.3, 4e-4, -1e-3, 1e-6, -2e-6, 1e-6, -1e-9, 1e-9, 2e-9, -1e-9, -1e-5, -2e-6, -1e-11, 1e-17]


def _platepar(dist_type, equal_aspect=True, asymmetry_corr=False, force_centre=False, decentering=False,
    refraction=True):
    """ A 1280x720 camera with an 82 deg wide field 60 deg above the horizon and the given distortion. """

    pp = Platepar()
    pp.lat, pp.lon, pp.elev = 45.0, 15.0, 300.0
    pp.X_res, pp.Y_res = 1280, 720
    pp.JD = JD
    pp.Ho = JD2HourAngle(JD)
    pp.RA_d = JD2LST(JD, pp.lon)[0]
    pp.dec_d = pp.lat - 30.0
    pp.pos_angle_ref = 30.0
    pp.F_scale = 15.6
    pp.refraction = refraction

    # The flags first: setDistortionType sizes the arrays from them
    pp.equal_aspect = equal_aspect
    pp.asymmetry_corr = asymmetry_corr
    pp.force_distortion_centre = force_centre
    pp.decentering_corr = decentering
    pp.setDistortionType(dist_type, reset_params=True)

    if dist_type.startswith("poly"):
        pp.x_poly_fwd = np.array(POLY_X[:pp.poly_length])
        pp.y_poly_fwd = np.array(POLY_Y[:pp.poly_length])

    else:
        coeffs = []
        if not force_centre:
            coeffs += [0.01, -0.008]
        if not equal_aspect:
            coeffs += [0.004]
        if asymmetry_corr:
            coeffs += [0.01, 0.3]
        coeffs += RADIAL_K["odd" if dist_type.endswith("odd") else "all"][:pp.radialTermCount(dist_type)]
        if decentering:
            coeffs += [0.002, -0.001, 0.0005, -0.0003]
        assert len(coeffs) == pp.poly_length
        pp.x_poly_fwd = np.array(coeffs)

    # The reverse polynomial is not used with exact_reverse; a copy of the forward one stands in for it
    pp.x_poly_rev = np.array(pp.x_poly_fwd)
    pp.y_poly_rev = np.array(pp.y_poly_fwd)
    pp.exact_reverse = True

    return pp


def _roundTrip(pp):
    """ Largest distance (px) of a grid of image points from where sky -> image puts them back. """

    gx, gy = (v.ravel() for v in np.meshgrid(np.linspace(20, pp.X_res - 20, 25),
        np.linspace(20, pp.Y_res - 20, 15)))
    gx, gy = np.append(gx, pp.X_res/2.0), np.append(gy, pp.Y_res/2.0)

    _, ra, dec, _ = astrometry.xyToRaDecPP(np.full(gx.size, JD), gx, gy, np.ones(gx.size), pp,
        extinction_correction=False, jd_time=True)
    x, y = astrometry.raDecToXYPP(np.array(ra), np.array(dec), JD, pp)

    return np.max(np.hypot(np.array(x) - gx, np.array(y) - gy))


def _cases():
    """ Every distortion type with every combination of the flags it takes. """

    for dist_type in Platepar().distortion_type_list:
        if dist_type.startswith("poly"):
            yield dist_type, True, False, False, False
        else:
            for flags in itertools.product([True, False], repeat=4):
                yield (dist_type,) + flags


def _refractionInverts():
    """ Whether the sky -> image refraction is the exact inverse of the image -> sky one. """

    elev = np.radians(np.array([2.0, 10.0, 30.0, 60.0]))
    back = np.array([pyRefractionApparentToTrue(pyRefractionTrueToApparent(e)) for e in elev])

    return np.max(np.abs(back - elev)) < 1e-12


@pytest.mark.parametrize("dist_type,equal_aspect,asymmetry_corr,force_centre,decentering", list(_cases()))
def test_the_round_trip_closes_without_refraction(dist_type, equal_aspect, asymmetry_corr, force_centre,
    decentering):

    pp = _platepar(dist_type, equal_aspect, asymmetry_corr, force_centre, decentering, refraction=False)

    assert _roundTrip(pp) < TOLERANCE_PX


@pytest.mark.parametrize("dist_type,equal_aspect,asymmetry_corr,force_centre,decentering", list(_cases()))
def test_the_round_trip_closes_with_refraction(dist_type, equal_aspect, asymmetry_corr, force_centre,
    decentering):

    if not _refractionInverts():
        pytest.skip("refractionTrueToApparent is not the inverse of refractionApparentToTrue")

    pp = _platepar(dist_type, equal_aspect, asymmetry_corr, force_centre, decentering, refraction=True)

    assert _roundTrip(pp) < TOLERANCE_PX


def test_the_option_is_off_by_default():

    assert Platepar().exact_reverse is False


def test_a_point_beyond_the_reach_of_the_distortion_is_nan():
    """ A strong barrel distortion folds back past the frame: sky beyond its first maximum has no image. """

    pp = _platepar("radial3-odd", refraction=False)
    pp.x_poly_fwd[-1] = -0.3

    # Far outside the field, and the field centre for comparison
    _, ra, dec, _ = astrometry.xyToRaDecPP(np.full(1, JD), np.array([pp.X_res/2.0]), np.array([pp.Y_res/2.0]),
        np.ones(1), pp, extinction_correction=False, jd_time=True)
    x, y = astrometry.raDecToXYPP(np.array([ra[0], ra[0]]), np.array([dec[0], dec[0] - 55.0]), JD, pp)

    assert np.isfinite(x[0]) and np.isfinite(y[0])
    assert np.isnan(x[1]) and np.isnan(y[1])


@pytest.mark.parametrize("dist_type", ["radial5-odd", "radial7-odd", "poly3+radial3"])
def test_fitting_returns_the_forward_mapping_that_made_the_stars(dist_type):
    """ Noise-free stars made through the forward mapping, refitted with exact_reverse from a perturbed platepar.

    The fit stops a few thousandths of a pixel from the truth (0.005 px for radial5-odd, 0.009 px for radial7-odd,
    0.027 px for poly3+radial3, against 0.05, 0.012 and 0.037 px through the reverse polynomial): the pointing and
    the distortion centre, or the linear terms of the polynomial, trade off along a nearly flat valley, and
    scaling the parameters does not change it. Far below any centroiding noise.
    """

    true_pp = _platepar(dist_type, refraction=False)

    rng = np.random.default_rng(1)
    img_x = rng.uniform(40, true_pp.X_res - 40, 60)
    img_y = rng.uniform(40, true_pp.Y_res - 40, 60)
    _, ra, dec, _ = astrometry.xyToRaDecPP(np.full(img_x.size, JD), img_x, img_y, np.ones(img_x.size), true_pp,
        extinction_correction=False, jd_time=True)
    img_stars = np.column_stack([img_x, img_y, np.ones(img_x.size)])
    catalog_stars = np.column_stack([np.array(ra), np.array(dec), np.ones(img_x.size)])

    pp = copy.deepcopy(true_pp)
    pp.RA_d += 0.05
    pp.dec_d -= 0.03
    pp.pos_angle_ref += 0.1
    pp.x_poly_fwd = pp.x_poly_fwd*0.9
    pp.y_poly_fwd = pp.y_poly_fwd*0.9
    pp.fitAstrometry(JD, img_stars, catalog_stars)

    # Compared through the forward mapping, on a grid over the frame
    gx, gy = (v.ravel() for v in np.meshgrid(np.linspace(40, pp.X_res - 40, 12), np.linspace(40, pp.Y_res - 40, 8)))
    _, ra_true, dec_true, _ = astrometry.xyToRaDecPP(np.full(gx.size, JD), gx, gy, np.ones(gx.size), true_pp,
        extinction_correction=False, jd_time=True)
    x, y = astrometry.raDecToXYPP(np.array(ra_true), np.array(dec_true), JD, pp)

    assert np.max(np.hypot(np.array(x) - gx, np.array(y) - gy)) < 0.05
