""" Tests for the pointing parameter helpers in RMS.Formats.Platepar. """

from __future__ import print_function, division, absolute_import

import pytest

np = pytest.importorskip("numpy")

from RMS.Formats.Platepar import normalizeRaDec


@pytest.mark.parametrize("ra, dec", [(0.0, 0.0), (123.4, 45.6), (359.9, -89.9), (10.0, 89.999)])
def testNormalizeRaDecInsideRange(ra, dec):
    """ Values already on the sphere are returned unchanged. """

    # Nothing to normalize, the input is already inside the valid range
    ra_out, dec_out, pos_angle_offset = normalizeRaDec(ra, dec)

    assert ra_out == pytest.approx(ra)
    assert dec_out == pytest.approx(dec)
    assert pos_angle_offset == 0.0


@pytest.mark.parametrize("dec", [90.0, -90.0])
def testNormalizeRaDecExactlyAtPoles(dec):
    """ The poles themselves are fixed points and the RA is not flipped. """

    # A declination of exactly +/- 90 deg is on the boundary and must not be reflected
    ra_out, dec_out, pos_angle_offset = normalizeRaDec(50.0, dec)

    assert ra_out == pytest.approx(50.0)
    assert dec_out == pytest.approx(dec)
    assert pos_angle_offset == 0.0


@pytest.mark.parametrize("ra, dec, ra_expected, dec_expected", [
    # Just past the north pole: reflect, RA advances by 180
    (50.0, 95.0, 230.0, 85.0),
    # Just past the south pole
    (50.0, -95.0, 230.0, -85.0),
    # Infinitesimally past the north pole must stay next to the north pole, not jump to -90
    (50.0, 90.000001, 230.0, 89.999999),
    # RA wraps around after the 180 deg advance
    (300.0, 95.0, 120.0, 85.0),
    # Dec of 180 is the antipode of dec 0 on the same meridian
    (0.0, 180.0, 180.0, 0.0),
    # Dec of 270 is the same as -90 (no RA flip needed)
    (0.0, 270.0, 0.0, -90.0),
    # Negative RA wraps into [0, 360)
    (-10.0, 20.0, 350.0, 20.0),
])
def testNormalizeRaDecReflectsAcrossPole(ra, dec, ra_expected, dec_expected):
    """ Declinations past a pole are reflected, with the RA advanced by 180 degrees. """

    # Every parametrized case crosses a pole, so the RA has to move by 180 deg
    ra_out, dec_out, pos_angle_offset = normalizeRaDec(ra, dec)

    assert ra_out == pytest.approx(ra_expected)
    assert dec_out == pytest.approx(dec_expected)

    # A reflection across the pole turns the local tangent frame by 180 deg
    reflected = abs(((ra - ra_expected + 180)%360) - 180) > 1e-9
    assert pos_angle_offset == (180.0 if reflected else 0.0)


def testNormalizeRaDecReflectionPreservesDirection():
    """ The reflected point must be the same unit vector as the raw (ra, dec) pair. """

    # A declination 7.5 deg past the north pole
    ra, dec = 37.0, 97.5
    ra_out, dec_out, pos_angle_offset = normalizeRaDec(ra, dec)

    def toVector(ra_d, dec_d):
        """ Convert an (RA, Dec) pair in degrees to a unit vector.

        Arguments:
            ra_d: [float] Right ascension (deg).
            dec_d: [float] Declination (deg).

        Return:
            vect: [ndarray] Unit vector pointing at the given direction.
        """

        ra_r, dec_r = np.radians(ra_d), np.radians(dec_d)
        return np.array([np.cos(dec_r)*np.cos(ra_r), np.cos(dec_r)*np.sin(ra_r), np.sin(dec_r)])

    # The normalized pair must point in exactly the same direction and be inside the valid ranges
    assert np.allclose(toVector(ra, dec), toVector(ra_out, dec_out))
    assert 0.0 <= ra_out < 360.0
    assert -90.0 <= dec_out <= 90.0


def testProjectionIsContinuousAcrossThePole():
    """ Stepping the declination parameter past the pole must not jump the projected field.

        The optimizers step the declination freely, so the parameter regularly crosses 90 deg for a
        camera pointing near the celestial pole. normalizeRaDec reflects the pointing back onto the
        sphere; without also rotating the position angle by the returned offset, the field would flip
        by 180 deg at the crossing and the cost function would be discontinuous.
    """

    from RMS.Formats.Platepar import Platepar
    from RMS.Astrometry.ApplyAstrometry import raDecToXYPP

    platepar = Platepar()
    platepar.X_res, platepar.Y_res = 1920, 1080
    platepar.F_scale = 27.0
    platepar.JD = 2460000.5
    platepar.refraction = False

    ra_param = 30.0
    pos_angle_param = 40.0

    # Catalog stars a couple of degrees from the pole, so they stay in the field the whole way
    star_ra = np.array([30.0, 120.0, 210.0])
    star_dec = np.array([88.0, 88.5, 88.2])


    def projectStars(dec_param):
        """ Project the stars with the pointing given by the raw declination parameter. """

        ra_d, dec_d, pos_angle_offset = normalizeRaDec(ra_param, dec_param)

        platepar.RA_d = ra_d
        platepar.dec_d = dec_d
        platepar.pos_angle_ref = (pos_angle_param + pos_angle_offset)%360

        x_arr, y_arr = raDecToXYPP(star_ra, star_dec, platepar.JD, platepar)

        return np.c_[x_arr, y_arr]


    # Walk the declination parameter across the pole in even steps
    dec_params = [89.90, 89.96, 89.99, 90.01, 90.04, 90.10]
    positions = [projectStars(dec_param) for dec_param in dec_params]

    # Every step of the parameter moves the stars by a comparable amount, the crossing included
    steps = [np.max(np.abs(positions[i + 1] - positions[i])) for i in range(len(positions) - 1)]

    assert max(steps) < 5.0


RADIAL_TYPES = ["radial3-all", "radial4-all", "radial5-all", "radial3-odd", "radial5-odd", "radial7-odd",
                "radial9-odd"]

RADIAL_FLAGS = [
    # (flag name, restricted value, free value)
    ('force_distortion_centre', True, False),
    ('equal_aspect', True, False),
    ('asymmetry_corr', False, True),
]


def _radialPlatepar(dist_type):
    """ A platepar with a radial distortion with all flags in the restricted state and non-zero ks. """

    from RMS.Formats.Platepar import Platepar

    pp = Platepar()
    pp.X_res, pp.Y_res = 1920, 1080
    pp.F_scale = 1920/90.0
    pp.JD = 2460000.5
    pp.RA_d, pp.dec_d = 100.0, 30.0
    pp.pos_angle_ref = 20.0
    pp.lat, pp.lon = 45.0, 15.0
    pp.refraction = False

    pp.force_distortion_centre = True
    pp.equal_aspect = True
    pp.asymmetry_corr = False
    pp.setDistortionType(dist_type, reset_params=True)

    # Set small, non-zero radial coefficients (the array only contains the ks in this flag state)
    n = pp.poly_length
    ks = 0.02*(np.arange(n) + 1)*(-1)**np.arange(n)
    pp.x_poly_fwd = ks.copy()
    pp.x_poly_rev = -ks.copy()
    pp.y_poly_fwd = ks.copy()
    pp.y_poly_rev = -ks.copy()
    pp.x_poly = pp.x_poly_fwd
    pp.y_poly = pp.y_poly_fwd

    return pp


def _projection(pp):
    """ Project an image grid to the sky and a sky grid to the image, both used as a fingerprint. """

    from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, xyToRaDecPP

    xs, ys = np.meshgrid(np.linspace(10, pp.X_res - 10, 9), np.linspace(10, pp.Y_res - 10, 7))
    xs, ys = xs.ravel(), ys.ravel()

    _, ra, dec, _ = xyToRaDecPP(len(xs)*[pp.JD], xs, ys, np.ones(len(xs)), pp, extinction_correction=False,
        jd_time=True)
    x_back, y_back = raDecToXYPP(ra, dec, pp.JD, pp)

    return np.c_[ra, dec], np.c_[x_back, y_back]


@pytest.mark.parametrize("dist_type", RADIAL_TYPES)
def testBuildRadialCoeffsMatchesPolyLength(dist_type):
    """ Rebuilding the coefficients must give an array of the length the distortion type uses. """

    pp = _radialPlatepar(dist_type)

    for flag_state in [(True, True, False), (False, False, True)]:
        pp.force_distortion_centre, pp.equal_aspect, pp.asymmetry_corr = flag_state
        pp.setDistortionType(dist_type, reset_params=False)

        coeffs = pp.buildRadialCoeffs(pp.extractRadialCoeffs(np.arange(1, 20, dtype=float)))

        assert len(coeffs) == pp.poly_length


@pytest.mark.parametrize("dist_type", RADIAL_TYPES)
@pytest.mark.parametrize("flag_name, restricted_value, free_value", RADIAL_FLAGS)
def testRadialFlagToggleKeepsProjection(dist_type, flag_name, restricted_value, free_value):
    """ Freeing a distortion flag (and restricting it back) must not change the projection.

        The newly freed parameters start at the values the restricted state implies (the forced centre is
        half a pixel from the image centre, zero aspect and asymmetry), and every radial coefficient must be
        carried over.
    """

    pp = _radialPlatepar(dist_type)
    sky_ref, img_ref = _projection(pp)

    # Free the flag
    assert pp.remapCoeffsForFlagChange(flag_name, free_value)
    assert len(pp.x_poly_fwd) == pp.poly_length
    sky, img = _projection(pp)
    assert np.max(np.abs(img - img_ref)) < 1e-6
    assert np.max(np.abs(sky - sky_ref)) < 1e-6/3600

    # Restrict it again
    assert pp.remapCoeffsForFlagChange(flag_name, restricted_value)
    sky, img = _projection(pp)
    assert np.max(np.abs(img - img_ref)) < 1e-6
    assert np.max(np.abs(sky - sky_ref)) < 1e-6/3600
