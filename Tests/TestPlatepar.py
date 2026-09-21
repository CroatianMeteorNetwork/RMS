""" Tests for the pointing parameter helpers in RMS.Formats.Platepar. """

from __future__ import print_function, division, absolute_import

import pytest

np = pytest.importorskip("numpy")

from RMS.Formats.Platepar import normalizeRaDec


@pytest.mark.parametrize("ra, dec", [(0.0, 0.0), (123.4, 45.6), (359.9, -89.9), (10.0, 89.999)])
def testNormalizeRaDecInsideRange(ra, dec):
    """ Values already on the sphere are returned unchanged. """

    ra_out, dec_out = normalizeRaDec(ra, dec)

    assert ra_out == pytest.approx(ra)
    assert dec_out == pytest.approx(dec)


@pytest.mark.parametrize("dec", [90.0, -90.0])
def testNormalizeRaDecExactlyAtPoles(dec):
    """ The poles themselves are fixed points and the RA is not flipped. """

    ra_out, dec_out = normalizeRaDec(50.0, dec)

    assert ra_out == pytest.approx(50.0)
    assert dec_out == pytest.approx(dec)


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

    ra_out, dec_out = normalizeRaDec(ra, dec)

    assert ra_out == pytest.approx(ra_expected)
    assert dec_out == pytest.approx(dec_expected)


def testNormalizeRaDecReflectionPreservesDirection():
    """ The reflected point must be the same unit vector as the raw (ra, dec) pair. """

    ra, dec = 37.0, 97.5
    ra_out, dec_out = normalizeRaDec(ra, dec)

    def toVector(ra_d, dec_d):
        ra_r, dec_r = np.radians(ra_d), np.radians(dec_d)
        return np.array([np.cos(dec_r)*np.cos(ra_r), np.cos(dec_r)*np.sin(ra_r), np.sin(dec_r)])

    assert np.allclose(toVector(ra, dec), toVector(ra_out, dec_out))
    assert 0.0 <= ra_out < 360.0
    assert -90.0 <= dec_out <= 90.0
