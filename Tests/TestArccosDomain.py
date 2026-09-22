""" Regression tests for the arccos domain guards.

    Several routines take the arc cosine of a quantity that is mathematically confined to
    [-1, 1] - a dot product of unit vectors, a spherical law of cosines, a direction cosine.
    Floating point rounding can push such a quantity just outside that interval, in which case
    arccos returns NaN instead of 0 or pi. These tests exercise the degenerate geometries where
    that happens.
"""

from __future__ import print_function, division, absolute_import

import pytest

np = pytest.importorskip("numpy")

from RMS.Astrometry.Conversions import ECEF2AltAz
from RMS.Math import cartesianToPolar, polarToCartesian


def test_cartesian_to_polar_at_the_poles():
    """ A unit vector pointing at a pole gives z = +-1, which is the value most likely to round
        out of the arccos domain.
    """

    theta, _ = cartesianToPolar(0.0, 0.0, 1.0)
    assert theta == 0.0

    theta, _ = cartesianToPolar(0.0, 0.0, -1.0)
    assert theta == np.pi


def test_cartesian_to_polar_tolerates_rounding_past_unity():
    """ A z component one ULP outside the unit interval must not produce NaN. """

    for z in (np.nextafter(1.0, 2.0), np.nextafter(-1.0, -2.0)):
        theta, _ = cartesianToPolar(0.0, 0.0, z)
        assert not np.isnan(theta)


def test_cartesian_to_polar_round_trip():
    """ The clip must not perturb well conditioned directions.

        Note that polarToCartesian takes (azimuth, inclination) while cartesianToPolar returns
        (inclination, azimuth), so the two are inverses only with the arguments swapped.
    """

    rng = np.random.default_rng(0)

    for _ in range(100):
        inclination = rng.uniform(0, np.pi)
        azimuth = rng.uniform(-np.pi, np.pi)

        incl_out, azim_out = cartesianToPolar(*polarToCartesian(azimuth, inclination))

        assert np.isclose(incl_out, inclination)
        assert np.isclose(np.cos(azim_out - azimuth), 1.0)


def test_ecef2altaz_zenith_and_nadir():
    """ A point directly above the station makes the normalized dot product exactly 1, where
        rounding used to give NaN instead of an altitude of 90 degrees.
    """

    # An arbitrary station position on the ECEF sphere (metres)
    s_vect = np.array([2297292.0, -4676529.0, 3593555.0])

    # 80 km and 120 km are heights at which the unclipped formula returned NaN for this station
    for height in (80000.0, 120000.0, 100000.0, 50000.0):

        # A point straight up, and one straight down
        zenith_vect = s_vect*(1.0 + height/np.linalg.norm(s_vect))
        nadir_vect = s_vect*(1.0 - height/np.linalg.norm(s_vect))

        _, alt_zenith = ECEF2AltAz(s_vect, zenith_vect)
        _, alt_nadir = ECEF2AltAz(s_vect, nadir_vect)

        assert np.isclose(alt_zenith, 90.0)
        assert np.isclose(alt_nadir, -90.0)


def test_ecef2altaz_random_zenith_points():
    """ No point directly above the station may give a NaN altitude (19% of them did before). """

    rng = np.random.default_rng(3)

    for _ in range(2000):

        s_vect = rng.normal(size=3)
        s_vect = s_vect/np.linalg.norm(s_vect)*6371000.0

        p_vect = s_vect*(1.0 + rng.uniform(1e3, 4e5)/np.linalg.norm(s_vect))

        _, alt = ECEF2AltAz(s_vect, p_vect)

        assert np.isclose(alt, 90.0)


def test_equatorial_coord_precession_at_the_pole():
    """ equatorialCoordPrecession switches to an acos branch within 0.5 deg of the pole. """

    cyfunctions = pytest.importorskip("RMS.Astrometry.CyFunctions")

    j2000 = 2451545.0
    j2020 = 2458849.5

    for dec_deg in (90.0, -90.0, 89.9, -89.9):
        for ra_deg in (0.0, 90.0, 180.0, 270.0):

            _, dec_corr = cyfunctions.equatorialCoordPrecession(
                j2000, j2020, np.radians(ra_deg), np.radians(dec_deg))

            assert not np.isnan(dec_corr)

            # Precession over 20 years moves a star by well under a degree
            assert abs(np.degrees(dec_corr) - dec_deg) < 1.0
