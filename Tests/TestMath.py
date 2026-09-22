""" Tests for the angular separation functions in RMS.Math and their Cython counterpart. """

from __future__ import print_function, division, absolute_import

import pytest

np = pytest.importorskip("numpy")

from RMS.Math import angularSeparation, angularSeparationDeg, angularSeparationVect


def test_angular_separation_vect_handles_non_unit_vectors():
    """ The vector form normalizes its inputs, so non-unit vectors give the right angle. """

    vect1 = np.array([2.0, 0.0, 0.0])
    vect2 = np.array([0.0, 3.0, 0.0])

    angle = angularSeparationVect(vect1, vect2)

    assert np.isclose(angle, np.pi / 2)


def test_angular_separation_coincident_directions_not_nan():
    """ The spherical law of cosines can round above 1 for coincident directions, which made the
        unclipped arccos return NaN. (0, -82) deg is one such pair.
    """

    assert angularSeparationDeg(0.0, -82.0, 0.0, -82.0) == 0.0


def test_angular_separation_coincident_grid_not_nan():
    """ No coincident pair on a 1 deg grid may produce NaN (2160 of them did before the clip). """

    ra, dec = np.meshgrid(np.arange(0.0, 360.0, 1.0), np.arange(-89.0, 90.0, 1.0))
    separations = angularSeparationDeg(ra, dec, ra, dec)

    assert not np.any(np.isnan(separations))

    # The classical formula resolves down to the float64 arccos floor (~0.004 arcsec)
    assert np.max(separations) < 1e-5


def test_angular_separation_antipodal_unchanged():
    """ Clipping the lower end of the cosine must not perturb the antipodal case. """

    assert angularSeparation(0.0, 0.0, np.pi, 0.0) == np.pi
    assert angularSeparationDeg(0.0, 0.0, 180.0, 0.0) == 180.0


def test_angular_separation_propagates_nan_inputs():
    """ The clip guards the arccos domain only - a NaN coordinate must still give NaN instead of
        being silently clamped to 0 or 180 deg.
    """

    assert np.isnan(angularSeparationDeg(np.nan, 10.0, 20.0, 30.0))


def test_cython_angular_separation_matches_numpy():
    """ RMS.Astrometry.CyFunctions carries a second copy of the formula (degrees in, degrees out).
        It must be free of the same NaN and agree with the NumPy version.
    """

    cyfunctions = pytest.importorskip("RMS.Astrometry.CyFunctions")

    cy_ang_sep = cyfunctions.angularSeparation

    # The coincident case that trips the unclipped acos
    assert cy_ang_sep(0.0, -82.0, 0.0, -82.0) == 0.0

    # Antipodal points are unaffected
    assert cy_ang_sep(0.0, 0.0, 180.0, 0.0) == 180.0

    # No coincident pair on a 1 deg grid may produce NaN
    for dec in np.arange(-89.0, 90.0, 1.0):
        for ra in np.arange(0.0, 360.0, 1.0):
            assert not np.isnan(cy_ang_sep(ra, dec, ra, dec))

    # The two copies must agree on general pairs
    rng = np.random.default_rng(0)
    ra1 = rng.uniform(0, 360, 2000)
    ra2 = rng.uniform(0, 360, 2000)
    dec1 = np.degrees(np.arcsin(rng.uniform(-1, 1, 2000)))
    dec2 = np.degrees(np.arcsin(rng.uniform(-1, 1, 2000)))

    cy_separations = np.array([cy_ang_sep(*pair) for pair in zip(ra1, dec1, ra2, dec2)])

    assert np.allclose(cy_separations, angularSeparationDeg(ra1, dec1, ra2, dec2), atol=1e-10)
