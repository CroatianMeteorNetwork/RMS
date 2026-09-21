""" Tests for the catalog pre-selection helpers in RMS.Astrometry.StarFilters. """

from __future__ import print_function, division, absolute_import

import numpy as np
import pytest

from RMS.Astrometry.StarFilters import fovRadiusDeg, catalogInFOVMask


class _FakePlatepar(object):
    """ Only the fields the helpers read. """

    def __init__(self, x_res, y_res, f_scale, ra_d=0.0, dec_d=0.0):
        self.X_res = x_res
        self.Y_res = y_res
        self.F_scale = f_scale
        self.RA_d = ra_d
        self.dec_d = dec_d


def testFovRadiusIsInDegrees():
    """ A 1920x1080 image at 27 px/deg has a half diagonal of about 40.8 deg; with the 1.5 margin about 61 deg. """

    pp = _FakePlatepar(1920, 1080, 27.0)

    half_diagonal_deg = np.sqrt(1920**2 + 1080**2)/2/27.0

    assert fovRadiusDeg(pp, margin=1.0) == pytest.approx(half_diagonal_deg)
    assert fovRadiusDeg(pp) == pytest.approx(1.5*half_diagonal_deg)

    # Must not saturate at the cap for an ordinary narrow-field camera
    assert fovRadiusDeg(pp) < 90.0


def testFovRadiusIsCappedForAllSky():
    """ An all-sky lens with a tiny plate scale is capped at 90 deg. """

    pp = _FakePlatepar(1920, 1080, 4.0)

    assert fovRadiusDeg(pp) == 90.0


def testCatalogInFOVMaskRejectsStarsBehindCamera():
    """ Stars far from the pointing are masked out, stars near it are kept. """

    pp = _FakePlatepar(1920, 1080, 27.0, ra_d=100.0, dec_d=20.0)

    catalog_ra = np.array([100.0, 110.0, 280.0, 100.0])
    catalog_dec = np.array([20.0, 25.0, -20.0, -75.0])

    in_fov = catalogInFOVMask(catalog_ra, catalog_dec, pp)

    assert in_fov.tolist() == [True, True, False, False]
