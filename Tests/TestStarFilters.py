""" Tests for the catalog pre-selection helpers in RMS.Astrometry.StarFilters. """

from __future__ import print_function, division, absolute_import

import numpy as np
import pytest

from RMS.Astrometry.StarFilters import fovRadiusDeg, catalogInFOVMask


class _FakePlatepar(object):
    """ Only the fields the helpers read. """

    def __init__(self, x_res, y_res, f_scale, ra_d=0.0, dec_d=0.0):
        """ Set up a minimal stand-in for a fitted platepar.

        Arguments:
            x_res: [int] Image width (px).
            y_res: [int] Image height (px).
            f_scale: [float] Plate scale (px/deg).

        Keyword arguments:
            ra_d: [float] Right ascension of the image centre (deg). 0 by default.
            dec_d: [float] Declination of the image centre (deg). 0 by default.
        """

        self.X_res = x_res
        self.Y_res = y_res
        self.F_scale = f_scale
        self.RA_d = ra_d
        self.dec_d = dec_d


def testFovRadiusIsInDegrees():
    """ A 1920x1080 image at 27 px/deg has a half diagonal of about 40.8 deg; with the 1.5 margin
        about 61 deg. """

    pp = _FakePlatepar(1920, 1080, 27.0)

    # Half the image diagonal, converted from px to deg with the plate scale
    half_diagonal_deg = np.sqrt(1920**2 + 1080**2)/2/27.0

    # Without a margin the radius is exactly the half diagonal, the default margin scales it by 1.5
    assert fovRadiusDeg(pp, margin=1.0) == pytest.approx(half_diagonal_deg)
    assert fovRadiusDeg(pp) == pytest.approx(1.5*half_diagonal_deg)

    # Must not saturate at the cap for an ordinary narrow-field camera
    assert fovRadiusDeg(pp) < 90.0


def testFovRadiusIsCappedForAllSky():
    """ An all-sky lens with a tiny plate scale is capped at 90 deg. """

    # At 4 px/deg the margined half diagonal is well over 90 deg
    pp = _FakePlatepar(1920, 1080, 4.0)

    assert fovRadiusDeg(pp) == 90.0


def testCatalogInFOVMaskRejectsStarsBehindCamera():
    """ Stars far from the pointing are masked out, stars near it are kept. """

    pp = _FakePlatepar(1920, 1080, 27.0, ra_d=100.0, dec_d=20.0)

    # Two stars near the pointing, one on the opposite side of the sky and one far to the south
    catalog_ra = np.array([100.0, 110.0, 280.0, 100.0])
    catalog_dec = np.array([20.0, 25.0, -20.0, -75.0])

    in_fov = catalogInFOVMask(catalog_ra, catalog_dec, pp)

    # Only the two stars near the pointing may survive
    assert in_fov.tolist() == [True, True, False, False]
