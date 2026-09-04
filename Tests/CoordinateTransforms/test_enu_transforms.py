#!/usr/bin/env python

""" Round-trip checks for the ENU/geodetic image transforms used by the GMN contrail pipeline.

These cover the functions ported from the test-coordinate-transforms branch:
xyHtToENUPP, enHtToXYPP, enuToXYPP, geoToENUPP, geoToXYPP, ENHt0ToENHt1 and xyToAltAzPP.

Run directly (python Tests/CoordinateTransforms/test_enu_transforms.py) or under pytest.
"""

from __future__ import print_function, division, absolute_import

import os
import sys

import numpy as np

# Add the RMS repository root to the path (this file lives in Tests/CoordinateTransforms/)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)

# A platepar shipped with the repo, so this test runs anywhere
PLATEPAR_PATH = os.path.join(REPO_ROOT, 'share', 'platepar_templates',
                             'template_generic_720p_4mm.cal')

from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.ApplyAstrometry import (xyHtToENUPP, enHtToXYPP, enuToXYPP, geoToENUPP,
                                            geoToXYPP, ENHt0ToENHt1, xyToAltAzPP)


# Test image points, kept well inside the frame
X_PTS = np.array([200., 400., 640., 800., 1000.])
Y_PTS = np.array([150., 250., 360., 450., 550.])

# Target height above the WGS84 ellipsoid (m), roughly cruise altitude
HT_WGS84 = 10000.0


def buildPlatepar():
    """ Load the bundled template platepar and point it somewhere realistic. """

    pp = Platepar()
    pp.read(PLATEPAR_PATH)

    # A mid-latitude station and an off-zenith pointing, so the geodetic maths is exercised
    pp.lat, pp.lon, pp.elev = 43.19, -81.32, 324.0
    pp.RA_d, pp.dec_d, pp.pos_angle_ref = 45.0, 55.0, 30.0

    # Drop any cached derived height, since lat/lon/elev were just changed
    pp._height_wgs84 = None

    return pp


def test_xy_enu_roundtrip():
    """ XY -> ENU -> XY must return the original pixel coordinates. """

    pp = buildPlatepar()
    ht = np.full_like(X_PTS, HT_WGS84)

    E, N, U = xyHtToENUPP(X_PTS, Y_PTS, ht, pp)[:3]

    # Back through the East/North + height solver
    x_en, y_en = enHtToXYPP(E, N, ht, pp)
    assert np.hypot(x_en - X_PTS, y_en - Y_PTS).max() < 0.05

    # And back through the full ENU vector
    x_enu, y_enu = enuToXYPP(E, N, U, pp)
    assert np.hypot(x_enu - X_PTS, y_enu - Y_PTS).max() < 0.05


def test_geo_enu_consistency():
    """ geoToENUPP -> enuToXYPP must agree with geoToXYPP for the same geodetic points. """

    pp = buildPlatepar()
    ht = np.full_like(X_PTS, HT_WGS84)

    E, N, U = xyHtToENUPP(X_PTS, Y_PTS, ht, pp)[:3]

    # Turn the ENU offsets back into geodetic coordinates with a local flat-Earth step. This only
    # has to be good enough to keep the points inside the FOV - the assertion below compares the
    # two projection paths against each other, not against the original pixels.
    earth_r = 6371000.0
    lat = pp.lat + np.degrees(N/earth_r)
    lon = pp.lon + np.degrees(E/(earth_r*np.cos(np.radians(pp.lat))))
    h = np.full_like(lat, HT_WGS84)

    x_geo, y_geo = geoToXYPP(lat, lon, h, pp)

    E_geo, N_geo, U_geo = geoToENUPP(lat, lon, h, pp)
    x_enu, y_enu = enuToXYPP(E_geo, N_geo, U_geo, pp)

    inside = (x_geo >= 0) & (x_geo < pp.X_res) & (y_geo >= 0) & (y_geo < pp.Y_res)
    assert inside.any(), "no test point projected inside the image"
    assert np.hypot(x_geo - x_enu, y_geo - y_enu)[inside].max() < 0.05

    # The flat-Earth step is approximate, but the points should still land near the originals
    assert np.hypot(x_geo - X_PTS, y_geo - Y_PTS)[inside].max() < 50.0


def test_enht0_to_enht1():
    """ ENHt0ToENHt1 must preserve the line of sight when changing the target height. """

    pp = buildPlatepar()
    ht0 = np.full_like(X_PTS, HT_WGS84)

    E, N = xyHtToENUPP(X_PTS, Y_PTS, ht0, pp)[:2]

    # Same height in and out is the identity (to solver tolerance)
    E_same, N_same = ENHt0ToENHt1(E, N, ht0, ht0, pp)[:2]
    assert np.sqrt((E_same - E)**2 + (N_same - N)**2).max() < 1.0

    # Moving along the ray to 20 km must reproject to the same pixel
    ht1 = np.full_like(X_PTS, 20000.0)
    E1, N1 = ENHt0ToENHt1(E, N, ht0, ht1, pp)[:2]
    x1, y1 = enHtToXYPP(E1, N1, ht1, pp)
    assert np.hypot(x1 - X_PTS, y1 - Y_PTS).max() < 0.05


def test_xy_to_altaz_matches_radec_path():
    """ xyToAltAzPP must agree with going XY -> RA/Dec -> Alt/Az to about an arcminute.

    xyToAltAzPP projects directly in Alt/Az (a different gnomonic path than xyToRaDecPP followed by
    a celestial-to-horizontal conversion), so the two are not expected to agree exactly - they
    handle refraction and precession at different points. Around 1 arcmin is the observed agreement;
    the 0.05 deg bound catches a genuine breakage without failing on that by-design difference.
    """

    from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP
    from RMS.Astrometry.Conversions import trueRaDec2ApparentAltAz, jd2Date

    pp = buildPlatepar()

    alt, azim = xyToAltAzPP(X_PTS, Y_PTS, pp)

    _, ra, dec, _ = xyToRaDecPP([jd2Date(pp.JD)]*len(X_PTS), X_PTS, Y_PTS,
                                np.ones_like(X_PTS), pp, extinction_correction=False)

    alt_ref, az_ref = [], []
    for ra_i, dec_i in zip(ra, dec):
        az_i, alt_i = trueRaDec2ApparentAltAz(ra_i, dec_i, pp.JD, pp.lat, pp.lon,
                                              refraction=pp.refraction)
        az_ref.append(az_i)
        alt_ref.append(alt_i)

    assert np.abs(alt - np.array(alt_ref)).max() < 0.05

    # Compare azimuth in the tangent plane, wrapping across 0/360
    d_az = np.abs((azim - np.array(az_ref) + 180) % 360 - 180)*np.cos(np.radians(alt))
    assert d_az.max() < 0.05


if __name__ == "__main__":

    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            fn()
            print("PASS  {:s}".format(name))

    print("\nAll ENU/geodetic transform checks passed.")
