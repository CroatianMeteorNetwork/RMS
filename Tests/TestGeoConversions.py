""" Geodetic conversions used by the ground projection: solving for the range at which a line of sight
    reaches a given height, and the ground position of a pixel.
"""

import math

import pytest

np = pytest.importorskip("numpy")

from RMS.Formats.Platepar import Platepar
from RMS.Misc import getRmsRootDir
from RMS.Astrometry.ApplyAstrometry import xyHt2Geo, geoHt2XY
from RMS.Astrometry.Conversions import AEH2Range, AEGeoidH2LatLonAlt, AER2ECEF, ecef2LatLonAlt

import os

TEMPLATE = os.path.join(getRmsRootDir(), 'share', 'platepar_templates', 'template_generic_720p_4mm.cal')

# A station on high ground, so that targets both above and below it are realistic
LAT, LON, ALT = 31.69, -110.9, 2610.0


def _heightAtRange(azim, elev, r):
    """ Helper: the WGS84 height reached at the given range along a line of sight. """

    return ecef2LatLonAlt(*AER2ECEF(azim, elev, r, LAT, LON, ALT))[2]


@pytest.mark.parametrize("elev, h", [
    (60.0, 100.0e3), (30.0, 100.0e3), (10.0, 100.0e3), (5.0, 100.0e3),
    (45.0, ALT + 3000.0), (20.0, ALT + 3000.0), (2.0, ALT + 3000.0),
    (-1.0, 2000.0), (-5.0, 1400.0), (-10.0, 500.0),
])
def test_range_lands_exactly_on_the_requested_height(elev, h):
    """ The solved range has to put the point at the requested height, for targets above and below the
        observer alike. The flat-Earth range (h - alt)/sin(elev) ignores the Earth curving away under the
        ray and is out by 100 km at 5 deg elevation, and the law of sines returns the far intersection for
        a target below the observer. """

    r = AEH2Range(180.0, elev, h, LAT, LON, ALT, accurate=True)

    assert np.isfinite(r)
    assert r > 0
    assert abs(_heightAtRange(180.0, elev, r) - h) < 1.0e-3


def test_flat_earth_range_would_be_wrong():
    """ The size of the error that the solve removes, so the test states what it is protecting against. """

    elev, h = 5.0, 100.0e3
    r_flat = (h - ALT)/math.sin(math.radians(elev))

    # The flat-Earth range overshoots by about 100 km in height at this elevation
    assert _heightAtRange(180.0, elev, r_flat) - h > 90.0e3


def test_downward_line_of_sight_takes_the_near_intersection():
    """ A line of sight pointing below the horizon crosses a lower height twice. The near crossing is the
        one in view; the far one is on the other side of the ray's lowest point, thousands of km away. """

    r = AEH2Range(180.0, -5.0, 1400.0, LAT, LON, ALT, accurate=True)

    assert r < 100.0e3


def test_unreachable_height_returns_nan():
    """ A line of sight pointing up never reaches a height below the observer. """

    assert not np.isfinite(AEH2Range(180.0, 20.0, 1000.0, LAT, LON, ALT, accurate=True))
    assert np.isnan(AEGeoidH2LatLonAlt(180.0, 20.0, 1000.0, LAT, LON, ALT)[0])


def test_default_range_path_is_unchanged():
    """ The analytical path is what EventMonitor and ShowerAssociation use, and it is left alone: accurate
        to of order 10 m for a target well above the observer. """

    for elev in (30.0, 45.0, 60.0):
        r = AEH2Range(180.0, elev, 100.0e3, LAT, LON, ALT)
        assert abs(_heightAtRange(180.0, elev, r) - 100.0e3) < 20.0


def test_geoid_conversion_lands_on_the_requested_height():
    """ AEGeoidH2LatLonAlt() has to return the point where the line of sight is at the given height. """

    for elev, h in ((45.0, ALT + 3000.0), (10.0, ALT + 3000.0), (5.0, 100.0e3), (-5.0, 1400.0)):

        lat_t, lon_t = AEGeoidH2LatLonAlt(180.0, elev, h, LAT, LON, ALT)
        r = AEH2Range(180.0, elev, h, LAT, LON, ALT, accurate=True)
        lat_r, lon_r, h_r = ecef2LatLonAlt(*AER2ECEF(180.0, elev, r, LAT, LON, ALT))

        assert abs(h_r - h) < 1.0e-3
        assert abs(lat_t - np.degrees(lat_r)) < 1.0e-9
        assert abs(lon_t - np.degrees(lon_r)) < 1.0e-9


@pytest.mark.parametrize("target_height", [ALT + 3000.0, 100.0e3])
def test_ground_projection_round_trips(target_height):
    """ xyHt2Geo() and geoHt2XY() have to be inverses of each other. They were not: xyHt2Geo() placed the
        target at the flat-Earth range while geoHt2XY() places it at exactly the given height, which on
        this platepar put the round trip about 3900 px away from where it started. The residual left is
        the epoch convention of geoHt2XY(), which evaluates the plate at J2000 so that the RA/Dec of date
        it computes may be taken as J2000, plus the forward/reverse polynomial mismatch of the plate
        itself. """

    pp = Platepar()
    pp.read(TEMPLATE)
    pp.lat, pp.lon, pp.elev = LAT, LON, ALT
    pp.JD = 2460000.5
    pp.refraction = True
    pp.az_centre, pp.alt_centre = 150.0, 25.0
    pp.updateRefRADec()

    x = np.array([100.0, 640.0, 1180.0])
    y = np.array([80.0, 360.0, 640.0])

    lat_t, lon_t = xyHt2Geo(pp, x, y, np.full(len(x), target_height))
    assert np.all(np.isfinite(lat_t))

    for i in range(len(x)):
        x_back, y_back = geoHt2XY(pp, lat_t[i], lon_t[i], target_height)
        assert np.hypot(x_back[0] - x[i], y_back[0] - y[i]) < 0.5, (x[i], y[i], x_back, y_back)
