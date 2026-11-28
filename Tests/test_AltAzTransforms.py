""" Image coordinates to and from alt/az, east-north-up (ENU) and geodetic coordinates at a height
    (xyToAltAzPP, xyHtToENUPP, geoToENUPP, geoToXYPP, enuToXYPP, enHtToXYPP, enHt0ToEnHt1), the refraction of
    targets at a known height or range, the station height above the WGS84 ellipsoid (Platepar.height_wgs84) and
    the EGM96 geoid model behind it.

    The transforms map the image through xyToRaDecPP and raDecToXYPP in the horizontal frame, so they are checked
    against their RA/Dec path, on plates with the decentering correction and the aberration flag too, and their
    geometry against ECEF coordinates computed here.
"""

import copy
import json
import math
import os

import pytest

np = pytest.importorskip("numpy")

import RMS.ConfigReader as cr
import RMS.GeoidHeightEGM96 as GeoidHeightEGM96
from RMS.Astrometry.ApplyAstrometry import (enHt0ToEnHt1, enHtToXYPP, enuToXYPP, geoToENUPP, geoToXYPP,
    plateRefractionScale, raDecToXYPP, targetRefractionApparentToTrue, targetRefractionTrueToApparent,
    xyHtToENUPP, xyToAltAzPP, xyToRaDecPP)
from RMS.Astrometry.Conversions import (apparentAltAz2TrueOfDateRaDec, ecef2LatLonAlt, enHeight2Up,
    enu2LatLonAlt, latLonAlt2ECEF, latLonAlt2ENU, rangeToHeightWGS84)
from RMS.Astrometry.CyFunctions import (applyAberration, cyAltAzHeight2ENU, cyTargetRefraction,
    cyTrueRaDec2ApparentAltAz, pyRefractionApparentToTrue, pyRefractionTrueToApparent, refractionScale,
    refractionTargetFraction)
from RMS.Formats.CAL import writeCAL
from RMS.Formats.Platepar import Platepar
from RMS.GeoidHeightEGM96 import mslToWGS84Height, wgs84toMSLHeight
from RMS.Misc import getRmsRootDir
from Tests.TestRefraction import _rayTrace

try:
    import erfa
except ImportError:
    erfa = None


# Templates shipped with RMS, so the tests do not depend on any station's data
TEMPLATE_DIR = os.path.join(getRmsRootDir(), 'share', 'platepar_templates')
TEMPLATES = ['template_generic_720p_4mm.cal', 'template_generic_720p_6mm.cal']

# Station of the tests (height above sea level)
TEST_LAT, TEST_LON, TEST_ELEV = 43.2, -81.3, 300.0

# Centre altitudes bracket real pointings; the rolled variants catch a wrong rotation, which a roll of 0 can mask
POINTINGS = [(alt, azim, roll) for alt in (69.0, 36.0, 12.0) for azim in (20.0, 200.0) for roll in (0.0, 45.0)]


def pointedPlatepar(template, alt_centre, azim_centre, roll):
    """ Helper: a template platepar at the test station with its centre at the given alt/az (deg). """

    pp = Platepar()
    pp.read(os.path.join(TEMPLATE_DIR, template), use_flat=False)
    pp.lat, pp.lon, pp.elev = TEST_LAT, TEST_LON, TEST_ELEV

    # The true-of-date reference pointing of the alt/az of the centre
    pp.RA_d, pp.dec_d = apparentAltAz2TrueOfDateRaDec(azim_centre, alt_centre, pp.JD, pp.lat, pp.lon,
        refraction=False)
    pp.pos_angle_ref = roll%360

    return pp


def frameGrid(platepar, n=7):
    """ Helper: image coordinates spanning the full frame, the corners included. """

    xs = np.linspace(0, platepar.X_res, n)
    ys = np.linspace(0, platepar.Y_res, n)
    xx, yy = np.meshgrid(xs, ys)

    return xx.ravel(), yy.ravel()


def referenceAltAz(platepar, x_data, y_data, aberration=False, refraction=True):
    """ Helper: alt/az through xyToRaDecPP and a conversion per point, with the refraction of a star at the
        station height put back (the apparent direction) or not (deg). """

    jd = platepar.JD
    _, ra_arr, dec_arr, _ = xyToRaDecPP(np.full(len(x_data), jd), np.asarray(x_data, dtype=np.float64),
        np.asarray(y_data, dtype=np.float64), np.ones(len(x_data)), platepar, extinction_correction=False,
        jd_time=True, aberration=aberration)

    altaz = [cyTrueRaDec2ApparentAltAz(np.radians(ra), np.radians(dec), jd, np.radians(platepar.lat),
        np.radians(platepar.lon), refraction, refractionScale(platepar.elev)) for ra, dec in zip(ra_arr, dec_arr)]

    return np.degrees([a[1] for a in altaz]), np.degrees([a[0] for a in altaz])


def separationArcsec(alt1, azim1, alt2, azim2):
    """ Helper: angular separation between alt/az directions (arcsec), from the cross and dot products. """

    a1, A1, a2, A2 = (np.radians(np.asarray(v, dtype=np.float64)) for v in (alt1, azim1, alt2, azim2))
    v1 = np.stack([np.cos(a1)*np.sin(A1), np.cos(a1)*np.cos(A1), np.sin(a1)], axis=-1)
    v2 = np.stack([np.cos(a2)*np.sin(A2), np.cos(a2)*np.cos(A2), np.sin(a2)], axis=-1)

    return np.degrees(np.arctan2(np.linalg.norm(np.cross(v1, v2), axis=-1), np.sum(v1*v2, axis=-1)))*3600


def enuAxesByHand(lat, lon):
    """ Helper: east, north and up unit vectors of the geodetic vertical in ECEF, written out here. """

    lat, lon = math.radians(lat), math.radians(lon)
    east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    up = np.array([math.cos(lat)*math.cos(lon), math.cos(lat)*math.sin(lon), math.sin(lat)])

    return east, np.cross(up, east), up


def distantPoint(alt, azim, distance=1.0e9):
    """ Helper: ENU coordinates of a point far away in the given direction (deg). """

    alt, azim = np.radians(alt), np.radians(azim)

    return distance*np.cos(alt)*np.sin(azim), distance*np.cos(alt)*np.cos(azim), distance*np.sin(alt)


@pytest.mark.parametrize("template", TEMPLATES)
@pytest.mark.parametrize("alt_centre, azim_centre, roll", POINTINGS)
def test_altaz_matches_the_radec_path(template, alt_centre, azim_centre, roll):
    """ xyToAltAzPP is xyToRaDecPP converted to alt/az with the refraction of a star put back (the direction
        the light arrives from), over the whole frame, rolled or not. """

    platepar = pointedPlatepar(template, alt_centre, azim_centre, roll)
    x_data, y_data = frameGrid(platepar)

    alt_ref, azim_ref = referenceAltAz(platepar, x_data, y_data)
    alt, azim = xyToAltAzPP(x_data, y_data, platepar)

    assert np.max(separationArcsec(alt_ref, azim_ref, alt, azim)) < 1e-3


def test_altaz_follows_the_decentering():
    """ On a plate with the decentering correction the alt/az follows xyToRaDecPP, which the correction moves by
        more than a pixel in the corners, and the image round trip closes. """

    platepar = pointedPlatepar(TEMPLATES[0], 36.0, 110.0, 30.0)
    x_data, y_data = frameGrid(platepar)
    alt_plain, azim_plain = xyToAltAzPP(x_data, y_data, platepar)

    # A decentered lens element
    platepar.remapCoeffsForFlagChange('decentering_corr', True)
    platepar.x_poly_fwd[-4:] = [0.002, -0.0015, 0.001, 0.0008]
    platepar.x_poly_rev[-4:] = [-0.002, 0.0015, -0.001, -0.0008]

    alt_ref, azim_ref = referenceAltAz(platepar, x_data, y_data)
    alt, azim = xyToAltAzPP(x_data, y_data, platepar)

    assert np.max(separationArcsec(alt_ref, azim_ref, alt, azim)) < 1e-3
    assert np.max(separationArcsec(alt_plain, azim_plain, alt, azim)) > 3600/platepar.F_scale

    # Through a height and back (the exact inverse), the corners below the horizon included
    platepar.exact_reverse = True
    E, N, U = xyHtToENUPP(x_data, y_data, 11000.0, platepar, min_el_deg=-90.0)[:3]
    x_back, y_back = enuToXYPP(E, N, U, platepar, min_el_deg=-90.0)
    assert np.all(np.isfinite(x_back))
    assert np.max(np.hypot(x_back - x_data, y_back - y_data)) < 1e-3


def test_altaz_is_in_the_frame_of_the_earth():
    """ On a plate that models the annual aberration, the alt/az is in the frame of the Earth (xyToRaDecPP with
        aberration=False), not the catalog direction of a star seen there, which differs by up to 20.5 arcsec. """

    platepar = pointedPlatepar(TEMPLATES[0], 36.0, 110.0, 0.0)
    platepar.aberration = True
    x_data, y_data = frameGrid(platepar)

    alt, azim = xyToAltAzPP(x_data, y_data, platepar)
    alt_earth, azim_earth = referenceAltAz(platepar, x_data, y_data, aberration=False)
    alt_cat, azim_cat = referenceAltAz(platepar, x_data, y_data, aberration=None)

    assert np.max(separationArcsec(alt_earth, azim_earth, alt, azim)) < 1e-3
    assert 5 < np.max(separationArcsec(alt_cat, azim_cat, alt, azim)) < 21


@pytest.mark.parametrize("aberration", [True, False])
def test_star_lands_on_its_pixel(aberration):
    """ A star in its apparent direction (the catalog direction displaced by the annual aberration, and refracted)
        is where xyToAltAzPP puts the pixel the platepar puts the star on, and a distant point in its unrefracted
        direction lands on that pixel (the plate through its exact inverse: the reverse distortion of the template
        is off by more than a pixel in the corners). A plate fitted without the aberration modelled has absorbed
        it, so there the catalog direction is the one that lands on the pixel. """

    platepar = pointedPlatepar(TEMPLATES[0], 36.0, 110.0, 20.0)
    platepar.aberration = aberration
    platepar.exact_reverse = True
    x_data, y_data = frameGrid(platepar, n=5)

    # Catalog directions of stars on the grid, and where the platepar puts them
    _, ra_arr, dec_arr, _ = xyToRaDecPP(np.full(len(x_data), platepar.JD), x_data, y_data, np.ones(len(x_data)),
        platepar, extinction_correction=False, jd_time=True)
    x_star, y_star = raDecToXYPP(ra_arr, dec_arr, platepar.JD, platepar)

    for x, y, ra, dec in zip(x_star, y_star, ra_arr, dec_arr):

        # The direction of the star in the frame of the Earth, refracted or not
        ra, dec = np.radians(ra), np.radians(dec)
        if aberration:
            ra, dec = applyAberration(ra, dec, platepar.JD)
        azim_app, alt_app = np.degrees(cyTrueRaDec2ApparentAltAz(ra, dec, platepar.JD,
            np.radians(platepar.lat), np.radians(platepar.lon), True, refractionScale(platepar.elev)))
        azim, alt = np.degrees(cyTrueRaDec2ApparentAltAz(ra, dec, platepar.JD, np.radians(platepar.lat),
            np.radians(platepar.lon), False))

        # As a direction and as a distant point
        alt_px, azim_px = xyToAltAzPP([x], [y], platepar)
        x_px, y_px = enuToXYPP(*distantPoint(alt, azim), platepar, min_el_deg=-90.0)

        assert separationArcsec(alt_px, azim_px, alt_app, azim_app)[0] < 0.01
        assert math.hypot(x_px[0] - x, y_px[0] - y) < 1e-3


@pytest.mark.parametrize("alt_centre, azim_centre, roll", POINTINGS[::3])
@pytest.mark.parametrize("ht_m", [1000.0, 11000.0])
@pytest.mark.parametrize("exact_reverse", [True, False])
def test_xy_enu_round_trip(alt_centre, azim_centre, roll, ht_m, exact_reverse):
    """ XY -> ENU -> XY through all three inverses. With the exact reverse flag the round trip returns the original
        pixels; without it the sky goes to the image through the reverse distortion of the plate, so the pixels
        come back where xyToRaDecPP -> raDecToXYPP takes them (off by the fit error of the reverse distortion,
        several pixels in the corners of the template). """

    platepar = pointedPlatepar(TEMPLATES[0], alt_centre, azim_centre, roll)
    platepar.exact_reverse = exact_reverse
    x_data, y_data = frameGrid(platepar, n=5)

    # Where the plate itself takes the pixels back
    _, ra, dec, _ = xyToRaDecPP(np.full(len(x_data), platepar.JD), x_data, y_data, None, platepar, jd_time=True)
    x_ref, y_ref = raDecToXYPP(ra, dec, platepar.JD, platepar)
    if exact_reverse:
        assert np.max(np.hypot(x_ref - x_data, y_ref - y_data)) < 1e-4
    else:
        assert np.max(np.hypot(x_ref - x_data, y_ref - y_data)) > 0.1

    E, N, U = xyHtToENUPP(x_data, y_data, ht_m, platepar)[:3]
    valid = np.isfinite(E)
    assert np.sum(valid) > 10

    x_back, y_back = enuToXYPP(E, N, U, platepar)
    assert np.nanmax(np.hypot(x_ref - x_back, y_ref - y_back)) < 1e-4

    x_back, y_back = enHtToXYPP(E, N, np.full(len(E), ht_m), platepar)
    assert np.nanmax(np.hypot(x_ref - x_back, y_ref - y_back)) < 1e-4

    lat, lon, h = enu2LatLonAlt(E, N, U, platepar.lat, platepar.lon, platepar.height_wgs84)
    x_back, y_back = geoToXYPP(lat, lon, h, platepar)
    assert np.nanmax(np.hypot(x_ref - x_back, y_ref - y_back)) < 1e-4

    # The same pixels are NaN in all of them
    assert np.array_equal(np.isfinite(x_back), valid)


def test_geo_paths_agree():
    """ geoToXYPP and geoToENUPP -> enuToXYPP land on the same pixels. """

    platepar = pointedPlatepar(TEMPLATES[0], 69.0, 110.0, 0.0)

    lat = platepar.lat + np.array([0.05, 0.10, -0.05, 0.02, -0.08])
    lon = platepar.lon + np.array([0.10, -0.05, 0.08, 0.15, -0.10])
    ht = np.full(len(lat), 11000.0)

    x_direct, y_direct = geoToXYPP(lat, lon, ht, platepar)
    x_via, y_via = enuToXYPP(*geoToENUPP(lat, lon, ht, platepar), platepar)

    assert np.all(np.isfinite(x_direct))
    assert np.max(np.hypot(x_direct - x_via, y_direct - y_via)) < 1e-6


def test_points_are_at_the_height_on_the_line_of_sight():
    """ The points of xyHtToENUPP, taken to ECEF with axes written out here, are at the requested height above the
        ellipsoid and in the direction returned with them. """

    platepar = pointedPlatepar(TEMPLATES[0], 36.0, 110.0, 0.0)
    x_data, y_data = frameGrid(platepar)
    ht_m = 11000.0

    E, N, U, Eu, Nu, Uu, az, el = xyHtToENUPP(x_data, y_data, ht_m, platepar)
    valid = np.isfinite(E)
    assert np.sum(valid) > 30

    # ECEF of the points from the station and its axes
    east, north, up = enuAxesByHand(platepar.lat, platepar.lon)
    station = np.array(latLonAlt2ECEF(math.radians(platepar.lat), math.radians(platepar.lon),
        platepar.height_wgs84))
    points = station + np.outer(E[valid], east) + np.outer(N[valid], north) + np.outer(U[valid], up)

    _, _, h = ecef2LatLonAlt(points[:, 0], points[:, 1], points[:, 2])
    assert np.max(np.abs(h - ht_m)) < 1e-3

    # The exact conversion of ERFA, independent of RMS
    if erfa is not None:
        _, _, h_erfa = erfa.gc2gd(1, points)
        assert np.max(np.abs(h_erfa - ht_m)) < 1e-3

    # The returned direction is the one of the point
    distance = np.sqrt(E**2 + N**2 + U**2)
    assert np.allclose(Eu[valid], E[valid]/distance[valid], atol=1e-12)
    assert np.allclose(Uu[valid], U[valid]/distance[valid], atol=1e-12)
    assert np.allclose(az[valid], np.arctan2(E[valid], N[valid]), atol=1e-12)
    assert np.allclose(el[valid], np.arcsin(Uu[valid]), atol=1e-12)


def test_geo_to_enu_against_ecef():
    """ geoToENUPP is the ECEF difference along the local axes of the station, and enu2LatLonAlt inverts it.
        Scalars give floats. """

    platepar = pointedPlatepar(TEMPLATES[0], 36.0, 110.0, 0.0)

    lat = platepar.lat + np.array([0.5, -1.0, 0.2, 2.0])
    lon = platepar.lon + np.array([-0.3, 0.7, 1.5, -2.0])
    h = np.array([500.0, 11000.0, 15000.0, -20.0])

    E, N, U = geoToENUPP(lat, lon, h, platepar)

    east, north, up = enuAxesByHand(platepar.lat, platepar.lon)
    station = np.array(latLonAlt2ECEF(math.radians(platepar.lat), math.radians(platepar.lon),
        platepar.height_wgs84))
    for i in range(len(lat)):
        diff = np.array(latLonAlt2ECEF(math.radians(lat[i]), math.radians(lon[i]), h[i])) - station
        assert np.allclose([E[i], N[i], U[i]], [diff.dot(east), diff.dot(north), diff.dot(up)], atol=1e-6)

    lat_back, lon_back, h_back = enu2LatLonAlt(E, N, U, platepar.lat, platepar.lon, platepar.height_wgs84)
    assert np.max(np.abs(lat_back - lat)) < 1e-9
    assert np.max(np.abs(lon_back - lon)) < 1e-9
    assert np.max(np.abs(h_back - h)) < 1e-3

    # Scalars in, floats out
    out = geoToENUPP(float(lat[1]), float(lon[1]), float(h[1]), platepar)
    assert all(isinstance(v, float) for v in out)
    assert np.allclose(out, [E[1], N[1], U[1]])


@pytest.mark.parametrize("ht_target", [5000.0, 10000.0])
def test_line_of_sight_bends_like_the_light(ht_target):
    """ The point at a height is where the light seen at the pixel comes from: the straight line to it is the one
        a ray traced through the standard atmosphere from the arrival direction (xyToAltAzPP) reaches the height
        on, to 2% of the refraction of a star. Taking out the refraction of a star instead misses it by more than
        40% of it. """

    platepar = pointedPlatepar(TEMPLATES[0], 20.0, 110.0, 0.0)
    platepar.elev = 0.0
    x_data, y_data = np.array([640.0, 640.0, 640.0]), np.array([360.0, 500.0, 600.0])

    # The direction the light arrives from, and the direction of a star seen there
    alt_app, _ = xyToAltAzPP(x_data, y_data, platepar)
    alt_star = np.degrees([pyRefractionApparentToTrue(np.radians(a), refractionScale(0.0)) for a in alt_app])

    # Heights above sea level: the geoid height of the station
    geoid_height = platepar.height_wgs84 - platepar.elev
    el = np.degrees(xyHtToENUPP(x_data, y_data, ht_target + geoid_height, platepar)[7])

    for i in range(len(x_data)):

        elev_target, elev_star = _rayTrace(alt_app[i], 0.0, ht_target)
        refraction_star = alt_app[i] - elev_star

        assert abs(el[i] - elev_target) < 0.02*refraction_star
        assert abs(alt_star[i] - elev_target) > 0.4*refraction_star


def test_enht0_to_enht1_follows_the_pixel():
    """ Moving a point to another height keeps it on the pixel it is seen at: the result is xyHtToENUPP of the
        pixel at the new height, the bearing does not change, moving it back returns it, and scalar heights apply
        to every point. """

    platepar = pointedPlatepar(TEMPLATES[0], 36.0, 110.0, 0.0)
    x_data, y_data = frameGrid(platepar, n=5)

    E0, N0 = xyHtToENUPP(x_data, y_data, 11000.0, platepar)[:2]
    valid = np.isfinite(E0)

    for ht1 in (2000.0, 8000.0, 15000.0):

        E1, N1, U1 = enHt0ToEnHt1(E0[valid], N0[valid], 11000.0, ht1, platepar)
        E1_ref, N1_ref, U1_ref = xyHtToENUPP(x_data[valid], y_data[valid], ht1, platepar)[:3]

        assert np.max(np.sqrt((E1 - E1_ref)**2 + (N1 - N1_ref)**2 + (U1 - U1_ref)**2)) < 0.01

        bearing_drift = (np.degrees(np.arctan2(E0[valid], N0[valid]) - np.arctan2(E1, N1)) + 180)%360 - 180
        assert np.max(np.abs(bearing_drift)) < 1e-9

        # And back to the first height
        E0_back, N0_back, _ = enHt0ToEnHt1(E1, N1, ht1, 11000.0, platepar)
        assert np.max(np.hypot(E0_back - E0[valid], N0_back - N0[valid])) < 0.01


def test_elevation_limit_and_unreachable_heights():
    """ Lines of sight below the elevation limit are NaN both ways, and a height below the station is never
        reached by a line of sight looking up (no point behind the station is returned). """

    platepar = pointedPlatepar(TEMPLATES[0], 15.0, 110.0, 0.0)
    x_data, y_data = frameGrid(platepar)

    E, N, U, _, _, _, _, el = xyHtToENUPP(x_data, y_data, 11000.0, platepar, min_el_deg=10.0)
    low = ~np.isfinite(E)
    assert 5 < np.sum(low) < len(E) - 5
    assert np.nanmin(np.degrees(el)) >= 10.0

    # The same points are NaN on the way back, given by ENU, by height or by geodetic coordinates
    E_all, N_all, U_all = xyHtToENUPP(x_data, y_data, 11000.0, platepar, min_el_deg=-5.0)[:3]
    lat, lon, h = enu2LatLonAlt(E_all, N_all, U_all, platepar.lat, platepar.lon, platepar.height_wgs84)
    for x_back, _ in (enuToXYPP(E_all, N_all, U_all, platepar, min_el_deg=10.0),
                      enHtToXYPP(E_all, N_all, 11000.0, platepar, min_el_deg=10.0),
                      geoToXYPP(lat, lon, h, platepar, min_el_deg=10.0)):
        assert np.array_equal(~np.isfinite(x_back), low)

    # A height below the station
    platepar.elev = 2000.0
    assert np.all(np.isnan(xyHtToENUPP(x_data, y_data, 1000.0, platepar)[0]))


def test_inputs_like_the_contrail_pipeline():
    """ Lists and scalar heights are accepted, and a platepar reached through a wrapper that forwards attributes
        (as the contrail pipeline uses) works and is left unchanged: nothing is stored on it. """

    class ForwardingPlatepar(object):
        """ Forwards reading the attributes of a platepar and refuses writing them. """

        def __init__(self, platepar):
            object.__setattr__(self, 'platepar', platepar)

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, 'platepar'), name)

        def __setattr__(self, name, value):
            raise AttributeError("read only: {:s}".format(name))

    platepar = pointedPlatepar(TEMPLATES[0], 36.0, 110.0, 0.0)
    platepar.exact_reverse = True
    state = copy.deepcopy(platepar.__dict__)
    json_before = platepar.jsonStr()
    wrapped = ForwardingPlatepar(platepar)

    x_list, y_list = [300.0, 640.0, 900.0], [200.0, 360.0, 500.0]

    alt, azim = xyToAltAzPP(x_list, y_list, wrapped)
    E, N, U, _, _, _, _, _ = xyHtToENUPP(x_list, y_list, 11000.0, wrapped)
    E_lst = xyHtToENUPP(x_list, y_list, [11000.0]*3, wrapped)[0]
    assert np.allclose(E, E_lst)

    lat, lon, h = enu2LatLonAlt(E, N, U, platepar.lat, platepar.lon, platepar.height_wgs84)
    for x_back, y_back in (enuToXYPP(list(E), list(N), list(U), wrapped), enHtToXYPP(E, N, 11000.0, wrapped),
                           geoToXYPP(list(lat), list(lon), list(h), wrapped)):
        assert np.max(np.hypot(x_back - x_list, y_back - y_list)) < 1e-3

    assert np.allclose(geoToENUPP(lat, lon, h, wrapped), [E, N, U], atol=1e-6)
    E1, N1, U1 = enHt0ToEnHt1(list(E), list(N), 11000.0, 9000.0, wrapped)
    assert len(E1) == 3

    # Nothing was stored on the platepar, nor would be written out
    assert platepar.__dict__.keys() == state.keys()
    assert platepar.jsonStr() == json_before


def test_height_wgs84():
    """ The station height above the ellipsoid comes from the height above sea level through EGM96 and follows it;
        an assigned value overrides it and None derives it again; it is never written into the platepar file, and
        a value in a file is not taken. """

    platepar = pointedPlatepar(TEMPLATES[0], 36.0, 110.0, 0.0)

    expected = mslToWGS84Height(math.radians(platepar.lat), math.radians(platepar.lon), platepar.elev)
    assert platepar.height_wgs84 == expected

    # The geoid is about 36 m below the ellipsoid at the station
    assert -40 < platepar.height_wgs84 - platepar.elev < -30

    # Follows the height above sea level
    platepar.elev += 1000.0
    assert abs(platepar.height_wgs84 - expected - 1000.0) < 1e-9

    # An assigned value overrides it, also in a copy, and None derives it again
    platepar.height_wgs84 = 123.0
    assert platepar.height_wgs84 == 123.0
    assert copy.deepcopy(platepar).height_wgs84 == 123.0
    assert 'height_wgs84' not in json.loads(platepar.jsonStr())
    platepar.height_wgs84 = None
    assert abs(platepar.height_wgs84 - expected - 1000.0) < 1e-9

    # Not written, and a value in a file is not taken
    platepar_dict = json.loads(platepar.jsonStr())
    assert 'height_wgs84' not in platepar_dict
    platepar_dict['height_wgs84'] = 5.0
    loaded = Platepar()
    loaded.loadFromDict(platepar_dict)
    assert abs(loaded.height_wgs84 - expected - 1000.0) < 1e-9
    assert 'height_wgs84' not in json.loads(loaded.jsonStr())


def test_geoid_model_default_and_cached():
    """ Without a configuration the EGM96 file shipped with RMS is used, giving the same heights as with the
        configuration, and the interpolated model is built once per file. """

    config = cr.Config()
    lat, lon = math.radians(43.2), math.radians(-81.3)

    h_default = mslToWGS84Height(lat, lon, 300.0)
    assert h_default == mslToWGS84Height(lat, lon, 300.0, config)
    assert abs(wgs84toMSLHeight(lat, lon, h_default) - 300.0) < 1e-9

    model = GeoidHeightEGM96.geoidModel()
    assert GeoidHeightEGM96.geoidModel(config) is model
    assert GeoidHeightEGM96.egm96FilePath() == GeoidHeightEGM96.egm96FilePath(config)


def test_write_cal_with_config(tmp_path):
    """ Guard (unchanged behaviour): the CAL file for CAMS-compatible stations is written with the WGS84 height of
        the station, the geoid file taken from the configuration. """

    config = cr.Config()
    config.cams_code = 1
    platepar = pointedPlatepar(TEMPLATES[0], 36.0, 110.0, 0.0)

    night_dir = tmp_path/'XX0001_20261119_010203_456789'
    night_dir.mkdir()
    file_name = writeCAL(str(night_dir), config, platepar)

    content = (night_dir/file_name).read_text()
    height = mslToWGS84Height(math.radians(platepar.lat), math.radians(platepar.lon), platepar.elev, config)
    assert '{:.1f}'.format(height/1000)[:4] in content or '{:.3f}'.format(height/1000) in content


def test_directions_without_photometry():
    """ xyToRaDecPP without levels gives the same directions and no magnitudes. """

    platepar = pointedPlatepar(TEMPLATES[0], 36.0, 110.0, 0.0)
    x_data, y_data = frameGrid(platepar)
    jd_data = np.full(len(x_data), platepar.JD)

    _, ra, dec, mag = xyToRaDecPP(jd_data, x_data, y_data, np.ones(len(x_data)), platepar, jd_time=True)
    _, ra_none, dec_none, mag_none = xyToRaDecPP(jd_data, x_data, y_data, None, platepar, jd_time=True)

    assert np.array_equal(ra, ra_none) and np.array_equal(dec, dec_none)
    assert mag_none is None and len(mag) == len(x_data)


def test_target_refraction_limits():
    """ The refraction of a target: none at the station height, that of a star far above the atmosphere, and in
        between refractionTargetFraction() of it (0.377 at 10 km seen from sea level). """

    platepar = pointedPlatepar(TEMPLATES[0], 36.0, 110.0, 0.0)
    geoid_height = platepar.height_wgs84 - platepar.elev
    elev_app = np.array([2.0, 10.0, 30.0, 70.0])
    scale = refractionScale(platepar.elev)
    star = np.degrees([pyRefractionApparentToTrue(np.radians(e), scale) for e in elev_app])

    # At the station height: no refraction; far away: that of a star
    assert np.allclose(targetRefractionApparentToTrue(elev_app, platepar, ht_data=platepar.height_wgs84),
        elev_app, atol=1e-12)
    assert np.allclose(targetRefractionApparentToTrue(elev_app, platepar, ht_data=1.0e12), star, atol=1e-9)
    assert np.allclose(targetRefractionApparentToTrue(elev_app, platepar, range_data=1.0e12, azim_data=110.0),
        star, atol=1e-9)

    # In between, the fraction of the target height
    for ht in (2000.0, 10000.0, 30000.0):
        elev = targetRefractionApparentToTrue(elev_app, platepar, ht_data=ht + geoid_height)
        fraction = (elev_app - elev)/(elev_app - star)
        assert np.allclose(fraction, refractionTargetFraction(platepar.elev, ht), atol=1e-4)

    # From sea level
    platepar.elev = 0.0
    elev = targetRefractionApparentToTrue(elev_app, platepar, ht_data=10000.0 + geoid_height)
    star = np.degrees([pyRefractionApparentToTrue(np.radians(e), 1.0) for e in elev_app])
    assert np.allclose((elev_app - elev)/(elev_app - star), 0.377, atol=0.01)


def test_target_refraction_grows_with_the_distance():
    """ The refraction of a target grows with its height and with its range, the two give the same where they
        describe the same point, and the two directions invert each other. """

    platepar = pointedPlatepar(TEMPLATES[0], 36.0, 110.0, 0.0)
    elev_app = 10.0

    # Monotonic in height and in range
    heights = np.linspace(platepar.height_wgs84, 50000.0, 30)
    ranges = np.linspace(10.0, 300000.0, 30)
    by_height = targetRefractionApparentToTrue(np.full(30, elev_app), platepar, ht_data=heights)
    by_range = targetRefractionApparentToTrue(elev_app, platepar, range_data=ranges, azim_data=30.0)
    assert np.all(np.diff(by_height) < 0) and np.all(np.diff(by_range) < 0)

    # The ranges of points at 11 km on the lines give the same elevations as their heights (both on the WGS84
    #   ellipsoid)
    azim = np.array([0.0, 75.0, 160.0, 290.0])
    elev = targetRefractionApparentToTrue(np.full(4, elev_app), platepar, ht_data=11000.0)
    E, N, U = cyAltAzHeight2ENU(np.radians(azim), np.radians(elev), np.full(4, 11000.0), platepar.lat,
        platepar.lon, platepar.height_wgs84)
    elev_range = targetRefractionApparentToTrue(elev_app, platepar, range_data=np.sqrt(E**2 + N**2 + U**2),
        azim_data=azim)
    assert np.max(np.abs(elev_range - elev)) < 1e-6/3600

    # Inverse pairs, scalars in and floats out
    for kwargs in ({'ht_data': 8000.0}, {'range_data': 50000.0, 'azim_data': 30.0}):
        elev = targetRefractionApparentToTrue(elev_app, platepar, **kwargs)
        assert isinstance(elev, float)
        assert abs(targetRefractionTrueToApparent(elev, platepar, **kwargs) - elev_app) < 1e-9

    # Either the height or the range, the range with the azimuth
    with pytest.raises(ValueError):
        targetRefractionApparentToTrue(elev_app, platepar)
    with pytest.raises(ValueError):
        targetRefractionApparentToTrue(elev_app, platepar, ht_data=1.0, range_data=1.0, azim_data=0.0)
    with pytest.raises(ValueError):
        targetRefractionApparentToTrue(elev_app, platepar, range_data=1.0)


def test_target_refraction_kernel():
    """ cyTargetRefraction: the two directions invert each other, and an infinite height is a star. """

    elev = np.radians(np.array([-0.4, 2.0, 10.0, 30.0, 70.0]))

    for height in (300.0, 10.0e3, np.inf):

        heights = np.full(len(elev), height)
        back = cyTargetRefraction(cyTargetRefraction(elev, heights, 300.0), heights, 300.0, apparent_to_true=False)
        assert np.max(np.abs(back - elev)) < 1e-12

    star = np.array([pyRefractionApparentToTrue(e, refractionScale(300.0)) for e in elev])
    assert np.array_equal(cyTargetRefraction(elev, np.full(len(elev), np.inf), 300.0), star)


def test_geometry_kernels():
    """ The line of sight to a height agrees with rangeToHeightWGS84, and the Conversions wrappers of the ENU
        kernels take scalars and arrays. """

    azim = np.radians(np.linspace(0.0, 350.0, 36))
    elev = np.radians(np.linspace(-1.0, 89.0, 36))
    for ht in (1500.0, 11000.0, 400.0):
        E, N, U = cyAltAzHeight2ENU(azim, elev, np.full(36, ht), TEST_LAT, TEST_LON, 700.0)
        r = rangeToHeightWGS84(np.degrees(azim), np.degrees(elev), ht, TEST_LAT, TEST_LON, 700.0)
        assert np.array_equal(np.isfinite(E), np.isfinite(r))

        # Both land within 1 mm of the height, which on a line of sight 1 deg below the horizon is 1 mm in 260 km
        assert np.nanmax(np.abs(np.sqrt(E**2 + N**2 + U**2) - r)) < 1e-2

    # Scalars give floats, arrays keep their shape
    out = latLonAlt2ENU(TEST_LAT + 0.1, TEST_LON, 9000.0, TEST_LAT, TEST_LON, 300.0)
    assert all(isinstance(v, float) for v in out)
    grid = latLonAlt2ENU(np.full((2, 3), TEST_LAT + 0.1), TEST_LON, 9000.0, TEST_LAT, TEST_LON, 300.0)
    assert grid[0].shape == (2, 3)
    assert np.allclose(grid[1], out[1])
    u = enHeight2Up(out[0], out[1], 9000.0, TEST_LAT, TEST_LON, 300.0)
    assert isinstance(u, float) and abs(u - out[2]) < 1e-3


def test_horizontal_kernel_paths():
    """ The horizontal frame of xyToRaDecPP and raDecToXYPP is their RA/Dec path converted to alt/az (the
        refraction of a star put back), through the reverse distortion as well, and with the exact inverse the two
        invert each other. """

    platepar = pointedPlatepar(TEMPLATES[1], 36.0, 110.0, 25.0)
    x_data, y_data = frameGrid(platepar)
    jd_data = np.full(len(x_data), platepar.JD)

    _, azim, alt, _ = xyToRaDecPP(jd_data, x_data, y_data, None, platepar, jd_time=True, horizontal=True)
    alt_ref, azim_ref = referenceAltAz(platepar, x_data, y_data)
    assert np.max(separationArcsec(alt_ref, azim_ref, alt, azim)) < 1e-6

    # Through the reverse distortion as the RA/Dec path, and through the exact inverse back to the pixels
    _, ra, dec, _ = xyToRaDecPP(jd_data, x_data, y_data, None, platepar, jd_time=True, aberration=False)
    x_rev, y_rev = raDecToXYPP(azim, alt, platepar.JD, platepar, horizontal=True)
    x_ref, y_ref = raDecToXYPP(ra, dec, platepar.JD, platepar, aberration=False)
    assert np.max(np.hypot(x_rev - x_ref, y_rev - y_ref)) < 1e-6

    platepar.exact_reverse = True
    x_back, y_back = raDecToXYPP(azim, alt, platepar.JD, platepar, horizontal=True)
    assert np.max(np.hypot(x_back - x_data, y_back - y_data)) < 1e-4


def test_plates_without_refraction():
    """ A plate fitted to the stars without refraction gives their directions, so the refraction of a star is put
        back for the arrival direction; one whose measurements are corrected for refraction afterwards maps the
        arrival direction itself. Both invert exactly. """

    platepar = pointedPlatepar(TEMPLATES[0], 36.0, 110.0, 0.0)
    platepar.exact_reverse = True
    assert plateRefractionScale(platepar) == 0.0
    platepar.refraction = False
    assert plateRefractionScale(platepar) == refractionScale(platepar.elev)

    x_data, y_data = frameGrid(platepar)
    for flag in (False, True):

        platepar.measurement_apparent_to_true_refraction = flag
        alt, azim = xyToAltAzPP(x_data, y_data, platepar)
        alt_ref, azim_ref = referenceAltAz(platepar, x_data, y_data, refraction=not flag)
        assert np.max(separationArcsec(alt_ref, azim_ref, alt, azim)) < 1e-3

        E, N, U = xyHtToENUPP(x_data, y_data, 11000.0, platepar)[:3]
        x_back, y_back = enuToXYPP(E, N, U, platepar)
        assert np.nanmax(np.hypot(x_back - x_data, y_back - y_data)) < 1e-3

    assert plateRefractionScale(platepar) == 0.0
