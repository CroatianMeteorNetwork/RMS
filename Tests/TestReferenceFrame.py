""" The frame of the platepar reference pointing. The kernel precesses RA_d/dec_d from the observation date
    to J2000 before comparing with the catalog, so the star fit defines them as true-of-date coordinates.
    Everything that reads them outside the kernel has to use that frame, and everything that builds a
    direction from Earth-fixed geometry has to take nutation out on the way to J2000.
"""

import copy
import os

import pytest

np = pytest.importorskip("numpy")

from RMS.Formats.Platepar import Platepar
from RMS.Misc import getRmsRootDir
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, geoHt2RaDec, geoHt2XY
from RMS.Astrometry.Conversions import (date2JD, JD2HourAngle, latLonAlt2ECEF, trueOfDateRaDec2ApparentAltAz,
    apparentAltAz2TrueOfDateRaDec, trueOfDateRaDec2J2000, j2000RaDec2TrueOfDate)
from RMS.Astrometry.CyFunctions import pointingCorrection, trueOfDateFromJ2000, cyraDec2AltAz, \
    refractionScale, removeAberration
from RMS.EventMonitor import platepar2AltAz

TEMPLATE = os.path.join(getRmsRootDir(), 'share', 'platepar_templates', 'template_generic_720p_4mm.cal')
LAT, LON, ELEV = 31.69, -110.9, 2610.0
JD = date2JD(2026, 9, 5, 6, 0, 0)

# TT - UT1 in days; RMS uses UTC as UT1 and so does the reference model here
TT_MINUS_UT = 69.2/86400.0


def _separation(ra1, dec1, ra2, dec2):
    """ Helper: angular separation in arc seconds, from atan2 so that it resolves below a milliarcsecond. """

    r1, d1, r2, d2 = np.radians(ra1), np.radians(dec1), np.radians(ra2), np.radians(dec2)
    v1 = np.stack([np.cos(d1)*np.cos(r1), np.cos(d1)*np.sin(r1), np.sin(d1)], -1)
    v2 = np.stack([np.cos(d2)*np.cos(r2), np.cos(d2)*np.sin(r2), np.sin(d2)], -1)

    return np.degrees(np.arctan2(np.linalg.norm(np.cross(v1, v2), axis=-1), np.sum(v1*v2, axis=-1)))*3600


def _templatePlatepar(refraction):
    """ Helper: a distortion-free platepar at the test site. """

    pp = Platepar()
    pp.read(TEMPLATE)
    pp.lat, pp.lon, pp.elev = LAT, LON, ELEV
    pp.resetDistortionParameters()
    pp.x_poly_fwd[0] = pp.x_poly_fwd[1] = pp.x_poly_rev[0] = pp.x_poly_rev[1] = 0.0
    pp.refraction = refraction
    pp.JD, pp.Ho = JD, JD2HourAngle(JD)
    pp.az_centre, pp.alt_centre = 150.0, 25.0
    pp.updateRefRADec()

    return pp


@pytest.mark.parametrize("refraction", [False, True])
def test_reference_alt_az_is_the_kernels_pointing(refraction):
    """ The alt/az that updateRefAltAz() derives from RA_d/dec_d is exactly where the kernel points the
        reference: pointingCorrection() at the reference time, taken back to the true-of-date frame. Reading
        RA_d as J2000, as before, put this 8-19 arcmin off in 2026. """

    pp = _templatePlatepar(refraction)
    lat, lon = np.radians(pp.lat), np.radians(pp.lon)

    # The kernel scales the refraction with the station height, as updateRefAltAz() does
    ra_k, dec_k, _ = pointingCorrection(pp.JD, lat, lon, np.radians(pp.Ho), pp.JD,
        np.radians(pp.RA_d), np.radians(pp.dec_d), 0.0, refraction, refractionScale(pp.elev))
    ra_t, dec_t = trueOfDateFromJ2000(pp.JD, ra_k, dec_k)
    az_k, alt_k = np.degrees(cyraDec2AltAz(ra_t, dec_t, pp.JD, lat, lon))

    q = copy.deepcopy(pp)
    q.updateRefAltAz()

    assert abs(((az_k - q.az_centre + 180)%360 - 180)*3600*np.cos(np.radians(alt_k))) < 1.0e-3
    assert abs((alt_k - q.alt_centre)*3600) < 1.0e-3

    # The same conversion is what EventMonitor uses
    az_e, alt_e = platepar2AltAz(q)
    assert az_e == pytest.approx(q.az_centre, abs=1.0e-9)
    assert alt_e == pytest.approx(q.alt_centre, abs=1.0e-9)


def test_reference_conversions_round_trip():
    """ Alt/az -> RA_d/dec_d -> alt/az closes, and the J2000 conversions are inverses. """

    pp = _templatePlatepar(False)

    q = copy.deepcopy(pp)
    q.updateRefAltAz()
    assert abs(q.az_centre - 150.0)*3600 < 1.0e-6
    assert abs(q.alt_centre - 25.0)*3600 < 1.0e-6

    ra, dec = apparentAltAz2TrueOfDateRaDec(150.0, 25.0, JD, LAT, LON, refraction=False)
    az, alt = trueOfDateRaDec2ApparentAltAz(ra, dec, JD, LAT, LON, refraction=False)
    assert abs(az - 150.0)*3600 < 1.0e-6 and abs(alt - 25.0)*3600 < 1.0e-6

    ra_j, dec_j = trueOfDateRaDec2J2000(ra, dec, JD)
    ra_b, dec_b = j2000RaDec2TrueOfDate(ra_j, dec_j, JD)
    assert _separation(ra, dec, ra_b, dec_b) < 1.0e-6

    # Array forms agree with the scalar ones
    az_a, alt_a = trueOfDateRaDec2ApparentAltAz(np.array([ra, ra + 1.0]), np.array([dec, dec]), JD,
        LAT, LON, refraction=False)
    assert az_a[0] == pytest.approx(az, abs=1.0e-12) and alt_a[0] == pytest.approx(alt, abs=1.0e-12)


def test_fitted_pointing_matches_the_camera():
    """ Fit the pointing of a synthetic camera whose star directions come from erfa, then check that the
        alt/az derived from the fitted RA_d/dec_d is the camera's real pointing. """

    erfa = pytest.importorskip("erfa")

    az0, alt0 = 150.0, 25.0
    x_res, y_res, f_scale = 1280, 720, 1280/39.8
    la, lo = np.radians(LAT), np.radians(LON)
    enu_to_ecef = np.array([[-np.sin(lo), -np.sin(la)*np.cos(lo), np.cos(la)*np.cos(lo)],
        [np.cos(lo), -np.sin(la)*np.sin(lo), np.cos(la)*np.sin(lo)], [0, np.cos(la), np.sin(la)]])
    a, h = np.radians(az0), np.radians(alt0)
    pointing = np.array([np.cos(h)*np.sin(a), np.cos(h)*np.cos(a), np.sin(h)])
    up = np.array([0, 0, 1.0]) - pointing[2]*pointing
    up /= np.linalg.norm(up)
    side = np.cross(pointing, up)
    gast = erfa.gst06a(JD, 0.0, JD + TT_MINUS_UT, 0.0)
    pnm = erfa.pnm06a(JD + TT_MINUS_UT, 0.0)

    def j2000(x, y):
        dx, dy = x - x_res/2, y - y_res/2
        rho = np.radians(np.hypot(dx, dy)/f_scale)
        theta = np.pi/2 + np.arctan2(dy, dx)
        d = np.cos(rho)[:, None]*pointing + np.sin(rho)[:, None]*(np.cos(theta)[:, None]*up
            + np.sin(theta)[:, None]*side)
        ecef = d.dot(enu_to_ecef.T)
        tod = np.stack([ecef[:, 0]*np.cos(gast) - ecef[:, 1]*np.sin(gast),
            ecef[:, 0]*np.sin(gast) + ecef[:, 1]*np.cos(gast), ecef[:, 2]], -1)
        gcrs = tod.dot(pnm)
        ra = np.arctan2(gcrs[:, 1], gcrs[:, 0])%(2*np.pi)
        dec = np.arcsin(gcrs[:, 2])

        # The camera sees the aberrated direction; the catalog direction the fit is given has it taken out
        out = np.array([removeAberration(r, d, JD) for r, d in zip(ra, dec)])

        return np.degrees(out[:, 0]), np.degrees(out[:, 1])

    rng = np.random.default_rng(5)
    sx, sy = rng.uniform(10, x_res - 10, 200), rng.uniform(10, y_res - 10, 200)
    ra_s, dec_s = j2000(sx, sy)

    pp = _templatePlatepar(False)
    pp.X_res, pp.Y_res, pp.F_scale = x_res, y_res, f_scale
    pp.pos_angle_ref = 0.0
    pp.fitAstrometry(JD, np.column_stack([sx, sy, np.ones(200)]),
        np.column_stack([ra_s, dec_s, 5*np.ones(200)]), fit_only_pointing=True)

    _, ra_m, dec_m, _ = xyToRaDecPP(200*[JD], sx, sy, np.ones(200), pp, extinction_correction=False,
        jd_time=True)
    assert _separation(ra_m, dec_m, ra_s, dec_s).max() < 0.5

    pp.updateRefAltAz()
    assert abs(((pp.az_centre - az0 + 180)%360 - 180)*3600*np.cos(np.radians(alt0))) < 0.5
    assert abs((pp.alt_centre - alt0)*3600) < 0.5


def test_ground_reference_direction_against_erfa():
    """ geoHt2RaDec() builds a direction in Earth-fixed geometry; on the way to J2000 both the equation of
        the equinoxes and the nutation have to come out. Mean precession alone left 7-10 arcsec. """

    erfa = pytest.importorskip("erfa")

    pp = _templatePlatepar(False)
    rng = np.random.default_rng(1)
    obs = np.array(latLonAlt2ECEF(np.radians(LAT), np.radians(LON), ELEV))

    for jd in (date2JD(2021, 3, 1, 3, 0, 0), JD, date2JD(2029, 6, 1, 3, 0, 0)):
        gast = erfa.gst06a(jd, 0.0, jd + TT_MINUS_UT, 0.0)
        pnm = erfa.pnm06a(jd + TT_MINUS_UT, 0.0)
        for _ in range(20):
            tlat, tlon = LAT + rng.uniform(-0.5, 0.5), LON + rng.uniform(-0.5, 0.5)
            th = rng.uniform(1000, 4000)
            ra, dec = geoHt2RaDec(pp, jd, tlat, tlon, th)
            v = np.array(latLonAlt2ECEF(np.radians(tlat), np.radians(tlon), th)) - obs
            v = np.array([v[0]*np.cos(gast) - v[1]*np.sin(gast), v[0]*np.sin(gast) + v[1]*np.cos(gast),
                v[2]])
            v = pnm.T.dot(v)
            ra_e = np.degrees(np.arctan2(v[1], v[0]))%360
            dec_e = np.degrees(np.arcsin(v[2]/np.linalg.norm(v)))
            assert _separation(ra, dec, ra_e, dec_e) < 0.3


def test_ground_projection_is_consistent_at_any_time():
    """ geoHt2XY() used to evaluate the plate at J2000 so that of-date coordinates could pass as J2000. Now
        geoHt2RaDec() returns J2000 and the plate is evaluated at the same time, so the pixel is the same
        whichever time is used, and the direction repeats after one sidereal day. """

    pp = _templatePlatepar(False)
    tlat, tlon, th = LAT + 0.2, LON + 0.1, ELEV + 500.0

    x1, y1 = geoHt2XY(pp, tlat, tlon, th)

    # Same target through the two halves at another time: the sky has turned, the J2000 direction with it
    jd = pp.JD + 0.3
    ra, dec = geoHt2RaDec(pp, jd, tlat, tlon, th)
    from RMS.Astrometry.ApplyAstrometry import raDecToXYPP
    pp_g = copy.deepcopy(pp)
    x2, y2 = raDecToXYPP(np.array([ra]), np.array([dec]), jd, pp_g, aberration=False)
    assert np.hypot(x2[0] - x1[0], y2[0] - y1[0]) < 0.01

    # One sidereal day later the Earth-fixed baseline points the same way in space, to the precession of
    #   the frame over a day
    ra0, dec0 = geoHt2RaDec(pp, pp.JD, tlat, tlon, th)
    ra1, dec1 = geoHt2RaDec(pp, pp.JD + 0.99726957, tlat, tlon, th)
    assert _separation(ra0, dec0, ra1, dec1) < 1.0
