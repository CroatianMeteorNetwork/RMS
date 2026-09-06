""" Precession + nutation between J2000 and the true equator and equinox of date, as used by the astrometric kernels.
    Run with: python -m unittest Tests.test_precession_nutation """

from __future__ import print_function, division, absolute_import

import os
import unittest

import numpy as np

from RMS.Astrometry.CyFunctions import (equatorialCoordAndRotPrecession, equatorialCoordPrecession,
    trueOfDateFromJ2000, j2000FromTrueOfDate, cyTrueRaDec2ApparentAltAz, pointingCorrection)
from RMS.Astrometry.Conversions import date2JD, JD2HourAngle
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP
from RMS.Formats.Platepar import Platepar
from RMS.Misc import getRmsRootDir

try:
    import erfa
except ImportError:
    erfa = None


J2000 = 2451545.0
TT_MINUS_UT = 69.2/86400.0
DATES = [date2JD(2021, 3, 1, 3, 0, 0), date2JD(2024, 9, 5, 3, 0, 0), date2JD(2026, 1, 15, 3, 0, 0),
    date2JD(2027, 7, 1, 3, 0, 0), date2JD(2029, 6, 1, 3, 0, 0)]


def unitVector(ra, dec):
    return np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])


def raDec(v):
    return np.arctan2(v[1], v[0])%(2*np.pi), np.arcsin(v[2]/np.linalg.norm(v))


def separationArcsec(ra1, dec1, ra2, dec2):
    # atan2 of the cross and dot products: arccos of the dot product alone cannot resolve below ~4 mas
    v1 = unitVector(ra1, dec1)
    v2 = unitVector(ra2, dec2)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2)))*3600


def randomDirections(n, seed=0):
    rng = np.random.default_rng(seed)
    return [(rng.uniform(0, 2*np.pi), np.arcsin(rng.uniform(-0.98, 0.98))) for _ in range(n)]


class TestPrecessionNutation(unittest.TestCase):

    def testRoundTrip(self):
        """ J2000 -> date -> J2000 closes, with both the kernel transform and the converters. """

        for jd in DATES:
            for ra, dec in randomDirections(100, 1):
                r1, d1, rot1 = equatorialCoordAndRotPrecession(J2000, jd, ra, dec, 0.3)
                r2, d2, rot2 = equatorialCoordAndRotPrecession(jd, J2000, r1, d1, rot1)
                self.assertLess(separationArcsec(ra, dec, r2, d2), 0.01)
                self.assertLess(abs(rot2 - 0.3)*206265, 0.01)

                r3, d3 = j2000FromTrueOfDate(jd, *trueOfDateFromJ2000(jd, ra, dec))
                self.assertLess(separationArcsec(ra, dec, r3, d3), 0.01)


    def testKernelAndConvertersAgree(self):
        """ The kernel transform and the scalar converters implement the same frame. """

        for jd in DATES:
            for ra, dec in randomDirections(50, 2):
                r1, d1, _ = equatorialCoordAndRotPrecession(jd, J2000, ra, dec, 0.0)
                r2, d2 = j2000FromTrueOfDate(jd, ra, dec)
                self.assertLess(separationArcsec(r1, d1, r2, d2), 0.001)


    def testNutationIsPresent(self):
        """ The true-of-date frame differs from the mean-of-date one by the nutation (up to ~17 arcsec). """

        jd = DATES[0]
        diffs = [separationArcsec(*trueOfDateFromJ2000(jd, ra, dec), *equatorialCoordPrecession(J2000, jd, ra, dec))
            for ra, dec in randomDirections(50, 3)]
        self.assertGreater(max(diffs), 5.0)
        self.assertLess(max(diffs), 25.0)


    @unittest.skipIf(erfa is None, "pyerfa not installed")
    def testAgainstErfa(self):
        """ Within 0.5 arcsec of the IAU 2006/2000A bias-precession-nutation (the 4-term nutation series limit). """

        for jd in DATES:
            pnm = erfa.pnm06a(jd + TT_MINUS_UT, 0.0)
            for ra, dec in randomDirections(100, 4):
                r_true, d_true = raDec(pnm.dot(unitVector(ra, dec)))          # J2000 -> true of date, erfa
                r1, d1 = trueOfDateFromJ2000(jd, ra, dec)
                self.assertLess(separationArcsec(r1, d1, r_true, d_true), 0.5, jd)
                r2, d2, _ = equatorialCoordAndRotPrecession(jd, J2000, r_true, d_true, 0.0)
                self.assertLess(separationArcsec(r2, d2, ra, dec), 0.5, jd)


    @unittest.skipIf(erfa is None, "pyerfa not installed")
    def testRotationAngleAgainstGeometry(self):
        """ The rotation change equals the angle between the two frames' north directions at the point. """

        for jd in DATES:
            pnm = erfa.pnm06a(jd + TT_MINUS_UT, 0.0)
            for ra, dec in randomDirections(50, 5):
                _, _, rot = equatorialCoordAndRotPrecession(jd, J2000, ra, dec, 0.0)
                north = np.array([-np.sin(dec)*np.cos(ra), -np.sin(dec)*np.sin(ra), np.cos(dec)])
                c2 = pnm.T.dot(unitVector(ra, dec)); n2 = pnm.T.dot(north)
                r2, d2 = raDec(c2)
                east2 = np.array([-np.sin(r2), np.cos(r2), 0.0])
                north2 = np.array([-np.sin(d2)*np.cos(r2), -np.sin(d2)*np.sin(r2), np.cos(d2)])
                self.assertLess(abs(rot - np.arctan2(n2.dot(east2), n2.dot(north2)))*206265, 0.5, jd)


    @unittest.skipIf(erfa is None, "pyerfa not installed")
    def testPointingCorrectionEndToEnd(self):
        """ A fixed camera: the kernel's J2000 centre matches the true direction to the sidereal-time and series
            limits, with no drift over a night. """

        lat, lon = np.radians(45.0), np.radians(15.0)
        az0, alt0 = np.radians(200.0), np.radians(45.0)
        sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
        R = np.array([[-so, -sl*co, cl*co], [co, -sl*so, cl*so], [0, cl, sl]])
        d_ecef = R.dot(np.array([np.cos(alt0)*np.sin(az0), np.cos(alt0)*np.cos(az0), np.sin(alt0)]))

        def rot3(t, v):
            c, s = np.cos(t), np.sin(t); return np.array([c*v[0] - s*v[1], s*v[0] + c*v[1], v[2]])

        def trueOfDate(jd):
            return raDec(rot3(erfa.gst06a(jd, 0.0, jd + TT_MINUS_UT, 0.0), d_ecef))

        def icrs(jd):
            return raDec(erfa.pnm06a(jd + TT_MINUS_UT, 0.0).T.dot(rot3(erfa.gst06a(jd, 0.0, jd + TT_MINUS_UT, 0.0), d_ecef)))

        jd_ref = date2JD(2026, 3, 1, 21, 0, 0)
        ra_d, dec_d = trueOfDate(jd_ref)
        h0 = np.radians(JD2HourAngle(jd_ref))
        errs = []
        for hours in np.arange(0, 10.01, 1.0):
            jd = jd_ref + hours/24.0
            r1, d1, _ = pointingCorrection(jd, lat, lon, h0, jd_ref, ra_d, dec_d, 0.0, False)
            errs.append(separationArcsec(r1, d1, *icrs(jd)))
        self.assertLess(max(errs), 2.0)
        self.assertLess(max(errs) - min(errs), 0.5)

# ---- A distortion-free synthetic camera, fixed in alt/az, checked at off-centre pixels ----------------------------

TEMPLATE = os.path.join(getRmsRootDir(), 'share', 'platepar_templates', 'template_generic_720p_4mm.cal')
SITE = (32.2, -110.9, 700.0)       # lat, lon (deg), elevation (m)
X_RES, Y_RES, FOV_DEG = 1280, 720, 82.0


def syntheticCamera(az, alt, rot):
    """ A distortion-free camera fixed in alt/az, with the kernels' projection (equidistant azimuthal about the image
        centre) built in the horizontal frame. Returns the pixel coordinates of a 9 x 7 grid over the frame and the
        ECEF unit vectors of their lines of sight. """

    lat, lon = np.radians(SITE[0]), np.radians(SITE[1])
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    enu_to_ecef = np.array([[-so, -sl*co, cl*co], [co, -sl*so, cl*so], [0, cl, sl]])

    a, h = np.radians(az), np.radians(alt)
    pointing = np.array([np.cos(h)*np.sin(a), np.cos(h)*np.cos(a), np.sin(h)])
    up = np.array([0, 0, 1.0]) - pointing[2]*pointing
    up /= np.linalg.norm(up)
    side = np.cross(pointing, up)

    f_scale = X_RES/FOV_DEG
    xs, ys = np.meshgrid(np.linspace(20, X_RES - 20, 9), np.linspace(20, Y_RES - 20, 7))
    X, Y = xs.ravel(), ys.ravel()
    dx, dy = X - X_RES/2, Y - Y_RES/2
    rho = np.radians(np.hypot(dx, dy)/f_scale)
    theta = np.pi/2 - np.radians(rot) + np.arctan2(dy, dx)
    d = np.cos(rho)[:, None]*pointing + np.sin(rho)[:, None]*(np.cos(theta)[:, None]*up
        + np.sin(theta)[:, None]*side)

    return X, Y, d.dot(enu_to_ecef.T), f_scale


def earthFrameJ2000(jd, d_ecef):
    """ J2000 (GCRS) directions of Earth-fixed lines of sight at jd: Earth rotation (GAST), then the IAU 2006/2000A
        precession-nutation, both from erfa. """

    tta = jd + TT_MINUS_UT
    gast = erfa.gst06a(jd, 0.0, tta, 0.0)
    pnm = erfa.pnm06a(tta, 0.0)
    r3 = np.array([[np.cos(gast), -np.sin(gast), 0], [np.sin(gast), np.cos(gast), 0], [0, 0, 1]])

    return d_ecef.dot(r3.T).dot(pnm)


def positionAngle(ra1, dec1, ra2, dec2):
    """ Position angle of (ra2, dec2) seen from (ra1, dec1), from north through east (radians). """

    return np.arctan2(np.sin(ra2 - ra1)*np.cos(dec2),
        np.cos(dec1)*np.sin(dec2) - np.sin(dec1)*np.cos(dec2)*np.cos(ra2 - ra1))


def syntheticPlatepar(jd0, X, Y, f_scale, j2000_dirs):
    """ Platepar of the synthetic camera referenced at jd0: the true pointing and rotation, expressed in the kernels'
        frame (true equator and equinox of date), no distortion, no refraction.

    Arguments:
        jd0: [float] Reference Julian date.
        X, Y: [ndarray] Grid pixel coordinates (the image centre and the point to its right must be in the grid).
        f_scale: [float] Pixel scale (px/deg).
        j2000_dirs: [ndarray] J2000 unit vectors of the grid's lines of sight at jd0 (the true pointing).
    """

    pp = Platepar()
    pp.read(TEMPLATE)
    pp.X_res, pp.Y_res, pp.F_scale = X_RES, Y_RES, f_scale
    pp.lat, pp.lon, pp.elev = SITE
    pp.resetDistortionParameters()

    # The template keeps a distortion-centre offset; put the projection centre on the image centre
    pp.x_poly_fwd[0] = pp.x_poly_fwd[1] = 0.0
    pp.refraction = False
    pp.measurement_apparent_to_true_refraction = False
    pp.JD = jd0
    pp.Ho = JD2HourAngle(jd0)

    ic = np.argmin(np.hypot(X - X_RES/2, Y - Y_RES/2))
    ix = np.argmin(np.hypot(X - X_RES/2 - 155, Y - Y_RES/2))
    ra_c, dec_c = trueOfDateFromJ2000(jd0, *raDec(j2000_dirs[ic]))
    ra_x, dec_x = trueOfDateFromJ2000(jd0, *raDec(j2000_dirs[ix]))
    pp.RA_d, pp.dec_d = np.degrees(ra_c), np.degrees(dec_c)

    # The kernels measure the image position angle as theta = pi/2 - pos_angle_ref + atan2(y, x), from north
    #   towards west (RA decreasing), so the point to the right of the centre (atan2 = 0) has position angle
    #   pos_angle_ref - pi/2 from north through east
    pp.pos_angle_ref = (90.0 + np.degrees(positionAngle(ra_c, dec_c, ra_x, dec_x)))%360

    return pp


def errorField(kernel_dirs, true_dirs, ic):
    """ Compare the kernel's directions with the true ones over the grid.

    Return:
        (centre, worst, roll, nonrigid): [tuple of floats] Error at the centre, worst error over the grid, the
            rigid rotation of the error field about the centre's line of sight (a pos_angle_ref error), and the
            largest residual after removing the best-fit rigid rotation (a scale or projection error). All arcsec.
    """

    sep = np.degrees(np.arctan2(np.linalg.norm(np.cross(kernel_dirs, true_dirs), axis=1),
        np.sum(kernel_dirs*true_dirs, axis=1)))*3600
    e = kernel_dirs - true_dirs
    A = np.concatenate([np.array([[0, t[2], -t[1]], [-t[2], 0, t[0]], [t[1], -t[0], 0]]) for t in true_dirs])
    w = np.linalg.lstsq(A, e.ravel(), rcond=None)[0]
    resid = e - np.cross(w, true_dirs)

    return sep[ic], sep.max(), np.degrees(w.dot(true_dirs[ic]))*3600, np.degrees(np.linalg.norm(resid, axis=1).max())*3600


class TestSyntheticCamera(unittest.TestCase):

    @unittest.skipIf(erfa is None, "pyerfa not installed")
    def testGridOverNight(self):
        """ A distortion-free camera fixed in alt/az, referenced at the start of the night: the kernel's XY -> J2000
            over a 9 x 7 grid on the 82 x 46 deg frame stays on the true (erfa) directions through the night. The
            off-centre points see rotation and scale errors that the centre alone cannot. """

        for az, alt in ((14.0, 44.0), (200.0, 30.0)):

            X, Y, d_ecef, f_scale = syntheticCamera(az, alt, 10.0)
            ic = np.argmin(np.hypot(X - X_RES/2, Y - Y_RES/2))

            for jd0 in DATES:

                pp = syntheticPlatepar(jd0, X, Y, f_scale, earthFrameJ2000(jd0, d_ecef))

                for hours in (0.0, 4.0, 8.0):

                    jd = jd0 + hours/24.0
                    truth = earthFrameJ2000(jd, d_ecef)
                    _, ra, dec, _ = xyToRaDecPP(len(X)*[jd], X, Y, np.ones(len(X)), pp, extinction_correction=False,
                        measurement=True, jd_time=True)
                    centre, worst, roll, nonrigid = errorField(unitVector(np.radians(ra), np.radians(dec)).T, truth, ic)

                    where = (az, alt, jd0, hours)
                    self.assertLess(centre, 0.5, where)
                    self.assertLess(worst, 0.5, where)
                    self.assertLess(abs(roll), 0.3, where)
                    self.assertLess(nonrigid, 0.1, where)


if __name__ == "__main__":
    unittest.main()
