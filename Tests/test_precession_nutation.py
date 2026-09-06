""" Precession + nutation between J2000 and the true equator and equinox of date, as used by the astrometric kernels.
    Run with: python -m unittest Tests.test_precession_nutation """

from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from RMS.Astrometry.CyFunctions import (equatorialCoordAndRotPrecession, equatorialCoordPrecession,
    trueOfDateFromJ2000, j2000FromTrueOfDate, cyTrueRaDec2ApparentAltAz, pointingCorrection)
from RMS.Astrometry.Conversions import date2JD, JD2HourAngle

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


if __name__ == "__main__":
    unittest.main()
