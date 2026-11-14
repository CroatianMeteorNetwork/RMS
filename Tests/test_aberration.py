""" Annual aberration in the astrometric kernels. Run with: python -m unittest Tests.test_aberration """

from __future__ import print_function, division, absolute_import

import json
import os
import unittest

import numpy as np

from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, raDecToXYPP
from RMS.Astrometry.CyFunctions import applyAberration, removeAberration
from RMS.Astrometry.Conversions import date2JD, jd2Date
from RMS.Misc import getRmsRootDir
from Tests.test_precession_nutation import (DATES, TT_MINUS_UT, X_RES, Y_RES, syntheticCamera, earthFrameJ2000,
    syntheticPlatepar, errorField, unitVector)

try:
    import erfa
except ImportError:
    erfa = None


# The template carries no aberration flag, so a platepar read from it does not model the aberration; the
#   tests that use it as a plate fitted with the aberration modelled turn the flag on
TEMPLATE = os.path.join(getRmsRootDir(), 'share', 'platepar_templates', 'template_generic_720p_4mm.cal')


def catalogDirections(jd, earth_frame_dirs):
    """ Helper: the catalog directions of the stars that appear along the given lines of sight (unit vectors,
        J2000 axes), by inverting erfa.ab with erfa's barycentric velocity of the Earth. """

    tta = jd + TT_MINUS_UT
    pvh, pvb = erfa.epv00(tta, 0.0)
    vel = pvb[1]*(erfa.DAU/erfa.DAYSEC)/erfa.CMPS
    bm1 = np.sqrt(1 - vel.dot(vel))
    out = []
    for v in earth_frame_dirs:
        q = v.copy()
        for _ in range(6):
            q = q - (erfa.ab(q, vel, 1.0, bm1) - v)
            q /= np.linalg.norm(q)
        out.append(q)

    return np.array(out)


def separationArcsec(ra1, dec1, ra2, dec2):
    # atan2 of the cross and dot products: arccos of the dot product alone cannot resolve below ~4 mas
    v1 = np.stack([np.cos(dec1)*np.cos(ra1), np.cos(dec1)*np.sin(ra1), np.sin(dec1)], axis=-1)
    v2 = np.stack([np.cos(dec2)*np.cos(ra2), np.cos(dec2)*np.sin(ra2), np.sin(dec2)], axis=-1)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(v1, v2), axis=-1), np.sum(v1*v2, axis=-1)))*3600


class TestAberration(unittest.TestCase):

    def testMagnitudeAndInverse(self):
        """ Shifts are at most the aberration constant (plus eccentricity), and removeAberration inverts them. """

        rng = np.random.default_rng(0)
        for jd in (date2JD(2026, 9, 4, 3, 0, 0), date2JD(2027, 1, 3, 3, 0, 0), date2JD(2027, 7, 4, 3, 0, 0)):
            shifts = []
            for _ in range(300):
                ra, dec = rng.uniform(0, 2*np.pi), np.arcsin(rng.uniform(-1, 1))
                ra1, dec1 = applyAberration(ra, dec, jd)
                ra2, dec2 = removeAberration(ra1, dec1, jd)
                shifts.append(separationArcsec(ra, dec, ra1, dec1))
                self.assertLess(separationArcsec(ra, dec, ra2, dec2), 0.01)

            # Max shift over the sky is the Earth's speed in units of c: 20.5" +- 1.7% over the year
            self.assertGreater(max(shifts), 20.0)
            self.assertLess(max(shifts), 21.0)


    @unittest.skipIf(erfa is None, "pyerfa not installed")
    def testAgainstErfa(self):
        """ Within 0.03 arcsec of the IAU 2006 barycentric velocity (erfa.epv00 + erfa.ab); 0.020 arcsec is
            the worst case found over 2020-2030. """

        rng = np.random.default_rng(1)
        for (y, m, d) in ((2026, 9, 4), (2026, 12, 1), (2027, 3, 15), (2027, 6, 20)):
            jd = date2JD(y, m, d, 3, 0, 0)
            pvh, pvb = erfa.epv00(jd + 69.2/86400.0, 0.0)
            v = pvb[1]*(erfa.DAU/erfa.DAYSEC)/erfa.CMPS
            bm1 = np.sqrt(1 - v.dot(v))
            for _ in range(100):
                ra, dec = rng.uniform(0, 2*np.pi), np.arcsin(rng.uniform(-1, 1))
                p = np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])
                q = erfa.ab(p, v, 1.0, bm1)
                ra_e, dec_e = np.arctan2(q[1], q[0])%(2*np.pi), np.arcsin(q[2])
                ra_r, dec_r = applyAberration(ra, dec, jd)
                self.assertLess(separationArcsec(ra_r, dec_r, ra_e, dec_e), 0.03, (y, m, d))


    def testStarsAndMeasurementsAreDifferentFrames(self):
        """ raDecToXYPP <-> xyToRaDecPP round-trip for stars, and a measurement (an object in the atmosphere,
            measurement=True) at the same pixel comes out at the star's apparent, aberrated direction. """

        pp = Platepar()
        pp.read(TEMPLATE)
        pp.aberration = True
        pp.lat, pp.lon, pp.elev = 40.0, -100.0, 300.0
        # High enough that the whole 82 x 46 deg field is above the horizon (the refraction functions are
        # only inverses of each other above it)
        pp.alt_centre, pp.az_centre, pp.pos_angle_ref = 50.0, 0.0, 15.0
        pp.updateRefRADec(skip_rot_update=True)

        # No lens distortion, so the forward and reverse mappings are exact inverses and any round-trip
        # residual is the kernels' own
        pp.resetDistortionParameters()

        x = np.linspace(30, pp.X_res - 30, 7)
        y = np.linspace(30, pp.Y_res - 30, 5)
        X, Y = [a.ravel() for a in np.meshgrid(x, y)]
        time_data = len(X)*[jd2Date(pp.JD)]

        # A star at this pixel: catalog direction
        _, ra_cat, dec_cat, _ = xyToRaDecPP(time_data, X, Y, np.ones(len(X)), pp, extinction_correction=False)
        # A meteor at the same pixel: geometric direction in the Earth's frame
        _, ra_geo, dec_geo, _ = xyToRaDecPP(time_data, X, Y, np.ones(len(X)), pp, extinction_correction=False,
            measurement=True)

        # Both project back to the pixel they came from (the refraction functions are inverses to ~0.015 px)
        x_cat, y_cat = raDecToXYPP(ra_cat, dec_cat, pp.JD, pp)
        x_geo, y_geo = raDecToXYPP(ra_geo, dec_geo, pp.JD, pp, measurement=True)
        self.assertLess(np.max(np.hypot(x_cat - X, y_cat - Y)), 0.05)
        self.assertLess(np.max(np.hypot(x_geo - X, y_geo - Y)), 0.05)

        # The meteor's direction is where the star appears, i.e. the catalog direction plus the aberration
        for rc, dc, rg, dg in zip(ra_cat, dec_cat, ra_geo, dec_geo):
            ra_app, dec_app = applyAberration(np.radians(rc), np.radians(dc), pp.JD)
            self.assertLess(separationArcsec(ra_app, dec_app, np.radians(rg), np.radians(dg)), 0.01)

        shift = separationArcsec(np.radians(ra_cat), np.radians(dec_cat), np.radians(ra_geo), np.radians(dec_geo))
        self.assertGreater(np.min(shift), 1.0)
        self.assertLess(np.max(shift), 21.0)


    def testDatePerPointEqualsOneCallPerPoint(self):
        """ raDecToXYPP with a Julian date per point gives the pixels of one call per point at its own date
            with the aberration modelled: the Earth velocity is taken at the date of every point. The dates
            span 30 days, over which the aberration of a star changes by up to 10 arcsec. """

        pp = Platepar()
        pp.read(TEMPLATE)
        pp.aberration = True
        pp.lat, pp.lon, pp.elev = 40.0, -100.0, 300.0
        pp.alt_centre, pp.az_centre, pp.pos_angle_ref = 50.0, 0.0, 15.0
        pp.updateRefRADec(skip_rot_update=True)

        # Catalog directions of the stars seen at a pixel grid, every point at its own date
        x = np.linspace(30, pp.X_res - 30, 7)
        y = np.linspace(30, pp.Y_res - 30, 5)
        X, Y = [a.ravel() for a in np.meshgrid(x, y)]
        jds = pp.JD + np.linspace(0, 30, len(X))
        _, ra, dec, _ = xyToRaDecPP(jds, X, Y, np.ones(len(X)), pp, extinction_correction=False, jd_time=True)

        # All points in one call, and one call per point
        x_arr, y_arr = raDecToXYPP(ra, dec, jds, pp)
        x_one = np.zeros_like(x_arr)
        y_one = np.zeros_like(y_arr)
        for i in range(len(X)):
            x_i, y_i = raDecToXYPP(ra[i:i + 1], dec[i:i + 1], jds[i], pp)
            x_one[i], y_one[i] = x_i[0], y_i[0]

        self.assertLess(np.max(np.hypot(x_arr - x_one, y_arr - y_one)), 1e-6)


    @unittest.skipIf(erfa is None, "pyerfa not installed")
    def testPlateparFlag(self):
        """ A platepar without the flag has absorbed the aberration, so none is applied to it; a full fit
            turns the flag on and it survives a write and read. """

        # New platepars model it, platepars read without the flag do not
        self.assertTrue(Platepar().aberration)
        pp = Platepar()
        pp.read(TEMPLATE)
        self.assertFalse(pp.aberration)

        pp.lat, pp.lon, pp.elev = 40.0, -100.0, 300.0
        pp.alt_centre, pp.az_centre, pp.pos_angle_ref = 50.0, 0.0, 15.0
        pp.updateRefRADec(skip_rot_update=True)
        pp.resetDistortionParameters()

        x = np.linspace(30, pp.X_res - 30, 7)
        y = np.linspace(30, pp.Y_res - 30, 5)
        X, Y = [a.ravel() for a in np.meshgrid(x, y)]
        time_data = len(X)*[jd2Date(pp.JD)]
        ones = np.ones(len(X))

        # Without the flag a star and a measurement at the same pixel are the same direction
        _, ra_cat, dec_cat, _ = xyToRaDecPP(time_data, X, Y, ones, pp, extinction_correction=False)
        _, ra_geo, dec_geo, _ = xyToRaDecPP(time_data, X, Y, ones, pp, extinction_correction=False,
            measurement=True)
        np.testing.assert_array_equal(ra_cat, ra_geo)
        np.testing.assert_array_equal(dec_cat, dec_geo)
        x_star, y_star = raDecToXYPP(ra_cat, dec_cat, pp.JD, pp)
        x_meas, y_meas = raDecToXYPP(ra_cat, dec_cat, pp.JD, pp, measurement=True)
        np.testing.assert_array_equal(x_star, x_meas)
        np.testing.assert_array_equal(y_star, y_meas)

        # A fit models the aberration, whether of the pointing alone or of the distortion as well
        img_stars = np.c_[X, Y, ones]
        catalog_stars = np.c_[ra_cat, dec_cat, ones]
        for fit_only_pointing in (True, False):
            pp.aberration = False
            pp.fitAstrometry(pp.JD, img_stars, catalog_stars, fit_only_pointing=fit_only_pointing)
            self.assertTrue(pp.aberration)

        # With the flag on, the two differ by the aberration
        _, ra_cat, dec_cat, _ = xyToRaDecPP(time_data, X, Y, ones, pp, extinction_correction=False)
        _, ra_geo, dec_geo, _ = xyToRaDecPP(time_data, X, Y, ones, pp, extinction_correction=False,
            measurement=True)
        sep = [separationArcsec(np.radians(a), np.radians(b), np.radians(c), np.radians(d)) \
            for a, b, c, d in zip(ra_cat, dec_cat, ra_geo, dec_geo)]
        self.assertGreater(min(sep), 1.0)

        # The flag is written to the file and read back
        pp_read = Platepar()
        pp_read.loadFromDict(json.loads(pp.jsonStr()))
        self.assertTrue(pp_read.aberration)


    @unittest.skipIf(erfa is None, "pyerfa not installed")
    def testFittedPlateMeasuresTheGeometricDirection(self):
        """ The star fit, then a measurement: a plate fitted to the catalog directions of the stars of a
            distortion-free camera fixed in alt/az (erfa: aberration removed from the lines of sight),
            starting from a pointing 0.1 deg off and without the aberration flag, measures a meteor at any
            pixel along the geometric line of sight of that pixel (measurement=True), which is not
            aberrated. """

        X, Y, d_ecef, f_scale = syntheticCamera(150.0, 40.0, 10.0)
        jd = DATES[2]
        lines_of_sight = earthFrameJ2000(jd, d_ecef)

        # The stars at the pixels of the grid: their catalog directions
        catalog = catalogDirections(jd, lines_of_sight)
        ra_cat = np.degrees(np.arctan2(catalog[:, 1], catalog[:, 0]))%360
        dec_cat = np.degrees(np.arcsin(catalog[:, 2]))

        # A plate of the camera with the pointing and the rotation 0.1 deg off, fitted to the stars
        pp = syntheticPlatepar(jd, X, Y, f_scale, lines_of_sight)

        # The fit goes through the reverse mapping, so its distortion-centre offset is taken off as well
        pp.x_poly_rev[0] = pp.x_poly_rev[1] = 0.0
        pp.RA_d += 0.1
        pp.dec_d -= 0.1
        pp.pos_angle_ref += 0.1
        pp.aberration = False
        ones = np.ones(len(X))
        pp.fitAstrometry(jd, np.c_[X, Y, ones], np.c_[ra_cat, dec_cat, ones], fit_only_pointing=True)
        self.assertTrue(pp.aberration)

        # A meteor at every pixel is measured along the line of sight of the pixel
        _, ra, dec, _ = xyToRaDecPP(len(X)*[jd], X, Y, ones, pp, extinction_correction=False,
            measurement=True, jd_time=True)
        sep = separationArcsec(np.radians(ra), np.radians(dec),
            np.arctan2(lines_of_sight[:, 1], lines_of_sight[:, 0]), np.arcsin(lines_of_sight[:, 2]))
        self.assertLess(np.max(sep), 0.02)


    def testPlateWithoutTheFlagGivesTheOutputsOfBefore(self):
        """ A platepar without the aberration flag (fitted before the aberration was modelled) gives the same
            outputs as the kernels without the aberration, to rounding, for stars and measurements alike. The
            values were computed with the kernels before the aberration was added. """

        pp = Platepar()
        pp.read(TEMPLATE)
        self.assertFalse(pp.aberration)
        pp.lat, pp.lon, pp.elev = 43.19, -81.32, 300.0
        pp.JD, pp.Ho = 2461000.5, 123.456
        pp.RA_d, pp.dec_d, pp.pos_angle_ref = 210.3, 35.2, 12.0
        pp.refraction = True
        jd = pp.JD + 0.1

        # Image -> sky, for stars and for measurements
        X = np.array([50.0, 640.0, 1230.0, 300.0, 1000.0])
        Y = np.array([40.0, 360.0, 680.0, 600.0, 120.0])
        ra_before = [249.10654097146963, 187.88242530004916, 140.97636634456615, 206.03977569909966,
            154.01999423770215]
        dec_before = [33.095384768115665, 37.806123507156826, 17.428652028987226, 15.428606996137892,
            56.31420121321367]
        for measurement in (False, True):
            _, ra, dec, _ = xyToRaDecPP(len(X)*[jd], X, Y, np.ones(len(X)), pp, extinction_correction=False,
                jd_time=True, measurement=measurement)
            np.testing.assert_allclose(ra, ra_before, rtol=0, atol=1e-11)
            np.testing.assert_allclose(dec, dec_before, rtol=0, atol=1e-11)

        # Sky -> image
        ra_c = np.array([200.0, 215.5, 190.0, 230.0, 205.0])
        dec_c = np.array([30.0, 40.0, 25.0, 45.0, 50.0])
        x_before = [454.57275796484555, 356.8874448629715, 560.3463150917519, 287.44068322266537,
            529.7649739583576]
        y_before = [425.8652471596938, 200.1094510630682, 544.605081820676, 54.25615351017734,
            126.57777939634846]
        for measurement in (False, True):
            x, y = raDecToXYPP(ra_c, dec_c, jd, pp, measurement=measurement)
            np.testing.assert_allclose(x, x_before, rtol=0, atol=1e-8)
            np.testing.assert_allclose(y, y_before, rtol=0, atol=1e-8)


    def testStarGridOverNight(self):
        """ The star fit as in real life, on a distortion-free camera fixed in alt/az and referenced at the start of
            the night: a star's catalog direction (erfa: velocity aberration removed) projects through xyToRaDecPP
            to the catalog direction over a 9 x 7 grid on the 82 x 46 deg frame, through the night, without
            rotation or scale errors that the centre alone cannot see. """

        for az, alt in ((14.0, 44.0), (200.0, 30.0)):

            X, Y, d_ecef, f_scale = syntheticCamera(az, alt, 10.0)
            ic = np.argmin(np.hypot(X - X_RES/2, Y - Y_RES/2))

            for jd0 in DATES:

                # The platepar holds the true pointing, which is what the star fit converges to with the aberration
                #   modelled
                pp = syntheticPlatepar(jd0, X, Y, f_scale, earthFrameJ2000(jd0, d_ecef))

                for hours in (0.0, 4.0, 8.0):

                    jd = jd0 + hours/24.0
                    truth = catalogDirections(jd, earthFrameJ2000(jd, d_ecef))
                    _, ra, dec, _ = xyToRaDecPP(len(X)*[jd], X, Y, np.ones(len(X)), pp, extinction_correction=False,
                        jd_time=True)
                    centre, worst, roll, nonrigid = errorField(unitVector(np.radians(ra), np.radians(dec)).T, truth, ic)

                    where = (az, alt, jd0, hours)
                    self.assertLess(centre, 0.5, where)
                    self.assertLess(worst, 0.5, where)
                    self.assertLess(abs(roll), 0.3, where)
                    self.assertLess(nonrigid, 0.1, where)


if __name__ == "__main__":
    unittest.main()
