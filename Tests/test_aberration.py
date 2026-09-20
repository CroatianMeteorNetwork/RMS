""" Annual aberration in the astrometric kernels. Run with: python -m unittest Tests.test_aberration """

from __future__ import print_function, division, absolute_import

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


TEMPLATE = os.path.join(getRmsRootDir(), 'share', 'platepar_templates', 'template_generic_720p_4mm.cal')


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
        """ Within 0.1 arcsec of the IAU 2006 barycentric velocity (erfa.epv00 + erfa.ab). """

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
                self.assertLess(separationArcsec(ra_r, dec_r, ra_e, dec_e), 0.1, (y, m, d))


    def testStarsAndMeasurementsAreDifferentFrames(self):
        """ raDecToXYPP <-> xyToRaDecPP round-trip for stars, and a measurement (an object in the atmosphere,
            measurement=True) at the same pixel comes out at the star's apparent, aberrated direction. """

        pp = Platepar()
        pp.read(TEMPLATE)
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


    @unittest.skipIf(erfa is None, "pyerfa not installed")
    def testStarGridOverNight(self):
        """ The star fit as in real life, on a distortion-free camera fixed in alt/az and referenced at the start of
            the night: a star's catalog direction (erfa: velocity aberration removed) projects through xyToRaDecPP
            to the catalog direction over a 9 x 7 grid on the 82 x 46 deg frame, through the night, without
            rotation or scale errors that the centre alone cannot see. """

        def catalogDirections(jd, earth_frame_dirs):
            # Invert erfa.ab: the catalog directions of stars that appear along the given lines of sight
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
