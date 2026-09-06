""" Refraction for targets at a finite height (meteors, contrails) in the alt/az, ENU and geodetic transforms:
    the true direction to a target depends on its height. Run with: python -m unittest Tests.test_target_refraction """

from __future__ import print_function, division, absolute_import

import copy
import math
import unittest

import numpy as np

from Tests.test_coordinate_transforms import makePlatepar, pixelGrid
from Tests.test_refraction import rayTrace
from RMS.Astrometry.ApplyAstrometry import (xyToAltAzPP, AltAzToXYPP, xyHtToENUPP, enHtToXYPP, enuToXYPP,
    xyToGeoPP, ENHt0ToENHt1, xyHt2Geo)
from RMS.Astrometry.CyFunctions import (refractionScale, refractionTargetFraction, pyRefractionTrueToApparent,
    pyRefractionApparentToTrue)

try:
    from RMS.Astrometry.GPUENHt import ENHt0ToENHt1_gpu, CUDA_AVAILABLE
    from numba import cuda
    GPU = CUDA_AVAILABLE and cuda.is_available()
except Exception:
    GPU = False


STATION_ELEV = 700.0        # m above sea level
CONTRAIL = 10000.0          # m above sea level


def platepar(alt=20.0):
    pp = makePlatepar(alt, 200.0, 0.0, refraction=True)
    pp.elev = STATION_ELEV
    return pp


def wgs84(h_msl, pp):
    """ Ellipsoidal height of a point at h_msl above sea level near the station. """
    return h_msl + (pp.height_wgs84 - pp.elev)


def enuAltitude(E, N, U):
    return np.degrees(np.arctan2(U, np.hypot(E, N)))


class TestTargetRefraction(unittest.TestCase):

    def testAltAzForATargetHeight(self):
        """ xyToAltAzPP with a target height applies the target fraction of the star refraction to each pixel
            (the centre stays star-referenced), AltAzToXYPP inverts it, and a contrail at 10 km comes out higher
            than a star on the same pixel by (1 - fraction) times the refraction. """

        pp = platepar(20.0)
        X, Y = pixelGrid(pp)
        alt_star, az_star = xyToAltAzPP(X, Y, pp)
        alt_c, az_c = xyToAltAzPP(X, Y, pp, target_height=CONTRAIL)
        scale = refractionScale(pp.elev)
        fraction = refractionTargetFraction(pp.elev, CONTRAIL)
        ok = np.isfinite(alt_star) & (alt_star > 3.0)
        self.assertGreater(ok.sum(), 20)

        for a_s, a_c in zip(alt_star[ok], alt_c[ok]):
            alt_app = pyRefractionTrueToApparent(math.radians(a_s), scale)
            expected = math.degrees(pyRefractionApparentToTrue(alt_app, scale*fraction))
            self.assertLess(abs(a_c - expected)*3600, 0.01)

        self.assertLess(np.max(np.abs(az_c - az_star)[ok])*3600, 0.01)
        self.assertGreater(np.min((alt_c - alt_star)[ok])*3600, 20.0)

        x, y = AltAzToXYPP(alt_c[ok], az_c[ok], pp, target_height=CONTRAIL)
        self.assertLess(np.max(np.hypot(x - X[ok], y - Y[ok])), 0.05)


    def testEnuAndGeodeticUseTheTargetHeight(self):
        """ The ENU-at-height and geodetic transforms refract for the target height they are given: the
            direction of the ENU point equals the target-aware alt/az, and their inverses close. """

        pp = platepar(20.0)
        X, Y = pixelGrid(pp)
        ht = wgs84(CONTRAIL, pp)
        E, N, U = xyHtToENUPP(X, Y, ht, pp)[:3]
        ok = np.isfinite(E)
        alt_c, _ = xyToAltAzPP(X, Y, pp, target_height=CONTRAIL)
        # The kernels take ellipsoidal heights, xyToAltAzPP heights above sea level: the geoid offset changes the
        #   fraction by ~0.3 arcsec here (under a metre on the ground)
        self.assertLess(np.max(np.abs(enuAltitude(E[ok], N[ok], U[ok]) - alt_c[ok]))*3600, 0.5)

        x, y = enHtToXYPP(E[ok], N[ok], ht, pp)
        self.assertLess(np.max(np.hypot(x - X[ok], y - Y[ok])), 0.03)
        x, y = enuToXYPP(E[ok], N[ok], U[ok], pp)
        self.assertLess(np.max(np.hypot(x - X[ok], y - Y[ok])), 0.15)

        lat, lon = xyToGeoPP(X[ok], Y[ok], ht, pp)
        lat2, lon2 = xyHt2Geo(pp, X[ok], Y[ok], ht)
        self.assertTrue(np.allclose(lat, lat2, atol=1e-12) and np.allclose(lon, lon2, atol=1e-12))

        # Against the star-refracted directions the target-aware ones sit higher, by tens of arcsec and more
        pp_star = copy.deepcopy(pp)
        alt_s, _ = xyToAltAzPP(X, Y, pp_star)
        self.assertGreater(np.min((alt_c - alt_s)[ok])*3600, 20.0)


    def testPixelIsInvariantAcrossHeights(self):
        """ Re-projecting a point from 8 to 12 km along its refracted line of sight lands on the same pixel; the
            straight (unrefracted) line does not, by the change of the target fraction with height. """

        pp = platepar(20.0)
        X, Y = pixelGrid(pp)
        h0, h1 = wgs84(8000.0, pp), wgs84(12000.0, pp)
        E0, N0 = xyHtToENUPP(X, Y, h0, pp)[:2]
        ok = np.isfinite(E0)
        E1, N1, U1 = ENHt0ToENHt1(E0[ok], N0[ok], h0, h1, pp)
        x, y = enHtToXYPP(E1, N1, h1, pp)
        good = np.hypot(x - X[ok], y - Y[ok])
        # Mean over the frame: the ENU-at-height solver has a 0.06 px residual of its own at one pixel
        self.assertLess(good.mean(), 0.03)

        pp_line = copy.deepcopy(pp)
        pp_line.refraction = False
        E1, N1, U1 = ENHt0ToENHt1(E0[ok], N0[ok], h0, h1, pp_line)
        x, y = enHtToXYPP(E1, N1, h1, pp)
        straight = np.hypot(x - X[ok], y - Y[ok])
        self.assertGreater(straight.mean(), 2*good.mean())


    def testGroundPositionAgainstRayTrace(self):
        """ For a contrail at 10 km seen between 15 and 40 deg, the target-aware direction places it within 15 m on
            the ground of a ray trace through the standard atmosphere; the star refraction is off by several times
            that. """

        pp = platepar(20.0)
        X, Y = pixelGrid(pp)
        alt_star, _ = xyToAltAzPP(X, Y, pp)
        alt_c, _ = xyToAltAzPP(X, Y, pp, target_height=CONTRAIL)
        scale = refractionScale(pp.elev)
        n = 0
        for a_s, a_c in zip(alt_star, alt_c):
            if not np.isfinite(a_s) or a_s < 15.0 or a_s > 40.0:
                continue
            alt_app = math.degrees(pyRefractionTrueToApparent(math.radians(a_s), scale))
            elev_target, elev_star = rayTrace(alt_app, STATION_ELEV, CONTRAIL)
            ground = lambda err_deg: abs(math.radians(err_deg))*(CONTRAIL - STATION_ELEV)/math.sin(math.radians(a_s))**2
            self.assertLess(ground(a_c - elev_target), 15.0, a_s)
            self.assertGreater(ground(a_s - elev_target), 3*ground(a_c - elev_target), a_s)
            n += 1
        self.assertGreater(n, 5)


    @unittest.skipUnless(GPU, "no CUDA GPU")
    def testGpuMatchesCpu(self):
        """ The GPU height-to-height re-projection matches the Cython kernel, with and without refraction. """

        pp = platepar(20.0)
        X, Y = pixelGrid(pp)
        h0, h1 = wgs84(8000.0, pp), wgs84(12000.0, pp)
        E0, N0 = xyHtToENUPP(X, Y, h0, pp)[:2]
        ok = np.isfinite(E0)
        for refraction in (True, False):
            pp_run = copy.deepcopy(pp)
            pp_run.refraction = refraction
            cpu = ENHt0ToENHt1(E0[ok], N0[ok], h0, h1, pp_run)
            gpu = ENHt0ToENHt1_gpu(E0[ok], N0[ok], np.full(ok.sum(), h0), np.full(ok.sum(), h1), pp_run)
            for c, g in zip(cpu, gpu):
                self.assertLess(np.max(np.abs(c - g)), 0.01, refraction)


if __name__ == "__main__":
    unittest.main()
