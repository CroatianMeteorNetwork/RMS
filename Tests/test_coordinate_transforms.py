""" Consistency tests for the direct image <-> Alt/Az, ENU and geodetic transforms.

The star-calibrated path (xyToRaDecPP -> Alt/Az) is the reference: every direct transform must reproduce it
to the kernel floor for radial and polynomial distortion models, with refraction on and off, rolled and
zenith-pointing platepars. Run with: python -m unittest Tests.test_coordinate_transforms
"""

from __future__ import print_function, division, absolute_import

import os
import sys
import json
import copy
import unittest

import numpy as np

from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.ApplyAstrometry import (xyToRaDecPP, xyToAltAzPP, xyHtToENUPP, enHtToXYPP, enuToXYPP,
    geoToXYPP, geoToENUPP, xyToGeoPP, ENHt0ToENHt1, rotationWrtHorizon, rotationWrtHorizonToPosAngle)
from RMS.Astrometry.Conversions import jd2Date
from RMS.Astrometry.CyFunctions import cyTrueRaDec2ApparentAltAz
from RMS.GeoidHeightEGM96 import mslToWGS84Height, wgs84toMSLHeight, geoidUndulation
from RMS.Misc import getRmsRootDir


TEMPLATE = os.path.join(getRmsRootDir(), 'share', 'platepar_templates', 'template_generic_720p_4mm.cal')

# Station used for all synthetic platepars
LAT, LON, ELEV = 45.0, 15.0, 300.0

# Target height for the ENU tests (m, WGS84)
HT = 10000.0


def makePlatepar(alt_centre, az_centre, pos_angle_ref, refraction=True, poly3=False):
    """ Synthetic platepar from the shipped template, pointed and rolled as requested. """

    pp = Platepar()
    pp.read(TEMPLATE)
    pp.lat, pp.lon, pp.elev = LAT, LON, ELEV
    pp.refraction = refraction

    if poly3:
        # A polynomial model with realistic magnitudes: pixel offsets in the constant terms, a small
        # linear rotation/shear, weak higher-order terms
        pp.setDistortionType("poly3+radial", reset_params=True)
        pp.x_poly_fwd = np.array([0.5, -0.02, -0.007, 2e-6, -1e-6, 3e-6, 1e-9, 2e-9, -1e-9, 3e-9, 2e-5, -1e-5])
        pp.y_poly_fwd = np.array([-2.3, -0.002, -0.03, -1e-6, 2e-6, -3e-6, -2e-9, 1e-9, 2e-9, -1e-9, -1e-5, 2e-5])
        pp.x_poly_rev = pp.x_poly_fwd.copy()
        pp.y_poly_rev = pp.y_poly_fwd.copy()

    pp.alt_centre, pp.az_centre, pp.pos_angle_ref = float(alt_centre), float(az_centre), float(pos_angle_ref)
    pp.updateRefRADec(skip_rot_update=True)

    return pp


def pixelGrid(pp, nx=9, ny=7, margin=10):
    xs = np.linspace(margin, pp.X_res - margin, nx)
    ys = np.linspace(margin, pp.Y_res - margin, ny)
    X, Y = np.meshgrid(xs, ys)
    return X.ravel(), Y.ravel()


def referenceAltAz(pp, X, Y, refraction):
    """ Alt/Az of pixels through the star-calibrated path (radians). """

    jd_arr, ra, dec, _ = xyToRaDecPP(len(X)*[jd2Date(pp.JD)], X, Y, np.ones(len(X)), pp,
        extinction_correction=False)

    az = np.zeros(len(X))
    alt = np.zeros(len(X))
    for i in range(len(X)):
        az[i], alt[i] = cyTrueRaDec2ApparentAltAz(np.radians(ra[i]), np.radians(dec[i]), jd_arr[i],
            np.radians(pp.lat), np.radians(pp.lon), refraction)

    return az, alt


def separationArcmin(az1, alt1, az2, alt2):
    cos_sep = np.sin(alt1)*np.sin(alt2) + np.cos(alt1)*np.cos(alt2)*np.cos(az1 - az2)
    return np.degrees(np.arccos(np.clip(cos_sep, -1, 1)))*60


# Pointings covering low, mid, high and zenith cameras, with and without roll
POINTINGS = [(20, 90, 0), (45, 200, 20), (45, 200, 65), (65, 350, 30), (85, 180, 45), (89.5, 0, 45)]


class TestDirectTransforms(unittest.TestCase):

    def _platepars(self):
        for alt, az, pa in POINTINGS:
            for refraction in (True, False):
                yield "radial alt={} pa={} refr={}".format(alt, pa, refraction), \
                    makePlatepar(alt, az, pa, refraction=refraction)

        for refraction in (True, False):
            yield "poly3 alt=45 pa=20 refr={}".format(refraction), makePlatepar(45, 200, 20, refraction, poly3=True)


    def testAltAzMatchesStarCalibratedPath(self):
        """ xyToAltAzPP must agree with xyToRaDecPP -> Alt/Az over the whole frame. """

        for label, pp in self._platepars():
            X, Y = pixelGrid(pp)

            # The direct transforms return true altitudes, so compare against the geometric direction
            az_r, alt_r = referenceAltAz(pp, X, Y, False)
            alt_p, az_p = xyToAltAzPP(X, Y, pp)

            sep = separationArcmin(az_r, alt_r, np.radians(az_p), np.radians(alt_p))
            self.assertLess(np.max(sep), 0.5, "{}: max {:.2f} arcmin".format(label, np.max(sep)))


    def testRotationIsAtTheKernelFloor(self):
        """ The solved rotation must be the one that best aligns the kernel with the reference. """

        for label, pp in self._platepars():
            X, Y = pixelGrid(pp)
            az_r, alt_r = referenceAltAz(pp, X, Y, False)

            rot = rotationWrtHorizon(pp)

            # Perturbing the rotation by a tenth of a degree must make the agreement visibly worse
            worse = []
            for d_rot in (-0.1, 0.1):
                pp2 = copy.deepcopy(pp)
                pp2.pos_angle_ref = rotationWrtHorizonToPosAngle(pp2, rot + d_rot)
                alt_p, az_p = xyToAltAzPP(X, Y, pp2)
                worse.append(np.max(separationArcmin(az_r, alt_r, np.radians(az_p), np.radians(alt_p))))

            alt_p, az_p = xyToAltAzPP(X, Y, pp)
            best = np.max(separationArcmin(az_r, alt_r, np.radians(az_p), np.radians(alt_p)))
            self.assertLess(best, min(worse), label)


    def testENUAndGeodeticRoundTrips(self):
        """ XY -> ENU -> XY, XY -> geodetic -> XY and geodetic -> ENU must close to a few hundredths of a pixel. """

        for label, pp in self._platepars():
            X, Y = pixelGrid(pp)

            E, N, U = xyHtToENUPP(X, Y, HT, pp)[:3]
            ok = np.isfinite(E)
            self.assertGreater(ok.sum(), len(X)//2, label)

            x2, y2 = enHtToXYPP(E[ok], N[ok], HT, pp)
            self.assertLess(np.max(np.hypot(x2 - X[ok], y2 - Y[ok])), 0.1, label)

            x3, y3 = enuToXYPP(E[ok], N[ok], U[ok], pp)
            self.assertLess(np.max(np.hypot(x3 - X[ok], y3 - Y[ok])), 0.1, label)

            lat, lon = xyToGeoPP(X[ok], Y[ok], HT, pp)
            h = np.full(len(lat), HT)

            x4, y4 = geoToXYPP(lat, lon, h, pp)
            self.assertLess(np.max(np.hypot(x4 - X[ok], y4 - Y[ok])), 0.1, label)

            E2, N2, U2 = geoToENUPP(lat, lon, h, pp)
            self.assertLess(np.max(np.hypot(np.hypot(E2 - E[ok], N2 - N[ok]), U2 - U[ok])), 0.05, label)


    def testENUMatchesGeometricRayIntersection(self):
        """ XY -> ENU must land where the reference ray meets the target height (rotation-independent check). """

        a, f = 6378137.0, 1/298.257223563
        e2 = f*(2 - f)

        def geo2ecef(lat, lon, h):
            n = a/np.sqrt(1 - e2*np.sin(lat)**2)
            return np.array([(n + h)*np.cos(lat)*np.cos(lon), (n + h)*np.cos(lat)*np.sin(lon),
                (n*(1 - e2) + h)*np.sin(lat)])

        def ecefHeight(P):
            x, y, z = P
            p = np.hypot(x, y)
            lat = np.arctan2(z, p*(1 - e2))
            for _ in range(10):
                n = a/np.sqrt(1 - e2*np.sin(lat)**2)
                h = p/np.cos(lat) - n
                lat = np.arctan2(z, p*(1 - e2*n/(n + h)))
            n = a/np.sqrt(1 - e2*np.sin(lat)**2)
            return p/np.cos(lat) - n

        for label, pp in self._platepars():
            X, Y = pixelGrid(pp, nx=5, ny=3, margin=100)
            az_r, alt_r = referenceAltAz(pp, X, Y, False)
            E, N, U = xyHtToENUPP(X, Y, HT, pp)[:3]

            lat, lon = np.radians(pp.lat), np.radians(pp.lon)
            C = geo2ecef(lat, lon, pp.height_wgs84)
            sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
            R = np.array([[-so, -sl*co, cl*co], [co, -sl*so, cl*so], [0, cl, sl]])

            for i in range(len(X)):
                if alt_r[i] < np.radians(20) or not np.isfinite(E[i]):
                    continue
                d = np.array([np.cos(alt_r[i])*np.sin(az_r[i]), np.cos(alt_r[i])*np.cos(az_r[i]), np.sin(alt_r[i])])
                dE = R.dot(d)
                lo, hi = 0.0, 3.0e6
                for _ in range(60):
                    mid = 0.5*(lo + hi)
                    if ecefHeight(C + mid*dE) < HT:
                        lo = mid
                    else:
                        hi = mid
                ref = 0.5*(lo + hi)*d
                err = np.linalg.norm(ref - np.array([E[i], N[i], U[i]]))
                self.assertLess(err, 5.0, "{}: {:.1f} m at alt {:.1f}".format(label, err, np.degrees(alt_r[i])))


    def testHeightShiftKeepsLineOfSightAndRejectsUnreachable(self):
        """ ENHt0ToENHt1 slides along the line of sight; heights the ray never reaches give NaN. """

        pp = makePlatepar(45, 200, 20)
        X, Y = pixelGrid(pp, nx=5, ny=3, margin=100)
        E0, N0, U0 = xyHtToENUPP(X, Y, HT, pp)[:3]

        E1, N1, U1 = ENHt0ToENHt1(E0, N0, HT, 8000.0, pp)
        x1, y1 = enHtToXYPP(E1, N1, 8000.0, pp)
        self.assertLess(np.max(np.hypot(x1 - X, y1 - Y)), 0.1)

        # Scalar heights broadcast, and unreachable heights (below the station on an upward ray) give NaN
        E2, N2, U2 = ENHt0ToENHt1(E0, N0, HT, 100.0, pp)
        self.assertTrue(np.all(np.isnan(E2)) and np.all(np.isnan(U2)))

        x_nan, _ = enHtToXYPP(E0[:1], N0[:1], -2000.0, pp)
        self.assertTrue(np.isnan(x_nan[0]))


    def testScalarInputs(self):
        pp = makePlatepar(45, 200, 20)
        E, N, U = xyHtToENUPP(640.0, 360.0, HT, pp)[:3]
        self.assertEqual(E.shape, (1,))
        x, y = enuToXYPP(E[0], N[0], U[0], pp)
        self.assertAlmostEqual(float(x[0]), 640.0, delta=0.1)
        self.assertAlmostEqual(float(y[0]), 360.0, delta=0.1)



class TestStationHeight(unittest.TestCase):

    def testHeightWGS84IsDerivedAndNeverStale(self):
        pp = Platepar()
        self.assertTrue(np.isfinite(pp.height_wgs84))

        pp.lat, pp.lon, pp.elev = LAT, LON, ELEV
        undulation = geoidUndulation(np.radians(LAT), np.radians(LON))
        self.assertAlmostEqual(pp.height_wgs84, ELEV + undulation, places=6)

        pp.elev += 100.0
        self.assertAlmostEqual(pp.height_wgs84, ELEV + 100.0 + undulation, places=6)

        # An explicit assignment overrides the derived value for this object only, and is never persisted
        pp.height_wgs84 = 999.0
        self.assertEqual(pp.height_wgs84, 999.0)
        d = json.loads(pp.jsonStr())
        self.assertNotIn('height_wgs84', d)
        self.assertNotIn('_height_wgs84_override', d)
        pp.height_wgs84 = None
        self.assertAlmostEqual(pp.height_wgs84, ELEV + 100.0 + undulation, places=6)

        # A stale value stored in a file is ignored
        d['height_wgs84'] = 12345.0
        pp2 = Platepar()
        pp2.loadFromDict(d)
        self.assertAlmostEqual(pp2.height_wgs84, ELEV + 100.0 + undulation, places=6)


    def testGeoidAcceptsPathConfigAndDefault(self):
        import RMS.ConfigReader as cr

        lat, lon = np.radians(LAT), np.radians(LON)
        config = cr.Config()
        h_default = mslToWGS84Height(lat, lon, ELEV)
        h_config = mslToWGS84Height(lat, lon, ELEV, config)
        h_path = mslToWGS84Height(lat, lon, ELEV, config.egm96_full_path)
        self.assertEqual(h_default, h_config)
        self.assertEqual(h_default, h_path)
        self.assertAlmostEqual(wgs84toMSLHeight(lat, lon, h_default, config), ELEV, places=9)



class TestGPUModuleImport(unittest.TestCase):

    def testImportsWithoutNumba(self):
        """ The optional CUDA module must import (and report unavailability) when numba is missing. """

        import importlib
        saved = {k: v for k, v in sys.modules.items() if k == 'numba' or k.startswith('numba.')}
        sys.modules.pop('RMS.Astrometry.GPUENHt', None)
        sys.modules['numba'] = None
        try:
            import RMS.Astrometry.GPUENHt as gpu
            importlib.reload(gpu)
            self.assertFalse(gpu.CUDA_AVAILABLE)
            with self.assertRaises(ImportError):
                gpu.ENHt0ToENHt1_gpu([0.0], [0.0], [HT], [HT], Platepar())
        finally:
            sys.modules.pop('numba', None)
            sys.modules.update(saved)
            sys.modules.pop('RMS.Astrometry.GPUENHt', None)



if __name__ == "__main__":
    unittest.main()
