""" Atmospheric refraction: the observer-height scale and the finite-target-height fraction of the star refraction.
    Run with: python -m unittest Tests.test_refraction """

from __future__ import print_function, division, absolute_import

import os
import math
import copy
import unittest

import numpy as np
import scipy.optimize

from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, raDecToXYPP
from RMS.Astrometry.Conversions import date2JD, JD2HourAngle
from RMS.Astrometry.CyFunctions import (refractionScale, refractionTargetFraction, pyRefractionApparentToTrue,
    pyRefractionTrueToApparent, equatorialCoordPrecession)

try:
    # Kernels that model the annual aberration: a star's catalog direction is its Earth-frame direction with the
    #   aberration removed
    from RMS.Astrometry.CyFunctions import removeAberration
except ImportError:
    removeAberration = None
from RMS.Misc import getRmsRootDir


EARTH_RADIUS = 6371.0e3
TEMPLATE = os.path.join(getRmsRootDir(), 'share', 'platepar_templates', 'template_generic_720p_4mm.cal')


def isaRefractivity(h):
    """ (n - 1) of the International Standard Atmosphere relative to sea level, from pressure over temperature. """

    if h <= 11000.0:
        temp = 288.15 - 0.0065*h
        pressure = 101325.0*(temp/288.15)**5.25588

    else:
        temp = 216.65
        pressure = 22632.0*math.exp(-(h - 11000.0)/6341.6)

    return (pressure/temp)/(101325.0/288.15)


def rayTrace(elev_app_deg, h_obs, h_target, step=5.0):
    """ Trace a ray from the observer through a spherically symmetric ISA atmosphere (n r sin z = const).

    Return:
        (true elevation of the straight line to the target, true elevation of a star on the same ray), degrees.
    """

    r = EARTH_RADIUS + h_obs
    z = math.radians(90.0 - elev_app_deg)
    invariant = (1 + 2.77e-4*isaRefractivity(h_obs))*r*math.sin(z)
    phi = 0.0
    elev_target = None

    while r < EARTH_RADIUS + 150.0e3:

        n = 1 + 2.77e-4*isaRefractivity(max(r - EARTH_RADIUS, 0.0))
        z = math.asin(min(invariant/(n*r), 1.0))
        r_new = r + step*math.cos(z)
        phi += step*math.sin(z)/r

        if (elev_target is None) and (r_new - EARTH_RADIUS >= h_target):
            elev_target = math.degrees(math.atan2(r_new*math.cos(phi) - (EARTH_RADIUS + h_obs), r_new*math.sin(phi)))
            step = 200.0

        r = r_new

    n = 1 + 2.77e-4*isaRefractivity(r - EARTH_RADIUS)
    z = math.asin(min(invariant/(n*r), 1.0))
    elev_star = 90.0 - math.degrees(z) - math.degrees(phi)

    return elev_target, elev_star


class TestRefraction(unittest.TestCase):

    def testScaleFollowsTheStandardAtmosphere(self):
        """ The refraction scale is the ISA pressure over temperature relative to sea level, exactly 1 at sea
            level (no change of behaviour for sea-level stations). """

        self.assertEqual(refractionScale(0.0), 1.0)

        for h in (700.0, 2074.0, 2796.0, 4000.0):
            self.assertAlmostEqual(refractionScale(h), isaRefractivity(h), places=4)

        self.assertAlmostEqual(refractionScale(2074.0), 0.816, places=3)


    def testTargetFractionAgainstRayTrace(self):
        """ The closed-form fraction of the star refraction that applies to a target at a finite height matches a
            ray trace through the standard atmosphere: within 2% of the star refraction for a contrail (10 km) and
            a target at 30 km at any elevation, within 2% for a meteor at 100 km above 10 deg. """

        for h_obs in (0.0, 2074.0):
            for h_target, elevations, tolerance in ((10.0e3, (5, 10, 20, 45), 0.02), (30.0e3, (10, 45), 0.02),
                    (100.0e3, (10, 45), 0.02)):

                for elev in elevations:
                    elev_target, elev_star = rayTrace(elev, h_obs, h_target)
                    fraction_traced = (elev - elev_target)/(elev - elev_star)
                    self.assertLess(abs(refractionTargetFraction(h_obs, h_target) - fraction_traced), tolerance,
                        (h_obs, h_target, elev, fraction_traced))

        self.assertEqual(refractionTargetFraction(0.0, float('inf')), 1.0)
        self.assertEqual(refractionTargetFraction(0.0, 1.0e7), 1.0)
        self.assertEqual(refractionTargetFraction(1000.0, 500.0), 0.0)
        self.assertAlmostEqual(refractionTargetFraction(0.0, 10.0e3), 0.377, places=2)
        self.assertAlmostEqual(refractionTargetFraction(0.0, 100.0e3), 0.916, places=2)


    def testScaledFormulasStayInverse(self):
        """ The true -> apparent conversion is the exact inverse of the apparent -> true one, with any scale. """

        for scale in (1.0, 0.75):
            for elev in range(5, 90, 5):
                back = pyRefractionApparentToTrue(pyRefractionTrueToApparent(math.radians(elev), scale), scale)
                self.assertLess(abs(math.degrees(back) - elev)*3600, 0.001)


    def testStarFitAtAltitudeLeavesNoBias(self):
        """ A camera at 2610 m (USV003) pointing at 35 deg altitude, stars refracted with the true (scaled) refraction
            and the platepar fitted by RMS: with the station elevation the meteor astrometry of the fitted platepar
            is right over the whole frame, with the sea-level model it is biased at the bottom of the frame. Only
            the pointing and scale are fitted: the reverse-polynomial derivation of the full fit leaves a ~0.05 px
            forward/inverse offset of its own, which would mask the comparison. (With the distortion fitted, the
            radial terms absorb most of the sea-level model's error and leave 5 to 11 arcsec on the USV cameras.) """

        lat, lon, elev_station = 31.69, -110.9, 2610.0
        az0, alt0 = 334.8, 34.8
        x_res, y_res, f_scale = 1280, 720, 1280/39.8
        jd = date2JD(2026, 9, 5, 6, 0, 0)
        true_scale = refractionScale(elev_station)

        # Camera fixed in alt/az, no distortion: pixel -> apparent direction (ENU), the kernels' projection
        la, lo = np.radians(lat), np.radians(lon)
        sl, cl, so, co = np.sin(la), np.cos(la), np.sin(lo), np.cos(lo)
        enu_to_ecef = np.array([[-so, -sl*co, cl*co], [co, -sl*so, cl*so], [0, cl, sl]])
        a, h = np.radians(az0), np.radians(alt0)
        pointing = np.array([np.cos(h)*np.sin(a), np.cos(h)*np.cos(a), np.sin(h)])
        up = np.array([0, 0, 1.0]) - pointing[2]*pointing
        up /= np.linalg.norm(up)
        side = np.cross(pointing, up)

        def apparentEnu(x, y):
            dx, dy = x - x_res/2, y - y_res/2
            rho = np.radians(np.hypot(dx, dy)/f_scale)
            theta = np.pi/2 + np.arctan2(dy, dx)
            return np.cos(rho)[:, None]*pointing + np.sin(rho)[:, None]*(np.cos(theta)[:, None]*up
                + np.sin(theta)[:, None]*side)

        def bennett(alt_deg):
            return 1.0/(60*np.tan(np.radians(alt_deg + 7.31/(alt_deg + 4.4))))

        def trueJ2000(x, y, scale):
            """ True J2000 directions of what is seen at the pixels, refracted with the given scale (erfa not needed:
                the Earth rotation and precession only need to be the same rigid frame for the fit and the check,
                so the kernels' own frame is used through a fixed hour angle). """
            d = apparentEnu(x, y)
            az = np.degrees(np.arctan2(d[:, 0], d[:, 1]))%360
            alt_app = np.degrees(np.arcsin(d[:, 2]))
            alt_true = alt_app - scale*bennett(alt_app)
            a_, h_ = np.radians(az), np.radians(alt_true)
            enu = np.stack([np.cos(h_)*np.sin(a_), np.cos(h_)*np.cos(a_), np.sin(h_)], -1)
            ecef = enu.dot(enu_to_ecef.T)
            # Earth-fixed -> equatorial of date (hour angle from the mean sidereal time), then to J2000
            gmst = np.radians(JD2HourAngle(jd))
            eq = np.stack([ecef[:, 0]*np.cos(gmst) - ecef[:, 1]*np.sin(gmst),
                ecef[:, 0]*np.sin(gmst) + ecef[:, 1]*np.cos(gmst), ecef[:, 2]], -1)
            ra = np.arctan2(eq[:, 1], eq[:, 0])%(2*np.pi)
            dec = np.arcsin(eq[:, 2])
            out = np.array([equatorialCoordPrecession(jd, 2451545.0, r_, d_) for r_, d_ in zip(ra, dec)])
            if removeAberration is not None:
                out = np.array([removeAberration(r_, d_, jd) for r_, d_ in out])
            return np.degrees(out[:, 0]), np.degrees(out[:, 1])

        def unit(ra, dec):
            r_, d_ = np.radians(ra), np.radians(dec)
            return np.stack([np.cos(d_)*np.cos(r_), np.cos(d_)*np.sin(r_), np.sin(d_)], -1)

        def separation(ra1, dec1, ra2, dec2):
            u1, u2 = unit(ra1, dec1), unit(ra2, dec2)
            return np.degrees(np.arctan2(np.linalg.norm(np.cross(u1, u2), axis=1), np.sum(u1*u2, axis=1)))*3600

        gx, gy = np.meshgrid(np.linspace(20, x_res - 20, 9), np.linspace(20, y_res - 20, 7))
        gx, gy = gx.ravel(), gy.ravel()
        rng = np.random.default_rng(3)
        sx, sy = rng.uniform(10, x_res - 10, 250), rng.uniform(10, y_res - 10, 250)

        # A platepar with the right pointing and rotation, pinned with no refraction anywhere
        pp = Platepar()
        pp.read(TEMPLATE)
        pp.X_res, pp.Y_res, pp.F_scale, pp.lat, pp.lon, pp.elev = x_res, y_res, f_scale, lat, lon, 0.0
        pp.resetDistortionParameters()
        pp.x_poly_fwd[0] = pp.x_poly_fwd[1] = pp.x_poly_rev[0] = pp.x_poly_rev[1] = 0.0
        pp.refraction = False
        pp.measurement_apparent_to_true_refraction = False
        pp.JD, pp.Ho = jd, JD2HourAngle(jd)
        ra0, dec0 = trueJ2000(gx, gy, 0.0)
        target = unit(ra0, dec0)
        ic = np.argmin(np.hypot(gx - x_res/2, gy - y_res/2))

        def residuals(params):
            pp.RA_d, pp.dec_d, pp.pos_angle_ref = params
            _, ra, dec, _ = xyToRaDecPP(len(gx)*[jd], gx, gy, np.ones(len(gx)), pp, extinction_correction=False,
                jd_time=True)
            return ((unit(ra, dec) - target)*206265).ravel()

        ra_c, dec_c = equatorialCoordPrecession(2451545.0, jd, np.radians(ra0[ic]), np.radians(dec0[ic]))
        best = min((scipy.optimize.least_squares(residuals, [np.degrees(ra_c), np.degrees(dec_c), rot0],
            xtol=1e-12, ftol=1e-12) for rot0 in (0, 90, 180, 270)), key=lambda r: r.cost)
        pp.RA_d, pp.dec_d, pp.pos_angle_ref = best.x
        self.assertLess(np.sqrt(2*best.cost/len(gx)/3), 0.5)

        # Stars as the camera sees them at this station, the fit as RMS does it, the platepar's meteor astrometry
        ra_s, dec_s = trueJ2000(sx, sy, true_scale)
        ra_t, dec_t = trueJ2000(gx, gy, true_scale)
        worst = {}
        for elev_model in (0.0, elev_station):
            pp_fit = copy.deepcopy(pp)
            pp_fit.elev = elev_model
            pp_fit.refraction = True
            pp_fit.fitAstrometry(jd, np.column_stack([sx, sy, np.ones(len(sx))]),
                np.column_stack([ra_s, dec_s, 5*np.ones(len(sx))]), fit_only_pointing=True)

            xs, ys = raDecToXYPP(ra_s, dec_s, jd, pp_fit)
            star_rms_arcsec = np.sqrt(np.mean((xs - sx)**2 + (ys - sy)**2))/f_scale*3600
            self.assertLess(star_rms_arcsec, 5.0, elev_model)

            # Evaluate through the star path (catalog frame, the same frame as the truth)
            _, ra, dec, _ = xyToRaDecPP(len(gx)*[jd], gx, gy, np.ones(len(gx)), pp_fit, extinction_correction=False,
                jd_time=True)
            worst[elev_model] = separation(ra, dec, ra_t, dec_t).max()

        self.assertLess(worst[elev_station], 1.0)
        self.assertGreater(worst[0.0], 4.0)


if __name__ == "__main__":
    unittest.main()
