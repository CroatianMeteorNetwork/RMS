""" Atmospheric refraction: the observer-height scale, the finite-target-height fraction of the star
    refraction, and the closure of the two refraction directions.
"""

import os
import math
import copy

import pytest

np = pytest.importorskip("numpy")

from RMS.Misc import getRmsRootDir
from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, raDecToXYPP, targetRaDecToPlateRaDec
from RMS.Astrometry.Conversions import date2JD, JD2HourAngle
from RMS.Astrometry.CyFunctions import (refractionScale, refractionTargetFraction,
    pyRefractionApparentToTrue, pyRefractionTrueToApparent, equatorialCoordPrecession)


EARTH_RADIUS = 6371.0e3
TEMPLATE = os.path.join(getRmsRootDir(), 'share', 'platepar_templates', 'template_generic_720p_4mm.cal')


def _isaRefractivity(h):
    """ Helper: (n - 1) of the International Standard Atmosphere relative to sea level, computed from the
        pressure and the temperature independently of the kernels. """

    if h <= 11000.0:
        temp = 288.15 - 0.0065*h
        pressure = 101325.0*(temp/288.15)**5.25588

    else:
        temp = 216.65
        pressure = 22632.0*math.exp(-(h - 11000.0)/6341.6)

    return (pressure/temp)/(101325.0/288.15)


def _rayTrace(elev_app_deg, h_obs, h_target, step=20.0):
    """ Helper: trace a ray from the observer through a spherically symmetric ISA atmosphere, using the
        invariant n*r*sin(z) = const.

    Return:
        (elev_target, elev_star): [tuple of floats] True elevation of the straight line to the target, and
            true elevation of a star on the same ray (degrees).
    """

    r = EARTH_RADIUS + h_obs
    z = math.radians(90.0 - elev_app_deg)
    invariant = (1 + 2.77e-4*_isaRefractivity(h_obs))*r*math.sin(z)
    phi = 0.0
    elev_target = None

    while r < EARTH_RADIUS + 150.0e3:

        n = 1 + 2.77e-4*_isaRefractivity(max(r - EARTH_RADIUS, 0.0))
        z = math.asin(min(invariant/(n*r), 1.0))
        r_new = r + step*math.cos(z)
        phi += step*math.sin(z)/r

        # Once the ray has reached the target height, record the direction of the straight line to it and
        #   take coarser steps for the rest of the way up
        if (elev_target is None) and (r_new - EARTH_RADIUS >= h_target):
            elev_target = math.degrees(math.atan2(r_new*math.cos(phi) - (EARTH_RADIUS + h_obs), \
                r_new*math.sin(phi)))
            step = 500.0

        r = r_new

    n = 1 + 2.77e-4*_isaRefractivity(r - EARTH_RADIUS)
    z = math.asin(min(invariant/(n*r), 1.0))
    elev_star = 90.0 - math.degrees(z) - math.degrees(phi)

    return elev_target, elev_star


def test_scale_is_exactly_one_at_sea_level():
    """ Sea-level stations must be unaffected by the height scaling. """

    assert refractionScale(0.0) == 1.0


@pytest.mark.parametrize("height", [700.0, 2074.0, 2610.0, 2796.0, 4000.0])
def test_scale_follows_the_standard_atmosphere(height):
    """ The refraction scale is the ISA pressure over temperature relative to sea level. """

    assert abs(refractionScale(height) - _isaRefractivity(height)) < 1e-4


def test_scale_matches_the_documented_values():
    """ Spot values quoted in the docstring of refractionScale. """

    assert abs(refractionScale(700.0) - 0.93) < 0.005
    assert abs(refractionScale(2074.0) - 0.816) < 0.001
    assert abs(refractionScale(2800.0) - 0.757) < 0.001


@pytest.mark.parametrize("h_obs", [0.0, 2074.0])
@pytest.mark.parametrize("h_target, elevations", [(10.0e3, (5, 10, 20, 45)), (30.0e3, (10, 45)),
    (100.0e3, (10, 45))])
def test_target_fraction_against_a_ray_trace(h_obs, h_target, elevations):
    """ The closed-form fraction of the star refraction that applies to a target at a finite height matches
        a ray trace through the standard atmosphere to within 2% of the star refraction. """

    for elev in elevations:

        elev_target, elev_star = _rayTrace(elev, h_obs, h_target)
        fraction_traced = (elev - elev_target)/(elev - elev_star)

        assert abs(refractionTargetFraction(h_obs, h_target) - fraction_traced) < 0.02, \
            (h_obs, h_target, elev, fraction_traced)


def test_target_fraction_edge_cases():
    """ A star gets the full refraction, a target at or below the observer gets none, and the values in
        between match the ones quoted in the docstring. """

    assert refractionTargetFraction(0.0, float('inf')) == 1.0
    assert refractionTargetFraction(0.0, 1.0e7) == 1.0
    assert refractionTargetFraction(1000.0, 1000.0) == 0.0

    # The closed form is only valid for a target above the observer
    assert refractionTargetFraction(1000.0, 500.0) == 0.0

    assert abs(refractionTargetFraction(0.0, 10.0e3) - 0.377) < 0.01
    assert abs(refractionTargetFraction(0.0, 30.0e3) - 0.722) < 0.01
    assert abs(refractionTargetFraction(0.0, 100.0e3) - 0.916) < 0.01


@pytest.mark.parametrize("scale", [1.0, 0.9, 0.75])
def test_refraction_directions_are_exact_inverses(scale):
    """ The true -> apparent conversion is the exact inverse of the apparent -> true one, at any scale and
        at any elevation. The convergence of the inversion is slowest at the horizon, so the low elevations
        are the ones that matter here. """

    for elev_deg in [-0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0] + list(range(5, 90, 5)):

        elev = math.radians(elev_deg)
        closed = pyRefractionApparentToTrue(pyRefractionTrueToApparent(elev, scale), scale)

        assert abs(math.degrees(closed) - elev_deg)*3600 < 1e-3, (scale, elev_deg)


@pytest.mark.parametrize("elev_deg", [1.0, 10.0, 45.0])
def test_scale_multiplies_the_refraction(elev_deg):
    """ The scale is a plain multiplier on the apparent-to-true refraction. """

    elev = math.radians(elev_deg)
    full = elev - pyRefractionApparentToTrue(elev, 1.0)
    scaled = elev - pyRefractionApparentToTrue(elev, 0.75)

    assert abs(scaled - 0.75*full) < 1e-12


def test_star_fit_at_altitude_leaves_no_bias():
    """ A camera at 2610 m (USV003) pointing at 35 deg altitude, with the stars refracted by the true
        (scaled) refraction and the platepar fitted by RMS: with the station elevation the meteor astrometry
        of the fitted platepar is right over the whole frame, with the sea-level model it is biased at the
        bottom of the frame.

        Only the pointing and the scale are fitted: the reverse-polynomial derivation of the full fit leaves
        a ~0.05 px forward/inverse offset of its own, which would mask the comparison. With the distortion
        fitted too, the radial terms absorb most of the sea-level model's error and leave 5 to 11 arcsec on
        the USV cameras.
    """

    scipy_optimize = pytest.importorskip("scipy.optimize")

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
        return np.cos(rho)[:, None]*pointing + np.sin(rho)[:, None]*(np.cos(theta)[:, None]*up \
            + np.sin(theta)[:, None]*side)

    def bennett(alt_deg):
        return 1.0/(60*np.tan(np.radians(alt_deg + 7.31/(alt_deg + 4.4))))

    def trueJ2000(x, y, scale):
        """ True J2000 directions of what is seen at the given pixels, refracted with the given scale. No
            need for erfa: the Earth rotation and the precession only have to put the fit and the check in
            the same rigid frame, so the kernels' own frame is used through a fixed hour angle. """

        d = apparentEnu(x, y)
        az = np.degrees(np.arctan2(d[:, 0], d[:, 1]))%360
        alt_app = np.degrees(np.arcsin(d[:, 2]))
        alt_true = alt_app - scale*bennett(alt_app)
        a_, h_ = np.radians(az), np.radians(alt_true)
        enu = np.stack([np.cos(h_)*np.sin(a_), np.cos(h_)*np.cos(a_), np.sin(h_)], -1)
        ecef = enu.dot(enu_to_ecef.T)

        # Earth-fixed -> equatorial of date (hour angle from the mean sidereal time), then to J2000
        gmst = np.radians(JD2HourAngle(jd))
        eq = np.stack([ecef[:, 0]*np.cos(gmst) - ecef[:, 1]*np.sin(gmst), \
            ecef[:, 0]*np.sin(gmst) + ecef[:, 1]*np.cos(gmst), ecef[:, 2]], -1)
        ra = np.arctan2(eq[:, 1], eq[:, 0])%(2*np.pi)
        dec = np.arcsin(eq[:, 2])
        out = np.array([equatorialCoordPrecession(jd, 2451545.0, r_, d_) for r_, d_ in zip(ra, dec)])

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
        _, ra, dec, _ = xyToRaDecPP(len(gx)*[jd], gx, gy, np.ones(len(gx)), pp, \
            extinction_correction=False, jd_time=True)
        return ((unit(ra, dec) - target)*206265).ravel()

    ra_c, dec_c = equatorialCoordPrecession(2451545.0, jd, np.radians(ra0[ic]), np.radians(dec0[ic]))
    best = min((scipy_optimize.least_squares(residuals, [np.degrees(ra_c), np.degrees(dec_c), rot0], \
        xtol=1e-12, ftol=1e-12) for rot0 in (0, 90, 180, 270)), key=lambda r: r.cost)
    pp.RA_d, pp.dec_d, pp.pos_angle_ref = best.x
    assert np.sqrt(2*best.cost/len(gx)/3) < 0.5

    # Stars as the camera sees them at this station, the fit as RMS does it, and the meteor astrometry of
    #   the resulting platepar
    ra_s, dec_s = trueJ2000(sx, sy, true_scale)
    ra_t, dec_t = trueJ2000(gx, gy, true_scale)
    worst = {}

    for elev_model in (0.0, elev_station):

        pp_fit = copy.deepcopy(pp)
        pp_fit.elev = elev_model
        pp_fit.refraction = True
        pp_fit.fitAstrometry(jd, np.column_stack([sx, sy, np.ones(len(sx))]), \
            np.column_stack([ra_s, dec_s, 5*np.ones(len(sx))]), fit_only_pointing=True)

        xs, ys = raDecToXYPP(ra_s, dec_s, jd, pp_fit)
        star_rms_arcsec = np.sqrt(np.mean((xs - sx)**2 + (ys - sy)**2))/f_scale*3600
        assert star_rms_arcsec < 5.0, elev_model

        # Evaluate through the star path (catalog frame, the same frame as the truth)
        _, ra, dec, _ = xyToRaDecPP(len(gx)*[jd], gx, gy, np.ones(len(gx)), pp_fit, \
            extinction_correction=False, jd_time=True)
        worst[elev_model] = separation(ra, dec, ra_t, dec_t).max()

    # With the station height the fit is unbiased, with the sea-level model it is not
    assert worst[elev_station] < 1.0
    assert worst[0.0] > 4.0


def _syntheticCamera(lat, lon, elev_station, az0, alt0, x_res, y_res, f_scale, jd):
    """ Helper: a camera fixed in alt/az with no distortion, as a function that returns the true J2000
        direction of whatever is seen at a given pixel, refracted with a given scale. Passing a scale of 0
        gives the direction of a target whose light is not refracted, i.e. a ground reference.
    """

    la, lo = np.radians(lat), np.radians(lon)
    sl, cl, so, co = np.sin(la), np.cos(la), np.sin(lo), np.cos(lo)
    enu_to_ecef = np.array([[-so, -sl*co, cl*co], [co, -sl*so, cl*so], [0, cl, sl]])
    a, h = np.radians(az0), np.radians(alt0)
    pointing = np.array([np.cos(h)*np.sin(a), np.cos(h)*np.cos(a), np.sin(h)])
    up = np.array([0, 0, 1.0]) - pointing[2]*pointing
    up /= np.linalg.norm(up)
    side = np.cross(pointing, up)
    gmst = np.radians(JD2HourAngle(jd))

    def trueJ2000(x, y, refr_scale):

        dx, dy = x - x_res/2, y - y_res/2
        rho = np.radians(np.hypot(dx, dy)/f_scale)
        theta = np.pi/2 + np.arctan2(dy, dx)
        d = np.cos(rho)[:, None]*pointing + np.sin(rho)[:, None]*(np.cos(theta)[:, None]*up \
            + np.sin(theta)[:, None]*side)

        az = np.degrees(np.arctan2(d[:, 0], d[:, 1]))%360
        alt_app = np.degrees(np.arcsin(d[:, 2]))
        alt_true = alt_app - refr_scale/(60*np.tan(np.radians(alt_app + 7.31/(alt_app + 4.4))))

        a_, h_ = np.radians(az), np.radians(alt_true)
        enu = np.stack([np.cos(h_)*np.sin(a_), np.cos(h_)*np.cos(a_), np.sin(h_)], -1)
        ecef = enu.dot(enu_to_ecef.T)
        eq = np.stack([ecef[:, 0]*np.cos(gmst) - ecef[:, 1]*np.sin(gmst), \
            ecef[:, 0]*np.sin(gmst) + ecef[:, 1]*np.cos(gmst), ecef[:, 2]], -1)
        ra = np.arctan2(eq[:, 1], eq[:, 0])%(2*np.pi)
        dec = np.arcsin(eq[:, 2])
        out = np.array([equatorialCoordPrecession(jd, 2451545.0, r_, d_) for r_, d_ in zip(ra, dec)])

        return np.degrees(out[:, 0]), np.degrees(out[:, 1])

    return trueJ2000


def _separation(ra1, dec1, ra2, dec2):
    """ Helper: angular separation between two directions, in arc seconds. """

    r1, d1, r2, d2 = np.radians(ra1), np.radians(dec1), np.radians(ra2), np.radians(dec2)
    cos_ang = np.sin(d1)*np.sin(d2) + np.cos(d1)*np.cos(d2)*np.cos(r1 - r2)

    return np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))*3600


@pytest.mark.parametrize("references", ["geo_only", "mixed"])
def test_ground_references_are_fitted_without_refraction(references):
    """ A plate fitted against ground references has to reproduce both kinds of measurement: the ground
        points themselves, whose light is not refracted, and objects on the sky, whose light is. Handing
        the fit the true direction of a ground point instead of its plate coordinates biases the whole
        plate by up to a full refraction.

        The camera here points at 20 deg altitude from 2610 m, where a star is refracted by about 160
        arcsec, so the bias is easy to separate from the numerical noise of the fit.
    """

    lat, lon, elev_station = 31.69, -110.9, 2610.0
    x_res, y_res, f_scale = 1280, 720, 1280/39.8
    jd = date2JD(2026, 9, 5, 6, 0, 0)
    scale = refractionScale(elev_station)
    trueJ2000 = _syntheticCamera(lat, lon, elev_station, 334.8, 20.0, x_res, y_res, f_scale, jd)

    # Stars and ground references seen by the camera. A ground reference is seen in its true direction, so
    #   its true J2000 direction is the one the camera's line of sight points at, with no refraction.
    rng = np.random.default_rng(3)
    sx, sy = rng.uniform(10, x_res - 10, 300), rng.uniform(10, y_res - 10, 300)
    ra_s, dec_s = trueJ2000(sx, sy, scale)
    rng = np.random.default_rng(11)
    gx, gy = rng.uniform(20, x_res - 20, 40), rng.uniform(20, y_res - 20, 40)
    ra_g, dec_g = trueJ2000(gx, gy, 0.0)

    # Check points spread over the frame, and the truth for both kinds of measurement
    cx, cy = np.meshgrid(np.linspace(30, x_res - 30, 7), np.linspace(30, y_res - 30, 5))
    cx, cy = cx.ravel(), cy.ravel()
    ra_ground_true, dec_ground_true = trueJ2000(cx, cy, 0.0)
    ra_sky_true, dec_sky_true = trueJ2000(cx, cy, scale)

    # Start from an ordinary refraction-on plate fitted on the stars, so that the distortion is already
    #   the one of this camera and the comparison below only sees the effect of the ground references
    pp = Platepar()
    pp.read(TEMPLATE)
    pp.X_res, pp.Y_res, pp.F_scale = x_res, y_res, f_scale
    pp.lat, pp.lon, pp.elev = lat, lon, elev_station
    pp.resetDistortionParameters()

    # The synthetic camera is a plain gnomonic projection, so pin the distortion to the identity and fit
    #   only the pointing. That keeps the comparison below free of any distortion-fit residual of its own.
    pp.x_poly_fwd[0] = pp.x_poly_fwd[1] = pp.x_poly_rev[0] = pp.x_poly_rev[1] = 0.0
    pp.refraction = True
    pp.JD, pp.Ho = jd, JD2HourAngle(jd)
    ra_c, dec_c = trueJ2000(np.array([x_res/2.0]), np.array([y_res/2.0]), scale)
    pp.RA_d, pp.dec_d, pp.pos_angle_ref = ra_c[0], dec_c[0], 0.0
    pp.fitAstrometry(jd, np.column_stack([sx, sy, np.ones(len(sx))]), \
        np.column_stack([ra_s, dec_s, 5*np.ones(len(sx))]), fit_only_pointing=True)
    pp.updateRefAltAz()
    xs, ys = raDecToXYPP(ra_s, dec_s, jd, pp)
    assert np.sqrt(np.mean((xs - sx)**2 + (ys - sy)**2)) < 0.01

    # The plate coordinates of the ground references: the refraction the plate applies, taken back out
    ra_plate, dec_plate = targetRaDecToPlateRaDec(ra_g, dec_g, jd, pp, refraction_fraction=0.0)
    assert _separation(ra_plate, dec_plate, ra_g, dec_g).max() > 100.0

    def fitAndMeasure(ra_ref, dec_ref, x_ref, y_ref):
        """ Fit a plate against the given references, then measure the check points both ways. """

        pp_fit = copy.deepcopy(pp)
        pp_fit.fitAstrometry(jd, np.column_stack([x_ref, y_ref, np.ones(len(x_ref))]), \
            np.column_stack([ra_ref, dec_ref, 5*np.ones(len(x_ref))]), fit_only_pointing=True)
        pp_fit.updateRefAltAz()

        # Ground picks go through the refraction-free representation of the same plate
        pp_ground = copy.deepcopy(pp_fit)
        pp_ground.switchToGroundPicks()
        _, ra_m, dec_m, _ = xyToRaDecPP(len(cx)*[jd], cx, cy, np.ones(len(cx)), pp_ground, \
            extinction_correction=False, measurement=True, jd_time=True)

        # Sky measurements go through the plate itself and come out as true J2000
        _, ra_k, dec_k, _ = xyToRaDecPP(len(cx)*[jd], cx, cy, np.ones(len(cx)), pp_fit, \
            extinction_correction=False, measurement=True, jd_time=True)

        return (_separation(ra_m, dec_m, ra_ground_true, dec_ground_true).max(),
                _separation(ra_k, dec_k, ra_sky_true, dec_sky_true).max())

    if references == "geo_only":
        x_ref, y_ref = gx, gy
        ra_true_ref, dec_true_ref = ra_g, dec_g
        ra_fit_ref, dec_fit_ref = ra_plate, dec_plate
        expected_bias = 100.0

    else:
        x_ref, y_ref = np.concatenate([sx, gx]), np.concatenate([sy, gy])
        ra_true_ref = np.concatenate([ra_s, ra_g])
        dec_true_ref = np.concatenate([dec_s, dec_g])
        ra_fit_ref = np.concatenate([ra_s, ra_plate])
        dec_fit_ref = np.concatenate([dec_s, dec_plate])
        expected_bias = 10.0

    # Handing the fit the true direction of the ground points biases the plate
    ground_err, sky_err = fitAndMeasure(ra_true_ref, dec_true_ref, x_ref, y_ref)
    assert ground_err > expected_bias
    assert sky_err > expected_bias

    # With the plate coordinates, both kinds of measurement come out right from the same plate
    ground_err, sky_err = fitAndMeasure(ra_fit_ref, dec_fit_ref, x_ref, y_ref)
    assert ground_err < 1.0, ground_err
    assert sky_err < 1.0, sky_err


def test_plate_coordinates_are_a_no_op_without_refraction():
    """ A plate fitted with the refraction off maps the arrival direction of a target directly, so a ground
        reference needs no correction at all. """

    pp = Platepar()
    pp.read(TEMPLATE)
    pp.lat, pp.lon, pp.elev = 31.69, -110.9, 2610.0
    pp.refraction = False

    ra = np.array([120.0, 121.0, 122.0])
    dec = np.array([20.0, 21.0, 22.0])
    ra_plate, dec_plate = targetRaDecToPlateRaDec(ra, dec, 2460000.5, pp)

    assert np.array_equal(ra_plate, ra)
    assert np.array_equal(dec_plate, dec)


def test_a_full_refraction_fraction_is_a_no_op():
    """ A target outside the atmosphere is refracted exactly like a star, so its plate coordinates are its
        true coordinates. This is the check that the two steps of targetRaDecToPlateRaDec are inverses. """

    pp = Platepar()
    pp.read(TEMPLATE)
    pp.lat, pp.lon, pp.elev = 31.69, -110.9, 2610.0
    pp.refraction = True

    ra = np.array([120.0, 150.0, 200.0])
    dec = np.array([5.0, 20.0, 60.0])
    ra_plate, dec_plate = targetRaDecToPlateRaDec(ra, dec, 2460000.5, pp, refraction_fraction=1.0)

    assert _separation(ra_plate, dec_plate, ra, dec).max() < 1e-6


def test_ground_projection_uses_the_apparent_pointing():
    """ xyHt2Geo() and geoHt2XY() project onto the ground without refraction, which is right for a target
        on the ground. Clearing the refraction flag alone is not enough though: that leaves the reference
        pointing at the true direction of the camera axis while the camera looks along the apparent one,
        which offsets the whole field by the refraction at the FOV centre, about 3 arc minutes at 20 deg
        elevation.

        The direction that the ground path recovers from a pixel is checked here against the line of sight
        of a synthetic camera, which is free of any Earth-model assumption.
    """

    lat, lon, elev_station = 31.69, -110.9, 2610.0
    x_res, y_res, f_scale = 1280, 720, 1280/39.8
    jd = date2JD(2026, 9, 5, 6, 0, 0)
    scale = refractionScale(elev_station)
    trueJ2000 = _syntheticCamera(lat, lon, elev_station, 334.8, 20.0, x_res, y_res, f_scale, jd)

    # An ordinary refraction-on plate fitted on the stars of this camera
    rng = np.random.default_rng(3)
    sx, sy = rng.uniform(10, x_res - 10, 200), rng.uniform(10, y_res - 10, 200)
    ra_s, dec_s = trueJ2000(sx, sy, scale)

    pp = Platepar()
    pp.read(TEMPLATE)
    pp.X_res, pp.Y_res, pp.F_scale = x_res, y_res, f_scale
    pp.lat, pp.lon, pp.elev = lat, lon, elev_station
    pp.resetDistortionParameters()
    pp.x_poly_fwd[0] = pp.x_poly_fwd[1] = pp.x_poly_rev[0] = pp.x_poly_rev[1] = 0.0
    pp.refraction = True
    pp.JD, pp.Ho = jd, JD2HourAngle(jd)
    ra_c, dec_c = trueJ2000(np.array([x_res/2.0]), np.array([y_res/2.0]), scale)
    pp.RA_d, pp.dec_d, pp.pos_angle_ref = ra_c[0], dec_c[0], 0.0
    pp.fitAstrometry(jd, np.column_stack([sx, sy, np.ones(len(sx))]), \
        np.column_stack([ra_s, dec_s, 5*np.ones(len(sx))]), fit_only_pointing=True)
    pp.updateRefAltAz()

    cx, cy = np.meshgrid(np.linspace(30, x_res - 30, 5), np.linspace(30, y_res - 30, 4))
    cx, cy = cx.ravel(), cy.ravel()

    # The light of a target on the ground is not refracted, so its true direction is the line of sight
    ra_ground, dec_ground = trueJ2000(cx, cy, 0.0)

    # The direction the ground path of xyHt2Geo() recovers from those pixels
    pp_ground = copy.deepcopy(pp)
    pp_ground.updateRefAltAz()
    pp_ground.switchToGroundPicks()
    _, ra_m, dec_m, _ = xyToRaDecPP(len(cx)*[jd], cx, cy, np.ones(len(cx)), pp_ground, \
        extinction_correction=False, measurement=False, jd_time=True)

    assert _separation(ra_m, dec_m, ra_ground, dec_ground).max() < 1.0

    # Leaving the reference pointing alone, as the ground helpers used to, offsets the whole field
    pp_flag_only = copy.deepcopy(pp)
    pp_flag_only.refraction = False
    _, ra_b, dec_b, _ = xyToRaDecPP(len(cx)*[jd], cx, cy, np.ones(len(cx)), pp_flag_only, \
        extinction_correction=False, measurement=False, jd_time=True)

    assert _separation(ra_b, dec_b, ra_ground, dec_ground).max() > 100.0
