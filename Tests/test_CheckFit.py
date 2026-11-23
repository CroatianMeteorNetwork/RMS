"""
Tests of the fit of the pointing in the calibration check (CheckFit.autoCheckFit) and in the recalibration on
individual images (ApplyRecalibrate.recalibrateFF), which share its residual function
(CheckFit._calcImageResidualsAstro).

The residual function evaluates the trial parameters of the fit on the platepar it is given. A fit which doesn't
succeed must leave that platepar as it was: the check hands it to the NN alignment and, if the check fails, to the
recalibration of the night, and the recalibration keeps the platepar of its last successful fit. The tests use
synthetic images of a 1280x720 camera with a 60 deg field: catalog stars at known sky positions, detected where
the true platepar puts them at the time of every image, and a starting platepar with the pointing off by 2 px.
"""

import copy
import datetime

import numpy as np
import pytest
import scipy.optimize

import RMS.ConfigReader as cr
from RMS.Astrometry import ApplyRecalibrate, CheckFit
from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, xyToRaDecPP
from RMS.Astrometry.Conversions import date2JD, JD2HourAngle, jd2Date
from RMS.Formats import FFfile
from RMS.Formats.Platepar import Platepar, PointingOffsets


# Julian date of the first synthetic image (2023-02-28 ~18 UT)
JD_START = 2460004.25

# Number of images of the synthetic night and of stars on every image
N_FF = 6
N_STARS = 40

# Number of frames in an FF file
FF_FRAMES = 256


def makePlatepar():
    """ A 1280x720 platepar with a 60 deg field pointed 60 deg up to the south, without distortion. """

    # The station and the image
    pp = Platepar()
    pp.lat, pp.lon, pp.elev = 45.0, 15.0, 100.0
    pp.X_res, pp.Y_res = 1280, 720
    pp.JD = JD_START

    # The reference hour angle goes with the reference time, as when a platepar is fitted or loaded
    pp.Ho = JD2HourAngle(pp.JD)

    # The scale and the pointing, with the reference RA and Dec of the pointing at the reference time
    pp.F_scale = pp.X_res/60.0
    pp.az_centre, pp.alt_centre = 180.0, 60.0
    pp.updateRefRADec(skip_rot_update=True)
    pp.updateRefAltAz()

    return pp


def makeConfig():
    """ The default configuration for a 1280x720 camera, with the number of images and stars of the check
        reduced to the size of the synthetic night. """

    # The camera, and the images and stars which the check needs
    config = cr.Config()
    config.width, config.height = 1280, 720
    config.calstars_files_N = N_FF
    config.calstars_min_stars = N_FF*N_STARS//2

    return config


def offsetPlatepar(pp, offset_px=2.0):
    """ The platepar with the pointing off by the given number of pixels in RA. """

    # Move the pointing in RA, on a copy
    pp = copy.deepcopy(pp)
    pp.RA_d += offset_px/pp.F_scale/np.cos(np.radians(pp.dec_d))
    pp.updateRefAltAz()

    return pp


def pointing(pp):
    """ The fitted parameters of a platepar: reference RA, Dec, position angle and scale. """

    return [pp.RA_d, pp.dec_d, pp.pos_angle_ref, pp.F_scale]


def makeNight(pp, config, seed=0):
    """ A synthetic night: catalog stars and their detections on N_FF images 10 s apart.

    Arguments:
        pp: [Platepar instance] The true platepar.
        config: [Config instance]

    Keyword arguments:
        seed: [int] Seed of the random numbers. 0 by default.

    Return:
        (catalog_stars, calstars_data): [tuple]
            - catalog_stars: [ndarray] Catalog stars (ra, dec, mag), sorted by descending declination.
            - calstars_data: [tuple] The CALSTARS content: a list of [ff_name, star_rows] and the number of
                frames in an FF file.
    """

    rng = np.random.default_rng(seed)

    # Names of the images, and the Julian dates of their middles
    dt_start = jd2Date(JD_START, dt_obj=True)
    ff_names = []
    for i in range(N_FF):
        dt = dt_start + datetime.timedelta(seconds=10*i)
        ff_names.append('FF_XX0001_{:s}_{:03d}_{:07d}.fits'.format(dt.strftime('%Y%m%d_%H%M%S'),
                                                                  dt.microsecond//1000, i*FF_FRAMES))
    jds = [date2JD(*FFfile.getMiddleTimeFF(name, config.fps, ret_milliseconds=True, ff_frames=FF_FRAMES))
           for name in ff_names]

    # Catalog stars at random positions in the first image, at least 20 px apart
    xy = []
    while len(xy) < N_STARS:
        x, y = rng.uniform(80, pp.X_res - 80), rng.uniform(80, pp.Y_res - 80)
        if all(np.hypot(x - xi, y - yi) > 20 for xi, yi in xy):
            xy.append((x, y))
    x, y = np.array(xy).T
    _, ra, dec, _ = xyToRaDecPP(np.full(N_STARS, jds[0]), x, y, np.ones(N_STARS), pp,
                                extinction_correction=False, jd_time=True)
    mag = rng.uniform(2.0, 5.0, N_STARS)
    catalog_stars = np.column_stack([ra, dec, mag]).astype(np.float64)
    catalog_stars = catalog_stars[np.argsort(-catalog_stars[:, 1])]

    # The detections on every image are where the true platepar puts the catalog stars at its time, as CALSTARS
    #   rows (y, x, intensity, amplitude, FWHM, background, SNR, saturated)
    calstars_list = []
    intensity = 10**(-0.4*(catalog_stars[:, 2] - 12.0))
    for name, jd in zip(ff_names, jds):
        x_img, y_img = raDecToXYPP(catalog_stars[:, 0], catalog_stars[:, 1], jd, pp)
        rows = np.column_stack([y_img, x_img, intensity, intensity/10, np.full(N_STARS, 2.5),
                                np.full(N_STARS, 20.0), np.full(N_STARS, 30.0), np.zeros(N_STARS)])
        calstars_list.append([name, rows.tolist()])

    return catalog_stars, (calstars_list, FF_FRAMES)


def useCatalog(monkeypatch, catalog_stars):
    """ Make the calibration check load the synthetic catalog instead of the star catalog on disk.

    Arguments:
        monkeypatch: [MonkeyPatch] The pytest fixture which undoes the change after the test.
        catalog_stars: [ndarray] The synthetic catalog stars (ra, dec, mag).
    """

    def _readStarCatalog(*args, **kwargs):
        """ The synthetic catalog, its name and the default band ratios, as readStarCatalog returns them. """

        return catalog_stars, 'synthetic', cr.Config().star_catalog_band_ratios

    monkeypatch.setattr(CheckFit.StarCatalog, 'readStarCatalog', _readStarCatalog)


def test_residual_function_leaves_the_platepar_unchanged():
    """ The residuals are computed with the trial parameters, and the platepar keeps its own. """

    # The stars of the first image
    pp = makePlatepar()
    config = makeConfig()
    catalog_stars, (calstars_list, _) = makeNight(pp, config)
    jd = date2JD(*FFfile.getMiddleTimeFF(calstars_list[0][0], config.fps, ret_milliseconds=True,
                                         ff_frames=FF_FRAMES))
    star_dict = {jd: np.array(calstars_list[0][1])}

    pp_start = offsetPlatepar(pp)
    params_start = pointing(pp_start)

    # Trial parameters at the true pointing: its offsets from the starting pointing (px)
    offsets = PointingOffsets(pp_start)
    params = list(offsets.fromPointing(pp.RA_d, pp.dec_d, pp.pos_angle_ref)) + [offsets.fromScale(pp.F_scale)]
    cost = CheckFit._calcImageResidualsAstro(params, config, pp_start, catalog_stars, star_dict, 5.0)

    assert pointing(pp_start) == params_start
    assert cost == pytest.approx(CheckFit.matchStarsResiduals(config, pp, catalog_stars, star_dict, 5.0),
                                 abs=1e-12)


def test_failed_check_hands_over_the_starting_platepar(monkeypatch):
    """ Every fit of the pointing stops after one iteration, so it doesn't succeed. The NN alignment (which then
        fails too) and the caller must get the starting platepar, not the last trial of the fit. """

    pp = makePlatepar()
    config = makeConfig()
    catalog_stars, calstars_data = makeNight(pp, config)
    useCatalog(monkeypatch, catalog_stars)

    # Stop every fit after one iteration
    minimize = scipy.optimize.minimize

    def _minimizeOneIteration(fun, x0, args=(), **kwargs):
        """ The minimization stopped after one iteration. """

        kwargs['options'] = dict(kwargs.get('options') or {}, maxiter=1)
        return minimize(fun, x0, args=args, **kwargs)

    monkeypatch.setattr(CheckFit.scipy.optimize, 'minimize', _minimizeOneIteration)

    # The NN alignment records the platepar it gets and fails, so it returns that platepar
    aligned = []

    def _alignPlatepar(config, platepar, *args, **kwargs):
        """ An NN alignment which doesn't converge, so it returns the given platepar. """

        aligned.append(pointing(platepar))
        return platepar, config.catalog_mag_limit

    monkeypatch.setattr(CheckFit, 'alignPlatepar', _alignPlatepar)

    # The check from a pointing 2 px off
    pp_start = offsetPlatepar(pp)
    params_start = pointing(pp_start)

    result, fit_status = CheckFit.autoCheckFit(config, pp_start, calstars_data)

    # The check failed, and both the NN alignment and the caller got the starting pointing
    assert not fit_status
    assert aligned == [params_start]
    assert pointing(result) == params_start
    assert result.aberration


def test_successful_check_returns_the_fitted_platepar(monkeypatch):
    """ A check which succeeds returns the fitted pointing (guard: the same before and after the change). """

    pp = makePlatepar()
    config = makeConfig()
    catalog_stars, calstars_data = makeNight(pp, config)
    useCatalog(monkeypatch, catalog_stars)

    # The check from a pointing 2 px off
    result, fit_status = CheckFit.autoCheckFit(config, offsetPlatepar(pp), calstars_data)

    # The fitted pointing puts the catalog stars where the true one does, within the tolerance of the check
    #   (the start is 2 px off)
    x_true, y_true = raDecToXYPP(catalog_stars[:, 0], catalog_stars[:, 1], JD_START, pp)
    x_fit, y_fit = raDecToXYPP(catalog_stars[:, 0], catalog_stars[:, 1], JD_START, result)

    assert fit_status
    assert np.max(np.hypot(x_fit - x_true, y_fit - y_true)) < 0.5


def test_recalibration_keeps_the_last_successful_fit(monkeypatch):
    """ The recalibration of an image fits the pointing with smaller and smaller match radii. When the fit with
        the 1.5 px radius doesn't succeed, the fits with the larger radii are kept, and the recalibrated platepar
        is the one of the last of them, not the last trial of the failed fit. """

    # The stars of the first image, recalibrated without the least squares refinement which would move the
    #   pointing of the search
    pp = makePlatepar()
    config = makeConfig()
    config.recalibration_lsq_refine = False
    catalog_stars, (calstars_list, _) = makeNight(pp, config)
    jd = date2JD(*FFfile.getMiddleTimeFF(calstars_list[0][0], config.fps, ret_milliseconds=True,
                                         ff_frames=FF_FRAMES))
    star_dict = {jd: np.array(calstars_list[0][1])}

    # Record the result of every fit of the pointing, and stop the one with the 1.5 px radius after one
    #   iteration (other fits, as the photometric one, are left alone)
    minimize = scipy.optimize.minimize
    fits = {}

    def _minimizeFailingAt15(fun, x0, args=(), **kwargs):
        """ The minimization, stopped after one iteration for the fit of the pointing with the 1.5 px radius. """

        # Other fits run as they are
        if fun is not CheckFit._calcImageResidualsAstro:
            return minimize(fun, x0, args=args, **kwargs)

        # The fit with the 1.5 px radius is stopped after one iteration. The fit gives the offsets of the
        #   pointing from the platepar it starts from, kept with its result
        match_radius = args[4]
        if match_radius == 1.5:
            kwargs['options'] = dict(kwargs.get('options') or {}, maxiter=1)
        res = minimize(fun, x0, args=args, **kwargs)
        fitted = copy.deepcopy(args[1])
        args[5].apply(fitted, res.x)
        fits[match_radius] = (res, fitted)

        return res

    monkeypatch.setattr(ApplyRecalibrate.scipy.optimize, 'minimize', _minimizeFailingAt15)

    # The recalibration from a pointing 2 px off
    result, min_match_radius = ApplyRecalibrate.recalibrateFF(config, offsetPlatepar(pp), jd, star_dict,
                                                              catalog_stars)

    # The fits with the radii down to 3 px succeeded, the one with 1.5 px didn't
    assert sorted(fits) == [1.5, 3, 5, 10]
    assert fits[3][0].success and (not fits[1.5][0].success)
    assert min_match_radius == 3

    # The recalibrated platepar is the result of the fit with the 3 px radius
    assert result is not None
    assert pointing(result) == pointing(fits[3][1])
