""" Tests for the star coverage gate in RMS.Astrometry.ApplyRecalibrate.recalibrateFF. """

from __future__ import print_function, division, absolute_import

import copy

import numpy as np

import RMS.ConfigReader as cr
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP
from RMS.Astrometry.ApplyRecalibrate import coverageMatchFraction, recalibrateFF
from RMS.Astrometry.Conversions import JD2HourAngle
from RMS.Formats.Platepar import Platepar


def _setup():
    """ Return a config and a consistent 60 deg FOV platepar at its own reference time. """

    config = cr.Config()
    config.width, config.height = 1280, 720

    pp = Platepar()
    pp.X_res, pp.Y_res = config.width, config.height
    pp.F_scale = pp.X_res/60.0
    pp.JD = 2460000.5
    pp.RA_d, pp.dec_d = 100.0, 30.0
    pp.pos_angle_ref = 10.0
    pp.lat, pp.lon = 45.0, 15.0
    pp.Ho = JD2HourAngle(pp.JD)%360

    config.latitude, config.longitude = pp.lat, pp.lon

    return config, pp


def _catalogInImage(pp, n_stars, rng):
    """ Return catalog stars (RA, dec, mag) at random positions inside the image, and their x, y. """

    xs = rng.uniform(30, pp.X_res - 30, n_stars)
    ys = rng.uniform(30, pp.Y_res - 30, n_stars)
    _, ra, dec, _ = xyToRaDecPP(n_stars*[pp.JD], xs, ys, np.ones(n_stars), pp, extinction_correction=False,
        jd_time=True)
    mags = rng.uniform(1.0, 4.4, n_stars)

    return np.c_[ra, dec, mags], xs, ys


def _imageStars(xs, ys, mags):
    """ Build CALSTARS-like rows (y, x, intensity, amplitude) for the given positions and magnitudes. """

    return np.c_[ys, xs, 10**(-0.4*(mags - 12)), np.ones(len(xs))]


def testCoverageGateAcceptsCameraDetectingBelowCatalogLimit():
    """ A camera detecting many stars fainter than the catalog limit must not be rejected.

        All 100 catalog stars in the image are detected and matched, but 150 more detections are
        fainter than the catalog limit, so only 100/250 = 0.40 of the detections can ever match.
    """

    config, pp = _setup()
    rng = np.random.default_rng(1)

    catalog, xs, ys = _catalogInImage(pp, 100, rng)

    # Detections of all catalog stars plus faint stars with no catalog counterpart
    fx = rng.uniform(30, pp.X_res - 30, 150)
    fy = rng.uniform(30, pp.Y_res - 30, 150)
    star_list = np.r_[_imageStars(xs, ys, catalog[:, 2]), _imageStars(fx, fy, np.full(150, 5.5))]

    result, _ = recalibrateFF(config, copy.deepcopy(pp), pp.JD, {pp.JD: star_list}, catalog, lim_mag=4.5,
        ignore_max_stars=True)

    assert result is not None
    assert len(result.star_list) == 100

    # The fraction is taken of the 100 possible matches
    fraction, n_possible = coverageMatchFraction(100, 250, catalog, pp.JD, pp)
    assert n_possible == 100
    assert np.isclose(fraction, 1.0)


def testCoverageGateRejectsSparseSpuriousMatch():
    """ A fit which matched only a small subset of the stars (29 of 308) must still be rejected. """

    config, pp = _setup()
    rng = np.random.default_rng(2)

    catalog, xs, ys = _catalogInImage(pp, 400, rng)

    # Only 29 detections coincide with catalog stars, the rest are at unrelated positions
    fx = rng.uniform(30, pp.X_res - 30, 279)
    fy = rng.uniform(30, pp.Y_res - 30, 279)
    star_list = np.r_[_imageStars(xs[:29], ys[:29], catalog[:29, 2]), _imageStars(fx, fy, np.full(279, 3.0))]

    # Force the goodness and distance checks so only the coverage gate can reject the fit
    result, _ = recalibrateFF(config, copy.deepcopy(pp), pp.JD, {pp.JD: star_list}, catalog, lim_mag=4.5,
        ignore_max_stars=True, force_platepar_save=True, ignore_distance_threshold=True)

    assert result is None

    # The fraction is taken of the 308 detections, as the catalog has more stars in the image
    fraction, n_possible = coverageMatchFraction(29, 308, catalog, pp.JD, pp)
    assert n_possible == 308
    assert fraction < config.recalibration_min_match_fraction


def testCoverageMatchFractionEmptyCatalog():
    """ No catalog stars in the image gives a zero fraction instead of a division by zero. """

    _, pp = _setup()

    fraction, n_possible = coverageMatchFraction(0, 100, np.zeros((0, 3)), pp.JD, pp)

    assert n_possible == 0
    assert fraction == 0.0


def testNeighbourPhotometryAveragingReplacesDegenerateZeroPoint():
    """ An FF with a degenerate photometric fit (stddev 0, mag_lev 10 from the initial guess) must not
        contribute to the neighbourhood average, but must receive it.
    """

    from types import SimpleNamespace

    from RMS.Astrometry.ApplyRecalibrate import averageNeighbourPhotometry

    calstars_ffs = ['FF_0', 'FF_1', 'FF_2', 'FF_3']
    recalibrated = {
        'FF_1': SimpleNamespace(mag_lev=11.0, mag_lev_stddev=0.1),
        'FF_2': SimpleNamespace(mag_lev=10.0, mag_lev_stddev=0.0),
        'FF_3': SimpleNamespace(mag_lev=11.2, mag_lev_stddev=0.1),
    }

    averageNeighbourPhotometry(recalibrated, ['FF_2'], calstars_ffs)

    # The average only uses FF_1 and FF_3, and all three FFs receive it
    for ff_name in calstars_ffs[1:]:
        assert np.isclose(recalibrated[ff_name].mag_lev, 11.1)
        assert recalibrated[ff_name].mag_lev_stddev > 0
