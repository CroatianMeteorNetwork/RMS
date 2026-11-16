"""
Tests of the least squares refinement of the pointing in the recalibration on individual images
(ApplyRecalibrate.refinePointingLSQ, used by recalibrateFF when recalibration_lsq_refine is on).

The search of recalibrateFF stops within its tolerance, so a recalibration keeps a part of a pixel of the error of
the platepar it starts from. The refinement fits the pointing exactly on the matched stars. The tests use a
synthetic image: stars of a real 1280x720 calibration (radial7-odd with the asymmetry correction) at known sky
positions, measured with a small noise, and a starting platepar whose scale, rotation and pointing are off by less
than the tolerance of the search.
"""

import copy
import types

import numpy as np

import RMS.ConfigReader as cr
from RMS.Astrometry import ApplyRecalibrate
from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, xyToRaDecPP
from RMS.Formats.Platepar import Platepar


# Number of stars of the synthetic image, and the noise of their positions (px)
N_STARS = 150
NOISE_PX = 0.05

# Smallest match radius of the recalibration of a 1280x720 image (px), the offset of the starting pointing
#   in the tests of the stars lost by the refinement (px), and how far inside the match radius the lost stars
#   sit at the start (px)
MATCH_RADIUS = 0.5
START_OFFSET_PX = 0.2
EDGE_MARGIN_PX = 0.05


def makePlatepar():
    """ A real 1280x720 calibration, radial7-odd with the asymmetry correction and an equal aspect. """

    pp = Platepar()
    pp.lat, pp.lon, pp.elev = 50.192509, 5.254718, 295.0
    pp.X_res, pp.Y_res = 1280, 720
    pp.F_scale = 15.600916448923877
    pp.RA_d, pp.dec_d, pp.pos_angle_ref = 349.8434689573242, -4.063815352866793, 19.34236856738213
    pp.JD, pp.Ho = 2461018.279334375, 358.3338455827907
    pp.refraction = True
    pp.equal_aspect, pp.asymmetry_corr, pp.force_distortion_centre = True, True, False
    pp.setDistortionType("radial7-odd", reset_params=True)
    pp.x_poly_fwd = np.array([-0.0017811686386760266, 0.031678484905771256, -0.007740143231304088,
                              -1.0140484087110697, 0.06835307986186301, 0.010258192855205479,
                              0.005386010898662939])
    pp.x_poly_rev = np.array([-0.0017856520990328477, 0.03170770867400316, 0.007074982440201784,
                              0.4820702423628897, 0.06802381881224333, -0.001903408962282129,
                              0.00014273818898309323])
    pp.y_poly_fwd = np.array([0.5, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    pp.y_poly_rev = np.array([0.5, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    pp.mag_lev = 10.5
    pp.vignetting_coeff = 0.001

    return pp


def makeConfig():
    """ The default configuration for a 1280x720 camera, with a catalog limit deeper than the synthetic stars. """

    config = cr.Config()
    config.width, config.height = 1280, 720
    config.catalog_mag_limit = 7.0
    config.recalibration_lsq_refine = True

    return config


def makeImage(pp, seed=0, n_stars=N_STARS):
    """ A synthetic image: catalog stars spread over the image (at least 20 px apart, so every star has one
        match), and the image stars at their positions with the noise.

    Keyword arguments:
        seed: [int] Seed of the random numbers. 0 by default.
        n_stars: [int] Number of stars. N_STARS by default.

    Return:
        (catalog_stars, star_rows): [tuple] Catalog stars (ra, dec, mag) and the image stars as rows of a
            CALSTARS file (y, x, intensity, amplitude, FWHM, background, SNR, saturated).
    """

    rng = np.random.default_rng(seed)

    # Random image positions far enough apart
    xy = []
    while len(xy) < n_stars:
        x, y = rng.uniform(30, pp.X_res - 30), rng.uniform(30, pp.Y_res - 30)
        if all(np.hypot(x - xi, y - yi) > 20 for xi, yi in xy):
            xy.append((x, y))
    x, y = np.array(xy).T

    # Their sky positions are the catalog, the image stars are where the platepar puts the catalog stars (the
    #   forward mapping, which is the one the recalibration fits), with the noise
    _, ra, dec, _ = xyToRaDecPP(np.full(n_stars, pp.JD), x, y, np.ones(n_stars), pp,
                                extinction_correction=False, jd_time=True)
    ra, dec = np.asarray(ra), np.asarray(dec)
    mag = rng.uniform(2.0, 6.0, n_stars)
    x_img, y_img = raDecToXYPP(ra, dec, pp.JD, pp)
    x_img = x_img + rng.normal(0, NOISE_PX, n_stars)
    y_img = y_img + rng.normal(0, NOISE_PX, n_stars)

    # The intensities follow the photometry of the platepar
    intensity = 10**((pp.mag_lev - mag)/2.5)
    rows = np.column_stack([y_img, x_img, intensity, intensity/10, np.full(n_stars, 2.5),
                            np.full(n_stars, 10.0), np.full(n_stars, 50.0), np.zeros(n_stars)])

    return np.column_stack([ra, dec, mag]), rows


def perturbed(pp):
    """ The platepar with the scale and the rotation off by 0.2 px at the corners and the pointing off by 0.1 px,
        all less than the tolerance of the search (dist_check_threshold, 0.33 px by default). """

    pp = copy.deepcopy(pp)
    r_corner = np.hypot(pp.X_res, pp.Y_res)/2
    pp.F_scale *= 1 + 0.2/r_corner
    pp.pos_angle_ref += np.degrees(0.2/r_corner)
    pp.RA_d += 0.1/pp.F_scale/np.cos(np.radians(pp.dec_d))
    pp.dec_d -= 0.1/pp.F_scale

    return pp


def displacement(pa, pb, catalog_stars):
    """ The largest distance (px) between the image positions of the catalog stars with two platepars. """

    xa, ya = raDecToXYPP(catalog_stars[:, 0], catalog_stars[:, 1], pa.JD, pa)
    xb, yb = raDecToXYPP(catalog_stars[:, 0], catalog_stars[:, 1], pa.JD, pb)

    return np.max(np.hypot(xa - xb, ya - yb))


def recalibrate(pp_start, rows, catalog_stars, config):
    """ The recalibration of the synthetic image from the starting platepar. """

    jd = pp_start.JD
    result, _ = ApplyRecalibrate.recalibrateFF(config, copy.deepcopy(pp_start), jd, {jd: rows}, catalog_stars)
    assert result is not None

    return result


def test_refinement_recovers_the_pointing():
    """ From a start off by less than the tolerance of the search, the refined recalibration is as close to the
        true pointing as the noise of the stars allows, and closer than the search alone. """

    pp = makePlatepar()
    catalog_stars, rows = makeImage(pp)
    config = makeConfig()

    # The recalibration with the refinement, and with the search alone
    refined = recalibrate(perturbed(pp), rows, catalog_stars, config)

    config.recalibration_lsq_refine = False
    searched = recalibrate(perturbed(pp), rows, catalog_stars, config)

    # The search alone keeps most of the error of the start, the refinement removes it

    assert displacement(pp, refined, catalog_stars) < 0.03
    assert displacement(pp, refined, catalog_stars) < displacement(pp, searched, catalog_stars)


def test_refinement_does_not_depend_on_the_start():
    """ The refined recalibrations from the true and from the perturbed platepar agree. """

    pp = makePlatepar()
    catalog_stars, rows = makeImage(pp, seed=1)
    config = makeConfig()

    # The same image recalibrated from two starts
    from_true = recalibrate(pp, rows, catalog_stars, config)
    from_perturbed = recalibrate(perturbed(pp), rows, catalog_stars, config)

    assert displacement(from_true, from_perturbed, catalog_stars) < 0.005


def test_wrong_pairs_move_the_pointing_less_than_their_mean_offset():
    """ A tenth of the image stars 0.45 px off their catalog stars in the same direction (inside the smallest
        match radius) move the refined pointing less than they would move an average of the offsets (0.045 px),
        as the loss of the fit is robust. """

    pp = makePlatepar()
    catalog_stars, rows = makeImage(pp, seed=2)
    config = makeConfig()
    clean = recalibrate(perturbed(pp), rows, catalog_stars, config)

    # The same image with the wrong pairs
    rows = rows.copy()
    n_wrong = N_STARS//10
    rows[:n_wrong, 1] += 0.45
    refined = recalibrate(perturbed(pp), rows, catalog_stars, config)

    assert displacement(clean, refined, catalog_stars) < n_wrong/N_STARS*0.45


def test_refinement_not_used_with_too_few_stars():
    """ With fewer matched stars than min_matched_stars, the refinement is not used. """

    pp = makePlatepar()
    catalog_stars, rows = makeImage(pp, seed=3)
    config = makeConfig()
    rows = rows[:config.min_matched_stars - 1]

    # The refinement called directly on the few stars
    refined, matched = ApplyRecalibrate.refinePointingLSQ(config, pp, pp.JD, {pp.JD: rows}, catalog_stars, 0.5,
                                                          0.5, len(rows))

    assert (refined is None) and (matched is None)


def test_refinement_off_in_the_config(monkeypatch):
    """ With recalibration_lsq_refine off, the recalibration doesn't refine the pointing. """

    pp = makePlatepar()
    catalog_stars, rows = makeImage(pp, seed=4)
    config = makeConfig()
    config.recalibration_lsq_refine = False

    def _fail(*args, **kwargs):
        """ The refinement must not be called. """

        raise AssertionError('the refinement was used')

    # The recalibration fails the test if it calls the refinement

    monkeypatch.setattr(ApplyRecalibrate, 'refinePointingLSQ', _fail)

    recalibrate(perturbed(pp), rows, catalog_stars, config)


def test_refinement_kept_out_when_not_good_enough(monkeypatch):
    """ A refined pointing which fails the goodness check (here 0.45 px off, above dist_check_threshold) doesn't
        replace the result of the search, so the recalibration still succeeds with it. """

    pp = makePlatepar()
    catalog_stars, rows = makeImage(pp, seed=5)
    config = makeConfig()

    # The result of the search alone
    config.recalibration_lsq_refine = False
    searched = recalibrate(perturbed(pp), rows, catalog_stars, config)

    def _offPointing(config, platepar, jd, star_dict_ff, catalog_stars, fit_radius, check_radius, n_check,
                     lim_mag=None):
        """ A refinement which moves the pointing by 0.45 px. """

        off = copy.deepcopy(platepar)
        off.dec_d += 0.45/off.F_scale
        _, _, _, matched = ApplyRecalibrate.CheckFit.matchStarsResiduals(config, off, catalog_stars, star_dict_ff,
                                                                         check_radius, ret_nmatch=True)

        return off, matched

    # The recalibration with the refinement replaced by the one which is off
    config.recalibration_lsq_refine = True
    monkeypatch.setattr(ApplyRecalibrate, 'refinePointingLSQ', _offPointing)
    result = recalibrate(perturbed(pp), rows, catalog_stars, config)

    assert displacement(searched, result, catalog_stars) < 1e-6


def test_forced_fallback_not_refined(monkeypatch):
    """ The fallback recalibration, limited to larger radii and saved without the goodness check, keeps the result
        of the search: its loose pairs would pull the least squares fit. """

    pp = makePlatepar()
    catalog_stars, rows = makeImage(pp, seed=6)
    config = makeConfig()

    def _fail(*args, **kwargs):
        """ The refinement must not be called. """

        raise AssertionError('the refinement was used')

    # The forced recalibration with the refinement on fails the test if it calls the refinement
    monkeypatch.setattr(ApplyRecalibrate, 'refinePointingLSQ', _fail)
    jd = pp.JD
    result, _ = ApplyRecalibrate.recalibrateFF(config, perturbed(pp), jd, {jd: rows}, catalog_stars,
                                               max_match_radius=1.5, force_platepar_save=True)
    assert result is not None



def makeLostStarsImage(pp, n_stars, n_lost, seed=0):
    """ A synthetic image on which the refinement loses some of the stars matched with the starting platepar.
        The start is off the true pointing by START_OFFSET_PX and n_lost image stars sit just inside the match
        radius from where the start puts their catalog stars, on the side away from the true position, so they
        are matched at the start and are out of the match radius once the pointing is refined.

    Arguments:
        pp: [Platepar instance] The true platepar.
        n_stars: [int] Number of stars.
        n_lost: [int] Number of stars which the refinement loses.

    Keyword arguments:
        seed: [int] Seed of the random numbers. 0 by default.

    Return:
        (pp_start, catalog_stars, star_rows): [tuple] The starting platepar, the catalog stars (ra, dec, mag)
            and the image stars as rows of a CALSTARS file.
    """

    catalog_stars, rows = makeImage(pp, seed=seed, n_stars=n_stars)

    # The start is off the true pointing in declination
    pp_start = copy.deepcopy(pp)
    pp_start.dec_d -= START_OFFSET_PX/pp.F_scale

    # Where the true and the starting platepar put the catalog stars
    x_true, y_true = raDecToXYPP(catalog_stars[:, 0], catalog_stars[:, 1], pp.JD, pp)
    x_start, y_start = raDecToXYPP(catalog_stars[:, 0], catalog_stars[:, 1], pp.JD, pp_start)

    # The stars to be lost are moved from the start position further away from the true one, to just inside
    #   the match radius
    dx, dy = x_start - x_true, y_start - y_true
    dist = np.hypot(dx, dy)
    edge = MATCH_RADIUS - EDGE_MARGIN_PX
    rows = rows.copy()
    rows[:n_lost, 1] = x_start[:n_lost] + edge*dx[:n_lost]/dist[:n_lost]
    rows[:n_lost, 0] = y_start[:n_lost] + edge*dy[:n_lost]/dist[:n_lost]

    return pp_start, catalog_stars, rows


def refineFromStart(pp_start, rows, catalog_stars, config):
    """ The refinement called directly from the starting platepar, with the smallest match radius both for the
        fit and for the checks after the recalibration.

    Return:
        (refined, matched, n_start): [tuple] The refined platepar and its matched stars (None, None if the
            refinement isn't used), and the number of stars matched with the starting platepar.
    """

    jd = pp_start.JD
    star_dict = {jd: rows}

    # The stars matched at the start, the count which the refinement compares its own with
    n_start, _, _, _ = ApplyRecalibrate.CheckFit.matchStarsResiduals(config, pp_start, catalog_stars,
                                                                     star_dict, MATCH_RADIUS, ret_nmatch=True)

    refined, matched = ApplyRecalibrate.refinePointingLSQ(config, pp_start, jd, star_dict, catalog_stars,
                                                          MATCH_RADIUS, MATCH_RADIUS, n_start)

    return refined, matched, n_start


def test_refinement_used_when_one_star_of_100_is_lost():
    """ A refinement which lowers the loss and moves one of 100 matched stars out of the match radius (fewer
        than the allowed 2 %) is used, and it finds the true pointing. """

    pp = makePlatepar()
    config = makeConfig()
    pp_start, catalog_stars, rows = makeLostStarsImage(pp, 100, 1, seed=7)

    refined, matched, n_start = refineFromStart(pp_start, rows, catalog_stars, config)

    # All stars are matched at the start, one fewer with the refined pointing, which is the true one
    assert n_start == 100
    assert refined is not None
    assert len(matched[pp.JD][0]) == 99
    assert displacement(pp, refined, catalog_stars) < 0.03


def test_refinement_not_used_when_3_stars_of_100_are_lost():
    """ A refinement which lowers the loss but moves 3 of 100 matched stars out of the match radius (more than
        the allowed 2 %) is not used. """

    pp = makePlatepar()
    config = makeConfig()
    pp_start, catalog_stars, rows = makeLostStarsImage(pp, 100, 3, seed=8)

    refined, matched, n_start = refineFromStart(pp_start, rows, catalog_stars, config)

    assert n_start == 100
    assert (refined is None) and (matched is None)


def test_refinement_used_when_one_star_of_20_is_lost():
    """ With few stars, where 2 % of them is less than one star, the refinement may still lose one. The
        minimum number of matched stars is lowered so that 19 stars are enough for the recalibration. """

    pp = makePlatepar()
    config = makeConfig()
    config.min_matched_stars = 10
    pp_start, catalog_stars, rows = makeLostStarsImage(pp, 20, 1, seed=9)

    refined, matched, n_start = refineFromStart(pp_start, rows, catalog_stars, config)

    assert n_start == 20
    assert refined is not None
    assert len(matched[pp.JD][0]) == 19
    assert displacement(pp, refined, catalog_stars) < 0.05


def test_refinement_not_used_with_a_higher_loss(monkeypatch):
    """ A refinement which loses no stars but has a higher loss on the stars matched at the start is not used.
        The fit is replaced by one which moves the true pointing by a quarter of a pixel, so all stars stay
        matched and fit worse. """

    pp = makePlatepar()
    config = makeConfig()
    catalog_stars, rows = makeImage(pp, seed=10, n_stars=100)

    calls = []

    def _offFit(fun, x0, **kwargs):
        """ A fit which moves the pointing by 0.25 px in RA in the first round and doesn't move it after. """

        x = np.zeros_like(x0)
        if not calls:
            x[0] = 0.25
        calls.append(x)

        return types.SimpleNamespace(x=x)

    # The refinement from the true pointing with the fit replaced
    monkeypatch.setattr(ApplyRecalibrate.scipy.optimize, 'least_squares', _offFit)
    refined, matched, n_start = refineFromStart(pp, rows, catalog_stars, config)

    assert n_start == 100
    assert (refined is None) and (matched is None)
