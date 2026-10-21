""" Recalibrate the platepar for every FF with detections and compute the astrometry with recalibrated
    values. 
"""

from __future__ import absolute_import, division, print_function

import argparse
import copy
import datetime
import json
import os
import shutil
import sys
import traceback
import logging
from collections import OrderedDict

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import RMS.ConfigReader as cr
import scipy.optimize
import Utils.CalibrationReport
import Utils.RMS2UFO
from RMS.Astrometry import CheckFit
from RMS.Astrometry.ApplyAstrometry import (
    applyAstrometryFTPdetectinfo,
    applyPlateparToCentroids,
    extinctionCorrectionTrueToApparent,
    getFOVSelectionRadius,
    photometryFitRobust,
    raDecToXYPP,
    rotationWrtHorizon,
    xyToRaDecPP,
)
from RMS.Astrometry.CyFunctions import subsetCatalog
from RMS.Astrometry.Conversions import date2JD, jd2Date
from RMS.Astrometry.NNalign import alignPlatepar
from RMS.Formats import CALSTARS, FFfile, FTPdetectinfo, Platepar, StarCatalog
from RMS.Formats.ECSV import writeECSV
from RMS.Formats.FTPdetectinfo import findFTPdetectinfoFile, validDefaultFTPdetectinfo
from RMS.Math import angularSeparation
from RMS.Logger import LoggingManager, getLogger
from RMS.Misc import RmsDateTime

# Neighbourhood size around individual FFs with detections which will be takes for recalibration
#   A size of e.g. 3 means that an FF before, the FF with the detection, an an FF after will be taken
RECALIBRATE_NEIGHBOURHOOD_SIZE = 3

# Least squares refinement of the pointing: at most this many rounds of matching and fitting, and the largest
#   change of the star positions (px) at which the refinement has converged
LSQ_REFINE_MAX_ROUNDS = 5
LSQ_REFINE_CONVERGED_PX = 0.01

# Get the logger from the main module
log = getLogger("rmslogger", level="INFO")


def loadRecalibratedPlatepar(dir_path, config, file_list=None, type='meteor'):
    """
    Gets recalibrated platepars. If they were already computed, load them, otherwise compute them and save

    Arguments:
        dir_path: [str] Path to the directory which contains the platepar and recalibrated platepars
            from ftpdetectinfo
        config: [config object]

    Keyword arguments:
        type: [str] 'meteor' or 'flux'

    Return:
        recalibrated_platepars: [dict] If platepar doesn't exist returns None

    """
    if type == 'meteor':
        platepar_file_name = config.platepars_recalibrated_name
    else:
        platepar_file_name = config.platepars_flux_recalibrated_name

    if not file_list:
        file_list = os.listdir(dir_path)

    # Find and load recalibrated platepars
    if platepar_file_name in file_list:
        with open(os.path.join(dir_path, platepar_file_name)) as f:

            try:
                # Load the JSON file with recalibrated platepars
                recalibrated_platepars_dict = json.load(f)
            
            except json.decoder.JSONDecodeError:
                return None

            log.info("Recalibrated platepars loaded!")
            # Convert the dictionary of recalibrated platepars to a dictionary of Platepar objects
            recalibrated_platepars = {}
            for ff_name in recalibrated_platepars_dict:
                pp = Platepar.Platepar()
                pp.loadFromDict(recalibrated_platepars_dict[ff_name], use_flat=config.use_flat)

                recalibrated_platepars[ff_name] = pp

        return recalibrated_platepars

    return None


def refinePointingLSQ(config, platepar, jd, star_dict_ff, catalog_stars, fit_radius, check_radius, n_check,
                      lim_mag=None):
    """ Refine the pointing (reference RA, Dec, position angle and scale) of a recalibrated platepar by least
        squares on the matched stars.

        The search in recalibrateFF minimizes a cost which changes in steps as stars enter and leave the match
        radius, so it stops within its tolerance (about dist_check_threshold) and keeps a part of a pixel of the
        error of the platepar it started from. With the pairs of image and catalog stars fixed, the positions
        change smoothly with the pointing and least squares finds the exact optimum. The stars are matched again
        with the refined pointing and the fit is repeated until the pointing no longer changes, so the choice of
        the stars doesn't tie the result to the starting pointing.

    Arguments:
        config: [Config instance]
        platepar: [Platepar instance] Recalibrated platepar.
        jd: [float] Julian date of the stars.
        star_dict_ff: [dict] A dictionary with one entry, where the key is 'jd' and the value is the list of
            star coordinates.
        catalog_stars: [ndarray] Catalog stars (ra, dec, mag).
        fit_radius: [float] Radius (px) within which the stars are matched for the fit.
        check_radius: [float] Radius (px) of the matched stars which are used after the recalibration (the
            coverage check and the photometry).
        n_check: [int] Number of stars matched within check_radius with the given platepar.

    Keyword arguments:
        lim_mag: [float] Override the limiting magnitude from config. None by default.

    Return:
        (platepar, matched_stars): [tuple] The refined platepar and its matched stars within check_radius, or
            (None, None) if the refinement isn't used: too few stars, fewer stars matched, or a higher loss
            (the robust one of the fit) on the stars matched with the given platepar.
    """

    # The stars matched with the given platepar, the pairs which the refinement must fit better. Too few stars
    #   can't constrain the four parameters
    n_start, _, _, matched = CheckFit.matchStarsResiduals(config, platepar, catalog_stars, star_dict_ff,
                                                          fit_radius, ret_nmatch=True, lim_mag=lim_mag)
    if (n_start < config.min_matched_stars) or (jd not in matched):
        log.debug('Least squares refinement of the pointing not used: {:d} stars within {:.2f} px'.format(
            n_start, fit_radius))
        return None, None

    img_start, cat_start, _ = matched[jd]


    def _positions(pp, cat_stars):
        """ Image positions (x, y) of catalog stars with the platepar. """

        return raDecToXYPP(cat_stars[:, 0], cat_stars[:, 1], jd, pp)


    # Scale of the robust loss (px): distances well below it count as squares, larger ones about linearly
    f_scale = fit_radius/2.0


    def _robustCost(pp, img_stars, cat_stars):
        """ The loss which the fit minimizes (soft L1 of the distances of the image stars from their catalog
            stars), to compare two pointings on the same pairs. """

        x_cat, y_cat = _positions(pp, cat_stars)
        z = ((img_stars[:, 1] - x_cat)**2 + (img_stars[:, 0] - y_cat)**2)/f_scale**2

        return np.sum(2*(np.sqrt(1 + z) - 1))


    # The fitted parameters, on a copy of the platepar (the given one is kept if the refinement isn't used)
    keys = ('RA_d', 'dec_d', 'pos_angle_ref', 'F_scale')
    refined = copy.deepcopy(platepar)

    # The parameters are fitted in steps which move the stars by about one pixel (the rotation and the scale at
    #   a quarter of the image diagonal), so the fit is equally well conditioned in all of them
    r_typ = np.hypot(platepar.X_res, platepar.Y_res)/4.0
    steps = np.array([1.0/(platepar.F_scale*max(np.cos(np.radians(platepar.dec_d)), 0.1)),
                      1.0/platepar.F_scale, np.degrees(1.0/r_typ), platepar.F_scale/r_typ])

    # Fit the pointing on the matched pairs, match the stars again with it and repeat until it converges
    img_stars, cat_stars = img_start, cat_start
    for _ in range(LSQ_REFINE_MAX_ROUNDS):

        # The pointing at the start of the round, and where it puts the stars
        p0 = np.array([getattr(refined, k) for k in keys])
        x_before, y_before = _positions(refined, cat_stars)

        def _residuals(q):
            """ Distances of the image stars from their catalog stars (px) with the pointing p0 + q*steps. """

            for k, v in zip(keys, p0 + q*steps):
                setattr(refined, k, v)
            x_cat, y_cat = _positions(refined, cat_stars)

            return np.concatenate([img_stars[:, 1] - x_cat, img_stars[:, 0] - y_cat])

        # The fit has a robust loss, so a wrong pair pulls the pointing less than its share of the stars
        fit = scipy.optimize.least_squares(_residuals, np.zeros(len(keys)), loss='soft_l1',
                                           f_scale=f_scale, diff_step=1e-2)
        for k, v in zip(keys, p0 + fit.x*steps):
            setattr(refined, k, v)

        # Converged when the stars no longer move
        x_after, y_after = _positions(refined, cat_stars)
        if np.max(np.hypot(x_after - x_before, y_after - y_before)) < LSQ_REFINE_CONVERGED_PX:
            break

        # Match the stars again with the refined pointing
        n_round, _, _, matched = CheckFit.matchStarsResiduals(config, refined, catalog_stars, star_dict_ff,
                                                              fit_radius, ret_nmatch=True, lim_mag=lim_mag)
        if (n_round < config.min_matched_stars) or (jd not in matched):
            log.debug('Least squares refinement of the pointing not used: {:d} stars within {:.2f} px after a '
                      'round'.format(n_round, fit_radius))
            return None, None
        img_stars, cat_stars, _ = matched[jd]

    # The refinement is used only if it matches at least as many stars, both for the fit and for the checks
    #   after the recalibration, and has a lower loss on the stars matched with the given platepar
    n_fit, _, _, _ = CheckFit.matchStarsResiduals(config, refined, catalog_stars, star_dict_ff, fit_radius,
                                                  ret_nmatch=True, lim_mag=lim_mag)
    n_refined, _, _, matched_check = CheckFit.matchStarsResiduals(config, refined, catalog_stars, star_dict_ff,
                                                                  check_radius, ret_nmatch=True, lim_mag=lim_mag)
    cost_start, cost_refined = _robustCost(platepar, img_start, cat_start), _robustCost(refined, img_start, cat_start)
    if (n_fit < n_start) or (n_refined < n_check) or (jd not in matched_check) or (cost_refined > cost_start):

        log.info('Least squares refinement of the pointing not used: stars within {:.2f} px {:d} (before {:d}), '
                 'within {:.2f} px {:d} (before {:d}), loss on the first pairs {:.3f} (before {:.3f})'.format(
                     fit_radius, n_fit, n_start, check_radius, n_refined, n_check, cost_refined, cost_start))

        return None, None

    # The change of the pointing, for the log
    log.info('Pointing refined by least squares on {:d} stars: RA {:+.4f}, Dec {:+.4f}, rotation {:+.4f} deg, '
             'scale {:+.6f} px/deg'.format(len(img_stars), refined.RA_d - platepar.RA_d,
                                           refined.dec_d - platepar.dec_d,
                                           refined.pos_angle_ref - platepar.pos_angle_ref,
                                           refined.F_scale - platepar.F_scale))

    return refined, matched_check


def recalibrateFF(
    config,
    working_platepar,
    jd,
    star_dict_ff,
    catalog_stars,
    max_match_radius=None,
    force_platepar_save=False,
    lim_mag=None,
    ignore_distance_threshold=False,
    ignore_max_stars=False,
    debug=False,
    ff_name=None,
    min_match_fraction=None,
):
    """Given the platepar and a list of stars on one image, try to recalibrate the platepar to achieve
        the best match by brute force star matching.
    Arguments:
        config: [Config instance]
        working_platepar: [Platepar instance] Platepar to recalibrate.
        jd: [float] Julian date of the star positions.
        star_dict_ff: [dict] A dictionary with only one entry, where the key is 'jd' and the value is the
            list of star coordinates.
        catalog_stars: [ndarray] A numpy array of catalog stars which should be on the image.
    Keyword arguments:
        max_match_radius: [float] Maximum radius used for star matching. None by default, which uses all 
            hardcoded values.
        force_platepar_save: [bool] Skip the goodness of fit check and save the platepar. Note that
            the star coverage gate (min_match_fraction) still applies, so a fit that matched only a
            small fraction of the detected stars is rejected even when this is True.
        ignore_distance_threshold: [bool] Don't consider the recalib as failed if the median distance
            is larger than the threshold.
        ignore_max_stars: [bool] Ignore the maximum number of image stars for recalibration.
        debug: [bool] If True, show a debug plot after each successful fit iteration showing catalog
            stars, calstars, and matched stars.
        ff_name: [str] Name of the FF file being processed. Used for debug plot labels. None by default.
        min_match_fraction: [float] Minimum fraction of the detected stars that the fit must match
            to be accepted. None by default, which uses config.recalibration_min_match_fraction.
            Set to 0 to disable the coverage gate (e.g. for interactive use).

    Return:
        result: [?] A Platepar instance if refinement is successful, None if it failed.
        min_match_radius: [float] Minimum radius that successfully matched the stars (pixels).
    """

    working_platepar = copy.deepcopy(working_platepar)

    # Resolve the coverage gate threshold from the config unless explicitly given
    if min_match_fraction is None:
        min_match_fraction = config.recalibration_min_match_fraction

    # If there more stars than a set limit, sample them randomly using the same seed for reproducibility
    if not ignore_max_stars and len(star_dict_ff[jd]) > config.recalibration_max_stars:

        # Make a copy so that the original star dictionary is not modified
        star_dict_ff = copy.deepcopy(star_dict_ff)

        # Python 3+
        if hasattr(np.random, 'default_rng'):

            # Use the newer Generator-based RNG
            rng = np.random.default_rng(seed=0)

            # Sample the stars and store them in a copy of the star dictionary
            star_dict_ff = {jd: rng.choice(star_dict_ff[jd], config.recalibration_max_stars, replace=False)}
        
        # Python 2
        else:

            # Use the older RandomState-based RNG
            rng = np.random.RandomState(seed=0)

            # RandomState.choice requires indices for complex data types
            indices = rng.choice(
                len(star_dict_ff[jd]),
                config.recalibration_max_stars,
                replace=False
            )

            # Use the indices to select the stars
            star_dict_ff[jd] = [star_dict_ff[jd][i] for i in indices]


    # A list of matching radiuses to try
    min_radius = 0.5
    max_radius = 10
    radius_list = [max_radius, 5, 3, 1.5, min_radius]

    # Scale these values to the image resolution, taking 720p as the base resolution
    scaling_factor = np.hypot(config.width, config.height)/np.hypot(1280, 720)
    if scaling_factor < 1: 
        scaling_factor = 1.0
    radius_list = [scaling_factor*r for r in radius_list]


    # Calculate the function tolerance, so the desired precision can be reached (the number is calculated
    # in the same regard as the cost function)
    fatol, xatol_ang = CheckFit.computeMinimizationTolerances(config, working_platepar, len(star_dict_ff))

    ### If the initial match is good enough, do only quick recalibration ###

    # Match the stars and calculate the residuals
    n_matched, avg_dist, cost, _ = CheckFit.matchStarsResiduals(
        config, working_platepar, catalog_stars, star_dict_ff, min_radius, ret_nmatch=True, lim_mag=lim_mag
    )

    log.info(
        'Initially matched stars within {:.1f} px: {:d}/{:d}'.format(min_radius, n_matched, len(star_dict_ff[jd]))
    )

    # If at least half the stars are matched with the smallest radius
    # Check if the average distance with the tightest radius is close
    if (n_matched >= 0.5*len(star_dict_ff[jd])) and (avg_dist < config.dist_check_quick_threshold):
        # Use a reduced set of initial radius values
        radius_list = [1.5, min_radius]

        log.info('Using a quick fit...')

    else:
        log.info(
            'Initial match failed, using radiuses: {}'.format(radius_list)
        )

    ##########

    # Pre-filter catalog to FOV region. This avoids re-subsetting the full catalog (potentially
    # 50k+ stars) on every optimizer iteration. Safe because matched stars are always well inside
    # the FOV boundary — stars aren't detected within ~15 px of image edges and the max match
    # radius is 10 px, so stars entering/leaving the subset boundary can never affect matching.
    fov_radius = getFOVSelectionRadius(working_platepar)
    _, RA_c, dec_c, _ = xyToRaDecPP(
        [jd2Date(jd)], [working_platepar.X_res/2], [working_platepar.Y_res/2], [1],
        working_platepar, extinction_correction=False
    )
    effective_lim_mag = lim_mag if lim_mag is not None else config.catalog_mag_limit
    _, catalog_stars = subsetCatalog(
        catalog_stars, RA_c[0], dec_c[0], jd,
        working_platepar.lat, working_platepar.lon,
        fov_radius, effective_lim_mag
    )

    log.info('Pre-filtered catalog to {:g} stars within {:.1f} deg FOV radius'.format(
        len(catalog_stars), fov_radius))

    # Go through all radii and match the stars
    min_match_radius = None
    matched_stars_good = None
    for match_radius in radius_list:

        # Skip radiuses that are too small if the radius filter is on
        if max_match_radius is not None:
            if match_radius < max_match_radius:
                log.info(
                    "Stopping radius decrements because {:.2f} < {:.2f}".format(
                        match_radius, max_match_radius
                    )
                )
                break

        # If the platepar is good and the radius is below a pixel, don't recalibrate anymore
        if (match_radius < 1.0) and CheckFit.checkFitGoodness(
            config, working_platepar, catalog_stars, star_dict_ff, match_radius, verbose=True
        ):
            log.info('The fit is good enough!')
            break

        # log.info('Trying to match stars with radius {:.1f} px...'.format(match_radius))

        # If there are no matched stars, give up
        n_matched, _, _, _ = CheckFit.matchStarsResiduals(
            config,
            working_platepar,
            catalog_stars,
            star_dict_ff,
            match_radius,
            ret_nmatch=True,
            verbose=False,
            lim_mag=lim_mag,
        )

        if n_matched == 0:
            log.info('No stars matched, stopping the fit!')
            result = None
            break

        ### Recalibrate the platepar just on these stars, use the default platepar for initial params ###

        # Init initial parameters
        p0 = [
            working_platepar.RA_d,
            working_platepar.dec_d,
            working_platepar.pos_angle_ref,
            working_platepar.F_scale,
        ]

        # Compute the minimization tolerance
        fatol, xatol_ang = CheckFit.computeMinimizationTolerances(config, working_platepar, len(star_dict_ff))

        simplex = CheckFit.buildNelderMeadSimplex(p0, working_platepar.dec_d, mode="floor")

        res = scipy.optimize.minimize(
            CheckFit._calcImageResidualsAstro,
            p0,
            args=(config, working_platepar, catalog_stars, star_dict_ff, match_radius),
            method='Nelder-Mead',
            options={'fatol': fatol, 'xatol': xatol_ang, 'initial_simplex': simplex},
        )

        ###

        # Compute matched stars
        temp_platepar = copy.deepcopy(working_platepar)

        ra_ref, dec_ref, pos_angle_ref, F_scale_ref = res.x
        temp_platepar.RA_d = ra_ref
        temp_platepar.dec_d = dec_ref
        temp_platepar.pos_angle_ref = pos_angle_ref
        temp_platepar.F_scale = F_scale_ref

        n_matched, dist, cost, matched_stars = CheckFit.matchStarsResiduals(
            config,
            temp_platepar,
            catalog_stars,
            star_dict_ff,
            match_radius,
            ret_nmatch=True,
            verbose=False,
            lim_mag=lim_mag,
        )

        # If the fit was not successful, stop further fitting on this FF file
        if (
            (not res.success)
            or (n_matched < config.min_matched_stars)
            or (not ignore_distance_threshold and (dist > config.dist_check_threshold))
        ):

            if not res.success:
                log.info('Astrometry fit failed!')
            elif (dist > config.dist_check_threshold) and (not ignore_distance_threshold):
                log.info(
                    'Average star match does not satisfy the threshold: {:.2f} > {:.2f} px'.format(dist, config.dist_check_threshold)
                )

            else:
                log.info(
                    'Number of matched stars after the fit is smaller than necessary: {:d} < {:d}'.format(n_matched, config.min_matched_stars)
                )

            # Indicate that the recalibration failed
            result = None
            break

        else:
            # If the fit was successful, use the new parameters from now on
            working_platepar = temp_platepar

            # Keep track of the minimum match radius
            min_match_radius = match_radius

            # Keep the matched stars from this successful iteration - a later iteration that fails
            #   overwrites matched_stars with matches of a platepar that was never accepted, and
            #   those must not feed the coverage gate or the photometry fit below
            matched_stars_good = matched_stars

            log.info('{:g}/{:g} match with avg distance {:.2f} px within radius {:.2f} px!'.format(
                n_matched, len(star_dict_ff[jd]), dist, match_radius
            ))

            # Debug plot: show catalog stars, calstars, and matched stars
            if debug:
                _plotRecalibDebug(config, working_platepar, jd, star_dict_ff, catalog_stars,
                                  matched_stars, match_radius, lim_mag=lim_mag, ff_name=ff_name)

    # Choose which radius will be chosen for the goodness of fit check
    if max_match_radius is None:
        goodness_check_radius = match_radius

    else:
        goodness_check_radius = max_match_radius

    # The search stops within its tolerance, so the pointing is refined by least squares on the stars matched
    #   within the smallest radius, scaled to the resolution (the quick fit's list of radii is not scaled). The
    #   matched stars of the refined pointing then replace those of the search (same radius) for the coverage
    #   check and the photometry. A recalibration limited to larger radii and saved without the goodness check
    #   (the fallback of recalibratePlateparsForFF) is not refined: its loose pairs would pull the fit, while the
    #   search's median cost resists them
    if config.recalibration_lsq_refine and (matched_stars_good is not None) and (max_match_radius is None) \
            and (not force_platepar_save):

        refined, matched_refined = refinePointingLSQ(config, working_platepar, jd, star_dict_ff, catalog_stars,
                                                     scaling_factor*min_radius, min_match_radius,
                                                     len(matched_stars_good[jd][0]), lim_mag=lim_mag)

        # The least squares fit can lower the distances while raising their median, which the goodness check below
        #   uses, so the refined pointing is used only if it is good enough on its own. Otherwise the result of the
        #   search is kept
        if (refined is not None) and CheckFit.checkFitGoodness(config, refined, catalog_stars, star_dict_ff,
                                                               goodness_check_radius):
            working_platepar = refined
            matched_stars_good = matched_refined

    # If the platepar is good, store it
    if (
        CheckFit.checkFitGoodness(config, working_platepar, catalog_stars, star_dict_ff, goodness_check_radius)
        or force_platepar_save
    ):

        # If no fit iteration succeeded, there is no matched star set consistent with the platepar,
        #   so neither the coverage gate nor the photometry fit can run - reject the platepar (the
        #   caller will fall back to the previous one)
        if matched_stars_good is None:
            log.info('Rejecting refined platepar, no successful fit iteration to take matched '
                     'stars from')
            return None, min_match_radius

        # Get a list of matched image and catalog stars, taken from the last successful fit
        #   iteration so that they are consistent with the platepar being saved
        image_stars, matched_catalog_stars, _ = matched_stars_good[jd]

        # Coverage gate: reject a fit that only matched a small fraction of the detected stars.
        # The goodness check above only weighs the residuals of the stars that matched, so a
        # sparse spurious subset (e.g. 29 of 308 detections aligned at a wrong pointing) can pass
        # it. When such a fit is stamped auto_recalibrated and chained forward as the seed for the
        # next FF, the pointing walks away over a marginal/cloudy night. Requiring a real coverage
        # fraction rejects those fits (the frame keeps the previous good platepar) while accepting
        # genuine fits - including after a real camera move, which re-matches the whole field at
        # the new pointing.
        n_detected = len(star_dict_ff[jd])
        match_fraction = len(image_stars)/n_detected if n_detected > 0 else 0.0

        if match_fraction < min_match_fraction:
            log.info('Rejecting refined platepar, only {:d}/{:d} = {:.2f} of detected stars '
                     'matched within {:.2f} px (< {:.2f})'.format(len(image_stars), n_detected,
                                                                  match_fraction, min_match_radius,
                                                                  min_match_fraction))
            return None, min_match_radius

        ### PHOTOMETRY FIT ###

        star_intensities = image_stars[:, 2]
        # Ensure it's a numpy array with at least 3 columns for unpacking
        matched_catalog_stars = np.array(matched_catalog_stars)
        if matched_catalog_stars.ndim == 1 and matched_catalog_stars.size == 0:
            matched_catalog_stars = np.empty((0, 3))
        elif matched_catalog_stars.ndim == 2 and matched_catalog_stars.shape[1] < 3:
            log.error("matched_catalog_stars has wrong number of columns: {}".format(matched_catalog_stars.shape[1]))
            matched_catalog_stars = np.empty((0, 3))

        ra_catalog, dec_catalog, catalog_mags = matched_catalog_stars.T[:3]

        # Compute apparent star magnitudes by including extinction
        corrected_catalog_mags = extinctionCorrectionTrueToApparent(
            catalog_mags, ra_catalog, dec_catalog, jd, working_platepar
        )

        # Compute radius of every star from image centre
        radius_arr = np.hypot(
            image_stars[:, 0] - working_platepar.Y_res / 2, image_stars[:, 1] - working_platepar.X_res / 2
        )

        # Fit the photometry on automated star intensities (use the fixed vignetting coeff, use robust fit)
        photom_params, fit_stddev, _, _, _, _ = photometryFitRobust(
            star_intensities,
            radius_arr,
            corrected_catalog_mags,
            fixed_vignetting=working_platepar.vignetting_coeff,
        )
        photom_offset = photom_params[0]

        # Store the fitted photometric offset and error
        working_platepar.mag_lev = photom_offset
        working_platepar.mag_lev_stddev = fit_stddev

        # Print photometry info
        log.info("Photometry")
        log.info("    Fit: {:+.1f}*LSP + {:.2f} +/- {:.2f}".format(-2.5, photom_offset, fit_stddev))

        ### ###

        log.info(
            "Platepar minimum error of {:.2f} with radius {:.1f} px PASSED!".format(
                config.dist_check_threshold, goodness_check_radius
            )
        )

        log.info('Saving improved platepar...')

        ### plot for checking fit quality ###
        # if date2JD(*FFfile.getMiddleTimeFF("FF_HR000N_20201214_203429_089_0416000.fits", config.fps)) == jd:
        #     plt.title('F_HR000K_20201213_225055_929_0631040.fits')
        #     plt.scatter(*matched_stars[jd][0][:,:2].T[::-1], label='matched')
        #     plt.scatter(*np.array(star_dict_ff[jd])[:,:2].T[::-1],c='r',s=1, label='detected')
        #     plt.legend()
        #     plt.show()

        # Mark the platepar to indicate that it was automatically recalibrated on an individual FF file
        working_platepar.auto_recalibrated = True
        working_platepar.star_list = []
        for star_vals, ra, dec, mag in zip(image_stars, ra_catalog, dec_catalog, catalog_mags):
            working_platepar.star_list.append([jd] + list(star_vals[:3]) + [ra, dec, mag])

        # Store the platepar to the list of recalibrated platepars
        result = working_platepar

    # Otherwise, indicate that the refinement was not successful
    else:
        log.info('Rejecting refined platepar, min error {:.2f} px NOT ACHIEVED!'.format(
            config.dist_check_threshold
        ))
        result = None

    return result, min_match_radius


def _plotRecalibDebug(config, platepar, jd, star_dict_ff, catalog_stars, matched_stars, match_radius,
                     lim_mag=None, ff_name=None):
    """ Show a debug plot with catalog stars, calstars, and matched stars.

    Arguments:
        config: [Config instance]
        platepar: [Platepar instance] Current (fitted) platepar.
        jd: [float] Julian date.
        star_dict_ff: [dict] {jd: list of star entries} for the current FF.
        catalog_stars: [ndarray] Full catalog stars array (ra, dec, mag).
        matched_stars: [dict] Output from matchStarsResiduals, keyed by jd.
        match_radius: [float] Current match radius in pixels.

    Keyword arguments:
        lim_mag: [float] Limiting magnitude. None uses config default.
        ff_name: [str] Name of the FF file being processed. None by default.
    """
    from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, getFOVSelectionRadius, xyToRaDecPP
    from RMS.Astrometry.Conversions import jd2Date
    from RMS.Astrometry.CyFunctions import subsetCatalog

    if lim_mag is None:
        lim_mag = config.catalog_mag_limit

    # Convert JD to a human-readable timestamp
    dt_tuple = jd2Date(jd)
    timestamp_str = '{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:05.2f} UTC'.format(
        int(dt_tuple[0]), int(dt_tuple[1]), int(dt_tuple[2]),
        int(dt_tuple[3]), int(dt_tuple[4]), dt_tuple[5]
    )

    # Get the FOV radius for catalog subset extraction
    fov_radius = getFOVSelectionRadius(platepar)

    # Get RA/Dec of the FOV centre
    _, RA_c, dec_c, _ = xyToRaDecPP([jd2Date(jd)], [platepar.X_res/2], [platepar.Y_res/2], [1],
                                     platepar, extinction_correction=False, precompute_pointing_corr=True)
    RA_c = RA_c[0]
    dec_c = dec_c[0]

    # Get catalog stars in the FOV
    _, extracted_catalog = subsetCatalog(catalog_stars, RA_c, dec_c, jd, platepar.lat, platepar.lon,
                                         fov_radius, lim_mag)
    
    # Ensure it's a numpy array with at least 3 columns for unpacking
    extracted_catalog = np.array(extracted_catalog)
    if extracted_catalog.ndim == 1 and extracted_catalog.size == 0:
        extracted_catalog = np.empty((0, 3))
    elif extracted_catalog.ndim == 2 and extracted_catalog.shape[1] < 3:
        log.error("extracted_catalog has wrong number of columns: {}".format(extracted_catalog.shape[1]))
        extracted_catalog = np.empty((0, 3))

    ra_cat, dec_cat, mag_cat = extracted_catalog.T[:3]

    # Project catalog stars to image coordinates
    cat_x, cat_y = raDecToXYPP(ra_cat, dec_cat, jd, platepar)

    # Filter to those within the image
    in_fov = (cat_x >= 0) & (cat_x < platepar.X_res) & (cat_y >= 0) & (cat_y < platepar.Y_res)
    cat_x_fov = cat_x[in_fov]
    cat_y_fov = cat_y[in_fov]
    mag_cat_fov = mag_cat[in_fov]

    # Extract calstars (detected stars on this image)
    stars_list = np.array(star_dict_ff[jd])
    # stars_list columns: (Y, X, bg_level, level, ...)
    calstars_y = stars_list[:, 0]
    calstars_x = stars_list[:, 1]

    # Extract matched stars
    if jd in matched_stars:
        matched_val = matched_stars[jd]
        if not isinstance(matched_val, (list, tuple)) or len(matched_val) != 3:
             log.error("matched_stars[jd] (debug plot) is not a 3-tuple! Value: {}, type: {}".format(matched_val, type(matched_val)))
        matched_img_stars, matched_cat_stars, dist_list = matched_val
        matched_img_y = matched_img_stars[:, 0]
        matched_img_x = matched_img_stars[:, 1]
        # Project matched catalog stars to image coordinates
        matched_cat_x, matched_cat_y = raDecToXYPP(matched_cat_stars[:, 0], matched_cat_stars[:, 1],
                                                    jd, platepar)
    else:
        matched_img_x = matched_img_y = np.array([])
        matched_cat_x = matched_cat_y = np.array([])

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor('black')

    # Plot catalog stars (blue crosses, size scaled by magnitude)
    mag_sizes = np.clip(20*(lim_mag - mag_cat_fov + 1), 5, 200)
    ax.scatter(cat_x_fov, cat_y_fov, s=mag_sizes, marker='+', c='dodgerblue', linewidths=0.8,
              alpha=0.6, label='Catalog stars ({:g})'.format(len(cat_x_fov)))

    # Plot calstars (green circles)
    ax.scatter(calstars_x, calstars_y, s=30, facecolors='none', edgecolors='lime', linewidths=1.0,
              label='Detected stars ({:g})'.format(len(calstars_x)))

    # Plot matched stars: draw lines connecting matched pairs
    n_matched = len(matched_img_x)
    for mx, my, cx, cy in zip(matched_img_x, matched_img_y, matched_cat_x, matched_cat_y):
        ax.plot([mx, cx], [my, cy], 'r-', linewidth=0.5, alpha=0.7)

    # Plot matched image stars (red filled dots)
    ax.scatter(matched_img_x, matched_img_y, s=15, c='red', zorder=5,
              label='Matched ({:g})'.format(n_matched))

    ax.set_xlim(0, platepar.X_res)
    ax.set_ylim(platepar.Y_res, 0)
    ax.set_aspect('equal')
    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')

    # Title with timestamp used for star position computation
    ax.set_title('Recalibration debug | {} | Radius={:.1f}px | Matched={:g}/{:g}'.format(
        timestamp_str, match_radius, n_matched, len(calstars_x)))

    # Add FF name below the image as figure text
    if ff_name is not None:
        fig.text(0.5, 0.01, ff_name, ha='center', va='bottom', fontsize=10, color='gray',
                 fontstyle='italic')

    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.07)  # Make room for the FF name text
    plt.show()


def recalibratePlateparsForFF(
    prev_platepar,
    ff_file_names,
    calstars,
    catalog_stars,
    config,
    lim_mag=None,
    ignore_distance_threshold=False,
    ignore_max_stars=False,
    ff_frames=256,
    debug=False,
):
    """
    Recalibrate platepars corresponding to ff files based on the stars.

    Arguments:
        prev_platepar: [platepar]
        ff_file_names: [list] list of ff file names
        calstars: [dict] A dictionary with only one entry, where the key is the FF file name and the value is 
            the list of star coordinates.
        catalog_stars: [list] A list of catalog stars which should be in the image.
        config: [config]

    Keyword arguments:
        lim_mag: [float] Limiting magnitude for the catalog. If None, use the default value from the config.
        ignore_distance_threshold: [bool] Don't consider the recalib as failed if the median distance
            is larger than the threshold.
        ignore_max_stars: [bool] Ignore the maximum number of image stars for recalibration.
        ff_frames: [int] Number of frames in the FF file or frame chunk. Default is 256.
        debug: [bool] If True, show debug plots for each fit iteration.

    Returns:
        recalibrated_platepars: [dict] A dictionary where one key is ff file name and the value is
            a calibrated corresponding platepar.
    """

    orig_platepar = copy.deepcopy(prev_platepar)

    # Go through all FF files with detections, recalibrate and apply astrometry
    recalibrated_platepars = {}
    for ff_name in ff_file_names:

        working_platepar = copy.deepcopy(prev_platepar)

        # Skip this meteor if its FF file was already recalibrated
        if ff_name in recalibrated_platepars:
            continue
        
        log.info('------------------------------------------------------------------------------')
        log.info('Processing: {}'.format(ff_name))

        # Find extracted stars on this image
        if not ff_name in calstars:
            log.info('Skipped because it was not in CALSTARS: {}'.format(ff_name))
            continue

        # Get stars detected on this FF file (create a dictionary with only one entry, the residuals function
        #   needs this format)
        calstars_time = FFfile.getMiddleTimeFF(ff_name, config.fps, ret_milliseconds=True, 
                                               ff_frames=ff_frames)
        jd = date2JD(*calstars_time)
        star_dict_ff = {jd: calstars[ff_name]}

        result = None

        # Skip recalibration if less than a minimum number of stars were detected
        if (len(calstars[ff_name]) >= config.ff_min_stars) and (
            len(calstars[ff_name]) >= config.min_matched_stars
        ):

            # Recalibrate the platepar using star matching
            result, min_match_radius = recalibrateFF(
                config,
                working_platepar,
                jd,
                star_dict_ff,
                catalog_stars,
                lim_mag=lim_mag,
                ignore_distance_threshold=ignore_distance_threshold,
                ignore_max_stars=ignore_max_stars,
                debug=debug,
                ff_name=ff_name,
            )

            # If the recalibration failed, try using NN alignment
            if result is None:

                log.info('Running NN alignment...')

                # Run NN alignment - pass full CALSTARS data so alignPlatepar can infer catalog LM
                calstars_data = np.array(star_dict_ff[jd])
                test_platepar, _ = alignPlatepar(
                    config, prev_platepar, calstars_time, calstars_data, show_plot=False
                )

                # Try to recalibrate after NN alignment
                result, _ = recalibrateFF(
                    config, test_platepar, jd, star_dict_ff, catalog_stars, lim_mag=lim_mag,
                    ignore_distance_threshold=True, debug=debug, ff_name=ff_name
                )

                # If the NN alignment failed, align the previous platepar using the smallest radius that
                # matched and force save the the platepar
                if (result is None) and (min_match_radius is not None):
                    log.info(
                        "Using the old platepar with the minimum match radius of: {:.2f}".format(
                            min_match_radius
                        )
                    )
                    result, _ = recalibrateFF(
                        config,
                        working_platepar,
                        jd,
                        star_dict_ff,
                        catalog_stars,
                        max_match_radius=min_match_radius,
                        force_platepar_save=True,
                        lim_mag=lim_mag,
                        ignore_distance_threshold=True,
                        debug=debug,
                        ff_name=ff_name,
                    )

                    if result is not None:
                        working_platepar = result

                # If the alignment succeeded, save the result
                else:
                    working_platepar = result

            else:
                working_platepar = result


            # If the working platepar keeps failing, try using the original platepar before any NN alignment
            if result is None:
                log.info('Recalibration failed, trying to use the original platepar...')
                result, _ = recalibrateFF(
                    config,
                    orig_platepar,
                    jd,
                    star_dict_ff,
                    catalog_stars,
                    lim_mag=lim_mag,
                    ignore_distance_threshold=True,
                    debug=debug,
                    ff_name=ff_name,
                )

                # If the recalibration succeeded, use the result
                if result is not None:
                    working_platepar = result

        # Store the platepar if the fit succeeded
        if result is not None:

            # Recompute alt/az of the FOV centre (apparent, epoch-of-date) using the platepar's own
            #   method, so the az/alt fields match the convention used by SkyFit/Platepar/ECSV. A plain
            #   raDec2AltAz here lacks precession and leaves az_centre ~0.3 deg off on aged platepars.
            working_platepar.updateRefAltAz()

            # Recompute the rotation wrt horizon
            working_platepar.rotation_from_horiz = rotationWrtHorizon(working_platepar)

            # Mark the platepar to indicate that it was automatically recalibrated on an individual FF file
            working_platepar.auto_recalibrated = True

            recalibrated_platepars[ff_name] = working_platepar
            prev_platepar = working_platepar

        else:

            log.info('Recalibration of {:s} failed, using the previous platepar...'.format(ff_name))

            # Mark the platepar to indicate that autorecalib failed
            prev_platepar_tmp = copy.deepcopy(prev_platepar)
            prev_platepar_tmp.auto_recalibrated = False

            # If the aligning failed, set the previous platepar as the one that should be used for this FF file
            recalibrated_platepars[ff_name] = prev_platepar_tmp

    return recalibrated_platepars


def recalibrateSelectedFF(dir_path, ff_file_names, calstars_data, config, lim_mag, \
    pp_recalib_name, ignore_distance_threshold=False, ignore_max_stars=False, platepar=None):
    """Recalibrate FF files, ignoring whether there are detections.

    Arguments:
        dir_path: [str] Path where the FF files are.
        ff_file_names: [str] List of ff files to recalibrate platepars to
        calstars_data: [tuple] (list, int)
            - A list of entries [[ff_name, star_coordinates], ...].
            - The number of frames in the FF file or frame chunk.
        config: [Config instance]
        lim_mag: [float] Limiting magnitude for the catalog.
        pp_recalib_name: [str] Name for the file where the recalibrated platepars will be stored as JSON.

    Keyword arguments:
        ignore_distance_threshold: [bool] Don't consider the recalib as failed if the median distance
            is larger than the threshold.
        ignore_max_stars: [bool] Ignore the maximum number of image stars for recalibration.
        platepar: [Platepar instance] Optional platepar to use as a starting point. None by default.

    Return:
        recalibrated_platepars: [dict] A dictionary where the keys are FF file names and values are
            recalibrated platepar instances for every FF file.
    """
    config = copy.deepcopy(config)

    calstars_list, ff_frames = calstars_data

    calstars = {ff_file: star_data for ff_file, star_data in calstars_list}

    if not ff_file_names:
        log.warning("recalibrateSelectedFF: no FF files after filtering - skipping recalibration")
        return {}
    
    ts = FFfile.getMiddleTimeFF(ff_file_names[0], fps=config.fps, ret_milliseconds=True, dt_obj=True)

    J2000 = datetime.datetime(2000, 1, 1, 12, 0, 0)

    # Compute the number of years from J2000
    years_from_J2000 = (ts - J2000).total_seconds()/(365.25*24*3600)
    # log.info('Loading star catalog with years from J2000: {:.2f}'.format(years_from_J2000))

    # load star catalog with increased catalog limiting magnitude
    star_catalog_status = StarCatalog.readStarCatalog(
        config.star_catalog_path,
        config.star_catalog_file,
        years_from_J2000=years_from_J2000,
        lim_mag=lim_mag,
        mag_band_ratios=config.star_catalog_band_ratios,
    )

    if not star_catalog_status:
        log.info("Could not load the star catalog!")
        log.info(os.path.join(config.star_catalog_path, config.star_catalog_file))
        return {}

    if not isinstance(star_catalog_status, (tuple, list)):
        log.error("star_catalog_status is not a tuple/list! Value: {}, type: {}".format(star_catalog_status, type(star_catalog_status)))
    elif len(star_catalog_status) != 3:
        log.error("star_catalog_status has wrong length! Length: {}, Value: {}".format(len(star_catalog_status), star_catalog_status))

    catalog_stars, _, config.star_catalog_band_ratios = star_catalog_status
    # log.info(catalog_stars)

    # If the platepar was not given, try to find it
    if platepar is None:
        prev_platepar = Platepar.findBestPlatepar(config, dir_path)

    else:
        prev_platepar = copy.deepcopy(platepar)

    if prev_platepar is None:
        log.error("recalibrateSelectedFF: Could not find a platepar file! Recalibration aborted.")
        return {}

    # Update the platepar coordinates from the config file
    prev_platepar.lat = config.latitude
    prev_platepar.lon = config.longitude
    prev_platepar.elev = config.elevation

    recalibrated_platepars = recalibratePlateparsForFF(
        prev_platepar,
        ff_file_names,
        calstars,
        catalog_stars,
        config,
        lim_mag=lim_mag,
        ignore_distance_threshold=ignore_distance_threshold,
        ignore_max_stars=ignore_max_stars,
        ff_frames=ff_frames
    )

    # Store recalibrated platepars in json
    all_pps = {}
    for ff_name in recalibrated_platepars:
        json_str = recalibrated_platepars[ff_name].jsonStr()
        all_pps[ff_name] = json.loads(json_str)

    with open(os.path.join(dir_path, pp_recalib_name), 'w') as f:
        
        # Convert all platepars to a JSON file
        out_str = json.dumps(all_pps, default=lambda o: o.__dict__, indent=4, sort_keys=True)
        f.write(out_str)

    return recalibrated_platepars


def writeMeteorECSV(dir_path, station_id, ff_name, platepar, meteor_picks, fps, used_names, log):
    """ Write a single recalibrated meteor detection as a GDEF ECSV file.

    The per-pick UTC time is reconstructed from the FF reference time and the (fractional) frame number
    using the given fps: time = filenameToDatetime(ff_name) + frame/fps, at full microsecond precision.
    For inputs with accurate per-frame timestamps this recovers the true GPS time (to within the
    FTPdetectinfo frame-field quantum, ~2 us at 32 fps).

    Arguments:
        dir_path: [str] Directory where the ECSV file will be written.
        station_id: [str] Station/camera code.
        ff_name: [str] FF file name (encodes the reference time and is stored as image_file in the meta).
        platepar: [Platepar instance] Platepar applied to this detection (for the ECSV meta header).
        meteor_picks: [ndarray] 12-column calibrated picks from applyPlateparToCentroids:
            frame, x, y, ra, dec, az, alt, level, mag, background, snr, saturated.
        fps: [float] Frames per second used to reconstruct the pick times.
        used_names: [set] Set of ECSV file names already written in this run (mutated for uniqueness).
        log: [Logger] Logger handle.

    Return:
        [str or None] Path to the written file, or None if there were no valid picks.
    """

    # Reference time encoded in the FF file name
    dt_ref = FFfile.filenameToDatetime(ff_name)

    # Build the list of picks (skip those without a valid centroid)
    picks = []
    for row in meteor_picks:

        frame, x, y, ra, dec, azim, alt, level, mag, background, snr, saturated = row

        if np.isnan(x) or np.isnan(y):
            continue

        picks.append({
            'datetime': dt_ref + datetime.timedelta(seconds=frame/fps),
            'ra': ra, 'dec': dec, 'azim': azim, 'alt': alt,
            'x': x, 'y': y,
            'intensity': level, 'background': background,
            'saturated': saturated > 0,
            'mag': mag, 'snr': snr,
        })

    # Nothing to write
    if not picks:
        return None

    # Number of stars used in the astrometric calibration
    if getattr(platepar, 'star_list', None) is not None:
        n_stars = len(platepar.star_list)
    else:
        n_stars = 0

    # Construct a unique file name from the time of the first pick
    base_name = picks[0]['datetime'].strftime("%Y-%m-%dT%H_%M_%S") + '_RMS_' + station_id
    file_name = base_name + '.ecsv'
    collision = 1
    while file_name in used_names:
        collision += 1
        file_name = '{:s}_{:d}.ecsv'.format(base_name, collision)
    used_names.add(file_name)

    ecsv_path = writeECSV(dir_path, file_name, station_id, platepar, picks,
        mag_band_string='V', n_stars=n_stars, image_file=ff_name,
        isodate_start_obs=dt_ref.strftime("%Y-%m-%dT%H:%M:%S.%f"))

    log.info('ECSV file saved: {:s}'.format(file_name))

    return ecsv_path


def plotCalibrationVariation(recalibrated_platepars, platepar, config, dir_path, plot_name, ff_frames=256):
    """ Plot the variation of the pointing (FOV centre drift and rotation) and of the photometric offset of
        the recalibrated platepars with respect to the reference platepar.

    Arguments:
        recalibrated_platepars: [dict] Recalibrated platepars (Platepar instances or None), keyed by the FF
            name.
        platepar: [Platepar instance] Reference platepar.
        config: [Config instance]
        dir_path: [str] Directory where the plots are saved.
        plot_name: [str] Prefix of the plot names, the plots are saved as
            <plot_name>_calibration_variation.png and <plot_name>_photometry_variation.png.

    Keyword arguments:
        ff_frames: [int] Number of frames in the FF files or frame chunks. 256 by default.

    Return:
        [bool] True if the plots were saved, False if there were less than 2 recalibrated platepars.
    """

    dt_list = []
    ang_dists = []
    rot_angles = []
    hour_list = []
    photom_offset_list = []
    photom_offset_std_list = []

    # If the length of the recalibrated platepars is less than 2, skip the plot generation
    if len(recalibrated_platepars) < 2:

        log.info('Less than 2 FF files were recalibrated, skipping the plot generation...')

        return False
    

    first_dt = np.min([FFfile.filenameToDatetime(ff_name) for ff_name in recalibrated_platepars])

    for ff_name in recalibrated_platepars:

        pp_temp = recalibrated_platepars[ff_name]

        # If the fitting failed, skip the platepar
        if pp_temp is None:
            continue

        # Add the datetime of the FF file to the list
        ff_dt = FFfile.filenameToDatetime(ff_name)
        dt_list.append(ff_dt)

        # Compute the angular separation from the reference platepar by projecting
        # both FOV centres at the FF file's observation time. This correctly handles
        # platepars with different JD reference epochs.
        ff_time = FFfile.getMiddleTimeFF(ff_name, config.fps, ret_milliseconds=True, ff_frames=ff_frames)
        _, ref_ra, ref_dec, _ = xyToRaDecPP(
            [ff_time], [platepar.X_res / 2], [platepar.Y_res / 2], [1], platepar,
            extinction_correction=False)
        _, temp_ra, temp_dec, _ = xyToRaDecPP(
            [ff_time], [pp_temp.X_res / 2], [pp_temp.Y_res / 2], [1], pp_temp,
            extinction_correction=False)
        ang_dist = np.degrees(
            angularSeparation(
                np.radians(ref_ra[0]),
                np.radians(ref_dec[0]),
                np.radians(temp_ra[0]),
                np.radians(temp_dec[0]),
            )
        )
        ang_dists.append(ang_dist*60)

        # Compute rotation difference
        rot_diff = (platepar.rotationWrtHorizon() - pp_temp.rotationWrtHorizon() + 180)%360 - 180
        rot_angles.append(rot_diff*60)

        # Compute the hour of the FF used for recalibration
        hour_list.append((ff_dt - first_dt).total_seconds()/3600)

        # Add the photometric offset to the list
        photom_offset_list.append(pp_temp.mag_lev)
        photom_offset_std_list.append(pp_temp.mag_lev_stddev)


    ### Plot difference from reference platepar in angular distance from (0, 0) vs rotation ###

    plt.figure(figsize=(6, 5))

    plt.scatter(0, 0, marker='o', edgecolor='k', label='Reference platepar', s=100, c='none', zorder=3)

    plt.scatter(ang_dists, rot_angles, c=hour_list, zorder=3)
    plt.colorbar(label="Hours from first FF file")

    plt.xlabel("Angular distance from reference (arcmin)")
    plt.ylabel("Rotation from reference (arcmin)")

    plt.title("FOV centre drift starting at {:s}".format(first_dt.strftime("%Y/%m/%d %H:%M:%S")))

    plt.grid()
    plt.legend()

    # Scale the aspect ratio so X and Y units are the same but the plot is not too narrow
    plt.axis('scaled')

    # Make the plot square by adjusting the limits to the maximum
    min_lim = min(plt.xlim()[0], plt.ylim()[0])
    max_lim = max(plt.xlim()[1], plt.ylim()[1])
    abs_lim = max_lim - min_lim
    plt.xlim(-0.1*abs_lim, 0.9*abs_lim)
    plt.ylim(min_lim, max_lim)


    plt.tight_layout()

    plt.savefig(os.path.join(dir_path, plot_name + '_calibration_variation.png'), dpi=150)

    # plt.show()

    plt.clf()
    plt.close()

    ### ###

    ### Plot the photometric offset variation ###

    # Skip the plot if there are not enough points (platepars which failed the fit are skipped above, so
    #   fewer points can remain than there are recalibrated platepars)
    if len(dt_list) < 2:

        log.info('Less than 2 photometric offsets available, skipping the photometry plot...')

        return True

    # Sanitize the error bars. Non-finite values break the error bar plotting in older versions of matplotlib
    #   (they raise a StopIteration), so replace them with zeros and omit the error bars completely if none of
    #   the values are usable
    photom_offset_std_arr = np.array(photom_offset_std_list, dtype=np.float64)
    if np.any(np.isfinite(photom_offset_std_arr)):
        yerr = np.where(np.isfinite(photom_offset_std_arr), photom_offset_std_arr, 0.0)

    else:
        yerr = None

    plt.figure()

    plt.errorbar(
        dt_list,
        photom_offset_list,
        yerr=yerr,
        fmt="o",
        ecolor='lightgray',
        elinewidth=2,
        capsize=0,
        ms=2,
    )

    # Format datetimes
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    # rotate and align the tick labels so they look better
    plt.gcf().autofmt_xdate()

    plt.xlabel("UTC time")
    plt.ylabel("Photometric offset")

    plt.title("Photometric offset variation")

    plt.grid()

    plt.tight_layout()

    plt.savefig(os.path.join(dir_path, plot_name + '_photometry_variation.png'), dpi=150)

    plt.clf()
    plt.close()

    return True


def ftpMatchTimes(meteor_list, meteor_fps_list, calstars_datetime_dict):
    """ Compute the times of the FF files in FTPdetectinfo which are used to find the closest CALSTARS entry
        (compared to the middle times of the CALSTARS entries).

    Arguments:
        meteor_list: [list] Meteors in the writeFTPdetectinfo input format: [ff_name, meteor_No, rho, phi,
            meteor_meas], where every measurement begins with the frame number.
        meteor_fps_list: [list] Frame rate of every meteor.
        calstars_datetime_dict: [dict] Middle times of the CALSTARS entries, keyed by the FF name.

    Return:
        [OrderedDict] {ff_name: datetime}
    """

    ftp_ff_datetime_dict = OrderedDict()

    for meteor_entry, meteor_fps in zip(meteor_list, meteor_fps_list):

        ff_name = meteor_entry[0]

        if ff_name in ftp_ff_datetime_dict:
            continue

        # If the FF file is also in CALSTARS, use the same time so it's matched to itself
        if ff_name in calstars_datetime_dict:
            ftp_ff_datetime_dict[ff_name] = calstars_datetime_dict[ff_name]

        # Otherwise (e.g. the name is the time of the first pick for non-FF inputs), use the time of the first
        #   pick, which is closest to the middle of the CALSTARS chunk that contains it
        else:
            first_frame = meteor_entry[4][0][0]
            ftp_ff_datetime_dict[ff_name] = FFfile.filenameToDatetime(ff_name) \
                + datetime.timedelta(seconds=first_frame/meteor_fps)

    return ftp_ff_datetime_dict


def recalibrateIndividualFFsAndApplyAstrometry(
    dir_path, ftpdetectinfo_path, calstars_data, config, platepar,
    generate_plot=True, load_all=False, debug=False, ecsv_out=False, calstars_fps=None
):
    """Recalibrate FF files with detections and apply the recalibrated platepar to those detections.
    
    Arguments:
        dir_path: [str] Path where the FTPdetectinfo file is.
        ftpdetectinfo_path: [str] Name of the FTPdetectinfo file.
        calstars_data: [tuple] (list, int)
            - A list of entries [[ff_name, star_coordinates], ...].
            - The number of frames in the FF file or frame chunk.
        config: [Config instance]
        platepar: [Platepar instance] Initial platepar.
        
    Keyword arguments:
        generate_plot: [bool] Generate the calibration variation plot. True by default.
        load_all: [bool] Load all FTPdetectinfo files in the directory and recalibrate them. False by default.
        debug: [bool] If True, show debug plots for each fit iteration.
        ecsv_out: [None/bool/str] If set, dump each calibrated detection as an ECSV file. True writes to the
            FTPdetectinfo directory; a string is used as the output directory. None disables. None by default.
        calstars_fps: [float or None] FPS read from the CALSTARS header. When given, it is used (instead of
            config.fps) to reconstruct the FF/chunk middle times, so the timing follows the real measured
            frame rate rather than a possibly-wrong config value. None by default (config.fps is used).

    Return:
        (recalibrated_platepars, ftpdetectinfo_file_list):
            - recalibrated_platepars: [dict] A dictionary where the keys are FF file names and values are
                recalibrated platepar instances for every FF file.
            - ftpdetectinfo_file_list: [list] List of FTPdetectinfo files that were loaded.
    """

    # Use a copy of the config file
    config = copy.deepcopy(config)

    # Use a copy of the platepar
    platepar = copy.deepcopy(platepar)

    # If the CALSTARS header recorded the measured fps, use it for all timing reconstruction instead of
    #   the config fps. getMiddleTimeFF() below derives the FF/chunk middle times from config.fps, and a
    #   wrong config fps shifts those times (and hence the recalibration star matching) off the truth.
    if calstars_fps is not None:
        if abs(config.fps - calstars_fps) > 1e-6:
            log.info('Using CALSTARS-recorded fps {:.6f} for timing (config fps was {:.6f})'.format(
                calstars_fps, config.fps))
        config.fps = calstars_fps

    ### Load CALSTARS data ###

    # Load the list of stars from the CALSTARS file
    calstars_list, calstars_ff_frames = calstars_data

    # Convert the list of stars to a per FF name dictionary
    calstars = {ff_file: star_data for ff_file, star_data in calstars_list}

    # Make a list of sorted FF files in CALSTARS
    calstars_ffs = sorted(calstars.keys())

    if not calstars_ffs:
        log.warning("No FF entries in CALSTARS - skipping recalibration")
        return {}, []

    # Create a dictionary mapping FF file names in CALSTARS to datetime objects
    calstars_datetime_dict = OrderedDict()
    for ff_name in calstars:
        calstars_datetime_dict[ff_name] = FFfile.getMiddleTimeFF(ff_name, config.fps, dt_obj=True, 
                                                                 ff_frames=calstars_ff_frames)

    ### Load catalog stars ##

    # Increase catalog limiting magnitude by one to get more stars for matching
    catalog_mag_limit = config.catalog_mag_limit + 1

    ts = calstars_datetime_dict[calstars_ffs[0]]
    J2000 = datetime.datetime(2000, 1, 1, 12, 0, 0)

    # Compute the number of years from J2000
    years_from_J2000 = (ts - J2000).total_seconds()/(365.25*24*3600)

    # Load catalog stars (overwrite the mag band ratios if specific catalog is used)
    star_catalog_status = StarCatalog.readStarCatalog(
        config.star_catalog_path,
        config.star_catalog_file,
        years_from_J2000=years_from_J2000,
        lim_mag=catalog_mag_limit,
        mag_band_ratios=config.star_catalog_band_ratios
    )

    if not star_catalog_status:
        log.info("Could not load the star catalog!")
        log.info(os.path.join(config.star_catalog_path, config.star_catalog_file))
        return {}, []

    if not isinstance(star_catalog_status, (tuple, list)):
        log.error("star_catalog_status is not a tuple/list! Value: {}, type: {}".format(star_catalog_status, type(star_catalog_status)))
    elif len(star_catalog_status) != 3:
        log.error("star_catalog_status has wrong length! Length: {}, Value: {}".format(len(star_catalog_status), star_catalog_status))

    catalog_stars, _, config.star_catalog_band_ratios = star_catalog_status

    ### ###
    
    # Update the platepar coordinates from the config file
    platepar.lat = config.latitude
    platepar.lon = config.longitude
    platepar.elev = config.elevation

    

    ### ###


    # Find FTPdetectinfo files to load
    if load_all:

        ftpdetectinfo_file_list = []

        # Load all FTPdetectinfo files in the directory
        for ftpdetectinfo_file in sorted(os.listdir(dir_path)):

            # Check that the file is a valid FTPdetectinfo file
            if validDefaultFTPdetectinfo(ftpdetectinfo_file):

                ftpdetectinfo_file_list.append(ftpdetectinfo_file)

    else:

        # If the given file does not exits, return nothing
        if not os.path.isfile(ftpdetectinfo_path):
            log.info('ERROR! The FTPdetectinfo file does not exist: {:s}'.format(ftpdetectinfo_path))
            log.info('    The recalibration on every file was not done!')

            return {}, []

        # If it exists, use it as the only file to load
        ftpdetectinfo_file_list = [os.path.basename(ftpdetectinfo_path)]


    recalibrated_platepars_all = {}

    # Go through every FTPdetectinfo file and recalibrate the platepars
    for ftpdetectinfo_file in ftpdetectinfo_file_list:

        log.info('Recalibrating FTPdetectinfo file: {:s}'.format(ftpdetectinfo_file))

        # Load the FTPdetectinfo data
        # NOTE: The assumption is that all files have the same camera code and FPS, as they should
        _, _, meteor_list = FTPdetectinfo.readFTPdetectinfo(
            dir_path, ftpdetectinfo_file, ret_input_format=True
        )

        # Also read the full format to recover the per-meteor fps from the header (index 4). For inputs
        #   with accurate per-frame timestamps this is the measured fps (e.g. 32.13), which must be used
        #   instead of config.fps to keep the timing accurate. Both reads parse the file in the same order.
        # The full format returns the meteor list directly; each entry is
        #   [ff_name, cam_code, meteor_No, n_segments, fps, hnr, mle, binn, px_fm, rho, phi, meteor_meas]
        meteor_list_full = FTPdetectinfo.readFTPdetectinfo(
            dir_path, ftpdetectinfo_file, ret_input_format=False
        )
        meteor_fps_list = [entry[4] for entry in meteor_list_full]

        # If the list is empty, skip the file
        if not meteor_list:
            log.info('No meteor detections in the FTPdetectinfo file!')
            continue

    
        # Create a dictionary mapping FF file names in FTPdetectinfo to the times which are compared to the
        #   middle times of the CALSTARS entries
        ftp_ff_datetime_dict = ftpMatchTimes(meteor_list, meteor_fps_list, calstars_datetime_dict)


        # Go through every FF file entry listed in the FTPdetectinfo and identify three FF entries in the 
        # CALSTARS file - one at the closest time to the FTPdetectinfo FF file, one before, and one after
        # The three files are add for better photometric offset estimation
        ff_processing_list = []

        for meteor_entry in meteor_list:

            ff_name = meteor_entry[0]

            # Define a function to compute time difference between FF files in FTPdetectinfo and CALSTARS
            time_diff_func = lambda x: abs(
                (ftp_ff_datetime_dict[ff_name] - calstars_datetime_dict[x]).total_seconds()
                )

            # Find the closest FF file in CALSTARS
            closest_ff_name = min(calstars_datetime_dict, key=lambda x: time_diff_func(x))

            # Find the index of the given FF file in the list of calstars
            ff_indx = list(calstars_datetime_dict.keys()).index(closest_ff_name)

            # Add the closest FF file to the processing list
            ff_processing_list.append(closest_ff_name)

            # Add the FF file before to the processing list
            if ff_indx > 0:
                ff_processing_list.append(list(calstars_datetime_dict.keys())[ff_indx - 1])

            # Add the FF file after to the processing list
            if ff_indx < len(calstars_datetime_dict) - 1:
                ff_processing_list.append(list(calstars_datetime_dict.keys())[ff_indx + 1])

        # Sort the processing list of FF files
        ff_processing_list = sorted(ff_processing_list)

        # ### ###

        prev_platepar = copy.deepcopy(platepar)

        # Go through all FF files with detections, recalibrate and apply astrometry
        recalibrated_platepars = recalibratePlateparsForFF(
            prev_platepar, ff_processing_list, calstars, catalog_stars, config, ff_frames=calstars_ff_frames,
            debug=debug
        )
        

        ### Average out photometric offsets within the given neighbourhood size ###

        # Go through the list of FF files with detections
        for meteor_entry in meteor_list:

            ff_name = meteor_entry[0]

            # Make sure the FF was successfully recalibrated
            if ff_name in recalibrated_platepars:

                # Find the index of the given FF file in the list of calstars
                ff_indx = calstars_ffs.index(ff_name)

                # Compute the average photometric offset and the improved standard deviation using all
                #   neighbors
                photom_offset_tmp_list = []
                photom_offset_std_tmp_list = []
                neighboring_ffs = []
                for k in range(-(RECALIBRATE_NEIGHBOURHOOD_SIZE // 2), RECALIBRATE_NEIGHBOURHOOD_SIZE // 2 + 1):

                    k_indx = ff_indx + k

                    if (k_indx > 0) and (k_indx < len(calstars_ffs)):

                        # Get the name of the FF file
                        ff_name_tmp = calstars_ffs[k_indx]

                        # Check that the neighboring FF was successfully recalibrated
                        if ff_name_tmp in recalibrated_platepars:

                            # Get the computed photometric offset and stddev
                            mag_lev_tmp = recalibrated_platepars[ff_name_tmp].mag_lev
                            mag_lev_stddev_tmp = recalibrated_platepars[ff_name_tmp].mag_lev_stddev

                            # Only use neighbours with a usable photometric solution. Averaging in a
                            #   non-finite value would make the average non-finite and, as the average is
                            #   written back to all neighbours, the non-finite value would spread to all
                            #   FF files of the night through overlapping neighbourhoods
                            if not (np.isfinite(mag_lev_tmp) and np.isfinite(mag_lev_stddev_tmp)):
                                continue

                            photom_offset_tmp_list.append(mag_lev_tmp)
                            photom_offset_std_tmp_list.append(mag_lev_stddev_tmp)
                            neighboring_ffs.append(ff_name_tmp)

                # If no neighbour had a usable photometric solution, keep the individual values as they are
                if len(photom_offset_tmp_list) == 0:
                    log.warning('No finite photometric offsets among the neighbours of {:s}, skipping the '
                                'photometric offset averaging!'.format(ff_name))
                    continue

                # Compute the new photometric offset and improved standard deviation (assume equal sample size)
                #   Source: https://stats.stackexchange.com/questions/55999/is-it-possible-to-find-the-combined-standard-deviation
                photom_offset_new = np.mean(photom_offset_tmp_list)
                photom_offset_std_new = np.sqrt(
                    np.sum(
                        [
                            st ** 2 + (mt - photom_offset_new) ** 2
                            for mt, st in zip(photom_offset_tmp_list, photom_offset_std_tmp_list)
                        ]
                    )
                    / len(photom_offset_tmp_list)
                )

                # Assign the new photometric offset and standard deviation to all FFs used for computation
                for ff_name_tmp in neighboring_ffs:
                    recalibrated_platepars[ff_name_tmp].mag_lev = photom_offset_new
                    recalibrated_platepars[ff_name_tmp].mag_lev_stddev = photom_offset_std_new


        # Add the recalibrated platepars to the list of all recalibrated platepars
        recalibrated_platepars_all.update(recalibrated_platepars)

        
        ### Apply platepars to FTPdetectinfo ###
        meteor_output_list = []

        # Track the ECSV file names used in this run to guarantee uniqueness
        used_ecsv_names = set()

        # Resolve the ECSV output directory: an explicit path if given, otherwise the FTPdetectinfo
        #   directory. ecsv_out is None (disabled), True (use default dir), or a path string.
        if ecsv_out:
            ecsv_dir = ecsv_out if isinstance(ecsv_out, str) else dir_path
            if not os.path.exists(ecsv_dir):
                os.makedirs(ecsv_dir)

        log.info('Applying recalibrated platepars to meteor detections...')
        for meteor_index, meteor_entry in enumerate(meteor_list):

            ff_name, meteor_No, rho, phi, meteor_meas = meteor_entry

            # Per-meteor fps from the input FTPdetectinfo header (measured fps for accurate-timestamp
            #   inputs, config.fps otherwise). Used so the recalibrated timing stays accurate.
            fps_meteor = meteor_fps_list[meteor_index]

            # Find the entry in the CALSTARS file that is closest in time to the FTPdetectinfo FF file
            time_diff_func = lambda x: abs((ftp_ff_datetime_dict[ff_name] - calstars_datetime_dict[x]).total_seconds())
            closest_calstars_ff_name = min(calstars_datetime_dict, key=lambda x: time_diff_func(x))

            time_diff_s = time_diff_func(closest_calstars_ff_name)
            log.info('{:s} -> CALSTARS: {:s}, time diff: {:.1f} s'.format(
                ff_name, closest_calstars_ff_name, time_diff_s
            ))

            # Choose the platepar that will be applied to this FF file
            if closest_calstars_ff_name in recalibrated_platepars:
                working_platepar = recalibrated_platepars[closest_calstars_ff_name]

            else:
                log.info('Could not find a recalibrated platepar for: {:s}, using default platepar.'.format(ff_name))
                working_platepar = platepar

            # Apply the recalibrated platepar to meteor centroids
            meteor_picks = applyPlateparToCentroids(
                ff_name, fps_meteor, meteor_meas, working_platepar, add_calstatus=True
            )

            # Carry the per-meteor fps so the rewritten FTPdetectinfo header reflects the measured fps
            meteor_output_list.append([ff_name, meteor_No, rho, phi, meteor_picks, fps_meteor])

            # Dump the calibrated detection as an ECSV file
            if ecsv_out:
                writeMeteorECSV(ecsv_dir, config.stationID, ff_name, working_platepar, meteor_picks,
                    fps_meteor, used_ecsv_names, log)

        # Calibration string to be written to the FTPdetectinfo file
        calib_str = 'Recalibrated with RMS on: ' + str(RmsDateTime.utcnow()) + ' UTC'

        # Back up the old FTPdetectinfo file
        try:
            ftpdetectinfo_path = os.path.join(dir_path, ftpdetectinfo_file)
            shutil.copy(
                ftpdetectinfo_path,
                ftpdetectinfo_path.strip('.txt')
                + '_backup_{:s}.txt'.format(RmsDateTime.utcnow().strftime('%Y%m%d_%H%M%S.%f')),
            )
        except:
            log.info('ERROR! The FTPdetectinfo file could not be backed up: {:s}'.format(ftpdetectinfo_path))

        # Save the updated FTPdetectinfo
        FTPdetectinfo.writeFTPdetectinfo(
            meteor_output_list,
            dir_path,
            os.path.basename(ftpdetectinfo_file),
            dir_path,
            config.stationID,
            config.fps,
            calibration=calib_str,
            celestial_coords_given=True,
        )

        # If no platepars were recalibrated, use the single platepar recalibration procedure
        if len(recalibrated_platepars) == 0:

            log.info('No FF images were used for recalibration, using the single platepar calibration function...')

            # Use the initial platepar for calibration
            applyAstrometryFTPdetectinfo(dir_path, ftpdetectinfo_file, None, platepar=platepar)

            return recalibrated_platepars, ftpdetectinfo_file_list

    ### ###

    ### Store all recalibrated platepars as a JSON file ###

    all_pps = {}
    for ff_name in recalibrated_platepars_all:

        json_str = recalibrated_platepars_all[ff_name].jsonStr()

        all_pps[ff_name] = json.loads(json_str)

    with open(os.path.join(dir_path, config.platepars_recalibrated_name), 'w') as f:

        # Convert all platepars to a JSON file
        out_str = json.dumps(all_pps, default=lambda o: o.__dict__, indent=4, sort_keys=True)

        f.write(out_str)

    ### ###


    ### GENERATE PLOTS ###

    if generate_plot:

        # The plots are only informative, so a failure to generate them must never abort the processing of
        #   the night (the astrometry has already been applied and the recalibrated platepars written out)
        try:

            # Generate the name the plots
            plot_name = os.path.basename(ftpdetectinfo_path).replace('FTPdetectinfo_', '').replace('.txt', '')

            plotCalibrationVariation(recalibrated_platepars_all, platepar, config, dir_path, plot_name,
                                     ff_frames=calstars_ff_frames)

        except Exception as e:

            log.warning('Generating the recalibration plots failed with the message:\n' + repr(e))
            log.debug(repr(traceback.format_exception(*sys.exc_info())))

            # Close any half-built figure so that it doesn't leak into the plots generated later
            plt.close('all')

    ### ###


    return recalibrated_platepars_all, ftpdetectinfo_file_list


def applyRecalibrate(ftpdetectinfo_path, config, generate_plot=True, load_all=False, generate_ufoorbit=True,
                     debug=False, ecsv_out=False):
    """Recalibrate FF files with detections and apply the recalibrated platepar to those detections.
    Arguments:
        ftpdetectinfo_path: [str] Path to an FTPdetectinfo file.
        config: [Config instance]

    Keyword arguments:
        generate_plot: [bool] Generate the calibration variation plot. True by default.
        load_all: [bool] Load all FTPdetectinfo files in the directory and recalibrate them.
        generate_ufoorbit: [bool] Generate the UFOOrbit file. True by default.
        debug: [bool] If True, show debug plots for each fit iteration.
        ecsv_out: [None/bool/str] If set, dump each calibrated detection as an ECSV file. True writes to the
            FTPdetectinfo directory; a string is used as the output directory. None disables. None by default.

    Return:
        recalibrated_platepars: [dict] A dictionary where the keys are FF file names and values are
            recalibrated platepar instances for every FF file.

    """

    # Extract parent directory
    dir_path = os.path.dirname(ftpdetectinfo_path)

    # Get a list of files in the night folder
    file_list = sorted(os.listdir(dir_path))

    # Find and load the platepar file
    if config.platepar_name in file_list:

        # Load the platepar
        platepar = Platepar.Platepar()
        platepar.read(os.path.join(dir_path, config.platepar_name), use_flat=config.use_flat)

    else:
        log.info('Cannot find the platepar file in the night directory: {}'.format(config.platepar_name))
        sys.exit()

    # Find all CALSTARS files in the given folder
    calstars_file_list = []
    for calstars_file in file_list:
        if ('CALSTARS' in calstars_file) and ('.txt' in calstars_file):
            calstars_file_list.append(calstars_file)

    if not len(calstars_file_list):
        log.info('CALSTARS file could not be found in the given directory!')
        return {}

    # Load all calstars files in the directory
    calstars_list = []
    calstars_fps = None
    for calstars_file in calstars_file_list:

        # Load the calstars file (also read the fps recorded in the header, if present)
        calstars_list_file, chunk_frames, file_fps = CALSTARS.readCALSTARS(dir_path, calstars_file,
                                                                           return_fps=True)

        # Remember the fps from the header (used to reconstruct the chunk timing from the real frame rate
        #   instead of the possibly-wrong config fps)
        if file_fps is not None:
            calstars_fps = file_fps

        # Merge the previously loaded data with the new one
        for ff_name, star_data in calstars_list_file:

            # Check that the FF file hasn't been added already
            if ff_name not in [entry[0] for entry in calstars_list]:
                calstars_list.append([ff_name, star_data])

        log.info('CALSTARS file: ' + calstars_file + ' loaded!')

    # Add the number of frames in the FF stack to the CALSTARS data lislt
    calstars_data = (calstars_list, chunk_frames)

    # Recalibrate and apply astrometry on every FF file with detections individually
    recalibrated_platepars, ftpdetectinfo_file_list = recalibrateIndividualFFsAndApplyAstrometry(
        dir_path, ftpdetectinfo_path, calstars_data, config, platepar,
        generate_plot=generate_plot, load_all=load_all, debug=debug, ecsv_out=ecsv_out,
        calstars_fps=calstars_fps
    )

    ### Generate the updated UFOorbit file ###
    if generate_ufoorbit:
        log.debug('Generating UFOOrbit file...')
        for ftpdetectinfo_file in ftpdetectinfo_file_list:
            Utils.RMS2UFO.FTPdetectinfo2UFOOrbitInput(
                dir_path, ftpdetectinfo_file, None, platepar_dict=recalibrated_platepars
        )

    ### ###

    return recalibrated_platepars


if __name__ == "__main__":

    # COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(
        description="Recalibrate the platepar for every FF with detections and apply it the detections, recompute the FTPdetectinfo and UFOOrbit file."
    )

    arg_parser.add_argument('ftpdetectinfo_path', metavar='FTPDETECTINFO_PATH', type=str, 
                            help='Path to the FF file or a directory containing multiple FTPdetectinfo files.'
    )

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str,
        help="Path to a config file which will be used instead of the default one.",
    )

    arg_parser.add_argument('-r', '--report', action="store_true", 
                            help="""Show the calibration report at the end."""
    )

    arg_parser.add_argument('-a', '--all', action="store_true", 
        help="""Load all FTPdetectinfo and CALSTARS files in the directory and recalibrate them."""
    )

    arg_parser.add_argument('--skipuforbit', action="store_true",
        help="""Skip the generation of the UFOOrbit file."""
    )

    arg_parser.add_argument('--ecsvout', nargs='?', const=True, default=None, metavar='OUTPUT_DIR', type=str,
        help="""Also dump every calibrated detection as a GDEF ECSV file (one per meteor), readable by ASTRA. """
             """Optionally takes an output directory; defaults to the FTPdetectinfo directory."""
    )

    arg_parser.add_argument('-d', '--debug', action="store_true", 
        help="""Show debug plots for each fit iteration, displaying catalog stars, calstars, and matched stars."""
    )

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    # Find at least one FTPdetectinfo file in the given path (either direct file or a directory)
    ftpdetectinfo_path = cml_args.ftpdetectinfo_path

    try:
        ftpdetectinfo_path = findFTPdetectinfoFile(ftpdetectinfo_path)
    except FileNotFoundError:
        print('No FTPdetectinfo file found in the given path: {}'.format(cml_args.ftpdetectinfo_path))
        sys.exit()

    # Check if the given FTPdetectinfo file exists
    if not os.path.isfile(ftpdetectinfo_path):
        print('No FTPdetectinfo file in: {}'.format(ftpdetectinfo_path))
        sys.exit()

    # Resolve to an absolute path so that a bare relative file name doesn't collapse the parent directory
    #   to an empty string
    ftpdetectinfo_path = os.path.abspath(ftpdetectinfo_path)

    # Extract parent directory
    dir_path = os.path.dirname(ftpdetectinfo_path)

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, dir_path)

    # Initialize the logger
    log_manager = LoggingManager()
    log_manager.initLogging(config, 'recalibrate_', safedir=dir_path)

    # Get the logger handle
    log = getLogger("rmslogger", level="INFO")

    # Run the recalibration and recomputation
    applyRecalibrate(ftpdetectinfo_path, config, load_all=cml_args.all, generate_ufoorbit=(not cml_args.skipuforbit and not cml_args.all), debug=cml_args.debug, ecsv_out=cml_args.ecsvout)

    # Show the calibration report
    if cml_args.report:

        Utils.CalibrationReport.generateCalibrationReport(config, dir_path, show_graphs=True)
