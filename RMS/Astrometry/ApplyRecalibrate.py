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
    photometryFitRobust,
    rotationWrtHorizon,
)
from RMS.Astrometry.Conversions import date2JD, raDec2AltAz
from RMS.Astrometry.FFTalign import alignPlatepar
from RMS.Formats import CALSTARS, FFfile, FTPdetectinfo, Platepar, StarCatalog
from RMS.Formats.FTPdetectinfo import findFTPdetectinfoFile, validDefaultFTPdetectinfo
from RMS.Math import angularSeparation
from RMS.Logger import initLogging, getLogger
from RMS.Misc import RmsDateTime

# Neighbourhood size around individual FFs with detections which will be takes for recalibration
#   A size of e.g. 3 means that an FF before, the FF with the detection, an an FF after will be taken
RECALIBRATE_NEIGHBOURHOOD_SIZE = 3

# Get the logger from the main module
log = getLogger("logger", level="INFO")

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
        force_platepar_save: [bool] Skip the goodness of fit check and save the platepar.
        ignore_distance_threshold: [bool] Don't consider the recalib as failed if the median distance
            is larger than the threshold.
        ignore_max_stars: [bool] Ignore the maximum number of image stars for recalibration.

    Return:
        result: [?] A Platepar instance if refinement is successful, None if it failed.
        min_match_radius: [float] Minimum radius that successfully matched the stars (pixels).
    """

    working_platepar = copy.deepcopy(working_platepar)

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
        'Initially match stars with {:.1f} px: {:d}/{:d}'.format(min_radius, n_matched, len(star_dict_ff[jd]))
    )

    # If at least half the stars are matched with the smallest radius
    # Check if the average distance with the tightest radius is close
    if n_matched >= 0.5 * len(star_dict_ff[jd]) and avg_dist < config.dist_check_quick_threshold:
        # Use a reduced set of initial radius values
        radius_list = [1.5, min_radius]

        log.info('Using a quick fit...')

    ##########

    # Go through all radii and match the stars
    min_match_radius = None
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

        res = scipy.optimize.minimize(
            CheckFit._calcImageResidualsAstro,
            p0,
            args=(config, working_platepar, catalog_stars, star_dict_ff, match_radius),
            method='Nelder-Mead',
            options={'fatol': fatol, 'xatol': xatol_ang},
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
                    'Fitted star is farther from catalog star than necessary: {:.2f} > {:.2f} px'.format(dist, config.dist_check_threshold)
                )

            else:
                log.info(
                    'Number of matched stars after the fit is smaller than necessary: {:d} < {:d}'.format(n_matched < config.min_matched_stars)
                )

            # Indicate that the recalibration failed
            result = None
            break

        else:
            # If the fit was successful, use the new parameters from now on
            working_platepar = temp_platepar

            # Keep track of the minimum match radius
            min_match_radius = match_radius

            log.info('Astrometry fit successful with radius {:.1f} px!'.format(match_radius))

    # Choose which radius will be chosen for the goodness of fit check
    if max_match_radius is None:
        goodness_check_radius = match_radius

    else:
        goodness_check_radius = max_match_radius

    # If the platepar is good, store it
    if (
        CheckFit.checkFitGoodness(config, working_platepar, catalog_stars, star_dict_ff, goodness_check_radius)
        or force_platepar_save
    ):

        ### PHOTOMETRY FIT ###

        # Get a list of matched image and catalog stars
        image_stars, matched_catalog_stars, _ = matched_stars[jd]
        star_intensities = image_stars[:, 2]
        ra_catalog, dec_catalog, catalog_mags = matched_catalog_stars.T

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
        log.info('Not using the refined platepar...')
        result = None

    return result, min_match_radius


def recalibratePlateparsForFF(
    prev_platepar,
    ff_file_names,
    calstars,
    catalog_stars,
    config,
    lim_mag=None,
    ignore_distance_threshold=False,
    ignore_max_stars=False,
    ff_frames=256
):
    """
    Recalibrate platepars corresponding to ff files based on the stars.

    Arguments:
        prev_platepar: [platepar]
        ff_file_names: [list] list of ff file names
        calstars: [dict] A dictionary with only one entry, where the key is 'jd' and the value is the
            list of star coordinates.
        catalog_stars: [list] A list of entries [[ff_name, star_coordinates], ...].
        config: [config]

    Keyword arguments:
        lim_mag: [float]
        ignore_distance_threshold: [bool] Don't consider the recalib as failed if the median distance
            is larger than the threshold.
        ignore_max_stars: [bool] Ignore the maximum number of image stars for recalibration.
        ff_frames: [int] Number of frames in the FF file or frame chunk. Default is 256.

    Returns:
        recalibrated_platepars: [dict] A dictionary where one key is ff file name and the value is
            a calibrated corresponding platepar.
    """
    # Go through all FF files with detections, recalibrate and apply astrometry
    recalibrated_platepars = {}
    for ff_name in ff_file_names:

        working_platepar = copy.deepcopy(prev_platepar)

        # Skip this meteor if its FF file was already recalibrated
        if ff_name in recalibrated_platepars:
            continue

        log.info('Processing: {}'.format(ff_name))
        log.info('------------------------------------------------------------------------------')

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
            )

            # If the recalibration failed, try using FFT alignment
            if result is None:

                log.info('Running FFT alignment...')

                # Run FFT alignment
                calstars_coords = np.array(star_dict_ff[jd])[:, :2]
                calstars_coords[:, [0, 1]] = calstars_coords[:, [1, 0]]
                log.info(calstars_time)
                test_platepar = alignPlatepar(
                    config, prev_platepar, calstars_time, calstars_coords, show_plot=False
                )

                # Try to recalibrate after FFT alignment
                result, _ = recalibrateFF(
                    config, test_platepar, jd, star_dict_ff, catalog_stars, lim_mag=lim_mag
                )

                # If the FFT alignment failed, align the original platepar using the smallest radius that matched
                #   and force save the the platepar
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
                    )

                    if result is not None:
                        working_platepar = result

                # If the alignment succeeded, save the result
                else:
                    working_platepar = result

            else:
                working_platepar = result

        # Store the platepar if the fit succeeded
        if result is not None:

            # Recompute alt/az of the FOV centre
            working_platepar.az_centre, working_platepar.alt_centre = raDec2AltAz(
                working_platepar.RA_d,
                working_platepar.dec_d,
                working_platepar.JD,
                working_platepar.lat,
                working_platepar.lon,
            )

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
    pp_recalib_name, ignore_distance_threshold=False, ignore_max_stars=False):
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
    log.info('Loading star catalog with years from J2000: {:.2f}'.format(years_from_J2000))

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

    catalog_stars, _, config.star_catalog_band_ratios = star_catalog_status
    # log.info(catalog_stars)
    prev_platepar = Platepar.Platepar()
    prev_platepar.read(os.path.join(dir_path, config.platepar_name), use_flat=config.use_flat)

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


def recalibrateIndividualFFsAndApplyAstrometry(
    dir_path, ftpdetectinfo_path, calstars_data, config, platepar, 
    generate_plot=True, load_all=False
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

        # If the list is empty, skip the file
        if not meteor_list:
            log.info('No meteor detections in the FTPdetectinfo file!')
            continue

    
        # Create a dictionary mapping FF file names in FTPdetectinfo to datetime objects
        ftp_ff_datetime_dict = OrderedDict()
        for meteor_entry in meteor_list:
            ff_name = meteor_entry[0]
            ftp_ff_datetime_dict[ff_name] = FFfile.getMiddleTimeFF(ff_name, config.fps, dt_obj=True)


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
            prev_platepar, ff_processing_list, calstars, catalog_stars, config, ff_frames=calstars_ff_frames
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
                            photom_offset_tmp_list.append(recalibrated_platepars[ff_name_tmp].mag_lev)
                            photom_offset_std_tmp_list.append(recalibrated_platepars[ff_name_tmp].mag_lev_stddev)
                            neighboring_ffs.append(ff_name_tmp)

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
        log.info('Applying recalibrated platepars to meteor detections...')
        for meteor_entry in meteor_list:

            ff_name, meteor_No, rho, phi, meteor_meas = meteor_entry

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
                ff_name, config.fps, meteor_meas, working_platepar, add_calstatus=True
            )

            meteor_output_list.append([ff_name, meteor_No, rho, phi, meteor_picks])

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

    dt_list = []
    ang_dists = []
    rot_angles = []
    hour_list = []
    photom_offset_list = []
    photom_offset_std_list = []

    # If the length of the recalibrated platepars is less than 2, skip the plot generation
    if len(recalibrated_platepars_all) < 2:

        log.info('Less than 2 FF files were recalibrated, skipping the plot generation...')

        return recalibrated_platepars_all, ftpdetectinfo_file_list
    

    first_dt = np.min([FFfile.filenameToDatetime(ff_name) for ff_name in recalibrated_platepars_all])

    for ff_name in recalibrated_platepars_all:

        pp_temp = recalibrated_platepars_all[ff_name]

        # If the fitting failed, skip the platepar
        if pp_temp is None:
            continue

        # Add the datetime of the FF file to the list
        ff_dt = FFfile.filenameToDatetime(ff_name)
        dt_list.append(ff_dt)

        # Compute the angular separation from the reference platepar
        ang_dist = np.degrees(
            angularSeparation(
                np.radians(platepar.RA_d),
                np.radians(platepar.dec_d),
                np.radians(pp_temp.RA_d),
                np.radians(pp_temp.dec_d),
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


    if generate_plot:

        # Generate the name the plots
        plot_name = os.path.basename(ftpdetectinfo_path).replace('FTPdetectinfo_', '').replace('.txt', '')

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

        plt.figure()

        plt.errorbar(
            dt_list,
            photom_offset_list,
            yerr=photom_offset_std_list,
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

    ### ###


    return recalibrated_platepars_all, ftpdetectinfo_file_list


def applyRecalibrate(ftpdetectinfo_path, config, generate_plot=True, load_all=False, generate_ufoorbit=True):
    """Recalibrate FF files with detections and apply the recalibrated platepar to those detections.
    Arguments:
        ftpdetectinfo_path: [str] Path to an FTPdetectinfo file.
        config: [Config instance]

    Keyword arguments:
        generate_plot: [bool] Generate the calibration variation plot. True by default.
        load_all: [bool] Load all FTPdetectinfo files in the directory and recalibrate them.
        generate_ufoorbit: [bool] Generate the UFOOrbit file. True by default.

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
    for calstars_file in calstars_file_list:

        # Load the calstars file
        calstars_list_file, chunk_frames = CALSTARS.readCALSTARS(dir_path, calstars_file)

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
        generate_plot=generate_plot, load_all=load_all

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

    # Extract parent directory
    dir_path = os.path.dirname(ftpdetectinfo_path)

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, dir_path)

    # Initialize the logger
    initLogging(config, 'recalibrate_', safedir=dir_path)

    # Get the logger handle
    log = getLogger("logger", level="INFO")

    # Run the recalibration and recomputation
    applyRecalibrate(ftpdetectinfo_path, config, load_all=cml_args.all, generate_ufoorbit=(not cml_args.skipuforbit))

    # Show the calibration report
    if cml_args.report:

        Utils.CalibrationReport.generateCalibrationReport(config, dir_path, show_graphs=True)
