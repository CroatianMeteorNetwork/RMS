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
from RMS.Formats.FTPdetectinfo import findFTPdetectinfoFile
from RMS.Math import angularSeparation
from RMS.Logger import initLogging

# Neighbourhood size around individual FFs with detections which will be takes for recalibration
#   A size of e.g. 3 means that an FF before, the FF with the detection, an an FF after will be taken
RECALIBRATE_NEIGHBOURHOOD_SIZE = 3

# Get the logger from the main module
log = logging.getLogger("logger")
log.setLevel(logging.INFO)

def loadRecalibratedPlatepar(dir_path, config, file_list=None, type='meteor'):
    """
    Gets recalibrated platpars. If they were already computed, load them, otherwise compute them and save

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
    Keyword argumnets:
        max_match_radius: [float] Maximum radius used for star matching. None by default, which uses all 
            hardcoded values.
        force_platepar_save: [bool] Skip the goodness of fit check and save the platepar.
        ignore_distance_threshold: [bool] Don't consider the recalib as failed if the median distance
            is larger than the threshold.
    Return:
        result: [?] A Platepar instance if refinement is successful, None if it failed.
        min_match_radius: [float] Minimum radius that successfuly matched the stars (pixels).
    """
    working_platepar = copy.deepcopy(working_platepar)

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

    ### If the initial match is good enough, do only quick recalibratoin ###

    # Match the stars and calculate the residuals
    n_matched, avg_dist, cost, _ = CheckFit.matchStarsResiduals(
        config, working_platepar, catalog_stars, star_dict_ff, min_radius, ret_nmatch=True, lim_mag=lim_mag
    )

    log.info(
        'Initally match stars with {:.1f} px: {:d}/{:d}'.format(min_radius, n_matched, len(star_dict_ff[jd]))
    )

    # If at least half the stars are matched with the smallest radius
    # Check if the average distance with the tightest radius is close
    if n_matched >= 0.5 * len(star_dict_ff[jd]) and avg_dist < config.dist_check_quick_threshold:
        # Use a reduced set of initial radius values
        radius_list = [1.5, min_radius]

        log.info('Using a quick fit...')

    ##########

    # Go through all radiia and match the stars
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
        goodnes_check_radius = match_radius

    else:
        goodnes_check_radius = max_match_radius

    # If the platepar is good, store it
    if (
        CheckFit.checkFitGoodness(config, working_platepar, catalog_stars, star_dict_ff, goodnes_check_radius)
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
                config.dist_check_threshold, goodnes_check_radius
            )
        )

        log.info('Saving improved platepar...')

        ### plot for checking fit quality ###
        # if date2JD(*FFfile.getMiddleTimeFF("FF_HR000N_20201214_203429_089_0416000.fits", config.fps)) == jd:
        #     plt.title(f'F_HR000K_20201213_225055_929_0631040.fits')
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

        # Get stars detected on this FF file (create a dictionaly with only one entry, the residuals function
        #   needs this format)
        calstars_time = FFfile.getMiddleTimeFF(ff_name, config.fps, ret_milliseconds=True)
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


def recalibrateSelectedFF(dir_path, ff_file_names, calstars_list, config, lim_mag, \
    pp_recalib_name, ignore_distance_threshold=False):
    """Recalibrate FF files, ignoring whether there are detections.

    Arguments:
        dir_path: [str] Path where the FF files are.
        ff_file_names: [str] List of ff files to recalibrate platepars to
        calstars_list: [list] A list of entries [[ff_name, star_coordinates], ...].
        config: [Config instance]
        lim_mag: [float] Limiting magnitude for the catalog.
        pp_recalib_name: [str] Name for the file where the recalibrated platepars will be stored as JSON.

    Keyword arguments:
        ignore_distance_threshold: [bool] Don't consider the recalib as failed if the median distance
            is larger than the threshold.

    Return:
        recalibrated_platepars: [dict] A dictionary where the keys are FF file names and values are
            recalibrated platepar instances for every FF file.
    """
    config = copy.deepcopy(config)
    calstars = {ff_file: star_data for ff_file, star_data in calstars_list}

    # load star catalog with increased catalog limiting magnitude
    star_catalog_status = StarCatalog.readStarCatalog(
        config.star_catalog_path,
        config.star_catalog_file,
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
    dir_path, ftpdetectinfo_path, calstars_list, config, platepar, generate_plot=True
):
    """Recalibrate FF files with detections and apply the recalibrated platepar to those detections.
    Arguments:
        dir_path: [str] Path where the FTPdetectinfo file is.
        ftpdetectinfo_path: [str] Name of the FTPdetectinfo file.
        calstars_list: [list] A list of entries [[ff_name, star_coordinates], ...].
        config: [Config instance]
        platepar: [Platepar instance] Initial platepar.
    Keyword arguments:
        generate_plot: [bool] Generate the calibration variation plot. True by default.
    Return:
        recalibrated_platepars: [dict] A dictionary where the keys are FF file names and values are
            recalibrated platepar instances for every FF file.
    """

    # Use a copy of the config file
    config = copy.deepcopy(config)

    # If the given file does not exits, return nothing
    if not os.path.isfile(ftpdetectinfo_path):
        log.info('ERROR! The FTPdetectinfo file does not exist: {:s}'.format(ftpdetectinfo_path))
        log.info('    The recalibration on every file was not done!')

        return {}

    # Read the FTPdetectinfo data
    cam_code, fps, meteor_list = FTPdetectinfo.readFTPdetectinfo(
        *os.path.split(ftpdetectinfo_path), ret_input_format=True
    )

    # Convert the list of stars to a per FF name dictionary
    calstars = {ff_file: star_data for ff_file, star_data in calstars_list}

    ### Add neighboring FF files for more robust photometry estimation ###

    ff_processing_list = []

    # Make a list of sorted FF files in CALSTARS
    calstars_ffs = sorted(calstars.keys())

    # Go through the list of FF files with detections and add neighboring FFs
    for meteor_entry in meteor_list:

        ff_name = meteor_entry[0]

        if ff_name in calstars_ffs:

            # Find the index of the given FF file in the list of calstars
            ff_indx = calstars_ffs.index(ff_name)

            # Add neighbours to the processing list
            for k in range(-(RECALIBRATE_NEIGHBOURHOOD_SIZE // 2), RECALIBRATE_NEIGHBOURHOOD_SIZE // 2 + 1):

                k_indx = ff_indx + k

                if (k_indx > 0) and (k_indx < len(calstars_ffs)):

                    ff_name_tmp = calstars_ffs[k_indx]
                    if ff_name_tmp not in ff_processing_list:
                        ff_processing_list.append(ff_name_tmp)

    # Sort the processing list of FF files
    ff_processing_list = sorted(ff_processing_list)

    ### ###

    # Globally increase catalog limiting magnitude
    config.catalog_mag_limit += 1

    # Load catalog stars (overwrite the mag band ratios if specific catalog is used)
    star_catalog_status = StarCatalog.readStarCatalog(
        config.star_catalog_path,
        config.star_catalog_file,
        lim_mag=config.catalog_mag_limit,
        mag_band_ratios=config.star_catalog_band_ratios,
    )

    if not star_catalog_status:
        log.info("Could not load the star catalog!")
        log.info(os.path.join(config.star_catalog_path, config.star_catalog_file))
        return {}

    catalog_stars, _, config.star_catalog_band_ratios = star_catalog_status
    log.info('MAX {}'.format(np.mean([np.percentile(data[1], 90) for data in catalog_stars])))
    # Update the platepar coordinates from the config file
    platepar.lat = config.latitude
    platepar.lon = config.longitude
    platepar.elev = config.elevation

    prev_platepar = copy.deepcopy(platepar)

    # Go through all FF files with detections, recalibrate and apply astrometry
    recalibrated_platepars = recalibratePlateparsForFF(
        prev_platepar, ff_processing_list, calstars, catalog_stars, config
    )

    ### Average out photometric offsets within the given neighbourhood size ###

    # Go through the list of FF files with detections
    for meteor_entry in meteor_list:

        ff_name = meteor_entry[0]

        # Make sure the FF was successfuly recalibrated
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

                    # Check that the neighboring FF was successfuly recalibrated
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

    ### ###

    ### Store all recalibrated platepars as a JSON file ###

    all_pps = {}
    for ff_name in recalibrated_platepars:

        json_str = recalibrated_platepars[ff_name].jsonStr()

        all_pps[ff_name] = json.loads(json_str)

    with open(os.path.join(dir_path, config.platepars_recalibrated_name), 'w') as f:

        # Convert all platepars to a JSON file
        out_str = json.dumps(all_pps, default=lambda o: o.__dict__, indent=4, sort_keys=True)

        f.write(out_str)

    ### ###

    # If no platepars were recalibrated, use the single platepar recalibration procedure
    if len(recalibrated_platepars) == 0:

        log.info('No FF images were used for recalibration, using the single platepar calibration function...')

        # Use the initial platepar for calibration
        applyAstrometryFTPdetectinfo(dir_path, os.path.basename(ftpdetectinfo_path), None, platepar=platepar)

        return recalibrated_platepars

    ### GENERATE PLOTS ###

    dt_list = []
    ang_dists = []
    rot_angles = []
    hour_list = []
    photom_offset_list = []
    photom_offset_std_list = []

    first_dt = np.min([FFfile.filenameToDatetime(ff_name) for ff_name in recalibrated_platepars])

    for ff_name in recalibrated_platepars:

        pp_temp = recalibrated_platepars[ff_name]

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

    ### Apply platepars to FTPdetectinfo ###

    meteor_output_list = []
    for meteor_entry in meteor_list:

        ff_name, meteor_No, rho, phi, meteor_meas = meteor_entry

        # Get the platepar that will be applied to this FF file
        if ff_name in recalibrated_platepars:
            working_platepar = recalibrated_platepars[ff_name]

        else:
            log.info('Using default platepar for: {}'.format(ff_name))
            working_platepar = platepar

        # Apply the recalibrated platepar to meteor centroids
        meteor_picks = applyPlateparToCentroids(
            ff_name, fps, meteor_meas, working_platepar, add_calstatus=True
        )

        meteor_output_list.append([ff_name, meteor_No, rho, phi, meteor_picks])

    # Calibration string to be written to the FTPdetectinfo file
    calib_str = 'Recalibrated with RMS on: ' + str(datetime.datetime.utcnow()) + ' UTC'

    # If no meteors were detected, set dummpy parameters
    if len(meteor_list) == 0:
        cam_code = ''
        fps = 0

    # Back up the old FTPdetectinfo file
    try:
        shutil.copy(
            ftpdetectinfo_path,
            ftpdetectinfo_path.strip('.txt')
            + '_backup_{:s}.txt'.format(datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S.%f')),
        )
    except:
        log.info('ERROR! The FTPdetectinfo file could not be backed up: {:s}'.format(ftpdetectinfo_path))

    # Save the updated FTPdetectinfo
    FTPdetectinfo.writeFTPdetectinfo(
        meteor_output_list,
        dir_path,
        os.path.basename(ftpdetectinfo_path),
        dir_path,
        cam_code,
        fps,
        calibration=calib_str,
        celestial_coords_given=True,
    )

    ### ###
    return recalibrated_platepars


def applyRecalibrate(ftpdetectinfo_path, config, generate_plot=True):
    """Recalibrate FF files with detections and apply the recalibrated platepar to those detections.
    Arguments:
        ftpdetectinfo_path: [str] Name of the FTPdetectinfo file.
        config: [Config instance]
    Keyword arguments:
        generate_plot: [bool] Generate the calibration variation plot. True by default.
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

    # Find the CALSTARS file in the given folder
    calstars_file = None
    for calstars_file in file_list:
        if ('CALSTARS' in calstars_file) and ('.txt' in calstars_file):
            break

    if calstars_file is None:
        log.info('CALSTARS file could not be found in the given directory!')
        sys.exit()

    # Load the calstars file
    calstars_list = CALSTARS.readCALSTARS(dir_path, calstars_file)

    log.info('CALSTARS file: ' + calstars_file + ' loaded!')

    # Recalibrate and apply astrometry on every FF file with detections individually
    recalibrated_platepars = recalibrateIndividualFFsAndApplyAstrometry(
        dir_path, ftpdetectinfo_path, calstars_list, config, platepar, generate_plot=generate_plot
    )

    ### Generate the updated UFOorbit file ###

    Utils.RMS2UFO.FTPdetectinfo2UFOOrbitInput(
        dir_path, os.path.basename(ftpdetectinfo_path), None, platepar_dict=recalibrated_platepars
    )

    ### ###

    return recalibrated_platepars


if __name__ == "__main__":

    # COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(
        description="Recalibrate the platepar for every FF with detections and apply it the detections, recompute the FTPdetectinfo and UFOOrbit file."
    )

    arg_parser.add_argument(
        'ftpdetectinfo_path', nargs=1, metavar='FTPDETECTINFO_PATH', type=str, help='Path to the FF file.'
    )

    arg_parser.add_argument(
        '-c',
        '--config',
        nargs=1,
        metavar='CONFIG_PATH',
        type=str,
        help="Path to a config file which will be used instead of the default one.",
    )

    arg_parser.add_argument(
        '-r', '--report', action="store_true", help="""Show the calibration report at the end."""
    )

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    ftpdetectinfo_path = cml_args.ftpdetectinfo_path[0]
    ftpdetectinfo_path = findFTPdetectinfoFile(ftpdetectinfo_path)

    # Check if the given FTPdetectinfo file exists
    if not os.path.isfile(ftpdetectinfo_path):
        print('No such file: {}'.format(ftpdetectinfo_path))
        sys.exit()

    # Extract parent directory
    dir_path = os.path.dirname(ftpdetectinfo_path)

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, dir_path)

    # Initialize the logger
    initLogging(config, 'recalibrate_', safedir=dir_path)

    # Get the logger handle
    log = logging.getLogger("logger")
    log.setLevel(logging.INFO)

    # Run the recalibration and recomputation
    applyRecalibrate(ftpdetectinfo_path, config)

    # Show the calibration report
    if cml_args.report:

        Utils.CalibrationReport.generateCalibrationReport(config, dir_path, show_graphs=True)
