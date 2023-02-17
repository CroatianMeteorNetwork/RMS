""" Automatic refining of astrometry calibration. The initial astrometric calibration is needed, which will be 
    refined by using all stars from a given night.
"""

from __future__ import absolute_import, division, print_function

import argparse
import copy
import os
import random
import shutil
import sys
import logging

import matplotlib.pyplot as plt # 
import numpy as np
# Import Cython functions
import pyximport
import RMS.ConfigReader as cr
import scipy.optimize
from RMS.Astrometry.ApplyAstrometry import (getFOVSelectionRadius, raDecToXYPP,
                                            rotationWrtHorizon, xyToRaDecPP)
from RMS.Astrometry.Conversions import date2JD, jd2Date, raDec2AltAz
from RMS.Astrometry.FFTalign import alignPlatepar
from RMS.Formats import CALSTARS, FFfile, Platepar, StarCatalog
from RMS.Math import angularSeparation
from RMS.Logger import initLogging

pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from RMS.Astrometry.CyFunctions import matchStars, subsetCatalog


# Get the logger from the main module
log = logging.getLogger("logger")


def computeMinimizationTolerances(config, platepar, star_dict_len):
    """ Compute tolerances for minimization. """

    # Calculate the function tolerance, so the desired precision can be reached (the number is calculated
    # in the same regard as the cost function)
    fatol = (config.dist_check_threshold**2)/np.sqrt(star_dict_len*config.min_matched_stars + 1)

    # Parameter estimation tolerance for angular values
    fov_w = platepar.X_res/platepar.F_scale
    xatol_ang = config.dist_check_threshold*fov_w/platepar.X_res

    return fatol, xatol_ang




def matchStarsResiduals(config, platepar, catalog_stars, star_dict, match_radius, ret_nmatch=False, \
    sky_coords=False, lim_mag=None, verbose=False):
    """ Match the image and catalog stars with the given astrometry solution and estimate the residuals
        between them.

    Arguments:
        config: [Config structure]
        platepar: [Platepar structure] Astrometry parameters.
        catalog_stars: [ndarray] An array of catalog stars (ra, dec, mag).
        star_dict: [ndarray] A dictionary where the keys are JDs when the stars were recorded and values are
            2D list of stars, each entry is (X, Y, bg_level, level, fwhm).
        match_radius: [float] Maximum radius for star matching (pixels).
        min_matched_stars: [int] Minimum number of matched stars on the image for the image to be accepted.
    Keyword arguments:
        ret_nmatch: [bool] If True, the function returns the number of matched stars and the average
            deviation. False by default.
        sky_coords: [bool] If True, sky coordinate residuals in RA, dec will be used to compute the cost,
            function, not image coordinates.
        lim_mag: [float] Override the limiting magnitude from config. None by default.
        verbose: [bool] Print results. True by default.
    Return:
        cost: [float] The cost function which weights the number of matched stars and the average deviation.
    """


    if lim_mag is None:
        lim_mag = config.catalog_mag_limit


    # Estimate the FOV radius
    fov_radius = getFOVSelectionRadius(platepar)


    # Dictionary containing the matched stars, the keys are JDs of every image
    matched_stars = {}


    # Go through every FF image and its stars
    for jd in star_dict:

        # Estimate RA,dec of the centre of the FOV
        _, RA_c, dec_c, _ = xyToRaDecPP([jd2Date(jd)], [platepar.X_res/2], [platepar.Y_res/2], [1], \
            platepar, extinction_correction=False)

        RA_c = RA_c[0]
        dec_c = dec_c[0]

        # Get stars from the catalog around the defined center in a given radius
        _, extracted_catalog = subsetCatalog(catalog_stars, RA_c, dec_c, jd, platepar.lat, platepar.lon, \
            fov_radius, lim_mag)
        ra_catalog, dec_catalog, mag_catalog = extracted_catalog.T


        # Extract stars for the given Julian date
        stars_list = star_dict[jd]
        stars_list = np.array(stars_list)

        # If the type is not float, it means something went wrong, so skip this
        if not (stars_list.dtype == np.float64):
            continue

        # Convert all catalog stars to image coordinates
        cat_x_array, cat_y_array = raDecToXYPP(ra_catalog, dec_catalog, jd, platepar)

        # Take only those stars which are within the FOV
        x_indices = np.argwhere((cat_x_array >= 0) & (cat_x_array < platepar.X_res))
        y_indices = np.argwhere((cat_y_array >= 0) & (cat_y_array < platepar.Y_res))
        cat_good_indices = np.intersect1d(x_indices, y_indices).astype(np.uint32)

        # cat_x_array = cat_x_array[good_indices]
        # cat_y_array = cat_y_array[good_indices]


        # # Plot image stars
        # im_y, im_x, _, _ = stars_list.T
        # plt.scatter(im_y, im_x, facecolors='none', edgecolor='g')

        # # Plot catalog stars
        # plt.scatter(cat_y_array[cat_good_indices], cat_x_array[cat_good_indices], c='r', s=20, marker='+')

        # plt.show()


        # Match image and catalog stars
        matched_indices = matchStars(stars_list, cat_x_array, cat_y_array, cat_good_indices, match_radius)

        # Skip this image is no stars were matched
        if len(matched_indices) < config.min_matched_stars:
            continue

        matched_indices = np.array(matched_indices)
        matched_img_inds, matched_cat_inds, dist_list = matched_indices.T

        # Extract data from matched stars
        matched_img_stars = stars_list[matched_img_inds.astype(int)]
        matched_cat_stars = extracted_catalog[matched_cat_inds.astype(int)]

        # Put the matched stars to a dictionary
        matched_stars[jd] = [matched_img_stars, matched_cat_stars, dist_list]

        # # Plot matched stars
        # im_y, im_x, _, _ = matched_img_stars.T
        # cat_y = cat_y_array[matched_cat_inds.astype(int)]
        # cat_x = cat_x_array[matched_cat_inds.astype(int)]

        # plt.scatter(im_x, im_y, c='r', s=5)
        # plt.scatter(cat_x, cat_y, facecolors='none', edgecolor='g')

        # plt.xlim([0, platepar.X_res])
        # plt.ylim([platepar.Y_res, 0])

        # plt.show()



    # If residuals on the image should be computed
    if not sky_coords:

        unit_label = 'px'

        # Extract all distances
        global_dist_list = []
        # level_list = []
        # mag_list = []
        for jd in matched_stars:
            # matched_img_stars, matched_cat_stars, dist_list = matched_stars[jd]

            _, _, dist_list = matched_stars[jd]

            global_dist_list += dist_list.tolist()

            # # TEST
            # level_list += matched_img_stars[:, 3].tolist()
            # mag_list += matched_cat_stars[:, 2].tolist()



        # # Plot levels vs. magnitudes
        # plt.scatter(mag_list, np.log10(level_list))
        # plt.xlabel('Magnitude')
        # plt.ylabel('Log10 level')
        # plt.show()

    # Compute the residuals on the sky
    else:

        unit_label = 'arcmin'

        global_dist_list = []

        # Go through all matched stars
        for jd in matched_stars:

            matched_img_stars, matched_cat_stars, dist_list = matched_stars[jd]

            # Go through all stars on the image
            for img_star_entry, cat_star_entry in zip(matched_img_stars, matched_cat_stars):

                # Extract star coords
                star_y = img_star_entry[0]
                star_x = img_star_entry[1]
                cat_ra = cat_star_entry[0]
                cat_dec = cat_star_entry[1]

                # Convert image coordinates to RA/Dec
                _, star_ra, star_dec, _ = xyToRaDecPP([jd2Date(jd)], [star_x], [star_y], [1], \
                    platepar, extinction_correction=False)

                # Compute angular distance between the predicted and the catalog position
                ang_dist = np.degrees(angularSeparation(np.radians(cat_ra), np.radians(cat_dec), \
                    np.radians(star_ra[0]), np.radians(star_dec[0])))

                # Store the angular separation in arc minutes
                global_dist_list.append(ang_dist*60)



    # Number of matched stars
    n_matched = len(global_dist_list)

    if n_matched == 0:

        if verbose:
            log.info('No matched stars with radius {:.1f} px!'.format(match_radius))

        if ret_nmatch:
            return 0, 9999.0, 9999.0, {}

        else:
            return 9999.0

    # Calculate the average distance
    avg_dist = np.median(global_dist_list)

    cost = avg_dist**2/((n_matched+1)/len(ra_catalog))
    # cost = avg_dist**2/np.sqrt(n_matched+1)

    if verbose:

        log.info("")
        log.info("Matched {:d} stars with radius of {:.1f} px".format(n_matched, match_radius))
        log.info("    Average distance = {:.3f} {:s}".format(avg_dist, unit_label))
        log.info("    Cost function    = {:.5f}".format(cost))


    if ret_nmatch:
        return n_matched, avg_dist, cost, matched_stars

    else:
        return cost



def checkFitGoodness(config, platepar, catalog_stars, star_dict, match_radius, verbose=False):
    """ Checks if the platepar is 'good enough', given the extracted star positions. Returns True if the
        fit is deemed good, False otherwise. The goodness of fit is determined by 2 criteria: the average
        star residual (in pixels) has to be below a certain threshold, and an average number of matched stars
        per image has to be above a predefined threshold as well.

    Arguments:
        config: [Config structure]
        platepar: [Platepar structure] Initial astrometry parameters.
        catalog_stars: [ndarray] An array of catalog stars (ra, dec, mag).
        star_dict: [ndarray] A dictionary where the keys are JDs when the stars were recorded and values are
            2D list of stars, each entry is (X, Y, bg_level, level).
        match_radius: [float] Maximum radius for star matching (pixels).
    Keyword arguments:
        verbose: [bool] If True, fit status will be printed on the screen. False by default.
    Return:
        [bool] True if the platepar is good, False otherwise.
    """

    if verbose:
        log.info("")
        log.info("CHECK FIT GOODNESS:")

    # Match the stars and calculate the residuals
    n_matched, avg_dist, cost, matched_stars = matchStarsResiduals(config, platepar, catalog_stars, \
        star_dict, match_radius, ret_nmatch=True, verbose=verbose)



    # ### Plot zenith distance vs. residual

    # # Go through all images
    # for jd in matched_stars:
    #     _, cat_stars, dists = matched_stars[jd]

    #     # Extract RA/Dec
    #     ra, dec, _ = cat_stars.T

    #     zangle_list = []
    #     for ra_t, dec_t in zip(ra, dec):

    #         # Compute zenith distance
    #         azim, elev = raDec2AltAz(ra_t, dec_t, jd, platepar.lat, platepar.lon)

    #         zangle = 90 - elev

    #         zangle_list.append(zangle)


    #     # Plot zangle vs. distance
    #     plt.scatter(zangle_list, dists, c='k', s=0.1)


    # plt.xlabel('Zenith angle')
    # plt.ylabel('Residual (px)')
    # plt.show()

    # ###


    # Check that the average distance is within the threshold
    if avg_dist <= config.dist_check_threshold:

        if verbose:
            log.info("")
            log.info('The minimum residual is satisfied!')

        # Check that the minimum number of stars is matched per every image
        if n_matched >= len(star_dict)*1:

            return True

        else:
            if verbose:
                log.info('But there are not enough stars on every image, recalibrating...')


    return False





def _calcImageResidualsAstro(params, config, platepar, catalog_stars, star_dict, match_radius):
    """ Calculates the differences between the stars on the image and catalog stars in image coordinates with 
        the given astrometrical solution. 

    Arguments:
        params: [list] Fit parameters - reference RA, Dec, position angle, and scale.
        config: [Config]
        platepar: [Platepar]
        catalog_stars: [list] List of (ra, dec, mag) entries (angles in degrees).
        star_dict: [dict] Dictionary which contains the JD, and a list of (X, Y, bg_intens, intens) of the
            stars on the image.
        match_radius: [float] Star match radius (px).

    Return:
        [float] The average pixel residual (difference between image and catalog positions) normalized
            by the square root of the total number of matched stars.
    """


    # Make a copy of the platepar
    pp = copy.deepcopy(platepar)

    # Extract fitting parameters
    ra_ref, dec_ref, pos_angle_ref, F_scale = params

    # Set the fitting parameters to the platepar clone
    pp.RA_d = ra_ref
    pp.dec_d = dec_ref
    pp.pos_angle_ref = pos_angle_ref
    pp.F_scale = F_scale

    # Match stars and calculate image residuals
    return matchStarsResiduals(config, pp, catalog_stars, star_dict, match_radius, verbose=False)




def starListToDict(config, calstars_list, max_ffs=None):
    """ Converts the list of calstars into dictionary where the keys are FF file JD and the values is
        a list of (X, Y, bg_intens, intens) of stars.
    """

    # Convert the list to a dictionary
    calstars = {ff_file: star_data for ff_file, star_data in calstars_list}

    # Dictionary which will contain the JD, and a list of (X, Y, bg_intens, intens) of the stars
    star_dict = {}

    # Take only those files with enough stars on them
    for ff_name in calstars:

        stars_list = calstars[ff_name]

        # Check if there are enough stars on the image
        if len(stars_list) >= config.ff_min_stars:

            # Calculate the JD time of the FF file
            dt = FFfile.getMiddleTimeFF(ff_name, config.fps, ret_milliseconds=True)
            jd = date2JD(*dt)

            # Add the time and the stars to the dict
            star_dict[jd] = stars_list


    if max_ffs is not None:

        # Limit the number of FF files used
        if len(star_dict) > max_ffs:

            # Randomly choose calstars_files_N image files from the whole list
            rand_keys = random.sample(list(star_dict), max_ffs)
            star_dict = {key: star_dict[key] for key in rand_keys}


    return star_dict




def autoCheckFit(config, platepar, calstars_list, _fft_refinement=False):
    """ Attempts to refine the astrometry fit with the given stars and and initial astrometry parameters.
    Arguments:
        config: [Config structure]
        platepar: [Platepar structure] Initial astrometry parameters.
        calstars_list: [list] A list containing stars extracted from FF files. See RMS.Formats.CALSTARS for
            more details.
    Keyword arguments:
        _fft_refinement: [bool] Internal flag indicating that autoCF is running the second time recursively
            after FFT platepar adjustment.

    Return:
        (platepar, fit_status):
            platepar: [Platepar structure] Estimated/refined platepar.
            fit_status: [bool] True if fit was successfuly, False if not.
    """


    def _handleFailure(config, platepar, calstars_list, catalog_stars, _fft_refinement):
        """ Run FFT alignment before giving up on ACF. """

        if not _fft_refinement:

            log.info("")
            log.info("-------------------------------------------------------------------------------")
            log.info('The initial platepar is bad, trying to refine it using FFT phase correlation...')
            log.info("")

            # Prepare data for FFT image registration

            calstars_dict = {ff_file: star_data for ff_file, star_data in calstars_list}

            # Extract star list from CALSTARS file from FF file with most stars
            max_len_ff = max(calstars_dict, key=lambda k: len(calstars_dict[k]))

            # Take only X, Y (change order so X is first)
            calstars_coords = np.array(calstars_dict[max_len_ff])[:, :2]
            calstars_coords[:, [0, 1]] = calstars_coords[:, [1, 0]]

            # Get the time of the FF file
            calstars_time = FFfile.getMiddleTimeFF(max_len_ff, config.fps, ret_milliseconds=True)


            # Try aligning the platepar using FFT image registration
            platepar_refined = alignPlatepar(config, platepar, calstars_time, calstars_coords)


            ### If there are still not enough stars matched, try FFT again ###
            min_radius = 10

            # Prepare star dictionary to check the match
            dt = FFfile.getMiddleTimeFF(max_len_ff, config.fps, ret_milliseconds=True)
            jd = date2JD(*dt)
            star_dict_temp = {}
            star_dict_temp[jd] = calstars_dict[max_len_ff]

            # Check the number of matched stars
            n_matched, _, _, _ = matchStarsResiduals(config, platepar_refined, catalog_stars, \
                star_dict_temp, min_radius, ret_nmatch=True, verbose=True)

            # Realign again if necessary
            if n_matched < config.min_matched_stars:
                log.info('')
                log.info("-------------------------------------------------------------------------------")
                log.info('Doing a second FFT pass as the number of matched stars was too small...')
                log.info('')
                platepar_refined = alignPlatepar(config, platepar_refined, calstars_time, calstars_coords)
                log.info('')

            ### ###


            # Redo autoCF
            return autoCheckFit(config, platepar_refined, calstars_list, _fft_refinement=True)

        else:
            log.info('Auto Check Fit failed completely, please redo the plate manually!')
            return platepar, False


    if _fft_refinement:
        log.info('Second ACF run with an updated platepar via FFT phase correlation...')


    # Load catalog stars (overwrite the mag band ratios if specific catalog is used)
    catalog_stars, _, config.star_catalog_band_ratios = StarCatalog.readStarCatalog(config.star_catalog_path, \
        config.star_catalog_file, lim_mag=config.catalog_mag_limit, \
        mag_band_ratios=config.star_catalog_band_ratios)


    # Dictionary which will contain the JD, and a list of (X, Y, bg_intens, intens) of the stars
    star_dict = starListToDict(config, calstars_list, max_ffs=config.calstars_files_N)

    # There has to be a minimum of 200 FF files for star fitting
    if len(star_dict) < config.calstars_files_N:
        log.info('Not enough FF files in CALSTARS for ACF!')
        return platepar, False


    # Calculate the total number of calibration stars used
    total_calstars = sum([len(star_dict[key]) for key in star_dict])
    log.info('Total calstars: {:d}'.format(total_calstars))

    if total_calstars < config.calstars_min_stars:
        log.info('Not enough calibration stars, need at least {}'.format(config.calstars_min_stars))
        return platepar, False


    # A list of matching radiuses to try
    min_radius = 0.5
    radius_list = [10, 5, 3, 1.5, min_radius]


    # Calculate the function tolerance, so the desired precision can be reached (the number is calculated
    # in the same regard as the cost function)
    fatol, xatol_ang = computeMinimizationTolerances(config, platepar, len(star_dict))


    ### If the initial match is good enough, do only quick recalibratoin ###

    # Match the stars and calculate the residuals
    n_matched, avg_dist, cost, _ = matchStarsResiduals(config, platepar, catalog_stars, star_dict, \
        min_radius, ret_nmatch=True)

    if n_matched >= config.calstars_files_N:

        # Check if the average distance with the tightest radius is close
        if avg_dist < config.dist_check_quick_threshold:

            log.info("Using quick fit with smaller radiia...")

            # Use a reduced set of initial radius values
            radius_list = [1.5, min_radius]

    ##########


    # Match increasingly smaller search radiia around image stars
    for i, match_radius in enumerate(radius_list):

        # Match the stars and calculate the residuals
        n_matched, avg_dist, cost, _ = matchStarsResiduals(config, platepar, catalog_stars, star_dict, \
            match_radius, ret_nmatch=True)

        log.info('')
        log.info("-------------------------------------------------------------")
        log.info("Refining camera pointing with max pixel deviation = {:.1f} px".format(match_radius))
        log.info("Initial values:")
        log.info("    Matched stars     = {:>6d}".format(n_matched))
        log.info("    Average deviation = {:>6.2f} px".format(avg_dist))


        # The initial number of matched stars has to be at least the number of FF imaages, otherwise it means
        #   that the initial platepar is no good
        if n_matched < config.calstars_files_N:
            log.info("The total number of initially matched stars is too small! Please manually redo the plate or make sure there are enough calibration stars.")

            # Try to refine the platepar with FFT phase correlation and redo the ACF
            return _handleFailure(config, platepar, calstars_list, catalog_stars, _fft_refinement)


        # Check if the platepar is good enough and do not estimate further parameters
        if checkFitGoodness(config, platepar, catalog_stars, star_dict, min_radius, verbose=True):

            # Print out notice only if the platepar is good right away
            if i == 0:
                log.info("Initial platepar is good enough!")

            return platepar, True


        # Initial parameters for the astrometric fit
        p0 = [platepar.RA_d, platepar.dec_d, platepar.pos_angle_ref, platepar.F_scale]

        # Fit the astrometric parameters
        res = scipy.optimize.minimize(_calcImageResidualsAstro, p0, args=(config, platepar, catalog_stars, \
            star_dict, match_radius), method='Nelder-Mead', \
            options={'fatol': fatol, 'xatol': xatol_ang})

        log.info(res)

        # If the fit was not successful, stop further fitting
        if not res.success:

            # Try to refine the platepar with FFT phase correlation and redo the ACF
            return _handleFailure(config, platepar, calstars_list, catalog_stars, _fft_refinement)


        else:
            # If the fit was successful, use the new parameters from now on
            ra_ref, dec_ref, pos_angle_ref, F_scale = res.x

            platepar.RA_d = ra_ref
            platepar.dec_d = dec_ref
            platepar.pos_angle_ref = pos_angle_ref
            platepar.F_scale = F_scale



        # Check if the platepar is good enough and do not estimate further parameters
        if checkFitGoodness(config, platepar, catalog_stars, star_dict, min_radius, verbose=True):
            return platepar, True



    # Match the stars and calculate the residuals
    n_matched, avg_dist, cost, matched_stars = matchStarsResiduals(config, platepar, catalog_stars, \
        star_dict, min_radius, ret_nmatch=True)

    log.info("FINAL SOLUTION with radius {:.1} px:".format(min_radius))
    log.info("    Matched stars     = {:>6d}".format(n_matched))
    log.info("    Average deviation = {:>6.2f} px".format(avg_dist))


    # Mark the platepar to indicate that it was automatically refined with CheckFit
    platepar.auto_check_fit_refined = True

    # Recompute alt/az of the FOV centre
    platepar.az_centre, platepar.alt_centre = raDec2AltAz(platepar.RA_d, platepar.dec_d, platepar.JD, \
        platepar.lat, platepar.lon)

    # Recompute the rotation wrt horizon
    platepar.rotation_from_horiz = rotationWrtHorizon(platepar)



    return platepar, True




if __name__ == "__main__":


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Check if the calibration file matches the stars, and improve it.")

    arg_parser.add_argument('dir_path', nargs=1, metavar='DIR_PATH', type=str, \
        help='Path to the folder with FF or image files. This folder also has to contain the platepar file.')

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    dir_path = cml_args.dir_path[0]

    # Check if the given directory is OK
    if not os.path.exists(dir_path):
        log.info('No such directory: {}'.format(dir_path))
        sys.exit()


    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, dir_path)

    # Initialize the logger
    initLogging(config, 'checkfit_')

    # Get the logger handle
    log = logging.getLogger("logger")
    log.setLevel(logging.INFO)

    # Get a list of files in the night folder
    file_list = os.listdir(dir_path)



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




    # Run the automatic astrometry fit
    pp, fit_status = autoCheckFit(config, platepar, calstars_list)


    # If the fit suceeded, save the platepar
    if fit_status:

        log.info('ACF sucessful!')

        # Save the old platepar
        shutil.move(os.path.join(dir_path, config.platepar_name), os.path.join(dir_path, 
            config.platepar_name + '.old'))

        # Save the new platepar
        pp.write(os.path.join(dir_path, config.platepar_name))
