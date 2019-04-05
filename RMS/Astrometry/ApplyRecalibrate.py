""" Recalibrate the platepar for every FF with detections and compute the astrometry with recalibrated
    values. 
"""

from __future__ import print_function, division, absolute_import

import os
import sys
import copy
import argparse
import json
import datetime
import shutil

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from RMS.Astrometry import CheckFit
from RMS.Astrometry.ApplyAstrometry import applyPlateparToCentroids
from RMS.Astrometry.Conversions import date2JD
import RMS.ConfigReader as cr
from RMS.Formats import CALSTARS
from RMS.Formats import FFfile
from RMS.Formats import FTPdetectinfo
from RMS.Formats import Platepar
from RMS.Formats import StarCatalog
from RMS.Math import angularSeparation
from Utils.RMS2UFO import FTPdetectinfo2UFOOrbitInput


def recalibrateIndividualFFsAndApplyAstrometry(dir_path, ftpdetectinfo_path, calstars_list, config, platepar):
    """ Recalibrate FF files with detections and apply the recalibrated platepar to those detections. """


    # Read the FTPdetectinfo data
    cam_code, fps, meteor_list = FTPdetectinfo.readFTPdetectinfo(*os.path.split(ftpdetectinfo_path), \
        ret_input_format=True)

    # Convert the list of stars to a per FF name dictionary
    calstars = {ff_file: star_data for ff_file, star_data in calstars_list}


    # Load catalog stars (overwrite the mag band ratios if specific catalog is used)
    catalog_stars, _, config.star_catalog_band_ratios = StarCatalog.readStarCatalog(config.star_catalog_path,\
        config.star_catalog_file, lim_mag=config.catalog_mag_limit, \
        mag_band_ratios=config.star_catalog_band_ratios)



    prev_platepar = copy.deepcopy(platepar)

    # Go through all FF files with detections, recalibrate and apply astrometry
    recalibrated_platepars = {}
    for meteor_entry in meteor_list:

        working_platepar = copy.deepcopy(platepar)

        ff_name, meteor_No, rho, phi, meteor_meas = meteor_entry

        # Skip this meteors if its FF file was already recalibrated
        if ff_name in recalibrated_platepars:
            continue

        print()
        print('Processing: ', ff_name)
        print('------------------------------------------------------------------------------')

        # Find extracted stars on this image
        if not ff_name in calstars:
            print('Skipped because it was not in CALSTARS:', ff_name)
            continue

        # Get stars detected on this FF file
        dt = FFfile.getMiddleTimeFF(ff_name, config.fps, ret_milliseconds=True)
        jd = date2JD(*dt)
        star_dict_ff = {jd: calstars[ff_name]}



        # A list of matching radiuses to try, pairs of [radius, fit_distorsion_flag]
        #   The distorsion will be fitted only if explicity requested
        min_radius = 0.5
        radius_list = [10, 5, 3, 1.5, min_radius]


        # Calculate the function tolerance, so the desired precision can be reached (the number is calculated
        # in the same regard as the cost function)
        fatol, xatol_ang = CheckFit.computeMinimizationTolerances(config, working_platepar, len(star_dict_ff))


        ### If the initial match is good enough, do only quick recalibratoin ###
         
        # Match the stars and calculate the residuals
        n_matched, avg_dist, cost, _ = CheckFit.matchStarsResiduals(config, working_platepar, catalog_stars, \
            star_dict_ff, min_radius, ret_nmatch=True)


        print('Initally match stars with {:.1f} px: {:d}/{:d}'.format(min_radius, n_matched, \
            len(star_dict_ff[jd])))

        # If at least half the stars are matched
        if n_matched >= 0.5*len(star_dict_ff[jd]):

            # Check if the average distance with the tightest radius is close
            if avg_dist < config.dist_check_quick_threshold:

                # Use a reduced set of initial radius values
                radius_list = [1.5, min_radius]

                print('Using quick fit on:', ff_name)
            

        ##########

        # Go through all radiia and match the stars
        for match_radius in radius_list:

            # If the platepar is good, don't recalibrate anymore
            if CheckFit.checkFitGoodness(config, working_platepar, catalog_stars, star_dict_ff, match_radius):
                break

            ### Recalibrate the platepar just on these stars, use the default platepar for initial params ###
            
            # Don't fit the scale
            p0 = [working_platepar.RA_d, working_platepar.dec_d, working_platepar.pos_angle_ref]
            fit_distorsion = False

            # Compute the minimization tolerance
            fatol, xatol_ang = CheckFit.computeMinimizationTolerances(config, working_platepar, \
                len(star_dict_ff))

            res = scipy.optimize.minimize(CheckFit._calcImageResidualsAstro, p0, args=(config, \
                working_platepar, catalog_stars, star_dict_ff, match_radius, fit_distorsion), \
                method='Nelder-Mead', options={'fatol': fatol, 'xatol': xatol_ang})


            ###

            #print(res)


            # If the fit was not successful, stop further fitting on this FF file
            if not res.success:

                # Indicate that the recalibration failed
                recalibrated_platepars[ff_name] = None
                break


            else:
                # If the fit was successful, use the new parameters from now on
                ra_ref, dec_ref, pos_angle_ref = res.x
                working_platepar.RA_d = ra_ref
                working_platepar.dec_d = dec_ref
                working_platepar.pos_angle_ref = pos_angle_ref


        # If the platepar is good, store it
        if CheckFit.checkFitGoodness(config, working_platepar, catalog_stars, star_dict_ff, match_radius):

            # Mark the platepar to indicate that it was automatically refined with CheckFit
            working_platepar.auto_check_fit_refined = True

            # Reset the star list
            working_platepar.star_list = []
            
            # Store the platepar to the list of recalibrated platepars
            recalibrated_platepars[ff_name] = working_platepar


        # Otherwise, store the previous successfully fitted platepar
        else:
            recalibrated_platepars[ff_name] = prev_platepar


    # Set the previous platepar if the fit succeeded
    if recalibrated_platepars[ff_name] is not None:
        prev_platepar = working_platepar

    else:
        # If the fit failed, set the previous platepar as the one that should be used for this FF file
        recalibrated_platepars[ff_name] = prev_platepar


    ### Store all recalibrated platepars as a JSON file ###

    all_pps = {}
    for ff_name in recalibrated_platepars:

        json_str = recalibrated_platepars[ff_name].jsonStr()
        
        all_pps[ff_name] = json.loads(json_str)

    with open(os.path.join(dir_path, 'platepars_all_recalibrated.json'), 'w') as f:
        
        # Convert all platepars to a JSON file
        out_str = json.dumps(all_pps, default=lambda o: o.__dict__, indent=4, sort_keys=True)

        f.write(out_str)

    ### ###


    ### Plot difference from reference platepar in angular distance from (0, 0) vs rotation ###

    ang_dists = []
    rot_angles = []
    hour_list = []

    first_jd = np.min([FFfile.filenameToDatetime(ff_name) for ff_name in recalibrated_platepars])

    for ff_name in recalibrated_platepars:
        
        pp_temp = recalibrated_platepars[ff_name]

        # If the fitting failed, skip the platepar
        if pp_temp is None:
            continue

        # Compute the angular separation from the reference platepar
        ang_dist = np.degrees(angularSeparation(np.radians(platepar.RA_d), np.radians(platepar.dec_d), \
            np.radians(pp_temp.RA_d), np.radians(pp_temp.dec_d)))
        ang_dists.append(ang_dist*60)

        rot_angles.append((platepar.pos_angle_ref - pp_temp.pos_angle_ref)*60)

        # Compute the hour of the FF used for recalibration
        hour_list.append((FFfile.filenameToDatetime(ff_name) - first_jd).total_seconds()/3600)



    plt.scatter(0, 0, marker='o', edgecolor='k', label='Reference platepar', s=100, c='none', zorder=3)

    plt.scatter(ang_dists, rot_angles, c=hour_list, zorder=3)
    plt.colorbar(label='Hours from first FF file')
    
    plt.xlabel("Angular distance from reference (arcmin)")
    plt.ylabel('Rotation from reference (arcmin)')

    plt.grid()
    plt.legend()

    plt.tight_layout()

    # Generate the name for the plot
    calib_plot_name = os.path.basename(ftpdetectinfo_path).replace('FTPdetectinfo_', '').replace('.txt', '') \
        + '_calibration_variation.png'

    plt.savefig(os.path.join(dir_path, calib_plot_name), dpi=150)

    # plt.show()

    ### ###



    ### Apply platepars to FTPdetectinfo ###

    meteor_output_list = []
    for meteor_entry in meteor_list:

        ff_name, meteor_No, rho, phi, meteor_meas = meteor_entry

        # Get the platepar that will be applied to this FF file
        if ff_name in recalibrated_platepars:
            working_platepar = recalibrated_platepars[ff_name]

        else:
            print('Using default platepar for:', ff_name)
            working_platepar = platepar

        # Apply the recalibrated platepar to meteor centroids
        meteor_picks = applyPlateparToCentroids(ff_name, fps, meteor_meas, working_platepar, \
            add_calstatus=True)

        meteor_output_list.append([ff_name, meteor_No, rho, phi, meteor_picks])


    # Calibration string to be written to the FTPdetectinfo file
    calib_str = 'Recalibrated with RMS on: ' + str(datetime.datetime.utcnow()) + ' UTC'

    # If no meteors were detected, set dummpy parameters
    if len(meteor_list) == 0:
        cam_code = ''
        fps = 0


    # Back up the old FTPdetectinfo file
    shutil.copy(ftpdetectinfo_path, ftpdetectinfo_path.strip('.txt') \
        + '_backup_{:s}.txt'.format(datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S.%f')))

    # Save the updated FTPdetectinfo
    FTPdetectinfo.writeFTPdetectinfo(meteor_output_list, dir_path, os.path.basename(ftpdetectinfo_path), \
        dir_path, cam_code, fps, calibration=calib_str, celestial_coords_given=True)


    ### ###


    
    ### Generate the updated UFOorbit file ###

    FTPdetectinfo2UFOOrbitInput(dir_path, os.path.basename(ftpdetectinfo_path), None, \
        platepar_dict=recalibrated_platepars)

    ### ###



        





if __name__ == "__main__":

        ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Recalibrate the platepar for every FF with detections and apply it the detections, recompute the FTPdetectinfo and UFOOrbit file.")

    arg_parser.add_argument('ftpdetectinfo_path', nargs=1, metavar='FTPDETECTINFO_PATH', type=str, \
        help='Path to the FF file.')

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    ftpdetectinfo_path = cml_args.ftpdetectinfo_path[0]

    # Check if the given FTPdetectinfo file exists
    if not os.path.isfile(ftpdetectinfo_path):
        print('No such file:', ftpdetectinfo_path)
        sys.exit()


    # Extract parent directory
    dir_path = os.path.dirname(ftpdetectinfo_path)


    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, dir_path)


    # Get a list of files in the night folder
    file_list = os.listdir(dir_path)



    # Find and load the platepar file
    if config.platepar_name in file_list:

        # Load the platepar
        platepar = Platepar.Platepar()
        platepar.read(os.path.join(dir_path, config.platepar_name))

    else:
        print('Cannot find the platepar file in the night directory: ', config.platepar_name)
        sys.exit()


    # Find the CALSTARS file in the given folder
    calstars_file = None
    for calstars_file in file_list:
        if ('CALSTARS' in calstars_file) and ('.txt' in calstars_file):
            break

    if calstars_file is None:
        print('CALSTARS file could not be found in the given directory!')
        sys.exit()

    # Load the calstars file
    calstars_list = CALSTARS.readCALSTARS(dir_path, calstars_file)

    print('CALSTARS file: ' + calstars_file + ' loaded!')

    # Recalibrate and apply astrometry on every FF file with detections individually
    recalibrateIndividualFFsAndApplyAstrometry(dir_path, ftpdetectinfo_path, calstars_list, config, platepar)