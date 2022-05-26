""" Generate a calibration report given the folder of the night. """


from __future__ import print_function, division, absolute_import

import os
import json

import numpy as np
import matplotlib.pyplot as plt

from RMS.Astrometry.ApplyAstrometry import computeFOVSize, xyToRaDecPP, raDecToXYPP, \
    photometryFitRobust, correctVignetting, photomLine, rotationWrtHorizon, \
    extinctionCorrectionTrueToApparent, getFOVSelectionRadius
from RMS.Astrometry.CheckFit import matchStarsResiduals
from RMS.Astrometry.Conversions import date2JD, jd2Date, raDec2AltAz
from RMS.Formats.CALSTARS import readCALSTARS
from RMS.Formats.FFfile import validFFName, getMiddleTimeFF
from RMS.Formats.FFfile import read as readFF
from RMS.Formats.Platepar import Platepar
from RMS.Formats import StarCatalog
from RMS.Routines import Image
from RMS.Routines.AddCelestialGrid import addEquatorialGrid

# Import Cython functions
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from RMS.Astrometry.CyFunctions import subsetCatalog


def generateCalibrationReport(config, night_dir_path, match_radius=2.0, platepar=None, show_graphs=False):
    """ Given the folder of the night, find the Calstars file, check the star fit and generate a report
        with the quality of the calibration. The report contains information about both the astrometry and
        the photometry calibration. Graphs will be saved in the given directory of the night.

    Arguments:
        config: [Config instance]
        night_dir_path: [str] Full path to the directory of the night.
    Keyword arguments:
        match_radius: [float] Match radius for star matching between image and catalog stars (px).
        platepar: [Platepar instance] Use this platepar instead of finding one in the folder.
        show_graphs: [bool] Show the graphs on the screen. False by default.
    Return:
        None
    """

    # Find the CALSTARS file in the given folder
    calstars_file = None
    for calstars_file in os.listdir(night_dir_path):
        if ('CALSTARS' in calstars_file) and ('.txt' in calstars_file):
            break

    if calstars_file is None:
        print('CALSTARS file could not be found in the given directory!')
        return None


    # Load the calstars file
    star_list = readCALSTARS(night_dir_path, calstars_file)



    ### Load recalibrated platepars, if they exist ###

    # Find recalibrated platepars file per FF file
    platepars_recalibrated_file = None
    for file_name in os.listdir(night_dir_path):
        if file_name == config.platepars_recalibrated_name:
            platepars_recalibrated_file = file_name
            break


    # Load all recalibrated platepars if the file is available
    recalibrated_platepars = None
    if platepars_recalibrated_file:
        with open(os.path.join(night_dir_path, platepars_recalibrated_file)) as f:
            recalibrated_platepars = json.load(f)
            print('Loaded recalibrated platepars JSON file for the calibration report...')

    ### ###


    ### Load the platepar file ###

    # Find the platepar file in the given directory if it was not given
    if platepar is None:

        # Find the platepar file
        platepar_file = None
        for file_name in os.listdir(night_dir_path):
            if file_name == config.platepar_name:
                platepar_file = file_name
                break

        if platepar_file is None:
            print('The platepar cannot be found in the night directory!')
            return None

        # Load the platepar file
        platepar = Platepar()
        platepar.read(os.path.join(night_dir_path, platepar_file), use_flat=config.use_flat)


    ### ###


    night_name = os.path.split(night_dir_path.strip(os.sep))[1]


    # Go one mag deeper than in the config
    lim_mag = config.catalog_mag_limit + 1

    # Load catalog stars (load one magnitude deeper)
    catalog_stars, mag_band_str, config.star_catalog_band_ratios = StarCatalog.readStarCatalog(\
        config.star_catalog_path, config.star_catalog_file, lim_mag=lim_mag, \
        mag_band_ratios=config.star_catalog_band_ratios)


    ### Take only those CALSTARS entires for which FF files exist in the folder ###

    # Get a list of FF files in the folder
    ff_list = []
    for file_name in os.listdir(night_dir_path):
        if validFFName(file_name):
            ff_list.append(file_name)


    # Filter out calstars entries, generate a star dictionary where the keys are JDs of FFs
    star_dict = {}
    ff_dict = {}
    for entry in star_list:

        ff_name, star_data = entry

        # Check if the FF from CALSTARS exists in the folder
        if ff_name not in ff_list:
            continue


        dt = getMiddleTimeFF(ff_name, config.fps, ret_milliseconds=True)
        jd = date2JD(*dt)

        # Add the time and the stars to the dict
        star_dict[jd] = star_data
        ff_dict[jd] = ff_name

    ### ###

    # If there are no FF files in the directory, don't generate a report
    if len(star_dict) == 0:
        print('No FF files from the CALSTARS file in the directory!')
        return None


    # If the recalibrated platepars file exists, take the one with the most stars
    max_jd = 0
    using_recalib_platepars = False
    if recalibrated_platepars is not None:
        max_stars = 0
        for ff_name_temp in recalibrated_platepars:

            # Compute the Julian date of the FF middle
            dt = getMiddleTimeFF(ff_name_temp, config.fps, ret_milliseconds=True)
            jd = date2JD(*dt)

            # Check that this file exists in CALSTARS and the list of FF files
            if (jd not in star_dict) or (jd not in ff_dict):
                continue

            # Make sure that the chosen file has been successfuly recalibrated
            if "auto_recalibrated" in recalibrated_platepars[ff_name_temp]:
                if not recalibrated_platepars[ff_name_temp]["auto_recalibrated"]:
                    continue

            # Check if the number of stars on this FF file is larger than the before
            if len(star_dict[jd]) > max_stars:
                max_jd = jd
                max_stars = len(star_dict[jd])


        # Set a flag to indicate if using recalibrated platepars has failed
        if max_jd == 0:
            using_recalib_platepars = False
        else:

            print('Using recalibrated platepars, file:', ff_dict[max_jd])
            using_recalib_platepars = True

            # Select the platepar where the FF file has the most stars
            platepar_dict = recalibrated_platepars[ff_dict[max_jd]]
            platepar = Platepar()
            platepar.loadFromDict(platepar_dict, use_flat=config.use_flat)

            filtered_star_dict = {max_jd: star_dict[max_jd]}

            # Match stars on the image with the stars in the catalog
            n_matched, avg_dist, cost, matched_stars = matchStarsResiduals(config, platepar, catalog_stars, \
                filtered_star_dict, match_radius, ret_nmatch=True, lim_mag=lim_mag)

            max_matched_stars = n_matched


    # Otherwise take the optimal FF file for evaluation
    if (recalibrated_platepars is None) or (not using_recalib_platepars):

        # If there are more than a set number of FF files to evaluate, choose only the ones with most stars on
        #   the image
        if len(star_dict) > config.calstars_files_N:

            # Find JDs of FF files with most stars on them
            top_nstars_indices = np.argsort([len(x) for x in star_dict.values()])[::-1][:config.calstars_files_N \
                - 1]

            filtered_star_dict = {}
            for i in top_nstars_indices:
                filtered_star_dict[list(star_dict.keys())[i]] = list(star_dict.values())[i]

            star_dict = filtered_star_dict


        # Match stars on the image with the stars in the catalog
        n_matched, avg_dist, cost, matched_stars = matchStarsResiduals(config, platepar, catalog_stars, \
            star_dict, match_radius, ret_nmatch=True, lim_mag=lim_mag)



    # If no recalibrated platepars where found, find the image with the largest number of matched stars
    if (not using_recalib_platepars) or (max_jd == 0):

        max_jd = 0
        max_matched_stars = 0
        for jd in matched_stars:
            _, _, distances = matched_stars[jd]
            if len(distances) > max_matched_stars:
                max_jd = jd
                max_matched_stars = len(distances)


        # If there are no matched stars, use the image with the largest number of detected stars
        if max_matched_stars <= 2:
            max_jd = max(star_dict, key=lambda x: len(star_dict[x]))
            distances = [np.inf]



    # Take the FF file with the largest number of matched stars
    ff_name = ff_dict[max_jd]

    # Load the FF file
    ff = readFF(night_dir_path, ff_name)
    img_h, img_w = ff.avepixel.shape

    dpi = 200
    plt.figure(figsize=(ff.avepixel.shape[1]/dpi, ff.avepixel.shape[0]/dpi), dpi=dpi)

    # Take the average pixel
    img = ff.avepixel

    # Slightly adjust the levels
    img = Image.adjustLevels(img, np.percentile(img, 1.0), 1.3, np.percentile(img, 99.99))

    plt.imshow(img, cmap='gray', interpolation='nearest')

    legend_handles = []


    # Plot detected stars
    for img_star in star_dict[max_jd]:

        y, x = img_star[:2]

        rect_side = 5*match_radius
        square_patch = plt.Rectangle((x - rect_side/2, y - rect_side/2), rect_side, rect_side, color='g', \
            fill=False, label='Image stars')

        plt.gca().add_artist(square_patch)

    legend_handles.append(square_patch)



    # If there are matched stars, plot them
    if max_matched_stars > 2:

        # Take the solution with the largest number of matched stars
        image_stars, matched_catalog_stars, distances = matched_stars[max_jd]

        # Plot matched stars
        for img_star in image_stars:
            x, y = img_star[:2]

            circle_patch = plt.Circle((y, x), radius=3*match_radius, color='y', fill=False, \
                label='Matched stars')

            plt.gca().add_artist(circle_patch)

        legend_handles.append(circle_patch)


        ### Plot match residuals ###

        # Compute preducted positions of matched image stars from the catalog
        x_predicted, y_predicted = raDecToXYPP(matched_catalog_stars[:, 0], \
            matched_catalog_stars[:, 1], max_jd, platepar)

        img_y = image_stars[:, 0]
        img_x = image_stars[:, 1]

        delta_x = x_predicted - img_x
        delta_y = y_predicted - img_y

        # Compute image residual and angle of the error
        res_angle = np.arctan2(delta_y, delta_x)
        res_distance = np.sqrt(delta_x**2 + delta_y**2)


        # Calculate coordinates of the beginning of the residual line
        res_x_beg = img_x + 3*match_radius*np.cos(res_angle)
        res_y_beg = img_y + 3*match_radius*np.sin(res_angle)

        # Calculate coordinates of the end of the residual line
        res_x_end = img_x + 100*np.cos(res_angle)*res_distance
        res_y_end = img_y + 100*np.sin(res_angle)*res_distance

        # Plot the 100x residuals
        for i in range(len(x_predicted)):
            res_plot = plt.plot([res_x_beg[i], res_x_end[i]], [res_y_beg[i], res_y_end[i]], color='orange', \
                lw=0.5, label='100x residuals')

        legend_handles.append(res_plot[0])

        ### ###

    else:

        distances = [np.inf]

        # If there are no matched stars, plot large text in the middle of the screen
        plt.text(img_w/2, img_h/2, "NO MATCHED STARS!", color='r', alpha=0.5, fontsize=20, ha='center',
            va='center')


    ### Plot positions of catalog stars to the limiting magnitude of the faintest matched star + 1 mag ###

    # Find the faintest magnitude among matched stars
    if max_matched_stars > 2:
        faintest_mag = np.max(matched_catalog_stars[:, 2]) + 1

    else:
        # If there are no matched stars, use the limiting magnitude from config
        faintest_mag = config.catalog_mag_limit + 1


    # Estimate RA,dec of the centre of the FOV
    _, RA_c, dec_c, _ = xyToRaDecPP([jd2Date(max_jd)], [platepar.X_res/2], [platepar.Y_res/2], [1], 
        platepar, extinction_correction=False)

    RA_c = RA_c[0]
    dec_c = dec_c[0]

    fov_radius = getFOVSelectionRadius(platepar)

    # Get stars from the catalog around the defined center in a given radius
    _, extracted_catalog = subsetCatalog(catalog_stars, RA_c, dec_c, max_jd, platepar.lat, platepar.lon, \
        fov_radius, faintest_mag)
    ra_catalog, dec_catalog, mag_catalog = extracted_catalog.T

    # Compute image positions of all catalog stars that should be on the image
    x_catalog, y_catalog = raDecToXYPP(ra_catalog, dec_catalog, max_jd, platepar)

    # Filter all catalog stars outside the image
    temp_arr = np.c_[x_catalog, y_catalog, mag_catalog]
    temp_arr = temp_arr[temp_arr[:, 0] >= 0]
    temp_arr = temp_arr[temp_arr[:, 0] <= ff.avepixel.shape[1]]
    temp_arr = temp_arr[temp_arr[:, 1] >= 0]
    temp_arr = temp_arr[temp_arr[:, 1] <= ff.avepixel.shape[0]]
    x_catalog, y_catalog, mag_catalog = temp_arr.T

    # Plot catalog stars on the image
    cat_stars_handle = plt.scatter(x_catalog, y_catalog, c='none', marker='D', lw=1.0, alpha=0.4, \
        s=((4.0 + (faintest_mag - mag_catalog))/3.0)**(2*2.512), edgecolor='r', label='Catalog stars')

    legend_handles.append(cat_stars_handle)

    ### ###


    # Add info text in the corner
    info_text = ff_dict[max_jd] + '\n' \
        + "Matched stars within {:.1f} px radius: {:d}/{:d} \n".format(match_radius, max_matched_stars, \
            len(star_dict[max_jd])) \
        + "Median distance = {:.2f} px\n".format(np.median(distances)) \
        + "Catalog lim mag = {:.1f}".format(lim_mag)

    plt.text(10, 10, info_text, bbox=dict(facecolor='black', alpha=0.5), va='top', ha='left', fontsize=4, \
        color='w', family='monospace')

    legend = plt.legend(handles=legend_handles, prop={'size': 4}, loc='upper right')
    legend.get_frame().set_facecolor('k')
    legend.get_frame().set_edgecolor('k')
    for txt in legend.get_texts():
        txt.set_color('w')



    ### Add FOV info (centre, size) ###

    # Mark FOV centre
    plt.scatter(platepar.X_res/2, platepar.Y_res/2, marker='+', s=20, c='r', zorder=4)

    # Compute FOV centre alt/az
    azim_centre, alt_centre = raDec2AltAz(RA_c, dec_c, max_jd, platepar.lat, platepar.lon)

    # Compute FOV size
    fov_h, fov_v = computeFOVSize(platepar)

    # Compute the rotation wrt. horizon
    rot_horizon = rotationWrtHorizon(platepar)

    fov_centre_text = "Azim  = {:6.2f}$\\degree$\n".format(azim_centre) \
                    + "Alt   = {:6.2f}$\\degree$\n".format(alt_centre) \
                    + "Rot h = {:6.2f}$\\degree$\n".format(rot_horizon) \
                    + "FOV h = {:6.2f}$\\degree$\n".format(fov_h) \
                    + "FOV v = {:6.2f}$\\degree$".format(fov_v) \

    plt.text(10, platepar.Y_res - 10, fov_centre_text, bbox=dict(facecolor='black', alpha=0.5), \
        va='bottom', ha='left', fontsize=4, color='w', family='monospace')



    ### ###


    # Plot RA/Dec gridlines #
    addEquatorialGrid(plt, platepar, max_jd)



    plt.axis('off')
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)

    plt.xlim([0, ff.avepixel.shape[1]])
    plt.ylim([ff.avepixel.shape[0], 0])

    # Remove the margins (top and right are set to 0.9999, as setting them to 1.0 makes the image blank in 
    #   some matplotlib versions)
    plt.subplots_adjust(left=0, bottom=0, right=0.9999, top=0.9999, wspace=0, hspace=0)

    plt.savefig(os.path.join(night_dir_path, night_name + '_calib_report_astrometry.jpg'), \
        bbox_inches='tight', pad_inches=0, dpi=dpi)


    if show_graphs:
        plt.show()

    else:
        plt.clf()
        plt.close()



    if max_matched_stars > 2:


        ### PHOTOMETRY FIT ###

        # If a flat is used, set the vignetting coeff to 0
        if config.use_flat:
            platepar.vignetting_coeff = 0.0

        # Extact intensities and mangitudes
        star_intensities = image_stars[:, 2]
        catalog_ra, catalog_dec, catalog_mags = matched_catalog_stars.T

        # Compute radius of every star from image centre
        radius_arr = np.hypot(image_stars[:, 0] - img_h/2, image_stars[:, 1] - img_w/2)

        # Compute apparent extinction corrected magnitudes
        catalog_mags = extinctionCorrectionTrueToApparent(catalog_mags, catalog_ra, catalog_dec, max_jd, \
            platepar)

        # Fit the photometry on automated star intensities (use the fixed vignetting coeff, use robust fit)
        photom_params, fit_stddev, fit_resid, star_intensities, radius_arr, catalog_mags = \
            photometryFitRobust(star_intensities, radius_arr, catalog_mags, \
            fixed_vignetting=platepar.vignetting_coeff)


        photom_offset, _ = photom_params

        ### ###



        ### PLOT PHOTOMETRY ###
        # Note: An almost identical code exists in RMS.Astrometry.SkyFit in the PlateTool.photometry function

        dpi = 130
        fig_p, (ax_p, ax_r) = plt.subplots(nrows=2, facecolor=None, figsize=(6.0, 7.0), dpi=dpi, \
            gridspec_kw={'height_ratios':[2, 1]})

        # Plot raw star intensities
        ax_p.scatter(-2.5*np.log10(star_intensities), catalog_mags, s=5, c='r', alpha=0.5, \
            label="Raw (extinction corrected)")

        # If a flat is used, disregard the vignetting
        if not config.use_flat:

            # Plot intensities of image stars corrected for vignetting
            lsp_corr_arr = np.log10(correctVignetting(star_intensities, radius_arr, \
                platepar.vignetting_coeff))
            ax_p.scatter(-2.5*lsp_corr_arr, catalog_mags, s=5, c='b', alpha=0.5, \
                label="Corrected for vignetting")


        # Plot photometric offset from the platepar
        x_min, x_max = ax_p.get_xlim()
        y_min, y_max = ax_p.get_ylim()

        x_min_w = x_min - 3
        x_max_w = x_max + 3
        y_min_w = y_min - 3
        y_max_w = y_max + 3

        photometry_info = "Platepar: {:+.1f}*LSP + {:.2f} +/- {:.2f}".format(platepar.mag_0, \
            platepar.mag_lev, platepar.mag_lev_stddev) \
            + "\nVignetting coeff = {:.5f}".format(platepar.vignetting_coeff) \
            + "\nGamma = {:.2f}".format(platepar.gamma)

        # Plot the photometry calibration from the platepar
        logsum_arr = np.linspace(x_min_w, x_max_w, 10)
        ax_p.plot(logsum_arr, logsum_arr + platepar.mag_lev, label=photometry_info, linestyle='--', \
            color='k', alpha=0.5)

        # Plot the fitted photometry calibration
        fit_info = "Fit: {:+.1f}*LSP + {:.2f} +/- {:.2f}".format(-2.5, photom_offset, fit_stddev)
        ax_p.plot(logsum_arr, logsum_arr + photom_offset, label=fit_info, linestyle='--', color='b',
            alpha=0.75)

        ax_p.legend()

        ax_p.set_ylabel("Catalog magnitude ({:s})".format(mag_band_str))
        ax_p.set_xlabel("Uncalibrated magnitude")

        # Set wider axis limits
        ax_p.set_xlim(x_min_w, x_max_w)
        ax_p.set_ylim(y_min_w, y_max_w)

        ax_p.invert_yaxis()
        ax_p.invert_xaxis()

        ax_p.grid()


        ### Plot photometry vs radius ###

        img_diagonal = np.hypot(img_h/2, img_w/2)

        # Plot photometry residuals (including vignetting)
        ax_r.scatter(radius_arr, fit_resid, c='b', alpha=0.75, s=5, zorder=3)

        # Plot a zero line
        ax_r.plot(np.linspace(0, img_diagonal, 10), np.zeros(10), linestyle='dashed', alpha=0.5, \
            color='k')



        # Plot only when no flat is used
        if not config.use_flat:

            #  Plot radius from centre vs. fit residual
            fit_resids_novignetting = catalog_mags - photomLine((np.array(star_intensities), \
                np.array(radius_arr)), photom_offset, 0.0)
            ax_r.scatter(radius_arr, fit_resids_novignetting, s=5, c='r', alpha=0.5, zorder=3)


            px_sum_tmp = 1000
            radius_arr_tmp = np.linspace(0, img_diagonal, 50)

            # Plot vignetting loss curve
            vignetting_loss = 2.5*np.log10(px_sum_tmp) \
                - 2.5*np.log10(correctVignetting(px_sum_tmp, radius_arr_tmp, \
                    platepar.vignetting_coeff))

            ax_r.plot(radius_arr_tmp, vignetting_loss, linestyle='dotted', alpha=0.5, color='k')


        ax_r.grid()

        ax_r.set_ylabel("Fit residuals (mag)")
        ax_r.set_xlabel("Radius from centre (px)")

        ax_r.set_xlim(0, img_diagonal)

        ### ###

        plt.tight_layout()

        plt.savefig(os.path.join(night_dir_path, night_name + '_calib_report_photometry.png'), dpi=150)


        if show_graphs:
            plt.show()

        else:
            plt.clf()
            plt.close()

        ### ###




if __name__ == "__main__":


    import argparse

    import RMS.ConfigReader as cr


    ### PARSE INPUT ARGUMENTS ###

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description=""" Generate the calibration report.
        """)

    arg_parser.add_argument('dir_path', type=str, help="Path to the folder of the night.")

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one.")


    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #############################


    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, cml_args.dir_path)


    generateCalibrationReport(config, cml_args.dir_path, show_graphs=True)
