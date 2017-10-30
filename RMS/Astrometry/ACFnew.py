""" Automatic refining of astrometry calibration. The initial astrometric calibration is needed, which will be 
    refined by using all stars from a given night.
"""

from __future__ import print_function, division, absolute_import

import os
import sys

import numpy as np

import RMS.ConfigReader as cr
from RMS.Formats import Platepar
from RMS.Formats import CALSTARS
from RMS.Formats import BSC
from RMS.Formats import FFfile
from RMS.Astrometry.Conversions import date2JD
from RMS.Astrometry.ApplyAstrometry import raDecToCorrectedXY




def matchStarsResiduals(platepar, catalog_stars, star_dict, max_radius=10):
    """
    
    Keyword arguments:
        max_radius: [float] Maximum radius for star matching (pixels).
    """


    import matplotlib.pyplot as plt


    ra_catalog, dec_catalog, mag_catalog = catalog_stars.T


    # Dictionary containing the mactched stars, the keys are JDs of every image
    matched_stars = {}


    # Go through every FF image and its stars
    for jd in star_dict:

        # Extract stars for the given Julian date
        stars_list = star_dict[jd]
        stars_list = np.array(stars_list)

        # Convert all catalog stars to image coordinates
        x_array, y_array = raDecToCorrectedXY(ra_catalog, dec_catalog, jd, platepar.lat, platepar.lon, \
            platepar.X_res, platepar.Y_res, platepar.RA_d, platepar.dec_d, platepar.JD, \
            platepar.pos_angle_ref, platepar.F_scale, platepar.x_poly, platepar.y_poly)


        # Take only those stars which are within the FOV
        x_indices = np.argwhere((x_array >= 0) & (x_array < platepar.X_res))
        y_indices = np.argwhere((y_array >= 0) & (y_array < platepar.Y_res))
        cat_good_indices = np.intersect1d(x_indices, y_indices)

        # x_array = x_array[good_indices]
        # y_array = y_array[good_indices]


        # # Plot image stars
        # im_y, im_x, _, _ = stars_list.T
        # plt.scatter(im_y, im_x, c='r', s=5)

        # # Plot catalog stars
        # plt.scatter(y_array[cat_good_indices], x_array[cat_good_indices], facecolors='none', edgecolor='g')

        # plt.show()
        
        
        matched_indices = []

        # Match image and catalog stars
        # Go through all image stars
        for i, entry in enumerate(stars_list):

            # Extract image star data
            im_star_y, im_star_x, _, level = entry

            min_dist = np.inf
            cat_match_indx = None

            # Check for the best match among catalog stars
            for k in cat_good_indices:

                cat_x = x_array[k]
                cat_y = y_array[k]


                # Calculate the distance between stars
                dist = np.sqrt((im_star_x - cat_x)**2 + (im_star_y - cat_y)**2)

                if (dist < min_dist):
                    min_dist = dist
                    cat_match_indx = k


            # Take the best matched star if the distance was within the maximum radius
            if min_dist < max_radius:
                matched_indices.append([i, cat_match_indx, min_dist])


        # Skip this image is no stars were matched
        if len(matched_indices) == 0:
            continue

        matched_indices = np.array(matched_indices)
        matched_img_inds, matched_cat_inds, dist_list = matched_indices.T

        # Extract data from matched stars
        matched_img_stars = stars_list[matched_img_inds.astype(np.int)]
        matched_cat_stars = catalog_stars[matched_cat_inds.astype(np.int)]

        # Put the matched stars to a dictionary
        matched_stars[jd] = [matched_img_stars, matched_cat_stars, dist_list]


        # # Plot matched stars
        # im_y, im_x, _, _ = matched_img_stars.T
        # cat_y = y_array[matched_cat_inds.astype(np.int)]
        # cat_x = x_array[matched_cat_inds.astype(np.int)]

        # plt.scatter(im_y, im_x, c='r', s=5)
        # plt.scatter(cat_y, cat_x, facecolors='none', edgecolor='g')

        # plt.show()




    # Extract all distances
    global_dist_list = []
    level_list = []
    mag_list = []
    for jd in matched_stars:
        matched_img_stars, matched_cat_stars, dist_list = matched_stars[jd]
        
        global_dist_list += dist_list.tolist()

        # TEST
        level_list += matched_img_stars[:, 3].tolist()
        mag_list += matched_cat_stars[:, 2].tolist()



    # # Plot levels vs. magnitudes
    # plt.scatter(mag_list, np.log10(level_list))
    # plt.xlabel('Magnitude')
    # plt.ylabel('Log10 level')
    # plt.show()


    # Number of matched stars
    n_matched = len(global_dist_list)

    # Calculate the average distance
    avg_dist = np.mean(global_dist_list)

    cost = avg_dist*(1.0/np.sqrt(n_matched + 1))

    print('Nmatched', n_matched)
    print('Avg dist', avg_dist)
    print('Cost:', cost)


    return cost







def autoCheckFit(config, platepar, calstars_list):
    """ Attempts to refine the astrometry fit with the given stars and and initial astrometry parameters.

    Arguments:
        config: [Config structure]
        platepar: [Platepar structure] Initial astrometry parameters.
        calstars_list: [list] A list containing stars extracted from FF files. See RMS.Formats.CALSTARS for
            more details.
    
    """


    # Convert the list to a dictionary
    calstars = {ff_file: star_data for ff_file, star_data in calstars_list}

    # Load catalog stars
    catalog_stars = BSC.readBSC(config.star_catalog_path, config.star_catalog_file, \
        lim_mag=config.catalog_mag_limit)


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


    # Match the stars and calculate the residuals
    res = matchStarsResiduals(platepar, catalog_stars, star_dict, max_radius=10)




        

    



if __name__ == "__main__":


    if len(sys.argv) < 2:
        print('Usage: python -m RMS.Astrometry.AstrometryCheckFit /path/to/FF/dir/')
        sys.exit()

    # Night directory
    dir_path = sys.argv[1].replace('"', '')


    # Check if the given directory is OK
    if not os.path.exists(dir_path):
        print('No such directory:', dir_path)
        sys.exit()


    # Load the configuration file
    config = cr.parse(".config")


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




    # Run the automatic astrometry fit
    autoCheckFit(config, platepar, calstars_list)