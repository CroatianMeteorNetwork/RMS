


import time
import datetime
import os
import sys
import numpy as np
import copy

import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit

# Import local modules
import RMS.ConfigReader as cr
import RMS.Formats.BSC as BSC
import RMS.Formats.CALSTARS as CALSTARS
import RMS.Formats.Platepar as Platepar
import RMS.Astrometry.ApplyAstrometry as Astrometry

# Import Cython functions
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from RMS.Astrometry.CyFunctions import angularSeparation, calcBearing, subsetCatalog, starsNNevaluation


def sampleCALSTARS(calstars_list, N):
    """ Randomly sample N number of FF files containing stars, weighted by the number of stars on the image
    """

    weights = np.array([len(star[1]) for star in calstars_list], dtype=np.float64)
    weights = weights/np.sum(weights)

    indices = np.random.choice(len(calstars_list), N, replace=False, p=weights)

    return [calstars_list[i] for i in indices]


def getMiddleTimeFF(ff_name, fps, ret_milliseconds=True, ff_frames=256):
    """ Converts a CAMS format FF file name to datetime object of its recording time. 

    @param ff_name: [str] name of the CAMS format FF file

    @return [datetime obj] moment of the file recording start
    """

    # Extract date and time of the FF file from its name
    ff_name = ff_name.split('_')

    year = int(ff_name[1][0:4])
    month = int(ff_name[1][4:6])
    day = int(ff_name[1][6:8])

    hour = int(ff_name[2][0:2])
    minute = int(ff_name[2][2:4])
    second = int(ff_name[2][4:6])
    millisecond = int(ff_name[3])

    # Convert to datetime
    dt_obj = datetime.datetime(year, month, day, hour, minute, second, millisecond*1000)

    # Time in seconds from the middle of the FF file
    middle_diff = datetime.timedelta(seconds=ff_frames/2.0/fps)

    # Add the difference in time
    dt_obj = dt_obj + middle_diff

    # Unpack datetime to individual values
    year, month, day, hour, minute, second, microsecond = (dt_obj.year, dt_obj.month, dt_obj.day, dt_obj.hour, 
        dt_obj.minute, dt_obj.second, dt_obj.microsecond)

    if ret_milliseconds:
        return (year, month, day, hour, minute, second, microsecond/1000)
    else:
        return (year, month, day, hour, minute, second, microsecond)


def hourDifferenceFF(ff_name, platepar_datetime, fps):
    """ Calculate the difference in hours from the middle time of the FF file and the time in the platepar. 
    """

    # Get the FF file middle time of integration
    ff_time_middle = datetime.datetime(*getMiddleTimeFF(ff_name, fps, ret_milliseconds=False))

    # Get the difference in hours between the platepar and the FF middle time
    time_diff = ff_time_middle - platepar_datetime
    
    return time_diff.total_seconds()/3600.0


def starsXY2RaDec(ff_name, star_list, platepar, fps, UT_corr):
    """ Convert CalibrationStars stars from XY coordinates to celestial equatorial coordinates (Ra, Dec) using
        the given calibration parameters. 

        The calculated Ra and Dec are corrected for time in the FF file.
        """

    # Get the middle time of the FF file
    ff_time_middle = getMiddleTimeFF(ff_name, fps)

    # Split the star list into columns
    y_data, x_data, bg_levels, level_data = np.hsplit(star_list, 4)

    # Generate a dummy list of times
    time_data = np.tile(ff_time_middle, (x_data.shape[0], 1))


    # Convert to Ra,Dec
    JD_data, RA_data, dec_data, magnitude_data = Astrometry.XY2CorrectedRADec(time_data, x_data.T[0], y_data.T[0], level_data.T[0], UT_corr, platepar.lat, 
        platepar.lon, platepar.Ho, platepar.X_res, platepar.Y_res, platepar.RA_d, platepar.dec_d, 
        platepar.rot_param, platepar.F_scale, platepar.w_pix, platepar.mag_0, platepar.mag_lev, 
        platepar.x_poly, platepar.y_poly)

    return RA_data, dec_data


def applyCoordinateCorrection(RA_c, dec_c, correction_separation, correction_angle):
    """ Shift platepar center RA and Dec by provided corrections. 

    Source of equations: http://www.movable-type.co.uk/scripts/latlong.html (May 1, 2016)
    """

    # Convert to radians
    correction_separation, correction_angle = map(np.radians, (correction_separation, correction_angle))

    # Calculate Ra and Dec components from the angular separation and bearing
    ra_diff = np.degrees(np.arctan2(np.sin(correction_angle)*np.sin(correction_separation), np.cos(correction_separation)))
    dec_diff = np.degrees(np.arcsin(np.sin(correction_separation)*np.cos(correction_angle)))

    # Apply correction to RA and Dec
    RA_c = (RA_c + ra_diff) % 360
    dec_c = dec_c + dec_diff

    if dec_c > 90:
        dec_c = 180 - dec_c
        RA_c = (RA_c + 180) % 360

    if dec_c < -90:
        dec_c = -180 - dec_c
        RA_c = (RA_c + 180) % 360

    return RA_c, dec_c


def solutionEvaluation(transformation_parameters, rot_param, platepar, catalog_stars, calstars_list, fps, UT_corr, catalog_extraction_radius, catalog_mag_limit, stars_NN_radius, min_matched_stars, x_poly=None, y_poly=None, matched_stars=None):
    """ Evaluate one Platepar solution.
    """

    # Copy the platepar, so no changes are made on the original one
    platepar = copy.deepcopy(platepar)

    #ra_d, dec_d, rot_param = transformation_parameters
    ra_d, dec_d = transformation_parameters

    # Assign given parameters to platepar
    platepar.RA_d = ra_d
    platepar.dec_d = dec_d
    platepar.rot_param = rot_param

    # Define X and Y distorsion palynomials if they are provided
    if x_poly != None:
        platepar.x_poly = x_poly

    if y_poly != None:
        platepar.y_poly = y_poly

    # Init temporary lists
    evaluation_list = []
    ra_mean_list = []
    ra_std_list = []
    dec_mean_list = []
    dec_std_list = []

    ### MAKE PART THIS PARALLEL BY USING A POOL OF WORKERS!
    # Go thourgh all files containing stars in the given CalibrationStars list
    for star_entry in calstars_list:

        # Unpack star data
        ff_name, star_data = star_entry

        # print ff_name

        # Convert CalibrationStars from XY image coordinates to celestial equatorial coordinates (Ra, Dec)
        RA_data, dec_data = starsXY2RaDec(ff_name, np.array(star_data), platepar, fps, UT_corr)

        # Get stars from the provided catalog if no matched stars were provided
        if matched_stars == None:

            # Calculate the difference in hours from the FF file recording time to the platepar time
            hour_diff = hourDifferenceFF(ff_name, platepar.time, fps)

            # Find the current RA center, by adding the time difference in hours from the platepar to the FF file
            RA_c = (platepar.RA_d + hour_diff*15) % 360
            dec_c = platepar.dec_d

            # Get stars from the catalog around the defined center in a given radius
            extracted_catalog = subsetCatalog(catalog_stars, RA_c, dec_c, catalog_extraction_radius, catalog_mag_limit)

        else:
            # Take the stars from the provided matched stars, if the entry exists
            if ff_name in matched_stars:
                extracted_catalog = matched_stars[ff_name]
            else:
                continue

        # Prepare stars coordinates for NN matching
        stars_coords = np.column_stack((RA_data, dec_data))
        ref_stars_coords = extracted_catalog[:,:2]


        # Calculate the difference between referent and calibration stars
        evaluation, ra_mean, ra_std, dec_mean, dec_std = starsNNevaluation(stars_coords, ref_stars_coords, stars_NN_radius, min_matched_stars)


        # Check if there were enough matched stars
        if ra_mean != None:

            # Add the correction parameters to the global list
            evaluation_list.append(evaluation)

            ra_mean_list.append(ra_mean)
            ra_std_list.append(ra_std)

            dec_mean_list.append(dec_mean)
            dec_std_list.append(dec_std)

    # Convert RA and Dec shift to angular shift
    mean_separation = angularSeparation(0, 0, np.mean(ra_mean_list), np.mean(dec_mean_list))

    # Check if the solution is NaN, if not return the values
    if not np.isnan(mean_separation):

        # Sum the evaluations
        evaluation_sum = np.sum(evaluation_list)

        # Calculate bearing
        mean_bearing = calcBearing(0, 0, np.mean(ra_mean_list), np.mean(dec_mean_list))

        print 'Evaluation: ', evaluation_sum
        print 'Separation and bearing: ', mean_separation, mean_bearing
        print 'RA: ', np.mean(ra_mean_list), np.mean(ra_std_list)
        print 'Dec: ', np.mean(dec_mean_list), np.mean(dec_std_list)
        print 'Rotation:', rot_param

        return evaluation_sum, mean_separation, mean_bearing, rot_param

    else:
        print 'NaN, skipping this iteration'
        return 999, np.nan, np.nan, np.nan


def findCorrectionDirection(platepar, catalog_stars, calstars_list, fps, UT_corr, catalog_extraction_radius, catalog_mag_limit, stars_NN_radius, min_matched_stars, rot_diff):
    """ Find the direction of the shift between referent and calibration stars.
    """

    best_evaluation = []
    best_separations = []
    best_angles = []
    best_rotations = []

    # Check for +/- rot_diff degree rotations
    rot_param_min = platepar.rot_param - rot_diff
    rot_param_max = platepar.rot_param + rot_diff

    for rot_param in np.linspace(rot_param_min, rot_param_max, 9)%360:

        evaluation_sum, mean_separation, mean_bearing, rot_param = solutionEvaluation((platepar.RA_d, platepar.dec_d), rot_param, platepar, catalog_stars, calstars_list, fps, UT_corr, catalog_extraction_radius, catalog_mag_limit, stars_NN_radius, min_matched_stars)

        # Check if the solution is NaN, if not add it to the list
        if not np.isnan(mean_separation):
            best_evaluation.append(evaluation_sum)
            best_separations.append(mean_separation)
            best_angles.append(mean_bearing)
            best_rotations.append(rot_param)
            

        # Break if rot_diff is 0
        if rot_diff == 0:
            break

    # Check if there are any good evaluations
    if best_evaluation:
        
        # Choose the lowest separation
        min_ind = np.argmin(best_evaluation)

    else:
        return None, None, None, None


    print 'Best values: '
    print 'EVAL, SEP, ANGLE, ROT'
    print best_evaluation[min_ind], best_separations[min_ind], best_angles[min_ind], best_rotations[min_ind]

    return best_evaluation[min_ind], best_separations[min_ind], best_angles[min_ind], best_rotations[min_ind]



def getMatchedStars(platepar, calstars_list, fps, UT_corr, catalog_stars, catalog_extraction_radius, catalog_mag_limit, stars_NN_radius, min_matched_stars):
    """ Get the matching catalogs stars with ones in each FF file from CALSTARS.
    """

    # Init the dectionary which contains matched stars from the catalog on each FF file
    catalog_matched_stars = {}

    # Init the list of all matching stars (catalog + FF file values)
    matched_list = []

    for calstars_entry in calstars_list:

        ff_name, star_data = calstars_entry

        # Convert CalibrationStars from XY image coordinates to celestial equatorial coordinates (Ra, Dec)
        RA_data, dec_data = starsXY2RaDec(ff_name, np.array(star_data), platepar, fps, UT_corr)

        # Extract levels data
        levels_data = np.array(star_data)[:,3]

        # Calculate the difference in hours from the FF file recording time to the platepar time
        hour_diff = hourDifferenceFF(ff_name, platepar.time, fps)

        # Find the current RA center, by adding the time difference in hours from the platepar to the FF file
        RA_c = (platepar.RA_d + hour_diff*15) % 360
        dec_c = platepar.dec_d

        #t1 = time.clock()
        # Get stars from the catalog around the defined center in a given radius
        extracted_catalog = subsetCatalog(catalog_stars, RA_c, dec_c, catalog_extraction_radius, catalog_mag_limit)
        #print 'Cat extract: ', time.clock() - t1

        # Prepare stars coordinates for NN matching
        stars_coords = np.column_stack((RA_data, dec_data))
        ref_stars_coords = extracted_catalog[:,:2]


        # Get the indices of the matching stars
        catalog_matched_indices, image_matched_indices = starsNNevaluation(stars_coords, ref_stars_coords, 
            stars_NN_radius, min_matched_stars, ret_indices=1)

        # Continue if no stars were matched
        if catalog_matched_indices == None:
            continue

        # Add matched stars to dictionary
        catalog_matched_stars[ff_name] = extracted_catalog[catalog_matched_indices,:]

        # Extract columns from the matched catalog stars
        RA_cat, dec_cat, mag_cat = np.hsplit(extracted_catalog[catalog_matched_indices,:], 3)

        # Add the matching entries to the list
        matched_list.append([RA_cat, dec_cat, mag_cat, RA_data[image_matched_indices], 
            dec_data[image_matched_indices], levels_data[image_matched_indices]])


    return catalog_matched_stars, matched_list


def photometryFit(matched_list):
    """ Perform the photometry procedure on the given matched star data.
    """

    def _stellarMagnitude(x, C2, m2):
        """ Equation that relates instrumental brightness level (intensity) to stellar magnitude. 

        @param x: [float] input instrumental level to be converted to stellar magnitude
        @param C2: [float] fitted level parameter
        @param m2: [float] fitted magnitude parameter
        """

        return -2.5*np.log10(x) + 2.5*np.log10(C2) + m2


    mag_list = []
    level_list = []

    # TESTING ###########
    for i, ff_stars in enumerate(matched_list):
        print '------------------'
        print 'Image: ', i+1

        ra_cat, dec_cat, mag_cat, ra_img, dec_img, level_img = ff_stars

        for j in range(len(ra_cat)):
            # print ra_cat[j][0], dec_cat[j][0], mag_cat[j][0], ra_img[j], dec_img[j], level_img[j]
            mag_list.append(mag_cat[j][0])
            level_list.append(level_img[j])

    # Convert brightness data to numpy arrays
    magnitude_data = np.array(mag_list)
    levels_data = np.array(level_list)

    # Fit the magnitude relation curve
    popt, pcov = curve_fit(_stellarMagnitude, levels_data, magnitude_data)

    print popt

    x_level_test = np.linspace(50, 10000, 100)

    # Plot the fitted curve
    plt.plot(x_level_test, _stellarMagnitude(x_level_test, *popt))

    # Plot levels vs magnitudes
    plt.scatter(levels_data, magnitude_data)
    plt.gca().set_xscale('log')

    plt.xlabel('Level')
    plt.ylabel('Stellar magnitude')
    plt.title('Magnitude fit')

    plt.show()

    return popt




def astrometryCheckFit(ff_directory, calstars_name, UT_corr, config):
    """ Checks the calibration parameters for the given night and recalibrates the calibration parameters if needed. 
    """

    # Load several parameters from config to local variables
    calstars_files_N = config.calstars_files_N
    stars_NN_radius = config.stars_NN_radius

    # Load CALSTARS
    calstars_list = CALSTARS.readCALSTARS(ff_directory, calstars_name)

    # Return False if there is no CALSTARS file
    if not calstars_list:
        return False

    # Sample N calstars files if sampling is given
    if calstars_files_N > 0:
        calstars_list = sampleCALSTARS(calstars_list, calstars_files_N)

    # Check if there is a minimum number of stars in calstars and increase the number of files until there is
    n_stars_prev = 0
    while True:

        # Calculate the number of stars in calstars_list
        n_stars = 0
        for star_list in calstars_list:
            n_stars_on_image = len(star_list[1])
            if n_stars_on_image >= config.min_matched_stars:
                n_stars += n_stars_on_image

        print n_stars

        # Declare the procedure unsuccessfull if the minimum number of stars cannot be reached
        if n_stars == n_stars_prev:
            print 'Not enough stars for calibration!'
            sys.exit()
            # return False

        # Check if there are enough stars
        if n_stars < config.calstars_min_stars:
            calstars_files_N = calstars_files_N + 10
            calstars_list = sampleCALSTARS(CALSTARS.readCALSTARS(ff_directory, calstars_name), 
                calstars_files_N)

            n_stars_prev = n_stars
        
        else:
            break

    # Import stars from the BSC catalog
    catalog_stars = BSC.readBSC(config.star_catalog_path, config.star_catalog_file, 
        lim_mag=config.catalog_mag_limit)

    # Check if the BSC exists
    if not catalog_stars.any():
        return False

    # Import platepar
    platepar_path = os.path.join(ff_directory, config.platepar_name)
    platepar = Platepar.PlateparCMN()
    platepar.read(platepar_path)

    # Check if the platepar exists
    if not platepar:
        return False


    total_time = time.clock()

    # Start initial iteration to find the rough coordinate offset and rotation
    previous_evaluation = 999
    iter_no = 0
    while 1:

        # On first iteration just run the Ra/dec aligment, without rotation parameter changes
        if iter_no == 0:
            rot_param_range_temp = 0
        elif iter_no == 2:
            rot_param_range_temp = config.rotation_param_range

        # Get the coordinate shift
        evaluation, separation_corr, angle_corr, platepar.rot_param = findCorrectionDirection(platepar, catalog_stars, calstars_list, config.fps, UT_corr, config.catalog_extraction_radius, config.catalog_mag_limit, stars_NN_radius, config.min_matched_stars, rot_param_range_temp)

        # Check if no good solutions were found
        if evaluation == None:
            
            print 'No good solutions were found, the astrometry procedure failed!'     
            return False   


        # Check if the separation is better than the last one, only then apply the correction
        if evaluation < previous_evaluation:
            
            # Apply the correction to platepar
            platepar.RA_d, platepar.dec_d = applyCoordinateCorrection(platepar.RA_d, platepar.dec_d, separation_corr, angle_corr)

            # Shrink the star search radius
            stars_NN_radius = stars_NN_radius/np.sqrt(2)

            print 'NEW CENTER'
            print platepar.RA_d, platepar.dec_d
            print platepar.rot_param
            print '--------'

        else:
            evaluation = previous_evaluation*1.2
            print 'No solution is acceptable!'

            # Widen the parameter search range
            stars_NN_radius *= 2
            rot_param_range_temp *= 2

        # End the estimation if the desired precision is reached
        if evaluation < config.min_estimation_value:
            break

        # End the estimation if the evaluation does not change
        if (iter_no > 5) and (abs(previous_evaluation - evaluation) < 0.01):
            
            return False



        # Stop after too many iterations
        if iter_no > config.max_initial_iterations:

            return False

        # Shrink the search radius of the rotational parameter
        if rot_param_range_temp:
            rot_param_range_temp = rot_param_range_temp/np.sqrt(2)

        previous_evaluation = evaluation
        iter_no += 1

    # Set a small search radius
    # config.refinement_star_NN_radius = 0.125 #deg

    # Get matched stars
    catalog_matched_stars, matched_list = getMatchedStars(platepar, calstars_list, config.fps, UT_corr, catalog_stars, config.catalog_extraction_radius, config.catalog_mag_limit, config.refinement_star_NN_radius, config.min_matched_stars)
        

    # ### Start RA, Dec refinement

    # # Define initial RA and Dec
    # initial_parameters = np.array([platepar.RA_d, platepar.dec_d])

    # extra_args = [platepar.rot_param, platepar, catalog_stars, calstars_list, config.fps, UT_corr, config.catalog_extraction_radius, config.catalog_mag_limit, config.refinement_star_NN_radius, config.min_matched_stars]

    # # Define an adapted function for refining
    # refineSoluton = lambda params: solutionEvaluation(params, *extra_args, matched_stars=catalog_matched_stars)[0]

    # # Run SIMPLEX parameter refining
    # res = minimize(refineSoluton, initial_parameters, method='Nelder-Mead', options={'xtol': 1e-4, 'disp': True})

    # print res.x, platepar.rot_param

    # # Update platepar with simplex results for Ra and Dec
    # platepar.RA_d, platepar.dec_d = res.x

    ### Start X distorsion parameter refinement
    initial_parameters = copy.deepcopy(platepar.x_poly)

    # Define an adapted function for refining
    refineSoluton = lambda params: solutionEvaluation((platepar.RA_d, platepar.dec_d), platepar.rot_param, platepar, catalog_stars, calstars_list, config.fps, UT_corr, config.catalog_extraction_radius, config.catalog_mag_limit, config.refinement_star_NN_radius, config.min_matched_stars, x_poly=params, matched_stars=catalog_matched_stars)[0]

    # Run SIMPLEX parameter refining
    res = minimize(refineSoluton, initial_parameters, method='Nelder-Mead', options={'xtol': 1e-4, 'disp': True})

    print res.x

    # Update platepar with simplex results for Ra and Dec
    platepar.x_poly = res.x

    ### Start Y distorsion parameter refinement
    initial_parameters = copy.deepcopy(platepar.y_poly)

    # Define an adapted function for refining
    refineSoluton = lambda params: solutionEvaluation((platepar.RA_d, platepar.dec_d), platepar.rot_param, platepar, catalog_stars, calstars_list, config.fps, UT_corr, config.catalog_extraction_radius, config.catalog_mag_limit, config.refinement_star_NN_radius, config.min_matched_stars, y_poly=params, matched_stars=catalog_matched_stars)[0]

    # Run SIMPLEX parameter refining
    res = minimize(refineSoluton, initial_parameters, method='Nelder-Mead', options={'xtol': 1e-4, 'disp': True})

    print res.x

    # Update platepar with simplex results for Ra and Dec
    platepar.y_poly = res.x


    ## Do the photometry calibration
    # Get matched stars
    catalog_matched_stars, matched_list = getMatchedStars(platepar, calstars_list, config.fps, UT_corr, catalog_stars, config.catalog_extraction_radius, config.catalog_mag_limit, config.refinement_star_NN_radius, config.min_matched_stars)
    

    # Fit the magnitude curve
    C2, m2 = photometryFit(matched_list)


    print 'FINAL RESULTS:'
    print platepar.RA_d, platepar.dec_d
    print platepar.rot_param
    print platepar.x_poly
    print platepar.y_poly

    print 'Photometry parameters: ', C2, m2

    # Calculate the number of used stars
    n_stars = 0
    for star_list in catalog_matched_stars.itervalues():
        n_stars += len(star_list)

    print 'Number of stars used: ', n_stars

    print 'Total time for completion: ', time.clock() - total_time

    return platepar


if __name__ == '__main__':

    #ff_directory = '/home/anonymus/CAMS/CapturedFiles/RVN2016_05_06_18_50_14'
    #ff_directory = '/home/anonymus/CAMS/CapturedFiles/OSE_2016_04_17_18_05_02/'
    ff_directory = '/home/anonymus/CAMS/CapturedFiles/VIB_2016_04_19_18_27_08/'
    #ff_directory = '/home/anonymus/CAMS/CapturedFiles/OSE_2016_05_01_18_24_35/'


    #calstars_name = 'CALSTARS0497_2016_05_05_18_48_52.txt'
    #calstars_name = 'CALSTARS0494OSE_2016_04_17_18_05_02.txt'
    calstars_name = 'CALSTARS0453VIB_2016_04_19_18_27_08.txt'
    #calstars_name = 'CALSTARS0494OSE_2016_05_01_18_24_35.txt'

    UT_corr = 0.0

    # Load config file
    config = cr.parse(".config")

    # Run the astrometry check procedure
    platepar = astrometryCheckFit(ff_directory, calstars_name, UT_corr, config)


    