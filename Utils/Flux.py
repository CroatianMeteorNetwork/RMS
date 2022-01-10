""" Compute single-station meteor shower flux. """

import os
import sys
import glob
import copy
import datetime
import json
import collections

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP
from RMS.Astrometry.ApplyRecalibrate import applyRecalibrate
from RMS.Astrometry.Conversions import areaGeoPolygon, jd2Date, datetime2JD, J2000_JD, raDec2AltAz
from RMS.ExtractStars import extractStarsAndSave
import RMS.Formats.CALSTARS as CALSTARS
from RMS.Formats import FFfile
from RMS.Formats.FTPdetectinfo import readFTPdetectinfo
from RMS.Formats import Platepar
from RMS.Math import angularSeparation
from RMS.Routines.FOVArea import xyHt2Geo, fovArea
from RMS.Routines.MaskImage import loadMask, MaskStructure
from RMS.Routines.SolarLongitude import jd2SolLonSteyaert
from Utils.ShowerAssociation import showerAssociation, heightModel


def generateColAreaJSONFileName(station_code, side_points, ht_min, ht_max, dht, elev_limit):
    """ Generate a file name for the collection area JSON file. """

    file_name = "flux_col_areas_{:s}_sp-{:d}_htmin-{:.1f}_htmax-{:.1f}_dht-{:.1f}_elemin-{:.1f}.json".format(\
        station_code, side_points, ht_min, ht_max, dht, elev_limit)

    return file_name


def saveRawCollectionAreas(dir_path, file_name, col_areas_ht):
    """ Save the raw collection area calculations so they don't have to be regenerated every time. """

    file_path = os.path.join(dir_path, file_name)

    with open(file_path, 'w') as f:

        # Convert tuple keys (x_mid, y_mid) to str keys
        col_areas_ht_strkeys = {}
        for key in col_areas_ht:
            col_areas_ht_strkeys[key] = {}

            for tuple_key in col_areas_ht[key]:

                str_key = "{:.2f}, {:.2f}".format(*tuple_key)

                col_areas_ht_strkeys[key][str_key] = col_areas_ht[key][tuple_key]


        # Add an explanation what each entry means
        col_areas_ht_strkeys[-1] = {"height (m)": {"x (px), y (px) of pixel block": \
            ["area (m^2)", "azimuth +E of due N (deg)", "elevation (deg)", "sensitivity", "range (m)"]}}

        # Convert collection areas to JSON
        out_str = json.dumps(col_areas_ht_strkeys, indent=4, sort_keys=True)

        # Save to disk
        f.write(out_str)
    

def loadRawCollectionAreas(dir_path, file_name):
    """ Read raw collection areas from disk. """

    file_path = os.path.join(dir_path, file_name)


    # Load the JSON file
    with open(file_path) as f:
        
        data = " ".join(f.readlines())

        col_areas_ht_strkeys = json.loads(data)

        # Convert tuple keys (x_mid, y_mid) to str keys
        col_areas_ht = collections.OrderedDict()
        for key in col_areas_ht_strkeys:

            # Skip heights below 0 (the info key)
            if float(key) < 0:
                continue

            col_areas_ht[float(key)] = collections.OrderedDict()

            for str_key in col_areas_ht_strkeys[key]:

                # Convert the string "x_mid, y_mid" to tuple of floats (x_mid, y_mid)
                tuple_key = tuple(map(float, str_key.split(", ")))

                col_areas_ht[float(key)][tuple_key] = col_areas_ht_strkeys[key][str_key]


        return col_areas_ht




class FluxConfig(object):
    def __init__(self):
        """ Container for flux calculations. """

        # How many points to use to evaluate the FOV on seach side of the image. Normalized to the longest 
        #   side.
        self.side_points = 20

        # Minimum height (km).
        self.ht_min = 60

        # Maximum height (km).
        self.ht_max = 130

        # Height sampling delta (km).
        self.dht = 2

        # Limit of meteor's elevation above horizon (deg). 25 degrees by default.
        self.elev_limit = 25

        # Minimum radiant elevation in the time bin (deg). 25 degreees by default
        self.rad_elev_limit = 25

        # Minimum distance of the end of the meteor to the radiant (deg)
        self.rad_dist_min = 15

        # Subdivide the time bin into the given number of subbins
        self.sub_time_bins = 2

        # Minimum number of meteors in the time bin
        self.meteros_min = 3



def collectingArea(platepar, mask=None, side_points=20, ht_min=60, ht_max=130, dht=2, elev_limit=10):
    """ Compute the collecting area for the range of given heights.
    
    Arguments:
        platepar: [Platepar object]

    Keyword arguments:
        mask: [Mask object] Mask object, None by default.
        side_points: [int] How many points to use to evaluate the FOV on seach side of the image. Normalized
            to the longest side.
        ht_min: [float] Minimum height (km).
        ht_max: [float] Maximum height (km).
        dht: [float] Height delta (km).
        elev_limit: [float] Limit of elevation above horizon (deg). 10 degrees by default.

    Return:
        col_areas_ht: [dict] A dictionary where the keys are heights of area evaluation, and values are
            segment dictionaries. Segment dictionaries have keys which are tuples of (x, y) coordinates of
            segment midpoints, and values are segment collection areas corrected for sensor effects.

    """


    # If the mask is not given, make a dummy mask with all white pixels
    if mask is None:
        mask = MaskStructure(255 + np.zeros((platepar.Y_res, platepar.X_res), dtype=np.uint8))


    # Compute the number of samples for every image axis
    longer_side_points = side_points
    shorter_side_points = int(np.ceil(side_points*platepar.Y_res/platepar.X_res))

    # Compute pixel delta for every side
    longer_dpx = int(platepar.X_res//longer_side_points)
    shorter_dpx = int(platepar.Y_res//shorter_side_points)


    # Distionary of collection areas per height
    col_areas_ht = collections.OrderedDict()

    # Estimate the collection area for a given range of heights
    for ht in np.arange(ht_min, ht_max + dht, dht):

        # Convert the height to meters
        ht = 1000*ht

        print(ht/1000, "km")

        total_area = 0

        # Dictionary of computed sensor-corrected collection areas where X and Y are keys
        col_areas_xy = collections.OrderedDict()

        # Sample the image
        for x0 in np.linspace(0, platepar.X_res, longer_side_points, dtype=np.int, endpoint=False):
            for y0 in np.linspace(0, platepar.Y_res, shorter_side_points, dtype=np.int, endpoint=False):
                
                # Compute lower right corners of the segment
                xe = x0 + longer_dpx
                ye = y0 + shorter_dpx

                # Compute geo coordinates of the image corners (if the corner is below the elevation limit,
                #   the *_elev value will be -1)
                _, ul_lat, ul_lon, ul_ht = xyHt2Geo(platepar, x0, y0, ht, indicate_limit=True, \
                    elev_limit=elev_limit)
                _, ll_lat, ll_lon, ll_ht = xyHt2Geo(platepar, x0, ye, ht, indicate_limit=True, \
                    elev_limit=elev_limit)
                _, lr_lat, lr_lon, lr_ht = xyHt2Geo(platepar, xe, ye, ht, indicate_limit=True, \
                    elev_limit=elev_limit)
                _, ur_lat, ur_lon, ur_ht = xyHt2Geo(platepar, xe, y0, ht, indicate_limit=True, \
                    elev_limit=elev_limit)


                # Skip the block if all corners are hitting the lower apparent elevation limit
                if np.all([ul_ht < 0, ll_ht < 0, lr_ht < 0, ur_ht < 0]):
                    continue


                # Make a polygon (clockwise direction)
                lats = [ul_lat, ll_lat, lr_lat, ur_lat]
                lons = [ul_lon, ll_lon, lr_lon, ur_lon]

                # Compute the area of the polygon
                area = areaGeoPolygon(lats, lons, ht)


                ### Apply sensitivity corrections to the area ###

                # Compute ratio of masked portion of the segment
                mask_segment = mask.img[y0:ye, x0:xe]
                unmasked_ratio = 1 - np.count_nonzero(~mask_segment)/mask_segment.size


                ## Compute the pointing direction and the vignetting and extinction loss for the mean location

                x_mean = (x0 + xe)/2
                y_mean = (y0 + ye)/2

                # Use a test pixel sum
                test_px_sum = 400

                # Compute the pointing direction and magnitude corrected for vignetting and extinction
                _, ra, dec, mag = xyToRaDecPP([jd2Date(J2000_JD.days)], [x_mean], [y_mean], [test_px_sum], \
                    platepar)
                azim, elev = raDec2AltAz(ra[0], dec[0], J2000_JD.days, platepar.lat, platepar.lon)

                # Compute the pixel sum back assuming no corrections
                rev_level = 10**((mag[0] - platepar.mag_lev)/(-2.5))
                
                # Compute the sensitivty loss due to vignetting and extinction
                sensitivity_ratio = test_px_sum/rev_level

                # print(np.abs(np.hypot(x_mean - platepar.X_res/2, y_mean - platepar.Y_res/2)), sensitivity_ratio, mag[0])

                ##


                # Compute the range correction (w.r.t 100 km) to the mean point
                r, _, _, _ = xyHt2Geo(platepar, x_mean, y_mean, ht, indicate_limit=True, \
                    elev_limit=elev_limit)


                # Correct the area for the masked portion
                area *= unmasked_ratio

                ### ###


                # Store the raw masked segment collection area, sensivitiy, and the range
                col_areas_xy[(x_mean, y_mean)] = [area, azim, elev, sensitivity_ratio, r]


                total_area += area


        # Store segments to the height dictionary (save a copy so it doesn't get overwritten)
        col_areas_ht[float(ht)] = dict(col_areas_xy)

        print("SUM:", total_area/1e6, "km^2")


        # Compare to total area computed from the whole area
        side_points_list = fovArea(platepar, mask=mask, area_ht=ht, side_points=side_points, \
            elev_limit=elev_limit)
        lats = []
        lons = []
        for side in side_points_list:
            for entry in side:
                lats.append(entry[0])
                lons.append(entry[1])
                
        print("DIR:", areaGeoPolygon(lats, lons, ht)/1e6)



    return col_areas_ht

    


def sensorCharacterization(config, dir_path):
    """ Characterize the standard deviation of the background and the FWHM of stars on every image. """

    
    # Find the CALSTARS file in the given folder that has FWHM information
    found_good_calstars = False
    for cal_file in os.listdir(dir_path):
        if ('CALSTARS' in cal_file) and ('.txt' in cal_file) and (not found_good_calstars):

            # Load the calstars file
            calstars_list = CALSTARS.readCALSTARS(dir_path, cal_file)

            if len(calstars_list) > 0:

                # Check that at least one image has good FWHM measurements
                for ff_name, star_data in calstars_list:

                    if len(star_data) > 1:

                        star_data = np.array(star_data)

                        # Check if the calstars file have FWHM information
                        fwhm = star_data[:, 4]

                        # Check that FWHM values have been computed well
                        if np.all(fwhm > 1):

                            found_good_calstars = True

                            print('CALSTARS file: ' + cal_file + ' loaded!')

                            break


    # If the FWHM information is not present, run the star extraction
    if not found_good_calstars:

        print()
        print("No FWHM information found in existing CALSTARS files!")
        print()
        print("Rerunning star detection...")
        print()

        found_good_calstars = False

        # Run star extraction again, and now FWHM will be computed
        calstars_list = extractStarsAndSave(config, dir_path)

        if len(calstars_list) == 0:
            found_good_calstars = False


        # Check for a minimum of detected stars
        for ff_name, star_data in calstars_list:
            if len(star_data) >= config.ff_min_stars:
                found_good_calstars = True
                break
            
    # If no good calstars exist, stop computing the flux
    if not found_good_calstars:

        print("No stars were detected in the data!")

        return False



    # Dictionary which holds information about FWHM and standard deviation of the image background
    sensor_data = {}

    # Compute median FWHM per FF file
    for ff_name, star_data in calstars_list:

        # Check that the FF file exists in the data directory
        if ff_name not in os.listdir(dir_path):
            continue


        star_data = np.array(star_data)

        # Compute the median star FWHM
        fwhm_median = np.median(star_data[:, 4])


        # Load the FF file and compute the standard deviation of the background
        ff = FFfile.read(dir_path, ff_name)

        # Compute the median stddev of the background
        stddev_median = np.median(ff.stdpixel)


        # Store the values to the dictionary
        sensor_data[ff_name] = [fwhm_median, stddev_median]


        print("{:s}, {:5.2f}, {:5.2f}".format(ff_name, fwhm_median, stddev_median))


    return sensor_data



def stellarLMModel(p0):
    """ Empirical model of the stellar limiting magnitude given the photometric zero point. 
        Derived using various RMS cameras.
    
    Arguments:
        p0: [float] Photometric zeropoint.

    Return:
        lm_s: [float] Stellar limiting magnitude.
    """

    #lm_s = 0.639*p0 - 0.858 # old with 3 points
    lm_s = 0.832*p0 - 2.585 # new on Nov 4, with 17 points

    return lm_s




def computeFlux(config, dir_path, ftpdetectinfo_path, shower_code, dt_beg, dt_end, timebin, mass_index, \
    timebin_intdt=0.25, ht_std_percent=5.0, mask=None, show_plots=True, confidence_interval=0.95):
    """ Compute flux using measurements in the given FTPdetectinfo file. 
    
    Arguments:
        config: [Config instance]
        dir_path: [str] Path to the working directory.
        ftpdetectinfo_path: [str] Path to a FTPdetectinfo file.
        shower_code: [str] IAU shower code (e.g. ETA, PER, SDA).
        dt_beg: [Datetime] Datetime object of the observation beginning.
        dt_end: [Datetime] Datetime object of the observation end.
        timebin: [float] Time bin in hours.
        mass_index: [float] Cumulative mass index of the shower.

    Keyword arguments:
        timebin_intdt: [float] Time step for computing the integrated collection area in hours. 15 minutes by
            default. If smaller than that, only one collection are will be computed.
        ht_std_percent: [float] Meteor height standard deviation in percent.
        mask: [Mask object] Mask object, None by default.
        show_plots: [bool] Show flux plots. True by default.
        confidence_interval: [float] Confidence interval for error estimation using Poisson statistics.
            0.95 by default (95% CI).

    Return:
        [tuple] sol_data, flux_lm_6_5_data
            - sol_data: [list] Array of solar longitudes (in degrees) of time bins.
            - flux_lm6_5_data: [list] Array of meteoroid flux at the limiting magnitude of +6.5 in 
                meteors/1000km^2/h.
            - flux_lm_6_5_ci_lower_data: [list] Flux, lower bound confidence interval.
            - flux_lm_6_5_ci_upper_data: [list] Flux, upper bound confidence interval.
            - meteor_num_data: [list] Number of meteors in every bin.
    """


    # Get a list of files in the night folder
    file_list = sorted(os.listdir(dir_path))



    # Find and load the platepar file
    if config.platepar_name in file_list:

        # Load the platepar
        platepar = Platepar.Platepar()
        platepar.read(os.path.join(dir_path, config.platepar_name), use_flat=config.use_flat)

    else:
        print("Cannot find the platepar file in the night directory: ", config.platepar_name)
        return None




    # # Load FTPdetectinfos
    # meteor_data = []
    # for ftpdetectinfo_path in ftpdetectinfo_list:

    #     if not os.path.isfile(ftpdetectinfo_path):
    #         print('No such file:', ftpdetectinfo_path)
    #         continue

    #     meteor_data += readFTPdetectinfo(*os.path.split(ftpdetectinfo_path))


    # Load meteor data from the FTPdetectinfo file
    meteor_data = readFTPdetectinfo(*os.path.split(ftpdetectinfo_path))

    if not len(meteor_data):
        print("No meteors in the FTPdetectinfo file!")
        return None




    # Find and load recalibrated platepars
    if config.platepars_recalibrated_name in file_list:
        with open(os.path.join(dir_path, config.platepars_recalibrated_name)) as f:
            recalibrated_platepars_dict = json.load(f)

            print("Recalibrated platepars loaded!")

    # If the file is not available, apply the recalibration procedure
    else:

        recalibrated_platepars_dict = applyRecalibrate(ftpdetectinfo_path, config)

        print("Recalibrated platepar file not available!")
        print("Recalibrating...")


    # Convert the dictionary of recalibrated platepars to a dictionary of Platepar objects
    recalibrated_platepars = {}
    for ff_name in recalibrated_platepars_dict:
        pp = Platepar.Platepar()
        pp.loadFromDict(recalibrated_platepars_dict[ff_name], use_flat=config.use_flat)

        recalibrated_platepars[ff_name] = pp


    # Compute nighly mean of the photometric zero point
    mag_lev_nightly_mean = np.mean([recalibrated_platepars[ff_name].mag_lev \
                                        for ff_name in recalibrated_platepars])




    # Locate and load the mask file
    if config.mask_file in file_list:
        mask_path = os.path.join(dir_path, config.mask_file)
        mask = loadMask(mask_path)
        print("Using mask:", mask_path)

    else:
        print("No mask used!")
        mask = None



    # Compute the population index using the classical equation
    population_index = 10**((mass_index - 1)/2.5) # Found to be more consistent when comparing fluxes
    #population_index = 10**((mass_index - 1)/2.3) # TEST !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1


    ### SENSOR CHARACTERIZATION ###
    # Computes FWHM of stars and noise profile of the sensor
    
    # File which stores the sensor characterization profile
    sensor_characterization_file = "flux_sensor_characterization.json"
    sensor_characterization_path = os.path.join(dir_path, sensor_characterization_file)

    # Load sensor characterization file if present, so the procedure can be skipped
    if os.path.isfile(sensor_characterization_path):

        # Load the JSON file
        with open(sensor_characterization_path) as f:
            
            data = " ".join(f.readlines())
            sensor_data = json.loads(data)

            # Remove the info entry
            if '-1' in sensor_data:
                del sensor_data['-1']

    else:

        # Run sensor characterization
        sensor_data = sensorCharacterization(config, dir_path)

        # Save to file for posterior use
        with open(sensor_characterization_path, 'w') as f:

            # Add an explanation what each entry means
            sensor_data_save = dict(sensor_data)
            sensor_data_save['-1'] = {"FF file name": ['median star FWHM', 'median background noise stddev']}

            # Convert collection areas to JSON
            out_str = json.dumps(sensor_data_save, indent=4, sort_keys=True)

            # Save to disk
            f.write(out_str)



    # Compute the nighly mean FWHM and noise stddev
    fwhm_nightly_mean = np.mean([sensor_data[key][0] for key in sensor_data])
    stddev_nightly_mean = np.mean([sensor_data[key][1] for key in sensor_data])

    ### ###



    # Perform shower association
    associations, _ = showerAssociation(config, [ftpdetectinfo_path], shower_code=shower_code, \
        show_plot=False, save_plot=False, plot_activity=False)

    # Init the flux configuration
    flux_config = FluxConfig()


    # Remove all meteors which begin below the limit height
    filtered_associations = {}
    for key in associations:
        meteor, shower = associations[key]

        if meteor.beg_alt > flux_config.elev_limit:
            print("Rejecting:", meteor.jdt_ref)
            filtered_associations[key] = [meteor, shower]

    associations = filtered_associations



    # If there are no shower association, return nothing
    if not associations:
        print("No meteors associated with the shower!")
        return None


    # Print the list of used meteors
    peak_mags = []
    for key in associations:
        meteor, shower = associations[key]

        if shower is not None:

            # Compute peak magnitude
            peak_mag = np.min(meteor.mag_array)

            peak_mags.append(peak_mag)

            print("{:.6f}, {:3s}, {:+.2f}".format(meteor.jdt_ref, shower.name, peak_mag))

    print()



    ### COMPUTE COLLECTION AREAS ###

    # Make a file name to save the raw collection areas
    col_areas_file_name = generateColAreaJSONFileName(platepar.station_code, flux_config.side_points, \
        flux_config.ht_min, flux_config.ht_max, flux_config.dht, flux_config.elev_limit)

    # Check if the collection area file exists. If yes, load the data. If not, generate collection areas
    if col_areas_file_name in os.listdir(dir_path):
        col_areas_ht = loadRawCollectionAreas(dir_path, col_areas_file_name)

        print("Loaded collection areas from:", col_areas_file_name)

    else:

        # Compute the collecting areas segments per height
        col_areas_ht = collectingArea(platepar, mask=mask, side_points=flux_config.side_points, \
            ht_min=flux_config.ht_min, ht_max=flux_config.ht_max, dht=flux_config.dht, \
            elev_limit=flux_config.elev_limit)

        # Save the collection areas to file
        saveRawCollectionAreas(dir_path, col_areas_file_name, col_areas_ht)

        print("Saved raw collection areas to:", col_areas_file_name)


    ### ###


    # Compute the raw collection area at the height of 100 km
    col_area_100km_raw = 0
    col_areas_100km_blocks = col_areas_ht[100000.0]
    for block in col_areas_100km_blocks:
        col_area_100km_raw += col_areas_100km_blocks[block][0]

    print("Raw collection area at height of 100 km: {:.2f} km^2".format(col_area_100km_raw/1e6))


    # Compute the pointing of the middle of the FOV
    _, ra_mid, dec_mid, _ = xyToRaDecPP([jd2Date(J2000_JD.days)], [platepar.X_res/2], [platepar.Y_res/2], \
        [1], platepar, extinction_correction=False)
    azim_mid, elev_mid = raDec2AltAz(ra_mid[0], dec_mid[0], J2000_JD.days, platepar.lat, platepar.lon)

    # Compute the range to the middle point
    ref_ht = 100000
    r_mid, _, _, _ = xyHt2Geo(platepar, platepar.X_res/2, platepar.Y_res/2, ref_ht, indicate_limit=True, \
        elev_limit=flux_config.elev_limit)

    print("Range at 100 km in the middle of the image: {:.2f} km".format(r_mid/1000))


    ### Compute the average angular velocity to which the flux variation throught the night will be normalized 
    #   The ang vel is of the middle of the FOV in the middle of observations

    # Middle Julian date of the night
    jd_night_mid = (datetime2JD(dt_beg) + datetime2JD(dt_end))/2

    # Compute the apparent radiant
    ra, dec, v_init = shower.computeApparentRadiant(platepar.lat, platepar.lon, jd_night_mid)

    # Compute the radiant elevation
    radiant_azim, radiant_elev = raDec2AltAz(ra, dec, jd_night_mid, platepar.lat, platepar.lon)

    # Compute the angular velocity in the middle of the FOV
    rad_dist_night_mid = angularSeparation(np.radians(radiant_azim), np.radians(radiant_elev), 
                np.radians(azim_mid), np.radians(elev_mid))
    ang_vel_night_mid = v_init*np.sin(rad_dist_night_mid)/r_mid

    ###




    # Compute the average limiting magnitude to which all flux will be normalized

    # Standard deviation of star PSF, nightly mean (px)
    star_stddev = fwhm_nightly_mean/2.355

    # # Compute the theoretical stellar limiting magnitude (nightly average)
    # star_sum = 2*np.pi*(config.k1_det*stddev_nightly_mean + config.j1_det)*star_stddev**2
    # lm_s_nightly_mean = -2.5*np.log10(star_sum) + mag_lev_nightly_mean

    # Compute the theoretical stellar limiting magnitude using an empirical model (nightly average)
    lm_s_nightly_mean = stellarLMModel(mag_lev_nightly_mean)


    # A meteor needs to be visible on at least 4 frames, thus it needs to have at least 4x the mass to produce
    #   that amount of light. 1 magnitude difference scales as -0.4 of log of mass, thus:
    #frame_min_loss = np.log10(config.line_minimum_frame_range_det)/(-0.4)
    frame_min_loss = 0.0 # TEST !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11

    print("Frame min loss: {:.2} mag".format(frame_min_loss))

    lm_s_nightly_mean += frame_min_loss

    # Compute apparent meteor magnitude
    lm_m_nightly_mean = lm_s_nightly_mean - 5*np.log10(r_mid/1e5) - 2.5*np.log10( \
        np.degrees(platepar.F_scale*v_init*np.sin(rad_dist_night_mid)/(config.fps*r_mid*fwhm_nightly_mean)) \
        )

    #
    print("Stellar lim mag using detection thresholds:", lm_s_nightly_mean)
    print("Apparent meteor limiting magnitude:", lm_m_nightly_mean)


    ### Apply time-dependent corrections ###

    # Track values used for flux
    sol_data = []
    flux_lm_6_5_data = []
    flux_lm_6_5_ci_lower_data = []
    flux_lm_6_5_ci_upper_data = []
    meteor_num_data = []
    effective_collection_area_data = []
    radiant_elev_data = []
    radiant_dist_mid_data = []
    ang_vel_mid_data = []
    lm_s_data = []
    lm_m_data = []
    sensitivity_corr_data = []
    range_corr_data = []
    radiant_elev_corr_data = []
    ang_vel_corr_data = []
    total_corr_data = []


    # Go through all time bins within the observation period
    total_time_hrs = (dt_end - dt_beg).total_seconds()/3600
    nbins = int(np.ceil(total_time_hrs/timebin))
    for t_bin in range(nbins):
        for subbin in range(flux_config.sub_time_bins):

            # Compute bin start and end time
            bin_dt_beg = dt_beg + datetime.timedelta(hours=(timebin*t_bin + timebin*subbin/flux_config.sub_time_bins))
            bin_dt_end = bin_dt_beg + datetime.timedelta(hours=timebin)

            if bin_dt_end > dt_end:
                bin_dt_end = dt_end


            # Compute bin duration in hours
            bin_hours = (bin_dt_end - bin_dt_beg).total_seconds()/3600

            # Convert to Julian date
            bin_jd_beg = datetime2JD(bin_dt_beg)
            bin_jd_end = datetime2JD(bin_dt_end)

                

            jd_mean = (bin_jd_beg + bin_jd_end)/2

            # Compute the mean solar longitude
            sol_mean = np.degrees(jd2SolLonSteyaert(jd_mean))

            ### Compute the radiant elevation at the middle of the time bin ###

            # Compute the apparent radiant
            ra, dec, v_init = shower.computeApparentRadiant(platepar.lat, platepar.lon, jd_mean)

            # Compute the mean meteor height
            meteor_ht_beg = heightModel(v_init, ht_type='beg')
            meteor_ht_end = heightModel(v_init, ht_type='end')
            meteor_ht = (meteor_ht_beg + meteor_ht_end)/2

            # Compute the standard deviation of the height
            meteor_ht_std = meteor_ht*ht_std_percent/100.0

            # Init the Gaussian height distribution
            meteor_ht_gauss = scipy.stats.norm(meteor_ht, meteor_ht_std)


            # Compute the radiant elevation
            radiant_azim, radiant_elev = raDec2AltAz(ra, dec, jd_mean, platepar.lat, platepar.lon)




            # Only select meteors in this bin and not too close to the radiant
            bin_meteors = []
            bin_ffs = []
            for key in associations:
                meteor, shower = associations[key]

                if shower is not None:
                    if (shower.name == shower_code) and (meteor.jdt_ref > bin_jd_beg) \
                        and (meteor.jdt_ref <= bin_jd_end):

                        # Filter out meteors ending too close to the radiant
                        if np.degrees(angularSeparation(np.radians(radiant_azim), np.radians(radiant_elev), \
                            np.radians(meteor.end_azim), np.radians(meteor.end_alt))) >= flux_config.rad_dist_min:
                        
                            bin_meteors.append([meteor, shower])
                            bin_ffs.append(meteor.ff_name)

            ### ###

            print()
            print()
            print("-- Bin information ---")
            print("Bin beg:", bin_dt_beg)
            print("Bin end:", bin_dt_end)
            print("Sol mid: {:.5f}".format(sol_mean))
            print("Radiant elevation: {:.2f} deg".format(radiant_elev))
            print("Apparent speed: {:.2f} km/s".format(v_init/1000))

            # If the elevation of the radiant is below the limit, skip this bin
            if radiant_elev < flux_config.rad_elev_limit:
                print("!!! Mean radiant elevation below {:.2f} deg threshold, skipping time bin!".format(flux_config.rad_elev_limit))
                continue

            # The minimum duration of the time bin should be larger than 50% of the given dt
            if bin_hours < 0.5*timebin:
                print("!!! Time bin duration of {:.2f} h is shorter than 0.5x of the time bin!".format(bin_hours))
                continue


            if len(bin_meteors) >= flux_config.meteros_min:
                print("Meteors:", len(bin_meteors))


                ### Weight collection area by meteor height distribution ###

                # Determine weights for each height
                weight_sum = 0
                weights = {}
                for ht in col_areas_ht:
                    wt = meteor_ht_gauss.pdf(float(ht))
                    weight_sum += wt
                    weights[ht] = wt

                # Normalize the weights so that the sum is 1
                for ht in weights:
                    weights[ht] /= weight_sum

                ### ###


                col_area_meteor_ht_raw = 0
                for ht in col_areas_ht:
                    for block in col_areas_ht[ht]:
                        col_area_meteor_ht_raw += weights[ht]*col_areas_ht[ht][block][0]

                print("Raw collection area at meteor heights: {:.2f} km^2".format(col_area_meteor_ht_raw/1e6))

                # Compute the angular velocity in the middle of the FOV
                rad_dist_mid = angularSeparation(np.radians(radiant_azim), np.radians(radiant_elev), 
                            np.radians(azim_mid), np.radians(elev_mid))
                ang_vel_mid = v_init*np.sin(rad_dist_mid)/r_mid


                # Skip time bin if the radiant is very close to the centre of the image
                if np.degrees(rad_dist_mid) < flux_config.rad_dist_min:
                    print("!!! Radiant too close to the centre of the image! {:.2f} < {:.2f}".format(np.degrees(rad_dist_mid), flux_config.rad_dist_min))
                    continue


                ### Compute the limiting magnitude ###

                # Compute the mean star FWHM in the given bin
                fwhm_bin_mean = np.mean([sensor_data[ff_name][0] for ff_name in bin_ffs])

                # Compute the mean background stddev in the given bin
                stddev_bin_mean = np.mean([sensor_data[ff_name][1] for ff_name in bin_ffs])

                # Compute the mean photometric zero point in the given bin
                mag_lev_bin_mean = np.mean([recalibrated_platepars[ff_name].mag_lev for ff_name in bin_ffs if ff_name in recalibrated_platepars])



                # # Standard deviation of star PSF, nightly mean (px)
                # star_stddev = fwhm_bin_mean/2.355

                # Compute the theoretical stellar limiting magnitude (bin average)
                # star_sum = 2*np.pi*(config.k1_det*stddev_bin_mean + config.j1_det)*star_stddev**2
                # lm_s = -2.5*np.log10(star_sum) + mag_lev_bin_mean
                
                # Use empirical LM calculation
                lm_s = stellarLMModel(mag_lev_bin_mean)

                lm_s += frame_min_loss


                # ### TEST !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11

                # # Artificialy increase limiting magnitude
                # lm_s += 1.2

                # #####

                # Compute apparent meteor magnitude
                lm_m = lm_s - 5*np.log10(r_mid/1e5) - 2.5*np.log10( \
                    np.degrees(platepar.F_scale*v_init*np.sin(rad_dist_mid)/(config.fps*r_mid*fwhm_bin_mean)))

                ### ###


                # Final correction area value (height-weightned)
                collection_area = 0

                # Keep track of the corrections
                sensitivity_corr_arr = []
                range_corr_arr = []
                radiant_elev_corr_arr = []
                ang_vel_corr_arr = []
                total_corr_arr = []
                col_area_raw_arr = []
                col_area_eff_arr = []
                col_area_eff_block_dict = {}

                # Go through all heights and segment blocks
                for ht in col_areas_ht:
                    for img_coords in col_areas_ht[ht]:

                        x_mean, y_mean = img_coords

                        # Unpack precomputed values
                        area, azim, elev, sensitivity_ratio, r = col_areas_ht[ht][img_coords]


                        # Compute the angular velocity (rad/s) in the middle of this block
                        rad_dist = angularSeparation(np.radians(radiant_azim), np.radians(radiant_elev), 
                            np.radians(azim), np.radians(elev))
                        ang_vel = v_init*np.sin(rad_dist)/r


                        # If the angular distance from the radiant is less than 15 deg, don't use the block
                        #   in the effective collection area
                        if np.degrees(rad_dist) < flux_config.rad_dist_min:
                            area = 0.0


                        # Compute the range correction
                        range_correction = (1e5/r)**2

                        #ang_vel_correction = ang_vel/ang_vel_mid
                        # Compute angular velocity correction relative to the nightly mean
                        ang_vel_correction = ang_vel/ang_vel_night_mid



                        ### Apply corrections

                        correction_ratio = 1.0
                        
                        # Correct the area for vignetting and extinction
                        sensitivity_corr_arr.append(sensitivity_ratio)
                        correction_ratio *= sensitivity_ratio


                        # Correct for the range (cap to an order of magnitude correction)
                        range_correction = max(range_correction, 0.1)
                        range_corr_arr.append(range_correction)
                        correction_ratio *= range_correction

                        # Correct for the radiant elevation (cap to an order of magnitude correction)
                        radiant_elev_correction = np.sin(np.radians(radiant_elev))
                        radiant_elev_correction = max(radiant_elev_correction, 0.1)
                        radiant_elev_corr_arr.append(radiant_elev_correction)
                        correction_ratio *= radiant_elev_correction


                        # Correct for angular velocity (cap to an order of magnitude correction)
                        ang_vel_correction = min(max(ang_vel_correction, 0.1), 10)
                        correction_ratio *= ang_vel_correction
                        ang_vel_corr_arr.append(ang_vel_correction)


                        # Add the collection area to the final estimate with the height weight
                        #   Raise the correction to the mass index power
                        total_correction = correction_ratio**(mass_index - 1)
                        total_correction = min(max(total_correction, 0.1), 10)
                        collection_area += weights[ht]*area*total_correction
                        total_corr_arr.append(total_correction)

                        col_area_raw_arr.append(weights[ht]*area)
                        col_area_eff_arr.append(weights[ht]*area*total_correction)

                        if img_coords not in col_area_eff_block_dict:
                            col_area_eff_block_dict[img_coords] = []

                        col_area_eff_block_dict[img_coords].append(weights[ht]*area*total_correction)




                # Compute mean corrections
                sensitivity_corr_avg = np.mean(sensitivity_corr_arr)
                range_corr_avg = np.mean(range_corr_arr)
                radiant_elev_corr_avg = np.mean(radiant_elev_corr_arr)
                ang_vel_corr_avg = np.mean(ang_vel_corr_arr)
                total_corr_avg = np.median(total_corr_arr)
                col_area_raw_sum = np.sum(col_area_raw_arr)
                col_area_eff_sum = np.sum(col_area_eff_arr)

                print("Raw collection area at meteor heights (CHECK): {:.2f} km^2".format(col_area_raw_sum/1e6))
                print("Eff collection area at meteor heights (CHECK): {:.2f} km^2".format(col_area_eff_sum/1e6))



                # ### PLOT HOW THE CORRECTION VARIES ACROSS THE FOV
                # x_arr = []
                # y_arr = []
                # col_area_eff_block_arr = []

                # for img_coords in col_area_eff_block_dict:
                    
                #     x_mean, y_mean = img_coords

                #     #if x_mean not in x_arr:
                #     x_arr.append(x_mean)
                #     #if y_mean not in y_arr:
                #     y_arr.append(y_mean)

                #     col_area_eff_block_arr.append(np.sum(col_area_eff_block_dict[img_coords]))

                # x_unique = np.unique(x_arr)
                # y_unique = np.unique(y_arr)
                # # plt.pcolormesh(x_arr, y_arr, np.array(col_area_eff_block_arr).reshape(len(x_unique), len(y_unique)).T, shading='auto')
                # plt.title("TOTAL = " + str(np.sum(col_area_eff_block_arr)/1e6))
                # plt.scatter(x_arr, y_arr, c=np.array(col_area_eff_block_arr)/1e6)
                # #plt.pcolor(np.array(x_arr).reshape(len(x_unique), len(y_unique)), np.array(y_arr).reshape(len(x_unique), len(y_unique)), np.array(col_area_eff_block_arr).reshape(len(x_unique), len(y_unique))/1e6)
                # plt.colorbar(label="km^2")
                # plt.gca().invert_yaxis()
                # plt.show()

                # ###


                # Compute the nominal flux at the bin LM (meteors/1000km^2/h)
                flux = 1e9*len(bin_meteors)/collection_area/bin_hours

                # Compute confidence interval of the flux
                ci = 1.0 - confidence_interval
                num_ci_lower = scipy.stats.chi2.ppf(ci/2, 2*len(bin_meteors))/2
                num_ci_upper = scipy.stats.chi2.ppf(1 - ci/2, 2*(len(bin_meteors) + 1))/2
                flux_ci_lower = 1e9*num_ci_lower/collection_area/bin_hours
                flux_ci_upper = 1e9*num_ci_upper/collection_area/bin_hours

                # Compute the flux scaled to the nightly mean LM
                flux_lm_nightly_mean = flux*population_index**(lm_m_nightly_mean - lm_m)

                # Compute the flux scaled to +6.5M
                flux_lm_6_5 = flux*population_index**(6.5 - lm_m)

                # Compute scaled confidence intervals purely based on Poisson statistics (LM errors not 
                #   taken into account yet)
                flux_lm_6_5_ci_lower = flux_ci_lower*population_index**(6.5 - lm_m)
                flux_lm_6_5_ci_upper = flux_ci_upper*population_index**(6.5 - lm_m)



                print("-- Sensor information ---")
                print("Star FWHM:  {:5.2f} px".format(fwhm_bin_mean))
                print("Bkg stddev: {:4.1f} ADU".format(stddev_bin_mean))
                print("Photom ZP:  {:+6.2f} mag".format(mag_lev_bin_mean))
                print("Stellar LM: {:+.2f} mag".format(lm_s))
                print("-- Flux ---")
                print("Meteors:  {:d}, {:.0f}% CI [{:.2f}, {:.2f}]".format(len(bin_meteors), \
                    100*confidence_interval, num_ci_lower, num_ci_upper))
                print("Col area: {:d} km^2".format(int(collection_area/1e6)))
                print("Ang vel:  {:.2f} deg/s".format(np.degrees(ang_vel_mid)))
                print("LM app:   {:+.2f} mag".format(lm_m))
                print("Flux:     {:.2f} meteors/1000km^2/h".format(flux))
                print("to {:+.2f}: {:.2f} meteors/1000km^2/h".format(lm_m_nightly_mean, flux_lm_nightly_mean))
                print("to +6.50: {:.2f}, {:.0f}% CI [{:.2f}, {:.2f}] meteors/1000km^2/h".format(flux_lm_6_5, \
                    100*confidence_interval, flux_lm_6_5_ci_lower, flux_lm_6_5_ci_upper))


                sol_data.append(sol_mean)
                flux_lm_6_5_data.append(flux_lm_6_5)
                flux_lm_6_5_ci_lower_data.append(flux_lm_6_5_ci_lower)
                flux_lm_6_5_ci_upper_data.append(flux_lm_6_5_ci_upper)
                meteor_num_data.append(len(bin_meteors))
                effective_collection_area_data.append(collection_area)
                radiant_elev_data.append(radiant_elev)
                radiant_dist_mid_data.append(np.degrees(rad_dist_mid))
                ang_vel_mid_data.append(np.degrees(ang_vel_mid))
                lm_s_data.append(lm_s)
                lm_m_data.append(lm_m)

                sensitivity_corr_data.append(sensitivity_corr_avg)
                range_corr_data.append(range_corr_avg)
                radiant_elev_corr_data.append(radiant_elev_corr_avg)
                ang_vel_corr_data.append(ang_vel_corr_avg)
                total_corr_data.append(total_corr_avg)


    # Print the results
    print("Solar longitude, Flux at LM +6.5:")
    for sol, flux_lm_6_5 in zip(sol_data, flux_lm_6_5_data):
        print("{:9.5f}, {:8.4f}".format(sol, flux_lm_6_5))


    if show_plots and len(sol_data):

        # Plot a histogram of peak magnitudes
        plt.hist(peak_mags, cumulative=True, log=True, bins=len(peak_mags), density=True)

        # Plot population index
        r_intercept = -0.7
        x_arr = np.linspace(np.min(peak_mags), np.percentile(peak_mags, 60))
        plt.plot(x_arr, 10**(np.log10(population_index)*x_arr + r_intercept))

        plt.title("r = {:.2f}".format(population_index))

        plt.show()


        # Plot how the derived values change throughout the night
        fig, axes \
            = plt.subplots(nrows=4, ncols=2, sharex=True, figsize=(10, 8))

        ((ax_met,      ax_lm),
         (ax_rad_elev, ax_corrs),
         (ax_rad_dist, ax_col_area),
         (ax_ang_vel,  ax_flux)) = axes


        fig.suptitle("{:s}, s = {:.2f}, r = {:.2f}".format(shower_code, mass_index, population_index))


        ax_met.scatter(sol_data, meteor_num_data)
        ax_met.set_ylabel("Meteors")

        ax_rad_elev.plot(sol_data, radiant_elev_data)
        ax_rad_elev.set_ylabel("Radiant elev (deg)")

        ax_rad_dist.plot(sol_data, radiant_dist_mid_data)
        ax_rad_dist.set_ylabel("Radiant dist (deg)")

        ax_ang_vel.plot(sol_data, ang_vel_mid_data)
        ax_ang_vel.set_ylabel("Ang vel (deg/s)")
        ax_ang_vel.set_xlabel("La Sun (deg)")


        ax_lm.plot(sol_data, lm_s_data, label="Stellar")
        ax_lm.plot(sol_data, lm_m_data, label="Meteor")
        ax_lm.set_ylabel("LM")
        ax_lm.legend()

        ax_corrs.plot(sol_data, sensitivity_corr_data, label="Sensitivity")
        ax_corrs.plot(sol_data, range_corr_data, label="Range")
        ax_corrs.plot(sol_data, radiant_elev_corr_data, label="Rad elev")
        ax_corrs.plot(sol_data, ang_vel_corr_data, label="Ang vel")
        ax_corrs.plot(sol_data, total_corr_data, label="Total (median)")
        ax_corrs.set_ylabel("Corrections")
        ax_corrs.legend()

        

        ax_col_area.plot(sol_data, np.array(effective_collection_area_data)/1e6)
        ax_col_area.plot(sol_data, len(sol_data)*[col_area_100km_raw/1e6], color='k', \
            label="Raw col area at 100 km")
        ax_col_area.plot(sol_data, len(sol_data)*[col_area_meteor_ht_raw/1e6], color='k', linestyle='dashed', \
            label="Raw col area at met ht")
        ax_col_area.set_ylabel("Eff. col. area (km^2)")
        ax_col_area.legend()


        ax_flux.scatter(sol_data, flux_lm_6_5_data, color='k', zorder=4)
        ax_flux.errorbar(sol_data, flux_lm_6_5_data, color='grey', capsize=5, zorder=3, linestyle='none', \
            yerr=[np.array(flux_lm_6_5_data) - np.array(flux_lm_6_5_ci_lower_data), \
                np.array(flux_lm_6_5_ci_upper_data) - np.array(flux_lm_6_5_data)])

        ax_flux.set_ylabel("Flux@+6.5M (met/1000km^2/h)")
        ax_flux.set_xlabel("La Sun (deg)")

        plt.tight_layout()

        plt.show()


    return sol_data, flux_lm_6_5_data, flux_lm_6_5_ci_lower_data, flux_lm_6_5_ci_upper_data, meteor_num_data
        



if __name__ == "__main__":

    import argparse

    import RMS.ConfigReader as cr

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Compute single-station meteor shower flux.")

    arg_parser.add_argument("ftpdetectinfo_path", metavar="FTPDETECTINFO_PATH", type=str, \
        help="Path to an FTPdetectinfo file. The directory also has to contain a platepar and mask file.")

    arg_parser.add_argument("shower_code", metavar="SHOWER_CODE", type=str, \
        help="IAU shower code (e.g. ETA, PER, SDA).")

    arg_parser.add_argument("tbeg", metavar="BEG_TIME", type=str, \
        help="Time of the observation beginning. YYYYMMDD_HHMMSS format.")

    arg_parser.add_argument("tend", metavar="END_TIME", type=str, \
        help="Time of the observation ending. YYYYMMDD_HHMMSS format.")

    arg_parser.add_argument("dt", metavar="TIME_BIN", type=float, \
        help="Time bin width in hours.")

    arg_parser.add_argument("s", metavar="MASS_INDEX", type=float, \
        help="Mass index of the shower.")

    arg_parser.add_argument("-c", "--config", metavar="CONFIG_PATH", type=str,
                            help="Path to a config file which will be used instead of the default one."
                                 " To load the .config file in the given data directory, write '.' (dot).")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    ftpdetectinfo_path = cml_args.ftpdetectinfo_path

    # # Apply wildcards to input
    # ftpdetectinfo_path_list = []
    # for entry in ftpdetectinfo_path:

    #     # Expand wildcards and find all paths
    #     paths = glob.glob(entry)

    #     # Only take paths to files, not directories
    #     paths = [entry for entry in paths if os.path.isfile(entry)]

    #     ftpdetectinfo_path_list += paths


    # # If there are no good files given, notify the user
    # if len(ftpdetectinfo_path_list) == 0:
    #     print("No FTPdetectinfo files given!")
    #     sys.exit()

    if not os.path.isfile(cml_args.ftpdetectinfo_path):
        print("The FTPdetectinfo file does not exist:", cml_args.ftpdetectinfo_path)
        print("Exiting...")
        sys.exit()


    # Parse the beg/end time
    dt_beg = datetime.datetime.strptime(cml_args.tbeg, "%Y%m%d_%H%M%S")
    dt_end = datetime.datetime.strptime(cml_args.tend, "%Y%m%d_%H%M%S")
        

    # Extract parent directory
    dir_path = os.path.dirname(cml_args.ftpdetectinfo_path)

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, dir_path)


    # Compute the flux
    computeFlux(config, dir_path, cml_args.ftpdetectinfo_path, cml_args.shower_code, dt_beg, dt_end, \
        cml_args.dt, cml_args.s)