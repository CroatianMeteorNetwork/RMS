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
import RMS.ConfigReader as cr
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

            col_areas_ht[key] = collections.OrderedDict()

            for str_key in col_areas_ht_strkeys[key]:

                # Convert the string "x_mid, y_mid" to tuple of floats (x_mid, y_mid)
                tuple_key = tuple(map(float, str_key.split(", ")))

                col_areas_ht[key][tuple_key] = col_areas_ht_strkeys[key][str_key]


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

        # Limit of meteor's elevation above horizon (deg). 10 degrees by default.
        self.elev_limit = 10

        # Minimum radiant elevation in the time bin (deg). 15 degreees by default
        self.rad_elev_limit = 15



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





def computeFlux(config, dir_path, ftpdetectinfo_path, shower_code, dt_beg, dt_end, timebin, mass_index, \
    timebin_intdt=0.25, ht_std_percent=5.0, mask=None):
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
    population_index = 10**((mass_index - 1)/2.5)


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
    associations, shower_counts = showerAssociation(config, [ftpdetectinfo_path], shower_code=shower_code, \
        show_plot=False, save_plot=False, plot_activity=False)

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


    # Init the flux configuration
    flux_config = FluxConfig()



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



    # Compute the pointing of the middle of the FOV
    _, ra_mid, dec_mid, _ = xyToRaDecPP([jd2Date(J2000_JD.days)], [platepar.X_res/2], [platepar.Y_res/2], \
        [1], platepar, extinction_correction=False)
    azim_mid, elev_mid = raDec2AltAz(ra_mid[0], dec_mid[0], J2000_JD.days, platepar.lat, platepar.lon)

    # Compute the range to the middle point
    ref_ht = 100000
    r_mid, _, _, _ = xyHt2Geo(platepar, platepar.X_res/2, platepar.Y_res/2, ref_ht, indicate_limit=True, \
        elev_limit=flux_config.elev_limit)


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

    # Compute the theoretical stellar limiting magnitude (nightly average)
    star_sum = 2*np.pi*(config.k1_det*stddev_nightly_mean + config.j1_det)*star_stddev**2
    lm_s_nightly_mean = -2.5*np.log10(star_sum) + mag_lev_nightly_mean

    # A meteor needs to be visible on at least 4 frames, thus it needs to have at least 4x the mass to produce
    #   that amount of light. 1 magnitude difference scales as -0.4 of log of mass, thus:
    frame_min_loss = np.log10(config.line_minimum_frame_range_det)/(-0.4)

    lm_s_nightly_mean += frame_min_loss

    # Compute apparent meteor magnitude
    lm_m_nightly_mean = lm_s_nightly_mean - 5*np.log10(r_mid/1e5) - 2.5*np.log10( \
        np.degrees(platepar.F_scale*v_init*np.sin(rad_dist_night_mid)/(config.fps*r_mid*fwhm_nightly_mean)) \
        )

    #
    print("Stellar lim mag using detection thresholds:", lm_s_nightly_mean)
    print("Apparent meteor limiting magnitude:", lm_m_nightly_mean)


    ### Apply time-dependent corrections ###

    sol_data = []
    flux_lm_6_5_data = []

    # Go through all time bins within the observation period
    total_time_hrs = (dt_end - dt_beg).total_seconds()/3600
    nbins = int(np.ceil(total_time_hrs/timebin))
    for t_bin in range(nbins):

        # Compute bin start and end time
        bin_dt_beg = dt_beg + datetime.timedelta(hours=timebin*t_bin)
        bin_dt_end = bin_dt_beg + datetime.timedelta(hours=timebin)

        if bin_dt_end > dt_end:
            bin_dt_end = dt_end


        # Compute bin duration in hours
        bin_hours = (bin_dt_end - bin_dt_beg).total_seconds()/3600

        # Convert to Julian date
        bin_jd_beg = datetime2JD(bin_dt_beg)
        bin_jd_end = datetime2JD(bin_dt_end)

        # Only select meteors in this bin
        bin_meteors = []
        bin_ffs = []
        for key in associations:
            meteor, shower = associations[key]

            if shower is not None:
                if (shower.name == shower_code) and (meteor.jdt_ref > bin_jd_beg) \
                    and (meteor.jdt_ref <= bin_jd_end):
                    
                    bin_meteors.append([meteor, shower])
                    bin_ffs.append(meteor.ff_name)




            

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

        ### ###

        print()
        print()
        print("-- Bin information ---")
        print("Bin beg:", bin_dt_beg)
        print("Bin end:", bin_dt_end)
        print("Sol mid: {:.5f}".format(sol_mean))
        print("Radiant elevation: {:.2f} deg".format(radiant_elev))

        # If the elevation of the radiant is below the limit, skip this bin
        if radiant_elev < flux_config.rad_elev_limit:
            print("!!! Mean radiant elevation below {:.2f} deg threshold, skipping time bin!".format(flux_config.rad_elev_limit))
            continue


        if len(bin_meteors) > 0:
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


            # Compute the angular velocity in the middle of the FOV
            rad_dist_mid = angularSeparation(np.radians(radiant_azim), np.radians(radiant_elev), 
                        np.radians(azim_mid), np.radians(elev_mid))
            ang_vel_mid = v_init*np.sin(rad_dist_mid)/r_mid



            ### Compute the limiting magnitude ###

            # Compute the mean star FWHM in the given bin
            fwhm_bin_mean = np.mean([sensor_data[ff_name][0] for ff_name in bin_ffs])

            # Compute the mean background stddev in the given bin
            stddev_bin_mean = np.mean([sensor_data[ff_name][1] for ff_name in bin_ffs])

            # Compute the mean photometric zero point in the given bin
            mag_lev_bin_mean = np.mean([recalibrated_platepars[ff_name].mag_lev for ff_name in bin_ffs if ff_name in recalibrated_platepars])



            # Standard deviation of star PSF, nightly mean (px)
            star_stddev = fwhm_bin_mean/2.355

            # Compute the theoretical stellar limiting magnitude (nightly average)
            star_sum = 2*np.pi*(config.k1_det*stddev_bin_mean + config.j1_det)*star_stddev**2
            lm_s = -2.5*np.log10(star_sum) + mag_lev_bin_mean
            lm_s += frame_min_loss

            # Compute apparent meteor magnitude
            lm_m = lm_s - 5*np.log10(r_mid/1e5) - 2.5*np.log10( \
                    np.degrees(platepar.F_scale*v_init*np.sin(rad_dist_mid)/(config.fps*r_mid*fwhm_bin_mean))\
                    )

            ### ###


            # Final correction area value (height-weightned)
            collection_area = 0

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


                    # Compute the range correction
                    range_correction = (1e5/r)**2

                    #ang_vel_correction = ang_vel/ang_vel_mid
                    # Compute angular velocity correction relative to the nightly mean
                    ang_vel_correction = ang_vel/ang_vel_night_mid


                    ### Apply corrections

                    correction_ratio = 1.0
                    
                    # Correct the area for vignetting and extinction
                    correction_ratio *= sensitivity_ratio

                    # Correct for the range
                    correction_ratio *= range_correction

                    # Correct for the radiant elevation
                    correction_ratio *= np.sin(np.radians(radiant_elev))

                    # Correct for angular velocity
                    correction_ratio *= ang_vel_correction


                    # Add the collection area to the final estimate with the height weight
                    #   Raise the correction to the mass index power
                    collection_area += weights[ht]*area*correction_ratio**(mass_index - 1)



            # Compute the flux at the bin LM (meteors/1000km^2/h)
            flux = 1e9*len(bin_meteors)/collection_area/bin_hours

            # Compute the flux scaled to the nightly mean LM
            flux_lm_nightly_mean = flux*population_index**(lm_m_nightly_mean - lm_m)

            # Compute the flux scaled to +6.5M
            flux_lm_6_5 = flux*population_index**(6.5 - lm_m)



            print("-- Sensor information ---")
            print("Star FWHM:  {:5.2f} px".format(fwhm_bin_mean))
            print("Bkg stddev: {:4.1f} ADU".format(stddev_bin_mean))
            print("Photom ZP:  {:+6.2f} mag".format(mag_lev_bin_mean))
            print("Stellar LM: {:+.2f} mag".format(lm_s))
            print("-- Flux ---")
            print("Col area: {:d} km^2".format(int(collection_area/1e6)))
            print("Ang vel:  {:.2f} deg/s".format(np.degrees(ang_vel_mid)))
            print("LM app:   {:+.2f} mag".format(lm_m))
            print("Flux:     {:.2f} meteors/1000km^2/h".format(flux))
            print("to {:+.2f}: {:.2f} meteors/1000km^2/h".format(lm_m_nightly_mean, flux_lm_nightly_mean))
            print("to +6.50: {:.2f} meteors/1000km^2/h".format(flux_lm_6_5))


            sol_data.append(sol_mean)
            flux_lm_6_5_data.append(flux_lm_6_5)


    # Print the results
    print("Solar longitude, Flux at LM +6.5:")
    for sol, flux_lm_6_5 in zip(sol_data, flux_lm_6_5_data):
        print("{:9.5f}, {:8.4f}".format(sol, flux_lm_6_5))

    # Plot a histogram of peak magnitudes
    plt.hist(peak_mags, cumulative=True)
    plt.show()
    





if __name__ == "__main__":

    import argparse

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
    #dir_path = os.path.dirname(ftpdetectinfo_path_list[0])
    dir_path = os.path.dirname(cml_args.ftpdetectinfo_path)

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, dir_path)


    # Compute the flux
    computeFlux(config, dir_path, cml_args.ftpdetectinfo_path, cml_args.shower_code, dt_beg, dt_end, \
        cml_args.dt, cml_args.s)