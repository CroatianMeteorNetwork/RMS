""" Compute single-station meteor shower flux. """

import os
import sys
import glob
import copy
import datetime
import json
import collections

import numpy as np
import scipy.stats

from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP
from RMS.Astrometry.Conversions import areaGeoPolygon, jd2Date, datetime2JD, J2000_JD, raDec2AltAz
import RMS.ConfigReader as cr
from RMS.Formats import Platepar
from RMS.Formats.FTPdetectinfo import readFTPdetectinfo
from RMS.Math import angularSeparation
from RMS.Routines.FOVArea import xyHt2Geo, fovArea
from RMS.Routines.MaskImage import loadMask, MaskStructure
from RMS.Routines.SolarLongitude import jd2SolLonSteyaert
from Utils.ShowerAssociation import showerAssociation, heightModel


def generateColAreaJSONFileName(station_code, side_points, ht_min, ht_max, dht, elev_limit):
    """ Generate a file name for the collection area JSON file. """

    file_name = "col_areas_{:s}_sp-{:d}_htmin-{:.1f}_htmax-{:.1f}_dht-{:.1f}_elemin-{:.1f}.json".format(\
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

        # Limit of elevation above horizon (deg). 10 degrees by default.
        self.elev_limit = 10



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

    





def computeFlux(config, dir_path, ftpdetectinfo_list, shower_code, dt_beg, dt_end, timebin, mass_index, \
    timebin_intdt=0.25, ht_std_percent=5.0, mask=None):
    """ Compute flux using measurements in the given FTPdetectinfo file. 
    
    Arguments:
        config: [Config instance]
        dir_path: [str] Path to the working directory.
        ftpdetectinfo_list: [list] A list of paths to FTPdetectinfo files.
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


    # Locate the mask file
    if config.mask_file in file_list:
        mask_path = os.path.join(dir_path, config.mask_file)
        mask = loadMask(mask_path)
        print("Using mask:", mask_path)

    else:
        print("No mask used!")
        mask = None



    # Load FTPdetectinfos
    meteor_data = []
    for ftpdetectinfo_path in ftpdetectinfo_list:

        if not os.path.isfile(ftpdetectinfo_path):
            print('No such file:', ftpdetectinfo_path)
            continue

        meteor_data += readFTPdetectinfo(*os.path.split(ftpdetectinfo_path))


    if not len(meteor_data):
        print("No meteors in the FTPdetectinfo file!")
        return None




    # Perform shower association
    associations, shower_counts = showerAssociation(config, ftpdetectinfo_list, shower_code=shower_code, \
        show_plot=False, save_plot=False, plot_activity=False)

    # If there are no shower association, return nothing
    if not associations:
        print("No meteors assocaited with a shower!")
        return None




    for key in associations:
        meteor, shower = associations[key]

        if shower is not None:

            print(meteor.jdt_ref, shower.name)



    # Init the flux configuration
    flux_config = FluxConfig()


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



    # Compute the pointing of the middle of the FOV
    _, ra_mid, dec_mid, _ = xyToRaDecPP([jd2Date(J2000_JD.days)], [platepar.X_res/2], [platepar.Y_res/2], \
        [1], platepar, extinction_correction=False)
    azim_mid, elev_mid = raDec2AltAz(ra_mid[0], dec_mid[0], J2000_JD.days, platepar.lat, platepar.lon)

    # Compute the range to the middle point
    ref_ht = 130000
    r_mid, _, _, _ = xyHt2Geo(platepar, platepar.X_res/2, platepar.Y_res/2, ref_ht, indicate_limit=True, \
        elev_limit=flux_config.elev_limit)



    ### Apply time-dependent corrections ###

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
        for key in associations:
            meteor, shower = associations[key]

            if shower is not None:
                if (shower.name == shower_code) and (meteor.jdt_ref > bin_jd_beg) \
                    and (meteor.jdt_ref <= bin_jd_end):
                    
                    bin_meteors.append([meteor, shower])


        if len(bin_meteors) > 0:


            ### Compute the radiant elevation at the middle of the time bin ###

            jd_mean = (bin_jd_beg + bin_jd_end)/2


            print(np.degrees(jd2SolLonSteyaert(jd_mean)), bin_dt_beg, bin_dt_end, len(bin_meteors))

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


            # Final correction area value (height-weightned)
            collection_area = 0

            # Go through all heights and segment blocks
            for ht in col_areas_ht:
                for img_coords in col_areas_ht[ht]:

                    x_mean, y_mean = img_coords

                    # Unpack precomputed values
                    area, azim, elev, sensitivity_ratio, r = col_areas_ht[ht][img_coords]


                    # Compute the angular velocity in the middle of this block
                    rad_dist = angularSeparation(np.radians(radiant_azim), np.radians(radiant_elev), 
                        np.radians(azim), np.radians(elev))
                    ang_vel = v_init*np.sin(rad_dist)/r


                    # Compute the range correction
                    range_correction = (1e5/r)**2

                    # Compute angular velocity correction
                    ang_vel_correction = ang_vel/ang_vel_mid


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


            print("Flux:", 1e9*len(bin_meteors)/collection_area/bin_hours, "meteors/1000km^2/h")
    





if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Compute single-station meteor shower flux.")

    arg_parser.add_argument("ftpdetectinfo_path", nargs="+", metavar="FTPDETECTINFO_PATH", type=str, \
        help="Path to one or more FTPdetectinfo files. The directory also has to contain a platepar and mask file.")

    arg_parser.add_argument("shower_code", metavar="SHOWER_CODE", type=str, \
        help="IAU shower code (e.g. ETA, PER, SDA).")

    arg_parser.add_argument("tbeg", metavar="BEG_TIME", type=str, \
        help="Time of the observation beginning. YYYYMMDD-HHMMSS format.")

    arg_parser.add_argument("tend", metavar="END_TIME", type=str, \
        help="Time of the observation ending. YYYYMMDD-HHMMSS format.")

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

    # Apply wildcards to input
    ftpdetectinfo_path_list = []
    for entry in ftpdetectinfo_path:

        # Expand wildcards and find all paths
        paths = glob.glob(entry)

        # Only take paths to files, not directories
        paths = [entry for entry in paths if os.path.isfile(entry)]

        ftpdetectinfo_path_list += paths


    # If there are no good files given, notify the user
    if len(ftpdetectinfo_path_list) == 0:
        print("No FTPdetectinfo files given!")
        sys.exit()


    # Parse the beg/end time
    dt_beg = datetime.datetime.strptime(cml_args.tbeg, "%Y%m%d-%H%M%S")
    dt_end = datetime.datetime.strptime(cml_args.tend, "%Y%m%d-%H%M%S")
        

    # Extract parent directory
    dir_path = os.path.dirname(ftpdetectinfo_path_list[0])

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, dir_path)


    # Compute the flux
    computeFlux(config, dir_path, ftpdetectinfo_path_list, cml_args.shower_code, dt_beg, dt_end, \
        cml_args.dt, cml_args.s)