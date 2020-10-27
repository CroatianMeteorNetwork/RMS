""" Compute single-station meteor shower flux. """

import os
import sys
import glob
import copy

import numpy as np

from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP
from RMS.Astrometry.Conversions import areaGeoPolygon, jd2Date, J2000_JD
import RMS.ConfigReader as cr
from RMS.Formats import Platepar
from RMS.Formats.FTPdetectinfo import readFTPdetectinfo
from RMS.Routines.FOVArea import xyHt2Geo, fovArea
from RMS.Routines.MaskImage import loadMask, MaskStructure
from Utils.ShowerAssociation import showerAssociation




def collectingArea(platepar, mask=None, side_points=20, ht_min=60, ht_max=120, dht=2, elev_limit=10):
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


    # Estimate the collection area for a given range of heights
    for ht in np.arange(ht_min, ht_max + dht, dht):

        # Convert the height to meters
        ht = 1000*ht

        print(ht/1000, "km")

        total_area = 0

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


                ## Compute the vignetting and extinction loss for the mean location

                x_mean = (x0 + xe)/2
                y_mean = (y0 + ye)/2

                # Use a test pixel sum
                test_px_sum = 400

                # Compute the magnitude corrected for vignetting and extinction
                _, _, _, mag = xyToRaDecPP([jd2Date(J2000_JD.days)], [x_mean], [y_mean], [test_px_sum], \
                    platepar)

                # Compute the pixel sum back assuming no corrections
                rev_level = 10**((mag[0] - platepar.mag_lev)/(-2.5))
                
                # Compute the sensitivty loss due to vignetting and extinction
                sensitivity_ratio = test_px_sum/rev_level

                # print(np.abs(np.hypot(x_mean - platepar.X_res/2, y_mean - platepar.Y_res/2)), sensitivity_ratio, mag[0])

                ##


                # Compute the range correction (w.r.t 100 km) to the mean point
                r, _, _, _ = xyHt2Geo(platepar, x_mean, y_mean, ht, indicate_limit=True, \
                    elev_limit=elev_limit)
                range_correction = (1e5/r)**2


                # Compute angular velocity loss using the estimated stddev at the given image segment



                # Compute radiant distance correction


                ## Apply collection area corrections ##

                # Correct the area for the masked portion
                area *= unmasked_ratio

                # Correct the area for vignetting and extinction
                area *= sensitivity_ratio

                # Correct for the range
                area *= range_correction

                ## ##
                

                ### ###


                total_area += area


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









        # Compute the ratio of masked and unmasked pixels



        pass



    





def computeFlux(config, ftpdetectinfo_list, shower_code=None, mask=None):
    """ Compute flux using measurements in the given FTPdetectinfo file. 
    
    Arguments:
        config: [Config instance]
        ftpdetectinfo_list: [list] A list of paths to FTPdetectinfo files.

    Keyword arguments:
        shower_code: [str] Only use this one shower for association (e.g. ETA, PER, SDA). None by default,
            in which case all active showers will be used.
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
    associations, shower_counts = showerAssociation(config, ftpdetectinfo_list, shower_code=shower_code, show_plot=False, save_plot=False, \
        plot_activity=False)

    # If there are no shower association, return nothing
    if not associations:
        print("No meteors assocaited with a shower!")
        return None



    # Compute the collecting area
    collecting_area = collectingArea(platepar, mask=mask)





if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Compute single-station meteor shower flux.")

    arg_parser.add_argument("ftpdetectinfo_path", nargs="+", metavar="FTPDETECTINFO_PATH", type=str, \
        help="Path to one or more FTPdetectinfo files. The directory also has to contain a platepar and mask file.")

    arg_parser.add_argument("-c", "--config", metavar="CONFIG_PATH", type=str,
                            help="Path to a config file which will be used instead of the default one."
                                 " To load the .config file in the given data directory, write '.' (dot).")

    arg_parser.add_argument("-s", "--shower", metavar="SHOWER", type=str, \
        help="Associate just this single shower given its code (e.g. PER, ORI, ETA).")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    ftpdetectinfo_path = cml_args.ftpdetectinfo_path

    # Apply wildcards to input
    ftpdetectinfo_path_list = []
    for entry in ftpdetectinfo_path:
        ftpdetectinfo_path_list += glob.glob(entry)


    # If there are no good files given, notify the user
    if len(ftpdetectinfo_path_list) == 0:
        print("No FTPdetectinfo files given!")
        sys.exit()
        

    # Extract parent directory
    dir_path = os.path.dirname(ftpdetectinfo_path_list[0])

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, dir_path)


    # Compute the flux
    computeFlux(config, ftpdetectinfo_path_list, shower_code=cml_args.shower)