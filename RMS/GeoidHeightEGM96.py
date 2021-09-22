#!/usr/bin/env python

"""Convert between mean sea level (EGM96) and WGS84 heights.
This file was taken from WesternMeteorPyLib, commit 5234439 (Dec 2020)"""

from __future__ import print_function, division, absolute_import

import os
import argparse

import numpy as np
import scipy.interpolate

import RMS.ConfigReader as cr


def loadEGM96Data(dir_path, file_name):
    """ Load a file with EGM96 data.

    EGM96 data source: http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/binary/binarygeoid.html
    """

    # Load the geoid heights
    geoid_heights = np.fromfile(os.path.join(dir_path, file_name), \
        dtype=np.int16).byteswap().astype(np.float64)

    # Reshape the data to 15 min grid
    geoid_heights = geoid_heights.reshape(721, 1440)

    # Compute the height in meters
    geoid_heights /= 100

    return geoid_heights



def interpolateEGM96Data(geoid_heights):
    """ Interpolate geoid heights on a sphere. """

    # Interpolate the data
    lat_points = np.radians(np.linspace(0.25, 179.25, 719))
    lon_points = np.radians(np.linspace(0, 359.75, 1440))

    # Extract pole values
    north_pole_value = geoid_heights[0][0]
    south_pole_value = geoid_heights[-1][0]

    # Remove points on the pole
    geoid_heights = geoid_heights[1:-1]

    # Construct an interpolation instance
    geoid_model = scipy.interpolate.RectSphereBivariateSpline(lat_points, lon_points, geoid_heights,
        pole_values=(north_pole_value, south_pole_value))

    return geoid_model



def mslToWGS84Height(lat, lon, msl_height, config):
    """ Given the height above sea level (using the EGM96 model), compute the height above the WGS84
        ellipsoid.

    Arguments:
        lat: [float] Latitude +N (rad).
        lon: [float] Longitude +E (rad).
        msl_height: [float] Height above sea level (meters).
        config: Config instance with the path to EGM96 coefficients

    Return:
        wgs84_height: [float] Height above the WGS84 ellipsoid.

    """

    # Load the geoid heights array
    GEOID_HEIGHTS = loadEGM96Data(config.egm96_path, config.egm96_file_name)

    # Init the interpolated geoid model
    GEOID_MODEL = interpolateEGM96Data(GEOID_HEIGHTS)

    # Get the difference between WGS84 and MSL height
    lat_mod = np.pi/2 - lat
    lon_mod = lon%(2*np.pi)
    msl_ht_diff = GEOID_MODEL(lat_mod, lon_mod)[0][0]

    # Compute the WGS84 height
    wgs84_height = msl_height + msl_ht_diff


    return wgs84_height



def wgs84toMSLHeight(lat, lon, wgs84_height, config):
    """ Given the height above the WGS84 ellipsoid compute the height above sea level (using the EGM96 model).

    Arguments:
        lat: [float] Latitude +N (rad).
        lon: [float] Longitude +E (rad).
        wgs84_height: [float] Height above the WGS84 ellipsoid (meters).
        config: Config instance with the path to EGM96 coefficients

    Return:
        msl_height: [float] Height above sea level (meters).

    """

    # Load the geoid heights array
    GEOID_HEIGHTS = loadEGM96Data(config.egm96_path, config.egm96_file_name)

    # Init the interpolated geoid model
    GEOID_MODEL = interpolateEGM96Data(GEOID_HEIGHTS)

    # Get the difference between WGS84 and MSL height
    lat_mod = np.pi/2 - lat
    lon_mod = lon%(2*np.pi)
    msl_ht_diff = GEOID_MODEL(lat_mod, lon_mod)[0][0]

    # Compute the sea level
    msl_height = wgs84_height - msl_ht_diff


    return msl_height



if __name__ == "__main__":

    import matplotlib.pyplot as plt

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Convert mean sea level (EGM96) to WGS84")

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one.")

    arg_parser.add_argument('-i', '--inverse', action="store_true", \
            help="Convert WGS84 to EGM96 (default is False)")

    arg_parser.add_argument("latitude", type=float, help="Latitude in degrees (east is positive)")
    arg_parser.add_argument("longitude", type=float, help="Longitude in degrees (north is positive)")
    arg_parser.add_argument("height", type=float, help="Height to convert (in meters)")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, ".")

    # Load latitude and longitude
    lat = cml_args.latitude
    lon = cml_args.longitude

    if not cml_args.inverse:
        print("Converting MSL height to WGS84 height")
        msl_height = cml_args.height
        wgs84_height = mslToWGS84Height(np.radians(lat), np.radians(lon), msl_height, config)
    else:
        print("Converting WGS84 height to MSL height")
        wgs84_height = cml_args.height
        msl_height = wgs84toMSLHeight(np.radians(lat), np.radians(lon), wgs84_height, config)

    print('Latitude:', lat)
    print('Longitude', lon)
    print('MSL height (m): {:.2f}'.format(msl_height))
    print('WGS84 height (m): {:.2f}'.format(wgs84_height))
