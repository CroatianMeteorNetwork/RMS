#!/usr/bin/env python

"""Convert between mean sea level (EGM96) and WGS84 heights.
This file was taken from WesternMeteorPyLib, commit 5234439 (Dec 2020)"""

from __future__ import print_function, division, absolute_import

import os
import argparse
import functools

import numpy as np
import scipy.interpolate

from RMS.Misc import getRmsRootDir


def defaultEGM96Path():
    """ Path of the EGM96 geoid file shipped with RMS. """

    return os.path.join(getRmsRootDir(), 'share', 'WW15MGH.DAC')


def resolveEGM96Path(egm96=None):
    """ Resolve the EGM96 geoid file from what callers pass in.

    Arguments:
        egm96: [None, str or Config] None uses the file shipped with RMS, a string is taken as the file path,
            and a Config object is honoured through its egm96_full_path, or egm96_path + egm96_file_name
            attributes (the historical calling convention of these functions).

    Return:
        file_path: [str] Path to the EGM96 geoid file.
    """

    if egm96 is None or egm96 == '':
        return defaultEGM96Path()

    if isinstance(egm96, str):
        return egm96

    full_path = getattr(egm96, 'egm96_full_path', None)
    if full_path:
        return full_path

    dir_path = getattr(egm96, 'egm96_path', None)
    file_name = getattr(egm96, 'egm96_file_name', None)
    if dir_path and file_name:
        return os.path.join(dir_path, file_name)

    raise TypeError("egm96 must be None, a file path or a Config with egm96_path/egm96_file_name, "
                    "got {}".format(type(egm96).__name__))


def loadEGM96Data(file_path=None):
    """ Load a file with EGM96 data.

    EGM96 data source: http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/binary/binarygeoid.html

    Arguments:
        file_path: [None, str or Config] See resolveEGM96Path.
    """

    file_path = resolveEGM96Path(file_path)

    # Load the geoid heights
    geoid_heights = np.fromfile(file_path, dtype=np.int16).byteswap().astype(np.float64)

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


@functools.lru_cache(maxsize=8)
def getEGM96Model(file_path):
    """ Geoid interpolation model for the given file, built once per file path. Building it costs about
        0.1 s on a desktop and up to a second on a Raspberry Pi, so it must not be repeated per call.

    Arguments:
        file_path: [str] Resolved path to the EGM96 file (use resolveEGM96Path first).
    """

    return interpolateEGM96Data(loadEGM96Data(file_path))


def geoidUndulation(lat, lon, egm96_file_path=None):
    """ Height of the EGM96 geoid above the WGS84 ellipsoid at the given location.

    Arguments:
        lat: [float] Latitude +N (rad).
        lon: [float] Longitude +E (rad).
        egm96_file_path: [None, str or Config] See resolveEGM96Path.

    Return:
        undulation: [float] Geoid height above the ellipsoid (meters).
    """

    geoid_model = getEGM96Model(resolveEGM96Path(egm96_file_path))

    lat_mod = np.pi/2 - lat
    lon_mod = lon%(2*np.pi)

    return float(geoid_model(lat_mod, lon_mod)[0][0])


def mslToWGS84Height(lat, lon, msl_height, egm96_file_path=None):
    """ Given the height above sea level (using the EGM96 model), compute the height above the WGS84
        ellipsoid.

    Arguments:
        lat: [float] Latitude +N (rad).
        lon: [float] Longitude +E (rad).
        msl_height: [float] Height above sea level (meters).
        egm96_file_path: [None, str or Config] EGM96 geoid file, see resolveEGM96Path. Defaults to
            RMS/share/WW15MGH.DAC.

    Return:
        wgs84_height: [float] Height above the WGS84 ellipsoid.
    """

    return msl_height + geoidUndulation(lat, lon, egm96_file_path)


def wgs84toMSLHeight(lat, lon, wgs84_height, egm96_file_path=None):
    """ Given the height above the WGS84 ellipsoid compute the height above sea level (using the EGM96 model).

    Arguments:
        lat: [float] Latitude +N (rad).
        lon: [float] Longitude +E (rad).
        wgs84_height: [float] Height above the WGS84 ellipsoid (meters).
        egm96_file_path: [None, str or Config] EGM96 geoid file, see resolveEGM96Path. Defaults to
            RMS/share/WW15MGH.DAC.

    Return:
        msl_height: [float] Height above sea level (meters).
    """

    return wgs84_height - geoidUndulation(lat, lon, egm96_file_path)



if __name__ == "__main__":

    import RMS.ConfigReader as cr

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Convert mean sea level (EGM96) to WGS84")

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file whose EGM96 settings will be used instead of the default file.")

    arg_parser.add_argument('--egm96', type=str,
            help="Path to EGM96 file (defaults to RMS/share/WW15MGH.DAC)")

    arg_parser.add_argument('-i', '--inverse', action="store_true", \
            help="Convert WGS84 to EGM96 (default is False)")

    arg_parser.add_argument("latitude", type=float, help="Latitude in degrees (north is positive)")
    arg_parser.add_argument("longitude", type=float, help="Longitude in degrees (east is positive)")
    arg_parser.add_argument("height", type=float, help="Height to convert (in meters)")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    # Pick the geoid file: explicit path, then config, then the default
    if cml_args.egm96:
        egm96 = cml_args.egm96

    elif cml_args.config:
        egm96 = cr.loadConfigFromDirectory(cml_args.config, ".")

    else:
        egm96 = None

    # Load latitude and longitude
    lat = cml_args.latitude
    lon = cml_args.longitude

    if not cml_args.inverse:
        print("Converting MSL height to WGS84 height")
        msl_height = cml_args.height
        wgs84_height = mslToWGS84Height(np.radians(lat), np.radians(lon), msl_height, egm96)

    else:
        print("Converting WGS84 height to MSL height")
        wgs84_height = cml_args.height
        msl_height = wgs84toMSLHeight(np.radians(lat), np.radians(lon), wgs84_height, egm96)

    print('Latitude:', lat)
    print('Longitude', lon)
    print('MSL height (m): {:.2f}'.format(msl_height))
    print('WGS84 height (m): {:.2f}'.format(wgs84_height))
