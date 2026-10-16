#!/usr/bin/env python

"""Convert between mean sea level (EGM96) and WGS84 heights.
This file was taken from WesternMeteorPyLib, commit 5234439 (Dec 2020)"""

from __future__ import print_function, division, absolute_import

import os
import sys
import argparse

import numpy as np
import scipy.interpolate

from RMS.Misc import getRmsRootDir
from RMS.Decorators import memoizeAll


# Name of the EGM96 geoid data file shipped with RMS
EGM96_FILE_NAME = 'WW15MGH.DAC'

# str/unicode under both Python 2 and 3
try:
    STRING_TYPES = (str, unicode)
except NameError:
    STRING_TYPES = (str,)


def egm96DefaultPaths():
    """ Candidate locations of the EGM96 data file shipped with RMS, in search order.

    A source checkout keeps it under the RMS root, but a conda/pip install puts package data under
    sys.prefix instead, so both have to be considered.

    Return:
        [list of str] Candidate full paths to the EGM96 data file.
    """

    candidates = []

    # Source checkout: <RMS root>/share/WW15MGH.DAC
    try:
        candidates.append(os.path.join(getRmsRootDir(), 'share', EGM96_FILE_NAME))

    except ImportError:
        pass

    # Installed into an environment (e.g. conda): <sys.prefix>/share/WW15MGH.DAC
    candidates.append(os.path.join(sys.prefix, 'share', EGM96_FILE_NAME))

    return candidates


def egm96FilePath(egm96_source=None):
    """ Resolve the EGM96 data file path from any of the accepted source forms.

    Keyword arguments:
        egm96_source: [None/str/Config] None (default) uses the file shipped with RMS; a string is
            taken as the full path to the data file; anything else is treated as a Config instance
            and its egm96_path/egm96_file_name are joined.

    Return:
        [str] Full path to the EGM96 data file.
    """

    # A Config instance carries its own path to the data file
    if hasattr(egm96_source, 'egm96_path'):
        return os.path.join(egm96_source.egm96_path, egm96_source.egm96_file_name)

    # An explicit path is used as given
    if isinstance(egm96_source, STRING_TYPES):
        return egm96_source

    if egm96_source is not None:
        raise TypeError("egm96_source must be None, a path, or a Config instance, got {}".format(
            type(egm96_source).__name__))

    # Fall back to the file shipped with RMS, wherever this install keeps it
    candidates = egm96DefaultPaths()

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    # Nothing found - return the first candidate so the failure names a sensible path
    return candidates[0]


def loadEGM96Data(file_path=None, file_name=None):
    """ Load a file with EGM96 data.

    EGM96 data source: http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/binary/binarygeoid.html

    Keyword arguments:
        file_path: [str] Full path to the data file, or the containing directory when file_name is
            also given. None (default) uses the file shipped with RMS.
        file_name: [str] File name, for the legacy (dir_path, file_name) call form.
    """

    # Legacy two-argument form: (dir_path, file_name)
    if file_name is not None:
        file_path = os.path.join(file_path, file_name)

    else:
        file_path = egm96FilePath(file_path)

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


@memoizeAll
def geoidModel(file_path):
    """ Interpolated EGM96 geoid model for a data file, built once per path.

    Loading the array and constructing the 721x1440 RectSphereBivariateSpline costs ~0.09 s on a
    desktop and appreciably more on a Pi, so it must not happen per conversion. Keyed on the
    resolved file path, which is the only thing the model depends on.

    Arguments:
        file_path: [str] Full path to the EGM96 data file.

    Return:
        [RectSphereBivariateSpline] Interpolated geoid model.
    """

    return interpolateEGM96Data(loadEGM96Data(file_path=file_path))


def mslToWGS84Height(lat, lon, msl_height, egm96_source=None):
    """ Given the height above sea level (using the EGM96 model), compute the height above the WGS84
        ellipsoid.

    Arguments:
        lat: [float] Latitude +N (rad).
        lon: [float] Longitude +E (rad).
        msl_height: [float] Height above sea level (meters).

    Keyword arguments:
        egm96_source: [None/str/Config] Where to get the EGM96 data file. None (default) uses the
            file shipped with RMS, a string is taken as its full path, and a Config instance is read
            for egm96_path/egm96_file_name.

    Return:
        wgs84_height: [float] Height above the WGS84 ellipsoid.

    """

    # Interpolated geoid model, built once per data file
    GEOID_MODEL = geoidModel(egm96FilePath(egm96_source))

    # Get the difference between WGS84 and MSL height
    lat_mod = np.pi/2 - lat
    lon_mod = lon%(2*np.pi)
    msl_ht_diff = GEOID_MODEL(lat_mod, lon_mod)[0][0]

    # Compute the WGS84 height
    wgs84_height = msl_height + msl_ht_diff


    return wgs84_height


def wgs84toMSLHeight(lat, lon, wgs84_height, egm96_source=None):
    """ Given the height above the WGS84 ellipsoid compute the height above sea level (using the EGM96 model).

    Arguments:
        lat: [float] Latitude +N (rad).
        lon: [float] Longitude +E (rad).
        wgs84_height: [float] Height above the WGS84 ellipsoid (meters).

    Keyword arguments:
        egm96_source: [None/str/Config] Where to get the EGM96 data file. None (default) uses the
            file shipped with RMS, a string is taken as its full path, and a Config instance is read
            for egm96_path/egm96_file_name.

    Return:
        msl_height: [float] Height above sea level (meters).

    """

    # Interpolated geoid model, built once per data file
    GEOID_MODEL = geoidModel(egm96FilePath(egm96_source))

    # Get the difference between WGS84 and MSL height
    lat_mod = np.pi/2 - lat
    lon_mod = lon%(2*np.pi)
    msl_ht_diff = GEOID_MODEL(lat_mod, lon_mod)[0][0]

    # Compute the sea level
    msl_height = wgs84_height - msl_ht_diff


    return msl_height



if __name__ == "__main__":

    import RMS.ConfigReader as cr

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Convert mean sea level (EGM96) to WGS84")

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one.")

    arg_parser.add_argument('--egm96', type=str, \
        help="Path to the EGM96 data file. Overrides --config. Defaults to the file shipped with RMS.")

    arg_parser.add_argument('-i', '--inverse', action="store_true", \
            help="Convert WGS84 to EGM96 (default is False)")

    arg_parser.add_argument("latitude", type=float, help="Latitude in degrees (north is positive)")
    arg_parser.add_argument("longitude", type=float, help="Longitude in degrees (east is positive)")
    arg_parser.add_argument("height", type=float, help="Height to convert (in meters)")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    # An explicit data file path wins; otherwise use the config if one was given, else the shipped file
    if cml_args.egm96 is not None:
        egm96_source = cml_args.egm96

    elif cml_args.config is not None:
        egm96_source = cr.loadConfigFromDirectory(cml_args.config, ".")

    else:
        egm96_source = None

    # Load latitude and longitude
    lat = cml_args.latitude
    lon = cml_args.longitude

    if not cml_args.inverse:
        print("Converting MSL height to WGS84 height")
        msl_height = cml_args.height
        wgs84_height = mslToWGS84Height(np.radians(lat), np.radians(lon), msl_height, egm96_source)
    else:
        print("Converting WGS84 height to MSL height")
        wgs84_height = cml_args.height
        msl_height = wgs84toMSLHeight(np.radians(lat), np.radians(lon), wgs84_height, egm96_source)

    print('Latitude:', lat)
    print('Longitude', lon)
    print('MSL height (m): {:.2f}'.format(msl_height))
    print('WGS84 height (m): {:.2f}'.format(wgs84_height))
