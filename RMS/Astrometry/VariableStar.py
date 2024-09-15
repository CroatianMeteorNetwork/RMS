""" This module contains procedures for collating data from multiple stations in a database
    of magnitudes and can produce a chart of magnitudes close to radec coordinates
"""

# The MIT License

# Copyright (c) 2024

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)



from RMS.DeleteOldObservations import getNightDirs
import argparse
import os
import sys

import numpy as np
# Import Cython functions
import pyximport
from RMS.Astrometry.Conversions import date2JD, jd2Date, raDec2AltAz

pyximport.install(setup_args={'include_dirs':[np.get_include()]})

import RMS.ConfigReader as cr
import glob as glob
import sqlite3
import datetime
import json

from RMS.Formats.CALSTARS import readCALSTARS
from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, correctVignetting, photometryFitRobust
from RMS.Astrometry.FFTalign import getMiddleTimeFF
from RMS.Astrometry.ApplyAstrometry import extinctionCorrectionTrueToApparent, xyToRaDecPP
from RMS.Astrometry.CheckFit import matchStarsResiduals
from RMS.Formats.StarCatalog import readStarCatalog
from RMS.Astrometry.Conversions import datetime2JD
from RMS.Routines.MaskImage import loadMask
from RMS.Formats.FFfile import read
from RMS.Math import angularSeparationDeg
from RMS.Misc import getRmsRootDir, mkdirP



from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter


# Handle Python 2/3 compatibility
if sys.version_info.major == 3:
    unicode = str

EM_RAISE = True

def checkMaskxy(x, y, mask_path, mask=None):

    """
    Discover if mask obstructs points on image

    Args:
        x (): x image coordinates
        y (): y image coordinates
        mask_path (): path to the mask file

    Returns:
        True if mask does not obstruct the coordinates on the image else False
        If mask is not found, returns True
    """
    if mask is None:
        if os.path.exists(mask_path):
            if os.path.splitext(mask_path) == "bmp":
                m = loadMask(mask_path)
            else:
                path_name = os.path.join(mask_path, "mask.bmp")
                default_mask = os.path.join(getRmsRootDir(), "mask.bmp")
                if os.path.exists(path_name):
                    m = loadMask(os.path.join(mask_path, "mask.bmp"))
                elif os.path.exists(default_mask):
                    m = loadMask(os.path.join(default_mask))
                else:
                    return True
        else:
            return True
    else:
        m = mask

    if m.img[y, x] == 255:
        return True
    else:
        return False

def rmsTimeExtractor(rms_time, asTuple = False, asJD = False, delimiter = None):
    """
    General purpose function to convert *20240819*010235*123 | 123456 into a datetime object or JD
    Offsets can be given for the positions of date, time, and fractional seconds, however
    the code will try to parse any string that is given.


    Args:
        rms_time (): Any string containing YYYYMMDD and HHMMSS separated by the delimited
        asJD (): optional, default false, if true return julian date, if false return datetime object

    Returns:
        a datetime object or a julian date number

    """
    dt = None
    rms_time = os.path.basename(rms_time)
    # remove any dots, might be filename extension
    rms_time = rms_time.split(".")[0] if "." in rms_time else rms_time

    # Initialise delim in case nothing is detected
    delim = "_"
    # find the delimiter, which is probably the first non alpha numeric character
    if delimiter is None:
        for c in rms_time:
            if c.isnumeric() or c.isalpha():
                continue
            else:
                delim = c
                break
    if delim not in rms_time:
        return None

    field_list = rms_time.split(delim)
    field_count = len(field_list)
    str_us = "0"

    consecutive_time_date_fields = 0

    # Parse rms filename, datestring into a date time object
    for field, field_no in zip(field_list, range (0, field_count)):
        field = field.split(".")[0] if "." in field else field
        if field.isnumeric():
            consecutive_time_date_fields += 1

        # Handle year month day
        if consecutive_time_date_fields == 1:
            if len(field) == 8 or len(field) == 6:
                # This looks like a date field so process the date field
                str_date = field_list[field_no]
                if len(str_date) == 8:
                    year, month, day = int(str_date[:4]), int(str_date[4:6]), int(str_date[6:8])
                    dt = datetime.datetime(year=int(year), month=int(month), day=int(day))
                # Handle 2 digit year format
                if len(str_date) == 6:
                    year, month, day = 2000 + int(str_date[:2]), int(str_date[2:4]), int(str_date[4:6])
                    dt = datetime.datetime(year=int(year), month=month, day=day)
            else:
                dt = 0

        # Handle hour minute second
        if consecutive_time_date_fields == 2:
            if len(field) == 6:
                # Found two consecutive numeric fields followed by a non numeric
                # These are date and time
                str_time = field_list[field_no]
                hour, minute, second = int(str_time[:2]), int(str_time[2:4]), int(str_time[4:6])
                dt = datetime.datetime(year, month , day, hour, minute, second)
            elif len(field) == 4:
                str_time = field_list[field_no]
                hour, minute, second = int(str_time[:2]), int(str_time[2:4]), 0
                dt = datetime.datetime(year, month, day, hour, minute, second)
            else:
                # if the second field is not of length 6 then reset the counter
                consecutive_time_date_fields = 0

        # Handle fractional seconds
        if consecutive_time_date_fields == 3:
            if field.isnumeric():
                # Convert any arbitrary length next field to microseconds
                us = int(field) * (10 ** (6 - len(field)))
                dt = datetime.datetime(year, month, day, hour, minute, second, microsecond=int(us))
                # Stop looping in all cases
                break
            else:
                # Stop looping in call cases
                break

    if dt is None:
        return dt

    if asTuple:
        return dt, datetime2JD(dt)

    if asJD:
        return datetime2JD(dt)
    else:
        return dt

def plateparContainsRaDec(r, d, source_pp, file_name, mask_dir, check_mask=True, mask=None):
    """

    Args:
        r (float): right ascension (degrees)
        d (float): declination (degrees)
        source_pp (platepar): instance of the source platepar
        file_name (string): name of the fits file
        mask_dir (string): directory holding the mask
        check_mask (bool): check to see if radec is obstructed by the mask
        mask (object): optional, default None, save time by passing a mask in memory

    Returns:

    """

    # Get the image time from the file_name
    source_JD = rmsTimeExtractor(file_name, asJD=True)

    # Convert r,d to source image coordinates
    r_array = np.array([r])
    d_array = np.array([d])
    jd_arr = np.array([rmsTimeExtractor(file_name, asJD=True)])
    x_arr = np.array([source_pp.X_res / 2])
    y_arr = np.array([source_pp.Y_res / 2])
    level_arr = np.array([1])
    _, r_centre_pp, dec_centre_pp, _ = xyToRaDecPP(jd_arr, x_arr, y_arr, level_arr, source_pp, jd_time=True)

    # Check the angle, to prevent false positives for objects behind the camera
    angle_from_centre = angularSeparationDeg(r, d, r_centre_pp, dec_centre_pp)[0]

    # this prevents spurious coordinates being generated for r, d outside fov
    if angle_from_centre > max(source_pp.fov_h, source_pp.fov_v) / 2:
        return False, 0, 0

    source_x, source_y = raDecToXYPP(r_array, d_array, source_JD, source_pp)
    source_x, source_y = round(source_x[0]), round(source_y[0])

    if 0 < source_x < source_pp.X_res and 0 < source_y < source_pp.Y_res:
        if check_mask:
            if checkMaskxy(source_x,source_y, mask_dir, mask=mask):

                return True, source_x, source_y
            else:
                return False, 0, 0
        else:
            return True, source_x, source_y
    else:
        return False, 0, 0

def filterDirectoriesByJD(path, earliest_jd, latest_jd = None):

    """
    Returns a list of directories inclusive of the earliest and latest jd
    The earliest directory returned will be the first directory dated
    before the earliest jd.
    The latest directory returned will be the last directory dated before
    the latest jd

    Args:
        path (): path to ierate over
        earliest_jd (): directory of the earliest jd to include
        latest_jd (): directory of the latest jd to include

    Returns:
        filtered list of directories
    """
    latest_jd = earliest_jd if latest_jd is None else latest_jd
    directory_list = []
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return directory_list
    for obj in os.listdir(os.path.expanduser(path)):
        if os.path.isdir(os.path.join(path, obj)):
            directory_list.append(os.path.join(path, obj))

    directory_list.sort(reverse=True)

    filtered_by_jd = []
    for directory in directory_list:

        # Get the jd of this directory, if it can't be parsed then continue
        jd_dir = rmsTimeExtractor(directory, asJD=True)
        if jd_dir is None:
            continue

        # If the start time of this directory is less than the latest_target append to the list
        if jd_dir < latest_jd:
            filtered_by_jd.append(directory)

        # As soon as a directory has been added which is before the earliest_jd
        # stop appending break the loop; everything else has already been processed
        if rmsTimeExtractor(directory, asJD=True) < earliest_jd:
            break


    # Sort the list so that the oldest is at the top.
    filtered_by_jd.sort()

    return filtered_by_jd

def readInArchivedCalstars(config, conn):


    """
    Iterates over the ArchivedDirectories for the station to load all
    the calstar files into the database in radec format

    Args:
        config(): config instance
        conn(): database connection instance
    Returns:

    """

    # Load the star catalogue
    catalogue = loadGaiaCatalog("~/source/RMS/Catalogs", "gaia_dr2_mag_11.5.npy", lim_mag=11)

    # Deduce the path to the archived directories for this station
    archived_directories_path = os.path.join(config.data_dir, config.archived_dir)
    archived_directories = getNightDirs(archived_directories_path, config.stationID)

    # Reverse this list so that the newest directories are at the front
    archived_directories.reverse()

    # Find the most recent jd in the database for this station
    latest_jd = findMostRecentEntry(config, conn)

    # Initialise the calstar list
    calstar_list, archived_directories_filtered_by_jd = [], []

    # Iterate through the list of archived directories newest first
    # appending to the list of directories to be considered
    for directory in archived_directories:
        archived_directories_filtered_by_jd.append(directory)
        # As soon as a directory has been added which is before the latest_jd
        # stop appending break the loop; everything else has already been processed
        if rmsTimeExtractor(directory, asJD=True) < latest_jd:
            print("Excluding directories before {}, already processed for {}".format(
                                os.path.basename(directory), config.stationID))
            break

    # Reverse the list again, so that the oldest is at the top.
    archived_directories_filtered_by_jd.reverse()

    # Working with each of the remaining archived directories write into the database
    print("\nIterating through the archived directories starting from {}\n"
                                                        .format(archived_directories_filtered_by_jd[0]))
    for dir in archived_directories_filtered_by_jd:

        # Get full paths to critical files
        full_path = os.path.join(archived_directories_path, dir)
        full_path_calstars = glob.glob(os.path.join(full_path,"*CALSTARS*" ))
        full_path_platepar = glob.glob(os.path.join(full_path, "platepar_cmn2010.cal"))

        # If no platepar is found or no calstars, then ignore this directory
        if len(full_path_platepar) != 1 or len(full_path_calstars) != 1:
            continue

        full_path_platepar, full_path_calstars = full_path_platepar[0], full_path_calstars[0]
        calstars_path = os.path.dirname(full_path_calstars)
        calstars_name = os.path.basename(full_path_calstars)

        # Read in the CALSTARS file
        calstar = readCALSTARS(calstars_path, calstars_name)

        # Put the CALSTARS list into the database
        calstar_list.append(calstarToDb(calstar, conn, full_path, latest_jd))

def getCatalogueID(r, d, conn, margin=0.3):
    """
    Get the local for the brightest star within margin degrees of passed radec
    Args:
        r ():  right ascension (degreees)
        d (): declination (degrees)
        conn (): database connection
        margin (): optional, default 0.3, degrees margin. This is not a skyarea, simply a box in the
                    interest of computational efficiency

    Returns:
        tuple (id, magnitude, catalogue right ascension, catalogue declination)
    """

    sql_command = ""
    sql_command += "SELECT id, mag, r, d FROM catalogue \n"
    sql_command += "WHERE \n"
    sql_command += "r < {} AND r > {} AND d < {} AND d > {}\n".format(r+margin, r-margin, d+margin, d-margin)
    sql_command += "ORDER BY mag ASC\n"
    id = conn.cursor().execute(sql_command).fetchone()
    if id is not None:
        if len(id):
            return id
        else:
            return 0, 0, 0, 0
    else:
        return 0, 0, 0, 0

def computePhotometry(config, pp_all, calstar, match_radius=2.0, star_margin = 1.2):

    """
    Compute photometric offset and vignetting coefficient from CALSTARS
    Best practice is to use the vignetting coefficient from the platepar
    not a computed number

    Args:
        config (): configuration instance
        pp_all (): a dictionary of all recomputed platepars
        calstar (): calstar data structure
        match_radius (): the pixel radius used by the recalibration routine

    Returns:
        tuple(photometric offset, vignetting coefficient)
    """

    # Extract stars from the catalogue one order of magnitude dimmer than config limit
    lim_mag = config.catalog_mag_limit + 1
    catalog_stars, mag_band_str, config.star_catalog_band_ratios = \
                        readStarCatalog(config.star_catalog_path, config.star_catalog_file,
                                        lim_mag=lim_mag, mag_band_ratios=config.star_catalog_band_ratios)

    # star_dict contains the star data from calstars - indexed by jd
    # ff_name contains the fits file name - indexed by jd
    star_dict, ff_dict = {}, {}
    max_stars = 0
    ff_most_stars = None
    for entry in calstar:
        ff_name, star_data = entry
        d = getMiddleTimeFF(ff_name, config.fps, ret_milliseconds=True)
        jd = date2JD(*d)
        star_dict[jd], ff_dict[jd] = star_data, ff_name
        star_count = len(star_dict[jd])
        if star_count > max_stars:
            if ff_name in pp_all:
                max_stars, ff_most_stars, jd_most = star_count, ff_name, jd

    # As the purpose of this code is to get the best magnitude information, discard observation sessions where
    # too few stars were observed, returning none from here will discard the whole observation session
    pp = Platepar()
    if ff_most_stars is None or max_stars < config.min_matched_stars * star_margin:
        print("Too few stars, moving on")
        return None, None

    # Build a list of matched stars for photometry computations
    pp.loadFromDict(pp_all[ff_most_stars])
    n_matched, avg_dist, cost, matched_stars = matchStarsResiduals(config, pp, catalog_stars,
                                        {jd_most: star_dict[jd_most]}, match_radius, ret_nmatch=True,
                                                                   lim_mag=lim_mag)

    # If jd_most is not in matched stars, then do not use this observation session.
    # This is probably caused by too few stars

    if jd_most not in matched_stars:
        print("Key error, moving on")
        return None, None

    # Split the data return from matched_stars into image stars and catalogue stars
    image_stars, matched_catalog_stars, distances = matched_stars[jd_most]

    # Get the star intensities
    star_intensities = image_stars[:, 2]

    # Transpose the matched_catalog_stars array and extract ra, dec, mag
    cat_ra, cat_dec, cat_mags = matched_catalog_stars.T

    # For every star on the image compute the radius from image centre
    radius_arr = np.hypot(image_stars[:, 0] - pp.Y_res / 2, image_stars[:, 1] - pp.X_res / 2)

    # Correct for extinction
    mag_cat = extinctionCorrectionTrueToApparent(cat_mags, cat_ra, cat_dec, jd, pp)

    # Conduct the photometry fit, probably should do something with the standard deviation
    photom_params, fit_stddev, fit_resid, star_intensities, radius_arr, catalog_mags = \
        photometryFitRobust(star_intensities, radius_arr, mag_cat)

    return photom_params

def getFitsPathsAndCoords(config, earliest_jd, latest_jd, r=None, d=None):
    """

    Args:
        config (obj): config instance
        earliest_jd (float): earliest_jd to find
        latest_jd (float): latest_jd to find
        r (float):
        d (float):

    Returns:

    """


    full_path_to_captured = os.path.expanduser(os.path.join(config.data_dir, config.captured_dir))
    directories_to_search = filterDirectoriesByJD(full_path_to_captured, earliest_jd, latest_jd)
    stationID = config.stationID
    last_pp_path = ""
    fits_paths = []
    print("\nGetting all directories to be searched\n")
    for directory in directories_to_search:

        mask_path = os.path.join(directory, config.mask_file)
        default_mask_path = os.path.join(getRmsRootDir(), config.mask_file)
        if os.path.exists(mask_path):
            mask = loadMask(mask_path)
        elif os.path.exists(default_mask_path):
            mask = loadMask(default_mask_path)


        pp_path = os.path.join(directory, "platepar_cmn2010.cal")
        if os.path.exists(pp_path):
            pp = Platepar()
            pp.read(pp_path)
        else:
            # No platepar found - continue
            continue
        directory_list = os.listdir(directory)
        directory_list.sort()
        for file_name in directory_list:
            if file_name.startswith('FF') and file_name.endswith('.fits') and len(file_name.split('_')) == 6:
                if file_name.split('_')[1] == stationID:
                    if r is None and d is None:
                        contains_radec, x, y = True, 0, 0
                    else:
                        contains_radec, x, y = plateparContainsRaDec(r, d, pp, file_name,
                                                                    config.mask_file, check_mask=True, mask=mask)
                    if contains_radec:
                        fits_paths.append([os.path.join(directory,file_name), x, y])

    return fits_paths

def getFitsPaths(path_to_search, jd_start, jd_end=None):
    """
    Search for fits files between jd
    Args:
        path_to_search (string): path to search for fits files
        jd_start (): starting jd
        jd_end (): ending jd

    Returns:
        [list] list of paths to fits_files
    """

    directories_to_search = filterDirectoriesByJD(path_to_search, jd_start, jd_end)
    jd_end = jd_start if jd_end is None else jd_start

    fits_paths = []
    for directory in directories_to_search:
        directory_list = os.listdir(directory)
        directory_list.sort()
        for file_name in directory_list:
            if file_name.startswith("FF") and file_name.endswith(".fits"):
                file_jd = rmsTimeExtractor(file_name, asJD=True)
                if file_jd is not None:
                    if jd_start <= file_jd <= jd_end:
                        fits_paths.append(os.path.join(path_to_search, directory, file_name))

    return fits_paths

def readCroppedFF(path, x, y, width=20, height=20, allow_drift_in = False):
    """

    Args:
        path (): full path to the ff file to be read
        x (): x coordinates of the centre of the cropped region
        y (): y coordinates of the centre of the cropped region
        width (): optional, default 50, width of crop
        height (): optional, default 50, height of crop

    Returns:
        2D list of pixel intensities
    """



    ff = read(os.path.dirname(path), os.path.basename(path), memmap=False)

    if ff is None:
        return ff

    return crop(ff, x, y, width, height, allow_drift_in)

def crop(ff, x_centre, y_centre, width = 50, height = 50, allow_drift_in=False):
    """

    Args:
        ff (object): instance of a fits file
        x_centre (): x image coordinates fo the centre of the crop
        y_centre (): y image coordinates of the centre of the crop
        width (): width in pixels of the crop
        height (): height in pixels
        allow_drift_in (): optional, default false, if true fill the crop by offsetting the centre of the crop, if the
                            object is off the edge of the image. If true, then always keep the object image centre

    Returns:

    """

    # Get resolution
    x_res, y_res = ff.ncols, ff.nrows

    if allow_drift_in:
        # This allows an object to drift into the field of view
        # Establish where we can safely place the centre
        x_centre = max(0.5 * width, x_centre)
        x_centre = min(x_res - 0.5 * width, x_centre)
        y_centre = max(0.5 * height, y_centre)
        y_centre = min(y_res - 0.5 * height, y_centre)

        # Compute crop bounds
        x_min, y_min = round(x_centre - 0.5 * width), round(y_centre - 0.5 * height)
        x_max, y_max = x_min + width, y_min + height

        # Crop into a new array
        print("Cropping from {},{} to {},{}".format(x_min, x_max, y_min, y_max))
        ff_cropped = ff.maxpixel[y_min:y_max, x_min:x_max]

    else:
        # This always keeps the centre of the target in the centre of thumbnail
        # This allows an object to drift into the field of view
        # Establish where we can safely place the centre


        x_min, y_min = round(x_centre - 0.5 * width), round(y_centre - 0.5 * height)
        x_max, y_max = x_min + width, y_min + height

        if 0 < x_min and x_max < x_res and  0 < y_min and y_max < y_res:
            # This is the simple case, the cropped section is fully contained within the source
            ff_cropped = ff.maxpixel[y_min:y_max, x_min:x_max]
        else:
            # Create a new array of zero
            ff_cropped = np.zeros((height, width))
            for y_source in range (y_min, y_max):
                for x_source in range (x_min, x_max):
                    if 0 < y_source and y_source < y_res and 0 < x_source and x_source < x_res:
                        x_dest, y_dest = x_source - x_min, y_source - y_min
                        ff_cropped[y_dest, x_dest] = ff.maxpixel[y_source, x_source]

    return ff_cropped

def createThumbnails(config, r, d, earliest_jd=0, latest_jd=np.inf):
    """

    Args:
        config (object): config instance
        r (float): right ascension (degrees)
        d (float): declinration (degrees)
        earliest_jd (float): optional, default 0, create a subset of thumbnails
        latest_jd (float): optional_default 0

    Returns:
        list of thumbnails
    """

    # get the paths to all the fits files in the jd window
    path_coords_list = getFitsPathsAndCoords(config, earliest_jd, latest_jd, r, d)

    # initialise a list to hold the cropped image data
    thumbnail_list = []
    print("Iterating over paths")
    for fits_path, x, y in path_coords_list:
        thumbnail_list.append([fits_path, readCroppedFF(fits_path, x, y)])

    return thumbnail_list

def calstarToDb(calstar, conn, archived_directory_path, latest_jd=0):

    """
    Parses a calstar data structures in archived directories path,
    converts to RaDec, corrects magnitude data and writes newer data to database

    Args:
        calstar (): calstar data structure for one observation session
        conn (): connection to database
        archived_directory_path ():
        latest_jd (): optional, default 0, latest jd for this station in the database

    Returns:
        calstar_radec (): list of stellar magnitude data in radec format
    """

    # Intialise calstar_radec list
    calstar_radec = []

    # Get the path to all the recalibrated platepars for the night and read them in
    platepars_all_recalibrated_path = os.path.join(archived_directory_path, "platepars_all_recalibrated.json")
    with open(platepars_all_recalibrated_path, 'r') as fh:
        pp_recal = json.load(fh)

    # Compute photometry offset and vignetting using the best data from the night
    # vignetting coefficient will be overwritten by platepar value
    offset, vignetting = computePhotometry(config, pp_recal, calstar)

    # If this can't be computed, then probably the night was a poor observation session, so reject all
    if offset is None or vignetting is None:
        print("Nothing found in {}, moving on".format(archived_directory_path))
        return

    # Iterate through the calstar data structure for each image in the whole night
    # print("Iterating through calstar list for {}".format(rmsTimeExtractor(archived_directory_path)))
    for fits_file, star_list in calstar:

        # If too few stars on this specific observation, then ignore
        if len(star_list) < config.min_matched_stars:
            continue

        # Get the data and time of this observation
        date_time, jd = rmsTimeExtractor(fits_file, asTuple=True)
        # Skip anything which has already been processed
        if jd < latest_jd:
            continue

        # If this fits_file does not have a recalibrated platepar, then skip
        if not fits_file in pp_recal:
            continue

        # If it does, then load the recalibrated platepar for this image
        pp = Platepar()
        pp.loadFromDict(pp_recal[fits_file])

        # Overwrite vignetting coefficient with platepar value
        vignetting = pp.vignetting_coeff
        jd_list, y_list, x_list, bg_list, amp_list, FWHM_list = [], [], [], [], [], []

        # Build up lists of data for this image
        for y, x, bg_intensity, amplitude, FWHM in star_list:
            jd_list.append(jd)
            x_list.append(x)
            y_list.append(y)
            bg_list.append(bg_intensity)
            amp_list.append(amplitude)
            FWHM_list.append(FWHM)

        # Convert to arrays
        jd_arr, x_data, y_data, level_data = np.array(jd_list), np.array(x_list), np.array(y_list), np.array(amp_list)

        # Process data into RaDec and apply magnitude corrections
        jd, ra, dec, mag = xyToRaDecPP(jd_arr, x_data, y_data, level_data, pp,
                                                jd_time=True, extinction_correction=False, measurement=True)

        star_list_radec = []
        for j, x, y, r, d, bg, amp, FWHM, mag in zip(jd, x_list, y_list, ra, dec, bg_list, amp_list, FWHM_list, mag):
            cat_id, cat_mag, cat_r, cat_d = getCatalogueID(r, d, conn)
            az, el = raDec2AltAz(r, d, j, pp.lat, pp.lon)
            radius = np.hypot(y - pp.Y_res / 2, x - pp.X_res / 2)
            mag = 0 - 2.5 * np.log10(correctVignetting(bg, radius, vignetting)) + offset
            if mag == np.inf:
                continue
            star_list_radec.append([j, date_time, fits_file, x, y, az, el, r, d, bg, amp,
                                                                    FWHM, mag, cat_id, cat_mag, cat_r, cat_d])
        # Check that we still have enough stars and write to database
        if len(star_list_radec) > config.min_matched_stars:
            insertDB(config, conn, star_list_radec)

        # Add the data to the calstar_radec list
        calstar_radec.append([fits_file, star_list_radec])
    return calstar_radec

def insertDB(config, conn, star_list_radec):
    """
    Write data into the stellar magnitudes database
    Args:
        config (): config instance
        conn (): database connection
        star_list_radec (): star_list in radec format with corrected magnitudes

    Returns:

    """

    for jd, date_time, fits, x, y, az, el, r, d,  bg, amp, FWHM, mag, cat_id, cat_mag, cat_r, cat_d in star_list_radec:
        sql_command = ""
        sql_command += "INSERT INTO star_observations \n"
        sql_command += "(jd, date_time, station_id, fits, x, y, az, el, r, d, bg, amp, FWHM, mag, cat_key, cat_mag, cat_r, cat_d )\n"
        sql_command += "VALUES\n"
        sql_command += ("({}, '{}', '{}', '{}', {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})"
                        .format(jd, date_time, config.stationID, fits, x, y, az, el, r, d,
                                                                bg, amp, FWHM, mag, cat_id, cat_mag, cat_r, cat_d))
        conn.execute(sql_command)
    conn.commit()

def getStationStarDBConn(db_path, force_delete=False):
    """
    Get the connection to the stellar magnitude database, if it does not exist, then create
    Args:
        db_path (): full path to database
        force_delete (): optional, default false, delete and create

    Returns:
        conn (): connection object instance
    """
    # Create the station star database

    if force_delete:
        os.unlink(db_path)

    if not os.path.exists(os.path.dirname(db_path)):
        # Handle the very rare case where this could run before any observation sessions
        # and RMS_data does not exist
        os.makedirs(os.path.dirname(db_path))

    try:
        conn = sqlite3.connect(db_path, timeout=60)
        createTableStarObservations(conn)
        createTableCatalogue(conn)
        return conn

    except:
        return None

def retrieveMagnitudesAroundRaDec(conn, r,d, window=0.5, start_time=None, end_time=None):

    """
    Query the database on conn to find magnitudes around r, d. This might return more than one star
    Args:
        r (): right ascension in degrees
        d (): declination in degrees
        window(): window width in degrees
        start_time (): jd of start
        end_time (): jd of end

    Returns:
        list of tuples (jd, stationID, r, d, amp, mag, cat_mag)
    """
    window = abs(window)
    sql_command = ""
    sql_command += "SELECT jd, station_id, r, d, amp, mag, cat_mag\n"
    sql_command += "FROM star_observations\n"
    sql_command += "WHERE\n"
    sql_command += "r > {} AND r < {} AND\n".format(r - window, r + window, )
    sql_command += "d > {} AND d < {}".format(d - window, d + window)

    values = conn.cursor().execute(sql_command).fetchall()

    return values

def createTableStarObservations(conn):

    """
    If the star_observations table does not exist, then create
    Args:
        conn (): connection to database

    Returns:

    """
    table_name = "star_observations"
    # Returns true if the table exists in the database
    try:
        tables = conn.cursor().execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' and name = '{}';".format(table_name)).fetchall()

        if len(tables) > 0:
            return conn
    except:
        if EM_RAISE:
            raise
        return None

    sql_command = ""
    sql_command += "CREATE TABLE {} \n".format(table_name)
    sql_command += "( \n"
    sql_command += "id INTEGER PRIMARY KEY AUTOINCREMENT, \n"
    # j, x, y, r, d, bg, amp, FWHM, mag
    sql_command += "jd FLOAT NOT NULL, \n"
    sql_command += "date_time DATETIME NOT NULL, \n"
    sql_command += "station_id TEXT NOT NULL, \n"
    sql_command += "fits TEXT NOT NULL, \n"
    sql_command += "x FLOAT NOT NULL, \n"
    sql_command += "y FLOAT NOT NULL, \n"
    sql_command += "az FLOAT NOT NULL, \n"
    sql_command += "el FLOAT NOT NULL, \n"
    sql_command += "r FLOAT NOT NULL, \n"
    sql_command += "d FLOAT NOT NULL, \n"
    sql_command += "bg FLOAT NOT NULL, \n"
    sql_command += "amp FLOAT NOT NULL, \n"
    sql_command += "FWHM FLOAT NOT NULL, \n"
    sql_command += "mag FLOAT NOT NULL, \n"
    sql_command += "cat_mag FLOAT NOT NULL, \n"
    sql_command += "cat_r FLOAT NOT NULL, \n"
    sql_command += "cat_d FLOAT NOT NULL, \n"
    sql_command += "cat_key INT NOT NULL \n"


    sql_command += ") \n"
    conn.execute(sql_command)

    return conn

def createTableCatalogue(conn):

    """
    Creates the catalogue table if it does not exist
    Args:
        conn (): connection to database

    Returns:
        connection to database
    """
    table_name = "catalogue"
    # Returns true if the table exists in the database
    try:
        tables = conn.cursor().execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' and name = '{}';".format(table_name)).fetchall()

        if len(tables) > 0:
            return conn
    except:
        if EM_RAISE:
            raise
        return None

    sql_command = ""
    sql_command += "CREATE TABLE {} \n".format(table_name)
    sql_command += "( \n"
    sql_command += "id INTEGER PRIMARY KEY AUTOINCREMENT, \n"
    sql_command += "r FLOAT NOT NULL, \n"
    sql_command += "d FLOAT NOT NULL, \n"
    sql_command += "mag FLOAT NOT NULL \n"

    sql_command += ") \n"
    conn.execute(sql_command)
    catalogueToDB(conn)

    return conn

def findMostRecentEntry(config, conn):

    """
    Get the most recent entry for the station id in config object
    in the stellar magnitude database

    Args:
        config (): config instance
        conn (): connection instance

    Returns:
        jd of most recent entry for this station
    """

    sql_command = ""
    sql_command += "SELECT max(jd) FROM star_observations \n"
    sql_command += "WHERE \n"
    sql_command += "station_id = '{}' \n".format(config.stationID)

    jd = conn.cursor().execute(sql_command).fetchone()[0]
    if jd is not None:
        return jd
    else:
        return 0

def loadGaiaCatalog(dir_path, file_name, lim_mag=None):
    """ Read star data from the GAIA catalog in the .npy format.
        This function copied here to avoid reading in whole of SkyFit2

    Arguments:
        dir_path: [str] Path to the directory where the catalog file is located.
        file_name: [str] Name of the catalog file.

    Keyword arguments:
        lim_mag: [float] Faintest magnitude to return. None by default, which will return all stars.

    Return:
        results: [2d ndarray] Rows of (ra, dec, mag), angular values are in degrees.
    """

    file_path = os.path.expanduser(os.path.join(dir_path, file_name))

    # Read the catalog
    results = np.load(str(file_path), allow_pickle=False)

    # Filter by limiting magnitude
    if lim_mag is not None:
        results = results[results[:, 2] <= lim_mag]

    # Sort stars by descending declination
    results = results[results[:, 1].argsort()[::-1]]

    return results

def catalogueToDB(conn):
    """
    Read catalogue into database
    Args:
        conn (): connection instance

    Returns:
        Nothing
    """
    catalogue = loadGaiaCatalog("~/source/RMS/Catalogs", "gaia_dr2_mag_11.5.npy", lim_mag=11)
    print("\nInserting catalgue data\n")
    for star in catalogue:
        sql_command = "INSERT INTO catalogue (r , d, mag) \n"
        sql_command += "Values ({} , {}, {})".format(star[0], star[1], star[2])
        conn.execute(sql_command)
    conn.commit()

def createPlot(values, r, d, w=0):
    """

    Args:
        values (): list of values to be plotted as (jd, stationID, ra, dec, mag, cat_mag)
        r (): right ascension, used only for title
        d (): declination, used only for title
        w (): window, used only for title

    Returns:
        (object) plot object
    """

    x_vals, y_vals = [], []
    title = "Plot of magnitudes at RA {} Dec {}, window {}".format(r,d, w)
    for jd, stationID, r, d, amp, mag, cat_mag in values:
        x_vals.append(jd)
        y_vals.append(mag)
    f, ax = plt.subplots()

    plt.title(title)
    plt.grid()
    plt.ylabel("Magnitude")
    plt.xlabel("Julian Date")
    plt.ylim((min(y_vals) * 0.8, max(y_vals) * 1.2))
    ax.scatter(x_vals, y_vals)

    return ax

def areaToGoldenRatioXY(count, rotate=False):
    """
    Calculate dimensions close to the golden ratio for a given pixel count
    Args:
        count (float): pixel count
        rotate (): optional, default false, gives landscape format

    Returns:
        (x, y) tuple of integers
    """

    gr = (1 + 5 ** 0.5) / 2
    down = np.ceil((count * gr) ** 0.5)
    across = np.ceil(count / down)

    if rotate:
        return int(down), int(across)
    else:
        return int(across), int(down)

def assembleContactSheet(thumbnail_list, x_across=None, border=1):
    """
    Create a plot of the thumbnails in the the list
    Args:
        thumbnail_list (list): list of thumbnails
        x_across (integer): optional, pixel count across
        border (): optional, default 1 size of border between thumbnails


    Returns:
        contact_sheet_array (array): array of pixels
        headings_list: list of headings - the timestamp of the first image in each column
        position_list: images coordinates for the headings
    """

    thumbnail_count = len(thumbnail_list)
    if thumbnail_count < 1:
        return [], [], []

    fits, thumbnail = thumbnail_list[0]

    y_res, x_res = len(thumbnail) + border * 2, len(thumbnail[0]) + border * 2
    pixels = y_res * x_res * thumbnail_count
    if x_across is None:
        # We are free to calculate our own dimensions, so use golden ratio
        across, down = areaToGoldenRatioXY(pixels)
        across = int(np.ceil(across / x_res) * x_res)
        down = int(np.ceil(down / y_res) * y_res)

        #create an array of zeros
        contact_sheet_array = np.zeros((across, down))

        tn = 0
        headings_list, position_list = [], []
        for y in range(0, down, y_res):
            if tn == thumbnail_count:
                break
            fits, _ = thumbnail_list[tn]
            headings_list.append(rmsTimeExtractor(fits).strftime("%Y%m%d_%H%M%S"))
            position_list.append(y)
            for x in range(0, across, x_res):
                if tn == thumbnail_count:
                    break

                fits, thumbnail = thumbnail_list[tn]
                contact_sheet_array[x + border:x + x_res - border, y + border:y + y_res - border] = thumbnail
                tn += 1


    return contact_sheet_array, headings_list, position_list

def renderContactSheet(config, contact_sheet_array, headings_list, position_list, r, d, e_jd, l_jd, plot_format='png'):
    """

    Args:
        config (object): config instance
        contact_sheet_array (array): contact sheet array
        headings_list (list): list of headings
        position_list (list): position for the headings
        r (float): right asccension, only used for filenames and headings
        d (float): declination, only used for filenames and headings
        e_jd (float): earliest julian date
        l_jd (float): latest julian date
        plot_format (str): plot format

    Returns:
        plt (object): plot object
        filename (string): path to file
    """

    r, d = round(r, 2), round(d, 2)
    if len(contact_sheet_array) and len(headings_list) and len(position_list):
        plt.figure(figsize=(16, 12))
        axes = plt.gca()
        plot_filename = "r_{}_d_{}_jd_{}_{}_{}_contact_sheet.{}".format(r, d, e_jd, l_jd,
                                                                        config.stationID, plot_format)
        axes.imshow(contact_sheet_array, cmap='gray')
        start_time = rmsTimeExtractor(headings_list[0]).strftime("%Y-%m-%d %H:%M:%S")
        end_time = rmsTimeExtractor(headings_list[-1]).strftime("%Y-%m-%d %H:%M:%S")
        axes.set_title("{} RA {}, Dec {} from {} to {}".format(config.stationID, r, d, start_time, end_time))
        axes.title.set_size(20)
        plt.xticks(position_list, headings_list, color='black', fontweight='normal', fontsize='10',
                   horizontalalignment='center',  rotation=90)
        return plt, plot_filename
    else:
        return None, ""

def renderMagnitudePlot(config, magnitude_list, elevation_list, r, d, e_jd, l_jd, plot_format='png'):
    """
    Render the magnitude plot from information passed in
    Args:
        config (object): magnitude plot object
        magnitude_list (list): list of magnitudes
        elevation_list (list): list of elevations
        r (float): right ascension, only used for plot titles and filenames
        d (float): declination, only used for plot titles and filenames
        e_jd (float): earliest julian date, only used for plot titles and filenames
        l_jd (float): latest julian date, only used for plot titles and filenames
        plot_format (string): plot format

    Returns:
        plt (object): plot object
        filename (string): filename to save plot
    """


    if len(magnitude_list):
        x_vals, y_vals = [], []
        plot_filename = "r_{}_d_{}_jd_{}_{}_{}_magnitude.{}".format(r, d, e_jd, l_jd,
                                                                    config.stationID, plot_format)
        for jd, mag in magnitude_list:
            x_vals.append(jd2Date(float(jd), dt_obj=True))
            y_vals.append(mag)

        start_time, end_time = min(x_vals).strftime("%Y-%m-%d %H:%M:%S") , max(x_vals).strftime("%Y-%m-%d %H:%M:%S")
        title = "Plot of magnitudes at RA {} Dec {} from {} to {}".format(r, d, start_time, end_time)
        plt.figure(figsize=(areaToGoldenRatioXY(16 * 12, rotate=True)))

        plt.plot(marker='o', edgecolor='k', label='Elevation', s=100, c='none', zorder=3)
        ax = plt.gca()
        plt.scatter(x_vals, y_vals, c=elevation_list, zorder=3)
        plt.gca().invert_yaxis()
        plt.colorbar(label="Elevation from Horizontal (degrees)")
        seconds_of_observation = (max(x_vals) - min(x_vals)).total_seconds()
        interval_between_ticks = seconds_of_observation / 6
        tick_offsets = np.arange(0,seconds_of_observation,interval_between_ticks)
        x_tick_list = []
        date_form = DateFormatter("%Y-%m-%d %H:%M:%S")
        ax.xaxis.set_major_formatter(date_form)
        for offset in tick_offsets:
            x_tick_list.append((min(x_vals) + datetime.timedelta(seconds=offset)).strftime("%Y-%m-%d %H:%M:%S"))
        plt.gca().set_xticks(x_tick_list)
        plt.gca().set_xticklabels(x_tick_list, color='black', fontweight='normal', fontsize='10',
                           horizontalalignment='center')
        plt.xlabel("Time (UTC)")
        plt.ylabel("Magnitude")

        plt.title(title)


        return plt, plot_filename

def saveThumbnailsRaDec(config, r, d, e_jd=0, l_jd=np.inf, file_path=None):
    """
    Create and save thumbnails of the right ascension and declination between julian datres
    Args:
        config (object): config instance
        r (float): right ascension
        d (float): declination
        e_jd (float): earliest julian date
        l_jd (float): latest julian date
        file_path (path): path

    Returns:
        Nothing
    """

    thumbnail_list = createThumbnails(config, r, d, earliest_jd=e_jd, latest_jd=l_jd)
    contact_sheet, headings_list, position_list = assembleContactSheet(thumbnail_list)
    plt, fn = renderContactSheet(config, contact_sheet, headings_list, position_list, r, d, e_jd, l_jd)
    if plt is None:
        print("No transits found - cannot plot")
        return None
    else:
        filename = fn if file_path is None else file_path
        plt.savefig(filename)

def jsonToThumbnails(config, observations_json, r, d, e_jd, l_jd,  file_path=None):
    """
    Plot thumbnails from json information
    Args:
        config (object): config object
        observations_json (json): json of observation information
        r (float): right ascension (degreees)
        d (float): declination (degrees)
        e_jd (float):
        l_jd (float):
        file_path (string):

    Returns:
        filename (string)
    """

    thumbnail_list = []
    print(len(observations_json))
    for j in observations_json:
        observations = observations_json.get(j)
        thumbnail_list.append([observations['fits'], observations['pixels']])
        r = observations['coords']['equatorial']['ra']
        d = observations['coords']['equatorial']['dec']

    contact_sheet, headings_list, position_list = assembleContactSheet(thumbnail_list)
    plt, fn = renderContactSheet(config, contact_sheet, headings_list, position_list, r, d, e_jd, l_jd)
    if plt is None:
        print("No transits found - cannot plot")
        return None
    else:
        filename = fn if file_path is None else os.path.join(file_path, fn)
        plt.savefig(filename)

    return filename

def jsonToMagnitudePlot(config, observations_json, r, d, e_jd, l_jd, file_path=None):
    """
    From a json of magnitude information produce a magnitude plot
    Args:
        config (object): config instance
        observations_json (json): json of observations
        r (): right ascension (degrees)
        d (): declination (degrees)
        e_jd (): earliest julian date
        l_jd (): latest julian date
        file_path (): optional, path to save files

    Returns:

    """
    magnitude_list, elevation_list = [], []
    if not len(observations_json):
        return None

    for j in observations_json:
        observations = observations_json.get(j)
        magnitude_list.append([j,observations['photometry']['mag']])
        elevation_list.append(observations['coords']['horizontal']['el'])
        r = observations['coords']['equatorial']['ra']
        d = observations['coords']['equatorial']['dec']

    plt, fn = renderMagnitudePlot(config, magnitude_list, elevation_list,
                                  r=round(r, 2), d=round(d, 2),
                                  e_jd=e_jd, l_jd=l_jd)

    if plt is None:
        print("No observations found - cannot plot")
        return
    else:
        filename = fn if file_path is None else os.path.join(file_path, fn)
        plt.savefig(filename)

    return filename

def filterCalstarByJD(config, calstar, e_jd, l_jd):
    """
    Filter a calstar by julian date
    Args:
        config (object): config instance
        calstar (structure): calstar structure
        e_jd (float): earlest julian date
        l_jd (float): latest julian date

    Returns:
        filtered_fits (list): list of fits files and star informaton per fits file
    """
    filtered_fits = []
    for fits_file, star_list in calstar:

        # Get date_time jd of this file
        date_time, jd = rmsTimeExtractor(fits_file, asTuple=True)

        # Skip anything which is not in the time window
        if not (e_jd < jd < l_jd):
            continue

        # If too few stars on this specific observation, then ignore
        if len(star_list) < config.min_matched_stars:
            continue
        filtered_fits.append([fits_file, star_list])

    return filtered_fits

def filterDirByJD(directory_path, e_jd, l_jd):
    """
    Given a directory return a list of fits between jd limits
    Args:
        directory_path (string): full path to a directory
        e_jd (): earliest julian date
        l_jd (): latest julian date

    Returns:
        filtered_fits (list): list of fits files
    """
    filtered_fits = []
    directory_list = os.listdir(directory_path)
    for fits_file in directory_list:
        # Get date_time jd of this file
        if fits_file.startswith("FF") and fits_file.endswith(".fits"):
            date_time, jd = rmsTimeExtractor(fits_file, asTuple=True)
            # Skip anything which is not in the time window
            if not (e_jd < jd < l_jd):
                continue
            filtered_fits.append(fits_file)

    filtered_fits.sort()

    return filtered_fits

def calstarRaDecToDict(data_dir_path, config, pp, pp_recal_json, r_target, d_target, e_jd, l_jd, calstar,
                       search_sky_radius_degrees=0.3, centre_on_calstar_coords=True):
    """
      Parses a calstar data structures in archived directories path,
      converts to RaDec, corrects magnitude data and writes newer data to database

      Args:
          calstar (): calstar data structure for one observation session
          conn (): connection to database
          archived_directory_path ():
          latest_jd (): optional, default 0, latest jd for this station in the database

      Returns:
          calstar_radec (): list of stellar magnitude data in radec format
      """


    captured_directory_path = os.path.join(config.data_dir, config.captured_dir)
    candidate_fits = filterCalstarByJD(config, calstar, e_jd, l_jd)
    candidate_fits.sort()

    sequence_dict = dict()

    # Iterate through the candidate fits files
    for fits_file, star_list in candidate_fits:

        date_time, jd = rmsTimeExtractor(fits_file, asTuple=True)
        if pp_recal_json is not None:
            if fits_file in pp_recal_json:
                # If we have a platepar in pp_recal then use it, else just use the last platepar
                pp.loadFromDict(pp_recal_json[fits_file])

        containsRaDec, _ ,_ = plateparContainsRaDec(r_target, d_target, pp, fits_file, data_dir_path, check_mask= False)
        if not containsRaDec:
            continue
        # Overwrite vignetting coefficient with platepar value

        jd_list, y_list, x_list, bg_list, amp_list, FWHM_list = [], [], [], [], [], []

        # Build up lists of data for this image
        for y, x, bg_intensity, amplitude, FWHM in star_list:
            jd_list.append(jd)
            x_list.append(x)
            y_list.append(y)
            bg_list.append(bg_intensity)
            amp_list.append(amplitude)
            FWHM_list.append(FWHM)

        # Convert to arrays
        jd_arr, x_data, y_data, level_data = np.array(jd_list), np.array(x_list), np.array(y_list), np.array(bg_list)

        # Process data into RaDec and apply magnitude corrections
        jd, ra, dec, mag = xyToRaDecPP(jd_arr, x_data, y_data, level_data, pp,
                                       jd_time=True, extinction_correction=False, measurement=True)

        for j, x, y, r, d, bg, amp, FWHM, mag, x_cs, y_cs in zip(jd, x_list, y_list, ra, dec, bg_list,
                                                                        amp_list, FWHM_list, mag, x_list, y_list):
            az, el = raDec2AltAz(r, d, j, pp.lat, pp.lon)
            radius = np.hypot(y - pp.Y_res / 2, x - pp.X_res / 2)
            actual_deviation_degrees = angularSeparationDeg(r_target, d_target, r, d)
            vignetting, offset = pp.vignetting_coeff, pp.mag_lev
            vignetting_correction = correctVignetting(bg, radius, vignetting)
            print("Vignetting correction value {}").format(vignetting_correction)
            mag = 0 - 2.5 * np.log10(correctVignetting(bg, radius, vignetting)) + offset
            if mag == np.inf:
                continue

            if actual_deviation_degrees < search_sky_radius_degrees:
                path_to_ff = os.path.join(data_dir_path, fits_file)
                if os.path.exists(path_to_ff):
                    path_to_ff = path_to_ff

                else:
                    fits_time_jd = rmsTimeExtractor(path_to_ff, asJD = True)
                    path_to_ff = getFitsPaths(captured_directory_path, fits_time_jd)
                    if len(path_to_ff):
                        path_to_ff = path_to_ff[0]
                    else:
                        continue
            else:
                continue

            if centre_on_calstar_coords:
                x_centre, y_centre = round(x), round(y)
            else:
                x_unrounded_arr, y_unrounded_arr = raDecToXYPP(np.array([r]), np.array([d]), np.array([j]), pp)
                x_centre, y_centre = round(x_unrounded_arr[0]), round(y_unrounded_arr[0])

            if os.path.exists(path_to_ff):
                    observation_dict = {"fits": fits_file,
                                        "coords": {
                                            "image": {"x": x, "y": y},
                                            "horizontal": {"az": az, "el": el},
                                            "equatorial": {"ra": r, "dec": d}},
                                        "radius": radius,
                                        "photometry": {"bg": bg,
                                                       "amp": amp,
                                                       "FWHM": FWHM,
                                                       "mag": mag,
                                                       "p_offset": offset,
                                                       "p_vig": vignetting},
                                        "actual_deviation_degrees": actual_deviation_degrees,
                                        "pixels": readCroppedFF(path_to_ff, x_centre, y_centre).tolist()}
                    sequence_dict[j] = observation_dict



    return sequence_dict

def dirRaDecToDict(data_dir_path, pp, pp_recal, r_target, d_target, e_jd, l_jd):
    """
      Parses a calstar data structures in archived directories path,
      converts to RaDec, corrects magnitude data and writes newer data to database

      Args:
          calstar (): calstar data structure for one observation session
          conn (): connection to database
          archived_directory_path ():
          latest_jd (): optional, default 0, latest jd for this station in the database

      Returns:
          calstar_radec (): list of stellar magnitude data in radec format
      """


    candidate_fits = filterDirByJD(data_dir_path, e_jd, l_jd)
    mask = loadMask(os.path.join(data_dir_path, "mask.bmp"))
    sequence_dict = dict()
    for fits_file in candidate_fits:
        date_time, j = rmsTimeExtractor(fits_file, asTuple=True)
        if pp_recal is not None:
            if fits_file in pp_recal:
                pp.loadFromDict(pp_recal[fits_file])


        containsRaDec, x ,y = plateparContainsRaDec(r_target, d_target, pp, fits_file, data_dir_path,
                                                    check_mask=True, mask=mask)

        if not containsRaDec:
            continue

        path_to_ff = os.path.join(data_dir_path, fits_file)

        if os.path.exists(path_to_ff):
            az, el = raDec2AltAz(r, d, j, pp.lat, pp.lon)
            radius = np.hypot(y - pp.Y_res / 2, x - pp.X_res / 2)
            observation_dict = {"fits": fits_file,
                                "coords": {
                                    "image": {"x": x, "y": y},
                                    "horizontal": {"az": az, "el": el},
                                    "equatorial": {"ra": r, "dec": d}
                                    },
                                "radius": radius,
                                "pixels": readCroppedFF(path_to_ff, x, y).tolist()}
            sequence_dict[j] = observation_dict

    return sequence_dict

def jsonMagsRaDec(config, r, d, e_jd=0, l_jd=np.inf, require_calstar=True, require_recalibrated_platepar=True):
    """
    Given a radec jd range, search for intensity information.
    Initially search in archived files using calstars, then search latest captured dir using
    fits files

    """

    full_path_to_archived = os.path.expanduser(os.path.join(config.data_dir, config.archived_dir))
    full_path_to_default_platepar = os.path.join(getRmsRootDir(), "platepar_cmn2010.cal")


    directories_to_search = filterDirectoriesByJD(full_path_to_archived, e_jd, l_jd)
    observation_sequence_dict = {}
    for search_dir in directories_to_search:

        full_path = os.path.join(full_path_to_archived, search_dir)
        full_path_to_session_platepar = os.path.join(full_path, "platepar_cmn2010.cal")
        platepars_all_recalibrated_path = os.path.join(full_path, "platepars_all_recalibrated.json")


        pp = Platepar()
        if os.path.exists(platepars_all_recalibrated_path):
            pp_mid_recal = Platepar()
            with open(platepars_all_recalibrated_path, 'r') as fh:
                pp_recal_json = json.load(fh)
                if len(pp_recal_json):
                    midpoint = int(len(pp_recal_json) / 2)
                    for fits, i in zip(pp_recal_json, range(0,midpoint)):
                        pass
                    pp_mid_recal.loadFromDict(pp_recal_json[fits])
                    recalibrated_platepar_loaded = True
                else:
                    recalibrated_platepar_loaded = False
        else:
            pp_mid_recal = None
            recalibrated_platepar_loaded = False
            if require_recalibrated_platepar:
                continue

        # Read in a platepar in the following preference order
        # session platepar, default platepar, the middle recalibrated platepar
        # if no success with any of these, then continue to the next directory
        if os.path.exists(full_path_to_session_platepar):
            pp.read(full_path_to_session_platepar)
        elif os.path.exists(full_path_to_default_platepar):
            os.path.exists(full_path_to_default_platepar)
            pp.read(full_path_to_default_platepar)
        elif recalibrated_platepar_loaded:
            pp = pp_mid_recal
        else:
            continue

        # Read in the CALSTARS file
        if require_calstar:
            full_path_calstar = glob.glob(os.path.join(full_path, "*CALSTARS*"))[0]
            if len(full_path_calstar):
                calstars_path, calstars_name = os.path.dirname(full_path_calstar), os.path.basename(full_path_calstar)
            else:
                continue

            if os.path.exists(full_path_calstar):
                calstar = readCALSTARS(calstars_path, calstars_name)
            else:
                continue

            dict_from_calstar = calstarRaDecToDict(full_path, config, pp, pp_recal_json, r, d, e_jd, l_jd, calstar)
            if dict_from_calstar is not None:
                observation_sequence_dict.update(dict_from_calstar)

        else:
            dict_from_dir = dirRaDecToDict(full_path, pp, pp_recal_json, r, d, e_jd, l_jd)
            if dict_from_dir is not None:
                observation_sequence_dict.update(dict_from_dir)

    return observation_sequence_dict


def processStarTrackEvent(log, config, ev):
    """
    Interface intended to be used by EventMonitor

    Args:
        log (instance): logger instance
        config (instance): config instance
        ev (obj): event object

    Returns:
        list of files to be uploaded
    """

    require_calstar = True if ev.use_calstar == 1 else False

    log.info("Processing star track event")
    log.info("===========================")
    log.info("JD start        : {}".format(ev.jd_start))
    log.info("JD end          : {}".format(ev.jd_end))
    log.info("RMS Style time  : {}".format(ev.dt))
    log.info("RA              : {}".format(ev.star_ra))
    log.info("Dec             : {}".format(ev.star_dec))
    log.info("Use Calstar     : {}".format(require_calstar))
    log.info("Suffix          : {}".format(ev.suffix))
    file_list = []



    json_name = "r_{}_d_{}_jd_{}_{}_{}.json".format(ev.star_ra, ev.star_dec,
                                                            ev.jd_start, ev.jd_end,
                                                                config.stationID)

    ev.suffix = "radec" if ev.suffix == "event" else ev.suffix

    star_track_working_directory = os.path.join(config.data_dir, "TrackingFiles")
    mkdirP(star_track_working_directory)
    json_path = os.path.join(star_track_working_directory, json_name)


    observation_sequence_dict = jsonMagsRaDec(config, ev.star_ra, ev.star_dec,
                                              e_jd=ev.jd_start, l_jd=ev.jd_end,
                                              require_calstar=require_calstar)

    if not len(observation_sequence_dict):
        log.info("No observations of this sky region")
        return []
    else:
        log.info("Found {} observations of this sky region".format(len(observation_sequence_dict)))

    with open(json_path, 'w') as json_fh:
        json_fh.write(json.dumps(observation_sequence_dict, indent=4, sort_keys=True))



    with open(json_path, 'r') as json_fh:
        observation_sequence_dict = json.loads(json_fh.read())

    file_list.append(json_path)
    file_list.append(jsonToThumbnails(config, observation_sequence_dict,
                                      ev.star_ra, ev.star_dec,
                                      ev.jd_start, ev.jd_end,
                                      file_path=star_track_working_directory))

    file_list.append(jsonToMagnitudePlot(config, observation_sequence_dict,
                                         ev.star_ra, ev.star_dec,
                                         ev.jd_start, ev.jd_end,
                                         file_path=star_track_working_directory))

    csv_name = "r_{}_d_{}_jd_{}_{}_{}.csv".format(ev.star_ra, ev.star_dec,
                                                    ev.jd_start, ev.jd_end,
                                                    config.stationID)
    csv_path = os.path.join(star_track_working_directory, csv_name)

    file_list.append(flattenDict(observation_sequence_dict, csv_path))

    return file_list


def flattenDict(input_dict, output_path):
    """
    Convert any recursively defined dictionary into a csv file

    Args:
        input_dict (dict): input dict
        output_path (string): file path

    Returns:
        output_path (string): path which has been saved
    """
    output_lines_list = []
    csv_header = ""

    # Iterate through entries in input_dict writing out to a list
    for entry in input_dict:
        value_list, csv_header, key_stack = flattenDictEntry(input_dict[entry])
        output_lines_list.append(value_list)

    # Reverse the list
    output_lines_list.reverse()
    # Put the csv header at the end
    output_lines_list.append(csv_header)
    # And reverse again
    output_lines_list.reverse()

    # Write to file
    with open(output_path, 'w') as fh_out:
        for out_line in output_lines_list:
            fh_out.write("{}\n".format(out_line))

    return output_path



def flattenDictEntry(node, key_list=None, key_stack=None, value_list=None):
    """
    Recursively walk a dictionary structure
    Args:
        node (): a dictionary or a key value pair
        key_list (): optional, list of keys to be used as a header in csv format
        key_stack (): optional, stack of keys level1.level2.level3 - only used during recursion
        value_list (): optional, list of values is csv format

    Returns:
        value_list (): list of values in csv format
    """
    for key, value in node.items():

        if isinstance(value, dict):
            # Recurse into the sub dictionaries
            if key_stack is None:
                value_list, key_list, key_stack = flattenDictEntry(value, key_list, key, value_list)
            else:
                value_list, key_list, key_stack = flattenDictEntry(value, key_list, "{}.{}".format(key_stack, key), value_list)

        else:
            # First value in the line
            if value_list is None:
                value_list = value
            # Or add to the value line
            else:
                value_list = "{},{}".format(value_list, value)

            # First level key
            if key_stack is None:
                key_stack = key
            # Subsequent level key
            else:
                key_stack = "{}.{}".format(key_stack, key)

            # First key in the line
            if key_list is None:
                key_list = key_stack
            # Subsequent key in the line
            else:
                key_list = "{},{}".format(key_list, key_stack)

        # Moving onto the next key, value pair get rid of the last entry in the key stack
        if "."  in key_stack:
            key_stack = key_stack[:key_stack.rindex(".")]
        else:
            key_stack = None

    return value_list, key_list, key_stack


if __name__ == "__main__":

    #ent = {'actual_deviation_degrees': 0.009614816231951446, 'coords': {'equatorial': {'dec': -81.37185347767333, 'ra': 341.5240561630952}, 'horizontal': {'az': 171.36898381118533, 'el': 37.28865759260427}, 'image': {'x': 780.4, 'y': 450.49}}, 'fits': 'FF_AU0006_20240913_120002_552_0083968.fits', 'photometry': {'FWHM': 3.71, 'amp': 50, 'bg': 361, 'mag': 4.342539211658279, 'p_offset': 10.766061944799436, 'p_vig': 0.0007}, 'pixels': [[90, 93, 92, 94, 95, 92, 92, 91, 91, 91, 90, 89, 90, 92, 91, 91, 90, 92, 93, 93], [90, 91, 91, 95, 96, 92, 91, 90, 92, 95, 90, 90, 90, 92, 92, 91, 90, 92, 92, 92], [91, 91, 91, 94, 94, 90, 90, 89, 91, 94, 90, 90, 90, 92, 92, 91, 92, 94, 92, 92], [95, 91, 90, 90, 90, 88, 88, 89, 91, 91, 90, 91, 92, 92, 91, 90, 94, 96, 90, 90], [93, 92, 89, 92, 91, 90, 89, 93, 97, 93, 89, 91, 95, 95, 91, 91, 96, 96, 89, 89], [91, 91, 91, 91, 89, 89, 89, 92, 93, 90, 90, 91, 92, 93, 92, 93, 96, 94, 92, 90], [92, 89, 90, 90, 91, 89, 90, 92, 92, 92, 91, 91, 90, 91, 91, 92, 93, 94, 94, 90], [93, 95, 93, 92, 93, 91, 92, 93, 90, 90, 92, 93, 90, 90, 93, 93, 93, 95, 94, 93], [92, 96, 93, 96, 92, 92, 90, 97, 92, 91, 95, 97, 96, 94, 98, 98, 96, 92, 91, 91], [91, 94, 96, 95, 92, 90, 92, 97, 95, 97, 98, 104, 98, 99, 94, 93, 92, 92, 90, 90], [92, 94, 99, 97, 90, 88, 91, 90, 90, 97, 137, 151, 109, 98, 95, 93, 92, 91, 89, 89], [92, 94, 100, 100, 91, 89, 95, 92, 94, 106, 152, 169, 142, 111, 95, 94, 93, 93, 93, 90], [91, 93, 98, 94, 91, 91, 93, 99, 94, 107, 130, 139, 126, 112, 95, 95, 98, 94, 90, 91], [91, 92, 93, 92, 90, 90, 89, 94, 94, 99, 104, 103, 99, 93, 95, 94, 94, 93, 90, 90], [91, 92, 91, 89, 89, 89, 91, 92, 91, 94, 91, 94, 92, 92, 92, 91, 91, 90, 88, 93], [95, 93, 91, 89, 89, 90, 95, 91, 98, 90, 90, 93, 91, 93, 92, 92, 91, 90, 95, 91], [92, 92, 92, 90, 90, 95, 94, 91, 93, 96, 90, 91, 90, 90, 94, 94, 90, 91, 93, 91], [91, 91, 90, 90, 91, 95, 94, 90, 91, 92, 90, 89, 89, 91, 95, 91, 92, 94, 92, 91], [90, 91, 89, 91, 91, 92, 92, 90, 89, 89, 89, 88, 88, 90, 91, 90, 92, 94, 92, 93], [90, 91, 91, 94, 91, 92, 92, 90, 90, 90, 89, 88, 87, 91, 89, 88, 88, 92, 97, 92]], 'radius': 167.03472722760378}
    #value_list, key_list, key_stack  = flattenDictEntry(ent)

    # Init the command line arguments parser


    description = "Iterate over archived directories, using the CALSTARS file to generate\n"
    description += "a database of stellar magnitudes against RaDec\n\n"
    description += "For multicamera operation, either start this as a process in each camera\n"
    description += "user account, pointing to the same database location\n"
    description += "Or run multiple proceses in one account, pointing to each cameras config file\n"

    arg_parser = argparse.ArgumentParser(description=description)

    arg_parser.add_argument('-r', '--ra', nargs=1, metavar='RA', type=float,
                            help="Right ascension to plot")

    arg_parser.add_argument('-d', '--dec', nargs=1, metavar='DEC', type=float,
                            help="Declination to plot")

    arg_parser.add_argument('-w', '--window', nargs=1, metavar='WINDOW', type=float,
                            help="Width to plot")

    arg_parser.add_argument("-p", '--dbpath', nargs=1, metavar='DBPATH', type=str,
                            help="Path to Database")

    arg_parser.add_argument("-c", '--config', nargs=1, metavar='CONFIGPATH', type=str,
                            help="Config file to load")

    arg_parser.add_argument("-f", '--format', nargs=1, metavar='FORMAT', type=str,
                            help="Chart output format - default png")

    arg_parser.add_argument("-j", '--jd_range', nargs=2, metavar='FORMAT', type=float,
                            help="Range of julian dates to plot")

    arg_parser.add_argument("-t", '--thumbnails', action="store_true",
                            help="Plot thumbnails around Radec")

    arg_parser.add_argument("-n", '--no_read', action="store_true",
                            help="Do not try to populate the database")





    # Parse the command line arguments
    cml_args = arg_parser.parse_args()
    if cml_args.config is None:
        config_path = "~/source/RMS/.config"
    else:
        config_path = cml_args.config[0]
    config_path = os.path.expanduser(config_path)
    config = cr.parse(config_path)

    if cml_args.format is None:
        plot_format = "png"
    else:
        plot_format = cml_args.format[0]

    if plot_format not in ['png', 'jpg', 'bmp']:
        plot_format = 'png'






        pass

    else:


        r, d = cml_args.ra[0], cml_args.dec[0]
        e_jd, l_jd = cml_args.jd_range[0], cml_args.jd_range[1]

        print("RaDec {},{} jd {} to {}".format(r, d, e_jd, l_jd))


        observation_sequence_dict = jsonMagsRaDec(config, r, d, e_jd=e_jd, l_jd=l_jd)

        observation_sequence_json = json.dumps(observation_sequence_dict, indent=4, sort_keys=True)
        with open("observation_sequence.json", 'w') as fh_observation_sequence_json:
            fh_observation_sequence_json.write(observation_sequence_json)

        with open("observation_sequence.json", 'r') as fh_observation_sequence_json:
            observation_sequence_json = json.loads(fh_observation_sequence_json.read())

        jsonToThumbnails(observation_sequence_json, r, d, e_jd, l_jd)
        jsonToMagnitudePlot(observation_sequence_json, r, d, e_jd, l_jd)

        if cml_args.thumbnails:
            saveThumbnailsRaDec(config, r, d, e_jd, l_jd)


        if cml_args.window is None:
            w = 0.1
        else:
            w = cml_args.window[0]






    exit()

    if cml_args.dbpath is None:
        dbpath = "~/RMS_data/magnitudes.db"
    else:
        dbpath = cml_args.dbpath


    dbpath = os.path.expanduser(dbpath)
    conn = getStationStarDBConn(dbpath)
    if cml_args.no_read:
        print("Skipping database population, no read selected")
    else:
        print("Started database population")
        archived_calstars = readInArchivedCalstars(config, conn)



        print("Producing plot around RaDec {}, {} width {}".format(r, d, w))

        values = retrieveMagnitudesAroundRaDec(conn, r, d, window=w)
        ax = createPlot(values, r, d, w)
        ax.plot()
        plt.savefig("magnitudes_at_Ra_{}_Dec_{}_Window_{}.{}".format(r, d, w, plot_format), format=plot_format)


