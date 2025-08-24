# RPi Meteor Station
# Copyright (C) 2025 David Rollinson Kristen Felker
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function

import os
import sys
import pickle
import argparse
import subprocess
import cv2
import numpy as np


import RMS.ConfigReader as cr
import datetime
import pathlib
import time
import imageio as imageio
import tqdm

from RMS.Astrometry.Conversions import altAz2RADec, raDec2AltAz, jd2Date, date2JD, J2000_JD
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, raDecToXYPP, correctVignetting
from RMS.Formats.FFfile import read as readFF
from RMS.Formats.Platepar import Platepar
from RMS.Math import angularSeparationDeg
from RMS.Misc import mkdirP
from RMS.Routines.MaskImage import loadMask
from RMS.Astrometry.CyFunctions import equatorialCoordPrecession


def cartesianToAltAz(x, y, dimension_x_min, dimension_x_max, dimension_y_min, dimension_y_max,  minimum_elevation_deg):

    """
    Convert Cartesian coordinates (x, y) on a polar plot to azimuth and altitude angles.

    Arguments:
        x: [int] x coordinate
        y: [int] y coordinate
        dimension_x_min: [int] minimum x value
        dimension_x_max: [int] maximum x value
        dimension_y_min: [int] minimum y value
        dimension_y_max: [int] maximum y value
        minimum_elevation_deg: [float] minimum elevation degrees

    Return:
        alt_deg   : Altitude angle in degrees (from horizon up)
        az_deg    : Azimuth angle in degrees (0° = right, 90° = up)

    """
    # Normalize coordinates to center
    x0 = (dimension_x_min + dimension_x_max) / 2
    y0 = (dimension_y_min + dimension_y_max) / 2
    dx = x - x0
    dy = (dimension_y_max - y) - y0

    # Compute azimuth (angle around center)
    az_rad = np.arctan2(dy, dx)
    az_deg = np.degrees(az_rad) % 360

    # Compute radial distance from center
    r = np.sqrt(dx ** 2 + dy ** 2)
    rmax = np.sqrt(((dimension_x_max - x0) ** 2 + (dimension_y_max - y0) ** 2))

    # Map radius to altitude (center = 90°, edge = min_elev_deg)
    alt_deg = 90 - (90 - minimum_elevation_deg) * (r / rmax)
    alt_deg = np.clip(alt_deg, minimum_elevation_deg, 90)

    return alt_deg, (az_deg - 90) % 360


def altAzToCartesian(az_deg, alt_deg, dimension_x_min, dimension_x_max, dimension_y_min, dimension_y_max,  minimum_elevaton_deg):
    """
    Convert azimuth and altitude angles to Cartesian coordinates on a polar plot.

    Arguments:
        alt_deg   : Altitude angle in degrees (from horizon up)
        az_deg    : Azimuth angle in degrees (0° = right, 90° = up)
        dimension_x_min: [int] minimum x value
        dimension_x_max: [int] maximum x value
        dimension_y_min: [int] minimum y value
        dimension_y_max: [int] maximum y value
        minimum_elevation_deg: [float] minimum elevation degrees

    Return:
        x: [int] x coordinate
        y: [int] y coordinate
    """

    az_deg += 90

    # Center of the plot
    x0 = (dimension_x_min + dimension_x_max) / 2
    y0 = (dimension_y_min + dimension_y_max) / 2

    # Max radius from center to edge
    rmax = np.sqrt((dimension_x_max - x0) ** 2 + (dimension_y_max - y0) ** 2)

    # Convert altitude to radial distance
    r = rmax * (90 - alt_deg) / (90 - minimum_elevaton_deg)

    # Convert azimuth to angle in radians
    az_rad = np.radians(az_deg)

    # Compute Cartesian coordinates
    x = x0 + r * np.cos(az_rad)
    y = y0 - r * np.sin(az_rad)



    return x, y


def getStationsInfoDict(path_list=None, print_activity=False):

    """
    Either load configs from a given path_list, or look for configs in a multi_cam linux
    or single camera per username style architecture.

    Keyword arguments:
        path_list: [list] List of paths to config files.
        print_activity: [bool] Optional, default false, print activity for debugging.

    Return:
        stations_info_dict: [dict] dictionary with station name as key and station config as value.
    """


    # Initialise an empty dict
    stations_info_dict = {}


    # If we have been given paths
    if len(path_list):
        if print_activity:
            print("Command line gave path lists  :")
        for p in path_list:
            if print_activity:
                print("                                 {}".format(p))
            station_info = {}
            if p.endswith('.config'):
                station_full_path = os.path.dirname(p)
                c = cr.parse(os.path.expanduser(p))
            else:
                station_full_path = p
                c = cr.parse(os.path.expanduser(os.path.join(p,".config")))

            platepar_full_path = os.path.join(station_full_path, c.platepar_name)
            mask_full_path = os.path.join(station_full_path, c.mask_file)

            if os.path.exists(platepar_full_path):
                pp = Platepar()
                pp.read(platepar_full_path)
            else:
                pp = None
                continue

            if os.path.exists(mask_full_path):
                m = loadMask(mask_full_path).img
            else:
                m = None

            data_dir_sections = pathlib.Path(c.data_dir).parts
            config_path_section = pathlib.Path(p).parts
            if "home" in data_dir_sections:
                user_name_index = data_dir_sections.index("home") + 1

                i = 0
                for section in data_dir_sections:

                    if i == user_name_index:
                        c.data_dir = os.path.join(c.data_dir, config_path_section[user_name_index])
                    else:
                        c.data_dir = os.path.join(c.data_dir, section)
                    i += 1

            station_info['mask'] = m
            station_info['pp'] = pp
            station_info['config'] = c
            stations_info_dict[c.stationID.lower()] = station_info

        return stations_info_dict


    else:
        # Test if this is a multicam linux station, i.e. ~/source/Stations/XX0001
        stations_base_directory = os.path.expanduser("~/source/Stations/")
        if os.path.exists(stations_base_directory):
            if os.path.isdir(stations_base_directory):
                candidate_stations_list = sorted(os.listdir(stations_base_directory))
                if print_activity:
                    print("Searching for multicam linux configs found :")
                for station in candidate_stations_list:
                    if print_activity:
                        print("                                             {}".format(station))
                    station_info = {}
                    if len(station) == 6:
                        station_full_path = os.path.join(stations_base_directory, station)
                        config_path = os.path.join(station_full_path, ".config")
                        if os.path.exists(config_path):
                            c = cr.parse(config_path)
                            if c.stationID.lower() != user_name.lower():
                                continue
                            platepar_full_path = os.path.join(station_full_path, c.platepar_name)
                            mask_full_path = os.path.join(station_full_path, c.mask_file)
                            if os.path.exists(platepar_full_path):
                                pp = Platepar()
                                pp.read(platepar_full_path)
                            else:
                                pp = None
                                continue

                            if os.path.exists(mask_full_path):
                                m = loadMask(mask_full_path).img
                            else:
                                m = None

                            station_info['mask'] = m
                            station_info['pp'] = pp
                            station_info['config'] = c
                            stations_info_dict[c.stationID.lower()] = station_info

        # If we found configs, then return what we have found
        if len(stations_info_dict):
            return stations_info_dict

    # Test if this is a one camera per user system
    stations_info_dict = {}
    if print_activity:
        print("Looking for a one camera per username style architecture")
    if not len(path_list):
        stations_base_directory = ("/home")
        if os.path.exists(stations_base_directory):
            if os.path.isdir(stations_base_directory):
                candidate_stations_list = sorted(os.listdir(stations_base_directory))
                for user_name in candidate_stations_list:
                    station_info = {}
                    if len(user_name) == 6:
                        if print_activity:
                            print("Testing {} to see if it a camera account".format(user_name))
                        if user_name[0:2].isalpha():
                            station_full_path = os.path.join(stations_base_directory, user_name)
                            station_config_dir = os.path.join(station_full_path, "source","RMS")
                            config_path = os.path.join(station_config_dir, ".config")
                            if print_activity:
                                print("Looking for {}".format(config_path))

                            if os.path.exists(config_path):
                                if print_activity:
                                    print("Found {}".format(config_path))
                                c = cr.parse(config_path)
                                if c.stationID.lower() == user_name:
                                    platepar_full_path = os.path.join(station_config_dir, c.platepar_name)
                                    mask_full_path = os.path.join(station_config_dir, c.mask_file)
                                    data_dir_sections = pathlib.Path(c.data_dir).parts
                                    if "home" in data_dir_sections:
                                        user_name_index = data_dir_sections.index("home") + 1

                                        i = 0
                                        for section in data_dir_sections:

                                            if i == user_name_index:
                                                c.data_dir = os.path.join(c.data_dir, user_name)
                                            else:
                                                c.data_dir = os.path.join(c.data_dir, section)
                                            i += 1
                                        if os.path.exists(platepar_full_path):
                                            pp = Platepar()
                                            pp.read(platepar_full_path)
                                            if pp.station_code == c.stationID:
                                                station_info['pp'] = pp
                                                if print_activity:
                                                    print("Loaded platepar for {}".format(user_name))
                                            else:
                                                station_info['pp'] = None
                                                if print_activity:
                                                    print("No platepar for {}".format(user_name))
                                        else:
                                            if print_activity:
                                                print("No platepar for {}".format(user_name))
                                            station_info['pp'] = None
                                    if os.path.exists(mask_full_path):
                                        m = loadMask(mask_full_path).img
                                        if print_activity:
                                            print("Loaded mask for {}".format(user_name))
                                    station_info['mask'] = m
                                    station_info['config'] = c
                                    stations_info_dict[c.stationID.lower()] = station_info
                            else:
                                if print_activity:
                                    print("No config found at {}".format(config_path))

        if len(stations_info_dict):
            return stations_info_dict

    return stations_info_dict


def makeTransformation(stations_info_dict, size_x, size_y, minimum_elevation_deg=20, stack_depth=3, time_steps_seconds=256 / 25, print_activity=False):

    """
    Make the transformation from the image coordinates of multiple cameras to the image coordinates of a destination
    polar project image of the sky.

    The calculations for transforming images through time are not robust, and should only be used for short
    offsets, not more than 10 hours. Over these durations the error will be acceptable for producing images.

    Arguments:
        stations_info_dict: [dict] Dictionary of station information.
        size_x: [int] X size of the image in pixels.
        size_y: [int] Y size of the image in pixels.

    Keyword arguments:
        minimum_elevation_deg:[float] Optional, default 20, minimum elevation angle in degrees.
        stack_depth:[int] Optional, default 3, number of fits files to get for stacking.
        time_steps_seconds:[int] Optional, default 256 / 25, number of seconds between images to use for stacking.

    Return:
        stations_list: [list] List of stations.
        source_coordinates_array: [Array] Array of source coordinates first column is the camera index.
        dest_coordinates_array: [array] Array of destination coordinates.
        intensity_scaling_array: [array] Array of number of source pixels mapped to a destination pixel.
    """

    # Intialise
    origin_x, origin_y = size_x / 2, size_y / 2
    elevation_range = 2 * (90 - minimum_elevation_deg)
    pixel_to_radius_scale_factor_x = elevation_range / size_x
    pixel_to_radius_scale_factor_y = elevation_range / size_y
    az_vals_deg, el_vals_deg = np.zeros((size_x, size_y)), np.zeros((size_x, size_y))

    # Make transform from target image coordinates to az and el.
    stations_list = sorted(list(stations_info_dict.keys()))

    # Define target image parameters
    pp_target = stations_info_dict[stations_list[0]]['pp']
    target_lat, target_lon, target_ele = pp_target.lat, pp_target.lon, pp_target.elev

    # Define source parameters
    source_coordinates_list, dest_coordinates_list, scaling_list, combined_coordinates_list = [], [], [], []
    transformation_layer_list, transformation_layer = [], 0

    # Form the transformation, working across stations, and then stacked images

    station_stack_count_list = []
    for station in stations_info_dict:
        for stack_count in range(0, stack_depth):
            station_stack_count_list.append([station, stack_count])

    for station, stack_count in station_stack_count_list:
        # Get the source platepar
        pp_source = stations_info_dict[station]['pp']

        # Compute the time offset - +ve is always forward in time
        time_offset_seconds = 0 - time_steps_seconds * stack_count
        offset_date = jd2Date(pp_source.JD, dt_obj=True) + datetime.timedelta(seconds=time_offset_seconds)

        # Convert the source image time to Julian Date relative to platepar reference
        jd_source = date2JD(offset_date.year, offset_date.month, offset_date.day, offset_date.hour, offset_date.minute, offset_date.second, int(offset_date.microsecond / 1000))

        if print_activity:
            print("Making transformation for {:s} with a time offset of {:.1f} seconds - {}".format(station.lower(), 0 - time_offset_seconds, jd2Date(jd_source)))

        # Get the centre of the platepar at creation time in JD - not compensated for time offsets
        _, r_source, d_source, _ = xyToRaDecPP([pp_source.JD], [pp_source.X_res / 2], [pp_source.Y_res / 2], [1], pp_source, jd_time=True, extinction_correction=False, measurement=False)
        r_list, d_list, x_dest_list, y_dest_list = [], [], [], []

        for y_dest in range(1, size_y - 1):
            for x_dest in range(1, size_x - 1):
                _x, _y, = x_dest - origin_x, y_dest - origin_y

                # Convert the target image (polar projection on cartesian axis) into azimuth and elevation
                el_deg = 90 - np.hypot(_x * pixel_to_radius_scale_factor_x, _y * pixel_to_radius_scale_factor_y)
                az_deg = np.degrees(np.arctan2(_x, _y))

                # el_deg, az_deg = cartesianToAltAz(x_dest, y_dest, 0, size_x, 0, size_y, minimum_elevation_deg)


                # print(x_dest, y_dest )

                # Store
                az_vals_deg[y_dest, x_dest], el_vals_deg[y_dest, x_dest] = az_deg, el_deg

                # Convert to ra and dec at the destination, including any time offset
                r_dest, d_dest = altAz2RADec(az_deg, el_deg, jd_source, target_lat, target_lon)

                # This time delta changes the position of the source pixel
                ang_sep_deg = angularSeparationDeg(r_dest, d_dest, r_source, d_source)

                # Is this still in the FoV
                if ang_sep_deg > np.hypot(pp_source.fov_h, pp_source.fov_v) / 2:
                    continue

                # Compute radec from azimuth and elevation at original platepar time
                r, d = altAz2RADec(az_deg, el_deg, pp_source.JD, pp_source.lat, pp_source.lon)
                r_list.append(r)
                d_list.append(d)
                x_dest_list.append(size_x - x_dest)
                y_dest_list.append(size_y - y_dest)

        # Compute source image pixels with the time offset
        x_source_array, y_source_array = raDecToXYPP(np.array(r_list), np.array(d_list), jd_source, pp_source)


        for x_source_float, y_source_float, x_dest, y_dest in zip(x_source_array, y_source_array, x_dest_list, y_dest_list):

            x_source, y_source = int(x_source_float), int(y_source_float)
            if not (5 < x_source < (pp_source.X_res - 5) and 5 < y_source < (pp_source.Y_res - 5)):
                continue

            m = stations_info_dict[station]['mask']
            if m is not None:
                if m[y_source, x_source] != 255:
                    continue

            station_index = stations_list.index(station)
            radius = np.hypot((x_source - pp_source.X_res) ** 2, (y_source - pp_source.Y_res) ** 2) ** 0.5
            vignetting_factor = correctVignetting(1,  radius, pp_source.vignetting_coeff)
            source_coordinates_list.append([int(transformation_layer), int(x_source), int(y_source), vignetting_factor])
            dest_coordinates_list.append([x_dest, y_dest])

        transformation_layer_list.append([station, time_offset_seconds])
        transformation_layer += 1

    source_coordinates_array = np.array(source_coordinates_list)
    dest_coordinates_array = np.array(dest_coordinates_list)

    pairs, counts = np.unique(dest_coordinates_array, axis=0, return_counts=True)
    intensity_scaling_array = np.zeros((size_x, size_y))
    for pair, count in zip(pairs, counts):
        intensity_scaling_array[pair[1]][pair[0]] = count

    return transformation_layer_list, source_coordinates_array, dest_coordinates_array, intensity_scaling_array, [target_lat, target_lon, target_ele]

def getFitsFiles(transformation_layer_list, stations_info_dict, target_image_time, print_activity=False):
    """
    Get the paths to fits files, in the same order as stations_list using info from stations_info_dict around target_image time.

    Arguments:
        transformation_layer_list: [[list]] list of [stations, time offsets]
        stations_info_dict: [dict] dictionary of station information.
        target_image_time: [datetime] target time for image. The closest fits files to this time will be selected.

    Return:
        station_files_list:[[list]] list of [station, path to fits file]
    """


    stations_files_list = []
    for s, time_offset_seconds in transformation_layer_list:
        if print_activity:
            print("Looking for fits files in station {} time offset {} from {}".format(s, time_offset_seconds, target_image_time))
        c = stations_info_dict[s]['config']
        captured_dir_path = os.path.join(c.data_dir, c.captured_dir)
        captured_dirs = sorted(os.listdir(captured_dir_path), reverse=True)
        if not len(captured_dirs):
            stations_files_list.append([s, None])
            continue

        for captured_dir in captured_dirs:
            dir_date, dir_time = captured_dir.split('_')[1], captured_dir.split('_')[2]
            year, month, day = int(dir_date[0:4]), int(dir_date[4:6]), int(dir_date[6:8])
            hour, minute, second = int(dir_time[0:2]), int(dir_time[2:4]), int(dir_time[4:6])
            dir_time = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second).replace(tzinfo=datetime.timezone.utc)
            dir_time += datetime.timedelta(seconds=time_offset_seconds)
            if dir_time < target_image_time:
                break

        if print_activity:
            print("Using {}".format(captured_dir))

        dir_files = sorted(os.listdir(os.path.join(captured_dir_path, captured_dir)))

        min_time_delta = np.inf

        closest_fits_file_full_path = None
        for file in dir_files:
            if file.startswith('FF_{}'.format(c.stationID)) and file.endswith('.fits'):

                file_date, file_time = file.split('_')[2], file.split('_')[3]
                year, month, day = int(file_date[0:4]), int(file_date[4:6]), int(file_date[6:8])
                hour, minute, second = int(file_time[0:2]), int(file_time[2:4]), int(file_time[4:6])
                file_time = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second).replace(tzinfo=datetime.timezone.utc)
                time_delta = abs(target_image_time + datetime.timedelta(seconds=time_offset_seconds) - file_time).total_seconds()
                if time_delta < min_time_delta:
                    closest_fits_file_full_path = os.path.join(c.data_dir, c.captured_dir, captured_dir, file)
                    min_time_delta = time_delta

        stations_files_list.append([s, closest_fits_file_full_path])
        if print_activity:
            if closest_fits_file_full_path is None:
                print("Could not find a file for {} for stations {}".format(target_image_time + datetime.timedelta(seconds=time_offset_seconds), s))
            else:
                print("Added {} with a time delta of {} seconds".format(os.path.basename(closest_fits_file_full_path), min_time_delta))

    return stations_files_list


def getFitsAsList(stations_files_list, stations_info_dict, print_activity=False, compensation=[50,80]):
    """
    Given a list of lists of stations and paths to fits files, return a list of images from
    the fits compensated to an average intensity of zero.

    Arguments:
        stations_files_list: [[list]] list of [station, path to fits file].
        stations_info_dict: [dict] dictionary of station information keyed by stationID.

    Keyword Arguments:
        print_activity: [bool] Optional, default False.

    Return:
        fits_list: [list] list of compensated fits images as arrays.
    """

    fits_dict, fits_list = {}, []
    for s, f in stations_files_list:
        if print_activity:
            print("Load fits {}".format(f))
        if f is None:
            pp = stations_info_dict[s]['pp']
            fits_list.append(np.array(np.zeros((pp.Y_res, pp.X_res))))
        else:
            ff = readFF(os.path.dirname(f), os.path.basename(f))

            max_pixel = ff.maxpixel.astype(np.float32)
            compensated_image = max_pixel
            min_threshold, max_threshold = np.percentile(compensated_image, compensation[0]), np.percentile(compensated_image, compensation[1])
            if min_threshold == max_threshold:
                compensated_image =  np.full_like(compensated_image, 128)
            else:
                compensated_image = (2 ** 16 * (compensated_image - min_threshold) / (max_threshold - min_threshold)) - 2 ** 15

            fits_list.append(compensated_image)

    return fits_list

def makeUpload(source_path, upload_to):
    """
    Make upload of source_path

    Arguments:
        source_path: [string] source path
        upload_to: [string] upload location i.e. user@host:path/on/remote

    Return:
        Nothing

    """

    cmd = [
        "rsync",
        "-avz",  # archive mode, verbose, compress
        source_path, upload_to
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("Upload failed with {}".format(e))


def SkyPolarProjection(config_paths, path_to_transform, force_recomputation=False, repeat=False, period=120,
                       print_activity=False, size=500, stack_depth=3, upload=None, annotate=True, minimum_elevation_deg=20,
                       target_jd=None, compensation=[50, 80, 80, 99.75], plot_constellations=True, write_image=True):

    """

    Arguments:
        config_paths: [list] list of config file paths.
        path_to_transform: [path] path to an existing source to destination transform, or the path where it should be saved.


    Keyword arguments:
        force_recomputation: [bool] Optional, default false, force recomputaion of the transform.
        period: [int] Optional default 120, period between plots.
        print_activity: [bool] Optional default False, print activity.
        size: [int] Optional default 500, size of both axes.
        stack_depth: [int] Optional default 3, number of images to stack.
        upload: [str] Optional, default None, if set where to upload finished image to.
        annotate: [bool] Optional, default True, annotate image.

    Return:
        target_image_array: [bool] Array of image.
    """

    # Load the config files into a dict
    stations_info_dict = getStationsInfoDict(config_paths)
    size_x, size_y = size, size


    # Load transform and check matches image size
    if os.path.exists(path_to_transform) and not force_recomputation:
        with open(path_to_transform, 'rb') as f:
            transform_data = pickle.load(f)
            stations_info_dict_loaded, transformation_layer_list, source_coordinates_array, dest_coordinates_array, intensity_scaling_array, cam_coords = transform_data
            if stations_info_dict_loaded.keys() != stations_info_dict.keys():
                force_recomputation = True


            if size != intensity_scaling_array.shape[0]:
                force_recomputation = True
                if print_activity:
                    print("Requested image size does not match size of intensity scaling array  - recomputing transform")

    if not os.path.exists(path_to_transform) or force_recomputation:
        transformation_layer_list, source_coordinates_array, dest_coordinates_array, intensity_scaling_array, cam_coords =  \
            makeTransformation(stations_info_dict, size_x, size_y, minimum_elevation_deg=minimum_elevation_deg, print_activity=print_activity, stack_depth=stack_depth)
        transform_data = [stations_info_dict, transformation_layer_list, source_coordinates_array, dest_coordinates_array,
                          intensity_scaling_array, cam_coords]

        with open(os.path.expanduser(path_to_transform), 'wb') as f:
            pickle.dump(transform_data, f)

    stations_info_dict, transformation_layer_list, source_coordinates_array, dest_coordinates_array, intensity_scaling_array, cam_coords = transform_data
    next_iteration_start_time = datetime.datetime.now(tz=datetime.timezone.utc)

    stations_as_text = ""
    for s in stations_info_dict.keys():
        stations_as_text = "{}, {}".format(stations_as_text,s.strip())

    if len(stations_as_text):
        stations_as_text = stations_as_text[2:]




    while True:
        this_iteration_start_time = next_iteration_start_time
        next_iteration_start_time += datetime.timedelta(seconds=period)
        # Compute epoch for this image
        if target_jd is None:
            target_image_time = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(seconds=20)
        else:
            target_image_time = jd2Date(target_jd, dt_obj=True).replace(tzinfo=datetime.timezone.utc)

            if print_activity:
                print("Target image time from julian date {} is {}".format(target_jd, target_image_time))
            repeat = False

        annotation_text_l1 = "{} Stack depth {:.0f}".format(target_image_time.replace(microsecond=0), len(transformation_layer_list) / len(stations_info_dict))


        annotation_text_l2 = "Lat:{:.3f} deg Lon:{:.3f} deg {}".format(cam_coords[0], cam_coords[1], stations_as_text)

        # Get the fits files as a stack of fits, one per camera
        fits_array = np.stack(getFitsAsList(getFitsFiles(transformation_layer_list, stations_info_dict, target_image_time), stations_info_dict, compensation=compensation), axis=0)

        target_image_time_jd = date2JD(*(target_image_time.timetuple()[:6]))

        # Form the uncompensated and target image arrays
        target_image_array, target_image_array_uncompensated = np.full_like(intensity_scaling_array, 0), np.full_like(
            intensity_scaling_array, 0)

        # Unwrap the source coordinates array into component lists
        camera_no, source_y, source_x, vignetting_factor_array = source_coordinates_array.T

        # And the destination coordinates list
        target_y, target_x = dest_coordinates_array.T

        # Build the uncompensated image by mappings coordinates from each camera
        intensities = fits_array[list(map(int, camera_no)), list(map(int, source_x)), list(map(int, source_y))]

        # Stack the images
        np.add.at(target_image_array_uncompensated, (target_x, target_y), intensities * vignetting_factor_array)

        div_zero_replacement = np.min(intensities)
        target_image_array = np.divide(target_image_array_uncompensated,
                                       intensity_scaling_array,
                                       out=np.full_like(target_image_array_uncompensated, div_zero_replacement, dtype=float),
                                       where=intensity_scaling_array!=0).astype(float)

        # Perform compensation
        min_threshold, max_threshold = np.percentile(intensities, float(compensation[2])), np.percentile(intensities, compensation[3])
        target_image_array = np.clip(255 * (target_image_array - min_threshold) / (max_threshold - min_threshold), 0, 255)

        if plot_constellations:
            constellation_coordinates_list = getConstellationsImageCoordinates(target_image_time_jd, cam_coords, size_x,
                                                                           size_y, minimum_elevation_deg)
            for x, y, x_, y_ in constellation_coordinates_list:
                cv2.line(target_image_array, (x, y), (x_, y_), 18, 1)



        if print_activity:
            print("Writing output to {:s}".format(output_path))


        target_image_array = target_image_array.astype(np.uint8)

        if annotate:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.3
            thickness = 1
            position_l1 = (3, size_y - 20)
            cv2.putText(target_image_array, annotation_text_l1, position_l1, font, font_scale, (55, 55, 55), thickness, cv2.LINE_AA)
            position_l2 = (3, size_y - 5)
            cv2.putText(target_image_array, annotation_text_l2, position_l2, font, font_scale, (55, 55, 55), thickness, cv2.LINE_AA)

        if write_image and (output_path.endswith(".png") or output_path.endswith(".bmp")):
            imageio.imwrite(output_path, target_image_array)
        if print_activity:
            print("Plotted in {:.1f} seconds".format((datetime.datetime.now(tz=datetime.timezone.utc) - this_iteration_start_time).total_seconds()))
            if repeat:
                print("Next run at {}".format(next_iteration_start_time.replace(microsecond=0)))

        if upload is not None and not make_timelapse:
            if print_activity:
                print("Uploading to {}".format(upload))
            makeUpload(output_path, upload)
            if print_activity:
                print("Uploaded")
        if not repeat:
            return target_image_array

        time.sleep(max((next_iteration_start_time - datetime.datetime.now(tz=datetime.timezone.utc)).total_seconds(),0))



def getConstellationsImageCoordinates(jd, cam_coords, size_x, size_y, minimum_elevation_deg, print_activity=True):

    lat, lon = cam_coords[0], cam_coords[1]



    if print_activity:
        print("Getting constellation coordinates at jd {} for location lat: {} lon: {}".format(jd, cam_coords[0], cam_coords[1]))
    constellations_path = os.path.join(os.path.expanduser("~/source/RMS/share/constellation_lines.csv"))
    lines = np.loadtxt(constellations_path, delimiter=",")
    array_ra_j2000, array_dec_j2000, array_ra_j2000_ ,array_dec_j2000_ = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]

    j2000=2451545

    list_ra, list_dec, list_ra_, list_dec_ = [], [], [] ,[]



    for ra_od, dec_od, ra_od_, dec_od_ in zip(array_ra_j2000, array_dec_j2000, array_ra_j2000_, array_dec_j2000_):
        ra_rads, dec_rads = equatorialCoordPrecession(j2000, jd, np.radians(ra_od), np.radians(dec_od))
        ra_rads_, dec_rads_ = equatorialCoordPrecession(j2000, jd, np.radians(ra_od_), np.radians(dec_od_))
        list_ra.append(np.degrees(ra_rads))
        list_dec.append(np.degrees(dec_rads))
        list_ra_.append(np.degrees(ra_rads_))
        list_dec_.append(np.degrees(dec_rads_))


    if False:
        list_ra = [220.35]
        list_dec = [-60.92]
        list_ra_ = [211.4]
        list_dec_ = [-60.50]


    array_ra, array_dec = np.array(list_ra), np.array(list_dec)
    array_ra_, array_dec_ = np.array(list_ra_), np.array(list_dec_)

    array_az, array_alt = raDec2AltAz(array_ra, array_dec, jd, lat, lon)
    array_az_, array_alt_ = raDec2AltAz(array_ra_ ,array_dec_ , jd, lat, lon)
    con = np.stack([array_alt, array_az, array_alt_, array_az_], axis=1)
    constellation_alt_az_above_horizon = con[(con[:, 0] >= 45) & (con[:, 2] >= 45)]




    image_coordinates = []




    """
    el_deg = 90 - np.hypot(_x * pixel_to_radius_scale_factor_x, _y * pixel_to_radius_scale_factor_y)
    az_deg = np.degrees(np.arctan2(_x, _y))
    """
    if print_activity:
        print("Creating constellation data for an image of size {},{}".format(size_x, size_y))

    origin_x, origin_y = size_x / 2, size_y / 2

    elevation_range = 2 * (90 - minimum_elevation_deg)
    pixel_to_radius_scale_factor_x = elevation_range / size_x
    pixel_to_radius_scale_factor_y = elevation_range / size_y

    for alt, az, alt_, az_ in constellation_alt_az_above_horizon:

        x, y = altAzToCartesian(az, alt, 0, size_x, 0, size_y, 20)

        alt_check, az_check = cartesianToAltAz(x, y, 0, size_x, 0, size_y, 20 )
        print(alt, alt_check, az, az_check)


        x_, y_ = altAzToCartesian(az_, alt_, 0, size_x, 0, size_y, 20)

        alt_check_, az_check_ = cartesianToAltAz(x_, y_, 0, size_x, 0, size_y, 20 )
        print(alt_, alt_check_, az_, az_check_)

        image_coordinates.append([int(x), int(y), int(x_), int(y_)])

        pass


    if False:
        img=np.zeros((size_x, size_y), dtype=np.uint8)

        for x, y, x_, y_ in image_coordinates:
            cv2.line(img, (x, y), (x_, y_), 20, 1 )

        imageio.imwrite('cons.png', img)

    return image_coordinates


if __name__ == "__main__":

    # ## PARSE INPUT ARGUMENTS ###
    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description=""" Produce a projection from multiple cameras""")

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, action="append",
                            help="Optional, paths to the config files. If no paths given then will search for config files"
                                 "in a multi-cam linux style file arrangement, or a one camera per usename arrangement.")


    arg_parser.add_argument('-r', '--repeat',  dest='repeat', default=False, action="store_true",
                    help="Run continuously, default false.")

    arg_parser.add_argument('-p', '--period', dest='period', default=[120], type=int, nargs=1,
                            help="Iteration period for continous running, default 120 seconds.")

    arg_parser.add_argument('-s', '--stack', dest='stack', default=[3], type=int, nargs=1,
                            help="Number of images to stack, default 3. This will only take affect if transform is recomputed.")

    arg_parser.add_argument('-o', '--output_file_name', dest='output_file_name', default=None,
                            nargs=1, help="Output filename and path. If only a path to a directory is given, then files will be saved with a timestamp."
                            "YYYYMMSS_HHMMDD. If no output path is given ~/RMS_data will be used.")

    arg_parser.add_argument('-t', '--transform', dest='transform', default=False, action="store_true",
                            help="Force recomputing of transform - needed if platepar has been changed")

    arg_parser.add_argument('-d', '--dimension', dest='dimension', default=[1000], type=int, nargs=1,
                            help="Output image size - only square images are permitted. Default 1000 x 1000.")

    arg_parser.add_argument('-q', '--quiet', dest='quiet', default=False, action="store_true",
                            help="Run quietly")

    arg_parser.add_argument('-u', '--upload', dest='upload', type=str, nargs=1,
                            help="Remote address to upload finished image to.")

    arg_parser.add_argument('-a', '--annotate', dest='annotate', default=False, action="store_true",
                            help="Annotate plot with image time, stations used, and projection origin.")

    arg_parser.add_argument('-n', '--constellations', dest='constellations', default=False, action="store_true",
                            help="Annotate plot with constellations.")

    arg_parser.add_argument('-e', '--elevation', dest='elevation', nargs=1, type=float, default=[20],
                            help="Minimum elevation to use for the plot")

    arg_parser.add_argument('-l', '--timelapse', dest='timelapse', nargs='*', type=float,
                            help="Generate timelapse over the past 24 hours of observations, including spanning"
                            "directories, or if two julian dates are specified, then timelapse between those two dates")

    arg_parser.add_argument('-j', '--julian-date', dest='julian_date', nargs=1, type=float,
                            help="Generate a single projection at the specified julian date")

    arg_parser.add_argument('-m', '--compensation', dest='compensation', nargs=4, type=float,
                            help="Image compensation values 50 80 90 99.85 work well")

    cml_args = arg_parser.parse_args()



    quiet = cml_args.quiet
    print_activity = not quiet
    path_to_transform = os.path.expanduser("~/RMS_data/camera_combination.transform")
    force_recomputation = cml_args.transform
    repeat = cml_args.repeat

    period = cml_args.period[0]

    if cml_args.dimension is not None:
        # round to even number
        size = int(cml_args.dimension[0] / 2) * 2
    else:
        size = 500

    config_paths = []

    if cml_args.config is None:
        path_list = None
    else:
        for path_list in cml_args.config:
            config_paths.append(path_list[0])

    stack_depth = cml_args.stack[0]
    quiet = cml_args.quiet

    if cml_args.upload is None:
        upload = None
    else:
        upload = cml_args.upload[0]

    annotate = cml_args.annotate

    if cml_args.elevation is None:
        minimum_elevation_deg = 0
    else:
        minimum_elevation_deg = cml_args.elevation[0] if cml_args.elevation[0] > 0 else 0

    # Initialise values - these should never be used
    timelapse_start, timelapse_end, seconds_per_frame = None, None, None
    make_timelapse = False


    if cml_args.timelapse is None:
        timelapse_start = None
        timelapse_end = None
        seconds_per_frame = None

    else:
        if len(cml_args.timelapse) == 0:
            timelapse_end = date2JD(*(datetime.datetime.now(datetime.timezone.utc).timetuple()[:6]))
            timelapse_start = timelapse_end - 1
            seconds_per_frame = 256/25
            make_timelapse = True

        elif len(cml_args.timelapse) == 1:
            timelapse_start = cml_args.timelapse[0]
            timelapse_end = date2JD(*(datetime.datetime.now(datetime.timezone.utc).timetuple()[:6]))
            seconds_per_frame = 256 / 25
            make_timelapse = True


        elif len(cml_args.timelapse) == 2:
            timelapse_start = cml_args.timelapse[0]
            timelapse_end = cml_args.timelapse[1]
            seconds_per_frame = 256 / 25
            make_timelapse = True


        elif len(cml_args.timelapse) == 3:
            timelapse_start = cml_args.timelapse[0]
            timelapse_end = cml_args.timelapse[1]
            seconds_per_frame = cml_args.timelapse[2]
            make_timelapse = True

    if cml_args.julian_date is None:
        target_jd = None
    else:
        target_jd = cml_args.julian_date[0]

    if cml_args.compensation is None:
        compensation = [80, 95, 50, 99.995]
    else:
        compensation = cml_args.compensation

    plot_constellations = cml_args.constellations

    if cml_args.output_file_name is None:
        output_file_name = None
    else:
        output_file_name = os.path.expanduser(cml_args.output_file_name[0])


    if output_file_name is None:
        mkdirP(os.path.expanduser("~/RMS_data/PolarPlot/Projection/"))
        if make_timelapse:
            output_path = os.path.expanduser(
                "~/RMS_data/PolarPlot/Projection/JD_{}_timelapse.png".format(timelapse_start))

        else:
            output_path = os.path.expanduser(
                "~/RMS_data/PolarPlot/Projection/{}.png".format(target_image_time.strftime("%Y%m%d_%H%M%S")))

    else:
        if os.path.exists(os.path.expanduser(output_file_name)):
            if os.path.isdir(os.path.expanduser(output_file_name)):
                if make_timelapse:
                    output_path = os.path.expanduser(
                        "~/RMS_data/PolarPlot/Projection/JD_{}_timelapse.png".format(timelapse_start))
                else:
                    output_path = os.path.join(os.path.expanduser(output_file_name),
                                           "{}.png".format(target_image_time.strftime("%Y%m%d_%H%M%S")))
            else:
                output_path = os.path.expanduser(output_file_name)
        elif not os.path.exists(os.path.dirname(os.path.expanduser(output_file_name))):
            mkdirP(os.path.dirname(os.path.expanduser(output_file_name)))
            output_path = os.path.expanduser(output_file_name)
        else:
            output_path = os.path.expanduser(output_file_name)

    if make_timelapse:
        repeat = False
        timelapse_frames = []
        frame_count = int(((jd2Date(timelapse_end, dt_obj=True) - jd2Date(timelapse_start, dt_obj=True)).total_seconds()) / seconds_per_frame)
        start_time_obj =  datetime.datetime(*jd2Date(timelapse_start, dt_obj=True).timetuple()[:6])
        print("Output file name is {}".format(output_file_name))
        with imageio.get_writer(output_file_name, fps=25, codec="libx264", quality=8) as writer:
            for frame_no in tqdm.tqdm(range(0, frame_count)):
                frame_time_obj = start_time_obj  + datetime.timedelta(seconds = frame_no * seconds_per_frame)
                target_jd = date2JD(*frame_time_obj.timetuple()[:6])

                if print_activity:
                    print("Making frame at time {}".format(jd2Date(target_jd, dt_obj=True)))
                writer.append_data(SkyPolarProjection(config_paths, path_to_transform, force_recomputation=force_recomputation,
                                        repeat=repeat, period=period, print_activity=not quiet,
                                        size=size, stack_depth=stack_depth, upload=upload, annotate=annotate,
                                        target_jd=target_jd, minimum_elevation_deg=minimum_elevation_deg,
                                        compensation=compensation, write_image=False,
                                        plot_constellations=plot_constellations).astype(np.uint8))
                # If recomputation was forced, then only do it once
                force_recomputation = False


        if upload is not None and make_timelapse:
            if print_activity:
                print("Uploading to {}".format(upload))
            makeUpload(output_path, upload)
            if print_activity:
                print("Uploaded")


        sys.exit()


    SkyPolarProjection(config_paths, path_to_transform, force_recomputation=force_recomputation,
                       repeat=repeat, period=period, print_activity=not quiet,
                       size=size, stack_depth=stack_depth, upload=upload, annotate=annotate,
                       target_jd=target_jd, minimum_elevation_deg=minimum_elevation_deg, compensation=compensation,
                       plot_constellations=plot_constellations)
