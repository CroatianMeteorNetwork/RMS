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
import pickle
import argparse
import subprocess
import cv2
import numpy as np
from ephem import julian_date

import RMS.ConfigReader as cr
import datetime
import pathlib
import time
import imageio as imageio

from RMS.Astrometry.Conversions import altAz2RADec, jd2Date, date2JD
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, raDecToXYPP, correctVignetting
from RMS.Formats.FFfile import read as readFF
from RMS.Formats.Platepar import Platepar
from RMS.Math import angularSeparationDeg
from RMS.Misc import mkdirP
from RMS.Routines.MaskImage import loadMask

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
    az_vals_deg, el_vals_deg = np.zeros((size_x + 1, size_y + 1)), np.zeros((size_x + 1, size_y + 1))

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
        _, r_source, d_source, _ = xyToRaDecPP([pp_source.JD], [pp_source.X_res / 2], [pp_source.Y_res / 2], [1], pp_source, jd_time=True)
        r_list, d_list, x_dest_list, y_dest_list = [], [], [], []

        for y_dest in range(0, size_y + 1):
            for x_dest in range(0, size_x + 1):
                _x, _y, = x_dest - origin_x, y_dest - origin_y

                # Convert the target image (polar projection on cartesian axis) into azimuth and elevation
                el_deg = 90 - np.hypot(_x * pixel_to_radius_scale_factor_x, _y * pixel_to_radius_scale_factor_y)
                az_deg = np.degrees(np.arctan2(_x, _y))

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
            if not (0 < x_source < (pp_source.X_res - 0) and 0 < y_source < (pp_source.Y_res - 0)):
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
    intensity_scaling_array = np.zeros((size_x + 1, size_y + 1))
    for pair, count in zip(pairs, counts):
        intensity_scaling_array[pair[1]][pair[0]] = count

    return transformation_layer_list, source_coordinates_array, dest_coordinates_array, intensity_scaling_array, [target_lat, target_lon, target_ele]

def getFitsFiles(transformation_layer_list, stations_info_dict, target_image_time, print_activity=True):
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
                print("Added {}".format(os.path.basename(closest_fits_file_full_path)))

    return stations_files_list


def getFitsAsList(stations_files_list, stations_info_dict, print_activity=False):
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

            max_pixel = ff.maxpixel
            compensation_value = np.mean(max_pixel)
            compensated_image = max_pixel - compensation_value
            min_threshold, max_threshold = np.percentile(compensated_image, 90), np.percentile(compensated_image, 99.5)
            if min_threshold == max_threshold:
                compensated_image =  np.full_like(compensated_image, 128)
            else:
                compensated_image = (255 * (compensated_image - min_threshold) / (max_threshold - min_threshold)) - 128

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
                       print_activity=False, size=500, stack_depth=3, upload=None, annotate=True,
                       min_elevation=20, timelapse_start=None, timelapse_end=None, target_jd=None, minimum_elevation_deg=20):

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
        Nothing.
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

            if size != intensity_scaling_array.shape[0] - 1:
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
            target_image_time = jd2Date(target_jd, dt_obj=True).astimezone(datetime.timezone.utc)
            repeat = False

        annotation_text_l1 = "{} Stack depth {:.0f}".format(target_image_time.replace(microsecond=0), len(transformation_layer_list) / len(stations_info_dict))
        print(annotation_text_l1)

        annotation_text_l2 = "Lat:{:.3f} deg Lon:{:.3f} deg {}".format(cam_coords[0], cam_coords[1], stations_as_text)

        # Get the fits files as a stack of fits, one per camera
        fits_array = np.stack(getFitsAsList(getFitsFiles(transformation_layer_list, stations_info_dict, target_image_time), stations_info_dict), axis=0)

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
                                       where=intensity_scaling_array!=0)

        # Perform compensation
        min_threshold, max_threshold = np.percentile(intensities, 50), np.percentile(intensities, 99.5)
        target_image_array = np.clip(254 * (target_image_array - min_threshold) / (max_threshold - min_threshold), 0, 255)

        if output_file_name is None:
            mkdirP(os.path.expanduser("~/RMS_data/PolarPlot/Projection/"))
            output_path = os.path.expanduser(
                "~/RMS_data/PolarPlot/Projection/{}.png".format(target_image_time.strftime("%Y%m%d_%H%M%S")))
        else:
            if os.path.exists(os.path.expanduser(output_file_name)):
                if os.path.isdir(os.path.expanduser(output_file_name)):
                    output_path = os.path.join(os.path.expanduser(output_file_name),
                                               "{}.png".format(target_image_time.strftime("%Y%m%d_%H%M%S")))
                else:
                    output_path = os.path.expanduser(output_file_name)
            elif not os.path.exists(os.path.dirname(os.path.expanduser(output_file_name))):
                mkdirP(os.path.dirname(os.path.expanduser(output_file_name)))
                output_path = os.path.expanduser(output_file_name)
            else:
                output_path = os.path.expanduser(output_file_name)

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

        imageio.imwrite(output_path, target_image_array)
        if print_activity:
            print("Plotted in {:.1f} seconds".format((datetime.datetime.now(tz=datetime.timezone.utc) - this_iteration_start_time).total_seconds()))
            print("Next run at {}".format(next_iteration_start_time.replace(microsecond=0)))

        if upload is not None:
            if print_activity:
                print("Uploading to {}".format(upload))
            makeUpload(output_path, upload)
            if print_activity:
                print("Uploaded")
        if not repeat:
            break

        time.sleep(max((next_iteration_start_time - datetime.datetime.now(tz=datetime.timezone.utc)).total_seconds(),0))

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

    cml_args = arg_parser.parse_args()


    if cml_args.output_file_name is None:
        output_file_name = None
    else:
        output_file_name = cml_args.output_file_name[0]
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
        minimum_elevation_deg = 20
    else:
        minimum_elevation_deg = cml_args.elevation[0] if cml_args.elevation[0] > 0 else 0

    if cml_args.timelapse is None:
        timelapse_start = None
        timelapse_end = None
    else:
        if len(cml_args.timelapse) == 0:
            timelapse_end = date2JD(*(datetime.datetime.now(datetime.timezone.utc).timetuple()[:6]))
            timelapse_start = timelapse_end - 1


        elif len(cml_args.timelapse) == 1:
            timelapse_start = cml_args.timelapse[0]
            timelapse_end = None
        elif len(cml_args.timelapse) == 2:
            timelapse_start = cml_args.timelapse[0]
            timelapse_end = cml_args.timelapse[1]

    if cml_args.julian_date is None:
        target_jd = None
    else:
        target_jd = cml_args.julian_date[0]


    SkyPolarProjection(config_paths, path_to_transform, force_recomputation=force_recomputation,
                       repeat=repeat, period=period, print_activity=not quiet,
                       size=size, stack_depth=stack_depth, upload=upload, annotate=annotate,
                       timelapse_start=timelapse_start, timelapse_end=timelapse_end,
                       target_jd=target_jd, minimum_elevation_deg=minimum_elevation_deg)
