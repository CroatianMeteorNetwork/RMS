from __future__ import print_function

import os, sys
import json
import copy
from glob import glob
import argparse


import cv2
import numpy as np
import matplotlib.pyplot as plt
import RMS.ConfigReader as cr
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, raDecToXYPP
from RMS.Astrometry.Conversions import date2JD, jd2Date, raDec2AltAz
from RMS.Formats.FFfile import validFFName, getMiddleTimeFF, filenameToDatetime
from RMS.Formats.FFfile import read as readFF
from RMS.Formats.Platepar import Platepar
from RMS.Math import angularSeparation
from Utils.ShowerAssociation import showerAssociation
from Utils.DrawConstellations import drawConstellations
from RMS.Routines.MaskImage import loadMask, MaskStructure
import multiprocessing as mp
import ctypes
from RMS.QueuedPool import QueuedPool
import time
import datetime


def dirNameToDatetime(file_name):
    """ Converts FF bin file name to a datetime object.

    Arguments:
        file_name: [str] Name of a FF file.

    Return:
        [datetime object] Date and time of the first frame in the FF file.

    """

    # e.g.  FF499_20170626_020520_353_0005120.bin
    # or FF_CA0001_20170626_020520_353_0005120.fits

    file_name = file_name.split('_')

    # Check the number of list elements, and the new fits format has one more underscore
    i = 0
    if len(file_name[0]) == 2:
        i = 1

    date = file_name[i + 1]
    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[6:8])

    time = file_name[i + 2]
    hour = int(time[:2])
    minute = int(time[2:4])
    seconds = int(time[4:6])

    us = int(file_name[i + 3])


    return datetime.datetime(year, month, day, hour, minute, seconds, microsecond= us)


def getDirectoryList(config, stack_time):
    """ Get the paths of directories which may contain files associated with a moment in time

         Arguments:
             event: [event]

         Return:
             directorylist: [list of paths] List of directories
    """

    directory_list = []

    # iterate across the folders in CapturedFiles and convert the directory time to posix time
    if os.path.exists(os.path.join(os.path.expanduser(config.data_dir), config.captured_dir)):
        for night_directory in os.listdir(
                os.path.join(os.path.expanduser(config.data_dir), config.captured_dir)):
            # Skip over any directory which does not start with the stationID and warn
            if night_directory[0:len(config.stationID)] != config.stationID:
                print("Skipping directory {} - not the expected format for a captured files directory".format(night_directory))
                continue
            directory_POSIX_time = dirNameToDatetime(night_directory)

            # if the POSIX time representation is before the event, and within 16 hours add to the list of directories
            # most unlikely that a single event could be split across two directories, unless there was large time uncertainty
            if directory_POSIX_time < stack_time and (stack_time - directory_POSIX_time).total_seconds() < 16 * 3600:
                directory_list.append(
                    os.path.join(os.path.expanduser(config.data_dir), config.captured_dir,
                                 night_directory))
    return directory_list


def camStack(config_path_list, stack_time = datetime.datetime.utcnow() - datetime.timedelta(seconds = 30), border=5, background_compensation=True,
        hide_plot=False, showers=None, darkbackground=False, out_dir=None,
        scalefactor=None, draw_constellations=False, one_core_free=False):
    """ Generate a stack with aligned stars, so the sky appears static. The folder should have a
        platepars_all_recalibrated.json file.

    Arguments:
        dir_paths: [list of str] Path to the directory with image files.
        config_file_names: [list of str] Paths to the .config files associated with each directory

    Keyword arguments:
        border: [int] Border around the image to exclude (px).
        background_compensation: [bool] Normalize the background by applying a median filter to avepixel and
            use it as a flat field. Slows down the procedure and may sometimes introduce artifacts. True
            by default.
        showers: [list[str]] List of showers to include, as code. E.g. or ["GEM","URS"].
            As a code for sporadics, use "..."
        darkbackground: [bool] force the sky background to be dark
        out_dir: target folder to save into
        scalefactor: factor to scale the canvas by; default 1, increase if image cropped
    """


    print("Computing a stack at {}".format(str(stack_time)))

    start_time = time.time()
    # normalise the path in a platform neutral way
    # done here so that camStack() can be called from other modules
    config_path_list = [os.path.normpath(config_path) for config_path in config_path_list]

    #iterate through the config paths and gather input data
    #get the .config file
    #get the platepar file
    #get the .fits files closest to the time of interest

    config_list, captured_files_dir_list, platepar_file_list, mask_file_list, matching_fits_list = [],[],[],[],[]

    for config_path in config_path_list:

        this_config = cr.parse(os.path.expanduser(config_path))
        config_list.append(this_config)

        captured_files_dir_list.append(os.path.join(this_config.data_dir, this_config.captured_dir))

        platepar_file_list.append(os.path.join(os.path.dirname(os.path.expanduser(config_path)), this_config.platepar_name))


        file_extension_list = ['.fits']
        directory_list = getDirectoryList(this_config, stack_time)
        print("Directories to search")
        print(directory_list)

        finding_first_file = True

        closest_file = ""
        for directory in directory_list:
            for file_extension in file_extension_list:
                # get the directory into name order
                dirlist = os.listdir(directory)
                dirlist.sort()
                if file_extension == ".fits":
                    fits_list = glob(os.path.join(directory, "*.fits"))
                    fits_list.sort()

                    if len(fits_list) == 0:
                        # If fits_list is empty then return an empty list
                        print("No fits files in {}".format(directory))


                for file in dirlist:

                    if file.endswith(file_extension):
                        file_time = filenameToDatetime(file)
                        if finding_first_file:
                            closest_file = file
                            smallest_delta_seconds = abs((file_time - stack_time).total_seconds())
                            finding_first_file = False
                        else:
                            if abs((file_time - stack_time).total_seconds()) < smallest_delta_seconds:
                                closest_file = file
                                if False:
                                    print("file         :{}".format(file))
                                    print("closest file :{}".format(closest_file))
                                smallest_delta_seconds = abs((file_time - stack_time).total_seconds())
        print("Closest file is ".format(closest_file))
        matching_fits_list.append(os.path.join(directory,closest_file))
        mask_file_list.append(os.path.join(os.path.dirname(os.path.expanduser(config_path)), this_config.mask_file))

    if False:
        for config, captured_files_dir, platepar_file, mask_file, fits in zip(config_list, captured_files_dir_list, platepar_file_list, mask_file_list, matching_fits_list):
            print("Using {}, {}, {}".format(platepar_file,mask_file,fits))


    pp_ref = Platepar()
    pp_ref.read(platepar_file_list[0])



    jd_middle = date2JD(stack_time.year, stack_time.month, stack_time.day,stack_time.hour,stack_time.minute, stack_time.second)

    img_size = 4000
    scale = 0.5

    pp_stack = copy.deepcopy(pp_ref)
    pp_stack.resetDistortionParameters()
    pp_stack.X_res = img_size
    pp_stack.Y_res = img_size
    pp_stack.F_scale *= scale
    pp_stack.refraction = False


    avg_stack_sum_shared = mp.Array(ctypes.c_float, img_size*img_size)
    avg_stack_count_shared = mp.Array(ctypes.c_int, img_size*img_size)
    max_deaveraged_shared = mp.Array(ctypes.c_uint8, img_size*img_size)
    finished_count = mp.Value(ctypes.c_int, 0)




    for ff_path, config, platepar_file, mask_file in zip(matching_fits_list, config_list, platepar_file_list, mask_file_list):
        
        # Load the platepar
        pp_temp = Platepar()
        pp_temp.read(platepar_file, use_flat=config.use_flat)

        mask = None
        if mask_file is not None:
            mask = loadMask(mask_file)


        print("Reading {}".format(ff_path))
        ff = readFF(os.path.dirname(ff_path), os.path.basename(ff_path))

        # Make a list of X and Y image coordinates
        x_coords, y_coords = np.meshgrid(np.arange(border, pp_ref.X_res - border),
                                         np.arange(border, pp_ref.Y_res - border))
        x_coords = x_coords.ravel()
        y_coords = y_coords.ravel()
        # Map image pixels to sky
        ff_basename = os.path.basename(ff_path)




        jd_arr, ra_coords, dec_coords, _ = xyToRaDecPP(
            len(x_coords) * [getMiddleTimeFF(ff_basename, config.fps, ret_milliseconds=True)], x_coords, y_coords,
            len(x_coords) * [1], pp_temp, extinction_correction=False)

        stack_x, stack_y = raDecToXYPP(ra_coords, dec_coords, jd_middle, pp_stack)

        # Round pixel coordinates
        stack_x = np.round(stack_x, decimals=0).astype(int)
        stack_y = np.round(stack_y, decimals=0).astype(int)

        # Cut the image to limits
        filter_arr = (stack_x > 0) & (stack_x < img_size) & (stack_y > 0) & (stack_y < img_size)
        x_coords = x_coords[filter_arr].astype(int)
        y_coords = y_coords[filter_arr].astype(int)
        stack_x = stack_x[filter_arr]
        stack_y = stack_y[filter_arr]

        # Apply the mask to maxpixel and avepixel
        maxpixel = copy.deepcopy(ff.maxpixel)
        avepixel = copy.deepcopy(ff.avepixel)


        # Compute deaveraged maxpixel image
        max_deavg = maxpixel - avepixel

        if True:
            # # Apply a median filter to the avepixel to get an estimate of the background brightness
            # avepixel_median = scipy.ndimage.median_filter(ff.avepixel, size=101)
            avepixel_median = cv2.medianBlur(ff.avepixel, 301)

            # Make sure to avoid zero division
            avepixel_median[avepixel_median < 1] = 1

            # Normalize the avepixel by subtracting out the background brightness
            avepixel = avepixel.astype(float)
            avepixel /= avepixel_median
            avepixel *= 50  # Normalize to a good background value, which is usually 50
            avepixel = np.clip(avepixel, 0, 255)
            avepixel = avepixel.astype(np.uint8)

        #plt.imshow(avepixel, cmap='gray', vmin=0, vmax=255)
        #plt.show()

        avg_stack_sum = getArray(img_size, avg_stack_sum_shared)
        avg_stack_count = getArray(img_size, avg_stack_count_shared)
        max_deaveraged = getArray(img_size, max_deaveraged_shared)

        avg_stack_sum[stack_y, stack_x] += avepixel[y_coords, x_coords]
        ones_img = np.ones_like(avepixel)
        ones_img[avepixel == 0] = 0
        avg_stack_count[stack_y, stack_x] += ones_img[y_coords, x_coords]
        max_deaveraged[stack_y, stack_x] = np.max(np.dstack([max_deaveraged[stack_y, stack_x],
                                                                 max_deavg[y_coords, x_coords]]), axis=2)



    stack_img = avg_stack_sum
    stack_img[avg_stack_count > 0] /= avg_stack_count[avg_stack_count > 0]
    stack_img += max_deaveraged
    stack_img = np.clip(stack_img, 0, 255)
    stack_img = stack_img.astype(np.uint8)


    # Plot and save the stack ###

    non_empty_columns = np.where(stack_img.max(axis=0) > 0)[0]
    non_empty_rows = np.where(stack_img.max(axis=1) > 0)[0]
    crop_box = (np.min(non_empty_rows), np.max(non_empty_rows), np.min(non_empty_columns),
        np.max(non_empty_columns))
    stack_img = stack_img[crop_box[0]:crop_box[1]+1, crop_box[2]:crop_box[3]+1]

    dpi = 200
    extrapix = 80 # space for annotations across the bottom (not handled by this module)
    fig = plt.figure(figsize=(stack_img.shape[1]/dpi, (stack_img.shape[0]+extrapix)/dpi), dpi=dpi)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1])

    vmin = 0
    if darkbackground is True:
        vmin = np.quantile(stack_img[stack_img>0], 0.05)
    plt.imshow(stack_img, cmap='gray', vmin=vmin, vmax=256, interpolation='nearest')
    if draw_constellations:
        constellations_img = constellations_img[crop_box[0]:crop_box[1]+1, crop_box[2]:crop_box[3]+1]
        plt.imshow(constellations_img)

    ax.set_axis_off()

    ax.set_xlim([0, stack_img.shape[1]])
    ax.set_ylim([stack_img.shape[0]+extrapix, 0])

    if showers is not None:
        msg = 'Filtered for {}'.format(showers)
        ax.text(10, stack_img.shape[0] - 10, msg, color='gray', fontsize=6, fontname='Source Sans Pro', weight='ultralight')

    # Remove the margins (top and right are set to 0.9999, as setting them to 1.0 makes the image blank in
    #   some matplotlib versions)
    plt.subplots_adjust(left=0, bottom=0, right=0.9999, top=0.9999, wspace=0, hspace=0)

    if out_dir is not None:
        filenam = os.path.join(os.path.expanduser(out_dir), stack_time.strftime("%Y%m%d_%H%M%S") + "_cam_stack.jpg")
    else:
        filenam = os.path.join(dir_path, os.path.basename(dir_path) + "_track_stack.jpg")
    plt.savefig(filenam, bbox_inches='tight', pad_inches=0, dpi=dpi, facecolor='k', edgecolor='k')
    print('saved to {}'.format(filenam))
    #
    print("Stacking time: {}".format(datetime.timedelta(seconds=(int(time.time() - start_time)))))
    if hide_plot is False:
        plt.show()

    return True


def stackFrame(ff_name, pp_temp, mask, border, pp_ref, img_size, jd_middle, pp_stack, conf, avg_stack_sum_arr,
               avg_stack_count_arr, max_deaveraged_arr, background_compensation, finished_count, num_ffs):
    ff_basename = os.path.basename(ff_name)

    avg_stack_sum = getArray(img_size, avg_stack_sum_arr)
    avg_stack_count = getArray(img_size, avg_stack_count_arr)
    max_deaveraged = getArray(img_size, max_deaveraged_arr)

    # Read the FF file
    ff = readFF(*os.path.split(ff_name))




    # Make a list of X and Y image coordinates
    x_coords, y_coords = np.meshgrid(np.arange(border, pp_ref.X_res - border),
                                     np.arange(border, pp_ref.Y_res - border))
    x_coords = x_coords.ravel()
    y_coords = y_coords.ravel()
    # Map image pixels to sky
    jd_arr, ra_coords, dec_coords, _ = xyToRaDecPP(
        len(x_coords) * [getMiddleTimeFF(ff_basename, conf.fps, ret_milliseconds=True)], x_coords, y_coords,
        len(x_coords) * [1], pp_temp, extinction_correction=False)
    # Map sky coordinates to stack image coordinates
    stack_x, stack_y = raDecToXYPP(ra_coords, dec_coords, jd_middle, pp_stack)

    # Round pixel coordinates
    stack_x = np.round(stack_x, decimals=0).astype(int)
    stack_y = np.round(stack_y, decimals=0).astype(int)

    # Cut the image to limits
    filter_arr = (stack_x > 0) & (stack_x < img_size) & (stack_y > 0) & (stack_y < img_size)
    x_coords = x_coords[filter_arr].astype(int)
    y_coords = y_coords[filter_arr].astype(int)
    stack_x = stack_x[filter_arr]
    stack_y = stack_y[filter_arr]

    # Apply the mask to maxpixel and avepixel
    maxpixel = copy.deepcopy(ff.maxpixel)
    maxpixel[mask.img == 0] = 0
    avepixel = copy.deepcopy(ff.avepixel)
    avepixel[mask.img == 0] = 0

    # Compute deaveraged maxpixel image
    max_deavg = maxpixel - avepixel

    # Normalize the backgroud brightness by applying a large-kernel median filter to avepixel
    if background_compensation:

        # # Apply a median filter to the avepixel to get an estimate of the background brightness
        # avepixel_median = scipy.ndimage.median_filter(ff.avepixel, size=101)
        avepixel_median = cv2.medianBlur(ff.avepixel, 301)

        # Make sure to avoid zero division
        avepixel_median[avepixel_median < 1] = 1

        # Normalize the avepixel by subtracting out the background brightness
        avepixel = avepixel.astype(float)
        avepixel /= avepixel_median
        avepixel *= 50 # Normalize to a good background value, which is usually 50
        avepixel = np.clip(avepixel, 0, 255)
        avepixel = avepixel.astype(np.uint8)

        # plt.imshow(avepixel, cmap='gray', vmin=0, vmax=255)
        # plt.show()

    with avg_stack_sum_arr.get_lock():
        # Add the average pixel to the sum
        avg_stack_sum[stack_y, stack_x] += avepixel[y_coords, x_coords]

    # Increment the counter image where the avepixel is not zero
    ones_img = np.ones_like(avepixel)
    ones_img[avepixel == 0] = 0
    with avg_stack_count_arr.get_lock():
        avg_stack_count[stack_y, stack_x] += ones_img[y_coords, x_coords]
    with max_deaveraged_arr.get_lock():
        # Set pixel values to the stack, only take the max values
        max_deaveraged[stack_y, stack_x] = np.max(np.dstack([max_deaveraged[stack_y, stack_x],
                                                             max_deavg[y_coords, x_coords]]), axis=2)
    with finished_count.get_lock():
        finished_count.value += 1
    # print progress
    printProgress(finished_count.value, num_ffs)


def shouldInclude(shower_list, ff_name, associations):
    if shower_list is None:
        return True
    else:
        ff_basename = os.path.basename(ff_name)
        try:
            shower = associations[(ff_basename, 1.0)][1]
            return shower.name in shower_list
        except:
            return False


def printProgress(current, total):
    progress_bar_len = 20
    progress = int(progress_bar_len * current / total)
    percent = 100 * current / total
    print("\rStacking : {:02.0f}%|{}{}| {}/{} ".format(percent, "#" * progress, " " * (progress_bar_len - progress), current, total), end="")
    if current == total:
        print("")


def getArray(size, shared_arr):
    numpy_arr = np.ctypeslib.as_array(shared_arr.get_obj())
    return numpy_arr.reshape(size, size)


if __name__ == "__main__":
    # ## PARSE INPUT ARGUMENTS ###
    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description=""" Generate a stack with aligned stars.
        """)

    arg_parser.add_argument('dir_paths', nargs='+', type=str, help="Path to the folder of the night.")

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str,
        help="Path to a config file which will be used instead of the default one.")

    arg_parser.add_argument('-b', '--bkgnormoff', action="store_true",
        help="""Disable background normalization.""")

    arg_parser.add_argument('-x', '--hideplot', action="store_true",
        help="""Don't show the stack on the screen after stacking. """)

    arg_parser.add_argument('-o', '--output', type=str,
        help="""folder to save the image in.""")

    arg_parser.add_argument('-f', '--scalefactor', type=int,
        help="""scale factor to apply. Increase if image is cropped""")

    arg_parser.add_argument('-s', '--showers', type=str,
        help="Show only meteors from specific showers (e.g. URS, PER, GEM, ... for sporadic). Comma-separated list. \
            Note that an RMS config file that matches the data is required for this option.")

    arg_parser.add_argument('--constellations', help="Overplot constellations", action="store_true")

    arg_parser.add_argument('-d', '--darkbackground', action="store_true",
        help="""Darken the background. """)

    arg_parser.add_argument('--freecore', action="store_true",
                            help="""Leave at least one core free""")


    arg_parser.add_argument('--start', type=str,
                            help="""Start time of plot e.g. 20231231_010203""")

    arg_parser.add_argument('--end', type=str,
                            help="""End time of plot e.g. 20231231_040506""")

    arg_parser.add_argument('--interval', type=int,
                            help="""Time between frames""")




    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #############################

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, cml_args.dir_paths[0])

    showers = cml_args.showers
    if showers is not None:
        showers = showers.split(",")
        showers = [s.upper() for s in showers]



    plot_date = cml_args.start.split('_')[0]
    plot_time = cml_args.start.split('_')[1]

    print("Plotting on date {}".format(plot_date))
    print("         at time {}".format(plot_time))

    pyr = int(plot_date[0:4])
    pmth = int(plot_date[4:6])
    pday= int(plot_date[6:8])

    phr = int(plot_time[0:2])
    pmin = int(plot_time[2:4])
    psec = int(plot_time[4:6])

    start = datetime.datetime(pyr,pmth,pday,phr,pmin,psec)

    plot_date = cml_args.end.split('_')[0]
    plot_time = cml_args.end.split('_')[1]



    pyr = int(plot_date[0:4])
    pmth = int(plot_date[4:6])
    pday = int(plot_date[6:8])

    phr = int(plot_time[0:2])
    pmin = int(plot_time[2:4])
    psec = int(plot_time[4:6])

    end = datetime.datetime(pyr, pmth, pday, phr, pmin, psec)

    timerange = int((end - start).total_seconds())

    if cml_args.interval is not None:
        interval = int(cml_args.interval)
    else:
        interval = 10

    for time_offset in range(0,timerange,interval):

        print(start + datetime.timedelta(seconds = time_offset))
        stack_time = start + datetime.timedelta(seconds = time_offset)
        dir_paths = [os.path.normpath(dir_path) for dir_path in cml_args.dir_paths]
        camStack(dir_paths, stack_time, background_compensation=(not cml_args.bkgnormoff),
            hide_plot=cml_args.hideplot, showers=showers,
            darkbackground=cml_args.darkbackground, out_dir=cml_args.output, scalefactor=cml_args.scalefactor,
            draw_constellations=cml_args.constellations, one_core_free=cml_args.freecore)
