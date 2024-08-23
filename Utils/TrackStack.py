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
from RMS.Astrometry.Conversions import date2JD, jd2Date
from RMS.Formats.FFfile import validFFName, getMiddleTimeFF
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


def trackStack(dir_paths, config, border=5, background_compensation=True, 
        hide_plot=False, showers=None, darkbackground=False, out_dir=None,
        scalefactor=None, draw_constellations=False, one_core_free=False,
        textoption=0):
    """ Generate a stack with aligned stars, so the sky appears static. The folder should have a
        platepars_all_recalibrated.json file.

    Arguments:
        dir_paths: [str] Path to the directory with image files.
        config: [Config instance]

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
        draw_constellations: [bool] Show constellation lines on stacked image
        one_core_free: [bool] leave one core free whilst processing
        overlay_file_name: [bool] show the filename on the completed image
    """
    start_time = time.time()
    # normalise the path in a platform neutral way
    # done here so that trackStack() can be called from other modules
    dir_paths = [os.path.normpath(dir_path) for dir_path in dir_paths]

    # Find recalibrated platepars file per FF file
    recalibrated_platepars = {}
    for dir_path in dir_paths: 
        platepars_recalibrated_file = glob(os.path.join(dir_path, config.platepars_recalibrated_name))
        if len(platepars_recalibrated_file) != 1:
            print('unable to find a unique platepars file in {}'.format(dir_path))
            return False
        print('loading {}'.format(platepars_recalibrated_file[0]))
        with open(platepars_recalibrated_file[0]) as f:
            pp_per_dir = json.load(f)
            # Put the full path in all the keys
            for key in pp_per_dir:
                recalibrated_platepars[os.path.join(os.path.dirname(platepars_recalibrated_file[0]), key)] = pp_per_dir[key]
    print('Loaded recalibrated platepars JSON file for the calibration report...')

    # ###
    associations = {}
    if showers is not None:

        # Get FTP file so we can filter by shower
        for dir_path in dir_paths: 

            if os.path.isfile(os.path.join(dir_path,'.config')):
                tmpcfg = cr.loadConfigFromDirectory('.config', dir_path)
            else:
                tmpcfg = config

            ftp_list = glob(os.path.join(dir_path, 'FTPdetectinfo_{}*.txt'.format(tmpcfg.stationID)))
            ftp_list = [x for x in ftp_list if 'backup' not in x and 'unfiltered' not in x]
            ftp_list.sort() 

            if len(ftp_list) < 1:
                print('unable to find FTPdetect file in {}'.format(dir_path))
                return False
            
            ftp_file = ftp_list[0] 

            print('Performing shower association using {}'.format(ftp_file))

            associations_per_dir, _ = showerAssociation(config, [ftp_file], 
                shower_code=None, show_plot=False, save_plot=False, plot_activity=False)
            associations.update(associations_per_dir)

        # Get a list of FF files in the folder
        ff_list = []
        for key in associations:
            ff_list.append(key[0])

    else:
        # Get a list of FF files in the folder
        ff_list = []
        for dir_path in dir_paths:
            for file_name in os.listdir(dir_path):
                if validFFName(file_name):
                    ff_list.append(file_name)    
    ff_list = list(set(ff_list))

    # Take the platepar with the middle time as the reference one
    ff_found_list = []
    jd_list = []
    for ff_name_temp in recalibrated_platepars:

        if os.path.basename(ff_name_temp) in ff_list:

            # Compute the Julian date of the FF middle
            dt = getMiddleTimeFF(os.path.basename(ff_name_temp), config.fps, ret_milliseconds=True)
            jd = date2JD(*dt)

            jd_list.append(jd)
            ff_found_list.append(ff_name_temp)



    if len(jd_list) < 2:
        print("Not more than 1 FF image!")
        return False



    # Take the FF file with the middle JD
    jd_list = np.array(jd_list)
    jd_middle = np.mean(jd_list)
    jd_mean_index = np.argmin(np.abs(jd_list - jd_middle))
    ff_mid = ff_found_list[jd_mean_index]

    # Load the middle platepar as the reference one
    pp_ref = Platepar()
    pp_ref.loadFromDict(recalibrated_platepars[ff_mid], use_flat=config.use_flat)



    # Try loading the mask
    mask_path = None
    mask_path_default = os.path.join(config.config_file_path, config.mask_file)
    if os.path.exists(os.path.join(dir_paths[0], config.mask_file)):
        mask_path = os.path.join(dir_paths[0], config.mask_file)

    # Try loading the default mask
    elif os.path.exists(mask_path_default):
        mask_path = os.path.abspath(mask_path_default)

    # Load the mask if given
    mask = None
    if mask_path is not None:
        mask = loadMask(mask_path)
        print("Loaded mask:", mask_path)

    # If the shape of the mask doesn't fit, init an empty mask
    if mask is not None:
        if (mask.img.shape[0] != pp_ref.Y_res) or (mask.img.shape[1] != pp_ref.X_res):
            print("Mask is of wrong shape!")
            mask = None


    if mask is None:
        mask = MaskStructure(255 + np.zeros((pp_ref.Y_res, pp_ref.X_res), dtype=np.uint8))


    # Compute the middle RA/Dec of the reference platepar
    _, ra_temp, dec_temp, _ = xyToRaDecPP([jd2Date(jd_middle)], [pp_ref.X_res/2], [pp_ref.Y_res/2], [1],
        pp_ref, extinction_correction=False)

    ra_mid, dec_mid = ra_temp[0], dec_temp[0]


    # Go through all FF files and find RA/Dec of image corners to find the size of the stack image ###
    
    # List of corners
    x_corns = [0, pp_ref.X_res,            0, pp_ref.X_res]
    y_corns = [0,            0, pp_ref.Y_res, pp_ref.Y_res]

    ra_list = []
    dec_list = []

    for ff_temp in ff_found_list:
        
        # Load the recalibrated platepar
        pp_temp = Platepar()
        pp_temp.loadFromDict(recalibrated_platepars[ff_temp], use_flat=config.use_flat)

        for x_c, y_c in zip(x_corns, y_corns):
            _, ra_temp, dec_temp, _ = xyToRaDecPP(
                [getMiddleTimeFF(os.path.basename(ff_temp), config.fps, ret_milliseconds=True)], [x_c], [y_c], [1], pp_ref,
                extinction_correction=False)
            ra_c, dec_c = ra_temp[0], dec_temp[0]

            ra_list.append(ra_c)
            dec_list.append(dec_c)


    # Compute the angular separation from the middle equatorial coordinates of the reference image to all
    #   RA/Dec corner coordinates
    ang_sep_list = []
    for ra_c, dec_c in zip(ra_list, dec_list):
        ang_sep = np.degrees(angularSeparation(np.radians(ra_mid), np.radians(dec_mid), np.radians(ra_c),
            np.radians(dec_c)))

        ang_sep_list.append(ang_sep)


    # Find the maximum angular separation and compute the image size using the plate scale
    #   The image size will be resampled to 1/2 of the original size to avoid interpolation
    scale = 0.5
    ang_sep_max = np.max(ang_sep_list)
    # scalefactor is a fudge factor to make the canvas large enough in some edge cases
    # default 1 works in most cases but multi-camera configs may need 2 or 3
    if scalefactor is None:
        scalefactor = 1
    img_size = int(scale*2*ang_sep_max*pp_ref.F_scale)*scalefactor

    #


    # Create the stack platepar with no distortion and a large image size
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

    # get number of images to include
    num_ffs = len(ff_found_list)
    if showers is not None:
        num_ffs = 0
        for acc in associations:
            shower = associations[acc][1]
            if shower is not None and shower.name in showers:
                num_ffs += 1

    # Load individual FFs and map them to the stack
    num_plotted = 0

    enumlist = ff_found_list
    # Create task pool
    cores = mp.cpu_count()
    if one_core_free and cores > 1:
        cores -= 1
    thead_pool = QueuedPool(stackFrame, cores=cores, backup_dir=None, print_state=False, func_extra_args=(recalibrated_platepars, mask, border,
                                                                                   pp_ref, img_size, jd_middle, pp_stack, config,
                                                                                   avg_stack_sum_shared, avg_stack_count_shared, max_deaveraged_shared,
                                                                                   background_compensation, finished_count, num_ffs))
    thead_pool.startPool()
    # add jobs
    for i, ff_name in enumerate(enumlist):
        if shouldInclude(showers, ff_name, associations):
            num_plotted += 1
            thead_pool.addJob([ff_name])
    printProgress(0, num_plotted)
    thead_pool.closePool()

    # End if the number of plotted FFs is zero
    if num_plotted == 0:
        print()
        print("No FFs plotted! Check the shower association or the detections.")
        return False

    avg_stack_sum = getArray(img_size, avg_stack_sum_shared)
    avg_stack_count = getArray(img_size, avg_stack_count_shared)
    max_deaveraged = getArray(img_size, max_deaveraged_shared)

    # Compute the blended avepixel background
    stack_img = avg_stack_sum
    stack_img[avg_stack_count > 0] /= avg_stack_count[avg_stack_count > 0]
    stack_img += max_deaveraged
    stack_img = np.clip(stack_img, 0, 255)
    stack_img = stack_img.astype(np.uint8)


    # Draw constellations
    if draw_constellations:
        constellations_img = drawConstellations(pp_stack, ff_mid,
                                                separation_deg=175)


    # Crop image
    non_empty_columns = np.where(stack_img.max(axis=0) > 0)[0]
    non_empty_rows = np.where(stack_img.max(axis=1) > 0)[0]
    crop_box = (np.min(non_empty_rows), np.max(non_empty_rows), np.min(non_empty_columns),
        np.max(non_empty_columns))
    stack_img = stack_img[crop_box[0]:crop_box[1]+1, crop_box[2]:crop_box[3]+1]


    # Plot and save the stack ###

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
        filenam = os.path.join(out_dir, os.path.basename(dir_path) + "_track_stack.jpg")
    else:
        filenam = os.path.join(dir_path, os.path.basename(dir_path) + "_track_stack.jpg")

    # Overlay the filename on the image\
    bf = os.path.basename(filenam)

    # Overlay filename only
    if textoption is not None:
        if textoption == 1:
            ax.text(10, stack_img.shape[0] + 30, bf, color='grey', fontsize=6, fontname='Source Sans Pro',
                    weight='ultralight')

    # Overlay stationID, YYYY-MM-DD Meteor count
        if textoption == 2:
            annotation = "{}  {}-{}-{}      Meteors: {}".format(bf[0:6],bf[7:11],bf[11:13],bf[13:15],num_plotted)
            ax.text(10, stack_img.shape[0] + 30, annotation, color='grey', fontsize=6, fontname='Source Sans Pro',
                weight='ultralight')

    plt.savefig(filenam, bbox_inches='tight', pad_inches=0, dpi=dpi, facecolor='k', edgecolor='k')
    print('saved to {}'.format(filenam))
    #
    print("Stacking time: {}".format(datetime.timedelta(seconds=(int(time.time() - start_time)))))
    if hide_plot is False:
        plt.show()

    return True


def stackFrame(ff_name, recalibrated_platepars, mask, border, pp_ref, img_size, jd_middle, pp_stack, conf, avg_stack_sum_arr,
               avg_stack_count_arr, max_deaveraged_arr, background_compensation, finished_count, num_ffs):
    ff_basename = os.path.basename(ff_name)

    avg_stack_sum = getArray(img_size, avg_stack_sum_arr)
    avg_stack_count = getArray(img_size, avg_stack_count_arr)
    max_deaveraged = getArray(img_size, max_deaveraged_arr)

    # Read the FF file
    ff = readFF(*os.path.split(ff_name))

    # Load the recalibrated platepar
    pp_temp = Platepar()
    pp_temp.loadFromDict(recalibrated_platepars[ff_name], use_flat=conf.use_flat)

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

    arg_parser.add_argument('-t', '--textoption', nargs=1, type=int,
                            help="""Add text beneath image. 
                                        0 - No text
                                        1 - Filename 
                                        2 - Station id, date, meteor count""")

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

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #############################

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, cml_args.dir_paths[0])

    showers = cml_args.showers
    if showers is not None:
        showers = showers.split(",")
        showers = [s.upper() for s in showers]

    text_option = 0
    if cml_args.textoption:
        text_option = cml_args.textoption[0]

    dir_paths = [os.path.normpath(dir_path) for dir_path in cml_args.dir_paths]
    trackStack(dir_paths, config, background_compensation=(not cml_args.bkgnormoff),
        hide_plot=cml_args.hideplot, showers=showers,
        darkbackground=cml_args.darkbackground, out_dir=cml_args.output, scalefactor=cml_args.scalefactor,
        draw_constellations=cml_args.constellations, one_core_free=cml_args.freecore,
        textoption = text_option)
