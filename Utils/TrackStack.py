

import os
import json
import copy

import cv2
import numpy as np
import matplotlib.pyplot as plt

from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, raDecToXYPP
from RMS.Astrometry.Conversions import date2JD, jd2Date
from RMS.Formats.FFfile import validFFName, getMiddleTimeFF
from RMS.Formats.FFfile import read as readFF
from RMS.Formats.Platepar import Platepar
from RMS.Math import angularSeparation
from RMS.Routines.MaskImage import loadMask, MaskStructure


def trackStack(dir_path, config, border=5, background_compensation=True):
    """ Generate a stack with aligned stars, so the sky appears static. The folder should have a
        platepars_all_recalibrated.json file.

    Arguments:
        dir_path: [str] Path to the directory with image files.
        config: [Config instance]

    Keyword arguments:
        border: [int] Border around the image to exclude (px).
        background_compensation: [bool] Normalize the background by applying a median filter to avepixel and
            use it as a flat field. Slows down the procedure and may sometimes introduce artifacts. True
            by default.
    """


    # Load recalibrated platepars, if they exist ###

    # Find recalibrated platepars file per FF file
    platepars_recalibrated_file = None
    for file_name in os.listdir(dir_path):
        if file_name == config.platepars_recalibrated_name:
            platepars_recalibrated_file = file_name
            break


    # Load all recalibrated platepars if the file is available
    recalibrated_platepars = None
    if platepars_recalibrated_file is not None:
        with open(os.path.join(dir_path, platepars_recalibrated_file)) as f:
            recalibrated_platepars = json.load(f)
            print('Loaded recalibrated platepars JSON file for the calibration report...')

    # ###


    # If the recalib platepars is not found, stop
    if recalibrated_platepars is None:
        print("The {:s} file was not found!".format(config.platepars_recalibrated_name))
        return False


    # Get a list of FF files in the folder
    ff_list = []
    for file_name in os.listdir(dir_path):
        if validFFName(file_name):
            ff_list.append(file_name)


    # Take the platepar with the middle time as the reference one
    ff_found_list = []
    jd_list = []
    for ff_name_temp in recalibrated_platepars:

        if ff_name_temp in ff_list:

            # Compute the Julian date of the FF middle
            dt = getMiddleTimeFF(ff_name_temp, config.fps, ret_milliseconds=True)
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
    if os.path.exists(os.path.join(dir_path, config.mask_file)):
        mask_path = os.path.join(dir_path, config.mask_file)

    # Try loading the default mask
    elif os.path.exists(config.mask_file):
        mask_path = os.path.abspath(config.mask_file)

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
                [getMiddleTimeFF(ff_temp, config.fps, ret_milliseconds=True)], [x_c], [y_c], [1], pp_ref,
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
    img_size = int(scale*2*ang_sep_max*pp_ref.F_scale)

    #


    # Create the stack platepar with no distortion and a large image size
    pp_stack = copy.deepcopy(pp_ref)
    pp_stack.resetDistortionParameters()
    pp_stack.X_res = img_size
    pp_stack.Y_res = img_size
    pp_stack.F_scale *= scale
    pp_stack.refraction = False


    # Init the image
    avg_stack_sum = np.zeros((img_size, img_size), dtype=np.float)
    avg_stack_count = np.zeros((img_size, img_size), dtype=np.int)
    max_deaveraged = np.zeros((img_size, img_size), dtype=np.uint8)


    # Load individual FFs and map them to the stack
    for i, ff_name in enumerate(ff_found_list):

        print("Stacking {:s}, {:.1f}% done".format(ff_name, 100*i/len(ff_found_list)))

        # Read the FF file
        ff = readFF(dir_path, ff_name)

        # Load the recalibrated platepar
        pp_temp = Platepar()
        pp_temp.loadFromDict(recalibrated_platepars[ff_name], use_flat=config.use_flat)

        # Make a list of X and Y image coordinates
        x_coords, y_coords = np.meshgrid(np.arange(border, pp_ref.X_res - border),
                                         np.arange(border, pp_ref.Y_res - border))
        x_coords = x_coords.ravel()
        y_coords = y_coords.ravel()

        # Map image pixels to sky
        jd_arr, ra_coords, dec_coords, _ = xyToRaDecPP(
            len(x_coords)*[getMiddleTimeFF(ff_name, config.fps, ret_milliseconds=True)], x_coords, y_coords,
            len(x_coords)*[1], pp_temp, extinction_correction=False)

        # Map sky coordinates to stack image coordinates
        stack_x, stack_y = raDecToXYPP(ra_coords, dec_coords, jd_middle, pp_stack)

        # Round pixel coordinates
        stack_x = np.round(stack_x, decimals=0).astype(np.int)
        stack_y = np.round(stack_y, decimals=0).astype(np.int)

        # Cut the image to limits
        filter_arr = (stack_x > 0) & (stack_x < img_size) & (stack_y > 0) & (stack_y < img_size)
        x_coords = x_coords[filter_arr].astype(np.int)
        y_coords = y_coords[filter_arr].astype(np.int)
        stack_x = stack_x[filter_arr]
        stack_y = stack_y[filter_arr]


        # Apply the mask to maxpixel and avepixel
        maxpixel = copy.deepcopy(ff.maxpixel)
        maxpixel[mask.img == 0] = 0
        avepixel = copy.deepcopy(ff.avepixel)
        avepixel[mask.img == 0] = 0

        # Compute deaveraged maxpixel
        max_deavg = maxpixel - avepixel


        # Normalize the backgroud brightness by applying a large-kernel median filter to avepixel
        if background_compensation:

            # # Apply a median filter to the avepixel to get an estimate of the background brightness
            # avepixel_median = scipy.ndimage.median_filter(ff.avepixel, size=101)
            avepixel_median = cv2.medianBlur(ff.avepixel, 301)

            # Make sure to avoid zero division
            avepixel_median[avepixel_median < 1] = 1

            # Normalize the avepixel by subtracting out the background brightness
            avepixel = avepixel.astype(np.float)
            avepixel /= avepixel_median
            avepixel *= 50 # Normalize to a good background value, which is usually 50
            avepixel = np.clip(avepixel, 0, 255)
            avepixel = avepixel.astype(np.uint8)

            # plt.imshow(avepixel, cmap='gray', vmin=0, vmax=255)
            # plt.show()


        # Add the average pixel to the sum
        avg_stack_sum[stack_y, stack_x] += avepixel[y_coords, x_coords]

        # Increment the counter image where the avepixel is not zero
        ones_img = np.ones_like(avepixel)
        ones_img[avepixel == 0] = 0
        avg_stack_count[stack_y, stack_x] += ones_img[y_coords, x_coords]

        # Set pixel values to the stack, only take the max values
        max_deaveraged[stack_y, stack_x] = np.max(np.dstack([max_deaveraged[stack_y, stack_x],
                                                             max_deavg[y_coords, x_coords]]), axis=2)


    # Compute the blended avepixel background
    stack_img = avg_stack_sum
    stack_img[avg_stack_count > 0] /= avg_stack_count[avg_stack_count > 0]
    stack_img += max_deaveraged
    stack_img = np.clip(stack_img, 0, 255)
    stack_img = stack_img.astype(np.uint8)


    # Crop image
    non_empty_columns = np.where(stack_img.max(axis=0) > 0)[0]
    non_empty_rows = np.where(stack_img.max(axis=1) > 0)[0]
    crop_box = (np.min(non_empty_rows), np.max(non_empty_rows), np.min(non_empty_columns),
        np.max(non_empty_columns))
    stack_img = stack_img[crop_box[0]:crop_box[1]+1, crop_box[2]:crop_box[3]+1]



    # Plot and save the stack ###

    dpi = 200
    plt.figure(figsize=(stack_img.shape[1]/dpi, stack_img.shape[0]/dpi), dpi=dpi)

    plt.imshow(stack_img, cmap='gray', vmin=0, vmax=256, interpolation='nearest')

    plt.axis('off')
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)

    plt.xlim([0, stack_img.shape[1]])
    plt.ylim([stack_img.shape[0], 0])

    # Remove the margins (top and right are set to 0.9999, as setting them to 1.0 makes the image blank in 
    #   some matplotlib versions)
    plt.subplots_adjust(left=0, bottom=0, right=0.9999, top=0.9999, wspace=0, hspace=0)

    # remove leading path separator if present
    dir_path = dir_path.rstrip('/').rstrip('\\')
    filenam = os.path.join(dir_path, os.path.basename(dir_path) + "_track_stack.jpg")
    plt.savefig(filenam, bbox_inches='tight', pad_inches=0, dpi=dpi)

    #

    plt.show()








if __name__ == "__main__":


    import argparse

    import RMS.ConfigReader as cr


    # ## PARSE INPUT ARGUMENTS ###

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description=""" Generate a stack with aligned stars.
        """)

    arg_parser.add_argument('dir_path', type=str, help="Path to the folder of the night.")

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str,
        help="Path to a config file which will be used instead of the default one.")

    arg_parser.add_argument('-b', '--bkgnormoff', action="store_true",
        help="""Disable background normalization.""")


    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #############################


    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, cml_args.dir_path)

    trackStack(cml_args.dir_path, config, background_compensation=(not cml_args.bkgnormoff))
