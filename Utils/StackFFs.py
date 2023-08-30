""" Stacks all maxpixles in the given folder to one image. """

from __future__ import print_function, division, absolute_import

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from RMS.Formats.FFfile import read as readFF
from RMS.Formats.FFfile import validFFName
from RMS.Formats.FTPdetectinfo import validDefaultFTPdetectinfo, readFTPdetectinfo
from RMS.Routines.Image import deinterlaceBlend, blendLighten, loadFlat, applyFlat, adjustLevels, saveImage
from RMS.Routines import MaskImage



def stackFFs(dir_path, file_format, deinterlace=False, subavg=False, filter_bright=False, flat_path=None,
    file_list=None, mask=None, captured_stack=False, print_progress=True):
    """ Stack FF files in the given folder. 

    Arguments:
        dir_path: [str] Path to the directory with FF files.
        file_format: [str] Image format for the stack. E.g. jpg, png, bmp

    Keyword arguments:
        deinterlace: [bool] True if the image shoud be deinterlaced prior to stacking. False by default.
        subavg: [bool] Whether the average pixel image should be subtracted form the max pixel image. False
            by default. 
        filter_bright: [bool] Whether images with bright backgrounds (after average subtraction) should be
            skipped. False by defualt.
        flat_path: [str] Path to the flat calibration file. None by default. Will only be used if subavg is
            False.
        file_list: [list] A list of file for stacking. False by default, in which case all FF files in the
            given directory will be used.
        mask: [MaskStructure] Mask to apply to the stack. None by default.
        captured_stack: [bool] True if all files are used and "_captured_stack" will be used in the file name.
            False by default.
        print_progress: [bool] Allow print calls to show files being stacked. True by default

    Return:
        stack_path, merge_img:
            - stack_path: [str] Path of the save stack.
            - merge_img: [ndarray] Numpy array of the stacked image.
    """

    # Load the flat if it was given
    flat = None
    if flat_path != '':

        # Try finding the default flat
        if flat_path is None:
            flat_path = dir_path
            flat_file = 'flat.bmp'

        else:
            flat_path, flat_file = os.path.split(flat_path)

        flat_full_path = os.path.join(flat_path, flat_file)
        if os.path.isfile(flat_full_path):

            # Load the flat
            flat = loadFlat(flat_path, flat_file)

            print('Loaded flat:', flat_full_path)


    first_img = True

    n_stacked = 0
    total_ff_files = 0
    merge_img = None

    # If the list of files was not given, take all files in the given folder
    if file_list is None:
        file_list = sorted(os.listdir(dir_path))


    # List all FF files in the current dir
    for ff_name in file_list:
        if validFFName(ff_name):

            # Load FF file
            ff = readFF(dir_path, ff_name)

            # Skip the file if it is corruped
            if ff is None:
                continue

            total_ff_files += 1

            maxpixel = ff.maxpixel
            avepixel = ff.avepixel

            # Dinterlace the images
            if deinterlace:
                maxpixel = deinterlaceBlend(maxpixel)
                avepixel = deinterlaceBlend(avepixel)

            # If the flat was given, apply it to the image, only if no subtraction is done
            if (flat is not None) and not subavg:
                maxpixel = applyFlat(maxpixel, flat)
                avepixel = applyFlat(avepixel, flat)


            # Reject the image if the median subtracted image is too bright. This usually means that there
            #   are clouds on the image which can ruin the stack
            if filter_bright:

                img = maxpixel - avepixel

                # Compute surface brightness
                median = np.median(img)

                # Compute top detection pixels
                top_brightness = np.percentile(img, 99.9)

                # Reject all images where the median brightness is high
                # Preserve images with very bright detections
                if (median > 10) and (top_brightness < (2**(8*img.itemsize) - 10)):
                    if print_progress:
                        print('Skipping: ', ff_name, 'median:', median, 'top brightness:', top_brightness)
                    continue


            # Subtract the average from maxpixel
            if subavg:
                img = maxpixel - avepixel

            else:
                img = maxpixel

            if first_img:
                merge_img = np.copy(img)
                first_img = False
                n_stacked += 1
                continue

            if print_progress:
                print('Stacking: ', ff_name)

            # Blend images 'if lighter'
            merge_img = blendLighten(merge_img, img)

            n_stacked += 1


    # If the number of stacked image is less than 20% of the given images, stack without filtering
    if filter_bright and (n_stacked < 0.2*total_ff_files):
        return stackFFs(dir_path, file_format, deinterlace=deinterlace, subavg=subavg, 
            filter_bright=False, flat_path=flat_path, file_list=file_list)

    # If no images were stacked, do nothing
    if n_stacked == 0:
        return None, None


    # Extract the name of the night directory which contains the FF files
    night_dir = os.path.basename(dir_path)

    # If the stack was captured, add "_captured_stack" to the file name
    if captured_stack:
        filename_suffix = "_captured_stack."
    else:
        filename_suffix = "_stack_{:d}_meteors.".format(n_stacked)


    stack_path = os.path.join(dir_path, night_dir + filename_suffix + file_format)

    if print_progress:
        print("Saving stack to:", stack_path)

    # Stretch the levels
    merge_img = adjustLevels(merge_img, np.percentile(merge_img, 0.5), 1.3, np.percentile(merge_img, 99.9))


    # Apply the mask, if given
    if mask is not None:
        merge_img = MaskImage.applyMask(merge_img, mask)

    
    # Save the blended image
    saveImage(stack_path, merge_img)


    return stack_path, merge_img



if __name__ == '__main__':

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Stacks all maxpixles in the given folder to one image.")

    arg_parser.add_argument('dir_path', metavar='DIR_PATH', type=str, \
        help='Path to directory with FF files.')

    arg_parser.add_argument('file_format', nargs=1, metavar='FILE_FORMAT', type=str, \
        help='File format of the image, e.g. jpg or png.')

    arg_parser.add_argument('-d', '--deinterlace', action="store_true", \
        help="""Deinterlace the image before stacking. """)

    arg_parser.add_argument('-s', '--subavg', action="store_true", \
        help="""Subtract the average image from maxpixel before stacking. """)

    arg_parser.add_argument('-b', '--brightfilt', action="store_true", \
        help="""Rejects images with very bright background, which are often clouds. """)

    arg_parser.add_argument('-x', '--hideplot', action="store_true", \
        help="""Don't show the stack on the screen after stacking. """)

    arg_parser.add_argument('-f', '--flat', nargs='?', metavar='FLAT_PATH', type=str, default='', 
        help="Apply a given flat frame. If no path to the flat is given, flat.bmp from the folder will be taken.")

    arg_parser.add_argument('-m', '--mask', metavar='MASK_PATH', type=str, 
        help="Apply a given mask. If no path to the mask is given, mask.bmp from the folder will be taken.")

    arg_parser.add_argument('--ftpdetectinfo', action="store_true", \
        help="""Only stack FF files given in the FTPdetectinfo file. """)

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    # Load the mask
    mask = None
    if cml_args.mask is not None:
        if os.path.exists(cml_args.mask):
            mask_path = os.path.abspath(cml_args.mask)
            print('Loading mask:', mask_path)
            mask = MaskImage.loadMask(mask_path)


    # If the FTPdetectinfo parameter is given, find the FTPdetectinfo file and only select those FF files
    # in it
    ff_list = None
    if cml_args.ftpdetectinfo:

        # Get a list of files in the directory
        dir_file_list = os.listdir(cml_args.dir_path)

        # Try to find the FTPdetectinfo file
        ftpdetectinfo_list = [fn for fn in dir_file_list if validDefaultFTPdetectinfo(fn)]

        if len(ftpdetectinfo_list) > 0:

            print("Using FTPdetectinfo for stacking!")

            ftpdetectinfo_name = ftpdetectinfo_list[0]

            # Load the FTPdetectinfo file
            meteor_list = readFTPdetectinfo(cml_args.dir_path, ftpdetectinfo_name)
            ff_list = [entry[0] for entry in meteor_list if entry[0] in dir_file_list]



    # Run stacking
    stack_path, merge_img = stackFFs(cml_args.dir_path, cml_args.file_format[0], \
        deinterlace=cml_args.deinterlace, subavg=cml_args.subavg, filter_bright=cml_args.brightfilt, \
        flat_path=cml_args.flat, file_list=ff_list, mask=mask)



    if not cml_args.hideplot:
        
        # Plot the blended image
        plt.imshow(merge_img, cmap='gray', vmin=0, vmax=255)

        plt.show()




