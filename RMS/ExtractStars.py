
# RPi Meteor Station
# Copyright (C) 2016 Denis Vida
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

from __future__ import print_function, division, absolute_import

import time
import sys
import os
import argparse
import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

# RMS imports
import RMS.ConfigReader as cr
from RMS.Formats import FFfile
from RMS.Formats import CALSTARS
from RMS.DetectionTools import loadImageCalibration
from RMS.Routines import MaskImage
from RMS.Routines import Image
from RMS.QueuedPool import QueuedPool

# Morphology - Cython init
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
#import RMS.Routines.MorphCy as morph


# Get the logger from the main module
log = logging.getLogger("logger")


def extractStars(ff_dir, ff_name, config=None, max_global_intensity=150, border=10, neighborhood_size=10, 
        intensity_threshold=5, flat_struct=None, dark=None, mask=None):
    """ Extracts stars on a given FF bin by searching for local maxima and applying PSF fit for star 
        confirmation.

        Source of one part of the code: 
    http://stackoverflow.com/questions/9111711/get-coordinates-of-local-maxima-in-2d-array-above-certain-value
    
    Arguments:
        ff_dir: [str] Path to directory where FF files are.
        ff_name: [str] Name of the FF file.
        config: [config object] configuration object (loaded from the .config file)
        max_global_intensity: [int] maximum mean intensity of an image before it is discared as too bright
        border: [int] apply a mask on the detections by removing all that are too close to the given image 
            border (in pixels)
        neighborhood_size: [int] size of the neighbourhood for the maximum search (in pixels)
        intensity_threshold: [float] a threshold for cutting the detections which are too faint (0-255)
        flat_struct: [Flat struct] Structure containing the flat field. None by default.
        dark: [ndarray] Dark frame. None by default.
        mask: [ndarray] Mask image. None by default.

    Return:
        x2, y2, background, intensity, fwhm: [list of ndarrays]
            - x2: X axis coordinates of the star
            - y2: Y axis coordinates of the star
            - background: background intensity
            - intensity: intensity of the star
            - Gaussian Full width at half maximum (FWHM) of fitted stars
    """

    # This will be returned if there was an error
    error_return = [[], [], [], [], [], []]

    # Load parameters from config if given
    if config:
        max_global_intensity = config.max_global_intensity
        border = config.border
        neighborhood_size = config.neighborhood_size
        intensity_threshold = config.intensity_threshold
        
    # Load the FF bin file
    ff = FFfile.read(ff_dir, ff_name)


    # If the FF file could not be read, skip star extraction
    if ff is None:
        return error_return


    # Apply the dark frame
    if dark is not None:
        ff.avepixel = Image.applyDark(ff.avepixel, dark)

    # Apply the flat
    if flat_struct is not None:
        ff.avepixel = Image.applyFlat(ff.avepixel, flat_struct)

    # Mask the FF file
    if mask is not None:
        ff = MaskImage.applyMask(ff, mask, ff_flag=True)


    # Calculate image mean and stddev
    global_mean = np.mean(ff.avepixel)

    # Check if the image is too bright and skip the image
    if global_mean > max_global_intensity:
        return error_return

    data = ff.avepixel.astype(np.float32)


    # Apply a mean filter to the image to reduce noise
    data = ndimage.filters.convolve(data, weights=np.full((2, 2), 1.0/4))

    # Locate local maxima on the image
    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > intensity_threshold)
    maxima[diff == 0] = 0

    # Apply a border mask
    border_mask = np.ones_like(maxima)*255
    border_mask[:border,:] = 0
    border_mask[-border:,:] = 0
    border_mask[:,:border] = 0
    border_mask[:,-border:] = 0
    maxima = MaskImage.applyMask(maxima, border_mask, image=True)

    # Remove all detections close to the mask image
    if mask is not None:
        erosion_kernel = np.ones((5, 5), mask.img.dtype)
        mask_eroded = cv2.erode(mask.img, erosion_kernel, iterations=1)

        maxima = MaskImage.applyMask(maxima, mask_eroded, image=True)


    # Find and label the maxima
    labeled, num_objects = ndimage.label(maxima)

    # Skip the image if there are too many maxima to process
    if num_objects > config.max_stars:
        log.warning('Too many candidate stars to process! {:d}/{:d}'.format(num_objects, config.max_stars))
        return error_return

    # Find centres of mass of each labeled objects
    xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)))

    # Remove all detection on the border
    #xy = xy[np.where((xy[:, 1] > border) & (xy[:,1] < ff.ncols - border) & (xy[:,0] > border) & (xy[:,0] < ff.nrows - border))]

    # Unpack star coordinates
    y, x = np.hsplit(xy, 2)

    # # Plot stars before the PSF fit
    # plotStars(ff, x, y)

    # Fit a PSF to each star
    x2, y2, amplitude, intensity, sigma_y_fitted, sigma_x_fitted = fitPSF(ff, global_mean, x, y, config)
    
    # x2, y2, amplitude, intensity = list(x), list(y), [], [] # Skip PSF fit

    # # Plot stars after PSF fit filtering
    # plotStars(ff, x2, y2)
    

    # Compute FWHM from one dimensional sigma
    sigma_x_fitted = np.array(sigma_x_fitted)
    sigma_y_fitted = np.array(sigma_y_fitted)
    sigma_fitted = np.sqrt(sigma_x_fitted**2 + sigma_y_fitted**2)
    fwhm = 2.355*sigma_fitted


    log.info('extracted ' + str(len(xy)) + ' stars from ' + ff_name)
    return ff_name, x2, y2, amplitude, intensity, fwhm



def twoDGaussian(params, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """ Defines a 2D Gaussian distribution. 
    
    Arguments:
        params: [tuple of floats] 
            - (x, y) independant variables, 
            - saturation: [int] Value at which saturation occurs
        amplitude: [float] amplitude of the PSF
        xo: [float] PSF center, X component
        yo: [float] PSF center, Y component
        sigma_x: [float] standard deviation X component
        sigma_y: [float] standard deviation Y component
        theta: [float] PSF rotation in radians
        offset: [float] PSF offset from the 0 (i.e. the "elevation" of the PSF)

    Return:
        g: [ndarray] values of the given Gaussian at (x, y) coordinates

    """

    x, y, saturation = params

    if isinstance(saturation, np.ndarray):
        saturation = saturation[0, 0]
    
    xo = float(xo)
    yo = float(yo)

    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp(-(a*((x - xo)**2) + 2*b*(x - xo)*(y - yo) + c*((y - yo)**2)))

    # Limit values to saturation level
    g[g > saturation] = saturation

    return g.ravel()



def fitPSF(ff, avepixel_mean, x2, y2, config):
    """ Fit a 2D Gaussian to the star candidate cutout to check if it's a star.
    
    Arguments:
        ff: [ff bin struct] FF bin file loaded in the FF bin structure
        avepixel_mean: [float] mean of the avepixel image
        x2: [list] a list of estimated star position (X axis)
        xy: [list] a list of estimated star position (Y axis)
        config: [config object] configuration object (loaded from the .config file)
    """

    # Load parameters form config if present
    if config is not None:
        # segment_radius: [int] radius (in pixels) of image segment around the detected star on which to 
        #     perform the fit
        # roundness_threshold: [float] minimum ratio of 2D Gaussian sigma X and sigma Y to be taken as a stars
        #     (hot pixels are narrow, while stars are round)
        # max_feature_ratio: [float] maximum ratio between 2 sigma of the star and the image segment area
        segment_radius = config.segment_radius
        roundness_threshold = config.roundness_threshold
        max_feature_ratio = config.max_feature_ratio


    x_fitted = []
    y_fitted = []
    amplitude_fitted = []
    intensity_fitted = []
    sigma_y_fitted = []
    sigma_x_fitted = []

    # Set the initial guess
    initial_guess = (30.0, segment_radius, segment_radius, 1.0, 1.0, 0.0, avepixel_mean)
    
    # Go through all stars
    for star in zip(list(y2), list(x2)):

        y, x = star

        y_min = y - segment_radius
        y_max = y + segment_radius
        x_min = x - segment_radius
        x_max = x + segment_radius

        if y_min < 0:
            y_min = np.array([0])
        if y_max > ff.nrows:
            y_max = np.array([ff.nrows])
        if x_min < 0:
            x_min = np.array([0])
        if x_max > ff.ncols:
            x_max = np.array([ff.ncols])

        # Check if any of these values is NaN and skip the star
        if np.any(np.isnan([x_min, x_max, y_min, y_max])):
            continue

        x_min = int(x_min)
        x_max = int(x_max)
        y_min = int(y_min)
        y_max = int(y_max)

        # Extract an image segment around each star
        star_seg = ff.avepixel[y_min:y_max, x_min:x_max]

        # Create x and y indices
        y_ind, x_ind = np.indices(star_seg.shape)

        # Estimate saturation level from image type
        saturation = (2**(8*star_seg.itemsize) - 1)*np.ones_like(y_ind)

        # Fit a PSF to the star
        try:
            # Fit the 2D Gaussian with the limited number of iterations - this reduces the processing time
            # and most of the bad star candidates take more iterations to fit
            popt, pcov = opt.curve_fit(twoDGaussian, (y_ind, x_ind, saturation), star_seg.ravel(), \
                p0=initial_guess, maxfev=200)
            # print(popt)
        except RuntimeError:
            # print('Fitting failed!')

            # Skip stars that can't be fitted in 200 iterations
            continue

        # Unpack fitted gaussian parameters
        amplitude, yo, xo, sigma_y, sigma_x, theta, offset = popt

        # Filter hot pixels by looking at the ratio between x and y sigmas (HPs are very narrow)
        if min(sigma_y/sigma_x, sigma_x/sigma_y) < roundness_threshold:
            # Skip if it is a hot pixel
            continue

        # Reject the star candidate if it is too large 
        if (4*sigma_x*sigma_y/segment_radius**2 > max_feature_ratio):
            continue


        ### If the fitting was successfull, compute the star intensity

        # Crop the star segment to take 3 sigma portion around the star
        crop_y_min = int(yo - 3*sigma_y) + 1
        if crop_y_min < 0: crop_y_min = 0
        
        crop_y_max = int(yo + 3*sigma_y) + 1
        if crop_y_max >= star_seg.shape[0]: crop_y_max = star_seg.shape[0] - 1

        crop_x_min = int(xo - 3*sigma_x) + 1
        if crop_x_min < 0: crop_x_min = 0

        crop_x_max = int(xo + 3*sigma_x) + 1
        if crop_x_max >= star_seg.shape[1]: crop_x_max = star_seg.shape[1] - 1

        # If the segment is too small, set a fixed size
        if (y_max - y_min) < 3:
            crop_y_min = int(yo - 2)
            crop_y_max = int(yo + 2)

        if (x_max - x_min) < 3:
            crop_x_min = int(xo - 2)
            crop_x_max = int(xo + 2)


        star_seg_crop = star_seg[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        # Skip the star if the shape is too small
        if (star_seg_crop.shape[0] == 0) or (star_seg_crop.shape[1] == 0):
            continue

        # Gamma correct the star segment
        star_seg_crop = Image.gammaCorrection(star_seg_crop.astype(np.float32), config.gamma)

        # Correct the background for gamma
        bg_corrected = Image.gammaCorrection(offset, config.gamma)

        # Subtract the background from the star segment and compute the total intensity
        intensity = np.sum(star_seg_crop - bg_corrected)

        # Skip stars with zero intensity
        if intensity <= 0:
            continue

        # print(intensity)
        # plt.imshow(star_seg_crop - bg_corrected, cmap='gray', vmin=0, vmax=255)
        # plt.show()


        ###

        # Calculate the intensity (as a volume under the 2D Gaussian) (OLD, before gamma correction)
        # intensity = 2*np.pi*amplitude*sigma_x*sigma_y



        # # Skip if the star intensity is below background level
        # if intensity < offset:
        #     continue

        # Add stars to the final list
        x_fitted.append(x_min + xo)
        y_fitted.append(y_min + yo)
        amplitude_fitted.append(amplitude)
        intensity_fitted.append(intensity)
        sigma_y_fitted.append(sigma_y)
        sigma_x_fitted.append(sigma_x)

        # # Plot fitted stars
        # data_fitted = twoDGaussian((y_ind, x_ind), *popt) - offset

        # fig, ax = plt.subplots(1, 1)
        # ax.hold(True)
        # plt.title('Center Y: '+str(y_min[0])+', X:'+str(x_min[0]))
        # ax.imshow(star_seg.reshape(segment_radius*2, segment_radius*2), cmap=plt.cm.inferno, origin='bottom',
        #     extent=(x_ind.min(), x_ind.max(), y_ind.min(), y_ind.max()))
        # # ax.imshow(data_fitted.reshape(segment_radius*2, segment_radius*2), cmap=plt.cm.jet, origin='bottom')
        # ax.contour(x_ind, y_ind, data_fitted.reshape(segment_radius*2, segment_radius*2), 8, colors='w')

        # plt.show()
        # plt.clf()
        # plt.close()

    return x_fitted, y_fitted, amplitude_fitted, intensity_fitted, sigma_y_fitted, sigma_x_fitted




def plotStars(ff, x2, y2):
    """ Plots detected stars on the input image.
    """

    # Plot image with adjusted levels to better see stars
    plt.imshow(Image.adjustLevels(ff.avepixel, 0, 1.3, 255), cmap='gray')

    # Plot stars
    for star in zip(list(y2), list(x2)):
        y, x = star
        c = plt.Circle((x, y), 5, fill=False, color='r')
        plt.gca().add_patch(c)

    plt.show()

    plt.clf()
    plt.close()




def extractStarsAndSave(config, ff_dir):
    """ Extract stars in the given folder and save the CALSTARS file. 
    
    Arguments:
        config: [config object] configuration object (loaded from the .config file)
        ff_dir: [str] Path to directory where FF files are.

    Return:
        star_list: [list] A list of [ff_name, star_data] entries, where star_data contains a list of 
            (column, row, amplitude, intensity, fwhm) values for every star.

    """




    time_start = time.time()

    # Load mask, dark, flat
    mask, dark, flat_struct = loadImageCalibration(ff_dir, config)
    

    extraction_list = []

    # Go through all files in the directory and add them to the detection list
    for ff_name in sorted(os.listdir(ff_dir)):

        # Check if the given file is a valid FF file
        if not FFfile.validFFName(ff_name):
            continue

        extraction_list.append(ff_name)



    # Run the QueuedPool for detection
    workpool = QueuedPool(extractStars, cores=-1, backup_dir=ff_dir, input_queue_maxsize=None)


    # Add jobs for the pool
    for ff_name in extraction_list:
        log.info('Adding for extraction: ' + ff_name)
        workpool.addJob([ff_dir, ff_name, config, None, None, None, None, flat_struct, dark, mask])


    log.info('Starting pool...')

    # Start the detection
    workpool.startPool()


    log.info('Waiting for the detection to finish...')

    # Wait for the detector to finish and close it
    workpool.closePool()


    # Get extraction results
    star_list = []
    for result in workpool.getResults():

        ff_name, x2, y2, amplitude, intensity, fwhm_data = result

        # Skip if no stars were found
        if not x2:
            continue


        # Construct the table of the star parameters
        star_data = list(zip(y2, x2, amplitude, intensity, fwhm_data))

        # Add star info to the star list
        star_list.append([ff_name, star_data])



    dir_name = os.path.basename(os.path.abspath(ff_dir))
    if dir_name.startswith(config.stationID):
        prefix = dir_name
    else:
        prefix = "{:s}_{:s}".format(config.stationID, dir_name)

    # Generate the name for the CALSTARS file
    calstars_name = 'CALSTARS_' + prefix + '.txt'


    # Write detected stars to the CALSTARS file
    CALSTARS.writeCALSTARS(star_list, ff_dir, calstars_name, config.stationID, config.height, config.width)

    # Delete QueudPool backed up files
    workpool.deleteBackupFiles()

    log.info('Total time taken: {:.2f} s'.format(time.time() - time_start))


    return star_list




if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Extract stars on FF files in the given folder.")

    arg_parser.add_argument('dir_path', nargs=1, metavar='DIR_PATH', type=str, \
        help='Path to the folder with FF files.')

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one.")

    arg_parser.add_argument('-s', '--showstd', action="store_true", help="""Show a histogram of stddevs of PSFs of all detected stars. """)

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, cml_args.dir_path)

    # Get paths to every FF bin file in a directory 
    ff_dir = os.path.abspath(cml_args.dir_path[0])
    ff_list = [ff_name for ff_name in os.listdir(ff_dir) if FFfile.validFFName(ff_name)]

    # Check if there are any file in the directory
    if len(ff_list) is None:
        log.warning("No files found!")
        sys.exit()



    # Run extraction and save the resulting CALSTARS file
    star_list = extractStarsAndSave(config, ff_dir)


    fwhm_list = []
    intensity_list = []
    x_list = []
    y_list = []


    # Print found stars
    for ff_name, star_data in star_list:

        print()
        print(ff_name)
        print('  ROW     COL       amp  intens FWHM')
        for x, y, max_ampl, level, fwhm in star_data:
            print(' {:7.2f} {:7.2f} {:6d} {:6d} {:5.2f}'.format(round(y, 2), round(x, 2), int(max_ampl), \
                int(level), fwhm))


        x2, y2, amplitude, intensity, fwhm_data = np.array(star_data).T

        # Store the star info to list
        fwhm_list += fwhm_data.tolist()
        intensity_list += intensity.tolist()
        x_list += x2.tolist()
        y_list += y2.tolist()


        # # Show stars if there are only more then 10 of them
        # if len(x2) < 20:
        #     continue

        # # Load the FF bin file
        # ff = FFfile.read(ff_dir, ff_name)

        # plotStars(ff, x2, y2)


    


    # Show the histogram of PSF FWHMs
    if cml_args.showstd:

        print('Median FWHM: {:.3f}'.format(np.median(fwhm_list)))

        # Compute the bin number
        nbins = int(np.ceil(np.sqrt(len(fwhm_list))))
        if nbins < 10:
            nbins = 10

        plt.hist(fwhm_list, bins=nbins)

        plt.xlabel('PSF FWHM')
        plt.ylabel('Count')

        plt.savefig(os.path.join(ff_dir, 'PSF_FWHM_hist.png'), dpi=300)

        plt.show()


        # Plot stddev by intensity
        
        hexbin_grid = int(1.0/np.sqrt(2)*nbins)
        lsp_list = -2.5*np.log10(np.array(intensity_list))
        fwhm_list = np.array(fwhm_list)

        # Compute plot limits
        x_min = np.percentile(lsp_list[~np.isnan(lsp_list)], 0.5)
        x_max = np.percentile(lsp_list[~np.isnan(lsp_list)], 99.5)
        y_min = np.percentile(fwhm_list[~np.isnan(fwhm_list)], 0.5)
        y_max = np.percentile(fwhm_list[~np.isnan(fwhm_list)], 99.5)

        plt.hexbin(lsp_list, fwhm_list, gridsize=(hexbin_grid, hexbin_grid), extent=(x_min, x_max, \
            y_min, y_max))
        plt.xlabel('Uncalibrated magnitude')
        plt.ylabel('PSF FWHM')

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.gca().invert_xaxis()

        plt.savefig(os.path.join(ff_dir, 'PSF_FWHM_vs_mag.png'), dpi=300)

        plt.show()


        # Plot stddev by X and Y
        fig, (ax1, ax2) = plt.subplots(nrows=2)

        x_min = np.min(x_list)
        x_max = np.max(x_list)
        ax1.hexbin(x_list, fwhm_list, gridsize=(hexbin_grid, hexbin_grid), extent=(x_min, x_max, \
            y_min, y_max))

        
        ax1.set_ylabel('PSF FWHM')
        ax1.set_xlabel('X')

        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)


        x_min = np.min(y_list)
        x_max = np.max(y_list)
        ax2.hexbin(y_list, fwhm_list, gridsize=(hexbin_grid, hexbin_grid), extent=(x_min, x_max, \
            y_min, y_max))

        ax2.set_ylabel('PSF FWHM')
        ax2.set_xlabel('Y')

        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)

        plt.tight_layout()

        plt.savefig(os.path.join(ff_dir, 'PSF_xy_vs_FWHM.png'), dpi=300)

        plt.show()


        # Plot stddev by radius from centre
        x_list = np.array(x_list)
        y_list = np.array(y_list)
        radius_list = np.hypot(x_list - config.width/2, y_list - config.height/2)

        radius_min = 0
        radius_max = np.hypot(config.width/2, config.height/2)

        plt.hexbin(radius_list, np.array(fwhm_list), gridsize=(hexbin_grid, hexbin_grid), \
            extent=(radius_min, radius_max, y_min, y_max))


        plt.xlabel("Radius from center (px)")
        plt.ylabel("PSF FWHM (px)")

        plt.xlim(radius_min, radius_max)
        plt.ylim(y_min, y_max)

        plt.tight_layout()

        plt.savefig(os.path.join(ff_dir, 'PSF_radius_vs_FWHM.png'), dpi=300)

        plt.show()



