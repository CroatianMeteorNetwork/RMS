
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
from RMS.Logger import getLogger
from RMS.Math import twoDGaussian
from RMS.Routines import MaskImage
from RMS.Routines import Image
from RMS.QueuedPool import QueuedPool

# Morphology - Cython init
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
#import RMS.Routines.MorphCy as morph


# Get the logger from the main module
log = getLogger("logger")



def extractStars(img, img_median=None, mask=None, gamma=1.0, max_star_candidates=1000, border=10,
                 neighborhood_size=10, intensity_threshold=18, 
                 segment_radius=4, roundness_threshold=0.5, max_feature_ratio=0.8, bit_depth=8):
    """ Extracts stars on a given image by searching for local maxima and applying PSF fit for star 
        confirmation.

    Arguments:
        img: [ndarray] Image data.

    Keyword arguments:
        img_median: [float] Median value of the image. If not given, it will be computed.
        mask: [ndarray] Mask image. None by default.
        gamma: [float] Gamma correction factor for the image.
        max_star_candidates: [int] Maximum number of star candidates to process. If the number of 
            candidates is larger than this number, the image will be skipped.
        border: [int] apply a mask on the detections by removing all that are too close to the given image 
            border (in pixels)
        neighborhood_size: [int] size of the neighbourhood for the maximum search (in pixels)
        intensity_threshold: [float] a threshold for cutting the detections which are too faint (0-255)
        segment_radius: [int] Radius (in pixels) of image segment around the detected star on which to 
            perform the fit.
        roundness_threshold: [float] Minimum ratio of 2D Gaussian sigma X and sigma Y to be taken as a stars
            (hot pixels are narrow, while stars are round).
        max_feature_ratio: [float] Maximum ratio between 2 sigma of the star and the image segment area.
        bit_depth: [int] Bit depth of the image. 8 bits by default.
    
    Return:
        x2, y2, background, intensity, fwhm: [list of ndarrays]
            - x2: X axis coordinates of the star
            - y2: Y axis coordinates of the star
            - background: background intensity
            - intensity: intensity of the star
            - Gaussian Full width at half maximum (FWHM) of fitted stars

    """


    # Compute the image median if not given
    if img_median is None:
        img_median = np.median(img)

    # Apply a mean filter to the image to reduce noise
    img_convolved = ndimage.filters.convolve(img, weights=np.full((2, 2), 1.0/4))

    # Locate local maxima on the image
    img_max = filters.maximum_filter(img_convolved, neighborhood_size)
    maxima = (img_convolved == img_max)
    img_min = filters.minimum_filter(img_convolved, neighborhood_size)
    diff = ((img_max - img_min) > intensity_threshold)
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
    if num_objects > max_star_candidates:
        log.warning('Too many candidate stars to process! {:d}/{:d}'.format(num_objects, max_star_candidates))
        return False

    # Find centres of mass of each labeled objects
    xy = np.array(ndimage.center_of_mass(img_convolved, labeled, range(1, num_objects + 1)))

    # Remove all detection on the border
    #xy = xy[np.where((xy[:, 1] > border) & (xy[:,1] < ff.ncols - border) & (xy[:,0] > border) & (xy[:,0] < ff.nrows - border))]

    # Unpack star coordinates
    y_init, x_init = np.hsplit(xy, 2)

    # Compensate for half-pixel shift caused by the 2x2 mean filter
    x_init = [x + 0.5 for x in x_init]
    y_init = [y + 0.5 for y in y_init]

    # # Plot stars before the PSF fit
    # plotStars(ff, x, y)

    # Fit a PSF to each star on the raw image
    (
        x_arr, y_arr, amplitude, intensity, 
        sigma_y_fitted, sigma_x_fitted, background, snr, saturated_count
    ) = fitPSF(
        img, img_median, x_init, y_init, 
        gamma=gamma,
        segment_radius=segment_radius, roundness_threshold=roundness_threshold, 
        max_feature_ratio=max_feature_ratio, bit_depth=bit_depth
        )
    
    # x_arr, y_arr, amplitude, intensity = list(x), list(y), [], [] # Skip PSF fit

    # # Plot stars after PSF fit filtering
    # plotStars(ff, x_arr, y_arr)
    

    # Compute FWHM from one dimensional sigma
    sigma_x_fitted = np.array(sigma_x_fitted)
    sigma_y_fitted = np.array(sigma_y_fitted)
    sigma_fitted = np.sqrt(sigma_x_fitted**2 + sigma_y_fitted**2)
    fwhm = 2.355*sigma_fitted

    return x_arr, y_arr, amplitude, intensity, fwhm, background, snr, saturated_count


def extractStarsAuto(img, mask=None, 
        max_star_candidates=1500, segment_radius=8, 
        min_stars_detect=50, max_stars_detect=150,
        bit_depth=8,
        verbose=False
        ):
    """ Automatically tried to extract stars from the given image by trying different intensity thresholds.
    
    Arguments:
        img: [ndarray] Image data.

    Keyword arguments:
        mask: [ndarray] Mask image. None by default.
        max_star_candidates: [int] Maximum number of star candidates when trying an intensity threshold.
            If there are too many, PSF fitting would take too long.
        segment_radius: [int] Radius (in pixels) of image segment around the detected star on which to 
            perform the fit.
        min_stars_detect: [int] Minimum number of stars retrieved with a given intensity threshold before
            a new one is tried.
        max_stars_detect: [int] Maximum number of stars to be detected before the process is stopped.
        bit_depth: [int] Bit depth of the image. 8 bits by default.
        verbose: [bool] Print verbose output.
    
    """

    # Precompute the median of the image
    img_median = np.median(img)

    x_data = []
    y_data = []
    amplitude = []
    intensity = []
    fwhm = []

    # Try different intensity thresholds until the greatest number of stars is found
    intens_thresh_list = [70, 50, 40, 30, 20, 10, 5]

    # Repeat the process until the number of returned stars falls within the range
    min_stars_detect = 50
    max_stars_detect = 150
    for intens_thresh in intens_thresh_list:

        if verbose:
            print("Detecting stars with intensity threshold: ", intens_thresh)

        status = extractStars(img, img_median=img_median, mask=mask, 
                                max_star_candidates=max_star_candidates, segment_radius=segment_radius, 
                                intensity_threshold=intens_thresh, bit_depth=bit_depth)

        if status == False:
            continue

        x_data, y_data, amplitude, intensity, fwhm, background, snr, saturated_count = status
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        if len(x_data) < min_stars_detect:

            if verbose:
                print("Skipping, the number of stars {:d} outside {:d} - {:d} range".format(
                    len(x_data), min_stars_detect, max_stars_detect))
            
            continue
        
        elif len(x_data) > max_stars_detect:
            
            # If too many stars are found even with the first very high threshold, take that solution
            break

        else:
            break


    return x_data, y_data, amplitude, intensity, fwhm, background, snr, saturated_count


def extractStarsFF(
        ff_dir, ff_name, 
        flat_struct=None, dark=None, mask=None,
        config=None, 
        border=10,
        max_global_intensity=150, 
        neighborhood_size=10, intensity_threshold=18, 
        segment_radius=4, roundness_threshold=0.5, max_feature_ratio=0.8
        ):
    """ Extracts stars on a given FF bin by searching for local maxima and applying PSF fit for star 
        confirmation.

        Source of one part of the code: 
    http://stackoverflow.com/questions/9111711/get-coordinates-of-local-maxima-in-2d-array-above-certain-value
    
    Arguments:
        ff_dir: [str] Path to directory where FF files are.
        ff_name: [str] Name of the FF file.
        config: [config object] configuration object (loaded from the .config file)
        max_global_intensity: [int] maximum mean intensity of an image before it is discarded as too bright
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
    error_return = [[], [], [], [], [], [], [], [], []]

    # Load parameters from config if given
    if config is not None:
        max_global_intensity = config.max_global_intensity
        border = config.border
        neighborhood_size = config.neighborhood_size
        intensity_threshold = config.intensity_threshold
        segment_radius = config.segment_radius
        roundness_threshold = config.roundness_threshold
        max_feature_ratio = config.max_feature_ratio
        
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
    img_median = np.median(ff.avepixel)

    # Check if the image is too bright and skip the image
    if img_median > max_global_intensity:
        return error_return

    # Get the image data from the average pixel image
    img = ff.avepixel.astype(np.float32)


    # Find the stars in the image
    status = extractStars(
        img, img_median=img_median, 
        mask=mask, gamma=config.gamma,
        max_star_candidates=config.max_stars, border=border,
        neighborhood_size=neighborhood_size, intensity_threshold=intensity_threshold, 
        segment_radius=segment_radius, roundness_threshold=roundness_threshold, 
        max_feature_ratio=max_feature_ratio, bit_depth=config.bit_depth
    )

    # If the star extraction failed, return an empty list
    if status is False:
        return error_return
    
    # Unpack the star data
    x_arr, y_arr, amplitude, intensity, fwhm, background, snr, saturated_count = status


    log.info('extracted ' + str(len(x_arr)) + ' stars from ' + ff_name)
    return ff_name, x_arr, y_arr, amplitude, intensity, fwhm, background, snr, saturated_count


def extractStarsImgHandle(img_handle,
        flat_struct=None, dark=None, mask=None,
        config=None, 
        border=10,
        max_global_intensity=150, 
        neighborhood_size=10, intensity_threshold=18, 
        segment_radius=4, roundness_threshold=0.5, max_feature_ratio=0.8
    ):

    """ Extracts stars on a given image handle by searching for local maxima and applying PSF fit for star 
        confirmation.

    Arguments:
        img_handle: [FrameInterface instance] Image data handle.

    Keyword arguments:
        flat_struct: [Flat struct] Structure containing the flat field. None by default.
        dark: [ndarray] Dark frame. None by default.
        mask: [ndarray] Mask image. None by default.
        config: [config object] configuration object (loaded from the .config file)
        max_global_intensity: [int] maximum mean intensity of an image before it is discarded as too bright
        border: [int] apply a mask on the detections by removing all that are too close to the given image 
            border (in pixels)
        neighborhood_size: [int] size of the neighbourhood for the maximum search (in pixels)
        intensity_threshold: [float] a threshold for cutting the detections which are too faint (0-255)
        segment_radius: [int] Radius (in pixels) of image segment around the detected star on which to 
            perform the fit.
        roundness_threshold: [float] Minimum ratio of 2D Gaussian sigma X and sigma Y to be taken as a stars
            (hot pixels are narrow, while stars are round).
        max_feature_ratio: [float] Maximum ratio between 2 sigma of the star and the image segment area.

    Return:
        x2, y2, background, intensity, fwhm: [list of ndarrays]
            - x2: X axis coordinates of the star
            - y2: Y axis coordinates of the star
            - background: background intensity
            - intensity: intensity of the star
            - Gaussian Full width at half maximum (FWHM) of fitted stars
    """

    # This will be returned if there was an error
    error_return = [[], [], [], [], [], [], [], [], []]

    # Load parameters from config if given
    if config is not None:
        max_global_intensity = config.max_global_intensity
        border = config.border
        neighborhood_size = config.neighborhood_size
        intensity_threshold = config.intensity_threshold
        segment_radius = config.segment_radius
        roundness_threshold = config.roundness_threshold
        max_feature_ratio = config.max_feature_ratio


    star_list = []


    # Set the reference frame to 0
    img_handle.setFrame(0)

    # Go through all the chunks in the image handle
    for chunk_no in range(img_handle.total_fr_chunks):

        # Load one video frame chunk
        ff_tmp = img_handle.loadChunk()

        # Extract the image to work on
        avepixel = ff_tmp.avepixel


        # Apply the dark frame
        if dark is not None:
            avepixel = Image.applyDark(avepixel, dark)

        # Apply the flat
        if flat_struct is not None:
            avepixel = Image.applyFlat(avepixel, flat_struct)

        # Mask the FF file
        if mask is not None:
            avepixel = MaskImage.applyMask(avepixel, mask, ff_flag=False)


        # Calculate image mean and stddev
        img_median = np.median(avepixel)

        # Check if the image is too bright and skip the image
        if img_median > max_global_intensity:
            return error_return

        # Get the image data from the average pixel image
        img = avepixel.astype(np.float32)

        # Extract stars from the average pixel image
        status = extractStars(
            img, img_median=img_median, 
            mask=mask, gamma=config.gamma,
            max_star_candidates=config.max_stars, border=border,
            neighborhood_size=neighborhood_size, intensity_threshold=intensity_threshold, 
            segment_radius=segment_radius, roundness_threshold=roundness_threshold, 
            max_feature_ratio=max_feature_ratio
        )

        # If the star extraction failed, return an empty list
        if status is False:
            return error_return
        
        # Unpack the star data
        x_arr, y_arr, amplitude, intensity, fwhm, background, snr, saturated_count = status


        # Construct an FF name from the chunk time
        ff_name = FFfile.constructFFName(
            config.stationID, img_handle.currentTime(dt_obj=True, beginning=True)
            )

        # Print the results
        print()
        print("FF name:", ff_name)
        print("Num frames:", img_handle.chunk_frames)
        print("Number of stars:", len(x_arr))
        for x, y, a, i, f, bg, s, satcnt in zip(x_arr, y_arr, amplitude, intensity, fwhm, background, snr, saturated_count):
            print("{:7.2f} {:7.2f} {:9d} {:6d} {:5.2f} {:6d} {:5.2f} {:6d}".format(
                round(y, 2), round(x, 2), 
                int(a), int(i), f, int(bg), s, int(satcnt)
                )
            )


        star_list.append(
            [ff_name, list(zip(y_arr, x_arr, amplitude, intensity, fwhm, background, snr, saturated_count))]
             )

        # Go to the next chunk
        img_handle.nextChunk()
    

    # If the star list is empty, return the error return
    if not star_list:
        return error_return

    return star_list




def fitPSF(img, img_median, x_init, y_init, gamma=1.0, segment_radius=4, roundness_threshold=0.5, 
           max_feature_ratio=0.8, bit_depth=8):
    """ Fit a 2D Gaussian to the star candidate cutout to check if it's a star.
    
    Arguments:
        img: [ndarray] Image data.
        x_init: [list] A list of estimated star position (X axis).
        y_init: [list] A list of estimated star position (Y axis).
        
    Keyword arguments:
        gamma: [float] Gamma correction factor for the image.
        segment_radius: [int] Radius (in pixels) of image segment around the detected star on which to 
            perform the fit.
        roundness_threshold: [float] Minimum ratio of 2D Gaussian sigma X and sigma Y to be taken as a stars
            (hot pixels are narrow, while stars are round).
        max_feature_ratio: [float] Maximum ratio between 2 sigma of the star and the image segment area.
        bit_depth: [int] Bit depth of the image.

    """


    x_fitted = []
    y_fitted = []
    amplitude_fitted = []
    intensity_fitted = []
    sigma_y_fitted = []
    sigma_x_fitted = []
    background_fitted = []
    snr_fitted = []
    saturated_count_fitted = []

    # Set the initial guess
    initial_guess = (30.0, segment_radius, segment_radius, 1.0, 1.0, 0.0, img_median)

    # Get the image dimensions
    nrows, ncols = img.shape

    # Threshold for the reported numbers of saturated pixels (98% of the dynamic range)
    saturation_threshold_report = int(round(0.98*(2**bit_depth - 1)))
    
    
    # Go through all stars
    for star in zip(list(y_init), list(x_init)):

        y, x = star

        y_min = y - segment_radius
        y_max = y + segment_radius
        x_min = x - segment_radius
        x_max = x + segment_radius

        if y_min < 0:
            y_min = np.array([0])
        if y_max > nrows:
            y_max = np.array([nrows])
        if x_min < 0:
            x_min = np.array([0])
        if x_max > ncols:
            x_max = np.array([ncols])

        # Check if any of these values is NaN and skip the star
        if np.any(np.isnan([x_min, x_max, y_min, y_max])):
            continue

        x_min = int(x_min)
        x_max = int(x_max)
        y_min = int(y_min)
        y_max = int(y_max)

        # Extract an image segment around each star
        star_seg = img[y_min:y_max, x_min:x_max]

        # Create x and y indices
        y_ind, x_ind = np.indices(star_seg.shape)

        # Estimate saturation level from image type
        saturation = (2**bit_depth - 1)*np.ones_like(y_ind)

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

        # Take absolute values of some parameters
        amplitude = abs(amplitude)
        sigma_x = abs(sigma_x)
        sigma_y = abs(sigma_y)

        # Filter hot pixels by looking at the ratio between x and y sigmas (HPs are very narrow)
        if min(sigma_y/sigma_x, sigma_x/sigma_y) < roundness_threshold:
            # Skip if it is a hot pixel
            continue

        # Reject the star candidate if it is too large 
        if (4*sigma_x*sigma_y/segment_radius**2 > max_feature_ratio):
            continue


        ### If the fitting was successful, compute the star intensity

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
        star_seg_crop_corr = Image.gammaCorrectionImage(star_seg_crop.astype(np.float32), gamma)

        # Correct the background for gamma
        bg_corrected = Image.gammaCorrectionScalar(offset, gamma)

        # Subtract the background from the star segment and compute the total intensity
        intensity = np.sum(star_seg_crop_corr - bg_corrected)

        # Skip stars with zero intensity
        if intensity <= 0:
            continue


        ### Compute the star's SNR

        # Compute the number of pixels inside the 3 sigma ellipse around the star
        star_px_area = np.pi*(3*sigma_x)*(3*sigma_y)

        # Estimate the standard deviation of the background, which is area outside the 3 sigma ellipse
        star_seg_crop_nan = np.copy(star_seg_crop_corr)
        star_seg_crop_nan[crop_y_min:crop_y_max, crop_x_min:crop_x_max] = np.nan
        bg_std = np.nanstd(star_seg_crop_nan)

        # Make sure the background standard deviation is not zero
        if (bg_std <= 0) or np.isnan(bg_std):
            bg_std = 1

        # Compute the SNR
        snr = Image.signalToNoise(intensity, star_px_area, bg_corrected, bg_std)

        ###


        ### Determine the number of saturated pixels ###

        # Count the number of saturated pixels (before gamma correction)
        saturated_count = np.sum(star_seg_crop >= saturation_threshold_report)

        ###


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
        background_fitted.append(bg_corrected)
        snr_fitted.append(snr)
        saturated_count_fitted.append(saturated_count)

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

    return (
            x_fitted, y_fitted, 
            amplitude_fitted, intensity_fitted, 
            sigma_y_fitted, sigma_x_fitted, 
            background_fitted, snr_fitted, saturated_count_fitted
            )




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


    # If just one file is given, run the extraction on it instead of using the QueuedPool
    workpool = None
    if len(extraction_list) == 1:
        ff_name = extraction_list[0]

        log.info('Extracting stars from ' + ff_name)

        # Run the extraction
        result = extractStarsFF(
            ff_dir, ff_name, flat_struct=flat_struct, dark=dark, mask=mask,
            config=config
        )

        results = [result]


    else:

        # The number of workers should be the minimum of cores and the number of tasks, so we don't have too many
        # workers waiting for the tasks to finish
        num_cores = min(config.num_cores, len(extraction_list))

        # Run the QueuedPool for detection
        workpool = QueuedPool(extractStarsFF, cores=num_cores, backup_dir=ff_dir, input_queue_maxsize=None)


        # Add jobs for the pool
        for ff_name in extraction_list:
            log.info('Adding for extraction: ' + ff_name)
            workpool.addJob([ff_dir, ff_name, flat_struct, dark, mask, config, None, None, None, None, None, None, None])


        log.info('Starting pool...')

        # Start the detection
        workpool.startPool()


        log.info('Waiting for the detection to finish...')

        # Wait for the detector to finish and close it
        workpool.closePool()

        results = workpool.getResults()


    # Get extraction results
    star_list = []
    for result in results:

        try:
            ff_name, x2, y2, amplitude, intensity, fwhm_data, background, snr, saturated_count = result
            
        except ValueError:
            ff_name, x2, y2, amplitude, intensity, fwhm_data = result
            background = np.zeros_like(x2)
            snr = np.zeros_like(x2)
            saturated_count = np.zeros_like(x2)

        # Skip if no stars were found
        if not x2:
            continue


        # Construct the table of the star parameters
        star_data = list(zip(y2, x2, amplitude, intensity, fwhm_data, background, snr, saturated_count))

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

    # Delete QueuedPool backed up files
    if workpool is not None:
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
    background_list = []
    snr_list = []
    saturated_count_list = []


    # Print found stars
    for ff_name, star_data in star_list:

        print()
        print(ff_name)
        print('  ROW     COL       amp  intens FWHM Bg SNR SatCount')
        for x, y, max_ampl, level, fwhm, background, snr, saturated_count in star_data:
            print(' {:7.2f} {:7.2f} {:6d} {:6d} {:5.2f} {:6d} {:5.2f} {:6d}'.format(round(y, 2), round(x, 2), int(max_ampl), \
                int(level), fwhm, int(background), snr, saturated_count))


        x2, y2, amplitude, intensity, fwhm_data, background, snr, saturated_count = np.array(star_data).T

        # Store the star info to list        
        x_list += x2.tolist()
        y_list += y2.tolist()
        intensity_list += intensity.tolist()
        fwhm_list += fwhm_data.tolist()
        background_list += background.tolist()
        snr_list += snr.tolist()
        saturated_count_list += saturated_count.tolist()


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



