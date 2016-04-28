
# RPi Meteor Station
# Copyright (C) 2016  Denis Vida
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

import time
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

# RMS imports
import RMS.ConfigReader as cr
from RMS.Formats import FFbin
from RMS.Formats import CALSTARS





def extractStars(ff_dir, ff_name, config=None, max_global_intensity=150, border=10, neighborhood_size=10, 
        intensity_threshold=5):
    """ Extracts stars on a given FF bin by searching for local maxima and applying PSF fit for star 
        confirmation.

    @param ff: [ff bin struct] FF bin file loaded in the FF bin structure

    @param config: [config object] configuration object (loaded from the .config file)
    @param max_global_intensity: [int] maximum mean intensity of an image before it is discared as too bright
    @param border: [int] apply a mask on the detections by removing all that are too close to the given image 
        border (in pixels)
    @param neighborhood_size: [int] size of the neighbourhood for the maximum search (in pixels)
    @param intensity_threshold: [float] a threshold for cutting the detections which are too faint (0-255)

    """

    # Load parameters from config if given
    if config:
        max_global_intensity = config.max_global_intensity
        border = config.border
        neighborhood_size = config.neighborhood_size
        intensity_threshold = config.intensity_threshold
        

    # Load the FF bin file
    ff = FFbin.read(ff_dir, ff_name)

    # Calculate image mean and stddev
    global_mean = np.mean(ff.avepixel)

    # Check if the image is too bright and skip the image
    if global_mean > max_global_intensity:
        return [[], [], [], []]
    
    # Stretch image intensity with arcsinh
    data = np.arcsinh(ff.avepixel.astype(np.float32))
    data = data / data.max() * 255

    # Locate local maximums on the image
    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > intensity_threshold)
    maxima[diff == 0] = 0

    # Calculate centroids of the maximums
    labeled, num_objects = ndimage.label(maxima)
    xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)))

    # Remove all detection on the border
    xy = xy[np.where((xy[:, 1] > border) & (xy[:,1] < ff.ncols - border) & (xy[:,0] > border) & (xy[:,0] < ff.nrows - border))]

    # Unpack star coordinates
    y, x = np.hsplit(xy, 2)

    # plotStars(ff, x, y)

    # Fit a PSF to each star
    x2, y2, background, intensity = fitPSF(ff, global_mean, x, y, config=config)
    # x2, y2, background, intensity = list(x), list(y), [], []

    return x2, y2, background, intensity



def twoDGaussian((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """ Defines a 2D Gaussian distribution. 

    @param (x, y): [tuple of floats] independant variables
    @param amplitude: [float] amplitude of the PSF
    @param xo: [float] PSF center, X component
    @param yo: [float] PSF center, Y component
    @param sigma_x: [float] standard deviation X component
    @param sigma_y: [float] standard deviation Y component
    @param theta: [float] PSF rotation in radians
    @param offset: [float] PSF offset from the 0 (i.e. the "elevation" of the PSF)

    """
    
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()


def fitPSF(ff, avepixel_mean, x2, y2, config=None, segment_radius=4, roundness_threshold=0.5, max_feature_ratio=0.8):
    """ Fit 2D Gaussian distribution as the PSF on the star image. 

    @param ff: [ff bin struct] FF bin file loaded in the FF bin structure
    @param avepixel_mean: [float] mean of the avepixel image
    @param x2: [list] a list of estimated star position (X axis)
    @param xy: [list] a list of estimated star position (Y axis)

    @param config: [config object] configuration object (loaded from the .config file)
    @param segment_radius: [int] radius (in pixels) of image segment around the detected star on which to 
        perform the fit
    @param roundness_threshold: [float] minimum ratio of 2D Gaussian sigma X and sigma Y to be taken as a stars
        (hot pixels are narrow, while stars are round)
    @param max_feature_ratio: [float] maximum ratio between 2 sigma of the star and the image segment area

    """

    # Load parameters form config if present
    if config:
        segment_radius = config.segment_radius
        roundness_threshold = config.roundness_threshold
        max_feature_ratio = config.max_feature_ratio


    x_fitted = []
    y_fitted = []
    background_fitted = []
    intensity_fitted = []

    # Set the initial guess
    initial_guess = (30.0, segment_radius, segment_radius, 1.0, 1.0, -10.0, avepixel_mean)
    
    for star in zip(list(y2), list(x2)):

        y, x = star

        y_min = y - segment_radius
        y_max = y + segment_radius
        x_min = x - segment_radius
        x_max = x + segment_radius

        if y_min < 0:
            y_min = 0
        if y_max > ff.nrows:
            y_max = ff.nrows
        if x_min < 0:
            x_min = 0
        if x_max > ff.ncols:
            x_max = ff.ncols

        # Extract an image segment around each star
        star_seg = ff.avepixel[y_min:y_max, x_min:x_max]

        # Create x and y indices
        y_ind, x_ind = np.indices(star_seg.shape)

        # Fit a PSF to the star

        try:
            popt, pcov = opt.curve_fit(twoDGaussian, (y_ind, x_ind), star_seg.ravel(), p0=initial_guess, 
                maxfev=150)
            # print popt
        except:
            # print 'Fitting failed!'
            continue

        # Unpack fitted gaussian parameters
        amplitude, yo, xo, sigma_y, sigma_x, theta, offset = popt

        # Filter hot pixels by looking at the ratio between x and y sigmas (HPs are very narrow)
        if min(sigma_y/sigma_x, sigma_x/sigma_y) < roundness_threshold:
            # Skip if it is a hot pixel
            continue

        # Reject the star candidate if it is too large 
        if (4*sigma_x*sigma_y / segment_radius**2 > max_feature_ratio):
            continue

        # Calculate intensity (as a volume under 2 sigma 2D Gauss curve) - background intensity
        intensity = 8 * 3.14159 * amplitude * sigma_x * sigma_y - offset

        # Skip if the star intensity is below background level
        if intensity < offset:
            continue

        # Add stars to the final list
        x_fitted.append(x_min + xo)
        y_fitted.append(y_min + yo)
        background_fitted.append(offset)
        intensity_fitted.append(intensity)

        # # Plot fitted stars
        # data_fitted = twoDGaussian((y_ind, x_ind), *popt) - offset

        # fig, ax = plt.subplots(1, 1)
        # ax.hold(True)
        # plt.title('Y: '+str(y_min)+', X:'+str(x_min))
        # ax.imshow(star_seg.reshape(segment_radius*2, segment_radius*2), cmap=plt.cm.jet, origin='bottom',
        #     extent=(x_ind.min(), x_ind.max(), y_ind.min(), y_ind.max()))
        # # ax.imshow(data_fitted.reshape(segment_radius*2, segment_radius*2), cmap=plt.cm.jet, origin='bottom')
        # ax.contour(x_ind, y_ind, data_fitted.reshape(segment_radius*2, segment_radius*2), 8, colors='w')

        # plt.show()
        # plt.clf()
        # plt.close()

    return x_fitted, y_fitted, background_fitted, intensity_fitted


def plotStars(ff, x2, y2):
    """ Plots detected stars on the input image. """

    # Plot image
    plt.imshow(np.arcsinh(ff.avepixel), cmap='gray')

    # Plot stars
    for star in zip(list(y2), list(x2)):
        y, x = star
        c = plt.Circle((x, y), 5, fill=False, color='r')
        plt.gca().add_patch(c)

    plt.show()

    plt.clf()
    plt.close()



if __name__ == "__main__":

    time_start = time.clock()

    # Load config file
    config = cr.parse(".config")

    if not len(sys.argv) == 2:
        print "Usage: python -m RMS.ExtractStars /path/to/bin/files/"
        sys.exit()
    
    # Get paths to every FF bin file in a directory 
    ff_dir = os.path.abspath(sys.argv[1])
    ff_list = [ff_name for ff_name in os.listdir(ff_dir) if ff_name[0:2]=="FF" and ff_name[-3:]=="bin"]

    # Check if there are any file in the directory
    if(len(ff_list) == None):
        print "No files found!"
        sys.exit()


    star_list = []

    # Go through all files in the directory
    for ff_name in sorted(ff_list):

        print ff_name

        t1 = time.clock()

        x2, y2, background, intensity = extractStars(ff_dir, ff_name, config)

        print 'Time for extraction: ', time.clock() - t1

        # Skip if no stars were found
        if not x2:
            continue

        # Construct the table of the star parameters
        star_data = zip(x2, y2, background, intensity)

        # Add star info to the star list
        star_list.append([ff_name, star_data])

        # Print found stars
        print '   ROW    COL intensity'
        for x, y, bg_level, level in star_data:
            print ' {:06.2f} {:06.2f} {:6d} {:6d}'.format(round(y, 2), round(x, 2), int(bg_level), int(level))


        # # Show stars if there are only more then 10 of them
        # if len(x2) < 20:
        #     continue

        # # Load the FF bin file
        # ff = FFbin.read(ff_dir, ff_name)

        # plotStars(ff, x2, y2)

    # Load data about the image
    ff = FFbin.read(ff_dir, ff_name)

    # Generate the name for the CALSTARS file
    calstars_name = 'CALSTARS' + "{:04d}".format(int(ff.camno)) + ff_dir.split(os.sep)[-1] + '.txt'


    # Write detected stars to the CALSTARS file
    CALSTARS.makeCALSTARS(star_list, calstars_name, ff_dir, ff.camno, ff.nrows, ff.ncols)

    print 'Total time taken: ', time.clock() - time_start


