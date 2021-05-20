# RPi Meteor Station
# Copyright (C) 2016  Dario Zubovic, Denis Vida
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

from __future__ import print_function, absolute_import, division

import argparse
import logging
from time import time
import datetime
import sys, os
import ctypes
import traceback

import numpy as np
import numpy.ctypeslib as npct
import cv2

# Plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

# RMS imports
from RMS.Astrometry.Conversions import jd2Date, raDec2AltAz
import RMS.ConfigReader as cr
from RMS.DetectionTools import getThresholdedStripe3DPoints, loadImageCalibration, binImageCalibration
from RMS.Formats.AsgardEv import writeEv
from RMS.Formats.AST import xyToRaDecAST
from RMS.Formats import FFfile
from RMS.Formats import FTPdetectinfo
from RMS.Formats.FrameInterface import detectInputType
from RMS.Formats.AST import loadAST
from RMS.Misc import mkdirP
from RMS.Routines.Grouping3D import find3DLines, getAllPoints
from RMS.Routines.CompareLines import compareLines
from RMS.Routines import MaskImage
from RMS.Routines import Image
from RMS.Routines import RollingShutterCorrection
from RMS.Routines.Image import thresholdFF

# Morphology - Cython init
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
import RMS.Routines.MorphCy as morph
from RMS.Routines.BinImageCy import binImage

# If True, all detection details will be logged
VERBOSE_DEBUG = False


# Get the logger from the main module
log = logging.getLogger("logger")



def logDebug(*log_str):
    """ Log detection debug messages. """

    if VERBOSE_DEBUG:

        log_str = map(str, log_str)

        log.debug(" ".join(log_str))




def getPolarLine(x1, y1, x2, y2, img_h, img_w):
    """ Calculate polar line coordinates (Hough transform coordinates) rho and theta given the 2 points that 
        define a line in Cartesian coordinates. Coordinate system starts in the image center, to replicate 
        the used HT implementation.
    
    Arguments:
        x1: [float] X component of the first point
        y1: [float] Y component of the first point
        x2: [float] X component of the second point
        y2: [float] Y component of the second point
    
    Return:
        (rho, theta): [tuple of floats] rho (distance in px) and theta (angle in degrees) polar line 
            coordinates
    """

    x0 = float(img_w)/2
    y0 = float(img_h)/2

    dx = float(x2 - x1)
    dy = float(y2 - y1)

    # Calculate polar line coordinates
    theta = -np.arctan2(dx, dy)
    rho = (dy * x0 - dx * y0 + x2*y1 - y2*x1) / np.sqrt(dy**2 + dx**2)
    
    # Correct for quadrant
    if rho > 0:
        theta += np.pi
    else:
        rho = -rho

    return rho, np.degrees(theta)



def _getCartesian(rho, theta):
        """ Convert rho and theta to cartesian x and y points.
        """
        
        return np.cos(np.radians(theta))*rho, np.sin(np.radians(theta))*rho



def mergeLines(line_list, min_distance, img_w, img_h, last_count=0):
    """ Merge similar lines defined by rho and theta.
    
    Arguments:
        line_list: [list] A list of (rho, phi, min_frame, max_frame) tuples which define a KHT line.
        min_distance: [float] Minimum distance between two vectors described by line parameters for the lines 
            to be joined.
        img_w: [int] Image width.
        img_h: [int] Image height.

    Keyword arguments:
        last_count: [int] Used for recursion, default is 0 and it should be left as is!

    Return:
        final_list: [list] List of (rho, phi, min_frame, max_frame) tuples after line merging.

    """


    # Return if less than 2 lines
    if len(line_list) < 2:
        return line_list

    final_list = []
    paired_indices = []

    for i, line1 in enumerate(line_list):

        # Skip if the line was paired
        if i in paired_indices:
            continue

        found_pair = False

        # Get polar coordinates of line
        rho1, theta1, min_frame1, max_frame1 = line1

        

        for j, line2 in enumerate(line_list[i+1:]):

            # Get real j index in respect to line_list
            j = i + j + 1

            # Skip if the line was paired
            if j in paired_indices:
                continue

            # Get polar coordinates of line
            rho2, theta2, min_frame2, max_frame2 = line2

            # If the minimum frame of this line is larger than the maximum frame of the reference line
            #   skip this loop because there is no time overlap, and we know the lines are ordered by frame
            if min_frame2 > max_frame1:
                break


            # If there is no time overlap, skip this pair
            if not ((max_frame1 >= min_frame2) and (min_frame1 <= max_frame2)):
                continue


            # Check if the points are close enough
            if compareLines(rho1, theta1, rho2, theta2, img_w, img_h) < min_distance:

                # Remove old lines
                paired_indices.append(i)
                paired_indices.append(j)

                # Get cartesian coordinates of a point described by the polar vector
                x1, y1 = _getCartesian(rho1, theta1)

                # Get cartesian coordinates of a point described by the polar vector
                x2, y2 = _getCartesian(rho2, theta2)

                # Merge line
                x_avg = (x1 + x2)/2
                y_avg = (y2 + y1)/2

                # Return to polar space
                theta_avg = np.degrees(np.arctan2(y_avg, x_avg))
                rho_avg = np.sqrt(x_avg**2 + y_avg**2)

                # Choose the frame range
                frame_min = min((line1[2], line2[2]))
                frame_max = max((line1[3], line2[3]))

                # Add merged lines to line list
                final_list.append([rho_avg, theta_avg, frame_min, frame_max])

                found_pair = True

                break

        # Add point back to the list if pair was not found
        if not found_pair:
            final_list.append([rho1, theta1, line1[2], line1[3]])

    # Use recursion until the number of lines stabilizes
    if len(final_list) != last_count:
        final_list = mergeLines(final_list, min_distance, img_w, img_h, len(final_list))

    return final_list



def merge3DLines(line_list, vect_angle_thresh, last_count=0):
    """ Merge similar lines found by the 3D detector. 

        Calculate the vector between the first point of the first line and the last point of the second line, 
        and then compares the angle difference to individual line vectors. If all vecters have angles that are 
        close enough, merge the line. Frame ranges also have to overlap to merge the line.
    
    Arguments:
        line_list: [list] A list of lines found by grouping3D algorithm.
        vect_angle_thresh: [float] Minimum angle between vectors to merge the lines.

    Keyword arguments:
        last_count: [int] Used for recursion, default is 0 and it should be left as is!
    
    Return:
        final_list: [list] A list of merged lines.

    """

    def _vectorAngle(v1, v2):
        """ Calculate an angle (in degrees) between two vectors.

        Arguments:
            v1: [ndarray] first vector
            v2: [ndarray] second vector

        Return:
            angle: [float] angle in degrees
        """

        # Calculate angle between vectors
        cosang = np.dot(v1, v2)
        sinang = np.linalg.norm(np.cross(v1, v2))
        angle = np.degrees(np.arctan2(sinang, cosang))

        if angle > 180:
            angle = abs(angle - 360)

        elif angle < 0:
            angle = abs(angle)

        return angle


    # Return if less than 2 lines
    if len(line_list) < 2:
        return line_list

    final_list = []
    paired_indices = []

    for i, line1 in enumerate(line_list):

        # Skip if the line is already paired
        if i in paired_indices:
            continue

        # Extract first point
        x11, y11, z11 = line1[0]

        # Extract second point
        x12, y12, z12 = line1[1]

        # Get frame range
        line1_fmin, line1_fmax = line1[4:6]

        # Create a vector from line points
        v1 = np.array([x12-x11, y12-y11, z12-z11])

        found_pair = False

        for j, line2 in enumerate(line_list[i+1:]):

            # Get real j index value with respect to line_list
            j = j + i + 1

            # Skip if the line is already paired
            if j in paired_indices:
                continue

            # Extract first point
            x21, y21, z21 = line2[0]

            # Extract second point
            x22, y22, z22 = line2[1]

            # Get frame range
            line2_fmin, line2_fmax = line2[4:6]

            # Create a vector from line points
            v2 = np.array([x22-x21, y22-y21, z22-z21])


            # Create a vector from the first point of the first line and the last point of the second line
            v_both = np.array([x22 - x11, y22 - y11, z22 - z11])

            # Calculate the angle between the v_both and v1
            vect_angle1 = _vectorAngle(v1, v_both)

            # Calculate the angle between the v_both and v1
            vect_angle2 = _vectorAngle(v2, v_both)


            # Check if the vector angles are close enough to the vector that connects them
            if (vect_angle1 < vect_angle_thresh) and (vect_angle2 < vect_angle_thresh):

                # Check if the frames overlap
                if set(range(line1_fmin, line1_fmax+1)).intersection(range(line2_fmin, line2_fmax+1)):

                    # Choose the frame range
                    frame_min = min((line1_fmin, line2_fmin))
                    frame_max = max((line1_fmax, line2_fmax))

                    # Remove old lines
                    paired_indices.append(i)
                    paired_indices.append(j)

                    found_pair = True

                    # Add megred line to the list
                    final_list.append([line1[0], line1[1], max(line1[2], line2[2]), max(line1[3], line2[3]), frame_min, frame_max])

                    break

        # Add point back to the list if pair was not found
        if not found_pair:
            final_list.append(line1)

    # Use recursion until the number of lines stabilizes
    if len(final_list) != last_count:
        final_list = merge3DLines(final_list, vect_angle_thresh, len(final_list))

    return final_list



def checkWhiteRatio(img_thres, ff, max_white_ratio):
    """ Checks if there are too many threshold passers on an image. """

    # Check if the image is too "white" and any futher processing makes no sense
    # Compute the radio between the number of threshold passers and all pixels
    white_ratio = np.count_nonzero(img_thres)/float(ff.nrows*ff.ncols)

    logDebug('white ratio: ' + str(white_ratio))

    if white_ratio > max_white_ratio:

        log.debug(("Too many threshold passers! White ratio is {:.2f}, which is higher than the "\
            "max_white_ratio threshold: {:.2f}").format(white_ratio, max_white_ratio))

        return False


    return True




def getLines(img_handle, k1, j1, time_slide, time_window_size, max_lines, max_white_ratio, kht_lib_path, \
    mask=None, flat_struct=None, dark=None, debug=False):
    """ Get (rho, phi) pairs for each meteor present on the image using KHT.
        
    Arguments:
        img_handle: [FrameInterface instance] Object with common interface to various input formats.
        k1: [float] weight parameter for the standard deviation during thresholding
        j1: [float] absolute threshold above average during thresholding
        time_slide: [int] subdivision size of the time axis (256 will be divided into 256/time_slide parts)
        time_window_size: [int] size of the time window which will be slided over the time axis
        max_lines: [int] maximum number of lines to find by KHT
        max_white_ratio: [float] max ratio between write and all pixels after thresholding
        kht_lib_path: [string] path to the compiled KHT library
        mask: [MaskStruct] Mask structure.
        flat_struct: [FlatStruct]  Flat frame sturcture.
        dark: [ndarray] Dark frame.

    
    Return:
        [list] a list of all found lines
    """

    # Load the KHT library
    kht = ctypes.cdll.LoadLibrary(kht_lib_path)
    kht.kht_wrapper.argtypes = [npct.ndpointer(dtype=np.double, ndim=2),
                                npct.ndpointer(dtype=np.byte, ndim=1),
                                ctypes.c_size_t,
                                ctypes.c_size_t,
                                ctypes.c_size_t,
                                ctypes.c_size_t,
                                ctypes.c_double,
                                ctypes.c_double,
                                ctypes.c_double,
                                ctypes.c_double]
    kht.kht_wrapper.restype = ctypes.c_size_t

    line_results = []


    # If the input is a single FF file, threshold the image right away
    if img_handle.input_type == 'ff':

        # Threshold the FF
        img_thres = thresholdFF(img_handle.ff, k1, j1, mask=mask)

        # # Show thresholded image
        # show("thresholded ALL", img_thres)

        # Check if there are too many threshold passers, if so report that no lines were found
        if not checkWhiteRatio(img_thres, img_handle.ff, max_white_ratio):
            return line_results


    # Subdivide the image by time into overlapping parts (decreases noise when searching for meteors)
    for i in range(0, int(np.ceil(img_handle.total_frames/time_slide)) - 1):

        frame_min = i*time_slide
        frame_max = i*time_slide + time_window_size


        # If an FF file is used
        if img_handle.input_type == 'ff':
            
            # Select the time range of the thresholded image
            img = FFfile.selectFFFrames(img_thres, img_handle.ff, frame_min, frame_max)


        # If not, load a range of frames and threshold it
        else:

            # Load the frame chunk
            img_handle.loadChunk(first_frame=frame_min, read_nframes=(frame_max - frame_min + 1))

            # If making the synthetic FF has failed, skip it
            if not img_handle.ff.successful:
                logDebug('Skipped frame range due to failed synthetic FF generation: frames {:d} to {:d}'.format(\
                    frame_min, frame_max))
                continue

            # Print the time
            logDebug('Time:', img_handle.name())

            # Apply the mask, dark, flat
            img_handle = preprocessFF(img_handle, mask, flat_struct, dark)

            # Threshold the frame chunk
            img = thresholdFF(img_handle.ff, k1, j1, mask=mask)

            # Check if there are too many threshold passers, if so report that no lines were found
            if not checkWhiteRatio(img, img_handle.ff, max_white_ratio):
                continue


        if debug:
            ### Show maxpixel and thresholded image

            if img_handle.input_type == 'ff':
                maxpix_img = FFfile.selectFFFrames(img_handle.ff.maxpixel, img_handle.ff, frame_min, frame_max)

            else:
                maxpix_img = img_handle.ff.maxpixel


            # Auto levels on maxpixel
            min_lvl = np.percentile(img_handle.ff.maxpixel[2:], 1)
            max_lvl = np.percentile(img_handle.ff.maxpixel[2:], 99.0)

            # Adjust levels
            maxpixel_autolevel = Image.adjustLevels(maxpix_img, min_lvl, 1.0, max_lvl)

            show2(str(frame_min) + "-" + str(frame_max) + " threshold", np.concatenate((maxpixel_autolevel, \
                img.astype(maxpix_img.dtype)*(2**(maxpix_img.itemsize*8) - 1)), axis=1))

            ###


        # # Show maxpixel of the thresholded part
        # mask = np.zeros(shape=img.shape)
        # mask[np.where(img)] = 1
        # show('thresh max', ff.maxpixel*mask)

        ### Apply morphological operations to prepare the image for KHT

        # Morphological operations:
            # 1 - clean (Remove lonely pixels)
            # 2 - bridge (Connect close pixels)
            # 3 - close (Close surrounded pixels)
            # 4 - thin (Thin all lines to 1px width)
            # 1 - Remove lonely pixels
        img = morph.morphApply(img, [1, 2, 3, 4, 1])


        if debug:
            # Show morphed image
            show(str(frame_min) + "-" + str(frame_max) + " morph", img)


        # Get image shape
        w, h = img.shape[1], img.shape[0]

        # Convert the image to feed it into the KHT
        img_flatten = (img.flatten().astype(np.byte)*255).astype(np.byte)
        
        # Predefine the line output
        lines = np.empty((max_lines, 2), np.double)
        
        # Call the KHT line finding
        # Parameters: cluster_min_size (px), cluster_min_deviation, delta, kernel_min_height, n_sigmas
        length = kht.kht_wrapper(lines, img_flatten, w, h, max_lines, 9, 2, 0.1, 0.004, 1)
        
        # Cut the line array to the number of found lines
        lines = lines[:length]


        # Skip further operations if there are no lines
        frame_lines = []
        if lines.any():
            for rho, theta in lines:
                line_results.append([rho, theta, frame_min, frame_max])
                frame_lines.append([rho, theta, frame_min, frame_max])


        if debug:
            if frame_lines:
                plotLines(img_handle.ff, frame_lines)


    return line_results


def filterCentroids(centroids, centroid_max_deviation, max_distance):
    """ Check for linearity in centroid data and reject the points which are too far off. 

    Arguments:
        centroids: [list] a list of [frame, X, Y, level] centroid coordinates
        centroid_max_deviation: [float] max deviation from the fitted line, centroids above this get rejected
        max_distance: [float] max distance between 2 ends of centroid chains which connects them
    
    Return:
        centroids: [list] a filtered list of centroids (see input centroids for details)

    """

    def _pointDistance(x1, y1, x2, y2):
        """ Distance between 2 points in 2D Cartesian coordinates.
        """

        return np.sqrt((x2-x1)**2 + (y2-y1)**2)



    def _LSQfit(y, x):
        """ Least squares fit.
        """

        A = np.vstack([x, np.ones(len(x)).astype(np.float64)]).T
        m, c = np.linalg.lstsq(A, y, rcond=-1)[0]

        return m, c



    def _connectBrokenChains(chains, max_distance, last_count=0):
        """ Connect broken chains of centroids.
        """

        filtered_chains = []
        paired_indices = []
        for i, chain1 in enumerate(chains):

            # SKip if chain already connected
            if i in paired_indices:
                continue

            # Get last point of first chain
            x1 = chain1[-1][2]
            y1 = chain1[-1][3]

            found_pair = False
            for j, chain2 in enumerate(chains[i+1:]):

                # Correct for real position in the chain
                j = j + i + 1

                # Skip if chain already connected
                if j in paired_indices:
                    continue

                # Get first point of the second chain
                x2 = chain2[0][2]
                y2 = chain2[0][3]

                # Check if chains connect
                if _pointDistance(x1, y1, x2, y2) <= max_distance:

                    # Concatenate chains
                    filtered_chains.append(chain1 + chain2)

                    paired_indices.append(i)
                    paired_indices.append(j)

                    found_pair = True

                    break

            if not found_pair:
                filtered_chains.append(chain1)

        # Iteratively connect chains until all chains are connected
        if len(filtered_chains) != last_count:
            filtered_chains = _connectBrokenChains(chains, max_distance, len(filtered_chains))

        return filtered_chains


    def _filterFrameGaps(centroids, frame_diff_multiplier=3.5):
        """ Filter centroid chains with gaps in frames. All chains with a gap of more than 2 frames will be 
            stripped at the edges. 
        """

        # Filter edge frame outliers until a stable set is achieved

        prev_len = np.inf
        while len(centroids) < prev_len:

            prev_len = len(centroids)
            frame_array = centroids[:,0]

            # Compute median frame difference
            frame_diffs = frame_array[1:] - frame_array[:-1]
            frame_diff_med = np.median(frame_diffs)
            
            mask_array = np.ones_like(frame_array)

            # If frame differences in the first half are larger than 2x the median frame difference, cull them
            for i in range(len(frame_diffs)//2):
                if frame_diffs[i] > frame_diff_multiplier*frame_diff_med:
                    mask_array[i] = 0

            # If frame differences in the last half are larger than 2x the median frame difference, cull them
            for i in range(len(frame_diffs)//2, len(frame_diffs)):
                if frame_diffs[i] > frame_diff_multiplier*frame_diff_med:
                    mask_array[i + 1] = 0

            # Filter centroids by mask
            centroids = centroids[np.where(mask_array)]

        return centroids


    # Skip centroid correction if there are not enough centroids
    if len(centroids) < 3:
        return centroids

    centroids_array = np.array(centroids).astype(np.float64)

    # Filter centroids of frame gaps
    centroids_array = _filterFrameGaps(centroids_array)

    # Skip centroid correction if there are not enough centroids
    if len(centroids) < 3:
        return centroids

        

    # Separate by individual columns of the centroid array
    frame_array = centroids_array[:,0]
    x_array = centroids_array[:,2]
    y_array = centroids_array[:,3]

    # LSQ fit by both axes
    try:
        mX, cX = _LSQfit(x_array, frame_array)
        mY, cY = _LSQfit(y_array, frame_array)
    except Exception as e:
        log.debug('Fitting centroid X and Y progressions in time failed with message:\n' + repr(e))
        log.debug(repr(traceback.format_exception(*sys.exc_info())))
        log.debug('Filtering centroids failed at fitting X and Y progressions in time, skipping filtering...')
        logDebug('x_array:', x_array)
        logDebug('y_array:', y_array)
        return centroids

    filtered_centroids = []

    # Distances between points and fitted line
    point_deviations = _pointDistance(x_array, y_array, mX*frame_array + cX, mY*frame_array + cY)

    # Calculate median deviation
    mean_deviation = np.median(point_deviations)

    # Take points with satisfactory deviation
    good_centroid_indices = np.where(np.logical_not(point_deviations > mean_deviation*centroid_max_deviation + 1))
    filtered_centroids = centroids_array[good_centroid_indices].tolist()

    # Go through all points and separate by chains of centroids (divided by max distance)
    chains = []
    chain_index = 0
    for i in range(len(filtered_centroids)-1):

        if i == 0:
            # Initialize the first chain on the first point
            chains.append([filtered_centroids[i]])

        # Current point position
        x1 = filtered_centroids[i][2]
        y1 = filtered_centroids[i][3]

        # Next point position
        x2 = filtered_centroids[i+1][2]
        y2 = filtered_centroids[i+1][3]

        # Check if the current and the next point are sufficiently close
        if _pointDistance(x1, y1, x2, y2) <= max_distance:
            chains[chain_index].append(filtered_centroids[i+1])
        else:
            # Make new chain
            chains.append([filtered_centroids[i+1]])
            chain_index += 1
            

    # Connect broken chains
    filtered_chains = _connectBrokenChains(chains, max_distance)


    # Choose the chain with the greatest length
    chain_lengths = [len(chain) for chain in filtered_chains]
    max_chain_length = max(chain_lengths)

    best_chain = filtered_chains[chain_lengths.index(max_chain_length)]


    return best_chain


def checkAngularVelocity3D(detected_line, config, correct_binning=False):
    """ Check the angular velocity of the detection, and reject those too slow or too fast to be meteors. 
        The minimum ang. velocity is 0.5 deg/s, while maximum is 35 deg/s (Peter Gural, private comm.).
    
    Arguments:
        detected_line: [list] A list which contains the 3D coordinates of the detected line.
        config: [config object] configuration object (loaded from the .config file)

    Keyword arguments:
        correct_binning: [bool] Correct the centroids for binning.
    
    Return:
        ang_vel, ang_vel_status: [float, bool]
            - ang_vel - angular velovity in deg/s
            - ang_vel_status - True if the velocity is in the meteor ang. velocity range, False otherwise

    """

    # Get coordinates of 2 points that describe the line
    x1, y1, z1 = detected_line[0]
    x2, y2, z2 = detected_line[1]

    # Correct the points for binning
    if correct_binning and (config.detection_binning_factor > 1):
        x1 *= config.detection_binning_factor
        y1 *= config.detection_binning_factor
        x2 *= config.detection_binning_factor
        y2 *= config.detection_binning_factor


    # Compute the average angular velocity in px per frame
    ang_vel = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)/(z2 - z1)

    # Convert to px/sec
    ang_vel = ang_vel*config.fps

    # Convert to deg/sec
    scale = (config.fov_h/float(config.height) + config.fov_w/float(config.width))/2.0
    ang_vel = ang_vel*scale

    # Check if the meteor is in the possible angular velocity range (deg/s)
    if (ang_vel >= config.ang_vel_min and ang_vel <= config.ang_vel_max):
        return ang_vel, True

    else:
        return ang_vel, False



def checkAngularVelocity(centroids, config):
    """ Check the angular velocity of the detected centroids, and reject those too slow or too fast to be 
        meteors. The minimum ang. velocity is 0.5 deg/s, while maximum is 35 deg/s (Peter Gural, private 
        comm.).
    
    Arguments:
        centroids: [ndarray] meteor centroids from the detector
        config: [config object] configuration object (loaded from the .config file)
    
    Return:
        ang_vel, ang_vel_status: [float, bool]
            - ang_vel - angular velovity in deg/s
            - ang_vel_status - True if the velocity is in the meteor ang. velocity range, False otherwise

    """

    # Calculate the angular velocity in px/frame
    first_centroid = centroids[0]
    last_centroid = centroids[-1]
    
    frame1, _, x1, y1, _ = first_centroid
    frame2, _, x2, y2, _ = last_centroid

    ang_vel = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)/float(frame2 - frame1 + 1)

    # Convert to px/sec
    ang_vel = ang_vel*config.fps

    # Convert to deg/sec
    scale = (config.fov_h/float(config.height) + config.fov_w/float(config.width))/2.0
    ang_vel = ang_vel*scale

    # Check if the meteor is in the possible angular velocity range (deg/s)
    if (ang_vel >= config.ang_vel_min and ang_vel <= config.ang_vel_max):
        return ang_vel, True

    else:
        return ang_vel, False




def show(name, img):
    """ COnvert the given image to uint8 and show it. """

    cv2.imshow(name, img.astype(np.uint8)*255)
    cv2.moveWindow(name, 0, 0)
    cv2.waitKey(0)
    cv2.destroyWindow(name)



def show2(name, img):
    """ Show the given image. """

    cv2.imshow(name, img)
    cv2.moveWindow(name, 0, 0)
    cv2.waitKey(0)
    cv2.destroyWindow(name)



def showAutoLevels(img):

    # Auto levels
    min_lvl = np.percentile(img[2:], 1)
    max_lvl = np.percentile(img[2:], 99.0)

    # Adjust levels
    img_autolevel = Image.adjustLevels(img, min_lvl, 1.0, max_lvl)

    plt.imshow(img_autolevel, cmap='gray')
    plt.show()

    ###




def plotLines(ff, line_list):
    """ Plot lines on the image.
    """

    img = np.copy(ff.maxpixel)


    # Auto adjust levels
    min_lvl = np.percentile(img[2:], 1)
    max_lvl = np.percentile(img[2:], 99.0)

    # Adjust levels
    img = Image.adjustLevels(img, min_lvl, 1.0, max_lvl)

    hh = img.shape[0]/2.0
    hw = img.shape[1]/2.0

    # Compute the maximum level value
    max_lvl = 2**(img.itemsize*8) - 1

    for rho, theta, frame_min, frame_max in line_list:
        theta = np.deg2rad(theta)
        
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho

        x1 = int(x0 + 1000*(-b) + hw)
        y1 = int(y0 + 1000*(a) + hh)
        x2 = int(x0 - 1000*(-b) + hw)
        y2 = int(y0 - 1000*(a) + hh)
        
        cv2.line(img, (x1, y1), (x2, y2), (max_lvl, 0, max_lvl), 1)
        
    show2("KHT", img)



def show3DCloud(ff, xs, ys, zs, detected_line=None, stripe_points=None, config=None):
    """ Shows 3D point cloud of stripe points.
    """

    if detected_line is None:
        detected_line = []

    logDebug('points: ', len(xs))

    # Plot points in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs, ys, zs)

    if detected_line and len(stripe_points):

        xs = [detected_line[0][1], detected_line[1][1]]
        ys = [detected_line[0][0], detected_line[1][0]]
        zs = [detected_line[0][2], detected_line[1][2]]
        ax.plot(ys, xs, zs, c = 'r')

        x1, x2 = ys
        y1, y2 = xs
        z1, z2 = zs

        detected_points = getAllPoints(stripe_points, x1, y1, z1, x2, y2, z2, config, 
            fireball_detection=False)

        if detected_points.any():

            detected_points = np.array(detected_points)

            xs = detected_points[:,0]
            ys = detected_points[:,1]
            zs = detected_points[:,2]

            ax.scatter(xs, ys, zs, c = 'r', s = 40)

    # Set limits
    plt.xlim((0, ff.ncols))
    plt.ylim((0, ff.nrows))
    ax.set_zlim((0, 255))

    plt.show()

    plt.clf()
    plt.close()



def preprocessFF(img_handle, mask, flat_struct, dark):
    """ Apply the mask, dark and flat to FF file in img_handle. """


    if not img_handle.ff.calibrated:

        # Apply the dark frame to maxpixel and avepixel    
        if dark is not None:
            img_handle.ff.maxpixel = Image.applyDark(img_handle.ff.maxpixel, dark)
            img_handle.ff.avepixel = Image.applyDark(img_handle.ff.avepixel, dark)


        # Apply the flat to maxpixel and avepixel
        if flat_struct is not None:

            img_handle.ff.maxpixel = Image.applyFlat(img_handle.ff.maxpixel, flat_struct)
            img_handle.ff.avepixel = Image.applyFlat(img_handle.ff.avepixel, flat_struct)


        # Mask the FF file
        img_handle.ff = MaskImage.applyMask(img_handle.ff, mask, ff_flag=True)

        # Set the FF calibration status to True
        img_handle.ff.calibrated = True


    return img_handle



def thresholdAndCorrectGammaFF(img_handle, config, mask):
    """ Prepare the FF for centroid extraction by performing gamma correction. """

    # Threshold the FF
    img_thres = thresholdFF(img_handle.ff, config.k1_det, config.j1_det)


    # Gamma correct image files
    maxpixel_gamma_corr = Image.gammaCorrection(img_handle.ff.maxpixel, config.gamma)
    avepixel_gamma_corr = Image.gammaCorrection(img_handle.ff.avepixel, config.gamma)
    stdpixel_gamma_corr = Image.gammaCorrection(img_handle.ff.stdpixel, config.gamma)

    # Make sure there are no zeros in standard deviation
    stdpixel_gamma_corr[stdpixel_gamma_corr == 0] = 1

    # Calculate weights for centroiding (apply gamma correction on both images)
    max_avg_corrected = maxpixel_gamma_corr - avepixel_gamma_corr
    flattened_weights = (max_avg_corrected).astype(np.float32)/stdpixel_gamma_corr


    # At the end, a check that the detection has a surface brightness above the background will be performed.
    # The assumption here is that the peak of the meteor should have the intensity which is at least
    # that of a patch of 4x4 pixels that are of the mean background brightness
    min_patch_intensity = 4*4*(np.mean(maxpixel_gamma_corr - avepixel_gamma_corr) \
        + config.k1_det*np.mean(stdpixel_gamma_corr) + config.j1)

    # Apply a special minimum path intensity multiplier
    min_patch_intensity *= config.min_patch_intensity_multiplier

    # Correct the minimum patch intensity if the image was binned with the 'avg' method
    if (img_handle != 'ff') and (config.detection_binning_method == 'avg'):
        min_patch_intensity *= config.detection_binning_factor**2


    return img_thres, max_avg_corrected, flattened_weights, min_patch_intensity




def detectMeteors(img_handle, config, flat_struct=None, dark=None, mask=None, asgard=False, debug=False):
    """ Detect meteors on the given image. Here are the steps in the detection:
            - input image (FF bin format file) is thresholded (converted to black and white)
            - several morphological operations are applied to clean the image
            - image is then broken into several image "windows" (these "windows" are reconstructed from the input FF file, given
              an input frame range (e.g. 64-128) which helps reduce the noise further)
            - on each "window" the Kernel-based Hough transform is performed to find any lines on the image
            - similar lines are joined
            - stripe around the lines is extracted
            - 3D line finding (third dimension is time) is applied to check if the line propagates in time
            - centroiding is performed, which calculates the position and intensity of meteor on each frame
    
    Arguments:
        img_handle: [FrameInterface instance] Object which has a common interface to various input files.
        config: [config object] configuration object (loaded from the .config file)

    Keyword arguments:
        flat_struct: [Flat struct] Structure containing the flat field. None by default.
        dark: [ndarray] Dark frame. None by default.
        mask: [ndarray] Mask image. None by default.
        asgard: [bool] If True, the vid file sequence number will be added in with the frame. False by 
            default, in which case only the frame number will be in the centroids.
        debug: [bool] If True, graphs for testing the detection settings will be shown. False by default.
    
    Return:
        meteor_detections: [list] a list of detected meteors, with these elements:
            - rho: [float] meteor line distance from image center (polar coordinates, in pixels)
            - theta: [float] meteor line angle from image center (polar coordinates, in degrees)
            - centroids: [list] [frame, X, Y, level] list of meteor points
    """


    t1 = time()
    t_all = time()


    # Bin the mask, dark and flat, only when not running on FF files
    if (img_handle.input_type != 'ff') and (config.detection_binning_factor > 1):

        # Bin the calibration images
        mask, dark, flat_struct = binImageCalibration(config, mask, dark, flat_struct)


    # Do all image processing on single FF file, if given
    # Otherwise, the image processing will be done on every frame chunk that is extracted
    if img_handle.input_type == 'ff':

        # If the FF file could not be loaded, skip it
        if img_handle.ff is None:
            logDebug("FF file cound not be loaded, skipping it...")
            return []

        # Apply mask and flat to FF
        img_handle = preprocessFF(img_handle, mask, flat_struct, dark)


    # Get lines on the image
    line_list = getLines(img_handle, config.k1_det, config.j1_det, config.time_slide, config.time_window_size, 
        config.max_lines_det, config.max_white_ratio, config.kht_lib_path, mask=mask, \
        flat_struct=flat_struct, dark=dark, debug=debug)

    # logDebug('List of lines:', line_list)


    # Init meteor list
    meteor_detections = []

    # Only if there are some lines in the image
    if len(line_list):

        # Join similar lines
        line_list = mergeLines(line_list, config.line_min_dist, img_handle.ff.ncols, img_handle.ff.nrows)

        logDebug('Time for finding lines:', time() - t1)

        logDebug('Number of KHT lines: ', len(line_list))

        # # Plot lines
        # plotLines(img_handle.ff, line_list)


        filtered_lines = []

        # Analyze stripes of each line
        # This step makes sure that there is a linear propagation of the detections in time
        for line in line_list:

            rho, theta, frame_min, frame_max = line

            logDebug('\n--------------------------------')
            logDebug('    rho,  theta, frame_min, frame_max')
            logDebug("{:7.2f}, {:6.2f}, {:9d}, {:9d}".format(rho, theta, frame_min, frame_max))


            # If FF files are not used as input, reconstruct it
            if img_handle.input_type != 'ff':

                # Compute the FF for this chunk
                img_handle.loadChunk(first_frame=frame_min, read_nframes=(frame_max - frame_min + 1))

                # Apply mask and flat to FF
                img_handle = preprocessFF(img_handle, mask, flat_struct, dark)

                # ### PLOT CHUNK
                # img = img_handle.ff.maxpixel - img_handle.ff.avepixel

                # # Auto adjust levels
                # min_lvl = np.percentile(img[2:], 1)
                # max_lvl = np.percentile(img[2:], 99.0)

                # # Adjust levels
                # img = Image.adjustLevels(img, min_lvl, 1.0, max_lvl)

                # # Show the image chunk, average subtracted
                # plt.imshow(img, cmap='gray')
                # plt.show()
                # ### ###

                logDebug('Checking temporal propagation at time:', img_handle.name())
                

            # Extract (x, y, frame) of thresholded frames, i.e. pixel and frame locations of threshold passers
            xs, ys, zs = getThresholdedStripe3DPoints(config, img_handle, frame_min, frame_max, rho, theta, \
                mask, flat_struct, dark, debug=False)

            # Limit the number of points to search if too large
            if len(zs) > config.max_points_det:

                # Extract weights of each point
                maxpix_elements = img_handle.ff.maxpixel[ys,xs].astype(np.float64)
                weights = maxpix_elements/np.sum(maxpix_elements)

                # Random sample the point, sampling is weighted by pixel intensity
                indices = np.random.choice(len(zs), config.max_points_det, replace=False, p=weights)
                ys = ys[indices]
                xs = xs[indices]
                zs = zs[indices]

            # Make an array to feed into the grouping algorithm
            stripe_points = np.vstack((xs, ys, zs))
            stripe_points = np.swapaxes(stripe_points, 0, 1)
            
            # Sort stripe points by frame
            stripe_points = stripe_points[stripe_points[:,2].argsort()]

            t1 = time()

            logDebug('finding lines...')

            # Find a single line in the point cloud
            detected_line = find3DLines(stripe_points, time(), config, fireball_detection=False)

            logDebug('time for GROUPING: {:.3f}'.format(time() - t1))

            # Extract the first and only line if any
            if detected_line:
                detected_line = detected_line[0]

                # logDebug(detected_line)
                

                # Check the detection if it has the proper angular velocity (correct for binning if not 
                #   using FF files as input)
                ang_vel, ang_vel_status = checkAngularVelocity3D(detected_line, config, 
                    correct_binning=(img_handle.input_type != 'ff'))

                if not ang_vel_status:
                    logDebug(detected_line)
                    logDebug('Rejected at initial stage due to the angular velocity: {:.2f} deg/s'.format(ang_vel))
                    continue

                # # Show 3D cloud
                # show3DCloud(img_handle.ff, xs, ys, zs, detected_line, stripe_points, config)

                # Add the line to the results list
                filtered_lines.append(detected_line)

            else:
                logDebug('No temporal propagation found!')


        # Merge similar lines in 3D
        filtered_lines = merge3DLines(filtered_lines, config.vect_angle_thresh)

        # logDebug('after filtering:')
        # logDebug(filtered_lines)


        # If the input is a single FF file, threshold the image right away and do gamma correction
        if img_handle.input_type == 'ff':

            img_thres, max_avg_corrected, flattened_weights, \
                min_patch_intensity = thresholdAndCorrectGammaFF(img_handle, config, mask)


        # Go through all detected and filtered lines and compute centroids
        for detected_line in filtered_lines:

            # Get frame range
            frame_min = detected_line[4]
            frame_max = detected_line[5]

            # Check if the line covers a minimum frame range
            if (abs(frame_max - frame_min) + 1 < config.line_minimum_frame_range_det):
                continue

            # Extend the frame range for several frames, just to be sure to catch all parts of the meteor
            frame_min -= config.frame_extension
            frame_max += config.frame_extension

            # Cap values
            frame_min = max(frame_min, 0)
            frame_max = min(frame_max, img_handle.total_frames - 1)

            logDebug(detected_line)

            # Get coordinates of 2 points that describe the line
            x1, y1, z1 = detected_line[0]
            x2, y2, z2 = detected_line[1]

            # Convert Cartesian line coordinates to polar
            rho, theta = getPolarLine(x1, y1, x2, y2, img_handle.ff.nrows, img_handle.ff.ncols)

            # Skip the line if rho could not be computed
            if np.isnan(rho):
                continue

            # Convert Cartesian line coordinate to CAMS compatible polar coordinates (flipped Y axis)
            rho_cams, theta_cams = getPolarLine(x1, img_handle.ff.nrows - y1, x2, img_handle.ff.nrows - y2, \
                img_handle.ff.nrows, img_handle.ff.ncols)


            # logDebug('converted rho, theta')
            # logDebug(rho, theta)

            # If other input types are given, load the frames and preprocess them
            if img_handle.input_type != 'ff':
                
                # Compute the FF for this chunk
                img_handle.loadChunk(first_frame=frame_min, read_nframes=(frame_max - frame_min + 1))

                # Preprocess image for this chunk
                img_handle = preprocessFF(img_handle, mask, flat_struct, dark)

                img_thres, max_avg_corrected, flattened_weights, \
                    min_patch_intensity = thresholdAndCorrectGammaFF(img_handle, config, mask)

                logDebug('Centroiding at time:', img_handle.name())
                


            # Extract (x, y, frame) of thresholded frames, i.e. pixel and frame locations of threshold passers
            xs, ys, zs = getThresholdedStripe3DPoints(config, img_handle, frame_min, frame_max, rho, theta, \
                mask, flat_struct, dark, stripe_width_factor=1.5, centroiding=True, \
                point1=detected_line[0], point2=detected_line[1], debug=False)


            # Make an array to feed into the centroiding algorithm
            stripe_points = np.vstack((xs, ys, zs))
            stripe_points = np.swapaxes(stripe_points, 0, 1)
            
            # Sort stripe points by frame
            stripe_points = stripe_points[stripe_points[:,2].argsort()]

            # # Show 3D cloud
            # show3DCloud(img_handle.ff, xs, ys, zs, detected_line, stripe_points, config)

            # Get points of the given line
            line_points = getAllPoints(stripe_points, x1, y1, z1, x2, y2, z2, config, 
                fireball_detection=False)

            # Skip if no points were returned
            if not line_points.any():
                logDebug('No line found in line refinement...')
                continue

            # Skip if the points cover too small a frame range
            frame_range = abs(np.max(line_points[:,2]) - np.min(line_points[:,2])) + 1
            if frame_range < config.line_minimum_frame_range_det:
                logDebug('Too small frame range! {:d} < {:d}'.format(frame_range, \
                    config.line_minimum_frame_range_det))
                continue


            # Calculate centroids
            centroids = []
            for i in range(frame_min, frame_max + 1):
                
                # Select pixel indicies belonging to a given frame
                frame_pixels_inds = np.where(line_points[:,2] == i)
                
                # Get pixel positions in a given frame (pixels belonging to a found line)
                frame_pixels = line_points[frame_pixels_inds].astype(np.int64)

                # Get pixel positions in a given frame (pixels belonging to the whole stripe)
                frame_pixels_stripe = stripe_points[np.where(stripe_points[:,2] == i)].astype(np.int64)

                # Skip if there are no pixels in the frame
                if not len(frame_pixels):
                    continue

                # Calculate centroids by half-frame
                for half_frame in range(2):

                    # Apply deinterlacing if it is present in the video
                    if config.deinterlace_order >= 0:

                        # Deinterlace by fields (line lixels)
                        half_frame_pixels = frame_pixels[frame_pixels[:,1]%2 == (config.deinterlace_order 
                            + half_frame)%2]

                        # Deinterlace by fields (stripe pixels)
                        half_frame_pixels_stripe = frame_pixels_stripe[frame_pixels_stripe[:,1] % 2 == (config.deinterlace_order 
                            + half_frame)%2]


                        # Skip if there are no pixels in the half-frame
                        if not len(half_frame_pixels):
                            continue

                        # Calculate half-frame value
                        frame_no = i + half_frame*0.5


                    # No deinterlacing
                    else:

                        # Skip the second half frame
                        if half_frame == 1:
                            continue

                        half_frame_pixels = frame_pixels
                        half_frame_pixels_stripe = frame_pixels_stripe
                        frame_no = i


                    # Get maxpixel-avepixel values of given pixel indices (this will be used as weights)
                    max_weights = flattened_weights[half_frame_pixels[:,1], half_frame_pixels[:,0]]

                    # Calculate weighted centroids
                    x_weighted = half_frame_pixels[:,0]*np.transpose(max_weights)
                    x_centroid = np.sum(x_weighted.astype(np.float64))/float(np.sum(max_weights))

                    y_weighted = half_frame_pixels[:,1]*np.transpose(max_weights)
                    y_centroid = np.sum(y_weighted.astype(np.float64))/float(np.sum(max_weights))


                    # Correct the rolling shutter effect
                    if config.deinterlace_order == -1:

                        # Compute the corrected frame time
                        frame_no = RollingShutterCorrection.correctRollingShutterTemporal(frame_no, \
                            y_centroid, img_handle.ff.maxpixel.shape[0])


                    # Get current frame if video or images are used as input
                    if img_handle.input_type != 'ff':

                        ### Extract intensity from frame ###

                        # Load the frame
                        img_handle.setFrame(int(frame_no))
                        fr_img = img_handle.loadFrame()

                        # Get the frame sequence number (frame number since the beginning of the recording)
                        seq_num = img_handle.getSequenceNumber()

                        # Apply dark frame
                        if dark is not None:
                            fr_img = Image.applyDark(fr_img, dark)


                        # Apply the flat to frame
                        if flat_struct is not None:
                            fr_img = Image.applyFlat(fr_img, flat_struct)


                        # Mask the image
                        fr_img = MaskImage.applyMask(fr_img, mask)


                        # Apply gamma correction
                        fr_img = Image.gammaCorrection(fr_img, config.gamma)

                        # Subtract average
                        max_avg_corrected = Image.applyDark(fr_img, img_handle.ff.avepixel)

                    else:

                        # If the FF file is used, set the sequence number to the current frame number
                        seq_num = i


                    # Calculate intensity as the sum of threshold passer pixels on the stripe
                    intensity_values = max_avg_corrected[half_frame_pixels_stripe[:,1], 
                            half_frame_pixels_stripe[:,0]]

                    intensity = int(np.sum(intensity_values))


                    # Rescale the centroid position and intensity back to the pre-binned size
                    if (img_handle.input_type != 'ff') and (config.detection_binning_factor > 1):
                        x_centroid *= config.detection_binning_factor
                        y_centroid *= config.detection_binning_factor

                        # Rescale the intensity only if the binning method was 'average'
                        if config.detection_binning_method == 'avg':
                            intensity *= config.detection_binning_factor**2

                    logDebug("centroid: fr {:>12.3f}, x {:>7.2f}, y {:>7.2f}, intens {:d}".format(frame_no, \
                        x_centroid, y_centroid, intensity))

                    # Add computed centroid to the centroid list
                    centroids.append([frame_no, seq_num, x_centroid, y_centroid, intensity])


            # Filter centroids
            centroids = filterCentroids(centroids, config.centroids_max_deviation, 
                config.centroids_max_distance)

            # Convert to numpy array for easy slicing
            centroids = np.array(centroids)

            # Reject the solution if there are too few centroids
            if len(centroids) < config.line_minimum_frame_range_det:
                logDebug('Rejected due to too few frames!')
                continue

            # Check that the detection has a surface brightness above the background
            # The assumption here is that the peak of the meteor should have the intensity which is at least
            # that of a patch of 4x4 pixels that are of the mean background brightness
            if np.max(centroids[:, 4]) < min_patch_intensity:
                logDebug('Rejected due to too low max patch intensity:', np.max(centroids[:, 4]), ' < ', \
                    min_patch_intensity)
                continue


            # Check the detection if it has the proper angular velocity
            ang_vel, ang_vel_status = checkAngularVelocity(centroids, config)
            if not ang_vel_status:
                logDebug('Rejected due to the angular velocity: {:.2f} deg/s'.format(ang_vel))
                continue


            # If the FTPdetectinfo format is requested, exclude the sequence number column from centroids
            if not asgard:
                centroids = np.delete(centroids, 1, axis=1)


            # Append the result to the meteor detections
            meteor_detections.append([rho_cams, theta_cams, centroids])


            logDebug('Time for processing:', time() - t_all)


            
            # # Plot centroids to image
            # fig, (ax1, ax2) = plt.subplots(nrows=2)
            
            # ax1.imshow(ff.maxpixel - ff.avepixel, cmap='gray')
            # ax1.scatter(centroids[:,1], centroids[:,2], s=5, c='r', edgecolors='none')

            # # Plot lightcurve
            # ax2.plot(centroids[:,0], centroids[:,3])

            # # # Plot relative angular velocity
            # # ang_vels = []
            # # fr_prev, x_prev, y_prev, _ = centroids[0]
            # # for fr, x, y, _ in centroids[1:]:
            # #     dx = x - x_prev
            # #     dy = y - y_prev
            # #     dfr = fr - fr_prev

            # #     ddist = np.sqrt(dx**2 + dy**2)
            # #     dt = dfr/config.fps

            # #     ang_vels.append(ddist/dt)

            # #     x_prev = x
            # #     y_prev = y
            # #     fr_prev = fr

            # # ax2.plot(ang_vels)

            # plt.show()


    
    return meteor_detections



if __name__ == "__main__":



    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Run RMS meteor detection on given data.")

    arg_parser.add_argument('dir_path', nargs='+', metavar='DIR_PATH', type=str, \
        help='Path to the folder with FF or image files, or path to a video file. If images or videos are given, their names must be in the format: YYYYMMDD_hhmmss.uuuuuu, or the beginning time has to be given.')

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one.")

    arg_parser.add_argument('-o', '--outdir', nargs=1, metavar='OUTDIR', type=str, \
        help="Path to the output directory where the detections will be saved.")

    arg_parser.add_argument('-t', '--timebeg', nargs=1, metavar='TIME', type=str, \
        help="The beginning time of the video file in the YYYYMMDD_hhmmss.uuuuuu format. Not needed for FF or vid files.")

    arg_parser.add_argument('-f', '--fps', metavar='FPS', type=float, \
        help="Frames per second when images are used. If not given, it will be read from the config file. Not needed for FF or vid files.")

    arg_parser.add_argument('-g', '--gamma', metavar='CAMERA_GAMMA', type=float, \
        help="Camera gamma value. Science grade cameras have 1.0, consumer grade cameras have 0.45. Adjusting this is essential for good photometry, and doing star photometry through SkyFit can reveal the real camera gamma.")

    arg_parser.add_argument('-a', '--asgard', nargs=1, metavar='ASGARD_PLATE', type=str, \
        help="""Write output as ASGARD event files, not CAMS FTPdetectinfo. The path to an AST plate is taken as the argument.""")

    arg_parser.add_argument('-p', '--photoff', metavar='ASGARD_PHOTOMETRIC_OFFSET', type=float, \
        help="""Photometric offset used when the ASGARD AST plate is given. Mandatory argument if --asgard is used.""")


    arg_parser.add_argument('-d', '--debug', action="store_true", \
        help="""Show debug info on the screen. """)

    arg_parser.add_argument('-i', '--debugplots', action="store_true", \
        help="""Show graphs (calibrated image, thresholded image, detected lines) which can help adjust the detection settings. """)

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, cml_args.dir_path)


    # If camera gamma was given, change the value in config
    if cml_args.gamma is not None:
        config.gamma = cml_args.gamma


    # Parse the beginning time into a datetime object
    if cml_args.timebeg is not None:

        beginning_time = datetime.datetime.strptime(cml_args.timebeg[0], "%Y%m%d_%H%M%S.%f")

    else:
        beginning_time = None



    # Check if a correct AST file was given for ASGARD data and if the photometric offset is given
    if cml_args.asgard is not None:

        # Extract the path to the platepar file
        ast_path = cml_args.asgard[0]

        # Quit if the AST file does not exist
        if not os.path.isfile(ast_path):
            print()
            print('The AST file could not be loaded: {:s}'.format(ast_path))
            print('Exiting...')
            sys.exit()

        # Load the AST plate
        ast = loadAST(*os.path.split(ast_path))

        

        # Check if the photometric offset is given
        photom_offset = cml_args.photoff

        if photom_offset is None:
            print()
            print('The photometric offset has to be given with argument --photoff if the AST plate is given with the --asgard argument.')
            print('Exiting...')
            sys.exit()



    # Measure time taken for the detection
    time_whole = time()


    ### Init the logger

    from RMS.Logger import initLogging
    initLogging(config, 'detection_')

    log = logging.getLogger("logger")

    ######

    # Check if a list of files is given
    if len(cml_args.dir_path) > 1:

        img_handle_list = []

        # If it is, load files individually as separate image handles
        for file_path in cml_args.dir_path:

            file_path = file_path.replace('"', '')

            img_handle = detectInputType(file_path, config, beginning_time=beginning_time, fps=cml_args.fps, \
                detection=True)

            img_handle_list.append(img_handle)


        # Set the first image handle as the main one
        img_handle_main = img_handle_list[0]


        # If folders with images are gicen, dump the detections in the parent directory
        if img_handle_main.input_type == 'images':
            
            # Set the main directory to be the parent directory of all files
            main_dir = os.path.abspath(os.path.join(img_handle_main.dir_path, os.pardir))

        # For all else, dump detections into the directory with the data file
        else:
            main_dir = img_handle_main.dir_path



    else:
        dir_path_input = cml_args.dir_path[0].replace('"', '')

        # Detect the input file format
        img_handle_main = detectInputType(dir_path_input, config, beginning_time=beginning_time, \
            fps=cml_args.fps, detection=True)

        img_handle_list = []

        # If the input format are FF files, break them down into single FF files
        if img_handle_main.input_type == 'ff':
            if not img_handle_main.single_ff:

                # Go through all FF files and add them as individual files
                for file_name in img_handle_main.ff_list:

                    img_handle = detectInputType(os.path.join(dir_path_input, file_name), config, \
                        skip_ff_dir=True, detection=True)

                    img_handle_list.append(img_handle)


        # Otherwise, just add the main image handle to the list
        if len(img_handle_list) == 0:
            img_handle_list.append(img_handle_main)


        # Set the main directory to the the given input directory
        main_dir = img_handle_main.dir_path



    # Check if the output directory was given. If not, the detection will be saved in the input directory
    if cml_args.outdir:
        out_dir = cml_args.outdir[0]

        # Make sure the output directory exists
        mkdirP(out_dir)

    else:
        out_dir = main_dir


    # If debug is on, enable debug logging
    if cml_args.debug:
        VERBOSE_DEBUG = True


    # Load mask, dark, flat
    mask, dark, flat_struct = loadImageCalibration(main_dir, config, dtype=img_handle_main.ff.dtype, \
        byteswap=img_handle_main.byteswap)


    # Init results list
    results_list = []

    # Open a file for results
    results_path = out_dir
    results_name = config.stationID + "_" + img_handle_main.beginning_datetime.strftime("%Y%m%d_%H%M%S_%f")

    if cml_args.debug:
        results_file = open(os.path.join(results_path, results_name + '_results.txt'), 'w')

    total_meteors = 0

    # Run meteor search on every file
    for img_handle in img_handle_list:

        logDebug('--------------------------------------------')
        logDebug(img_handle.name())

        # Run the meteor detection algorithm
        meteor_detections = detectMeteors(img_handle, config, flat_struct=flat_struct, dark=dark, mask=mask, \
            debug=cml_args.debugplots, asgard=(cml_args.asgard is not None))

        # Supress numpy scientific notation printing
        np.set_printoptions(suppress=True)

        meteor_No = 1
        for meteor in meteor_detections:

            rho, theta, centroids = meteor

            first_pick_time = img_handle.currentFrameTime(frame_no=int(centroids[0][0]), dt_obj=True)

            if cml_args.debug:
                # Print detection to file
                results_file.write('-------------------------------------------------------\n')
                results_file.write(str(first_pick_time) + '\n')
                results_file.write(str(rho) + ',' + str(theta) + '\n')

            # Write the time in results file instead of the frame
            res_centroids = centroids.tolist()
            for entry in res_centroids:
                entry[0] = (img_handle.currentFrameTime(frame_no=int(entry[0]), \
                    dt_obj=True) - first_pick_time).total_seconds()

            if cml_args.debug:
                results_file.write(str(np.array(res_centroids)) + '\n')

            # Append to the results list
            results_list.append([img_handle.name(beginning=True), meteor_No, rho, theta, centroids])
            meteor_No += 1

            total_meteors += 1



        # Write output as ASGARD event file
        if cml_args.asgard is not None:

            # count of multiple simultaneous detections (within the same second)
            multi_event = 0
            prev_fn_time = 0

            # Go through all centroids
            for meteor in meteor_detections:

                rho, theta, centroids = meteor


                # Extract centroid columns
                frame_array, seq_array, x_array, y_array, intensity_array = centroids.T


                # Compute frame time for every centroid
                time_array = []
                for entry in centroids:

                    # Compute the datetime for every frame
                    frame_time = img_handle.currentFrameTime(frame_no=int(entry[0]))
                    time_array.append(frame_time)


                ### Compute alt/az and magnitudes
                
                # Compute ra/dec
                jd_array, ra_array, dec_array, mag_array = xyToRaDecAST(time_array, x_array, y_array, \
                    intensity_array, ast, photom_offset)

                # Compute alt/az
                azim_array = []
                alt_array = []
                for jd, ra, dec in zip(jd_array, ra_array, dec_array):

                    azim, alt = raDec2AltAz(ra, dec, jd, np.degrees(ast.lat), np.degrees(ast.lon))

                    azim_array.append(azim)
                    alt_array.append(alt)

                ###

                # Construct an input array for ASGARD event file function
                ev_array = np.array([frame_array, seq_array, jd_array, intensity_array, x_array, y_array, \
                    azim_array, alt_array, mag_array]).T


                ### Construct the file name for the event file

                # Find the time of the peak
                jd_peak = jd_array[mag_array.argmin()]


                # Construct the time part of the file name
                fn_time = jd2Date(jd_peak, dt_obj=True)
                fn_time = fn_time.strftime('%Y%m%d_%H%M%S')

                # If the previous file name was the same, increment the multi detect count
                if fn_time == prev_fn_time:
                    multi_event += 1
                else:
                    # Reset when at least one second ticks over
                    multi_event = 0

                prev_fn_time = fn_time

                # Construct a file name for the event
                # multi_event is used to assign a letter to multiple detections
                # first 0 = 'A', 1 = 'B', 2 = 'C', etc.
                # ex. ev_20190604_010203A_01A.txt
                file_name = 'ev_' + fn_time + chr(ord('A') + multi_event) + "_" + config.stationID + '.txt'

                ###


                # Write the ev file - use .vid file vidinfo for metadata if available
                if hasattr(img_handle, "vidinfo"):
                    writeEv(results_path, file_name, ev_array, ast, multi_event, \
                        ast_input=True, vidinfo=img_handle.vidinfo)
                else:
                    writeEv(results_path, file_name, ev_array, ast, multi_event, ast_input=True)


    if cml_args.debug:
        results_file.close()


    # Write output as CAMS FTPdetectinfo files
    if cml_args.asgard is None:

        ftpdetectinfo_name = 'FTPdetectinfo_' + results_name + '.txt'

        # Write FTPdetectinfo file
        FTPdetectinfo.writeFTPdetectinfo(results_list, results_path, ftpdetectinfo_name, results_path, 
            config.stationID, config.fps)
                


    print('Time for the whole directory:', time() - time_whole)
    print('Detected meteors:', total_meteors)