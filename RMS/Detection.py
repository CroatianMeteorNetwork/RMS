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

import logging
from time import time
import sys, os
import ctypes

import numpy as np
import numpy.ctypeslib as npct
import cv2

# Plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

# RMS imports
from RMS.DetectionTools import getThresholdedStripe3DPoints
from RMS.Formats import FFfile
from RMS.Formats import FTPdetectinfo
from RMS.Formats.FrameInterface import detectInputType
import RMS.ConfigReader as cr
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

# If True, all detection details will be logged
VERBOSE_DEBUG = True


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
        rho1, theta1 = line1[0:2]

        

        for j, line2 in enumerate(line_list[i+1:]):

            # Get real j index in respect to line_list
            j = i + j + 1

            # Skip if the line was paired
            if j in paired_indices:
                continue

            # Get polar coordinates of line
            rho2, theta2 = line2[0:2]

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
    mask, flat_struct):
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
        img_thres = thresholdFF(img_handle.ff, k1, j1)

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

            # Mask the FF file
            img_handle.ff = MaskImage.applyMask(img_handle.ff, mask, ff_flag=True)

            # Apply the flat to maxpixel and avepixel
            if flat_struct is not None:

                img_handle.ff.maxpixel = Image.applyFlat(img_handle.ff.maxpixel, flat_struct)
                img_handle.ff.avepixel = Image.applyFlat(img_handle.ff.avepixel, flat_struct)

            # Threshold the frame chunk
            img = thresholdFF(img_handle.ff, k1, j1)

            # Check if there are too many threshold passers, if so report that no lines were found
            if not checkWhiteRatio(img, img_handle.ff, max_white_ratio):
                return line_results



        # # Show thresholded image
        # show(str(frame_min) + "-" + str(frame_max) + " threshold", img)

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

        # # Show morphed over maxpixel
        # temp = ff.maxpixel - img.astype(np.int16)*255
        # temp[temp>0] = 0
        # show('compare', temp-ff.maxpixel)

        # # Show morphed image
        # show(str(frame_min) + "-" + str(frame_max) + " morph", img)

        ###

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
        if lines.any():
            for rho, theta in lines:
                line_results.append([rho, theta, frame_min, frame_max])


        # if line_results:
        #     plotLines(ff, line_results)


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

        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y)[0]

        return m, c



    def _connectBrokenChains(centroids, max_distance, last_count=0):
        """ Connect broken chains of centroids.
        """

        filtered_chains = []
        paired_indices = []
        for i, chain1 in enumerate(chains):

            # SKip if chain already connected
            if i in paired_indices:
                continue

            # Get last point of first chain
            x1 = chain1[-1][1]
            y1 = chain1[-1][2]

            found_pair = False
            for j, chain2 in enumerate(chains[i+1:]):

                # Correct for real position in the chain
                j = j + i + 1

                # Skip if chain already connected
                if j in paired_indices:
                    continue

                # Get first point of the second chain
                x2 = chain2[0][1]
                y2 = chain2[0][2]

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
            filtered_chains = _connectBrokenChains(centroids, max_distance, len(filtered_chains))

        return filtered_chains



    # Skip centroid correction if there are not conteroids, of there's only one
    if len(centroids) < 2:
        return centroids

    centroids_array = np.array(centroids)

    # Separate by individual columns of the centroid array
    frame_array = centroids_array[:,0]
    x_array = centroids_array[:,1]
    y_array = centroids_array[:,2]

    # LSQ fit by both axes
    mX, cX = _LSQfit(x_array, frame_array)
    mY, cY = _LSQfit(y_array, frame_array)

    filtered_centroids = []

    # Distances between points and fitted line
    point_deviations = _pointDistance(x_array, y_array, mX*frame_array + cX, mY*frame_array + cY)

    # Calculate average deviation
    mean_deviation = np.mean(point_deviations)

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
        x1 = filtered_centroids[i][1]
        y1 = filtered_centroids[i][2]

        # Next point position
        x2 = filtered_centroids[i+1][1]
        y2 = filtered_centroids[i+1][2]

        # Check if the current and the next point are sufficiently close
        if _pointDistance(x1, y1, x2, y2) <= max_distance:
            chains[chain_index].append(filtered_centroids[i+1])
        else:
            # Make new chain
            chains.append([filtered_centroids[i+1]])
            chain_index += 1
            

    # Connect broken chains
    filtered_chains = _connectBrokenChains(centroids, max_distance)


    # Choose the chain with the greatest length
    chain_lengths = [len(chain) for chain in filtered_chains]
    max_chain_length = max(chain_lengths)

    best_chain = filtered_chains[chain_lengths.index(max_chain_length)]


    return best_chain



def checkAngularVelocity(centroids, config):
    """ Check the angular velocity of the detection, and reject those too slow or too fast to be meteors. 
        The minimum ang. velocity is 0.5 deg/s, while maximum is 35 deg/s (Peter Gural, private comm.).
    
    Arguments:
        centroids: [ndarray] meteor centroids from the detector
        config: [config object] configuration object (loaded from the .config file)
    
    Return:
        [bool] True if the velocity is in the meteor ang. velocity range, False otherwise

    """

    # Calculate the angular velocity in px/frame
    first_centroid = centroids[0]
    last_centroid = centroids[-1]
    
    frame1, x1, y1, _ = first_centroid
    frame2, x2, y2, _ = last_centroid

    ang_vel = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)/float(frame2 - frame1 + 1)

    # Convert to px/sec
    ang_vel = ang_vel*config.fps

    # Convert to deg/sec
    scale = (config.fov_h/float(config.height) + config.fov_w/float(config.width))/2.0
    ang_vel = ang_vel*scale

    # Check if the meteor is in the possible angular velocity range (deg/s)
    if (ang_vel >= config.ang_vel_min and ang_vel <= config.ang_vel_max):
        return True

    else:
        return False



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



def plotLines(ff, line_list):
    """ Plot lines on the image.
    """

    img = np.copy(ff.maxpixel)

    hh = img.shape[0]/2.0
    hw = img.shape[1]/2.0

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
        
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
        
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



def preprocessFF(img_handle, mask, flat_struct):
    """ Apply the mask and flat to FF file in img_handle. """

    # Mask the FF file
    img_handle.ff = MaskImage.applyMask(img_handle.ff, mask, ff_flag=True)

    # Apply the flat to maxpixel and avepixel
    if flat_struct is not None:

        img_handle.ff.maxpixel = Image.applyFlat(img_handle.ff.maxpixel, flat_struct)
        img_handle.ff.avepixel = Image.applyFlat(img_handle.ff.avepixel, flat_struct)


    return img_handle





def detectMeteors(img_handle, config, flat_struct=None):
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
    
    Return:
        meteor_detections: [list] a list of detected meteors, with these elements:
            - rho: [float] meteor line distance from image center (polar coordinates, in pixels)
            - theta: [float] meteor line angle from image center (polar coordinates, in degrees)
            - centroids: [list] [frame, X, Y, level] list of meteor points
    """


    t1 = time()
    t_all = time()

    # Load the mask file
    mask = MaskImage.loadMask(config.mask_file)


    # Do all image processing on single FF file, if given
    #   Otherwise, the image processing will be done on every frame chunk that is extracted
    if img_handle == 'ff':

        # Apply mask and flat to FF
        img_handle = preprocessFF(img_handle, mask, flat_struct)


    # Get lines on the image
    line_list = getLines(img_handle, config.k1_det, config.j1_det, config.time_slide, config.time_window_size, 
        config.max_lines_det, config.max_white_ratio, config.kht_lib_path, mask, flat_struct)

    logDebug('List of lines:', line_list)


    # Init meteor list
    meteor_detections = []

    # Only if there are some lines in the image
    if len(line_list):

        # Join similar lines
        line_list = mergeLines(line_list, config.line_min_dist, img_handle.ff.ncols, img_handle.ff.nrows)

        logDebug('Time for finding lines:', time() - t1)

        logDebug('Number of KHT lines: ', len(line_list))
        logDebug(line_list)

        # Plot lines
        # plotLines(img_handle.ff, line_list)


        filtered_lines = []

        # Analyze stripes of each line
        # This step makes sure that there is a linear propagation of the detections in time
        for line in line_list:

            rho, theta, frame_min, frame_max = line

            logDebug('rho, theta, frame_min, frame_max')
            logDebug(rho, theta, frame_min, frame_max)


            # If FF files are not used as input, reconstruct it
            if img_handle.input_type != 'ff':

                # Compute the FF for this chunk
                img_handle.loadChunk(first_frame=frame_min, read_nframes=(frame_max - frame_min))

                # Apply mask and flat to FF
                img_handle = preprocessFF(img_handle, mask, flat_struct)


                # # Show the image chunk, average subtracted
                # plt.imshow(img_handle.ff.maxpixel - img_handle.ff.avepixel, cmap='gray')
                # plt.show()
                

            # Extract (x, y, frame) of thresholded frames, i.e. pixel and frame locations of threshold passers
            xs, ys, zs = getThresholdedStripe3DPoints(config, img_handle, frame_min, frame_max, rho, theta, \
                mask, flat_struct)
            

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

            logDebug('time for GROUPING: ', time() - t1)

            # Extract the first and only line if any
            if detected_line:
                detected_line = detected_line[0]

                # logDebug(detected_line)

                # # Show 3D cloud
                # show3DCloud(img_handle.ff, xs, ys, zs, detected_line, stripe_points, config)

                # Add the line to the results list
                filtered_lines.append(detected_line)

        # Merge similar lines in 3D
        filtered_lines = merge3DLines(filtered_lines, config.vect_angle_thresh)

        logDebug('after filtering:')
        logDebug(filtered_lines)


        ### TEST!!!
        print('EXITING...')
        sys.exit()

        ###


        # If the input is a single FF file, threshold the image right away
        if img_handle.input_type == 'ff':

            # Threshold the FF
            img_thres = thresholdFF(img_handle.ff, config.k1_det, config.j1_det)


            # Gamma correct image files
            maxpixel_gamma_corr = Image.gammaCorrection(img_handle.ff.maxpixel, config.gamma)
            avepixel_gamma_corr = Image.gammaCorrection(img_handle.ff.avepixel, config.gamma)
            stdpixel_gamma_corr = Image.gammaCorrection(img_handle.ff.stdpixel, config.gamma)


            # Calculate weights for centroiding (apply gamma correction on both images)
            max_avg_corrected = maxpixel_gamma_corr - avepixel_gamma_corr
            flattened_weights = (max_avg_corrected).astype(np.float32)/stdpixel_gamma_corr


            # At the end, a check that the detection has a surface brightness above the background will be performed.
            # The assumption here is that the peak of the meteor should have the intensity which is at least
            # that of a patch of 4x4 pixels that are of the mean background brightness
            min_patch_intensity = 4*4*(np.mean(maxpixel_gamma_corr - avepixel_gamma_corr) \
                + config.k1_det*np.mean(stdpixel_gamma_corr) + config.j1)


        # For other input types
        else:
            pass



        for detected_line in filtered_lines:

            # Get frame range
            frame_min = detected_line[4]
            frame_max = detected_line[5]

            # Check if the line covers a minimum frame range
            if (abs(frame_max - frame_min) + 1 < config.line_minimum_frame_range_det):
                continue

            # Extend the frame range for several frames, just to be sure to catch all parts of a meteor
            frame_min -= config.frame_extension
            frame_max += config.frame_extension

            # Cap values to 0-255
            frame_min = max(frame_min, 0)
            frame_max = min(frame_max, 255)

            logDebug(detected_line)

            # Get coordinates of 2 points that describe the line
            x1, y1, z1 = detected_line[0]
            x2, y2, z2 = detected_line[1]

            # Convert Cartesian line coordinates to polar
            rho, theta = getPolarLine(x1, y1, x2, y2, ff.nrows, ff.ncols)

            # Convert Cartesian line coordinate to CAMS compatible polar coordinates (flipped Y axis)
            rho_cams, theta_cams = getPolarLine(x1, ff.nrows - y1, x2, ff.nrows - y2, ff.nrows, ff.ncols)


            logDebug('converted rho, theta')
            logDebug(rho, theta)

            # Bounded the thresholded image by min and max frames
            img = FFfile.selectFFFrames(np.copy(img_thres), ff, frame_min, frame_max)

            # Remove lonely pixels
            img = morph.clean(img)


            # Get indices of stripe pixels around the line
            stripe_indices = getStripeIndices(rho, theta, int(config.stripe_width*1.5), img.shape[0], img.shape[1])

            # Extract the stripe from the thresholded image
            stripe = np.zeros((ff.nrows, ff.ncols), np.uint8)
            stripe[stripe_indices] = img[stripe_indices]

            # Show detected line
            # show('detected line: '+str(frame_min)+'-'+str(frame_max), stripe)

            # Get stripe positions
            stripe_positions = stripe.nonzero()
            xs = stripe_positions[1]
            ys = stripe_positions[0]
            zs = ff.maxframe[stripe_positions]

            # Make an array to feed into the centroiding algorithm
            stripe_points = np.vstack((xs, ys, zs))
            stripe_points = np.swapaxes(stripe_points, 0, 1)
            
            # Sort stripe points by frame
            stripe_points = stripe_points[stripe_points[:,2].argsort()]

            # Show 3D cloud
            # show3DCloud(ff, stripe, detected_line, stripe_points, config)

            # Get points of the given line
            line_points = getAllPoints(stripe_points, x1, y1, z1, x2, y2, z2, config, 
                fireball_detection=False)

            # Skip if no points were returned
            if not line_points.any():
                continue

            # Skip if the points cover too small a frame range
            if abs(np.max(line_points[:,2]) - np.min(line_points[:,2])) + 1 < config.line_minimum_frame_range_det:
                continue


            # Compute the average angular velocity in px per frame
            ang_vel_avg = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)/(z2 - z1)

            # Calculate centroids
            centroids = []


            for i in range(frame_min, frame_max+1):
                
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
                        half_frame_pixels = frame_pixels[frame_pixels[:,1] % 2 == (config.deinterlace_order 
                            + half_frame) % 2]

                        # Deinterlace by fields (stripe pixels)
                        half_frame_pixels_stripe = frame_pixels_stripe[frame_pixels_stripe[:,1] % 2 == (config.deinterlace_order 
                            + half_frame) % 2]


                        # Skip if there are no pixels in the half-frame
                        if not len(half_frame_pixels):
                            continue

                        # Calculate half-frame value
                        frame_no = i+half_frame*0.5


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
                    x_centroid = np.sum(x_weighted)/float(np.sum(max_weights))

                    y_weighted = half_frame_pixels[:,1]*np.transpose(max_weights)
                    y_centroid = np.sum(y_weighted)/float(np.sum(max_weights))


                    # Correct the rolling shutter effect
                    if config.deinterlace_order == -1:

                        # Compute the corrected frame time
                        frame_no = RollingShutterCorrection.correctRollingShutterTemporal(frame_no, \
                            y_centroid, ff.maxpixel.shape[0])


                    # Calculate intensity as the sum of threshold passer pixels on the stripe
                    #intensity_values = max_avg_corrected[half_frame_pixels[:,1], half_frame_pixels[:,0]]
                    intensity_values = max_avg_corrected[half_frame_pixels_stripe[:,1], 
                        half_frame_pixels_stripe[:,0]]
                    intensity = np.sum(intensity_values)
                    
                    logDebug("centroid: ", frame_no, x_centroid, y_centroid, intensity)

                    centroids.append([frame_no, x_centroid, y_centroid, intensity])


            # Filter centroids
            centroids = filterCentroids(centroids, config.centroids_max_deviation, 
                config.centroids_max_distance)

            # Convert to numpy array for easy slicing
            centroids = np.array(centroids)

            # Reject the solution if there are too few centroids
            if len(centroids) < config.line_minimum_frame_range_det:
                continue


            # Check that the detection has a surface brightness above the background
            # The assumption here is that the peak of the meteor should have the intensity which is at least
            # that of a patch of 4x4 pixels that are of the mean background brightness
            if np.max(centroids[:, 3]) < min_patch_intensity:
                continue


            # Check the detection if it has the proper angular velocity
            if not checkAngularVelocity(centroids, config):
                continue


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

    # Load config file
    config = cr.parse(".config")


    # Measure the time of the whole operation
    time_whole = time()


    ### Init the logger

    from RMS.Logger import initLogging
    initLogging('detection_')

    log = logging.getLogger("logger")

    ######

    
    if len(sys.argv) == 1:
        print("Usage: python -m RMS.Detection /path/to/ff/files/")
        sys.exit()


    dir_path = sys.argv[1]
        

    # Detect the input file format
    img_handle_main = detectInputType(dir_path, config)

    img_handle_list = []

    # If the input format are FF files, break them down into single FF files
    if img_handle_main.input_type == 'ff':
        if not img_handle_main.single_ff:

            # Go through all FF files and add them as individual files
            for file_name in img_handle_main.ff_list:

                img_handle = detectInputType(os.path.join(dir_path, file_name), config, skip_ff_dir=True)

                img_handle_list.append(img_handle)


    # Otherwise, just add the main image handle to the list
    if len(img_handle_list) == 0:
        img_handle_list.append(img_handle_main)



    # Try loading a flat field image
    flat_struct = None

    if config.use_flat:
        
        # Check if there is flat in the data directory
        if os.path.exists(os.path.join(dir_path, config.flat_file)):
            flat_struct = Image.loadFlat(dir_path, config.flat_file)

        # Try loading the default flat
        elif os.path.exists(config.flat_file):
            flat_struct = Image.loadFlat(os.getcwd(), config.flat_file)



    # Init results list
    results_list = []

    # Open a file for results
    results_path = os.path.abspath(dir_path) + os.sep
    results_name = results_path.split(os.sep)[-2]
    results_file = open(results_path + results_name + '_results.txt', 'w')

    total_meteors = 0

    # Run meteor search on every file
    for img_handle in img_handle_list:

        print('--------------------------------------------')
        print(img_handle.name())

        # Run the meteor detection algorithm
        meteor_detections = detectMeteors(img_handle, config, flat_struct=flat_struct)

        meteor_No = 1
        for meteor in meteor_detections:

            rho, theta, centroids = meteor

            # Print detection to file
            results_file.write('-------------------------------------------------------\n')
            results_file.write(img_handle.name() + '\n')
            results_file.write(str(rho) + ',' + str(theta) + '\n')
            results_file.write(str(centroids)+'\n')

            # Append to the results list
            results_list.append([img_handle.name(), meteor_No, rho, theta, centroids])
            meteor_No += 1

            total_meteors += 1


    results_file.close()


    ftpdetectinfo_name = 'FTPdetectinfo_' + results_name + '.txt'

    # Write FTPdetectinfo file
    FTPdetectinfo.writeFTPdetectinfo(results_list, results_path, ftpdetectinfo_name, results_path, 
        config.stationID, config.fps)
                


    print('Time for the whole directory:', time() - time_whole)
    print('Detected meteors:', total_meteors)