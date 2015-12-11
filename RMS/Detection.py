# RPi Meteor Station
# Copyright (C) 2015  Dario Zubovic
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

import numpy as np
import numpy.ctypeslib as npct
import cv2
from time import time
import sys, os
import ctypes

# Plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

# RMS imports
from RMS.Formats import FFbin
import RMS.ConfigReader as cr
from RMS.Routines import MorphologicalOperations as morph
from RMS.Routines.Grouping3D import find3DLines, getAllPoints

def show(name, img):
    cv2.imshow(name, img.astype(np.uint8)*255)
    cv2.moveWindow(name, 0, 0)
    cv2.waitKey(0)
    cv2.destroyWindow(name)

def show2(name, img):
    cv2.imshow(name, img)
    cv2.moveWindow(name, 0, 0)
    cv2.waitKey(0)
    cv2.destroyWindow(name)

def thresholdImg(ff, k1, j1):
    """ Threshold the image with given parameters. """

    return ff.maxpixel - ff.avepixel > (k1 * ff.stdpixel + j1)

def selectFrames(img_thres, ff, frame_min, frame_max):
    """ Select only pixels in a given frame range. """

    # Get the indices of image positions with times correspondng to the subdivision
    indices = np.where((ff.maxframe >= frame_min) & (ff.maxframe <= frame_max))

    # Reconstruct the image with given indices
    img = np.zeros((ff.nrows, ff.ncols), dtype=np.uint8)
    img[indices] = img_thres[indices]

    return img

def getPolarLine(x1, y1, x2, y2, img_h, img_w):
    """ Calculate polar line coordinates (Hough transform coordinates) rho and theta given the 2 points that 
        define a line. Coordinate system starts in the image center, to replicate the used HT implementation.
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

def getStripeIndices(rho, theta, stripe_width, img_h, img_w):
    """ Get indices of a stripe centered on a lines.
    @param rho: [float] line distance from the center in HT space (pixels)
    @param theta: [float] angle in degrees in HT space
    @param stripe_width: [int] width of the stripe around the line
    @param img_h: [int] original image height
    @param img_w: [int] original image width
    @return (indicesx, indicesy): [tuple] a tuple of x and y indices of stripe pixels
    """

    # Check for vertical lines
    if (theta % 180 == 0):
        theta += 0.001

    hh = img_h / 2.0
    hw = img_w / 2.0

    indicesy = []
    indicesx = []
     
    if theta < 45 or (theta > 90 and theta < 135):
        theta = np.deg2rad(theta)
        half_limit = stripe_width/2 / np.cos(theta)
        a = -np.tan(theta)
        b = rho/np.cos(theta)
         
        for y in xrange(int(-hh), int(hh)):
            x0 = a*y + b
             
            x1 = int(x0 - half_limit + hw)
            x2 = int(x0 + half_limit + hw)
             
            if x1 > x2:
                x1, x2 = x2, x1
             
            if x2 < 0 or x1 >= img_w:
                continue
             
            for x in xrange(x1, x2):
                if x < 0 or x >= img_w:
                    continue
                 
                indicesy.append(y+hh)
                indicesx.append(x)
                 
    else:
        theta = np.deg2rad(theta)
        half_limit = stripe_width/2 / np.sin(theta)
        a = -1/np.tan(theta)
        b = rho/np.sin(theta)
         
        for x in xrange(int(-hw), int(hw)):
            y0 = a*x + b
             
            y1 = int(y0 - half_limit + hh)                        
            y2 = int(y0 + half_limit + hh)
             
            if y1 > y2:
                y1, y2 = y2, y1
             
            if y2 < 0 or y1 >= img_h:
                continue
                
            for y in xrange(y1, y2):
                if y < 0 or y >= img_h:
                    continue
                 
                indicesy.append(y)
                indicesx.append(x+hw)

    return (indicesy, indicesx)

def mergeLines(line_list, min_distance, last_count=0):
    """ Merge similar lines defined by rho and theta. 

    @param: line_list: [list] a list of (rho, phi, min_frame, max_frame) tuples which define a KHT line
    @param: min_distance: [float] minimum distance between two vectors described by line parameters for the 
        lines to be joined
    @param: last_count: [int] used for recursion, default is 0 and it should be left as is

    @return final_list: [list] a list of (rho, phi, min_frame, max_frame) tuples after line merging
    """

    def _getCartesian(rho, theta):
        """ Convert rho and theta to cartesian x and y points. """
        return np.cos(np.radians(theta)) * rho, np.sin(np.radians(theta)) * rho

    def _pointDist(x1, y1, x2, y2):
        """ Euclidian distance between two points. """
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

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

        # Get cartesian coordinates of a point described by the polar vector
        x1, y1 = _getCartesian(rho1, theta1)

        for j, line2 in enumerate(line_list[i+1:]):

            # Skip if the line was paired
            if j in paired_indices:
                continue

            # Get polar coordinates of line
            rho2, theta2 = line2[0:2]

            # Get cartesian coordinates of a point described by the polar vector
            x2, y2 = _getCartesian(rho2, theta2)

            # Check if the points are close enough
            if _pointDist(x1, y1, x2, y2) <= min_distance:

                # Remove old lines
                paired_indices.append(i)
                paired_indices.append(i + j + 1)

                # Merge line
                x_avg = (x1 + x2) / 2
                y_avg = (y2 + y1) / 2

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
        final_list = mergeLines(final_list, min_distance, len(final_list))

    return final_list


def merge3DLines(line_list, vect_angle_thresh, last_count=0):
    """ Merges similar lines found by the 3D detector. It calculates the vector between the first point of the
        first line and the last point of the second line, and then compares the angle difference to individual
        line vectors. If all vecters have angles that are close enough, merge the line. Frame ranges also have
        to overlap to merge the line.

        @param: line_list: [list] a list of lines found by grouping3D algorithm
        @param: vect_angle_thresh: [float] minimum angle between vectors to merge the lines
        @param: last_count: [int] used for recursion, default is 0 and it should be left as is

        @return: final_list: [list] a list of merged lines

    """

    def _vectorAngle(v1, v2):
        """ Calculates an angle between two vectors in degrees. """

        # Calculate angle between vectors
        cosang = np.dot(v1, v2)
        sinang = np.linalg.norm(np.cross(v1, v2))
        return np.degrees(np.arctan2(sinang, cosang))


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
                    paired_indices.append(i + j + 1)

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

def getLines(ff, k1, j1, time_slide, time_window_size, max_lines, max_white_ratio, kht_lib_path):
    """ Get (rho, phi) pairs for each meteor present on the image using KHT.
    @param: ff: [FF bin object] FF bin file loaded into the FF bin class
    @param: k1: [float] weight parameter for the standard deviation during thresholding
    @param: j1: [float] absolute threshold above average during thresholding
    @param time_slide: [int] subdivision size of the time axis (256 will be divided into 256/time_slide parts)
    @param time_window_size: [int] size of the time window which will be slided over the time axis
    @param max_lines: [int] maximum number of lines to find by KHT
    @param max_white_ratio: [float] max ratio between write and all pixels after thresholding
    @param kht_lib_path: [string] path to the compiled KHT library
    @return

    """

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

    # Threshold the image
    img_thres = thresholdImg(ff, k1, j1)

    # Check if the image is too "white" and any futher processing makes no sense
    # This checks the max percentage of white pixels in the thresholded image
    print 'white ratio', np.count_nonzero(img_thres) / float(ff.nrows * ff.ncols)
    if np.count_nonzero(img_thres) / float(ff.nrows * ff.ncols) > max_white_ratio:
        return line_results

    # Subdivide the image by time into overlapping parts (decreases noise when searching for meteors)
    for i in range(0, 256/time_slide-1):

        frame_min = i*time_slide
        frame_max = i*time_slide+time_window_size

        # Select the time range of the thresholded image
        img = selectFrames(img_thres, ff, frame_min, frame_max)

        # Show thresholded image
        # show(str(frame_min) + "-" + str(frame_max) + " treshold", img)

        ### Apply morphological operations to prepare the image for KHT

        # Remove lonely pixels
        img = morph.clean(img)
        
        # Connect close pixels
        img = morph.bridge(img)
        
        # Close surrounded pixels
        img = morph.close(img)
        
        # Thin all lines to 1px width
        img = morph.repeat(morph.thin, img, None)
        
        # Remove lonely pixels
        img = morph.clean(img)

        # Show morphed image
        # show(str(frame_min) + "-" + str(frame_max) + " morph", img)

        ###

        # Get image shape
        w, h = img.shape[1], img.shape[0]

        # Convert the image to feed it into the KHT
        img = (img.flatten().astype(np.byte)*255).astype(np.byte)
        
        # Predefine the line output
        lines = np.empty((max_lines, 2), np.double)
        
        # Call the KHT line finding
        length = kht.kht_wrapper(lines, img, w, h, max_lines, 9, 2, 0.1, 0.004, 1)
        
        # Cut the line array to the number of found lines
        lines = lines[:length]

        # Skip further operations if there are no lines
        if lines.any():
            for rho, theta in lines:
                line_results.append([rho, theta, frame_min, frame_max])


    return line_results


def plotLines(ff, line_list):
    """ Plot lines on the image. """

    img = np.copy(ff.maxpixel)

    hh = img.shape[0] / 2.0
    hw = img.shape[1] / 2.0

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


def show3DCloud(ff, stripe, detected_line=[], stripe_points=None, config=None):
    """ Shows 3D point cloud of stripe points. """

    stripe_indices = stripe.nonzero()

    xs = stripe_indices[1]
    ys = stripe_indices[0]
    zs = ff.maxframe[stripe_indices]

    print 'points:', len(xs)

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




if __name__ == "__main__":
    # time_window_size = 64
    # time_slide = 32
    # k1 = 1.5
    # j1 = 9
    # stripe_width = 20

    # max_points = 700

    # Minimum distance between KHT lines in Cartesian space to merge them
    # line_min_dist = 40

    # kht_lib_path = "build/lib.linux-x86_64-2.7/kht_module.so"


    
    if len(sys.argv) == 1:
        print "Usage: python -m RMS.Detection /path/to/bin/files/"
        sys.exit()
    
    # Get paths to every FF bin file in a directory 
    ff_list = [ff for ff in os.listdir(sys.argv[1]) if ff[0:2]=="FF" and ff[-3:]=="bin"]

    # Check if there are any file in the directory
    if(len(ff_list) == None):
        print "No files found!"
        sys.exit()

    # Load config file
    config = cr.parse(".config")

    vect_angle_thresh = 5

    # Run meteor search on every file
    for ff_name in ff_list:

        print ff_name

        t1 = time()

        # Load the FF bin file
        ff = FFbin.read(sys.argv[1], ff_name)

        # Get lines on the image
        line_list = getLines(ff, config.k1_det, config.j1, config.time_slide, config.time_window_size, 
            config.max_lines_det, config.max_white_ratio, config.kht_lib_path)

        # Only if there are some lines in the image
        if len(line_list):

            # Join similar lines
            line_list = mergeLines(line_list, config.line_min_dist)

            print 'time for finding lines', time() - t1

            print 'number of KHT lines:', len(line_list)
            print line_list

            # Plot lines
            plotLines(ff, line_list)

            # Threshold the image
            img_thres = thresholdImg(ff, config.k1_det, config.j1)

            filtered_lines = []

            # Analyze stripes of each line
            for line in line_list:
                rho, theta, frame_min, frame_max = line

                print 'rho, theta, frame_min, frame_max'
                print rho, theta, frame_min, frame_max

                # Bounded the thresholded image by min and max frames
                img = selectFrames(np.copy(img_thres), ff, frame_min, frame_max)

                # Remove lonely pixels
                img = morph.clean(img)

                # Get indices of stripe pixels around the line
                stripe_indices = getStripeIndices(rho, theta, config.stripe_width, img.shape[0], img.shape[1])

                # Extract the stripe from the thresholded image
                stripe = np.zeros((ff.nrows, ff.ncols), np.uint8)
                stripe[stripe_indices] = img[stripe_indices]

                # Show stripe
                # show2("stripe", stripe*255)

                # Show 3D could
                # show3DCloud(ff, stripe)

                # Get stripe positions
                stripe_positions = stripe.nonzero()
                xs = stripe_positions[1]
                ys = stripe_positions[0]
                zs = ff.maxframe[stripe_positions]

                # Limit the number of points to search if too large
                if len(zs) > config.max_points_det:
                    indices = np.random.choice(len(zs), config.max_points_det, replace=False)
                    ys = ys[indices]
                    xs = xs[indices]
                    zs = zs[indices]

                # Make an array to feed into the gropuing algorithm
                stripe_points = np.vstack((xs, ys, zs))
                stripe_points = np.swapaxes(stripe_points, 0, 1)
                
                # Sort stripe points by frame
                stripe_points = stripe_points[stripe_points[:,2].argsort()]

                t1 = time()

                # Find a single line in the point cloud
                detected_line = find3DLines(stripe_points, time(), config, fireball_detection=False)

                print 'time for GROUPING: ', time() - t1

                # Extract the first and only line if any
                if detected_line:
                    detected_line = detected_line[0]

                    # print detected_line

                    # # Show 3D cloud
                    # show3DCloud(ff, stripe, detected_line, stripe_points, config)

                    # Add the line to the results list
                    filtered_lines.append(detected_line)

            # Merge similar lines in 3D
            filtered_lines = merge3DLines(filtered_lines, vect_angle_thresh)

            print 'after filtering:'
            print filtered_lines

            for detected_line in filtered_lines:

                print detected_line

                # Get coordinates of 2 points that describe the line
                x1, y1, z1 = detected_line[0]
                x2, y2, z2 = detected_line[1]

                # Convert Cartesian line coordinates to polar
                rho, theta = getPolarLine(x1, y1, x2, y2, ff.nrows, ff.ncols)

                print 'converted rho, theta'
                print rho, theta

                # Get frame range
                frame_min = detected_line[4]
                frame_max = detected_line[5]

                # Bounded the thresholded image by min and max frames
                img = selectFrames(np.copy(img_thres), ff, frame_min, frame_max)

                # Remove lonely pixels
                img = morph.clean(img)

                # Get indices of stripe pixels around the line
                stripe_indices = getStripeIndices(rho, theta, config.stripe_width, img.shape[0], img.shape[1])

                # Extract the stripe from the thresholded image
                stripe = np.zeros((ff.nrows, ff.ncols), np.uint8)
                stripe[stripe_indices] = img[stripe_indices]


                show('detected line', stripe)

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

                # Calculate centroids
                centroids = []

                for i in range(frame_min, frame_max+1):
                    
                    # Select pixel indicies belonging to a given frame
                    frame_pixels_inds = np.where(line_points[:,2] == i)
                    
                    # Get pixel positions in a given frame
                    frame_pixels = line_points[frame_pixels_inds].astype(np.int64)

                    # Skip if there are no pixels in the frame
                    if not len(frame_pixels):
                        continue

                    # Calculate centroids by half-frame
                    for half_frame in range(2):
                        half_frame_pixels = frame_pixels[frame_pixels[:,1] % 2 == (config.deinterlace_order + half_frame) % 2]

                        # Skip if there are no pixels in the half-frame
                        if not len(half_frame_pixels):
                            continue

                        # Calculate half-frame value
                        frame_no = i+half_frame*0.5

                        # Get maxpixel-avepixel values of given pixel indices (this will be used as weights)
                        max_weights = (ff.maxpixel-ff.avepixel)[half_frame_pixels[:,1], half_frame_pixels[:,0]]

                        # Calculate weighted centroids
                        x_weighted = half_frame_pixels[:,0] * np.transpose(max_weights)
                        x_centroid = np.sum(x_weighted) / float(np.sum(max_weights))

                        y_weighted = half_frame_pixels[:,1] * np.transpose(max_weights)
                        y_centroid = np.sum(y_weighted) / float(np.sum(max_weights))

                        # Calculate intensity as the sum of white pixels
                        intensity = np.sum(max_weights)
                        
                        print "centroid: ", frame_no, x_centroid, y_centroid, intensity

                        centroids.append([frame_no, x_centroid, y_centroid, intensity])

                centroids = np.array(centroids)

                gs = gridspec.GridSpec(2, 1, width_ratios=[2,2], height_ratios=[2,1])
                # Plot centroids to image
                plt.subplot(gs[0])
                plt.imshow(img_thres, cmap='gray')
                plt.scatter(centroids[:,1], centroids[:,2], s=5, c='r', edgecolors='none')

                plt.subplot(gs[1])
                # Plot lightcurve
                plt.plot(centroids[:,0], centroids[:,3])
                plt.show()
                plt.clf()
                plt.close()