# RPi Meteor Station
# Copyright (C) 2018  Dario Zubovic, Denis Vida
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

import numpy as np
from math import sqrt
from time import time
import logging


# Get the logger from the main module
log = logging.getLogger("logger")


# Cython init
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})

from RMS.Routines.Grouping3Dcy import find3DLines as find3DLinesCy
from RMS.Routines.Grouping3Dcy import getAllPoints as getAllPointsCy
from RMS.Routines.Grouping3Dcy import thresholdAndSubsample as thresholdAndSubsampleCy
from RMS.Routines.Grouping3Dcy import testPoints as testPointsCy
from RMS.Routines.Grouping3Dcy import detectionCutOut as detectionCutOutCy



def getAllPoints(point_list, x1, y1, z1, x2, y2, z2, config, fireball_detection=True):
    """ Only a Cython wrapper function!
        Returns all points describing a particular line. 
    
    Arguments:
        point_list: [ndarray] list of all points
        x1
        ...
        z2: [int] points defining a line in 3D space (begin, end in X, Y, Z).
        config: [config object] defines configuration parameters fro the config file
    
    Return:
        [ndarray] array of all points belonging to a given line
    """

    # Check if working with fireball data or HT data
    if fireball_detection:
        distance_threshold = config.distance_threshold
        gap_threshold = config.gap_threshold

    else:
        distance_threshold = config.distance_threshold_det
        gap_threshold = config.gap_threshold_det

    # Convert the point list to numpy array
    point_list = np.array(point_list, dtype = np.uint16)

    return getAllPointsCy(point_list, x1, y1, z1, x2, y2, z2, distance_threshold, gap_threshold)




def find3DLines(point_list, start_time, config, fireball_detection=True):
    """ Only a Cython wrapper function!
        Iteratively find N straight lines in 3D space.
    
    Arguments:
        point_list: [ndarray] list of all points
        start_time: [time.time() object] starting time of the loop
        config: [config object] defines configuration parameters fro the config file
        get_single: [bool] returns only 1 line, does not perform recusrive line searching
    
    Return:
        [list] list of found lines
    """

    class GroupingConfig(object):
        """ A special config used only for grouping3D algorithm. Default values are fireball detection values.
        """
        def __init__(self, config):
            super(GroupingConfig, self).__init__()

            self.max_lines = config.max_lines
            self.point_ratio_threshold = config.point_ratio_threshold

            self.max_time = config.max_time
            self.distance_threshold = config.distance_threshold
            self.gap_threshold = config.gap_threshold
            self.min_points = config.min_points
            self.min_frames = config.min_frames
            self.line_minimum_frame_range = config.line_minimum_frame_range
            self.line_distance_const = config.line_distance_const


    # Convert the point list to numpy array
    point_list = np.array(point_list, dtype = np.uint16)

    line_list = []

    # Choose proper algorithm parameters, whether finding fireballs or faint meteors
    grouping_config = GroupingConfig(config) # Loads fireball detection parameters by default
    if fireball_detection:
        get_single = False

    else:
        ## Used for faint meteor detection

        # Find only one line
        get_single = True

        # Load faint meteor detecion parameters instead of fireball detection parameters
        grouping_config.max_time = config.max_time_det
        grouping_config.distance_threshold = config.distance_threshold_det
        grouping_config.gap_threshold = config.gap_threshold_det
        grouping_config.min_points = config.min_pixels_det
        grouping_config.line_minimum_frame_range = config.line_minimum_frame_range_det
        grouping_config.line_distance_const = config.line_distance_const_det

        # These parameters are important only in fireball detection, use these values for faint detection
        grouping_config.max_lines = 1
        grouping_config.point_ratio_threshold = 1


    # Call a fast cython function for finding lines in 3D
    return find3DLinesCy(point_list, start_time, grouping_config, get_single, line_list)




def findCoefficients(line_list):
    """ Extract coefficients from list of lines that can be consumed by RMS.VideoExtraction.
    
    Arguments:
        line_list: [list] list of detected lines
    
    Return:
        coeff: [list] coefficients for each detected line in format: [first point, slope of XZ, slope of YZ, 
            first frame, last frame]
    """
    
    coeff = []
    
    for detected_line in line_list:
        
        if detected_line[0][2] < detected_line[1][2]:
            point1 = np.array(detected_line[0], dtype=np.float64)
            point2 = np.array(detected_line[1], dtype=np.float64)
        elif detected_line[0][2] > detected_line[1][2]:
            point1 = np.array(detected_line[1], dtype=np.float64)
            point2 = np.array(detected_line[0], dtype=np.float64)
        else:
        # skip if points are on the same frame (that shouldn't happen, though)
            log.debug("Points on the same frame!")
            continue
        
        # difference between last point and first point that represent a line
        point3 = point2 - point1
        
        # slope
        slopeXZ = point3[1]/point3[2] # speed on X axis
        slopeYZ = point3[0]/point3[2] # speed on Y axis
        
        # length of velocity vector
        total = sqrt(slopeXZ**2 + slopeYZ**2)
        
        print('Fireball slope:', total)

        # ignore line if too fast
        # TODO: this limit should be read from config file and calculated for FOV
        # 1.6 is better estimate on upper speed limit, set to 2 for safety
        if total > 2:
            continue
        
        coeff.append([point1, slopeXZ, slopeYZ, detected_line[4], detected_line[5]]) #first point, slope of XZ, slope of YZ, first frame, last frame
        
    return coeff



def thresholdAndSubsample(frames, compressed, min_level, min_points, k1, j1, f):
    """ This is only a Cython wrapper function!
        Given the list of frames, threshold them, subsample the time and check if there are enough threshold
        passers on the given frame. 

    Arguments:
        frames: [3D ndarray] Numpy array containing video frames. Structure: (nframe, y, x).
        compressed: [3D ndarray] Numpy array containing compressed video frames. Structure: (frame, y, x), 
            where frames are: maxpixel, maxframe, avepixel, stdpixel
        min_level: [int] The point will be subsampled if it has this minimum pixel level (i.e. brightness).
        min_points: [int] Minimum number of points in the subsampled block that is required to pass the 
            threshold.
        k1: [float] Threhsold max > avg + k1*stddev
        j1: [float] Constant offset in the threshold levels.
        f: [int] Decimation scale

    Return:
        num: [int] Number threshold passers.
        pointsx: [ndarray] X coordinate of the subsampled point.
        pointsy: [ndarray] Y coordinate of the subsampled point.
        pointsz: [ndarray] frame of the subsampled point.
    """

    return thresholdAndSubsampleCy(frames, compressed, min_level, min_points, k1, j1, f)



def testPoints(gap_threshold, pointsy, pointsx, pointsz):
    """ This is only a Cython wrapper function!
        Test if the given 3D point cloud contains a line by testing if there is a large gap between the points
        in time or not.

    Arguments:
        gap_threshold: [int] Maximum gap between points in 3D space.
        pointsy: [ndarray] X coordinates of points.
        pointsx: [ndarray] Y coordinates of points.
        pointsx: [ndarray] Z coordinates of points.

    Return:
        count: [int] Number of points within the gap threshold. 

    """

    return testPointsCy(gap_threshold, pointsy, pointsx, pointsz)



def detectionCutOut(frames, compressed, point, slopeXZ, slopeYZ, first_frame, last_frame, f, \
    intensity_size_threshold, size_min, size_max):
    """ This is only a Cython wrapper function!
        Compute the locations and the size of fireball frame crops. The computed values will be used to
        crop out raw video frames.

    Arguments:
        frames: [3D ndarray]: Raw video frames.
        compressed: [3D array]: FTP compressed 256 frame block.
        point: [ndarray] Coordinates of the first point of the event.
        slopeXZ: [float] Speed of the fireball in X direction in px/frame.
        slopeYZ: [float] Speed of the fireball in Y direction in px/frame.
        first_frame: [int] No. of the first frame.
        last_frame: [int] No. of the last frame.
        f: [int] Decimation factor.
        intensity_size_threshold: [float] Threshold for dynamically estimating the window size based on the
            pixel intensity.
        size_min: [int] Minimum size of the window.
        size_max: [int] Maximum size of the window.

    Return:
        num: [int] Number of extracted windows.
        cropouts: [3D ndarray] Cropped out windows.
        sizepos: [3D ndarray] Array of positions and size of cropouts within the context of the whole frame.

    """

    return detectionCutOutCy(frames, compressed, point, slopeXZ, slopeYZ, first_frame, last_frame, f, \
        intensity_size_threshold, size_min, size_max)