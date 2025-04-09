""" Cython functions for 3D line detection. """

from __future__ import division, print_function

from time import time

import numpy as np
cimport numpy as np
cimport cython


# Define numpy types
UINT16_TYPE = np.uint16
ctypedef np.uint16_t UINT16_TYPE_t

UINT8_TYPE = np.uint8
ctypedef np.uint8_t UINT8_TYPE_t


# Declare math functions
cdef extern from "math.h":
    double floor(double)
    double abs(double)
    double sqrt(double)



@cython.cdivision(True) # Don't check for zero division
cdef float line3DDistance_simple(int x1, int y1, int z1, int x2, int y2, int z2, int x0, int y0, int z0):
    """ Calculate distance from line to a point in 3D using simple operations.
    
    @param x1: X coordinate of first point representing line
    @param y1: Y coordinate of first point representing line
    @param z1: Z coordinate of first point representing line
    @param x2: X coordinate of second point representing line
    @param y2: Y coordinate of second point representing line
    @param z2: Z coordinate of second point representing line
    @param x0: X coordinate of a point whose distance is to be calculated
    @param y0: Y coordinate of a point whose distance is to be calculated
    @param z0: Z coordinate of a point whose distance is to be calculated
    
    @return: squared distance
    """

    # Original function:
    # np.linalg.norm(np.cross((point0 - point1), (point0 - point2))) / np.linalg.norm(point2 - point1)

    # Length of vector in the numerator
    cdef int dx1 = x0 - x1
    cdef int dy1 = y0 - y1
    cdef int dz1 = z0 - z1

    cdef int dx2 = x0 - x2
    cdef int dy2 = y0 - y2
    cdef int dz2 = z0 - z2

    

    cdef int n_len = (dx1*dy2 - dx2*dy1)**2 + (dx2*dz1 - dx1*dz2)**2 + (dy1*dz2 - dy2*dz1)**2

    # Length of denominator vector
    cdef int d_len = (x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2

    cdef float result = (<float> n_len) / (<float> d_len)

    return result




cdef int point3DDistance(int x1, int y1, int z1, int x2, int y2, int z2):
    """ Calculate distance between two points in 3D space.
    
    @param x1: X coordinate of first point
    @param y1: Y coordinate of first point
    @param z1: Z coordinate of first point
    @param x2: X coordinate of second point
    @param y2: Y coordinate of second point
    @param z2: Z coordinate of second point
    
    @return: squared distance
    """

    return (x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2




cdef class Line:
    """ Structure that defines a line.
    """

    cdef int x1, y1, z1, x2, y2, z2
    cdef public int counter
    cdef public float line_quality

    def __cinit__(self, x1=0, y1=0, z1=0, x2=0, y2=0, z2=0, counter=0, line_quality=0):
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1

        self.x2 = x2
        self.y2 = y2
        self.z2 = z2

        self.counter = counter
        self.line_quality = line_quality

    def __str__(self):
        """ String to print.
        """

        return " ".join(map(str, (self.x1, self.y1, self.z1, self.x2, self.y2, self.z2, self.counter, 
            self.line_quality)))

    def get_points(self):
        """ Return the starting and ending points of the line.
        """

        return self.x1, self.y1, self.z1, self.x2, self.y2, self.z2




@cython.boundscheck(False)
def getAllPoints(np.ndarray[UINT16_TYPE_t, ndim=2] point_list, x1, y1, z1, x2, y2, z2, distance_threshold, gap_threshold, max_array_size=0):
    """ Returns all points describing a particular line. 
    
    @param point_list: [ndarray] list of all points
    @param x1, y1, z1, x2, y2, z2: [int] points defining a line in 3D space
    @param distance_threshold: [int] maximum distance between the line and the point to be takes as a part of the same line
    @param gap_threshold: [float] maximum allowed gap between points
    @param max_array_size: [float] predefined size of max_line_points array (optional)
    
    @return: [ndarray] array of points belonging to a certain line
    """

    cdef int i = 0

    # Number of points in the point list
    point_list_size = point_list.shape[0]

    # Check if the point list is empty
    if point_list_size == 0:
        return np.array([[]])

    def propagateLine(np.ndarray[UINT16_TYPE_t, ndim=2] max_line_points, np.ndarray[UINT16_TYPE_t, ndim=2] propagation_list, int i):
        """ Finds all points present on a line starting from a point on that line.
        """

        cdef int x3, y3, z3, x_prev, y_prev, z_prev, z

        x_prev, y_prev, z_prev = x1, y1, z1

        for z in range(len(propagation_list)):

            # This point defines a single point from a point cloud
            x3 = propagation_list[z, 0]
            y3 = propagation_list[z, 1]
            z3 = propagation_list[z, 2]

            # Check if the distance between the line and the point is close enough
            line_dist = line3DDistance_simple(x1, y1, z1, x2, y2, z2, x3, y3, z3)

            if line_dist < distance_threshold:

                # Calculate the gap from the previous point and reject the solution if the point is too far
                if point3DDistance(x_prev, y_prev, z_prev, x3, y3, z3) > gap_threshold:
                    break

                max_line_points[i,0] = x3
                max_line_points[i,1] = y3
                max_line_points[i,2] = z3
                i += 1

                x_prev, y_prev, z_prev = x3, y3, z3

        return max_line_points, i


    if max_array_size == 0:
        max_array_size = point_list_size

    # Get all points belonging to the best line
    cdef np.ndarray[UINT16_TYPE_t, ndim=2] max_line_points = np.zeros(shape=(max_array_size, 3), dtype=UINT16_TYPE)

    # Get the index of the first point
    point1_index = np.where(np.all(point_list == np.array((x1, y1, z1)), axis=1))[0]

    # Check if the first point exists, if not start from the point closes to the given point
    if not point1_index:

        best_distance = np.inf

        for j in range(len(point_list)):
            x_temp = point_list[j, 0]
            y_temp = point_list[j, 1]
            z_temp = point_list[j, 2]

            temp_dist = point3DDistance(x1, y1, z1, x_temp, y_temp, z_temp)
            
            if temp_dist < best_distance:
                best_distance = temp_dist
                point1_index = [j]

    # Extract the first point
    point1_index = point1_index[0]

    # Spread point cloud forward
    max_line_points, i = propagateLine(max_line_points, point_list[point1_index:], i)

    # Spread point cloud backwards
    max_line_points, i = propagateLine(max_line_points, (point_list[:point1_index])[::-1], i)

    return max_line_points[:i]



def remove3DPoints(np.ndarray[UINT16_TYPE_t, ndim=2] point_list, Line max_line, distance_threshold, gap_threshold):
    """ Remove points from a point list that belong to the given line.
    
    @param point_list: [ndarray] list of all points
    @param max_line: [Line object] given line
    @param distance_threshold: [int] maximum distance between the line and the point to be takes as a part of the same line
    @param gap_threshold: [int] maximum allowed gap between points
    
    @return: [tuple of ndarrays] (array of all points minus the ones in the max_line), (points in the max_line)
    """

    cdef int x1, y1, z1, x2, y2, z2

    # Get max_line ending points
    x1, y1, z1, x2, y2, z2 = max_line.get_points()
    
    # Get all points belonging to the max_line
    max_line_points = getAllPoints(point_list, x1, y1, z1, x2, y2, z2, distance_threshold, gap_threshold, 
        max_array_size=max_line.counter)

    # Get the point could minus points in the max_line
    point_list_copy = point_list.copy()
    point_list_rows = point_list_copy.view([('', point_list_copy.dtype)] * point_list_copy.shape[1])
    max_line_points_rows = max_line_points.view([('', max_line_points.dtype)] * max_line_points.shape[1])
    point_list = np.setdiff1d(point_list_rows, max_line_points_rows).view(point_list_copy.dtype).reshape(-1, 
        point_list_copy.shape[1])

    # Sort max point only if there are any
    if max_line_points.size:
        max_line_points = max_line_points[max_line_points[:,2].argsort()]

    # Sort_points by frame 
    point_list = point_list[point_list[:,2].argsort()]

    return (point_list, max_line_points)




def _formatLine(line, first_frame, last_frame):
    """ Converts Line object to a list of format:
    (point1, point2, counter, line_quality), first_frame, last_frame
    """

    x1, y1, z1, x2, y2, z2 = line.get_points()

    return [(x1, y1, z1), (x2, y2, z2), line.counter, line.line_quality, first_frame, last_frame]




@cython.boundscheck(False)
@cython.wraparound(False) 
def find3DLines(np.ndarray[UINT16_TYPE_t, ndim=2] point_list, start_time, config, get_single=False, line_list=[]):
    """ Iteratively find N straight lines in 3D space.
    
    @param point_list: [ndarray] list of all points
    @param start_time: [time.time() object] starting time of the loop
    @param config: [config object] defines configuration parameters fro the config file
    @param get_single: [bool] returns only 1 line, does not perform recusrive line searching
    @param line_list: [list] list of lines found previously
    
    @return: list of found lines
    """

    # Load config parameters
    cdef float distance_threshold = config.distance_threshold
    cdef float gap_threshold = config.gap_threshold
    cdef int min_points = config.min_points
    cdef int min_frames = config.min_frames
    cdef int line_minimum_frame_range = config.line_minimum_frame_range
    cdef float line_distance_const = config.line_distance_const

    cdef int i, j, z

    cdef int x1, y1, z1, x2, y2, z2, x3, y3, z3, x_prev, y_prev, z_prev

    cdef int counter = 0
    cdef int results_counter = 0
    cdef float line_dist_sum = 0
    cdef float line_dist
    cdef float line_dist_avg

    # stop iterating if too many lines 
    if len(line_list) >= config.max_lines:
        return line_list

    # stop iterating if running for too long
    if time() - start_time > config.max_time:
        if len(line_list) > 0:
            return line_list
        else:
            return None

    cdef int point_list_size = point_list.shape[0]

    # Define a list for results
    results_list = np.zeros(shape=((point_list_size*(point_list_size-1))//2), dtype=Line)

    for i in range(point_list_size):
        for j in range(point_list_size - i - 1):

            # These 2 points define the line
            x1 = point_list[i, 0]
            y1 = point_list[i, 1]
            z1 = point_list[i, 2]
            # x1, y1, z1 = point_list[i]

            x2 = point_list[i + j + 1, 0]
            y2 = point_list[i + j + 1, 1]
            z2 = point_list[i + j + 1, 2]

            # Don't check point pairs on the same frame, as the velocity can't be computed then
            if z1 == z2:
                continue

            # Include 2 points that make the line in the count
            counter = 0

            # Track average distance from the line
            line_dist_sum = 0

            x_prev, y_prev, z_prev = x1, y1, z1

            for z in range(point_list_size):

                # # Skip if the lines are the same
                # if (i == z) or (z == i+j+1):
                #     continue

                # This point defines a single point from a point cloud
                x3 = point_list[z, 0]
                y3 = point_list[z, 1]
                z3 = point_list[z, 2]

                # Check if the distance between the line and the point is close enough
                line_dist = line3DDistance_simple(x1, y1, z1, x2, y2, z2, x3, y3, z3)

                if line_dist < distance_threshold:

                    # Calculate the gap from the previous point and reject the solution if the point is too far
                    if point3DDistance(x_prev, y_prev, z_prev, x3, y3, z3) > gap_threshold:

                        # Reject solution (reset counter) if the last point is too far
                        if point3DDistance(x2, y2, z2, x_prev, y_prev, z_prev) > gap_threshold:
                            counter = 0

                        break

                    counter += 1
                    line_dist_sum += line_dist

                    x_prev, y_prev, z_prev = x3, y3, z3

            # Skip if too little points were found
            if counter < min_points:
                continue

            # Average distance between points and the line
            line_dist_avg = line_dist_sum / <float> (counter)

            # calculate a parameter for line quality
            # larger average distance = less quality
            line_quality = <float> counter - line_distance_const*line_dist_avg
            results_list[results_counter] = Line(x1, y1, z1, x2, y2, z2, counter, line_quality)
            results_counter += 1

    # Return empty if no good match was found
    if results_counter == 0:
        return None

    # Get Line with the best quality
    max_line = results_list[0]
    for i in range(results_counter):
        if results_list[i].line_quality > max_line.line_quality:
            max_line = results_list[i]

    # Ratio of points inside and and all points
    cdef float line_ratio = max_line.counter / results_counter

    # Remove points from the point cloud that belong to line with the best quality
    point_list, max_line_points = remove3DPoints(point_list, max_line, distance_threshold, gap_threshold)

    # Return nothing if no points were found
    if not max_line_points.size:
        return None

    # Get the first and the last frame from the max_line point could
    first_frame = max_line_points[0,2]
    last_frame = max_line_points[len(max_line_points) - 1,2]


    # Reject the line if all points are only in very close frames (eliminate flashes):
    if abs(last_frame - first_frame) + 1 >= line_minimum_frame_range:

        # Add max_line to results, as well as the first and the last frame of a meteor
        line_list.append(_formatLine(max_line, first_frame, last_frame))

    # If only one line was desired, return it
    # if there are more lines on the image, recursively find lines
    if (line_ratio < config.point_ratio_threshold) and (results_counter > 10) and (not get_single):
        # Recursively find lines until there are no more points or no lines is found to be good
        find3DLines(point_list, start_time, config, get_single=get_single, line_list = line_list)

    return line_list





@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.cdivision(True)
def thresholdAndSubsample(np.ndarray[UINT8_TYPE_t, ndim=3] frames, \
    np.ndarray[UINT8_TYPE_t, ndim=3] compressed, int min_level, int min_points, float k1, float j1, int f):
    """ Given the list of frames, threshold them, subsample the time and check if there are enough threshold
        passers on the given frame. 

    Arguments:
        frames: [3D ndarray] Numpy array containing video frames. Structure: (nframe, y, x).
        compressed: [3D ndarray] Numpy array containing compressed video frames. Structure: (frame, y, x), 
            where frames are: maxpixel, maxframe, avepixel, stdpixel
        min_level: [int] The point will be subsampled if it has this minimum pixel level (i.e. brightness).
        min_points: [int] Minimum number of points in the subsampled block that is required to pass the 
            threshold.
        k1: [float] Threhsold max > avg + k1*stddev
        j1: [float] Constant level offset in the threshold
        f: [int] Decimation scale

    Return:
        num: [int] Number threshold passers.
        pointsx: [ndarray] X coordinate of the subsampled point.
        pointsy: [ndarray] Y coordinate of the subsampled point. 
        pointsz: [ndarray] frame of the subsampled point.
    """

    cdef unsigned int x, y, x2, y2, n, max_val, nframes, x_size, y_size
    cdef unsigned int num = 0
    cdef unsigned int avg_std

    # Calculate the shapes of the subsamples image
    cdef shape_z = frames.shape[0]
    cdef shape_y = int(floor(frames.shape[1]//f))
    cdef shape_x = int(floor(frames.shape[2]//f))
    
    # Init subsampled image arrays
    cdef np.ndarray[np.int32_t, ndim=3] count = np.zeros((shape_z, shape_y, shape_x), np.int32)
    cdef np.ndarray[UINT16_TYPE_t, ndim=1] pointsy = np.zeros((shape_z*shape_y*shape_x), UINT16_TYPE)
    cdef np.ndarray[UINT16_TYPE_t, ndim=1] pointsx = np.zeros((shape_z*shape_y*shape_x), UINT16_TYPE)
    cdef np.ndarray[UINT16_TYPE_t, ndim=1] pointsz = np.zeros((shape_z*shape_y*shape_x), UINT16_TYPE)

    # Extract frames dimensions 
    nframes = frames.shape[0]
    y_size = frames.shape[1]
    x_size = frames.shape[2]
    
    for y in range(y_size):
        for x in range(x_size):

            max_val = compressed[0, y, x]

            # Compute the threshold limit
            avg_std = int(float(compressed[2, y, x]) + k1*float(compressed[3, y, x])) + j1

            # Make sure the threshold limit is not above the maximum possible value
            if avg_std > 255:
                avg_std = 255
            
            if ((max_val > min_level) and (max_val >= avg_std)):

                # Extract frame of maximum intensity
                n = compressed[1, y, x]
                
                # Subsample frame in f*f squares
                y2 = int(floor(y//f))
                x2 = int(floor(x//f))
                
                # Check if there are enough of threshold passers inside of this square
                if count[n, y2, x2] >= min_points:

                    # Put this point to the final list
                    pointsy[num] = y2
                    pointsx[num] = x2
                    pointsz[num] = n
                    num += 1

                    # Don't repeat this number
                    count[n, y2, x2] = -1

                # Increase counter if not enough threshold passers and this number isn't written already
                elif count[n, y2, x2] != -1:
                    count[n, y2, x2] += 1
                
    
    # Cut point arrays to their maximum size
    pointsy = pointsy[:num]
    pointsx = pointsx[:num]
    pointsz = pointsz[:num]

    return num, pointsx, pointsy, pointsz



@cython.boundscheck(False)
@cython.wraparound(False) 
def testPoints(int gap_threshold, np.ndarray[UINT16_TYPE_t, ndim=1] pointsy, \
    np.ndarray[UINT16_TYPE_t, ndim=1] pointsx, np.ndarray[UINT16_TYPE_t, ndim=1] pointsz):
    """ Test if the given 3D point cloud contains a line by testing if there is a large gap between the points
        in time or not.

    Arguments:
        gap_threshold: [int] Maximum gap between points in 3D space.
        pointsy: [ndarray] X coordinates of points.
        pointsx: [ndarray] Y coordinates of points.
        pointsz: [ndarray] Z coordinates of points.

    Return:
        count: [int] Number of points within the gap threshold. 

    """

    cdef unsigned int size, distance, i, count = 0, y_dist, x_dist, z_dist, y_prev = 0, x_prev = 0, z_prev = 0
    
    # Extract the size of arrays
    size = pointsx.shape[0]

    for i in range(size):

        # Compute the distance from the previous point
        x_dist = pointsx[i] - x_prev
        z_dist = pointsz[i] - z_prev
        y_dist = pointsy[i] - y_prev
        
        distance = x_dist**2 + y_dist**2 + z_dist**2
        
        # Count the point if there is no gap from the previous point
        if(distance < gap_threshold):
            count += 1
        
        y_prev = pointsy[i]
        x_prev = pointsx[i]
        z_prev = pointsz[i]

    
    return count



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def detectionCutOut(np.ndarray[UINT8_TYPE_t, ndim=3] frames, np.ndarray[UINT8_TYPE_t, ndim=3] compressed, \
    np.ndarray[UINT16_TYPE_t, ndim=1] point, float slopeXZ, float slopeYZ, int first_frame, int last_frame, \
    int f, float intensity_size_threshold, int size_min, int size_max):
    """ Compute the locations and the size of fireball frame crops. The computed values will be used to
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


    cdef float k
    cdef int x_m, x_p, x_t, y_m, y_p, y_t, half_max_size = size_max//2, half_f = f//2
    cdef int x, y, i, x2, y2, num = 0, max_val, pixel, limit, max_width, max_height, size, half_size, \
        num_equal, frames_ysize, frames_xsize, prev_size, prev_size_counter


    # Init the output crops array
    cdef np.ndarray[UINT8_TYPE_t, ndim=3] cropouts = np.zeros((frames.shape[0], size_max, size_max), \
        UINT8_TYPE)

    # Init the array holding X and Y sizes
    cdef np.ndarray[UINT16_TYPE_t, ndim=2] sizepos = np.zeros((frames.shape[0], 4), UINT16_TYPE)
    

    # Extract frame size
    frames_ysize = frames.shape[1]
    frames_xsize = frames.shape[2]

    # Go though all frames
    prev_size = 0
    for i in range(first_frame, last_frame):
        
        # Calculate position of the detection at current time
        k = <float> (i - point[2])
        y_t = <int> ((<float> point[0] + slopeYZ*k)*f + half_f)
        x_t = <int> ((<float> point[1] + slopeXZ*k)*f + half_f)
            
        # Skip if out of bounds
        if (y_t < 0) or (x_t < 0) or (y_t >= frames_ysize) or (x_t >= frames_xsize):
            continue

        
        # Calculate boundaries for finding max value
        y_m = y_t - half_f
        y_p = y_t + half_f
        x_m = x_t - half_f
        x_p = x_t + half_f

        if y_m < 0:
            y_m = 0
        
        if x_m < 0:
            x_m = 0
        
        if y_p >= frames_ysize:
            y_p = frames_ysize - 1
        

        if x_p >= frames_xsize:
            x_p = frames_xsize - 1
        
        
        # Find max value
        max_val = 0

        for y in range(y_m, y_p):
            for x in range(x_m, x_p):

                pixel = frames[i, y, x]

                if pixel > max_val:
                    max_val = pixel

        
        # Calculate boundaries for finding size
        y_m = y_t - half_max_size
        y_p = y_t + half_max_size
        x_m = x_t - half_max_size
        x_p = x_t + half_max_size

        if y_m < 0:
            y_m = 0
        
        if x_m < 0:
            x_m = 0
        
        if y_p >= frames_ysize:
            y_p = frames_ysize - 1
        
        if x_p >= frames_xsize:
            x_p = frames_xsize - 1
        
        
        # Calculate mean distance from center
        max_width = 0 
        max_height = 0
        num_equal = 1
        limit = <int> (intensity_size_threshold*max_val)

        for y in range(y_m, y_p):
            for x in range(x_m, x_p):
                
                # If the pixel intensity above average is above the limit, increase the size
                if (frames[i, y, x] - compressed[2, y, x]) >= limit:

                    max_height += <int> abs(y_t - y)
                    max_width += <int> abs(x_t - x)
                    num_equal += 1
        
        # Compute size
        if max_height > max_width:
            size = max_height//(<int>sqrt(<float>num_equal))
        else:
            size = max_width//(<int>sqrt(<float>num_equal))

        
        if size < size_min:
            size = size_min

        elif size > half_max_size:
            size = half_max_size



        # If the current size is > than the previous size, use this size of larger for the next 4 frames
        if size >= prev_size:
            prev_size_counter = 4

        else:
            if prev_size_counter > 0:
                prev_size_counter -= 1

        if prev_size_counter > 0:
            if prev_size > size:
                size = prev_size


        
        # Save size
        sizepos[num, 3] = size
        half_size = size//2
        
        # Adjust position for frame extraction if out of borders
        if y_t < half_size:
            y_t = half_size
        
        if x_t < half_size:
            x_t = half_size
        
        if y_t >= frames_ysize - half_size:
            y_t = frames_ysize - 1 - half_size
        
        if x_t >= frames_xsize - half_size:
            x_t = frames_xsize - 1 - half_size
        
        
        # Save location
        sizepos[num, 0] = y_t
        sizepos[num, 1] = x_t
        sizepos[num, 2] = i
        
        # Calculate bounds for frame extraction
        y_m = y_t - half_size
        y_p = y_t + half_size
        x_m = x_t - half_size
        x_p = x_t + half_size
        
        # Crop part of frame
        y2 = 0
        x2 = 0

        for y in range(y_m, y_p):

            x2 = 0

            for x in range(x_m, x_p):

                cropouts[num, y2, x2] = frames[i, y, x]
                x2 += 1
            
            y2 +=1 
        
        
        num += 1


        # Keep the previous frame size
        prev_size = size
    


    return num, cropouts, sizepos