
from time import time

import numpy as np
cimport numpy as np
cimport cython

INT_TYPE = np.uint16
ctypedef np.uint16_t INT_TYPE_t

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

    

    cdef int n_len = (dx1*dy2 - dx2*dy1)**2+(dx2*dz1 - dx1*dz2)**2 + (dy1*dz2 - dy2*dz1)**2

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
    """ Structure that defines a line. """

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
        """ String to print. """

        return " ".join(map(str, (self.x1, self.y1, self.z1, self.x2, self.y2, self.z2, self.counter, 
            self.line_quality)))

    def get_points(self):
        """ Return the starting and ending points of the line. """

        return self.x1, self.y1, self.z1, self.x2, self.y2, self.z2

@cython.boundscheck(False)
def getAllPoints(np.ndarray[INT_TYPE_t, ndim=2] point_list, x1, y1, z1, x2, y2, z2, distance_treshold, gap_treshold, max_array_size=0):
    """ Returns all points describing a particular line. 
    @param point_list: [ndarray] list of all points
    @params x1 to z2: [int] points defining a line in 3D space
    @param distance_treshold: [int] maximum distance between the line and the point to be takes as a part of 
        the same line
    @param gap_treshold: [float] maximum allowed gap between points
    @param max_array_size: [float] predefined size of max_line_points array (optional)
    @return: [ndarray] array of points belonging to a certain line
    """

    cdef int i = 0

    # Number of points in the point list
    point_list_size = point_list.shape[0]

    # Check if the point list is empty
    if point_list_size == 0:
        return np.array([[]])

    def propagateLine(np.ndarray[INT_TYPE_t, ndim=2] max_line_points, np.ndarray[INT_TYPE_t, ndim=2] propagation_list, int i):
        """ Finds all points present on a line starting from a point on that line. """

        cdef int x3, y3, z3, x_prev, y_prev, z_prev, z

        x_prev, y_prev, z_prev = x1, y1, z1

        for z in range(len(propagation_list)):

            # This point defines a single point from a point cloud
            x3 = propagation_list[z, 0]
            y3 = propagation_list[z, 1]
            z3 = propagation_list[z, 2]

            # Check if the distance between the line and the point is close enough
            line_dist = line3DDistance_simple(x1, y1, z1, x2, y2, z2, x3, y3, z3)

            if line_dist < distance_treshold:

                # Calculate the gap from the previous point and reject the solution if the point is too far
                if point3DDistance(x_prev, y_prev, z_prev, x3, y3, z3) > gap_treshold:
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
    cdef np.ndarray[INT_TYPE_t, ndim=2] max_line_points = np.zeros(shape=(max_array_size, 3), dtype = INT_TYPE)

    # Get the index of the first point
    point1_index = np.where(np.all(point_list==np.array((x1, y1, z1)),axis=1))[0]

    # Check if the first point exists, if not start from the point closes to the given point
    if not point1_index:
        best_distance = 999

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



def remove3DPoints(np.ndarray[INT_TYPE_t, ndim=2] point_list, Line max_line, distance_treshold, gap_treshold):
    """ Remove points from a point list that belong to the given line.
    
    @param point_list: [ndarray] list of all points
    @param max_line: [Line object] given line
    @param distance_treshold: [int] maximum distance between the line and the point to be takes as a part of 
        the same line
    @param gap_treshold: [int] maximum allowed gap between points
    @return: [tuple of ndarrays] (array of all points minus the ones in the max_line), (points in the max_line)
    """

    cdef int x1, y1, z1, x2, y2, z2

    # Get max_line ending points
    x1, y1, z1, x2, y2, z2 = max_line.get_points()
    
    # Get all points belonging to the max_line
    max_line_points = getAllPoints(point_list, x1, y1, z1, x2, y2, z2, distance_treshold, gap_treshold, 
        max_array_size=max_line.counter)

    # Get the point could minus points in the max_line
    point_list_rows = point_list.view([('', point_list.dtype)] * point_list.shape[1])
    max_line_points_rows = max_line_points.view([('', max_line_points.dtype)] * max_line_points.shape[1])
    point_list = np.setdiff1d(point_list_rows, max_line_points_rows).view(point_list.dtype).reshape(-1, 
        point_list.shape[1])

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
def find3DLines(np.ndarray[INT_TYPE_t, ndim=2] point_list, start_time, config, get_single=False, line_list=[]):
    """ Iteratively find N straight lines in 3D space.
    
    @param point_list: [ndarray] list of all points
    @param start_time: [time.time() object] starting time of the loop
    @param config: [config object] defines configuration parameters fro the config file
    @param get_single: [bool] returns only 1 line, does not perform recusrive line searching
    @param line_list: [list] list of lines found previously
    @return: list of found lines
    """

    # Load config parameters
    cdef float distance_treshold = config.distance_treshold
    cdef float gap_treshold = config.gap_treshold
    cdef int min_points = config.min_points
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
    results_list = np.zeros(shape=((point_list_size*(point_list_size-1))/2), dtype=Line)

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

                if line_dist < distance_treshold:

                    # Calculate the gap from the previous point and reject the solution if the point is too far
                    if point3DDistance(x_prev, y_prev, z_prev, x3, y3, z3) > gap_treshold:

                        # Reject solution (reset counter) if the last point is too far
                        if point3DDistance(x2, y2, z2, x_prev, y_prev, z_prev) > gap_treshold:
                            counter = 0

                        break

                    counter += 1
                    line_dist_sum += line_dist

                    x_prev, y_prev, z_prev = x3, y3, z3

            # Skip if too little points were found
            if (counter) < min_points:
                continue

            # Average distance between points and the line
            line_dist_avg = line_dist_sum / <float> (counter)

            # calculate a parameter for line quality
            # larger average distance = less quality
            line_quality = <float> counter - line_distance_const * line_dist_avg
            results_list[results_counter] = Line(x1, y1, z1, x2, y2, z2, counter, line_quality)
            results_counter += 1

    # Return empty if no good match was found
    if not results_counter:
        return None

    # Get Line with the best quality
    max_line = results_list[0]
    for i in range(results_counter):
        if results_list[i].line_quality > max_line.line_quality:
            max_line = results_list[i]

    # Ratio of points inside and and all points
    cdef float line_ratio = max_line.counter / results_counter

    # Remove points from the point cloud that belong to line with the best quality
    point_list, max_line_points = remove3DPoints(point_list, max_line, distance_treshold, gap_treshold)

    # Return nothing if no points were found
    if not max_line_points.size:
        return None

    # Get the first and the last frame from the max_line point could
    first_frame = max_line_points[0,2]
    last_frame = max_line_points[len(max_line_points) - 1,2]

    # Reject the line if all points are only in very close frames (eliminate flashes):
    if abs(last_frame - first_frame)+1 >= line_minimum_frame_range:

        # Add max_line to results, as well as the first and the last frame of a meteor
        line_list.append(_formatLine(max_line, first_frame, last_frame))

    # If only one line was desired, return it
    # if there are more lines on the image, recursively find lines
    if (line_ratio < config.point_ratio_treshold) and (results_counter > 10) and (not get_single):
        # Recursively find lines until there are no more points or no lines is found to be good
        find3DLines(point_list, start_time, config, get_single=get_single, line_list = line_list)

    return line_list