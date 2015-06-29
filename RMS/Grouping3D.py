import numpy as np

def line3DDistance_simple(x1, y1, z1, x2, y2, z2, x0, y0, z0):
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
    @return: distance
    """

    # Original function:
    # np.linalg.norm(np.cross((point0 - point1), (point0 - point2))) / np.linalg.norm(point2 - point1)

    # Length of vector in the numerator
    dx1 = x0 - x1
    dy1 = y0 - y1
    dz1 = z0 - z1

    dx2 = x0 - x2
    dy2 = y0 - y2
    dz2 = z0 - z2

    n_len = (dx1*dy2 - dx2*dy1)**2+(dx2*dz1 - dx1*dz2)**2 + (dy1*dz2 - dy2*dz1)**2

    # Length of denominator vector
    d_len = point3DDistance(x1, y1, z1, x2, y2, z2)

    return n_len / d_len

def point3DDistance(x1, y1, z1, x2, y2, z2):
    """ Calculate distance between two points in 3D space.
    
    @param x1: X coordinate of first point
    @param y1: Y coordinate of first point
    @param z1: Z coordinate of first point
    @param x2: X coordinate of second point
    @param y2: Y coordinate of second point
    @param z2: Z coordinate of second point
    @return: distance
    """

    return (x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2

def getAllPoints(point_list, x1, y1, z1, x2, y2, z2, distance_treshold=2500, gap_treshold=10000):
    """ Return all points describing a particular line.
    
    @param point_list: list of all points
    @param x1: X coordinate of first point representing line
    @param y1: Y coordinate of first point representing line
    @param z1: Z coordinate of first point representing line
    @param x2: X coordinate of second point representing line
    @param y2: Y coordinate of second point representing line
    @param z2: Z coordinate of second point representing line
    @param distance_treshold: maximum distance between the line and the point to be takes as a part of the same line
    @return: list of points
    """

    def propagateLine(propagation_list):
        """ Finds all points present on a line starting from a point on that line.
        """

        x_prev, y_prev, z_prev = x1, y1, z1

        for point3 in propagation_list:

            y3, x3, z3 = point3

            line_dist = line3DDistance_simple(x1, y1, z1, x2, y2, z2, x3, y3, z3)

            if line_dist < distance_treshold:

                # Calculate the gap from the previous point and reject the solution if the point is too far
                if point3DDistance(x_prev, y_prev, z_prev, x3, y3, z3) > gap_treshold:

                    if point3DDistance(x_prev, y_prev, z_prev, x2, y2, z2) > gap_treshold:
                        counter = 0

                    break

                x_prev, y_prev, z_prev = x3, y3, z3

                line_points.append(point3)


    point1 = [y1, x1, z1]
    point2 = [y2, x2, z2]

    line_points = []

    # Spread point cloud forward
    point1_index = point_list.index(point1)
    propagateLine(point_list[point1_index:])

    # Spread point cloud backwards
    propagateLine(reversed(point_list[:point1_index]))

    return line_points


def find3DLines(point_list, line_list=[], distance_treshold=2500, line_distance_const=9, gap_treshold=10000, minimum_points=3, point_ratio_treshold=0.7, max_lines=5):
    """ Iteratively find N straight lines in 3D space.
    
    @param point_list: list of all points
    @param line_list: list of lines found previously
    @param distance_treshold: maximum distance between the line and the point to be takes as a part of the same line
    @param line_distance_const: constant that determines the influence of average point distance on the line quality
    @param gap_treshold: maximum gap between consecutive points allowed
    @param minimum_points: minimum points required to form a line
    @param point_ratio_treshold: ratio of how many points must be close to the line before considering searching for another line
    @return: list of found lines
    """
    
    # return None if too many lines 
    if len(line_list) >= max_lines:
        return None

    results_list = []
    for i, point1 in enumerate(point_list):
        for point2 in point_list[i:]:
            
            # Skip until the point is in another frame 
            if point2[2] <= point1[2]:
                continue

            x1, y1, z1 = point1
            x2, y2, z2 = point2

            # Include 2 points that make the line in the count
            counter = 2

            # Track average distance from the line
            line_dist_sum = 0

            x_prev, y_prev, z_prev = x1, y1, z1

            for point3 in point_list:

                if point1 == point3:
                    continue

                x3, y3, z3 = point3

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
            if (counter-2) < minimum_points:
                continue

            # Average distance between points and the line
            line_dist_avg = line_dist_sum / float(counter - 2)

            # calculate a parameter for line quality
            # larger average distance = less quality
            line_quality = counter - line_distance_const * line_dist_avg
            results_list.append((point1, point2, counter, line_quality))

    # Return empty if no good match was found
    if not results_list:
        return None

    # Line with the best quality    
    max_line = max(results_list, key=lambda x: x[3])

    point_num = len(point_list)
    
    # ratio of points inside and and all points
    line_ratio = max_line[2] / float(point_num)
    
    # remove points that belong to line with the best quality
    point_list, removed_points = remove3DPoints(point_list, max_line[0], max_line[1], distance_treshold, gap_treshold)
    
    # sort removed points by frame
    removed_points = sorted(removed_points, key=lambda x:x[2])
    # append line, first frame and last frame of meteor
    line_list.append([max_line, removed_points[0][2], removed_points[-1][2]])
    
    # if there are more lines on the image
    if line_ratio < point_ratio_treshold and point_num > 10:
        # Recursively find lines until the condition is met
        find3DLines(point_list, line_list, distance_treshold, line_distance_const, gap_treshold, minimum_points, point_ratio_treshold, max_lines)

    return line_list

def remove3DPoints(point_list, point1, point2, distance_treshold=2500, gap_treshold=10000):
    """ Remove points from a point list that belong to the given line described by point1 and point2.
    
    @param point_list: list of all points
    @param point1: first point describing a line
    @param point2: second point describing a line
    @param distance_treshold: maximum distance between the line and the point to be takes as a part of the same line
    @param gap_treshold: maximum gap between consecutive points allowed
    @return: tuple of list of points without those that belong to the given line and list of removed points
    """

    x1, y1, z1 = point1
    x2, y2, z2 = point2

    line_points = getAllPoints(point_list, y1, x1, z1, y2, x2, z2, distance_treshold)

    return [x for x in point_list if x not in line_points], line_points

def normalizeParameter(param, y_dim, x_dim):
    """ Normalize detection parameter to be size independent.
    
    @param param: parameter to be normalized
    @param y_dim: frame heigth
    @param x_dim: frame width
    @return: normalized param
    """

    return param**2 * y_dim * x_dim / (720*576)


def findCoefficients(event_points, line_list):
    """ Extract coefficients from list of lines that can be consumed by RMS.VideoExtractor
    
    @param event_points: list of all points
    @param line_list: list of detected lines
    @return: coefficients for each detected line in format: [first point, slope of XZ, slope of YZ, first frame, last frame]
    """
    
    coeff = []
    
    for i, detected_line in enumerate(line_list):
        point1 = np.array(detected_line[0][0])
        point2 = np.array(detected_line[0][1])        
        
        # difference between last point and first point that represent a line
        point3 = point2 - point1
        
        # slope
        slope1 = point3[1]/point3[2] # speed on Y axis
        slope2 = point3[0]/point3[2] # speed on X axis
        
        coeff.append([point1, slope1, slope2, detected_line[1], detected_line[2]]) #first point, frame of first point, slope of XZ, slope of YZ, first frame, last frame
        
    return coeff