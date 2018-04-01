""" A set of functions for calculating the similarity of 2 lines using the Frechet distance.
See the compareLines function for more details about the usage.
"""

from __future__ import division, absolute_import, print_function

import numpy as np
import math


def frechetDist(P,Q):
    """ Calculates the Frechet distance between 2 polygonal lines.

    Source: http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf

    @param P: ndarray; (x, y) pair of coordinates which make up the first line
    @param Q: ndarray; (x, y) pair of coordinates which make up the secondline

    @return: Frechet distance of the given lines
    """

    def eucDist(pt1, pt2):
        """ Eucledian distance between 2 points in 2D space.
        """

        return math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)

    def calcFrechet(ca, i, j, P, Q):
        if ca[i, j] > -1:
            return ca[i,j]
        elif i == 0 and j == 0:
            ca[i, j] = eucDist(P[0], Q[0])
        elif i > 0 and j == 0:
            ca[i, j] = max(calcFrechet(ca, i-1, 0, P, Q), eucDist(P[i], Q[0]))
        elif i == 0 and j > 0:
            ca[i, j] = max(calcFrechet(ca, 0, j-1, P, Q), eucDist(P[0], Q[j]))
        elif i > 0 and j > 0:
            ca[i, j] = max(min(calcFrechet(ca, i-1, j, P, Q), calcFrechet(ca, i-1, j-1, P, Q), 
                calcFrechet(ca, i, j-1, P, Q)), eucDist(P[i], Q[j]))
        else:
            ca[i, j] = float("inf")
        return ca[i, j]

    ca = -np.ones((len(P), len(Q)))

    return calcFrechet(ca, len(P)-1, len(Q)-1, P, Q)


def samplePolarLine(rho, theta, img_h, img_w, nsamples=10):
    """ Samples the line given in polar coordinates and returns Caretsian coordinates of the samples.

    @param rho: distance parameter of the line in polar coordinates (px)
    @param theta: angle parameter of the line in polar coordinates (degrees)
    @param img_h: image height (px)
    @param img_w: image width (px)
    @param nsamples: number of samples to be taken on the line.

    @return: 2D numpy array containing (x, y) coordinates of each sample
    """

    # Convert theta to radians
    theta = np.deg2rad(theta)

    # Image half widths and heights
    hh = img_h/2.0
    hw = img_w/2.0

    # Convert line to Cartesian coordinates
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho

    # Sample points on the line
    samples = np.linspace(0, img_h, nsamples) - hh

    # Calculate the line array
    x_array = x0 + samples*(-b) + hw
    y_array = y0 + samples*(a) + hh

    return np.column_stack((x_array, y_array))


def compareLines(rho1, theta1, rho2, theta2, img_h, img_w):
    """ Calculates the similarity between 2 straight lines.

    The similarity is defined as a Frechet distance between 10 sampled points on those lines. The sampled
    points are inside the image size which originally contained that line.

    @param rho1: distance parameter of the first line in polar coordinates (px)
    @param theta1: angle parameter of the first line in polar coordinates (degrees)
    @param rho2: distance parameter of the second line in polar coordinates (px)
    @param theta2: angle parameter of the second line in polar coordinates (degrees)
    @param img_h: image height (px)
    @param img_w: image width (px)

    @return: Frechet distance of the given lines.
    """

    # Make sample points of each given line
    P = samplePolarLine(rho1, theta1, img_h, img_w)
    Q = samplePolarLine(rho2, theta2, img_h, img_w)

    # As the order of the points is important during Frechet distance calculation, flip one of the lines
    # and take the minimum Frechet distance between the normal and the flipped solutions
    return min(frechetDist(P, Q), frechetDist(np.flipud(P), Q))




if __name__ == '__main__':

    # Image dimensions
    img_w = 720
    img_h = 576

    # Lines in polar coordinates
    line_list = [
    [-11.62494509512873, 23.600000000000065, 160, 224], 
    [-11.824945095128729, 23.700000000000067, 160, 224], 
    [-12.024945095128729, 23.800000000000068, 160, 224], 
    [-14.024945095128722, 22.900000000000055, 192, 256], 
    [-17.924945095128741, 26.100000000000101, 192, 256], 
    [-21.924945095128798, 6.9999999999999911, 192, 256], 
    [-22.324945095128804, 6.2999999999999936, 192, 256],
    [-100.0,              90, 0, 0],
    [-100.0,              85+180, 0, 0]]

    # Test lines in polar coordinates
    rho1, theta1 = line_list[0][0:2]
    rho2, theta2 = line_list[5][0:2]

    # test line comparison
    print(compareLines(rho1, theta1, rho2, theta2, img_h, img_w))