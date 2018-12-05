""" Common math functins. """

import numpy as np
from numpy.core.umath_tests import inner1d


def angularSeparation(ra1, dec1, ra2, dec2):
    """ Calculates the angle between two points on a sphere. 
    
    Arguments:
        dec1: [float] Declination 1 (radians).
        ra1: [float] Right ascension 1 (radians).
        dec2: [float] Declination 2 (radians).
        ra2: [float] Right ascension 2 (radians).

    Return:
        [float] Angle between two coordinates (radians).
    """

    return np.arccos(np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra2 - ra1))




### VECTORS ###
##############################################################################################################


def vectNorm(vect):
    """ Convert a given vector to a unit vector. """

    return vect/vectMag(vect)



def vectMag(vect):
    """ Calculate the magnitude of the given vector. """

    return np.sqrt(inner1d(vect, vect))


##############################################################################################################


def cartesianToPolar(x, y, z):
    """ Converts 3D cartesian coordinates to polar coordinates. 

    Arguments:
        x: [float] Px coordinate.
        y: [float] Py coordinate.
        z: [float] Pz coordinate.

    Return:
        (theta, phi): [float] Polar angles in radians (inclination, azimuth).

    """

    theta = np.arccos(z)
    phi = np.arctan2(y, x)

    return theta, phi



def polarToCartesian(theta, phi):
    """ Converts 3D spherical coordinates to 3D cartesian coordinates. 

    Arguments:
        theta: [float] Inclination in radians.
        phi: [float] Azimuth angle in radians.

    Return:
        (x, y, z): [tuple of floats] Coordinates of the point in 3D cartiesian coordinates.
    """


    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)

    return x, y, z

