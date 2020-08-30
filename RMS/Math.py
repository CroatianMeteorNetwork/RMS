""" Common math functions. """

import numpy as np
from numpy.core.umath_tests import inner1d


def angularSeparation(ra1, dec1, ra2, dec2):
    """ Calculates the angle between two points on a sphere. 
    
    Arguments:
        ra1: [float] Right ascension 1 (radians).
        dec1: [float] Declination 1 (radians).
        ra2: [float] Right ascension 2 (radians).
        dec2: [float] Declination 2 (radians).

    Return:
        [float] Angle between two coordinates (radians).
    """

    return np.arccos(np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra2 - ra1))


def angularSeparationVect(vect1, vect2):
    """ Calculates angle between vectors in radians. """

    return np.abs(np.arccos(np.dot(vect1, vect2)))



### VECTORS ###
##############################################################################################################


def vectNorm(vect):
    """ Convert a given vector to a unit vector. """

    return vect/vectMag(vect)



def vectMag(vect):
    """ Calculate the magnitude of the given vector. """

    return np.sqrt(inner1d(vect, vect))


##############################################################################################################


def rotatePoint(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.

    Source: http://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python

    Arguments:
        origin: [tuple of floats] (x, y) pair of Cartesian coordinates of the origin
        point: [tuple of floats] (x, y) pair of Cartesian coordinates of the point
        angle: [float] angle of rotation in radians

    Return:
        (qx, qy): [tuple of floats] Cartesian coordinates of the rotated point
    """

    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle)*(px - ox) - np.sin(angle)*(py - oy)
    qy = oy + np.sin(angle)*(px - ox) + np.cos(angle)*(py - oy)

    return qx, qy


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


def isAngleBetween(left, ang, right):
    """ Checks if ang is between the angle on the left anf right. 
    
    Arguments:
        left: [float] Left (counter-clockwise) angle (radians).
        ang: [float] Angle to check (radians),
        right: [float] Right (clockwise) angle (radiant).

    Return:
        [bool] True if the angle is in between, false otherwise.
    """

    if right - left < 0:
        right = right - left + 2*np.pi
    else:
        right = right - left


    if ang - left < 0:
        ang = ang - left + 2*np.pi
    else:
        ang = ang - left


    return ang < right



@np.vectorize
def sphericalPointFromHeadingAndDistance(ra1, dec1, heading, distance):
    """ Given RA and Dec, a heading and angular distance, compute coordinates of the point.

    Arguments:
        ra1: [float] Right Ascension (deg).
        dec1: [float] Declination (deg).
        heading: [float] Heading +E of due N in degrees (deg).
        distance: [float] Distance (deg).

    Return:
        ra, dec: [float] Coordinates of the new point (deg)

    """

    ra1 = np.radians(ra1)
    dec1 = np.radians(dec1)
    heading = np.radians(heading)
    distance = np.radians(distance)

    # Compute the new declination
    dec = np.arcsin(np.sin(dec1)*np.cos(distance) + np.cos(dec1)*np.sin(distance)*np.cos(heading))

    # Handle poles and compute right ascension
    if np.cos(dec) == 0:
       ra = ra1

    else:
       ra = (ra1 - np.arcsin(np.sin(heading)*np.sin(distance)/np.cos(dec)) + np.pi)% (2*np.pi) - np.pi


    return np.degrees(ra)%360, np.degrees(dec)
    