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