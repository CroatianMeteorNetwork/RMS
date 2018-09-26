""" Common math functins. """

import numpy as np

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