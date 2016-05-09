import numpy as np
import cv2

# Cython import
cimport numpy as np
cimport cython

# Define numpy types
FLOAT_TYPE = np.float64
ctypedef np.float64_t FLOAT_TYPE_t

# Define Pi
cdef double pi = np.pi

# Declare trigonometric functions
cdef extern from "math.h":
    double sin(double)
    double asin(double)
    double cos(double)
    double acos(double)
    double atan2(double, double)


@cython.cdivision(True)
cdef double radians(double deg):
    """Converts degrees to radians. """
    
    return deg/180.0*(pi)

@cython.cdivision(True)
cdef double degrees(double deg):
    """Converts radians to degrees. """
    
    return deg*180.0/pi

@cython.boundscheck(False)
@cython.wraparound(False) 
cpdef double angularSeparation(double ra1, double dec1, double ra2, double dec2):
    """ Calculate the angular separation between 2 stars in equatorial celestial coordinates. 

    Source of the equation: http://www.astronomycafe.net/qadir/q1890.html (May 1, 2016)

    @param ra1: [float] right ascension of the first stars (in degrees)
    @param dec1: [float] decliantion of the first star (in degrees)
    @param ra2: [float] right ascension of the decons stars (in degrees)
    @param dec2: [float] decliantion of the decons star (in degrees)

    @return angular_separation: [float] angular separation (in degrees)
    """

    # Convert input coordinates to radians
    ra1 = radians(ra1)
    dec1 =  radians(dec1)
    ra2 = radians(ra2)
    dec2 = radians(dec2)

    return degrees(acos(sin(dec1)*sin(dec2) + cos(dec1)*cos(dec2)*cos(ra2 - ra1)))


cpdef double calcBearing(double ra1, double dec1, double ra2, double dec2):
    """ Calculate the bearing angle between 2 stars in equatorial celestial coordinates. """

    # Convert input coordinates to radians
    ra1 = radians(ra1)
    dec1 =  radians(dec1)
    ra2 = radians(ra2)
    dec2 = radians(dec2)

    return degrees(atan2(sin(ra2 - ra1)*cos(dec2), cos(dec1)*sin(dec2) - sin(dec1)*cos(dec2)*cos(ra2 - ra1))) % 360


@cython.boundscheck(False)
@cython.wraparound(False) 
def subsetCatalog(np.ndarray[FLOAT_TYPE_t, ndim=2] catalog_list, double ra_c, double dec_c, double radius, double mag_limit):
    """ Make a subset of stras from the given star catalog around the given coordinates with a given radius. """


    # Define variables
    cdef int i, k
    cdef double dec_min, dec_max
    cdef double ra, dec, mag
    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] filtered_list = np.zeros(shape=(catalog_list.shape[0], catalog_list.shape[1]), dtype=FLOAT_TYPE)

    # Calculate minimum and maximum declination
    dec_min = dec_c - radius
    if dec_min < -90:
        dec_min = -90

    dec_max = dec_c + radius
    if dec_max > 90:
        dec_max = 90

    k = 0
    for i in range(catalog_list.shape[0]):

        ra = catalog_list[i,0]
        dec = catalog_list[i,1]
        mag = catalog_list[i,2]

        # Skip if the declination is too large
        if dec > dec_max:
            continue

        # End the loop if the declination is too small
        if dec < dec_min:
            break

        # Add star to the list if it is within a given radius and has a certain brightness
        if (angularSeparation(ra, dec, ra_c, dec_c) <= radius) and (mag <= mag_limit):
            
            filtered_list[k,0] = ra
            filtered_list[k,1] = dec
            filtered_list[k,2] = mag

            # Increment filtered list counter
            k += 1

    return filtered_list[:k]


@cython.boundscheck(False)
@cython.wraparound(False)
def starsNNevaluation(np.ndarray[FLOAT_TYPE_t, ndim=2] stars, np.ndarray[FLOAT_TYPE_t, ndim=2] ref_stars, double consideration_radius, int min_matched_stars, int ret_indices=0):
    """ Finds nearest neighbours between the catalog stars and the calibration stars and evaluate their matching. """

    # Get the size of each point set
    cdef int stars_len = stars.shape[0]
    cdef int ref_stars_len = ref_stars.shape[0]

    # Init evaluation parameter
    cdef double evaluation = 0

    # Define difference vector's magnitude and directions
    cdef np.ndarray[np.uint16_t, ndim=1] vect_idx = np.zeros(shape=(stars_len), dtype=np.uint16)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] vect_separation = np.zeros(shape=(stars_len), dtype=FLOAT_TYPE)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] vect_bearing = np.zeros(shape=(stars_len), dtype=FLOAT_TYPE)
    cdef int k = 0
    cdef int i, j, min_idx
    cdef double min_dist, ang_sep

    for i in range(stars_len):

        min_dist = consideration_radius
        min_idx = 0

        for j in range(ref_stars_len):

            # Calculate the angular separation between the stars
            ang_sep = angularSeparation(stars[i, 0], stars[i, 1], ref_stars[j, 0], ref_stars[j, 1])

            if ang_sep <= min_dist:
                min_dist = ang_sep
                min_idx = j

        # Add to the evaluation if the neighbour is close enough
        if min_dist < consideration_radius:
            vect_idx[k] = min_idx
            vect_separation[k] = min_dist
            vect_bearing[k] = calcBearing(stars[i, 0], stars[i, 1], ref_stars[min_idx, 0], ref_stars[min_idx, 1])
            k += 1

    # Check if there is a minimum number of matched stars
    if k < min_matched_stars:
        
        if ret_indices:
            return None
        else:
            return (None, None, None, None, None)
    
    # Crop the vectors to their real size and convert to radians
    vect_separation = np.radians(vect_separation[:k])
    vect_bearing = np.radians(vect_bearing[:k])

    # Calculate the mean of the given vector
    # Calculate Ra and Dec components from the angular separation and bearing
    ra_diff = np.degrees(np.arctan2(np.sin(vect_bearing)*np.sin(vect_separation), np.cos(vect_separation)))
    dec_diff = np.degrees(np.arcsin(np.sin(vect_separation)*np.cos(vect_bearing)))

    ra_mean = np.mean(ra_diff)
    ra_std = np.std(ra_diff)
    dec_mean = np.mean(dec_diff)
    dec_std = np.std(dec_diff)

    # Evaluate the solution (smaller STDDEV the better, more stars the better) -> smaller evaluation is better
    evaluation = (ra_std + dec_std)/k

    # If ret_indices is 1, then only return indices of the matched stars
    if ret_indices:
        return vect_idx[:k]

    else:
        return evaluation, ra_mean, ra_std, dec_mean, dec_std