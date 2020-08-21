#!python
#cython: language_level=3

import numpy as np
# import cv2

# Cython import
cimport numpy as np
cimport cython

# Import the Python bool type
from cpython cimport bool

# Define numpy types
INT_TYPE = np.uint32
ctypedef np.uint32_t INT_TYPE_t

FLOAT_TYPE = np.float64
ctypedef np.float64_t FLOAT_TYPE_t

ctypedef np.uint8_t BOOL_TYPE_t

# Define Pi
cdef double pi = np.pi

# Define the Julian date at the J2000 epoch
cdef double J2000_DAYS = 2451545.0

# Declare math functions
cdef extern from "math.h":
    double fabs(double)
    double sin(double)
    double asin(double)
    double cos(double)
    double acos(double)
    double tan(double)
    double atan2(double, double)
    double sqrt(double)


@cython.cdivision(True)
cdef double radians(double deg):
    """Converts degrees to radians.
    """

    return deg/180.0*(pi)

  
@cython.cdivision(True)
cdef double degrees(double deg):
    """Converts radians to degrees.
    """

    return deg*180.0/pi


cdef double sign(double x):    
    if (x >= 1):
        return 1.0

    return -1.0


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


@cython.boundscheck(False)
@cython.wraparound(False)
def angularSeparation_fv(double ra1, double dec1,
                             np.ndarray[FLOAT_TYPE_t, ndim=1] ra2, np.ndarray[FLOAT_TYPE_t, ndim=1] dec2):
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
    ra2 = np.radians(ra2)
    dec2 = np.radians(dec2)

    return np.degrees(np.arccos(sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra2 - ra1)))


@cython.boundscheck(False)
@cython.wraparound(False)
def subsetCatalog(np.ndarray[FLOAT_TYPE_t, ndim=2] catalog_list, double ra_c, double dec_c, double jd,
        double lat, double lon, double radius, double mag_limit):
    """ Make a subset of stars from the given star catalog around the given coordinates with a given radius.

    Arguments:
        catalog_list: [ndarray] An array of (ra, dec, mag) pairs for stars (J2000, degrees).
        ra_c: [float] Centre of extraction RA (degrees).
        dec_c: [float] Centre of extraction dec (degrees).
        jd: [float] Julian date of observations.
        lat: [float] Observer latitude (deg).
        lon: [float] Observer longitude (deg).
        radius: [float] Extraction radius (degrees).
        mag_limit: [float] Limiting magnitude.

    Return:
        filtered_indices, filtered_list: (ndarray, ndarray)
            - filtered_indices - Indices of catalog_list entries which satifly the filters.
            - filtered_list - catalog_list entires that satifly the filters.
        ...
    """


    # Define variables
    cdef int i, k
    cdef double dec_min, dec_max
    cdef double ra, dec, mag, elev
    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] filtered_list = np.zeros(shape=(catalog_list.shape[0], \
        catalog_list.shape[1]), dtype=FLOAT_TYPE)

    cdef np.ndarray[INT_TYPE_t, ndim=1] filtered_indices = np.zeros(shape=(catalog_list.shape[0]), \
        dtype=INT_TYPE)

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

            # Compute the local star elevation
            _, elev = cyraDec2AltAz(radians(ra), radians(dec), jd, radians(lat), radians(lon))


            # Only take stars above -20 degrees
            if degrees(elev) > -20:

                filtered_list[k,0] = ra
                filtered_list[k,1] = dec
                filtered_list[k,2] = mag

                # Add index to the list of indices which passed the filter
                filtered_indices[k] = i;

                # Increment filtered list counter
                k += 1


    return filtered_indices[:k], filtered_list[:k]



@cython.boundscheck(False)
@cython.wraparound(False)
def matchStars(np.ndarray[FLOAT_TYPE_t, ndim=2] stars_list, np.ndarray[FLOAT_TYPE_t, ndim=1] cat_x_array, \
    np.ndarray[FLOAT_TYPE_t, ndim=1] cat_y_array, np.ndarray[INT_TYPE_t, ndim=1] cat_good_indices, \
    double max_radius):


    cdef int i, j
    cdef unsigned int cat_idx
    cdef int k = 0
    cdef double min_dist, dist
    cdef double cat_match_indx, im_star_y, im_star_x, cat_x, cat_y

    # Get the lenghts of input arrays
    cdef int stars_len = stars_list.shape[0]
    cdef int cat_len = cat_good_indices.shape[0]

    # List for matched indices
    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] matched_indices = np.zeros(shape=(stars_list.shape[0], 3), \
        dtype=FLOAT_TYPE)


    ### Match image and catalog stars ###

    # Go through all image stars
    for i in range(stars_len):

        # Extract image star coordinates
        im_star_y = stars_list[i, 0]
        im_star_x = stars_list[i, 1]

        min_dist = max_radius
        cat_match_indx = -1

        # Check for the best match among catalog stars
        for j in range(cat_len):

            cat_idx = cat_good_indices[j]

            # Extract catalog coordinates
            cat_x = cat_x_array[cat_idx]
            cat_y = cat_y_array[cat_idx]


            # Calculate the distance between stars
            dist = sqrt((im_star_x - cat_x)**2 + (im_star_y - cat_y)**2)

            # Set the catalog star as the best match if it is the closest to the image star than any previous
            if (dist < min_dist):
                min_dist = dist
                cat_match_indx = cat_idx


        # Take the best matched star if the distance was within the maximum radius
        if min_dist < max_radius:

            # Add the matched indices to the output list
            matched_indices[k, 0] = i
            matched_indices[k, 1] = cat_match_indx
            matched_indices[k, 2] = min_dist

            k += 1



    # Cut the output list to the number of matched stars
    matched_indices = matched_indices[:k]

    return matched_indices



@cython.cdivision(True)
cdef double cyjd2LST(double jd, double lon):
    """ Convert Julian date to apparent Local Sidereal Time. The times is apparent, not mean!
    Source: J. Meeus: Astronomical Algorithms
    Arguments:
        jd: [float] Decimal julian date, epoch J2000.0.
        lon: [float] Longitude of the observer in degrees.
    
    Return:
        lst [float] Apparent Local Sidereal Time (deg).
    """

    cdef double gst

    cdef double t = (jd - J2000_DAYS)/36525.0

    # Calculate the Mean sidereal rotation of the Earth in radians (Greenwich Sidereal Time)
    gst = 280.46061837 + 360.98564736629*(jd - J2000_DAYS) + 0.000387933*t**2 - (t**3)/38710000.0
    gst = (gst + 360)%360


    # Compute the apparent Local Sidereal Time (LST)
    return (gst + lon + 360)%360



@cython.cdivision(True)
cpdef (double, double) equatorialCoordPrecession(double start_epoch, double final_epoch, double ra, \
    double dec):
    """ Corrects Right Ascension and Declination from one epoch to another, taking only precession into 
        account.

        Implemented from: Jean Meeus - Astronomical Algorithms, 2nd edition, pages 134-135
    
    Arguments:
        start_epoch: [float] Julian date of the starting epoch.
        final_epoch: [float] Julian date of the final epoch.
        ra: [float] Input right ascension (radians).
        dec: [float] Input declination (radians).
    
    Return:
        (ra, dec): [tuple of floats] Precessed equatorial coordinates (radians).
    """

    cdef double T, t, zeta, z, theta, A, B, C, ra_corr, dec_corr


    T = (start_epoch - J2000_DAYS )/36525.0
    t = (final_epoch - start_epoch)/36525.0

    # Calculate correction parameters in degrees
    zeta  = ((2306.2181 + 1.39656*T - 0.000139*T**2)*t + (0.30188 - 0.000344*T)*t**2 + 0.017998*t**3)/3600
    z     = ((2306.2181 + 1.39656*T - 0.000139*T**2)*t + (1.09468 + 0.000066*T)*t**2 + 0.018203*t**3)/3600
    theta = ((2004.3109 - 0.85330*T - 0.000217*T**2)*t - (0.42665 + 0.000217*T)*t**2 - 0.041833*t**3)/3600

    # Convert parameters to radians
    zeta  = radians(zeta)
    z     = radians(z)
    theta = radians(theta)

    # Calculate the next set of parameters
    A = cos(dec  )*sin(ra + zeta)
    B = cos(theta)*cos(dec)*cos(ra + zeta) - sin(theta)*sin(dec)
    C = sin(theta)*cos(dec)*cos(ra + zeta) + cos(theta)*sin(dec)

    # Calculate right ascension
    ra_corr = (atan2(A, B) + z + 2*pi)%(2*pi)

    # Calculate declination (apply a different equation if close to the pole, closer then 0.5 degrees)
    if (pi/2 - fabs(dec)) < radians(0.5):
        dec_corr = np.sign(dec)*acos(sqrt(A**2 + B**2))
    else:
        dec_corr = asin(C)


    return ra_corr, dec_corr

@cython.cdivision(True)
def equatorialCoordPrecession_vect(double start_epoch, double final_epoch, np.ndarray[FLOAT_TYPE_t, ndim=1] ra,
    np.ndarray[FLOAT_TYPE_t, ndim=1] dec):
    """ Corrects Right Ascension and Declination from one epoch to another, taking only precession into 
        account.
        Implemented from: Jean Meeus - Astronomical Algorithms, 2nd edition, pages 134-135
    
    Arguments:
        start_epoch: [float] Julian date of the starting epoch.
        final_epoch: [float] Julian date of the final epoch.
        ra: [float] Input right ascension (radians).
        dec: [float] Input declination (radians).
    
    Return:
        (ra, dec): [tuple of floats] Precessed equatorial coordinates (radians).
    """

    cdef double T, t, zeta, z, theta
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] A, B, C, ra_corr, dec_corr
    cdef np.ndarray[BOOL_TYPE_t, ndim=1] filter

    T = (start_epoch - J2000_DAYS )/36525.0
    t = (final_epoch - start_epoch)/36525.0

    # Calculate correction parameters in degrees
    zeta  = ((2306.2181 + 1.39656*T - 0.000139*T**2)*t + (0.30188 - 0.000344*T)*t**2 + 0.017998*t**3)/3600
    z     = ((2306.2181 + 1.39656*T - 0.000139*T**2)*t + (1.09468 + 0.000066*T)*t**2 + 0.018203*t**3)/3600
    theta = ((2004.3109 - 0.85330*T - 0.000217*T**2)*t - (0.42665 + 0.000217*T)*t**2 - 0.041833*t**3)/3600

    # Convert parameters to radians
    zeta  = radians(zeta)
    z     = radians(z)
    theta = radians(theta)

    # Calculate the next set of parameters
    A = np.cos(dec  )*np.sin(ra + zeta)
    B = cos(theta)*np.cos(dec)*np.cos(ra + zeta) - sin(theta)*np.sin(dec)
    C = sin(theta)*np.cos(dec)*np.cos(ra + zeta) + cos(theta)*np.sin(dec)

    # Calculate right ascension
    ra_corr = (np.arctan2(A, B) + z + 2*pi)%(2*pi)

    # Calculate declination (apply a different equation if close to the pole, closer then 0.5 degrees)
    dec_corr = np.arcsin(C)

    filter = (pi/2 - np.abs(dec)) < radians(0.5)
    dec_corr[filter] = np.sign(dec[filter])*np.arccos(np.sqrt(A[filter]**2 + B[filter]**2))

    return ra_corr, dec_corr


@cython.cdivision(True)
cdef double refractionApparentToTrue(double elev):
    """ Correct the apparent elevation of a star for refraction to true elevation. The temperature and air
        pressure are assumed to be unknown. 
        Source: Explanatory Supplement to the Astronomical Almanac (1992), p. 144.
    Arguments:
        elev: [float] Apparent elevation (radians).
    Return:
        [float] True elevation (radians).
    """

    cdef double refraction

    # Don't apply refraction for elevation below -0.5 deg
    if elev > radians(-0.5):

        # Refraction in radians
        refraction = radians(1.0/(60*tan(radians(degrees(elev) + 7.31/(degrees(elev) + 4.4)))))

    else:
        refraction = 0.0

    # Correct the elevation
    return elev - refraction


@cython.cdivision(True)
cpdef np.ndarray[FLOAT_TYPE_t, ndim=1] refractionApparentToTrue_vect(np.ndarray[FLOAT_TYPE_t, ndim=1] elev):
    """ Correct the apparent elevation of a star for refraction to true elevation. The temperature and air
        pressure are assumed to be unknown. 
        Source: Explanatory Supplement to the Astronomical Almanac (1992), p. 144.
    Arguments:
        elev: [float] Apparent elevation (radians).
    Return:
        [float] True elevation (radians).
    """

    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] refraction
    cdef np.ndarray[BOOL_TYPE_t, ndim=1] filter

    # Don't apply refraction for elevation below -0.5 deg
    filter = elev > radians(-0.5)

    # Refraction in radians
    refraction = np.zeros_like(elev)
    refraction[filter] = np.radians(1.0/(60*np.tan(np.radians(np.degrees(elev[filter]) + 7.31/(np.degrees(elev[filter]) + 4.4)))))

    # Apply the refraction
    return elev + refraction

cpdef (double, double) eqRefractionApparentToTrue(double ra, double dec, double jd, double lat, double lon):
    """ Correct the equatorial coordinates for refraction. The correction is done from apparent to true
        coordinates.
    
    Arguments:
        ra: [float] J2000 right ascension in radians.
        dec: [float] J2000 declination in radians.
        jd: [float] Julian date.
        lat: [float] latitude in radians.
        lon: [float] longitude in radians.
    Return:
        (ra, dec):
            - ra: [float] Refraction corrected (true) right ascension in radians.
            - dec: [float] Refraction corrected (true) declination in radians.
    """

    cdef double azim, alt

    # Precess RA/Dec from J2000 to the epoch of date
    ra, dec = equatorialCoordPrecession(J2000_DAYS, jd, ra, dec)

    # Convert coordinates to alt/az
    azim, alt = cyraDec2AltAz(ra, dec, jd, lat, lon)

    # Correct the elevation
    alt = refractionApparentToTrue(alt)

    # Convert back to equatorial
    ra, dec = cyaltAz2RADec(azim, alt, jd, lat, lon)

    # Precess RA/Dec from the epoch of date to J2000
    ra, dec = equatorialCoordPrecession(jd, J2000_DAYS, ra, dec)


    return (ra, dec)



@cython.cdivision(True)
cdef double refractionTrueToApparent(double elev):
    """ Correct the true elevation of a star for refraction to apparent elevation. The temperature and air
        pressure are assumed to be unknown. 
        Source: https://en.wikipedia.org/wiki/Atmospheric_refraction
    Arguments:
        elev: [float] Apparent elevation (radians).
    Return:
        [float] True elevation (radians).
    """

    cdef double refraction

    # Don't apply refraction for elevation below -0.5 deg
    if elev > radians(-0.5):

        # Refraction in radians
        refraction = radians(1.02/(60*tan(radians(degrees(elev) + 10.3/(degrees(elev) + 5.11)))))

    else:
        refraction = 0.0

    # Apply the refraction
    return elev + refraction


@cython.cdivision(True)
cpdef np.ndarray[FLOAT_TYPE_t, ndim=1] refractionTrueToApparent_vect(np.ndarray[FLOAT_TYPE_t, ndim=1] elev):
    """ Correct the true elevation of a star for refraction to apparent elevation. The temperature and air
        pressure are assumed to be unknown. 
        Source: https://en.wikipedia.org/wiki/Atmospheric_refraction
    Arguments:
        elev: [float] Apparent elevation (radians).
    Return:
        [float] True elevation (radians).
    """

    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] refraction
    cdef np.ndarray[BOOL_TYPE_t, ndim=1] filter

    # Don't apply refraction for elevation below -0.5 deg
    filter = elev > radians(-0.5)

    # Refraction in radians
    refraction = np.zeros_like(elev)
    refraction[filter] = np.radians(1.02/(60*np.tan(np.radians(np.degrees(elev[filter]) + 10.3/(np.degrees(elev[filter]) + 5.11)))))

    # Apply the refraction
    return elev + refraction



cpdef (double, double) eqRefractionTrueToApparent(double ra, double dec, double jd, double lat, double lon):
    """ Correct the equatorial coordinates for refraction. The correction is done from true to apparent
        coordinates.
    
    Arguments:
        ra: [float] J2000 Right ascension in radians.
        dec: [float] J2000 Declination in radians.
        jd: [float] Julian date.
        lat: [float] Latitude in radians.
        lon: [float] Longitude in radians.
    Return:
        (ra, dec):
            - ra: [float] Apparent right ascension in radians.
            - dec: [float] Apparent declination in radians.
    """

    cdef double azim, alt

    # Precess RA/Dec from J2000 to the epoch of date
    ra, dec = equatorialCoordPrecession(J2000_DAYS, jd, ra, dec)

    # Convert coordinates to alt/az
    azim, alt = cyraDec2AltAz(ra, dec, jd, lat, lon)

    # Correct the elevation
    alt = refractionTrueToApparent(alt)

    # Convert back to equatorial
    ra, dec = cyaltAz2RADec(azim, alt, jd, lat, lon)

    # Precess RA/Dec from the epoch of date to J2000
    ra, dec = equatorialCoordPrecession(jd, J2000_DAYS, ra, dec)


    return (ra, dec)




@cython.cdivision(True)
cpdef (double, double) cyraDec2AltAz(double ra, double dec, double jd, double lat, double lon):
    """ Convert right ascension and declination to azimuth (+East of due North) and altitude. Same epoch is
        assumed, no correction for refraction is done.

    Arguments:
        ra: [float] Right ascension in radians.
        dec: [float] Declination in radians.
        jd: [float] Julian date.
        lat: [float] Latitude in radians.
        lon: [float] Longitude in radians.
    Return:
        (azim, elev): [tuple]
            azim: [float] Azimuth (+east of due north) in radians.
            elev: [float] Elevation above horizon in radians.
        """

    cdef double lst, ha, azim, sin_elev, elev

    # Calculate Local Sidereal Time
    lst = radians(cyjd2LST(jd, degrees(lon)))

    # Calculate the hour angle
    ha = lst - ra

    # Constrain the hour angle to [-pi, pi] range
    ha = (ha + pi)%(2*pi) - pi

    # Calculate the azimuth
    azim = pi + atan2(sin(ha), cos(ha)*sin(lat) - tan(dec)*cos(lat))

    # Calculate the sine of elevation
    sin_elev = sin(lat)*sin(dec) + cos(lat)*cos(dec)*cos(ha)

    # Wrap the sine of elevation in the [-1, +1] range
    sin_elev = (sin_elev + 1)%2 - 1

    elev = asin(sin_elev)

    return (azim, elev)


@cython.cdivision(True)
def cyraDec2AltAz_vect(np.ndarray[FLOAT_TYPE_t, ndim=1] ra, np.ndarray[FLOAT_TYPE_t, ndim=1] dec,
                       double jd, double lat, double lon):
    """ Convert right ascension and declination to azimuth (+East of due North) and altitude. Same epoch is
        assumed, no correction for refraction is done.
    Arguments:
        ra: [float] Right ascension in radians.
        dec: [float] Declination in radians.
        jd: [float] Julian date.
        lat: [float] Latitude in radians.
        lon: [float] Longitude in radians.
    Return:
        (azim, elev): [tuple]
            azim: [float] Azimuth (+east of due north) in radians.
            elev: [float] Elevation above horizon in radians.
        """

    cdef double lst
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] azim, elev, sin_elev, ha

    # Calculate Local Sidereal Time
    lst = radians(cyjd2LST(jd, degrees(lon)))

    # Calculate the hour angle
    ha = lst - ra

    # Constrain the hour angle to [-pi, pi] range
    ha = (ha + pi)%(2*pi) - pi

    # Calculate the azimuth
    azim = pi + np.arctan2(np.sin(ha), np.cos(ha)*sin(lat) - np.tan(dec)*cos(lat))

    # Calculate the sine of elevation
    sin_elev = sin(lat)*np.sin(dec) + cos(lat)*np.cos(dec)*np.cos(ha)

    # Wrap the sine of elevation in the [-1, +1] range
    sin_elev = (sin_elev + 1)%2 - 1

    elev = np.arcsin(sin_elev)

    return (azim, elev)


cpdef (double, double) trueRaDec2ApparentAltAz(double ra, double dec, double jd, double lat, double lon, \
    bool refraction=True):
    """ Convert the true right ascension and declination in J2000 to azimuth (+East of due North) and 
        altitude in the epoch of date. The correction for refraction is performed.
    Arguments:
        ra: [float] Right ascension in radians (J2000).
        dec: [float] Declination in radians (J2000).
        jd: [float] Julian date.
        lat: [float] Latitude in radians.
        lon: [float] Longitude in radians.
    Keyword arguments:
        refraction: [bool] Apply refraction correction. True by default.
    Return:
        (azim, elev): [tuple]
            azim: [float] Azimuth (+east of due north) in radians (epoch of date).
            elev: [float] Elevation above horizon in radians (epoch of date).
        """

    cdef double azim, elev

    # Precess RA/Dec to the epoch of date
    ra, dec = equatorialCoordPrecession(J2000_DAYS, jd, ra, dec)

    # Convert to alt/az
    azim, elev = cyraDec2AltAz(ra, dec, jd, lat, lon)

    # Correct elevation for refraction
    if refraction:
        elev = refractionTrueToApparent(elev)


    return (azim, elev)



cpdef trueRaDec2ApparentAltAz_vect(np.ndarray[FLOAT_TYPE_t, ndim=1] ra, np.ndarray[FLOAT_TYPE_t, ndim=1] dec,
    double jd, double lat, double lon, bool refraction=True):
    """ Convert the true right ascension and declination in J2000 to azimuth (+East of due North) and 
        altitude in the epoch of date. The correction for refraction is performed.
    Arguments:
        ra: [float] Right ascension in radians (J2000).
        dec: [float] Declination in radians (J2000).
        jd: [float] Julian date.
        lat: [float] Latitude in radians.
        lon: [float] Longitude in radians.
    Keyword arguments:
        refraction: [bool] Apply refraction correction. True by default.
    Return:
        (azim, elev): [tuple]
            azim: [float] Azimuth (+east of due north) in radians (epoch of date).
            elev: [float] Elevation above horizon in radians (epoch of date).
        """

    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] azim, elev

    # Precess RA/Dec to the epoch of date
    ra, dec = equatorialCoordPrecession_vect(J2000_DAYS, jd, ra, dec)

    # Convert to alt/az
    azim, elev = cyraDec2AltAz_vect(ra, dec, jd, lat, lon)

    # Correct elevation for refraction
    if refraction:
        elev = refractionTrueToApparent_vect(elev)


    return (azim, elev)




cpdef (double, double) trueRaDec2ApparentAltAz(double ra, double dec, double jd, double lat, double lon, \
    bool refraction=True):
    """ Convert the true right ascension and declination in J2000 to azimuth (+East of due North) and 
        altitude in the epoch of date. The correction for refraction is performed.

    Arguments:
        ra: [float] Right ascension in radians (J2000).
        dec: [float] Declination in radians (J2000).
        jd: [float] Julian date.
        lat: [float] Latitude in radians.
        lon: [float] Longitude in radians.

    Keyword arguments:
        refraction: [bool] Apply refraction correction. True by default.

    Return:
        (azim, elev): [tuple]
            azim: [float] Azimuth (+east of due north) in radians (epoch of date).
            elev: [float] Elevation above horizon in radians (epoch of date).

        """

    cdef double azim, elev


    # Precess RA/Dec to the epoch of date
    ra, dec = equatorialCoordPrecession(J2000_DAYS, jd, ra, dec)

    # Convert to alt/az
    azim, elev = cyraDec2AltAz(ra, dec, jd, lat, lon)

    # Correct elevation for refraction
    if refraction:
        elev = refractionTrueToApparent(elev)


    return (azim, elev)




@cython.cdivision(True)
cpdef (double, double) cyaltAz2RADec(double azim, double elev, double jd, double lat, double lon):
    """ Convert azimuth and altitude in a given time and position on Earth to right ascension and 
        declination. 
    Arguments:
        azim: [float] Azimuth (+east of due north) in radians.
        elev: [float] Elevation above horizon in radians.
        jd: [float] Julian date.
        lat: [float] Latitude of the observer in radians.
        lon: [float] Longitde of the observer in radians.
    Return:
        (RA, dec): [tuple]
            RA: [float] Right ascension (radians).
            dec: [float] Declination (radians).
    """


    cdef double lst, ha, ra, dec

    # Calculate Local Sidereal Time
    lst = radians(cyjd2LST(jd, degrees(lon)))

    # Calculate hour angle
    ha = atan2(-sin(azim), tan(elev)*cos(lat) - cos(azim)*sin(lat))

    # Calculate right ascension
    ra = (lst - ha + 2*pi)%(2*pi)

    # Calculate declination
    dec = asin(sin(lat)*sin(elev) + cos(lat)*cos(elev)*cos(azim))

    return (ra, dec)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cyaltAz2RADec_vect(np.ndarray[FLOAT_TYPE_t, ndim=1] azim, np.ndarray[FLOAT_TYPE_t, ndim=1] elev, double jd,
    double lat, double lon):
    """ Convert azimuth and altitude in a given time and position on Earth to right ascension and 
        declination. 
    Arguments:
        azim: [float] Azimuth (+east of due north) in radians.
        elev: [float] Elevation above horizon in radians.
        jd: [float] Julian date.
        lat: [float] Latitude of the observer in radians.
        lon: [float] Longitde of the observer in radians.
    Return:
        (RA, dec): [tuple]
            RA: [float] Right ascension (radians).
            dec: [float] Declination (radians).
    """



    cdef double lst
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] ha, ra, dec

    # Calculate Local Sidereal Time
    lst = radians(cyjd2LST(jd, degrees(lon)))

    # Calculate hour angle
    ha = np.arctan2(-np.sin(azim), np.tan(elev)*cos(lat) - np.cos(azim)*sin(lat))

    # Calculate right ascension
    ra = (lst - ha + 2*pi)%(2*pi)

    # Calculate declination
    dec = np.arcsin(sin(lat)*np.sin(elev) + cos(lat)*np.cos(elev)*np.cos(azim))

    return (ra, dec)


cpdef (double, double) apparentAltAz2TrueRADec(double azim, double elev, double jd, double lat, double lon, \
    bool refraction=True):
    """ Convert the apparent azimuth and altitude in the epoch of date to true (refraction corrected) right 
        ascension and declination in J2000.
    Arguments:
        azim: [float] Azimuth (+East of due North) in radians (epoch of date).
        elev: [float] Elevation above horizon in radians (epoch of date).
        jd: [float] Julian date.
        lat: [float] Latitude of the observer in radians.
        lon: [float] Longitde of the observer in radians.
    Keyword arguments:
        refraction: [bool] Apply refraction correction. True by default.
    Return:
        (ra, dec): [tuple]
            ra: [float] Right ascension (radians, J2000).
            dec: [float] Declination (radians, J2000).
    """


    cdef double ra, dec


    # Correct elevation for refraction
    if refraction:
        elev = refractionApparentToTrue(elev)

    # Convert to RA/Dec (true, epoch of date)
    ra, dec = cyaltAz2RADec(azim, elev, jd, lat, lon)

    # Precess RA/Dec to J2000
    ra, dec = equatorialCoordPrecession(jd, J2000_DAYS, ra, dec)


    return (ra, dec)



def apparentAltAz2TrueRADec_vect(np.ndarray[FLOAT_TYPE_t, ndim=1] azim, np.ndarray[FLOAT_TYPE_t, ndim=1] elev,
    double jd, double lat, double lon, bool refraction=True):
    """ Convert the apparent azimuth and altitude in the epoch of date to true (refraction corrected) right 
        ascension and declination in J2000.
    Arguments:
        azim: [float] Azimuth (+East of due North) in radians (epoch of date).
        elev: [float] Elevation above horizon in radians (epoch of date).
        jd: [float] Julian date.
        lat: [float] Latitude of the observer in radians.
        lon: [float] Longitde of the observer in radians.
    Keyword arguments:
        refraction: [bool] Apply refraction correction. True by default.
    Return:
        (ra, dec): [tuple]
            ra: [float] Right ascension (radians, J2000).
            dec: [float] Declination (radians, J2000).
    """
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] ra, dec


    # Correct elevation for refraction
    if refraction:
        elev = refractionApparentToTrue_vect(elev)

    # Convert to RA/Dec (true, epoch of date)
    ra, dec = cyaltAz2RADec_vect(azim, elev, jd, lat, lon)

    # Precess RA/Dec to J2000
    ra, dec = equatorialCoordPrecession_vect(jd, J2000_DAYS, ra, dec)


    return (ra, dec)


cpdef (double, double) apparentAltAz2TrueRADec(double azim, double elev, double jd, double lat, double lon, \
    bool refraction=True):
    """ Convert the apparent azimuth and altitude in the epoch of date to true (refraction corrected) right 
        ascension and declination in J2000.

    Arguments:
        azim: [float] Azimuth (+East of due North) in radians (epoch of date).
        elev: [float] Elevation above horizon in radians (epoch of date).
        jd: [float] Julian date.
        lat: [float] Latitude of the observer in radians.
        lon: [float] Longitde of the observer in radians.

    Keyword arguments:
        refraction: [bool] Apply refraction correction. True by default.

    Return:
        (ra, dec): [tuple]
            ra: [float] Right ascension (radians, J2000).
            dec: [float] Declination (radians, J2000).
    """


    cdef double ra, dec


    # Correct elevation for refraction
    if refraction:
        elev = refractionApparentToTrue(elev)

    # Convert to RA/Dec (true, epoch of date)
    ra, dec = cyaltAz2RADec(azim, elev, jd, lat, lon)

    # Precess RA/Dec to J2000
    ra, dec = equatorialCoordPrecession(jd, J2000_DAYS, ra, dec)


    return (ra, dec)




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cyraDecToXY(np.ndarray[FLOAT_TYPE_t, ndim=1] ra_data, \
    np.ndarray[FLOAT_TYPE_t, ndim=1] dec_data, double jd, double lat, double lon, double x_res, \
    double y_res, double h0, double ra_ref, double dec_ref, double pos_angle_ref, double pix_scale, \
    np.ndarray[FLOAT_TYPE_t, ndim=1] x_poly_rev, np.ndarray[FLOAT_TYPE_t, ndim=1] y_poly_rev, \
    str dist_type, bool refraction=True, bool equal_aspect=False, bool force_distortion_centre=False):

    """ Convert RA, Dec to distorion corrected image coordinates.

    Arguments:
        RA_data: [ndarray] Array of right ascensions (degrees).
        dec_data: [ndarray] Array of declinations (degrees).
        jd: [float] Julian date.
        lat: [float] Latitude of station in degrees.
        lon: [float] Longitude of station in degrees.
        x_res: [int] X resolution of the camera.
        y_res: [int] Y resolution of the camera.
        h0: [float] Reference hour angle (deg).
        ra_ref: [float] Reference right ascension of the image centre (degrees).
        dec_ref: [float] Reference declination of the image centre (degrees).
        pos_angle_ref: [float] Rotation from the celestial meridial (degrees).
        pix_scale: [float] Image scale (px/deg).
        x_poly_rev: [ndarray float] Distortion polynomial in X direction for reverse mapping.
        y_poly_rev: [ndarray float] Distortion polynomail in Y direction for reverse mapping.
        dist_type: [str] Distortion type. Can be: poly3+radial, radial3, radial4, or radial5.

    Keyword arguments:
        refraction: [bool] Apply refraction correction. True by default.
        equal_aspect: [bool] Force the X/Y aspect ratio to be equal. Used only for radial distortion. \
            False by default.
        force_distortion_centre: [bool] Force the distortion centre to the image centre. False by default.

    Return:
        (x, y): [tuple of ndarrays] Image X and Y coordinates.
    """

    cdef int i
    cdef double ra_centre, dec_centre, ra, dec
    cdef double radius, sin_ang, cos_ang, theta, x, y, r, dx, dy, x_img, y_img, r_corr, r_scale
    cdef double x0, y0, xy, k1, k2, k3, k4

    # Init output arrays
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] x_array = np.zeros_like(ra_data)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] y_array = np.zeros_like(ra_data)

    # Precalculate some parameters
    cdef double sl = sin(radians(lat))
    cdef double cl = cos(radians(lat))


    # Compute the current RA of the FOV centre by adding the difference in between the current and the
    #   reference hour angle
    ra_centre = radians((ra_ref + cyjd2LST(jd, 0) - h0 + 360)%360)
    dec_centre = radians(dec_ref)

    # Correct the reference FOV centre for refraction
    if refraction:
        ra_centre, dec_centre = eqRefractionTrueToApparent(ra_centre, dec_centre, jd, radians(lat), \
            radians(lon))


    # If the radial distortion is used, unpack radial parameters
    if dist_type.startswith("radial"):


        # Force the distortion centre to the image centre
        if force_distortion_centre:
            x0 = 0.5
            y0 = 0.5
        else:
            # Read distortion offsets
            x0 = x_poly_rev[0]
            y0 = x_poly_rev[1]


        # Aspect ratio
        if equal_aspect:
            xy = 0.0
        else:
            xy = x_poly_rev[2]


        # Distortion coeffs
        k1 = x_poly_rev[3]
        k2 = x_poly_rev[4]
        k3 = x_poly_rev[5]
        k4 = x_poly_rev[6]

    # If the polynomial distortion was used, unpack the offsets
    else:
        x0 = x_poly_rev[0]
        y0 = y_poly_rev[0]


    # Convert all equatorial coordinates to image coordinates
    for i in range(ra_data.shape[0]):

        ra = radians(ra_data[i])
        dec = radians(dec_data[i])

        ### Gnomonization of star coordinates to image coordinates ###

        # Apply refraction
        if refraction:
            ra, dec = eqRefractionTrueToApparent(ra, dec, jd, radians(lat), radians(lon))


        # Compute the distance from the FOV centre to the sky coordinate
        radius = radians(angularSeparation(degrees(ra), degrees(dec), degrees(ra_centre),
            degrees(dec_centre)))

        # Compute theta - the direction angle between the FOV centre, sky coordinate, and the image vertical
        sin_ang = cos(dec)*sin(ra - ra_centre)/sin(radius)
        cos_ang = (sin(dec) - sin(dec_centre)*cos(radius))/(cos(dec_centre)*sin(radius))
        theta   = -atan2(sin_ang, cos_ang) + radians(pos_angle_ref) - pi/2.0

        # Calculate the standard coordinates
        x = degrees(radius)*cos(theta)*pix_scale
        y = degrees(radius)*sin(theta)*pix_scale

        ### ###

        # Set initial distorsion values
        dx = 0
        dy = 0

        # Apply 3rd order polynomial + one radial term distortion
        if dist_type == "poly3+radial":

            # Compute the radius
            r = sqrt((x - x0)**2 + (y - y0)**2)

            # Calculate the distortion in X direction
            dx = (x0
                + x_poly_rev[1]*x
                + x_poly_rev[2]*y
                + x_poly_rev[3]*x**2
                + x_poly_rev[4]*x*y
                + x_poly_rev[5]*y**2
                + x_poly_rev[6]*x**3
                + x_poly_rev[7]*x**2*y
                + x_poly_rev[8]*x*y**2
                + x_poly_rev[9]*y**3
                + x_poly_rev[10]*x*r
                + x_poly_rev[11]*y*r)

            # Calculate the distortion in Y direction
            dy = (y0
                + y_poly_rev[1]*x
                + y_poly_rev[2]*y
                + y_poly_rev[3]*x**2
                + y_poly_rev[4]*x*y
                + y_poly_rev[5]*y**2
                + y_poly_rev[6]*x**3
                + y_poly_rev[7]*x**2*y
                + y_poly_rev[8]*x*y**2
                + y_poly_rev[9]*y**3
                + y_poly_rev[10]*y*r
                + y_poly_rev[11]*x*r)


        # Apply a radial distortion
        elif dist_type.startswith("radial"):

            # Compute the normalized radius to horizontal size
            r = sqrt(x**2 + y**2)/(x_res/2.0)
            r_corr = r

            # Apply the 3rd order radial distortion
            if dist_type == "radial3":

                # Compute the new radius
                r_corr = (1.0 - k1 - k2)*r + k1*r**2 - k2*r**3

            # Apply the 4th order radial distortion
            elif dist_type == "radial4":

                # Compute the new radius
                r_corr = (1.0 - k1 - k2 - k3)*r + k1*r**2 - k2*r**3 + k3*r**4


            # Apply the 5th order radial distortion
            elif dist_type == "radial5":

                # Compute the new radius
                r_corr = (1.0 - k1 - k2 - k3 - k4)*r + k1*r**2 - k2*r**3 + k3*r**4 - k4*r**5


            # Compute the scaling term
            if r == 0:
                r_scale = 0
            else:
                r_scale = (r_corr/r - 1)

            # Compute distortion offsets
            dx = (x - x0)*r_scale
            dy = (y - y0)*r_scale/(1.0 + xy)

        # Add the distortion
        x_img = x - dx
        y_img = y - dy

        # Calculate X image coordinates
        x_array[i] = x_img + x_res/2.0

        # Calculate Y image coordinates
        y_array[i] = y_img + y_res/2.0


    return x_array, y_array



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cyXYToRADec(np.ndarray[FLOAT_TYPE_t, ndim=1] jd_data, np.ndarray[FLOAT_TYPE_t, ndim=1] x_data, \
    np.ndarray[FLOAT_TYPE_t, ndim=1] y_data, double lat, double lon, double x_res, double y_res, \
    double h0, double ra_ref, double dec_ref, double pos_angle_ref, double pix_scale, \
    np.ndarray[FLOAT_TYPE_t, ndim=1] x_poly_fwd, np.ndarray[FLOAT_TYPE_t, ndim=1] y_poly_fwd, \
    str dist_type, bool refraction=True, bool equal_aspect=False, bool force_distortion_centre=False):
    """
    Arguments:
        jd_data: [ndarray] Julian date of each data point.
        x_data: [ndarray] 1D numpy array containing the image column.
        y_data: [ndarray] 1D numpy array containing the image row.
        lat: [float] Latitude of the observer in degrees.
        lon: [float] Longitde of the observer in degress.
        x_res: [int] Image size, X dimension (px).
        y_res: [int] Image size, Y dimenstion (px).
        h0: [float] Reference hour angle (deg).
        ra_ref: [float] Reference right ascension of the image centre (degrees).
        dec_ref: [float] Reference declination of the image centre (degrees).
        pos_angle_ref: [float] Field rotation parameter (degrees).
        pix_scale: [float] Plate scale (px/deg).
        x_poly_fwd: [ndarray] 1D numpy array of 12 elements containing forward X axis polynomial parameters.
        y_poly_fwd: [ndarray] 1D numpy array of 12 elements containing forward Y axis polynomial parameters.
        dist_type: [str] Distortion type. Can be: poly3+radial, radial3, radial4, or radial5.

    Keyword arguments:
        refraction: [bool] Apply refraction correction. True by default.
        equal_aspect: [bool] Force the X/Y aspect ratio to be equal. Used only for radial distortion. \
            False by default.
        force_distortion_centre: [bool] Force the distortion centre to the image centre. False by default.

    Return:
        (ra_data, dec_data): [tuple of ndarrays]

            ra_data: [ndarray] Right ascension of each point (deg).
            dec_data: [ndarray] Declination of each point (deg).
            magnitude_data: [ndarray] Array of meteor's lightcurve apparent magnitudes.
    """

    cdef int i
    cdef double jd, x_img, y_img, r, dx, x_corr, dy, y_corr, r_corr, r_scale
    cdef double x0, y0, xy, k1, k2, k3, k4
    cdef double radius, theta, sin_t, cos_t
    cdef double ha, ra_ref_now, ra_ref_now_corr, ra, dec, dec_ref_corr

    # Convert the reference pointing direction to radians
    ra_ref  = radians(ra_ref)
    dec_ref = radians(dec_ref)

    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] ra_data = np.zeros_like(jd_data)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] dec_data = np.zeros_like(jd_data)


    # If the radial distortion is used, unpack radial parameters
    if dist_type.startswith("radial"):

        # Force the distortion centre to the image centre
        if force_distortion_centre:
            x0 = 0.5
            y0 = 0.5
        else:
            # Read distortion offsets
            x0 = x_poly_fwd[0]
            y0 = x_poly_fwd[1]


        # Aspect ratio
        if equal_aspect:
            xy = 0.0
        else:
            # Read aspect ratio
            xy = x_poly_fwd[2]


        # Distortion coeffs
        k1 = x_poly_fwd[3]
        k2 = x_poly_fwd[4]
        k3 = x_poly_fwd[5]
        k4 = x_poly_fwd[6]


    # If the polynomial distortion was used, unpack the offsets
    else:
        x0 = x_poly_fwd[0]
        y0 = y_poly_fwd[0]


    # Go through all given data points and convert them from X, Y to RA, Dec
    for i in range(jd_data.shape[0]):

        # Choose time and image coordiantes
        jd = jd_data[i]
        x_img = x_data[i]
        y_img = y_data[i]


        ### APPLY DISTORTION CORRECTION ###

        # Normalize image coordinates to the image centre and compute the radius from image centre
        x_img = x_img - x_res/2.0
        y_img = y_img - y_res/2.0


        # Apply 3rd order polynomial + one radial term distortion
        if dist_type == "poly3+radial":

            # Compute the radius
            r = sqrt((x_img - x0)**2 + (y_img - y0)**2)

            # Compute offset in X direction
            dx = (x0
                + x_poly_fwd[1]*x_img
                + x_poly_fwd[2]*y_img
                + x_poly_fwd[3]*x_img**2
                + x_poly_fwd[4]*x_img*y_img
                + x_poly_fwd[5]*y_img**2
                + x_poly_fwd[6]*x_img**3
                + x_poly_fwd[7]*x_img**2*y_img
                + x_poly_fwd[8]*x_img*y_img**2
                + x_poly_fwd[9]*y_img**3
                + x_poly_fwd[10]*x_img*r
                + x_poly_fwd[11]*y_img*r)

            # Compute offset in Y direction
            dy = (y0
                + y_poly_fwd[1]*x_img
                + y_poly_fwd[2]*y_img
                + y_poly_fwd[3]*x_img**2
                + y_poly_fwd[4]*x_img*y_img
                + y_poly_fwd[5]*y_img**2
                + y_poly_fwd[6]*x_img**3
                + y_poly_fwd[7]*x_img**2*y_img
                + y_poly_fwd[8]*x_img*y_img**2
                + y_poly_fwd[9]*y_img**3
                + y_poly_fwd[10]*y_img*r
                + y_poly_fwd[11]*x_img*r)


        # Apply a radial distortion
        elif dist_type.startswith("radial"):

            # Compute the radius normalized to the horizontal image size
            r = sqrt(x_img**2 + (1.0 + xy)*y_img**2)/(x_res/2.0)
            r_corr = r

            # Apply the 3rd order radial distortion
            if dist_type == "radial3":

                # Compute the new radius
                r_corr = (1.0 - k1 - k2)*r + k1*r**2 - k2*r**3

            # Apply the 4th order radial distortion
            elif dist_type == "radial4":

                # Compute the new radius
                r_corr = (1.0 - k1 - k2 - k3)*r + k1*r**2 - k2*r**3 + k3*r**4

            # Apply the 5th order radial distortion
            elif dist_type == "radial5":

                # Compute the new radius
                r_corr = (1.0 - k1 - k2 - k3 - k4)*r + k1*r**2 - k2*r**3 + k3*r**4 - k4*r**5


            # Compute the scaling term
            if r == 0:
                r_scale = 0
            else:
                r_scale = (r_corr/r - 1)

            # Compute offsets
            dx = (x_img - x0)*r_scale
            dy = (y_img - y0)*r_scale*(1.0 + xy)


        # Correct image coordinates for distortion
        x_corr = x_img + dx
        y_corr = y_img + dy


        # Gnomonize coordinates
        x_corr = x_corr/pix_scale
        y_corr = y_corr/pix_scale

        ### ###


        ### Convert gnomonic X, Y to RA, Dec ###

        # Radius from FOV centre to sky coordinate
        radius = radians(sqrt(x_corr**2 + y_corr**2))

        # Compute theta - the direction angle between the FOV centre, sky coordinate, and the north

        #   celestial pole
        theta = (pi/2 - radians(pos_angle_ref) + atan2(y_corr, x_corr))%(2*pi)


        # Compute the reference RA centre at the given JD by adding the hour angle difference
        ra_ref_now = (ra_ref + radians(cyjd2LST(jd, 0)) - radians(h0) + 2*pi)%(2*pi)


        # Correct the FOV centre for refraction
        if refraction:
            ra_ref_now_corr, dec_ref_corr = eqRefractionTrueToApparent(ra_ref_now, dec_ref, jd, \
                radians(lat), radians(lon))

        else:
            ra_ref_now_corr = ra_ref_now
            dec_ref_corr = dec_ref


        # Compute declination
        sin_t = sin(dec_ref_corr)*cos(radius) + cos(dec_ref_corr)*sin(radius)*cos(theta)
        dec = atan2(sin_t, sqrt(1 - sin_t**2))

        # Compute right ascension
        sin_t = sin(theta)*sin(radius)/cos(dec)
        cos_t = (cos(radius) - sin(dec)*sin(dec_ref_corr))/(cos(dec)*cos(dec_ref_corr))
        ra = (ra_ref_now_corr - atan2(sin_t, cos_t) + 2*pi)%(2*pi)


        # Apply refraction correction
        if refraction:
            ra, dec = eqRefractionApparentToTrue(ra, dec, jd, radians(lat), radians(lon))



        # Convert coordinates to degrees
        ra = degrees(ra)
        dec = degrees(dec)


        # Assign values to output list
        ra_data[i] = ra
        dec_data[i] = dec


    return ra_data, dec_data


# @cython.boundscheck(False)
# @cython.cdivision(True)
# def generateRaDecGrid(double ra_d, double dec_d, double jd, double lat, double lon, double x_res, \
#     double y_res, double h0, double ra_ref, double dec_ref, double pos_angle_ref, double pix_scale, \
#     double fov_radius, np.ndarray[FLOAT_TYPE_t, ndim=1] x_poly_fwd, np.ndarray[FLOAT_TYPE_t, ndim=1] y_poly_fwd, \
#     np.ndarray[FLOAT_TYPE_t, ndim=1] x_poly_rev, np.ndarray[FLOAT_TYPE_t, ndim=1] y_poly_rev, \
#     str dist_type, bool refraction=True, bool equal_aspect=False):
#
#     cdef np.ndarray[FLOAT_TYPE_t, ndim=1] RA_c, dec_c
#     RA_c, dec_c = cyXYToRADec(np.array([jd]),
#                               np.array(x_res/2, dtype=np.float64), np.array(y_res/2, dtype=np.float64),
#                               lat, lon, x_res, y_res, h0, ra_d, dec_d, pos_angle_ref, pix_scale, x_poly_fwd,
#                               y_poly_fwd, dist_type, refraction=refraction,
#                               equal_aspect=equal_aspect)
#
#
#     # Compute FOV centre alt/az
#     cdef double azim_centre, alt_centre
#     azim_centre, alt_centre = cyraDec2AltAz(RA_c[0], dec_c[0], jd, lat, lon)
#
#     # Determine gridline frequency (double the gridlines if the number is < 4eN)
#     cdef double grid_freq = 10**np.floor(np.log10(fov_radius))
#     if 10**(np.log10(fov_radius) - np.floor(np.log10(fov_radius))) < 4:
#         grid_freq /= 2
#
#     # Set a maximum grid frequency of 15 deg
#     if grid_freq > 15:
#         grid_freq = 15
#
#     # Grid plot density
#     cdef double plot_dens = grid_freq/100
#
#     # Compute the range of declinations to consider
#     cdef double dec_min = dec_d - fov_radius/2
#     if dec_min < -90:
#         dec_min = -90
#
#     cdef double dec_max = dec_d + fov_radius/2
#     if dec_max > 90:
#         dec_max = 90
#
#     cdef np.ndarray[FLOAT_TYPE_t, ndim=1] ra_grid_arr = np.arange(0, 360, grid_freq)
#     cdef np.ndarray[FLOAT_TYPE_t, ndim=1]dec_grid_arr = np.arange(-90, 90, grid_freq)
#
#     # Filter out the dec grid for min/max declination
#     dec_grid_arr = dec_grid_arr[(dec_grid_arr >= dec_min) & (dec_grid_arr <= dec_max)]
#
#     cdef list x = []
#     cdef list y = []
#     cdef list cuts = []
#
#     cdef np.ndarray[FLOAT_TYPE_t, ndim=1] ra_grid_plot
#     cdef np.ndarray[FLOAT_TYPE_t, ndim=1] dec_grid_plot
#     cdef np.ndarray[FLOAT_TYPE_t, ndim=1] az_grid_plot
#     cdef np.ndarray[FLOAT_TYPE_t, ndim=1] alt_grid_plot
#     cdef np.ndarray[BOOL_TYPE_t, ndim=1] filter_arr
#
#     cdef list ra_grid_plot_list
#     cdef list dec_grid_plot_list
#
#     cdef np.ndarray[INT_TYPE_t, ndim=2] gap_indices
#
#
#     # Plot the celestial parallel grid (circles)
#     for dec_grid in dec_grid_arr:
#
#         ra_grid_plot = np.arange(0, 360, plot_dens)
#         dec_grid_plot = np.zeros_like(ra_grid_plot) + dec_grid
#
#         # Compute alt/az
#         az_grid_plot, alt_grid_plot = cyraDec2AltAz_vect(ra_grid_plot, dec_grid_plot, jd, lat, lon)
#
#         # Filter out points below the horizon  and outside the FOV
#         filter_arr = (alt_grid_plot > 0) & (angularSeparation_fv(alt_centre,
#                                                                 azim_centre,
#                                                                 alt_grid_plot,
#                                                                 az_grid_plot) < fov_radius)
#         ra_grid_plot = ra_grid_plot[filter_arr]
#         dec_grid_plot = dec_grid_plot[filter_arr]
#
#         # Find gaps in continuity and break up plotting individual lines
#         gap_indices = np.argwhere(np.abs(ra_grid_plot[1:] - ra_grid_plot[:-1]) > fov_radius)
#         if len(gap_indices):
#
#             ra_grid_plot_list = []
#             dec_grid_plot_list = []
#
#             # Separate gridlines with large gaps
#             prev_gap_indx = 0
#             for entry in gap_indices:
#                 gap_indx = entry[0]
#
#                 ra_grid_plot_list.append(ra_grid_plot[prev_gap_indx:gap_indx + 1])
#                 dec_grid_plot_list.append(dec_grid_plot[prev_gap_indx:gap_indx + 1])
#
#                 prev_gap_indx = gap_indx
#
#             # Add the last segment
#             ra_grid_plot_list.append(ra_grid_plot[prev_gap_indx + 1:-1])
#             dec_grid_plot_list.append(dec_grid_plot[prev_gap_indx + 1:-1])