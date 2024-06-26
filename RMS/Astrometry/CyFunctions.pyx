#!python
#cython: language_level=3

import numpy as np

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
    double hypot(double, double)
    double fmod(double, double)
    double M_PI "M_PI"


# Define Pi
cdef double pi = M_PI

# Define the Julian date at the J2000 epoch
cdef double J2000_DAYS = 2451545.0



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
@cython.cdivision(True)
cpdef double angularSeparation(double ra1, double dec1, double ra2, double dec2):
    """ Calculate the angular separation between 2 stars in equatorial celestial coordinates. 

    Source of the equation: http://www.astronomycafe.net/qadir/q1890.html (May 1, 2016)

    @param ra1: [float] right ascension of the first stars (in degrees)
    @param dec1: [float] decliantion of the first star (in degrees)
    @param ra2: [float] right ascension of the decons stars (in degrees)
    @param dec2: [float] decliantion of the decons star (in degrees)

    @return angular_separation: [float] angular separation (in degrees)
    """

    cdef double deldec2
    cdef double delra2
    cdef double sindis

    # Convert input coordinates to radians
    ra1 = radians(ra1)
    dec1 =  radians(dec1)
    ra2 = radians(ra2)
    dec2 = radians(dec2)


    # Classical method
    return degrees(acos(sin(dec1)*sin(dec2) + cos(dec1)*cos(dec2)*cos(ra2 - ra1)))


    # # Compute the angular separation using the haversine formula
    # #   Source: https://idlastro.gsfc.nasa.gov/ftp/pro/astro/gcirc.pro
    # deldec2 = (dec2 - dec1)/2.0
    # delra2 =  (ra2 - ra1)/2.0
    # sindis = sqrt(sin(deldec2)*sin(deldec2) + cos(dec1)*cos(dec2)*sin(delra2)*sin(delra2))

    # return degrees(2.0*asin(sindis))



@cython.boundscheck(False)
@cython.wraparound(False) 
def subsetCatalog(np.ndarray[FLOAT_TYPE_t, ndim=2] catalog_list, double ra_c, double dec_c, double jd,
        double lat, double lon, double radius, double mag_limit, bool remove_under_horizon=True):
    """ Make a subset of stars from the given star catalog around the given coordinates with a given radius.
    
    Arguments:
        catalog_list: [ndarray] An array of (ra, dec, mag) pairs for stars (J2000, degrees). Note that the 
            array needs to be sorted by descending declination!
        ra_c: [float] Centre of extraction RA (degrees).
        dec_c: [float] Centre of extraction dec (degrees).
        jd: [float] Julian date of observations.
        lat: [float] Observer latitude (deg).
        lon: [float] Observer longitude (deg).
        radius: [float] Extraction radius (degrees).
        mag_limit: [float] Limiting magnitude.

    Keyword arguments:
        remove_under_horizon: [bool] Remove stars below the horizon (-5 deg below).

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


            # Only take stars above -5 degrees, if the filtering is on
            if not (remove_under_horizon and (degrees(elev) < -5)):
            
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
        dec_corr = sign(dec)*acos(sqrt(A**2 + B**2))
    else:
        dec_corr = asin(C)


    return ra_corr, dec_corr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[np.float64_t, ndim=2] precessionMatrix(double zeta, double theta, double z):
    """ Calculate the precession matrix based on precession angles.

    Arguments:
        zeta: [double] Precession angle zeta in radians.
        theta: [double] Precession angle theta in radians.
        z: [double] Precession angle z in radians.

    Return:
        [np.ndarray] A 3x3 rotation matrix representing the precession transformation.

    Notes:
        - This matrix is used to transform coordinates from one epoch to another, 
          accounting for the precession of the Earth's rotational axis.
        - The matrix is calculated using the formulation from the IAU 1976 precession model.
        - Input angles should be calculated for the time span between the initial and final epochs.
    """

    cdef np.ndarray[np.float64_t, ndim=2, mode="c"] P = np.empty((3, 3), dtype=np.float64)
    cdef double czeta = cos(zeta)
    cdef double szeta = sin(zeta)
    cdef double ctheta = cos(theta)
    cdef double stheta = sin(theta)
    cdef double cz = cos(z)
    cdef double sz = sin(z)

    # Calculate matrix elements
    P[0, 0] = czeta*ctheta*cz - szeta*sz
    P[0, 1] = -szeta*ctheta*cz - czeta*sz
    P[0, 2] = -stheta*cz
    
    P[1, 0] = czeta*ctheta*sz + szeta*cz
    P[1, 1] = -szeta*ctheta*sz + czeta*cz
    P[1, 2] = -stheta*sz
    
    P[2, 0] = czeta*stheta
    P[2, 1] = -szeta*stheta
    P[2, 2] = ctheta
    
    return P



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef (double, double, double) equatorialCoordAndRotPrecession(double start_epoch, double final_epoch,
                                                               double ra, double dec, double rot_angle):
    """ Corrects Right Ascension, Declination, and Rotation wrt Standard angle from one epoch to another,
    taking only precession into account.
    
    Arguments:
        start_epoch: [float] Julian date of the starting epoch.
        final_epoch: [float] Julian date of the final epoch.
        ra: [float] Input right ascension (radians).
        dec: [float] Input declination (radians).
        rot_angle [float] the rotation wrt Standard angle, aka pos_angle_ref (radians).
    
    Return:
        (ra, dec, rot_angle): [tuple of floats] Precessed equatorial coordinates and rotation angle (radians).
    """
    
    # Don't precess if the start and final epoch are the same
    if start_epoch == final_epoch:
        return ra, dec, rot_angle

    cdef:
        np.ndarray[double, ndim=1] initial_vector, transformed_vector, parallel_vec, parallel_vec_precessed
        np.ndarray[double, ndim=1] normal_vector, transformed_normal
        np.ndarray[double, ndim=1] proj_parallel, proj_parallel_precessed
        np.ndarray[double, ndim=2] P
        double ra_precessed, dec_precessed, T, t, zeta, z, theta
        double new_rot_angle, angle1, angle2, rotation_change
        int i

    # Calculate precession parameters
    T = (start_epoch - 2451545.0)/36525.0  # J2000.0 epoch
    t = (final_epoch - start_epoch)/36525.0

    # Calculate correction parameters in degrees
    zeta = ((2306.2181 + 1.39656*T - 0.000139*T ** 2)*t + (0.30188 - 0.000344*T)*t ** 2 + 0.017998*t ** 3)/3600
    z = ((2306.2181 + 1.39656*T - 0.000139*T ** 2)*t + (1.09468 + 0.000066*T)*t ** 2 + 0.018203*t ** 3)/3600
    theta = ((2004.3109 - 0.85330*T - 0.000217*T ** 2)*t - (0.42665 + 0.000217*T)*t ** 2 - 0.041833*t ** 3)/3600

    # Convert parameters to radians
    zeta = radians(zeta)
    z = radians(z)
    theta = radians(theta)

    # Calculate precession matrix
    P = precessionMatrix(zeta, theta, z)

    # Convert RA, Dec to cartesian coordinates
    initial_vector = raDecToCartesian(ra, dec)

    # Apply precession
    transformed_vector = np.dot(P, initial_vector)

    # Calculate normal vector to the plane of precession
    normal_vector = np.cross(initial_vector, transformed_vector)
    transformed_normal = np.dot(P, normal_vector)

    # Normalize vectors
    initial_vector /= np.linalg.norm(initial_vector)
    transformed_vector /= np.linalg.norm(transformed_vector)
    normal_vector /= np.linalg.norm(normal_vector)
    transformed_normal /= np.linalg.norm(transformed_normal)

    # Convert precessed vector back to RA, Dec
    ra_precessed, dec_precessed = cartesianToRaDec(transformed_vector)

    # Calculate vector tangent to the parallel of declination
    parallel_vec = np.array([-sin(ra), cos(ra), 0])
    parallel_vec_precessed = np.array([-sin(ra_precessed), cos(ra_precessed), 0])

    # Project parallel vectors onto the plane perpendicular to the line of sight
    proj_parallel = parallel_vec - np.dot(parallel_vec, initial_vector)*initial_vector
    proj_parallel_precessed = parallel_vec_precessed - np.dot(parallel_vec_precessed, transformed_vector)*transformed_vector

    # Normalize projected vectors
    proj_parallel /= np.linalg.norm(proj_parallel)
    proj_parallel_precessed /= np.linalg.norm(proj_parallel_precessed)

    # Calculate the angles between the normal vector and projected parallels
    # in the plane perpendicular to the line of sight (sensor plane).
    #
    # The normal vectors are fixed relative to the camera sensor.
    # The projected parallels represent how the celestial parallels appear on the sensor plane.
    #
    # Since the rotationWrtStandard function computes pos_angle_ref as the angle between
    # a row of pixels and the parallel passing through the center of the FOV,
    # we want to determine how precession affects this angle.
    #
    # By comparing these angles before and after precession, we can quantify
    # the change in field orientation due to precession.

    # Angle for the initial position
    angle1 = atan2(np.dot(np.cross(normal_vector, proj_parallel), initial_vector), 
                   np.dot(normal_vector, proj_parallel))

    # Angle for the precessed position
    angle2 = atan2(np.dot(np.cross(transformed_normal, proj_parallel_precessed), transformed_vector), 
                   np.dot(transformed_normal, proj_parallel_precessed))

    # Calculate the change in angle
    rotation_change = angle2 - angle1

    # Apply the rotation change to the initial rotation angle
    new_rot_angle = rot_angle + rotation_change

    # Normalize the new rotation angle to be between -pi and pi
    new_rot_angle = fmod(new_rot_angle + M_PI, 2*M_PI) - M_PI

    # print(f"Rotation change due to precession: {degrees(rotation_change)*60} arcmin")

    return ra_precessed, dec_precessed, new_rot_angle



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

    cdef double refraction, elev_calc

    # Don't apply refraction for elevations below -0.5 deg
    if elev <= radians(-0.5):
        elev_calc = radians(-0.5)
    else:
        elev_calc = elev

    # Refraction in radians
    refraction = radians(1.0/(60*tan(radians(degrees(elev_calc) + 7.31/(degrees(elev_calc) + 4.4)))))

    # Correct the elevation
    return elev - refraction



cpdef double pyRefractionApparentToTrue(double elev):
    """ Python version of the refraction correction (apparent to true).

    Arguments:
        elev: [float] Apparent elevation (radians).

    Return:
        [float] True elevation (radians).

    """

    return refractionApparentToTrue(elev)



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

    cdef double refraction, elev_calc

    # Don't apply refraction for elevation below -0.5 deg
    if elev <= radians(-0.5):
        elev_calc = radians(-0.5)
    else:
        elev_calc = elev

    # Refraction in radians
    refraction = radians(1.02/(60*tan(radians(degrees(elev_calc) + 10.3/(degrees(elev_calc) + 5.11)))))

    # Apply the refraction
    return elev + refraction



cpdef double pyRefractionTrueToApparent(double elev):
    """ Python version of the refraction correction (true to apparent).

    Arguments:
        elev: [float] Apparent elevation (radians).

    Return:
        [float] True elevation (radians).

    """

    return refractionTrueToApparent(elev)



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



cpdef (double, double) cyTrueRaDec2ApparentAltAz(double ra, double dec, double jd, double lat, double lon, \
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



@cython.boundscheck(False)
@cython.wraparound(False)
def cyTrueRaDec2ApparentAltAz_vect(np.ndarray[FLOAT_TYPE_t, ndim=1] ra_arr, \
    np.ndarray[FLOAT_TYPE_t, ndim=1] dec_arr, np.ndarray[FLOAT_TYPE_t, ndim=1] jd_arr, \
    double lat, double lon, bool refraction=True):
    """ Convert the true right ascension and declination in J2000 to azimuth (+East of due North) and 
        altitude in the epoch of date. The correction for refraction is performed.
    Arguments:
        ra_arr: [ndarray] Right ascension in radians (J2000).
        dec:_arr [ndarray] Declination in radians (J2000).
        jd_arr: [ndarray] Julian date.
        lat: [float] Latitude in radians.
        lon: [float] Longitude in radians.
    Keyword arguments:
        refraction: [bool] Apply refraction correction. True by default.
    Return:
        (azim, elev): [tuple]
            azim: [ndarray] Azimuth (+east of due north) in radians (epoch of date).
            elev: [ndarray] Elevation above horizon in radians (epoch of date).
        """

    cdef int i
    cdef double azim, elev
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] azim_arr = np.zeros_like(ra_arr)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] elev_arr = np.zeros_like(ra_arr)


    # Convert all entries
    for i in range(len(ra_arr)):

        # Compute alt/az
        azim, elev = cyTrueRaDec2ApparentAltAz(ra_arr[i], dec_arr[i], jd_arr[i], lat, lon, \
            refraction=refraction)

        # Assign alt/az to array
        azim_arr[i] = azim
        elev_arr[i] = elev


    return (azim_arr, elev_arr)




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



cpdef (double, double) cyApparentAltAz2TrueRADec(double azim, double elev, double jd, double lat, double lon, \
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
def cyApparentAltAz2TrueRADec_vect(np.ndarray[FLOAT_TYPE_t, ndim=1] azim_arr, np.ndarray[FLOAT_TYPE_t, ndim=1] elev_arr,
    double jd, double lat, double lon, bool refraction=True):
    """ Convert the apparent azimuth and altitude in the epoch of date to true (refraction corrected) right 
        ascension and declination in J2000.
    Arguments:
        azim_arr: [float] Azimuth (+East of due North) in radians (epoch of date).
        elev_arr: [float] Elevation above horizon in radians (epoch of date).
        jd: [float] Julian date.
        lat: [float] Latitude of the observer in radians.
        lon: [float] Longitde of the observer in radians.
    Keyword arguments:
        refraction: [bool] Apply refraction correction. True by default.
    Return:
        (ra_arr, dec_arr): [tuple]
            ra: [float] Right ascension (radians, J2000).
            dec: [float] Declination (radians, J2000).
    """

    cdef int i
    cdef double ra, dec
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] ra_arr = np.zeros_like(azim_arr)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] dec_arr = np.zeros_like(azim_arr)


    for i in range(len(azim_arr)):

        # Compute RA/Dec
        ra, dec = cyApparentAltAz2TrueRADec(azim_arr[i], elev_arr[i], jd, lat, lon, \
            refraction=refraction)

        ra_arr[i] = ra
        dec_arr[i] = dec

    
    return (ra_arr, dec_arr)



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] raDecToCartesian(double ra, double dec):
    """ Convert RA, Dec to Cartesian coordinates.
    
    Arguments:
        ra: [double] Right ascension in radians.
        dec: [double] Declination in radians.

    Return:
        [np.ndarray] A 3D vector [x, y, z] representing the position in Cartesian coordinates.
            The vector is normalized (unit vector).

    Notes:
        The coordinate system follows the convention:
        x-axis points to RA = 0, Dec = 0
        y-axis points to RA = 90°, Dec = 0
        z-axis points to the North Celestial Pole (Dec = 90°)
    """
    cdef:
        double x = cos(dec)*cos(ra)
        double y = cos(dec)*sin(ra)
        double z = sin(dec)
    return np.array([x, y, z], dtype=np.float64)



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef (double, double) cartesianToRaDec(np.ndarray[np.float64_t, ndim=1] vec):
    """ Convert Cartesian coordinates to RA, Dec.

    Arguments:
        vec: [np.ndarray] A 3D vector [x, y, z] representing the position in Cartesian coordinates.

    Return:
        (ra, dec): [tuple]
            ra: [double] Right ascension in radians, range [0, 2π).
            dec: [double] Declination in radians, range [-π/2, π/2].

    Notes:
        - The function returns (0, 0) if the input vector is [0, 0, 0] to avoid division by zero.
        - The returned RA is normalized to be within [0, 2π).
        - The coordinate system assumes:
          x-axis points to RA = 0, Dec = 0
          y-axis points to RA = 90°, Dec = 0
          z-axis points to the North Celestial Pole (Dec = 90°)
    """
    cdef:
        double x = vec[0]
        double y = vec[1]
        double z = vec[2]
        double distance = hypot(hypot(x, y), z)
        double ra, dec
        
    if distance == 0:
        return 0.0, 0.0
    
    ra = atan2(y, x)
    dec = asin(z/distance)
    
    # Normalize RA to be within [0, 2π)
    ra = fmod(ra + 2*M_PI, 2*M_PI)
    
    return ra, dec



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cyraDecToXY(np.ndarray[FLOAT_TYPE_t, ndim=1] ra_data,
    np.ndarray[FLOAT_TYPE_t, ndim=1] dec_data, double jd, double lat, double lon, double x_res,
    double y_res, double h0, double jd_ref, double ra_ref, double dec_ref, double pos_angle_ref, 
    double pix_scale, np.ndarray[FLOAT_TYPE_t, ndim=1] x_poly_rev, 
    np.ndarray[FLOAT_TYPE_t, ndim=1] y_poly_rev, str dist_type, bool refraction=True, bool equal_aspect=False, 
    bool force_distortion_centre=False, bool asymmetry_corr=True):
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
        jd_ref: [float] Reference Julian date of plate solution.
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
        asymmetry_corr: [bool] Correct the distortion for asymmetry. Only for radial distortion. True by
            default.
    
    Return:
        (x, y): [tuple of ndarrays] Image X and Y coordinates.
    """

    cdef int i
    cdef double ra_centre, dec_centre, ra, dec, ra_centre_j2000, dec_centre_j2000, pos_angle_ref_corr
    cdef double radius, sin_ang, cos_ang, theta, x, y, r, dx, dy, x_img, y_img, r_corr, r_scale
    cdef double x0, y0, xy, a1, a2, k1, k2, k3, k4, k5
    cdef int index_offset

    # Init output arrays
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] x_array = np.zeros_like(ra_data)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] y_array = np.zeros_like(ra_data)


    # Compute the current RA of the FOV centre by adding the difference in between the current and the 
    #   reference hour angle
    ra_centre = radians((ra_ref + cyjd2LST(jd, 0) - h0 + 360)%360)
    dec_centre = radians(dec_ref)

    # Correct the reference FOV centre for refraction
    if refraction:
        ra_centre, dec_centre = eqRefractionTrueToApparent(ra_centre, dec_centre, jd, radians(lat), \
            radians(lon))
            

    # Precess the FOV centre and rotation angle to J2000 (otherwise the FOV centre drifts with time)
    ra_centre_j2000, dec_centre_j2000, pos_angle_ref_corr = equatorialCoordAndRotPrecession(jd, J2000_DAYS,
                                                            ra_centre, dec_centre, radians(pos_angle_ref))

    # The position angle needs to be corrected for precession, otherwise the FOV rotates with time
    # Applying the difference in RA between the current and the reference epoch fixes the position angle
    pos_angle_ref_corr = degrees(pos_angle_ref_corr)

    ra_centre = ra_centre_j2000
    dec_centre = dec_centre_j2000


    # If the radial distortion is used, unpack radial parameters
    if dist_type.startswith("radial"):

        # Index offset for reading distortion parameters. May change as equal aspect or asymmetry correction
        #   is toggled on/off
        index_offset = 0

        # Force the distortion centre to the image centre
        if force_distortion_centre:
            x0 = 0.5/(x_res/2.0)
            y0 = 0.5/(y_res/2.0)
            index_offset += 2
        else:
            # Read distortion offsets
            x0 = x_poly_rev[0]
            y0 = x_poly_rev[1]


        # Normalize offsets
        x0 *= (x_res/2.0)
        y0 *= (y_res/2.0)

        # Wrap offsets to always be within the image
        x0 = -x_res/2.0 + (x0 + x_res/2.0)%x_res
        y0 = -y_res/2.0 + (y0 + y_res/2.0)%y_res


        # Aspect ratio
        if equal_aspect:
            xy = 0.0
            index_offset += 1
        else:
            xy = x_poly_rev[2 - index_offset]


        # Asymmetry correction
        if asymmetry_corr:

            # Asymmetry amplitude
            a1 = x_poly_rev[3 - index_offset]

            # Asymmetry angle - normalize so full circle fits within 0-1
            a2 = (x_poly_rev[4 - index_offset]*(2*pi))%(2*pi)

        else:
            a1 = 0.0
            a2 = 0.0
            index_offset += 2

        # Distortion coeffs
        k1 = x_poly_rev[5 - index_offset]
        k2 = x_poly_rev[6 - index_offset]

        if x_poly_rev.shape[0] > (7 - index_offset):
            k3 = x_poly_rev[7 - index_offset]

        if x_poly_rev.shape[0] > (8 - index_offset):
            k4 = x_poly_rev[8 - index_offset]

        # if x_poly_rev.shape[0] > (9 - index_offset):
        #     k5 = x_poly_rev[9 - index_offset]

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
        radius = radians(angularSeparation(degrees(ra), degrees(dec), degrees(ra_centre), \
            degrees(dec_centre)))

        # Compute theta - the direction angle between the FOV centre, sky coordinate, and the image vertical
        sin_ang = cos(dec)*sin(ra - ra_centre)/sin(radius)
        cos_ang = (sin(dec) - sin(dec_centre)*cos(radius))/(cos(dec_centre)*sin(radius))
        theta   = -atan2(sin_ang, cos_ang) + radians(pos_angle_ref_corr) - pi/2.0

        # Calculate the standard coordinates
        x = degrees(radius)*cos(theta)*pix_scale
        y = degrees(radius)*sin(theta)*pix_scale

        ### ###

        # Set initial distorsion values
        dx = 0
        dy = 0

        # Apply 3rd order polynomial + one radial term distortion
        if dist_type.startswith("poly3+radial"):

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

            # If the 3rd order radial term is used, apply it
            if dist_type.endswith("+radial3") or dist_type.endswith("+radial5"):
                dx += x_poly_rev[12]*x*r**3
                dy += y_poly_rev[12]*y*r**3


            # If the 5th order radial term is used, apply it
            if dist_type.endswith("+radial5"):
                dx += x_poly_rev[13]*x*r**5
                dy += y_poly_rev[13]*y*r**5


        # Apply a radial distortion
        elif dist_type.startswith("radial"):

            # Compute the radius
            r = sqrt(x**2 + y**2)

            # Apply the asymmetry correction
            r = r + a1*y*cos(a2) - a1*x*sin(a2)

            # Normalize radius to horizontal size
            r = r/(x_res/2.0)

            r_corr = r

            # Apply the 3rd order radial distortion, all powers
            if dist_type == "radial3-all":

                # Compute the new radius
                r_corr = r + k1*r**2 + k2*r**3

            # Apply the 4th order radial distortion, all powers
            elif dist_type == "radial4-all":

                # Compute the new radius
                r_corr = r + k1*r**2 + k2*r**3 + k3*r**4

            # Apply the 5th order radial distortion, all powers
            elif dist_type == "radial5-all":

                # Compute the new radius
                r_corr = r + k1*r**2 + k2*r**3 + k3*r**4 + k4*r**5

            # Apply the 3rd order radial distortion, only odd powers
            elif dist_type == "radial3-odd":

                # Compute the new radius
                r_corr = r + k1*r**3

            # Apply the 5th order radial distortion, only odd powers
            elif dist_type == "radial5-odd":

                # Compute the new radius
                r_corr = r + k1*r**3 + k2*r**5


            # Apply the 7th order radial distortion, only odd powers
            elif dist_type == "radial7-odd":

                # Compute the new radius
                r_corr = r + k1*r**3 + k2*r**5 + k3*r**7


            # Apply the 9th order radial distortion, only odd powers
            elif dist_type == "radial9-odd":

                # Compute the new radius
                r_corr = r + k1*r**3 + k2*r**5 + k3*r**7 + k4*r**9


            # Compute the scaling term
            if r == 0:
                r_scale = 0
            else:
                r_scale = (r_corr/r - 1)


            # Compute distortion offsets
            dx = x*r_scale - x0
            dy = y*r_scale/(1.0 + xy) - y0 + y*(1.0 - 1.0/(1.0 + xy))



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
    double h0, double jd_ref, double ra_ref, double dec_ref, double pos_angle_ref, double pix_scale, \
    np.ndarray[FLOAT_TYPE_t, ndim=1] x_poly_fwd, np.ndarray[FLOAT_TYPE_t, ndim=1] y_poly_fwd, \
    str dist_type, bool refraction=True, bool equal_aspect=False, bool force_distortion_centre=False,\
    bool asymmetry_corr=True):
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
        jd_ref: [float] Reference Julian date when the plate was fit.
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
        asymmetry_corr: [bool] Correct the distortion for asymmetry. Only for radial distortion. True by
            default.
    
    Return:
        (ra_data, dec_data): [tuple of ndarrays]
            
            ra_data: [ndarray] Right ascension of each point (deg).
            dec_data: [ndarray] Declination of each point (deg).
            magnitude_data: [ndarray] Array of meteor's lightcurve apparent magnitudes.
    """

    cdef int i
    cdef double jd, x_img, y_img, r, dx, x_corr, dy, y_corr, r_corr, r_scale
    cdef double x0, y0, xy, a1, a2, k1, k2, k3, k4, k5, ra_ref_now_corr_j2000,dec_ref_corr_j2000
    cdef int index_offset
    cdef double radius, theta, sin_t, cos_t
    cdef double ra_ref_now, ra_ref_now_corr, ra, dec, dec_ref_corr, pos_angle_ref_now_corr

    # Convert the reference pointing direction to radians
    ra_ref  = radians(ra_ref)
    dec_ref = radians(dec_ref)

    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] ra_data = np.zeros_like(jd_data)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] dec_data = np.zeros_like(jd_data)


    # If the radial distortion is used, unpack radial parameters
    if dist_type.startswith("radial"):


        # Index offset for reading distortion parameters. May change as equal aspect or asymmetry correction
        #   is toggled on/off
        index_offset = 0

        # Force the distortion centre to the image centre
        if force_distortion_centre:
            x0 = 0.5/(x_res/2.0)
            y0 = 0.5/(y_res/2.0)
            index_offset += 2
        else:
            # Read distortion offsets
            x0 = x_poly_fwd[0]
            y0 = x_poly_fwd[1]


        # Normalize offsets
        x0 *= (x_res/2.0)
        y0 *= (y_res/2.0)

        # Wrap offsets to always be within the image
        x0 = -x_res/2.0 + (x0 + x_res/2.0)%x_res
        y0 = -y_res/2.0 + (y0 + y_res/2.0)%y_res

        # Aspect ratio
        if equal_aspect:
            xy = 0.0
            index_offset += 1
        else:
            # Read aspect ratio
            xy = x_poly_fwd[2 - index_offset]


        # Asymmetry correction
        if asymmetry_corr:

            # Asymmetry amplitude
            a1 = x_poly_fwd[3 - index_offset]

            # Asymmetry angle - normalize so full circle fits within 0-1
            a2 = (x_poly_fwd[4 - index_offset]*(2*pi))%(2*pi)

        else:
            a1 = 0.0
            a2 = 0.0
            index_offset += 2


        # Distortion coeffs
        k1 = x_poly_fwd[5 - index_offset]
        k2 = x_poly_fwd[6 - index_offset]

        if x_poly_fwd.shape[0] > (7 - index_offset):
            k3 = x_poly_fwd[7 - index_offset]

        if x_poly_fwd.shape[0] > (8 - index_offset):
            k4 = x_poly_fwd[8 - index_offset]

        # if x_poly_fwd.shape[0] > (9 - index_offset):
        #     k5 = x_poly_fwd[9 - index_offset]


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
        if dist_type.startswith("poly3+radial"):

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

            # If the 3rd order radial term is used, apply it
            if dist_type.endswith("+radial3") or dist_type.endswith("+radial5"):
                dx += x_poly_fwd[12]*x_img*r**3
                dy += y_poly_fwd[12]*y_img*r**3

            # If the 5th order radial term is used, apply it
            if dist_type.endswith("+radial5"):
                dx += x_poly_fwd[13]*x_img*r**5
                dy += y_poly_fwd[13]*y_img*r**5


        # Apply a radial distortion
        elif dist_type.startswith("radial"):

            # Compute the radius
            r = sqrt((x_img - x0)**2 + ((1.0 + xy)*(y_img - y0))**2)

            # Apply the asymmetry correction
            r = r + a1*(1.0 + xy)*(y_img - y0)*cos(a2) - a1*(x_img - x0)*sin(a2)

            # Normalize radius to horizontal size
            r = r/(x_res/2.0)

            r_corr = r


            # Apply the 3rd order radial distortion, all powers
            if dist_type == "radial3-all":

                # Compute the new radius
                r_corr = r + k1*r**2 + k2*r**3

            # Apply the 4th order radial distortion, all powers
            elif dist_type == "radial4-all":

                # Compute the new radius
                r_corr = r + k1*r**2 + k2*r**3 + k3*r**4

            # Apply the 5th order radial distortion, all powers
            elif dist_type == "radial5-all":

                # Compute the new radius
                r_corr = r + k1*r**2 + k2*r**3 + k3*r**4 + k4*r**5

            # Apply the 3rd order radial distortion, only odd powers
            elif dist_type == "radial3-odd":

                # Compute the new radius
                r_corr = r + k1*r**3

            # Apply the 5th order radial distortion, only odd powers
            elif dist_type == "radial5-odd":

                # Compute the new radius
                r_corr = r + k1*r**3 + k2*r**5

            # Apply the 7th order radial distortion, only odd powers
            elif dist_type == "radial7-odd":

                # Compute the new radius
                r_corr = r + k1*r**3 + k2*r**5 + k3*r**7

            # Apply the 9th order radial distortion, only odd powers
            elif dist_type == "radial9-odd":

                # Compute the new radius
                r_corr = r + k1*r**3 + k2*r**5 + k3*r**7 + k4*r**9


            # Compute the scaling term
            if r == 0:
                r_scale = 0
            else:
                r_scale = (r_corr/r - 1)

            # Compute offsets
            dx = (x_img - x0)*r_scale - x0
            dy = (y_img - y0)*r_scale*(1.0 + xy) - y0*(1.0 + xy) + y_img*xy


        # Correct image coordinates for distortion
        x_corr = x_img + dx
        y_corr = y_img + dy


        # Gnomonize coordinates
        x_corr = x_corr/pix_scale
        y_corr = y_corr/pix_scale

        ### ###


        ### Convert gnomonic X, Y to RA, Dec ###

        # Compute the reference RA centre at the given JD by adding the hour angle difference
        ra_ref_now = (ra_ref + radians(cyjd2LST(jd, 0)) - radians(h0) + 2*pi)%(2*pi)

        # Correct the FOV centre for refraction
        if refraction:
            ra_ref_now_corr, dec_ref_corr = eqRefractionTrueToApparent(ra_ref_now, dec_ref, jd, \
                radians(lat), radians(lon))

        else:
            ra_ref_now_corr = ra_ref_now
            dec_ref_corr = dec_ref


        # Precess the reference RA, dec, position angle to J2000 (needs to be used to avoid FOV centre drift
        # over time)
        ra_ref_now_corr_j2000, dec_ref_corr_j2000, pos_angle_ref_now_corr = equatorialCoordAndRotPrecession(jd,
                                            J2000_DAYS, ra_ref_now_corr, dec_ref_corr, radians(pos_angle_ref))

        # The position angle needs to be corrected for precession, otherwise the FOV rotates with time
        # Applying the difference in RA between the current and the reference epoch fixes the position angle
        pos_angle_ref_now_corr = degrees(pos_angle_ref_now_corr)

        ra_ref_now_corr = ra_ref_now_corr_j2000
        dec_ref_corr = dec_ref_corr_j2000

        # Radius from FOV centre to sky coordinate
        radius = radians(sqrt(x_corr**2 + y_corr**2))

        # Compute theta - the direction angle between the FOV centre, sky coordinate, and the north 
        #   celestial pole
        theta = (pi/2 - radians(pos_angle_ref_now_corr) + atan2(y_corr, x_corr))%(2*pi)


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