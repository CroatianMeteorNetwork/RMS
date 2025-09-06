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
    @param dec1: [float] declination of the first star (in degrees)
    @param ra2: [float] right ascension of the decons stars (in degrees)
    @param dec2: [float] declination of the decons star (in degrees)

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
            - filtered_indices - Indices of catalog_list entries which satisfy the filters.
            - filtered_list - catalog_list entires that satisfy the filters.
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

    # Get the lengths of input arrays
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
cdef (double, double) nutationComponents(double T):
    """ Calculate nutation corrections """

    cdef double omega, L, Ll, delta_psi, delta_eps

    # Longitude of the ascending node of the Moon's mean orbit on the ecliptic, measured from the mean equinox
    # of the date
    omega = radians(125.04452 - 1934.136261*T)

    # Mean longitude of the Sun
    L = radians(280.4665 + 36000.7698*T)

    # Mean longitude of the Moon
    Ll = radians(218.3165 + 481267.8813*T)

    # Nutation in longitude
    delta_psi = -17.2*sin(omega) - 1.32*sin(2*L) - 0.23*sin(2*Ll) + 0.21*sin(2*omega)

    # Nutation in obliquity
    delta_eps = 9.2*cos(omega) + 0.57*cos(2*L) + 0.1*cos(2*Ll) - 0.09*cos(2*omega)

    # Convert to radians
    delta_psi = radians(delta_psi/3600)
    delta_eps = radians(delta_eps/3600)

    return delta_psi, delta_eps


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
        np.ndarray[double, ndim=2] epsilon_matrix, psi_matrix

        double ra_precessed, dec_precessed, T, t, zeta, z, theta, Delta_psi, Delta_epsilon
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



    # Calculate nutation corrections
    Delta_psi, Delta_epsilon = nutationComponents(T)

    # Construct the nutation matrices
    epsilon_matrix = np.array([[1,                  0,                   0],
                               [0, cos(Delta_epsilon),  sin(Delta_epsilon)],
                               [0, -sin(Delta_epsilon),  cos(Delta_epsilon)]])

    psi_matrix = np.array([[cos(Delta_psi), -sin(Delta_psi), 0],
                           [sin(Delta_psi),  cos(Delta_psi), 0],
                           [0,                            0, 1]])

    # Apply nutation
    transformed_vector = np.dot(epsilon_matrix, transformed_vector)
    transformed_vector = np.dot(psi_matrix, transformed_vector)


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

    # print("Rotation change due to precession: {} arcmin".format(degrees(rotation_change) * 60))

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
cpdef double refractionTrueToApparent(double elev):
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
        lon: [float] Longitude of the observer in radians.

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
        lon: [float] Longitude of the observer in radians.

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
        lon: [float] Longitude of the observer in radians.
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
        y-axis points to RA = 90, Dec = 0
        z-axis points to the North Celestial Pole (Dec = 90)
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
            ra: [double] Right ascension in radians, range [0, 2pi).
            dec: [double] Declination in radians, range [-pi/2, pi/2].

    Notes:
        - The function returns (0, 0) if the input vector is [0, 0, 0] to avoid division by zero.
        - The returned RA is normalized to be within [0, 2pi).
        - The coordinate system assumes:
          x-axis points to RA = 0, Dec = 0
          y-axis points to RA = 90, Dec = 0
          z-axis points to the North Celestial Pole (Dec = 90)
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
    
    # Normalize RA to be within [0, 2pi)
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
    """ Convert RA, Dec to distortion corrected image coordinates. 

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
        pos_angle_ref: [float] Rotation from the celestial meridian (degrees).
        pix_scale: [float] Image scale (px/deg).
        x_poly_rev: [ndarray float] Distortion polynomial in X direction for reverse mapping.
        y_poly_rev: [ndarray float] Distortion polynomial in Y direction for reverse mapping.
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
    cdef double ra_centre, dec_centre, ra, dec
    cdef double radius, sin_ang, cos_ang, theta, x, y, r, dx, dy, x_img, y_img, r_corr, r_scale
    cdef double x0, y0, xy, a1, a2, k1, k2, k3, k4, k5
    cdef int index_offset

    # Init output arrays
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] x_array = np.zeros_like(ra_data)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] y_array = np.zeros_like(ra_data)


    # Correct the pointing for precession (output in radians)
    ra_centre, dec_centre, pos_angle_ref = pointingCorrection(
            jd, radians(lat), radians(lon), 
            radians(h0), jd_ref, radians(ra_ref), radians(dec_ref), radians(pos_angle_ref), 
            refraction=refraction
            )

    # # Compute the current RA of the FOV centre by adding the difference in between the current and the 
    # #   reference hour angle
    # ra_centre = radians((ra_ref + cyjd2LST(jd, 0) - h0 + 360)%360)
    # dec_centre = radians(dec_ref)

    # # Correct the reference FOV centre for refraction
    # if refraction:
    #     ra_centre, dec_centre = eqRefractionTrueToApparent(ra_centre, dec_centre, jd, radians(lat), \
    #         radians(lon))
            

    # # Precess the FOV centre and rotation angle to J2000 (otherwise the FOV centre drifts with time)
    # ra_centre_j2000, dec_centre_j2000, pos_angle_ref_corr = equatorialCoordAndRotPrecession(jd, J2000_DAYS,
    #                                                         ra_centre, dec_centre, radians(pos_angle_ref))

    # # The position angle needs to be corrected for precession, otherwise the FOV rotates with time
    # # Applying the difference in RA between the current and the reference epoch fixes the position angle
    # pos_angle_ref_corr = degrees(pos_angle_ref_corr)

    # ra_centre = ra_centre_j2000
    # dec_centre = dec_centre_j2000


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
        theta   = -atan2(sin_ang, cos_ang) + pos_angle_ref - pi/2.0

        # Calculate the standard coordinates
        x = degrees(radius)*cos(theta)*pix_scale
        y = degrees(radius)*sin(theta)*pix_scale

        ### ###

        # Set initial distortion values
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
def cyRaDecToXY_iter(np.ndarray[FLOAT_TYPE_t, ndim=1] ra_data,
    np.ndarray[FLOAT_TYPE_t, ndim=1] dec_data, double jd, double lat, double lon, double x_res,
    double y_res, double h0, double jd_ref, double ra_ref, double dec_ref, double pos_angle_ref, 
    double pix_scale, np.ndarray[FLOAT_TYPE_t, ndim=1] x_poly_fwd, 
    np.ndarray[FLOAT_TYPE_t, ndim=1] y_poly_fwd, str dist_type, bool refraction=True, bool equal_aspect=False, 
    bool force_distortion_centre=False, bool asymmetry_corr=True):
    """ Convert RA, Dec to distortion corrected image coordinates using iterative solver for radial distortions.

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
        pos_angle_ref: [float] Rotation from the celestial meridian (degrees).
        pix_scale: [float] Image scale (px/deg).
        x_poly_fwd: [ndarray float] Distortion polynomial in X direction for reverse mapping.
        y_poly_fwd: [ndarray float] Distortion polynomial in Y direction for reverse mapping.
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

    cdef int i, j
    cdef double ra_centre, dec_centre, ra, dec
    cdef double radius, sin_ang, cos_ang, theta, x, y, r, dx, dy, x_img, y_img, r_corr, r_scale
    cdef double x0, y0, xy, a1, a2, k1, k2, k3, k4
    cdef int index_offset
    cdef double delta_r, lens_dist, r1, r2, x_img1, y_img1, x_img2, y_img2
    cdef double x_img1_est, y_img1_est, x_img2_est, y_img2_est
    cdef double x_corr1, y_corr1, sin_t, cos_t

    # Init output arrays
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] x_array = np.zeros_like(ra_data)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] y_array = np.zeros_like(ra_data)

    # Correct the pointing for precession (output in radians)
    ra_centre, dec_centre, pos_angle_ref = pointingCorrection(
            jd, radians(lat), radians(lon), 
            radians(h0), jd_ref, radians(ra_ref), radians(dec_ref), radians(pos_angle_ref), 
            refraction=refraction
            )

    # If the radial distortion is used, unpack radial parameters
    if dist_type.startswith("radial"):

        # Index offset for reading distortion parameters. May change as equal aspect or asymmetry correction
        #   is toggled on/off
        index_offset = 0

        # Force the distortion centre to the image centre
        if force_distortion_centre:
            x0 = 0.5/(x_res/2.0)  # 0.5 pixel offset to true center
            y0 = 0.5/(y_res/2.0)
            index_offset += 2
        else:
            # Read distortion offsets
            x0 = x_poly_fwd[0]
            y0 = x_poly_fwd[1]
            
        # Convert offsets to pixel coordinates
        x0 *= (x_res/2.0)
        y0 *= (y_res/2.0)

        # Check if X/Y have equal aspect ratio
        if equal_aspect:
            xy = 0
            index_offset += 1
        else:
            # Read the aspect ratio
            xy = x_poly_fwd[2 - index_offset]

        # Check if the assymmetry correction was used
        if asymmetry_corr:
            # Read the assymetry values
            a1 = x_poly_fwd[3 - index_offset]
            a2 = (x_poly_fwd[4 - index_offset]*(2*pi))%(2*pi)
        else:
            a1 = 0.0
            a2 = 0.0
            index_offset += 2

        # Read distortion coefficients
        if dist_type == "radial3-all":
            k1 = x_poly_fwd[5 - index_offset]
            k2 = x_poly_fwd[6 - index_offset]
            k3 = 0.0
            k4 = 0.0

        elif dist_type == "radial4-all":
            k1 = x_poly_fwd[5 - index_offset]
            k2 = x_poly_fwd[6 - index_offset]
            k3 = x_poly_fwd[7 - index_offset]
            k4 = 0.0

        elif dist_type == "radial5-all":
            k1 = x_poly_fwd[5 - index_offset]
            k2 = x_poly_fwd[6 - index_offset]
            k3 = x_poly_fwd[7 - index_offset]
            k4 = x_poly_fwd[8 - index_offset]

        elif dist_type == "radial3-odd":
            k1 = x_poly_fwd[5 - index_offset]
            k2 = 0.0
            k3 = 0.0
            k4 = 0.0

        elif dist_type == "radial5-odd":
            k1 = x_poly_fwd[5 - index_offset]
            k2 = x_poly_fwd[6 - index_offset]
            k3 = 0.0
            k4 = 0.0

        elif dist_type == "radial7-odd":
            k1 = x_poly_fwd[5 - index_offset]
            k2 = x_poly_fwd[6 - index_offset]
            k3 = x_poly_fwd[7 - index_offset]
            k4 = 0.0

        elif dist_type == "radial9-odd":
            k1 = x_poly_fwd[5 - index_offset]
            k2 = x_poly_fwd[6 - index_offset]
            k3 = x_poly_fwd[7 - index_offset]
            k4 = x_poly_fwd[8 - index_offset]

    # Convert all equatorial coordinates to image coordinates
    for i in range(ra_data.shape[0]):

        # Read the next coordinate
        ra = radians(ra_data[i])
        dec = radians(dec_data[i])

        # Apply refraction
        if refraction:
            ra, dec = eqRefractionTrueToApparent(ra, dec, jd, radians(lat), radians(lon))

        # Compute the distance from the FOV centre to the sky coordinate
        radius = radians(angularSeparation(degrees(ra), degrees(dec), degrees(ra_centre), degrees(dec_centre)))

        # Compute theta - the direction angle between the FOV centre, sky coordinate, and the image vertical
        if radius < 1e-8:
            theta = 0.0
        else:
            sin_ang = cos(dec)*sin(ra - ra_centre)/sin(radius)
            cos_ang = (sin(dec) - sin(dec_centre)*cos(radius))/(cos(dec_centre)*sin(radius))
            theta = -atan2(sin_ang, cos_ang) + pos_angle_ref - pi/2.0

        # Calculate the standard coordinates
        x_corr = degrees(radius)*cos(theta)*pix_scale
        y_corr = degrees(radius)*sin(theta)*pix_scale

        # Apply polynomial distortion
        if dist_type.startswith("poly3+radial"):

            # Compute the radius from pixel coordinates
            r = sqrt((x_corr - x0)**2 + (y_corr - y0)**2)

            # Calculate the distortion in X direction (using pixel coordinates)
            dx = (x0
                + x_poly_fwd[1]*x_corr
                + x_poly_fwd[2]*y_corr
                + x_poly_fwd[3]*x_corr**2
                + x_poly_fwd[4]*x_corr*y_corr
                + x_poly_fwd[5]*y_corr**2
                + x_poly_fwd[6]*x_corr**3
                + x_poly_fwd[7]*x_corr**2*y_corr
                + x_poly_fwd[8]*x_corr*y_corr**2
                + x_poly_fwd[9]*y_corr**3
                + x_poly_fwd[10]*x_corr*r
                + x_poly_fwd[11]*y_corr*r)
                
            # Calculate the distortion in Y direction (using pixel coordinates)
            dy = (y0
                + y_poly_fwd[1]*x_corr
                + y_poly_fwd[2]*y_corr
                + y_poly_fwd[3]*x_corr**2
                + y_poly_fwd[4]*x_corr*y_corr
                + y_poly_fwd[5]*y_corr**2
                + y_poly_fwd[6]*x_corr**3
                + y_poly_fwd[7]*x_corr**2*y_corr
                + y_poly_fwd[8]*x_corr*y_corr**2
                + y_poly_fwd[9]*y_corr**3
                + y_poly_fwd[10]*y_corr*r
                + y_poly_fwd[11]*x_corr*r)

            # If the 3rd order radial term is used, apply it
            if dist_type.endswith("+radial3") or dist_type.endswith("+radial5"):
                dx += x_poly_fwd[12]*x_corr*r**3
                dy += y_poly_fwd[12]*y_corr*r**3

            # If the 5th order radial term is used, apply it
            if dist_type.endswith("+radial5"):
                dx += x_poly_fwd[13]*x_corr*r**5
                dy += y_poly_fwd[13]*y_corr*r**5

            x_img = x_corr - dx
            y_img = y_corr - dy

        # Apply radial distortion using iterative solver
        elif dist_type.startswith("radial"):
            
            # Initialize the reverse radial iteration loop
            delta_r = 1.0
            j = 0

            # Set initial guess (undistorted coordinates in pixels)
            x_img = x_corr
            y_img = y_corr

            # Iterate to find the distorted position
            while delta_r > 0.01 and j < 100:  # 0.01 pixel tolerance
                j += 1

                # Compute the radius (with aspect ratio and asymmetry, in pixels then normalized)
                r = sqrt((x_img - x0)**2 + ((1.0 + xy)*(y_img - y0))**2)
                r = r + a1*(1.0 + xy)*(y_img - y0)*cos(a2) - a1*(x_img - x0)*sin(a2)
                r = r/(x_res/2.0)  # Normalize to horizontal size

                r_corr = r
                
                # Apply the appropriate radial distortion model
                if dist_type == "radial3-all":
                    r_corr = r + k1*r**2 + k2*r**3

                elif dist_type == "radial4-all":
                    r_corr = r + k1*r**2 + k2*r**3 + k3*r**4

                elif dist_type == "radial5-all":
                    r_corr = r + k1*r**2 + k2*r**3 + k3*r**4 + k4*r**5

                elif dist_type == "radial3-odd":
                    r_corr = r + k1*r**3

                elif dist_type == "radial5-odd":
                    r_corr = r + k1*r**3 + k2*r**5

                elif dist_type == "radial7-odd":
                    r_corr = r + k1*r**3 + k2*r**5 + k3*r**7

                elif dist_type == "radial9-odd":
                    r_corr = r + k1*r**3 + k2*r**5 + k3*r**7 + k4*r**9
                
                # Compute the scaling factor
                if r == 0:
                    r_scale = 0
                else:
                    r_scale = (r_corr/r - 1)
                
                # Stop iterating if distortion is negligible
                if fabs(r_scale) < 1e-8:
                    break
                
                # Compute distortion offsets (matching cyXYToRADec)
                dx = (x_img - x0)*r_scale - x0
                dy = (y_img - y0)*r_scale*(1.0 + xy) - y0*(1.0 + xy) + y_img*xy
                
                # Compute new estimate by inverting: x_corr = x_img + dx
                x_img_est = x_corr - dx
                y_img_est = y_corr - dy

                # Compute distance between current and last guess
                delta_r = sqrt((x_img - x_img_est)**2 + (y_img - y_img_est)**2)

                # Update guess
                x_img = x_img_est
                y_img = y_img_est

        else:
            # No distortion
            x_img = x_corr
            y_img = y_corr

        # Shift to image coordinate system (0,0 at top-left)
        x_array[i] = x_img + x_res/2.0
        y_array[i] = y_img + y_res/2.0

    return x_array, y_array


cpdef (double, double, double) pointingCorrection(
    double jd, double lat, double lon, 
    double h0, double jd_ref, double ra_ref, double dec_ref, double pos_angle_ref, 
    bool refraction=True
    ):
    """ Compute the pointing correction for the given Julian date. The correction is done by computing the
        difference in RA between the current and the reference epoch. The correction is done in the J2000
        epoch to avoid the drift of the FOV centre over time.

    Arguments:
        jd: [float] Julian date.
        lat: [float] Latitude of the observer in radians.
        lon: [float] Longitude of the observer in radians.
        h0: [float] Reference hour angle in radians.
        jd_ref: [float] Reference Julian date of the plate solution.
        ra_ref: [float] Reference right ascension of the image centre in radians.
        dec_ref: [float] Reference declination of the image centre in radians.
        pos_angle_ref: [float] Rotation from the celestial meridian in radians.
    
    Keyword arguments:
        refraction: [bool] Apply refraction correction. True by default.

    Return:
        (ra_ref_now_corr, dec_ref_corr, pos_angle_ref_now_corr): [tuple]
            ra_ref_now_corr: [float] Corrected right ascension of the image centre in J2000 (radians).
            dec_ref_corr: [float] Corrected declination of the image centre in J2000 (radians).
            pos_angle_ref_now_corr: [float] Corrected position angle in J2000 (radians).

    """

    cdef double ra_ref_now, ra_ref_now_corr, dec_ref_corr, ra_ref_now_corr_j2000, dec_ref_corr_j2000
    cdef double pos_angle_ref_now_corr

    # Compute the reference RA centre at the given JD by adding the hour angle difference
    ra_ref_now = (ra_ref + radians(cyjd2LST(jd, 0)) - h0 + 2*pi)%(2*pi)

    # Correct the FOV centre for refraction
    if refraction:
        ra_ref_now_corr, dec_ref_corr = eqRefractionTrueToApparent(ra_ref_now, dec_ref, jd, lat, lon)

    else:
        ra_ref_now_corr = ra_ref_now
        dec_ref_corr = dec_ref


    # Precess the reference RA, dec, position angle to J2000 (needs to be used to avoid FOV centre drift
    # over time)
    # The position angle needs to be corrected for precession, otherwise the FOV rotates with time
    # Applying the difference in RA between the current and the reference epoch fixes the position angle
    ra_ref_now_corr_j2000, dec_ref_corr_j2000, pos_angle_ref_now_corr = equatorialCoordAndRotPrecession(jd,
                                        J2000_DAYS, ra_ref_now_corr, dec_ref_corr, pos_angle_ref)

    return ra_ref_now_corr_j2000, dec_ref_corr_j2000, pos_angle_ref_now_corr



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cyXYToRADec(np.ndarray[FLOAT_TYPE_t, ndim=1] jd_data, np.ndarray[FLOAT_TYPE_t, ndim=1] x_data, \
    np.ndarray[FLOAT_TYPE_t, ndim=1] y_data, double lat, double lon, double x_res, double y_res, \
    double h0, double jd_ref, double ra_ref, double dec_ref, double pos_angle_ref, double pix_scale, \
    np.ndarray[FLOAT_TYPE_t, ndim=1] x_poly_fwd, np.ndarray[FLOAT_TYPE_t, ndim=1] y_poly_fwd, \
    str dist_type, bool refraction=True, bool equal_aspect=False, bool force_distortion_centre=False,\
    bool asymmetry_corr=True, bool precompute_pointing_corr=False):
    """
    Arguments:
        jd_data: [ndarray] Julian date of each data point.
        x_data: [ndarray] 1D numpy array containing the image column.
        y_data: [ndarray] 1D numpy array containing the image row.
        lat: [float] Latitude of the observer in degrees.
        lon: [float] Longitude of the observer in degrees.
        x_res: [int] Image size, X dimension (px).
        y_res: [int] Image size, Y dimension (px).
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
        precompute_pointing_corr: [bool] Precompute the pointing correction. False by default. This is used
            to speed up the calculation when the input JD is the same for all data points, e.g. during
            plate solving.
    
    Return:
        (ra_data, dec_data): [tuple of ndarrays]
            
            ra_data: [ndarray] Right ascension of each point (deg).
            dec_data: [ndarray] Declination of each point (deg).
            magnitude_data: [ndarray] Array of meteor's lightcurve apparent magnitudes.
    """

    cdef int i
    cdef double jd, x_img, y_img, r, dx, x_corr, dy, y_corr, r_corr, r_scale
    cdef double x0, y0, xy, a1, a2, k1, k2, k3, k4, k5
    cdef int index_offset
    cdef double radius, theta, sin_t, cos_t
    cdef double ra_ref_now_corr, ra, dec, dec_ref_corr, pos_angle_ref_now_corr

    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] ra_data = np.zeros_like(jd_data)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] dec_data = np.zeros_like(jd_data)

    if precompute_pointing_corr:

        # Correct the pointing for precession (output in radians)
        ra_ref_now_corr, dec_ref_corr, pos_angle_ref_now_corr = pointingCorrection(
            np.mean(jd_data), radians(lat), radians(lon), 
            radians(h0), jd_ref, radians(ra_ref), radians(dec_ref), radians(pos_angle_ref), 
            refraction=refraction
            )


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

        # Choose time and image coordinates
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

        if not precompute_pointing_corr:

            # Correct the pointing for precession (output in radians)
            ra_ref_now_corr, dec_ref_corr, pos_angle_ref_now_corr = pointingCorrection(
                jd, radians(lat), radians(lon), 
                radians(h0), jd_ref, radians(ra_ref), radians(dec_ref), radians(pos_angle_ref), 
                refraction=refraction
                )


        # Radius from FOV centre to sky coordinate
        radius = radians(sqrt(x_corr**2 + y_corr**2))

        # Compute theta - the direction angle between the FOV centre, sky coordinate, and the north 
        #   celestial pole
        theta = (pi/2 - pos_angle_ref_now_corr + atan2(y_corr, x_corr))%(2*pi)


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



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cyAltAzToXY(np.ndarray[FLOAT_TYPE_t, ndim=1] alt_data, np.ndarray[FLOAT_TYPE_t, ndim=1] az_data, \
    double x_res, double y_res, double alt_ref, double az_ref, double rotation_from_horiz, double pix_scale, \
    np.ndarray[FLOAT_TYPE_t, ndim=1] x_poly_fwd, np.ndarray[FLOAT_TYPE_t, ndim=1] y_poly_fwd, \
    str dist_type, bool refraction=True, bool equal_aspect=False, bool force_distortion_centre=False, \
    bool asymmetry_corr=True):
    """
    Convert Azimuth, Altitude to distortion corrected image coordinates.

    Arguments:
        az_data: [ndarray] Array of azimuth (degrees).
        alt_data: [ndarray] Array of altitude (degrees).
        x_res: [int] X resolution of the camera.
        y_res: [int] Y resolution of the camera.
        az_ref: [float] Reference azimuth of the image centre (degrees).
        alt_ref: [float] Reference altitude of the image centre (degrees).
        rotation_from_horiz: [float] Rotation from the horizontal (degrees).
        pix_scale: [float] Image scale (px/deg).
        x_poly_fwd: [ndarray float] Distortion polynomial in X direction for forward mapping.
        y_poly_fwd: [ndarray float] Distortion polynomail in Y direction for forward mapping.
        dist_type: [str] Distortion type. Can be: poly3+radial, radial3, radial4, or radial5.
        
    Keyword arguments:
        refraction: [bool] Apply refraction correction. True by default.
        equal_aspect: [bool] Force the X/Y aspect ratio to be equal. Used only for radial distortion. \
            False by default.
        force_distortion_centre: [bool] Force the distortion centre to the image centre. False by default.
        asymmetry_corr: [bool] Correct the distortion for asymmetry. Only for radial distortion. True by
            default.
    
    Return:
        (x, y): [tuple of ndarrays] Image X and Y coordinates.    """

    cdef int i, j, index_offset
    cdef double x0, y0, xy, a1, a2, k1, k2, k3, k4
    cdef double r, r1, r2, dx, dy, lens_dist, x_corr, y_corr, x_corr1, y_corr1
    cdef double x_img, y_img, x_img1, y_img1, x_img2, y_img2
    cdef double x_img1_est, y_img1_est, x_img2_est, y_img2_est
    cdef double radius, sin_ang, cos_ang, theta, x, y
    cdef double az, alt

    # Init output arrays
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] x_array = np.zeros_like(az_data)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] y_array = np.zeros_like(az_data)

    az_ref = radians(az_ref)
    alt_ref = radians(alt_ref)

    # Correct the reference FOV centre for refraction
    if refraction:
        alt_ref = refractionTrueToApparent(alt_ref)

    # If the radial distortion is used, unpack radial parameters
    if dist_type.startswith("radial"):

        # Index offset for reading distortion parameters. May change as equal aspect or asymmetry correction
        #   is toggled on/off
        index_offset = 0

        # Force the distortion centre to the image centre
        if force_distortion_centre:
            x0 = 0.5/(x_res/2.0)  # 0.5 pixel offset to true center (matching cyRaDecToXY_iter)
            y0 = 0.5/(y_res/2.0)
            index_offset += 2
        else:
            # Read distortion offsets
            x0 = x_poly_fwd[0]
            y0 = x_poly_fwd[1]
            
        # Convert offsets to pixel coordinates
        x0 *= (x_res/2.0)
        y0 *= (y_res/2.0)

        # Aspect ratio
        if equal_aspect:
            xy = 0.0
            index_offset += 1
        else:
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

        # Read distortion coefficients
        if dist_type == "radial3-all":
            k1 = x_poly_fwd[5 - index_offset]
            k2 = x_poly_fwd[6 - index_offset]
            k3 = 0.0
            k4 = 0.0

        elif dist_type == "radial4-all":
            k1 = x_poly_fwd[5 - index_offset]
            k2 = x_poly_fwd[6 - index_offset]
            k3 = x_poly_fwd[7 - index_offset]
            k4 = 0.0

        elif dist_type == "radial5-all":
            k1 = x_poly_fwd[5 - index_offset]
            k2 = x_poly_fwd[6 - index_offset]
            k3 = x_poly_fwd[7 - index_offset]
            k4 = x_poly_fwd[8 - index_offset]

        elif dist_type == "radial3-odd":
            k1 = x_poly_fwd[5 - index_offset]
            k2 = 0.0
            k3 = 0.0
            k4 = 0.0

        elif dist_type == "radial5-odd":
            k1 = x_poly_fwd[5 - index_offset]
            k2 = x_poly_fwd[6 - index_offset]
            k3 = 0.0
            k4 = 0.0

        elif dist_type == "radial7-odd":
            k1 = x_poly_fwd[5 - index_offset]
            k2 = x_poly_fwd[6 - index_offset]
            k3 = x_poly_fwd[7 - index_offset]
            k4 = 0.0

        elif dist_type == "radial9-odd":
            k1 = x_poly_fwd[5 - index_offset]
            k2 = x_poly_fwd[6 - index_offset]
            k3 = x_poly_fwd[7 - index_offset]
            k4 = x_poly_fwd[8 - index_offset]

    # Convert all horizontal coordinates to image coordinates
    for i in range(az_data.shape[0]):

        az = radians(az_data[i])
        alt = radians(alt_data[i])

        # Apply refraction correction
        if refraction:
            alt = refractionTrueToApparent(alt)

        ### Gnomonization of coordinates to image coordinates ###

        # Compute the distance from the FOV centre to the sky coordinate
        radius = radians(angularSeparation(degrees(az), degrees(alt), degrees(az_ref), degrees(alt_ref)))

        # Compute theta - the direction angle between the FOV centre, sky coordinate, and the image vertical
        # sin_ang = cos(alt) * sin(az_ref - az) / sin(radius)
        # cos_ang = (sin(alt) - sin(alt_ref) * cos(radius)) / (cos(alt_ref) * sin(radius))
        # theta = -atan2(sin_ang, cos_ang) + radians(rotation_from_horiz) - pi/2.0

        # Avoid division by zerp
        if radius < 1e-8:
            theta = 0.0
        else:
            sin_ang = cos(alt)*sin(az_ref - az)/sin(radius)
            cos_ang = (sin(alt) - sin(alt_ref)*cos(radius))/(cos(alt_ref)*sin(radius))
            theta = -atan2(sin_ang, cos_ang) + radians(rotation_from_horiz) - pi/2

        # Calculate the standard coordinates
        x_corr = degrees(radius)*cos(theta)*pix_scale
        y_corr = degrees(radius)*sin(theta)*pix_scale

        # Apply polynomial distortion
        if dist_type.startswith("poly3+radial"):

            # Compute the radius from pixel coordinates
            r = sqrt((x_corr - x0)**2 + (y_corr - y0)**2)

            # Calculate the distortion in X direction (using pixel coordinates)
            dx = (x0
                + x_poly_fwd[1]*x_corr
                + x_poly_fwd[2]*y_corr
                + x_poly_fwd[3]*x_corr**2
                + x_poly_fwd[4]*x_corr*y_corr
                + x_poly_fwd[5]*y_corr**2
                + x_poly_fwd[6]*x_corr**3
                + x_poly_fwd[7]*x_corr**2*y_corr
                + x_poly_fwd[8]*x_corr*y_corr**2
                + x_poly_fwd[9]*y_corr**3
                + x_poly_fwd[10]*x_corr*r
                + x_poly_fwd[11]*y_corr*r)
                
            # Calculate the distortion in Y direction (using pixel coordinates)
            dy = (y0
                + y_poly_fwd[1]*x_corr
                + y_poly_fwd[2]*y_corr
                + y_poly_fwd[3]*x_corr**2
                + y_poly_fwd[4]*x_corr*y_corr
                + y_poly_fwd[5]*y_corr**2
                + y_poly_fwd[6]*x_corr**3
                + y_poly_fwd[7]*x_corr**2*y_corr
                + y_poly_fwd[8]*x_corr*y_corr**2
                + y_poly_fwd[9]*y_corr**3
                + y_poly_fwd[10]*y_corr*r
                + y_poly_fwd[11]*x_corr*r)

            # If the 3rd order radial term is used, apply it
            if dist_type.endswith("+radial3") or dist_type.endswith("+radial5"):
                dx += x_poly_fwd[12]*x_corr*r**3
                dy += y_poly_fwd[12]*y_corr*r**3

            # If the 5th order radial term is used, apply it
            if dist_type.endswith("+radial5"):
                dx += x_poly_fwd[13]*x_corr*r**5
                dy += y_poly_fwd[13]*y_corr*r**5

            x_img = x_corr - dx
            y_img = y_corr - dy


        # Apply radial distortion using iterative solver
        elif dist_type.startswith("radial"):
            
            # Initialize the reverse radial iteration loop
            delta_r = 1.0
            j = 0

            # Set initial guess (undistorted coordinates in pixels)
            x_img = x_corr
            y_img = y_corr

            # Iterate to find the distorted position
            while delta_r > 0.01 and j < 100:  # 0.01 pixel tolerance
                j += 1

                # Compute the radius (with aspect ratio and asymmetry, in pixels then normalized)
                r = sqrt((x_img - x0)**2 + ((1.0 + xy)*(y_img - y0))**2)
                r = r + a1*(1.0 + xy)*(y_img - y0)*cos(a2) - a1*(x_img - x0)*sin(a2)
                r = r/(x_res/2.0)  # Normalize to horizontal size

                r_corr = r
                
                # Apply the appropriate radial distortion model
                if dist_type == "radial3-all":
                    r_corr = r + k1*r**2 + k2*r**3

                elif dist_type == "radial4-all":
                    r_corr = r + k1*r**2 + k2*r**3 + k3*r**4

                elif dist_type == "radial5-all":
                    r_corr = r + k1*r**2 + k2*r**3 + k3*r**4 + k4*r**5

                elif dist_type == "radial3-odd":
                    r_corr = r + k1*r**3

                elif dist_type == "radial5-odd":
                    r_corr = r + k1*r**3 + k2*r**5

                elif dist_type == "radial7-odd":
                    r_corr = r + k1*r**3 + k2*r**5 + k3*r**7

                elif dist_type == "radial9-odd":
                    r_corr = r + k1*r**3 + k2*r**5 + k3*r**7 + k4*r**9
                
                # Compute the scaling factor
                if r == 0:
                    r_scale = 0
                else:
                    r_scale = (r_corr/r - 1)
                
                # Stop iterating if distortion is negligible
                if fabs(r_scale) < 1e-8:
                    break
                
                # Compute distortion offsets (matching cyXYToRADec)
                dx = (x_img - x0)*r_scale - x0
                dy = (y_img - y0)*r_scale*(1.0 + xy) - y0*(1.0 + xy) + y_img*xy
                
                # Compute new estimate by inverting: x_corr = x_img + dx
                x_img_est = x_corr - dx
                y_img_est = y_corr - dy

                # Compute distance between current and last guess
                delta_r = sqrt((x_img - x_img_est)**2 + (y_img - y_img_est)**2)

                # Update guess
                x_img = x_img_est
                y_img = y_img_est

        else:
            # No distortion
            x_img = x_corr
            y_img = y_corr

        # Shift to image coordinate system (0,0 at top-left)
        x_array[i] = x_img + x_res/2.0
        y_array[i] = y_img + y_res/2.0

    return x_array, y_array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cyXYToAltAz(np.ndarray[FLOAT_TYPE_t, ndim=1] x_data, \
    np.ndarray[FLOAT_TYPE_t, ndim=1] y_data, double x_res, double y_res, \
    double alt_centre, double az_centre, double rotation_from_horiz, double pix_scale, \
    np.ndarray[FLOAT_TYPE_t, ndim=1] x_poly_fwd, np.ndarray[FLOAT_TYPE_t, ndim=1] y_poly_fwd, \
    str dist_type, bool refraction=True, bool equal_aspect=False, bool force_distortion_centre=False,\
    bool asymmetry_corr=True):
    """
    Arguments:
        x_data: [ndarray] 1D numpy array containing the image column.
        y_data: [ndarray] 1D numpy array containing the image row.
        x_res: [int] Image size, X dimension (px).
        y_res: [int] Image size, Y dimenstion (px).
        az_centre: [float] Reference right ascension of the image centre (degrees).
        alt_centre: [float] Reference declination of the image centre (degrees).
        rotation_from_horiz: [float] Field rotation parameter (degrees).
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
        (alt_data, az_data): [tuple of ndarrays]
            
            alt_data: [ndarray] Altitude of each point (deg).
            az_data: [ndarray] Azimuth of each point (deg).
    """

    cdef int i, index_offset
    cdef double x0, y0, xy, a1, a2, k1, k2, k3, k4
    cdef double r, r1, r2, dx, dy, lens_dist, x_corr, y_corr, x_corr1, y_corr1
    cdef double x_img, y_img, x_img1, y_img1, x_img2, y_img2
    cdef double radius, theta, sin_t, cos_t
    cdef double alt_centre_corr, az, alt

    # Convert the reference pointing direction to radians
    az_centre  = radians(az_centre)
    alt_centre = radians(alt_centre)

    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] az_data = np.zeros_like(x_data)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] alt_data = np.zeros_like(x_data)


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

    # Go through all given data points and convert them from X, Y to Az, Alt
    for i in range(x_data.shape[0]):

        # Choose time and image coordiantes
        x_img = x_data[i]
        y_img = y_data[i]


        ### APPLY DISTORTION CORRECTION ###

        # Normalize image coordinates to the image centre
        x_img = x_img - x_res/2.0
        y_img = y_img - y_res/2.0

        # Apply 3rd order polynomial + one radial term distortion
        if dist_type.startswith("poly3+radial"):

            # Compute the radius
            r = sqrt((x_img - x0)**2 + (y_img - y0)**2)

            # Compute offset in X direction
            dx = (x_poly_fwd[1]*x_img
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
            dy = (y_poly_fwd[1]*x_img
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
            
            x_corr1 = x_img + dx
            y_corr1 = y_img + dy

        # Apply a radial distortion
        elif dist_type.startswith("radial"):

            # Compute the radius from distortion center
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


        ### Convert gnomonic X, Y to Az, Alt ###

        # Radius from FOV centre to sky coordinate
        radius = radians(sqrt(x_corr**2 + y_corr**2))

        # Compute theta - the direction angle between the FOV centre, sky coordinate, and the north 
        #   celestial pole
        theta = (pi/2 - radians(rotation_from_horiz) + atan2(y_corr, x_corr))%(2*pi)

        # Correct the FOV centre for refraction
        if refraction:
            alt_centre_corr = refractionTrueToApparent(alt_centre)

        else:
            alt_centre_corr = alt_centre

        # Compute altitude
        sin_t = sin(alt_centre_corr)*cos(radius) + cos(alt_centre_corr)*sin(radius)*cos(theta)
        alt = atan2(sin_t, sqrt(1 - sin_t**2))

        # Compute azimuth
        sin_t = sin(theta)*sin(radius)/cos(alt)
        cos_t = (cos(radius) - sin(alt)*sin(alt_centre_corr))/(cos(alt)*cos(alt_centre_corr))
        az = (az_centre + atan2(sin_t, cos_t) + 2*pi)%(2*pi)

        # Apply refraction correction
        if refraction:
            alt = refractionApparentToTrue(alt)

        # Convert coordinates to degrees
        az = degrees(az)
        alt = degrees(alt)


        # Assign values to output list
        az_data[i] = az
        alt_data[i] = alt


    return alt_data, az_data

# === XY -> ENU at WGS-84 height (low-elevation safe) ===
@cython.cdivision(True)
cdef inline void geodetic_to_ecef(double lat, double lon, double h,
                                  double* X, double* Y, double* Z):
    cdef double a = 6378137.0
    cdef double f = 1.0/298.257223563
    cdef double e2 = f*(2.0 - f)
    cdef double s = sin(lat), c = cos(lat)
    cdef double cl = cos(lon), sl = sin(lon)
    cdef double N = a / sqrt(1.0 - e2*s*s)
    X[0] = (N + h)*c*cl
    Y[0] = (N + h)*c*sl
    Z[0] = (N*(1.0 - e2) + h)*s

@cython.cdivision(True)
cdef inline void ecef_to_geodetic_bowring(double X, double Y, double Z,
                                          double* lat, double* lon, double* h):
    cdef double a = 6378137.0
    cdef double f = 1.0/298.257223563
    cdef double b = a*(1.0 - f)
    cdef double e2 = f*(2.0 - f)
    cdef double ep2 = (a*a - b*b)/(b*b)
    cdef double p = sqrt(X*X + Y*Y)
    lon[0] = atan2(Y, X)
    cdef double theta = atan2(Z*a, p*b)
    cdef double st = sin(theta), ct = cos(theta)
    lat[0] = atan2(Z + ep2*b*st*st*st, p - e2*a*ct*ct*ct)
    cdef double s = sin(lat[0])
    cdef double N = a / sqrt(1.0 - e2*s*s)
    h[0] = p/cos(lat[0]) - N

@cython.cdivision(True)
cdef inline void R_ecef_from_enu(double lat, double lon, double[:, :] R):
    cdef double sL = sin(lon), cL = cos(lon), sF = sin(lat), cF = cos(lat)
    R[0,0] = -sL;         R[1,0] =  cL;        R[2,0] = 0.0
    R[0,1] = -sF*cL;      R[1,1] = -sF*sL;     R[2,1] =  cF
    R[0,2] =  cF*cL;      R[1,2] =  cF*sL;     R[2,2] =  sF


@cython.boundscheck(False)
@cython.wraparound(False)
def cyXYHttoENU_wgs84(
    np.ndarray[FLOAT_TYPE_t, ndim=1] x_data,
    np.ndarray[FLOAT_TYPE_t, ndim=1] y_data,
    double x_res, double y_res,
    double alt_ref, double az_ref, double rotation_from_horiz, double pix_scale,
    np.ndarray[FLOAT_TYPE_t, ndim=1] x_poly_fwd, np.ndarray[FLOAT_TYPE_t, ndim=1] y_poly_fwd,
    str dist_type,
    # station geodetic & target height (WGS-84)
    double lat_sta_deg, double lon_sta_deg, double h_sta_m, np.ndarray[FLOAT_TYPE_t, ndim=1] ht_wgs84_m,
    # options
    bint refraction=True, bint equal_aspect=False,
    bint force_distortion_centre=False, bint asymmetry_corr=True,
    double min_el_deg=0.0
):
    """
    Pixels (x,y) + WGS-84 height -> ENU (meters).
    Steps (identical normalization to cyXYToAltAz):
      1) center-subtract
      2) UNDISTORT (poly3+radial forward polys, or radial forward model)
      3) gnomonic: R, theta = (pi/2 - rot + atan2(y,x))  → Alt/Az using the Alt/Az spherical form
      4) (optional) convert apparent->true altitude if refraction=True (mirrors your code)
      5) build ENU unit ray (east, north, up)
      6) ECEF ray ∩ WGS-84 ellipsoidal height via bisection → back to ENU meters.
    Returns (E_m[], N_m[], U_m[]). NaNs for rays below min_el_deg or failed intersections.
    """

    # -------- HOISTED DECLARATIONS --------
    cdef Py_ssize_t n, i
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] E, N, U
    # WGS-84 constants
    cdef double a, f, e2, b, ep2
    # Station ECEF + rotation columns (ECEF <- ENU; ENU = R^T (ECEF-C))
    cdef double latS, lonS, sS, cS, Nsta, Xc, Yc, Zc
    cdef double RE0, RE1, RE2, RN0, RN1, RN2, RU0, RU1, RU2
    # Centre & rotation (Alt/Az)
    cdef double A0, h0, rotH
    # Distortion params
    cdef int index_offset
    cdef double x0, y0, xy, a1, a2, k1, k2, k3, k4
    # Per-sample vars
    cdef double x_img, y_img, r, dx, dy, x_corr, y_corr
    cdef double R, theta, A, h, el_gate
    cdef double sin_t, cos_t, sin_ang, cos_ang
    cdef double east, north, up
    cdef double dxe, dye, dze
    # Intersection helpers
    cdef double C2, Cdotd, disc
    cdef double num, den, Rgeo, r_guess
    cdef double s_lo, s_hi, s_mid
    cdef double Xi, Yi, Zi, pval, theta_b, st, ct
    cdef double latP, Ncur, hP, f_lo, f_hi, f_mid
    cdef int it
    # --------------------------------------

    n = x_data.shape[0]
    E = np.empty(n, dtype=FLOAT_TYPE)
    N = np.empty(n, dtype=FLOAT_TYPE)
    U = np.empty(n, dtype=FLOAT_TYPE)

    # --- WGS-84 constants ---
    a  = 6378137.0
    f  = 1.0/298.257223563
    e2 = f*(2.0 - f)
    b  = a*(1.0 - f)
    ep2 = (a*a - b*b)/(b*b)

    # --- Station ECEF ---
    latS = radians(lat_sta_deg)
    lonS = radians(lon_sta_deg)
    sS = sin(latS); cS = cos(latS)
    Nsta = a / sqrt(1.0 - e2*sS*sS)
    Xc = (Nsta + h_sta_m)*cS*cos(lonS)
    Yc = (Nsta + h_sta_m)*cS*sin(lonS)
    Zc = (Nsta*(1.0 - e2) + h_sta_m)*sS

    # ECEF <- ENU columns
    RE0 = -sin(lonS); RE1 =  cos(lonS); RE2 = 0.0
    RN0 = -sS*cos(lonS); RN1 = -sS*sin(lonS); RN2 = cS
    RU0 =  cS*cos(lonS); RU1 =  cS*sin(lonS); RU2 = sS

    # --- Centre & rotation (match cyXYToAltAz) ---
    A0 = radians(az_ref)
    h0 = radians(alt_ref)
    if refraction:
        h0 = refractionTrueToApparent(h0)   # apparent centre for spherical part
    rotH = radians(rotation_from_horiz)
    el_gate = radians(min_el_deg)

    # --- Distortion params (identical unpack to cyXYToAltAz) ---
    index_offset = 0
    if force_distortion_centre:
        x0 = 0.5/(x_res/2.0); y0 = 0.5/(y_res/2.0); index_offset += 2
    else:
        x0 = x_poly_fwd[0]; y0 = x_poly_fwd[1]
    x0 *= (x_res/2.0); y0 *= (y_res/2.0)
    x0 = -x_res/2.0 + (x0 + x_res/2.0)%x_res
    y0 = -y_res/2.0 + (y0 + y_res/2.0)%y_res

    if equal_aspect:
        xy = 0.0; index_offset += 1
    else:
        xy = x_poly_fwd[2 - index_offset]

    if asymmetry_corr:
        a1 = x_poly_fwd[3 - index_offset]
        a2 = (x_poly_fwd[4 - index_offset]*(2*pi))%(2*pi)
    else:
        a1 = 0.0; a2 = 0.0; index_offset += 2

    k1 = k2 = k3 = k4 = 0.0
    if   dist_type == "radial3-all": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]
    elif dist_type == "radial4-all": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]; k3 = x_poly_fwd[7 - index_offset]
    elif dist_type == "radial5-all": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]; k3 = x_poly_fwd[7 - index_offset]; k4 = x_poly_fwd[8 - index_offset]
    elif dist_type == "radial3-odd": k1 = x_poly_fwd[5 - index_offset]
    elif dist_type == "radial5-odd": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]
    elif dist_type == "radial7-odd": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]; k3 = x_poly_fwd[7 - index_offset]
    elif dist_type == "radial9-odd": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]; k3 = x_poly_fwd[7 - index_offset]; k4 = x_poly_fwd[8 - index_offset]

    # --- intersection helpers ---
    C2 = Xc*Xc + Yc*Yc + Zc*Zc
    num = (a*a*cS)*(a*a*cS) + (b*b*sS)*(b*b*sS)
    den = (a*cS)*(a*cS) + (b*sS)*(b*sS)
    Rgeo = sqrt(num/den)
    # Note: r_guess will be set per point in the loop

    for i in range(n):
        # 1) center-subtract
        x_img = x_data[i] - x_res/2.0
        y_img = y_data[i] - y_res/2.0

        # 2) UNDISTORT (IDENTICAL to cyXYToAltAz)
        if dist_type.startswith("poly3+radial"):
            r  = sqrt((x_img - x0)**2 + (y_img - y0)**2)
            dx = (x_poly_fwd[1]*x_img + x_poly_fwd[2]*y_img
                + x_poly_fwd[3]*x_img**2 + x_poly_fwd[4]*x_img*y_img + x_poly_fwd[5]*y_img**2
                + x_poly_fwd[6]*x_img**3 + x_poly_fwd[7]*x_img**2*y_img + x_poly_fwd[8]*x_img*y_img**2 + x_poly_fwd[9]*y_img**3
                + x_poly_fwd[10]*x_img*r + x_poly_fwd[11]*y_img*r)
            dy = (y_poly_fwd[1]*x_img + y_poly_fwd[2]*y_img
                + y_poly_fwd[3]*x_img**2 + y_poly_fwd[4]*x_img*y_img + y_poly_fwd[5]*y_img**2
                + y_poly_fwd[6]*x_img**3 + y_poly_fwd[7]*x_img**2*y_img + y_poly_fwd[8]*x_img*y_img**2 + y_poly_fwd[9]*y_img**3
                + y_poly_fwd[10]*y_img*r + y_poly_fwd[11]*x_img*r)
            x_corr = (x_img + dx)/pix_scale
            y_corr = (y_img + dy)/pix_scale

        elif dist_type.startswith("radial"):
            r  = sqrt((x_img - x0)**2 + ((1.0 + xy)*(y_img - y0))**2)
            r  = r + a1*(1.0 + xy)*(y_img - y0)*cos(a2) - a1*(x_img - x0)*sin(a2)
            r  = r/(x_res/2.0)

            # forward radial model (same as your code)
            r_corr = r
            if   dist_type == "radial3-all": r_corr = r + k1*r**2 + k2*r**3
            elif dist_type == "radial4-all": r_corr = r + k1*r**2 + k2*r**3 + k3*r**4
            elif dist_type == "radial5-all": r_corr = r + k1*r**2 + k2*r**3 + k3*r**4 + k4*r**5
            elif dist_type == "radial3-odd": r_corr = r + k1*r**3
            elif dist_type == "radial5-odd": r_corr = r + k1*r**3 + k2*r**5
            elif dist_type == "radial7-odd": r_corr = r + k1*r**3 + k2*r**5 + k3*r**7
            elif dist_type == "radial9-odd": r_corr = r + k1*r**3 + k2*r**5 + k3*r**7 + k4*r**9

            if r == 0.0:
                dx = dy = 0.0
            else:
                dx = (x_img - x0)*(r_corr/r - 1.0) - x0
                dy = (y_img - y0)*(r_corr/r - 1.0)*(1.0 + xy) - y0*(1.0 + xy) + y_img*xy
            x_corr = (x_img + dx)/pix_scale
            y_corr = (y_img + dy)/pix_scale
        else:
            x_corr = x_img/pix_scale
            y_corr = y_img/pix_scale

        # 3) gnomonic → Alt/Az (IDENTICAL to cyXYToAltAz)
        R = radians(sqrt(x_corr*x_corr + y_corr*y_corr))
        if R < 1e-12:
            A = A0; h = h0
        else:
            theta = (pi/2.0 - rotH + atan2(y_corr, x_corr))%(2*pi)
            # altitude
            sin_t = sin(h0)*cos(R) + cos(h0)*sin(R)*cos(theta)
            h = atan2(sin_t, sqrt(1.0 - sin_t*sin_t))
            # azimuth
            sin_t = sin(theta)*sin(R)/cos(h)
            cos_t = (cos(R) - sin(h)*sin(h0))/(cos(h)*cos(h0))
            A = (A0 + atan2(sin_t, cos_t) + 2*pi)%(2*pi)

        # 4) finish refraction handling same as your code: true altitude for ray
        if refraction:
            h = refractionApparentToTrue(h)
        if h < el_gate:
            E[i]=N[i]=U[i]=np.nan
            continue

        # 5) ENU unit ray
        east  = cos(h)*sin(A)
        north = cos(h)*cos(A)
        up    = sin(h)

        # 6) ECEF ray + intersection with WGS-84 height
        dxe = RE0*east + RN0*north + RU0*up
        dye = RE1*east + RN1*north + RU1*up
        dze = RE2*east + RN2*north + RU2*up

        # bracket with sphere guess
        r_guess = Rgeo + ht_wgs84_m[i] + 1000.0
        C2 = Xc*Xc + Yc*Yc + Zc*Zc
        Cdotd = Xc*dxe + Yc*dye + Zc*dze
        disc  = Cdotd*Cdotd - (C2 - r_guess*r_guess)
        if disc <= 0.0:
            E[i]=N[i]=U[i]=np.nan
            continue
        s_hi = -Cdotd + sqrt(disc)
        s_lo = 0.0
        f_lo = h_sta_m - ht_wgs84_m[i]

        Xi = Xc + s_hi*dxe; Yi = Yc + s_hi*dye; Zi = Zc + s_hi*dze
        pval = sqrt(Xi*Xi + Yi*Yi)
        theta_b = atan2(Zi*a, pval*b); st = sin(theta_b); ct = cos(theta_b)
        latP = atan2(Zi + ep2*b*st*st*st, pval - e2*a*ct*ct*ct)
        Ncur = a / sqrt(1.0 - e2*sin(latP)*sin(latP))
        hP   = pval/cos(latP) - Ncur
        f_hi = hP - ht_wgs84_m[i]

        it = 0
        while f_lo*f_hi > 0.0 and it < 6:
            s_hi *= 1.5
            Xi = Xc + s_hi*dxe; Yi = Yc + s_hi*dye; Zi = Zc + s_hi*dze
            pval = sqrt(Xi*Xi + Yi*Yi)
            theta_b = atan2(Zi*a, pval*b); st = sin(theta_b); ct = cos(theta_b)
            latP = atan2(Zi + ep2*b*st*st*st, pval - e2*a*ct*ct*ct)
            Ncur = a / sqrt(1.0 - e2*sin(latP)*sin(latP))
            hP   = pval/cos(latP) - Ncur
            f_hi = hP - ht_wgs84_m[i]
            it += 1

        if f_lo*f_hi > 0.0:
            E[i]=N[i]=U[i]=np.nan
            continue

        for it in range(20):
            s_mid = 0.5*(s_lo + s_hi)
            Xi = Xc + s_mid*dxe; Yi = Yc + s_mid*dye; Zi = Zc + s_mid*dze
            pval = sqrt(Xi*Xi + Yi*Yi)
            theta_b = atan2(Zi*a, pval*b); st = sin(theta_b); ct = cos(theta_b)
            latP = atan2(Zi + ep2*b*st*st*st, pval - e2*a*ct*ct*ct)
            Ncur = a / sqrt(1.0 - e2*sin(latP)*sin(latP))
            hP   = pval/cos(latP) - Ncur
            f_mid = hP - ht_wgs84_m[i]
            if f_lo*f_mid <= 0.0:
                s_hi = s_mid; f_hi = f_mid
            else:
                s_lo = s_mid; f_lo = f_mid
            if fabs(f_mid) < 1e-3:
                break

        # return ENU meters relative to station
        dX = Xi - Xc; dY = Yi - Yc; dZ = Zi - Zc
        E[i] = RE0*dX + RE1*dY + RE2*dZ
        N[i] = RN0*dX + RN1*dY + RN2*dZ
        U[i] = RU0*dX + RU1*dY + RU2*dZ

    return E, N, U




@cython.boundscheck(False)
@cython.wraparound(False)
def cyGeoToXY_wgs84_iter(
    np.ndarray[FLOAT_TYPE_t, ndim=1] lat_geo_deg,   # target geodetic lat (deg)
    np.ndarray[FLOAT_TYPE_t, ndim=1] lon_geo_deg,   # target geodetic lon (deg)
    np.ndarray[FLOAT_TYPE_t, ndim=1] h_geo_m,       # target WGS-84 height (m)
    # plate / camera (match cyAltAzToXY parameter order/names)
    double x_res, double y_res,
    double alt_ref, double az_ref, double rotation_from_horiz, double pix_scale,
    np.ndarray[FLOAT_TYPE_t, ndim=1] x_poly_fwd, np.ndarray[FLOAT_TYPE_t, ndim=1] y_poly_fwd,
    str dist_type,
    # station geodetic
    double lat_sta_deg, double lon_sta_deg, double h_sta_m,
    # options
    bint refraction=True, bint equal_aspect=False,
    bint force_distortion_centre=False, bint asymmetry_corr=True,
    double min_el_deg=0.0
):
    """
    GEO (lat,lon,h) -> image (x,y), using the SAME gnomonic + distortion flow as cyAltAzToXY:
      GEO -> ECEF -> ENU -> Alt/Az (apparent if refraction=True) ->
      gnomonic (radius/sin_ang/cos_ang/theta) -> forward poly OR iterative radial -> pixels.
    """

    # ---------- HOISTED DECLARATIONS ----------
    cdef Py_ssize_t n, i
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] x_array, y_array

    # WGS-84 constants
    cdef double a, f, e2

    # Station ECEF + rotation columns (ECEF <- ENU, columns are E,N,U in ECEF)
    cdef double latS, lonS, sS, cS, Nsta, Xc, Yc, Zc
    cdef double RE0, RE1, RE2, RN0, RN1, RN2, RU0, RU1, RU2

    # Centre/rotation like cyAltAzToXY
    cdef double A0, h0, rotH

    # Distortion params (identical unpack)
    cdef int index_offset
    cdef double x0, y0, xy, a1, a2, k1, k2, k3, k4

    # Per-sample variables
    cdef double latT, lonT, hT, sT, cT, NT, Xt, Yt, Zt
    cdef double dX, dY, dZ, E, Nn, U
    cdef double az, alt, radius, theta, sin_ang, cos_ang, x_corr, y_corr
    cdef double r, dx, dy, x_img, y_img
    cdef double el_gate

    # Iterative radial loop temps
    cdef double delta_r, r_corr, r_scale, x_est, y_est
    cdef int j
    # ------------------------------------------

    n = lat_geo_deg.shape[0]
    x_array = np.zeros(n, dtype=FLOAT_TYPE)
    y_array = np.zeros(n, dtype=FLOAT_TYPE)

    # WGS-84 constants
    a  = 6378137.0
    f  = 1.0/298.257223563
    e2 = f*(2.0 - f)

    # Station ECEF
    latS = radians(lat_sta_deg)
    lonS = radians(lon_sta_deg)
    sS = sin(latS); cS = cos(latS)
    Nsta = a / sqrt(1.0 - e2*sS*sS)
    Xc = (Nsta + h_sta_m)*cS*cos(lonS)
    Yc = (Nsta + h_sta_m)*cS*sin(lonS)
    Zc = (Nsta*(1.0 - e2) + h_sta_m)*sS

    # ECEF <- ENU rotation columns (so ENU = R^T * (ECEF-C))
    RE0 = -sin(lonS); RE1 =  cos(lonS); RE2 = 0.0
    RN0 = -sS*cos(lonS); RN1 = -sS*sin(lonS); RN2 = cS
    RU0 =  cS*cos(lonS); RU1 =  cS*sin(lonS); RU2 = sS

    # Centre/rotation (exactly like cyAltAzToXY)
    A0 = radians(az_ref)
    h0 = radians(alt_ref)
    if refraction:
        h0 = refractionTrueToApparent(h0)   # apparent centre for spherical step
    rotH = radians(rotation_from_horiz)

    # Distortion params (same unpack as cyAltAzToXY)
    index_offset = 0
    if force_distortion_centre:
        x0 = 0.5/(x_res/2.0); y0 = 0.5/(y_res/2.0); index_offset += 2
    else:
        x0 = x_poly_fwd[0]; y0 = x_poly_fwd[1]
    x0 *= (x_res/2.0); y0 *= (y_res/2.0)
    x0 = -x_res/2.0 + (x0 + x_res/2.0)%x_res
    y0 = -y_res/2.0 + (y0 + y_res/2.0)%y_res

    if equal_aspect:
        xy = 0.0; index_offset += 1
    else:
        xy = x_poly_fwd[2 - index_offset]

    if asymmetry_corr:
        a1 = x_poly_fwd[3 - index_offset]
        a2 = (x_poly_fwd[4 - index_offset]*(2*pi))%(2*pi)
    else:
        a1 = 0.0; a2 = 0.0; index_offset += 2

    k1 = k2 = k3 = k4 = 0.0
    if   dist_type == "radial3-all": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]
    elif dist_type == "radial4-all": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]; k3 = x_poly_fwd[7 - index_offset]
    elif dist_type == "radial5-all": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]; k3 = x_poly_fwd[7 - index_offset]; k4 = x_poly_fwd[8 - index_offset]
    elif dist_type == "radial3-odd": k1 = x_poly_fwd[5 - index_offset]
    elif dist_type == "radial5-odd": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]
    elif dist_type == "radial7-odd": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]; k3 = x_poly_fwd[7 - index_offset]
    elif dist_type == "radial9-odd": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]; k3 = x_poly_fwd[7 - index_offset]; k4 = x_poly_fwd[8 - index_offset]

    el_gate = radians(min_el_deg)

    # Main loop
    for i in range(n):
        # GEO -> ECEF
        latT = radians(lat_geo_deg[i]); lonT = radians(lon_geo_deg[i]); hT = h_geo_m[i]
        sT = sin(latT); cT = cos(latT)
        NT = a / sqrt(1.0 - e2*sT*sT)
        Xt = (NT + hT)*cT*cos(lonT)
        Yt = (NT + hT)*cT*sin(lonT)
        Zt = (NT*(1.0 - e2) + hT)*sT

        # ECEF -> ENU (using columns; ENU = R^T*(ECEF - C))
        dX = Xt - Xc; dY = Yt - Yc; dZ = Zt - Zc
        E  = RE0*dX + RE1*dY + RE2*dZ
        Nn = RN0*dX + RN1*dY + RN2*dZ
        U  = RU0*dX + RU1*dY + RU2*dZ

        # ENU -> Alt/Az (apparent if refraction=True)
        az  = atan2(E, Nn)
        if az < 0.0: az += 2*pi
        alt = atan2(U, sqrt(E*E + Nn*Nn))
        if refraction:
            alt = refractionTrueToApparent(alt)
        if alt < el_gate:
            x_array[i] = np.nan; y_array[i] = np.nan
            continue

        # Gnomonic (IDENTICAL to cyAltAzToXY)
        radius = radians(angularSeparation(degrees(az), degrees(alt), degrees(A0), degrees(h0)))
        if radius < 1e-8:
            theta = 0.0
        else:
            # Alt/Az form you use
            # sin_ang =  cos(alt) * sin(az_ref - az) / sin(radius)
            # cos_ang = (sin(alt) - sin(alt_ref) * cos(radius)) / (cos(alt_ref) * sin(radius))
            # theta   = -atan2(sin_ang, cos_ang) + rotH - pi/2.0
            sin_ang =  cos(alt) * sin(A0 - az) / sin(radius)
            cos_ang = (sin(alt) - sin(h0)*cos(radius)) / (cos(h0)*sin(radius))
            theta = -atan2(sin_ang, cos_ang) + rotH - pi/2.0

        x_corr = degrees(radius)*cos(theta)*pix_scale
        y_corr = degrees(radius)*sin(theta)*pix_scale

        # Distortion (match cyAltAzToXY exactly)
        if dist_type.startswith("poly3+radial"):
            r  = sqrt((x_corr - x0)**2 + (y_corr - y0)**2)

            dx = (x0
                + x_poly_fwd[1]*x_corr + x_poly_fwd[2]*y_corr
                + x_poly_fwd[3]*x_corr**2 + x_poly_fwd[4]*x_corr*y_corr + x_poly_fwd[5]*y_corr**2
                + x_poly_fwd[6]*x_corr**3 + x_poly_fwd[7]*x_corr**2*y_corr + x_poly_fwd[8]*x_corr*y_corr**2 + x_poly_fwd[9]*y_corr**3
                + x_poly_fwd[10]*x_corr*r + x_poly_fwd[11]*y_corr*r)

            dy = (y0
                + y_poly_fwd[1]*x_corr + y_poly_fwd[2]*y_corr
                + y_poly_fwd[3]*x_corr**2 + y_poly_fwd[4]*x_corr*y_corr + y_poly_fwd[5]*y_corr**2
                + y_poly_fwd[6]*x_corr**3 + y_poly_fwd[7]*x_corr**2*y_corr + y_poly_fwd[8]*x_corr*y_corr**2 + y_poly_fwd[9]*y_corr**3
                + y_poly_fwd[10]*y_corr*r + y_poly_fwd[11]*x_corr*r)

            if dist_type.endswith("+radial3") or dist_type.endswith("+radial5"):
                dx += x_poly_fwd[12]*x_corr*r**3
                dy += y_poly_fwd[12]*y_corr*r**3
            if dist_type.endswith("+radial5"):
                dx += x_poly_fwd[13]*x_corr*r**5
                dy += y_poly_fwd[13]*y_corr*r**5

            x_img = x_corr - dx
            y_img = y_corr - dy

        elif dist_type.startswith("radial"):
            # iterative radial (same tolerance and math as your cyAltAzToXY)
            delta_r = 1.0
            j = 0
            x_img = x_corr
            y_img = y_corr
            while (delta_r > 0.01) and (j < 100):   # ~0.01 px tolerance
                j += 1

                r = sqrt((x_img - x0)**2 + ((1.0 + xy)*(y_img - y0))**2)
                r = r + a1*(1.0 + xy)*(y_img - y0)*cos(a2) - a1*(x_img - x0)*sin(a2)
                r = r/(x_res/2.0)

                # forward radial model
                r_corr = r
                if   dist_type == "radial3-all": r_corr = r + k1*r**2 + k2*r**3
                elif dist_type == "radial4-all": r_corr = r + k1*r**2 + k2*r**3 + k3*r**4
                elif dist_type == "radial5-all": r_corr = r + k1*r**2 + k2*r**3 + k3*r**4 + k4*r**5
                elif dist_type == "radial3-odd": r_corr = r + k1*r**3
                elif dist_type == "radial5-odd": r_corr = r + k1*r**3 + k2*r**5
                elif dist_type == "radial7-odd": r_corr = r + k1*r**3 + k2*r**5 + k3*r**7
                elif dist_type == "radial9-odd": r_corr = r + k1*r**3 + k2*r**5 + k3*r**7 + k4*r**9

                if r == 0.0:
                    r_scale = 0.0
                else:
                    r_scale = (r_corr/r - 1.0)

                dx = (x_img - x0)*r_scale - x0
                dy = (y_img - y0)*r_scale*(1.0 + xy) - y0*(1.0 + xy) + y_img*xy

                x_est = x_corr - dx
                y_est = y_corr - dy
                delta_r = sqrt((x_img - x_est)**2 + (y_img - y_est)**2)
                x_img = x_est; y_img = y_est
        else:
            x_img = x_corr
            y_img = y_corr

        # to image coordinates
        x_array[i] = x_img + x_res/2.0
        y_array[i] = y_img + y_res/2.0

    return x_array, y_array



@cython.boundscheck(False)
@cython.wraparound(False)
def cyENUToXY_iter(
    np.ndarray[FLOAT_TYPE_t, ndim=1] E_m,   # ENU east  (m)
    np.ndarray[FLOAT_TYPE_t, ndim=1] N_m,   # ENU north (m)
    np.ndarray[FLOAT_TYPE_t, ndim=1] U_m,   # ENU up    (m)
    double x_res, double y_res,
    double alt_ref, double az_ref, double rotation_from_horiz, double pix_scale,
    np.ndarray[FLOAT_TYPE_t, ndim=1] x_poly_fwd, np.ndarray[FLOAT_TYPE_t, ndim=1] y_poly_fwd,
    str dist_type,
    bint refraction=True, bint equal_aspect=False,
    bint force_distortion_centre=False, bint asymmetry_corr=True,
    double min_el_deg=0.0
):
    """
    ENU (meters) -> image (x,y), using the SAME spherical/gnomonic + distortion flow as cyAltAzToXY.
    """

    # -------- HOISTED DECLARATIONS --------
    cdef Py_ssize_t n, i
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] x_array, y_array
    cdef double A0, h0, rotH
    cdef int index_offset
    cdef double x0, y0, xy, a1, a2, k1, k2, k3, k4
    cdef double A, h, el_gate
    cdef double radius, theta, x_corr, y_corr
    cdef double r, dx, dy, x_img, y_img
    cdef double sin_ang, cos_ang
    # iterative radial temps
    cdef double delta_r, r_corr, r_scale, x_est, y_est
    cdef int j
    # --------------------------------------

    n = E_m.shape[0]
    x_array = np.zeros(n, dtype=FLOAT_TYPE)
    y_array = np.zeros(n, dtype=FLOAT_TYPE)

    # Centre & rotation (match cyAltAzToXY)
    A0 = radians(az_ref)
    h0 = radians(alt_ref)
    if refraction:
        h0 = refractionTrueToApparent(h0)   # apparent centre for spherical math
    rotH = radians(rotation_from_horiz)
    el_gate = radians(min_el_deg)

    # Distortion params (identical unpack to cyAltAzToXY)
    index_offset = 0
    if force_distortion_centre:
        x0 = 0.5/(x_res/2.0); y0 = 0.5/(y_res/2.0); index_offset += 2
    else:
        x0 = x_poly_fwd[0]; y0 = x_poly_fwd[1]
    x0 *= (x_res/2.0); y0 *= (y_res/2.0)
    x0 = -x_res/2.0 + (x0 + x_res/2.0)%x_res
    y0 = -y_res/2.0 + (y0 + y_res/2.0)%y_res

    if equal_aspect:
        xy = 0.0; index_offset += 1
    else:
        xy = x_poly_fwd[2 - index_offset]

    if asymmetry_corr:
        a1 = x_poly_fwd[3 - index_offset]
        a2 = (x_poly_fwd[4 - index_offset]*(2*pi))%(2*pi)
    else:
        a1 = 0.0; a2 = 0.0; index_offset += 2

    k1 = k2 = k3 = k4 = 0.0
    if   dist_type == "radial3-all": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]
    elif dist_type == "radial4-all": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]; k3 = x_poly_fwd[7 - index_offset]
    elif dist_type == "radial5-all": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]; k3 = x_poly_fwd[7 - index_offset]; k4 = x_poly_fwd[8 - index_offset]
    elif dist_type == "radial3-odd": k1 = x_poly_fwd[5 - index_offset]
    elif dist_type == "radial5-odd": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]
    elif dist_type == "radial7-odd": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]; k3 = x_poly_fwd[7 - index_offset]
    elif dist_type == "radial9-odd": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]; k3 = x_poly_fwd[7 - index_offset]; k4 = x_poly_fwd[8 - index_offset]

    for i in range(n):
        # ENU -> Alt/Az (apparent if refraction=True)
        A = atan2(E_m[i], N_m[i])
        if A < 0.0: A += 2*pi
        h = atan2(U_m[i], sqrt(E_m[i]*E_m[i] + N_m[i]*N_m[i]))
        if refraction:
            h = refractionTrueToApparent(h)
        if h < el_gate:
            x_array[i] = np.nan; y_array[i] = np.nan
            continue

        # Gnomonic (IDENTICAL to cyAltAzToXY)
        radius = radians(angularSeparation(degrees(A), degrees(h), degrees(A0), degrees(h0)))
        if radius < 1e-8:
            theta = 0.0
        else:
            sin_ang =  cos(h) * sin(A0 - A) / sin(radius)
            cos_ang = (sin(h) - sin(h0) * cos(radius)) / (cos(h0) * sin(radius))
            theta   = -atan2(sin_ang, cos_ang) + rotH - pi/2.0

        x_corr = degrees(radius)*cos(theta)*pix_scale
        y_corr = degrees(radius)*sin(theta)*pix_scale

        # Distortion (same branches/tolerance as cyAltAzToXY)
        if dist_type.startswith("poly3+radial"):
            r  = sqrt((x_corr - x0)**2 + (y_corr - y0)**2)

            dx = (x0
                + x_poly_fwd[1]*x_corr + x_poly_fwd[2]*y_corr
                + x_poly_fwd[3]*x_corr**2 + x_poly_fwd[4]*x_corr*y_corr + x_poly_fwd[5]*y_corr**2
                + x_poly_fwd[6]*x_corr**3 + x_poly_fwd[7]*x_corr**2*y_corr + x_poly_fwd[8]*x_corr*y_corr**2 + x_poly_fwd[9]*y_corr**3
                + x_poly_fwd[10]*x_corr*r + x_poly_fwd[11]*y_corr*r)

            dy = (y0
                + y_poly_fwd[1]*x_corr + y_poly_fwd[2]*y_corr
                + y_poly_fwd[3]*x_corr**2 + y_poly_fwd[4]*x_corr*y_corr + y_poly_fwd[5]*y_corr**2
                + y_poly_fwd[6]*x_corr**3 + y_poly_fwd[7]*x_corr**2*y_corr + y_poly_fwd[8]*x_corr*y_corr**2 + y_poly_fwd[9]*y_corr**3
                + y_poly_fwd[10]*y_corr*r + y_poly_fwd[11]*x_corr*r)

            if dist_type.endswith("+radial3") or dist_type.endswith("+radial5"):
                dx += x_poly_fwd[12]*x_corr*r**3
                dy += y_poly_fwd[12]*y_corr*r**3
            if dist_type.endswith("+radial5"):
                dx += x_poly_fwd[13]*x_corr*r**5
                dy += y_poly_fwd[13]*y_corr*r**5

            x_img = x_corr - dx
            y_img = y_corr - dy

        elif dist_type.startswith("radial"):
            delta_r = 1.0
            j = 0
            x_img = x_corr
            y_img = y_corr
            while (delta_r > 0.01) and (j < 100):   # ~0.01 px tolerance
                j += 1

                r = sqrt((x_img - x0)**2 + ((1.0 + xy)*(y_img - y0))**2)
                r = r + a1*(1.0 + xy)*(y_img - y0)*cos(a2) - a1*(x_img - x0)*sin(a2)
                r = r/(x_res/2.0)

                r_corr = r
                if   dist_type == "radial3-all": r_corr = r + k1*r**2 + k2*r**3
                elif dist_type == "radial4-all": r_corr = r + k1*r**2 + k2*r**3 + k3*r**4
                elif dist_type == "radial5-all": r_corr = r + k1*r**2 + k2*r**3 + k3*r**4 + k4*r**5
                elif dist_type == "radial3-odd": r_corr = r + k1*r**3
                elif dist_type == "radial5-odd": r_corr = r + k1*r**3 + k2*r**5
                elif dist_type == "radial7-odd": r_corr = r + k1*r**3 + k2*r**5 + k3*r**7
                elif dist_type == "radial9-odd": r_corr = r + k1*r**3 + k2*r**5 + k3*r**7 + k4*r**9

                if r == 0.0:
                    r_scale = 0.0
                else:
                    r_scale = (r_corr/r - 1.0)

                dx = (x_img - x0)*r_scale - x0
                dy = (y_img - y0)*r_scale*(1.0 + xy) - y0*(1.0 + xy) + y_img*xy

                x_est = x_corr - dx
                y_est = y_corr - dy
                delta_r = sqrt((x_img - x_est)**2 + (y_img - y_est)**2)
                x_img = x_est; y_img = y_est
        else:
            x_img = x_corr
            y_img = y_corr

        x_array[i] = x_img + x_res/2.0
        y_array[i] = y_img + y_res/2.0

    return x_array, y_array


@cython.boundscheck(False)
@cython.wraparound(False)
def cyENHtToXY_iter(
    np.ndarray[FLOAT_TYPE_t, ndim=1] E_m,    # ENU east  (m)
    np.ndarray[FLOAT_TYPE_t, ndim=1] N_m,    # ENU north (m)
    np.ndarray[FLOAT_TYPE_t, ndim=1] Ht_m,   # target WGS-84 ellipsoidal height PER POINT (m)
    double x_res, double y_res,
    double alt_ref, double az_ref, double rotation_from_horiz, double pix_scale,
    np.ndarray[FLOAT_TYPE_t, ndim=1] x_poly_fwd, np.ndarray[FLOAT_TYPE_t, ndim=1] y_poly_fwd,
    str dist_type,
    double lat_sta_deg, double lon_sta_deg, double h_sta_m,
    bint refraction=True, bint equal_aspect=False,
    bint force_distortion_centre=False, bint asymmetry_corr=True,
    double min_el_deg=0.0
):
    """
    (E, N, h_ellip[i]) -> XY.
    Solves U per point so that geodetic height equals Ht_m[i], then Alt/Az -> gnomonic -> forward distortion.
    Matches cyAltAzToXY normalization/flow. Iteration only in radial forward branch.
    """

    # -------- HOISTED DECLARATIONS --------
    cdef Py_ssize_t n, i
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] x_array, y_array

    # WGS-84 constants
    cdef double a, f, e2, b, ep2

    # station ECEF + columns of (ECEF <- ENU)
    cdef double latS, lonS, sS, cS, Nsta, Xc, Yc, Zc
    cdef double RE0, RE1, RE2, RN0, RN1, RN2, RU0, RU1, RU2

    # center/rotation (Alt/Az)
    cdef double A0, h0, rotH, el_gate

    # forward distortion params
    cdef int index_offset
    cdef double x0, y0, xy, a1, a2, k1, k2, k3, k4

    # per-point temps
    cdef double E, Nn, U
    cdef double dxe_base, dye_base, dze_base
    cdef double U_lo, U_hi, U_mid, f_lo, f_hi, f_mid
    cdef double Xi, Yi, Zi, pval, theta_b, st, ct, latP, Ncur, hP

    cdef double A, h, radius, theta, sin_ang, cos_ang
    cdef double x_corr, y_corr, r, dx, dy, x_img, y_img

    cdef int it, j
    cdef double delta_r, r_corr, r_scale, x_est, y_est
    # --------------------------------------

    n = E_m.shape[0]
    x_array = np.zeros(n, dtype=FLOAT_TYPE)
    y_array = np.zeros(n, dtype=FLOAT_TYPE)

    # WGS-84
    a  = 6378137.0
    f  = 1.0/298.257223563
    e2 = f*(2.0 - f)
    b  = a*(1.0 - f)
    ep2 = (a*a - b*b)/(b*b)

    # station ECEF
    latS = radians(lat_sta_deg); lonS = radians(lon_sta_deg)
    sS = sin(latS); cS = cos(latS)
    Nsta = a / sqrt(1.0 - e2*sS*sS)
    Xc = (Nsta + h_sta_m)*cS*cos(lonS)
    Yc = (Nsta + h_sta_m)*cS*sin(lonS)
    Zc = (Nsta*(1.0 - e2) + h_sta_m)*sS

    # ECEF <- ENU (columns are E,N,U in ECEF)
    RE0 = -sin(lonS); RE1 =  cos(lonS); RE2 = 0.0
    RN0 = -sS*cos(lonS); RN1 = -sS*sin(lonS); RN2 = cS
    RU0 =  cS*cos(lonS); RU1 =  cS*sin(lonS); RU2 = sS

    # center/rotation like cyAltAzToXY
    A0 = radians(az_ref)
    h0 = radians(alt_ref)
    if refraction:
        h0 = refractionTrueToApparent(h0)
    rotH = radians(rotation_from_horiz)
    el_gate = radians(min_el_deg)

    # forward distortion params (offsets from x_poly_fwd)
    index_offset = 0
    if force_distortion_centre:
        x0 = 0.5/(x_res/2.0); y0 = 0.5/(y_res/2.0); index_offset += 2
    else:
        x0 = x_poly_fwd[0]; y0 = x_poly_fwd[1]
    x0 *= (x_res/2.0); y0 *= (y_res/2.0)
    x0 = -x_res/2.0 + (x0 + x_res/2.0)%x_res
    y0 = -y_res/2.0 + (y0 + y_res/2.0)%y_res

    if equal_aspect:
        xy = 0.0; index_offset += 1
    else:
        xy = x_poly_fwd[2 - index_offset]

    if asymmetry_corr:
        a1 = x_poly_fwd[3 - index_offset]
        a2 = (x_poly_fwd[4 - index_offset]*(2*pi))%(2*pi)
    else:
        a1 = 0.0; a2 = 0.0; index_offset += 2

    k1 = k2 = k3 = k4 = 0.0
    if   dist_type == "radial3-all": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]
    elif dist_type == "radial4-all": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]; k3 = x_poly_fwd[7 - index_offset]
    elif dist_type == "radial5-all": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]; k3 = x_poly_fwd[7 - index_offset]; k4 = x_poly_fwd[8 - index_offset]
    elif dist_type == "radial3-odd": k1 = x_poly_fwd[5 - index_offset]
    elif dist_type == "radial5-odd": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]
    elif dist_type == "radial7-odd": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]; k3 = x_poly_fwd[7 - index_offset]
    elif dist_type == "radial9-odd": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]; k3 = x_poly_fwd[7 - index_offset]; k4 = x_poly_fwd[8 - index_offset]

    for i in range(n):
        E = E_m[i]; Nn = N_m[i]

        # 1) Build base ECEF direction for this (E,N), then 1-D solve U to hit Ht_m[i]
        dxe_base = RE0*E + RN0*Nn
        dye_base = RE1*E + RN1*Nn
        dze_base = RE2*E + RN2*Nn

        # bracket U (m). Start with a broad span.
        U_lo = -50000.0
        U_hi =  400000.0

        # f(U_lo)
        Xi = Xc + dxe_base + RU0*U_lo; Yi = Yc + dye_base + RU1*U_lo; Zi = Zc + dze_base + RU2*U_lo
        pval = sqrt(Xi*Xi + Yi*Yi)
        theta_b = atan2(Zi*a, pval*b); st = sin(theta_b); ct = cos(theta_b)
        latP = atan2(Zi + ep2*b*st*st*st, pval - e2*a*ct*ct*ct)
        Ncur = a / sqrt(1.0 - e2*sin(latP)*sin(latP))
        hP   = pval/cos(latP) - Ncur
        f_lo = hP - Ht_m[i]

        # f(U_hi)
        Xi = Xc + dxe_base + RU0*U_hi; Yi = Yc + dye_base + RU1*U_hi; Zi = Zc + dze_base + RU2*U_hi
        pval = sqrt(Xi*Xi + Yi*Yi)
        theta_b = atan2(Zi*a, pval*b); st = sin(theta_b); ct = cos(theta_b)
        latP = atan2(Zi + ep2*b*st*st*st, pval - e2*a*ct*ct*ct)
        Ncur = a / sqrt(1.0 - e2*sin(latP)*sin(latP))
        hP   = pval/cos(latP) - Ncur
        f_hi = hP - Ht_m[i]

        it = 0
        while f_lo*f_hi > 0.0 and it < 8:
            if fabs(f_lo) < fabs(f_hi):
                U_lo -= 0.5*(U_hi - U_lo)
            else:
                U_hi += 0.5*(U_hi - U_lo)
            Xi = Xc + dxe_base + RU0*U_hi; Yi = Yc + dye_base + RU1*U_hi; Zi = Zc + dze_base + RU2*U_hi
            pval = sqrt(Xi*Xi + Yi*Yi)
            theta_b = atan2(Zi*a, pval*b); st = sin(theta_b); ct = cos(theta_b)
            latP = atan2(Zi + ep2*b*st*st*st, pval - e2*a*ct*ct*ct)
            Ncur = a / sqrt(1.0 - e2*sin(latP)*sin(latP))
            hP   = pval/cos(latP) - Ncur
            f_hi = hP - Ht_m[i]
            it += 1

        for it in range(20):
            U_mid = 0.5*(U_lo + U_hi)
            Xi = Xc + dxe_base + RU0*U_mid; Yi = Yc + dye_base + RU1*U_mid; Zi = Zc + dze_base + RU2*U_mid
            pval = sqrt(Xi*Xi + Yi*Yi)
            theta_b = atan2(Zi*a, pval*b); st = sin(theta_b); ct = cos(theta_b)
            latP = atan2(Zi + ep2*b*st*st*st, pval - e2*a*ct*ct*ct)
            Ncur = a / sqrt(1.0 - e2*sin(latP)*sin(latP))
            hP   = pval/cos(latP) - Ncur
            f_mid = hP - Ht_m[i]
            if f_lo*f_mid <= 0.0:
                U_hi = U_mid; f_hi = f_mid
            else:
                U_lo = U_mid; f_lo = f_mid
            if fabs(f_mid) < 1e-3:   # ~1 mm height
                break

        U = 0.5*(U_lo + U_hi)

        # 2) ENU -> Alt/Az (apparent if refraction=True)
        A = atan2(E, Nn)
        if A < 0.0: A += 2*pi
        h = atan2(U, sqrt(E*E + Nn*Nn))
        if refraction:
            h = refractionTrueToApparent(h)
        if h < el_gate:
            x_array[i] = np.nan; y_array[i] = np.nan
            continue

        # 3) gnomonic (match cyAltAzToXY)
        radius = radians(angularSeparation(degrees(A), degrees(h), degrees(A0), degrees(h0)))
        if radius < 1e-8:
            theta = 0.0
        else:
            # keep same sign convention you use elsewhere
            sin_ang =  cos(h) * sin(A0 - A) / sin(radius)
            cos_ang = (sin(h) - sin(h0) * cos(radius)) / (cos(h0) * sin(radius))
            theta   = -atan2(sin_ang, cos_ang) + rotH - pi/2.0

        x_corr = degrees(radius)*cos(theta)*pix_scale
        y_corr = degrees(radius)*sin(theta)*pix_scale

        # 4) forward distortion (same as cyAltAzToXY)
        if dist_type.startswith("poly3+radial"):
            r  = sqrt((x_corr - x0)**2 + (y_corr - y0)**2)

            dx = (x0
                + x_poly_fwd[1]*x_corr + x_poly_fwd[2]*y_corr
                + x_poly_fwd[3]*x_corr**2 + x_poly_fwd[4]*x_corr*y_corr + x_poly_fwd[5]*y_corr**2
                + x_poly_fwd[6]*x_corr**3 + x_poly_fwd[7]*x_corr**2*y_corr + x_poly_fwd[8]*x_corr*y_corr**2 + x_poly_fwd[9]*y_corr**3
                + x_poly_fwd[10]*x_corr*r + x_poly_fwd[11]*y_corr*r)

            dy = (y0
                + y_poly_fwd[1]*x_corr + y_poly_fwd[2]*y_corr
                + y_poly_fwd[3]*x_corr**2 + y_poly_fwd[4]*x_corr*y_corr + y_poly_fwd[5]*y_corr**2
                + y_poly_fwd[6]*x_corr**3 + y_poly_fwd[7]*x_corr**2*y_corr + y_poly_fwd[8]*x_corr*y_corr**2 + y_poly_fwd[9]*y_corr**3
                + y_poly_fwd[10]*y_corr*r + y_poly_fwd[11]*x_corr*r)

            if dist_type.endswith("+radial3") or dist_type.endswith("+radial5"):
                dx += x_poly_fwd[12]*x_corr*r**3
                dy += y_poly_fwd[12]*y_corr*r**3
            if dist_type.endswith("+radial5"):
                dx += x_poly_fwd[13]*x_corr*r**5
                dy += y_poly_fwd[13]*y_corr*r**5

            x_img = x_corr - dx
            y_img = y_corr - dy

        elif dist_type.startswith("radial"):
            delta_r = 1.0
            j = 0
            x_img = x_corr; y_img = y_corr
            while (delta_r > 0.01) and (j < 100):
                j += 1
                r = sqrt((x_img - x0)**2 + ((1.0 + xy)*(y_img - y0))**2)
                r = r + a1*(1.0 + xy)*(y_img - y0)*cos(a2) - a1*(x_img - x0)*sin(a2)
                r = r/(x_res/2.0)

                r_corr = r
                if   dist_type == "radial3-all": r_corr = r + k1*r**2 + k2*r**3
                elif dist_type == "radial4-all": r_corr = r + k1*r**2 + k2*r**3 + k3*r**4
                elif dist_type == "radial5-all": r_corr = r + k1*r**2 + k2*r**3 + k3*r**4 + k4*r**5
                elif dist_type == "radial3-odd": r_corr = r + k1*r**3
                elif dist_type == "radial5-odd": r_corr = r + k1*r**3 + k2*r**5
                elif dist_type == "radial7-odd": r_corr = r + k1*r**3 + k2*r**5 + k3*r**7
                elif dist_type == "radial9-odd": r_corr = r + k1*r**3 + k2*r**5 + k3*r**7 + k4*r**9

                if r == 0.0:
                    r_scale = 0.0
                else:
                    r_scale = (r_corr/r - 1.0)

                dx = (x_img - x0)*r_scale - x0
                dy = (y_img - y0)*r_scale*(1.0 + xy) - y0*(1.0 + xy) + y_img*xy

                x_est = x_corr - dx
                y_est = y_corr - dy
                delta_r = sqrt((x_img - x_est)**2 + (y_img - y_est)**2)
                x_img = x_est; y_img = y_est
        else:
            x_img = x_corr; y_img = y_corr

        x_array[i] = x_img + x_res/2.0
        y_array[i] = y_img + y_res/2.0

    return x_array, y_array



@cython.boundscheck(False)
@cython.wraparound(False)
def cyXYToGeo_wgs84(
    np.ndarray[FLOAT_TYPE_t, ndim=1] x_data,
    np.ndarray[FLOAT_TYPE_t, ndim=1] y_data,
    double x_res, double y_res,
    double alt_centre, double az_centre,               # degrees (match cyXYToAltAz)
    double rotation_from_horiz,                        # degrees
    double pix_scale,                                  # px/deg
    np.ndarray[FLOAT_TYPE_t, ndim=1] x_poly_fwd,       # SAME layout as cyXYToAltAz
    np.ndarray[FLOAT_TYPE_t, ndim=1] y_poly_fwd,
    str dist_type,
    # station geodetic & target ellipsoidal height
    double lat_sta_deg, double lon_sta_deg, double h_sta_m, np.ndarray[FLOAT_TYPE_t, ndim=1] ht_wgs84_m,
    # options
    bint refraction=True, bint equal_aspect=False,
    bint force_distortion_centre=False, bint asymmetry_corr=True,
    double min_el_deg=0.0
):
    """
    Pixels (x,y) -> WGS-84 (lat, lon) at specified height.
    Undistort/gnomonic step is IDENTICAL to cyXYToAltAz (same normalization & formulas).
    Then: build ENU ray -> ECEF -> intersect WGS-84 h = ht_wgs84_m by bisection.

    Returns:
        (lat_deg[], lon_deg[]) ; NaNs where elevation < min_el_deg or no intersection.
    """

    # -------------------- HOISTED DECLARATIONS --------------------
    cdef Py_ssize_t n, i
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] lat_out, lon_out
    # WGS-84
    cdef double a, f, e2, b, ep2
    # Station ECEF, rotation mats
    cdef double latS, lonS, sS, cS, Nsta, Xc, Yc, Zc
    cdef double RE0, RE1, RE2, RN0, RN1, RN2, RU0, RU1, RU2
    # Center & rotation (Alt/Az)
    cdef double A0, h0, rotH
    # Distortion params
    cdef int index_offset
    cdef double x0, y0, xy, a1, a2, k1, k2, k3, k4
    # Per-sample working vars
    cdef double x_img, y_img, r, dx, dy, x_corr, y_corr
    cdef double R, sin_t, cos_t, theta, A, h, el_gate
    cdef double east, north, up
    cdef double dxe, dye, dze
    # Intersection helpers
    cdef double C2, Cdotd, disc
    cdef double num, den, Rgeo, r_guess
    cdef double s_lo, s_hi, s_mid
    cdef double Xi, Yi, Zi, pval, theta_b, st, ct
    cdef double latP, lonP, Ncur, hP, f_lo, f_hi, f_mid
    cdef int it
    # --------------------------------------------------------------

    n = x_data.shape[0]
    lat_out = np.empty(n, dtype=FLOAT_TYPE)
    lon_out = np.empty(n, dtype=FLOAT_TYPE)

    # --- WGS-84 constants ---
    a  = 6378137.0
    f  = 1.0/298.257223563
    e2 = f*(2.0 - f)
    b  = a*(1.0 - f)
    ep2 = (a*a - b*b)/(b*b)

    # --- Station ECEF ---
    latS = radians(lat_sta_deg)
    lonS = radians(lon_sta_deg)
    sS = sin(latS); cS = cos(latS)
    Nsta = a / sqrt(1.0 - e2*sS*sS)
    Xc = (Nsta + h_sta_m)*cS*cos(lonS)
    Yc = (Nsta + h_sta_m)*cS*sin(lonS)
    Zc = (Nsta*(1.0 - e2) + h_sta_m)*sS

    # ECEF <- ENU (columns are E,N,U in ECEF)
    RE0 = -sin(lonS); RE1 =  cos(lonS); RE2 = 0.0
    RN0 = -sS*cos(lonS); RN1 = -sS*sin(lonS); RN2 = cS
    RU0 =  cS*cos(lonS); RU1 =  cS*sin(lonS); RU2 = sS

    # --- Centre & rotation (match cyXYToAltAz) ---
    A0 = radians(az_centre)
    h0 = radians(alt_centre)
    if refraction:
        h0 = refractionTrueToApparent(h0)  # apparent centre for spherical math, as in your code
    rotH = radians(rotation_from_horiz)

    # --- Distortion params (IDENTICAL unpack to cyXYToAltAz) ---
    index_offset = 0
    if force_distortion_centre:
        x0 = 0.5/(x_res/2.0); y0 = 0.5/(y_res/2.0); index_offset += 2
    else:
        x0 = x_poly_fwd[0];   y0 = x_poly_fwd[1]
    x0 *= (x_res/2.0); y0 *= (y_res/2.0)
    x0 = -x_res/2.0 + (x0 + x_res/2.0)%x_res
    y0 = -y_res/2.0 + (y0 + y_res/2.0)%y_res

    if equal_aspect:
        xy = 0.0; index_offset += 1
    else:
        xy = x_poly_fwd[2 - index_offset]

    if asymmetry_corr:
        a1 = x_poly_fwd[3 - index_offset]
        a2 = (x_poly_fwd[4 - index_offset]*(2*pi))%(2*pi)
    else:
        a1 = 0.0; a2 = 0.0; index_offset += 2

    k1 = k2 = k3 = k4 = 0.0
    if   dist_type == "radial3-all": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]
    elif dist_type == "radial4-all": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]; k3 = x_poly_fwd[7 - index_offset]
    elif dist_type == "radial5-all": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]; k3 = x_poly_fwd[7 - index_offset]; k4 = x_poly_fwd[8 - index_offset]
    elif dist_type == "radial3-odd": k1 = x_poly_fwd[5 - index_offset]
    elif dist_type == "radial5-odd": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]
    elif dist_type == "radial7-odd": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]; k3 = x_poly_fwd[7 - index_offset]
    elif dist_type == "radial9-odd": k1 = x_poly_fwd[5 - index_offset]; k2 = x_poly_fwd[6 - index_offset]; k3 = x_poly_fwd[7 - index_offset]; k4 = x_poly_fwd[8 - index_offset]

    # --- helpers for intersection ---
    C2 = Xc*Xc + Yc*Yc + Zc*Zc
    el_gate = radians(min_el_deg)
    # geocentric radius near station for sphere guess
    num = (a*a*cS)*(a*a*cS) + (b*b*sS)*(b*b*sS)
    den = (a*cS)*(a*cS) + (b*sS)*(b*sS)
    Rgeo = sqrt(num/den)
    # Note: r_guess will be set per point in the loop

    # --- main loop ---
    for i in range(n):
        # 1) center-subtract
        x_img = x_data[i] - x_res/2.0
        y_img = y_data[i] - y_res/2.0

        # 2) UNDISTORT (IDENTICAL to cyXYToAltAz)
        if dist_type.startswith("poly3+radial"):
            r  = sqrt((x_img - x0)**2 + (y_img - y0)**2)
            dx = (x_poly_fwd[1]*x_img + x_poly_fwd[2]*y_img
                + x_poly_fwd[3]*x_img**2 + x_poly_fwd[4]*x_img*y_img + x_poly_fwd[5]*y_img**2
                + x_poly_fwd[6]*x_img**3 + x_poly_fwd[7]*x_img**2*y_img + x_poly_fwd[8]*x_img*y_img**2 + x_poly_fwd[9]*y_img**3
                + x_poly_fwd[10]*x_img*r + x_poly_fwd[11]*y_img*r)
            dy = (y_poly_fwd[1]*x_img + y_poly_fwd[2]*y_img
                + y_poly_fwd[3]*x_img**2 + y_poly_fwd[4]*x_img*y_img + y_poly_fwd[5]*y_img**2
                + y_poly_fwd[6]*x_img**3 + y_poly_fwd[7]*x_img**2*y_img + y_poly_fwd[8]*x_img*y_img**2 + y_poly_fwd[9]*y_img**3
                + y_poly_fwd[10]*y_img*r + y_poly_fwd[11]*x_img*r)
            x_corr = (x_img + dx)/pix_scale
            y_corr = (y_img + dy)/pix_scale

        elif dist_type.startswith("radial"):
            r  = sqrt((x_img - x0)**2 + ((1.0 + xy)*(y_img - y0))**2)
            r  = r + a1*(1.0 + xy)*(y_img - y0)*cos(a2) - a1*(x_img - x0)*sin(a2)
            r  = r/(x_res/2.0)

            # forward radial model (same as your code)
            r_corr = r
            if   dist_type == "radial3-all": r_corr = r + k1*r**2 + k2*r**3
            elif dist_type == "radial4-all": r_corr = r + k1*r**2 + k2*r**3 + k3*r**4
            elif dist_type == "radial5-all": r_corr = r + k1*r**2 + k2*r**3 + k3*r**4 + k4*r**5
            elif dist_type == "radial3-odd": r_corr = r + k1*r**3
            elif dist_type == "radial5-odd": r_corr = r + k1*r**3 + k2*r**5
            elif dist_type == "radial7-odd": r_corr = r + k1*r**3 + k2*r**5 + k3*r**7
            elif dist_type == "radial9-odd": r_corr = r + k1*r**3 + k2*r**5 + k3*r**7 + k4*r**9

            if r == 0.0:
                dx = dy = 0.0
            else:
                dx = (x_img - x0)*(r_corr/r - 1.0) - x0
                dy = (y_img - y0)*(r_corr/r - 1.0)*(1.0 + xy) - y0*(1.0 + xy) + y_img*xy
            x_corr = (x_img + dx)/pix_scale
            y_corr = (y_img + dy)/pix_scale
        else:
            x_corr = x_img/pix_scale
            y_corr = y_img/pix_scale

        # 3) gnomonic → Alt/Az (IDENTICAL to cyXYToAltAz)
        R = radians(sqrt(x_corr*x_corr + y_corr*y_corr))
        if R < 1e-12:
            A = A0; h = h0
        else:
            theta = (pi/2.0 - rotH + atan2(y_corr, x_corr))%(2*pi)
            # altitude
            sin_t = sin(h0)*cos(R) + cos(h0)*sin(R)*cos(theta)
            h = atan2(sin_t, sqrt(1.0 - sin_t*sin_t))
            # azimuth
            sin_t = sin(theta)*sin(R)/cos(h)
            cos_t = (cos(R) - sin(h)*sin(h0))/(cos(h)*cos(h0))
            A = (A0 + atan2(sin_t, cos_t) + 2*pi)%(2*pi)

        # Use TRUE elevation for the ray if you mirrored cyXYToAltAz finishing step
        if refraction:
            h = refractionApparentToTrue(h)
        if h < el_gate:
            lat_out[i] = np.nan; lon_out[i] = np.nan
            continue

        # 4) ENU ray → ECEF direction
        east  = cos(h)*sin(A)
        north = cos(h)*cos(A)
        up    = sin(h)
        dxe = RE0*east + RN0*north + RU0*up
        dye = RE1*east + RN1*north + RU1*up
        dze = RE2*east + RN2*north + RU2*up

        # 5) Intersect WGS-84 h = ht_wgs84_m   (sphere guess + bisection)
        r_guess = Rgeo + ht_wgs84_m[i] + 1000.0
        C2 = Xc*Xc + Yc*Yc + Zc*Zc
        Cdotd = Xc*dxe + Yc*dye + Zc*dze
        disc  = Cdotd*Cdotd - (C2 - r_guess*r_guess)
        if disc <= 0.0:
            lat_out[i] = np.nan; lon_out[i] = np.nan
            continue
        s_hi = -Cdotd + sqrt(disc)
        s_lo = 0.0
        f_lo = h_sta_m - ht_wgs84_m[i]

        Xi = Xc + s_hi*dxe; Yi = Yc + s_hi*dye; Zi = Zc + s_hi*dze
        pval = sqrt(Xi*Xi + Yi*Yi)
        theta_b = atan2(Zi*a, pval*b); st = sin(theta_b); ct = cos(theta_b)
        latP = atan2(Zi + ep2*b*st*st*st, pval - e2*a*ct*ct*ct)
        Ncur = a / sqrt(1.0 - e2*sin(latP)*sin(latP))
        hP   = pval/cos(latP) - Ncur
        f_hi = hP - ht_wgs84_m[i]

        it = 0
        while f_lo*f_hi > 0.0 and it < 6:
            s_hi *= 1.5
            Xi = Xc + s_hi*dxe; Yi = Yc + s_hi*dye; Zi = Zc + s_hi*dze
            pval = sqrt(Xi*Xi + Yi*Yi)
            theta_b = atan2(Zi*a, pval*b); st = sin(theta_b); ct = cos(theta_b)
            latP = atan2(Zi + ep2*b*st*st*st, pval - e2*a*ct*ct*ct)
            Ncur = a / sqrt(1.0 - e2*sin(latP)*sin(latP))
            hP   = pval/cos(latP) - Ncur
            f_hi = hP - ht_wgs84_m[i]
            it += 1

        if f_lo*f_hi > 0.0:
            lat_out[i] = np.nan; lon_out[i] = np.nan
            continue

        for it in range(20):
            s_mid = 0.5*(s_lo + s_hi)
            Xi = Xc + s_mid*dxe; Yi = Yc + s_mid*dye; Zi = Zc + s_mid*dze
            pval = sqrt(Xi*Xi + Yi*Yi)
            theta_b = atan2(Zi*a, pval*b); st = sin(theta_b); ct = cos(theta_b)
            latP = atan2(Zi + ep2*b*st*st*st, pval - e2*a*ct*ct*ct)
            Ncur = a / sqrt(1.0 - e2*sin(latP)*sin(latP))
            hP   = pval/cos(latP) - Ncur
            f_mid = hP - ht_wgs84_m[i]
            if f_lo*f_mid <= 0.0:
                s_hi = s_mid; f_hi = f_mid
            else:
                s_lo = s_mid; f_lo = f_mid
            if fabs(f_mid) < 1e-3:  # ~1 mm
                break

        lonP = atan2(Yi, Xi)
        lat_out[i] = degrees(latP)
        lon_out[i] = (degrees(lonP) + 540.0) % 360.0 - 180.0

    return lat_out, lon_out

