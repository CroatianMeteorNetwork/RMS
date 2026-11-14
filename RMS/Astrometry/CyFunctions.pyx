#!python
#cython: language_level=3

import numpy as np

# Cython import
cimport numpy as np

# Initialize the NumPy C API (explicit for clarity: Cython 3 (required to build against NumPy 2) emits it itself)
np.import_array()
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

    Note that RMS.Math.angularSeparation has the same name but takes and returns radians.

    @param ra1: [float] right ascension of the first stars (in degrees)
    @param dec1: [float] declination of the first star (in degrees)
    @param ra2: [float] right ascension of the decons stars (in degrees)
    @param dec2: [float] declination of the decons star (in degrees)

    @return angular_separation: [float] angular separation (in degrees)
    """

    cdef double deldec2
    cdef double delra2
    cdef double sindis
    cdef double cos_sep

    # Convert input coordinates to radians
    ra1 = radians(ra1)
    dec1 =  radians(dec1)
    ra2 = radians(ra2)
    dec2 = radians(dec2)


    # Classical method
    # Rounding can push the cosine slightly above 1 for (nearly) coincident directions, which
    # would make acos return NaN, so clamp it to the closed interval [-1, 1]. The explicit
    # comparisons are used instead of fmax(-1.0, fmin(1.0, cos_sep)) because the C fmin/fmax
    # return the non-NaN operand, which would silently turn a NaN input into 0 or 180 deg.
    # Mirrors the clip in RMS.Math.angularSeparation (which takes radians, not degrees).
    cos_sep = sin(dec1)*sin(dec2) + cos(dec1)*cos(dec2)*cos(ra2 - ra1)

    if cos_sep > 1.0:
        cos_sep = 1.0
    elif cos_sep < -1.0:
        cos_sep = -1.0

    return degrees(acos(cos_sep))


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

    cdef double T, t, zeta, z, theta, A, B, C, ra_corr, dec_corr, cos_pole_dist


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

        # A, B and C are components of a unit vector, so sqrt(A**2 + B**2) is in [0, 1], but
        #   rounding can push it just above 1 and make acos return NaN
        cos_pole_dist = sqrt(A**2 + B**2)

        if cos_pole_dist > 1.0:
            cos_pole_dist = 1.0

        dec_corr = sign(dec)*acos(cos_pole_dist)
    else:
        dec_corr = asin(C)


    return ra_corr, dec_corr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void precessionMatrix(double zeta, double theta, double z, double P[3][3]):
    """ Calculate the precession matrix based on precession angles.

    Arguments:
        zeta: [double] Precession angle zeta in radians.
        theta: [double] Precession angle theta in radians.
        z: [double] Precession angle z in radians.
        P: [double[3][3]] Output: a 3x3 rotation matrix representing the precession transformation.

    Return:
        None

    Notes:
        - This matrix is used to transform coordinates from one epoch to another, 
          accounting for the precession of the Earth's rotational axis.
        - The matrix is calculated using the formulation from the IAU 1976 precession model.
        - Input angles should be calculated for the time span between the initial and final epochs.
    """

    # Sines and cosines of the precession angles
    cdef double czeta = cos(zeta)
    cdef double szeta = sin(zeta)
    cdef double ctheta = cos(theta)
    cdef double stheta = sin(theta)
    cdef double cz = cos(z)
    cdef double sz = sin(z)

    # Calculate matrix elements
    P[0][0] = czeta*ctheta*cz - szeta*sz
    P[0][1] = -szeta*ctheta*cz - czeta*sz
    P[0][2] = -stheta*cz
    
    P[1][0] = czeta*ctheta*sz + szeta*cz
    P[1][1] = -szeta*ctheta*sz + czeta*cz
    P[1][2] = -stheta*sz
    
    P[2][0] = czeta*stheta
    P[2][1] = -szeta*stheta
    P[2][2] = ctheta



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


cdef inline double dotProduct3(double *a, double *b):
    """ Dot product of two 3D vectors.

    Arguments:
        a: [double[3]] First vector.
        b: [double[3]] Second vector.

    Return:
        [double] The dot product.
    """

    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


cdef inline void crossProduct3(double *a, double *b, double *out):
    """ Cross product of two 3D vectors.

    Arguments:
        a: [double[3]] First vector.
        b: [double[3]] Second vector.
        out: [double[3]] Output: the cross product a x b.

    Return:
        None
    """

    out[0] = a[1]*b[2] - a[2]*b[1]
    out[1] = a[2]*b[0] - a[0]*b[2]
    out[2] = a[0]*b[1] - a[1]*b[0]


cdef inline void matrixVectorProduct3(double M[3][3], double *v, double *out):
    """ Product of a 3x3 matrix and a 3D vector.

    Arguments:
        M: [double[3][3]] Matrix.
        v: [double[3]] Vector.
        out: [double[3]] Output: the product M v (must not be v).

    Return:
        None
    """

    cdef int i

    # Every element of the result is the dot product of a matrix row and the vector
    for i in range(3):
        out[i] = M[i][0]*v[0] + M[i][1]*v[1] + M[i][2]*v[2]


cdef inline void normalizeVector3(double *v):
    """ Normalize a 3D vector to unit length, in place.

    Arguments:
        v: [double[3]] Vector, normalized in place.

    Return:
        None
    """

    cdef double norm = sqrt(dotProduct3(v, v))

    v[0] /= norm
    v[1] /= norm
    v[2] /= norm


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

    # The vectors and matrices are plain C arrays: small numpy arrays made this function, which is called for
    #   every coordinate conversion, several times more expensive
    cdef:
        double initial_vector[3]
        double transformed_vector[3]
        double nutated_vector[3]
        double parallel_vec[3]
        double parallel_vec_precessed[3]
        double normal_vector[3]
        double transformed_normal[3]
        double proj_parallel[3]
        double proj_parallel_precessed[3]
        double cross_vec[3]
        double P[3][3]
        double epsilon_matrix[3][3]
        double psi_matrix[3][3]
        double parallel_dot, distance

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
    precessionMatrix(zeta, theta, z, P)

    # Convert RA, Dec to cartesian coordinates (as raDecToCartesian)
    initial_vector[0] = cos(dec)*cos(ra)
    initial_vector[1] = cos(dec)*sin(ra)
    initial_vector[2] = sin(dec)

    # Apply precession
    matrixVectorProduct3(P, initial_vector, transformed_vector)



    # Calculate nutation corrections
    Delta_psi, Delta_epsilon = nutationComponents(T)

    # Construct the nutation matrices
    epsilon_matrix[0][0], epsilon_matrix[0][1], epsilon_matrix[0][2] = 1, 0, 0
    epsilon_matrix[1][0], epsilon_matrix[1][1], epsilon_matrix[1][2] = 0, cos(Delta_epsilon), sin(Delta_epsilon)
    epsilon_matrix[2][0], epsilon_matrix[2][1], epsilon_matrix[2][2] = 0, -sin(Delta_epsilon), cos(Delta_epsilon)

    psi_matrix[0][0], psi_matrix[0][1], psi_matrix[0][2] = cos(Delta_psi), -sin(Delta_psi), 0
    psi_matrix[1][0], psi_matrix[1][1], psi_matrix[1][2] = sin(Delta_psi), cos(Delta_psi), 0
    psi_matrix[2][0], psi_matrix[2][1], psi_matrix[2][2] = 0, 0, 1

    # Apply nutation (obliquity first, then longitude)
    matrixVectorProduct3(epsilon_matrix, transformed_vector, nutated_vector)
    matrixVectorProduct3(psi_matrix, nutated_vector, transformed_vector)


    # Calculate normal vector to the plane of precession
    crossProduct3(initial_vector, transformed_vector, normal_vector)
    matrixVectorProduct3(P, normal_vector, transformed_normal)

    # Normalize vectors
    normalizeVector3(initial_vector)
    normalizeVector3(transformed_vector)
    normalizeVector3(normal_vector)
    normalizeVector3(transformed_normal)

    # Convert precessed vector back to RA, Dec (as cartesianToRaDec)
    distance = hypot(hypot(transformed_vector[0], transformed_vector[1]), transformed_vector[2])
    if distance == 0:
        ra_precessed, dec_precessed = 0.0, 0.0
    else:
        ra_precessed = fmod(atan2(transformed_vector[1], transformed_vector[0]) + 2*M_PI, 2*M_PI)
        dec_precessed = asin(transformed_vector[2]/distance)

    # Calculate vector tangent to the parallel of declination
    parallel_vec[0], parallel_vec[1], parallel_vec[2] = -sin(ra), cos(ra), 0
    parallel_vec_precessed[0], parallel_vec_precessed[1], parallel_vec_precessed[2] = -sin(ra_precessed), \
        cos(ra_precessed), 0

    # Project parallel vectors onto the plane perpendicular to the line of sight
    parallel_dot = dotProduct3(parallel_vec, initial_vector)
    for i in range(3):
        proj_parallel[i] = parallel_vec[i] - parallel_dot*initial_vector[i]

    parallel_dot = dotProduct3(parallel_vec_precessed, transformed_vector)
    for i in range(3):
        proj_parallel_precessed[i] = parallel_vec_precessed[i] - parallel_dot*transformed_vector[i]

    # Normalize projected vectors
    normalizeVector3(proj_parallel)
    normalizeVector3(proj_parallel_precessed)

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
    crossProduct3(normal_vector, proj_parallel, cross_vec)
    angle1 = atan2(dotProduct3(cross_vec, initial_vector), dotProduct3(normal_vector, proj_parallel))

    # Angle for the precessed position
    crossProduct3(transformed_normal, proj_parallel_precessed, cross_vec)
    angle2 = atan2(dotProduct3(cross_vec, transformed_vector), dotProduct3(transformed_normal,
        proj_parallel_precessed))

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



cdef int radialTermCount(str dist_type):
    """ Number of the radial distortion coefficients (k1, k2, ...) of a radial distortion type (see
        Platepar.radialTermCount): the decentering coefficients follow them in the distortion arrays.
    """

    # The radial coefficients of every radial type (k1 for radial3-odd up to k1...k4 for radial9-odd and radial5-all)
    if dist_type == "radial3-odd":
        return 1
    elif (dist_type == "radial5-odd") or (dist_type == "radial3-all"):
        return 2
    elif (dist_type == "radial7-odd") or (dist_type == "radial4-all"):
        return 3
    else:
        return 4



### EXACT INVERSE OF THE FORWARD DISTORTION ###
#
# cyraDecToXY can take the sky to the image through the forward distortion (x_poly_fwd) instead of the separately
#   fitted reverse one (x_poly_rev), so that the two directions are exact inverses of each other. A radial model
#   keeps the direction from the distortion centre and only rescales the radius, which makes its inverse one root
#   in one variable; the polynomial models and the decentering correction are inverted by Newton's method in the
#   image plane.
#
# Every catalog star of a pointing fit or of the star matching goes through the inverse, so its work per point
#   is kept to plain arithmetic: the model is unpacked once per call into a C structure, the radial functions
#   are evaluated by Horner's rule, Newton's method takes the Jacobian of the distortion analytically and
#   starts with the linear part of the distortion already undone, and the rising branch of a radial model is
#   kept between calls.

# Not-a-number and infinity, for points the forward distortion cannot reach and for the first error of a
#   search
cdef double NAN_VALUE = float("nan")
cdef double INF_VALUE = float("inf")

# Upper end, in radius normalized to half the image width, of the search for the rising branch of a radial model
cdef double RADIAL_BRANCH_LIMIT = 4.0

# Newton's method in the image plane stops once its step is shorter than this (px). The error it leaves is of
#   the order of the square of the last step times the curvature of the distortion (below 1e-3 /px even for a
#   strong lens), so below 1e-13 px
cdef double NEWTON_STEP_TOL = 1e-5

# The root of a radial model stops once its step is shorter than this, in radius normalized to half the image
#   width
cdef double RADIAL_STEP_TOL = 1e-10


# A distortion model of cyXYToRADec, unpacked for the exact inverse
cdef struct ForwardModel:

    # -1 for a radial model; for poly3+radial, the number of its higher radial terms (0, 1 for +radial3, 2 for
    #   +radial5)
    int poly_radial

    # The polynomial coefficients in X and Y
    double cx[14]
    double cy[14]

    # Distortion centre (px), aspect and asymmetry of a radial model
    double x0
    double y0
    double xy
    double a1
    double sin_a2
    double cos_a2

    # Radial type (radialKind) and coefficients, the ones the type does not have set to zero
    int kind
    double k1
    double k2
    double k3
    double k4

    # Decentering of a radial model
    int decentering
    double p1
    double p2
    double p3
    double p4

    # Half the image width (px), the unit of the radius of a radial model
    double half_width


cdef int radialKind(str dist_type):
    """ The radial distortion type as a number, so that the per-point root does not compare strings. """

    if dist_type == "radial3-all":
        return 0
    elif dist_type == "radial4-all":
        return 1
    elif dist_type == "radial5-all":
        return 2
    elif dist_type == "radial3-odd":
        return 3
    elif dist_type == "radial5-odd":
        return 4
    elif dist_type == "radial7-odd":
        return 5
    else:
        return 6


@cython.cdivision(True)
cdef double radialRatio(double r, int kind, double k1, double k2, double k3, double k4, double *dg):
    """ The ratio f(r)/r of the corrected radius of a radial model, as cyXYToRADec computes it, to the radius,
        by Horner's rule. The coefficients the type does not have must be zero.

    Arguments:
        r: [float] Radius, normalized to half the image width.
        kind: [int] Radial distortion type (radialKind).
        k1: [float] First radial coefficient.
        k2: [float] Second radial coefficient.
        k3: [float] Third radial coefficient.
        k4: [float] Fourth radial coefficient.
        dg: [double pointer] The derivative of the ratio in r is written to dg[0].

    Return:
        g: [float] The ratio f(r)/r, 1 at r = 0.
    """

    cdef double t

    # Odd powers: f(r) = r + k1*r^3 + k2*r^5 + k3*r^7 + k4*r^9
    if kind >= 3:
        t = r*r
        dg[0] = r*(2*k1 + t*(4*k2 + t*(6*k3 + t*8*k4)))
        return 1.0 + t*(k1 + t*(k2 + t*(k3 + t*k4)))

    # All powers: f(r) = r + k1*r^2 + k2*r^3 + k3*r^4 + k4*r^5
    dg[0] = k1 + r*(2*k2 + r*(3*k3 + r*4*k4))
    return 1.0 + r*(k1 + r*(k2 + r*(k3 + r*k4)))


# The rising branch of the last radial model, kept between calls: the pointing fits and the star matching take
#   the same model to the image many times
cdef int branch_kind = -1
cdef double branch_k1 = 0, branch_k2 = 0, branch_k3 = 0, branch_k4 = 0, branch_r_end = 0, branch_f_end = 0


@cython.cdivision(True)
cdef double radialBranchEnd(int kind, double k1, double k2, double k3, double k4, double *f_end):
    """ Where the rising branch of f, starting at f(0) = 0, ends: the first point where f' falls to zero. Beyond
        it the model folds back and has no inverse.

    Arguments:
        kind: [int] Radial distortion type (radialKind).
        k1: [float] First radial coefficient.
        k2: [float] Second radial coefficient.
        k3: [float] Third radial coefficient.
        k4: [float] Fourth radial coefficient.
        f_end: [double pointer] The corrected radius at the end of the branch is written to f_end[0].

    Return:
        r_end: [float] The radius at the end of the branch, normalized to half the image width.
    """

    global branch_kind, branch_k1, branch_k2, branch_k3, branch_k4, branch_r_end, branch_f_end

    cdef int i, j, n = 4000
    cdef double r, lo, hi, mid, g, dg

    # The same model as the last call
    if (kind == branch_kind) and (k1 == branch_k1) and (k2 == branch_k2) and (k3 == branch_k3) \
        and (k4 == branch_k4):

        f_end[0] = branch_f_end
        return branch_r_end

    # Scan f' = g + r*g' outward from the centre for its first zero
    lo = RADIAL_BRANCH_LIMIT
    for i in range(1, n + 1):
        r = RADIAL_BRANCH_LIMIT*i/n
        g = radialRatio(r, kind, k1, k2, k3, k4, &dg)
        if g + r*dg <= 0:

            # Refine the turning point by bisection
            lo = RADIAL_BRANCH_LIMIT*(i - 1)/n
            hi = r
            for j in range(60):
                mid = 0.5*(lo + hi)
                g = radialRatio(mid, kind, k1, k2, k3, k4, &dg)
                if g + mid*dg > 0:
                    lo = mid
                else:
                    hi = mid

            break

    # Keep the branch for the next call
    branch_kind = kind
    branch_k1, branch_k2, branch_k3, branch_k4 = k1, k2, k3, k4
    branch_r_end = lo
    branch_f_end = lo*radialRatio(lo, kind, k1, k2, k3, k4, &dg)

    f_end[0] = branch_f_end
    return branch_r_end


@cython.cdivision(True)
cdef double radialRoot(double target, int kind, double k1, double k2, double k3, double k4, double r_end,
    double f_end):
    """ The radius r >= 0 on the rising branch of f with f(r) = target, by Newton's method kept inside a
        bisection bracket.

    Arguments:
        target: [float] Corrected radius, normalized to half the image width.
        kind: [int] Radial distortion type (radialKind).
        k1: [float] First radial coefficient.
        k2: [float] Second radial coefficient.
        k3: [float] Third radial coefficient.
        k4: [float] Fourth radial coefficient.
        r_end: [float] End of the rising branch (radialBranchEnd).
        f_end: [float] Corrected radius at the end of the rising branch.

    Return:
        r: [float] The radius, NaN if the target lies beyond the branch.
    """

    cdef int i
    cdef double lo = 0.0, hi = r_end, r, g, dg, f, df, res, nxt

    if (target < 0) or (target > f_end):
        return NAN_VALUE

    # Newton's method from the target itself (the distortion is small next to the radius), falling back to
    #   bisection whenever its step leaves the bracket
    r = target if target < r_end else r_end
    for i in range(100):
        g = radialRatio(r, kind, k1, k2, k3, k4, &dg)
        f = r*g
        df = g + r*dg
        res = f - target
        if res == 0:
            return r
        elif res < 0:
            lo = r
        else:
            hi = r

        nxt = r - res/df if df > 0 else 0.5*(lo + hi)
        if not ((nxt > lo) and (nxt < hi)):
            nxt = 0.5*(lo + hi)

        # A step this short leaves an error of the order of its square
        if fabs(nxt - r) <= RADIAL_STEP_TOL:
            return nxt

        r = nxt

    return r


@cython.cdivision(True)
cdef double forwardJacobian(ForwardModel *m, double x_img, double y_img, double *out):
    """ The distortion of cyXYToRADec, image coordinates relative to the image centre to corrected ones (both
        in pixels), and its Jacobian. The same expressions as the block in cyXYToRADec, which the round-trip
        tests check.

    Arguments:
        m: [ForwardModel pointer] The distortion model.
        x_img: [float] Image X coordinate relative to the image centre (px).
        y_img: [float] Image Y coordinate relative to the image centre (px).
        out: [double pointer] Six values are written to it: the corrected X and Y, and the derivatives
            dX/dx, dX/dy, dY/dx, dY/dy.

    Return:
        0: [float] Always zero.
    """

    cdef double *c
    cdef double *d
    cdef double x2, y2, xy2, r, rx, ry, r2, r3, r4, r5, fx, fy, jxx, jxy, jyx, jyy
    cdef double u, w, rho, rn, drx, dry, g, dg, uu, vv, q1, q2

    # Polynomial with radial terms
    if m.poly_radial >= 0:

        c = m.cx
        d = m.cy

        # Radius from the distortion centre and its derivatives (none at the centre itself)
        r = sqrt((x_img - m.x0)*(x_img - m.x0) + (y_img - m.y0)*(y_img - m.y0))
        if r > 0:
            rx = (x_img - m.x0)/r
            ry = (y_img - m.y0)/r
        else:
            rx = 0
            ry = 0

        x2 = x_img*x_img
        y2 = y_img*y_img
        xy2 = x_img*y_img

        # The third order polynomial and the first radial terms, in X and Y
        fx = (x_img + m.x0 + c[1]*x_img + c[2]*y_img + c[3]*x2 + c[4]*xy2 + c[5]*y2 + c[6]*x2*x_img
            + c[7]*x2*y_img + c[8]*x_img*y2 + c[9]*y2*y_img + c[10]*x_img*r + c[11]*y_img*r)
        jxx = (1.0 + c[1] + 2*c[3]*x_img + c[4]*y_img + 3*c[6]*x2 + 2*c[7]*xy2 + c[8]*y2
            + c[10]*(r + x_img*rx) + c[11]*y_img*rx)
        jxy = (c[2] + c[4]*x_img + 2*c[5]*y_img + c[7]*x2 + 2*c[8]*xy2 + 3*c[9]*y2 + c[10]*x_img*ry
            + c[11]*(r + y_img*ry))

        fy = (y_img + m.y0 + d[1]*x_img + d[2]*y_img + d[3]*x2 + d[4]*xy2 + d[5]*y2 + d[6]*x2*x_img
            + d[7]*x2*y_img + d[8]*x_img*y2 + d[9]*y2*y_img + d[10]*y_img*r + d[11]*x_img*r)
        jyx = (d[1] + 2*d[3]*x_img + d[4]*y_img + 3*d[6]*x2 + 2*d[7]*xy2 + d[8]*y2 + d[10]*y_img*rx
            + d[11]*(r + x_img*rx))
        jyy = (1.0 + d[2] + d[4]*x_img + 2*d[5]*y_img + d[7]*x2 + 2*d[8]*xy2 + 3*d[9]*y2
            + d[10]*(r + y_img*ry) + d[11]*x_img*ry)

        # The third order radial term
        if m.poly_radial >= 1:
            r2 = r*r
            r3 = r2*r
            fx += c[12]*x_img*r3
            jxx += c[12]*(r3 + 3*x_img*r2*rx)
            jxy += 3*c[12]*x_img*r2*ry
            fy += d[12]*y_img*r3
            jyx += 3*d[12]*y_img*r2*rx
            jyy += d[12]*(r3 + 3*y_img*r2*ry)

        # The fifth order radial term
        if m.poly_radial >= 2:
            r4 = r2*r2
            r5 = r4*r
            fx += c[13]*x_img*r5
            jxx += c[13]*(r5 + 5*x_img*r4*rx)
            jxy += 5*c[13]*x_img*r4*ry
            fy += d[13]*y_img*r5
            jyx += 5*d[13]*y_img*r4*rx
            jyy += d[13]*(r5 + 5*y_img*r4*ry)

    # Radial model: the offsets from the distortion centre are scaled by g = f(r)/r of the normalized radius,
    #   which the asymmetry stretches in one direction
    else:

        u = x_img - m.x0
        w = (1.0 + m.xy)*(y_img - m.y0)
        rho = sqrt(u*u + w*w)
        rn = (rho + m.a1*(w*m.cos_a2 - u*m.sin_a2))/m.half_width

        # Derivatives of the normalized radius (only the asymmetry part at the centre itself)
        if rho > 0:
            drx = (u/rho - m.a1*m.sin_a2)/m.half_width
            dry = (1.0 + m.xy)*(w/rho + m.a1*m.cos_a2)/m.half_width
        else:
            drx = -m.a1*m.sin_a2/m.half_width
            dry = (1.0 + m.xy)*m.a1*m.cos_a2/m.half_width

        g = radialRatio(rn, m.kind, m.k1, m.k2, m.k3, m.k4, &dg)

        fx = u*g
        fy = w*g
        jxx = g + u*dg*drx
        jxy = u*dg*dry
        jyx = w*dg*drx
        jyy = (1.0 + m.xy)*g + w*dg*dry

        # Decentering, in the normalized coordinates of the radial distortion
        if m.decentering:
            uu = u/m.half_width
            vv = (y_img - m.y0)/m.half_width
            r2 = uu*uu + vv*vv
            q1 = m.p1 + m.p3*r2
            q2 = m.p2 + m.p4*r2
            fx += (q1*(r2 + 2*uu*uu) + 2*q2*uu*vv)*m.half_width
            fy += (2*q1*uu*vv + q2*(r2 + 2*vv*vv))*m.half_width
            jxx += 2*m.p3*uu*(r2 + 2*uu*uu) + 6*q1*uu + 4*m.p4*uu*uu*vv + 2*q2*vv
            jxy += 2*m.p3*vv*(r2 + 2*uu*uu) + 2*q1*vv + 4*m.p4*uu*vv*vv + 2*q2*uu
            jyx += 4*m.p3*uu*uu*vv + 2*q1*vv + 2*m.p4*uu*(r2 + 2*vv*vv) + 2*q2*uu
            jyy += 4*m.p3*uu*vv*vv + 2*q1*uu + 2*m.p4*vv*(r2 + 2*vv*vv) + 6*q2*vv

    out[0] = fx
    out[1] = fy
    out[2] = jxx
    out[3] = jxy
    out[4] = jyx
    out[5] = jyy

    return 0


@cython.cdivision(True)
cdef double invertDistortionNewton(ForwardModel *m, double x_corr, double y_corr, double *img):
    """ The image coordinates (relative to the image centre) that the forward distortion takes to
        (x_corr, y_corr), by Newton's method with the analytic Jacobian, the step halved whenever it would not
        reduce the error.

    Arguments:
        m: [ForwardModel pointer] The distortion model.
        x_corr: [float] Corrected X coordinate (px).
        y_corr: [float] Corrected Y coordinate (px).
        img: [double pointer] The start, img[0] and img[1]; the image coordinates are written back to it, NaN
            if the search does not converge or ends beyond a fold of the distortion.

    Return:
        n_eval: [float] The number of evaluations of the distortion.
    """

    cdef int n_eval = 0, n_steps = 0, n_halvings = 0
    cdef double out[6]
    cdef double x = img[0], y = img[1], x_prev = img[0], y_prev = img[1], sx = 0, sy = 0
    cdef double gx, gy, err, det, err_prev = INF_VALUE, det_prev = 0

    while True:

        forwardJacobian(m, x, y, out)
        n_eval += 1
        gx = out[0] - x_corr
        gy = out[1] - y_corr
        err = gx*gx + gy*gy

        # A step that did not reduce the error is halved, from where it started. After 30 halvings the search
        #   ends at that start
        if not (err < err_prev):
            n_halvings += 1
            if n_halvings > 30:
                x = x_prev
                y = y_prev
                err = err_prev
                det = det_prev
                break

            sx *= 0.5
            sy *= 0.5
            x = x_prev + sx
            y = y_prev + sy
            continue

        n_halvings = 0
        det = out[2]*out[5] - out[3]*out[4]

        # Converged, or out of steps
        if (err < 1e-20) or (n_steps >= 50):
            break

        if det == 0:
            img[0] = NAN_VALUE
            img[1] = NAN_VALUE
            return n_eval

        # The Newton step
        sx = -(gx*out[5] - gy*out[3])/det
        sy = -(gy*out[2] - gx*out[4])/det
        x_prev = x
        y_prev = y
        err_prev = err
        det_prev = det
        x += sx
        y += sy
        n_steps += 1

        # A step this short leaves an error of the order of its square: done, without evaluating there. The
        #   Jacobian of its start stands for the one at its end
        if sx*sx + sy*sy < NEWTON_STEP_TOL*NEWTON_STEP_TOL:
            err = 0
            break

    # Not converged (the error is squared)
    if err > 1e-12:
        img[0] = NAN_VALUE
        img[1] = NAN_VALUE
        return n_eval

    # A solution where the distortion reverses the orientation of the image (it does not at the image centre)
    #   lies beyond a fold of the polynomial, on a second sheet that is not the image of the sky: no inverse
    if det <= 0:
        img[0] = NAN_VALUE
        img[1] = NAN_VALUE
        return n_eval

    img[0] = x
    img[1] = y

    return n_eval

### ###


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cyraDecToXY(np.ndarray[FLOAT_TYPE_t, ndim=1] ra_data,
    np.ndarray[FLOAT_TYPE_t, ndim=1] dec_data, double jd, double lat, double lon, double x_res,
    double y_res, double h0, double jd_ref, double ra_ref, double dec_ref, double pos_angle_ref, 
    double pix_scale, np.ndarray[FLOAT_TYPE_t, ndim=1] x_poly_rev, 
    np.ndarray[FLOAT_TYPE_t, ndim=1] y_poly_rev, str dist_type, bool refraction=True, bool equal_aspect=False, 
    bool force_distortion_centre=False, bool asymmetry_corr=True, bool decentering_corr=False,
    bool invert_forward=False, np.ndarray[FLOAT_TYPE_t, ndim=1] jd_data=None):
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
        decentering_corr: [bool] Correct the radial distortion for decentering (a lens element off the optical
            axis, Brown-Conrady): the coefficients p1, p2, p3, p4 follow the radial ones. False by default.
        invert_forward: [bool] x_poly_rev and y_poly_rev hold the FORWARD distortion (x_poly_fwd, y_poly_fwd),
            and the image coordinates are those the forward distortion takes to the sky coordinates, so that this
            function is the exact inverse of cyXYToRADec. A point the forward distortion cannot reach (beyond the
            first maximum of a radial model) is returned as NaN. False by default.
        jd_data: [ndarray] Julian date of every point. None by default, in which case all points are at jd. If
            given, jd is ignored and every point is converted at its own date (the camera is fixed in the
            horizontal frame, so the sky moves across the image), the same as one call per point.
    
    Return:
        (x, y): [tuple of ndarrays] Image X and Y coordinates.
    """

    cdef int i
    cdef double ra_centre, dec_centre, ra, dec

    # The reference position angle as given (it is replaced by its value corrected for the date below), for the
    #   correction at the date of every point
    cdef double pos_angle_ref_in = pos_angle_ref
    cdef double radius, sin_ang, cos_ang, theta, x, y, r, dx, dy, x_img, y_img, r_corr, r_scale
    cdef double x0, y0, xy, a1, a2, k1 = 0, k2 = 0, k3 = 0, k4 = 0, k5 = 0
    cdef double radius_corr, cos_phi, sin_phi, asym, rho, q, r_end = 0, f_end = 0
    cdef double lin_xx = 1, lin_xy = 0, lin_yx = 0, lin_yy = 1, lin_det = 1
    cdef double img[2]
    cdef bint radial_root = False
    cdef int j, n_k
    cdef ForwardModel model
    cdef double p1 = 0, p2 = 0, p3 = 0, p4 = 0, u, v, r2, q1, q2
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

        # Decentering coefficients, after the radial ones
        if decentering_corr:
            p1 = x_poly_rev[5 - index_offset + radialTermCount(dist_type)]
            p2 = x_poly_rev[6 - index_offset + radialTermCount(dist_type)]
            p3 = x_poly_rev[7 - index_offset + radialTermCount(dist_type)]
            p4 = x_poly_rev[8 - index_offset + radialTermCount(dist_type)]

    # If the polynomial distortion was used, unpack the offsets
    else:
        x0 = x_poly_rev[0]
        y0 = y_poly_rev[0]


    # The forward distortion unpacked for its inverse
    if invert_forward:

        # The polynomial with its higher radial terms, or a radial model
        if dist_type.startswith("radial"):
            model.poly_radial = -1
        elif dist_type.endswith("+radial5"):
            model.poly_radial = 2
        elif dist_type.endswith("+radial3"):
            model.poly_radial = 1
        else:
            model.poly_radial = 0

        for j in range(14):
            model.cx[j] = x_poly_rev[j] if j < x_poly_rev.shape[0] else 0.0
            model.cy[j] = y_poly_rev[j] if j < y_poly_rev.shape[0] else 0.0

        # The parameters of a radial model, with zeros for the radial coefficients its type does not have
        model.x0 = x0
        model.y0 = y0
        model.half_width = x_res/2.0
        model.kind = radialKind(dist_type)
        model.decentering = decentering_corr
        if model.poly_radial < 0:
            n_k = radialTermCount(dist_type)
            model.xy = xy
            model.a1 = a1
            model.sin_a2 = sin(a2)
            model.cos_a2 = cos(a2)
            model.k1 = k1
            model.k2 = k2 if n_k >= 2 else 0.0
            model.k3 = k3 if n_k >= 3 else 0.0
            model.k4 = k4 if n_k >= 4 else 0.0
            model.p1, model.p2, model.p3, model.p4 = p1, p2, p3, p4
        else:
            model.xy = model.a1 = model.sin_a2 = model.k1 = model.k2 = model.k3 = model.k4 = 0.0
            model.cos_a2 = 1.0
            model.p1 = model.p2 = model.p3 = model.p4 = 0.0

            # The linear part of the polynomial, whose inverse starts Newton's method
            lin_xx = 1.0 + model.cx[1]
            lin_xy = model.cx[2]
            lin_yx = model.cy[1]
            lin_yy = 1.0 + model.cy[2]
            lin_det = lin_xx*lin_yy - lin_xy*lin_yx

        # The rising branch of a radial model, for its inverse
        if (model.poly_radial < 0) and (not decentering_corr):
            radial_root = True
            r_end = radialBranchEnd(model.kind, model.k1, model.k2, model.k3, model.k4, &f_end)

    # Convert all equatorial coordinates to image coordinates
    for i in range(ra_data.shape[0]):

        # With a date per point, correct the pointing for the date of this point
        if jd_data is not None:
            jd = jd_data[i]
            ra_centre, dec_centre, pos_angle_ref = pointingCorrection(
                    jd, radians(lat), radians(lon),
                    radians(h0), jd_ref, radians(ra_ref), radians(dec_ref), radians(pos_angle_ref_in),
                    refraction=refraction
                    )

        ra = radians(ra_data[i])
        dec = radians(dec_data[i])

        ### Gnomonization of star coordinates to image coordinates ###

        # Apply refraction
        if refraction:
            ra, dec = eqRefractionTrueToApparent(ra, dec, jd, radians(lat), radians(lon))


        # Compute the distance from the FOV centre to the sky coordinate
        radius = radians(angularSeparation(degrees(ra), degrees(dec), degrees(ra_centre), \
            degrees(dec_centre)))

        # Compute theta - the direction angle between the FOV centre, sky coordinate, and the image vertical.
        #   It is undefined (0/0) for the FOV centre itself, which projects to x = y = 0 at any angle.
        if radius == 0:
            theta = 0
        else:
            sin_ang = cos(dec)*sin(ra - ra_centre)/sin(radius)
            cos_ang = (sin(dec) - sin(dec_centre)*cos(radius))/(cos(dec_centre)*sin(radius))
            theta   = -atan2(sin_ang, cos_ang) + pos_angle_ref - pi/2.0

        # Calculate the standard coordinates
        x = degrees(radius)*cos(theta)*pix_scale
        y = degrees(radius)*sin(theta)*pix_scale

        ### ###

        # Invert the forward distortion: (x, y) are the corrected coordinates the forward distortion produces
        if invert_forward:

            if radial_root:

                # Direction from the distortion centre is kept; only the radius is rescaled. The distortion
                #   centre itself is its own image
                radius_corr = sqrt(x*x + y*y)
                if radius_corr == 0:
                    x_img = x0
                    y_img = y0

                else:
                    cos_phi = x/radius_corr
                    sin_phi = y/radius_corr
                    asym = 1.0 + model.a1*(sin_phi*model.cos_a2 - cos_phi*model.sin_a2)
                    rho = radialRoot(radius_corr*asym/model.half_width, model.kind, model.k1, model.k2,
                        model.k3, model.k4, r_end, f_end) if asym > 0 else NAN_VALUE
                    q = rho*model.half_width/asym
                    x_img = q*cos_phi + x0
                    y_img = q*sin_phi/(1.0 + xy) + y0

            else:

                # Polynomial models and decentering: Newton's method in the image plane, from the corrected
                #   position with the linear part of the distortion undone (for a radial model, its centre and
                #   aspect)
                if model.poly_radial < 0:
                    img[0] = x + x0
                    img[1] = y/(1.0 + xy) + y0
                elif lin_det > 0:
                    img[0] = (lin_yy*(x - x0) - lin_xy*(y - y0))/lin_det
                    img[1] = (lin_xx*(y - y0) - lin_yx*(x - x0))/lin_det
                else:
                    img[0] = x
                    img[1] = y

                invertDistortionNewton(&model, x, y, img)
                x_img = img[0]
                y_img = img[1]

            x_array[i] = x_img + x_res/2.0
            y_array[i] = y_img + y_res/2.0
            continue

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

            # Decentering (Brown-Conrady): a lens element off the optical axis displaces the image by p1, p2 (the
            #   amount and direction of the shift) times the square of the radius, a radial part three times the
            #   tangential one, and by p3, p4 times its fourth power. In the normalized coordinates of the radial
            #   distortion, converted to pixels
            if decentering_corr:
                u = x/(x_res/2.0)
                v = y/(x_res/2.0)
                r2 = u*u + v*v
                q1 = p1 + p3*r2
                q2 = p2 + p4*r2
                dx += (q1*(r2 + 2*u*u) + 2*q2*u*v)*(x_res/2.0)
                dy += (2*q1*u*v + q2*(r2 + 2*v*v))*(x_res/2.0)



        # Add the distortion
        x_img = x - dx
        y_img = y - dy


        # Calculate X image coordinates
        x_array[i] = x_img + x_res/2.0

        # Calculate Y image coordinates
        y_array[i] = y_img + y_res/2.0


    return x_array, y_array



cdef (double, double, double) pointingCorrection(
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
    bool asymmetry_corr=True, bool precompute_pointing_corr=False, bool decentering_corr=False):
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
        decentering_corr: [bool] Correct the radial distortion for decentering (a lens element off the optical
            axis, Brown-Conrady): the coefficients p1, p2, p3, p4 follow the radial ones. False by default.
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
    cdef double p1 = 0, p2 = 0, p3 = 0, p4 = 0, u, v, r2, q1, q2
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

        # Decentering coefficients, after the radial ones
        if decentering_corr:
            p1 = x_poly_fwd[5 - index_offset + radialTermCount(dist_type)]
            p2 = x_poly_fwd[6 - index_offset + radialTermCount(dist_type)]
            p3 = x_poly_fwd[7 - index_offset + radialTermCount(dist_type)]
            p4 = x_poly_fwd[8 - index_offset + radialTermCount(dist_type)]


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

            # Decentering (Brown-Conrady), see cyraDecToXY, around the distortion centre
            if decentering_corr:
                u = (x_img - x0)/(x_res/2.0)
                v = (y_img - y0)/(x_res/2.0)
                r2 = u*u + v*v
                q1 = p1 + p3*r2
                q2 = p2 + p4*r2
                dx += (q1*(r2 + 2*u*u) + 2*q2*u*v)*(x_res/2.0)
                dy += (2*q1*u*v + q2*(r2 + 2*v*v))*(x_res/2.0)


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