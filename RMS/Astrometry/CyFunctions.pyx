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
    double exp(double)
    int isinf(double)


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
    cdef double cos_sep

    # Convert input coordinates to radians
    ra1 = radians(ra1)
    dec1 =  radians(dec1)
    ra2 = radians(ra2)
    dec2 = radians(dec2)


    # Classical method
    # Rounding can push the cosine slightly above 1 for (nearly) coincident directions, which
    # would make acos return NaN, so clamp it to the closed interval [-1, 1].
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
        lst [float] Mean Local Sidereal Time (deg). (Mean, not apparent: the equation of the equinoxes, ~1
            arcsec, is not applied.)
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



cdef double meanObliquity(double T):
    """ Mean obliquity of the ecliptic (Meeus, Astronomical Algorithms, eq. 22.2).

    Arguments:
        T: [float] Julian centuries since J2000.0.

    Return:
        eps: [float] Mean obliquity (radians).
    """

    return radians(23.0 + 26.0/60.0 + 21.448/3600.0 - (46.8150*T + 0.00059*T*T - 0.001813*T*T*T)/3600.0)


cdef (double, double, double) nutationRotate(double jd, double x, double y, double z, bint inverse):
    """ Rotate a unit vector between the mean and the true equator and equinox of date.

        Nutation is a rotation about the ecliptic pole by the nutation in longitude, combined with the change of
        obliquity: N = R1(-(eps + deps)) * R3(-dpsi) * R1(eps) takes a vector from the mean equator and equinox of
        date to the true equator and equinox of date (the frame the Earth actually rotates in, and the one the
        sidereal time refers to). inverse=True applies N^T (true -> mean).

    Arguments:
        jd: [float] Julian date of the epoch.
        x, y, z: [float] Unit vector components (equatorial axes).
        inverse: [bool] False: mean -> true, True: true -> mean.

    Return:
        (x, y, z): [tuple of floats] Rotated vector.
    """

    cdef double T, dpsi, deps, eps, eps1, x1, y1, z1, x2, y2, z2

    T = (jd - J2000_DAYS)/36525.0
    dpsi, deps = nutationComponents(T)
    eps = meanObliquity(T)
    eps1 = eps + deps

    if not inverse:

        # R1(eps): rotate into the ecliptic frame
        x1 = x
        y1 = y*cos(eps) + z*sin(eps)
        z1 = -y*sin(eps) + z*cos(eps)

        # R3(-dpsi): nutation in longitude, about the ecliptic pole
        x2 = x1*cos(dpsi) - y1*sin(dpsi)
        y2 = x1*sin(dpsi) + y1*cos(dpsi)
        z2 = z1

        # R1(-(eps + deps)): back to the equator, with the nutated obliquity
        return x2, y2*cos(eps1) - z2*sin(eps1), y2*sin(eps1) + z2*cos(eps1)

    else:

        # N^T = R1(-eps) * R3(dpsi) * R1(eps + deps)
        x1 = x
        y1 = y*cos(eps1) + z*sin(eps1)
        z1 = -y*sin(eps1) + z*cos(eps1)

        x2 = x1*cos(dpsi) + y1*sin(dpsi)
        y2 = -x1*sin(dpsi) + y1*cos(dpsi)
        z2 = z1

        return x2, y2*cos(eps) - z2*sin(eps), y2*sin(eps) + z2*cos(eps)


cpdef (double, double) trueOfDateFromJ2000(double jd, double ra, double dec):
    """ Catalog (mean J2000) right ascension and declination -> true equator and equinox of date, i.e. precession
        followed by nutation. This is the frame that goes with the sidereal time when computing hour angles and
        alt/az, and the frame in which the platepar reference pointing RA_d/dec_d is expressed.

    Arguments:
        jd: [float] Julian date of the epoch of date.
        ra: [float] Right ascension (radians, J2000).
        dec: [float] Declination (radians, J2000).

    Return:
        (ra, dec): [tuple of floats] True equatorial coordinates of date (radians).
    """

    cdef double x, y, z

    ra, dec = equatorialCoordPrecession(J2000_DAYS, jd, ra, dec)

    x = cos(dec)*cos(ra)
    y = cos(dec)*sin(ra)
    z = sin(dec)
    x, y, z = nutationRotate(jd, x, y, z, False)

    return (atan2(y, x) + 2*pi)%(2*pi), atan2(z, sqrt(x*x + y*y))


cpdef (double, double) j2000FromTrueOfDate(double jd, double ra, double dec):
    """ Inverse of trueOfDateFromJ2000: true equator and equinox of date -> catalog (mean J2000) coordinates.

    Arguments:
        jd: [float] Julian date of the epoch of date.
        ra: [float] Right ascension (radians, true of date).
        dec: [float] Declination (radians, true of date).

    Return:
        (ra, dec): [tuple of floats] J2000 equatorial coordinates (radians).
    """

    cdef double x, y, z

    x = cos(dec)*cos(ra)
    y = cos(dec)*sin(ra)
    z = sin(dec)
    x, y, z = nutationRotate(jd, x, y, z, True)
    ra = (atan2(y, x) + 2*pi)%(2*pi)
    dec = atan2(z, sqrt(x*x + y*y))

    return equatorialCoordPrecession(jd, J2000_DAYS, ra, dec)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef (double, double, double) equatorialCoordAndRotPrecession(double start_epoch, double final_epoch,
                                                               double ra, double dec, double rot_angle):
    """ Transforms right ascension, declination and the rotation wrt standard angle (pos_angle_ref) from the true
        equator and equinox of one epoch to that of another: nutation of the start epoch is removed, the mean
        precession applied, and the nutation of the final epoch added. J2000 (J2000_DAYS) is treated as the mean,
        catalog frame and gets no nutation. The rotation angle changes by the angle between the two frames' north
        directions at the transformed point, so a gnomonic projection defined in one frame is exactly the same
        projection in the other.
    
    Arguments:
        start_epoch: [float] Julian date of the starting epoch.
        final_epoch: [float] Julian date of the final epoch.
        ra: [float] Input right ascension (radians).
        dec: [float] Input declination (radians).
        rot_angle [float] the rotation wrt Standard angle, aka pos_angle_ref (radians).
    
    Return:
        (ra, dec, rot_angle): [tuple of floats] Precessed equatorial coordinates and rotation angle (radians).
    """
    
    # Don't transform if the start and final epoch are the same
    if start_epoch == final_epoch:
        return ra, dec, rot_angle

    cdef:
        np.ndarray[double, ndim=1] vec, north
        np.ndarray[double, ndim=2] P
        double ra_precessed, dec_precessed, T, t, zeta, z, theta
        double x, y, zz, new_rot_angle, rotation_change

    # Calculate precession parameters (mean equinox of the start epoch to that of the final epoch)
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

    # Pointing direction, and the local north direction at it (a physical direction fixed to the sensor)
    vec = raDecToCartesian(ra, dec)
    north = np.array([-sin(dec)*cos(ra), -sin(dec)*sin(ra), cos(dec)])

    # Full rotation: true of date (start) -> mean (start) -> mean (final) -> true of date (final)
    if start_epoch != J2000_DAYS:
        x, y, zz = nutationRotate(start_epoch, vec[0], vec[1], vec[2], True)
        vec = np.array([x, y, zz])
        x, y, zz = nutationRotate(start_epoch, north[0], north[1], north[2], True)
        north = np.array([x, y, zz])

    vec = np.dot(P, vec)
    north = np.dot(P, north)

    if final_epoch != J2000_DAYS:
        x, y, zz = nutationRotate(final_epoch, vec[0], vec[1], vec[2], False)
        vec = np.array([x, y, zz])
        x, y, zz = nutationRotate(final_epoch, north[0], north[1], north[2], False)
        north = np.array([x, y, zz])

    # Convert the transformed pointing back to RA, Dec
    vec /= np.linalg.norm(vec)
    ra_precessed, dec_precessed = cartesianToRaDec(vec)

    # The transported north direction, measured against the final frame's east and north at the new point,
    # gives the rotation of the field: this is how much a row of pixels turns relative to the local parallel
    rotation_change = atan2(
        np.dot(north, np.array([-sin(ra_precessed), cos(ra_precessed), 0.0])),
        np.dot(north, np.array([-sin(dec_precessed)*cos(ra_precessed), -sin(dec_precessed)*sin(ra_precessed),
            cos(dec_precessed)]))
        )

    # Apply the rotation change to the initial rotation angle
    new_rot_angle = rot_angle + rotation_change

    # Normalize the new rotation angle to be between -pi and pi
    new_rot_angle = fmod(new_rot_angle + M_PI, 2*M_PI) - M_PI

    return ra_precessed, dec_precessed, new_rot_angle


# International Standard Atmosphere in the troposphere: T/T0 = 1 - ISA_LAPSE*h and P/P0 = (T/T0)**ISA_PRESSURE_EXP,
#   so the refractivity of the air (n - 1), proportional to P/T, falls as (T/T0)**(ISA_PRESSURE_EXP - 1). Above the
#   tropopause the temperature is constant and the pressure falls exponentially.
cdef double ISA_LAPSE = 2.25577e-5
cdef double ISA_PRESSURE_EXP = 5.25588
cdef double ISA_TROPOPAUSE = 11000.0
cdef double ISA_STRATOSPHERE_SCALE = 6341.6


cdef double isaRefractivity(double h):
    """ Refractivity of the International Standard Atmosphere at height h (m above sea level), relative to sea
        level. """

    if h < -500.0:
        h = -500.0

    if h <= ISA_TROPOPAUSE:
        return (1.0 - ISA_LAPSE*h)**(ISA_PRESSURE_EXP - 1.0)

    return (1.0 - ISA_LAPSE*ISA_TROPOPAUSE)**(ISA_PRESSURE_EXP - 1.0)*exp(-(h - ISA_TROPOPAUSE)/ISA_STRATOSPHERE_SCALE)


cdef double isaRefractivityIntegral(double h):
    """ Integral of the relative ISA refractivity from sea level to height h (m). Tends to 8434 m. """

    cdef double j_tropopause

    if h < -500.0:
        h = -500.0

    if h <= ISA_TROPOPAUSE:
        return (1.0 - (1.0 - ISA_LAPSE*h)**ISA_PRESSURE_EXP)/(ISA_LAPSE*ISA_PRESSURE_EXP)

    j_tropopause = (1.0 - (1.0 - ISA_LAPSE*ISA_TROPOPAUSE)**ISA_PRESSURE_EXP)/(ISA_LAPSE*ISA_PRESSURE_EXP)

    return j_tropopause + isaRefractivity(ISA_TROPOPAUSE)*ISA_STRATOSPHERE_SCALE \
        *(1.0 - exp(-(h - ISA_TROPOPAUSE)/ISA_STRATOSPHERE_SCALE))


cpdef double refractionScale(double elev_obs):
    """ Scale of the atmospheric refraction at the observer's height, relative to sea level. The refraction is
        proportional to the refractivity of the air at the observer, i.e. to its pressure over temperature, taken
        here from the International Standard Atmosphere: 1.0 at sea level, 0.93 at 700 m, 0.82 at 2000 m, 0.76 at
        2800 m. The weather is not modelled: +-10 K or +-10 hPa change the refraction by about 3.5% or 1%.

    Arguments:
        elev_obs: [float] Height of the observer above sea level (m).

    Return:
        [float] Factor to multiply the sea-level refraction with.
    """

    if elev_obs > ISA_TROPOPAUSE:
        elev_obs = ISA_TROPOPAUSE

    return isaRefractivity(elev_obs)


cpdef double refractionTargetFraction(double elev_obs, double target_height):
    """ Fraction of the refraction of a star that applies to a target at a finite height, e.g. a meteor or a
        contrail. The light from a target inside or just above the atmosphere is bent only by the air between
        the target and the observer, and the straight line to the target differs from the arrival direction of
        the light by the path average of the bending accumulated along the way. With the ISA refractivity
        profile this gives, seen from sea level: 1 for a star, 0.92 for a meteor at 100 km, 0.72 for a target at
        30 km and 0.38 for a contrail at 10 km. Flat-atmosphere approximation, independent of the elevation
        angle: within 2% of the star refraction for targets below 30 km at any elevation, and within 2% (6") of
        it above 10 deg elevation for a target at 100 km, where the curvature of the Earth starts to matter.

    Arguments:
        elev_obs: [float] Height of the observer above sea level (m).
        target_height: [float] Height of the target above sea level (m). Infinity, or anything above 1000 km,
            means a star.

    Return:
        [float] Factor in [0, 1] to multiply the star refraction with (on top of refractionScale).
    """

    cdef double height_diff

    if isinf(target_height) or (target_height > 1.0e6):
        return 1.0

    height_diff = target_height - elev_obs

    if height_diff <= 1.0:
        return 0.0

    if elev_obs > ISA_TROPOPAUSE:
        elev_obs = ISA_TROPOPAUSE

    return 1.0 - (isaRefractivityIntegral(target_height) - isaRefractivityIntegral(elev_obs)) \
        /(height_diff*isaRefractivity(elev_obs))



@cython.cdivision(True)
cdef double refractionApparentToTrue(double elev, double scale=1.0):
    """ Correct the apparent elevation of a star for refraction to true elevation. Standard conditions at sea
        level (Bennett's formula); the observer's height is taken into account through the scale. 

        Source: Explanatory Supplement to the Astronomical Almanac (1992), p. 144.

    Arguments:
        elev: [float] Apparent elevation (radians).
        scale: [float] Scale of the refraction for the observer's height, from refractionScale(). 1.0 by
            default (sea level). Multiply by refractionTargetFraction() for a target at a finite height.

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
    refraction = scale*radians(1.0/(60*tan(radians(degrees(elev_calc) + 7.31/(degrees(elev_calc) + 4.4)))))

    # Correct the elevation
    return elev - refraction



cpdef double pyRefractionApparentToTrue(double elev, double scale=1.0):
    """ Python version of the refraction correction (apparent to true).

    Arguments:
        elev: [float] Apparent elevation (radians).

    Return:
        [float] True elevation (radians).

    """

    return refractionApparentToTrue(elev, scale)



cpdef (double, double) eqRefractionApparentToTrue(double ra, double dec, double jd, double lat, double lon, \
    double scale=1.0):
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
    alt = refractionApparentToTrue(alt, scale)

    # Convert back to equatorial
    ra, dec = cyaltAz2RADec(azim, alt, jd, lat, lon)

    # Precess RA/Dec from the epoch of date to J2000
    ra, dec = equatorialCoordPrecession(jd, J2000_DAYS, ra, dec)


    return (ra, dec)



@cython.cdivision(True)
cdef double refractionTrueToApparent(double elev, double scale=1.0):
    """ Correct the true elevation of a star for refraction to apparent elevation, as the exact inverse of
        refractionApparentToTrue (solved by fixed-point iteration, which converges to well below 0.001 arcsec
        in a few steps because the refraction changes slowly with elevation). The two directions therefore
        close on each other; the previous separate formula (Saemundsson 1986) differed from the inverse of the
        apparent-to-true one by up to 4 arcsec, which put a bias of that size between the star fit (done in
        image coordinates, true to apparent) and the measurements (apparent to true).

    Arguments:
        elev: [float] True elevation (radians).
        scale: [float] Scale of the refraction for the observer's height, from refractionScale(). 1.0 by
            default (sea level). Multiply by refractionTargetFraction() for a target at a finite height.

    Return:
        [float] Apparent elevation (radians).

    """

    cdef double elev_app = elev
    cdef int i

    for i in range(5):
        elev_app = elev + (elev_app - refractionApparentToTrue(elev_app, scale))

    return elev_app



cpdef double pyRefractionTrueToApparent(double elev, double scale=1.0):
    """ Python version of the refraction correction (true to apparent).

    Arguments:
        elev: [float] Apparent elevation (radians).

    Return:
        [float] True elevation (radians).

    """

    return refractionTrueToApparent(elev, scale)



cpdef (double, double) eqRefractionTrueToApparent(double ra, double dec, double jd, double lat, double lon, \
    double scale=1.0):
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
    alt = refractionTrueToApparent(alt, scale)

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
    bool refraction=True, double refraction_scale=1.0):
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
        refraction_scale: [float] Scale of the refraction for the observer's height above sea level, from
            refractionScale(). 1.0 by default (sea level).

    Return:
        (azim, elev): [tuple]
            azim: [float] Azimuth (+east of due north) in radians (epoch of date).
            elev: [float] Elevation above horizon in radians (epoch of date).

        """

    cdef double azim, elev


    # Precession and nutation: catalog J2000 -> true equator and equinox of date
    ra, dec = trueOfDateFromJ2000(jd, ra, dec)

    # Convert to alt/az
    azim, elev = cyraDec2AltAz(ra, dec, jd, lat, lon)

    # Correct elevation for refraction
    if refraction:
        elev = refractionTrueToApparent(elev, refraction_scale)


    return (azim, elev)



@cython.boundscheck(False)
@cython.wraparound(False)
def cyTrueRaDec2ApparentAltAz_vect(np.ndarray[FLOAT_TYPE_t, ndim=1] ra_arr, \
    np.ndarray[FLOAT_TYPE_t, ndim=1] dec_arr, np.ndarray[FLOAT_TYPE_t, ndim=1] jd_arr, \
    double lat, double lon, bool refraction=True, double refraction_scale=1.0):
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
        refraction_scale: [float] Scale of the refraction for the observer's height above sea level, from
            refractionScale(). 1.0 by default (sea level).
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
            refraction=refraction, refraction_scale=refraction_scale)

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
    bool refraction=True, double refraction_scale=1.0):
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
        refraction_scale: [float] Scale of the refraction for the observer's height above sea level, from
            refractionScale(). 1.0 by default (sea level).

    Return:
        (ra, dec): [tuple]
            ra: [float] Right ascension (radians, J2000).
            dec: [float] Declination (radians, J2000).
    """


    cdef double ra, dec


    # Correct elevation for refraction
    if refraction:
        elev = refractionApparentToTrue(elev, refraction_scale)

    # Convert to RA/Dec (true, epoch of date)
    ra, dec = cyaltAz2RADec(azim, elev, jd, lat, lon)

    # Nutation and precession: true equator and equinox of date -> catalog J2000
    ra, dec = j2000FromTrueOfDate(jd, ra, dec)


    return (ra, dec)


@cython.boundscheck(False)
@cython.wraparound(False)
def cyApparentAltAz2TrueRADec_vect(np.ndarray[FLOAT_TYPE_t, ndim=1] azim_arr, np.ndarray[FLOAT_TYPE_t, ndim=1] elev_arr,
    double jd, double lat, double lon, bool refraction=True, double refraction_scale=1.0):
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
        refraction_scale: [float] Scale of the refraction for the observer's height above sea level, from
            refractionScale(). 1.0 by default (sea level).
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
            refraction=refraction, refraction_scale=refraction_scale)

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




### Annual aberration ###

# Constant of aberration kappa = n*a / (c*sqrt(1 - e^2)), the mean orbital speed of the Earth in units of c
cdef double ABERRATION_CONSTANT = radians(20.49552/3600.0)

# Mean obliquity of the ecliptic at J2000
cdef double MEAN_OBLIQUITY_J2000 = radians(23.4392911)


cdef (double, double, double) earthVelocityJ2000(double jd):
    """ Heliocentric velocity of the Earth at the given Julian date, in units of c, in equatorial axes
        (mean equinox and obliquity of J2000).

        Keplerian two-body velocity from the low-precision solar solution (Meeus, Astronomical Algorithms,
        ch. 25) with the longitudes referred to the J2000 equinox, within 0.05 arcsec of the IAU 2006
        barycentric velocity. The Moon's ~13 m/s and the planetary perturbations are neglected.

    Arguments:
        jd: [float] Julian date.

    Return:
        (vx, vy, vz): [tuple of floats] Velocity components in units of c (equatorial J2000 axes).
    """

    cdef double T, L0, M, e, C, sun_lon, earth_lon, perihelion_lon, prec, vx_ecl, vy_ecl

    T = (jd - J2000_DAYS)/36525.0

    # Geometric mean longitude and mean anomaly of the Sun, eccentricity of the Earth's orbit
    L0 = radians((280.46646 + 36000.76983*T + 0.0003032*T*T)%360.0)
    M = radians((357.52911 + 35999.05029*T - 0.0001537*T*T)%360.0)
    e = 0.016708634 - 0.000042037*T - 0.0000001267*T*T

    # Equation of the centre, true geocentric longitude of the Sun
    C = radians((1.914602 - 0.004817*T - 0.000014*T*T)*sin(M) + (0.019993 - 0.000101*T)*sin(2*M) \
        + 0.000289*sin(3*M))
    sun_lon = L0 + C

    # True heliocentric longitude of the Earth, and the longitude of the Earth's perihelion: the Sun's perigee
    #   longitude (mean longitude - mean anomaly) plus 180 deg. Both are referred to the mean equinox of date;
    #   subtract the general precession in longitude to refer them to the J2000 equinox, so the velocity comes
    #   out in J2000 axes
    prec = radians(1.3969713*T)
    earth_lon = sun_lon + pi - prec
    perihelion_lon = L0 - M + pi - prec

    # Keplerian velocity in the ecliptic plane: kappa*(-(sin L + e sin w), cos L + e cos w)
    vx_ecl = -ABERRATION_CONSTANT*(sin(earth_lon) + e*sin(perihelion_lon))
    vy_ecl = ABERRATION_CONSTANT*(cos(earth_lon) + e*cos(perihelion_lon))

    # Ecliptic -> equatorial axes
    return vx_ecl, vy_ecl*cos(MEAN_OBLIQUITY_J2000), vy_ecl*sin(MEAN_OBLIQUITY_J2000)


cdef (double, double) shiftDirection(double ra, double dec, double vx, double vy, double vz):
    """ Direction (ra, dec) shifted by a small velocity vector: the first-order aberration formula
        p' = (p + v/c)/|p + v/c|.
    """

    cdef double x, y, z, n

    x = cos(dec)*cos(ra) + vx
    y = cos(dec)*sin(ra) + vy
    z = sin(dec) + vz
    n = sqrt(x*x + y*y + z*z)

    return (atan2(y, x) + 2*pi)%(2*pi), asin(z/n)


cpdef (double, double) applyAberration(double ra, double dec, double jd):
    """ Annual aberration: catalog (barycentric) direction -> apparent direction seen from the moving Earth.
        Stars are displaced by up to 20.5 arcsec towards the apex of the Earth's motion.

    Arguments:
        ra: [float] Right ascension (radians, J2000).
        dec: [float] Declination (radians, J2000).
        jd: [float] Julian date.

    Return:
        (ra, dec): [tuple of floats] Apparent direction (radians, J2000 axes).
    """

    cdef double vx, vy, vz

    vx, vy, vz = earthVelocityJ2000(jd)

    return shiftDirection(ra, dec, vx, vy, vz)


cpdef (double, double) removeAberration(double ra, double dec, double jd):
    """ Inverse of applyAberration: apparent direction -> catalog direction. First-order inverse, exact to
        ~0.002 arcsec.
    """

    cdef double vx, vy, vz

    vx, vy, vz = earthVelocityJ2000(jd)

    return shiftDirection(ra, dec, -vx, -vy, -vz)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cyraDecToXY(np.ndarray[FLOAT_TYPE_t, ndim=1] ra_data,
    np.ndarray[FLOAT_TYPE_t, ndim=1] dec_data, double jd, double lat, double lon, double x_res,
    double y_res, double h0, double jd_ref, double ra_ref, double dec_ref, double pos_angle_ref, 
    double pix_scale, np.ndarray[FLOAT_TYPE_t, ndim=1] x_poly_rev, 
    np.ndarray[FLOAT_TYPE_t, ndim=1] y_poly_rev, str dist_type, bool refraction=True, bool equal_aspect=False, 
    bool force_distortion_centre=False, bool asymmetry_corr=True, bool aberration=True, double refraction_scale=1.0):
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
        refraction_scale: [float] Scale of the refraction for the observer's height above sea level, from
            refractionScale(). 1.0 by default (sea level).
        equal_aspect: [bool] Force the X/Y aspect ratio to be equal. Used only for radial distortion. \
            False by default.
        force_distortion_centre: [bool] Force the distortion centre to the image centre. False by default.
        aberration: [bool] Displace the input directions by the annual aberration of light (up to 20.5 arcsec
            towards the apex of the Earth's motion), i.e. treat them as catalog directions of distant sources.
            True by default. Set False for directions already in the Earth's frame (e.g. an object in the
            atmosphere), which are not aberrated.
        asymmetry_corr: [bool] Correct the distortion for asymmetry. Only for radial distortion. True by
            default.
    
    Return:
        (x, y): [tuple of ndarrays] Image X and Y coordinates.
    """

    cdef int i
    cdef double ra_centre, dec_centre, ra, dec
    cdef double vx, vy, vz
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
            refraction=refraction, refraction_scale=refraction_scale
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
    # Earth velocity for the annual aberration of the catalog directions
    vx = vy = vz = 0.0
    if aberration:
        vx, vy, vz = earthVelocityJ2000(jd)

    for i in range(ra_data.shape[0]):

        ra = radians(ra_data[i])
        dec = radians(dec_data[i])

        # Annual aberration: catalog direction -> apparent direction seen from the moving Earth
        if aberration:
            ra, dec = shiftDirection(ra, dec, vx, vy, vz)

        ### Gnomonization of star coordinates to image coordinates ###

        # Apply refraction
        if refraction:
            ra, dec = eqRefractionTrueToApparent(ra, dec, jd, radians(lat), radians(lon), refraction_scale)


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


def cyRaDecToXY_iter(np.ndarray[FLOAT_TYPE_t, ndim=1] ra_data,
    np.ndarray[FLOAT_TYPE_t, ndim=1] dec_data, double jd, double lat, double lon, double x_res,
    double y_res, double h0, double jd_ref, double ra_ref, double dec_ref, double pos_angle_ref, 
    double pix_scale, np.ndarray[FLOAT_TYPE_t, ndim=1] x_poly_fwd, 
    np.ndarray[FLOAT_TYPE_t, ndim=1] y_poly_fwd, str dist_type, bool refraction=True, bool equal_aspect=False, 
    bool force_distortion_centre=False, bool asymmetry_corr=True, bool aberration=True, double refraction_scale=1.0):
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
        aberration: [bool] Displace the input directions by the annual aberration of light, as cyraDecToXY
            does. True by default (catalog directions); False for measured directions.
        refraction_scale: [float] Scale of the refraction for the observer's height above sea level, from
            refractionScale(elev). 1.0 by default (sea level).
    
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
    cdef double vx, vy, vz

    # Init output arrays
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] x_array = np.zeros_like(ra_data)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] y_array = np.zeros_like(ra_data)

    # Correct the pointing for precession (output in radians)
    ra_centre, dec_centre, pos_angle_ref = pointingCorrection(
            jd, radians(lat), radians(lon), 
            radians(h0), jd_ref, radians(ra_ref), radians(dec_ref), radians(pos_angle_ref), 
            refraction=refraction, refraction_scale=refraction_scale
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
    # Earth velocity for the annual aberration of the catalog directions
    vx = vy = vz = 0.0
    if aberration:
        vx, vy, vz = earthVelocityJ2000(jd)

    for i in range(ra_data.shape[0]):

        # Read the next coordinate
        ra = radians(ra_data[i])
        dec = radians(dec_data[i])

        # Annual aberration: catalog direction -> apparent direction seen from the moving Earth
        if aberration:
            ra, dec = shiftDirection(ra, dec, vx, vy, vz)

        # Apply refraction
        if refraction:
            ra, dec = eqRefractionTrueToApparent(ra, dec, jd, radians(lat), radians(lon), refraction_scale)

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
    bool refraction=True, double refraction_scale=1.0
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
        refraction_scale: [float] Scale of the refraction for the observer's height above sea level, from
            refractionScale(). 1.0 by default (sea level).

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
        ra_ref_now_corr, dec_ref_corr = eqRefractionTrueToApparent(ra_ref_now, dec_ref, jd, lat, lon, refraction_scale)

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
    bool asymmetry_corr=True, bool precompute_pointing_corr=False, bool aberration=True, double refraction_scale=1.0):
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
        refraction_scale: [float] Scale of the refraction for the observer's height above sea level, from
            refractionScale(). 1.0 by default (sea level).
        equal_aspect: [bool] Force the X/Y aspect ratio to be equal. Used only for radial distortion. \
            False by default.
        force_distortion_centre: [bool] Force the distortion centre to the image centre. False by default.
        asymmetry_corr: [bool] Correct the distortion for asymmetry. Only for radial distortion. True by
            default.
        aberration: [bool] Remove the annual aberration of light (up to 20.5 arcsec) so the output is the
            catalog direction of a distant source (a star). True by default. Set False for an object in the
            atmosphere (a meteor), whose light is not aberrated: the output is then the geometric direction
            in the Earth's frame, which is what a trajectory solver uses.
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
            refraction=refraction, refraction_scale=refraction_scale
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
                refraction=refraction, refraction_scale=refraction_scale
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
            ra, dec = eqRefractionApparentToTrue(ra, dec, jd, radians(lat), radians(lon), refraction_scale)

        # Annual aberration: apparent direction seen from the moving Earth -> catalog direction
        if aberration:
            ra, dec = removeAberration(ra, dec, jd)



        # Convert coordinates to degrees
        ra = degrees(ra)
        dec = degrees(dec)


        # Assign values to output list
        ra_data[i] = ra
        dec_data[i] = dec


    return ra_data, dec_data