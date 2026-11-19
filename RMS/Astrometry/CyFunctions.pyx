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
    cdef double ra, dec, mag, elev, lst
    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] filtered_list = np.zeros(shape=(catalog_list.shape[0], \
        catalog_list.shape[1]), dtype=FLOAT_TYPE)

    cdef np.ndarray[INT_TYPE_t, ndim=1] filtered_indices = np.zeros(shape=(catalog_list.shape[0]), \
        dtype=INT_TYPE)

    # Local sidereal time, the same for all stars
    lst = radians(cyjd2LST(jd, lon))

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
            _, elev = raDec2AltAzLST(radians(ra), radians(dec), lst, radians(lat))


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


cdef double equationOfEquinoxes(double jd):
    """ Equation of the equinoxes: apparent minus mean sidereal time, dpsi*cos(eps), from the same nutation
        series as the frame rotation below. Up to about 1.1 s of time, i.e. 15.8 arcsec of hour angle, over
        the 18.6-year nutation cycle. Hour angles of true-of-date right ascensions must be taken against the
        apparent sidereal time, otherwise the alt/az is off by this much.

    Arguments:
        jd: [float] Julian date.

    Return:
        [float] Equation of the equinoxes (deg).

    """

    cdef double T, dpsi, deps

    T = (jd - J2000_DAYS)/36525.0
    dpsi, deps = nutationComponents(T)

    return degrees(dpsi*cos(meanObliquity(T) + deps))



cpdef double cyjd2GST(double jd):
    """ Apparent Greenwich sidereal time: the mean sidereal time (Meeus, Astronomical Algorithms, eq. 12.4)
        plus the equation of the equinoxes. This is the sidereal time that goes with true-of-date right
        ascensions, which is the frame of the platepar reference pointing and of trueOfDateFromJ2000.
        JD2HourAngle and JD2LST in RMS.Astrometry.Conversions call this, so the three stay identical.

    Arguments:
        jd: [float] Julian date.

    Return:
        [float] Apparent Greenwich sidereal time (deg), in [0, 360).

    """

    cdef double gst

    # Mean sidereal rotation of the Earth
    gst = meanSiderealTime(jd)

    # Apparent sidereal time
    gst += equationOfEquinoxes(jd)

    return (gst%360 + 360)%360



cdef double meanSiderealTime(double jd):
    """ Mean Greenwich sidereal time (Meeus, Astronomical Algorithms, eq. 12.4), not normalized.

    Arguments:
        jd: [float] Julian date.

    Return:
        [float] Mean Greenwich sidereal time (deg).

    """

    cdef double t = (jd - J2000_DAYS)/36525.0

    return 280.46061837 + 360.98564736629*(jd - J2000_DAYS) + 0.000387933*t**2 - (t**3)/38710000.0



cdef double cyjd2LST(double jd, double lon):
    """ Apparent local sidereal time.

    Arguments:
        jd: [float] Julian date.
        lon: [float] Longitude of the observer in degrees.

    Return:
        [float] Apparent local sidereal time (deg), in [0, 360).

    """

    return (cyjd2GST(jd) + lon + 360)%360



cdef struct FrameRotation:
    # Mean precession angles between the two epochs (Meeus, Astronomical Algorithms, eq. 21.2)
    double zeta
    double z
    double theta
    # Nutation of the start epoch, removed first (not applied when the start epoch is J2000)
    bint nutate_start
    double dpsi_start
    double deps_start
    double eps_start
    # Nutation of the final epoch, applied last (not applied when the final epoch is J2000)
    bint nutate_final
    double dpsi_final
    double deps_final
    double eps_final



cdef FrameRotation frameRotation(double start_epoch, double final_epoch):
    """ Parameters of the rotation from the true equator and equinox of one epoch to that of another. J2000
        (J2000_DAYS) is treated as the mean, catalog frame and gets no nutation. Evaluated once so that
        several vectors can be rotated with the same parameters (see frameRotateVector).

    Arguments:
        start_epoch: [float] Julian date of the starting epoch.
        final_epoch: [float] Julian date of the final epoch.

    Return:
        [FrameRotation] Precession angles and nutation components of both epochs.

    """

    cdef FrameRotation f
    cdef double T

    # Mean precession from the mean equinox of the start epoch to that of the final epoch
    f = precessionRotation(start_epoch, final_epoch)

    # Nutation of each epoch that is not J2000
    f.nutate_start = start_epoch != J2000_DAYS
    if f.nutate_start:
        T = (start_epoch - J2000_DAYS)/36525.0
        f.dpsi_start, f.deps_start = nutationComponents(T)
        f.eps_start = meanObliquity(T)

    f.nutate_final = final_epoch != J2000_DAYS
    if f.nutate_final:
        T = (final_epoch - J2000_DAYS)/36525.0
        f.dpsi_final, f.deps_final = nutationComponents(T)
        f.eps_final = meanObliquity(T)

    return f



cdef FrameRotation precessionRotation(double start_epoch, double final_epoch):
    """ The mean precession part of frameRotation, without the nutation of either epoch.

    Arguments:
        start_epoch: [float] Julian date of the starting epoch.
        final_epoch: [float] Julian date of the final epoch.

    Return:
        [FrameRotation] Precession angles, nutation off at both epochs.

    """

    cdef FrameRotation f
    cdef double T, t

    # Mean precession from the mean equinox of the start epoch to that of the final epoch
    T = (start_epoch - J2000_DAYS)/36525.0
    t = (final_epoch - start_epoch)/36525.0

    f.zeta = radians(((2306.2181 + 1.39656*T - 0.000139*T**2)*t + (0.30188 - 0.000344*T)*t**2 \
        + 0.017998*t**3)/3600)
    f.z = radians(((2306.2181 + 1.39656*T - 0.000139*T**2)*t + (1.09468 + 0.000066*T)*t**2 \
        + 0.018203*t**3)/3600)
    f.theta = radians(((2004.3109 - 0.85330*T - 0.000217*T**2)*t - (0.42665 + 0.000217*T)*t**2 \
        - 0.041833*t**3)/3600)

    # No nutation
    f.nutate_start = f.nutate_final = False
    f.dpsi_start = f.deps_start = f.eps_start = 0.0
    f.dpsi_final = f.deps_final = f.eps_final = 0.0

    return f



cdef (double, double, double) nutationRotate(double dpsi, double deps, double eps, double x, double y, \
    double z, bint inverse):
    """ Rotate a unit vector between the mean and the true equator and equinox of date.

        Nutation is a rotation about the ecliptic pole by the nutation in longitude, combined with the
        change of obliquity: N = R1(-(eps + deps)) * R3(-dpsi) * R1(eps) takes a vector from the mean
        equator and equinox of date to the true equator and equinox of date (the frame the Earth actually
        rotates in, and the one the sidereal time refers to). inverse=True applies N^T (true -> mean).

    Arguments:
        dpsi: [float] Nutation in longitude (radians).
        deps: [float] Nutation in obliquity (radians).
        eps: [float] Mean obliquity of the ecliptic (radians).
        x, y, z: [float] Unit vector components (equatorial axes).
        inverse: [bool] False: mean -> true, True: true -> mean.

    Return:
        (x, y, z): [tuple of floats] Rotated vector.

    """

    cdef double eps1, x1, y1, z1, x2, y2, z2

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



cdef (double, double, double) frameRotateVector(FrameRotation f, double x, double y, double z):
    """ Rotate a unit vector from the true equator and equinox of the start epoch to that of the final
        epoch: nutation of the start epoch removed, mean precession applied, nutation of the final epoch
        added. The precession is the same Meeus formulation as equatorialCoordPrecession, written for a
        vector: R3(-z) * R2(theta) * R3(-zeta).

    Arguments:
        f: [FrameRotation] Parameters from frameRotation().
        x, y, z: [float] Unit vector components (equatorial axes).

    Return:
        (x, y, z): [tuple of floats] Rotated vector.

    """

    cdef double x1, y1, z1, a, b, c

    if f.nutate_start:
        x, y, z = nutationRotate(f.dpsi_start, f.deps_start, f.eps_start, x, y, z, True)

    # R3(-zeta): right ascension increases by zeta
    x1 = x*cos(f.zeta) - y*sin(f.zeta)
    y1 = x*sin(f.zeta) + y*cos(f.zeta)
    z1 = z

    # R2(theta): tilt towards the new pole (the A, B, C of Meeus eq. 21.4)
    a = y1
    b = cos(f.theta)*x1 - sin(f.theta)*z1
    c = sin(f.theta)*x1 + cos(f.theta)*z1

    # R3(-z): right ascension increases by z
    x = b*cos(f.z) - a*sin(f.z)
    y = b*sin(f.z) + a*cos(f.z)
    z = c

    if f.nutate_final:
        x, y, z = nutationRotate(f.dpsi_final, f.deps_final, f.eps_final, x, y, z, False)

    return x, y, z



cdef (double, double) frameRotateRaDec(FrameRotation f, double ra, double dec):
    """ frameRotateVector for a direction given as right ascension and declination.

    Arguments:
        f: [FrameRotation] Parameters from frameRotation().
        ra: [float] Right ascension in the start frame (radians).
        dec: [float] Declination in the start frame (radians).

    Return:
        (ra, dec): [tuple of floats] Right ascension and declination in the final frame (radians).

    """

    cdef double x, y, z

    x, y, z = frameRotateVector(f, cos(dec)*cos(ra), cos(dec)*sin(ra), sin(dec))

    return (atan2(y, x) + 2*pi)%(2*pi), atan2(z, sqrt(x*x + y*y))



cpdef (double, double) trueOfDateFromJ2000(double jd, double ra, double dec):
    """ Catalog (mean J2000) right ascension and declination -> true equator and equinox of date, i.e.
        precession followed by nutation. This is the frame that goes with the apparent sidereal time when
        computing hour angles and alt/az, and the frame in which the platepar reference pointing RA_d/dec_d
        is expressed.

    Arguments:
        jd: [float] Julian date of the epoch of date.
        ra: [float] Right ascension (radians, J2000).
        dec: [float] Declination (radians, J2000).

    Return:
        (ra, dec): [tuple of floats] True equatorial coordinates of date (radians).

    """

    return frameRotateRaDec(frameRotation(J2000_DAYS, jd), ra, dec)



cpdef (double, double) j2000FromTrueOfDate(double jd, double ra, double dec):
    """ Inverse of trueOfDateFromJ2000: true equator and equinox of date -> catalog (mean J2000)
        coordinates.

    Arguments:
        jd: [float] Julian date of the epoch of date.
        ra: [float] Right ascension (radians, true of date).
        dec: [float] Declination (radians, true of date).

    Return:
        (ra, dec): [tuple of floats] J2000 equatorial coordinates (radians).

    """

    return frameRotateRaDec(frameRotation(jd, J2000_DAYS), ra, dec)



cdef struct DateFrame:
    # Everything in a J2000 <-> alt/az conversion that depends only on the time and the longitude. The
    #   kernels evaluate it once per Julian date instead of once per point (see dateFrame).
    FrameRotation to_date
    FrameRotation to_j2000
    double lst
    double gst



cdef DateFrame dateFrame(double jd, double lon):
    """ Frame rotations and sidereal times for the given time and place.

    Arguments:
        jd: [float] Julian date.
        lon: [float] Longitude of the observer in radians.

    Return:
        [DateFrame] J2000 -> true of date and true of date -> J2000 rotations, and the local and the
            Greenwich apparent sidereal time (radians).

    """

    cdef DateFrame d
    cdef double T, dpsi, deps, eps, gst

    # The nutation of the date, evaluated once for both rotations and the sidereal time
    T = (jd - J2000_DAYS)/36525.0
    dpsi, deps = nutationComponents(T)
    eps = meanObliquity(T)

    # J2000 -> true of date and back: the mean precession both ways, with the nutation of the date (none at
    #   J2000 itself, as in frameRotation)
    d.to_date = precessionRotation(J2000_DAYS, jd)
    d.to_j2000 = precessionRotation(jd, J2000_DAYS)
    if jd != J2000_DAYS:
        d.to_date.nutate_final = True
        d.to_date.dpsi_final, d.to_date.deps_final, d.to_date.eps_final = dpsi, deps, eps
        d.to_j2000.nutate_start = True
        d.to_j2000.dpsi_start, d.to_j2000.deps_start, d.to_j2000.eps_start = dpsi, deps, eps

    # Apparent sidereal time, mean plus the equation of the equinoxes (as cyjd2GST), for Greenwich and the
    #   observer (as cyjd2LST(jd, 0) and cyjd2LST(jd, lon))
    gst = meanSiderealTime(jd)
    gst += degrees(dpsi*cos(eps + deps))
    gst = (gst%360 + 360)%360
    d.lst = radians((gst + degrees(lon) + 360)%360)
    d.gst = radians((gst + 360)%360)

    return d



cpdef (double, double, double) equatorialCoordAndRotPrecession(double start_epoch, double final_epoch,
                                                               double ra, double dec, double rot_angle):
    """ Transforms right ascension, declination and the rotation wrt standard angle (pos_angle_ref) from
        the true equator and equinox of one epoch to that of another, with the same rotation as
        trueOfDateFromJ2000 / j2000FromTrueOfDate. The rotation angle changes by the angle between the two
        frames' north directions at the transformed point, so a gnomonic projection defined in one frame is
        exactly the same projection in the other.

    Arguments:
        start_epoch: [float] Julian date of the starting epoch.
        final_epoch: [float] Julian date of the final epoch.
        ra: [float] Input right ascension (radians).
        dec: [float] Input declination (radians).
        rot_angle [float] the rotation wrt Standard angle, aka pos_angle_ref (radians).

    Return:
        (ra, dec, rot_angle): [tuple of floats] Precessed equatorial coordinates and rotation angle
            (radians).

    """

    # Don't transform if the start and final epoch are the same
    if start_epoch == final_epoch:
        return ra, dec, rot_angle

    return frameRotateRaDecRot(frameRotation(start_epoch, final_epoch), ra, dec, rot_angle)



cdef (double, double, double) frameRotateRaDecRot(FrameRotation f, double ra, double dec, double rot_angle):
    """ equatorialCoordAndRotPrecession with the frame rotation (frameRotation) given instead of the two
        epochs, so that it is evaluated once per Julian date.

    Arguments:
        f: [FrameRotation] Parameters from frameRotation().
        ra: [float] Input right ascension (radians).
        dec: [float] Input declination (radians).
        rot_angle: [float] The rotation wrt standard angle, aka pos_angle_ref (radians).

    Return:
        (ra, dec, rot_angle): [tuple of floats] Transformed equatorial coordinates and rotation angle
            (radians).

    """

    cdef double x, y, z, nx, ny, nz, ra_precessed, dec_precessed, rotation_change, new_rot_angle
    cdef double ex, ey, px, py, pz

    # Pointing direction, and the local north direction at it (a physical direction fixed to the sensor)
    x, y, z = frameRotateVector(f, cos(dec)*cos(ra), cos(dec)*sin(ra), sin(dec))
    nx, ny, nz = frameRotateVector(f, -sin(dec)*cos(ra), -sin(dec)*sin(ra), cos(dec))

    # Transformed pointing
    ra_precessed = (atan2(y, x) + 2*pi)%(2*pi)
    dec_precessed = atan2(z, sqrt(x*x + y*y))

    # The transported north direction, measured against the final frame's east and north at the new
    #   point, gives the rotation of the field: this is how much a row of pixels turns relative to the
    #   local parallel
    ex, ey = -sin(ra_precessed), cos(ra_precessed)
    px, py, pz = -sin(dec_precessed)*cos(ra_precessed), -sin(dec_precessed)*sin(ra_precessed), \
        cos(dec_precessed)
    rotation_change = atan2(nx*ex + ny*ey, nx*px + ny*py + nz*pz)

    # Apply the rotation change to the initial rotation angle
    new_rot_angle = rot_angle + rotation_change

    # Normalize the new rotation angle to be between -pi and pi
    new_rot_angle = fmod(new_rot_angle + M_PI, 2*M_PI) - M_PI

    return ra_precessed, dec_precessed, new_rot_angle


# International Standard Atmosphere in the troposphere: T/T0 = 1 - ISA_LAPSE*h and
#   P/P0 = (T/T0)**ISA_PRESSURE_EXP, so the refractivity of the air (n - 1), proportional to P/T, falls as
#   (T/T0)**(ISA_PRESSURE_EXP - 1). Above the tropopause the temperature is constant and the pressure
#   falls exponentially.
cdef double ISA_LAPSE = 2.25577e-5
cdef double ISA_PRESSURE_EXP = 5.25588
cdef double ISA_TROPOPAUSE = 11000.0
cdef double ISA_STRATOSPHERE_SCALE = 6341.6


cdef double isaRefractivity(double h):
    """ Refractivity of the International Standard Atmosphere at the given height, relative to sea level.

    Arguments:
        h: [float] Height above sea level (m).

    Return:
        [float] Refractivity (n - 1) relative to its sea-level value.

    """

    cdef double refractivity_tropopause

    # Keep the profile defined below sea level
    if h < -500.0:
        h = -500.0

    # In the troposphere the refractivity falls as a power of the temperature ratio
    if h <= ISA_TROPOPAUSE:
        return (1.0 - ISA_LAPSE*h)**(ISA_PRESSURE_EXP - 1.0)

    # Above the tropopause the temperature is constant, so the refractivity follows the pressure
    refractivity_tropopause = (1.0 - ISA_LAPSE*ISA_TROPOPAUSE)**(ISA_PRESSURE_EXP - 1.0)

    return refractivity_tropopause*exp(-(h - ISA_TROPOPAUSE)/ISA_STRATOSPHERE_SCALE)


cdef double isaRefractivityIntegral(double h):
    """ Integral of the relative ISA refractivity from sea level up to the given height.

    Arguments:
        h: [float] Height above sea level (m).

    Return:
        [float] Integral (m). Tends to 8434 m for large heights.

    """

    cdef double j_tropopause, tail_scale

    # Keep the profile defined below sea level
    if h < -500.0:
        h = -500.0

    # Closed form of the troposphere power law
    if h <= ISA_TROPOPAUSE:
        return (1.0 - (1.0 - ISA_LAPSE*h)**ISA_PRESSURE_EXP)/(ISA_LAPSE*ISA_PRESSURE_EXP)

    # Above the tropopause, add the exponential tail to the integral up to the tropopause
    j_tropopause = (1.0 - (1.0 - ISA_LAPSE*ISA_TROPOPAUSE)**ISA_PRESSURE_EXP)/(ISA_LAPSE*ISA_PRESSURE_EXP)
    tail_scale = isaRefractivity(ISA_TROPOPAUSE)*ISA_STRATOSPHERE_SCALE

    return j_tropopause + tail_scale*(1.0 - exp(-(h - ISA_TROPOPAUSE)/ISA_STRATOSPHERE_SCALE))


cpdef double refractionScale(double elev_obs):
    """ Scale of the atmospheric refraction at the observer's height, relative to sea level. The refraction
        is proportional to the refractivity of the air at the observer, i.e. to its pressure over
        temperature, taken here from the International Standard Atmosphere: 1.0 at sea level, 0.93 at 700 m,
        0.82 at 2000 m and 0.76 at 2800 m. The weather is not modelled, and +-10 K or +-10 hPa change the
        refraction by about 3.5% or 1%.

    Arguments:
        elev_obs: [float] Height of the observer above sea level (m).

    Return:
        [float] Factor to multiply the sea-level refraction with.

    """

    # Clamp to the tropopause, above which the ground-observer model no longer applies
    if elev_obs > ISA_TROPOPAUSE:
        elev_obs = ISA_TROPOPAUSE

    return isaRefractivity(elev_obs)


cpdef double refractionTargetFraction(double elev_obs, double target_height):
    """ Fraction of the refraction of a star that applies to a target at a finite height, e.g. a meteor.
        The light from a target inside or just above the atmosphere is bent only by the air between the
        target and the observer, and the straight line to the target differs from the arrival direction of
        the light by the path average of the bending accumulated along the way. With the ISA refractivity
        profile this gives, seen from sea level: 1 for a star, 0.92 for a meteor at 100 km, 0.72 for a
        target at 30 km and 0.38 for a target at 10 km. Beyond the atmosphere the fraction keeps growing
        towards 1 as 1 - 8.4 km/height, without a jump, and is exactly 1 for an infinite height.

        Flat-atmosphere approximation, independent of the elevation angle. Compared with a ray trace
        through the US Standard Atmosphere, it is within 2% of the star refraction for targets below
        30 km at any elevation; for targets at 100 km and higher it is within 0.6 arcsec at 20 deg elevation
        and above, 1.5-4 arcsec at 10 deg and up to 17 arcsec at 5 deg, where the curvature of the Earth
        starts to matter. For such targets it should not be used much below 15-20 deg elevation.

        Only valid for a target above the observer. The closed form breaks down for a target below the
        observer, where the mean refractivity along the path exceeds the one at the observer and the
        fraction comes out negative, so that case returns 0.

    Arguments:
        elev_obs: [float] Height of the observer above sea level (m).
        target_height: [float] Height of the target above sea level (m). Infinity means a star.

    Return:
        [float] Factor in [0, 1] to multiply the star refraction with (on top of refractionScale).

    """

    cdef double height_diff, path_integral

    # Clamp to the tropopause, above which the ground-observer model no longer applies
    if elev_obs > ISA_TROPOPAUSE:
        elev_obs = ISA_TROPOPAUSE

    height_diff = target_height - elev_obs

    # No air between the observer and the target, or a target below the observer (see the note above)
    if height_diff <= 1.0:
        return 0.0

    # Refractivity integrated over the path, relative to the refractivity at the observer. It is finite
    #   for an infinite height, so a star comes out with exactly the full refraction
    path_integral = isaRefractivityIntegral(target_height) - isaRefractivityIntegral(elev_obs)

    return 1.0 - path_integral/(height_diff*isaRefractivity(elev_obs))



@cython.cdivision(True)
cdef (double, double) bennettRefraction(double elev, double refraction_scale):
    """ Atmospheric refraction at the given apparent elevation, and its derivative with respect to that
        elevation. Standard conditions at sea level; the observer's height enters through the scale. The
        derivative is returned alongside because inverting the formula (refractionTrueToApparent) needs it
        and it costs only a few operations once the tangent has been computed.

        Source: Explanatory Supplement to the Astronomical Almanac (1992), p. 144.

    Arguments:
        elev: [float] Apparent elevation (radians).
        refraction_scale: [float] Scale of the refraction for the observer's height, from
            refractionScale().

    Return:
        (refraction, derivative): [tuple]
            refraction: [float] Refraction to subtract from the apparent elevation (radians).
            derivative: [float] Derivative of the refraction with respect to the apparent elevation.

    """

    cdef double elev_deg, arg_slope, cotangent, refraction

    # Don't apply refraction for elevations below -0.5 deg. The refraction is held at its value there, so
    #   it no longer varies with the elevation and its derivative is zero.
    if elev <= radians(-0.5):
        elev_deg = -0.5
        arg_slope = 0.0

    else:
        elev_deg = degrees(elev)
        arg_slope = 1.0 - 7.31/(elev_deg + 4.4)**2

    # Bennett's formula gives the refraction in arc minutes
    cotangent = 1.0/tan(radians(elev_deg + 7.31/(elev_deg + 4.4)))
    refraction = refraction_scale*radians(cotangent/60)

    # d(cot x)/dx = -(1 + cot**2 x), and the degree-to-radian conversions of the two elevations cancel
    return (refraction, -refraction_scale*radians((1.0 + cotangent*cotangent)*arg_slope/60))



cdef double refractionApparentToTrue(double elev, double refraction_scale=1.0):
    """ Correct the apparent elevation of a star for refraction to true elevation. Standard conditions at
        sea level (Bennett's formula); the observer's height is taken into account through the scale.

    Arguments:
        elev: [float] Apparent elevation (radians).

    Keyword arguments:
        refraction_scale: [float] Scale of the refraction for the observer's height, from
            refractionScale(). 1.0 by default (sea level). Multiply by refractionTargetFraction() for a
            target at a finite height.

    Return:
        [float] True elevation (radians).

    """

    cdef double refraction, derivative

    refraction, derivative = bennettRefraction(elev, refraction_scale)

    # Correct the elevation
    return elev - refraction



cpdef double pyRefractionApparentToTrue(double elev, double refraction_scale=1.0):
    """ Python version of the refraction correction (apparent to true).

    Arguments:
        elev: [float] Apparent elevation (radians).

    Keyword arguments:
        refraction_scale: [float] Scale of the refraction for the observer's height, from
            refractionScale(). 1.0 by default (sea level).

    Return:
        [float] True elevation (radians).

    """

    return refractionApparentToTrue(elev, refraction_scale)



cdef (double, double) eqRefraction(DateFrame d, double ra, double dec, double lat, bint true_to_apparent, \
    double refraction_scale):
    """ Refract J2000 equatorial coordinates at their altitude. The body of eqRefractionApparentToTrue and
        eqRefractionTrueToApparent, with the frame rotations and the sidereal time taken from a DateFrame so
        that the kernels compute them once per Julian date.

    Arguments:
        d: [DateFrame] From dateFrame() for the time and the observer's longitude.
        ra: [float] J2000 right ascension in radians.
        dec: [float] J2000 declination in radians.
        lat: [float] Latitude in radians.
        true_to_apparent: [bool] True: true -> apparent, False: apparent -> true.
        refraction_scale: [float] Scale of the refraction for the observer's height, from
            refractionScale().

    Return:
        (ra, dec): [tuple of floats] Refracted J2000 right ascension and declination in radians.

    """

    cdef double azim, alt

    # Precession and nutation: J2000 -> true equator and equinox of date, the frame of the sidereal time
    ra, dec = frameRotateRaDec(d.to_date, ra, dec)

    # Convert coordinates to alt/az
    azim, alt = raDec2AltAzLST(ra, dec, d.lst, lat)

    # Correct the elevation
    if true_to_apparent:
        alt = refractionTrueToApparent(alt, refraction_scale)
    else:
        alt = refractionApparentToTrue(alt, refraction_scale)

    # Convert back to equatorial
    ra, dec = altAz2RADecLST(azim, alt, d.lst, lat)

    # Nutation and precession: true equator and equinox of date -> J2000
    ra, dec = frameRotateRaDec(d.to_j2000, ra, dec)


    return (ra, dec)



cpdef (double, double) eqRefractionApparentToTrue(double ra, double dec, double jd, double lat, double lon, \
    double refraction_scale=1.0):
    """ Correct the equatorial coordinates for refraction. The correction is done from apparent to true
        coordinates.
    
    Arguments:
        ra: [float] J2000 right ascension in radians.
        dec: [float] J2000 declination in radians.
        jd: [float] Julian date.
        lat: [float] latitude in radians.
        lon: [float] longitude in radians.

    Keyword arguments:
        refraction_scale: [float] Scale of the refraction for the observer's height, from
            refractionScale(). 1.0 by default (sea level).

    Return:
        (ra, dec):
            - ra: [float] Refraction corrected (true) right ascension in radians.
            - dec: [float] Refraction corrected (true) declination in radians.

    """

    return eqRefraction(dateFrame(jd, lon), ra, dec, lat, False, refraction_scale)



@cython.cdivision(True)
cdef double refractionTrueToApparent(double elev, double refraction_scale=1.0):
    """ Correct the true elevation of a star for refraction to apparent elevation, as the exact inverse of
        refractionApparentToTrue, so that the two directions close on each other.

        Solves elev_app - refraction(elev_app) = elev for elev_app by Newton's method on Bennett's formula.
        The refraction is smooth and its derivative is at most 0.263 in magnitude at sea level (at -0.5 deg,
        the lowest elevation the formula is evaluated at), so two or three steps reach double precision at
        any elevation, the horizon included.

    Arguments:
        elev: [float] True elevation (radians).

    Keyword arguments:
        refraction_scale: [float] Scale of the refraction for the observer's height, from
            refractionScale(). 1.0 by default (sea level). Multiply by refractionTargetFraction() for a
            target at a finite height.

    Return:
        [float] Apparent elevation (radians).

    """

    cdef double elev_app, refraction, derivative, step
    cdef int i

    elev_app = elev

    for i in range(10):

        refraction, derivative = bennettRefraction(elev_app, refraction_scale)

        # Newton step on elev_app - refraction(elev_app) - elev
        step = (elev + refraction - elev_app)/(1.0 - derivative)
        elev_app += step

        # Newton converges quadratically: after a step below 1e-8 rad the error left is of the order of
        #   1e-15 rad (1e-10 arcsec), so another evaluation would not change the result in practice
        if fabs(step) < 1e-8:
            break

    return elev_app



cpdef double pyRefractionTrueToApparent(double elev, double refraction_scale=1.0):
    """ Python version of the refraction correction (true to apparent).

    Arguments:
        elev: [float] True elevation (radians).

    Keyword arguments:
        refraction_scale: [float] Scale of the refraction for the observer's height, from
            refractionScale(). 1.0 by default (sea level).

    Return:
        [float] Apparent elevation (radians).

    """

    return refractionTrueToApparent(elev, refraction_scale)



cpdef (double, double) eqRefractionTrueToApparent(double ra, double dec, double jd, double lat, double lon, \
    double refraction_scale=1.0):
    """ Correct the equatorial coordinates for refraction. The correction is done from true to apparent
        coordinates.
    
    Arguments:
        ra: [float] J2000 Right ascension in radians.
        dec: [float] J2000 Declination in radians.
        jd: [float] Julian date.
        lat: [float] Latitude in radians.
        lon: [float] Longitude in radians.

    Keyword arguments:
        refraction_scale: [float] Scale of the refraction for the observer's height, from
            refractionScale(). 1.0 by default (sea level).

    Return:
        (ra, dec):
            - ra: [float] Apparent right ascension in radians.
            - dec: [float] Apparent declination in radians.

    """

    return eqRefraction(dateFrame(jd, lon), ra, dec, lat, True, refraction_scale)



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

    # Calculate Local Sidereal Time
    return raDec2AltAzLST(ra, dec, radians(cyjd2LST(jd, degrees(lon))), lat)



@cython.boundscheck(False)
@cython.wraparound(False)
def cyraDec2AltAz_vect(np.ndarray[FLOAT_TYPE_t, ndim=1] ra_arr, np.ndarray[FLOAT_TYPE_t, ndim=1] dec_arr, \
    np.ndarray[FLOAT_TYPE_t, ndim=1] jd_arr, np.ndarray[FLOAT_TYPE_t, ndim=1] lat_arr, \
    np.ndarray[FLOAT_TYPE_t, ndim=1] lon_arr):
    """ cyraDec2AltAz for arrays of the same length. The sidereal time is evaluated only when the time or
        the longitude changes.

    Arguments:
        ra_arr: [ndarray] Right ascension of every point (radians).
        dec_arr: [ndarray] Declination of every point (radians).
        jd_arr: [ndarray] Julian date of every point.
        lat_arr: [ndarray] Latitude of the observer of every point (radians).
        lon_arr: [ndarray] Longitude of the observer of every point (radians).

    Return:
        (azim_arr, elev_arr): [tuple of ndarrays] Azimuth (+east of due north) and elevation in radians.
    """

    cdef int i
    cdef double lst, azim, elev
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] azim_arr = np.zeros_like(ra_arr)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] elev_arr = np.zeros_like(ra_arr)

    for i in range(ra_arr.shape[0]):

        if (i == 0) or (jd_arr[i] != jd_arr[i - 1]) or (lon_arr[i] != lon_arr[i - 1]):
            lst = radians(cyjd2LST(jd_arr[i], degrees(lon_arr[i])))

        azim, elev = raDec2AltAzLST(ra_arr[i], dec_arr[i], lst, lat_arr[i])

        azim_arr[i] = azim
        elev_arr[i] = elev

    return azim_arr, elev_arr



@cython.cdivision(True)
cdef (double, double) raDec2AltAzLST(double ra, double dec, double lst, double lat):
    """ cyraDec2AltAz with the local sidereal time given instead of the time and longitude.

    Arguments:
        ra: [float] Right ascension (radians), in the frame of the sidereal time (true of date).
        dec: [float] Declination (radians).
        lst: [float] Local apparent sidereal time (radians).
        lat: [float] Latitude of the observer (radians).

    Return:
        (azim, elev): [tuple of floats] Azimuth (+east of due north) and elevation (radians).

    """

    cdef double ha, azim, sin_elev, elev

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

    return trueRaDec2ApparentAltAzOfDate(frameRotation(J2000_DAYS, jd), radians(cyjd2LST(jd, degrees(lon))), \
        ra, dec, lat, refraction, refraction_scale)



cdef (double, double) trueRaDec2ApparentAltAzOfDate(FrameRotation to_date, double lst, double ra, double dec, \
    double lat, bint refraction, double refraction_scale):
    """ cyTrueRaDec2ApparentAltAz with the J2000 -> true of date rotation (frameRotation) and the local
        sidereal time given instead of the time and longitude, so that they are evaluated once for all
        points of the same time.

    Arguments:
        to_date: [FrameRotation] J2000 -> true of date rotation, frameRotation(J2000_DAYS, jd).
        lst: [float] Local apparent sidereal time (radians).
        ra: [float] Right ascension (radians, J2000).
        dec: [float] Declination (radians, J2000).
        lat: [float] Latitude of the observer (radians).
        refraction: [bool] Apply the refraction (true -> apparent elevation).
        refraction_scale: [float] Scale of the refraction for the observer's height above sea level, from
            refractionScale().

    Return:
        (azim, elev): [tuple of floats] Azimuth (+east of due north) and elevation (radians).

    """

    cdef double azim, elev


    # Precession and nutation: catalog J2000 -> true equator and equinox of date
    ra, dec = frameRotateRaDec(to_date, ra, dec)

    # Convert to alt/az
    azim, elev = raDec2AltAzLST(ra, dec, lst, lat)

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
    cdef double azim, elev, lst
    cdef FrameRotation to_date
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] azim_arr = np.zeros_like(ra_arr)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] elev_arr = np.zeros_like(ra_arr)


    # Convert all entries
    for i in range(len(ra_arr)):

        # The frame rotation and the sidereal time depend only on the time, evaluated when it changes
        if (i == 0) or (jd_arr[i] != jd_arr[i - 1]):
            to_date = frameRotation(J2000_DAYS, jd_arr[i])
            lst = radians(cyjd2LST(jd_arr[i], degrees(lon)))

        # Compute alt/az
        azim, elev = trueRaDec2ApparentAltAzOfDate(to_date, lst, ra_arr[i], dec_arr[i], lat, refraction, \
            refraction_scale)

        # Assign alt/az to array
        azim_arr[i] = azim
        elev_arr[i] = elev


    return (azim_arr, elev_arr)




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

    # Calculate Local Sidereal Time
    return altAz2RADecLST(azim, elev, radians(cyjd2LST(jd, degrees(lon))), lat)



@cython.boundscheck(False)
@cython.wraparound(False)
def cyaltAz2RADec_vect(np.ndarray[FLOAT_TYPE_t, ndim=1] azim_arr, np.ndarray[FLOAT_TYPE_t, ndim=1] elev_arr, \
    np.ndarray[FLOAT_TYPE_t, ndim=1] jd_arr, np.ndarray[FLOAT_TYPE_t, ndim=1] lat_arr, \
    np.ndarray[FLOAT_TYPE_t, ndim=1] lon_arr):
    """ cyaltAz2RADec for arrays of the same length. The sidereal time is evaluated only when the time or
        the longitude changes.

    Arguments:
        azim_arr: [ndarray] Azimuth (+east of due north) of every point (radians).
        elev_arr: [ndarray] Elevation of every point (radians).
        jd_arr: [ndarray] Julian date of every point.
        lat_arr: [ndarray] Latitude of the observer of every point (radians).
        lon_arr: [ndarray] Longitude of the observer of every point (radians).

    Return:
        (ra_arr, dec_arr): [tuple of ndarrays] Right ascension and declination in radians.
    """

    cdef int i
    cdef double lst, ra, dec
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] ra_arr = np.zeros_like(azim_arr)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] dec_arr = np.zeros_like(azim_arr)

    for i in range(azim_arr.shape[0]):

        if (i == 0) or (jd_arr[i] != jd_arr[i - 1]) or (lon_arr[i] != lon_arr[i - 1]):
            lst = radians(cyjd2LST(jd_arr[i], degrees(lon_arr[i])))

        ra, dec = altAz2RADecLST(azim_arr[i], elev_arr[i], lst, lat_arr[i])

        ra_arr[i] = ra
        dec_arr[i] = dec

    return ra_arr, dec_arr



@cython.cdivision(True)
cdef (double, double) altAz2RADecLST(double azim, double elev, double lst, double lat):
    """ cyaltAz2RADec with the local sidereal time given instead of the time and longitude.

    Arguments:
        azim: [float] Azimuth (+east of due north) (radians).
        elev: [float] Elevation (radians).
        lst: [float] Local apparent sidereal time (radians).
        lat: [float] Latitude of the observer (radians).

    Return:
        (ra, dec): [tuple of floats] Right ascension and declination (radians), in the frame of the
            sidereal time (true of date).

    """

    cdef double ha, ra, dec

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

    return apparentAltAz2TrueRADecOfDate(frameRotation(jd, J2000_DAYS), radians(cyjd2LST(jd, degrees(lon))), \
        azim, elev, lat, refraction, refraction_scale)



cdef (double, double) apparentAltAz2TrueRADecOfDate(FrameRotation to_j2000, double lst, double azim, \
    double elev, double lat, bint refraction, double refraction_scale):
    """ cyApparentAltAz2TrueRADec with the true of date -> J2000 rotation (frameRotation) and the local
        sidereal time given instead of the time and longitude, so that they are evaluated once for all
        points of the same time.

    Arguments:
        to_j2000: [FrameRotation] True of date -> J2000 rotation, frameRotation(jd, J2000_DAYS).
        lst: [float] Local apparent sidereal time (radians).
        azim: [float] Azimuth (+east of due north) (radians).
        elev: [float] Elevation (radians).
        lat: [float] Latitude of the observer (radians).
        refraction: [bool] Remove the refraction (apparent -> true elevation).
        refraction_scale: [float] Scale of the refraction for the observer's height above sea level, from
            refractionScale().

    Return:
        (ra, dec): [tuple of floats] Right ascension and declination (radians, J2000).

    """

    cdef double ra, dec


    # Correct elevation for refraction
    if refraction:
        elev = refractionApparentToTrue(elev, refraction_scale)

    # Convert to RA/Dec (true, epoch of date)
    ra, dec = altAz2RADecLST(azim, elev, lst, lat)

    # Nutation and precession: true equator and equinox of date -> catalog J2000
    ra, dec = frameRotateRaDec(to_j2000, ra, dec)


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
    cdef double ra, dec, lst
    cdef FrameRotation to_j2000
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] ra_arr = np.zeros_like(azim_arr)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] dec_arr = np.zeros_like(azim_arr)


    # The frame rotation and the sidereal time are the same for all points
    to_j2000 = frameRotation(jd, J2000_DAYS)
    lst = radians(cyjd2LST(jd, degrees(lon)))

    for i in range(len(azim_arr)):

        # Compute RA/Dec
        ra, dec = apparentAltAz2TrueRADecOfDate(to_j2000, lst, azim_arr[i], elev_arr[i], lat, refraction, \
            refraction_scale)

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

# Constant of aberration kappa = n*a / (c*sqrt(1 - e^2)), the Earth's mean speed around the Sun in units of c
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

    # Geometric mean longitude and mean anomaly of the Sun, eccentricity of the Earth's path around the Sun
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

    Arguments:
        ra: [float] Right ascension (radians).
        dec: [float] Declination (radians).
        vx, vy, vz: [float] Velocity in units of c, in the axes of ra and dec (see earthVelocityJ2000).

    Return:
        (ra, dec): [tuple of floats] Shifted direction (radians).
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

    Arguments:
        ra: [float] Right ascension (radians, J2000 axes), apparent direction.
        dec: [float] Declination (radians, J2000 axes), apparent direction.
        jd: [float] Julian date.

    Return:
        (ra, dec): [tuple of floats] Catalog direction (radians, J2000).
    """

    cdef double vx, vy, vz

    vx, vy, vz = earthVelocityJ2000(jd)

    return shiftDirection(ra, dec, -vx, -vy, -vz)


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

# Samples along the ray from the distortion centre to an image point found by Newton's method beyond the reach
#   of the radial inverse, which show whether a fold of the mapping lies in between
cdef int FOLD_SAMPLES = 32

# Distance (in half widths of the image) that Newton's method may move a radial model with decentering from the
#   seed given by its radial part, with only the end point checked for a fold. The decentering of fitted plates
#   moves it by a few thousandths at most; farther moves are checked along the whole ray
cdef double FOLD_FREE_MOVE = 0.01

# Fraction of the distance of its seed (the inverse of the linear part) from the image of the corrected origin that
#   Newton's method may move a polynomial model, with only the orientation at the solution checked: the higher
#   terms of fitted plates move it by much less, while a solution on a folded sheet lies far from the seed
cdef double POLY_FOLD_FREE_FRACTION = 0.25


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

    # A target beyond the reach of the branch has no inverse
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
cdef double radialInverse(ForwardModel *m, double x_corr, double y_corr, double r_end, double f_end,
    double *img):
    """ The inverse of the radial part of a radial model (its decentering left out): the direction from the
        distortion centre is kept and only the radius is rescaled, on the rising branch of the model.

    Arguments:
        m: [ForwardModel pointer] The distortion model.
        x_corr: [float] Corrected X coordinate (px).
        y_corr: [float] Corrected Y coordinate (px).
        r_end: [float] End of the rising branch (radialBranchEnd).
        f_end: [float] Corrected radius at the end of the rising branch.
        img: [double pointer] The image coordinates relative to the image centre are written to img[0] and
            img[1], NaN beyond the reach of the branch.

    Return:
        asym: [float] The factor by which the asymmetry stretches the radius in the direction of the corrected
            point (1 without asymmetry). Where it is not positive there is no root, and NaN is written.
    """

    cdef double radius_corr, cos_phi, sin_phi, asym, rho, q

    # The distortion centre is its own image
    radius_corr = sqrt(x_corr*x_corr + y_corr*y_corr)
    if radius_corr == 0:
        img[0] = m.x0
        img[1] = m.y0
        return 1.0

    # The radius of the image point, in the direction of the corrected one, which the asymmetry stretches
    cos_phi = x_corr/radius_corr
    sin_phi = y_corr/radius_corr
    asym = 1.0 + m.a1*(sin_phi*m.cos_a2 - cos_phi*m.sin_a2)
    rho = radialRoot(radius_corr*asym/m.half_width, m.kind, m.k1, m.k2, m.k3, m.k4, r_end, f_end) \
        if asym > 0 else NAN_VALUE
    q = rho*m.half_width/asym

    img[0] = q*cos_phi + m.x0
    img[1] = q*sin_phi/(1.0 + m.xy) + m.y0

    return asym


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

        # Second powers of the coordinates
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

        # The distortion and its Jacobian at the current point, and the squared error there
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


@cython.cdivision(True)
cdef bint beforeFirstFold(ForwardModel *m, double x_start, double y_start, double x_img, double y_img,
    int n_samples):
    """ Whether an image point of a radial model lies before the first fold of its forward distortion, i.e. on
        the sheet of the mapping that holds the image of the sky. The start of the ray (the distortion centre)
        goes to the corrected origin, and along the ray from it to the point the corrected point has to move
        away from that origin all the way: past a fold it turns back, and may come out again on the far side of
        the origin. Checked at evenly spaced samples of the ray, both from one sample to the next and in the
        direction of the ray at each sample.

    Arguments:
        m: [ForwardModel pointer] The distortion model, a radial one.
        x_start: [float] Image X coordinate of the start of the ray, relative to the image centre (px).
        y_start: [float] Image Y coordinate of the start of the ray, relative to the image centre (px).
        x_img: [float] Image X coordinate relative to the image centre (px).
        y_img: [float] Image Y coordinate relative to the image centre (px).
        n_samples: [int] Number of samples, the last one at the point (1 checks only the point itself).

    Return:
        [bool] True if no fold lies between the start of the ray and the point.
    """

    cdef int i
    cdef double out[6]
    cdef double dx = x_img - x_start, dy = y_img - y_start, t, dist2, dist2_prev = 0.0, outward

    # The start of the ray is its own image
    if (dx == 0) and (dy == 0):
        return True

    # The squared distance of the corrected point from the corrected origin has to grow from each sample to
    #   the next, and its derivative along the ray (the corrected point dotted with the Jacobian times the
    #   direction of the ray) has to be positive at each sample
    for i in range(1, n_samples + 1):
        t = <double>i/n_samples
        forwardJacobian(m, x_start + t*dx, y_start + t*dy, out)
        dist2 = out[0]*out[0] + out[1]*out[1]
        outward = out[0]*(out[2]*dx + out[3]*dy) + out[1]*(out[4]*dx + out[5]*dy)
        if (dist2 <= dist2_prev) or (outward <= 0):
            return False

        dist2_prev = dist2

    return True

@cython.cdivision(True)
cdef bint orientedAlongSegment(ForwardModel *m, double x_start, double y_start, double x_img, double y_img,
    int n_samples):
    """ Whether the forward distortion keeps the orientation of the image (a positive Jacobian determinant) all
        along the segment from a start point to an image point. A fold of the mapping is where the determinant
        changes sign, so a point reached from the image of the corrected origin without such a change is on the
        sheet of the mapping that holds the image of the sky. Unlike the radial distance of beforeFirstFold, it
        does not take a shear for a fold, so it is the check of the polynomial models.

    Arguments:
        m: [ForwardModel pointer] The distortion model.
        x_start: [float] Image X coordinate of the start of the segment, relative to the image centre (px).
        y_start: [float] Image Y coordinate of the start of the segment, relative to the image centre (px).
        x_img: [float] Image X coordinate relative to the image centre (px).
        y_img: [float] Image Y coordinate relative to the image centre (px).
        n_samples: [int] Number of samples, the last one at the point.

    Return:
        [bool] True if the determinant is positive at every sample.
    """

    cdef int i
    cdef double out[6]
    cdef double t

    for i in range(1, n_samples + 1):
        t = <double>i/n_samples
        forwardJacobian(m, x_start + t*(x_img - x_start), y_start + t*(y_img - y_start), out)
        if out[2]*out[5] - out[3]*out[4] <= 0:
            return False

    return True

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
    bool invert_forward=False, np.ndarray[FLOAT_TYPE_t, ndim=1] jd_data=None, double refraction_scale=1.0,
    bool aberration=True):
    """ Convert RA, Dec to distortion corrected image coordinates. 

    Arguments:
        RA_data: [ndarray] Array of right ascensions (degrees).
        dec_data: [ndarray] Array of declinations (degrees).
        jd: [float] Julian date.
        lat: [float] Latitude of station in degrees.
        lon: [float] Longitude of station in degrees.
        x_res: [int] X resolution of the camera.
        y_res: [int] Y resolution of the camera.
        h0: [float] Reference hour angle (deg): the apparent Greenwich sidereal time at jd_ref (mean
            sidereal time plus the equation of the equinoxes, see Conversions.JD2HourAngle).
        jd_ref: [float] Reference Julian date of plate solution.
        ra_ref: [float] Reference right ascension of the image centre (degrees), true equator and equinox
            of jd_ref.
        dec_ref: [float] Reference declination of the image centre (degrees), true equator and equinox of
            jd_ref.
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
    cdef double vx, vy, vz

    # The reference position angle as given (it is replaced by its value corrected for the date below), for the
    #   correction at the date of every point
    cdef double pos_angle_ref_in = pos_angle_ref
    cdef double radius, sin_ang, cos_ang, theta, x, y, r, dx, dy, x_img, y_img, r_corr, r_scale
    cdef double x0, y0, xy, a1, a2, k1 = 0, k2 = 0, k3 = 0, k4 = 0, k5 = 0
    cdef double r_end = 0, f_end = 0, asym_factor = 1, seed_x, seed_y
    cdef int fold_samples
    cdef bint moved_far
    cdef double lin_xx = 1, lin_xy = 0, lin_yx = 0, lin_yy = 1, lin_det = 1
    cdef double img[2]
    cdef double poly_origin[2]
    cdef bint seed_beyond_radial = False
    cdef bint radial_root = False
    cdef int j, n_k
    cdef ForwardModel model
    cdef double p1 = 0, p2 = 0, p3 = 0, p4 = 0, u, v, r2, q1, q2
    cdef int index_offset
    cdef DateFrame date_frame

    # Init output arrays
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] x_array = np.zeros_like(ra_data)
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] y_array = np.zeros_like(ra_data)


    # Frame rotations and sidereal time for the pointing and the refraction, the same for all stars
    date_frame = dateFrame(jd, radians(lon))

    # Correct the pointing for precession (output in radians)
    ra_centre, dec_centre, pos_angle_ref = pointingCorrectionOfDate(date_frame, jd, radians(lat), radians(h0),
        radians(ra_ref), radians(dec_ref), radians(pos_angle_ref), refraction, refraction_scale)

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

        # radial3-odd has no k2: its polynomial ends at k1, and reading past it is out of bounds
        k2 = 0.0
        if x_poly_rev.shape[0] > (6 - index_offset):
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

        # The coefficients, zeros past the end of the arrays
        for j in range(14):
            model.cx[j] = x_poly_rev[j] if j < x_poly_rev.shape[0] else 0.0
            model.cy[j] = y_poly_rev[j] if j < y_poly_rev.shape[0] else 0.0

        # The distortion centre (the offsets of a polynomial) and the unit of the radius
        model.x0 = x0
        model.y0 = y0
        model.half_width = x_res/2.0
        model.kind = radialKind(dist_type)
        model.decentering = decentering_corr

        # The parameters of a radial model, with zeros for the radial coefficients its type does not have
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
        # A polynomial has none of them
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

            # The image of the corrected origin, where the rays of the fold check start: Newton's method from
            #   the inverse of the linear part (that seed itself if there is no solution)
            if lin_det > 0:
                poly_origin[0] = (lin_yy*(0.0 - x0) - lin_xy*(0.0 - y0))/lin_det
                poly_origin[1] = (lin_xx*(0.0 - y0) - lin_yx*(0.0 - x0))/lin_det
            else:
                poly_origin[0] = 0.0
                poly_origin[1] = 0.0
            img[0] = poly_origin[0]
            img[1] = poly_origin[1]
            invertDistortionNewton(&model, 0.0, 0.0, img)
            if img[0] == img[0]:
                poly_origin[0] = img[0]
                poly_origin[1] = img[1]

        # The rising branch of a radial model, for its inverse (which with decentering starts Newton's method)
        if model.poly_radial < 0:
            radial_root = not decentering_corr
            r_end = radialBranchEnd(model.kind, model.k1, model.k2, model.k3, model.k4, &f_end)

    # Earth velocity for the annual aberration of the catalog directions
    vx = vy = vz = 0.0
    if aberration:
        vx, vy, vz = earthVelocityJ2000(jd)

    # Convert all equatorial coordinates to image coordinates
    for i in range(ra_data.shape[0]):

        # With a date per point, the pointing, the refraction and the aberration are computed at the date of
        #   this point, when it differs from the date they were last computed at
        if (jd_data is not None) and (jd_data[i] != jd):
            jd = jd_data[i]
            date_frame = dateFrame(jd, radians(lon))
            ra_centre, dec_centre, pos_angle_ref = pointingCorrectionOfDate(date_frame, jd, radians(lat),
                radians(h0), radians(ra_ref), radians(dec_ref), radians(pos_angle_ref_in), refraction,
                refraction_scale)

            if aberration:
                vx, vy, vz = earthVelocityJ2000(jd)

        ra = radians(ra_data[i])
        dec = radians(dec_data[i])

        # Annual aberration: catalog direction -> apparent direction seen from the moving Earth
        if aberration:
            ra, dec = shiftDirection(ra, dec, vx, vy, vz)

        ### Gnomonization of star coordinates to image coordinates ###

        # Apply refraction
        if refraction:
            ra, dec = eqRefraction(date_frame, ra, dec, radians(lat), True, refraction_scale)


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

            # Radial models: the direction from the distortion centre is kept; only the radius is rescaled
            if radial_root:
                asym_factor = radialInverse(&model, x, y, r_end, f_end, img)

                # No root on the rising branch. Past the maximum of the radial function, with a positive
                #   asymmetry factor, every solution lies past the first fold of the mapping (on the side of
                #   the corrected point its radius is beyond the maximum, on the other side the radial
                #   function has turned back): no image. Otherwise (no maximum within the radius searched for
                #   it, which an asymmetry factor larger than 1 can carry the radius beyond, or an asymmetry
                #   larger than 1 that makes the factor negative: a fit that left the asymmetry free over
                #   negligible radial terms, whose mapping is still regular) the solution is searched by
                #   Newton's method from the corrected position, which checks the orientation of the mapping,
                #   and kept only within 1.5 times the frame and with no fold of the mapping between it and
                #   the distortion centre: past a fold it lies on a folded sheet, which is not the image of
                #   the sky even where its orientation is that of the image
                if (img[0] != img[0]) and ((asym_factor <= 0) or (r_end >= RADIAL_BRANCH_LIMIT)):
                    img[0] = x
                    img[1] = y
                    invertDistortionNewton(&model, x, y, img)
                    if (img[0] == img[0]) and ((fabs(img[0]) > 0.75*x_res) or (fabs(img[1]) > 0.75*y_res)
                        or (not beforeFirstFold(&model, model.x0, model.y0, img[0], img[1], FOLD_SAMPLES))):

                        img[0] = NAN_VALUE
                        img[1] = NAN_VALUE

            # Polynomial models and decentering: Newton's method in the image plane. A polynomial starts from
            #   the corrected position with its linear part undone; a radial model with decentering from the
            #   inverse of its radial part, which keeps the solution on the rising branch
            else:
                seed_beyond_radial = False
                if model.poly_radial < 0:
                    radialInverse(&model, x, y, r_end, f_end, img)

                    # Beyond the maximum of the radial part alone the decentering can still keep the whole
                    #   mapping rising (near the edge of the frame), so there (and with a negative asymmetry
                    #   factor) the solution is searched from the corrected position instead, and kept only
                    #   within 1.5 times the frame
                    if img[0] != img[0]:
                        img[0] = x
                        img[1] = y
                        seed_beyond_radial = True
                elif lin_det > 0:
                    img[0] = (lin_yy*(x - x0) - lin_xy*(y - y0))/lin_det
                    img[1] = (lin_xx*(y - y0) - lin_yx*(x - x0))/lin_det
                else:
                    img[0] = x
                    img[1] = y

                seed_x = img[0]
                seed_y = img[1]
                if img[0] == img[0]:
                    invertDistortionNewton(&model, x, y, img)

                # The decentering can also lower the maximum of the whole mapping below that of its radial part,
                #   so even from a seed on the rising branch Newton's method can end past the first fold, on a
                #   sheet that is not the image of the sky. A solution of a radial model is kept only with no
                #   fold between it and the distortion centre: checked along the whole ray when it was searched
                #   from beyond the radial maximum or moved far from its seed, otherwise at the solution alone
                #   (just past a fold near the seed the corrected point runs back toward the origin there)
                if (model.poly_radial < 0) and (img[0] == img[0]):
                    if seed_beyond_radial or (sqrt((img[0] - seed_x)**2 + (img[1] - seed_y)**2)
                        > FOLD_FREE_MOVE*0.5*x_res):

                        fold_samples = FOLD_SAMPLES
                    else:
                        fold_samples = 1

                    if (seed_beyond_radial and ((fabs(img[0]) > 0.75*x_res) or (fabs(img[1]) > 0.75*y_res))) \
                        or (not beforeFirstFold(&model, model.x0, model.y0, img[0], img[1], fold_samples)):

                        img[0] = NAN_VALUE
                        img[1] = NAN_VALUE

                # A polynomial with strong higher terms folds as well, and Newton's method can end on a folded
                #   sheet inside the frame (with the orientation of the image, which Newton's method checks at
                #   the solution). When it moved the solution far from the inverse of the linear part (relative
                #   to the distance of that seed from the origin), the solution is kept only if the orientation
                #   holds all along the segment from the image of the corrected origin; the radial distance of
                #   the radial models would take a shear for a fold
                elif (model.poly_radial >= 0) and (img[0] == img[0]):
                    moved_far = sqrt((img[0] - seed_x)**2 + (img[1] - seed_y)**2) > max(FOLD_FREE_MOVE*0.5*x_res,
                        POLY_FOLD_FREE_FRACTION*sqrt((seed_x - poly_origin[0])**2 + (seed_y - poly_origin[1])**2))
                    if moved_far and (not orientedAlongSegment(&model, poly_origin[0], poly_origin[1], img[0],
                        img[1], FOLD_SAMPLES)):

                        img[0] = NAN_VALUE
                        img[1] = NAN_VALUE

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



cpdef (double, double, double) pointingCorrection(
    double jd, double lat, double lon, 
    double h0, double jd_ref, double ra_ref, double dec_ref, double pos_angle_ref, 
    bool refraction=True, double refraction_scale=1.0
    ):
    """ Compute the pointing correction for the given Julian date. The camera is fixed in the horizontal
        frame, so the reference right ascension advances by the sidereal time elapsed since the reference
        time, at the same hour angle and declination: the result is the true-of-date pointing at jd. It is
        then refracted (if enabled) and rotated to J2000, together with the position angle, so that the
        field does not drift or rotate with time.

    Arguments:
        jd: [float] Julian date.
        lat: [float] Latitude of the observer in radians.
        lon: [float] Longitude of the observer in radians.
        h0: [float] Reference hour angle in radians: the apparent Greenwich sidereal time at jd_ref (mean
            sidereal time plus the equation of the equinoxes, see Conversions.JD2HourAngle).
        jd_ref: [float] Reference Julian date of the plate solution.
        ra_ref: [float] Reference right ascension of the image centre in radians: the true (unrefracted)
            direction in the true equator and equinox of jd_ref.
        dec_ref: [float] Reference declination of the image centre in radians, true equator and equinox
            of jd_ref.
        pos_angle_ref: [float] Rotation from the celestial meridian in radians, in the true equator and
            equinox of jd_ref.
    
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

    return pointingCorrectionOfDate(dateFrame(jd, lon), jd, lat, h0, ra_ref, dec_ref, pos_angle_ref, \
        refraction, refraction_scale)



cdef (double, double, double) pointingCorrectionOfDate(DateFrame d, double jd, double lat, double h0, \
    double ra_ref, double dec_ref, double pos_angle_ref, bint refraction, double refraction_scale):
    """ pointingCorrection with the frame rotations and sidereal times (dateFrame) given instead of the
        longitude, so that the kernels evaluate them once per Julian date for both the pointing and the
        refraction of the points.

    Arguments:
        d: [DateFrame] From dateFrame() for the time and the observer's longitude.
        jd: [float] Julian date.
        lat: [float] Latitude of the observer in radians.
        h0: [float] Reference hour angle in radians (see pointingCorrection).
        ra_ref: [float] Reference right ascension of the image centre in radians (see pointingCorrection).
        dec_ref: [float] Reference declination of the image centre in radians (see pointingCorrection).
        pos_angle_ref: [float] Rotation from the celestial meridian in radians.
        refraction: [bool] Apply refraction correction.
        refraction_scale: [float] Scale of the refraction for the observer's height above sea level, from
            refractionScale().

    Return:
        (ra_ref_now_corr, dec_ref_corr, pos_angle_ref_now_corr): [tuple of floats] See pointingCorrection.

    """

    cdef double ra_ref_now, ra_ref_now_corr, dec_ref_corr, azim, alt

    # Compute the reference RA centre at the given JD by adding the hour angle difference
    ra_ref_now = (ra_ref + d.gst - h0 + 2*pi)%(2*pi)

    # Correct the FOV centre for refraction. The reference is a true-of-date direction, so it is taken to
    #   alt/az directly: eqRefractionTrueToApparent expects J2000 input and would precess it a second
    #   time, evaluating the refraction about 22 arcmin (in 2026) away from the real altitude.
    if refraction:
        azim, alt = raDec2AltAzLST(ra_ref_now, dec_ref, d.lst, lat)
        alt = refractionTrueToApparent(alt, refraction_scale)
        ra_ref_now_corr, dec_ref_corr = altAz2RADecLST(azim, alt, d.lst, lat)

    else:
        ra_ref_now_corr = ra_ref_now
        dec_ref_corr = dec_ref


    # Precess the reference RA, dec, position angle to J2000 (needs to be used to avoid FOV centre drift
    #   over time). The position angle needs to be corrected for precession, otherwise the FOV rotates with
    #   time. No rotation at J2000 itself, as in equatorialCoordAndRotPrecession
    if jd == J2000_DAYS:
        return ra_ref_now_corr, dec_ref_corr, pos_angle_ref

    return frameRotateRaDecRot(d.to_j2000, ra_ref_now_corr, dec_ref_corr, pos_angle_ref)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cyXYToRADec(np.ndarray[FLOAT_TYPE_t, ndim=1] jd_data, np.ndarray[FLOAT_TYPE_t, ndim=1] x_data, \
    np.ndarray[FLOAT_TYPE_t, ndim=1] y_data, double lat, double lon, double x_res, double y_res, \
    double h0, double jd_ref, double ra_ref, double dec_ref, double pos_angle_ref, double pix_scale, \
    np.ndarray[FLOAT_TYPE_t, ndim=1] x_poly_fwd, np.ndarray[FLOAT_TYPE_t, ndim=1] y_poly_fwd, \
    str dist_type, bool refraction=True, bool equal_aspect=False, bool force_distortion_centre=False,\
    bool asymmetry_corr=True, bool precompute_pointing_corr=False, bool decentering_corr=False,
    double refraction_scale=1.0, bool aberration=True):
    """
    Arguments:
        jd_data: [ndarray] Julian date of each data point.
        x_data: [ndarray] 1D numpy array containing the image column.
        y_data: [ndarray] 1D numpy array containing the image row.
        lat: [float] Latitude of the observer in degrees.
        lon: [float] Longitude of the observer in degrees.
        x_res: [int] Image size, X dimension (px).
        y_res: [int] Image size, Y dimension (px).
        h0: [float] Reference hour angle (deg): the apparent Greenwich sidereal time at jd_ref (mean
            sidereal time plus the equation of the equinoxes, see Conversions.JD2HourAngle).
        jd_ref: [float] Reference Julian date when the plate was fit.
        ra_ref: [float] Reference right ascension of the image centre (degrees), true equator and equinox
            of jd_ref.
        dec_ref: [float] Reference declination of the image centre (degrees), true equator and equinox of
            jd_ref.
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
    cdef double vx, vy, vz
    cdef DateFrame date_frame

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

        # radial3-odd has no k2: its polynomial ends at k1, and reading past it is out of bounds
        k2 = 0.0
        if x_poly_fwd.shape[0] > (6 - index_offset):
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


        # What depends only on the time is evaluated when it changes, i.e. once per frame
        if (i == 0) or (jd != jd_data[i - 1]):

            # Frame rotations and sidereal time for the pointing and the refraction
            if refraction or (not precompute_pointing_corr):
                date_frame = dateFrame(jd, radians(lon))

            if not precompute_pointing_corr:

                # Correct the pointing for precession (output in radians)
                ra_ref_now_corr, dec_ref_corr, pos_angle_ref_now_corr = pointingCorrectionOfDate(date_frame,
                    jd, radians(lat), radians(h0), radians(ra_ref), radians(dec_ref), radians(pos_angle_ref),
                    refraction, refraction_scale)

            # Earth velocity for the annual aberration
            if aberration:
                vx, vy, vz = earthVelocityJ2000(jd)


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
            ra, dec = eqRefraction(date_frame, ra, dec, radians(lat), False, refraction_scale)

        # Annual aberration: apparent direction seen from the moving Earth -> catalog direction
        if aberration:
            ra, dec = shiftDirection(ra, dec, -vx, -vy, -vz)



        # Convert coordinates to degrees
        ra = degrees(ra)
        dec = degrees(dec)


        # Assign values to output list
        ra_data[i] = ra
        dec_data[i] = dec


    return ra_data, dec_data