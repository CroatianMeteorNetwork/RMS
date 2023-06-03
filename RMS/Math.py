""" Common math functions. """

import numpy as np
from numpy.core.umath_tests import inner1d


from datetime import datetime, timedelta, MINYEAR
import math

J2000_JD = timedelta(2451545)
JULIAN_EPOCH = datetime(2000, 1, 1, 12) # J2000.0 noon

def lineFunc(x, m, k):
    """ Linear function.
    
    Arguments:
        x: [float or ndarray] Independant variable.
        m: [float] Slope.
        k: [float] Y-intercept.

    Return:
        y: [float or ndarray] Dependant variable.
    """

    return m*x + k


def logLineFunc(x, m, k):
    """ Logarithmic linear function.

    Arguments:
        x: [float or ndarray] Independant variable.
        m: [float] Slope.
        k: [float] Y-intercept.

    Return:
        y: [float or ndarray] Dependant variable.
    """

    return 10**lineFunc(x, m, k)


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

    # Classical method
    return np.arccos(np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra2 - ra1))

    # # Compute the angular separation using the haversine formula
    # #   Source: https://idlastro.gsfc.nasa.gov/ftp/pro/astro/gcirc.pro
    # deldec2 = (dec2 - dec1)/2.0
    # delra2 =  (ra2 - ra1)/2.0
    # sindis = np.sqrt(np.sin(deldec2)*np.sin(deldec2) \
    #     + np.cos(dec1)*np.cos(dec2)*np.sin(delra2)*np.sin(delra2))
    # dis = 2.0*np.arcsin(sindis) 

    # return dis


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


def angleBetweenVectors(a, b):
    """ Compute the angle between two vectors.

    Arguments:
        a: [ndarray] First vector.
        b: [ndarray] Second vector.

    Return:
        [float] Angle between a and b (radians).
    """

    return np.arccos(np.dot(a, b) / (vectMag(a) * vectMag(b)))


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
        theta: [float] Longitude in radians.
        phi: [float] Latitude in radians.

    Return:
        (x, y, z): [tuple of floats] Coordinates of the point in 3D cartiesian coordinates.
    """


    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)

    return x, y, z


def isAngleBetween(left, ang, right):
    """ Checks if ang is between the angle on the left and right. 
    
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

    # Compute the new RA
    dra = np.arctan2(np.sin(heading)*np.sin(distance)*np.cos(dec1), np.cos(distance) \
        - np.sin(dec1)*np.sin(dec))
    ra = (ra1 - dra + np.pi)%(2*np.pi) - np.pi


    return np.degrees(ra)%360, np.degrees(dec)
    


def RMSD(x, weights=None):
    """ Root-mean-square deviation of measurements vs. model. 
    
    Arguments:
        x: [ndarray] An array of model and measurement differences.

    Return:
        [float] RMSD
    """

    if isinstance(x, list):
        x = np.array(x)

    if weights is None:
        weights = np.ones_like(x)

    return np.sqrt(np.sum(weights*x**2)/np.sum(weights))


### 3D functions ###
##############################################################################################################

def sphericalToCartesian(r, theta, phi):
    """ Convert spherical coordinates to cartesian coordinates. 
        
    Arguments:
        r: [float] Radius
        theta: [float] Inclination in radians.
        phi: [float] Azimuth angle in radians.

    Return:
        (x, y, z): [tuple of floats] Coordinates of the point in 3D cartiesian coordinates.
    """

    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    return x, y, z


def pointInsideConvexPolygonSphere(points, vertices):
    """
    Polygon must be convex
    https://math.stackexchange.com/questions/4012834/checking-that-a-point-is-in-a-spherical-polygon


    Arguments:
        points: [array] points with dimension (npoints, 2). The two dimensions are ra and dec
        vertices: [array] vertices of convex polygon with dimension (nvertices, 2)
        
    Return:
        filter: [array of bool] Array of booleans on whether a given point is inside the polygon on 
            the sphere.
    """
    # convert ra dec to spherical
    points = points[:, ::-1]
    vertices = vertices[:, ::-1]
    points[:, 0] = 90 - points[:, 0]
    vertices[:, 0] = 90 - vertices[:, 0]
    points = np.array(sphericalToCartesian(*np.hstack((np.ones((len(points), 1)), np.radians(points))).T))
    vertices = np.array(sphericalToCartesian(*np.hstack((np.ones((len(vertices), 1)), np.radians(vertices))).T))
    
    great_circle_normal = np.cross(vertices, np.roll(vertices, 1, axis=1), axis=0)
    dot_prod = np.dot(great_circle_normal.T, points)
    return np.sum(dot_prod < 0, axis=0, dtype=int) == 0  # inside if n . p < 0 for no n


##############################################################################################################

def histogramEdgesEqualDataNumber(x, nbins):
    """ Given the data, divide the histogram edges in such a way that every bin has the same number of
        data points. 

        Source: https://stackoverflow.com/questions/37649342/matplotlib-how-to-make-a-histogram-with-bins-of-equal-area/37667480

    Arguments:
        x: [list] Input data.
        nbins: [int] Number of bins.

    """

    npt = len(x)
    return np.interp(np.linspace(0, npt, nbins + 1), np.arange(npt), np.sort(x))


def histogramEdgesDataNumber(x, points_per_bin):
    """ Given the data, divides the histogram edges in such a way that every bin contains at least a
    minimum number of points
    
    Arguments:
        x: [list] Input data.
        points_per_bin: [int] Number of point per bin
    """
    
    nbins = len(x)//points_per_bin
    return histogramEdgesEqualDataNumber(x, nbins)

#########

def rollingAverage2d(x, y, x_window):
    """
    Rolling average where the window is on the x axis rather than index
    
    Arguments:
        x: [list or ndarray] sorted x values
        y: [list or ndarray] y values corresponding to x
        x_window: [float]
        
    Returns:
        output_x: [list]
        output_y: [list]
    """
    assert len(x) == len(y)
    cum_y = np.cumsum(np.insert(y, 0, 0))
    output_y = []
    output_x = []
    j = 0
    for i in range(len(y)):
        for j in range(j, len(y)):
            if x_window >= x[j] - x[i]:
                output_y.append((cum_y[j + 1] - cum_y[i])/(j + 1 - i))
                output_x.append((x[j] + x[i])/2)
            if j > i and (j == len(x) - 1 or (x[j + 1] - x[i] > x_window and x[j] - x[i + 1] < x[j + 1] - x[i])):
                break

    return output_x, output_y

def raDec2AltAz(ra, dec, jd, lat, lon):
    """ Convert right ascension and declination to azimuth (+east of sue north) and altitude.

    Arguments:
        ra: [float] right ascension in radians
        dec: [float] declination in radians
        jd: [float] Julian date
        lat: [float] latitude in radians
        lon: [float] longitude in radians

    Return:
        (azim, elev): [tuple]
            azim: [float] azimuth (+east of due north) in radians
            elev: [float] elevation above horizon in radians

        """

    # Calculate Local Sidereal Time
    lst = np.radians(jd2LST(jd, np.degrees(lon))[0])

    # Calculate the hour angle
    ha = lst - ra

    # Constrain the hour angle to [-pi, pi] range
    ha = (ha + np.pi)%(2*np.pi) - np.pi

    # Calculate the azimuth
    azim = np.pi + np.arctan2(np.sin(ha), np.cos(ha)*np.sin(lat) - np.tan(dec)*np.cos(lat))

    # Calculate the sine of elevation
    sin_elev = np.sin(lat)*np.sin(dec) + np.cos(lat)*np.cos(dec)*np.cos(ha)

    # Wrap the sine of elevation in the [-1, +1] range
    sin_elev = (sin_elev + 1)%2 - 1

    elev = np.arcsin(sin_elev)

    return azim, elev


def jd2LST(julian_date, lon):
    """ Convert Julian date to Local Sidereal Time and Greenwich Sidereal Time. The times used are apparent
        times, not mean times.

    Source: J. Meeus: Astronomical Algorithms

    Arguments:
        julian_date: [float] decimal julian date, epoch J2000.0
        lon: [float] longitude of the observer in degrees

    Return:
        (LST, GST): [tuple of floats] a tuple of Local Sidereal Time and Greenwich Sidereal Time
    """

    # t = (julian_date - J2000_JD.days)/36525.0

    # Greenwich Sidereal Time
    # GST = 280.46061837 + 360.98564736629*(julian_date - J2000_JD.days) + 0.000387933*t**2 - (t**3)/38710000
    # GST = (GST + 360)%360

    GST = np.degrees(calcApparentSiderealEarthRotation(julian_date))

    # Local Sidereal Time
    LST = (GST + lon + 360) % 360

    return LST, GST


def calcApparentSiderealEarthRotation(julian_date):
    """ Calculate apparent sidereal rotation GST of the Earth.

        Calculated according to:
        Clark, D. L. (2010). Searching for fireball pre-detections in sky surveys. The School of Graduate and
        Postdoctoral Studies. University of Western Ontario, London, Ontario, Canada, MSc Thesis.

    """

    t = (julian_date - J2000_JD.days) / 36525.0

    # Calculate the Mean sidereal rotation of the Earth in radians (Greenwich Sidereal Time)
    GST = 280.46061837 + 360.98564736629 * (
                julian_date - J2000_JD.days) + 0.000387933 * t ** 2 - (t ** 3) / 38710000
    GST = (GST + 360) % 360
    GST = math.radians(GST)

    # print('GST:', np.degrees(GST), 'deg')

    # Calculate the dynamical time JD
    jd_dyn = jd2DynamicalTimeJD(julian_date)

    # Calculate Earth's nutation components
    delta_psi, delta_eps = calcNutationComponents(jd_dyn)

    # print('Delta Psi:', np.degrees(delta_psi), 'deg')
    # print('Delta Epsilon:', np.degrees(delta_eps), 'deg')

    # Calculate the mean obliquity (in arcsec)
    u = (jd_dyn - 2451545.0) / 3652500.0
    eps0 = 84381.448 - 4680.93 * u - 1.55 * u ** 2 + 1999.25 * u ** 3 - 51.38 * u ** 4 - 249.67 * u ** 5 - 39.05 * u ** 6 \
           + 7.12 * u ** 7 + 27.87 * u ** 8 + 5.79 * u ** 9 + 2.45 * u ** 10

    # Convert to radians
    eps0 /= 3600
    eps0 = np.radians(eps0)

    # print('Mean obliquity:', np.degrees(eps0), 'deg')

    # Calculate apparent sidereal Earth's rotation
    app_sid_rot = (GST + delta_psi * math.cos(eps0 + delta_eps)) % (2 * math.pi)

    return app_sid_rot


def calcNutationComponents(jd_dyn):
    """ Calculate Earth's nutation components from the given Julian date.

    Source: Meeus (1998) Astronomical algorithms, 2nd edition, chapter 22.

    The precision is limited to 0.5" in nutation in longitude and 0.1" in nutation in obliquity. The errata
    for the 2nd edition was used to correct the equation for delta_psi.

    Arguments:
        jd_dyn: [float] Dynamical Julian date. See wmpl.Utils.TrajConversions.jd2DynamicalTimeJD function.

    Return:
        (delta_psi, delta_eps): [tuple of floats] Differences from mean nutation due to the influence of
            the Moon and minor effects (radians).
    """

    T = (jd_dyn - J2000_JD.days) / 36525.0

    # # Mean Elongation of the Moon from the Sun
    # D = 297.85036 + 445267.11148*T - 0.0019142*T**2 + (T**3)/189474

    # # Mean anomaly of the Earth with respect to the Sun
    # M = 357.52772 + 35999.05034*T - 0.0001603*T**2 - (T**3)/300000

    # # Mean anomaly of the Moon
    # Mm = 134.96298 + 477198.867398*T + 0.0086972*T**2 + (T**3)/56250

    # # Argument of latitude of the Moon
    # F = 93.27191  + 483202.017538*T - 0.0036825*T**2 + (T**3)/327270

    # Longitude of the ascending node of the Moon's mean orbit on the ecliptic, measured from the mean equinox
    # of the date
    omega = 125.04452 - 1934.136261 * T

    # Mean longitude of the Sun (deg)
    L = 280.4665 + 36000.7698 * T

    # Mean longitude of the Moon (deg)
    Ll = 218.3165 + 481267.8813 * T

    # Nutation in longitude
    delta_psi = -17.2 * math.sin(math.radians(omega)) - 1.32 * math.sin(np.radians(2 * L)) \
                - 0.23 * math.sin(math.radians(2 * Ll)) + 0.21 * math.sin(math.radians(2 * omega))

    # Nutation in obliquity
    delta_eps = 9.2 * math.cos(math.radians(omega)) + 0.57 * math.cos(math.radians(2 * L)) \
                + 0.1 * math.cos(math.radians(2 * Ll)) - 0.09 * math.cos(math.radians(2 * omega))

    # Convert to radians
    delta_psi = np.radians(delta_psi / 3600)
    delta_eps = np.radians(delta_eps / 3600)

    return delta_psi, delta_eps


def jd2DynamicalTimeJD(jd):
    """ Converts the given Julian date to dynamical time (i.e. Terrestrial Time, TT) Julian date. The
        conversion takes care of leap seconds.

    Arguments:
        jd: [float] Julian date.

    Return:
        [float] Dynamical time Julian date.
    """

    # Leap seconds as of 2017 (default)
    leap_secs = 37.0 #that's enough



    # Calculate the dynamical JD
    jd_dyn = jd + (leap_secs + 32.184) / 86400.0

    return jd_dyn



def datetime2JD(dt):
    """ Converts a datetime object to Julian date.

    Arguments:
        dt: [datetime object]

    Return:
        jd: [float] Julian date
    """

    return date2JD(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond/1000.0)


def date2JD(year, month, day, hour, minute, second, millisecond=0, UT_corr=0.0):
    """ Convert date and time to Julian Date in J2000.0.

    Arguments:
        year: [int] year
        month: [int] month
        day: [int] day of the date
        hour: [int] hours
        minute: [int] minutes
        second: [int] seconds

    Kwargs:
        millisecond: [int] milliseconds (optional)
        UT_corr: [float] UT correction in hours (difference from local time to UT)

    Return:
        [float] julian date, J2000.0 epoch
    """

    # Convert all input arguments to integer (except milliseconds)
    year, month, day, hour, minute, second = map(int, (year, month, day, hour, minute, second))

    # Create datetime object of current time
    dt = datetime(year, month, day, hour, minute, second, int(millisecond * 1000))

    # Calculate Julian date
    julian = dt - JULIAN_EPOCH + J2000_JD - timedelta(hours=UT_corr)

    # Convert seconds to day fractions
    return julian.days + (julian.seconds + julian.microseconds / 1000000.0) / 86400.0


def altAz2RADec(azim, elev, jd, lat, lon):
    """ Convert azimuth and altitude in a given time and position on Earth to right ascension and
        declination.

    Arguments:
        azim: [float] azimuth (+east of due north) in radians
        elev: [float] elevation above horizon in radians
        jd: [float] Julian date
        lat: [float] latitude of the observer in radians
        lon: [float] longitde of the observer in radians

    Return:
        (RA, dec): [tuple]
            RA: [float] right ascension (radians)
            dec: [float] declination (radians)
    """

    # Calculate hour angle
    ha = np.arctan2(-np.sin(azim), np.tan(elev) * np.cos(lat) - np.cos(azim) * np.sin(lat))

    # Calculate Local Sidereal Time
    lst = np.radians(jd2LST(jd, np.degrees(lon))[0])

    # Calculate right ascension
    ra = (lst - ha) % (2 * np.pi)

    # Calculate declination
    dec = np.arcsin(np.sin(lat) * np.sin(elev) + np.cos(lat) * np.cos(elev) * np.cos(azim))



def raDec2ECI(ra, dec):
    """ Convert right ascension and declination to Earth-centered inertial vector.

    Arguments:
        ra: [float] right ascension in radians
        dec: [float] declination in radians

    Return:
        (x, y, z): [tuple of floats] Earth-centered inertial coordinates

    """

    x = np.cos(dec)*np.cos(ra)
    y = np.cos(dec)*np.sin(ra)
    z = np.sin(dec)

    return x, y, z

def latLonAlt2ECEF(lat, lon, h):
    """ Convert geographical coordinates to Earth centered - Earth fixed coordinates.

    Arguments:
        lat: [float] latitude in radians (+north)
        lon: [float] longitude in radians (+east)
        h: [float] elevation in meters (WGS84)

    Return:
        (x, y, z): [tuple of floats] ECEF coordinates

    """

    # Get distance from Earth centre to the position given by geographical coordinates, in WGS84
    N = EARTH.EQUATORIAL_RADIUS/math.sqrt(1.0 - (EARTH.E**2)*math.sin(lat)**2)

    # Calculate ECEF coordinates
    ecef_x = (N + h)*math.cos(lat)*math.cos(lon)
    ecef_y = (N + h)*math.cos(lat)*math.sin(lon)
    ecef_z = ((1 - EARTH.E**2)*N + h)*math.sin(lat)

    return ecef_x, ecef_y, ecef_z

class EARTH_CONSTANTS(object):
    """ Holds Earth's shape and physical parameters. """

    def __init__(self):

        # Earth elipsoid parameters in meters (source: WGS84, the GPS standard)
        self.EQUATORIAL_RADIUS = 6378137.0
        self.POLAR_RADIUS = 6356752.314245
        self.E = math.sqrt(1.0 - self.POLAR_RADIUS**2/self.EQUATORIAL_RADIUS**2)
        self.RATIO = self.EQUATORIAL_RADIUS/self.POLAR_RADIUS
        self.SQR_DIFF = self.EQUATORIAL_RADIUS**2 - self.POLAR_RADIUS**2

        # Earth mass (kg)
        self.MASS = 5.9722e24

# Initialize Earth shape constants object
EARTH = EARTH_CONSTANTS()