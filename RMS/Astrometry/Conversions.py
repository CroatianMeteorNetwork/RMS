""" 
A set of tools of working with meteor data. 
Includes:
    - Julian date conversion
    - LST calculation
    - Coordinate transformations
    - RA and Dec precession correction
    - ...

"""

# The MIT License

# Copyright (c) 2016 Denis Vida

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import print_function, division, absolute_import, unicode_literals

import math
from datetime import datetime, timedelta, MINYEAR

import numpy as np
import scipy.optimize

from RMS.Math import vectMag, vectNorm
from RMS.Misc import UTCFromTimestamp

# Import Cython functions
import pyximport
pyximport.install(setup_args={'include_dirs': [np.get_include()]})
from RMS.Astrometry.CyFunctions import cyaltAz2RADec, cyraDec2AltAz, cyApparentAltAz2TrueRADec, \
    cyApparentAltAz2TrueRADec_vect, cyTrueRaDec2ApparentAltAz, cyTrueRaDec2ApparentAltAz_vect

# Vectorize some functions
cyaltAz2RADec_vect = np.vectorize(cyaltAz2RADec, excluded=["jd", "lat", "lon"])
cyraDec2AltAz_vect = np.vectorize(cyraDec2AltAz, excluded=["jd", "lat", "lon"])

### CONSTANTS ###

# Define Julian epoch
JULIAN_EPOCH = datetime(2000, 1, 1, 12)  # noon (the epoch name is unrelated)
J2000_JD = timedelta(2451545)  # julian epoch in julian dates


class EARTH_CONSTANTS(object):
    """ Holds Earth's shape and physical parameters. """

    def __init__(self):

        # Earth ellipsoid parameters in meters (source: WGS84, the GPS standard)
        self.EQUATORIAL_RADIUS = 6378137.0
        self.POLAR_RADIUS = 6356752.314245
        self.E = math.sqrt(1.0 - self.POLAR_RADIUS**2/self.EQUATORIAL_RADIUS**2)
        self.RATIO = self.EQUATORIAL_RADIUS/self.POLAR_RADIUS
        self.SQR_DIFF = self.EQUATORIAL_RADIUS**2 - self.POLAR_RADIUS**2


# Initialize Earth shape constants object
EARTH = EARTH_CONSTANTS()


#################


### DECORATORS ###

def floatArguments(func):
    """ A decorator that converts all function arguments to float.

    @param func: a function to be decorated
    @return :[function object] the decorated function
    """

    def inner_func(*args):
        args = map(float, args)
        return func(*args)

    return inner_func


##################


### Time conversions ###


def unixTime2Date(ts, tu, dt_obj=False):
    """ Convert UNIX time given in ts and tu to date and time.

    Arguments:
        ts: [int] UNIX time, seconds part
        tu: [int] UNIX time, microsecond part
    Kwargs:
        dt_obj: [bool] default False, function returns a datetime object if True
    Return:
        if dt_obj == False (default): [tuple] (year, month, day, hours, minutes, seconds, milliseconds)
        else: [datetime object]
    """

    # Convert the UNIX timestamp to datetime object
    dt = UTCFromTimestamp.utcfromtimestamp(float(ts) + float(tu)/1000000)

    if dt_obj:
        return dt

    else:
        return dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, float(tu)/1000


def datetime2UnixTime(dt):
    """ Convert the given datetime to UNIX time.

    Arguments:
        dt: [datetime]
    Return:
        [float] Unix time.
    """

    # UTC unix timestamp
    unix_timestamp = (dt - datetime(1970, 1, 1)).total_seconds()

    return unix_timestamp


def date2UnixTime(year, month, day, hour, minute, second, millisecond=0, UT_corr=0.0):
    """ Convert date and time to Unix time. 
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
        [float] Unix time
    """  # Convert all input arguments to integer (except milliseconds)
    year, month, day, hour, minute, second = map(int, (year, month, day, hour, minute, second))

    # Create datetime object of current time
    dt = datetime(year, month, day, hour, minute, second, int(millisecond*1000)) - timedelta(hours=UT_corr)

    return datetime2UnixTime(dt)


def date2JD(year, month, day, hour, minute, second, millisecond=0, UT_corr=0.0):
    """ Convert date and time to Julian Date with epoch J2000.0.
    @param year: [int] year
    @param month: [int] month
    @param day: [int] day of the date
    @param hour: [int] hours
    @param minute: [int] minutes
    @param second: [int] seconds
    @param millisecond: [int] milliseconds (optional)
    @param UT_corr: [float] UT correction in hours (difference from local time to UT)
    @return :[float] julian date, epoch 2000.0
    """

    # Convert all input arguments to integer (except milliseconds)
    year, month, day, hour, minute, second = map(int, (year, month, day, hour, minute, second))

    # Create datetime object of current time
    dt = datetime(year, month, day, hour, minute, second, int(millisecond*1000))

    # Calculate Julian date
    julian = dt - JULIAN_EPOCH + J2000_JD - timedelta(hours=UT_corr)

    # Convert seconds to day fractions
    return julian.days + (julian.seconds + julian.microseconds/1000000.0)/86400.0


def datetime2JD(dt, UT_corr=0.0):
    """ Converts a datetime object to Julian date.
    Arguments:
        dt: [datetime object]
    Keyword arguments:
        UT_corr: [float] UT correction in hours (difference from local time to UT)
    Return:
        jd: [float] Julian date
    """

    return date2JD(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond/1000.0,
                   UT_corr=UT_corr)


def jd2Date(jd, UT_corr=0, dt_obj=False):
    """ Converts the given Julian date to (year, month, day, hour, minute, second, millisecond) tuple.
    Arguments:
        jd: [float] Julian date
    Keyword arguments:
        UT_corr: [float] UT correction in hours (difference from local time to UT)
        dt_obj: [bool] returns a datetime object if True. False by default.
    Return:
        (year, month, day, hour, minute, second, millisecond)
    """

    dt = timedelta(days=jd)

    try:
        date = dt + JULIAN_EPOCH - J2000_JD + timedelta(hours=UT_corr)

    # If the date is out of range (i.e. before year 1) use year 1. This is the limitation in the datetime
    # library. Time handling should be switched to astropy.time
    except OverflowError:
        date = datetime(MINYEAR, 1, 1, 0, 0, 0)

    # Return a datetime object if dt_obj == True
    if dt_obj:
        return date

    return date.year, date.month, date.day, date.hour, date.minute, date.second, date.microsecond/1000.0


def unixTime2JD(ts, tu):
    """ Converts UNIX time to Julian date.

    Arguments:
        ts: [int] UNIX time, seconds part
        tu: [int] UNIX time, microsecond part
    Return:
        [float] julian date, epoch 2000.0
    """

    return date2JD(*unixTime2Date(ts, tu))


def jd2UnixTime(jd, UT_corr=0):
    """ Converts the given Julian date to Unix timestamp.
    Arguments:
        jd: [float] Julian date
    Keyword arguments:
        UT_corr: [float] UT correction in hours (difference from local time to UT)
    Return:
        [float] Unix timestamp.
    """

    return date2UnixTime(*jd2Date(jd, UT_corr=UT_corr))


def JD2LST(julian_date, lon):
    """ Convert Julian date to Local Sidereal Time and Greenwich Sidereal Time.

    Arguments;
        julian_date: [float] decimal julian date, epoch J2000.0
        lon: [float] longitude of the observer in degrees

    Return:
        [tuple]: (LST, GST): [tuple of floats] a tuple of Local Sidereal Time and Greenwich Sidereal Time
            (degrees)
    """

    t = (julian_date - J2000_JD.days)/36525.0

    # Greenwich Sidereal Time
    GST = 280.46061837 + 360.98564736629*(julian_date - 2451545) + 0.000387933*t**2 - ((t**3)/38710000)
    GST = (GST + 360)%360

    # Local Sidereal Time
    LST = (GST + lon + 360)%360

    return LST, GST


def JD2HourAngle(jd):
    """ Convert the given Julian date to hour angle.
    Arguments:
        jd: [float] Julian date.
    Return:
        hour_angle: [float] Hour angle (deg).
    """

    T = (jd - 2451545)/36525.0
    hour_angle = 280.46061837 + 360.98564736629*(jd - 2451545.0) + 0.000387933*T**2 \
                 - (T**3)/38710000.0

    return hour_angle


############################


### Spatial coordinates transformations ###


def LST2LongitudeEast(julian_date, LST):
    """ Convert Julian date and Local Sidereal Time to east longitude. 
    
    Arguments:
        julian_date: [float] decimal julian date, epoch J2000.0
        LST: [float] Local Sidereal Time in degrees

    Return:
        lon: [float] longitude of the observer in degrees
    """

    # Greenwich Sidereal Time (apparent)
    _, GST = JD2LST(julian_date, 0)

    # Calculate longitude
    lon = (LST - GST + 180)%360 - 180

    return lon, GST



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


@floatArguments
def geo2Cartesian(lat, lon, h, julian_date):
    """ Convert geographical Earth coordinates to Cartesian ECI coordinate system (Earth center as origin).
        The Earth is considered as an ellipsoid.
    
    Arguments:
        lat_rad: [float] Latitude of the observer in degrees (+N), WGS84.
        lon_rad: [float] Longitude of the observer in degrees (+E), WGS84.
        h: [int or float] Elevation of the observer in meters (WGS84 convention).
        julian_date: [float] Julian date, epoch J2000.0.
    
    Return:
        (x, y, z): [tuple of floats] A tuple of X, Y, Z Cartesian ECI coordinates in meters.
        
    """

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Calculate ECEF coordinates
    ecef_x, ecef_y, ecef_z = latLonAlt2ECEF(lat_rad, lon_rad, h)


    # Get Local Sidereal Time
    LST_rad = math.radians(JD2LST(julian_date, np.degrees(lon_rad))[0])

    # Calculate the Earth radius at given latitude
    Rh = math.sqrt(ecef_x**2 + ecef_y**2 + ecef_z**2)

    # Calculate the geocentric latitude (latitude which considers the Earth as an ellipsoid)
    lat_geocentric = math.atan2(ecef_z, math.sqrt(ecef_x**2 + ecef_y**2))

    # Calculate Cartesian ECI coordinates (in meters), in the epoch of date
    x = Rh*np.cos(lat_geocentric)*np.cos(LST_rad)
    y = Rh*np.cos(lat_geocentric)*np.sin(LST_rad)
    z = Rh*np.sin(lat_geocentric)

    return x, y, z



def ecef2LatLonAlt(x, y, z):
    """ Convert Earth centered - Earth fixed coordinates to geographical coordinates (latitude, longitude, 
        elevation).

    Arguments:
        x: [float] ECEF x coordinate
        y: [float] ECEF y coordinate
        z: [float] ECEF z coordinate

    Return:
        (lat, lon, alt): [tuple of floats] latitude and longitude in radians, WGS84 elevation in meters

    """

    # Calculate the polar eccentricity
    ep = np.sqrt((EARTH.EQUATORIAL_RADIUS**2 - EARTH.POLAR_RADIUS**2)/(EARTH.POLAR_RADIUS**2))

    # Calculate the longitude
    lon = np.arctan2(y, x)

    p = np.sqrt(x**2  +  y**2)

    theta = np.arctan2( z*EARTH.EQUATORIAL_RADIUS, p*EARTH.POLAR_RADIUS)

    # Calculate the latitude
    lat = np.arctan2(z + (ep**2)*EARTH.POLAR_RADIUS*np.sin(theta)**3, \
        p - (EARTH.E**2)*EARTH.EQUATORIAL_RADIUS*np.cos(theta)**3)

    # Get distance from Earth centre to the position given by geographical coordinates, in WGS84
    N = EARTH.EQUATORIAL_RADIUS/math.sqrt(1.0 - (EARTH.E**2)*math.sin(lat)**2)

    
    # Calculate the height in meters

    # Correct for numerical instability in altitude near exact poles (and make sure cos(lat) is not 0!)
    if((np.abs(x) < 1000) and (np.abs(y) < 1000)):
        alt = np.abs(z) - EARTH.POLAR_RADIUS

    else:
        # Calculate altitude anywhere else
        alt = p/np.cos(lat) - N


    return lat, lon, alt



def ECEF2AltAz(s_vect, p_vect):
    """ Given two sets of ECEF coordinates, compute alt/az which point from the point S to the point P.

    Source: https://gis.stackexchange.com/a/58926
    
    Arguments:
        s_vect: [ndarray] sx, sy, sz - S point ECEF coordinates
        p_vect: [ndarray] px, py, pz - P point ECEF coordinates

    Return:
        (azim, alt): Horizontal coordinates in degrees.

    """


    sx, sy, sz = s_vect
    px, py, pz = p_vect

    # Compute the pointing vector from S to P
    dx = px - sx
    dy = py - sy
    dz = pz - sz

    # Compute the elevation
    alt = np.degrees(
        np.pi/2 - np.arccos((sx*dx + sy*dy + sz*dz)/np.sqrt((sx**2 + sy**2 + sz**2)*(dx**2 + dy**2 + dz**2)))
        )

    # Compute the azimuth
    
    cos_az = (-sz*sx*dx - sz*sy*dy + (sx**2 + sy**2)*dz)/np.sqrt(
                                            (sx**2 + sy**2)*(sx**2 + sy**2 + sz**2)*(dx**2 + dy**2 + dz**2)
                                            )
    
    sin_az = (-sy*dx + sx*dy)/np.sqrt((sx**2 + sy**2)*(dx**2 + dy**2 + dz**2))

    azim = np.degrees(np.arctan2(sin_az, cos_az))%360


    return azim, alt



def AER2ECEF(azim, elev, r, lat, lon, alt):
    """ Given an azimuth, altitude, and range, compute the ECEF coordinate of that point given a location
        of the observer by lat, lon, alt.

        Source: https://stackoverflow.com/questions/15954978/ecef-from-azimuth-elevation-range-and-observer-lat-lon-alt

    Arguments:
        azim: [float] Azimuth (+E of due N) in degrees.
        elev: [float] Elevation in degrees.
        r: [float] Range in meters.
        lat: [float] Latitude of observer in degrees.
        lon: [float] Longitude of observer in degrees.
        alt: [float] Altitude of observer in meters.

    Return:
        (x, y, z): [list of floats] ECEF coordinates of the given point.

    """

    # Observer ECEF coordinates
    obs_x, obs_y, obs_z = latLonAlt2ECEF(np.radians(lat), np.radians(lon), alt)

    # Precalculate some values
    slat = np.sin(np.radians(lat))
    slon = np.sin(np.radians(lon))
    clat = np.cos(np.radians(lat))
    clon = np.cos(np.radians(lon))

    azim_rad = np.radians(azim)
    elev_rad = np.radians(elev)

    # Convert alt/az to direction components
    south  = -r*np.cos(elev_rad)*np.cos(azim_rad)
    east   =  r*np.cos(elev_rad)*np.sin(azim_rad)
    zenith =  r*np.sin(elev_rad)


    x = obs_x + ( slat*clon*south) + (-slon*east) + (clat*clon*zenith)
    y = obs_y + ( slat*slon*south) + ( clon*east) + (clat*slon*zenith)
    z = obs_z + (-clat*     south)                + (slat*     zenith)

    return x, y, z



def AEH2Range(azim, elev, h, lat, lon, alt, accurate=False):
    """ Given an azimuth and altitude, compute the range to a point along the given line of sight
        that has the specified height above the ground.

    Arguments:
        azim: [float] Azimuth (+E of due N) in degrees.
        elev: [float] Elevation in degrees.
        h: [float] Height of the point on the line of sight (meters).
        lat: [float] Latitude of observer in degrees.
        lon: [float] Longitude of observer in degrees.
        alt: [float] Altitude of observer in meters.

    Keyword arguments:
        accurate: [bool] Minimize the range for very accurate solution. False by default, in which case
            the accuracy is +/- 10 m using an analytical approach.

    Return:
        r: [float] Range to point in meters.

    """


    def _heightCostFunction(params, azim, elev, h, lat, lon, alt):

        # Get the guessed range
        r = params

        # Compute the ECEF coordinates with the given range
        x, y, z = AER2ECEF(azim, elev, r, lat, lon, alt)

        # Compute the height
        _, _, h_computed = ecef2LatLonAlt(x, y, z)

        # Return residual between the heights
        return (h_computed - h)**2



    ### Law of sines solution ###

    # Get distance from Earth centre to the position given by geographical coordinates, in WGS84
    N = EARTH.EQUATORIAL_RADIUS/math.sqrt(1.0 - (EARTH.E**2)*math.sin(np.radians(lat))**2)

    # Compute the distance from Earth centre to the observer
    rs = N + alt

    # Compute the distance from Earth centre to the point
    rm = N + h

    # Compute the angle between the observer and the point
    beta = np.radians(elev) + np.arcsin((rs*np.cos(np.radians(elev)))/rm)

    # Compute the range
    r = rm*np.cos(beta)/np.cos(np.radians(elev))

    ### ###


    # Compute an accurate numerical solution if needed
    if accurate:

        # First guess of range if the elevation is higher than 10 degrees
        if elev < np.radians(10):

            # Flat-Earth assumption
            r0 = r

        else:
            # Otherwise, use a distance of 1000 km
            r0 = 1e6

        # Numerically find the range which corresponds to the given height above the ground
        res = scipy.optimize.minimize(_heightCostFunction, r0, \
            args=(azim, elev, h, lat, lon, alt))

        # Minimized range
        r = res.x[0]


    # Return the minimized solution
    return r


def AER2LatLonAlt(azim, elev, r, lat, lon, alt):
    """ Given an azimuth and altitude, compute lat, lon, and lat to a point along the given line of sight
        that is a given distance far away.

    Arguments:
        azim: [float] Azimuth (+E of due N) in degrees.
        elev: [float] Elevation in degrees.
        r: [float] Range along the line of sight (meters).
        lat: [float] Latitude of observer in degrees.
        lon: [float] Longitude of observer in degrees.
        alt: [float] Altitude of observer in meters.

    Return:
        (lat, lon, alt): [tuple of floats] range in meters, latitude and longitude in degrees,
            WGS84 elevation in meters

    """


    # Compute lat/lon/alt of the point on the line of sight
    x, y, z = AER2ECEF(azim, elev, r, lat, lon, alt)
    lat2, lon2, alt2 = ecef2LatLonAlt(x, y, z)
    lat2, lon2 = np.degrees(lat2), np.degrees(lon2)


    return lat2, lon2, alt2


def AEH2LatLonAlt(azim, elev, h, lat, lon, alt):
    """ Given an azimuth and altitude, compute lat, lon, and lat to a point along the given line of sight
        that has the specified height above the ground.

    Arguments:
        azim: [float] Azimuth (+E of due N) in degrees.
        elev: [float] Elevation in degrees.
        h: [float] Height of the point on the line of sight (meters).
        lat: [float] Latitude of observer in degrees.
        lon: [float] Longitude of observer in degrees.
        alt: [float] Altitude of observer in meters.

    Return:
        (r, lat, lon, alt): [tuple of floats] range in meteors, latitude and longitude in degrees, 
            WGS84 elevation in meters

    """

    # Compute the range to the point
    r = AEH2Range(azim, elev, h, lat, lon, alt)


    # Compute lat/lon/alt of the point on the line of sight
    lat2, lon2, alt2 = AER2LatLonAlt(azim, elev, r, lat, lon, alt)


    return r, lat2, lon2, alt2


def AEGeoidH2LatLonAlt(azim, elev, h, lat, lon, alt):
    """ Given an azimuth and altitude, and Height above Geoid compute lat, lon, and lat to a point.

    Arguments:
        azim: [float] Azimuth (+E of due N) in degrees.
        elev: [float] Elevation in degrees.
        h: [float] Height of the point above the geoid (meters).
        lat: [float] Latitude of observer in degrees.
        lon: [float] Longitude of observer in degrees.
        alt: [float] Altitude of observer in meters.

    Return:
        (lat, lon): [tuple of floats] latitude and longitude in degrees

    """

    # Convert azimuth and elevation to radians
    azim = np.radians(azim)
    elev = np.radians(elev)
    lat = np.radians(lat)
    lon = np.radians(lon)

    # Convert observer's geodetic coordinates to ECEF
    obs_x, obs_y, obs_z = latLonAlt2ECEF(lat, lon, alt)

    # Calculate line-of-sight unit vector in ENU coordinates
    los_vector_enu = np.array([
        np.cos(elev)*np.sin(azim),  # East component
        np.cos(elev)*np.cos(azim),  # North component
        np.sin(elev)                # Up component
    ])

    # Transform ENU to ECEF coordinates
    R_enu2ecef = np.array([
        [-np.sin(lon),  -np.sin(lat)*np.cos(lon),  np.cos(lat)*np.cos(lon)],
        [np.cos(lon), -np.sin(lat)*np.sin(lon),  np.cos(lat)*np.sin(lon)],
        [0, np.cos(lat), np.sin(lat)]
    ])

    los_vector = np.dot(R_enu2ecef, los_vector_enu)

    # Compute the range to the point
    r = (h - alt)/np.sin(elev)
    
    # Find the target ECEF coordinates using the optimized range
    target_x = obs_x + r*los_vector[0]
    target_y = obs_y + r*los_vector[1]
    target_z = obs_z + r*los_vector[2]
      
    # Convert target ECEF coordinates to geodetic coordinates
    target_lat, target_lon, h2 = ecef2LatLonAlt(target_x, target_y, target_z)
    target_lat, target_lon = np.degrees(target_lat), np.degrees(target_lon)

    return target_lat, target_lon


def cartesian2Geo(julian_date, x, y, z):
    """ Convert Cartesian ECI coordinates of a point (origin in Earth's centre) to geographical coordinates.
    
    Arguments:
        julian_date: [float] decimal julian date
        X: [float] X coordinate of a point in space (meters)
        Y: [float] Y coordinate of a point in space (meters)
        Z: [float] Z coordinate of a point in space (meters)
    
    Return:
        (lon, lat, ele): [tuple of floats]
            lat: longitude of the point in degrees
            lon: latitude of the point in degrees
            ele: elevation in meters
    """


    # Calculate LLA
    lat, r_LST, ele = ecef2LatLonAlt(x, y, z)

    # Calculate proper longitude from the given JD
    lon, _ = LST2LongitudeEast(julian_date, np.degrees(r_LST))

    # Convert longitude to radians
    lon = np.radians(lon)


    return np.degrees(lat), np.degrees(lon), ele



def areaGeoPolygon(lats, lons, ht):
    """ Computes area of spherical polygon given by geo coordinates, assuming spherical Earth. 
        Line integral based on Green's Theorem.

        Source: https://stackoverflow.com/a/61184491

    Arguments:
        lats: [list/ndarray] A list of latitudes (degrees).
        lons: [list/ndarray] A list of longitudes (degrees).
        ht: [float] Height above sea level (meters).

    Return:
        area: [float] Area enclosed by the polygon in m^2.
    
    """

    lats = np.radians(np.array(lats))
    lons = np.radians(np.array(lons))

    # Compute the mean latitude
    lat_mean = np.mean(lats)

    # Get distance from Earth centre to the position given by mean geographical coordinates, in WGS84 (m)
    N = EARTH.EQUATORIAL_RADIUS/np.sqrt(1.0 - (EARTH.E**2)*np.sin(lat_mean)**2)

    # Compute the total radius including the height
    radius = N + ht


    # Check if a closed polygon is given, and if not, close it
    if (lats[0] != lats[-1]) or (lons[0] != lons[-1]):
        lats = np.append(lats, lats[0])
        lons = np.append(lons, lons[0])

    # Get colatitude (a measure of surface distance as an angle)
    a = np.sin(lats/2)**2 + np.cos(lats)*np.sin(lons/2)**2
    colat = 2*np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Azimuth of each point in segment from the arbitrary origin
    az = np.arctan2(np.cos(lats)*np.sin(lons), np.sin(lats))%(2*np.pi)

    # Calculate step sizes
    daz = np.diff(az)
    daz = (daz + np.pi)%(2*np.pi) - np.pi

    # Determine average surface distance for each step
    deltas = np.diff(colat)/2
    colat = colat[0:-1] + deltas

    # Integral over azimuth is 1-cos(colatitudes)
    integrands = (1 - np.cos(colat))*daz

    # Integrate and save the answer as a fraction of the unit sphere.
    # Note that the sum of the integrands will include a factor of 4pi.
    area = abs(sum(integrands))/(4*np.pi)

    # Could be area of inside or outside the polygon, choose the smaller value aka. the inner area
    area = min(area, 1 - area)

    # Compute the area in square meters
    return area*4*np.pi*radius**2



def raDec2Vector(ra, dec):
    """ Convert stellar equatorial coordinates to a vector with X, Y and Z components.
    @param ra: [float] right ascension in degrees
    @param dec: [float] declination in degrees
    @return (x, y, z): [tuple of floats]
    """

    ra_rad = math.radians(ra)
    dec_rad = math.radians(dec)

    xt = math.cos(dec_rad)*math.cos(ra_rad)
    yt = math.cos(dec_rad)*math.sin(ra_rad)
    zt = math.sin(dec_rad)

    return xt, yt, zt


def vector2RaDec(eci):
    """ Convert Earth-centered inertial vector to right ascension and declination.
    Arguments:
        eci: [3 element ndarray] Vector coordinates in Earth-centered inertial system
    Return:
        (ra, dec): [tuple of floats] right ascension and declination (degrees)
    """

    # Normalize the ECI coordinates
    eci = vectNorm(eci)

    # Calculate declination
    dec = np.arcsin(eci[2])

    # Calculate right ascension
    ra = np.arctan2(eci[1], eci[0])%(2*np.pi)

    return np.degrees(ra), np.degrees(dec)


def altAz2RADec(azim, elev, jd, lat, lon):
    """ Convert azimuth and altitude in a given time and position on Earth to right ascension and
        declination.
    Arguments:
        azim: [float] azimuth (+east of due north) in degrees
        elev: [float] elevation above horizon in degrees
        jd: [float] Julian date
        lat: [float] latitude of the observer in degrees
        lon: [float] longitude of the observer in degrees
    Return:
        (RA, dec): [tuple]
            RA: [float] right ascension (degrees)
            dec: [float] declination (degrees)
    """
    azim = np.radians(azim)
    elev = np.radians(elev)
    lat = np.radians(lat)
    lon = np.radians(lon)

    if isinstance(azim, float) or isinstance(azim, int) or isinstance(azim, np.float64):
        ra, dec = cyaltAz2RADec(azim, elev, jd, lat, lon)
    elif isinstance(azim, np.ndarray):
        ra, dec = cyaltAz2RADec_vect(azim, elev, jd, lat, lon)
    else:
        raise TypeError("azim must be a number or np.ndarray, given: {}".format(type(azim)))

    return np.degrees(ra), np.degrees(dec)


def apparentAltAz2TrueRADec(azim, elev, jd, lat, lon, refraction=True):
    """ Convert the apparent azimuth and altitude in the epoch of date to true (refraction corrected) right 
        ascension and declination in J2000.
    Arguments:
        azim: [float] Azimuth (+East of due North) in degrees (epoch of date).
        elev: [float] Elevation above horizon in degrees (epoch of date).
        jd: [float] Julian date.
        lat: [float] Latitude of the observer in degrees.
        lon: [float] Longitude of the observer in degrees.
    Keyword arguments:
        refraction: [bool] Apply refraction correction. True by default.
    Return:
        (ra, dec): [tuple]
            ra: [float] Right ascension (degrees, J2000).
            dec: [float] Declination (degrees, J2000).
    """

    azim = np.radians(azim)
    elev = np.radians(elev)
    lat = np.radians(lat)
    lon = np.radians(lon)

    if isinstance(azim, float) or isinstance(azim, int) or isinstance(azim, np.float64):
        ra, dec = cyApparentAltAz2TrueRADec(azim, elev, jd, lat, lon, refraction)
    
    elif isinstance(azim, np.ndarray):
        ra, dec = cyApparentAltAz2TrueRADec_vect(azim, elev, jd, lat, lon, refraction)

    else:
        raise TypeError("azim must be a number or np.ndarray, given: {}".format(type(azim)))

    return np.degrees(ra), np.degrees(dec)


def raDec2AltAz(ra, dec, jd, lat, lon):
    """ Calculate the reference azimuth and altitude of the centre of the FOV from the given RA/Dec.
    Arguments:
        ra:  [float] Right ascension in degrees.
        dec: [float] Declination in degrees.
        jd: [float] Reference Julian date.
        lat: [float] Latitude +N in degrees.
        lon: [float] Longitude +E in degrees.
    Return:
        (azim, elev): [tuple of float]: Azimuth and elevation (degrees).
    """
    ra = np.radians(ra)
    dec = np.radians(dec)
    lat = np.radians(lat)
    lon = np.radians(lon)

    # Compute azim and elev using a fast cython function
    if isinstance(ra, float) or isinstance(ra, int) or isinstance(ra, np.float64):
        azim, elev = cyraDec2AltAz(ra, dec, jd, lat, lon)

    elif isinstance(ra, np.ndarray):
        # Compute it for numpy arrays
        azim, elev = cyraDec2AltAz_vect(ra, dec, jd, lat, lon)

    else:
        raise TypeError("ra must be a number or np.ndarray, given: {}".format(type(ra)))

    return np.degrees(azim), np.degrees(elev)


def trueRaDec2ApparentAltAz(ra, dec, jd, lat, lon, refraction=True):
    """ Convert the true right ascension and declination in J2000 to azimuth (+East of due North) and 
        altitude in the epoch of date. The correction for refraction is performed.
    Arguments:
        ra: [float] Right ascension in degrees (J2000).
        dec: [float] Declination in degrees (J2000).
        jd: [float] Julian date.
        lat: [float] Latitude in degrees.
        lon: [float] Longitude in degrees.
    Keyword arguments:
        refraction: [bool] Apply refraction correction. True by default.
    Return:
        (azim, elev): [tuple]
            azim: [float] Azimuth (+east of due north) in degrees (epoch of date).
            elev: [float] Elevation above horizon in degrees (epoch of date).
    """

    ra = np.radians(ra)
    dec = np.radians(dec)
    lat = np.radians(lat)
    lon = np.radians(lon)

    if isinstance(ra, float) or isinstance(ra, int) or isinstance(ra, np.float64):
        azim, elev = cyTrueRaDec2ApparentAltAz(ra, dec, jd, lat, lon, refraction)

    elif isinstance(ra, np.ndarray):

        # Convert JD to appropriate format
        if isinstance(jd, float):
            jd = np.zeros_like(ra) + jd

        # Compute it for numpy arrays
        azim, elev = cyTrueRaDec2ApparentAltAz_vect(ra, dec, jd, lat, lon, refraction)

    else:
        raise TypeError("ra must be a number or np.ndarray, given: {}".format(type(ra)))

    return np.degrees(azim), np.degrees(elev)


def geocentricToApparentRadiantAndVelocity(ra_g, dec_g, vg, lat, lon, elev, jd, include_rotation=True):
    """ Converts the geocentric into apparent meteor radiant and velocity. The conversion is not perfect
        as the zenith attraction correction should be done after the radiant has been derotated for Earth's
        velocity, but it's precise to about 0.1 deg.

    Arguments:
        ra_g: [float] Geocentric right ascension (deg).
        dec_g: [float] Declination (deg).
        vg: [float] Geocentric velocity (m/s).
        lat: [float] State vector latitude (deg)
        lon: [float] State vector longitude (deg).
        ele: [float] State vector elevation (meters).
        jd: [float] Julian date.
    Keyword arguments:
        include_rotation: [bool] Whether the velocity should be corrected for Earth's rotation.
            True by default.
    Return:
        (ra, dec, v_init): Apparent radiant (deg) and velocity (m/s).
    """

    # Compute ECI coordinates of the meteor state vector
    state_vector = geo2Cartesian(lat, lon, elev, jd)

    eci_x, eci_y, eci_z = state_vector

    # Assume that the velocity at infinity corresponds to the initial velocity
    v_init = np.sqrt(vg**2 + (2*6.67408*5.9722)*1e13/vectMag(state_vector))

    # Calculate the geocentric latitude (latitude which considers the Earth as an ellipsoid) of the reference
    # trajectory point
    lat_geocentric = np.degrees(math.atan2(eci_z, math.sqrt(eci_x**2 + eci_y**2)))

    ### Uncorrect for zenith attraction ###

    # Compute the radiant in the local coordinates
    azim, elev = raDec2AltAz(ra_g, dec_g, jd, lat_geocentric, lon)

    # Compute the zenith angle
    eta = np.radians(90.0 - elev)

    # Numerically correct for zenith attraction
    diff = 10e-5
    zc = eta
    while diff > 10e-6:
        # Update the zenith distance
        zc -= diff

        # Calculate the zenith attraction correction
        delta_zc = 2*math.atan((v_init - vg)*math.tan(zc/2.0)/(v_init + vg))
        diff = zc + delta_zc - eta

    # Compute the uncorrected geocentric radiant for zenith attraction
    ra, dec = altAz2RADec(azim, 90.0 - np.degrees(zc), jd, lat_geocentric, lon)

    ### ###

    # Apply the rotation correction
    if include_rotation:
        # Calculate the velocity of the Earth rotation at the position of the reference trajectory point (m/s)
        v_e = 2*math.pi*vectMag(state_vector)*math.cos(np.radians(lat_geocentric))/86164.09053

        # Calculate the equatorial coordinates of east from the reference position on the trajectory
        azimuth_east = 90.0
        altitude_east = 0
        ra_east, dec_east = altAz2RADec(azimuth_east, altitude_east, jd, lat, lon)

        # Compute the radiant vector in ECI coordinates of the apparent radiant
        v_ref_vect = v_init*np.array(raDec2Vector(ra, dec))

        v_ref_nocorr = np.zeros(3)

        # Calculate the derotated reference velocity vector/radiant
        v_ref_nocorr[0] = v_ref_vect[0] + v_e*np.cos(np.radians(ra_east))
        v_ref_nocorr[1] = v_ref_vect[1] + v_e*np.sin(np.radians(ra_east))
        v_ref_nocorr[2] = v_ref_vect[2]

        # Compute the radiant without Earth's rotation included
        ra_norot, dec_norot = vector2RaDec(vectNorm(v_ref_nocorr))
        v_init_norot = vectMag(v_ref_nocorr)

        ra = ra_norot
        dec = dec_norot
        v_init = v_init_norot

    return ra, dec, v_init


###########################################


if __name__ == "__main__":
    # Test the geocentric to apparent radiant function
    ra_g = 108.67522
    dec_g = 31.91152
    vg = 33073.41

    lat = 43.991023
    lon = -80.485553
    elev = 90149.53

    jd = 2456274.636704600416

    print('Geocentric radiant:')
    print('ra_g = ', ra_g)
    print('dec_g = ', dec_g)
    print('vg = ', vg)

    ra, dec, v_init = geocentricToApparentRadiantAndVelocity(ra_g, dec_g, vg, lat, lon, elev, jd, \
                                                             include_rotation=True)

    print('Apparent radiant:')
    print('ra = ', ra)
    print('dec = ', dec)
    print('v_init = ', v_init)



    ### Test computing Lat/Lon/Alt given an azim, elev and height ###

    azim = 0
    elev = 45
    h = 100000
    lat = 45.0
    lon = 13.0
    alt = 90.0

    # Compute lat/lon/alt of the point along the LOS
    r, lat2, lon2, alt2 = AEH2LatLonAlt(azim, elev, h, lat, lon, alt)
    print(r, lat2, lon2, alt2)


    ### ###