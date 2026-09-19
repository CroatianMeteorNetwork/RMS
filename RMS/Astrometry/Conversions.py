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

from RMS.Math import vectMag, vectNorm
from RMS.Misc import UTCFromTimestamp
from RMS.GeoidHeightEGM96 import mslToWGS84Height, wgs84toMSLHeight

# Import Cython functions
import pyximport
pyximport.install(setup_args={'include_dirs': [np.get_include()]})
from RMS.Astrometry.CyFunctions import cyaltAz2RADec, cyraDec2AltAz, cyApparentAltAz2TrueRADec, \
    cyApparentAltAz2TrueRADec_vect, cyTrueRaDec2ApparentAltAz, cyTrueRaDec2ApparentAltAz_vect, cyjd2GST, \
    trueOfDateFromJ2000, j2000FromTrueOfDate, pyRefractionTrueToApparent, pyRefractionApparentToTrue

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


def JD2GST(jd):
    """ Apparent Greenwich sidereal time, the same definition as the astrometric kernels use (mean sidereal
        time plus the equation of the equinoxes). All sidereal times in RMS come from here or from the
        kernel's cyjd2GST, which this calls, so hour angles of true-of-date right ascensions are consistent
        everywhere.

    Arguments:
        jd: [float or ndarray] Julian date.

    Return:
        gst: [float or ndarray] Apparent Greenwich sidereal time (deg), in [0, 360).

    """

    if np.ndim(jd) == 0:
        return cyjd2GST(float(jd))

    return np.array([cyjd2GST(float(j)) for j in np.asarray(jd).ravel()]).reshape(np.shape(jd))


def JD2LST(julian_date, lon):
    """ Convert Julian date to apparent Local Sidereal Time and Greenwich Sidereal Time.

    Arguments;
        julian_date: [float] decimal julian date, epoch J2000.0
        lon: [float] longitude of the observer in degrees

    Return:
        [tuple]: (LST, GST): [tuple of floats] a tuple of Local Sidereal Time and Greenwich Sidereal Time
            (degrees)
    """

    # Greenwich Sidereal Time (apparent)
    GST = JD2GST(julian_date)

    # Local Sidereal Time
    LST = (GST + lon + 360)%360

    return LST, GST


def JD2HourAngle(jd):
    """ Reference hour angle of a platepar: the apparent Greenwich sidereal time at its reference time. The
        kernel adds cyjd2LST(jd) - Ho to RA_d to follow the sky, so this has to be the same sidereal time
        that the kernel uses.

    Arguments:
        jd: [float] Julian date.

    Return:
        hour_angle: [float] Hour angle (deg), normalized to [0, 360).

    """

    return JD2GST(jd)


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
    # The normalized dot product can round just outside [-1, 1] when P is directly overhead or
    #   directly below S, which would make arccos return NaN, so clamp it
    cos_zenith = (sx*dx + sy*dy + sz*dz)/np.sqrt((sx**2 + sy**2 + sz**2)*(dx**2 + dy**2 + dz**2))
    alt = np.degrees(np.pi/2 - np.arccos(np.clip(cos_zenith, -1.0, 1.0)))

    # Compute the azimuth
    
    cos_az = (-sz*sx*dx - sz*sy*dy + (sx**2 + sy**2)*dz)/np.sqrt(
                                            (sx**2 + sy**2)*(sx**2 + sy**2 + sz**2)*(dx**2 + dy**2 + dz**2)
                                            )
    
    sin_az = (-sy*dx + sx*dy)/np.sqrt((sx**2 + sy**2)*(dx**2 + dy**2 + dz**2))

    azim = np.degrees(np.arctan2(sin_az, cos_az))%360


    return azim, alt

def addECEFVectortoLatLonEle(lat, lon, ele_egm96, x ,y, z, config, radians=False):
    """

    Add an ECEF vector to a latitude, longitude and elevation.

    Given a latitude and longitude in degrees, optionally radians, wgs84 and elevation in egm96 in meters
    and an ECEF vector, return a new position in latitude and longitude in degrees, optionally radians,
    and an elevation in egm96 in meters.


    Arguments:
        lat: [float] latitude wgs84 degrees, optionally radians
        lon: [float] longitude wgs84 degrees, optionally radians
        ele_egm96: [float] elevation in meters egm96 basis
        x: [float] component of ECEF coordinate vector in meters
        y: [float]  component of ECEF coordinate vector in meters
        z: [float]  component of ECEF coordinate vector in meters
        config: Config instance with the path to EGM96 coefficients

    Keyword arguments:
        radians: [bool] optional, default False

    Return:
        lat_: [float] latitude wgs84 degrees, optionally radians
        lon_: [float] longitude wgs84, optionally radians
        ele_egm96_: [float]  in meters egm96 basis
    """

    if not radians:
        # Convert to radians and elevation and elevation in egm96 to altitude in wgs84
        lat_rads, lon_rads = np.radians(lat), np.radians(lon)
    else:
        # Pass through as radians
        lat_rads, lon_rads = lat, lon

    # Convert elevation in egm96 to altitude in wgs84
    alt_wgs84 = mslToWGS84Height(lat_rads, lon_rads, ele_egm96, config)

    # Compute ecef coordinates
    ecef_x, ecef_y, ecef_z = latLonAlt2ECEF(lat_rads, lon_rads, alt_wgs84)

    # Add the ecef vector to the original position
    ecef_x_, ecef_y_, ecef_z_ = x + ecef_x, y + ecef_y, z + ecef_z

    # Convert back to latitude and longitude in radians, altitude wgs84
    lat_rads_, lon_rads_, alt_wgs84_ = ecef2LatLonAlt(ecef_x_, ecef_y_, ecef_z_)

    # Convert wgs84 altitude to egm96 elevation at the new position
    ele_egm96_ = wgs84toMSLHeight(lat_rads_, lon_rads_, alt_wgs84_, config)

    if not radians:
        # Convert to degrees
        lat_, lon_ = np.degrees(lat_rads_), np.degrees(lon_rads_)
    else:
        # Pass through as rads
        lat_, lon_ = lat_rads_, lon_rads_

    # Return computed values
    return lat_, lon_, ele_egm96_

def getECEFVectorBetweenGeoPoints(lat_1, lon_1, ele_1_egm96, lat_2, lon_2, ele_2_egm96, config, radians=False):
    """
    Compute ECEF vector between two points.

    From a pair of lat, lon in degrees, optionally radians, and elevation in egm96 basis
    compute the ECEF vector between the two points.

    Point 2 is the end of the vector, Point 1 is the start of the vector.

    Arguments:
        lat_1: [float] latitude in degrees optionally radians
        lon_1: [float] longitude in degrees optionally radians
        ele_1_egm96: [float] elevation in meters egm96 basis
        lat_2: [float] latitude in degrees optionally radians
        lon_2: [float] longitude in degrees optionally radians
        ele_2_egm96: [float] elevation in meters egm96 basis
        config: Config instance with the path to EGM96 coefficients

    Keyword arguments:
        radians: [bool] optional default False

    Returns:
        [float] ecef vector x component
        [float] ecef vector y component
        [float] ecef vector z component
    """

    if not radians:
        # Convert to degrees
        lat_1_rads, lon_1_rads = np.radians(lat_1), np.radians(lon_1)
        lat_2_rads, lon_2_rads = np.radians(lat_2), np.radians(lon_2)
    else:
        # Pass through as radians
        lat_1_rads, lon_1_rads = lat_1, lon_1
        lat_2_rads, lon_2_rads = lat_2, lon_2

    # Compute wgs84 elevations
    alt_1_wgs84 = mslToWGS84Height(lat_1_rads, lon_1_rads, ele_1_egm96, config)
    alt_2_wgs84 = mslToWGS84Height(lat_2_rads, lon_2_rads, ele_2_egm96, config)

    # Compute ecef coordinates
    ecef_x_1, ecef_y_1, ecef_z_1 = latLonAlt2ECEF(lat_1_rads, lon_1_rads, alt_1_wgs84)
    ecef_x_2, ecef_y_2, ecef_z_2 = latLonAlt2ECEF(lat_2_rads, lon_2_rads, alt_2_wgs84)

    # Compute vector
    x_, y_, z_ = ecef_x_2 - ecef_x_1, ecef_y_2 - ecef_y_1, ecef_z_2 - ecef_z_1

    # Return computed values
    return x_, y_, z_

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
        accurate: [bool] Solve for the range on the WGS84 ellipsoid instead of using the analytical
            approximation, which is accurate to about +/- 10 m for a target above the observer. False by
            default. The approximation also picks the wrong intersection for a target below the observer,
            so this has to be set for a target that the camera looks down at.

    Return:
        r: [float] Range to point in meters. NaN if the line of sight never reaches the given height.

    """


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


    # Solve on the ellipsoid if the approximation is not good enough, or if the target is below the
    #   observer, where the law of sines above returns the far intersection instead of the near one
    if accurate:

        # Observer position and the unit line of sight, both in ECEF
        obs = np.array(latLonAlt2ECEF(np.radians(lat), np.radians(lon), alt))
        los = np.array(AER2ECEF(azim, elev, 1.0, lat, lon, alt)) - obs

        # Geocentric distances of the observer and of the target, the latter through the Earth radius
        #   under the observer. Only the starting guess needs this, the iteration below works on the
        #   ellipsoid itself.
        rs = np.sqrt(np.dot(obs, obs))
        rm = (rs - alt) + h

        # The ray meets the sphere of radius rm where r**2 + 2*r*(obs . los) + rs**2 - rm**2 = 0. For a
        #   target above the observer only one root is in front of the camera; for one below there are two
        #   and the near one is wanted.
        b = np.dot(obs, los)
        disc = b*b + rm*rm - rs*rs

        # The line of sight never reaches that height
        if disc < 0:
            return np.nan

        if rm >= rs:
            r = -b + np.sqrt(disc)

        else:
            r = -b - np.sqrt(disc)

        if r <= 0:
            return np.nan

        # Refine on the WGS84 ellipsoid. The height along the line of sight grows with the range at the
        #   rate of the component of the line of sight along the local vertical, which is the derivative
        #   Newton's method needs.
        for _ in range(20):

            lat_p, lon_p, h_p = ecef2LatLonAlt(*(obs + r*los))
            up = np.array([np.cos(lat_p)*np.cos(lon_p), np.cos(lat_p)*np.sin(lon_p), np.sin(lat_p)])
            slope = np.dot(los, up)

            # The ray is grazing, so the height barely changes with the range and Newton cannot step
            if abs(slope) < 1e-9:
                break

            dr = (h - h_p)/slope
            r += dr

            if abs(dr) < 1e-6:
                break


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

    # Range at which the line of sight is at the given height above the ellipsoid. Solved on the
    #   ellipsoid rather than taken as (h - alt)/sin(elev): that flat-Earth range ignores the Earth
    #   curving away under the ray, which puts a target at 100 km and 5 deg elevation at 200 km instead,
    #   and it cannot be inverted by geoHt2XY(), which places the target at exactly h.
    r = AEH2Range(azim, elev, h, lat, lon, alt, accurate=True)

    if not np.isfinite(r):
        return np.nan, np.nan

    # Convert the target's ECEF coordinates to geodetic coordinates
    target_lat, target_lon, _ = ecef2LatLonAlt(*AER2ECEF(azim, elev, r, lat, lon, alt))

    return np.degrees(target_lat), np.degrees(target_lon)


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


def apparentAltAz2TrueRADec(azim, elev, jd, lat, lon, refraction=True, refraction_scale=1.0):
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
        refraction_scale: [float] Scale of the refraction for the observer's height above sea level, from
            RMS.Astrometry.CyFunctions.refractionScale(elev). 1.0 by default (sea level).
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
        ra, dec = cyApparentAltAz2TrueRADec(azim, elev, jd, lat, lon, refraction, refraction_scale)
    
    elif isinstance(azim, np.ndarray):
        ra, dec = cyApparentAltAz2TrueRADec_vect(azim, elev, jd, lat, lon, refraction, refraction_scale)

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


def trueRaDec2ApparentAltAz(ra, dec, jd, lat, lon, refraction=True, refraction_scale=1.0):
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
        refraction_scale: [float] Scale of the refraction for the observer's height above sea level, from
            RMS.Astrometry.CyFunctions.refractionScale(elev). 1.0 by default (sea level).
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
        azim, elev = cyTrueRaDec2ApparentAltAz(ra, dec, jd, lat, lon, refraction, refraction_scale)

    elif isinstance(ra, np.ndarray):

        # Convert JD to appropriate format
        if isinstance(jd, float):
            jd = np.zeros_like(ra) + jd

        # Compute it for numpy arrays
        azim, elev = cyTrueRaDec2ApparentAltAz_vect(ra, dec, jd, lat, lon, refraction, refraction_scale)

    else:
        raise TypeError("ra must be a number or np.ndarray, given: {}".format(type(ra)))

    return np.degrees(azim), np.degrees(elev)


def _perPoint(func, *args):
    """ Helper: apply a scalar (ra, dec)-style function to scalars or to arrays of the first two arguments,
        returning floats for scalars and arrays otherwise. """

    a, b = args[0], args[1]
    rest = args[2:]

    if np.ndim(a) == 0 and np.ndim(b) == 0:
        return func(float(a), float(b), *rest)

    a, b = np.broadcast_arrays(np.atleast_1d(a).astype(np.float64), np.atleast_1d(b).astype(np.float64))
    out = np.array([func(float(x), float(y), *rest) for x, y in zip(a.ravel(), b.ravel())])

    return out[:, 0].reshape(a.shape), out[:, 1].reshape(a.shape)


def trueOfDateRaDec2ApparentAltAz(ra, dec, jd, lat, lon, refraction=True, refraction_scale=1.0):
    """ Convert a true-of-date right ascension and declination, such as the platepar reference pointing
        RA_d/dec_d, to apparent azimuth and altitude. No precession or nutation is applied, only the hour
        angle against the apparent sidereal time and, optionally, the refraction.

        This is the conversion that matches the astrometric kernel: pointingCorrection takes RA_d/dec_d to
        be true-of-date and precesses them to J2000 before comparing with the catalog, so the fit defines
        them in that frame. Feeding them to trueRaDec2ApparentAltAz, which expects J2000, precesses them a
        second time and lands 8-19 arcmin from the real pointing in 2026.

    Arguments:
        ra: [float or ndarray] Right ascension in degrees (true of date).
        dec: [float or ndarray] Declination in degrees (true of date).
        jd: [float] Julian date.
        lat: [float] Latitude in degrees.
        lon: [float] Longitude in degrees.

    Keyword arguments:
        refraction: [bool] Apply refraction correction. True by default.
        refraction_scale: [float] Scale of the refraction for the observer's height, from
            refractionScale(). 1.0 by default (sea level).

    Return:
        (azim, elev): [tuple of floats or ndarrays] Azimuth (+east of due north) and apparent altitude in
            degrees.

    """

    def one(ra_i, dec_i):

        azim, elev = cyraDec2AltAz(np.radians(ra_i), np.radians(dec_i), jd, np.radians(lat), np.radians(lon))

        if refraction:
            elev = pyRefractionTrueToApparent(elev, refraction_scale)

        return np.degrees(azim), np.degrees(elev)

    return _perPoint(one, ra, dec)


def apparentAltAz2TrueOfDateRaDec(azim, elev, jd, lat, lon, refraction=True, refraction_scale=1.0):
    """ Inverse of trueOfDateRaDec2ApparentAltAz: apparent azimuth and altitude to a true-of-date right
        ascension and declination, the frame of the platepar reference pointing RA_d/dec_d.

    Arguments:
        azim: [float or ndarray] Azimuth (+east of due north) in degrees.
        elev: [float or ndarray] Apparent altitude in degrees.
        jd: [float] Julian date.
        lat: [float] Latitude in degrees.
        lon: [float] Longitude in degrees.

    Keyword arguments:
        refraction: [bool] Remove the refraction first. True by default.
        refraction_scale: [float] Scale of the refraction for the observer's height, from
            refractionScale(). 1.0 by default (sea level).

    Return:
        (ra, dec): [tuple of floats or ndarrays] Right ascension and declination in degrees (true of date).

    """

    def one(azim_i, elev_i):

        elev_i = np.radians(elev_i)

        if refraction:
            elev_i = pyRefractionApparentToTrue(elev_i, refraction_scale)

        ra, dec = cyaltAz2RADec(np.radians(azim_i), elev_i, jd, np.radians(lat), np.radians(lon))

        return np.degrees(ra), np.degrees(dec)

    return _perPoint(one, azim, elev)


def trueOfDateRaDec2J2000(ra, dec, jd):
    """ True equator and equinox of date -> J2000, in degrees (see j2000FromTrueOfDate).

    Arguments:
        ra: [float or ndarray] Right ascension in degrees (true of date).
        dec: [float or ndarray] Declination in degrees (true of date).
        jd: [float] Julian date of the epoch of date.

    Return:
        (ra, dec): [tuple of floats or ndarrays] Right ascension and declination in degrees (J2000).

    """

    def one(ra_i, dec_i):
        ra_j, dec_j = j2000FromTrueOfDate(jd, np.radians(ra_i), np.radians(dec_i))
        return np.degrees(ra_j), np.degrees(dec_j)

    return _perPoint(one, ra, dec)


def j2000RaDec2TrueOfDate(ra, dec, jd):
    """ J2000 -> true equator and equinox of date, in degrees (see trueOfDateFromJ2000).

    Arguments:
        ra: [float or ndarray] Right ascension in degrees (J2000).
        dec: [float or ndarray] Declination in degrees (J2000).
        jd: [float] Julian date of the epoch of date.

    Return:
        (ra, dec): [tuple of floats or ndarrays] Right ascension and declination in degrees (true of date).

    """

    def one(ra_i, dec_i):
        ra_t, dec_t = trueOfDateFromJ2000(jd, np.radians(ra_i), np.radians(dec_i))
        return np.degrees(ra_t), np.degrees(dec_t)

    return _perPoint(one, ra, dec)


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

    # required for testing getECEFVectorBetweenGeoPoints and addECEFVectortoLatLonEle
    import os
    import RMS.ConfigReader as cr

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


    # Test of getECEFVectorBetweenGeoPoints and addECEFVectortoLatLonEle

    config = cr.loadConfigFromDirectory(".config", os.path.abspath('.'))
    # location one
    lat_deg_1, lon_deg_1, elevation_egm_96_1 = -32.007433, 116.134826, 380
    lat_rad_1, lon_rad_1 = np.radians(lat_deg_1), np.radians(lon_deg_1)
    # location two
    lat_deg_2, lon_deg_2, elevation_egm_96_2 = -31.952561, 115.844618, 70
    lat_rad_2, lon_rad_2 = np.radians(lat_deg_2), np.radians(lon_deg_2)

    print("Test in degrees\n")
    x, y, z = getECEFVectorBetweenGeoPoints(lat_deg_1, lon_deg_1, elevation_egm_96_1,
                                            lat_deg_2, lon_deg_2, elevation_egm_96_2,
                                            config)
    lat_deg_3, lon_deg_3, ele_egm96_3 = addECEFVectortoLatLonEle(lat_deg_1, lon_deg_1,elevation_egm_96_1,
                                                                 x ,y, z,  config)

    print("Start    lat, lon, ele: {:.6f}, {:.6f}, {:.0f}".format(lat_deg_1,lon_deg_1,elevation_egm_96_1))
    print("End      lat, lon, ele: {:.6f}, {:.6f}, {:.0f}".format(lat_deg_2, lon_deg_2, elevation_egm_96_2))
    print("Returned lat, lon, ele: {:.6f}, {:.6f}, {:.0f}\n".format(lat_deg_3, lon_deg_3, ele_egm96_3))

    print("Test in radians\n")
    x, y, z = getECEFVectorBetweenGeoPoints(lat_rad_1, lon_rad_1, elevation_egm_96_1,
                                            lat_rad_2, lon_rad_2, elevation_egm_96_2,
                                            config, radians=True)
    lat_rad_3, lon_rad_3, ele_egm96_3 = addECEFVectortoLatLonEle(lat_rad_1, lon_rad_1, elevation_egm_96_1,
                                                                 x, y, z, config, radians=True)

    print("Start    lat, lon, ele: {:.6f}, {:.6f}, {:.0f}".format(lat_rad_1,lon_rad_1,elevation_egm_96_1))
    print("End      lat, lon, ele: {:.6f}, {:.6f}, {:.0f}".format(lat_rad_2, lon_rad_2, elevation_egm_96_2))
    print("Returned lat, lon, ele: {:.6f}, {:.6f}, {:.0f}\n".format(lat_rad_3, lon_rad_3, ele_egm96_3))

    print("Length of vector is {:.2f} km".format(((x ** 2 + y ** 2 + z ** 2) ** 0.5) / 1000))

    ### ###