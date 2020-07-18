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

import math
from datetime import datetime, timedelta, MINYEAR

import numpy as np

from RMS.Math import vectMag, vectNorm

# Import Cython functions
import pyximport

pyximport.install(setup_args={'include_dirs': [np.get_include()]})
import RMS.Astrometry.CyFunctions as cy

### CONSTANTS ###

# Define Julian epoch
JULIAN_EPOCH = datetime(2000, 1, 1, 12)  # noon (the epoch name is unrelated)
J2000_JD = timedelta(2451545)  # julian epoch in julian dates


class EARTH_CONSTANTS(object):
    """ Holds Earth's shape parameters. """

    def __init__(self):
        # Earth elipsoid parameters in meters (source: IERS 2003)
        self.EQUATORIAL_RADIUS = 6378136.6
        self.POLAR_RADIUS = 6356751.9
        self.RATIO = self.EQUATORIAL_RADIUS/self.POLAR_RADIUS
        self.SQR_DIFF = self.EQUATORIAL_RADIUS**2 - self.POLAR_RADIUS**2


# Initialize Earth shape constants object
EARTH = EARTH_CONSTANTS()


#################


### DECORATORS ###

def floatArguments(func):
    """ A decorator that converts all function arguments to float.

    @param func: a function to be decorated
    @return :[funtion object] the decorated function
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
    dt = datetime.utcfromtimestamp(float(ts) + float(tu)/1000000)

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
    """ Convert Julian date to Local Sidreal Time and Greenwich Sidreal Time.

    Arguments;
        julian_date: [float] decimal julian date, epoch J2000.0
        lon: [float] longitude of the observer in degrees

    Return:
        [tuple]: (LST, GST): [tuple of floats] a tuple of Local Sidreal Time and Greenwich Sidreal Time
            (degrees)
    """

    t = (julian_date - J2000_JD.days)/36525.0

    # Greenwich Sidreal Time
    GST = 280.46061837 + 360.98564736629*(julian_date - 2451545) + 0.000387933*t**2 - ((t**3)/38710000)
    GST = (GST + 360)%360

    # Local Sidreal Time
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

@floatArguments
def geo2Cartesian(lat, lon, h, julian_date):
    """ Convert geographical Earth coordinates to Cartesian coordinate system (Earth center as origin).
        The Earth is considered as an elipsoid.
    @param lat: [float] latitude of the observer in degrees
    @param lon: [float] longitde of the observer in degress
    @param h: [int or float] elevation of the observer in meters
    @param julian_date: [float] decimal julian date, epoch J2000.0
    @return (x, y, z): [tuple of floats] a tuple of X, Y, Z Cartesian coordinates
    """

    lat_rad = math.radians(lat)

    # Get Local Sidreal Time
    LST_rad = math.radians(JD2LST(julian_date, lon)[0])

    # Get distance from Earth centre to the position given by geographical coordinates
    Rh = h + math.sqrt(EARTH.POLAR_RADIUS**2 + (EARTH.SQR_DIFF/((EARTH.RATIO*math.tan(lat_rad))*
                                                                (EARTH.RATIO*math.tan(lat_rad)) + 1)))

    # Calculate Cartesian coordinates (in meters)
    x = Rh*math.cos(lat_rad)*math.cos(LST_rad)
    y = Rh*math.cos(lat_rad)*math.sin(LST_rad)
    z = Rh*math.sin(lat_rad)

    return x, y, z


def cartesian2Geographical(julian_date, lon, Xi, Yi, Zi):
    """ Convert Cartesian coordinates of a point (origin in Earth's centre) to geographical coordinates.
    @param julian_date: [float] decimal julian date, epoch J2000.0
    @param lon: [float] longitde of the observer in degress
    @param Xi: [float] X coordinate of a point in space (meters)
    @param Yi: [float] Y coordinate of a point in space (meters)
    @param Zi: [float] Z coordinate of a point in space (meters)
    @return (lon_p, lat_p): [tuple of floats]
        lon_p: longitude of the point in degrees
        lat_p: latitude of the point in degrees
    """

    # Get LST and GST
    LST, GST = JD2LST(julian_date, lon)

    # Convert Cartesian coordinates to latitude and longitude
    lon_p = math.degrees(math.atan2(Yi, Xi) - math.radians(GST))
    lat_p = math.degrees(math.atan2(math.sqrt(Xi**2 + Yi**2), Zi))

    return lon_p, lat_p


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
    """ Convert Earth-centered intertial vector to right ascension and declination.
    Arguments:
        eci: [3 element ndarray] Vector coordinates in Earth-centered inertial system
    Return:
        (ra, dec): [tuple of floats] right ascension and declinaton (degrees)
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
        lon: [float] longitde of the observer in degrees
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
        ra, dec = cy.cyaltAz2RADec(azim, elev, jd, lat, lon)
    elif isinstance(azim, np.ndarray):
        ra, dec = cy.cyaltAz2RADec_vect(azim, elev, jd, lat, lon)
    else:
        raise TypeError("azim must be a number or np.ndarray, given: {}".format(type(azim)))

    return np.degrees(ra), np.degrees(dec)


def apparentAltAz2TrueRADec(azim, elev, jd, lat, lon, refraction=True):
    azim = np.radians(azim)
    elev = np.radians(elev)
    lat = np.radians(lat)
    lon = np.radians(lon)

    if isinstance(azim, float) or isinstance(azim, int) or isinstance(azim, np.float64):
        ra, dec = cy.apparentAltAz2TrueRADec(azim, elev, jd, lat, lon, refraction)
    elif isinstance(azim, np.ndarray):
        ra, dec = cy.apparentAltAz2TrueRADec_vect(azim, elev, jd, lat, lon, refraction)
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
        azim, elev = cy.cyraDec2AltAz(ra, dec, jd, lat, lon)
    elif isinstance(ra, np.ndarray):
        azim, elev = cy.cyraDec2AltAz_vect(ra, dec, jd, lat, lon)
    else:
        raise TypeError("ra must be a number or np.ndarray, given: {}".format(type(ra)))

    return np.degrees(azim), np.degrees(elev)


def trueRaDec2ApparentAltAz(ra, dec, jd, lat, lon, refraction=True):
    ra = np.radians(ra)
    dec = np.radians(dec)
    lat = np.radians(lat)
    lon = np.radians(lon)

    if isinstance(ra, float) or isinstance(ra, int) or isinstance(ra, np.float64):
        azim, elev = cy.trueRaDec2ApparentAltAz(ra, dec, jd, lat, lon, refraction)
    elif isinstance(ra, np.ndarray):
        azim, elev = cy.trueRaDec2ApparentAltAz_vect(ra, dec, jd, lat, lon, refraction)
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

    # Calculate the geocentric latitude (latitude which considers the Earth as an elipsoid) of the reference
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
