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
from datetime import datetime, timedelta

### CONSTANTS ###

# Define Julian epoch
JULIAN_EPOCH = datetime(2000, 1, 1, 12) # noon (the epoch name is unrelated)
J2000_JD = timedelta(2451545) # julian epoch in julian dates

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


### Time transformations ###

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



def JD2LST(julian_date, lon):
    """ Convert Julian date to Local Sidreal Time and Greenwich Sidreal Time. 

    @param julian_date: [float] decimal julian date, epoch J2000.0
    @param lon: [float] longitude of the observer in degrees

    @return (LST, GST): [tuple of floats] a tuple of Local Sidreal Time and Greenwich Sidreal Time
    """

    t = (julian_date - J2000_JD.days)/36525

    # Greenwich Sidreal Time
    GST = 280.46061837 + 360.98564736629 * (julian_date - 2451545) + 0.000387933 *t**2 - ((t**3) / 38710000)
    GST = (GST+360) % 360

    # Local Sidreal Time
    LST = (GST + lon + 360) % 360
    
    return LST, GST


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

    lon_rad = math.radians(lon)
    lat_rad = math.radians(lat)

    # Get Local Sidreal Time
    LST_rad = math.radians(JD2LST(julian_date, lon)[0])

    # Get distance from Earth centre to the position given by geographical coordinates
    Rh = h + math.sqrt(EARTH.POLAR_RADIUS**2 + (EARTH.SQR_DIFF/((EARTH.RATIO * math.tan(lat_rad)) * 
        (EARTH.RATIO * math.tan(lat_rad)) + 1)))

    # Calculate Cartesian coordinates (in meters)
    x = Rh * math.cos(lat_rad) * math.cos(LST_rad)
    y = Rh * math.cos(lat_rad) * math.sin(LST_rad)
    z = Rh * math.sin(lat_rad)

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


def stellar2Vector(ra, dec):
    """ Convert stellar equatorial coordinates to a vector with X, Y and Z components. 

    @param ra: [float] right ascension in degrees
    @param dec: [float] declination in degrees

    @return (x, y, z): [tuple of floats]
    """
    
    ra_rad = math.radians(ra)
    dec_rad = math.radians(dec)

    xt = math.cos(dec_rad) * math.cos(ra_rad)
    yt = math.cos(dec_rad) * math.sin(ra_rad)
    zt = math.sin(dec_rad)

    return xt, yt, zt


###########################################


### Precession ###

def equatorialCoordPrecession(start_epoch, final_epoch, ra, dec):
    """ Corrects Right Ascension and Declination from one epoch to another, taking only precession into 
        account.

        Implemented from: Jean Meeus - Astronomical Algorithms, 2nd edition, pages 134-135

    @param start_epoch: [float] Julian date of the starting epoch
    @param final_epoch: [float] Julian date of the final epoch
    @param ra: [float] non-corrected right ascension in degrees
    @param dec: [float] non-corrected declination in degrees

    @return (ra, dec): [tuple of floats] precessed equatorial coordinates in degrees
    """

    ra = math.radians(ra)
    dec = math.radians(dec)

    T = (start_epoch - 2451545) / 36525.0
    t = (final_epoch - start_epoch) / 36525.0

    # Calculate correction parameters
    zeta  = ((2306.2181 + 1.39656*T - 0.000139*T**2)*t + (0.30188 - 0.000344*T)*t**2 + 0.017998*t**3)/3600
    z     = ((2306.2181 + 1.39656*T - 0.000139*T**2)*t + (1.09468 + 0.000066*T)*t**2 + 0.018203*t**3)/3600
    theta = ((2004.3109 - 0.85330*T - 0.000217*T**2)*t - (0.42665 + 0.000217*T)*t**2 - 0.041833*t**3)/3600

    # Convert parameters to radians
    zeta, z, theta = map(math.radians, (zeta, z, theta))

    # Calculate the next set of parameters
    A = math.cos(dec) * math.sin(ra + zeta)
    B = math.cos(theta)*math.cos(dec)*math.cos(ra + zeta) - math.sin(theta)*math.sin(dec)
    C = math.sin(theta)*math.cos(dec)*math.cos(ra + zeta) + math.cos(theta)*math.sin(dec)

    # Calculate right ascension
    ra_corr = math.atan2(A, B) + z

    # Calculate declination (apply a different equation if close to the pole, closer then 0.5 degrees)
    if (math.pi/2 - abs(dec)) < math.radians(0.5):
        dec_corr = math.acos(math.sqrt(A**2 + B**2))
    else:
        dec_corr = math.asin(C)


    return math.degrees(ra_corr), math.degrees(dec_corr)


##################