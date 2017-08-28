""" This module contains procedures for applying astrometry and field corrections to meteor data.
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

from __future__ import print_function, division, absolute_import

import os
import sys
import math
import datetime
import shutil

import numpy as np

from RMS.Astrometry.Conversions import date2JD, datetime2JD
from RMS.Formats.Platepar import PlateparCMN
from RMS.Formats.FTPdetectinfo import readFTPdetectinfo, writeFTPdetectinfo
from RMS.Formats.FFbin import filenameToDatetime as filenameToDatetimeFF


def applyFieldCorrection(x_poly, y_poly, X_res, Y_res, F_scale, X_data, Y_data, level_data):
    """ Apply field correction and vignetting correction to all given image points. 

    @param x_poly: [ndarray] 1D numpy array of 12 elements containing X axis polynomial parameters
    @param y_poly: [ndarray] 1D numpy array of 12 elements containing Y axis polynomial parameters
    @param X_res: [int] camera X axis resolution (longer)
    @param Y_res: [int] camera Y axis resolution (shorter)
    @param F_scale: [float] sum of image scales per each image axis (arcsec per px)
    @param X_data: [ndarray] 1D float numpy array containing X component of the detection point
    @param Y_data: [ndarray] 1D float numpy array containing Y component of the detection point
    @param level_data: [ndarray] 1D int numpy array containing levels of detection points

    @return (X_corrected, Y_corrected, levels_corrected): [tuple of ndarrays]
        X_corrected: 1D numpy array containing distortion corrected X component
        Y_corrected: 1D numpy array containing distortion corrected Y component
        level_data: 1D numpy array containing vignetting corrected levels
    """

    # Scale the resolution to CIF
    X_scale = X_res/384.0
    Y_scale = Y_res/288.0

    # Initialize final values containers
    X_corrected = np.zeros_like(X_data, dtype=np.float64)
    Y_corrected = np.zeros_like(Y_data, dtype=np.float64)
    levels_corrected = np.zeros_like(level_data, dtype=np.float64)

    i = 0

    data_matrix = np.vstack((X_data, Y_data, level_data)).T

    # Go through all given data points
    for Xdet, Ydet, level in data_matrix:

        # Scale the point coordinates to CIF resolution
        Xdet = (Xdet - X_res/2)/X_scale
        Ydet = (Ydet - Y_res/2)/Y_scale

        # # Apply vignetting correction
        # if (np.sqrt((Xdet - 192)**2 + (Ydet - 192)**2) > 120):
        #     level = level * (1 + 0.00245*(np.sqrt((Xdet - 192)**2 + (Ydet - 192)**2) - 120))

        dX = (x_poly[0]
            + x_poly[1]*Xdet
            + x_poly[2]*Ydet
            + x_poly[3]*Xdet**2
            + x_poly[4]*Xdet*Ydet
            + x_poly[5]*Ydet**2
            + x_poly[6]*Xdet**3
            + x_poly[7]*Xdet**2*Ydet
            + x_poly[8]*Xdet*Ydet**2
            + x_poly[9]*Ydet**3
            + x_poly[10]*Xdet*np.sqrt(Xdet**2 + Ydet**2)
            + x_poly[11]*Ydet*np.sqrt(Xdet**2 + Ydet**2))

        # Add the distortion correction
        X_pix = Xdet + dX

        dY = (y_poly[0]
            + y_poly[1]*Xdet
            + y_poly[2]*Ydet
            + y_poly[3]*Xdet**2
            + y_poly[4]*Xdet*Ydet
            + y_poly[5]*Ydet**2
            + y_poly[6]*Xdet**3
            + y_poly[7]*Xdet**2*Ydet
            + y_poly[8]*Xdet*Ydet**2
            + y_poly[9]*Ydet**3
            + y_poly[10]*Ydet*np.sqrt(Xdet**2 + Ydet**2)
            + y_poly[11]*Xdet*np.sqrt(Xdet**2 + Ydet**2))

        # Add the distortion correction
        Y_pix = Ydet + dY

        # Scale back image coordinates
        X_pix = X_pix/F_scale
        Y_pix = Y_pix/F_scale

        # Store values to final arrays
        X_corrected[i] = X_pix
        Y_corrected[i] = Y_pix
        levels_corrected[i] = level

        i += 1

    return X_corrected, Y_corrected, levels_corrected



def XY2altAz(lat, lon, RA_d, dec_d, Ho, rot_param, X_data, Y_data):
    """ Convert image coordinates (X, Y) to sky altitude and azimuth. 

    @param lat: [float] latitude of the observer in degrees
    @param lon: [float] longitde of the observer in degress
    @param RA_d: [float] right ascension of the image centre (degrees)
    @param dec_d: [float] declination of the image centre (degrees)
    @param Ho: [float] referent hour angle
    @param rot_param: [float] field rotation parameter (degrees)
    @param X_data: [ndarray] 1D numpy array containing distortion corrected X component
    @param Y_data: [ndarray] 1D numpy array containing distortion corrected Y component

    @return (azimuth_data, altitude_data): [tuple of ndarrays]
        azimuth_data: [ndarray] 1D numpy array containing the azimuth of each data point (degrees)
        altitude_data: [ndarray] 1D numyp array containing the altitude of each data point (degrees)
    """

    # Initialize final values containers
    az_data = np.zeros_like(X_data, dtype=np.float64)
    alt_data = np.zeros_like(X_data, dtype=np.float64)

    # Convert declination to radians
    dec_rad = math.radians(dec_d)

    # Precalculate some parameters
    sl = math.sin(math.radians(lat))
    cl = math.cos(math.radians(lat))

    i = 0
    data_matrix = np.vstack((X_data, Y_data)).T

    # Go through all given data points
    for X_pix, Y_pix in data_matrix:

        # Caulucate the needed parameters
        radius = math.radians(np.sqrt(X_pix**2 + Y_pix**2))
        theta = math.radians((90 - rot_param + math.degrees(math.atan2(Y_pix, X_pix)))%360)

        sin_t = math.sin(dec_rad)*math.cos(radius) + math.cos(dec_rad)*math.sin(radius)*math.cos(theta)
        Dec0det = math.atan2(sin_t, math.sqrt(1 - sin_t**2))

        sin_t = math.sin(theta)*math.sin(radius)/math.cos(Dec0det)
        cos_t = (math.cos(radius) - math.sin(Dec0det)*math.sin(dec_rad))/(math.cos(Dec0det)*math.cos(dec_rad))
        RA0det = (RA_d - math.degrees(math.atan2(sin_t, cos_t)))%360

        h = math.radians(Ho + lon - RA0det)
        sh = math.sin(h)
        sd = math.sin(Dec0det)
        ch = math.cos(h)
        cd = math.cos(Dec0det)

        x = -ch*cd*sl + sd*cl
        y = -sh*cd
        z = ch*cd*cl + sd*sl

        r = math.sqrt(x**2 + y**2)

        # Calculate azimuth and altitude
        azimuth = math.degrees(math.atan2(y, x))%360
        altitude = math.degrees(math.atan2(z, r))

        # Save calculated values to an output array
        az_data[i] = azimuth
        alt_data[i] = altitude
        
        i += 1

    return az_data, alt_data



def altAz2RADec(lat, lon, UT_corr, time_data, azimuth_data, altitude_data, dt_time=False):
    """ Convert the azimuth and altitude in a given time and position on Earth to right ascension and 
        declination. 
    
    Arguments:
        lat: [float] latitude of the observer in degrees
        lon: [float] longitde of the observer in degress
        UT_corr: [float] UT correction in hours (difference from local time to UT)
        time_data: [2D ndarray] numpy array containing time tuples of each data point (year, month, day, 
            hour, minute, second, millisecond)
        azimuth_data: [ndarray] 1D numpy array containing the azimuth of each data point (degrees)
        altitude_data: [ndarray] 1D numpy array containing the altitude of each data point (degrees)

    Keyword arguments:
        dt_time: [bool] If True, datetime objects can be passed for time_data.

    Return: 
        (JD_data, RA_data, dec_data): [tuple of ndarrays]
            JD_data: [ndarray] julian date of each data point
            RA_data: [ndarray] right ascension of each point
            dec_data: [ndarray] declination of each point
    """

    # Initialize final values containers
    JD_data = np.zeros_like(azimuth_data, dtype=np.float64)
    RA_data = np.zeros_like(azimuth_data, dtype=np.float64)
    dec_data = np.zeros_like(azimuth_data, dtype=np.float64)

    # Precalculate some parameters
    sl = math.sin(math.radians(lat))
    cl = math.cos(math.radians(lat))

    i = 0
    data_matrix = np.vstack((azimuth_data, altitude_data)).T

    # Go through all given data points
    for azimuth, altitude in data_matrix:

        if dt_time:
            JD = datetime2JD(time_data[i], UT_corr=UT_corr)

        else:
            # Extract time
            Y, M, D, h, m, s, ms = time_data[i]
            JD = date2JD(Y, M, D, h, m, s, ms, UT_corr=UT_corr)

        # Convert altitude and azimuth to radians
        az_rad = math.radians(azimuth)
        alt_rad = math.radians(altitude)

        saz = math.sin(az_rad)
        salt = math.sin(alt_rad)
        caz = math.cos(az_rad)
        calt = math.cos(alt_rad)

        x = -saz*calt
        y = -caz*sl*calt + salt*cl
        HA = math.degrees(math.atan2(x, y))

        # Calculate the referent hour angle
        
        T = (JD - 2451545.0)/36525.0
        Ho = (280.46061837 + 360.98564736629*(JD - 2451545.0) + 0.000387933*T**2 - T**3/38710000.0)%360

        RA = (Ho + lon - HA)%360
        dec = math.degrees(math.asin(sl*salt + cl*calt*caz))

        # Save calculated values to an output array
        JD_data[i] = JD
        RA_data[i] = RA
        dec_data[i] = dec

        i += 1

    return JD_data, RA_data, dec_data


def calculateMagnitudes(level_data, C2, m2):
    """ Calculate the magnitude of the data points with given magnitude calibration parameters. 

    @param level_data: [ndarray] levels of the meteor centroid (arbirtary units)
    @param C2: [float] magnitude calibration equation parameter (fitted intensity)
    @param m2: [float] magnitude calibration equation parameter (intercept)

    @return magnitude_data: [ndarray] array of meteor's lightcurve apparent magnitudes
    """

    magnitude_data = np.zeros_like(level_data, dtype=np.float64)

    # Go through all levels of a meteor
    for i, level in enumerate(level_data):

        # Save magnitude data to the output array
        magnitude_data[i] = -2.5*np.log10(level) + 2.5*np.log10(C2) + m2


    return magnitude_data



def calculateMagnitudes_old(level_data, ra_beg, ra_end, dec_beg, dec_end, duration, mag_0, mag_lev, 
    w_pix):
    """ OLD CMN FUNCTION, DO NOT USE, HERE FOR LEGACY!
    Calculate the magnitude of the data points with given magnitude calibration parameters. 

    @param level_data: [ndarray] levels of the meteor centroid (arbirtary units)
    @param ra_beg: [float] right ascension of the meteor's beginning (degrees)
    @param ra_end: [float] right ascension of the meteor's end (degrees)
    @param dec_beg: [float] declination of the meteor's beginning (degrees)
    @param dec_end: [float] declination of the meteor's end (degrees)
    @param duration: [float] duration of the meteor in seconds
    @param mag_0: [float] magnitude calibration equation parameter (slope)
    @param mag_lev: [float] magnitude calibration equation parameter (intercept)
    @param w_pix: [float] minimum angular velocity of which to apply magnitude correction (arcsec/sec)

    @return magnitude_data: [ndarray] array of meteor's lightcurve apparent magnitudes
    """

    magnitude_data = np.zeros_like(level_data, dtype=np.float64)

    # Convert RA and Dec to radians
    ra_beg, ra_end, dec_beg, dec_end = map(np.radians, (ra_beg, ra_end, dec_beg, dec_end))

    # Calculate the length of the meteor trail
    length = np.degrees(np.arccos(np.sin(dec_beg)*np.sin(dec_end) + 
        np.cos(dec_beg)*np.cos(dec_end)*np.cos(ra_beg - ra_end)))

    # Calculate the angular velocty
    angular_v = length / float(duration)

    # Go through all levels of a meteor
    for i, level in enumerate(level_data):

        # Calculate the non-angular-velocity-corrected magnitude
        if np.log10(level) <= 3.2:
            magnitude = mag_0*np.log10(level) + mag_lev
        else:
            magnitude = -20*np.log10(level) + 64.5


        # Not needed! There's a Bob Hawkes paper on this, don't mind the work of Bagrov
        # # Correct the magnitude for angular velocity
        # if angular_v > w_pix:
        #     magnitude = magnitude -2.5*np.log10(angular_v/w_pix)

        # Save magnitude data to the output array
        magnitude_data[i] = magnitude


    return magnitude_data



def XY2CorrectedRADec(time_data, X_data, Y_data, level_data, UT_corr, lat, lon, Ho, X_res, Y_res, RA_d, dec_d, 
    rot_param, F_scale, w_pix, mag_0, mag_lev, x_poly, y_poly):
    """ A function that does the complete calibration and coordinate transformations of a meteor detection.

    First, it applies field distortion and vignetting correction on the data, then converts the XY coordinates
    to altitude and azimuth. Then it converts the altitude and azimuth data to right ascension and 
    declination. The resulting coordinates are in J2000.0 epoch.

    @param time_data: [2D ndarray] numpy array containing time tuples of each data point (year, month, day, 
        hour, minute, second, millisecond)
    @param X_data: [ndarray] 1D numpy array containing the image X component
    @param Y_data: [ndarray] 1D numpy array containing the image Y component
    @param level_data: [ndarray] levels of the meteor centroid (arbirtary units)
    @param UT_corr: [float] UT correction in hours (difference from local time to UT)
    @param lat: [float] latitude of the observer in degrees
    @param lon: [float] longitde of the observer in degress
    @param Ho: [float] referent hour angle
    @param X_res: [int] camera X axis resolution (longer)
    @param Y_res: [int] camera Y axis resolution (shorter)
    @param RA_d: [float] right ascension of the image centre (degrees)
    @param dec_d: [float] declination of the image centre (degrees)
    @param rot_param: [float] field rotation parameter (degrees)
    @param F_scale: [float] sum of image scales per each image axis (arcsec per px)
    @param w_pix: [float] minimum angular velocity of which to apply magnitude correction (arcsec/sec)
    @param mag_0: [float] magnitude calibration equation parameter (slope)
    @param mag_lev: [float] magnitude calibration equation parameter (intercept)
    @param x_poly: [ndarray] 1D numpy array of 12 elements containing X axis polynomial parameters
    @param y_poly: [ndarray] 1D numpy array of 12 elements containing Y axis polynomial parameters

    @return (JD_data, RA_data, dec_data, magnitude_data): [tuple of ndarrays]
        JD_data: [ndarray] julian date of each data point
        RA_data: [ndarray] right ascension of each point
        dec_data: [ndarray] declination of each point
        magnitude_data: [ndarray] array of meteor's lightcurve apparent magnitudes
    """

    # Apply field correction
    X_corrected, Y_corrected, levels_corrected = applyFieldCorrection(x_poly, y_poly, X_res, Y_res, F_scale, 
        X_data, Y_data, level_data)

    # Convert XY image coordinates to azimuth and altitude
    az_data, alt_data = XY2altAz(lat, lon, RA_d, dec_d, Ho, rot_param, X_corrected, Y_corrected)

    # Convert azimuth and altitude data to right ascension and declination
    JD_data, RA_data, dec_data = altAz2RADec(lat, lon, UT_corr, time_data, az_data, alt_data)

    # Find the beginning and ending points of the meteor and its duration in seconds
    ra_beg, ra_end = RA_data[0], RA_data[-1]
    dec_beg, dec_end = dec_data[0], dec_data[-1]
    duration = (JD_data[-1] - JD_data[0])*86400

    # Calculate magnitudes
    magnitude_data = calculateMagnitudes(levels_corrected, mag_0, mag_lev)


    return JD_data, RA_data, dec_data, magnitude_data


# NOT TESTED!!!
def raDecToXY(RA_data, dec_data, RA_d, dec_d, jd, ref_jd, rot_param, F_scale):
    """ Convert RA, Dec to image coordinates. 

    Arguments:
        RA: [ndarray] Right ascension (degrees).
        dec: [ndarray] Declination (degrees).
        RA_d: [float] Right ascension of the FOV centre (degrees).
        dec_d: [float] Declination of the FOV centre (degrees).
        jd: [float] Julian date.
        ref_jd: [float] Referent Julian date from platepar.
        rot_param: [float] Rotation from the celestial meridial (degrees).
        F_scale: [float] Sum of image scales per each image axis (arcsec per px).
    
    Return:
        (x, y): [tuple of ndarrays] Image X and Y coordinates.
    """
    
    RA_data = np.copy(RA_data)
    dec_data = np.copy(dec_data)

    # Correct RA for the hour angle
    delta_RA_H = 24.0657098244*15*(jd - ref_jd)
    RA_data -= delta_RA_H
    RA_data = RA_data%360

    RA_data = np.radians(RA_data)
    dec_data = np.radians(dec_data)

    RA_d = np.radians(RA_d)
    dec_d = np.radians(dec_d)
    rot_param = np.radians(rot_param)

    x_array = []
    y_array = []

    for RA, dec in zip(RA_data, dec_data):
        ad = np.arccos(np.sin(dec_d)*np.sin(dec) + np.cos(dec_d)*np.cos(dec)*np.cos(RA - RA_d))

        radius = np.degrees(ad)

        sinA = np.cos(dec)*np.sin(RA - RA_d)/np.sin(ad)

        cosA = (np.sin(dec) - np.sin(dec_d)*np.cos(ad))/(np.cos(dec_d)*np.sin(ad))

        theta = -np.arctan2(sinA, cosA)

        theta = theta + rot_param - np.pi/2

        x = radius*np.cos(theta)*F_scale
        y = radius*np.sin(theta)*F_scale

        x_array.append(x)
        y_array.append(y)

    x_array = np.array(x_array)
    y_array = np.array(y_array)

    return x_array, y_array



def applyAstrometryFTPdetectinfo(dir_path, ftp_detectinfo_file, platepar_file, UT_corr=0):
    """ Use the given platepar to calculate the celestial coordinates of detected meteors from a FTPdetectinfo
        file and save the updates values.

    Arguments:
        dir_path: [str] Path to the night.
        ftp_detectinfo_file: [str] Name of the FTPdetectinfo file.
        platepar_file: [str] Name of the platepar file.

    Keyword arguments:
        UT_corr: [float] Difference of time from UTC in hours.

    Return:
        None
    """

    # Save a copy of the uncalibrated FTPdetectinfo
    ftp_detectinfo_copy = "".join(ftp_detectinfo_file.split('.')[:-1]) + "_uncalibrated.txt"
    shutil.copy2(os.path.join(dir_path, ftp_detectinfo_file), os.path.join(dir_path, ftp_detectinfo_copy))

    # Load the platepar
    platepar = PlateparCMN()
    platepar.read(os.path.join(dir_path, platepar_file))

    # Load the FTPdetectinfo file
    meteor_data = readFTPdetectinfo(dir_path, ftp_detectinfo_file)

    # List for final meteor data
    meteor_list = []

    # Go through every meteor
    for meteor in meteor_data:

        ff_name, cam_code, meteor_No, n_segments, fps, hnr, mle, binn, px_fm, rho, phi, meteor_meas = meteor

        meteor_meas = np.array(meteor_meas)

        # Extract frame number, x, y, intensity
        frames = meteor_meas[:, 0]
        X_data = meteor_meas[:, 1]
        Y_data = meteor_meas[:, 2]
        level_data = meteor_meas[:, 7]

        # Get the beginning time of the FF file
        time_beg = filenameToDatetimeFF(ff_name)

        # Calculate time data of every point
        time_data = []
        for frame_n in frames:
            t = time_beg + datetime.timedelta(seconds=frame_n/fps)
            time_data.append([t.year, t.month, t.day, t.hour, t.minute, t.second, int(t.microsecond/1000)])

        # Apply field correction
        X_corrected, Y_corrected, levels_corrected = applyFieldCorrection(platepar.x_poly, platepar.y_poly, \
            platepar.X_res, platepar.Y_res, platepar.F_scale, X_data, Y_data, level_data)

        # Convert XY image coordinates to azimuth and altitude
        az_data, alt_data = XY2altAz(platepar.lat, platepar.lon, platepar.RA_d, platepar.dec_d, platepar.Ho, \
            platepar.pos_angle_ref, X_corrected, Y_corrected)

        # Convert azimuth and altitude data to right ascension and declination
        JD_data, RA_data, dec_data = altAz2RADec(platepar.lat, platepar.lon, UT_corr, time_data, az_data, 
            alt_data)

        # Construct the meteor measurements array
        meteor_picks = np.c_[frames, X_data, Y_data, RA_data, dec_data, az_data, alt_data, levels_corrected]

        # Add the calculated values to the final list
        meteor_list.append([ff_name, meteor_No, rho, phi, meteor_picks])


    # Calibration string to be written to the FTPdetectinfo file
    calib_str = 'Calibrated with RMS on: ' + str(datetime.datetime.now())

    # Save the updated FTPdetectinfo
    writeFTPdetectinfo(meteor_list, dir_path, ftp_detectinfo_file, dir_path, cam_code, fps, 
        calibration=calib_str, celestial_coords_given=True)





if __name__ == "__main__":

    # Read the path to the FTPdetectinfo file
    if not len(sys.argv) == 2:
        print("Usage: python -m RMS.Astrometry.ApplyAstrometry /path/to/FTPdetectinfo.txt")
        sys.exit()


    # Extract the directory path
    dir_path, ftp_detectinfo_file = os.path.split(os.path.abspath(sys.argv[1]))

    # Find the platepar file
    platepar_file = None
    for file_name in os.listdir(dir_path):
        if 'platepar' in file_name:
            platepar_file = file_name
            break

    if platepar_file is None:
        print('ERROR! Could not find the platepar file!')
        sys.exit()


    # Apply the astrometry to the given FTPdetectinfo file
    applyAstrometryFTPdetectinfo(dir_path, ftp_detectinfo_file, platepar_file)


    # # Load the platepar
    # platepar = PlateparCMN()
    # platepar.read("C:\\Users\\delorayn1\\Desktop\\20170813_213506_620678\\platepar_cmn2010.cal")

    # from RMS.Astrometry.AstrometryCheckFit import getMiddleTimeFF
    # time = getMiddleTimeFF('FF100_20170813_213508_532_0000000.bin', 25)

    # # Deneb
    # star_x = 281.14
    # star_y = 57.97

    # # # Vega
    # # star_x = 546.27
    # # star_y = 54.74

    # jd, ra, dec, mag = XY2CorrectedRADec([time], [star_x], [star_y], [0], 0, platepar.lat, platepar.lon, platepar.Ho, platepar.X_res, platepar.Y_res, platepar.RA_d, platepar.dec_d, platepar.pos_angle_ref, platepar.F_scale, platepar.w_pix, platepar.mag_0, platepar.mag_lev, platepar.x_poly, platepar.y_poly)

    # ra_h = int(ra/15)
    # ra_min = int((ra/15 - ra_h)*60)
    # ra_sec = ((ra/15 - ra_h)*60 - ra_min)*60

    # dec_d = int(dec)
    # dec_min = int((dec - dec_d)*60)
    # dec_sec = ((dec - dec_d)*60 - dec_min)*60


    # print(ra_h, ra_min, ra_sec)
    # print(dec_d, dec_min, dec_sec)



    # # Load the platepar
    # platepar = PlateparCMN()
    # platepar.read("C:\\Users\\delorayn1\\Desktop\\2017071819-Processed\\platepar_cmn2010.cal")

    # # X_scale = platepar.X_res/384
    # # Y_scale = platepar.Y_res/288

    # # x = (617.71 - platepar.X_res/2)/X_scale/platepar.F_scale
    # # y = (139.99 - platepar.Y_res/2)/Y_scale/platepar.F_scale

    # x = 617.71
    # y = 139.99

    # secs = 28.788 + 164.0/25
    # time_data = [[2017, 7, 18, 20, 46, int(secs), int(1000*(secs - int(secs)))]]

    # X_corrected, Y_corrected, levels_corrected = applyFieldCorrection(platepar.x_poly, platepar.y_poly, platepar.X_res, platepar.Y_res, platepar.F_scale, [x], [y], [0])

    # az_data, alt_data = XY2altAz(platepar.lat, platepar.lon, platepar.RA_d, platepar.dec_d, platepar.Ho, platepar.pos_angle_ref, X_corrected, Y_corrected)

    # # Convert azimuth and altitude data to right ascension and declination
    # JD_data, RA_data, dec_data = altAz2RADec(platepar.lat, platepar.lon, 0, time_data, az_data, alt_data)

    # print(az_data, alt_data)
    # print(RA_data, dec_data)