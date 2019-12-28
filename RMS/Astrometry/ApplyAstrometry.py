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
import copy
import argparse

import numpy as np
import scipy.optimize

from RMS.Astrometry.Conversions import date2JD, datetime2JD, jd2Date
from RMS.Astrometry.AtmosphericExtinction import atmosphericExtinctionCorrection
import RMS.Formats.Platepar
from RMS.Formats.FTPdetectinfo import readFTPdetectinfo, writeFTPdetectinfo
from RMS.Formats.FFfile import filenameToDatetime
from RMS.Math import angularSeparation
import Utils.RMS2UFO

# Import Cython functions
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from RMS.Astrometry.CyFunctions import cyraDecToXY, cyXYToRADec



def correctVignetting(px_sum, radius, vignetting_coeff):
    """ Given a pixel sum, radius from focal plane centre and the vignetting coefficient, correct the pixel
        sum for the vignetting effect.

    Arguments:
        px_sum: [float] Pixel sum.
        radius: [float] Radius (px) from focal plane centre.
        vignetting_coeff: [float] Vignetting ceofficient (deg/px).

    Return:
        px_sum_corr: [float] Corrected pixel sum.
    """

    # Make sure the vignetting coefficient is a number
    if vignetting_coeff is None:
        vignetting_coeff = 0.0

    return px_sum/(np.cos(vignetting_coeff*radius)**4)



def photomLine(input_params, photom_offset, vignetting_coeff):
    """ Line used for photometry, the slope is fixed to -2.5, only the photometric offset is given. 
    
    Arguments:
        input_params: [tuple]
            - px_sum: [float] sum of pixel intensities.
            - radius: [float] Radius from the centre of the focal plane to the centroid.
        photom_offset: [float] The photometric offet.
        vignetting_coeff: [float] Vignetting coefficient.

    Return: 
        [float] Magnitude.
    """

    px_sum, radius = input_params

    # Apply the vignetting correction and compute the LSP
    lsp = np.log10(correctVignetting(px_sum, radius, vignetting_coeff))
    
    # The slope is fixed to -2.5, this comes from the definition of magnitude
    return -2.5*lsp + photom_offset



def photomLineMinimize(params, px_sum, radius, catalog_mags, fixed_vignetting):
    """ Modified photomLine function used for minimization. The function uses the L1 norm for minimization. 
    """

    photom_offset, vignetting_coeff = params

    if fixed_vignetting is not None:
        vignetting_coeff = fixed_vignetting

    # Compute the sum of squred residuals
    return np.sum(np.abs(catalog_mags - photomLine((px_sum, radius), photom_offset, vignetting_coeff)))



def photometryFit(px_intens_list, radius_list, catalog_mags, fixed_vignetting=None):
    """ Fit the photometry on given data. 
    
    Arguments:
        px_intens_list: [list] A list of sums of pixel intensities.
        radius_list: [list] A list of raddia from the focal plane centre (px).
        catalog_mags: [list] A list of corresponding catalog magnitudes of stars.

    Keyword arguments:
        fixed_vignetting: [float] Fixed vignetting coefficient. None by default, in which case it will be
            computed.

    Return:
        (photom_offset, fit_stddev, fit_resid):
            photom_params: [list]
                - photom_offset: [float] The photometric offset.
                - vignetting_coeff: [float] Vignetting coefficient.
            fit_stddev: [float] The standard deviation of the fit.
            fit_resid: [float] Magnitude fit residuals.
    """

    # Fit a line to the star data, where only the intercept has to be estimated
    p0 = [10.0, 0.0]
    res = scipy.optimize.minimize(photomLineMinimize, p0, args=(np.array(px_intens_list), \
        np.array(radius_list), np.array(catalog_mags), fixed_vignetting), method='Nelder-Mead')
    photom_offset, vignetting_coeff = res.x

    # Handle the vignetting coeff
    vignetting_coeff = np.abs(vignetting_coeff)
    if fixed_vignetting is not None:
        vignetting_coeff = fixed_vignetting

    photom_params = (photom_offset, vignetting_coeff)

    # Calculate the standard deviation
    fit_resids = np.array(catalog_mags) - photomLine((np.array(px_intens_list), np.array(radius_list)), \
        *photom_params)
    fit_stddev = np.std(fit_resids)

    return photom_params, fit_stddev, fit_resids



def computeFOVSize(platepar):
    """ Computes the size of the FOV in deg from the given platepar. 
        
    Arguments:
        platepar: [Platepar instance]

    Return:
        fov_h: [float] Horizontal FOV in degrees.
        fov_v: [float] Vertical FOV in degrees.
    """

    # Construct poinits on the middle of every side of the image
    time_data = np.array(4*[jd2Date(platepar.JD)])
    x_data = np.array([0, platepar.X_res, platepar.X_res/2, platepar.X_res/2])
    y_data = np.array([platepar.Y_res/2, platepar.Y_res/2, 0, platepar.Y_res])
    level_data = np.ones(4)

    # Compute RA/Dec of the points
    _, ra_data, dec_data, _ = xyToRaDecPP(time_data, x_data, y_data, level_data, platepar)

    ra1, ra2, ra3, ra4 = ra_data
    dec1, dec2, dec3, dec4 = dec_data

    # Compute horizontal FOV
    fov_h = np.degrees(angularSeparation(np.radians(ra1), np.radians(dec1), np.radians(ra2), \
        np.radians(dec2)))

    # Compute vertical FOV
    fov_v = np.degrees(angularSeparation(np.radians(ra3), np.radians(dec3), np.radians(ra4), \
        np.radians(dec4)))


    return fov_h, fov_v




def rotationWrtHorizon(platepar):
    """ Given the platepar, compute the rotation of the FOV with respect to the horizon. 
    
    Arguments:
        pletepar: [Platepar object] Input platepar.

    Return:
        rot_angle: [float] Rotation w.r.t. horizon (degrees).
    """

    # Image coordiantes of the center
    img_mid_w = platepar.X_res/2
    img_mid_h = platepar.Y_res/2

    # Image coordinate slighty right of the center (horizontal)
    img_up_w = img_mid_w + 10
    img_up_h = img_mid_h

    # Compute alt/az
    azim, alt = XY2altAz([img_mid_w, img_up_w], [img_mid_h, img_up_h], platepar.lat, platepar.lon, \
        platepar.RA_d, platepar.dec_d, platepar.Ho, platepar.X_res, platepar.Y_res, platepar.pos_angle_ref, \
        platepar.F_scale, platepar.x_poly_fwd, platepar.y_poly_fwd)
    azim_mid = azim[0]
    alt_mid = alt[0]
    azim_up = azim[1]
    alt_up = alt[1]

    # Compute the rotation wrt horizon (deg)    
    rot_angle = np.degrees(np.arctan2(np.radians(alt_up) - np.radians(alt_mid), \
        np.radians(azim_up) - np.radians(azim_mid)))

    # Wrap output to <-180, 180] range
    if rot_angle > 180:
        rot_angle -= 360

    return rot_angle



def rotationWrtHorizonToPosAngle(platepar, rot_angle):
    """ Given the rotation angle w.r.t horizon, numerically compute the position angle. 
    
    Arguments:
        pletepar: [Platepar object] Input platepar.
        rot_angle: [float] The rotation angle w.r.t. horizon (deg)>

    Return:
        pos_angle: [float] Position angle (deg).

    """

    platepar = copy.deepcopy(platepar)
    rot_angle = rot_angle%360


    def _rotAngleResidual(params, rot_angle):

        # Set the given position angle to the platepar
        platepar.pos_angle_ref = params[0]

        # Compute the rotation angle with the given guess of the position angle
        rot_angle_computed = rotationWrtHorizon(platepar)%360

        # Compute the deviation between computed and desired angle
        return 180 - abs(abs(rot_angle - rot_angle_computed) - 180)



    # Numerically find the position angle
    res = scipy.optimize.minimize(_rotAngleResidual, [platepar.pos_angle_ref], args=(rot_angle), \
        method='Nelder-Mead')


    return res.x[0]%360




def rotationWrtStandard(platepar):
    """ Given the platepar, compute the rotation from the celestial meridian passing through the centre of 
        the FOV.
    
    Arguments:
        pletepar: [Platepar object] Input platepar.

    Return:
        rot_angle: [float] Rotation from the meridian (degrees).
    """

    # Image coordiantes of the center
    img_mid_w = platepar.X_res/2
    img_mid_h = platepar.Y_res/2

    # Image coordinate slighty right of the centre
    img_up_w = img_mid_w + 10
    img_up_h = img_mid_h

    # Compute ra/dec
    _, ra, dec, _ = xyToRaDecPP(2*[jd2Date(platepar.JD)], [img_mid_w, img_up_w], [img_mid_h, img_up_h], \
        2*[1], platepar)
    ra_mid = ra[0]
    dec_mid = dec[0]
    ra_up = ra[1]
    dec_up = dec[1]

    # Compute the equatorial orientation
    rot_angle = np.degrees(np.arctan2(np.radians(dec_mid) - np.radians(dec_up), \
        np.radians(ra_mid) - np.radians(ra_up)))

    # Wrap output to 0-360 range
    rot_angle = rot_angle%360

    return rot_angle




def rotationWrtStandardToPosAngle(platepar, rot_angle):
    """ Given the rotation angle w.r.t horizon, numerically compute the position angle. 
    
    Arguments:
        pletepar: [Platepar object] Input platepar.
        rot_angle: [float] The rotation angle w.r.t. horizon (deg)>

    Return:
        pos_angle: [float] Position angle (deg).

    """

    platepar = copy.deepcopy(platepar)
    rot_angle = rot_angle%360


    def _rotAngleResidual(params, rot_angle):

        # Set the given position angle to the platepar
        platepar.pos_angle_ref = params[0]

        # Compute the rotation angle with the given guess of the position angle
        rot_angle_computed = rotationWrtStandard(platepar)%360

        # Compute the deviation between computed and desired angle
        return 180 - abs(abs(rot_angle - rot_angle_computed) - 180)



    # Numerically find the position angle
    res = scipy.optimize.minimize(_rotAngleResidual, [platepar.pos_angle_ref], args=(rot_angle), \
        method='Nelder-Mead')


    return res.x[0]%360



def raDec2AltAz(JD, lon, lat, ra, dec):
    """ Calculate the reference azimuth and altitude of the centre of the FOV from the given RA/Dec. 

    Arguments:
        JD: [float] Reference Julian date.
        lon: [float] Longitude +E in degrees.
        lat: [float] Latitude +N in degrees.
        ra_: [float] Right ascension in degrees.
        dec: [float] Declination in degrees.

    Return:
        (azim, elev): [tuple of float]: Azimuth and elevation (degrees).
    """

    # Compute the LST (local sidereal time)
    T = (JD - 2451545)/36525.0
    lst = (280.46061837 + 360.98564736629*(JD - 2451545.0) + 0.000387933*T**2 - (T**3)/38710000)%360
    lst = lst + lon

    # Convert all values to radians
    lst = np.radians(lst)
    lat = np.radians(lat)    
    ra = np.radians(ra)
    dec = np.radians(dec)

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
    

    # Convert alt/az to degrees
    azim = np.degrees(azim)
    elev = np.degrees(elev)

    return azim, elev




def applyFieldCorrection(x_poly_fwd, y_poly_fwd, X_res, Y_res, F_scale, X_data, Y_data):
    """ Apply field correction and vignetting correction to all given image points. 
`
    Arguments:
        x_poly_fwd: [ndarray] 1D numpy array of 12 elements containing forward X axis polynomial parameters.
        y_poly_fwd: [ndarray] 1D numpy array of 12 elements containing forward Y axis polynomial parameters.
        X_res: [int] Image size, X dimension (px).
        Y_res: [int] Image size, Y dimenstion (px).
        F_scale: [float] Sum of image scales per each image axis (px/deg).
        X_data: [ndarray] 1D float numpy array containing X component of the detection point.
        Y_data: [ndarray] 1D float numpy array containing Y component of the detection point.
    
    Return:
        (X_corrected, Y_corrected, levels_corrected): [tuple of ndarrays]
            X_corrected: 1D numpy array containing distortion corrected X component.
            Y_corrected: 1D numpy array containing distortion corrected Y component.
            
    """

    # Initialize final values containers
    X_corrected = np.zeros_like(X_data, dtype=np.float64)
    Y_corrected = np.zeros_like(Y_data, dtype=np.float64)

    i = 0

    data_matrix = np.vstack((X_data, Y_data)).T

    # Go through all given data points
    for Xdet, Ydet in data_matrix:

        Xdet = Xdet - X_res/2.0
        Ydet = Ydet - Y_res/2.0

        dX = (x_poly_fwd[0]
            + x_poly_fwd[1]*Xdet
            + x_poly_fwd[2]*Ydet
            + x_poly_fwd[3]*Xdet**2
            + x_poly_fwd[4]*Xdet*Ydet
            + x_poly_fwd[5]*Ydet**2
            + x_poly_fwd[6]*Xdet**3
            + x_poly_fwd[7]*Xdet**2*Ydet
            + x_poly_fwd[8]*Xdet*Ydet**2
            + x_poly_fwd[9]*Ydet**3
            + x_poly_fwd[10]*Xdet*np.sqrt(Xdet**2 + Ydet**2)
            + x_poly_fwd[11]*Ydet*np.sqrt(Xdet**2 + Ydet**2))

        # Add the distortion correction
        X_pix = Xdet + dX

        dY = (y_poly_fwd[0]
            + y_poly_fwd[1]*Xdet
            + y_poly_fwd[2]*Ydet
            + y_poly_fwd[3]*Xdet**2
            + y_poly_fwd[4]*Xdet*Ydet
            + y_poly_fwd[5]*Ydet**2
            + y_poly_fwd[6]*Xdet**3
            + y_poly_fwd[7]*Xdet**2*Ydet
            + y_poly_fwd[8]*Xdet*Ydet**2
            + y_poly_fwd[9]*Ydet**3
            + y_poly_fwd[10]*Ydet*np.sqrt(Xdet**2 + Ydet**2)
            + y_poly_fwd[11]*Xdet*np.sqrt(Xdet**2 + Ydet**2))

        # Add the distortion correction
        Y_pix = Ydet + dY

        # Scale back image coordinates
        X_pix = X_pix/F_scale
        Y_pix = Y_pix/F_scale

        # Store values to final arrays
        X_corrected[i] = X_pix
        Y_corrected[i] = Y_pix

        i += 1

    return X_corrected, Y_corrected



def XY2altAz(X_data, Y_data, lat, lon, RA_d, dec_d, Ho, X_res, Y_res, pos_angle_ref, F_scale, x_poly_fwd, \
    y_poly_fwd):
    """ Convert image coordinates (X, Y) to celestial altitude and azimuth. 
    
    Arguments:
        X_data: [ndarray] 1D numpy array containing the image pixel column.
        Y_data: [ndarray] 1D numpy array containing the image pixel row.
        lat: [float] Latitude of the observer +N (degrees).
        lon: [float] Longitde of the observer +E (degress).
        RA_d: [float] Reference right ascension of the image centre (degrees).
        dec_d: [float] Reference declination of the image centre (degrees).
        Ho: [float] Reference hour angle.
        X_res: [int] Image size, X dimension (px).
        Y_res: [int] Image size, Y dimenstion (px).
        pos_angle_ref: [float] Field rotation parameter (degrees).
        F_scale: [float] Sum of image scales per each image axis (px/deg).
        x_poly_fwd: [ndarray] 1D numpy array of 12 elements containing forward X axis polynomial parameters.
        y_poly_fwd: [ndarray] 1D numpy array of 12 elements containing forward Y axis polynomial parameters.
        
    
    Return:
        (azimuth_data, altitude_data): [tuple of ndarrays]
            azimuth_data: [ndarray] 1D numpy array containing the azimuth of each data point (degrees).
            altitude_data: [ndarray] 1D numyp array containing the altitude of each data point (degrees).
    """


    # Apply distorsion correction
    X_corrected, Y_corrected = applyFieldCorrection(x_poly_fwd, y_poly_fwd, X_res, Y_res, F_scale, X_data, \
        Y_data)

    # Initialize final values containers
    az_data = np.zeros_like(X_corrected, dtype=np.float64)
    alt_data = np.zeros_like(X_corrected, dtype=np.float64)

    # Convert declination to radians
    dec_rad = math.radians(dec_d)

    # Precalculate some parameters
    sl = math.sin(math.radians(lat))
    cl = math.cos(math.radians(lat))

    i = 0
    data_matrix = np.vstack((X_corrected, Y_corrected)).T

    # Go through all given data points
    for X_pix, Y_pix in data_matrix:

        # Caulucate the needed parameters
        radius = math.radians(np.sqrt(X_pix**2 + Y_pix**2))
        theta = math.radians((90 - pos_angle_ref + math.degrees(math.atan2(Y_pix, X_pix)))%360)

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



def altAzToRADec(lat, lon, UT_corr, time_data, azimuth_data, altitude_data, dt_time=False):
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
            JD = datetime2JD(time_data[i], UT_corr=-UT_corr)

        else:
            # Extract time
            Y, M, D, h, m, s, ms = time_data[i]
            JD = date2JD(Y, M, D, h, m, s, ms, UT_corr=-UT_corr)

        # Never allow the altitude to be exactly 90 deg due to numerical issues
        if altitude == 90:
            altitude = 89.9999

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

        # Calculate the reference hour angle
        
        T = (JD - 2451545.0)/36525.0
        hour_angle = (280.46061837 + 360.98564736629*(JD - 2451545.0) + 0.000387933*T**2 - T**3/38710000.0)%360

        RA = (hour_angle + lon - HA)%360
        dec = math.degrees(math.asin(sl*salt + cl*calt*caz))

        # Save calculated values to an output array
        JD_data[i] = JD
        RA_data[i] = RA
        dec_data[i] = dec

        i += 1

    return JD_data, RA_data, dec_data



def calculateMagnitudes(px_sum_arr, radius_arr, photom_offset, vignetting_coeff):
    """ Calculate the magnitude of the data points with given magnitude calibration parameters. 
    
    Arguments:
        px_sum_arr: [ndarray] Sum of pixel intensities of the meteor centroid (arbitrary units).
        radius_arr: [ndarray] A list of raddia from image centre (px).
        photom_offset: [float] Magnitude intercept, i.e. the photometric offset.
        vignetting_coeff: [float] Vignetting ceofficient (deg/px).


    Return:
        magnitude_data: [ndarray] Apparent magnitude.
    """

    magnitude_data = np.zeros_like(px_sum_arr, dtype=np.float64)

    # Go through all levels of a meteor
    for i, (px_sum, radius) in enumerate(zip(px_sum_arr, radius_arr)):

        # Correct vignetting
        px_sum_corr = correctVignetting(px_sum, radius, vignetting_coeff)

        # Save magnitude data to the output array
        magnitude_data[i] = -2.5*np.log10(px_sum_corr) + photom_offset


    return magnitude_data



def xyToRaDec(time_data, X_data, Y_data, level_data, lat, lon, Ho, X_res, Y_res, RA_d, dec_d, \
    pos_angle_ref, F_scale, mag_lev, vignetting_coeff, x_poly_fwd, y_poly_fwd, station_ht):
    """ A function that does the complete calibration and coordinate transformations of a meteor detection.

    First, it applies field distortion on the data, then converts the XY coordinates
    to altitude and azimuth. Then it converts the altitude and azimuth data to right ascension and 
    declination. The resulting coordinates are in J2000.0 epoch.
    
    Arguments:
        time_data: [2D ndarray] Numpy array containing time tuples of each data point (year, month, day, 
            hour, minute, second, millisecond).
        X_data: [ndarray] 1D numpy array containing the image X component.
        Y_data: [ndarray] 1D numpy array containing the image Y component.
        level_data: [ndarray] Levels of the meteor centroid.
        lat: [float] Latitude of the observer in degrees.
        lon: [float] Longitde of the observer in degress.
        Ho: [float] Reference hour angle (deg).
        X_res: [int] Image size, X dimension (px).
        Y_res: [int] Image size, Y dimenstion (px).
        RA_d: [float] Reference right ascension of the image centre (degrees).
        dec_d: [float] Reference declination of the image centre (degrees).
        pos_angle_ref: [float] Field rotation parameter (degrees).
        F_scale: [float] Image scale (px/deg).
        mag_lev: [float] Magnitude calibration equation parameter (intercept).
        vignetting_coeff: [float] Vignetting ceofficient (deg/px).
        x_poly_fwd: [ndarray] 1D numpy array of 12 elements containing forward X axis polynomial parameters.
        y_poly_fwd: [ndarray] 1D numpy array of 12 elements containing forward Y axis polynomial parameters.
        station_ht: [float] Height above sea level of the station (m).
    
    Return:
        (JD_data, RA_data, dec_data, magnitude_data): [tuple of ndarrays]
            JD_data: [ndarray] Julian date of each data point.
            RA_data: [ndarray] Right ascension of each point (deg).
            dec_data: [ndarray] Declination of each point (deg).
            magnitude_data: [ndarray] Array of meteor's lightcurve apparent magnitudes.

    """


    # Convert time to Julian date
    JD_data = np.array([date2JD(*time_data_entry) for time_data_entry in time_data])

    # Convert x,y to RA/Dec using a fast cython function
    RA_data, dec_data = cyXYToRADec(JD_data, np.array(X_data), np.array(Y_data), float(lat), float(lon), \
        float(Ho), float(X_res), float(Y_res), float(RA_d), float(dec_d), float(pos_angle_ref), \
        float(F_scale), x_poly_fwd, y_poly_fwd)

    # Compute radiia from image centre
    radius_arr = np.hypot(np.array(X_data) - X_res/2, np.array(Y_data) - Y_res/2)

    # Calculate magnitudes
    magnitude_data = calculateMagnitudes(level_data, radius_arr, mag_lev, vignetting_coeff)

    # CURRENTLY DISABLED!
    # Compute the apparent magnitudes corrected to relative atmospheric extinction
    # magnitude_data -= atmosphericExtinctionCorrection(alt_data, station_ht) \
    #   - atmosphericExtinctionCorrection(90, station_ht)

    
    return JD_data, RA_data, dec_data, magnitude_data
 


def xyToRaDecPP(time_data, X_data, Y_data, level_data, platepar):
    """ Converts image XY to RA,Dec, but it takes a platepar instead of individual parameters. 
    
    Arguments:
        time_data: [2D ndarray] Numpy array containing time tuples of each data point (year, month, day, 
            hour, minute, second, millisecond).
        X_data: [ndarray] 1D numpy array containing the image X component.
        Y_data: [ndarray] 1D numpy array containing the image Y component.
        level_data: [ndarray] Levels of the meteor centroid.
        platepar: [Platepar structure] Astrometry parameters.


    Return:
        (JD_data, RA_data, dec_data, magnitude_data): [tuple of ndarrays]
            JD_data: [ndarray] Julian date of each data point.
            RA_data: [ndarray] Right ascension of each point (deg).
            dec_data: [ndarray] Declination of each point (deg).
            magnitude_data: [ndarray] Array of meteor's lightcurve apparent magnitudes.
    """


    return xyToRaDec(time_data, X_data, Y_data, level_data, platepar.lat, \
        platepar.lon, platepar.Ho, platepar.X_res, platepar.Y_res, platepar.RA_d, platepar.dec_d, \
        platepar.pos_angle_ref, platepar.F_scale, platepar.mag_lev, platepar.vignetting_coeff, \
        platepar.x_poly_fwd, platepar.y_poly_fwd, platepar.elev)




def raDecToXY(RA_data, dec_data, jd, lat, lon, x_res, y_res, RA_d, dec_d, ref_jd, pos_angle_ref, \
    F_scale, x_poly_rev, y_poly_rev, UT_corr=0):
    """ Convert RA, Dec to distorion corrected image coordinates. 

    Arguments:
        RA: [ndarray] Array of right ascensions (degrees).
        dec: [ndarray] Array of declinations (degrees).
        jd: [float] Julian date.
        lat: [float] Latitude of station in degrees.
        lon: [float] Longitude of station in degrees.
        x_res: [int] X resolution of the camera.
        y_res: [int] Y resolution of the camera.
        RA_d: [float] Right ascension of the FOV centre (degrees).
        dec_d: [float] Declination of the FOV centre (degrees).
        ref_jd: [float] Reference Julian date from platepar.
        pos_angle_ref: [float] Rotation from the celestial meridial (degrees).
        F_scale: [float] Image scale (px/deg).
        x_poly_rev: [ndarray float] Distorsion polynomial in X direction for reverse mapping.
        y_poly_rev: [ndarray float] Distorsion polynomail in Y direction for reverse mapping.

    Keyword arguments:
        UT_corr: [float] UT correction (hours).
    
    Return:
        (x, y): [tuple of ndarrays] Image X and Y coordinates.
    """
    
    # Calculate the azimuth and altitude of the FOV centre
    az_centre, alt_centre = raDec2AltAz(ref_jd, lon, lat, RA_d, dec_d)

    # Apply the UT correction
    jd -= UT_corr/24.0

    # Use the cythonized funtion insted of the Python function
    return cyraDecToXY(RA_data, dec_data, jd, lat, lon, x_res, y_res, az_centre, alt_centre, 
        pos_angle_ref, F_scale, x_poly_rev, y_poly_rev)



def raDecToXYPP(RA_data, dec_data, jd, platepar):
    """ Converts RA, Dec to image coordinates, but the platepar is given instead of individual parameters.

    Arguments:
        RA: [ndarray] Array of right ascensions (degrees).
        dec: [ndarray] Array of declinations (degrees).
        jd: [float] Julian date.
        platepar: [Platepar structure] Astrometry parameters.

    Return:
        (x, y): [tuple of ndarrays] Image X and Y coordinates.

    """

    return raDecToXY(RA_data, dec_data, jd, platepar.lat, platepar.lon, platepar.X_res, \
        platepar.Y_res, platepar.RA_d, platepar.dec_d, platepar.JD, platepar.pos_angle_ref, \
        platepar.F_scale, platepar.x_poly_rev, platepar.y_poly_rev, UT_corr=platepar.UT_corr)




def applyPlateparToCentroids(ff_name, fps, meteor_meas, platepar, add_calstatus=False):
    """ Given the meteor centroids and a platepar file, compute meteor astrometry and photometry (RA/Dec, 
        alt/az, mag).

    Arguments:
        ff_name: [str] Name of the FF file with the meteor.
        fps: [float] Frames per second of the video.
        meteor_meas: [list] A list of [calib_status, frame_n, x, y, ra, dec, azim, elev, inten, mag].
        platepar: [Platepar instance] Platepar which will be used for astrometry and photometry.

    Keyword arguments:
        add_calstatus: [bool] Add a column with calibration status at the beginning. False by default.

    Return:
        meteor_picks: [ndarray] A numpy 2D array of: [frames, X_data, Y_data, RA_data, dec_data, az_data, 
        alt_data, level_data, magnitudes]

    """


    meteor_meas = np.array(meteor_meas)

    # Add a line which is indicating the calibration status
    if add_calstatus:
        meteor_meas = np.c_[np.ones((meteor_meas.shape[0], 1)), meteor_meas]


    # Remove all entries where levels are equal to or smaller than 0, unless all are zero
    level_data = meteor_meas[:, 8]
    if np.any(level_data):
        meteor_meas = meteor_meas[level_data > 0, :]

    # Extract frame number, x, y, intensity
    frames = meteor_meas[:, 1]
    X_data = meteor_meas[:, 2]
    Y_data = meteor_meas[:, 3]
    level_data = meteor_meas[:, 8]

    # Get the beginning time of the FF file
    time_beg = filenameToDatetime(ff_name)

    # Calculate time data of every point
    time_data = []
    for frame_n in frames:
        t = time_beg + datetime.timedelta(seconds=frame_n/fps)
        time_data.append([t.year, t.month, t.day, t.hour, t.minute, t.second, int(t.microsecond/1000)])



    # Convert image cooredinates to RA and Dec, and do the photometry
    JD_data, RA_data, dec_data, magnitudes = xyToRaDecPP(np.array(time_data), X_data, Y_data, \
        level_data, platepar)


    # Compute azimuth and altitude of centroids
    az_data = np.zeros_like(RA_data)
    alt_data = np.zeros_like(RA_data)

    for i in range(len(az_data)):

        jd = JD_data[i]
        ra_tmp = RA_data[i]
        dec_tmp = dec_data[i]

        # Alt and az are kept in the J2000 epoch, which is the CAMS standard!
        az_tmp, alt_tmp = raDec2AltAz(jd, platepar.lon, platepar.lat, ra_tmp, dec_tmp)

        az_data[i] = az_tmp
        alt_data[i] = alt_tmp


    # print(ff_name, cam_code, meteor_No, fps)
    # print(X_data, Y_data)
    # print(RA_data, dec_data)
    # print('------------------------------------------')

    # Construct the meteor measurements array
    meteor_picks = np.c_[frames, X_data, Y_data, RA_data, dec_data, az_data, alt_data, level_data, \
        magnitudes]


    return meteor_picks






def applyAstrometryFTPdetectinfo(dir_path, ftp_detectinfo_file, platepar_file, UT_corr=0, platepar=None):
    """ Use the given platepar to calculate the celestial coordinates of detected meteors from a FTPdetectinfo
        file and save the updates values.

    Arguments:
        dir_path: [str] Path to the night.
        ftp_detectinfo_file: [str] Name of the FTPdetectinfo file.
        platepar_file: [str] Name of the platepar file.

    Keyword arguments:
        UT_corr: [float] Difference of time from UTC in hours.
        platepar: [Platepar obj] Loaded platepar. None by default. If given, the platepar file won't be read, 
            but this platepar structure will be used instead.

    Return:
        None
    """

    # If the FTPdetectinfo file does not exist, skip everything
    if not os.path.isfile(os.path.join(dir_path, ftp_detectinfo_file)):
        print('The given FTPdetectinfo file does not exist:', os.path.join(dir_path, ftp_detectinfo_file))
        print('The astrometry was not computed!')
        return None

    # Save a copy of the uncalibrated FTPdetectinfo
    ftp_detectinfo_copy = "".join(ftp_detectinfo_file.split('.')[:-1]) + "_uncalibrated.txt"

    # Back up the original FTPdetectinfo, only if a backup does not exist already
    if not os.path.isfile(os.path.join(dir_path, ftp_detectinfo_copy)):
        shutil.copy2(os.path.join(dir_path, ftp_detectinfo_file), os.path.join(dir_path, ftp_detectinfo_copy))

    # Load platepar from file if not given
    if platepar is None:

        # Load the platepar
        platepar = RMS.Formats.Platepar.Platepar()
        platepar.read(os.path.join(dir_path, platepar_file), use_flat=None)


    # Load the FTPdetectinfo file
    meteor_data = readFTPdetectinfo(dir_path, ftp_detectinfo_file)

    # List for final meteor data
    meteor_list = []

    # Go through every meteor
    for meteor in meteor_data:

        ff_name, cam_code, meteor_No, n_segments, fps, hnr, mle, binn, px_fm, rho, phi, meteor_meas = meteor

        # Apply the platepar to the given centroids
        meteor_picks = applyPlateparToCentroids(ff_name, fps, meteor_meas, platepar)

        # Add the calculated values to the final list
        meteor_list.append([ff_name, meteor_No, rho, phi, meteor_picks])


    # Calibration string to be written to the FTPdetectinfo file
    calib_str = 'Calibrated with RMS on: ' + str(datetime.datetime.utcnow()) + ' UTC'

    # If no meteors were detected, set dummpy parameters
    if len(meteor_list) == 0:
        cam_code = ''
        fps = 0

    # Save the updated FTPdetectinfo
    writeFTPdetectinfo(meteor_list, dir_path, ftp_detectinfo_file, dir_path, cam_code, fps, 
        calibration=calib_str, celestial_coords_given=True)





if __name__ == "__main__":


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Apply the platepar to the given FTPdetectinfo file.")

    arg_parser.add_argument('ftpdetectinfo_path', nargs=1, metavar='FTPDETECTINFO_PATH', type=str, \
        help='Path to the FF file.')

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    ftpdetectinfo_path = cml_args.ftpdetectinfo_path[0]

    # Extract the directory path
    dir_path, ftp_detectinfo_file = os.path.split(os.path.abspath(ftpdetectinfo_path))

    if not ftp_detectinfo_file.endswith('.txt'):
        print("Please provide a FTPdetectinfo file! It has to end with .txt")
        sys.exit()

    # Find the platepar file
    platepar_file = None
    for file_name in os.listdir(dir_path):
        if 'platepar_' in file_name:
            platepar_file = file_name
            break

    if platepar_file is None:
        print('ERROR! Could not find the platepar file!')
        sys.exit()


    # Apply the astrometry to the given FTPdetectinfo file
    applyAstrometryFTPdetectinfo(dir_path, ftp_detectinfo_file, platepar_file)


    # Recompute the UFOOrbit file
    Utils.RMS2UFO.FTPdetectinfo2UFOOrbitInput(dir_path, ftp_detectinfo_file, os.path.join(dir_path, \
        platepar_file))

    print('Done!')



    # sys.exit()


    # # TEST CONVERSION FUNCTIONS

    # # Load the platepar
    # platepar = RMS.Formats.Platepar.Platepar()
    # platepar.read("/home/dvida/Desktop/HR000A_20181214_170136_990012_detected/platepar_cmn2010.cal")

    # from RMS.Formats.FFfile import getMiddleTimeFF
    # from RMS.Astrometry.Conversions import date2JD, jd2Date
    # time = getMiddleTimeFF('FF_HR000A_20181215_015724_739_0802560.fits', 25)

    # # Convert time to UT
    # #time = jd2Date(date2JD(*time, UT_corr=platepar.UT_corr))

    # # Star
    # star_x = 435.0
    # star_y = 285.0

    # print('Star X, Y:', star_x, star_y)

    # jd, ra_array, dec_array, mag = xyToRaDecPP(np.array([time, time]), np.array([star_x, 100]), np.array([star_y, 100]), np.array([1, 1]), platepar)

    # print(ra_array, dec_array)
    # ra = ra_array[0]
    # dec = dec_array[0]

    # ra_h = int(ra/15)
    # ra_min = int((ra/15 - ra_h)*60)
    # ra_sec = ((ra/15 - ra_h)*60 - ra_min)*60

    # dec_d = int(dec)
    # dec_min = int((dec - dec_d)*60)
    # dec_sec = ((dec - dec_d)*60 - dec_min)*60

    # print('Computed RA, Dec:')
    # print(ra_h, ra_min, ra_sec)
    # print(dec_d, dec_min, dec_sec)


    # # Convert the coordinates back to image coordinates
    # # ra_star = (6 + (45 + 8/60)/60)*15
    # # dec_star = -(16 + (43 + 21/60)/60)
    # ra_star = ra
    # dec_star = dec
    # x_star, y_star = raDecToXYPP(np.array([ra_star]), np.array([dec_star]), \
    #     np.array([date2JD(*time)]), platepar)

    # print('Star X, Y computed:', x_star, y_star)