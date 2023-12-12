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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import copy
import datetime
import os
import shutil
import sys

import numpy as np
# Import Cython functions
import pyximport
import RMS.Formats.Platepar
import scipy.optimize
from RMS.Astrometry.AtmosphericExtinction import \
    atmosphericExtinctionCorrection
from RMS.Astrometry.Conversions import J2000_JD, date2JD, jd2Date, raDec2AltAz
from RMS.Formats.FFfile import filenameToDatetime
from RMS.Formats.FTPdetectinfo import (findFTPdetectinfoFile,
                                       readFTPdetectinfo, writeFTPdetectinfo)
from RMS.Math import angularSeparation, cartesianToPolar, polarToCartesian

pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from RMS.Astrometry.CyFunctions import (cyraDecToXY, cyTrueRaDec2ApparentAltAz,
                                        cyXYToRADec,
                                        eqRefractionApparentToTrue,
                                        equatorialCoordPrecession)

# Handle Python 2/3 compability
if sys.version_info.major == 3:
    unicode = str


def limitVignettingCoefficient(x_res, y_res, vignetting_coeff, delta_mag=2.5):
    """ Limit the vignetting coefficient so that the drop in brigness in the corner of the image is not
        does not exceed delta_mag magnitudes.
        
    Arguments:
        x_res: [int] Image width (px).
        y_res: [int] Image height (px).
        vignetting_coeff: [float] Vignetting coefficient (rad/px).

    Keyword arguments:
        delta_mag: [float] Maximum drop in brightness in the corner of the image (magnitudes). Default: 2.5.

    Return:
        vignetting_coeff: [float] Limited vignetting coefficient (rad/px).

    """

    # Take an absolute value of the vignetting coefficient to make sure it is positive
    vignetting_coeff = abs(vignetting_coeff)

    # Compute the distance from the center to the corner of the image
    radius = np.sqrt((x_res/2)**2 + (y_res/2)**2)

    # Restrict the vignetting coefficient to have a maximum drop of delta_mag magnitudes in the corner of 
    # the image
    vignetting_coeff_max = np.arccos((10**(-0.4*delta_mag))**(1/4))/radius

    # Limit the vignetting coefficient
    if vignetting_coeff > vignetting_coeff_max:
        vignetting_coeff = vignetting_coeff_max

    return vignetting_coeff



def correctVignetting(px_sum, radius, vignetting_coeff):
    """ Given a pixel sum, radius from focal plane centre and the vignetting coefficient, correct the pixel
        sum for the vignetting effect.
    Arguments:
        px_sum: [float] Pixel sum.
        radius: [float] Radius (px) from focal plane centre.
        vignetting_coeff: [float] Vignetting coefficient (rad/px).
    Return:
        px_sum_corr: [float] Corrected pixel sum.
    """

    # Make sure the vignetting coefficient is a number
    if vignetting_coeff is None:
        vignetting_coeff = 0.0

    return px_sum/(np.cos(vignetting_coeff*radius)**4)


def extinctionCorrectionTrueToApparent(catalog_mags, ra_data, dec_data, jd, platepar):
    """ Compute apparent magnitudes by applying extinction correction to catalog magnitudes. 

    Arguments:
        catalog_mags: [list] A list of catalog magnitudes.
        ra_data: [list] A list of catalog right ascensions (J2000) in degrees.
        dec_data: [list] A list of catalog declinations (J2000) in degrees.
        jd: [float] Julian date.
        platepar: [Platepar object]

    Return:
        corrected_catalog_mags: [list] Extinction corrected catalog magnitudes.

    """


    ### Compute star elevations above the horizon (epoch of date, true) ###

    # Compute elevation above the horizon
    elevation_data = []
    for ra, dec in zip(ra_data, dec_data):

        # Precess to epoch of date
        ra, dec = equatorialCoordPrecession(J2000_JD.days, jd, np.radians(ra), np.radians(dec))

        # Compute elevation
        _, elev = raDec2AltAz(np.degrees(ra), np.degrees(dec), jd, platepar.lat, platepar.lon)

        if elev < 0:
            elev = 0

        elevation_data.append(elev)

    ### ###

    # Correct catalog magnitudes for extinction
    extinction_correction = atmosphericExtinctionCorrection(np.array(elevation_data), platepar.elev) \
        - atmosphericExtinctionCorrection(90, platepar.elev)
    corrected_catalog_mags = np.array(catalog_mags) + platepar.extinction_scale*extinction_correction

    return corrected_catalog_mags



def extinctionCorrectionApparentToTrue(mags, x_data, y_data, jd, platepar):
    """ Compute true magnitudes by applying extinction correction to apparent magnitudes. 
    
    Arguments:
        mags: [list] A list of apparent magnitudes.
        x_data: [list] A list of pixel columns.
        y_data: [list] A list of pixel rows.
        jd: [float] Julian date.
        platepar: [Platepar object]

    Return:
        corrected_mags: [list] A list of extinction corrected mangitudes.

    """


    ### Compute star elevations above the horizon (epoch of date, true) ###

    # Compute RA/Dec in J2000
    _, ra_data, dec_data, _ = xyToRaDecPP(len(x_data)*[jd2Date(jd)], x_data, y_data, len(x_data)*[1], \
        platepar, extinction_correction=False)

    # Compute elevation above the horizon
    elevation_data = []
    for ra, dec in zip(ra_data, dec_data):

        # Precess to epoch of date
        ra, dec = equatorialCoordPrecession(J2000_JD.days, jd, np.radians(ra), np.radians(dec))

        # Compute elevation
        _, elev = raDec2AltAz(np.degrees(ra), np.degrees(dec), jd, platepar.lat, platepar.lon)

        if elev < 0:
            elev = 0
            
        elevation_data.append(elev)

    ### ###

    # Correct catalog magnitudes for extinction
    extinction_correction = atmosphericExtinctionCorrection(np.array(elevation_data), platepar.elev) \
        - atmosphericExtinctionCorrection(90, platepar.elev)
    corrected_mags = np.array(mags) - platepar.extinction_scale*extinction_correction

    return corrected_mags



def photomLine(input_params, photom_offset, vignetting_coeff):
    """ Line used for photometry, the slope is fixed to -2.5, only the photometric offset is given.

    Arguments:
        input_params: [tuple]
            - px_sum: [float] sum of pixel intensities.
            - radius: [float] Radius from the centre of the focal plane to the centroid.
        photom_offset: [float] The photometric offet.
        vignetting_coeff: [float] Vignetting coefficient (rad/px).
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
                - vignetting_coeff: [float] Vignetting coefficient (rad/px).
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



def photometryFitRobust(px_intens_list, radius_list, catalog_mags, fixed_vignetting=None):
    """ Fit the photometry on given data robustly by rejecting 2 sigma residuals several times.

    Arguments:
        px_intens_list: [list] A list of sums of pixel intensities.
        radius_list: [list] A list of raddia from the focal plane centre (px).
        catalog_mags: [list] A list of corresponding catalog magnitudes of stars.
    Keyword arguments:
        fixed_vignetting: [float] Fixed vignetting coefficient. None by default, in which case it will be
            computed.
    Return:
        (photom_offset, fit_stddev, fit_resid, px_intens_list, radius_list, catalog_mags):
            photom_params: [list]
                - photom_offset: [float] The photometric offset.
                - vignetting_coeff: [float] Vignetting coefficient (rad/px).
            fit_stddev: [float] The standard deviation of the fit.
            fit_resid: [float] Magnitude fit residuals.
            px_intens_list: [ndarray] A list of filtered pixel intensities.
            radius_list: [ndarray] A list of filtered radiia.
            catalog_mags: [ndarray] A list of filtered catalog magnitudes.
    """

    # Convert everything to numpy arrays
    px_intens_list = np.array(px_intens_list)
    radius_list = np.array(radius_list)
    catalog_mags = np.array(catalog_mags)


    # Reject outliers and re-fit the photometry several times
    reject_iters = 5
    for i in range(reject_iters):

        # Fit the photometry on automated star intensities (use the fixed vignetting coeff)
        photom_params, fit_stddev, fit_resid = photometryFit(px_intens_list, radius_list, catalog_mags, \
            fixed_vignetting=fixed_vignetting)

        # Skip the rejection in the last iteration
        if i < reject_iters - 1:

            # Reject all 2 sigma residuals and all larger than 1.0 mag, and re-fit the photometry
            filter_indices = (np.abs(fit_resid) < 2*fit_stddev) & (np.abs(fit_resid) < 1.0)
            px_intens_list = px_intens_list[filter_indices]
            radius_list = radius_list[filter_indices]
            catalog_mags = catalog_mags[filter_indices]


    return photom_params, fit_stddev, fit_resid, px_intens_list, radius_list, catalog_mags



def computeFOVSize(platepar):
    """ Computes the size of the FOV in deg from the given platepar.

    Arguments:
        platepar: [Platepar instance]
    Return:
        fov_h: [float] Horizontal FOV in degrees.
        fov_v: [float] Vertical FOV in degrees.
    """

    # Construct poinits on the middle of every side of the image
    x_data = np.array([               0,  platepar.X_res,  platepar.X_res/2, platepar.X_res/2, platepar.X_res/2.0])
    y_data = np.array([platepar.Y_res/2, platepar.Y_res/2,                0, platepar.Y_res,   platepar.Y_res/2.0])
    time_data = np.array(len(x_data)*[jd2Date(platepar.JD)])
    level_data = np.ones(len(x_data))

    # Compute RA/Dec of the points
    _, ra_data, dec_data, _ = xyToRaDecPP(time_data, x_data, y_data, level_data, platepar, \
        extinction_correction=False)

    ra1, ra2, ra3, ra4, ra_mid = ra_data
    dec1, dec2, dec3, dec4, dec_mid = dec_data

    # Compute horizontal FOV
    fov_hl = np.degrees(angularSeparation(np.radians(ra1), np.radians(dec1), np.radians(ra_mid), \
        np.radians(dec_mid)))
    fov_hr = np.degrees(angularSeparation(np.radians(ra2), np.radians(dec2), np.radians(ra_mid), \
        np.radians(dec_mid)))
    fov_h = fov_hl + fov_hr

    # Compute vertical FOV
    fov_vu = np.degrees(angularSeparation(np.radians(ra3), np.radians(dec3), np.radians(ra_mid), \
        np.radians(dec_mid)))
    fov_vd = np.degrees(angularSeparation(np.radians(ra4), np.radians(dec4), np.radians(ra_mid), \
        np.radians(dec_mid)))
    fov_v = fov_vu + fov_vd

    return fov_h, fov_v



def getFOVSelectionRadius(platepar):
    """ Get a radius around the centre of the FOV which includes the FOV, but excludes stars outside the FOV.
    Arguments:
        platepar: [Platepar instance]

    Return:
        fov_radius: [float] Radius in degrees.
    """

    # Construct poinits on the middle of every side of the image
    x_data = np.array([0, platepar.X_res, platepar.X_res,              0, platepar.X_res/2.0])
    y_data = np.array([0, platepar.Y_res,              0, platepar.Y_res, platepar.Y_res/2.0])
    time_data = np.array(len(x_data)*[jd2Date(platepar.JD)])
    level_data = np.ones(len(x_data))

    # Compute RA/Dec of the points
    _, ra_data, dec_data, _ = xyToRaDecPP(time_data, x_data, y_data, level_data, platepar, \
        extinction_correction=False)

    ra1, ra2, ra3, ra4, ra_mid = ra_data
    dec1, dec2, dec3, dec4, dec_mid = dec_data

    # Angular separation between the centre of the FOV and corners
    ul_sep = np.degrees(angularSeparation(np.radians(ra1), np.radians(dec1), np.radians(ra_mid), np.radians(dec_mid)))
    lr_sep = np.degrees(angularSeparation(np.radians(ra2), np.radians(dec2), np.radians(ra_mid), np.radians(dec_mid)))
    ur_sep = np.degrees(angularSeparation(np.radians(ra3), np.radians(dec3), np.radians(ra_mid), np.radians(dec_mid)))
    ll_sep = np.degrees(angularSeparation(np.radians(ra4), np.radians(dec4), np.radians(ra_mid), np.radians(dec_mid)))

    # Take the average radius
    fov_radius = np.mean([ul_sep, lr_sep, ur_sep, ll_sep])

    return fov_radius



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

    # Compute apparent alt/az in the epoch of date from X,Y
    jd_arr, ra_arr, dec_arr, _ = xyToRaDecPP(2*[jd2Date(platepar.JD)], [img_mid_w, img_up_w], \
        [img_mid_h, img_up_h], [1, 1], platepar, extinction_correction=False)
    azim_mid, alt_mid = cyTrueRaDec2ApparentAltAz(np.radians(ra_arr[0]), np.radians(dec_arr[0]), jd_arr[0], \
        np.radians(platepar.lat), np.radians(platepar.lon), platepar.refraction)
    azim_up, alt_up = cyTrueRaDec2ApparentAltAz(np.radians(ra_arr[1]), np.radians(dec_arr[1]), jd_arr[1], \
        np.radians(platepar.lat), np.radians(platepar.lon), platepar.refraction)

    # Compute the rotation wrt horizon (deg)
    rot_angle = np.degrees(np.arctan2(alt_up - alt_mid, azim_up - azim_mid))

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



def calculateMagnitudes(px_sum_arr, radius_arr, photom_offset, vignetting_coeff):
    """ Calculate the magnitude of the data points with given magnitude calibration parameters.

    Arguments:
        px_sum_arr: [ndarray] Sum of pixel intensities of the meteor centroid (arbitrary units).
        radius_arr: [ndarray] A list of raddia from image centre (px).
        photom_offset: [float] Magnitude intercept, i.e. the photometric offset.
        vignetting_coeff: [float] Vignetting ceofficient (rad/px).
    Return:
        magnitude_data: [ndarray] Apparent magnitude.
    """

    magnitude_data = np.zeros_like(px_sum_arr, dtype=np.float64)

    # Go through all levels of a meteor
    for i, (px_sum, radius) in enumerate(zip(px_sum_arr, radius_arr)):

        # Make sure the pixel sum is a number
        if px_sum is None:
            px_sum = 1

        # Correct vignetting
        px_sum_corr = correctVignetting(px_sum, radius, vignetting_coeff)

        # Save magnitude data to the output array
        magnitude_data[i] = -2.5*np.log10(px_sum_corr) + photom_offset


    return magnitude_data



def xyToRaDecPP(time_data, X_data, Y_data, level_data, platepar, extinction_correction=True, \
    measurement=False, jd_time=False):
    """ Converts image XY to RA,Dec, but it takes a platepar instead of individual parameters. 

    Arguments:
        time_data: [2D ndarray] Numpy array containing either: 
            if jd_time is False - time tuples of each data point (year, month, day,hour, minute, second, 
                millisecond).
            if jd_time is True - Julian dates.
        X_data: [ndarray] 1D numpy array containing the image X component.
        Y_data: [ndarray] 1D numpy array containing the image Y component.
        level_data: [ndarray] Levels of the meteor centroid.
        platepar: [Platepar structure] Astrometry parameters.

    Keyword arguments:
        extinction_correction: [bool] Apply extinction correction. True by default. False is set to prevent 
            infinite recursion in extinctionCorrectionApparentToTrue when set to True.
        measurement: [bool] Indicates if the given images values are image measurements. Used for correcting
            celestial coordinates for refraction if the refraction was not taken into account during
            plate fitting.
        jd_time: [bool] If True, time_data is expected as a list of Julian dates. False by default.

    Return:
        (JD_data, RA_data, dec_data, magnitude_data): [tuple of ndarrays]
            JD_data: [ndarray] Julian date of each data point.
            RA_data: [ndarray] Right ascension of each point (deg).
            dec_data: [ndarray] Declination of each point (deg).
            magnitude_data: [ndarray] Array of meteor's lightcurve apparent magnitudes.
    """


    # Convert time to Julian date
    if jd_time:
        JD_data = np.array(time_data)
    else:
        JD_data = np.array([date2JD(*time_data_entry) for time_data_entry in time_data], dtype=np.float64)

    # Convert x,y to RA/Dec using a fast cython function
    RA_data, dec_data = cyXYToRADec(JD_data, np.array(X_data, dtype=np.float64), \
        np.array(Y_data, dtype=np.float64), float(platepar.lat), float(platepar.lon), float(platepar.X_res), \
        float(platepar.Y_res), float(platepar.Ho), float(platepar.JD), float(platepar.RA_d), 
        float(platepar.dec_d), float(platepar.pos_angle_ref), float(platepar.F_scale), platepar.x_poly_fwd, 
        platepar.y_poly_fwd, unicode(platepar.distortion_type), refraction=platepar.refraction, \
        equal_aspect=platepar.equal_aspect, force_distortion_centre=platepar.force_distortion_centre, \
        asymmetry_corr=platepar.asymmetry_corr)

    # Correct the coordinates for refraction if it wasn't taken into account during the astrometry calibration
    #   procedure
    if (not platepar.refraction) and measurement and platepar.measurement_apparent_to_true_refraction:
        for i, entry in enumerate(zip(JD_data, RA_data, dec_data)):
            jd, ra, dec = entry
            ra, dec = eqRefractionApparentToTrue(np.radians(ra), np.radians(dec), jd, \
                np.radians(platepar.lat), np.radians(platepar.lon))

            RA_data[i] = np.degrees(ra)
            dec_data[i] = np.degrees(dec)
            

    # Compute radiia from image centre
    radius_arr = np.hypot(np.array(X_data) - platepar.X_res/2, np.array(Y_data) - platepar.Y_res/2)

    # Calculate magnitudes
    magnitude_data = calculateMagnitudes(level_data, radius_arr, platepar.mag_lev, platepar.vignetting_coeff)


    # Extinction correction
    if extinction_correction:
        magnitude_data = extinctionCorrectionApparentToTrue(magnitude_data, X_data, Y_data, JD_data[0], \
            platepar)


    return JD_data, RA_data, dec_data, magnitude_data





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

    # Use the cythonized funtion insted of the Python function
    X_data, Y_data = cyraDecToXY(RA_data, dec_data, float(jd), float(platepar.lat), float(platepar.lon),
        float(platepar.X_res), float(platepar.Y_res), float(platepar.Ho), float(platepar.JD),  
        float(platepar.RA_d), float(platepar.dec_d), float(platepar.pos_angle_ref), platepar.F_scale, 
        platepar.x_poly_rev, platepar.y_poly_rev, unicode(platepar.distortion_type), 
        refraction=platepar.refraction, equal_aspect=platepar.equal_aspect, 
        force_distortion_centre=platepar.force_distortion_centre, asymmetry_corr=platepar.asymmetry_corr)

    return X_data, Y_data




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
        level_data, platepar, measurement=True)


    # Compute azimuth and altitude of centroids
    az_data = np.zeros_like(RA_data)
    alt_data = np.zeros_like(RA_data)

    for i in range(len(az_data)):

        jd = JD_data[i]
        ra_tmp = RA_data[i]
        dec_tmp = dec_data[i]

        # Precess RA/Dec to epoch of date
        ra_tmp, dec_tmp = equatorialCoordPrecession(J2000_JD.days, jd, np.radians(ra_tmp), \
            np.radians(dec_tmp))

        # Alt/Az are apparent (in the epoch of date, corresponding to geographical azimuths)
        az_tmp, alt_tmp = raDec2AltAz(np.degrees(ra_tmp), np.degrees(dec_tmp), jd, platepar.lat, platepar.lon)

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
    
    import Utils.RMS2UFO

    ### COMMAND LINE ARGUMENTS
    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Apply the platepar to the given FTPdetectinfo file.")

    arg_parser.add_argument('ftpdetectinfo_path', nargs=1, metavar='FTPDETECTINFO_PATH', type=str, \
        help='Path to the FF file.')

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    ftpdetectinfo_path = cml_args.ftpdetectinfo_path[0]
    ftpdetectinfo_path = findFTPdetectinfoFile(ftpdetectinfo_path)

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
