""" CMN-style astrometric calibration.
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

from __future__ import absolute_import, division, print_function

import copy
import datetime
import json
import os

import matplotlib.pyplot as plt
import numpy as np

# Import Cython functions
import pyximport
import RMS.Astrometry.ApplyAstrometry
import scipy.optimize
from RMS.Astrometry.Conversions import date2JD, jd2Date, trueRaDec2ApparentAltAz
from RMS.Math import angularSeparation, sphericalPointFromHeadingAndDistance

pyximport.install(setup_args={'include_dirs': [np.get_include()]})
from RMS.Astrometry.CyFunctions import (
    cyApparentAltAz2TrueRADec,
    cyTrueRaDec2ApparentAltAz,
    pyRefractionTrueToApparent,
)


class stationData(object):
    """Holds information about one meteor station (location) and observed points."""

    def __init__(self, file_name):
        self.file_name = file_name
        self.station_code = ''
        self.lon = 0
        self.lat = 0
        self.h = 0
        self.points = []

    def __str__(self):
        return 'Station: ' + self.station_code + ' data points: ' + str(len(self.points))


def parseInf(file_name):
    """Parse information from an INF file to a stationData object."""

    station_data_obj = stationData(file_name)

    with open(file_name) as f:
        for line in f.readlines()[2:]:

            line = line.split()

            if 'Station_Code' in line[0]:
                station_data_obj.station_code = line[1]

            elif 'Long' in line[0]:
                station_data_obj.lon = float(line[1])

            elif 'Lati' in line[0]:
                station_data_obj.lat = float(line[1])

            elif 'Height' in line[0]:
                station_data_obj.h = int(line[1])

            else:
                station_data_obj.points.append(map(float, line))

    return station_data_obj


def getCatalogStarsImagePositions(catalog_stars, jd, platepar):
    """Get image positions of catalog stars using the current platepar values.
    Arguments:
        catalog_stars: [2D list] A list of (ra, dec, mag) pairs of catalog stars.
        jd: [float] Julian date for transformation.
        platepar: [Platepar]
    Return:
        (x_array, y_array mag_catalog): [tuple of ndarrays] X, Y positons and magnitudes of stars on the
            image.
    """

    ra_catalog, dec_catalog, mag_catalog = catalog_stars.T

    # Convert star RA, Dec to image coordinates
    x_array, y_array = RMS.Astrometry.ApplyAstrometry.raDecToXYPP(ra_catalog, dec_catalog, jd, platepar)

    return x_array, y_array, mag_catalog


def getPairedStarsSkyPositions(img_x, img_y, jd, platepar):
    """Compute RA, Dec of all paired stars on the image given the platepar.
    Arguments:
        img_x: [ndarray] Array of column values of the stars.
        img_y: [ndarray] Array of row values of the stars.
        jd: [float] Julian date for transformation.
        platepar: [Platepar instance] Platepar object.
    Return:
        (ra_array, dec_array): [tuple of ndarrays] Arrays of RA and Dec of stars on the image.
    """

    # Compute RA, Dec of image stars
    img_time = jd2Date(jd)
    _, ra_array, dec_array, _ = RMS.Astrometry.ApplyAstrometry.xyToRaDecPP(
        len(img_x) * [img_time], img_x, img_y, len(img_x) * [1], platepar, extinction_correction=False
    )

    return ra_array, dec_array


class Platepar(object):
    def __init__(self, distortion_type="poly3+radial"):
        """Astrometric and photometric calibration plate parameters. Several distortion types are supported.

        Arguments:
            file_name: [string] Path to the platepar file.
        Keyword arguments:
            distortion_type: [str] Distortion type. It can be one of the following:
                - "poly3+radial" - 3rd order polynomial fit including a single radial term
                - "poly3+radial3" - 3rd order polynomial fit including two radial terms (r + r^3)
                - "radial3-all" - 3rd order radial distortion, all powers
                - "radial4-all" - 4rd order radial distortion, all powers
                - "radial5-all" - 5rd order radial distortion, all powers
                - "radial3-odd" - 3rd order radial distortion, only odd powers
                - "radial5-odd" - 5th order radial distortion, only odd powers
                - "radial7-odd" - 7th order radial distortion, only odd powers
                - "radial9-odd" - 7th order radial distortion, only odd powers

        Return:
            self: [object] Instance of this class with loaded platepar parameters.
        """

        self.version = 2

        # Set the distortion type
        self.distortion_type = distortion_type
        self.setDistortionType(self.distortion_type)

        # Station coordinates
        self.lat = self.lon = self.elev = 0

        # Reference time and date
        self.time = 0
        self.JD = 2451545.0

        # UT correction
        self.UT_corr = 0

        self.Ho = 0
        self.X_res = 1280
        self.Y_res = 720

        self.fov_h = 88
        self.fov_v = 45

        # FOV centre
        self.RA_d = 0
        self.dec_d = 0
        self.pos_angle_ref = 0
        self.rotation_from_horiz = 0

        self.az_centre = 0
        self.alt_centre = 0

        # FOV scale (px/deg)
        self.F_scale = 1.0

        # Refraction on/off
        self.refraction = True

        # If the calibration was done without then refraction and points on the sky are measured, then they
        #   need to be corrected for refraction. WARNING: This should not be used if the distortion model
        #   itself compensates for the refraction (e.g. the polynomial model)
        self.measurement_apparent_to_true_refraction = False

        # Equal aspect (X and Y scales are equal) - used ONLY for radial distortion
        self.equal_aspect = True

        # Force distortion centre to image centre
        self.force_distortion_centre = False

        # Asymmetry correction - used ONLY for radial distortion
        self.asymmetry_corr = False

        # Photometry calibration
        self.mag_0 = -2.5
        self.mag_lev = 1.0
        self.mag_lev_stddev = 0.0
        self.gamma = 1.0
        self.vignetting_coeff = 0.001
        self.vignetting_fixed = True

        # Extinction correction scaling
        self.extinction_scale = 0.6

        self.station_code = "None"

        self.star_list = None

        # Flag to indicate that the platepar was refined with CheckFit
        self.auto_check_fit_refined = False

        # Flag to indicate that the platepar was successfuly auto recalibrated on an individual FF files
        self.auto_recalibrated = False

        # Init the distortion parameters
        self.resetDistortionParameters()

    def resetDistortionParameters(self, preserve_centre=False):
        """Set the distortion parameters to zero.

        Keyword arguments:
            preserve_centre: [bool] Don't reset the distortion centre. False by default, in which case it will
                be reset.
        """

        # Store the distortion centre if it needs to be preserved
        if preserve_centre:

            # Preserve centre for the radial distortion
            if self.distortion_type.startswith("radial"):

                # Note that the radial distortion parameters are kept in the X poly array
                x_centre_fwd, y_centre_fwd = self.x_poly_fwd[0], self.x_poly_fwd[1]
                x_centre_rev, y_centre_rev = self.x_poly_rev[0], self.x_poly_rev[1]

            else:

                # Preserve centre for the polynomial distortion
                x_centre_fwd, x_centre_rev = self.x_poly_fwd[0], self.x_poly_rev[0]
                y_centre_fwd, y_centre_rev = self.y_poly_fwd[0], self.y_poly_rev[0]

        # Reset distortion fit (forward and reverse)
        self.x_poly_fwd = np.zeros(shape=(self.poly_length,), dtype=np.float64)
        self.y_poly_fwd = np.zeros(shape=(self.poly_length,), dtype=np.float64)
        self.x_poly_rev = np.zeros(shape=(self.poly_length,), dtype=np.float64)
        self.y_poly_rev = np.zeros(shape=(self.poly_length,), dtype=np.float64)

        # Preserve the image centre
        if preserve_centre:

            # Preserve centre for the radial distortion
            if self.distortion_type.startswith("radial") and (not self.force_distortion_centre):

                # Note that the radial distortion parameters are kept in the X poly array
                self.x_poly_fwd[0], self.x_poly_fwd[1] = x_centre_fwd, y_centre_fwd
                self.x_poly_rev[0], self.x_poly_rev[1] = x_centre_rev, y_centre_rev

            else:

                # Preserve centre for the polynomial distortion
                self.x_poly_fwd[0], self.x_poly_rev[0] = x_centre_fwd, x_centre_rev
                self.y_poly_fwd[0], self.y_poly_rev[0] = y_centre_fwd, y_centre_rev

        # Reset the image centre
        else:
            # Set the first coeffs to 0.5, as that is the real centre of the FOV
            self.x_poly_fwd[0] = 0.5
            self.y_poly_fwd[0] = 0.5
            self.x_poly_rev[0] = 0.5
            self.y_poly_rev[0] = 0.5

            # If the distortion is radial, set the second X parameter to 0.5, as x_poly[1] is used for the Y
            #   offset in the radial models
            if self.distortion_type.startswith("radial"):
                self.x_poly_fwd[0] /= self.X_res / 2
                self.x_poly_rev[0] /= self.X_res / 2
                self.x_poly_fwd[1] = 0.5 / (self.Y_res / 2)
                self.x_poly_rev[1] = 0.5 / (self.Y_res / 2)

                # If the distortion center is forced to the center of the image, reset all parameters to zero
                if self.force_distortion_centre:
                    self.x_poly_fwd *= 0
                    self.x_poly_rev *= 0

        self.x_poly = self.x_poly_fwd
        self.y_poly = self.y_poly_fwd

    def setDistortionType(self, distortion_type, reset_params=True):
        """Sets the distortion type."""

        # List of distortion types
        self.distortion_type_list = [
            "poly3+radial",
            "poly3+radial3",
            "poly3+radial5",
            "radial3-all",
            "radial4-all",
            "radial5-all",
            "radial3-odd",
            "radial5-odd",
            "radial7-odd",
            "radial9-odd",
        ]

        # Lenghts of full polynomials, (including distortion center, aspect, and asymmetry correction for
        #   radial distortions)
        self.distortion_type_poly_length = [12, 13, 14, 7, 8, 9, 6, 7, 8, 9]

        # Set the length of the distortion polynomial depending on the distortion type
        if distortion_type in self.distortion_type_list:

            # If the new distortion type (poly vs radial) is different from the old, reset the parameters
            if distortion_type[:4] != self.distortion_type[:4]:
                reset_params = True

            # If the all vs odd only radial powers type is changed, reset the distortion
            if distortion_type.startswith("radial"):
                if distortion_type[-3:] != self.distortion_type[-3:]:
                    reset_params = True

            self.distortion_type = distortion_type

            # Get the polynomial length
            self.poly_length = self.distortion_type_poly_length[
                self.distortion_type_list.index(distortion_type)
            ]

            # Remove distortion center for radial distortions if it's not used
            if distortion_type.startswith("radial"):
                if self.force_distortion_centre:
                    self.poly_length -= 2

            # Remove aspect parameter for radial distortions if it's not used
            if distortion_type.startswith("radial"):
                if self.equal_aspect:
                    self.poly_length -= 1

            # Remove asymmetry correction parameters for radial distortions if they are not used
            if distortion_type.startswith("radial"):
                if not self.asymmetry_corr:
                    self.poly_length -= 2

        else:
            raise ValueError("The distortion type is not recognized: {:s}".format(self.distortion_type))

        # Reset distortion parameters
        if reset_params:
            self.resetDistortionParameters()

        # Set the correct polynomial size
        self.padDictParams()

    def addVignettingCoeff(self, use_flat):
        """Add a vignetting coeff to the platepar if it doesn't have one.

        Arguments:
            use_flat: [bool] Is the flat used or not.
        """

        # Add a vignetting coefficient if it's not set
        if self.vignetting_coeff is None:

            # Only add it if a flat is not used
            if use_flat:
                self.vignetting_coeff = 0.0

            else:

                # Use 0.001 rad/px as the default coefficeint, as that's the one for 3.6 mm f/0.95 and 16 mm
                #   f/1.0 lenses. The vignetting coeff is dependent on the resolution, the default value of
                #   0.001 rad/px is for 720p.
                self.vignetting_coeff = 0.001 * np.hypot(1280, 720) / np.hypot(self.X_res, self.Y_res)

    def fitPointing(self, jd, img_stars, catalog_stars, fixed_scale=False):
        """Fit pointing parameters to the list of star image and celectial catalog coordinates.
        At least 4 stars are needed to fit the rigid body parameters.

        New parameters are saved to the given object (self).

        Arguments:
            jd: [float] Julian date of the image.
            img_stars: [list] A list of (x, y, intensity_sum) entires for every star.
            catalog_stars: [list] A list of (ra, dec, mag) entries for every star (degrees).

        Keyword arguments:
            fixed_scale: [bool] Keep the scale fixed. False by default.

        """

        def _calcImageResidualsAstro(params, platepar, jd, catalog_stars, img_stars):
            """Calculates the differences between the stars on the image and catalog stars in image
            coordinates with the given astrometrical solution.
            """

            # Extract fitting parameters
            ra_ref, dec_ref, pos_angle_ref = params[:3]
            if not fixed_scale:
                F_scale = params[3]

            img_x, img_y, _ = img_stars.T

            pp_copy = copy.deepcopy(platepar)

            # Assign guessed parameters
            pp_copy.RA_d = ra_ref
            pp_copy.dec_d = dec_ref
            pp_copy.pos_angle_ref = pos_angle_ref

            if not fixed_scale:
                pp_copy.F_scale = abs(F_scale)

            # Get image coordinates of catalog stars
            catalog_x, catalog_y, catalog_mag = getCatalogStarsImagePositions(catalog_stars, jd, pp_copy)

            # Calculate the sum of squared distances between image stars and catalog stars
            dist_sum = np.sum((catalog_x - img_x) ** 2 + (catalog_y - img_y) ** 2)

            return dist_sum

        def _calcSkyResidualsAstro(params, platepar, jd, catalog_stars, img_stars):
            """Calculates the differences between the stars on the image and catalog stars in sky
            coordinates with the given astrometrical solution.
            """

            # Extract fitting parameters
            ra_ref, dec_ref, pos_angle_ref = params[:3]
            if not fixed_scale:
                F_scale = params[3]

            img_x, img_y, _ = img_stars.T

            pp_copy = copy.deepcopy(platepar)

            # Assign guessed parameters
            pp_copy.RA_d = ra_ref
            pp_copy.dec_d = dec_ref
            pp_copy.pos_angle_ref = pos_angle_ref

            if not fixed_scale:
                pp_copy.F_scale = abs(F_scale)

            img_x, img_y, _ = img_stars.T

            # Get image coordinates of catalog stars
            ra_array, dec_array = getPairedStarsSkyPositions(img_x, img_y, jd, pp_copy)

            ra_catalog, dec_catalog, _ = catalog_stars.T

            # Compute the sum of the angular separation
            separation_sum = np.sum(
                angularSeparation(
                    np.radians(ra_array),
                    np.radians(dec_array),
                    np.radians(ra_catalog),
                    np.radians(dec_catalog),
                )
                ** 2
            )

            return separation_sum

        # Initial parameters for the astrometric fit
        p0 = [self.RA_d, self.dec_d, self.pos_angle_ref]

        # Add fitting scale if not fixed
        if not fixed_scale:
            p0 += [abs(self.F_scale)]

        # Fit the astrometric parameters using the reverse transform for reference
        res = scipy.optimize.minimize(
            _calcImageResidualsAstro, p0, args=(self, jd, catalog_stars, img_stars), method='SLSQP'
        )

        # # Fit the astrometric parameters using the forward transform for reference
        #   WARNING: USING THIS MAKES THE FIT UNSTABLE
        # res = scipy.optimize.minimize(_calcSkyResidualsAstro, p0, args=(self, jd, \
        #     catalog_stars, img_stars), method='Nelder-Mead')

        # Update fitted astrometric parameters
        self.RA_d, self.dec_d, self.pos_angle_ref = res.x[:3]
        if not fixed_scale:
            self.F_scale = res.x[3]

        # Force scale to be positive
        self.F_scale = abs(self.F_scale)

        # Update alt/az of pointing
        self.updateRefAltAz()

    def fitAstrometry(
        self,
        jd,
        img_stars,
        catalog_stars,
        first_platepar_fit=False,
        fit_only_pointing=False,
        fixed_scale=False,
    ):
        """Fit astrometric parameters to the list of star image and celectial catalog coordinates.
        At least 4 stars are needed to fit the rigid body parameters.

        New parameters are saved to the given object (self).

        Arguments:
            jd: [float] Julian date of the image.
            img_stars: [list] A list of (x, y, intensity_sum) entires for every star.
            catalog_stars: [list] A list of (ra, dec, mag) entries for every star (degrees).

        Keyword arguments:
            first_platepar_fit: [bool] Fit a platepar from scratch. False by default.
            fit_only_pointing: [bool] Only fit the pointing parameters, and not distortion.
            fixed_scale: [bool] Keep the scale fixed. False by default.

        """

        def _calcImageResidualsDistortion(params, platepar, jd, catalog_stars, img_stars, dimension):
            """Calculates the differences between the stars on the image and catalog stars in image
                coordinates with the given astrometrical solution.
            Arguments:
                ...
                dimension: [str] 'x' for X polynomial fit, 'y' for Y polynomial fit
            """

            # Set distortion parameters
            pp_copy = copy.deepcopy(platepar)

            if (dimension == 'x') or (dimension == 'radial'):
                pp_copy.x_poly_rev = params
                pp_copy.y_poly_rev = np.zeros(platepar.poly_length)

            else:
                pp_copy.x_poly_rev = np.zeros(platepar.poly_length)
                pp_copy.y_poly_rev = params

            img_x, img_y, _ = img_stars.T

            # Get image coordinates of catalog stars
            catalog_x, catalog_y, catalog_mag = getCatalogStarsImagePositions(catalog_stars, jd, pp_copy)

            # Calculate the sum of squared distances between image stars and catalog stars, per every
            #   dimension
            if dimension == 'x':
                dist_sum = np.sum((catalog_x - img_x) ** 2)

            elif dimension == 'y':
                dist_sum = np.sum((catalog_y - img_y) ** 2)

            # Minimization for the radial distortion
            else:

                # Compute the image fit error
                dist_sum = np.sum((catalog_x - img_x) ** 2 + (catalog_y - img_y) ** 2)

            return dist_sum

        # Modify the residuals function so that it takes a list of arguments
        def _calcImageResidualsDistortionListArguments(params, *args, **kwargs):
            return [_calcImageResidualsDistortion(param_line, *args, **kwargs) for param_line in params]

        def _calcSkyResidualsDistortion(params, platepar, jd, catalog_stars, img_stars, dimension):
            """Calculates the differences between the stars on the image and catalog stars in sky
                coordinates with the given astrometrical solution.
            Arguments:
                ...
                dimension: [str] 'x' for X polynomial fit, 'y' for Y polynomial fit
            """

            pp_copy = copy.deepcopy(platepar)

            if (dimension == 'x') or (dimension == 'radial'):
                pp_copy.x_poly_fwd = params

            else:
                pp_copy.y_poly_fwd = params

            img_x, img_y, _ = img_stars.T

            # Get image coordinates of catalog stars
            ra_array, dec_array = getPairedStarsSkyPositions(img_x, img_y, jd, pp_copy)

            ra_catalog, dec_catalog, _ = catalog_stars.T

            # Compute the sum of the angular separation
            separation_sum = np.sum(
                angularSeparation(
                    np.radians(ra_array),
                    np.radians(dec_array),
                    np.radians(ra_catalog),
                    np.radians(dec_catalog),
                )
                ** 2
            )

            return separation_sum

        # Modify the residuals function so that it takes a list of arguments
        def _calcSkyResidualsDistortionListArguments(params, *args, **kwargs):
            return [_calcSkyResidualsDistortion(param_line, *args, **kwargs) for param_line in params]

        def _calcImageResidualsAstroAndDistortionRadial(params, platepar, jd, catalog_stars, img_stars):
            """Calculates the differences between the stars on the image and catalog stars in image
                coordinates with the given astrometrical solution. Pointing and distortion paramters are used
                in the fit.

            Arguments:
                ...
                dimension: [str] 'x' for X polynomial fit, 'y' for Y polynomial fit
            """

            # Set distortion parameters
            pp_copy = copy.deepcopy(platepar)

            # Unpack pointing parameters and assign to the copy of platepar used for the fit
            ra_ref, dec_ref, pos_angle_ref, F_scale = params[:4]

            pp_copy = copy.deepcopy(platepar)

            # Unnormalize the pointing parameters
            pp_copy.RA_d = (360 * ra_ref) % (360)
            pp_copy.dec_d = -90 + (90 * dec_ref + 90) % (180.000001)
            pp_copy.pos_angle_ref = (360 * pos_angle_ref) % (360)
            pp_copy.F_scale = abs(F_scale)

            # Assign distortion parameters
            pp_copy.x_poly_rev = params[4:]

            img_x, img_y, _ = img_stars.T

            # Get image coordinates of catalog stars
            catalog_x, catalog_y, catalog_mag = getCatalogStarsImagePositions(catalog_stars, jd, pp_copy)

            # Calculate the sum of squared distances between image stars and catalog stars
            dist_sum = np.sum((catalog_x - img_x) ** 2 + (catalog_y - img_y) ** 2)

            return dist_sum

        def _calcSkyResidualsAstroAndDistortionRadial(params, platepar, jd, catalog_stars, img_stars):
            """Calculates the differences between the stars on the image and catalog stars in celestial
            coordinates with the given astrometrical solution. Pointing and distortion paramters are used
            in the fit.

            """

            # Set distortion parameters
            pp_copy = copy.deepcopy(platepar)

            # Unpack pointing parameters and assign to the copy of platepar used for the fit
            ra_ref, dec_ref, pos_angle_ref, F_scale = params[:4]

            pp_copy = copy.deepcopy(platepar)

            # Unnormalize the pointing parameters
            pp_copy.RA_d = (360 * ra_ref) % (360)
            pp_copy.dec_d = -90 + (90 * dec_ref + 90) % (180.000001)
            pp_copy.pos_angle_ref = (360 * pos_angle_ref) % (360)
            pp_copy.F_scale = abs(F_scale)

            # Assign distortion parameters
            pp_copy.x_poly_fwd = params[4:]

            img_x, img_y, _ = img_stars.T

            # Get image coordinates of catalog stars
            ra_array, dec_array = getPairedStarsSkyPositions(img_x, img_y, jd, pp_copy)

            ra_catalog, dec_catalog, _ = catalog_stars.T

            # Compute the sum of the angular separation
            separation_sum = np.sum(
                angularSeparation(
                    np.radians(ra_array),
                    np.radians(dec_array),
                    np.radians(ra_catalog),
                    np.radians(dec_catalog),
                )
                ** 2
            )

            return separation_sum

        # print('ASTRO', _calcImageResidualsAstro([self.RA_d, self.dec_d,
        #     self.pos_angle_ref, self.F_scale],  catalog_stars, img_stars))

        # print('DIS_X', _calcImageResidualsDistortion(self.x_poly_rev,  catalog_stars, \
        #     img_stars, 'x'))

        # print('DIS_Y', _calcImageResidualsDistortion(self.y_poly_rev,  catalog_stars, \
        #     img_stars, 'y'))

        ### ASTROMETRIC PARAMETERS FIT ###

        # Fit the pointing parameters (RA, Dec, rotation, scale)
        #   Only do the fit for the polynomial distortion model, or the first time if the radial distortion
        #   is used
        if (
            self.distortion_type.startswith("poly")
            or (not self.distortion_type.startswith("poly") and first_platepar_fit)
            or fit_only_pointing
        ):

            self.fitPointing(jd, img_stars, catalog_stars, fixed_scale=fixed_scale)

        ### ###

        ### DISTORTION FIT ###

        # Fit the polynomial distortion parameters if there are enough picked stars
        min_fit_stars = self.poly_length + 1

        if (len(img_stars) >= min_fit_stars) and (not fit_only_pointing):

            # Fit the polynomial distortion
            if self.distortion_type.startswith("poly"):

                ### REVERSE MAPPING FIT ###

                # Fit distortion parameters in X direction, reverse mapping
                res = scipy.optimize.minimize(
                    _calcImageResidualsDistortion,
                    self.x_poly_rev,
                    args=(self, jd, catalog_stars, img_stars, 'x'),
                    method='Nelder-Mead',
                    options={'maxiter': 10000, 'adaptive': True},
                )

                # Exctact fitted X polynomial
                self.x_poly_rev = res.x

                # Fit distortion parameters in Y direction, reverse mapping
                res = scipy.optimize.minimize(
                    _calcImageResidualsDistortion,
                    self.y_poly_rev,
                    args=(self, jd, catalog_stars, img_stars, 'y'),
                    method='Nelder-Mead',
                    options={'maxiter': 10000, 'adaptive': True},
                )

                # Extract fitted Y polynomial
                self.y_poly_rev = res.x

                ### ###

                # If this is the first fit of the distortion, set the forward parametrs to be equal to the reverse
                if first_platepar_fit:

                    self.x_poly_fwd = np.array(self.x_poly_rev)
                    self.y_poly_fwd = np.array(self.y_poly_rev)

                ### FORWARD MAPPING FIT ###

                # Fit distortion parameters in X direction, forward mapping
                res = scipy.optimize.minimize(
                    _calcSkyResidualsDistortion,
                    self.x_poly_fwd,
                    args=(self, jd, catalog_stars, img_stars, 'x'),
                    method='Nelder-Mead',
                    options={'maxiter': 10000, 'adaptive': True},
                )

                # Extract fitted X polynomial
                self.x_poly_fwd = res.x

                # Fit distortion parameters in Y direction, forward mapping
                res = scipy.optimize.minimize(
                    _calcSkyResidualsDistortion,
                    self.y_poly_fwd,
                    args=(self, jd, catalog_stars, img_stars, 'y'),
                    method='Nelder-Mead',
                    options={'maxiter': 10000, 'adaptive': True},
                )

                # IMPORTANT NOTE - the X polynomial is used to store the fit paramters
                self.y_poly_fwd = res.x

                ### ###

            # Fit radial distortion (+ pointing)
            else:

                ### FORWARD MAPPING FIT ###

                # # Fit the radial distortion - the X polynomial is used to store the fit paramters
                # res = scipy.optimize.minimize(_calcSkyResidualsDistortion, self.x_poly_fwd, \
                #     args=(self, jd, catalog_stars, img_stars, 'radial'), method='Nelder-Mead', \
                #     options={'maxiter': 10000, 'adaptive': True})

                # # Extract distortion parameters, IMPORTANT NOTE - the X polynomial is used to store the
                # #   fit paramters
                # self.x_poly_fwd = res.x

                # Fitting the pointing direction below! - if used, it should be put BEFORE the reverse fit!
                # Initial parameters for the pointing and distortion fit (normalize to the 0-1 range)
                p0 = [self.RA_d / 360, self.dec_d / 90, self.pos_angle_ref / 360, abs(self.F_scale)]
                p0 += self.x_poly_fwd.tolist()

                # Fit the radial distortion - the X polynomial is used to store the fit paramters
                res = scipy.optimize.minimize(
                    _calcSkyResidualsAstroAndDistortionRadial,
                    p0,
                    args=(self, jd, catalog_stars, img_stars),
                    method='Nelder-Mead',
                    options={'maxiter': 10000, 'adaptive': True},
                )

                # Update fitted astrometric parameters (Unnormalize the pointing parameters)
                ra_ref, dec_ref, pos_angle_ref, F_scale = res.x[:4]
                self.RA_d = (360 * ra_ref) % (360)
                self.dec_d = -90 + (90 * dec_ref + 90) % (180.000001)
                self.pos_angle_ref = (360 * pos_angle_ref) % (360)
                self.F_scale = abs(F_scale)

                self.updateRefAltAz()

                # Extract distortion parameters, IMPORTANT NOTE - the X polynomial is used to store the
                #   fit paramters
                self.x_poly_fwd = res.x[4:]

                ### ###

                # If this is the first fit of the distortion, set the forward parametrs to be equal to the reverse
                if first_platepar_fit:
                    self.x_poly_rev = np.array(self.x_poly_fwd)

                ### REVERSE MAPPING FIT ###

                # # Initial parameters for the pointing and distortion fit (normalize to the 0-1 range)
                # p0  = [self.RA_d/360.0, self.dec_d/90.0, self.pos_angle_ref/360.0, abs(self.F_scale)]
                # p0 += self.x_poly_rev.tolist()

                # # Fit the radial distortion - the X polynomial is used to store the fit paramters
                # res = scipy.optimize.minimize(_calcImageResidualsAstroAndDistortionRadial, p0, \
                #     args=(self, jd, catalog_stars, img_stars), method='Nelder-Mead', \
                #     options={'maxiter': 10000, 'adaptive': True})

                # # Update fitted astrometric parameters (Unnormalize the pointing parameters)
                # ra_ref, dec_ref, pos_angle_ref, F_scale = res.x[:4]
                # self.RA_d = (360*ra_ref)%(360)
                # self.dec_d = -90 + (90*dec_ref + 90)%(180.000001)
                # self.pos_angle_ref = (360*pos_angle_ref)%(360)
                # self.F_scale = abs(F_scale)

                # # Compute reference Alt/Az to apparent coordinates, epoch of date
                # self.updateRefAltAz()

                # # Extract distortion parameters, IMPORTANT NOTE - the X polynomial is used to store the
                # #   fit paramters
                # self.x_poly_rev = res.x[4:]

                ## Distortion-only fit below!

                # Fit the radial distortion - the X polynomial is used to store the fit paramters
                res = scipy.optimize.minimize(
                    _calcImageResidualsDistortion,
                    self.x_poly_rev,
                    args=(self, jd, catalog_stars, img_stars, 'radial'),
                    method='Nelder-Mead',
                    options={'maxiter': 10000, 'adaptive': True},
                )

                # Extract distortion parameters, IMPORTANT NOTE - the X polynomial is used to store the
                #   fit paramters
                self.x_poly_rev = res.x

                ### ###

        else:
            if len(img_stars) < min_fit_stars:
                print('Too few stars to fit the distortion, only the astrometric parameters where fitted!')

        # Set the list of stars used for the fit to the platepar
        fit_star_list = []
        for img_coords, cat_coords in zip(img_stars, catalog_stars):

            # Store time, image coordinate x, y, intensity, catalog ra, dec, mag
            fit_star_list.append([jd] + img_coords.tolist() + cat_coords.tolist())

        self.star_list = fit_star_list

        # Set the flag to indicate that the platepar was manually fitted
        self.auto_check_fit_refined = False
        self.auto_recalibrated = False

        ### ###

    def parseLine(self, f):
        """Read next line, split the line and convert parameters to float.
        @param f: [file handle] file we want to read
        @return (a1, a2, ...): [tuple of floats] parsed data from the line
        """

        return map(float, f.readline().split())

    def padDictParams(self):
        """Update the array length if an old platepar version was loaded which was shorter/longer."""

        # Extend the array if it's too short
        if self.x_poly_fwd.shape[0] < self.poly_length:
            self.x_poly_fwd = np.pad(
                self.x_poly_fwd,
                (0, self.poly_length - self.x_poly_fwd.shape[0]),
                'constant',
                constant_values=0,
            )
            self.x_poly_rev = np.pad(
                self.x_poly_rev,
                (0, self.poly_length - self.x_poly_rev.shape[0]),
                'constant',
                constant_values=0,
            )
            self.y_poly_fwd = np.pad(
                self.y_poly_fwd,
                (0, self.poly_length - self.y_poly_fwd.shape[0]),
                'constant',
                constant_values=0,
            )
            self.y_poly_rev = np.pad(
                self.y_poly_rev,
                (0, self.poly_length - self.y_poly_rev.shape[0]),
                'constant',
                constant_values=0,
            )

        # Cut the array if it's too long
        if self.x_poly_fwd.shape[0] > self.poly_length:
            self.x_poly_fwd = self.x_poly_fwd[: self.poly_length]
            self.x_poly_rev = self.x_poly_rev[: self.poly_length]
            self.y_poly_fwd = self.y_poly_fwd[: self.poly_length]
            self.y_poly_rev = self.y_poly_rev[: self.poly_length]

    def loadFromDict(self, platepar_dict, use_flat=None):
        """Load the platepar from a dictionary."""

        # Parse JSON into an object with attributes corresponding to dict keys
        self.__dict__ = platepar_dict

        # Add the version if it was not in the platepar (v1 platepars didn't have a version)
        if not 'version' in self.__dict__:
            self.version = 1

        # If the refraction was not used for the fit, assume it is disabled
        if not 'refraction' in self.__dict__:
            self.refraction = False

        # If the measurement correction for refraction (if it was not taken into account during calibration)
        #   is not present, assume it's false
        if not 'measurement_apparent_to_true_refraction' in self.__dict__:
            self.measurement_apparent_to_true_refraction = False

        # Add equal aspect
        if not 'equal_aspect' in self.__dict__:
            self.equal_aspect = False

        # Add asymmetry correction
        if not 'asymmetry_corr' in self.__dict__:
            self.asymmetry_corr = False

        # Add forcing distortion centre to image center
        if not 'force_distortion_centre' in self.__dict__:
            self.force_distortion_centre = False

        # Add the distortion type if not present (assume it's the polynomal type with the radial term)
        if not 'distortion_type' in self.__dict__:

            # Check if the variable with the typo was used and correct it
            if 'distortion_type' in self.__dict__:
                self.distortion_type = self.distortion_type
                del self.distortion_type

            # Otherwise, assume the polynomial type
            else:
                self.distortion_type = "poly3+radial"

        # Add UT correction if it was not in the platepar
        if not 'UT_corr' in self.__dict__:
            self.UT_corr = 0

        # Add the gamma if it was not in the platepar
        if not 'gamma' in self.__dict__:
            self.gamma = 1.0

        # Add the vignetting coefficient if it was not in the platepar
        if not 'vignetting_coeff' in self.__dict__:
            self.vignetting_coeff = None

            # Add the default vignetting coeff
            self.addVignettingCoeff(use_flat=use_flat)

        # Limit the vignetting coefficient in case a too high value was set
        if self.vignetting_coeff is not None:

            self.vignetting_coeff = RMS.Astrometry.ApplyAstrometry.limitVignettingCoefficient(
                self.X_res, self.Y_res, self.vignetting_coeff
                )

        # Add keeping the vignetting coefficient fixed
        if not 'vignetting_fixed' in self.__dict__:
            self.vignetting_fixed = False

        # Add extinction scale
        if not 'extinction_scale' in self.__dict__:
            self.extinction_scale = 1.0

        # Add the list of calibration stars if it was not in the platepar
        if not 'star_list' in self.__dict__:
            self.star_list = []

        # If v1 only the backward distortion coeffs were fitted, so use load them for both forward and
        #   reverse if nothing else is available
        if not 'x_poly_fwd' in self.__dict__:

            self.x_poly_fwd = np.array(self.x_poly)
            self.x_poly_rev = np.array(self.x_poly)
            self.y_poly_fwd = np.array(self.y_poly)
            self.y_poly_rev = np.array(self.y_poly)

        # Convert lists to numpy arrays
        self.x_poly_fwd = np.array(self.x_poly_fwd)
        self.x_poly_rev = np.array(self.x_poly_rev)
        self.y_poly_fwd = np.array(self.y_poly_fwd)
        self.y_poly_rev = np.array(self.y_poly_rev)

        # Set the distortion type
        self.setDistortionType(self.distortion_type, reset_params=False)

        # Set polynomial parameters used by the old code
        self.x_poly = self.x_poly_fwd
        self.y_poly = self.y_poly_fwd

        # Add rotation from horizontal
        if not 'rotation_from_horiz' in self.__dict__:
            self.rotation_from_horiz = RMS.Astrometry.ApplyAstrometry.rotationWrtHorizon(self)

        # Calculate the datetime
        self.time = jd2Date(self.JD, dt_obj=True)

    def read(self, file_name, fmt=None, use_flat=None):
        """Read the platepar.

        Arguments:
            file_name: [str] Path and the name of the platepar to read.
        Keyword arguments:
            fmt: [str] Format of the platepar file. 'json' for JSON format and 'txt' for the usual CMN textual
                format.
            use_flat: [bool] Indicates wheter a flat is used or not. None by default.
        Return:
            fmt: [str]
        """

        # Check if platepar exists
        if not os.path.isfile(file_name):
            return False

        # Determine the type of the platepar if it is not given
        if fmt is None:

            with open(file_name) as f:
                data = " ".join(f.readlines())

                # Try parsing the file as JSON
                try:
                    json.loads(data)
                    fmt = 'json'

                except:
                    fmt = 'txt'

        # Load the file as JSON
        if fmt == 'json':

            # Load the JSON file
            with open(file_name) as f:
                data = " ".join(f.readlines())

            # Load the platepar from the JSON dictionary
            self.loadFromDict(json.loads(data), use_flat=use_flat)

        # Load the file as TXT (old CMN format)
        else:

            with open(file_name) as f:

                self.UT_corr = 0
                self.gamma = 1.0
                self.star_list = []

                # Parse latitude, longitude, elevation
                self.lon, self.lat, self.elev = self.parseLine(f)

                # Parse date and time as int
                D, M, Y, h, m, s = map(int, f.readline().split())

                # Calculate the datetime of the platepar time
                self.time = datetime.datetime(Y, M, D, h, m, s)

                # Convert time to JD
                self.JD = date2JD(Y, M, D, h, m, s)

                # Calculate the reference hour angle
                T = (self.JD - 2451545.0) / 36525.0
                self.Ho = (
                    280.46061837
                    + 360.98564736629 * (self.JD - 2451545.0)
                    + 0.000387933 * T ** 2
                    - T ** 3 / 38710000.0
                ) % 360

                # Parse camera parameters
                self.X_res, self.Y_res, self.focal_length = self.parseLine(f)

                # Parse the right ascension of the image centre
                self.RA_d, self.RA_H, self.RA_M, self.RA_S = self.parseLine(f)

                # Parse the declination of the image centre
                self.dec_d, self.dec_D, self.dec_M, self.dec_S = self.parseLine(f)

                # Parse the rotation parameter
                self.pos_angle_ref = self.parseLine(f)[0]

                # Parse the image scale (convert from arcsec/px to px/deg)
                self.F_scale = self.parseLine(f)[0]
                self.F_scale = 3600 / self.F_scale

                # Load magnitude slope parameters
                self.mag_0, self.mag_lev = self.parseLine(f)

                # Load X axis polynomial parameters
                self.x_poly_fwd = self.x_poly_rev = np.zeros(shape=(self.poly_length,), dtype=np.float64)
                for i in range(self.poly_length):
                    self.x_poly_fwd[i] = self.x_poly_fwd[i] = self.parseLine(f)[0]

                # Load Y axis polynomial parameters
                self.y_poly_fwd = self.y_poly_rev = np.zeros(shape=(self.poly_length,), dtype=np.float64)
                for i in range(self.poly_length):
                    self.y_poly_fwd[i] = self.y_poly_rev[i] = self.parseLine(f)[0]

                # Read station code
                self.station_code = f.readline().replace('\r', '').replace('\n', '')

        # Add a default vignetting coefficient if it already doesn't exist
        self.addVignettingCoeff(use_flat)

        return fmt

    def jsonStr(self):
        """Returns the JSON representation of the platepar as a string."""

        # Make a copy of the platepar object, which will be modified for writing
        self2 = copy.deepcopy(self)

        # Convert numpy arrays to list, which can be serialized
        self2.x_poly_fwd = self.x_poly_fwd.tolist()
        self2.x_poly_rev = self.x_poly_rev.tolist()
        self2.y_poly_fwd = self.y_poly_fwd.tolist()
        self2.y_poly_rev = self.y_poly_rev.tolist()
        del self2.time

        # For compatibility with old procedures, write the forward distortion parameters as x, y
        self2.x_poly = self.x_poly_fwd.tolist()
        self2.y_poly = self.y_poly_fwd.tolist()

        out_str = json.dumps(self2, default=lambda o: o.__dict__, indent=4, sort_keys=True)

        return out_str

    def write(self, file_path, fmt=None, fov=None, ret_written=False):
        """Write platepar to file.

        Arguments:
            file_path: [str] Path and the name of the platepar to write.
        Keyword arguments:
            fmt: [str] Format of the platepar file. 'json' for JSON format and 'txt' for the usual CMN textual
                format. The format is JSON by default.
            fov: [tuple] Tuple of horizontal and vertical FOV size in degree. None by default.
            ret_written: [bool] If True, the JSON string of the platepar instead of writing it to disk.
        Return:
            fmt: [str] Platepar format.
        """

        # If the FOV size was given, store it
        if fov is not None:
            self.fov_h, self.fov_v = fov

        # Set JSON to be the defualt format
        if fmt is None:
            fmt = 'json'

        # If the format is JSON, write a JSON file
        if fmt == 'json':

            out_str = self.jsonStr()

            with open(file_path, 'w') as f:
                f.write(out_str)

            if ret_written:
                return fmt, out_str

        # Old CMN format
        else:

            with open(file_path, 'w') as f:

                # Write geo coords
                f.write('{:9.6f} {:9.6f} {:04d}\n'.format(self.lon, self.lat, int(self.elev)))

                # Calculate reference time from reference JD
                Y, M, D, h, m, s, ms = list(map(int, jd2Date(self.JD)))

                # Write the reference time
                f.write('{:02d} {:02d} {:04d} {:02d} {:02d} {:02d}\n'.format(D, M, Y, h, m, s))

                # Write resolution and focal length
                f.write('{:d} {:d} {:f}\n'.format(int(self.X_res), int(self.Y_res), self.focal_length))

                # Write reference RA
                self.RA_H = int(self.RA_d / 15)
                self.RA_M = int((self.RA_d / 15 - self.RA_H) * 60)
                self.RA_S = int(((self.RA_d / 15 - self.RA_H) * 60 - self.RA_M) * 60)

                f.write("{:7.3f} {:02d} {:02d} {:02d}\n".format(self.RA_d, self.RA_H, self.RA_M, self.RA_S))

                # Write reference Dec
                self.dec_D = int(self.dec_d)
                self.dec_M = int((self.dec_d - self.dec_D) * 60)
                self.dec_S = int(((self.dec_d - self.dec_D) * 60 - self.dec_M) * 60)

                f.write(
                    "{:+7.3f} {:02d} {:02d} {:02d}\n".format(self.dec_d, self.dec_D, self.dec_M, self.dec_S)
                )

                # Write rotation parameter
                f.write('{:<7.3f}\n'.format(self.pos_angle_ref))

                # Write F scale
                f.write('{:<5.1f}\n'.format(3600 / self.F_scale))

                # Write magnitude fit
                f.write("{:.3f} {:.3f}\n".format(self.mag_0, self.mag_lev))

                # Write X distortion polynomial
                for x_elem in self.x_poly_fwd:
                    f.write('{:+E}\n'.format(x_elem))

                # Write y distortion polynomial
                for y_elem in self.y_poly_fwd:
                    f.write('{:+E}\n'.format(y_elem))

                # Write station code
                f.write(str(self.station_code) + '\n')

            if ret_written:
                with open(file_path) as f:
                    out_str = "\n".join(f.readlines())

                return fmt, out_str

        return fmt

    def updateRefAltAz(self):
        """Update the reference apparent azimuth and altitude from the reference RA and Dec."""

        # Compute reference Alt/Az to apparent coordinates, epoch of date
        az_centre, alt_centre = cyTrueRaDec2ApparentAltAz(
            np.radians(self.RA_d),
            np.radians(self.dec_d),
            self.JD,
            np.radians(self.lat),
            np.radians(self.lon),
            self.refraction,
        )
        self.az_centre, self.alt_centre = np.degrees(az_centre), np.degrees(alt_centre)

        # Update the rotation wrt horizon
        self.rotation_from_horiz = RMS.Astrometry.ApplyAstrometry.rotationWrtHorizon(self)

    def updateRefRADec(self, skip_rot_update=False, preserve_rotation=False):
        """Update the reference RA and Dec (true in J2000) from Alt/Az (apparent in epoch of date)."""

        if (not skip_rot_update) and (not preserve_rotation):

            # Save the current rotation w.r.t horizon value
            self.rotation_from_horiz = RMS.Astrometry.ApplyAstrometry.rotationWrtHorizon(self)

        # Convert the reference apparent Alt/Az in the epoch of date to true RA/Dec in J2000
        ra, dec = cyApparentAltAz2TrueRADec(
            np.radians(self.az_centre),
            np.radians(self.alt_centre),
            self.JD,
            np.radians(self.lat),
            np.radians(self.lon),
            self.refraction,
        )

        # Assign the computed RA/Dec to platepar
        self.RA_d = np.degrees(ra)
        self.dec_d = np.degrees(dec)

        if not skip_rot_update:

            # Update the position angle so that the rotation wrt horizon doesn't change
            self.pos_angle_ref = RMS.Astrometry.ApplyAstrometry.rotationWrtHorizonToPosAngle(
                self, self.rotation_from_horiz
            )

    def switchToGroundPicks(self):
        """Switch the reference pointing so that points on the ground may be correctly measured."""

        # If the refraction was on, turn if off and correct the centre
        if self.refraction:

            self.refraction = False

            # Preserve the reference elevation of the pointing as the apparent pointing
            # self.alt_centre = np.degrees(pyRefractionTrueToApparent(np.radians(self.alt_centre)))

            self.updateRefRADec(preserve_rotation=True)


    def rotationWrtHorizon(self):
        """Return the rotation of the camera wrt horizon in degrees."""

        # Compute the rotation of the camera wrt horizon
        return RMS.Astrometry.ApplyAstrometry.rotationWrtHorizon(self)

    def __repr__(self):

        # Compute alt/az pointing
        azim, elev = trueRaDec2ApparentAltAz(
            self.RA_d, self.dec_d, self.JD, self.lat, self.lon, refraction=self.refraction
        )

        out_str = "Platepar\n"
        out_str += "--------\n"
        out_str += "Camera info:\n"
        out_str += "    Lat (+N)  = {:+11.6f} deg\n".format(self.lat)
        out_str += "    Lon (+E)  = {:+11.6f} deg\n".format(self.lon)
        out_str += "    Ele (MSL) = {:11.2f} m\n".format(self.elev)
        out_str += "    FOV       = {:6.2f} x {:6.2f} deg\n".format(
            *RMS.Astrometry.ApplyAstrometry.computeFOVSize(self)
        )
        out_str += "    Img res   = {:6d} x {:6d} px\n".format(self.X_res, self.Y_res)
        out_str += "Reference pointing - equatorial (J2000):\n"
        out_str += "    JD      = {:.10f} \n".format(self.JD)
        out_str += "    RA      = {:11.6f} deg\n".format(self.RA_d)
        out_str += "    Dec     = {:+11.6f} deg\n".format(self.dec_d)
        out_str += "    Pos ang = {:.6f} deg\n".format(self.pos_angle_ref)
        out_str += "    Pix scl = {:.2f} arcmin/px\n".format(60 / self.F_scale)
        out_str += "Reference pointing - apparent azimuthal (date):\n"
        out_str += "    Azim    = {:.6f} deg (+E of N)\n".format(azim)
        out_str += "    Alt     = {:.6f} deg\n".format(elev)
        out_str += "    Rot/hor = {:.6f} deg\n".format(
            RMS.Astrometry.ApplyAstrometry.rotationWrtHorizon(self)
        )
        out_str += "Distortion:\n"
        out_str += "    Type = {:s}\n".format(self.distortion_type)

        # If the polynomial is used, the X axis parameters are stored in x_poly, otherwise radials paramters
        #   are used
        if self.distortion_type.startswith("poly"):
            out_str += "    Distortion coeffs (polynomial):\n"
            dist_string = "X"

            # Poly parameters for printing (needed for radial which will be modified)
            x_poly_fwd_print = self.x_poly_fwd
            x_poly_rev_print = self.x_poly_rev

        # Radial coefficients
        else:
            out_str += "    Distortion coeffs (radial):\n"

            out_str += "           "
            if not self.force_distortion_centre:
                out_str += " x0 (px),  y0 (px), "

            if not self.equal_aspect:
                out_str += "aspect-1, "

            if self.asymmetry_corr:
                out_str += "      a1, a2 (deg), "

            out_str += "      k1,       k2,       k3,       k4\n"

            dist_string = ""

            x_poly_fwd_print = np.array(self.x_poly_fwd)
            x_poly_rev_print = np.array(self.x_poly_rev)

            if not self.force_distortion_centre:

                # Report x0 and y0 in px (unnormalize and wrap)
                x_poly_fwd_print[0] *= self.X_res / 2
                x_poly_fwd_print[1] *= self.Y_res / 2
                x_poly_rev_print[0] *= self.X_res / 2
                x_poly_rev_print[1] *= self.Y_res / 2
                x_poly_fwd_print[0] = (
                    -self.X_res / 2.0 + (x_poly_fwd_print[0] + self.X_res / 2.0) % self.X_res
                )
                x_poly_fwd_print[1] = (
                    -self.Y_res / 2.0 + (x_poly_fwd_print[1] + self.Y_res / 2.0) % self.Y_res
                )
                x_poly_rev_print[0] = (
                    -self.X_res / 2.0 + (x_poly_rev_print[0] + self.X_res / 2.0) % self.X_res
                )
                x_poly_rev_print[1] = (
                    -self.Y_res / 2.0 + (x_poly_rev_print[1] + self.Y_res / 2.0) % self.Y_res
                )

            # Convert the asymmetry correction parameter to degrees
            if self.asymmetry_corr:
                asym_ang_index = 4

                if self.force_distortion_centre:
                    asym_ang_index -= 2

                if self.equal_aspect:
                    asym_ang_index -= 1

                x_poly_fwd_print[asym_ang_index] = np.degrees(
                    (2 * np.pi * x_poly_fwd_print[asym_ang_index]) % (2 * np.pi)
                )
                x_poly_rev_print[asym_ang_index] = np.degrees(
                    (2 * np.pi * x_poly_rev_print[asym_ang_index]) % (2 * np.pi)
                )

        out_str += "img2sky {:s} = {:s}\n".format(
            dist_string,
            ", ".join(
                ["{:+8.3f}".format(c) if abs(c) > 10e-4 else "{:+8.1e}".format(c) for c in x_poly_fwd_print]
            ),
        )
        out_str += "sky2img {:s} = {:s}\n".format(
            dist_string,
            ", ".join(
                ["{:+8.3f}".format(c) if abs(c) > 10e-4 else "{:+8.1e}".format(c) for c in x_poly_rev_print]
            ),
        )

        # Only print the rest if the polynomial fit is used
        if self.distortion_type.startswith("poly"):
            out_str += "img2sky Y = {:s}\n".format(
                ", ".join(
                    [
                        "{:+8.3f}".format(c) if abs(c) > 10e-4 else "{:+8.1e}".format(c)
                        for c in self.y_poly_fwd
                    ]
                )
            )
            out_str += "sky2img Y = {:s}\n".format(
                ", ".join(
                    [
                        "{:+8.3f}".format(c) if abs(c) > 10e-4 else "{:+8.1e}".format(c)
                        for c in self.y_poly_rev
                    ]
                )
            )

        return out_str


if __name__ == "__main__":

    import argparse

    # Init argument parser
    arg_parser = argparse.ArgumentParser(
        description="Test the astrometry functions using the given platepar."
    )
    arg_parser.add_argument('platepar_path', metavar='PLATEPAR', type=str, help='Path to the platepar file')

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    # Load the platepar file
    pp = Platepar()
    pp.read(cml_args.platepar_path)

    # Try with using standard coordinates by resetting the distortion coeffs
    pp.resetDistortionParameters()

    # # Reset distortion fit (forward and reverse)
    # pp.x_poly_fwd = np.zeros(shape=(12,), dtype=np.float64)
    # pp.y_poly_fwd = np.zeros(shape=(12,), dtype=np.float64)
    # pp.x_poly_rev = np.zeros(shape=(12,), dtype=np.float64)
    # pp.y_poly_rev = np.zeros(shape=(12,), dtype=np.float64)

    print(pp)

    # Try forward and reverse mapping, and compare results
    for i in range(5):

        # Randomly generate a pick inside the image
        x_img = np.random.uniform(0, pp.X_res)
        y_img = np.random.uniform(0, pp.Y_res)

        # Take current time
        time_data = [2020, 5, 30, 1, 20, 34, 567]

        # Map to RA/Dec
        jd_data, ra_data, dec_data, _ = RMS.Astrometry.ApplyAstrometry.xyToRaDecPP(
            [time_data], [x_img], [y_img], [1], pp, extinction_correction=False
        )

        # Map back to X, Y
        x_data, y_data = RMS.Astrometry.ApplyAstrometry.raDecToXYPP(ra_data, dec_data, jd_data[0], pp)

        # Map forward to sky again
        _, ra_data_rev, dec_data_rev, _ = RMS.Astrometry.ApplyAstrometry.xyToRaDecPP(
            [time_data], x_data, y_data, [1], pp, extinction_correction=False
        )

        print()
        print("-----------------------")
        print("Init image coordinates:")
        print("X = {:.3f}".format(x_img))
        print("Y = {:.3f}".format(y_img))
        print("Sky coordinates:")
        print("RA  = {:.4f}".format(ra_data[0]))
        print("Dec = {:+.4f}".format(dec_data[0]))
        print("Reverse image coordinates:")
        print("X = {:.3f}".format(x_data[0]))
        print("Y = {:.3f}".format(y_data[0]))
        print("Reverse sky coordinates:")
        print("RA  = {:.4f}".format(ra_data_rev[0]))
        print("Dec = {:+.4f}".format(dec_data_rev[0]))
        print("Image diff:")
        print("X = {:.3f}".format(x_img - x_data[0]))
        print("Y = {:.3f}".format(y_img - y_data[0]))
        print("Sky diff:")
        print("RA  = {:.4f} amin".format(60 * (ra_data[0] - ra_data_rev[0])))
        print("Dec = {:+.4f} amin".format(60 * (dec_data[0] - dec_data_rev[0])))
