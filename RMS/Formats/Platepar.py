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
from scipy.spatial import cKDTree
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
        (x_array, y_array mag_catalog): [tuple of ndarrays] X, Y positions and magnitudes of stars on the
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
        len(img_x) * [img_time], img_x, img_y, len(img_x) * [1], platepar, extinction_correction=False,
        precompute_pointing_corr=True
    )

    return ra_array, dec_array


class Platepar(object):
    def __init__(self, distortion_type="radial7-odd"):
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

        # Store distortion type for later initialization
        self._init_distortion_type = distortion_type

        # Initialize distortion-related attributes (needed by setDistortionType)
        self.equal_aspect = True
        self.force_distortion_centre = False
        self.asymmetry_corr = True

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

        # Note: equal_aspect, force_distortion_centre, asymmetry_corr are initialized earlier
        # (before setDistortionType is called)

        # Photometry calibration
        self.mag_0 = -2.5
        self.mag_lev = 1.0
        self.mag_lev_stddev = 0.0
        self.gamma = 1.0
        self.vignetting_coeff = None  # Will be set to resolution-scaled default by addVignettingCoeff
        self.vignetting_fixed = True

        # Extinction correction scaling
        self.extinction_scale = 0.6

        self.station_code = "None"

        self.star_list = None

        # Flag to indicate that the platepar was refined with CheckFit
        self.auto_check_fit_refined = False

        # Flag to indicate that the platepar was successfully auto recalibrated on an individual FF files
        self.auto_recalibrated = False

        # Initialize distortion type (must be done before resetDistortionParameters)
        # First set a placeholder value so setDistortionType can compare against it
        self.distortion_type = ""
        self.poly_length = 0

        # Now set the actual distortion type
        self.setDistortionType(self._init_distortion_type, reset_params=True)
        del self._init_distortion_type

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

                # If force_distortion_centre is True, there are no centre coefficients to preserve
                if self.force_distortion_centre:
                    x_centre_fwd, y_centre_fwd = 0.0, 0.0
                    x_centre_rev, y_centre_rev = 0.0, 0.0
                else:
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

        # Lengths of full polynomials, (including distortion center, aspect, and asymmetry correction for
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

                # Use 0.001 rad/px as the default coefficient, as that's the one for 3.6 mm f/0.95 and 16 mm
                #   f/1.0 lenses. The vignetting coeff is dependent on the resolution, the default value of
                #   0.001 rad/px is for 720p.
                self.vignetting_coeff = 0.001 * np.hypot(1280, 720) / np.hypot(self.X_res, self.Y_res)

    def getDistortionCentre(self):
        """Get the distortion center (optical axis) in pixel coordinates.

        For radial distortion models with force_distortion_centre=False, the distortion center
        is fitted and stored in x_poly_fwd[0] and x_poly_fwd[1] as normalized offsets.
        Otherwise, the distortion center is at the image center.

        Returns:
            (x_centre, y_centre): [tuple of floats] Distortion center in pixel coordinates.
        """

        # If distortion center is forced to image center, or not using radial distortion
        if self.force_distortion_centre or not self.distortion_type.startswith("radial"):
            return self.X_res / 2.0, self.Y_res / 2.0

        # For radial distortion, extract the center offsets from coefficients
        # The offsets are stored normalized by X_res/2 and Y_res/2
        x0 = self.x_poly_fwd[0] * (self.X_res / 2.0)
        y0 = self.x_poly_fwd[1] * (self.Y_res / 2.0)

        # Wrap offsets to always be within the image bounds
        x0 = -self.X_res / 2.0 + (x0 + self.X_res / 2.0) % self.X_res
        y0 = -self.Y_res / 2.0 + (y0 + self.Y_res / 2.0) % self.Y_res

        # Return center in pixel coordinates (image center + offset)
        return self.X_res / 2.0 + x0, self.Y_res / 2.0 + y0

    def fitPointing(self, jd, img_stars, catalog_stars, fixed_scale=False):
        """Fit pointing parameters to the list of star image and celestial catalog coordinates.
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

    def fitPointingNN(self, jd, img_stars, catalog_stars, fixed_scale=True):
        """Fit pointing parameters using nearest-neighbor cost function in pixel space.

        Unlike fitPointing() which requires pre-matched star pairs, this method finds the
        nearest catalog star for each detected star and optimizes to minimize the sum of
        NN distances. This is more robust when the initial pointing is off.

        Use this as a replacement for NNalign - it fits RA_d, dec_d, pos_angle_ref, and
        optionally F_scale without requiring explicit star matching.

        The cost function works in pixel space (like matchStarsResiduals) by projecting
        catalog stars onto the image and computing pixel distances. This is more robust
        than working in sky coordinates because it correctly handles distortion and timing.

        Arguments:
            jd: [float] Julian date of the image.
            img_stars: [ndarray] Detected stars as Nx3 array (x, y, intensity).
            catalog_stars: [ndarray] Catalog stars in FOV as Mx3 array (ra, dec, mag).

        Keyword arguments:
            fixed_scale: [bool] Keep scale fixed. True by default (for camera drift correction).

        """

        from RMS.Astrometry.ApplyAstrometry import raDecToXYPP

        # Create a single working copy of platepar to reuse in cost function
        # This avoids expensive deepcopy on every optimizer iteration
        pp_work = copy.deepcopy(self)

        # CRITICAL: Update the platepar's reference time to match the observation time.
        # The coordinate transformation uses (JD, Ho) as the reference frame. If we don't
        # update these to match the observation JD, changing RA_d won't have the expected
        # effect on the catalog star projections.
        #
        # When we change JD/Ho, we must also adjust RA_d to maintain the same pointing direction.
        # The relationship is: new_RA_d = old_RA_d + delta_Ho (sidereal rotation)
        old_Ho = pp_work.Ho

        pp_work.JD = jd
        T = (jd - 2451545.0) / 36525.0
        new_Ho = (
            280.46061837
            + 360.98564736629 * (jd - 2451545.0)
            + 0.000387933 * T ** 2
            - T ** 3 / 38710000.0
        ) % 360
        pp_work.Ho = new_Ho

        # Compute delta Ho and adjust RA_d to preserve pointing direction
        delta_Ho = (new_Ho - old_Ho) % 360
        pp_work.RA_d = (self.RA_d + delta_Ho) % 360

        # Pre-extract catalog coordinates (constant across iterations)
        ra_catalog, dec_catalog, _ = catalog_stars.T

        # Pre-extract detected star positions (constant across iterations)
        img_x, img_y, _ = img_stars.T

        def _calcPointingNNCostPixel(params, pp_work, jd, ra_catalog, dec_catalog, img_x, img_y, fixed_scale):
            """NN cost function in pixel space for pointing fit.

            Projects catalog stars to image coordinates and computes pixel-space NN distances.
            This is more robust than sky-space comparison because it correctly handles
            distortion and timing through the standard coordinate transforms.

            Uses a pre-allocated working platepar to avoid deepcopy overhead.
            """

            # Update working platepar with current parameters (no copy needed)
            pp_work.RA_d, pp_work.dec_d, pp_work.pos_angle_ref = params[:3]
            if not fixed_scale:
                pp_work.F_scale = abs(params[3])

            # Project catalog stars to image coordinates
            cat_x, cat_y = raDecToXYPP(ra_catalog, dec_catalog, jd, pp_work)

            # Filter out catalog stars that project outside the image
            valid_mask = (cat_x >= 0) & (cat_x < pp_work.X_res) & (cat_y >= 0) & (cat_y < pp_work.Y_res)
            cat_x_valid = cat_x[valid_mask]
            cat_y_valid = cat_y[valid_mask]

            if len(cat_x_valid) < 3:
                return 1e10  # Return large cost if too few valid catalog stars

            # Use KD-tree for fast nearest-neighbor search (O(N log M) vs O(N*M))
            cat_coords = np.column_stack([cat_x_valid, cat_y_valid])
            tree = cKDTree(cat_coords)
            img_coords = np.column_stack([img_x, img_y])
            nn_distances, _ = tree.query(img_coords, k=1)

            # Use RMSD (root mean square deviation) as cost
            # This normalizes by number of stars and gives interpretable units (pixels)
            rmsd = np.sqrt(np.mean(nn_distances ** 2))

            return rmsd

        # Initial parameters - use pp_work.RA_d which has been adjusted for the new JD/Ho
        p0 = [pp_work.RA_d, self.dec_d, self.pos_angle_ref]
        if not fixed_scale:
            p0.append(abs(self.F_scale))

        # Debug: compute initial cost (RMSD in pixels)
        initial_cost = _calcPointingNNCostPixel(p0, pp_work, jd, ra_catalog, dec_catalog, img_x, img_y, fixed_scale)
        print("    fitPointingNN: BEFORE RA={:.2f} Dec={:.2f} Rot={:.2f} RMSD={:.2f} px".format(
            p0[0], p0[1], p0[2], initial_cost))

        # Fit using Nelder-Mead (robust for NN cost landscape)
        res = scipy.optimize.minimize(
            _calcPointingNNCostPixel,
            p0,
            args=(pp_work, jd, ra_catalog, dec_catalog, img_x, img_y, fixed_scale),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'adaptive': True},
        )

        # Debug: show result (RMSD in pixels)
        print("    fitPointingNN: AFTER  RA={:.2f} Dec={:.2f} Rot={:.2f} RMSD={:.2f} px success={}".format(
            res.x[0], res.x[1], res.x[2], res.fun, res.success))

        # Update fitted parameters
        self.RA_d, self.dec_d, self.pos_angle_ref = res.x[:3]
        if not fixed_scale:
            self.F_scale = abs(res.x[3])

        # Update JD and Ho to match the observation time
        # This ensures the fitted RA_d is consistent with the new reference time
        self.JD = jd
        T = (jd - 2451545.0) / 36525.0
        self.Ho = (
            280.46061837
            + 360.98564736629 * (jd - 2451545.0)
            + 0.000387933 * T ** 2
            - T ** 3 / 38710000.0
        ) % 360

        # Update alt/az of pointing
        self.updateRefAltAz()

        return res.success

    def fitAstrometry(
        self,
        jd,
        img_stars,
        catalog_stars,
        first_platepar_fit=False,
        fit_only_pointing=False,
        fixed_scale=False,
        use_nn_cost=False,
    ):
        """Fit astrometric parameters to the list of star image and celestial catalog coordinates.
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
                coordinates with the given astrometrical solution. Pointing and distortion parameters are used
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
            coordinates with the given astrometrical solution. Pointing and distortion parameters are used
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
            pp_copy.x_poly_fwd = np.array(params[4:])

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

        def _calcSkyResidualsAstroAndDistortionRadialNN(params, platepar, jd, catalog_stars, img_stars):
            """Like _calcSkyResidualsAstroAndDistortionRadial but uses nearest-neighbor matching.

            Instead of requiring pre-matched pairs, this finds the nearest catalog star for each
            detected star and sums those distances. This eliminates the need for explicit matching.

            Args:
                params: [RA_d/360, dec_d/90, pos_angle_ref/360, F_scale, x_poly_fwd...]
                platepar: Platepar object
                jd: Julian date
                catalog_stars: Pre-filtered catalog stars in FOV (Mx3 array [ra, dec, mag])
                img_stars: detected image stars (Nx3 array [x, y, intensity])

            Returns:
                Sum of squared angular separations to nearest catalog star for each detected star
            """
            # Set distortion parameters
            pp_copy = copy.deepcopy(platepar)

            # Unpack pointing parameters and assign to the copy of platepar used for the fit
            ra_ref, dec_ref, pos_angle_ref, F_scale = params[:4]

            # Unnormalize the pointing parameters
            pp_copy.RA_d = (360 * ra_ref) % (360)
            pp_copy.dec_d = -90 + (90 * dec_ref + 90) % (180.000001)
            pp_copy.pos_angle_ref = (360 * pos_angle_ref) % (360)
            pp_copy.F_scale = abs(F_scale)

            # Assign distortion parameters
            pp_copy.x_poly_fwd = np.array(params[4:])

            img_x, img_y, _ = img_stars.T

            # Convert detected image positions to sky coordinates
            ra_det, dec_det = getPairedStarsSkyPositions(img_x, img_y, jd, pp_copy)

            # Catalog is pre-filtered to FOV by caller - no need to re-filter here
            ra_catalog, dec_catalog, _ = catalog_stars.T

            # Vectorized NN: compute NxM angular separation matrix, then take row-wise min
            ra_det_rad = np.radians(ra_det)[:, np.newaxis]  # (N, 1)
            dec_det_rad = np.radians(dec_det)[:, np.newaxis]  # (N, 1)
            ra_cat_rad = np.radians(ra_catalog)[np.newaxis, :]  # (1, M)
            dec_cat_rad = np.radians(dec_catalog)[np.newaxis, :]  # (1, M)

            # angularSeparation broadcasts to (N, M) separation matrix
            sep_matrix = angularSeparation(ra_det_rad, dec_det_rad, ra_cat_rad, dec_cat_rad)
            nn_distances = np.min(sep_matrix, axis=1)  # (N,)
            total_cost = np.sum(nn_distances ** 2)

            return total_cost

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
        #   Skip when using NN cost since pointing is included in the radial fit
        if (
            self.distortion_type.startswith("poly")
            or (not self.distortion_type.startswith("poly") and first_platepar_fit)
            or fit_only_pointing
        ) and not use_nn_cost:

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

                # Extract fitted X polynomial
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

                # If this is the first fit of the distortion, set the forward parameters to be equal to the reverse
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

                # IMPORTANT NOTE - the X polynomial is used to store the fit parameters
                self.y_poly_fwd = res.x

                ### ###

            # Fit radial distortion (+ pointing)
            else:

                ### FORWARD MAPPING FIT ###

                # # Fit the radial distortion - the X polynomial is used to store the fit parameters
                # res = scipy.optimize.minimize(_calcSkyResidualsDistortion, self.x_poly_fwd, \
                #     args=(self, jd, catalog_stars, img_stars, 'radial'), method='Nelder-Mead', \
                #     options={'maxiter': 10000, 'adaptive': True})

                # # Extract distortion parameters, IMPORTANT NOTE - the X polynomial is used to store the
                # #   fit parameters
                # self.x_poly_fwd = res.x

                # Fitting the pointing direction below! - if used, it should be put BEFORE the reverse fit!
                # Initial parameters for the pointing and distortion fit (normalize to the 0-1 range)
                p0 = [self.RA_d / 360, self.dec_d / 90, self.pos_angle_ref / 360, abs(self.F_scale)]
                p0 += self.x_poly_fwd.tolist()

                # Choose cost function based on use_nn_cost flag
                if use_nn_cost:
                    # Use nearest-neighbor cost function - no explicit star matching needed
                    cost_func = _calcSkyResidualsAstroAndDistortionRadialNN
                else:
                    # Use original matched-pair cost function
                    cost_func = _calcSkyResidualsAstroAndDistortionRadial

                # Fit the radial distortion - the X polynomial is used to store the fit parameters
                # Tiered tolerances: RANSAC uses looser tolerances, final fit uses tighter
                opt_options_final = {'maxiter': 5000, 'fatol': 1e-8, 'adaptive': True}
                opt_options_ransac_r3 = {'maxiter': 700, 'fatol': 1e-5, 'adaptive': True}  # radial3-odd: rough
                opt_options_ransac_r5 = {'maxiter': 1000, 'fatol': 1e-7, 'adaptive': True}  # radial5-odd
                opt_options_ransac_r7 = {'maxiter': 1200, 'fatol': 1e-7, 'adaptive': True}  # radial7-odd

                if use_nn_cost:
                    # Three-stage RANSAC-style outlier detection with radial-weighted threshold
                    # Stage 1: radial3-odd (stable, fewer params) - identifies outliers
                    # Stage 2: radial5-odd (refines distortion) - works only on Stage 1 inliers
                    # Stage 3: radial7-odd (captures edge distortion) - final refinement
                    # Outliers can't hide because they don't influence the fit

                    n_stars = len(img_stars)
                    # Smaller fraction for initial iterations to reduce outlier contamination
                    initial_subset_fraction = 0.25  # First 2 iterations: 25%
                    normal_subset_fraction = 0.5    # Remaining iterations: 50%
                    all_indices = np.arange(n_stars)

                    # Compute radial distance from image center for weighted threshold
                    # Edge stars have larger errors due to lens distortion
                    cx, cy = self.X_res / 2.0, self.Y_res / 2.0
                    img_x_all, img_y_all, _ = img_stars.T
                    r = np.sqrt((img_x_all - cx)**2 + (img_y_all - cy)**2)
                    r_max = np.sqrt(cx**2 + cy**2)  # Corner distance
                    # Scale threshold: allow 2x error at edges vs center
                    radial_scale = 1.0 + (r / r_max)  # Range [1.0, 2.0]

                    # Save original distortion type for final fit
                    original_dist_type = self.distortion_type

                    # Keep track of the best result across all stages
                    best_res = None
                    best_cost = float('inf')

                    # RANSAC outlier detection: radial3-odd -> radial5-odd -> radial7-odd with warm start
                    # radial3-odd stabilizes quickly, radial5-odd refines, radial7-odd captures edges
                    # Pre-filter catalog to current FOV for count display
                    ra_cat_all, dec_cat_all, _ = catalog_stars.T
                    cat_x_init, cat_y_init = RMS.Astrometry.ApplyAstrometry.raDecToXYPP(
                        ra_cat_all, dec_cat_all, jd, self)
                    in_fov_init = (cat_x_init >= 0) & (cat_x_init < self.X_res) & \
                                  (cat_y_init >= 0) & (cat_y_init < self.Y_res)
                    n_catalog_fov = np.sum(in_fov_init)
                    print("    NN input: {:d} detected stars, {:d} catalog stars in FOV".format(
                        len(img_stars), n_catalog_fov))

                    # Safety check: need minimum stars to proceed
                    min_stars_required = 10
                    if len(img_stars) < min_stars_required:
                        print("    -> Not enough detected stars ({} < {}), skipping NN fit".format(
                            len(img_stars), min_stars_required))
                        return None
                    if n_catalog_fov < min_stars_required:
                        print("    -> Not enough catalog stars in FOV ({} < {}), skipping NN fit".format(
                            n_catalog_fov, min_stars_required))
                        return None

                    print("    RANSAC outlier detection: 21 iterations")
                    print("    Radial-weighted threshold: 1.0x at center, 2.0x at corners")
                    print("    Iterations 1-7: radial3-odd, 8-14: radial5-odd, 15-21: radial7-odd")

                    # ========== RANSAC - identify outliers ==========
                    total_iters = 21  # 7 + 7 + 7 iterations
                    switch_iter_r5 = 7   # Switch from radial3-odd to radial5-odd after iteration 7
                    switch_iter_r7 = 14  # Switch from radial5-odd to radial7-odd after iteration 14

                    # Start with radial3-odd - convert coefficients intelligently
                    # This preserves k1 when converting from e.g. radial7-odd to radial3-odd
                    if self.distortion_type.startswith("radial") and self.distortion_type != "radial3-odd":
                        # Convert coefficients before changing distortion type
                        radial3_x, radial3_y = self.convertDistortionCoeffs("radial3-odd")
                        self.setDistortionType("radial3-odd", reset_params=False)
                        self.x_poly_fwd = radial3_x
                        self.x_poly_rev = radial3_x.copy()
                        self.y_poly_fwd = radial3_y
                        self.y_poly_rev = radial3_y.copy()
                    else:
                        self.setDistortionType("radial3-odd", reset_params=False)

                    p0_current = [
                        self.RA_d / 360.0,
                        self.dec_d / 90,
                        self.pos_angle_ref / 360.0,
                        self.F_scale,
                    ] + self.x_poly_fwd.tolist()

                    # Weighted outlier scoring: later iterations count more (fit is more refined)
                    # Outliers get +weight, inliers get -weight (redemption)
                    # Weighted scoring: radial3-odd=1, radial5-odd=2, radial7-odd=3
                    # This gives more influence to the more accurate higher-order fits
                    outlier_scores = np.zeros(n_stars, dtype=float)

                    # Track current iteration's outliers - exclude from next iteration's fit
                    current_outlier_mask = np.zeros(n_stars, dtype=bool)
                    prev_outlier_mask = None
                    stable_count = 0  # Consecutive iterations with unchanged outlier mask

                    iteration = 0
                    while iteration < total_iters:
                        # Compute weight for this iteration - higher order counts more
                        if iteration < switch_iter_r5:
                            weight = 1  # radial3-odd phase
                        elif iteration < switch_iter_r7:
                            weight = 2  # radial5-odd phase
                        else:
                            weight = 3  # radial7-odd phase - most accurate, counts most

                        # Switch to radial5-odd after iteration 7 with warm start
                        if iteration == switch_iter_r5:
                            # Save current best params from radial3-odd
                            if best_res is not None:
                                ra_ref, dec_ref, pos_angle_ref, F_scale = best_res.x[:4]
                                self.RA_d = (360 * ra_ref) % (360)
                                self.dec_d = -90 + (90 * dec_ref + 90) % (180.000001)
                                self.pos_angle_ref = (360 * pos_angle_ref) % (360)
                                self.F_scale = abs(F_scale)
                                # Update x_poly_fwd with best radial3-odd coefficients
                                self.x_poly_fwd = np.array(best_res.x[4:])
                                self.x_poly_rev = self.x_poly_fwd.copy()

                            # Convert coefficients from radial3-odd to radial5-odd intelligently
                            # This preserves x0, y0, xy, a1, a2, k1 and adds k2=0
                            radial5_x, radial5_y = self.convertDistortionCoeffs("radial5-odd")
                            self.setDistortionType("radial5-odd", reset_params=False)
                            self.x_poly_fwd = radial5_x
                            self.x_poly_rev = radial5_x.copy()
                            self.y_poly_fwd = radial5_y
                            self.y_poly_rev = radial5_y.copy()

                            p0_current = [
                                self.RA_d / 360.0,
                                self.dec_d / 90,
                                self.pos_angle_ref / 360.0,
                                self.F_scale,
                            ] + self.x_poly_fwd.tolist()
                            best_res = None  # Reset best result for new distortion type
                            best_cost = float('inf')
                            print("      --- Switching to radial5-odd (warm start with coefficient conversion) ---")

                        # Switch to radial7-odd after iteration 14 with warm start
                        elif iteration == switch_iter_r7:
                            # Save current best params from radial5-odd
                            if best_res is not None:
                                ra_ref, dec_ref, pos_angle_ref, F_scale = best_res.x[:4]
                                self.RA_d = (360 * ra_ref) % (360)
                                self.dec_d = -90 + (90 * dec_ref + 90) % (180.000001)
                                self.pos_angle_ref = (360 * pos_angle_ref) % (360)
                                self.F_scale = abs(F_scale)
                                # Update x_poly_fwd with best radial5-odd coefficients
                                self.x_poly_fwd = np.array(best_res.x[4:])
                                self.x_poly_rev = self.x_poly_fwd.copy()

                            # Convert coefficients from radial5-odd to radial7-odd intelligently
                            # This preserves x0, y0, xy, a1, a2, k1, k2 and adds k3=0
                            radial7_x, radial7_y = self.convertDistortionCoeffs("radial7-odd")
                            self.setDistortionType("radial7-odd", reset_params=False)
                            self.x_poly_fwd = radial7_x
                            self.x_poly_rev = radial7_x.copy()
                            self.y_poly_fwd = radial7_y
                            self.y_poly_rev = radial7_y.copy()

                            p0_current = [
                                self.RA_d / 360.0,
                                self.dec_d / 90,
                                self.pos_angle_ref / 360.0,
                                self.F_scale,
                            ] + self.x_poly_fwd.tolist()
                            best_res = None  # Reset best result for new distortion type
                            best_cost = float('inf')
                            print("      --- Switching to radial7-odd (warm start with coefficient conversion) ---")

                        # Exclude stars that were outliers in the previous iteration
                        # This ensures outliers don't contaminate subsequent fits
                        available_mask = ~current_outlier_mask
                        available_indices = all_indices[available_mask]

                        # Safety check: need minimum stars to fit
                        min_stars_for_fit = 5
                        if len(available_indices) < min_stars_for_fit:
                            print("      -> Not enough non-outlier stars ({} < {}), exiting RANSAC early".format(
                                len(available_indices), min_stars_for_fit))
                            break

                        # Use smaller subset for first 2 iterations to reduce outlier contamination
                        if iteration < 2:
                            iter_subset_fraction = initial_subset_fraction
                        else:
                            iter_subset_fraction = normal_subset_fraction

                        n_subset = max(10, int(n_stars * iter_subset_fraction))

                        if len(available_indices) < n_subset:
                            subset_indices = available_indices
                        else:
                            # Bias selection towards brighter stars (higher intensity)
                            # Use intensity as weight, with sqrt to reduce extreme bias
                            intensities = img_stars[available_indices, 2]
                            # Clip extreme values (satellites/planets at top, noise at bottom)
                            intensity_median = np.median(intensities)
                            intensity_clipped = np.clip(intensities,
                                                       intensity_median * 0.1,  # floor at 10% of median
                                                       intensity_median * 10)   # cap at 10x median
                            weights = np.sqrt(intensity_clipped)
                            weights = weights / weights.sum()  # normalize to probabilities

                            np.random.seed(42 + iteration)
                            subset_indices = np.random.choice(available_indices, n_subset, replace=False,
                                                             p=weights)

                        img_stars_subset = img_stars[subset_indices]

                        start_params = best_res.x if best_res is not None else p0_current
                        if len(start_params) != len(p0_current):
                            start_params = p0_current

                        # Pre-filter catalog to FOV using current pointing (from start_params)
                        # This avoids re-filtering inside every optimizer function evaluation
                        pp_filter = copy.deepcopy(self)
                        pp_filter.RA_d = (360 * start_params[0]) % 360
                        pp_filter.dec_d = -90 + (90 * start_params[1] + 90) % (180.000001)
                        pp_filter.pos_angle_ref = (360 * start_params[2]) % 360
                        pp_filter.F_scale = abs(start_params[3])
                        ra_cat_all, dec_cat_all, _ = catalog_stars.T
                        cat_x, cat_y = RMS.Astrometry.ApplyAstrometry.raDecToXYPP(
                            ra_cat_all, dec_cat_all, jd, pp_filter
                        )
                        in_fov = (cat_x >= 0) & (cat_x < self.X_res) & \
                                 (cat_y >= 0) & (cat_y < self.Y_res)
                        catalog_stars_fov = catalog_stars[in_fov]

                        # Safety check: need minimum catalog stars for NN matching
                        if len(catalog_stars_fov) < min_stars_for_fit:
                            print("      -> Not enough catalog stars in FOV ({} < {}), exiting RANSAC early".format(
                                len(catalog_stars_fov), min_stars_for_fit))
                            break

                        # Use tiered tolerances: looser for radial3-odd, tighter for radial5/7-odd
                        if iteration < switch_iter_r5:
                            ransac_opts = opt_options_ransac_r3
                        elif iteration < switch_iter_r7:
                            ransac_opts = opt_options_ransac_r5
                        else:
                            ransac_opts = opt_options_ransac_r7
                        res = scipy.optimize.minimize(
                            cost_func, start_params,
                            args=(self, jd, catalog_stars_fov, img_stars_subset),
                            method='Nelder-Mead', options=ransac_opts,
                        )

                        # Debug: show optimizer exit reason
                        exit_reason = "maxiter" if res.nit >= ransac_opts['maxiter'] else "converged"
                        print("        opt: {} iters, {} fev, {} (fatol={})".format(
                            res.nit, res.nfev, exit_reason, ransac_opts['fatol']))

                        # Score on ALL stars
                        pp_temp = copy.deepcopy(self)
                        ra_ref, dec_ref, pos_angle_ref, F_scale = res.x[:4]
                        pp_temp.RA_d = (360 * ra_ref) % (360)
                        pp_temp.dec_d = -90 + (90 * dec_ref + 90) % (180.000001)
                        pp_temp.pos_angle_ref = (360 * pos_angle_ref) % (360)
                        pp_temp.F_scale = abs(F_scale)
                        pp_temp.x_poly_fwd = np.array(res.x[4:])

                        img_x, img_y, _ = img_stars.T
                        ra_det, dec_det = getPairedStarsSkyPositions(img_x, img_y, jd, pp_temp)

                        # Re-filter catalog using current pp_temp (strict XY filter)
                        # This allows edge stars to "appear" as distortion improves
                        ra_cat_ext, dec_cat_ext, _ = catalog_stars.T
                        cat_x, cat_y = RMS.Astrometry.ApplyAstrometry.raDecToXYPP(
                            ra_cat_ext, dec_cat_ext, jd, pp_temp
                        )
                        in_fov_iter = (cat_x >= 0) & (cat_x < pp_temp.X_res) & \
                                      (cat_y >= 0) & (cat_y < pp_temp.Y_res)
                        catalog_stars_iter = catalog_stars[in_fov_iter]
                        ra_catalog, dec_catalog, _ = catalog_stars_iter.T

                        # Vectorized NN: compute NxM separation matrix
                        ra_det_rad = np.radians(ra_det)[:, np.newaxis]  # (N, 1)
                        dec_det_rad = np.radians(dec_det)[:, np.newaxis]  # (N, 1)
                        ra_cat_rad = np.radians(ra_catalog)[np.newaxis, :]  # (1, M)
                        dec_cat_rad = np.radians(dec_catalog)[np.newaxis, :]  # (1, M)
                        sep_matrix = angularSeparation(ra_det_rad, dec_det_rad, ra_cat_rad, dec_cat_rad)
                        nn_seps = np.min(sep_matrix, axis=1)  # (N,)

                        median_sep = np.median(nn_seps)
                        base_threshold = 3.0 * median_sep
                        per_star_threshold = base_threshold * radial_scale
                        iteration_outliers = nn_seps > per_star_threshold

                        # Weighted scoring: outliers get +weight, inliers get -weight (redemption!)
                        outlier_scores[iteration_outliers] += weight
                        outlier_scores[~iteration_outliers] -= weight

                        # Update current outlier mask for next iteration's exclusion
                        current_outlier_mask = iteration_outliers

                        # RMSD in arcminutes on INLIERS only (nn_seps is in radians)
                        inlier_seps = nn_seps[~iteration_outliers]
                        if len(inlier_seps) > 0:
                            rmsd_arcmin = np.degrees(np.sqrt(np.mean(inlier_seps**2))) * 60
                        else:
                            rmsd_arcmin = np.inf
                        if rmsd_arcmin < best_cost:
                            best_cost = rmsd_arcmin
                            best_res = res

                        if iteration < switch_iter_r5:
                            dist_label = "radial3-odd"
                        elif iteration < switch_iter_r7:
                            dist_label = "radial5-odd"
                        else:
                            dist_label = "radial7-odd"
                        # Debug: show RA/Dec at each iteration
                        iter_ra = (360 * res.x[0]) % 360
                        iter_dec = -90 + (90 * res.x[1] + 90) % 180.000001
                        print("      Iter {}: {} (w={}) fit on {}, {} outliers, RMSD={:.2f}', RA={:.2f} Dec={:.2f}".format(
                            iteration + 1, dist_label, weight, len(subset_indices),
                            np.sum(iteration_outliers), rmsd_arcmin, iter_ra, iter_dec))

                        # Early exit: if outlier mask unchanged for 2 consecutive iterations within phase
                        if prev_outlier_mask is not None and np.array_equal(iteration_outliers, prev_outlier_mask):
                            stable_count += 1
                            if stable_count >= 2:
                                if iteration < switch_iter_r5:
                                    # In radial3-odd phase: skip to radial5-odd phase
                                    print("      -> Outliers stable for 2 iterations, skipping to radial5-odd")
                                    iteration = switch_iter_r5  # Jump to start of radial5-odd
                                    stable_count = 0  # Reset for next phase
                                    prev_outlier_mask = None  # Reset to avoid false stability detection
                                    continue
                                elif iteration < switch_iter_r7:
                                    # In radial5-odd phase: skip to radial7-odd phase
                                    print("      -> Outliers stable for 2 iterations, skipping to radial7-odd")
                                    iteration = switch_iter_r7  # Jump to start of radial7-odd
                                    stable_count = 0  # Reset for next phase
                                    prev_outlier_mask = None  # Reset to avoid false stability detection
                                    continue
                                else:
                                    # In radial7-odd phase: exit entirely
                                    print("      -> Outliers stable for 2 iterations, exiting RANSAC")
                                    break
                        else:
                            stable_count = 0
                        prev_outlier_mask = iteration_outliers.copy()
                        iteration += 1

                    # Outlier mask: positive score = more outlier votes than inlier votes
                    # Max possible score: 7*1 + 7*2 + 7*3 = 7+14+21 = 42 (all outlier)
                    # Min possible score: -42 (all inlier)
                    final_outlier_mask = outlier_scores > 0
                    nn_inlier_mask = ~final_outlier_mask
                    n_inliers = np.sum(nn_inlier_mask)
                    n_outliers = np.sum(final_outlier_mask)

                    print("    RANSAC result: {}/{} inliers (removed {} outliers with score > 0), RMSD={:.2f}'".format(
                        n_inliers, n_stars, n_outliers, best_cost))

                    # Safety check: if RMSD is too large, bail out
                    max_rmsd_arcmin = 10.0
                    if best_cost > max_rmsd_arcmin:
                        print("    -> RMSD too large ({:.2f}' > {:.0f}'), skipping final fit".format(
                            best_cost, max_rmsd_arcmin))
                        return None

                    # Apply best RANSAC params to platepar
                    ra_ref, dec_ref, pos_angle_ref, F_scale = best_res.x[:4]
                    self.RA_d = (360 * ra_ref) % (360)
                    self.dec_d = -90 + (90 * dec_ref + 90) % (180.000001)
                    self.pos_angle_ref = (360 * pos_angle_ref) % (360)
                    self.F_scale = abs(F_scale)
                    self.x_poly_fwd = np.array(best_res.x[4:])  # Ensure numpy array
                    self.x_poly_rev = np.array(self.x_poly_fwd)  # Sync reverse with forward!
                    self.updateRefAltAz()

                    # Create matched pairs from NN assignments using RANSAC result
                    img_stars_clean = img_stars[~final_outlier_mask]
                    img_x, img_y, _ = img_stars_clean.T
                    ra_det, dec_det = getPairedStarsSkyPositions(img_x, img_y, jd, self)

                    # Re-filter catalog using final platepar (strict XY filter)
                    ra_cat_ext, dec_cat_ext, _ = catalog_stars.T
                    cat_x, cat_y = RMS.Astrometry.ApplyAstrometry.raDecToXYPP(
                        ra_cat_ext, dec_cat_ext, jd, self
                    )
                    in_fov_final = (cat_x >= 0) & (cat_x < self.X_res) & \
                                   (cat_y >= 0) & (cat_y < self.Y_res)
                    catalog_stars_fov = catalog_stars[in_fov_final]
                    ra_catalog, dec_catalog, _ = catalog_stars_fov.T

                    # Vectorized NN matching: compute NxM separation matrix
                    ra_det_rad = np.radians(ra_det)[:, np.newaxis]  # (N, 1)
                    dec_det_rad = np.radians(dec_det)[:, np.newaxis]  # (N, 1)
                    ra_cat_rad = np.radians(ra_catalog)[np.newaxis, :]  # (1, M)
                    dec_cat_rad = np.radians(dec_catalog)[np.newaxis, :]  # (1, M)
                    sep_matrix = angularSeparation(ra_det_rad, dec_det_rad, ra_cat_rad, dec_cat_rad)
                    nearest_indices = np.argmin(sep_matrix, axis=1)  # (N,)
                    matched_catalog = catalog_stars_fov[nearest_indices]

                    # Restore original distortion type for final matched-pair fit
                    self.setDistortionType(original_dist_type, reset_params=False)

                    # Final fit: call fitAstrometry recursively with matched pairs
                    # Use first_platepar_fit=True to ensure fitPointing runs and x_poly_rev syncs
                    print("    Final fit on {} matched stars with {} (recursive call)...".format(
                        len(img_stars_clean), original_dist_type))
                    self.fitAstrometry(jd, img_stars_clean, matched_catalog,
                                       first_platepar_fit=True, use_nn_cost=False)

                    # Return the actual matched pairs from RANSAC
                    return (img_stars_clean, matched_catalog)
                else:
                    res = scipy.optimize.minimize(
                        cost_func,
                        p0,
                        args=(self, jd, catalog_stars, img_stars),
                        method='Nelder-Mead',
                        options=opt_options_final,
                    )

                # Update fitted astrometric parameters (Unnormalize the pointing parameters)
                ra_ref, dec_ref, pos_angle_ref, F_scale = res.x[:4]
                self.RA_d = (360 * ra_ref) % (360)
                self.dec_d = -90 + (90 * dec_ref + 90) % (180.000001)
                self.pos_angle_ref = (360 * pos_angle_ref) % (360)
                self.F_scale = abs(F_scale)

                self.updateRefAltAz()

                # Extract distortion parameters, IMPORTANT NOTE - the X polynomial is used to store the
                #   fit parameters
                self.x_poly_fwd = np.array(res.x[4:])

                ### ###

                # If this is the first fit of the distortion, set the forward parameters to be equal to the reverse
                if first_platepar_fit:
                    self.x_poly_rev = np.array(self.x_poly_fwd)

                ### REVERSE MAPPING FIT ###
                # Skip reverse fit when using NN cost (no matched pairs available)
                if not use_nn_cost:

                    # # Initial parameters for the pointing and distortion fit (normalize to the 0-1 range)
                    # p0  = [self.RA_d/360.0, self.dec_d/90.0, self.pos_angle_ref/360.0, abs(self.F_scale)]
                    # p0 += self.x_poly_rev.tolist()

                    # # Fit the radial distortion - the X polynomial is used to store the fit parameters
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
                    # #   fit parameters
                    # self.x_poly_rev = res.x[4:]

                    ## Distortion-only fit below!

                    # Fit the radial distortion - the X polynomial is used to store the fit parameters
                    res = scipy.optimize.minimize(
                        _calcImageResidualsDistortion,
                        self.x_poly_rev,
                        args=(self, jd, catalog_stars, img_stars, 'radial'),
                        method='Nelder-Mead',
                        options={'maxiter': 10000, 'adaptive': True},
                    )

                    # Extract distortion parameters, IMPORTANT NOTE - the X polynomial is used to store the
                    #   fit parameters
                    self.x_poly_rev = res.x

                ### ###

        else:
            if len(img_stars) < min_fit_stars:
                print('Too few stars to fit the distortion, only the astrometric parameters where fitted!')

        # Set the list of stars used for the fit to the platepar
        # Note: use_nn_cost=True returns early after RANSAC, so this only runs for matched-pair fits
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

    def extractRadialCoeffs(self, x_poly=None):
        """Extract logical radial distortion coefficients from the array.

        The coefficient array structure depends on flags (force_distortion_centre, equal_aspect,
        asymmetry_corr) and distortion type. This method extracts the logical values regardless
        of array layout.

        Arguments:
            x_poly: [ndarray] Optional coefficient array to extract from. If None, uses self.x_poly_fwd.

        Returns:
            dict: Dictionary with keys 'x0', 'y0', 'xy', 'a1', 'a2', 'k1', 'k2', 'k3', 'k4'
                  Values are 0.0 for coefficients not present in the current distortion type.
        """
        if x_poly is None:
            x_poly = self.x_poly_fwd

        # Only works for radial distortion types
        if not self.distortion_type.startswith("radial"):
            return None

        # Compute index offset based on flags (same logic as CyFunctions.pyx)
        index_offset = 0
        if self.force_distortion_centre:
            index_offset += 2
        if self.equal_aspect:
            index_offset += 1
        if not self.asymmetry_corr:
            index_offset += 2

        result = {'x0': 0.0, 'y0': 0.0, 'xy': 0.0, 'a1': 0.0, 'a2': 0.0,
                  'k1': 0.0, 'k2': 0.0, 'k3': 0.0, 'k4': 0.0}

        # Extract coefficients based on current flags
        idx = 0

        # Distortion center
        if not self.force_distortion_centre:
            if idx < len(x_poly):
                result['x0'] = x_poly[idx]
            idx += 1
            if idx < len(x_poly):
                result['y0'] = x_poly[idx]
            idx += 1

        # Aspect ratio
        if not self.equal_aspect:
            if idx < len(x_poly):
                result['xy'] = x_poly[idx]
            idx += 1

        # Asymmetry correction
        if self.asymmetry_corr:
            if idx < len(x_poly):
                result['a1'] = x_poly[idx]
            idx += 1
            if idx < len(x_poly):
                result['a2'] = x_poly[idx]
            idx += 1

        # Radial distortion coefficients (k1, k2, k3, k4)
        # Number depends on distortion type: radial3-odd has k1, radial5-odd has k1,k2, etc.
        if idx < len(x_poly):
            result['k1'] = x_poly[idx]
        idx += 1
        if idx < len(x_poly):
            result['k2'] = x_poly[idx]
        idx += 1
        if idx < len(x_poly):
            result['k3'] = x_poly[idx]
        idx += 1
        if idx < len(x_poly):
            result['k4'] = x_poly[idx]

        return result

    def buildRadialCoeffs(self, coeffs_dict, target_dist_type=None):
        """Build a coefficient array from logical radial distortion coefficients.

        Arguments:
            coeffs_dict: [dict] Dictionary with keys 'x0', 'y0', 'xy', 'a1', 'a2', 'k1', 'k2', 'k3', 'k4'
            target_dist_type: [str] Target distortion type. If None, uses current self.distortion_type.

        Returns:
            ndarray: Coefficient array for the target distortion type.
        """
        if target_dist_type is None:
            target_dist_type = self.distortion_type

        # Only works for radial distortion types
        if not target_dist_type.startswith("radial"):
            return None

        result = []

        # Distortion center
        if not self.force_distortion_centre:
            result.append(coeffs_dict.get('x0', 0.0))
            result.append(coeffs_dict.get('y0', 0.0))

        # Aspect ratio
        if not self.equal_aspect:
            result.append(coeffs_dict.get('xy', 0.0))

        # Asymmetry correction
        if self.asymmetry_corr:
            result.append(coeffs_dict.get('a1', 0.0))
            result.append(coeffs_dict.get('a2', 0.0))

        # Radial distortion coefficients - number depends on target type
        result.append(coeffs_dict.get('k1', 0.0))

        if target_dist_type in ["radial5-odd", "radial7-odd", "radial9-odd",
                                "radial4-all", "radial5-all"]:
            result.append(coeffs_dict.get('k2', 0.0))

        if target_dist_type in ["radial7-odd", "radial9-odd", "radial5-all"]:
            result.append(coeffs_dict.get('k3', 0.0))

        if target_dist_type in ["radial9-odd"]:
            result.append(coeffs_dict.get('k4', 0.0))

        return np.array(result)

    def convertDistortionCoeffs(self, target_dist_type):
        """Convert current distortion coefficients to a new distortion type.

        This properly maps coefficients when switching between radial distortion types,
        preserving k1 when going from e.g. radial7-odd to radial3-odd.

        Arguments:
            target_dist_type: [str] Target distortion type (e.g., "radial3-odd", "radial5-odd").

        Returns:
            tuple: (x_poly_new, y_poly_new) - New coefficient arrays for the target type.
        """
        # Only works for radial-to-radial conversions
        if not self.distortion_type.startswith("radial") or not target_dist_type.startswith("radial"):
            return self.x_poly_fwd.copy(), self.y_poly_fwd.copy()

        # Extract logical coefficients from current arrays
        x_coeffs = self.extractRadialCoeffs(self.x_poly_fwd)
        y_coeffs = self.extractRadialCoeffs(self.y_poly_fwd)

        # Build new arrays for target type
        x_poly_new = self.buildRadialCoeffs(x_coeffs, target_dist_type)
        y_poly_new = self.buildRadialCoeffs(y_coeffs, target_dist_type)

        return x_poly_new, y_poly_new

    def remapCoeffsForFlagChange(self, flag_name, new_value):
        """Remap distortion coefficients when a flag changes.

        This method properly remaps coefficients when toggling flags like
        force_distortion_centre, equal_aspect, or asymmetry_corr. It extracts
        the logical coefficient values BEFORE changing the flag, changes the flag,
        then rebuilds the coefficient arrays with the new flag state.

        Arguments:
            flag_name: [str] Name of flag to change ('force_distortion_centre',
                       'equal_aspect', or 'asymmetry_corr')
            new_value: [bool] New value for the flag

        Returns:
            bool: True if remapping was performed, False if not applicable
        """
        # Only works for radial distortion types
        if not self.distortion_type.startswith("radial"):
            # Just set the flag directly for non-radial types
            setattr(self, flag_name, new_value)
            return False

        # Get current flag value
        old_value = getattr(self, flag_name)

        # If flag isn't changing, nothing to do
        if old_value == new_value:
            return False

        # Extract logical coefficients BEFORE changing the flag
        x_coeffs_fwd = self.extractRadialCoeffs(self.x_poly_fwd)
        x_coeffs_rev = self.extractRadialCoeffs(self.x_poly_rev)
        y_coeffs_fwd = self.extractRadialCoeffs(self.y_poly_fwd)
        y_coeffs_rev = self.extractRadialCoeffs(self.y_poly_rev)

        if x_coeffs_fwd is None:
            # Not radial, just set flag
            setattr(self, flag_name, new_value)
            return False

        # Change the flag
        setattr(self, flag_name, new_value)

        # Update poly_length using setDistortionType (which recalculates it)
        # but with reset_params=False to not zero out coefficients
        self.setDistortionType(self.distortion_type, reset_params=False)

        # Rebuild coefficient arrays with the new flag state
        self.x_poly_fwd = self.buildRadialCoeffs(x_coeffs_fwd, self.distortion_type)
        self.x_poly_rev = self.buildRadialCoeffs(x_coeffs_rev, self.distortion_type)
        self.y_poly_fwd = self.buildRadialCoeffs(y_coeffs_fwd, self.distortion_type)
        self.y_poly_rev = self.buildRadialCoeffs(y_coeffs_rev, self.distortion_type)

        # Ensure proper length (in case of edge cases)
        self.padDictParams()

        # Update x_poly and y_poly references
        self.x_poly = self.x_poly_fwd
        self.y_poly = self.y_poly_fwd

        return True

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

        # Add the distortion type if not present (assume it's the polynomial type with the radial term)
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
            use_flat: [bool] Indicates whether a flat is used or not. None by default.
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

        # Set JSON to be the default format
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

        # If the polynomial is used, the X axis parameters are stored in x_poly, otherwise radials parameters
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

                # Only convert if the array actually has the asymmetry parameter
                if asym_ang_index < len(x_poly_fwd_print):
                    x_poly_fwd_print[asym_ang_index] = np.degrees(
                        (2 * np.pi * x_poly_fwd_print[asym_ang_index]) % (2 * np.pi)
                    )
                if asym_ang_index < len(x_poly_rev_print):
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

        # Add the photometric parameters
        out_str += "Photometry:\n"
        out_str += "    Slope     = {:6.3f}\n".format(self.mag_0)
        out_str += "    Zeropoint = {:6.3f} +/- {:.3f}\n".format(self.mag_lev, self.mag_lev_stddev)
        out_str += "    Gamma     = {:6.3f}\n".format(self.gamma)
        out_str += "    Vignetting coeff = {:6.3f}\n".format(self.vignetting_coeff)
        out_str += "    Vignetting fixed = {:s}\n".format(
            "True" if self.vignetting_fixed else "False"
        )
        out_str += "    Extinction scale = {:6.3f}\n".format(self.extinction_scale)


        return out_str


def findBestPlatepar(config, night_data_dir=None):
    """ Try loading the platepar from the night data directory, and if it doesn't exist, load the default one
    from the config directory.
    
    Arguments:
        config: [Config] Configuration object.

    Keyword arguments:
        night_data_dir: [str] Path to the night data directory. None by default.

    Return:
        platepar: [Platepar] Loaded platepar object. None if not found.
    """

    # Get the platepar (first from the night directory, then the default one)
    platepar = Platepar()
    
    if night_data_dir is not None:
        night_platepar_path = os.path.join(night_data_dir, config.platepar_name)

    default_platepar_path = os.path.join(config.config_file_path, config.platepar_name)

    if (night_data_dir is not None) and os.path.exists(night_platepar_path):
        platepar.read(night_platepar_path, use_flat=config.use_flat)

        return platepar

    elif os.path.exists(default_platepar_path):
        platepar.read(default_platepar_path, use_flat=config.use_flat)

        return platepar

    
    return None


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
            [time_data], [x_img], [y_img], [1], pp, extinction_correction=False, precompute_pointing_corr=True
        )

        # Map back to X, Y
        x_data, y_data = RMS.Astrometry.ApplyAstrometry.raDecToXYPP(ra_data, dec_data, jd_data[0], pp)

        # Map forward to sky again
        _, ra_data_rev, dec_data_rev, _ = RMS.Astrometry.ApplyAstrometry.xyToRaDecPP(
            [time_data], x_data, y_data, [1], pp, extinction_correction=False, precompute_pointing_corr=True
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
