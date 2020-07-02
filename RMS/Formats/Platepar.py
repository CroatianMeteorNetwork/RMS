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

from __future__ import print_function, division, absolute_import

import os
import json
import copy
import datetime


import numpy as np
import scipy.optimize

from RMS.Astrometry.Conversions import date2JD, jd2Date, raDec2AltAz
import RMS.Astrometry.ApplyAstrometry
from RMS.Math import angularSeparation



class stationData(object):
    """ Holds information about one meteor station (location) and observed points.
    """

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
    """ Parse information from an INF file to a stationData object.
    """

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
    """ Get image positions of catalog stars using the current platepar values. 

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
    """ Compute RA, Dec of all paired stars on the image given the platepar. 

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
    _, ra_array, dec_array, _ = RMS.Astrometry.ApplyAstrometry.xyToRaDecPP(len(img_x)*[img_time], img_x, 
        img_y, len(img_x)*[1], platepar, extinction_correction=False)

    return ra_array, dec_array


class Platepar(object):
    def __init__(self, distortion_type="poly3+radial"):
        """ Astrometric and photometric calibration plate parameters. Several distortion types are supported.
    
        Arguments:
            file_name: [string] Path to the platepar file.

        Keyword arguments:
            distortion_type: [str] Distortion type. It can be one of the following:
                - "poly3+radial" - 3rd order polynomial fit including a single radial term
                - "radial3" - 3rd order radial distortion
                - "radial4" - 4th order radial distortion
                - "radial5" - 5th order radial distortion
    
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
        self.JD = 0

        # UT correction
        self.UT_corr = 0

        self.Ho = 0
        self.X_res = 0
        self.Y_res = 0

        self.fov_h = 0
        self.fov_v = 0

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

        # Equal aspect (X and Y scales are equal) - used ONLY for radial distortion
        self.equal_aspect = False

        # Photometry calibration
        self.mag_0 = -2.5
        self.mag_lev = 1.0
        self.mag_lev_stddev = 0.0
        self.gamma = 1.0
        self.vignetting_coeff = 0.0

        # Extinction correction scaling
        self.extinction_scale = 1.0

        self.station_code = None

        self.star_list = None

        # Flag to indicate that the platepar was refined with CheckFit
        self.auto_check_fit_refined = False

        # Flag to indicate that the platepar was successfuly auto recalibrated on an individual FF files
        self.auto_recalibrated = False

        # Init the distortion parameters
        self.resetDistortionParameters()


    def resetDistortionParameters(self, preserve_centre=False):
        """ Set the distortion parameters to zero. 
    
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
        self.x_poly_fwd = np.zeros(shape=(12,), dtype=np.float64)
        self.y_poly_fwd = np.zeros(shape=(12,), dtype=np.float64)
        self.x_poly_rev = np.zeros(shape=(12,), dtype=np.float64)
        self.y_poly_rev = np.zeros(shape=(12,), dtype=np.float64)



        # Preserve the image centre
        if preserve_centre:

            # Preserve centre for the radial distortion
            if self.distortion_type.startswith("radial"):

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
                self.x_poly_fwd[1] = 0.5
                self.x_poly_rev[1] = 0.5



        self.x_poly = self.x_poly_fwd
        self.y_poly = self.y_poly_fwd



    def setDistortionType(self, distortion_type, reset_params=True):
        """ Sets the distortion type. """

        # List of distortion types an number of parameters for each
        self.distortion_type_list = [
             "poly3+radial",
                  "radial3",
                  "radial4",
                  "radial5"
            ]

        # Set the length of the distortion polynomial depending on the distortion type
        if distortion_type in self.distortion_type_list:
            self.distortion_type = distortion_type

        else:
            raise ValueError("The distortion type is not recognized: {:s}".format(self.distortion_type))


        # Reset distortion parameters
        if reset_params:
            self.resetDistortionParameters()


    def addVignettingCoeff(self, use_flat):
        """ Add a vignetting coeff to the platepar if it doesn't have one. 
        
        Arguments:
            use_flat: [bool] Is the flat used or not.
        """

        # Add a vignetting coefficient if it's not set
        if self.vignetting_coeff is None:

            # Only add it if a flat is not used
            if use_flat:
                self.vignetting_coeff = 0.0

            else:

                # Use 0.001 deg/px as the default coefficeint, as that's the one for 3.6 mm f/0.95 and 16 mm
                #   f/1.0 lenses. The vignetting coeff is dependent on the resolution, the default value of 
                #   0.001 deg/px is for 720p.
                self.vignetting_coeff = 0.001*np.hypot(1280, 720)/np.hypot(self.X_res, self.Y_res)



    def fitAstrometry(self, jd, img_stars, catalog_stars, first_platepar_fit=False):
        """ Fit astrometric parameters to the list of star image and celectial catalog coordinates. 
        At least 4 stars are needed to fit the rigid body parameters, and 12 to fit the distortion.
        New parameters are saved to the given object.

        Arguments:
            jd: [float] Julian date of the image.
            img_stars: [list] A list of (x, y, intensity_sum) entires for every star.
            catalog_stars: [list] A list of (ra, dec, mag) entries for every star (degrees).

        Keyword arguments:
            first_platepar_fit: [bool] Fit a platepar from scratch. False by default.

        """


        def _calcImageResidualsAstro(params, platepar, jd, catalog_stars, img_stars):
            """ Calculates the differences between the stars on the image and catalog stars in image 
                coordinates with the given astrometrical solution. 

            """

            # Extract fitting parameters
            ra_ref, dec_ref, pos_angle_ref, F_scale = params

            img_x, img_y, _ = img_stars.T

            pp_copy = copy.deepcopy(platepar)

            pp_copy.RA_d = ra_ref
            pp_copy.dec_d = dec_ref
            pp_copy.pos_angle_ref = pos_angle_ref
            pp_copy.F_scale = F_scale

            # Get image coordinates of catalog stars
            catalog_x, catalog_y, catalog_mag = getCatalogStarsImagePositions(catalog_stars, jd, pp_copy)
            
            # Calculate the sum of squared distances between image stars and catalog stars
            dist_sum = np.sum((catalog_x - img_x)**2 + (catalog_y - img_y)**2)

            return dist_sum


        def _calcSkyResidualsAstro(params, platepar, jd, catalog_stars, img_stars):
            """ Calculates the differences between the stars on the image and catalog stars in sky 
                coordinates with the given astrometrical solution. 

            """

            # Extract fitting parameters
            ra_ref, dec_ref, pos_angle_ref, F_scale = params

            pp_copy = copy.deepcopy(platepar)

            pp_copy.RA_d = ra_ref
            pp_copy.dec_d = dec_ref
            pp_copy.pos_angle_ref = pos_angle_ref
            pp_copy.F_scale = F_scale

            img_x, img_y, _ = img_stars.T

            # Get image coordinates of catalog stars
            ra_array, dec_array = getPairedStarsSkyPositions(img_x, img_y, jd, pp_copy)

            ra_catalog, dec_catalog, _ = catalog_stars.T

            # Compute the sum of the angular separation
            separation_sum = np.sum(angularSeparation(np.radians(ra_array), np.radians(dec_array), \
                np.radians(ra_catalog), np.radians(dec_catalog))**2)


            return separation_sum



        def _calcImageResidualsDistortion(params, platepar, jd, catalog_stars, img_stars, dimension):
            """ Calculates the differences between the stars on the image and catalog stars in image 
                coordinates with the given astrometrical solution. 

            Arguments:
                ...
                dimension: [str] 'x' for X polynomial fit, 'y' for Y polynomial fit

            """

            # Set distortion parameters
            pp_copy = copy.deepcopy(platepar)

            if (dimension == 'x') or (dimension == 'radial'):
                pp_copy.x_poly_rev = params
                pp_copy.y_poly_rev = np.zeros(12)

            else:
                pp_copy.x_poly_rev = np.zeros(12)
                pp_copy.y_poly_rev = params


            img_x, img_y, _ = img_stars.T


            # Get image coordinates of catalog stars
            catalog_x, catalog_y, catalog_mag = getCatalogStarsImagePositions(catalog_stars, jd, pp_copy)


            # Calculate the sum of squared distances between image stars and catalog stars, per every
            #   dimension
            if dimension == 'x':
                dist_sum = np.sum((catalog_x - img_x)**2)

            elif dimension == 'y':
                dist_sum = np.sum((catalog_y - img_y)**2)

            # Minimization for the radial distortion
            else:
                dist_sum = np.sum((catalog_x - img_x)**2 + (catalog_y - img_y)**2)



            return dist_sum


        def _calcSkyResidualsDistortion(params, platepar, jd, catalog_stars, img_stars, dimension):
            """ Calculates the differences between the stars on the image and catalog stars in sky 
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
            separation_sum = np.sum(angularSeparation(np.radians(ra_array), np.radians(dec_array), \
                np.radians(ra_catalog), np.radians(dec_catalog))**2)

            return separation_sum


        # print('ASTRO', _calcImageResidualsAstro([self.RA_d, self.dec_d, 
        #     self.pos_angle_ref, self.F_scale],  catalog_stars, img_stars))

        # print('DIS_X', _calcImageResidualsDistortion(self.x_poly_rev,  catalog_stars, \
        #     img_stars, 'x'))

        # print('DIS_Y', _calcImageResidualsDistortion(self.y_poly_rev,  catalog_stars, \
        #     img_stars, 'y'))


        ### ASTROMETRIC PARAMETERS FIT ###

        # Initial parameters for the astrometric fit
        p0 = [self.RA_d, self.dec_d, self.pos_angle_ref, self.F_scale]

        # Fit the astrometric parameters using the reverse transform for reference        
        res = scipy.optimize.minimize(_calcImageResidualsAstro, p0, \
            args=(self, jd, catalog_stars, img_stars), method='SLSQP')

        # # Fit the astrometric parameters using the forward transform for reference
        #   WARNING: USING THIS MAKES THE FIT UNSTABLE
        # res = scipy.optimize.minimize(_calcSkyResidualsAstro, p0, args=(self, jd, \
        #     catalog_stars, img_stars), method='Nelder-Mead')

        # Update fitted astrometric parameters
        self.RA_d, self.dec_d, self.pos_angle_ref, self.F_scale = res.x

        # Recalculate FOV centre
        self.az_centre, self.alt_centre = raDec2AltAz(self.RA_d, self.dec_d, self.JD, self.lat, self.lon)

        ### ###


        ### DISTORTION FIT ###

        # fit the polynomial distortion parameters if there are more than 12 stars picked
        if self.distortion_type.startswith("poly"):
            min_fit_stars = 12

        # fit the radial distortion parameters if there are more than 7 stars picked
        else:
            min_fit_stars = 7


        if len(img_stars) >= min_fit_stars:

            ### REVERSE MAPPING FIT ###

            # Fit the polynomial distortion
            if self.distortion_type.startswith("poly"):

                # Fit distortion parameters in X direction, reverse mapping
                res = scipy.optimize.minimize(_calcImageResidualsDistortion, self.x_poly_rev, \
                    args=(self, jd, catalog_stars, img_stars, 'x'), method='Nelder-Mead', \
                    options={'maxiter': 10000, 'adaptive': True})

                # Exctact fitted X polynomial
                self.x_poly_rev = res.x

                # Fit distortion parameters in Y direction, reverse mapping
                res = scipy.optimize.minimize(_calcImageResidualsDistortion, self.y_poly_rev, \
                    args=(self, jd, catalog_stars, img_stars, 'y'), method='Nelder-Mead', \
                    options={'maxiter': 10000, 'adaptive': True})

                # Extract fitted Y polynomial
                self.y_poly_rev = res.x


            # Fit radial distortion
            else:

                # Fit the radial distortion - the X polynomial is used to store the fit paramters
                res = scipy.optimize.minimize(_calcImageResidualsDistortion, self.x_poly_rev, \
                    args=(self, jd, catalog_stars, img_stars, 'radial'), method='Nelder-Mead', \
                    options={'maxiter': 10000, 'adaptive': True})

                # IMPORTANT NOTE - the X polynomial is used to store the fit paramters
                self.x_poly_rev = res.x

                # Force aspect ratio to 0 if axes are set to be equal
                if self.equal_aspect:
                    self.x_poly_rev[2] = 0

                # Set all parameters not used by the radial fit to 0
                n_params = int(self.distortion_type[-1])
                self.x_poly_rev[(n_params + 2):] *= 0


            ### ###

            

            # If this is the first fit of the distortion, set the forward parametrs to be equal to the reverse
            if first_platepar_fit:

                self.x_poly_fwd = np.array(self.x_poly_rev)
                self.y_poly_fwd = np.array(self.y_poly_rev)


            ### FORWARD MAPPING FIT ###
            
            # Fit the polynomial distortion
            if self.distortion_type.startswith("poly"):

                # Fit distortion parameters in X direction, forward mapping
                res = scipy.optimize.minimize(_calcSkyResidualsDistortion, self.x_poly_fwd, \
                    args=(self, jd, catalog_stars, img_stars, 'x'), method='Nelder-Mead', \
                    options={'maxiter': 10000, 'adaptive': True})

                # Extract fitted X polynomial
                self.x_poly_fwd = res.x


                # Fit distortion parameters in Y direction, forward mapping
                res = scipy.optimize.minimize(_calcSkyResidualsDistortion, self.y_poly_fwd, \
                    args=(self, jd, catalog_stars, img_stars, 'y'), method='Nelder-Mead', \
                    options={'maxiter': 10000, 'adaptive': True})

                # IMPORTANT NOTE - the X polynomial is used to store the fit paramters
                self.y_poly_fwd = res.x


            # Fit the radial distortion
            else:

                # Fit the radial distortion - the X polynomial is used to store the fit paramters
                res = scipy.optimize.minimize(_calcSkyResidualsDistortion, self.x_poly_fwd, \
                    args=(self, jd, catalog_stars, img_stars, 'radial'), method='Nelder-Mead', \
                    options={'maxiter': 10000, 'adaptive': True})

                # Extract fitted X polynomial
                self.x_poly_fwd = res.x

                # Force aspect ratio to 0 if axes are set to be equal
                if self.equal_aspect:
                    self.x_poly_fwd[2] = 0

                # Set all parameters not used by the radial fit to 0
                n_params = int(self.distortion_type[-1])
                self.x_poly_fwd[(n_params + 2):] *= 0

            ### ###

        else:
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
        """ Read next line, split the line and convert parameters to float.

        @param f: [file handle] file we want to read

        @return (a1, a2, ...): [tuple of floats] parsed data from the line
        """

        return map(float, f.readline().split())


    def loadFromDict(self, platepar_dict, use_flat=None):
        """ Load the platepar from a dictionary. """

        # Parse JSON into an object with attributes corresponding to dict keys
        self.__dict__ = platepar_dict

        # Add the version if it was not in the platepar (v1 platepars didn't have a version)
        if not 'version' in self.__dict__:
            self.version = 1


        # If the refraction was not used for the fit, assume it is disabled
        if not 'refraction' in self.__dict__:
            self.refraction = False


        # Add equal aspect
        if not 'equal_aspect' in self.__dict__:
            self.equal_aspect = False


        # Add the distortion type if not present (assume it's the polynomal type with the radial term)
        if not 'distortion_type' in self.__dict__:

            # Check if the variable with the typo was used and correct it
            if 'distortion_type' in self.__dict__:
                self.distortion_type = self.distortion_type
                del self.distortion_type

            # Otherwise, assume the polynomial type
            else:
                self.distortion_type = "poly3+radial"
        
        self.setDistortionType(self.distortion_type, reset_params=False)

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

        # Set polynomial parameters used by the old code
        self.x_poly = self.x_poly_fwd
        self.y_poly = self.y_poly_fwd


        # Add rotation from horizontal
        if not 'rotation_from_horiz' in self.__dict__:
            self.rotation_from_horiz = RMS.Astrometry.ApplyAstrometry.rotationWrtHorizon(self)

        # Calculate the datetime
        self.time = jd2Date(self.JD, dt_obj=True)




    def read(self, file_name, fmt=None, use_flat=None):
        """ Read the platepar. 
            
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
                T = (self.JD - 2451545.0)/36525.0
                self.Ho = (280.46061837 + 360.98564736629*(self.JD - 2451545.0) + 0.000387933*T**2 - 
                    T**3/38710000.0)%360

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
                self.F_scale = 3600/self.F_scale

                # Load magnitude slope parameters
                self.mag_0, self.mag_lev = self.parseLine(f)

                # Load X axis polynomial parameters
                self.x_poly_fwd = self.x_poly_rev = np.zeros(shape=(12,), dtype=np.float64)
                for i in range(12):
                    self.x_poly_fwd[i] = self.x_poly_fwd[i] = self.parseLine(f)[0]

                # Load Y axis polynomial parameters
                self.y_poly_fwd = self.y_poly_rev = np.zeros(shape=(12,), dtype=np.float64)
                for i in range(12):
                    self.y_poly_fwd[i] = self.y_poly_rev[i] = self.parseLine(f)[0]

                # Read station code
                self.station_code = f.readline().replace('\r', '').replace('\n', '')


        # Add a default vignetting coefficient if it already doesn't exist
        self.addVignettingCoeff(use_flat)

        return fmt


    def jsonStr(self):
        """ Returns the JSON representation of the platepar as a string. """

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
        """ Write platepar to file. 
        
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
                self.RA_H = int(self.RA_d/15)
                self.RA_M = int((self.RA_d/15 - self.RA_H)*60)
                self.RA_S = int(((self.RA_d/15 - self.RA_H)*60 - self.RA_M)*60)

                f.write("{:7.3f} {:02d} {:02d} {:02d}\n".format(self.RA_d, self.RA_H, self.RA_M, self.RA_S))

                # Write reference Dec
                self.dec_D = int(self.dec_d)
                self.dec_M = int((self.dec_d - self.dec_D)*60)
                self.dec_S = int(((self.dec_d - self.dec_D)*60 - self.dec_M)*60)

                f.write("{:+7.3f} {:02d} {:02d} {:02d}\n".format(self.dec_d, self.dec_D, self.dec_M, self.dec_S))

                # Write rotation parameter
                f.write('{:<7.3f}\n'.format(self.pos_angle_ref))

                # Write F scale
                f.write('{:<5.1f}\n'.format(3600/self.F_scale))

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


    def __repr__(self):

        out_str  = "Platepar\n"
        out_str += "--------\n"
        out_str += "Reference pointing equatorial (J2000):\n"
        out_str += "    JD      = {:.10f} \n".format(self.JD)
        out_str += "    RA      = {:.6f} deg\n".format(self.RA_d)
        out_str += "    Dec     = {:.6f} deg\n".format(self.dec_d)
        out_str += "    Pos ang = {:.6f} deg\n".format(self.pos_angle_ref)
        out_str += "    Pix scl = {:.2f} arcmin/px\n".format(60/self.F_scale)
        out_str += "Distortion:\n"
        out_str += "    Type = {:s}\n".format(self.distortion_type)

        # If the polynomial is used, the X axis parameters are stored in x_poly, otherwise radials paramters
        #   are used
        if self.distortion_type.startswith("poly"):
            out_str += "    Distortion coeffs (polynomial):\n"
            dist_string = "X"
        else:
            out_str += "    Distortion coeffs (radial):\n"
            out_str += "                 x0,       y0, aspect-1,       k1,       k2,       k3,       k4\n"
            dist_string = ""

        out_str += "img2sky {:s} = {:s}\n".format(dist_string, ", ".join(["{:+8.3f}".format(c) \
            if abs(c) > 10e-4 else "{:+8.1e}".format(c) for c in self.x_poly_fwd]))
        out_str += "sky2img {:s} = {:s}\n".format(dist_string, ", ".join(["{:+8.3f}".format(c) \
            if abs(c) > 10e-4 else "{:+8.1e}".format(c) for c in self.x_poly_rev]))

        # Only print the rest if the polynomial fit is used
        if self.distortion_type.startswith("poly"):
            out_str += "img2sky Y = {:s}\n".format(", ".join(["{:+8.3f}".format(c) \
                if abs(c) > 10e-4 else "{:+8.1e}".format(c) for c in self.y_poly_fwd]))
            out_str += "sky2img Y = {:s}\n".format(", ".join(["{:+8.3f}".format(c) \
                if abs(c) > 10e-4 else "{:+8.1e}".format(c) for c in self.y_poly_rev]))

        return out_str



if __name__ == "__main__":


    import argparse

    # Init argument parser
    arg_parser = argparse.ArgumentParser(description="Test the astrometry functions using the given platepar.")
    arg_parser.add_argument('platepar_path', metavar='PLATEPAR', type=str, \
        help='Path to the platepar file')

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
        jd_data, ra_data, dec_data, _ = RMS.Astrometry.ApplyAstrometry.xyToRaDecPP([time_data], [x_img], \
            [y_img], [1], pp, extinction_correction=False)

        # Map back to X, Y
        x_data, y_data = RMS.Astrometry.ApplyAstrometry.raDecToXYPP(ra_data, dec_data, jd_data[0], pp)

        # Map forward to sky again
        _, ra_data_rev, dec_data_rev, _ = RMS.Astrometry.ApplyAstrometry.xyToRaDecPP([time_data], x_data, \
            y_data, [1], pp, extinction_correction=False)


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
        print("RA  = {:.4f} amin".format(60*(ra_data[0] - ra_data_rev[0])))
        print("Dec = {:+.4f} amin".format(60*(dec_data[0] - dec_data_rev[0])))

