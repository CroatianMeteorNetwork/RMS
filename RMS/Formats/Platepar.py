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
import math
import os

import matplotlib.pyplot as plt
import numpy as np

# Import Cython functions
import pyximport
import RMS.Astrometry.ApplyAstrometry
import scipy.optimize
from scipy.spatial import cKDTree
from RMS.Astrometry.Conversions import date2JD, jd2Date, JD2HourAngle, trueOfDateRaDec2ApparentAltAz, \
    apparentAltAz2TrueOfDateRaDec
from RMS.GeoidHeightEGM96 import mslToWGS84Height
from RMS.Math import angularSeparation, sphericalPointFromHeadingAndDistance
from RMS.Misc import RmsDateTime, rmsCommit

pyximport.install(setup_args={'include_dirs': [np.get_include()]})
from RMS.Astrometry.CyFunctions import (
    pyRefractionApparentToTrue,
    pyRefractionTrueToApparent,
    refractionScale,
)


# Version of the astrometric model of the kernels. The fitted pointing, scale and distortion take up
#   whatever the model leaves out, so their meaning depends on it. Increment it whenever a change of the
#   kernels alters what the fitted coefficients mean, and add it to the history below. The model a platepar
#   was fitted with is kept in its fit record (Platepar.fit_record, Platepar.astrometryModel). History:
#   1 - Platepars fitted before the fit record existed: the reference hour angle from the mean sidereal
#       time, the reference pointing taken as a J2000 direction by the refraction, the Bennett/Saemundsson
#       refraction pair without a scale for the station height, and the annual aberration absorbed by the fit.
#   2 - True-of-date frames with the apparent sidereal time, the exact inverse of the refraction scaled with
#       the station height, and the annual aberration modelled.
ASTROMETRY_MODEL = 2

# The fits kept in the fit record of a platepar: the distortion (with the scale and the rest of the lens
#   model) and the pointing
FIT_RECORD_KEYS = ('distortion', 'pointing')


def fitRecordEntry(fit_by):
    """ The record of a fit made now by this code.

    Arguments:
        fit_by: [str] Short label of what made the fit, e.g. 'SkyFit2' or 'recalibrateFF'.

    Return:
        [dict] 'model': the astrometric model (ASTROMETRY_MODEL), 'commit': the commit of the RMS code (see
            RMS.Misc.gitCommit), 'utc': the time of the fit (ISO 8601, UTC) and 'by': the label.
    """

    # The model and the code of this process, and the time now
    return {
        'model': ASTROMETRY_MODEL,
        'commit': rmsCommit(),
        'utc': RmsDateTime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'by': str(fit_by),
    }


def emptyFitRecord():
    """ The fit record of a platepar with no recorded fit (e.g. fitted before the record existed).

    Return:
        [dict] None for every fit of FIT_RECORD_KEYS.
    """

    return {key: None for key in FIT_RECORD_KEYS}


def normalFitRecord(fit_record):
    """ A fit record as read from a file or a pickle, with an entry for every fit (None if it is missing or
        is not a record).

    Arguments:
        fit_record: [dict] Fit record, or None.

    Return:
        [dict] The fit record with only the keys of FIT_RECORD_KEYS.
    """

    # Only the entries which are records are kept, the others are not known
    record = emptyFitRecord()
    if isinstance(fit_record, dict):
        for key in FIT_RECORD_KEYS:
            if isinstance(fit_record.get(key), dict):
                record[key] = fit_record[key]

    return record


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


def normalizeRaDec(ra_d, dec_d, refraction=False):
    """ Bring a pointing whose declination was stepped past a pole back onto the celestial sphere.

    The fits step the declination of the pointing freely, so near a celestial pole it can go past 90 deg.
    The point (ra, 90 + x) is the point (ra + 180, 90 - x), so the declination is reflected across the pole
    and the right ascension turned by 180 deg (a modulo wrap of the declination would jump to the opposite
    pole instead).

    The position angle is measured from the celestial meridian at the pointing the projection uses. Without
    the refraction that is the pointing itself, whose meridian turns by 180 deg when it goes over the pole,
    so the position angle has to be turned by 180 deg with the reflection to keep the same field. With the
    refraction it is the refracted pointing, which is lifted towards the zenith and goes beside the pole, so
    its meridian does not turn and the position angle is kept. Either way the projected field stays
    continuous when the declination is stepped over the pole.

    Arguments:
        ra_d: [float] Right ascension (deg), any value.
        dec_d: [float] Declination (deg), any value.

    Keyword arguments:
        refraction: [bool] The projection corrects the pointing for the refraction (Platepar.refraction).
            False by default.

    Return:
        (ra_d, dec_d, pos_angle_offset): [tuple of floats] Right ascension in [0, 360) (deg), declination in
            [-90, 90] (deg), and the angle to add to the position angle (deg, 180 for a reflection without the
            refraction, otherwise 0).
    """

    # Bring the declination into one turn starting at the south pole. For a declination already in
    #   [-90, 90] this is the same arithmetic as the decoding of the fit parameters it replaced, so fits that
    #   stay away from the poles give the same numbers as before
    dec_d = (dec_d + 90.0)%360.0 - 90.0

    # Reflect the pointing across the pole it went past. Without the refraction the meridian of the pointing
    #   turns by 180 deg with it, so does the position angle
    pos_angle_offset = 0.0
    if dec_d > 90.0:
        dec_d = 180.0 - dec_d
        ra_d = ra_d + 180.0
        if not refraction:
            pos_angle_offset = 180.0

    # Wrap the right ascension
    ra_d = ra_d%360.0

    return ra_d, dec_d, pos_angle_offset


def pointingFrame(ra_d, dec_d, pos_angle_ref):
    """ The axes of the camera for a pointing, as the projection uses them: the directions on the sky of the +x
        and +y standard coordinates at the pointing (the image x and y directions at the centre of the field),
        and the pointing itself.

    Arguments:
        ra_d: [float] Right ascension of the pointing (deg).
        dec_d: [float] Declination of the pointing (deg).
        pos_angle_ref: [float] Position angle, the rotation of the field from the celestial meridian at the
            pointing (deg).

    Return:
        [ndarray] Rotation matrix (3x3) whose columns are the +x, +y and pointing directions, in the
            equatorial axes of the right ascension and declination.
    """

    ra, dec, pos_angle = np.radians(ra_d), np.radians(dec_d), np.radians(pos_angle_ref)

    # The north and east directions at the pointing, and the pointing
    north = np.array([-np.sin(dec)*np.cos(ra), -np.sin(dec)*np.sin(ra), np.cos(dec)])
    east = np.array([-np.sin(ra), np.cos(ra), 0.0])
    axis = np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])

    # The standard coordinates are turned by the position angle from them (as in CyFunctions.cameraAxes)
    x_dir = np.sin(pos_angle)*north - np.cos(pos_angle)*east
    y_dir = -np.cos(pos_angle)*north - np.sin(pos_angle)*east

    return np.column_stack([x_dir, y_dir, axis])


def frameToPointing(x_dir, axis):
    """ The pointing of a camera from its axes, the inverse of pointingFrame.

    Arguments:
        x_dir: [list] The +x direction (unit vector in the equatorial axes, the first column of pointingFrame).
        axis: [list] The pointing (unit vector, the last column of pointingFrame).

    Return:
        (ra_d, dec_d, pos_angle_ref): [tuple of floats] Right ascension in [0, 360), declination in [-90, 90] and
            position angle in [0, 360) (deg).
    """

    # The pointing, with the declination from the distance to the pole, which is accurate near the poles too
    ra = math.atan2(axis[1], axis[0])
    dec = math.atan2(axis[2], math.hypot(axis[0], axis[1]))

    # The position angle is the angle of the +x direction from the north and east directions at the pointing.
    #   At a pole they follow the right ascension found above, so the field comes out the same
    north = (-math.sin(dec)*math.cos(ra), -math.sin(dec)*math.sin(ra), math.cos(dec))
    east = (-math.sin(ra), math.cos(ra))
    pos_angle = math.atan2(x_dir[0]*north[0] + x_dir[1]*north[1] + x_dir[2]*north[2],
                           -(x_dir[0]*east[0] + x_dir[1]*east[1]))

    return math.degrees(ra)%360.0, math.degrees(dec), math.degrees(pos_angle)%360.0


def offsetAxes(dx, dy, droll):
    """ The axes of a camera turned by angles in its own axes: the optical axis is turned by dx towards
        the +x direction and by dy towards the +y direction (along the great circle in the direction (dx, dy),
        by the angle hypot(dx, dy)), and the camera is then rolled by droll about the new optical axis, which
        turns the +x direction towards +y.

    Arguments:
        dx: [float] Turn of the optical axis towards +x (radians).
        dy: [float] Turn of the optical axis towards +y (radians).
        droll: [float] Roll about the new optical axis (radians).

    Return:
        (x_dir, axis): [tuple] The +x direction and the optical axis of the turned camera (unit vectors in the
            axes of the camera before the turn, see offsetRotation).
    """

    tilt = math.hypot(dx, dy)

    # sin(tilt)/tilt and (1 - cos(tilt))/tilt**2 = 2*sin(tilt/2)**2/tilt**2, which are 1 and 1/2 for no tilt
    if tilt > 0:
        sin_ratio = math.sin(tilt)/tilt
        cos_ratio = 2*(math.sin(tilt/2)/tilt)**2
    else:
        sin_ratio, cos_ratio = 1.0, 0.5

    # The +x and +y directions turned with the optical axis about the axis perpendicular to the direction of
    #   the turn (Rodrigues' formula)
    x_tilt = (1.0 - cos_ratio*dx*dx, -cos_ratio*dx*dy, -sin_ratio*dx)
    y_tilt = (-cos_ratio*dx*dy, 1.0 - cos_ratio*dy*dy, -sin_ratio*dy)

    # The roll turns the +x direction towards +y about the turned optical axis
    cos_roll, sin_roll = math.cos(droll), math.sin(droll)
    x_dir = tuple(cos_roll*x_comp + sin_roll*y_comp for x_comp, y_comp in zip(x_tilt, y_tilt))
    axis = (sin_ratio*dx, sin_ratio*dy, math.cos(tilt))

    return x_dir, axis


def offsetRotation(dx, dy, droll):
    """ The rotation of a camera by angles in its own axes (see offsetAxes).

    Arguments:
        dx: [float] Turn of the optical axis towards +x (radians).
        dy: [float] Turn of the optical axis towards +y (radians).
        droll: [float] Roll about the new optical axis (radians).

    Return:
        [ndarray] Rotation matrix (3x3) in the axes of the camera (+x, +y, optical axis): the axes of the turned
            camera are frame.dot(rotation) for the axes of the camera before the turn (pointingFrame).
    """

    x_dir, axis = offsetAxes(dx, dy, droll)

    # The +y direction completes the right-handed axes
    return np.column_stack([x_dir, np.cross(axis, x_dir), axis])


def rotationOffsets(rotation):
    """ The angles of a rotation of a camera in its own axes, the inverse of offsetRotation. The turn of the
        optical axis has to be less than 180 deg.

    Arguments:
        rotation: [ndarray] Rotation matrix (3x3) in the axes of the camera.

    Return:
        (dx, dy, droll): [tuple of floats] Turn of the optical axis towards +x and +y, and the roll about the new
            optical axis (radians, the roll in (-pi, pi]).
    """

    # The turned optical axis is (sin(tilt)/tilt*dx, sin(tilt)/tilt*dy, cos(tilt)) in the axes of the camera
    axis = rotation[:, 2]
    tilt = np.arctan2(np.hypot(axis[0], axis[1]), axis[2])
    sin_ratio = np.sinc(tilt/np.pi)
    dx, dy = axis[0]/sin_ratio, axis[1]/sin_ratio

    # What is left after the turn of the axis is the roll about it
    roll_rotation = offsetRotation(dx, dy, 0.0).T.dot(rotation)
    droll = np.arctan2(roll_rotation[1, 0], roll_rotation[0, 0])

    return float(dx), float(dy), float(droll)


class PointingOffsets(object):
    """ The pointing of a platepar as small offsets from a reference pointing, for the fits of the pointing.

    Stepping the right ascension, declination and position angle of the pointing is badly conditioned: a step
    of the right ascension is cos(dec) of it on the sky, which shrinks to nothing at a celestial pole, where the
    right ascension and the position angle turn the field the same way, and the direction in which a step moves
    the field on the image depends on the position angle. The fits instead turn the camera in its own axes
    about the reference pointing: the optical axis is moved along the image x and y directions, and the camera
    is rolled about it (offsetAxes). A step then moves the field by the same amount in the same direction on the
    image wherever the camera points, the poles and the zenith included.

    The camera is turned as the projection sees it. With the refraction, the projection refracts the reference
    pointing and measures the position angle from the meridian at the refracted pointing, so the refracted
    pointing is turned and the result is taken back through the refraction, keeping the position angle (as
    CyFunctions.pointingCorrection does). The right ascension and declination are those of the platepar (true
    of date at its reference time), and the refraction is the same at every time, as the projection keeps the
    hour angle of the pointing.

    The offsets are given in pixels: dx and dy move the optical axis by dx/F_scale and dy/F_scale (deg) towards
    the +x and +y directions of the image (the stars move the other way), and a roll of droll turns the field
    so that the stars at a quarter of the image diagonal from the centre move by droll. The fits which step the
    scale with the pointing in pixels use a fourth offset, dscale, which changes the scale so that those stars
    move by dscale.
    """

    def __init__(self, platepar):
        """ The offsets about the pointing of the given platepar.

        Arguments:
            platepar: [Platepar] Platepar with the reference pointing (RA_d, dec_d, pos_angle_ref), the
                refraction setting, the place and reference time (lat, lon, elev, Ho) and the scale and size of
                the image (F_scale, X_res, Y_res). It is not changed, and not kept.
        """

        # The reference pointing
        self.ra_d, self.dec_d, self.pos_angle_ref = platepar.RA_d, platepar.dec_d, platepar.pos_angle_ref

        # The refraction of the pointing as the projection does it: the hour angle of the pointing is
        #   lon + Ho - RA at any time, i.e. the local sidereal time is lon + Ho in the frame of RA_d
        self.refraction = platepar.refraction
        self.refraction_scale = refractionScale(platepar.elev)
        lat, lst = math.radians(platepar.lat), math.radians(platepar.lon + platepar.Ho)

        # Rotation from the equatorial axes to the east, north and up axes of the observer (as
        #   CyFunctions.equatorialToHorizontal, after the frame rotation)
        self.horizontal = (
            (-math.sin(lst), math.cos(lst), 0.0),
            (-math.sin(lat)*math.cos(lst), -math.sin(lat)*math.sin(lst), math.cos(lat)),
            (math.cos(lat)*math.cos(lst), math.cos(lat)*math.sin(lst), math.sin(lat)))

        # The units of the offsets: the angle of a pixel at the scale of the platepar, and of a pixel at a
        #   quarter of the image diagonal from the centre for the roll (radians)
        self.F_scale = abs(platepar.F_scale)
        self.radius = math.hypot(platepar.X_res, platepar.Y_res)/4.0
        self.shift_unit = math.radians(1.0/self.F_scale)
        self.roll_unit = 1.0/self.radius

        # The axes of the camera at the reference pointing, as the projection uses them
        ra_app, dec_app = self._refract(self.ra_d, self.dec_d, True)
        self.frame = pointingFrame(ra_app, dec_app, self.pos_angle_ref)
        self.axes = tuple(tuple(float(v) for v in self.frame[:, i]) for i in range(3))

    def _refract(self, ra_d, dec_d, true_to_apparent):
        """ The pointing refracted as the projection refracts it, or taken back through the refraction.

        Arguments:
            ra_d: [float] Right ascension (deg).
            dec_d: [float] Declination (deg).
            true_to_apparent: [bool] Refract the true pointing (True) or take the refraction out (False).

        Return:
            (ra_d, dec_d): [tuple of floats] The pointing, unchanged without the refraction (deg).
        """

        if not self.refraction:
            return ra_d, dec_d

        # The pointing in the east, north and up axes, and its elevation and azimuth
        ra, dec = math.radians(ra_d), math.radians(dec_d)
        vec = (math.cos(dec)*math.cos(ra), math.cos(dec)*math.sin(ra), math.sin(dec))
        e, n, u = [row[0]*vec[0] + row[1]*vec[1] + row[2]*vec[2] for row in self.horizontal]
        elev = math.atan2(u, math.hypot(e, n))
        azim = math.atan2(e, n)

        # Refract the elevation (the exact inverse of the correction from apparent to true), or correct it
        if true_to_apparent:
            elev = pyRefractionTrueToApparent(elev, self.refraction_scale)
        else:
            elev = pyRefractionApparentToTrue(elev, self.refraction_scale)

        # Back to the equatorial axes, and to the right ascension and declination
        vec = (math.cos(elev)*math.sin(azim), math.cos(elev)*math.cos(azim), math.sin(elev))
        x, y, z = [sum(self.horizontal[k][i]*vec[k] for k in range(3)) for i in range(3)]

        return math.degrees(math.atan2(y, x))%360.0, math.degrees(math.atan2(z, math.hypot(x, y)))

    def toPointing(self, offsets):
        """ The pointing of the camera turned by the given offsets from the reference pointing.

        Arguments:
            offsets: [list] Offsets (dx, dy, droll) (px).

        Return:
            (ra_d, dec_d, pos_angle_ref): [tuple of floats] Right ascension in [0, 360), declination in
                [-90, 90] and position angle in [0, 360) (deg).
        """

        # The +x direction and the optical axis of the turned camera, in its axes at the reference pointing
        #   and then in the equatorial axes
        x_cam, axis_cam = offsetAxes(offsets[0]*self.shift_unit, offsets[1]*self.shift_unit,
                                     offsets[2]*self.roll_unit)
        x_dir = [sum(x_cam[k]*self.axes[k][i] for k in range(3)) for i in range(3)]
        axis = [sum(axis_cam[k]*self.axes[k][i] for k in range(3)) for i in range(3)]

        # Its pointing, taken back through the refraction (the position angle is kept)
        ra_d, dec_d, pos_angle_ref = frameToPointing(x_dir, axis)
        ra_d, dec_d = self._refract(ra_d, dec_d, False)

        return ra_d, dec_d, pos_angle_ref

    def fromPointing(self, ra_d, dec_d, pos_angle_ref):
        """ The offsets of the given pointing from the reference pointing, the inverse of toPointing. The
            pointing has to be less than 180 deg from the reference.

        Arguments:
            ra_d: [float] Right ascension (deg).
            dec_d: [float] Declination (deg).
            pos_angle_ref: [float] Position angle (deg).

        Return:
            [ndarray] Offsets (dx, dy, droll) (px).
        """

        # The rotation from the axes of the reference camera to those of the given pointing
        ra_app, dec_app = self._refract(ra_d, dec_d, True)
        dx, dy, droll = rotationOffsets(self.frame.T.dot(pointingFrame(ra_app, dec_app, pos_angle_ref)))

        return np.array([dx/self.shift_unit, dy/self.shift_unit, droll/self.roll_unit])

    def toScale(self, dscale):
        """ The scale changed by the given offset from the scale of the reference platepar.

        Arguments:
            dscale: [float] Offset of the scale (px).

        Return:
            [float] The scale (px/deg).
        """

        return self.F_scale*(1.0 + dscale/self.radius)

    def fromScale(self, F_scale):
        """ The offset of the given scale from the scale of the reference platepar, the inverse of toScale.

        Arguments:
            F_scale: [float] Scale (px/deg).

        Return:
            [float] Offset of the scale (px).
        """

        return (abs(F_scale)/self.F_scale - 1.0)*self.radius

    def apply(self, platepar, offsets):
        """ Set the pointing turned by the given offsets, and the scale changed by the fourth offset if there is
            one, to a platepar.

        Arguments:
            platepar: [Platepar] Platepar whose RA_d, dec_d and pos_angle_ref (and F_scale) are set.
            offsets: [list] Offsets (dx, dy, droll) or (dx, dy, droll, dscale) (px).

        Return:
            None
        """

        platepar.RA_d, platepar.dec_d, platepar.pos_angle_ref = self.toPointing(offsets)

        if len(offsets) > 3:
            platepar.F_scale = self.toScale(offsets[3])


# A RANSAC fit (Platepar.fitAstrometry with use_nn_cost) is clearly good if at least this share of the image stars
#   ended in its matched pairs, and the final fit on the pairs left an RMS distance of at most RANSAC_GOOD_RMS_PX
RANSAC_GOOD_FRACTION = 0.6
RANSAC_GOOD_RMS_PX = 1.0


def ransacQuality(platepar, jd, pairs):
    """ The quality of the result of a RANSAC fit, to compare two of them: the number of matched pairs, and the RMS
        distance of the image stars from their catalog stars with the fitted platepar.

    Arguments:
        platepar: [Platepar or dict] The fitted platepar, or the dictionary of its attributes.
        jd: [float] Julian date of the image.
        pairs: [tuple] The matched pairs (img_stars, catalog_stars) the fit returned, or None.

    Return:
        (n_pairs, -rms): [tuple] Larger is better; (0, -inf) without a result.
    """

    if (pairs is None) or (len(pairs[0]) == 0):
        return (0, -np.inf)

    # A platepar from the dictionary of its attributes
    if isinstance(platepar, dict):
        pp = Platepar()
        pp.__dict__.update(platepar)
        platepar = pp

    img_stars, catalog_stars = pairs
    x, y, _ = getCatalogStarsImagePositions(np.asarray(catalog_stars)[:, :3], jd, platepar)
    dist_sq = (x - img_stars[:, 0])**2 + (y - img_stars[:, 1])**2
    rms = np.sqrt(np.mean(dist_sq)) if np.all(np.isfinite(dist_sq)) else np.inf

    return (len(img_stars), -rms)


def ransacResultGood(platepar, jd, img_stars, pairs):
    """ Whether the result of a RANSAC fit is clearly good: enough of the image stars in its matched pairs, with
        small residuals (RANSAC_GOOD_FRACTION, RANSAC_GOOD_RMS_PX).

    Arguments:
        platepar: [Platepar] The fitted platepar.
        jd: [float] Julian date of the image.
        img_stars: [ndarray] The image stars given to the fit.
        pairs: [tuple] The matched pairs (img_stars, catalog_stars) the fit returned, or None.

    Return:
        [bool] True if the result is clearly good.
    """

    n_pairs, neg_rms = ransacQuality(platepar, jd, pairs)

    return (n_pairs >= RANSAC_GOOD_FRACTION*len(img_stars)) and (-neg_rms <= RANSAC_GOOD_RMS_PX)


def _lstsqFit(residual_func, x0, args):
    """Levenberg-Marquardt least-squares fit on a vector-valued residual.

    Replaces Nelder-Mead on the smooth, matched-pair cost functions: it uses the residual's
    Jacobian structure, so it converges in far fewer evaluations and reaches the same (or a
    lower-residual) least-squares minimum. Uses 'lm' when there are at least as many residuals
    as parameters (lm's requirement); otherwise falls back to the robust 'trf'. Tight
    tolerances ensure full convergence.

    Arguments:
        residual_func: [callable] Returns the per-observation residual VECTOR (not the sum of
            squares). least_squares minimizes 0.5*sum(residual**2), so this must match the
            squared-and-summed scalar cost it replaces.
        x0: [array] Initial parameters.
        args: [tuple] Extra args passed to residual_func.

    Return:
        scipy.optimize.OptimizeResult
    """
    x0 = np.asarray(x0, dtype=float)
    n_resid = len(np.atleast_1d(residual_func(x0, *args)))
    method = 'lm' if n_resid >= len(x0) else 'trf'
    return scipy.optimize.least_squares(
        residual_func, x0, args=args, method=method, ftol=1e-12, xtol=1e-12, gtol=1e-12)


def _raDecToUnitVectors(ra_deg, dec_deg):
    """Convert RA/Dec arrays (degrees) to 3D unit vectors. Euclidean nearest-neighbour on unit
    vectors gives the same result as angular nearest-neighbour, so a cKDTree of these vectors
    replaces an O(N*M) angular-separation matrix with an O(N log M) query."""
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    cos_dec = np.cos(dec)
    return np.column_stack([cos_dec*np.cos(ra), cos_dec*np.sin(ra), np.sin(dec)])


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

        # Version of the file format: 1 had none, 2 added the forward and reverse distortion, 3 the fit record
        self.version = 3

        # Store distortion type for later initialization
        self._init_distortion_type = distortion_type

        # Initialize distortion-related attributes (needed by setDistortionType)
        self.equal_aspect = True
        self.force_distortion_centre = False
        self.asymmetry_corr = True

        # Decentering correction of the radial distortion (Brown-Conrady): a lens element shifted or tilted off the
        #   optical axis displaces the image by p1, p2 (amount and direction of the shift) times the square of the
        #   radius, and p3, p4 times its fourth power. Its coefficients follow the radial ones in the distortion
        #   arrays. Off by default
        self.decentering_corr = False

        # Take the sky to the image as the exact inverse of the forward distortion, instead of through the
        #   separately fitted reverse one, so that the two directions agree exactly. Off by default
        self.exact_reverse = False

        # Station coordinates
        self.lat = self.lon = self.elev = 0

        # Reference time and date
        self.time = 0
        self.JD = 2451545.0

        # UT correction
        self.UT_corr = 0

        # Reference hour angle (deg): the apparent Greenwich sidereal time at JD (mean sidereal time plus the
        #   equation of the equinoxes, see Conversions.JD2HourAngle), the sidereal time of the true-of-date
        #   reference pointing RA_d/dec_d. Recomputed from JD when a platepar is loaded
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

        # Annual aberration of light modelled in the fit. If True, the plate maps the apparent directions of
        #   the stars and the aberration is applied to (or removed from) catalog directions explicitly. If
        #   False, the plate was fitted to catalog directions as they are and has absorbed the aberration of
        #   the fit epoch in its pointing, scale and distortion
        self.aberration = True

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

        # What made the last fit of the distortion and of the pointing, when, and with which astrometric
        #   model (see fitRecordEntry and recordFit). None for a fit that is not known
        self.fit_record = emptyFitRecord()

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
            # If the distortion is radial, center may be in x_poly[0] and x_poly[1]
            if self.distortion_type.startswith("radial"):
                # Only set center coefficients if force_distortion_centre is False
                # When force_distortion_centre=True, there are no center coefficients in the array
                if not self.force_distortion_centre:
                    # Match CyFunctions which uses 0.5/(res/2) when force_distortion_centre=True
                    # After normalization (coeff * res/2), this gives x0=0.5 offset
                    self.x_poly_fwd[0] = 0.5 / (self.X_res / 2.0)
                    self.x_poly_rev[0] = 0.5 / (self.X_res / 2.0)
                    self.x_poly_fwd[1] = 0.5 / (self.Y_res / 2.0)
                    self.x_poly_rev[1] = 0.5 / (self.Y_res / 2.0)
                # y_poly stays all zeros for radial
            else:
                # For polynomial distortion, center is in x_poly[0] and y_poly[0]
                self.x_poly_fwd[0] = 0.5
                self.y_poly_fwd[0] = 0.5
                self.x_poly_rev[0] = 0.5
                self.y_poly_rev[0] = 0.5

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

            # With the decentering correction its coefficients follow the radial ones, so a change of the radial
            #   order moves them: the coefficients are read in the layout of the current type, to be rebuilt in the
            #   layout of the new one below
            remap = None
            if (not reset_params) and getattr(self, 'decentering_corr', False) \
                    and self.distortion_type.startswith("radial") and (distortion_type != self.distortion_type):
                remap = [self.extractRadialCoeffs(arr) for arr in (self.x_poly_fwd, self.x_poly_rev,
                                                                    self.y_poly_fwd, self.y_poly_rev)]

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

            # Add the decentering coefficients (p1, p2, p3, p4) for radial distortions if they are used
            if distortion_type.startswith("radial"):
                if self.decentering_corr:
                    self.poly_length += 4

        else:
            raise ValueError("The distortion type is not recognized: {:s}".format(self.distortion_type))

        # Reset distortion parameters
        if reset_params:
            self.resetDistortionParameters()

        # Rebuild the coefficients read above in the layout of the new type (radial coefficients beyond its order
        #   are dropped, the decentering coefficients kept)
        elif remap is not None:
            self.x_poly_fwd, self.x_poly_rev, self.y_poly_fwd, self.y_poly_rev = [
                self.buildRadialCoeffs(coeffs, target_dist_type=distortion_type) for coeffs in remap]

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
            # CyFunctions uses x0=0.5 offset when forced, so match that here
            return self.X_res / 2.0 + 0.5, self.Y_res / 2.0 + 0.5

        # For radial distortion, extract the center offsets from coefficients
        # The offsets are stored normalized by X_res/2 and Y_res/2
        x0 = self.x_poly_fwd[0] * (self.X_res / 2.0)
        y0 = self.x_poly_fwd[1] * (self.Y_res / 2.0)

        # Wrap offsets to always be within the image bounds
        x0 = -self.X_res / 2.0 + (x0 + self.X_res / 2.0) % self.X_res
        y0 = -self.Y_res / 2.0 + (y0 + self.Y_res / 2.0) % self.Y_res

        # Return center in pixel coordinates (image center + offset)
        return self.X_res / 2.0 + x0, self.Y_res / 2.0 + y0

    def _withReverseProjection(self, func, *args, **kwargs):
        """ Call func with the sky -> image projection through the reverse polynomial, as without exact_reverse,
            so that x_poly_rev is fitted with the meaning it has for readers that do not know the option. The
            residual functions copy the platepar on every call, and the copies inherit the switch.
        """

        exact = getattr(self, "exact_reverse", False)
        self.exact_reverse = False

        try:
            return func(*args, **kwargs)

        finally:
            self.exact_reverse = exact


    def recordFit(self, pointing=True, distortion=False, fit_by='fitAstrometry'):
        """ Record a fit made now by this code in the fit record (see fitRecordEntry).

        Keyword arguments:
            pointing: [bool] The pointing was fitted. True by default.
            distortion: [bool] The distortion was fitted. False by default.
            fit_by: [str] Short label of what made the fit. 'fitAstrometry' by default.
        """

        # A platepar made without the constructor (e.g. from an old pickle) may not have the record yet
        self.fit_record = normalFitRecord(getattr(self, 'fit_record', None))

        # One entry for the fit, the same for the distortion and the pointing if both were fitted
        entry = fitRecordEntry(fit_by)
        if distortion:
            self.fit_record['distortion'] = dict(entry)
        if pointing:
            self.fit_record['pointing'] = dict(entry)

    def astrometryModel(self):
        """ The astrometric model the distortion of the platepar was fitted with (see ASTROMETRY_MODEL).
            The pointing is refitted on every image, the distortion is what ties a platepar to the model of
            its fit.

        Return:
            [int] The model of the distortion fit, 1 if it is not known (a platepar fitted before the fit
                record existed).
        """

        # The record of the distortion fit, if there is one
        entry = normalFitRecord(getattr(self, 'fit_record', None))['distortion']
        if entry is None:
            return 1

        # A record without a valid model is taken as the oldest model
        try:
            return int(entry.get('model', 1))
        except (TypeError, ValueError):
            return 1

    def fitPointing(self, jd, img_stars, catalog_stars, fixed_scale=False, final_catalog_stars=None,
                    fit_by='fitPointing'):
        """Fit pointing parameters to the list of star image and celestial catalog coordinates.
        At least 4 stars are needed to fit the rigid body parameters.

        New parameters are saved to the given object (self), and the pointing fit is recorded in the fit
        record.

        Arguments:
            jd: [float] Julian date of the image.
            img_stars: [list] A list of (x, y, intensity_sum) entires for every star.
            catalog_stars: [list] A list of (ra, dec, mag) entries for every star (degrees).

        Keyword arguments:
            fixed_scale: [bool] Keep the scale fixed. False by default.
            final_catalog_stars: [list] Optional deeper catalog to use for final radial7 fitting.
                                 If provided, switches to this catalog after radial7 switch.
            fit_by: [str] Short label of what made the fit, for the fit record. 'fitPointing' by default.

        """

        # The pointing is fitted as offsets (px) from the pointing of the platepar, which move the field by the
        #   same amount in the same direction on the image wherever the camera points (see PointingOffsets)
        pointing = PointingOffsets(self)

        def _calcImageResidualsAstro(params, platepar, jd, catalog_stars, img_stars):
            """Calculates the differences between the stars on the image and catalog stars in image
            coordinates with the given astrometrical solution.
            """

            # Extract fitting parameters
            if not fixed_scale:
                F_scale = params[3]

            img_x, img_y, _ = img_stars.T

            pp_copy = copy.copy(platepar)

            # Assign guessed parameters: the pointing turned by the offsets
            pointing.apply(pp_copy, params[:3])

            if not fixed_scale:
                pp_copy.F_scale = abs(F_scale)

            # Get image coordinates of catalog stars
            catalog_x, catalog_y, catalog_mag = getCatalogStarsImagePositions(catalog_stars, jd, pp_copy)

            # Calculate the sum of squared distances between image stars and catalog stars. A catalog star
            #   which has no image (beyond the reach of the exact inverse of the distortion) counts as off by
            #   the image diagonal, so the sum stays finite and the fit is not drawn to lose stars
            dist_sq = (catalog_x - img_x) ** 2 + (catalog_y - img_y) ** 2
            dist_sq[~np.isfinite(dist_sq)] = platepar.X_res ** 2 + platepar.Y_res ** 2
            dist_sum = np.sum(dist_sq)

            return dist_sum

        def _calcSkyResidualsAstro(params, platepar, jd, catalog_stars, img_stars):
            """Calculates the differences between the stars on the image and catalog stars in sky
            coordinates with the given astrometrical solution.
            """

            # Extract fitting parameters
            if not fixed_scale:
                F_scale = params[3]

            img_x, img_y, _ = img_stars.T

            pp_copy = copy.copy(platepar)

            # Assign guessed parameters: the pointing turned by the offsets
            pointing.apply(pp_copy, params[:3])

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

        # Initial parameters for the astrometric fit: no offset of the pointing
        p0 = [0.0, 0.0, 0.0]

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
        pointing.apply(self, res.x[:3])
        if not fixed_scale:
            self.F_scale = res.x[3]

        # Force scale to be positive
        self.F_scale = abs(self.F_scale)

        # Update alt/az of pointing
        self.updateRefAltAz()

        # Record the pointing fit
        self.recordFit(pointing=True, fit_by=fit_by)

    def fitPointingNN(self, jd, img_stars, catalog_stars, fixed_scale=True, fit_by='fitPointingNN'):
        """Fit pointing parameters using nearest-neighbor cost function in pixel space.

        Unlike fitPointing() which requires pre-matched star pairs, this method finds the
        nearest catalog star for each detected star and optimizes to minimize the sum of
        NN distances. This is more robust when the initial pointing is off.

        Use this as a replacement for NNalign - it fits RA_d, dec_d, pos_angle_ref, and
        optionally F_scale without requiring explicit star matching.

        The cost function works in pixel space (like matchStarsResiduals) by projecting
        catalog stars onto the image and computing pixel distances. This is more robust
        than working in sky coordinates because it correctly handles distortion and timing.

        The pointing is fitted as offsets of the pointing (see PointingOffsets). If that fit is not accepted, the
        fit of RA, Dec and position angle of earlier versions is run from the given pointing, as a safety net, and
        its result is used.

        Arguments:
            jd: [float] Julian date of the image.
            img_stars: [ndarray] Detected stars as Nx3 array (x, y, intensity).
            catalog_stars: [ndarray] Catalog stars in FOV as Mx3 array (ra, dec, mag).

        Keyword arguments:
            fixed_scale: [bool] Keep scale fixed. True by default (for camera drift correction).
            fit_by: [str] Short label of what made the fit, for the fit record. 'fitPointingNN' by default.

        """

        # The fit of the offsets of the pointing, and if it is not accepted, the fit of RA, Dec and position angle
        result = self._fitPointingNNOnce(jd, img_stars, catalog_stars, fixed_scale, legacy=False)
        if not (result[0] and result[1]):
            result = self._fitPointingNNOnce(jd, img_stars, catalog_stars, fixed_scale, legacy=True)
        accepted, success, cost, inlier_fraction, inlier_rmsd, pp_fit = result

        if not accepted:
            return False, cost, inlier_fraction, 0.0

        # Update fitted parameters
        self.RA_d, self.dec_d, self.pos_angle_ref = pp_fit.RA_d, pp_fit.dec_d, pp_fit.pos_angle_ref
        if not fixed_scale:
            self.F_scale = pp_fit.F_scale

        # Update JD and Ho to match the observation time
        # This ensures the fitted RA_d is consistent with the new reference time
        self.JD = jd
        self.Ho = JD2HourAngle(jd)

        # Update alt/az of pointing
        self.updateRefAltAz()

        # Record the pointing fit
        self.recordFit(pointing=True, fit_by=fit_by)

        return success, cost, inlier_fraction, inlier_rmsd

    def _fitPointingNNOnce(self, jd, img_stars, catalog_stars, fixed_scale, legacy=False):
        """ One NN fit of the pointing (see fitPointingNN), on a copy of the platepar.

        Arguments:
            jd: [float] Julian date of the image.
            img_stars: [ndarray] Detected stars as Nx3 array (x, y, intensity).
            catalog_stars: [ndarray] Catalog stars in FOV as Mx3 array (ra, dec, mag).
            fixed_scale: [bool] Keep scale fixed.

        Keyword arguments:
            legacy: [bool] Fit RA, Dec and position angle from the given pointing, as earlier versions did, instead
                of the offsets of the pointing. False by default.

        Return:
            (accepted, success, cost, inlier_fraction, inlier_rmsd, platepar): [tuple] Whether enough stars have a
                catalog star nearby (inliers), whether the optimizer converged, its cost, the inlier fraction and
                RMSD (px), and the copy of the platepar with the fitted pointing at the time of the image.
        """
        # Import here to avoid circular import (ApplyAstrometry imports Platepar)
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
        new_Ho = JD2HourAngle(jd)
        pp_work.Ho = new_Ho

        # Compute delta Ho and adjust RA_d to preserve pointing direction
        delta_Ho = (new_Ho - old_Ho) % 360
        pp_work.RA_d = (self.RA_d + delta_Ho) % 360

        # Pre-extract catalog coordinates (constant across iterations)
        ra_catalog, dec_catalog, _ = catalog_stars.T

        # Pre-extract detected star positions (constant across iterations)
        img_x, img_y, _ = img_stars.T

        # The pointing is fitted as offsets (px) from the starting pointing at the time of the image, which
        #   move the field by the same amount in the same direction on the image wherever the camera points
        #   (see PointingOffsets)
        pointing = PointingOffsets(pp_work)

        def _setNNPointing(pp, params):
            """ Set the pointing of the given platepar from the fit parameters.

            Arguments:
                pp: [Platepar] Platepar which is updated in place.
                params: [list] Offsets of the pointing (dx, dy, droll, px, see PointingOffsets), or with legacy
                    RA_d, dec_d and pos_angle_ref, and if the scale is not fixed, F_scale.

            Return:
                None
            """

            if legacy:

                # The declination is stepped freely, so a pointing past a pole is reflected back onto the sphere
                #   and the position angle turned with it
                pp.RA_d, pp.dec_d, pos_angle_offset = normalizeRaDec(params[0], params[1],
                                                                     refraction=pp.refraction)
                pp.pos_angle_ref = (params[2] + pos_angle_offset)%360

            else:
                pointing.apply(pp, params[:3])

            if not fixed_scale:
                pp.F_scale = abs(params[3])

        def _calcPointingNNCostPixel(params, pp_work, jd, ra_catalog, dec_catalog, img_x, img_y, fixed_scale):
            """NN cost function in pixel space for pointing fit.

            Projects catalog stars to image coordinates and computes pixel-space NN distances.
            This is more robust than sky-space comparison because it correctly handles
            distortion and timing through the standard coordinate transforms.

            Uses a pre-allocated working platepar to avoid deepcopy overhead.
            """

            # Update working platepar with current parameters (no copy needed)
            _setNNPointing(pp_work, params)

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

        # Initial parameters: no offset from the pointing of pp_work, which has been adjusted for the new JD/Ho.
        #   The first simplex moves the pointing by 1 deg on the sky along x and y and rolls it by 2 deg, for the
        #   large corrections this fit is used for, and changes the scale by 10%
        if not legacy:
            p0 = [0.0, 0.0, 0.0]
            steps = [np.radians(1.0)/pointing.shift_unit]*2 + [np.radians(2.0)/pointing.roll_unit]

        # The fit of earlier versions: RA, Dec and position angle, from a simplex of 2 deg in Dec and the position
        #   angle and 2/cos(dec) deg in RA (cos(dec) at least 0.1)
        else:
            p0 = [pp_work.RA_d, self.dec_d, self.pos_angle_ref]
            steps = [2.0/max(np.cos(np.radians(self.dec_d)), 0.1), 2.0, 2.0]

        if not fixed_scale:
            p0.append(abs(self.F_scale))
            steps.append(0.1*abs(self.F_scale))
        simplex = np.vstack([p0, np.array(p0) + np.diag(steps)])

        # Fit using Nelder-Mead (robust for NN cost landscape)
        res = scipy.optimize.minimize(
            _calcPointingNNCostPixel,
            p0,
            args=(pp_work, jd, ra_catalog, dec_catalog, img_x, img_y, fixed_scale),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'adaptive': True, 'initial_simplex': simplex},
        )

        # Inliers: image stars with a catalog star within this distance (px); at least this share of the
        #   detected stars has to be one for an accepted fit
        inlier_threshold = 5.0
        min_inlier_fraction = 0.5

        def _inliers(pp):
            """ The NN distances of the image stars (px) and the indices of their nearest catalog stars inside the
                image with the given platepar, or None with fewer than 3 such catalog stars. """

            cat_x, cat_y = raDecToXYPP(ra_catalog, dec_catalog, jd, pp)
            valid = np.where((cat_x >= 0) & (cat_x < pp.X_res) & (cat_y >= 0) & (cat_y < pp.Y_res))[0]
            if len(valid) < 3:
                return None

            nn_distances, nearest = cKDTree(np.column_stack([cat_x[valid], cat_y[valid]])).query(
                np.column_stack([img_x, img_y]), k=1)

            return nn_distances, valid[np.minimum(nearest, len(valid) - 1)]

        def _inlierStats(nn_distances):
            """ The number of inliers and their RMS distance (px). """

            inlier_mask = nn_distances < inlier_threshold
            n_inliers = int(np.sum(inlier_mask))
            rmsd = np.sqrt(np.mean(nn_distances[inlier_mask]**2)) if n_inliers > 0 else 0.0

            return n_inliers, rmsd

        # Check inlier fraction - more robust than RMSD which is skewed by outliers
        # Recompute NN distances with fitted parameters
        _setNNPointing(pp_work, res.x)
        nn_fit = _inliers(pp_work)
        if nn_fit is None:
            return False, False, res.fun, 0.0, 0.0, pp_work
        nn_distances, _ = nn_fit
        n_inliers, inlier_rmsd = _inlierStats(nn_distances)

        inlier_fraction = n_inliers/len(nn_distances)

        return (inlier_fraction >= min_inlier_fraction), res.success, res.fun, inlier_fraction, inlier_rmsd, \
            pp_work

    def fitAstrometry(
        self,
        jd,
        img_stars,
        catalog_stars,
        first_platepar_fit=False,
        fit_only_pointing=False,
        fixed_scale=False,
        use_nn_cost=False,
        final_catalog_stars=None,
        iteration_callback=None,
        fit_by='fitAstrometry',
    ):
        """Fit astrometric parameters to the list of star image and celestial catalog coordinates.
        At least 4 stars are needed to fit the rigid body parameters.

        New parameters are saved to the given object (self). The fit models the annual aberration: the
        aberration flag is turned on where the pointing is fitted. A fit that stops without a result or fails
        with an exception leaves the flag as it was. What was fitted (the pointing, and the distortion unless
        only the pointing is fitted) is recorded in the fit record.

        A NN fit (use_nn_cost) that stops without a result or fails with an exception leaves the whole platepar
        as it was, with an empty star list, as no star pairs came out of it. The NN fit only works with a radial
        distortion and enough stars for it, and is not run for a fit of the pointing alone.

        Arguments:
            jd: [float] Julian date of the image.
            img_stars: [list] A list of (x, y, intensity_sum) entires for every star.
            catalog_stars: [list] A list of (ra, dec, mag) entries for every star (degrees).

        Keyword arguments:
            first_platepar_fit: [bool] Fit a platepar from scratch. False by default.
            fit_only_pointing: [bool] Only fit the pointing parameters, and not distortion.
            fixed_scale: [bool] Keep the scale fixed. False by default.
            use_nn_cost: [bool] Fit on the nearest catalog neighbours of the stars (RANSAC) instead of given
                pairs. False by default.
            final_catalog_stars: [list] Optional deeper catalog to use for final radial7 fitting.
                                 If provided, switches to this catalog after radial7 switch.
            iteration_callback: [callable] Optional callback called after each RANSAC iteration.
                                Signature: callback(iteration, platepar_copy, outlier_mask, rmsd_arcmin)
                                Used for visual debugging of the fitting process.
            fit_by: [str] Short label of what made the fit, for the fit record (e.g. 'SkyFit2').
                'fitAstrometry' by default.

        Return:
            None, or with use_nn_cost the matched pairs (img_stars, catalog_stars) of the RANSAC fit (None if
                it stopped without a result, the platepar then being as it was before the fit).

        """

        # The aberration flag of the plate before the fit, restored if the fit fails with an exception (a fit
        #   that stops without a result restores it itself)
        aberration_prev = self.aberration
        completed = False

        # The NN fit changes the distortion type, the pointing and the distortion while it runs, so the whole
        #   state of the plate before it is kept, for a NN fit that stops without a result or fails
        state_prev = copy.deepcopy(self.__dict__) if use_nn_cost else None

        try:
            fit_args = dict(first_platepar_fit=first_platepar_fit, fit_only_pointing=fit_only_pointing,
                fixed_scale=fixed_scale, use_nn_cost=use_nn_cost, final_catalog_stars=final_catalog_stars,
                iteration_callback=iteration_callback, fit_by=fit_by)
            result = self._fitAstrometry(jd, img_stars, catalog_stars, **fit_args)

            # Safety net of the RANSAC fit: if its result is not clearly good, the RANSAC fit of earlier versions
            #   (RA, Dec and position angle from the given pointing) is run as well, from the platepar as it was
            #   given, and the result with more matched pairs (and then the smaller residuals) is kept
            if use_nn_cost and (not ransacResultGood(self, jd, img_stars, result)):
                print("    RANSAC result not clearly good, running the RANSAC fit of RA, Dec and position angle")
                result_new, state_new = result, copy.deepcopy(self.__dict__)
                self.__dict__.clear()
                self.__dict__.update(copy.deepcopy(state_prev))
                result = self._fitAstrometry(jd, img_stars, catalog_stars, legacy=True, **fit_args)

                # The new result is kept unless the other one is better
                if ransacQuality(self, jd, result) <= ransacQuality(state_new, jd, result_new):
                    result = result_new
                    self.__dict__.clear()
                    self.__dict__.update(state_new)

            completed = True

        finally:
            if not completed:
                self.aberration = aberration_prev

            # A NN fit without a result puts the plate back as it was. No star pairs came out of it, so the star
            #   list is emptied (callers read it as the pairs of the fit)
            if use_nn_cost and ((not completed) or (result is None)):
                self.__dict__.clear()
                self.__dict__.update(state_prev)
                self.star_list = []

        return result

    def _fitAstrometry(self, jd, img_stars, catalog_stars, first_platepar_fit=False, fit_only_pointing=False,
        fixed_scale=False, use_nn_cost=False, final_catalog_stars=None, iteration_callback=None,
        fit_by='fitAstrometry', legacy=False):
        """ The fit of fitAstrometry, which has the description and the arguments. With legacy, the RANSAC fit
            (use_nn_cost) fits RA, Dec and position angle from the given pointing, as earlier versions did, instead
            of the offsets of the pointing.

        Return:
            See fitAstrometry.

        """

        def _calcImageResidualsDistortion(params, platepar, jd, catalog_stars, img_stars, dimension):
            """Calculates the differences between the stars on the image and catalog stars in image
                coordinates with the given astrometrical solution.
            Arguments:
                ...
                dimension: [str] 'x' for X polynomial fit, 'y' for Y polynomial fit
            """

            # Set distortion parameters
            pp_copy = copy.copy(platepar)

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

        def _calcImageResidualsDistortionVect(params, platepar, jd, catalog_stars, img_stars, dimension):
            """Vector-residual form of _calcImageResidualsDistortion for scipy.least_squares.
               Returns per-star residuals; least_squares minimizes their sum of squares, which
               equals the scalar dist_sum above, so the optimum is identical."""

            pp_copy = copy.copy(platepar)

            if (dimension == 'x') or (dimension == 'radial'):
                pp_copy.x_poly_rev = params
                pp_copy.y_poly_rev = np.zeros(platepar.poly_length)
            else:
                pp_copy.x_poly_rev = np.zeros(platepar.poly_length)
                pp_copy.y_poly_rev = params

            img_x, img_y, _ = img_stars.T
            catalog_x, catalog_y, _ = getCatalogStarsImagePositions(catalog_stars, jd, pp_copy)

            if dimension == 'x':
                return catalog_x - img_x
            elif dimension == 'y':
                return catalog_y - img_y
            else:
                return np.concatenate([catalog_x - img_x, catalog_y - img_y])

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

            pp_copy = copy.copy(platepar)

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

        def _calcSkyResidualsDistortionVect(params, platepar, jd, catalog_stars, img_stars, dimension):
            """Vector-residual form of _calcSkyResidualsDistortion for scipy.least_squares.
               Returns per-star angular separations (sum of squares == the scalar cost above)."""

            pp_copy = copy.copy(platepar)

            if (dimension == 'x') or (dimension == 'radial'):
                pp_copy.x_poly_fwd = params
            else:
                pp_copy.y_poly_fwd = params

            img_x, img_y, _ = img_stars.T
            ra_array, dec_array = getPairedStarsSkyPositions(img_x, img_y, jd, pp_copy)
            ra_catalog, dec_catalog, _ = catalog_stars.T

            return angularSeparation(
                np.radians(ra_array), np.radians(dec_array),
                np.radians(ra_catalog), np.radians(dec_catalog),
            )

        # Modify the residuals function so that it takes a list of arguments
        def _calcSkyResidualsDistortionListArguments(params, *args, **kwargs):
            return [_calcSkyResidualsDistortion(param_line, *args, **kwargs) for param_line in params]

        # The pointing of the radial fits is fitted as offsets from a reference pointing (see PointingOffsets),
        #   which move the field by the same amount in the same direction on the image wherever the camera
        #   points. The optimizers get them as the parameters 1 + offset/diagonal (offset and image diagonal in
        #   px): both optimizers step a parameter in proportion to its value (the finite differences of the least
        #   squares fits, the first simplex of Nelder-Mead), which for a parameter of about 1 moves the stars by
        #   about 2e-5 px for the finite differences, well above the rounding of the projection, and the
        #   tolerance of Nelder-Mead (1e-4) is 1e-4 of the image diagonal (0.15 px on a 720p image)
        diagonal = np.hypot(self.X_res, self.Y_res)
        pointing_start = [1.0, 1.0, 1.0]

        def _setPointingAndScale(pp, params, pointing):
            """ Set the pointing and the scale of the fit parameters to a platepar.

            Arguments:
                pp: [Platepar] Platepar which is updated in place.
                params: [list] Fit parameters: the three pointing parameters, the scale and the distortion.
                pointing: [PointingOffsets] Offsets about the reference pointing of the fit, or None for the
                    parameters of the RANSAC fit of earlier versions (RA/360, Dec/90 and position angle/360).

            Return:
                None
            """

            # The parameters of earlier versions: the declination is stepped freely, so a pointing past a pole is
            #   reflected back onto the sphere and the position angle turned with it
            if pointing is None:
                pp.RA_d, pp.dec_d, pos_angle_offset = normalizeRaDec(360*params[0], 90*params[1],
                                                                     refraction=pp.refraction)
                pp.pos_angle_ref = (360*params[2] + pos_angle_offset)%360

            else:
                pointing.apply(pp, [(param - 1.0)*diagonal for param in params[:3]])

            pp.F_scale = f_scale_fixed if fixed_scale else abs(params[3])

        def ransacStart():
            """ The fit parameters of the RANSAC fit at the pointing, scale and distortion of the platepar: no
                offset of the pointing, or with legacy RA/360, Dec/90 and position angle/360. """

            if legacy:
                start = [self.RA_d/360.0, self.dec_d/90, self.pos_angle_ref/360.0]
            else:
                start = list(pointing_start)

            return start + [self.F_scale] + self.x_poly_fwd.tolist()

        def _calcImageResidualsAstroAndDistortionRadial(params, platepar, jd, catalog_stars, img_stars, pointing):
            """Calculates the differences between the stars on the image and catalog stars in image
                coordinates with the given astrometrical solution. Pointing and distortion parameters are used
                in the fit.

            Arguments:
                ...
                dimension: [str] 'x' for X polynomial fit, 'y' for Y polynomial fit
            """

            # Shallow copy: the residual only rebinds scalar/poly attributes below and the
            # projection reads the rest, so a full deepcopy (per optimizer eval) is wasted.
            pp_copy = copy.copy(platepar)

            # Assign the pointing and the scale to the copy of platepar used for the fit
            _setPointingAndScale(pp_copy, params, pointing)

            # Assign distortion parameters
            pp_copy.x_poly_rev = params[4:]

            img_x, img_y, _ = img_stars.T

            # Get image coordinates of catalog stars
            catalog_x, catalog_y, catalog_mag = getCatalogStarsImagePositions(catalog_stars, jd, pp_copy)

            # Calculate the sum of squared distances between image stars and catalog stars
            dist_sum = np.sum((catalog_x - img_x) ** 2 + (catalog_y - img_y) ** 2)

            return dist_sum

        def _calcImageResidualsAstroAndDistortionRadialVect(params, platepar, jd, catalog_stars, img_stars,
                                                            pointing):
            """Vector-residual form of _calcImageResidualsAstroAndDistortionRadial for least_squares:
               fits pointing + reverse distortion on the per-star pixel residual."""

            pp_copy = copy.copy(platepar)
            _setPointingAndScale(pp_copy, params, pointing)
            pp_copy.x_poly_rev = np.array(params[4:])

            img_x, img_y, _ = img_stars.T
            catalog_x, catalog_y, _ = getCatalogStarsImagePositions(catalog_stars, jd, pp_copy)

            return np.concatenate([catalog_x - img_x, catalog_y - img_y])

        def _calcSkyResidualsAstroAndDistortionRadial(params, platepar, jd, catalog_stars, img_stars, pointing):
            """Calculates the differences between the stars on the image and catalog stars in celestial
            coordinates with the given astrometrical solution. Pointing and distortion parameters are used
            in the fit.

            """

            # Shallow copy: the residual only rebinds scalar/poly attributes below and the
            # projection reads the rest, so a full deepcopy (per optimizer eval) is wasted.
            pp_copy = copy.copy(platepar)

            # Assign the pointing and the scale to the copy of platepar used for the fit
            _setPointingAndScale(pp_copy, params, pointing)

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

        def _calcSkyResidualsAstroAndDistortionRadialVect(params, platepar, jd, catalog_stars, img_stars,
                                                          pointing):
            """Vector-residual form of _calcSkyResidualsAstroAndDistortionRadial for scipy.least_squares.
               Returns per-star angular separations; least_squares minimizes their sum of squares,
               which equals the scalar separation_sum above, so the optimum is identical."""

            pp_copy = copy.copy(platepar)

            _setPointingAndScale(pp_copy, params, pointing)
            pp_copy.x_poly_fwd = np.array(params[4:])

            img_x, img_y, _ = img_stars.T
            ra_array, dec_array = getPairedStarsSkyPositions(img_x, img_y, jd, pp_copy)
            ra_catalog, dec_catalog, _ = catalog_stars.T

            return angularSeparation(
                np.radians(ra_array), np.radians(dec_array),
                np.radians(ra_catalog), np.radians(dec_catalog),
            )

        def _calcSkyResidualsAstroAndDistortionRadialNN(params, platepar, jd, catalog_stars, img_stars,
                                                        pointing, cat_tree=None):
            """Like _calcSkyResidualsAstroAndDistortionRadial but uses nearest-neighbor matching.

            Instead of requiring pre-matched pairs, this finds the nearest catalog star for each
            detected star and sums those distances. This eliminates the need for explicit matching.

            Args:
                params: [pointing parameters (3, 1 + offset/diagonal), F_scale, x_poly_fwd...]
                platepar: Platepar object
                jd: Julian date
                catalog_stars: Pre-filtered catalog stars in FOV (Mx3 array [ra, dec, mag])
                img_stars: detected image stars (Nx3 array [x, y, intensity])
                pointing: PointingOffsets about the reference pointing of the fit

            Returns:
                Sum of squared angular separations to nearest catalog star for each detected star
            """
            # Set distortion parameters
            pp_copy = copy.copy(platepar)

            # Assign the pointing and the scale to the copy of platepar used for the fit
            _setPointingAndScale(pp_copy, params, pointing)

            # Assign distortion parameters
            pp_copy.x_poly_fwd = np.array(params[4:])

            img_x, img_y, _ = img_stars.T

            # Convert detected image positions to sky coordinates
            ra_det, dec_det = getPairedStarsSkyPositions(img_x, img_y, jd, pp_copy)

            if cat_tree is not None:
                # Fast path: query a prebuilt KD-tree of catalog unit vectors. Same nearest
                # neighbour as the matrix below, but O(N log M) instead of O(N*M) per eval.

                # The tree query rejects non-finite points (a star without a sky position with the trial
                #   parameters), so only the finite ones are queried, the others are given a separation below
                det_vectors = _raDecToUnitVectors(ra_det, dec_det)
                finite = np.all(np.isfinite(det_vectors), axis=1)
                nn_distances = np.full(len(det_vectors), np.nan)
                if np.any(finite):
                    chord_dist, _ = cat_tree.query(det_vectors[finite], k=1)
                    nn_distances[finite] = 2.0*np.arcsin(np.clip(chord_dist/2.0, 0.0, 1.0))  # chord -> angle
            else:
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

            # A star without a finite sky position is given the largest possible separation, so the cost stays
            #   finite and steers the optimizer away from such parameters
            nn_distances = np.where(np.isfinite(nn_distances), nn_distances, np.pi)

            total_cost = np.sum(nn_distances ** 2)

            return total_cost

        # print('ASTRO', _calcImageResidualsAstro([self.RA_d, self.dec_d,
        #     self.pos_angle_ref, self.F_scale],  catalog_stars, img_stars))

        # print('DIS_X', _calcImageResidualsDistortion(self.x_poly_rev,  catalog_stars, \
        #     img_stars, 'x'))

        # print('DIS_Y', _calcImageResidualsDistortion(self.y_poly_rev,  catalog_stars, \
        #     img_stars, 'y'))

        # The distortion is fitted below unless only the pointing is fitted, or there are too few stars for it
        min_fit_stars = self.poly_length + 1
        fit_distortion = (len(img_stars) >= min_fit_stars) and (not fit_only_pointing)

        # The NN fit only exists as the RANSAC of the radial distortion fit. With a polynomial distortion, too few
        #   stars for the distortion, or a fit of the pointing alone, the stars would be fitted to the catalog stars
        #   of the same index, which are not their matches, so such a NN fit is stopped before any fit
        if use_nn_cost and (self.distortion_type.startswith("poly") or (not fit_distortion)):
            print("    -> NN fit not run ({:s}, {:d} stars, {:d} needed, fit_only_pointing={})".format(
                self.distortion_type, len(img_stars), min_fit_stars, fit_only_pointing))
            return None

        # The fit models the aberration, whatever the plate had absorbed before: the flag is turned on below
        #   where the pointing is fitted. The flag before the fit is kept for the fits that stop without a
        #   result
        aberration_prev = self.aberration

        # What this fit changes, for the fit record at the end (the RANSAC fit records it in its final fit)
        pointing_fitted = False
        distortion_fitted = False

        ### ASTROMETRIC PARAMETERS FIT ###

        # Fit the pointing parameters (RA, Dec, rotation, scale)
        #   Only do the fit for the polynomial distortion model, or if only pointing is requested
        #   For radial distortion, pointing is fitted jointly with distortion - no separate step needed
        if (
            self.distortion_type.startswith("poly")
            or fit_only_pointing
        ) and not use_nn_cost:

            # The pointing is fitted with the aberration modelled
            self.aberration = True

            # With the exact reverse, the polynomial models first go through their usual sequence, which needs
            #   the reverse polynomial (see the refinement after the forward fit). A fit of the pointing alone
            #   (also with too few stars for the distortion) goes through the mapping of the platepar
            if getattr(self, "exact_reverse", False) and self.distortion_type.startswith("poly") \
                and fit_distortion:

                self._withReverseProjection(self.fitPointing, jd, img_stars, catalog_stars, fixed_scale=fixed_scale,
                    fit_by=fit_by)

            else:
                self.fitPointing(jd, img_stars, catalog_stars, fixed_scale=fixed_scale, fit_by=fit_by)
            pointing_fitted = True

        ### ###

        ### DISTORTION FIT ###

        # Fit the polynomial distortion parameters if there are enough picked stars
        if fit_distortion:

            # The distortion, and with the radial models the pointing as well, is fitted with the aberration
            #   modelled
            self.aberration = True
            distortion_fitted = True
            if not self.distortion_type.startswith("poly"):
                pointing_fitted = True

            # Fit the polynomial distortion
            if self.distortion_type.startswith("poly"):

                ### REVERSE MAPPING FIT ###

                # Fit distortion parameters in X direction, reverse mapping
                res = self._withReverseProjection(
                    _lstsqFit,
                    _calcImageResidualsDistortionVect,
                    self.x_poly_rev,
                    (self, jd, catalog_stars, img_stars, 'x'),
                )

                # Extract fitted X polynomial
                self.x_poly_rev = res.x

                # Fit distortion parameters in Y direction, reverse mapping
                res = self._withReverseProjection(
                    _lstsqFit,
                    _calcImageResidualsDistortionVect,
                    self.y_poly_rev,
                    (self, jd, catalog_stars, img_stars, 'y'),
                )

                # Extract fitted Y polynomial
                self.y_poly_rev = res.x

                ### ###

                # Always seed the forward fit from the freshly-fit reverse polynomial. The reverse
                # fit is robust (pixel residual) and forward ~ reverse for the distortion, so this
                # is a good starting point. Seeding from a stale/garbage x_poly_fwd (e.g. on a
                # manual re-fit, or after a distortion-type change) can leave the forward fit stuck
                # far from the solution because the sky-residual objective saturates -- producing a
                # broken forward mapping (catalog overlay far off) while the reverse, and the RMSD,
                # still look fine.
                self.x_poly_fwd = np.array(self.x_poly_rev)
                self.y_poly_fwd = np.array(self.y_poly_rev)

                ### FORWARD MAPPING FIT ###

                # Fit distortion parameters in X direction, forward mapping
                res = _lstsqFit(
                    _calcSkyResidualsDistortionVect,
                    self.x_poly_fwd,
                    (self, jd, catalog_stars, img_stars, 'x'),
                )

                # Extract fitted X polynomial
                self.x_poly_fwd = res.x

                # Fit distortion parameters in Y direction, forward mapping
                res = _lstsqFit(
                    _calcSkyResidualsDistortionVect,
                    self.y_poly_fwd,
                    (self, jd, catalog_stars, img_stars, 'y'),
                )

                # IMPORTANT NOTE - the X polynomial is used to store the fit parameters
                self.y_poly_fwd = res.x

                # With the exact reverse, the pointing above was fitted through the reverse polynomial. Fit it
                #   through the exact inverse, and the forward polynomial on the sky with it, in turn until
                #   neither moves: the two then describe one mapping
                if getattr(self, "exact_reverse", False):

                    for _ in range(10):

                        prev = np.concatenate([[self.RA_d, self.dec_d, self.pos_angle_ref],
                            self.x_poly_fwd, self.y_poly_fwd])

                        self.fitPointing(jd, img_stars, catalog_stars, fixed_scale=fixed_scale)

                        res = _lstsqFit(
                            _calcSkyResidualsDistortionVect,
                            self.x_poly_fwd,
                            (self, jd, catalog_stars, img_stars, 'x'),
                        )
                        self.x_poly_fwd = res.x

                        res = _lstsqFit(
                            _calcSkyResidualsDistortionVect,
                            self.y_poly_fwd,
                            (self, jd, catalog_stars, img_stars, 'y'),
                        )
                        self.y_poly_fwd = res.x

                        change = np.max(np.abs(np.concatenate([[self.RA_d, self.dec_d, self.pos_angle_ref],
                            self.x_poly_fwd, self.y_poly_fwd]) - prev))
                        if change < 1e-9:
                            break

                    # Refit the reverse polynomial to the final pointing, as it is fitted without the option, for
                    #   readers that do not know it (the pointing moved since the reverse fit above)
                    res = self._withReverseProjection(
                        _lstsqFit,
                        _calcImageResidualsDistortionVect,
                        self.x_poly_rev,
                        (self, jd, catalog_stars, img_stars, 'x'),
                    )
                    self.x_poly_rev = res.x

                    res = self._withReverseProjection(
                        _lstsqFit,
                        _calcImageResidualsDistortionVect,
                        self.y_poly_rev,
                        (self, jd, catalog_stars, img_stars, 'y'),
                    )
                    self.y_poly_rev = res.x

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
                # Hold F_scale when fixed_scale is set. The radial joint fit below otherwise carries F_scale
                # as a free parameter; the patched residual functions make the objective independent of it,
                # and every extraction/temporary below reads this held value instead of the (drifting under
                # Nelder-Mead) parameter. fixed_scale is forwarded through the recursive matched-pair fit too.
                f_scale_fixed = abs(self.F_scale)
                p0 = pointing_start + [abs(self.F_scale)]
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

                    # Get detected star positions (used for radial distance computation)
                    img_x_all, img_y_all, _ = img_stars.T

                    # Initial radial_scale using image center (updated in loop with distortion center)
                    cx_init, cy_init = self.X_res / 2.0, self.Y_res / 2.0
                    r_init = np.sqrt((img_x_all - cx_init)**2 + (img_y_all - cy_init)**2)
                    r_max = np.sqrt(cx_init**2 + cy_init**2)  # Corner distance
                    radial_scale = 1.0 + (r_init / r_max)  # Range [1.0, 2.0]

                    # Save original distortion type for final fit
                    original_dist_type = self.distortion_type

                    # Keep track of the best result across all stages, with the offsets of the pointing its
                    #   parameters are about
                    best_res = None
                    best_pointing = None
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
                        self.aberration = aberration_prev
                        return None
                    if n_catalog_fov < min_stars_required:
                        print("    -> Not enough catalog stars in FOV ({} < {}), skipping NN fit".format(
                            n_catalog_fov, min_stars_required))
                        self.aberration = aberration_prev
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

                    # The parameters at the start of the stage, with the pointing of the platepar
                    p0_current = ransacStart()

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
                                _setPointingAndScale(self, best_res.x, best_pointing)
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

                            p0_current = ransacStart()
                            best_res = None  # Reset best result for new distortion type
                            best_cost = float('inf')
                            print("      --- Switching to radial5-odd (warm start with coefficient conversion) ---")

                        # Switch to radial7-odd after iteration 14 with warm start
                        elif iteration == switch_iter_r7:
                            # Save current best params from radial5-odd
                            if best_res is not None:
                                _setPointingAndScale(self, best_res.x, best_pointing)
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

                            p0_current = ransacStart()
                            best_res = None  # Reset best result for new distortion type
                            best_cost = float('inf')
                            print("      --- Switching to radial7-odd (warm start with coefficient conversion) ---")

                            # Switch to deeper catalog for final iterations if provided
                            if final_catalog_stars is not None:
                                catalog_stars = final_catalog_stars
                                print(f"      --- Using final catalog ({len(catalog_stars)} stars) for radial7 fitting ---")

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
                            # The stars are selected uniformly if their intensities cannot be used as weights (a
                            #   median intensity <= 0, or NaN intensities)
                            if not ((intensity_median > 0) and np.all(np.isfinite(weights))):
                                weights = None

                            # Draw with a local generator seeded per iteration (the same draws as seeding the
                            #   global one), so the fit leaves the global NumPy random state alone
                            rng = np.random.RandomState(42 + iteration)
                            subset_indices = rng.choice(available_indices, n_subset, replace=False, p=weights)

                        img_stars_subset = img_stars[subset_indices]

                        # Start from the best fit of the stage so far, or from the start of the stage. The fit is
                        #   centred on its starting pointing: the pointing parameters are offsets about it
                        pp_filter = copy.deepcopy(self)
                        start_params = p0_current
                        pointing = None
                        if legacy:
                            if (best_res is not None) and (len(best_res.x) == len(p0_current)):
                                start_params = best_res.x
                            _setPointingAndScale(pp_filter, start_params, None)
                        else:
                            if (best_res is not None) and (len(best_res.x) == len(p0_current)):
                                _setPointingAndScale(pp_filter, best_res.x, best_pointing)
                                start_params = pointing_start + list(best_res.x[3:])
                            pointing = PointingOffsets(pp_filter)

                        # Pre-filter catalog to FOV using current pointing (from start_params)
                        # This avoids re-filtering inside every optimizer function evaluation
                        pp_filter.F_scale = f_scale_fixed if fixed_scale else abs(start_params[3])
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
                        # Kept on Nelder-Mead: the NN cost is non-smooth (nearest-neighbour
                        # assignment jumps as params move), so a gradient/least-squares method
                        # is unreliable here. The smooth matched-pair paths use _lstsqFit.
                        # Speedup: build a catalog KD-tree once per iteration so each NN cost
                        # eval is O(N log M) instead of rebuilding an N×M matrix (~80x fewer
                        # ops/eval, identical nearest-neighbour result).
                        ra_cat_fov, dec_cat_fov, _ = catalog_stars_fov.T
                        cat_tree = cKDTree(_raDecToUnitVectors(ra_cat_fov, dec_cat_fov))

                        # The first simplex moves the pointing by 1% of the image diagonal in each offset (about
                        #   15 px on a 720p image), as the fit starts from a plate solution or a fitted platepar,
                        #   and the nearest catalog stars of a wider simplex are other stars. The scale and the
                        #   distortion take Nelder-Mead's default steps (5% of the value, 0.00025 for a zero)
                        #   The fit of earlier versions takes Nelder-Mead's default first simplex
                        ransac_steps = [0.01]*3 + [0.05*value if value != 0 else 0.00025
                                                   for value in start_params[3:]]
                        ransac_simplex = np.vstack([start_params,
                                                    np.array(start_params) + np.diag(ransac_steps)])
                        res = scipy.optimize.minimize(
                            cost_func, start_params,
                            args=(self, jd, catalog_stars_fov, img_stars_subset, pointing, cat_tree),
                            method='Nelder-Mead',
                            options=ransac_opts if legacy else dict(ransac_opts, initial_simplex=ransac_simplex),
                        )

                        # Debug: show optimizer exit reason
                        exit_reason = "maxiter" if res.nit >= ransac_opts['maxiter'] else "converged"
                        print("        opt: {} iters, {} fev, {} (fatol={})".format(
                            res.nit, res.nfev, exit_reason, ransac_opts['fatol']))

                        # Score on ALL stars
                        pp_temp = copy.deepcopy(self)
                        _setPointingAndScale(pp_temp, res.x, pointing)
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

                        # Without catalog stars in the field of this fit no star has a nearest one
                        if len(catalog_stars_iter) > 0:
                            nn_seps = np.min(sep_matrix, axis=1)  # (N,)
                        else:
                            nn_seps = np.full(len(img_stars), np.nan)

                        # Flag the stars further from their nearest catalog star than 3x the median, with the
                        #   threshold relaxed towards the image corners. A star without a finite separation (no
                        #   sky position, or no catalog stars in the field) is always an outlier
                        finite_seps = np.isfinite(nn_seps)
                        median_sep = np.median(nn_seps[finite_seps]) if np.any(finite_seps) else np.inf
                        base_threshold = 3.0 * median_sep
                        per_star_threshold = base_threshold * radial_scale
                        iteration_outliers = ~(nn_seps <= per_star_threshold)

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
                            best_pointing = pointing

                        if iteration < switch_iter_r5:
                            dist_label = "radial3-odd"
                        elif iteration < switch_iter_r7:
                            dist_label = "radial5-odd"
                        else:
                            dist_label = "radial7-odd"
                        # Debug: show RA/Dec at each iteration
                        print("      Iter {}: {} (w={}) fit on {}, {} outliers, RMSD={:.2f}', RA={:.2f} Dec={:.2f}".format(
                            iteration + 1, dist_label, weight, len(subset_indices),
                            np.sum(iteration_outliers), rmsd_arcmin, pp_temp.RA_d, pp_temp.dec_d))

                        # Call iteration callback for visual debugging if provided
                        if iteration_callback is not None:
                            # Create a platepar copy with current iteration's parameters, decoded as the NN cost
                            #   function does
                            pp_iter = copy.deepcopy(self)
                            _setPointingAndScale(pp_iter, res.x, pointing)
                            if len(res.x) > 4:
                                pp_iter.x_poly_fwd = np.array(res.x[4:])
                                pp_iter.x_poly_rev = np.array(res.x[4:])
                            pp_iter.updateRefAltAz()
                            iteration_callback(iteration + 1, pp_iter, iteration_outliers, rmsd_arcmin)

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
                        self.aberration = aberration_prev
                        return None

                    # Apply best RANSAC params to platepar
                    _setPointingAndScale(self, best_res.x, best_pointing)
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

                    # Nothing can be matched if no catalog star is inside the image with the fitted parameters
                    if len(catalog_stars_fov) == 0:
                        print("    -> No catalog stars in the field of the RANSAC fit, skipping final fit")
                        return None

                    # Vectorized NN matching: compute NxM separation matrix
                    ra_det_rad = np.radians(ra_det)[:, np.newaxis]  # (N, 1)
                    dec_det_rad = np.radians(dec_det)[:, np.newaxis]  # (N, 1)
                    ra_cat_rad = np.radians(ra_catalog)[np.newaxis, :]  # (1, M)
                    dec_cat_rad = np.radians(dec_catalog)[np.newaxis, :]  # (1, M)
                    sep_matrix = angularSeparation(ra_det_rad, dec_det_rad, ra_cat_rad, dec_cat_rad)
                    nearest_indices = np.argmin(sep_matrix, axis=1)  # (N,)
                    nearest_sep = np.min(sep_matrix, axis=1)         # (N,)

                    # Enforce one-to-one matching. argmin is per-detection and non-injective, so two
                    #   detected stars can claim the same catalog star (a duplicate) - which then
                    #   shows up as a gross residual and inflates the RMSD. Resolve conflicts by
                    #   keeping, for each catalog star, only the closest detection; drop the rest.
                    keep_mask = np.zeros(len(nearest_indices), dtype=bool)
                    seen_catalog = set()
                    for det_i in np.argsort(nearest_sep):
                        cat_i = int(nearest_indices[det_i])
                        if cat_i in seen_catalog:
                            continue
                        seen_catalog.add(cat_i)
                        keep_mask[det_i] = True

                    n_dupes = int(np.sum(~keep_mask))
                    if n_dupes > 0:
                        print("    Dropped {} duplicate matches (multiple detections -> one catalog "
                              "star)".format(n_dupes))

                    # A star without a finite sky position has no real nearest catalog star
                    keep_mask &= np.isfinite(nearest_sep)

                    img_stars_clean = img_stars_clean[keep_mask]
                    matched_catalog = catalog_stars_fov[nearest_indices[keep_mask]]

                    # Restore original distortion type for final matched-pair fit
                    self.setDistortionType(original_dist_type, reset_params=False)

                    # Final fit: call fitAstrometry recursively with matched pairs
                    print("    Final fit on {} matched stars with {} (recursive call)...".format(
                        len(img_stars_clean), original_dist_type))
                    self.fitAstrometry(jd, img_stars_clean, matched_catalog,
                                       first_platepar_fit=False, use_nn_cost=False, fixed_scale=fixed_scale,
                                       fit_by=fit_by)

                    # Return the actual matched pairs from RANSAC
                    return (img_stars_clean, matched_catalog)
                else:
                    # Iterate forward-reverse fits until convergence
                    # Forward fits pointing + distortion, reverse fits distortion only.
                    # Without iteration, one pass leaves forward/reverse inconsistent.
                    max_fwd_rev_iter = 3
                    convergence_threshold = 1e-6

                    for fwd_rev_iter in range(max_fwd_rev_iter):
                        # Store previous params to check convergence
                        prev_x_poly_fwd = self.x_poly_fwd.copy()
                        prev_x_poly_rev = self.x_poly_rev.copy()

                        ### FORWARD MAPPING FIT ###
                        # Prototype: Levenberg-Marquardt least-squares on the vector residual.
                        # Converges in far fewer evals than Nelder-Mead and lands on the same
                        # (or lower-residual) least-squares minimum.
                        # The pointing is fitted as offsets about the pointing of the platepar, which every
                        #   iteration starts from
                        pointing = PointingOffsets(self)
                        res = _lstsqFit(
                            _calcSkyResidualsAstroAndDistortionRadialVect,
                            p0,
                            (self, jd, catalog_stars, img_stars, pointing),
                        )
                        fwd_status = "converged" if res.success else "stopped"

                        # Update fitted astrometric parameters
                        # Hold the scale when fixed_scale is set. The patched residual makes the fit
                        # objective independent of the F_scale parameter, so LM leaves res.x[3] at its
                        # seed; read the held value explicitly rather than relying on that.
                        _setPointingAndScale(self, res.x, pointing)

                        self.updateRefAltAz()

                        # Extract distortion parameters
                        self.x_poly_fwd = np.array(res.x[4:])

                        # Update p0 for next iteration with current fitted values (no offset from the fitted
                        #   pointing)
                        p0 = pointing_start + [abs(self.F_scale)]
                        p0 += self.x_poly_fwd.tolist()

                        ### REVERSE MAPPING FIT ###
                        # Initialize reverse fit from forward coefficients as starting point
                        # This provides a much better initial guess than zeros, especially for
                        # cameras with extreme distortion center offset where the optimizer
                        # can diverge when starting from zeros
                        rev_init = self.x_poly_fwd.copy()
                        res_rev = self._withReverseProjection(
                            _lstsqFit,
                            _calcImageResidualsDistortionVect,
                            rev_init,
                            (self, jd, catalog_stars, img_stars, 'radial'),
                        )
                        rev_status = "converged" if res_rev.success else "stopped"

                        self.x_poly_rev = res_rev.x

                        # Check convergence
                        fwd_change = np.max(np.abs(self.x_poly_fwd - prev_x_poly_fwd))
                        rev_change = np.max(np.abs(self.x_poly_rev - prev_x_poly_rev))

                        print("    Fwd-rev iteration {}: fwd_change={:.2e} ({}), rev_change={:.2e} ({})".format(
                            fwd_rev_iter + 1, fwd_change, fwd_status, rev_change, rev_status))

                        if fwd_change < convergence_threshold and rev_change < convergence_threshold:
                            if fwd_rev_iter > 0:
                                print("    Converged after {} iterations".format(fwd_rev_iter + 1))
                            break

                    # Final refinement: fit pointing + reverse distortion jointly on the IMAGE
                    # (pixel) residual -- the metric reported to users. The fwd-rev loop fits the
                    # forward mapping on SKY angular separation, whose optimum can leave the
                    # pointing slightly off for the pixel residual (forward/reverse mappings are
                    # not exact inverses), which showed up as a worse RMSD with least_squares.
                    # This recovers the old Nelder-Mead pixel accuracy while keeping the speedup.
                    # With the exact reverse the image residual goes through the forward distortion, and the
                    #   reverse polynomial plays no part in it: only the pointing is refined
                    exact_reverse = getattr(self, "exact_reverse", False)
                    rev_fixed = self.x_poly_rev.tolist()

                    def _calcImageResidualsAstroVect(params, *args):
                        res = _calcImageResidualsAstroAndDistortionRadialVect(list(params) + rev_fixed, *args)

                        # A catalog star with no image (beyond the reach of the exact inverse) counts as off by
                        #   the image diagonal, as in the pointing fit, so the residuals stay finite for least
                        #   squares
                        res[~np.isfinite(res)] = np.hypot(self.X_res, self.Y_res)/np.sqrt(2)

                        return res

                    pointing = PointingOffsets(self)
                    p_final = pointing_start + [abs(self.F_scale)] \
                        + ([] if exact_reverse else self.x_poly_rev.tolist())
                    res_final = _lstsqFit(
                        _calcImageResidualsAstroVect if exact_reverse
                            else _calcImageResidualsAstroAndDistortionRadialVect,
                        p_final,
                        (self, jd, catalog_stars, img_stars, pointing),
                    )
                    xf = np.concatenate([res_final.x, rev_fixed]) if exact_reverse else res_final.x
                    # Hold the scale when fixed_scale is set (see fwd-rev loop note above).
                    _setPointingAndScale(self, xf, pointing)
                    self.x_poly_rev = np.array(xf[4:])
                    self.updateRefAltAz()

                    # The joint fit above moved the pointing, which leaves the forward distortion
                    # (fit to the previous pointing) stale -- the forward and reverse mappings
                    # would disagree (broken image<->sky round-trip). Re-fit the forward
                    # distortion to the refined pointing so the two mappings stay consistent.
                    res_fwd = _lstsqFit(
                        _calcSkyResidualsDistortionVect,
                        self.x_poly_fwd,
                        (self, jd, catalog_stars, img_stars, 'radial'),
                    )
                    self.x_poly_fwd = res_fwd.x

                    # With the exact reverse, refit the reverse polynomial to the final pointing, as it is fitted
                    #   without the option, for readers that do not know it
                    if exact_reverse:
                        res_rev = self._withReverseProjection(
                            _lstsqFit,
                            _calcImageResidualsDistortionVect,
                            self.x_poly_fwd.copy(),
                            (self, jd, catalog_stars, img_stars, 'radial'),
                        )
                        self.x_poly_rev = res_rev.x

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

        # Record what was fitted (one entry for the distortion and the pointing of a full fit)
        if pointing_fitted or distortion_fitted:
            self.recordFit(pointing=pointing_fitted, distortion=distortion_fitted, fit_by=fit_by)

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
                  'k1': 0.0, 'k2': 0.0, 'k3': 0.0, 'k4': 0.0, 'p1': 0.0, 'p2': 0.0, 'p3': 0.0, 'p4': 0.0}

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
        # Number depends on distortion type: radial3-odd has k1, radial5-odd has k1,k2, etc. With the decentering
        #   correction only the radial coefficients of the type are read, the decentering coefficients follow them
        n_radial = self.radialTermCount() if self.decentering_corr else 4
        for i, key in enumerate(('k1', 'k2', 'k3', 'k4')[:n_radial]):
            if idx + i < len(x_poly):
                result[key] = x_poly[idx + i]

        # Decentering coefficients (p1, p2, p3, p4), after the radial ones
        if self.decentering_corr:
            for i, key in enumerate(('p1', 'p2', 'p3', 'p4')):
                if idx + n_radial + i < len(x_poly):
                    result[key] = x_poly[idx + n_radial + i]

        return result

    def radialTermCount(self, dist_type=None):
        """Number of the radial distortion coefficients (k1, k2, ...) of a radial distortion type.

        Arguments:
            dist_type: [str] Radial distortion type. None by default (the current distortion type).

        Returns:
            int: Number of the radial coefficients: the full length of the type without the distortion centre
                (2), the aspect (1) and the asymmetry (2) parameters.
        """
        # The current distortion type by default
        if dist_type is None:
            dist_type = self.distortion_type

        # The full length of the type, without the centre, the aspect and the asymmetry parameters
        return self.distortion_type_poly_length[self.distortion_type_list.index(dist_type)] - 5

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

        # Radial distortion coefficients - number depends on target type (radial3-odd has k1, radial5-odd and
        #   radial3-all have k1, k2, etc.)
        for key in ('k1', 'k2', 'k3', 'k4')[:self.radialTermCount(target_dist_type)]:
            result.append(coeffs_dict.get(key, 0.0))

        # The decentering coefficients follow the radial ones
        if self.decentering_corr:
            for key in ('p1', 'p2', 'p3', 'p4'):
                result.append(coeffs_dict.get(key, 0.0))

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

    def __setstate__(self, state):
        """ Restore a pickled platepar (e.g. from a SkyFit2 state file). A platepar pickled before an option existed
            gets its default, and the reference hour angle is recomputed from the reference time, as when a
            platepar file is loaded (see loadFromDict).

        Arguments:
            state: [dict] The attributes of the pickled platepar.
        """

        self.__dict__.update(state)

        # The reference hour angle is a function of the reference time only, so it is recomputed as in
        #   loadFromDict: a platepar pickled with the mean sidereal time keeps pointing where it was fitted, and
        #   gives the same coordinates as the same calibration loaded from its platepar file
        if 'JD' in self.__dict__:
            self.Ho = JD2HourAngle(self.JD)

        # The decentering correction (platepars pickled before it existed have none)
        if 'decentering_corr' not in self.__dict__:
            self.decentering_corr = False

        # The exact reverse mapping (platepars pickled before it existed have none)
        if 'exact_reverse' not in self.__dict__:
            self.exact_reverse = False

        # The aberration flag: a platepar pickled before it existed was fitted without the aberration modelled
        #   and has absorbed it, as when a platepar file without the flag is loaded
        if 'aberration' not in self.__dict__:
            self.aberration = False

        # The fit record: a platepar pickled before it existed has none, its fits are not known
        self.fit_record = normalFitRecord(self.__dict__.get('fit_record'))

    def loadFromDict(self, platepar_dict, use_flat=None):
        """Load the platepar from a dictionary."""

        # Parse JSON into an object with attributes corresponding to dict keys
        self.__dict__ = platepar_dict

        # The height above the ellipsoid follows from the station coordinates (see height_wgs84), a value in
        #   the file is not taken
        self.__dict__.pop('height_wgs84', None)

        # The reference hour angle is a function of the reference time only, so it is recomputed rather
        #   than taken from the file: this way a platepar written with an older definition of the sidereal
        #   time (e.g. without the equation of the equinoxes) keeps pointing where it was fitted
        if 'JD' in self.__dict__:
            self.Ho = JD2HourAngle(self.JD)

        # Add the version if it was not in the platepar (v1 platepars didn't have a version)
        if not 'version' in self.__dict__:
            self.version = 1

        # If the refraction was not used for the fit, assume it is disabled
        if not 'refraction' in self.__dict__:
            self.refraction = False

        # A platepar without the flag was fitted before the aberration was modelled, so it has absorbed it
        if not 'aberration' in self.__dict__:
            self.aberration = False

        # The fit record: a platepar from before it existed has none, its fits are not known and it was
        #   fitted with the astrometric model 1 (see astrometryModel)
        self.fit_record = normalFitRecord(self.__dict__.get('fit_record'))

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

        # Add the decentering correction (platepars from before it existed have none)
        if not 'decentering_corr' in self.__dict__:
            self.decentering_corr = False

        # Add the exact reverse mapping (platepars from before it existed have none)
        if not 'exact_reverse' in self.__dict__:
            self.exact_reverse = False

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
                self.Ho = JD2HourAngle(self.JD)

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

                # The CMN format has no aberration flag: such a plate was fitted to the catalog directions as
                #   they are and has absorbed the aberration
                self.aberration = False

                # Nor a fit record: its fits are not known
                self.fit_record = emptyFitRecord()

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

        # The height above the ellipsoid is derived from the station coordinates, an assigned value is not
        #   written
        self2.__dict__.pop('height_wgs84', None)

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
            fov: [tuple] Tuple of (fov_h, fov_v) in degrees. If given, used directly to avoid recomputation.
                If None, FOV is computed from the current calibration.
            ret_written: [bool] If True, the JSON string of the platepar instead of writing it to disk.
        Return:
            fmt: [str] Platepar format.
        """

        # Update FOV — use provided value to avoid recomputation, otherwise compute from calibration
        if fov is not None:
            self.fov_h, self.fov_v = fov
        else:
            try:
                # Local import to avoid circular dependency (ApplyAstrometry imports Platepar)
                from RMS.Astrometry.ApplyAstrometry import computeFOVSize
                self.fov_h, self.fov_v = computeFOVSize(self)
            except Exception:
                pass  # Keep existing values if computation fails (e.g. uncalibrated platepar)

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

    def computeRefAltAz(self):
        """Apparent azimuth and altitude (deg) of the reference pointing, from RA_d/dec_d. No side effects.

        RA_d/dec_d are the true (unrefracted) equatorial coordinates of the reference point in the epoch of
        date of the platepar JD: pointingCorrection shifts them by sidereal time and precesses them to J2000
        itself, so that is the frame the star fit defines them in. They are therefore converted without any
        precession; the J2000 converters would put the centre ~20 arcmin (the precession since 2000) away
        from where the camera points.

        Return:
            (az_centre, alt_centre): [tuple of floats] Apparent azimuth (+E of N) and altitude (deg).
        """

        return trueOfDateRaDec2ApparentAltAz(
            self.RA_d, self.dec_d, self.JD, self.lat, self.lon, refraction=self.refraction,
            refraction_scale=refractionScale(self.elev)
        )

    def updateRefAltAz(self):
        """Update the reference apparent azimuth and altitude from the reference RA and Dec (see
        computeRefAltAz for the epoch convention), and the rotation wrt horizon."""

        self.az_centre, self.alt_centre = self.computeRefAltAz()

        # Update the rotation wrt horizon
        self.rotation_from_horiz = RMS.Astrometry.ApplyAstrometry.rotationWrtHorizon(self)

    def updateRefRADec(self, skip_rot_update=False, preserve_rotation=False):
        """Update the reference RA and Dec (true of date) from Alt/Az (apparent in epoch of date)."""

        if (not skip_rot_update) and (not preserve_rotation):

            # Save the current rotation w.r.t horizon value
            self.rotation_from_horiz = RMS.Astrometry.ApplyAstrometry.rotationWrtHorizon(self)

        # Convert the reference apparent Alt/Az to a true-of-date RA/Dec, the frame of RA_d/dec_d
        self.RA_d, self.dec_d = apparentAltAz2TrueOfDateRaDec(
            self.az_centre, self.alt_centre, self.JD, self.lat, self.lon, refraction=self.refraction,
            refraction_scale=refractionScale(self.elev)
        )

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

    @property
    def height_wgs84(self):
        """ Height of the station above the WGS84 ellipsoid (m). Derived from the height above sea level (elev)
            through the EGM96 geoid, so that it follows any change of the station coordinates. A value
            assigned to it takes precedence; it is not written into the platepar file.
        """

        # An assigned value is kept in the instance dictionary under the name of the property
        if self.__dict__.get('height_wgs84') is not None:
            return self.__dict__['height_wgs84']

        # Otherwise the height above sea level is taken to the ellipsoid
        return mslToWGS84Height(np.radians(self.lat), np.radians(self.lon), self.elev)

    @height_wgs84.setter
    def height_wgs84(self, value):
        """ Assign the height of the station above the WGS84 ellipsoid (m), None to derive it again. """

        self.__dict__['height_wgs84'] = value

    def __repr__(self):

        # Compute alt/az pointing
        azim, elev = self.computeRefAltAz()

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
        out_str += "Reference pointing - equatorial (true of date):\n"
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

            # The radial coefficients, followed by the decentering ones if they are used
            if self.decentering_corr:
                out_str += "      k1, ..., p1, p2, p3, p4 (decentering)\n"
            else:
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
