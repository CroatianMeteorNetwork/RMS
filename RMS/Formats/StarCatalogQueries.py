# RPi Meteor Station
# Copyright (C) 2026 David Rollinson
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.



from __future__ import print_function, absolute_import

import numpy as np
import cv2
import scipy
import FFfits
import os
import tarfile

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime, UTC, timezone, timedelta
from copy import deepcopy
from scipy.spatial import cKDTree


from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, xyToRaDecPP
from RMS.Astrometry.AstrometryNet import astrometryNetSolveLocal
from RMS.ConfigReader import loadConfigFromDirectory
from RMS.Formats.FFfile import getMiddleTimeFF
from RMS.Formats.Platepar import Platepar
from RMS.Formats.StarCatalog import readStarCatalog
from RMS.Astrometry.Conversions import date2JD, datetime2JD, raDec2AltAz, vector2RaDec
from RMS.Math import angularSeparationDeg, vectShort

J2000 = datetime(2000, 1, 1, 12, 0, 0).replace(tzinfo=timezone.utc)


def getPlateparSkyParameters(pp, jd):

    """ Given a platepar and a julian date calculate the sky coordinates
    of the centre of the image and the maximum angular distance between corners.

    Arguments:
        pp: [platepar] RMS platepar object.
        jd: [float] Julian date.

    Return:
        ra_c: [array] Right ascension (degrees) of centre.
        dec_c: [array] Declination (degrees) of centre.
        max_separation: [float] Maximum angular separation (degrees) between corners.
        ra_array: [array] Array of corner right ascension coordinates (degrees).
        dec_array: [array] Array of corner declination coordinates (degrees).

    """

    # Get the optical centre of the platepar in sky coordinates
    x_c, y_c = pp.getDistortionCentre()
    _, ra_c, dec_c, _ = xyToRaDecPP([jd], [x_c], [y_c], [1], pp, jd_time=True)

    # Find the corners of the platepar in image coordinates
    cx_array, cy_array = np.array([0, pp.X_res, pp.X_res, 0]), np.array([0, 0, pp.Y_res, pp.Y_res])

    # Convert the corners to sky coordinates (represented as an array)
    jd_array, mag_array = np.full_like(cy_array, jd, dtype=float), np.full_like(cy_array, 1, dtype=float)
    _, ra_array, dec_array, _ = xyToRaDecPP(jd_array, cx_array, cy_array, mag_array, pp, jd_time=True)

    # Compute the all the angular separations, and take the maximum
    max_separation = np.max(angularSeparationDeg(ra_c, dec_c, ra_array, dec_array))

    return ra_c, dec_c, max_separation, ra_array, dec_array

def vecsInsideConvexPolygon(vecs, normals):

    """Determine which vectors lie inside a convex spherical polygon.

    The polygon is represented by a set of inward-pointing unit normals,
    one for each edge plane. A vector is considered inside the polygon if
    it lies on the inward side of every edge plane.

    Arguments:
        vecs: [ndarray] Array of Cartesian unit vectors with shape (N, 3).
        normals: [ndarray] Array of inward-pointing unit normals, one for each edge.

    Return:
        polygon_mask: [ndarray] Boolean array with one entry for each input
            vector. True indicates that the vector lies inside the polygon.
    """

    polygon_mask = np.ones(len(vecs), dtype=bool)

    for n in normals:
        polygon_mask &= ((vecs @ n) >= 0)

    return polygon_mask


def repointPlatepar(pp, az_new, el_new):

    """Given a platepar, new azimuth and new elevation, repoint the field of
    view to the new azimuth and elevation preserving rotation with respect to
    the local horizon.

    Arguments:
        pp: [Platepar] RMS platepar object.
        az_new: [Float] Azimuth angle (degrees).
        el_new: [Float] Elevation angle (degrees)
    """

    # Update the azimuth and elevation
    pp.az_centre, pp.alt_centre = az_new, el_new

    # Update platepar internal sky and horizontal coordinate references
    pp.updateRefRADec(skip_rot_update=False, preserve_rotation=True)
    pp.updateRefAltAz()

    return pp




def computePolygonNormals(corner_vecs, inside_vec):

    """ Given vectors to the corners of a convex spherical polygon and a
    vector to a point within the polygon (typically, but not essentially
    the centre), compute inward-pointing unit normals to the great-circle planes
    defining the polygon edges.

    Arguments:
        corner_vecs: [array] Array of vectors to the polygon corners.
        inside_vec: [vector] Vector to a point within the polygon.

    Return:
        normals: [ndarray] Array of inward-pointing unit vectors.

    """

    # Initialise the list of normals
    normals = []

    # Compute the number of sides
    n_sides = len(corner_vecs)

    # Iterate over the polygon corners
    for i in range(n_sides):

        # Extract adjacent polygon corners
        p1, p2 = corner_vecs[i], corner_vecs[(i + 1) % n_sides]

        # Compute normal to each edge plane and normalize
        n = np.cross(p1, p2)
        n /= np.linalg.norm(n)

        # If required, invert the vector
        if np.dot(n, inside_vec) < 0:
            n *= -1

        normals.append(n)

    return np.array(normals)


def radecToUnitVectors(ra_deg, dec_deg):

    """ Given one or more RA/Dec coordinates, return the
    corresponding Cartesian unit vectors.

    Arguments:
        ra_deg: [array] Right ascension (degrees)
        dec_deg: [array] Declination (degrees)

    Return:
        [ndarray] Cartesian unit vectors with shape (N, 3).

    """

    ra = np.radians(np.atleast_1d(ra_deg))
    dec = np.radians(np.atleast_1d(dec_deg))

    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)

    return np.column_stack((x, y, z))


class Catalog:
    """
    Load a star catalogue, build a spherical KD-tree, and expose query methods.
    """

    def __init__(self, config, catalogue_time=None, ra_col=0, dec_col=1, mag_col=2, lim_mag=None):

        """Initialise a star catalog, expose Cartesian unit vectors, and construct a spherical KD-tree

        Arguments:
            config[config]: RMS config instance.

        Keyword Arguments:
            catalogue_time: Time point for the catalogue generation if none build for now.
            ra_col: Optional, default 0 - array column with ra data in degrees.
            dec_col: Optional, default 1 - array column with dec data in degrees.
            mag_col: Optional, default 2 - array column of magnitude data.
            name_col: Optional, default 3 - array column of star names.

        """

        self.ra_col, self.dec_col, self.mag_col = ra_col, dec_col, mag_col

        if catalogue_time is None:
            catalogue_time = datetime.now(timezone.utc)

        years_from_J2000 = (catalogue_time - J2000).total_seconds() / (365.25 * 24 * 3600)

        # Read in the star catalog
        star_catalog_status = readStarCatalog(
            config.star_catalog_path,
            config.star_catalog_file,
            lim_mag=lim_mag,
            years_from_J2000=years_from_J2000,
            mag_band_ratios=config.star_catalog_band_ratios,
            additional_fields=['preferred_name', 'common_name', 'bayer_name', 'B', 'V', 'R'])

        if not star_catalog_status:
            print("Could not load star catalogue")

        catalog_stars, _, config.star_catalog_band_ratios, extras = star_catalog_status

        # Produce a mask for stars in expected coordinate ranges and finite
        mask_finite = np.isfinite(catalog_stars[:, 0:3]).all(axis=1)
        maskRange = (
                # Right ascension
                (catalog_stars[:, 0] >= 0.0) & (catalog_stars[:, 0] < 360.0) &
                # Declination
                (catalog_stars[:, 1] >= -90.0) & (catalog_stars[:, 1] <= 90.0)
        )

        mask = mask_finite & maskRange

        # Apply the mask
        self.cat = catalog_stars[mask]
        self.names = extras['preferred_name'][mask]
        self.b, self.v, self.r = extras['B'][mask], extras['V'][mask], extras['R'][mask]

        # Expose the unit vectors
        self.unit_vecs = radecToUnitVectors(self.cat[:, ra_col],self.cat[:, dec_col])

        # Build the tree
        self.tree = cKDTree(self.unit_vecs)

    def queryRaDec(self, ra_deg, dec_deg, radius_deg=0.1, n_brightest=1):

        """Search for stars within the radii of given arrays of coordinates.

        Return a list with one entry for each set of coordinates. Each entry contains a list of stars
        within the radius, ordered by increasing magnitude (brightest first). The return format is
        always the same, even for a single set of coordinates return a single star.

        Arguments:
            ra_deg: [array] Right ascension (degrees).
            dec_deg: [array] Declination (degrees).

        Keyword Arguments:
            radius_deg: [float] Optional default 0.1, radius (degrees).
            n_brightest: [int] Optional default 1, n_brightest (degrees).

        Return: [list]
            One entry per query coordinate.
            Each entry is a list of:

            [name, ra, dec, mag, theta, B, V, R]

            Where theta is the angular separation between that star and the search coordinates.

            If no star is found to match the criteria, an empty list is returned for that query.



        """

        # Normalize inputs to arrays of radians
        ra_deg, dec_deg = np.atleast_1d(ra_deg), np.atleast_1d(dec_deg)

        # Build query vectors array
        query_vectors = radecToUnitVectors(ra_deg, dec_deg)

        # Euclidean chord distance for spherical radius
        ecd = 2 * np.sin(np.radians(radius_deg) / 2)

        # Initialise list for results
        results = []

        # Iterate over the array of queries
        for qvec, ra0, dec0 in zip(query_vectors, ra_deg, dec_deg):

            # Perform a single query and return array of catalogue indices
            idx = np.array(self.tree.query_ball_point(qvec, ecd), dtype=int)

            # If nothing returned by the query append an empty list
            if len(idx) == 0:
                results.append([])
                continue

            # Extract the query results
            query_result_mags = self.cat[idx, self.mag_col]

            # Sort the query result indices by magnitude increasing (brightest stars first)
            brightness_order = np.argsort(query_result_mags)

            # Truncate to the number of brightest stars and retrun catalog indices
            selected_catalog_indices = idx[brightness_order[:n_brightest]]

            # Extract the data from the catalog
            names = [x.decode("utf-8", errors="replace").strip() for x in self.names[selected_catalog_indices]]
            ras = self.cat[selected_catalog_indices, self.ra_col].astype(float)
            decs = self.cat[selected_catalog_indices, self.dec_col].astype(float)
            mags = self.cat[selected_catalog_indices, self.mag_col].astype(float)
            b = self.b[selected_catalog_indices].astype(float)
            v = self.v[selected_catalog_indices].astype(float)
            r = self.r[selected_catalog_indices].astype(float)

            # Compute the angular separation as a query return, not used for star selection
            thetas = angularSeparationDeg(ra0, dec0, ras, decs).astype(float)

            # Build a list of stars to return
            matched_stars = []

            for i in range(len(names)):
                star = [names[i], ras[i], decs[i], mags[i], thetas[i], b[i], v[i], r[i]]
                matched_stars.append(star)

            # Append the stars returned by this query to the list of the results of all queries
            results.append(matched_stars)

        return results

    def queryPlateparSimple(self, pp, jd, n_brightest=1000, az=None, el=None):

        """Query a platepar for the brightest stars visible within its field of
        view at a specified Julian date.


        Stars are initially selected using a circular sky search centred on the
        platepar optical axis. They are then projected into image coordinates
        and filtered to those whose image coordinates lie within the image
        bounds.

        The platepar may optionally be repointed before the query. Any repointing
        operation is applied to a copy of the platepar, leaving the original
        object unchanged.


        Arguments:
            pp: [Platepar] RMS platepar object.
            jd: [float] Julian date.

        Keyword Arguments:
            n_brightest: [int] Optional, default 1000. Number of brightest stars to return.
            az: [float] Optional azimuth (degrees) used to repoint the platepar.
            el: [float] Optional elevation (degrees) used to repoint the platepar.

        Return:
            p: [Platepar] Platepar used for the query.
            ra_array: [array] Right ascension coordinates (degrees).
            dec_array: [array] Declination coordinates (degrees).
            x_array: [array] X image coordinates (pixels).
            y_array: [array] Y image coordinates (pixels).
            mag_array: [array] Stellar magnitudes.
            names_array: [array] Star names.
        """

        # Create a copy, so that the original object is unchanged
        p = deepcopy(pp)

        if az is not None and  el is not None:
            # Repoint the platepar
            p = repointPlatepar(p, az, el)

        # Get the platepar sky parameters at jd
        ra_c, dec_c, max_separation, _, _ = getPlateparSkyParameters(p, jd)

        # Query for stars and map to arrays
        stars_list = self.queryRaDec(ra_c, dec_c, max_separation, n_brightest=n_brightest * 4)[0]

        # If nothing returned, then exit the function returning empty arrays
        if not stars_list:
            return p, np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        names_array, ra_array, dec_array, mag_array, theta_array, b_array, v_array, r_array = map(np.array, zip(*stars_list))

        # Compute image coordinates.
        x_array, y_array = raDecToXYPP(ra_array, dec_array, jd, p)

        # Create a mask to cut any stars which are outside the image
        mask = ((x_array >= 0) & (x_array < p.X_res) & (y_array >= 0) & (y_array < p.Y_res))

        # Return only those within the image and cut to the number of brightest stars
        ra_array, dec_array = ra_array[mask][:n_brightest], dec_array[mask][:n_brightest]
        x_array, y_array = x_array[mask][:n_brightest], y_array[mask][:n_brightest]
        mag_array, names_array = mag_array[mask][:n_brightest], names_array[mask][:n_brightest]

        return p, ra_array, dec_array, x_array, y_array, mag_array, names_array




    def queryPlatepar(self, pp, jd, n_brightest=1000, az=None, el=None):

        """Query a platepar for the brightest stars visible within its field of
        view at a specified Julian date.

        Stars are initially selected using a circular sky search centred on the
        platepar optical axis. The platepar field of view is then represented
        as a convex spherical polygon and candidate stars outside this polygon
        are rejected. The remaining stars are projected into image coordinates
        and filtered to those whose image coordinates lie within the image
        bounds.

        The platepar may optionally be repointed before the query. Any repointing
        operation is applied to a copy of the platepar, leaving the original
        object unchanged.

        Arguments:
            pp: [Platepar] RMS platepar object.
            jd: [float] Julian date.

        Keyword Arguments:
            n_brightest: [int] Optional, default 1000. Number of brightest stars
                to return.
            az: [float] Optional azimuth (degrees) used to repoint the platepar.
            el: [float] Optional elevation (degrees) used to repoint the platepar.

        Return:
            p: [Platepar] Platepar used for the query.
            ra_array: [array] Right ascension coordinates (degrees).
            dec_array: [array] Declination coordinates (degrees).
            x_array: [array] X image coordinates (pixels).
            y_array: [array] Y image coordinates (pixels).
            mag_array: [array] Stellar magnitudes.
            names_array: [array] Star names.

        Notes:
            Candidate stars are first selected using a circular sky query whose
            radius is determined from the platepar footprint. A convex spherical
            polygon test is then used to reject stars outside the true field of
            view before projection into image coordinates.
        """

        p = deepcopy(pp)

        if az is not None and el is not None:
            p = repointPlatepar(p, az, el)


        # Get the platepar sky parameters
        ra_c, dec_c, max_separation, corner_ra_array, corner_dec_array = getPlateparSkyParameters(p, jd)

        # Compute the Euclidean vectors in sky coordinates of the platepar corners and centre
        corner_vecs = radecToUnitVectors(corner_ra_array, corner_dec_array)
        centre_vec = radecToUnitVectors(ra_c, dec_c)[0]

        # Compute inward-pointing normals to the polygon edge planes
        normals = computePolygonNormals(corner_vecs, centre_vec)

        # Query for stars
        stars_list = self.queryRaDec(ra_c, dec_c, max_separation, n_brightest=n_brightest * 4)[0]

        # If nothing returned, then exit the function returning empty arrays
        if not stars_list:
            return p, np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        # Extract into arrays and convert to vectors
        names_array, ra_array, dec_array, mag_array, theta_array, b_array, v_array, r_array = map(np.array, zip(*stars_list))
        star_vecs = radecToUnitVectors(ra_array, dec_array)

        # Create the polygon mask
        polygon_mask = vecsInsideConvexPolygon(star_vecs, normals)

        # Extract into arrays and mask
        names_array = names_array[polygon_mask]
        ra_array, dec_array, mag_array = ra_array[polygon_mask], dec_array[polygon_mask], mag_array[polygon_mask]
        theta_array = theta_array[polygon_mask]
        b_array, v_array, r_array = b_array[polygon_mask], v_array[polygon_mask], r_array[polygon_mask]

        # Compute image coordinates for remaining selection
        x_array, y_array = raDecToXYPP(ra_array, dec_array, jd, p)

        # Identify stars which are outside the image
        image_mask = ((x_array >= 0) & (x_array < p.X_res) & (y_array >= 0) & (y_array < p.Y_res))

        # Apply mask
        names_array = names_array[image_mask]
        ra_array, dec_array, mag_array = ra_array[image_mask], dec_array[image_mask], mag_array[image_mask]
        x_array, y_array = x_array[image_mask], y_array[image_mask]
        theta_array, b_array, v_array, r_array = theta_array[image_mask], b_array[image_mask], v_array[image_mask], r_array[image_mask]

        return p, ra_array, dec_array, x_array, y_array, mag_array, names_array




### TEST CODE START ###

def loadConstellationBoundaries(config, path=None):
    """Load IAU constellation boundary vertices.

    The boundary file is expected to contain rows consisting of right
    ascension (hours), declination (degrees), constellation abbreviation,
    and an edge flag. Boundary vertices are grouped by constellation.

    Arguments:
        config: [Config] RMS configuration object.

    Keyword Arguments:
        path: [str] Optional path to the constellation boundary file.
            If None, the default RMS boundary file is used.

    Return:
        constellation_boundaries: [dict] Dictionary keyed by constellation
            abbreviation. Each entry contains:

            {
                "ra_deg": [...],
                "dec_deg": [...],
                "flags": [...]
            }


    """

    constellation_boundaries = {}

    if path is None:
        path = os.path.join(config.rms_root_dir, "share/constellation_boundaries_J2000.dat")


    with open(path) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            ra_hours, dec_deg, name, flag = line.split()

            if name not in constellation_boundaries:
                constellation_boundaries[name] = {
                    "ra_deg": [],
                    "dec_deg": [],
                    "flags": []
                }

            constellation_boundaries[name]["ra_deg"].append(
                float(ra_hours) * 15.0
            )

            constellation_boundaries[name]["dec_deg"].append(
                float(dec_deg)
            )

            constellation_boundaries[name]["flags"].append(
                flag
            )

    return constellation_boundaries

def computeConstellationCentres(constellation_boundaries):
    """Compute representative RA/Dec coordinates for constellation labels.

    The centre of each constellation is computed by converting all
    boundary vertices to Cartesian unit vectors, averaging those
    vectors, then converting the resulting unit vector back to sky
    coordinates.

    Arguments:
        constellation_boundaries: [dict] Constellation boundary
            dictionary returned by loadConstellationBoundaries().

    Return:
        constellation_centres: [dict] Dictionary keyed by constellation
            abbreviation. Each entry contains:

            {
                "ra_deg": float,
                "dec_deg": float
            }

    """

    constellation_centres = {}

    for name, boundary in constellation_boundaries.items():

        vecs = radecToUnitVectors(boundary["ra_deg"], boundary["dec_deg"])

        centre_vec = vecs.mean(axis=0)
        centre_vec /= np.linalg.norm(centre_vec)

        ra_deg, dec_deg = vector2RaDec(centre_vec)

        constellation_centres[name] = {"ra_deg": ra_deg, "dec_deg": dec_deg}

    return constellation_centres

def extractTarBz2(path):
    """Extract a .tar.bz2 archive into a directory alongside the archive.

    For example:

        /tmp/test_data.tar.bz2

    is extracted into:

        /tmp/test_data/

    Arguments:
        path: [str] Path to the .tar.bz2 archive.

    Return:
        extract_dir: [str] Path to the extracted directory.
    """

    path = os.path.abspath(path)

    filename = os.path.basename(path)

    if not filename.endswith(".tar.bz2"):
        raise ValueError(f"Not a .tar.bz2 file: {path}")

    extract_dir = os.path.join(
        os.path.dirname(path),
        filename[:-8]  # strip ".tar.bz2"
    )

    os.makedirs(extract_dir, exist_ok=True)

    with tarfile.open(path, "r:bz2") as tar:
        tar.extractall(extract_dir)

    return extract_dir


def renderStarsSimple(x_array, y_array, shape):
    """
    Render stars as single bright pixels.

    x_array, y_array : pixel coordinates
    shape            : (height, width)
    """

    H, W = shape

    img = np.zeros((H, W), dtype=np.uint8)

    x = np.round(x_array).astype(int)
    y = np.round(y_array).astype(int)

    # Keep only stars inside image bounds
    mask = (
        (x >= 0) & (x < W) &
        (y >= 0) & (y < H)
    )

    img[y[mask], x[mask]] = 255

    return img





def renderStarsGaussian(
        x_array,
        y_array,
        mag_array,
        names_array,
        shape,
        sigma=1.5,
        min_brightness=0,
        max_brightness=400,
        sensor_saturation=255,
        add_noise=True,
        noise_sigma=1,
        sky_background=75, agc_max=40):
    """
    Render a synthetic star field.

    Parameters
    ----------
    x_array, y_array : ndarray
        Star pixel coordinates.

    mag_array : ndarray
        Stellar magnitudes (smaller = brighter).

    shape : tuple
        (height, width)

    sigma : float
        Gaussian PSF width in pixels.

    min_brightness : float
        Brightness assigned to faintest star.

    max_brightness : float
        Brightness assigned to brightest star.

    add_noise : bool
        Add Gaussian image noise.

    noise_sigma : float
        Noise standard deviation.
    """


    H, W = shape

    image = np.zeros(
        (H, W),
        dtype=np.float32
    )

    #
    # Pixel coordinates
    #
    x = np.round(x_array).astype(int)
    y = np.round(y_array).astype(int)



    #
    # Clip to image bounds
    #
    mask = (
        (x >= 10) & (x < W - 10) &
        (y >= 10) & (y < H - 10)
    )

    #
    # Convert magnitude -> brightness
    #
    # Bright stars have SMALL magnitudes,
    # so invert the scale.
    #

    range_min, range_max = mag_array[mask].min(), mag_array[mask].max()
    flux = np.interp(mag_array,(range_min,range_max),(max_brightness, min_brightness)
    )

    #
    # Render delta-function stars
    #
    image[y[mask], x[mask]] += flux[mask]


    #
    # Apply common stellar PSF
    #
    image = scipy.ndimage.gaussian_filter(
        image,
        sigma=sigma
    )

    image_max = np.max(image)
    agc_max = agc_max + (image_max - agc_max) / 50


    # image -= image.min()
    if image_max > 0:
        image *= max_brightness / agc_max

    image += sky_background

    #
    # Optional sensor noise
    #
    if add_noise:
        image += np.random.normal(
            0,
            noise_sigma,
            image.shape
        )



    #
    # Clip to display range
    #
    image = np.clip(
        image,
        0,
        sensor_saturation
    )





    return image.astype(np.uint8)



def writePNG(image, filename):
    """
    Save a numpy image array to a PNG.

    Parameters
    ----------
    image : ndarray
        Image data. Can be uint8, uint16, float32, etc.

    filename : str or Path
        Output filename.
    """

    image = np.asarray(image)

    if image.dtype.kind == "f":
        if image.max() > image.min():
            image = (
                255 * (image - image.min())
                / (image.max() - image.min())
            ).astype(np.uint8)
        else:
            image = np.zeros_like(image, dtype=np.uint8)

    Image.fromarray(image).save(filename)




def inferColumnPositions(line):
    """
    From a line of text infer column positions

    Arguments:
        line: [str] line of text

    Return:
        names: [list] list of column names
        positions: [list] list of column positions
    """

    positions,names = [], []
    field_count = 0
    in_field = False
    for i, char in enumerate(line):
        if not in_field and not char.isspace():
            field_count += 1
            if field_count > 1:
                end = i
                field_name = line[start:end].strip()
                if field_name.replace("#","") != "":
                    field_name = field_name.replace("#","")
                names.append(field_name)
                positions.append((start, end))
            if i > 0:
                start = i - 1
            else:
                start = i
            in_field = True
        elif in_field and char.isspace():
            in_field = False
    end = i
    names.append(line[start:end].strip())
    positions.append((start, len(line)))

    if in_field:
        positions.append((start, len(line)))
    return names, positions


def loadIAUCSN(catalog_path=os.path.expanduser("~/source/RMS/Catalogs/"), catalog_name="IAU-CSN.txt", additional_fields = []):
    """

    Arguments:
        catalog_path: [str] path to catalog
        catalog_name: [str] name of catalog file
        additional_fields: [list] names of additional fields to return

    Returns:
        star_data: [list] of star data [ra, dec. m]
        mag_band: [list] of mag band data per star
        mag_band_ratios: [list] each entry zero, not supported for this catalog
        extra_values_dict:[dict] dictionary indexed of extra values
    """

    extra_values_dict = {}

    for field in additional_fields:
        extra_values_dict[field] = []


    star_data, name_list, mag_band, mag_band_ratios = [], [], [], []
    with open(os.path.join(os.path.expanduser(catalog_path), catalog_name), 'r') as catalog_file:

        for line in catalog_file:
            line_counter = 0
            if line.startswith('#') or line.startswith('$') or line == '\n':
                pass
            else:
                break
            header_line = line
        names, positions = inferColumnPositions(header_line)

        for line in catalog_file:
            if line.startswith('#') or line.startswith('$'):
                continue
            if not line[0].isalnum():
                continue
            f = []
            for col in range(0,len(positions)):
                f.append(line[positions[col][0]:positions[col][1]].strip())
            try:
                r, d, m, mb = float(f[12]), float(f[13]), float(f[8]), f[9]
                star_data.append(np.array([float(r), float(d), float(m)]))
                mag_band.append(mb)
                mag_band_ratios.append(0)
                for additional_field in additional_fields:
                    additional_field_column = names.index(additional_field)
                    value = line[positions[additional_field_column][0]:positions[additional_field_column][1]].strip()
                    if "name" in additional_field.lower():
                        extra_values_dict[additional_field].append(value.encode('utf-8'))
                    else:
                        extra_values_dict[additional_field].append(value)
            except:
                print("Error loading IAU-CSN line {}".format(line))

        star_data = np.array(star_data)

        for field in additional_fields:
            extra_values_dict[field] = np.array(extra_values_dict[field])


    return star_data, mag_band, mag_band_ratios, extra_values_dict

def renderIAUNames(iau_csn_coordinates, iau_extras, ra_array, dec_array, x_array, y_array, pp, max_sep_deg=0.05):

    iau_ra, iau_dec = iau_csn_coordinates[:, 0], iau_csn_coordinates[:, 1]
    iau_names = iau_extras['Name/ASCII']

    results = []

    for ra, dec, x, y in zip(ra_array, dec_array, x_array, y_array):

        sep = angularSeparationDeg(ra, dec, iau_ra, iau_dec)

        idx = np.argmin(sep)

        if sep[idx] < max_sep_deg:
            results.append((iau_names[idx], x, y))

        label_img = np.zeros((pp.Y_res, pp.X_res), dtype=np.uint8)

        for name, x, y in results:
            cv2.putText(label_img, name.decode("utf-8"), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.4,255,1, cv2.LINE_AA)
    return label_img



def renderConstellationLabels(config, pp, frame_time, intensity=150, constellation_centres=None, three_letter_names=False):

    """Render constellation labels onto an image.

    Arguments:
        config: [Config] RMS configuration object.
        pp: [Platepar] RMS platepar object.
        frame_time: [datetime] Frame time.

    Keyword Arguments:
        intensity: [int] Optional label intensity.
        constellation_centres: [dict] Optional constellation centre
            dictionary returned by computeConstellationCentres().

    Return:
        img: [ndarray] Rendered label image.
    """

    CONSTELLATION_NAMES = {
        "AND": "Andromeda",
        "ANT": "Antlia",
        "APS": "Apus",
        "AQL": "Aquila",
        "AQR": "Aquarius",
        "ARA": "Ara",
        "ARI": "Aries",
        "AUR": "Auriga",
        "BOO": "Boötes",
        "CAE": "Caelum",
        "CAM": "Camelopardalis",
        "CAP": "Capricornus",
        "CAR": "Carina",
        "CAS": "Cassiopeia",
        "CEN": "Centaurus",
        "CEP": "Cepheus",
        "CET": "Cetus",
        "CHA": "Chamaeleon",
        "CIR": "Circinus",
        "CMA": "Canis Major",
        "CMI": "Canis Minor",
        "CNC": "Cancer",
        "COL": "Columba",
        "COM": "Coma Berenices",
        "CRA": "Corona Australis",
        "CRB": "Corona Borealis",
        "CRT": "Crater",
        "CRU": "Crux",
        "CRV": "Corvus",
        "CVN": "Canes Venatici",
        "CYG": "Cygnus",
        "DEL": "Delphinus",
        "DOR": "Dorado",
        "DRA": "Draco",
        "EQU": "Equuleus",
        "ERI": "Eridanus",
        "FOR": "Fornax",
        "GEM": "Gemini",
        "GRU": "Grus",
        "HER": "Hercules",
        "HOR": "Horologium",
        "HYA": "Hydra",
        "HYI": "Hydrus",
        "IND": "Indus",
        "LAC": "Lacerta",
        "LEO": "Leo",
        "LEP": "Lepus",
        "LIB": "Libra",
        "LMI": "Leo Minor",
        "LUP": "Lupus",
        "LYN": "Lynx",
        "LYR": "Lyra",
        "MEN": "Mensa",
        "MIC": "Microscopium",
        "MON": "Monoceros",
        "MUS": "Musca",
        "NOR": "Norma",
        "OCT": "Octans",
        "OPH": "Ophiuchus",
        "ORI": "Orion",
        "PAV": "Pavo",
        "PEG": "Pegasus",
        "PER": "Perseus",
        "PHE": "Phoenix",
        "PIC": "Pictor",
        "PSA": "Piscis Austrinus",
        "PSC": "Pisces",
        "PUP": "Puppis",
        "PYX": "Pyxis",
        "RET": "Reticulum",
        "SCL": "Sculptor",
        "SCO": "Scorpius",
        "SCT": "Scutum",
        "SER": "Serpens",
        "SEX": "Sextans",
        "SGE": "Sagitta",
        "SGR": "Sagittarius",
        "TAU": "Taurus",
        "TEL": "Telescopium",
        "TRA": "Triangulum Australe",
        "TRI": "Triangulum",
        "TUC": "Tucana",
        "UMA": "Ursa Major",
        "UMI": "Ursa Minor",
        "VEL": "Vela",
        "VIR": "Virgo",
        "VOL": "Volans",
        "VUL": "Vulpecula",
    }


    if constellation_centres is None:
        constellation_centres = computeConstellationCentres(
            loadConstellationBoundaries(config)
        )

    # Get the frame time in JD
    jd = datetime2JD(frame_time)

    # Get the platepar sky parameters
    ra_c, dec_c, max_separation, _, _ = getPlateparSkyParameters(pp, jd)

    # Create image
    img = np.zeros((pp.Y_res, pp.X_res), dtype=np.uint8)

    for name, centre in constellation_centres.items():

        ra_deg = centre["ra_deg"]
        dec_deg = centre["dec_deg"]

        # Only process labels near the field of view
        if angularSeparationDeg(ra_deg, dec_deg, ra_c, dec_c) > max_separation:
            continue

        x, y = raDecToXYPP(np.array([ra_deg]), np.array([dec_deg]), jd, pp)
        x, y = x[0], y[0]

        # Skip labels outside the image
        if x < 0 or x >= pp.X_res or y < 0 or y >= pp.Y_res:
            continue

        if not three_letter_names:
            name =  CONSTELLATION_NAMES.get(name, name)
        cv2.putText(img, name, (int(round(x)), int(round(y))), cv2.FONT_HERSHEY_SIMPLEX, 0.4, intensity, 1, cv2.LINE_AA)

    return img


def renderConstellations(config, pp, frame_time, intensity=150, roll_back=40, constellation_data=None, constellation_centres=None):

    if constellation_data is None:
        constellations_path = os.path.expanduser("~/source/RMS/share/constellation_lines.csv")
        lines = np.loadtxt(constellations_path, delimiter=",")
    else:
        lines = constellation_data




    # Get the time of the frame in jd
    jd = datetime2JD(frame_time)

    ra_c, dec_c, max_separation, _, _ = getPlateparSkyParameters(pp, jd)

    from_ra, from_dec = lines[:, 0], lines[:, 1]
    to_ra, to_dec = lines[:, 2], lines[:, 3]
    ang_sep_to = angularSeparationDeg(to_ra, to_dec, ra_c, dec_c)
    ang_sep_from = angularSeparationDeg(from_ra, from_dec, ra_c, dec_c)
    line_mask = (ang_sep_to < max_separation) & (ang_sep_from < max_separation)

    from_ra, from_dec = from_ra[line_mask], from_dec[line_mask]
    to_ra, to_dec = to_ra[line_mask], to_dec[line_mask]


    from_x, from_y = raDecToXYPP(from_ra, from_dec, jd, pp)
    to_x, to_y = raDecToXYPP(to_ra, to_dec, jd, pp)

    img = np.zeros((pp.Y_res, pp.X_res), dtype=np.uint8)


    for i in range(len(to_x)):
            ok, p1, p2 = cv2.clipLine((0, 0, pp.X_res, pp.Y_res),(int(round(from_x[i])), int(round(from_y[i]))), (int(round(to_x[i])), int(round(to_y[i]))))
            try:
                p1, p2 = vectShort(p1, p2, roll_back)
                x1, y1 = p1[0], p1[1]
                x2, y2 = p2[0], p2[1]
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color=150, thickness=1)
            except:
                continue

    return img

def renderVideo(start_time, end_time, seconds_per_frame, fps, pp):

    print(f"Rendering video for station {pp.station_code} with az {pp.az_centre:.2f} el {pp.alt_centre:.2f}")
    video = cv2.VideoWriter(
        os.path.expanduser("~/tmp/sidebyside/synthetic_sky.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (pp.X_res, pp.Y_res),
        isColor=False
    )

    duration_seconds = (end_time - start_time).total_seconds()
    number_of_frames_to_render = int(np.floor(duration_seconds / seconds_per_frame))

    iau_csn_coordinates, iau_mag_band, iau_mag_band_ratios, iau_extras = loadIAUCSN(additional_fields=['Name/ASCII'])

    render_start_time = datetime.now(UTC)
    for frame_number in range(number_of_frames_to_render):
        if frame_number % 100 == 0 and frame_number > 1:
            time_now = datetime.now(UTC)
            elapsed_seconds = (time_now - render_start_time).total_seconds()
            frames_processed_per_second = elapsed_seconds / frame_number
            completion_time = time_now + timedelta(seconds=frames_processed_per_second * (number_of_frames_to_render - frame_number))
            print(f"{frame_number}/{number_of_frames_to_render}/{completion_time}")
        seconds_advanced = frame_number * seconds_per_frame
        frame_time = start_time + timedelta(seconds=seconds_advanced)
        jd_frame = datetime2JD(frame_time)
        pp, ra_array, dec_array, x_array, y_array, mag_array, names_array = cat.queryPlatepar(pp, jd_frame, n_brightest=200)
        stars = renderStarsGaussian(x_array, y_array, mag_array, names_array,(pp.Y_res, pp.X_res))
        cons = renderConstellations(config, pp, frame_time)
        iau_names = renderIAUNames(iau_csn_coordinates, iau_extras, ra_array, dec_array, x_array, y_array, pp)
        timestamp = frame_time.strftime("%Y-%m-%d %H:%M:%S")

        # Draw text to image
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = pp.station_code + " " + timestamp + " UTC (IMX291 simulation)"
        position = (10, pp.Y_res - 6)
        font_scale = 0.4
        thickness = 1

        img = np.maximum.reduce([stars, cons , iau_names])
        cv2.putText(img, text, position, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, position, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        video.write(img)


    video.release()



def annotateRadec(img, jd):

    solve_result = astrometryNetSolveLocal(img=img)

    ra, dec = solve_result[0], solve_result[1]

    text = f"RA: {ra:.4f}  Dec: {dec:.4f}"
    cv2.putText(
        img,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        0,
        4,
        cv2.LINE_AA
    )

    cv2.putText(
        img,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        255,
        2,
        cv2.LINE_AA
    )

    return img


def testGenerateSynthetic(cat, az=None, el=None, n_brightest=100):


    print("Generate synthetic images now and use plate solver to test accuracy")

    datetime_now = datetime.now(UTC)
    jd_now = datetime2JD(datetime_now)
    pp = Platepar()
    pp.read(os.path.join(config.rms_root_dir, config.platepar_name))

    pp, ra_array, dec_array, x_array, y_array, mag_array, names_array = cat.queryPlatepar(pp, jd_now, az=az, el=el, n_brightest=n_brightest)

    # Create images with the default platepar pointing

    constellation_image = renderConstellations(config, pp, datetime_now, roll_back=20)
    constellation_label_image = renderConstellationLabels(config, pp, datetime_now)
    img = renderStarsSimple(x_array, y_array, (pp.Y_res, pp.X_res))
    img = np.maximum.reduce([img, constellation_image])
    writePNG(img, os.path.join(os.path.expanduser("~/simple_output.png")))
    stars = renderStarsGaussian(x_array, y_array, mag_array, names_array,(pp.Y_res, pp.X_res))
    iau_csn_coordinates, iau_mag_band, iau_mag_band_ratios, iau_extras = loadIAUCSN(additional_fields=['Name/ASCII'])

    names = renderIAUNames(iau_csn_coordinates, iau_extras, ra_array, dec_array, x_array, y_array, pp)
    img = np.maximum.reduce([stars, constellation_image, names, constellation_label_image])
    writePNG(img, os.path.join(os.path.expanduser("~/gaussian_output.png")))

    solve_result = astrometryNetSolveLocal(img=img)
    ra_centre_solve = solve_result[0]
    dec_centre_solve = solve_result[1]
    print(f"Found a starfield at ra:{ra_centre_solve:.2f}, dec:{dec_centre_solve:.2f} jd:{jd_now}")


    # Compute az and el
    az, el = raDec2AltAz(ra_centre_solve, dec_centre_solve, jd_now, pp.lat, pp.lon)
    print(f"Measured as local azimuth and elevation of {az:.2f},{el:.2f}")
    print(f"Station az and el is {pp.az_centre:.2f},{pp.alt_centre:.2f}")
    az_el_angle_error = angularSeparationDeg(az, el, pp.az_centre, pp.alt_centre)
    print(f"Computed azimuth elevation angle error of {az_el_angle_error:.2f}")


def generateComparisonImages(cat):

    path_to_test_dir = os.path.join(config.rms_root_dir, "Tests/ExampleStationData/stations.tar.bz2")
    extractTarBz2(path_to_test_dir)
    path_to_stations =Path(os.path.join(config.rms_root_dir, "Tests/ExampleStationData/stations/Stations"))

    # Get the paths to each of the stations
    station_path_list = []
    for station_dir in path_to_stations.iterdir():
        if not station_dir.is_dir():
            continue
        station_path_list.append(station_dir)

    station_path_list.sort()
    # Iterate over each of the station paths



    for station_path in station_path_list:

        # Load the config for this station
        test_config = loadConfigFromDirectory(".config", str(station_path))

        # Load the platepar for this station
        test_pp = Platepar()
        test_pp.read(os.path.join(station_path, config.platepar_name))

        # Get the fits file
        fits_file = next(station_path.glob("*.fits"))


        # Get the jd of the test image
        test_jd = date2JD(*getMiddleTimeFF(fits_file.name, test_config.fps))

        # Query for stars in the platepar
        pp, ra_array, dec_array, x_array, y_array, mag_array, names_array = cat.queryPlatepar(test_pp, test_jd, n_brightest=100)
        synthetic_img = renderStarsGaussian(x_array, y_array, mag_array, names_array, (pp.Y_res, pp.X_res))

        max_pixel = FFfits.read(str(station_path),fits_file.name).maxpixel
        comparison_img = np.hstack((max_pixel, synthetic_img))
        output_file = fits_file.with_name(f"{fits_file.stem}_comparison.png")
        writePNG(comparison_img, output_file)
        print(f"Written comparison image to {output_file}")


if __name__ == "__main__":


    # Extract the directory of test data in the RMS repo


    config = loadConfigFromDirectory(".config", ".")
    cat = Catalog(config)

    generateComparisonImages(cat)
    testGenerateSynthetic(cat)
    testGenerateSynthetic(cat, az=237 , el=-39, n_brightest=100)




    start_time = datetime(year=2026, month=10, day=29, hour=11, minute=18, second=57)
    end_time = datetime(year=2026, month=10, day=29, hour=20, minute=38, second=55)

    pp = Platepar()
    pp.read(os.path.join(config.rms_root_dir, config.platepar_name))
    renderVideo(start_time, end_time, 256/25, 30, pp)

    pass