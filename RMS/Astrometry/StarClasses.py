"""
Shared star container classes for astrometry operations.

This module provides container classes for catalog stars, geo points, and paired stars
that are used by both SkyFit2 and AutoPlatepar.
"""

from __future__ import print_function, division, absolute_import

import numpy as np


class CatalogStar(object):
    def __init__(self, ra, dec, mag):
        """ Container for a catalog star.

        Arguments:
            ra: [float] Right ascension in degrees.
            dec: [float] Declination in degrees.
            mag: [float] Magnitude.
        """

        self.pick_type = "star"

        self.ra = ra
        self.dec = dec
        self.mag = mag


    def coords(self):
        """ Return sky coordinates.

        Returns:
            tuple: (ra, dec, mag)
        """

        return self.ra, self.dec, self.mag



class GeoPoint(object):
    def __init__(self, geo_points_obj, geo_point_index):
        """ Container for a geo point.

        Arguments:
            geo_points_obj: [object] GeoPoints object containing coordinate arrays.
            geo_point_index: [int] Index into the geo_points_obj arrays.
        """

        self.pick_type = "geopoint"

        self.geo_points_obj = geo_points_obj
        self.geo_point_index = geo_point_index


    def coords(self):
        """ Return sky coordinates.

        Returns:
            tuple: (ra, dec, mag) where mag is always 1.0 for geo points.
        """

        ra = self.geo_points_obj.ra_data[self.geo_point_index]
        dec = self.geo_points_obj.dec_data[self.geo_point_index]
        mag = 1.0

        return ra, dec, mag



class PlanetPoint(object):
    def __init__(self, name, ra, dec, mag):
        """ Container for a solar system body (planet, Moon, Sun).

        Arguments:
            name: [str] Name of the body (e.g., 'Jupiter', 'Moon').
            ra: [float] Right ascension in degrees.
            dec: [float] Declination in degrees.
            mag: [float] Apparent magnitude.
        """

        self.pick_type = "planet"

        self.name = name
        self.ra = ra
        self.dec = dec
        self.mag = mag


    def coords(self):
        """ Return sky coordinates.

        Returns:
            tuple: (ra, dec, mag)
        """

        return self.ra, self.dec, self.mag



class PairedStars(object):
    def __init__(self):
        """ Container for picked stars and geo points. """

        self.paired_stars = []


    def addPair(self, x, y, fwhm, intens_acc, obj, snr=0, saturated=False):
        """ Add a pair between image coordinates and a star or a geo point.

        Arguments:
            x: [float] Image X coordinate.
            y: [float] Image Y coordinate.
            fwhm: [float] Full width at half maximum (px).
            intens_acc: [float] Sum of pixel intensities.
            obj: [object] Instance of CatalogStar or GeoPoint.
            snr: [float] Signal-to-noise ratio. Default is 0.
            saturated: [bool] Whether the star is saturated. Default is False.
        """

        self.paired_stars.append([x, y, fwhm, intens_acc, obj, snr, saturated])


    def removeGeoPoints(self):
        """ Remove all geo points from the list of pairs. """

        self.paired_stars = [entry for entry in self.paired_stars if entry[4].pick_type != "geopoint"]


    def findClosestPickedStarIndex(self, pos_x, pos_y):
        """ Finds the index of the closest picked star on the image to the given image position.

        Arguments:
            pos_x: [float] Image X coordinate.
            pos_y: [float] Image Y coordinate.

        Returns:
            int: Index of the closest star.
        """

        min_index = 0
        min_dist = np.inf

        picked_x = [star[0] for star in self.paired_stars]
        picked_y = [star[1] for star in self.paired_stars]

        # Find the index of the closest catalog star to the given image coordinates
        for i, (x, y) in enumerate(zip(picked_x, picked_y)):

            dist = (pos_x - x)**2 + (pos_y - y)**2

            if dist < min_dist:
                min_dist = dist
                min_index = i

        return min_index


    def removeClosestPair(self, pos_x, pos_y):
        """ Remove pair closest to the given image coordinates.

        Arguments:
            pos_x: [float] Image X coordinate.
            pos_y: [float] Image Y coordinate.

        Returns:
            The removed pair, or None if no pairs exist.
        """

        if not len(self.paired_stars):
            return None

        # Find the closest star to the coordinates
        min_index = self.findClosestPickedStarIndex(pos_x, pos_y)

        # Remove the star from the list
        return self.paired_stars.pop(min_index)


    def imageCoords(self, draw=False):
        """ Return a list of image coordinates of the pairs.

        Keyword arguments:
            draw: [bool] Add an offset of 0.5 px for drawing using pyqtgraph.

        Returns:
            list: List of (x, y, intens_acc) tuples.
        """

        offset = 0
        if draw:
            offset = 0.5

        img_coords = [(x + offset, y + offset, intens_acc) for x, y, _, intens_acc, _, _, _
                      in self.paired_stars]

        return img_coords


    def skyCoords(self):
        """ Return a list of sky coordinates.

        Returns:
            list: List of (ra, dec, mag) tuples.
        """

        return [obj.coords() for _, _, _, _, obj, _, _ in self.paired_stars]


    def allCoords(self):
        """ Return all coordinates, image and sky in the [(x, y, fwhm, intens_acc, snr, saturated), (ra, dec, mag)]
            list form for every entry.

        Returns:
            list: List of [(img_data), (sky_data)] pairs.
        """

        return [
            [(x, y, fwhm, intens_acc, snr, saturated), obj.coords()]
            for x, y, fwhm, intens_acc, obj, snr, saturated in self.paired_stars
            ]

    def snr(self):
        """ Return a list of SNR values.

        Returns:
            list: List of SNR values.
        """

        return [snr for _, _, _, _, _, snr, _ in self.paired_stars]


    def __len__(self):
        """ Return the total number of paired stars. """

        return len(self.paired_stars)
