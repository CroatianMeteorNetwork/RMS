# RPi Meteor Station
# Copyright (C) 2023
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

from __future__ import print_function, division, absolute_import


import os
import sys
import shutil

import datetime
import time
import dateutil
import glob
import sqlite3
import multiprocessing
import logging
import copy
import uuid
import random
import string


if sys.version_info[0] < 3:

    import urllib2

    # Fix Python 2 SSL certs
    try:
        import os, ssl
        if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
            getattr(ssl, '_create_unverified_context', None)): 
            ssl._create_default_https_context = ssl._create_unverified_context
    except:
        # Print the error
        print("Error: {}".format(sys.exc_info()[0]))

else:
    import urllib.request


import numpy as np

import RMS.ConfigReader as cr

from RMS.Astrometry.Conversions import datetime2JD, geo2Cartesian, altAz2RADec, vectNorm, raDec2Vector
from RMS.Astrometry.Conversions import latLonAlt2ECEF, AER2LatLonAlt, AEH2Range, ECEF2AltAz, ecef2LatLonAlt
from RMS.Math import angularSeparationVect
from RMS.Formats.FFfile import convertFRNameToFF
from RMS.Formats.Platepar import Platepar
from RMS.UploadManager import uploadSFTP
from Utils.StackFFs import stackFFs
from Utils.FRbinViewer import view
from Utils.BatchFFtoImage import batchFFtoImage
from RMS.CaptureDuration import captureDuration
from RMS.Misc import sanitise


# Import Cython functions
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from RMS.Astrometry.CyFunctions import cyTrueRaDec2ApparentAltAz

log = logging.getLogger("logger")
EM_RAISE = False

"""

# Reference event

# Required
EventTime                : 20230821_110845	#Time as YYYYMMDD_HHMMSS
TimeTolerance (s)        : 20			    #Width of time to take
EventLat (deg +N)        : -33			    #Event start latitude
EventLon (deg +E)        : 116.0		    #Event start longitude
EventHt (km)             : 100			    #Event height in km
CloseRadius(km)          : 10			    #Radius of stations to trajectory which must upload information for time
FarRadius (km)           : 1000			    #Radius of stations to trajectory which must upload if trajectory passed through field of view

#Either
EventLat2Std (deg)       : 1.0			    #Event end latitude standard deviation
EventLon2 (deg +E)       : 116.0		    #Event end longitude
EventHt2 (km)            : 90			    #Event height standard deviation

#Or
EventAzim                : 45			    #Azimuth from North +E from point of view of object
EventElev		         : 20			    #Elevation as perceived by observer on ground, hence always +ve

#Optional
EventCartStd		     : 10000		    #Event start cartesian standard deviation (m)
EventCart2Std		     : 10000		    #Event end cartesian standard deviation (m)
RequireFR                : 0                #If not zero only upload if a file FR*.bin exists 

#Optional - not preferred as sensitive to different latitudes
EventLatStd (deg)	     : 1.0			    #Event start latitude polar standard deviation
EventLonStd (deg)	     : 1.0		        #Event start longitude polar standard deviation
EventLat2 (deg +N)       : -31			    #Event end latitude
EventLon2Std (deg)	     : 1.0			    #Event end longitude standard deviation
Suffix                   : event            #Free text suffix to the uploaded archive

END						                    #Event delimiter - everything after this is associated with a new event

"""

""" Automatically uploads data files based on search specification information given on a website. """

class EventContainer(object):

    """ Contains the specification of the search for an event.

    """
    def __init__(self, dt, lat, lon, ht):

        # Required parameters
        self.dt, self.time_tolerance = dt, 0
        self.lat, self.lat_std, self.lon, self.lon_std, self.ht, self.ht_std, self.cart_std = lat, 0, lon, 0, ht, 0, 0
        self.lat2, self.lat2_std, self.lon2, self.lon2_std, self.ht2, self.ht2_std, self.cart2_std = 0, 0, 0, 0, 0, 0, 0
        self.close_radius, self.far_radius = 0, 0
        self.require_FR = 0

        # Or trajectory information from the first point
        self.azim, self.azim_std, self.elev, self.elev_std, self.elev_is_max = 0, 0, 0, 0, False
        self.stations_required = ""
        self.respond_to = ""

        # These are internal control properties
        self.uuid = ""
        self.event_spec_type = 0
        self.files_uploaded = []
        self.time_completed = None
        self.observed_status = None
        self.processed_status = False

        self.start_distance, self.start_angle, self.end_distance, self.end_angle = 0, 0, 0, 0
        self.fovra, self.fovdec = 0, 0
        self.suffix = "event"

    def setValue(self, variable_name, value):

        """ Receive a name and value pair, and put them into this event

        Arguments:
            variable_name: Name of the variable
            value        : Value to be assigned

        Return:
            Nothing
        """
        # Extract the variable name, truncate before any '(' used for units
        variable_name = variable_name.strip().split('(')[0].strip()

        if value == "":
            return

        # Mandatory parameters
        self.dt = value if "EventTime" == variable_name else self.dt
        self.time_tolerance = value if "TimeTolerance" == variable_name else self.time_tolerance
        self.lat = float(value) if "EventLat" == variable_name else self.lat
        self.lat_std = float(value) if "EventLatStd" == variable_name else self.lat_std
        self.lon = float(value) if "EventLon" == variable_name else self.lon
        self.lon_std = float(value) if "EventLonStd" == variable_name else self.lon_std
        self.ht = float(value) if "EventHt" == variable_name else self.ht
        self.ht_std = float(value) if "EventHtStd" == variable_name else self.ht_std
        self.cart_std = float(value) if "EventCartStd" == variable_name else self.cart_std
        if "RequireFR" == variable_name:
            if str(value) == "0" :
                self.require_FR = 0
            else:
                self.require_FR = 1

        # Radii
        self.close_radius = float(value) if "CloseRadius" == variable_name else self.close_radius
        self.far_radius = float(value) if "FarRadius" == variable_name else self.far_radius

        # Optional parameters, if trajectory is set by a start and an end
        self.lat2 = float(value) if "EventLat2" == variable_name else self.lat2
        self.lat2_std = float(value) if "EventLat2Std" == variable_name else self.lat2_std
        self.lon2 = float(value) if "EventLon2" == variable_name else self.lon2
        self.lon2_std = float(value) if "EventLon2Std" == variable_name else self.lon2_std
        self.ht2 = float(value) if "EventHt2" == variable_name else self.ht2
        self.ht2_std = float(value) if "EventHt2Std" == variable_name else self.ht2_std
        self.cart2_std = float(value) if "EventCart2Std" == variable_name else self.cart2_std

        # Optional parameters for defining trajectory by a start point, and a direction
        if "EventAzim" == variable_name:
            self.azim = 0 if value is None else float(value)

        if "EventAzimStd" == variable_name:
            self.azim_std = 0 if value is None else float(value)

        if "EventElev" == variable_name:
            self.elev = 0 if value is None else float(value)

        if "EventElevStd" == variable_name:
            self.elev_std = 0 if value is None else float(value)

        # This code is used for reading event_watchlist.txt and database queries
        # Text stores as True, database stores as 0 and 1
        if "EventElevIsMax" == variable_name:
            self.elev_is_max = True if value == "True" or value == 1 else False

        # Control information
        self.stations_required = str(value) if "StationsRequired" == variable_name else self.stations_required
        self.uuid = str(value) if "uuid" == variable_name else self.uuid
        self.respond_to = str(value) if "RespondTo" == variable_name else self.respond_to

    def eventToString(self):

        """ Turn an event into a string

        Arguments:

        Return:
            String representation of an event
        """

        output = "# Required \n"
        output += ("EventTime                : {}\n".format(self.dt))
        output += ("TimeTolerance (s)        : {}\n".format(self.time_tolerance))
        output += ("EventLat (deg +N)        : {:3.2f}\n".format(self.lat))
        output += ("EventLatStd (deg)        : {:3.2f}\n".format(self.lat_std))
        output += ("EventLon (deg +E)        : {:3.2f}\n".format(self.lon))
        output += ("EventLonStd (deg)        : {:3.2f}\n".format(self.lon_std))
        output += ("EventHt (km)             : {:3.2f}\n".format(self.ht))
        output += ("EventHtStd (km)          : {:3.2f}\n".format(self.ht_std))
        output += ("EventCartStd (km)          : {:3.2f}\n".format(self.cart_std))
        output += ("CloseRadius(km)          : {:3.2f}\n".format(self.close_radius))
        output += ("FarRadius (km)           : {:3.2f}\n".format(self.far_radius))
        output += "\n"
        output += "# Optional second point      \n"
        output += ("EventLat2 (deg +N)       : {:3.2f}\n".format(self.lat2))
        output += ("EventLat2Std (deg)       : {:3.2f}\n".format(self.lat2_std))
        output += ("EventLon2 (deg +E)       : {:3.2f}\n".format(self.lon2))
        output += ("EventLon2Std (deg)       : {:3.2f}\n".format(self.lon2_std))
        output += ("EventHt2 (km)            : {:3.2f}\n".format(self.ht2))
        output += ("EventHtStd2 (km)         : {:3.2f}\n".format(self.ht2_std))
        output += ("EventCartStd2 (km)         : {:3.2f}\n".format(self.cart2_std))
        output += "\n"
        output += "# Or a trajectory instead    \n"
        output += ("EventAzim (deg +E of N)  : {:3.2f}\n".format(self.azim))
        output += ("EventAzimStd (deg)       : {:3.2f}\n".format(self.azim_std))
        output += ("EventElev (deg)          : {:3.2f}\n".format(self.elev))
        output += ("EventElevStd (deg):      : {:3.2f}\n".format(self.elev_std))
        output += ("EventElevIsMax           : {:3.2f}\n".format(self.elev_is_max))
        output += "\n"
        output += "# Control information        \n"
        output += ("StationsRequired         : {}\n".format(self.stations_required))
        output += ("uuid                     : {}\n".format(self.uuid))
        output += ("RespondTo                : {}\n".format(self.respond_to))
        output += "# Trajectory information     \n"
        output += ("Start Distance (km)      : {:3.2f}\n".format(self.start_distance / 1000))
        output += ("Start Angle              : {:3.2f}\n".format(self.start_angle))
        output += ("End Distance (km)        : {:3.2f}\n".format(self.end_distance / 1000))
        output += ("End Angle                : {:3.2f}\n".format(self.end_angle))
        output += "# Station information        \n"
        output += ("Field of view RA         : {:3.2f}\n".format(self.fovra))
        output += ("Field of view Dec        : {:3.2f}\n".format(self.fovdec))
        output += ("Suffix                   : {}\n".format(self.suffix))
        output += "\n"
        output += "END"
        output += "\n"
        return output

    def isReasonable(self):

        """ Check if self is reasonable, and optionally try to fix it up
            Crucially, this function prevents any excessive requests being made that may compromise capture

        Arguments:

        Return:
            reasonable: [bool] The event is reasonable
        """

        reasonable = True
        reasonable = False if self.lat == "" else reasonable
        reasonable = False if self.lat is None else reasonable
        reasonable = False if self.lon == "" else reasonable
        reasonable = False if self.lon is None else reasonable
        reasonable = False if 0 < float(self.time_tolerance) > 300 else reasonable
        reasonable = False if self.close_radius > self.far_radius else reasonable

        return reasonable

    def hasCartSD(self):

        """
        Event contains any non-zero cartesian deviation parameters

        returns:
            [bool]

        """

        return self.cart_std != 0 or self.cart2_std != 0

    def hasPolarSD(self):

        """
        Event contains any non-zero polar deviation parameters

        returns:
            [bool]

        """

        sd_used = False
        sd_used = True if self.lat_std != 0 else sd_used
        sd_used = True if self.lon_std != 0 else sd_used
        sd_used = True if self.ht_std != 0 else sd_used
        sd_used = True if self.lat2_std != 0 else sd_used
        sd_used = True if self.lon2_std != 0 else sd_used
        sd_used = True if self.ht2_std != 0 else sd_used
        sd_used = True if self.azim_std != 0 else sd_used
        sd_used = True if self.elev_std != 0 else sd_used

        return sd_used

    def appendPopulation(self, population, population_size):

        """
        Append to a population identical copies of self event

        arguments:
            population: [list] population of events

        return:
            population [list] population of events

        """

        for pop_num in range(0, population_size):
            population.append(copy.copy(self))
        return population

    def eventToECEFVector(self):

        """
        Return ECEF vector (meters) representation of search trajectory

        return:
            [vector] ECEF vector
        """

        v1 = latLonAlt2ECEFDeg(self.lat, self.lon, self.ht * 1000)
        v2 = latLonAlt2ECEFDeg(self.lat2, self.lon2, self.ht2 * 1000)

        return [v1, v2]

    def applyCartesianSDToPoint(self, pt, std):

        """
        Apply random number from normal distribution to each component of a 3 dimension vector
        If this fails, just return the point.

        arguments:
            pt: [vector] vector
            std: [float] sigma to apply

        """

        try:
            return pt + np.random.normal(scale=std, size=3)
        except:
            return pt

    def applyCartesianSD(self, population, seed = None):

        """
        Apply standard deviation to the Cartesian coordinate of a population of trajectories
        Take the absolute value, in case a negative value was passed

        arguments:
            population: [list] population of events

        returns:
            population: [list] population of events

        """

        if seed is not None:
            np.random.seed(seed)
        ecef_vector = self.eventToECEFVector()
        if self.hasCartSD():
            for tr in population:
                start_vect = self.applyCartesianSDToPoint(ecef_vector[0], abs(self.cart_std))
                end_vect = self.applyCartesianSDToPoint(ecef_vector[1], abs(self.cart2_std))
                tr.lat, tr.lon, tr.ht = ecefV2LatLonAlt(start_vect)
                tr.lat2, tr.lon2, tr.ht2 = ecefV2LatLonAlt(end_vect)

        return population

    def applyPolarSD(self, population, seed = None):


        """
        Apply standard deviation to the Polar coordinates of a population of trajectories
        This function can handle negative standard deviations

        arguments:
            population: [list] of events
            seed: optional, set the seed for the standard deviation generation

        returns:
            population: [list] of events

        """

        if seed is not None:
            np.random.seed(seed)

        for tr in population:
            tr.lat = tr.lat + np.random.normal(scale=1) * self.lat_std
            tr.lon = tr.lon + np.random.normal(scale=1) * self.lon_std
            tr.ht = tr.ht + np.random.normal(scale=1) * self.ht_std
            tr.lat2 = tr.lat2 + np.random.normal(scale=1) * self.lat2_std
            tr.lon2 = tr.lon2 + np.random.normal(scale=1) * self.lon2_std
            tr.ht2 = tr.ht2 + np.random.normal(scale=1) * self.ht2_std
            tr.azim = tr.azim + np.random.normal(scale=1) * self.azim_std
            tr.elev = tr.elev + np.random.normal(scale=1) * self.elev_std

        return population

    def hasAzEl(self):

        """
        Are polar end points non-zero for trajectory?

        Arguments:

        Returns: [bool]
            True if end point latitudes or longitudes or heights are not zero

        """

        azim_elev_definition = True
        azim_elev_definition = False if self.lon2 != 0 else azim_elev_definition
        azim_elev_definition = False if self.lat2 != 0 else azim_elev_definition
        azim_elev_definition = False if self.ht2 != 0 else azim_elev_definition

        return azim_elev_definition

    def limitAzEl(self, min_elev_hard, min_elev, prob_elev, max_elev):

        """
        Acts on self to correct any strange elevations

        Arguments
            min_elev_hard: [float] minimum elevation considered reasonable
            min_elev: [float] minimum elevation considered correct
            prob_elev: [float] set any unreasonable elevations
            max_elev: [float] maximum elevation considered reasonable

        Returns:
            Nothing

        """

        # Detect, fix and log elevations outside range
        if min_elev < self.elev < max_elev:
            pass
        else:
            # If elevation is not within min_elev_hard and max_elev degrees set to prob_elev
            self.elev = self.elev if min_elev_hard < self.elev < max_elev else prob_elev

            # If elevation is min_elev_hard - min_elev degrees set to min_elev
            self.elev = min_elev if min_elev_hard < self.elev < min_elev else self.elev

    def limitHeights(self, obsvd_ht, min_lum_flt_ht, max_lum_flt_ht, gap):

        """
        Adjust default illuminated flight heights to match event specification. Leave a gap
        between the observation and the limit to allow accurate angles to be calculated

        Arguments
            observd_ht: [float] height of observation
            min_lum_flt_ht: [float] minimum expected illuminated flight
            max_lum_flt_ht: [float] maximum expected illuminated flight
            gap : [float] minimum gap between the observed_bt and either of the limits

            All must be specified with the same unit multiplier

        """

        max_lum_flt_ht = obsvd_ht + gap if obsvd_ht >= (max_lum_flt_ht - gap) else max_lum_flt_ht
        min_lum_flt_ht = obsvd_ht - gap if obsvd_ht <= (min_lum_flt_ht + gap) else min_lum_flt_ht

        return min_lum_flt_ht, max_lum_flt_ht

    def getRanges(self, obs_lat, obs_lon, obs_ht, min_lum_flt_ht, max_lum_flt_ht):

        """
        For an event containing a trajectory specified with two lat,lon, heights, calculate the range from
        the observed point to the maximum luminous flight height, and to the minimum luminous flight height

        arguments:
            obsvd_lat : [float] latitude (degrees) of observed point
            obsvd_lon : [float] longitude (degrees) of observed point
            obsvd_ht : [float] height (meters) of observed point
            min_lum_flt_ht: [float] height (meters) of minimum luminous flight
            max_lum_flt_ht: [float] height (meters) of maximum luminous flight

        returns:
            bwd_range : [float] range (meters) from observed point to maximum luminous height
            fwd_range : [float] range (meters) from observed point to minimum luminous height


        """

        # Find range to maximum heights in reverse trajectory direction
        bwd_range = AEH2Range(self.azim, self.elev, max_lum_flt_ht, obs_lat, obs_lon, obs_ht)

        # Find range to minimum height in forward trajectory direction.
        # This is done by reflecting the trajectory in a horizontal plane midway between obs_ht and min_lum_flt_ht
        # This simplifies the calculation, but introduces a small imprecision, caused by curvature of earth
        fwd_range = AEH2Range(self.azim, self.elev, obs_ht, obs_lat, obs_lon, min_lum_flt_ht)

        # Iterate to find accurate solution - limit iterations to 100, generally requires fewer than 10 iterations
        for n in range(100):
            self.lat2, self.lon2, ht2_m = AER2LatLonAlt(self.azim, 0 - self.elev, fwd_range, obs_lat, obs_lon, obs_ht)
            # Use trigonometry to estimate the error - vertical error is the opposite side to the elevation
            # so vertical error / sin(elev) gives the hypotenuse, which is the trajectory error
            traj_error = (ht2_m - min_lum_flt_ht) / np.sin(np.radians(self.elev))
            fwd_range = fwd_range + traj_error
            if traj_error < 1e-8:
                break

        return bwd_range, fwd_range

    def addElevationRange(self, population, ob_ev, min_elevation):

        """
        Take a single observed point on a trajectory, and a minimum elevation
        Create a population of trajectories from the min_elevation through to observed elevation
        in steps of 1 degree.

        The trajectories pivot around the observed lat, lon and height, and this function checks that
        this point is close to all the produced trajectories

        arguments:
            population: [list] list of trajectories to be appended to
            ob_ev: An observed event, specified as a lat (degrees), lon (degress), ht (km) and elevation (degrees)
            min_elevation: [degrees] observed elevation, which will always be the minimum

        returns:
            population: [list] list of trajectories

        """


        ob_ev.azim, ob_ev.elev = ob_ev.latLonlatLonToLatLonAzEl()
        for elev in range(min_elevation, int(ob_ev.elev), 1):
            s = copy.copy(ob_ev)
            s.elev = elev
            s.latLonAzElToLatLonLatLon(force=True)
            population.append(s)
            ch_az, ch_el = s.latLonlatLonToLatLonAzEl()
            start, end, closest = calculateClosestPoint(s.lat, s.lon, s.ht * 1000, s.lat2, s.lon2, s.ht2 * 1000,
                                                        ob_ev.lat, ob_ev.lon, ob_ev.ht * 1000)
            start, end, closest = start / 1000, end / 1000, closest / 1000

            if start > 1000 or end > 1000 or closest > 0.2:
                log.error("Original             Az, El {:.3f},{:.3f} degrees".format(ob_ev.azim, ob_ev.elev))
                log.error("Final                Az, El {:.3f},{:.3f} degrees".format(ch_az, ch_el))
                log.error("Final    Start Lat,Lon,Alt  {:.3f},{:.3f},{:.3f}".format(s.lat, s.lon, s.ht))
                log.error("Original Start Lat,Lon,Alt  {:.3f},{:.3f},{:.3f}".format(ob_ev.lat, ob_ev.lon, ob_ev.ht))
                log.error("Original End   Lat,Lon,Alt  {:.3f},{:.3f},{:.3f}".format(ob_ev.lat2, ob_ev.lon2, ob_ev.ht2))
                log.error("Final    End   Lat,Lon,Alt  {:.3f},{:.3f},{:.3f}".format(s.lat2, s.lon2, s.ht2))
                log.error("Distance from original start to trajectory")
                log.error("Start, End, Closest, Elev {:.2f},{:.2f},{:.2f},{:.2f}".format(start, end, closest, ch_el))
            pass

        return population

    def adjustTrajectoryLimits(self, bwd_range, fwd_range, obs_lat, obs_lon, obs_ht):

        """
        Move the start and end of a trajectory

        Extend the trajectory of this event backwards by bwd_range and forwards by fwd_range maintaining the same
        azimuth and elevation. One application for this function could be to extend a trajectory to the expected
        limits of illuminated flight.

        arguments:
            bwd_range: [float] range (meters) to extend a trajectory backwards - in line with the trajectory
            fwd_range: [float] range (meters) to extend a trajectory forwards - in line with the trajectory
            obsvd_lat: [float] observed latitude (degrees) of reference point
            obsvd_lon: [float] observed longitude (degrees) of reference point
            obsvd_ht: [float] observed height (metres) of reference point

        reurns:
            nothing

        """

        # Move event start point back to intersection with max_lum_flt_ht
        self.lat, self.lon, ht_m = AER2LatLonAlt(revAz(self.azim), self.elev, bwd_range, obs_lat, obs_lon, obs_ht)
        # Calculate end point of trajectory and convert to km
        self.lat2, self.lon2, ht2_m = AER2LatLonAlt(self.azim, 0 - self.elev, fwd_range, obs_lat, obs_lon, obs_ht)

        # Convert to km and store in event
        self.ht, self.ht2 = ht_m / 1000, ht2_m / 1000

    def latLonAzElToLatLonLatLon(self, force=False):

        """Take an event, establish how it has been defined, and convert to representation as
        a pair of Lat,Lon,Ht parameters. If force is True (default False), then any existing end point
        lat(deg), lon(deg), ht(km) will be overwritten by the value calculated from the azimuth.

        Results are checked, and significant errors are sent to error log.

        arguments:
            force: [bool] force conversion

        return:
            nothing

        """

        if not self.hasAzEl() and not force:
            return

        # Copy observed lat, lon and height local variables for ease of comprehension and convert to meters
        obs_lat, obs_lon, obs_ht = self.lat, self.lon, self.ht * 1000

        # For this routine elevation must always be within 10 - 90 degrees
        min_elev_hard, min_elev, prob_elev, max_elev = 0, 10, 45, 90
        self.limitAzEl(min_elev_hard, min_elev, prob_elev, max_elev)

        # Handle estimated start heights outside normal range of luminous flight
        # Need to add gap so that angles can be calculated for consistency checks
        # Set minimum and maximum luminous flight heights
        min_lum_flt_ht, max_lum_flt_ht, gap = 20000, 100000, 1000
        min_lum_flt_ht, max_lum_flt_ht = self.limitHeights(obs_ht, min_lum_flt_ht, max_lum_flt_ht, gap)
        bwd_range, fwd_range = self.getRanges(obs_lat, obs_lon, obs_ht, min_lum_flt_ht, max_lum_flt_ht)

        # Move the end points
        self.adjustTrajectoryLimits(bwd_range, fwd_range, obs_lat, obs_lon, obs_ht)

        # Post calculation checks - not required for operation
        # Convert to ECEF
        x1, y1, z1 = latLonAlt2ECEFDeg(self.lat, self.lon, self.ht * 1000)
        x2, y2, z2 = latLonAlt2ECEFDeg(self.lat2, self.lon2, self.ht2 * 1000)
        x_obs, y_obs, z_obs = latLonAlt2ECEFDeg(obs_lat, obs_lon, obs_ht)

        # Calculate vectors of three points on trajectory
        max_pt, min_pt, obs_pt = np.array([x1, y1, z1]), np.array([x2, y2, z2]), np.array([x_obs, y_obs, z_obs])

        # Calculate Alt Az between three points
        min_obs_az, min_obs_el = ECEF2AltAz(obs_pt, min_pt)
        min_max_az, min_max_el = ECEF2AltAz(max_pt, min_pt)
        obs_max_az, obs_max_el = ECEF2AltAz(max_pt, obs_pt)

        # Log any errors

        # Check that az from the minimum to the observation height as the same as the minimum to the maximum height
        # And the minimum to the observation height is the same as the observation to the maximum height
        if angDif(min_obs_az, min_max_az) > 1 or angDif(min_obs_az, obs_max_az) > 1:

            log.error("Error in Azimuth calculations")
            log.error("Observation at lat,lon,ht {:3.5f},{:3.5f},{:.0f}".format(obs_lat, obs_lon, obs_ht))
            log.error("Propagate fwds, bwds {:.0f},{:.0f} metres".format(fwd_range, bwd_range))
            log.error("At az, az_rev, el {:.4f} ,{:.4f} , {:.4f}".format(self.azim, revAz(self.azim) , self.elev))
            log.error("Start lat,lon,ht {:3.5f},{:3.5f},{:.0f}".format(self.lat, self.lon, self.ht * 1000))
            log.error("End   lat,lon,ht {:3.5f},{:3.5f},{:.0f}".format(self.lat2, self.lon2, self.ht2 * 1000))
            log.error("Minimum height to Observed height az,el {},{}".format(min_obs_az, min_obs_el))
            log.error("Minimum height to Maximum height az,el {},{}".format(min_max_az, min_max_el))
            log.error("Observed height to Maximum height az,el {},{}".format(obs_max_az, obs_max_el))

        # Check that el from the minimum to the observation height as the same as the minimum to the maximum height
        # And the minimum to the observation height is the same as the observation to the maximum height
        if angDif(min_obs_el, min_max_el) > 1 or angDif(min_obs_el, obs_max_el) > 1:
            log.error("Error in Elevation calculations")
            log.error("Traj from observation at lat,lon,ht {:3.5f},{:3.5f},{:.0f}".format(obs_lat, obs_lon, obs_ht))
            log.error("Propagate fwds, bwds {:.0f},{:.0f} metres".format(fwd_range, bwd_range))
            log.error("At az, az_rev, el {:.4f} ,{:.4f} , {:.4f}".format(self.azim, revAz(self.azim), self.elev))
            log.error("Start lat,lon,ht {:3.5f},{:3.5f},{:.0f}".format(self.lat, self.lon, self.ht * 1000))
            log.error("End   lat,lon,ht {:3.5f},{:3.5f},{:.0f}".format(self.lat2, self.lon2, self.ht2 * 1000))
            log.error("Minimum height to Observed height az,el {},{}".format(min_obs_az, min_obs_el))
            log.error("Minimum height to Maximum height az,el {},{}".format(min_max_az, min_max_el))
            log.error("Observed height to Maximum height az,el {},{}".format(obs_max_az, obs_max_el))

    def latLonlatLonToLatLonAzEl(self):

        """
        Populate azimuth and elevation for an event defined with two Lat,Lons and Hts

        Elevation is always positive.

        arguments:

        return:
            azimuth(degrees), elevation(degrees)
        """

        x1, y1, z1 = latLonAlt2ECEFDeg(self.lat, self.lon, self.ht * 1000)
        x2, y2, z2 = latLonAlt2ECEFDeg(self.lat2, self.lon2, self.ht2 * 1000)
        start_pt, end_pt = np.array([x1, y1, z1]), np.array([x2, y2, z2])
        end_start_az, end_start_el = ECEF2AltAz(end_pt, start_pt)
        return revAz(end_start_az), end_start_el

class EventMonitor(multiprocessing.Process):

    def __init__(self, config):
        """ Automatically uploads data files of an event (e.g. fireball) as given on the website.
        Arguments:
            config: [Config] Configuration object.
        """

        super(EventMonitor, self).__init__()
        # Hold two configs - one for the locations of folders - syscon, and one for the lats and lons etc. - config
        self.config = config        # the config that will be used for all data processing - lats, lons etc.
        self.syscon = config        # the config that describes where the folders are

        # The path to the EventMonitor database
        self.event_monitor_db_path = os.path.join(os.path.abspath(self.syscon.data_dir),
                                                  self.syscon.event_monitor_db_name)

        self.createDB()

        # Load the EventMonitor database. Any problems, delete and recreate.
        self.db_conn = self.getConnectionToEventMonitorDB()
        self.upgradeDB(self.db_conn)
        self.check_interval = self.syscon.event_monitor_check_interval
        self.exit = multiprocessing.Event()

        log.info("EventMonitor is starting")
        log.info("Monitoring {} ".format(self.syscon.event_monitor_webpage))
        log.info("At {:3.2f} minute intervals".format(self.syscon.event_monitor_check_interval))
        log.info("Reporting data to {}/{}".format(self.syscon.hostname, self.syscon.event_monitor_remote_dir))

    def createDB(self):

        """
        Attempt multiple times to create a database to hold event search specifications.

        """

        for createdb_attempts in range(30):
            self.conn = self.createEventMonitorDB()
            if self.conn is not None:
                break
            log.info("Database creation failed, waiting to retry")
            # try block because os.path.exists seems to lag behind the file system state
            try:
                if os.path.exists(self.event_monitor_db_path):
                    os.unlink(self.event_monitor_db_path)
            except:
                pass
            time.sleep(30)
            log.info("Retrying database creation")

    def createEventMonitorDB(self, test_mode=False):

        """ Creates the EventMonitor database. Tries only once.

        arguments:

        returns:
            conn: [connection] connection to database if success else None

        """

        # Create the EventMonitor database
        if test_mode:
            self.event_monitor_db_path = os.path.expanduser(os.path.join(self.syscon.data_dir, self.syscon.event_monitor_db_name))
            if os.path.exists(self.event_monitor_db_path):
                os.unlink(self.event_monitor_db_path)

        if not os.path.exists(os.path.dirname(self.event_monitor_db_path)):
            # Handle the very rare case where this could run before any observation sessions
            # and RMS_data does not exist
            os.makedirs(os.path.dirname(self.event_monitor_db_path))

        try:
            conn = sqlite3.connect(self.event_monitor_db_path)

        except:
            log.error("Failed to create event_monitor database")
            return None

        # Returns true if the table event_monitor exists in the database
        try:
            tables = conn.cursor().execute(
                """SELECT name FROM sqlite_master WHERE type = 'table' and name = 'event_monitor';""").fetchall()

            if tables:
                self.upgradeDB(conn)
                return conn
        except:
            if EM_RAISE:
                raise
            return None

        conn.execute("""CREATE TABLE event_monitor (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,   
                            EventTime TEXT NOT NULL,
                            TimeTolerance REAL NOT NULL,
                            EventLat REAL NOT NULL,
                            EventLatStd REAL NOT NULL,
                            EventLon REAL NOT NULL,
                            EventLonStd REAL NOT NULL,
                            EventHt REAL NOT NULL,
                            EventHtStd REAL NOT NULL,
                            EventCartStd REAL NOT NULL,
                            CloseRadius REAL NOT NULL,
                            FarRadius REAL NOT NULL,
                            EventLat2 REAL NOT NULL,
                            EventLat2Std REAL NOT NULL,
                            EventLon2 REAL NOT NULL,
                            EventLon2Std REAL NOT NULL,
                            EventHt2 REAL NOT NULL,
                            EventHt2Std REAL NOT NULL,
                            EventCart2Std REAL NOT NULL,
                            EventAzim REAL NOT NULL,
                            EventAzimStd REAL NOT NULL,
                            EventElev REAL NOT NULL,
                            EventElevStd REAL NOT NULL,
                            EventElevIsMax BOOL,
                            RequireFR TEXT,
                            StationsRequired TEXT,
                            filesuploaded TEXT,
                            timeadded TEXT,
                            timecompleted TEXT,
                            observedstatus BOOL,
                            processedstatus BOOL,
                            uploadedstatus BOOL,
                            receivedbyserver BOOL,
                            uuid TEXT,              
                            RespondTo TEXT,
                            Suffix TEXT
                            )""")


        # Commit the changes
        conn.commit()

        # Set the connection
        self.db_conn = conn
        return conn

    def recoverFromDatabaseError(self):

        """

        Called if a database error is detected, and tries to recreate the database and connection.
        """

        log.error("Attempting to recover from database error")
        if self.delEventMonitorDB():
            log.warning("Deleted EventMonitor database at {}".format(self.event_monitor_db_path))
        else:
            log.warning("No EventMonitor database found at {}".format(self.event_monitor_db_path))
        time.sleep(20)
        self.createDB()
        log.info("Database recovered")

    def delEventMonitorDB(self):

        """ Delete the EventMonitor database.

        Arguments:

        Return:
            Status: [bool] True if a db was found at that location, otherwise false

        """

        # This check is to prevent accidental deletion of the working directory

        if os.path.isfile(self.event_monitor_db_path):
            os.remove(self.event_monitor_db_path)
            return True
        else:
            return False

    def upgradeDB(self,conn):


        """

        Checks that any columns required by subsequent releases of EventMonitor are present. If they are not,
        then they are added.

        Args:
            conn: Connection to the EventMonitor database

        Returns: Nothing

        """

        if not self.checkDBcol(conn,"Suffix"):
            log.info("Missing db column Suffix")
            self.addDBcol("Suffix","TEXT")

        if not self.checkDBcol(conn,"Ra"):
            log.info("Missing db column Ra")
            self.addDBcol("Ra","REAL")

        if not self.checkDBcol(conn,"Dec"):
            log.info("Missing db column Dec")
            self.addDBcol("Dec","REAL")


        if not self.checkDBcol(conn,"RequireFR"):
            log.info("Missing db column RequireFR")
            self.addDBcol("RequireFR","bool")


    def addDBcol(self, column, coltype):


        """ Add a new column to the database. This is used when upgrading the database to later features.

        Arguments:
            column: [string] Name of column to add
            coltype: [string] type of column to add

        Return:
            Status: [bool] True if successful otherwise false

        """

        sql_command = ""
        sql_command += "ALTER TABLE event_monitor  \n"
        sql_command += "ADD {} {}; ".format(column, coltype)

        try:

            conn = sqlite3.connect(self.event_monitor_db_path)
            conn.execute(sql_command)
            conn.close()
            return True
        except:
            if EM_RAISE:
                raise
            return False

    def checkDBcol(self, conn,column):

        """ Check column exists

        Arguments:
            conn: [connection] database connection
            column: [string] Name of column to check for


        Return:
            Status: [bool] True if column exists

        """

        sql_command = ""
        sql_command += "SELECT COUNT(*) AS COL"
        sql_command += " FROM pragma_table_info('event_monitor')"
        sql_command += " WHERE name='{}'  \n".format(column)

        try:
            return (conn.cursor().execute(sql_command).fetchone())[0] != 0
        except:
            return False

    def deleteDBoldrecords(self):

        """
        Remove old record from the database, notional time of 14 days selected.
        The deletion is made on the criteria of when the record was added to the database, not the event date
        If the event is still listed on the website, then it will be added, and uploaded.

        """

        sql_statement = ""
        sql_statement += "DELETE from event_monitor \n"
        sql_statement += "WHERE                     \n"
        sql_statement += "timeadded < date('now', '-14 day')"

        try:
            cursor = self.db_conn.cursor()
            cursor.execute(sql_statement)
            self.db_conn.commit()

        except:
            log.info("Database purge failed")
            self.delEventMonitorDB()
            self.createEventMonitorDB()
        return None

    def getConnectionToEventMonitorDB(self):
        """ Loads the EventMonitor database

        Arguments:

        Return:
             connection: [connection] A connection to the database
        """

        # Create the EventMonitor database if it does not exist
        if not os.path.isfile(self.event_monitor_db_path):
            self.createEventMonitorDB()


        # Load the EventMonitor database - only gets done here
        try:
            self.conn = sqlite3.connect(self.event_monitor_db_path)
            self.upgradeDB(self.conn)
        except:
            os.unlink(self.event_monitor_db_path)
            self.createEventMonitorDB()

        return self.conn

    def eventExists(self, event):

        """
        Returns True if an event is already in the database. Checks most of the parameters.

        returns:
            exists: [bool]

        """

        sql_statement = ""
        sql_statement += "SELECT COUNT(*) FROM event_monitor \n"
        sql_statement += "WHERE \n"
        sql_statement += "EventTime = '{}'              AND \n".format(event.dt)
        sql_statement += "EventLat = '{}'               AND \n".format(event.lat)
        sql_statement += "EventLon = '{}'               AND \n".format(event.lon)
        sql_statement += "EventHt = '{}'                AND \n".format(event.ht)
        sql_statement += "EventLatStd = '{}'            AND \n".format(event.lat_std)
        sql_statement += "EventLonStd = '{}'            AND \n".format(event.lon_std)
        sql_statement += "EventHtStd = '{}'             AND \n".format(event.ht_std)
        sql_statement += "EventLat2 = '{}'              AND \n".format(event.lat2)
        sql_statement += "EventLon2 = '{}'              AND \n".format(event.lon2)
        sql_statement += "EventHt2 = '{}'               AND \n".format(event.ht2)
        sql_statement += "EventLat2Std = '{}'           AND \n".format(event.lat2_std)
        sql_statement += "EventLon2Std = '{}'           AND \n".format(event.lon2_std)
        sql_statement += "EventHt2Std = '{}'            AND \n".format(event.ht2_std)
        sql_statement += "FarRadius = '{}'              AND \n".format(event.far_radius)
        sql_statement += "CloseRadius = '{}'            AND \n".format(event.close_radius)
        sql_statement += "TimeTolerance = '{}'          AND \n".format(event.time_tolerance)
        sql_statement += "StationsRequired = '{}'       AND \n".format(event.stations_required)
        sql_statement += "RespondTo = '{}'                  \n".format(event.respond_to)

        # does a similar event exist
        # query gets the number of rows matching, not the actual rows

        try:
            return (self.db_conn.cursor().execute(sql_statement).fetchone())[0] != 0
        except:
            log.info("Check for event exists failed")
            if EM_RAISE:
                raise
            return False

    def delOldRecords(self):

        """

        Remove old record from the database, notional time of 14 days selected.
        The deletion is made on the criteria of when the record was added to the database, not the event date
        If the event is still listed on the website, then it will be added, and uploaded.

        """

        sql_statement = ""
        sql_statement += "DELETE from event_monitor \n"
        sql_statement += "WHERE                     \n"
        sql_statement += "timeadded < date('now', '-14 day')"

        try:
            cursor = self.db_conn.cursor()
            cursor.execute(sql_statement)
            self.db_conn.commit()

        except:
            log.info("Database purge failed")
            self.delEventMonitorDB()
            self.createEventMonitorDB()
        return None

    def addEvent(self, event):

        """

        Checks to see if an event exists, if not then add to the database

        Arguments:
            event: [event] Event to be added to the database

        Return:
            added: [bool] True if added, else false

            """


        self.delOldRecords()

        # required for Jessie
        qry_elev_is_max = 1 if event.elev_is_max else 0

        if not self.eventExists(event):
            sql_statement = ""
            sql_statement += "INSERT INTO event_monitor \n"
            sql_statement += "("
            sql_statement += "EventTime, TimeTolerance,                   \n"
            sql_statement += "EventLat ,EventLatStd ,EventLon , EventLonStd , EventHt ,EventHtStd, EventCartStd,     \n"
            sql_statement += "CloseRadius, FarRadius,                     \n"
            sql_statement += "EventLat2, EventLat2Std, EventLon2, EventLon2Std,EventHt2, EventHt2Std, EventCart2Std,    \n"
            sql_statement += "EventAzim, EventAzimStd, EventElev, EventElevStd, EventElevIsMax,    \n"
            sql_statement += "processedstatus, uploadedstatus, uuid, RespondTo, StationsRequired, RequireFR, timeadded \n"
            sql_statement += ")                                           \n"

            sql_statement += "VALUES "
            sql_statement += "(                            \n"
            sql_statement += "'{}',{},                     \n".format(event.dt, event.time_tolerance)
            sql_statement += "{},  {}, {}, {}, {}, {}, {}, \n".format(event.lat, event.lat_std, event.lon, event.lon_std,
                                                                      event.ht, event.ht_std, event.cart_std)
            sql_statement += "{},  {},                     \n".format(event.close_radius, event.far_radius)
            sql_statement += "{},  {}, {}, {}, {}, {}, {}, \n".format(event.lat2, event.lat2_std, event.lon2,
                                                                      event.lon2_std, event.ht2, event.ht2_std, event.cart2_std)
            sql_statement += "{},  {}, {}, {}, {} ,        \n".format(event.azim, event.azim_std, event.elev,
                                                                      event.elev_std,
                                                                      qry_elev_is_max)
            sql_statement += "{},  {}, '{}', '{}', '{}' , '{}', \n".format(0, 0,uuid.uuid4(), event.respond_to, event.stations_required, event.require_FR)
            sql_statement += "CURRENT_TIMESTAMP ) \n"

            try:
                cursor = self.db_conn.cursor()
                cursor.execute(sql_statement)
                self.db_conn.commit()

            except:
                print(sql_statement)
                if EM_RAISE:
                    raise
                log.info("Add event failed")
                self.recoverFromDatabaseError()
                return False
            log.info("Added event at {} to the database".format(event.dt))
            return True
        else:
            return False

    def markEventAsProcessed(self, event):

        """ Marks an event as having been processed

        Arguments:
            event: [event] Event to be marked as processed

        Return:
        """

        sql_statement = ""
        sql_statement += "UPDATE event_monitor                 \n"
        sql_statement += "SET                                  \n"
        sql_statement += "processedstatus = 1,                 \n"
        sql_statement += "timecompleted   = CURRENT_TIMESTAMP  \n".format(datetime.datetime.utcnow())
        sql_statement += "                                     \n"
        sql_statement += "WHERE                                \n"
        sql_statement += "uuid = '{}'                          \n".format(event.uuid)
        try:
            self.db_conn.cursor().execute(sql_statement)
            self.db_conn.commit()
            log.info("Event at {} marked as processed".format(event.dt))
        except:
            log.info("Database error")
            self.recoverFromDatabaseError()

    def eventProcessed(self, uuid):

        """ Return processed status from uuid

        Arguments:
            uuid: [event] Locally generated uuid of the event to be queried

        Return: [bool] True if processed, False if unprocessed or does not exist

        """

        sql_statement = ""
        sql_statement += "SELECT COUNT(processedstatus)                        \n"
        sql_statement += "  from event_monitor                                 \n"
        sql_statement += "    WHERE                                            \n"
        sql_statement += "    processedstatus = 1                              \n"
        sql_statement += "    AND                                              \n"
        sql_statement += "    uuid   = '{}'                                    \n".format(uuid)

        try:
            return (self.db_conn.cursor().execute(sql_statement).fetchone())[0] != 0

        except:
            log.info("Database error - attempting to recreate database")
            self.recoverFromDatabaseError()

    def eventUploaded(self, uuid):

        """ Return uploaded status from uuid

        Arguments:
            uuid: [event] Locally generated uuid of the event to be queried

        Return:
            [bool] True if uploaded, False if not uploaded or does not exist

        """

        sql_statement = ""
        sql_statement += "SELECT COUNT(processedstatus)                        \n"
        sql_statement += "  from event_monitor                                 \n"
        sql_statement += "    WHERE                                            \n"
        sql_statement += "    uploadedstatus = 1                               \n"
        sql_statement += "    AND                                              \n"
        sql_statement += "    uuid   = '{}'                                    \n".format(uuid)

        try:
            return (self.db_conn.cursor().execute(sql_statement).fetchone())[0] != 0

        except:
            log.info("Database error in eventUploaded")
            self.recoverFromDatabaseError()

    def markEventAsUploaded(self, event, file_list):

        """ Mark an event as uploaded in the database

        Arguments:
            event: [event] Event to be marked as uploaded
            file_list: [list of strings] Files uploaded

        Return:
        """

        files_uploaded = ""
        for file in file_list:
            files_uploaded += os.path.basename(file) + " "

        sql_statement = ""
        sql_statement += "UPDATE event_monitor  \n"
        sql_statement += "SET                   \n"
        sql_statement += "filesuploaded  = '{}',\n".format(files_uploaded)
        sql_statement += "uploadedstatus = 1    \n"
        sql_statement += "                      \n"
        sql_statement += "WHERE                 \n"
        sql_statement += "uuid = '{}'           \n".format(event.uuid)

        try:
            cursor = self.db_conn.cursor()
            cursor.execute(sql_statement)
            self.db_conn.commit()
            log.info("Event at {} marked as uploaded".format(event.dt))
        except:
            log.info("Database error")

    def markEventAsReceivedByServer(self, uuid):

        """ Updates table when server publishes UUID of an event which has been sent
            This allows public acknowledgement of a stations transmission to be obfuscated

        rguments:
             uuid: [string] uuid of event received by server

        Return:
             Nothing
        """

        sql_statement = ""
        sql_statement += "UPDATE event_monitor     \n"
        sql_statement += "SET                      \n"
        sql_statement += "receivedbyserver =   '{}'\n".format("1")
        sql_statement += "                         \n"
        sql_statement += "WHERE                    \n"
        sql_statement += "uuid = '{}'              \n".format(uuid)

        cursor = self.db_conn.cursor()
        cursor.execute(sql_statement)
        self.db_conn.commit()

    def getEventsfromWebPage(self, testmode=False):

        """ Reads a webpage, and generates a list of events

            Arguments:

        Return:
            events : [list of events]
        """

        event = EventContainer(0, 0, 0, 0)  # Initialise it empty
        events = []

        if not testmode:
            try:
                if sys.version_info[0] < 3:
                    web_page = urllib2.urlopen(self.syscon.event_monitor_webpage).read().splitlines()
                else:
                    web_page = urllib.request.urlopen(self.syscon.event_monitor_webpage).read().decode("utf-8").splitlines()

            except:
                # Return an empty list
                log.info("EventMonitor found no page at {}".format(self.syscon.event_monitor_webpage))
                return events
        else:
            f = open(os.path.expanduser("~/RMS_data/event_watchlist.txt"), "r")
            web_page = f.read().splitlines()
            f.close()

        for line in web_page:

            line = line.split('#')[0]  # remove anything to the right of comments

            # Protect database against primitive injection techniques

            if ";" in line:
                log.warning("Detected attempt to use ; in database query")
                continue
            if "--" in line:
                log.warning("Detected attempt to use -- in database query")
                continue
            if "=" in line:
                log.warning("Detected attempt to use = in database query")
                continue

            if ":" in line:  # then it is a value pair

                # All this in a try block, in case a type conversion fails
                try:
                    variable_name = line.split(":")[0].strip()  # get variable name
                    value = line.split(":")[1].strip()  # get value
                    event.setValue(variable_name, value)  # and put into this event container
                except:
                    log.error("Unable to read line from webpage...")
                    log.error("{}".format(line))

            else:
                if "END" in line:
                    events.append(copy.copy(event))  # copy, because these are references, not objects
                    event = EventContainer(0, 0, 0, 0)  # Initialise it empty
        #log.info("Read {} events from {}".format(len(events), self.syscon.event_monitor_webpage))

        return events

    def getUnprocessedEventsfromDB(self):

        """ Get the unprocessed events from the database

            Arguments:

            Return:
                events : [list of events]
        """


        sql_statement = ""
        sql_statement += "SELECT "
        sql_statement += ""
        sql_query_cols = ""
        sql_query_cols += "EventTime,TimeTolerance,EventLat,EventLatStd,EventLon, EventLonStd, EventHt, EventHtStd, "
        sql_query_cols += "FarRadius,CloseRadius, uuid,"
        sql_query_cols += "EventLat2, EventLat2Std, EventLon2, EventLon2Std,EventHt2, EventHt2Std, "
        sql_query_cols += "EventAzim, EventAzimStd, EventElev, EventElevStd, EventElevIsMax, RespondTo, StationsRequired,"
        sql_query_cols += "EventCartStd, EventCart2Std, RequireFR"
        sql_statement += sql_query_cols
        sql_statement += " \n"
        sql_statement += "FROM event_monitor "
        sql_statement += "WHERE processedstatus = 0"

        try:
            cursor = self.db_conn.cursor().execute(sql_statement)
        except:
            if EM_RAISE:
                raise
            log.info("Database access error. Delete and recreate.")
            self.delEventMonitorDB()
            self.createEventMonitorDB()
            return[]
        events = []

        # iterate through the rows, one row to an event

        for row in cursor:
            event = EventContainer(0, 0, 0, 0)
            col_count = 0
            cols_list = sql_query_cols.split(',')
            # iterate through the columns, one column to a value
            for col in row:
                event.setValue(cols_list[col_count].strip(), col)
                col_count += 1
                # this is the end of an event
            events.append(copy.copy(event))

        # iterated through all events
        return events

    def getFile(self, file_name, directory):

        """ Get the path to the file in the directory if it exists.
            If not, then return the path to RMS root directory


            Arguments:
                file_name: [string] name of file
                directory: [string] path to preferred directory

            Return:
                 file: [string] Path to platepar
        """

        file_list = []
        if os.path.isfile(os.path.join(directory, file_name)):
            file_list.append(str(os.path.join(directory, file_name)))
            return file_list
        else:

            if os.path.isfile(os.path.join(os.path.expanduser(self.config.config_file_name), file_name)):
                file_list.append(str(os.path.join(os.path.expanduser(self.config.config_file_name), file_name)))
                log.info("Was looking for {}, returning {} as fallback .config file".format(file_name, self.config.config_file_name))
                return file_list
        return []

    def getPlateparFilePath(self, event):

        """ Get the path to the best platepar from the directory matching the event time


            Arguments:
                event: [event]

            Return:
                file: [string] Path to platepar
        """

        platepar_file = ""

        if len(self.getDirectoryList(event)) > 0:
            platepar_file_list = self.getFile(self.syscon.platepar_name, self.getDirectoryList(event)[0])
            if len(platepar_file_list) > 0:
                platepar_file = platepar_file_list[0]
            else:
                platepar_file = os.path.join(self.syscon.rms_root_dir, self.syscon.platepar_name)
                pass

        return platepar_file

    def getDirectoryList(self, event):

        """ Get the paths of directories which may contain files associated with an event

             Arguments:
                 event: [event]

             Return:
                 directorylist: [list of paths] List of directories
        """

        directory_list = []
        event_time = convertGMNTimeToPOSIX(event.dt)

        # iterate across the folders in CapturedFiles and convert the directory time to posix time
        if os.path.exists(os.path.join(os.path.expanduser(self.config.data_dir), self.config.captured_dir)):
            for night_directory in os.listdir(
                    os.path.join(os.path.expanduser(self.config.data_dir), self.config.captured_dir)):
                # Skip over any directory which does not start with the stationID and warn
                if night_directory[0:len(self.config.stationID)] != self.config.stationID:
                    continue
                # Do not add any files, only add directories
                if not os.path.isdir(os.path.join(os.path.expanduser(self.config.data_dir), self.config.captured_dir,night_directory)):
                    continue
                directory_POSIX_time = convertGMNTimeToPOSIX(night_directory[7:22])
                # if the POSIX time representation is before the event, and within 16 hours add to the list of directories
                # most unlikely that a single event could be split across two directories, unless there was large time uncertainty
                if directory_POSIX_time < event_time and (event_time - directory_POSIX_time).total_seconds() < 16 * 3600:
                    directory_list.append(
                        os.path.join(os.path.expanduser(self.config.data_dir), self.config.captured_dir,
                                     night_directory))
        return directory_list



    def findEventFiles(self, event, directory_list, file_extension_list):

        """Take an event, directory list and an extension list and return paths to files

           For .fits files always return at least the closest previous event
           The previous file compared to the event time is held in a variable.
           If the file being compared is the first file after the event time, put the previous file into the list,
           if it is not already there.

           Arguments:
                event: [event] Event of interest
                directory_list: [list of paths] List of directories which may contain the files sought
                file_extension_list: [list of extensions] List of file extensions such as .fits, .bin

           Return:
                file_list: [list of paths] List of paths to files
        """
        
        try:
            event_time = dateutil.parser.parse(event.dt)

        except:
            event_time = convertGMNTimeToPOSIX(event.dt)

        seeking_first_fits_after_event = True # to prevent warning of possibly uninitialised variable
        file_list = []
        # Iterate through the directory list, appending files with the correct extension


        last_fits_file = None
        for directory in directory_list:
            for file_extension in file_extension_list:
                # get the directory into name order
                dirlist = os.listdir(directory)
                dirlist.sort()
                if file_extension == ".fits":
                    fits_list = glob.glob(os.path.join(directory,"*.fits"))
                    fits_list.sort()

                    if len(fits_list) == 0:
                        # If fits_list is empty then return an empty list
                        log.info("No fits files in {}".format(directory))
                        return file_list
                    else:
                        # Initialise last_fits_file with the first from the list
                        last_fits_file = fits_list[0]

                        seeking_first_fits_after_event = True
                for file in dirlist:
                    if file.endswith(file_extension):
                        file_POSIX_time = convertGMNTimeToPOSIX(file[10:25])
                        if abs((file_POSIX_time - event_time).total_seconds()) < event.time_tolerance:
                            file_list.append(os.path.join(directory, file))

                        if file_extension == ".fits":
                        # if this is the first fits file after the event time, add the previous fits file
                        # unless already in the list
                            if file_POSIX_time > event_time and seeking_first_fits_after_event:
                                if last_fits_file not in file_list:
                                    file_list.append(os.path.join(directory, last_fits_file))
                                    seeking_first_fits_after_event = False
                            last_fits_file = file
        return file_list

    def getFileList(self, event):

        """Take an event, return paths to files

           Arguments:
               event: [event] Event of interest


           Return:
               file_list: [list of paths] List of paths to files
        """

        file_list = []

        file_list += self.findEventFiles(event, self.getDirectoryList(event), [".fits", ".bin"])
        #have to use system .config file_name here because we have not yet identified the files for the event
        #log.info("Using {} as .config file name".format(self.syscon.config_file_name))
        if len(self.getDirectoryList(event)) > 0:
            file_list += self.getFile(os.path.basename(self.syscon.config_file_name), self.getDirectoryList(event)[0])
            file_list += [self.getPlateparFilePath(event)]
            #log.info("File list {}".format(file_list))
        return file_list

    def trajectoryVisible(self, rp, event):

        """
        Given a platepar and an event, calculate the centiles of the trajectory which would be in the FoV.
        Working is in ECI, relative to the station coordinates.

        Args:
            rp: [platepar] reference platepar
            event: [event] event of interest

        Returns:
            points_in_fov: [integer] the number of points out of 100 in the field of view
            start_distance: [float] the distance in metres from the station to the trajectory start
            start_angle: [float] the angle between the vector from the station to start of the trajectory
                        and the vector of the centre of the FOV
            end_distance: [float] the distance in metres from the station to the trajectort end
            end_angle: [float] the angle between the vector from the station to end of the trajectory
                        and the vector of the centre of the FOV
            fov_ra: [float]  field of view Ra (degrees)
            fov_dec: [float] fov_dec of view Dec (degrees)

        """
        # Calculate diagonal FoV of camera
        diagonal_fov = np.sqrt(rp.fov_v ** 2 + rp.fov_h ** 2)

        # Calculation origin will be the ECI of the station taken from the platepar
        jul_date = datetime2JD(convertGMNTimeToPOSIX(event.dt))
        origin = np.array(geo2Cartesian(rp.lat, rp.lon, rp.elev, jul_date))

        # Convert trajectory start and end point coordinates to cartesian ECI at JD of event
        traj_sta_pt = np.array(geo2Cartesian(event.lat, event.lon, event.ht * 1000, jul_date))
        traj_end_pt = np.array(geo2Cartesian(event.lat2, event.lon2, event.ht2 * 1000, jul_date))

        # Make relative (_rel) to station coordinates
        stapt_rel, endpt_rel = traj_sta_pt - origin, traj_end_pt - origin

        # trajectory vector, and vector for traverse
        traj_vec = traj_end_pt - traj_sta_pt
        traj_inc = traj_vec / 100

        # the az_centre, alt_centre of the camera
        az_centre, alt_centre = platepar2AltAz(rp)

        # calculate Field of View RA and Dec at event time, and
        fov_ra, fov_dec = altAz2RADec(az_centre, alt_centre, jul_date, rp.lat, rp.lon)

        fov_vec = np.array(raDec2Vector(fov_ra, fov_dec))

        # iterate along the trajectory counting points in the field of view
        points_in_fov = 0
        for i in range(0, 100):
            point = (stapt_rel + i * traj_inc)
            point_fov = angularSeparationVectDeg(vectNorm(point), vectNorm(fov_vec))
            if point_fov < diagonal_fov / 2:
                points_in_fov += 1

        # calculate some additional information for confidence
        start_distance = (np.sqrt(np.sum(stapt_rel ** 2)))
        start_angle = angularSeparationVectDeg(vectNorm(stapt_rel), vectNorm(fov_vec))
        end_distance = (np.sqrt(np.sum(endpt_rel ** 2)))
        end_angle = angularSeparationVectDeg(vectNorm(endpt_rel), vectNorm(fov_vec))

        return points_in_fov, start_distance, start_angle, end_distance, end_angle, fov_ra, fov_dec

    def trajectoryThroughFOV(self, event):

        """
        For the trajectory contained in the event, calculate if it passed through the FoV defined by the
        of the time of the event

        Args:
            event: [event] Calculate if the trajectory of this event passed through the field of view

        Returns:
            pts_in_FOV: [integer] Number of points of the trajectory split into 100 parts
                                   apparently in the FOV of the camera
            sta_dist: [float] Distance from station to the start of the trajectory
            sta_ang: [float] Angle from the centre of the FoV to the start of the trajectory
            end_dist: [float] Distance from station to the end of the trajectory
            end_ang: [float] Angle from the centre of the FoV to the end of the trajectory
        """

        # Read in the platepar for the event
        rp = Platepar()
        if self.getPlateparFilePath(event) == "":
            rp.read(os.path.abspath('.'))
        else:
            rp.read(self.getPlateparFilePath(event))

        pts_in_FOV, sta_dist, sta_ang, end_dist, end_ang, fov_RA, fov_DEC = self.trajectoryVisible(rp, event)
        return pts_in_FOV, sta_dist, sta_ang, end_dist, end_ang, fov_RA, fov_DEC

    def doUpload(self, event, evcon, file_list, keep_files=False, no_upload=False, test_mode=False):

        """Move all the files to a single directory. Make MP4s, stacks and jpgs
           Archive into a bz2 file and upload, using paramiko. Delete all working folders.

        Args:
            event: [event] the event to be uploaded
            evcon: [path] path to the config file for the event
            file_list: [list of paths] the files to be uploaded
            keep_files: [bool] keep the files after upload
            no_upload: [bool] if True do everything apart from uploading
            test_mode: [bool] if True prevents upload

        Returns:
            uploadstatus: [bool] status of upload

        """

        if self.eventUploaded(event.uuid):
            log.warning("Call to doUpload for already uploaded event {}".format(event.dt))

        if self.eventProcessed(event.uuid):
            log.warning("Call to doUpload for already processed event {}".format(event.dt))

        event_monitor_directory = os.path.expanduser(os.path.join(self.syscon.data_dir, "EventMonitor"))
        upload_filename = "{}_{}_{}".format(evcon.stationID, event.dt, sanitise(event.suffix))
        # Try and bake the camera network name and group name into the path structure of the archive
        if evcon.network_name is not None and evcon.camera_group_name is not None:
            #create path for this_event_directory
            #get rid of spaces from network name and group name
            this_event_directory = os.path.join(event_monitor_directory,
                                                    upload_filename,
                                                        sanitise(evcon.network_name),
                                                            sanitise(evcon.camera_group_name),
                                                                sanitise(evcon.stationID))

            log.info("Network {} and group {} so creating {}"
                                .format(sanitise(evcon.network_name),sanitise(evcon.camera_group_name), this_event_directory))
        else:
            this_event_directory = os.path.join(event_monitor_directory, upload_filename, sanitise(evcon.stationID))
            log.info("Network and group not defined so creating {}".format(this_event_directory))

        # get rid of the eventdirectory, should never be needed
        if not keep_files:
            if os.path.exists(this_event_directory) and event_monitor_directory != "" and upload_filename != "":
                shutil.rmtree(this_event_directory)


        # create a new event directory
        if not os.path.exists(this_event_directory):
            os.makedirs(this_event_directory)

        # put all the files from the filelist into the event directory
        pack_size = 0
        for file in file_list:
            pack_size += os.path.getsize(file)
        log.info("File pack ({:.0f}MB) assembly started".format(pack_size/1024/1024))

        # Don't upload things which are too large
        if pack_size > 1000*1024*1024:
            log.error("File pack too large")
            return False

        for file in file_list:
            shutil.copy(file, this_event_directory)
        log.info("File pack assembled")

        stackFFs(this_event_directory, "jpg", captured_stack=True, print_progress=False)

        # convert bins to MP4
        for file in file_list:
            #Guard against FS files getting into binViewer
            if file.endswith(".bin") and sys.version_info[0] >= 3 and os.path.basename(file)[0:2] != "FS":
                fr_file = os.path.basename(file)
                ff_file = convertFRNameToFF(fr_file)

                try:
                    log.info("this_event_directory {}".format(this_event_directory))
                    log.info("ff_file {}, fr_file {}".format(ff_file, fr_file))
                    view(this_event_directory, ff_file, fr_file, self.syscon, hide=True, add_timestamp=True, extract_format="mp4")
                except:
                    log.error("Converting {} to mp4 failed".format(file))
                    log.error("this_event_directory {}".format(this_event_directory))
                    log.error("convertFRNameToFF {}".format(ff_file))
                    log.error("fr_file {}".format(fr_file))

        if True:
            image_note = event.suffix
            batchFFtoImage(os.path.join(this_event_directory), "jpg", add_timestamp=True,
                           ff_component='maxpixel')

        with open(os.path.join(this_event_directory, "event_report.txt"), "w") as info:
            info.write(event.eventToString())

        # remove any leftover .bz2 files
        if not keep_files:
            if os.path.isfile(os.path.join(event_monitor_directory, "{}.tar.bz2".format(upload_filename))):
                os.remove(os.path.join(event_monitor_directory, "{}.tar.bz2".format(upload_filename)))

        if not test_mode:
            if os.path.isdir(event_monitor_directory) and upload_filename != "":
             log.info("Making archive of {}".format(os.path.join(event_monitor_directory, upload_filename)))
             base_name = os.path.join(event_monitor_directory,upload_filename)
             root_dir = os.path.join(event_monitor_directory,upload_filename)
             archive_name = shutil.make_archive(base_name, 'bztar', root_dir, logger=log)
            else:
             log.info("Not making an archive of {}, not sensible.".format(os.path.join(event_monitor_directory)))

        # Remove the directory where the files were assembled
        if not keep_files:
            if os.path.exists(this_event_directory) and this_event_directory != "":
                shutil.rmtree(this_event_directory)

        # Set the upload status to false - every path through the code will set this.
        upload_status = False

        if not no_upload and not test_mode:
            # Loop round for a maximum of 30 tries to carry out the upload
            # Progressively lengthen the delay time, with some random element
            # Return the status of the upload
            # Don't include a delay before uploading
            log.info("Upload of {} - first attempt".format(event_monitor_directory))
            for retry in range(1,30):
                archives = glob.glob(os.path.join(event_monitor_directory,"*.bz2"))

                # Make the upload



                upload_status = uploadSFTP(self.syscon.hostname, self.syscon.stationID.lower(),
                                 event_monitor_directory,self.syscon.event_monitor_remote_dir,archives,
                                 rsa_private_key=self.config.rsa_private_key, allow_dir_creation=True)


                if upload_status:
                    log.info("Upload of {} - attempt no {} was successful".format(event_monitor_directory, retry))
                    # set to the fast check rate after an upload,
                    # unless already set to run faster than that, possibly for future event reporting
                    self.check_interval = self.syscon.event_monitor_check_interval_fast if self.check_interval > self.syscon.event_monitor_check_interval_fast else self.check_interval
                    log.info("Now checking at {:.1f} minute intervals".format(self.check_interval))
                    # Exit loop if upload was successful
                    break
                else:
                    retry_delay = (retry * 180 * (1+ random.random()))
                    log.error("Upload failed on attempt {}. Retry after {:.1f} seconds.".format(retry, retry_delay))
                    time.sleep(retry_delay)
                    log.info("Retrying upload of {}. This is retry {}".format(event_monitor_directory, retry))
        else:
            upload_status = False

        # Remove the directory - even if the upload was not successful
        if not keep_files:
            shutil.rmtree(event_monitor_directory)
        return upload_status

    def frFileInList(self, file_list):

        found = False
        for file_to_check in file_list:
            if os.path.basename(file_to_check)[0:2] == "FR" and file_to_check.endswith('bin'):
                found = True

        return found

    def checkEvents(self, ev_con, test_mode = False):

        """
        argunments:
            ev_con: configuration object at the time of this event

        returns:
            Nothing
        """

        # Get the work to be done

        unprocessed = self.getUnprocessedEventsfromDB()

        future_events = 0
        for observed_event in unprocessed:

            # check to see if the end of this event is in the future, if it is then do not process
            # if the end of the event is before the next scheduled execution of event monitor loop,
            # then set the loop to execute after the event ends
            if convertGMNTimeToPOSIX(observed_event.dt) + \
                    datetime.timedelta(seconds=int(observed_event.time_tolerance)) > datetime.datetime.utcnow():
                time_until_event_end_seconds = (convertGMNTimeToPOSIX(observed_event.dt) -
                                                    datetime.datetime.utcnow() +
                                                    datetime.timedelta(seconds=int(observed_event.time_tolerance))).total_seconds()
                future_events += 1
                log.info("The end of event at {} is in the future by {:.1f} minutes"
                         .format(observed_event.dt, time_until_event_end_seconds / 60))
                if time_until_event_end_seconds < float(self.check_interval) * 60:
                    log.info("Check interval is set to {:.1f} minutes, however end of future event is only {:.1f} minutes away"
                             .format(float(self.check_interval),time_until_event_end_seconds / 60))
                    # set the check_interval to the time until the end of the event
                    self.check_interval = float(time_until_event_end_seconds) / 60
                    # random time offset to reduce congestion
                    self.check_interval += random.randint(20, 60) / 60
                    log.info("Check interval set to {:.1f} minutes, so that future event is reported quickly"
                             .format(float(self.check_interval)))
                else:
                    log.info("Check interval is set to {:.1f} minutes, end of future event {:.1f} minutes away, no action required"
                             .format(float(self.check_interval),time_until_event_end_seconds / 60 ))
                continue


            log.info("Checks on trajectories for event at {}".format(observed_event.dt))
            check_time_start = datetime.datetime.utcnow()
            # Iterate through the work
            # Events can be specified in different ways, make sure converted to LatLon
            observed_event.latLonAzElToLatLonLatLon()
            # Get the files
            file_list = self.getFileList(observed_event)

            # If there are no files based on time, then mark as processed and continue
            if (len(file_list) == 0 or file_list == [None]) and not test_mode:
                log.info("No files for event - marking {} as processed".format(observed_event.dt))
                self.markEventAsProcessed(observed_event)
                # This moves to next observed_event
                continue

            # move to the next event if we required an FR file but do not have one

            if observed_event.require_FR == 1:
                log.info("Event at {} requires FR file".format(observed_event.dt))
                if not self.frFileInList(file_list):
                    log.info("Event at {} skipped - FR required and none found".format(observed_event.dt))
                    self.markEventAsProcessed(observed_event)
                    continue
                else:
                    log.info("Event at {} required FR file and file was found".format(observed_event.dt))
            else:
                log.info("FR file not required for event at {}".format(observed_event.dt))

            # If there is a .config file then parse it as evcon - not the station config
            for file in file_list:
                if file.endswith(".config"):
                    ev_con = cr.parse(file)

            # Look for the station code in the stations_required string
            if observed_event.stations_required.find(ev_con.stationID) != -1:
                if self.doUpload(observed_event, ev_con, file_list, test_mode):
                    log.info("In Stations_Required - marking {} as processed".format(observed_event.dt))
                    self.markEventAsProcessed(observed_event)
                    if len(file_list) > 0:
                        self.markEventAsUploaded(observed_event, file_list)
                else:
                    log.error("Upload failed for event at {}. Event retained in database for retry.".format(observed_event.dt))
                continue

            # Initialise the population of trajectories
            event_population = []
            # If we have any standard deviation definitions then create a population of 1000, else create a population of 1
            if observed_event.hasCartSD() or observed_event.hasPolarSD():
                log.info("Working with standard deviations")
                event_population = observed_event.appendPopulation(event_population,1000)
            else:
                log.info("Working without standard deviations")
                event_population = observed_event.appendPopulation(event_population,1)



            # Apply SD to the population
            if observed_event.hasCartSD():
                log.info("Applying cartesian standard deviations")
                event_population = observed_event.applyCartesianSD(event_population)
            if observed_event.hasPolarSD():
                log.info("Applying polar standard deviations")
                event_population = observed_event.applyPolarSD(event_population)

            # Add trajectories with elevations from observed value to 15 deg
            if observed_event.elev_is_max:
                log.info("Rotating trajectory around observed point")
                event_population = observed_event.addElevationRange(event_population, observed_event, 15)

            # Start testing trajectories from the population
            for event in event_population:
                # check if this has already been handled
                if self.eventProcessed(observed_event.uuid):
                    break # do no more work on any version of this trajectory - break exits loop
                # From the infinitely extended trajectory, work out the closest point to the camera
                # ev_con.elevation is the height above sea level of the station in metres, no conversion required
                start_dist, end_dist, atmos_dist = calculateClosestPoint(event.lat, event.lon, event.ht * 1000,
                                                                              event.lat2, event.lon2, event.ht2 * 1000,
                                                                              ev_con.latitude, ev_con.longitude, ev_con.elevation)
                min_dist = min([start_dist, end_dist, atmos_dist])

                # If this version of the trajectory outside the farradius, continue
                if min_dist > event.far_radius * 1000 and not test_mode:
                    # Do no more work on this version of the trajectory
                    continue

            # If trajectory inside the closeradius, then do the upload and mark as processed
                if min_dist < event.close_radius * 1000 and not test_mode:
                    # this is just for info
                    log.info("Event at {} was {:.0f}km away, inside {:.0f}km so is uploaded with no further checks.".format(event.dt, min_dist / 1000, event.close_radius))
                    check_time_end = datetime.datetime.utcnow()
                    check_time_seconds = (check_time_end- check_time_start).total_seconds()
                    log.info("Check of trajectories time elapsed {:.2f} seconds".format(check_time_seconds))
                    count, event.start_distance, event.start_angle, event.end_distance, event.end_angle, event.fovra, event.fovdec = self.trajectoryThroughFOV(
                        event)
                    # If doUpload returned True mark the event as processed and uploaded
                    if self.doUpload(event, ev_con, file_list, test_mode):
                        log.info("Inside close radius - marking {} as processed".format(observed_event.dt))
                        self.markEventAsProcessed(observed_event)
                        if len(file_list) > 0:
                            self.markEventAsUploaded(observed_event, file_list)
                        break # Do no more work on any version of this trajectory - break exits loop
                    else:
                        log.error("Upload failed for event at {}. Event retained in database for retry.".format(event.dt))

            # If trajectory inside the farradius, then check if the trajectory went through the FoV
            # The returned count is the number of 100th parts of the trajectory observed through the FoV
                if min_dist < event.far_radius * 1000 or test_mode:
                    #log.info("Event at {} was {:4.1f}km away, inside {:4.1f}km, consider FOV.".format(event.dt, min_dist / 1000, event.far_radius))
                    count, event.start_distance, event.start_angle, event.end_distance, event.end_angle, event.fovra, event.fovdec = self.trajectoryThroughFOV(event)
                    if count != 0:
                        log.info("Event at {} had {} points out of 100 in the trajectory in the FOV. Uploading.".format(event.dt, count))
                        check_time_end = datetime.datetime.utcnow()
                        check_time_seconds = (check_time_end - check_time_start).total_seconds()
                        log.info("Check of trajectories took {:2f} seconds".format(check_time_seconds))
                        if self.doUpload(observed_event, ev_con, file_list, test_mode=test_mode):
                            self.markEventAsUploaded(observed_event, file_list)
                            if not test_mode:
                                log.info("Trajectory passed through FoV - marking {} as processed".format(observed_event.dt))
                                self.markEventAsProcessed(observed_event)
                            break # Do no more work on any version of this trajectory
                        else:
                            log.error("Upload failed for event at {}. Event retained in database for retry.".format(event.dt))
                        if test_mode:
                            rp = Platepar()
                            rp.read(self.getPlateparFilePath(event))
                            with open(os.path.expanduser(os.path.join(self.syscon.data_dir, "testlog")), 'at') as logfile:
                                logfile.write(
                                    "{} LOC {} Az:{:3.1f} El:{:3.1f} sta_lat:{:3.4f} sta_lon:{:3.4f} sta_dist:{:3.0f} end_dist:{:3.0f} fov_h:{:3.1f} fov_v:{:3.1f} sa:{:3.1f} ea::{:3.1f} \n".format(
                                    convertGMNTimeToPOSIX(event.dt), ev_con.stationID, rp.az_centre, rp.alt_centre,
                                    rp.lat, rp.lon, event.start_distance / 1000, event.end_distance / 1000, rp.fov_h,
                                    rp.fov_v, event.start_angle, event.end_angle))
                    else:

                        if not test_mode:
                            pass

                    # Continue with other trajectories from this population
                    continue

            # End of the processing loop for this event
            if self.eventProcessed(observed_event.uuid):
                log.info("Reached end of checks - {} is processed".format(observed_event.dt))
                check_time_end = datetime.datetime.utcnow()
                check_time_seconds = (check_time_end - check_time_start).total_seconds()
                log.info("Check of trajectories time elapsed {:.2f} seconds".format(check_time_seconds))

            else:
                check_time_end = datetime.datetime.utcnow()
                check_time_seconds = (check_time_end - check_time_start).total_seconds()
                log.info("Reached end of checks - {} is processed, nothing to upload".format(observed_event.dt))
                log.info("Check of trajectories time elapsed {:.2f} seconds".format(check_time_seconds))
                self.markEventAsProcessed(observed_event)

        if len(unprocessed) - future_events > 1:
            log.info("{} events were processed, EventMonitor work completed"
                     .format(len(unprocessed) - future_events))
        if len(unprocessed) - future_events == 1:
            log.info("{} event was processed, EventMonitor work completed"
                     .format(len(unprocessed) - future_events))

        next_run = (datetime.datetime.utcnow() + datetime.timedelta(minutes=self.check_interval)).replace(microsecond = 0)
        if future_events == 1:
            log.info("{} future event is scheduled, running again at {}"
                     .format(future_events, next_run))
        if future_events > 1:
            log.info("{} future events are scheduled, running again at {}"
                     .format(future_events, next_run))

        return None

    def start(self):
        """ Starts the EventMonitor """

        if testIndividuals(logging = False):
            log.info("EventMonitor function test success")
            super(EventMonitor, self).start()
            log.info("EventMonitor was started")
            log.info("Using {} as fallback directory".format(os.path.join(os.path.abspath("."))))
            log.info("Using {} as config filename".format(self.syscon.config_file_name))
            log.info("Using {} as platepar filename".format(self.syscon.platepar_name))
        else:
            log.error("EventMonitor function test fail - not starting EventMonitor")

    def stop(self):
        """ Stops the EventMonitor. """

        self.db_conn.close()
        time.sleep(2)
        self.exit.set()
        self.join()
        log.info("EventMonitor has stopped")

    def checkDBExists(self):

        """
        Check that the database file exists
        """

        if not os.path.exists(self.event_monitor_db_path):
            self.conn = self.createEventMonitorDB()

        return True

    def getEventsAndCheck(self, start_time, end_time, testmode=False):
        """
        Gets event(s) from the webpage, or a local file.
        Calls self.addevent to add them to the database
        Calls self.checkevents to see if the database holds any unprocessed events

        Args:
            start_time: time to start checking from
            end_time: time to start checking to
            testmode: [bool] if set true looks for a local file, rather than a web address

        Returns:
            Nothing
        """

        events = self.getEventsfromWebPage(testmode)
        # Don't try to iterate over None - this check should never be needed
        if events is None:
            log.warning("Attempt to iterate over None")
            return
        for event in events:
            if event.isReasonable():
                self.addEvent(event)

        # Go through all events and check if they need to be uploaded - this iterates through the database
        self.checkEvents(self.config, test_mode=testmode)

    def run(self):

        """
        Call to start the EventMonitor loop. If the loop has been accelerated following a match
        then this loop slows it down by multiplying the check interval by 1.1.

        The time between checks is the sum of the delay interval, and the time to perform the check and upload.
        No further randomisation is applied, as this is a congestion, not contention problem.

        """

        # Delay to allow capture to check existing folders - keep the logs tidy


        time.sleep(60)
        last_check_start_time = datetime.datetime.utcnow()
        while not self.exit.is_set():
            check_start_time = datetime.datetime.utcnow()
            next_check_start_time = (datetime.datetime.utcnow() + datetime.timedelta(minutes=self.check_interval))
            next_check_start_time_str = next_check_start_time.replace(microsecond=0).strftime('%H:%M:%S')
            self.checkDBExists()
            self.getEventsAndCheck(last_check_start_time,next_check_start_time)
            last_check_start_time = check_start_time

            start_time, duration = captureDuration(self.syscon.latitude, self.syscon.longitude, self.syscon.elevation)

            if not isinstance(start_time, bool):

                time_left_before_start = (start_time - datetime.datetime.utcnow())
                time_left_before_start = time_left_before_start - datetime.timedelta(microseconds=time_left_before_start.microseconds)
                time_left_before_start_minutes = int(time_left_before_start.total_seconds() / 60)
                next_check_start_time = (datetime.datetime.utcnow() + datetime.timedelta(minutes=self.check_interval))
                next_check_start_time_str = next_check_start_time.replace(microsecond=0).strftime('%H:%M:%S')
                log.info('Next EventMonitor run : {} UTC; {:3.1f} minutes from now'.format(next_check_start_time_str, int(self.check_interval)))
                if time_left_before_start_minutes < 120:
                    log.info('Next Capture start    : {} UTC; {:3.1f} minutes from now'.format(str(start_time.strftime('%H:%M:%S')),time_left_before_start_minutes))
                else:
                    log.info('Next Capture start    : {} UTC'.format(str(start_time.strftime('%H:%M:%S'))))
            else:
                next_check_start_time = (datetime.datetime.utcnow() + datetime.timedelta(minutes=self.check_interval))
                next_check_start_time_str = next_check_start_time.replace(microsecond=0).strftime('%H:%M:%S')
                log.info('Next EventMonitor run : {} UTC {:3.1f} minutes from now'.format(next_check_start_time_str, self.check_interval))
            # Wait for the next check
            self.exit.wait(60 * self.check_interval)
            # Increase the check interval
            if self.check_interval < self.syscon.event_monitor_check_interval:
                self.check_interval = self.check_interval * 1.1

def latLonAlt2ECEFDeg(lat, lon, h):
    """ Convert geographical coordinates to Earth centered - Earth fixed coordinates.

    Arguments:
        lat: [float] latitude in degrees (+north)
        lon: [float] longitude in degrees (+east)
        h: [float] elevation in metres (WGS84)

    Return:
        (x, y, z): [tuple of floats] ECEF coordinates

    """

    # Call library function, after converting to radians
    return latLonAlt2ECEF(np.radians(lat), np.radians(lon), h)

def angularSeparationVectDeg(vect1, vect2):
    """ Calculates angle between vectors in radians.
        Uses library function, in radions , but converts return to degrees
        This function is to reduce inline conversion calls """

    return np.degrees(angularSeparationVect(vect1,vect2))

def calculateClosestPoint(beg_lat, beg_lon, beg_ele, end_lat, end_lon, end_ele, ref_lat, ref_lon, ref_ele):

        """
        Calculate the closest approach of a trajectory to a reference point

        refer to https://globalmeteornetwork.groups.io/g/main/topic/96374390#8250


        Args:
            beg_lat: [float] Starting latitude of the trajectory
            beg_lon: [float] Starting longitude of the trajectory
            beg_ele: [float] Beginning height of the trajectory
            end_lat: [float] Ending latitude of the trajectory
            end_lon: [float] Ending longitude of the trajectory
            end_ele: [float] Ending height of the trajectory
            ref_lat: [float] Station latitude
            ref_lon: [float] Station longitude
            ref_ele: [float] Station height

        Returns:
            start_dis: Distance from station to start of trajectory
            end_dist: Distance from station to end of trajectory
            closest_dist: Distance at the closest point (possibly outside the start and end)

        """

        # Convert coordinates to ECEF
        beg_ecef = np.array(latLonAlt2ECEFDeg(beg_lat, beg_lon, beg_ele))
        end_ecef = np.array(latLonAlt2ECEFDeg(end_lat, end_lon, end_ele))
        ref_ecef = np.array(latLonAlt2ECEFDeg(ref_lat, ref_lon, ref_ele))

        traj_vec = vectNorm(end_ecef - beg_ecef)
        start_vec, end_vec = (ref_ecef - beg_ecef), (ref_ecef - end_ecef)
        start_dist, end_dist = (np.sqrt((np.sum(start_vec ** 2)))), (np.sqrt((np.sum(end_vec ** 2))))

        # Consider whether vector is zero length by looking at start and end
        if [beg_lat, beg_lon, beg_ele] != [end_lat, end_lon, end_ele]:

            # Vector start and end points are different, calculate the projection of the ref vect onto the traj vector
            proj_vec = beg_ecef + np.dot(start_vec, traj_vec) * traj_vec

            # Hence, calculate the vector at the nearest point, and the closest distance
            closest_vec = ref_ecef - proj_vec
            closest_dist = (np.sqrt(np.sum(closest_vec ** 2)))

        else:

            # Vector has zero length, do not try to calculate projection
            closest_dist = start_dist

        return start_dist, end_dist, closest_dist

def revAz(azim):

    """
        Reverse an azimuth by normalising and reversing

        arguments:
            azim: [float] azimuth in degrees
        returns:
            azim_rev: [float] azimuth in the reverse direction in degrees

        """

    azim_nrm = azim % 360
    azim_rev = azim_nrm + 180 if azim_nrm < 180 else azim_nrm - 180
    return azim_rev

def ecefV2LatLonAlt(ecef_vect):
    """
        Convert ECEF vector (meters) to lat(deg),lon(deg),ht(km)

        arguments: ecef_vector

        returns: lat,lon,ht

        """

    lat, lon, ht = ecef2LatLonAlt(ecef_vect[0], ecef_vect[1], ecef_vect[2])
    return np.degrees(lat), np.degrees(lon), ht / 1000

def randomword(length):

    """
    https://stackoverflow.com/questions/2030053/how-to-generate-random-strings-in-python
    
    Generates a random string of letters    
    
    """


    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def platepar2AltAz(rp):

    """

    arguments:
        rp: Platepar

    returns:
        Ra_d : [degrees] Ra of the platepar at its creation date
        dec_d : [degrees] Dec of the platepar at its creation date
        JD : [float] JD of the platepar creation
        lat : [float] lat of the station
        lon : [float] lon of the station

    """

    RA_d = np.radians(rp.RA_d)
    dec_d = np.radians(rp.dec_d)
    JD = rp.JD
    lat = np.radians(rp.lat)
    lon = np.radians(rp.lon)

    return np.degrees(cyTrueRaDec2ApparentAltAz(RA_d, dec_d, JD, lat, lon))

def angDif(a1, a2):

    """
    Get the minimum difference between two angles, always positive

    arguments:
        a1:[float] angle(degrees)
        a2:[float] angle(degrees)

    return:
        [float] difference between angles


    """

    normalised = abs(a1-a2) % 360
    return min(360-normalised, normalised)

def convertGMNTimeToPOSIX(timestring):

    """
    Converts the filenaming time convention used by GMN into posix

    arguments:
        timestring: [string] time represented as a string e.g. 20230527_032115

    returns:
        posix compatible time
    """
    try:
        dt_object = datetime.datetime.strptime(timestring.strip(), "%Y%m%d_%H%M%S")
    except:
        log.error("Badly formatted time {}".format(timestring))
        # return a time which will be safe but cannot produce any output
        dt_object = datetime.datetime.strptime("20000101_000000".strip(), "%Y%m%d_%H%M%S")
    return dt_object

def createATestEvent07():


    """
    Creates an event for testing

    arguments:

    returns:
        event: [event]

    """

    test_event = EventContainer("", 0, 0, 0)
    test_event.setValue("EventTime", "20230526_205441")
    test_event.setValue("TimeTolerance", 60)
    test_event.setValue("EventLat", -32.263726)
    test_event.setValue("EventLatStd", 0.31)
    test_event.setValue("EventLon", 116.016066)
    test_event.setValue("EventLonStd", 0.32)
    test_event.setValue("EventHt", 89.0537)
    test_event.setValue("EventHtStd", 15)
    test_event.setValue("CloseRadius", 152)
    test_event.setValue("FarRadius", 153)

    test_event.setValue("EventLat2", -32.187818)
    test_event.setValue("EventLat2Std", 0.33)
    test_event.setValue("EventLon2", 116.111370)
    test_event.setValue("EventLon2Std", 0.34)
    test_event.setValue("EventHt2", 80.8778)
    test_event.setValue("EventHt2Std", 0.35)

    test_uuid = "28e4a2d7-4111-4a72-8a30-969f71fc9207"
    test_event.setValue("uuid", test_uuid)

    return test_event

def createATestEvent08():

    """
    Creates an event for testing

    arguments:

    returns:
        event: [event]

    """

    test_event = EventContainer("", 0, 0, 0)
    test_event.setValue("EventTime", "20230601_124235")
    test_event.setValue("TimeTolerance", 60)
    test_event.setValue("EventLat", 45)
    test_event.setValue("EventLatStd", 0)
    test_event.setValue("EventLon", 179)
    test_event.setValue("EventLonStd", 0)
    test_event.setValue("EventHt", 100)
    test_event.setValue("EventHtStd", 0)
    test_event.setValue("CloseRadius", 152)
    test_event.setValue("FarRadius", 153)

    test_event.setValue("EventLat2", 0)
    test_event.setValue("EventLat2Std", 0)
    test_event.setValue("EventLon2", 0)
    test_event.setValue("EventLon2Std", 0)
    test_event.setValue("EventHt2", 0)
    test_event.setValue("EventHt2Std", 0)

    test_event.setValue("EventAzim", 0)
    test_event.setValue("EventElev", 0)



    test_uuid = "28e4a2d7-4111-4a72-8a30-969f71fc9207"
    test_event.setValue("uuid", test_uuid)

    return test_event

def gcDistDeg(lat1, lon1, lat2, lon2):

    """
    Uses Haversine formula to return great circle distance. This function is only used for testing other
    functions

    arguments:
        lat1: [float] latitude of point 1 (degrees)
        lon1: [float] latitude of point 1 (degrees)
        lat2: [float] latitude of point 2 (degrees)
        lon2: [float] latitude of point 2 (degrees)

    returns:
        [float] great circle distance (km)


    """

    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    delta_lat, delta_lon = (lat2 - lat1)/2 , (lon2 - lon1)/2

    t1 = np.sin(delta_lat) ** 2
    t2 = np.sin(delta_lon) ** 2 * np.cos(lat1) * np.cos(lat2)

    if (abs(t1) - abs(t2)) < 1e-10:
        return 0
    else:
        return 2 * np.arcsin((t1 + t2) ** 0.5) * 6371.009

def testRevAz():

    """
    Generates random angles and confirms that the angle difference to reversed is 180 degrees


    return:
        [bool]

    """
    t = EventContainer("",0,0,0)
    success = True

    for test in range(3000):
        test_azim = random.uniform(-5000,5000)
        success = success if angDif(test_azim, revAz(test_azim)) == 180 else False
        if not success:
            return False


    return success

def testIsReasonable():
    """
    Checks isReasonble function by generating events


    return:
        [bool]

    """

    t = EventContainer("",0,0,0)
    t.setValue("EventLat", "45")
    t.setValue("EventLon", "-27")
    t.setValue("TimeTolerance", "5")

    success = True
    success = success if t.isReasonable() else False
    t.setValue("TimeTolerance", "301")
    success = success if not t.isReasonable() else False
    t.setValue("TimeTolerance", "299")
    success = success if t.isReasonable() else False
    t.lat = ""
    success = success if not t.isReasonable() else False
    t.setValue("EventLat", "-180")
    success = success if t.isReasonable() else False
    t.lon = ""
    success = success if not t.isReasonable() else False
    t.setValue("EventLon", "-32")
    success = success if t.isReasonable() else False

    return success

def testHasCartSD():

    """
    tests hasCartSD function by testing events
    tests hasCardSD function by testing events


    return:
        [bool]

    """

    t = EventContainer("", 0, 0, 0)
    t.setValue("EventLat", "45")
    t.setValue("EventLon", "-27")
    t.setValue("TimeTolerance", "5")

    success = True
    success = success if not t.hasCartSD() else False
    t.setValue("EventCartStd",1)
    success = success if t.hasCartSD() else False
    t.setValue("EventCartStd", 0)
    success = success if not t.hasCartSD() else False
    t.setValue("EventCart2Std", 1)
    success = success if t.hasCartSD() else False
    t.setValue("EventCart2Std", 0)
    success = success if not t.hasCartSD() else False

    return success

def testHasPolarSD():

    t = EventContainer("", 0, 0, 0)
    t.setValue("EventLat", "45")
    t.setValue("EventLon", "-27")
    t.setValue("TimeTolerance", "5")

    success = True
    success = success if not t.hasPolarSD() else False
    t.setValue("EventLatStd",1)
    success = success if t.hasPolarSD() else False
    t.setValue("EventLatStd", 0)
    success = success if not t.hasPolarSD() else False
    t.setValue("EventLonStd", 1)
    success = success if t.hasPolarSD() else False
    t.setValue("EventLonStd", 0)
    success = success if not t.hasPolarSD() else False
    t.setValue("EventLat2Std", 1)
    success = success if t.hasPolarSD() else False
    t.setValue("EventLat2Std", 0)
    success = success if not t.hasPolarSD() else False
    t.setValue("EventLon2Std", 1)
    success = success if t.hasPolarSD() else False
    t.setValue("EventLon2Std", 0)
    success = success if not t.hasPolarSD() else False

    return success

def testEventToECEFVector():

    """
    Tests conversion of events to ECEF vectors by reversing and using haversine


    return:
        [bool]

    """

    t = EventContainer("", 0, 0, 0)

    success = True

    for test in range(1000):
        iLon, iLon2, iLat,iLat2 = np.random.uniform(-1000,1000,4)

        iHt, iHt2 = np.random.uniform(-50,1000,2)

        t.setValue("EventLat", iLat)
        t.setValue("EventLon", iLon)
        t.setValue("EventHt", iHt)
        t.setValue("EventLat2", iLat2)
        t.setValue("EventLon2", iLon2)
        t.setValue("EventHt2", iHt2)

        v1,v2 = t.eventToECEFVector()
        lat,lon,ht = ecef2LatLonAlt(v1[0],v1[1],v1[2])
        lat2, lon2, ht2 = ecef2LatLonAlt(v2[0], v2[1], v2[2])
        ht, ht2 = ht / 1000, ht2 / 1000
        lat, lon, lat2, lon2 = np.degrees(lat), np.degrees(lon), np.degrees(lat2), np.degrees(lon2)

        success = success if gcDistDeg(iLat, iLon, lat, lon) < 0.1  else False
        success = success if gcDistDeg(iLat2, iLon2, lat2, lon2) < 0.1 else False

        success = success if abs(iHt-ht) < 0.1 else False
        success = success if abs(iHt2 - ht2) < 0.1 else False

        if not success:
            print("fail")
            print(gcDistDeg(iLat, iLon, lat, lon))
            print(gcDistDeg(iLat2, iLon2, lat2, lon2))
            time.sleep(30)

    return success


def testEventCreation():

    """
    Tests creation and modification of events


    return:
        [bool]
    """

    success = True


    event = createATestEvent08()
    # print(event.eventToString())
    event.latLonAzElToLatLonLatLon()

    success = success if event.azim == 0 and event.elev == 45 else False

    # print(event.eventToString())
    event = createATestEvent08()
    event.setValue("EventAzim", 91)
    event.setValue("EventElev", 0)
    event.latLonAzElToLatLonLatLon()

    success = success if event.azim == 91 and event.elev == 45 else False
    success = success if gcDistDeg(45, 178, event.lat, event.lon) < 0.1 else False
    success = success if gcDistDeg(45, -179, event.lat2, event.lon2) < 0.1 else False
    success = success if abs(event.ht - 101) < 0.1 else False
    success = success if abs(event.ht2 - 20) < 0.1 else False

    # print(event.eventToString())
    event = createATestEvent08()
    event.setValue("EventAzim", 179)
    event.setValue("EventElev", 0)
    event.latLonAzElToLatLonLatLon()

    success = success if event.azim == 179 and event.elev == 45 else False
    success = success if gcDistDeg(45, 178, event.lat, event.lon) < 0.1 else False
    success = success if gcDistDeg(44.278, 179.017, event.lat2, event.lon2) < 0.1 else False
    success = success if abs(event.ht - 101) < 0.1 else False
    success = success if abs(event.ht2 - 20) < 0.1 else False

    event = createATestEvent08()
    event.setValue("EventAzim", 270)
    event.setValue("EventElev", 0)
    event.latLonAzElToLatLonLatLon()
    success = success if event.azim == 270 and event.elev == 45 else False
    success = success if gcDistDeg(45, 179, event.lat, event.lon) < 0.1 else False
    success = success if gcDistDeg(45, 178, event.lat2, event.lon2) < 0.1 else False
    success = success if abs(event.ht - 101) < 0.1 else False
    success = success if abs(event.ht2 - 20) < 0.1 else False

    event = createATestEvent08()
    event.setValue("EventAzim", 1)
    event.setValue("EventElev", 45)
    event.latLonAzElToLatLonLatLon()
    success = success if event.azim == 1 and event.elev == 45 else False

    success = success if gcDistDeg(44.99, 179, event.lat, event.lon) < 0.1 else False

    success = success if gcDistDeg(45.722, 179.018, event.lat2, event.lon2) < 0.1 else False
    success = success if abs(event.ht - 101) < 0.1 else False
    success = success if abs(event.ht2 - 20) < 0.1 else False

    # print(event.eventToString())
    event = createATestEvent08()
    event.setValue("EventAzim", -1)
    event.setValue("EventElev", 90)
    event.latLonAzElToLatLonLatLon()

    success = success if event.azim == -1 and event.elev == 45 else False
    success = success if gcDistDeg(44.99, 179, event.lat, event.lon) < 0.1 else False
    success = success if gcDistDeg(45.722, 179, event.lat2, event.lon2) < 0.1 else False
    success = success if abs(event.ht - 101) < 0.1 else False
    success = success if abs(event.ht2 - 20) < 0.1 else False


    # print(event.eventToString())
    event = createATestEvent08()
    event.setValue("EventAzim", 30)
    event.setValue("EventElev", 78)
    event.latLonAzElToLatLonLatLon()

    success = success if event.azim == 30 and event.elev == 78 else False
    success = success if event.azim == 30 and event.elev == 78 else False
    success = success if gcDistDeg(44.998, 178.998, event.lat, event.lon) < 0.1 else False
    success = success if gcDistDeg(45.132, 179.108, event.lat2, event.lon2) < 0.1 else False
    success = success if abs(event.ht - 101) < 0.1 else False
    success = success if abs(event.ht2 - 20) < 0.1 else False

    return success

def testApplyCartesianSD():


    """
    Tests application of standard deviation to cartesian coordinates


    return:
        [bool]
    """


    success = True
    event = createATestEvent07()
    event_population = []
    event.cart_std, event.cart2_std = 1000,2000
    event_population = event.appendPopulation(event_population, 1000)
    event_population = event.applyCartesianSD(event_population, seed = 0) # pass a seed for repeatability

    x1l,y1l,z1l = [],[],[]
    x2l,y2l,z2l = [],[],[]

    e = event
    for e in event_population:

        x1, y1, z1 = latLonAlt2ECEFDeg(e.lat, e.lon, e.ht * 1000)
        x2, y2, z2 = latLonAlt2ECEFDeg(e.lat2, e.lon2, e.ht2 * 1000)
        x1l.append(x1)
        y1l.append(y1)
        z1l.append(z1)
        x2l.append(x2)
        y2l.append(y2)
        z2l.append(z2)
    xstd, ystd, zstd = np.std(x1l), np.std(y1l), np.std(z1l)
    success = success if abs(xstd - event.cart_std) < 100 else False
    success = success if abs(ystd - event.cart_std) < 100 else False
    success = success if abs(zstd - event.cart_std) < 100 else False

    x2std, y2std, z2std = np.std(x2l), np.std(y2l), np.std(z2l)
    success = success if abs(x2std - event.cart2_std) < 100 else False
    success = success if abs(y2std - event.cart2_std) < 100 else False
    success = success if abs(z2std - event.cart2_std) < 100 else False



    xmn, ymn, zmn = np.mean(x1l), np.mean(y1l), np.mean(z1l)
    x2mn, y2mn, z2mn = np.mean(x2l), np.mean(y2l), np.mean(z2l)
    lat,lon,ht = ecef2LatLonAlt(xmn,ymn,zmn)
    lat2, lon2, ht2 = ecef2LatLonAlt(x2mn+50, y2mn+50 , z2mn+50 )
    lat, lon, lat2, lon2 = np.degrees(lat) , np.degrees(lon),np.degrees(lat2), np.degrees(lon2)
    success = success if gcDistDeg(lat, lon, event.lat, event.lon) < 0.1 else False
    success = success if gcDistDeg(lat2, lon2, event.lat2, event.lon2) < 0.1 else False
    success = success if abs(e.ht - ht/1000) < 10 and (e.ht2 -  ht2/1000) < 10 else False

    return success

def testApplyPolarSD():

    """
    Tests application of standard deviation to polar coordinates


    return:
         [bool]
    """

    success = True
    event = createATestEvent07()
    event_population = []
    event.lat_std, event.lon_std, event.ht_std, event.lat2_std, event.lon2_std,event.ht2_std = 0.01,0.02,1,0.05,0.6,5
    event_population = event.appendPopulation(event_population, 10000)
    event_population = event.applyPolarSD(event_population, seed = 0) # pass a seed for repeatbility

    lat1l,lon1l,ht1l = [],[],[]
    lat2l,lon2l,ht2l = [],[],[]

    e = event
    for e in event_population:

        lat1l.append(e.lat)
        lon1l.append(e.lon)
        ht1l.append(e.ht)

        lat2l.append(e.lat2)
        lon2l.append(e.lon2)
        ht2l.append(e.ht2)



    lat1std, lon1std, ht1std = np.std(lat1l), np.std(lon1l), np.std(ht1l)
    success = success if abs(lat1std - event.lat_std) < 0.01 else False
    success = success if abs(lon1std - event.lon_std) < 0.01 else False
    success = success if abs(ht1std - event.ht_std) < 0.1 else False

    lat2std, lon2std, ht2std = np.std(lat2l), np.std(lon2l), np.std(ht2l)
    success = success if abs(lat2std - event.lat2_std) < 0.01 else False
    success = success if abs(lon2std - event.lon2_std) < 0.01 else False
    success = success if abs(ht2std - event.ht2_std) < 0.1 else False






    lat1mn, lon1mn, ht1mn = np.mean(lat1l), np.mean(lon1l), np.mean(ht1l)
    lat2mn, lon2mn, ht2mn = np.mean(lat2l), np.mean(lon2l), np.mean(ht2l)

    success = success if gcDistDeg(lat1mn, lon1mn, event.lat, event.lon) < 0.1 else False
    success = success if gcDistDeg(lat2mn, lon2mn, event.lat2, event.lon2) < 0.1 else False
    success = success if abs(e.ht - ht1mn) < 5 and (e.ht2 - ht2mn) < 10 else False
    return success

def testsanitise():

    success = True

    if "This_string_is_safe" == sanitise("This string is safe", space_substitution="_", log_changes= False):
        success = success
    else:
        success = False

    if "Thisstringis_alsosafebuthasspacesstripped" == sanitise("This string is _ also safe but has spaces stripped", log_changes= False):
        success = success
    else:
        success = False

    if "This_string_is_-_also_safe_but_has_spaces_converted_to_us" == sanitise("This string is - also safe but has spaces converted to us", space_substitution="_", log_changes= False):
        success = success
    else:
        success = False


    if "This string is - not  safe but has spaces left in place" == sanitise("This string is - not ?.,`?/%%^&* safe but has spaces left in place", space_substitution=" ", log_changes= False):
        success = success
    else:
        success = False

    return success

def testIndividuals(logging = True):


    """
    Control function to check individual functions

    arguments:

    return:
        [bool]


    """

    individuals_success = True



    if testsanitise():
        if logging:
            log.info("santise passed tests")
    else:
        log.error("sanitise failed tests")
        individuals_success = False


    if testRevAz():
        if logging:
            log.info("revAz passed tests")
    else:
        log.error("revAz failed tests")
        individuals_success = False

    if testIsReasonable():
        if logging:
            log.info("isReasonable passed tests")
    else:
        log.error("isReasonable failed tests")
        individuals_success = False

    if testHasCartSD():
        if logging:
            log.info("hasCartSD passed tests")
    else:
        log.error("hasCartSD failed tests")
        individuals_success = False


    if testHasPolarSD():
        if logging:
            log.info("hasPolarSD passed tests")
    else:
        log.error("hasPolarSD failed tests")
        individuals_success = False

    if abs(gcDistDeg(31.7, 26.3, 45.1, 31.2) - 1549.2) < 0.5:
        if logging:
            log.info("GC Dist passed test")
    else:
        log.error("GC Dist failed test")
        individuals_success = False



    if testEventToECEFVector():
        if logging:
            log.info("eventToECEFVector passed tests")
    else:
        log.error("eventToECEFVector failed tests")
        individuals_success = False


    if convertGMNTimeToPOSIX("20210925_163127") == datetime.datetime(2021, 9, 25, 16, 31, 27):
        if logging:
            log.info("convertgmntimetoposix success")
    else:
        log.error("convertgmntimetoposix fail")
        individuals_success = False


    if testEventCreation():
        if logging:
            log.info("Event Creation success")
    else:
        log.error("Event Creation fail")

    if testApplyCartesianSD():
        if logging:
            log.info("Apply Cartesian SD success")
    else:
        log.error("Apply Cartesian SD fail")
        individuals_success = False

    if testApplyPolarSD():
        if logging:
            log.info("Apply Polar SD success")
    else:
        log.error("Apply Polar SD fail")
        individuals_success = False

    return individuals_success

if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="""Check a web page for trajectories, and upload relevant data. \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str,
                            help="Path to a config file which will be used instead of the default one.")

    arg_parser.add_argument('-o', '--oneshot', dest='one_shot', default=False, action="store_true",
                            help="Run once, and terminate.")

    arg_parser.add_argument('-d', '--deletedb', dest='delete_db', default=False, action="store_true",
                            help="Delete the event_monitor database at initialisation.")

    arg_parser.add_argument('-k', '--keepfiles', dest='keepfiles', default=False, action="store_true",
                            help="Keep working files")

    arg_parser.add_argument('-n', '--noupload', dest='noupload', default=False, action="store_true",
                            help="Do not upload")


    cml_args = arg_parser.parse_args()

    # Load the config file
    syscon = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('.'))

    # Set the web page to monitor

    if testIndividuals():
        log.info("Individual function test success")
        print("Individual function test success")
    else:
        log.error("Individual function test fail")
        print("Individual function test fail")


    try:
        # Add a random string after the URL to defeat caching


        if sys.version_info[0] < 3:
            web_page = urllib2.urlopen(syscon.event_monitor_webpage).read().splitlines()
        else:
            web_page = urllib.request.urlopen(syscon.event_monitor_webpage).read().decode("utf-8").splitlines()


    except:

        log.info("Nothing found at {}".format(syscon.event_monitor_webpage))

    if cml_args.delete_db and os.path.isfile(syscon.event_monitor_db_path):
        os.unlink(syscon.event_monitor_db_path)

    em = EventMonitor(syscon)



    if cml_args.one_shot:

        em.getEventsAndCheck()

    else:

        em.start()
