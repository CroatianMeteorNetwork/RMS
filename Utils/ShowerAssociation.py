""" Single station shower association. """

from __future__ import absolute_import, division, print_function

import copy
import datetime
import glob
import os
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from RMS.Astrometry.Conversions import (EARTH_CONSTANTS, datetime2JD,
                                        jd2Date, raDec2AltAz, raDec2Vector,
                                        vector2RaDec, AEH2Range)
from RMS.Formats.FFfile import filenameToDatetime
from RMS.Formats.FTPdetectinfo import findFTPdetectinfoFile, readFTPdetectinfo
from RMS.Formats.Showers import (generateActivityDiagram, loadShowers,
                                 makeShowerColors, Shower)
from RMS.Math import (angularSeparation, angularSeparationVect,
                      cartesianToPolar, isAngleBetween,
                      sphericalPointFromHeadingAndDistance, vectNorm)
from RMS.Routines.AllskyPlot import AllSkyPlot
from RMS.Routines.GreatCircle import (fitGreatCircle, greatCircle,
                                      greatCirclePhase)
from RMS.Routines.SolarLongitude import jd2SolLonSteyaert

EARTH = EARTH_CONSTANTS()


class MeteorSingleStation(object):
    def __init__(self, station_id, lat, lon, elev, ff_name):
        """ Container for single station observations which enables great circle fitting. 

        Arguments:
            station_id: [str] Station ID.
            lat: [float] Station +N latitude (deg).
            lon: [float] Station +E longitude (deg).
            elev: [float] Station elevation (m).
            ff_name: [str] Name of the FF file on which the meteor was recorded.
        """

        self.station_id = station_id
        self.lat = lat
        self.lon = lon
        self.elev = elev
        self.ff_name = ff_name

        self.jd_array = []
        self.ra_array = []
        self.dec_array = []
        self.mag_array = []

        self.cartesian_points = None

        self.normal = None

        self.meteor_begin_cartesian = None
        self.meteor_end_cartesian = None

        self.beg_azim, self.beg_alt = None, None
        self.end_azim, self.end_alt = None, None

        self.duration = None

        self.jdt_ref = None

        # Solar longitude of the beginning (degrees)
        self.lasun = None

        # Phases on the great circle of the beginning and the end
        self.gc_beg_phase = None
        self.gc_end_phase = None

        # Approx apparent shower radiant (only for associated meteors)
        self.radiant_ra = None
        self.radiant_dec = None



    def addPoint(self, jd, ra, dec, mag):

        self.jd_array.append(jd)
        self.ra_array.append(ra)
        self.dec_array.append(dec)
        self.mag_array.append(mag)




    def fitGC(self):
        """ Fits great circle to observations. """

        self.cartesian_points = []

        self.ra_array = np.array(self.ra_array)
        self.dec_array = np.array(self.dec_array)

        for ra, dec in zip(self.ra_array, self.dec_array):

            vect = vectNorm(raDec2Vector(ra, dec))

            self.cartesian_points.append(vect)


        self.cartesian_points = np.array(self.cartesian_points)

        # Set begin and end pointing vectors
        self.beg_vect = self.cartesian_points[0]
        self.end_vect = self.cartesian_points[-1]

        # Compute alt of the begining and the last point
        self.beg_azim, self.beg_alt = raDec2AltAz(self.ra_array[0], self.dec_array[0], self.jd_array[0], \
            self.lat, self.lon)
        self.end_azim, self.end_alt = raDec2AltAz(self.ra_array[-1], self.dec_array[-1], self.jd_array[-1], \
            self.lat, self.lon)


        # Fit a great circle through observations
        x_arr, y_arr, z_arr = self.cartesian_points.T
        coeffs, self.theta0, self.phi0 = fitGreatCircle(x_arr, y_arr, z_arr)

        # Calculate the plane normal
        self.normal = np.array([coeffs[0], coeffs[1], -1.0])

        # Norm the normal vector to unit length
        self.normal = vectNorm(self.normal)

        # Compute RA/Dec of the normal direction
        self.normal_ra, self.normal_dec = vector2RaDec(self.normal)


        # Take pointing directions of the beginning and the end of the meteor
        self.meteor_begin_cartesian = vectNorm(self.cartesian_points[0])
        self.meteor_end_cartesian = vectNorm(self.cartesian_points[-1])

        # Compute angular distance between begin and end (radians)
        self.ang_be = angularSeparationVect(self.beg_vect, self.end_vect)

        # Compute meteor duration in seconds
        self.duration = (self.jd_array[-1] - self.jd_array[0])*86400.0

        # Set the reference JD as the JD of the beginning
        self.jdt_ref = self.jd_array[0]

        # Compute the solar longitude of the beginning (degrees)
        self.lasun = np.degrees(jd2SolLonSteyaert(self.jdt_ref))

        # Meteor peak magnitude
        self.peak_mag = None

        # Only computed if the meteor belongs to a shower (height has to be assumed)
        self.peak_azim = None
        self.peak_alt = None
        self.abs_mag = None

        # Radiant elevation above the horizon (degrees) if the meteor belongs to a shower
        self.shower_elev = None


    def sampleGC(self, phase_angles):
        """ Sample the fitted great circle and return RA/dec of points for the given phase angles. 
        
        Arguments:
            phase_angles: [ndarray] An array of phase angles (degrees).

        Return:
            ra, dec: [ndarrays] Arrays of RA and Dec (degrees).
        """


        # Sample the great circle
        x_array, y_array, z_array = greatCircle(np.radians(phase_angles), self.theta0, self.phi0)

        if isinstance(x_array, float):
            x_array = [x_array]
            y_array = [y_array]
            z_array = [z_array]

        # Compute RA/Dec of every points
        ra_array = []
        dec_array = []
        for x, y, z in zip(x_array, y_array, z_array):
            ra, dec = vector2RaDec(np.array([x, y, z]))

            ra_array.append(ra)
            dec_array.append(dec)


        return np.array(ra_array), np.array(dec_array)


    def findGCPhase(self, ra, dec):
        """ Finds the phase of the great circle that is closest to the given RA/Dec. 
    
        Arguments;
            ra: [float] RA (deg).
            dec: [float] Declination (deg).

        Return:
            phase: [float] Phase (deg).
        """

        
        x, y, z = raDec2Vector(ra, dec)
        theta, phi = cartesianToPolar(x, y, z)

        # Find the corresponding phase
        phase = np.degrees(greatCirclePhase(theta, phi, self.theta0, self.phi0))

        return phase



    def angularSeparationFromGC(self, ra, dec):
        """ Compute the angular separation from the given coordinaes to the great circle. 
    
        Arguments;
            ra: [float] RA (deg).
            dec: [float] Declination (deg).

        Return:
            ang_separation: [float] Radiant distance (deg).
        """

        ang_separation = np.degrees(abs(np.pi/2 - angularSeparation(np.radians(ra), \
                np.radians(dec), np.radians(self.normal_ra), np.radians(self.normal_dec))))

        return ang_separation
    

    def computeMag(self, shower=None):
        """ Compute peak and absolute magnitude (if the shower is given).
        
        Keyword arguments:
            shower: [Shower instance] Shower object. Optional.
        """

        # Compute the apparent peak magnitude
        self.peak_mag = None
        peak_mag_index = None
        if np.any(self.mag_array):

            # Get the index of the peak magnitude
            peak_mag_index = np.argmin(self.mag_array)

            # Get the peak magnitude
            self.peak_mag = self.mag_array[peak_mag_index]


        # If the meteor belongs to a meteor shower
        if (shower is not None) and (self.peak_mag is not None):

            ### Compute the absolute magnitude assuming the meteor height ###

            # Get the reference meteor height
            if shower.ref_height is not None:
                    
                # Read from table if given
                ref_height = 1000*shower.ref_height

            else:
                    
                # Otherwise compute from the model
                ref_height = (heightModel(shower.v_init, ht_type='beg') + \
                    heightModel(shower.v_init, ht_type='end'))/2
            

            # Compute alt/az at the peak magnitude point
            self.peak_azim, self.peak_alt = raDec2AltAz(
                self.ra_array[peak_mag_index], self.dec_array[peak_mag_index], \
                self.jd_array[peak_mag_index], self.lat, self.lon)

            # Get the range to the meteor at the reference height
            r = AEH2Range(self.peak_azim, self.peak_alt, ref_height, self.lat, self.lon, self.elev)

            # Compute the peak absolute magnitude (apparent mag at 100 km distance)
            self.abs_mag = self.peak_mag + 5*np.log10((10**5)/r)

            ### ###



def heightModel(v_init, ht_type='beg'):
    """ Function that takes a velocity and returns an extreme begin/end meteor height that was fit on CAMS
        data.

    Arguments:
        v_init: [float] Meteor initial velocity (m/s).

    Keyword arguments:
        ht_type: [str] 'beg' or 'end'

    Return:
        ht: [float] Height (m).

    """

    def _htVsVelModel(v_init, c, a, b):
        return c + a*v_init + b/(v_init**3)


    # Convert velocity to km/s
    v_init /= 1000

    if ht_type.lower() == 'beg':

        # Begin height fit
        fit_params = [97.8411, 0.4081, -20919.3867]

    else:
        # End height fit
        fit_params = [59.4751, 0.3743, -11193.7365]


    # Compute the height in meters
    ht = 1000*_htVsVelModel(v_init, *fit_params)

    return ht




def estimateMeteorHeight(config, meteor_obj, shower):
    """ Estimate the height of a meteor from single station give a candidate shower. 

    Arguments:
        config: [Config instance]
        meteor_obj: [MeteorSingleStation instance]
        shower: [Shower instance]

    Return:
        (ht_b, ht_e): [tuple of floats] Estimated begin and end heights in meters.
    """

    ### Compute all needed values in alt/az coordinates ###
    
    # Compute beginning point vector in alt/az
    beg_ra, beg_dec = vector2RaDec(meteor_obj.beg_vect)
    beg_azim, beg_alt = raDec2AltAz(beg_ra, beg_dec, meteor_obj.jdt_ref, meteor_obj.lat, meteor_obj.lon)
    beg_vect_horiz = raDec2Vector(beg_azim, beg_alt)

    # Compute end point vector in alt/az
    end_ra, end_dec = vector2RaDec(meteor_obj.end_vect)
    end_azim, end_alt = raDec2AltAz(end_ra, end_dec, meteor_obj.jdt_ref, meteor_obj.lat, meteor_obj.lon)
    end_vect_horiz = raDec2Vector(end_azim, end_alt)

    # Compute radiant vector in alt/az
    radiant_azim, radiant_alt = raDec2AltAz(shower.ra, shower.dec, meteor_obj.jdt_ref, meteor_obj.lat, \
        meteor_obj.lon)
    radiant_vector_horiz = raDec2Vector(radiant_azim, radiant_alt)


    # Reject the pairing if the radiant is below the horizon
    if radiant_alt < 0:
        return -1


    # Get distance from Earth's centre to the position given by geographical coordinates for the 
    #   observer's latitude
    earth_radius = EARTH.EQUATORIAL_RADIUS/np.sqrt(1.0 - (EARTH.E**2)*np.sin(np.radians(config.latitude))**2)

    # Compute the distance from Earth's centre to the station (including the sea level height of the station)
    re_dist = earth_radius + config.elevation

    ### ###


    # Compute the distance the meteor traversed during its duration (meters)
    dist = shower.v_init*meteor_obj.duration

    # Compute the angle between the begin and the end point of the meteor (rad)
    theta_met = np.arccos(np.dot(vectNorm(beg_vect_horiz), vectNorm(end_vect_horiz)))

    # Compute the angle between the radiant vector and the end point (rad)
    theta_beg = np.arccos(np.dot(vectNorm(radiant_vector_horiz), -vectNorm(end_vect_horiz)))

    # Compute the angle between the radiant vector and the begin point (rad)
    theta_end = np.arccos(np.dot(-vectNorm(radiant_vector_horiz), -vectNorm(beg_vect_horiz)))

    # Compute the distance from the station to the begin point (meters)
    dist_beg = dist*np.sin(theta_beg)/np.sin(theta_met)

    # Compute the distance from the station to the end point (meters)
    dist_end = dist*np.sin(theta_end)/np.sin(theta_met)


    # Compute the height of the begin point using the law of cosines
    ht_b  = np.sqrt(dist_beg**2 + re_dist**2 - 2*dist_beg*re_dist*np.cos(np.radians(90 + meteor_obj.beg_alt)))
    ht_b -= earth_radius
    ht_b  = abs(ht_b)


    # Compute the height of the end point using the law of cosines
    ht_e  = np.sqrt(dist_end**2 + re_dist**2 - 2*dist_end*re_dist*np.cos(np.radians(90 + meteor_obj.end_alt)))
    ht_e -= earth_radius
    ht_e  = abs(ht_e)


    return ht_b, ht_e


def showerMatchScore(radiant_distance, height_difference):
    """ Given the distance of the great circle from the reference radiant and the difference in height,
        compute the match score.

    Arguments:
        radiant_distance: [float] Distance of the great circle from the reference radiant in degrees.
        height_difference: [float] Difference in height in meters.

    Return:
        score: [float] Match score.
    """

    # Add the radiant distance as is
    score = radiant_distance

    # Add the height difference in km with a weight
    ht_weight = 0.1
    score += ht_weight*(height_difference/1000.0)

    return score


def showerAssociation(config, ftpdetectinfo_list, shower_code=None, show_plot=False, save_plot=False, \
    plot_activity=False, flux_showers=False, color_map='viridis'):
    """ Do single station shower association based on radiant direction and height. 
    
    Arguments:
        config: [Config instance]
        ftpdetectinfo_list: [list] A list of paths to FTPdetectinfo files.

    Keyword arguments:
        shower_code: [str or Shower] Only use this one shower for association (e.g. ETA, PER, SDA). None by default,
            in which case all active showers will be associated. It can either by the three letter shower
            code or a Shower object.
        show_plot: [bool] Show the plot on the screen. False by default.
        save_plot: [bool] Save the plot in the folder with FTPdetectinfos. False by default.
        plot_activity: [bool] Whether to plot the shower activity plot of not. False by default.
        flux_showers: [bool] Use the set of showers for flux, not the default list. False by default.

    Return:
        associations, shower_counts: [tuple]
            - associations: [dict] A dictionary where the FF name and the meteor ordinal number on the FF
                file are keys, and the associated Shower object are values.
            - shower_counts: [list] A list of shower code and shower count pairs.
    """

    # If the shower code is given as the three letter code (or one at all), load the shower list from disk
    if isinstance(shower_code, str) or (shower_code is None):

        # Load the list of meteor showers
        if flux_showers:
            shower_table = loadShowers(config.shower_path, config.showers_flux_file_name)
        else:
            shower_table = loadShowers(config.shower_path, config.shower_file_name)

        shower_list = [Shower(shower_entry) for shower_entry in shower_table]

    # If the the Shower object was given, use it
    else:
        shower_list = [shower_code]
        shower_code = shower_code.name


    # Load FTPdetectinfos
    meteor_data = []
    for ftpdetectinfo_path in ftpdetectinfo_list:

        if not os.path.isfile(ftpdetectinfo_path):
            print('No such file:', ftpdetectinfo_path)
            continue

        meteor_data += readFTPdetectinfo(*os.path.split(ftpdetectinfo_path))

    if not len(meteor_data):
        return {}, []


    # Dictionary which holds FF names as keys and meteor measurements + associated showers as values
    associations = {}

    for meteor in meteor_data:

        ff_name, cam_code, meteor_No, n_segments, fps, hnr, mle, binn, px_fm, rho, phi, meteor_meas = meteor

        # Skip very short meteors
        if len(meteor_meas) < 4:
            continue

        # Check if the data is calibrated
        if not meteor_meas[0][0]:
            print('Data is not calibrated! Meteors cannot be associated to showers!')
            break


        # Init container for meteor observation
        meteor_obj = MeteorSingleStation(cam_code, config.latitude, config.longitude, config.elevation, 
                                         ff_name)

        # Infill the meteor structure
        for entry in meteor_meas:
            
            calib_status, frame_n, x, y, ra, dec, azim, elev, inten, mag = entry

            # Compute the Julian data of every point
            jd = datetime2JD(filenameToDatetime(ff_name) + datetime.timedelta(seconds=float(frame_n)/fps))

            meteor_obj.addPoint(jd, ra, dec, mag)

            
        # Fit the great circle and compute the geometrical parameters
        meteor_obj.fitGC()


        # Skip all meteors with beginning heights below 15 deg
        if meteor_obj.beg_alt < 15:
            continue

        
        # Go through all showers in the list and find the best match
        best_match_shower = None
        # best_match_dist = np.inf
        best_match_score = np.inf
        for shower in shower_list:

            # If the shower code was given, only check this one shower
            if shower_code is not None:
                if shower.name.lower() != shower_code.lower():
                    continue



            ### Solar longitude filter

            # If the shower doesn't have a stated beginning or end, check if the meteor is within a preset
            # threshold solar longitude difference
            if np.any(np.isnan([shower.lasun_beg, shower.lasun_end])):

                shower.lasun_beg = (shower.lasun_max - config.shower_lasun_threshold)%360
                shower.lasun_end = (shower.lasun_max + config.shower_lasun_threshold)%360


            # Filter out all showers which are not active    
            if not isAngleBetween(np.radians(shower.lasun_beg), np.radians(meteor_obj.lasun), 
                np.radians(shower.lasun_end)):

                continue

            ### ###


            ### Radiant filter ###

            # Get the reference height for the given shower
            if shower.ref_height is not None:

                # Read from table if given
                meteor_fixed_ht = 1000*shower.ref_height

            else:

                # Otherwise compute the height from the model
                meteor_ht_beg = heightModel(1000*shower.vg, ht_type='beg')
                meteor_ht_end = heightModel(1000*shower.vg, ht_type='end')
                meteor_fixed_ht = (meteor_ht_beg + meteor_ht_end)/2

            # Compute the apparent radiant
            shower.computeApparentRadiant(config.latitude, config.longitude, meteor_obj.jdt_ref, \
                meteor_fixed_ht=meteor_fixed_ht)

            # Compute the angle between the meteor radiant and the great circle normal
            radiant_separation = meteor_obj.angularSeparationFromGC(shower.ra, shower.dec)

            # Load the appropriate association radius
            if hasattr(shower, 'association_radius'):
                association_radius = shower.association_radius
            else:
                association_radius = config.shower_max_radiant_separation

            # Make sure the meteor is within the radiant distance threshold
            if radiant_separation > association_radius:
                continue


            # Compute angle between the meteor's beginning and end, and the shower radiant
            shower.radiant_vector = vectNorm(raDec2Vector(shower.ra, shower.dec))
            begin_separation = np.degrees(angularSeparationVect(shower.radiant_vector, \
                meteor_obj.meteor_begin_cartesian))
            end_separation = np.degrees(angularSeparationVect(shower.radiant_vector, \
                meteor_obj.meteor_end_cartesian))


            # Make sure the beginning of the meteor is closer to the radiant than it's end
            if begin_separation > end_separation:
                continue

            ### ###


            ### Height filter ###

            # Estimate the limiting meteor height from the velocity (meters)
            filter_beg_ht = heightModel(shower.v_init, ht_type='beg')
            filter_end_ht = heightModel(shower.v_init, ht_type='end')

            # Get the reference meteor height
            if shower.ref_height is not None:

                # Read from table if given
                ref_height = 1000*shower.ref_height

            else:

                # Otherwise compute from the model
                ref_height = (filter_beg_ht + filter_end_ht)/2


            ### Estimate the meteor beginning height with +/- 1 frame, otherwise some short meteor may get
            ###   rejected

            meteor_obj_orig = copy.deepcopy(meteor_obj)

            # Shorter
            meteor_obj_m1 = copy.deepcopy(meteor_obj_orig)
            meteor_obj_m1.duration -= 1.0/config.fps
            meteor_ht_m1 = np.mean(estimateMeteorHeight(config, meteor_obj_m1, shower))

            # Nominal
            meteor_ht = np.mean(estimateMeteorHeight(config, meteor_obj_orig, shower))

            # Longer
            meteor_obj_p1 = copy.deepcopy(meteor_obj_orig)
            meteor_obj_p1.duration += 1.0/config.fps
            meteor_ht_p1 = np.mean(estimateMeteorHeight(config, meteor_obj_p1, shower))


            meteor_obj = meteor_obj_orig


            ### ###

            # If all heights (even those with +/- 1 frame) are outside the height range, reject the meteor
            if ((meteor_ht_p1 < filter_end_ht) or (meteor_ht_p1 > filter_beg_ht)) and \
                ((meteor_ht    < filter_end_ht) or (meteor_ht    > filter_beg_ht)) and \
                ((meteor_ht_m1 < filter_end_ht) or (meteor_ht_m1 > filter_beg_ht)):

                continue

            ### ###


            # Compute the radiant elevation above the horizon
            shower.azim, shower.elev = raDec2AltAz(shower.ra, shower.dec, meteor_obj.jdt_ref, \
                config.latitude, config.longitude)


            ### Compute the association score ###

            # Compute the difference between the observed and thoretical height
            meteor_ht_diff = abs(meteor_ht - ref_height)

            # Compute the association score
            association_score = showerMatchScore(radiant_separation, meteor_ht_diff)

            ### ###


            # Take the shower with the smallest match score if there are multiple candidates
            if association_score < best_match_score:
                best_match_score = association_score
                best_match_shower = copy.deepcopy(shower)

            # # Take the shower that's closest to the great circle if there are multiple candidates
            # if radiant_separation < best_match_dist:
            #     best_match_dist = radiant_separation
            #     best_match_shower = copy.deepcopy(shower)


        # If a shower is given and the match is not this shower, skip adding the meteor to the list
        # If no specific shower is give for association, add all meteors
        if ((shower_code is not None) and (best_match_shower is not None)) or (shower_code is None):

            # Store the associated shower
            associations[(ff_name, meteor_No)] = [meteor_obj, best_match_shower]


    # Compute apparent and absolute magnitudes for all meteors (if they belong to a shower)
    for key in associations:
        meteor_obj, shower = associations[key]
        meteor_obj.computeMag(shower=shower)


    # Find shower frequency and sort by count
    shower_name_list_temp = []
    shower_list_temp = []
    for key in associations:
        _, shower = associations[key]

        if shower is None:
            shower_name = '...'
        else:
            shower_name = shower.name

        shower_name_list_temp.append(shower_name)
        shower_list_temp.append(shower)

    _, unique_showers_indices = np.unique(shower_name_list_temp, return_index=True)
    unique_shower_names = np.array(shower_name_list_temp)[unique_showers_indices]
    unique_showers = np.array(shower_list_temp)[unique_showers_indices]
    shower_counts = [[shower_obj, shower_name_list_temp.count(shower_name)] for shower_obj, \
        shower_name in zip(unique_showers, unique_shower_names)]
    shower_counts = sorted(shower_counts, key=lambda x: x[1], reverse=True)


    # Create a plot of showers
    if show_plot or save_plot:

        # Generate consistent colours
        colors_by_name = makeShowerColors(shower_list, color_map=color_map)
        def get_shower_color(shower):
            try:
                return colors_by_name[shower.name] if shower else "0.4"
            except KeyError:
                return 'gray'

        # Init the figure
        plt.figure()

        # Init subplots depending on if the activity plot is done as well
        if plot_activity:

            # Scale the shower plot according to number
            shower_ax_scale = np.sqrt(len(shower_list))/12 + 0.25

            gs = gridspec.GridSpec(2, 1, height_ratios=[3, shower_ax_scale])
            ax_allsky = plt.subplot(gs[0], facecolor='black')
            ax_activity = plt.subplot(gs[1], facecolor='black')
        else:
            ax_allsky = plt.subplot(111, facecolor='black')


        # Init the all-sky plot
        allsky_plot = AllSkyPlot(ax_handle=ax_allsky)

        # Plot all meteors
        for key in associations:

            meteor_obj, shower = associations[key]


            ### Plot the observed meteor points ###
            color = get_shower_color(shower)
            allsky_plot.plot(meteor_obj.ra_array, meteor_obj.dec_array, color=color, linewidth=1, zorder=4)

            # Head color of sporadics
            peak_color = '0.5'
            peak_alpha = 0.5

            # Plot the peak of shower meteors a different color
            if shower is not None:
                peak_color = 'tomato'
                peak_alpha = 0.8

            allsky_plot.scatter(meteor_obj.ra_array[-1], meteor_obj.dec_array[-1], c=peak_color, marker='+', \
                s=2, zorder=5, alpha=peak_alpha)

            ### ###


            ### Plot fitted great circle points ###

            # Find the GC phase angle of the beginning of the meteor
            gc_beg_phase = meteor_obj.findGCPhase(meteor_obj.ra_array[0], meteor_obj.dec_array[0])[0]%360

            # If the meteor belongs to a shower, find the GC phase which ends at the shower
            if shower is not None:
                gc_end_phase = meteor_obj.findGCPhase(shower.ra, shower.dec)[0]%360

                # Fix 0/360 wrap
                if abs(gc_end_phase - gc_beg_phase) > 180:
                    if gc_end_phase > gc_beg_phase:
                        gc_end_phase -= 360
                    else:
                        gc_beg_phase -= 360

                gc_alpha = 1.0


            else:

                # If it's a sporadic, find the direction to which the meteor should extend
                gc_end_phase = meteor_obj.findGCPhase(meteor_obj.ra_array[-1], \
                    meteor_obj.dec_array[-1])[0]%360

                # Find the correct direction
                if (gc_beg_phase - gc_end_phase)%360 > (gc_end_phase - gc_beg_phase)%360:
                    gc_end_phase = gc_beg_phase - 170

                else:
                    gc_end_phase = gc_beg_phase + 170

                gc_alpha = 0.5


            # Store great circle beginning and end phase
            meteor_obj.gc_beg_phase = gc_beg_phase
            meteor_obj.gc_end_phase = gc_end_phase

            # Get phases 180 deg before the meteor
            phase_angles = np.linspace(gc_end_phase, gc_beg_phase, 100)%360

            # Compute RA/Dec of points on the great circle
            ra_gc, dec_gc = meteor_obj.sampleGC(phase_angles)

            # Cull all points below the horizon
            azim_gc, elev_gc = raDec2AltAz(ra_gc, dec_gc, meteor_obj.jdt_ref, config.latitude, \
                config.longitude)
            temp_arr = np.c_[ra_gc, dec_gc]
            temp_arr = temp_arr[elev_gc > 0]
            ra_gc, dec_gc = temp_arr.T

            # Plot the great circle fitted on the radiant
            gc_color = get_shower_color(shower)
            allsky_plot.plot(ra_gc, dec_gc, linestyle='dotted', color=gc_color, alpha=gc_alpha, linewidth=1)

            # Plot the point closest to the shower radiant
            if shower is not None:
                allsky_plot.plot(ra_gc[0], dec_gc[0], color='r', marker='+', ms=5, mew=1)

                # Store shower radiant point
                meteor_obj.radiant_ra = ra_gc[0]
                meteor_obj.radiant_dec = dec_gc[0]


            ### ###


        ### Plot all showers ###

        # Find unique showers and their apparent radiants computed at highest radiant elevation
        # (otherwise the apparent radiants can be quite off)
        shower_dict = {}
        for key in associations:
            meteor_obj, shower = associations[key]

            if shower is None:
                continue

            # If the shower name is in dict, find the shower with the highest radiant elevation
            if shower.name in shower_dict:
                if shower.elev > shower_dict[shower.name].elev:
                    shower_dict[shower.name] = shower
                
            else:
                shower_dict[shower.name] = shower


        # Plot the location of shower radiants
        for shower_name in shower_dict:
            
            shower = shower_dict[shower_name]

            heading_arr = np.linspace(0, 360, 50)

            # Load the appropriate association radius
            if hasattr(shower, 'association_radius'):
                association_radius = shower.association_radius
            else:
                association_radius = config.shower_max_radiant_separation

            # Compute coordinates on a circle around the given RA, Dec
            ra_circle, dec_circle = sphericalPointFromHeadingAndDistance(shower.ra, shower.dec, \
                heading_arr, association_radius)


            # Plot the shower circle
            allsky_plot.plot(ra_circle, dec_circle, color=colors_by_name[shower_name])


            # Plot the shower name
            x_text, y_text = allsky_plot.raDec2XY(shower.ra, shower.dec)
            allsky_plot.ax.text(x_text, y_text, shower.name, color='w', size=8, va='center', \
                ha='center', zorder=6)



        # Plot station name and solar longiutde range
        allsky_plot.ax.text(-180, 89, "{:s}".format(cam_code), color='w', family='monospace')

        # Get a list of JDs of meteors
        jd_list = [associations[key][0].jdt_ref for key in associations]

        if len(jd_list):

            # Get the range of solar longitudes
            jd_min = min(jd_list)
            sol_min = np.degrees(jd2SolLonSteyaert(jd_min))
            jd_max = max(jd_list)
            sol_max = np.degrees(jd2SolLonSteyaert(jd_max))

            # Plot the date and solar longitude range
            date_sol_beg = u"Beg: {:s} (sol = {:.2f}\u00b0)".format(jd2Date(jd_min, dt_obj=True).strftime("%Y%m%d %H:%M:%S"), sol_min)
            date_sol_end = u"End: {:s} (sol = {:.2f}\u00b0)".format(jd2Date(jd_max, dt_obj=True).strftime("%Y%m%d %H:%M:%S"), sol_max)
            
            allsky_plot.ax.text(-180, 85, date_sol_beg, color='w', family='monospace')
            allsky_plot.ax.text(-180, 81, date_sol_end, color='w', family='monospace')
            allsky_plot.ax.text(-180, 77, "-"*len(date_sol_end), color='w', family='monospace')

            # Plot shower counts
            for i, (shower, count) in enumerate(shower_counts):

                if shower is not None:
                    shower_name = shower.name
                else:
                    shower_name = "..."

                allsky_plot.ax.text(-180, 73 - i*4, "{:s}: {:d}".format(shower_name, count), color='w', \
                    family='monospace')


            ### ###

            # Plot yearly meteor shower activity
            if plot_activity:

                # Plot the activity diagram
                generateActivityDiagram(config, shower_list, ax_handle=ax_activity, \
                    sol_marker=[sol_min, sol_max], color_map=color_map)


        

        # Save plot and text file
        if save_plot:

            dir_path, ftpdetectinfo_name = os.path.split(ftpdetectinfo_path)
            ftpdetectinfo_base_name = ftpdetectinfo_name.replace('FTPdetectinfo_', '').replace('.txt', '')
            plot_name = ftpdetectinfo_base_name + '_radiants.png'

            # Increase figure size
            allsky_plot.fig.set_size_inches(18, 9, forward=True)

            allsky_plot.beautify()

            plt.savefig(os.path.join(dir_path, plot_name), dpi=100, facecolor='k')



            # Save the text file with shower info
            if len(jd_list):
                with open(os.path.join(dir_path, ftpdetectinfo_base_name + "_radiants.txt"), 'w') as f:

                    # Print station code
                    f.write("# RMS single station association\n")
                    f.write("# \n")
                    f.write("# Station: {:s}\n".format(cam_code))

                    # Print date range
                    f.write("#                    Beg          |            End            \n")
                    f.write("#      -----------------------------------------------------\n")
                    f.write("# Date | {:24s} | {:24s} \n".format(jd2Date(jd_min, \
                        dt_obj=True).strftime("%Y%m%d %H:%M:%S.%f"), jd2Date(jd_max, \
                        dt_obj=True).strftime("%Y%m%d %H:%M:%S.%f")))
                    f.write("# Sol  | {:>24.2f} | {:>24.2f} \n".format(sol_min, sol_max))

                    # Write shower counts
                    f.write("# \n")
                    f.write("# Shower counts:\n")
                    f.write("# --------------\n")
                    f.write("# Code, Count, IAU link\n")

                    for i, (shower, count) in enumerate(shower_counts):

                        if shower is not None:
                            shower_name = shower.name

                            # Create link to the IAU database of showers
                            iau_link = "https://www.ta3.sk/IAUC22DB/MDC2022/Roje/pojedynczy_obiekt.php?kodstrumienia={:05d}".format(shower.iau_code_int)

                        else:
                            shower_name = "..."
                            iau_link = "None"

                        f.write("# {:>4s}, {:>5d}, {:s}\n".format(shower_name, count, iau_link))



                    f.write("# \n")
                    f.write("# Meteor parameters:\n")
                    f.write("# ------------------\n")
                    f.write("#          Date And Time,      Beg Julian date,     La Sun, Shower, RA beg, Dec beg, RA end, Dec end, RA rad, Dec rad, GC theta0,  GC phi0, GC beg phase, GC end phase, AppMag, AbsMag, RadElev\n")


                    # Create a sorted list of meteor associations by time
                    associations_list = [associations[key] for key in associations]
                    associations_list = sorted(associations_list, key=lambda x: x[0].jdt_ref)

                    # Write out meteor parameters
                    for meteor_obj, shower in associations_list:

                        # Format the peak magnitude
                        if meteor_obj.peak_mag is not None:
                            peak_mag = "{:+.2f}".format(meteor_obj.peak_mag)

                        else:
                            peak_mag = "None"

                        # Format the absolute magnitude
                        if meteor_obj.abs_mag is not None:
                            abs_mag = "{:+.2f}".format(meteor_obj.abs_mag)

                        else:
                            abs_mag = "None"

                        # Format the radiant elevation
                        if shower is not None:
                            rad_elev = "{:.2f}".format(shower.elev)

                        else:
                            rad_elev = "None"
                        


                        if shower is not None:

                            f.write("{:24s}, {:20.12f}, {:>10.6f}, {:>6s}, {:6.2f}, {:+7.2f}, {:6.2f}, {:+7.2f}, {:6.2f}, {:+7.2f}, {:9.3f}, {:8.3f}, {:12.3f}, {:12.3f}, {:>6s}, {:>6s}, {:>7s}\n".format(jd2Date(meteor_obj.jdt_ref, dt_obj=True).strftime("%Y%m%d %H:%M:%S.%f"), \
                                meteor_obj.jdt_ref, meteor_obj.lasun, shower.name, \
                                meteor_obj.ra_array[0]%360, meteor_obj.dec_array[0], \
                                meteor_obj.ra_array[-1]%360, meteor_obj.dec_array[-1], \
                                meteor_obj.radiant_ra%360, meteor_obj.radiant_dec, \
                                np.degrees(meteor_obj.theta0), np.degrees(meteor_obj.phi0), \
                                meteor_obj.gc_beg_phase, meteor_obj.gc_end_phase, peak_mag, abs_mag, rad_elev))

                        else:
                            f.write("{:24s}, {:20.12f}, {:>10.6f}, {:>6s}, {:6.2f}, {:+7.2f}, {:6.2f}, {:+7.2f}, {:>6s}, {:>7s}, {:9.3f}, {:8.3f}, {:12.3f}, {:12.3f}, {:>6s}, {:>6s}, {:>7s}\n".format(jd2Date(meteor_obj.jdt_ref, dt_obj=True).strftime("%Y%m%d %H:%M:%S.%f"), \
                                meteor_obj.jdt_ref, meteor_obj.lasun, '...', meteor_obj.ra_array[0]%360, \
                                meteor_obj.dec_array[0], meteor_obj.ra_array[-1]%360, \
                                meteor_obj.dec_array[-1], "None", "None", np.degrees(meteor_obj.theta0), \
                                np.degrees(meteor_obj.phi0), meteor_obj.gc_beg_phase, \
                                meteor_obj.gc_end_phase, peak_mag, abs_mag, rad_elev))





        if show_plot:
            allsky_plot.show()

        else:
            plt.clf()
            plt.close()



    return associations, shower_counts






if __name__ == "__main__":

    import argparse

    import RMS.ConfigReader as cr

    ### COMMAND LINE ARGUMENTS
    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Perform single-station established shower association on an FTPdetectinfo file.")

    arg_parser.add_argument('ftpdetectinfo_path', metavar='FTPDETECTINFO_PATH', nargs='+', type=str, \
        help='Path to an FTPdetectinfo file or a directory with an FTPdetectinfo file.')

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one.")

    arg_parser.add_argument('-s', '--shower', metavar='SHOWER', type=str, \
        help="Associate just this single shower given its code (e.g. PER, ORI, ETA).")

    arg_parser.add_argument('-f', '--fluxshowers', action="store_true", \
        help="""Only show shower association for showers used for flux estimation.""")

    arg_parser.add_argument('-x', '--hideplot', action="store_true", \
        help="""Do not show the plot on the screen.""")

    arg_parser.add_argument('-p', '--palette', metavar='PALETTE', type=str, \
        help="color palette to use - one of viridis, gist_ncar, rainbow etc")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    ftpdetectinfo_path = cml_args.ftpdetectinfo_path


    # Find an FTPdetectinfo file if only a directory is given
    if len(ftpdetectinfo_path) == 1:

        # If a directory is given, find the FTPdetectinfo
        if os.path.isdir(ftpdetectinfo_path[0]):
        
            ftpdetectinfo_path = findFTPdetectinfoFile(ftpdetectinfo_path[0])
            ftpdetectinfo_path_list = [ftpdetectinfo_path]

        # If a file is given
        elif os.path.isfile(ftpdetectinfo_path[0]):
            ftpdetectinfo_path_list = ftpdetectinfo_path

        # If a wildcard was given
        else:
            ftpdetectinfo_path_list = glob.glob(ftpdetectinfo_path[0])


    # If multiple files are given, find them all
    else:
    
        # Apply wildcards to input if more are given
        ftpdetectinfo_path_list = []
        for entry in ftpdetectinfo_path:
            ftpdetectinfo_path_list += glob.glob(entry)


    # If there are no good files given, notify the user
    if len(ftpdetectinfo_path_list) == 0:
        print("No FTPdetectinfo files given!")
        sys.exit()

    else:
        print()
        print("Using FTPdetectinfo files:")
        for ftpdetectinfo_path in ftpdetectinfo_path_list:
            print(ftpdetectinfo_path)
        

    # Extract parent directory
    dir_path = os.path.dirname(ftpdetectinfo_path_list[0])

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, dir_path)
    
    # choose a colour scheme to use
    if cml_args.palette is None:
        color_map = config.shower_color_map
    else:
        color_map = cml_args.palette

    # Perform shower association
    associations, shower_counts = showerAssociation(config, ftpdetectinfo_path_list, \
        shower_code=cml_args.shower, show_plot=(not cml_args.hideplot), save_plot=True, plot_activity=True,
        flux_showers=cml_args.fluxshowers, color_map=color_map)


    # Print results to screen
    if shower_counts:
        print()
        print('Shower ranking:')
        for shower, count in shower_counts:

            if shower is None:
                shower_name = '...'
            else:
                shower_name = shower.name

            print(shower_name, count)

    else:
        print("No meteors!")



