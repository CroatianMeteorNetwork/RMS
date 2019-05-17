""" Single station shower association. """

from __future__ import print_function, division, absolute_import



import os
import sys
import datetime
import copy

import numpy as np

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

from RMS.Astrometry.Conversions import raDec2Vector, vector2RaDec, datetime2JD, \
    geocentricToApparentRadiantAndVelocity, raDec2AltAz, raDec2AltAz_vect, EARTH_CONSTANTS
from RMS.Formats.FFfile import filenameToDatetime
from RMS.Formats.FTPdetectinfo import readFTPdetectinfo
from RMS.Formats.Showers import loadShowers
from RMS.Math import vectNorm, angularSeparation, angularSeparationVect, isAngleBetween, \
    sphericalPointFromHeadingAndDistance, cartesianToPolar
from RMS.Routines.GreatCircle import fitGreatCircle, greatCircle, greatCirclePhase
from RMS.Routines.SolarLongitude import jd2SolLonSteyaert
from RMS.Routines.AllskyPlot import AllSkyPlot


EARTH = EARTH_CONSTANTS()


class MeteorSingleStation(object):
    def __init__(self, station_id, lat, lon):
        """ Container for single station observations which enables great circle fitting. 

        Arguments:
            station_id: [str]
            lat: [float] +N latitude (deg).
            lon: [float] +E longitude (deg).
        """

        self.station_id = station_id
        self.lat = lat
        self.lon = lon

        self.jd_array = []
        self.ra_array = []
        self.dec_array = []

        self.cartesian_points = None

        self.normal = None

        self.meteor_begin_cartesian = None
        self.meteor_end_cartesian = None

        self.duration = None

        self.jdt_ref = None

        # Solar longitude of the beginning (degrees)
        self.lasun = None



    def addPoint(self, jd, ra, dec):

        self.jd_array.append(jd)
        self.ra_array.append(ra)
        self.dec_array.append(dec)




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
            ang_separation: [float] Radiant dsitance (deg).
        """

        ang_separation = np.degrees(abs(np.pi/2 - angularSeparation(np.radians(ra), \
                np.radians(dec), np.radians(self.normal_ra), np.radians(self.normal_dec))))

        return ang_separation





class Shower(object):
    def __init__(self, shower_entry):

        self.name = shower_entry[1]
        self.lasun_beg = shower_entry[3] # deg
        self.lasun_max = shower_entry[4] # deg
        self.lasun_end = shower_entry[5] # deg
        self.ra_g = shower_entry[6] # deg
        self.dec_g = shower_entry[7] # deg
        self.dra = shower_entry[8] # deg
        self.ddec = shower_entry[9] # deg
        self.vg = shower_entry[10] # km/s

        # Apparent radiant
        self.ra = None # deg
        self.dec = None # deg
        self.v_init = None # m/s
        self.azim = None # deg
        self.elev = None # deg
        self.shower_vector = None




def heightModel(v_init, ht_type='beg'):
    """ Function that takes a velocity and returns an extreme begin/end meteor height that was fit on CAMS
        data.

    Arguments:
        v_init: [float] Meteor initial velocity (m/s).

    Keyword arguments:
        ht_type: [str] 'beg' or 'end'

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




def estimateMeteorHeight(meteor_obj, shower):
    """ Estimate the height of a meteor from single station give a candidate shower. 

    Arguments:
        meteor_obj: [MeteorSingleStation instance]
        shower: [Shower instance]

    Return:
        ht: [float] Estimated height in meters.
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

    # Compute normal vector in alt/az
    normal_azim, normal_alt = raDec2AltAz(meteor_obj.normal_ra, meteor_obj.normal_dec, meteor_obj.jdt_ref, \
        meteor_obj.lat, meteor_obj.lon)
    normal_horiz = raDec2Vector(normal_azim, normal_alt)

    # Compute radiant vector in alt/az
    radiant_azim, radiant_alt = raDec2AltAz(shower.ra, shower.dec, meteor_obj.jdt_ref, meteor_obj.lat, \
        meteor_obj.lon)
    radiant_vector_horiz = raDec2Vector(radiant_azim, radiant_alt)


    # Reject the pairing if the radiant is below the horizon
    if radiant_alt < 0:
        return -1

    ### ###


    # Compute cartesian coordinates of the pointing at the beginning of the meteor
    pt = vectNorm(beg_vect_horiz)

    # Compute reference vector perpendicular to the plane normal and the radiant
    vec = vectNorm(np.cross(normal_horiz, radiant_vector_horiz))

    # Compute angles between the reference vector and the pointing
    dot_vb = np.dot(vec, beg_vect_horiz)
    dot_ve = np.dot(vec, end_vect_horiz)
    dot_vp = np.dot(vec, pt)

    # Compute distance to the radiant intersection line
    r_mag  = 1.0/(dot_vb**2)
    r_mag += 1.0/(dot_ve**2)
    r_mag += -2*np.cos(meteor_obj.ang_be)/(dot_vb*dot_ve)
    r_mag  = np.sqrt(r_mag)
    r_mag  = shower.v_init*meteor_obj.duration/r_mag
    pt_mag = r_mag/dot_vp

    # Compute the height
    ht  = pt_mag**2 + EARTH.EQUATORIAL_RADIUS**2 \
        - 2*pt_mag*EARTH.EQUATORIAL_RADIUS*np.cos(np.radians(90 - meteor_obj.beg_alt))
    ht  = np.sqrt(ht)
    ht -= EARTH.EQUATORIAL_RADIUS
    ht  = abs(ht)

    return ht




def showerAssociation(config, ftpdetectinfo_path, shower_code=None, plot_showers=False):
    """ """


    # Load FTPdetectinfo
    meteor_data = readFTPdetectinfo(*os.path.split(ftpdetectinfo_path))


    # Load the list of meteor showers
    shower_list = loadShowers(config.shower_path, config.shower_file_name)


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
        meteor_obj = MeteorSingleStation(cam_code, config.latitude, config.longitude)

        # Infill the meteor structure
        for entry in meteor_meas:
            
            calib_status, frame_n, x, y, ra, dec, azim, elev, inten, mag = entry

            # Compute the Julian data of every point
            jd = datetime2JD(filenameToDatetime(ff_name) + datetime.timedelta(seconds=float(frame_n)/fps))

            meteor_obj.addPoint(jd, ra, dec)

            
        # Fit the great circle and compute the geometrical parameters
        meteor_obj.fitGC()

        
        # Go through all showers in the list and find the best match
        best_match_shower = None
        best_match_dist = np.inf
        for shower_entry in shower_list:

            # Extract shower parameters
            shower = Shower(shower_entry)


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

            # Compute the location of the radiant due to radiant drift
            if not np.any(np.isnan([shower.dra, shower.ddec])):
                
                # Solar longitude difference form the peak
                lasun_diff = abs(meteor_obj.lasun - shower.lasun_max)%360
                if lasun_diff > 180:
                    lasun_diff = 360 - lasun_diff


                shower.ra_g = shower.ra_g + lasun_diff*shower.dra
                shower.dec_g = shower.dec_g + lasun_diff*shower.ddec


            # Compute the apparent radiant - assume that the meteor is directly above the station
            meteor_fixed_ht = 100000 # 100 km
            shower.ra, shower.dec, shower.v_init = geocentricToApparentRadiantAndVelocity(shower.ra_g, \
                shower.dec_g, 1000*shower.vg, config.latitude, config.longitude, meteor_fixed_ht, \
                meteor_obj.jdt_ref, include_rotation=True)

            # Compute the angle between the meteor radiant and the great circle normal
            radiant_separation = meteor_obj.angularSeparationFromGC(shower.ra, shower.dec)


            # Make sure the meteor is within the radiant distance threshold
            if radiant_separation > config.shower_max_radiant_separation:
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


            # Estimate the meteor beginning height
            meteor_beg_ht = estimateMeteorHeight(meteor_obj, shower)

            # print('Height filter:')
            # print('    Max: {:.2f} km'.format(filter_beg_ht/1000))
            # print('    Met: {:.2f} km'.format(meteor_beg_ht/1000))
            # print('    Min: {:.2f} km'.format(filter_end_ht/1000))

            
            # If the height is outside the height range, reject the meteor
            if (meteor_beg_ht < filter_end_ht) or (meteor_beg_ht > filter_beg_ht):
                continue

            ### ###


            # Compute the radiant elevation above the horizon
            shower.azim, shower.elev = raDec2AltAz(shower.ra, shower.dec, meteor_obj.jdt_ref, \
                config.latitude, config.longitude)


            # Take the shower that's closest to the great circle if there are multiple candidates
            if radiant_separation < best_match_dist:
                best_match_dist = radiant_separation
                best_match_shower = copy.deepcopy(shower)



        # Store the associated shower
        associations[(ff_name, meteor_No)] = [meteor_obj, best_match_shower]



    # Create a plot of showers
    if plot_showers:


        # Init the all-sky plot
        allsky_plot = AllSkyPlot()

        # Plot all meteors
        for key in associations:

            meteor_obj, shower = associations[key]


            ### Plot the observed meteor points ###

            allsky_plot.plot(meteor_obj.ra_array, meteor_obj.dec_array, color='r', linewidth=1, zorder=4)

            # Plot the peak of shower meteors a different color
            peak_color = 'blue'
            if shower is not None:
                peak_color = 'tomato'

            allsky_plot.scatter(meteor_obj.ra_array[-1], meteor_obj.dec_array[-1], c=peak_color, marker='+', \
                s=5, zorder=5)

            ### ###


            ### Plot fitted great circle points ###

            # Find the GC phase angle of the beginning of the meteor
            gc_beg_phase = meteor_obj.findGCPhase(meteor_obj.ra_array[0], meteor_obj.dec_array[0])[0]%360

            # If the meteor belongs to a shower, find the GC phase which ends at the shower
            if shower is not None:
                gc_end_phase = meteor_obj.findGCPhase(shower.ra, shower.dec)[0]%360

                # Fix 0/360 wrap
                if abs(gc_end_phase - gc_beg_phase) > 180:
                    gc_beg_phase -= 360

                gc_color = 'purple'
                gc_alpha = 0.7


            else:

                # If it's a sporadic, find the direction to which the meteor should extend
                gc_end_phase = meteor_obj.findGCPhase(meteor_obj.ra_array[-1], \
                    meteor_obj.dec_array[-1])[0]%360

                # Find the correct direction
                if (gc_beg_phase - gc_end_phase)%360 > (gc_end_phase - gc_beg_phase)%360:
                    gc_end_phase = gc_beg_phase - 170

                else:
                    gc_end_phase = gc_beg_phase + 170

                gc_color = 'green'
                gc_alpha = 0.5


            # Get phases 180 deg before the meteor
            phase_angles = np.linspace(gc_end_phase, gc_beg_phase, 100)%360

            # Compute RA/Dec of points on the great circle
            ra_gc, dec_gc = meteor_obj.sampleGC(phase_angles)

            # Cull all points below the horizon
            azim_gc, elev_gc = raDec2AltAz_vect(ra_gc, dec_gc, meteor_obj.jdt_ref, config.latitude, \
                config.longitude)
            temp_arr = np.c_[ra_gc, dec_gc]
            temp_arr = temp_arr[elev_gc > 0]
            ra_gc, dec_gc = temp_arr.T

            # Plot the great circle fitted on the radiant
            allsky_plot.plot(ra_gc, dec_gc, linestyle='dotted', color=gc_color, alpha=gc_alpha, linewidth=1)

            # Plot the point closest to the shower radiant
            if shower is not None:
                allsky_plot.plot(ra_gc[0], dec_gc[0], color='r', marker='+', ms=5, mew=1)


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

            # Compute coordinates on a circle around the given RA, Dec
            ra_circle, dec_circle = sphericalPointFromHeadingAndDistance(shower.ra, shower.dec, \
                heading_arr, config.shower_max_radiant_separation)


            # Plot the shower circle
            allsky_plot.plot(ra_circle, dec_circle)


            # Plot the shower name
            x_text, y_text = allsky_plot.raDec2XY(shower.ra, shower.dec)
            allsky_plot.ax.text(x_text, y_text, shower.name, color='w', size=8, va='center', \
                ha='center', zorder=6)

        ### ###
                


        allsky_plot.show()



    return associations






if __name__ == "__main__":

    import argparse

    import RMS.ConfigReader as cr


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Perform single-station shower association on the FTPdetectinfo file.")

    arg_parser.add_argument('ftpdetectinfo_path', nargs=1, metavar='FTPDETECTINFO_PATH', type=str, \
        help='Path to the FF file.')

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    ftpdetectinfo_path = cml_args.ftpdetectinfo_path[0]

    # Check if the given FTPdetectinfo file exists
    if not os.path.isfile(ftpdetectinfo_path):
        print('No such file:', ftpdetectinfo_path)
        sys.exit()


    # Extract parent directory
    dir_path = os.path.dirname(ftpdetectinfo_path)


    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, dir_path)



    # Perform shower association
    associations = showerAssociation(config, ftpdetectinfo_path, plot_showers=True)


    # Find shower frequency and sort by count
    shower_list = []
    for key in associations:
        _, shower = associations[key]

        if shower is None:
            shower_name = '...'
        else:
            shower_name = shower.name

        shower_list.append(shower_name)

    unique_showers = set(shower_list)
    shower_counts = [[name, shower_list.count(name)] for name in unique_showers]
    shower_counts = sorted(shower_counts, key=lambda x: x[1], reverse=True)

    print('Shower ranking:')
    for shower_name, count in shower_counts:
        print(shower_name, count)



