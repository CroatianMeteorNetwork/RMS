""" Convert the FTPdetectinfo format to UFOorbit input CSV. """

from __future__ import print_function, division, absolute_import

from __future__ import print_function, division, absolute_import

import os
import argparse
import datetime

import numpy as np

from RMS.Astrometry import ApplyAstrometry
from RMS.Formats import FTPdetectinfo
from RMS.Formats import FFfile
from RMS.Formats import Platepar
from RMS.Formats import UFOOrbit
from RMS import Math

from RMS.Routines import GreatCircle



def FTPdetectinfo2UFOOrbitInput(dir_path, file_name, platepar_path, platepar_dict=None):
    """ Convert the FTPdetectinfo file into UFOOrbit input CSV file. 
        
    Arguments:
        dir_path: [str] Path of the directory which contains the FTPdetectinfo file.
        file_name: [str] Name of the FTPdetectinfo file.
        platepar_path: [str] Full path to the platepar file.

    Keyword arguments:
        platepar_dict: [dict] Dictionary of Platepar instances where keys are FF file names. This will be 
            used instead of the platepar at platepar_path. None by default.
    """

    # Load the FTPdetecinfo file
    meteor_list = FTPdetectinfo.readFTPdetectinfo(dir_path, file_name)


    # Load the platepar file
    if platepar_dict is None:

        pp = Platepar.Platepar()
        pp.read(platepar_path)


    # Init the UFO format list
    ufo_meteor_list = []

    # Go through every meteor in the list
    for meteor in meteor_list:

        ff_name, cam_code, meteor_No, n_segments, fps, hnr, mle, binn, px_fm, rho, phi, \
            meteor_meas = meteor

        # Load the platepar from the platepar dictionary, if given
        if platepar_dict is not None:
            if ff_name in platepar_dict:
                pp = platepar_dict[ff_name]

            else:
                print('Skipping {:s} becuase no platepar was found for this FF file!'.format(ff_name))

        # Convert the FF file name into time
        dt = FFfile.filenameToDatetime(ff_name)

        # Extract measurements
        calib_status, frame_n, x, y, ra, dec, azim, elev, inten, mag = np.array(meteor_meas).T

        # If the meteor wasn't calibrated, skip it
        if not np.all(calib_status):
            print('Meteor {:d} was not calibrated, skipping it...'.format(meteor_No))
            continue

        # Compute the peak magnitude
        peak_mag = np.min(mag)

        # Compute the total duration
        first_frame = np.min(frame_n)
        last_frame = np.max(frame_n) 
        duration = (last_frame - first_frame)/fps


        # Compute times of first and last points
        dt1 = dt + datetime.timedelta(seconds=first_frame/fps)
        dt2 = dt + datetime.timedelta(seconds=last_frame/fps)

        
        ### Fit a great circle to Az/Alt measurements and compute model beg/end RA and Dec ###

        # Convert the measurement Az/Alt to cartesian coordinates
        # NOTE: All values that are used for Great Circle computation are:
        #   theta - the zenith angle (90 deg - altitude)
        #   phi - azimuth +N of due E, which is (90 deg - azim)
        x, y, z = Math.polarToCartesian(np.radians((90 - azim)%360), np.radians(90 - elev))

        # Fit a great circle
        C, theta0, phi0 = GreatCircle.fitGreatCircle(x, y, z)

        # Get the first point on the great circle
        phase1 = GreatCircle.greatCirclePhase(np.radians(90 - elev[0]), np.radians((90 - azim[0])%360), \
            theta0, phi0)
        alt1, azim1 = Math.cartesianToPolar(*GreatCircle.greatCircle(phase1, theta0, phi0))
        alt1 = 90 - np.degrees(alt1)
        azim1 = (90 - np.degrees(azim1))%360



        # Get the last point on the great circle
        phase2 = GreatCircle.greatCirclePhase(np.radians(90 - elev[-1]), np.radians((90 - azim[-1])%360),\
            theta0, phi0)
        alt2, azim2 = Math.cartesianToPolar(*GreatCircle.greatCircle(phase2, theta0, phi0))
        alt2 = 90 - np.degrees(alt2)
        azim2 = (90 - np.degrees(azim2))%360

        # Compute RA/Dec from Alt/Az
        _, ra1, dec1 = ApplyAstrometry.altAz2RADec(pp.lat, pp.lon, pp.UT_corr, [dt1], [azim1], [alt1], \
            dt_time=True)
        _, ra2, dec2 = ApplyAstrometry.altAz2RADec(pp.lat, pp.lon, pp.UT_corr, [dt2], [azim2], [alt2], \
            dt_time=True)


        ### ###


        ufo_meteor_list.append([dt1, peak_mag, duration, azim1[0], alt1[0], azim2[0], alt2[0], \
            ra1[0][0], dec1[0][0], ra2[0][0], dec2[0][0], cam_code, pp.lon, pp.lat, pp.elev, pp.UT_corr])


    # Construct a file name for the UFO file, which is the FTPdetectinfo file without the FTPdetectinfo 
    #   part
    ufo_file_name = file_name.replace('FTPdetectinfo_', '').replace('.txt', '') + '.csv'

    # Write the UFOorbit file
    UFOOrbit.writeUFOOrbit(dir_path, ufo_file_name, ufo_meteor_list)





if __name__ == "__main__":


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Converts the given FTPdetectinfo file into UFOorbit input format.")

    arg_parser.add_argument('file_path', nargs='+', metavar='FILE_PATH', type=str, \
        help='Path to one or more FTPdetectinfo files.')

    arg_parser.add_argument('platepar', nargs=1, metavar='PLATEPAR', type=str, \
        help='Path to the platepar file.')

    
    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    # Go though all FTPdetectinfo files and convert them to UFOOrbit input files
    for file_path in cml_args.file_path:

        dir_path, file_name = os.path.split(file_path)

        FTPdetectinfo2UFOOrbitInput(dir_path, file_name, cml_args.platepar[0])
            




