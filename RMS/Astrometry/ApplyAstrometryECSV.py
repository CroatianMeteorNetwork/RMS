""" Apply a platepar conversion to measurements from an ECSV files. """

import os
import shutil
import datetime

import numpy as np
from astropy.table import QTable

import RMS.Formats.Platepar
from RMS.Astrometry.Conversions import trueRaDec2ApparentAltAz
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP, computeFOVSize, rotationWrtHorizon


ECSV_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"



def applyAstrometryECSV(dir_path, ecsv_file, platepar_file, platepar=None, time_corr=None):
    """ Use the given platepar to calculate the celestial coordinates of detected meteors from an ECSV
        file and save the updated values.

    Arguments:
        dir_path: [str] Path to the directory with the ECSV file.
        ecsv_file: [str] Name of the ECSV file.
        platepar_file: [str] Name of the platepar file.
    Keyword arguments:
        platepar: [Platepar obj] Loaded platepar. None by default. If given, the platepar file won't be read,
            but this platepar structure will be used instead.
        time_corr: [float] Apply a time correction to the time in the file. If none, the time will not be
            changed.
    Return:
        None
    """



    # If the ECSV file does not exist, skip everything
    if not os.path.isfile(os.path.join(dir_path, ecsv_file)):
        print('The given ECSV file does not exist:', os.path.join(dir_path, ecsv_file))
        print('The astrometry was not computed!')
        return None

    # Save a copy of the uncalibrated ECSV
    ecsv_copy = "".join(ecsv_file.split('.')[:-1]) + "_uncalibrated.ecsv"

    # Back up the original ECSV, only if a backup does not exist already
    if not os.path.isfile(os.path.join(dir_path, ecsv_copy)):
        shutil.copy2(os.path.join(dir_path, ecsv_file), os.path.join(dir_path, ecsv_copy))

    # Load platepar from file if not given
    if platepar is None:

        # Load the platepar
        platepar = RMS.Formats.Platepar.Platepar()
        platepar.read(os.path.join(dir_path, platepar_file), use_flat=None)


    # Load the ECSV file
    with open(os.path.join(dir_path, ecsv_file)) as f:
        ecsv_content = f.readlines()

        data = QTable.read(ecsv_content, format='ascii.ecsv')


    # Compute alt/az pointing
    azim, elev = trueRaDec2ApparentAltAz(platepar.RA_d, platepar.dec_d, platepar.JD, \
        platepar.lat, platepar.lon, refraction=False)

    # Compute FOV size
    fov_horiz, fov_vert = computeFOVSize(platepar)



    data.meta['obs_latitude'] = platepar.lat                        # Decimal signed latitude (-90 S to +90 N)
    data.meta['obs_longitude'] = platepar.lon                       # Decimal signed longitude (-180 W to +180 E)
    data.meta['obs_elevation'] = platepar.elev                      # Altitude in metres above MSL. Note not WGS84
    data.meta['camera_id'] = platepar.station_code                  # The code name of the camera, likely to be network-specific
    data.meta['cx'] = platepar.X_res                               # Horizontal camera resolution in pixels
    data.meta['cy'] = platepar.Y_res                               # Vertical camera resolution in pixels
    data.meta['astrometry_number_stars'] = len(platepar.star_list) # The number of stars identified and used in the astrometric calibration
    data.meta['mag_label'] = 'mag'                                       # The label of the Magnitude column in the Point Observation data
    data.meta['no_frags'] = 1                                            # The number of meteoroid fragments described in this data
    data.meta['obs_az'] = azim                                           # The azimuth of the centre of the field of view in decimal degrees. North = 0, increasing to the East
    data.meta['obs_ev'] = elev                                           # The elevation of the centre of the field of view in decimal degrees. Horizon =0, Zenith = 90
    data.meta['obs_rot'] = rotationWrtHorizon(platepar)             # Rotation of the field of view from horizontal, decimal degrees. Clockwise is positive
    data.meta['fov_horiz'] = fov_horiz                                   # Horizontal extent of the field of view, decimal degrees
    data.meta['fov_vert'] = fov_vert                                     # Vertical extent of the field of view, decimal degrees

    
    ### Extract input parameters ###
    
    # Get the time in the appropriate format
    dt_data = data['datetime']
    time_data = []
    dt_updated = []
    for dt_str in dt_data:

        # Read in the datetime
        dt = datetime.datetime.strptime(dt_str, ECSV_DATETIME_FORMAT)

        # Apply a time correction if necessary
        if time_corr is not None:
            dt += datetime.timedelta(seconds=time_corr)

        dt_updated.append(dt)
        time_data.append([dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond/1000])

    time_data = np.array(time_data)

    # Extract image coordinates
    x_picks = np.array(data['x_image'])
    y_picks = np.array(data['y_image'])

    # Check if the raw pixel values are given for magnitude computation
    mag_write = True
    if 'integrated_pixel_value' in data.columns:
        px_intens_data = data['integrated_pixel_value']

    else:
        px_intens_data = np.ones_like(x_picks)

        # If the intensity data is not given and the magnitudes already exist, don't write them again
        mag_write = False


    ###

    
    # Compute RA/Dec using the platepar file
    jd_data, ra_data, dec_data, mag_data = xyToRaDecPP(time_data, x_picks, y_picks, px_intens_data, platepar, 
        measurement=True)

    # Compute alt/az (topocentric, i.e. without refraction)
    azim_data, alt_data = trueRaDec2ApparentAltAz(ra_data, dec_data, jd_data, platepar.lat, platepar.lon, \
        refraction=False)


    # Update the dataframe
    data['datetime'] = [dt.strftime(ECSV_DATETIME_FORMAT) for dt in dt_updated]
    data['JD']       = jd_data
    data['ra']       = ra_data
    data['dec']      = dec_data
    data['azimuth']  = azim_data
    data['altitude'] = alt_data

    # Add magnitude information if raw intensities were available
    if mag_write:
        data['mag_data'] = mag_data


    # Save the updated ECSV file to disk
    data.write(os.path.join(dir_path, ecsv_file), format='ascii.ecsv', delimiter=',', overwrite=True)




if __name__ == "__main__":

    import sys
    import argparse

    ### COMMAND LINE ARGUMENTS
    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Apply the platepar to the given ECSV file.")

    arg_parser.add_argument('ecsv_path', metavar='ECSV_PATH', type=str, help='Path to the FF file.')

    arg_parser.add_argument('-t', '--timecorr', metavar='TIME_CORRECTION', type=float,
                            help="Apply a time correction in second. The time will be added to the time in the file.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    ecsv_path = cml_args.ecsv_path

    # Extract the directory path
    dir_path, ecsv_file = os.path.split(os.path.abspath(ecsv_path))

    if not ecsv_file.endswith('.ecsv'):
        print("Please provide a ECSV file! It has to end with .ecsv")
        sys.exit()

    # Find the platepar file
    platepar_file = None
    for file_name in os.listdir(dir_path):
        if 'platepar_' in file_name:
            platepar_file = file_name
            print("Using platepar:", platepar_file)
            break

    if platepar_file is None:
        print('ERROR! Could not find the platepar file!')
        sys.exit()


    # Apply the astrometry to the given ECSV file
    applyAstrometryECSV(dir_path, ecsv_file, platepar_file, time_corr=cml_args.timecorr)