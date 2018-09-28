# RPi Meteor Station
# Copyright (C) 2016  Dario Zubovic, Denis Vida
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

import sys
import os
import time
import datetime
import logging
import argparse

# RMS imports
import RMS.ConfigReader as cr
from RMS.Formats import FTPdetectinfo
from RMS.Formats import CALSTARS
from RMS.Formats.FFfile import validFFName
from RMS.ExtractStars import extractStars
from RMS.Detection import detectMeteors
from RMS.QueuedPool import QueuedPool
from RMS.Routines import Image


# Get the logger from the main module
log = logging.getLogger("logger")


def detectStarsAndMeteors(ff_directory, ff_name, config, flat_struct=None):
    """ Run the star extraction and subsequently runs meteor detection on the FF bin file if there are enough
        stars on the image.

    Arguments:
        ff_directory: [str] path to the directory where the FF files are located.
        ff_name: [str] name of the FF file.
        config: [Configuration object] configuration object.

    Keyword arguments:
        flat_struct: [Flat struct] Structure containing the flat field. None by default.

    Return:
        [ff_name, star_list, meteor_list] detected stars and meteors

    """

    log.info('Running detection on file: ' + ff_name)

    # Run star extraction on the FF bin
    star_list = extractStars(ff_directory, ff_name, config, flat_struct=flat_struct)

    log.info('Detected stars: ' + str(len(star_list[0])))

    # Run meteor detection if there are enough stars on the image
    if len(star_list[0]) >= config.ff_min_stars:

        log.debug('More than ' + str(config.ff_min_stars) + ' stars, detecting meteors...')

        meteor_list = detectMeteors(ff_directory, ff_name, config, flat_struct=flat_struct)

        log.info(ff_name + ' detected meteors: ' + str(len(meteor_list)))

    else:
        meteor_list = []


    return ff_name, star_list, meteor_list





if __name__ == "__main__":

    time_start = datetime.datetime.utcnow()


    ### Init the logger

    from RMS.Logger import initLogging
    initLogging('detection_')

    log = logging.getLogger("logger")

    ######


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Detect stars and meteors in the given folder.")

    arg_parser.add_argument('dir_path', nargs=1, metavar='DIR_PATH', type=str, \
        help='Path to the folder with FF files.')

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    if cml_args.config is not None:

        config_file = os.path.abspath(cml_args.config[0].replace('"', ''))

        print('Loading config file:', config_file)

        # Load the given config file
        config = cr.parse(config_file)

    else:
        # Load the default configuration file
        config = cr.parse(".config")


    

    # Get paths to every FF bin file in a directory 
    ff_dir = cml_args.dir_path[0]
    ff_dir = os.path.abspath(ff_dir)
    ff_list = [ff_name for ff_name in sorted(os.listdir(ff_dir)) if validFFName(ff_name)]


    # Check if there are any file in the directory
    if(len(ff_list) == None):
        print("No files found!")
        sys.exit()


    # Try loading a flat field image
    flat_struct = None

    if config.use_flat:
        
        # Check if there is flat in the data directory
        if os.path.exists(os.path.join(ff_dir, config.flat_file)):
            flat_struct = Image.loadFlat(ff_dir, config.flat_file)

        # Try loading the default flat
        elif os.path.exists(config.flat_file):
            flat_struct = Image.loadFlat(os.getcwd(), config.flat_file)


    print('Starting detection...')

    # Initialize the detector
    detector = QueuedPool(detectStarsAndMeteors, cores=-1, log=log)

    # Give detector jobs
    for ff_name in ff_list:
        print('Adding for detection:', ff_name)
        detector.addJob([ff_dir, ff_name, config, flat_struct])


    # Start the detection
    detector.startPool()


    log.info('Waiting for the detection to finish...')

    # Wait for the detector to finish and close it
    detector.closePool()

    log.info('Detection finished!')

    ### SAVE DETECTIONS TO FILE

    # Init data lists
    star_list = []
    meteor_list = []
    ff_detected = []


    log.info('Collecting results...')

    # Get the detection results from the queue
    detection_results = detector.getResults()

    # Remove all 'None' results, which were errors
    detection_results = [res for res in detection_results if res is not None]

    # Count the number of detected meteors
    meteors_num = 0
    for _, _, meteor_data in detection_results:
        for meteor in meteor_data:
            meteors_num += 1

    log.info('TOTAL: ' + str(meteors_num) + ' detected meteors.')


    # Save the detections to a file
    for ff_name, star_data, meteor_data in detection_results:

        x2, y2, background, intensity = star_data

        # Skip if no stars were found
        if not x2:
            continue

        # Construct the table of the star parameters
        star_data = zip(x2, y2, background, intensity)

        # Add star info to the star list
        star_list.append([ff_name, star_data])

        # Handle the detected meteors
        meteor_No = 1
        for meteor in meteor_data:

            rho, theta, centroids = meteor

            # Append to the results list
            meteor_list.append([ff_name, meteor_No, rho, theta, centroids])
            meteor_No += 1


        # Add the FF file to the archive list if a meteor was detected on it
        if meteor_data:
            ff_detected.append(ff_name)


    # Generate the name for the CALSTARS file
    calstars_name = 'CALSTARS_' + "{:s}".format(str(config.stationID)) + '_' + os.path.basename(ff_dir) + '.txt'

    # Write detected stars to the CALSTARS file
    CALSTARS.writeCALSTARS(star_list, ff_dir, calstars_name, config.stationID, config.height, 
        config.width)

    # Generate FTPdetectinfo file name
    ftpdetectinfo_name = 'FTPdetectinfo_' + os.path.basename(ff_dir) + '.txt'

    # Write FTPdetectinfo file
    FTPdetectinfo.writeFTPdetectinfo(meteor_list, ff_dir, ftpdetectinfo_name, ff_dir, 
        config.stationID, config.fps)


    print('Total time taken: ', datetime.datetime.utcnow() - time_start)
