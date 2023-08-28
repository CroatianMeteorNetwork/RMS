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

import numpy as np

# RMS imports
import RMS.ConfigReader as cr
from RMS.Formats import FTPdetectinfo
from RMS.Formats import CALSTARS
from RMS.Formats.FFfile import validFFName
from RMS.Formats.FrameInterface import detectInputType
from RMS.ExtractStars import extractStars
from RMS.Detection import detectMeteors
from RMS.DetectionTools import loadImageCalibration
from RMS.QueuedPool import QueuedPool


# Get the logger from the main module
log = logging.getLogger("logger")


def detectStarsAndMeteors(ff_directory, ff_name, config, flat_struct=None, dark=None, mask=None):
    """ Run the star extraction and subsequently runs meteor detection on the FF file if there are enough
        stars on the image.

    Arguments:
        ff_directory: [str] path to the directory where the FF files are located.
        ff_name: [str] name of the FF file.
        config: [Configuration object] configuration object.

    Keyword arguments:
        flat_struct: [Flat struct] Structure containing the flat field. None by default.
        dark: [ndarray]
        mask: [MaskStruct]

    Return:
        [ff_name, star_list, meteor_list] detected stars and meteors

    """

    log.info('Running detection on file: ' + ff_name)


    # Construct the image handle for the detection
    img_handle = detectInputType(os.path.join(ff_directory, ff_name), config, skip_ff_dir=True, \
        detection=True)

    
    # If the FF file could not be loaded, skip processing
    if img_handle.input_type == 'ff':

        # If the FF file could not be loaded, skip it
        if img_handle.ff is None:
            return ff_name, [[], [], [], []], []



    # Load mask, dark, flat
    mask, dark, flat_struct = loadImageCalibration(ff_directory, config, dtype=img_handle.ff.dtype, \
        byteswap=img_handle.byteswap)


    # Run star extraction
    star_list = extractStars(ff_directory, ff_name, config, flat_struct=flat_struct, dark=dark, mask=mask)


    log.info('Detected stars: ' + str(len(star_list[1])))


    # Run meteor detection if there are enough stars on the image
    if len(star_list[1]) >= config.ff_min_stars:

        log.debug('At least ' + str(config.ff_min_stars) + ' stars, detecting meteors...')

        # Run the detection
        meteor_list = detectMeteors(img_handle, config, flat_struct=flat_struct, dark=dark, mask=mask)

        log.info(ff_name + ' detected meteors: ' + str(len(meteor_list)))

    else:
        meteor_list = []



    return ff_name, star_list, meteor_list



def saveDetections(detection_results, ff_dir, config):
    """ Save detection to CALSTARS and FTPdetectinfo files. 
    
    Arguments:
        detection_results: [list] A list of outputs from detectStarsAndMeteors function.
        ff_dir: [str] Path to the night directory.
        config: [Config obj]

    Return:
        calstars_name: [str] Name of the CALSTARS file.
        ftpdetectinfo_name: [str] Name of the FTPdetectinfo file.
        ff_detected: [list] A list of FF files with detections.
    """


    ### SAVE DETECTIONS TO FILE

    # Init data lists
    star_list = []
    meteor_list = []
    ff_detected = []

    # Remove all 'None' results, which were errors
    detection_results = [res for res in detection_results if res is not None]

    # Sort by FF name
    detection_results = sorted(detection_results, key=lambda x: x[0])


    # Count the number of detected meteors
    meteors_num = 0
    for _, _, meteor_data in detection_results:
        for meteor in meteor_data:
            meteors_num += 1

    log.info('TOTAL: ' + str(meteors_num) + ' detected meteors.')


    # Save the detections to a file
    for ff_name, star_data, meteor_data in detection_results:


        if len(star_data) == 4:
            x2, y2, background, intensity = star_data
            fwhm = (np.zeros_like(x2) - 1).tolist()
        else:
            _, x2, y2, background, intensity, fwhm = star_data
            

        # Skip if no stars were found
        if not x2:
            continue

        # Construct the table of the star parameters
        star_data = zip(y2, x2, background, intensity, fwhm)

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



    dir_name = os.path.basename(os.path.abspath(ff_dir))
    if dir_name.startswith(config.stationID):
        prefix = dir_name
    else:
        prefix = "{:s}_{:s}".format(config.stationID, dir_name)

    # Generate the name for the CALSTARS file
    calstars_name = 'CALSTARS_' + prefix + '.txt'

    # Write detected stars to the CALSTARS file
    CALSTARS.writeCALSTARS(star_list, ff_dir, calstars_name, config.stationID, config.height, 
        config.width)

    # Generate FTPdetectinfo file name
    ftpdetectinfo_name = 'FTPdetectinfo_' + os.path.basename(ff_dir) + '.txt'

    # Write FTPdetectinfo file
    FTPdetectinfo.writeFTPdetectinfo(meteor_list, ff_dir, ftpdetectinfo_name, ff_dir, 
        config.stationID, config.fps)


    return calstars_name, ftpdetectinfo_name, ff_detected




def detectStarsAndMeteorsDirectory(dir_path, config):
    """ Extract stars and detect meteors on all FF files in the given folder. 

    Arguments:
        dir_path: [str] Path to the directory with FF files.
        config: [Config obj]

    Return:
        calstars_name: [str] Name of the CALSTARS file.
        ftpdetectinfo_name: [str] Name of the FTPdetectinfo file.
        ff_detected: [list] A list of FF files with detections.
    """

    # Get paths to every FF bin file in a directory 
    ff_dir = dir_path
    ff_dir = os.path.abspath(ff_dir)
    ff_list = [ff_name for ff_name in sorted(os.listdir(ff_dir)) if validFFName(ff_name)]


    # Check if there are any file in the directory
    if not len(ff_list):

        log.info("No files for processing found!")
        return None, None, None, None


    log.info('Starting detection...')

    # Initialize the detector
    detector = QueuedPool(detectStarsAndMeteors, cores=-1, log=log, backup_dir=ff_dir, \
        input_queue_maxsize=None)

    # Start the detection
    detector.startPool()

    # Give detector jobs
    for ff_name in ff_list:

        while True:
            
            # Add a job as long as there are available workers to receive it
            if detector.available_workers.value() > 0:
                log.info('Adding for detection: {}'.format(ff_name))
                detector.addJob([ff_dir, ff_name, config], wait_time=0)
                break
            else:
                time.sleep(0.1)



    log.info('Waiting for the detection to finish...')

    # Wait for the detector to finish and close it
    detector.closePool()

    log.info('Detection finished!')

    log.info('Collecting results...')

    # Get the detection results from the queue
    detection_results = detector.getResults()


    # Save detection to disk
    calstars_name, ftpdetectinfo_name, ff_detected = saveDetections(detection_results, ff_dir, config)


    return calstars_name, ftpdetectinfo_name, ff_detected, detector





if __name__ == "__main__":

    time_start = datetime.datetime.utcnow()


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

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, cml_args.dir_path)


    ### Init the logger

    from RMS.Logger import initLogging
    initLogging(config, 'detection_')

    log = logging.getLogger("logger")

    ######


    # Run detection on the folder
    _, _, _, detector = detectStarsAndMeteorsDirectory(cml_args.dir_path[0], config)

    # Delete backup files
    detector.deleteBackupFiles()

    log.info('Total time taken: {}'.format(datetime.datetime.utcnow() - time_start))
