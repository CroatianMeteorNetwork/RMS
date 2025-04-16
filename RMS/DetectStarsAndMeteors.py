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
import gc
import os
import time
import argparse

import numpy as np

# RMS imports
import RMS.ConfigReader as cr
from RMS.Formats import FTPdetectinfo
from RMS.Formats import CALSTARS
from RMS.Formats.FFfile import validFFName, constructFFName
from RMS.Formats.FrameInterface import detectInputType, detectInputTypeFile, checkIfVideoFile
from RMS.ExtractStars import extractStarsFF
from RMS.ExtractStarsFrameInterface import extractStarsFrameInterface
from RMS.Detection import detectMeteors
from RMS.DetectionTools import loadImageCalibration
from RMS.QueuedPool import QueuedPool
from RMS.Logger import getLogger
from RMS.Misc import RmsDateTime


# Get the logger from the main module
log = getLogger("logger")



def detectStarsAndMeteorsFrameInterface(
        img_handle, config, 
        flat_struct=None, dark=None, mask=None, chunk_frames=128
        ):
    """ Extract stars and detect meteors on the given image handle. This is most useful for videos and 
        directories with images.

    Arguments:
        img_handle: [ImageHandle] Image handle object.
        config: [Configuration object] configuration object.

    Keyword arguments:
        flat_struct: [Flat struct] Structure containing the flat field. None by default.
        dark: [ndarray]
        mask: [MaskStruct]
        chunk_frames: [int] Number of frames to stacked image on which the stars will be extracted.

    Return:
        [img_handle, star_list, meteor_list]:
            - img_handle: [ImageHandle] Image handle object.
            - star_list: [list] List of stars detected in the image.
            - meteor_list: [list] List of detected meteors.
        
    """

    log.info('Running detection on file: ' + img_handle.file_name)
    
    # Load mask, dark, flat
    mask, dark, flat_struct = loadImageCalibration(img_handle.dir_path, config, dtype=img_handle.ff.dtype, 
                                                  byteswap=img_handle.byteswap)
    
    # Run star extraction on the image handle
    star_list = extractStarsFrameInterface(img_handle, config, chunk_frames=chunk_frames, 
        flat_struct=flat_struct, dark=dark, mask=mask, save_calstars=False)
    

    # Get the maximum number of stars on any chunks
    try:
        
        max_stars = max([len(star_entry[1]) for star_entry in star_list if len(star_entry) > 1]) \
                    if star_list else 0
    
    except (IndexError, TypeError, ValueError):
        max_stars = 0

    
    log.info('Max. detected stars on all frame chunks: {:d}'.format(max_stars))

    # Rewind the video to the beginning
    img_handle.setFrame(0)

    # Run meteor detection if there are enough stars on the image
    if max_stars >= config.ff_min_stars:
            
        log.debug('At least ' + str(config.ff_min_stars) + ' stars, detecting meteors...')
        
        # Run the detection
        meteor_list = detectMeteors(img_handle, config, flat_struct=flat_struct, dark=dark, mask=mask)
        
        log.info(img_handle.file_name + ' detected meteors: ' + str(len(meteor_list)))

    else:
        log.info('Not enough stars for meteor detection: {:d} < {:d}'.format(max_stars, config.ff_min_stars))
        meteor_list = []


    return star_list, meteor_list


def saveResultsFrameInterface(star_list, meteor_list, img_handle, config, chunk_frames=128, output_suffix=''):
    """ Save detection results to CALSTARS and FTPdetectinfo files.
    
    Arguments:
        star_list: [list] List of stars detected in the image.
        meteor_list: [list] List of detected meteors.
        img_handle: [ImageHandle] Image handle object.
        config: [Configuration object] configuration object.

    Keyword arguments:
        chunk_frames: [int] Number of frames to stacked image on which the stars will be extracted.
        output_suffix: [str] Suffix to add to the output files.

    Return:
        None
    
    """

     # Construct the name of the CALSTARS file by using the camera code and the time of the first frame
    timestamp = img_handle.beginning_datetime.strftime("%Y%m%d_%H%M%S_%f")
    prefix = "{:s}_{:s}".format(config.stationID, timestamp)

    suffix = ''
    if len(output_suffix):
        suffix = '_' + output_suffix

    # Generate the name for the CALSTARS file
    calstars_name = 'CALSTARS_' + prefix + suffix + '.txt'

    # Write detected stars to the CALSTARS file
    CALSTARS.writeCALSTARS(star_list, img_handle.dir_path, calstars_name, 
                        config.stationID, config.height, config.width, chunk_frames=chunk_frames)
    
    log.info("Stars extracted and written to {:s}".format(calstars_name))

    # Generate FTPdetectinfo file name
    ftpdetectinfo_name = 'FTPdetectinfo_' + prefix + suffix + '.txt'

    results_list = []
    meteor_No = 1
    for meteor in meteor_list:

        rho, theta, centroids = meteor

        # Get the time of the first frame in the detection
        first_pick_time = img_handle.currentFrameTime(frame_no=int(centroids[0][0]), dt_obj=True)

        # Construct FF file name if it's not available
        if img_handle.input_type == 'ff':
            ff_file_name = img_handle.name()

        # For non-FF inputs, construct the FF name from the station ID and the first pick time
        # To keep an accurate time, reset the frames so that the first pick is at frame 0
        else:
            ff_file_name = constructFFName(config.stationID, first_pick_time)

            # Reset the frame numbers so that the first pick is at frame 0 
            # frame[i] - int(frame[0]) to preserve the rolling shutter correction encoded as the 
            #   fractional part of the frame number
            centroids[:,0] -= int(centroids[0,0])

        # Append to the results list
        results_list.append([ff_file_name, meteor_No, rho, theta, centroids])

    # Write FTPdetectinfo file
    FTPdetectinfo.writeFTPdetectinfo(results_list, img_handle.dir_path, ftpdetectinfo_name, 
                                     img_handle.dir_path, config.stationID, config.fps)


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


    # Run star extraction on FF files
    star_list = extractStarsFF(ff_directory, ff_name, config=config, 
                               flat_struct=flat_struct, dark=dark, mask=mask)


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



def saveDetections(detection_results, ff_dir, config, output_suffix=''):
    """ Save detection to CALSTARS and FTPdetectinfo files. 
    
    Arguments:
        detection_results: [list] A list of outputs from detectStarsAndMeteors function.
        ff_dir: [str] Path to the night directory.
        config: [Config obj]

    Keyword arguments:
        output_suffix: [str] Suffix to add to the output files.

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
            
            amplitude = np.zeros_like(x2).tolist()
            fwhm = (np.zeros_like(x2) - 1).tolist()
            snr = np.ones_like(x2).tolist()
            saturated_count = np.zeros_like(x2).tolist()

        elif len(star_data) == 6:
            _, x2, y2, background, intensity, fwhm = star_data

            amplitude = np.zeros_like(x2).tolist()
            snr = np.ones_like(x2).tolist()
            saturated_count = np.zeros_like(x2).tolist()

        else:
            _, x2, y2, amplitude, intensity, fwhm, background, snr, saturated_count = star_data
            

        # Skip if no stars were found
        if not x2:
            continue

        # Construct the table of the star parameters
        star_data = zip(y2, x2, amplitude, intensity, fwhm, background, snr, saturated_count)

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

    suffix = ''
    if len(output_suffix):
        suffix = '_' + output_suffix

    # Generate the name for the CALSTARS file
    calstars_name = 'CALSTARS_' + prefix + suffix + '.txt'

    # Create the ff_dir if it doesn't exist
    if not os.path.exists(ff_dir):
        os.makedirs(ff_dir)

    # Write detected stars to the CALSTARS file
    CALSTARS.writeCALSTARS(star_list, ff_dir, calstars_name, config.stationID, config.height, 
        config.width)

    # Generate FTPdetectinfo file name
    ftpdetectinfo_name = 'FTPdetectinfo_' + os.path.basename(ff_dir) + suffix + '.txt'

    # Write FTPdetectinfo file
    FTPdetectinfo.writeFTPdetectinfo(meteor_list, ff_dir, ftpdetectinfo_name, ff_dir, 
        config.stationID, config.fps)


    return calstars_name, ftpdetectinfo_name, ff_detected




def detectStarsAndMeteorsDirectory(dir_path, config, output_suffix=''):
    """ Extract stars and detect meteors on all FF files in the given folder. 

    Arguments:
        dir_path: [str] Path to the directory with FF files.
        config: [Config obj]

    Keyword arguments:
        output_suffix: [str] Suffix to add to the output files.

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
    detector = QueuedPool(detectStarsAndMeteors, cores=config.num_cores, log=log, backup_dir=ff_dir, \
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
    calstars_name, ftpdetectinfo_name, ff_detected = saveDetections(detection_results, ff_dir, config, 
                                                                    output_suffix=output_suffix)


    return calstars_name, ftpdetectinfo_name, ff_detected, detector





if __name__ == "__main__":

    time_start = RmsDateTime.utcnow()


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Detect stars and meteors in the given folder.")

    arg_parser.add_argument('dir_path', metavar='DIR_PATH', type=str, \
        help='Path to the folder with FF files.')

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one.")
    
    arg_parser.add_argument('--multivid', action='store_true', \
        help="Flag to indicate that the data path is a directory containing multiple video files.")
    
    arg_parser.add_argument('--chunk_frames', type=int, default=128, \
        help="Number of frames to use to stack an image on which the stars will be extracted. Only "
        "applicable for non-FF files.")
    
    arg_parser.add_argument('--suffix', type=str, default='', \
        help="Suffix to add to the output files.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, cml_args.dir_path)


    ### Init the logger

    from RMS.Logger import initLogging
    initLogging(config, 'detection_', safedir=cml_args.dir_path)

    log = getLogger("logger")

    ######

    if cml_args.multivid:

        log.info('Running detection on a directory with multiple video files...')

        video_paths = []
        for file_name in sorted(os.listdir(cml_args.dir_path)):

            file_path = os.path.join(cml_args.dir_path, file_name)

            if checkIfVideoFile(file_path):
                video_paths.append(file_path)

        # Run the detection on each video file
        for video_path in video_paths:

            # Load the video file
            img_handle = detectInputTypeFile(video_path, config, detection=True, preload_video=True, 
                                             chunk_frames=cml_args.chunk_frames)

            # Load the calibration files
            mask, dark, flat_struct = loadImageCalibration(img_handle.dir_path, config, 
                dtype=img_handle.ff.dtype, byteswap=img_handle.byteswap)

            # Run detection on the video
            star_list, meteor_list = detectStarsAndMeteorsFrameInterface(img_handle, config, 
                flat_struct=flat_struct, dark=dark, mask=mask, chunk_frames=img_handle.chunk_frames)
            
            # Save the results
            saveResultsFrameInterface(star_list, meteor_list, img_handle, config, 
                chunk_frames=img_handle.chunk_frames, output_suffix=cml_args.suffix)
            
            # Release the video handle
            if img_handle.input_type == 'video':

                print("Releasing video handle... ", end="")
                img_handle.cap.release()
                print("Done!")
            
            # Delete the image handle to free up memory
            del img_handle

            # Collect garbage
            gc.collect()

    else:

        # Detect the file type
        img_handle = detectInputType(cml_args.dir_path, config, skip_ff_dir=False, detection=True, 
                                     chunk_frames=cml_args.chunk_frames)

        # If the directory contains FF files, run detection on them using a pool of workers
        if img_handle.input_type == 'ff':

            # Run detection on the folder
            _, _, _, detector = detectStarsAndMeteorsDirectory(cml_args.dir_path, config, 
                                                               output_suffix=cml_args.suffix)

            # Delete backup files
            detector.deleteBackupFiles()

            log.info('Total time taken: {}'.format(RmsDateTime.utcnow() - time_start))

        # Otherwise, run the detection on the given input type (e.g. directory with images)
        else:

            # Load the calibration files
            mask, dark, flat_struct = loadImageCalibration(img_handle.dir_path, config, 
                dtype=img_handle.ff.dtype, byteswap=img_handle.byteswap)

            # Run detection on the image
            star_list, meteor_list = detectStarsAndMeteorsFrameInterface(
                img_handle, config, flat_struct=flat_struct, dark=dark, mask=mask,
                chunk_frames=img_handle.chunk_frames
                )
            
            # Save the results
            saveResultsFrameInterface(star_list, meteor_list, img_handle, config, 
                chunk_frames=img_handle.chunk_frames, output_suffix=cml_args.suffix)
