""" Perform star extraction and meteor detection on a given folder, archive detections, upload to server. """


from __future__ import print_function, division, absolute_import


import os
import argparse
import logging

import scipy.misc

from RMS.ArchiveDetections import archiveDetections, archiveFieldsums
from RMS.Astrometry.ApplyAstrometry import applyAstrometryFTPdetectinfo
from RMS.Astrometry.CheckFit import autoCheckFit
import RMS.ConfigReader as cr
from RMS.DownloadPlatepar import downloadNewPlatepar
from RMS.DetectStarsAndMeteors import detectStarsAndMeteorsDirectory, saveDetections
from RMS.Formats.CAL import writeCAL
from RMS.Formats.FTPdetectinfo import readFTPdetectinfo, writeFTPdetectinfo
from RMS.Formats.Platepar import Platepar
from RMS.Formats import CALSTARS
from RMS.UploadManager import UploadManager
from Utils.MakeFlat import makeFlat
from Utils.PlotFieldsums import plotFieldsums
from Utils.RMS2UFO import FTPdetectinfo2UFOOrbitInput



# Get the logger from the main module
log = logging.getLogger("logger")



def getPlatepar(config):
    """ Downloads a new platepar from the server of uses an existing one. """


    # Download a new platepar from the server, if present
    downloadNewPlatepar(config)


    # Load the default platepar if it is available
    platepar = None
    platepar_fmt = None
    platepar_path = os.path.join(os.getcwd(), config.platepar_name)
    if os.path.exists(platepar_path):
        platepar = Platepar()
        platepar_fmt = platepar.read(platepar_path)

        log.info('Loaded platepar: ' + platepar_path)

    else:

        log.info('No platepar file found!')


    return platepar, platepar_path, platepar_fmt




def processNight(night_data_dir, config, detection_results=None, nodetect=False):
    """ Given the directory with FF files, run detection and archiving. 
    
    Arguments:
        night_data_dir: [str] Path to the directory with FF files.
        config: [Config obj]

    Keyword arguments:
        detection_results: [list] An optional list of detection. If None (default), detection will be done
            on the the files in the folder.
        nodetect: [bool] True if detection should be skipped. False by default.

    Return:
        archive_name: [str] Path to the archive.
    """

    # Remove final slash in the night dir
    if night_data_dir.endswith(os.sep):
        night_data_dir = night_data_dir[:-1]

    # Extract the name of the night
    night_data_dir_name = os.path.basename(night_data_dir)
    
    # If the detection should be run
    if (not nodetect):

        # If no detection was performed, run it
        if detection_results is None:

            # Run detection on the given directory
            calstars_name, ftpdetectinfo_name, ff_detected, \
                detector = detectStarsAndMeteorsDirectory(night_data_dir, config)

        # Otherwise, save detection results
        else:

            # Save CALSTARS and FTPdetectinfo to disk
            calstars_name, ftpdetectinfo_name, ff_detected = saveDetections(detection_results, \
                night_data_dir, config)

            # If the files were previously detected, there is no detector
            detector = None


        # Get the platepar file
        platepar, platepar_path, platepar_fmt = getPlatepar(config)


        # Run calibration check and auto astrometry refinement
        if platepar is not None:

            # Read in the CALSTARS file
            calstars_list = CALSTARS.readCALSTARS(night_data_dir, calstars_name)

            # Run astrometry check and refinement
            platepar, fit_status = autoCheckFit(config, platepar, calstars_list)

            # If the fit was sucessful, apply the astrometry to detected meteors
            if fit_status:

                log.info('Astrometric calibration SUCCESSFUL!')

                # Save the refined platepar to the night directory and as default
                platepar.write(os.path.join(night_data_dir, config.platepar_name), fmt=platepar_fmt)
                platepar.write(platepar_path, fmt=platepar_fmt)

            else:
                log.info('Astrometric calibration FAILED!, Using old platepar for calibration...')    


            # Calculate astrometry for meteor detections
            applyAstrometryFTPdetectinfo(night_data_dir, ftpdetectinfo_name, platepar_path)


            # Convert the FTPdetectinfo into UFOOrbit input file
            FTPdetectinfo2UFOOrbitInput(night_data_dir, ftpdetectinfo_name, platepar_path)



    log.info('Plotting field sums...')

    # Plot field sums to a graph
    plotFieldsums(night_data_dir, config)

    # Archive all fieldsums to one archive
    archiveFieldsums(night_data_dir)


    # List for any extra files which will be copied to the night archive directory. Full paths have to be 
    #   given
    extra_files = []


    log.info('Making a flat...')

    # Make a new flat field
    flat_img = makeFlat(night_data_dir, config)

    # If making flat was sucessfull, save it
    if flat_img is not None:

        # Save the flat in the root directory, to keep the operational flat updated
        scipy.misc.imsave(config.flat_file, flat_img)
        flat_path = os.path.join(os.getcwd(), config.flat_file)
        log.info('Flat saved to: ' + flat_path)

        # Copy the flat to the night's directory as well
        extra_files.append(flat_path)

    else:
        log.info('Making flat image FAILED!')


    # Make a CAL file if full CAMS compatibility is desired
    if config.cams_code > 0:

        # Write the CAL file to disk
        cal_file_name = writeCAL(night_data_dir, config, platepar)

        # Load the FTPdetectinfo
        cam_code, fps, meteor_list = readFTPdetectinfo(night_data_dir, ftpdetectinfo_name, ret_input_format=True)

        # Write the CAL file in FTPdetectinfo
        writeFTPdetectinfo(meteor_list, night_data_dir, ftpdetectinfo_name, night_data_dir, \
            cam_code, fps, calibration=cal_file_name, celestial_coords_given=(platepar is not None))
        


    ### Add extra files to archive

    # Add the platepar to the archive if it exists
    if os.path.exists(platepar_path):
        extra_files.append(platepar_path)


    # Add the config file to the archive too
    extra_files.append(os.path.join(os.getcwd(), '.config'))


    ### ###


    night_archive_dir = os.path.join(os.path.abspath(config.data_dir), config.archived_dir, 
        night_data_dir_name)


    log.info('Archiving detections to ' + night_archive_dir)
    
    # Archive the detections
    archive_name = archiveDetections(night_data_dir, night_archive_dir, ff_detected, config, \
        extra_files=extra_files)


    return archive_name, detector




if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Reprocess the given folder, perform detection, archiving and server upload.")

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


    
    ### Init the logger

    from RMS.Logger import initLogging
    initLogging('reprocess_')

    log = logging.getLogger("logger")

    ######


    # Process the night
    archive_name, detector = processNight(cml_args.dir_path[0], config)


    # Upload the archive, if upload is enabled
    if config.upload_enabled:

        # Init the upload manager
        print('Starting the upload manager...')

        upload_manager = UploadManager(config)
        upload_manager.start()

        # Add file for upload
        print('Adding file on upload list: ' + archive_name)
        upload_manager.addFiles([archive_name])

        # Stop the upload manager
        if upload_manager.is_alive():
            upload_manager.stop()
            print('Closing upload manager...')


        # Delete detection backup files
        if detector is not None:
            detector.deleteBackupFiles()