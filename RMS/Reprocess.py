""" Perform star extraction and meteor detection on a given folder, archive detections, upload to server. """


from __future__ import print_function, division, absolute_import


import os
import sys
import traceback
import argparse
import logging

from RMS.ArchiveDetections import archiveDetections, archiveFieldsums
# from RMS.Astrometry.ApplyAstrometry import applyAstrometryFTPdetectinfo
from RMS.Astrometry.ApplyRecalibrate import recalibrateIndividualFFsAndApplyAstrometry
from RMS.Astrometry.CheckFit import autoCheckFit
import RMS.ConfigReader as cr
from RMS.DownloadPlatepar import downloadNewPlatepar
from RMS.DetectStarsAndMeteors import detectStarsAndMeteorsDirectory, saveDetections
from RMS.Formats.CAL import writeCAL
from RMS.Formats.FTPdetectinfo import readFTPdetectinfo, writeFTPdetectinfo
from RMS.Formats.Platepar import Platepar
from RMS.Formats import CALSTARS
from RMS.UploadManager import UploadManager
from RMS.Routines.Image import saveImage
from RMS.Routines.MaskImage import loadMask
from Utils.CalibrationReport import generateCalibrationReport
from Utils.FOVKML import fovKML
from Utils.MakeFlat import makeFlat
from Utils.PlotFieldsums import plotFieldsums
from Utils.RMS2UFO import FTPdetectinfo2UFOOrbitInput
from Utils.ShowerAssociation import showerAssociation


# Get the logger from the main module
log = logging.getLogger("logger")



def getPlatepar(config, night_data_dir):
    """ Downloads a new platepar from the server of uses an existing one. 
    
    Arguments:
        Config: [Config instance]
        night_data_dir: [str] Full path to the data directory.

    Return:
        platepar, platepar_path, platepar_fmt
    """


    # Download a new platepar from the server, if present
    downloadNewPlatepar(config)


    # Construct path to the platepar in the night directory
    platepar_night_dir_path = os.path.join(night_data_dir, config.platepar_name)

    # Load the default platepar from the RMS if it is available
    platepar = None
    platepar_fmt = None
    platepar_path = os.path.join(os.getcwd(), config.platepar_name)
    if os.path.exists(platepar_path):
        platepar = Platepar()
        platepar_fmt = platepar.read(platepar_path, use_flat=config.use_flat)

        log.info('Loaded platepar from RMS directory: ' + platepar_path)


    # Otherwise, try to find the platepar in the data directory
    elif os.path.exists(platepar_night_dir_path):

        platepar_path = platepar_night_dir_path

        platepar = Platepar()
        platepar_fmt = platepar.read(platepar_path, use_flat=config.use_flat)

        log.info('Loaded platepar from night directory: ' + platepar_path)

    else:

        log.info('No platepar file found!')


    if platepar is not None:
        
        # Make sure that the station code from the config and the platepar match
        if platepar.station_code is not None:
            if config.stationID != platepar.station_code:

                # If they don't match, don't use this platepar
                log.info("The station code in the platepar doesn't match the station code in config file! Not using the platepar...")

                platepar = None
                platepar_fmt = None


    # Make sure the image resolution matches
    if platepar is not None:
        if (int(config.width) != int(platepar.X_res)) or (int(config.height) != int(platepar.Y_res)):

            # If they don't match, don't use this platepar
            log.info("The image resolution in config and platepar don't match! Not using the platepar...")

            platepar = None
            platepar_fmt = None

        

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
        night_archive_dir: [str] Path to the night directory in ArchivedFiles.
        archive_name: [str] Path to the archive.
        detector: [QueuedPool instance] Handle to the detector.
    """

    # Remove final slash in the night dir
    if night_data_dir.endswith(os.sep):
        night_data_dir = night_data_dir[:-1]

    # Extract the name of the night
    night_data_dir_name = os.path.basename(os.path.abspath(night_data_dir))

    platepar = None
    kml_file = None
    
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
        platepar, platepar_path, platepar_fmt = getPlatepar(config, night_data_dir)


        # Run calibration check and auto astrometry refinement
        if (platepar is not None) and (calstars_name is not None):

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


            # # Calculate astrometry for meteor detections
            # applyAstrometryFTPdetectinfo(night_data_dir, ftpdetectinfo_name, platepar_path)

            # If a flat is used, disable vignetting correction
            if config.use_flat:
                platepar.vignetting_coeff = 0.0

            log.info("Recalibrating astrometry on FF files with detections...")

            # Recalibrate astrometry on every FF file and apply the calibration to detections
            recalibrateIndividualFFsAndApplyAstrometry(night_data_dir, os.path.join(night_data_dir, \
                ftpdetectinfo_name), calstars_list, config, platepar)

            

            log.info("Converting RMS format to UFOOrbit format...")

            # Convert the FTPdetectinfo into UFOOrbit input file
            FTPdetectinfo2UFOOrbitInput(night_data_dir, ftpdetectinfo_name, platepar_path)



            # Generate a calibration report
            log.info("Generating a calibration report...")
            try:
                generateCalibrationReport(config, night_data_dir, platepar=platepar)

            except Exception as e:
                log.debug('Generating calibration report failed with the message:\n' + repr(e))
                log.debug(repr(traceback.format_exception(*sys.exc_info())))


            # Perform single station shower association
            log.info("Performing single station shower association...")
            try:
                showerAssociation(config, [os.path.join(night_data_dir, ftpdetectinfo_name)], \
                    save_plot=True, plot_activity=True)

            except Exception as e:
                log.debug('Shower association failed with the message:\n' + repr(e))
                log.debug(repr(traceback.format_exception(*sys.exc_info())))



            # Generate the FOV KML file
            log.info("Generating a FOV KML file...")
            try:

                mask_path = None
                mask = None

                # Try loading the mask
                if os.path.exists(os.path.join(night_data_dir, config.mask_file)):
                    mask_path = os.path.join(night_data_dir, config.mask_file)

                # Try loading the default mask
                elif os.path.exists(config.mask_file):
                    mask_path = os.path.abspath(config.mask_file)

                # Load the mask if given
                if mask_path:
                    mask = loadMask(mask_path)

                if mask is not None:
                    log.info("Loaded mask: {:s}".format(mask_path))

                # Generate the KML (only the FOV is shown, without the station)
                kml_file = fovKML(config, night_data_dir, platepar, mask=mask, plot_station=False)


            except Exception as e:
                log.debug("Generating a FOV KML file failed with the message:\n" + repr(e))
                log.debug(repr(traceback.format_exception(*sys.exc_info())))


    else:
        ff_detected = []
        detector = None



    log.info('Plotting field sums...')

    # Plot field sums
    try:
        plotFieldsums(night_data_dir, config)

    except Exception as e:
        log.debug('Plotting field sums failed with message:\n' + repr(e))
        log.debug(repr(traceback.format_exception(*sys.exc_info())))



    # Archive all fieldsums to one archive
    archiveFieldsums(night_data_dir)


    # List for any extra files which will be copied to the night archive directory. Full paths have to be 
    #   given
    extra_files = []


    log.info('Making a flat...')

    # Make a new flat field image
    try:
        flat_img = makeFlat(night_data_dir, config)

    except Exception as e:
        log.debug('Making a flat failed with message:\n' + repr(e))
        log.debug(repr(traceback.format_exception(*sys.exc_info())))
        flat_img = None
        

    # If making flat was sucessfull, save it
    if flat_img is not None:

        # Save the flat in the night directory, to keep the operational flat updated
        flat_path = os.path.join(night_data_dir, os.path.basename(config.flat_file))
        saveImage(flat_path, flat_img)
        log.info('Flat saved to: ' + flat_path)

        # Copy the flat to the night's directory as well
        extra_files.append(flat_path)

    else:
        log.info('Making flat image FAILED!')


    ### Add extra files to archive

    # Add the config file to the archive too
    extra_files.append(os.path.join(os.getcwd(), '.config'))

    # Add the mask
    if (not nodetect):
        if os.path.exists(config.mask_file):
            mask_path = os.path.abspath(config.mask_file)
            extra_files.append(mask_path)


    # Add the platepar to the archive if it exists
    if (not nodetect):
        if os.path.exists(platepar_path):
            extra_files.append(platepar_path)


    # Add the json file with recalibrated platepars to the archive
    if (not nodetect):
        recalibrated_platepars_path = os.path.join(night_data_dir, config.platepars_recalibrated_name)
        if os.path.exists(recalibrated_platepars_path):
            extra_files.append(recalibrated_platepars_path)

    # Add the FOV KML file
    if kml_file is not None:
        extra_files.append(kml_file)
        

    ### ###



    # If the detection should be run
    if (not nodetect):

        # Make a CAL file and a special CAMS FTPdetectinfo if full CAMS compatibility is desired
        if (config.cams_code > 0) and (platepar is not None):

            log.info('Generating a CAMS FTPdetectinfo file...')

            # Write the CAL file to disk
            cal_file_name = writeCAL(night_data_dir, config, platepar)

            # Check if the CAL file was successfully generated
            if cal_file_name is not None:

                cams_code_formatted = "{:06d}".format(int(config.cams_code))

                # Load the FTPdetectinfo
                _, fps, meteor_list = readFTPdetectinfo(night_data_dir, ftpdetectinfo_name, \
                    ret_input_format=True)

                # Replace the camera code with the CAMS code
                for met in meteor_list:

                    # Replace the station name and the FF file format
                    ff_name = met[0]
                    ff_name = ff_name.replace('.fits', '.bin')
                    ff_name = ff_name.replace(config.stationID, cams_code_formatted)
                    met[0] = ff_name


                # Write the CAMS compatible FTPdetectinfo file
                writeFTPdetectinfo(meteor_list, night_data_dir, \
                    ftpdetectinfo_name.replace(config.stationID, cams_code_formatted),\
                    night_data_dir, cams_code_formatted, fps, calibration=cal_file_name, \
                    celestial_coords_given=(platepar is not None))



    night_archive_dir = os.path.join(os.path.abspath(config.data_dir), config.archived_dir, 
        night_data_dir_name)


    log.info('Archiving detections to ' + night_archive_dir)
    
    # Archive the detections
    archive_name = archiveDetections(night_data_dir, night_archive_dir, ff_detected, config, \
        extra_files=extra_files)


    return night_archive_dir, archive_name, detector




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

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, cml_args.dir_path)

    
    ### Init the logger

    from RMS.Logger import initLogging
    initLogging(config, 'reprocess_')

    log = logging.getLogger("logger")

    ######


    # Process the night
    _, archive_name, detector = processNight(cml_args.dir_path[0], config)


    # Upload the archive, if upload is enabled
    if config.upload_enabled:

        # Init the upload manager
        print('Starting the upload manager...')

        upload_manager = UploadManager(config)
        upload_manager.start()

        # Add file for upload
        print('Adding file to upload list: ' + archive_name)
        upload_manager.addFiles([archive_name])

        # Stop the upload manager
        if upload_manager.is_alive():
            upload_manager.stop()
            print('Closing upload manager...')


        # Delete detection backup files
        if detector is not None:
            detector.deleteBackupFiles()
