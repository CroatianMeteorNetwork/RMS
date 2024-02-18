""" Selecting and zipping files with detections. """
import datetime
import os
import sys
import logging
import traceback


import RMS.Formats.FFfile
from RMS.Formats.FFfile import validFFName
from RMS.Misc import archiveDir
from RMS.Routines import MaskImage
from Utils.GenerateThumbnails import generateThumbnails
from Utils.StackFFs import stackFFs
from glob import glob

# Get the logger from the main module
log = logging.getLogger("logger")


def reduceTimeGaps(file_list, captured_path, max_time_between_fits = 900):

    """
    Function takes a list of files, calculates the difference in time between
    each fits file, and adds additional files from the CapturedPath to reduce the gap
    to the max_time_between fits

    Arguments:
        file_list: [list] list of files to be uploaded.
        captured_path: [str] path of the captured files directory corresponding to this list of files.
        max_time_between_fits: [int] maximum time between two fits files default 900 seconds.

    Return:
         file_list: [list] The original list of files augmented by fits files to reduce the gaps.

    """


    fits_list = []
    minimum_time_between_fits = 900

    if max_time_between_fits < minimum_time_between_fits:
        log.warning("Setting max_time_between_fits to {} seconds is less than coded minimum of {} seconds".format(max_time_between_fits, minimum_time_between_fits))
        max_time_between_fits = minimum_time_between_fits
    log.info("max_time_between_fits is set to {} seconds".format(max_time_between_fits))

    # make a list of only fits files, sorted by time
    for path_file_to_check in file_list:
        file_to_check = os.path.basename(path_file_to_check)
        if file_to_check.endswith('.fits'):
            fits_list.append(file_to_check)

    fits_list.sort()

    # get a list of all the fits files that are available in the captured files directory, and sort
    captured_fits_list = glob(os.path.join(captured_path,"*.fits"))


    # if there are no fits files, then return from this function
    if len(captured_fits_list) == 0:
        log.warning("No captured fits files so no extra files to add")
        return file_list

    # sort the list of files
    captured_fits_list.sort()

    # calculate some statistics
    original_fits_list_length = len(fits_list)

    first_captured_fits_file = os.path.basename(captured_fits_list[0])
    time_previous_fits_file = RMS.Formats.FFfile.filenameToDatetime(first_captured_fits_file)
    final_captured_fits_file = os.path.basename(captured_fits_list[-1])
    time_final_fits_file = RMS.Formats.FFfile.filenameToDatetime(final_captured_fits_file)

    # add the final captured file to the fits list
    fits_list.append(final_captured_fits_file)

    target_time_list = []

    file_list.sort()

    # compute the initial maximum interval between fits files
    initial_max_interval = 0
    last_time = RMS.Formats.FFfile.filenameToDatetime(first_captured_fits_file)
    for file in file_list:
        if file.endswith(".fits"):
            interval = round((RMS.Formats.FFfile.filenameToDatetime(os.path.basename(file)) - last_time).total_seconds())
            initial_max_interval = interval if interval > initial_max_interval else initial_max_interval
            last_time = RMS.Formats.FFfile.filenameToDatetime(os.path.basename(file))


    # go through all the fits files
    for fits in fits_list:

        time_of_this_fits_file = RMS.Formats.FFfile.filenameToDatetime(fits)
        time_elapsed = int((time_of_this_fits_file - time_previous_fits_file).total_seconds())
        # if the time gap is too large
        if time_elapsed > max_time_between_fits:
            # work out how many intervals there are
            number_of_additional_files = time_elapsed // max_time_between_fits
            # how long is the interval, make 1 second shorter so that we will always add at least one file at middle
            # of interval, for protection against edge case
            interval_seconds = int(time_elapsed // (number_of_additional_files+1) -1)

            # iterate across this interval, missing the first - because that fits file is already in place
            # and add target times for fits files to find

            # intialise target_time for edge case protection
            target_time = time_previous_fits_file + datetime.timedelta(seconds=interval_seconds)


            files_added = 0
            for offset in range(interval_seconds + 1,time_elapsed - interval_seconds,interval_seconds):
                target_time = time_previous_fits_file + datetime.timedelta(seconds = offset)
                target_time_list.append(target_time)
                files_added += 1

            if files_added == 0:
                log.warning("Loop did not execute with input values of")
                log.warning("time_of_this_fits_file {}".format(time_of_this_fits_file))
                log.warning("time_elapsed {}".format(time_elapsed))
                log.warning("number_of_additional_fits_files {}".format(number_of_additional_files))
                log.warning("interval_seconds {}".format(interval_seconds))

            # the time of the previous fits file is the time of the file we are going to add
            time_previous_fits_file = target_time
        else:
            # otherwise the time of the previous fits file is the time of this file
            time_previous_fits_file = time_of_this_fits_file


    # find files immediately after the target times - the intervals will not be perfect
    for target_time in target_time_list:
        for fits_file in captured_fits_list:
            if RMS.Formats.FFfile.filenameToDatetime(os.path.basename(fits_file)) > target_time:
                file_list.append(os.path.basename(fits_file))
                break

    file_list.sort()

    final_max_interval,final_fits_count = 0,0

    # recalculate the statistics
    last_time = RMS.Formats.FFfile.filenameToDatetime(first_captured_fits_file)
    for file in file_list:
        if file.endswith(".fits"):
            interval = round((RMS.Formats.FFfile.filenameToDatetime(os.path.basename(file)) - last_time).total_seconds())
            final_max_interval = interval if interval > final_max_interval else final_max_interval
            last_time = RMS.Formats.FFfile.filenameToDatetime(os.path.basename(file))
            final_fits_count += 1

    log.info("Intervals before / after including extra files {} / {} seconds".format(initial_max_interval, final_max_interval))
    log.info("Original / added / final fits file count {} / {} / {}".format(original_fits_list_length, len(target_time_list), final_fits_count))


    return file_list


def selectFiles(config, dir_path, ff_detected):
    """ Make a list of all files which should be zipped in the given night directory. 
    
        In the list are included:
            - all TXT files
            - all FR bin files and their parent FF bin files
            - all FF bin files with detections

    Arguments:
        config: [conf object] Configuration.
        dir_path: [str] Path to the night directory.
        ff_detected: [list] A list of FF bin file with detections on them.

    Return:
        selected_files: [list] A list of files selected for compression.

    """

    ### Decide what to upload, given the upload mode ###
    
    upload_ffs = True
    upload_frs = True
    
    if config.upload_mode == 2:
        upload_ffs = False

    elif config.upload_mode == 3:
        upload_ffs = False
        upload_frs = False

    elif config.upload_mode == 4:
        upload_frs = False

    ### ###



    selected_list = []

    # Go through all files in the night directory
    for file_name in os.listdir(dir_path):

        # Take all .txt and .csv files
        if (file_name.lower().endswith('.txt')) or (file_name.lower().endswith('.csv')):
            selected_list.append(file_name)


        # Take all PNG, JPG, BMP images
        if ('.png' in file_name) or ('.jpg' in file_name) or ('.bmp' in file_name):
            selected_list.append(file_name)


        # Take all field sum files
        if ('FS' in file_name) and ('fieldsum' in file_name):
            selected_list.append(file_name)


        # Take all FR bin files, and their parent FF bin files
        if upload_frs and ('FR' in file_name) and ('.bin' in file_name):

            fr_split = file_name.split('_')

            # FR file identifier which it shares with the FF bin file
            fr_id = '_'.join(fr_split[1:3])

            ff_match = None

            # Locate the parent FF bin file
            for ff_file_name in os.listdir(dir_path):

                if validFFName(ff_file_name) and (fr_id in ff_file_name):
                    
                    ff_match = ff_file_name
                    break


            # Add the FR bin file and it's parent FF file to the list
            selected_list.append(file_name)

            if ff_match is not None:
                selected_list.append(ff_match)


        # Add FF file which contain detections to the list
        if upload_ffs and (ff_detected is not None) and (file_name in ff_detected):
            selected_list.append(file_name)


    # Take only the unique elements in the list, sorted by name
    selected_list = sorted(list(set(selected_list)))


    return selected_list



def archiveFieldsums(dir_path):
    """ Put all FS fieldsum files in one archive. """

    fieldsum_files = []

    # Find all fieldsum FS files
    for file_name in os.listdir(dir_path):

        # Take all field sum files
        if ('FS' in file_name) and ('fieldsum' in file_name):
            fieldsum_files.append(file_name)


    # Path to the fieldsum directory
    fieldsum_archive_dir = os.path.abspath(os.path.join(dir_path, 'Fieldsums'))


    # Name of the fieldsum archive
    fieldsum_archive_name = os.path.join(os.path.abspath(os.path.join(fieldsum_archive_dir, os.pardir)), \
        'FS_' + os.path.basename(dir_path) + '_fieldsums')

    # Archive all FS files
    archiveDir(dir_path, fieldsum_files, fieldsum_archive_dir, fieldsum_archive_name, delete_dest_dir=False)

    # Delete FS files in the main directory
    for fs_file in fieldsum_files:
        os.remove(os.path.join(dir_path, fs_file))





def archiveDetections(captured_path, archived_path, ff_detected, config, extra_files=None):
    """ Create thumbnails and compress all files with detections and the accompanying files in one archive.

    Arguments:
        captured_path: [str] Path where the captured files are located.
        archived_path: [str] Path where the detected files will be archived to.
        ff_detected: [str] A list of FF files with detections.
        config: [conf object] Configuration.

    Keyword arguments:
        extra_files: [list] A list of extra files (with fill paths) which will be be saved to the night 
            archive.

    Return:
        archive_name: [str] Name of the archive where the files were compressed to.

    """

    # Get the list of files to archive
    file_list = selectFiles(config, captured_path, ff_detected)

    
    log.info('Generating thumbnails...')

    try:

        # Generate captured thumbnails
        captured_mosaic_file = generateThumbnails(captured_path, config, 'CAPTURED')

        # Generate detected thumbnails
        detected_mosaic_file = generateThumbnails(captured_path, config, 'DETECTED', \
            file_list=sorted(file_list), no_stack=True)

        # Add the detected mosaic file to the selected list
        file_list.append(captured_mosaic_file)
        file_list.append(detected_mosaic_file)

    except Exception as e:
        log.error('Generating thumbnails failed with error:' + repr(e))
        log.error("".join(traceback.format_exception(*sys.exc_info())))



    log.info('Generating a stack of detections...')

    try:

        # Load the mask for stack
        mask = None
        mask_path_default = os.path.join(config.config_file_path, config.mask_file)
        if os.path.exists(mask_path_default) and config.stack_mask:
            mask_path = os.path.abspath(mask_path_default)
            mask = MaskImage.loadMask(mask_path)


        # Make a co-added image of all captured images
        captured_stack_path, _ = stackFFs(captured_path, 'jpg', deinterlace=(config.deinterlace_order > 0), 
            subavg=True, mask=mask, captured_stack=True)

        if captured_stack_path is not None:

            log.info("Captured stack saved to: {:s}".format(captured_stack_path))

            # Extract the name of the stack image
            stack_file = os.path.basename(captured_stack_path)
            
            # Add the stack path to the list of files to put in the archive
            file_list.append(stack_file)

        else:
            log.info("Captured stack could not be saved!")


        # Make a co-added image of all detections. Filter out possible clouds
        detected_stack_path, _ = stackFFs(captured_path, 'jpg', deinterlace=(config.deinterlace_order > 0), 
            subavg=True, filter_bright=True, file_list=sorted(file_list), mask=mask)

        if detected_stack_path is not None:

            log.info("Detected stack saved to: {:s}".format(detected_stack_path))

            # Extract the name of the stack image
            stack_file = os.path.basename(detected_stack_path)
            
            # Add the stack path to the list of files to put in the archive
            file_list.append(stack_file)

        else:
            log.info("Detected stack could not be saved!")


    except Exception as e:
        log.error('Generating stack failed with error:' + repr(e))
        log.error("".join(traceback.format_exception(*sys.exc_info())))

    if config.upload_mode == 1:
        try:
            file_list = reduceTimeGaps(file_list, captured_path, config.max_time_between_fits)
        except:
            log.warning("Could not reduce time gaps")

    if file_list:

        # Create the archive ZIP in the parent directory of the archive directory
        archive_name = os.path.join(os.path.abspath(os.path.join(archived_path, os.pardir)), 
            os.path.basename(captured_path) + '_detected')

        # Archive the files
        archive_name = archiveDir(captured_path, file_list, archived_path, archive_name, \
            extra_files=extra_files)

        return archive_name

    return None



if __name__ == "__main__":

    import RMS.ConfigReader as cr

    # Load the configuration file
    config = cr.parse(".config")

    if False:
        # this code is for testing reduceTimeGaps function
        # captured path points to a captured directory
        # file_list is a list of all the files that are to be uploaded, for test purposes
        # use all the files in the already existing archived directory

        captured_path = "/home/user/RMS_data/CapturedFiles/sample_night_dir"
        file_list = os.listdir("/home/user/RMS_data/ArchivedFiles/sample_archived_dir")
        reduceTimeGaps(file_list, captured_path)


    ### Test the archive function

    # captured_path = "/home/dvida/RMS_data/CapturedFiles/20170903_203323_142567"
    # archiveFieldsums(captured_path)

    captured_path = "/home/dvida/RMS_data/CapturedFiles/CA0001_20170905_094706_920438"

    archived_path = "/home/dvida/RMS_data/ArchivedFiles/CA0001_20170905_094706_920438"

    ff_detected = ['FF_CA0001_20170905_094707_004_0000000.fits', 'FF_CA0001_20170905_094716_491_0000256.fits']

    archive_name = archiveDetections(captured_path, archived_path, ff_detected, config)

    print(archive_name)

