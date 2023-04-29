""" Selecting and zipping files with detections. """


import os
import sys
import logging
import traceback


from RMS.Formats.FFfile import validFFName
from RMS.Misc import archiveDir
from RMS.Routines import MaskImage
from Utils.GenerateThumbnails import generateThumbnails
from Utils.StackFFs import stackFFs


# Get the logger from the main module
log = logging.getLogger("logger")


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



    ### Test the archive function

    # captured_path = "/home/dvida/RMS_data/CapturedFiles/20170903_203323_142567"
    # archiveFieldsums(captured_path)

    captured_path = "/home/dvida/RMS_data/CapturedFiles/CA0001_20170905_094706_920438"

    archived_path = "/home/dvida/RMS_data/ArchivedFiles/CA0001_20170905_094706_920438"

    ff_detected = ['FF_CA0001_20170905_094707_004_0000000.fits', 'FF_CA0001_20170905_094716_491_0000256.fits']

    archive_name = archiveDetections(captured_path, archived_path, ff_detected, config)

    print(archive_name)

