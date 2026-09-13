""" Selecting and zipping files with detections. """


import os
import sys
import traceback



from RMS.Formats.FFfile import validFFName
from RMS.Formats.FFfile import read as readFF
from RMS.Logger import getLogger
from RMS.Misc import archiveDir, tarWithProgress
from RMS.Routines import MaskImage
from Utils.GenerateThumbnails import generateThumbnails, ThumbnailMosaic
from Utils.GenerateTimelapse import TimelapseWriter
from Utils.StackFFs import stackFFs, FFStacker
from Utils.LogArchiver import makeLogArchives


# Get the logger from the main module
log = getLogger("rmslogger")


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

    dir_file_list = os.listdir(dir_path)

    # FF file names, listed once for the FR parent lookup below
    ff_file_names = [file_name for file_name in dir_file_list if validFFName(file_name)]

    # Go through all files in the night directory
    for file_name in dir_file_list:

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
            for ff_file_name in ff_file_names:

                if fr_id in ff_file_name:

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

        # Take all field sum .bin files
        if file_name.startswith("FS") and file_name.endswith(".bin") and ('fieldsum' in file_name):
            fieldsum_files.append(file_name)


    # Path to the fieldsum directory
    fieldsum_archive_dir = os.path.abspath(os.path.join(dir_path, 'Fieldsums'))


    # Name of the fieldsum archive
    fieldsum_archive_name = os.path.join(os.path.abspath(os.path.join(fieldsum_archive_dir, os.pardir)), \
        'FS_' + os.path.basename(dir_path) + '_fieldsums')

    # Archive all FS files straight from the night directory; nothing is staged on disk
    archiveDir(dir_path, fieldsum_files, fieldsum_archive_dir, fieldsum_archive_name, delete_dest_dir=True)

    # Delete FS files in the main directory
    for fs_file in fieldsum_files:
        os.remove(os.path.join(dir_path, fs_file))





def generateCapturedProducts(captured_path, config, mask=None, make_timelapse=True):
    """ Make the products that need every FF file of the night, in one pass over the files: the captured
        thumbnail mosaic, the captured stack, and the timelapse. Each FF is read once, maxpixel and
        avepixel only, and fed to all three.

    Arguments:
        captured_path: [str] Path where the captured files are located.
        config: [conf object] Configuration.

    Keyword arguments:
        mask: [MaskStructure] Mask to apply to the stack. None by default.
        make_timelapse: [bool] Whether to make the timelapse. True by default.

    Return:
        mosaic_file, stack_path, timelapse_path:
            - mosaic_file: [str] Name of the captured thumbnail mosaic, None if it failed.
            - stack_path: [str] Path of the captured stack, None if it failed.
            - timelapse_path: [str] Path of the timelapse, None if it was not made.
    """

    ff_list = [ff_name for ff_name in sorted(os.listdir(captured_path)) if validFFName(ff_name)]

    log.info('Generating the captured thumbnails, stack{:s} in one pass over {:d} FF files...'.format(
        ' and timelapse' if make_timelapse else '', len(ff_list)))

    mosaic = ThumbnailMosaic(config, 'CAPTURED', len(ff_list))

    # The per-file 'Stacking:' lines are not logged here, one line per FF for the whole night was noise
    stacker = FFStacker(deinterlace=(config.deinterlace_order > 0), subavg=True, print_progress=False)

    writer = None
    if make_timelapse:
        dir_name = os.path.basename(os.path.abspath(captured_path))
        mp4_path = os.path.join(captured_path, dir_name.replace("_detected", "") + "_timelapse.mp4")
        writer = TimelapseWriter(captured_path, mp4_path)

    mosaic_file, stack_path, timelapse_path = None, None, None

    try:
        for ff_name in ff_list:

            ff = readFF(captured_path, ff_name, planes=('maxpixel', 'avepixel'))

            if ff is None:
                # A corrupted FF still takes its slot in the mosaic
                mosaic.add(ff_name, None)
                continue

            stacker.add(ff_name, ff.maxpixel, ff.avepixel)
            mosaic.add(ff_name, ff.maxpixel)

            # Last, as the timestamp is stamped onto the maxpixel in place
            if writer is not None:
                writer.add(ff_name, ff.maxpixel)

    except Exception as e:
        log.error('Reading the captured FF files failed with error:' + repr(e))
        log.error("".join(traceback.format_exception(*sys.exc_info())))

    finally:
        # Always finish ffmpeg, so a failure above never leaves it waiting on its input
        if writer is not None:
            try:
                timelapse_path = writer.close()
            except Exception as e:
                log.error('Finishing the timelapse failed with error:' + repr(e))
                log.error("".join(traceback.format_exception(*sys.exc_info())))


    try:
        mosaic_file = mosaic.save(captured_path)
    except Exception as e:
        log.error('Generating thumbnails failed with error:' + repr(e))
        log.error("".join(traceback.format_exception(*sys.exc_info())))

    try:
        stack_path, _ = stacker.save(captured_path, 'jpg', mask=mask, captured_stack=True)
    except Exception as e:
        log.error('Generating stack failed with error:' + repr(e))
        log.error("".join(traceback.format_exception(*sys.exc_info())))

    return mosaic_file, stack_path, timelapse_path



def archiveDetections(captured_path, archived_path, ff_detected, config, extra_files=None):
    """ Create thumbnails and compress all files with detections and the accompanying files
        in one archive, suffix _detected

        or

        only the fr*.bin, ff*.fits and extra_files in one archive, suffix _imgdata,
        and everything apart from fr*.bin and ff*.fits in another archive, suffix _metadata

    Arguments:
        captured_path: [str] Path where the captured files are located.
        archived_path: [str] Path where the detected files will be archived to.
        ff_detected: [str] A list of FF files with detections.
        config: [conf object] Configuration.

    Keyword arguments:
        extra_files: [list] A list of extra files (with full paths) which will be saved to the night
            archive.

    Return:
        archive_name: [str] Name of the archive where the files were compressed to.
        imgdata_archive_name: [str] Name of the archive where the images were compressed to.
        metadata_archive_name: [str] Name of the archive where the metadata was compressed to.

    """

    # Get the list of files to archive
    file_list = selectFiles(config, captured_path, ff_detected)

    # Copy the extra files list, the timelapse is added to it below
    extra_files = list(extra_files) if extra_files else []

    # Load the mask for the stack
    mask = None
    try:
        mask_path_default = os.path.join(config.config_file_path, config.mask_file)
        if os.path.exists(mask_path_default) and config.stack_mask:
            mask_path = os.path.abspath(mask_path_default)
            mask = MaskImage.loadMask(mask_path)

    except Exception as e:
        log.error('Loading the mask for the stack failed with error:' + repr(e))
        log.error("".join(traceback.format_exception(*sys.exc_info())))


    # Captured thumbnails, captured stack and the timelapse, from one pass over the FF files
    captured_mosaic_file, captured_stack_path, timelapse_path = generateCapturedProducts(captured_path,
        config, mask=mask, make_timelapse=config.timelapse_generate_captured)

    if captured_mosaic_file is not None:
        file_list.append(captured_mosaic_file)
    else:
        log.info("Captured thumbnails could not be saved!")

    if captured_stack_path is not None:
        log.info("Captured stack saved to: {:s}".format(captured_stack_path))
        file_list.append(os.path.basename(captured_stack_path))
    else:
        log.info("Captured stack could not be saved!")

    if timelapse_path is not None:
        # Goes with the extra files, as it did when processNight made it
        extra_files.append(timelapse_path)
    elif config.timelapse_generate_captured:
        log.info("Timelapse could not be saved!")


    log.info('Generating detected thumbnails...')

    try:
        # Generate detected thumbnails
        detected_mosaic_file = generateThumbnails(captured_path, config, 'DETECTED', \
            file_list=sorted(file_list), no_stack=True)

        file_list.append(detected_mosaic_file)

    except Exception as e:
        log.error('Generating thumbnails failed with error:' + repr(e))
        log.error("".join(traceback.format_exception(*sys.exc_info())))


    log.info('Generating a stack of {:d} detections...'.format(len(ff_detected)))

    try:

        # Make a co-added image of all detections. Filter out possible clouds
        detected_stack_path, _ = stackFFs(captured_path, 'jpg', deinterlace=(config.deinterlace_order > 0), 
            subavg=True, filter_bright=True, file_list=sorted(ff_detected), mask=mask)

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

    log.info("Generating an archive file of most recent logs...")

    try:


        log_archive_path = makeLogArchives(config, captured_path)
        log.info(f"Log archive saved to: {log_archive_path}")
        file_list.append(os.path.basename(log_archive_path))

    except Exception as e:
        log.error('Generating log archives failed with error:' + repr(e))
        log.error("".join(traceback.format_exception(*sys.exc_info())))



    if file_list:

        # Create the archive ZIP in the parent directory of the archive directory
        archive_base = os.path.join(os.path.abspath(os.path.join(archived_path, os.pardir)),
            os.path.basename(captured_path))

        # Create the imgdata set which is the union of the sets of FF files and FR files
        imgdata_set = (set([item for item in file_list if item.startswith("FF") and item.endswith(".fits")]) |
                       set([item for item in file_list if item.startswith("FR") and item.endswith(".bin")]))

        # Create the metadata set which is all the files from _detected excluding the files in imgdata_set,
        # (*.fits and *.bin)

        metadata_set = set([item for item in file_list if item not in imgdata_set])

        # Create all the required archive names and paths
        archive_name = f"{archive_base}_detected"
        metadata_archived_path = f"{archived_path}_metadata"
        metadata_archive_name = f"{archive_base}_metadata"
        imgdata_archived_path = f"{archived_path}_imgdata"
        imgdata_archive_name = f"{archive_base}_imgdata"

        # In all cases, generate the _detected directory and archive as a record for the station

        # Make the archive directory and compress into _detected.tar.bz2 if config.upload_split is False.

        create_detected_tar_bz2 = not config.upload_split
        archive_name = archiveDir(captured_path, file_list, archived_path, archive_name, extra_files=extra_files,
                                  create_archive=create_detected_tar_bz2)

        if config.upload_split:
            # Create a directory to hold the metadata files, archive, then remove the directory
            metadata_archive_name = archiveDir(captured_path, metadata_set, metadata_archived_path,
                                               metadata_archive_name, extra_files=extra_files, delete_dest_dir=True)

            # Create a directory to hold the imgdata files, archive, then remove the directory
            imgdata_archive_name = archiveDir(captured_path, imgdata_set, imgdata_archived_path,
                                              imgdata_archive_name, extra_files=extra_files, delete_dest_dir=True)

            # Set to archive_name to None to prevent upload from this execution path
            archive_name = None

        else:
            # Set to None, as these archives will not be generated in this path
            imgdata_archive_name, metadata_archive_name = None, None

        return archive_name, imgdata_archive_name, metadata_archive_name

    return None, None, None


def archiveFrameTimelapse(frames_root,
                          video_json_pairs,
                          remove_source=False):
    """Tar-up each (mp4, json) pair **without compression**.

    Arguments:
        frames_root: [str] Directory where the archives will be placed.
        video_json_pairs: [list[tuple[str, str] | None]] Output from
            generateTimelapseFromFrameBlocks; each element is either
            (mp4_path, json_path) or None if that block failed.

    Keyword arguments:
        remove_source: [bool] If True, delete the mp4 and json after a verified
            archive is created. False by default.

    Return:
        archive_paths: [list[str]] Paths of archives successfully created.
    """
    archive_paths = []

    for pair in video_json_pairs:
        if not pair:
            continue

        mp4_path, json_path = pair
        if not (os.path.isfile(mp4_path) and os.path.isfile(json_path)):
            log.warning("Skipping archive: missing file(s) %s  %s",
                        mp4_path, json_path)
            continue

        # Build archive name: strip suffix, keep in same root, plain .tar
        base_name = os.path.basename(mp4_path).replace(".mp4", ".tar")
        archive_path = os.path.join(frames_root, base_name)
        tmp_archive = archive_path + ".tmp"

        log.info("Archiving %s and %s to %s",
                 os.path.basename(mp4_path),
                 os.path.basename(json_path),
                 os.path.basename(archive_path))

        try:
            success = tarWithProgress(
                None,
                tmp_archive,
                None,                  # None = no gzip/bz2 compression for mp4
                remove_source,
                file_list=[mp4_path, json_path]
            )

            if success:
                if os.path.exists(archive_path):
                    os.remove(archive_path)
                os.rename(tmp_archive, archive_path)
                archive_paths.append(archive_path)
                log.info("Archive created: %s", archive_path)
            else:
                log.warning("Archive verification failed: %s", archive_path)
                if os.path.exists(tmp_archive):
                    os.remove(tmp_archive)

        except Exception as exc:
            log.error("Archiving error for %s: %s", mp4_path, exc)
            if os.path.exists(tmp_archive):
                os.remove(tmp_archive)

    return archive_paths



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

    archive_name, metadata_name, imgdata_name = archiveDetections(captured_path, archived_path, ff_detected, config)

    print(archive_name, metadata_name, imgdata_name)

