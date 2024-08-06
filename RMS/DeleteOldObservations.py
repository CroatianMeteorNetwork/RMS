""" Freeing up space for new observations by deleting old files. """


import sys
import ctypes
import os
import platform
import shutil
import datetime
import time
import logging
import glob
import argparse

import ephem

from RMS.CaptureDuration import captureDuration
from RMS.ConfigReader import loadConfigFromDirectory
from RMS.Logger import initLogging

# Get the logger from the main module
log = logging.getLogger("logger")


# Python 2 doesn't have the timestamp function, so make one
if (sys.version_info[0] < 3) or (sys.version_info[1] < 4):
    
    # python version < 3.3
    def timestamp(date):
        return time.mktime(date.timetuple())

else:

    def timestamp(date):
        return date.timestamp()



def availableSpace(dirname):
    """
    Returns the number of free bytes on the drive that p is on.

    Source: https://atlee.ca/blog/posts/blog20080223getting-free-diskspace-in-python.html
    """

    if platform.system() == 'Windows':

        free_bytes = ctypes.c_ulonglong(0)

        ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(dirname), None, None,
            ctypes.pointer(free_bytes))

        return free_bytes.value

    else:
        st = os.statvfs(dirname)

        return st.f_bavail*st.f_frsize


def usedSpace(obj_path):

    """

    Args:
        obj_path : path from which to start searching

    Returns:
        file size of files found in directory path
    """

    n = 0
    if os.path.isdir(obj_path):
        for directory_entry in os.listdir(obj_path):
            directory_entry_path = os.path.join(obj_path, directory_entry)
            if os.path.islink(directory_entry_path):
                continue
            n += os.path.getsize(directory_entry_path) if os.path.isfile(directory_entry_path) else usedSpace(directory_entry_path)
    else:
        n += os.path.getsize(obj_path)

    return n / (1024 * 1024)


def objectsToDelete(object_path, stationID, quota_gb = 0, bz2=False):

    """

    Args:
        object_path: path to directory to be checked
        quota_gb: target size of objects in directory
        bz2: look at bz2 files if set true, else only work on directories

    Returns:

    """

    if quota_gb == 0 or quota_gb == None:
        log.info("Disc quota system disabled for {:s}".format(object_path))
        return []
    else:
        if bz2:
            log.info("Quota for bz2 files set at {:.1f}".format(quota_gb))
        else:
            log.info("Quota for directory {:s} set at {:.1f}".format(object_path,quota_gb))


    # get a list of objects
    if bz2:
        object_list = getBz2Files(object_path, stationID)
    else:
        object_list = getNightDirs(object_path, stationID)

    # reverse it to put newest at top
    object_list.reverse()

    # initialise variables and a list
    n, objects_to_delete = 0, []

    # iterate through building up an accumulator, once the accumulator passes quota
    # append items to delete to the list. This means that the space used will be under quota
    # if all these items are deleted
    for obj in object_list:
        obj_size = usedSpace(os.path.join(object_path,obj))
        n += obj_size / 1024
        if n > quota_gb:
            log.info("{}, size {:.1f}GB marked for deletion".format(obj, obj_size / 1024))
            objects_to_delete.append(os.path.join(object_path,obj))

    return objects_to_delete

def rmList(delete_list, dummy_run=True):

    for full_path in delete_list:
        try:
            if dummy_run:
                log.info("Config setting inhibited deletion of {}".format(os.path.basename(full_path)))
            else:
                shutil.rmtree(full_path)
                log.info("Deleted {}".format(os.path.basename(full_path)))
        except:
            log.info("Could not delete {}".format(os.path.basename(full_path)))


def sizeArchivedDirs():
    """

    Returns:
        size: byte size of all archived directories and subdirectories
    """
    archived_path = os.path.join(config.data_dir, config.archived_dir)
    directory_list = getNightDirs(os.path.join(config.data_dir, config.archived_dir), config.stationID).reverse()
    n = 0
    print("Archived directory sizes")
    for directory in directory_list:
        dir_size = usedSpace(os.path.join(archived_path, directory))
        print("{} {:.2f} Gb".format(directory, dir_size / 1024))
        n += dir_size
    return n / 1024

def sizeBz2Files(target_size):


    """

    Args:
        target_size: target total size of bz2 files

    Returns:
        list of files to delete
    """


    file_list = getBz2Files(os.path.join(config.data_dir, config.archived_dir), config.stationID).reverse()
    bz2_path = os.path.join(config.data_dir, config.archived_dir)
    n = 0
    print("Captured directory sizes")
    for bz2_file in file_list:
        file_size = os.path.getsize(os.path.join(bz2_path, bz2_file))
        print("{:s} {:.1f} Gb".format(bz2_file, file_size / 1024))
        n += file_size

    return n





def getNightDirs(dir_path, stationID):
    """ Returns a sorted list of directories in the given directory which conform to the captured directories
        names. 

    Arguments:
        dir_path: [str] Path to the data directory.
        stationID: [str] Name of the station. The directory will have to contain this string to be taken
            as the night directory.

    Return:
        dir_list: [list] A list of night directories in the data directory.

    """

    # Get a list of directories in the given directory
    dir_list = [dir_name for dir_name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, dir_name))]

    # Get a list of directories which conform to the captured directories names
    dir_list = [dir_name for dir_name in dir_list if (len(dir_name.split('_')) > 3) and (stationID in dir_name)]
    dir_list = sorted(dir_list)

    return dir_list



def getBz2Files(dir_path, stationID):
    """ Returns a sorted list of bz2 files in the given directory which conform to the RMS compress archdir names. 

    Arguments:
        dir_path: [str] Path to the data directory.
        stationID: [str] Name of the station. The file will have to contain this string to be taken 
        as a compressed archdir.

    Return:
        dir_list: [list] A list of bz2 files in the data directory.

    """

    # Get a list of files in the given directory
    bz2_list = [bz2_name for bz2_name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, bz2_name))]

    # Get a list of files which conform to the required pattern
    bz2_list = [bz2_name for bz2_name in bz2_list if (len(bz2_name.split('_')) > 3) and (stationID in bz2_name)]
    bz2_list = sorted(bz2_list)

    return bz2_list



def deleteNightFolders(dir_path, config, delete_all=False):
    """ Deletes captured data directories to free up disk space. Either only one directory will be deleted
        (the oldest one), or all directories will be deleted (if delete_all = True).

    Arguments:
        dir_path: [str] Path to the data directory.
        config: [Configuration object]

    Keyword arguments:
        delete_all: [bool] If True, all data folders will be deleted. False by default.

    Return:
        dir_list: [list] A list of remaining night directories in the data directory.

    """

    # Get the list of night directories
    dir_list = getNightDirs(dir_path, config.stationID)

    # Delete the night directories
    for dir_name in dir_list:
        
        # Delete the next directory in the list, i.e. the oldes one
        try:
            shutil.rmtree(os.path.join(dir_path, dir_name))
        except OSError:
            continue

        # If only one (first) directory should be deleted, break the loop
        if not delete_all:
            break


    # Return the list of remaining night directories
    return getNightDirs(dir_path, config.stationID)



def getFiles(dir_path, stationID):
    """ Returns a sorted list of files in the given directory which conform to the captured file names.

    Arguments:
        dir_path: [str] Path to the data directory.
        stationID: [str] Name of the station. The file name will have to contain this string

    Return:
        file_list: [list] A list of files the data directory.

    """

    # Get list of files in the given directory
    file_list = [file_name for file_name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, file_name))]

    # Filter files containing station ID in its name
    file_list = [file_name for file_name in file_list if (len(file_name.split('_')) > 3) and (stationID in file_name)]

    return sorted(file_list)



def deleteFiles(dir_path, config, delete_all=False):
    """ Deletes data files free up disk space. Either only one file will be deleted
        (the oldest one), or all files will be deleted (if delete_all = True).

    Arguments:
        dir_path: [str] Path to the data directory.
        config: [Configuration object]

    Keyword arguments:
        delete_all: [bool] If True, all data folders will be deleted. False by default.

    Return:
        file_list: [list] A list of remaining files in the data directory.

    """

    # Get sorted list of files in dir_path
    file_list = getFiles(dir_path, config.stationID)

    # Delete first file or all files
    for file_name in file_list:

        try:
            os.remove(os.path.join(dir_path, file_name))
        except:
            print('There was an error deleting the file: ', os.path.join(dir_path, file_name))

        # break for loop when deleting only one file
        if not delete_all:
            break

    return getFiles(dir_path, config.stationID)




def deleteOldObservations(data_dir, captured_dir, archived_dir, config, duration=None):
    """ Deletes old observation directories to free up space for new ones.

    Arguments:
        data_dir: [str] Path to the RMS data directory which contains the Captured and Archived diretories
        captured_dir: [str] Captured directory name.
        archived_dir: [str] Archived directory name.
        config: [Configuration object]

    Keyword arguments:
        duration: [float] Duration of next video capturing in seconds. If None (by default), duration will
            be calculated for the next night.

    Return:
        [bool]: True if there's enough space for the next night's data, False if not.

    """

    captured_dir = os.path.join(data_dir, captured_dir)
    archived_dir = os.path.join(data_dir, archived_dir)

    # clear down logs first
    log.info('clearing down log files')
    deleteOldLogfiles(data_dir, config)

    # next purge out any old ArchivedFiles folders and compressed files
    log.info('clearing down old data')
    deleteOldDirs(data_dir, config)

    # calculate the captured directory allowance and print to log
    if config.rms_data_quota is None or config.arch_dir_quota is None or config.bz2_files_quota is None:
        log.info("Deleting files by space quota is not enabled, some quota is None.")
    else:
        capt_dir_quota = config.rms_data_quota - (config.arch_dir_quota + config.bz2_files_quota)
        if capt_dir_quota <= 0:
            log.warning("No quota allocation remains for captured directories, please increase rms_data_quota")
            capt_dir_quota = 0

        arch_dir_use = usedSpace(archived_dir)
        capt_dir_use = usedSpace(captured_dir)

        log.info("Directory quotas")
        log.info("----------------------------------------")
        log.info("Space used                              ")
        log.info("    Archive directory space used: {:.0f)GB".format(usedSpace(archived_dir)))
        log.info("    Archive directory space used: {:.0f)GB".format(usedSpace(captured_dir)))
        log.info("Quotas allowed                          ")
        log.info("        Total quota for RMS_data: {:.0f}GB".format(config.rms_data_quota))
        log.info("      Archived directories quota: {:.0f}GB".format(config.arch_dir_quota))
        log.info("                  bz2 file quota: {:.0f}GB".format(config.bz2_files_quota))
        log.info("          Remaining for captured: {:.0f}GB".format(capt_dir_quota))
        log.info("----------------------------------------")



        delete_list = objectsToDelete(captured_dir, config.stationID, capt_dir_quota,bz2=False)
        log.info("Deleting {:.0f} CapturedFiles directories".format(len(delete_list)))
        rmList(delete_list, dummy_run=config.quota_management_disabled)

        delete_list = objectsToDelete(archived_dir, config.stationID, config.arch_dir_quota, bz2=False)
        log.info("Deleting {:.0f} ArchivedFiles directories".format(len(delete_list)))
        rmList(delete_list, dummy_run=config.quota_management_disabled)

        delete_list_bz2 = objectsToDelete(archived_dir, config.stationID, config.bz2_files_quota, bz2=True)
        log.info("Deleting {:.0f} bz2 file".format(len(delete_list_bz2)))
        rmList(delete_list, dummy_run=config.quota_management_disabled)








    # Calculate the approximate needed disk space for the next night

    # If the duration of capture is not given
    if duration is None:

        # Time of next local noon
        #ct = datetime.datetime.utcnow()
        #noon_time = datetime.datetime(ct.year, ct.month, ct.date, 12)

        # Initialize the observer and find the time of next noon
        o = ephem.Observer()  
        o.lat = str(config.latitude)
        o.long = str(config.longitude)
        o.elevation = config.elevation
        sun = ephem.Sun()

        sunrise = o.previous_rising(sun, start=ephem.now())
        noon_time = o.next_transit(sun, start=sunrise).datetime()

        # if ct.hour > 12:
        #     noon_time += datetime.timedelta(days=1)


        # Get the duration of the next night
        _, duration = captureDuration(config.latitude, config.longitude, config.elevation, 
            current_time=noon_time)


    # Calculate the approx. size for the night night
    next_night_bytes = (duration*config.fps)/256*config.width*config.height*4

    # Always leave at least 2 GB free for archive
    next_night_bytes += config.extra_space_gb*(1024**3)

    ######

    log.info("Need {:.2f} GB for next night".format(next_night_bytes/1024/1024/1024))


    # If there's enough free space, don't do anything
    if availableSpace(data_dir) > next_night_bytes:
        return True


    # Intermittently delete captured and archived directories until there's enough free space
    prev_available_space = availableSpace(data_dir)
    log.info("Available space before deleting: {:.2f} GB".format(prev_available_space/1024/1024/1024))
    nothing_deleted_count = 0
    free_space_status = False
    while True:

        # Delete one captured directory
        captured_dirs_remaining = deleteNightFolders(captured_dir, config)

        log.info("Deleted dir captured directory: {:s}".format(captured_dir))
        log.info("Free space: {:.2f} GB".format(availableSpace(data_dir)/1024/1024/1024))

        # Break the there's enough space
        if availableSpace(data_dir) > next_night_bytes:
            free_space_status = True
            break

        # Delete one archived directory
        archived_dirs_remaining = deleteNightFolders(archived_dir, config)

        log.info("Deleted dir in archived directory: {:s}".format(archived_dir))
        log.info("Free space: {:.2f} GB".format(availableSpace(data_dir)/1024/1024/1024))


        # Break if there's enough space
        if availableSpace(data_dir) > next_night_bytes:
            free_space_status = True
            break

        # Wait 10 seconds between deletes. This helps to balance out the space distribution if multiple
        #   instances of RMS are running on the same system
        log.info("Still not enough space, waiting 10 s...")
        time.sleep(10)

        # If no folders left to delete, try to delete archived files
        if (len(captured_dirs_remaining) == 0) and (len(archived_dirs_remaining) == 0):

            log.info("Deleted all Capture and Archived directories, deleting archived bz2 files...")

            archived_files_remaining = deleteFiles(archived_dir, config)

            # If there's nothing left to delete, return False
            if len(archived_files_remaining) == 0:
                free_space_status = False
                break


        # Break the there's enough space
        if availableSpace(data_dir) > next_night_bytes:
            free_space_status = True
            break


        # If nothing was deleted in this loop, count how may time this happened
        if availableSpace(data_dir) == prev_available_space:
            log.info("Nothing got deleted...")
            nothing_deleted_count += 1

        else:
            nothing_deleted_count = 0


        # If nothing was deleted for 100 loops, indicate that no more space can be freed
        if nothing_deleted_count >= 100:
            free_space_status = False
            break

        prev_available_space = availableSpace(data_dir)


    # If there is still not enough space, wait 10 seconds to see if perhaps other users are clearing their
    #   space if this is a multiuser setup
    if free_space_status is False:

        time.sleep(10)

        # If there's still not enough space, return False
        if availableSpace(data_dir) < next_night_bytes:
            return False


    return True


def deleteOldDirs(data_dir, config):

    archived_dir = os.path.join(data_dir, config.archived_dir)
    orig_count = 0
    final_count = 0
    if config.arch_dirs_to_keep > 0:
        archdir_list = getNightDirs(archived_dir, config.stationID)
        orig_count = len(archdir_list)
        while len(archdir_list) > config.arch_dirs_to_keep:
            archdir_list = deleteNightFolders(archived_dir, config)
        final_count = len(archdir_list)
    log.info('Purged {} older folders from ArchivedFiles'.format(orig_count - final_count))

    orig_count = 0
    final_count = 0
    captured_dir = os.path.join(data_dir, config.captured_dir)
    if config.capt_dirs_to_keep > 0:
        captdir_list = getNightDirs(captured_dir, config.stationID)
        orig_count = len(captdir_list)
        while len(captdir_list) > config.capt_dirs_to_keep:
            captdir_list = deleteNightFolders(captured_dir, config)
        final_count = len(captdir_list)
    log.info('Purged {} older folders from Captured Files'.format(orig_count - final_count))


    if config.bz2_files_to_keep > 0:
        bz2_list = getBz2Files(archived_dir, config.stationID)
        orig_count = len(bz2_list)
        while len(bz2_list) > config.bz2_files_to_keep:
            os.remove(os.path.join(archived_dir, bz2_list[0]))
            bz2_list.pop(0)
        final_count = len(bz2_list)
    log.info('Purged {} older bz2 files from ArchivedFiles'.format(orig_count - final_count))
    return


def deleteOldLogfiles(data_dir, config, days_to_keep=None):
    """ Deletes old observation directories to free up space for new ones.

    Arguments:
        data_dir: [str] Path to the RMS data directory which contains the Captured and Archived diretories
        config: [Configuration object]
        duration: [int] number of days to retain, default None means read from config file
    """
    log_dir = os.path.join(data_dir, config.log_dir)
    
    # Date to purge before
    if days_to_keep is None:
        days_to_keep = int(config.logdays_to_keep)
    date_to_purge_to = datetime.datetime.now() - datetime.timedelta(days=days_to_keep)
    date_to_purge_to = timestamp(date_to_purge_to)

    # Only going to purge RMS log files
    flist = glob.glob1(log_dir, 'log*.log*')

    for fl in flist:

        log_file_path = os.path.join(log_dir, fl)

        # Check if the file exists and check if it should be purged
        if os.path.isfile(log_file_path):

            # Get the file modification time
            file_mtime = os.stat(log_file_path).st_mtime

            # If the file is older than the date to purge to, delete it
            if file_mtime < date_to_purge_to:
                try:
                    os.remove(log_file_path)
                    log.info("deleted {}".format(fl))
                except Exception as e:
                    log.warning('unable to delete {}: '.format(log_file_path) + repr(e)) 
                

if __name__ == '__main__':
    """ Delete old data to free up space for next night's run
    """
    arg_parser = argparse.ArgumentParser(description=""" Deleting old observations.""")
    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, help="Path to a config file")
    cml_args = arg_parser.parse_args()



    cfg_path = os.path.abspath('.') # default to using config from current folder
    cfg_file = '.config'



    if cml_args.config:
        if os.path.isfile(cml_args.config[0]):
            cfg_path, cfg_file = os.path.split(cml_args.config[0])
    config = loadConfigFromDirectory(cfg_file, cfg_path)

    # Initialize the logger
    initLogging(config)
    log = logging.getLogger("logger")


    if not os.path.isdir(config.data_dir):
        log.info('Data Dir not found {}'.format(config.data_dir))
    else:
        log.info('deleting obs from {}'.format(config.data_dir))
        deleteOldObservations(config.data_dir, config.captured_dir, config.archived_dir, config)

