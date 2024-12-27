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
import subprocess
import re

import ephem

from RMS.CaptureDuration import captureDuration
from RMS.ConfigReader import loadConfigFromDirectory
from RMS.Logger import initLogging
from RMS.Misc import RmsDateTime

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


def quotaReport(capt_dir_quota, config, after=False):
    """
    Args:
        capt_dir_quota : GB allowance for captured directories
        config : station configuration file
        after : optional, default false, selects an appropriate report header

    Returns:
        str : \n delimited report suitable for printing or logging
    """

    captured_dir = os.path.join(config.data_dir, config.captured_dir)
    archived_dir = os.path.join(config.data_dir, config.archived_dir)


    rep = "\n\n"
    rep += ("--------------------------------------------\n")
    if after:
        rep += ("Directory quotas after management\n")
    else:
        rep += ("Directory quotas before management\n")
    rep += ("--------------------------------------------\n")
    rep += ("Space used                              \n")
    rep += ("   Total space used for RMS_data : {:7.02f}GB\n".format(usedSpace(config.data_dir)))
    rep += ("            bz2 files space used : {:7.02f}GB\n".format(sizeBz2Files(config)))
    rep += (" archived directories space used : {:7.02f}GB\n".format(sizeArchivedDirs(config)))
    rep += ("              total for archives : {:7.02f}GB\n".format(usedSpace(archived_dir)))
    rep += ("   Captured directory space used : {:7.02f}GB\n".format(usedSpace(captured_dir)))
    rep += ("Quotas allowed                          \n")
    rep += ("        Total quota for RMS_data : {:7.02f}GB\n".format(config.rms_data_quota))
    rep += ("                  bz2 file quota : {:7.02f}GB\n".format(config.bz2_files_quota))
    rep += ("      Archived directories quota : {:7.02f}GB\n".format(config.arch_dir_quota))
    rep += ("          Remaining for captured : {:7.02f}GB\n".format(capt_dir_quota))
    rep += ("Space on drive                          \n")
    rep += ("        Available space on drive : {:7.02f}GB\n".format(availableSpace(config.data_dir) / (1024 ** 3)))
    rep += ("--------------------------------------------\n")

    return rep

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


def usedSpaceRecursive(obj_path):
    """
    Calculates space used by recursion

    Args:
        obj_path : path from which to start searching

    Returns:
        GB file size of files found in directory path
    """

    obj_path = os.path.expanduser(obj_path)
    n = 0
    if os.path.isdir(obj_path):
        n += 4.024 / (1024 ** 2) # allowance for size of directory node 4.0k
        for directory_entry in os.listdir(obj_path):
            directory_entry_path = os.path.join(obj_path, directory_entry)
            if os.path.islink(directory_entry_path):
                continue
            if os.path.isfile(directory_entry_path):
                n += os.path.getsize(directory_entry_path) / 1024 ** 3
            else:
                n += usedSpace(directory_entry_path)
    else:
        n += os.path.getsize(obj_path) / 1024 ** 3
    return n

def usedSpaceFromOS(obj_path):
    """
    Calculates used space by a call to du

    Args:
        obj_path : path from which to start searching

    Returns:
        GB file size of files found in directory path
    """
    obj_path = os.path.expanduser(obj_path)
    byte_encode = subprocess.check_output(["du", "-d0", obj_path])
    text = byte_encode.decode(encoding="utf-8")
    value = text.split("\t")[0]

    return float(value) / (1024 ** 2)


def usedSpaceNoRecursion(obj_path):
    """
    Calculates used space using os.walk()

    Args:
        obj_path : path from which to start searching

    Returns:
        GB file size of files found in directory path
        """

    obj_path = os.path.expanduser(obj_path)
    path_list, n = os.walk(obj_path), 0
    for root, directory_list, file_list in path_list:
        for _ in directory_list:
            n += 4.024 / 1024 ** 2
        for file_name in file_list:
            n += os.path.getsize(os.path.join(root, file_name)) / 1024 ** 3
    return n

def usedSpace(obj):
    """
    Args:
        obj (): file system object from where to start searching

    Returns:
        size beneath this object in GB (bytes / 1024 **3 )
    """

    obj = os.path.expanduser((obj))
    return usedSpaceNoRecursion(obj)


def objectsToDelete(object_path, stationID, quota_gb=0, bz2=False):
    """
    Args:
        object_path: path to directory to be checked
        quota_gb: target size of objects in directory
        bz2: look at bz2 files if set true, else only work on directories

    Returns:
        list of full paths to file system objects for deletion
    """

    if quota_gb == 0 or quota_gb == None:
        log.info("Disc quota system disabled for {:s}".format(object_path))
        return []

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
        n += obj_size
        if n > quota_gb:
            log.info("{}, size {:.1f}GB marked for deletion".format(obj, obj_size))
            objects_to_delete.append(os.path.join(object_path,obj))

    return objects_to_delete

def rmList(delete_list, dummy_run=True):
    """
    Args:
        delete_list (): list of full paths to objects to be deleted
        dummy_run (): optional, default True, conducts a dummy run with no deletions

    Returns:
        None
    """

    for full_path in delete_list:
        full_path = os.path.expanduser(full_path)
        try:
            if dummy_run:
                log.info("Config setting inhibited deletion of {}".format(os.path.basename(full_path)))
            else:
                if os.path.exists(full_path):
                    shutil.rmtree(full_path)
                    log.info("Deleted {}".format(os.path.basename(full_path)))
                else:
                    log.warning("Attempted to delete {}, which did not exist".format(full_path))
        except:
            log.info("Could not delete {}".format(os.path.basename(full_path)))


def sizeArchivedDirs(config):
    """
    Args:
        config (): Station config file

    Returns:
        size in bytes of the archived directories
    """

    archived_path = os.path.join(config.data_dir, config.archived_dir)
    directory_list = getNightDirs(os.path.join(config.data_dir, config.archived_dir), config.stationID)
    directory_list.reverse()

    n = 0
    for directory in directory_list:
        dir_size = usedSpace(os.path.join(archived_path, directory))
        n += dir_size
    return n


def sizeBz2Files(config):
    """
    Args:
        config ():Station config file

    Returns:
        size in GB of all the .bz2 files in the archived directory
    """

    file_list = getBz2Files(os.path.join(config.data_dir, config.archived_dir), config.stationID)
    file_list.reverse()
    bz2_path = os.path.join(config.data_dir, config.archived_dir)

    n = 0
    for bz2_file in file_list:
        file_size = os.path.getsize(os.path.join(bz2_path, bz2_file))

        n += file_size / (1024 ** 3)

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
    if not os.path.exists(dir_path):
        return []
    dir_list = [dir_name for dir_name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, dir_name))]

    # Get a list of directories which conform to the captured directories names
    dir_list = [dir_name for dir_name in dir_list if (len(dir_name.split('_')) > 3) and (stationID in dir_name)]
    dir_list = sorted(dir_list)

    return dir_list


def getRawItems(dir_path, in_frame_dir=False, unique=False):
    """ Returns a sorted list of video directories / frame files from within the raw video / frames directories respectively.
        In case of frames directory, only adds to the list the processed frames-(archive, timelapse, json) files and
        not directories as they may be unprocessed.

    Arguments:
        dir_path: [str] Path to the raw video / frames directory.
        in_frame_dir: [bool] Set this to True when dir_path is frames directory. False by default.
        unique: [bool] Set this to True to get unique files by date

    Return:
        dir_list: [list] A list of directories / files in the raw video / frame directories, each corresponding to one day of data

    """

    # Helper function to check frames file conditions
    def isProcessedFrameFile(path):
        suffix = ['_frametimes.json', '_frames_timelapse.mp4', '_frames.tar.gz', '_frames.tar.bz2']
        return (os.path.isfile(path) and any(path.endswith(end) for end in suffix))

    # Get a list of directories / files in the given directory
    if not os.path.exists(dir_path):
        return []
    
    raw_list = []
    
    for year in os.listdir(dir_path):
        year_path = os.path.join(dir_path, year)

        if in_frame_dir:
            raw_list += [os.path.join(year_path, day_file) for day_file in os.listdir(year_path) if isProcessedFrameFile(os.path.join(year_path, day_file))]
        else:
            raw_list += [os.path.join(year_path, day_dir) for day_dir in os.listdir(year_path) if os.path.isdir(os.path.join(year_path, day_dir))]


    # Output files with unique dates
    if unique:
        unique_dates = []
        temp_path_list = []

        for raw_item in raw_list:
            date = os.path.basename(raw_item).split('_')[1] # YYYYMMDD-DoY
            if date not in unique_dates:
                temp_path_list += [raw_item]
                unique_dates += [date]

        raw_list = temp_path_list

    return sorted(raw_list)


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
    if not os.path.exists(dir_path):
        return []
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
        
        # Delete the next directory in the list, i.e. the oldest one
        try:
            shutil.rmtree(os.path.join(dir_path, dir_name))
        except OSError:
            continue

        # If only one (first) directory should be deleted, break the loop
        if not delete_all:
            break


    # Return the list of remaining night directories
    return getNightDirs(dir_path, config.stationID)


def deleteRawItems(dir_path, delete_all=False, in_frame_dir=False, unique=False):
    """ Deletes raw video / frames data directories / files respectively, to free up disk space. In case of frames directory,
        it only will check for and delete old processed files (timelapses, archives, and json files) per day
        and not directories as they may be unprocessed.

    Arguments:
        dir_path: [str] Path to the data directory

    Keyword arguments:
        delete_all: [bool] If True, all raw video / frame folders will be deleted. False by default.
        in_frame_dir: [bool] Set this to True when dir_path is frames directory. False by default.
        unique: [bool] Set this to True to get unique files by date

    Return:
        dir_list: [list] A list of remaining raw video/frame directories/files in the data directory.
    """

    # Get the list of raw directories/files
    del_list = getRawItems(dir_path, in_frame_dir=in_frame_dir)

    # Delete the raw video / frames directories / files, respectively
    for item in del_list:
        
        # Delete the next directory or file(s) in the list, i.e. the oldest ones
        try:
            if in_frame_dir:
                
                # For frame files, each day has a triplet of files: an archive (gz or bz2), a timelapse, and a json file
                # We match file date for this batch of files for one day. The file base names are of type STATIONID_YYYYMMDD-DoY_*
                date_str = os.path.basename(item).split('_')[1]

                # Delete all files with this target date
                files_to_delete = [path for path in del_list if date_str in os.path.basename(path)]

                for file in files_to_delete:
                    os.remove(file)

            else:
                shutil.rmtree(item)

        except OSError:
            continue

        # If only one (first) directory should be deleted, break the loop
        if not delete_all:
            break

    # Return the list of remaining raw frame/video directories
    return getRawItems(dir_path, in_frame_dir=in_frame_dir, unique=unique)



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
        data_dir: [str] Path to the RMS data directory which contains the Captured and Archived directories
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
    frame_dir = os.path.join(data_dir, config.frame_dir)
    video_dir = os.path.join(data_dir, config.video_dir)

    # clear down logs first
    log.info('clearing down log files')
    deleteOldLogfiles(data_dir, config)

    # next purge out any old ArchivedFiles folders, compressed files, and video clips
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

        deleteByQuota(archived_dir, capt_dir_quota, captured_dir, config)

    # Calculate the approximate needed disk space for the next night

    # If the duration of capture is not given
    if duration is None:

        # Time of next local noon
        #ct = RmsDateTime.utcnow()
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


    # Calculate the approx. size for the next night
    next_night_bytes = (duration*config.fps)/256*config.width*config.height*4

    # Always leave at least 2 GB free for archive
    next_night_bytes += config.extra_space_gb*(1024**3)


    # Calculating space for raw frames and videos.
    # In case of continuous capture, frames and videos will be captured throughout the day (86400 seconds).
    # Else, they will happen as long as nighttime capture is runnning

    if config.continuous_capture:
        duration = 86400

    if config.raw_video_save:

        # Taking a ~0.25 Mbps average video bitrate (default 720p capture @ 25 fps)
        raw_video_bytes = duration*0.25*(1024**2)

        # Roughly scaling for higher resolutions
        raw_video_bytes *= (config.width*config.height)/(1280*720)

        # Roughly scaling for fps
        raw_video_bytes *= (config.fps)/(25)

        next_night_bytes += raw_video_bytes


    if config.save_frames: # (estimated value of 3GB maximum for 24 hours)
        next_night_bytes += 3*(1024**3)

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

        # Delete one day of video directory data
        video_dirs_remaining = deleteRawItems(video_dir)

        log.info("Deleted dir in video directory: {:s}".format(video_dir))
        log.info("Free space: {:.2f} GB".format(availableSpace(data_dir)/1024/1024/1024))

        # Break the there's enough space
        if availableSpace(data_dir) > next_night_bytes:
            free_space_status = True
            break


        # Delete one day of frame directory data
        frame_dirs_remaining = deleteRawItems(frame_dir, in_frame_dir=True)

        log.info("Deleted files in frame directory: {:s}".format(frame_dir))
        log.info("Free space: {:.2f} GB".format(availableSpace(data_dir)/1024/1024/1024))

        # Break the there's enough space
        if availableSpace(data_dir) > next_night_bytes:
            free_space_status = True
            break


        # Delete one captured directory
        captured_dirs_remaining = deleteNightFolders(captured_dir, config)

        log.info("Deleted dir in captured directory: {:s}".format(captured_dir))
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
        if (len(captured_dirs_remaining) + len(archived_dirs_remaining) + 
            len(frame_dirs_remaining) + len(video_dirs_remaining) == 0):

            log.info("Deleted all Capture, Archived, Frame and Video directories, deleting archived bz2 files...")

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


def deleteByQuota(archived_dir, capt_dir_quota, captured_dir, config):
    """
    By quotas deletes the oldest files in each section. This method does
    not delete log files.

    Args:
        archived_dir (): Full path to the archive directory
        capt_dir_quota (): Quota for captured directories
        captured_dir (): Path to the captured directoreis
        config (): Station config

    Returns:
        None
    """

    log.info(quotaReport(capt_dir_quota, config, after=False))

    delete_list = objectsToDelete(captured_dir, config.stationID, capt_dir_quota, bz2=False)
    rmList(delete_list, dummy_run=config.quota_management_disabled)

    delete_list = objectsToDelete(archived_dir, config.stationID, config.arch_dir_quota, bz2=False)
    rmList(delete_list, dummy_run=config.quota_management_disabled)

    delete_list = objectsToDelete(archived_dir, config.stationID, config.bz2_files_quota, bz2=True)
    rmList(delete_list, dummy_run=config.quota_management_disabled)

    log.info(quotaReport(capt_dir_quota, config, after=True))


def deleteOldDirs(data_dir, config):

    # Deleting old archived dirs
    archived_dir = os.path.join(data_dir, config.archived_dir)
    orig_count = 0
    final_count = 0
    if config.arch_dirs_to_keep > 0:
        archdir_list = getNightDirs(archived_dir, config.stationID)
        orig_count = len(archdir_list)
        while len(archdir_list) > config.arch_dirs_to_keep:
            prev_length = len(archdir_list)
            archdir_list = deleteNightFolders(archived_dir, config)
            if len(archdir_list) == prev_length:
                log.error("Failed to delete folder from ArchivedFiles. Exiting loop.")
                break
        final_count = len(archdir_list)
    log.info('Purged {} older folders from ArchivedFiles'.format(orig_count - final_count))


    # Deleting old captured dirs
    orig_count = 0
    final_count = 0
    captured_dir = os.path.join(data_dir, config.captured_dir)
    if config.capt_dirs_to_keep > 0:
        captdir_list = getNightDirs(captured_dir, config.stationID)
        orig_count = len(captdir_list)
        while len(captdir_list) > config.capt_dirs_to_keep:
            prev_length = len(captdir_list)
            captdir_list = deleteNightFolders(captured_dir, config)
            if len(captdir_list) == prev_length:
                log.error("Failed to delete folder from CapturedFiles. Exiting loop.")
                break
        final_count = len(captdir_list)
    log.info('Purged {} older folders from CapturedFiles'.format(orig_count - final_count))


    # Deleting old frame dir files
    orig_count = 0
    final_count = 0
    frame_dir = os.path.join(data_dir, config.frame_dir)
    if config.frame_days_to_keep > 0:
        framedir_list = getRawItems(frame_dir, in_frame_dir=True, unique=True)
        orig_count = len(framedir_list)
        while len(framedir_list) > config.frame_days_to_keep:
            prev_length = len(framedir_list)
            framedir_list = deleteRawItems(frame_dir, in_frame_dir=True, unique=True)
            if len(framedir_list) == prev_length:
                log.error("Failed to delete folder from FrameFiles. Exiting loop.")
                break
        final_count = len(framedir_list)
    log.info('Purged old files from {} days in FrameFiles'.format(orig_count - final_count))


    # Deleting old video dirs
    orig_count = 0
    final_count = 0
    video_dir = os.path.join(data_dir, config.video_dir)
    if config.video_days_to_keep > 0:
        videodir_list = getRawItems(video_dir)
        orig_count = len(videodir_list)
        while len(videodir_list) > config.video_days_to_keep:
            prev_length = len(videodir_list)
            videodir_list = deleteRawItems(video_dir)
            if len(videodir_list) == prev_length:
                log.error("Failed to delete folder from VideoFiles. Exiting loop.")
                break
        final_count = len(videodir_list)
    log.info('Purged {} days of old folders from VideoFiles'.format(orig_count - final_count))

    # Deleting old bz2 files
    orig_count = 0
    final_count = 0
    if config.bz2_files_to_keep > 0:
        bz2_list = getBz2Files(archived_dir, config.stationID)
        orig_count = len(bz2_list)
        while len(bz2_list) > config.bz2_files_to_keep:
            try:
                os.remove(os.path.join(archived_dir, bz2_list[0]))
                bz2_list.pop(0)
            except OSError as e:
                log.error("Failed to delete file {}: {}. Exiting loop.".format(bz2_list[0], e))
                break
        final_count = len(bz2_list)
    log.info('Purged {} older bz2 files from ArchivedFiles'.format(orig_count - final_count))
    return


def deleteOldLogfiles(data_dir, config, days_to_keep=None):
    """ Deletes old observation directories to free up space for new ones.

    Arguments:
        data_dir: [str] Path to the RMS data directory which contains the Captured and Archived directories
        config: [Configuration object]
        days_to_keep: [int] number of days to retain, default None means read from config file
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

    print("Disc use routine checks -these should all produce similar results")
    print("Used space no recursion   {:.5f}GB".format(usedSpaceNoRecursion("~/source/RMS")))
    print("Used space with recursion {:.5f}GB".format(usedSpaceRecursive("~/source/RMS")))
    print("Used space from OS        {:.5f}GB".format(usedSpaceFromOS("~/source/RMS")))

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
