""" Freeing up space for new observations by deleting old files. """


import os
import shutil
import datetime

from RMS.CaptureDuration import captureDuration



def availableSpace(p):
    """
    Returns the number of free bytes on the drive that p is on.

    Source: https://atlee.ca/blog/posts/blog20080223getting-free-diskspace-in-python.html
    """

    s = os.statvfs(p)

    return s.f_bsize*s.f_bavail




def getNightDirs(dir_path):
    """ Returns a sorted list of directories in the given directory which conform to the captured directories
        names. 

    Arguments:
        dir_path: [str] Path to the data directory.

    Return:
        dir_list: [list] A list of night directories in the data directory.

    """

    # Get a list of directories in the given directory
    dir_list = [dir_name for dir_name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, dir_name))]

    # Get a list of directories which conform to the captured directories names
    dir_list = [dir_name for dir_name in dir_list if (len(dir_name.split('_')) == 3) and len(dir_name) == 22]
    dir_list = sorted(dir_list)

    return dir_list



def deleteNightFolders(dir_path, delete_all=False):
    """ Deletes captured data directories to free up disk space. Either only one directory will be deleted
        (the oldest one), or all directories will be deleted (if delete_all = True).

    Arguments:
        dir_path: [str] Path to the data directory.

    Keyword arguments:
        delete_all: [bool] If True, all data folders will be deleted. False by default.

    Return:
        dir_list: [list] A list of remaining night directories in the data directory.

    """

    # Get the list of night directories
    dir_list = getNightDirs(dir_path)

    # Delete the night directories
    for dir_name in dir_list:
        
        # Delete the next directory in the list, i.e. the oldes one
        shutil.rmtree(os.path.join(dir_path, dir_name))

        # If only one (first) directory should be deleted, break the loop
        if not delete_all:
            break


    # Return the list of remaining night directories
    return getNightDirs(dir_path)



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


    ### Calculate the approximate needed disk space for the next night

    # If the duration of capture is not given
    if duration is None:

        # Time of next noon
        ct = datetime.datetime.now()
        noon_time = datetime.datetime(ct.year, ct.month, ct.date, 12)

        if ct.hour > 12:
            noon_time += datetime.timedelta(days=1)


        # Get the duration of the next night
        _, duration = captureDuration(config.latitude, config.longitude, config.elevation, 
            current_time=noon_time)


    # Calculate the approx. size for the night night
    next_night_bytes = (duration*config.fps)/256*config.width*config.height*4

    # Always leave at least 1 GB free
    next_night_bytes += 1*(1024**3)


    ######


    # If there's enough free space, don't do anything
    if availableSpace(data_dir) > next_night_bytes:
        return True


    # Intermittently delete captured and archived directories until there's enough free space
    while True:

        # Delete one captured directory
        captured_dirs_remaining = deleteNightFolders(captured_dir)

        # Break the there's enough space
        if availableSpace(data_dir) > next_night_bytes:
            break

        # Delete one archived directory
        archived_dirs_remaining = deleteNightFolders(archived_dir)


        # Break the there's enough space
        if availableSpace(data_dir) > next_night_bytes:
            break


        # If there's nothing left to delete, return False
        if (len(captured_dirs_remaining) == 0) and (len(archived_dirs_remaining) == 0):
            return False


    return True


    
