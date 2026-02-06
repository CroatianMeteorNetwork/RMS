# Skeleton external script for adaption
# Copyright (C) 2025 David Rollinson, Kristen Felker
#
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

# This module provides a skeleton of an External Script. It only
# handles applying and removing the reboot lock, and logs a few entries
# to the log file, including the number of files in CapturedFiles path and
# ArchivedFiles path



import os
from RMS.Logger import getLogger
import datetime
import Utils.TrackStack

import RMS.ConfigReader as cr
from Utils.TrackStack import trackStack

LOG_FILE_PREFIX = "EXTERNAL"

log = getLogger("rmslogger", stdout=False)


def createLock(config, log):
    """ If no file config.reboot_lock_file exists in config.data_dir, create one

    Arguments:
        config: [config] RMS config instance
        log: [logger] logger instance

    Returns:
        Nothing
    """


    log.info("Applying reboot lock")
    lockfile = os.path.join(os.path.expanduser(config.data_dir), config.reboot_lock_file)
    with open(lockfile, 'w') as _:
        pass

    pass

def removeLock(config, log):
    """ If the file config.reboot_lock_file exists in config.data_dir, remove it

    Arguments:
        config: [config] RMS config instance
        log: [logger] logger instance

    Returns:
        Nothing
    """

    log.info("Removing reboot lock")
    lockfile = os.path.join(os.path.expanduser(config.data_dir), config.reboot_lock_file)
    if os.path.exists(lockfile):
        os.remove(lockfile)
    else:
        log.warning("No reboot lock file found at {}".format(lockfile))




def rmsExternal(captured_night_dir, archived_night_dir, config):
    """ Function for launch from main RMS process

    Arguments:
        captured_night_dir: [path] to the captured night directory folder
        archived_night_dir: [path] to the archived night directory folder
        config: [config] RMS config instance

    This function logs a few entries to the log file, applies and removes the reboot lock

    """
    captured_night_dir = os.path.expanduser(captured_night_dir)
    archived_night_dir = os.path.expanduser(archived_night_dir)

    log.info(f"Starting External Script at {datetime.datetime.now(datetime.timezone.utc)}")
    createLock(config, log)
    log.warning(f"RMS External Script applied the reboot lock at {datetime.datetime.now(datetime.timezone.utc)}")
    log.info(f"The reboot lock file is at {config.reboot_lock_file}")
    log.info("The parameters passed to this script were :")
    log.info(f"                                             Captured Night Dir    {captured_night_dir}")
    log.info(f"                                             Archived Night Dir    {archived_night_dir}")

    # Assume the standard multi-cam file structure is in use

    stations_dir = os.path.dirname(config.data_dir)
    stations_list = os.listdir(stations_dir)

    # Get newest CapturedFiles Directory

    captured_files_directory_list = []
    latest_directory_full_path_list = []
    for station in stations_list:
        test_path = os.path.join(stations_dir, station)
        if os.path.isdir(test_path) and len(os.listdir(test_path)):
            captured_files_directory_list.append(test_path)
            latest_directory = sorted(os.listdir(os.path.expanduser(os.path.join(test_path, config.captured_dir))), reverse=True)
            latest_directory_full_path_list.append(os.path.join(os.path.join(stations_dir, station, config.captured_dir, latest_directory[0])))

    print(latest_directory_full_path_list)

    log.info(f"Launching trackStack on {latest_directory_full_path_list}")
    trackStack(latest_directory_full_path_list, config)

    log.info(f"All the work is done, remove the lock")
    removeLock(config, log)

    log.info(f"External Script removed the reboot lock at {datetime.datetime.now(datetime.timezone.utc)}")
    log.info("External Script terminating")

if __name__ == '__main__':
    # test launch
    config = cr.loadConfigFromDirectory(".config", os.path.expanduser("~/source/RMS"))

    # Find the latest CapturedFiles and ArchivedFiles directory
    captured_dirs = sorted(os.listdir(os.path.expanduser(os.path.join(config.data_dir, config.captured_dir))), reverse=True)

    rmsExternal(captured_dirs[0], captured_dirs[0], config)

