# rsync based uploader
# Copyright (C) 2026 David Rollinson
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
import subprocess
from RMS.Logger import getLogger
import datetime

import RMS.ConfigReader as cr


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

    Upload files in priority order to a remote server using rsync

    """
    createLock(config, log)

    station_id = config.stationID
    station_id_lower = station_id.lower()
    key_path = os.path.expanduser(config.rsa_private_key)

    remote_path = os.path.join("/", "home",station_id_lower,"files","incoming")
    local_path = os.path.join(config.data_dir, config.archived_dir)
    with open(os.path.expanduser(os.path.join(config.data_dir, "rsync_remote_host.txt"))) as f:
        rsync_remote_host = f.readline()
        user_host = f"{station_id_lower}@{rsync_remote_host}:".replace("\n","")

    log.info(f"Using key from {key_path}")
    log.info(f"To copy files from {local_path} to {user_host}{remote_path}")

    local_path_modifier_list = ["*_metadata.tar.bz2",
                                "*_detected.tar.bz2",
                                "*_imgdata.tar.bz2",
                                "*.tar.bz2"]


    for local_path_modifier in local_path_modifier_list:


        # modify the local path to send files in the right order
        local_path_modified = os.path.join(local_path, local_path_modifier)
        log.info(f"Sending {local_path_modified}")
        # build rsync command
        command_string = f"rsync --progress -av -e 'ssh -i {key_path}'  {local_path_modified} {user_host}{remote_path}"
        subprocess.run(command_string, shell=True)

    # Now send the frame_dir

    local_path = os.path.join(config.data_dir, config.frame_dir, "*.tar")
    command_string = f"rsync --progress -av -e 'ssh -i {key_path}' {local_path} {user_host}{remote_path}"
    subprocess.run(command_string, shell=True)
    removeLock(config, log)


if __name__ == '__main__':
    # test launch
    config = cr.loadConfigFromDirectory(".config", os.path.expanduser("~/source/RMS"))

    # Find the latest CapturedFiles and ArchivedFiles directory
    captured_dirs = sorted(os.listdir(os.path.expanduser(os.path.join(config.data_dir, config.captured_dir))), reverse=True)

    rmsExternal(captured_dirs[0], captured_dirs[0], config)