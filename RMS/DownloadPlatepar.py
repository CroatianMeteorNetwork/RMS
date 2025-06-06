""" Download a platepar from the server if a new plate exists. """

from __future__ import print_function, division, absolute_import

import logging
import os
from os.path import exists as file_exists

import paramiko


from RMS.UploadManager import getSSHAndSFTP
from RMS.Misc import RmsDateTime

# Get the logger from the main module
log = logging.getLogger("logger")




def downloadNewPlatepar(config):
    """ Connect to the central server and download a new platepar calibration file, if available. """

    log.info('Checking for new platepar on the server...')
    if file_exists(config.rsa_private_key) is False:
        log.debug("Can't contact the server: RSA private key file not found.")
        return False

    ssh = None
    sftp = None

    try:
        # Connect to host
        # Connect with timeouts
        ssh, sftp = getSSHAndSFTP(
            config.hostname,
            port=config.host_port,
            username=config.stationID.lower(),
            key_filename=config.rsa_private_key,
            timeout=60,
            banner_timeout=60,
            auth_timeout=60
        )

        # Check that the remote directory exists
        try:
            sftp.stat(config.remote_dir)

        except Exception as e:
            log.error("Remote directory '" + config.remote_dir + "' does not exist!")
            return False


        # Construct path to remote platepar directory
        remote_platepar_path = config.remote_dir + '/' + config.remote_platepar_dir + '/'

        # Change the directory into file
        remote_platepar = remote_platepar_path + config.platepar_remote_name


        # Check if the remote platepar file exists
        try:
            sftp.lstat(remote_platepar)
        
        except IOError as e:
            log.info('No new platepar on the server!')
            return False


        # Download the remote platepar
        sftp.get(remote_platepar, os.path.join(config.config_file_path, config.platepar_name))

        log.info('Latest platepar downloaded!')


        ### Rename the downloaded platepar file, add a timestamp of download

        # Construct a new name with the time of the download included
        dl_pp_name = remote_platepar_path + 'platepar_dl_' \
            + RmsDateTime.utcnow().strftime('%Y%m%d_%H%M%S.%f') + '.cal'

        sftp.posix_rename(remote_platepar, dl_pp_name)

        log.info('Remote platepar renamed to: ' + dl_pp_name)

    finally:
        if sftp:
            log.debug("Closing SFTP channel")
            sftp.close()
        if ssh:
            log.debug("Closing SSH client connection")
            ssh.close()

    return True




if __name__ == "__main__":

    from RMS.Logger import initLogging

    # Set up a fake config file
    class FakeConf(object):
        def __init__(self):

            self.username = 'dvida'

            # remote hostname where SSH server is running
            self.hostname = 'gmn.uwo.ca'
            self.host_port = 22
            self.remote_dir = 'files'
            self.remote_platepar_dir = 'platepars'
            self.stationID = 'dvida'
            self.rsa_private_key = os.path.expanduser("~/.ssh/id_rsa")

            self.upload_queue_file = 'FILES_TO_UPLOAD.inf'
            self.platepar_name = 'platepar_cmn2010.cal'
            self.platepar_remote_name = 'platepar_latest.cal'

            self.data_dir = os.path.join(os.path.expanduser('~'), 'RMS_data')
            self.log_dir = 'logs'


    config = FakeConf()

    # Init the logger
    initLogging(config)


    # Test platepar downloading
    downloadNewPlatepar(config)
