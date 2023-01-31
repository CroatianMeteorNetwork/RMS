#!/usr/bin/python
import os
import sys
import traceback
import subprocess
import datetime
import logging
from RMS.CaptureDuration import captureDuration
from RMS.Logger import initLogging
from RMS.Misc import isRaspberryPi


def rmsExternal(captured_night_dir, archived_night_dir, config):
    """ This function is called by RMS when the capture is finished. It is used to run external scripts
        after the capture is finished.

    Arguments:
        captured_night_dir: [str] Path to the directory where the captured night is stored.
        archived_night_dir: [str] Path to the directory where the archived night is stored.
        config: [Config] Configuration object.
    """


    # Initialize the logger
    initLogging(config, 'iStream_')
    log = logging.getLogger("logger")
    log.info('iStream external script started')

    # Create lock file to avoid RMS rebooting the system
    lockfile = os.path.join(config.data_dir, config.reboot_lock_file)
    with open(lockfile, 'w') as _:
        pass

    # Compute the capture duration from now
    start_time, duration = captureDuration(config.latitude, config.longitude, config.elevation)

    timenow = datetime.datetime.utcnow()
    remaining_seconds = 0

    # Compute how long to wait before capture
    if start_time is not True:
        waitingtime = start_time - timenow
        remaining_seconds = int(waitingtime.total_seconds())		

    # Run the Istrastream shell script
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iStream.sh")
    log.info('Calling {}'.format(script_path))

    command = [
        script_path,
        config.stationID,
        captured_night_dir,
        archived_night_dir,
        '{:.6f}'.format(config.latitude),
        '{:.6f}'.format(config.longitude),
        '{:.1f}'.format(config.elevation),
        str(config.width),
        str(config.height),
        str(remaining_seconds)
    ]

    proc = subprocess.Popen(command,stdout=subprocess.PIPE)
   
    # Read iStream script output and append to log file
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        log.info(line.rstrip().decode("utf-8"))

    exit_code = proc.wait()
    log.info('Exit status: {}'.format(exit_code))
    log.info('iStream external script finished')

    # Relase lock file so RMS is authorized to reboot, if needed
    os.remove(lockfile)

    # Only reboot RPis, don't reboot Linux machines
    if isRaspberryPi():

        # Reboot the computer (script needs sudo priviledges, works only on Pis)
        try:
            log.info("Rebooting system...")
            os.system('sudo shutdown -r now')
            
        except Exception as e:
            log.debug('Rebooting failed with message:\n' + repr(e))
            log.debug(repr(traceback.format_exception(*sys.exc_info())))
