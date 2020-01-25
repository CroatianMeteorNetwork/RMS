#!/usr/bin/python
import os
import getpass
import datetime
from RMS.CaptureDuration import captureDuration

# Get the current user
user = getpass.getuser()


def rmsExternal(captured_night_dir, archived_night_dir, config):

	# Compute the capture duration from now
	start_time, duration = captureDuration(config.latitude, config.longitude, config.elevation)

	timenow = datetime.datetime.utcnow()
	remaining_seconds = 0

	# Compute how long to wait before capture
	if start_time != True:
		
		waitingtime = start_time - timenow
		remaining_seconds = int(waitingtime.total_seconds())		

	# Run the Istrastream shell script
	script_path = os.path.abspath(os.path.join(os.sep, "home", user, "iStream.sh"))
	os.system(script_path + " {:s} {:s} {:s} {:.6f} {:.6f} {:.1f} {:d} {:d} {:d}".format(config.stationID, \
		captured_night_dir, archived_night_dir, config.latitude, config.longitude, config.elevation, \
		config.width, config.height, remaining_seconds))