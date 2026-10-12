#!/bin/bash

# This software is part of the Linux port of RMS
# Copyright (C) 2023  Ed Harman
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
#
# Version 1.2 - fixed bug whereby crontab runs this script under /bin/sh instead of /bin/bash
#
# Version 1.1
# Changes - added station and delay arguments
#
# first command line arg should be stationID: XX000X
# second command line arg is optional, can be used to adjust the sleep time in seconds
# so that there can be a delay between starting each station

delay_by_station_sequence() {

# implement a delay based on position of station in directory
# this is not used presently

delay=$(ls /home/$(whoami)/source/Stations/ | grep -n $1 | cut -d':' -f1)
delay=$((delay*30))
echo Additional $delay second delay will be added.
sleep $delay

}


if [[  -z "$1" ]]	# called with no args
then
	echo " No Station directory specified, quitting now"
	exit
fi

echo "Starting RMS..."

if [[  -z "$2" ]]	# called with no second arg
then
	sleep 2
else
	sleep $2
fi

source "$HOME/vRMS/bin/activate"
cd "$HOME/source/RMS"

configpath="/home/$(whoami)/source/Stations/$1/.config"
echo "Using config from $configpath"

# ----- decide how we were launched ---------------------------------
# real TTY (manual or .desktop launch)
if [[ -t 1 ]]; then
    # TTY mode: output goes to screen (no additional logging, RMS logs internally)

    python -u -m RMS.StartCapture -c "$configpath" &
    child=$!

    # Handle signals: SIGINT (user Ctrl-C) shows message, SIGTERM (GRMSUpdater) doesn't
    handle_signal() {
        local sig=$1
        
        # Always forward SIGINT to Python (it handles this better than SIGTERM)
        if [[ "$sig" == "TERM" ]]; then
            kill -INT "$child" 2>/dev/null  # Forward SIGINT instead of SIGTERM
        else
            kill -"$sig" "$child" 2>/dev/null
        fi
        
        wait "$child"
        status=$?
        
        # Only show message for SIGINT (user Ctrl-C), not SIGTERM (automated)
        if [[ "$sig" == "INT" ]]; then
            read -n1 -r -p "Capture ended (exit $status) - press any key to close..."
        fi
        exit "$status"
    }
    
    trap 'handle_signal INT' INT
    trap 'handle_signal TERM' TERM
    wait "$child"
    status=$?

    # keep the window open for inspection only when user-started (GRMS_AUTO is unset)
    if [[ -z "${GRMS_AUTO:-}" ]]; then
        read -n1 -r -p "Capture ended (exit $status) - press any key to close..."
    fi

    exit "$status"

else
    # no TTY (cron / GRMSUpdater / nohup etc.)
    # Run Python as a child process so bash stays alive with the station ID
    # in its command line - this allows GRMSUpdater to find and signal the process.
    # Use job control to ensure proper signal handling and output inheritance.
    set -m  # Enable job control
    python -u -m RMS.StartCapture -c "$configpath" &
    child=$!

    # Forward SIGTERM as SIGINT to Python for graceful shutdown
    trap 'kill -INT "$child" 2>/dev/null; wait "$child"; exit $?' TERM INT
    wait "$child"
fi
