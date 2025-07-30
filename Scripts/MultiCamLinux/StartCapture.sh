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

source ~/vRMS/bin/activate
cd ~/source/RMS

LOGDIR=~/RMS_data/logs
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/$(date +%F_%T)_startcap.log"

configpath="/home/$(whoami)/source/Stations/$1/.config"
echo "Using config from $configpath"

# ----- decide how we were launched ---------------------------------
if [[ -t 1 ]]; then            # we have a real TTY → manual or .desktop launch
    echo "Logging to $LOGFILE"
    
    # duplicate output but shield tee from SIGINT
    exec > >(bash -c 'trap "" INT TERM; tee -a "$1"' _ "$LOGFILE") 2>&1

    python -u -m RMS.StartCapture -c "$configpath" &
    child=$!

    # forward INT/TERM from user to the child, then wait for it
    forward() { kill -"$1" "$child" 2>/dev/null; }
    trap 'forward INT'  INT
    trap 'forward TERM' TERM
    wait "$child"
    status=$?

    # keep the window open for inspection
    read -n1 -r -p "Capture ended (exit $status) - press any key to close…"

else                            # cron / GRMSUpdater / nohup etc.
    # no TTY – just append to the log file
    { exec -a "StartCapture.sh $1" \
        python -u -m RMS.StartCapture -c "$configpath"; } 2>&1 | tee -a "$LOGFILE"
fi
