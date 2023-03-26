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
cd ~/source/Stations/$1

# Init log file
LOGPATH=~/RMS_data/logs/
LOGDATE=$(date +"%Y%m%d_%H%M%S")
LOGSUFFIX="_log.txt"
LOGFILE=$LOGPATH$LOGDATE$LOGSUFFIX

mkdir -p $LOGPATH

# Log the output to a file (warning: this breaks Ctrl+C passing to StartCapture)
#python -m RMS.StartCapture 2>&1 | tee $LOGFILE

python -m RMS.StartCapture

read -p "Press any key to continue... "

$SHELL
