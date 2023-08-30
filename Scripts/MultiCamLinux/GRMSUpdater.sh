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

# Version 1.7 changed parsing of the current username and display number to be agnostic to either the display manager or the session manager
#
# Version 1.6 bug fixes -
# fixed this script failing when run under cron
# changed behaviour to allow for consistent run-line parsing irrespective of how a capture was initiated
# be it autostart, desktop icon or this script itself
#
# Version 1.5 fixed bugs introduced by version 1.4 changes.
#
# Version 1.5 fixed untested issue introduced by 1.4
#
# Version 1.4 fixed path issue in Run/Pid list variables
#
# Version 1.3 moved codebase into RMS/Scripts/MultiCamLinux
#
# Version 1.2 numerous fixes -
# added support for a delayed start
# fixed bug  parsing the running processes, used when called with an argument
# Version 1.1
# Changes: Fixed a bug whereby the list of running RMS processes was incorrect when the script was called with 
# an argument and the script was invoked by cron (i.e. root)
#
# read the  PID's of all RMS processes into an array and then read the number of running stations into 
# array RunList so that after killing the instances we can then update RMS and then restart the stations.
# Default behaviour if called with no arguments, - capture all the running RMS processes, kill them, update RMS, then start 
# all that are configured within directory ~/source/Stations -

UserDisp=($(w -h | awk '/\ :[0-9]/ {print $1,$3}')) # grab the RMS username and current display number
RunList=( $( ps -ef|grep -E -w -o '\/bin\/bash .*\/source\/RMS\/Scripts\/MultiCamLinux\/StartCapture.sh\ [[:alnum:]]{6}'| awk '{print $NF}' | sort -u )) # create an array of the running station names


PidList=( $(ps -ef|awk '/\/bin\/bash .*\/source\/RMS\/Scripts\/MultiCamLinux\/StartCapture.sh/ {print $2}') ) 	# create an array of the running RMS processes


        for Pid in "${PidList[@]}"
                do
		kill $Pid
                done

# This script when run as a crontab entry runs under root's account so we need to set some variables to allow the 
# script to launch apps onto the users desktop -which it doesn't own. We also use this to build the full paths to any
# files since jobs running under cron don't inherit normal users ENV variables
# Note: this may well break if the distro used doesn't use xdm or if the user e.g. connects via xrdp....

export XAUTHORITY=/home/${UserDisp[0]}/.Xauthority
export DISPLAY=${UserDisp[1]}


if [[  -z "$1" ]] 					# called with no args
then
echo " ....Will restart all configured stations post-update"
unset RunList
	for Dir in /home/${UserDisp[0]}/source/Stations/*
		do
			if [[  "${Dir##*/}" != "Scripts" ]]
			then
			RunList+=(${Dir##*/}) 		# build an array of the configured stations
			fi
		done
                /home/${UserDisp[0]}/source/RMS/Scripts/RMS_Update.sh >/dev/null
		sleep 10
		for Station in "${RunList[@]}"		# restart all that are configured
        	do
		# get the runline from this stations Desktop link
		Delay=$(grep Exec ~/Desktop/${Station}_StartCap.desktop| sed 's/Exec=\(.*\).*\"$/\1/g'|awk '{print $(NF)}')
		if [[ "$Delay" != "$Station" ]]
		then
			sleep 5
			lxterminal --title=${Station} -e "/home/${UserDisp[0]}/source/RMS/Scripts/MultiCamLinux/StartCapture.sh ${Station} $Delay"  &
			else
			sleep 5
			lxterminal --title=${Station} -e "/home/${UserDisp[0]}/source/RMS/Scripts/MultiCamLinux/StartCapture.sh ${Station}"  &
		fi
	 	done
else

# If called with any argument then all running RMS processes are killed and RMS updated
# however only those only those stations that were actually running when called are restarted
# and not all the stations that are configured. 
# Handy if say out of 3 stations, one is not running due to a camera issue



	/home/${UserDisp[0]}/source/RMS/Scripts/RMS_Update.sh >/dev/null 
	sleep 10
	for Station in "${RunList[@]}"
		do
			sleep 5
                        lxterminal --title=${Station} -e "/home/${UserDisp[0]}/source/RMS/Scripts/MultiCamLinux/StartCapture.sh ${Station} $Delay"  &
		done
fi

