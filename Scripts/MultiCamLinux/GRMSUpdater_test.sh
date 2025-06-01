#!/bin/bash

# Debug log file
exec 1> >/tmp/rms_restart_debug.log 2>&1
date

echo "Script starting..."

# Debug: Print all environment variables
echo "Environment variables:"
env | sort

echo "Checking who output:"
who
echo "---"

echo "Checking display-having users:"
who | awk '/\ :[0-9]/'
echo "---"

UserDisp=($(who | awk '/\ :[0-9]/ {print $1,$2}'))
echo "UserDisp array contents:"
printf '%s\n' "${UserDisp[@]}"

echo "Current running RMS processes:"
ps -ef | grep -E "RMS.StartCapture|StartCapture.sh"

# Just do the kill part for now
PidList=( $(ps -ef|awk '/\/bin\/bash .*\/source\/RMS\/Scripts\/MultiCamLinux\/StartCapture.sh/ {print $2}') )
echo "PIDs to kill:"
printf '%s\n' "${PidList[@]}"

echo "Attempting to kill processes..."
for Pid in "${PidList[@]}"
do
    echo "Killing $Pid"
    kill $Pid
    echo "Kill exit code: $?"
done

echo "Processes after kill:"
ps -ef | grep -E "RMS.StartCapture|StartCapture.sh"

# Comment out the rest of the script for now
: <<'END_COMMENT'
export XAUTHORITY=/home/${UserDisp[0]}/.Xauthority
export DISPLAY=${UserDisp[1]}
... rest of original script ...
END_COMMENT

echo "Script finished"