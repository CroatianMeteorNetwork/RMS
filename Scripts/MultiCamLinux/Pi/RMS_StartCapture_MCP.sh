#!/bin/bash

# On a single camera system there is no risk of concurrent first-start updates,
# so the long delay is only needed when more than one station is configured
dircount=$(find ~/source/Stations -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
if [[ $dircount -le 1 ]]; then
    seconds=2
else
    seconds=70
fi

echo " Starting all configured stations post-update..."

loop=0
for Dir in ~/source/Stations/*
  do
	Station=$(basename $Dir)
	echo " Starting camera ${Station}"
	lxterminal --title=${Station} -e "$HOME/source/RMS/Scripts/MultiCamLinux/StartCapture.sh ${Station}"  &
	echo "  waiting $seconds seconds..."
	sleep ${seconds}
	if [[ $loop = 0 ]] ; then
	    seconds=10
	fi
	let loop++
  done
echo " All cameras started"
