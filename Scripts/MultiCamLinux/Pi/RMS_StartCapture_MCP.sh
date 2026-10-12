#!/bin/bash

seconds=70
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
