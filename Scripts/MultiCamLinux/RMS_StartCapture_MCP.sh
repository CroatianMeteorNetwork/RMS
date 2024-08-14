#!/bin/bash

echo " Starting all configured stations post-update..."
for Dir in ~/source/Stations/*
  do
	Station=$(basename $Dir)
	echo "Starting camera ${Station}"
	lxterminal --title=${Station} -c "~/source/RMS/Scripts/MultiCamLinux/StartCapture.sh ${Station}"  &
	sleep 30
  done
