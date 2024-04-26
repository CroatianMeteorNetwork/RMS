#!/bin/bash
if [[ -d /home/${USER}/source/Stations ]]
then
for Dir in /home/${USER}/source/Stations/*
    do screen -dmS $(basename $Dir) /home/${USER}/source/RMS/Scripts/MultiCamLinux/StartCapture.sh $(basename $Dir)
    echo "Launched and detached station $(basename $Dir)"
    done
fi
