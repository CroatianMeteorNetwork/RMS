#!/bin/bash
echo "Starting RMS..."
sleep 10
source ~/vRMS/bin/activate
cd ~/source/RMS

# Init log file
LOGPATH=~/RMS_data/logs/
LOGDATE=$(date +"%Y%m%d_%H%M%S")
LOGSUFFIX="_log.txt"
LOGFILE=$LOGPATH$LOGDATE$LOGSUFFIX

mkdir -p $LOGPATH

echo ""
echo ""
echo "If you need to update the RMS config file, you can do it now."
echo "Any changes to the config file will be read only after this script is started again or the Pi is rebooted."
echo ""
sleep 5

python -m RMS.StartCapture "$@"

read -p "Press any key to continue... "

$SHELL
