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

# Log the output to a file (warning: this breaks Ctrl+C passing to StartCapture)
#python -m RMS.StartCapture 2>&1 | tee $LOGFILE

# start a new tmux session if not already inside one and tmux is available
if [ -z "$TMUX" ] && type tmux >/dev/null 2>&1; then
  tmux new-session -s RMS python -m RMS.StartCapture "$@"
else
  python -m RMS.StartCapture "$@"
fi

read -p "Press any key to continue... "

$SHELL
