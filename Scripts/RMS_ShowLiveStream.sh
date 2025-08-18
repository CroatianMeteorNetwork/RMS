#!/bin/bash
echo "Starting RMS live stream..."

source ~/vRMS/bin/activate
cd ~/source/RMS
python -m Utils.ShowLiveStream

read -p "Press any key to continue... "
$SHELL