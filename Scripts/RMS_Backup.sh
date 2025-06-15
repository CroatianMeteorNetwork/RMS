#!/bin/bash
#
# RMS_Backup.sh: Stores RMS station config files in a compressed .zip files
#
# It tries to detect the default RMS config file in default path. A custom
# RMS config file can be passed as parameter to this script


# Read a property from the config file
function read_property() {
    grep -e "^$1:" $CONFIG_FILE  | cut -d":" -f2 | cut -d ";" -f1 | xargs
}

SCRIPT=$(realpath ${BASH_SOURCE[0]})
SCRIPT_DIR="$(dirname $SCRIPT)/../"

if [ -z $1 ]; then
    RMS_DIR=$(realpath $SCRIPT_DIR)
    CONFIG_FILE="${RMS_DIR}/.config"
else
    CONFIG_FILE="$1"
fi

if [ ! -f $CONFIG_FILE ]; then
    echo "ERROR: RMS config file $CONFIG_FILE not found."
    exit 1
fi

MASK_FILE="${RMS_DIR}/mask.bmp"
PLATEPAR_FILE="${RMS_DIR}/platepar_cmn2010.cal"

STATIONS="${HOME}/source/Stations"

RSA="${HOME}/.ssh"

cd $(dirname $CONFIG_FILE)

STATION_ID=$(read_property "stationID")
TIMESTAMP=$(date +%Y-%m-%d)
BACKUP_FILE=~/RMS_Backup_${STATION_ID}_${TIMESTAMP}.zip

echo "RMS Backup"
echo "----------"
echo
echo "Station ID:....: " $STATION_ID
echo "RMS config.....: " $CONFIG_FILE
echo "Mask...........: " $MASK_FILE
echo "Platepar.......: " $PLATEPAR_FILE
echo "RSA ...........: " $RSA
echo "Stations.......: " $STATIONS
echo
echo "Compressing files..."

zip -q -r $BACKUP_FILE $CONFIG_FILE $MASK_$PLATEPAR_FILE FILE $RSA $STATIONS

echo "Backup saved to: $BACKUP_FILE"

exit 0
