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

cd $(dirname $CONFIG_FILE)

RSA_PRIV_FILE=$(read_property "rsa_private_key")
RSA_PRIV_FILE="${RSA_PRIV_FILE/#\~/$HOME}"

RSA_PUB_FILE=${RSA_PRIV_FILE}.pub

PLATEPAR_FILE=$(read_property "platepar_name")
PLATEPAR_FILE=$(realpath "${PLATEPAR_FILE/#\~/$HOME}")

MASK_FILE=$(read_property "mask")
MASK_FILE=$(realpath "${MASK_FILE/#\~/$HOME}")

STATION_ID=$(read_property "stationID")
TIMESTAMP=$(date +%Y-%m-%d)
BACKUP_FILE=~/RMS_Backup_${STATION_ID}_${TIMESTAMP}.zip

echo "RMS Backup"
echo "----------"
echo
echo "Station ID:....: " $STATION_ID
echo "RMS config.....: " $CONFIG_FILE
echo "RSA private key: " $RSA_PRIV_FILE
echo "RSA public key.: " $RSA_PUB_FILE
echo "Platepar.......: " $PLATEPAR_FILE
echo "Mask...........: " $MASK_FILE
echo
echo "Compressing files..."


zip -q --junk-paths $BACKUP_FILE $CONFIG_FILE $RSA_PRIV_FILE $RSA_PUB_FILE $PLATEPAR_FILE $MASK_FILE
echo "Backup saved to: $BACKUP_FILE"

exit 0
