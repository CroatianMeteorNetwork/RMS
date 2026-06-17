#!/bin/bash

echo "Waiting a few seconds for the system to initialize..."
echo ""
echo "IMPORTANT: RMS will first update itself."
echo "Do not touch any file during the update and do not close this window."
sleep 12

# Google DNS server IP
IP='8.8.8.8'

# Contact e-mail
RMSEMAIL="denis.vida@gmail.com"

# RMS config file
RMSCONFIG=~/source/RMS/.config

# Auto run enable flag file
RMSAUTORUNFILE=~/.rmsautorunflag

# The StartCapture Desktop entry is a symlink that add_Station.sh retargets
# to the multi-camera launcher after a conversion, so it is used on purpose
RMSSTARTCAPTURE=~/Desktop/RMS_StartCapture.sh
RMSUPDATESCRIPT=~/source/RMS/Scripts/RMS_Update.sh


# Check that a value is a number within the given range
isNumberInRange () {   # $1 = value, $2 = min, $3 = max
  [[ "$1" =~ ^-?[0-9]+(\.[0-9]+)?$ ]] || return 1
  awk -v v="$1" -v lo="$2" -v hi="$3" 'BEGIN { exit !(v >= lo && v <= hi) }'
}



# If the autorun file does not exist, create it and run the configuration
if [ ! -f $RMSAUTORUNFILE ]; then
  echo "0" > $RMSAUTORUNFILE
else
  # If the autorun file exists, check if the configuration was already done
  AUTORUNSTATUS=$(cat $RMSAUTORUNFILE)

  if [ "$AUTORUNSTATUS" = "1" ]; then

    echo ""
    echo "Updating to the latest version of RMS..."
    bash $RMSUPDATESCRIPT

    # If the configuration was done, run recording
    bash $RMSSTARTCAPTURE
    exit 0
  fi
fi

# If autorun is not enabled, run the first setup

reset

echo "Hey, welcome to the Raspberry Pi Meteor Station (RMS) project!

This guide will help you to get your system up and running in no time!


IMPORTANT! Before you proceed make sure you have the following:

 1. Geo location of your camera (latitude +N, longitude +E, elevation).
   The latitude and longitude should be in degrees to at least 5 decimal
   places. Be careful that the longitude of places in the western hemisphere
   is negative. E.g. if your camera was installed on the Statue of Liberty,
   the latitude would be 40.689298 and the longitude would be -74.044479.
   The elevation should be given in meters (NOT feet!) and in the mean sea level
   (MSL) convention (not WGS84).
   The easiest way to measure the coordinates is in Google Earth. Make sure to
   pinpoint the actual location of the camera to within a precision of at least
   10 meters.

 2. Unique station code. To obtain a station code, send an e-mail containing
    the following to Denis Vida ($RMSEMAIL):
      a) Geo coordinates from step 1.
      b) Your country.
      c) Brief description of the camera system (location, owner's name),
         e.g. Mike Henderson's camera in London, Ontario, Canada.

    You will then be given a unique station code which will look something
    like this: US01AB. The first two letters is the ISO code of the country,
    and the last 4 characters are the unique alphanumeric identifier.

IMPORTANT: READ THE INSTRUCTIONS ABOVE BEFORE CONTINUING!

Press Q to continue.
" | less -X

echo "-----------------"
echo ""

echo "If you DON'T have the geo coordinates and/or the station code, press CTRL + C
to exit this guide.

Otherwise, press ENTER to continue."

read -p ""


echo ""
echo "This is a brief overview of this guide:"
echo "  0. Expand the file system (if you flashed this SD card yourself)."
echo "  1. Connect your Pi to the Internet. "
echo "  2. Change the default password for security reasons."
echo "  3. Generate a new SSH key."
echo "  4. Enter the station ID and the geo coordinates of your camera."
echo "  5. Convert to the multi-camera data structure (recommended)."
echo ""
echo "At the end of these steps the first data capture session will start."
echo ""

# Expanding the file system only applies to Raspberry Pi SD card images
if command -v raspi-config >/dev/null 2>&1; then

echo "
0) Expanding the file system
----------------------------
If you have bought a system that was already assembled, or the file system
has already been expanded, press any key to skip this step.
"

read -n1 -r -p 'If you have flashed this SD card yourself, press ENTER.' key

if [[ "$key" = "" ]]; then
    lxterminal -e "sudo raspi-config"

    echo "
    Another window has opened where you can expand the file system.

    Go to:

    7 ADVANCED OPTIONS -->
      A1 EXPAND FILE SYSTEM -->
        < OK >

    < FINISH > (at the bottom) -->
      < YES > (reboot)

    Your Raspberry Pi will reboot.
    "

    read -p "Press ENTER to continue..."
fi

fi  # raspi-config available

echo ""
echo ""

# Check if connected to the Internet
echo "1) Internet connection"
echo "----------------------"
echo "Checking if connected to the Internet..."
# Some networks block ping, so fall back to an HTTP check of the server
# the update needs to reach anyway
if ping -c 1 $IP &> /dev/null || wget -q --spider --timeout=10 https://github.com
then
  echo "Success!"
else
  echo "The device is not connected to the internet! Please connect the device to the Internet to proceed!"
  read -p "Press ENTER to continue..."
  exit 1
fi

echo ""
echo "2) Changing the default password"
echo "--------------------------------"
echo "The default password is either 'raspberry' or 'rmsraspberry'. Please change it so nobody can connect to your Raspberry Pi and hack the computers on your network!"

echo ""
read -n1 -r -p 'Press ENTER to change the password (recommended), or any other key to skip this step...' key

if [[ "$key" = "" ]]; then
    passwd
fi

echo ""
echo ""
echo "3) Generating a new SSH key"
echo "---------------------------"

read -n1 -r -p 'Press ENTER to generate the SSH key (recommended), or any other key to skip this step...' key

if [[ "$key" = "" ]]; then

  if [[ -f ~/.ssh/id_rsa ]]; then

    # Never overwrite an existing key - GMN already has its public half on
    # file, and replacing it would break the data uploads
    echo ""
    echo "An SSH key already exists in ~/.ssh, not overwriting it."

  else
    echo ""
    echo "Generating a new SSH key..."

    # Generate an SSH key without a passphrase
    ssh-keygen -t rsa -m PEM -N "" -f ~/.ssh/id_rsa >/dev/null
  fi

  # Link the public SSH key to desktop
  ln -sf ~/.ssh/id_rsa.pub ~/Desktop/id_rsa.pub

  echo ""
  echo "A file called id_rsa.pub appeared on Desktop, please send this file to Denis "
  echo "Vida ($RMSEMAIL) before continuing!"

  read -p "Press ENTER to continue"

fi

echo ""
echo "Updating to the latest version of RMS..."
bash $RMSUPDATESCRIPT

echo ""
echo "4) Station configuration"
echo "------------------------"
echo "Enter the station ID and the geo coordinates of your camera."
echo ""

echo "
If you need to change any other setting in the future, you can find a
shortcut to the configuration file on desktop (RMS_config), or open it
directly in $RMSCONFIG.
"

while true; do

  # Station ID
  while true; do
    read -p "Station ID (e.g. US01AB): " statID || exit 1
    statID=$(echo "$statID" | xargs | tr '[:lower:]' '[:upper:]')
    if [[ "$statID" =~ ^[A-Z]{2}[A-Z0-9]{4}$ && "$statID" != "XX0001" ]]; then
      break
    fi
    echo "  A station code has 2 country letters followed by 4 characters, e.g. US01AB."
    if [[ -n "$statID" ]]; then
      read -n1 -r -p "  Use '$statID' anyway? Press Y to accept it, any other key to retype... " key
      echo ""
      if [[ "$key" = "y" || "$key" = "Y" ]]; then
        break
      fi
    fi
  done

  # Latitude
  while true; do
    read -p "Latitude (+N, in degrees, e.g. 40.689298): " lat || exit 1
    lat=$(echo "$lat" | xargs)
    isNumberInRange "$lat" -90 90 && break
    echo "  The latitude must be a number between -90 and 90."
  done

  # Longitude
  while true; do
    read -p "Longitude (+E, in degrees, NEGATIVE in the western hemisphere, e.g. -74.044479): " lon || exit 1
    lon=$(echo "$lon" | xargs)
    isNumberInRange "$lon" -180 180 && break
    echo "  The longitude must be a number between -180 and 180."
  done

  # Elevation
  while true; do
    read -p "Elevation (mean sea level, in meters, NOT feet): " elev || exit 1
    elev=$(echo "$elev" | xargs)
    isNumberInRange "$elev" -500 9000 && break
    echo "  The elevation must be a number in meters, e.g. 95.3."
  done

  echo "
Station ID: $statID
Latitude:   $lat
Longitude:  $lon
Elevation:  $elev m
"
  read -n1 -r -p 'Press ENTER to save these values, or R to re-enter them... ' key
  echo ""
  if [[ "$key" = "" ]]; then
    break
  fi

done

# Write the values into the config file
sed -i "s/^stationID:.*$/stationID: $statID/" $RMSCONFIG
sed -i "s/^latitude:.*$/latitude: $lat/" $RMSCONFIG
sed -i "s/^longitude:.*$/longitude: $lon/" $RMSCONFIG
sed -i "s/^elevation:.*$/elevation: $elev/" $RMSCONFIG

echo "Configuration saved to $RMSCONFIG"

# Offer the conversion to the multi-camera data structure, unless the
# system already uses it
if [[ ! -d ~/source/Stations ]]; then

echo "
5) Converting the data structure
--------------------------------
The new recommended data structure keeps important camera files in a safer
place. Converting to this multi-camera data structure is recommended for
both single and multiple camera systems.

The important files (.config, mask.bmp and platepar_cmn2010.cal) will be
copied to ~/source/Stations/<camera_ID>, and captured data will be stored
in ~/RMS_data/<camera_ID>, where <camera_ID> is the unique station code
you were given by GMN. Nothing in ~/source/RMS is modified.
"
sleep 1

# clear the input buffer
while read -t 0.01; do :; done

read -n1 -r -p 'Press ENTER to convert to the new data structure, or any other key to stay with the legacy structure... ' key
if [[ "$key" != "" ]]; then
  echo ""
  echo ""
  echo "Are you sure you want to stay with the legacy structure?"
  read -n1 -r -p 'Press any key to confirm, or ENTER to convert to the new data structure... ' key
  echo ""
fi

if [[ "$key" = "" ]]; then
  echo ""
  echo "Next, this script will run to convert to the recommended data structure"
  echo "(it will show the details and ask you to confirm):"
  echo "  ~/source/RMS/Scripts/add_Station.sh"
  echo ""
  echo "Note: you can run the same script later to add more cameras."
  echo ""
  sleep 2
  bash ~/source/RMS/Scripts/add_Station.sh
else
  echo ""
  echo "You have chosen to keep the legacy data structure."
  echo "If you decide to convert in the future, run this command from a terminal:"
  echo "  ~/source/RMS/Scripts/add_Station.sh"
fi

fi  # if no ~/source/Stations yet

# Enable autorun. This is done only after the optional data structure
# conversion so an interrupted conversion does not get skipped over by
# the autorun path on the next boot.
echo "1" > $RMSAUTORUNFILE

sleep 2
echo ""
echo "Configuration is done"
echo ""
read -n1 -r -p 'Press ENTER when you are ready to start camera data capture... '
echo ""

# If the configuration was done, run recording
bash $RMSSTARTCAPTURE
