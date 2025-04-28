#!/bin/bash
#
# bash script to set an IMX291 camera up from scratch
#
echo "This script will set your camera to the recommended settings"
echo "for brightness, video style, gain, and so on. "
echo ""
echo "NB: The script requires that your camera is -already- set to the "
echo "right IP address and that this address has been added to the RMS .config file."
echo ""
echo "If you have not yet configured the camera IP address, press Ctrl-C. "
echo ""
echo "otherwise press any key to continue."
read goonthen

currip=$(python -m Utils.CameraControl GetIP)

echo "Camera Address is $currip"

# Set camera settings using the camera settings json file specified in teh config file
python -m Utils.CameraControl SwitchMode init
python -m Utils.CameraControl reboot


