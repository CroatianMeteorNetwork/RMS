#!/bin/bash
#
# bash script to set an IMX291 camera up from scratch
#
if [ $# -lt 1 ] ; then
    echo "usage1 python -m Utils.CameraControl DIRECT"
    echo "    configure the camera for direct connection to the pi"
    echo "usage1 python -m Utils.CameraControl targetipaddress routeripaddress"
    echo "    configure the camera for connection via your router"
    echo "    the two parameters are the IP address you want the camera to have"
    echo "    and the address of your router." 
    exit 1
fi 

currip=$(python -m Utils.CameraControl GetIP)
if [ "$1" == "DIRECT" ] ; then
    echo Setting direct connection
    echo Warning: you will lose connection to the camera once this completes
else
    if [ $# -lt 2 ] ; then 
        echo direct mode requires you to provide a Camera IP address and your routers IP address
        exit 1
    fi
    echo Setting via-router connection
    camip=$1
    routerip=$2
fi 
echo  "------------------------"

# a few miscellaneous things - onscreen date/camera Id off, colour settings, autoreboot at 1500 every day
python -m Utils.CameraControl SetOSD off
python -m Utils.CameraControl SetColor 100,50,50,50,0,0
python -m Utils.CameraControl SetAutoReboot Everyday,15

# set the Video Encoder parameters
python -m Utils.CameraControl SetParam Encode Video Compression H.264
python -m Utils.CameraControl SetParam Encode Video Resolution 720P
python -m Utils.CameraControl SetParam Encode Video BitRateControl VBR
python -m Utils.CameraControl SetParam Encode Video FPS 25
python -m Utils.CameraControl SetParam Encode Video Quality 6
python -m Utils.CameraControl SetParam Encode AudioEnable 0
python -m Utils.CameraControl SetParam Encode VideoEnable 1
python -m Utils.CameraControl SetParam Encode SecondStream 0

# camera parameters
python -m Utils.CameraControl SetParam Camera Style type1
python -m Utils.CameraControl SetParam Camera AeSensitivity 1
python -m Utils.CameraControl SetParam Camera ApertureMode 0
python -m Utils.CameraControl SetParam Camera BLCMode 0
python -m Utils.CameraControl SetParam Camera DayNightColor 2
python -m Utils.CameraControl SetParam Camera Day_nfLevel 0
python -m Utils.CameraControl SetParam Camera DncThr 50
python -m Utils.CameraControl SetParam Camera ElecLevel 100
python -m Utils.CameraControl SetParam Camera EsShutter 0
python -m Utils.CameraControl SetParam Camera ExposureParam LeastTime 40000
python -m Utils.CameraControl SetParam Camera ExposureParam Level 0
python -m Utils.CameraControl SetParam Camera ExposureParam MostTime 40000
python -m Utils.CameraControl SetParam Camera GainParam AutoGain 1
python -m Utils.CameraControl SetParam Camera GainParam Gain 60
python -m Utils.CameraControl SetParam Camera IRCUTMode 0
python -m Utils.CameraControl SetParam Camera IrcutSwap 0
python -m Utils.CameraControl SetParam Camera Night_nfLevel 0
python -m Utils.CameraControl SetParam Camera RejectFlicker 0
python -m Utils.CameraControl SetParam Camera WhiteBalace 2
python -m Utils.CameraControl SetParam Camera PictureFlip 0
python -m Utils.CameraControl SetParam Camera PictureMirror 0

# network parameters
python -m Utils.CameraControl SetParam Network EnableDHCP 0
python -m Utils.CameraControl SetParam Network TransferPlan Fluency

echo  "------------------------"
echo "about to update the camera IP address. You will see a timeout message"
if [ "$1" == "DIRECT" ] ; then
    python -m Utils.CameraControl SetParam Network GateWay 192.168.42.1
    python -m Utils.CameraControl SetParam Network HostIP 192.168.42.10
    python -m Utils.CameraControl SetParam Network EnableDHCP 1
else
    python -m Utils.CameraControl SetParam Network GateWay $routerip
    python -m Utils.CameraControl SetParam Network HostIP $camip
fi
echo "------------------------"
echo "updating config file"
cat .config | sed "s/$currip/$camip/g" > tmp.tmp
mv .config .config.orig
mv tmp.tmp .config

echo "------------------------"
echo "the camera will now reboot.... "
sleep 5

if [ "$1" == "DIRECT" ] ; then
    echo "now plug the camera into the Pi"
else
    currip=$(python -m Utils.CameraControl GetIP)
    echo Camera ip is now $currip
fi