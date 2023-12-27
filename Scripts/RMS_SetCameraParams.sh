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

# a few miscellaneous things - onscreen date/camera Id off, colour settings, autoreboot at 1500 every day, set time
python -m Utils.CameraControl SetOSD off
python -m Utils.CameraControl SetColor 100,50,50,50,0,0
python -m Utils.CameraControl SetAutoReboot Everyday,15
python -m Utils.CameraControl CameraTime set

# disable phone-home remote connectivity to server in China
python -m Utils.CameraControl CloudConnection off

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
python -m Utils.CameraControl SetParam Camera WhiteBalance 2
python -m Utils.CameraControl SetParam Camera PictureFlip 0
python -m Utils.CameraControl SetParam Camera PictureMirror 0

# network parameters
python -m Utils.CameraControl SetParam Network TransferPlan Fluency

