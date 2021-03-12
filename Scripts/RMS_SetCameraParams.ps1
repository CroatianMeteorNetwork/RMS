#
# powershell script to set an IMX291 camera up from scratch
#
if ( $args.count -eq 0 ){
    Write-Output "usage1 python -m Utils.CameraControl DIRECT"
    Write-Output "    configure the camera for direct connection to the pi"
    Write-Output "usage1 python -m Utils.CameraControl targetipaddress routeripaddress"
    Write-Output "    configure the camera for connection via your router"
    write-output "    the two parameters are the IP address you want the camera to have"
    write-output "    and the address of your router." 
    exit 1
}
Write-Output "------------------------"
Write-Output "this script assumes that the RMS config file has not been changed and the"
Write-Output "device string still contains the IP address 192.168.42.10"
Write-Output ""
Write-Output "if this is not the case, press Ctrl-C and edit it before proceeding"
read-host -prompt "or press any other key to continue"

Write-Output "updating config file to use factory IP address 192.168.1.10"
$currip="192.168.42.10"
$defaultip="192.168.1.10"

(Get-Content .config).replace("$currip", "$defaultip") | Set-Content tmp.tmp
Move-Item .config .config.orig -force
Move-Item tmp.tmp .config -force

#$currip=$(python -m Utils.CameraControl GetIP)
if ($args[0] -eq "DIRECT"){
    write-output Setting direct connection
    write-output Warning: you will lose connection to the camera once this completes
}else{
    if ($args.count -lt 2){
        write-output "direct mode requires you to provide a Camera IP address and your routers IP address"
        exit 1
    }
    write-output "Setting via-router connection"
    $camip=$args[0]
    $routerip=$args[1]
}
write-output "------------------------"
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

write-output "------------------------"
write-output "about to update the camera IP address. You will see a timeout message"
if ($args[0] -eq "DIRECT"){
    python -m Utils.CameraControl SetParam Network GateWay 192.168.42.1
    python -m Utils.CameraControl SetParam Network HostIP 192.168.42.10
    python -m Utils.CameraControl SetParam Network EnableDHCP 1
}else{
    python -m Utils.CameraControl SetParam Network GateWay $routerip
    python -m Utils.CameraControl SetParam Network HostIP $camip
}
write-output "------------------------"
write-output "updating config file"
(Get-Content .config).replace("$defaultip", "$camip") | Set-Content tmp.tmp
Move-Item tmp.tmp .config -force
write-output "------------------------"
write-output "the camera will now reboot.... "
Start-Sleep 5

if ($args[0] -eq "DIRECT"){
    write-output "now plug the camera into the Pi"
}else{
    $camip=$(python -m Utils.CameraControl GetIP)
    write-output "Camera ip is now $camip"
}