""" Controls CMS-compatible IP camera 
    This module can read and control the IMX291 Cameras and possibly other IMX cameras
    that comply with the dvrip protocol and can be controlled from CMS.

    Note:the DVRIP module used by this code requires Python 3.5 or later.

# Usage examples:
#
# pip install onvif_zeep
# python -m RMS.CameraControl reboot
# python -m RMS.CameraControl getDeviceInformation

from __future__ import print_function

    Parameters:
    command - the command you want to execute. 
    opts - array containing field and value to use when calling SetParam

    example
    python -m Utils.CameraControl GetCameraParams
    python -m Utils.CameraControl GetEncodeParams
    will return JSON showing you the contents of the Camera Params block and Video Encoding blocks 

    You can set these values using SetParam eg
    set the AE Reference to 100
    python -m Utils.CameraControl SetParam Camera ElecLevel 100
    Set the Gain to 70
    python -m Utils.CameraControl SetParam Camera GainParam Gain 70
    Set the minimum exposure time to 40ms
    python -m Utils.CameraControl SetParam Camera ExposureParam LeastTime 40000

    Note that some fields have subfields as shown in the last example
"""

import sys, os
if sys.version_info.major < 3 or (sys.version_info.major > 2 and sys.version_info.minor < 5) :
        print('This module can only be used with Python 3.5 or later')
        print('Please use CameraControl27 for older versions of Python')
        exit()

import ipaddress as ip
import argparse
import re
import RMS.ConfigReader as cr
from time import sleep
import git, importlib  #used to import python-dvr as it has a dash in the name

# if not present, force update of the submodule
try:
    dvr = importlib.import_module("python-dvr.dvrip")
except:
    print("updating python-dvr")
    rmsloc,_= os.path.split(os.path.realpath(__file__))
    rmsrepo=git.Repo(rmsloc)
    for sm in rmsrepo.submodules:
        sm.update(init=True, force=True)
    try:
        dvr = importlib.import_module("python-dvr.dvrip")
    except:
        print('unable to update python-dvr - can\'t continue')
        exit()

def rebootCamera(cam):
    """Reboot the Camera

    Args:
        cam : The camera
    """
    print('rebooting, please wait....')
    cam.reboot()
    sleep(60) # wait while camera starts
    if cam.login():
        print('reboot successful')
    else:
        print('camera nonresponsive, please wait 30s and reconnect')

def iptoString(s):
    """Convert an IP address in hex network order to a human readable string 

    Args:
        s (string): the encoded IP address eg '0x0A01A8C0' 

    Returns:
        string: human readable IP in host order eg '192.169.1.10'
    """
    a=s[2:]
    addr='0x'+''.join([a[x:x+2] for x in range(0,len(a),2)][::-1])
    ipaddr=ip.IPv4Address(int(addr,16))
    return ipaddr

def saveToFile(nc,dh,cs,vs):
    """Save the camera config to JSON files 

    Args:
        nc : network config
        dh : dhcp config
        cs : camera config
        vs : video encoding config
    """
    if not os.path.exists('./camerasettings/'):
        os.makedirs('./camerasettings/')
    with open('./camerasettings/net1.json','w') as f:
        json.dump(nc,f)
    with open('./camerasettings/net2.json','w') as f:
        json.dump(dh,f)
    with open('./camerasettings/cam.json','w') as f:
        json.dump(cs,f)
    with open('./camerasettings/vid.json','w') as f:
        json.dump(vs,f)
    print('Settings saved to ./camerasettings/')


def getHostname(cam):
    resp = cam.devicemgmt.GetHostname()
    print('getHostname:\n' + str(resp))

def getDeviceInformation(cam):
    resp = cam.devicemgmt.GetDeviceInformation()
    print('getDeviceInformation:\n' + str(resp))

def systemReboot(cam):
    resp = cam.devicemgmt.SystemReboot()
    print('systemReboot: ' + str(resp))



def onvifCommand(config, cmd):
    """ Execute ONVIF command to the IP camera.

    Arguments:
        config: [COnfiguration]
        cmd: [str] Command:
            - reboot
            - GetHostname
            - GetDeviceInformation
    """

    cam = None

    # extract IP from config file
    camera_ip = re.findall(r"[0-9]+(?:\.[0-9]+){3}", config.deviceID)[0]
    camera_onvif_port = 8899

    try:
        print('Connecting to {}:{}'.format(camera_ip, camera_onvif_port))
        cam = ONVIFCamera(camera_ip, camera_onvif_port, 'admin', '', '/home/pi/vRMS/wsdl')
    except:
        print('Could not connect to camera!')
        exit(1)

    print('Connected.')

    # process commands
    if cmd == 'reboot':
        systemReboot(cam)

    if cmd == 'GetHostname':
        getHostname(cam)

    if cmd == 'GetDeviceInformation':
        getDeviceInformation(cam)

    exit(0)





if __name__ == '__main__':

    # list of supported commands
    cmd_list = ['reboot', 'GetHostname', 'GetDeviceInformation']

    # command line argument parser
    parser = argparse.ArgumentParser(description='Controls ONVIF-Compatible IP camera')
    parser.add_argument('command', metavar='command', type=str, nargs=1, help=' | '.join(cmd_list))
    args = parser.parse_args()
    cmd = args.command[0]

    if not cmd in cmd_list:
        print('Error: command "{}" not supported'.format(cmd))
        exit(1)

    config = cr.parse('.config')

    if str(config.deviceID).isdigit():
        print('Error: this utility only works with IP cameras')
        exit(1)
    # extract IP from config file
    camera_ip = re.findall(r"[0-9]+(?:\.[0-9]+){3}", config.deviceID)[0]

    CameraControl(camera_ip, cmd, opts)    

"""Known Field mappings
These are available in Guides/imx2910config-maps.md

To set these values pass split at the dot if there is one, then call SetParam 
eg 
  SetParam ExposureParam Level 0
  SetParam DayNightColor 0
Decimals will be converted to hex strings if necesssary. 
"""
