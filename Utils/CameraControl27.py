""" Controls ONVIF-compatible IP camera """

# Prerequisites:
# Python2 
#   pip install onvif
# python3
#   pip install onvif_zeep
# note that if you install onvif_zeep on python2 it will not work

# Usage examples:
#
# python -m Utils.CameraControl27 reboot
# python -m Utils.CameraControl27 getDeviceInformation

from __future__ import print_function

import argparse
import re
import RMS.ConfigReader as cr
from onvif import ONVIFCamera
import os
import platform


def getHostname(cam):
    """get the hostname - seems pointless tbh 

    Args:
        cam : The camera
    """
    resp = cam.devicemgmt.GetHostname()
    print('getHostname:\n' + str(resp))


def getDeviceInformation(cam):
    """get limited device information

    Args:
        cam : The camera
    """
    resp = cam.devicemgmt.GetDeviceInformation()
    print('getDeviceInformation:\n' + str(resp))


def systemReboot(cam):
    """Reboot the Camera

    Args:
        cam : The camera
    """
    resp = cam.devicemgmt.SystemReboot()
    print('systemReboot: ' + str(resp))


# function to find where your WSDL files are. 
def getOnvifWsdlLocation():
    """Locate the ONVIF WSDL files required to use this module

    """

    platf = platform.system()
    if platf == 'Linux':
        basedir=os.path.join(os.environ['HOME'],'vRMS/lib')

    elif platf == 'Windows':
        # assumes you're using Anaconda
        try:
            condadir=os.environ['CONDA_PREFIX']
        except:
            print('I don\'t know how to find the WSDL files on your platform')
            exit(1)
        basedir=os.path.join(condadir, 'lib\\site-packages')

    elif platf == 'Darwin':
        # MacOS - should be the same as Linux i hope
        basedir=os.path.join(os.environ['HOME'],'vRMS/lib/')

    else:
        print('unknown platform {:s} - assuming linux-like' % platform)
        basedir=os.path.join(os.environ['HOME'],'vRMS/lib')

    wsdl_loc=''   
    for root, dirs, _ in os.walk(basedir, topdown=False):
        for name in dirs:
            loc = os.path.join(root,name)
            # on the Pi, the correct version is in an onvif-x.xx.x folder
            if name =='wsdl' and 'onvif' in root:
                return loc
            if name == 'wsdl' and platf == 'Windows' and 'zeep' not in root:
                return loc
    if wsdl_loc == '':
        print('Unable to find WSDL files, unable to continue')
        exit(1)
    return wsdl_loc


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
    
    wsdl_loc = getOnvifWsdlLocation()

    # extract IP from config file
    camera_ip = re.findall(r"[0-9]+(?:\.[0-9]+){3}", config.deviceID)[0]
    camera_onvif_port = 8899

    try:
        print('Connecting to {}:{}'.format(camera_ip, camera_onvif_port))
        print('WSDL location is', wsdl_loc)
        cam = ONVIFCamera(camera_ip, camera_onvif_port, 'admin', '', wsdl_loc)
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


if __name__ == '__main__':

    # list of supported commands
    cmd_list = ['reboot', 'GetHostname', 'GetDeviceInformation']

    # command line argument parser
    parser = argparse.ArgumentParser(description='Controls ONVIF-Compatible IP camera')
    parser.add_argument('command', metavar='command', type=str, nargs=1, help=' | '.join(cmd_list))
    args = parser.parse_args()
    cmd = args.command[0]

    if cmd not in cmd_list:
        print('Error: command "{}" not supported'.format(cmd))
        exit(1)

    config = cr.parse('.config')

    if str(config.deviceID).isdigit():
        print('Error: this utility only works with IP cameras')
        exit(1)
    
    
    # Process the IP camera control command
    onvifCommand(config, cmd)
