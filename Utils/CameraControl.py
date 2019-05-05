""" Controls ONVIF-compatible IP camera """

# Usage examples:
#
# pip install client
# pip install onvif
# python -m RMS.CameraControl reboot
# python -m RMS.CameraControl GetDeviceInformation

from __future__ import print_function

import argparse
import re
import RMS.ConfigReader as cr
from onvif import ONVIFCamera

cam = None

def GetHostname():
    resp = cam.devicemgmt.GetHostname()
    print('GetHostname:\n' + str(resp))

def GetDeviceInformation():
    resp = cam.devicemgmt.GetDeviceInformation()
    print('GetDeviceInformation:\n' + str(resp))

def SystemReboot():
    resp = cam.devicemgmt.SystemReboot()
    print('SystemReboot: ' + str(resp))

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

    if config.deviceID.isdigit():
        print('Error: this utility only works with IP cameras')
        exit(1)
    
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
        SystemReboot()

    if cmd == 'GetHostname':
	GetHostname()

    if cmd == 'GetDeviceInformation':
	GetDeviceInformation()

    exit(0)

