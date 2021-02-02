""" Controls CMS-compatible IP camera 
    This module can read and control the IMX291 Cameras and possibly other IMX cameras
    that comply with the dvrip protocol and can be controlled from CMS.

    Note: if you're using Python 2 then only the GetHostname and reboot parameters are 
    supported. This is a limitation of the camera control library

    usage 1: 
    Python -m Utils.CameraControl command {opts}
    call with -h to get a list of supported commands

    Usage 2:
    >>> import Utils.CameraControl as cc
    >>> cc.CameraControl(ip_address,command, [opts]) 
    >>> cc.CameraControlV2(config, command, [opts])

    Parameters:
    ipaddress: string ip address in dotted form eg 1.2.3.4
    config: RMS config object

    command: the command you want to execute. 
    opts: field and value to use when calling SetParam

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

    Network parameters:
    ==================
    You can retrieve the current network settings using GetNetConfig

    You can also set the IP Address, netmask and gateway using SetParam
        eg SetParam Network HostIP 1.2.3.4
    You can turn DHCP on and off using SetParam Network EnableDHCP 1 or 0

    Note that turning on DHCP will cause the camera to lose connection
    and you will need to scan your network or check your router to find out
    what its address has changed to. 

"""

import sys
import os

import ipaddress as ip
import binascii
import socket
import argparse
import json
import pprint
import re
import RMS.ConfigReader as cr
from time import sleep

# if not present, force update of the submodule

if sys.version_info.major > 2:
    import git
    import importlib  #used to import python-dvr as it has a dash in the name
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
else:
    # Python2 compatible version with much restricted capabilities
    import Utils.CameraControl27 as cc27


def rebootCamera(cam):
    """Reboot the Camera

    Args:
        cam : The camera
    """
    print('rebooting, please wait....')
    cam.reboot()
    retry = 0
    while retry < 5:
        sleep(5) # wait while camera starts
        if cam.login():
            break
        retry += 1
    if retry < 5: 
        print('reboot successful')
    else:
        print('camera nonresponsive, please wait 30s and reconnect')


def strIPtoHex(ip):
    a = binascii.hexlify(socket.inet_aton(ip)).decode().upper()
    addr='0x'+''.join([a[x:x+2] for x in range(0,len(a),2)][::-1])
    return addr


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


def loadFromFile():
    """Load the camera config from JSON files saved earlier

    """
    if not os.path.exists('./camerasettings/'):
        print('Settings files not found in ./camerasettings/')
        return 
    print('Loading settings....')
    with open('./camerasettings/net1.json','r') as f:
        nc = json.load(f)
    with open('./camerasettings/net2.json','r') as f:
        dh = json.load(f)
    with open('./camerasettings/cam.json','r') as f:
        cs = json.load(f)
    with open('./camerasettings/vid.json','r') as f:
        vs = json.load(f)
    print('Loaded')
    return nc, dh, cs, vs 


def getNetworkParams(cam, showit=True):
    """ retrieve or display the camera network settings

    Args:
        cam : canera object
        showit (bool, optional): whether to show the info or just return it. Defaults to True 

    Returns:
        json block containing the config
    """
    nc=cam.get_info("NetWork.NetCommon")
    dh=cam.get_info("NetWork.NetDHCP")

    if showit is True:
        print('IP Address  : ', iptoString(nc['HostIP']))
        print('---------')
        pprint.pprint(nc)
        print('---------')
        pprint.pprint(dh)
    return nc, dh


def getEncodeParams(cam, showit=True):
    """ Read the Encode section of the camera config

    Args:
        cam - the camera 
        saveit (bool, optional): whether we're being called so the data can be saved. Defaults to True.

    Returns:
        json block containing the config
    """
    vidinfo = cam.get_info("Simplify.Encode")
    if showit is True:
        pprint.pprint(vidinfo)
    return vidinfo


def getCameraParams(cam, showit=True):
    """ display or retrueve the Camera section of the camera config

    Args:
        cam - the camera 
        saveit (bool, optional): whether we're being called so the data can be saved. Defaults to True.

    Returns:
        json block containing the config
    """
    caminfo = cam.get_info("Camera")
    p1 = caminfo['Param'][0]
    p2 = caminfo['ParamEx'][0]
    if showit is True:
        pprint.pprint(p1)
        pprint.pprint(p2)
    return caminfo


def setEncodeParam(cam, opts):
    """ Set a parameter in the Encode section of the camera config
        Note: a different approach has to be taken here than for camera params
        as its not possible to set individual parameters without crashing the camera
    Args:
        cam - the camera 
        opts - array of fields, subfields and the value to set
    """
    intflds=['FPS','BitRate','GOP','Quality']

    params = cam.get_info("Simplify.Encode")
    fld=opts[1]
    if fld == 'Video':
        subfld=opts[2]
        val = opts[3]
        if subfld=='Compression' and val !='H.264' and val != 'H.265':
            print('Compression must be H.264 or H.265')
            return 
        if subfld=='Resolution' and val !='720P' and val != '1080P' and val !='3M':
            print('Resolution must be 720P, 1080P or 3M')
            return 
        if subfld=='BitRateControl' and val !='CBR' and val !='VBR':
            print('BitRateControl must be VBR or CBR')
            return
        if subfld in intflds:
            val = int(val)
        params[0]['MainFormat']['Video'][subfld] = val
    else:
        val = opts[2]
        if val!=0 and val!=1:
            print('AudioEnable and VideoEnable must be 1 or 0')
            return 
        subfld=''
        params[0]['MainFormat'][fld]=val
    cam.set_info("Simplify.Encode", params)
    print('Set {} {} to {}'.format(fld, subfld, val))
    print('Camera Reboot required to take effect')


def setNetworkParam(cam, opts):
    """ Set a parameter in the Network section of the camera config

    Args:
        cam - the camera 
        opts - array of fields, subfields and the value to set
    """
    # top level field name
    fld=opts[1]
    # these fields are stored in the ParamEx.[0] block
    if fld == 'HostIP':
        val = opts[2]
        hexval = strIPtoHex(val)
        cam.set_info("NetWork.NetCommon.HostIP", hexval)

    elif fld == 'GateWay':
        val = opts[2]
        hexval = strIPtoHex(val)
        cam.set_info("NetWork.NetCommon.GateWay", hexval)

    elif fld == 'Submask':
        val = opts[2]
        hexval = strIPtoHex(val)
        cam.set_info("NetWork.NetCommon.Submask", hexval)

    elif fld == 'EnableDHCP':
        val = int(opts[2])
        if val == 1:
            cam.set_info("NetWork.NetDHCP.[0].Enable", 1)
        else:
            cam.set_info("NetWork.NetDHCP.[0].Enable", 0)
        dh = cam.get_info("NetWork.NetDHCP.[0]")
        print(dh)

    else:
        print('Options for SetParam Network are: HostIP, GateWay, Submask')
        print('followed by an ipaddress in xxx.xxx.xxx.xxx format')
        print('or EnableDHCP followed by 1 or 0')



def setCameraParam(cam, opts):
    """ Set a parameter in the Camera section of the camera config
        Individual parameters can be set and the change will take effect immediately 
    Args:
        cam - the camera 
        opts - array of fields, subfields and the value to set
    """
    # these fields are stored as integers. Others are Hex strings
    intfields=['AeSensitivity','Day_nfLevel','DncThr','ElecLevel',
        'IRCUTMode','IrcutSwap','Night_nfLevel', 'Level','AutoGain','Gain']
    styleFlds='typedefault','type1','type2'
    # top level field name
    fld=opts[1]
    # these fields are stored in the ParamEx.[0] block
    if fld == 'Style':
        val = opts[2]
        if val not in styleFlds:
            print('style must be one of ', styleFlds)
            return
        cam.set_info("Camera.ParamEx.[0]",{fld:val})
        print('Set Camera.ParamEx.[0].{} to {}'.format(fld, val))
    elif fld == 'BroadTrends': 
        subfld=opts[2]
        val = int(opts[3])
        fldToSet='Camera.ParamEx.[0].'+fld
        cam.set_info(fldToSet,{subfld:val})
        print('Set {} to {}'.format(fldToSet, val))
                
    # Exposuretime and gainparam have subfields
    elif fld == 'ExposureParam' or fld == 'GainParam':
        subfld=opts[2]
        val = int(opts[3])
        if subfld not in intfields:
            # the two non-int fields in ExposureParam are the exposure times. 
            # These are stored in microsconds converted ito hex strings.
            if val < 100 or val > 80000: 
                print('Exposure must be between 100 and 80000 microsecs')
                return
            val ="0x%8.8X" % (int(val))
        fldToSet='Camera.Param.[0].'+fld
        cam.set_info(fldToSet,{subfld:val})
        print('Set {} to {}'.format(fldToSet, val))
    else:
        # other fields do not have subfields
        val = int(opts[2])
        if fld not in intfields:
            val ="0x%8.8X" % val
        cam.set_info("Camera.Param.[0]",{fld:val})
        print('Set Camera.Param.[0].{} to {}'.format(fld, val))


def setParameter(cam, opts):
    """ Set a parameter in various sections of the camera config

    Args:
        cam - the camera 
        opts - array of fields, subfields and the value to set
    """
    if len(opts) < 3:
        print('Not enough parameters, need at least block, field, value')
    if opts[0] == 'Camera':
        setCameraParam(cam, opts)
    elif opts[0] == 'Encode':
        setEncodeParam(cam, opts)
    elif opts[0] == 'Network':
        setNetworkParam(cam, opts)
    else:
        print('Setting not currently supported for', opts)


def dvripCall(cam, cmd, opts):
    """ retrieve or display the camera network settings

    Args:
        cam  - the camera
        cmd  - the command to execute
        opts - optional list of parameters to be passed to SetParam
    """
    if cmd == 'GetHostname':
        nc,_=getNetworkParams(cam, False)
        print(nc['HostName'])

    elif cmd == 'GetNetConfig':
        getNetworkParams(cam, True)

    elif cmd == 'reboot':
        rebootCamera(cam)

    elif cmd == 'GetCameraParams':
        getCameraParams(cam, True)
    
    elif cmd == 'GetEncodeParams':
        getEncodeParams(cam, True)

    elif cmd == 'GetSettings':
        getNetworkParams(cam, True)
        getCameraParams(cam, True)
        getEncodeParams(cam, True)

    elif cmd =='SaveSettings':
        nc, dh = getNetworkParams(cam, False)
        cs = getCameraParams(cam, False)
        vs = getEncodeParams(cam, False)
        saveToFile(nc, dh, cs, vs)

    elif cmd == 'LoadSettings':
        nc, dh, cs, vs = loadFromFile()
        cam.set_info("NetWork.NetCommon", nc)
        cam.set_info("NetWork.NetDHCP", dh)
        cam.set_info("Camera",cs)
        cam.set_info("Simplify.Encode", vs)
        rebootCamera(cam)

    elif cmd == 'SetParam':
        setParameter(cam, opts)

    else:
        print('System Info')
        ugi=cam.get_upgrade_info()
        print(ugi['Hardware'])


def cameraControl(camera_ip, cmd, opts=''):
    """CameraControl - main entry point to the module

    Args:
        camera_ip (string): IPAddress of camera in dotted form eg 192.168.1.10
        cmd (string): Command to be executed
        opts (array of strings): Optional array of field, subfield and value for the SetParam command
    """
    # Process the IP camera control command
    cam = dvr.DVRIPCam(camera_ip, "admin", "")
    if cam.login():
        try:
            dvripCall(cam, cmd, opts)
        except:
            print('error executing command - probably not supported')
    else:
        print("Failure. Could not connect.")
    cam.close()


def cameraControlV2(config, cmd, opts=''):
    if str(config.deviceID).isdigit():
        print('Error: this utility only works with IP cameras')
        exit(1)
    # extract IP from config file
    camera_ip = re.findall(r"[0-9]+(?:\.[0-9]+){3}", config.deviceID)[0]

    if sys.version_info.major < 3:
        cc27.onvifCommand(config, cmd)
    else:
        cameraControl(camera_ip, cmd, opts)


if __name__ == '__main__':
    """Main function
    Args:
        command - the command you want to execute
        opts - optional list of fields and a value to pass to SetParam
    """

    # list of supported commands
    if sys.version_info.major < 3:
        cmd_list = ['reboot', 'GetHostname', 'GetDeviceInformation']
        opthelp=''
    else:
        cmd_list = ['reboot', 'GetHostname', 'GetSettings','GetDeviceInformation','GetNetConfig',
            'GetCameraParams','GetEncodeParams','SetParam','SaveSettings', 'LoadSettings']
        opthelp='optional parameters for SetParam for example Camera ElecLevel 70 \n' \
            'will set the AE Ref to 70.\n To see possibilities, execute GetSettings first'

    usage = "Available commands " + str(cmd_list) + '\n' + opthelp
    parser = argparse.ArgumentParser(description='Controls CMS-Compatible IP camera',
        usage=usage)
    parser.add_argument('command', metavar='command', type=str, nargs=1, help=' | '.join(cmd_list))
    parser.add_argument('options', metavar='opts', type=str, nargs='*', help=opthelp)
    args = parser.parse_args()
    cmd = args.command[0]
    if args.options is not None:
        opts = args.options
    else:
        opts=''
        
    if cmd not in cmd_list:
        print('Error: command "{}" not supported'.format(cmd))
        exit(1)

    config = cr.parse('.config')

    cameraControlV2(config, cmd, opts)
    

"""Known Field mappings
These are available in Guides/imx2910config-maps.md

To set these values pass split at the dot if there is one, then call SetParam 
eg 
  SetParam ExposureParam Level 0
  SetParam DayNightColor 0
Decimals will be converted to hex strings if necesssary. 
"""
