# noqa:E501,W291
""" Controls CMS-compatible IP camera 
    This module can read and control the IMX291 Cameras and possibly other IMX cameras
    that comply with the dvrip protocol and can be controlled from CMS.

    Note: if you're using Python 2 then only the GetHostname and reboot parameters are 
    supported. This is a limitation of the camera control library

    usage 1: 
    python -m Utils.CameraControl command {opts}
    call with -h to get a list of supported commands

    Usage 2:
    >>> import Utils.CameraControl as cc
    >>> cc.CameraControl(ip_address,command, [opts]) 
    >>> cc.CameraControlV2(config, command, [opts])

    Parameters:
    ip_address: dotted ipaddress of the camera eg 1.2.3.4
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

    API details : https://oppf.xmcsrv.com/#/api?md=readProtocol

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
from RMS.Logger import getLogger, initLogging
from time import sleep
import datetime

# if not present, force update of the submodule

if sys.version_info.major > 2:
    import dvrip as dvr
else:
    # Python2 compatible version 
    import Utils.CameraControl27 as dvr

# Get the logger from the main module
log = getLogger("logger")


def rebootCamera(cam):
    """Reboot the Camera

    Args:
        cam : The camera
    """
    log.info('rebooting, please wait....')
    cam.reboot()
    retry = 0
    while retry < 5:
        sleep(5) # wait while camera starts
        if cam.login():
            break
        retry += 1
    if retry < 5: 
        log.info('reboot successful')
    else:
        log.info('camera nonresponsive, please wait 30s and reconnect')


def strIPtoHex(ip_str):
    a = binascii.hexlify(socket.inet_aton(ip_str)).decode().upper()
    addr = '0x' + ''.join([a[x:x+2] for x in range(0, len(a), 2)][::-1])
    return addr


def iptoString(s):
    """Convert an IP address in hex network order to a human readable string 

    Args:
        s (string): the encoded IP address eg '0x0A01A8C0' 

    Returns:
        string: human readable IP in host order eg '192.169.1.10'
    """
    a = s[2:]
    addr = '0x' + ''.join([a[x:x+2] for x in range(0, len(a), 2)][::-1])
    ipaddr = ip.IPv4Address(int(addr, 16))
    return ipaddr


def saveToFile(nc, dh, nt, cs, vs, gu, cp, rb, lc):
    """Save the camera config to pretty-printed JSON files
    Args:
    nc : NetWork NetCommon config
    dh : NetWork DHCP config
    nt : NetWork NTP config
    cs : camera config
    vs : video encoding config
    gu : gui display params
    cp : color settings dialog
    rb : autoreboot params
    lc : location params
    """
    if not os.path.exists('./camerasettings/'):
        os.makedirs('./camerasettings/')

    configs = {
        'netcommon.json': nc,
        'netdhcp.json': dh,
        'netntp.json': nt,
        'cam.json': cs,
        'vid.json': vs,
        'gui.json': gu,
        'color.json': cp,
        'autoreboot.json': rb,
        'location.json': lc
    }

    for filename, config in configs.items():
        with open(os.path.join('./camerasettings/', filename), 'w') as f:
            json.dump(config, f, indent=4, sort_keys=True)

    log.info('Settings saved to ./camerasettings/')


def loadFromFile():
    """Load the camera config from JSON files saved earlier"""
    if not os.path.exists('./camerasettings/'):
        log.info('Settings files not found in ./camerasettings/')
        return None

    log.info('Loading settings....')

    config_files = {
        'netcommon.json': 'nc',
        'netdhcp.json': 'dh',
        'netntp.json': 'nt',
        'cam.json': 'cs',
        'vid.json': 'vs',
        'gui.json': 'gu',
        'color.json': 'cp',
        'autoreboot.json': 'rb',
        'location.json': 'lc'
    }

    configs = {}

    for filename, config_name in config_files.items():
        file_path = os.path.join('./camerasettings/', filename)
        if not os.path.exists(file_path):
            log.info("Warning: {} not found. Skipping.".format(filename))
            configs[config_name] = None
            continue

        with open(file_path, 'r') as f:
            configs[config_name] = json.load(f)

    log.info('Loaded')
    return (configs['nc'], configs['dh'], configs['nt'], configs['cs'], configs['vs'],
            configs['gu'], configs['cp'], configs['rb'], configs['lc'])


def getNetworkParams(cam, showit=True):
    """ retrieve or display the camera network settings

    Args:
        cam : camera object
        showit (bool, optional): whether to log out the settings.

    Returns:
        json block containing the config
    """
    nc = cam.get_info("NetWork.NetCommon")
    dh = cam.get_info("NetWork.NetDHCP")
    nt = cam.get_info("NetWork.NetNTP")

    if showit is True:
        log.info('IP Address  : ' + str(iptoString(nc['HostIP'])))
        log.info('---------')
        log.info(pprint.pformat(nc))
        log.info('---------')
        log.info(pprint.pformat(dh))
        log.info('---------')
        log.info(pprint.pformat(nt))
    return nc, dh, nt


def getIP(cam):
    nc = cam.get_info("NetWork.NetCommon")
    log.info(str(iptoString(nc['HostIP'])))
    return


def getEncodeParams(cam, showit=True):
    """ Read the Encode section of the camera config

    Args:
        cam - the camera 
        showit (bool, optional): whether to log out the settings.

    Returns:
        json block containing the config
    """
    vidinfo = cam.get_info("Simplify.Encode")
    if showit is True:
        log.info(pprint.pformat(vidinfo))
    return vidinfo


def getCameraParams(cam, showit=True):
    """ display or retrieve the Camera section of the camera config

    Args:
        cam - the camera 
        showit (bool, optional): whether to log out the settings.

    Returns:
        json block containing the config
    """
    caminfo = cam.get_info("Camera")
    fog = caminfo["ClearFog"][0]
    p1 = caminfo['Param'][0]
    p2 = caminfo['ParamEx'][0]
    if showit is True:
        log.info(pprint.pformat(fog))
        log.info(pprint.pformat(p1))
        log.info(pprint.pformat(p2))
    return caminfo


def getGuiParams(cam, showit=True):
    """ display or retrieve the Gui Params 

    Args:
        cam - the camera 
        showit (bool, optional): whether to log out the settings.

    Returns:
        json block containing the config
    """
    caminfo = cam.get_info("AVEnc.VideoWidget")
    if showit is True:
        log.info(pprint.pformat(caminfo))
    return caminfo


def getGeneralParams(cam, showit=True):
    """ display or retrieve the Autoreboot and Location Params

    Args:
        cam - the camera
        showit (bool, optional): whether to log out the settings.

    Returns:
        json block containing the config
    """
    rb = cam.get_info("General.AutoMaintain")
    lc = cam.get_info("General.Location")

    if showit is True:
        log.info(pprint.pformat(rb))
        log.info('---------')
        log.info(pprint.pformat(lc))

    return rb, lc


def getColorParams(cam, showit=True):
    """ display or retrieve the Color Settings section of the camera config

    Args:
        cam - the camera 
        showit (bool, optional): whether to log out the settings.

    Returns:
        json block containing the config
    """
    caminfo = cam.get_info("AVEnc.VideoColor.[0]")
    if showit is True:
        log.info(pprint.pformat(caminfo))
    return caminfo


def setEncodeParam(cam, opts):
    """ Set a parameter in the Encode section of the camera config
        Note: a different approach is used than for camera params
        as it's not possible to set individual parameters without crashing the camera
    Args:
        cam - the camera 
        opts - array of fields, subfields and the value to set
    """
    intflds = ['FPS','BitRate','GOP','Quality']

    params = cam.get_info("Simplify.Encode")
    fld = opts[1]
    if fld == 'Video':
        subfld = opts[2]
        val = opts[3]
        if subfld == 'Compression' and val not in ('H.264', 'H.265'):
            log.info('Compression must be H.264 or H.265')
            return 
        if subfld == 'Resolution' and val not in ('720P', '1080P', '3M'):
            log.info('Resolution must be 720P, 1080P or 3M')
            return 
        if subfld == 'BitRateControl' and val not in ('CBR', 'VBR'):
            log.info('BitRateControl must be VBR or CBR')
            return
        if subfld in intflds:
            val = int(val)
        params[0]['MainFormat']['Video'][subfld] = val

    elif fld == 'SecondStream':
        val = int(opts[2])
        if val not in (0, 1):
            log.info('SecondStream must be 1 or 0')
            return 
        subfld = ''
        params[0]['ExtraFormat']['VideoEnable'] = val
        params[0]['ExtraFormat']['AudioEnable'] = val
    else:
        val = int(opts[2])
        subfld = ''
        params[0]['MainFormat'][fld] = val

    cam.set_info("Simplify.Encode", params)
    log.info('Set {} {} to {}'.format(fld, subfld, val))


def setNetworkParam(cam, opts):
    """ Set a parameter in the Network section of the camera config

    Args:
        cam - the camera 
        opts - array of fields, subfields and the value to set
    """
    # top level field name
    fld = opts[1]
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
            log.info('DHCP enabled')
        else:
            cam.set_info("NetWork.NetDHCP.[0].Enable", 0)
            log.info('DHCP disabled')

    elif fld == 'setTimezone':
        val = opts[2]
        cam.set_info("NetWork.NetNTP.TimeZone", val)

    elif fld == 'EnableNTP':
        val = opts[2]
        if val == "0":
            cam.set_info("NetWork.NetNTP.Enable", False)
            log.info('NTP disabled')
        else:
            cam.set_info("NetWork.NetNTP.Server.Name", val)
            cam.set_info("NetWork.NetNTP.Enable", True)
            cam.set_info("NetWork.NetNTP.UpdatePeriod", 60)
            log.info('NTP enabled')
    elif fld == 'TransferPlan':
        val = opts[2]
        cam.set_info("NetWork.NetCommon.TransferPlan", val)

    else:
        log.info('usage: SetParam Network option,value: ')
        log.info('HostIP, GateWay or Submask followed by a dotted IP address')
        log.info('EnableDHCP followed by 1 or 0')
        log.info('EnableNTP followed by a dotted IP address to enable or 0 to disable')


def setVideoFormatParam(cam, opts):
    """ Set a parameter in the VideoFormat section of the camera config

    Args:
        cam - the camera 
        opts - array of fields, subfields and the value to set
    """

    fld = opts[1]
    if fld == 'VideoFormat':
        val = opts[2]
        if val not in ('PAL', 'NTSC'):
            log.info('VideoFormat must be PAL or NTSC')
            return
        cam.set_info("General.Location.VideoFormat", val)
        log.info("Video Format set to {}".format(val))

    else:
        log.info('usage: SetParam General VideoFormat PAL')


def setCameraParam(cam, opts):
    """ Set a parameter in the Camera section of the camera config
        Individual parameters can be set and the change will take effect immediately 
    Args:
        cam - the camera 
        opts - array of fields, subfields and the value to set
    """
    # these fields are stored as integers. Others are Hex strings
    intfields = [
        'AeSensitivity','Day_nfLevel','DncThr','ElecLevel','IRCUTMode',
        'IrcutSwap','Night_nfLevel','Level','AutoGain','Gain'
    ]
    styleFlds = ('typedefault','type1','type2')

    fld = opts[1]
    if fld == 'ClearFog':
        subfld = opts[2].lower()
        val = int(opts[3])
        if subfld == 'enable':
            val = True if val == 1 else False

        elif subfld == 'level':
            val = int(val)

        else:
            log.info('Invalid ClearFog subfield. Use "enable" or "level".')
            return

        log.info('Set Camera.ClearFog.[0].{} to {}'.format(subfld, val))
        cam.set_info("Camera.ClearFog.[0]", {subfld: val})

    # these fields are stored in the ParamEx.[0] block
    elif fld == 'Style':
        val = opts[2]
        if val not in styleFlds:
            log.info('style must be one of {}'.format(styleFlds))
            return
        log.info('Set Camera.ParamEx.[0].{} to {}'.format(fld, val))
        cam.set_info("Camera.ParamEx.[0]", {fld: val})

    elif fld == 'BroadTrends':
        subfld = opts[2]
        val = int(opts[3])
        if subfld in ('AutoGain', 'Gain'):
            fldToSet = 'Camera.ParamEx.[0].' + fld
            log.info('Set {}.{} to {}'.format(fldToSet, subfld, val))
            cam.set_info(fldToSet, {subfld: val})
        else:
            log.info("BroadTrends option must be 'AutoGain' or 'Gain'")

    # Exposuretime and gainparam have subfields
    elif fld in ('ExposureParam', 'GainParam'):

        subfld = opts[2]
        val = int(opts[3])
        if subfld not in intfields:
            # the two non-int fields in ExposureParam are the exposure times. 
            # These are stored in microseconds converted to hex strings.
            if val < 100 or val > 80000: 
                log.info('Exposure must be between 100 and 80000 microsecs')
                return
            val = "0x%8.8X" % (int(val))
        fldToSet = 'Camera.Param.[0].' + fld
        log.info('Set {}.{} to {}'.format(fldToSet, subfld, val))
        cam.set_info(fldToSet, {subfld: val})

    else:
        # other fields do not have subfields
        val = int(opts[2])
        if fld not in intfields:
            val = "0x%8.8X" % val
        log.info('Set Camera.Param.[0].{} to {}'.format(fld, val))
        cam.set_info("Camera.Param.[0]", {fld: val})


def setOSD(cam, opts):
    """ Set a parameter in the Gui Params section of the camera config
    Args:
        cam - the camera 
        opts - array of fields, subfields and the value to set
    """

    info = cam.get_info("AVEnc.VideoWidget")
    if len(opts) == 0:
        log.info('usage: setOSD on|off')
        return

    if opts[0] == 'on':
        info[0]["TimeTitleAttribute"]["EncodeBlend"] = True
        info[0]["ChannelTitleAttribute"]["EncodeBlend"] = True
        log.info('Set osd enabled')
    else:
        info[0]["TimeTitleAttribute"]["EncodeBlend"] = False 
        info[0]["ChannelTitleAttribute"]["EncodeBlend"] = False 
        log.info('Set osd disabled')

    cam.set_info("AVEnc.VideoWidget", info)


def setColor(cam, opts):
    """ Set a parameter in the Color Settings 
    Args:
        cam - the camera 
        opts - array of fields, subfields and the value to set
    """

    info = cam.get_info("AVEnc.VideoColor.[0]")
    b,c,s,h,g,a = 100,50,0,50,0,0

    if len(opts) > 0:
        spls = opts[0].split(',')
        try:
            b = int(spls[0])
            c = int(spls[1])
            s = int(spls[2])
            h = int(spls[3])
            g = int(spls[4])
            a = int(spls[5])
        except Exception:
            pass
    else:
        log.info('usage: setColor brightness,contrast,saturation,hue,gain,acutance')
        log.info('  b,c,s,h,g all numbers from 1 to 100')
        log.info('  acutance sets both horiz and vert sharpness')
        log.info('  the lower 8 bits set horiz and the upper 8 bits set vert')
        return

    n = 0
    info[n]["VideoColorParam"]["Brightness"] = b
    info[n]["VideoColorParam"]["Contrast"] = c
    info[n]["VideoColorParam"]["Saturation"] = s
    info[n]["VideoColorParam"]["Hue"] = h
    info[n]["VideoColorParam"]["Gain"] = g
    info[n]["VideoColorParam"]["Acutance"] = a
    # print(json.dumps(info[n], ensure_ascii=False, indent=4, sort_keys=True))
    log.info('Set color configuration %s %s %s %s %s %s', b, c, s, h, g, a)
    cam.set_info("AVEnc.VideoColor.[0]", info)


def setAutoReboot(cam, opts):
    """ Set a parameter in the Color Settings 
    Args:
        cam - the camera 
        opts - array of fields, subfields and the value to set
    """

    info = cam.get_info("General.AutoMaintain") 
    # print(json.dumps(info, ensure_ascii=False, indent=4, sort_keys=True))
    if len(opts) < 1:
        log.info('usage: setAutoReboot dayofweek,hour')
        log.info('  where dayofweek is Never EveryDay Monday Tuesday etc')
        log.info('  and hour is a number between 0 and 23')
        return
    spls = opts[0].split(',')
    day = spls[0]
    hour = 0
    if len(spls) > 1:
        hour = int(spls[1])
    valid_days = [
        'Everyday','Monday','Tuesday','Wednesday','Thursday','Friday',
        'Saturday','Sunday','Never'
    ]
    if day not in valid_days or hour < 0 or hour > 23:
        log.info('usage: SetAutoReboot dayofweek,hour')
        log.info('  where dayofweek is Never, Everyday, Monday, Tuesday, Wednesday etc')
        log.info('  and hour is a number between 0 and 23')
        return

    info["AutoRebootDay"] = day
    info["AutoRebootHour"] = hour
    log.info('Set autoreboot: %s at %s', day, hour*100)
    cam.set_info("General.AutoMaintain", info)


def manageCloudConnection(cam, opts):
    if len(opts) < 1 or opts[0] not in ['on', 'off', 'get']:
        log.info('usage: CloudConnection on|off|get')
        return

    info = cam.get_info("NetWork.Nat") 
    if opts[0] == 'get':
        log.info('Enabled %s', info['NatEnable'])
        return 
    if opts[0] == 'on':
        info["NatEnable"] = True
    else:
        info["NatEnable"] = False
    cam.set_info("NetWork.Nat", info)
    info = cam.get_info("NetWork.Nat")
    log.info('Enabled %s', info['NatEnable'])


def setParameter(cam, opts):
    """ Set a parameter in various sections of the camera config

    Args:
        cam - the camera
        opts - array of fields, subfields and the value to set
    """
    if len(opts) < 3:
        log.info('Not enough parameters, need at least block, field, value')
    if opts[0] == 'Camera':
        setCameraParam(cam, opts)
    elif opts[0] == 'Encode':
        setEncodeParam(cam, opts)
    elif opts[0] == 'Network':
        setNetworkParam(cam, opts)
    elif opts[0] == 'General':
        setVideoFormatParam(cam, opts)

    else:
        log.info('Setting not currently supported for %s', opts)


def switchMode(cam, mode_name, path='./camera_settings.json'):
    """
    Switch the camera to a named mode by executing the commands in the JSON file.
    Filters out any "SwitchMode" entries to avoid recursion.

    Args:
        cam: An authenticated camera object.
        mode_name (str): The name of the mode to switch to ("day", "night", etc.).
        path (str): Path to the JSON file containing mode definitions.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError("Camera settings file '{}' not found.".format(path))

    with open(path, 'r') as f:
        modes = json.load(f)

    if mode_name not in modes:
        raise ValueError("Mode '{}' not found in '{}'. Available modes: {}"
                         .format(mode_name, path, list(modes.keys())))

    # Loop over each command array in the specified mode
    for param in modes[mode_name]:
        cmd = param[0]
        opts = param[1:]

        # Avoid calling "SwitchMode" from within switchMode
        if cmd == "SwitchMode":
            log.warning("Ignoring SwitchMode command inside JSON to prevent recursion.")
            continue

        # Pass everything else directly to dvripCall
        dvripCall(cam, cmd, opts)


def dvripCall(cam, cmd, opts, camera_settings_path='./camera_settings.json'):
    """ retrieve or display the camera network settings

    Args:
        cam  - the camera
        cmd  - the command to execute
        opts - optional list of parameters to be passed to SetParam
    """
    if cmd == 'GetHostname':
        nt, _, _ = getNetworkParams(cam, False)
        log.info(nt['HostName'])
        return

    elif cmd == 'GetNetConfig':
        getNetworkParams(cam, True)
        return

    elif cmd == 'reboot':
        rebootCamera(cam)
        return

    elif cmd == 'GetIP':
        getIP(cam)
        return

    elif cmd == 'GetAutoReboot':
        getGeneralParams(cam, True)
        return

    elif cmd == 'CloudConnection':
        manageCloudConnection(cam, opts)
        return

    elif cmd == 'GetCameraParams':
        getCameraParams(cam, True)
        return

    elif cmd == 'GetEncodeParams':
        getEncodeParams(cam, True)
        return

    elif cmd == 'GetSettings':
        getNetworkParams(cam, True)
        getCameraParams(cam, True)
        getEncodeParams(cam, True)
        getGuiParams(cam, True)
        getColorParams(cam, True)
        getGeneralParams(cam, True)
        return

    elif cmd == 'SaveSettings':
        nc, dh, nt = getNetworkParams(cam, False)
        cs = getCameraParams(cam, False)
        vs = getEncodeParams(cam, False)
        gu = getGuiParams(cam, False)
        cp = getColorParams(cam, False)
        rb, lc = getGeneralParams(cam, False)
        saveToFile(nc, dh, nt, cs, vs, gu, cp, rb, lc)
        return

    elif cmd == 'LoadSettings':
        loaded = loadFromFile()
        if not loaded:
            return
        nc, dh, nt, cs, vs, gu, cp, rb, lc = loaded
        cam.set_info("NetWork.NetCommon", nc)
        cam.set_info("NetWork.NetDHCP", dh)
        cam.set_info("NetWork.NetNTP", nt)
        cam.set_info("Camera", cs)
        cam.set_info("Simplify.Encode", vs)
        cam.set_info("AVEnc.VideoWidget", gu)
        cam.set_info("AVEnc.VideoColor.[0]", cp)
        cam.set_info("General.AutoMaintain", rb)
        cam.set_info("General.Location", lc)
        rebootCamera(cam)
        return

    elif cmd == 'SetParam':
        setParameter(cam, opts)
        return

    elif cmd == 'CameraTime':
        if opts[0] == 'get':
            log.info(str(cam.get_time()))
        elif opts[0] == 'set':
            if cam.get_info("NetWork.NetNTP.Enable") is True:
                log.info('cant set the camera time - NTP enabled')
            else:
                try:
                    reqtime = datetime.datetime.strptime(opts[1], '%Y%m%d_%H%M%S')
                except:
                    reqtime = datetime.datetime.now()
                cam.set_time(reqtime)
                log.info('time set to %s', reqtime)
        else:
            log.info('usage CameraTime get|set')
        return

    elif cmd == 'SetColor':
        setColor(cam, opts)
        return

    elif cmd == 'SetOSD':
        setOSD(cam, opts)
        return

    elif cmd == 'SetAutoReboot':
        setAutoReboot(cam, opts)
        return

    elif cmd == 'SwitchMode':
        if not opts:
            log.error("No mode specified for SwitchMode.")
            return
        
        # If opts is just a string, use it directly; if it's a list, pull the first element
        if isinstance(opts, str):
            mode_name = opts
        else:
            mode_name = opts[0]
        
        switchMode(cam, mode_name, camera_settings_path)
        return
    
    # -- If we get here, command is not recognized:
    else:
        log.error("Unrecognized command '%s' in dvripCall. Options were: %s", cmd, opts)
        log.info('System Info')
        ugi = cam.get_upgrade_info()
        log.info(ugi['Hardware'])
        return


def cameraControl(camera_ip, cmd, opts='', camera_settings_path='./camera_settings.json'):
    """CameraControl - main entry point to the module

    Args:
        camera_ip (string): IPAddress of camera in dotted form eg 192.168.1.10
        cmd (string): Command to be executed
        opts (array of strings): Optional array of field, subfield and value for the SetParam command
    """
    # Process the IP camera control command
    cam = dvr.DVRIPCam(camera_ip)
    if cam.login():
        try:
            dvripCall(cam, cmd, opts, camera_settings_path)
        except Exception as e:
            log.error("Error executing command: %s", e)
            log.error("This command may not be supported.")
    else:
        log.info("Failure. Could not connect.")
    cam.close()


def cameraControlV2(config, cmd, opts=''):
    """High-level entry point that uses config to figure out IP and path."""

    if str(config.deviceID).isdigit():
        log.info('Error: this utility only works with IP cameras')
        exit(1)
    # extract IP from config file
    camera_ip = re.findall(r"[0-9]+(?:\.[0-9]+){3}", config.deviceID)[0]

    if not hasattr(config, 'camera_settings_path') or not os.path.isfile(config.camera_settings_path):
        camera_settings_path = './camera_settings.json'
    else:
        camera_settings_path = config.camera_settings_path

    cameraControl(camera_ip, cmd, opts, camera_settings_path=camera_settings_path)


if __name__ == '__main__':
    """Main function
    Args:
        command - the command you want to execute
        opts - optional list of fields and a value to pass to SetParam
    """

    # list of supported commands
    cmd_list = [
        'reboot', 'GetHostname', 'GetSettings','GetDeviceInformation','GetNetConfig',
        'GetCameraParams','GetEncodeParams','SetParam','SaveSettings','LoadSettings',
        'SetColor','SetOSD','SetAutoReboot','GetIP','GetAutoReboot','CloudConnection',
        'CameraTime','SwitchMode'
    ]
    opthelp = (
        'optional parameters for SetParam for example Camera ElecLevel 70 \n'
        'will set the AE Ref to 70.\n To see possibilities, execute GetSettings first. '
        'Call a function with no parameters to see the possibilities'
    )

    usage = "Available commands " + str(cmd_list) + '\n' + opthelp
    parser = argparse.ArgumentParser(
        description='Controls CMS-Compatible IP camera',
        usage=usage
    )
    parser.add_argument(
        'command',
        metavar='command',
        type=str,
        nargs=1,
        help=' | '.join(cmd_list)
    )
    parser.add_argument(
        'options',
        metavar='opts',
        type=str,
        nargs='*',
        help=opthelp
    )

    parser.add_argument(
        '-c', '--config',
        nargs=1,
        metavar='CONFIG_PATH',
        type=str,
        help="Path to a config file which will be used instead of the default one."
    )

    cml_args = parser.parse_args()
    cmd = cml_args.command[0]
    if cml_args.options is not None:
        opts = cml_args.options
    else:
        opts = ''

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, 'notused')
    # initialise a logger, when running in standalone mode, to avoid DVRip's excessive debug messages
    logger = initLogging(config, log_file_prefix='camControl_')
    
    if cmd not in cmd_list:
        log.info('Error: command "%s" not supported', cmd)
        exit(1)

    cameraControlV2(config, cmd, opts)


"""Known Field mappings
These are available in Guides/imx2910config-maps.md

To set these values pass split at the dot if there is one, then call SetParam
eg
  SetParam ExposureParam Level 0
  SetParam DayNightColor 0
Decimals will be converted to hex strings if necessary.
"""
