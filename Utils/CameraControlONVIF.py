# noqa:E501,W291
""" Controls ONVIF-compatible IP cameras
    This module provides camera control functionality for ONVIF-compliant cameras,
    mirroring the interface of CameraControl.py (which uses DVRIP protocol).

    Note: Requires the onvif-zeep library: pip install onvif-zeep

    usage 1:
    python -m Utils.CameraControlONVIF command {opts}
    call with -h to get a list of supported commands

    Usage 2:
    >>> import Utils.CameraControlONVIF as cc
    >>> cc.cameraControl(ip_address, port, username, password, command, [opts])
    >>> cc.cameraControlV2(config, command, [opts])

    Parameters:
    ip_address: dotted ipaddress of the camera eg 192.168.1.10
    port: ONVIF port (typically 80 or 8080)
    username: ONVIF username
    password: ONVIF password
    command: the command you want to execute
    opts: field and value to use when calling SetParam

    example
    python -m Utils.CameraControlONVIF GetCameraParams
    python -m Utils.CameraControlONVIF GetEncodeParams
    will return JSON showing you the contents of the Camera Params block and Video Encoding blocks

    Supported commands:
    - reboot: Reboot the camera
    - GetHostname: Get the camera hostname
    - GetIP: Get the camera IP address
    - GetNetConfig: Get network configuration
    - GetCameraParams: Get imaging parameters (brightness, contrast, exposure, etc.)
    - GetEncodeParams: Get video encoding parameters
    - GetSettings: Get all settings
    - SetParam: Set a parameter (Camera/Network/Encode block)
    - CameraTime: Get or set camera time
    - SetColor: Set color/brightness settings

    - SwitchMode: Switch between day/night modes using camera_settings_onvif.json

    Note: Some DVRIP-specific commands may not have direct ONVIF equivalents.
    Commands like SaveSettings, LoadSettings, SetOSD, SetAutoReboot, CloudConnection
    are not supported in this ONVIF implementation.

"""

import sys
import os
import argparse
import json
import pprint
import re
import datetime

try:
    from onvif import ONVIFCamera
except ImportError:
    print("Error: onvif-zeep library not installed. Install with: pip install onvif-zeep")
    sys.exit(1)

import RMS.ConfigReader as cr
from RMS.Logger import getLogger, initLogging
from time import sleep

# Get the logger from the main module
log = getLogger("logger")


def connectCamera(ip_address, port, username, password):
    """Connect to an ONVIF camera and return the camera object.

    Args:
        ip_address: IP address of the camera
        port: ONVIF port (typically 80 or 8080)
        username: ONVIF username
        password: ONVIF password

    Returns:
        ONVIFCamera object or None if connection failed
    """
    try:
        cam = ONVIFCamera(ip_address, port, username, password)
        return cam
    except Exception as e:
        log.error("Failed to connect to camera: %s", e)
        return None


def rebootCamera(cam):
    """Reboot the Camera.

    Args:
        cam: The ONVIFCamera object
    """
    log.info('Rebooting camera, please wait...')
    try:
        cam.devicemgmt.SystemReboot()
        log.info('Reboot command sent successfully')
        log.info('Camera will be unavailable for 30-60 seconds')
    except Exception as e:
        log.error("Failed to reboot camera: %s", e)


def getHostname(cam):
    """Get the camera hostname.

    Args:
        cam: The ONVIFCamera object

    Returns:
        Hostname string
    """
    try:
        resp = cam.devicemgmt.GetHostname()
        hostname = resp.Name if hasattr(resp, 'Name') else str(resp)
        log.info('Hostname: %s', hostname)
        return hostname
    except Exception as e:
        log.error("Failed to get hostname: %s", e)
        return None


def getNetworkParams(cam, showit=True):
    """Retrieve or display the camera network settings.

    Args:
        cam: ONVIFCamera object
        showit (bool, optional): whether to log out the settings.

    Returns:
        dict containing network configuration
    """
    try:
        # Get network interfaces
        interfaces = cam.devicemgmt.GetNetworkInterfaces()

        # Get DNS configuration
        dns = cam.devicemgmt.GetDNS()

        # Get NTP configuration
        try:
            ntp = cam.devicemgmt.GetNTP()
        except Exception:
            ntp = None

        # Get hostname
        hostname = cam.devicemgmt.GetHostname()

        result = {
            'Hostname': hostname.Name if hasattr(hostname, 'Name') else str(hostname),
            'Interfaces': [],
            'DNS': {},
            'NTP': {}
        }

        for iface in interfaces:
            iface_info = {
                'Name': iface.Info.Name if hasattr(iface.Info, 'Name') else 'Unknown',
                'Enabled': iface.Enabled if hasattr(iface, 'Enabled') else True,
            }

            if hasattr(iface, 'IPv4') and iface.IPv4:
                if hasattr(iface.IPv4, 'Config') and iface.IPv4.Config:
                    config = iface.IPv4.Config
                    iface_info['DHCP'] = config.DHCP if hasattr(config, 'DHCP') else False
                    if hasattr(config, 'Manual') and config.Manual:
                        manual = config.Manual[0] if isinstance(config.Manual, list) else config.Manual
                        iface_info['IPAddress'] = manual.Address if hasattr(manual, 'Address') else 'Unknown'
                        iface_info['PrefixLength'] = manual.PrefixLength if hasattr(manual, 'PrefixLength') else 24
                    elif hasattr(config, 'FromDHCP') and config.FromDHCP:
                        dhcp = config.FromDHCP
                        iface_info['IPAddress'] = dhcp.Address if hasattr(dhcp, 'Address') else 'Unknown'
                        iface_info['PrefixLength'] = dhcp.PrefixLength if hasattr(dhcp, 'PrefixLength') else 24

            result['Interfaces'].append(iface_info)

        if dns:
            result['DNS']['FromDHCP'] = dns.FromDHCP if hasattr(dns, 'FromDHCP') else False
            if hasattr(dns, 'DNSManual') and dns.DNSManual:
                result['DNS']['Servers'] = [s.IPv4Address for s in dns.DNSManual if hasattr(s, 'IPv4Address')]

        if ntp:
            result['NTP']['FromDHCP'] = ntp.FromDHCP if hasattr(ntp, 'FromDHCP') else False
            if hasattr(ntp, 'NTPManual') and ntp.NTPManual:
                result['NTP']['Servers'] = [s.IPv4Address for s in ntp.NTPManual if hasattr(s, 'IPv4Address')]

        if showit:
            log.info('Network Configuration:')
            log.info('---------')
            log.info(pprint.pformat(result))

        return result

    except Exception as e:
        log.error("Failed to get network params: %s", e)
        return None


def getIP(cam):
    """Get the camera IP address.

    Args:
        cam: ONVIFCamera object
    """
    try:
        interfaces = cam.devicemgmt.GetNetworkInterfaces()
        for iface in interfaces:
            if hasattr(iface, 'IPv4') and iface.IPv4:
                if hasattr(iface.IPv4, 'Config') and iface.IPv4.Config:
                    config = iface.IPv4.Config
                    if hasattr(config, 'Manual') and config.Manual:
                        manual = config.Manual[0] if isinstance(config.Manual, list) else config.Manual
                        log.info(manual.Address)
                        return manual.Address
                    elif hasattr(config, 'FromDHCP') and config.FromDHCP:
                        log.info(config.FromDHCP.Address)
                        return config.FromDHCP.Address
        log.info('Could not determine IP address')
        return None
    except Exception as e:
        log.error("Failed to get IP: %s", e)
        return None


def getVideoSourceToken(cam):
    """Get the video source token needed for imaging service.

    Args:
        cam: ONVIFCamera object

    Returns:
        Video source token string
    """
    try:
        media = cam.create_media_service()
        video_sources = media.GetVideoSources()
        if video_sources:
            return video_sources[0].token
        return None
    except Exception as e:
        log.error("Failed to get video source token: %s", e)
        return None


def getCameraParams(cam, showit=True):
    """Display or retrieve the imaging parameters of the camera.

    Args:
        cam: ONVIFCamera object
        showit (bool, optional): whether to log out the settings.

    Returns:
        dict containing imaging parameters
    """
    try:
        imaging = cam.create_imaging_service()
        media = cam.create_media_service()
        video_sources = media.GetVideoSources()

        if not video_sources:
            log.error("No video sources found")
            return None

        token = video_sources[0].token
        settings = imaging.GetImagingSettings({'VideoSourceToken': token})

        # Get imaging options to understand valid ranges
        try:
            options = imaging.GetOptions({'VideoSourceToken': token})
        except Exception:
            options = None

        result = {
            'ImagingSettings': {},
            'Options': {}
        }

        # Extract settings into a clean dict
        if hasattr(settings, 'Brightness'):
            result['ImagingSettings']['Brightness'] = settings.Brightness
        if hasattr(settings, 'Contrast'):
            result['ImagingSettings']['Contrast'] = settings.Contrast
        if hasattr(settings, 'ColorSaturation'):
            result['ImagingSettings']['ColorSaturation'] = settings.ColorSaturation
        if hasattr(settings, 'Sharpness'):
            result['ImagingSettings']['Sharpness'] = settings.Sharpness

        if hasattr(settings, 'Exposure') and settings.Exposure:
            exp = settings.Exposure
            result['ImagingSettings']['Exposure'] = {
                'Mode': exp.Mode if hasattr(exp, 'Mode') else 'Unknown',
            }
            if hasattr(exp, 'MinExposureTime'):
                result['ImagingSettings']['Exposure']['MinExposureTime'] = exp.MinExposureTime
            if hasattr(exp, 'MaxExposureTime'):
                result['ImagingSettings']['Exposure']['MaxExposureTime'] = exp.MaxExposureTime
            if hasattr(exp, 'MinGain'):
                result['ImagingSettings']['Exposure']['MinGain'] = exp.MinGain
            if hasattr(exp, 'MaxGain'):
                result['ImagingSettings']['Exposure']['MaxGain'] = exp.MaxGain
            if hasattr(exp, 'ExposureTime'):
                result['ImagingSettings']['Exposure']['ExposureTime'] = exp.ExposureTime
            if hasattr(exp, 'Gain'):
                result['ImagingSettings']['Exposure']['Gain'] = exp.Gain

        if hasattr(settings, 'WhiteBalance') and settings.WhiteBalance:
            wb = settings.WhiteBalance
            result['ImagingSettings']['WhiteBalance'] = {
                'Mode': wb.Mode if hasattr(wb, 'Mode') else 'Unknown',
            }

        if hasattr(settings, 'WideDynamicRange') and settings.WideDynamicRange:
            wdr = settings.WideDynamicRange
            result['ImagingSettings']['WideDynamicRange'] = {
                'Mode': wdr.Mode if hasattr(wdr, 'Mode') else 'Unknown',
                'Level': wdr.Level if hasattr(wdr, 'Level') else None
            }

        if hasattr(settings, 'IrCutFilter'):
            result['ImagingSettings']['IrCutFilter'] = settings.IrCutFilter

        if hasattr(settings, 'BacklightCompensation') and settings.BacklightCompensation:
            blc = settings.BacklightCompensation
            result['ImagingSettings']['BacklightCompensation'] = {
                'Mode': blc.Mode if hasattr(blc, 'Mode') else 'Unknown',
                'Level': blc.Level if hasattr(blc, 'Level') else None
            }

        # Extract options if available
        if options:
            if hasattr(options, 'Brightness'):
                result['Options']['Brightness'] = {
                    'Min': options.Brightness.Min if hasattr(options.Brightness, 'Min') else 0,
                    'Max': options.Brightness.Max if hasattr(options.Brightness, 'Max') else 100
                }
            if hasattr(options, 'Contrast'):
                result['Options']['Contrast'] = {
                    'Min': options.Contrast.Min if hasattr(options.Contrast, 'Min') else 0,
                    'Max': options.Contrast.Max if hasattr(options.Contrast, 'Max') else 100
                }

        if showit:
            log.info('Camera/Imaging Parameters:')
            log.info('---------')
            log.info(pprint.pformat(result))

        return result

    except Exception as e:
        log.error("Failed to get camera params: %s", e)
        return None


def getEncodeParams(cam, showit=True):
    """Read the video encoding parameters of the camera.

    Args:
        cam: ONVIFCamera object
        showit (bool, optional): whether to log out the settings.

    Returns:
        dict containing encoding parameters
    """
    try:
        media = cam.create_media_service()
        profiles = media.GetProfiles()

        result = {'Profiles': []}

        for profile in profiles:
            profile_info = {
                'Name': profile.Name if hasattr(profile, 'Name') else 'Unknown',
                'Token': profile.token if hasattr(profile, 'token') else 'Unknown',
            }

            if hasattr(profile, 'VideoEncoderConfiguration') and profile.VideoEncoderConfiguration:
                vec = profile.VideoEncoderConfiguration
                profile_info['VideoEncoder'] = {
                    'Name': vec.Name if hasattr(vec, 'Name') else 'Unknown',
                    'Encoding': vec.Encoding if hasattr(vec, 'Encoding') else 'Unknown',
                    'Quality': vec.Quality if hasattr(vec, 'Quality') else None,
                }

                if hasattr(vec, 'Resolution') and vec.Resolution:
                    profile_info['VideoEncoder']['Resolution'] = {
                        'Width': vec.Resolution.Width,
                        'Height': vec.Resolution.Height
                    }

                if hasattr(vec, 'RateControl') and vec.RateControl:
                    profile_info['VideoEncoder']['RateControl'] = {
                        'FrameRateLimit': vec.RateControl.FrameRateLimit if hasattr(vec.RateControl, 'FrameRateLimit') else None,
                        'BitrateLimit': vec.RateControl.BitrateLimit if hasattr(vec.RateControl, 'BitrateLimit') else None,
                    }

                if hasattr(vec, 'H264') and vec.H264:
                    profile_info['VideoEncoder']['H264'] = {
                        'GovLength': vec.H264.GovLength if hasattr(vec.H264, 'GovLength') else None,
                        'H264Profile': vec.H264.H264Profile if hasattr(vec.H264, 'H264Profile') else None,
                    }

            result['Profiles'].append(profile_info)

        if showit:
            log.info('Video Encoding Parameters:')
            log.info('---------')
            log.info(pprint.pformat(result))

        return result

    except Exception as e:
        log.error("Failed to get encode params: %s", e)
        return None


def getDeviceInfo(cam, showit=True):
    """Get device information.

    Args:
        cam: ONVIFCamera object
        showit: Whether to log the output

    Returns:
        dict containing device information
    """
    try:
        info = cam.devicemgmt.GetDeviceInformation()

        result = {
            'Manufacturer': info.Manufacturer if hasattr(info, 'Manufacturer') else 'Unknown',
            'Model': info.Model if hasattr(info, 'Model') else 'Unknown',
            'FirmwareVersion': info.FirmwareVersion if hasattr(info, 'FirmwareVersion') else 'Unknown',
            'SerialNumber': info.SerialNumber if hasattr(info, 'SerialNumber') else 'Unknown',
            'HardwareId': info.HardwareId if hasattr(info, 'HardwareId') else 'Unknown',
        }

        if showit:
            log.info('Device Information:')
            log.info('---------')
            log.info(pprint.pformat(result))

        return result

    except Exception as e:
        log.error("Failed to get device info: %s", e)
        return None


def getCameraTime(cam):
    """Get the camera's current date and time.

    Args:
        cam: ONVIFCamera object

    Returns:
        datetime object or None
    """
    try:
        dt = cam.devicemgmt.GetSystemDateAndTime()

        if hasattr(dt, 'UTCDateTime') and dt.UTCDateTime:
            utc = dt.UTCDateTime
            camera_time = datetime.datetime(
                utc.Date.Year, utc.Date.Month, utc.Date.Day,
                utc.Time.Hour, utc.Time.Minute, utc.Time.Second
            )
            log.info('Camera time (UTC): %s', camera_time)
            return camera_time
        elif hasattr(dt, 'LocalDateTime') and dt.LocalDateTime:
            local = dt.LocalDateTime
            camera_time = datetime.datetime(
                local.Date.Year, local.Date.Month, local.Date.Day,
                local.Time.Hour, local.Time.Minute, local.Time.Second
            )
            log.info('Camera time (Local): %s', camera_time)
            return camera_time
        else:
            log.info('Could not parse camera time')
            return None

    except Exception as e:
        log.error("Failed to get camera time: %s", e)
        return None


def setCameraTime(cam, new_time=None):
    """Set the camera's date and time.

    Args:
        cam: ONVIFCamera object
        new_time: datetime object (defaults to current system time)
    """
    try:
        if new_time is None:
            new_time = datetime.datetime.utcnow()

        # Create the request
        request = cam.devicemgmt.create_type('SetSystemDateAndTime')
        request.DateTimeType = 'Manual'
        request.DaylightSavings = False
        request.UTCDateTime = {
            'Date': {
                'Year': new_time.year,
                'Month': new_time.month,
                'Day': new_time.day
            },
            'Time': {
                'Hour': new_time.hour,
                'Minute': new_time.minute,
                'Second': new_time.second
            }
        }

        cam.devicemgmt.SetSystemDateAndTime(request)
        log.info('Camera time set to: %s', new_time)

    except Exception as e:
        log.error("Failed to set camera time: %s", e)


def setImagingParam(cam, param_name, value):
    """Set an imaging parameter.

    Args:
        cam: ONVIFCamera object
        param_name: Parameter name (Brightness, Contrast, ColorSaturation, Sharpness)
        value: Value to set
    """
    try:
        imaging = cam.create_imaging_service()
        token = getVideoSourceToken(cam)

        if not token:
            log.error("Could not get video source token")
            return

        request = imaging.create_type('SetImagingSettings')
        request.VideoSourceToken = token
        request.ImagingSettings = {param_name: float(value)}

        imaging.SetImagingSettings(request)
        log.info('Set %s to %s', param_name, value)

    except Exception as e:
        log.error("Failed to set imaging param %s: %s", param_name, e)


def setExposureParam(cam, param_name, value):
    """Set an exposure parameter.

    Args:
        cam: ONVIFCamera object
        param_name: Parameter name (Mode, MinExposureTime, MaxExposureTime, MinGain, MaxGain, etc.)
        value: Value to set
    """
    try:
        imaging = cam.create_imaging_service()
        token = getVideoSourceToken(cam)

        if not token:
            log.error("Could not get video source token")
            return

        request = imaging.create_type('SetImagingSettings')
        request.VideoSourceToken = token

        # For exposure mode, value is a string (AUTO or MANUAL)
        if param_name == 'Mode':
            request.ImagingSettings = {'Exposure': {'Mode': value}}
        else:
            request.ImagingSettings = {'Exposure': {param_name: float(value)}}

        imaging.SetImagingSettings(request)
        log.info('Set Exposure.%s to %s', param_name, value)

    except Exception as e:
        log.error("Failed to set exposure param %s: %s", param_name, e)


def setColor(cam, opts):
    """Set color/imaging parameters (brightness, contrast, saturation, sharpness).

    Args:
        cam: ONVIFCamera object
        opts: array containing comma-separated values: brightness,contrast,saturation,sharpness
    """
    if not opts:
        log.info('usage: SetColor brightness,contrast,saturation,sharpness')
        log.info('  all values typically range from 0 to 100')
        return

    try:
        parts = opts[0].split(',')
        brightness = float(parts[0]) if len(parts) > 0 else None
        contrast = float(parts[1]) if len(parts) > 1 else None
        saturation = float(parts[2]) if len(parts) > 2 else None
        sharpness = float(parts[3]) if len(parts) > 3 else None

        imaging = cam.create_imaging_service()
        token = getVideoSourceToken(cam)

        if not token:
            log.error("Could not get video source token")
            return

        settings = {}
        if brightness is not None:
            settings['Brightness'] = brightness
        if contrast is not None:
            settings['Contrast'] = contrast
        if saturation is not None:
            settings['ColorSaturation'] = saturation
        if sharpness is not None:
            settings['Sharpness'] = sharpness

        request = imaging.create_type('SetImagingSettings')
        request.VideoSourceToken = token
        request.ImagingSettings = settings

        imaging.SetImagingSettings(request)
        log.info('Set color configuration: brightness=%s, contrast=%s, saturation=%s, sharpness=%s',
                 brightness, contrast, saturation, sharpness)

    except Exception as e:
        log.error("Failed to set color: %s", e)


def setNetworkParam(cam, opts):
    """Set a network parameter.

    Args:
        cam: ONVIFCamera object
        opts: array of field and value to set
    """
    if len(opts) < 2:
        log.info('usage: SetParam Network option value')
        log.info('Options: Hostname, EnableDHCP, EnableNTP')
        return

    field = opts[1]

    try:
        if field == 'Hostname':
            value = opts[2]
            request = cam.devicemgmt.create_type('SetHostname')
            request.Name = value
            cam.devicemgmt.SetHostname(request)
            log.info('Set hostname to %s', value)

        elif field == 'EnableDHCP':
            log.info('DHCP configuration via ONVIF requires interface-specific setup')
            log.info('Use the camera web interface for network configuration')

        elif field == 'EnableNTP':
            value = opts[2]
            if value == "0":
                request = cam.devicemgmt.create_type('SetNTP')
                request.FromDHCP = False
                request.NTPManual = []
                cam.devicemgmt.SetNTP(request)
                log.info('NTP disabled')
            else:
                request = cam.devicemgmt.create_type('SetNTP')
                request.FromDHCP = False
                request.NTPManual = [{'Type': 'IPv4', 'IPv4Address': value}]
                cam.devicemgmt.SetNTP(request)
                log.info('NTP enabled with server %s', value)

        else:
            log.info('Network parameter %s not supported', field)
            log.info('Supported: Hostname, EnableNTP')

    except Exception as e:
        log.error("Failed to set network param: %s", e)


def setCameraParam(cam, opts):
    """Set a camera/imaging parameter.

    Args:
        cam: ONVIFCamera object
        opts: array of fields and value to set
    """
    if len(opts) < 3:
        log.info('usage: SetParam Camera field value')
        log.info('Fields: Brightness, Contrast, ColorSaturation, Sharpness')
        log.info('Or: SetParam Camera ExposureParam subfield value')
        return

    field = opts[1]

    # Simple fields
    simple_fields = ['Brightness', 'Contrast', 'ColorSaturation', 'Sharpness']

    if field in simple_fields:
        value = opts[2]
        setImagingParam(cam, field, value)

    elif field == 'ExposureParam':
        if len(opts) < 4:
            log.info('usage: SetParam Camera ExposureParam subfield value')
            log.info('Subfields: Mode (AUTO/MANUAL), MinExposureTime, MaxExposureTime, MinGain, MaxGain')
            return
        subfield = opts[2]
        value = opts[3]
        setExposureParam(cam, subfield, value)

    elif field == 'IrCutFilter':
        value = opts[2].upper()
        if value not in ('ON', 'OFF', 'AUTO'):
            log.info('IrCutFilter must be ON, OFF, or AUTO')
            return
        try:
            imaging = cam.create_imaging_service()
            token = getVideoSourceToken(cam)
            request = imaging.create_type('SetImagingSettings')
            request.VideoSourceToken = token
            request.ImagingSettings = {'IrCutFilter': value}
            imaging.SetImagingSettings(request)
            log.info('Set IrCutFilter to %s', value)
        except Exception as e:
            log.error("Failed to set IrCutFilter: %s", e)

    else:
        log.info('Camera parameter %s not supported', field)
        log.info('Supported: Brightness, Contrast, ColorSaturation, Sharpness, ExposureParam, IrCutFilter')


def setParameter(cam, opts):
    """Set a parameter in various sections of the camera config.

    Args:
        cam: ONVIFCamera object
        opts: array of block, fields, and value to set
    """
    if len(opts) < 3:
        log.info('Not enough parameters, need at least block, field, value')
        log.info('Blocks: Camera, Network')
        return

    block = opts[0]

    if block == 'Camera':
        setCameraParam(cam, opts)
    elif block == 'Network':
        setNetworkParam(cam, opts)
    else:
        log.info('Block %s not supported. Use Camera or Network', block)


def switchMode(cam, mode_name, path='./camera_settings_onvif.json'):
    """Switch the camera to a named mode by executing the commands in the JSON file.

    This mirrors the functionality of CameraControl.switchMode for DVRIP cameras,
    but uses ONVIF-compatible commands.

    Args:
        cam: An authenticated ONVIFCamera object.
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

    log.info("Switching to mode '%s'", mode_name)

    # Loop over each command array in the specified mode
    for param in modes[mode_name]:
        cmd = param[0]
        opts = param[1:] if len(param) > 1 else []

        # Avoid calling "SwitchMode" from within switchMode to prevent recursion
        if cmd == "SwitchMode":
            log.warning("Ignoring SwitchMode command inside JSON to prevent recursion.")
            continue

        # Pass everything else to onvifCall
        try:
            onvifCall(cam, cmd, opts, path)
        except Exception as e:
            log.error("Error executing command '%s' with opts %s: %s", cmd, opts, e)

    log.info("Mode switch to '%s' completed", mode_name)


def onvifCall(cam, cmd, opts, camera_settings_path='./camera_settings_onvif.json'):
    """Execute an ONVIF command.

    Args:
        cam: ONVIFCamera object
        cmd: the command to execute
        opts: optional list of parameters
        camera_settings_path: path to the camera settings JSON file for SwitchMode
    """
    if cmd == 'GetHostname':
        getHostname(cam)
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

    elif cmd == 'GetCameraParams':
        getCameraParams(cam, True)
        return

    elif cmd == 'GetEncodeParams':
        getEncodeParams(cam, True)
        return

    elif cmd == 'GetDeviceInformation':
        getDeviceInfo(cam, True)
        return

    elif cmd == 'GetSettings':
        getDeviceInfo(cam, True)
        getNetworkParams(cam, True)
        getCameraParams(cam, True)
        getEncodeParams(cam, True)
        return

    elif cmd == 'SetParam':
        setParameter(cam, opts)
        return

    elif cmd == 'CameraTime':
        if not opts:
            log.info('usage: CameraTime get|set [YYYYMMDD_HHMMSS]')
            return
        if opts[0] == 'get':
            getCameraTime(cam)
        elif opts[0] == 'set':
            if len(opts) > 1:
                try:
                    reqtime = datetime.datetime.strptime(opts[1], '%Y%m%d_%H%M%S')
                except Exception:
                    reqtime = datetime.datetime.utcnow()
            else:
                reqtime = datetime.datetime.utcnow()
            setCameraTime(cam, reqtime)
        else:
            log.info('usage: CameraTime get|set [YYYYMMDD_HHMMSS]')
        return

    elif cmd == 'SetColor':
        setColor(cam, opts)
        return

    elif cmd == 'SwitchMode':
        if not opts:
            log.error("No mode specified for SwitchMode.")
            log.info("Usage: SwitchMode <mode_name>")
            log.info("Modes are defined in camera_settings_onvif.json (e.g., 'day', 'night')")
            return

        # If opts is just a string, use it directly; if it's a list, pull the first element
        if isinstance(opts, str):
            mode_name = opts
        else:
            mode_name = opts[0]

        switchMode(cam, mode_name, camera_settings_path)
        return

    # Unsupported commands (DVRIP-specific)
    elif cmd in ('SaveSettings', 'LoadSettings', 'SetOSD', 'SetAutoReboot',
                 'GetAutoReboot', 'CloudConnection'):
        log.info('Command %s is not supported for ONVIF cameras', cmd)
        log.info('This command is specific to DVRIP/CMS cameras')
        return

    else:
        log.info('Unknown command: %s', cmd)
        getDeviceInfo(cam, True)
        return


def cameraControl(camera_ip, port, username, password, cmd, opts='',
                   camera_settings_path='./camera_settings_onvif.json'):
    """CameraControl - main entry point to the module.

    Args:
        camera_ip (string): IP Address of camera in dotted form eg 192.168.1.10
        port (int): ONVIF port (typically 80 or 8080)
        username (string): ONVIF username
        password (string): ONVIF password
        cmd (string): Command to be executed
        opts (array of strings): Optional array of field, subfield and value for SetParam
        camera_settings_path (string): Path to camera settings JSON file for SwitchMode
    """
    cam = connectCamera(camera_ip, port, username, password)
    if cam:
        try:
            onvifCall(cam, cmd, opts, camera_settings_path)
        except Exception as e:
            log.error("Error executing command: %s", e)
            log.error("This command may not be supported by your camera.")
    else:
        log.info("Failure. Could not connect.")


def cameraControlV2(config, cmd, opts=''):
    """High-level entry point that uses config to figure out IP, port, and credentials.

    Args:
        config: RMS config object (must have onvif_ip, onvif_port, onvif_user, onvif_password)
        cmd: Command to execute
        opts: Optional parameters
    """
    # Check for required ONVIF config options
    if not hasattr(config, 'onvif_ip') or not config.onvif_ip:
        # Try to extract from deviceID if it's an IP address
        if hasattr(config, 'deviceID') and not str(config.deviceID).isdigit():
            ip_match = re.findall(r"[0-9]+(?:\.[0-9]+){3}", str(config.deviceID))
            if ip_match:
                camera_ip = ip_match[0]
            else:
                log.error('Error: onvif_ip not configured and could not extract IP from deviceID')
                return
        else:
            log.error('Error: onvif_ip not configured')
            return
    else:
        camera_ip = config.onvif_ip

    port = getattr(config, 'onvif_port', 80)
    username = getattr(config, 'onvif_user', 'admin')
    password = getattr(config, 'onvif_password', '')

    # Determine camera settings path
    if hasattr(config, 'camera_settings_path_onvif') and os.path.isfile(config.camera_settings_path_onvif):
        camera_settings_path = config.camera_settings_path_onvif
    elif hasattr(config, 'camera_settings_path') and config.camera_settings_path:
        # Try ONVIF variant of the DVRIP settings path
        base_path = os.path.splitext(config.camera_settings_path)[0]
        onvif_path = base_path + '_onvif.json'
        if os.path.isfile(onvif_path):
            camera_settings_path = onvif_path
        else:
            camera_settings_path = './camera_settings_onvif.json'
    else:
        camera_settings_path = './camera_settings_onvif.json'

    cameraControl(camera_ip, port, username, password, cmd, opts,
                  camera_settings_path=camera_settings_path)


if __name__ == '__main__':
    """Main function.
    Args:
        command - the command you want to execute
        opts - optional list of fields and a value to pass to SetParam
    """

    # List of supported commands
    cmd_list = [
        'reboot', 'GetHostname', 'GetSettings', 'GetDeviceInformation', 'GetNetConfig',
        'GetCameraParams', 'GetEncodeParams', 'SetParam', 'SetColor', 'GetIP', 'CameraTime',
        'SwitchMode'
    ]

    # Unsupported DVRIP commands (for reference)
    unsupported = ['SaveSettings', 'LoadSettings', 'SetOSD', 'SetAutoReboot',
                   'GetAutoReboot', 'CloudConnection']

    opthelp = (
        'optional parameters for SetParam for example Camera Brightness 70\n'
        'will set the brightness to 70.\n To see possibilities, execute GetSettings first. '
        'Call a function with no parameters to see the possibilities'
    )

    usage = "Available commands: " + str(cmd_list) + '\n' + opthelp
    parser = argparse.ArgumentParser(
        description='Controls ONVIF-Compatible IP camera',
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
    parser.add_argument(
        '--ip',
        type=str,
        help="Camera IP address (overrides config)"
    )
    parser.add_argument(
        '--port',
        type=int,
        default=80,
        help="ONVIF port (default: 80)"
    )
    parser.add_argument(
        '--user',
        type=str,
        default='admin',
        help="ONVIF username (default: admin)"
    )
    parser.add_argument(
        '--password',
        type=str,
        default='',
        help="ONVIF password"
    )
    parser.add_argument(
        '--settings',
        type=str,
        default='./camera_settings_onvif.json',
        help="Path to camera settings JSON file for SwitchMode (default: ./camera_settings_onvif.json)"
    )

    cml_args = parser.parse_args()
    cmd = cml_args.command[0]
    opts = cml_args.options if cml_args.options else []

    # If IP provided via command line, use direct connection
    if cml_args.ip:
        # Initialize logging
        log = getLogger("logger")

        if cmd not in cmd_list:
            log.info('Error: command "%s" not supported', cmd)
            log.info('Supported commands: %s', cmd_list)
            sys.exit(1)

        cameraControl(cml_args.ip, cml_args.port, cml_args.user, cml_args.password, cmd, opts,
                      camera_settings_path=cml_args.settings)
    else:
        # Load the config file
        config = cr.loadConfigFromDirectory(cml_args.config, 'notused')
        # Initialize a logger
        logger = initLogging(config, log_file_prefix='camControlONVIF_')

        if cmd not in cmd_list:
            log.info('Error: command "%s" not supported', cmd)
            log.info('Supported commands: %s', cmd_list)
            sys.exit(1)

        cameraControlV2(config, cmd, opts)


"""
ONVIF vs DVRIP Command Mapping:

ONVIF Supported:
- reboot -> SystemReboot
- GetHostname -> GetHostname
- GetIP -> GetNetworkInterfaces
- GetNetConfig -> GetNetworkInterfaces + GetDNS + GetNTP
- GetCameraParams -> GetImagingSettings (brightness, contrast, exposure, etc.)
- GetEncodeParams -> GetProfiles (video encoder configuration)
- GetDeviceInformation -> GetDeviceInformation
- GetSettings -> All of the above
- SetParam Camera -> SetImagingSettings
- SetParam Network -> SetHostname, SetNTP
- SetColor -> SetImagingSettings (brightness, contrast, saturation, sharpness)
- CameraTime -> GetSystemDateAndTime / SetSystemDateAndTime
- SwitchMode -> Executes commands from camera_settings_onvif.json

DVRIP-Only (Not Available in ONVIF):
- SaveSettings / LoadSettings (camera-side config backup)
- SetOSD (on-screen display - vendor specific in ONVIF)
- SetAutoReboot (vendor specific)
- GetAutoReboot (vendor specific)
- CloudConnection (DVRIP/XMeye specific)

DVRIP to ONVIF Parameter Mapping for SwitchMode:
- DayNightColor (0=Auto, 1=Day, 2=Night) -> IrCutFilter (AUTO/ON/OFF)
- ElecLevel (AE Reference) -> Brightness
- ExposureParam.LeastTime/MostTime -> Exposure.MinExposureTime/MaxExposureTime (microseconds)
- GainParam.Gain -> Exposure.Gain
- GainParam.AutoGain -> Exposure.Mode (AUTO/MANUAL)
- BroadTrends.AutoGain -> Exposure.Mode (AUTO/MANUAL)
- ClearFog -> WideDynamicRange (partial equivalent, vendor-specific)
"""
