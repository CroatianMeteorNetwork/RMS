from __future__ import print_function, division, absolute_import

import os
import json
import time
from datetime import timedelta
import ephem
import Utils.CameraControl as cc
from RMS.Logger import getLogger
from RMS.Misc import RmsDateTime

# Get the logger from the main module
log = getLogger("logger")

# Sun altitude (in degrees) that defines the switch point.
# Negative numbers mean the Sun is below the horizon.
SWITCH_HORIZON_DEG = "-9"


def switchCameraMode(config, daytime_mode, camera_mode_switch_trigger):
    """
    Attempt to switch the camera to 'day' or 'night' using external JSON-based mode definitions.

    Arguments:
        config: RMS config object
        daytime_mode: multiprocessing.Value(bool) indicating day/night
        camera_mode_switch_trigger: multiprocessing.Value(bool) flag to trigger switching
    """
    mode_name = "day" if daytime_mode.value else "night"

    mode_path = config.camera_settings_path

    try:
        if not os.path.exists(mode_path):
            raise FileNotFoundError("Mode file {} not found.".format(mode_path))

        with open(mode_path, 'r') as f:
            modes = json.load(f)

        if mode_name not in modes:
            raise KeyError("Mode '{}' not defined in {}.".format(mode_name, mode_path))

        try:
            cc.cameraControlV2(config, "SwitchMode", mode_name)
        except Exception as e:
            raise RuntimeError("Failed to switch camera mode: {}".format(e))

        # After successful camera mode switching, don't keep trying
        camera_mode_switch_trigger.value = False
        log.info("Successfully switched camera mode to %s", mode_name)

    except Exception as e:
        log.warning("Camera switch to %s mode failed: %s. Will retry later.", mode_name, e)

        # After failure, retry on next opportunity
        camera_mode_switch_trigger.value = True

# Function to switch capture between day and night modes
def captureModeSwitcher(config, daytime_mode, camera_mode_switch_trigger):
    """ Wait and switch between day and night capture modes based on current time.
    
    Arguments:
        config: [Config] config object for determining location and controlling camera settings if specified
        daytime_mode: [multiprocessing.Value] shared boolean variable to communicate the mode switch with other processes
                            True = Day time, False = Night time
    """
    is_first_switch = True  # Track whether it's the initial switch

    try:
        while True:

            # Initialize observer
            o = ephem.Observer()
            o.lat = str(config.latitude)
            o.long = str(config.longitude)
            o.elevation = config.elevation

            # The Sun should be about 9 degrees below the horizon when the capture modes switch
            o.horizon = SWITCH_HORIZON_DEG

            # Set the current time
            current_time = RmsDateTime.utcnow()
            o.date = current_time

            # Calculate sun positions
            s = ephem.Sun()
            s.compute()

            # Based on whether next event is a sunrise or sunset, set the value for daytime_mode
            # Except on initial run, apply the capture_wait_seconds from the config to stagger the mode
            # switching. This can help prevent disconnects in multi-camera setups
            try:
                next_rise = o.next_rising(s).datetime()
                next_set = o.next_setting(s).datetime()

                if next_set < next_rise:
                    log.info("Next event is a sunset ({}), switching to daytime mode".format(next_set))

                    if config.switch_camera_modes:
                        # Delay before switching camera modes to prevent multiple cameras 
                        # from switching simultaneously (avoids power spikes)
                        # Initial run already has startup delay, so skip additional wait
                        if not is_first_switch:
                            time.sleep(config.capture_wait_seconds)
                        camera_mode_switch_trigger.value = True

                    daytime_mode.value = True
                    time_to_wait = (next_set - current_time).total_seconds()

                else:
                    log.info("Next event is a sunrise ({}), switching to nighttime mode".format(next_rise))

                    if config.switch_camera_modes:
                        # Delay before switching camera modes to prevent multiple cameras 
                        # from switching simultaneously (avoids power spikes)
                        # Initial run already has startup delay, so skip additional wait
                        if not is_first_switch:
                            time.sleep(config.capture_wait_seconds)
                        camera_mode_switch_trigger.value = True

                    daytime_mode.value = False
                    time_to_wait = (next_rise - current_time).total_seconds()


            # If the day last more than 24 hours, continue daytime capture for the whole day
            except ephem.AlwaysUpError:

                if config.switch_camera_modes:
                    # Switch immediately in polar day conditions
                    # No sunset to wait for, and startup delay already applied
                    camera_mode_switch_trigger.value = True

                daytime_mode.value = True
                time_to_wait = 86400


            # If the night lasts more than 24 hours, continue nighttime capture for the whole day
            except ephem.NeverUpError:

                if config.switch_camera_modes:
                    # Switch immediately in polar night conditions
                    # No sunrise to wait for, and startup delay already applied
                    camera_mode_switch_trigger.value = True

                daytime_mode.value = False
                time_to_wait = 86400

            # Mark that the first switch has occurred
            is_first_switch = False

            # Sleep until the next switch time
            time.sleep(time_to_wait)


    except Exception as e:

        log.error('CaptureModeSwitcher thread failed with following error: ' + repr(e))



def lastNightToDaySwitch(config, whenUtc=None):
    """Return the UTC timestamp of the most recent night-to-day switch.

    Arguments:
        config: [Config] RMS configuration object; must expose latitude,
            longitude and elevation attributes.

    Keyword arguments:
        whenUtc: [datetime] Naive UTC time used as the upper bound of the
            search window. None by default, which means the current
            ``RmsDateTime.utcnow()`` is used.

    Return:
        last_switch: [datetime] UTC time at which the Sun last rose above
            ``SWITCH_HORIZON_DEG`` before *whenUtc* (or the preceding midnight
            if the Sun is always up/down at that location).
    """
    if whenUtc is None:
        whenUtc = RmsDateTime.utcnow()

    obs = ephem.Observer()
    obs.lat = str(config.latitude)
    obs.long = str(config.longitude)
    obs.elevation = config.elevation
    obs.horizon = SWITCH_HORIZON_DEG
    obs.date = ephem.Date(whenUtc)

    sun = ephem.Sun()
    try:
        # Account for programmed delay in mode switching
        wait = timedelta(seconds=config.capture_wait_seconds)
        previous_sunrise = obs.previous_rising(sun).datetime()
        return previous_sunrise + wait
    
    except (ephem.AlwaysUpError, ephem.NeverUpError):
        # Fallback: last midnight before whenUtc
        midnight = whenUtc.replace(hour=0, minute=0, second=0, microsecond=0)
        if midnight >= whenUtc:
            midnight -= timedelta(days=1)
        return midnight


    ### For testing switching only ###

    # wait_interval = 5*60
    
    # while True:

    #     if not daytime_mode.value:
    #         log.info('Switching to day time mode')
    #         daytime_mode.value = True

    #         if config.switch_camera_modes:
    #             cc.cameraControlV2(config, 'SwitchDayTime')

    #     else:
    #         log.info('Switching to night time mode')
    #         daytime_mode.value = False
    
    #         if config.switch_camera_modes:
    #             cc.cameraControlV2(config, 'SwitchNightTime')

    #     time.sleep(wait_interval)


if __name__ == "__main__":
    
    import RMS.ConfigReader as cr
    import os

    config = cr.loadConfigFromDirectory('.', os.path.abspath('.'))

    config.latitude = -80.833763
    config.longitude = -44.674523
    config.elevation = -20
    config.switch_camera_modes = False

    # Test the time now - remove daytime_mode above
    # captureModeSwitcher(config)
    
