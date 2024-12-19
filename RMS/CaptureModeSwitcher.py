from __future__ import print_function, division, absolute_import

import time
import ephem
import logging
import Utils.CameraControl as cc
from RMS.Misc import RmsDateTime

# Get the logger from the main module
log = logging.getLogger("logger")

# Function to switch capture between day and night modes
def captureModeSwitcher(config, daytime_mode):
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
            o.horizon = '-9'

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
                        if not is_first_switch:
                            time.sleep(config.capture_wait_seconds)
                        cc.cameraControlV2(config, 'SwitchDayTime')

                    daytime_mode.value = True
                    time_to_wait = (next_set - current_time).total_seconds()

                else:
                    log.info("Next event is a sunrise ({}), switching to nighttime mode".format(next_rise))

                    if config.switch_camera_modes:
                        if not is_first_switch:
                            time.sleep(config.capture_wait_seconds)
                        cc.cameraControlV2(config, 'SwitchNightTime')

                    daytime_mode.value = False
                    time_to_wait = (next_rise - current_time).total_seconds()


            # If the day last more than 24 hours, continue daytime capture for the whole day
            except ephem.AlwaysUpError:

                if config.switch_camera_modes:
                    if not is_first_switch:
                        time.sleep(config.capture_wait_seconds)
                    cc.cameraControlV2(config, 'SwitchDayTime')

                daytime_mode.value = True
                time_to_wait = 86400


            # If the night lasts more than 24 hours, continue nighttime capture for the whole day
            except ephem.NeverUpError:

                if config.switch_camera_modes:
                    if not is_first_switch:
                        time.sleep(config.capture_wait_seconds)
                    cc.cameraControlV2(config, 'SwitchNightTime')

                daytime_mode.value = False
                time_to_wait = 86400

            # Mark that the first switch has occurred
            is_first_switch = False

            # Sleep until the next switch time
            time.sleep(time_to_wait)


    except Exception as e:

        log.error('CaptureModeSwitcher thread failed with following error: ' + repr(e))


    ### For testing switching only ###

    # wait_interval = 5*60
    
    # while True:

    #     if not daytime_mode.value:
    #         log.info(f'Switching to day time mode')
    #         daytime_mode.value = True

    #         if config.switch_camera_modes:
    #             cc.cameraControlV2(config, 'SwitchDayTime')

    #     else:
    #         log.info(f'Switching to night time mode')
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
    
