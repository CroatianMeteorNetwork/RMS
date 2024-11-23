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
        next_rise = o.next_rising(s).datetime()
        next_set = o.next_setting(s).datetime()

        if next_set < next_rise:
            log.info("Next event is a sunset ({}), switching to daytime mode".format(next_set))
            daytime_mode.value = True

            if config.switch_camera_modes:
                cc.cameraControlV2(config, 'SwitchDayTime')

            time_to_wait = (next_set - current_time).total_seconds()

        else:
            log.info("Next event is a sunrise ({}), switching to nighttime mode".format(next_rise))
            daytime_mode.value = False

            if config.switch_camera_modes:
                cc.cameraControlV2(config, 'SwitchNightTime')
            
            time_to_wait = (next_rise - current_time).total_seconds()

        # Sleep until the next switch time
        time.sleep(time_to_wait)


    ### For testing ###

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

    config.latitude = 35.0572167
    config.longitude = -106.6837667
    config.elevation = 1520

    # Test the time now - remove daytime_mode
    # captureModeSwitcher(config)
    