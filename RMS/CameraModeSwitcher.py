import time
import ephem
import logging
import Utils.CameraControl as cc
from RMS.Misc import RmsDateTime

# Get the logger from the main module
log = logging.getLogger("logger")

# Function to switch between day and night modes
def cameraModeSwitcher(config, daytime_mode):
    """ Wait and switch between day and night camera modes based on current time.
    
    Arguments:
        config: [Config] config object for determining location and camera
        daytime_mode: [multiprocessing.Value] shared boolean variable to communicate mode switch with other processes
                            True = Day time, False = Night time
    """

    while True:

        # Initialize observer
        o = ephem.Observer()  
        o.lat = str(config.latitude)
        o.long = str(config.longitude)
        o.elevation = config.elevation

        # The Sun should be about 10 degrees below the horizon when the camera modes switch
        o.horizon = '-10'

        # Set the current time
        current_time = RmsDateTime.utcnow()
        o.date = current_time

        # Calculate sun positions
        s = ephem.Sun()
        s.compute()

        next_rise = o.next_rising(s).datetime()
        next_set = o.next_setting(s).datetime()

        if next_set < next_rise:
            log.info(f'Next event is a sunset ({next_set}), switching to daytime mode')
            daytime_mode.value = True
            cc.cameraControlV2(config, 'SwitchDayTime')
            time_to_wait = (next_set - current_time).total_seconds()

        else:
            log.info(f'Next event is a sunrise ({next_rise}), switching to nighttime mode')
            daytime_mode.value = False
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
    #         cc.cameraControlV2(config, 'SwitchDayTime')

    #     else:
    #         log.info(f'Switching to night time mode')
    #         daytime_mode.value = False
    #         cc.cameraControlV2(config, 'SwitchNightTime')

    #     time.sleep(wait_interval)


if __name__ == "__main__":
    
    import RMS.ConfigReader as cr
    import os

    config = cr.loadConfigFromDirectory('.', os.path.abspath('.'))

    config.latitude = 35.0572167
    config.longitude = -106.6837667
    config.elevation = 1520

    # Test the time now - remove daytime_mode
    # cameraModeSwitcher(config)
    