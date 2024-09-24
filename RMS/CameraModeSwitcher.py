import time
import ephem
import Utils.CameraControl as cc
from RMS.Misc import RmsDateTime


# Function to switch between day and night modes
def cameraModeSwitcher(config):
    """ Wait and switch between day and night camera modes based on current time.
    
    Arguments:
        config: [Config] config object for determining location and camera
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
            # Next event is a sunset, switch to daytime mode
            cc.cameraControlV2(config, 'SwitchDayTime')
            time_to_wait = (next_set - current_time).total_seconds()

        else:
            # Next event is a sunrise, switch to nighttime mode
            cc.cameraControlV2(config, 'SwitchNightTime')
            time_to_wait = (next_rise - current_time).total_seconds()

        # Sleep until the next switch time
        time.sleep(time_to_wait)



if __name__ == "__main__":
    
    import RMS.ConfigReader as cr
    import os

    config = cr.loadConfigFromDirectory('.', os.path.abspath('.'))

    config.latitude = 15.9
    config.longitude = 102.12
    config.elevation = 327

    # Test the time now
    cameraModeSwitcher(config)
    