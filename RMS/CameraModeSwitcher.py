import time
import ephem
import Utils.CameraControl as cc
from RMS.Misc import RmsDateTime


# Function to switch between day and night modes
def cameraModeSwitcher(lat, lon, elevation, config):
    """ Continuously switch between day and night camera modes based on current time.
    
    Arguments:
        lat: [float] latitude +N in degrees
        lon: [float] longitude +E in degrees
        elevation: [float] elevation above sea level in meters
        config: [Config] config object for determining camera control
    """

    while True:
        # Initialize observer
        o = ephem.Observer()  
        o.lat = str(lat)
        o.long = str(lon)
        o.elevation = elevation

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

        if current_time < next_set:
            # Daytime mode
            cc.CameraControlV2(config, 'SwitchDayTime')
            time_to_wait = (next_set - current_time).total_seconds()

        else:
            # Nighttime mode
            cc.CameraControlV2(config, 'SwitchNightTime')
            time_to_wait = (next_rise - current_time).total_seconds()

        # Sleep until the next switch time
        time.sleep(time_to_wait)



if __name__ == "__main__":
    
    # Test the time now
    cameraModeSwitcher(43, -81, 265)
    