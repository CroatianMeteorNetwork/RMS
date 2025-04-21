from __future__ import absolute_import, print_function

import datetime

import ephem
from RMS.Misc import RmsDateTime


def captureDuration(lat, lon, elevation, current_time=None, max_hours=23):
    """ Calculates the start time and the duration of capturing, for the given geographical coordinates. 
    
    Arguments:
        lat: [float] latitude +N in degrees
        lon: [float] longitude +E in degrees
        elevation: [float] elevation above sea level in meters
    
    Keyword arguments:
        current_time: [datetime object] The given date and time of reference for the capture duration
            calculation. If not given, the current time is used. None by default
        max_hours: [float] Maximum number of hours of capturing time. If the calculated duration is longer
            than this, the duration is set to this value. 23 by default, to give enough time for the
            rest of the processing.
    
    Return:
        (start_time, duration):
            - start_time: [datetime object] time when the capturing should start, True if capturing should
                start right away
            - duration: [float] seconds of capturing time
    """

    # Initialize the observer
    o = ephem.Observer()  
    o.lat = str(lat)
    o.long = str(lon)
    o.elevation = elevation

    # The Sun should be about 5.5 degrees below the horizon when the capture should begin/end
    o.horizon = '-5:26'

    # If the current time is not given, use the current time
    if current_time is None:
        current_time = RmsDateTime.utcnow()

    # Set the current time
    o.date = current_time

    # Calculate the locations of the Sun
    s = ephem.Sun()  
    s.compute()

    # Calculate the time of next sunrise and sunset
    try:
        next_rise = o.next_rising(s).datetime()

    # If the night lasts more than 24 hours, start capturing immediately for the maximum allowed time
    except ephem.NeverUpError:
        start_time = True
        duration = 3600*max_hours
        return start_time, duration
    
    # If the day last more than 24, then the start of the capture is at the next sunset (which may be in days)
    except ephem.AlwaysUpError:

        # Search in 1 hour increments until the next sunset is found (search for a maximum of 6 months)
        print("Searching for the next sunset...")
        for i in range(0, 6*30*24):
            
            # Increment the time by 1 hour
            o.date = o.date.datetime() + datetime.timedelta(hours=1)

            try:
                next_set = o.next_setting(s, start=o.date).datetime()
                break

            except ephem.AlwaysUpError:
                print("Still day at ", o.date.datetime(), "...")
                pass

        # Compute the next sunrise
        next_rise = o.next_rising(s, start=next_set).datetime()

        # Compute the total capture duration
        duration = (next_rise - next_set).total_seconds()

        return next_set, duration
        


    next_set = o.next_setting(s).datetime()
    

    # If the next sunset is later than the next sunrise, it means that it is night, and capturing should start immediately
    if next_set > next_rise:

        start_time = True

    # Otherwise, start capturing after the next sunset
    else:

        start_time = next_set
        

    # Calculate how long should the capture run
    if start_time == True:
        duration = next_rise - current_time

    else:
        duration = next_rise - next_set

    # Calculate the duration of capture in seconds
    duration = duration.total_seconds()

    # If the duration is longer than the maximum allowed, set it to the maximum
    max_duration = 3600*max_hours
    if duration > max_duration:
        duration = max_duration

    return start_time, duration
        

if __name__ == "__main__":
    

    # Test the time now
    start_time, duration = captureDuration(43, -81, 265)

    # # Test the capture duration on e.g. Greenland during the winter solstice
    # start_time, duration = captureDuration(72.0, -40.0, 0, 
    #                                        current_time=datetime.datetime(2022, 12, 21, 15, 0, 0))
    
    # # # Test the capture duration on e.g. Greenland during the summer solstice
    # start_time, duration = captureDuration(72.0, -40.0, 0,
    #                                          current_time=datetime.datetime(2022, 6, 21, 15, 0, 0))

    # # Test the capture duration on the South Pole during the summer solstice
    # start_time, duration = captureDuration(-89.0, 0.0, 0,
    #                                          current_time=datetime.datetime(2022, 6, 21, 0, 0, 0))

    # # Test the capture duration on the South Pole during the winter solstice
    # start_time, duration = captureDuration(-89.0, 0.0, 0,
    #                                          current_time=datetime.datetime(2022, 12, 21, 0, 0, 0))
    

    
    print("Start time: ", start_time)
    print("Duration: ", duration/3600, " hours")
