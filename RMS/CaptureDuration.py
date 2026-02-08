from __future__ import absolute_import, print_function

import datetime

import ephem
import RMS.ConfigReader as cr
import os

from RMS.Astrometry.Conversions import trueRaDec2ApparentAltAz
from RMS.Misc import RmsDateTime


SWITCH_HORIZON_DEG = "-9"  # Used for continuous capture mode switching
CAPTURE_HORIZON_DEG = "-5:26"  # Used for standard capture start/stop (matches CaptureDuration.py)

def captureDuration(lat, lon, elevation, continuous_capture=None, sun_angle=None, current_time=None, max_hours=23):

    """ Calculates the start time and the duration of capturing, for the given geographical coordinates, and optional
    sun angle
    
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
        continuous_capture: [bool] If False the sun rise angle is set to -5:26 degress below the horizon.
                                   If True the run rise angle is set to -9 degrees
                                   If None then the value in sun_angle is used
        sun_angle: [float] Sun angle in degrees below the horizon. Default -5:26 degrees below the horizon

    
    Return:
        (start_time, duration):
            - start_time: [datetime object] time when the capturing should start, True if capturing should
                start right away
            - duration: [float] seconds of capturing time
    """

    # Handle keyword parameters
    # If a sun_angle is given, it always takes priority.
    # If a continuous_capture is given, the appropriate constant is chosen.
    # If nothing is given, then default to night time only capture settings

    if continuous_capture is None and sun_angle is None:
        sun_angle = CAPTURE_HORIZON_DEG

    elif continuous_capture is not None and sun_angle is None:
        sun_angle = SWITCH_HORIZON_DEG if continuous_capture is True else CAPTURE_HORIZON_DEG

    elif continuous_capture is None and sun_angle is not None:
        sun_angle = sun_angle

    elif continuous_capture is not None and sun_angle is not None:
        sun_angle = sun_angle

    # Initialize the observer
    o = ephem.Observer()  
    o.lat = str(lat)
    o.long = str(lon)
    o.elevation = elevation

    # The Sun should be about 5.5 degrees below the horizon when the capture should begin/end
    o.horizon = sun_angle

    # If the current time is not given, use the current time
    if current_time is None:
        current_time = RmsDateTime.utcnow()

    # Set the current time
    o.date = current_time

    # Calculate the locations of the Sun
    s = ephem.Sun(o)
    s.compute(o)

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

    import argparse

    arg_parser = argparse.ArgumentParser(description="""Compute start time and duration for continuous capture and 
                                                    night time only capture for the location in the passed config file"
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str,
                            help="Path to a config file which will be used instead of the default one.")

    cml_args = arg_parser.parse_args()

    # Load the config file
    if cml_args.config is None:
        config = cr.loadConfigFromDirectory(".config", os.getcwd())
    else:
        config = cr.loadConfigFromDirectory(cml_args.config, os.getcwd())
    # Set the web page to monitor



    print(f"For location {config.latitude}, {config.longitude}, {config.elevation}, ")

    start_time, duration = captureDuration(config.latitude, config.longitude, config.elevation)
    duration = datetime.timedelta(seconds=round(duration))

    print(f"No keyword argument     Start time: {start_time} Duration: {duration}")

    start_time, duration = captureDuration(config.latitude, config.longitude, config.elevation,
                                           continuous_capture=False)
    duration = datetime.timedelta(seconds=round(duration))
    print(f"Night time capture mode Start time: {start_time} Duration: {duration}")

    start_time, duration = captureDuration(config.latitude, config.longitude, config.elevation,
                                           continuous_capture=True)
    duration = datetime.timedelta(seconds=round(duration))
    print(f"Continuous capture mode Start time: {start_time} Duration: {duration}")

    sun_angle = '-10'
    start_time, duration = captureDuration(config.latitude, config.longitude, config.elevation,
                                           sun_angle=sun_angle)
    duration = datetime.timedelta(seconds=round(duration))
    print(f"Specify a sun angle of {sun_angle} - which is lower, so should lead to a later start and shorter capture")
    print(f"                        Start time: {start_time} Duration: {duration}")

    sun_angle = '-1'
    start_time, duration = captureDuration(config.latitude, config.longitude, config.elevation,
                                           sun_angle=sun_angle)
    duration = datetime.timedelta(seconds=round(duration))
    print(f"Specify a sun angle of {sun_angle} - which is higher, so should lead to an earlier start and longer capture")
    print(f"                        Start time: {start_time} Duration: {duration}")

