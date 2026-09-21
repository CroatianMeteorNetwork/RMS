from __future__ import absolute_import, print_function

import os
import datetime

import ephem

import RMS.ConfigReader as cr
from RMS.Logger import getLogger
from RMS.Misc import RmsDateTime


log = getLogger("rmslogger")


# Sun altitude thresholds (deg, or deg:min, as ephem horizon strings). Defined here, at the bottom
# of the import graph, and imported by CaptureModeSwitcher and ObservationSummary.
SWITCH_HORIZON_DEG = "-9"  # Used for continuous capture mode switching
CAPTURE_HORIZON_DEG = "-5:26"  # Used for standard capture start/stop


def captureDuration(lat, lon, elevation, current_time=None, continuous_capture=None, sun_angle=None, \
    max_hours=23):
    """ Calculates the start time and the duration of capturing, for the given geographical coordinates
        and an optional Sun angle.
    
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
        continuous_capture: [bool] If False the Sun rise angle is set to -5:26 degrees below the horizon
            (CAPTURE_HORIZON_DEG). If True the Sun rise angle is set to -9 degrees (SWITCH_HORIZON_DEG).
            None by default, in which case the value in sun_angle is used.
        sun_angle: [str] Sun angle in deg:min below the horizon (ephem horizon string). None by default,
            which means -5:26 degrees below the horizon unless continuous_capture is given. If given,
            it takes priority over continuous_capture.

    Return:
        (start_time, duration):
            - start_time: [datetime object or bool] time when the capturing should start, or the bool
                True if capturing should start right away (already night, polar night, or no sunset
                found). Callers must test `isinstance(start_time, bool)` before doing datetime
                arithmetic on it.
            - duration: [float] seconds of capturing time
    """

    # Choose the Sun altitude threshold:
    #   - if a sun_angle is given, it always takes priority
    #   - if continuous_capture is given, the appropriate constant is chosen
    #   - if nothing is given, default to the night time only capture settings
    if sun_angle is None:
        if continuous_capture is None:
            sun_angle = CAPTURE_HORIZON_DEG
        else:
            sun_angle = SWITCH_HORIZON_DEG if continuous_capture is True else CAPTURE_HORIZON_DEG

    # Initialize the observer
    o = ephem.Observer()  
    o.lat = str(lat)
    o.long = str(lon)
    o.elevation = elevation

    # The Sun should be below the chosen altitude threshold when the capture should begin/end
    o.horizon = sun_angle

    # If the current time is not given, use the current time
    if current_time is None:
        current_time = RmsDateTime.utcnow()

    # Set the current time
    o.date = current_time

    # Calculate the location of the Sun as seen by the observer
    s = ephem.Sun()
    s.compute(o)

    # Calculate the time of next sunrise and sunset
    try:
        next_rise = o.next_rising(s).datetime()

    # If the night lasts more than 24 hours, start capturing immediately for the maximum allowed time
    except ephem.NeverUpError:
        start_time = True
        duration = 3600*max_hours
        return start_time, duration
    
    # If the day lasts more than 24 hours (polar day), the next sunset may be days or even
    # months away. The start of the capture is then at the next sunset.
    except ephem.AlwaysUpError:

        # Search in 1 hour increments until the next sunset is found. The window covers ~13 months
        # so that a full polar day at the exact poles (where day/night each last ~6 months) is
        # always spanned.
        log.info("Polar day: searching for the next sunset...")
        next_set = None
        for i in range(0, 13*30*24):

            # Increment the time by 1 hour
            o.date = o.date.datetime() + datetime.timedelta(hours=1)

            try:
                next_set = o.next_setting(s, start=o.date).datetime()
                break

            except ephem.AlwaysUpError:
                # Still polar day, keep searching forwards
                continue

            except ephem.NeverUpError:
                # The search stepped from polar day straight into polar night without pinning a
                # discrete sunset (happens at the exact pole, where the crossing is instantaneous).
                # o.date is now the start of the dark period, which may be months in the future, so
                # schedule the capture to start then rather than immediately during polar day.
                return o.date.datetime(), 3600*max_hours

        # No sunset found within the search window: capture immediately for the maximum allowed
        # time instead of crashing.
        if next_set is None:
            return True, 3600*max_hours

        # Compute the next sunrise after that sunset. If no discrete sunrise can be pinned (polar
        # transitions at extreme latitudes), fall back to the maximum allowed duration.
        try:
            next_rise = o.next_rising(s, start=next_set).datetime()
        except (ephem.AlwaysUpError, ephem.NeverUpError):
            return next_set, 3600*max_hours

        # Compute the total capture duration
        duration = (next_rise - next_set).total_seconds()

        # At very high latitudes the first night after a polar day can be very long (up to ~6
        # months at the exact pole). Cap it to the maximum allowed time, matching the polar-night
        # (NeverUpError) branch above.
        max_duration = 3600*max_hours
        if duration > max_duration:
            duration = max_duration

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

    import argparse

    arg_parser = argparse.ArgumentParser(description="Compute start time and duration for continuous capture "
                                                     "and night time only capture for the location in the passed "
                                                     "config file" , formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str,
                            help="Path to a config file which will be used instead of the default one.")

    cml_args = arg_parser.parse_args()

    # Load the config file (from the current directory if not given)
    if cml_args.config is None:
        config = cr.loadConfigFromDirectory(".config", os.getcwd())
    else:
        config = cr.loadConfigFromDirectory(cml_args.config, os.getcwd())

    # Test the time now
    start_time, duration = captureDuration(43, -81, 265)

    print("Start time: ", start_time)
    print("Duration: ", duration / 3600, " hours")

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

    # Compare the capture windows for the station location under the different Sun angle settings
    print(f"For location {config.latitude}, {config.longitude}, {config.elevation}, ")

    # Default (night time only)
    start_time, duration = captureDuration(config.latitude, config.longitude, config.elevation)
    duration = datetime.timedelta(seconds=round(duration))

    print(f"No keyword argument     Start time: {start_time} Duration: {duration}")

    # Explicit night time only capture
    start_time, duration = captureDuration(config.latitude, config.longitude, config.elevation,
                                           continuous_capture=False)
    duration = datetime.timedelta(seconds=round(duration))
    print(f"Night time capture mode Start time: {start_time} Duration: {duration}")

    # Continuous capture (switch point at -9 degrees)
    start_time, duration = captureDuration(config.latitude, config.longitude, config.elevation,
                                           continuous_capture=True)
    duration = datetime.timedelta(seconds=round(duration))
    print(f"Continuous capture mode Start time: {start_time} Duration: {duration}")

    # A lower Sun angle should give a later start and a shorter capture
    sun_angle = '-10'
    start_time, duration = captureDuration(config.latitude, config.longitude, config.elevation,
                                           sun_angle=sun_angle)
    duration = datetime.timedelta(seconds=round(duration))
    print(f"Specify a sun angle of {sun_angle} - which is lower, so should lead to a later start and shorter capture")
    print(f"                        Start time: {start_time} Duration: {duration}")

    # A higher Sun angle should give an earlier start and a longer capture
    sun_angle = '-1'
    start_time, duration = captureDuration(config.latitude, config.longitude, config.elevation,
                                           sun_angle=sun_angle)
    duration = datetime.timedelta(seconds=round(duration))
    print(f"Specify a sun angle of {sun_angle} - which is higher, so should lead to an earlier start and longer capture")
    print(f"                        Start time: {start_time} Duration: {duration}")

