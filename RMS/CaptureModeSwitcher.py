from __future__ import print_function, division, absolute_import

import os
import json
import time
from datetime import datetime, timedelta
import ephem
import Utils.CameraControl as cc
from RMS.Logger import getLogger
from RMS.Misc import RmsDateTime

# Get the logger from the main module
log = getLogger("rmslogger")

# Sun altitude (in degrees) that defines the switch point.
# Negative numbers mean the Sun is below the horizon.
SWITCH_HORIZON_DEG = "-9"  # Used for continuous capture mode switching
CAPTURE_HORIZON_DEG = "-5:26"  # Used for standard capture start/stop (matches CaptureDuration.py)

# Buffer time (seconds) to account for capture pipeline shutdown inertia.
# Capture continues briefly after the calculated end time while the pipeline shuts down.
# This is added to the timelapse cutoff to ensure all captured frames are included.
SHUTDOWN_INERTIA_SECONDS = 300  # 5 minutes - generous buffer since mode-based splitting prevents mixing

# During polar day or night the Sun is re-checked at this interval, so the end of the polar period (the
# first real sunset or sunrise) is not missed
POLAR_RECHECK_SECONDS = 3600

# A capture session is not split if it ended less than this long before the split time, so a split never
# leaves an almost empty session (e.g. at the pole, when the mode changes shortly before a split)
MIN_SESSION_SECONDS = 12*3600

# During polar day or night a capture session is split at the latest this long after the last split (at the
# pole the culmination is not defined, and a polar day may turn into a polar night with a single switch)
MAX_POLAR_SESSION_SECONDS = 25*3600


def captureModeAt(config, when_utc):
    """ Determine the capture mode (day or night) at the given time, and when it next changes.

    The mode is decided from the current altitude of the Sun relative to the switching horizon. During
    polar day or night the Sun does not cross the switching horizon and there is no next change.

    Arguments:
        config: [Config] RMS config object, the station location is used.
        when_utc: [datetime] Naive UTC time.

    Return:
        (daytime, next_change, polar): [tuple]
            - daytime: [bool] True if the Sun is above the switching horizon (daytime mode).
            - next_change: [datetime] Naive UTC time of the next mode change, None during polar day or night.
            - polar: [str] None normally, 'day' during polar day, 'night' during polar night.
    """

    o = ephem.Observer()
    o.lat = str(config.latitude)
    o.long = str(config.longitude)
    o.elevation = config.elevation
    o.horizon = SWITCH_HORIZON_DEG
    o.date = ephem.Date(when_utc)

    s = ephem.Sun()
    s.compute(o)

    # Use the altitude of the upper limb of the Sun (with refraction), the same as ephem uses for the rising
    # and setting times, so the mode at a rising/setting time matches the event
    daytime = (float(s.alt) + float(s.radius)) > float(o.horizon)

    try:
        if daytime:
            next_change = o.next_setting(s).datetime()
        else:
            next_change = o.next_rising(s).datetime()

        return daytime, next_change, None

    # The Sun does not cross the switching horizon - the current state lasts: polar day if the Sun is up,
    # polar night if it is down. The state is taken from the altitude and not from the error type, as close
    # to the pole the Sun altitude barely changes during the day and ephem may raise either error
    except (ephem.AlwaysUpError, ephem.NeverUpError):
        return daytime, None, ('day' if daytime else 'night')


def polarSessionBoundary(config, when_utc, polar):
    """ Time of the next daily split of the capture session during polar day or night.

    In continuous capture mode, a session normally ends when the camera switches between night and day.
    During polar day or night there is no switch, so the session is split once a day: at the lower
    culmination of the Sun during polar day (local solar midnight, the Sun is at its lowest - this is
    where the night appears first when the polar day ends), and at the upper culmination during polar
    night (local solar noon, the sky is at its brightest).

    Arguments:
        config: [Config] RMS config object, the station location is used.
        when_utc: [datetime] Naive UTC time, the boundary is searched after this time.
        polar: [str] 'day' or 'night'.

    Return:
        boundary: [datetime] Naive UTC time of the next session split, within about 24 hours of when_utc.
    """

    o = ephem.Observer()
    o.lat = str(config.latitude)
    o.long = str(config.longitude)
    o.elevation = config.elevation
    o.date = ephem.Date(when_utc)

    s = ephem.Sun()

    boundary = None
    try:
        if polar == 'day':
            boundary = o.next_antitransit(s).datetime()
        else:
            boundary = o.next_transit(s).datetime()

    except Exception as e:
        log.warning("Computing the Sun culmination failed: {}".format(repr(e)))

    # Fall back to the next UTC midnight if the culmination is not defined (e.g. at the pole). The solar
    # day is not exactly 24 hours, so allow some margin
    if (boundary is None) or (boundary <= when_utc) or (boundary > when_utc + timedelta(hours=25)):
        boundary = (when_utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    return boundary


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

def switcherStep(config, state, daytime_mode, camera_mode_switch_trigger, session_cut=None,
    session_cut_time=None, now_func=RmsDateTime.utcnow, sleep_func=time.sleep):
    """ One iteration of the capture mode switcher: set the day/night mode, split the capture session if it
        would otherwise last longer than a day (polar day or night), and return how long to wait until the
        next iteration.

    A capture session normally ends when the camera switches from night to day. If the camera is still in
    daytime mode at the lower culmination of the Sun (local solar midnight), or still in nighttime mode at the
    upper culmination (local solar noon), the Sun has not crossed the switching horizon for a day and the
    session is split there. This only happens during polar day/night and on the days around them.

    Arguments:
        config: [Config] RMS config object.
        state: [dict] State kept between iterations, start with an empty dict.
        daytime_mode: [multiprocessing.Value] Shared bool, True = daytime, False = nighttime.
        camera_mode_switch_trigger: [multiprocessing.Value] Shared bool, set to True to switch the camera mode.

    Keyword arguments:
        session_cut: [multiprocessing.Value] Shared int, incremented when the capture session should be split.
            None by default, in which case the session is never split.
        session_cut_time: [multiprocessing.Value] Shared double, Unix time of the last session split. Set
            before session_cut is incremented.
        now_func: [function] Returns the current naive UTC time. Used for testing.
        sleep_func: [function] Sleeps for the given number of seconds. Used for testing.

    Return:
        time_to_wait: [float] Seconds to wait before the next iteration.
    """

    is_first_switch = state.get('is_first_switch', True)

    now = now_func()
    daytime, next_change, polar = captureModeAt(config, now)


    # Switch the mode on the first run and when it changes. During polar day or night the Sun is
    # re-checked every hour, the camera must not be switched (and rebooted) on every check
    if is_first_switch or (daytime_mode.value != daytime):

        mode_name = "daytime" if daytime else "nighttime"

        if polar is None:
            log.info("Next event is a {:s} ({}), switching to {:s} mode".format(
                "sunset" if daytime else "sunrise", next_change, mode_name))

        else:
            log.info("Polar {:s}: the Sun does not cross the switching horizon, switching to {:s} mode".format(
                polar, mode_name))

        if config.switch_camera_modes:

            # Delay before switching camera modes to prevent multiple cameras from switching
            # simultaneously (avoids power spikes). Initial run already has startup delay, so skip the wait
            if not is_first_switch:
                sleep_func(config.capture_wait_seconds)

            camera_mode_switch_trigger.value = True

        # A night -> day switch ends the capture session
        if (not is_first_switch) and daytime:
            state['last_session_end'] = now

        daytime_mode.value = daytime

        # Refresh the current time after the stagger wait
        now = now_func()

    state['is_first_switch'] = False


    # Split the session at the scheduled culmination if the mode has not changed since it was scheduled
    cut_at = state.get('cut_at')
    if (cut_at is not None) and (now >= cut_at):

        if state.get('cut_daytime') == daytime:

            last_end = state.get('last_session_end')
            if (last_end is None) or (cut_at - last_end >= timedelta(seconds=MIN_SESSION_SECONDS)):

                log.info("The Sun has not crossed the switching horizon for a day ({:s}), splitting the "
                    "capture session at {}".format("daytime" if daytime else "nighttime", cut_at))

                if session_cut is not None:

                    # Set the time first, the counter change signals that a new split is available
                    if session_cut_time is not None:
                        session_cut_time.value = (cut_at - datetime(1970, 1, 1)).total_seconds()

                    session_cut.value += 1

                state['last_session_end'] = cut_at

        cut_at = None


    # Schedule the next possible split: solar midnight in daytime, solar noon in nighttime
    if (cut_at is None) or (state.get('cut_daytime') != daytime):
        cut_at = polarSessionBoundary(config, now, 'day' if daytime else 'night')

        last_end = state.get('last_session_end')
        if (polar is not None) and (last_end is not None):
            cut_at = max(now, min(cut_at, last_end + timedelta(seconds=MAX_POLAR_SESSION_SECONDS)))

        if polar is not None:
            log.info("Polar {:s}: the capture session will be split daily, next split at {}".format(
                polar, cut_at))

    state['cut_at'] = cut_at
    state['cut_daytime'] = daytime


    # Wake up just after the next mode change or the next possible split, whichever is first
    wake_times = [cut_at]
    if next_change is not None:
        wake_times.append(next_change)

    time_to_wait = max(0, (min(wake_times) - now).total_seconds()) + 1

    # During polar day or night re-check every hour, to catch the first sunset/sunrise when it ends
    if polar is not None:
        time_to_wait = min(time_to_wait, POLAR_RECHECK_SECONDS)

    return time_to_wait


# Function to switch capture between day and night modes
def captureModeSwitcher(config, daytime_mode, camera_mode_switch_trigger, stop_event=None, session_cut=None,
    session_cut_time=None):
    """ Wait and switch between day and night capture modes based on current time.

    Arguments:
        config: [Config] config object for determining location and controlling camera settings if specified
        daytime_mode: [multiprocessing.Value] shared boolean variable to communicate the mode switch with other processes
                            True = Day time, False = Night time
        stop_event: [threading.Event] optional event to signal this thread to exit
        session_cut: [multiprocessing.Value] optional shared int, incremented when the capture session should
            be split once a day during polar day or night
        session_cut_time: [multiprocessing.Value] optional shared double, Unix time of the last session split
    """

    state = {}

    try:
        while not (stop_event is not None and stop_event.is_set()):

            time_to_wait = switcherStep(config, state, daytime_mode, camera_mode_switch_trigger,
                session_cut=session_cut, session_cut_time=session_cut_time)

            # Sleep until the next switch time (interruptible via stop_event)
            if stop_event is not None:
                stop_event.wait(time_to_wait)
            else:
                time.sleep(time_to_wait)


    except Exception as e:

        log.error('CaptureModeSwitcher thread failed with following error: ' + repr(e))



def lastNightToDaySwitch(config, whenUtc=None):
    """Return the UTC timestamp of the most recent night-to-day switch.

    Arguments:
        config: [Config] RMS configuration object; must expose latitude,
            longitude, elevation, and continuous_capture attributes.

    Keyword arguments:
        whenUtc: [datetime] Naive UTC time used as the upper bound of the
            search window. None by default, which means the current
            ``RmsDateTime.utcnow()`` is used.

    Return:
        (sunrise, max_cutoff): [tuple[datetime, datetime]] Tuple of:
            - sunrise: The actual sunrise time (sun crossing horizon threshold)
            - max_cutoff: sunrise + inertia delay (maximum possible cutoff)
        For polar regions where sun doesn't rise/set, returns (whenUtc, whenUtc).
    """
    if whenUtc is None:
        whenUtc = RmsDateTime.utcnow()

    # Use different horizon based on capture mode:
    # - Continuous capture: mode switches at -9 degrees
    # - Standard capture: capture ends at -5:26 degrees
    if config.continuous_capture:
        horizon = SWITCH_HORIZON_DEG
    else:
        horizon = CAPTURE_HORIZON_DEG

    obs = ephem.Observer()
    obs.lat = str(config.latitude)
    obs.long = str(config.longitude)
    obs.elevation = config.elevation
    obs.horizon = horizon
    obs.date = ephem.Date(whenUtc)

    sun = ephem.Sun()
    try:
        # Account for programmed delay in mode switching and capture pipeline shutdown inertia
        wait = timedelta(seconds=config.capture_wait_seconds + SHUTDOWN_INERTIA_SECONDS)
        previous_sunrise = obs.previous_rising(sun).datetime()
        return previous_sunrise, previous_sunrise + wait

    except (ephem.AlwaysUpError, ephem.NeverUpError):
        # Fallback for polar regions: use current time as cutoff
        # This ensures the entire just-completed capture session is processed
        # immediately, rather than being split across UTC day boundaries
        return whenUtc, whenUtc


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
    
