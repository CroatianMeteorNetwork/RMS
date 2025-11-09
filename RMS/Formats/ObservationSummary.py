# The MIT License

# Copyright (c) 2025

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

""" Summary text and json files for station and observation session
"""

from __future__ import print_function, division, absolute_import



import os
import sys
import socket
import subprocess
import platform

import git
import shutil
import glob
import json
import re
import sqlite3
import datetime

import struct
import time
import tempfile
import ephem

from RMS.ConfigReader import parse
from RMS.Misc import niceFormat, isRaspberryPi, sanitise, getRMSStyleFileName, getRmsRootDir, UTCFromTimestamp
from RMS.Formats.FFfits import filenameToDatetimeStr
from RMS.Formats.Platepar import Platepar
from RMS.CaptureDuration import captureDuration
from RMS.CaptureModeSwitcher import SWITCH_HORIZON_DEG


if sys.version_info.major > 2:
    import dvrip as dvr
else:
    # Python2 compatible version
    import Utils.CameraControl27 as dvr

EM_RAISE = True
DEBUG_PRINT = False


def roundWithoutTrailingZero(value, no):
    """Given a float, round to specified number of decimal places, then remove trailing zeroes.

    Arguments:
        value: [float] value.
        no: [integer] number of decimal places to round.

    Return:
        string: [string]: value rounded number of decimal places without trailing zero.
    """

    value = round(value,no)
    return str("{0:g}".format(value))

def getObservationDuration(config, start_time):
    """Get the theoretical duration of the observation session for a specific night.

    Calculates sunset-to-sunrise duration using the appropriate horizon angle
    based on whether continuous capture mode is enabled.

    Arguments:
        config: [object] RMS configuration instance.
        start_time: [datetime] A time during the observation session (typically from directory name).

    Return:
        (sunset_time, duration, sunrise_time): Tuple of sunset time, duration in seconds, and sunrise time.

    """
    
    # Initialize the observer
    o = ephem.Observer()
    o.lat = str(config.latitude)
    o.long = str(config.longitude)
    o.elevation = config.elevation
    
    # Set horizon angle based on capture mode
    if config.continuous_capture:
        o.horizon = '-9'  # 9 degrees below horizon for continuous mode
    else:
        o.horizon = '-5:26'  # 5.5 degrees below horizon for nighttime-only mode
    
    o.date = start_time
    
    s = ephem.Sun()
    s.compute()
    
    try:
        # Find the sunset that preceded this time (start of the night)
        sunset_time = o.previous_setting(s).datetime()
        
        # Find the sunrise that follows the sunset (end of the night)
        o.date = sunset_time
        sunrise_time = o.next_rising(s).datetime()
        
        duration = (sunrise_time - sunset_time).total_seconds()
    except:
        # Handle polar night/day cases
        sunset_time = None
        duration = 0
        sunrise_time = None
    
    if DEBUG_PRINT:
        print("Horizon angle: {}".format(o.horizon))
        print("start_time_ephem {}".format(sunset_time))
        print("duration_ephem {:.1f} hours".format(duration / 3600 if duration else 0))
        print("end_time_ephem {}".format(sunrise_time))

    return sunset_time, duration, sunrise_time

def getTimeClient():
    """Attempt to identify which time service client, if any is providing a service.

    This function is aware of systemd-timesyncd, chronyd, ntpd.

    Return:
        name: [string] Name of the time client.
    """

    clients = {
        'systemd-timesyncd': ['systemctl', 'is-active', 'systemd-timesyncd'],
        'chronyd': ['systemctl', 'is-active', 'chronyd'],
        'ntpd': ['systemctl', 'is-active', 'ntp']
    }

    for name, cmd in clients.items():
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
            if output == 'active':
                return name
        except subprocess.CalledProcessError:
            # Not active or not recognised
            pass
    return "Not recognized"

def timeSyncStatus(config, conn, night_directory=None, force_client=None):

    """
    Add time sync information to the observation summary.

    Arguments:
        config: [Config] Configuration object.
        conn: [Connection] database connection.
        night_directory: [string] optional, name of the night directory.

    Keyword arguments:
        force_client: [string] optional, string to force resolution by ntpd, chrony, or a query on a remote server.

    Return:
        ahead_ms: [float] time local clock ahead (+ve) milliseconds, or "Unknown" if delta cannot be determined.
    """

    time_client = getTimeClient()

    if force_client is None:
        pass
    else:
        time_client = force_client

    if time_client =="ntpd":
        synchronized, uncertainty, ahead_ms = getNTPStatistics()
        addObsParam(conn, "clock_measurement_source", "ntp", night_directory=night_directory)
        addObsParam(conn, "clock_synchronized", synchronized, night_directory=night_directory)
        addObsParam(conn, "clock_ahead_ms", ahead_ms, night_directory=night_directory)
        addObsParam(conn, "clock_error_uncertainty_ms", uncertainty, night_directory=night_directory)

    elif time_client == "chronyd":
        synchronized, ahead_ms, uncertainty_ms = getChronyUncertainty()
        addObsParam(conn, "clock_measurement_source", "chrony", night_directory=night_directory)
        addObsParam(conn, "clock_synchronized", synchronized, night_directory=night_directory)
        addObsParam(conn, "clock_ahead_ms", ahead_ms, night_directory=night_directory)
        addObsParam(conn, "clock_error_uncertainty_ms", uncertainty_ms, night_directory=night_directory)

    else:
        addObsParam(conn, "clock_measurement_source", "Not detected", night_directory=night_directory)
        remote_time_query, uncertainty = timestampFromNTP()
        if remote_time_query is not None:
            local_time_query = (datetime.datetime.now(datetime.timezone.utc)
                                - datetime.datetime(1970, 1, 1)
                                        .replace(tzinfo=datetime.timezone.utc)).total_seconds()
            ahead_ms = (local_time_query - remote_time_query) * 1000
            addObsParam(conn, "clock_error_uncertainty_ms", uncertainty * 1000, night_directory=night_directory)

        else:
            ahead_ms, uncertainty = "Unknown", "Unknown"
            addObsParam(conn, "clock_error_uncertainty_ms", uncertainty, night_directory=night_directory)
        addObsParam(conn, "clock_ahead_ms", ahead_ms, night_directory=night_directory)

        result_list = subprocess.run(['timedatectl','status'], capture_output = True).stdout.splitlines()

        for raw_result in result_list:
            result = raw_result.decode('ascii')
            if "synchronized" in result:
                if result.split(":")[1].strip() == "no":
                    addObsParam(conn, "clock_synchronized", False, night_directory=night_directory)
                else:
                    addObsParam(conn, "clock_synchronized", True, night_directory=night_directory)

    return ahead_ms

def getNTPStatistics():
    """Acquire the statistics of the ntp client.

    Tries to use ntpstat, if not available, falls back to ntpq, if not available returns Unknown.

    Argyments:
        None

    Return:
        synchronized: [bool] true if reported as synchronized.
        uncertainty_ms: [float] uncertainty in milliseconds.
        time_error_ms: [str] always Unknown, unable to discern actual time error using ntp tools.
    """

    try:
        cmd = ["ntpstat"]
        lines = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip().splitlines()

        # ntpstat uses the UK spelling of synchronised.
        synchronized = False
        if lines[0].startswith("synchronised"):
            synchronized = True
        else:
            synchronized = False
        # ntpstat return milliseconds rather than base units, do not multiply 1000
        uncertainty_ms = float(lines[1].split()[4])
        return synchronized, uncertainty_ms, "Unknown"
    except:
        pass

    try:
        cmd = ["ntpq", '-p']
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        lines = output.splitlines()
        for line in lines:
            if line[0] == "*":
                fields = line.split()
                uncertainty =  float(fields[7]) + float(fields[8]) + float(fields[9])
                return "True", uncertainty, "Unknown"
    except:
        pass

    return "Unknown", "Unknown", "Unknown"

def getChronyUncertainty():
    """Acquire the statistics of the chrony ntp client.

        uncertainty implementation is taken from
        https://chrony-project.org/doc/3.3/chronyc.html

        Root dispersion

            This is the total dispersion accumulated through all the computers back to the
            stratum-1 computer from which the computer is ultimately synchronised. Dispersion is due
            to system clock resolution, statistical measurement variations, etc.

            An absolute bound on the computers clock accuracy (assuming the stratum-1 computer is correct) is given by:
            clock_error <= |system_time_offset| + root_dispersion + (0.5 * root_delay).


        Uncertainty is very high at initial synchronisation, as root dispersion dominates.

    Arguments:
        None

    Return:
        synchronized: [bool] true if reported as synchronized.
        ahead_ms: [str] time in milliseconds that computer clock is reported to be ahead of superior reference.
        uncertainty_ms: [float] uncertainty in milliseconds.
    """

    synchronized = False
    system_time_offset, root_dispersion, root_delay = 0, 0, 0
    try:
        cmd = ["chronyc", "tracking"]
        lines = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip().splitlines()
        ahead_ms = "Unknown"

        for line in lines:
            if line.startswith("Last offset"):
                system_time_offset = float(line.split(":")[1].strip().split()[0])
            if line.startswith("Root dispersion"):
                root_dispersion = float(line.split(":")[1].strip().split()[0])
            if line.startswith("Root delay"):
                root_delay = float(line.split(":")[1].strip().split()[0])
            if line.startswith("System time"):
                if "slow" in line:
                    ahead_ms = 0 - float(line.split(":")[1].strip().split()[0]) * 1000
                else:
                    ahead_ms = 0 + float(line.split(":")[1].strip().split()[0]) * 1000
            if line.startswith("Leap status"):
                if "Not synchronised" in line:
                    synchronized = False
                else:
                    synchronized = True

        if synchronized:
            uncertainty_ms = (abs(system_time_offset) + root_dispersion + (0.5 * root_delay)) * 1000
        else:
            uncertainty_ms = "Unknown"
            ahead_ms = "Unknown"

        return synchronized, ahead_ms, uncertainty_ms

    except:
        return "False", "Unknown", "Unknown"

def timestampFromNTP(addr='time.cloudflare.com'):
    """Get the timestamp from the NTP server by a direct query.

    refer https://stackoverflow.com/questions/36500197/how-to-get-time-from-an-ntp-server
    and also https://github.com/CroatianMeteorNetwork/RMS/issues/624


    Arguments:
        None

    Keyword arguments:
        addr: optional, address of ntp server to use.

    Return:
        adjusted_time: [float] time in seconds since epoch.
        estimated_network_delay: [float] estimated network delay (average of outgoing and return legs).
    """


    REF_TIME_1970 = 2208988800  # Reference time
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.settimeout(5)
    data = b'\x1b' + 47 * b'\0'
    try:
        local_clock_transmit_timestamp = time.time()
        client.sendto(data, (addr, 123))
        data, address = client.recvfrom(1024)
        local_clock_receive_timestamp = time.time()
    except socket.timeout:
        print("NTP request timed out")
        return None, None
    except Exception as e:
        print("NTP request failed: {}".format(e))
        return None, None
    if data:

        # For NTP the fractional seconds is a 32 bit counter
        fractional_second_factor = ( 1 / 2 ** 32)

        # Unpack data
        remote_clock_time_receive_timestamp_seconds = struct.unpack('!12I', data)[8] - REF_TIME_1970
        remote_clock_time_receive_timestamp_fractional_seconds = struct.unpack('!12I', data)[9] * fractional_second_factor

        remote_clock_time_transmit_timestamp_seconds = struct.unpack('!12I', data)[10] - REF_TIME_1970
        remote_clock_time_transmit_timestamp_fractional_seconds = struct.unpack('!12I', data)[11] * fractional_second_factor

        remote_clock_time_receive_timestamp = remote_clock_time_receive_timestamp_seconds + remote_clock_time_receive_timestamp_fractional_seconds
        remote_clock_time_transmit_timestamp = remote_clock_time_transmit_timestamp_seconds + remote_clock_time_transmit_timestamp_fractional_seconds

        local_clock_measured_response_time = (local_clock_receive_timestamp - local_clock_transmit_timestamp)
        remote_clock_measured_processing_time = (remote_clock_time_transmit_timestamp - remote_clock_time_receive_timestamp)

        if DEBUG_PRINT:
            print("Rx Fractional {}, Tx fractional {}".format(remote_clock_time_receive_timestamp_fractional_seconds, remote_clock_time_transmit_timestamp_fractional_seconds))
        # Next calculation assumes that remote and local clock are running at identical rates
        estimated_network_delay = local_clock_measured_response_time - remote_clock_measured_processing_time
        if estimated_network_delay < 0:
            return None, None

        # Now calculate estimated clock offsets
        clock_offset_out_leg = remote_clock_time_receive_timestamp - local_clock_transmit_timestamp
        clock_offset_return_leg = remote_clock_time_transmit_timestamp - local_clock_receive_timestamp
        estimated_offset = (clock_offset_out_leg + clock_offset_return_leg) / 2
        adjusted_time = remote_clock_time_transmit_timestamp + estimated_offset
        return adjusted_time, estimated_network_delay
    else:
        return None, None

def getObsDBConn(config, force_delete=False):
    """Creates the Observation Summary database. Tries only once.

    Arguments:
        config: [config] config instance.

    Keyword arguments:
        force_delete: [bool] default false, if set then deletes the database before recreating.

    Return:
        conn: [connection] connection to database if success else None.

    """

    # Create the Observation Summary database
    observation_records_db_path = os.path.join(config.data_dir,"observation.db")

    if force_delete:
        os.unlink(observation_records_db_path)

    if not os.path.exists(os.path.dirname(observation_records_db_path)):
        # Handle the very rare case where this could run before any observation sessions
        # and RMS_data does not exist
        try:
            # Create the required directory
            os.makedirs(os.path.dirname(observation_records_db_path))
        except:
            return None

    try:
        conn = sqlite3.connect(observation_records_db_path)

    except:
        return None

    # Returns true if the table observation_records exists in the database
    try:
        tables = conn.cursor().execute(
            """SELECT name FROM sqlite_master WHERE type = 'table' and name = 'records';""").fetchall()

        if len(tables) > 0:
            # Check if NightDirectory column exists, if not add it (for migration)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(records)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'NightDirectory' not in columns:
                print("Migrating observation database to add NightDirectory column...")
                try:
                    conn.execute("ALTER TABLE records ADD COLUMN NightDirectory TEXT")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_night_directory ON records(NightDirectory)")
                    conn.commit()
                    print("Database migration completed successfully.")
                except Exception as e:
                    print("Warning: Could not add NightDirectory column: {}".format(e))
            
            return conn
    except:
        if EM_RAISE:
            raise
        return None

    sql_command = ""
    sql_command += "CREATE TABLE records \n"
    sql_command += "( \n"
    sql_command += "id INTEGER PRIMARY KEY AUTOINCREMENT, \n"
    sql_command += "TimeStamp TEXT NOT NULL, \n"
    sql_command += "NightDirectory TEXT, \n"
    sql_command += "Key TEXT NOT NULL, \n"
    sql_command += "Value TEXT NOT NULL \n"
    sql_command += ") \n"
    conn.execute(sql_command)
    
    # Create index on NightDirectory for faster queries
    conn.execute("CREATE INDEX idx_night_directory ON records(NightDirectory)")

    return conn

def addObsParam(conn, key, value, timestamp=None, night_directory=None):
    """Add a single key value pair into the database.

    Arguments:
        conn [connection]: the connection to the database
        key [str]: the key for the value to be added
        value [str]: the value to be added

    Keyword arguments:
        timestamp [str]: optional timestamp to use instead of CURRENT_TIMESTAMP
        night_directory [str]: optional night directory name to associate with this record

    Return:
        Nothing

    """

    sql_statement = ""
    sql_statement += "INSERT INTO records \n"
    sql_statement += "(\n"
    sql_statement += "      TimeStamp, NightDirectory, Key, Value \n"
    sql_statement += ")\n\n"

    sql_statement += "      VALUES "
    sql_statement += "      (                            \n"
    if timestamp is not None:
        if night_directory is not None:
            sql_statement += "      '{}','{}','{}','{}'   \n".format(timestamp, night_directory, key, value)
        else:
            sql_statement += "      '{}',NULL,'{}','{}'   \n".format(timestamp, key, value)
    else:
        if night_directory is not None:
            sql_statement += "      CURRENT_TIMESTAMP,'{}','{}','{}'   \n".format(night_directory, key, value)
        else:
            sql_statement += "      CURRENT_TIMESTAMP,NULL,'{}','{}'   \n".format(key, value)
    sql_statement += "      )"

    if conn is None:
        return
    else:
        try:
            cursor = conn.cursor()
            cursor.execute(sql_statement)
            conn.commit()

        except:

            if EM_RAISE:
                raise

def estimateLens(fov_h):
    """Estimate the focal length of the lens in use.

    Arguments:
        fov_h: [float] horizontal field of view.

    Feturns:
        lens_type: [str] The focal length of the lens in mm.

    """

    lens_types = ["25mm", "16mm", "8mm", "6mm", "4mm"]
    lens_fov_h = [15, 30, 45, 60, 90]
    for lens_type, fov in zip(lens_types, lens_fov_h):
        if fov_h < fov:
            return lens_type
    return None

def getEphemTimesFromCaptureDirectory(config, capture_directory):
    """Examine config file in a capture directory to determine start, duration, end.

        Reads config file to use the correct calculation for continuous capture
        or night time only.

    Arguments:
        conn: [connection] connection to database.
        obs_time: [datetime] A time before an observation session.

    Return:
        start_time: [datetime] The start time of the observation session.
        duration: [integer]  seconds The duration of the observation session.
        end_time: [datetime] The end time of the observation session.

    """

    capture_directory_full_path = os.path.join(config.data_dir, config.captured_dir, capture_directory)
    if DEBUG_PRINT:
        print("Capture directory full path: {}".format(capture_directory_full_path))
    config_file_name = getattr(config, "config_file_name", None)
    if config_file_name:
        nightly_config_filename = os.path.basename(config_file_name)
        night_config_path = os.path.join(capture_directory_full_path, nightly_config_filename)
    else:
        night_config_path = os.path.join(capture_directory_full_path, ".config")

    if not os.path.isfile(night_config_path):
        # Fall back to the full config path if the nightly file is missing.
        night_config_path = config_file_name or os.path.join(capture_directory_full_path, ".config")

    night_config = parse(night_config_path)
    if DEBUG_PRINT:
        print("Making a time from {}".format(capture_directory))
    capture_directory_start_time = filenameToDatetimeStr(os.path.basename(capture_directory))
    if DEBUG_PRINT:
        print("Capture directory start time: {}".format(capture_directory_start_time))
        print("Type is {}".format(type(capture_directory_start_time)))
    capture_directory_start_time = datetime.datetime.strptime(capture_directory_start_time, "%Y-%m-%d %H:%M:%S.%f")
    if DEBUG_PRINT:
        print("Capture directory start time: {}".format(capture_directory_start_time))
    start_time, duration, end_time = getObservationDuration(night_config, capture_directory_start_time)

    return start_time, duration, end_time

def getNextStartTime(conn, time_point, tz_naive=True):
    """Query the database to discover the next start time.

    Arguments:
        conn: [connection] connection to database.
        obs_time: [datetime] A time before an observation session.

    Return:
        result: [string] the first entry in the next observation.

    """


    sql_statement = ""
    sql_statement += "SELECT Value from records \n"
    sql_statement += "      WHERE Key = 'start_time' \n"
    sql_statement += "      AND Value > '{}'\n".format(time_point)
    sql_statement += "      ORDER BY TimeStamp asc \n"

    # print(sql_statement)
    result_list = conn.cursor().execute(sql_statement).fetchall()
    # print(result_list)
    if len(result_list) > 2:
        result = result_list[1]
        return result[0]
    else:
        result = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        return result

    return result

def gatherCameraInformation(config, attempts=6, delay=10, sock_timeout=3):
    """ Gather information about the sensor in use.
        Retry the DVRIP handshake until it works, or we exhaust attempts.

    Arguments:
        config: [config] config object.

    Keyword arguments:
        attempts: [int] optional, default 6, number of attempts to connect.
        delay: [float] optional, default 10, delay between attempts.
        sock_timeout: [float] optional, default 3, socket timeout in seconds.

    Return:
        sensor type: [string] sensor type.

    """

    ip = re.search(r'(?:\d{1,3}\.){3}\d{1,3}', config.deviceID).group()
    for _ in range(attempts):
        try:
            cam = dvr.DVRIPCam(ip, timeout=sock_timeout)
            if cam.login():
                sensor = cam.get_upgrade_info()['Hardware']
                cam.close()
                return sensor
        except (socket.timeout, OSError, ConnectionError):
            # Camera may still be rebooting - ignore and retry
            pass
        time.sleep(delay)

    return "Unavailable"

def captureDirectories(captured_dir, stationID):
    """Counts the captured directories.

    Arguments:
        captured_dir: [path] to the captured directories.
        stationID: [str] stationID to identify only relevant directories.
.
    Return:
        capture_directories: [int] count of directories.

        """

    capture_directories = 0
    if not os.path.exists(captured_dir):
        return 0

    if len(os.listdir(captured_dir)) < 1:
        return 0

    for item in os.listdir(captured_dir):
        if item.startswith(stationID) and os.path.isdir(os.path.join(captured_dir, item)):
            capture_directories += 1

    return capture_directories

def nightSummaryData(config, night_data_dir):
    """ Calculate the summary data for the night.

    This is based on work by others and translated from the original source code.

    Arguments:
        config: [config] RMS config instance.
        night_data_dir: [path] the directory of captured files.


    Return:
        capture_duration_from_fits: [int] the duration from the start of first fits to the end of the last.
        capture_duration_from_ephemeris: [int] the duration from the start of first fits to the end of the last.
        fits_count: [int] the count of *.fits files in the directory.
        fits_file_shortfall: [int] the number of expected fits expected less the number actually found.
        fits_file_shortfall_ephemeris: [int] the number of expected fits expected less the number actually found,
                                             from the ephemeris computed duration
        fits_file_shortfall_as_time: [int] this shortfall expressed in seconds, never negative.
        fits_file_shortfall_as_time_ephemeris: [int] this shortfall expressed in seconds, never negative,
                                            from the ephemeris computed duration.
        time_first_fits_file: [str] the time of the first fits file.
        time_last_fits_file: [str] the time of the last fits file.
        total_expected_fits: [int] the number of fits files expected.
        total_expected_fits_ephermeris: [int] the number of fits files expected from the
                                                ephemeris computed duration
    """

    duration_one_fits_file = 256 / config.fps
    fits_files_list = glob.glob(os.path.join(night_data_dir, "*.fits"))
    fits_files_list.sort()
    fits_count = len(fits_files_list)
    if fits_count < 1:
        return 0,0,0,0,0,0,0,0,0,0,0

    time_first_fits_file = datetime.datetime.strptime(filenameToDatetimeStr(os.path.basename(fits_files_list[0])),
                                                      "%Y-%m-%d %H:%M:%S.%f")
    time_last_fits_file = datetime.datetime.strptime(filenameToDatetimeStr(
        os.path.basename(fits_files_list[-1])), "%Y-%m-%d %H:%M:%S.%f")

    # Compute key values using the first and last fits files to mark the start and end of observations
    capture_duration_from_fits = (time_last_fits_file - time_first_fits_file).total_seconds() + duration_one_fits_file
    total_expected_fits = round(capture_duration_from_fits / duration_one_fits_file)
    fits_file_shortfall = total_expected_fits - fits_count
    fits_file_shortfall = 0 if fits_file_shortfall < 1 else fits_file_shortfall
    fits_file_shortfall_as_time = str(datetime.timedelta(seconds=fits_file_shortfall * duration_one_fits_file))

    # Compute key values from the ephemeris values

    start_ephem, duration_ephem, end_ephem = getObservationDuration(config, time_first_fits_file)
    
    # Account for normal startup delay between sunset and first FITS file
    # This is the typical time needed for camera initialization, buffer allocation, etc.
    normal_startup_delay = 9  # seconds for normal mode
    if hasattr(config, 'switch_camera_modes') and config.switch_camera_modes:
        normal_startup_delay = 30  # seconds for camera mode switching
    duration_ephem = max(0, duration_ephem - normal_startup_delay)
    
    # Also account for any programmed capture delay (for staggered multi-camera systems)
    # This is in addition to the normal startup delay
    if hasattr(config, 'capture_wait_seconds') and config.capture_wait_seconds > 0:
        duration_ephem = max(0, duration_ephem - config.capture_wait_seconds)
    
    total_expected_fits_ephemeris = round(duration_ephem / duration_one_fits_file)
    fits_file_shortfall_ephemeris = total_expected_fits_ephemeris - fits_count
    fits_file_shortfall_ephemeris = 0 if fits_file_shortfall_ephemeris < 1 else fits_file_shortfall_ephemeris
    fits_file_shortfall_as_time_ephemeris = str(datetime.timedelta(seconds=fits_file_shortfall_ephemeris * duration_one_fits_file))


    return  capture_duration_from_fits, start_ephem, duration_ephem, end_ephem, \
            fits_count, \
            fits_file_shortfall, fits_file_shortfall_ephemeris, \
            fits_file_shortfall_as_time, fits_file_shortfall_as_time_ephemeris, \
            time_first_fits_file, time_last_fits_file, total_expected_fits, total_expected_fits_ephemeris



def getCommit(repo):
    """Get the most recent commit from the local repository's active branch.

    Arguments:
        repo: [path] file location of a repository.

    Return:
        commit: [string] latest commit hash
    """

    commit = subprocess.check_output(["git", "log", "-n 1", "--pretty=format:%H"], cwd=repo).decode(
        "utf-8")

    return commit

def getDateOfCommit(repo, commit):
    """Get the date of a commit

    Arguments:
        repo: [path] directory of repository.
        commit: [string] commit hash.

    Return:
        commit_time : [datetime object] python datetime object of the time and date of that commit
    """

    if commit is None:
        return datetime.datetime.strptime("2000-01-01 00:00:00 +00:00", "%Y-%m-%d %H:%M:%S %z")
    commit_date  = subprocess.check_output(["git", "show", "-s", "--format=%ci", commit], cwd=repo).decode('utf8').replace("\n","")
    return datetime.datetime.strptime(commit_date, "%Y-%m-%d %H:%M:%S %z")

def getRemoteUrls(repo):
    """Get the urls of the remotes for the local repository.
    Arguments:
        repo: directory of repository.

    Return:
        list of [remote, url] where remote is the local name of a remote and URL is the URL of the remote
    """

    urls_and_remotes = subprocess.check_output(["git", "remote", "-v"], cwd=repo).decode("utf-8").split("\n")
    url_remote_list_to_return = []
    for url_and_remote in urls_and_remotes:
        url_and_remote = url_and_remote.split("\t")
        if len(url_and_remote) == 2:
            remote, url = [url_and_remote[0], url_and_remote[1]]
            url = url.split(" ")[0]
            if not [remote, url] in url_remote_list_to_return:
                url_remote_list_to_return.append([remote, url])
    return url_remote_list_to_return

def _get_local_head_commit_and_date(repo_path):
    """Get the local HEAD commit hash and date."""
    # Local HEAD commit hash
    local_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_path
    ).decode().strip()

    # Local HEAD committer date (timezone-aware ISO 8601)
    local_date_str = subprocess.check_output(
        ["git", "log", "-1", "--pretty=%cI", "HEAD"], cwd=repo_path
    ).decode().strip()
    local_date = datetime.datetime.strptime(local_date_str, "%Y-%m-%dT%H:%M:%S%z")
    return local_commit, local_date

def _get_upstream_ref(repo_path):
    """Get the upstream tracking branch reference."""
    # e.g. "origin/alpha1"
    upstream = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        cwd=repo_path, stderr=subprocess.STDOUT
    ).decode().strip()
    return upstream

def _get_remote_url(repo_path, remote_name):
    """Get the URL for a remote."""
    return subprocess.check_output(
        ["git", "remote", "get-url", remote_name], cwd=repo_path
    ).decode().strip()

def daysBehind():
    """Measure how far behind the latest commit on the active branch is behind a branch with that commit on the remote
    repository.

    Arguments:
        syscon: [config] RMS config object.

    Return:
        number of days behind the latest remote commit that the latest local commit is on the active branch.
    """
    repo_path = os.getcwd()

    try:
        local_commit, local_date = _get_local_head_commit_and_date(repo_path)
        upstream_ref = _get_upstream_ref(repo_path)  # e.g. origin/alpha1

        remote_name, branch_name = upstream_ref.split("/", 1)
        remote_url = _get_remote_url(repo_path, remote_name)

        # Temp repo to avoid mutating the working repo
        tmpdir_obj = tempfile.TemporaryDirectory()
        tmp = tmpdir_obj.name
        
        # Create minimal temp repo and fetch only the branch tip at depth 1
        subprocess.check_call(["git", "init", "-q"], cwd=tmp, 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.check_call(["git", "remote", "add", remote_name, remote_url], cwd=tmp,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Fetch only the upstream branch tip (depth=1) with timeout
        try:
            subprocess.check_call(["git", "fetch", "-q", "--depth=1", remote_name, branch_name],
                                cwd=tmp, timeout=60,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            tmpdir_obj.cleanup()
            return "Unable to determine - timeout", "Unable to determine"

        # Remote tip committer date
        remote_date_str = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%cI", f"{remote_name}/{branch_name}"],
            cwd=tmp
        ).decode().strip()
        remote_date = datetime.datetime.strptime(remote_date_str, "%Y-%m-%dT%H:%M:%S%z")

        # Compute "days behind" (0 if local is newer/ahead)
        delta_days = (remote_date - local_date).total_seconds() / (60*60*24)
        days_behind = max(0.0, delta_days)

        tmpdir_obj.cleanup()
        return days_behind, upstream_ref
        
    except subprocess.CalledProcessError as e:
        return "Unable to determine - git error", "Unable to determine"
    except Exception as e:
        return f"Unable to determine - {str(e)}", "Unable to determine"

def retrieveObservationData(conn, config, night_directory=None, ordering=None):
    """ Query the database to get the data more recent than the time passed in.

        Usually this will be the start of the most recent observation session.
        If no ordering is passed, then a default ordering is returned.

    Arguments:
            conn:  [object] connection to database.

    Keyword arguments:
            night_directory: [str] optional, directory of night directory if none, assume most recent
            ordering: [list] optional, default None, sequence to order the keys.

    return:
            key value pairs committed to the database since the obs_start_time.
    """

    if night_directory is None:
        captured_data_dir = os.path.join(config.data_dir, config.captured_dir)
        night_dir_list = os.listdir(captured_data_dir)
        night_dir_list.sort(reverse=True)

        for night_directory in night_dir_list:
            if night_directory.startswith(config.stationID) and os.path.isdir(os.path.join(captured_data_dir, night_directory)):
                break
    
    # Extract just the directory name if a full path was provided
    night_directory = os.path.basename(night_directory)

    # print("Night directory was {}".format(night_directory))

    if ordering is None:
        # Be sure to add a comma after each list entry, IDE will not pick up this error as Python will concatenate
        # the two items into one.

        ordering = ['stationID',
                    'commit_date', 'commit_hash', 'remote_branch', 'repository_lag_remote_days',
                    'media_backend','star_catalog_file',
                    'hardware_version',
                    'captured_directories',
                    'storage_used_gb', 'storage_free_gb', 'storage_total_gb',
                    'camera_lens','camera_fov_h','camera_fov_v',
                    'camera_pointing_alt','camera_pointing_az',
                    'camera_information',
                    'clock_measurement_source', 'clock_synchronized', 'clock_ahead_ms', 'clock_error_uncertainty_ms',
                    'start_time', 'duration_from_start_of_observation', 'continuous_capture',
                    'photometry_good', 'star_catalog_file',
                    'time_start_ephem', 'time_first_fits_file',
                    'time_end_ephem', 'time_last_fits_file',
                    'total_expected_fits','total_fits',
                    'fits_files_from_duration','fits_file_shortfall', 'fits_file_shortfall_as_time',
                    'capture_duration_from_fits',
                    'capture_duration_from_ephemeris', 'total_expected_fits_ephemeris', 'fits_file_shortfall_ephemeris',
                    'fits_file_shortfall_as_time_ephemeris',
                    'detections_after_ml',
                    'media_backend','protocol_in_use','jitter_quality','dropped_frame_rate']

    # Use this print call to check the ordering
    # print("Ordering {}".format(ordering))

    sql_statement = ""
    sql_statement += "SELECT Key, Value from records \n"
    # Handle both new records with NightDirectory and old records without it
    if night_directory:
        sql_statement += "           WHERE NightDirectory = '{}' \n".format(night_directory)
        sql_statement += "              OR (NightDirectory IS NULL \n"
        # For old records, fall back to time-based filtering
        obs_start_time, obs_duration, obs_end_time = getEphemTimesFromCaptureDirectory(config, night_directory)
        if obs_start_time and obs_end_time:
            # Use a 2-hour buffer before start time to catch records added at capture start
            query_start_time = obs_start_time - datetime.timedelta(hours=2)
            next_start_time = getNextStartTime(conn, obs_end_time)
            sql_statement += "                  AND TimeStamp >= '{}' \n".format(query_start_time)
            sql_statement += "                  AND TimeStamp <= '{}') \n".format(next_start_time)
        else:
            sql_statement += "                  ) \n"
    else:
        # If no night_directory specified, use time-based query (backward compatibility)
        obs_start_time, obs_duration, obs_end_time = getEphemTimesFromCaptureDirectory(config, night_directory)
        if obs_start_time and obs_end_time:
            query_start_time = obs_start_time - datetime.timedelta(hours=2)
            next_start_time = getNextStartTime(conn, obs_end_time)
            sql_statement += "           WHERE TimeStamp >= '{}' \n".format(query_start_time)
            sql_statement += "           AND   TimeStamp <= '{}' \n".format(next_start_time)
    sql_statement += "           GROUP BY KEY \n"
    sql_statement += "           ORDER BY \n"
    sql_statement += "              CASE Key \n"

    # This SQL applies an ordering to all the keys in the ordering list. Any extra keys will be at the end.
    count = 1
    for ordering_key in ordering:
        sql_statement += "                  WHEN '{:s}' THEN {:03d} \n".format(ordering_key,count)
        count += 1

    sql_statement += "                  ELSE {:03d} \n".format(count)
    sql_statement += "              END"

    # print(sql_statement)

    return conn.cursor().execute(sql_statement).fetchall()

def serialize(config, format_nicely=True, as_json=False, night_directory = None):
    """ Returns the data from the most recent observation session as either colon
        delimited text file, ar as a json.

    Arguments:
        config: [config] station config file.
        format_nicely: [bool] optional, default true, present the data with delimiter characters aligned.
        as_json: [bool] optional, default false, return the data as a json.

    Return:
        string of key value pairs committed to the database since the start of the previous observation session.
    """

    conn = getObsDBConn(config)
    data = retrieveObservationData(conn, config, night_directory)
    conn.close()

    if as_json:
        return json.dumps(dict(data), default=lambda o: o.__dict__, indent=4, sort_keys=True)

    output = ""
    for key,value in data:
        # Does this look like a float
        if not re.match(r'^-?\d+(?:\.\d+)$', value) is None:
            # Handle as float
            try:
                value_as_float = float(value)
                output += "{}:{:s} \n".format(key, roundWithoutTrailingZero(value_as_float, 3))
            except:
                pass
        else:
            try:
                # Convert to a time
                time_object = time.strptime(value, "%Y-%m-%d %H:%M:%S.%f")
                value_as_time = time.strftime("%Y-%m-%d %H:%M:%S", time_object)
                output += "{}:{:s} \n".format(key, value_as_time)

            except:
                try:
                # Convert to a time
                    time_object = time.strptime(value, "%H:%M:%S.%f")
                    value_as_time = time.strftime("%H:%M:%S", time_object)
                    output += "{}:{:s} \n".format(key, value_as_time)
                    # if it didn't work, then handle as a string
                except:
                    pass
                    try:
                        output += "{}:{:s} \n".format(key, value)
                    except:
                        # If we can't output as a string, then move on
                        pass

    if format_nicely:
        return niceFormat(output)


    return output

def writeToFile(config, file_path_and_name, night_dir):

    """Write colon delimited text to file.

    Arguments:
        config: [config] station config file.
        file_path_and_name: [path full path to the target file.

    Return:
        [string] string of key value pairs committed to the database since the start of the observation session.
        """


    with open(file_path_and_name, "w") as summary_file_handle:
        as_ascii = serialize(config, night_directory=night_dir).encode("ascii", errors="ignore").decode("ascii")
        summary_file_handle.write(as_ascii)


def writeToJSON(config, file_path_and_name, night_dir):

    """Write as a json.
    Arguments:
        config: [config] station config file.
        file_path_and_name: [path] full path to the target file.

    Return:
        Nothing
    """

    with open(file_path_and_name, "w") as summary_file_handle:
        as_ascii = serialize(config, as_json=True, night_directory=night_dir).encode("ascii", errors="ignore").decode("ascii")
        summary_file_handle.write(as_ascii)

def startObservationSummaryReport(config, duration, night_directory=None, force_delete=False):
    """ Enters the parameters known at the start of observation into the database.

    Arguments:
        config: [config] config file.
        duration: [int]the initially calculated duration seconds.

    Keyword arguments:
        night_directory: [str] optional name of the night directory (e.g. 'US0001_20240101_123456_789012')
        force_delete: [bool] forces deletion of the observation summary database, default False.

    Return:
        [str] message about session.

    """


    conn = getObsDBConn(config, force_delete=force_delete)
    start_time_object = (datetime.datetime.now(datetime.timezone.utc) -
                         datetime.timedelta(seconds=1)).replace(tzinfo=datetime.timezone.utc)
    start_time_object_rounded = start_time_object.replace(microsecond=0)
    
    # Extract just the directory name if a full path was provided
    if night_directory:
        night_directory = os.path.basename(night_directory)
    
    addObsParam(conn, "start_time", start_time_object_rounded, night_directory=night_directory)
    addObsParam(conn, "duration_from_start_of_observation", duration, night_directory=night_directory)
    addObsParam(conn, "stationID", sanitise(config.stationID, space_substitution=""), night_directory=night_directory)

    if isRaspberryPi():
        with open('/sys/firmware/devicetree/base/model', 'r') as m:
            hardware_version = sanitise(m.read().lower(), space_substitution=" ")
    else:
        hardware_version = sanitise(platform.machine(), space_substitution=" ")

    addObsParam(conn, "hardware_version", hardware_version, night_directory=night_directory)

    try:
        repo_path = getRmsRootDir()
        repo = git.Repo(repo_path)
        if repo:
            addObsParam(conn, "commit_date",
                        UTCFromTimestamp.utcfromtimestamp(repo.head.object.committed_date).strftime('%Y%m%d_%H%M%S'),
                        night_directory=night_directory)
            addObsParam(conn, "commit_hash", repo.head.object.hexsha, night_directory=night_directory)
        else:
            print("RMS Git repository not found. Skipping Git-related information.")
    except:
        print("Error getting Git information. Skipping Git-related information.")
    
    # Get the disk usage info (only in Python 3.3+) for the data_dir disc
    if (sys.version_info.major > 2) and (sys.version_info.minor > 2):

        try:
            storage_total, storage_used, storage_free = shutil.disk_usage(config.data_dir)
            addObsParam(conn, "storage_total_gb", round(storage_total/(1024**3), 2), night_directory=night_directory)
            addObsParam(conn, "storage_used_gb", round(storage_used/(1024**3), 2), night_directory=night_directory)
            addObsParam(conn, "storage_free_gb", round(storage_free/(1024**3), 2), night_directory=night_directory)
        except:
            addObsParam(conn, "storage_total_gb", "Not available", night_directory=night_directory)
            addObsParam(conn, "storage_used_gb", "Not available", night_directory=night_directory)
            addObsParam(conn, "storage_free_gb", "Not available", night_directory=night_directory)

    captured_directories = captureDirectories(os.path.join(config.data_dir, config.captured_dir), config.stationID)
    addObsParam(conn, "captured_directories", captured_directories, night_directory=night_directory)
    try:
        addObsParam(conn, "camera_information", gatherCameraInformation(config))
    except:
        addObsParam(conn, "camera_information", "Unavailable")

    # Hardcoded for now, but should be calculated based on the config value
    no_of_frames_per_fits_file = 256

    # Calculate the number of fits files expected for the duration
    fps = config.fps


    # Testing running without this code
    """
    if duration is None:
        fits_files_from_duration = "None (Continuous Capture)"
    else:
        fits_files_from_duration = duration*fps/no_of_frames_per_fits_file
    
    addObsParam(conn, "fits_files_from_duration", fits_files_from_duration)
    """

    if not conn is None:
        conn.close()

    return "Opening a new observations summary"

def finalizeObservationSummary(config, night_data_dir, platepar=None):

    """ Enters the parameters known at the end of observation into the database.

    Arguments:
        config: [config] config file.
        night_data_dir: [path] the directory of captured files.

    Keyword arguments:
        platepar: [object] optional, default None.

    Return:
        [str] filename of text file.
        [str] filename of json.

            """

    capture_duration_from_fits, start_ephem, capture_duration_from_ephemeris, end_ephem, \
    fits_count, \
    fits_file_shortfall, fits_file_shortfall_ephemeris, \
    fits_file_shortfall_as_time, fits_file_shortfall_as_time_ephemeris, \
    time_first_fits_file, time_last_fits_file, \
    total_expected_fits, total_expected_fits_ephemeris = nightSummaryData(config, night_data_dir)

    obs_db_conn = getObsDBConn(config)
    
    # Extract just the directory name if a full path was provided
    night_dir_name = os.path.basename(night_data_dir)

    try:
        timeSyncStatus(config, obs_db_conn, night_dir_name)
    except Exception as e:
        print(repr(e))


    platepar_path = os.path.join(config.config_file_path, config.platepar_name)
    if os.path.exists(platepar_path):
        platepar = Platepar()
        platepar.read(platepar_path, use_flat=config.use_flat)
        addObsParam(obs_db_conn, "camera_pointing_az", format("{:.2f} degrees".format(platepar.az_centre)), night_directory=night_dir_name)
        addObsParam(obs_db_conn, "camera_pointing_alt", format("{:.2f} degrees".format(platepar.alt_centre)), night_directory=night_dir_name)
        addObsParam(obs_db_conn, "camera_fov_h","{:.2f}".format(platepar.fov_h), night_directory=night_dir_name)
        addObsParam(obs_db_conn, "camera_fov_v","{:.2f}".format(platepar.fov_v), night_directory=night_dir_name)
        addObsParam(obs_db_conn, "camera_lens", estimateLens(platepar.fov_h), night_directory=night_dir_name)

    addObsParam(obs_db_conn, "continuous_capture", config.continuous_capture, night_directory=night_dir_name)
    addObsParam(obs_db_conn, "time_start_ephem", start_ephem, night_directory=night_dir_name)
    addObsParam(obs_db_conn, "time_first_fits_file", time_first_fits_file, night_directory=night_dir_name)
    addObsParam(obs_db_conn, "time_end_ephem", end_ephem, night_directory=night_dir_name)
    addObsParam(obs_db_conn, "time_last_fits_file", time_last_fits_file, night_directory=night_dir_name)
    addObsParam(obs_db_conn, "capture_duration_from_fits", capture_duration_from_fits, night_directory=night_dir_name)
    addObsParam(obs_db_conn, "capture_duration_from_ephemeris", capture_duration_from_ephemeris, night_directory=night_dir_name)
    addObsParam(obs_db_conn, "total_expected_fits", round(total_expected_fits), night_directory=night_dir_name)
    addObsParam(obs_db_conn, "total_expected_fits_ephemeris", round(total_expected_fits_ephemeris), night_directory=night_dir_name)
    addObsParam(obs_db_conn, "total_fits", fits_count, night_directory=night_dir_name)
    addObsParam(obs_db_conn, "fits_file_shortfall", fits_file_shortfall, night_directory=night_dir_name)
    addObsParam(obs_db_conn, "fits_file_shortfall_ephemeris", fits_file_shortfall_ephemeris, night_directory=night_dir_name)
    addObsParam(obs_db_conn, "fits_file_shortfall_as_time", fits_file_shortfall_as_time, night_directory=night_dir_name)
    addObsParam(obs_db_conn, "fits_file_shortfall_as_time_ephemeris", fits_file_shortfall_as_time_ephemeris, night_directory=night_dir_name)
    addObsParam(obs_db_conn, "protocol_in_use", config.protocol, night_directory=night_dir_name)
    addObsParam(obs_db_conn, "star_catalog_file", config.star_catalog_file, night_directory=night_dir_name)

    try:
        days_behind, remote_branch = daysBehind()
        addObsParam(obs_db_conn, "repository_lag_remote_days", days_behind, night_directory=night_dir_name)
        addObsParam(obs_db_conn, "remote_branch", os.path.basename(remote_branch), night_directory=night_dir_name)
    except:
        addObsParam(obs_db_conn, "repository_lag_remote_days", "Not determined", night_directory=night_dir_name)
    
    obs_db_conn.close()

    writeToFile(config, getRMSStyleFileName(night_data_dir, "observation_summary.txt"), night_data_dir)
    writeToJSON(config, getRMSStyleFileName(night_data_dir, "observation_summary.json"), night_data_dir)

    return getRMSStyleFileName(night_data_dir, "observation_summary.txt"), \
                getRMSStyleFileName(night_data_dir, "observation_summary.json")

if __name__ == "__main__":

    config = parse(os.path.expanduser("~/source/RMS/.config"))

    obs_db_conn = getObsDBConn(config)

    capture_directory = os.path.join(config.data_dir, config.captured_dir)
    start_time = datetime.datetime.strptime("2025-06-25 08:03:37", "%Y-%m-%d %H:%M:%S")
    start_time, duration, end_time = getObservationDurationContinuous(config, start_time)

    dir_list = os.listdir(capture_directory)
    dir_list.sort(reverse=True)
    latest_dir = os.path.join(capture_directory, dir_list[0])
    start_time, duration, end_time = getEphemTimesFromCaptureDirectory(config, latest_dir)
    print("For directory {}".format(latest_dir))
    print("Start time was {}".format(start_time))
    print("Duration time was {:.2f} hours".format(duration/3600))
    print("End time was {}".format(end_time))



    startObservationSummaryReport(config, 100, force_delete=False)
    pp = Platepar()
    finalizeObservationSummary(config, latest_dir , pp)
    print("Summary as colon delimited text")
    print(serialize(config, as_json=False, night_directory=latest_dir))
    print("Summary as json")
    print(serialize(config, as_json=True, night_directory=latest_dir))
