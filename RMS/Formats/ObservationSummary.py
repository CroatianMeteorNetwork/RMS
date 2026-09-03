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
from RMS.Misc import niceFormat, isRaspberryPi, sanitise, getRMSStyleFileName, getRmsRootDir, \
    UTCFromTimestamp
from RMS.Misc import runGitCommand, GitCommandTimeout, hostReachable, gitUrlHost, \
    gitUrlHostAndPort, sanitiseGitUrl, anonymousHttpsGitUrl, GIT_ALLOWED_HOSTS, \
    GIT_NETWORK_TIMEOUT, GIT_LOCAL_TIMEOUT
from RMS.Logger import getLogger
from RMS.Formats.FFfits import filenameToDatetimeStr
from RMS.Formats.Platepar import Platepar
from RMS.CaptureDuration import captureDuration
from RMS.CaptureModeSwitcher import SWITCH_HORIZON_DEG


# Get the logger from the main module
log = getLogger("rmslogger")


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

def getObservationDurationNightTime(config, start_time):
    """Get the duration of an observation session not in continuous capture mode.


    Arguments:
        conn: [object] database connection instance.
        config: [object] RMS configuration instance.

    Return:
        duration: [float] duration of observation in seconds.
    """

    _, duration = captureDuration(config.latitude, config.longitude, config.elevation,start_time)

    end_time = start_time + datetime.timedelta(seconds=duration)

    return start_time, duration, end_time

def getObservationDurationContinuous(config, start_time):
    """Get the duration of an observation session in continuous capture mode.

        o.date is initialised to the start time of the observation session, rather
        than an arbitrary time during the previous capture session.

        Arguments:
            config: [object] RMS configuration instance.
            start_time: [object] time within, but near to the start of the observation session

        Return:
            duration: [float] duration of observation in seconds. If cannot be computed, return 0.
        """

    # convert start_time to a python object
    if DEBUG_PRINT:
        print("Passed a start time of {}".format(start_time))

    # Initialize sun and observer
    o = ephem.Observer()
    o.lat, o.long, o.elevation  = str(config.latitude), str(config.longitude), config.elevation
    s, o.horizon, o.date = ephem.Sun(), SWITCH_HORIZON_DEG, start_time

    # Is this start time during night time capture hours
    s.compute()
    while o.next_setting(s).datetime() < o.next_rising(s).datetime():
        if DEBUG_PRINT:
            print("{} is not at night time".format(start_time))
        start_time +=datetime.timedelta(minutes=1)
        o.date = start_time
        s.compute()
    if DEBUG_PRINT:
        print("Advanced time to {}".format(o.date))

    # Compute duration
    try:
        s.compute()

        start_time_ephem = o.previous_setting(s).datetime()
        end_time_ephem = o.next_rising(s).datetime()
        duration_ephem = (end_time_ephem - start_time_ephem).total_seconds()
    except:
        start_time_ephem = None
        duration_ephem = 0
        end_time_ephem = None

    if DEBUG_PRINT:
        print("start_time_ephem {}".format(start_time_ephem))
        print("duration_ephem {:.1f} hours".format(duration_ephem/3600))
        print("end_time_ephem {}".format(end_time_ephem))

    return start_time_ephem, duration_ephem, end_time_ephem

def getObservationDuration(config, start_time):
    """Get the duration of the observation session.

    Capture can operate in two modes. Continuous capture, where the capture runs all day,
    and nighttime only mode. The duration of the observation sessions is computed in a
    slightly different way in these two cases. This function calls the correct function
    to compute the duration of the observation session, based on the RMS configuration
    instance.

    Arguments:
        config: [object] RMS configuration instance.
        start_time: [object] A time during the observation session.

    Return:
        duration: [int] duration of the observation session in seconds.

    """

    if config.continuous_capture:
        start_time_ephem, duration_ephem, end_time_ephem = getObservationDurationContinuous(config, start_time)
    else:
        start_time_ephem, duration_ephem, end_time_ephem = getObservationDurationNightTime(config, start_time)

    return start_time_ephem, duration_ephem, end_time_ephem

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

def timeSyncStatus(config, conn, force_client=None):

    """
    Add time sync information to the observation summary.

    Arguments:
        config: [Config] Configuration object.
        conn: [Connection] database connection.

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
        addObsParam(conn, "clock_measurement_source", "ntp")
        addObsParam(conn, "clock_synchronized", synchronized)
        addObsParam(conn, "clock_ahead_ms", ahead_ms)
        addObsParam(conn, "clock_error_uncertainty_ms", uncertainty)

    elif time_client == "chronyd":
        synchronized, ahead_ms, uncertainty_ms = getChronyUncertainty()
        addObsParam(conn, "clock_measurement_source", "chrony")
        addObsParam(conn, "clock_synchronized", synchronized)
        addObsParam(conn, "clock_ahead_ms", ahead_ms)
        addObsParam(conn, "clock_error_uncertainty_ms", uncertainty_ms)

    else:
        addObsParam(conn, "clock_measurement_source", "Not detected")
        remote_time_query, uncertainty = timestampFromNTP()
        if remote_time_query is not None:
            local_time_query = (datetime.datetime.now(datetime.timezone.utc)
                                - datetime.datetime(1970, 1, 1)
                                        .replace(tzinfo=datetime.timezone.utc)).total_seconds()
            ahead_ms = (local_time_query - remote_time_query) * 1000
            addObsParam(conn, "clock_error_uncertainty_ms", uncertainty * 1000)

        else:
            ahead_ms, uncertainty = "Unknown", "Unknown"
            addObsParam(conn, "clock_error_uncertainty_ms", uncertainty)
        addObsParam(conn, "clock_ahead_ms", ahead_ms)

        result_list = subprocess.run(['timedatectl','status'], capture_output = True).stdout.splitlines()

        for raw_result in result_list:
            result = raw_result.decode('ascii')
            if "synchronized" in result:
                conn = getObsDBConn(config)
                if result.split(":")[1].strip() == "no":
                    addObsParam(conn, "clock_synchronized", False)
                else:
                    addObsParam(conn, "clock_synchronized", True)

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
        fractional_second_factor = ( 1/2 ** 32)

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
        estimated_offset = (clock_offset_out_leg + clock_offset_return_leg)/2
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
    sql_command += "Key TEXT NOT NULL, \n"
    sql_command += "Value TEXT NOT NULL \n"
    sql_command += ") \n"
    conn.execute(sql_command)

    return conn

def addObsParam(conn, key, value):
    """Add a single key value pair into the database.

    Arguments:
        conn [connection]: the connection to the database
        key [str]: the key for the value to be added
        value [str]: the value to be added

    Return:
        Nothing

    """

    sql_statement = ""
    sql_statement += "INSERT INTO records \n"
    sql_statement += "(\n"
    sql_statement += "      TimeStamp, Key, Value \n"
    sql_statement += ")\n\n"

    sql_statement += "      VALUES "
    sql_statement += "      (                            \n"
    sql_statement += "      CURRENT_TIMESTAMP,'{}','{}'   \n".format(key, value)
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

    duration_one_fits_file = 256/config.fps
    fits_files_list = glob.glob(os.path.join(night_data_dir, "*.fits"))
    fits_files_list.sort()
    fits_count = len(fits_files_list)
    if fits_count < 1:
        return 0,0,0,0,0,0,0,0,0,0,0,0,0

    time_first_fits_file = datetime.datetime.strptime(filenameToDatetimeStr(os.path.basename(fits_files_list[0])),
                                                      "%Y-%m-%d %H:%M:%S.%f")
    time_last_fits_file = datetime.datetime.strptime(filenameToDatetimeStr(
        os.path.basename(fits_files_list[-1])), "%Y-%m-%d %H:%M:%S.%f")

    # Compute key values using the first and last fits files to mark the start and end of observations
    capture_duration_from_fits = (time_last_fits_file - time_first_fits_file).total_seconds() + duration_one_fits_file
    total_expected_fits = round(capture_duration_from_fits/duration_one_fits_file)
    fits_file_shortfall = total_expected_fits - fits_count
    fits_file_shortfall = 0 if fits_file_shortfall < 1 else fits_file_shortfall
    fits_file_shortfall_as_time = str(datetime.timedelta(seconds=fits_file_shortfall * duration_one_fits_file))

    # Compute key values from the ephemeris values

    start_ephem, duration_ephem, end_ephem = getObservationDuration(config, time_first_fits_file)
    total_expected_fits_ephemeris = round(duration_ephem/duration_one_fits_file)
    fits_file_shortfall_ephemeris = total_expected_fits_ephemeris - fits_count
    fits_file_shortfall_ephemeris = 0 if fits_file_shortfall_ephemeris < 1 else fits_file_shortfall_ephemeris
    fits_file_shortfall_as_time_ephemeris = str(datetime.timedelta(seconds=fits_file_shortfall_ephemeris * duration_one_fits_file))


    return  capture_duration_from_fits, start_ephem, duration_ephem, end_ephem, \
            fits_count, \
            fits_file_shortfall, fits_file_shortfall_ephemeris, \
            fits_file_shortfall_as_time, fits_file_shortfall_as_time_ephemeris, \
            time_first_fits_file, time_last_fits_file, total_expected_fits, total_expected_fits_ephemeris


def filterCloneableRemotes(remote_list, allowed_hosts=GIT_ALLOWED_HOSTS):

    """ Keep only the remotes which can be cloned anonymously, i.e. without any chance of a
        credential prompt.

        Only remotes on known hosts and without embedded credentials are kept, as a remote on an
        unknown host may well be a private fork which would ask for credentials. An ssh remote on
        an allowed host is rewritten to its anonymous https equivalent, so that checkouts which
        were cloned over ssh are still covered, without ever needing a key or a passphrase.

    Arguments:
        remote_list: [list] list of [remote_name, url] pairs, as returned by getRemoteUrls.

    Keyword arguments:
        allowed_hosts: [tuple of str] host names from which cloning is allowed.

    Return:
        [list] list of [remote_name, url] pairs with anonymous https URLs.
    """

    allowed_hosts_lower = [host.lower() for host in allowed_hosts]

    cloneable_remotes = []

    for remote_name, url in remote_list:

        host = gitUrlHost(url)

        if (host is None) or (host.lower() not in allowed_hosts_lower):
            log.debug("Skipping git remote {:s}, the host {:s} is not in the list of allowed hosts"
                      .format(remote_name, str(host)))
            continue

        # Rewrite the URL so that it can be cloned without any credentials
        https_url = anonymousHttpsGitUrl(url)

        if https_url is None:
            log.debug("Skipping git remote {:s}, its URL cannot be used anonymously".format(remote_name))
            continue

        if [remote_name, https_url] not in cloneable_remotes:
            cloneable_remotes.append([remote_name, https_url])

    return cloneable_remotes


def updateCommitHistoryDirectory(remote_urls, target_directory, timeout=GIT_NETWORK_TIMEOUT):

    """ Clone only the commit history of a remote repository.

        Every git operation runs with the interactive prompts disabled and with a hard timeout, so
        that an unhealthy remote can never block the processing chain.

    Arguments:
        remote_urls: [list] list of [remote_name, url] pairs to be cloned/fetched.
        target_directory: [path] the directory into which to clone.

    Keyword arguments:
        timeout: [float] hard timeout of every git operation which touches the network, in seconds.

    Return:
        commit_repo_directory: [path] directory of the repository
    """

    # Keep only the remotes which can be cloned without any chance of a credential prompt
    remote_urls = filterCloneableRemotes(remote_urls)

    if len(remote_urls) == 0:
        raise RuntimeError("No anonymously cloneable git remote was found")

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Clone into a directory named here, so that the location of the clone does not have to be
    # guessed from a directory listing afterwards
    clone_directory = os.path.join(target_directory, "commit_history")

    if os.path.exists(clone_directory):
        shutil.rmtree(clone_directory)

    commit_repo_directory = None

    for local_name, url in remote_urls:

        host, port = gitUrlHostAndPort(url)

        # Fail fast if the host cannot even be reached, instead of waiting out the git timeout
        if not hostReachable(host, port):
            log.warning("Skipping git remote {:s}, the host {:s} is not reachable".format(
                local_name, str(host)))
            continue

        try:

            if commit_repo_directory is None:

                # Clone the commit history of the first reachable remote
                returncode, _, stderr = runGitCommand(
                    ["clone", url, "--filter=blob:none", "--no-checkout", clone_directory],
                    timeout=timeout, network=True)

                if returncode != 0:
                    log.warning("Cloning the commit history from {:s} failed with code {:d}: {:s}"
                                .format(sanitiseGitUrl(url), returncode, stderr.strip()[:500]))
                    continue

                commit_repo_directory = clone_directory

                # This first remote might have been pulled in with the wrong local name, so rename it
                returncode, stdout, _ = runGitCommand(["remote"], cwd=commit_repo_directory,
                                                      timeout=GIT_LOCAL_TIMEOUT)
                downloaded_remote_name = stdout.strip()

                if (returncode == 0) and (len(downloaded_remote_name) > 0) \
                        and (downloaded_remote_name != local_name):

                    runGitCommand(["remote", "rename", downloaded_remote_name, local_name],
                                  cwd=commit_repo_directory, timeout=GIT_LOCAL_TIMEOUT)

            else:

                # The history is already there, so only add the additional remote and fetch it. A
                # failure here is not fatal, as the lag can still be measured against the remote
                # which was cloned first.
                returncode, _, stderr = runGitCommand(["remote", "add", local_name, url],
                                                      cwd=commit_repo_directory,
                                                      timeout=GIT_LOCAL_TIMEOUT)

                if returncode != 0:
                    log.warning("Adding the git remote {:s} failed: {:s}".format(
                        local_name, stderr.strip()[:500]))
                    continue

                returncode, _, stderr = runGitCommand(["fetch", "--filter=blob:none", local_name],
                                                      cwd=commit_repo_directory, timeout=timeout,
                                                      network=True)

                if returncode != 0:
                    log.warning("Fetching the git remote {:s} failed: {:s}".format(
                        local_name, stderr.strip()[:500]))

        except GitCommandTimeout as e:
            log.warning("Git operation on the remote {:s} timed out: {:s}".format(local_name, repr(e)))
            continue

    if commit_repo_directory is None:
        raise RuntimeError("Could not clone the commit history from any git remote")

    return commit_repo_directory

def getCommit(repo):
    """Get the most recent commit from the local repository's active branch.

    Arguments:
        repo: [path] file location of a repository.

    Return:
        commit: [string] latest commit hash
    """

    returncode, stdout, stderr = runGitCommand(["log", "-n 1", "--pretty=format:%H"], cwd=repo,
                                               timeout=GIT_LOCAL_TIMEOUT)

    if returncode != 0:
        raise RuntimeError("Could not read the latest local commit: " + stderr.strip()[:500])

    return stdout

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

    returncode, stdout, stderr = runGitCommand(["show", "-s", "--format=%ci", commit], cwd=repo,
                                               timeout=GIT_LOCAL_TIMEOUT)

    if returncode != 0:
        raise RuntimeError("Could not read the date of the commit {:s}: {:s}".format(
            str(commit), stderr.strip()[:500]))

    commit_date = stdout.replace("\n", "")

    return datetime.datetime.strptime(commit_date, "%Y-%m-%d %H:%M:%S %z")

def getRemoteUrls(repo):
    """Get the urls of the remotes for the local repository.
    Arguments:
        repo: directory of repository.

    Return:
        list of [remote, url] where remote is the local name of a remote and URL is the URL of the remote
    """

    returncode, stdout, stderr = runGitCommand(["remote", "-v"], cwd=repo, timeout=GIT_LOCAL_TIMEOUT)

    if returncode != 0:
        raise RuntimeError("Could not read the git remotes: " + stderr.strip()[:500])

    urls_and_remotes = stdout.split("\n")
    url_remote_list_to_return = []
    for url_and_remote in urls_and_remotes:
        url_and_remote = url_and_remote.split("\t")
        if len(url_and_remote) == 2:
            remote, url = [url_and_remote[0], url_and_remote[1]]
            url = url.split(" ")[0]
            if not [remote, url] in url_remote_list_to_return:
                url_remote_list_to_return.append([remote, url])
    return url_remote_list_to_return

def getBranchOfCommit(repo, commit):
    """Find a branch where a commit exists.

    Arguments:
        repo: [path] directory of repository.
        commit: [str] commit hash

    Return:
        local_branch: [str] A local branch where a commit exists.
    """

    _, stdout, _ = runGitCommand(["branch", "-a", "--contains", commit], cwd=repo,
                                 timeout=GIT_LOCAL_TIMEOUT)

    local_branch = stdout.split("\n")[0].replace("*", "").strip()
    return local_branch

def getLatestCommit(repo, commit_branch):
    """Get the latest commit on a specific branch on the local repository.

    Arguments:
        repo: [path] repository directory.
        commit_branch: [str] branch.

    Return:
        commit: [str] the hash of the latest commit on commit_branch in repository
    """

    if commit_branch.startswith("remotes/"):
        commit_branch = commit_branch[len("remotes/"):]

    _, stdout, _ = runGitCommand(["branch", "-r", "-v"], cwd=repo, timeout=GIT_LOCAL_TIMEOUT)

    commit_list = stdout.split("\n")
    commit = None
    for branch in commit_list:

        branch_list = branch.split()
        if len(branch_list) > 1:
            remote_branch = branch_list[0]
            remote_commit = branch_list[1]

            if commit_branch == remote_branch:
                commit = remote_commit
                break
    return commit

def getRemoteBranchNameForCommit(repo, commit):
    """Get the remote branch name for a commit on a local branch.

    Arguments:
        repo: [path] directory of repository.
        commit: [str] commit hash.

    Return:
        remote_branch_name: [str] the full name of the remote branch where commit exists.
    """

    local_branch_list = []
    try:
        _, stdout, _ = runGitCommand(["branch", "-a", "--contains", commit], cwd=repo,
                                     timeout=GIT_LOCAL_TIMEOUT)
        local_branch_list = stdout.split("\n")

    except Exception as e:
        log.debug("Could not list the branches containing the commit: " + repr(e))

    remote_branch_name = None
    for branch in local_branch_list:
        branch_stripped = branch.strip()
        if branch_stripped.startswith("remotes/"):
            remote_branch_name = branch_stripped

    return remote_branch_name

def daysBehind(timeout=GIT_NETWORK_TIMEOUT):
    """Measure how far behind the latest commit on the active branch is behind a branch with that commit on the remote
    repository.

    Keyword arguments:
        timeout: [float] hard timeout of every git operation which touches the network, in seconds.

    Return:
        (days_behind, remote_branch): number of days behind the latest remote commit that the latest
            local commit is on the active branch, and the remote branch it was measured against.
            (None, None) if the lag could not be determined.
    """

    repo_directory = getRmsRootDir()

    latest_local_commit = getCommit(repo_directory)
    latest_local_date = getDateOfCommit(repo_directory, latest_local_commit)

    remote_urls = getRemoteUrls(repo_directory)

    target_directory_obj = tempfile.TemporaryDirectory()

    try:

        commit_repo_directory = updateCommitHistoryDirectory(remote_urls, target_directory_obj.name,
                                                             timeout=timeout)

        remote_branch_of_commit = getRemoteBranchNameForCommit(commit_repo_directory, latest_local_commit)

        if remote_branch_of_commit is None:
            log.warning("The latest local commit was not found on any remote branch, "
                        "the repository lag could not be measured")
            return None, None

        latest_remote_date = getDateOfCommit(commit_repo_directory, remote_branch_of_commit)
        days_behind = (latest_remote_date - latest_local_date).total_seconds()/(60 * 60 * 24)

        return days_behind, remote_branch_of_commit

    finally:

        try:
            target_directory_obj.cleanup()

        except Exception as e:
            log.debug("Could not clean up the temporary commit history directory: " + repr(e))


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

    obs_start_time, obs_duration, obs_end_time = getEphemTimesFromCaptureDirectory(config, night_directory)

    # print("Night directory was {}".format(night_directory))
    # print("Observation start time was {}".format(obs_start_time))
    # print("Observation duration was {}".format(obs_duration))
    # print("Observation end time was {}".format(obs_end_time))

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

    next_start_time = getNextStartTime(conn, obs_end_time)
    # print("Observation start time was {}".format(obs_start_time))
    # print("Next start time was {}".format(next_start_time))

    sql_statement = ""
    sql_statement += "SELECT Key, Value from records \n"
    sql_statement += "           WHERE TimeStamp >= '{}' \n".format(obs_start_time)
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

def startObservationSummaryReport(config, duration, force_delete=False):
    """ Enters the parameters known at the start of observation into the database.

    Arguments:
        config: [config] config file.
        duration: [int]the initially calculated duration seconds.

    Keyword arguments:
        force_delete: [bool] forces deletion of the observation summary database, default False.

    Return:
        [str] message about session.

    """


    conn = getObsDBConn(config, force_delete=force_delete)
    start_time_object = (datetime.datetime.now(datetime.timezone.utc) -
                         datetime.timedelta(seconds=1)).replace(tzinfo=datetime.timezone.utc)
    start_time_object_rounded = start_time_object.replace(microsecond=0)
    addObsParam(conn, "start_time", start_time_object_rounded)
    addObsParam(conn, "duration_from_start_of_observation", duration)
    addObsParam(conn, "stationID", sanitise(config.stationID, space_substitution=""))

    if isRaspberryPi():
        with open('/sys/firmware/devicetree/base/model', 'r') as m:
            hardware_version = sanitise(m.read().lower(), space_substitution=" ")
    else:
        hardware_version = sanitise(platform.machine(), space_substitution=" ")

    addObsParam(conn, "hardware_version", hardware_version)

    try:
        repo_path = getRmsRootDir()
        repo = git.Repo(repo_path)
        if repo:
            addObsParam(conn, "commit_date",
                        UTCFromTimestamp.utcfromtimestamp(repo.head.object.committed_date).strftime('%Y%m%d_%H%M%S'))
            addObsParam(conn, "commit_hash", repo.head.object.hexsha)
        else:
            print("RMS Git repository not found. Skipping Git-related information.")
    except:
        print("Error getting Git information. Skipping Git-related information.")
    
    # Get the disk usage info (only in Python 3.3+) for the data_dir disc
    if (sys.version_info.major > 2) and (sys.version_info.minor > 2):

        try:
            storage_total, storage_used, storage_free = shutil.disk_usage(config.data_dir)
            addObsParam(conn, "storage_total_gb", round(storage_total/(1024**3), 2))
            addObsParam(conn, "storage_used_gb", round(storage_used/(1024**3), 2))
            addObsParam(conn, "storage_free_gb", round(storage_free/(1024**3), 2))
        except:
            addObsParam(conn, "storage_total_gb", "Not available")
            addObsParam(conn, "storage_used_gb", "Not available")
            addObsParam(conn, "storage_free_gb", "Not available")

    captured_directories = captureDirectories(os.path.join(config.data_dir, config.captured_dir), config.stationID)
    addObsParam(conn, "captured_directories", captured_directories)
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

    try:
        timeSyncStatus(config, obs_db_conn)
    except Exception as e:
        print(repr(e))


    platepar_path = os.path.join(config.config_file_path, config.platepar_name)
    if os.path.exists(platepar_path):
        platepar = Platepar()
        platepar.read(platepar_path, use_flat=config.use_flat)
        addObsParam(obs_db_conn, "camera_pointing_az", format("{:.2f} degrees".format(platepar.az_centre)))
        addObsParam(obs_db_conn, "camera_pointing_alt", format("{:.2f} degrees".format(platepar.alt_centre)))
        addObsParam(obs_db_conn, "camera_fov_h","{:.2f}".format(platepar.fov_h))
        addObsParam(obs_db_conn, "camera_fov_v","{:.2f}".format(platepar.fov_v))
        addObsParam(obs_db_conn, "camera_lens", estimateLens(platepar.fov_h))

    addObsParam(obs_db_conn, "continuous_capture", config.continuous_capture)
    addObsParam(obs_db_conn, "time_start_ephem", start_ephem)
    addObsParam(obs_db_conn, "time_first_fits_file", time_first_fits_file)
    addObsParam(obs_db_conn, "time_end_ephem", end_ephem)
    addObsParam(obs_db_conn, "time_last_fits_file", time_last_fits_file)
    addObsParam(obs_db_conn, "capture_duration_from_fits", capture_duration_from_fits)
    addObsParam(obs_db_conn, "capture_duration_from_ephemeris", capture_duration_from_ephemeris)
    addObsParam(obs_db_conn, "total_expected_fits", round(total_expected_fits))
    addObsParam(obs_db_conn, "total_expected_fits_ephemeris", round(total_expected_fits_ephemeris))
    addObsParam(obs_db_conn, "total_fits", fits_count)
    addObsParam(obs_db_conn, "fits_file_shortfall", fits_file_shortfall)
    addObsParam(obs_db_conn, "fits_file_shortfall_ephemeris", fits_file_shortfall_ephemeris)
    addObsParam(obs_db_conn, "fits_file_shortfall_as_time", fits_file_shortfall_as_time)
    addObsParam(obs_db_conn, "fits_file_shortfall_as_time_ephemeris", fits_file_shortfall_as_time_ephemeris)
    addObsParam(obs_db_conn, "protocol_in_use", config.protocol)
    addObsParam(obs_db_conn, "star_catalog_file", config.star_catalog_file)

    try:
        days_behind, remote_branch = daysBehind()

        if days_behind is None:
            addObsParam(obs_db_conn, "repository_lag_remote_days", "Not determined")

        else:
            addObsParam(obs_db_conn, "repository_lag_remote_days", days_behind)

            if remote_branch is not None:
                addObsParam(obs_db_conn, "remote_branch", os.path.basename(remote_branch))

    except Exception as e:
        log.warning("Could not determine the repository lag: " + repr(e))
        addObsParam(obs_db_conn, "repository_lag_remote_days", "Not determined")
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
