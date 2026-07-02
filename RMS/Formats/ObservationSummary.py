# The MIT License

# Copyright (c) 2026

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
import traceback
import argparse

from RMS.ConfigReader import parse
from RMS.Misc import niceFormat, isRaspberryPi, sanitise, getRMSStyleFileName, getRmsRootDir, UTCFromTimestamp
from RMS.Formats.FFfits import filenameToDatetimeStr
from RMS.Formats.Platepar import Platepar
from RMS.CaptureDuration import captureDuration
from RMS.CaptureModeSwitcher import SWITCH_HORIZON_DEG
from RMS.Formats.FTPdetectinfo import findFTPdetectinfoFile, readFTPdetectinfo
from RMS.Logger import getLogger
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import RMS.ConfigReader as cr
import subprocess

import dvrip as dvr

# File locking for the per-night working JSON (POSIX only; degrade gracefully elsewhere)
try:
    import fcntl
except ImportError:
    fcntl = None

# Get the logger from the main module
log = getLogger("rmslogger")

DEBUG_PRINT = False
RUNNING_FROM_CONSOLE = False

OBSERVATION_SUMMARY_WORKING_NAME_JSON = "observation_summary_working.json"
OBSERVATION_SUMMARY_NAME_JSON = "observation_summary.json"
OBSERVATION_SUMMARY_NAME_TXT = "observation_summary.txt"
OBSERVATION_SUMMARY_NAME_PNG = "observation_summary.png"
OBSERVATIONS_TABLE_NAME = "observations"
OBSERVATION_DB_FILE_NAME = "observation.db"
NIGHT_DATA_DIR_COL = "night_data_dir"


def pingOnce(host):
    """Quickly detect if a host is pingable

    Arguments:
        host: [str} ip address of host to be pinged.

    Return:
        [bool]: True if pinged, otherwise False.

    """
    try:
        result = subprocess.run(
            ["ping", "-c", "1", "-W", "1", host],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return result.returncode == 0
    except Exception:
        return False


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
    observation_records_db_path = os.path.join(config.data_dir,OBSERVATION_DB_FILE_NAME)
    log.info(f"Opening database at {observation_records_db_path}")
    if force_delete and os.path.exists(observation_records_db_path):
        os.unlink(observation_records_db_path)

    if not os.path.exists(os.path.dirname(observation_records_db_path)):
        # Handle the very rare case where this could run before any observation sessions
        # and RMS_data does not exist
        try:
            # Create the required directory
            os.makedirs(os.path.dirname(observation_records_db_path))

        except Exception as e:
            log.error(f'Unable to create {observation_records_db_path}:' + repr(e))
            log.error("".join(traceback.format_exception(*sys.exc_info())))
            return None

    try:
        conn = sqlite3.connect(observation_records_db_path)

    except Exception as e:
        log.error('Unable to get database connection:' + repr(e))
        log.error("".join(traceback.format_exception(*sys.exc_info())))

        return None

    # Returns true if the table observations exists in the database
    try:
        sql_command = f"SELECT name FROM sqlite_master WHERE type='table' and name='{OBSERVATIONS_TABLE_NAME}';"

        tables = conn.cursor().execute(sql_command).fetchall()

        if len(tables) > 0:
            return conn
    except:
        print(f"{OBSERVATIONS_TABLE_NAME} does not exist")


    sql_command = ""
    sql_command += f"CREATE TABLE {OBSERVATIONS_TABLE_NAME} \n"
    sql_command += f"( \n"
    sql_command += f"{NIGHT_DATA_DIR_COL} TEXT PRIMARY KEY \n"
    sql_command += f") \n"

    conn.execute(sql_command)

    return conn

def getColumns(conn):
    """Get the columns in the observation table.

    Arguments:
        conn: connection to database.

    Return:
        [set]: Set of columns in table.
    """

    cursor = conn.execute(f"PRAGMA table_info({OBSERVATIONS_TABLE_NAME})")
    return {row[1] for row in cursor.fetchall()}

def addRequiredColumns(conn, d):
    """For each key in d if not already a column in table, add as a column.

    Arguments:
        conn: connection to database.
        d: [dict] Dictionary of keys and values for the observation summary.

     Return:
        [set] columns existing in the table
    """

    # If d has not yet been initialised, return to prevent iterating over None
    if d is None:
        log.info("Not adding columns for observation summary dictionary which is None")
        return set()

    existing = getColumns(conn)
    for key in d:
        # SQLite cannot bind identifiers in DDL, so guard against anything that is not a plain
        # column name before interpolating it into the ALTER TABLE statement.
        if not re.match(r'^[A-Za-z0-9_]+$', key):
            log.warning("Skipping observation summary key with unsafe column name: {!r}".format(key))
            continue
        if key.lower() not in existing:
            sql_command = f"ALTER TABLE {OBSERVATIONS_TABLE_NAME} ADD COLUMN {key.lower()} TEXT"
            conn.execute(sql_command)

    return set(getColumns(conn))

def storeDictInDB(conn, d, debug=False):
    """Store the dict d in the observation summary database, create new columns if needed.

    Arguments:
        conn: connection to database.
        d: [dict] Dictionary of keys and values for the observation summary.

    Keyword Arguments:
        debug: [bool] Optional, default False, print debugging information

    Return:
        Nothing.
    """

    # Nothing to store if the dict is None, return early
    if d is None:
        log.info("Not storing an empty observation summary in the database")
        return

    # Ensure schema is up to date
    existing_columns = addRequiredColumns(conn, d)

    # Columns are always created lower-cased (see addRequiredColumns), and SQLite column
    # names are case-insensitive, so match case-insensitively and key the filtered dict by
    # the lower-cased name. Comparing the original-case key would silently drop mixed-case
    # keys such as "stationID".
    dict_filtered_by_columns = {k.lower(): v for k, v in d.items() if k.lower() in existing_columns}

    dropped = {k for k in d.keys() if k.lower() not in existing_columns}
    if len(dropped) != 0:
        log.warning(f"No columns for following keys: {sorted(dropped)}")

    # Normalise booleans safely (TEXT columns expect strings)
    clean = {
        k: ("True" if v is True else "False" if v is False else v)
        for k, v in dict_filtered_by_columns.items()
    }

    # Only store the basename for night_data_dir
    if "night_data_dir" in clean:
        clean["night_data_dir"] = os.path.basename(clean["night_data_dir"])

    columns = list(clean.keys())
    placeholders = ", ".join("?" for _ in columns)
    values = [clean[col] for col in columns]

    assignments = ", ".join(f"{col}=excluded.{col}" for col in columns if col != "night_data_dir")

    if debug:
        for c, v in zip(columns, values):
            print(f"{c:40} -> {repr(v)}")

    sql_command = ""
    sql_command += f"INSERT INTO {OBSERVATIONS_TABLE_NAME} ({', '.join(columns)})\n"
    sql_command += f"VALUES ({placeholders})\n"
    sql_command += f"ON CONFLICT(night_data_dir) DO UPDATE SET {assignments}\n"

    # Show the SQL with placeholders
    if debug:
        print(sql_command)
        print(values)

    try:
        conn.execute(sql_command, values)
        conn.commit()

    except Exception as e:
        log.error('Storing observation summary into database failed with error:' + repr(e))
        log.error("".join(traceback.format_exception(*sys.exc_info())))

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

    original_start_time = start_time
    ephemeris_start_time, duration = captureDuration(config.latitude, config.longitude, config.elevation, start_time)

    # captureDuration returns a bool (not a datetime) as the first element when it cannot pin a
    # concrete sunset (we are already inside the dark window, or it is polar day/night). Walk back
    # to find the sunset that began the current dark period, but cap the search - during polar
    # night captureDuration always returns a bool regardless of start_time, so an unbounded loop
    # would spin forever.
    max_backoff_minutes = 24*60
    backoff = 0
    while isinstance(ephemeris_start_time, bool) and backoff < max_backoff_minutes:
        start_time -= datetime.timedelta(minutes=1)
        backoff += 1
        # Go backwards through time until we are before the start time
        ephemeris_start_time, duration = captureDuration(config.latitude, config.longitude, config.elevation, start_time)

    if isinstance(ephemeris_start_time, bool):
        # No concrete sunset within the search window (e.g. polar night) - fall back to the passed time.
        log.warning("getObservationDurationNightTime: no concrete sunset found; falling back to start_time")
        ephemeris_start_time = original_start_time

    end_time = ephemeris_start_time + datetime.timedelta(seconds=duration)

    return ephemeris_start_time, duration, end_time

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

    # Bound the search so it can never spin (during polar night/day captureDuration
    # never finds a concrete boundary), mirroring getObservationDurationNightTime.
    max_advance_minutes = 24*60
    advanced = 0
    try:
        while o.next_setting(s).datetime() < o.next_rising(s).datetime() and advanced < max_advance_minutes:
            if DEBUG_PRINT:
                print("{} is not at night time".format(start_time))
            start_time += datetime.timedelta(minutes=1)
            advanced += 1
            o.date = start_time
            s.compute()

    except (ephem.AlwaysUpError, ephem.NeverUpError):
        # Polar day/night: the Sun never sets/rises, so the start time cannot be
        # refined. The duration block below falls back to duration=0.
        log.warning("Polar day/night: no Sun setting/rising; cannot refine continuous-capture start time")

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
            # Other error such as systemctl not available
            pass
    return "Not recognized"

def timeSyncStatus(config, d, force_client=None):
    """Add time sync information to the observation summary.

    Arguments:
        config: [Config] Configuration object.
        d: [dict] Observation summary dictionary

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
        addObsParam(d, "clock_measurement_source", "ntp")
        addObsParam(d, "clock_synchronized", synchronized)
        addObsParam(d, "clock_ahead_ms", ahead_ms)
        addObsParam(d, "clock_error_uncertainty_ms", uncertainty)

    elif time_client == "chronyd":
        synchronized, ahead_ms, uncertainty_ms = getChronyUncertainty()
        addObsParam(d, "clock_measurement_source", "chrony")
        addObsParam(d, "clock_synchronized", synchronized)
        addObsParam(d, "clock_ahead_ms", ahead_ms)
        addObsParam(d, "clock_error_uncertainty_ms", uncertainty_ms)

    else:
        addObsParam(d, "clock_measurement_source", "Not detected")
        try:
            remote_time_query, uncertainty, time_server = timestampFromNTP(config.time_server)
        except Exception:
            remote_time_query, uncertainty, time_server = (None, None, None)
        addObsParam(d, "time_server", time_server)
        if remote_time_query is not None:
            local_time_query = (datetime.datetime.now(datetime.timezone.utc)
                                - datetime.datetime(1970, 1, 1)
                                        .replace(tzinfo=datetime.timezone.utc)).total_seconds()
            ahead_ms = (local_time_query - remote_time_query) * 1000
            addObsParam(d, "clock_error_uncertainty_ms", uncertainty * 1000)

        else:
            ahead_ms, uncertainty = "Unknown", "Unknown"
            addObsParam(d, "clock_error_uncertainty_ms", uncertainty)
        addObsParam(d, "clock_ahead_ms", ahead_ms)

        try:
            result_list = subprocess.run(['timedatectl','status'], capture_output = True).stdout.splitlines()
        except Exception:
            result_list = []

        for raw_result in result_list:
            result = raw_result.decode('ascii')
            if "synchronized" in result:

                if result.split(":")[1].strip() == "no":
                    addObsParam(d, "clock_synchronized", False)
                else:
                    addObsParam(d, "clock_synchronized", True)

    return ahead_ms

def getDaysSinceLastDetection(config, data_dir, d=None, debug=False):
    """Get the number of days since the last meteor detection

    Arguments:
        config: [config] RMS configuration instance.
        data_dir: [path] path to the data_dir.
        d: [dict] Obseravation summary dictonary.
        debug: [bool] Run in debug mode.

    Returns:
        days_since_last_detection: [int].
    """


    last_fits_file_for_session_sql = ""
    last_fits_file_for_session_sql += f"SELECT time_last_fits_file\n"
    last_fits_file_for_session_sql += f"        FROM {OBSERVATIONS_TABLE_NAME}\n"
    last_fits_file_for_session_sql += f"        WHERE night_data_dir = ?\n"
    last_fits_file_for_session_sql += f"        LIMIT 1; "


    if debug:
        log.info("Last fits file for session SQL")
        log.info(last_fits_file_for_session_sql)

    try:
        conn = getObsDBConn(config)
        result = conn.execute(last_fits_file_for_session_sql, (os.path.basename(data_dir),)).fetchone()
        if result is None:
            return "Unknown"
        else:
            result = str(result[0])
        log.info(f"SQL query is \n {last_fits_file_for_session_sql}")
        log.info(f"SQL query result is \n {result}")
        # Keep microseconds
        if '.' in result:
            time_last_fits_file_for_session = datetime.datetime.strptime(result,  "%Y-%m-%d %H:%M:%S.%f")
        else:
            time_last_fits_file_for_session = datetime.datetime.strptime(result, "%Y-%m-%d %H:%M:%S")
        conn.close()

    except Exception as e:
        log.error('Failed to calculate time since last detection:' + repr(e))
        log.error("".join(traceback.format_exception(*sys.exc_info())))
        return "Error"


    last_detection_time_for_session_sql = ""
    last_detection_time_for_session_sql += "SELECT time_last_detection\n"
    last_detection_time_for_session_sql += f"   FROM {OBSERVATIONS_TABLE_NAME}\n"
    last_detection_time_for_session_sql += f"   WHERE COALESCE(detections_after_ml, '0') != '0'\n"
    last_detection_time_for_session_sql += f"   AND detections_after_ml IS NOT NULL\n"
    last_detection_time_for_session_sql += f"   AND time_last_fits_file <= '{time_last_fits_file_for_session}'\n"
    last_detection_time_for_session_sql += f"   ORDER BY time_last_detection DESC LIMIT 1;\n"

    if debug:
        log.info("Last detection time for session SQL")
        log.info(last_detection_time_for_session_sql)

    log.info("Write dict to db before doing SQL")

    try:
        conn = getObsDBConn(config)
        storeDictInDB(conn,d, debug=False)

        cursor = conn.execute(last_detection_time_for_session_sql)
        result =  cursor.fetchone()[0]
        last_detection_time_for_session = datetime.datetime.strptime(result, "%Y-%m-%d %H:%M:%S")

        # Guard against missing fits files causing negative time since last detection
        seconds_since_last_detection = max((time_last_fits_file_for_session - last_detection_time_for_session).total_seconds(), 0)

        # Express the gap in solar days (24 h), which is the intuitive unit for "days since".
        days_since_last_detection = seconds_since_last_detection / (60 * 60 * 24.0)
        conn.close()

    except Exception as e:
        log.error('Failed to calculate time since last detection:' + repr(e))
        log.error("".join(traceback.format_exception(*sys.exc_info())))
        return "Error"

    log.info(f"Time since last detection is {days_since_last_detection} days")


    return days_since_last_detection

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
        addr: [str] optional, address of ntp server to use.

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
            return None, None, addr

        # Now calculate estimated clock offsets
        clock_offset_out_leg = remote_clock_time_receive_timestamp - local_clock_transmit_timestamp
        clock_offset_return_leg = remote_clock_time_transmit_timestamp - local_clock_receive_timestamp
        estimated_offset = (clock_offset_out_leg + clock_offset_return_leg)/2
        adjusted_time = remote_clock_time_transmit_timestamp + estimated_offset
        return adjusted_time, estimated_network_delay, addr
    else:
        return None, None, addr

def addObsParam(d, key, value):
    """Add a single key value pair into the observation summary dictionary

    Arguments:
        observation_summary_dict [c]: the dict holding the information
        key [str]: the key for the value to be added
        value [str]: the value to be added

    Return:
        Nothing

    """

    if 'night_data_dir' in d and key == 'night_data_dir':
        if d['night_data_dir'] != value:
            log.warning("Observation summary night_data_dir is changing - this is unexpected")


    d[key] = str(value)
    saveObservationSummaryDict(d)

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

def countKeyStringsInLogs(session_start, config, key_string="Traceback (most recent call last)"):
    """Count the number of occurences of key_string in log files from the current session.

    Scans all log files in the log directory that were modified after the session's
    start_time (from the observation database) for lines containing 'Traceback
    (most recent call last)'.

    Arguments:
        session_start: [datetime] Time object for session start
        config: [config] RMS configuration instance.

    Keyword arguments:
        key_string: [str] Optional default "Traceback (most recent call last)" - string to be sought

    Return:
        count: [int] Number of tracebacks found, or 0 if logs cannot be read.
    """

    log_dir = os.path.join(config.data_dir, config.log_dir)

    if not os.path.isdir(log_dir):
        return 0

    # Find log files modified after the session start
    key_string_count = 0
    log_pattern = "log_{}_".format(config.stationID)

    for filename in sorted(os.listdir(log_dir)):
        if not filename.endswith(".log") or log_pattern not in filename:
            continue

        filepath = os.path.join(log_dir, filename)

        # Only consider log files modified after the session started
        file_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(filepath),tz=datetime.timezone.utc)
        if file_mtime < session_start:
            continue

        try:
            with open(filepath, 'r', errors='replace') as f:
                for line in f:
                    if key_string in line:
                        key_string_count += 1
        except Exception:
            continue

    return key_string_count


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
        (sensor, firmware, build_date): [tuple of strings]
            sensor: hardware/sensor identifier
            firmware: firmware version string, or "" if not available
            build_date: firmware build date string, or "" if not available

    """

    ip = re.search(r'(?:\d{1,3}\.){3}\d{1,3}', config.deviceID).group()

    if RUNNING_FROM_CONSOLE and not pingOnce(ip):
        return ("Unavailable", "Unavailable", "Unavailable")

    for _ in range(attempts):
        try:
            cam = dvr.DVRIPCam(ip, timeout=sock_timeout)
            if cam.login():
                sys_info = cam.get_system_info()
                cam.close()
                sensor = sys_info.get('HardWare', 'Unknown')
                fw = sys_info.get('SoftWareVersion', '')
                build_time = sys_info.get('BuildTime', '')
                return (sensor, fw, build_time)
        except (socket.timeout, OSError, ConnectionError):
            # Camera may still be rebooting - ignore and retry
            pass
        time.sleep(delay)

    return ("Unavailable", "Unavailable", "Unavailable")

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

def updateCommitHistoryDirectory(remote_urls, target_directory):

    """ Clone only the commit history of a remote repository.

    Arguments:
        remote_urls: [url] the remote url to be cloned/
        target_directory: [path] the directory into which to clone.

    Return:
        commit_repo_directory: [path] directory of the repository
    """


    if os.path.exists(target_directory):
        shutil.rmtree(target_directory)

    os.makedirs(target_directory)
    first_remote = True
    for remote_url in remote_urls:
        local_name, url = remote_url[0], remote_url[1]

        if first_remote:
            first_remote = False
            p = subprocess.Popen(["git", "clone", url, "--filter=blob:none", "--no-checkout"], cwd=target_directory,
                             stdout=subprocess.PIPE)
            p.wait()
            # this first remote might have been pulled in with the wrong local_name so rename it
            commit_repo_directory = os.path.join(target_directory, os.listdir(target_directory)[0])
            downloaded_remote_name = subprocess.check_output(["git", "remote"], cwd = commit_repo_directory).strip().decode('utf-8')

            if downloaded_remote_name != local_name:
                p = subprocess.Popen(["git", "remote", "rename", downloaded_remote_name, local_name], cwd = commit_repo_directory)
                p.wait()

        else:
            # this is not the first remote so add another remote

            p = subprocess.Popen(["git", "remote", "add", local_name, url], cwd = commit_repo_directory)
            p.wait()
            p = subprocess.Popen(["git", "fetch", "--filter=blob:none", local_name], cwd = commit_repo_directory)
            p.wait()
    return commit_repo_directory

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

def getBranchOfCommit(repo, commit):
    """Find a branch where a commit exists.

    Arguments:
        repo: [path] directory of repository.
        commit: [str] commit hash

    Return:
        local_branch: [str] A local branch where a commit exists.
    """

    local_branch = subprocess.check_output(["git", "branch", "-a", "--contains", commit], cwd=repo).decode(
         "utf-8").split("\n")[0].replace("*", "").strip()
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

    commit_list = subprocess.check_output(["git", "branch", "-r", "-v"], cwd=repo).decode("utf-8").split("\n")
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


    # This is the simple case, our latest commit is the HEAD

    local_branch_list = []
    try:
        local_branch_list = subprocess.check_output(["git", "branch", "-r", "--points-at", commit], cwd=repo).decode(
            "utf-8").split("\n")
    except:
        pass

    remote_branch_name = None
    for branch in local_branch_list:
        branch_stripped = branch.strip()
        if branch_stripped.startswith("remotes/"):
            remote_branch_name = branch_stripped

    # If we are not at the HEAD, then get all the branches which contain the commit.

    # 2. Branches that *contain* the commit
    try:
        contains = subprocess.check_output(
            ["git", "branch", "-r", "--contains", commit],
            cwd=repo
        ).decode().splitlines()
    except Exception:
        contains = []

    contains = [c.strip() for c in contains if c.strip()]

    if contains:
        # If the branch is origin/main or origin/pre-release; then that is almost certainly where we are
        for preferred in ["origin/master", "origin/prerelease"]:
            if preferred in contains:
                return preferred
        return contains[0]

    # Fall back return
    return remote_branch_name

def daysBehind():
    """Measure how far behind the latest commit on the active branch is behind a branch with that commit on the remote
    repository.

    Arguments:
        syscon: [config] RMS config object.

    Return:
        number of days behind the latest remote commit that the latest local commit is on the active branch.
    """

    latest_local_commit = getCommit(os.getcwd())
    latest_local_date = getDateOfCommit(os.getcwd(), latest_local_commit)
    target_directory_obj = tempfile.TemporaryDirectory()
    target_directory = target_directory_obj.name
    remote_urls = getRemoteUrls(os.getcwd())
    commit_repo_directory = updateCommitHistoryDirectory(remote_urls, target_directory)
    remote_branch_of_commit = getRemoteBranchNameForCommit(commit_repo_directory, latest_local_commit)
    if not remote_branch_of_commit is None:
        latest_remote_date = getDateOfCommit(commit_repo_directory, remote_branch_of_commit)
        days_behind = (latest_remote_date - latest_local_date).total_seconds()/(60 * 60 * 24)
        target_directory_obj.cleanup()
        return days_behind, remote_branch_of_commit
    else:
        target_directory_obj.cleanup()
        return "Unable to determine"

def serialize(config, format_nicely=True, as_json=False, night_directory=None, drop_keys_list=None, ordering=None, final=False):
    """ Returns the data from the most recent observation session as either colon
        delimited text file, ar as a json.

    Arguments:
        config: [config] station config file.

    Keyword Arguments:
        format_nicely: [bool] optional, default true, present the data with delimiter characters aligned.
        as_json: [bool] optional, default false, return the data as a json.
        night_directory: [string] optional, default None, the night directory to use
        drop_keys_list: [string] any keys to exclude
        ordering: [list] List of keys showing the order they should be written for text files

    Return:
        string of key value pairs committed to the database since the start of the previous observation session.
    """

    d = getObservationSummaryDict(night_directory, final=final)



    if ordering is None:
        ordering = ['stationID',
                    'commit_date', 'commit_hash', 'remote_branch', 'repository_lag_remote_days',
                    'star_catalog_file',
                    'hardware_version',
                    'captured_directories',
                    'storage_used_gb', 'storage_free_gb', 'storage_total_gb',
                    'camera_lens','camera_fov_h','camera_fov_v',
                    'camera_pointing_alt','camera_pointing_az',
                    'camera_information', 'camera_firmware_build_date', 'camera_firmware_version',
                    'clock_measurement_source', 'clock_synchronized', 'clock_ahead_ms', 'clock_error_uncertainty_ms', 'time_server',
                    'start_time', 'duration_from_start_of_observation', 'continuous_capture', 'photometry_good',
                    'time_start_ephem', 'time_first_fits_file', 'time_first_detection', 'time_last_detection',
                    'time_end_ephem', 'time_last_fits_file', 'days_since_last_detection',
                    'total_expected_fits','total_fits',
                    'fits_files_from_duration','fits_file_shortfall', 'fits_file_shortfall_as_time',
                    'capture_duration_from_fits',
                    'capture_duration_from_ephemeris', 'total_expected_fits_ephemeris', 'fits_file_shortfall_ephemeris',
                    'fits_file_shortfall_as_time_ephemeris',
                    'detections_after_ml',
                    'media_backend','protocol_in_use','jitter_quality','dropped_frame_rate','kht_wrapper_count',
                    'traceback_count']


    # Warn for duplicated keys in ordering list
    seen = set()
    for key_name in ordering:
        if key_name in seen:
            log.warning(f"Duplicated key {key_name} in ordering list")
        else:
            seen.add(key_name)

    # Dedupe while preserving first occurrence - the ordering list contains a few repeats
    # (e.g. media_backend, star_catalog_file) which would otherwise produce duplicate output lines.
    ordering = list(dict.fromkeys(ordering))

    if drop_keys_list:
        if isinstance(drop_keys_list, str):
            drop_keys_list = [drop_keys_list]

        for key in drop_keys_list:
            d.pop(key, None)

    if as_json:
        return json.dumps(d, default=lambda o: o.__dict__, indent=4, sort_keys=True)

    output = ""

    # Use list to make a copy - rather than iterating over the list we are modifying
    output_ordering = list(ordering)
    seen = set(ordering)

    for key in d:
        if key not in seen:
            output_ordering.append(key)
            seen.add(key)

    for key in output_ordering:
        if key not in d:
            continue
        value = d[key]
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
        file_path_and_name: [path] full path to the target file.
        night_dir: [path] path to capture directory for the night

    Return:
        [string] string of key value pairs committed to the database since the start of the observation session.
        """


    with open(file_path_and_name, "w") as summary_file_handle:
        as_ascii = serialize(config, night_directory=night_dir, drop_keys_list="night_data_dir").encode("ascii", errors="ignore").decode("ascii")
        summary_file_handle.write(as_ascii)
        summary_file_handle.flush()


def writeToPNG(config, file_path_and_name, night_dir, font_size=16, line_gap=4, padding=10,
               col_gap=20, char_height=15, char_width=10,
               text_colour=(255, 140, 0), bg_colour=(25, 10, 0), alpha_blur=0.8, radius_blur=2.0):

    """Write colon delimited text to png image.

    Arguments:
        config: [config] station config file.
        file_path_and_name: [path full path to the target file.
        night_dir: [path] path to capture directory for the night.

    Keyword arguments:
        font_size: [int] Font size.
        line_gap: [int] Gap between lines.
        padding: [int] Border around image.
        col_gap: [int] gap between columns.
        char_height: [int] height of characters.
        char_width: [int] width of characters used to compute column width.
        text_colour: (r,g,b) Colour for text, optional default (255,140,0)
        bg_colour: (r,g,b) Colour for text, optional default (25,10,0) - VT320 style
        alpha_blur: [float] alpha for blurring overlay
        radius_blur: [float] pixel radius for blurring

    Return:
        [string] string of key value pairs committed to the database since the start of the observation session.
        """

    # Rendering the PNG is a nice-to-have for the weblog; never let it break finalization.
    try:
        as_ascii = serialize(
            config,
            night_directory=night_dir,
            drop_keys_list="night_data_dir"
        ).encode("ascii", errors="ignore").decode("ascii")

        lines_list = as_ascii.split("\n")

        # Remove final empty line if present
        if lines_list and lines_list[-1].strip() == "":
            lines_list.pop()

        # Split into two columns
        mid = (len(lines_list) + 1) // 2
        col1_list, col2_list = lines_list[:mid], lines_list[mid:]


        # Monospace font - fall back to the PIL default if the DejaVu font is not installed.
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        # Measure column widths
        col1_width = max(char_width * len(line) for line in col1_list) if col1_list else 0
        col2_width = max(char_width * len(line) for line in col2_list) if col2_list else 0

        # Total image size
        img_width = padding + col1_width + col_gap + col2_width + padding
        img_height = padding + (char_height + line_gap) * max(len(col1_list), len(col2_list)) + padding

        # Background
        img = Image.new("RGB", (img_width, img_height), bg_colour)
        draw = ImageDraw.Draw(img)

        # Draw column 1
        y = padding
        for line in col1_list:
            draw.text((padding, y), line, font=font, fill=text_colour)
            y += char_height + line_gap

        # Draw column 2
        x2 = padding + col1_width + col_gap
        y = padding
        for line in col2_list:
            draw.text((x2, y), line, font=font, fill=text_colour)
            y += char_height + line_gap


        glow = img.filter(ImageFilter.GaussianBlur(radius=radius_blur))
        img = Image.blend(glow, img, alpha=alpha_blur)

        img.save(file_path_and_name)

        return os.path.basename(file_path_and_name)

    except Exception as e:
        log.warning("Could not render observation summary PNG: {}".format(e))
        return None





def writeToJSON(config, file_path_and_name, night_dir):
    """Write as a json.
    Arguments:
        config: [config] station config file.
        file_path_and_name: [path] full path to the target file.

    Return:
        Nothing
    """

    with open(file_path_and_name, "w") as summary_file_handle:
        as_ascii = serialize(config, as_json=True, night_directory=night_dir, drop_keys_list=["night_data_dir"]).encode("ascii", errors="ignore").decode("ascii")
        summary_file_handle.write(as_ascii)
        summary_file_handle.flush()


def getTimeOfFirstAndLastDetectionInDir(data_dir):
    """Get the time of the first and last meteor detections in the data_dir

    Arguments:
        data_dir:[path] Path to the data_dir to be checked

    Return:
        [str] First detection time
        [str] Last detection time

    """

    first_detection, last_detection = "0", "0"
    log.info(f"Looking for FTP file in {data_dir}")
    ftp_file = findFTPdetectinfoFile(data_dir)
    log.info(f"Found FTP file {ftp_file}")
    ftp_detect_info = readFTPdetectinfo(data_dir, ftp_file)
    if len(ftp_detect_info):
        first_detection, last_detection = ftp_detect_info[0][0], ftp_detect_info[-1][0]
        log.info("First detection info: {}".format(first_detection))
        log.info("Last detection info: {}".format(last_detection))

        first_detection = datetime.datetime.strptime(filenameToDatetimeStr(first_detection), "%Y-%m-%d %H:%M:%S.%f")
        last_detection = datetime.datetime.strptime(filenameToDatetimeStr(last_detection), "%Y-%m-%d %H:%M:%S.%f")

        log.info("First detection info: {}".format(first_detection))
        log.info("Last detection info: {}".format(last_detection))

        return str(first_detection.replace(microsecond=0)), str(last_detection.replace(microsecond=0))
    return '0', '0'

def getObservationSummaryDict(data_dir, final=False, config=None):
    """

    Arguments:
        data_dir: [path] to the data directory, if none, then the latest confirming directory in
        captured files is used.

    Keyword Arguments:
        final: [bool] Optional, default false, if true write to the final filename, rather than working, and delete
            working.
        config: [config] Optional, default None. If a config is passed, and data_dir is None, then attempt to guess
            the appropriate data_dir to use.

    Return:
        [dict]: Observation summary dict.
    """

    if data_dir is None and config is not None:
        p = Path(os.path.join(config.data_dir, config.captured_dir))
        regex = re.compile(rf"^{config.stationID}_[0-9]{{8}}_[0-9]{{6}}_[0-9]{{6}}$")

        if p.exists() and p.is_dir():

            candidate_dirs = [cd for cd in p.iterdir() if cd.is_dir() and regex.match(cd.name)]
            candidate_dirs.sort(key=lambda d: d.stat().st_ctime, reverse=True)
            if len(candidate_dirs):
                data_dir = str(candidate_dirs[0].resolve())
            else:
                log.warning("Found no matching captured dirs, unable to determine directory to use")
                return {}
        else:
            return {}

    json_name = OBSERVATION_SUMMARY_NAME_JSON if final else OBSERVATION_SUMMARY_WORKING_NAME_JSON

    observation_summary_json_path = os.path.join(data_dir, getRMSStyleFileName(data_dir, json_name))
    if os.path.exists(observation_summary_json_path):
        if os.path.isfile(observation_summary_json_path):
            with open(observation_summary_json_path, "r") as f:
                try:
                    d = json.load(f)

                    # A file containing e.g. the literal "null" parses fine but yields a
                    # non-dict (None); reject it so it is recovered like a corrupt file
                    # rather than being returned and crashing every downstream consumer.
                    if not isinstance(d, dict):
                        raise ValueError("observation summary JSON is not a dict (got {})".format(
                            type(d).__name__))

                    log.info(f"Loaded {os.path.basename(observation_summary_json_path)}")

                except:
                    # Don't silently delete - back up the unparseable/invalid file so data is
                    # not lost, then start fresh.
                    corrupt_path = observation_summary_json_path + ".corrupt"
                    try:
                        os.replace(observation_summary_json_path, corrupt_path)
                        log.warning("Could not parse {}; backed up to {}".format(
                            os.path.basename(observation_summary_json_path), os.path.basename(corrupt_path)))
                    except Exception as e:
                        log.warning("Could not parse {}; backup failed: {}".format(
                            os.path.basename(observation_summary_json_path), e))
                    d = {'night_data_dir': data_dir}
                    saveObservationSummaryDict(d, data_dir)

            return d

    log.info("Creating a new observation summary dictionary")
    d = {'night_data_dir': data_dir}
    saveObservationSummaryDict(d, data_dir)

    return d

def saveObservationSummaryDict(d, night_dir=None):
    """Save the observation summary dictionary as a json.

    The working JSON is read-modify-written by several processes (the capture child writes
    media_backend while the main process writes the start-of-session values, and Reprocess runs
    later). To avoid lost updates and torn reads (the failure class behind issue #882) this:
      - takes an exclusive file lock (POSIX; degrades gracefully where fcntl is unavailable),
      - merges the in-memory dict on top of whatever is already on disk (so concurrent writers
        adding distinct keys do not clobber each other),
      - writes to a temp file and atomically replaces the target.
    The caller's dict is updated in place to stay consistent with what was written.

    Arguments:
        d: Observation summary dict.

    Keyword arguments:
        night_dir: [path] optional, the night directory; if None it is taken from d['night_data_dir'].

    Return:
        Nothing
    """
    if night_dir is None:
        night_dir = d.get("night_data_dir")

    if night_dir is None:
        log.warning("saveObservationSummaryDict: no night_data_dir available; skipping save")
        return

    if not os.path.isdir(night_dir):
        log.warning("saveObservationSummaryDict: night_data_dir does not exist, skipping save: {}".format(night_dir))
        return

    observation_summary_json_path = os.path.join(night_dir, getRMSStyleFileName(night_dir, OBSERVATION_SUMMARY_WORKING_NAME_JSON))
    lock_path = observation_summary_json_path + ".lock"

    lock_f = open(lock_path, "w")
    try:
        if fcntl is not None:
            fcntl.flock(lock_f, fcntl.LOCK_EX)

        # Merge with whatever is already on disk; in-memory values win for the keys they set.
        merged = {}
        if os.path.isfile(observation_summary_json_path):
            try:
                with open(observation_summary_json_path, "r") as rf:
                    merged = json.load(rf)
            except Exception:
                merged = {}
        merged.update(d)

        # Keep the caller's dict consistent with what is persisted.
        d.clear()
        d.update(merged)

        tmp_path = observation_summary_json_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(merged, f, default=lambda o: o.__dict__, indent=4, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, observation_summary_json_path)

    except Exception as e:
        log.error("Saving observation summary working JSON failed: " + repr(e))
        log.error("".join(traceback.format_exception(*sys.exc_info())))

    finally:
        if fcntl is not None:
            try:
                fcntl.flock(lock_f, fcntl.LOCK_UN)
            except Exception:
                pass
        lock_f.close()

def startObservationSummaryReport(config, night_data_dir, duration, force_delete=False):
    """ Enters the parameters known at the start of observation into the database.

    Arguments:
        config: [config] config file.
        duration: [int]the initially calculated duration seconds.

    Keyword arguments:
        force_delete: [bool] forces deletion of the observation summary database, default False.

    Return:
        [str] message about session.

    """


    d = getObservationSummaryDict(night_data_dir)

    start_time_object = (datetime.datetime.now(datetime.timezone.utc) -
                         datetime.timedelta(seconds=1)).replace(tzinfo=datetime.timezone.utc)
    start_time_object_rounded = start_time_object.replace(microsecond=0)
    addObsParam(d, "start_time", start_time_object_rounded.isoformat())
    addObsParam(d, "duration_from_start_of_observation", duration)
    addObsParam(d, "stationID", sanitise(config.stationID, space_substitution=""))

    if isRaspberryPi():
        with open('/sys/firmware/devicetree/base/model', 'r') as m:
            hardware_version = sanitise(m.read().lower(), space_substitution=" ")
    else:
        hardware_version = sanitise(platform.machine(), space_substitution=" ")

    addObsParam(d, "hardware_version", hardware_version)

    try:
        repo_path = getRmsRootDir()
        repo = git.Repo(repo_path)
        if repo:
            addObsParam(d, "commit_date",
                        UTCFromTimestamp.utcfromtimestamp(repo.head.object.committed_date).strftime('%Y%m%d_%H%M%S'))
            addObsParam(d, "commit_hash", repo.head.object.hexsha)
        else:
            print("RMS Git repository not found. Skipping Git-related information.")
    except:
        print("Error getting Git information. Skipping Git-related information.")
    
    # Get the disk usage info (only in Python 3.3+) for the data_dir disc
    if (sys.version_info.major > 2) and (sys.version_info.minor > 2):

        try:
            storage_total, storage_used, storage_free = shutil.disk_usage(config.data_dir)
            addObsParam(d, "storage_total_gb", round(storage_total/(1024**3), 2))
            addObsParam(d, "storage_used_gb", round(storage_used/(1024**3), 2))
            addObsParam(d, "storage_free_gb", round(storage_free/(1024**3), 2))
        except:
            addObsParam(d, "storage_total_gb", "Not available")
            addObsParam(d, "storage_used_gb", "Not available")
            addObsParam(d, "storage_free_gb", "Not available")

    captured_directories = captureDirectories(os.path.join(config.data_dir, config.captured_dir), config.stationID)
    addObsParam(d, "captured_directories", captured_directories)
    try:
        sensor, firmware, build_date = gatherCameraInformation(config)
        addObsParam(d, "camera_information", sensor)
        addObsParam(d, "camera_firmware_version", firmware)
        addObsParam(d, "camera_firmware_build_date", build_date)
    except:
        addObsParam(d, "camera_information", "Unavailable")
        addObsParam(d, "camera_firmware_version", "Unavailable")
        addObsParam(d, "camera_firmware_build_date", "Unavailable")


    saveObservationSummaryDict(d)
    try:
        conn = getObsDBConn(config)
        storeDictInDB(conn, d, debug=False)
        conn.close()

    except Exception as e:
        log.error('Storing initial observation summary into database failed with error:' + repr(e))
        log.error("".join(traceback.format_exception(*sys.exc_info())))

    return "Opening a new observations summary"

def finalizeObservationSummary(config, night_data_dir, platepar=None):
    """Enters the parameters known at the end of observation into the database.

    Arguments:
        config: [config] config file.
        night_data_dir: [path] the directory of captured files.

    Keyword arguments:
        platepar: [object] optional, default None.

    Return:
        [str] filename of text file.
        [str] filename of json.

            """

    d = getObservationSummaryDict(night_data_dir)
    capture_duration_from_fits, start_ephem, capture_duration_from_ephemeris, end_ephem, \
    fits_count, \
    fits_file_shortfall, fits_file_shortfall_ephemeris, \
    fits_file_shortfall_as_time, fits_file_shortfall_as_time_ephemeris, \
    time_first_fits_file, time_last_fits_file, \
    total_expected_fits, total_expected_fits_ephemeris = nightSummaryData(config, night_data_dir)

    # Convert AU0004_20260612_100206_674582 into a python time object
    _, time_section = os.path.basename(d['night_data_dir']).split("_",maxsplit=1)
    session_start_time = datetime.datetime.strptime(time_section, "%Y%m%d_%H%M%S_%f").replace(tzinfo=datetime.timezone.utc)
    addObsParam(d, "traceback_count", countKeyStringsInLogs(session_start_time, config, key_string="Traceback (most recent call last)"))
    addObsParam(d, "kht_wrapper_count", countKeyStringsInLogs(session_start_time, config, key_string="undefined symbol: kht_wrapper"))

    try:
        timeSyncStatus(config, d)
    except Exception as e:
        print(repr(e))


    platepar_path = os.path.join(config.config_file_path, config.platepar_name)
    if os.path.exists(platepar_path):
        platepar = Platepar()
        platepar.read(platepar_path, use_flat=config.use_flat)
        addObsParam(d, "camera_pointing_az", format("{:.2f} degrees".format(platepar.az_centre)))
        addObsParam(d, "camera_pointing_alt", format("{:.2f} degrees".format(platepar.alt_centre)))
        addObsParam(d, "camera_fov_h", "{:.2f}".format(platepar.fov_h))
        addObsParam(d, "camera_fov_v", "{:.2f}".format(platepar.fov_v))
        addObsParam(d, "camera_lens", estimateLens(platepar.fov_h))

    addObsParam(d, "continuous_capture", config.continuous_capture)
    addObsParam(d, "time_start_ephem", start_ephem)
    addObsParam(d, "time_first_fits_file", time_first_fits_file)
    addObsParam(d, "time_end_ephem", end_ephem)
    addObsParam(d, "time_last_fits_file", time_last_fits_file)
    addObsParam(d, "capture_duration_from_fits", capture_duration_from_fits)
    addObsParam(d, "capture_duration_from_ephemeris", capture_duration_from_ephemeris)
    addObsParam(d, "total_expected_fits", round(total_expected_fits))
    addObsParam(d, "total_expected_fits_ephemeris", round(total_expected_fits_ephemeris))
    addObsParam(d, "total_fits", fits_count)
    addObsParam(d, "fits_file_shortfall", fits_file_shortfall)
    addObsParam(d, "fits_file_shortfall_ephemeris", fits_file_shortfall_ephemeris)
    addObsParam(d, "fits_file_shortfall_as_time", fits_file_shortfall_as_time)
    addObsParam(d, "fits_file_shortfall_as_time_ephemeris", fits_file_shortfall_as_time_ephemeris)
    addObsParam(d, "protocol_in_use", config.protocol)
    addObsParam(d, "star_catalog_file", config.star_catalog_file)

    try:
        first_detection, last_detection = getTimeOfFirstAndLastDetectionInDir(night_data_dir)
        addObsParam(d, "time_first_detection", first_detection)
        addObsParam(d, "time_last_detection", last_detection)
    except Exception as e:
        log.error('Storing first and last detections failed with error:' + repr(e))
        log.error("".join(traceback.format_exception(*sys.exc_info())))

    try:
        days_behind, remote_branch = daysBehind()
        addObsParam(d, "repository_lag_remote_days", days_behind)
        addObsParam(d, "remote_branch", os.path.basename(remote_branch))
    except:
        addObsParam(d, "repository_lag_remote_days", "Not determined")

    # Persist the values gathered so far so getDaysSinceLastDetection can query time_last_fits_file.
    try:
        conn = getObsDBConn(config, force_delete=False)
        storeDictInDB(conn, d, debug=False)
        conn.close()

    except Exception as e:
        log.error('Storing final observation summary into database failed with error:' + repr(e))
        log.error("".join(traceback.format_exception(*sys.exc_info())))


    addObsParam(d, 'days_since_last_detection', getDaysSinceLastDetection(config, night_data_dir, d=d))
    saveObservationSummaryDict(d)

    writeToFile(config, getRMSStyleFileName(night_data_dir, OBSERVATION_SUMMARY_NAME_TXT), night_data_dir)
    writeToJSON(config, getRMSStyleFileName(night_data_dir, OBSERVATION_SUMMARY_NAME_JSON), night_data_dir)
    writeToPNG(config, getRMSStyleFileName(night_data_dir, OBSERVATION_SUMMARY_NAME_PNG), night_data_dir)
    working_json_path = getRMSStyleFileName(night_data_dir, OBSERVATION_SUMMARY_WORKING_NAME_JSON)
    if os.path.exists(working_json_path):
        if os.path.isfile(working_json_path):
            os.unlink(working_json_path)

    try:
        conn = getObsDBConn(config, force_delete=False)
        storeDictInDB(conn, d, debug=False)
        conn.close()

    except Exception as e:
        log.error('Storing final observation summary into database failed with error:' + repr(e))
        log.error("".join(traceback.format_exception(*sys.exc_info())))


    return getRMSStyleFileName(night_data_dir, "observation_summary.txt"), \
                getRMSStyleFileName(night_data_dir, "observation_summary.json")

if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Test run observation summary.")

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
                            help="Path to a config file which will be used instead of the default one.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    RUNNING_FROM_CONSOLE = True

    # Load the config file

    config = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('.'))

    conn = getObsDBConn(config, force_delete=False)
    full_path_capture_directory = os.path.join(config.data_dir, config.captured_dir)
    d = getObservationSummaryDict(None, config=config)

    ftp_detect_info_file = None
    dir_list = os.listdir(full_path_capture_directory)
    dir_list.sort(reverse=True)
    for directory_to_search in dir_list:
        try:
            ftp_detect_info_file = findFTPdetectinfoFile(os.path.join(full_path_capture_directory, directory_to_search))
            break
        except:
            pass

    if ftp_detect_info_file is None:
        log.info("Unable to find a directory with a FTP file")
    else:
        log.info(f"Directory {directory_to_search} has a FTP file {ftp_detect_info_file}")

    capture_directory = directory_to_search

    latest_dir = os.path.join(full_path_capture_directory, capture_directory)
    print(f"Days since last detection {getDaysSinceLastDetection(config, latest_dir, debug=True)}")
    start_time, duration, end_time = getEphemTimesFromCaptureDirectory(config, latest_dir)
    print("For directory {}".format(latest_dir))
    print("Start time was {}".format(start_time))
    print("Duration time was {:.2f} hours".format(duration/3600))
    print("End time was {}".format(end_time))
    print(f"Days since last detection {getDaysSinceLastDetection(config, latest_dir, debug=True)}")
    try:
        print(getTimeOfFirstAndLastDetectionInDir(latest_dir))
    except:
        pass

    startObservationSummaryReport(config, latest_dir, duration, force_delete=False)
    pp = Platepar()
    finalizeObservationSummary(config, latest_dir , pp)
    print("Summary as colon delimited text")
    print(serialize(config, as_json=False, night_directory=latest_dir))
    print("Summary as json")
    obs_sum_json = serialize(config, as_json=True, night_directory=latest_dir, final=True)
    print(obs_sum_json)

