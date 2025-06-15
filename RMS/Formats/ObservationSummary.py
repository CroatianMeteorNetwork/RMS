# The MIT License

# Copyright (c) 2024

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


import sys
import os
import subprocess
from time import strftime

from RMS.Misc import niceFormat, isRaspberryPi, sanitise, getRMSStyleFileName, getRmsRootDir, UTCFromTimestamp
import re
import sqlite3
from RMS.ConfigReader import parse
from datetime import datetime, timezone
import platform
import git
import shutil

import glob
import json


from RMS.Formats.FFfits import filenameToDatetimeStr
import datetime
from RMS.Formats.Platepar import Platepar

if sys.version_info.major > 2:
    import dvrip as dvr
else:
    # Python2 compatible version
    import Utils.CameraControl27 as dvr

EM_RAISE = True

import socket
import struct
import sys
import time


def roundWithoutTrailingZero(value, no):

    """
    Given a float, round to specified number of decimal places, then remove trailing zeroes

    Args:
        value: float value
        no: number of decimal places to round to

    Returns:
        [string]: value rounded number of decimal places without trailing zero
    """

    value = round(value,no)
    return str("{0:g}".format(value))

def getTimeClient():

    """
    Attempt to identify which time service client, if any is providing sync service, or Not recognized.
    Aware of systemd-timesyncd, chronyd, ntpd

    Returns:
        [string]: Name of the time client

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
            pass  # Not active or not installed
    return "Not recognized"

def timeSyncStatus(config, conn, force_client=None):

    """
    Add timeSyncStatus to the database, and return howe far the machine clock
    is ahead of reference time in milliseconds

    Args:
        config:rms config file
        conn: database connection
        force_client: optional, string to force resolution by ntpd, chrony, or a query on a remote server

    Returns:
        time local clock ahead (+ve) milliseconds, or "Unknown" if delta cannot be determined
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
            local_time_query = (datetime.datetime.now(timezone.utc) - datetime.datetime(1970, 1, 1).replace(tzinfo=timezone.utc)).total_seconds()
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

    """
    Acquire the statistics of the ntp client.
    Tries to use ntpstat, if not available, falls back to ntpq, if not available returns Unknown

    Returns:
        [bool]: true if reported as synchronised
        [float]: uncertainty in milliseconds
        [str]: always Unknown, unable to discern actual time error using ntp tools

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

    """
        Acquire the statistics of the ntp client.
        Tries to use ntpstat, if not available, falls back to ntpq, if not available returns Unknown

        uncertainty implementation is taken from
        https://chrony-project.org/doc/3.3/chronyc.html

        Root dispersion

            This is the total dispersion accumulated through all the computers back to the
            stratum-1 computer from which the computer is ultimately synchronised. Dispersion is due
            to system clock resolution, statistical measurement variations, etc.

            An absolute bound on the computers clock accuracy (assuming the stratum-1 computer is correct) is given by:
            clock_error <= |system_time_offset| + root_dispersion + (0.5 * root_delay)

            This is very high at initial synchronisation, as root dispersion dominates.


        Returns:
            [bool]: true if reported as synchronised
            [float]: uncertainty in milliseconds
            [str]: time in milliseconds that computer clock is reported to be ahead of superior reference

        """

    synchronized = False
    system_time_offset, root_dispersion, root_delay = 0, 0, 0
    try:
        cmd = ["chronyc", "tracking"]
        lines = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip().splitlines()
        print(lines)
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

    """
    refer https://stackoverflow.com/questions/36500197/how-to-get-time-from-an-ntp-server
    and also https://github.com/CroatianMeteorNetwork/RMS/issues/624


    Args:
        addr: optional, address of ntp server to use

    Returns:
        [float]: time in seconds since epoch
        [float]: estimated network delay (average of outgoing and return legs)
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

        # for NTP the fractional seconds is a 32 bit counter
        fractional_second_factor = ( 1 / 2 ** 32)


        # unpack data
        remote_clock_time_receive_timestamp_seconds = struct.unpack('!12I', data)[8] - REF_TIME_1970
        remote_clock_time_receive_timestamp_fractional_seconds = struct.unpack('!12I', data)[9] * fractional_second_factor

        remote_clock_time_transmit_timestamp_seconds = struct.unpack('!12I', data)[10] - REF_TIME_1970
        remote_clock_time_transmit_timestamp_fractional_seconds = struct.unpack('!12I', data)[11] * fractional_second_factor

        remote_clock_time_receive_timestamp = remote_clock_time_receive_timestamp_seconds + remote_clock_time_receive_timestamp_fractional_seconds
        remote_clock_time_transmit_timestamp = remote_clock_time_transmit_timestamp_seconds + remote_clock_time_transmit_timestamp_fractional_seconds

        local_clock_measured_response_time = (local_clock_receive_timestamp - local_clock_transmit_timestamp)
        remote_clock_measured_processing_time = (remote_clock_time_transmit_timestamp - remote_clock_time_receive_timestamp)

        # print("Rx Fractional {}, Tx fractional {}".format(remote_clock_time_receive_timestamp_fractional_seconds, remote_clock_time_transmit_timestamp_fractional_seconds))
        # next calculation assumes that remote and local clock are running at identical rates
        estimated_network_delay = local_clock_measured_response_time - remote_clock_measured_processing_time
        if estimated_network_delay < 0:
            return None, None

        # now calculate estimated clock offsets
        clock_offset_out_leg = remote_clock_time_receive_timestamp - local_clock_transmit_timestamp
        clock_offset_return_leg = remote_clock_time_transmit_timestamp - local_clock_receive_timestamp
        estimated_offset = (clock_offset_out_leg + clock_offset_return_leg) / 2
        adjusted_time = remote_clock_time_transmit_timestamp + estimated_offset
        return adjusted_time, estimated_network_delay
    else:
        return None, None

def getObsDBConn(config, force_delete=False):
    """ Creates the Observation Summary database. Tries only once.

    arguments:
        config: config file
        force_delete: if set then deletes the database before recreating

    returns:
        conn: [connection] connection to database if success else None

    """

    # Create the Observation Summary database
    observation_records_db_path = os.path.join(config.data_dir,"observation.db")

    if force_delete:
        os.unlink(observation_records_db_path)

    if not os.path.exists(os.path.dirname(observation_records_db_path)):
        # Handle the very rare case where this could run before any observation sessions
        # and RMS_data does not exist
        os.makedirs(os.path.dirname(observation_records_db_path))

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

    """ Add a single key value pair into the database

            arguments:
                conn: the connection to the database
                key: the key for the value to be added
                value: the value to be added

            returns:
                conn: [connection] connection to database if success else None

            """

    print(key,value)
    sql_statement = ""
    sql_statement += "INSERT INTO records \n"
    sql_statement += "(\n"
    sql_statement += "TimeStamp, Key, Value \n"
    sql_statement += ")\n\n"

    sql_statement += "VALUES "
    sql_statement += "(                            \n"
    sql_statement += "CURRENT_TIMESTAMP,'{}','{}'   \n".format(key, value)
    sql_statement += ")"

    try:
        cursor = conn.cursor()
        cursor.execute(sql_statement)
        conn.commit()

    except:

        if EM_RAISE:
            raise

def estimateLens(fov_h):

    """ Estimate the focal length of the lens in use

        arguments:
                fov_h: horizontal field of view

        returns:
                an estimate of the focal length of the lens

    """

    lens_types = ["25mm", "16mm", "8mm", "6mm", "4mm"]
    lens_fov_h = [15, 30, 45, 60, 90]
    for type, fov in zip(lens_types, lens_fov_h):
        if fov_h < fov:
            return type
    return None

def getLastStartTime(conn):
    """ Query the database to discover the previous start time
        arguments:
                conn: connection to database

        returns:
                the previous start time

    """


    sql_statement = ""
    sql_statement += "SELECT Value from records \n"
    sql_statement += "      WHERE Key = 'start_time' \n"
    sql_statement += "      ORDER BY TimeStamp desc \n"

    result = conn.cursor().execute(sql_statement).fetchone()

    if result is None:
        sql_statement = ""
        sql_statement += "SELECT Timestamp from records \n"
        sql_statement += "  ORDER BY Timestamp asc \n"

        result =  conn.cursor().execute(sql_statement).fetchone()

    return result[0]

def gatherCameraInformation(config, attempts=6, delay=10, sock_timeout=3):

    """ Gather information about the sensor in use
        Retry the DVRIP handshake until it works, or we exhaust attempts.

                arguments:
                    config: config object
                    attempts: optional, default 6, number of attempts to connect
                    delay: optional, default 10, delay between attempts
                    sock_timeout: optional, default 3, socket timeout in seconds

                returns:
                    sensor type string

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
    """ Counts the captured directories

        arguments:
            captured_dir: path to the captured directories
            stationID: stationID to identify only relevant directories

        returns:
            conn: count of directories

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
    """ Calculate the summary data for the night. This is based on work by others
        and translated from the original source code

                arguments:
                    config: config file
                    night_data_dir: the directory of captured files


                returns:
                    capture_duration_from_fits: the duration from the start of first fits to the end of the last
                    fits_count: the count of *.fits files in the directory
                    fits_file_shortfall: the number of expected fits expected vs the number actually found
                    fits_file_shortfall_as_time: this shortfall expressed in seconds, never negative
                    time_first_fits_file: the time of the first fits file
                    time_last_fits_file: the time of the last fits file
                    total_expected_fits: the number of fits files expected

                """

    duration_one_fits_file = 256 / config.fps
    fits_files_list = glob.glob(os.path.join(night_data_dir, "*.fits"))
    fits_files_list.sort()
    fits_count = len(fits_files_list)
    if fits_count < 1:
        return 0,0,0,0,0,0,0

    time_first_fits_file = datetime.datetime.strptime(filenameToDatetimeStr(os.path.basename(fits_files_list[0])),
                                                      "%Y-%m-%d %H:%M:%S.%f")
    time_last_fits_file = datetime.datetime.strptime(filenameToDatetimeStr(
        os.path.basename(fits_files_list[-1])), "%Y-%m-%d %H:%M:%S.%f")
    capture_duration_from_fits = (time_last_fits_file - time_first_fits_file).total_seconds() + duration_one_fits_file
    total_expected_fits = round(capture_duration_from_fits / duration_one_fits_file)
    fits_file_shortfall = total_expected_fits - fits_count
    fits_file_shortfall = 0 if fits_file_shortfall < 1 else fits_file_shortfall
    fits_file_shortfall_as_time = str(datetime.timedelta(seconds=fits_file_shortfall * duration_one_fits_file))
    return capture_duration_from_fits, fits_count, fits_file_shortfall, fits_file_shortfall_as_time, \
                        time_first_fits_file, time_last_fits_file, total_expected_fits

def retrieveObservationData(conn, obs_start_time, ordering=None):
    """ Query the database to get the data more recent than the time passed in.
        Usually this will be the start of the most recent observation session
        If no ordering is passed, then a default ordering is returned

            arguments:
                    conn: connection to database
                    ordering: optional, default None, sequence to order the keys.

            returns:
                    key value pairs committed to the database since the obs_start_time
    """

    if ordering is None:
        # Be sure to add a comma after each list entry, IDE will not pick up this error as Python will concatenate
        # the two items into one.

        ordering = ['stationID',
                    'commit_date', 'commit_hash','media_backend',
                    'hardware_version',
                    'captured_directories',
                    'storage_used_gb', 'storage_free_gb', 'storage_total_gb',
                    'camera_lens','camera_fov_h','camera_fov_v',
                    'camera_pointing_alt','camera_pointing_az',
                    'camera_information',
                    'clock_measurement_source', 'clock_synchronized', 'clock_ahead_ms', 'clock_error_uncertainty_ms',
                    'start_time', 'duration', 'photometry_good',
                    'time_first_fits_file', 'time_last_fits_file', 'total_expected_fits','total_fits',
                    'fits_files_from_duration','fits_file_shortfall', 'fits_file_shortfall_as_time',
                    'capture_duration_from_fits',
                    'detections_after_ml',
                    'media_backend','jitter_quality','dropped_frame_rate']

    # use this print call to check the ordering
    #print("Ordering {}".format(ordering))

    sql_statement = ""
    sql_statement += "SELECT Key, Value from records \n"
    sql_statement += "           WHERE TimeStamp >= '{}' \n".format(obs_start_time)
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

    #print(sql_statement)

    return conn.cursor().execute(sql_statement).fetchall()

def serialize(config, format_nicely=True, as_json=False):
    """ Returns the data from the most recent observation session as either colon
        delimited text file, ar as a json
                arguments:
                        config: station config file
                        format_nicely: optional, default true, present the data with
                                        delimiter characters aligned
                        as_json: optional, default false, return the data as a json

                returns:
                        string of key value pairs committed to the database since the
                        start of the previous observation session
        """

    conn = getObsDBConn(config)
    data = retrieveObservationData(conn, getLastStartTime(conn))
    conn.close()

    if as_json:
        return json.dumps(dict(data), default=lambda o: o.__dict__, indent=4, sort_keys=True)

    output = ""
    for key,value in data:
        #does this look like a float
        if not re.match(r'^-?\d+(?:\.\d+)$', value) is None:
            # handle as float
            try:
                value_as_float = float(value)
                output += "{}:{:s} \n".format(key, roundWithoutTrailingZero(value_as_float, 3))
            except:
                pass
        else:
            try:
                # convert to a time
                time_object = time.strptime(value, "%Y-%m-%d %H:%M:%S.%f")
                value_as_time = strftime("%Y-%m-%d %H:%M:%S", time_object)
                output += "{}:{:s} \n".format(key, value_as_time)

            except:
                try:
                # convert to a time
                    time_object = time.strptime(value, "%H:%M:%S.%f")
                    value_as_time = strftime("%H:%M:%S", time_object)
                    output += "{}:{:s} \n".format(key, value_as_time)
                    # if it didn't work, then handle as a string
                except:
                    pass
                    try:
                        output += "{}:{:s} \n".format(key, value)
                    except:
                        # if we can't output as a string, then move on
                        pass

    if format_nicely:
        return niceFormat(output)


    return output

def writeToFile(config, file_path_and_name):
    """Write colon delimited text to file
                arguments:
                        config: station config file
                        file_path_and_name: full path to the target file

                returns:
                        string of key value pairs committed to the database since the
                        start of the previous observation session
        """


    with open(file_path_and_name, "w") as summary_file_handle:
        as_ascii = serialize(config).encode("ascii", errors="ignore").decode("ascii")
        summary_file_handle.write(as_ascii)


def writeToJSON(config, file_path_and_name):
    """Write as a json
                    arguments:
                            config: station config file
                            file_path_and_name: full path to the target file

                    returns:
                            string of key value pairs committed to the database since the
                            start of the previous observation session
            """
    with open(file_path_and_name, "w") as summary_file_handle:
        as_ascii = serialize(config, as_json=True).encode("ascii", errors="ignore").decode("ascii")
        summary_file_handle.write(as_ascii)

def startObservationSummaryReport(config, duration, force_delete=False):
    """ Enters the parameters known at the start of observation into the database

        arguments:
            config: config file
            duration: the initially calculated duration
            force_delete: forces deletion of the observation summary database, default False

        returns:
            conn: [connection] connection to database

        """


    conn = getObsDBConn(config, force_delete=force_delete)
    addObsParam(conn, "start_time", (datetime.datetime.now(timezone.utc) - datetime.timedelta(seconds=1)).replace(tzinfo=timezone.utc))
    addObsParam(conn, "duration", duration)
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
    
    # Get the disk usage info (only in Python 3.3+)
    if (sys.version_info.major > 2) and (sys.version_info.minor > 2):

        storage_total, storage_used, storage_free = shutil.disk_usage("/")
        addObsParam(conn, "storage_total_gb", round(storage_total/(1024**3), 2))
        addObsParam(conn, "storage_used_gb", round(storage_used/(1024**3), 2))
        addObsParam(conn, "storage_free_gb", round(storage_free/(1024**3), 2))

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

    if duration is None:
        fits_files_from_duration = "None (Continuous Capture)"
    else:
        fits_files_from_duration = duration*fps/no_of_frames_per_fits_file
    
    addObsParam(conn, "fits_files_from_duration", fits_files_from_duration)

    conn.close()

    return "Opening a new observations summary for duration {} seconds".format(duration)

def finalizeObservationSummary(config, night_data_dir, platepar=None):

    """ Enters the parameters known at the end of observation into the database

            arguments:
                config: config file
                night_data_dir: the directory of captured files
                platepar: optional, default None

            returns:
                conn: [connection] connection to database if success else None

            """

    capture_duration_from_fits, fits_count, fits_file_shortfall, fits_file_shortfall_as_time, time_first_fits_file, \
        time_last_fits_file, total_expected_fits = nightSummaryData(config, night_data_dir)



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
        addObsParam(obs_db_conn, "camera_fov_v","{:2f}".format(platepar.fov_v))
        addObsParam(obs_db_conn, "camera_lens", estimateLens(platepar.fov_h))


    addObsParam(obs_db_conn, "time_first_fits_file", time_first_fits_file)
    addObsParam(obs_db_conn, "time_last_fits_file", time_last_fits_file)
    addObsParam(obs_db_conn, "capture_duration_from_fits", capture_duration_from_fits)
    addObsParam(obs_db_conn, "total_expected_fits", round(total_expected_fits))
    addObsParam(obs_db_conn, "total_fits", fits_count)
    addObsParam(obs_db_conn, "fits_file_shortfall", fits_file_shortfall)
    addObsParam(obs_db_conn, "fits_file_shortfall_as_time", fits_file_shortfall_as_time)
    obs_db_conn.close()

    writeToFile(config, getRMSStyleFileName(night_data_dir, "observation_summary.txt"))
    writeToJSON(config, getRMSStyleFileName(night_data_dir, "observation_summary.json"))

    return getRMSStyleFileName(night_data_dir, "observation_summary.txt"), \
                getRMSStyleFileName(night_data_dir, "observation_summary.json")





if __name__ == "__main__":

    config = parse(os.path.expanduser("~/source/RMS/.config"))

    obs_db_conn = getObsDBConn(config)
    print(timeSyncStatus(config, obs_db_conn))
    startObservationSummaryReport(config, 100, force_delete=False)
    pp = Platepar()
    pp.read(os.path.expanduser(os.path.join(config.rms_root_dir, "platepar_cmn2010.cal")))
    night_data_dir = os.path.join(config.data_dir, config.captured_dir)
    target = os.path.join(night_data_dir, os.listdir(night_data_dir)[-1])
    finalizeObservationSummary(config, target , pp)
    output_directory = os.path.join(os.path.expanduser(config.data_dir), os.listdir(night_data_dir)[-1])
    writeToFile(config, output_directory)
    writeToJSON(config, output_directory)
    print("Summary as colon delimited text")
    print(serialize(config, as_json=False))
    print("Summary as json")
    print(serialize(config, as_json=True))