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



import sys
import os
from RMS.Misc import niceFormat, isRaspberryPi, sanitise, getRMSStyleFileName
import re
import sqlite3
from RMS.ConfigReader import parse
from datetime import datetime
import platform
import git
import shutil
import time
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

EM_RAISE = False




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

    for items in os.listdir(captured_dir):
        if items[0:len(stationID)] == stationID:
            capture_directories += 1

    return capture_directories


def startObservationSummaryReport(config, duration, force_delete = False):
    """ Enters the parameters known at the start of observation into the database

        arguments:
            config: config file
            duration: the initially calculated duration
            force_delete: forces deletion of the obseration summary database, default False

        returns:
            conn: [connection] connection to database if success else None

        """


    conn = getObsDBConn(config, force_delete=force_delete)
    addObsParam(conn, "StartTime", datetime.datetime.utcnow() - datetime.timedelta(seconds=1))
    addObsParam(conn, "Duration", duration)

    if isRaspberryPi():
        with open('/sys/firmware/devicetree/base/model', 'r') as m:
            hardware_version = sanitise(m.read().lower(), space_substitution=" ")
    else:
        hardware_version = sanitise(platform.machine(), space_substitution=" ")

    addObsParam(conn, "hardware_version", hardware_version)

    repo = git.Repo(search_parent_directories=True)
    addObsParam(conn, "commit_date",
                datetime.datetime.fromtimestamp(repo.head.object.committed_date).strftime('%Y%m%d_%H%M%S'))

    repo = git.Repo(search_parent_directories=True)
    addObsParam(conn, "commit_hash", repo.head.object.hexsha)

    storage_total, storage_used, storage_free = shutil.disk_usage("/")
    addObsParam(conn, "storage_total_gb", round(storage_total / (1024 **3 ),2))
    addObsParam(conn, "storage_used_gb", round(storage_used / (1024 **3 ),2))
    addObsParam(conn, "storage_free_gb", round(storage_free / (1024 **3 ),2))

    captured_directories = captureDirectories(os.path.join(config.data_dir, config.captured_dir), config.stationID)
    addObsParam(conn, "captured_directories", captured_directories)
    camera_informatoin = gatherCameraInformation(config)

    no_of_frames_per_fits_file = 256
    fps = config.fps
    fits_files_from_duration = duration * fps / no_of_frames_per_fits_file
    addObsParam(conn, "fits_files_from_duration", fits_files_from_duration)
    conn.close()
    return "Opening a new observations summary for duration {}".format(duration)

def finalizeObservationSummary(config, night_data_dir, platepar=None):

    """ Enters the parameters known at the end of observation into the database

            arguments:
                config: config file
                night_data_dir: the directory of captured files
                force_delete: forces deletion of the obseration summary database, default False

            returns:
                conn: [connection] connection to database if success else None

            """

    capture_duration_from_fits, fits_count, fits_file_shortfall, fits_file_shortfall_as_time, time_first_fits_file, \
        time_last_fits_file, total_expected_fits = nightSummaryData(config, night_data_dir)


    obs_db_conn = getObsDBConn(config)
    if not platepar is None:

        addObsParam(obs_db_conn, "camera_pointing_az", format("{:.2f}".format(platepar.az_centre)))
        addObsParam(obs_db_conn, "camera_pointing_alt", format("{:.2f} degrees".format(platepar.alt_centre)))
        addObsParam(obs_db_conn, "camera_fov_h","{:.2f}".format(platepar.fov_h))
        addObsParam(obs_db_conn, "camera_fov_v","{:2f}".format(platepar.fov_v))
        addObsParam(obs_db_conn, "lens", estimateLens(platepar.fov_h))


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





def nightSummaryData(config, night_data_dir):

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
    capture_duration_from_fits = (time_last_fits_file - time_first_fits_file).total_seconds()
    total_expected_fits = round(capture_duration_from_fits / duration_one_fits_file)
    fits_file_shortfall = total_expected_fits - fits_count
    fits_file_shortfall_as_time = str(datetime.timedelta(seconds=fits_file_shortfall * duration_one_fits_file))
    return capture_duration_from_fits, fits_count, fits_file_shortfall, fits_file_shortfall_as_time, \
                        time_first_fits_file, time_last_fits_file, total_expected_fits


def addObsParam(conn, key, value):

    """ Add a single key value pair into the database

            arguments:
                conn: the connnection to the database
                key: the key for the value to be added
                value: the value to be added

            returns:
                conn: [connection] connection to database if success else None

            """

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



def gatherCameraInformation(config):

    """ Gather information about the sensor in use

                arguments:
                    config: config object

                returns:
                    sensor type string

                """

    try:
        cam = dvr.DVRIPCam(re.findall(r"[0-9]+(?:\.[0-9]+){3}", config.deviceID)[0])
        if cam.login():
            sensor_type = cam.get_upgrade_info()['Hardware']
        else:
            sensor_type = "Unable to login"
    except:
        sensor_type = "Error"

    return sensor_type

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

    sql_statement = ""
    sql_statement += "SELECT Value from records \n"
    sql_statement += "      WHERE Key = 'StartTime' \n"
    sql_statement += "      ORDER BY TimeStamp desc \n"

    result = conn.cursor().execute(sql_statement).fetchone()

    if result is None:
        sql_statement = ""
        sql_statement += "SELECT Timestamp from records \n"
        sql_statement += "  ORDER BY Timestamp asc \n"

        result =  conn.cursor().execute(sql_statement).fetchone()

    return result[0]

def retrieveObservationData(conn, obs_start_time):

    sql_statement = ""
    sql_statement += "SELECT Key, Value from records \n"
    sql_statement += "           WHERE TimeStamp >= '{}' \n".format(obs_start_time)
    sql_statement += "           GROUP BY Key \n"

    return conn.cursor().execute(sql_statement).fetchall()


def serialize(config, format_nicely=True, consider_security=True, utf8=True, as_json=False):

    conn = getObsDBConn(config)
    data = retrieveObservationData(conn, getLastStartTime(conn))
    conn.close()

    if as_json:
        return json.dumps(dict(data), default=lambda o: o.__dict__, indent=4, sort_keys=True)

    output = ""
    for key,value in data:
        output += "{}:{} \n".format(key,value)

    if format_nicely:
        return niceFormat(output)


    return output


def writeToFile(config, file_path_and_name):
    with open(file_path_and_name, "w") as summary_file_handle:
        as_ascii = serialize(config).encode("ascii", errors="ignore").decode("ascii")
        summary_file_handle.write(as_ascii)


def writeToJSON(config, file_path_and_name):
    with open(file_path_and_name, "w") as summary_file_handle:
        as_ascii = serialize(config, as_json=True).encode("ascii", errors="ignore").decode("ascii")
        summary_file_handle.write(as_ascii)




def readFromFile(self, file_path_and_name):
    # todo: this is required at server end
    string = ""
    return string


def unserialize(self):
    # todo: this is required at server end
    return self


if __name__ == "__main__":
    config = parse("/home/david/source/RMS/.config")

    if False:
        obs_db_conn = getObsDBConn(config)
        startObservationSummaryReport(config, 100, force_delete=False)
        night_data_dir = "/home/david/RMS_data/CapturedFiles/AU0004_20240810_185301_776066"
        pp = Platepar()
        pp.read("/home/david/source/RMS/platepar_cmn2010.cal")
        finalizeObservationSummary(config, night_data_dir, pp)
        writeToFile(config, "/home/david/RMS_data/observation_summary.txt")
    print(serialize(config, as_json=True))
    writeToJSON(config, "/home/david/RMS_data/summary.json")
