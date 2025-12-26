# RPi Meteor Station
# Copyright (C) 2025 David Rollinson
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function, absolute_import

import os
import json
import tempfile
import argparse
import datetime
import shutil
import RMS.ConfigReader as cr
from RMS.Logger import getLogger


LATEST_LOG_UPLOADS_FILE_NAME = ".latestloguploads.json"
log = getLogger("logger")


def getTimeOfLastLogEntry(config, log_file):
    """Get the time in python date time format of the last entry in a log file

    Args:
        log_file_path: [path] path to the log file

    Returns:
        [datetime.datetime] time of the last entry in the log file
    """

    log_file_path = os.path.join(config.data_dir, config.log_dir, log_file)

    # It is quicker to open the whole file using readlines, and pick the last line; but this uses less memory
    with open(log_file_path, "r") as f:
        line = ""
        for line in f:
            pass
    last_log_time_string = line.split("-")[0]
    if last_log_time_string:
        try:
            last_log_entry_time_object = datetime.datetime.strptime(last_log_time_string, "%Y/%m/%d %H:%M:%S")
        except:
            last_log_entry_time_object = datetime.datetime.strptime("2000/01/01 00:00", "%Y/%m/%d %H:%M")
    else:
        last_log_entry_time_object = datetime.datetime.strptime("2000/01/01 00:00", "%Y/%m/%d %H:%M")

    return  last_log_entry_time_object



def getLogTypes(config):
    """ Scan the log directory given in the config file, return a list of unique log types
    sorted alphabetically.

    Arguments:
        config: [config] RMS config instance

    Return:
        [list] list of log file types
    """
    log_dir = os.path.join(config.data_dir, config.log_dir)
    log_file_list, log_types_list = os.listdir(log_dir), []
    unique_log_type_set  = {item.split("_")[0] for item in log_file_list if item.endswith(".log")}
    for log_type in unique_log_type_set:
        log.info(f"                 : {log_type}")

    return sorted(list(unique_log_type_set))

def getLogFileListOfLists(config, reverse=False):
    """ Scan the log directory given in the config file, and return a list of
    lists of log files. The top level list is ordered by log type,
    the next level of list, is by log name.

    Arguments:
        config: [config] RMS config instance

    Keyword Arguments:
        reverse: [bool] Reverse the order of the lists, default False, so newest is at the top

    Return:
        [[list]] list of lists of log file paths
    """

    log_types_list = getLogTypes(config)
    log_file_list_of_lists = []
    log_dir_list = sorted(os.listdir(os.path.join(config.data_dir, config.log_dir)))
    for log_type in log_types_list:
        log_list_by_type = [log_name for log_name in log_dir_list if log_name.split("_")[0] == log_type]
        if reverse:
            log_file_list_of_lists.append(sorted(log_list_by_type, reverse=True))
        else:
            log_file_list_of_lists.append(sorted(log_list_by_type, reverse=False))

    return log_file_list_of_lists

def getEarliestDates(config):
    """Scan the log file directory given in the config file, and return a list of
    the earliest dates of each type of log

    Arguments:
        config: [config] RMS config instance

    Return:
        [list] of dates of each log type, ordered alphabetically by log type
    """

    log_file_list_of_lists = getLogFileListOfLists(config)
    earliest_dates = []
    for log_file_list in log_file_list_of_lists:
        earliest_dates.append(extractDateFromLogName(config, log_file_list[0]))
    return earliest_dates


def extractDateFromLogName(config, log_name):
    """Extracts date and time from a log filename

    Arguments:
        log_name: [str] name of log file

    Return:
        [str] iosoformat date and time extracted from log name
    """


    log_name_fields = log_name.split("_")
    next_field_is_date, next_field_is_time = False, False

    year, month, day, hour, minute, second = 2000,1,1,0,0,0

    for field in log_name_fields:
        if config.stationID.upper() == field.upper():
            next_field_is_date = True
            continue
        if next_field_is_date and len(field) == 8:
            year, month, day = field[0:4], field[4:6], field[6:8]
            next_field_is_time, next_field_is_date = True, False
            continue
        if next_field_is_time and len(field) >= 6:
            hour, minute, second = field[0:2], field[2:4], field[4:6]
            next_field_is_time = False
            break
    return datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second)).isoformat()


def getLatestLogUploads(config):
    """If a json file exists with the latest log uploads, then read it and return a dict
    keys are log types, values are latest log file uploaded.

    If the file does not exist, find the earliest logs of each type, create a dictionary,
    and save as a json

    Arguments:
        config: [config] RMS config instance

    Return:
        [dict] dictionary of latest upload dates
    """
    latest_log_uploads_file_full_path = os.path.join(config.data_dir, config.log_dir, LATEST_LOG_UPLOADS_FILE_NAME)

    if os.path.exists(latest_log_uploads_file_full_path):
        if os.path.isfile(latest_log_uploads_file_full_path):
            with open(latest_log_uploads_file_full_path, "r") as f:
                try:
                    latest_log_uploads_dict = json.load(f)
                except:
                    latest_log_uploads_dict = dict(zip(getLogTypes(config), getEarliestDates(config)))

    else:
        # create a dictionary
        latest_log_uploads_dict  = dict(zip(getLogTypes(config), getEarliestDates(config)))
        with open(latest_log_uploads_file_full_path, "w") as f:
            json.dump(latest_log_uploads_dict, f, indent=4, sort_keys=True)

    return latest_log_uploads_dict

def makeLogArchives(config, dest_dir, update_tracker=True):
    """Given a config file and a destination directory, get the logs later
    than the logs last uploaded, package into a bz2 file, with each log type
    in a separate directory, archive to bz2, and copy to a destination directory


    Arguments:
        config: [config] RMS config instance
        dest_dir: [path] full path to the destination directory, normally the latest captured directory

    Keyword Arguments:
        update_tracker: [bool] Whether or not to update the tracker (default: {True})


    Return:
        [path] : path to created archive

    """

    if not update_tracker:
        log.info("Running without updating tracker file")
    latest_log_uploads_dict = getLatestLogUploads(config)
    for log_type, latest_date in latest_log_uploads_dict.items():
        log.info(f"Log type {log_type} latest info uploaded was {latest_date}")

    log_list_of_lists = getLogFileListOfLists(config, reverse=True)
    log_type_list = getLogTypes(config)

    logs_to_send_by_type = []
    for log_file_type, log_list in zip(log_type_list, log_list_of_lists):
        logs_to_send = []
        date_for_this_log_type = datetime.datetime.fromisoformat(latest_log_uploads_dict[log_file_type])

        pass
        for log_name in log_list:
            date_for_this_log_file = datetime.datetime.fromisoformat(extractDateFromLogName(config, log_name))
            if date_for_this_log_file < date_for_this_log_type:
                continue
            else:
                if date_for_this_log_file == date_for_this_log_type:
                    log.info(f"Adding {log_name} to log upload archive to ensure overlap with last upload")
                else:
                    log.info(f"Adding {log_name} to log upload archive as it is newer than last upload")
                logs_to_send.append(log_name)
        logs_to_send_by_type.append(logs_to_send)


    with tempfile.TemporaryDirectory() as temp_dir:
        os.mkdir(os.path.join(temp_dir, "logs"))
        for log_file_type, log_file_list in zip(log_type_list, logs_to_send_by_type):
            os.mkdir(os.path.join(temp_dir, "logs", log_file_type))
            for log_file in sorted(log_file_list):
                source_file_path = os.path.join(config.data_dir, config.log_dir, log_file)
                if os.path.exists(source_file_path):
                    shutil.copy(os.path.join(source_file_path), os.path.join(temp_dir, "logs", log_file_type))
                else:
                    log.warning(f"Could not find log file in {source_file_path}")
                pass
        log.info(f"Log directory structure created at {temp_dir}")

        archive_filename = shutil.make_archive(os.path.join(dest_dir, f"{os.path.basename(dest_dir)}_logs"), 'bztar', root_dir=temp_dir, base_dir=".")

    # Now update the dictionary of last times

    latest_log_uploads_file_full_path = os.path.join(config.data_dir, config.log_dir, LATEST_LOG_UPLOADS_FILE_NAME)

    log_file_type_list, last_log_file_timestamp_list = [], []
    for log_file_type, log_file_list in zip(log_type_list, log_list_of_lists):
        newest_log_file_for_this_type = sorted(log_file_list, reverse=True)[0]

        last_log_entry_time = getTimeOfLastLogEntry(config, newest_log_file_for_this_type)
        log.info(f"For log file type {log_file_type} the newest file is {newest_log_file_for_this_type}, last entry is {last_log_entry_time}")
        last_log_file_timestamp = extractDateFromLogName(config, newest_log_file_for_this_type)
        log_file_type_list.append(log_file_type)
        last_log_file_timestamp_list.append(last_log_file_timestamp)

    latest_log_uploads_dict = dict(zip(log_file_type_list, last_log_file_timestamp_list))
    with open(latest_log_uploads_file_full_path, "w") as f:
        json.dump(latest_log_uploads_dict, f, indent=4, sort_keys=True)

    return archive_filename


if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description=""" Generate an archive of log files """)

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one.")

    arg_parser.add_argument('-t', '--tracker', action="store_true", help="""Update upload tracker file, 
     default false when run from console""")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('.'))

    # Get the logger handle
    log = getLogger("rmslogger")
    p = os.path.join(config.data_dir, config.captured_dir)

    # Get the latest captured directory, this is only for testing purposes
    newest_dir = sorted([name for name in os.listdir(p) if os.path.isdir(os.path.join(p, name))], reverse=True)[0]
    latest_captured_directory_full_path = os.path.join(p, newest_dir)

    # Run a test on making log Archives
    archive_file_name = makeLogArchives(config, latest_captured_directory_full_path, update_tracker=cml_args.tracker)
    log.info(f"Logs archived at {archive_file_name}")


