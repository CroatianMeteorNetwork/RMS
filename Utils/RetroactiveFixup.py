""" Gives a path to the station directory and a date, copy a config file from the night folder closest to the
given date to all previous night folders. 
"""

from __future__ import print_function, division, absolute_import

import os
import sys
import shutil
import datetime

from RMS.Astrometry.ApplyRecalibrate import applyRecalibrate
from RMS.ConfigReader import parse as parseConfig


CONFIG_NAME = ".config"
CONFIG_BAK = "bak.config"


if __name__ == "__main__":

    import argparse


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Copy .config file from directory closest in time to the reference time to all older data directories and recalibrate the data.")

    arg_parser.add_argument('station_path', metavar='DIR_PATH', type=str, \
        help="Path to the station directory.")

    arg_parser.add_argument('date', metavar='DATE', type=str, \
        help="Reference date in the YYYYMMDD format.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    if not os.path.exists(cml_args.station_path):
        print("Directory does not exist:", cml_args.station_path)
        print("Exiting...")
        sys.exit()


    # Parse the reference date into a datetime object
    dt_ref = datetime.datetime.strptime(cml_args.date, "%Y%m%d")


    # Create a list of directory paths and their parsed dates
    dir_dt_list = []
    for dir_name in sorted(os.listdir(cml_args.station_path)):

        dir_path = os.path.join(cml_args.station_path, dir_name)

        # Only take directories
        if not os.path.isdir(dir_path):
            continue


        # Parse the date, e.g. HR000Q_20190814_184253_344231_detected
        date_str = dir_name.split("_")[1]
        dt = datetime.datetime.strptime(date_str, "%Y%m%d")

        dir_dt_list.append([dir_name, dt])


    # Find the directory closest to the reference date
    dt_diff_list = [abs((dt - dt_ref).total_seconds()) for _, dt in dir_dt_list]
    ref_dir = dir_dt_list[dt_diff_list.index(min(dt_diff_list))][0]


    # Check if the config file exists in the refernce directory
    ref_config_path = os.path.join(cml_args.station_path, ref_dir, CONFIG_NAME)
    if not os.path.exists(ref_config_path):
        print("No config file in the reference directory:", ref_config_path)
        print("Exiting...")
        sys.exit()


    # Go through all directories and copy the config file from the reference directory into them
    for dir_name in sorted(os.listdir(cml_args.station_path)):

        dir_path = os.path.join(cml_args.station_path, dir_name)

        # Only take directories
        if not os.path.isdir(dir_path):
            continue

        # Stop if the reference directory has been reached
        if dir_name == ref_dir:
            break


        config_path = os.path.join(dir_path, CONFIG_NAME)
        if not os.path.exists(config_path):
            print("No config file in {:s}, skipping...".format(dir_name))
            continue

        # Back up the config file unless it already exists
        config_bak_path = os.path.join(dir_path, CONFIG_BAK)
        if not os.path.exists(config_bak_path):
            shutil.copy2(config_path, config_bak_path)

        # Copy the refernce config file in the place of the existing config file
        shutil.copy2(ref_config_path, config_path)

        print("Replaced config in:", dir_name)



        ### Recalibrate the FTPdetectinfo ###

        # Find the FTPdetectinfo file in the directory
        ftpdetectinfo_name = None
        for file_name in sorted(os.listdir(dir_path)):

            # Find FTPdetectinfo
            if file_name.startswith("FTPdetectinfo") and file_name.endswith('.txt') and \
                (not "backup" in file_name) and (not "uncalibrated" in file_name):
                ftpdetectinfo_name = file_name
                break

        
        # Skip directories which have no FTPdetectinfo files
        if ftpdetectinfo_name is None:
            print("No FTPdetectinfo file found in:", dir_name)
            print("Skipping...")
            continue


        # Load the config file
        config = parseConfig(config_path)

        # Run the recalibration
        applyRecalibrate(os.path.join(dir_path, ftpdetectinfo_name), config)

        ### ###

    




