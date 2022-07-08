""" Automatically runs the flux code and produces graphs on available data from multiple stations. """

import os

import datetime

import numpy as np

from RMS.Astrometry.Conversions import datetime2JD
from RMS.Formats.Showers import FluxShowers
from RMS.Math import isAngleBetween
from RMS.Routines.SolarLongitude import jd2SolLonSteyaert


def fluxAutoRun(config, data_path, ref_time, days_prev=2, days_next=1):
    """

    Arguments:
        config: [Config]
        data_path: [str] Path to the directory with the data used for flux computation.
        ref_file: [datetime] Reference time to compute the flux for all active showers. E.g. this can be now,
            or some manually specified point in time.

    Keyword arguments:
        days_prev: [int] Produce graphs for showers active N days before.
        days_next: [int] Produce graphs for showers active N days in the future.

    """


    # Load the showers for flux
    flux_showers = FluxShowers(config)


    # Determine the time range for shower activity check
    dt_beg = ref_time - datetime.timedelta(days=days_prev)
    dt_end = ref_time + datetime.timedelta(days=days_next)

    # Get a list of showers active now
    active_showers = flux_showers.activeShowers(dt_beg, dt_end)

    print([shower.name for shower in active_showers])


    ### Load all data folders ###

    # Determine which data folders should be used for each shower
    shower_dirs = {}
    for entry in os.walk(data_path):

        dir_path, _, file_list = entry

        print("Inspecting:", dir_path)

        # Check that the dir name is long enough to contain the station code and the timestamp
        if len(dir_path) < 23:
            continue

        # Parse the timestamp from the directory name and determine the capture date
        dir_split = os.path.basename(dir_path).split("_")
        if len(dir_split) < 3:
            continue

        try:
            dir_dt = datetime.datetime.strptime(dir_split[1] + "_" + dir_split[2], "%Y%m%d_%H%M%S")
        except ValueError:
            continue

        # Make sure the directory time is after 2018 (to avoid 1970 unix time 0 dirs)
        #   2018 is when the GMN was established
        if dir_dt.year < 2018:
            continue

        # Compute the solar longitude of the directory time stamp
        sol_dir = jd2SolLonSteyaert(datetime2JD(dir_dt))

        # Go through all showers and take the appropriate directories
        for shower in active_showers:

            # Add a list for dirs for this shower, if it doesn't exist
            if shower.name not in shower_dirs:
                shower_dirs[shower.name] = []

            # Check that the directory time is within the activity period of the shower (+/- 1 deg sol)
            if isAngleBetween(np.radians(shower.lasun_beg - 1), sol_dir, np.radians(shower.lasun_end + 1)):

                # Take the folder only if it has a platepar file inside it
                if len([file_name for file_name in file_list if file_name == config.platepar_name]):
                    shower_dirs[shower.name].append(dir_path)


    ### ###


    # Process fluxes of active showers
    #   - combine data from previous years, plot this year data separately



if __name__ == "__main__":

    import argparse

    import RMS.ConfigReader as cr

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser

    arg_parser = argparse.ArgumentParser(description="Compute single-station meteor shower flux.")

    arg_parser.add_argument('dir_path', metavar='DIR_PATH', type=str,
        help="Path to the directory with the data used for flux. The directories can either be flat, or "
        "organized in STATIONID/NIGHTDIR structure.")

    arg_parser.add_argument('-t', '--time', nargs=1, metavar='TIME', type=str,
        help="Give the time in the YYYYMMDD_hhmmss.uuuuuu format at which the flux will be computed (instead of now).")

    # arg_parser.add_argument(
    #     "-c",
    #     "--config",
    #     nargs=1,
    #     metavar="CONFIG_PATH",
    #     type=str,
    #     default='.',
    #     help="Path to a config file which will be used instead of the default one."
    #     " To load the .config file in the given data directory, write '.' (dot).",
    # )

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    # Load the default config file
    config = cr.Config()
    config = cr.parse(config.config_file_name)

    if cml_args.time is not None:
        ref_time = datetime.datetime.strptime(cml_args.time[0], "%Y%m%d_%H%M%S.%f")

    # If no manual time was given, use current time.
    else:
        ref_time = datetime.datetime.utcnow()


    print("Computing flux using reference time:", ref_time)

    # Run auto flux
    fluxAutoRun(config, cml_args.dir_path, ref_time)
