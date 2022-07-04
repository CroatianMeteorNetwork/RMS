""" Automatically runs the flux code and produces graphs on available data from multiple stations. """


import datetime


from RMS.Formats.Showers import FluxShowers


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


    # Load all data folders

    # Determine which data folders should be used for each shower


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
