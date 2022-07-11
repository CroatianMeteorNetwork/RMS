""" Automatically runs the flux code and produces graphs on available data from multiple stations. """

import os

import datetime

import numpy as np

from RMS.Astrometry.Conversions import datetime2JD
from RMS.Formats.Showers import FluxShowers
from RMS.Math import isAngleBetween
from RMS.Routines.SolarLongitude import jd2SolLonSteyaert
from Utils.FluxBatch import fluxBatch, plotBatchFlux, FluxBatchBinningParams, saveBatchFluxCSV


def fluxAutoRun(config, data_path, ref_dt, days_prev=2, days_next=1, metadata_dir=None, output_dir=None):
    """ Given the reference time, automatically identify active showers and produce the flux graphs and
        CSV files.

    Arguments:
        config: [Config]
        data_path: [str] Path to the directory with the data used for flux computation.
        ref_dt: [datetime] Reference time to compute the flux for all active showers. E.g. this can be now,
            or some manually specified point in time.

    Keyword arguments:
        days_prev: [int] Produce graphs for showers active N days before.
        days_next: [int] Produce graphs for showers active N days in the future.
        metadata_dir: [str] A separate directory for flux metadata. If not given, the data directory will be
            used.
        output_dir: [str] Directory where the final data products will be saved. If None, data_path directory
            will be used.
    """


    if output_dir is None:
        output_dir = data_path


    # Load the showers for flux
    flux_showers = FluxShowers(config)

    # Compute the solar longitude of the reference time
    sol_ref = np.degrees(jd2SolLonSteyaert(datetime2JD(ref_dt)))


    # Determine the time range for shower activity check
    dt_beg = ref_dt - datetime.timedelta(days=days_prev)
    dt_end = ref_dt + datetime.timedelta(days=days_next)

    # Get a list of showers active now
    active_showers = flux_showers.activeShowers(dt_beg, dt_end, use_zhr_threshold=False)
    active_showers_dict = {shower.name:shower for shower in active_showers}
    print([shower.name for shower in active_showers])


    # Compute the range of dates for this year's activity of every active shower
    for shower in active_showers:

        # Compute the date range for this year's activity
        sol_diff_beg = abs((shower.lasun_beg - sol_ref + 180)%360 - 180)
        sol_diff_end = abs((sol_ref - shower.lasun_end + 180)%360 - 180)
        sol_diff_max = (shower.lasun_max - sol_ref + 180)%360 - 180

        # Add activity during the given year
        shower.dt_beg_ref_year = ref_dt - datetime.timedelta(days=sol_diff_beg*360/365.24219)
        shower.dt_end_ref_year = ref_dt + datetime.timedelta(days=sol_diff_end*360/365.24219)
        shower.dt_max_ref_year = ref_dt + datetime.timedelta(days=sol_diff_max*360/365.24219)


    ### Load all data folders ###

    # Determine which data folders should be used for each shower
    shower_dirs = {}
    shower_dirs_ref_year = {}
    for entry in sorted(os.walk(data_path)):

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
                shower_dirs_ref_year[shower.name] = []

            # Check that the directory time is within the activity period of the shower (+/- 1 deg sol)
            if isAngleBetween(np.radians(shower.lasun_beg - 1), sol_dir, np.radians(shower.lasun_end + 1)):

                # Take the folder only if it has a platepar file inside it
                if len([file_name for file_name in file_list if file_name == config.platepar_name]):
                    shower_dirs[shower.name].append(dir_path)


                    print("Ref year check:")
                    print(dir_dt, shower.dt_beg_ref_year - datetime.timedelta(days=1)) 
                    print(dir_dt, shower.dt_end_ref_year + datetime.timedelta(days=1))
                    print()

                    # Store only the given year's directories, to generate the plot of the latest activity
                    if (dir_dt >= shower.dt_beg_ref_year - datetime.timedelta(days=1)) and \
                       (dir_dt <= shower.dt_end_ref_year + datetime.timedelta(days=1)):

                       shower_dirs_ref_year[shower.name].append(dir_path)


    ### ###

    # Define binning parameters for all years, and individual years
    fluxbatch_binning_params_all_years = FluxBatchBinningParams(
        min_meteors=100, 
        min_tap=20, 
        min_bin_duration=0.5, 
        max_bin_duration=24
        )

    fluxbatch_binning_params_one_year = FluxBatchBinningParams(
        min_meteors=20,
        min_tap=5, 
        min_bin_duration=0.5, 
        max_bin_duration=12
        )


    # Process batch fluxes for all showers
    #   2 sets of plots and CSV files will be saved: one set with all years combined, and one set with the
    #   reference year
    for shower_dir_dict, plot_suffix_status, fb_bin_params in [
        [shower_dirs, "ALL", fluxbatch_binning_params_all_years], 
        [shower_dirs_ref_year, "REF", fluxbatch_binning_params_one_year]
        ]:
        
        for shower_code in shower_dir_dict:

            shower = active_showers_dict[shower_code]
            dir_list = shower_dir_dict[shower_code]

            ref_height = -1
            if shower.ref_height is not None:
                ref_height = shower.ref_height

            # Construct the dir input list
            dir_params = [(night_dir_path, None, None, None, None, None) for night_dir_path in dir_list]

            # Compute the batch flux
            fbr = fluxBatch(shower_code, shower.mass_index, dir_params, ref_ht=ref_height, 
                min_meteors=fb_bin_params.min_meteors, 
                min_tap=fb_bin_params.min_tap, 
                min_bin_duration=fb_bin_params.min_bin_duration, 
                max_bin_duration=fb_bin_params.max_bin_duration, 
                compute_single=False,
                metadata_dir=metadata_dir,
                )


            if plot_suffix_status == "ALL":
                plot_suffix = "all_years"
            else:
                plot_suffix = "year_{:d}".format(shower.dt_max_ref_year.year)

            # Make a name for the plot to save
            batch_flux_output_filename = "flux_{:s}_sol={:.6f}-{:.6f}_{:s}".format(shower_code, 
                fbr.comb_sol_bins[0], fbr.comb_sol_bins[-1], plot_suffix)

            # Show and save the batch flux plot
            plotBatchFlux(
                fbr, 
                output_dir,
                batch_flux_output_filename,
                only_flux=False,
                compute_single=False,
                show_plot=False,
            )

            # Save the results to a CSV file
            saveBatchFluxCSV(fbr, output_dir, batch_flux_output_filename)


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

    arg_parser.add_argument('-m', '--metadir', metavar='FLUX_METADATA_DIRECTORY', type=str,
        help="Path to a directory with flux metadata (ECSV files). If not given, the data directory will be used.")

    arg_parser.add_argument('-o', '--outdir', metavar='FLUX_METADATA_DIRECTORY', type=str,
        help="Path to a directory where the plots and CSVs will be saved. If not given, the data directory will be used.")

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
        ref_dt = datetime.datetime.strptime(cml_args.time[0], "%Y%m%d_%H%M%S.%f")

    # If no manual time was given, use current time.
    else:
        ref_dt = datetime.datetime.utcnow()


    print("Computing flux using reference time:", ref_dt)

    # Run auto flux
    fluxAutoRun(config, cml_args.dir_path, ref_dt, metadata_dir=cml_args.metadir, output_dir=cml_args.outdir)
