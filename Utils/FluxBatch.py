""" Batch run the flux code using a flux batch file. """

import datetime
import os
import shlex
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy
from RMS.Astrometry.Conversions import datetime2JD, jd2Date
from RMS.Formats.FTPdetectinfo import findFTPdetectinfoFile
from RMS.Math import rollingAverage2d
from RMS.Routines.SolarLongitude import jd2SolLonSteyaert, solLon2jdSteyaert, unwrapSol

from Utils.Flux import computeFlux, detectClouds, fluxParser, loadForcedBinFluxData


def combineFixedBinsAndComputeFlux(sol_bins, meteors, time_area_prod, min_meteors=50, ci=0.95, min_tap=2):
    """
    Computes flux values and their corresponding solar longitude based on bins containing
    number of meteors, and time-area product. Bins will be combined so that each bin has the
    minimum number of meteors

    Arguments:
        sol_bins: [ndarray] Solar longitude of bins start and end (the length must be 1 more than meteors)
        meteors: [ndarray] Number of meteors in a bin
        time_area_prod: [ndarray] Time multiplied by LM corrected collecting area added for each station
            which contains each bin

    Keyword arguments:
        min_meteors: [int] Minimum number of meteors to have in a bin
        ci: [float] Confidence interval for calculating the flux error bars (from 0 to 1)
        min_tap: [float] Minimum time area product in 1000 km^2*h.
    Return:
        [tuple] sol, flux, flux_lower, flux_upper, meteors, ta_prod
            - sol: [ndarray] Solar longitude
            - flux: [ndarray] Flux corresponding to solar longitude
            - flux_lower: [ndarray] Lower bound of flux corresponding to sol
            - flux_upper: [ndarray] Upper bound of flux corresponding to sol
            - meteor_count: [ndarray] Number of meteors in bin
            - time_area_product: [ndarray] Time area product of bin

    """
    middle_bin_sol = (sol_bins[1:] + sol_bins[:-1]) / 2

    flux_list = []
    flux_upper_list = []
    flux_lower_list = []
    sol_list = []
    meteor_count_list = []
    area_time_product_list = []

    start_idx = 0
    for end_idx in range(1, len(meteors)):
        sl = slice(start_idx, end_idx)
        if np.sum(meteors[sl]) >= min_meteors and np.nansum(time_area_prod[sl]) / 1e9 >= min_tap:
            start_idx = end_idx
            ta_prod = np.sum(time_area_prod[sl])

            num_meteors = np.sum(meteors[sl])
            meteor_count_list.append(num_meteors)
            area_time_product_list.append(ta_prod)
            if ta_prod == 0:
                flux_list.append(0)
                flux_upper_list.append(0)
                flux_lower_list.append(0)
            else:
                n_meteors_upper = scipy.stats.chi2.ppf(0.5 + ci / 2, 2 * (num_meteors + 1)) / 2
                n_meteors_lower = scipy.stats.chi2.ppf(0.5 - ci / 2, 2 * num_meteors) / 2
                flux_list.append(1e9 * num_meteors / ta_prod)
                flux_upper_list.append(1e9 * n_meteors_upper / ta_prod)
                flux_lower_list.append(1e9 * n_meteors_lower / ta_prod)

            sol_list.append(np.mean(middle_bin_sol[sl]))

    return (
        np.array(sol_list),
        np.array(flux_list),
        np.array(flux_lower_list),
        np.array(flux_upper_list),
        np.array(meteor_count_list),
        np.array(area_time_product_list),
    )


def calculateFixedBins(all_time_intervals, dir_list, bin_duration=5):
    """
    Function to calculate the bins that any amount of stations over any number of years for one shower
    can be put into.

    Arguments:
        file_data: [list]

    Keyword arguments:
        bin_duration: [float] Bin duration in minutes (this is only an approximation since the bins are
            fixed to solar longitude)

    Return:
        [tuple] sol_bins, bin_datetime_dict
            - sol_bins: [ndarray] array of solar longitudes corresponding to each of the bins
            - bin_datetime_dict: [list] Each element contains a list of two elements: datetime_range, bin_datetime
                - datetime_range: [tuple] beg_time, end_time
                    - beg_time: [datetime] starting time of the bins
                    - end_time: [datetime] ending time of the last bin
                - bin_datetime: [list] list of datetime for each bin start and end time. The first element
                    is beg_time and the last element is end_time
    """

    # calculate bins for summary calculations
    if not all_time_intervals:
        return np.array([]), []

    sol_delta = 2 * np.pi / 60 / 24 / 365.24219 * bin_duration
    sol_beg = np.array([jd2SolLonSteyaert(datetime2JD(beg)) for beg, _ in all_time_intervals])
    sol_end = np.array([jd2SolLonSteyaert(datetime2JD(end)) for _, end in all_time_intervals])

    # even if the solar longitude wrapped around, make sure that you know what the smallest sol are
    if np.max(sol_beg) - np.min(sol_beg) > np.pi or np.max(sol_end) - np.min(sol_end) > np.pi:
        start_idx = np.argmin(np.where(sol_beg > np.pi, sol_beg, 2 * np.pi))
        end_idx = np.argmax(np.where(sol_end <= np.pi, sol_beg, 0))
    else:
        start_idx = np.argmin(sol_beg)
        end_idx = np.argmax(sol_end)
    min_sol = sol_beg[start_idx]
    max_sol = sol_end[end_idx] if sol_beg[start_idx] < sol_end[end_idx] else sol_end[end_idx] + 2 * np.pi
    sol_bins = np.arange(min_sol, max_sol, sol_delta)
    sol_bins = np.append(sol_bins, sol_bins[-1] + sol_delta)  # all events should be within the bins

    # make sure that fixed bins fit with already existing bins saved
    existing_sol = []
    for _dir in dir_list:
        if os.path.exists(os.path.join(_dir, 'fixedbinsflux.csv')):
            existing_sol.append(loadForcedBinFluxData(_dir, 'fixedbinsflux.csv')[0])

    ## calculating sol_bins
    if existing_sol:
        if len(existing_sol) == 1:
            starting_sol = existing_sol[0]
        else:
            # if there's more than one array of sol values, make sure they all agree with each other and
            # take the first
            failed = False
            comparison_sol = existing_sol[0]
            for sol, _dir in zip(existing_sol[1:], dir_list[1:]):
                min_len = min(len(comparison_sol), len(sol), 5)
                a, b = (
                    (comparison_sol[:min_len], sol[:min_len])
                    if comparison_sol[0] > sol[0]
                    else (sol[:min_len], comparison_sol[:min_len])
                )
                epsilon = 1e-12
                goal = sol_delta / 2
                val = (np.median(a - b if a[0] - b[0] < np.pi else b + 2 * np.pi - a) + goal) % sol_delta
                if np.abs(goal - val) > epsilon:
                    print(
                        f'!!! fixedbinsflux.csv in {dir_list[0]} and {_dir} don\'t match solar longitude values'
                    )
                    print('\tSolar longitude difference:', np.abs(goal - val))
                    failed = True

            if failed:
                print()
                raise Exception(
                    'Flux bin solar longitudes didn\'t match. To fix this, at least one of the'
                    ' fixedbinsflux.csv must be deleted.'
                )
            # filter only sol values that are inside the solar longitude
            starting_sol = comparison_sol[
                np.searchsorted(comparison_sol, min_sol, side='left') : np.searchsorted(
                    comparison_sol, max_sol, side='right'
                )
            ]

        sol_bins = starting_sol + np.mean(sol_bins[: len(starting_sol)] - starting_sol) % sol_delta
        sol_bins = np.append(sol_bins[0] - sol_delta, sol_bins)  # assume that it doesn't wrap around

    ## calculating datetime corresponding to sol_bins for each year
    bin_datetime_dict = []
    bin_datetime = []
    for sol in sol_bins:
        curr_time = all_time_intervals[start_idx][0] + datetime.timedelta(
            minutes=(sol - sol_bins[0]) / (2 * np.pi) * 365.24219 * 24 * 60
        )
        bin_datetime.append(jd2Date(solLon2jdSteyaert(curr_time.year, curr_time.month, sol), dt_obj=True))
    bin_datetime_dict.append([(bin_datetime[0], bin_datetime[-1]), bin_datetime])

    for start_time, _ in all_time_intervals:
        if all(
            [
                year_start > start_time or start_time > year_end
                for (year_start, year_end), _ in bin_datetime_dict
            ]
        ):
            delta_years = int(
                np.floor(
                    (start_time - all_time_intervals[start_idx][0]).total_seconds()
                    / (365.24219 * 24 * 60 * 60)
                )
            )
            bin_datetime = [
                jd2Date(solLon2jdSteyaert(dt.year + delta_years, dt.month, sol), dt_obj=True)
                for sol, dt in zip(sol_bins, bin_datetime_dict[0][1])
            ]
            bin_datetime_dict.append([(bin_datetime[0], bin_datetime[-1]), bin_datetime])

    return sol_bins, bin_datetime_dict


class StationPlotParams:
    '''Class to give plots specific appearances based on the station'''

    def __init__(self):
        self.color_dict = {}
        self.marker_dict = {}
        self.markers = ['o', 'x', '+']

        self.color_cycle = [plt.get_cmap("tab10")(i) for i in range(10)]

    def __call__(self, station):
        if station not in self.color_dict:
            # Generate a new color
            color = self.color_cycle[len(self.color_dict) % (len(self.color_cycle))]
            label = station
            marker = self.markers[(len(self.marker_dict) // 10) % (len(self.markers))]

            # Assign plot color
            self.color_dict[station] = color
            self.marker_dict[station] = marker

        else:
            color = self.color_dict[station]
            marker = self.marker_dict[station]
            # label = str(config.stationID)
            label = None

        return {'color': color, 'marker': marker, 'label': label}


if __name__ == "__main__":

    import argparse

    import RMS.ConfigReader as cr

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(
        description="Compute single-station meteor shower flux from a batch file."
    )

    arg_parser.add_argument(
        "batch_path", metavar="BATCH_PATH", type=str, help="Path to the flux batch file or csv file."
    )
    arg_parser.add_argument(
        "--output_filename",
        metavar="FILENAME",
        type=str,
        default='fluxbatch_output',
        help="Filename to export images and data (exclude file extensions)",
    )
    arg_parser.add_argument(
        "-csv", action='store_true', help="Whether batch_path is a .csv file that will be loaded from"
    )

    # Parse the command line arguments
    fluxbatch_cml_args = arg_parser.parse_args()

    #########################

    # Check if the batch file exists
    if not os.path.isfile(fluxbatch_cml_args.batch_path):
        print("The given batch file does not exist!", fluxbatch_cml_args.batch_path)
        sys.exit()

    dir_path = os.path.dirname(fluxbatch_cml_args.batch_path)

    output_data = []
    ci = 0.95

    plot_info = StationPlotParams()

    plt.figure(figsize=(15, 8))
    if not fluxbatch_cml_args.csv:
        with open(fluxbatch_cml_args.batch_path) as f:
            file_data = []

            # Parse the batch entries
            for line in f:
                line = line.replace("\n", "").replace("\r", "")

                if not len(line):
                    continue

                if line.startswith("#"):
                    continue

                flux_cml_args = fluxParser().parse_args(shlex.split(line))
                (
                    ftpdetectinfo_path,
                    shower_code,
                    s,
                    binduration,
                    binmeteors,
                    time_intervals,
                    fwhm,
                    ratio_threshold,
                ) = (
                    flux_cml_args.ftpdetectinfo_path,
                    flux_cml_args.shower_code,
                    flux_cml_args.s,
                    flux_cml_args.binduration,
                    flux_cml_args.binmeteors,
                    flux_cml_args.timeinterval,
                    flux_cml_args.fwhm,
                    flux_cml_args.ratiothres,
                )
                ftpdetectinfo_path = findFTPdetectinfoFile(ftpdetectinfo_path)

                if not os.path.isfile(ftpdetectinfo_path):
                    print("The FTPdetectinfo file does not exist:", ftpdetectinfo_path)
                    print("Exiting...")
                    sys.exit()

                # Extract parent directory
                ftp_dir_path = os.path.dirname(ftpdetectinfo_path)

                # Load the config file
                config = cr.loadConfigFromDirectory('.', ftp_dir_path)
                if time_intervals is None:
                    # find time intervals to compute flux with
                    time_intervals = detectClouds(
                        config, ftp_dir_path, show_plots=False, ratio_threshold=ratio_threshold
                    )
                else:
                    time_intervals = [(*time_intervals,)]

                file_data.append(
                    [
                        config,
                        ftp_dir_path,
                        ftpdetectinfo_path,
                        shower_code,
                        time_intervals,
                        s,
                        binduration,
                        binmeteors,
                        fwhm,
                    ]
                )

            sol_bins, bin_datetime_dict = calculateFixedBins(
                [time_interval for data in file_data for time_interval in data[4]],
                [data[1] for data in file_data],
            )

            all_bin_information = []
            # Compute the flux
            for (
                config,
                ftp_dir_path,
                ftpdetectinfo_path,
                shower_code,
                time_intervals,
                s,
                binduration,
                binmeteors,
                fwhm,
            ) in file_data:
                # print(config.stationID, )
                for interval in time_intervals:
                    dt_beg, dt_end = interval
                    forced_bins = (
                        bin_datetime_dict[
                            np.argmax(
                                [
                                    year_start < dt_beg < year_end
                                    for (year_start, year_end), _ in bin_datetime_dict
                                ]
                            )
                        ][1],
                        sol_bins,
                    )
                    ret = computeFlux(
                        config,
                        ftp_dir_path,
                        ftpdetectinfo_path,
                        shower_code,
                        dt_beg,
                        dt_end,
                        s,
                        binduration,
                        binmeteors,
                        show_plots=False,
                        default_fwhm=fwhm,
                        forced_bins=forced_bins,
                        confidence_interval=ci,
                    )
                    if ret is None:
                        continue
                    (
                        sol_data,
                        flux_lm_6_5_data,
                        flux_lm_6_5_ci_lower_data,
                        flux_lm_6_5_ci_upper_data,
                        meteor_num_data,
                        population_index,
                        bin_information,
                    ) = ret

                    # Add computed flux to the output list
                    output_data += [
                        [config.stationID, sol, flux, lower, upper, population_index]
                        for (sol, flux, lower, upper) in zip(
                            sol_data, flux_lm_6_5_data, flux_lm_6_5_ci_lower_data, flux_lm_6_5_ci_upper_data
                        )
                    ]
                    all_bin_information.append(bin_information)

                    plot_params = plot_info(config.stationID)
                    line = plt.plot(sol_data, flux_lm_6_5_data, **plot_params, linestyle='dashed')

                    plt.errorbar(
                        sol_data,
                        flux_lm_6_5_data,
                        color=plot_params['color'],
                        alpha=0.5,
                        capsize=5,
                        zorder=3,
                        linestyle='none',
                        yerr=[
                            np.array(flux_lm_6_5_data) - np.array(flux_lm_6_5_ci_lower_data),
                            np.array(flux_lm_6_5_ci_upper_data) - np.array(flux_lm_6_5_data),
                        ],
                    )

    else:
        # load data from .csv file and plot it
        if not fluxbatch_cml_args.batch_path.endswith('.csv'):
            print('WARNING!!! File given does not end with .csv, so it might not work properly with -csv')

        data = np.genfromtxt(
            fluxbatch_cml_args.batch_path, delimiter=',', dtype=None, encoding=None, skip_header=1
        )

        for stationID, sol, flux, lower, upper in data:
            plot_params = plot_info(stationID)

            plt.errorbar(
                sol,
                flux,
                **plot_params,
                alpha=0.5,
                capsize=5,
                zorder=3,
                linestyle='none',
                yerr=[[flux - lower], [upper - flux]],
            )

    # # Save output
    if not fluxbatch_cml_args.csv:
        data_out_path = os.path.join(dir_path, f"{fluxbatch_cml_args.output_filename}_1.csv")
        with open(data_out_path, 'w') as fout:
            fout.write(
                "#Station, Sol (deg), Flux@+6.5M (met/1000km^2/h), Flux lower bound, Flux upper bound\n"
            )
            for entry in output_data:
                stationID, sol, flux, lower, upper, population_index = entry

                fout.write(
                    "{:s},{:.8f},{:.3f},{:.3f},{:.3f},{}\n".format(
                        stationID, sol, flux, lower, upper, population_index
                    )
                )

    if all_bin_information:
        num_meteors = sum(np.array(meteors) for meteors, _, _ in all_bin_information)

        area_time_product = sum(np.array(area) * np.array(time) for _, area, time in all_bin_information)
        sol, flux, flux_lower, flux_upper, num_meteors, ta_prod = combineFixedBinsAndComputeFlux(
            sol_bins, num_meteors, area_time_product, ci=ci
        )

        plt.errorbar(
            np.degrees(sol) % 360,
            flux,
            yerr=[flux - flux_lower, flux_upper - flux],
            label='weighted average flux',
            c='k',
            marker='o',
            linestyle='none',
        )

        if not fluxbatch_cml_args.csv:
            data_out_path = os.path.join(dir_path, f"{fluxbatch_cml_args.output_filename}_2.csv")
            with open(data_out_path, 'w') as fout:
                fout.write(
                    "Sol (rad), Flux@+6.5M (met/1000km^2/h), Flux lower bound, Flux upper bound, Area-time product (corrected to +6.5M) (m^3/h), Meteor Count\n"
                )
                for _sol, _flux, _flux_lower, _flux_upper, _at, _nmeteors in zip(
                    sol, flux, flux_lower, flux_upper, ta_prod, num_meteors
                ):

                    fout.write(f"{_sol},{_flux},{_flux_lower},{_flux_upper},{_at},{_nmeteors}\n")

            print("Data saved to:", data_out_path)

    # Show plot
    plt.legend()

    plt.ylabel("Flux@+6.5M (met/1000km^2/h)")
    plt.xlabel("La Sun (deg)")

    # plt.tight_layout()

    fig_path = os.path.join(dir_path, f"{fluxbatch_cml_args.output_filename}.png")
    print("Figure saved to:", fig_path)
    plt.savefig(fig_path, dpi=300)

    plt.show()
