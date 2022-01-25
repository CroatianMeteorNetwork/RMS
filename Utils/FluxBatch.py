""" Batch run the flux code using a flux batch file. """

import datetime
import os
import shlex
import sys

import matplotlib.pyplot as plt
import numpy as np
from RMS.Formats.FTPdetectinfo import findFTPdetectinfoFile

from Utils.Flux import (computeFlux, computeTimeIntervals, detectClouds,
                        fluxParser)

if __name__ == "__main__":

    import argparse

    import RMS.ConfigReader as cr

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Compute single-station meteor shower flux from a batch file.")

    arg_parser.add_argument("batch_path", metavar="BATCH_PATH", type=str, \
        help="Path to the flux batch file.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    # Check if the batch file exists
    if not os.path.isfile(cml_args.batch_path):
        print("The given batch file does not exist!", cml_args.batch_path)
        sys.exit()


    dir_path = os.path.dirname(cml_args.batch_path)


    with open(cml_args.batch_path) as f:

        color_dict = {}
        marker_dict = {}
        markers = ["o", 'x', '+']

        output_data = []

        color_cycle = [plt.get_cmap("tab10")(i) for i in range(10)]

        # Parse the batch entries
        for line in f:
            line = line.replace("\n", "").replace("\r", "")

            if not len(line):
                continue

            if line.startswith("#"):
                continue

            clm_args = fluxParser().parse_args(shlex.split(line))
            ftpdetectinfo_path, shower_code, s, binduration, binmeteors, time_intervals = clm_args.ftpdetectinfo_path, \
                    clm_args.shower_code, clm_args.s, clm_args.binduration, clm_args.binmeteors, clm_args.time_intervals
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
                detect_clouds = detectClouds(config, dir_path)
                time_intervals = computeTimeIntervals(detect_clouds)
            else:
                time_intervals = [(*time_intervals,)]
            
            # Compute the flux
            for interval in time_intervals:
                print(f'Using interval: {interval}')
                dt_beg, dt_end = interval
                sol_data, flux_lm_6_5_data, flux_lm_6_5_ci_lower_data, flux_lm_6_5_ci_upper_data, \
                meteor_num_data = computeFlux(config, ftp_dir_path, ftpdetectinfo_path, shower_code, \
                    dt_beg, dt_end, s, binduration, binmeteors, show_plots=False)

                # Add computed flux to the output list
                output_data += [[config.stationID, sol, flux] for (sol, flux) in zip(sol_data, flux_lm_6_5_data)]


            # Make all stations the same color
            if config.stationID not in color_dict:
                
                # Generate a new color
                color = color_cycle[len(color_dict)%(len(color_cycle))]
                label = str(config.stationID)
                marker = markers[(len(marker_dict)//10)%(len(markers))]

                # Assign plot color
                color_dict[config.stationID] = color
                marker_dict[config.stationID] = marker

            else:
                color = color_dict[config.stationID]
                marker = marker_dict[config.stationID]
                #label = str(config.stationID)
                label = None


            print(config.stationID, color)

            # Plot the flux
            plt.plot(sol_data, flux_lm_6_5_data, label=label, color=color, marker=marker, linestyle='dashed')

            plt.errorbar(sol_data, flux_lm_6_5_data, color=color, alpha=0.5, capsize=5, zorder=3, linestyle='none', \
            yerr=[np.array(flux_lm_6_5_data) - np.array(flux_lm_6_5_ci_lower_data), \
                np.array(flux_lm_6_5_ci_upper_data) - np.array(flux_lm_6_5_data)])


        # plt.gca().set_yscale('log')


        # Save output
        data_out_path = os.path.join(dir_path, "flux_output.csv")
        with open(data_out_path, 'w') as fout:
            fout.write("#Station, Sol (deg), Flux@+6.5M (met/1000km^2/h)\n")
            for entry in output_data:
                stationID, sol, flux = entry

                fout.write("{:s},{:.8f},{:.3f}\n".format(stationID, sol, flux))

        print("Data saved to:", data_out_path)




        # Show plot
        plt.legend()

        plt.ylabel("Flux@+6.5M (met/1000km^2/h)")
        plt.xlabel("La Sun (deg)")

        plt.tight_layout()

        fig_path = os.path.join(dir_path, "batch_flux.png")
        print("Figure saved to:", fig_path)
        plt.savefig(fig_path, dpi=300)

        plt.show()
