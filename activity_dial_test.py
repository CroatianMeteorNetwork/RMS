import os
import datetime
import collections

import numpy as np
import matplotlib.pyplot as plt


from Utils.FluxAuto import generateZHRDialSVG
from Utils.FluxFitActivityCurve import computeCurrentPeakZHR, loadFluxActivity, plotYearlyZHR


if __name__ == "__main__":

    from RMS.ConfigReader import Config


    ###########################

    out_path = "C:/Users/denis/Dropbox/UWO/Projects/flux/dial"

    # Output SVG dial
    flux_dial_out_file = "flux_dial_latest.svg"

    # Set the sporadic ZHR
    sporadic_zhr = 25

    ###########################

    config = Config()

    # Load the flux activity file
    print("Loading flux activity file...")
    shower_models = loadFluxActivity(config)

    # Print shower name and ZHR of all shower models
    print("Shower models:")
    for shower_name, shower_model in shower_models.items():
        print()
        print("Shower: {}".format(shower_name))
        print("Base ZHR @ sol {:.2f} = {:.2f}".format(shower_model.base_fit.sol_peak, shower_model.base_fit.peak_zhr))

        for add_params in shower_model.additional_fits:
            print("   + ZHR @ sol {:.2f} = {:.2f}".format(add_params.sol_peak, add_params.peak_zhr))


    print()
    print("-----------------")

    # Compute the current peak ZHR
    print("Computing current peak ZHR...")
    peak_zhr = computeCurrentPeakZHR(shower_models, sporadic_zhr=sporadic_zhr)
    print("Current peak ZHR: {:.2f}".format(peak_zhr))

    # ZHR at the peak of the Perseids
    print("ZHR at Persieds peak: {:.2f}".format(computeCurrentPeakZHR(shower_models, datetime.datetime(2020, 8, 12, 0, 0, 0), sporadic_zhr=sporadic_zhr)))

    
    # Set the ZHR dial
    svg = generateZHRDialSVG(config.flux_dial_template_svg, peak_zhr, sporadic_zhr)

    # Save the SVG file
    with open(os.path.join(out_path, flux_dial_out_file), "w") as f:
        f.writelines(svg)

    
    # Plot the ZHR throughout the year
    plotYearlyZHR(config, os.path.join(out_path, config.yearly_zhr_plot_name), sporadic_zhr=sporadic_zhr)