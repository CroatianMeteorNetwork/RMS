""" Functions to load the shower catalog. """

from __future__ import print_function, division, absolute_import



import os
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from RMS.Astrometry.Conversions import datetime2JD
from RMS.Routines.SolarLongitude import jd2SolLonSteyaert


def loadShowers(dir_path, file_name):
    """ Loads the given shower CSV file. """

    # Older versions of numpy don't have the encoding parameter
    try:
        shower_data = np.genfromtxt(os.path.join(dir_path, file_name), delimiter='|', dtype=None, \
            autostrip=True, encoding=None)
    except:
        shower_data = np.genfromtxt(os.path.join(dir_path, file_name), delimiter='|', dtype=None, \
            autostrip=True)

    return shower_data


def makeShowerColors(shower_data, color_map='gist_ncar'):
    """ Generates a map of distinct colours indexed by shower name """
    names = sorted([s[1] for s in shower_data])
    cmap = plt.get_cmap(color_map)
    colors = cmap(np.linspace(0, 1, len(names)))
    colors_by_name = {n:colors[i] for i,n in enumerate(names)}
    return colors_by_name


def generateActivityDiagram(config, shower_data, ax_handle=None, sol_marker=None, colors=None):
    """ Generates a plot of shower activity across all solar longitudes. """

    shower_data = np.array(shower_data)

    # Fill in min/max solar longitudes if they are not present
    for shower in shower_data:

        sol_min, sol_peak, sol_max = list(shower)[3:6]

        if np.isnan(sol_min):
            shower[3] = (sol_peak - config.shower_lasun_threshold)%360

        if np.isnan(sol_max):
            shower[5] = (sol_peak + config.shower_lasun_threshold)%360


    # Sort showers by duration
    durations = [(shower[5] - shower[3] + 180) % 360 - 180 for shower in shower_data]
    shower_data = shower_data[np.argsort(durations)][::-1]


    # Generate an array of shower activity per 1 deg of solar longitude
    code_name_dict = {}

    activity_stack = np.zeros((20, 360), dtype=np.uint16)
    if not colors: colors = makeShowerColors(shower_data)
    shower_index = 0

    for i in range(20):
        for sol_plot in range(0, 360):

            # If the cell is unassigned, check to see if a shower is active
            if activity_stack[i, sol_plot] > 0:
                continue

            for shower in shower_data:
                code = int(shower[0])
                name = shower[1]

                # Skip assigned showers
                if code in code_name_dict:
                    continue

                sol_min, sol_peak, sol_max = list(shower)[3:6]
                sol_min = int(np.floor(sol_min))%360
                sol_max = int(np.ceil(sol_max))%360

                # If the shower is active at the given solar longitude and there aren't any other showers
                # in the same activity period, assign shower code to this solar longitude
                shower_active = False
                if (sol_max - sol_min) < 180:

                    # Check if the shower is active
                    if (sol_plot >= sol_min) and (sol_plot <= sol_max):

                        # Leave a buffer of +/- 3 deg around the shower
                        sol_min_check = sol_min - 3
                        if sol_min_check < 0:
                            sol_min_check = 0

                        sol_max_check = sol_max + 3
                        if sol_max_check > 360:
                            sol_max_check = 360

                        # Check if the solar longitue range is free of other showers
                        if not np.any(activity_stack[i, sol_min_check:sol_max_check]):

                            # Assign the shower code to activity stack
                            activity_stack[i, sol_min:sol_max] = code
                            code_name_dict[code] = [name, sol_peak]

                            shower_active = True

                else:
                    if (sol_plot >= sol_min) or (sol_plot <= sol_max):

                        # Check if the solar longitue range is free of other showers
                        if (not np.any(activity_stack[i, 0:sol_max])) and \
                            (not np.any(activity_stack[i, sol_min:])):

                            # Assign shower code to activity stack
                            activity_stack[i, 0:sol_max] = code
                            activity_stack[i, sol_min:] = code

                            shower_active = True
                        


                if shower_active:

                    # Get shower color
                    color = colors[name]
                    shower_index += 1

                    # Assign shower params
                    code_name_dict[code] = [name, sol_min, sol_peak, sol_max, color]

    
    # If no axis was given, crate one
    if ax_handle is None:
        ax_handle = plt.subplot(111, facecolor='black')

    # Set background color
    plt.gcf().patch.set_facecolor('black')

    # Change axis color
    ax_handle.spines['bottom'].set_color('w')
    ax_handle.spines['top'].set_color('w') 
    ax_handle.spines['right'].set_color('w')
    ax_handle.spines['left'].set_color('w')

    # Change tick color
    ax_handle.tick_params(axis='x', colors='w')
    ax_handle.tick_params(axis='y', colors='w')

    # Change axis label color
    ax_handle.yaxis.label.set_color('w')
    ax_handle.xaxis.label.set_color('w')
    
    # Plot the activity graph
    active_shower = 0
    vertical_scale_line = 0.5
    vertical_shift_text = 0.01
    text_size = 8
    for i, line in enumerate(activity_stack):

        for shower_block in line:

            # If a new shower was found, plot it
            if (shower_block != active_shower) and (shower_block > 0):

                # Get shower parameters
                name, sol_min, sol_peak, sol_max, color = code_name_dict[shower_block]
                    
                # Plot the shower activity period
                if (sol_max - sol_min) < 180:
                    x_arr = np.arange(sol_min, sol_max + 1, 1)
                    ax_handle.plot(x_arr, np.zeros_like(x_arr) + i*vertical_scale_line, linewidth=3, \
                        color=color, zorder=3)

                    ax_handle.text(round((sol_max + sol_min)/2), i*vertical_scale_line + vertical_shift_text, \
                        name, ha='center', va='bottom', color='w', size=text_size, zorder=3)

                else:

                    x_arr = np.arange(0, sol_max + 1, 1)
                    ax_handle.plot(x_arr, np.zeros_like(x_arr) + i*vertical_scale_line, linewidth=3, \
                        color=color, zorder=3)

                    x_arr = np.arange(sol_min, 361, 1)
                    ax_handle.plot(x_arr, np.zeros_like(x_arr) + i*vertical_scale_line, linewidth=3, \
                        color=color, zorder=3)

                    ax_handle.text(0, i*vertical_scale_line + vertical_shift_text, name, ha='center', \
                        va='bottom', color='w', size=text_size, zorder=2)

                # Plot peak location
                ax_handle.scatter(sol_peak, i*vertical_scale_line, marker='+', c="w", zorder=4)

                active_shower = shower_block

    # Hide y axis
    ax_handle.get_yaxis().set_visible(False)
    ax_handle.set_xlabel('Solar longitude (deg)')

    # Get the plot Y limits
    y_min, y_max = ax_handle.get_ylim()

    # Plot a line with given solver longitude
    if sol_marker is not None:

        # Plot the solar longitude line behind everything else
        y_arr = np.linspace(y_min, y_max, 5)

        if not isinstance(sol_marker, list):
            sol_marker = [sol_marker]

        for sol_value in sol_marker:
            ax_handle.plot(np.zeros_like(y_arr) + sol_value, y_arr, color='r', linestyle='dashed', zorder=2, \
                linewidth=1)


    # Plot month names at the 1st of that month (start in April of this year)
    for month_no, year_modifier in [[ 4, 0],
                                    [ 5, 0],
                                    [ 6, 0],
                                    [ 7, 0],
                                    [ 8, 0],
                                    [ 9, 0],
                                    [10, 0],
                                    [11, 0],
                                    [12, 0],
                                    [ 1, 1],
                                    [ 2, 1],
                                    [ 3, 1]]:

        # Get the solar longitude of the 15th date of the month
        curr_year = datetime.datetime.now().year
        dt = datetime.datetime(curr_year + year_modifier, month_no, 15, 0, 0, 0)
        sol = np.degrees(jd2SolLonSteyaert(datetime2JD(dt)))%360

        # Plot the month name in the background of the plot
        plt.text(sol, y_max, dt.strftime("%b").upper(), alpha=0.3, rotation=90, size=20, \
            zorder=1, color='w', va='top', ha='center')

        # Get the solar longitude of the 1st date of the month
        curr_year = datetime.datetime.now().year
        dt = datetime.datetime(curr_year + year_modifier, month_no, 1, 0, 0, 0)
        sol = np.degrees(jd2SolLonSteyaert(datetime2JD(dt)))%360

        # Plot the manth beginning line
        y_arr = np.linspace(y_min, y_max, 5)
        plt.plot(np.zeros_like(y_arr) + sol, y_arr, linestyle='dotted', alpha=0.3, zorder=3, color='w')
    

    # Force Y limits
    ax_handle.set_ylim([y_min, y_max])
    ax_handle.set_xlim([0, 360])



if __name__ == "__main__":

    import RMS.ConfigReader as cr


    shower_data = loadShowers("share", "established_showers.csv")


    # Generate activity diagram
    config = cr.parse('.config')
    generateActivityDiagram(config, shower_data, \
        sol_marker=np.degrees(jd2SolLonSteyaert(datetime2JD(datetime.datetime.now()))))

    plt.show()