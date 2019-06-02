""" Functions to load the shower catalog. """

from __future__ import print_function, division, absolute_import



import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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



def generateActivityDiagram(config, shower_data):
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
    colors = cm.tab10(np.linspace(0, 1.0, 10))
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
                        if (not np.any(activity_stack[i, 0:sol_max])) and (not np.any(activity_stack[i, sol_min:])):

                            # Assign shower code to activity stack
                            activity_stack[i, 0:sol_max] = code
                            activity_stack[i, sol_min:] = code

                            shower_active = True
                        


                if shower_active:

                    # Get shower color
                    color = colors[shower_index%10]
                    shower_index += 1

                    # Assign shower params
                    code_name_dict[code] = [name, sol_min, sol_peak, sol_max, color]

    
    # Plot the activity graph
    active_shower = 0
    vertical_scale_line = 0.1
    vertical_shift_text = 0.01
    for i, line in enumerate(activity_stack):

        for shower_block in line:

            # If a new shower was found, plot it
            if (shower_block != active_shower) and (shower_block > 0):

                # Get shower parameters
                name, sol_min, sol_peak, sol_max, color = code_name_dict[shower_block]
                    
                # Plot the shower activity period
                if (sol_max - sol_min) < 180:
                    x_arr = np.arange(sol_min, sol_max + 1, 1)
                    plt.plot(x_arr, np.zeros_like(x_arr) + i*vertical_scale_line, linewidth=5, color=color)

                    plt.text(round((sol_max + sol_min)/2), i*vertical_scale_line + vertical_shift_text, name, ha='center', va='bottom')

                else:

                    x_arr = np.arange(0, sol_max + 1, 1)
                    plt.plot(x_arr, np.zeros_like(x_arr) + i*vertical_scale_line, linewidth=5, color=color)

                    x_arr = np.arange(sol_min, 361, 1)
                    plt.plot(x_arr, np.zeros_like(x_arr) + i*vertical_scale_line, linewidth=5, color=color)

                    plt.text(0, i*vertical_scale_line + vertical_shift_text, name, ha='center', va='bottom')

                # Plot peak location
                plt.scatter(sol_peak, i*vertical_scale_line, marker='+', c="w", zorder=4)

                active_shower = shower_block


    plt.xlabel('Solar longitude (deg)')
    
    plt.show()



        














if __name__ == "__main__":

    import RMS.ConfigReader as cr


    shower_data = loadShowers("share", "established_showers.csv")


    # Generate activity diagram
    config = cr.parse('.config')
    generateActivityDiagram(config, shower_data)