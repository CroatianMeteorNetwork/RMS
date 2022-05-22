""" Functions to load the shower catalog. """

from __future__ import print_function, division, absolute_import


import os
import copy
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from random import Random

from RMS.Astrometry.Conversions import datetime2JD, geocentricToApparentRadiantAndVelocity
from RMS.Routines.SolarLongitude import jd2SolLonSteyaert



class Shower(object):
    def __init__(self, shower_entry):

        # Indicates wheter the flux parameters are defined (False by default)
        self.flux_entry = False

        self.iau_code = shower_entry[0]
        self.name = shower_entry[1]
        self.name_full = shower_entry[2]

        # Generate a unique integer code based on the IAU code (which may have letters)
        self.iau_code_int_unique = ""
        self.iau_code_int = ""
        for c in str(self.iau_code):
            if c.isdigit():
                self.iau_code_int_unique += c
                self.iau_code_int += c
            else:
                self.iau_code_int_unique += str(ord(c))

        self.iau_code_int = int(self.iau_code_int)
        self.iau_code_int_unique = int(self.iau_code_int_unique)
        

        self.lasun_beg = float(shower_entry[3]) # deg
        self.lasun_max = float(shower_entry[4]) # deg
        self.lasun_end = float(shower_entry[5]) # deg
        self.ra_g = float(shower_entry[6]) # deg
        self.dec_g = float(shower_entry[7]) # deg
        self.dra = float(shower_entry[8]) # deg
        self.ddec = float(shower_entry[9]) # deg
        self.vg = float(shower_entry[10]) # km/s

        # Load parameters for flux, if that type of shower entry is loaded
        if len(shower_entry) > 13:

            self.flux_entry = True

            self.flux_year = shower_entry[11]
            self.flux_lasun_peak = float(shower_entry[12])
            self.flux_zhr_peak = float(shower_entry[13])
            self.flux_bp = float(shower_entry[14])
            self.flux_bm = float(shower_entry[15])
            self.population_index = float(shower_entry[16])
            self.mass_index = 1 + 2.5*np.log10(self.population_index)

        # Apparent radiant
        self.ra = None # deg
        self.dec = None # deg
        self.v_init = None # m/s
        self.azim = None # deg
        self.elev = None # deg
        self.shower_vector = None


        # Add a vactorized version of computeZHRFloat
        self.computeZHR = np.vectorize(self.computeZHRFloat) 


    def computeApparentRadiant(self, latitude, longitude, jdt_ref, meteor_fixed_ht=100000):
        """ Compute the apparent radiant of the shower at the given location and time.

        Arguments:
            latitude: [float] Latitude of the observer (deg).
            longitude: [float] Longitude of the observer (deg).
            jdt_ref: [float] Julian date.

        Keyword arguments:
            meteor_fixed_ht: [float] Assumed height of the meteor (m). 100 km by default.

        Return;
            ra, dec, v_init: [tuple of floats] Apparent radiant (deg and m/s).

        """


        # Compute the location of the radiant due to radiant drift
        if not np.any(np.isnan([self.dra, self.ddec])):
            
            # Solar longitude difference form the peak
            lasun_diff = (np.degrees(jd2SolLonSteyaert(jdt_ref)) - self.lasun_max + 180)%360 - 180

            ra_g = self.ra_g + lasun_diff*self.dra
            dec_g = self.dec_g + lasun_diff*self.ddec


        # Compute the apparent radiant - assume that the meteor is directly above the station
        self.ra, self.dec, self.v_init = geocentricToApparentRadiantAndVelocity(ra_g, \
            dec_g, 1000*self.vg, latitude, longitude, meteor_fixed_ht, \
            jdt_ref, include_rotation=True)

        return self.ra, self.dec, self.v_init


    def computeZHRFloat(self, la_sun):
        """ Compute the ZHR activity of the shower given the solar longitude. Only works for showers which
            have the flux parameters. Only takes floats!

        Arguments:
            la_sun: [float] Solar longitude (degrees).

        Return:
            zhr: [float]
        """

        # This can only be done for showers with the flux parameters
        if not self.flux_entry:
            return None

        # Determine if the given solar longitude is before or after the peak
        angle_diff = (la_sun%360 - self.flux_lasun_peak + 180 + 360)%360 - 180

        if angle_diff <= 0:
            b = self.flux_bp
            sign = 1

        else:
            b = self.flux_bm
            sign = -1

            # Handle symmetric activity which is defined as Bm being zero and thus Bp should be used too
            if self.flux_bm == 0:
                b = self.flux_bp

        # Compute the ZHR
        zhr = self.flux_zhr_peak*10**(sign*b*angle_diff)

        return zhr


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


def loadRadiantShowers(config):
    """ Load showers for single-station shower association, not flux. """
    
    return [Shower(shower_entry) for shower_entry in loadShowers(config.shower_path, config.shower_file_name)]


class FluxShowers(object):
    def __init__(self, config):

        # Load the list of showers used for flux
        shower_data = loadShowers(config.shower_path, config.showers_flux_file_name)

        # Init showers
        self.showers = [Shower(entry) for entry in shower_data]


    def activeShowers(self, dt_beg, dt_end, min_zhr=1):
        """ Return a list of active showers given a range of solar longitudes. 
    
        Arguments:
            dt_beg: [float] Starting datetime.
            dt_end: [float] End datetime.

        Keyword arguments:
            min_zhr: [float] Minimum ZHR for the shower to be considered active.

        Return:
            [list] A list of Shower objects with the modified activity period according to the ZHR threshold.
        """

        # Convert dates to solar longitudes
        la_sun_beg = np.degrees(jd2SolLonSteyaert(datetime2JD(dt_beg)))
        la_sun_end = np.degrees(jd2SolLonSteyaert(datetime2JD(dt_end)))


        # Sample the range with a 0.02 deg delta in sol (~30 minutes)
        if la_sun_beg > la_sun_end:
            la_sun_beg -= 360
        sol_array = np.arange(la_sun_beg, la_sun_end, 0.02)

        # Compute ZHR values for every shower, accunting for the peak and base component
        shower_codes = np.unique([shower.name for shower in self.showers])
        shower_zhrs = {shower_code: np.zeros_like(sol_array) for shower_code in shower_codes}

        for shower in self.showers:

            # Only take the shower if it's annual, or the year the correct
            if not shower.flux_year == "annual":
                if not (int(shower.flux_year) == dt_beg.year) and not (int(shower.flux_year) == dt_end.year):
                    continue

            # Compute the ZHR profile
            zhr_arr = shower.computeZHR(sol_array)

            # Add the profile to the shower dictionary
            shower_zhrs[shower.name] += zhr_arr


        # List of showers with a total ZHR above the threshold
        active_showers = []

        # Go through all showers and deterine if they were active or not
        for shower in self.showers:

            # Don't add already added showers
            if shower.name not in [shower.name for shower in active_showers]:
                
                # Determine the activity period
                activity_above_threshold = shower_zhrs[shower.name] > min_zhr

                if np.any(activity_above_threshold):

                    # Determine the activity period within the given range of solar longitudes
                    la_sun_min = np.min(sol_array[activity_above_threshold])%360
                    la_sun_max = np.max(sol_array[activity_above_threshold])%360

                    # Create a copy of the shower object with the modified activity period
                    shower_active = copy.deepcopy(shower)
                    shower_active.lasun_beg = la_sun_min
                    shower_active.lasun_end = la_sun_max

                    active_showers.append(shower_active)


        return active_showers


        # for shower in active_showers:
        #     print(shower.name, shower.lasun_beg, shower.lasun_end)


        # # Plot the activity profile for all showers
        # plt.plot(sol_array, np.sum([shower_zhrs[shower_code] for shower_code in shower_codes], axis=0), color='k', label='TOTAL')

        # # Plot individual activites
        # for shower in active_showers:
        #     plt.plot(sol_array, shower_zhrs[shower.name], label=shower.name)

        # plt.legend()
        # plt.xlabel("Sol")
        # plt.ylabel("ZHR")

        # plt.show()





# The seed ensures shower colours are the same each run
rng = Random(1) 


def getColorList(num, color_map=None):
    """ Return a list of colors for showers.

    Arguments:
        num: [int] Number of colors to return. 

    Return:
        colors: [list] A list of colors for matplotlib.

    """

    if color_map is None:
        color_list = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]
        colors = [color_list[i%len(color_list)] for i in range(num)]

    else:
        cmap = plt.get_cmap(color_map)
        colors = cmap(np.linspace(0.1, 0.9, num))[::-1]

    return colors


def makeShowerColors(shower_data, color_map):
    """ Generates a map of distinct colours indexed by shower name """

    # Sort showers into non-overlaping rows and assign them unique colors
    _, code_name_dict = sortShowersIntoRows(shower_data, color_map)

    # Assign each color to shower name
    colors_by_name = {}
    for code in code_name_dict:
        name, sol_min, sol_peak, sol_max, color = code_name_dict[code]
        colors_by_name[name] = color

    return colors_by_name



def sortShowersIntoRows(shower_data, color_map):

    # Generate an array of shower activity per 1 deg of solar longitude
    code_name_list = []
    activity_stack = np.zeros((20, 360), dtype=np.uint32)

    for i in range(20):
        for sol_plot in range(0, 360):

            # If the cell is unassigned, check to see if a shower is active
            if activity_stack[i, sol_plot] > 0:
                continue

            for shower in shower_data:
                code = shower.iau_code_int_unique
                name = shower.name

                # Skip already assigned showers
                if code in code_name_list:
                    continue

                # skip if shower doesn't have a stated start or end
                if np.any(np.isnan([shower.lasun_beg, shower.lasun_end])):
                    continue 
                
                sol_min, sol_peak, sol_max = shower.lasun_beg, shower.lasun_max, shower.lasun_end
                sol_min = int(np.floor(sol_min))%360
                sol_max = int(np.ceil(sol_max))%360

                # If the shower is active at the given solar longitude and there aren't any other showers
                # in the same activity period, assign shower code to this solar longitude
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
                            code_name_list.append(code)

                else:
                    if (sol_plot >= sol_min) or (sol_plot <= sol_max):

                        # Check if the solar longitue range is free of other showers
                        if (not np.any(activity_stack[i, 0:sol_max])) and \
                            (not np.any(activity_stack[i, sol_min:])):

                            # Assign shower code to activity stack
                            activity_stack[i, 0:sol_max] = code
                            activity_stack[i, sol_min:] = code


    # Count the number of populated rows in the activity stack
    active_rows = np.count_nonzero([np.any(row) for row in activity_stack])

    # Assign shower colors by row
    row_colors = getColorList(active_rows, color_map)

    # Assign a color to each shower
    code_name_dict = {}
    for shower in shower_data:
        code = shower.iau_code_int_unique
        name = shower.name

        # Skip assigned showers
        if code in code_name_dict:
            continue

        # skip if shower doesn't have a stated start or end
        if np.any(np.isnan([shower.lasun_beg, shower.lasun_end])):
            continue 

        sol_min, sol_peak, sol_max = shower.lasun_beg, shower.lasun_max, shower.lasun_end
        sol_min = int(np.floor(sol_min))%360
        sol_max = int(np.ceil(sol_max))%360

        for i, row in enumerate(activity_stack):

            # Check if the shower is in the row
            if code in row:

                # Grab the color from the list
                color = row_colors[i]

                # Assign a color to the shower
                code_name_dict[code] = [name, sol_min, sol_peak, sol_max, color]


    return activity_stack, code_name_dict




def generateActivityDiagram(config, shower_data, ax_handle=None, sol_marker=None, color_map='viridis'):
    """ Generates a plot of shower activity across all solar longitudes. """

    shower_data = np.array(shower_data)

    # Fill in min/max solar longitudes if they are not present
    for shower in shower_data:

        sol_min, sol_peak, sol_max = shower.lasun_beg, shower.lasun_max, shower.lasun_end

        if np.isnan(sol_min):
            shower.lasun_beg = (sol_peak - config.shower_lasun_threshold)%360

        if np.isnan(sol_max):
            shower.lasun_end = (sol_peak + config.shower_lasun_threshold)%360


    # Sort showers by duration
    durations = [(shower.lasun_end - shower.lasun_beg + 180) % 360 - 180 for shower in shower_data]
    shower_data = shower_data[np.argsort(durations)][::-1]


    # Sort showers into rows so that they do no overlap on the graph. This will also generate colors
    #   for every shower, sorted per row
    activity_stack, code_name_dict = sortShowersIntoRows(shower_data, color_map=color_map)

    
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
    vertical_shift_text = 0.02
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

    # Shift the plot maximum to accomodate the upper text
    y_max *= 1.25

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

        # Plot the month begin line
        y_arr = np.linspace(y_min, y_max, 5)
        plt.plot(np.zeros_like(y_arr) + sol, y_arr, linestyle='dotted', alpha=0.3, zorder=3, color='w')
    

    # Force Y limits
    ax_handle.set_ylim([y_min, y_max])
    ax_handle.set_xlim([0, 360])



if __name__ == "__main__":

    import argparse
    import RMS.ConfigReader as cr

    arg_parser = argparse.ArgumentParser(description="Plot chart of showers by date.")

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one.")

    arg_parser.add_argument('-p', '--palette', metavar='PALETTE', type=str, \
        help="color palette to use - one of viridis, gist_ncar, rainbow etc")

    cml_args = arg_parser.parse_args()

    # Load the list of all showers
    shower_table = loadShowers("share", "established_showers.csv")
    shower_list = [Shower(shower_entry) for shower_entry in shower_table]


    # Generate activity diagram
    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, 'notused')

    if cml_args.palette is None:
        color_map = config.shower_color_map
    else:
        color_map = cml_args.palette

    generateActivityDiagram(config, shower_list, \
        sol_marker=np.degrees(jd2SolLonSteyaert(datetime2JD(datetime.datetime.now()))), \
        color_map=color_map)

    plt.show()


    # Test the flux shower option
    flux_showers = FluxShowers(config)

    # Check for active showers in a range of dates
    active_showers = flux_showers.activeShowers(datetime.datetime(2021, 1, 1, 0, 0, 0), \
        datetime.datetime(2022, 1, 1, 0, 0, 0))

    for shower in active_showers:
        print(shower.name, shower.lasun_beg, shower.lasun_end)

