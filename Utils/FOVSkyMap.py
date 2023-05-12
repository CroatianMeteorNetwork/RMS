
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from RMS.Routines.FOVArea import fovArea
from RMS.Astrometry.Conversions import latLonAlt2ECEF, ECEF2AltAz


def plotFOVSkyMap(platepars, out_dir, north_up=False,  masks=None):
    """ Plot all given platepar files on an Alt/Az sky map. 
    

    Arguments:
        platepars: [dict] A dictionary of Platepar objects where keys are station codes.
        out_dir: [str] Path to where the graph will be saved.

    Keyword arguments:
        masks: [dict] A dictionary of mask objects where keys are station codes.

    """

    # Change plotting style
    plt.style.use('ggplot')

    # Init an alt/az polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    if north_up:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

    # Set up elevation limits
    ax.set_rlim(bottom=90, top=0)


    # Reference height for FOV lat/lon
    ref_ht = 100000 # m

    for station_code in platepars:

        pp = platepars[station_code]

        if station_code in masks:
            mask = masks[station_code]
        else:
            mask = None

        print("Computing FOV for {:s}".format(pp.station_code))

        # Compute the edges of the
        side_points_latlon = fovArea(pp, mask=mask, area_ht=ref_ht, side_points=50, elev_limit=0)

        # Convert the station location to ECEF
        s_vect = np.array(latLonAlt2ECEF(np.radians(pp.lat), np.radians(pp.lon), pp.elev))

        # Convert lat/lon to local alt/az from the perspective of the station
        azims = []
        alts = []
        for side in side_points_latlon:

            for p_lat, p_lon, p_ht in side:

                # Convert FOV side point to ECEF
                p_vect = np.array(latLonAlt2ECEF(np.radians(p_lat), np.radians(p_lon), p_ht))

                # Compute the azimuth and altitude of P as viewed from S
                azim, alt = ECEF2AltAz(s_vect, p_vect)

                azims.append(azim)
                alts.append(alt)


        # If the polygon is not closed, close it
        if (azims[0] != azims[-1]) or (alts[0] != alts[-1]):
            azims.append(azims[0])
            alts.append(alts[0])


        # Plot the FOV alt/az
        line_handle, = ax.plot(np.radians(azims), alts, alpha=0.75)

        # Fill the FOV
        ax.fill(np.radians(azims), alts, color='0.5', alpha=0.3)


        # Plot the station name at the middle of the FOV
        ax.text(np.radians(pp.az_centre), pp.alt_centre, pp.station_code, va='center', ha='center', 
            color=line_handle.get_color(), weight='bold', size=8)


    ax.grid(True, color='0.98')
    ax.set_xlabel("Azimuth (deg)")
    ax.yaxis.set_major_formatter(StrMethodFormatter(u"{x:.0f}°"))
    ax.tick_params(axis='y', which='major', labelsize=8, direction='out')

    plt.tight_layout()


    # Save the plot to disk
    plot_file_name = "fov_sky_map.png"
    plot_path = os.path.join(out_dir, plot_file_name)
    plt.savefig(plot_path, dpi=150)
    print("FOV sky map saved to: {:s}".format(plot_path))




if __name__ == "__main__":

    from RMS.ConfigReader import Config
    from RMS.Formats.Platepar import Platepar
    from RMS.Routines.MaskImage import loadMask

    import os

    import argparse

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="""Compute the FOV area given the platepar and mask files. \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('dir_path', metavar='DIR_PATH', type=str, \
                    help="Path to the directory with platepar files. All platepar files will be found recursively.")

    arg_parser.add_argument('-n', '--northup', dest='north_up', default=False, action="store_true", \
                    help="Plot the chart with north up, azimuth increasing clockwise.")



    ###

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()


    # Init the default config file
    config = Config()


    # Find all platepar files
    platepars = {}
    masks = {}
    for entry in os.walk(cml_args.dir_path):

        dir_path, _, file_list = entry

        # Add platepar to the list if found
        if config.platepar_name in file_list:

            pp_path = os.path.join(dir_path, config.platepar_name)

            # Load the platepar file
            pp = Platepar()
            pp.read(pp_path)

            # If the station code already exists, skip it
            if pp.station_code in platepars:
                print("Skipping already added station: {:s}".format(pp_path))
                continue

            print()
            print("Loaded platepar for {:s}: {:s}".format(pp.station_code, pp_path))


            platepars[pp.station_code] = pp


            # Also add a mask if it's available
            if config.mask_file in file_list:
                masks[pp.station_code] = loadMask(os.path.join(dir_path, config.mask_file))
                print("Loaded the mask too!")



    # Plot all plateaprs on an alt/az sky map
    plotFOVSkyMap(platepars, cml_args.dir_path, cml_args.north_up, masks=masks)

            



    
