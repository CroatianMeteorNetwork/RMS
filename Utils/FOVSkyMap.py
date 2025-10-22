import datetime
import numpy as np
import matplotlib.pyplot as plt
import RMS.ConfigReader as cr

from matplotlib.ticker import StrMethodFormatter
from RMS.Routines.FOVArea import fovArea
from RMS.Routines.FOVSkyArea import fovSkyArea

from RMS.Astrometry.Conversions import latLonAlt2ECEF, ECEF2AltAz, raDec2AltAz, datetime2JD, jd2Date
import ephem
import os

# Approximate duration of a lunar month in seconds
LUNAR_MONTH_PERIOD_SECONDS = int(29.5 * 24 * 60 * 60)

# Approximate duration of a day in seconds
DAY_IN_SECONDS = int(24 * 60 * 60)

# Approximate duration of a year in seconds
YEAR_IN_SECONDS = int(365.25 * DAY_IN_SECONDS)

# Astromical dusk, when centre of sun is 18 degrees below local horizon
ASTRONOMICAL_DUSK = np.radians(-18)


def plotMoon(ax, configs, show_moon, station_code, moon_plotted):
    """
    Plots the position of the moon every 5 minutes for the next lunar month

    Arguments:
        ax: [axis] axis on which to plot
        configs: [dict] dictionary of config files
        show_moon: [bool] whether to show moon
        station_code: [str] station code
        moon_plotted: [bool] if True, moon will not be plotted again, if False, moon will be plotted

    Return:
        moon_plotted: [bool] Generally returned true, unless the moon never rose above 0 elevation
    """

    if show_moon and station_code in configs and not moon_plotted:
        c = configs[station_code]
        o = ephem.Observer()
        o.lat = str(c.latitude)
        o.long = str(c.longitude)
        o.elevation = c.elevation


        current_time = datetime.datetime.now(datetime.timezone.utc)

        moon_azim_list, moon_elev_list = [], []

        for elapsed_time in range(0, LUNAR_MONTH_PERIOD_SECONDS, 300):
            time_to_evaluate = current_time + datetime.timedelta(seconds=elapsed_time)
            o.date = time_to_evaluate
            moon = ephem.Moon(o)

            moon.compute(o)

            if moon.alt > 0:
                moon_azim_list.append(moon.az)
                moon_elev_list.append(np.degrees(moon.alt))

        if len(moon_elev_list):
            moon_plotted = True
            ax.scatter(moon_azim_list, moon_elev_list, color='blue')
    return moon_plotted

def plotSun(ax, configs, show_sun, station_code, sun_plotted):
    """
        Plots the position of the sun every hour for the next 365 days

        Arguments:
            ax: [axis] axis on which to plot
            configs: [dict] dictionary of config files
            show_sun: [bool] whether to show sun
            station_code: [str] station code
            sun_plotted: [bool] if True, sun will not be plotted again, if False, sun will be plotted

        Return:
            sun_plotted: [bool] Generally returned true, unless the sun never rose above 0 elevation
        """

    if show_sun and station_code in configs and not sun_plotted:
        c = configs[station_code]
        o = ephem.Observer()
        o.lat = str(c.latitude)
        o.long = str(c.longitude)
        o.elevation = c.elevation

        # Set the time to the previous midnight
        current_time = datetime.datetime.now(datetime.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

        sun_azim_list, sun_elev_list = [], []

        for elapsed_time in range(0, YEAR_IN_SECONDS, 3600):
            time_to_evaluate = current_time + datetime.timedelta(seconds=elapsed_time)
            o.date = time_to_evaluate
            sun = ephem.Sun(o)
            sun.compute(o)

            if sun.alt > 0:
                sun_azim_list.append(sun.az)
                sun_elev_list.append(np.degrees(sun.alt))

        if len(sun_elev_list):
            sun_plotted = True
            ax.scatter(sun_azim_list, sun_elev_list, color='yellow')
    return sun_plotted

def plotRaDec(ax, configs, show_radec, radec_list, radec_name_list, station_code, radec_plotted, jd_start=None, jd_end=None):

    """
        Plots the position of a radec point every minute for the next 24 hours, or jd range specified, as long as Sun is
        below local horizon

        Arguments:
            ax: [axis] axis on which to plot
            configs: [dict] dictionary of config files
            show_radec: [bool] whether to show radec
            station_code: [str] station code
            radec_plotted: [bool] if True, radec will not be plotted again, if False, radec will be plotted

        Keyword arguments:
            jd_start: Optional, default None, start of the jd range to plot the radec
            jd_end: Optional, default None, start of the jd range to plot the radec

        Return:
            radec_plotted: [bool] Generally returned true, unless the sun never set, or the object
            never rose
        """

    if show_radec and station_code in configs and not radec_plotted:
        c = configs[station_code]
        o = ephem.Observer()
        o.lat = str(c.latitude)
        o.long = str(c.longitude)
        o.elevation = c.elevation

        if jd_start is None or jd_end is None:
            # Use the current time
            start_time = datetime.datetime.now(datetime.timezone.utc)
            duration_seconds = DAY_IN_SECONDS
        else:
            start_time = jd2Date(jd_start, UT_corr=0, dt_obj=True)
            end_time = jd2Date(jd_end, UT_corr=0, dt_obj=True)
            duration_seconds = (end_time - start_time).total_seconds()

        # Title the plot. If only RADEC is give, then put that in the title.
        if len(radec_name_list) == 1 and len(radec_list):
            ax.set_title("Plot of {} RADEC ({:.1f},{:.1f}) degrees starting at {} ".format(radec_name_list[0], radec_list[0][0], radec_list[0][1], start_time.replace(microsecond=0)), fontsize=10)
        else:
            ax.set_title(
                "Plot of {} objects starting at {} ".format(len(radec_name_list), start_time.replace(microsecond=0)), fontsize=10)

        object_rise, object_set = False, False
        for radec, radec_name in zip(radec_list, radec_name_list):
            radec_azim_list, radec_elev_list = [], []
            change_state, plot_arrow, plot_count, last_values_initialized = True, False, 1, False
            last_plotted_initialized = False

            for elapsed_time in range(0, duration_seconds, 60):

                # Initialize the observer
                time_to_evaluate = start_time + datetime.timedelta(seconds=elapsed_time)
                time_str = time_to_evaluate.strftime('%H:%M')
                o.date = time_to_evaluate
                sun = ephem.Sun(o)
                sun.compute(o)

                # Compute az and el of object
                jd = datetime2JD(time_to_evaluate)
                ra_degrees, dec_degrees = radec[0], radec[1]
                az, el = raDec2AltAz(ra_degrees, dec_degrees, jd, c.latitude, c.longitude)

                # Is the sun below the dusk threshold?
                if sun.alt < ASTRONOMICAL_DUSK:

                    # Plot arrows each hour, or when plot_arrow is set to show the direction of motion and passage of time
                    if plot_arrow or (plot_count % 60 == 0 and plot_count != 0):
                        plot_arrow = False
                        start, end = (np.radians(_az), _el), (np.radians(az), el)
                        ax.annotate('',
                                    xy=end,
                                    xytext=start,
                                    arrowprops=dict(arrowstyle='->', color='orange', lw=2))
                        plot_count = 0
                    # Annotate the plot with information at dusk, dawn, and each rise and set of the object, provided
                    # plot is not more than 1.5 days
                    if last_values_initialized and duration_seconds < 1.5 * DAY_IN_SECONDS:
                        if change_state:
                            change_state = False

                            if _sun_alt < ASTRONOMICAL_DUSK and sun.alt > ASTRONOMICAL_DUSK:
                                ax.annotate("{} (dawn)".format(time_str),
                                            xy=(np.radians(az), el - 2), color="black", fontsize=8)
                                plot_arrow = False
                            elif _sun_alt > ASTRONOMICAL_DUSK and sun.alt < ASTRONOMICAL_DUSK:
                                ax.annotate(" {}: UTC {} (dusk)".format(radec_name, time_str),
                                            xy=(np.radians(az + 2), el + 4), color="black", fontsize=8)
                                plot_arrow = True
                            elif object_set:
                                ax.annotate(" {} (setting)".format(time_str),
                                            xy=(np.radians(az - 2), 0), color="black", fontsize=8)
                                plot_arrow = False
                            elif object_rise:
                                ax.annotate(" {}: UTC {} (rising)".format(radec_name, time_str),
                                            xy=(np.radians(az), 0), color="black", fontsize=8)
                                plot_arrow = True

                        else:
                            # Has the object risen or set
                            if _el < 0.1 and el > 0.1:
                                change_state = True
                                object_rise, object_set = True, False
                                plot_count = 0
                            elif _el > 0.1 and el < 0.1:
                                change_state = True
                                object_rise, object_set = False, True
                                plot_count = 0

                    # Append to the plot list
                    radec_azim_list.append(np.radians(az))
                    radec_elev_list.append(el)

                    # Store previous plotted values to use for arrow directions etc
                    if last_values_initialized:
                        last_plotted_az, _last_plotted_az = az, _az
                        last_plotted_el, _last_plotted_el = el, _el
                        last_plotted_sun_alt, last_plotted_time_str = sun.alt, time_str
                        last_plotted_initialized = True

                    # Increment the plot counter, used for plotting arrows
                    plot_count += 1

                # Store the previous elevation, azimuth, time and sun elevation for determining dusk and dawn
                _el, _az, _time_str, _sun_alt = el, az, time_str, sun.alt
                last_values_initialized = True

            # Annotate the final point which was plotted, provided this has been initialised

            if last_plotted_initialized:
                if last_plotted_sun_alt > _sun_alt:
                    ax.annotate(" {}: UTC {} (dusk)".format(radec_name, last_plotted_time_str),
                            xy=(np.radians(last_plotted_az + 2),last_plotted_el + 4), color="black", fontsize=8)

                elif last_plotted_sun_alt < _sun_alt:
                    ax.annotate(" {} (dawn)".format(last_plotted_time_str),
                            xy=(np.radians(last_plotted_az),last_plotted_el - 4), color="black", fontsize=8)

                start, end = (np.radians(_last_plotted_az), _last_plotted_el), (np.radians(last_plotted_az), last_plotted_el)
                ax.annotate('',
                            xy=end,
                            xytext=start,
                            arrowprops=dict(arrowstyle='->', color='orange', lw=2))

            if len(radec_elev_list):
                radec_plotted = True
                ax.scatter(radec_azim_list, radec_elev_list, color='black', s=1)
    return radec_plotted


def plotFOVSkyMap(platepars, configs, out_dir, north_up=False, show_pointing=False, show_fov=False,
                  rotate_text=False, flip_text=False, show_ip=False, show_coordinates=False, masks=None,
                  output_file_name="fov_sky_map.png", show_sun=False, show_moon=False,
                  show_radec=False, radec_list=[], radec_name_list=[], jd_start=None, jd_end=None):
    """ Plot all given platepar files on an Alt/Az sky map.


    Arguments:
        platepars: [dict] A dictionary of Platepar objects where keys are station codes.
        configs: [dics] A dictionary of RMS config objects where keys are station codes.
        out_dir: [str] Path to where the graph will be saved.

    Keyword arguments:
        north_up: [bool] If true, plot the north upward.
        show_pointing: [bool] If true, annotate plot with azimuth and elevation per camera
        show_fov: [bool] If true, annotate plot with field of view per camera
        rotate_text: [bool] If true, rotate text so that is normal to the camera
        flip_text: [bool] If true, flip text so that it is never upside down
        show_ip: [bool] If true, annotate plot with address of camera
        show_coordinates: [bool] If true, annotate plot with coordinates
        masks: [dict] A dictionary of mask objects where keys are station codes.
        output_file_name: [str] Name of output file default "fov_sky_map.png"
        show_sun: [bool] If true, annotate plot with sun track, default False
        show_radec: [bool] If true, annotate plot with the radec object given
        radec_list: [list] A list of [ra, dec] to be plotted
        radec_name_list: [list] A list of names, in the same order as the radec_list

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
    sun_plotted, moon_plotted, radec_plotted = False, False, False

    print()
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

        # Compute the area of the FOV polygon in square degrees
        fov_area = fovSkyArea(pp, mask=mask)

        # Plot the FOV alt/az
        line_handle, = ax.plot(np.radians(azims), alts, alpha=0.75)

        # Fill the FOV
        ax.fill(np.radians(azims), alts, color='0.5', alpha=0.3)

        # Label for FOV
        fov_label, label_size = pp.station_code, 8

        # Compute text rotation

        rot = 0
        if rotate_text:

            if north_up:
                rot = 0 - pp.az_centre
                if -270 < rot < -90 and flip_text:
                    rot += 180

            else:
                rot = 90 + pp.az_centre
                if  90 < rot < 270 and flip_text:
                    rot -= 180

        if show_pointing:
            fov_label += "\n az:{:.1f} el:{:.1f}".format(pp.az_centre, pp.alt_centre)
            label_size -= 1

        if show_fov:
            fov_label += "\n{:.1f} x {:.1f}, {:.1f} sq deg".format(pp.fov_h, pp.fov_v, fov_area)
            label_size -= 1

        sun_plotted = plotSun(ax, configs, show_sun, station_code, sun_plotted)
        moon_plotted = plotMoon(ax, configs, show_moon, station_code, moon_plotted)
        radec_plotted = plotRaDec(ax, configs, show_radec, radec_list, radec_name_list, station_code, radec_plotted, jd_start, jd_end)

        if station_code in configs:
            c = configs[station_code]
            if show_ip:
                try:
                    ip = c.deviceID.split("rtsp://")[1].split(":")[0]
                    label_size -= 1
                except:
                    ip = ""
                fov_label += "\n{}".format(ip)
            if show_coordinates:
                if c.latitude > 0:
                    lat = "N{:.6f}".format(0 + c.latitude)
                else:
                    lat = "S{:.6f}".format(0 - c.latitude)
                if c.longitude > 0:
                    lon = "E{:.6f}".format(0 + c.longitude)
                else:
                    lon = "W{:.6f}".format(0 - c.longitude)
                fov_label += "\n{} {}".format(lat, lon)

        # Plot the station name at the middle of the FOV
        ax.text(np.radians(pp.az_centre), pp.alt_centre, fov_label, va='center', ha='center',
            color=line_handle.get_color(), weight='bold', size=label_size, rotation=rot)


    ax.grid(True, color='0.98')
    ax.set_xlabel("Azimuth (deg)")
    ax.yaxis.set_major_formatter(StrMethodFormatter(u"{x:.0f}\u00b0"))
    ax.tick_params(axis='y', which='major', labelsize=8, direction='out')

    plt.tight_layout()


    if output_file_name is None:
        output_file_name = os.path.join(out_dir, "fov_sky_map.png")
    else:
        output_file_name = os.path.join(os.path.abspath('.'), os.path.expanduser(output_file_name))

    if os.path.isdir(output_file_name):
        output_file_name = os.path.join(output_file_name, "fov_sky_map.png")



    # Save the plot to disk
    plot_path = os.path.expanduser(output_file_name)
    if os.path.isdir(plot_path):
        plot_path = os.path.join(plot_path, "fov_sky_map.png")
    plt.savefig(plot_path, dpi=150)
    print()
    print("FOV sky map saved to: {:s}".format(plot_path))


if __name__ == "__main__":

    from RMS.ConfigReader import Config
    from RMS.Formats.Platepar import Platepar
    from RMS.Routines.MaskImage import loadMask

    import os

    import argparse

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="""Compute the FOV area given the platepar and mask files.
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('dir_path', metavar='DIR_PATH', type=str,
                    help="Path to the directory with platepar files. All platepar files will be found recursively.")

    arg_parser.add_argument('-n', '--northup',  dest='north_up', default=False, action="store_true",
                    help="Plot the chart with north up, azimuth increasing clockwise.")

    arg_parser.add_argument('-p', '--pointing', dest='pointing', default=False, action="store_true",
                            help="Show pointing on degrees in chart.")

    arg_parser.add_argument('-f','--fov', dest='fov', default=False, action="store_true",
                            help="Show field of view and area in degrees on chart.")

    arg_parser.add_argument('-r', '--rotate', dest='rotate', default=False, action="store_true",
                            help="Rotate text in line with camera pointing.")

    arg_parser.add_argument('-l', '--flip_text', dest='flip_text', default=False, action="store_true",
                            help="Flip text so it is never upside down.")

    arg_parser.add_argument('-o', '--output_file_name', dest='output_file_name', default=None,
                            nargs=1, help="Output filename and path.")

    arg_parser.add_argument('-i', '--show_ip', dest='show_ip', default=False, action="store_true",
                            help="Show ip address of the camera.")

    arg_parser.add_argument('-c', '--show_coordinates', dest='show_coordinates', default=False, action="store_true",
                            help="Show coordinates of the camera.")

    arg_parser.add_argument('-s', '--show_sun', dest='show_sun', default=False, action="store_true",
                            help="Plot the position of the sun in each hour for the next year as seen from one station.")

    arg_parser.add_argument('-m', '--show_moon', dest='show_moon', default=False, action="store_true",
                            help="Plot the position of the moon every 5 minutes for the next 29.5 days as seen from one station.")


    arg_parser.add_argument('-d', '--show_radec', dest='ra_dec', nargs=3, type=str, action="append",
                            help="Show the path of multiple radecs across the FoV, pass as ra, dec, name. The name passed will be used for annotation only.")


    ###

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()


    # Init the default config file
    config = Config()

    show_radec = False
    radec_list, radec_name_list = [], []
    if cml_args.ra_dec is not None:

        show_radec = True
        for radec_group in cml_args.ra_dec:
            radec_list.append([float(radec_group[0]), float(radec_group[1])])
            radec_name_list.append(radec_group[2])

    else:
        ra, dec, radec_name = None, None, None

    # Find all platepar files
    platepars = {}
    masks = {}
    configs = {}

    cml_args.dir_path = os.path.expanduser(cml_args.dir_path)
    if not os.path.isdir(cml_args.dir_path):
        print("Input directory {:s} does not exist, quitting.".format(cml_args.dir_path))
        quit()

    if cml_args.output_file_name is None:
        output_file_name = cml_args.dir_path
    else:
        output_file_name = cml_args.output_file_name[0]

    if not os.path.isdir(output_file_name) and not os.path.isdir(os.path.dirname(output_file_name)):
        print("Output directory {:s} does not exist, quitting.".format(os.path.dirname(output_file_name)))
        quit()



    for entry in sorted(os.walk(os.path.expanduser(cml_args.dir_path), topdown=True)):

        dir_path, dirs , file_list = entry

        # Add platepar to the list if found
        if config.platepar_name in file_list:

            pp_path = os.path.join(dir_path, config.platepar_name)
            config_path = os.path.join(dir_path, ".config")

            # Load the platepar file
            pp = Platepar()
            pp.read(pp_path)


            # If the station
            # code already exists, skip it
            if pp.station_code in platepars:
                print("Skipping already added station: {:s}".format(pp_path))
                continue

            print()
            print("Loaded platepar for {:s}: {:s}".format(pp.station_code, pp_path))
            platepars[pp.station_code] = pp


            if os.path.exists(config_path):
                configs[pp.station_code] = cr.parse(config_path)
                print("Loaded config for   {:s}: {:s}".format(pp.station_code, config_path))

            # Also add a mask if it's available
            if config.mask_file in file_list:
                masks[pp.station_code] = loadMask(os.path.join(dir_path, config.mask_file))
                print("Loaded mask for     {:s}: {:s}".format(pp.station_code, pp_path))



    # Plot all platepars on an alt/az sky map
    plotFOVSkyMap(platepars, configs, cml_args.dir_path, north_up=cml_args.north_up,
                  show_pointing=cml_args.pointing, show_fov=cml_args.fov,
                  rotate_text=cml_args.rotate, masks=masks,
                  flip_text=cml_args.flip_text, output_file_name=output_file_name,
                  show_ip=cml_args.show_ip, show_coordinates=cml_args.show_coordinates,
                  show_sun=cml_args.show_sun, show_moon=cml_args.show_moon,
                  show_radec=show_radec, radec_list=radec_list, radec_name_list=radec_name_list, jd_start=None, jd_end=None)





