""" Automatically runs the flux code and produces graphs on available data from multiple stations. """

from __future__ import print_function, division, absolute_import

import os
import sys
import ast
import time
import datetime
import copy
import json
import traceback

import numpy as np

from RMS.Astrometry.Conversions import datetime2JD
from RMS.Formats.Showers import FluxShowers
from RMS.Math import isAngleBetween
from RMS.Routines.SolarLongitude import jd2SolLonSteyaert
from Utils.FluxBatch import fluxBatch, plotBatchFlux, FluxBatchBinningParams, saveBatchFluxCSV, \
    reportCameraTally
from RMS.Misc import mkdirP, walkDirsToDepth
from Utils.FluxFitActivityCurve import computeCurrentPeakZHR, loadFluxActivity, plotYearlyZHR


def generateZHRDialSVG(ref_svg_path, zhr, sporadic_zhr):
    """ Load the ZHR dial SVG and set the ZHR to the given number. 
    
    Arguments:
        ref_svg_path: [str] Path to the reference SVG file.
        zhr: [float] Current ZHR.
        sporadic_zhr: [float] Current sporadic ZHR.

    Return:
        svg: [list of str] A string containing the SVG file.
        
    """

    # Load the reference SVG file as a string
    with open(ref_svg_path, "r") as f:
        svg = f.readlines()

    # ZHR cannot be negative
    if zhr < 0:
        zhr = 0

    # Maximum ZHR on the dial
    max_zhr = 110

    # Compute the angle of the ZHR hand
    # 0 ZHR = -180 deg
    # 50 ZHR = -90 deg
    # 100 ZHR = 0 deg
    # >110 ZHR = 9 deg (this is the upper limit)
    hand_angle = -180 + (zhr*9/5)
    if zhr > max_zhr:
        hand_angle = -180 + (max_zhr*9/5)

    # Compute the angle of the sporadic zhr
    sporadic_pie_angle = -180 + (sporadic_zhr*9/5)

    
    for i, line in enumerate(svg):

        # Find a line with the dial hand (id="hand") and replace the angle
        if "id=\"hand\"" in line:
            svg[i] = line.replace("rotate(-90 ", "rotate({:.2f} ".format(hand_angle))

        # Insert the ZHR value
        if "ZHR_NUM" in line:
            svg[i] = line.replace("ZHR_NUM", "{:.0f}".format(zhr))

        # Set the size of the sporadic pie
        if "id=\"sporadic-portion\"" in line:
            svg[i] = line.replace("270 315 1@2e9970e6", "270 {:.2f} 1@2e9970e6".format(sporadic_pie_angle))

    
    # Merge the list of strings into a single string
    svg_str = "\n".join(svg)
            

    return svg_str



def generateWebsite(index_dir, flux_showers, ref_dt, results_all_years, results_ref_year, 
    website_plot_url, dial_svg_str, yearly_zhr_plot_name):
    

    # Decide which joining function to use, considering the given website URL or local path
    if os.path.isdir(website_plot_url):
        joinFunc = os.path.join

    else:
        joinFunc = lambda *pieces: '/'.join(s.strip('/') for s in pieces)

    html_code = ""

    # Define the website header
    website_header = r"""
<!DOCTYPE html>
<html>
<center>
<head>
        <meta charset="utf-8">
        <title>Meteor shower flux</title>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css" />
</head>

<body>


<body>

<table class="paddingBetweenCols">
<tbody>
<tr>
    <td>
        <!--<a href="https://www.nasa.gov/offices/meo/home/index.html" target="_blank"><img class="logo-imgl" src="https://fireballs.ndc.nasa.gov/static/nasa_logo.png" height="126" width="157" /></a>-->
    </td>
    <td>
        <a href="https://uwo.ca/" target="_blank"><img class="logo-imgl" src="https://globalmeteornetwork.org/static/images/uwo_logo_stacked_small.png" height="126" width="119" /></a> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 
    </td>

    <td>
        <center>
        <p><h1 class="heading">Meteor Shower <br> Flux Monitoring</h1></p>
                Supporting data supplied by the <a href="https://globalmeteornetwork.org/" target="_blank">Global Meteor Network</a>
        </center>
    </td>

    <td>
        &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 
        <a href="https://aquarid.physics.uwo.ca/index.html" target="_blank"><img class="logo-imgl" src="https://globalmeteornetwork.org/static/images/wmpg_banner_small.png" height="126" width="126" /></a>
    </td>
    <td>
        <a href="https://globalmeteornetwork.org/" target="_blank"><img class="logo-imgl" src="https://globalmeteornetwork.org/static/images/GMN_logo_white_text_dark_transparent_small.png" height="126" width="216" /></a>
    </td>
</tr>
</tbody>
</table>

    <hr>
    
    <h1>Meteor shower activity level</h1>
    <br>

"""
    html_code += website_header


    # Add the ZHR dial
    html_code += """
<p style="max-width: 600px; width: 100%;">
    """
    html_code += dial_svg_str
    html_code += """
    </p>
    """


    html_code += """
<div style="max-width: 800px; margin: 0px auto; text-align: left;">
    <br>
    <p>
        The dial shows the peak sum of activity from all currently active showers and the sporadic background 
        in the next 24 hours. The numbers are based on previous observations and predictions of future activity.
        The real activity might be lower or higher than the predicted value, as meteor showers can have 
        unexpected outbursts. See the plots below for the latest real-time meteor shower activity measurements.
    </p>

    <h3>How many meteors will I see?</h3>
    <p>
        ZHR stands for Zenithal Hourly Rate and is the number of meteors a single observer would see in 
        one hour in ideal conditions: under a clear, moonless dark sky with the radiant directly overhead. 
    </p>
    <p>
        The ZHR is not the number of meteors an observer will see in reality - for example, during the 
        peak of the Perseids when their ZHR is about 100, you can expect to see about one Persied every minute.
        However, if the Moon is full this number is easily halved, and if the radiant is low in the sky
        (below 30 degrees) you will see even less.
    </p>

    <h3>When should I observe?</h3>
    <p>
        An average observer will notice significantly increased meteor activity when the ZHR is above ~50 
        (the needle is in the green), which corresponds to the peak activity of the top 10 most active 
        showers.
    </p>
    <p>
        Outside rare meteor shower outbursts, the three showers which put on a regular annual show that is
        worth watching (ZHR > 100) are the Perseids (Aug 11 - 13), Geminids (Dec 13 - 14) and Quadrantids (Jan 3 - 4).
        <br>
        The plot below which summarizes the usual activity of annual meteor showers but excludes outbursts.
        Northern hemisphere showers (declination > 30 deg) are shown in blue, while southern hemisphere showers
        (declination < -30 deg) are shown in red (none currently). The showers in between are shown in black and are usually visible
        from both hemispheres.
    </p>
</div>
<br> """

    # Add the image with the yearly ZHR
    html_code += """
        <a href="{:s}" target="_blank"><img src="{:s}" style="width: 80%; height: auto;"/></a>""".format(
            joinFunc(website_plot_url, yearly_zhr_plot_name), 
            joinFunc(website_plot_url, yearly_zhr_plot_name)
            )
    
    html_code += """

<h1> Currently active showers </h1>
    """

    # Add the time of latest update (UTC) and solar longitude
    sol_ref = np.degrees(jd2SolLonSteyaert(datetime2JD(ref_dt)))
    update_time_str = """
<p><b>Last update: </b>
<br>{:s} UTC 
<br>Solar longitude {:.4f} deg
<br>(indicated by a red vertical line on the flux plots)
</p>""".format(
        ref_dt.strftime("%Y-%m-%d %H:%M:%S"), 
        sol_ref
    )
    html_code += update_time_str

    html_code += """
<br>
<p> 
Previous plots can be found here: <a href="{:s}">Archival data</a>
</p>
<p> 
Information about the data is provided in a section below: <a href="#about">About</a>
</p>
    """.format(website_plot_url)

    # Generate HTML with the latest results
    for shower_code in results_ref_year:

        # Extract reference year results object and names of the plots files
        shower_ref, _, plot_name_ref, plot_name_ref_full = results_ref_year[shower_code]

        # Print shower name
        shower_info = "<br><h2>#{:d} {:s} - {:s}</h2>".format(
            shower_ref.iau_code_int, 
            shower_ref.name, 
            shower_ref.name_full
            )
        html_code += shower_info

        # Add the image with latest flux
        img_ref_html = """
        <h3>Year {:d}</h3>
        <a href="{:s}" target="_blank"><img src="{:s}" style="width: 80%; height: auto;"/></a>""".format(
            ref_dt.year, 
            joinFunc(website_plot_url, plot_name_ref_full), 
            joinFunc(website_plot_url, plot_name_ref)
            )
        html_code += img_ref_html


        # Add the image with all years combined
        if shower_code in results_all_years:

            # Extract reference year results object and name of the plot file
            shower_all, dir_list_all, plot_name_all, plot_name_all_full = results_all_years[shower_code]

            # Determine the range of used years
            dt_list = [dt for _, dt in dir_list_all]
            year_min = min(dt_list).year
            year_max = max(dt_list).year

            # Add image with all years combined
            img_all_html = """
            <br>
            <h3>Years {:d} - {:d}</h3>
            <a href="{:s}" target="_blank"><img src="{:s}" style="width: 80%; height: auto;"/></a>""".format(
                year_min, year_max, 
                joinFunc(website_plot_url, plot_name_all_full),
                joinFunc(website_plot_url, plot_name_all)
                )
            html_code += img_all_html

    
    ### Generate a table with the used showers ###
    shower_table_html = """
<h1>Operational shower table</h1>
<div style="width: 1000px; margin: 0px auto;">
<div class="table-container">
    <table class="table table-striped table-responsive">
    <thead class="thead-default" >
        <tr>
        <th class="desc orderable">IAU #</th> 
        <th class="desc orderable">IAU code</th> 
        <th class="desc orderable">Name </th> 
        <th class="desc orderable"> Sol begin </th> 
        <th class="desc orderable"> Sol max </th> 
        <th class="desc orderable"> Sol end </th> 
        <th class="desc orderable"> Year </th> 
        <th class="desc orderable"> Population index </th>
        </tr>
    </thead>
<tbody>
    """

    for i, shower in enumerate(flux_showers.showers):

        if i%2 == 0:
            tr_class = "even"
        else:
            tr_class = "odd"

        shower_table_html += """<tr scope="row" class="{:s}">""".format(tr_class)
        #               <td>IAU #</td> <td>IAU code</td> <td>Name </td> <td> Sol begin </td> <td> Sol max </td> <td> Sol end </td> <td> Population index </td>
        shower_table_html += "<td>{:d}</td> <td>{:s}</td> <td>{:s}</td> <td> {:.2f} </td> <td> {:.2f} </td> <td> {:.2f} </td> <td> {:s} </td> <td> {:.2f} </td>".format(
            shower.iau_code_int, shower.name, shower.name_full, shower.lasun_beg, shower.lasun_max, 
            shower.lasun_end, shower.flux_year, shower.population_index)
        shower_table_html += "</tr>"

    shower_table_html += """
</tbody>
</table>
</div>
</div>
    """

    html_code += shower_table_html

    ###


    # Define the website footer
    website_footer = """
<h1 id="about">About</h1>
<div style="max-width: 800px; margin: 0px auto; text-align: left;">
    <p>
        The plots above show near-real time flux estimates computed as part of a collaboration between the NASA Meteoroid Environment Office and the Western University Meteor Physics group for major meteor showers. These data are gathered by video cameras of the Global Meteor Network - the hardware, software and flux methodology background are provided in the references below.
    </p>

    <p>
        The plots show the measured physical meteor flux (left hand axis) while the equivalent Zenithal Hourly Rate (number of meteors an observer would see under ideal skies with the radiant overhead in an hour) is shown on the right axis as a function of solar longitude (J2000). The assumed population index (which is used to convert flux to ZHR) is assumed constant over the duration of the shower is shown in the header. The equivalent flux to a limiting meteor absolute magnitude of +6.5 and the average limiting meteor magnitude for the measurements are shown, where the equivalent meteoroid mass uses the Mass-Magnitude-Velocity relationship of Verniani (1973). Uncertainties reflect Poisson statistics only (95% confidence interval).
    </p>

    <p>
        When a plot on the front page is clicked, a full plot including additional metadata will open. The second inset in the full plot shows the number of single station meteors associated to the shower by all cameras in varying time bins (black dots). The per bin equivalent time-area-product (TAP) of coverage in the atmosphere (in units of 1000 km<sup>2</sup> hr) is also shown. The time bin sizes are computed by requiring each bin to contain a minimum number of meteors (min meteors) and a minimum TAP.
    </p>

    <p>
        The third inset in the full plot show the average radiant distance from the cameras center field of view, the apparent elevation of the radiant (weighted by the TAP) and the moon phase (0 = new and 100 = full).
    </p>

    <p>
    Finally, the fourth inset shows the average meteor angular velocity (in degrees per second) measured in the cameras field center (black crosses) and the theoretical limiting detectable meteor magnitude taking all corrections into account also at the field center.
    </p>

    <p>
        Data are updated once per day on a global basis.
    </p>
     

    <b>References</b>
    <ul>
    <li>Verniani, F., 1973. An analysis of the physical parameters of 5759 faint radio meteors. <i>Journal of Geophysical Research</i>, 78(35), pp.8429-8462.</li>
    
    <li>Vida, D., Erskine, R.C.B., Brown, P.G., Kambulow, J., Campbell-Brown, M. and Mazur, M.J., 2022. Computing optical meteor flux using global meteor network data. <i>Monthly Notices of the Royal Astronomical Society</i>, in press. <a href="https://doi.org/10.1093/mnras/stac1766" target="_blank">MNRAS</a>, <a href="https://arxiv.org/abs/2206.11365" target="_blank">arxiv</a>.</li>
    
    <li>Vida, D., Segon, D., Gural, P.S., Brown, P.G., McIntyre, M.J., Dijkema, T.J., Pavletic, L., Kukic, P., Mazur, M.J., Eschman, P. and Roggemans, P., 2021. The Global Meteor Network - Methodology and first results. <i>Monthly Notices of the Royal Astronomical Society</i>, 506(4), pp.5046-5074. <a href="https://academic.oup.com/mnras/article-abstract/506/4/5046/6347233" target="_blank">MNRAS</a>, <a href="https://arxiv.org/abs/2107.12335" target="_blank">arxiv</a>.</li>
    </ul>
</div>
<br>
<h1 id="about">Data usage</h1>
<div style="max-width: 800px; margin: 0px auto; text-align: left;">
    <p>
    The data are released under the <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank">CC BY 4.0 license</a>. If you are using the data for scientific purposes, we kindly ask you to reference the following papers:

    <ul>
        <li>Vida, D., Segon, D., Gural, P.S., Brown, P.G., McIntyre, M.J., Dijkema, T.J., Pavletic, L., Kukic, P., Mazur, M.J., Eschman, P. and Roggemans, P., 2021. The Global Meteor Network - Methodology and first results. Monthly Notices of the Royal Astronomical Society, 506(4), pp.5046-5074. </li>
        <li>Vida, D., Blaauw Erskine, R.C., Brown, P.G., Kambulow, J., Campbell-Brown, M. and Mazur, M.J., 2022. Computing optical meteor flux using global meteor network data. Monthly Notices of the Royal Astronomical Society, 515(2), pp.2322-2339. </li>
    </ul>
    </p>
    <p>
    Also, we kindly ask you to add this text in the acknowledgements of any publications: 
    <br>
    <i>The Global Meteor Network (GMN) data are released under the CC BY 4.0 license. The authors acknowledge that the GMN data collection was supported in part by the NASA Meteoroid Environment Office under cooperative agreement 80NSSC21M0073 with the Western Meteor Physics Group.</i>
    </p>
</div>

<hr>
<footer>
<center>Supporting data supplied by the <a href="https://globalmeteornetwork.org/" target="_blank">Global Meteor Network</a>
    <br>    
<!--For more information, please email <a href="mailto:MSFC-fireballs@mail.nasa.gov?Subject=Flux%20Webpage" target="_top">MSFC-fireballs@mail.nasa.gov</a></center>-->
</footer>
</body>
</center>
</html>
    """
    html_code += website_footer


    # Save the HTML file
    with open(os.path.join(index_dir, "index.html"), 'w') as f:
        f.write(html_code)


def loadExcludedStations(dir_path, excluded_stations_file="excluded_stations.txt"):
    """ Load a list  of stations and dates to exclude from flux calculations. """

    # A dictionary where the keys are stations codes and the values are lists of periods of exclusion
    excluded_stations = {}
    with open(os.path.join(dir_path, excluded_stations_file)) as f:
        for line in f:

            # Skip comments
            if line.startswith("#"):
                continue

            line = line.replace("\n", "").replace("\r", "").strip()

            # Skip empty lines
            if not len(line):
                continue

            line = line.split("#", 1)[0]
            line = line.split(";")

            if len(line) < 2:
                continue

            station_code, exclusion_periods_temp = line
            exclusion_periods_temp = ast.literal_eval(exclusion_periods_temp)

            exclusion_periods = []
            for exclusion_range in exclusion_periods_temp:

                beg, end = exclusion_range

                # Convert time to string
                entries = []
                for entry in [beg, end]:
                    if entry is None:
                        entries.append(entry)
                    else:
                        time_str = "{:06d}_{:06d}.0".format(int(entry/1e6), int(entry%1e6))
                        entries.append(datetime.datetime.strptime(time_str, "%Y%m%d_%H%M%S.%f"))

                exclusion_periods.append(entries)

            excluded_stations[station_code] = exclusion_periods


        return excluded_stations


def fluxAutoRun(config, data_path, ref_dt, days_prev=2, days_next=1, all_prev_year_limit=3, \
    metadata_dir=None, output_dir=None, csv_dir=None, index_dir=None, generate_website=False, 
    website_plot_url=None, shower_code=None, shower_suffix_filename=None, custom_binning_dict=None,
    cpu_cores=1, excluded_stations_file="excluded_stations.txt",
    skip_allyear=False, publication_quality=False):
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
        all_prev_year_limit: [int] Only go back N years for the all data plot.
        metadata_dir: [str] A separate directory for flux metadata. If not given, the data directory will be
            used.
        output_dir: [str] Directory where the final data products will be saved. If None, data_path directory
            will be used.
        csv_dir: [str] Directory where the CSV files will be save. If None, output_dir will be used.
        index_dir: [str] Directory where index.html will be placed. If None, output_dir will be used.
        generate_website: [bool] Generate HTML code for the website. It will be saved in the output dir.
        website_plot_url: [str] Public URL to the plots, so they can be accessed online.
        shower_code: [str] Force a specific shower. None by default, in which case active showers will be
            automatically determined.
        shower_suffix_filename: [str] Suffix to add to the shower code in the file name.
        custom_binning_dict: [str] Custom binning parameters to be used instead of those given in the
            flux showers files. Needs to be a dictionary in the same format as in that file.
        cpu_cores: [int] Number of CPU cores to use. If -1, all availabe cores will be used. 1 by default.
        excluded_stations_file: [str] File with excluded stations and periods. It should be in the metadata
            directory.
        skip_allyear: [bool] Skip computing the flux with all years.
        publication_quality: [bool] Produce plots in publication quality. False by default.
    """


    if output_dir is None:
        output_dir = data_path

    else:
        if not os.path.exists(output_dir):
            mkdirP(output_dir)


    if csv_dir is None:
        csv_dir = output_dir

    else:
        if not os.path.exists(csv_dir):
            mkdirP(csv_dir)


    if index_dir is None:
        index_dir = output_dir

    else:
        if not os.path.exists(index_dir):
            mkdirP(index_dir)


    if website_plot_url is None:
        website_plot_url = output_dir


    if shower_suffix_filename is None:
        shower_suffix_filename = ''


    # Load the showers for flux
    flux_showers = FluxShowers(config)


    # Load excluded stations
    excluded_stations = {}
    excluded_stations_dir = metadata_dir
    if metadata_dir is None:
        excluded_stations_dir = data_path
    if os.path.isfile(os.path.join(excluded_stations_dir, excluded_stations_file)):
        excluded_stations = loadExcludedStations(
            excluded_stations_dir, 
            excluded_stations_file=excluded_stations_file
            )



    # If a specific shower was given, load it
    if shower_code is not None:

        # Find the given shower
        shower = flux_showers.showerObjectFromCode(shower_code)

        if shower is None:
            print("The shower {:s} could not be found in the list of showers for flux!")
            sys.exit()


        # Take the peak of the shower as reference
        sol_ref = shower.lasun_max

        # Compute the solar longitude of the reference time
        sol_ref_time = np.degrees(jd2SolLonSteyaert(datetime2JD(ref_dt)))

        # Compute the range of datetimes for the activity closest to the reference time
        sol_diff_beg = (shower.lasun_beg - sol_ref_time + 180)%360 - 180
        sol_diff_end = (shower.lasun_end - sol_ref_time + 180)%360 - 180
        sol_diff_max = (shower.lasun_max - sol_ref_time + 180)%360 - 180
        shower.dt_beg_ref_year = ref_dt + datetime.timedelta(days=sol_diff_beg*360/365.24219)
        shower.dt_end_ref_year = ref_dt + datetime.timedelta(days=sol_diff_end*360/365.24219)
        shower.dt_max_ref_year = ref_dt + datetime.timedelta(days=sol_diff_max*360/365.24219)

        # Add the shower to active showers
        active_showers = [shower]


    else:

        # Compute the solar longitude of the reference time
        sol_ref = np.degrees(jd2SolLonSteyaert(datetime2JD(ref_dt)))


        # Determine the time range for shower activity check
        dt_beg = ref_dt - datetime.timedelta(days=days_prev)
        dt_end = ref_dt + datetime.timedelta(days=days_next)

        # Get a list of showers active now
        active_showers = flux_showers.activeShowers(dt_beg, dt_end, use_zhr_threshold=False)

        print("Active showers:")
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


    # Create a dictionary of active showers where the shower codes are the key
    active_showers_dict = {shower.name:shower for shower in active_showers}


    ### Load all data folders ###

    # Determine which data folders should be used for each shower (don't search deeper than a depth of 2)
    shower_dirs = {}
    shower_dirs_ref_year = {}
    for entry in walkDirsToDepth(data_path, depth=2):

        dir_entry, dir_list, file_list = entry

        # Go though all directories
        for dir_name in dir_list:

            dir_path = os.path.join(dir_entry, dir_name)

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


            # Check if the station should be excluded at this time, according to the station exclusion file
            station_code = dir_split[0]
            skip_dir = False
            if station_code in excluded_stations:

                # Check all exclusion periods
                for excluded_period in excluded_stations[station_code]:

                    beg, end = excluded_period

                    # Skip if the station is always excluded
                    if (beg is None) and (end is None):
                        skip_dir = True
                        break

                    elif (beg is None):
                        if dir_dt < end:
                            skip_dir = True
                            break

                    elif (end is None):
                        if dir_dt > beg:
                            skip_dir = True
                            break

                    else:
                        if (dir_dt > beg) and (dir_dt < end):
                            skip_dir = True
                            break

            if skip_dir:
                print("Excluding:", dir_name)
                continue



            # Make sure the directory time is after 2018 (to avoid 1970 unix time 0 dirs)
            #   2018 is when the GMN was established
            if dir_dt.year < 2018:
                continue

            # Skip dirs that are too old to add to the all year plot
            if (ref_dt.year - dir_dt.year) >= all_prev_year_limit:
                print("Skipping due to {:d} year limit!".format(all_prev_year_limit))
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
                    if len([file_name for file_name in os.listdir(dir_path)
                            if file_name == config.platepar_name]):

                        # Add the directory to the list if it doesn't exist already
                        shower_dirs_entry = [dir_path, dir_dt]

                        if shower_dirs_entry not in shower_dirs[shower.name]:
                            shower_dirs[shower.name].append(shower_dirs_entry)

                            # print("Ref year check:")
                            # print(dir_dt, shower.dt_beg_ref_year - datetime.timedelta(days=1)) 
                            # print(dir_dt, shower.dt_end_ref_year + datetime.timedelta(days=1))
                            # print()

                            # Store the reference year's directories separately
                            if (dir_dt >= shower.dt_beg_ref_year - datetime.timedelta(days=1)) and \
                               (dir_dt <= shower.dt_end_ref_year + datetime.timedelta(days=1)):

                               shower_dirs_ref_year[shower.name].append([dir_path, dir_dt])


    ### ###

    # Define binning parameters for all years
    fluxbatch_binning_params_all_years = FluxBatchBinningParams(
        min_meteors=20,
        min_tap=100,
        min_bin_duration=1.0,
        max_bin_duration=12
        )

    # Define binning parameters for individual years
    fluxbatch_binning_params_one_year = FluxBatchBinningParams(
        min_meteors=10,
        min_tap=50,
        min_bin_duration=1.0,
        max_bin_duration=12
        )


    # Store results in a dictionary where the keys are shower codes
    results_all_years = {}
    results_ref_year = {}


    # Make a list of shower parameters
    shower_params = []
    
    # Add yearly data
    shower_params.append([shower_dirs_ref_year, "REF", fluxbatch_binning_params_one_year])

    # Add all year data
    if not skip_allyear:
        shower_params.append([     shower_dirs, "ALL", fluxbatch_binning_params_all_years])


    # Process batch fluxes for all showers
    #   2 sets of plots and CSV files will be saved: one set with all years combined, and one set with the
    #   reference year
    for shower_dir_dict, time_extent_flag, fb_bin_params in shower_params:
        
        for shower_code in shower_dir_dict:

            shower = active_showers_dict[shower_code]
            dir_list = shower_dir_dict[shower_code]


            # Skip all-year fluxes if a one-year outburst way given
            if time_extent_flag == "ALL":
                if not shower.isAnnual():
                    continue


            # Load the reference height if given
            ref_height = -1
            if shower.ref_height is not None:
                ref_height = shower.ref_height



            # Use custom binning parameters if they are given
            if custom_binning_dict is not None:
                shower_flux_binning_params = custom_binning_dict

            # Otherwise, use the binning parameters specified in the flux showers csv file
            else:
                shower_flux_binning_params = shower.flux_binning_params


            # Load the binning parameters if given
            if shower_flux_binning_params is not None:

                # Select multi-year plotting options
                if time_extent_flag == "ALL":
                    if 'all_years' in shower_flux_binning_params:
                        fb_bin_params = FluxBatchBinningParams(**shower_flux_binning_params['all_years'])

                else:
                    if 'yearly' in shower_flux_binning_params:
                        fb_bin_params = FluxBatchBinningParams(**shower_flux_binning_params['yearly'])

                


            # Construct the dir input list
            dir_params = [(night_dir_path, None, None, None, None, None) for night_dir_path, _ in dir_list]

            # Compute the batch flux
            fbr = fluxBatch(config, shower_code, shower.mass_index, dir_params, ref_ht=ref_height, 
                min_meteors=fb_bin_params.min_meteors, 
                min_tap=fb_bin_params.min_tap, 
                min_bin_duration=fb_bin_params.min_bin_duration, 
                max_bin_duration=fb_bin_params.max_bin_duration, 
                compute_single=False,
                metadata_dir=metadata_dir,
                cpu_cores=cpu_cores,
                )


            if time_extent_flag == "ALL":
                plot_suffix = "all_years"
            else:
                plot_suffix = "year_{:d}".format(shower.dt_max_ref_year.year)

            # Make a name for the plots to save (only flux + full metadata plot)
            batch_flux_output_filename = "flux_{:s}{:s}_sol={:.2f}-{:.2f}_{:s}".format(shower_code, 
                shower_suffix_filename, fbr.shower.lasun_beg, fbr.shower.lasun_end, plot_suffix)
            batch_flux_output_filename_full = batch_flux_output_filename + "_full"

            # Save the results metadata to a dictionary
            if time_extent_flag == "ALL":
                results_all_years[shower_code] = [
                    shower, 
                    dir_list, 
                    batch_flux_output_filename + '.png',
                    batch_flux_output_filename_full + '.png'
                    ]

            else:
                results_ref_year[shower_code] = [
                    shower, 
                    dir_list, 
                    batch_flux_output_filename + '.png',
                    batch_flux_output_filename_full + '.png'
                    ]

            # Save the batch flux plot (only flux)
            plotBatchFlux(
                fbr, 
                output_dir,
                batch_flux_output_filename,
                only_flux=True,
                compute_single=False,
                show_plot=False,
                xlim_shower_limits=True,
                sol_marker=sol_ref,
                publication_quality=publication_quality
            )

            # Save the batch flux plot (full metadata)
            plotBatchFlux(
                fbr, 
                output_dir,
                batch_flux_output_filename_full,
                only_flux=False,
                compute_single=False,
                show_plot=False,
                xlim_shower_limits=True,
                sol_marker=None,
                publication_quality=publication_quality
            )

            # Save the results to a CSV file
            saveBatchFluxCSV(fbr, csv_dir, batch_flux_output_filename)

            # Save the per-camera tally results
            tally_string = reportCameraTally(fbr, top_n_stations=5)
            with open(os.path.join(output_dir, batch_flux_output_filename + "_camera_tally.txt"), 'w') as f:
                f.write(tally_string)

            # Delete the flux results object to free up memory
            del fbr


    # Generate the website HTML code
    if generate_website:

        print("Generating website...")

        # Load the flux activity file
        shower_models = loadFluxActivity(config)

        # Compute the current peak ZHR
        peak_zhr = computeCurrentPeakZHR(shower_models, sporadic_zhr=config.background_sporadic_zhr)
        
        # Set the ZHR dial
        dial_svg_str = generateZHRDialSVG(config.flux_dial_template_svg, peak_zhr, 
                                          config.background_sporadic_zhr)
        
        try:
            # Make a plot of the ZHR across the year
            plotYearlyZHR(config, os.path.join(output_dir, config.yearly_zhr_plot_name), 
                          sporadic_zhr=config.background_sporadic_zhr)
        
        except:
            # Log the error
            print("Error generating yearly ZHR plot!")
            traceback.print_exc()

        # Generate the website
        generateWebsite(index_dir, flux_showers, ref_dt, results_all_years, results_ref_year, 
            website_plot_url, dial_svg_str, config.yearly_zhr_plot_name)
        

        print("   ... done!")



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

    arg_parser.add_argument('-o', '--outdir', metavar='OUTPUT_DIRECTORY', type=str,
        help="Path to a directory where the plots will be saved. If not given, the data directory will be used.")

    arg_parser.add_argument('-c', '--csvdir', metavar='CSV_DIRECTORY', type=str,
        help="Path to a directory where the CSV files will be saved. If not given, the output directory will be used.")

    arg_parser.add_argument('-i', '--indexdir', metavar='INDEX_DIRECTORY', type=str,
        help="Path to the directory where index.html will be placed. If not given, the output directory will be used.")

    arg_parser.add_argument('-w', '--weburl', metavar='WEBSITE_PLOT_PUBLIC_URL', type=str,
        help="Generate a website HTML with the given public URL to where the plots are stored on the website.")

    arg_parser.add_argument('-s', '--shower', metavar='SHOWER', type=str,
        help="Force a specific shower. 3-letter IAU shower code is expected.")

    arg_parser.add_argument('--suffix', metavar='SUFFIX', type=str,
        help="Add a suffix to the shower name to differentiate the flux run it in case something special was done.")

    arg_parser.add_argument('--binning', metavar='BINNING', type=str,
        help="""Specify custom binning parameters instead of those used in the flux shower file. Usage example:
        --binning "{'all_years': {'min_tap':   50, 'min_meteors':  35, 'min_bin_duration': 0.5, 'max_bin_duration': 12}, 'yearly': {'min_tap':  30, 'min_meteors':  20, 'min_bin_duration': 0.5, 'max_bin_duration': 12}}" """)

    arg_parser.add_argument('-a', '--auto', metavar='H_FREQ', type=float, default=None, const=1.0, 
        nargs='?',
        help="""Run continously every H_FREQ hours. If argument not given, the code will run every hour."""
        )

    arg_parser.add_argument(
        "--cpucores",
        type=int,
        default=1,
        help="Number of CPU codes to use for computation. -1 to use all cores. 1 by default.",
    )

    arg_parser.add_argument('--skipallyear', action="store_true", \
        help="""Skip computing multi-year fluxes. Only compute fluxes for the given year.""")
    
    arg_parser.add_argument('--publication', action="store_true", \
        help="""Produce plots in publication quality.""")


    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    # Load the default config file
    config = cr.Config()
    config = cr.parse(config.config_file_name)


    # Load the custom binning as a dictionary
    custom_binning_dict = None
    if cml_args.binning is not None:
        
        # Strip quotes from ends of string, convert all apostrophes to quotes
        binning_formatted = cml_args.binning.strip('"').strip("'").replace("'", '"')

        # Load JSON as dictionary
        custom_binning_dict = json.loads(binning_formatted)



    previous_start_time = None
    while True:

        # Clock for measuring script time
        t1 = datetime.datetime.utcnow()


        if cml_args.time is not None:
            ref_dt = datetime.datetime.strptime(cml_args.time[0], "%Y%m%d_%H%M%S.%f")

        # If no manual time was given, use current time.
        else:
            ref_dt = datetime.datetime.utcnow()


        print("Computing flux using reference time:", ref_dt)

        # Run auto flux
        fluxAutoRun(config, cml_args.dir_path, ref_dt, metadata_dir=cml_args.metadir,
            output_dir=cml_args.outdir, csv_dir=cml_args.csvdir, 
            generate_website=(cml_args.weburl is not None), index_dir=cml_args.indexdir, 
            website_plot_url=cml_args.weburl, shower_code=cml_args.shower, \
            shower_suffix_filename=cml_args.suffix, custom_binning_dict=custom_binning_dict,
            cpu_cores=cml_args.cpucores,
            skip_allyear=cml_args.skipallyear,
            publication_quality=cml_args.publication
            )


        ### <// DETERMINE NEXT RUN TIME ###

        # Store the previous start time
        previous_start_time = copy.deepcopy(t1)

        # Break if only running once or a specific time or shower was given
        if (cml_args.auto is None) or (cml_args.time is not None) or (cml_args.shower is not None):
            break

        else:

            # Otherwise wait to run
            wait_time = (datetime.timedelta(hours=cml_args.auto) \
                - (datetime.datetime.utcnow() - t1)).total_seconds()

            # Run immediately if the wait time has elapsed
            if wait_time < 0:
                continue

            # Otherwise wait to run
            else:

                # Compute next run time
                next_run_time = datetime.datetime.now() + datetime.timedelta(seconds=wait_time)

                # Wait to run
                while next_run_time > datetime.datetime.now():
                    print("Waiting {:s} to run the fluxes...                ".format(str(next_run_time \
                        - datetime.datetime.now())), end='\r')
                    time.sleep(2)


        ### DETERMINE NEXT RUN TIME //> ###