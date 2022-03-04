""" Compute single-station meteor shower flux. """

import argparse
import collections
import datetime
import json
import os
import sys

import astropy.table
import astropy.units
import astropy.coordinates
import ephem
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


from RMS.Astrometry.ApplyAstrometry import (
    getFOVSelectionRadius,
    raDecToXYPP,
    xyToRaDecPP,
)
from RMS.Astrometry.ApplyRecalibrate import applyRecalibrate, loadRecalibratedPlatepar, recalibrateSelectedFF
from RMS.Astrometry.Conversions import J2000_JD, areaGeoPolygon, date2JD, datetime2JD, jd2Date, raDec2AltAz
from RMS.ExtractStars import extractStarsAndSave
from RMS.Formats import FFfile, Platepar, StarCatalog
from RMS.Formats.CALSTARS import readCALSTARS
import RMS.Formats.CALSTARS as CALSTARS
from RMS.Formats.FTPdetectinfo import findFTPdetectinfoFile, readFTPdetectinfo
from RMS.Formats.Showers import FluxShowers, loadRadiantShowers
from RMS.Math import angularSeparation, pointInsideConvexPolygonSphere
from RMS.Routines.FOVArea import fovArea, xyHt2Geo
from RMS.Routines.MaskImage import MaskStructure, getMaskFile
from RMS.Routines.SolarLongitude import jd2SolLonSteyaert, solLon2jdSteyaert, unwrapSol

from Utils.ShowerAssociation import heightModel, showerAssociation



# CONSTANTS

FIXED_BINS_NAME = "flux_fixed_bins"

FLUX_TIME_INTERVALS_JSON = "flux_time_intervals.json"

#


class FluxConfig(object):
    def __init__(self):
        """Container for flux calculations."""

        # How many points to use to evaluate the FOV on seach side of the image. Normalized to the longest
        #   side.
        self.side_points = 20

        # Minimum height (km).
        self.ht_min = 60

        # Maximum height (km).
        self.ht_max = 130

        # Height sampling delta (km).
        self.dht = 2

        # Limit of meteor's elevation above horizon (deg). 25 degrees by default.
        self.elev_limit = 20

        # Minimum radiant elevation in the time bin (deg). 25 degreees by default
        self.rad_elev_limit = 15

        # Radiant elevation correction exponent
        # See: Molau & Barentsen (2013) - Meteoroid stream flux densities and the zenith exponent
        self.gamma = 1.5

        # Minimum distance of the end of the meteor to the radiant (deg)
        self.rad_dist_min = 15

        # Subdivide the time bin into the given number of subbins
        self.sub_time_bins = 2

        # Minimum number of meteors in the time bin
        self.meteors_min = 3

        # Default star FWHM, it it's not available (pz)
        self.default_fwhm = 3


class FluxMeasurements(object):
    def __init__(self):
        """ Container for flux measurements. """

        # Init the data arrays
        self.resetDataArrays()


        ### ECSV parameters ###

        self.format_version = 1.0
        self.format_date = "2022-02-24"
        self.format_version_str = "{:.3f} ({:s})".format(self.format_version, self.format_date)

        ### ###

    def initMetadata(self, shower_code, mass_index, population_index, gamma, shower_velocity, shower_height, \
        star_fwhm, mean_ang_vel, mean_sensitivity, mean_range, raw_col_area, ci):

        self.shower_code = shower_code
        self.mass_index = mass_index
        self.population_index = population_index
        self.gamma = gamma
        self.shower_velocity = shower_velocity
        self.shower_height = shower_height
        self.star_fwhm = star_fwhm
        self.mean_ang_vel = mean_ang_vel
        self.mean_sensitivity = mean_sensitivity
        self.mean_range = mean_range
        self.raw_col_area = raw_col_area
        self.ci = ci

    def resetDataArrays(self):

        # Initialize empty table
        self.table = astropy.table.Table()

        # Init data arrays
        self.sol_lon_data = []
        self.times_data = []
        self.meteors_data = []
        self.rad_elev_data = []
        self.rad_dist_data = []
        self.ang_vel_data = []
        self.total_corrections_data = []
        self.eff_col_area_data = []
        self.eff_col_area_6_5_lm_data = []
        self.time_bin_data = []
        self.stellar_lm_data = []
        self.meteor_lm_data = []
        self.flux_meteor_lm_data = []
        self.flux_6_5_lm_data = []
        self.flux_6_5_lm_ci_lower_data = []
        self.flux_6_5_lm_ci_upper_data = []


    def addEntry(self, sol_lon, times, meteors, rad_elev, rad_dist, ang_vel, total_corrections, \
        eff_col_area, eff_col_area_6_5_lm, time_bin, stellar_lm, meteor_lm, flux_meteor_lm, flux_6_5_lm, \
        flux_6_5_lm_ci_lower, flux_6_5_lm_ci_upper):
        """ Add entry to the flux data. """


        self.sol_lon_data.append(sol_lon)
        self.times_data.append(times)
        self.meteors_data.append(meteors)
        self.rad_elev_data.append(rad_elev)
        self.rad_dist_data.append(rad_dist)
        self.ang_vel_data.append(ang_vel)
        self.total_corrections_data.append(total_corrections)
        self.eff_col_area_data.append(eff_col_area)
        self.eff_col_area_6_5_lm_data.append(eff_col_area_6_5_lm)
        self.time_bin_data.append(time_bin)
        self.stellar_lm_data.append(stellar_lm)
        self.meteor_lm_data.append(meteor_lm)
        self.flux_meteor_lm_data.append(flux_meteor_lm)
        self.flux_6_5_lm_data.append(flux_6_5_lm)
        self.flux_6_5_lm_ci_lower_data.append(flux_6_5_lm_ci_lower)
        self.flux_6_5_lm_ci_upper_data.append(flux_6_5_lm_ci_upper)


    def saveECSV(self, ecsv_file_path):
        """ Save the ECSV file with flux data. """

        ### Initialize metadata ###

        self.table.meta['description'] = "Global Meteor Network flux measurements"
        self.table.meta['format_version'] = self.format_version_str
        self.table.meta['shower'] = str(self.shower_code)
        self.table.meta['mass_index'] = self.mass_index
        self.table.meta['population_index'] = self.population_index
        self.table.meta['gamma'] = self.gamma
        self.table.meta['shower_velocity'] = astropy.units.Quantity(self.shower_velocity, \
            unit=astropy.units.km/astropy.units.s)
        self.table.meta['shower_height'] = astropy.units.Quantity(self.shower_height, unit=astropy.units.km)
        self.table.meta['star_fwhm'] = astropy.units.Quantity(self.star_fwhm, unit=astropy.units.pix)
        self.table.meta['mean_ang_vel'] = astropy.units.Quantity(self.mean_ang_vel, \
            unit=astropy.units.deg/astropy.units.s)
        self.table.meta['mean_sensitivity'] = self.mean_sensitivity
        self.table.meta['mean_range'] = astropy.units.Quantity(self.mean_range, unit=astropy.units.km)
        self.table.meta['raw_col_area'] = astropy.units.Quantity(self.raw_col_area, unit=1000*astropy.units.km**2)
        self.table.meta['confidence_interval'] = self.ci

        ### ###

        ### Initialize columns ###

        formats = {}

        # Solar longitude (deg)
        sol_lon_col = astropy.coordinates.Angle(self.sol_lon_data, unit=astropy.units.deg)
        sol_lon_col.info.description = "Solar longitude of bin edges"
        self.table['sol'] = sol_lon_col
        formats['sol'] = "%.8f"

        # Time (UTC)
        time_col = astropy.time.Time(self.times_data, format='datetime')
        time_col.info.description = "UTC time of bin edges"
        self.table['time'] = time_col

        # Meteors
        num_meteors_col = astropy.units.Quantity(self.meteors_data)
        num_meteors_col.info.description = "Numer of meteors in the bin"
        self.table['meteors'] = num_meteors_col
        formats['meteors'] = "%d"

        # Radiant elevation
        rad_elev_col = astropy.coordinates.Angle(self.rad_elev_data, unit=astropy.units.deg)
        rad_elev_col.info.description = "Radiant elevation"
        self.table['rad_elev'] = rad_elev_col
        formats['rad_elev'] = "%.3f"

        # Radiant distance
        rad_dist_col = astropy.coordinates.Angle(self.rad_dist_data, unit=astropy.units.deg)
        rad_dist_col.info.description = "Radiant distance from the FOV center"
        self.table['rad_dist'] = rad_dist_col
        formats['rad_dist'] = "%.3f"

        # Angular velocity
        ang_vel_col = astropy.units.Quantity(self.ang_vel_data, unit=astropy.units.deg/astropy.units.s)
        ang_vel_col.info.description = "Angular velocity at the FOV center"
        self.table['ang_vel'] = ang_vel_col
        formats['ang_vel'] = "%.3f"

        # Total corrections
        total_corrections_col = astropy.units.Quantity(self.total_corrections_data)
        total_corrections_col.info.description = "Total correction applied to the raw collecting area"
        self.table['total_corrections'] = total_corrections_col
        formats['total_corrections'] = "%.4f"

        # Corrected effective collection area to the observed limiting magnitude
        eff_col_area_col = astropy.units.Quantity(self.eff_col_area_data, unit=astropy.units.km**2)
        eff_col_area_col.info.description = "Effective collecting area corrected to observed meteor magnitude"
        self.table['eff_col_area'] = eff_col_area_col
        formats['eff_col_area'] = "%.2f"

        # Corrected effective collection area to +6.5M (km^2)
        eff_col_area_6_5_lm_col = astropy.units.Quantity(self.eff_col_area_6_5_lm_data, \
            unit=astropy.units.km**2)
        eff_col_area_6_5_lm_col.info.description = "Effective collecting area corrected to +6.5M meteor magnitude"
        self.table['eff_col_area_6_5_lm'] = eff_col_area_6_5_lm_col
        formats['eff_col_area_6_5_lm'] = "%.2f"

        # Time bin (hours)
        time_bin_col = astropy.units.Quantity(self.time_bin_data, unit=astropy.units.h)
        time_bin_col.info.description = "Bin duration"
        self.table['time_bin'] = time_bin_col
        formats['time_bin'] = "%.6f"

        # Stellar LM
        stellar_lm_col = astropy.units.Quantity(self.stellar_lm_data, unit=astropy.units.mag)
        stellar_lm_col.info.description = "Stellar limiting magnitude"
        self.table['stellar_lm'] = stellar_lm_col
        formats['stellar_lm'] = "%.3f"

        # Meteor LM (mag)
        meteor_lm_col = astropy.units.Quantity(self.meteor_lm_data, unit=astropy.units.mag)
        meteor_lm_col.info.description = "Meteor limiting magnitude at the FOV center"
        self.table['meteor_lm'] = meteor_lm_col
        formats['meteor_lm'] = "%.3f"

        # Flux at meteor magnitude
        flux_meteor_lm_col = astropy.units.Quantity(self.flux_meteor_lm_data, \
            unit=1.0/((1000*astropy.units.km**2)*astropy.units.h))
        flux_meteor_lm_col.info.description = "Flux at the meteor LM"
        self.table['flux_meteor_lm'] = flux_meteor_lm_col
        formats['flux_meteor_lm'] = "%.3f"

        # Flux at +6.5M
        flux_6_5_lm_col = astropy.units.Quantity(self.flux_6_5_lm_data, \
            unit=1.0/((1000*astropy.units.km**2)*astropy.units.h))
        flux_6_5_lm_col.info.description = "Flux at +6.5M"
        self.table['flux_6_5_lm'] = flux_6_5_lm_col
        formats['flux_6_5_lm'] = "%.3f"

        # Flux at +6.5M, lower confidence interval limit
        flux_6_5_lm_ci_lower_col = astropy.units.Quantity(self.flux_6_5_lm_ci_lower_data, \
            unit=1.0/((1000*astropy.units.km**2)*astropy.units.h))
        flux_6_5_lm_ci_lower_col.info.description = "Flux at +6.5M, lower confidence interval limit"
        self.table['flux_6_5_lm_ci_lower'] = flux_6_5_lm_ci_lower_col
        formats['flux_6_5_lm_ci_lower'] = "%.3f"

        # Flux at +6.5M, upper confidence interval limit
        flux_6_5_lm_ci_upper_col = astropy.units.Quantity(self.flux_6_5_lm_ci_upper_data, \
            unit=1.0/((1000*astropy.units.km**2)*astropy.units.h))
        flux_6_5_lm_ci_upper_col.info.description = "Flux at +6.5M, upper confidence interval limit"
        self.table['flux_6_5_lm_ci_upper'] = flux_6_5_lm_ci_upper_col
        formats['flux_6_5_lm_ci_upper'] = "%.3f"

        ### ###


        # Save the ECSV file to disk
        self.table.write(ecsv_file_path, format='ascii.ecsv', delimiter=',', overwrite=True, formats=formats,
            serialize_method={astropy.time.Time: 'jd1_jd2'})


    def loadECSV(self, ecsv_file_path):

        # Load the ECSV data
        self.table =  astropy.table.Table.read(ecsv_file_path, delimiter=',', format='ascii.ecsv', \
            guess=False)



def calculatePopulationIndex(mass_index):
    """ Compute the population index given the mass index. """

    return 10**((mass_index - 1)/2.5)


def generateColAreaJSONFileName(station_code, side_points, ht_min, ht_max, dht, elev_limit):
    """Generate a file name for the collection area JSON file."""

    file_name = "flux_col_areas_{:s}_sp-{:d}_htmin-{:.1f}_htmax-{:.1f}_dht-{:.1f}_elemin-{:.1f}.json".format(
        station_code, side_points, ht_min, ht_max, dht, elev_limit
    )

    return file_name


def saveRawCollectionAreas(dir_path, file_name, col_areas_ht):
    """Save the raw collection area calculations so they don't have to be regenerated every time."""

    file_path = os.path.join(dir_path, file_name)

    with open(file_path, 'w') as f:

        # Convert tuple keys (x_mid, y_mid) to str keys
        col_areas_ht_strkeys = {}
        for key in col_areas_ht:
            col_areas_ht_strkeys[key] = {}

            for tuple_key in col_areas_ht[key]:

                str_key = "{:.2f}, {:.2f}".format(*tuple_key)

                col_areas_ht_strkeys[key][str_key] = col_areas_ht[key][tuple_key]

        # Add an explanation what each entry means
        col_areas_ht_strkeys[-1] = {
            "height (m)": {
                "x (px), y (px) of pixel block": [
                    "area (m^2)",
                    "azimuth +E of due N (deg)",
                    "elevation (deg)",
                    "sensitivity",
                    "range (m)",
                ]
            }
        }

        # Convert collection areas to JSON
        out_str = json.dumps(col_areas_ht_strkeys, indent=4, sort_keys=True)

        # Save to disk
        f.write(out_str)


def loadRawCollectionAreas(dir_path, file_name):
    """Read raw collection areas from disk. """

    file_path = os.path.join(dir_path, file_name)

    # Load the JSON file
    with open(file_path) as f:
        data = " ".join(f.readlines())

        col_areas_ht_strkeys = json.loads(data)

        # Convert tuple keys (x_mid, y_mid) to str keys
        col_areas_ht = collections.OrderedDict()
        for key in col_areas_ht_strkeys:

            # Skip heights below 0 (the info key)
            if float(key) < 0:
                continue

            col_areas_ht[float(key)] = collections.OrderedDict()

            for str_key in col_areas_ht_strkeys[key]:

                # Convert the string "x_mid, y_mid" to tuple of floats (x_mid, y_mid)
                tuple_key = tuple(map(float, str_key.split(", ")))

                col_areas_ht[float(key)][tuple_key] = col_areas_ht_strkeys[key][str_key]

    return col_areas_ht




def saveTimeInvervals(config, dir_path, time_intervals):
    """ Save observing time intervals as determined by the cloud detector. 
    
    Arguments:
        config: [Config]
        dir_path: [str] Data directory
        time_intervals: [list] A list of observing time intervals in the (dt_beg, dt_end) format.

    Return:
        None

    """


    def _jsonFormatter(o):

        # Convert datetimes to string
        if isinstance(o, datetime.datetime):
            return o.strftime("%Y-%m-%dT%H:%M:%S.%f")


    time_inverval_dict = {}
    time_inverval_dict["time_intervals"] = time_intervals
    time_inverval_dict["stationID"] = config.stationID

    # Convert time intervals to a JSON string
    time_interval_json = json.dumps(time_inverval_dict, default=_jsonFormatter, indent=4, sort_keys=True)

    with open(os.path.join(dir_path, FLUX_TIME_INTERVALS_JSON), 'w') as f:
        f.write(time_interval_json)



def loadTimeInvervals(config, dir_path):
    """ Load observing time intervals as determined by the cloud detector. 
    
    Arguments:
        config: [Config]
        dir_path: [str] Data directory

    Return:
        time_intervals: [list] A list of observing time intervals in the (dt_beg, dt_end) format.

    """

    json_file_path = os.path.join(dir_path, FLUX_TIME_INTERVALS_JSON)

    if os.path.isfile(json_file_path):

        with open(json_file_path) as f:

            time_interval_json = " ".join(f.readlines())

            # Load time intervals back in
            dt_json = json.loads(time_interval_json)

            # Check that the station ID is correct
            if str(dt_json['stationID']) == str(config.stationID):

                dt_list = []
                for entry in dt_json["time_intervals"]:
                    dt_range = []
                    for dt in entry:
                        dt_range.append(datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S.%f"))
                    dt_list.append(dt_range)


                return dt_list


    # If the file is not found or the station ID does not match, return nothing
    return None



def calculateFixedBins(all_time_intervals, dir_list, shower, bin_duration=5):
    """
    Function to calculate the bins that any amount of stations over any number of years for one shower
    can be put into.

    Arguments:
        all_time_intervals: [list] A list of observing time intervals in the (dt_beg, dt_end) format.
        dir_list: [list] List of directories to check for existing flux fixed bin files.
        shower: [Shower object]

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

    # Compute the bin duration in solar longitudes
    sol_delta = 2*np.pi/60/24/365.24219*bin_duration

    # Convert begin and end of all time intervals into solar longitudes
    sol_beg = np.array([jd2SolLonSteyaert(datetime2JD(beg)) for beg, _ in all_time_intervals])
    sol_end = np.array([jd2SolLonSteyaert(datetime2JD(end)) for _, end in all_time_intervals])

    # Even if the solar longitude wrapped around 0/360, make sure that you know what the smallest sol are
    # The assumption here is that the interval is never longer than 180 deg sol
    if ((np.max(sol_beg) - np.min(sol_beg)) > np.pi) or ((np.max(sol_end) - np.min(sol_end)) > np.pi):
        start_idx = np.argmin(np.where(sol_beg > np.pi, sol_beg, 2*np.pi))
        end_idx = np.argmax(np.where(sol_end <= np.pi, sol_beg, 0))

    else:
        start_idx = np.argmin(sol_beg)
        end_idx = np.argmax(sol_end)

    min_sol = sol_beg[start_idx]
    max_sol = sol_end[end_idx] if sol_beg[start_idx] < sol_end[end_idx] else sol_end[end_idx] + 2*np.pi
    sol_bins = np.arange(min_sol, max_sol, sol_delta)
    sol_bins = np.append(sol_bins, sol_bins[-1] + sol_delta)  # all events should be within the bins

    # Make sure that fixed bins fit with already existing bins saved
    existing_sol = []
    dirs_with_found_files = []
    for dir_name in dir_list:
        loaded_sol = []
        for file_name in sorted(os.listdir(dir_name)):

            # Take precomputed time bins for the right shower and mass index
            if checkFluxFixedBinsName(file_name, shower.name, shower.mass_index):

                # Load the solar longitude edges
                sol_bins_loaded = loadForcedBinFluxData(dir_name, file_name)[0]

                if len(sol_bins_loaded):

                    loaded_sol.append(sol_bins_loaded)
                    dirs_with_found_files.append(dir_name)


        if loaded_sol:
            existing_sol.append(loaded_sol)

        # does not check to make sure that none of the intervals during a single night overlap. User must
        # make sure of this

    ## calculating sol_bins
    if existing_sol:
        # select a starting_sol to transform sol_bins to so that it matched what already exists
        
        if len(existing_sol) == 1:
            starting_sol = existing_sol[0][0]

        else:
            # if there's more than one array of sol values, make sure they all agree with each other and
            # take the first
            failed = False
            comparison_sol = existing_sol[0][0]
            for loaded_sol_list, dir_name in zip(existing_sol[1:], dirs_with_found_files[1:]):

                # Take the first of the list and assume the others agree with it
                sol = loaded_sol_list[0]
                min_len = min(len(comparison_sol), len(sol), 5)
                
                a, b = (
                    (comparison_sol[:min_len], sol[:min_len])
                    if comparison_sol[0] > sol[0]
                    else (sol[:min_len], comparison_sol[:min_len])
                )

                epsilon = 1e-7
                goal = sol_delta/2
                
                val = (np.median(a - b if a[0] - b[0] < np.pi else b + 2*np.pi - a) + goal)%sol_delta

                if np.abs(goal - val) > epsilon:
                    print(
                        "!!! {:s} CSV in {:s} and {:s} don't match solar longitude values".format( \
                            FIXED_BINS_NAME, dirs_with_found_files[0], dir_name)
                    )
                    print('\tSolar longitude difference:', np.abs(goal - val))
                    failed = True

            if failed:
                print()
                raise Exception(
                    "Flux bin solar longitudes didn't match. To fix this, at least one of the"
                    " {:s} CSV files must be deleted.".format(FIXED_BINS_NAME)
                )
            # filter only sol values that are inside the solar longitude
            starting_sol = comparison_sol

        # adjust bins to fit existing bins
        length = min(len(starting_sol), len(sol_bins))
        sol_bins += np.mean(starting_sol[:length] - sol_bins[:length])%sol_delta

        # make sure that sol_bins satisfies the range even with the fit
        sol_bins = np.append(sol_bins[0] - sol_delta, sol_bins)  # assume that it doesn't wrap around

    ## calculating datetime corresponding to sol_bins for each year
    bin_datetime_dict = []
    bin_datetime = []
    for sol in sol_bins:
        curr_time = all_time_intervals[start_idx][0] + datetime.timedelta(
            minutes=(sol - sol_bins[0])/(2*np.pi)*365.24219*24*60
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
                    (start_time - all_time_intervals[start_idx][0]).total_seconds()/(365.24219*24*60*60)
                )
            )
            bin_datetime = [
                jd2Date(solLon2jdSteyaert(dt.year + delta_years, dt.month, sol), dt_obj=True)
                for sol, dt in zip(sol_bins, bin_datetime_dict[0][1])
            ]
            bin_datetime_dict.append([(bin_datetime[0], bin_datetime[-1]), bin_datetime])

    return sol_bins, bin_datetime_dict



def generateFluxPlotName(station_code, shower_code, mass_index, sol_beg, sol_end):
    """ Generate a file name for the flux plot."""

    return "flux_{:s}_{:s}_s={:.2f}_sol={:.6f}-{:.6f}.png".format(station_code, shower_code, \
        mass_index, np.degrees(sol_beg), np.degrees(sol_end))


def generateFluxECSVName(station_code, shower_code, mass_index, sol_beg, sol_end):
    """ Generate a file name for the flux ECSV file. """

    return "flux_{:s}_{:s}_s={:.2f}_sol={:.6f}-{:.6f}.ecsv".format(station_code, shower_code, \
        mass_index, np.degrees(sol_beg), np.degrees(sol_end))



def generateFluxFixedBinsName(station_code, shower_code, mass_index, sol_beg, sol_end):
    """ Generate a file name for the fixed bins flux file. """

    return FIXED_BINS_NAME + "_{:s}_{:s}_s={:.2f}_sol={:.6f}-{:.6f}.ecsv".format(station_code, shower_code, \
        mass_index, np.degrees(sol_beg), np.degrees(sol_end))


def checkFluxFixedBinsName(file_name, shower_code, mass_index):

    shower_string = "_{:s}_s={:.2f}".format(shower_code, mass_index)

    if file_name.startswith(FIXED_BINS_NAME) and (shower_string in file_name) and file_name.endswith(".ecsv"):
        return True
    else:
        return False


# def saveForcedBinFluxData(dir_path, file_name, sol_list, meteor_n_list, area_list, time_list, meteor_lm_list):
#     """Save solar longitude and other parameters in a solar longitude range.

#     Arguments:
#         sol_list: [list] Solar longitude bin edges (length is one more than other lists) in radians.
#             It is assumed that this value does not wrap around.
#     """

#     # TO DO:
#     # Add to header:
#     #   shower
#     #   station code 
#     #   lat/lon/elev
#     #   used mass and population indices
#     #   star FWHM

#     file_path = os.path.join(dir_path, file_name)

#     with open(file_path, 'w') as f:

#         # TO DO:
#         # Add entires
#         #   Stellar LM
#         #   raw collection area
#         #   corrections before scaling with mass index
#         #   radiant elevation
#         #   radiant distance

#         f.write('# Solar longitude (deg), Meteors, Corrected collection area to +6.5M (km^2), Time (hours), Meteor LM (mag)\n')


#         for sol, meteors, area, time, lm in zip(sol_list, meteor_n_list, area_list, time_list, meteor_lm_list):

#             lm_str = "None"
#             if lm is not None:
#                 lm_str = "{:.3f}".format(lm)

#             f.write("{:.8f}, {:d}, {:.6f}, {:.6f}, {:s}\n".format(np.degrees(sol), meteors, area/1e6, time, \
#                 lm_str))


#         # sol_list has one more element than meteor_list
#         f.write("{:.8f},,,,".format(np.degrees(sol_list[-1])))



def loadFluxData(dir_path, file_name):
    """Load previously computed flux data.

    Return:
        [tuple] sol, meteor_list, area_list, time_list, meteor_lm_list
            - sol: [ndarray] Array of solar longitude bin edges (length is one more than other arrays)
            - meteor_list: [ndarray] Number of meteors in bin
            - area_list: [ndarray] Effective collecting area corresponding to bin
            - time_list: [ndarray] Duration of bin (in hours)
            - meteor_lm_list: [ndarray] Meteor limiting magnitude corresponding to bin
    """

    # Load previously computed collection areas and flux metadata
    flux_table = FluxMeasurements()
    flux_table.loadECSV(os.path.join(dir_path, file_name))

    ### Extract the data ###

    sol_data = flux_table.table['sol'].data.tolist()
    flux_lm_6_5_data = flux_table.table['flux_6_5_lm'].data.tolist()
    flux_lm_6_5_ci_lower_data = flux_table.table['flux_6_5_lm_ci_lower'].data.tolist()
    flux_lm_6_5_ci_upper_data = flux_table.table['flux_6_5_lm_ci_upper'].data.tolist()
    meteor_num_data = flux_table.table['meteors'].data.tolist()
    
    population_index = flux_table.table.meta['population_index']

    ### ###

    return sol_data, flux_lm_6_5_data, flux_lm_6_5_ci_lower_data, flux_lm_6_5_ci_upper_data, \
        meteor_num_data, population_index


def loadForcedBinFluxData(dir_path, file_name):
    """Load solar longitude, number of meteors, collecting area and time values for fixed bins

    Return:
        [tuple] sol, meteor_list, area_list, time_list, meteor_lm_list
            - sol: [ndarray] Array of solar longitude bin edges (length is one more than other arrays)
            - meteor_list: [ndarray] Number of meteors in bin
            - area_list: [ndarray] Effective collecting area corresponding to bin
            - time_list: [ndarray] Duration of bin (in hours)
            - meteor_lm_list: [ndarray] Meteor limiting magnitude corresponding to bin
    """

    # Load previously computed collection areas and flux metadata
    flux_table = FluxMeasurements()
    flux_table.loadECSV(os.path.join(dir_path, file_name))

    ### Extract the data ###

    # Add the ending bin to the solar longitdes, so they represent bin edges
    sol_bins = flux_table.table['sol'].data
    if len(sol_bins):
        sol_bins = np.radians(np.append(sol_bins, [flux_table.table.meta['sol_range'][1]]))

    meteor_list = flux_table.table['meteors'].data.astype(np.int).tolist()
    area_list = (1e6*flux_table.table['eff_col_area'].data).tolist()
    time_list = flux_table.table['time_bin'].data
    meteor_lm_list = flux_table.table['meteor_lm'].data.tolist()

    ### ###


    # file_path = os.path.join(dir_path, filename)

    # data = np.genfromtxt(file_path, delimiter=',', encoding=None, skip_header=1)

    # sol = np.radians(data[:, 0])
    # meteor_list = data[:-1, 1]
    # area_list = 1e6*data[:-1, 2]
    # time_list = data[:-1, 3]
    # meteor_lm_list = data[:-1, 4]

    return sol_bins, meteor_list, area_list, time_list, meteor_lm_list


def computeClearSkyTimeIntervals(cloud_ratio_dict, ratio_threshold=0.5, time_gap_threshold=15, clearing_threshold=90):
    """ Calculate sets of time intervals using the detected to predicted star ratios.

    Arguments:
        cloud_ratio_dict: [dict] ff_file: ratio
        ratio_threshold: [float] Minimum ratio required for the FF file to be considered having clear skies.
        time_gap_threshold: [float] Maximum time gap in seconds between ff files from cloud_ratio_dict
            before it's used to stop an interval. This is because if there is a gap that's too large, it
            can be because there are clouds in between. And to not risk this affecting flux values
            this is cut out.
        clearing_threshold: [float] Minimum time (in minutes) required for the ratio to be above the
            threshold before it will be considered.

    Return:
        intervals: [list of tuple of datetime] Structured like [(start_datetime, end_datetime), ...]

    """

    intervals = []
    start_interval = None
    prev_date = None


    for filename, ratio in cloud_ratio_dict.items():

        date = FFfile.filenameToDatetime(filename)

        if prev_date is None:
            prev_date = date

        if start_interval is None and ratio >= ratio_threshold:
            start_interval = date

        # Make an interval if FF has a >10 min gap (suggests clouds in between) or if the ratio is too low.
        # However the interval must be at least an hour to be kept.
        if (date - prev_date).total_seconds()/60 > time_gap_threshold or ratio < ratio_threshold:
            if start_interval is not None and (prev_date - start_interval).total_seconds()/60 > 60:
                intervals.append((start_interval, prev_date))

            # If ratio is less than threshold, you want to discard so it shouldn't be the start of an interval
            if ratio < ratio_threshold:
                start_interval = None
            else:
                start_interval = date

        prev_date = date

    # If you run out of images, that counts as a cutoff
    if start_interval is not None and (prev_date - start_interval).total_seconds()/60 > clearing_threshold:
        intervals.append((start_interval, prev_date))

    return intervals


def detectMoon(file_list, platepar, config):
    """ If moon is within 3 degrees of the FOV and the phase of the moon is above 25% then the moon
    is visible in view, and a picture with the moon in it will not be used

    Arguments:
        file_list: [list] List of FF files to detect the moon in
        platepar: [platepar object]
        config: [config object]

    Returns:
        new_file_list: [list] FF file list which don't have a moon in it
    """
    # setting up observer
    o = ephem.Observer()
    o.lat = str(config.latitude)
    o.long = str(config.longitude)
    o.elevation = config.elevation
    o.horizon = '0:0'

    radius = getFOVSelectionRadius(platepar)
    new_file_list = []

    # going through all ff files to check if moon is in fov
    for filename in file_list:
        # getting right ascension and declination of middle of fov
        _, ra_mid, dec_mid, _ = xyToRaDecPP(
            [FFfile.getMiddleTimeFF(filename, config.fps)],
            [platepar.X_res/2],
            [platepar.Y_res/2],
            [1],
            platepar,
        )
        ra_mid, dec_mid = np.radians(ra_mid[0]), np.radians(dec_mid[0])

        o.date = FFfile.filenameToDatetime(filename)
        m = ephem.Moon()
        m.compute(o)

        # Calculating fraction of moon which is visible
        nnm = ephem.next_new_moon(o.date)
        pnm = ephem.previous_new_moon(o.date)
        phase = (o.date - pnm)/(nnm - pnm)  # from 0 to 1 for 360 deg
        lunar_area = 1 - np.abs(2*phase - 1)  # using sawtooth function for fraction of moon visible

        # Calculating angular distance from middle of fov to correct for checking after the xy mapping
        angular_distance = np.degrees(angularSeparation(ra_mid, dec_mid, float(m.ra), float(m.dec)))

        # Always take observations if the Moon is at less than 25% illumination, regardless of where it is
        if lunar_area < 0.25:

            new_file_list.append(filename)
            continue

        # Always take observations if the Moon is not above the horizon
        elif o.next_rising(m) < o.next_setting(m):

            new_file_list.append(filename)
            continue

        # If it's brighter and up, only take observations when the Moon is outside the FOV
        elif angular_distance > radius:

            new_file_list.append(filename)
            continue

        # If it's witin the radius, check that it's not within the actual FOV
        else:

            # Compute X, Y coordinates of the Moon in the image
            x, y = raDecToXYPP(
                np.array([np.degrees(m.ra)]),
                np.array([np.degrees(m.dec)]),
                datetime2JD(o.date.datetime()),
                platepar,
            )

            x = x[0]
            y = y[0]

            # Compute the exclusion border in pixels (5 degrees)
            border = 5*platepar.F_scale

            if not (
                ((x > -border) and (x < platepar.X_res + border))
                and ((y > -border) and (y < platepar.Y_res + border))
            ):

                new_file_list.append(filename)
                continue

        print("Skipping {:s}, Moon in the FOV!".format(filename))

    return new_file_list


def detectClouds(config, dir_path, N=5, mask=None, show_plots=True, save_plots=False, ratio_threshold=0.5):
    """Detect clouds based on the number of stars detected in images compared to how many are
    predicted.

    Arguments:
        dir_path: [str] folder to search for FF files, CALSTARS files
        platepar: [Platepar object]

    keyword arguments:
        N: [float] Recalibrate on stars every N minutes to determine if it's cloudy or not.
        mask: [2d array]
        show_plots: [Bool] Whether to show plots (defaults to true)
        save_plots: [bool] Save the plots to disk. False by default
        ratio_threshold: [float] If the ratio of matched/predicted number of stars below this threshold,
            it is assumed that the sky is cloudy.

    Return:
        time_intervals [list of tuple]: list of datetime pairs in tuples, representing the starting
            and ending times of a time interval
    """


    if not show_plots:

        # Try loading already computed time intervals and skip computing them anew
        time_intervals = loadTimeInvervals(config, dir_path)

        if time_intervals is not None:
            print("Loaded already computed time intervals!")
            return time_intervals


    # Collect detected stars
    file_list = sorted(os.listdir(dir_path))

    # Get detected stars
    calstars_file = None
    for calstars_file in file_list:
        if ('CALSTARS' in calstars_file) and calstars_file.endswith('.txt'):
            break
    star_list = readCALSTARS(dir_path, calstars_file)
    print('CALSTARS file: ' + calstars_file + ' loaded!')


    # Get FF file every N minutes
    starting_time = None
    recorded_files = []
    bin_used = -1
    for ff_file_name, _ in star_list:
        date = FFfile.filenameToDatetime(ff_file_name)
        if starting_time is None:
            starting_time = date

        # store the first file of each bin
        new_bin = int(((date - starting_time).total_seconds()/60)//N)
        if new_bin > bin_used:
            recorded_files.append(ff_file_name)
            bin_used = new_bin


    # Get the platepar
    platepar = Platepar.Platepar()
    platepar.read(os.path.join(dir_path, config.platepar_name), use_flat=config.use_flat)


    # Locate and load the mask file
    mask = getMaskFile(dir_path, config, file_list=file_list)

    if mask is not None:
        mask.checkMask(platepar.X_res, platepar.Y_res)


    # Detect which images don't have a moon visible, and filter the file list based on this
    recorded_files = detectMoon(recorded_files, platepar, config)

    # Try loading previously recalibrated platepars on N minute intervals
    recalibrated_platepars = loadRecalibratedPlatepar(dir_path, config, file_list, type='flux')

    # If the file is not available, apply the recalibration procedure
    if not recalibrated_platepars:

        print("Recalibrated platepar file not available!")
        print("Recalibrating...")

        recalibrated_platepars = recalibrateSelectedFF(
            dir_path,
            recorded_files,
            star_list,
            config,
            stellarLMModel(platepar.mag_lev),
            ignore_distance_threshold=True,
        )
    recorded_files = list(recalibrated_platepars.keys())

    # Extract the number of matches stars between the catalog and the image
    matched_count = {ff: len(recalibrated_platepars[ff].star_list) \
        if recalibrated_platepars[ff].star_list is not None else 0 for ff in recorded_files}

    # Compute the correction between the visible limiting magnitude and the LM produced by the star detector
    #   - normalize the LM to intensity_threshold of 18
    #   - correct for the sensitivity at intensity threshold of 18 (empirical), which translates to -1.2 mag
    star_det_mag_corr = -2.5*np.log10(config.intensity_threshold/18) - 1.2

    # Compute the limiting magnitude of the star detector
    ff_limiting_magnitude = {
        ff_file: (
            stellarLMModel(recalibrated_platepars[ff_file].mag_lev) + star_det_mag_corr
            if recalibrated_platepars[ff_file].auto_recalibrated
            else None
        )
        for ff_file in recorded_files
    }


    # if show_plots:

    #     # Compute the limiting magnitude of matched stars as the 90th percentile of the faintest matched stars
    #     matched_star_LM = {
    #         ff: np.percentile(np.array(recalibrated_platepars[ff].star_list)[:, 6], 90)
    #         for ff in recorded_files
    #         if len(recalibrated_platepars[ff].star_list)
    #     }

    #     # Compute the limiting magnitude from the photometric offset using the empirical function
    #     empirical_LM = {
    #         ff_file: (
    #             stellarLMModel(recalibrated_platepars[ff_file].mag_lev)
    #             if recalibrated_platepars[ff_file].auto_recalibrated
    #             else None
    #         )
    #         for ff_file in recorded_files
    #     }

    #     plot_format = mdates.DateFormatter('%H:%M')

    #     plt.gca().xaxis.set_major_formatter(plot_format)

    #     # Plot the stellar LM
    #     plt.gca().scatter(
    #         [FFfile.filenameToDatetime(ff) for ff in empirical_LM],
    #         empirical_LM.values(),
    #         label='Stellar LM',
    #         s=5,
    #         c='k',
    #     )

    #     # Plot the theoretical LM of the star detector
    #     plt.gca().scatter(
    #         [FFfile.filenameToDatetime(ff) for ff in ff_limiting_magnitude],
    #         ff_limiting_magnitude.values(),
    #         marker='+',
    #         label='Star detection LM',
    #         c='orange',
    #     )

    #     # Plot the 90th percentile magnitude of matched stars
    #     plt.gca().scatter(
    #         [FFfile.filenameToDatetime(ff) for ff in matched_star_LM],
    #         matched_star_LM.values(),
    #         marker='x',
    #         label="90th percentile detected stars",
    #         c='green',
    #     )

    #     plt.gca().set_ylabel('Magnitude')
    #     plt.gca().set_xlabel('Time (UTC)')

    #     plt.gca().legend()

    #     plt.show()


    # Compute the predicted number of stars on every recalibrated FF file
    predicted_stars = predictStarNumberInFOV(
        recalibrated_platepars, ff_limiting_magnitude, config, mask, show_plot=show_plots
    )
    # for ff in predicted_stars:
    #     print(ff, matched_count.get(ff), predicted_stars.get(ff), ff_limiting_magnitude.get(ff))

    # Compute the ratio between matched and predicted stars
    ratio = {
        ff_file: (matched_count[ff_file]/predicted_stars[ff_file] if ff_file in predicted_stars else 0)
        for ff_file in recorded_files
    }

    # Find the time intervals of clear weather
    time_intervals = computeClearSkyTimeIntervals(ratio, ratio_threshold=ratio_threshold)

    # Save the computed time intervals so they don't have to be recomputed later on
    saveTimeInvervals(config, dir_path, time_intervals)


    if (show_plots or save_plots) and predicted_stars:

        # Compute the observing time in hours
        total_observing_time = 0
        for beg, end in time_intervals:
            total_observing_time += (end - beg).total_seconds()/3600

        fig, ax = plt.subplots(2, sharex=True)
        plot_format = mdates.DateFormatter('%H:%M')

        ax[0].set_title("Total observing time = {:.2f} h".format(total_observing_time))

        ax[0].xaxis.set_major_formatter(plot_format)

        # Plot the computed ratio
        ax[0].scatter([FFfile.filenameToDatetime(x) for x in ratio.keys()], list(ratio.values()), \
            marker='o', s=5, c='k', zorder=6, label='Measurements')
        ax[0].set_ylabel("Matched/Predicted stars")

        # Plot the radio threshold
        times = [FFfile.filenameToDatetime(x) for x in ratio.keys()]
        time_arr = [min(times), max(times)]
        ax[0].plot(time_arr, [ratio_threshold, ratio_threshold], linestyle='dashed', color='r', zorder=5, \
            alpha=0.5, label='Threshold')

        # Shade the regions with clear skies
        if len(time_intervals):

            for beg, end in time_intervals:
                ax[0].axvspan(beg, end, alpha=0.5, color='lightblue', zorder=4)

        ax[0].legend()
        ax[0].set_ylim(ymin=0)



        ax[1].xaxis.set_major_formatter(plot_format)

        # Plot the number of matched stars
        ax[1].scatter(
            [FFfile.filenameToDatetime(ff) for ff in matched_count],
            [matched_count[ff] for ff in matched_count],
            label='Matched stars', marker='+', color='r', zorder=5,
        )

        # Plot the number of predicted stars
        ax[1].scatter(
            [FFfile.filenameToDatetime(ff) for ff in predicted_stars],
            [predicted_stars[ff] for ff in predicted_stars],
            label='Predicted stars', marker='x', color='k', zorder=5,
        )
        

        # Shade the regions with clear skies
        if len(time_intervals):

            for beg, end in time_intervals:
                ax[1].axvspan(beg, end, alpha=0.5, color='lightblue', zorder=4)


        ax[1].set_xlabel("Time (UTC)")
        ax[1].set_ylabel("Count")
        ax[1].legend()

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.05)


        if save_plots:

            night_timestamp = "_".join(calstars_file.replace(".txt", "").split("_")[2:5])
            plot_name = "{:s}_{:s}_observing_periods.png".format(str(config.stationID), night_timestamp)

            plt.savefig(os.path.join(dir_path, plot_name), dpi=150)

        if show_plots:
            plt.show()

        else:
            plt.clf()
            plt.close()


    # calculating the ratio of observed starts to the number of predicted stars
    return time_intervals



def predictStarNumberInFOV(recalibrated_platepars, ff_limiting_magnitude, config, mask=None, show_plot=True):
    """ Predicts the number of stars that should be in the FOV, considering limiting magnitude,
        FOV and mask, and returns a dictionary mapping FF files to the number of predicted stars

    Arguments:
        recalibrated_platepars: [dict] FF_file: platepar
        ff_limiting_magnitude: [dict] FF_file: limiting_magnitude
        config: [Config object]

    Keyword Arguments:
        mask: [Mask object] Mask to filter stars to
        show_plot: [Bool] Whether to show plots (defaults to true)

    Return:
        pred_star_count: [dict] FF_file: number_of_stars_in_FOV
    """

    pred_star_count = {}

    ff_files = list(recalibrated_platepars.keys())

    if len(ff_files):

        # Using a blank mask if nothing is given
        if mask is None:
            mask = MaskStructure(None)
            mask.resetEmpty(recalibrated_platepars[ff_files[0]].X_res, \
                recalibrated_platepars[ff_files[0]].Y_res)


        # Go through all FF files and compute the number of predicted stars
        star_mag = {}
        for i, ff_file in enumerate(ff_files):

            platepar = recalibrated_platepars[ff_file]
            lim_mag = ff_limiting_magnitude[ff_file]

            if lim_mag is None:
                continue

            date = FFfile.getMiddleTimeFF(ff_file, config.fps, ret_milliseconds=True)
            jd = date2JD(*date)

            # # make a polygon on a sphere out of 5 points on each side
            # n_points = 5
            # y_vert = [platepar.Y_res*i/n_points for i in range(n_points)] \
            #             + [platepar.Y_res]*n_points \
            #             + [platepar.Y_res*(1 - i/n_points) for i in range(n_points)] + [0]*n_points
            # x_vert = [0]*n_points + [platepar.X_res*i/n_points for i in range(n_points)] \
            #             + [platepar.X_res]*n_points \
            #             + [platepar.X_res*(1-i/n_points) for i in range(n_points)]

            # Make a polygon on a the sky using the outline of the image, 5 points on each side
            x = platepar.X_res
            y = platepar.Y_res
            x_vert = [0, x/4, x/2, 3/4*x, x,   x,   x,     x, x, 3/4*x, x/2, x/4, 0,     0,   0,   0]
            y_vert = [0,   0,   0,     0, 0, y/4, y/2, 3/4*y, y,     y,   y,   y, y, 3/4*y, y/2, y/4]

            _, ra_vertices, dec_vertices, _ = xyToRaDecPP(
                [date]*len(x_vert),
                #[0, 0, platepar.X_res, platepar.X_res], # only 4 corners
                #[0, platepar.Y_res, platepar.Y_res, 0], # only 4 corners
                list(reversed(x_vert)),
                list(reversed(y_vert)),
                [1]*len(x_vert),
                platepar,
                extinction_correction=False,
            )

            # Collect and filter catalog stars
            catalog_stars, _, _ = StarCatalog.readStarCatalog(
                config.star_catalog_path,
                config.star_catalog_file,
                lim_mag=lim_mag,
                mag_band_ratios=config.star_catalog_band_ratios,
            )

            # Filter out stars that are outside of the polygon on the sphere made by the FOV
            ra_catalog, dec_catalog, mag = catalog_stars.T
            inside = pointInsideConvexPolygonSphere(
                np.array([ra_catalog, dec_catalog]).T, np.array([ra_vertices, dec_vertices]).T
            )

            # Compute catalog stars in X, Y coordinates and filter out in X, Y
            x, y = raDecToXYPP(ra_catalog[inside], dec_catalog[inside], jd, platepar)
            mag = mag[inside]


            # Compute star image levels from catalog magnitudes without any vignetting or extinction
            star_levels = 10**((mag - platepar.mag_lev)/(-2.5))

            # Compute star magnitudes using vignetting and extinction
            _, _, _, mag_corrected = xyToRaDecPP(len(x)*[date], x, y, star_levels, platepar, extinction_correction=True)
            mag_corrected = np.array(mag_corrected)

            # Filter coordinates to be in FOV and make sure that the stars that are too dim are filtered
            bounds = (mag_corrected <= lim_mag) & (y >= 0) & (y < platepar.Y_res) & (x >= 0) & (x < platepar.X_res)
            x = x[bounds]
            y = y[bounds]
            mag = mag[bounds]


            # Filter stars with mask
            mask_filter = np.take(
                np.floor(mask.img/255),
                np.ravel_multi_index(
                    np.floor(np.array([y, x])).astype(int), (platepar.Y_res, platepar.X_res)
                ),
            ).astype(bool)


            # # Plot one example of matched and predicted stars
            # if show_plot and (i == int(len(ff_files)//2)):

            #     plt.title("{:s}, LM = {:.2f}".format(ff_file, lim_mag))
            #     plt.scatter(
            #         *np.array(recalibrated_platepars[ff_file].star_list)[:, 1:3].T[::-1], label='matched'
            #     )
            #     plt.scatter(x[mask_filter], y[mask_filter], c='r', marker='+', label='catalog')
            #     plt.legend()
            #     plt.show()

            #     # print(np.sum(mask_filter))
            #     # print(val[inside][bounds][mask_filter])
            #     # plt.scatter(x[mask_filter], y[mask_filter], c=mag[inside & (mag <= lim_mag)][bounds][mask_filter])
            #     # plt.show()


            pred_star_count[ff_file] = np.sum(mask_filter)
            star_mag[ff_file] = mag


    return pred_star_count


def collectingArea(platepar, mask=None, side_points=20, ht_min=60, ht_max=130, dht=2, elev_limit=10):
    """Compute the collecting area for the range of given heights.

    Arguments:
        platepar: [Platepar object]

    Keyword arguments:
        mask: [Mask object] Mask object, None by default.
        side_points: [int] How many points to use to evaluate the FOV on seach side of the image. Normalized
            to the longest side.
        ht_min: [float] Minimum height (km).
        ht_max: [float] Maximum height (km).
        dht: [float] Height delta (km).
        elev_limit: [float] Limit of elevation above horizon (deg). 10 degrees by default.

    Return:
        col_areas_ht: [dict] A dictionary where the keys are heights of area evaluation, and values are
            segment dictionaries. Segment dictionaries have keys which are tuples of (x, y) coordinates of
            segment midpoints, and values are segment collection areas corrected for sensor effects.

    """

    # If the mask is not given, make a dummy mask with all white pixels
    if mask is None:
        mask = MaskStructure(None)
        mask.resetEmpty(platepar.Y_res, platepar.X_res)

    # Compute the number of samples for every image axis
    longer_side_points = side_points
    shorter_side_points = int(np.ceil(side_points*platepar.Y_res/platepar.X_res))

    # Compute pixel delta for every side
    longer_dpx = int(platepar.X_res//longer_side_points)
    shorter_dpx = int(platepar.Y_res//shorter_side_points)

    # Distionary of collection areas per height
    col_areas_ht = collections.OrderedDict()

    # Estimate the collection area for a given range of heights
    for ht in np.arange(ht_min, ht_max + dht, dht):

        # Convert the height to meters
        ht = 1000*ht

        total_area = 0

        # Dictionary of computed sensor-corrected collection areas where X and Y are keys
        col_areas_xy = collections.OrderedDict()

        # Sample the image
        for x0 in np.linspace(0, platepar.X_res, longer_side_points, dtype=np.int, endpoint=False):
            for y0 in np.linspace(0, platepar.Y_res, shorter_side_points, dtype=np.int, endpoint=False):

                # Compute lower right corners of the segment
                xe = x0 + longer_dpx
                ye = y0 + shorter_dpx

                # Compute geo coordinates of the image corners (if the corner is below the elevation limit,
                #   the *_elev value will be -1)
                _, ul_lat, ul_lon, ul_ht = xyHt2Geo(
                    platepar, x0, y0, ht, indicate_limit=True, elev_limit=elev_limit
                )
                _, ll_lat, ll_lon, ll_ht = xyHt2Geo(
                    platepar, x0, ye, ht, indicate_limit=True, elev_limit=elev_limit
                )
                _, lr_lat, lr_lon, lr_ht = xyHt2Geo(
                    platepar, xe, ye, ht, indicate_limit=True, elev_limit=elev_limit
                )
                _, ur_lat, ur_lon, ur_ht = xyHt2Geo(
                    platepar, xe, y0, ht, indicate_limit=True, elev_limit=elev_limit
                )

                # Skip the block if all corners are hitting the lower apparent elevation limit
                if np.all([ul_ht < 0, ll_ht < 0, lr_ht < 0, ur_ht < 0]):
                    continue

                # Make a polygon (clockwise direction)
                lats = [ul_lat, ll_lat, lr_lat, ur_lat]
                lons = [ul_lon, ll_lon, lr_lon, ur_lon]

                # Compute the area of the polygon
                area = areaGeoPolygon(lats, lons, ht)

                ### Apply sensitivity corrections to the area ###

                # Get the relevant mask segment
                mask_segment = mask.img[y0:ye, x0:xe]

                # If the mask segment is empty, skip this segment
                if mask_segment.size == 0:
                    continue

                # Compute ratio of masked portion of the segment
                unmasked_ratio = 1 - np.count_nonzero(~mask_segment)/mask_segment.size

                # Compute the pointing direction and the vignetting and extinction loss for the mean location

                x_mean = (x0 + xe)/2
                y_mean = (y0 + ye)/2

                # Use a test pixel sum
                test_px_sum = 400

                # Compute the pointing direction and magnitude corrected for vignetting and extinction
                _, ra, dec, mag = xyToRaDecPP(
                    [jd2Date(J2000_JD.days)], [x_mean], [y_mean], [test_px_sum], platepar
                )
                azim, elev = raDec2AltAz(ra[0], dec[0], J2000_JD.days, platepar.lat, platepar.lon)

                # Compute the pixel sum back assuming no corrections
                rev_level = 10**((mag[0] - platepar.mag_lev)/(-2.5))

                # Compute the sensitivty loss due to vignetting and extinction
                sensitivity_ratio = test_px_sum/rev_level

                # print(np.abs(np.hypot(x_mean - platepar.X_res/2, y_mean - platepar.Y_res/2)), sensitivity_ratio, mag[0])

                ##

                # Compute the range correction (w.r.t 100 km) to the mean point
                r, _, _, _ = xyHt2Geo(
                    platepar, x_mean, y_mean, ht, indicate_limit=True, elev_limit=elev_limit
                )

                # Correct the area for the masked portion
                area *= unmasked_ratio

                ### ###

                # Store the raw masked segment collection area, sensivitiy, and the range
                col_areas_xy[(x_mean, y_mean)] = [area, azim, elev, sensitivity_ratio, r]

                total_area += area

        # Store segments to the height dictionary (save a copy so it doesn't get overwritten)
        col_areas_ht[float(ht)] = dict(col_areas_xy)

        # print("SUM:", total_area/1e6, "km^2")

        # Compare to total area computed from the whole area
        side_points_list = fovArea(
            platepar, mask=mask, area_ht=ht, side_points=side_points, elev_limit=elev_limit
        )
        lats = []
        lons = []
        for side in side_points_list:
            for entry in side:
                lats.append(entry[0])
                lons.append(entry[1])

        # print("DIR:", areaGeoPolygon(lats, lons, ht)/1e6)

    return col_areas_ht


def sensorCharacterization(config, flux_config, dir_path, meteor_data, default_fwhm=None):
    """Characterize the standard deviation of the background and the FWHM of stars on every image."""

    exists_FF_files = any(FFfile.validFFName(filename) for filename in os.listdir(dir_path))

    # Find the CALSTARS file in the given folder that has FWHM information
    found_good_calstars = False
    for cal_file in os.listdir(dir_path):

        if ('CALSTARS' in cal_file) and ('.txt' in cal_file) and (not found_good_calstars):

            # Load the calstars file
            calstars_list = CALSTARS.readCALSTARS(dir_path, cal_file)

            # Check that at least one image has good FWHM measurements
            for ff_name, star_data in calstars_list:

                if len(star_data) > 0 and star_data[0][4] > -1:  # if stars were detected
                    star_data = np.array(star_data)

                    # Check if the calstars file have FWHM information
                    fwhm = star_data[:, 4]

                    # Check that FWHM values have been computed well
                    if np.all(fwhm > 1):
                        found_good_calstars = True
                        print('CALSTARS file: ' + cal_file + ' loaded!')
                        break

                elif not exists_FF_files and len(star_data) > 0 and star_data[0][4] == -1:
                    if default_fwhm is not None:
                        found_good_calstars = True
                        print('CALSTARS file: ' + cal_file + ' loaded!')
                        break

                    else:

                        # Use the default FWHM from the config file
                        fwhm = flux_config.default_fwhm

                        found_good_calstars = True

                        # raise Exception(
                        #     'CALSTARS file does not have fwhm and FF files do not exist in'
                        #     'directory. You must give a fwhm value with "--fwhm 3"'
                        # )

    # If the FWHM information is not present, run the star extraction
    if not found_good_calstars and exists_FF_files:
        print()
        print("No FWHM information found in existing CALSTARS files!")
        print()
        print("Rerunning star detection...")
        print()

        # Run star extraction again, and now FWHM will be computed
        calstars_list = extractStarsAndSave(config, dir_path)

        # Check for a minimum of detected stars
        for ff_name, star_data in calstars_list:
            if len(star_data) >= config.ff_min_stars:
                found_good_calstars = True
                break

    # If no good calstars exist, stop computing the flux
    if not found_good_calstars:

        print("No stars were detected in the data!")

        return False

    # Dictionary which holds information about FWHM and standard deviation of the image background
    sensor_data = {}
    meteor_ff = [data[0] for data in meteor_data]

    # Compute median FWHM per FF file
    for ff_name, star_data in calstars_list:

        # Check that the FF file exists in the data directory
        if ff_name not in meteor_ff:
            continue

        if star_data[0][4] == -1 and default_fwhm is not None and not exists_FF_files:  # data is old and fw
            fwhm_median = default_fwhm  # both these parameters are arbitrary
        else:
            star_data = np.array(star_data)

            # Compute the median star FWHM
            fwhm_median = np.median(star_data[:, 4])

        # Store the values to the dictionary
        sensor_data[ff_name] = [fwhm_median]
        print("{:s}, {:5.2f}".format(ff_name, fwhm_median))

    return sensor_data



def getCollectingArea(dir_path, config, flux_config, platepar, mask):
    """ Make a file name to save the raw collection areas. """

    col_areas_file_name = generateColAreaJSONFileName(config.stationID, flux_config.side_points, \
        flux_config.ht_min, flux_config.ht_max, flux_config.dht, flux_config.elev_limit,
    )

    # Check if the collection area file exists. If yes, load the data. If not, generate collection areas
    if col_areas_file_name in os.listdir(dir_path):
        col_areas_ht = loadRawCollectionAreas(dir_path, col_areas_file_name)
        print("Loaded collection areas from:", col_areas_file_name)

    else:

        # Compute the collecting areas segments per height
        col_areas_ht = collectingArea(platepar, mask=mask, side_points=flux_config.side_points, \
            ht_min=flux_config.ht_min, ht_max=flux_config.ht_max, dht=flux_config.dht, \
            elev_limit=flux_config.elev_limit)

        # Save the collection areas to file
        saveRawCollectionAreas(dir_path, col_areas_file_name, col_areas_ht)

        print("Saved raw collection areas to:", col_areas_file_name)

    ### ###

    # Compute the raw collection area at the height of 100 km
    col_area_100km_raw = 0
    col_areas_100km_blocks = col_areas_ht[100000.0]
    for block in col_areas_100km_blocks:
        col_area_100km_raw += col_areas_100km_blocks[block][0]

    print("Raw collection area at height of 100 km: {:.2f} km^2".format(col_area_100km_raw/1e6))

    return col_areas_ht, col_area_100km_raw


def getSensorCharacterization(dir_path, config, flux_config, meteor_data, default_fwhm=None):
    """ File which stores the sensor characterization profile. """

    sensor_characterization_file = "flux_sensor_characterization.json"
    sensor_characterization_path = os.path.join(dir_path, sensor_characterization_file)

    # Load sensor characterization file if present, so the procedure can be skipped
    if os.path.isfile(sensor_characterization_path):

        # Load the JSON file
        with open(sensor_characterization_path) as f:

            data = " ".join(f.readlines())
            sensor_data = json.loads(data)

            # Remove the info entry
            if '-1' in sensor_data:
                del sensor_data['-1']

    else:

        # Run sensor characterization
        sensor_data = sensorCharacterization(config, flux_config, dir_path, meteor_data, \
            default_fwhm=default_fwhm)

        # Save to file for posterior use
        with open(sensor_characterization_path, 'w') as f:

            # Add an explanation what each entry means
            sensor_data_save = dict(sensor_data)
            sensor_data_save['-1'] = {"FF file name": ['median star FWHM']}

            # Convert collection areas to JSON
            out_str = json.dumps(sensor_data_save, indent=4, sort_keys=True)

            # Save to disk
            f.write(out_str)

    return sensor_data


def stellarLMModel(p0):
    """Empirical model of the stellar limiting magnitude given the photometric zero point.
        Derived using various RMS cameras.

    Arguments:
        p0: [float] Photometric zeropoint.

    Return:
        lm_s: [float] Stellar limiting magnitude.
    """

    lm_s = 0.832*p0 - 2.585  # on Nov 4 2021, with 17 measured points

    return lm_s


def computeFluxCorrectionsOnBins(
    bin_meteor_information,
    bin_intervals,
    mass_index,
    population_index,
    shower,
    ht_std_percent,
    flux_config,
    config,
    col_areas_ht,
    v_init,
    azim_mid,
    elev_mid,
    r_mid,
    recalibrated_platepars,
    recalibrated_flux_platepars,
    platepar,
    frame_min_loss,
    ang_vel_night_mid,
    sensor_data,
    lm_m_nightly_mean,
    confidence_interval=0.95,
    binduration=None,
    verbose=True,
    fixed_bins=False,
):
    """

    Keyword arguments:
        verbose: [bool] Whether to print info as function is running
        fixed_bins: [bool] Compute fixed bins.

    """
    # Track values used for flux
    sol_data = []
    flux_lm_6_5_data = []
    flux_lm_6_5_ci_lower_data = []
    flux_lm_6_5_ci_upper_data = []
    meteor_num_data = []
    effective_collection_area_data = []
    radiant_elev_data = []
    radiant_dist_mid_data = []
    ang_vel_mid_data = []
    lm_s_data = []
    lm_m_data = []
    sensitivity_corr_data = []
    range_corr_data = []
    radiant_elev_corr_data = []
    ang_vel_corr_data = []
    total_corr_data = []
    mag_median_data = []
    mag_90_perc_data = []
    col_area_meteor_ht_raw = 0


    ### Init flux data container ###

    # Compute the mean meteor height
    meteor_ht_beg = heightModel(v_init, ht_type='beg')
    meteor_ht_end = heightModel(v_init, ht_type='end')
    meteor_ht = (meteor_ht_beg + meteor_ht_end)/2

    # Compute the mean FWHM
    all_ff_files = []
    for _, bin_ffs in bin_meteor_information:
        all_ff_files += bin_ffs

    fwhm_mean = np.mean([sensor_data[ff_name][0] for ff_name in all_ff_files])

    # Init the flux measurement table (sensitivity and the raw collection area are set to None, they have to
    #   the computed below)
    flux_table = FluxMeasurements()
    flux_table.initMetadata(shower.name, mass_index, population_index, flux_config.gamma, v_init/1000, 
        meteor_ht/1000, fwhm_mean, np.degrees(ang_vel_night_mid), None, r_mid/1000, None, confidence_interval)


    ### ###

    # Compute corrections and flux in all given bins
    for ((bin_meteor_list, bin_ffs), (bin_dt_beg, bin_dt_end)) in zip(bin_meteor_information, bin_intervals):


        bin_jd_beg = datetime2JD(bin_dt_beg)
        bin_jd_end = datetime2JD(bin_dt_end)

        jd_mean = (bin_jd_beg + bin_jd_end)/2

        # Compute the mean solar longitude
        sol_mean = np.degrees(jd2SolLonSteyaert(jd_mean))


        # If the fixed bins are used, saved the beginning of the bin as the relevant sol and time
        if fixed_bins:
            sol_entry = np.degrees(jd2SolLonSteyaert(datetime2JD(bin_dt_beg)))
            dt_entry = bin_dt_beg

        # Otherwise, use the middle time
        else:

            # Compute datetime of the mean
            dt_mean = jd2Date(jd_mean, dt_obj=True)

            sol_entry = sol_mean
            dt_entry = dt_mean


        ### Compute the radiant elevation at the middle of the time bin ###
        ra, dec, v_init = shower.computeApparentRadiant(platepar.lat, platepar.lon, jd_mean)
        radiant_azim, radiant_elev = raDec2AltAz(ra, dec, jd_mean, platepar.lat, platepar.lon)

        # Compute the mean meteor height
        meteor_ht_beg = heightModel(v_init, ht_type='beg')
        meteor_ht_end = heightModel(v_init, ht_type='end')
        meteor_ht = (meteor_ht_beg + meteor_ht_end)/2

        # Compute the standard deviation of the height
        meteor_ht_std = meteor_ht*ht_std_percent/100.0

        # Init the Gaussian height distribution
        meteor_ht_gauss = scipy.stats.norm(meteor_ht, meteor_ht_std)



        # Compute the total duration of the bin
        bin_hours = (bin_dt_end - bin_dt_beg).total_seconds()/3600


        # Compute the radiant distance from the middle of the FOV
        rad_dist_mid = angularSeparation(
            np.radians(radiant_azim), np.radians(radiant_elev), np.radians(azim_mid), np.radians(elev_mid)
        )

        # Compute the angular velocity in the middle of the FOV (rad/s)
        ang_vel_mid = v_init*np.sin(rad_dist_mid)/r_mid


        if verbose:
            print()
            print()
            print("-- Bin information ---")
            print("Bin beg:", bin_dt_beg)
            print("Bin end:", bin_dt_end)
            print("Sol mid: {:.5f}".format(sol_mean))
            print("Radiant elevation: {:.2f} deg".format(radiant_elev))
            print("Apparent speed: {:.2f} km/s".format(v_init/1000))

        if (not bin_ffs) and (not fixed_bins):
            if verbose:
                print("!!! Bin doesn't have any meteors!")
            continue

        # If the elevation of the radiant is below the limit, skip this bin
        if radiant_elev < flux_config.rad_elev_limit:

            if verbose:
                print(
                    "!!! Mean radiant elevation below {:.2f} deg threshold, skipping time bin!".format(
                        flux_config.rad_elev_limit
                    )
                )


            # Add zeros to the flux table
            if fixed_bins:
                meteor_num_data.append(0)
                lm_m_data.append(None)
                effective_collection_area_data.append(0)

                flux_table.addEntry(sol_entry, dt_entry, 0, radiant_elev, np.degrees(rad_dist_mid), \
                    np.degrees(ang_vel_mid), 0, 0, 0, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

            continue

        # The minimum duration of the time bin should be larger than 50% of the given dt
        if (binduration is not None) and (bin_hours < 0.5*binduration):

            if verbose:
                print(
                    "!!! Time bin duration of {:.2f} h is shorter than 0.5x of the inputted time bin!".format(bin_hours)
                )

            if fixed_bins:
                meteor_num_data.append(0)
                lm_m_data.append(None)
                effective_collection_area_data.append(0)

                flux_table.addEntry(sol_entry, dt_entry, 0, radiant_elev, np.degrees(rad_dist_mid), \
                    np.degrees(ang_vel_mid), 0, 0, 0, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

            continue

        # Continue running if there are enough meteors
        if fixed_bins or (len(bin_meteor_list) >= flux_config.meteors_min):
            
            if verbose:
                print("Meteors:", len(bin_meteor_list))

            ### Weight collection area by meteor height distribution ###

            # Determine weights for each height, but skip very small weights
            weight_sum = 0
            weights = {}
            for ht in col_areas_ht:
                wt = meteor_ht_gauss.pdf(float(ht))

                # Skip weights outside 3 sigma to speed up calculation
                if (ht < meteor_ht - 3*meteor_ht_std) or (ht > meteor_ht + 3*meteor_ht_std):
                    wt = 0

                weight_sum += wt
                weights[ht] = wt

            # Normalize the weights so that the sum is 1
            for ht in weights:
                weights[ht] /= weight_sum

            ### ###


            # Compute the raw collection area at the height meteors of of the given shower
            col_area_meteor_ht_raw = 0
            for ht in col_areas_ht:

                if weights[ht] == 0:
                    continue

                for block in col_areas_ht[ht]:
                    col_area_meteor_ht_raw += weights[ht]*col_areas_ht[ht][block][0]

            if verbose:
                print(
                    "Raw collection area at meteor heights: {:.2f} km^2".format(col_area_meteor_ht_raw/1e6)
                )


            # Skip time bin if the radiant is very close to the centre of the image
            if np.degrees(rad_dist_mid) < flux_config.rad_dist_min:
                if verbose:
                    print(
                        "!!! Radiant too close to the centre of the image! {:.2f} < {:.2f}".format(
                            np.degrees(rad_dist_mid), flux_config.rad_dist_min
                        )
                    )

                if fixed_bins:
                    meteor_num_data.append(0)
                    lm_m_data.append(None)
                    effective_collection_area_data.append(0)

                    flux_table.addEntry(sol_entry, dt_entry, 0, radiant_elev, np.degrees(rad_dist_mid), \
                    np.degrees(ang_vel_mid), 0, 0, 0, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)


                continue


            ### Compute the limiting magnitude ###

            # Compute the mean star FWHM in the given bin
            if bin_ffs:
                fwhm_bin_mean = np.mean([sensor_data[ff_name][0] for ff_name in bin_ffs])
                mag_lev_bin = np.mean(
                    [
                        recalibrated_platepars[ff_name].mag_lev
                        for ff_name in bin_ffs
                        if ff_name in recalibrated_platepars
                    ]
                )
            else:
                fwhm_bin_mean = min(
                    sensor_data.items(),
                    key=lambda x: np.abs(datetime2JD(FFfile.filenameToDatetime(x[0])) - jd_mean),
                )[1][0]
                # Compute the mean photometric zero point in the given bin
                mag_lev_bin = min(
                    recalibrated_flux_platepars.items(),
                    key=lambda x: np.abs(datetime2JD(FFfile.filenameToDatetime(x[0])) - jd_mean),
                )[1].mag_lev


            # Use empirical LM calculation
            lm_s = stellarLMModel(mag_lev_bin)

            # Add a loss due to minimum number of frames used
            lm_s += frame_min_loss

            # Compute apparent meteor magnitude
            lm_m = (
                lm_s - 5*np.log10(r_mid/1e5) - 2.5*np.log10(
                    np.degrees(
                        platepar.F_scale*v_init*np.sin(rad_dist_mid)
                            /(config.fps*r_mid*fwhm_bin_mean)
                    )))

            ### ###

            # Final correction area value (height-weightned)
            collection_area = 0

            # Keep track of the corrections
            sensitivity_corr_arr = []
            range_corr_arr = []
            radiant_elev_corr_arr = []
            ang_vel_corr_arr = []
            total_corr_arr = []
            col_area_raw_arr = []
            col_area_eff_arr = []
            col_area_eff_block_dict = {}

            # Go through all heights and segment blocks
            for ht in col_areas_ht:

                # Skip heights for which the weights are zero
                if weights[ht] == 0:
                    continue

                for img_coords in col_areas_ht[ht]:
                    # Unpack precomputed values
                    area, azim, elev, sensitivity_ratio, r = col_areas_ht[ht][img_coords]

                    # Compute the angular velocity (rad/s) in the middle of this block
                    rad_dist = angularSeparation(
                        np.radians(radiant_azim), np.radians(radiant_elev), np.radians(azim), np.radians(elev)
                    )
                    ang_vel = v_init*np.sin(rad_dist)/r

                    # If the angular distance from the radiant is less than 15 deg, don't use the block
                    #   in the effective collection area
                    if np.degrees(rad_dist) < flux_config.rad_dist_min:
                        area = 0.0

                    # Compute the range correction
                    range_correction = (1e5/r)**2

                    # ang_vel_correction = ang_vel/ang_vel_mid
                    # Compute angular velocity correction relative to the nightly mean
                    ang_vel_correction = ang_vel/ang_vel_night_mid

                    # Apply corrections

                    correction_ratio = 1.0

                    # Correct the area for vignetting and extinction
                    sensitivity_corr_arr.append(sensitivity_ratio)
                    correction_ratio *= sensitivity_ratio

                    # Correct for the range (cap to an order of magnitude correction)
                    range_correction = max(range_correction, 0.1)
                    range_corr_arr.append(range_correction)
                    correction_ratio *= range_correction

                    # Correct for the radiant elevation (cap to an order of magnitude correction)
                    #   Apply the zenith exponent gamma
                    radiant_elev_correction = np.sin(np.radians(radiant_elev))**flux_config.gamma
                    radiant_elev_correction = max(radiant_elev_correction, 0.1)
                    radiant_elev_corr_arr.append(radiant_elev_correction)
                    correction_ratio *= radiant_elev_correction

                    # Correct for angular velocity (cap to an order of magnitude correction)
                    ang_vel_correction = min(max(ang_vel_correction, 0.1), 10)
                    correction_ratio *= ang_vel_correction
                    ang_vel_corr_arr.append(ang_vel_correction)

                    # Add the collection area to the final estimate with the height weight
                    #   Raise the correction to the mass index power
                    total_correction = correction_ratio**(mass_index - 1)
                    collection_area += weights[ht]*area*total_correction
                    total_corr_arr.append(total_correction)

                    col_area_raw_arr.append(weights[ht]*area)
                    col_area_eff_arr.append(weights[ht]*area*total_correction)

                    if img_coords not in col_area_eff_block_dict:
                        col_area_eff_block_dict[img_coords] = []

                    col_area_eff_block_dict[img_coords].append(weights[ht]*area*total_correction)

            # Compute mean corrections
            sensitivity_corr_avg = np.mean(sensitivity_corr_arr)
            range_corr_avg = np.mean(range_corr_arr)
            radiant_elev_corr_avg = np.mean(radiant_elev_corr_arr)
            ang_vel_corr_avg = np.mean(ang_vel_corr_arr)
            total_corr_avg = np.median(total_corr_arr)
            col_area_raw_sum = np.sum(col_area_raw_arr)
            col_area_eff_sum = np.sum(col_area_eff_arr)

            if verbose:
                print(
                    "Raw collection area at meteor heights (CHECK): {:.2f} km^2".format(
                        col_area_raw_sum/1e6
                    )
                )
                print(
                    "Eff collection area at meteor heights (CHECK): {:.2f} km^2".format(
                        col_area_eff_sum/1e6
                    )
                )

            # ### PLOT HOW THE CORRECTION VARIES ACROSS THE FOV
            # x_arr = []
            # y_arr = []
            # col_area_eff_block_arr = []

            # for img_coords in col_area_eff_block_dict:

            #     x_mean, y_mean = img_coords

            #     #if x_mean not in x_arr:
            #     x_arr.append(x_mean)
            #     #if y_mean not in y_arr:
            #     y_arr.append(y_mean)

            #     col_area_eff_block_arr.append(np.sum(col_area_eff_block_dict[img_coords]))

            # x_unique = np.unique(x_arr)
            # y_unique = np.unique(y_arr)
            # # plt.pcolormesh(x_arr, y_arr, np.array(col_area_eff_block_arr).reshape(len(x_unique), len(y_unique)).T, shading='auto')
            # plt.title("TOTAL = " + str(np.sum(col_area_eff_block_arr)/1e6))
            # plt.scatter(x_arr, y_arr, c=np.array(col_area_eff_block_arr)/1e6)
            # #plt.pcolor(np.array(x_arr).reshape(len(x_unique), len(y_unique)), np.array(y_arr).reshape(len(x_unique), len(y_unique)), np.array(col_area_eff_block_arr).reshape(len(x_unique), len(y_unique))/1e6)
            # plt.colorbar(label="km^2")
            # plt.gca().invert_yaxis()
            # plt.show()

            # ###

            # Compute the nominal flux at the bin LM (meteors/1000km^2/h)
            collection_area_lm_nightly = collection_area/population_index**(lm_m_nightly_mean - lm_m)
            collection_area_lm_6_5 = collection_area/population_index**(6.5 - lm_m)

            flux = 1e9*len(bin_meteor_list)/collection_area/bin_hours
            flux_lm_nightly_mean = 1e9*len(bin_meteor_list)/collection_area_lm_nightly/bin_hours
            flux_lm_6_5 = 1e9*len(bin_meteor_list)/collection_area_lm_6_5/bin_hours

            # Compute confidence interval of the flux
            ci = 1.0 - confidence_interval
            num_ci_lower = scipy.stats.chi2.ppf(ci/2, 2*len(bin_meteor_list))/2
            num_ci_upper = scipy.stats.chi2.ppf(1 - ci/2, 2*(len(bin_meteor_list) + 1))/2
            flux_lm_6_5_ci_lower = 1e9*num_ci_lower/collection_area_lm_6_5/bin_hours
            flux_lm_6_5_ci_upper = 1e9*num_ci_upper/collection_area_lm_6_5/bin_hours

            mag_bin_median = (
                np.median([np.min(meteor.mag_array) for meteor in bin_meteor_list])
                if bin_meteor_list
                else None
            )
            mag_median_data.append(mag_bin_median)
            mag_90_perc = (
                np.percentile([np.min(meteor.mag_array) for meteor in bin_meteor_list], 90)
                if bin_meteor_list
                else None
            )
            mag_90_perc_data.append(mag_90_perc)

            if verbose:
                print("-- Sensor information ---")
                print("Star FWHM:  {:5.2f} px".format(fwhm_bin_mean))
                print("Photom ZP:  {:+6.2f} mag".format(mag_lev_bin))
                print("Stellar LM: {:+.2f} mag".format(lm_s))
                print("-- Flux ---")
                print(
                    "Meteors:  {:d}, {:.0f}% CI [{:.2f}, {:.2f}]".format(
                        len(bin_meteor_list), 100*confidence_interval, num_ci_lower, num_ci_upper
                    )
                )
                print("Col area: {:d} km^2".format(int(collection_area/1e6)))
                print("Ang vel:  {:.2f} deg/s".format(np.degrees(ang_vel_mid)))
                print("LM app:   {:+.2f} mag".format(lm_m))
                print("Flux:     {:.2f} meteors/1000km^2/h".format(flux))
                print("to {:+.2f}: {:.2f} meteors/1000km^2/h".format(lm_m_nightly_mean, flux_lm_nightly_mean))
                print(
                    "to +6.50: {:.2f}, {:.0f}% CI [{:.2f}, {:.2f}] meteors/1000km^2/h".format(
                        flux_lm_6_5, 100*confidence_interval, flux_lm_6_5_ci_lower, flux_lm_6_5_ci_upper
                    )
                )

            sol_data.append(sol_mean)
            flux_lm_6_5_data.append(flux_lm_6_5)
            flux_lm_6_5_ci_lower_data.append(flux_lm_6_5_ci_lower)
            flux_lm_6_5_ci_upper_data.append(flux_lm_6_5_ci_upper)
            meteor_num_data.append(len(bin_meteor_list))
            effective_collection_area_data.append(collection_area)
            radiant_elev_data.append(radiant_elev)
            radiant_dist_mid_data.append(np.degrees(rad_dist_mid))
            ang_vel_mid_data.append(np.degrees(ang_vel_mid))
            lm_s_data.append(lm_s)
            lm_m_data.append(lm_m)

            sensitivity_corr_data.append(sensitivity_corr_avg)
            range_corr_data.append(range_corr_avg)
            radiant_elev_corr_data.append(radiant_elev_corr_avg)
            ang_vel_corr_data.append(ang_vel_corr_avg)
            total_corr_data.append(total_corr_avg)

            # Compute the collection area scaled to meteor LM of +6.5M
            collection_area_6_5_lm = collection_area/population_index**(6.5 - lm_m)

            # Add entry to the flux data container
            flux_table.addEntry(sol_entry, dt_entry, len(bin_meteor_list), radiant_elev, \
                np.degrees(rad_dist_mid), np.degrees(ang_vel_mid), total_corr_avg, \
                collection_area/1e6, collection_area_6_5_lm/1e6, bin_hours, lm_s, lm_m, flux, flux_lm_6_5, 
                flux_lm_6_5_ci_lower, flux_lm_6_5_ci_upper)


        elif verbose:
            print(
                '!!! Insufficient meteors in bin: {:d} observed vs min {:d}'.format(len(bin_meteor_list), flux_config.meteors_min)
            )


    # Add sensitivity and raw collection area to the flux table
    flux_table.mean_sensitivity = np.mean(sensitivity_corr_data)
    flux_table.raw_col_area = col_area_meteor_ht_raw/1e6

    return (
        flux_table,
        sol_data,
        flux_lm_6_5_data,
        flux_lm_6_5_ci_lower_data,
        flux_lm_6_5_ci_upper_data,
        meteor_num_data,
        effective_collection_area_data,
        radiant_elev_data,
        radiant_dist_mid_data,
        ang_vel_mid_data,
        lm_s_data,
        lm_m_data,
        sensitivity_corr_data,
        range_corr_data,
        radiant_elev_corr_data,
        ang_vel_corr_data,
        total_corr_data,
        col_area_meteor_ht_raw,
        mag_median_data,
        mag_90_perc_data,
    )


def computeFlux(config, dir_path, ftpdetectinfo_path, shower_code, dt_beg, dt_end, mass_index, \
    binduration=None, binmeteors=None, timebin_intdt=0.25, ht_std_percent=5.0, mask=None, show_plots=True, \
    save_plots=False, confidence_interval=0.95, default_fwhm=None, forced_bins=None, compute_single=True):
    """Compute flux using measurements in the given FTPdetectinfo file.

    Arguments:
        config: [Config instance]
        dir_path: [str] Path to the working directory.
        ftpdetectinfo_path: [str] Path to a FTPdetectinfo file.
        shower_code: [str or Shower object] IAU shower code (e.g. ETA, PER, SDA), or an instace of the Shower
            class.
        dt_beg: [Datetime] Datetime object of the observation beginning.
        dt_end: [Datetime] Datetime object of the observation end.
        mass_index: [float] Cumulative mass index of the shower.

    Keyword arguments:
        binduration: [float] Time in hours for each bin
        binmeteors: [int] Number of meteors to have in each bin (cannot have both this and binduration).
            If set to -1, determine the number of bins by using the Rice rule.
        timebin_intdt: [float] Time step for computing the integrated collection area in hours. 15 minutes by
            default. If smaller than that, only one collection are will be computed.
        ht_std_percent: [float] Meteor height standard deviation in percent.
        mask: [Mask object] Mask object, None by default.
        show_plots: [bool] Show flux plots. True by default.
        save_plots: [bool] Save the plots to disk. False by default.
        confidence_interval: [float] Confidence interval for error estimation using Poisson statistics.
            0.95 by default (95% CI).
        default_fwhm: [float] If calstars file doesn't contain fwhm data, a value can be inputted for
            what they can be defaulted to (actual fwhm values are always prioritized)
        forced_bins: [tuple] sol_bins, binduration
            - bin_time: [list] Datetime objects corresponding to each bin.
            - sol_bins: [list] List of sol corresponding the bins. These values must NOT wrap around
            If this is given, meteor detections will be put in bins given by the sol_bins, where each bin
            is 5ish minutes long. sol_bins also shouldn't wrap around once it hits 360.
            The extra parameter, bin_information, will be returned when this parameter is given. These bins
            are independent of the binduration and binmeteors parameters
        compute_single: [bool] Only considered if forced bins are given. If False, single station flux will
            not be computed. True by default.

    Return:
        [tuple] sol_data, flux_lm_6_5_data, flux_lm_6_5_ci_lower_data, flux_lm_6_5_ci_upper_data, bin_information
            - sol_data: [list] Array of solar longitudes (in degrees) of time bins.
            - flux_lm6_5_data: [list] Array of meteoroid flux at the limiting magnitude of +6.5 in
                meteors/1000km^2/h.
            - flux_lm_6_5_ci_lower_data: [list] Flux, lower bound confidence interval.
            - flux_lm_6_5_ci_upper_data: [list] Flux, upper bound confidence interval.
            - meteor_num_data: [list] Number of meteors in every bin.
            - bin_information: [tuple] (only if forced_bins is given) meteors, collecting_area, collection_time
                - meteors: [list] Number of meteors in bin (given by forced_bins)
                - collecting_area: [list] Effective collecting area for a bin
                - collection_time: [list] Time which meteors can be present (only changes when the
                    bin encapsulates the beginning or end time) in hours

    """

    # Init the flux configuration
    flux_config = FluxConfig()


    # Extract the shower code if the Shower object was given
    shower = shower_code
    if not isinstance(shower_code, str):
        shower_code = shower_code.name

    else:
        # Otherwise, get the shower object
        shower_list = loadRadiantShowers(config)
        shower = [sh for sh in shower_list if sh.name == shower_code][0]


    ### Generate 5 minute bins ###
    loaded_forced_bins = False
    if forced_bins:

        bin_datetime_all, sol_bins_all = forced_bins
        sol_bins_all = np.array(sol_bins_all)
        starting_sol = unwrapSol(jd2SolLonSteyaert(datetime2JD(dt_beg)), sol_bins_all[0], sol_bins_all[-1])
        ending_sol = unwrapSol(jd2SolLonSteyaert(datetime2JD(dt_end)), sol_bins_all[0], sol_bins_all[-1])

        # Make a name for the forced bins file
        forced_bins_file = generateFluxFixedBinsName(config.stationID, shower_code, mass_index, \
            starting_sol, ending_sol)

        print("Forced bins file:", forced_bins_file)

        # Load previous computed bins, if available
        if os.path.exists(os.path.join(dir_path, forced_bins_file)):

            # Load previously computed collection areas and flux metadata
            sol_bins, forced_bins_meteor_num, forced_bins_area, forced_bins_time, \
                forced_bins_lm_m = loadForcedBinFluxData(dir_path, forced_bins_file)

            print("    ... loaded!")

            # Skips area calculation
            loaded_forced_bins = True  

        # Compute bins
        else:

            # Filtering sol bins so that bin edges contain just starting_sol and ending_sol
            bin_filter_min = np.searchsorted(sol_bins_all, starting_sol, side='right') - 1 
            bin_filter_max = np.searchsorted(sol_bins_all, ending_sol, side='left') + 1
            sol_bins = sol_bins_all[bin_filter_min:bin_filter_max]

            # Also filter the datetime bins
            dt_bins = bin_datetime_all[bin_filter_min:bin_filter_max]


            # Time calculated as fraction of bin time that the starting_sol end ending_sol is
            forced_bins_time = np.array(
                [
                    min(
                        max(
                            min(sol_bins[i + 1] - starting_sol, ending_sol - sol_bins[i])
                           /(sol_bins[i + 1] - sol_bins[i]),
                            0,
                        ),
                        1,
                    )
                   *(dt_bins[i + 1] - dt_bins[i]).total_seconds()/3600
                    for i in range(len(sol_bins[:-1]))
                ]
            )

            
            forced_bin_meteor_information = [[[], []] for _ in sol_bins[:-1]]

            # Compute datetime intervals of forced bins
            forced_bin_intervals = [
                [dt_bins[i], dt_bins[i + 1]] for i in range(len(dt_bins) - 1)
            ]

            # Change the begin and end times of the fixed intervals to correspond to the orbserving period 
            #   range
            forced_bin_intervals[0][0] = dt_beg
            forced_bin_intervals[-1][-1] = dt_end



    # Make the name for the flux file
    sol_beg = jd2SolLonSteyaert(datetime2JD(dt_beg))
    sol_end = jd2SolLonSteyaert(datetime2JD(dt_end))
    ecsv_file_name = generateFluxECSVName(config.stationID, shower_code, mass_index, sol_beg, sol_end)

    print("Flux ECSV file:", ecsv_file_name)

    # If the flux file was already computed and the plots won't be shown, load the flux file from disk
    loaded_flux_computations = False
    sol_data = []
    flux_lm_6_5_data = []
    flux_lm_6_5_ci_lower_data = []
    flux_lm_6_5_ci_upper_data = []
    meteor_num_data = []
    population_index = calculatePopulationIndex(mass_index)
    if os.path.isfile(os.path.join(dir_path, ecsv_file_name)) and not (show_plots or save_plots):
        
        sol_data, flux_lm_6_5_data, flux_lm_6_5_ci_lower_data, flux_lm_6_5_ci_upper_data, meteor_num_data, \
            population_index = loadFluxData(dir_path, ecsv_file_name)

        print("   ... loaded!")

        loaded_flux_computations = True



    # Compute the flux
    if (compute_single and not loaded_flux_computations) \
        or (forced_bins and loaded_flux_computations and not loaded_forced_bins):

        # Get a list of files in the night folder
        file_list = sorted(os.listdir(dir_path))

        # Load meteor data from the FTPdetectinfo file
        meteor_data = readFTPdetectinfo(*os.path.split(ftpdetectinfo_path))

        if not len(meteor_data):
            print("No meteors in the FTPdetectinfo file!")
            return None

        platepar = Platepar.Platepar()
        platepar.read(os.path.join(dir_path, config.platepar_name), use_flat=config.use_flat)

        recalibrated_platepars = loadRecalibratedPlatepar(dir_path, config, file_list, type='meteor')
        recalibrated_flux_platepars = loadRecalibratedPlatepar(dir_path, config, file_list, type='flux')

        if not recalibrated_platepars:
            print("Recalibrated platepar file not available!")
            print("Recalibrating...")
            recalibrated_platepars = applyRecalibrate(ftpdetectinfo_path, config)

        # Compute nighly mean of the photometric zero point
        mag_lev_nightly_mean = np.mean(
            [recalibrated_platepars[ff_name].mag_lev for ff_name in recalibrated_platepars]
        )

        # Locate and load the mask file
        mask = getMaskFile(dir_path, config, file_list=file_list)

        # If the resolution of the loaded mask doesn't match the resolution in the platepar, reset the mask
        if mask is not None:
            mask.checkMask(platepar.X_res, platepar.Y_res)

        # Compute the population index using the classical equation
        # Found to be more consistent when comparing fluxes
        population_index = calculatePopulationIndex(mass_index)

        ### SENSOR CHARACTERIZATION ###
        # Computes FWHM of stars and noise profile of the sensor
        sensor_data = getSensorCharacterization(dir_path, config, flux_config, meteor_data, \
            default_fwhm=default_fwhm)

        # Compute the nighly mean FWHM
        fwhm_nightly_mean = np.mean([sensor_data[key][0] for key in sensor_data])

        ### ###

        # Perform shower association
        associations, _ = showerAssociation(config, [ftpdetectinfo_path], shower_code=shower_code, \
            show_plot=False, save_plot=False, plot_activity=False)


        # Remove all meteors which begin below the limit height
        filtered_associations = {}
        for key in associations:
            meteor, shower = associations[key]

            if meteor.beg_alt > flux_config.elev_limit:
                # print("Rejecting:", meteor.jdt_ref)
                filtered_associations[key] = (meteor, shower)

        associations = filtered_associations




        ### Go through all time bins within the observation period ###

        bin_meteor_information = []  # [[[meteor, ...], [meteor.ff_name, ...]], ...]
        bin_intervals = []  # [(start_time, end_time), ...]
        num_meteors = 0


        # Automatically deterine the number of meteors in the bin, if it's not given
        if binmeteors < 0:

            if len(associations) > 0:

                binmeteors = len(associations)/np.ceil(np.sqrt(len(associations)))

            # Use a minimum of 5 meteors per bin
            if binmeteors < 5:
                binmeteors = 5


        # If using fixed bins in time, generate them
        if binduration is not None:

            curr_bin_start = dt_beg
            dt = datetime.timedelta(hours=(binduration/flux_config.sub_time_bins))

            while curr_bin_start < dt_end:

                bin_intervals.append(
                    (curr_bin_start, min(curr_bin_start + flux_config.sub_time_bins*dt, dt_end))
                )

                bin_meteor_information.append([[], []])

                curr_bin_start += dt



        ### Sort meteors into bins ###

        for meteor, shower in associations.values():
            meteor_date = jd2Date(meteor.jdt_ref, dt_obj=True)

            # Filter out meteors ending too close to the radiant
            ra, dec, _ = shower.computeApparentRadiant(platepar.lat, platepar.lon, meteor.jdt_ref)
            radiant_azim, radiant_elev = raDec2AltAz(ra, dec, meteor.jdt_ref, platepar.lat, platepar.lon)

            # Skip meteors don't belonging to the shower, outside the bin, and too close to the radiant
            if (
                shower is None
                or (shower.name != shower_code)
                or (meteor_date < dt_beg)
                or (meteor_date > dt_end)
                or np.degrees(
                    angularSeparation(
                        np.radians(radiant_azim),
                        np.radians(radiant_elev),
                        np.radians(meteor.end_azim),
                        np.radians(meteor.end_alt),
                    )
                )
                < flux_config.rad_dist_min
            ):
                continue

            num_meteors += 1

            if forced_bins and not loaded_forced_bins:

                sol = unwrapSol(jd2SolLonSteyaert(meteor.jdt_ref), sol_bins[0], sol_bins[-1])
                indx = min(np.searchsorted(sol_bins, sol, side='right') - 1, len(sol_bins) - 2)
                forced_bin_meteor_information[indx][0].append(meteor)
                forced_bin_meteor_information[indx][1].append(meteor.ff_name)

            
            # Sort meteors into bins by bin duration
            if binduration is not None:

                bin_num = min(
                    int(
                        ((meteor_date - dt_beg).total_seconds()/3600)/binduration*flux_config.sub_time_bins
                    ),
                    len(bin_intervals) - 1,
                )
                for i in range(min(flux_config.sub_time_bins, bin_num)):
                    bin_meteor_information[bin_num - i][0].append(meteor)
                    bin_meteor_information[bin_num - i][1].append(meteor.ff_name)

            # Sort meteors into bins by meteor count
            else: 
                if ((num_meteors - 1)*flux_config.sub_time_bins)%binmeteors < flux_config.sub_time_bins:
                    for i in range(min(flux_config.sub_time_bins - 1, len(bin_meteor_information))):
                        bin_meteor_information[-i - 1][0].append(meteor)
                        bin_meteor_information[-i - 1][1].append(meteor.ff_name)

                    bin_meteor_information.append([[meteor], [meteor.ff_name]])

                    if len(bin_intervals) < flux_config.sub_time_bins:
                        bin_intervals.append([dt_beg])
                    else:
                        bin_intervals[-flux_config.sub_time_bins].append(meteor_date)
                        bin_intervals.append([meteor_date])

                else:
                    for i in range(min(flux_config.sub_time_bins, len(bin_meteor_information))):
                        bin_meteor_information[-i - 1][0].append(meteor)
                        bin_meteor_information[-i - 1][1].append(meteor.ff_name)


        # Closing off all bin intervals
        if binmeteors is not None:
            for _bin in bin_intervals[-flux_config.sub_time_bins :]:
                if len(_bin) == 1:
                    _bin.append(dt_end)

        ### ###


        

        # Print the list of used meteors
        peak_mags = []
        for meteor, shower in associations.values():
            if shower is not None:
                # Compute peak magnitude
                peak_mag = np.min(meteor.mag_array)
                peak_mags.append(peak_mag)
                print("{:.6f}, {:3s}, {:+.2f}".format(meteor.jdt_ref, shower.name, peak_mag))

        print()


        ### COMPUTE COLLECTION AREAS ###

        col_areas_ht, col_area_100km_raw = getCollectingArea(dir_path, config, flux_config, platepar, mask)

        # Compute the pointing of the middle of the FOV
        _, ra_mid, dec_mid, _ = xyToRaDecPP(
            [jd2Date(J2000_JD.days)],
            [platepar.X_res/2],
            [platepar.Y_res/2],
            [1],
            platepar,
            extinction_correction=False,
        )
        azim_mid, elev_mid = raDec2AltAz(ra_mid[0], dec_mid[0], J2000_JD.days, platepar.lat, platepar.lon)


        # Compute the range to the middle point
        ref_ht = 100000
        r_mid, _, _, _ = xyHt2Geo(
            platepar,
            platepar.X_res/2,
            platepar.Y_res/2,
            ref_ht,
            indicate_limit=True,
            elev_limit=flux_config.elev_limit,
        )

        print("Range at 100 km in the middle of the image: {:.2f} km".format(r_mid/1000))



        # Compute the average angular velocity to which the flux variation throught the night will be normalized
        #   The ang vel is of the middle of the FOV in the middle of observations

        # Middle Julian date of the night
        jd_night_mid = (datetime2JD(dt_beg) + datetime2JD(dt_end))/2

        # Compute the apparent radiant
        ra, dec, v_init = shower.computeApparentRadiant(platepar.lat, platepar.lon, jd_night_mid)

        # Compute the radiant elevation
        radiant_azim, radiant_elev = raDec2AltAz(ra, dec, jd_night_mid, platepar.lat, platepar.lon)

        # Compute the angular velocity in the middle of the FOV
        rad_dist_night_mid = angularSeparation(
            np.radians(radiant_azim), np.radians(radiant_elev), np.radians(azim_mid), np.radians(elev_mid)
        )
        ang_vel_night_mid = v_init*np.sin(rad_dist_night_mid)/r_mid

        ###



        # Compute the average limiting magnitude to which all flux will be normalized

        # Compute the theoretical stellar limiting magnitude using an empirical model (nightly average)
        lm_s_nightly_mean = stellarLMModel(mag_lev_nightly_mean)

        # A meteor needs to be visible on at least 4 frames, thus it needs to have at least 4x the mass to produce
        #   that amount of light. 1 magnitude difference scales as -0.4 of log of mass, thus:
        # frame_min_loss = np.log10(config.line_minimum_frame_range_det)/(-0.4)
        # However this makes the flux too high and is not consistent with other measurements (that doesn't make
        #   those other measurements correct, something to be investiaged...)
        frame_min_loss = 0.0 
        # print("Frame min loss: {:.2} mag".format(frame_min_loss))
        lm_s_nightly_mean += frame_min_loss


        # Compute the nightly mean apparent meteor magnitude
        lm_m_nightly_mean = (lm_s_nightly_mean - 5*np.log10(r_mid/1e5) - 2.5*np.log10(np.degrees(
                    platepar.F_scale*v_init*np.sin(rad_dist_night_mid)
                   /(config.fps*r_mid*fwhm_nightly_mean)
                ))
            )

        print("Average stellar LM during the night: {:+.2f}".format(lm_s_nightly_mean))
        print("        meteor  LM during the night: {:+.2f}".format(lm_m_nightly_mean))


        ##### Apply time-dependent corrections #####



        # If there are no shower association, don't compute single-station flux
        if not associations:
            print("No meteors associated with the shower!")

            # Save empty flux files
            flux_table = FluxMeasurements()
            flux_table.initMetadata(shower_code, mass_index, calculatePopulationIndex(mass_index), \
                flux_config.gamma, 0, 0, 0, 0, 0, 0, 0, confidence_interval)

            flux_table.table.meta['fixed_bins'] = False
            flux_table.table.meta['sol_range'] = [0, 0]
            flux_table.saveECSV(os.path.join(dir_path, ecsv_file_name))


        elif (not forced_bins and not loaded_flux_computations) or (forced_bins and compute_single):

            # Apply corrections and compute the flux
            (   
                flux_table,
                sol_data,
                flux_lm_6_5_data,
                flux_lm_6_5_ci_lower_data,
                flux_lm_6_5_ci_upper_data,
                meteor_num_data,
                effective_collection_area_data,
                radiant_elev_data,
                radiant_dist_mid_data,
                ang_vel_mid_data,
                lm_s_data,
                lm_m_data,
                sensitivity_corr_data,
                range_corr_data,
                radiant_elev_corr_data,
                ang_vel_corr_data,
                total_corr_data,
                col_area_meteor_ht_raw,
                mag_median_data,
                mag_90_perc_data,
            ) = computeFluxCorrectionsOnBins(
                bin_meteor_information,
                bin_intervals,
                mass_index,
                population_index,
                shower,
                ht_std_percent,
                flux_config,
                config,
                col_areas_ht,
                v_init,
                azim_mid,
                elev_mid,
                r_mid,
                recalibrated_platepars,
                recalibrated_flux_platepars,
                platepar,
                frame_min_loss,
                ang_vel_night_mid,
                sensor_data,
                lm_m_nightly_mean,
                confidence_interval=confidence_interval,
                binduration=binduration,
            )


            # Save ECSV with flux measurements
            flux_table.table.meta['fixed_bins'] = False
            flux_table.table.meta['sol_range'] = [np.degrees(sol_beg), np.degrees(sol_end)]


            # Save the flux table to disk
            flux_table.saveECSV(os.path.join(dir_path, ecsv_file_name))


    # # Compute ZHR (Rentdel & Koschak, 1990 paper 2 method)
    # zhr_data = (np.array(flux_lm_6_5_data)/1000)*37200/((13.1*population_index - 16.5)*(population_index - 1.3)**0.748)

    if forced_bins:

        if not loaded_forced_bins:

            print("Calculating fixed bins...")


            (
                forced_flux_table,
                forced_sol_data,
                _,
                _,
                _,
                forced_bins_meteor_num,
                forced_bins_area,
                _,
                _,
                _,
                _,
                forced_bins_lm_m,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = computeFluxCorrectionsOnBins(
                forced_bin_meteor_information,
                forced_bin_intervals,
                mass_index,
                population_index,
                shower,
                ht_std_percent,
                flux_config,
                config,
                col_areas_ht,
                v_init,
                azim_mid,
                elev_mid,
                r_mid,
                recalibrated_platepars,
                recalibrated_flux_platepars,
                platepar,
                frame_min_loss,
                ang_vel_night_mid,
                sensor_data,
                lm_m_nightly_mean,
                confidence_interval=confidence_interval,
                fixed_bins=True,
                verbose=False
            )
            
            print('Finished computing collecting areas for fixed bins')


            # Add the range of solar longitudes to the ECSV table
            sol_beg = np.degrees(sol_bins[0])
            sol_end = np.degrees(sol_bins[-1])
            forced_flux_table.table.meta['sol_range'] = [sol_beg, sol_end]
            forced_flux_table.table.meta['fixed_bins'] = True

            # Override the original range of the solar longitudes (exclude the final bin)
            forced_flux_table.sol_lon_data = np.degrees(np.array(sol_bins[:-1]))

            # Save the fixed bin as an ECSV table
            forced_flux_table.saveECSV(os.path.join(dir_path, forced_bins_file))


            # ### TEST !!!!!1

            # test_sol_bins, test_forced_bins_meteor_num, test_forced_bins_area, test_forced_bins_time, \
            #     test_forced_bins_lm_m = loadForcedBinFluxData(dir_path, forced_bins_file)

            # print("sol bins", sol_bins, test_sol_bins)
            # print("bins_meteor_num", forced_bins_meteor_num, test_forced_bins_meteor_num)
            # print("bins_area", forced_bins_area, test_forced_bins_area)
            # print("time_bins", forced_bins_time, test_forced_bins_time)
            # print("bins_lm_m", forced_bins_lm_m, test_forced_bins_lm_m)

            # print()
            # input("Press ENTER to continue...")

            # ### ###



        # Scale the collecting area to +6.5M
        forced_bins_area = np.array(
            [
                area/population_index**(6.5 - lm) if lm is not None and not np.isnan(lm) else 0
                for area, lm in zip(forced_bins_area, forced_bins_lm_m)
            ]
        )

    #######################################


    if (show_plots or save_plots) and len(sol_data):


        # Print the results
        print("Solar longitude, Flux at LM +6.5:")
        for sol, flux_lm_6_5 in zip(sol_data, flux_lm_6_5_data):
            print("{:9.5f}, {:8.4f}".format(sol, flux_lm_6_5))


        # # Plot a histogram of peak magnitudes
        # plt.hist(peak_mags, cumulative=True, log=True, bins=len(peak_mags), density=True)

        # # Plot population index
        # r_intercept = -0.7
        # x_arr = np.linspace(np.min(peak_mags), np.percentile(peak_mags, 60))
        # plt.plot(x_arr, 10**(np.log10(population_index)*x_arr + r_intercept))

        # plt.title("r = {:.2f}".format(population_index))

        # plt.show()



        # Plot how the derived values change throughout the night
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10, 8))
        ((ax_met, ax_lm), (ax_rad, ax_corrs), (ax_ang_vel, ax_col_area), (ax_mag, ax_flux)) = axes

        # Set up shared axes (all except the magnitude plot)
        sharex_list = [ax_met, ax_lm, ax_rad, ax_corrs, ax_ang_vel, ax_col_area, ax_flux]
        sharex_list[0].get_shared_x_axes().join(*sharex_list)


        fig.suptitle("{:s}, s = {:.2f}, r = {:.2f}, $\\gamma = {:.2f}$".format(shower_code, mass_index, \
            population_index, flux_config.gamma))

        ax_met.scatter(sol_data, meteor_num_data)
        ax_met.set_ylabel("Meteors")

        ax_rad.plot(sol_data, radiant_elev_data, label="Radiant elevation", color='k', linestyle='solid')
        ax_rad.plot(sol_data, radiant_dist_mid_data, label="Radiant distance", color='k', linestyle='dashed')
        ax_rad.legend()
        ax_rad.set_ylabel("Angle (deg)")

        ax_ang_vel.plot(sol_data, ang_vel_mid_data)
        ax_ang_vel.set_ylabel("Ang vel (deg/s)")
        ax_ang_vel.set_xlabel("La Sun (deg)")



        # Plot a histogram of peak magnitudes
        nums, mag_bins, _ = ax_mag.hist(peak_mags, cumulative=True, log=True, bins=len(peak_mags), \
            density=True)

        # Constrain the intercept so that it matchs the median magnitude
        median_mag = np.median(peak_mags)

        # Find the bin closest to the median magnitude
        median_bin = np.argmin(np.abs(mag_bins - median_mag))
        median_mag_bin = mag_bins[median_bin]
        median_value = nums[median_bin]

        # Plot population index
        #r_intercept = -0.7
        r_intercept = np.log10(median_value) - np.log10(population_index)*median_mag_bin
        x_arr = np.linspace(np.min(peak_mags), np.percentile(peak_mags, 90))
        ax_mag.plot(x_arr, 10**(np.log10(population_index)*x_arr + r_intercept))

        # Only show the portion between the edge percentiles
        ax_mag.set_xlim(np.percentile(peak_mags, 10) - 1, np.percentile(peak_mags, 90) + 1)

        ax_mag.set_xlabel("Magnitude")
        ax_mag.set_ylabel("Density")


        ax_lm.plot(sol_data, lm_s_data, label="Stellar")
        ax_lm.plot(sol_data, lm_m_data, label="Meteor")
        ax_lm.plot(sol_data, mag_median_data, label='Median Meteor')
        ax_lm.plot(sol_data, mag_90_perc_data, label='90 Percentile Meteor')
        ax_lm.set_ylabel("LM")
        ax_lm.legend()

        ax_corrs.plot(sol_data, sensitivity_corr_data, label="Sensitivity")
        ax_corrs.plot(sol_data, range_corr_data, label="Range")
        ax_corrs.plot(sol_data, radiant_elev_corr_data, label="Rad elev")
        ax_corrs.plot(sol_data, ang_vel_corr_data, label="Ang vel")
        ax_corrs.plot(sol_data, total_corr_data, label="Total (median)")
        ax_corrs.set_ylabel("Corrections")
        ax_corrs.legend()


        # Plot the collection area
        ax_col_area.plot(sol_data, np.array(effective_collection_area_data)/1e9)
        ax_col_area.plot(
            sol_data, len(sol_data)*[col_area_100km_raw/1e9], color='k', label="Raw col area at 100 km"
        )
        ax_col_area.plot(
            sol_data,
            len(sol_data)*[col_area_meteor_ht_raw/1e9],
            color='k',
            linestyle='dashed',
            label="Raw col area at met ht",
        )
        ax_col_area.set_ylabel("Eff. col. area (1000 km$^2$)")
        ax_col_area.legend()

        # Plot the flux
        ax_flux.scatter(sol_data, flux_lm_6_5_data, color='k', zorder=4)
        ax_flux.errorbar(
            sol_data,
            flux_lm_6_5_data,
            color='grey',
            capsize=5,
            zorder=3,
            linestyle='none',
            yerr=[
                np.array(flux_lm_6_5_data) - np.array(flux_lm_6_5_ci_lower_data),
                np.array(flux_lm_6_5_ci_upper_data) - np.array(flux_lm_6_5_data),
            ],
        )

        ax_flux.set_ylabel("Flux (met/1000km$^2$/h)")
        ax_flux.set_xlabel("La Sun (deg)")

        # ### Add a ZHR axis ###
        # ax_flux_zhr = ax_flux.twinx()

        # # Plot ZHR to be invisible, just to align the axes
        # ax_flux_zhr.scatter(sol_data, zhr_data, alpha=1)
        # #ax_flux_zhr.set_ylim(ht_min/1000, ht_max/1000)
        # ax_flux_zhr.set_ylabel("ZHR")

        # ### ###

        plt.tight_layout()

        if save_plots:

            sol_beg = jd2SolLonSteyaert(datetime2JD(dt_beg))
            sol_end = jd2SolLonSteyaert(datetime2JD(dt_end))

            plt.savefig(os.path.join(dir_path, generateFluxPlotName(config.stationID, shower_code, \
                mass_index, sol_beg, sol_end)), dpi=150)

        if show_plots:
            plt.show()
        else:
            plt.clf()
            plt.close()


    if forced_bins:
        return (
            sol_data,
            flux_lm_6_5_data,
            flux_lm_6_5_ci_lower_data,
            flux_lm_6_5_ci_upper_data,
            meteor_num_data,
            population_index,
            (sol_bins, forced_bins_meteor_num, forced_bins_area, forced_bins_time),
        )
    return (
        sol_data,
        flux_lm_6_5_data,
        flux_lm_6_5_ci_lower_data,
        flux_lm_6_5_ci_upper_data,
        meteor_num_data,
        population_index,
    )




def prepareFluxFiles(config, dir_path, ftpdetectinfo_path):
    """ Prepare files necessary for quickly computing the flux. 
    
    Arguments:
        config: [Config]
        dir_path: [str] Path to the data directory.
        ftpdetectinfo_path: [str] Path to the FTPdetectinfo file.

    Return:
        None

    """

    # Init the flux configuration
    flux_config = FluxConfig()


    file_list = sorted(os.listdir(dir_path))


    # Load meteor data from the FTPdetectinfo file
    meteor_data = readFTPdetectinfo(*os.path.split(ftpdetectinfo_path))


    # Load the platepar file
    platepar = Platepar.Platepar()
    platepar.read(os.path.join(dir_path, config.platepar_name), use_flat=config.use_flat)


    # Locate and load the mask file
    mask = getMaskFile(dir_path, config, file_list=file_list)

    if mask is not None:
        mask.checkMask(platepar.X_res, platepar.Y_res)


    # Computes FWHM of stars
    getSensorCharacterization(dir_path, config, flux_config, meteor_data)

    # Compute collecting areas
    getCollectingArea(dir_path, config, flux_config, platepar, mask)

    # Run cloud detection and store the approprite files
    time_intervals = detectClouds(config, dir_path, mask=mask, save_plots=True, show_plots=False)


    ### Go through every shower that was active and prepare compute the flux ###

    # Load the list of showers used for flux computation
    flux_showers = FluxShowers(config)

    # Check for active showers in the observation periods
    for interval in time_intervals:
            
        dt_beg, dt_end = interval

        # Get a list of active showers in the time interval
        active_showers = flux_showers.activeShowers(dt_beg, dt_end)

        # Compute the flux for all active showers
        for shower in active_showers:

            # Calculate fixed 5 minute bins
            sol_bins_all, bin_datetime_dict = calculateFixedBins(time_intervals, [dir_path], shower)

            # Extract datetimes of forced bins relevant for this time interval
            dt_bins = bin_datetime_dict[np.argmax([year_start < dt_beg < year_end \
                for (year_start, year_end), _ in bin_datetime_dict])][1]
            forced_bins = (dt_bins, sol_bins_all)


            computeFlux(config, dir_path, ftpdetectinfo_path, shower, dt_beg, dt_end, shower.mass_index, \
                binmeteors=-1, forced_bins=forced_bins, save_plots=True, show_plots=False)


    ###


def fluxParser():
    """Returns arg parser for Flux.py __main__"""

    flux_parser = argparse.ArgumentParser(description="Compute single-station meteor shower flux.")

    flux_parser.add_argument(
        "ftpdetectinfo_path",
        metavar="FTPDETECTINFO_PATH",
        type=str,
        help="Path to an FTPdetectinfo file or path to folder. The directory also has to contain"
        "a platepar and mask file.",
    )


    flux_parser.add_argument(
        "shower_code", metavar="SHOWER_CODE", type=str, \
        help="IAU shower code (e.g. ETA, PER, SDA). Use 'auto' to compute the flux using the table in RMS."
    )

    flux_parser.add_argument("-s", "--massindex", metavar="MASS_INDEX", type=float, \
        help="Mass index of the shower. Only used when a specific shower is specified.")

    flux_parser.add_argument(
        "--timeinterval",
        nargs=2,
        metavar='INTERVAL',
        help="Time of the observation start and ending. YYYYMMDD_HHMMSS format for each",
    )

    # Decide on the binning if the flux is computed manually
    binning_group = flux_parser.add_mutually_exclusive_group(required=False)
    binning_group.add_argument(
        "--binduration", type=float, metavar='DURATION', help="Time bin width in hours."
    )
    binning_group.add_argument("--binmeteors", type=int, metavar='COUNT', help="Number of meteors per bin. Automatically determined by default.", \
        default=-1)

    flux_parser.add_argument(
        "-c",
        "--config",
        metavar="CONFIG_PATH",
        type=str,
        default='.',
        help="Path to a config file which will be used instead of the default one."
        " To load the .config file in the given data directory, write '.' (dot).",
    )
    flux_parser.add_argument(
        "--fwhm",
        metavar="DEFAULT_FWHM",
        type=float,
        help="For old datasets fwhm was not measured for CALSTARS files, in these cases, "
        "fwhm must be given (will be used only when necessary)",
    )
    flux_parser.add_argument(
        "--ratiothres",
        metavar="RATIO",
        type=float,
        default=0.5,
        help="Define a specific ratio threshold that will decide whether there are clouds. "
        "0.5 has been tested to be good and it is the default",
    )

    return flux_parser


if __name__ == "__main__":

    import RMS.ConfigReader as cr

    # COMMAND LINE ARGUMENTS
    # Init the command line arguments parser

    flux_parser = fluxParser()
    cml_args = flux_parser.parse_args()

    #########################

    ftpdetectinfo_path = cml_args.ftpdetectinfo_path
    ftpdetectinfo_path = findFTPdetectinfoFile(ftpdetectinfo_path)

    if not os.path.isfile(ftpdetectinfo_path):
        print("The FTPdetectinfo file does not exist:", ftpdetectinfo_path)
        print("Exiting...")
        sys.exit()

    # Extract parent directory
    dir_path = os.path.dirname(ftpdetectinfo_path)

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, dir_path)


    # Automatically prepare the files and compute the flux
    if cml_args.shower_code.lower() == 'auto':

        print("Automatically computing the flux...")

        prepareFluxFiles(config, dir_path, ftpdetectinfo_path)

        sys.exit()



    datetime_pattern = "%Y/%M/%d %H:%M:%S"
    
    # Use manually defined time intervals
    if cml_args.timeinterval is not None:
        dt_beg = datetime.datetime.strptime(cml_args.timeinterval[0], "%Y%m%d_%H%M%S")
        dt_end = datetime.datetime.strptime(cml_args.timeinterval[1], "%Y%m%d_%H%M%S")
        time_intervals = [(dt_beg, dt_end)]

    # Automatically deterine time intervals
    else:

        time_intervals = detectClouds(config, dir_path, show_plots=True, save_plots=False, \
            ratio_threshold=cml_args.ratiothres)

        for i, interval in enumerate(time_intervals):
            print(
                'interval {:d}/{:d}: '.format(i + 1, len(time_intervals)),
                '({:s},{:s})'.format(interval[0].strftime(datetime_pattern), interval[1].strftime(datetime_pattern))
            )

        # print('display ff with clouds')
        # for ff in detect_clouds:
        #     print(ff, detect_clouds[ff])


    # Compute the flux
    for dt_beg, dt_end in time_intervals:
        print('Using interval: ({:s},{:s})'.format(dt_beg.strftime(datetime_pattern), dt_end.strftime(datetime_pattern)))
        computeFlux(
            config,
            dir_path,
            ftpdetectinfo_path,
            cml_args.shower_code,
            dt_beg,
            dt_end,
            cml_args.massindex,
            cml_args.binduration,
            cml_args.binmeteors,
            default_fwhm=cml_args.fwhm,
        )
