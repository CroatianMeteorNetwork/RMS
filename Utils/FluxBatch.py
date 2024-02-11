""" Batch run the flux code using a flux batch file. """

from __future__ import print_function, division, absolute_import

import datetime
import os
import shlex
import sys
import collections
import copy
import configparser
import multiprocessing    

import ephem
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import scale as mscale
from cycler import cycler


from RMS.Astrometry.Conversions import datetime2JD, jd2Date
import RMS.ConfigReader as cr
from RMS.Formats.FTPdetectinfo import findFTPdetectinfoFile
from RMS.Formats.Showers import FluxShowers, loadRadiantShowers
from Utils.Flux import calculatePopulationIndex, calculateMassIndex, computeFlux, detectClouds, fluxParser, \
    calculateFixedBins, calculateZHR, massVerniani, loadShower
from RMS.Routines.SolarLongitude import unwrapSol
from RMS.Misc import formatScientific, roundToSignificantDigits, SegmentedScale, mkdirP
from RMS.QueuedPool import QueuedPool

# Now that the Scale class has been defined, it must be registered so
# that ``matplotlib`` can find it.
mscale.register_scale(SegmentedScale)



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
            color = self.color_cycle[len(self.color_dict)%(len(self.color_cycle))]
            label = station
            marker = self.markers[(len(self.marker_dict) // 10)%(len(self.markers))]

            # Assign plot color
            self.color_dict[station] = color
            self.marker_dict[station] = marker

        else:
            color = self.color_dict[station]
            marker = self.marker_dict[station]
            # label = str(config.stationID)
            label = None

        #return {'color': color, 'marker': marker, 'label': label}

        # Don't include the station name in the legend
        return {'color': color, 'marker': marker}


class FluxBatchBinningParams(object):
    def __init__(self, min_meteors=None, min_tap=None, min_bin_duration=None, max_bin_duration=None):
        """ Container for fluxBatch binning parameters. """

        if min_meteors is None:
            self.min_meteors = 50
        else:
            self.min_meteors = min_meteors

        if min_tap is None:            
            self.min_tap = 2
        else:
            self.min_tap = min_tap

        if min_bin_duration is None:        
            self.min_bin_duration = 0.5
        else:
            self.min_bin_duration = min_bin_duration

        if max_bin_duration is None:        
            self.max_bin_duration = 12
        else:
            self.max_bin_duration = max_bin_duration



class FluxBatchResults(object):
    def __init__(self, 
        # Shower object
        shower,
        # Input parameters used to compute the flux
        ref_ht, atomic_bin_duration, ci, min_meteors, min_tap, min_bin_duration, max_bin_duration, 
        compute_single,
        # Solar longitude bins
        sol_bins, bin_datetime_yearly, comb_sol, comb_sol_tap_weighted, comb_sol_bins, 
        # Per fixed bin numers and TAP
        num_meteors, time_area_product,
        # Flux data products
        comb_flux, comb_flux_lower, comb_flux_upper, 
        comb_flux_lm_m, comb_flux_lm_m_lower, comb_flux_lm_m_upper, 
        comb_zhr, comb_zhr_lower, comb_zhr_upper,
        # TAP-averaged parameters per bin
        comb_ta_prod, comb_num_meteors, comb_rad_elev, comb_rad_dist, comb_lm_m, comb_ang_vel,
        # Mag/mass limit information
        lm_m_mean, lm_m_to_6_5_factor, mass_lim, mass_lim_lm_m_mean,
        # Supplementary information
        v_init, summary_population_index, population_index_mean, single_fixed_bin_information,
        single_station_flux
        ):
        """ Container for results computed by the flux batch script. """

        # Shower object
        self.shower = shower

        # Input parameters used to compute the flux
        self.ref_ht = ref_ht
        self.atomic_bin_duration = atomic_bin_duration
        self.ci = ci
        self.min_meteors = min_meteors
        self.min_tap = min_tap
        self.min_bin_duration = min_bin_duration
        self.max_bin_duration = max_bin_duration
        self.compute_single = compute_single

        # Solar longitude bins
        self.sol_bins = sol_bins
        self.bin_datetime_yearly = bin_datetime_yearly
        self.comb_sol = comb_sol
        self.comb_sol_tap_weighted = comb_sol_tap_weighted
        self.comb_sol_bins = comb_sol_bins

        # Per fixed bin numers and TAP
        self.num_meteors = num_meteors
        self.time_area_product = time_area_product
        
        # Flux data products
        self.comb_flux = comb_flux
        self.comb_flux_lower = comb_flux_lower
        self.comb_flux_upper = comb_flux_upper
        self.comb_flux_lm_m = comb_flux_lm_m
        self.comb_flux_lm_m_lower = comb_flux_lm_m_lower 
        self.comb_flux_lm_m_upper = comb_flux_lm_m_upper
        self.comb_zhr = comb_zhr
        self.comb_zhr_lower = comb_zhr_lower
        self.comb_zhr_upper = comb_zhr_upper

        # TAP-averaged parameters per bin
        self.comb_ta_prod = comb_ta_prod
        self.comb_num_meteors = comb_num_meteors
        self.comb_rad_elev = comb_rad_elev
        self.comb_rad_dist = comb_rad_dist
        self.comb_lm_m = comb_lm_m
        self.comb_ang_vel = comb_ang_vel

        # Mag/mass limit information
        self.lm_m_mean = lm_m_mean
        self.lm_m_to_6_5_factor = lm_m_to_6_5_factor
        self.mass_lim = mass_lim
        self.mass_lim_lm_m_mean = mass_lim_lm_m_mean

        # Supplementary information
        self.v_init = v_init
        self.summary_population_index = summary_population_index
        self.population_index_mean = population_index_mean
        self.single_fixed_bin_information = single_fixed_bin_information
        self.single_station_flux = single_station_flux


    def unpack(self):
        """ Return all parameters. """

        (
        # Shower object
        self.shower,
        # Solar longitude bins
        self.sol_bins, self.bin_datetime_yearly, self.comb_sol, self.comb_sol_bins, 
        # Flux data products
        self.comb_flux, self.comb_flux_lower, self.comb_flux_upper, 
        self.comb_flux_lm_m, self.comb_flux_lm_m_lower, self.comb_flux_lm_m_upper, 
        self.comb_zhr, self.comb_zhr_lower, self.comb_zhr_upper,
        # TAP-averaged parameters per bin
        self.comb_ta_prod, self.comb_num_meteors, self.comb_rad_elev, self.comb_rad_dist, self.comb_lm_m, 
        self.comb_ang_vel,
        # Mag/mass limit information
        self.lm_m_mean, self.lm_m_to_6_5_factor, self.mass_lim, self.mass_lim_lm_m_mean,
        # Supplementary information
        self.v_init, self.summary_population_index, self.population_index_mean, 
        self.single_fixed_bin_information, self.single_station_flux,
        )




def addFixedBins(sol_bins, small_sol_bins, small_dt_bins, meteor_num_arr, collecting_area_arr, obs_time_arr, \
    lm_m_arr, rad_elev_arr, rad_dist_arr, ang_vel_arr, v_init_arr):
    """ Sort data into fixed bins by solar longitude. 

    For a larger array of solar longitudes sol_bins, fits parameters to an empty array of its size (minus 1)
    so that small_sol_bins agrees with sol_bins

    Assumes that for some index i, sol_bins[i:i+len(small_sol_bins)] = small_sol_bins. If this is not true,
    then the values are invalid and different small arrays should be used

    Arguments:
        sol_bins: [ndarray] Array of solar longitude bin edges. Does not wrap around
        small_sol_bins: [ndarray] Array of solar longitude bin edges which is smaller in length than
            sol_bins but can be transformed to sol_bins if shifted by a certain index. Does not wrap
            around.
        small_dt_bins: [ndarray] Datetime objects corresponding to the small_sol_bins edges. NOT USED.
        *params: [ndarray] Physical quantities such as number of meteors, collecting area.

    Return:
        [tuple] Same variables corresponding to params
            - val: [ndarray] Array of where any index that used to correspond to a sol in small_sol_bins,
                now corresponds to an index in sol_bins, padding all other values with zeros
    """

    # if sol_bins wraps would wrap around but forced_bins_sol doesn't
    if sol_bins[0] > small_sol_bins[0]:
        i = np.argmax(sol_bins - (small_sol_bins[0] + 360) > -1e-7)
    else:
        i = np.argmax(sol_bins - small_sol_bins[0] > -1e-7)  # index where they are equal


    # # Sort datetime edges into bins
    # dt_binned = np.zeros(len(sol_bins), dtype="datetime64[ms]")
    # dt_binned[i:i + len(small_dt_bins)] = small_dt_bins

    # Sort collecting area into bins
    collecting_area_binned = np.zeros(len(sol_bins) - 1)
    collecting_area_binned[i:i + len(collecting_area_arr)] = collecting_area_arr

    # Sort observation time into bins
    obs_time_binned = np.zeros(len(sol_bins) - 1)
    obs_time_binned[i:i + len(obs_time_arr)] = obs_time_arr

    # Sort meteor limiting magnitude into bins
    lm_m_binned = np.zeros(len(sol_bins) - 1) + np.nan
    lm_m_binned[i:i + len(obs_time_arr)] = lm_m_arr

    # Sort radiant elevation into bins
    rad_elev_binned = np.zeros(len(sol_bins) - 1) + np.nan
    rad_elev_binned[i:i + len(obs_time_arr)] = rad_elev_arr

    # Sort radiant distance into bins
    rad_dist_binned = np.zeros(len(sol_bins) - 1) + np.nan
    rad_dist_binned[i:i + len(obs_time_arr)] = rad_dist_arr

    # Sort angular velocity into bins
    ang_vel_binned = np.zeros(len(sol_bins) - 1) + np.nan
    ang_vel_binned[i:i + len(obs_time_arr)] = ang_vel_arr

    # Sort initial velocity into bins
    v_init_binned = np.zeros(len(sol_bins) - 1) + np.nan
    v_init_binned[i:i + len(obs_time_arr)] = v_init_arr


    # Sort meteor numbers into bins
    meteor_num_binned = np.zeros(len(sol_bins) - 1)
    meteor_num_binned[i:i + len(meteor_num_arr)] = meteor_num_arr

    # Set the number of meteors to zero where either the time or the collecting area is also zero
    meteor_num_binned[(collecting_area_binned == 0) | (obs_time_binned == 0)] = 0

    #data_arrays = []
    # for p in params:
    #     forced_bin_param = np.zeros(len(sol_bins) - 1)
    #     forced_bin_param[i:i + len(p)] = p
    #     data_arrays.append(forced_bin_param)

    return [meteor_num_binned, collecting_area_binned, obs_time_binned, lm_m_binned, rad_elev_binned, \
        rad_dist_binned, ang_vel_binned, v_init_binned]


def combineFixedBinsAndComputeFlux(
    sol_bins, meteors, time_area_prod, lm_m_data, rad_elev_data, rad_dist_data, ang_vel_data, v_init_data,
    ci=0.95,
    min_meteors=50, min_tap=2, min_bin_duration=0.5, max_bin_duration=12):
    """
    Computes flux values and their corresponding solar longitude based on bins containing
    number of meteors, and time-area product. Bins will be combined so that each bin has the
    minimum number of meteors, minimum TAP, and that other conditions are met.

    Arguments:
        sol_bins: [ndarray] Solar longitude of bins start and end (the length must be 1 more than meteors)
        meteors: [ndarray] Number of meteors in a bin
        time_area_prod: [ndarray] Time multiplied by LM corrected collecting area added for each station
            which contains each bin
        lm_m_data: [ndarray]
        rad_elev_data: [ndarray]
        rad_dist_data: [ndarray]
        ang_vel_data: [ndarray]
        v_init_data: [ndarray]

    Keyword arguments:
        ci: [float] Confidence interval for calculating the flux error bars (from 0 to 1)
        min_meteors: [int] Minimum number of meteors to have in a bin
        min_tap: [float] Minimum time area product in 1000 km^2*h.
        min_bin_duration: [float] Minimum bin duration in hours.
        max_bin_duration: [float] Maximum bin duration in hours.

    Return:
        [tuple] sol, flux, flux_lower, flux_upper, meteors, ta_prod
            - sol: [ndarray] Solar longitude
            - flux: [ndarray] Flux corresponding to solar longitude
            - flux_lower: [ndarray] Lower bound of flux corresponding to sol
            - flux_upper: [ndarray] Upper bound of flux corresponding to sol
            - meteor_count: [ndarray] Number of meteors in bin
            - time_area_product: [ndarray] Time area product of bin

    """
    middle_bin_sol = (sol_bins[1:] + sol_bins[:-1])/2

    flux_list = []
    flux_upper_list = []
    flux_lower_list = []
    sol_list = []
    sol_tap_weighted_list = []
    sol_bin_list = []
    meteor_count_list = []
    time_area_product_list = []
    lm_m_list = []
    rad_elev_list = []
    rad_dist_list = []
    ang_vel_list = []
    v_init_list = []

    # In some cases meteors can be an integer and the program crashes, so we need to check
    if not isinstance(meteors, int):

        start_idx = 0
        for end_idx in range(1, len(meteors)):

            sl = slice(start_idx, end_idx)

            # Compute the total duration of the bin (convert from solar longitude)
            bin_hours = (middle_bin_sol[end_idx] - middle_bin_sol[start_idx])/(2*np.pi)*24*365.24219

            # If the number of meteors, time-area product, and duration are larger than the limits, add this  
            #   as a new bin
            if (np.sum(meteors[sl]) >= min_meteors) and (np.nansum(time_area_prod[sl])/1e9 >= min_tap) \
                and (bin_hours >= min_bin_duration):

                # Sum up the values in the bin
                ta_prod = np.sum(time_area_prod[sl])
                num_meteors = np.sum(meteors[sl])

                meteor_count_list.append(num_meteors)
                time_area_product_list.append(ta_prod)

                if ta_prod == 0:
                    flux_list.append(np.nan)
                    flux_upper_list.append(np.nan)
                    flux_lower_list.append(np.nan)
                    lm_m_list.append(np.nan)
                    rad_elev_list.append(np.nan)
                    rad_dist_list.append(np.nan)
                    ang_vel_list.append(np.nan)
                    v_init_list.append(np.nan)

                else:

                    # Compute Poisson errors
                    n_meteors_upper = scipy.stats.chi2.ppf(0.5 + ci/2, 2*(num_meteors + 1))/2
                    n_meteors_lower = scipy.stats.chi2.ppf(0.5 - ci/2, 2*num_meteors)/2

                    # Compute the flux
                    flux_list.append(1e9*num_meteors/ta_prod)
                    flux_upper_list.append(1e9*n_meteors_upper/ta_prod)
                    flux_lower_list.append(1e9*n_meteors_lower/ta_prod)

                    # Compute the TAP-weighted meteor limiting magnitude
                    lm_m_select = lm_m_data[sl]*time_area_prod[sl]
                    lm_m_weighted = np.sum(lm_m_select[~np.isnan(lm_m_select)])/ta_prod
                    lm_m_list.append(lm_m_weighted)

                    # Compute the TAP-weighted radiant elevation
                    rad_elev_select = rad_elev_data[sl]*time_area_prod[sl]
                    rad_elev_weighted = np.sum(rad_elev_select[~np.isnan(rad_elev_select)])/ta_prod
                    rad_elev_list.append(rad_elev_weighted)

                    # Compute the TAP-weighted radiant distance
                    rad_dist_select = rad_dist_data[sl]*time_area_prod[sl]
                    rad_dist_weighted = np.sum(rad_dist_select[~np.isnan(rad_dist_select)])/ta_prod
                    rad_dist_list.append(rad_dist_weighted)

                    # Compute the TAP-weighted angular velocity
                    ang_vel_select = ang_vel_data[sl]*time_area_prod[sl]
                    ang_vel_weighted = np.sum(ang_vel_select[~np.isnan(ang_vel_select)])/ta_prod
                    ang_vel_list.append(ang_vel_weighted)

                    # Compute the TAP-weighted initial velocity
                    v_init_select = v_init_data[sl]*time_area_prod[sl]
                    v_init_weighted = np.sum(v_init_select[~np.isnan(v_init_select)])/ta_prod
                    v_init_list.append(v_init_weighted)


                sol_list.append(np.mean(middle_bin_sol[sl]))
                sol_tap_weighted_list.append(np.average(middle_bin_sol[sl], weights=time_area_prod[sl]))
                sol_bin_list.append(sol_bins[start_idx])
                start_idx = end_idx

            # If the total duration is over the maximum duration, skip the bin
            elif bin_hours >= max_bin_duration:
                start_idx = end_idx

        sol_bin_list.append(sol_bins[start_idx])

    return (
        np.array(sol_list),
        np.array(sol_tap_weighted_list),
        np.array(sol_bin_list),
        np.array(flux_list),
        np.array(flux_lower_list),
        np.array(flux_upper_list),
        np.array(meteor_count_list),
        np.array(time_area_product_list),
        np.array(lm_m_list),
        np.array(rad_elev_list),
        np.array(rad_dist_list),
        np.array(ang_vel_list),
        np.array(v_init_list)
    )



def cameraTally(comb_sol, comb_sol_bins, single_fixed_bin_information):
    """ Tally contributions from individual cameras in every time bin. 
        
    Arguments:
        comb_sol: [list] List of combined mean solar longitues (degrees).
        comb_sol_bins: [list] List of combined solar longitue bin edges (degrees).
        single_fixed_bin_information: [list] A list of [station, [sol (rad)], [meteor_num], [area (m^2)], 
            [time_bin (hour)]] entries for every station.
    """

    def _sortByParam(bin_tally, sol_bin, param, reverse=True):
        """ Compute top stations by the given parameter. """

        # Extract the stations in the given bin
        bin_cams_sorted = bin_tally[sol_bin]

        # Skip stations with a zero TAP
        bin_cams_sorted = {key:bin_cams_sorted[key] for key in bin_cams_sorted if bin_cams_sorted[key]['tap'] > 0}

        # Sort the stations in the given bin by the given parameter
        bin_cams_sorted = collections.OrderedDict(sorted(bin_cams_sorted.items(),
            key=lambda item: item[1][param], reverse=reverse))
        
        return bin_cams_sorted



    bin_tally = collections.OrderedDict()
    bin_tally_top_meteors      = collections.OrderedDict()
    bin_tally_top_tap          = collections.OrderedDict()
    bin_tally_faintest_lm      = collections.OrderedDict()
    bin_tally_brightest_lm     = collections.OrderedDict()
    bin_tally_highest_rad_elev = collections.OrderedDict()
    bin_tally_lowest_rad_elev  = collections.OrderedDict()
    bin_tally_highest_rad_dist = collections.OrderedDict()
    bin_tally_lowest_rad_dist  = collections.OrderedDict()
    bin_tally_highest_ang_vel  = collections.OrderedDict()
    bin_tally_lowest_ang_vel   = collections.OrderedDict()

    # Go through all solar longitude bins
    for i in range(len(comb_sol_bins) - 1):

        sol_start = np.radians(comb_sol_bins[i])
        sol_end   = np.radians(comb_sol_bins[i + 1])
        sol_mean  = comb_sol[i]

        # Add an entry for the bin
        if sol_mean not in bin_tally:
            bin_tally[sol_mean] = collections.OrderedDict()


        # Compute station contributions
        for station, (
                sol_arr, 
                _, 
                met_num, 
                area, 
                time_bin, 
                lm_m, 
                rad_elev, 
                rad_dist, 
                ang_vel, 
                _
                ) in single_fixed_bin_information:


            sol_arr  = np.array(sol_arr)
            met_num  = np.array(met_num)
            area     = np.array(area)
            time_bin = np.array(time_bin)
            lm_m     = np.array([np.nan if lm_tmp is None else lm_tmp for lm_tmp in lm_m])
            rad_elev = np.array(rad_elev)
            rad_dist = np.array(rad_dist)
            ang_vel  = np.array(ang_vel)

            # If there are no good bins, skip this
            if lm_m is None:
                continue

            if np.count_nonzero(~np.isnan(lm_m)) == 0:
                continue

            # Select data in the solar longitude range and with non-nan limiting magnitude
            sol_arr_unwrapped = unwrapSol(sol_arr[:-1], sol_start, sol_end)
            mask_arr = (sol_arr_unwrapped >= sol_start) & (sol_arr_unwrapped <= sol_end) & ~np.isnan(lm_m)

            # Set the number of meteors to 0 where the TAP or the observing duration are 0
            met_num[(area == 0) | (time_bin == 0)] = 0

            if np.any(mask_arr):

                if station not in bin_tally[sol_mean]:
                    
                    # Add an entry for the station, if it doesn't exist
                    bin_tally[sol_mean][station] = {
                        'meteors':  0, 
                        'tap':      0, 
                        'lm_m':     0, 
                        'rad_elev': 0, 
                        'rad_dist': 0, 
                        'ang_vel':  0, 
                        'tap_sum':  0
                    }

                # Add meteors and TAP numbers to the tally
                bin_tally[sol_mean][station]['meteors']  += np.sum(met_num[mask_arr])
                bin_tally[sol_mean][station]['tap']      += np.sum(area[mask_arr]*time_bin[mask_arr])

                # Add radiant information to compute TAP-weighted values
                tap_weight = area[mask_arr]*time_bin[mask_arr]
                bin_tally[sol_mean][station]['lm_m']     += np.sum(tap_weight*lm_m[mask_arr])
                bin_tally[sol_mean][station]['rad_elev'] += np.sum(tap_weight*rad_elev[mask_arr])
                bin_tally[sol_mean][station]['rad_dist'] += np.sum(tap_weight*rad_dist[mask_arr])
                bin_tally[sol_mean][station]['ang_vel']  += np.sum(tap_weight*ang_vel[mask_arr])
                bin_tally[sol_mean][station]['tap_sum']  += np.sum(tap_weight)


        # Compute the TAP-weighted radiant information values
        for station in bin_tally[sol_mean]:
            bin_tally[sol_mean][station]['lm_m']     /= bin_tally[sol_mean][station]['tap_sum']
            bin_tally[sol_mean][station]['rad_elev'] /= bin_tally[sol_mean][station]['tap_sum']
            bin_tally[sol_mean][station]['rad_dist'] /= bin_tally[sol_mean][station]['tap_sum']
            bin_tally[sol_mean][station]['ang_vel']  /= bin_tally[sol_mean][station]['tap_sum']

        # Sort by the number of meteors
        bin_tally_top_meteors[sol_mean]      = _sortByParam(bin_tally, sol_mean, 'meteors', reverse=True)
        bin_tally_top_tap[sol_mean]          = _sortByParam(bin_tally, sol_mean, 'tap', reverse=True)
        bin_tally_faintest_lm[sol_mean]      = _sortByParam(bin_tally, sol_mean, 'lm_m', reverse=True)
        bin_tally_brightest_lm[sol_mean]     = _sortByParam(bin_tally, sol_mean, 'lm_m', reverse=False)
        bin_tally_highest_rad_elev[sol_mean] = _sortByParam(bin_tally, sol_mean, 'rad_elev', reverse=True)
        bin_tally_lowest_rad_elev[sol_mean]  = _sortByParam(bin_tally, sol_mean, 'rad_elev', reverse=False)
        bin_tally_highest_rad_dist[sol_mean] = _sortByParam(bin_tally, sol_mean, 'rad_dist', reverse=True)
        bin_tally_lowest_rad_dist[sol_mean]  = _sortByParam(bin_tally, sol_mean, 'rad_dist', reverse=False)
        bin_tally_highest_ang_vel[sol_mean]  = _sortByParam(bin_tally, sol_mean, 'ang_vel', reverse=True)
        bin_tally_lowest_ang_vel[sol_mean]   = _sortByParam(bin_tally, sol_mean, 'ang_vel', reverse=False)


    return (
        bin_tally_top_meteors, 
        bin_tally_top_tap, 
        bin_tally_faintest_lm, 
        bin_tally_brightest_lm, 
        bin_tally_highest_rad_elev, 
        bin_tally_lowest_rad_elev, 
        bin_tally_highest_rad_dist, 
        bin_tally_lowest_rad_dist, 
        bin_tally_highest_ang_vel, 
        bin_tally_lowest_ang_vel
        )



def reportCameraTally(fbr, top_n_stations=5):
    """ Generate string report of top N stations per number of meteors and TAP from the camera tally results. """


    def _formatResults(bin_cams_top, station_id):
        """ Format a line given a sorted dictionary of station information. """

        station_data = bin_cams_top[station_id]

        n_meteors = station_data['meteors']
        tap       = station_data['tap']/1e6
        lm_m      = station_data['lm_m']
        rad_elev  = station_data['rad_elev']
        rad_dist  = station_data['rad_dist']
        ang_vel   = station_data['ang_vel']

        return "    {:s}, {:5d} meteors, TAP = {:10.2f} km^2 h, lm_m = {:+6.2f} M, rad elev = {:5.1f} deg, rad dist = {:6.1f} deg, ang vel = {:5.1f} deg/s\n".format(station_id, n_meteors, tap, lm_m, rad_elev, rad_dist, ang_vel)



    # Tally up contributions from individual cameras in each bin
    (
        bin_tally_top_meteors, 
        bin_tally_top_tap, 
        bin_tally_faintest_lm, 
        bin_tally_brightest_lm, 
        bin_tally_highest_rad_elev, 
        bin_tally_lowest_rad_elev, 
        bin_tally_highest_rad_dist, 
        bin_tally_lowest_rad_dist, 
        bin_tally_highest_ang_vel, 
        bin_tally_lowest_ang_vel
    ) = cameraTally(
        fbr.comb_sol, 
        fbr.comb_sol_bins,
        fbr.single_fixed_bin_information
        )

    out_str = ""

    out_str += "# Shower parameters:\n"
    out_str += "# Shower         = {:s}\n".format(fbr.shower.name)
    out_str += "# r              = {:.2f}\n".format(fbr.population_index_mean)
    out_str += "# s              = {:.2f}\n".format(calculateMassIndex(fbr.population_index_mean))
    out_str += "# m_lim @ +6.5M  = {:.2e} kg\n".format(fbr.mass_lim)
    out_str += "# Met LM mean    = {:.2e}\n".format(fbr.lm_m_mean)
    out_str += "# m_lim @ {:+.2f}M = {:.2e} kg\n".format(fbr.lm_m_mean, fbr.mass_lim_lm_m_mean)
    out_str += "# CI int.        = {:.1f} %\n".format(100*fbr.ci)
    out_str += "# Binning parameters:\n"
    out_str += "# Min. meteors     = {:d}\n".format(fbr.min_meteors)
    out_str += "# Min TAP          = {:.2f} x 1000 km^2 h\n".format(fbr.min_tap)
    out_str += "# Min bin duration = {:.2f} h\n".format(fbr.min_bin_duration)
    out_str += "# Max bin duration = {:.2f} h\n".format(fbr.max_bin_duration)

    out_str += "Camera tally per bin:\n"
    out_str += "---------------------\n"

    # Print cameras with most meteors per bin
    for sol_bin_mean in bin_tally_top_meteors:

        # Get cameras with most meteors
        bin_cams_topmeteors = bin_tally_top_meteors[sol_bin_mean]

        # Get cameras with the highest TAP
        bin_cams_toptap = bin_tally_top_tap[sol_bin_mean]

        # Extract radiant information
        bin_cams_faintest_lm      = bin_tally_faintest_lm[sol_bin_mean]
        bin_cams_brightest_lm     = bin_tally_brightest_lm[sol_bin_mean]
        bin_cams_highest_rad_elev = bin_tally_highest_rad_elev[sol_bin_mean]
        bin_cams_lowest_rad_elev  = bin_tally_lowest_rad_elev[sol_bin_mean]
        bin_cams_highest_rad_dist = bin_tally_highest_rad_dist[sol_bin_mean]
        bin_cams_lowest_rad_dist  = bin_tally_lowest_rad_dist[sol_bin_mean]
        bin_cams_highest_ang_vel  = bin_tally_highest_ang_vel[sol_bin_mean]
        bin_cams_lowest_ang_vel   = bin_tally_lowest_ang_vel[sol_bin_mean]


        ### Write the string

        out_str += "\n"
        out_str += "Sol = {:.4f} deg\n".format(sol_bin_mean)

        # Write stations with the most number of meteors
        out_str += "Top {:d} by meteor number:\n".format(top_n_stations)
        for i, station_id in enumerate(bin_cams_topmeteors):
            
            # Add a line for the top number of meteors
            out_str += _formatResults(bin_cams_topmeteors, station_id)

            if i == top_n_stations - 1:
                break

        # Write stations with the highest TAP
        out_str += "Top {:d} by TAP:\n".format(top_n_stations)
        for i, station_id in enumerate(bin_cams_toptap):

            # Add a line for the top TAP
            out_str += _formatResults(bin_cams_toptap, station_id)

            if i == top_n_stations - 1:
                break

        out_str += "Top {:d} by faintest meteor LM:\n".format(top_n_stations)
        for i, station_id in enumerate(bin_cams_faintest_lm):

            # Add a line for the top TAP
            out_str += _formatResults(bin_cams_faintest_lm, station_id)

            if i == top_n_stations - 1:
                break

        out_str += "Top {:d} by brightest meteor LM:\n".format(top_n_stations)
        for i, station_id in enumerate(bin_cams_brightest_lm):

            # Add a line for the top TAP
            out_str += _formatResults(bin_cams_brightest_lm, station_id)

            if i == top_n_stations - 1:
                break

        out_str += "Top {:d} by highest radiant elevation:\n".format(top_n_stations)
        for i, station_id in enumerate(bin_cams_highest_rad_elev):

            # Add a line for the top TAP
            out_str += _formatResults(bin_cams_highest_rad_elev, station_id)

            if i == top_n_stations - 1:
                break

        out_str += "Top {:d} by lowest radiant elevation:\n".format(top_n_stations)
        for i, station_id in enumerate(bin_cams_lowest_rad_elev):

            # Add a line for the top TAP
            out_str += _formatResults(bin_cams_lowest_rad_elev, station_id)

            if i == top_n_stations - 1:
                break

        out_str += "Top {:d} by highest radiant distance:\n".format(top_n_stations)
        for i, station_id in enumerate(bin_cams_highest_rad_dist):

            # Add a line for the top TAP
            out_str += _formatResults(bin_cams_highest_rad_dist, station_id)

            if i == top_n_stations - 1:
                break

        out_str += "Top {:d} by lowest radiant distance:\n".format(top_n_stations)
        for i, station_id in enumerate(bin_cams_lowest_rad_dist):

            # Add a line for the top TAP
            out_str += _formatResults(bin_cams_lowest_rad_dist, station_id)

            if i == top_n_stations - 1:
                break

        out_str += "Top {:d} by highest angular velocity:\n".format(top_n_stations)
        for i, station_id in enumerate(bin_cams_highest_ang_vel):

            # Add a line for the top TAP
            out_str += _formatResults(bin_cams_highest_ang_vel, station_id)

            if i == top_n_stations - 1:
                break

        out_str += "Top {:d} by lowest angular velocity:\n".format(top_n_stations)
        for i, station_id in enumerate(bin_cams_lowest_ang_vel):

            # Add a line for the top TAP
            out_str += _formatResults(bin_cams_lowest_ang_vel, station_id)

            if i == top_n_stations - 1:
                break


        ###

    return out_str



def computeTimeIntervalsPerStation(night_dir_path, time_intervals, binduration, binmeteors, fwhm, \
    ratio_threshold):
    """ Go though the given data folder and compute the time intervals when the flux should be computed.

    Arguments:
        night_dir_path: [str] Path to the night directory
        time_intervals: [tuple] (dt_beg, dt_end) pairs, if None automatically computed intervals will be taken
        binduration: [float] For single-station fluxes only
        binmeteors: [int] For single-station fluxes only
        fwhm: [float] Manual star FWHM, if not computed in CALSTARS files. None to take a default value
        ratio_threshold: [float] Star match ratio for determining cloudiness. None to take a default value
    """

    # Find the FTPdetectinfo file
    try:
        ftpdetectinfo_path = findFTPdetectinfoFile(night_dir_path)
    except FileNotFoundError:
        print("An FTPdetectinfo file could not be found! Skipping...")
        return None

    if not os.path.isfile(ftpdetectinfo_path):
        print("The FTPdetectinfo file does not exist:", ftpdetectinfo_path, "Skipping...")
        return None

    # Extract parent directory
    ftp_dir_path = os.path.dirname(ftpdetectinfo_path)

    # Load the config file
    try:
        config_station = cr.loadConfigFromDirectory('.', ftp_dir_path)

    except RuntimeError:
        print("The config file could not be loaded! Skipping...")
        return None

    except FileNotFoundError:
        print("The config file could not be loaded! Skipping...")
        return None

    except ValueError:
        print("The config file could not be loaded! Skipping...")
        return None

    except configparser.MissingSectionHeaderError:
        print("The config file could not be loaded! Skipping...")
        return None

    except configparser.DuplicateOptionError:
        print("The config file could not be loaded! Skipping...")
        return None

    except configparser.ParsingError:
        print("The config file could not be loaded! Skipping...")
        return None


    if time_intervals is None:
        
        # Find time intervals to compute flux with
        print('Detecting whether clouds are present...')

        time_intervals = detectClouds(
            config_station, ftp_dir_path, show_plots=False, ratio_threshold=ratio_threshold
        )

        print('Cloud detection complete!')
        print()

    else:
        dt_beg_temp = datetime.datetime.strptime(time_intervals[0], "%Y%m%d_%H%M%S")
        dt_end_temp = datetime.datetime.strptime(time_intervals[1], "%Y%m%d_%H%M%S")
        time_intervals = [[dt_beg_temp, dt_end_temp]]


    return config_station, ftp_dir_path, ftpdetectinfo_path, time_intervals, binduration, binmeteors, fwhm



def computeTimeIntervalsPerStationPoolFunc(args):
    """ Modify to one argument so the function works with the multiprocessing Pool. """

    return computeTimeIntervalsPerStation(*args)



def computeTimeIntervalsParallel(dir_params, cpu_cores=1):
    """ Find time intervals for given folders, using multiple CPUs.
    Arguments:
        dir_params: [list] A list of lists, per input directory:
            (night_dir_path, time_intervals, binduration, binmeteors, fwhm)
            - night_dir_path - path to the night directory
            - time_intervals - (dt_beg, dt_end) pairs, if None automatically computed intervals will be taken
            - binduration - for single-station fluxes only
            - binmeteors - for single-station fluxes only
            - fwhm - manual star FWHM, if not computed in CALSTARS files. None to take a default value
            - ratio_threshold - star match ratio for determining cloudiness. None to take a default value

    Keyword arguments:
        cpu_cores: [int] Number of CPU cores to use. If -1, all availabe cores will be used. 1 by default.
    """

    # Compute the time intervals using the given number of CPU cores
    file_data = []
    with multiprocessing.Pool(cpu_cores) as pool:

        results = pool.map(computeTimeIntervalsPerStationPoolFunc, dir_params)

        # Ignore entries for which there were no good time intervals
        for entry in results:
            if entry is not None:
                file_data.append(entry)

    return file_data



def computeFluxPerStation(file_entry, metadata_dir, shower_code, mass_index, ref_ht, bin_datetime_yearly, \
    sol_bins, ci, compute_single):
    """ Compute the flux for individual stations. """

    all_fixed_bin_information = []
    single_fixed_bin_information = []
    single_station_flux = []
    summary_population_index = []

    ## Compute the flux

    # Unpack the data
    config_station, ftp_dir_path, ftpdetectinfo_path, time_intervals, binduration, \
        binmeteors, fwhm = file_entry

    # Compute the flux in every observing interval
    for interval in time_intervals:

        dt_beg, dt_end = interval

        # Extract datetimes of forced bins relevant for this time interval
        dt_bins = bin_datetime_yearly[np.argmax([year_start < dt_beg < year_end \
            for (year_start, year_end), _ in bin_datetime_yearly])][1]

        forced_bins = (dt_bins, sol_bins)

        ret = computeFlux(
            config_station,
            ftp_dir_path,
            ftpdetectinfo_path,
            shower_code,
            dt_beg,
            dt_end,
            mass_index,
            binduration=binduration,
            binmeteors=binmeteors,
            ref_height=ref_ht,
            show_plots=False,
            default_fwhm=fwhm,
            confidence_interval=ci,
            forced_bins=forced_bins,
            compute_single=compute_single,
            metadata_dir=metadata_dir,
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

        # Skip observations with no computed fixed bins
        if len(bin_information[0]) == 0:
            continue

        # Sort measurements into fixed bins
        all_fixed_bin_information.append(addFixedBins(sol_bins, *bin_information))

        single_fixed_bin_information.append([config_station.stationID, bin_information])
        summary_population_index.append(population_index)


        # Add computed single-station flux to the output list
        single_station_flux += [
            [config_station.stationID, sol, flux, lower, upper, population_index]
            for (sol, flux, lower, upper) in zip(
                sol_data, flux_lm_6_5_data, flux_lm_6_5_ci_lower_data, flux_lm_6_5_ci_upper_data
            )
        ]

    return all_fixed_bin_information, single_fixed_bin_information, single_station_flux, \
        summary_population_index



def computeBatchFluxParallel(file_data, shower_code, mass_index, ref_ht, bin_datetime_yearly, sol_bins, ci, \
    compute_single, metadata_dir, cpu_cores=1):
    """ Compute flux in batch by distributing the computations on multiple CPU cores. 
    """

    if cpu_cores < 0:
        cpu_cores = multiprocessing.cpu_count()


    # If only one core is given, don't use multiprocessing
    if cpu_cores == 1:

        total_all_fixed_bin_information = []
        total_single_fixed_bin_information = []
        total_single_station_flux = []
        total_summary_population_index = []

        for file_entry in file_data:

            # Compute the flux per each station
            (
                all_fixed_bin_information, 
                single_fixed_bin_information, 
                single_station_flux,
                summary_population_index
            ) = computeFluxPerStation(
                file_entry, 
                metadata_dir, 
                shower_code, 
                mass_index, 
                ref_ht, 
                bin_datetime_yearly, 
                sol_bins, 
                ci, 
                compute_single
                )

            total_all_fixed_bin_information += all_fixed_bin_information
            total_single_fixed_bin_information += single_fixed_bin_information
            total_single_station_flux += single_station_flux
            total_summary_population_index += summary_population_index


    # Use multiple cores
    else:

        # Run the QueuedPool for detection (limit the input queue size for better memory management)
        workpool = QueuedPool(computeFluxPerStation, cores=cpu_cores, backup_dir=None, \
            func_extra_args=(shower_code, mass_index, ref_ht, bin_datetime_yearly, sol_bins, ci, compute_single),
            input_queue_maxsize=2*cpu_cores, worker_wait_inbetween_jobs=0.01,
            )

        print('Starting pool...')

        # Start the detection
        workpool.startPool()

        print('Adding jobs...')

        # Add jobs for the pool
        for file_entry in file_data:
            workpool.addJob([file_entry, metadata_dir], wait_time=0)


        print('Waiting for the batch flux computation to finish...')

        # Wait for the detector to finish and close it
        workpool.closePool()

        total_all_fixed_bin_information = []
        total_single_fixed_bin_information = []
        total_single_station_flux = []
        total_summary_population_index = []

        # Get extraction results
        for result in workpool.getResults():

            if result is None:
                continue

            all_fixed_bin_information, single_fixed_bin_information, single_station_flux, \
                summary_population_index = result

            total_all_fixed_bin_information += all_fixed_bin_information
            total_single_fixed_bin_information += single_fixed_bin_information
            total_single_station_flux += single_station_flux
            total_summary_population_index += summary_population_index


    return total_all_fixed_bin_information, total_single_fixed_bin_information, total_single_station_flux, \
        total_summary_population_index


def fluxBatch(config, shower_code, mass_index, dir_params, ref_ht=-1, atomic_bin_duration=5, ci=0.95, min_meteors=50, 
    min_tap=2, min_bin_duration=0.5, max_bin_duration=12, compute_single=False, metadata_dir=None, 
    cpu_cores=1):
    """ Compute flux by combining flux measurements from multiple stations.
    
    Arguments:
        shower_code: [str] Three letter IAU shower code (or whatever is defined in the flux table).
        mass_index: [float] Differential mass index of the shower.
        dir_params: [list] A list of lists, per input directory:
            (night_dir_path, time_intervals, binduration, binmeteors, fwhm)
            - night_dir_path - path to the night directory
            - time_intervals - (dt_beg, dt_end) pairs, if None automatically computed intervals will be taken
            - binduration - for single-station fluxes only
            - binmeteors - for single-station fluxes only
            - fwhm - manual star FWHM, if not computed in CALSTARS files. None to take a default value
            - ratio_threshold - star match ratio for determining cloudiness. None to take a default value

    Keyword arguments:
        ref_ht: [float] Reference height for the collection area (in km). If -1, a velocity dependent height
            model will be used.
        atomic_bin_duration: [float] Duration of the elemental bins (in minutes). 5 minutes by default.
        ci: [float] Confidence interval for calculating the flux error bars (from 0 to 1)
        min_meteors: [int] Minimum number of meteors to have in a bin
        min_tap: [float] Minimum time area product in 1000 km^2*h.
        min_bin_duration: [float] Minimum bin duration in hours.
        max_bin_duration: [float] Maximum bin duration in hours.
        compute_single: [bool] Compute single-station flux. False by default.
        metadata_dir: [str] A separate directory for flux metadata. If not given, the data directory will be
            used.
        cpu_cores: [int] Number of CPU cores to use. If -1, all availabe cores will be used. 1 by default.
    """

    # Make the metadata directory, if given
    if metadata_dir is not None:
        if not os.path.exists(metadata_dir):
            mkdirP(metadata_dir)


    # Go through all directories containing the flux data and prepare the time intervals
    print("Computing time intervals...")
    file_data = computeTimeIntervalsParallel(dir_params, cpu_cores=cpu_cores)

    # Load the shower object from the given shower code
    shower = loadShower(config, shower_code, mass_index, force_flux_list=True)

    # # Init the apparent speed
    # _, _, v_init = shower.computeApparentRadiant(0, 0, 2451545.0)

    # Override the mass index if given
    if mass_index is not None:
        shower.mass_index = mass_index



    print()
    print("Calculating fixed bins...")

    # Compute 5 minute bins of equivalent solar longitude every year
    sol_bins, bin_datetime_yearly = calculateFixedBins(
        [time_interval for data in file_data for time_interval in data[3]],
        [data[1] for data in file_data],
        shower,
        atomic_bin_duration=atomic_bin_duration,
        metadata_dir=metadata_dir
        )


    # Compute the batch flux using multiple CPU cores
    print("Computing flux...")
    (
        all_fixed_bin_information, 
        single_fixed_bin_information, 
        single_station_flux, 
        summary_population_index,
    ) = computeBatchFluxParallel(
        file_data, 
        shower_code, 
        mass_index, 
        ref_ht,
        bin_datetime_yearly, 
        sol_bins, 
        ci, 
        compute_single, 
        metadata_dir, 
        cpu_cores=cpu_cores
        )

    # Sum meteors in every bin (this is a 2D along the first axis, producing an array)
    num_meteors = np.sum(np.array(meteors, dtype=float) \
        for meteors, _, _, _, _, _, _, _ in all_fixed_bin_information)

    # Compute time-area product in every bin
    time_area_product = np.sum(np.array(area, dtype=float)*np.array(time, dtype=float) \
        for _, area, time, _, _, _, _, _ in all_fixed_bin_information)

    # Compute TAP-weighted meteor limiting magnitude in every bin
    lm_m_data = np.zeros_like(num_meteors, dtype=float)
    for _, area, time, lm_m, _, _, _, _ in all_fixed_bin_information:

        lm_m_data[~np.isnan(lm_m)] += (
             np.array(lm_m[~np.isnan(lm_m)])
            *np.array(area[~np.isnan(lm_m)])
            *np.array(time[~np.isnan(lm_m)])
            )

    lm_m_data /= time_area_product

    # Compute TAP-weighted radiant elevation in every bin
    rad_elev_data = np.zeros_like(num_meteors, dtype=float)
    for _, area, time, _, rad_elev, _, _, _ in all_fixed_bin_information:

        rad_elev_data[~np.isnan(rad_elev)] += (
             np.array(rad_elev[~np.isnan(rad_elev)])
            *np.array(area[~np.isnan(rad_elev)])
            *np.array(time[~np.isnan(rad_elev)])
            )

    rad_elev_data /= time_area_product


    # Compute TAP-weighted radiant distance in every bin
    rad_dist_data = np.zeros_like(num_meteors, dtype=float)
    for _, area, time, _, _, rad_dist, _, _ in all_fixed_bin_information:

        rad_dist_data[~np.isnan(rad_dist)] += (
             np.array(rad_dist[~np.isnan(rad_dist)])
            *np.array(area[~np.isnan(rad_dist)])
            *np.array(time[~np.isnan(rad_dist)])
            )

    rad_dist_data /= time_area_product


    # Compute TAP-weighted angular velocity in every bin
    ang_vel_data = np.zeros_like(num_meteors, dtype=float)
    for _, area, time, _, _, _, ang_vel, _ in all_fixed_bin_information:

        ang_vel_data[~np.isnan(ang_vel)] += (
             np.array(ang_vel[~np.isnan(ang_vel)])
            *np.array(area[~np.isnan(ang_vel)])
            *np.array(time[~np.isnan(ang_vel)])
            )

    ang_vel_data /= time_area_product

    # Compute the TAP-weighted initial velocity across all bins
    v_init_data = np.zeros_like(num_meteors, dtype=float)
    for _, area, time, _, _, _, _, v0 in all_fixed_bin_information:
            
        v_init_data[~np.isnan(v0)] += (
             np.array(v0[~np.isnan(v0)])
            *np.array(area[~np.isnan(v0)])
            *np.array(time[~np.isnan(v0)])
            )

    v_init_data /= time_area_product



    (
        comb_sol,
        comb_sol_tap_weighted,
        comb_sol_bins,
        comb_flux,
        comb_flux_lower,
        comb_flux_upper,
        comb_num_meteors,
        comb_ta_prod,
        comb_lm_m,
        comb_rad_elev,
        comb_rad_dist,
        comb_ang_vel,
        comb_v_init,
    ) = combineFixedBinsAndComputeFlux(
        sol_bins,
        num_meteors,
        time_area_product,
        lm_m_data,
        rad_elev_data,
        rad_dist_data,
        ang_vel_data,
        v_init_data,
        ci=ci,
        min_tap=min_tap,
        min_meteors=min_meteors,
        min_bin_duration=min_bin_duration,
        max_bin_duration=max_bin_duration,
    )
    comb_sol = np.degrees(comb_sol)
    comb_sol_tap_weighted = np.degrees(comb_sol_tap_weighted)
    comb_sol_bins = np.degrees(comb_sol_bins)

    # Computed the weidghted mean initial velocity
    summary_v_init = np.sum(
        comb_v_init[~np.isnan(comb_v_init)]*comb_ta_prod[~np.isnan(comb_v_init)]) \
            /np.sum(comb_ta_prod[~np.isnan(comb_v_init)]
            )


    # Compute the mass limit at 6.5 mag
    mass_lim = massVerniani(6.5, summary_v_init/1000)

    # Compute the weighted mean meteor magnitude
    lm_m_mean = np.sum(
        comb_lm_m[~np.isnan(comb_lm_m)]*comb_ta_prod[~np.isnan(comb_lm_m)]) \
            /np.sum(comb_ta_prod[~np.isnan(comb_lm_m)]
        )

    # Compute the mass limit at the mean meteor LM
    mass_lim_lm_m_mean = massVerniani(lm_m_mean, summary_v_init/1000)

    print("Mean TAP-weighted meteor limiting magnitude = {:.2f}M".format(lm_m_mean))
    print("                         limiting mass      = {:.2e} g".format(1000*mass_lim_lm_m_mean))

    # Compute the mean population index
    population_index_mean = np.mean(summary_population_index)

    # Compute the flux conversion factor
    lm_m_to_6_5_factor = population_index_mean**(6.5 - lm_m_mean)

    # Compute the flux to the mean meteor limiting magnitude
    comb_flux_lm_m = comb_flux/lm_m_to_6_5_factor
    comb_flux_lm_m_lower = comb_flux_lower/lm_m_to_6_5_factor
    comb_flux_lm_m_upper = comb_flux_upper/lm_m_to_6_5_factor


    # Compute the ZHR
    comb_zhr = calculateZHR(comb_flux, population_index_mean)
    comb_zhr_lower = calculateZHR(comb_flux_lower, population_index_mean)
    comb_zhr_upper = calculateZHR(comb_flux_upper, population_index_mean)


    # Store the results into a structure
    flux_batch_results = FluxBatchResults(
        # Shower object
        shower,
        # Input parameters used to compute the flux
        ref_ht, atomic_bin_duration, ci, min_meteors, min_tap, min_bin_duration, max_bin_duration, 
        compute_single,
        # Solar longitude bins
        sol_bins, bin_datetime_yearly, comb_sol, comb_sol_tap_weighted, comb_sol_bins, 
        # Per fixed bin numers and TAP
        num_meteors, time_area_product,
        # Flux data products
        comb_flux, comb_flux_lower, comb_flux_upper, 
        comb_flux_lm_m, comb_flux_lm_m_lower, comb_flux_lm_m_upper, 
        comb_zhr, comb_zhr_lower, comb_zhr_upper,
        # TAP-averaged parameters per bin
        comb_ta_prod, comb_num_meteors, comb_rad_elev, comb_rad_dist, comb_lm_m, comb_ang_vel,
        # Mag/mass limit information
        lm_m_mean, lm_m_to_6_5_factor, mass_lim, mass_lim_lm_m_mean,
        # Supplementary information
        summary_v_init, summary_population_index, population_index_mean, single_fixed_bin_information,
        single_station_flux)

    return flux_batch_results



def plotBatchFlux(fbr, dir_path, output_filename, only_flux=False, compute_single=False, show_plot=True,
    xlim_shower_limits=False, sol_marker=None, publication_quality=False):
    """ Make a plot showing the batch flux results. 
    
    Arguments:
        fbr: [FluxBatchResults object]
        dir_path: [str] Path to where the plot will be saved.
        output_filename: [str] Plot file name. .png will be added to it.

    Keyword arguments:
        only_flux: [bool] Only plot the flux graph, skip other metadata. False by default.
        compute_single: [bool] Also plot per-station single-station fluxes. False by default.
        show_plot: [bool] Show the plot on the screen. True by default.
        xlim_shower_limits: [bool] If True, set the plot x axis limits to the shower activity extent. False
            by default.
        sol_marker: [float] Plot a red vertical line on the flux plot at the given solar longitude (deg).
            None by default, in which case the marker will not be plotted.
        publication_quality: [bool] If True, make the plot publication quality with no grid, larger font,
            and only keeping the flux and TAP subplots, not the individual parameters. False by default.
    """


    ### Init the plot ###

    plot_info = StationPlotParams()
    
    if only_flux:

        subplot_rows = 1
        figsize = (15, 5)

    else:

        if publication_quality:
            subplot_rows = 2
            figsize = (15, 10)

        else:
            subplot_rows = 4
            figsize = (15, 10)

    fig, ax = plt.subplots(nrows=subplot_rows, figsize=figsize, sharex=True, \
        gridspec_kw={'height_ratios': [3, 1, 1, 1][:subplot_rows]})
    

    # If the plot is for publication, set the plot parameters
    if publication_quality:

        plt.rcParams.update({'font.size': 16})
        plt.rcParams.update({'axes.labelsize': 16})
        plt.rcParams.update({'xtick.labelsize': 16})
        plt.rcParams.update({'ytick.labelsize': 16})
        plt.rcParams.update({'legend.fontsize': 16})
        plt.rcParams.update({'figure.autolayout': True})
        plt.rcParams.update({'axes.grid': False})


    if not isinstance(ax, np.ndarray):
        ax = [ax]

    ### ###


    # Plot single-station data
    if compute_single:

        for (station_id, sol_data, flux_lm_6_5_data, flux_lm_6_5_ci_lower_data, flux_lm_6_5_ci_upper_data, 
            _) in fbr.single_station_flux:

            # plot data for night and interval
            plot_params = plot_info(station_id)

            # Plot the single-station flux line
            ax[0].plot(sol_data, flux_lm_6_5_data, linestyle='dashed', **plot_params)

            # Plot single-station error bars
            ax[0].errorbar(
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


    # If data was able to be combined, plot the weighted flux
    if len(fbr.comb_sol):

        # Plotting weigthed flux
        ax[0].errorbar(
            fbr.comb_sol_tap_weighted%360,
            fbr.comb_flux,
            yerr=[fbr.comb_flux - fbr.comb_flux_lower, fbr.comb_flux_upper - fbr.comb_flux],
            label="Weighted average flux at:\n" \
                + "LM = +6.5$^{\\mathrm{M}}$, " \
                + r"(${:s}$ g)".format(formatScientific(1000*fbr.mass_lim, 0)),
                #+ "$m_{\\mathrm{lim}} = $" + "${:s}$".format(formatScientific(1000*mass_lim, 0)) + " g (+6.5$^{\\mathrm{M}}$)",
            c='k',
            marker='o',
            linestyle='none',
            zorder=4,
            capsize=3,
        )

        # Plot the flux to the meteor LM
        lm_flux_plot = ax[0].errorbar(
            fbr.comb_sol_tap_weighted%360,
            fbr.comb_flux_lm_m,
            yerr=[fbr.comb_flux_lm_m - fbr.comb_flux_lm_m_lower, 
                  fbr.comb_flux_lm_m_upper - fbr.comb_flux_lm_m],
            label="Flux (1/{:.2f}x) at:\n".format(fbr.lm_m_to_6_5_factor) \
                + "LM = {:+.2f}".format(fbr.lm_m_mean) + "$^{\\mathrm{M}}$, " \
                + r"(${:s}$ g)".format(formatScientific(1000*fbr.mass_lim_lm_m_mean, 0)),
                #+ "$m_{\\mathrm{lim}} = $" + "${:s}$".format(formatScientific(1000*mass_lim_lm_m_mean, 0)) + " g ({:+.2f}".format(lm_m_mean) + "$^{\\mathrm{M}}$) ", \
            c='0.5',
            marker='o',
            linestyle='none',
            zorder=4,
            capsize=3,
        )

        ax[0].legend()

        # Set the minimum flux to 0
        ax[0].set_ylim(bottom=0)

        ax[0].set_ylabel("Flux (meteoroids / 1000 $\\cdot$ km$^2$ $\\cdot$ h)")


        if not publication_quality:

            # Add the grid
            ax[0].grid(color='0.9')
            
            # Set the title
            ax[0].set_title("{:s}, v = {:.1f} km/s, s = {:.2f}, r = {:.2f}".format(fbr.shower.name_full, 
                fbr.v_init/1000, calculateMassIndex(np.mean(fbr.summary_population_index)), 
                np.mean(fbr.summary_population_index)) 
                        # + ", $\\mathrm{m_{lim}} = $" + r"${:s}$ g ".format(formatScientific(1000*mass_lim, 0))
                        # + "at LM = +6.5$^{\\mathrm{M}}$"
                        )
            

            # Plot the marker
            if sol_marker is not None:
                ax[0].axvline(x=sol_marker, color='r', linewidth=1)


        ### Plot the ZHR on another axis ###

        # Create the right axis
        zhr_ax = ax[0].twinx()

        population_index = np.mean(fbr.summary_population_index)

        # Set the same range on the Y axis
        y_min, y_max = ax[0].get_ylim()
        zhr_min, zhr_max = calculateZHR([y_min, y_max], population_index)
        zhr_ax.set_ylim(zhr_min, zhr_max)

        # Get the flux ticks and set them to the zhr axis
        flux_ticks = ax[0].get_yticks()
        zhr_ax.set_yscale('segmented', 
            points=roundToSignificantDigits(calculateZHR(flux_ticks, population_index), n=2))

        zhr_ax.set_ylabel("ZHR at +6.5$^{\\mathrm{M}}$")

        ### ###


        if not only_flux:


            ##### SUBPLOT 1 #####

            # Plot time-area product in the bottom plot
            ax[1].bar(
                ((fbr.comb_sol_bins[1:] + fbr.comb_sol_bins[:-1])/2)%360,
                fbr.comb_ta_prod/1e9,
                fbr.comb_sol_bins[1:] - fbr.comb_sol_bins[:-1],
                label='Time-area product (TAP)',
                color='0.65',
                edgecolor='0.45'
            )



            # Plot TAP distribution within individual bins
            for i, sol_end_temp in enumerate(fbr.comb_sol_bins[1:]):

                sol_beg_temp = fbr.comb_sol_bins[i]

                # Select the current TAP height
                tap_temp = fbr.comb_ta_prod[i]/1e9

                # Select the range of sols for the current bar
                selection_mask = (np.degrees(fbr.sol_bins) >= sol_beg_temp) \
                                    & (np.degrees(fbr.sol_bins) <= sol_end_temp)

                sol_range = np.degrees(fbr.sol_bins[selection_mask])
                tap_range = fbr.time_area_product[selection_mask[1:]]

                if len(sol_range) > len(tap_range):
                    sol_range = sol_range[(len(sol_range) - len(tap_range)):]

                ax[1].bar(
                    ((sol_range[1:] + sol_range[:-1])/2)%360,
                    tap_range[1:]/np.max(tap_range)*tap_temp,
                    sol_range[1:] - sol_range[:-1],
                    color='0.35',
                    edgecolor='none',
                    alpha=0.5
                )


            # Plot the minimum time-area product as a horizontal line
            ax[1].hlines(
                fbr.min_tap,
                np.min(fbr.comb_sol%360),
                np.max(fbr.comb_sol%360),
                colors='k',
                linestyles='solid',
                label="Min. TAP",
            )

            ax[1].set_ylabel("TAP (1000 $\\cdot$ km$^2$ $\\cdot$ h)")


            # Plot the number of meteors on the right axis
            side_ax = ax[1].twinx()
            side_ax.scatter(fbr.comb_sol_tap_weighted%360, fbr.comb_num_meteors, c='k', label='Meteors', s=8)

            # Plot the minimum meteors line
            side_ax.hlines(
                fbr.min_meteors,
                np.min(fbr.comb_sol%360),
                np.max(fbr.comb_sol%360),
                colors='k',
                linestyles='--',
                label="Min. meteors"
            )
            side_ax.set_ylabel('Num meteors')
            side_ax.set_ylim(bottom=0)


            # Add a combined legend
            lines, labels = ax[1].get_legend_handles_labels()
            lines2, labels2 = side_ax.get_legend_handles_labels()
            side_ax.legend(lines + lines2, labels + labels2)

            
            if not publication_quality:

                ##### SUBPLOT 2 #####

                # Plot the radiant elevation
                ax[2].scatter(fbr.comb_sol_tap_weighted%360, fbr.comb_rad_elev, \
                    label="Rad. elev. (TAP-weighted)", color='0.75', s=15, marker='s')

                # Plot the radiant distance
                ax[2].scatter(fbr.comb_sol_tap_weighted%360, fbr.comb_rad_dist, \
                    label="Rad. dist.", color='0.25', s=20, marker='x')

                ax[2].set_ylabel("Angle (deg)")

                ### Plot lunar phases per year ###

                moon_ax = ax[2].twinx()

                # Set line plot cycler
                line_cycler   = (
                    cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                    cycler(linestyle=["-", "--", "-.", ":", "-", "--", "-."])
                    )

                moon_ax.set_prop_cycle(line_cycler)


                # Set up observer
                o = ephem.Observer()
                o.lat = str(0)
                o.long = str(0)
                o.elevation = 0
                o.horizon = '0:0'

                year_list = []
                for dt_range, dt_arr in fbr.bin_datetime_yearly:

                    dt_bin_beg, dt_bin_end = dt_range
                    dt_mid = jd2Date((datetime2JD(dt_bin_beg) + datetime2JD(dt_bin_end))/2, dt_obj=True)

                    # Make sure the years are not repeated
                    if dt_mid.year in year_list:
                        continue

                    year_list.append(dt_mid.year)

                    moon_phases = []

                    for dt in dt_arr:

                        o.date = dt
                        m = ephem.Moon()
                        m.compute(o)

                        moon_phases.append(m.phase)

                    # Plot Moon phases
                    moon_ax.plot(np.degrees(fbr.sol_bins), moon_phases, label="{:d} moon phase".format(dt_mid.year))

                moon_ax.set_ylabel("Moon phase")
                moon_ax.set_ylim([0, 100])

                # Add a combined legend
                lines, labels = ax[2].get_legend_handles_labels()
                lines2, labels2 = moon_ax.get_legend_handles_labels()
                moon_ax.legend(lines + lines2, labels + labels2)

                ### ###


                ##### SUBPLOT 3 #####

                ### Plot the TAP-weighted limiting magnitude ###

                lm_ax = ax[3].twinx()

                lm_ax.scatter(fbr.comb_sol_tap_weighted%360, fbr.comb_lm_m, label="Meteor LM", color='0.5', s=20)

                lm_ax.invert_yaxis()
                lm_ax.set_ylabel("Meteor LM")
                #lm_ax.legend()

                # Add one magnitude of buffer to every end, round to 0.5
                lm_min, lm_max = lm_ax.get_ylim()
                lm_ax.set_ylim(np.ceil(2*(lm_min))/2, np.floor(2*(lm_max))/2)


                # Plot the TAP-weighted meteor LM

                lm_ax.hlines(
                    fbr.lm_m_mean,
                    np.min(fbr.comb_sol%360),
                    np.max(fbr.comb_sol%360),
                    colors='k',
                    alpha=0.5,
                    linestyles='dashed',
                    label="Mean meteor LM = {:+.2f}".format(fbr.lm_m_mean) + "$^{\\mathrm{M}}$",
                )


                ###

                
                # Plot the angular velocity
                ax[3].scatter(fbr.comb_sol_tap_weighted%360, fbr.comb_ang_vel, \
                    label="Angular velocity", color='0.0', s=30, marker='+')
                ax[3].set_ylabel("Ang. vel. (deg/s)")


                # Add a combined legend
                lines, labels = ax[3].get_legend_handles_labels()
                lines2, labels2 = lm_ax.get_legend_handles_labels()
                lm_ax.legend(lines + lines2, labels + labels2)


        ax[subplot_rows - 1].set_xlabel("Solar longitude (deg)")


    # Set X axis limits
    if xlim_shower_limits:
        ax[0].set_xlim(fbr.shower.lasun_beg, fbr.shower.lasun_end)

    
    plt.tight_layout()

    fig_path = os.path.join(dir_path, output_filename + ".png")
    print("Figure saved to:", fig_path)
    plt.savefig(fig_path, dpi=300)

    # Also save a PDF file for publication-quality plots
    if publication_quality:

        fig_path = os.path.join(dir_path, output_filename + ".pdf")
        print("Figure saved to:", fig_path)
        plt.savefig(fig_path, format='pdf', dpi=300)


    if show_plot:
        plt.show()

    else:
        plt.clf()
        plt.close()



def saveBatchFluxCSV(fbr, dir_path, output_filename):
    """ Save the binned flux batch results to a CSV file. """

    # Write the computed weigthed flux to disk
    if len(fbr.comb_sol):

        data_out_path = os.path.join(dir_path, output_filename + ".csv")
        with open(data_out_path, 'w') as fout:
            fout.write("# Shower parameters:\n")
            fout.write("# Shower         = {:s}\n".format(fbr.shower.name))
            fout.write("# r              = {:.2f}\n".format(fbr.population_index_mean))
            fout.write("# s              = {:.2f}\n".format(calculateMassIndex(fbr.population_index_mean)))
            fout.write("# m_lim @ +6.5M  = {:.2e} kg\n".format(fbr.mass_lim))
            fout.write("# Met LM mean    = {:.2e}\n".format(fbr.lm_m_mean))
            fout.write("# m_lim @ {:+.2f}M = {:.2e} kg\n".format(fbr.lm_m_mean, fbr.mass_lim_lm_m_mean))
            fout.write("# CI int.        = {:.1f} %\n".format(100*fbr.ci))
            fout.write("# Binning parameters:\n")
            fout.write("# Min. meteors     = {:d}\n".format(fbr.min_meteors))
            fout.write("# Min TAP          = {:.2f} x 1000 km^2 h\n".format(fbr.min_tap))
            fout.write("# Min bin duration = {:.2f} h\n".format(fbr.min_bin_duration))
            fout.write("# Max bin duration = {:.2f} h\n".format(fbr.max_bin_duration))
            fout.write(
                "# Sol bin start (deg), Mean Sol (deg), TAP-weighted Sol (deg), Flux@+6.5M (met / 1000 km^2 h), Flux CI low, Flux CI high, Flux@+{:.2f}M (met / 1000 km^2 h), Flux CI low, Flux CI high, ZHR, ZHR CI low, ZHR CI high, Meteor Count, Time-area product (corrected to +6.5M) (1000 km^2/h), Meteor LM, Radiant elev (deg), Radiat dist (deg), Ang vel (deg/s)\n".format(fbr.lm_m_mean)
            )
            for (
                _sol_bin_start,
                _tap_weighted_sol,
                _mean_sol,
                _flux,
                _flux_lower,
                _flux_upper,
                _flux_lm,
                _flux_lm_lower,
                _flux_lm_upper,
                _zhr,
                _zhr_lower,
                _zhr_upper,
                _nmeteors,
                _tap,
                _lm_m,
                _rad_elev,
                _rad_dist,
                _ang_vel) \
            in zip(
                    fbr.comb_sol_bins,
                    fbr.comb_sol,
                    fbr.comb_sol_tap_weighted,
                    fbr.comb_flux,
                    fbr.comb_flux_lower,
                    fbr.comb_flux_upper,
                    fbr.comb_flux_lm_m,
                    fbr.comb_flux_lm_m_lower,
                    fbr.comb_flux_lm_m_upper,
                    fbr.comb_zhr,
                    fbr.comb_zhr_lower,
                    fbr.comb_zhr_upper,
                    fbr.comb_num_meteors,
                    fbr.comb_ta_prod,
                    fbr.comb_lm_m,
                    fbr.comb_rad_elev,
                    fbr.comb_rad_dist,
                    fbr.comb_ang_vel
                    ):

                fout.write(
                    "{:.8f},{:.8f},{:.8f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:d},{:.3f},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(
                    _sol_bin_start,
                    _mean_sol,
                    _tap_weighted_sol,
                    _flux,
                    _flux_lower,
                    _flux_upper,
                    _flux_lm,
                    _flux_lm_lower,
                    _flux_lm_upper,
                    _zhr,
                    _zhr_lower,
                    _zhr_upper,
                    int(_nmeteors),
                    _tap/1e9,
                    _lm_m,
                    _rad_elev,
                    _rad_dist,
                    _ang_vel,
                    ))

            fout.write("{:.8f},,,,,,,,,,,,,,,,\n".format(fbr.comb_sol_bins[-1]))


        return data_out_path






if __name__ == "__main__":

    import argparse

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(
        description="Compute multi-station and multi-year meteor shower flux from a batch file."
    )

    arg_parser.add_argument("batch_path", metavar="BATCH_PATH", type=str, help="Path to the flux batch file.")

    arg_parser.add_argument(
        "--output_filename",
        metavar="FILENAME",
        type=str,
        default='fluxbatch_output',
        help="Filename to export images and data (exclude file extensions), defaults to fluxbatch_output",
    )

    # NOTE: The CSV option is disabled for now, more work needs to be done to fully support it
    # arg_parser.add_argument(
    #     "-csv",
    #     action='store_true',
    #     help="If given, will read from the csv files defined with output_filename (defaults to fluxbatch_output)",
    # )

    arg_parser.add_argument(
        "--single",
        action='store_true',
        help="Show single-station fluxes.",
    )

    arg_parser.add_argument(
        "--onlyflux",
        action='store_true',
        help="Only plot the flux, without the additional plots.",
    )

    arg_parser.add_argument(
        "--minmeteors",
        type=int,
        default=30,
        help="Minimum meteors per bin. If this is not satisfied the bin will be made larger. Default = 30 meteors.",
    )

    arg_parser.add_argument(
        "--mintap",
        type=float,
        default=3,
        help="Minimum time-area product per bin. If this is not satisfied the bin will be made larger. Default = 3 x 1000 km^2 h.",
    )

    arg_parser.add_argument(
        "--minduration",
        type=float,
        default=0.5,
        help="Minimum time per bin in hours. If this is not satisfied the bin will be made larger. Default = 0.5 h.",
    )

    arg_parser.add_argument(
        "--maxduration",
        type=float,
        default=12,
        help="Maximum time per bin in hours. If this is not satisfied, the bin will be discarded. Default = 12 h.",
    )

    arg_parser.add_argument('-m', '--metadir', metavar='FLUX_METADATA_DIRECTORY', type=str,
        help="Path to a directory with flux metadata (ECSV files). If not given, the data directory will be used.")

    arg_parser.add_argument(
        "--cpucores",
        type=int,
        default=1,
        help="Number of CPU codes to use for computation. -1 to use all cores. 1 by default.",
    )

    # Parse the command line arguments
    fluxbatch_cml_args = arg_parser.parse_args()

    #########################


    # Only run in Python 3+
    if sys.version_info[0] < 3:
        print("The flux code can only run in Python 3+ !")
        sys.exit()


    ### Binning parameters ###

    # Confidence interval
    ci = 0.95

    # Base bin duration (minutes)
    atomic_bin_duration = 5

    # Minimum number of meteors in the bin
    min_meteors = fluxbatch_cml_args.minmeteors

    # Minimum time-area product (1000 km^2 h)
    min_tap = fluxbatch_cml_args.mintap

    # Minimum bin duration (hours)
    min_bin_duration = fluxbatch_cml_args.minduration

    # Maximum bin duration (hours)
    max_bin_duration = fluxbatch_cml_args.maxduration

    # Init the binning parametrs into a container
    fb_bin_params = FluxBatchBinningParams(
        min_meteors=min_meteors, 
        min_tap=min_tap, 
        min_bin_duration=min_bin_duration, 
        max_bin_duration=max_bin_duration,
        )

    ### ###


    # Check if the batch file exists
    if not os.path.isfile(fluxbatch_cml_args.batch_path):
        print("The given batch file does not exist!", fluxbatch_cml_args.batch_path)
        sys.exit()

    dir_path = os.path.dirname(fluxbatch_cml_args.batch_path)

    shower_code = None

    # Load the default config file
    config = cr.Config()
    config = cr.parse(config.config_file_name)


    # NOTE: CSV option not supported
    # # If an input CSV file was not given, compute the data
    # if not fluxbatch_cml_args.csv:



    # Loading commands from batch file and collecting information to run batchFlux
    dir_params = []
    with open(fluxbatch_cml_args.batch_path) as f:

        # Parse the batch entries
        for line in f:
            line = line.replace("\n", "").replace("\r", "")

            if not len(line):
                continue

            if line.startswith("#"):
                continue

            flux_cml_args = fluxParser().parse_args(shlex.split(line, posix=0))
            (
                ftpdetectinfo_path,
                shower_code,
                mass_index,
                binduration,
                binmeteors,
                time_intervals,
                fwhm,
                ratio_threshold,
                ref_ht
            ) = (
                flux_cml_args.ftpdetectinfo_path,
                flux_cml_args.shower_code,
                flux_cml_args.massindex,
                flux_cml_args.binduration,
                flux_cml_args.binmeteors,
                flux_cml_args.timeinterval,
                flux_cml_args.fwhm,
                flux_cml_args.ratiothres,
                flux_cml_args.ht,
            )

            dir_params.append([ftpdetectinfo_path, time_intervals, binduration, binmeteors, fwhm, 
                ratio_threshold])




    # Compute the batch flux
    fbr = fluxBatch(config, shower_code, mass_index, dir_params, ref_ht=ref_ht, ci=ci,
            atomic_bin_duration=atomic_bin_duration, min_meteors=fb_bin_params.min_meteors, 
            min_tap=fb_bin_params.min_tap, min_bin_duration=fb_bin_params.min_bin_duration, 
            max_bin_duration=fb_bin_params.max_bin_duration, compute_single=fluxbatch_cml_args.single,
            metadata_dir=fluxbatch_cml_args.metadir, cpu_cores=fluxbatch_cml_args.cpucores)


    ### Print camera tally
    print()

    # Save the per-camera tally results
    tally_string = reportCameraTally(fbr, top_n_stations=5)
    # print(tally_string)
    with open(os.path.join(dir_path, fluxbatch_cml_args.output_filename + "_camera_tally.txt"), 'w') as f:
        f.write(tally_string)

    ###

    # Show and save the batch flux plot
    plotBatchFlux(
        fbr, 
        dir_path,
        fluxbatch_cml_args.output_filename, 
        only_flux=fluxbatch_cml_args.onlyflux, 
        compute_single=fluxbatch_cml_args.single
        )

    # Save the results to a CSV file
    data_out_path = saveBatchFluxCSV(fbr, dir_path, fluxbatch_cml_args.output_filename)
    print("Data saved to:", data_out_path)


    # NOTE: CSV option not supported
    # # If a CSV files was given, load the fluxes from the disk
    # else:

    #     # get list of directories so that fixedfluxbin csv files can be found
    #     with open(fluxbatch_cml_args.batch_path) as f:
    #         # Parse the batch entries
    #         for line in f:
    #             line = line.replace("\n", "").replace("\r", "")

    #             if not len(line):
    #                 continue

    #             if line.startswith("#"):
    #                 continue

    #             flux_cml_args = fluxParser().parse_args(shlex.split(line, posix=0))
    #             shower_code = flux_cml_args.shower_code
    #             summary_population_index.append(calculatePopulationIndex(flux_cml_args.s))

    #     # Load data from single-station .csv file and plot it
    #     if fluxbatch_cml_args.single:
    #         dirname = os.path.dirname(fluxbatch_cml_args.batch_path)
    #         data1 = np.genfromtxt(
    #             os.path.join(dirname, fluxbatch_cml_args.output_filename + "_single.csv"),
    #             delimiter=',',
    #             dtype=None,
    #             encoding=None,
    #             skip_header=1,
    #         )

    #         station_list = []
    #         for stationID, sol, flux, lower, upper, _ in data1:
    #             plot_params = plot_info(stationID)

    #             ax[0].errorbar(
    #                 sol,
    #                 flux,
    #                 alpha=0.5,
    #                 capsize=5,
    #                 zorder=3,
    #                 linestyle='none',
    #                 yerr=[[flux - lower], [upper - flux]],
    #                 **plot_params
    #             )

    #     if os.path.exists(os.path.join(dirname, fluxbatch_cml_args.output_filename + "_combined.csv")):
    #         data2 = np.genfromtxt(
    #             os.path.join(dirname, fluxbatch_cml_args.output_filename + "_combined.csv"),
    #             delimiter=',',
    #             encoding=None,
    #             skip_header=1,
    #         )

    #         comb_sol_bins = data2[:, 0]
    #         comb_sol = data2[:-1, 1]
    #         comb_flux = data2[:-1, 2]
    #         comb_flux_lower = data2[:-1, 3]
    #         comb_flux_upper = data2[:-1, 4]
    #         comb_ta_prod = data2[:-1, 5]
    #         comb_num_meteors = data2[:-1, 6]
    #     else:
    #         comb_sol = []
    #         comb_sol_bins = []
    #         comb_flux = []
    #         comb_flux_lower = []
    #         comb_flux_upper = []
    #         comb_num_meteors = []
    #         comb_ta_prod = []


    # Save the single-station fluxes
    if fluxbatch_cml_args.single:

        data_out_path = os.path.join(dir_path, fluxbatch_cml_args.output_filename + "_single.csv")
        with open(data_out_path, 'w') as fout:
            fout.write(
                "# Station, Sol (deg), Flux@+6.5M (met/1000km^2/h), Flux lower bound, Flux upper bound, Population Index\n"
            )
            for entry in fbr.single_station_flux:
                print(entry)
                stationID, sol, flux, lower, upper, population_index = entry

                fout.write(
                    "{:s},{:.8f},{:.3f},{:.3f},{:.3f},{}\n".format(
                        stationID, sol, flux, lower, upper, population_index
                    )
                )
        print("Data saved to:", data_out_path)
