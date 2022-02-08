""" Compute single-station meteor shower flux. """

import argparse
import collections
import copy
import datetime
import glob
import json
import os
import sys
from pathlib import Path

import ephem
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import RMS.Formats.CALSTARS as CALSTARS
import scipy.stats
from git.objects.base import Object
from RMS.Astrometry.ApplyAstrometry import (correctVignettingTrueToApparent,
                                            extinctionCorrectionTrueToApparent,
                                            getFOVSelectionRadius, raDecToXYPP,
                                            xyToRaDecPP)
from RMS.Astrometry.ApplyRecalibrate import (applyRecalibrate,
                                             getRecalibratedPlatepar,
                                             recalibrateSelectedFF)
from RMS.Astrometry.Conversions import (J2000_JD, areaGeoPolygon, date2JD,
                                        datetime2JD, jd2Date, raDec2AltAz)
from RMS.Astrometry.CyFunctions import subsetCatalog
from RMS.ExtractStars import extractStarsAndSave
from RMS.Formats import FFfile, Platepar, StarCatalog
from RMS.Formats.CALSTARS import readCALSTARS
from RMS.Formats.FTPdetectinfo import findFTPdetectinfoFile, readFTPdetectinfo
from RMS.Math import angularSeparation, pointInsideConvexPolygonSphere
from RMS.Routines.FOVArea import fovArea, xyHt2Geo
from RMS.Routines.MaskImage import MaskStructure, getMaskFile, loadMask
from RMS.Routines.SolarLongitude import (jd2SolLonSteyaert, solLon2jdSteyaert,
                                         unwrapSol)

from Utils.ShowerAssociation import heightModel, showerAssociation


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
    """Read raw collection areas from disk."""

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


def saveForcedBinFluxData(
    dir_path, file_name, sol_list, meteor_n_list, area_list, time_list, meter_lm_list, sol_range=None
):
    """Save solar longitude, number of meteors, collecting area and time for fixed bins

    Keyword arguments:
        sol_range: [tuple] lower, upper
            - lower: [float] Minimum solar longitude which the loaded bins must contain
            - upper: [float] Maximum solar longitude which the loaded bins must contain
    """
    if sol_range:
        lower_i = np.searchsorted(sol_list, sol_range[0], side='left')
        upper_i = np.searchsorted(sol_list, sol_range[1], side='right')
        sol_list = sol_list[lower_i:upper_i]

        sl = slice(lower_i, upper_i - 1)
        meteor_n_list, area_list, time_list, meter_lm_list = (
            meteor_n_list[sl],
            area_list[sl],
            time_list[sl],
            meter_lm_list[sl],
        )
    file_path = os.path.join(dir_path, f"{file_name}.csv")
    with open(file_path, 'w') as f:
        f.write('Sol (rad), Meteors, Area (m^2), Time (hours), Meteor LM (mag)\n')
        for sol, meteors, area, time, lm in zip(sol_list, meteor_n_list, area_list, time_list, meter_lm_list):
            f.write(f"{sol},{meteors},{area},{time},{lm}\n")
        f.write(f'{sol_list[-1]},,,,')  # sol_list has one more element thatn meteor_list


def loadForcedBinFluxData(dir_path, filename):
    """Load solar longitude, number of meteors, collecting area and time values for fixed bins

    Keyword arguments:
        sol_range: [tuple] lower, upper
            - lower: [float] Minimum solar longitude which the loaded bins must contain
            - upper: [float] Maximum solar longitude which the loaded bins must contain

    Return:
        [tuple] sol, meteor_list, area_list, time_list, meteor_lm_list
            - sol: [ndarray] Array of solar longitude bin edges (length is one more than other arrays)
            - meteor_list: [ndarray] Number of meteors in bin
            - area_list: [ndarray] Effective collecting area corresponding to bin
            - time_list: [ndarray] Duration of bin (in hours)
            - meteor_lm_list: [ndarray] Meteor limiting magnitude corresponding to bin
    """
    file_path = os.path.join(dir_path, filename)

    data = np.genfromtxt(file_path, delimiter=',', encoding=None, skip_header=1)

    sol = data[:, 0]
    meteor_list = data[:-1, 1]
    area_list = data[:-1, 2]
    time_list = data[:-1, 3]
    meteor_lm_list = data[:-1, 4]

    return sol, meteor_list, area_list, time_list, meteor_lm_list


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

        # Minimum distance of the end of the meteor to the radiant (deg)
        self.rad_dist_min = 15

        # Subdivide the time bin into the given number of subbins
        self.sub_time_bins = 2

        # Minimum number of meteors in the time bin
        self.meteors_min = 3


def computeTimeIntervals(cloud_ratio_dict, ratio_threshold=0.5, time_gap_threshold=15, clearing_threshold=90):
    """
    Calculate sets of time intervals using the detected to predicted star ratios

    Arguments:
        cloud_ratio_dict: [dict] ff_file: ratio
        ratio_threshold: [float] minimum ratio required to be above. 0.15 is reasonable but 0.2 is more safe
        time_gap_threshold: [float] maximum time gap between ff files from clodu_ratio_dict
            before it's used to stop an interval. This is because if there is a gap that's too large, it
            can be because there are clouds in between. And to not risk this affecting flux values
            this is cut out.
        clearing_threshold: [float] The time (in hours) required for the ratio to be above the
            ratio threshold before it will be considered

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

        # make an interval if FF has a >10 min gap (suggests clouds in between) or if the ratio is too low.
        # However the interval must be at least an hour to be kept.
        if (date - prev_date).total_seconds() / 60 > time_gap_threshold or ratio < ratio_threshold:
            if start_interval is not None and (prev_date - start_interval).total_seconds() / 60 > 60:
                intervals.append((start_interval, prev_date))

            # If ratio is less than threshold, you want to discard so it shouldn't be the start of an interval
            if ratio < ratio_threshold:
                start_interval = None
            else:
                start_interval = date

        prev_date = date

    # if you run out of images, that counts as a cutoff
    if start_interval is not None and (prev_date - start_interval).total_seconds() / 60 > clearing_threshold:
        intervals.append((start_interval, prev_date))

    return intervals


def detectMoon(file_list, platepar, config):
    """
    If moon is within 3 degrees of the FOV and the phase of the moon is above 25% then the moon
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
            [platepar.X_res / 2],
            [platepar.Y_res / 2],
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

        # print()
        # print(filename)
        # print("Area:", lunar_area)
        # print(o.next_rising(m) < o.next_setting(m))
        # print("Ang dist:", angular_distance, radius)

        
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
            x, y = raDecToXYPP(np.array([np.degrees(m.ra)]), np.array([np.degrees(m.dec)]),
                               datetime2JD(o.date.datetime()), platepar)

            x = x[0]
            y = y[0]

            # print(x, y)

            # Compute the exclusion border in pixels (always scale to 720p)
            border = 100*platepar.Y_res/720

            if not (((x > -border) and (x < platepar.X_res + border)) \
                and ((y > -border) and (y < platepar.Y_res + border))):

                new_file_list.append(filename)
                continue
        

        print("Skipping {:s}, Moon in the FOV!".format(filename))

    return new_file_list


def detectClouds(config, dir_path, N=5, mask=None, show_plots=True, ratio_threshold=0.5):
    """Detect clouds based on the number of stars detected in images compared to how many are
    predicted.

    Arguments:
        dir_path: [str] folder to search for FF files, CALSTARS files
        platepar: [Platepar object]

    keyword arguments:
        mask: [2d array]
        N: [float] Time duration of bins to separate FF files into
        show_plots: [Bool] Whether to show plots (defaults to true)

    Return:
        time_intervals [list of tuple]: list of datetime pairs in tuples, representing the starting
            and ending times of a time interval
    """
    # collect detected stars
    file_list = sorted(os.listdir(dir_path))

    # Locate and load the mask file
    mask = getMaskFile(dir_path, config, file_list=file_list)

    # get detected stars
    calstars_file = None
    for calstars_file in file_list:
        if ('CALSTARS' in calstars_file) and calstars_file.endswith('.txt'):
            break
    star_list = readCALSTARS(dir_path, calstars_file)
    print('CALSTARS file: ' + calstars_file + ' loaded!')

    # get FF file every N minute interval
    starting_time = None
    recorded_files = []
    bin_used = -1
    for ff_file_name, _ in star_list:
        date = FFfile.filenameToDatetime(ff_file_name)
        if starting_time is None:
            starting_time = date

        # store the first file of each bin
        new_bin = int(((date - starting_time).total_seconds() / 60) // N)
        if new_bin > bin_used:
            recorded_files.append(ff_file_name)
            bin_used = new_bin

    # detect which images don't have a moon visible, and filter the file list based on this
    platepar = Platepar.Platepar()
    platepar.read(os.path.join(dir_path, config.platepar_name), use_flat=config.use_flat)

    recorded_files = detectMoon(recorded_files, platepar, config)

    # Find and load recalibrated platepars
    if config.platepars_flux_recalibrated_name in file_list:
        with open(os.path.join(dir_path, config.platepars_flux_recalibrated_name)) as f:
            recalibrated_platepar_dict = json.load(f)
            # Convert the dictionary of recalibrated platepars to a dictionary of Platepar objects
            recalibrated_platepars = {}
            for ff_name in recalibrated_platepar_dict:
                pp = Platepar.Platepar()
                pp.loadFromDict(recalibrated_platepar_dict[ff_name], use_flat=config.use_flat)

                recalibrated_platepars[ff_name] = pp
                recorded_files = list(recalibrated_platepars.keys())
            print("Recalibrated platepars loaded!")

    # If the file is not available, apply the recalibration procedure
    else:
        print("Recalibrated platepar file not available!")
        print("Recalibrating...")
        recalibrated_platepars = recalibrateSelectedFF(dir_path, recorded_files, star_list, config, \
            stellarLMModel(platepar.mag_lev), ignore_distance_threshold=True)
        recorded_files = list(recalibrated_platepars.keys())

    matched_count = {ff: len(recalibrated_platepars[ff].star_list) for ff in recorded_files}

    # Compute the correction between the visible limiting magnitude and the LM produced by the star detector
    #   - normalize the LM to the intensity threshold of 18
    #   - correct for the sensitivity at intensity threshold of 18 (empirical)
    star_det_mag_corr = -2.5*np.log10(config.intensity_threshold/18) - 1.3

    # Compute the limiting magnitude of the star detector
    ff_limiting_magnitude = {
        ff_file: (
            stellarLMModel(recalibrated_platepars[ff_file].mag_lev) + star_det_mag_corr
            if recalibrated_platepars[ff_file].auto_recalibrated
            else None
        )
        for ff_file in recorded_files
    }

    if show_plots:
        # matched_pred_LM = {
        #     ff: np.percentile(
        #         xyToRaDecPP(
        #             [FFfile.getMiddleTimeFF(ff, config.fps, ret_milliseconds=True)]
        #             * len(star_data.star_list),
        #             [star[1] for star in star_data.star_list],
        #             [star[2] for star in star_data.star_list],
        #             [star[3] for star in star_data.star_list],
        #             platepar,
        #         )[3],
        #         90,
        #     )
        #     for ff, star_data in recalibrated_platepars.items()
        #     if len(star_data.star_list)
        # }

        # Compute the limiting magnitude of matched stars as the 90th percentile of the faintest matched stars
        matched_star_LM = {
            ff: np.percentile(np.array(recalibrated_platepars[ff].star_list)[:, 6], 90)
            for ff in recorded_files
            if len(recalibrated_platepars[ff].star_list)
        }

        empirical_LM = {
            ff_file: (
                stellarLMModel(recalibrated_platepars[ff_file].mag_lev)
                if recalibrated_platepars[ff_file].auto_recalibrated
                else None
            )
            for ff_file in recorded_files
        }

        plot_format = mdates.DateFormatter('%H:%M')

        plt.gca().xaxis.set_major_formatter(plot_format)

        plt.gca().scatter(
            [FFfile.filenameToDatetime(ff) for ff in empirical_LM],
            empirical_LM.values(),
            label='Stellar LM',
            s=5,
            c='k',
        )

        plt.gca().scatter(
            [FFfile.filenameToDatetime(ff) for ff in ff_limiting_magnitude],
            ff_limiting_magnitude.values(),
            marker='+',
            label='Star detection LM',
            c='orange',
        )

        plt.gca().scatter(
            [FFfile.filenameToDatetime(ff) for ff in matched_star_LM],
            matched_star_LM.values(),
            marker='x',
            label="90th percentile detected stars",
            c='green',
        )

        plt.gca().set_ylabel('Magnitude')
        plt.gca().set_xlabel('Time')

        plt.gca().legend()

        plt.show()

    predicted_stars = predictStarNumberInFOV(
        recalibrated_platepars, ff_limiting_magnitude, config, mask, show_plot=show_plots
    )
    # for ff in predicted_stars:
    #     print(ff, matched_count.get(ff), predicted_stars.get(ff), ff_limiting_magnitude.get(ff))

    ratio = {
        ff_file: (matched_count[ff_file] / predicted_stars[ff_file] if ff_file in predicted_stars else 0)
        for ff_file in recorded_files
    }
    time_intervals = computeTimeIntervals(ratio, ratio_threshold=ratio_threshold)

    if show_plots and predicted_stars:
        fig, ax = plt.subplots(2, sharex=True)
        plot_format = mdates.DateFormatter('%H:%M')
        ax[0].xaxis.set_major_formatter(plot_format)
        ax[0].plot([FFfile.filenameToDatetime(x) for x in ratio.keys()], list(ratio.values()), marker='o')
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("stars observed at LM-1/stars predicted")
        ax[0].vlines(
            np.array(time_intervals).flatten(),
            ymin=min(ratio.values()),
            ymax=max(ratio.values()),
            linestyles='dashed',
            colors='r',
        )

        ax[1].xaxis.set_major_formatter(plot_format)
        ax[1].scatter(
            [FFfile.filenameToDatetime(ff) for ff in matched_count],
            [matched_count[ff] for ff in matched_count],
            label='fitted count',
        )
        ax[1].scatter(
            [FFfile.filenameToDatetime(ff) for ff in predicted_stars],
            [predicted_stars[ff] for ff in predicted_stars],
            label='predicted count',
        )
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Count")
        ax[1].legend()
        ax[1].vlines(
            np.array(time_intervals).flatten(),
            ymin=min(predicted_stars.values()),
            ymax=max(predicted_stars.values()),
            linestyles='dashed',
            colors='r',
        )
        plt.show()

    # calculating the ratio of observed starts to the number of predicted stars
    return time_intervals


def predictStarNumberInFOV(recalibrated_platepars, ff_limiting_magnitude, config, mask=None, show_plot=True):
    """Predicts the number of stars that should be in the FOV, considering limiting magnitude,
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

        # using a blank mask if nothing is given
        if mask is None:
            mask = MaskStructure(
                np.full(
                    (recalibrated_platepars[ff_files[0]].Y_res, recalibrated_platepars[ff_files[0]].X_res),
                    255,
                    dtype=np.uint8,
                )
            )

        star_mag = {}
        for i, ff_file in enumerate(ff_files):
            platepar = recalibrated_platepars[ff_file]
            lim_mag = ff_limiting_magnitude[ff_file]
            if lim_mag is None:
                continue

            date = FFfile.getMiddleTimeFF(ff_file, config.fps, ret_milliseconds=True)
            jd = date2JD(*date)

            # make a polygon on a sphere out of 5 points on each side
            # n_points = 5
            # y_points = [platepar.Y_res * i/n_points for i in range(n_points)] + [platepar.Y_res]*n_points + \
            #             [platepar.Y_res * (1-i/n_points) for i in range(n_points)] + [0]*n_points
            # x_points = [0]*n_points + [platepar.X_res * i/n_points for i in range(n_points)] + \
            #             [platepar.X_res]*n_points + [platepar.X_res * (1-i/n_points) for i in range(n_points)]
            _, ra_vertices, dec_vertices, _ = xyToRaDecPP(
                [date] * 4,
                [0, 0, platepar.X_res, platepar.X_res],
                [0, platepar.Y_res, platepar.Y_res, 0],
                [1] * 4,
                platepar,
                extinction_correction=False,
            )

            # collect and filter catalog stars
            catalog_stars, _, _ = StarCatalog.readStarCatalog(
                config.star_catalog_path,
                config.star_catalog_file,
                lim_mag=lim_mag,
                mag_band_ratios=config.star_catalog_band_ratios,
            )

            # filter out stars that are outside of the polygon on the sphere made by the fov
            ra_catalog, dec_catalog, mag = catalog_stars.T
            inside = pointInsideConvexPolygonSphere(
                np.array([ra_catalog, dec_catalog]).T, np.array([ra_vertices, dec_vertices]).T
            )
            x, y = raDecToXYPP(ra_catalog, dec_catalog, jd, platepar)
            x = x[inside]
            y = y[inside]
            mag = mag[inside]
            # correct for extinction and vignetting so that dim stars can be filtered
            # (not necessary since limiting magnitude already matches with matched star LM)
            # mag = extinctionCorrectionTrueToApparent(mag[inside], ra_catalog[inside], dec_catalog[inside], jd, platepar)
            # mag = correctVignettingTrueToApparent(mag, x, y, platepar)
            # filter coordinates to be in FOV and make sure that the stars that are too dim are filtered
            bounds = (mag <= lim_mag) & (y >= 0) & (y < platepar.Y_res) & (x >= 0) & (x < platepar.X_res)
            x = x[bounds]
            y = y[bounds]
            mag = mag[bounds]

            # filter stars with mask
            mask_filter = np.take(
                np.floor(mask.img / 255),
                np.ravel_multi_index(
                    np.floor(np.array([y, x])).astype(int), (platepar.Y_res, platepar.X_res)
                ),
            ).astype(bool)

            if show_plot and i == int(len(ff_files) // 2):
                plt.title(f"{ff_file}, lim_mag={lim_mag:.2f}")
                plt.scatter(
                    *np.array(recalibrated_platepars[ff_file].star_list)[:, 1:3].T[::-1], label='matched'
                )
                plt.scatter(x[mask_filter], y[mask_filter], c='r', marker='+', label='catalog')
                plt.legend()
                plt.show()

                # print(np.sum(mask_filter))
                # print(val[inside][bounds][mask_filter])
                # plt.scatter(x[mask_filter], y[mask_filter], c=mag[inside & (mag <= lim_mag)][bounds][mask_filter])
                # plt.show()
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
        mask = MaskStructure(np.full((platepar.Y_res, platepar.X_res), 255, dtype=np.uint8))

    # Compute the number of samples for every image axis
    longer_side_points = side_points
    shorter_side_points = int(np.ceil(side_points * platepar.Y_res / platepar.X_res))

    # Compute pixel delta for every side
    longer_dpx = int(platepar.X_res // longer_side_points)
    shorter_dpx = int(platepar.Y_res // shorter_side_points)

    # Distionary of collection areas per height
    col_areas_ht = collections.OrderedDict()

    # Estimate the collection area for a given range of heights
    for ht in np.arange(ht_min, ht_max + dht, dht):

        # Convert the height to meters
        ht = 1000 * ht

        print(ht / 1000, "km")

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

                # Compute ratio of masked portion of the segment
                mask_segment = mask.img[y0:ye, x0:xe]
                unmasked_ratio = 1 - np.count_nonzero(~mask_segment) / mask_segment.size

                # Compute the pointing direction and the vignetting and extinction loss for the mean location

                x_mean = (x0 + xe) / 2
                y_mean = (y0 + ye) / 2

                # Use a test pixel sum
                test_px_sum = 400

                # Compute the pointing direction and magnitude corrected for vignetting and extinction
                _, ra, dec, mag = xyToRaDecPP(
                    [jd2Date(J2000_JD.days)], [x_mean], [y_mean], [test_px_sum], platepar
                )
                azim, elev = raDec2AltAz(ra[0], dec[0], J2000_JD.days, platepar.lat, platepar.lon)

                # Compute the pixel sum back assuming no corrections
                rev_level = 10 ** ((mag[0] - platepar.mag_lev) / (-2.5))

                # Compute the sensitivty loss due to vignetting and extinction
                sensitivity_ratio = test_px_sum / rev_level

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

        print("SUM:", total_area / 1e6, "km^2")

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

        print("DIR:", areaGeoPolygon(lats, lons, ht) / 1e6)

    return col_areas_ht


def sensorCharacterization(config, dir_path, meteor_data, default_fwhm=None):
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
                        raise Exception(
                            'CALSTARS file does not have fwhm and FF files do not exist in'
                            'directory. You must give a fwhm value with "--fwhm 3"'
                        )

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
    # Make a file name to save the raw collection areas
    col_areas_file_name = generateColAreaJSONFileName(
        config.stationID,
        flux_config.side_points,
        flux_config.ht_min,
        flux_config.ht_max,
        flux_config.dht,
        flux_config.elev_limit,
    )

    # Check if the collection area file exists. If yes, load the data. If not, generate collection areas
    if col_areas_file_name in os.listdir(dir_path):
        col_areas_ht = loadRawCollectionAreas(dir_path, col_areas_file_name)
        print("Loaded collection areas from:", col_areas_file_name)

    else:

        # Compute the collecting areas segments per height
        col_areas_ht = collectingArea(
            platepar,
            mask=mask,
            side_points=flux_config.side_points,
            ht_min=flux_config.ht_min,
            ht_max=flux_config.ht_max,
            dht=flux_config.dht,
            elev_limit=flux_config.elev_limit,
        )

        # Save the collection areas to file
        saveRawCollectionAreas(dir_path, col_areas_file_name, col_areas_ht)

        print("Saved raw collection areas to:", col_areas_file_name)

    ### ###

    # Compute the raw collection area at the height of 100 km
    col_area_100km_raw = 0
    col_areas_100km_blocks = col_areas_ht[100000.0]
    for block in col_areas_100km_blocks:
        col_area_100km_raw += col_areas_100km_blocks[block][0]

    print("Raw collection area at height of 100 km: {:.2f} km^2".format(col_area_100km_raw / 1e6))

    return col_areas_ht, col_area_100km_raw


def getSensorCharacterization(dir_path, config, meteor_data, default_fwhm=None):
    # File which stores the sensor characterization profile
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
        sensor_data = sensorCharacterization(config, dir_path, meteor_data, default_fwhm=default_fwhm)

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

    # lm_s = 0.639*p0 - 0.858 # old with 3 points
    lm_s = 0.832 * p0 - 2.585  # new on Nov 4, with 17 points

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
    platepar,
    frame_min_loss,
    ang_vel_night_mid,
    sensor_data,
    lm_m_nightly_mean,
    confidence_interval=0.95,
    binduration=None,
    print_info=True,
    no_skip=False,
):
    """

    Keyword arguments:
        print_info: [bool] Whether to print info as function is running
        no_skip: [bool]

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

    for ((bin_meteor_list, bin_ffs), (bin_dt_beg, bin_dt_end)) in zip(bin_meteor_information, bin_intervals):
        bin_jd_beg = datetime2JD(bin_dt_beg)
        bin_jd_end = datetime2JD(bin_dt_end)

        jd_mean = (bin_jd_beg + bin_jd_end) / 2

        # Compute the mean solar longitude
        sol_mean = np.degrees(jd2SolLonSteyaert(jd_mean))

        ### Compute the radiant elevation at the middle of the time bin ###
        ra, dec, v_init = shower.computeApparentRadiant(platepar.lat, platepar.lon, jd_mean)
        radiant_azim, radiant_elev = raDec2AltAz(ra, dec, jd_mean, platepar.lat, platepar.lon)

        # Compute the mean meteor height
        meteor_ht_beg = heightModel(v_init, ht_type='beg')
        meteor_ht_end = heightModel(v_init, ht_type='end')
        meteor_ht = (meteor_ht_beg + meteor_ht_end) / 2

        # Compute the standard deviation of the height
        meteor_ht_std = meteor_ht * ht_std_percent / 100.0

        # Init the Gaussian height distribution
        meteor_ht_gauss = scipy.stats.norm(meteor_ht, meteor_ht_std)

        if print_info:
            print()
            print()
            print("-- Bin information ---")
            print("Bin beg:", bin_dt_beg)
            print("Bin end:", bin_dt_end)
            print("Sol mid: {:.5f}".format(sol_mean))
            print("Radiant elevation: {:.2f} deg".format(radiant_elev))
            print("Apparent speed: {:.2f} km/s".format(v_init / 1000))

        if not bin_ffs:
            if print_info:
                print("!!! Bin doesn't have any meteors!")
            if no_skip:
                meteor_num_data.append(0)
                lm_m_data.append(None)
                effective_collection_area_data.append(0)
            continue

        # If the elevation of the radiant is below the limit, skip this bin
        if radiant_elev < flux_config.rad_elev_limit:
            if print_info:
                print(
                    "!!! Mean radiant elevation below {:.2f} deg threshold, skipping time bin!".format(
                        flux_config.rad_elev_limit
                    )
                )

            if no_skip:
                meteor_num_data.append(0)
                lm_m_data.append(None)
                effective_collection_area_data.append(0)
            continue

        bin_hours = (bin_dt_end - bin_dt_beg).total_seconds() / 3600
        # The minimum duration of the time bin should be larger than 50% of the given dt
        if binduration is not None and bin_hours < 0.5 * binduration:
            if print_info:
                print(
                    f"!!! Time bin duration of {bin_hours:.2f} h is shorter than 0.5x of the inputted time bin!"
                )
            if no_skip:
                meteor_num_data.append(0)
                lm_m_data.append(None)
                effective_collection_area_data.append(0)
            continue

        if no_skip or len(bin_meteor_list) >= flux_config.meteors_min:
            if print_info:
                print("Meteors:", len(bin_meteor_list))

            ### Weight collection area by meteor height distribution ###

            # Determine weights for each height
            weight_sum = 0
            weights = {}
            for ht in col_areas_ht:
                wt = meteor_ht_gauss.pdf(float(ht))
                weight_sum += wt
                weights[ht] = wt

            # Normalize the weights so that the sum is 1
            for ht in weights:
                weights[ht] /= weight_sum

            ### ###

            col_area_meteor_ht_raw = 0
            for ht in col_areas_ht:
                for block in col_areas_ht[ht]:
                    col_area_meteor_ht_raw += weights[ht] * col_areas_ht[ht][block][0]

            if print_info:
                print(
                    "Raw collection area at meteor heights: {:.2f} km^2".format(col_area_meteor_ht_raw / 1e6)
                )

            # Compute the angular velocity in the middle of the FOV
            rad_dist_mid = angularSeparation(
                np.radians(radiant_azim), np.radians(radiant_elev), np.radians(azim_mid), np.radians(elev_mid)
            )
            ang_vel_mid = v_init * np.sin(rad_dist_mid) / r_mid

            # Skip time bin if the radiant is very close to the centre of the image
            if np.degrees(rad_dist_mid) < flux_config.rad_dist_min:
                if print_info:
                    print(
                        "!!! Radiant too close to the centre of the image! {:.2f} < {:.2f}".format(
                            np.degrees(rad_dist_mid), flux_config.rad_dist_min
                        )
                    )
                if no_skip:
                    meteor_num_data.append(0)
                    lm_m_data.append(None)
                    effective_collection_area_data.append(0)
                continue

            ### Compute the limiting magnitude ###

            # Compute the mean star FWHM in the given bin
            fwhm_bin_mean = np.mean([sensor_data[ff_name][0] for ff_name in bin_ffs])

            # Compute the mean photometric zero point in the given bin
            mag_lev_bin_mean = np.mean(
                [
                    recalibrated_platepars[ff_name].mag_lev
                    for ff_name in bin_ffs
                    if ff_name in recalibrated_platepars
                ]
            )

            # # Standard deviation of star PSF, nightly mean (px)
            # star_stddev = fwhm_bin_mean/2.355

            # Use empirical LM calculation
            lm_s = stellarLMModel(mag_lev_bin_mean)

            lm_s += frame_min_loss

            # ### TEST !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11

            # # Artificialy increase limiting magnitude
            # lm_s += 1.2

            # #####

            # Compute apparent meteor magnitude
            lm_m = (
                lm_s
                - 5 * np.log10(r_mid / 1e5)
                - 2.5
                * np.log10(
                    np.degrees(
                        platepar.F_scale
                        * v_init
                        * np.sin(rad_dist_mid)
                        / (config.fps * r_mid * fwhm_bin_mean)
                    )
                )
            )

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
                for img_coords in col_areas_ht[ht]:
                    # Unpack precomputed values
                    area, azim, elev, sensitivity_ratio, r = col_areas_ht[ht][img_coords]

                    # Compute the angular velocity (rad/s) in the middle of this block
                    rad_dist = angularSeparation(
                        np.radians(radiant_azim), np.radians(radiant_elev), np.radians(azim), np.radians(elev)
                    )
                    ang_vel = v_init * np.sin(rad_dist) / r

                    # If the angular distance from the radiant is less than 15 deg, don't use the block
                    #   in the effective collection area
                    if np.degrees(rad_dist) < flux_config.rad_dist_min:
                        area = 0.0

                    # Compute the range correction
                    range_correction = (1e5 / r) ** 2

                    # ang_vel_correction = ang_vel/ang_vel_mid
                    # Compute angular velocity correction relative to the nightly mean
                    ang_vel_correction = ang_vel / ang_vel_night_mid

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
                    radiant_elev_correction = np.sin(np.radians(radiant_elev))
                    radiant_elev_correction = max(radiant_elev_correction, 0.1)
                    radiant_elev_corr_arr.append(radiant_elev_correction)
                    correction_ratio *= radiant_elev_correction

                    # Correct for angular velocity (cap to an order of magnitude correction)
                    ang_vel_correction = min(max(ang_vel_correction, 0.1), 10)
                    correction_ratio *= ang_vel_correction
                    ang_vel_corr_arr.append(ang_vel_correction)

                    # Add the collection area to the final estimate with the height weight
                    #   Raise the correction to the mass index power
                    total_correction = correction_ratio ** (mass_index - 1)
                    total_correction = min(max(total_correction, 0.1), 10)
                    collection_area += weights[ht] * area * total_correction
                    total_corr_arr.append(total_correction)

                    col_area_raw_arr.append(weights[ht] * area)
                    col_area_eff_arr.append(weights[ht] * area * total_correction)

                    if img_coords not in col_area_eff_block_dict:
                        col_area_eff_block_dict[img_coords] = []

                    col_area_eff_block_dict[img_coords].append(weights[ht] * area * total_correction)

            # Compute mean corrections
            sensitivity_corr_avg = np.mean(sensitivity_corr_arr)
            range_corr_avg = np.mean(range_corr_arr)
            radiant_elev_corr_avg = np.mean(radiant_elev_corr_arr)
            ang_vel_corr_avg = np.mean(ang_vel_corr_arr)
            total_corr_avg = np.median(total_corr_arr)
            col_area_raw_sum = np.sum(col_area_raw_arr)
            col_area_eff_sum = np.sum(col_area_eff_arr)

            if print_info:
                print(
                    "Raw collection area at meteor heights (CHECK): {:.2f} km^2".format(
                        col_area_raw_sum / 1e6
                    )
                )
                print(
                    "Eff collection area at meteor heights (CHECK): {:.2f} km^2".format(
                        col_area_eff_sum / 1e6
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
            collection_area_lm_nightly = collection_area / population_index ** (lm_m_nightly_mean - lm_m)
            collection_area_lm_6_5 = collection_area / population_index ** (6.5 - lm_m)

            flux = 1e9 * len(bin_meteor_list) / collection_area / bin_hours
            flux_lm_nightly_mean = 1e9 * len(bin_meteor_list) / collection_area_lm_nightly / bin_hours
            flux_lm_6_5 = 1e9 * len(bin_meteor_list) / collection_area_lm_6_5 / bin_hours

            # Compute confidence interval of the flux
            ci = 1.0 - confidence_interval
            num_ci_lower = scipy.stats.chi2.ppf(ci / 2, 2 * len(bin_meteor_list)) / 2
            num_ci_upper = scipy.stats.chi2.ppf(1 - ci / 2, 2 * (len(bin_meteor_list) + 1)) / 2
            flux_lm_6_5_ci_lower = 1e9 * num_ci_lower / collection_area_lm_6_5 / bin_hours
            flux_lm_6_5_ci_upper = 1e9 * num_ci_upper / collection_area_lm_6_5 / bin_hours

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

            if print_info:
                print("-- Sensor information ---")
                print("Star FWHM:  {:5.2f} px".format(fwhm_bin_mean))
                print("Photom ZP:  {:+6.2f} mag".format(mag_lev_bin_mean))
                print("Stellar LM: {:+.2f} mag".format(lm_s))
                print("-- Flux ---")
                print(
                    "Meteors:  {:d}, {:.0f}% CI [{:.2f}, {:.2f}]".format(
                        len(bin_meteor_list), 100 * confidence_interval, num_ci_lower, num_ci_upper
                    )
                )
                print("Col area: {:d} km^2".format(int(collection_area / 1e6)))
                print("Ang vel:  {:.2f} deg/s".format(np.degrees(ang_vel_mid)))
                print("LM app:   {:+.2f} mag".format(lm_m))
                print("Flux:     {:.2f} meteors/1000km^2/h".format(flux))
                print("to {:+.2f}: {:.2f} meteors/1000km^2/h".format(lm_m_nightly_mean, flux_lm_nightly_mean))
                print(
                    "to +6.50: {:.2f}, {:.0f}% CI [{:.2f}, {:.2f}] meteors/1000km^2/h".format(
                        flux_lm_6_5, 100 * confidence_interval, flux_lm_6_5_ci_lower, flux_lm_6_5_ci_upper
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

        elif print_info:
            print(
                f'!!! Insufficient meteors in bin: {len(bin_meteor_list)} observed vs min {flux_config.meteors_min}'
            )

    if no_skip:
        return meteor_num_data, effective_collection_area_data, lm_m_data
    return (
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


def computeFlux(
    config,
    dir_path,
    ftpdetectinfo_path,
    shower_code,
    dt_beg,
    dt_end,
    mass_index,
    binduration=None,
    binmeteors=None,
    timebin_intdt=0.25,
    ht_std_percent=5.0,
    mask=None,
    show_plots=True,
    confidence_interval=0.95,
    default_fwhm=None,
    forced_bins=None,
):
    """Compute flux using measurements in the given FTPdetectinfo file.

    Arguments:
        config: [Config instance]
        dir_path: [str] Path to the working directory.
        ftpdetectinfo_path: [str] Path to a FTPdetectinfo file.
        shower_code: [str] IAU shower code (e.g. ETA, PER, SDA).
        dt_beg: [Datetime] Datetime object of the observation beginning.
        dt_end: [Datetime] Datetime object of the observation end.
        mass_index: [float] Cumulative mass index of the shower.

    Keyword arguments:
        binduration: [float] Time in hours for each bin
        binmeteors: [int] Number of meteors to have in each bin (cannot have both this and binduration)
        timebin_intdt: [float] Time step for computing the integrated collection area in hours. 15 minutes by
            default. If smaller than that, only one collection are will be computed.
        ht_std_percent: [float] Meteor height standard deviation in percent.
        mask: [Mask object] Mask object, None by default.
        show_plots: [bool] Show flux plots. True by default.
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

    # Get a list of files in the night folder
    file_list = sorted(os.listdir(dir_path))

    # Load meteor data from the FTPdetectinfo file
    meteor_data = readFTPdetectinfo(*os.path.split(ftpdetectinfo_path))

    if not len(meteor_data):
        print("No meteors in the FTPdetectinfo file!")
        return None

    platepar, recalibrated_platepars = getRecalibratedPlatepar(dir_path, config, file_list)

    # Compute nighly mean of the photometric zero point
    mag_lev_nightly_mean = np.mean(
        [recalibrated_platepars[ff_name].mag_lev for ff_name in recalibrated_platepars]
    )

    # Locate and load the mask file
    mask = getMaskFile(dir_path, config, file_list=file_list)

    # Compute the population index using the classical equation
    # Found to be more consistent when comparing fluxes
    population_index = 10 ** ((mass_index - 1) / 2.5)
    # population_index = 10**((mass_index - 1)/2.3) # TEST !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

    ### SENSOR CHARACTERIZATION ###
    # Computes FWHM of stars and noise profile of the sensor
    sensor_data = getSensorCharacterization(dir_path, config, meteor_data, default_fwhm=default_fwhm)

    # Compute the nighly mean FWHM
    fwhm_nightly_mean = np.mean([sensor_data[key][0] for key in sensor_data])

    ### ###

    # Perform shower association
    associations, _ = showerAssociation(
        config,
        [ftpdetectinfo_path],
        shower_code=shower_code,
        show_plot=False,
        save_plot=False,
        plot_activity=False,
    )

    # Init the flux configuration
    flux_config = FluxConfig()

    # Remove all meteors which begin below the limit height
    filtered_associations = {}
    for key in associations:
        meteor, shower = associations[key]

        if meteor.beg_alt > flux_config.elev_limit:
            print("Rejecting:", meteor.jdt_ref)
            filtered_associations[key] = (meteor, shower)

    associations = filtered_associations

    # If there are no shower association, return nothing
    if not associations:
        print("No meteors associated with the shower!")
        return None

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
        [platepar.X_res / 2],
        [platepar.Y_res / 2],
        [1],
        platepar,
        extinction_correction=False,
    )
    azim_mid, elev_mid = raDec2AltAz(ra_mid[0], dec_mid[0], J2000_JD.days, platepar.lat, platepar.lon)

    # Compute the range to the middle point
    ref_ht = 100000
    r_mid, _, _, _ = xyHt2Geo(
        platepar,
        platepar.X_res / 2,
        platepar.Y_res / 2,
        ref_ht,
        indicate_limit=True,
        elev_limit=flux_config.elev_limit,
    )

    print("Range at 100 km in the middle of the image: {:.2f} km".format(r_mid / 1000))

    # Compute the average angular velocity to which the flux variation throught the night will be normalized
    #   The ang vel is of the middle of the FOV in the middle of observations

    # Middle Julian date of the night
    jd_night_mid = (datetime2JD(dt_beg) + datetime2JD(dt_end)) / 2

    # Compute the apparent radiant
    ra, dec, v_init = shower.computeApparentRadiant(platepar.lat, platepar.lon, jd_night_mid)

    # Compute the radiant elevation
    radiant_azim, radiant_elev = raDec2AltAz(ra, dec, jd_night_mid, platepar.lat, platepar.lon)

    # Compute the angular velocity in the middle of the FOV
    rad_dist_night_mid = angularSeparation(
        np.radians(radiant_azim), np.radians(radiant_elev), np.radians(azim_mid), np.radians(elev_mid)
    )
    ang_vel_night_mid = v_init * np.sin(rad_dist_night_mid) / r_mid

    ###

    # Compute the average limiting magnitude to which all flux will be normalized

    # Compute the theoretical stellar limiting magnitude using an empirical model (nightly average)
    lm_s_nightly_mean = stellarLMModel(mag_lev_nightly_mean)

    # A meteor needs to be visible on at least 4 frames, thus it needs to have at least 4x the mass to produce
    #   that amount of light. 1 magnitude difference scales as -0.4 of log of mass, thus:
    # frame_min_loss = np.log10(config.line_minimum_frame_range_det)/(-0.4)
    frame_min_loss = 0.0  # TEST !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11

    print("Frame min loss: {:.2} mag".format(frame_min_loss))

    lm_s_nightly_mean += frame_min_loss

    # Compute apparent meteor magnitude
    lm_m_nightly_mean = (
        lm_s_nightly_mean
        - 5 * np.log10(r_mid / 1e5)
        - 2.5
        * np.log10(
            np.degrees(
                platepar.F_scale
                * v_init
                * np.sin(rad_dist_night_mid)
                / (config.fps * r_mid * fwhm_nightly_mean)
            )
        )
    )

    print("Stellar lim mag using detection thresholds:", lm_s_nightly_mean)
    print("Apparent meteor limiting magnitude:", lm_m_nightly_mean)

    ### Apply time-dependent corrections ###

    # Go through all time bins within the observation period
    bin_meteor_information = []  # [[[meteor, ...], [meteor.ff_name, ...]], ...]
    bin_intervals = []  # [(start_time, end_time), ...]
    num_meteors = 0

    # if using fixed bins, generate them
    if binduration is not None:
        curr_bin_start = dt_beg
        dt = datetime.timedelta(hours=(binduration / flux_config.sub_time_bins))
        while curr_bin_start < dt_end:
            bin_intervals.append(
                (curr_bin_start, min(curr_bin_start + flux_config.sub_time_bins * dt, dt_end))
            )
            bin_meteor_information.append([[], []])
            curr_bin_start += dt

    # generating forced bins to fill
    loaded_forced_bins = False
    if forced_bins:
        bin_datetime, sol_bins = forced_bins
        sol_bins = np.array(sol_bins)
        starting_sol = unwrapSol(jd2SolLonSteyaert(datetime2JD(dt_beg)), sol_bins[0], sol_bins[-1])
        ending_sol = unwrapSol(jd2SolLonSteyaert(datetime2JD(dt_end)), sol_bins[0], sol_bins[-1])

        # if you can load data from a file, use those bins
        if os.path.exists(
            os.path.join(
                dir_path, f'fixedbinsflux_{config.stationID}_{starting_sol:.5f}_{ending_sol:.5f}.csv'
            )
        ):
            loaded_forced_bins = True
            (
                forced_bins_sol,
                _forced_bins_meteor_num,
                _forced_bins_area,
                _forced_bins_time,
                _forced_bins_lm_m,
            ) = loadForcedBinFluxData(
                dir_path, f'fixedbinsflux_{config.stationID}_{starting_sol:.5f}_{ending_sol:.5f}.csv'
            )
            # if sol_bins wraps would wrap around but forced_bins_sol doesn't
            if sol_bins[0] > forced_bins_sol[0]:
                i = np.argmax(sol_bins - (forced_bins_sol[0] + 360) > -1e-7)
            else:
                i = np.argmax(sol_bins - forced_bins_sol[0] > -1e-7)  # index where they are equal

            forced_bins_meteor_num = np.zeros(len(sol_bins) - 1)
            forced_bins_meteor_num[i : i + len(_forced_bins_meteor_num)] = _forced_bins_meteor_num
            forced_bins_area = np.zeros(len(sol_bins) - 1)
            forced_bins_area[i : i + len(_forced_bins_area)] = _forced_bins_area
            forced_bins_time = np.zeros(len(sol_bins) - 1)
            forced_bins_time[i : i + len(_forced_bins_time)] = _forced_bins_time
            forced_bins_lm_m = np.zeros(len(sol_bins) - 1)
            forced_bins_lm_m[i : i + len(_forced_bins_lm_m)] = _forced_bins_lm_m

        else:
            forced_bins_time = np.array(
                [
                    min(
                        max(min(sol - starting_sol, ending_sol - sol) / (sol_bins[i + 1] - sol_bins[i]), 0), 1
                    )
                    * (bin_datetime[i + 1] - bin_datetime[i]).total_seconds()
                    / 3600
                    for i, sol in enumerate(sol_bins[:-1])
                ]
            )

            forced_bin_meteor_information = [[[], []] for _ in sol_bins[:-1]]
            forced_bin_intervals = [
                (bin_datetime[i], bin_datetime[i + 1]) for i in range(len(bin_datetime) - 1)
            ]

    # mapping meteors to bins
    for meteor, shower in associations.values():
        meteor_date = jd2Date(meteor.jdt_ref, dt_obj=True)

        # Filter out meteors ending too close to the radiant
        ra, dec, _ = shower.computeApparentRadiant(platepar.lat, platepar.lon, meteor.jdt_ref)
        radiant_azim, radiant_elev = raDec2AltAz(ra, dec, meteor.jdt_ref, platepar.lat, platepar.lon)
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

        # finding how to put the meteor in a bin
        if binduration is not None:
            bin_num = min(
                int(
                    ((meteor_date - dt_beg).total_seconds() / 3600) / binduration * flux_config.sub_time_bins
                ),
                len(bin_intervals) - 1,
            )
            for i in range(min(flux_config.sub_time_bins, bin_num)):
                bin_meteor_information[bin_num - i][0].append(meteor)
                bin_meteor_information[bin_num - i][1].append(meteor.ff_name)

        else:  # meteor count
            if ((num_meteors - 1) * flux_config.sub_time_bins) % binmeteors < flux_config.sub_time_bins:
                for i in range(min(flux_config.sub_time_bins - 1, len(bin_meteor_information))):
                    bin_meteor_information[-i - 1][0].append(meteor)
                    bin_meteor_information[-i - 1][1].append(meteor.ff_name)

                bin_meteor_information.append([[meteor], [meteor.ff_name]])

                if len(bin_intervals) == 0:
                    bin_intervals.append([dt_beg])
                else:
                    bin_intervals[-1].append(meteor_date)
                    bin_intervals.append([meteor_date])

            else:
                for i in range(min(flux_config.sub_time_bins, len(bin_meteor_information))):
                    bin_meteor_information[-i - 1][0].append(meteor)
                    bin_meteor_information[-i - 1][1].append(meteor.ff_name)

    # closing off all bin intervals
    if binmeteors is not None:
        for _bin in bin_intervals[-flux_config.sub_time_bins :]:
            if len(_bin) == 1:
                _bin.append(dt_end)

    ### ###
    (
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
        platepar,
        frame_min_loss,
        ang_vel_night_mid,
        sensor_data,
        lm_m_nightly_mean,
        confidence_interval=confidence_interval,
        binduration=binduration,
    )

    if forced_bins:
        if not loaded_forced_bins:
            print('Calculating collecting area for fixed bins')
            forced_bins_meteor_num, forced_bins_area, forced_bins_lm_m = computeFluxCorrectionsOnBins(
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
                platepar,
                frame_min_loss,
                ang_vel_night_mid,
                sensor_data,
                lm_m_nightly_mean,
                confidence_interval=confidence_interval,
                no_skip=True,
            )
            print('Finished computing collecting areas for fixed bins')
            saveForcedBinFluxData(
                dir_path,
                f'fixedbinsflux_{config.stationID}_{starting_sol:.5f}_{ending_sol:.5f}',
                sol_bins,
                forced_bins_meteor_num,
                forced_bins_area,
                forced_bins_time,
                forced_bins_lm_m,
                sol_range=(starting_sol, ending_sol),
            )

        forced_bins_area = np.array(
            [
                area / population_index ** (6.5 - lm) if lm is not None and not np.isnan(lm) else 0
                for area, lm in zip(forced_bins_area, forced_bins_lm_m)
            ]
        )

    #######################################

    # Print the results
    print("Solar longitude, Flux at LM +6.5:")
    for sol, flux_lm_6_5 in zip(sol_data, flux_lm_6_5_data):
        print("{:9.5f}, {:8.4f}".format(sol, flux_lm_6_5))

    if show_plots and len(sol_data):

        # Plot a histogram of peak magnitudes
        plt.hist(peak_mags, cumulative=True, log=True, bins=len(peak_mags), density=True)

        # Plot population index
        r_intercept = -0.7
        x_arr = np.linspace(np.min(peak_mags), np.percentile(peak_mags, 60))
        plt.plot(x_arr, 10 ** (np.log10(population_index) * x_arr + r_intercept))

        plt.title("r = {:.2f}".format(population_index))

        plt.show()

        # Plot how the derived values change throughout the night
        fig, axes = plt.subplots(nrows=4, ncols=2, sharex=True, figsize=(10, 8))

        ((ax_met, ax_lm), (ax_rad_elev, ax_corrs), (ax_rad_dist, ax_col_area), (ax_ang_vel, ax_flux)) = axes

        fig.suptitle("{:s}, s = {:.2f}, r = {:.2f}".format(shower_code, mass_index, population_index))

        ax_met.scatter(sol_data, meteor_num_data)
        ax_met.set_ylabel("Meteors")

        ax_rad_elev.plot(sol_data, radiant_elev_data)
        ax_rad_elev.set_ylabel("Radiant elev (deg)")

        ax_rad_dist.plot(sol_data, radiant_dist_mid_data)
        ax_rad_dist.set_ylabel("Radiant dist (deg)")

        ax_ang_vel.plot(sol_data, ang_vel_mid_data)
        ax_ang_vel.set_ylabel("Ang vel (deg/s)")
        ax_ang_vel.set_xlabel("La Sun (deg)")

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

        ax_col_area.plot(sol_data, np.array(effective_collection_area_data) / 1e6)
        ax_col_area.plot(
            sol_data, len(sol_data) * [col_area_100km_raw / 1e6], color='k', label="Raw col area at 100 km"
        )
        ax_col_area.plot(
            sol_data,
            len(sol_data) * [col_area_meteor_ht_raw / 1e6],
            color='k',
            linestyle='dashed',
            label="Raw col area at met ht",
        )
        ax_col_area.set_ylabel("Eff. col. area (km^2)")
        ax_col_area.legend()

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

        ax_flux.set_ylabel("Flux@+6.5M (met/1000km^2/h)")
        ax_flux.set_xlabel("La Sun (deg)")

        plt.tight_layout()

        plt.show()

    if forced_bins:
        return (
            sol_data,
            flux_lm_6_5_data,
            flux_lm_6_5_ci_lower_data,
            flux_lm_6_5_ci_upper_data,
            meteor_num_data,
            population_index,
            (forced_bins_meteor_num, forced_bins_area, forced_bins_time),
        )
    return (
        sol_data,
        flux_lm_6_5_data,
        flux_lm_6_5_ci_lower_data,
        flux_lm_6_5_ci_upper_data,
        meteor_num_data,
        population_index,
    )


def prepareFluxFiles(
    config,
    dir_path,
    ftpdetectinfo_path,
    shower_code,
    dt_beg,
    dt_end,
    mass_index,
    forced_bins,
    ht_std_percent=5.0,
    mask=None,
    confidence_interval=0.95,
    default_fwhm=None,
):
    """
    Computes fluxes for 5 minute bins and saves them to files. This is similar to computeFlux, except
    doesn't return anything and doesn't make bins for the specific dataset.
    """

    file_list = sorted(os.listdir(dir_path))

    # Load meteor data from the FTPdetectinfo file
    meteor_data = readFTPdetectinfo(*os.path.split(ftpdetectinfo_path))

    if not len(meteor_data):
        print("No meteors in the FTPdetectinfo file!")
        return None

    platepar, recalibrated_platepars = getRecalibratedPlatepar(dir_path, config, file_list)

    # Compute nighly mean of the photometric zero point
    mag_lev_nightly_mean = np.mean(
        [recalibrated_platepars[ff_name].mag_lev for ff_name in recalibrated_platepars]
    )

    # Locate and load the mask file
    mask = getMaskFile(dir_path, config, file_list=file_list)

    # Compute the population index using the classical equation
    # Found to be more consistent when comparing fluxes
    population_index = 10 ** ((mass_index - 1) / 2.5)
    # population_index = 10**((mass_index - 1)/2.3) # TEST !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

    ### SENSOR CHARACTERIZATION ###
    # Computes FWHM of stars and noise profile of the sensor
    sensor_data = getSensorCharacterization(dir_path, config, meteor_data, default_fwhm=default_fwhm)

    # Compute the nighly mean FWHM
    fwhm_nightly_mean = np.mean([sensor_data[key][0] for key in sensor_data])

    ### ###

    # Perform shower association
    associations, _ = showerAssociation(
        config,
        [ftpdetectinfo_path],
        shower_code=shower_code,
        show_plot=False,
        save_plot=False,
        plot_activity=False,
    )

    # Init the flux configuration
    flux_config = FluxConfig()

    # Remove all meteors which begin below the limit height
    filtered_associations = {}
    for key in associations:
        meteor, shower = associations[key]

        if meteor.beg_alt > flux_config.elev_limit:
            print("Rejecting:", meteor.jdt_ref)
            filtered_associations[key] = (meteor, shower)

    associations = filtered_associations

    # If there are no shower association, return nothing
    if not associations:
        print("No meteors associated with the shower!")
        return None

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

    col_areas_ht, _ = getCollectingArea(dir_path, config, flux_config, platepar, mask)

    # Compute the pointing of the middle of the FOV
    _, ra_mid, dec_mid, _ = xyToRaDecPP(
        [jd2Date(J2000_JD.days)],
        [platepar.X_res / 2],
        [platepar.Y_res / 2],
        [1],
        platepar,
        extinction_correction=False,
    )
    azim_mid, elev_mid = raDec2AltAz(ra_mid[0], dec_mid[0], J2000_JD.days, platepar.lat, platepar.lon)

    # Compute the range to the middle point
    ref_ht = 100000
    r_mid, _, _, _ = xyHt2Geo(
        platepar,
        platepar.X_res / 2,
        platepar.Y_res / 2,
        ref_ht,
        indicate_limit=True,
        elev_limit=flux_config.elev_limit,
    )

    print("Range at 100 km in the middle of the image: {:.2f} km".format(r_mid / 1000))

    # Compute the average angular velocity to which the flux variation throught the night will be normalized
    #   The ang vel is of the middle of the FOV in the middle of observations

    # Middle Julian date of the night
    jd_night_mid = (datetime2JD(dt_beg) + datetime2JD(dt_end)) / 2

    # Compute the apparent radiant
    ra, dec, v_init = shower.computeApparentRadiant(platepar.lat, platepar.lon, jd_night_mid)

    # Compute the radiant elevation
    radiant_azim, radiant_elev = raDec2AltAz(ra, dec, jd_night_mid, platepar.lat, platepar.lon)

    # Compute the angular velocity in the middle of the FOV
    rad_dist_night_mid = angularSeparation(
        np.radians(radiant_azim), np.radians(radiant_elev), np.radians(azim_mid), np.radians(elev_mid)
    )
    ang_vel_night_mid = v_init * np.sin(rad_dist_night_mid) / r_mid

    ###

    # Compute the average limiting magnitude to which all flux will be normalized

    # Compute the theoretical stellar limiting magnitude using an empirical model (nightly average)
    lm_s_nightly_mean = stellarLMModel(mag_lev_nightly_mean)

    # A meteor needs to be visible on at least 4 frames, thus it needs to have at least 4x the mass to produce
    #   that amount of light. 1 magnitude difference scales as -0.4 of log of mass, thus:
    # frame_min_loss = np.log10(config.line_minimum_frame_range_det)/(-0.4)
    frame_min_loss = 0.0  # TEST !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11

    print("Frame min loss: {:.2} mag".format(frame_min_loss))

    lm_s_nightly_mean += frame_min_loss

    # Compute apparent meteor magnitude
    lm_m_nightly_mean = (
        lm_s_nightly_mean
        - 5 * np.log10(r_mid / 1e5)
        - 2.5
        * np.log10(
            np.degrees(
                platepar.F_scale
                * v_init
                * np.sin(rad_dist_night_mid)
                / (config.fps * r_mid * fwhm_nightly_mean)
            )
        )
    )

    print("Stellar lim mag using detection thresholds:", lm_s_nightly_mean)
    print("Apparent meteor limiting magnitude:", lm_m_nightly_mean)

    ### Apply time-dependent corrections ###

    # Go through all time bins within the observation period
    num_meteors = 0

    # generating forced bins to fill
    bin_datetime, sol_bins = forced_bins

    starting_sol = unwrapSol(jd2SolLonSteyaert(datetime2JD(dt_beg)), sol_bins[0], sol_bins[-1])
    ending_sol = unwrapSol(jd2SolLonSteyaert(datetime2JD(dt_end)), sol_bins[0], sol_bins[-1])
    forced_bins_time = np.array(
        [
            min(max(min(sol - starting_sol, ending_sol - sol) / (sol_bins[i + 1] - sol_bins[i]), 0), 1)
            * (bin_datetime[i + 1] - bin_datetime[i]).total_seconds()
            / 3600
            for i, sol in enumerate(sol_bins[:-1])
        ]
    )

    forced_bin_meteor_information = [[[], []] for _ in sol_bins[:-1]]
    forced_bin_intervals = [(bin_datetime[i], bin_datetime[i + 1]) for i in range(len(bin_datetime) - 1)]

    # mapping meteors to bins
    for meteor, shower in associations.values():
        meteor_date = jd2Date(meteor.jdt_ref, dt_obj=True)

        # Filter out meteors ending too close to the radiant
        ra, dec, _ = shower.computeApparentRadiant(platepar.lat, platepar.lon, meteor.jdt_ref)
        radiant_azim, radiant_elev = raDec2AltAz(ra, dec, meteor.jdt_ref, platepar.lat, platepar.lon)
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

        sol = unwrapSol(jd2SolLonSteyaert(meteor.jdt_ref), sol_bins[0], sol_bins[-1])
        indx = min(np.searchsorted(sol_bins, sol, side='right') - 1, len(sol_bins) - 2)
        forced_bin_meteor_information[indx][0].append(meteor)
        forced_bin_meteor_information[indx][1].append(meteor.ff_name)

    print('Calculating collecting area for fixed bins')
    forced_bins_meteor_num, forced_bins_area, forced_bins_lm_m = computeFluxCorrectionsOnBins(
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
        platepar,
        frame_min_loss,
        ang_vel_night_mid,
        sensor_data,
        lm_m_nightly_mean,
        confidence_interval=confidence_interval,
        no_skip=True,
    )
    print('Finished computing collecting areas for fixed bins')
    saveForcedBinFluxData(
        dir_path,
        f'fixedbinsflux_{config.stationID}_{starting_sol:.5f}_{ending_sol:.5f}',
        sol_bins,
        forced_bins_meteor_num,
        forced_bins_area,
        forced_bins_time,
        forced_bins_lm_m,
        sol_range=(starting_sol, ending_sol),
    )


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
        "shower_code", metavar="SHOWER_CODE", type=str, help="IAU shower code (e.g. ETA, PER, SDA)."
    )

    flux_parser.add_argument("s", metavar="MASS_INDEX", type=float, help="Mass index of the shower.")

    flux_parser.add_argument(
        "--timeinterval",
        nargs=2,
        metavar='INTERVAL',
        help="Time of the observation start and ending. YYYYMMDD_HHMMSS format for each",
    )

    parser_group2 = flux_parser.add_mutually_exclusive_group(required=True)
    parser_group2.add_argument(
        "--binduration", type=float, metavar='DURATION', help="Time bin width in hours."
    )
    parser_group2.add_argument("--binmeteors", type=int, metavar='COUNT', help="Number of meteors per bin")

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
        "0.4 has been tested to be good and it is the default",
    )
    return flux_parser


if __name__ == "__main__":


    import RMS.ConfigReader as cr

    # COMMAND LINE ARGUMENTS
    # Init the command line arguments parser
    flux_parser = fluxParser()
    cml_args = flux_parser.parse_args()

    # if cml_args.binmeteors is not None:
    #     raise NotImplementedError("--binmeteors not implemented")

    #########################

    ftpdetectinfo_path = cml_args.ftpdetectinfo_path
    ftpdetectinfo_path = findFTPdetectinfoFile(ftpdetectinfo_path)

    # # Apply wildcards to input
    # ftpdetectinfo_path_list = []
    # for entry in ftpdetectinfo_path:

    #     # Expand wildcards and find all paths
    #     paths = glob.glob(entry)

    #     # Only take paths to files, not directories
    #     paths = [entry for entry in paths if os.path.isfile(entry)]

    #     ftpdetectinfo_path_list += paths

    # # If there are no good files given, notify the user
    # if len(ftpdetectinfo_path_list) == 0:
    #     print("No FTPdetectinfo files given!")
    #     sys.exit()

    if not os.path.isfile(ftpdetectinfo_path):
        print("The FTPdetectinfo file does not exist:", ftpdetectinfo_path)
        print("Exiting...")
        sys.exit()

    # Extract parent directory
    dir_path = os.path.dirname(ftpdetectinfo_path)

    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, dir_path)

    datetime_pattern = "%Y/%M/%d %H:%M:%S"
    # find time intervals to compute flux with
    if cml_args.timeinterval is not None:
        dt_beg = datetime.datetime.strptime(cml_args.timeinterval[0], "%Y%m%d_%H%M%S")
        dt_end = datetime.datetime.strptime(cml_args.timeinterval[1], "%Y%m%d_%H%M%S")
        time_intervals = [(dt_beg, dt_end)]
    else:
        time_intervals = detectClouds(config, dir_path, show_plots=True, ratio_threshold=cml_args.ratiothres)
        for i, interval in enumerate(time_intervals):
            print(
                f'interval {i+1}/{len(time_intervals)}: '
                f'({interval[0].strftime(datetime_pattern)},{interval[1].strftime(datetime_pattern)})'
            )

        # print('display ff with clouds')
        # for ff in detect_clouds:
        #     print(ff, detect_clouds[ff])

    # Compute the flux
    for dt_beg, dt_end in time_intervals:
        print(f'Using interval: ({dt_beg.strftime(datetime_pattern)},{dt_end.strftime(datetime_pattern)})')
        computeFlux(
            config,
            dir_path,
            ftpdetectinfo_path,
            cml_args.shower_code,
            dt_beg,
            dt_end,
            cml_args.s,
            cml_args.binduration,
            cml_args.binmeteors,
            default_fwhm=cml_args.fwhm,
        )
