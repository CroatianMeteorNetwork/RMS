# RPi Meteor Station
# Copyright (C) 2026
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" Persistent hot pixel blacklist.

Cameras with in-camera hot pixel filtering disabled produce constant bright pixels which the star
extractor cannot distinguish from stars on a single FF file. Across a night they are trivially
separable: real stars drift at the sidereal rate (tens of pixels per hour), while hot pixels recur
at the same sub-pixel position for hours.

The blacklist lives in a small JSON file (config.hot_pixels_file, resolved night-dir-first with the
config directory as the fallback, like the mask). It is consulted live during capture so a field of
hot pixels cannot trip the ff_min_stars meteor-detection gate on an otherwise starless image, and it
is rebuilt every night from the full star list before the CALSTARS file is written: newly found
stationary detections are added, entries seen again are refreshed, and entries with no detections
for hot_pixels_max_age_days are dropped. No manual curation is needed.

A star very close to the celestial pole is the one celestial source that could look stationary, but
with the default gates (recurrence at one spot in >=20% of the night's FFs over >=45 min within a
~2 px radius) even Polaris drifts out of the matching circle, so only true sensor defects qualify.
"""

from __future__ import print_function, division, absolute_import

import os
import sys
import json
import datetime

import numpy as np

from RMS.Formats.FFfile import filenameToDatetime
from RMS.Logger import getLogger
from RMS.Misc import RmsDateTime

log = getLogger("rmslogger")


DATE_FMT = "%Y-%m-%d"


def hotPixelFilePath(dir_path, config):
    """ Resolve the hot pixel file path, night directory first, config directory as fallback.

    Return:
        path: [str] Path to an existing hot pixel file, or None if none exists.
    """

    # getattr guards keep this working with stale unpickled config objects predating the feature
    file_name = getattr(config, 'hot_pixels_file', 'hotpixels.json')

    for candidate_dir in [dir_path, getattr(config, 'config_file_path', None)]:

        if candidate_dir is None:
            continue

        path = os.path.join(candidate_dir, file_name)

        if os.path.isfile(path):
            return path

    return None


def loadHotPixels(dir_path, config):
    """ Load the hot pixel data structure.

    Return:
        hp_data: [dict] {"version": 1, "updated": ..., "pixels": [...]}. An empty structure is
            returned if no file exists or it cannot be parsed.
    """

    empty = {"version": 1, "updated": None, "pixels": []}

    path = hotPixelFilePath(dir_path, config)

    if path is None:
        return empty

    try:
        with open(path) as f:
            hp_data = json.load(f)

    except (ValueError, OSError) as e:
        log.warning("Could not read hot pixel file {:s}: {:s}".format(path, repr(e)))
        return empty

    if not isinstance(hp_data.get("pixels"), list):
        return empty

    return hp_data


def saveHotPixels(hp_data, dir_path, config):
    """ Atomically write the hot pixel data structure to dir_path. """

    hp_data["updated"] = RmsDateTime.utcnow().isoformat()

    path = os.path.join(dir_path, getattr(config, 'hot_pixels_file', 'hotpixels.json'))
    tmp_path = path + ".tmp"

    try:
        with open(tmp_path, "w") as f:
            json.dump(hp_data, f, indent=2)

        os.replace(tmp_path, path)

    except OSError as e:
        log.warning("Could not write hot pixel file {:s}: {:s}".format(path, repr(e)))


def hotPixelCoords(hp_data):
    """ Extract an Nx2 array of (x, y) coordinates from the hot pixel data structure. """

    pixels = hp_data.get("pixels", [])

    if not pixels:
        return np.empty((0, 2))

    return np.array([[p["x"], p["y"]] for p in pixels], dtype=np.float64)


def loadHotPixelCoords(dir_path, config):
    """ Load just the blacklisted (x, y) coordinates - the live capture-time entry point. """

    return hotPixelCoords(loadHotPixels(dir_path, config))


def matchHotPixels(x_arr, y_arr, hp_xy, radius):
    """ Match detections against the blacklist.

    Arguments:
        x_arr, y_arr: [list/ndarray] Detection coordinates.
        hp_xy: [ndarray] Nx2 array of blacklisted (x, y) positions.
        radius: [float] Match radius in pixels.

    Return:
        matched: [ndarray of bool] True for every detection within radius of any blacklisted pixel.
    """

    x_arr = np.asarray(x_arr, dtype=np.float64)
    y_arr = np.asarray(y_arr, dtype=np.float64)

    if len(hp_xy) == 0 or len(x_arr) == 0:
        return np.zeros(len(x_arr), dtype=bool)

    dx = x_arr[:, None] - hp_xy[None, :, 0]
    dy = y_arr[:, None] - hp_xy[None, :, 1]

    return np.any(dx**2 + dy**2 <= radius**2, axis=1)


def _ffTime(ff_name):
    """ FF file name -> datetime, or None if the name cannot be parsed. """

    try:
        return filenameToDatetime(ff_name)

    except (ValueError, IndexError, TypeError):
        return None


# A cluster must have detections in at least this many quarters of the observed night. A star
# drifting slowly through the match circle (e.g. near the celestial pole) produces one contiguous
# block of hits confined to a single quarter; a hot pixel fires throughout the night, clouds
# included, because it is a sensor defect
MIN_NIGHT_QUARTERS = 3

def findStationaryDetections(star_list, config):
    """ Find detections which recur at the same pixel position across the night.

    Verified against a real US005A night (2026-08-18, in-camera filtering disabled): the two
    genuine hot pixels recur on ~9% of FFs but across the whole night, while a slow near-pole star
    produced MORE within-radius hits (~380) confined to one ~65 min window. Raw counts therefore
    cannot separate the two - the temporal structure (quarter coverage + net drift) can.

    Arguments:
        star_list: [list] CALSTARS-style list of [ff_name, rows] pairs, where each row starts with
            (y, x, ...).
        config: [Config]

    Return:
        clusters: [list of dict] One entry per hot pixel candidate:
            {"x", "y", "n_det", "n_ff", "span_minutes"}
    """

    radius = config.hot_pixels_radius

    # Flatten all detections; remember which FF each came from
    det_x, det_y, det_ff = [], [], []
    ff_times = {}

    for ff_name, rows in star_list:

        rows = list(rows)

        if not rows:
            continue

        ff_times[ff_name] = _ffTime(ff_name)

        for row in rows:
            det_y.append(float(row[0]))
            det_x.append(float(row[1]))
            det_ff.append(ff_name)

    n_ff_total = len(ff_times)

    valid_times = [t for t in ff_times.values() if t is not None]

    if n_ff_total == 0 or len(valid_times) < 2:
        return []

    night_t0 = min(valid_times)
    night_t1 = max(valid_times)

    det_x = np.array(det_x)
    det_y = np.array(det_y)

    # A pixel must recur in a meaningful fraction of the night's star-bearing FFs to qualify.
    # The absolute floor protects short or mostly cloudy nights from promoting noise.
    min_hits = max(config.hot_pixels_min_count, int(np.ceil(config.hot_pixels_min_frac*n_ff_total)))

    # Bin detections on an integer pixel grid; sub-pixel jitter of a hot pixel can split it across
    # adjacent bins, so each candidate absorbs its 8 neighbours before the radius test
    bins = {}
    for i in range(len(det_x)):
        key = (int(round(det_x[i])), int(round(det_y[i])))
        bins.setdefault(key, []).append(i)

    reach = max(1, int(np.ceil(radius)))
    consumed = np.zeros(len(det_x), dtype=bool)
    clusters = []

    # Visit the densest bins first so each hot pixel is claimed by its own centre bin
    for key in sorted(bins, key=lambda k: len(bins[k]), reverse=True):

        candidates = []
        for dx_bin in range(-reach, reach + 1):
            for dy_bin in range(-reach, reach + 1):
                candidates.extend(bins.get((key[0] + dx_bin, key[1] + dy_bin), []))

        candidates = [i for i in candidates if not consumed[i]]

        if len(candidates) < min_hits:
            continue

        idx = np.array(candidates)
        cx = np.median(det_x[idx])
        cy = np.median(det_y[idx])

        # Keep only detections truly within the match radius of the cluster centre
        in_radius = (det_x[idx] - cx)**2 + (det_y[idx] - cy)**2 <= radius**2
        idx = idx[in_radius]

        # Count distinct FFs - multiple detections on one FF must not inflate the recurrence
        cluster_ffs = set(det_ff[i] for i in idx)

        if len(cluster_ffs) < min_hits:
            continue

        # A real star drifts through the match circle in minutes; require the recurrence to span
        # a large part of the night
        times = [ff_times[ff] for ff in cluster_ffs if ff_times[ff] is not None]

        if len(times) < 2:
            continue

        span_minutes = (max(times) - min(times)).total_seconds()/60

        if span_minutes < config.hot_pixels_min_span_minutes:
            continue

        # Require detections spread across the night - a slow-moving star (e.g. near the pole)
        # can rack up an hour of hits inside the match circle, but only as one contiguous block
        night_span_s = (night_t1 - night_t0).total_seconds()
        quarters = set()
        for t in times:
            q = int(4*(t - night_t0).total_seconds()/(night_span_s + 1e-9))
            quarters.add(min(q, 3))

        if len(quarters) < MIN_NIGHT_QUARTERS:
            continue

        # Net drift test: order the cluster's detections by time and compare the median position
        # of the first and last thirds. A hot pixel stays put (sub-pixel jitter only); a star
        # crosses the match circle, displacing by ~the circle diameter
        idx_times = np.array([
            (ff_times[det_ff[i]] - night_t0).total_seconds() if ff_times[det_ff[i]] is not None
            else -1.0 for i in idx])
        timed = idx[idx_times >= 0]
        timed = timed[np.argsort(idx_times[idx_times >= 0])]

        third = max(1, len(timed)//3)
        dx_net = np.median(det_x[timed[-third:]]) - np.median(det_x[timed[:third]])
        dy_net = np.median(det_y[timed[-third:]]) - np.median(det_y[timed[:third]])

        if dx_net**2 + dy_net**2 > (radius/2)**2:
            continue

        consumed[idx] = True

        clusters.append({
            "x": round(float(np.median(det_x[idx])), 2),
            "y": round(float(np.median(det_y[idx])), 2),
            "n_det": int(len(idx)),
            "n_ff": int(len(cluster_ffs)),
            "span_minutes": round(span_minutes, 1),
        })

    return clusters


def updateHotPixelList(hp_data, clusters, night_date, config):
    """ Merge tonight's stationary clusters into the persistent list and age out stale entries.

    Reprocessing an old night is safe: last_seen only ever moves forward (max-merge), and ageing is
    measured relative to the night being processed, so an old night can never age out entries that
    a newer night has refreshed.

    Arguments:
        hp_data: [dict] Existing hot pixel structure.
        clusters: [list of dict] Output of findStationaryDetections.
        night_date: [datetime.date] Date of the night being processed.
        config: [Config]

    Return:
        hp_data: [dict] Updated structure.
    """

    radius = config.hot_pixels_radius
    night_str = night_date.strftime(DATE_FMT)

    pixels = list(hp_data.get("pixels", []))

    unmatched_clusters = list(clusters)

    for p in pixels:

        # Find the closest cluster matching this existing entry
        best = None
        for c in unmatched_clusters:
            if (c["x"] - p["x"])**2 + (c["y"] - p["y"])**2 <= radius**2:
                best = c
                break

        if best is not None:

            unmatched_clusters.remove(best)

            # Refresh the position and the last-seen date (never move last_seen backwards)
            p["x"], p["y"] = best["x"], best["y"]
            p["nights_seen"] = int(p.get("nights_seen", 0)) + 1
            p["hits_last_night"] = best["n_ff"]

            try:
                prev_seen = datetime.datetime.strptime(p["last_seen"], DATE_FMT).date()
            except (KeyError, ValueError):
                prev_seen = night_date

            p["last_seen"] = max(prev_seen, night_date).strftime(DATE_FMT)

    # Add newly discovered hot pixels
    for c in unmatched_clusters:
        pixels.append({
            "x": c["x"],
            "y": c["y"],
            "first_seen": night_str,
            "last_seen": night_str,
            "nights_seen": 1,
            "hits_last_night": c["n_ff"],
        })

    # Age out entries not seen for too long (relative to the night being processed, so
    # reprocessing old data never ages anything)
    kept = []
    for p in pixels:

        try:
            last_seen = datetime.datetime.strptime(p["last_seen"], DATE_FMT).date()
        except (KeyError, ValueError):
            last_seen = night_date

        if (night_date - last_seen).days > config.hot_pixels_max_age_days:
            log.info("Hot pixel ({:.1f}, {:.1f}) not seen since {:s}, removing from the blacklist".format(
                p["x"], p["y"], p["last_seen"]))
        else:
            kept.append(p)

    hp_data["pixels"] = kept

    return hp_data


def filterStarList(star_list, hp_xy, radius):
    """ Remove all star rows within radius of any blacklisted pixel.

    Arguments:
        star_list: [list] CALSTARS-style list of [ff_name, rows] pairs, rows starting with (y, x).

    Return:
        filtered: [list] Same structure with matching rows removed (FF entries left with no stars
            are dropped, matching writeCALSTARS's expectations).
        n_removed: [int] Number of rows removed.
    """

    if len(hp_xy) == 0:
        return star_list, 0

    filtered = []
    n_removed = 0

    for ff_name, rows in star_list:

        rows = list(rows)

        if not rows:
            continue

        ys = [row[0] for row in rows]
        xs = [row[1] for row in rows]

        matched = matchHotPixels(xs, ys, hp_xy, radius)
        n_removed += int(np.count_nonzero(matched))

        kept_rows = [row for row, m in zip(rows, matched) if not m]

        if kept_rows:
            filtered.append([ff_name, kept_rows])

    return filtered, n_removed


def applyHotPixels(star_list, night_dir, config):
    """ Nightly entry point: analyze the night, update the persistent blacklist, filter the stars.

    The master list in the config directory is updated and an audit copy is written into the night
    directory, so archived nights record exactly which pixels were blacklisted when they were
    processed.

    Arguments:
        star_list: [list] CALSTARS-style list of [ff_name, rows] pairs (rows must be materialized
            lists, not generators).
        night_dir: [str] Path to the night directory.
        config: [Config]

    Return:
        star_list: [list] Filtered star list, safe to pass to writeCALSTARS.
    """

    # Derive the night date from the earliest FF file
    times = [t for t in (_ffTime(ff_name) for ff_name, _ in star_list) if t is not None]

    if not times:
        return star_list

    night_date = min(times).date()

    # Load the master list (config dir; the night dir has no copy yet at this point)
    hp_data = loadHotPixels(night_dir, config)

    # Find tonight's stationary detections and merge them in
    clusters = findStationaryDetections(star_list, config)

    if clusters:
        log.info("Found {:d} stationary (hot pixel) positions: {:s}".format(
            len(clusters), ", ".join("({:.1f}, {:.1f}) on {:d} FFs".format(
                c["x"], c["y"], c["n_ff"]) for c in clusters)))

    hp_data = updateHotPixelList(hp_data, clusters, night_date, config)

    # Persist the master list and an audit copy in the night dir
    if config.config_file_path:
        saveHotPixels(hp_data, config.config_file_path, config)

    saveHotPixels(hp_data, night_dir, config)

    # Filter the night's stars with the updated list
    hp_xy = hotPixelCoords(hp_data)
    star_list, n_removed = filterStarList(star_list, hp_xy, config.hot_pixels_radius)

    if n_removed:
        log.info("Removed {:d} hot pixel detections from CALSTARS ({:d} blacklisted pixels)".format(
            n_removed, len(hp_xy)))

    return star_list


def primeFromCALSTARS(night_dir, config):
    """ Prime or refresh the master blacklist from an already-processed night's CALSTARS file.

    Runs the same stationary-detection analysis the nightly pipeline runs and merges the result
    into the master list in the config directory. Useful for bootstrapping the list from an
    archived night (e.g. one processed before the filter existed) without waiting for the next
    capture - the live gate picks the file up on the next FF, even mid-capture.

    Return:
        clusters: [list of dict] The stationary clusters found, or None if no CALSTARS file exists.
    """

    from RMS.Formats import CALSTARS as CALSTARSFormat

    calstars_files = [f for f in sorted(os.listdir(night_dir))
        if f.startswith('CALSTARS') and f.endswith('.txt')]

    if not calstars_files:
        return None

    calstars_data = CALSTARSFormat.readCALSTARS(night_dir, calstars_files[0])

    if not calstars_data:
        return None

    star_list = calstars_data[0]

    # Derive the night date from the earliest FF entry
    times = [t for t in (_ffTime(ff_name) for ff_name, _ in star_list) if t is not None]

    if not times:
        return None

    clusters = findStationaryDetections(star_list, config)

    hp_data = loadHotPixels(config.config_file_path, config)
    hp_data = updateHotPixelList(hp_data, clusters, min(times).date(), config)

    saveHotPixels(hp_data, config.config_file_path, config)

    # Also drop a copy into the night dir - tools opened on the night dir (e.g. SkyFit2 loading
    # the night's archived .config copy) resolve the blacklist night-dir-first
    saveHotPixels(hp_data, night_dir, config)

    return clusters


if __name__ == "__main__":

    import argparse

    import RMS.ConfigReader as cr

    arg_parser = argparse.ArgumentParser(description="Manage the persistent hot pixel blacklist. "
        "With no arguments, lists the current blacklist.")

    arg_parser.add_argument('-p', '--prime', metavar='NIGHT_DIR', type=str,
        help="Prime/refresh the blacklist from the CALSTARS file in the given night directory.")

    arg_parser.add_argument('-r', '--remove', nargs=2, metavar=('X', 'Y'), type=float,
        help="Remove the blacklist entry within the match radius of the given X Y position.")

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str,
        help="Path to a config file which will be used instead of the default one.")

    cml_args = arg_parser.parse_args()

    config = cr.loadConfigFromDirectory(cml_args.config,
        cml_args.prime if cml_args.prime else os.getcwd())

    master_path = os.path.join(config.config_file_path, config.hot_pixels_file)

    if cml_args.prime:

        clusters = primeFromCALSTARS(os.path.abspath(cml_args.prime), config)

        if clusters is None:
            print("No usable CALSTARS file found in {:s}".format(cml_args.prime))
            sys.exit(1)

        print("Found {:d} stationary (hot pixel) positions:".format(len(clusters)))
        for c in clusters:
            print("  ({:7.2f}, {:7.2f})  on {:d} FFs over {:.0f} min".format(
                c["x"], c["y"], c["n_ff"], c["span_minutes"]))
        print("Master list updated: {:s}".format(master_path))

    elif cml_args.remove:

        rx, ry = cml_args.remove
        hp_data = loadHotPixels(config.config_file_path, config)

        kept = [p for p in hp_data["pixels"]
            if (p["x"] - rx)**2 + (p["y"] - ry)**2 > config.hot_pixels_radius**2]

        n_removed = len(hp_data["pixels"]) - len(kept)
        hp_data["pixels"] = kept

        saveHotPixels(hp_data, config.config_file_path, config)
        print("Removed {:d} entries within {:.1f} px of ({:.1f}, {:.1f})".format(
            n_removed, config.hot_pixels_radius, rx, ry))

    else:

        hp_data = loadHotPixels(config.config_file_path, config)

        if not hp_data["pixels"]:
            print("Blacklist is empty ({:s})".format(master_path))
        else:
            print("{:d} blacklisted pixels in {:s}:".format(len(hp_data["pixels"]), master_path))
            print("  {:>8s} {:>8s}  {:>10s} {:>10s} {:>7s} {:>9s}".format(
                "X", "Y", "first", "last", "nights", "hits"))
            for p in hp_data["pixels"]:
                print("  {:8.2f} {:8.2f}  {:>10s} {:>10s} {:7d} {:9d}".format(
                    p["x"], p["y"], p.get("first_seen", "?"), p.get("last_seen", "?"),
                    p.get("nights_seen", 0), p.get("hits_last_night", 0)))
