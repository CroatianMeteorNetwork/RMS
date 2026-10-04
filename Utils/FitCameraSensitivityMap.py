""" Fit a per-camera image-plane sensitivity map.

The site light-dome model shares one alt-az brightness pattern across all co-located
cameras and gives each camera a single scalar LM0. But a camera's limiting magnitude
varies across its own image plane (corner optics, PSF degradation, vignetting): measured
on USC0G4, block-wise fits span LM 4.7-5.5 across one FOV. For a fixed camera the image
position maps 1:1 to alt-az, so a SITE-shared sky model structurally cannot represent
per-camera FOV quality - six differently-aimed cameras project their good/bad image
regions onto six different sky patches.

This tool fits a block-wise logistic detection model per camera from archived nights:
    P(detected | m, block) = 1/(1 + exp((m - LM_block)/s))
over catalog-star hit/miss trials (same construction as FitLightDome, but per image
block and only on fully dark, moonless frames). Use clear nights: there is no cloud
filter, and a cloudy frame reads as a shallow block.

The map is written to <data_dir>/<stationID>_sensitivity_map.json and its presence
activates it in the flux collection area (RMS.SensitivityMap, Utils.Flux): block
limiting magnitudes replace the platepar's vignetting-and-extinction loss and the
zero-point LM inference. The map also records the photometric zero point and the
vignetting coefficient of the frames it was fitted on, so later nights can be placed
relative to it through their own zero point. That reference is in the intensity units
of the camera configuration at fit time, so refit the map after any firmware, gain or
gamma change, and after a change of the star-extraction gate (the map measures what
the extractor detects).

Usage:
    python -m Utils.FitCameraSensitivityMap /path/to/station_config_dir \\
        --nights 20260710,20260711 [--blocks 4x3]
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import json
import os
import re
import time

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

import RMS.ConfigReader as cr
from RMS.Formats import FFfile, StarCatalog
from RMS.Formats.CALSTARS import readCALSTARS
from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.ApplyRecalibrate import loadRecalibratedPlatepar
from RMS.Astrometry.Conversions import date2JD
from RMS.Logger import getLogger
from RMS.Routines.MaskImage import getMaskFile
from RMS.SensitivityMap import (SensitivityMap, configFingerprint, frameDepth, mapStaleness,
                                nightPhotometry, sensitivityMapQualityIssues, ZP_EPOCH_TOL)
from Utils.FitLightDome import sunAltitude, moonAltPhase, SUN_ALT_MAX, MOON_PHASE_MAX
from Utils.Flux import projectCatalogStarsInFOV

log = getLogger("rmslogger")

MATCH_RADIUS_PX = 3.0
FF_PER_NIGHT = 30
TRIAL_LIM_MAG = 8.0     # deep enough to sample the rolloff of any current camera


def fitSensitivityMap(config, night_dirs, nbx=4, nby=3, lim_mag=TRIAL_LIM_MAG):
    """ Fit per-block LM with a shared logistic width from archived nights.

    Arguments:
        config: [Config]
        night_dirs: [list of str] Night directories with CALSTARS and flux platepars.

    Keyword arguments:
        nbx, nby: [int] Image blocks in x and y.
        lim_mag: [float] Catalog depth for the trials.

    Return:
        map_dict: [dict] stationID, nbx, nby, LM (nby*nbx, row-major), s, n_trials,
            fit_date, nights - or None if no usable trials.
    """

    catalog_stars, _, _ = StarCatalog.readStarCatalog(config.star_catalog_path,
        config.star_catalog_file, lim_mag=lim_mag,
        mag_band_ratios=config.star_catalog_band_ratios)

    mags, blocks, hits = [], [], []
    zero_points, vig_coeffs, depths = [], [], []
    nights_used = []
    w = h = None
    pp_last = None

    for night_dir in night_dirs:

        file_list = sorted(os.listdir(night_dir))
        try:
            calstars_file = next(f for f in file_list
                                 if "CALSTARS" in f and f.endswith(".txt"))
        except StopIteration:
            continue
        calstars_list, _ = readCALSTARS(night_dir, calstars_file)
        calstars = {ff: (np.array(st)[:, [1, 0]].astype(float) if len(st)
                         else np.zeros((0, 2))) for ff, st in calstars_list}

        pps = loadRecalibratedPlatepar(night_dir, config, file_list, type="flux")
        pps = {ff: pp for ff, pp in (pps or {}).items()
               if getattr(pp, "auto_recalibrated", False)}
        if len(pps) < 3:
            continue
        nights_used.append(os.path.basename(night_dir))
        mask = getMaskFile(night_dir, config, file_list=file_list,
                           default_as_backup=True)

        valid = sorted((FFfile.filenameToDatetime(ff), ff) for ff in pps)
        valid_times = np.array([t for t, _ in valid])

        ffs = sorted(calstars.keys())
        picks = [ffs[i] for i in
                 np.unique(np.linspace(0, len(ffs) - 1, FF_PER_NIGHT).astype(int))]

        for ff in picks:
            date = FFfile.getMiddleTimeFF(ff, config.fps, ret_milliseconds=True)
            jd = date2JD(*date)

            pp0 = pps[valid[0][1]]
            if sunAltitude(jd, pp0.lat, pp0.lon) > SUN_ALT_MAX:
                continue
            moon_alt, moon_phase = moonAltPhase(jd, pp0.lat, pp0.lon)
            if (moon_alt > 0) and (moon_phase > MOON_PHASE_MAX):
                continue

            t = FFfile.filenameToDatetime(ff)
            i = np.searchsorted(valid_times, t)
            cand = [j for j in (i - 1, i) if 0 <= j < len(valid)]
            if not cand:
                continue
            j = min((abs((valid[k][0] - t).total_seconds()), k) for k in cand)[1]
            pp = pps[valid[j][1]]
            pp_last = pp
            w, h = pp.X_res, pp.Y_res
            zero_points.append(float(pp.mag_lev))
            vig_coeffs.append(float(pp.vignetting_coeff) if pp.vignetting_coeff is not None else 0.0)
            depth = frameDepth(getattr(pp, "star_list", None))
            if depth is not None:
                depths.append(depth)

            x, y, mag, az, alt, _ = projectCatalogStarsInFOV(pp, date, jd, catalog_stars,
                mask=mask)
            if not len(x):
                continue

            det = calstars[ff]
            if len(det):
                dd, _ = cKDTree(det).query(np.column_stack([x, y]), k=1)
                hit = dd <= MATCH_RADIUS_PX
            else:
                hit = np.zeros(len(x), bool)

            blk = (np.clip((np.asarray(y)*nby/h).astype(int), 0, nby - 1)*nbx
                   + np.clip((np.asarray(x)*nbx/w).astype(int), 0, nbx - 1))

            mags.extend(mag)
            blocks.extend(blk)
            hits.extend(hit)

    if len(mags) < 500*nbx*nby:
        print("Too few trials ({:d}) for a {:d}x{:d} map".format(len(mags), nbx, nby))
        return None

    m = np.array(mags)
    blk = np.array(blocks)
    hit = np.array(hits, float)

    def nll(p):
        s = max(p[-1], 0.05)
        lmb = np.array(p[:-1])[blk]
        pr = np.clip(1.0/(1.0 + np.exp((m - lmb)/s)), 1e-6, 1 - 1e-6)
        return -np.sum(hit*np.log(pr) + (1 - hit)*np.log(1 - pr))

    p0 = [5.0]*(nbx*nby) + [0.4]
    res = minimize(nll, p0, method="Nelder-Mead",
                   options=dict(maxiter=40000, xatol=1e-3, fatol=1e-2))

    return dict(
        stationID=str(config.stationID),
        nbx=nbx, nby=nby,
        X_res=int(w), Y_res=int(h),
        LM=[round(float(v), 3) for v in res.x[:-1]],
        s=round(float(max(res.x[-1], 0.05)), 3),
        n_trials=int(len(m)),
        trial_lim_mag=lim_mag,
        # Photometric reference of the fitted frames: later nights are placed relative to it
        # through their zero point, after moving it to the then-current vignetting coefficient
        mag_lev_ref=round(float(np.median(zero_points)), 3),
        vignetting_coeff_ref=float(np.median(vig_coeffs)),
        # Matched-star depth of the fitted frames (max over frames, the cloud-immune statistic
        # nightPhotometry measures), so a camera that later reaches clearly deeper is noticed
        depth_ref=(round(float(max(depths)), 3) if depths else None),
        # The configuration the map describes; a change in its units or geometry retires the map
        fingerprint=configFingerprint(config, pp_last),
        pointing=([round(float(pp_last.az_centre), 2), round(float(pp_last.alt_centre), 2)]
                  if pp_last is not None else None),
        fit_date=time.strftime("%Y-%m-%d", time.gmtime()),
        # Only nights that contributed trials (a night without flux platepars is skipped)
        nights=sorted(nights_used),
    )



# Self-maintenance (ensureSensitivityMap)
AUTO_SELECT_WINDOW = 28     # trailing night directories considered: one lunar cycle, so a dark
                            # fortnight is always inside the window
AUTO_MAX_NIGHTS = 3         # nights pooled into one fit
AUTO_MIN_NIGHTS = 1         # one clear dark night already gives tens of thousands of trials
AUTO_REL_CLARITY = 0.5      # keep nights at least this clear (median matched stars) relative
                            # to the clearest selected one
AUTO_DEPTH_TOL = 0.4        # mag - nights this much shallower than the deepest candidate are a
                            # different extraction regime (or haze) and are left out
AUTO_MIN_FRAMES = 10        # recalibrated frames a night needs to be a candidate
AUTO_ATTEMPT_MARKER = "sensitivity_map_fit_attempt.json"


def selectMapNights(config, window=AUTO_SELECT_WINDOW, max_nights=AUTO_MAX_NIGHTS,
                    min_rel_clarity=AUTO_REL_CLARITY, reference_zp=None):
    """ The clearest recent archived nights that belong to the camera's current intensity epoch.

    A firmware, gain or gamma change moves the recalibrated zero point by magnitudes; nights from
    before such a change must not train a map for after it. The epoch is defined by the reference
    zero point (tonight's, when given) or else by the most recent candidate night, and only nights
    within ZP_EPOCH_TOL of it qualify. Nights clearly shallower than the deepest candidate are also
    left out: they are either hazy or from before an extraction-gate improvement.

    Arguments:
        config: [Config] Station config (data_dir locates ArchivedFiles).

    Keyword arguments:
        window: [int] Trailing night directories to consider.
        max_nights: [int] Nights to select.
        min_rel_clarity: [float] Keep only nights at least this clear relative to the best one.
        reference_zp: [float] Zero point defining the current intensity epoch (tonight's).

    Return:
        night_dirs: [list of str] Selected night directories, clearest first.
    """

    archive_dir = os.path.join(os.path.expanduser(config.data_dir), "ArchivedFiles")
    if not os.path.isdir(archive_dir):
        return []

    all_dirs = sorted(d for d in os.listdir(archive_dir)
                      if d.startswith(str(config.stationID) + "_")
                      and os.path.isdir(os.path.join(archive_dir, d)))

    candidates = []
    for d in all_dirs[-window:]:
        path = os.path.join(archive_dir, d)
        night = nightPhotometry(path)
        if (night is None) or (night["n_frames"] < AUTO_MIN_FRAMES):
            continue
        candidates.append((path, night))

    if not candidates:
        return []

    epoch_zp = reference_zp if reference_zp is not None else candidates[-1][1]["zp"]
    epoch = [c for c in candidates if abs(c[1]["zp"] - epoch_zp) <= ZP_EPOCH_TOL]

    depths = [c[1]["depth"] for c in epoch if c[1]["depth"] is not None]
    if depths:
        best_depth = max(depths)
        epoch = [c for c in epoch
                 if (c[1]["depth"] is None) or (c[1]["depth"] >= best_depth - AUTO_DEPTH_TOL)]

    epoch.sort(key=lambda c: c[1]["matched"], reverse=True)
    epoch = epoch[:max_nights]

    if epoch and (min_rel_clarity is not None):
        best = epoch[0][1]["matched"]
        epoch = [c for c in epoch if c[1]["matched"] >= min_rel_clarity*best]

    return [c[0] for c in epoch]



def ensureSensitivityMap(config, platepar=None, night_dir=None):
    """ Keep the station's sensitivity map describing its camera, with no operator involved.

    Called nightly from the flux preparation. The installed map is checked against the current
    configuration fingerprint and the night's own photometry (RMS.SensitivityMap.mapStaleness).
    When it is missing, hard-stale (intensity units or geometry changed) or soft-stale (old,
    extraction gate retuned, camera deeper than at fit time), the clearest recent archived nights
    of the current intensity epoch are selected and the map is refitted and installed. Attempts
    are rate-limited to one per day, so a camera whose change has not yet been followed by a clear
    night retries cheaply until one arrives. Until a refit succeeds, a hard-stale map is refused by
    SensitivityMap.load and the flux falls back to the platepar vignetting.

    Arguments:
        config: [Config] Station config.

    Keyword arguments:
        platepar: [Platepar] Current platepar (image size, pointing, vignetting coefficient).
        night_dir: [str] Tonight's directory: its photometry defines the intensity epoch and is
            compared with the map's references.

    Return:
        [bool] True if a usable map is in place after the call.
    """

    data_dir = os.path.expanduser(config.data_dir)
    station = str(config.stationID)
    path = SensitivityMap.stationMapPath(config)

    current = None
    if os.path.isfile(path):
        try:
            with open(path) as f:
                current = json.load(f)
        except Exception:
            current = None

    hard, soft = [], []
    if current is not None:
        hard, soft = mapStaleness(current, config, platepar=platepar, night_dir=night_dir)
        if not hard and not soft:
            return True
        for msg in hard:
            log.info("Sensitivity map retired: {:s}".format(msg))
        for msg in soft:
            log.info("Sensitivity map due for a refit: {:s}".format(msg))
    else:
        log.info("No sensitivity map for {:s} - fitting one from the archive".format(station))

    usable = (current is not None) and (not hard)

    # One attempt per day
    marker_path = os.path.join(data_dir, "{:s}_{:s}".format(station, AUTO_ATTEMPT_MARKER))
    today = time.strftime("%Y-%m-%d", time.gmtime())
    try:
        with open(marker_path) as f:
            if json.load(f).get("date") == today:
                return usable
    except Exception:
        pass
    try:
        with open(marker_path, "w") as f:
            json.dump(dict(date=today), f)
    except Exception:
        pass

    reference_zp = None
    if night_dir is not None:
        tonight = nightPhotometry(night_dir)
        if tonight is not None:
            reference_zp = tonight["zp"]

    night_dirs = selectMapNights(config, reference_zp=reference_zp)
    if len(night_dirs) < AUTO_MIN_NIGHTS:
        log.info("Sensitivity map: no clear archived night of the current intensity epoch yet - "
                 "retrying tomorrow")
        return usable

    dates = sorted(set(m.group(1) for m in
                       (re.search(r"_(\d{8})(?:_|$)", os.path.basename(d)) for d in night_dirs) if m))
    log.info("Sensitivity map fit from the {:d} clearest archived night(s) of this epoch: {:s}".format(
        len(night_dirs), ", ".join(dates)))

    map_dict = fitSensitivityMap(config, night_dirs)
    if map_dict is None:
        log.info("Sensitivity map fit produced too few trials - keeping the previous state")
        return usable

    issues = sensitivityMapQualityIssues(map_dict)
    if issues:
        log.info("Sensitivity map fit rejected: " + "; ".join(issues))
        return usable

    # The fresh fit must describe the camera as it is now (e.g. the archive could predate a move)
    fresh_hard, _ = mapStaleness(map_dict, config, platepar=platepar, night_dir=night_dir)
    if fresh_hard:
        log.info("Sensitivity map fit does not match the current camera - not installed: "
                 + "; ".join(fresh_hard))
        return usable

    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(map_dict, f, indent=1)
    os.replace(tmp_path, path)

    lm = np.array(map_dict["LM"])
    log.info("Sensitivity map installed: {:s} (best block LM {:.2f}, spread {:.2f} mag, {:d} trials)".format(
        path, float(lm.max()), float(lm.max() - lm.min()), int(map_dict["n_trials"])))

    return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fit a per-camera sensitivity map")
    parser.add_argument("config_dirs", nargs="+", help="station config directories")
    parser.add_argument("--nights", default=None,
                        help="comma-separated YYYYMMDD training dates (not with --ensure)")
    parser.add_argument("--ensure", action="store_true",
                        help="run the nightly self-maintenance instead: refit from the clearest "
                             "recent nights of the current intensity epoch only if the installed "
                             "map is missing or stale")
    parser.add_argument("--blocks", default="4x3")
    parser.add_argument("--out", default=None, help="output file (single config directory only)")
    args = parser.parse_args()

    if args.ensure == (args.nights is not None):
        parser.error("give either --nights or --ensure")
    if args.out and (len(args.config_dirs) > 1):
        parser.error("--out needs a single config directory")

    nbx, nby = (int(v) for v in args.blocks.lower().split("x"))

    for config_dir in args.config_dirs:

        config = cr.loadConfigFromDirectory(".", config_dir)

        if args.ensure:
            platepar = None
            pp_path = os.path.join(config.config_file_path, config.platepar_name)
            if os.path.isfile(pp_path):
                platepar = Platepar()
                platepar.read(pp_path, use_flat=None)
            present = ensureSensitivityMap(config, platepar=platepar)
            print("{:s}: {:s}".format(config.stationID,
                "usable map in place" if present else "no usable map (see log)"))
            continue

        arch = os.path.join(os.path.expanduser(config.data_dir), "ArchivedFiles")
        dates = args.nights.split(",")
        # Directories only: the archive also holds the night's tarballs next to the directory
        night_dirs = [d for d in sorted(glob.glob(os.path.join(arch, "*_*")))
                      if os.path.isdir(d) and (os.path.basename(d).split("_")[1] in dates)]
        print("{:s}: {:d} night dir(s)".format(config.stationID, len(night_dirs)))

        map_dict = fitSensitivityMap(config, night_dirs, nbx=nbx, nby=nby)
        if map_dict is None:
            print("{:s}: too few trials, no map written".format(config.stationID))
            continue

        out = args.out or SensitivityMap.stationMapPath(config)
        with open(out, "w") as f:
            json.dump(map_dict, f, indent=1)

        print("map written to {:s}".format(out))
        lm = np.array(map_dict["LM"]).reshape(nby, nbx)
        print("LM map (image rows top to bottom):")
        for row in lm:
            print("  " + "  ".join("{:.2f}".format(v) for v in row))
        print("s = {:.3f}, spread = {:.2f} mag".format(map_dict["s"], lm.max() - lm.min()))
