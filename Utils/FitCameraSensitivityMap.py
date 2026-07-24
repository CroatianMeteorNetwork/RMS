""" Fit a per-camera image-plane sensitivity map (EXPERIMENTAL - not yet consumed by
the pipeline).

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
block and only on fully dark, moonless frames). The map is written as JSON; intended
consumers (once wired): the dome model as a per-camera LM offset field, and any
predictor of per-position detectability.

Usage:
    python -m Utils.FitCameraSensitivityMap /path/to/station_config_dir \\
        --nights 20260710,20260711 [--blocks 4x3]
"""

from __future__ import absolute_import, division, print_function

import argparse
import datetime
import glob
import json
import os

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

import RMS.ConfigReader as cr
from RMS.Formats import FFfile, StarCatalog
from RMS.Formats.CALSTARS import readCALSTARS
from RMS.Astrometry.ApplyRecalibrate import loadRecalibratedPlatepar
from RMS.Astrometry.Conversions import date2JD
from RMS.Routines.MaskImage import getMaskFile
from Utils.FitLightDome import sunAltitude, moonAltPhase, SUN_ALT_MAX, MOON_PHASE_MAX
from Utils.Flux import projectCatalogStarsInFOV

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
    w = h = None

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
            w, h = pp.X_res, pp.Y_res

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
        LM=[round(float(v), 3) for v in res.x[:-1]],
        s=round(float(max(res.x[-1], 0.05)), 3),
        n_trials=int(len(m)),
        trial_lim_mag=lim_mag,
        fit_date=datetime.datetime.utcnow().strftime("%Y-%m-%d"),
        nights=sorted(os.path.basename(d) for d in night_dirs),
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fit a per-camera sensitivity map")
    parser.add_argument("config_dir")
    parser.add_argument("--nights", required=True,
                        help="comma-separated YYYYMMDD training dates")
    parser.add_argument("--blocks", default="4x3")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    nbx, nby = (int(v) for v in args.blocks.lower().split("x"))
    config = cr.loadConfigFromDirectory(".", args.config_dir)

    arch = os.path.join(os.path.expanduser(config.data_dir), "ArchivedFiles")
    dates = args.nights.split(",")
    night_dirs = [d for d in sorted(glob.glob(os.path.join(arch, "*_*")))
                  if os.path.basename(d).split("_")[1] in dates]
    print("{:d} night dir(s)".format(len(night_dirs)))

    map_dict = fitSensitivityMap(config, night_dirs, nbx=nbx, nby=nby)
    if map_dict is None:
        raise SystemExit(1)

    out = args.out or os.path.join(os.path.expanduser(config.data_dir),
        "{:s}_sensitivity_map.json".format(config.stationID))
    with open(out, "w") as f:
        json.dump(map_dict, f, indent=1)

    print("map written to {:s}".format(out))
    lm = np.array(map_dict["LM"]).reshape(nby, nbx)
    print("LM map (image rows top to bottom):")
    for row in lm:
        print("  " + "  ".join("{:.2f}".format(v) for v in row))
    print("s = {:.3f}, spread = {:.2f} mag".format(map_dict["s"], lm.max() - lm.min()))
