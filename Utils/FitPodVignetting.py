""" Fit one vignetting coefficient for a multi-camera pod from stars that two cameras see at the same
time in their field-of-view overlap.

The single-camera photometry fit in SkyFit2 has to separate the radial vignetting falloff from the
atmospheric extinction gradient, catalog band errors and star colour, all on one image. Where two
cameras of a pod overlap, the same star is measured at the same instant through the same air, so
every one of those terms cancels in the difference of the two instrumental magnitudes. What is left is

    m_A - m_B = V(r_A) - V(r_B) + (ZP_A - ZP_B)

with V(r) = -10 log10 cos(k r) the cos^4 vignetting loss in magnitudes at radius r (px) from the image
centre. The star catalog is never used photometrically: the platepars only give sky positions for
matching the two detections. On the US005 pod the overlaps cover about half of the outer field of
each ring camera and over 90% of the two zenith cameras, and one night gives 1e5 to 2e5 pairs.

The fit profiles out one zero-point per camera pair (the gain difference), clips outliers, and
reports per-camera deviations from the shared coefficient as a hardware monitor: a camera whose
own radial residual departs by more than ~0.1 mag has dew, a focus change or a decentred lens.

Faint stars carry a measurement bias that correlates with the fitted FWHM (noise that broadens the
Gaussian fit also enlarges the crop and the summed counts), so the fit uses stars above an S/N
cut by default. The optional FWHM^2 nuisance term absorbs the remainder.

Usage:
    python -m Utils.FitPodVignetting ~/source/Stations/US005A ~/source/Stations/US005B ... \\
        --night 20261001 [--snr 20] [--fwhm-term] [--json report.json]

    # write the fitted coefficient (or a chosen one) to every station platepar, with the
    # photometric offset compensated so calibrated magnitudes stay the same on average
    python -m Utils.FitPodVignetting <config dirs> --night 20261001 --write
    python -m Utils.FitPodVignetting <config dirs> --set 0.00059
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import json
import os
import shutil
import time

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial import cKDTree

import RMS.ConfigReader as cr
from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP
from RMS.Astrometry.ApplyRecalibrate import loadRecalibratedPlatepar
from RMS.Astrometry.Conversions import date2JD
from RMS.Formats import FFfile
from RMS.Formats.CALSTARS import readCALSTARS
from RMS.Formats.Platepar import Platepar


# Two FF blocks are "simultaneous" when their middle times differ by less than this (s). Half an
# FF block at 25 FPS; a star moves 1.5 arcmin in that time
MAX_PAIR_DT_S = 6.0

# Sky matching radius between the two cameras (deg). Corner astrometry of the recalibrated platepars
# is good to a few px, i.e. a few arcmin
MATCH_RADIUS_DEG = 0.25

# Stars kept at load time. The fit applies its own, stricter S/N cut
LOAD_SNR_MIN = 5.0
FWHM_MAX_PX = 8.0

# Default S/N cut for the fit (both cameras). Above ~20 the FWHM-correlated bias is small
FIT_SNR_MIN = 20.0

# Radius bins for the per-camera residual trend, as fractions of the corner radius
TREND_BIN_FRACTIONS = (0.3, 0.5, 0.65, 0.77, 0.86, 1.0)

# Residual trend above which a camera is flagged as departing from the shared profile (mag), judged
# only on bins with at least TREND_FLAG_MIN_N pairs. Sparse bins at the edge of an overlap strip
# sample one partner's extreme corner and are reported but not used for the flag
TREND_FLAG_MAG = 0.1
TREND_FLAG_MIN_N = 1000


def vignettingLoss(radius, k):
    """ Vignetting loss in magnitudes of the cos^4 model at a given radius.

    Arguments:
        radius: [float or ndarray] Radius from the image centre (px).
        k: [float] Vignetting coefficient (rad/px).

    Return:
        [float or ndarray] Loss (mag), positive = dimmer.
    """

    return -10.0*np.log10(np.clip(np.cos(k*np.asarray(radius, dtype=np.float64)), 1e-6, 1.0))



def _findNightDir(config, night):
    """ Find the archived night directory of a station for a YYYYMMDD night string. """

    arch = os.path.join(os.path.expanduser(config.data_dir), "ArchivedFiles")
    cands = [d for d in sorted(glob.glob(os.path.join(arch, "{:s}_{:s}_*".format(config.stationID, night))))
             if os.path.isdir(d)]

    return cands[0] if cands else None



def _loadBasePlatepar(config, night_dir):
    """ Load the night's platepar, falling back to the station's config-dir platepar. """

    for path in (os.path.join(night_dir, config.platepar_name),
                 os.path.join(config.config_file_path, config.platepar_name)):

        if os.path.isfile(path):
            pp = Platepar()
            pp.read(path, use_flat=None)
            return pp

    return None



def loadNightStars(config, night_dir, snr_min=LOAD_SNR_MIN, fwhm_max=FWHM_MAX_PX, every=1):
    """ Load the CALSTARS detections of one night and put them on the sky.

    Arguments:
        config: [Config] Station config.
        night_dir: [str] Archived night directory with the CALSTARS file.

    Keyword arguments:
        snr_min: [float] Minimum star S/N to keep.
        fwhm_max: [float] Maximum FWHM (px) to keep.
        every: [int] Use every N-th FF file only (speed).

    Return:
        ff_stars: [list of dict] One entry per FF, sorted by time, with keys:
            jd, vec (N x 3 unit vectors), r (px from image centre), mag (instrumental, -2.5 log10 of
            the intensity sum), fwhm, snr.
        Returns an empty list if the night has no CALSTARS file.
    """

    file_list = sorted(os.listdir(night_dir))

    calstars_files = [f for f in file_list if f.startswith("CALSTARS") and f.endswith(".txt")]
    if not calstars_files:
        return []

    star_list, chunk_frames = readCALSTARS(night_dir, calstars_files[0])

    # The flux recalibration covers the whole night in time bins; the meteor one only FFs with
    # detections. Both are only used for sky positions here, so the night's base platepar is a fine
    # fallback (pointing is fixed, recalibration moves by arcminutes)
    recal = loadRecalibratedPlatepar(night_dir, config, file_list, type="flux")
    if not recal:
        recal = loadRecalibratedPlatepar(night_dir, config, file_list, type="meteor")
    recal = recal or {}

    pp_base = _loadBasePlatepar(config, night_dir)
    if (pp_base is None) and (not recal):
        return []

    ff_stars = []

    for ff_name, stars in star_list[::max(1, int(every))]:

        if not len(stars):
            continue

        pp = recal.get(ff_name, pp_base)
        if pp is None:
            continue

        rows = np.array(stars, dtype=np.float64)
        y, x, intens = rows[:, 0], rows[:, 1], rows[:, 2]
        fwhm = rows[:, 4] if rows.shape[1] > 4 else np.full(len(x), -1.0)
        snr = rows[:, 6] if rows.shape[1] > 6 else np.full(len(x), np.inf)
        nsat = rows[:, 7] if rows.shape[1] > 7 else np.zeros(len(x))

        good = (intens > 0) & (snr >= snr_min) & (nsat == 0) & (fwhm > 0) & (fwhm < fwhm_max)
        if not np.any(good):
            continue

        x, y, intens, fwhm, snr = x[good], y[good], intens[good], fwhm[good], snr[good]

        t = FFfile.getMiddleTimeFF(ff_name, config.fps, ret_milliseconds=True, ff_frames=chunk_frames)
        jd = date2JD(*t)

        _, ra, dec, _ = xyToRaDecPP([t]*len(x), x, y, [1]*len(x), pp, extinction_correction=False)
        ra = np.radians(np.array(ra))
        dec = np.radians(np.array(dec))
        vec = np.column_stack([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])

        ff_stars.append({
            "jd": jd,
            "vec": vec,
            "r": np.hypot(x - pp.X_res/2.0, y - pp.Y_res/2.0),
            "mag": -2.5*np.log10(intens),
            "fwhm": fwhm,
            "snr": snr,
            })

    ff_stars.sort(key=lambda d: d["jd"])

    return ff_stars



def pairStations(stars_a, stars_b, max_dt=MAX_PAIR_DT_S, match_radius=MATCH_RADIUS_DEG):
    """ Pair the detections of two cameras: same instant, same sky position.

    Arguments:
        stars_a, stars_b: [list of dict] Output of loadNightStars for the two cameras.

    Keyword arguments:
        max_dt: [float] Maximum FF middle-time difference (s).
        match_radius: [float] Sky matching radius (deg).

    Return:
        pairs: [dict] Arrays r_a, r_b (px), dmag (m_a - m_b, instrumental), fwhm_a, fwhm_b, snr_a,
            snr_b, jd. None if nothing matched.
    """

    if (not stars_a) or (not stars_b):
        return None

    jd_b = np.array([d["jd"] for d in stars_b])
    chord = 2.0*np.sin(np.radians(match_radius)/2.0)

    out = {key: [] for key in ("r_a", "r_b", "dmag", "fwhm_a", "fwhm_b", "snr_a", "snr_b", "jd")}

    for fa in stars_a:

        j = int(np.argmin(np.abs(jd_b - fa["jd"])))
        if abs(jd_b[j] - fa["jd"])*86400.0 > max_dt:
            continue

        fb = stars_b[j]

        dist, idx = cKDTree(fb["vec"]).query(fa["vec"], distance_upper_bound=chord)
        ok = np.isfinite(dist)
        if not np.any(ok):
            continue

        # One-to-one: keep the first (closest, as query returns the nearest) claim on each B star
        ia = np.where(ok)[0]
        ib = idx[ok]
        _, first = np.unique(ib, return_index=True)
        ia, ib = ia[first], ib[first]

        out["r_a"].append(fa["r"][ia])
        out["r_b"].append(fb["r"][ib])
        out["dmag"].append(fa["mag"][ia] - fb["mag"][ib])
        out["fwhm_a"].append(fa["fwhm"][ia])
        out["fwhm_b"].append(fb["fwhm"][ib])
        out["snr_a"].append(fa["snr"][ia])
        out["snr_b"].append(fb["snr"][ib])
        out["jd"].append(np.full(len(ia), fa["jd"]))

    if not out["dmag"]:
        return None

    return {key: np.concatenate(val) for key, val in out.items()}



def _stackPairs(pairs, snr_min):
    """ Stack the per-pair dictionaries into flat arrays with pair and camera indices. """

    keys = sorted(pairs.keys())
    stations = sorted({s for key in keys for s in key})

    cols = {c: [] for c in ("r_a", "r_b", "dmag", "fwhm_a", "fwhm_b")}
    pid, cam_a, cam_b = [], [], []

    for i, key in enumerate(keys):
        p = pairs[key]
        sel = (p["snr_a"] >= snr_min) & (p["snr_b"] >= snr_min)
        for c in cols:
            cols[c].append(p[c][sel])
        n = int(np.count_nonzero(sel))
        pid.append(np.full(n, i))
        cam_a.append(np.full(n, stations.index(key[0])))
        cam_b.append(np.full(n, stations.index(key[1])))

    flat = {c: np.concatenate(v) for c, v in cols.items()}
    flat["pid"] = np.concatenate(pid)
    flat["cam_a"] = np.concatenate(cam_a)
    flat["cam_b"] = np.concatenate(cam_b)

    return keys, stations, flat



def _profileOffsets(resid, pid, n_pairs, mask):
    """ Remove the L1-optimal zero point (inlier median) of every camera pair from the residuals. """

    resid = resid.copy()
    offsets = np.zeros(n_pairs)

    for i in range(n_pairs):
        sel = pid == i
        selm = sel & mask
        if np.any(selm):
            offsets[i] = np.median(resid[selm])
            resid[sel] -= offsets[i]

    return resid, offsets



def _robustFit(resid_fn, p0, pid, n_pairs, bounds=None, n_iter=4):
    """ L1 fit with per-pair zero points profiled out, iterated with 2.5 sigma clipping.

    Arguments:
        resid_fn: [callable] params -> raw residuals (before zero-point removal).
        p0: [list] Initial parameters. A single parameter uses a bounded scalar minimizer.
        pid: [ndarray] Pair index per data point.
        n_pairs: [int] Number of pairs.

    Keyword arguments:
        bounds: [tuple] Bounds for the scalar case.
        n_iter: [int] Clipping iterations.

    Return:
        params, resid, offsets, mask, scatter
    """

    mask = np.ones(len(pid), dtype=bool)
    p = np.array(p0, dtype=np.float64)

    for _ in range(n_iter):

        def objective(q):
            rr, _ = _profileOffsets(resid_fn(np.atleast_1d(q)), pid, n_pairs, mask)
            return np.sum(np.abs(rr[mask]))

        if len(p) == 1:
            p = np.array([minimize_scalar(objective, bounds=bounds, method="bounded",
                                          options={"xatol": 1e-8}).x])
        else:
            p = minimize(objective, p, method="Nelder-Mead",
                         options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-6}).x

        resid, offsets = _profileOffsets(resid_fn(p), pid, n_pairs, mask)
        scatter = 1.4826*np.median(np.abs(resid[mask]))
        mask = np.abs(resid) < 2.5*max(scatter, 1e-3)

    return p, resid, offsets, mask, scatter



def fitPodVignetting(pairs, resolutions, snr_min=FIT_SNR_MIN, fwhm_term=False, per_camera=True):
    """ Fit the shared vignetting coefficient of a pod on cross-camera star pairs.

    Arguments:
        pairs: [dict] {(station_a, station_b): pair arrays} from pairStations.
        resolutions: [dict] {station: (X_res, Y_res)}.

    Keyword arguments:
        snr_min: [float] Minimum S/N in both cameras for a pair to enter the fit.
        fwhm_term: [bool] Also fit a b*(FWHM_a^2 - FWHM_b^2) nuisance term for the faint-star bias.
        per_camera: [bool] Also fit one coefficient per camera (diagnostic only).

    Return:
        result: [dict] k, scatter, n_used, per_pair {key: {zp, scatter, n}}, per_camera_k,
            per_camera_trend {station: {bins_px, median_resid, n}}, flagged (stations whose trend
            exceeds TREND_FLAG_MAG), closure_max (mag), and k_fwhm / fwhm_coeff when fwhm_term.
            None if there are too few pairs.
    """

    keys, stations, d = _stackPairs(pairs, snr_min)
    n_pairs = len(keys)

    if len(d["dmag"]) < 50:
        return None

    r_a, r_b, dmag, pid = d["r_a"], d["r_b"], d["dmag"], d["pid"]

    def residK(p):
        return dmag - (vignettingLoss(r_a, abs(p[0])) - vignettingLoss(r_b, abs(p[0])))

    k_arr, resid, offsets, mask, scatter = _robustFit(residK, [6e-4], pid, n_pairs, bounds=(1e-5, 1.5e-3))
    k = abs(float(k_arr[0]))

    result = {
        "k": k,
        "scatter": float(scatter),
        "n_used": int(np.count_nonzero(mask)),
        "n_pairs_total": int(len(dmag)),
        "snr_min": snr_min,
        "stations": stations,
        "per_pair": {},
        "per_camera_k": None,
        "per_camera_trend": {},
        "flagged": [],
        "closure_max": None,
        "k_fwhm": None,
        "fwhm_coeff": None,
        }

    for i, key in enumerate(keys):
        sel = mask & (pid == i)
        result["per_pair"]["{:s}-{:s}".format(*key)] = {
            "zp": float(offsets[i]),
            "scatter": float(1.4826*np.median(np.abs(resid[sel]))) if np.any(sel) else None,
            "n": int(np.count_nonzero(sel)),
            }

    # Zero-point closure around every triangle of cameras: zp_ab + zp_bc - zp_ac should vanish
    zp = {key: offsets[i] for i, key in enumerate(keys)}
    closures = []
    for a in stations:
        for b in stations:
            for c in stations:
                if (a < b < c) and ((a, b) in zp) and ((b, c) in zp) and ((a, c) in zp):
                    closures.append(abs(zp[(a, b)] + zp[(b, c)] - zp[(a, c)]))
    if closures:
        result["closure_max"] = float(max(closures))

    # Per-camera residual trend vs own radius under the shared coefficient. A camera appears as
    # "a" in some pairs and "b" in others; flip the sign so + always means the camera is dimmer
    # than the model
    for ci, station in enumerate(stations):

        is_a = d["cam_a"] == ci
        is_b = d["cam_b"] == ci
        sel_cam = (is_a | is_b) & mask
        own_r = np.where(is_a, r_a, r_b)
        signed = np.where(is_a, resid, -resid)

        x_res, y_res = resolutions[station]
        r_corner = np.hypot(x_res/2.0, y_res/2.0)
        edges = np.array(TREND_BIN_FRACTIONS)*r_corner

        medians, counts = [], []
        for j in range(len(edges) - 1):
            sel = sel_cam & (own_r >= edges[j]) & (own_r < edges[j + 1])
            counts.append(int(np.count_nonzero(sel)))
            medians.append(float(np.median(signed[sel])) if counts[-1] >= 200 else None)

        result["per_camera_trend"][station] = {
            "bins_px": [float(e) for e in edges],
            "median_resid": medians,
            "n": counts,
            }

        valid = [m for m, n in zip(medians, counts) if (m is not None) and (n >= TREND_FLAG_MIN_N)]
        if (len(valid) >= 2) and (max(valid) - min(valid) > TREND_FLAG_MAG):
            result["flagged"].append(station)

    if per_camera:

        cam_a, cam_b = d["cam_a"], d["cam_b"]

        def residPerCam(p):
            ks = np.abs(p)
            return dmag - (vignettingLoss(r_a, ks[cam_a]) - vignettingLoss(r_b, ks[cam_b]))

        p_cam, _, _, _, scatter_cam = _robustFit(residPerCam, [k]*len(stations), pid, n_pairs)
        result["per_camera_k"] = {s: float(abs(v)) for s, v in zip(stations, p_cam)}
        result["per_camera_scatter"] = float(scatter_cam)

    if fwhm_term:

        f2 = d["fwhm_a"]**2 - d["fwhm_b"]**2

        def residFwhm(p):
            return residK(p) - p[1]*f2

        p_f, _, _, _, scatter_f = _robustFit(residFwhm, [k, 0.0], pid, n_pairs)
        result["k_fwhm"] = float(abs(p_f[0]))
        result["fwhm_coeff"] = float(p_f[1])
        result["scatter_fwhm"] = float(scatter_f)

    return result



def magLevShift(star_xy, x_res, y_res, k_old, k_new):
    """ Change of the photometric offset that keeps calibrated magnitudes unchanged on average when
        the vignetting coefficient changes.

        mag = -2.5 log10(I) - V(r, k) + mag_lev, so a smaller k (less correction) needs a smaller
        mag_lev by the mean of V(r, k_old) - V(r, k_new) over the stars the offset was fitted on.

    Arguments:
        star_xy: [ndarray or None] N x 2 image positions of the photometry stars. If None or fewer
            than 10 stars, a uniform distribution over the image area is used instead.
        x_res, y_res: [int] Image size.
        k_old, k_new: [float] Old and new vignetting coefficients (rad/px).

    Return:
        [float] Amount to ADD to mag_lev (negative when k decreases).
    """

    if (star_xy is None) or (len(star_xy) < 10):
        xs = np.linspace(0.5, x_res - 0.5, 192)
        ys = np.linspace(0.5, y_res - 0.5, 108)
        gx, gy = np.meshgrid(xs, ys)
        star_xy = np.column_stack([gx.ravel(), gy.ravel()])

    star_xy = np.asarray(star_xy, dtype=np.float64)
    r = np.hypot(star_xy[:, 0] - x_res/2.0, star_xy[:, 1] - y_res/2.0)

    return float(-np.mean(vignettingLoss(r, k_old) - vignettingLoss(r, k_new)))



def applyVignettingToPlatepars(k, configs, backup=True):
    """ Write a fixed vignetting coefficient into every station's live platepar, compensating the
        photometric offset.

    Arguments:
        k: [float] Vignetting coefficient (rad/px).
        configs: [list of Config] Station configs; the platepar is config_file_path/platepar_name.

    Keyword arguments:
        backup: [bool] Copy the platepar to <name>.bak.<timestamp> first.

    Return:
        changes: [list of dict] stationID, path, k_old, k_new, mag_lev_old, mag_lev_new, backup.
    """

    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    changes = []

    for config in configs:

        path = os.path.join(config.config_file_path, config.platepar_name)
        if not os.path.isfile(path):
            continue

        pp = Platepar()
        pp.read(path, use_flat=None)

        # Radii of the stars the offset was fitted on; with fewer than 10 the shift falls back to a
        # uniform image-area average (reported as 0 offset stars)
        star_xy = None
        star_list = getattr(pp, "star_list", None)
        if star_list:
            try:
                star_xy = np.array([[row[1], row[2]] for row in star_list], dtype=np.float64)
            except (IndexError, TypeError, ValueError):
                star_xy = None
        if (star_xy is not None) and (len(star_xy) < 10):
            star_xy = None

        k_old = pp.vignetting_coeff if pp.vignetting_coeff is not None else 0.0
        shift = magLevShift(star_xy, pp.X_res, pp.Y_res, k_old, k)

        backup_path = None
        if backup:
            backup_path = "{:s}.bak.{:s}".format(path, stamp)
            shutil.copy2(path, backup_path)

        mag_lev_old = pp.mag_lev
        pp.vignetting_coeff = float(k)
        pp.vignetting_fixed = True
        pp.mag_lev = float(mag_lev_old + shift)
        pp.write(path)

        changes.append({
            "stationID": config.stationID,
            "path": path,
            "k_old": float(k_old),
            "k_new": float(k),
            "mag_lev_old": float(mag_lev_old),
            "mag_lev_new": float(pp.mag_lev),
            "n_offset_stars": int(len(star_xy)) if star_xy is not None else 0,
            "backup": backup_path,
            })

    return changes



def printReport(result, resolutions, platepar_k=None):
    """ Print a human-readable fit report. """

    any_res = next(iter(resolutions.values()))
    r_corner = np.hypot(any_res[0]/2.0, any_res[1]/2.0)

    print("Pod vignetting fit on {:d} of {:d} pairs (S/N >= {:g} in both cameras)".format(
        result["n_used"], result["n_pairs_total"], result["snr_min"]))
    print("  shared k = {:.6f} rad/px   corner loss {:.2f} mag   scatter {:.3f} mag".format(
        result["k"], vignettingLoss(r_corner, result["k"]), result["scatter"]))

    if result.get("k_fwhm") is not None:
        print("  with FWHM^2 term: k = {:.6f} (corner {:.2f} mag), b = {:+.4f} mag/px^2, scatter {:.3f}".format(
            result["k_fwhm"], vignettingLoss(r_corner, result["k_fwhm"]), result["fwhm_coeff"],
            result["scatter_fwhm"]))

    if result.get("per_camera_k"):
        print("  per-camera k (diagnostic): " + "  ".join(
            "{:s}={:.6f}".format(s, v) for s, v in sorted(result["per_camera_k"].items())))

    if platepar_k:
        print("  platepar k now:            " + "  ".join(
            "{:s}={:.6f}".format(s, v) for s, v in sorted(platepar_k.items()) if v is not None))

    print()
    print("  pair      zero point   scatter      n")
    for key, p in sorted(result["per_pair"].items()):
        print("  {:9s}  {:+8.3f}     {:6.3f}  {:7d}".format(
            key, p["zp"], p["scatter"] if p["scatter"] is not None else float("nan"), p["n"]))
    if result["closure_max"] is not None:
        print("  worst zero-point closure around a camera triangle: {:.3f} mag".format(result["closure_max"]))

    print()
    print("  per-camera residual vs own radius under the shared k (median mag, + = dimmer than model)")
    for station, tr in sorted(result["per_camera_trend"].items()):
        cells = []
        for j, m in enumerate(tr["median_resid"]):
            lo, hi = tr["bins_px"][j], tr["bins_px"][j + 1]
            cells.append("{:4.0f}-{:4.0f}: {:s}".format(lo, hi, "{:+.3f}".format(m) if m is not None else "   -  "))
        flag = "   <-- departs from the shared profile" if station in result["flagged"] else ""
        print("  {:8s} ".format(station) + "  ".join(cells) + flag)



def runPod(config_dirs, night, snr_min=FIT_SNR_MIN, fwhm_term=False, every=1):
    """ Load, pair and fit one night for a list of station config directories.

    Return:
        (result, resolutions, platepar_k, configs): the fit result (or None), image sizes, the
        current platepar coefficient per station, and the loaded configs.
    """

    configs = [cr.loadConfigFromDirectory(".", d) for d in config_dirs]

    stars = {}
    resolutions = {}
    platepar_k = {}

    for config in configs:

        night_dir = _findNightDir(config, night)
        if night_dir is None:
            print("{:s}: no archived night {:s}".format(config.stationID, night))
            continue

        pp = _loadBasePlatepar(config, night_dir)
        if pp is None:
            print("{:s}: no platepar".format(config.stationID))
            continue

        resolutions[config.stationID] = (pp.X_res, pp.Y_res)
        platepar_k[config.stationID] = pp.vignetting_coeff

        ff_stars = loadNightStars(config, night_dir, every=every)
        print("{:s}: {:d} FF files with stars".format(config.stationID, len(ff_stars)))
        if ff_stars:
            stars[config.stationID] = ff_stars

    pairs = {}
    ids = sorted(stars.keys())
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            p = pairStations(stars[a], stars[b])
            if p is not None:
                pairs[(a, b)] = p
                print("{:s}-{:s}: {:d} simultaneous star pairs".format(a, b, len(p["dmag"])))

    if not pairs:
        return None, resolutions, platepar_k, configs

    result = fitPodVignetting(pairs, resolutions, snr_min=snr_min, fwhm_term=fwhm_term)

    return result, resolutions, platepar_k, configs



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fit one vignetting coefficient for a camera pod from "
                                                 "stars seen simultaneously by two cameras")
    parser.add_argument("config_dirs", nargs="+", help="station config directories (one per camera)")
    parser.add_argument("--night", help="YYYYMMDD night to fit (required unless --set is used alone)")
    parser.add_argument("--snr", type=float, default=FIT_SNR_MIN, help="minimum S/N in both cameras")
    parser.add_argument("--every", type=int, default=1, help="use every N-th FF file (speed)")
    parser.add_argument("--fwhm-term", action="store_true", help="also fit the FWHM^2 nuisance term")
    parser.add_argument("--json", default=None, help="write the fit report to this JSON file")
    parser.add_argument("--write", action="store_true",
                        help="write the fitted k into every station platepar (fixed), compensating mag_lev")
    parser.add_argument("--set", type=float, default=None, metavar="K",
                        help="write this k instead of fitting (no night needed)")
    parser.add_argument("--no-backup", action="store_true", help="do not back up platepars before writing")
    args = parser.parse_args()

    result = None
    configs = None

    if args.night:
        result, resolutions, platepar_k, configs = runPod(args.config_dirs, args.night, snr_min=args.snr,
                                                          fwhm_term=args.fwhm_term, every=args.every)
        if result is None:
            print("Not enough cross-camera pairs to fit")
        else:
            print()
            printReport(result, resolutions, platepar_k)

            if args.json:
                with open(args.json, "w") as f:
                    json.dump(result, f, indent=1)
                print("report written to {:s}".format(args.json))

    elif args.set is None:
        parser.error("--night is required unless --set is given")

    k_write = None
    if args.set is not None:
        k_write = args.set
    elif args.write and (result is not None):
        k_write = result["k"]

    if k_write is not None:

        if configs is None:
            configs = [cr.loadConfigFromDirectory(".", d) for d in args.config_dirs]

        changes = applyVignettingToPlatepars(k_write, configs, backup=not args.no_backup)

        print()
        print("Platepars updated with vignetting_coeff = {:.6f} (fixed):".format(k_write))
        for ch in changes:
            sample = "{:d} offset stars".format(ch["n_offset_stars"]) if ch["n_offset_stars"] else "image-area mean"
            print("  {:s}: k {:.6f} -> {:.6f}, mag_lev {:.3f} -> {:.3f} ({:s}){:s}".format(
                ch["stationID"], ch["k_old"], ch["k_new"], ch["mag_lev_old"], ch["mag_lev_new"],
                sample, "  backup " + ch["backup"] if ch["backup"] else ""))
