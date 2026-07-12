""" Fit the site light-dome limiting magnitude model from archived nights.

Builds (star, frame) hit/miss trials from the flux-recalibrated platepars of one or more
co-located stations: every catalog star predicted inside a frame's FOV (spherical polygon
guarded, mask filtered) is one binary trial - matched within the match radius or not.
The site model (see RMS.LightDomeModel) is then fitted by maximum likelihood with a
logistic detection probability, so non-detections carry information and no binning or
censoring bias enters.

Fit on CLEAR nights: pass explicit dates with --nights, or let the tool pick the
clearest recent nights per station (highest median matched-star count). Cloudy training
nights bias the model shallow; when in doubt, pass known-clear dates.

The fitted model is written as <stationID>_light_dome.json into every station's data
directory, one identical site file per station.

Usage:
    python -m Utils.FitLightDome /path/to/station_config_dir1 [dir2 ...] \\
        --nights 20260708,20260709

Note: the model absorbs the CURRENT intensity_threshold/focus/gain of each camera into
its LM0. Refit after retuning a camera, refocusing, or re-aiming.
"""

from __future__ import absolute_import, division, print_function

import argparse
import datetime
import json
import os
import re

import ephem
import numpy as np
from scipy.optimize import minimize

import RMS.ConfigReader as cr
from RMS.Formats import FFfile, StarCatalog
from RMS.Formats.Platepar import Platepar
from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, xyToRaDecPP
from RMS.Astrometry.Conversions import date2JD, jd2Date, raDec2AltAz
from RMS.Routines.MaskImage import getMaskFile
from RMS.Math import pointInsideConvexPolygonSphere
from RMS.LightDomeModel import DOME_CATALOG_LIM_MAG, LIGHT_DOME_FILE_SUFFIX, LightDomeModel


# Trial building
MATCH_RADIUS_PX = 3.0    # px - predicted star counts as matched within this radius
BORDER_PX = 10           # px - skip stars near the frame border (extractor border)
ALT_MIN = 5.0            # deg - ignore sky below this altitude
FF_PER_NIGHT = 40        # frames sampled per night per station
SUN_ALT_MAX = -18.0      # deg - only fully dark frames (twilight frames train the model
                         # on a temporarily bright sky and bias it shallow, which makes
                         # cloudy nights pass the ratio threshold)
MOON_PHASE_MAX = 25.0    # percent illumination - frames with a brighter moon above the
                         # horizon are excluded: scattered moonlight brightens the whole
                         # sky, which the static site model would absorb as permanent
                         # light pollution. Matches the detectMoon phase convention.

# Fitting
MIN_DOME_SIGNIFICANCE = 50.0   # NLL improvement required to accept one more harmonic order


def sunAltitude(jd, lat, lon):
    """ Solar altitude (deg) at the given time and location. """

    obs = ephem.Observer()
    obs.lat = str(lat)
    obs.lon = str(lon)
    obs.date = jd2Date(jd, dt_obj=True)

    sun = ephem.Sun()
    sun.compute(obs)

    return np.degrees(float(sun.alt))


def moonAltPhase(jd, lat, lon):
    """ Lunar altitude (deg) and illuminated fraction (percent) at the given time and
        location. """

    obs = ephem.Observer()
    obs.lat = str(lat)
    obs.lon = str(lon)
    obs.date = jd2Date(jd, dt_obj=True)

    moon = ephem.Moon()
    moon.compute(obs)

    return np.degrees(float(moon.alt)), float(moon.phase)


def buildStationTrials(config, night_dirs, n_ff=FF_PER_NIGHT, moon_phase_max=MOON_PHASE_MAX):
    """ Build hit/miss trials for one station over the given night directories.

    Arguments:
        config: [Config] Station config.
        night_dirs: [list of str] Paths to night directories with
            platepars_flux_recalibrated.json.

    Keyword arguments:
        n_ff: [int] Frames sampled per night.
        moon_phase_max: [float] Exclude frames with a moon above the horizon illuminated
            more than this (percent). 100 disables the filter.

    Return:
        trials: [dict of ndarray] az, alt, mag, det - or None if no usable data.
    """

    n_moon_excluded = 0
    footprint = None

    catalog_stars, _, _ = StarCatalog.readStarCatalog(config.star_catalog_path,
        config.star_catalog_file, lim_mag=DOME_CATALOG_LIM_MAG,
        mag_band_ratios=config.star_catalog_band_ratios)
    cat_ra, cat_dec, cat_mag = catalog_stars[:, 0], catalog_stars[:, 1], catalog_stars[:, 2]

    az_all, alt_all, mag_all, det_all = [], [], [], []

    for night_dir in night_dirs:

        pp_path = os.path.join(night_dir, "platepars_flux_recalibrated.json")
        if not os.path.isfile(pp_path):
            continue

        with open(pp_path) as f:
            ppr = json.load(f)

        ff_names = sorted(k for k in ppr if k.startswith("FF_") and isinstance(ppr[k], dict)
            and ppr[k].get("auto_recalibrated") and ppr[k].get("star_list"))

        if len(ff_names) < 3:
            continue

        mask = getMaskFile(night_dir, config, default_as_backup=True)

        indices = np.unique(np.linspace(0, len(ff_names) - 1,
            min(n_ff, len(ff_names))).astype(int))

        for ff_name in [ff_names[i] for i in indices]:

            pp = Platepar()
            pp.loadFromDict(ppr[ff_name], use_flat=config.use_flat)
            w, h = pp.X_res, pp.Y_res

            if mask is not None:
                mask.checkMask(w, h)

            date = FFfile.getMiddleTimeFF(ff_name, config.fps, ret_milliseconds=True)
            jd = date2JD(*date)

            # Fully dark frames only
            if sunAltitude(jd, pp.lat, pp.lon) > SUN_ALT_MAX:
                continue

            # Exclude moonlit frames: scattered moonlight brightens the whole sky and a
            # static site model would absorb it as permanent light pollution
            moon_alt, moon_phase = moonAltPhase(jd, pp.lat, pp.lon)
            if (moon_alt > 0) and (moon_phase > moon_phase_max):
                n_moon_excluded += 1
                continue

            # Spherical FOV polygon guard - without it, catalog stars far outside the FOV
            # fold back to in-frame pixel coordinates through the distortion polynomial
            x_vert = [0, w/4, w/2, 3*w/4, w, w, w, w, w, 3*w/4, w/2, w/4, 0, 0, 0, 0, 0]
            y_vert = [0, 0, 0, 0, 0, h/4, h/2, 3*h/4, h, h, h, h, h, 3*h/4, h/2, h/4, 0]
            _, ra_vert, dec_vert, _ = xyToRaDecPP([date]*len(x_vert),
                list(reversed(x_vert)), list(reversed(y_vert)), [1]*len(x_vert), pp,
                extinction_correction=False, precompute_pointing_corr=True)

            inside = pointInsideConvexPolygonSphere(np.array([cat_ra, cat_dec]).T,
                np.array([ra_vert, dec_vert]).T)

            # FOV footprint on the sky (az/alt border polygon), captured once at a
            # consistent epoch - the polygon RA/Dec and the jd MUST come from the same
            # frame, or the sidereal mismatch smears the footprint
            if footprint is None:
                fp_az, fp_alt = raDec2AltAz(np.array(ra_vert), np.array(dec_vert), jd,
                    pp.lat, pp.lon)
                footprint = [[round(float(a), 2) for a in fp_az],
                             [round(float(a), 2) for a in fp_alt]]

            x, y = raDecToXYPP(cat_ra[inside], cat_dec[inside], jd, pp)
            m = cat_mag[inside]
            ra_in = cat_ra[inside]
            dec_in = cat_dec[inside]

            in_frame = (x >= BORDER_PX) & (x < w - BORDER_PX) \
                & (y >= BORDER_PX) & (y < h - BORDER_PX)

            if mask is not None:
                xi = np.clip(x.astype(int), 0, w - 1)
                yi = np.clip(y.astype(int), 0, h - 1)
                in_frame &= mask.img[yi, xi] > 0

            x, y, m = x[in_frame], y[in_frame], m[in_frame]
            az, alt = raDec2AltAz(ra_in[in_frame], dec_in[in_frame], jd, pp.lat, pp.lon)

            up = alt >= ALT_MIN
            x, y, m, az, alt = x[up], y[up], m[up], az[up], alt[up]

            if not len(x):
                continue

            # Matched star positions (star_list rows are [jd, x, y, ...] - detect a
            # transposed variant by coordinate range)
            star_list = np.array(ppr[ff_name]["star_list"])
            c1, c2 = star_list[:, 1], star_list[:, 2]
            if (c1.max() <= h + 1) and (c2.max() > h + 1):
                sx, sy = c2, c1
            else:
                sx, sy = c1, c2

            dist2 = (x[:, None] - sx[None, :])**2 + (y[:, None] - sy[None, :])**2
            detected = (dist2.min(axis=1) <= MATCH_RADIUS_PX**2).astype(np.int8)

            az_all.append(az)
            alt_all.append(alt)
            mag_all.append(m)
            det_all.append(detected)

    if n_moon_excluded:
        print("  moon filter: excluded {:d} frame(s) (moon up, phase > {:.0f}%)".format(
            n_moon_excluded, moon_phase_max))

    if not az_all:
        return None

    return dict(az=np.concatenate(az_all), alt=np.concatenate(alt_all),
        mag=np.concatenate(mag_all), det=np.concatenate(det_all),
        pointing=(float(pp.az_centre), float(pp.alt_centre)),
        footprint=footprint)


def selectNightDirs(config, dates=None, window=30, max_nights=4, min_rel_clarity=None):
    """ Pick night directories to fit on: explicit dates, or the clearest recent nights
        (highest median matched-star count over the trailing window).

    Arguments:
        config: [Config] Station config (data_dir locates ArchivedFiles).

    Keyword arguments:
        dates: [list of str] YYYYMMDD dates. If given, all segments of those dates.
        window: [int] Trailing night directories to consider for auto-selection.
        max_nights: [int] Number of auto-selected nights.
        min_rel_clarity: [float] If given, only keep nights whose median matched count is
            at least this fraction of the best selected night's - excludes cloudy nights
            from the training set even when few nights are available (training on cloudy
            frames biases the model shallow, which can read cloudy nights as clear).

    Return:
        night_dirs: [list of str] Paths to selected night directories.
    """

    archive_dir = os.path.join(os.path.expanduser(config.data_dir), "ArchivedFiles")

    if not os.path.isdir(archive_dir):
        return []

    all_dirs = sorted(d for d in os.listdir(archive_dir)
        if d.startswith(str(config.stationID) + "_")
        and os.path.isdir(os.path.join(archive_dir, d)))

    if dates is not None:
        # Locate the date chunk by pattern, not position - robust to any stationID scheme
        matched = []
        for d in all_dirs:
            m = re.search(r"_(\d{8})(?:_|$)", d)
            if m and (m.group(1) in dates):
                matched.append(os.path.join(archive_dir, d))
        return matched

    # Auto-select: median matched count per night dir, take the top max_nights
    scored = []
    for d in all_dirs[-window:]:
        pp_path = os.path.join(archive_dir, d, "platepars_flux_recalibrated.json")
        if not os.path.isfile(pp_path):
            continue
        try:
            with open(pp_path) as f:
                ppr = json.load(f)
        except Exception:
            continue
        # Only successfully recalibrated frames count: failed entries carry the SkyFit
        # calibration star_list (a fossil of a clear night), which makes fully cloudy
        # nights look clear to this selector
        counts = [len(v["star_list"]) for k, v in ppr.items()
                  if k.startswith("FF_") and isinstance(v, dict) and v.get("star_list")
                  and v.get("auto_recalibrated")]
        if len(counts) >= 10:
            scored.append((float(np.median(counts)), d))

    scored.sort(reverse=True)
    scored = scored[:max_nights]

    if min_rel_clarity is not None and scored:
        best = scored[0][0]
        scored = [(s, d) for s, d in scored if s >= min_rel_clarity*best]

    return [os.path.join(archive_dir, d) for _, d in scored]


def fitLightDome(station_configs, dates=None, max_order=3, moon_phase_max=MOON_PHASE_MAX):
    """ Fit the site model over multiple co-located stations.

    Arguments:
        station_configs: [list of Config] One config per station.

    Keyword arguments:
        dates: [list of str] Explicit YYYYMMDD training dates (recommended: clear nights).
        max_order: [int] Maximum azimuthal harmonic order to try (1=dipole, 2=quadrupole,
            ...). Each order is kept only if it clears the significance gate.
        moon_phase_max: [float] Exclude frames with a moon above the horizon illuminated
            more than this (percent). 100 disables the filter.

    Return:
        model_dict: [dict] Fitted site model (LightDomeModel format), or None.
    """

    cams = [str(c.stationID) for c in station_configs]
    ncam = len(cams)

    az_l, alt_l, mag_l, det_l, ci_l = [], [], [], [], []
    fit_nights = []
    pointing = {}
    thresholds = {}
    footprints = {}

    for i, config in enumerate(station_configs):

        night_dirs = selectNightDirs(config, dates=dates)
        fit_nights += [os.path.basename(d) for d in night_dirs]

        trials = buildStationTrials(config, night_dirs, moon_phase_max=moon_phase_max)

        if trials is None:
            print("{:s}: no usable trials, station excluded from fit".format(cams[i]))
            continue

        n = len(trials["az"])
        print("{:s}: {:d} trials from {:d} night dir(s), detected fraction {:.2f}".format(
            cams[i], n, len(night_dirs), float(np.mean(trials["det"]))))

        az_l.append(trials["az"])
        alt_l.append(trials["alt"])
        mag_l.append(trials["mag"])
        det_l.append(trials["det"])
        ci_l.append(np.full(n, i, dtype=np.int64))

        # Metadata for staleness detection (see ensureLightDomeModel) and the model plot
        pointing[cams[i]] = list(trials["pointing"])
        thresholds[cams[i]] = float(config.intensity_threshold)
        footprints[cams[i]] = trials["footprint"]

    if not az_l:
        return None

    az = np.concatenate(az_l)
    alt = np.concatenate(alt_l)
    mag = np.concatenate(mag_l)
    det = np.concatenate(det_l)
    ci = np.concatenate(ci_l)

    print("TOTAL: {:d} trials".format(len(az)))

    def modelLM(p, azv, altv, civ, norder):
        # Harmonics basis: LP = bowl + sum_k A_k*cos(k*(az - phi_k))*exp(-alt/h_k),
        # clamped non-negative (the cosine terms go negative on the dark side). The
        # dipole (k=1) is the leading term for an observer EMBEDDED in the glow field;
        # for a distant city, higher orders approximate a localized dome.
        k = p[ncam]
        q0, hb = p[ncam + 2], p[ncam + 3]
        alt_c = np.clip(altv, 5.0, 90.0)
        z2 = np.sin(np.radians(90.0 - alt_c))**2
        vr = 1.0/np.sqrt(1.0 - 0.97*z2)
        lp = (10.0**q0)*np.exp(-alt_c/hb)
        for j in range(norder):
            qa, phi, hh = p[ncam + 4 + 3*j:ncam + 4 + 3*j + 3]
            lp = lp + (10.0**qa)*np.cos(np.radians((j + 1)*(azv - phi)))*np.exp(-alt_c/hh)
        B = vr + np.maximum(lp, 0.0)
        return p[civ] - k*(1.0/np.sin(np.radians(alt_c)) - 1.0) - 1.25*np.log10(B)

    def nll(p, norder):
        s = p[ncam + 1]
        pr = 1.0/(1.0 + np.exp(-(modelLM(p, az, alt, ci, norder) - mag)/s))
        pr = np.clip(pr, 1e-6, 1 - 1e-6)
        return -np.sum(det*np.log(pr) + (1 - det)*np.log(1 - pr))

    base_bounds = [(4.0, 7.0)]*ncam + [(0.0, 1.5), (0.12, 1.2), (-2.0, 3.0), (5.0, 60.0)]
    order_bounds = [(-2.0, 3.5), (-360.0, 720.0), (3.0, 60.0)]   # log10 A, phase, alt scale

    results = {}
    for norder in range(max_order + 1):

        base0 = [5.5]*ncam + [0.2, 0.35, 1.0, 20.0]

        if norder == 0:
            starts = [base0]
        elif norder == 1:
            # Dipole phase multi-start around the compass
            starts = [base0 + [1.0, phi, 20.0] for phi in (0, 90, 180, 270)]
        else:
            # Seed from the previous order's best fit; phase inits offset within the
            # order's periodicity
            prev = list(results[norder - 1]["p"])
            period = 360.0/norder
            starts = [prev + [0.7, phi, 15.0] for phi in (0.0, period/2)]

        bounds = base_bounds + order_bounds*norder

        best = None
        for p0 in starts:
            r = minimize(nll, np.array(p0), args=(norder,), method="L-BFGS-B",
                bounds=bounds, options=dict(maxiter=120))
            if (best is None) or (r.fun < best.fun):
                best = r

        r = minimize(nll, best.x, args=(norder,), method="L-BFGS-B", bounds=bounds,
            options=dict(maxiter=500))

        results[norder] = {"p": r.x, "nll": float(r.fun)}
        print("order={:d}  NLL={:.0f}".format(norder, r.fun))

    # Keep adding harmonic orders only while each one earns its keep
    use = 0
    while (use + 1 in results) \
            and (results[use]["nll"] - results[use + 1]["nll"] > MIN_DOME_SIGNIFICANCE):
        use += 1

    p = results[use]["p"]

    harmonics = []
    for j in range(use):
        qa, phi, hh = p[ncam + 4 + 3*j:ncam + 4 + 3*j + 3]
        harmonics.append(dict(order=j + 1, A=float(10.0**qa),
            phi=float(phi%(360.0/(j + 1))), h=float(hh)))

    model_dict = dict(
        cams=cams,
        LM0=[float(p[i]) for i in range(ncam)],
        k=float(p[ncam]),
        s=float(p[ncam + 1]),
        q0=float(p[ncam + 2]),
        h0=float(p[ncam + 3]),
        norder=use,
        harmonics=harmonics,
        model="vanrhijn_harmonics",
        fit_nights=sorted(set(fit_nights)),
        fit_date=datetime.datetime.utcnow().strftime("%Y-%m-%d"),
        n_trials=int(len(az)),
        pointing=pointing,
        intensity_threshold=thresholds,
        footprints=footprints,
        nll={str(kk): v["nll"] for kk, v in results.items()},
    )

    print("\nFitted model ({:d} harmonic order(s)):".format(use))
    for i, cam in enumerate(cams):
        print("  LM0[{:s}] = {:.2f}".format(cam, model_dict["LM0"][i]))
    print("  k={:.2f}  s={:.2f}  LP bowl B={:.1f} (h0={:.0f} deg)".format(
        model_dict["k"], model_dict["s"], 10.0**model_dict["q0"], model_dict["h0"]))
    for h in harmonics:
        name = {1: "dipole", 2: "quadrupole", 3: "octupole"}.get(h["order"],
            "order {:d}".format(h["order"]))
        print("  {:s}: A={:.1f}  toward az {:.0f} deg  alt_scale={:.0f} deg".format(
            name, h["A"], h["phi"], h["h"]))

    # Per-camera calibration sanity: aggregate detected/expected on the training trials
    # must be ~1.00 for every camera; a deviation means the shared terms misallocate
    print("Training calibration (detected/expected per camera, want ~1.00):")
    s_soft = model_dict["s"]
    lm_all = modelLM(p, az, alt, ci, use)
    p_det = 1.0/(1.0 + np.exp(-(lm_all - mag)/s_soft))
    for i, cam in enumerate(cams):
        sel = ci == i
        if np.any(sel):
            print("  {:s}: {:.2f}".format(cam, float(np.sum(det[sel])/np.sum(p_det[sel]))))

    return model_dict


# Model plot (see renderLightDomeModel)
NATURAL_ZENITH_SQM = 21.8   # mag/arcsec^2 - moonless natural zenith sky reference used to
                            # express the model's relative brightness in absolute units.
                            # NOTE: the absolute anchor is approximate (star detectability
                            # cannot fully separate camera depth from sky brightness);
                            # the relative structure and the fixed scale are the point.
PLOT_SQM_MIN = 16.0         # fixed color scale so every station and site renders
PLOT_SQM_MAX = 22.0         # comparably (brighter sky = brighter color)


def renderLightDomeModel(model_dict, station_id, out_path):
    """ Render the fitted site model as a sky-brightness hemisphere map.

    Fixed absolute color scale (PLOT_SQM_MIN..MAX mag/arcsec^2, light pollution bright)
    so plots from different stations and sites are directly comparable. The station's own
    FOV footprint is highlighted; co-located sister cameras are drawn in grey for context.

    Arguments:
        model_dict: [dict] Fitted model (LightDomeModel format, with footprints metadata).
        station_id: [str] Station to highlight.
        out_path: [str] Output PNG path.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    model = LightDomeModel(model_dict)

    fig = plt.figure(figsize=(9.5, 8.2))
    ax = fig.add_subplot(1, 1, 1, projection="polar")

    az_grid = np.radians(np.arange(0, 361, 2))
    alt_grid = np.arange(5, 86, 2)
    az_mesh, alt_mesh = np.meshgrid(np.degrees(az_grid), alt_grid)

    sqm = NATURAL_ZENITH_SQM - 2.5*np.log10(model.skyBrightness(az_mesh, alt_mesh))

    cmap = cm.get_cmap("inferno_r")
    pc = ax.pcolormesh(az_grid, 90 - alt_grid, sqm, cmap=cmap,
        vmin=PLOT_SQM_MIN, vmax=PLOT_SQM_MAX, shading="auto")

    # FOV footprints: this station highlighted, sisters grey
    for cam, fp in sorted(model_dict.get("footprints", {}).items()):
        fp_az = np.array(fp[0])
        fp_r = 90 - np.clip(np.array(fp[1]), 5, 90)

        if str(cam) == str(station_id):
            ax.plot(np.radians(fp_az), fp_r, "-", lw=2.6, color="#00ff88", zorder=6)

            # Label at the footprint centroid (circular mean azimuth), safely inside
            mean_az = np.degrees(np.arctan2(np.mean(np.sin(np.radians(fp_az))),
                                            np.mean(np.cos(np.radians(fp_az)))))
            ax.annotate(str(cam), (np.radians(mean_az), float(np.mean(fp_r))),
                color="#00ff88", fontsize=12, fontweight="bold",
                ha="center", va="center", zorder=6)
        else:
            ax.plot(np.radians(fp_az), fp_r, "-", lw=1.0, color="#999999",
                alpha=0.6, zorder=5)
            ax.annotate(str(cam), (np.radians(fp_az[len(fp_az)//2]),
                float(fp_r[len(fp_az)//2])), color="#aaaaaa", fontsize=6, alpha=0.9,
                zorder=5)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 85)
    ax.set_rticks([20, 40, 60, 80])
    ax.set_yticklabels(["70", "50", "30", "10"], fontsize=7)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"], fontsize=11)

    cb = fig.colorbar(pc, ax=ax, shrink=0.8, pad=0.09,
        ticks=np.arange(PLOT_SQM_MIN, PLOT_SQM_MAX + 0.1, 1.0))
    cb.set_label("sky brightness (mag/arcsec$^2$) - brighter sky $\\rightarrow$ brighter color",
        fontsize=9)

    zenith_sqm = float(NATURAL_ZENITH_SQM - 2.5*np.log10(model.skyBrightness(0.0, 90.0)))
    lm0 = model.lm0_map.get(str(station_id), model.lm0_default)
    fit_kind = "auto-fit" if model_dict.get("auto_fitted") else "site fit"

    # Describe the directional glow structure in plain terms, whichever basis fitted it
    parts = []
    for h in model_dict.get("harmonics", []):
        name = {1: "dipole", 2: "quadrupole"}.get(int(h["order"]),
            "order {:d}".format(int(h["order"])))
        parts.append("{:s} az {:.0f}".format(name, h["phi"]))
    for d in model_dict.get("domes", []):
        parts.append("lobe az {:.0f}".format(d["az"]))
    glow_str = ", ".join(parts) or "uniform (no directional glow)"

    ax.set_title("{:s} sky brightness ({:s} {:s})\nzenith {:.1f} mag/arcsec$^2$ | "
        "glow: {:s} | LM0[{:s}]={:.2f}".format(str(station_id), fit_kind,
        str(model_dict.get("fit_date", "")), zenith_sqm, glow_str, str(station_id), lm0),
        fontsize=10)

    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# Self-priming (see ensureLightDomeModel)
AUTO_MIN_NIGHTS = 3        # usable archived nights required before an auto-fit is attempted
AUTO_MIN_TRIALS = 10000    # star trials required for the fit to be trusted
AUTO_REFIT_DAYS = 45       # dead-man backstop only - the PRIMARY refit trigger is
                           # measured: the nightly ratio normalization drifting
                           # persistently away from 1 (see modelIsStale), which detects
                           # atmospheric epoch changes when they actually happen instead
                           # of on a calendar
NORM_DRIFT_LIMIT = 0.25    # |median dratio - 1| beyond this = the model no longer
                           # represents the current sky epoch
NORM_DRIFT_MIN_NIGHTS = 5  # dratio entries (for the current model) needed to call drift
LM_HISTORY_WINDOW_DRIFT = 14   # trailing dratio entries the drift check considers
AUTO_SELECT_WINDOW = 14    # nights - auto-fits train on the clearest of the RECENT nights
                           # only: picking the clearest nights of a long window selects the
                           # most transparent atmospheric epoch ever seen, and a model fit
                           # there under-scores every hazier (but clear) night that follows
AUTO_MAX_NIGHTS = 6        # nights - more current-epoch data makes the bootstrap fit better
AUTO_REL_CLARITY = 0.5     # keep only nights at least this clear relative to the best one
                           # (cloudy training frames bias the model shallow, which can read
                           # cloudy nights as clear)
AUTO_POINTING_TOL = 3.0    # deg - pointing change that invalidates an auto-fitted model
AUTO_ATTEMPT_MARKER = "light_dome_fit_attempt.json"


def modelIsStale(model_dict, config, platepar=None):
    """ Decide whether an AUTO-FITTED station model needs a refit.

    Arguments:
        model_dict: [dict] The loaded model dictionary.
        config: [Config] Station config.

    Keyword arguments:
        platepar: [Platepar] Current platepar for the pointing check (skipped if None).

    Return:
        reason: [str] Why the model is stale, or None if it is fresh.
    """

    station = str(config.stationID)

    # Age
    fit_date = model_dict.get("fit_date")
    if fit_date is not None:
        try:
            age = (datetime.datetime.utcnow()
                   - datetime.datetime.strptime(fit_date, "%Y-%m-%d")).days
            if age > AUTO_REFIT_DAYS:
                return "age {:d} d > {:d} d".format(age, AUTO_REFIT_DAYS)
        except ValueError:
            pass

    # Star extractor retuned - the model absorbs the threshold into LM0
    fit_threshold = model_dict.get("intensity_threshold", {}).get(station)
    if (fit_threshold is not None) and (float(fit_threshold) != float(config.intensity_threshold)):
        return "intensity_threshold changed {:g} -> {:g}".format(
            fit_threshold, config.intensity_threshold)

    # Camera re-aimed - the FOV samples a different part of the dome
    fit_pointing = model_dict.get("pointing", {}).get(station)
    if (fit_pointing is not None) and (platepar is not None):
        d_az = abs((float(platepar.az_centre) - fit_pointing[0] + 180)%360 - 180)
        d_alt = abs(float(platepar.alt_centre) - fit_pointing[1])
        if max(d_az, d_alt) > AUTO_POINTING_TOL:
            return "pointing moved {:.1f} deg".format(max(d_az, d_alt))

    # Measured epoch drift: the nightly clear-sky ratio normalization (Utils.Flux) tracks
    # how far reality sits from this model. A persistent departure from 1 means the
    # atmospheric epoch has moved (e.g. monsoon onset) and the model should be re-based -
    # detected when it happens, not on a calendar.
    fit_date = model_dict.get("fit_date")
    if fit_date is not None:
        try:
            history_path = os.path.join(os.path.expanduser(config.data_dir),
                "{:s}_flux_lm_history.json".format(station))
            if os.path.isfile(history_path):
                with open(history_path) as f:
                    history = json.load(f)
                dratios = [v["dratio"] for v in history.values() if isinstance(v, dict)
                           and ("dratio" in v) and (v.get("dmodel") == str(fit_date))]
                if len(dratios) >= NORM_DRIFT_MIN_NIGHTS:
                    drift = abs(float(np.median(dratios[-LM_HISTORY_WINDOW_DRIFT:])) - 1.0)
                    if drift > NORM_DRIFT_LIMIT:
                        return "ratio normalization drift {:.2f}".format(drift)
        except Exception:
            pass

    return None


def ensureLightDomeModel(config, platepar=None):
    """ Self-prime the station's light-dome model from its own archives, no operator needed.

    Called nightly from detectClouds. If no model file exists (or a previously auto-fitted
    one went stale), the clearest recent archived nights are selected and a single-station
    model is fitted and installed. A single camera constrains the model only inside its own
    FOV footprint - which is exactly and only where it is evaluated for that camera, so a
    station-local fit is valid on its own. A manually fitted multi-station (site-pooled)
    model is NEVER overwritten: pooling is strictly better, so if one is present and stale
    this only logs a recommendation to refit it manually.

    Attempts are rate-limited to one per day via a marker file, so stations without enough
    archived nights retry cheaply until history accumulates; until then detectClouds falls
    back to the previous scalar behavior.

    Arguments:
        config: [Config] Station config.

    Keyword arguments:
        platepar: [Platepar] Current platepar, used for the pointing staleness check.

    Return:
        present: [bool] True if a usable model file is in place after the call.
    """

    data_dir = os.path.expanduser(config.data_dir)
    station = str(config.stationID)
    model_path = os.path.join(data_dir, "{:s}_{:s}".format(station, LIGHT_DOME_FILE_SUFFIX))

    # Existing model: keep it unless it is an auto-fitted one that went stale
    model = LightDomeModel.load(config)
    if model is not None:

        stale = modelIsStale(model.model, config, platepar=platepar)

        if stale is None:
            return True

        if (not model.model.get("auto_fitted")) or (len(model.model.get("cams", [])) > 1):
            print("Light-dome model is stale ({:s}) but was fitted manually/site-pooled - "
                  "keeping it; consider refitting with Utils.FitLightDome".format(stale))
            return True

        print("Auto-fitted light-dome model is stale ({:s}) - refitting".format(stale))

    # Rate-limit attempts to one per day
    marker_path = os.path.join(data_dir, "{:s}_{:s}".format(station, AUTO_ATTEMPT_MARKER))
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    try:
        with open(marker_path) as f:
            if json.load(f).get("date") == today:
                return model is not None
    except Exception:
        pass
    try:
        with open(marker_path, "w") as f:
            json.dump(dict(date=today), f)
    except Exception:
        pass

    # Enough usable archived nights? Prefer the recent window (current atmospheric epoch),
    # widen only if a young station does not have enough history yet
    night_dirs = selectNightDirs(config, window=AUTO_SELECT_WINDOW,
        max_nights=AUTO_MAX_NIGHTS, min_rel_clarity=AUTO_REL_CLARITY)
    if len(night_dirs) < AUTO_MIN_NIGHTS:
        night_dirs = selectNightDirs(config, window=30,
            max_nights=AUTO_MAX_NIGHTS, min_rel_clarity=AUTO_REL_CLARITY)
    if len(night_dirs) < AUTO_MIN_NIGHTS:
        print("Light-dome auto-fit: only {:d} usable archived night(s), need {:d} - "
              "waiting for more history".format(len(night_dirs), AUTO_MIN_NIGHTS))
        return model is not None

    dates = sorted(set(m.group(1) for m in
        (re.search(r"_(\d{8})(?:_|$)", os.path.basename(d)) for d in night_dirs) if m))
    print("Light-dome auto-fit from the {:d} clearest archived nights: {:s}".format(
        len(night_dirs), ", ".join(dates)))

    model_dict = fitLightDome([config], dates=dates)

    if (model_dict is None) or (model_dict.get("n_trials", 0) < AUTO_MIN_TRIALS):
        print("Light-dome auto-fit produced insufficient trials - keeping previous behavior")
        return model is not None

    model_dict["auto_fitted"] = True

    with open(model_path, "w") as f:
        json.dump(model_dict, f, indent=1)
    print("Light-dome model auto-fitted and installed: {:s}".format(model_path))

    try:
        renderLightDomeModel(model_dict, station,
            os.path.splitext(model_path)[0] + ".png")
    except Exception as e:
        print("Light-dome model plot failed ({}) - model itself is installed".format(e))

    return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Fit the site light-dome LM model from archived nights.")
    parser.add_argument("station_dirs", nargs="+",
        help="Station config directories (one per co-located camera).")
    parser.add_argument("--nights", type=str, default=None,
        help="Comma-separated YYYYMMDD training dates (clear nights). "
             "Auto-selects the clearest recent nights if omitted.")
    parser.add_argument("--max-order", type=int, default=3,
        help="Maximum azimuthal harmonic order (default 3; each order kept only if "
             "statistically significant).")
    parser.add_argument("--moon-phase-max", type=float, default=MOON_PHASE_MAX,
        help="Exclude frames with a moon above the horizon illuminated more than this "
             "percentage (default {:.0f}; 100 disables the filter).".format(MOON_PHASE_MAX))

    args = parser.parse_args()

    configs = [cr.loadConfigFromDirectory(".", os.path.abspath(d))
               for d in args.station_dirs]

    dates = args.nights.split(",") if args.nights else None

    model_dict = fitLightDome(configs, dates=dates, max_order=args.max_order,
        moon_phase_max=args.moon_phase_max)

    if model_dict is None:
        print("No usable data - nothing written.")
    else:
        for config in configs:
            out_path = os.path.join(os.path.expanduser(config.data_dir),
                "{:s}_{:s}".format(str(config.stationID), LIGHT_DOME_FILE_SUFFIX))
            with open(out_path, "w") as f:
                json.dump(model_dict, f, indent=1)
            print("wrote {:s}".format(out_path))

            try:
                renderLightDomeModel(model_dict, str(config.stationID),
                    os.path.splitext(out_path)[0] + ".png")
                print("wrote {:s}".format(os.path.splitext(out_path)[0] + ".png"))
            except Exception as e:
                print("plot failed ({}) - model itself is written".format(e))
