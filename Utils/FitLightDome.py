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
from RMS.LightDomeModel import (DOME_CATALOG_LIM_MAG, FIT_BOUND_TOL, LIGHT_DOME_FILE_SUFFIX,
    LM0_FIT_MIN, S_FIT_MAX, LightDomeModel, fitQualityIssues)


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


def buildStationTrials(config, night_dirs, n_ff=FF_PER_NIGHT, moon_phase_max=MOON_PHASE_MAX,
        lim_mag=DOME_CATALOG_LIM_MAG):
    """ Build hit/miss trials for one station over the given night directories.

    Arguments:
        config: [Config] Station config.
        night_dirs: [list of str] Paths to night directories with
            platepars_flux_recalibrated.json.

    Keyword arguments:
        n_ff: [int] Frames sampled per night.
        moon_phase_max: [float] Exclude frames with a moon above the horizon illuminated
            more than this (percent). 100 disables the filter.
        lim_mag: [float] Catalog depth for the trials. Must match the depth the model is
            scored with (see LightDomeModel.catalogLimMag).

    Return:
        trials: [dict of ndarray] az, alt, mag, det - or None if no usable data.
    """

    n_moon_excluded = 0
    footprint = None

    catalog_stars, _, _ = StarCatalog.readStarCatalog(config.star_catalog_path,
        config.star_catalog_file, lim_mag=lim_mag,
        mag_band_ratios=config.star_catalog_band_ratios)
    cat_ra, cat_dec, cat_mag = catalog_stars[:, 0], catalog_stars[:, 1], catalog_stars[:, 2]

    az_all, alt_all, mag_all, det_all, pchance_all = [], [], [], [], []

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

            # Chance-match floor of this frame: a catalog star with no true detection
            # still "matches" a random detection within the radius. At adaptive depths
            # the faint trials are numerous enough (hundreds of thousands) that this
            # ~1% floor, left unmodeled, teaches the fit a fat logistic tail - the
            # July-22 refit wave over-predicted faint expectations ~2-7x this way.
            if (mask is not None) and (getattr(mask, "img", None) is not None):
                usable_px = float(np.count_nonzero(mask.img > 0))
            else:
                usable_px = float(w*h)
            p_chance = 1.0 - np.exp(-len(sx)*np.pi*MATCH_RADIUS_PX**2/max(usable_px, 1.0))

            az_all.append(az)
            alt_all.append(alt)
            mag_all.append(m)
            det_all.append(detected)
            pchance_all.append(np.full(len(m), p_chance))

    if n_moon_excluded:
        print("  moon filter: excluded {:d} frame(s) (moon up, phase > {:.0f}%)".format(
            n_moon_excluded, moon_phase_max))

    if not az_all:
        return None

    return dict(az=np.concatenate(az_all), alt=np.concatenate(alt_all),
        mag=np.concatenate(mag_all), det=np.concatenate(det_all),
        pchance=np.concatenate(pchance_all),
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




MAX_FIT_TRIALS = 300000    # star trials the optimizer sees; larger sets are randomly
                           # subsampled (deterministic seed). ~20 fit parameters need
                           # nowhere near this many binomial trials, and an uncapped
                           # 6-camera site fit on adaptive-depth CALSTARS reaches
                           # millions of trials (observed: 8 h of CPU on a pod computer)

_LN10 = np.log(10.0)


def _domeModelLM(p, ncam, norder, az, alt, ci):
    """ Evaluate the dome model LM per trial (see fitLightDome for the parameter layout).

    Harmonics basis: LP = bowl + sum_k A_k*cos(k*(az - phi_k))*exp(-alt/h_k), clamped
    non-negative (the cosine terms go negative on the dark side). The dipole (k=1) is
    the leading term for an observer EMBEDDED in the glow field; for a distant city,
    higher orders approximate a localized dome.
    """

    k = p[ncam]
    q0, hb = p[ncam + 2], p[ncam + 3]
    alt_c = np.clip(alt, 5.0, 90.0)
    z2 = np.sin(np.radians(90.0 - alt_c))**2
    vr = 1.0/np.sqrt(1.0 - 0.97*z2)
    lp = (10.0**q0)*np.exp(-alt_c/hb)
    for j in range(norder):
        qa, phi, hh = p[ncam + 4 + 3*j:ncam + 4 + 3*j + 3]
        lp = lp + (10.0**qa)*np.cos(np.radians((j + 1)*(az - phi)))*np.exp(-alt_c/hh)
    B = vr + np.maximum(lp, 0.0)
    return p[ci] - k*(1.0/np.sin(np.radians(alt_c)) - 1.0) - 1.25*np.log10(B)


def _domeNLLAndGrad(p, ncam, norder, az, alt, mag, det, ci, pchance=None):
    """ Binomial negative log-likelihood of the dome model AND its analytic gradient.

    The observation model includes the per-frame chance-match floor: a catalog star
    matches either by true detection (logistic in the model LM) or by a random
    detection landing within the match radius,

        P_match = p_c + (1 - p_c) * P_det(LM, m)

    Omitting p_c lets hundreds of thousands of faint deep-catalog trials (matched at
    the ~1% chance rate) inflate the fitted logistic tail.

    The closed-form gradient is what makes the fit tractable: L-BFGS-B with numerical
    differentiation costs (n_params + 1) full likelihood evaluations per gradient - a
    ~20x multiplier on a 6-camera site fit that once kept a pod computer at 100% CPU
    for 8 hours. Correctness is pinned against numerical differentiation in the tests.

    Return:
        (f, grad): NLL value and d(NLL)/dp, same layout as p.
    """

    if pchance is None:
        pchance = np.zeros(len(az))

    k = p[ncam]
    s = p[ncam + 1]
    q0, hb = p[ncam + 2], p[ncam + 3]

    alt_c = np.clip(alt, 5.0, 90.0)
    z2 = np.sin(np.radians(90.0 - alt_c))**2
    vr = 1.0/np.sqrt(1.0 - 0.97*z2)

    bowl = (10.0**q0)*np.exp(-alt_c/hb)
    lp = bowl.copy()
    harmonic_terms = []
    for j in range(norder):
        qa, phi, hh = p[ncam + 4 + 3*j:ncam + 4 + 3*j + 3]
        cosj = np.cos(np.radians((j + 1)*(az - phi)))
        expj = np.exp(-alt_c/hh)
        tj = (10.0**qa)*cosj*expj
        harmonic_terms.append((qa, phi, hh, expj, tj))
        lp = lp + tj

    pos = lp > 0.0
    B = vr + np.where(pos, lp, 0.0)
    airmass = 1.0/np.sin(np.radians(alt_c)) - 1.0

    lm = p[ci] - k*airmass - 1.25*np.log10(B)

    u = (lm - mag)/s
    pr = 1.0/(1.0 + np.exp(-u))
    pm = pchance + pr*(1.0 - pchance)
    pmc = np.clip(pm, 1e-6, 1.0 - 1e-6)
    f = -np.sum(det*np.log(pmc) + (1 - det)*np.log(1 - pmc))

    # dNLL/du = w*(pm - det) with w = pr*(1-pr)*(1-p_c)/(pm*(1-pm)); reduces to the
    # familiar (pr - det) when p_c = 0. Zeroed where the clip flattens the likelihood.
    inside = (pm > 1e-6) & (pm < 1.0 - 1e-6)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = pr*(1.0 - pr)*(1.0 - pchance)/(pm*(1.0 - pm))
    r = np.where(inside, w*(pm - det), 0.0)
    rs = r/s

    grad = np.zeros_like(p)

    # Per-camera LM0: dLM/dLM0_c = 1 on that camera's trials
    np.add.at(grad, ci, rs)

    grad[ncam] = -np.sum(rs*airmass)
    grad[ncam + 1] = -np.sum(r*(lm - mag))/s**2

    # Shared chain factor for every brightness-field parameter:
    # dLM/dB = -1.25/(ln10 * B), through the non-negativity clamp
    dB = np.where(pos, -1.25/(_LN10*B), 0.0)

    grad[ncam + 2] = np.sum(rs*dB*_LN10*bowl)
    grad[ncam + 3] = np.sum(rs*dB*bowl*alt_c/hb**2)

    for j, (qa, phi, hh, expj, tj) in enumerate(harmonic_terms):
        base = ncam + 4 + 3*j
        grad[base] = np.sum(rs*dB*_LN10*tj)
        sinj = np.sin(np.radians((j + 1)*(az - phi)))
        grad[base + 1] = np.sum(rs*dB*(10.0**qa)*expj*sinj*np.radians(j + 1))
        grad[base + 2] = np.sum(rs*dB*tj*alt_c/hh**2)

    return f, grad


def fitLightDome(station_configs, dates=None, max_order=3, moon_phase_max=MOON_PHASE_MAX,
        lim_mag=None, _depth_iter=0):
    """ Fit the site model over multiple co-located stations.

    Arguments:
        station_configs: [list of Config] One config per station.

    Keyword arguments:
        dates: [list of str] Explicit YYYYMMDD training dates (recommended: clear nights).
        max_order: [int] Maximum azimuthal harmonic order to try (1=dipole, 2=quadrupole,
            ...). Each order is kept only if it clears the significance gate.
        moon_phase_max: [float] Exclude frames with a moon above the horizon illuminated
            more than this (percent). 100 disables the filter.
        lim_mag: [float] Catalog depth for trials and fit bounds. None (default) starts at
            DOME_CATALOG_LIM_MAG; if the fitted LM demands a deeper catalog (the logistic
            tail must be fully sampled or LM0 saturates against the catalog ceiling and
            the fit degenerates - observed fleet-wide as LM0 pinned at the old 7.0 bound
            with inflated s), the fit is repeated once at the required depth. The depth
            actually used is stored in the model as catalog_lim_mag, and scoring MUST use
            that stored value (LightDomeModel.catalogLimMag) so expected counts stay
            calibrated to the fit.

    Return:
        model_dict: [dict] Fitted site model (LightDomeModel format), or None.
    """

    if lim_mag is None:
        lim_mag = DOME_CATALOG_LIM_MAG

    az_l, alt_l, mag_l, det_l, ci_l, pchance_l = [], [], [], [], [], []
    fit_nights = []
    pointing = {}
    thresholds = {}
    footprints = {}

    # Stations without usable trials are dropped here, BEFORE the parameter vector is
    # built - otherwise the written model would carry an unconstrained (initial-value)
    # LM0 for a camera the data never touched
    cams = []

    for config in station_configs:

        station = str(config.stationID)

        night_dirs = selectNightDirs(config, dates=dates)

        trials = buildStationTrials(config, night_dirs, moon_phase_max=moon_phase_max,
            lim_mag=lim_mag)

        if trials is None:
            print("{:s}: no usable trials, station excluded from fit".format(station))
            continue

        n = len(trials["az"])
        print("{:s}: {:d} trials from {:d} night dir(s), detected fraction {:.2f}".format(
            station, n, len(night_dirs), float(np.mean(trials["det"]))))

        fit_nights += [os.path.basename(d) for d in night_dirs]

        az_l.append(trials["az"])
        alt_l.append(trials["alt"])
        mag_l.append(trials["mag"])
        det_l.append(trials["det"])
        pchance_l.append(trials["pchance"])
        ci_l.append(np.full(n, len(cams), dtype=np.int64))

        # Metadata for staleness detection (see ensureLightDomeModel) and the model plot
        pointing[station] = list(trials["pointing"])
        thresholds[station] = float(config.intensity_threshold)
        footprints[station] = trials["footprint"]

        cams.append(station)

    if not cams:
        return None

    ncam = len(cams)

    az = np.concatenate(az_l)
    alt = np.concatenate(alt_l)
    mag = np.concatenate(mag_l)
    det = np.concatenate(det_l)
    ci = np.concatenate(ci_l)
    pchance = np.concatenate(pchance_l)

    n_trials_total = len(az)
    print("TOTAL: {:d} trials".format(n_trials_total))

    # Cap the trial count the optimizer sees - the parameters are constrained just as
    # well by a large random subsample, at a fraction of the cost (deterministic seed
    # so refits on the same data reproduce)
    if n_trials_total > MAX_FIT_TRIALS:
        sel = np.random.RandomState(0).choice(n_trials_total, MAX_FIT_TRIALS, replace=False)
        az, alt, mag, det, ci, pchance = (az[sel], alt[sel], mag[sel], det[sel],
                                          ci[sel], pchance[sel])
        print("Subsampled to {:d} trials for the fit".format(len(az)))

    print("Chance-match floor: median {:.4f}, p90 {:.4f}".format(
        float(np.median(pchance)), float(np.percentile(pchance, 90))))

    def nll(p, norder):
        return _domeNLLAndGrad(p, ncam, norder, az, alt, mag, det, ci, pchance)

    # LM0 may exceed the catalog depth by a little (the logistic tail still constrains
    # it there), but far beyond it the trials carry no information - bound accordingly
    base_bounds = [(LM0_FIT_MIN, lim_mag + 1.0)]*ncam \
        + [(0.0, 1.5), (0.12, S_FIT_MAX), (-2.0, 3.0), (5.0, 60.0)]
    order_bounds = [(-2.0, 3.5), (-360.0, 720.0), (3.0, 60.0)]   # log10 A, phase, alt scale

    # Identifiability gate: an order-k azimuthal harmonic is only measurable if
    # the trials' azimuth coverage leaves no gap wider than half its period
    # (180/k deg) - otherwise amplitude and phase trade off freely, and the fit
    # buys a small in-FOV likelihood gain with an unphysical global harmonic
    # (observed on a single-camera station: a 57x-the-bowl dipole whose
    # non-negativity clamp cut an LM cliff through the FOV's own edge). A
    # single ~90 deg FOV therefore gets the azimuth-symmetric bowl only, while
    # multi-camera rings that surround the compass keep their harmonics.
    az_bins = np.zeros(36, dtype=bool)
    az_bins[(np.asarray(az, dtype=np.float64)%360.0/10.0).astype(np.intp) % 36] = True
    if az_bins.all():
        max_az_gap = 0.0
    else:
        # Largest circular run of empty 10 deg bins
        empty = np.concatenate([~az_bins, ~az_bins])
        run, max_run = 0, 0
        for e in empty:
            run = run + 1 if e else 0
            max_run = max(max_run, run)
        max_az_gap = min(max_run, 36)*10.0
    identifiable_order = 0
    for j in range(1, max_order + 1):
        if max_az_gap <= 180.0/j:
            identifiable_order = j
        else:
            break
    if identifiable_order < max_order:
        print("Azimuth coverage gap {:.0f} deg: harmonics above order {:d} are "
              "unidentifiable - capping (was {:d})".format(
              max_az_gap, identifiable_order, max_order))
    max_order = identifiable_order

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
                jac=True, bounds=bounds, options=dict(maxiter=120))
            if (best is None) or (r.fun < best.fun):
                best = r

        r = minimize(nll, best.x, args=(norder,), method="L-BFGS-B", jac=True,
            bounds=bounds, options=dict(maxiter=500))

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
        n_trials_total=int(n_trials_total),
        pointing=pointing,
        intensity_threshold=thresholds,
        footprints=footprints,
        nll={str(kk): v["nll"] for kk, v in results.items()},
        # Feature marker: this fit models the chance-match floor. Models without
        # it (the 2026-07-22 refit wave) are chance-contaminated at depth -
        # inflated s, dragged-down LM0 - and isStale retires them on sight.
        floor_modeled=True,
    )

    model_dict["catalog_lim_mag"] = float(lim_mag)

    # Adaptive catalog depth: the logistic tail must be fully sampled or the fit
    # saturates. If the fitted LM wants a deeper catalog than was used, refit once at
    # the required depth (recursion depth-limited; the deeper fit re-derives everything
    # so trials, bounds and stored depth stay consistent)
    from RMS.LightDomeModel import domeCatalogLimMag
    wanted = domeCatalogLimMag(model_dict["LM0"], model_dict["s"])
    if (wanted > lim_mag + 0.25) and (_depth_iter < 2):
        print("\nCatalog depth {:.2f} too shallow for the fitted LM "
              "(max LM0 {:.2f}, s {:.2f}) - refitting at depth {:.2f}".format(
              lim_mag, max(model_dict["LM0"]), model_dict["s"], wanted))
        return fitLightDome(station_configs, dates=dates, max_order=max_order,
            moon_phase_max=moon_phase_max, lim_mag=wanted, _depth_iter=_depth_iter + 1)

    # Ceiling guard: catches both a saturated final fit and any future regression of
    # the adaptive-depth logic (this exact symptom - LM0 pinned with inflated s - went
    # unnoticed fleet-wide against the old fixed depth)
    if max(model_dict["LM0"]) + 2.0*model_dict["s"] > lim_mag - 0.25:
        print("WARNING: fitted LM0 approaches the catalog depth ({:.2f} + 2s vs "
              "{:.2f}) - the model may be saturated against the catalog ceiling".format(
              max(model_dict["LM0"]), lim_mag))

    # Degenerate-fit diagnosis travels with the model - the auto-fit path refuses to
    # adopt a model with issues, and a manual fit prints them for the operator
    model_dict["quality_issues"] = fitQualityIssues(model_dict)
    for msg in model_dict["quality_issues"]:
        print("WARNING: degenerate fit: {:s}".format(msg))

    print("\nFitted model ({:d} harmonic order(s), catalog depth {:.2f}):".format(
        use, lim_mag))
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
    lm_all = _domeModelLM(p, ncam, use, az, alt, ci)
    p_det = 1.0/(1.0 + np.exp(-(lm_all - mag)/s_soft))
    p_match = pchance + p_det*(1.0 - pchance)
    for i, cam in enumerate(cams):
        sel = ci == i
        if np.any(sel):
            print("  {:s}: {:.2f}".format(cam,
                float(np.sum(det[sel])/np.sum(p_match[sel]))))

    # Magnitude-resolved empirical calibration: parameters at their bounds are one
    # failure class (fitQualityIssues); a model that fits its parameters comfortably
    # while lying about the sky is another. Compare claimed match probability against
    # the measured rate per magnitude bin on the training trials - the July-22 refit
    # wave (P=0.33 claimed vs 0.14 delivered at mag 5-6) is exactly what this catches.
    # The table travels in the model file; gross bins block adoption via
    # quality_issues.
    calibration = []
    for lo in np.arange(np.floor(mag.min()), np.ceil(lim_mag)):
        b = (mag >= lo) & (mag < lo + 1)
        if b.sum() < 500:
            continue
        mean_pm = float(np.mean(p_match[b]))
        rate = float(np.mean(det[b]))
        calibration.append(dict(mag_lo=float(lo), n=int(b.sum()),
            expected=round(mean_pm, 4), measured=round(rate, 4)))
        if (mean_pm >= 0.02) and (abs(rate - mean_pm) > max(0.05, 0.5*mean_pm)):
            model_dict["quality_issues"].append(
                "calibration: mag {:.0f}-{:.0f} claims P={:.3f} but measures "
                "{:.3f} on the training trials".format(lo, lo + 1, mean_pm, rate))
    model_dict["calibration"] = calibration
    for msg in model_dict["quality_issues"]:
        if msg.startswith("calibration"):
            print("WARNING: " + msg)

    return model_dict


# Model plot (see renderLightDomeModel)
NATURAL_ZENITH_SQM = 21.8   # mag/arcsec^2 - moonless natural zenith sky reference used to
                            # express the model's relative brightness in absolute units.
                            # NOTE: the absolute anchor is approximate (star detectability
                            # cannot fully separate camera depth from sky brightness);
                            # the relative structure and the fixed scale are the point.
PLOT_SQM_MIN = 16.0         # fixed color scale so every station and site renders
PLOT_SQM_MAX = 22.0         # comparably (brighter sky = brighter color)


ANCHOR_WINDOW_NIGHTS = 14   # recent SQM measurements per camera the anchor considers
ANCHOR_RESIDUAL_WARN = 0.3  # mag - a camera's median departing this far from the site
                            # anchor suggests the glow field misallocates brightness
                            # between the cameras' FOVs


def _radiometricAnchorOffset(model, configs):
    """ Offset (mag) that pins the model's soft absolute scale to the site's MEASURED
        sky quality history. The fit cannot separate camera depth from absolute sky
        brightness (LM0 degeneracy), so the model-implied zenith runs too dark; the
        radiometric SQM measurements supply the missing anchor.

        The measurements of ALL co-located cameras are pooled: each camera measures its
        own FOV patch, transposed to zenith through the shared glow field, so one site
        anchor emerges and every camera's map shows the same site zenith - the anchor
        analog of the pooled model fit. Returns 0 when no usable history exists (map
        stays model-relative, labeled).

    Arguments:
        model: [LightDomeModel] The site model.
        configs: [Config or list of Config] The site's station configs, whose sky
            quality histories are pooled.
    """

    if not isinstance(configs, (list, tuple)):
        configs = [configs]

    try:
        b_zen = float(model.skyBrightness(0.0, 90.0))
        measured = []
        per_station = {}

        for config in configs:

            try:
                path = os.path.join(os.path.expanduser(config.data_dir),
                    "{:s}_sky_quality_history.json".format(str(config.stationID)))
                with open(path) as f:
                    nights = json.load(f).get("nights", {})
            except Exception:
                continue

            station_measured = []
            for entry in nights.values():
                if entry.get("sqm") is None or not entry.get("absolute", False):
                    continue
                if (entry.get("az") is None) or (entry.get("alt") is None):
                    continue
                b_patch = float(model.skyBrightness(float(entry["az"]), float(entry["alt"])))
                station_measured.append(float(entry["sqm"]) + 2.5*np.log10(b_patch/b_zen))

            # Each camera contributes its recent window only (current atmospheric epoch)
            if station_measured:
                per_station[str(config.stationID)] = station_measured[-ANCHOR_WINDOW_NIGHTS:]
                measured += station_measured[-ANCHOR_WINDOW_NIGHTS:]

        if len(measured) < 3:
            return 0.0

        measured_zenith = float(np.median(measured))
        model_zenith = NATURAL_ZENITH_SQM - 2.5*np.log10(b_zen)

        # Per-camera residuals vs the site anchor. Every camera measures the same site
        # zenith once transposed through the shared glow field, so a systematic residual
        # is a diagnostic that the field misallocates brightness between the FOVs
        if len(per_station) > 1:
            print("Radiometric site anchor: zenith {:.2f} mag/arcsec^2 pooled from "
                  "{:d} cameras".format(measured_zenith, len(per_station)))
            for sid in sorted(per_station):
                res = float(np.median(per_station[sid])) - measured_zenith
                flag = "  <- check the glow field allocation" \
                    if abs(res) > ANCHOR_RESIDUAL_WARN else ""
                print("  {:s}: {:+.2f} mag vs site (n={:d}){:s}".format(
                    sid, res, len(per_station[sid]), flag))

        return measured_zenith - model_zenith

    except Exception:
        return 0.0


def renderLightDomeModel(model_dict, station_id, out_path, config=None):
    """ Render the fitted site model as a sky-brightness hemisphere map.

    Fixed absolute color scale (PLOT_SQM_MIN..MAX mag/arcsec^2, light pollution bright)
    so plots from different stations and sites are directly comparable. The station's own
    FOV footprint is highlighted; co-located sister cameras are drawn in grey for context.
    When the station has a measured sky quality history, the map's absolute scale is
    radiometrically anchored to it (the model alone carries a soft absolute anchor).

    Arguments:
        model_dict: [dict] Fitted model (LightDomeModel format, with footprints metadata).
        station_id: [str] Station to highlight.
        out_path: [str] Output PNG path.

    Keyword arguments:
        config: [Config or list of Config] Enables the radiometric anchor lookup; pass
            ALL the site's configs so the anchor pools their measurements and every
            camera's map shows the same site zenith. None = model-relative.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    model = LightDomeModel(model_dict)

    anchor = _radiometricAnchorOffset(model, config) if config is not None else 0.0

    fig = plt.figure(figsize=(9.5, 8.2))
    ax = fig.add_subplot(1, 1, 1, projection="polar")

    az_grid = np.radians(np.arange(0, 361, 2))
    alt_grid = np.arange(5, 86, 2)
    az_mesh, alt_mesh = np.meshgrid(np.degrees(az_grid), alt_grid)

    sqm = NATURAL_ZENITH_SQM - 2.5*np.log10(model.skyBrightness(az_mesh, alt_mesh)) + anchor

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

    zenith_sqm = float(NATURAL_ZENITH_SQM - 2.5*np.log10(model.skyBrightness(0.0, 90.0))) + anchor
    anchor_str = "radiometric anchor" if abs(anchor) > 0.001 else "model-relative, unanchored"
    lm0 = model.lm0_map.get(str(station_id), model.lm0_default)
    fit_kind = "auto-fit" if model_dict.get("auto_fitted") else "site fit"

    # Describe the directional glow structure in plain terms
    parts = []
    for h in model_dict.get("harmonics", []):
        name = {1: "dipole", 2: "quadrupole"}.get(int(h["order"]),
            "order {:d}".format(int(h["order"])))
        parts.append("{:s} az {:.0f}".format(name, h["phi"]))
    glow_str = ", ".join(parts) or "uniform (no directional glow)"

    ax.set_title("{:s} sky brightness ({:s} {:s})\nzenith {:.1f} mag/arcsec$^2$ ({:s}) | "
        "glow: {:s} | LM0[{:s}]={:.2f}".format(str(station_id), fit_kind,
        str(model_dict.get("fit_date", "")), zenith_sqm, anchor_str, glow_str,
        str(station_id), lm0), fontsize=10)

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


def findSiblingStationConfigs(config):
    """ Discover co-located sibling stations on a multi-station computer.

    Uses the multi-camera layout convention: each station lives in a directory named by
    its stationID (e.g. ~/source/Stations/XX0001/.config). A sibling is accepted only if
    its directory name matches its stationID (this rules out unrelated RMS checkouts with
    template configs) and it sits at the same site (within ~1 km) - the fitted glow field
    is shared only by co-located cameras.

    Arguments:
        config: [Config] Station config.

    Return:
        configs: [list of Config] This station's config first, then any co-located
            siblings. A single-camera station simply gets a one-entry list.
    """

    configs = [config]

    cfg_path = getattr(config, "config_file_name", None)
    if (not cfg_path) or (not os.path.isfile(cfg_path)):
        return configs

    station_dir = os.path.dirname(os.path.abspath(cfg_path))
    root = os.path.dirname(station_dir)

    # A duplicated data_dir would double-count its trials in the pooled fit
    seen_data_dirs = {os.path.realpath(os.path.expanduser(config.data_dir))}

    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return configs

    for name in entries:

        sib_dir = os.path.join(root, name)
        sib_cfg_path = os.path.join(sib_dir, ".config")

        if (os.path.realpath(sib_dir) == os.path.realpath(station_dir)) \
                or (not os.path.isfile(sib_cfg_path)):
            continue

        try:
            sib = cr.parse(sib_cfg_path)
        except Exception:
            continue

        # The directory must be named by the station it hosts (the multi-camera layout)
        if name.lower() != str(sib.stationID).lower():
            continue

        # Same site only
        if (abs(float(sib.latitude) - float(config.latitude)) > 0.01) \
                or (abs(float(sib.longitude) - float(config.longitude)) > 0.01):
            continue

        sib_data_dir = os.path.realpath(os.path.expanduser(sib.data_dir))
        if sib_data_dir in seen_data_dirs:
            continue
        seen_data_dirs.add(sib_data_dir)

        configs.append(sib)

    return configs


def findLegacyModelFile(data_dir, station):
    """ Return the path of a model file from the retired dome basis, or None.

    Arguments:
        data_dir: [str] Station data directory.
        station: [str] Station ID.

    Return:
        path: [str] Path of the legacy-basis model file, or None if none is present.
    """

    # Same precedence as LightDomeModel.load: only the file that would actually be used
    # matters - a shadowed legacy file behind a harmonic one must not refit-loop forever
    for name in ["{:s}_{:s}".format(station, LIGHT_DOME_FILE_SUFFIX), LIGHT_DOME_FILE_SUFFIX]:

        path = os.path.join(data_dir, name)
        if not os.path.isfile(path):
            continue

        try:
            with open(path) as f:
                model_dict = json.load(f)
        except Exception:
            continue

        if model_dict.get("model") != "vanrhijn_harmonics":
            return path

        return None

    return None


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

    # Fit predates the chance-match floor: without the floor term, deep-catalog
    # fits are contaminated by chance coincidences (inflated s, dragged-down
    # LM0 - the 2026-07-22 refit wave; USV001 read normalized ratios of 3).
    # The nightly normalization absorbs the level error but not the shape
    # error, so retire these models on sight rather than waiting for the
    # drift check to accumulate history. Scoped to machine-fit models (ones
    # carrying fit metadata): bare pre-metadata/manual models stay never-stale.
    if (model_dict.get("fit_date") is not None) \
            and not model_dict.get("floor_modeled", False):
        return "fit predates the chance-match floor"

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
    """ Self-prime the site's light-dome model from the stations' own archives, no
        operator needed.

    Called nightly from detectClouds. If no model file exists, or the present one went
    stale, or it is from the retired dome basis, or it does not cover every co-located
    sibling station, the clearest recent archived nights are selected and the model is
    refitted. All co-located sibling stations found on this
    computer (multi-camera layout: sibling config directories named by stationID) are
    pooled into one site fit, so every camera's FOV constrains the shared glow field and
    no camera relies on an extrapolation into sky it cannot see. A single-camera station
    fits alone - it constrains the model only inside its own FOV footprint, which is
    exactly and only where the model is evaluated for it. The identical site model is
    installed for every station that contributed trials, and their attempt markers are
    stamped so the siblings do not redo the same fit.

    Attempts are rate-limited to one per day per station via a marker file, so stations
    without enough archived nights retry cheaply until history accumulates; until then
    detectClouds falls back to the previous scalar behavior.

    Arguments:
        config: [Config] Station config.

    Keyword arguments:
        platepar: [Platepar] Current platepar, used for the pointing staleness check.

    Return:
        present: [bool] True if a usable model file is in place after the call.
    """

    data_dir = os.path.expanduser(config.data_dir)
    station = str(config.stationID)

    # Every co-located sibling station on this computer joins the site fit
    station_configs = findSiblingStationConfigs(config)

    # A model file from the retired dome basis is refitted outright - the harmonic basis
    # is the only supported one, and old files no longer evaluate
    legacy_path = findLegacyModelFile(data_dir, station)
    if legacy_path is not None:
        print("Light-dome model {:s} uses the retired dome basis - refitting with the "
              "harmonic basis".format(os.path.basename(legacy_path)))

    # Existing harmonic model: keep it while it is fresh AND it covers every co-located
    # sibling. A fresh-but-narrower model (e.g. a single-camera fit from before the
    # siblings existed, or before pooling was automatic) is superseded by the pooled
    # site fit rather than left to age out.
    model = LightDomeModel.load(config)
    if (model is not None) and (legacy_path is None):

        stale = modelIsStale(model.model, config, platepar=platepar)

        # An installed degenerate model (fit at a bound - e.g. LM0 pinned low by an
        # all-cloudy fit window, or saturated against the old fixed catalog depth) is
        # refit outright rather than left to poison verdicts until the drift check trips
        degenerate = fitQualityIssues(model.model)

        missing = [str(c.stationID) for c in station_configs
                   if str(c.stationID) not in model.cams]

        if (stale is None) and (not missing) and (not degenerate):
            return True

        if degenerate:
            print("Light-dome model is degenerate - refitting:")
            for msg in degenerate:
                print("  " + msg)
        elif stale is not None:
            print("Light-dome model is stale ({:s}) - refitting".format(stale))
        else:
            print("Light-dome model does not cover co-located station(s) {:s} - "
                  "refitting the site pooled".format(", ".join(missing)))

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

    if len(station_configs) > 1:
        print("Pooling co-located stations for the site fit: {:s}".format(
            ", ".join(str(c.stationID) for c in station_configs)))

    model_dict = fitLightDome(station_configs, dates=dates)

    if (model_dict is None) or (model_dict.get("n_trials", 0) < AUTO_MIN_TRIALS):
        print("Light-dome auto-fit produced insufficient trials - keeping previous behavior")
        return model is not None

    # Never adopt a degenerate fit - keep whatever was in place (the previous model, or
    # the scalar fallback) and let the daily attempt marker retry when new nights arrive
    if model_dict.get("quality_issues"):
        print("Light-dome auto-fit is degenerate - not installing it:")
        for msg in model_dict["quality_issues"]:
            print("  " + msg)
        return model is not None

    model_dict["auto_fitted"] = True

    # Install the identical site model for every station that contributed trials, and
    # stamp their attempt markers so the siblings do not redo the same fit tonight
    for sib_config in station_configs:

        sib_station = str(sib_config.stationID)
        if sib_station not in model_dict["cams"]:
            continue

        sib_data_dir = os.path.expanduser(sib_config.data_dir)
        sib_model_path = os.path.join(sib_data_dir,
            "{:s}_{:s}".format(sib_station, LIGHT_DOME_FILE_SUFFIX))

        with open(sib_model_path, "w") as f:
            json.dump(model_dict, f, indent=1)
        print("Light-dome model auto-fitted and installed: {:s}".format(sib_model_path))

        try:
            # Pass the whole site's configs so the radiometric anchor pools their SQM
            # histories - one site zenith on every camera's map
            renderLightDomeModel(model_dict, sib_station,
                os.path.splitext(sib_model_path)[0] + ".png", config=station_configs)
        except Exception as e:
            print("Light-dome model plot failed ({}) - model itself is installed".format(e))

        try:
            with open(os.path.join(sib_data_dir,
                    "{:s}_{:s}".format(sib_station, AUTO_ATTEMPT_MARKER)), "w") as f:
                json.dump(dict(date=today), f)
        except Exception:
            pass

    return (station in model_dict["cams"]) or (model is not None)


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
                # All configs, so the radiometric anchor pools the site's SQM histories
                renderLightDomeModel(model_dict, str(config.stationID),
                    os.path.splitext(out_path)[0] + ".png", config=configs)
                print("wrote {:s}".format(os.path.splitext(out_path)[0] + ".png"))
            except Exception as e:
                print("plot failed ({}) - model itself is written".format(e))
