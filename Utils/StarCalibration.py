""" Trailing per-star calibration: each star's MEASURED behavior on this camera,
maintained nightly.

The dome model predicts detection probability from sky position and magnitude -
a model of the average star through the average of the training epochs. Real
stars depart from it in ways that matter to the transparency estimators: the
extractor chronically drops saturated stars (a model-p 0.97 star matching 59%
of the time reads as permanent haze wherever it drifts), unresolved doubles are
brighter than their catalog magnitude, and glare or vignette zones suppress
whole regions. The cure everywhere is the same: weight each star's absence by
what THAT star measurably does when its own sky is clear.

Per star (keyed by star_cat_id, schema v4) and per evidence channel:
    rate_calstars - CALSTARS match rate on clear frames.
    rate_forced   - forced-photometry (FF avepixel) detection rate on clear
                    frames, where measured (the bright set).
    base_mag      - median instrumental-minus-catalog magnitude on clear
                    matched frames, AFTER removing the night's measured
                    extinction slope k*(airmass - 1) - so the baseline means
                    "this star at reference airmass" and a star sliding into
                    low elevation is not misread as dimmed (see k_fit).
    sigma_mag     - robust scatter of the same residual.
    n_nights      - nights contributing.

"Clear" is conditioned on the night's own transparency map: only frames where
the star's cell reads dm < CLEAR_DM_MAX contribute - a star behind a cloud is
not evidence about the star. (First night of a station bootstraps from the
whole-night statistics of the frames the map judged clear.)

The nightly extinction slope k_fit is measured from the night's own clear
matched residuals (pooled per-star-demeaned residual vs demeaned airmass): it
lumps true atmospheric extinction with any altitude-correlated vignette and
edge-PSF loss, which is exactly the systematic that must not read as haze.
Measured 0.68 mag/airmass on a vignetting-heavy camera vs ~0.2 for clean sky -
a per-camera constant worth tracking in its own right.

Values are EMA-merged across nights (EMA_ALPHA), so the calibration tracks
slow drift (focus, aging, seasonal extinction) with ~2-week memory. A catalog
depth change (model refit) resets the file: star ids are catalog row indices
at a fixed depth, and one warmup night is cheaper than a mis-join.

File: <data_dir>/<stationID>_star_calibration.npz
"""

from __future__ import absolute_import, division, print_function

import json
import os

import numpy as np


SCHEMA_VERSION = 1

FILE_SUFFIX = "star_calibration.npz"

CLEAR_DM_MAX = 0.3     # mag - a star's frame is "clear" if its cell reads below this
MIN_CLEAR_FRAMES = 20  # clear sightings before a star's rate is trusted
MIN_PHOT_FRAMES = 10   # clear matched frames before a baseline is trusted
EMA_ALPHA = 0.2        # nightly EMA weight (~10-night memory)
K_FIT_MAX = 1.5        # mag/airmass - physical clamp on the nightly slope


def calibrationFileName(station_id):
    return "{:s}_{:s}".format(str(station_id), FILE_SUFFIX)


def _kastenYoung(alt_deg):
    """ Airmass from altitude (deg), Kasten & Young 1989. """

    h = np.maximum(alt_deg, 2.0)
    return 1.0/(np.sin(np.radians(h)) + 0.50572*(h + 6.07995)**-1.6364)


def computeNightStarStats(config, night_dir):
    """ Per-star clear-conditioned statistics for one processed night.

    Requires the night's star-scoring product (schema v4), transparency map,
    CALSTARS file and recalibrated platepars to exist in night_dir.

    Return:
        stats: [dict] cat_id-indexed arrays (see updateStarCalibration), plus
            'k_fit' and provenance - or None if prerequisites are missing.
    """

    from RMS.Formats import CALSTARS as CALSTARSFormat
    from RMS.Formats.StarScoring import loadStarScoring, scoringFileName
    from RMS.Astrometry.Conversions import raDec2AltAz
    from RMS.Formats import StarCatalog

    night_name = os.path.basename(os.path.normpath(night_dir))

    scoring_path = os.path.join(night_dir, scoringFileName(night_name))
    if not os.path.isfile(scoring_path):
        return None

    header, frames, stars = loadStarScoring(scoring_path)
    if int(header.get("schema_version", 0)) < 4 or "star_cat_id" not in stars:
        return None

    # Union channel: fuse the stills sidecar's instantaneous detections first,
    # so rates and conditioning see the same evidence the tree consumes
    from Utils.StillsSampler import fuseSidecarDetections
    stars = fuseSidecarDetections(night_dir, frames, stars)

    n_frames = len(frames["frame_names"])

    cat_id = np.asarray(stars["star_cat_id"], dtype=np.intp)
    sf = np.asarray(stars["star_frame"], dtype=np.intp)
    sx = np.asarray(stars["star_x"], dtype=np.float64)
    sy = np.asarray(stars["star_y"], dtype=np.float64)
    row = np.asarray(stars["calstars_row"], dtype=np.intp)
    smag = np.asarray(stars["star_mag"], dtype=np.float32)
    snr = np.asarray(stars["star_flux_snr"], dtype=np.float32)
    n_cat = int(cat_id.max()) + 1

    # ---- Instrumental magnitudes: CALSTARS join, per-knot mag_lev zero ------
    # The knots are fitted on clear moments, so this removes the night's
    # photometric zero-point drift while preserving cloud dimming (the
    # harness's thin-cloud photometric engine lived in this convention).
    inst_mag = np.full(len(row), np.nan, dtype=np.float32)
    knot_t, knot_ml = [], []
    pp_path = os.path.join(night_dir, config.platepars_flux_recalibrated_name)
    if os.path.isfile(pp_path):
        with open(pp_path) as f:
            _ppr = json.load(f)
        from RMS.Formats import FFfile as _FF
        import calendar as _cal
        for ffn, v in _ppr.items():
            if isinstance(v, dict) and v.get("auto_recalibrated") \
                    and ("mag_lev" in v):
                knot_t.append(_cal.timegm(
                    _FF.filenameToDatetime(ffn).timetuple()))
                knot_ml.append(float(v["mag_lev"]))
    knot_t = np.asarray(knot_t, dtype=np.float64)
    knot_ml = np.asarray(knot_ml, dtype=np.float64)
    t_fr = np.asarray(frames["frame_time_unix"], dtype=np.float64)
    if len(knot_t):
        ml_frame = knot_ml[np.argmin(np.abs(knot_t[None, :]
            - t_fr[:, None]), axis=1)]
    else:
        ml_frame = np.zeros(len(t_fr))
    calstars_files = [f for f in sorted(os.listdir(night_dir))
                      if f.startswith("CALSTARS") and f.endswith(".txt")]
    if calstars_files:
        calstars_list, _ = CALSTARSFormat.readCALSTARS(night_dir, calstars_files[0])
        by_ff = {ff: np.asarray(data, dtype=np.float64)
                 for ff, data in calstars_list if len(data)}
        frame_names = [str(n) for n in frames["frame_names"]]
        for fi, ff_name in enumerate(frame_names):
            data = by_ff.get(ff_name)
            if data is None:
                continue
            m = (sf == fi) & (row >= 0)
            if not np.any(m):
                continue
            rows_here = row[m]
            valid = rows_here < len(data)
            # CALSTARS columns: y, x, intensity, amplitude, ...
            inten = data[rows_here[valid], 2]
            good = inten > 0
            idx = np.where(m)[0][valid][good]
            inst_mag[idx] = ml_frame[fi] - 2.5*np.log10(inten[good])

    # ---- Fine conditioning map (16x9 cumsum, the harness scheme) ------------
    # The 8x5 product map is too thin-cloud-blind to condition a calibration
    # on (mislabeled thin frames poison every rate and baseline - measured on
    # the A6 parity lab). This internal fine map uses per-record model p as
    # logits and drift-free photometric residuals; its only job is a clean
    # "clear" mask.
    width = float(np.ceil(np.nanmax(sx)/16.0)*16.0)
    height = float(np.ceil(np.nanmax(sy)/16.0)*16.0)
    NXF, NYF = 16, 9
    dome_s = 0.4
    grid_dm = np.arange(-0.3, 4.501, 0.03)
    ngd = len(grid_dm)
    sp = np.asarray(stars["star_p"], dtype=np.float64)
    lgm = np.log(np.clip(sp, 1e-4, 0.999)/(1 - np.clip(sp, 1e-4, 0.999)))
    detected_u = row != -1
    res_w = inst_mag - smag
    base_w = np.full(n_cat, np.nan)
    sig_w = np.full(n_cat, np.nan)
    mw = np.isfinite(res_w)
    if np.any(mw):
        order = np.argsort(cat_id[mw])
        cids = cat_id[mw][order]
        rvw = res_w[mw][order]
        uniq, starts = np.unique(cids, return_index=True)
        bounds = np.append(starts, len(cids))
        for u, a0, a1 in zip(uniq, bounds[:-1], bounds[1:]):
            if a1 - a0 >= MIN_PHOT_FRAMES:
                med = np.median(rvw[a0:a1])
                base_w[u] = med
                sig_w[u] = max(1.4826*np.median(np.abs(rvw[a0:a1] - med)), 0.05)
    cxf = np.clip((sx/(width/NXF)).astype(np.intp), 0, NXF - 1)
    cyf = np.clip((sy/(height/NYF)).astype(np.intp), 0, NYF - 1)
    cellf = cyf*NXF + cxf
    usable_rec = sp >= 0.03
    resid_wr = res_w - base_w[cat_id]

    # Records presorted by frame: per-frame indexing is a slice, not a
    # whole-night boolean scan per frame
    u_idx = np.where(usable_rec)[0]
    u_idx = u_idx[np.argsort(sf[u_idx], kind="stable")]
    frame_bounds = np.searchsorted(sf[u_idx], np.arange(n_frames + 1))

    def _frameCellLL(j):
        """ Per-cell (log-likelihood, count) contribution of one frame. """
        cll = np.zeros((NYF*NXF, ngd), dtype=np.float32)
        cn = np.zeros(NYF*NXF, dtype=np.int32)
        ridx = u_idx[frame_bounds[j]:frame_bounds[j + 1]]
        if len(ridx):
            lg = lgm[ridx][:, None]
            det = detected_u[ridx].astype(np.float64)[:, None]
            pr = np.clip(1.0/(1.0 + np.exp(-(lg - grid_dm[None, :]/dome_s))),
                1e-6, 1 - 1e-6)
            q = det*pr + (1 - det)*(1 - pr)
            ll = np.log(0.98*q + 0.01)
            selp = np.isfinite(resid_wr[ridx]) & (row[ridx] >= 0)
            if np.any(selp):
                rr = resid_wr[ridx][selp][:, None]
                ss_ = sig_w[cat_id[ridx][selp]][:, None]
                ll[selp] += -0.5*np.minimum(
                    ((rr - grid_dm[None, :])/ss_)**2, 9.0)
            np.add.at(cll, cellf[ridx], ll.astype(np.float32))
            np.add.at(cn, cellf[ridx], 1)
        return cll, cn

    # Rolling +-6-frame windows instead of a whole-night cumsum: the adaptive
    # windows never reach further, so a 13-frame ring cache replaces the
    # [n_frames, cells, grid] tensor whose cumsum peaked over a gigabyte on a
    # long night - a Pi-class OOM hazard
    MAX_HALF = 6
    cache = {}
    dm_fine = np.full((n_frames, NYF*NXF), np.nan, dtype=np.float32)
    for j in range(n_frames):
        for k in range(max(0, j - MAX_HALF), min(n_frames, j + MAX_HALF + 1)):
            if k not in cache:
                cache[k] = _frameCellLL(k)
        for k in list(cache):
            if k < j - MAX_HALF:
                del cache[k]
        s_ll = cache[j][0].copy()
        s_n = cache[j][1].astype(np.int64)
        done = np.zeros(NYF*NXF, dtype=bool)
        for half in range(1, MAX_HALF + 1):
            for k in (j - half, j + half):
                if 0 <= k < n_frames:
                    s_ll += cache[k][0]
                    s_n += cache[k][1]
            ready = (~done) & (s_n >= 12)
            if np.any(ready):
                dm_fine[j, ready] = np.maximum(0.0,
                    grid_dm[np.argmax(s_ll[ready], axis=1)])
                done |= ready
            if done.all():
                break
    del cache

    cell_dm = dm_fine[sf, cellf]
    clear = np.isfinite(cell_dm) & (cell_dm < CLEAR_DM_MAX)

    # A night with essentially no clear sky teaches nothing about clear-sky
    # behavior. Skip such nights entirely (the EMA just waits).
    clear_frac = float(np.mean(np.bincount(sf[clear], minlength=n_frames) > 0))
    if clear_frac < 0.15:
        print("Star calibration: only {:.0%} of frames have any clear cell - "
              "skipping this night".format(clear_frac))
        return None

    # Channel rates on clear frames: the UNION channel is what the tree
    # consumes (CALSTARS + FF-forced + stills)
    seen_cs = np.bincount(cat_id[clear], minlength=n_cat)
    match_cs = np.bincount(cat_id[clear & (row != -1)], minlength=n_cat)

    meas_f = clear & np.isfinite(snr)
    seen_fo = np.bincount(cat_id[meas_f], minlength=n_cat)
    det_fo = np.bincount(cat_id[meas_f & (snr >= 5.0)], minlength=n_cat)

    # Airmass per clear matched record, from catalog positions and frame times
    resid_ok = clear & (row >= 0) & np.isfinite(inst_mag)
    X = np.full(len(row), np.nan, dtype=np.float64)
    if np.any(resid_ok):
        catalog_stars, _, _ = StarCatalog.readStarCatalog(
            config.star_catalog_path, config.star_catalog_file,
            lim_mag=float(header["catalog_lim_mag"]),
            mag_band_ratios=config.star_catalog_band_ratios)
        cat_ra = catalog_stars[:, 0]
        cat_dec = catalog_stars[:, 1]

        # lat/lon from any recalibrated platepar
        pp_path = os.path.join(night_dir, config.platepars_flux_recalibrated_name)
        lat = lon = None
        if os.path.isfile(pp_path):
            with open(pp_path) as f:
                ppr = json.load(f)
            for v in ppr.values():
                if isinstance(v, dict) and ("lat" in v):
                    lat, lon = float(v["lat"]), float(v["lon"])
                    break
        if lat is not None:
            t_unix = np.asarray(frames["frame_time_unix"], dtype=np.float64)
            jd_frame = t_unix/86400.0 + 2440587.5
            for fi in np.unique(sf[resid_ok]):
                m = resid_ok & (sf == fi)
                _, alt = raDec2AltAz(cat_ra[cat_id[m]], cat_dec[cat_id[m]],
                    float(jd_frame[fi]), lat, lon)
                X[m] = _kastenYoung(np.asarray(alt))

    # Nightly extinction slope from pooled per-star-demeaned residuals
    resid_raw = inst_mag - smag
    k_fit = 0.0
    fit_ok = resid_ok & np.isfinite(X)
    if np.any(fit_ok):
        order = np.argsort(cat_id[fit_ok])
        cids = cat_id[fit_ok][order]
        rr = resid_raw[fit_ok][order].astype(np.float64)
        xx = X[fit_ok][order]
        # per-star means via segmented reduction
        uniq, starts = np.unique(cids, return_index=True)
        counts = np.diff(np.append(starts, len(cids)))
        r_mean = np.repeat(np.add.reduceat(rr, starts)/counts, counts)
        x_mean = np.repeat(np.add.reduceat(xx, starts)/counts, counts)
        dr, dx_ = rr - r_mean, xx - x_mean
        denom = float(np.sum(dx_**2))
        if denom > 1.0:
            k_fit = float(np.clip(np.sum(dr*dx_)/denom, 0.0, K_FIT_MAX))

    # Per-star baseline and scatter of the airmass-corrected residual
    resid_corr = np.where(fit_ok, resid_raw - k_fit*(X - 1.0), np.nan)
    base = np.full(n_cat, np.nan, dtype=np.float32)
    sigma = np.full(n_cat, np.nan, dtype=np.float32)
    if np.any(fit_ok):
        order = np.argsort(cat_id[fit_ok])
        cids = cat_id[fit_ok][order]
        rv = resid_corr[fit_ok][order]
        uniq, starts = np.unique(cids, return_index=True)
        bounds = np.append(starts, len(cids))
        for u, s0, s1 in zip(uniq, bounds[:-1], bounds[1:]):
            if s1 - s0 >= MIN_PHOT_FRAMES:
                vals = rv[s0:s1]
                med = np.median(vals)
                base[u] = med
                sigma[u] = max(1.4826*np.median(np.abs(vals - med)), 0.05)

    with np.errstate(invalid="ignore"):
        rate_cs = np.where(seen_cs >= MIN_CLEAR_FRAMES,
            match_cs/np.maximum(seen_cs, 1), np.nan)
        rate_fo = np.where(seen_fo >= MIN_CLEAR_FRAMES,
            det_fo/np.maximum(seen_fo, 1), np.nan)

    return dict(
        n_cat=n_cat,
        rate_calstars=rate_cs.astype(np.float32),
        rate_forced=rate_fo.astype(np.float32),
        base_mag=base,
        sigma_mag=sigma,
        seen_clear=seen_cs.astype(np.int32),
        k_fit=k_fit,
        catalog_lim_mag=float(header["catalog_lim_mag"]),
        night=night_name,
        # Diagnostic view of the internal conditioner (not persisted -
        # updateStarCalibration only saves its named fields). The parity
        # regression test diffs these against a harness reference.
        _dm_fine=dm_fine,
        _clear=clear,
        _rec_sf=sf,
        _rec_cat_id=cat_id,
    )


def updateStarCalibration(config, night_dir):
    """ Fold one processed night into the station's trailing calibration file.

    Return:
        path: [str] Calibration file path, or None if the night lacked
            prerequisites (never raises - callers wrap in a guarded try).
    """

    stats = computeNightStarStats(config, night_dir)
    if stats is None:
        return None

    path = os.path.join(os.path.expanduser(config.data_dir),
        calibrationFileName(config.stationID))

    prev = None
    if os.path.isfile(path):
        try:
            prev_header, prev_arrays = loadStarCalibration(path)
            # A depth change means the ids no longer join - reset
            if abs(prev_header.get("catalog_lim_mag", -99)
                    - stats["catalog_lim_mag"]) < 0.01:
                prev = prev_arrays
        except Exception:
            prev = None

    n_cat = stats["n_cat"]
    fields = ("rate_calstars", "rate_forced", "base_mag", "sigma_mag")

    if prev is not None:
        n_cat = max(n_cat, len(prev["rate_calstars"]))

    merged = {}
    n_nights = np.zeros(n_cat, dtype=np.int16)
    if prev is not None:
        n_prev = len(prev["rate_calstars"])
        n_nights[:n_prev] = prev.get("n_nights", np.zeros(n_prev, dtype=np.int16))

    for f in fields:
        new = np.full(n_cat, np.nan, dtype=np.float32)
        new[:len(stats[f])] = stats[f]
        if prev is None:
            merged[f] = new
            continue
        old = np.full(n_cat, np.nan, dtype=np.float32)
        old[:len(prev[f])] = prev[f]
        # EMA where both exist; adopt whichever side exists alone
        both = np.isfinite(old) & np.isfinite(new)
        out = np.where(both, (1 - EMA_ALPHA)*old + EMA_ALPHA*new,
            np.where(np.isfinite(new), new, old))
        merged[f] = out.astype(np.float32)

    contributed = np.zeros(n_cat, dtype=bool)
    contributed[:len(stats["rate_calstars"])] = np.isfinite(stats["rate_calstars"])
    n_nights[contributed] += 1

    k_prev = prev_header.get("k_ema") if (prev is not None) else None
    k_ema = stats["k_fit"] if k_prev is None \
        else (1 - EMA_ALPHA)*float(k_prev) + EMA_ALPHA*stats["k_fit"]

    header = dict(
        schema_version=SCHEMA_VERSION,
        stationID=str(config.stationID),
        catalog_lim_mag=stats["catalog_lim_mag"],
        k_ema=float(k_ema),
        k_last_night=float(stats["k_fit"]),
        last_night=stats["night"],
        ema_alpha=EMA_ALPHA,
        clear_dm_max=CLEAR_DM_MAX,
    )

    np.savez_compressed(path.replace(".npz", ""),
        header=json.dumps(header),
        n_nights=n_nights,
        **merged)

    print("Star calibration updated: {:d} stars with rates, k_fit {:.2f} "
          "(EMA {:.2f})".format(int(np.isfinite(merged["rate_calstars"]).sum()),
          stats["k_fit"], k_ema))

    return path


def loadStarCalibration(path):
    """ Load the trailing calibration.

    Return:
        (header, arrays): header dict and cat_id-indexed arrays.
    """

    with np.load(path, allow_pickle=False) as z:
        header = json.loads(str(z["header"]))
        arrays = {k: z[k] for k in z.files if k != "header"}

    return header, arrays
