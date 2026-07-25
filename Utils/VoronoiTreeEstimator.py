""" The multiscale Voronoi tree transparency estimator - the production port of
the estimator selected by the synthetic ground-truth benchmark (clean-data
results: RMS 0.29/0.48/0.80 vs the grid's 0.39/0.59/0.98 on clear/discs/bank,
boundary IoU 0.86 vs 0.77, opaque-core bias -0.14 vs -1.72, and edge timing
within one frame vs the grid's ~6 minutes early).

Structure (validated in the demo harness, judged on real nights):

- Anchors are STARS, in celestial coordinates (3-D unit vectors; chord metric -
  flat charts distort near the poles). Nested nearest-anchor membership builds
  a 3-level Voronoi tree: sparse reliable stars own large cells, dense faint
  stars refine locally. Strata are chosen by MEASURED detectability (the
  trailing per-star calibration; model-p only when no history exists) - no
  magnitude constants.
- Per frame, each leaf accumulates log-likelihood over an extinction grid from
  its stars' evidence: detection bits (CALSTARS or forced-photometry, weighted
  by each star's own measured clear rate) and photometric residuals against
  the star's airmass-corrected baseline.
- Scale-recursive belief propagation with ROOT DECOUPLING: an effectively flat
  top prior (TAU[0] wide) so a solid overcast saturates ("at least this
  opaque" - one-sided evidence must not be shrunk toward the sky average), and
  robust heavy-tailed transitions (a flat mixture component) so sharp spatial
  steps survive. Posterior MODE per leaf.
- Leaves aggregate to the SAME 8x5 cell grid and npz schema as the grid
  product - consumers (transparencyAt) never see the difference. During the
  parallel run the tree writes <night>_transparency_map_tree.npz alongside
  the grid product; the flip renames it to the canonical file.
"""

from __future__ import absolute_import, division, print_function

import json
import os

import numpy as np


GRID_DM = np.arange(-0.3, 4.501, 0.03)     # extinction grid (mag)
TAU = (3.0, 0.6, 0.3)                      # scale-transition widths, root first
                                           # (root wide = decoupled)
EPS_MIX = 0.1                              # heavy-tail flat mixture in transitions
EVIDENCE_MIX = 0.98                        # per-star likelihood robustness floor
RATE_MIN_LEAF = 0.15                       # measured clear rate to carry evidence
RATE_CLIP = (0.03, 0.97)
WINDOW_FRAMES = 1                          # +/- frames pooled per BP solve
S_DEFAULT = 0.4                            # logistic width fallback (mag)

CELL_NX, CELL_NY = 8, 5

TREE_MAP_SUFFIX = "transparency_map_tree.npz"


def treeMapFileName(night_name):
    return "{:s}_{:s}".format(night_name, TREE_MAP_SUFFIX)


def _unitVectors(ra_deg, dec_deg):
    r, d = np.radians(ra_deg), np.radians(dec_deg)
    return np.column_stack([np.cos(d)*np.cos(r), np.cos(d)*np.sin(r), np.sin(d)])


def _smooth(p, tau):
    """ Gaussian smoothing along the dm axis plus the robust flat mixture. """

    from scipy.ndimage import gaussian_filter1d

    q = gaussian_filter1d(p, tau/0.03, axis=-1, mode="nearest")
    q = (1.0 - EPS_MIX)*q + EPS_MIX/p.shape[-1]
    return q


def computeTreeSeries(config, night_dir, header, frames, stars, calibration=None):
    """ Run the tree estimator over a night's scoring product.

    Arguments:
        config: [Config]
        night_dir: [str] Night directory (CALSTARS join for photometry).
        header, frames, stars: [dicts] From StarScoring.loadStarScoring
            (schema >= 4 required: star identity and the forced channel).

    Keyword arguments:
        calibration: [(header, arrays) or None] Trailing star calibration
            (Utils.StarCalibration). In-night statistics are used where it is
            absent or the catalog depth mismatches.

    Return:
        (t_unix, dm_cells, flags, leaf_cat_id, leaf_dm): the cell arrays per
            the TransparencyMap schema, plus the LEAF channel - each leaf's
            anchor star id and its per-frame posterior-mode dm. The cells are
            the consumer product; the leaves are the estimator's native
            resolution (rendered by the demo video, and the basis of any
            future finer-grained consumer). None if the product predates
            schema v4.
    """

    from scipy.spatial import cKDTree
    from RMS.Formats import StarCatalog
    from RMS.Formats.TransparencyMap import (DM_CEILING, FLAG_CEILING,
        FLAG_MOON_DOMAIN, FLAG_NO_DATA, MOON_PHASE_BRIGHT)

    if int(header.get("schema_version", 0)) < 4 or "star_cat_id" not in stars:
        return None

    cat_id = np.asarray(stars["star_cat_id"], dtype=np.int64)
    sf = np.asarray(stars["star_frame"], dtype=np.int64)
    sx = np.asarray(stars["star_x"], dtype=np.float64)
    sy = np.asarray(stars["star_y"], dtype=np.float64)
    row = np.asarray(stars["calstars_row"], dtype=np.int64)
    n_frames = len(frames["frame_names"])
    n_cat = int(cat_id.max()) + 1

    detected = row != -1

    # ---- Per-star rates: trailing calibration first, in-night fallback -------
    rate = np.full(n_cat, np.nan, dtype=np.float64)
    base = np.full(n_cat, np.nan, dtype=np.float64)
    sigma = np.full(n_cat, np.nan, dtype=np.float64)
    k_ext = 0.0
    if calibration is not None:
        cal_h, cal = calibration
        if abs(cal_h.get("catalog_lim_mag", -99)
                - float(header["catalog_lim_mag"])) < 0.01:
            n = min(n_cat, len(cal["rate_calstars"]))
            rc = cal["rate_calstars"][:n].astype(np.float64)
            rf = cal["rate_forced"][:n].astype(np.float64)
            # Combined-channel rate: the product's detection bit is the union
            with np.errstate(invalid="ignore"):
                rate[:n] = np.where(np.isfinite(rf),
                    1.0 - (1.0 - np.fmin(rc, 0.99))*(1.0 - np.fmin(rf, 0.99)),
                    rc)
            base[:n] = cal["base_mag"][:n]
            sigma[:n] = cal["sigma_mag"][:n]
            k_ext = float(cal_h.get("k_ema", 0.0))

    # Night-1 fallback: the MODEL detection probability, not the in-night
    # measured rate. In-night rates conflate weather with detectability - on a
    # night overcast from dusk they make the overcast the baseline and the
    # whole deck reads dm ~ 0 (observed on a recovered CAWEC4 night). The dome
    # model was fit on clear archived nights, so its per-star p IS a clear-sky
    # baseline; the trailing calibration replaces it from night 2 on.
    p_sum = np.bincount(cat_id, weights=stars["star_p"].astype(np.float64),
        minlength=n_cat)
    seen_n = np.bincount(cat_id, minlength=n_cat)
    with np.errstate(invalid="ignore"):
        p_model = np.where(seen_n >= 10, p_sum/np.maximum(seen_n, 1), np.nan)
    rate = np.where(np.isfinite(rate), rate, p_model)

    qualified = np.isfinite(rate) & (rate >= RATE_MIN_LEAF)
    if qualified.sum() < 30:
        return None

    p_clear = np.clip(rate, *RATE_CLIP)
    logit0 = np.log(p_clear/(1.0 - p_clear))
    dome_s = float(header.get("dome_s", S_DEFAULT)) or S_DEFAULT

    # ---- Photometric channel: CALSTARS intensity join ------------------------
    inst_resid = np.full(len(row), np.nan, dtype=np.float64)
    try:
        from RMS.Formats import CALSTARS as CALSTARSFormat
        calstars_files = [f for f in sorted(os.listdir(night_dir))
                          if f.startswith("CALSTARS") and f.endswith(".txt")]
        if calstars_files:
            calstars_list, _ = CALSTARSFormat.readCALSTARS(night_dir,
                calstars_files[0])
            by_ff = {ff: np.asarray(d, dtype=np.float64)
                     for ff, d in calstars_list if len(d)}
            fnames = [str(n) for n in frames["frame_names"]]
            smag = np.asarray(stars["star_mag"], dtype=np.float64)
            for fi, ff_name in enumerate(fnames):
                data = by_ff.get(ff_name)
                if data is None:
                    continue
                m = (sf == fi) & (row >= 0)
                if not np.any(m):
                    continue
                rows_here = row[m]
                valid = rows_here < len(data)
                inten = data[rows_here[valid], 2]
                good = inten > 0
                idx = np.where(m)[0][valid][good]
                inst_resid[idx] = -2.5*np.log10(inten[good]) - smag[idx]
    except Exception:
        pass

    # Baselines: in-night medians where the trailing file has none
    has_resid = np.isfinite(inst_resid)
    if np.any(has_resid):
        order = np.argsort(cat_id[has_resid])
        cids = cat_id[has_resid][order]
        rv = inst_resid[has_resid][order]
        uniq, starts = np.unique(cids, return_index=True)
        bounds = np.append(starts, len(cids))
        for u, s0, s1 in zip(uniq, bounds[:-1], bounds[1:]):
            if not np.isfinite(base[u]) and (s1 - s0) >= 10:
                vals = rv[s0:s1]
                med = np.median(vals)
                base[u] = med
                sigma[u] = max(1.4826*np.median(np.abs(vals - med)), 0.05)
    phot_ok = np.isfinite(base) & np.isfinite(sigma)

    # ---- Tree construction ---------------------------------------------------
    catalog_stars, _, _ = StarCatalog.readStarCatalog(
        config.star_catalog_path, config.star_catalog_file,
        lim_mag=float(header["catalog_lim_mag"]),
        mag_band_ratios=config.star_catalog_band_ratios)
    pos_all = _unitVectors(catalog_stars[:, 0], catalog_stars[:, 1])

    q_ids = np.where(qualified)[0]
    q_ids = q_ids[q_ids < len(pos_all)]
    pos = pos_all[q_ids]

    # Strata by measured reliability: leaves = all qualified; mid = upper half;
    # root level = the most reliable, thinned for coverage (cap for BP cost)
    r_q = rate[q_ids]
    mid_thr = np.nanmedian(r_q)
    l1_mask = r_q >= mid_thr
    l0_thr = np.nanpercentile(r_q, 90)
    l0_mask = r_q >= l0_thr
    if l0_mask.sum() > 250:
        keep = np.zeros(len(q_ids), dtype=bool)
        idx0 = np.where(l0_mask)[0]
        keep[idx0[::max(1, len(idx0)//250)]] = True
        l0_mask = keep
    if l1_mask.sum() < 10 or l0_mask.sum() < 3:
        return None

    A2 = np.arange(len(q_ids))
    A1 = np.where(l1_mask)[0]
    A0 = np.where(l0_mask)[0]

    leaf_parent = A1[cKDTree(pos[A1]).query(pos)[1]]      # leaf -> L1 anchor
    l1_parent = A0[cKDTree(pos[A0]).query(pos[A1])[1]]    # L1 -> L0 anchor
    l1_index = {a: i for i, a in enumerate(A1)}
    l0_index = {a: i for i, a in enumerate(A0)}
    leaf_parent_i = np.array([l1_index[a] for a in leaf_parent])
    l1_parent_i = np.array([l0_index[a] for a in l1_parent])
    n_leaf, n_l1, n_l0 = len(A2), len(A1), len(A0)

    leaf_of_cat = np.full(n_cat, -1, dtype=np.int64)
    leaf_of_cat[q_ids] = A2

    NG = len(GRID_DM)

    # ---- Per-frame star evidence, accumulated to leaves ----------------------
    rec_leaf = leaf_of_cat[cat_id]
    usable = rec_leaf >= 0

    def frameLeafLL(fi):
        out = np.zeros((n_leaf, NG))
        m = usable & (sf == fi)
        if not np.any(m):
            return out
        cid = cat_id[m]
        lg = logit0[cid][:, None]
        det = detected[m].astype(np.float64)[:, None]
        pr = 1.0/(1.0 + np.exp(-(lg - GRID_DM[None, :]/dome_s)))
        pr = np.clip(pr, 1e-6, 1 - 1e-6)
        q = det*pr + (1 - det)*(1 - pr)
        ll = np.log(EVIDENCE_MIX*q + (1 - EVIDENCE_MIX)/2)
        pm = m & (row >= 0) & np.isfinite(inst_resid) & phot_ok[cat_id]
        if np.any(pm):
            sel = pm[m]
            rr = (inst_resid[pm] - base[cat_id[pm]])[:, None]
            ss = sigma[cat_id[pm]][:, None]
            ll[sel] += -0.5*np.minimum(((rr - GRID_DM[None, :])/ss)**2, 9.0)
        np.add.at(out, rec_leaf[m], ll)
        return out

    # ---- BP over the night with a ring cache for the +/- window --------------
    t_unix = np.asarray(frames["frame_time_unix"], dtype=np.float64)
    dm_cells = np.full((n_frames, CELL_NY, CELL_NX), np.nan, dtype=np.float32)
    leaf_dm_all = np.full((n_frames, n_leaf), np.nan, dtype=np.float32)

    # Leaf -> cell assignment per frame from the stars' image positions: a
    # leaf's cell is where its own star currently sits (median if several recs)
    width = float(np.ceil(np.nanmax(sx)/16.0)*16.0)
    height = float(np.ceil(np.nanmax(sy)/16.0)*16.0)
    cxr = np.clip((sx/(width/CELL_NX)).astype(np.intp), 0, CELL_NX - 1)
    cyr = np.clip((sy/(height/CELL_NY)).astype(np.intp), 0, CELL_NY - 1)

    cache = {}
    for fi in range(n_frames):

        lls = None
        for k in range(fi - WINDOW_FRAMES, fi + WINDOW_FRAMES + 1):
            if not (0 <= k < n_frames):
                continue
            if k not in cache:
                cache[k] = frameLeafLL(k)
            lls = cache[k] if lls is None else lls + cache[k]
        for k in list(cache):
            if k < fi - WINDOW_FRAMES:
                del cache[k]

        # Upward pass
        up2 = _smooth(lls, TAU[2])
        l1_in = np.zeros((n_l1, NG))
        np.add.at(l1_in, leaf_parent_i, up2)
        up1 = _smooth(l1_in, TAU[1])
        l0_in = np.zeros((n_l0, NG))
        np.add.at(l0_in, l1_parent_i, up1)
        up0 = _smooth(l0_in, TAU[0])
        root = up0.sum(axis=0)

        # Downward pass
        down0 = _smooth(root[None, :] - up0, TAU[0])
        down1 = _smooth((l0_in + down0)[l1_parent_i] - up1, TAU[1])
        down2 = _smooth((l1_in + down1)[leaf_parent_i] - up2, TAU[2])

        belief = lls + down2
        leaf_dm = np.maximum(0.0, GRID_DM[np.argmax(belief, axis=1)])
        # A leaf with no in-window evidence has a flat belief - do not report it
        has_ev = np.abs(lls).sum(axis=1) > 0
        leaf_dm_all[fi, has_ev] = leaf_dm[has_ev]

        # Aggregate leaves to cells by the leaves' current image positions
        m = usable & (sf == fi)
        if not np.any(m):
            continue
        lf = rec_leaf[m]
        vals = leaf_dm[lf]
        flat = cyr[m]*CELL_NX + cxr[m]
        sums = np.bincount(flat, weights=vals, minlength=CELL_NY*CELL_NX)
        cnts = np.bincount(flat, minlength=CELL_NY*CELL_NX)
        with np.errstate(invalid="ignore"):
            cell = np.where(cnts > 0, sums/np.maximum(cnts, 1), np.nan)
        dm_cells[fi] = cell.reshape(CELL_NY, CELL_NX)

    # ---- Flags (same semantics as the grid product) --------------------------
    flags = np.zeros(dm_cells.shape, dtype=np.uint8)
    flags[~np.isfinite(dm_cells)] = FLAG_NO_DATA
    ceiling = np.isfinite(dm_cells) & (dm_cells >= DM_CEILING)
    dm_cells = np.where(ceiling, DM_CEILING, dm_cells)
    flags[ceiling] |= FLAG_CEILING
    moon = (np.asarray(frames["moon_alt"]) > 0) \
        & (np.asarray(frames["moon_phase"]) > MOON_PHASE_BRIGHT)
    flags[moon, :, :] |= FLAG_MOON_DOMAIN

    leaf_cat_id = q_ids[A2]

    return (t_unix, dm_cells.astype(np.float16), flags,
        leaf_cat_id.astype(np.int32), leaf_dm_all.astype(np.float16))


def computeAndSaveTreeMap(config, night_dir):
    """ Full station entry point: compute the tree map for a processed night and
        write <night>_transparency_map_tree.npz. Returns the path or None.
    """

    from RMS.Formats.StarScoring import loadStarScoring, scoringFileName

    night_name = os.path.basename(os.path.normpath(night_dir))
    scoring_path = os.path.join(night_dir, scoringFileName(night_name))
    if not os.path.isfile(scoring_path):
        return None

    header, frames, stars = loadStarScoring(scoring_path)

    calibration = None
    try:
        from Utils.StarCalibration import calibrationFileName, loadStarCalibration
        cal_path = os.path.join(os.path.expanduser(config.data_dir),
            calibrationFileName(config.stationID))
        if os.path.isfile(cal_path):
            calibration = loadStarCalibration(cal_path)
    except Exception:
        calibration = None

    result = computeTreeSeries(config, night_dir, header, frames, stars,
        calibration=calibration)
    if result is None:
        return None
    t_unix, dm_cells, flags, leaf_cat_id, leaf_dm = result

    out_header = dict(
        schema_version=1,
        estimator="voronoi_tree/1",
        stationID=str(config.stationID),
        night=night_name,
        nx=CELL_NX, ny=CELL_NY,
        dome_s=float(header.get("dome_s", S_DEFAULT)),
        catalog_lim_mag=float(header["catalog_lim_mag"]),
        cadence=str(header.get("cadence", "per_ff")),
        note="dm is extinction (mag) vs the night's clear baseline; "
             "flags: 1=no_data 2=moon_domain 4=ceiling(lower bound); "
             "multiscale Voronoi tree estimator (parallel-run product)",
    )

    path = os.path.join(night_dir, treeMapFileName(night_name))
    np.savez_compressed(path.replace(".npz", ""),
        header=json.dumps(out_header),
        t_unix=t_unix,
        dm=dm_cells,
        flags=flags,
        leaf_cat_id=leaf_cat_id,
        leaf_dm=leaf_dm)

    print("Tree transparency map: {:s}".format(os.path.basename(path)))

    return path
