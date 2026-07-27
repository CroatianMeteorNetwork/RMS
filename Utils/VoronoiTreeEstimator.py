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


_SMOOTH_MAT = {}


def _smooth(msgs, tau):
    """ BP message smoothing: exponentiate, Gaussian-smooth along the dm axis
    in probability space, NORMALIZE, then apply the robust flat mixture and
    return to log space. The normalization is load-bearing: it caps every
    node's message at log((1-eps)*NG/eps) nats regardless of how many records
    the branch aggregates, which is what stops a well-observed clear region
    from flooding a covered minority through the root (smoothing raw
    log-likelihoods instead lets message strength grow without bound with
    record count).

    The grid is fixed, so each tau's filter (with its boundary handling) is a
    constant linear operator - one precomputed matrix per tau, applied as a
    BLAS matmul. The taps-based filter was the BP hot spot: tau 3.0 on the
    0.03 grid is an 801-tap kernel over a 161-bin axis.
    """

    ng = msgs.shape[-1]
    M = _SMOOTH_MAT.get((tau, ng))
    if M is None:
        from scipy.ndimage import gaussian_filter1d
        M = gaussian_filter1d(np.eye(ng), tau/0.03, axis=0,
            mode="nearest").T.astype(np.float32)
        _SMOOTH_MAT[(tau, ng)] = M
    p = np.exp(msgs - msgs.max(axis=-1, keepdims=True)).astype(np.float32)
    p = p @ M
    p /= np.maximum(p.sum(axis=-1, keepdims=True), 1e-30)
    p = (1.0 - EPS_MIX)*p + EPS_MIX/ng
    return np.log(p)


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

    # Empty product (no scored frames / no records): nothing to estimate
    if (len(frames.get("frame_names", [])) == 0) \
            or (len(np.asarray(stars["star_cat_id"])) == 0):
        return None

    cat_id = np.asarray(stars["star_cat_id"], dtype=np.intp)
    sf = np.asarray(stars["star_frame"], dtype=np.intp)
    sx = np.asarray(stars["star_x"], dtype=np.float64)
    sy = np.asarray(stars["star_y"], dtype=np.float64)
    row = np.asarray(stars["calstars_row"], dtype=np.intp)
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

    # Normalize the model seed to the NIGHT'S OWN clear level (the same
    # high-percentile scheme as the grid detector's norms): the fleet's
    # models sit anywhere from 2.5x over-prediction (fresh refit, warmup
    # norm ~0.4 - raw seeding paints clear sky solid) to 1.5x under-
    # prediction (MW transit - thin clouds become invisible because stars
    # beat expectations). A high percentile of the per-frame ratio tracks
    # the clearest moments, so ordinary cloud does not drag the zero point;
    # a night overcast END TO END clamps at NORM_MIN and still saturates.
    # The trailing calibration (measured rates) needs none of this.
    sp = stars["star_p"].astype(np.float64)
    dark = np.asarray(frames["sun_alt"], dtype=np.float64) <= -18.0
    fr_exp = np.bincount(sf, weights=sp, minlength=n_frames)
    fr_det = np.bincount(sf[detected], weights=np.ones(int(detected.sum())),
        minlength=n_frames)
    informative = dark & (fr_exp >= 5.0)
    if np.any(informative):
        seed_norm = float(np.clip(np.percentile(
            fr_det[informative]/fr_exp[informative], 80), 0.5, 3.0))
    else:
        seed_norm = 1.0
    p_model = np.clip(p_model*seed_norm, 0.0, 0.97)

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
            # Per-knot mag_lev zero point (clear-moment fits): removes the
            # night's photometric drift while preserving cloud dimming - the
            # same convention as the calibration baselines
            knot_t, knot_ml = [], []
            pp_path = os.path.join(night_dir,
                config.platepars_flux_recalibrated_name)
            if os.path.isfile(pp_path):
                with open(pp_path) as f:
                    _ppr = json.load(f)
                from RMS.Formats import FFfile as _FF
                import calendar as _cal
                for ffn, v in _ppr.items():
                    if isinstance(v, dict) and v.get("auto_recalibrated")                             and ("mag_lev" in v):
                        knot_t.append(_cal.timegm(
                            _FF.filenameToDatetime(ffn).timetuple()))
                        knot_ml.append(float(v["mag_lev"]))
            knot_t = np.asarray(knot_t, dtype=np.float64)
            knot_ml = np.asarray(knot_ml, dtype=np.float64)
            t_fr = np.asarray(frames["frame_time_unix"], dtype=np.float64)
            ml_frame = (knot_ml[np.argmin(np.abs(knot_t[None, :]
                - t_fr[:, None]), axis=1)] if len(knot_t)
                else np.zeros(len(t_fr)))
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
                inst_resid[idx] = (ml_frame[fi]
                    - 2.5*np.log10(inten[good])) - smag[idx]
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
                # Clear-biased baseline: the 30th percentile tracks the star's
                # BRIGHT (clear) states - a whole-night median on a cloudy
                # night absorbs the dimming into the baseline and kills the
                # photometric channel's thin-cloud sensitivity
                med = np.percentile(vals, 30)
                base[u] = med
                sigma[u] = max(1.4826*np.median(np.abs(vals - med)), 0.05)
    phot_ok = np.isfinite(base) & np.isfinite(sigma)

    # ---- Tree construction ---------------------------------------------------
    catalog_stars, _, _ = StarCatalog.readStarCatalog(
        config.star_catalog_path, config.star_catalog_file,
        lim_mag=float(header["catalog_lim_mag"]),
        mag_band_ratios=config.star_catalog_band_ratios)
    pos_all = _unitVectors(catalog_stars[:, 0], catalog_stars[:, 1])

    # Airmass consistency: the calibration stores baselines at reference
    # airmass (it subtracts k*(X-1) before fitting) - the evidence residuals
    # must live in the same frame or every low-elevation star reads its own
    # extinction as dimming
    if abs(k_ext) > 1e-3 and np.any(np.isfinite(inst_resid)):
        from RMS.Astrometry.Conversions import raDec2AltAz
        lat = lon = None
        try:
            with open(os.path.join(night_dir,
                    config.platepars_flux_recalibrated_name)) as f:
                _ppr = json.load(f)
            for v in _ppr.values():
                if isinstance(v, dict) and ("lat" in v):
                    lat, lon = float(v["lat"]), float(v["lon"])
                    break
        except Exception:
            pass
        if lat is not None:
            t_all = np.asarray(frames["frame_time_unix"], dtype=np.float64)
            jd_frame = t_all/86400.0 + 2440587.5
            r_ok = np.isfinite(inst_resid)
            for fi in np.unique(sf[r_ok]):
                m_ = r_ok & (sf == fi)
                _, alt_ = raDec2AltAz(
                    catalog_stars[cat_id[m_], 0], catalog_stars[cat_id[m_], 1],
                    float(jd_frame[fi]), lat, lon)
                h_ = np.maximum(np.asarray(alt_), 2.0)
                X_ = 1.0/(np.sin(np.radians(h_))
                          + 0.50572*(h_ + 6.07995)**-1.6364)
                inst_resid[m_] = inst_resid[m_] - k_ext*(X_ - 1.0)


    q_ids = np.where(qualified)[0]
    q_ids = q_ids[q_ids < len(pos_all)]
    pos = pos_all[q_ids]

    # Upper strata must be SPATIALLY BALANCED, not globally rate-ranked: seeded
    # rates fall with altitude (the dome model predicts lower p at low
    # elevation), so a global rate cut concentrates every parent anchor in the
    # high-altitude field. Children in an uncovered region then attach to
    # far-away parents whose sibling majority is clear, and BP's sibling flood
    # crushes a locally-covered minority to dm 0 (observed: a half-covered
    # USV001 night reading fully clear). Greedy best-rate-first selection with
    # a minimum angular separation gives every region its own parents - the
    # spatial spread the harness got for free from magnitude strata.
    r_q = rate[q_ids]

    def _spreadSelect(min_sep_chord, max_n):
        order = np.argsort(-np.nan_to_num(r_q))
        chosen = []
        for i in order:
            if len(chosen) >= max_n:
                break
            if chosen:
                d2 = np.sum((pos[np.array(chosen)] - pos[i])**2, axis=1)
                if d2.min() < min_sep_chord**2:
                    continue
            chosen.append(int(i))
        return np.array(chosen, dtype=np.intp)

    # Footprint angular radius from the qualified stars themselves
    centroid = pos.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    max_chord = float(np.sqrt(np.max(np.sum((pos - centroid)**2, axis=1))))

    # Stratum densities match the validated harness geometry (leaf:L1:L0 ~
    # 15:5:1). Coarser parents aggregate too many siblings and BP's sibling
    # flood crushes locally-covered minorities (measured on the A6 A/B:
    # record-level likelihoods agreed while leaf modes read 0 - the crush was
    # pure tree geometry). Separation radii follow uniform-coverage scaling.
    A2 = np.arange(len(q_ids))
    n0 = max(20, len(q_ids)//15)
    n1 = max(100, len(q_ids)//3)
    A0 = _spreadSelect(max_chord*np.sqrt(2.0/n0), n0)
    A1 = _spreadSelect(max_chord*np.sqrt(2.0/n1), n1)
    if len(A1) < 10 or len(A0) < 3:
        return None

    leaf_parent = A1[cKDTree(pos[A1]).query(pos)[1]]      # leaf -> L1 anchor
    l1_parent = A0[cKDTree(pos[A0]).query(pos[A1])[1]]    # L1 -> L0 anchor
    l1_index = {a: i for i, a in enumerate(A1)}
    l0_index = {a: i for i, a in enumerate(A0)}
    leaf_parent_i = np.array([l1_index[a] for a in leaf_parent])
    l1_parent_i = np.array([l0_index[a] for a in l1_parent])
    n_leaf, n_l1, n_l0 = len(A2), len(A1), len(A0)

    # EVERY star with a usable rate feeds its nearest leaf (the harness pooled
    # all calibrated stars this way) - the anchor gate selects LEAVES, it must
    # not discard the sub-anchor stars' evidence, which on deep catalogs is a
    # large fraction of the total
    leaf_of_cat = np.full(n_cat, -1, dtype=np.intp)
    leaf_of_cat[q_ids] = A2
    other = np.where(np.isfinite(rate) & ~qualified)[0]
    other = other[other < len(pos_all)]
    if len(other):
        leaf_of_cat[other] = A2[cKDTree(pos).query(pos_all[other])[1]]

    NG = len(GRID_DM)

    t_unix = np.asarray(frames["frame_time_unix"], dtype=np.float64)
    width = float(np.ceil(np.nanmax(sx)/16.0)*16.0)
    height = float(np.ceil(np.nanmax(sy)/16.0)*16.0)
    cxr = np.clip((sx/(width/CELL_NX)).astype(np.intp), 0, CELL_NX - 1)
    cyr = np.clip((sy/(height/CELL_NY)).astype(np.intp), 0, CELL_NY - 1)

    rec_leaf = leaf_of_cat[cat_id]
    usable = rec_leaf >= 0

    def runBP(logit0v, basev, sigmav, phot_okv):
        """ One full BP sweep of the night with the given per-star weights. """

        def frameLeafLL(fi):
            out = np.zeros((n_leaf, NG))
            m = usable & (sf == fi)
            if not np.any(m):
                return out
            cid = cat_id[m]
            lg = logit0v[cid][:, None]
            det = detected[m].astype(np.float64)[:, None]
            pr = 1.0/(1.0 + np.exp(-(lg - GRID_DM[None, :]/dome_s)))
            pr = np.clip(pr, 1e-6, 1 - 1e-6)
            q = det*pr + (1 - det)*(1 - pr)
            ll = np.log(EVIDENCE_MIX*q + (1 - EVIDENCE_MIX)/2)
            pm = m & (row >= 0) & np.isfinite(inst_resid) & phot_okv[cat_id]
            if np.any(pm):
                sel = pm[m]
                rr = (inst_resid[pm] - basev[cat_id[pm]])[:, None]
                ss = sigmav[cat_id[pm]][:, None]
                ll[sel] += -0.5*np.minimum(((rr - GRID_DM[None, :])/ss)**2, 9.0)
            np.add.at(out, rec_leaf[m], ll)
            return out

        dm_cells = np.full((n_frames, CELL_NY, CELL_NX), np.nan, dtype=np.float32)
        leaf_dm_all = np.full((n_frames, n_leaf), np.nan, dtype=np.float32)
        # Parent topology is fixed for the whole night: presorted group
        # boundaries turn the per-frame scatter-adds into reduceat segment
        # sums (np.add.at is an unbuffered scatter and much slower)
        l1_ord = np.argsort(leaf_parent_i, kind="stable")
        l1_grp, l1_starts = np.unique(leaf_parent_i[l1_ord], return_index=True)
        l0_ord = np.argsort(l1_parent_i, kind="stable")
        l0_grp, l0_starts = np.unique(l1_parent_i[l0_ord], return_index=True)
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

            up2 = _smooth(lls, TAU[2])
            l1_in = np.zeros((n_l1, NG))
            l1_in[l1_grp] = np.add.reduceat(up2[l1_ord], l1_starts, axis=0)
            up1 = _smooth(l1_in, TAU[1])
            l0_in = np.zeros((n_l0, NG))
            l0_in[l0_grp] = np.add.reduceat(up1[l0_ord], l0_starts, axis=0)
            up0 = _smooth(l0_in, TAU[0])
            root = up0.sum(axis=0)

            down0 = _smooth(root[None, :] - up0, TAU[0])
            down1 = _smooth((l0_in + down0)[l1_parent_i] - up1, TAU[1])
            down2 = _smooth((l1_in + down1)[leaf_parent_i] - up2, TAU[2])

            belief = lls + down2
            leaf_dm = np.maximum(0.0, GRID_DM[np.argmax(belief, axis=1)])
            has_ev = np.abs(lls).sum(axis=1) > 0
            leaf_dm_all[fi, has_ev] = leaf_dm[has_ev]

            m = usable & (sf == fi)
            if not np.any(m):
                continue
            lf = rec_leaf[m]
            vals = leaf_dm[lf]
            flat = cyr[m]*CELL_NX + cxr[m]
            cell = np.full(CELL_NY*CELL_NX, np.nan, dtype=np.float64)
            order = np.argsort(flat)
            fs, vs = flat[order], vals[order]
            uniq, starts = np.unique(fs, return_index=True)
            bounds = np.append(starts, len(fs))
            for u, s0, s1 in zip(uniq, bounds[:-1], bounds[1:]):
                cell[u] = np.median(vs[s0:s1])
            dm_cells[fi] = cell.reshape(CELL_NY, CELL_NX)

        return dm_cells, leaf_dm_all

    dm_cells, leaf_dm_all = runBP(logit0, base, sigma, phot_ok)

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

    # A night with zero recalibrated platepars produces a scoring product
    # with no frames and no records (observed on marginal stations after a
    # fully rejected recalibration night) - nothing to estimate from
    if (len(frames.get("frame_names", [])) == 0) \
            or (len(stars.get("star_cat_id", [])) == 0):
        print("Tree estimator: empty scoring product (no scored frames) - "
              "skipping")
        return None

    # Fuse the stills sidecar's instantaneous detections into the detection
    # bits (calstars_row -1 -> -2 where a still saw the star within the FF
    # window). The union channel is what the validated harness consumed; the
    # FF window alone under-reads thin fast cloud.
    n_fused = 0
    try:
        from Utils.StillsSampler import loadStillStarStates, sidecarFileName
        sc_path = os.path.join(night_dir, sidecarFileName(night_name))
        if os.path.isfile(sc_path):
            _, sc = loadStillStarStates(sc_path)
            cat_id_f = np.asarray(stars["star_cat_id"], dtype=np.intp)
            sf_f = np.asarray(stars["star_frame"], dtype=np.intp)
            row_f = np.asarray(stars["calstars_row"], dtype=np.intp).copy()
            t_ff = np.asarray(frames["frame_time_unix"], dtype=np.float64)
            b_ids = sc["bright_cat_id"].astype(np.intp)
            b_det = sc["bright_detected"]
            t_st = sc["t_unix"]
            smap = np.argmin(np.abs(t_ff[None, :] - t_st[:, None]), axis=1)
            ok_s = np.abs(t_ff[smap] - t_st) < 8.0
            det_ff = np.zeros((int(cat_id_f.max()) + 1, len(t_ff)), dtype=bool)
            for si in np.where(ok_s)[0]:
                det_ff[b_ids[b_det[:, si]], smap[si]] = True
            fused = (row_f == -1) & det_ff[cat_id_f, sf_f]
            row_f[fused] = -2
            stars = dict(stars)
            stars["calstars_row"] = row_f
            n_fused = int(fused.sum())
            print("Tree estimator: fused {:d} stills detections".format(
                n_fused))
    except Exception as e:
        print("Tree estimator: stills fusion skipped ({})".format(e))

    # Provenance for the product header: which expectations the tree ran on
    # ("trailing" EMA file / "in_night" same-night stats / "seeded" model-p
    # only) - answerable from any harvested product, not just station logs
    cal_prov = dict(source="seeded", n_rate_stars=0, k=0.0)

    calibration = None
    try:
        from Utils.StarCalibration import calibrationFileName, loadStarCalibration
        cal_path = os.path.join(os.path.expanduser(config.data_dir),
            calibrationFileName(config.stationID))
        if os.path.isfile(cal_path):
            calibration = loadStarCalibration(cal_path)
            if abs(calibration[0].get("catalog_lim_mag", -99)
                    - float(header["catalog_lim_mag"])) >= 0.01:
                calibration = None
            else:
                cal_prov = dict(source="trailing",
                    n_rate_stars=int(np.isfinite(
                        calibration[1]["rate_calstars"]).sum()),
                    k=float(calibration[0].get("k_ema", 0.0)),
                    last_night=str(calibration[0].get("last_night", "")))
    except Exception:
        calibration = None

    # Night-1 (no usable trailing file): measure THIS night's per-star stats,
    # clear-conditioned on the GRID detector's map - the harness's recipe.
    # The grid map is thin-cloud sensitive, so the conditioning excludes
    # cloudy frames properly (a tree-pass-1 conditioner inherits the seed's
    # own thin-blindness; the model-p seed alone is too mushy - measured on
    # the A6 A/B: thin-cloud band harness 0.55, seed 0.00). The overcast
    # guard inside computeNightStarStats keeps unlearnable nights on the
    # saturating seed path.
    if calibration is None:
        try:
            from Utils.StarCalibration import computeNightStarStats
            stats = computeNightStarStats(config, night_dir)
            if stats is not None:
                calibration = (
                    dict(catalog_lim_mag=stats["catalog_lim_mag"],
                         k_ema=stats["k_fit"]),
                    dict(rate_calstars=stats["rate_calstars"],
                         rate_forced=stats["rate_forced"],
                         base_mag=stats["base_mag"],
                         sigma_mag=stats["sigma_mag"]),
                )
                cal_prov = dict(source="in_night",
                    n_rate_stars=int(np.isfinite(
                        stats["rate_calstars"]).sum()),
                    k=float(stats["k_fit"]))
                print("Tree estimator: in-night fine-map-conditioned calibration "
                      "({} stars)".format(cal_prov["n_rate_stars"]))
        except Exception as e:
            print("Tree estimator: in-night calibration failed ({}), "
                  "seeded path".format(e))

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
        calibration=cal_prov,
        stills_fused=n_fused,
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
