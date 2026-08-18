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

from RMS.Logger import getLogger

# On-station the capture process owns the handlers and these messages land
# in the nightly log with timestamps (bare print() lines do not - an 8-hour
# fit once ran invisibly because of that); CLI mains add a stdout handler
log = getLogger("rmslogger")


GRID_DM = np.arange(-0.3, 4.501, 0.03)     # extinction grid (mag)
TAU = (3.0, 0.6, 0.3)                      # scale-transition widths, root first
                                           # (root wide = decoupled)
EPS_MIX = 0.1                              # heavy-tail flat mixture in transitions
EVIDENCE_MIX = 0.98                        # per-star likelihood robustness floor
RATE_MIN_LEAF = 0.15                       # measured clear rate to carry evidence
RATE_CLIP = (0.03, 0.97)
WINDOW_FRAMES = 1                          # +/- frames pooled per BP solve

# OPEN ITEM - deep uniform layers saturate to the ceiling instead of reading
# the bright-star bound. When a solid semi-transparent layer covers the FOV
# and only the brightest star is barely visible, the correct inference is
# dm ~ that star's bound for the WHOLE field - the effective cell is the
# whole FOV. The uniform-layer suite (built on a real night's geometry;
# see tests and the investigation notes) measures: uniform 1.0/2.0 read
# correctly with the trailing-history seed norm, but uniform 3.0 and the
# brightest-only case read the 4.5 ceiling ("at least" - conservative, so
# verdicts err opaque). Three global-anchor designs were measured and
# rejected: a root prior (the tau-3 down smoothing smears a 0.25-mag anchor
# ~12x - no effect), the same scaled by consistency^2 (no effect), and a
# belief-level anchor at the pooled pure argmax (fixed deep layers but
# dragged the half-disc scenario from 1.80/0.00 to 0.42/0.30 - per-star
# leaves hold too few records to out-argue even a capped bump). The right
# construction is a structure-aware uniform component (consistent-subset
# refit), not a global bump; until built, the ceiling is the honest answer.
S_DEFAULT = 0.4                            # logistic width fallback (mag)

# Kriging fusion of the leaf posteriors: ordinary kriging over each frame's
# (mean, sigma) in image coordinates, GLS constant mean. Neighboring leaves
# see the same sky, but their raw posteriors carry independent per-star
# noise and calibration error - fusing them with uncertainty weights is
# what turned the leaf quilt into coherent fields (and halved the
# rotating-with-the-sky identity artifact) in the offline validation. The
# solve is subsampled: at KRIG_ELL the FOV holds ~15 independent patches,
# so ~350 lowest-sigma leaves saturate the resolvable structure and keep
# the per-frame Cholesky Pi-priced; prediction covers every evidence leaf.
KRIG_ELL = 250.0        # px - exponential kernel length scale
KRIG_MAX_OBS = 350      # leaves entering the solve (lowest sigma first)
KRIG_MIN_OBS = 40       # below this the frame keeps its raw posteriors
KRIG_NUGGET_FLOOR = 0.05  # mag - minimum per-leaf observation noise

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

    # Records grouped by frame, once. Every per-frame loop below used to select its
    # frame with "sf == fi" - a scan of the whole night's record array, re-run for
    # every frame, allocating a full-length boolean each time. Measured on a deep
    # catalog synthetic night at 3.15M records: 59.7 s -> 49.6 s.
    #
    # This does NOT lower the resident peak, which was measured as working set
    # proportional to the record count (~370 MB fixed + ~130 MB per million records,
    # 1.7 GB at 8.4M) rather than allocation churn - the per-frame temporaries are
    # megabytes each, so numpy mmaps them and the kernel takes them back on free.
    # Bounding the concurrent peak across co-located stations is a separate problem.
    #
    # The sort is stable, so records keep their original relative order inside a frame
    # and every downstream accumulation (np.add.at, reduceat) sums in the same
    # sequence as before - the outputs are bit-identical, not merely equivalent.
    rec_order = np.argsort(sf, kind="stable")
    rec_frame_starts = np.searchsorted(sf[rec_order], np.arange(n_frames + 1))

    def recordsInFrame(fi):
        """ Indices of the records belonging to frame fi, in ascending record order.

        Arguments:
            fi: [int] Frame index.

        Return:
            [ndarray] Record indices, empty if the frame carries none.
        """

        return rec_order[rec_frame_starts[fi]:rec_frame_starts[fi + 1]]

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

    # Normalize the model seed to the site's clear level - from the TRAILING
    # ratio history (prior nights, cloud-immune), NOT from tonight's own
    # frames: tonight's percentile absorbs any uniform semi-transparent layer
    # into the baseline (measured on the synthetic suite: a uniform 1.0 mag
    # layer read 0.30, a 2.0 layer read 1.17). The model-miscalibration this
    # norm exists to fix (2.5x fresh-refit over-prediction, MW-transit
    # under-prediction) is a property of the MODEL, not of tonight - so
    # prior nights measure it correctly and tonight's weather cannot leak in.
    # True night 1 (no history) falls back to tonight's percentile with the
    # documented limitation: a uniform layer on the station's very first
    # night partially reads as the baseline.
    seed_norm = None
    try:
        history_path = os.path.join(os.path.expanduser(config.data_dir),
            "{:s}_flux_lm_history.json".format(str(config.stationID)))
        with open(history_path) as f:
            _hist = json.load(f)
        _ver = str(header.get("dome_fit_date"))
        _drs = [v["dratio"] for k, v in sorted(_hist.items())[-30:]
                if isinstance(v, dict) and ("dratio" in v)
                and (str(v.get("dmodel", "")) == _ver)]
        if len(_drs) >= 5:
            seed_norm = float(np.clip(np.percentile(_drs, 80), 0.5, 3.0))
    except Exception:
        seed_norm = None

    if seed_norm is None:
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

    # PROTOTYPE - uniform-component channel: uncapped logits (the RATE_CLIP /
    # seed caps amputate bright-star bits precisely where a deep uniform layer
    # is measured; the pooled whole-FOV estimator needs their full leverage)
    rate_unc = np.where(np.isfinite(rate), rate, np.nan)
    if calibration is None:
        # the seed path capped p_model at 0.97; recover the uncapped medians
        with np.errstate(invalid="ignore"):
            p_model_unc = np.where(seen_n >= 10, p_sum/np.maximum(seen_n, 1), np.nan)
        p_model_unc = np.clip(p_model_unc*(seed_norm if seed_norm else 1.0),
            0.0, 0.9995)
        rate_unc = np.where(np.isfinite(rate_unc), rate_unc, p_model_unc)
    p_unc = np.clip(rate_unc, 1e-3, 0.9995)
    logit0_unc = np.log(p_unc/(1.0 - p_unc))
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
                idx_f = recordsInFrame(fi)
                joined = row[idx_f] >= 0
                if not np.any(joined):
                    continue
                idx_f = idx_f[joined]
                rows_here = row[idx_f]
                valid = rows_here < len(data)
                inten = data[rows_here[valid], 2]
                good = inten > 0
                idx = idx_f[valid][good]
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
                idx_f = recordsInFrame(fi)
                idx_f = idx_f[r_ok[idx_f]]
                if not len(idx_f):
                    continue
                _, alt_ = raDec2AltAz(
                    catalog_stars[cat_id[idx_f], 0], catalog_stars[cat_id[idx_f], 1],
                    float(jd_frame[fi]), lat, lon)
                h_ = np.maximum(np.asarray(alt_), 2.0)
                X_ = 1.0/(np.sin(np.radians(h_))
                          + 0.50572*(h_ + 6.07995)**-1.6364)
                inst_resid[idx_f] = inst_resid[idx_f] - k_ext*(X_ - 1.0)


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

    # Per-frame anchor pixel positions for the kriging distance matrix,
    # filled from the anchor stars' own records (+/-2 frame tolerance). The
    # sky rotates hundreds of px over a night - a static position degrades
    # the geometry (measured as sliver leaves and a corrupted kriging
    # kernel in the offline validation). float32: 2 x n_frames x n_leaf.
    anchor_of_cat = np.full(n_cat, -1, dtype=np.intp)
    anchor_of_cat[q_ids] = A2
    rec_anchor = anchor_of_cat[cat_id]
    ma = rec_anchor >= 0
    kx = np.full((n_frames, n_leaf), np.nan, dtype=np.float32)
    ky = np.full((n_frames, n_leaf), np.nan, dtype=np.float32)
    kx[sf[ma], rec_anchor[ma]] = sx[ma]
    ky[sf[ma], rec_anchor[ma]] = sy[ma]
    for arr in (kx, ky):
        for shift in (1, -1, 2, -2):
            src_ = np.roll(arr, shift, axis=0)
            if shift > 0:
                src_[:shift] = np.nan
            else:
                src_[shift:] = np.nan
            gap = np.isnan(arr)
            arr[gap] = src_[gap]

    # Usable records grouped by frame, and the frame-independent half of the
    # photometric-channel mask. Both were rebuilt inside frameLeafLL on every frame,
    # which cost several full-length temporaries per call for a result that never
    # changed (see recordsInFrame above for why that matters here).
    use_order = rec_order[usable[rec_order]]
    use_frame_starts = np.searchsorted(sf[use_order], np.arange(n_frames + 1))
    phot_rec_static = (row >= 0) & np.isfinite(inst_resid)

    def runBP(logit0v, basev, sigmav, phot_okv):
        """ One full BP sweep of the night with the given per-star weights. """

        # phot_okv is per-call, the rest of the mask is not
        phot_rec = phot_rec_static & phot_okv[cat_id]

        def frameLeafLL(fi):
            out = np.zeros((n_leaf, NG))
            g_unif = np.zeros(NG)
            idx = use_order[use_frame_starts[fi]:use_frame_starts[fi + 1]]
            if not len(idx):
                return out, g_unif
            cid = cat_id[idx]
            lg = logit0v[cid][:, None]
            det = detected[idx].astype(np.float64)[:, None]
            pr = 1.0/(1.0 + np.exp(-(lg - GRID_DM[None, :]/dome_s)))
            pr = np.clip(pr, 1e-6, 1 - 1e-6)
            q = det*pr + (1 - det)*(1 - pr)
            ll = np.log(EVIDENCE_MIX*q + (1 - EVIDENCE_MIX)/2)
            sel = phot_rec[idx]

            # Uniform-component channel: pure pooled likelihood, uncapped
            # logits, flux with a 4-sigma robustness clamp
            lg_u = logit0_unc[cid][:, None]
            pr_u = np.clip(1.0/(1.0 + np.exp(-(lg_u - GRID_DM[None, :]/dome_s))),
                1e-9, 1 - 1e-9)
            q_u = det*pr_u + (1 - det)*(1 - pr_u)
            g_unif = np.log(q_u).sum(axis=0)

            if np.any(sel):
                idx_p = idx[sel]
                cid_p = cat_id[idx_p]
                rr = (inst_resid[idx_p] - basev[cid_p])[:, None]
                ss = sigmav[cid_p][:, None]
                ll[sel] += -0.5*np.minimum(((rr - GRID_DM[None, :])/ss)**2, 9.0)
                g_unif = g_unif + (-0.5*np.minimum(
                    ((rr - GRID_DM[None, :])/ss)**2, 16.0)).sum(axis=0)
            np.add.at(out, rec_leaf[idx], ll)
            return out, g_unif

        dm_cells = np.full((n_frames, CELL_NY, CELL_NX), np.nan, dtype=np.float32)
        leaf_dm_all = np.full((n_frames, n_leaf), np.nan, dtype=np.float32)
        dm_u_all = np.full(n_frames, np.nan, dtype=np.float32)
        leaf_sd_all = np.full((n_frames, n_leaf), np.nan, dtype=np.float32)
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
            g_unif = None
            for k in range(fi - WINDOW_FRAMES, fi + WINDOW_FRAMES + 1):
                if not (0 <= k < n_frames):
                    continue
                if k not in cache:
                    cache[k] = frameLeafLL(k)
                c_ll, c_g = cache[k]
                lls = c_ll if lls is None else lls + c_ll
                g_unif = c_g if g_unif is None else g_unif + c_g
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

            # Whole-FOV pooled uniform level, kept as a per-frame
            # diagnostic (the shrinkage that used it was superseded by the
            # kriging fusion below)
            if g_unif is not None:
                pu = np.exp(g_unif - g_unif.max())
                pu /= max(pu.sum(), 1e-30)
                dm_u_all[fi] = max(0.0, float(pu @ GRID_DM))

            # Posterior mean + sigma per leaf (the belief carries a full
            # curve; the argmax discarded its shape - on censored plateaus
            # that was the arbitrary-edge/ceiling behavior)
            pb = np.exp(belief - belief.max(axis=1, keepdims=True))
            pb /= np.maximum(pb.sum(axis=1, keepdims=True), 1e-30)
            mu = pb @ GRID_DM
            sd = np.sqrt(np.maximum((pb @ (GRID_DM**2)) - mu**2, 1e-6))
            has_ev = np.abs(lls).sum(axis=1) > 0

            # Kriging fusion (see the constants block). Observations: the
            # lowest-sigma evidence leaves with a position this frame;
            # prediction: every evidence leaf with a position. Leaves
            # without a position this frame keep their raw posterior.
            okk = has_ev & np.isfinite(mu) & np.isfinite(sd) \
                & np.isfinite(kx[fi]) & np.isfinite(ky[fi])
            if okk.sum() >= KRIG_MIN_OBS:
                cand = np.where(okk)[0]
                if len(cand) > KRIG_MAX_OBS:
                    # Spatially stratified subsample: lowest-sigma-first
                    # selection starved whole regions (under a half-field
                    # layer the occluded side's censored posteriors carry
                    # the largest sigmas - the suite read its opaque half
                    # as CLEAR). Cap picks per image tile instead: coverage
                    # first, precision within a neighborhood.
                    tx = np.clip((kx[fi][cand]/max(width, 1.0)*16).astype(
                        np.intp), 0, 15)
                    ty = np.clip((ky[fi][cand]/max(height, 1.0)*9).astype(
                        np.intp), 0, 8)
                    tile = ty*16 + tx
                    order = np.lexsort((sd[cand], tile))
                    t_sorted = tile[order]
                    new_tile = np.r_[True, t_sorted[1:] != t_sorted[:-1]]
                    grp = np.cumsum(new_tile) - 1
                    starts = np.where(new_tile)[0]
                    rank = np.arange(len(order)) - starts[grp]
                    per_tile = max(2, KRIG_MAX_OBS//max(
                        int(new_tile.sum()), 1))
                    cand = cand[order[rank < per_tile]]
                ox_, oy_ = kx[fi][cand], ky[fi][cand]
                om, on = mu[cand], np.maximum(sd[cand], KRIG_NUGGET_FLOOR)
                dxk = ox_[:, None] - ox_[None, :]
                dyk = oy_[:, None] - oy_[None, :]
                Kb = np.exp(-np.hypot(dxk, dyk)/KRIG_ELL)
                sf2 = max(float(np.var(om) - np.mean(on**2)), 0.03**2)
                Kn = sf2*Kb + np.diag(on**2)
                try:
                    from scipy.linalg import cho_factor, cho_solve
                    cf = cho_factor(Kn, lower=True)
                    one = np.ones(len(om))
                    beta = float(one @ cho_solve(cf, om)) \
                        / max(float(one @ cho_solve(cf, one)), 1e-9)
                    alpha = cho_solve(cf, om - beta)
                    pxk = kx[fi][okk], ky[fi][okk]
                    ddx = pxk[0][:, None] - ox_[None, :]
                    ddy = pxk[1][:, None] - oy_[None, :]
                    Ks = sf2*np.exp(-np.hypot(ddx, ddy)/KRIG_ELL)
                    mu = mu.copy()
                    mu[okk] = beta + Ks @ alpha
                except Exception:
                    pass

            leaf_dm = np.maximum(0.0, mu)
            leaf_dm_all[fi, has_ev] = leaf_dm[has_ev]
            leaf_sd_all[fi, has_ev] = sd[has_ev]

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

        computeTreeSeries.last_dm_u = dm_u_all
        computeTreeSeries.last_leaf_sd = leaf_sd_all
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
        log.info("Tree estimator: empty scoring product (no scored frames) - "
              "skipping")
        return None

    # Fuse the stills sidecar's instantaneous detections into the detection
    # bits (calstars_row -1 -> -2 where a still saw the star within the FF
    # window). The union channel is what the validated harness consumed; the
    # FF window alone under-reads thin fast cloud.
    n_fused = 0
    try:
        from Utils.StillsSampler import fuseSidecarDetections
        n_before = int((np.asarray(stars["calstars_row"]) == -2).sum())
        stars = fuseSidecarDetections(night_dir, frames, stars)
        n_fused = int((np.asarray(stars["calstars_row"]) == -2).sum()) \
            - n_before
        if n_fused:
            log.info("Tree estimator: fused {:d} stills detections".format(
                n_fused))
    except Exception as e:
        log.info("Tree estimator: stills fusion skipped ({})".format(e))

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
                log.info("Tree estimator: in-night fine-map-conditioned calibration "
                      "({} stars)".format(cal_prov["n_rate_stars"]))
        except Exception as e:
            log.info("Tree estimator: in-night calibration failed ({}), "
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

    log.info("Tree transparency map: {:s}".format(os.path.basename(path)))

    return path
