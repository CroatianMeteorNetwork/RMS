""" Cross-frame validation of a fitted platepar.

The in-frame overfit check and the forward/reverse round-trip error both operate on the
calibration frame alone - neither measures how well the fit generalizes to the rest of the
night, and in particular to the image corners, where the calibration frame often has few or
no stars. Over a night the sky rotates through the field of view, so the union of all CALSTARS
detections usually covers the corners even when no single frame does.

This module selects a coverage-driven subset of frames (explicitly making sure the corner
cells get filled whenever the dataset has stars there), optionally refits only the pointing
per frame (distortion frozen) so mount drift does not masquerade as distortion error, and
reports spatially binned residuals of catalog-matched detected stars.
"""

from __future__ import print_function, division, absolute_import

import copy

import numpy as np
from scipy.spatial import cKDTree

from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, xyToRaDecPP
from RMS.Astrometry.Conversions import date2JD
from RMS.Formats.FFfile import filenameToDatetime
from RMS.Math import angularSeparation


def _cellCounts(star_data, x_res, y_res, n_grid):
    """ Count stars per cell of an n_grid x n_grid image grid.

    Arguments:
        star_data: [ndarray] CALSTARS rows, columns [y, x, ...].
        x_res, y_res: [int] Image dimensions.
        n_grid: [int] Grid divisions per axis.

    Return:
        counts: [ndarray] (n_grid, n_grid) star counts, indexed [row, col].
    """

    counts = np.zeros((n_grid, n_grid), dtype=int)

    if len(star_data) == 0:
        return counts

    star_data = np.asarray(star_data)
    ys, xs = star_data[:, 0], star_data[:, 1]

    cy = np.clip((ys*n_grid/y_res).astype(int), 0, n_grid - 1)
    cx = np.clip((xs*n_grid/x_res).astype(int), 0, n_grid - 1)
    np.add.at(counts, (cy, cx), 1)

    return counts


def selectValidationFrames(calstars, x_res, y_res, n_grid=8, min_per_cell=150, max_frames=800,
                           min_stars_frame=10):
    """ Select a subset of frames whose union of detected stars covers the image, with an
        explicit guarantee that the corner cells are filled whenever the dataset has stars there.

    Arguments:
        calstars: [dict] {ff_name: star rows} as read from CALSTARS.
        x_res, y_res: [int] Image dimensions.

    Keyword arguments:
        n_grid: [int] Grid divisions per axis for the coverage accounting.
        min_per_cell: [int] Target number of stars per grid cell.
        max_frames: [int] Frame budget for the general coverage pass. The corner top-up pass
            may exceed it - corner coverage takes priority over the budget.
        min_stars_frame: [int] Skip frames with fewer detected stars (clouds, gaps).

    Return:
        selected: [list] Selected FF names, in selection order.
        coverage: [ndarray] (n_grid, n_grid) star counts of the selected union.
    """

    # Precompute per-frame cell counts
    frame_counts = {}
    for ff_name, star_data in calstars.items():
        star_data = np.asarray(star_data)
        if len(star_data) < min_stars_frame:
            continue
        frame_counts[ff_name] = _cellCounts(star_data, x_res, y_res, n_grid)

    coverage = np.zeros((n_grid, n_grid), dtype=int)
    selected = []
    remaining = dict(frame_counts)

    def gain(counts):
        # How many still-needed stars this frame contributes
        return int(np.minimum(counts, np.maximum(min_per_cell - coverage, 0)).sum())

    # General coverage pass: greedily add the frame that fills the most still-needed cells
    while (len(selected) < max_frames) and remaining and (coverage < min_per_cell).any():

        best_ff = max(remaining, key=lambda ff: gain(remaining[ff]))
        if gain(remaining[best_ff]) == 0:
            break

        coverage += remaining.pop(best_ff)
        selected.append(best_ff)

    # Corner top-up pass: the four corner cells get filled whenever ANY frame in the dataset
    # still has stars there, regardless of the frame budget
    corner_cells = [(0, 0), (0, n_grid - 1), (n_grid - 1, 0), (n_grid - 1, n_grid - 1)]
    for cy, cx in corner_cells:

        while (coverage[cy, cx] < min_per_cell) and remaining:

            best_ff = max(remaining, key=lambda ff: remaining[ff][cy, cx])
            if remaining[best_ff][cy, cx] == 0:
                break

            coverage += remaining.pop(best_ff)
            selected.append(best_ff)

    return selected, coverage


def validateFit(platepar, calstars, catalog_stars, frames=None, match_radius=10.0,
                pointing_refit=True, min_match_stars=8, progress_callback=None):
    """ Measure how well the platepar generalizes to other frames of the night.

    For every frame, catalog stars are projected with the platepar at the frame's time and
    matched one-to-one to the detected stars. If pointing_refit is enabled, the pointing
    (not the distortion, not the scale) is refit on the matches first, so mount drift over
    the night is separated from distortion error - the residuals then measure distortion
    generalization alone, and the per-frame pointing shift is reported as drift.

    Arguments:
        platepar: [Platepar] The fitted platepar to validate. Not modified.
        calstars: [dict] {ff_name: star rows}, CALSTARS format [y, x, intensity, ...].
        catalog_stars: [ndarray] (ra, dec, mag) rows, degrees.

    Keyword arguments:
        frames: [list] FF names to use. If None, all frames in calstars.
        match_radius: [float] Match radius in px. Detected stars with a projected catalog
            neighbour within 3x this radius but no match within it count as match failures,
            so large-residual censoring is visible in the match fraction.
        pointing_refit: [bool] Refit pointing per frame before measuring. True by default.
        min_match_stars: [int] Minimum initial matches for a frame to be used.
        progress_callback: [callable] Called with (i_frame, n_frames, ff_name) per frame.

    Return:
        results: [dict]
            star_x, star_y: [ndarray] Image positions of matched stars (all frames pooled).
            star_res: [ndarray] Matched-star residuals (px).
            star_frame: [ndarray] Frame index of each matched star.
            unmatched_x, unmatched_y: [ndarray] Detected stars with a catalog neighbour within
                3x match_radius that failed to match within match_radius.
            frames: [list of dict] Per-frame: ff_name, jd, n_matched, drift_arcmin (None if
                pointing_refit is off or the refit was skipped).
    """

    if frames is None:
        frames = sorted(calstars.keys())

    catalog_stars = np.asarray(catalog_stars, dtype=np.float64)

    star_x, star_y, star_res, star_frame = [], [], [], []
    star_ra, star_dec, star_mag, star_intens = [], [], [], []
    unmatched_x, unmatched_y, unmatched_frame = [], [], []
    unmatched_ra, unmatched_dec = [], []
    frame_reports = []

    for i, ff_name in enumerate(frames):

        if progress_callback is not None:
            progress_callback(i, len(frames), ff_name)

        star_data = np.asarray(calstars[ff_name], dtype=np.float64)
        if len(star_data) < min_match_stars:
            continue

        det_y, det_x = star_data[:, 0], star_data[:, 1]
        det_intens = star_data[:, 2] if star_data.shape[1] > 2 else np.ones(len(det_x))

        ff_dt = filenameToDatetime(ff_name)
        jd = date2JD(ff_dt.year, ff_dt.month, ff_dt.day, ff_dt.hour, ff_dt.minute, ff_dt.second,
                     millisecond=ff_dt.microsecond/1000)

        def projectCatalog(pp):
            cat_x, cat_y = raDecToXYPP(catalog_stars[:, 0], catalog_stars[:, 1], jd, pp)
            inside = (cat_x >= 0) & (cat_x < pp.X_res) & (cat_y >= 0) & (cat_y < pp.Y_res)
            return cat_x, cat_y, inside

        def matchStars(cat_x, cat_y, inside, radius):
            """One-to-one nearest match of detected stars to projected catalog stars."""
            idx_inside = np.where(inside)[0]
            if len(idx_inside) == 0:
                return []
            tree = cKDTree(np.column_stack([cat_x[idx_inside], cat_y[idx_inside]]))
            dist, nn = tree.query(np.column_stack([det_x, det_y]), k=1)
            matches = []
            taken = set()
            # Assign closest pairs first so each catalog star is used once
            for d_i in np.argsort(dist):
                if dist[d_i] > radius:
                    break
                cat_i = idx_inside[nn[d_i]]
                if cat_i in taken:
                    continue
                taken.add(cat_i)
                matches.append((d_i, cat_i, dist[d_i]))
            return matches

        # Initial match with the platepar as fitted
        cat_x, cat_y, inside = projectCatalog(platepar)
        matches = matchStars(cat_x, cat_y, inside, match_radius)

        if len(matches) < min_match_stars:
            continue

        drift_arcmin = None
        centre_radec_ref = None
        centre_radec_frame = None
        pp_frame = platepar

        if pointing_refit:

            # Refit ONLY the pointing on this frame's matches, with the scale fixed - the
            # distortion and scale stay exactly as fitted, so the residuals below measure
            # how the frozen distortion generalizes to this frame
            pp_frame = copy.deepcopy(platepar)
            img_stars = np.array([[det_x[d], det_y[d], det_intens[d]] for d, c, _ in matches])
            cat_matched = catalog_stars[[c for _, c, _ in matches]]

            try:
                pp_frame.fitAstrometry(jd, img_stars, cat_matched, fit_only_pointing=True,
                                       fixed_scale=True)
            except Exception:
                pp_frame = platepar

            if pp_frame is not platepar:

                # Drift = angular shift of the image-centre pointing introduced by the refit
                time_tuple = (ff_dt.year, ff_dt.month, ff_dt.day, ff_dt.hour, ff_dt.minute,
                              ff_dt.second, ff_dt.microsecond/1000)
                _, ra0, dec0, _ = xyToRaDecPP([time_tuple],
                    [platepar.X_res/2.0], [platepar.Y_res/2.0], [1], platepar,
                    extinction_correction=False)
                _, ra1, dec1, _ = xyToRaDecPP([time_tuple],
                    [pp_frame.X_res/2.0], [pp_frame.Y_res/2.0], [1], pp_frame,
                    extinction_correction=False)
                drift_arcmin = 60*np.degrees(angularSeparation(np.radians(ra0[0]),
                    np.radians(dec0[0]), np.radians(ra1[0]), np.radians(dec1[0])))
                centre_radec_ref = (float(ra0[0]), float(dec0[0]))
                centre_radec_frame = (float(ra1[0]), float(dec1[0]))

                # Re-project and re-match with the drift-corrected pointing
                cat_x, cat_y, inside = projectCatalog(pp_frame)
                matches = matchStars(cat_x, cat_y, inside, match_radius)

        ### Sanity filters mirroring the auto-fit pipeline (RMS.Astrometry.StarFilters) ###

        det_fwhm = star_data[:, 4] if star_data.shape[1] > 4 else None

        # Blend rejection: drop pairs whose catalog star has ANOTHER catalog star within
        # 2x the median detected FWHM - the detection is likely a blend of both and its
        # centroid sits between them
        n_blend = 0
        rejected_det = set()
        if len(matches) >= 5:
            good_fwhm = det_fwhm[det_fwhm > 0] if det_fwhm is not None else np.array([])
            blend_radius = 2.0*float(np.median(good_fwhm)) if len(good_fwhm) else 6.0
            idx_inside = np.where(inside)[0]
            cat_tree = cKDTree(np.column_stack([cat_x[idx_inside], cat_y[idx_inside]]))
            kept = []
            for d_i, c_i, dist in matches:
                neighbours = cat_tree.query_ball_point((cat_x[c_i], cat_y[c_i]), blend_radius)

                # A neighbour is a blend threat only if it pulls the blended centroid enough
                # to corrupt the residual: a neighbour with flux ratio f at separation s
                # shifts the photocentre by ~s*f/(1 + f)
                threat = False
                for j in neighbours:
                    n_i = idx_inside[j]
                    if n_i == c_i:
                        continue
                    sep = np.hypot(cat_x[n_i] - cat_x[c_i], cat_y[n_i] - cat_y[c_i])
                    flux_ratio = 10.0**(-0.4*(catalog_stars[n_i, 2] - catalog_stars[c_i, 2]))
                    if sep*flux_ratio/(1.0 + flux_ratio) > 0.5:
                        threat = True
                        break

                if threat:
                    n_blend += 1
                    rejected_det.add(d_i)
                else:
                    kept.append((d_i, c_i, dist))
            matches = kept

        # Photometric consistency: a pairing whose instrumental brightness disagrees with the
        # catalog magnitude is likely a wrong pairing (junk detection or a transient matched
        # to a star by chance). 2.5 sigma clip around the frame's photometric offset.
        n_phot = 0
        if len(matches) >= 8:
            m_inst = np.array([-2.5*np.log10(max(det_intens[d], 1.0)) for d, _, _ in matches])
            m_cat = np.array([catalog_stars[c, 2] for _, c, _ in matches])
            phot_res = (m_cat - m_inst) - np.median(m_cat - m_inst)
            sigma = 1.4826*np.median(np.abs(phot_res))  # robust MAD sigma
            if sigma > 0:
                keep_mask = np.abs(phot_res) <= 2.5*sigma
                n_phot = int((~keep_mask).sum())
                rejected_det.update(m[0] for m, k in zip(matches, keep_mask) if not k)
                matches = [m for m, k in zip(matches, keep_mask) if k]

        ###

        # Record matched residuals (and the pair identities, so the pairs can also be
        # reused for a cross-frame astrometric refit)
        for d_i, c_i, dist in matches:
            star_x.append(det_x[d_i])
            star_y.append(det_y[d_i])
            star_res.append(dist)
            star_frame.append(i)
            star_ra.append(catalog_stars[c_i, 0])
            star_dec.append(catalog_stars[c_i, 1])
            star_mag.append(catalog_stars[c_i, 2])
            star_intens.append(det_intens[d_i])

        # Censoring accounting: detected stars whose nearest projected catalog star is within
        # 3x the radius, UNCLAIMED by any other detection, and yet no match happened within
        # the radius. These are exactly the large residuals a naive average would silently
        # drop. Contention losers (nearest catalog star claimed by a closer detection -
        # doubles, blends) are counted separately and are not failures.
        matched_det = set(d for d, _, _ in matches)
        claimed_cat = set(c for _, c, _ in matches)
        n_contention = 0
        fail_idx = []
        idx_inside = np.where(inside)[0]
        if len(idx_inside):
            tree_cens = cKDTree(np.column_stack([cat_x[idx_inside], cat_y[idx_inside]]))
            dist_all, nn_all = tree_cens.query(np.column_stack([det_x, det_y]), k=1)
            for d_i in range(len(det_x)):

                # Matched pairs and filter-rejected pairs are already accounted for
                if (d_i in matched_det) or (d_i in rejected_det):
                    continue

                # No plausible counterpart at all - not a failure, no story to tell
                if dist_all[d_i] > 3*match_radius:
                    continue

                c_i = idx_inside[nn_all[d_i]]
                if c_i in claimed_cat:
                    # Nearest catalog star was won by a closer detection (double/blend)
                    n_contention += 1
                else:
                    fail_idx.append(d_i)

        # Sky positions of the failures, so the same problem star can be recognized across
        # frames wherever the sky rotation carries it in the image
        if fail_idx:
            time_tuple = (ff_dt.year, ff_dt.month, ff_dt.day, ff_dt.hour, ff_dt.minute,
                          ff_dt.second, ff_dt.microsecond/1000)
            _, f_ra, f_dec, _ = xyToRaDecPP([time_tuple]*len(fail_idx),
                [det_x[k] for k in fail_idx], [det_y[k] for k in fail_idx],
                [1]*len(fail_idx), pp_frame, extinction_correction=False)
            for k, ra, dec in zip(fail_idx, f_ra, f_dec):
                unmatched_x.append(det_x[k])
                unmatched_y.append(det_y[k])
                unmatched_frame.append(i)
                unmatched_ra.append(ra)
                unmatched_dec.append(dec)

        frame_reports.append(dict(ff_name=ff_name, jd=jd, frame_index=i,
                                  n_matched=len(matches),
                                  n_blend_rejected=n_blend, n_photometric_rejected=n_phot,
                                  n_contention=n_contention, drift_arcmin=drift_arcmin,
                                  centre_radec_ref=centre_radec_ref,
                                  centre_radec_frame=centre_radec_frame))

    # Separate failures that are the SAME sky star recurring across frames (a catalog gap,
    # tight double or variable - tells us nothing about the calibration) from the rest.
    # The same star clusters to arcseconds on the sky no matter where the rotation carries
    # it in the image, so cluster on sky unit vectors.
    ux = np.array(unmatched_x); uy = np.array(unmatched_y)
    uframe = np.array(unmatched_frame)
    recurring = np.zeros(len(ux), dtype=bool)
    if len(ux) > 1:
        ra = np.radians(np.array(unmatched_ra)); dec = np.radians(np.array(unmatched_dec))
        vec = np.column_stack([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])

        # Angular clustering radius: the match radius expressed on the sky
        ang_radius = np.radians(2.0*match_radius/platepar.F_scale)
        chord = 2.0*np.sin(ang_radius/2.0)

        sky_tree = cKDTree(vec)
        for k, neighbours in enumerate(sky_tree.query_ball_point(vec, r=chord)):
            if len(set(uframe[j] for j in neighbours)) >= 3:
                recurring[k] = True

    return dict(
        star_x=np.array(star_x), star_y=np.array(star_y), star_res=np.array(star_res),
        star_frame=np.array(star_frame),
        star_ra=np.array(star_ra), star_dec=np.array(star_dec),
        star_mag=np.array(star_mag), star_intens=np.array(star_intens),
        unmatched_x=ux[~recurring], unmatched_y=uy[~recurring],
        unmatched_frame=uframe[~recurring],
        recurring_x=ux[recurring], recurring_y=uy[recurring],
        n_recurring_detections=int(recurring.sum()),
        frames=frame_reports,
    )


def summarizeValidation(results, x_res, y_res, n_annuli=8, corner_radius_frac=None):
    """ Aggregate validation results into radius-binned statistics and headline numbers.

    Arguments:
        results: [dict] Output of validateFit.
        x_res, y_res: [int] Image dimensions.

    Keyword arguments:
        n_annuli: [int] Number of radius bins between the centre and the corner.
        corner_radius_frac: [float] Radii beyond this fraction of the half-diagonal count as
            "corner". If None (default), it is derived from the image aspect: radii beyond the
            farthest edge midpoint, which are geometrically reachable only in the corner wedges
            (~0.87 for 16:9). A looser threshold dilutes the corner statistic with the healthy
            outer ring.

    Return:
        summary: [dict]
            annuli: list of (r_lo_frac, r_hi_frac, n, median_res, rmsd, match_fraction)
            rmsd_global, rmsd_corner, n_corner, corner_match_fraction, max_drift_arcmin
    """

    cx, cy = x_res/2.0, y_res/2.0
    r_max = np.hypot(cx, cy)

    # Default corner threshold: beyond the farthest edge midpoint only the corner wedges remain
    if corner_radius_frac is None:
        corner_radius_frac = max(cx, cy)/r_max

    def radii(x, y):
        return np.hypot(np.asarray(x) - cx, np.asarray(y) - cy)/r_max

    res = results["star_res"]
    r_matched = radii(results["star_x"], results["star_y"]) if len(res) else np.array([])
    r_unmatched = radii(results["unmatched_x"], results["unmatched_y"]) \
        if len(results["unmatched_x"]) else np.array([])

    annuli = []
    edges = np.linspace(0, 1, n_annuli + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_m = (r_matched >= lo) & (r_matched < hi)
        n_m = int(in_m.sum())
        n_u = int(((r_unmatched >= lo) & (r_unmatched < hi)).sum())
        med = float(np.median(res[in_m])) if n_m else None
        rmsd = float(np.sqrt(np.mean(res[in_m]**2))) if n_m else None
        frac = n_m/(n_m + n_u) if (n_m + n_u) else None
        annuli.append((float(lo), float(hi), n_m, med, rmsd, frac))

    corner = r_matched >= corner_radius_frac
    corner_u = int((r_unmatched >= corner_radius_frac).sum()) if len(r_unmatched) else 0
    n_corner = int(corner.sum())

    drifts = [f["drift_arcmin"] for f in results["frames"] if f["drift_arcmin"] is not None]

    return dict(
        annuli=annuli,
        corner_radius_frac=float(corner_radius_frac),
        rmsd_global=float(np.sqrt(np.mean(res**2))) if len(res) else None,
        median_global=float(np.median(res)) if len(res) else None,
        rmsd_corner=float(np.sqrt(np.mean(res[corner]**2))) if n_corner else None,
        median_corner=float(np.median(res[corner])) if n_corner else None,
        n_corner=n_corner,
        corner_match_fraction=n_corner/(n_corner + corner_u) if (n_corner + corner_u) else None,
        max_drift_arcmin=float(np.max(drifts)) if drifts else None,
    )


def buildRefitGroups(results, platepar, max_per_cell=15, n_grid=8):
    """ Turn the validated cross-frame matches into per-image groups for
        Platepar.fitAstrometryMultiImage.

    Unlike an epoch-transfer approach, the multi-image fit projects the catalog at each
    image's own Julian date, so the pairs are used exactly as observed - no coordinate
    round trips, no drift approximation baked into the data.

    The set is spatially balanced with a per-cell cap so the star-rich image centre does
    not dominate the fit; corner cells rarely reach the cap, so their pairs are all kept.

    Arguments:
        results: [dict] Output of validateFit.
        platepar: [Platepar] Provides the image dimensions for the spatial cap.

    Keyword arguments:
        max_per_cell: [int] Cap on pairs per image grid cell.
        n_grid: [int] Grid divisions per axis for the spatial cap.

    Return:
        image_groups: [list] (image_id, jd, img_stars, catalog_stars) tuples, as expected by
            fitAstrometryMultiImage.
    """

    by_index = {f["frame_index"]: f for f in results["frames"]}

    n = len(results["star_x"])
    if n == 0:
        return []

    # Spatial cap: deterministic shuffle, keep up to max_per_cell per grid cell
    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    cell_counts = {}
    keep = []
    for k in order:
        cx = int(np.clip(results["star_x"][k]*n_grid/platepar.X_res, 0, n_grid - 1))
        cy = int(np.clip(results["star_y"][k]*n_grid/platepar.Y_res, 0, n_grid - 1))
        if cell_counts.get((cy, cx), 0) < max_per_cell:
            cell_counts[(cy, cx)] = cell_counts.get((cy, cx), 0) + 1
            keep.append(k)

    # Group the kept pairs by source frame
    groups = {}
    for k in keep:
        i = int(results["star_frame"][k])
        report = by_index.get(i)
        if report is None:
            continue
        g = groups.setdefault(i, dict(ff_name=report["ff_name"], jd=report["jd"],
                                      img=[], cat=[]))
        g["img"].append([results["star_x"][k], results["star_y"][k],
                         results["star_intens"][k]])
        g["cat"].append([results["star_ra"][k], results["star_dec"][k],
                         results["star_mag"][k]])

    image_groups = []
    for i in sorted(groups):
        g = groups[i]
        if len(g["img"]) < 3:
            continue
        image_groups.append((g["ff_name"], g["jd"], np.array(g["img"]), np.array(g["cat"])))

    return image_groups
