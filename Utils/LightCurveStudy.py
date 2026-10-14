""" Studies of the per-block star light curves written by Utils.LightCurveReplay (RMS.StarLightCurves).

'photometry' (clear stretches): the slope of instrumental against catalog magnitude, and each
star's frame-to-frame scatter against a noise model built from measured quantities only:
    sigma^2 = sigma_sky^2 + F/g + (s*F)^2
sigma_sky is the scatter of the same box statistic on the sky probes near the star (it carries
read noise, quantisation, codec loss and the pixel-to-pixel correlation of the demosaic, none of
which a per-pixel sigma shows), F/g the star's own shot noise and s*F a noise proportional to the
signal (scintillation); g and s are fitted over all stars.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import json
import os
import sys

import numpy as np

from RMS.StarLightCurves import APERTURE_AREA


def blockStats(path, probe_radius=150.0):
    """ Reduce one block file to per-star statistics.

    Background estimators: 'fr' = the frame's own ring median, 'ave' = the ring median on the
    block's average image (constant over the block). The bias of each estimator (aperture mean
    minus ring median: +0.36 and +0.10 linear codes on US05E1's clear sky, from the skew of the
    linear-domain sky) is measured on the probes within probe_radius px and subtracted.

    Return:
        [dict] of per-star arrays, and the header.
    """

    d = np.load(path)
    h = json.loads(str(d['header']))
    flux, bg, bg_ave = d['flux'], d['bg'], d['bg_ave']
    n = flux.shape[0]
    x = 0.5*(d['x_first'] + d['x_last'])
    y = 0.5*(d['y_first'] + d['y_last'])

    pf, pb = d['probe_flux'], d['probe_bg']
    p_net_fr = pf - APERTURE_AREA*pb
    p_mean_fr = p_net_fr.mean(axis=0)
    p_mean_ave = pf.mean(axis=0) - APERTURE_AREA*d['probe_bg_ave']
    p_sig_fr = p_net_fr.std(axis=0, ddof=1)
    p_sig_ave = pf.std(axis=0, ddof=1)

    # Probes near each star (median of their statistics)
    dist = np.hypot(x[:, None] - d['probe_x'][None, :], y[:, None] - d['probe_y'][None, :])
    near = dist <= probe_radius
    def local(v):
        m = np.where(near, v[None, :], np.nan)
        with np.errstate(all='ignore'):
            return np.nanmedian(m, axis=1)

    net_fr = flux - APERTURE_AREA*bg
    net_ave = flux - APERTURE_AREA*bg_ave[None, :]

    f = flux - flux.mean(axis=0)
    with np.errstate(all='ignore'):
        lag1 = (f[1:]*f[:-1]).mean(axis=0)/(f*f).mean(axis=0)

    return {'oid': d['oid'], 'mag': d['mag'], 'ra': d['ra'], 'dec': d['dec'], 'x': x, 'y': y,
            't': float(np.mean(d['t'])), 'n_sat': d['n_sat'], 'bg_ave': bg_ave,
            'F_fr': net_fr.mean(axis=0) - local(p_mean_fr),
            'F_ave': net_ave.mean(axis=0) - local(p_mean_ave),
            'sig_fr': net_fr.std(axis=0, ddof=1), 'sig_ave': net_ave.std(axis=0, ddof=1),
            'sky_fr': local(p_sig_fr), 'sky_ave': local(p_sig_ave), 'lag1': lag1,
            'n_probe': near.sum(axis=1), 'probe_bias_fr': local(p_mean_fr),
            'probe_bias_ave': local(p_mean_ave)}, h


def robustFit(A, b, clip=3.0, n_iter=6):
    """ Least squares with iterative sigma clipping. Return (coefficients, kept mask, rms). """

    keep = np.isfinite(b) & np.all(np.isfinite(A), axis=1)
    for _ in range(n_iter):
        c = np.linalg.lstsq(A[keep], b[keep], rcond=None)[0]
        r = b - A.dot(c)
        s = 1.4826*np.median(np.abs(r[keep] - np.median(r[keep])))
        new = keep & (np.abs(r) <= clip*s)
        if np.array_equal(new, keep):
            break
        keep = new
    r = b - A.dot(c)

    return c, keep, float(np.sqrt(np.mean(r[keep]**2)))


def photometry(args):

    from RMS.Astrometry.Conversions import raDec2AltAz
    from RMS.Formats.Platepar import Platepar
    from RMS.Formats import StarCatalog
    import RMS.ConfigReader as cr

    paths = sorted(glob.glob(os.path.join(args.lc_dir, 'LC_*.npz')))
    if args.max_blocks:
        paths = paths[:args.max_blocks]

    blocks = []
    for p in paths:
        s, h = blockStats(p)
        blocks.append(s)
    print("{:d} blocks, {:s} .. {:s}".format(len(blocks), os.path.basename(paths[0]),
                                            os.path.basename(paths[-1])))

    config = cr.parse(args.config)
    pp = Platepar()
    pp.read(args.platepar, use_flat=False)
    cat, _, _, ex = StarCatalog.readStarCatalog(config.star_catalog_path, config.star_catalog_file,
        lim_mag=h['lim_mag'], mag_band_ratios=config.star_catalog_band_ratios,
        additional_fields=['oid', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'Simbad_OType'])
    row = {int(o): i for i, o in enumerate(ex['oid'])}

    # Per star over the stretch: medians of the per-block statistics
    oids = np.unique(np.concatenate([b['oid'] for b in blocks]))
    pos = {int(o): i for i, o in enumerate(oids)}
    keys = ('F_fr', 'F_ave', 'sig_fr', 'sig_ave', 'sky_fr', 'sky_ave', 'lag1', 'x', 'y', 'bg_ave')
    acc = {k: [[] for _ in oids] for k in keys}
    nsat = np.zeros(len(oids), dtype=np.int64)
    alt = [[] for _ in oids]
    for b in blocks:
        jd = b['t']/86400.0 + 2440587.5
        _, a = raDec2AltAz(b['ra'], b['dec'], jd, pp.lat, pp.lon)
        for j, o in enumerate(b['oid']):
            i = pos[int(o)]
            for k in keys:
                acc[k][i].append(b[k][j])
            alt[i].append(a[j])
            nsat[i] += int(b['n_sat'][j])
    med = {k: np.array([np.nanmedian(v) if len(v) else np.nan for v in acc[k]]) for k in keys}
    nblk = np.array([len(v) for v in acc['F_ave']])
    alt = np.array([np.median(v) for v in alt])

    ci = np.array([row[int(o)] for o in oids])
    mag = cat[ci, 2]
    bp_rp = ex['phot_bp_mean_mag'][ci] - ex['phot_rp_mean_mag'][ci]
    otype = np.array([t.decode() if isinstance(t, bytes) else str(t) for t in ex['Simbad_OType'][ci]])
    variable = np.array([('V*' in t) or ('Var' in t) or ('Cepheid' in t) or ('RR' in t)
                         or ('EB' in t) or ('Mira' in t) for t in otype])

    # Crowding: another measured star within 10 px brighter than mag + 2.5
    from scipy.spatial import cKDTree
    xy = np.column_stack([med['x'], med['y']])
    crowded = np.zeros(len(oids), dtype=bool)
    for i, nb in enumerate(cKDTree(xy).query_ball_point(xy, 10.0)):
        crowded[i] = any((j != i) and (mag[j] < mag[i] + 2.5) for j in nb)

    airmass = 1.0/np.sin(np.radians(np.clip(alt, 5, 90)))
    r2 = ((med['x'] - pp.X_res/2.0)**2 + (med['y'] - pp.Y_res/2.0)**2)/(pp.Y_res/2.0)**2

    clean = (nsat == 0) & ~crowded & ~variable & (nblk >= max(3, len(blocks)//2))
    print("stars: {:d} measured, {:d} clean (unsaturated {:d}, uncrowded {:d}, non-variable {:d})"
          .format(len(oids), int(clean.sum()), int((nsat == 0).sum()), int((~crowded).sum()),
                  int((~variable).sum())))

    # 1) Linearity: instrumental vs catalog magnitude. The GMN catalog's Gaia BP/RP are unreliable
    # brighter than G ~3.5 (saturation), which spoils both the band magnitude and the colour
    # there, hence the lower bound
    for est in ('F_ave', 'F_fr'):
        F = med[est]
        ok = clean & (F > 0) & (mag >= args.slope_mag_min) & (mag <= args.slope_mag_max)
        m_inst = -2.5*np.log10(np.where(ok, F, np.nan))
        print("\n[{:s}] slope of instrumental vs catalog magnitude, {:.1f} <= mag <= {:.1f}:".format(
            est, args.slope_mag_min, args.slope_mag_max))
        for name, cols in (('mag only', [mag]),
                           ('+ airmass + vignetting (r^2, r^4)', [mag, airmass, r2, r2**2]),
                           ('+ airmass + vignetting + BP-RP', [mag, airmass, r2, r2**2, bp_rp])):
            A = np.column_stack(cols + [np.ones(len(mag))])
            sel = ok & np.all(np.isfinite(A), axis=1)
            c, keep, rms = robustFit(A[sel], m_inst[sel])
            print("  {:36s} slope {:.4f}  n {:d}/{:d}  rms {:.3f} mag  terms {}".format(name, c[0],
                  int(keep.sum()), int(sel.sum()), rms, np.round(c[1:-1], 4).tolist()))

        # Slope 1 with an additive flux offset: F = 10^(-0.4 (mag + k X + v r^2 + c (BP-RP) + z)) + delta.
        # A background bias delta makes the plain log fit steeper than 1 at the faint end
        from scipy.optimize import least_squares
        sel = ok & np.isfinite(bp_rp)
        def resid(p, frac=False):
            model = 10**(-0.4*(mag[sel] + p[0]*airmass[sel] + p[1]*r2[sel] + p[2]*bp_rp[sel] + p[3])) + p[4]
            return (F[sel] - model)/np.maximum(med[est.replace('F', 'sig')][sel]/np.sqrt(max(nblk.min(), 1)*256.0), 1e-3)
        sol = least_squares(resid, [0.25, 0.3, 0.0, -2.5*np.log10(np.median(F[sel])) - np.median(mag[sel]), 0.0],
                            loss='soft_l1')
        print("  {:36s} delta {:+.3f} linear codes (k {:.3f}, v {:.3f}, colour {:.3f})".format(
              'slope fixed at 1, additive offset:', sol.x[4], sol.x[0], sol.x[1], sol.x[2]))

    # 2) Scatter against the noise model. The excess over the local sky noise is split per star
    # into its shot-noise equivalent (excess^2/F, constant for pure shot noise) and its fraction of
    # the signal (excess/F, constant for a multiplicative noise such as scintillation)
    for est, sig, sky in (('F_ave', 'sig_ave', 'sky_ave'), ('F_fr', 'sig_fr', 'sky_fr')):
        F, S, K = med[est], med[sig], med[sky]
        ok = clean & (F > 0) & np.isfinite(S) & np.isfinite(K)
        ex2 = S**2 - K**2
        print("\n[{:s}] per-frame scatter sigma vs the probes' sky noise at the star:".format(est))
        print("   mag    n       F    sigma     sky  sigma/sky  excess^2/F  excess/F   lag1")
        for lo in np.arange(np.floor(np.nanmin(mag[ok])*2)/2, h['lim_mag'], 0.5):
            k = ok & (mag >= lo) & (mag < lo + 0.5)
            if k.sum() < 3:
                continue
            e2 = np.median(ex2[k])
            print("  {:4.1f} {:5d} {:7.2f} {:8.2f} {:7.2f} {:9.3f} {:11.3f} {:9.3f} {:7.3f}".format(lo,
                  int(k.sum()), np.median(F[k]), np.median(S[k]), np.median(K[k]),
                  np.median(S[k]/K[k]), e2/np.median(F[k]), np.sqrt(max(e2, 0))/np.median(F[k]),
                  np.median(med['lag1'][k])))

        # Scintillation scales with airmass (~X^1.5..1.75); a detector or codec noise does not
        b = ok & (F > 10.0*K)
        frac = np.sqrt(np.maximum(ex2[b], 0))/F[b]
        good = frac > 0
        if np.count_nonzero(good) >= 10:
            A = np.column_stack([np.log(airmass[b][good]), np.ones(np.count_nonzero(good))])
            c, keep, rms = robustFit(A, np.log(frac[good]))
            print("  bright stars (F > 10 sky): excess/F = {:.3f} * X^{:.2f} (n {:d}, airmass {:.2f}..{:.2f}, "
                  "rms {:.2f} in ln)".format(np.exp(c[1]), c[0], int(keep.sum()), airmass[b].min(),
                                             airmass[b].max(), rms))

    # The clear-sky model (slope fixed at 1, F_fr): saved for the transmission of other stretches
    F = med['F_fr']
    ok = clean & (F > 0) & (mag >= args.slope_mag_min) & (mag <= args.slope_mag_max) & np.isfinite(bp_rp)
    A = np.column_stack([airmass, r2, r2**2, bp_rp, np.ones(len(mag))])
    c, keep, rms = robustFit(A[ok], -2.5*np.log10(F[ok]) - mag[ok])
    model = {'k': c[0], 'v2': c[1], 'v4': c[2], 'colour': c[3], 'zp': c[4], 'rms': rms,
             'n': int(keep.sum()), 'mag_range': [args.slope_mag_min, args.slope_mag_max],
             'lc_dir': args.lc_dir, 'X_res': pp.X_res, 'Y_res': pp.Y_res}
    print("\nclear-sky model (slope 1): m_inst = mag + {:.3f} X + {:.3f} r^2 + {:.4f} r^4 + {:.3f} (BP-RP) "
          "+ {:.3f}, rms {:.3f} mag, n {:d}".format(c[0], c[1], c[2], c[3], c[4], rms, int(keep.sum())))

    if args.out:
        np.savez(args.out, oid=oids, mag=mag, bp_rp=bp_rp, airmass=airmass, r2=r2, clean=clean,
                 nsat=nsat, crowded=crowded, variable=variable, model=json.dumps(model), **med)


BIN_FRAMES = 16             # 0.64 s summary bins at 25 fps
CELL_NY, CELL_NX = 5, 8     # the transparency map's grid (Utils.VoronoiTreeEstimator)


def catalogColumns(config, lim_mag, fields):
    """ {oid: row} and the requested extra columns of the catalog filtered at lim_mag. """

    from RMS.Formats import StarCatalog

    cat, _, _, ex = StarCatalog.readStarCatalog(config.star_catalog_path, config.star_catalog_file,
        lim_mag=lim_mag, mag_band_ratios=config.star_catalog_band_ratios,
        additional_fields=['oid'] + list(fields))

    return cat, ex


def clouds(args):
    """ Per-star transmission against the clear-sky model, per 0.64 s bin and per block, and its
    comparison with the transparency tree, the stills photometry and the flux cloud intervals. """

    from RMS.Astrometry.Conversions import raDec2AltAz
    from RMS.Formats.Platepar import Platepar
    from RMS.PatchPhotometry import CLEAR_FLUX_FRAC, SNR_MIN
    import RMS.ConfigReader as cr

    model = json.loads(str(np.load(args.model)['model']))
    config = cr.parse(args.config)
    pp = Platepar()
    pp.read(args.platepar, use_flat=False)

    paths = sorted(glob.glob(os.path.join(args.lc_dir, 'LC_*.npz')))
    h0 = json.loads(str(np.load(paths[0])['header']))
    cat, ex = catalogColumns(config, h0['lim_mag'], ['phot_bp_mean_mag', 'phot_rp_mean_mag'])
    colour = dict(zip(ex['oid'].astype(int), ex['phot_bp_mean_mag'] - ex['phot_rp_mean_mag']))

    W, H = model['X_res'], model['Y_res']
    blk_t, blk_t0, blk_T, blk_n, cell_T, cell_n, cell_bg = [], [], [], [], [], [], []
    frames_t, frames_F, frames_Fm, frames_oid, frames_sky = [], [], [], [], []
    bin_T = []
    probe_noise = []
    for p in paths:
        d = np.load(p)
        s, _ = blockStats(p)
        t = d['t']
        jd = float(np.mean(t))/86400.0 + 2440587.5
        _, alt = raDec2AltAz(d['ra'], d['dec'], jd, pp.lat, pp.lon)
        X = 1.0/np.sin(np.radians(np.clip(alt, 5, 90)))
        r2 = ((s['x'] - W/2.0)**2 + (s['y'] - H/2.0)**2)/(H/2.0)**2
        col = np.array([colour.get(int(o), np.nan) for o in d['oid']])
        col = np.where(np.isfinite(col), col, np.nanmedian(col))
        Fm = 10**(-0.4*(d['mag'] + model['k']*X + model['v2']*r2 + model['v4']*r2**2
                        + model['colour']*col + model['zp']))

        # Bias-corrected per-frame net flux (frame ring), the clear-sky model, the sky noise
        Ff = d['flux'] - APERTURE_AREA*d['bg'] - s['probe_bias_fr'][None, :]
        n_bins = Ff.shape[0]//BIN_FRAMES
        Fb = Ff[:n_bins*BIN_FRAMES].reshape(n_bins, BIN_FRAMES, -1).mean(axis=1)

        # Usable for the block cloud metric: block S/N of the clear-sky flux >= 10
        use = (Fm/(s['sky_fr']/np.sqrt(Ff.shape[0])) >= 10) & (d['n_sat'] == 0)
        T = Ff.mean(axis=0)/Fm
        blk_t.append(float(np.mean(t)))
        blk_t0.append(float(t[0]))
        blk_T.append(np.median(T[use]) if use.any() else np.nan)
        blk_n.append(int(use.sum()))
        bin_T.append(np.median(Fb[:, use]/Fm[use][None, :], axis=1) if use.any() else np.full(n_bins, np.nan))

        cx = np.clip((s['x']/(W/float(CELL_NX))).astype(np.intp), 0, CELL_NX - 1)
        cy = np.clip((s['y']/(H/float(CELL_NY))).astype(np.intp), 0, CELL_NY - 1)
        cT = np.full((CELL_NY, CELL_NX), np.nan)
        cN = np.zeros((CELL_NY, CELL_NX), dtype=np.intp)
        cB = np.full((CELL_NY, CELL_NX), np.nan)
        for j in range(CELL_NY):
            for i in range(CELL_NX):
                k = use & (cy == j) & (cx == i)
                cN[j, i] = k.sum()
                if k.any():
                    cT[j, i] = np.median(T[k])
                kk = (cy == j) & (cx == i)
                if kk.any():
                    cB[j, i] = np.median(d['bg_ave'][kk])
        cell_T.append(cT)
        cell_n.append(cN)
        cell_bg.append(cB)

        frames_t.append(t)
        frames_F.append(Ff)
        frames_Fm.append(Fm)
        frames_oid.append(d['oid'])
        frames_sky.append(s['sky_fr'])
        probe_noise.append(np.median((d['probe_flux'] - APERTURE_AREA*d['probe_bg']).std(axis=0)))

    blk_t, blk_T, blk_n = np.array(blk_t), np.array(blk_T), np.array(blk_n)
    cell_T, cell_n, cell_bg = np.array(cell_T), np.array(cell_n), np.array(cell_bg)
    bin_T = np.array(bin_T)

    print("{:d} blocks: block median transmission (usable stars per block {:d}..{:d})".format(
          len(paths), int(blk_n.min()), int(blk_n.max())))
    q = np.nanpercentile(blk_T, [5, 25, 50, 75, 95])
    print("  T percentiles 5/25/50/75/95: " + " ".join("{:.2f}".format(v) for v in q))
    print("  0.64 s bins: T scatter within a block (median over blocks of the bins' std): {:.3f}".format(
          float(np.nanmedian(np.nanstd(bin_T, axis=1)))))

    # Per-minute timeline: median T, the fraction of cells clear (T > 0.5), sky background
    print("\n  minute  median_T  min_bin_T  cells_clear  cells_with_stars  sky_bg(median cell)")
    minute = ((blk_t - blk_t[0])//60).astype(np.intp)
    for m in np.unique(minute):
        k = minute == m
        ct = cell_T[k]
        print("  {:6d} {:9.2f} {:10.2f} {:12.2f} {:17.1f} {:10.2f}".format(int(m), np.nanmedian(blk_T[k]),
              np.nanmin(bin_T[k]), np.nanmean(ct[np.isfinite(ct)] > 0.5) if np.isfinite(ct).any() else np.nan,
              np.mean(np.isfinite(ct).reshape(k.sum(), -1).sum(axis=1)), np.nanmedian(cell_bg[k])))

    # Transparency tree: its per-cell delta magnitude of the same FF (t_unix = FF start) vs ours
    if args.tree:
        tr = np.load(args.tree, allow_pickle=True)
        tt, dm = tr['t_unix'], tr['dm']
        ours, theirs = [], []
        for b in range(len(blk_t)):
            j = int(np.argmin(np.abs(tt - blk_t0[b])))
            if abs(tt[j] - blk_t0[b]) > 0.5:
                continue
            ok = np.isfinite(cell_T[b]) & (cell_T[b] > 0) & np.isfinite(dm[j])
            ours.append(-2.5*np.log10(cell_T[b][ok]))
            theirs.append(dm[j][ok])
        if ours:
            o, th = np.concatenate(ours), np.concatenate(theirs)
            print("\ntree vs light curves, per cell and block: n {:d}, Pearson r {:.3f}, median(ours - tree) "
                  "{:+.3f} mag, MAD {:.3f} mag".format(len(o), np.corrcoef(o, th)[0, 1], np.median(o - th),
                                                     1.4826*np.median(np.abs(o - th - np.median(o - th)))))
            for lo, hi in ((-9, 0.25), (0.25, 0.75), (0.75, 1.5), (1.5, 9)):
                k = (th >= lo) & (th < hi)
                if k.sum():
                    print("  tree dm {:5.2f}..{:5.2f}: n {:5d}  ours median {:.2f}".format(max(lo, -1), min(hi, 5),
                          int(k.sum()), np.median(o[k])))

    # Stills: the same frames, photometred on the saved JPEG stills (code domain, PatchPhotometry)
    if args.stills:
        st = np.load(args.stills, allow_pickle=True)
        sh = json.loads(str(st['header']))
        s_cat, s_ex = catalogColumns(config, float(sh['catalog_lim_mag']), [])
        s_oid = s_ex['oid'][st['bright_cat_id']].astype(int)
        s_t = st['t_unix']

        # Read each array once: indexing an NpzFile decompresses the array again every time
        s_flux = st['bright_flux'].astype(np.float64)
        s_det = st['bright_detected']
        s_ref = np.nanmedian(s_flux, axis=1)
        floor = float(np.median(probe_noise))
        pairs_T, pairs_det, n_match = [], [], 0
        for b, t in enumerate(frames_t):
            for j in np.flatnonzero((s_t >= t[0] - 0.001) & (s_t <= t[-1] + 0.001)):
                n = int(np.argmin(np.abs(t - s_t[j])))
                if not (0 <= t[n] - s_t[j] < 0.001):
                    continue
                n_match += 1
                pos = {int(o): i for i, o in enumerate(frames_oid[b])}
                for k, o in enumerate(s_oid):
                    i = pos.get(int(o))
                    if (i is None) or not np.isfinite(s_flux[k, j]):
                        continue
                    F = frames_F[b][n, i]
                    Fm = frames_Fm[b][i]
                    noise = max(frames_sky[b][i], floor)
                    det = (F/noise >= SNR_MIN) and (F >= CLEAR_FLUX_FRAC*Fm)
                    pairs_T.append((F/Fm, s_flux[k, j]/s_ref[k], Fm/noise))
                    pairs_det.append((det, bool(s_det[k, j])))
        if pairs_T:
            a = np.array(pairs_T)
            dd = np.array(pairs_det)
            ok = np.all(np.isfinite(a), axis=1)
            print("\nstills vs light curves at the same frames: {:d} stills matched to a frame (|dt| < 1 ms), "
                  "{:d} star-frames".format(n_match, len(a)))
            print("  transmission, each normalised to its own clear reference (LC: clear-sky model, linear; "
                  "still: the star's night median, code domain, JPEG), by expected single-frame S/N:")
            for lo, hi in ((0, 3), (3, 6), (6, 12), (12, 1e9)):
                k = ok & (a[:, 2] >= lo) & (a[:, 2] < hi) & (a[:, 1] > 0)
                if k.sum() > 10:
                    print("    S/N {:>4s}..{:<4s}: n {:6d}  Pearson r {:.3f}  median LC/still {:.3f}  "
                          "MAD of LC - still {:.3f}".format(str(lo), str(hi) if hi < 1e9 else 'inf', int(k.sum()),
                          np.corrcoef(a[k, 0], a[k, 1])[0, 1], np.median(a[k, 0]/a[k, 1]),
                          1.4826*np.median(np.abs(a[k, 0] - a[k, 1] - np.median(a[k, 0] - a[k, 1])))))
            print("  detection state agreement: {:.3f} (LC det & still det {:d}, LC only {:d}, still only {:d}, "
                  "neither {:d})".format(np.mean(dd[:, 0] == dd[:, 1]), int(np.sum(dd[:, 0] & dd[:, 1])),
                  int(np.sum(dd[:, 0] & ~dd[:, 1])), int(np.sum(~dd[:, 0] & dd[:, 1])),
                  int(np.sum(~dd[:, 0] & ~dd[:, 1]))))

    # Flux cloud intervals (Utils.Flux.detectClouds), clear periods
    if args.intervals:
        with open(args.intervals) as f:
            iv = json.load(f)
        import datetime
        clear = np.zeros(len(blk_t), dtype=bool)
        for a, b in iv:
            ta = datetime.datetime.strptime(a, "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc).timestamp()
            tb = datetime.datetime.strptime(b, "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc).timestamp()
            clear |= (blk_t >= ta) & (blk_t <= tb)
        print("\nflux clear intervals: {:d} of {:d} blocks inside".format(int(clear.sum()), len(blk_t)))
        for name, k in (('inside (clear)', clear), ('outside', ~clear)):
            if k.any():
                print("  {:16s} block median T: median {:.2f}, 10th pct {:.2f}, fraction T>0.7 {:.2f}".format(
                      name, np.nanmedian(blk_T[k]), np.nanpercentile(blk_T[k], 10), np.nanmean(blk_T[k] > 0.7)))

    if args.out:
        np.savez(args.out, blk_t=blk_t, blk_T=blk_T, blk_n=blk_n, cell_T=cell_T, cell_n=cell_n,
                 cell_bg=cell_bg, bin_T=bin_T)


def limitingMagnitude(args):
    """ Single-frame stellar limiting magnitude per block and region, from the light curves:
        LM = magnitude whose clear-sky flux, dimmed by the region's transmission, equals
             snr * (the probes' per-frame box-sum noise in the region),
    checked against the stars themselves (the magnitude where their per-frame S/N crosses snr) and
    compared with the flux model's stellar LM of the same FFs (Utils.Flux.stellarLMModel of the
    recalibrated platepar's zero point). Also the clear fraction of each block per region: the
    fraction of its 0.64 s bins whose median transmission is above clear_t. """

    from RMS.Astrometry.ApplyAstrometry import xyToRaDecPP
    from RMS.Astrometry.Conversions import raDec2AltAz
    from RMS.Formats import FFfile
    from RMS.Formats.Platepar import Platepar
    from Utils.Flux import stellarLMModel
    import RMS.ConfigReader as cr

    model = json.loads(str(np.load(args.model)['model']))
    config = cr.parse(args.config)
    pp = Platepar()
    pp.read(args.platepar, use_flat=False)
    W, H = model['X_res'], model['Y_res']

    paths = sorted(glob.glob(os.path.join(args.lc_dir, 'LC_*.npz')))
    h0 = json.loads(str(np.load(paths[0])['header']))
    cat, ex = catalogColumns(config, h0['lim_mag'], ['phot_bp_mean_mag', 'phot_rp_mean_mag'])
    colour = dict(zip(ex['oid'].astype(int), ex['phot_bp_mean_mag'] - ex['phot_rp_mean_mag']))
    col_med = float(np.nanmedian(list(colour.values())))

    # Cell centres: radial term and airmass (at each block's time)
    cxc = (np.arange(CELL_NX) + 0.5)*W/CELL_NX
    cyc = (np.arange(CELL_NY) + 0.5)*H/CELL_NY
    gx, gy = np.meshgrid(cxc, cyc)
    r2c = ((gx - W/2.0)**2 + (gy - H/2.0)**2)/(H/2.0)**2

    # Flux model: stellar LM of the recalibrated platepars' zero points
    with open(args.flux_platepars) as f:
        ppr = json.load(f)
    flux_t, flux_lm = [], []
    for name, d in ppr.items():
        if isinstance(d, dict) and d.get('auto_recalibrated') and ('mag_lev' in d):
            t = FFfile.filenameToDatetime(name)
            flux_t.append((t - __import__('datetime').datetime(1970, 1, 1)).total_seconds())
            flux_lm.append(stellarLMModel(float(d['mag_lev'])))
    flux_t, flux_lm = np.array(flux_t), np.array(flux_lm)

    rows = []
    for p in paths:
        d = np.load(p)
        s, _ = blockStats(p)
        t0, tm = float(d['t'][0]), float(np.mean(d['t']))
        jd = tm/86400.0 + 2440587.5

        _, ra, dec, _ = xyToRaDecPP([jd]*gx.size, gx.ravel(), gy.ravel(), [1]*gx.size, pp,
                                    extinction_correction=False, jd_time=True)
        _, alt = raDec2AltAz(np.array(ra), np.array(dec), jd, pp.lat, pp.lon)
        Xc = (1.0/np.sin(np.radians(np.clip(np.array(alt), 5, 90)))).reshape(gx.shape)

        # Per-frame noise of a box sum on the sky (block background), per cell from the probes
        p_sig = d['probe_flux'].std(axis=0, ddof=1)
        pcx = np.clip((d['probe_x']/(W/float(CELL_NX))).astype(np.intp), 0, CELL_NX - 1)
        pcy = np.clip((d['probe_y']/(H/float(CELL_NY))).astype(np.intp), 0, CELL_NY - 1)

        # Stars: transmission per block and per 0.64 s bin, per-frame S/N
        jd_s = jd
        _, alt_s = raDec2AltAz(d['ra'], d['dec'], jd_s, pp.lat, pp.lon)
        Xs = 1.0/np.sin(np.radians(np.clip(alt_s, 5, 90)))
        r2s = ((s['x'] - W/2.0)**2 + (s['y'] - H/2.0)**2)/(H/2.0)**2
        col = np.array([colour.get(int(o), col_med) for o in d['oid']])
        col = np.where(np.isfinite(col), col, col_med)
        Fm = 10**(-0.4*(d['mag'] + model['k']*Xs + model['v2']*r2s + model['v4']*r2s**2
                        + model['colour']*col + model['zp']))
        Ff = d['flux'] - APERTURE_AREA*d['bg_ave'][None, :] - s['probe_bias_ave'][None, :]
        n_bins = Ff.shape[0]//BIN_FRAMES
        Tb = Ff[:n_bins*BIN_FRAMES].reshape(n_bins, BIN_FRAMES, -1).mean(axis=1)/Fm[None, :]
        use = (Fm/(s['sky_ave']/np.sqrt(Ff.shape[0])) >= 10) & (d['n_sat'] == 0)
        scx = np.clip((s['x']/(W/float(CELL_NX))).astype(np.intp), 0, CELL_NX - 1)
        scy = np.clip((s['y']/(H/float(CELL_NY))).astype(np.intp), 0, CELL_NY - 1)
        snr_star = Ff.mean(axis=0)/s['sky_ave']

        for j in range(CELL_NY):
            for i in range(CELL_NX):
                kp = (pcy == j) & (pcx == i)
                ks = use & (scy == j) & (scx == i)
                if (not kp.any()) or (ks.sum() < args.min_stars):
                    continue
                sig = float(np.median(p_sig[kp]))
                T = float(np.median(Ff.mean(axis=0)[ks]/Fm[ks]))
                Tbin = np.median(Tb[:, ks], axis=1)
                clear_frac = float(np.mean(Tbin > args.clear_t))
                if T <= 0:
                    lm = np.nan
                else:
                    lm = -2.5*np.log10(args.snr*sig/T) - (model['k']*Xc[j, i] + model['v2']*r2c[j, i]
                         + model['v4']*r2c[j, i]**2 + model['colour']*col_med + model['zp'])

                # Empirical: the stars of the cell (all, not only 'use'), S/N vs magnitude
                ke = (scy == j) & (scx == i) & (d['n_sat'] == 0) & (snr_star > 0)
                lm_emp = np.nan
                if ke.sum() >= 8:
                    c = np.polyfit(d['mag'][ke], np.log10(snr_star[ke]), 1)
                    if c[0] < 0:
                        lm_emp = (np.log10(args.snr) - c[1])/c[0]
                rows.append((t0, j, i, sig, T, clear_frac, lm, lm_emp, int(ks.sum())))

    a = np.array(rows)
    print("{:d} blocks, {:d} block-cells with >= {:d} usable stars (S/N {:.0f} per frame)".format(
          len(paths), len(a), args.min_stars, args.snr))
    ok = np.isfinite(a[:, 6]) & np.isfinite(a[:, 7])
    print("model LM vs the stars' own S/N crossing (same cells): median diff {:+.3f} mag, MAD {:.3f}, n {:d}".format(
          np.median(a[ok, 6] - a[ok, 7]), 1.4826*np.median(np.abs(a[ok, 6] - a[ok, 7] - np.median(a[ok, 6] - a[ok, 7]))),
          int(ok.sum())))

    # Per block: the field's LM (median over cells), clear fraction, and the flux model's stellar LM
    print("\n  block start (UTC)     LM_frame(median cells)  LM_frame range   clear_frac  flux-model stellar LM")
    blocks_t = np.unique(a[:, 0])
    out = []
    for b, t0 in enumerate(blocks_t):
        k = a[:, 0] == t0
        j = int(np.argmin(np.abs(flux_t - t0))) if len(flux_t) else -1
        f_lm = flux_lm[j] if (j >= 0) and (abs(flux_t[j] - t0) < 1.0) else np.nan
        out.append((t0, np.nanmedian(a[k, 6]), np.nanmin(a[k, 6]), np.nanmax(a[k, 6]), np.mean(a[k, 5]), f_lm))
        if b % args.every == 0:
            import datetime
            print("  {:s}   {:8.2f}          {:5.2f}..{:5.2f}     {:6.2f}       {:s}".format(
                  datetime.datetime.fromtimestamp(t0, tz=datetime.timezone.utc).strftime('%H:%M:%S'),
                  out[-1][1], out[-1][2], out[-1][3], out[-1][4],
                  '{:.2f}'.format(f_lm) if np.isfinite(f_lm) else '-'))
    o = np.array(out)
    k = np.isfinite(o[:, 5]) & np.isfinite(o[:, 1])
    if k.any():
        print("\nflux-model stellar LM - single-frame LM (median over cells), {:d} blocks with a recalibrated "
              "platepar: median {:+.2f} mag, range {:+.2f}..{:+.2f}; r = {:.2f}".format(int(k.sum()),
              np.median(o[k, 5] - o[k, 1]), np.min(o[k, 5] - o[k, 1]), np.max(o[k, 5] - o[k, 1]),
              np.corrcoef(o[k, 5], o[k, 1])[0, 1] if k.sum() > 2 else np.nan))

    if args.out:
        np.savez(args.out, cells=a, blocks=o)


DIP_WIDTHS = (2, 3, 5, 8, 12, 20, 32, 50, 80)     # box widths searched [frames]


def dipSearch(x, sigma, widths=DIP_WIDTHS, margin=4):
    """ Box matched filter for dips in a star's block curve.

    Arguments:
        x: [2D ndarray] (N, S) net flux per frame.
        sigma: [ndarray] (S,) per-frame noise of each star (white: lag-1 autocorrelation ~0 measured).

    Return:
        (z, w, i0): [ndarray] (S,) the largest dip significance (baseline - box mean)*sqrt(w)/sigma
            over all widths and positions, its width and its first frame.
    """

    n, s = x.shape
    base = np.median(x, axis=0)
    c = np.vstack([np.zeros((1, s)), np.cumsum(x, axis=0)])
    best_z = np.full(s, -np.inf)
    best_w = np.zeros(s, dtype=np.intp)
    best_i = np.zeros(s, dtype=np.intp)
    for w in widths:
        if w > n - 2*margin:
            continue
        m = (c[w + margin:n - margin + 1] - c[margin:n - margin - w + 1])/w
        z = (base[None, :] - m)*np.sqrt(w)/sigma[None, :]
        i = np.argmax(z, axis=0)
        zz = z[i, np.arange(s)]
        better = zz > best_z
        best_z[better] = zz[better]
        best_w[better] = w
        best_i[better] = i[better] + margin

    return best_z, best_w, best_i


def overlapFraction(t_start, t_exp, t_in, t_out):
    """ Fraction of each exposure [t_start, t_start + t_exp] inside [t_in, t_out]. """

    return np.clip((np.minimum(t_start + t_exp, t_out) - np.maximum(t_start, t_in))/t_exp, 0, 1)


def fitEdges(x, base, t_start, t_exp, t_in0, t_out0, depth=1.0):
    """ Least-squares ingress and egress times of a box dip (sub-frame), around first guesses from
    the box search (which can be off by up to the spacing of the searched widths): a coarse pass
    over +/-10 frames, then a fine one over +/-10 ms; ingress and egress alternately, twice. """

    def sse(t_in, t_out):
        return np.sum((x - base*(1 - depth*overlapFraction(t_start, t_exp, t_in, t_out)))**2)

    t_in, t_out = t_in0, t_out0
    for half, step in ((10*t_exp, 0.005), (0.010, 0.00025)):
        for _ in range(2):
            g = np.arange(t_in - half, min(t_in + half, t_out), step)
            if len(g):
                t_in = g[int(np.argmin([sse(v, t_out) for v in g]))]
            g = np.arange(max(t_out - half, t_in + step), t_out + half, step)
            if len(g):
                t_out = g[int(np.argmin([sse(t_in, v) for v in g]))]

    return t_in, t_out


def occultation(args):
    """ Dip detection efficiency and ingress/egress timing on injected dips, the false-dip rate of
    the real curves, and (with --other) the coincidence of the strongest dips across two cameras. """

    rng = np.random.RandomState(args.seed)
    paths = sorted(glob.glob(os.path.join(args.lc_dir, 'LC_*.npz')))
    if args.max_blocks:
        paths = paths[:args.max_blocks]

    zmax_real, mag_real = [], []
    inj = []
    n_star_blocks = 0
    for p in paths:
        d = np.load(p)
        h = json.loads(str(d['header']))
        s, _ = blockStats(p)
        t_row = h['t_row_ns']*1e-9
        t_exp = h['exp_s']

        B = APERTURE_AREA*d['bg_ave'] + s['probe_bias_ave']
        x = d['flux'] - B[None, :]
        F = np.median(x, axis=0)
        sigma = 1.4826*np.median(np.abs(x - F[None, :]), axis=0)
        ok = (d['n_sat'] == 0) & (sigma > 0) & np.isfinite(sigma)
        n_star_blocks += int(ok.sum())

        # False dips: the real curves
        z, _, _ = dipSearch(x[:, ok], sigma[ok])
        zmax_real.append(z)
        mag_real.append(d['mag'][ok])

        # Injections: one per star in a random subset, total dip of a random duration, sub-frame
        # ingress; the star's signal is removed and the sky noise kept at its full level with the
        # nearest probe's real fluctuation
        y = 0.5*(d['y_first'] + d['y_last'])
        xs = 0.5*(d['x_first'] + d['x_last'])
        pf = d['probe_flux']
        pr = pf - pf.mean(axis=0)
        from scipy.spatial import cKDTree
        _, near = cKDTree(np.column_stack([d['probe_x'], d['probe_y']])).query(np.column_stack([xs, y]))
        cand = np.flatnonzero(ok & (F > 0))
        for i in rng.choice(cand, size=min(args.inject_per_block, len(cand)), replace=False):
            dur = float(rng.choice(args.durations))
            ts = d['t'] + y[i]*t_row
            t_in = ts[0] + rng.uniform(0.2, 0.8)*(ts[-1] - ts[0] - dur)
            t_out = t_in + dur
            o = overlapFraction(ts, t_exp, t_in, t_out)
            keep = 1.0 - args.depth*o
            xi = keep*x[:, i] + np.sqrt(np.clip(1 - keep**2, 0, 1))*pr[:, near[i]]
            zi, wi, ii = dipSearch(xi[:, None], sigma[i:i + 1])
            hit = (zi[0] >= args.z_thr) and (ii[0] < np.searchsorted(ts, t_out)) \
                and (ii[0] + wi[0] > np.searchsorted(ts, t_in) - 1)
            e_in = e_out = np.nan
            if hit:
                fin, fout = fitEdges(xi, F[i], ts, t_exp, ts[ii[0]], ts[min(ii[0] + wi[0], len(ts) - 1)],
                                     depth=args.depth)
                e_in, e_out = (fin - t_in)*1e3, (fout - t_out)*1e3
            inj.append((d['mag'][i], dur, F[i]/sigma[i], hit, e_in, e_out))

    zr = np.concatenate(zmax_real)
    mr = np.concatenate(mag_real)
    hours = len(paths)*256/25.0/3600.0
    print("{:d} blocks ({:.2f} h), {:d} star-blocks".format(len(paths), hours, n_star_blocks))
    print("\nfalse dips on the real curves (largest box-dip significance per star and block):")
    print("  z_thr   dips   per star-hour   by mag (<4 / 4-5 / 5-6 / >6)")
    sh = n_star_blocks*256/25.0/3600.0
    for thr in (5, 6, 7, 8, 10):
        k = zr >= thr
        print("  {:5.1f} {:6d}   {:.2e}       {}".format(thr, int(k.sum()), k.sum()/sh,
              " / ".join(str(int((k & (mr >= lo) & (mr < hi)).sum())) for lo, hi in ((-9, 4), (4, 5), (5, 6), (6, 99)))))

    a = np.array(inj, dtype=np.float64)
    print("\ninjected total dips (depth {:.2f}), detected at z >= {:.1f} overlapping the dip; timing error of the "
          "sub-frame edge fit [ms] (median |error|, 90th pct):".format(args.depth, args.z_thr))
    print("   mag   dur[s]     n   eff   S/N/frame   ingress |err|  p90   egress |err|  p90")
    for lo, hi in ((0, 3), (3, 4), (4, 5), (5, 6), (6, 7)):
        for dur in args.durations:
            k = (a[:, 0] >= lo) & (a[:, 0] < hi) & (np.abs(a[:, 1] - dur) < 1e-9)
            if k.sum() < 5:
                continue
            hk = k & (a[:, 3] > 0)
            ei, eo = np.abs(a[hk, 4]), np.abs(a[hk, 5])
            print("  {:.0f}-{:.0f}  {:6.2f} {:6d} {:5.2f} {:9.2f}     {:8.1f} {:6.1f}     {:8.1f} {:6.1f}".format(
                  lo, hi, dur, int(k.sum()), np.mean(a[k, 3]), np.median(a[k, 2]),
                  np.median(ei) if len(ei) else np.nan, np.percentile(ei, 90) if len(ei) else np.nan,
                  np.median(eo) if len(eo) else np.nan, np.percentile(eo, 90) if len(eo) else np.nan))

    if args.out:
        np.savez(args.out, zmax_real=zr, mag_real=mr, injections=a)


def coincidence(args):
    """ The same stars on two cameras (by catalog oid, same blocks by time): correlation of their
    per-frame fluctuations, and whether the strongest dips of one camera recur on the other. """

    def load(lc_dir):
        out = {}
        for p in sorted(glob.glob(os.path.join(lc_dir, 'LC_*.npz'))):
            d = np.load(p)
            s, _ = blockStats(p)
            x = d['flux'] - (APERTURE_AREA*d['bg_ave'] + s['probe_bias_ave'])[None, :]
            out[float(d['t'][0])] = (d['t'], d['oid'], x, d['mag'], d['n_sat'])
        return out

    a, b = load(args.lc_dir), load(args.other)
    tb = np.array(sorted(b.keys()))
    corr, z_pairs, n_common, dt_frames = [], [], 0, []
    for t0, (ta, oa, xa, ma, sa) in sorted(a.items()):
        j = int(np.argmin(np.abs(tb - t0)))
        tb0 = tb[j]
        if abs(tb0 - t0) > 5.0:
            continue
        tbv, ob, xb, mb, sb = b[tb0]

        # Frame alignment by time (the two cameras run free; nearest frames)
        idx = np.searchsorted(tbv, ta)
        idx = np.clip(idx, 1, len(tbv) - 1)
        idx = np.where(np.abs(tbv[idx - 1] - ta) < np.abs(tbv[idx] - ta), idx - 1, idx)
        good = np.abs(tbv[idx] - ta) < 0.02
        dt_frames.append(np.median(tbv[idx[good]] - ta[good]))

        common, ia, ib = np.intersect1d(oa, ob, return_indices=True)
        keep = (sa[ia] == 0) & (sb[ib] == 0)
        ia, ib = ia[keep], ib[keep]
        n_common += len(ia)
        if not len(ia):
            continue
        A = xa[good][:, ia]
        Bv = xb[idx[good]][:, ib]
        A = A - A.mean(axis=0)
        Bv = Bv - Bv.mean(axis=0)
        with np.errstate(invalid='ignore'):
            r = (A*Bv).sum(axis=0)/np.sqrt((A*A).sum(axis=0)*(Bv*Bv).sum(axis=0))
        corr.append(np.column_stack([ma[ia], r]))

        sa_ = 1.4826*np.median(np.abs(A), axis=0)
        sb_ = 1.4826*np.median(np.abs(Bv), axis=0)
        za, wa, iA = dipSearch(xa[good][:, ia], sa_)
        zb, wb, iB = dipSearch(xb[idx[good]][:, ib], sb_)
        z_pairs.append(np.column_stack([ma[ia], za, zb, np.abs(iA - iB)]))

    c = np.vstack(corr)
    zp = np.vstack(z_pairs)
    print("{:d} star-blocks on both cameras; frame time offset (other - this) median {:.1f} ms".format(
          n_common, 1e3*np.median(dt_frames)))
    print("per-frame flux correlation of the same star on the two cameras, by magnitude:")
    for lo, hi in ((0, 3), (3, 4), (4, 5), (5, 6), (6, 7)):
        k = (c[:, 0] >= lo) & (c[:, 0] < hi) & np.isfinite(c[:, 1])
        if k.sum() >= 5:
            print("  mag {:.0f}-{:.0f}: n {:5d}  median r {:+.3f}  (90th pct {:+.3f})".format(lo, hi, int(k.sum()),
                  np.median(c[k, 1]), np.percentile(c[k, 1], 90)))
    for thr in (4, 5, 6):
        k = zp[:, 1] >= thr
        both = k & (zp[:, 2] >= thr) & (zp[:, 3] <= 3)
        print("  dips z >= {:.0f} on this camera: {:d}; also on the other within 3 frames: {:d}".format(
              thr, int(k.sum()), int(both.sum())))


def main(argv=None):

    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd')
    sub.required = True

    p = sub.add_parser('photometry', help="linearity slope and scatter vs noise model, clear sky")
    p.add_argument('--lc-dir', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--platepar', required=True)
    p.add_argument('--slope-mag-min', type=float, default=3.5)
    p.add_argument('--slope-mag-max', type=float, default=6.0)
    p.add_argument('--max-blocks', type=int, default=0)
    p.add_argument('--out', help="npz of the per-star results")
    p.set_defaults(func=photometry)

    c = sub.add_parser('clouds', help="transmission vs the clear-sky model; tree, stills, flux intervals")
    c.add_argument('--lc-dir', required=True)
    c.add_argument('--model', required=True, help="npz written by 'photometry --out' on a clear stretch")
    c.add_argument('--config', required=True)
    c.add_argument('--platepar', required=True)
    c.add_argument('--tree', help="<night>_transparency_map_tree.npz")
    c.add_argument('--stills', help="<night>_still_star_states.npz")
    c.add_argument('--intervals', help="flux_time_intervals.json (clear intervals)")
    c.add_argument('--out', help="npz of the per-block results")
    c.set_defaults(func=clouds)

    m = sub.add_parser('lm', help="single-frame stellar LM and clear fraction per block and region")
    m.add_argument('--lc-dir', required=True)
    m.add_argument('--model', required=True, help="npz written by 'photometry --out' on a clear stretch")
    m.add_argument('--config', required=True)
    m.add_argument('--platepar', required=True)
    m.add_argument('--flux-platepars', required=True, help="platepars_flux_recalibrated.json of the night")
    m.add_argument('--snr', type=float, default=5.0, help="single-frame detection S/N")
    m.add_argument('--clear-t', type=float, default=0.5, help="transmission above which a bin is clear")
    m.add_argument('--min-stars', type=int, default=3, help="usable stars a cell needs")
    m.add_argument('--every', type=int, default=30, help="print every Nth block")
    m.add_argument('--out', help="npz of the per-cell results")
    m.set_defaults(func=limitingMagnitude)

    o = sub.add_parser('occultation', help="injected-dip efficiency and timing, false-dip rate")
    o.add_argument('--lc-dir', required=True)
    o.add_argument('--z-thr', type=float, default=7.0, help="dip detection significance")
    o.add_argument('--depth', type=float, default=1.0, help="injected dip depth (1 = total)")
    o.add_argument('--durations', type=float, nargs='+', default=[0.1, 0.2, 0.5, 1.0, 2.0])
    o.add_argument('--inject-per-block', type=int, default=40)
    o.add_argument('--max-blocks', type=int, default=0)
    o.add_argument('--seed', type=int, default=1)
    o.add_argument('--out')
    o.set_defaults(func=occultation)

    k = sub.add_parser('coincidence', help="the same stars on two cameras")
    k.add_argument('--lc-dir', required=True)
    k.add_argument('--other', required=True, help="the other camera's light-curve directory")
    k.set_defaults(func=coincidence)

    args = ap.parse_args(argv)
    args.func(args)

    return 0


if __name__ == '__main__':
    sys.exit(main())
