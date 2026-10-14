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

    if args.out:
        np.savez(args.out, oid=oids, mag=mag, bp_rp=bp_rp, airmass=airmass, r2=r2, clean=clean,
                 nsat=nsat, crowded=crowded, variable=variable, **med)


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

    args = ap.parse_args(argv)
    args.func(args)

    return 0


if __name__ == '__main__':
    sys.exit(main())
