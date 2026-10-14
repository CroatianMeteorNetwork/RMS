""" Per-star light curves measured on a frame block while it is in memory.

An FF keeps max/mean/std per pixel over its 256-frame block, so a star's brightness over the
10.24 s is lost. This pass measures every catalog star in the field on every frame of the block,
before the capture buffer is released:

- Positions: catalog RA/Dec through the station platepar at the first and the last frame of the
  block, linear in between (the sidereal motion over a block is ~1 px at 2.6 arcmin/px, ~4 px with
  16 mm lenses). One block offset, fitted to the bright stars centroided on the block's average
  image, moves all stars, so a faint or occluded star never chases noise. When too few bright
  stars centroid (cloud), the previous block's offset is kept.
- Photometry: the frames are linearised with the decode table the compressor uses
  (ResponseCurve.decodeLUT, or the power law of config.gamma). Aperture and background ring are
  those of RMS.PatchPhotometry: a 5x5 px aperture, here centred on the star's fractional
  position (pixel weights are the overlap of each pixel with the 5x5 px square, so the sum does
  not jump when the star drifts across a pixel boundary), and the median of the 8 ring pixels at
  +/-7 px around the rounded position. Both are stored raw, per star and frame; the net flux is
  sum - APERTURE_AREA*background. Any detection derived from them must apply the two noise
  floors of RMS.PatchPhotometry.

Pixel coordinates are numpy's: pixel (row j, column i) is centred on (x, y) = (i, j).
"""

from __future__ import absolute_import, division, print_function

import json

import numpy as np

from RMS.Astrometry.ApplyAstrometry import raDecToXYPP
from RMS.Astrometry.Conversions import raDec2AltAz
from RMS.Formats import StarCatalog
from RMS.PatchPhotometry import APERTURE_HALF, EDGE_MARGIN, RING_OFFSETS
from RMS.Routines.Response import responseOf
from RMS.StarLightCurvesCy import boxPhotometry


SCHEMA_VERSION = 1

APERTURE_SIZE = 2*APERTURE_HALF + 1
APERTURE_AREA = float(APERTURE_SIZE**2)
RING = np.array(RING_OFFSETS, dtype=np.intp)

# The catalog is projected in full only every FIELD_REFRESH_S of sky time; the stars that can be
# in the frame until the next refresh (their own motion over the interval plus FIELD_MARGIN px)
# are projected for every block
FIELD_REFRESH_S = 600.0
FIELD_MARGIN = 40.0

# Offset fit: bright stars centroided on the block's average image
OFFSET_MAG_MAX = 5.0        # catalog magnitude limit of the centroid stars
OFFSET_N_MAX = 80           # brightest stars used
OFFSET_SEARCH_FIRST = 6     # px search half-width before the first good fit
OFFSET_SEARCH = 3           # px search half-width around the previous offset
OFFSET_SNR_MIN = 10.0       # centroid star S/N on the average image
OFFSET_N_MIN = 5            # good centroids needed to update the offset
OFFSET_CLIP = 1.0           # px; centroids further than this from the median offset are dropped
CENTROID_RADIUS = 3         # px; centroid window half-width

# Sky probes: blank-sky positions measured exactly like stars, one per cell of a PROBE_NX x
# PROBE_NY grid, at least PROBE_CLEAR px from every catalog star over the whole block. They give
# the per-frame noise of the aperture statistic (PatchPhotometry's frame-noise floor, and the
# spatially correlated aperture noise that the per-pixel sigma does not show) per region
PROBE_NX = 24
PROBE_NY = 16
PROBE_CLEAR = 8.0
PROBE_AVOID_MAG = 8.5       # catalog stars to this magnitude are kept clear of the probes


def unixToJD(t):
    return np.asarray(t, dtype=np.float64)/86400.0 + 2440587.5


class StarField(object):

    def __init__(self, config, platepar, lim_mag, years_from_J2000=None):
        """ The catalog stars to be measured with a platepar.

        Arguments:
            config: [Config] Station config (star catalog, gamma, response).
            platepar: [Platepar] Station astrometry.
            lim_mag: [float] Catalog limiting magnitude of the measured stars.

        Keyword arguments:
            years_from_J2000: [float] Proper-motion epoch; None for the platepar's epoch.
        """

        self.config = config
        self.platepar = platepar
        self.lim_mag = float(lim_mag)

        if years_from_J2000 is None:
            years_from_J2000 = (platepar.JD - 2451545.0)/365.25

        stars, band, ratios, extras = StarCatalog.readStarCatalog(config.star_catalog_path,
            config.star_catalog_file, years_from_J2000=years_from_J2000, lim_mag=self.lim_mag,
            mag_band_ratios=config.star_catalog_band_ratios,
            additional_fields=['oid', 'designation', 'preferred_name'])

        self.catalog_name = "{:s} ({:s})".format(config.star_catalog_file, band)
        self.ra = stars[:, 0].astype(np.float64)
        self.dec = stars[:, 1].astype(np.float64)
        self.mag = stars[:, 2].astype(np.float32)
        self.oid = np.asarray(extras['oid'], dtype=np.int32)
        self.designation = np.asarray(extras['designation'])
        self.preferred_name = np.asarray(extras['preferred_name'])

        # Deeper positions, only to keep the sky probes clear of fainter stars
        avoid, _, _ = StarCatalog.readStarCatalog(config.star_catalog_path,
            config.star_catalog_file, years_from_J2000=years_from_J2000,
            lim_mag=max(PROBE_AVOID_MAG, self.lim_mag), mag_band_ratios=config.star_catalog_band_ratios)
        self.avoid_ra = avoid[:, 0].astype(np.float64)
        self.avoid_dec = avoid[:, 1].astype(np.float64)

        # Stars that can be in the frame until valid_until (indices into the catalogs)
        self.near = np.arange(len(self.ra), dtype=np.intp)
        self.near_avoid = np.arange(len(self.avoid_ra), dtype=np.intp)
        self.valid_until = None


    def refresh(self, t, w, h):
        """ Select the stars that can be within FIELD_MARGIN px of the w x h frame between t and
        t + FIELD_REFRESH_S. """

        def near(ra, dec):
            x0, y0 = self._project(ra, dec, t)
            x1, y1 = self._project(ra, dec, t + FIELD_REFRESH_S)
            m = FIELD_MARGIN + np.nan_to_num(np.hypot(x1 - x0, y1 - y0), nan=0.0)
            keep = np.zeros(len(ra), dtype=bool)
            for x, y in ((x0, y0), (x1, y1)):
                with np.errstate(invalid='ignore'):
                    keep |= (x > -m) & (x < w + m) & (y > -m) & (y < h + m)
            return np.flatnonzero(keep).astype(np.intp)

        self.near = near(self.ra, self.dec)
        self.near_avoid = near(self.avoid_ra, self.avoid_dec)
        self.valid_until = t + FIELD_REFRESH_S

        # The probe-avoidance stars move linearly enough over the interval (they only need to be
        # placed to well within PROBE_CLEAR), so they are interpolated between its two ends
        ra, dec = self.avoid_ra[self.near_avoid], self.avoid_dec[self.near_avoid]
        self.avoid_t0 = t
        self.avoid_xy0 = np.array(self._project(ra, dec, t))
        self.avoid_xy1 = np.array(self._project(ra, dec, t + FIELD_REFRESH_S))


    def project(self, t, avoid=False):
        """ Image positions at unix time t of the selected catalog stars (or of the probe-avoidance
        stars); NaN behind the camera.

        Return:
            (idx, x, y): idx [ndarray np.intp] indices into the catalog, x, y their positions.
        """

        if avoid:
            idx = self.near_avoid
            a = (t - self.avoid_t0)/FIELD_REFRESH_S
            x, y = self.avoid_xy0 + a*(self.avoid_xy1 - self.avoid_xy0)
        else:
            idx = self.near
            x, y = self._project(self.ra[idx], self.dec[idx], t)

        return idx, x, y


    def _project(self, ra, dec, t):

        jd = float(unixToJD(t))
        pp = self.platepar
        x, y = raDecToXYPP(ra, dec, jd, pp)

        # The distortion polynomial folds stars far off the axis back into the frame (see
        # Utils.StillsSampler.inFrontOfCamera)
        az, alt = raDec2AltAz(ra, dec, jd, pp.lat, pp.lon)
        az, alt = np.radians(az), np.radians(alt)
        az_c, alt_c = np.radians(pp.az_centre), np.radians(pp.alt_centre)
        front = (np.sin(alt)*np.sin(alt_c) + np.cos(alt)*np.cos(alt_c)*np.cos(az - az_c)) > 0.3

        x = np.where(front, x, np.nan)
        y = np.where(front, y, np.nan)

        return x, y


def ringBackground(img, x, y):
    """ Median of the PatchPhotometry ring pixels around the rounded positions (1D x, y). """

    xi = np.floor(x + 0.5).astype(np.intp)
    yi = np.floor(y + 0.5).astype(np.intp)
    ring = np.stack([img[yi + oy, xi + ox] for oy, ox in RING_OFFSETS], axis=-1)

    return np.median(ring, axis=-1)


def skyProbes(w, h, star_x, star_y):
    """ One blank-sky position per cell of the PROBE_NX x PROBE_NY grid: the first of a fixed list
    of candidates in the cell (centre first) that is PROBE_CLEAR px from every star position given
    and inside the measurable area. Cells without such a candidate get no probe.

    Return:
        (x, y): [ndarray float64] probe positions (integers, so the box sum is a plain 5x5 sum).
    """

    from scipy.spatial import cKDTree

    cw, ch = w/float(PROBE_NX), h/float(PROBE_NY)
    fr = np.array([(0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75),
                   (0.5, 0.25), (0.5, 0.75), (0.25, 0.5), (0.75, 0.5)])
    gx, gy = np.meshgrid(np.arange(PROBE_NX), np.arange(PROBE_NY))
    cand_x = np.floor((gx.reshape(-1, 1) + fr[None, :, 0])*cw)
    cand_y = np.floor((gy.reshape(-1, 1) + fr[None, :, 1])*ch)

    ok = ((cand_x >= EDGE_MARGIN) & (cand_x < w - EDGE_MARGIN)
          & (cand_y >= EDGE_MARGIN) & (cand_y < h - EDGE_MARGIN))
    if len(star_x):
        d, _ = cKDTree(np.column_stack([star_x, star_y])).query(
            np.column_stack([cand_x.ravel(), cand_y.ravel()]))
        ok &= (d.reshape(cand_x.shape) >= PROBE_CLEAR)

    cell = np.flatnonzero(ok.any(axis=1))
    first = np.argmax(ok[cell], axis=1)

    return cand_x[cell, first], cand_y[cell, first]


def peak(img, x, y, half):
    """ Position of the brightest 3x3 box sum in windows of half-width `half` around (x, y). """

    xi = np.round(x).astype(np.intp)
    yi = np.round(y).astype(np.intp)
    d = np.arange(-half - 1, half + 2)
    win = img[yi[:, None, None] + d[None, :, None], xi[:, None, None] + d[None, None, :]]

    # 3x3 box sums at offsets -half..half
    c = np.cumsum(np.cumsum(np.pad(win, ((0, 0), (1, 0), (1, 0))), axis=1), axis=2)
    box = c[:, 3:, 3:] - c[:, :-3, 3:] - c[:, 3:, :-3] + c[:, :-3, :-3]
    k = np.argmax(box.reshape(len(xi), -1), axis=1)
    n = box.shape[1]

    return xi + (k % n) - half, yi + (k//n) - half


def centroid(img, x, y, half):
    """ Background-subtracted intensity centroids in windows of half-width `half` around (x, y).

    Return:
        (cx, cy, snr): [ndarray] centroids and the S/N of the window's net sum (noise from the MAD
            of the window border, which is also the background).
    """

    xi = np.round(x).astype(np.intp)
    yi = np.round(y).astype(np.intp)
    d = np.arange(-half, half + 1)
    win = img[yi[:, None, None] + d[None, :, None], xi[:, None, None] + d[None, None, :]]

    border = np.concatenate([win[:, 0, :], win[:, -1, :], win[:, 1:-1, 0], win[:, 1:-1, -1]], axis=1)
    bg = np.median(border, axis=1)
    sig = 1.4826*np.median(np.abs(border - bg[:, None]), axis=1)

    net = win - bg[:, None, None]
    pos = np.clip(net, 0, None)
    s = pos.sum(axis=(1, 2))
    with np.errstate(invalid='ignore', divide='ignore'):
        cx = xi + (pos.sum(axis=1)*d[None, :]).sum(axis=1)/s
        cy = yi + (pos.sum(axis=2)*d[None, :]).sum(axis=1)/s
        snr = net.sum(axis=(1, 2))/(np.maximum(sig, 1e-3)*win.shape[1])

    return cx, cy, snr


class StarLightCurves(object):

    def __init__(self, config, platepar, lim_mag):
        """ The per-block light-curve pass of one station.

        Arguments:
            config: [Config] Station config.
            platepar: [Platepar] Station astrometry.
            lim_mag: [float] Catalog limiting magnitude of the measured stars.
        """

        self.config = config
        self.field = StarField(config, platepar, lim_mag)

        response = responseOf(config)
        self.response_id = response.ident() if not response.isPower else ''
        self.lut = np.ascontiguousarray(response.decodeLUT(255), dtype=np.float32)
        self.response = response

        # Block offset carried from block to block
        self.offset = np.zeros(2)
        self.offset_good = False


    def fitOffset(self, ave_lin, x_mid, y_mid, mag):
        """ Fit the block offset (measured - predicted) from the bright stars on the block's linear
        average image. Updates self.offset when at least OFFSET_N_MIN stars centroid.

        Return:
            [dict] offset fit provenance.
        """

        h, w = ave_lin.shape
        search = OFFSET_SEARCH if self.offset_good else OFFSET_SEARCH_FIRST
        half = search + CENTROID_RADIUS
        px = x_mid + self.offset[0]
        py = y_mid + self.offset[1]

        cand = np.flatnonzero((mag <= OFFSET_MAG_MAX) & (px >= half + 1) & (px < w - half - 1)
                              & (py >= half + 1) & (py < h - half - 1))
        cand = cand[np.argsort(mag[cand])][:OFFSET_N_MAX]

        info = {'n_cand': int(len(cand)), 'n_used': 0, 'dx': float(self.offset[0]),
                'dy': float(self.offset[1]), 'rms': float('nan'), 'updated': False}
        if len(cand) < OFFSET_N_MIN:
            return info

        # Coarse: the brightest 3x3 box of the search window; then centroids around it
        cx, cy = peak(ave_lin, px[cand], py[cand], search)
        for _ in range(2):
            cx, cy, snr = centroid(ave_lin, cx, cy, CENTROID_RADIUS)
            ok = np.isfinite(cx) & np.isfinite(cy)
            cx = np.where(ok, np.clip(cx, half, w - half - 1), px[cand])
            cy = np.where(ok, np.clip(cy, half, h - half - 1), py[cand])

        ok = ok & (snr >= OFFSET_SNR_MIN)
        if np.count_nonzero(ok) < OFFSET_N_MIN:
            return info

        dx = cx[ok] - x_mid[cand[ok]]
        dy = cy[ok] - y_mid[cand[ok]]
        mx, my = np.median(dx), np.median(dy)
        keep = np.hypot(dx - mx, dy - my) <= OFFSET_CLIP
        if np.count_nonzero(keep) < OFFSET_N_MIN:
            return info

        self.offset = np.array([np.mean(dx[keep]), np.mean(dy[keep])])
        self.offset_good = True
        res = np.hypot(dx[keep] - self.offset[0], dy[keep] - self.offset[1])
        info.update({'n_used': int(np.count_nonzero(keep)), 'dx': float(self.offset[0]),
                     'dy': float(self.offset[1]), 'rms': float(np.sqrt(np.mean(res**2))),
                     'updated': True})

        return info


    def measureBlock(self, frames, ave16, t_frames):
        """ Measure all stars of the field on every frame of a block.

        Arguments:
            frames: [3D ndarray uint8] (N, H, W) frame block (the compressor's buffer).
            ave16: [2D ndarray uint16] The block's average image in 8.8 fixed point (compressFrames).
            t_frames: [ndarray] (N,) unix time of each frame (start of integration of row 0).

        Return:
            [dict] of arrays (see saveBlock) and 'offset' provenance.
        """

        n_frames, h, w = frames.shape
        t_frames = np.asarray(t_frames, dtype=np.float64)

        if (self.field.valid_until is None) or (t_frames[-1] > self.field.valid_until):
            self.field.refresh(t_frames[0], w, h)

        # Positions at the first and the last frame, linear in between
        idx, x_a, y_a = self.field.project(t_frames[0])
        _, x_b, y_b = self.field.project(t_frames[-1])
        frac = ((t_frames - t_frames[0])/(t_frames[-1] - t_frames[0])).astype(np.float64)

        # The block's average image, linear: the decode table interpolated at the 8.8 fixed-point
        # codes (the piecewise-linear ResponseCurve.decode)
        ave = np.asarray(ave16)
        i = (ave >> 8).astype(np.intp)
        ave_lin = self.lut[i] + (ave & 0xFF).astype(np.float32)/256.0 \
            *(self.lut[np.minimum(i + 1, 255)] - self.lut[i])

        # Offset from the block's average image, at the block's mid position
        off = self.fitOffset(ave_lin, 0.5*(x_a + x_b), 0.5*(y_a + y_b), self.field.mag[idx])
        x_a = x_a + self.offset[0]
        x_b = x_b + self.offset[0]
        y_a = y_a + self.offset[1]
        y_b = y_b + self.offset[1]

        # Stars whose aperture and ring stay inside the frame for the whole block. Positions are
        # interpolated from the end points as stored (float32), so a stored block reproduces them
        lo = EDGE_MARGIN
        with np.errstate(invalid='ignore'):
            inside = (np.isfinite(x_a) & np.isfinite(x_b)
                      & (np.minimum(x_a, x_b) >= lo) & (np.maximum(x_a, x_b) < w - lo)
                      & (np.minimum(y_a, y_b) >= lo) & (np.maximum(y_a, y_b) < h - lo))
        k = np.flatnonzero(inside).astype(np.intp)
        sel = idx[k]
        xa, xb, ya, yb = (v[k].astype(np.float32) for v in (x_a, x_b, y_a, y_b))

        flux, bg, nsat = self.photometry(frames, xa, xb, ya, yb, frac)

        # The block's background around each star from the average image (256x less noisy than a
        # frame's ring)
        bg_ave = ringBackground(ave_lin, 0.5*(xa.astype(np.float64) + xb),
                                0.5*(ya.astype(np.float64) + yb)).astype(np.float32)

        # Sky probes, clear of every catalog star to PROBE_AVOID_MAG during the block
        sx, sy = [], []
        for t in (t_frames[0], t_frames[-1]):
            _, ax, ay = self.field.project(t, avoid=True)
            fin = np.isfinite(ax) & np.isfinite(ay)
            sx.append(ax[fin] + self.offset[0])
            sy.append(ay[fin] + self.offset[1])
        px, py = skyProbes(w, h, np.concatenate(sx), np.concatenate(sy))
        px, py = px.astype(np.float32), py.astype(np.float32)
        p_flux, p_bg, _ = self.photometry(frames, px, px, py, py, frac)

        return {'sel': sel, 'oid': self.field.oid[sel], 'ra': self.field.ra[sel],
                'dec': self.field.dec[sel], 'mag': self.field.mag[sel],
                'x_first': xa, 'y_first': ya, 'x_last': xb, 'y_last': yb,
                'flux': flux, 'bg': bg, 'bg_ave': bg_ave, 'n_sat': nsat, 'offset': off,
                'probe_x': px, 'probe_y': py, 'probe_flux': p_flux, 'probe_bg': p_bg}


    def photometry(self, frames, x_first, x_last, y_first, y_last, frac):
        """ Linear aperture sums, ring backgrounds and saturated-frame counts of positions moving
        linearly from (x_first, y_first) to (x_last, y_last) over the block (frame n at fraction
        frac[n]); see RMS.StarLightCurvesCy.boxPhotometry.

        Return:
            (flux, bg, n_sat): flux [N, S] float32 aperture sum, bg [N, S] float32 ring median
                (per pixel), n_sat [S] uint16 frames with a code-255 pixel in the aperture.
        """

        # Measured in row order, which keeps the frame accesses local (4x faster than catalog
        # order at 2000 stars), and returned in the caller's order
        order = np.argsort(np.asarray(y_first) + np.asarray(y_last), kind='stable')
        f64 = lambda v: np.ascontiguousarray(np.asarray(v)[order], dtype=np.float64)

        flux, bg, nsat = boxPhotometry(np.ascontiguousarray(frames), self.lut, f64(x_first),
            f64(x_last), f64(y_first), f64(y_last), np.ascontiguousarray(frac, dtype=np.float64),
            APERTURE_HALF, RING)

        inv = np.empty_like(order)
        inv[order] = np.arange(len(order), dtype=order.dtype)

        return flux[:, inv], bg[:, inv], nsat[inv]


def saveBlock(path, res, t_frames, frame_seq, header):
    """ Write one block's light curves (test format: everything, uncompressed npz).

    Arrays:
        oid [S] int32 catalog object id; ra, dec [S] float64 deg; mag [S] float32 catalog mag.
        x_first, y_first, x_last, y_last [S] float32 position on the first and last frame.
        n_sat [S] uint16 frames with a saturated aperture pixel.
        bg_ave [S] float32 ring median per pixel on the block's average image.
        flux [N, S] float32 aperture sum (linear codes); bg [N, S] float32 ring median per pixel.
        probe_x, probe_y [P] float32 sky-probe positions (static over the block).
        probe_flux, probe_bg [N, P] float32 the probes' aperture sums and ring medians.
        t [N] float64 unix start of integration of row 0; frame_seq [N] int64 camera frame number.
        header: JSON string.
    """

    np.savez(path, header=json.dumps(header), oid=res['oid'], ra=res['ra'], dec=res['dec'],
             mag=res['mag'], x_first=res['x_first'], y_first=res['y_first'], x_last=res['x_last'],
             y_last=res['y_last'], n_sat=res['n_sat'], bg_ave=res['bg_ave'], flux=res['flux'],
             bg=res['bg'], probe_x=res['probe_x'], probe_y=res['probe_y'],
             probe_flux=res['probe_flux'], probe_bg=res['probe_bg'],
             t=np.asarray(t_frames, dtype=np.float64), frame_seq=np.asarray(frame_seq, dtype=np.int64))
