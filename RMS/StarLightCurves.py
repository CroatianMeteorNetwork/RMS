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


SCHEMA_VERSION = 1

APERTURE_SIZE = 2*APERTURE_HALF + 1
APERTURE_AREA = float(APERTURE_SIZE**2)

# Offset fit: bright stars centroided on the block's average image
OFFSET_MAG_MAX = 5.0        # catalog magnitude limit of the centroid stars
OFFSET_N_MAX = 80           # brightest stars used
OFFSET_SEARCH_FIRST = 6     # px search half-width before the first good fit
OFFSET_SEARCH = 3           # px search half-width around the previous offset
OFFSET_SNR_MIN = 10.0       # centroid star S/N on the average image
OFFSET_N_MIN = 5            # good centroids needed to update the offset
OFFSET_CLIP = 1.0           # px; centroids further than this from the median offset are dropped
CENTROID_RADIUS = 3         # px; centroid window half-width


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


    def project(self, t):
        """ Image positions of all catalog stars at unix time t; NaN behind the camera. """

        jd = float(unixToJD(t))
        pp = self.platepar
        x, y = raDecToXYPP(self.ra, self.dec, jd, pp)

        # The distortion polynomial folds stars far off the axis back into the frame (see
        # Utils.StillsSampler.inFrontOfCamera)
        az, alt = raDec2AltAz(self.ra, self.dec, jd, pp.lat, pp.lon)
        az, alt = np.radians(az), np.radians(alt)
        az_c, alt_c = np.radians(pp.az_centre), np.radians(pp.alt_centre)
        front = (np.sin(alt)*np.sin(alt_c) + np.cos(alt)*np.cos(alt_c)*np.cos(az - az_c)) > 0.3

        x = np.where(front, x, np.nan)
        y = np.where(front, y, np.nan)

        return x, y


def apertureWeights(c):
    """ First pixel index and the 1D weights of the APERTURE_SIZE px box centred on c.

    The box [c - APERTURE_SIZE/2, c + APERTURE_SIZE/2] covers APERTURE_SIZE + 1 pixels; the inner
    ones fully, the two end ones by u and 1 - u.

    Arguments:
        c: [ndarray] Fractional centres (float64).

    Return:
        (i0, w): i0 [ndarray np.intp] first pixel, w [ndarray float32] (..., APERTURE_SIZE + 1).
    """

    left = c - APERTURE_SIZE/2.0
    i0 = np.floor(left + 0.5).astype(np.intp)
    u = (left + 0.5 - i0).astype(np.float32)

    w = np.ones(c.shape + (APERTURE_SIZE + 1,), dtype=np.float32)
    w[..., 0] = 1.0 - u
    w[..., -1] = u

    return i0, w


def ringBackground(img, x, y):
    """ Median of the PatchPhotometry ring pixels around the rounded positions (1D x, y). """

    xi = np.round(x).astype(np.intp)
    yi = np.round(y).astype(np.intp)
    ring = np.stack([img[yi + oy, xi + ox] for oy, ox in RING_OFFSETS], axis=-1)

    return np.median(ring, axis=-1)


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

        # Positions at the first and the last frame, linear in between
        x_a, y_a = self.field.project(t_frames[0])
        x_b, y_b = self.field.project(t_frames[-1])
        frac = ((t_frames - t_frames[0])/(t_frames[-1] - t_frames[0])).astype(np.float64)

        # Offset from the block's average image (linear), at the block's mid position
        ave_lin = self.response.decode(ave16/256.0).astype(np.float32)*255.0
        x_mid = 0.5*(x_a + x_b)
        y_mid = 0.5*(y_a + y_b)
        off = self.fitOffset(ave_lin, x_mid, y_mid, self.field.mag)
        x_a = x_a + self.offset[0]
        x_b = x_b + self.offset[0]
        y_a = y_a + self.offset[1]
        y_b = y_b + self.offset[1]

        # Stars whose aperture and ring stay inside the frame for the whole block
        lo = EDGE_MARGIN
        inside = (np.isfinite(x_a) & np.isfinite(x_b)
                  & (np.minimum(x_a, x_b) >= lo) & (np.maximum(x_a, x_b) < w - lo)
                  & (np.minimum(y_a, y_b) >= lo) & (np.maximum(y_a, y_b) < h - lo))
        sel = np.flatnonzero(inside).astype(np.intp)

        # (frame, star) positions, from the end points as stored (float32), so a stored block
        # reproduces them exactly
        xa, xb = (v[sel].astype(np.float32).astype(np.float64) for v in (x_a, x_b))
        ya, yb = (v[sel].astype(np.float32).astype(np.float64) for v in (y_a, y_b))
        x = xa[None, :] + frac[:, None]*(xb - xa)[None, :]
        y = ya[None, :] + frac[:, None]*(yb - ya)[None, :]

        flux, bg, nsat = self.photometry(frames, x, y)

        # The block's background around each star from the average image (256x less noisy than a
        # frame's ring)
        bg_ave = ringBackground(ave_lin, 0.5*(x[0] + x[-1]), 0.5*(y[0] + y[-1])).astype(np.float32)

        return {'sel': sel, 'oid': self.field.oid[sel], 'ra': self.field.ra[sel],
                'dec': self.field.dec[sel], 'mag': self.field.mag[sel],
                'x_first': x[0].astype(np.float32), 'y_first': y[0].astype(np.float32),
                'x_last': x[-1].astype(np.float32), 'y_last': y[-1].astype(np.float32),
                'flux': flux, 'bg': bg, 'bg_ave': bg_ave, 'n_sat': nsat, 'offset': off}


    def photometry(self, frames, x, y):
        """ Linear aperture sums, ring backgrounds and saturated-frame counts.

        Arguments:
            frames: [3D ndarray uint8] (N, H, W).
            x, y: [2D ndarray] (N, S) star positions on each frame.

        Return:
            (flux, bg, n_sat): flux [N, S] float32 aperture sum, bg [N, S] float32 ring median
                (per pixel), n_sat [S] uint16 frames with a code-255 pixel in the aperture.
        """

        n_frames, h, w = frames.shape
        flat = frames.reshape(-1)
        lut = self.lut

        ix0, wx = apertureWeights(x)
        iy0, wy = apertureWeights(y)
        k = np.arange(APERTURE_SIZE + 1, dtype=np.intp)

        # (N, S, 6, 6) footprint codes, gathered through one flat index per pixel
        base = (np.arange(n_frames, dtype=np.intp)[:, None]*h + iy0)*w + ix0
        offs = (k[:, None]*w + k[None, :]).reshape(-1)
        codes = flat[base[..., None] + offs].reshape(x.shape + (APERTURE_SIZE + 1, APERTURE_SIZE + 1))
        flux = np.einsum('nsij,nsi,nsj->ns', lut[codes], wy, wx, dtype=np.float32)

        # Saturation: a code 255 inside the aperture (weight > 0)
        wgt = (wy[..., :, None]*wx[..., None, :]) > 0
        nsat = np.count_nonzero(((codes == 255) & wgt).any(axis=(2, 3)), axis=0).astype(np.uint16)

        xi = np.round(x).astype(np.intp)
        yi = np.round(y).astype(np.intp)
        base = (np.arange(n_frames, dtype=np.intp)[:, None]*h + yi)*w + xi
        roffs = np.array([oy*w + ox for oy, ox in RING_OFFSETS], dtype=np.intp)
        bg = np.median(lut[flat[base[..., None] + roffs]], axis=-1).astype(np.float32)

        return flux, bg, nsat


def saveBlock(path, res, t_frames, frame_seq, header):
    """ Write one block's light curves (test format: everything, uncompressed npz).

    Arrays:
        oid [S] int32 catalog object id; ra, dec [S] float64 deg; mag [S] float32 catalog mag.
        x_first, y_first, x_last, y_last [S] float32 position on the first and last frame.
        n_sat [S] uint16 frames with a saturated aperture pixel.
        bg_ave [S] float32 ring median per pixel on the block's average image.
        flux [N, S] float32 aperture sum (linear codes); bg [N, S] float32 ring median per pixel.
        t [N] float64 unix start of integration of row 0; frame_seq [N] int64 camera frame number.
        header: JSON string.
    """

    np.savez(path, header=json.dumps(header), oid=res['oid'], ra=res['ra'], dec=res['dec'],
             mag=res['mag'], x_first=res['x_first'], y_first=res['y_first'], x_last=res['x_last'],
             y_last=res['y_last'], n_sat=res['n_sat'], bg_ave=res['bg_ave'], flux=res['flux'],
             bg=res['bg'],
             t=np.asarray(t_frames, dtype=np.float64), frame_seq=np.asarray(frame_seq, dtype=np.int64))
