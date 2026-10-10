""" Per-frame meteor photometry from an FF file: an unthresholded aperture plus the frame-split
correction.

Why this exists
---------------
An FF stores, per pixel, the maximum over 256 frames (maxpixel) and the frame it came from (maxframe).
The detector reconstructs frame i from the pixels whose maxframe is i. Two things then lose meteor
light, and neither is cancelled by the photometric zero point, because stars are measured with an
unthresholded box aperture on a static image (ExtractStars.fitPSF):

1. Threshold-mask photometry (the large one). The detector used to report the sum of the THRESHOLD
   PASSER pixels of the frame. Faint meteors have few passers, so the reported intensity fell with
   the meteor's signal-to-noise: against raw video it recovered a median 0.61 of the light (0.07 to
   0.85, correlation with log SNR 0.93), i.e. meteor magnitudes read +0.53 mag too faint on median
   and up to ~3 mag for the faintest. This module sums a fixed aperture around the frame's segment
   instead, with no threshold, the way stars are summed.

2. Frame split (the bias that remains after 1). A pixel the meteor crosses while its row switches
   from frame i to i+1 gets part of its light in each frame; maxpixel keeps the larger part only and
   the smaller part is stored nowhere (avepixel is trimmed, so it drops it too). These are the dark
   "beads" seen along fast streaks in maxpixel, one per frame. For a Gaussian PSF of sigma s (px)
   and a meteor moving v px/frame, the light lost per frame boundary is integral of
   min(Phi(x/s), 1 - Phi(x/s)) dx = sqrt(2/pi)*s px of streak, so the fraction of a frame's light
   lost is sqrt(2/pi)*s/v. It is NOT sensor dead time: the exposure is the full frame less a few
   lines (IMX307: VMAX - 2 lines = 59 us of 40 ms), which moves even a fast meteor < 0.1 px.

Validation (2026-10-10, UST002 = IMX662 on GK7605V100, night 2026-10-08, 22 unsaturated meteors at
5-27 px/frame, against the station's saved raw video, which reproduces the FF exactly:
raw[maxframe] == maxpixel on 100% of the meteor pixels):
    - split loss measured vs sqrt(2/pi)*s/v: median -0.007, rms 0.029
    - magnitude bias vs truth (median, p90-p10 spread):
          threshold passers (old)            +0.53 mag   0.94
          old x split correction             +0.38       0.84
          aperture, no threshold             +0.21       0.24
          aperture x split correction        +0.06       0.20
    - the remaining ~5 % is PSF wing light whose maxframe fell in an unrelated frame (noise won the
      max), median 7 %, speed independent. Not corrected here.
    - end to end, the patched detector on the same 22 meteors: median 0.947 of the true light,
      +0.06 mag, spread 0.15 mag (was 0.613, +0.53 mag, 0.94).
    - detection unchanged: A/B over UST002 (100 FFs, 115 meteors) and US05D1 (IMX307, 20 FFs,
      37 meteors) gave identical detections, frames, positions, background, SNR and saturation
      counts; only the intensity differs (new/old median 1.55 and 1.34, never below 1). Detection
      time +-2 %.

Scope
-----
- FF input only. Video input reconstructs real frames and is unaffected.
- The split correction is applied only where the model was validated: a loss fraction of at most
  0.25 (about v >= 5 px/frame for these cameras). Slower meteors light a pixel over several frames,
  the model under-predicts there, and no correction is applied.
- Saturated pixels are summed as they are (clipped).
- The detector floors each frame at the old threshold-passer sum: the aperture contains the passers,
  so it can only come out lower if the geometry failed (e.g. a long curved aircraft track); the old
  value is kept then.
"""

from __future__ import absolute_import, division, print_function

import math

import numpy as np


# Light lost per frame boundary, in px of streak per px of PSF sigma: integral over x of
# min(Phi(x/s), 1 - Phi(x/s)) = 2*s*phi(0) = sqrt(2/pi)*s
SPLIT_K = math.sqrt(2.0/math.pi)


def splitLossFraction(sigma, v):
    """ Fraction of a frame's meteor light the FF drops at the frame boundaries.

    Arguments:
        sigma: [float] Gaussian sigma of the meteor's cross-track profile (px).
        v: [float] Meteor speed on the image (px/frame).

    Return:
        [float] sqrt(2/pi)*sigma/v, or nan if v <= 0.
    """

    if v <= 0:
        return float('nan')

    return SPLIT_K*sigma/v


class FFAperturePhotometry(object):
    """ Aperture photometry of one meteor in an FF, frame by frame.

    The aperture of frame i is every pixel whose maxframe is i, within halfwidth px across the track
    and within v/2 + margin px along it of the frame's position. The track direction and the speed v
    (px/frame) come from a line fitted to the detected line points (threshold passers of the meteor,
    with their frame). The frame's position is its centroid when the caller passes it, so the
    aperture follows curved or decelerating tracks; otherwise the fitted position. The aperture is
    widened to take in all of the frame's own line points (+2 px), so bright, bloomed objects wider
    than halfwidth are not clipped.

    Arguments:
        maxframe: [2D ndarray] FF maxframe plane.
        max_avg_corrected: [2D ndarray] maxpixel - avepixel in linear light (the detector's
            max_avg_corrected; masked pixels are already zero in both planes).
        line_points: [ndarray] N x 3 array of (x, y, frame) of the meteor's line points.
        halfwidth: [float] Aperture half width across the track (px), the minimum.
        margin: [float] Aperture extension beyond the frame's segment along the track (px).

    Keyword arguments:
        deinterlace_order: [int] -1 for progressive video, else the field order (rows of the field
            are selected the same way the centroiding does).
    """

    def __init__(self, maxframe, max_avg_corrected, line_points, halfwidth=4.5, margin=6.0,
        deinterlace_order=-1):

        self.halfwidth = float(halfwidth)
        self.margin = float(margin)
        self.deinterlace_order = deinterlace_order

        self.lp = np.asarray(line_points, dtype=np.float64)
        x, y, fr = self.lp[:, 0], self.lp[:, 1], self.lp[:, 2]

        # Linear motion fitted to the line points: position(frame) = slope*frame + intercept
        self.ax, self.bx = np.polyfit(fr, x, 1)
        self.ay, self.by = np.polyfit(fr, y, 1)
        self.v = math.hypot(self.ax, self.ay)

        # Along-track unit vector (any direction if the meteor does not move) and its normal
        if self.v > 1e-6:
            self.u = np.array([self.ax, self.ay])/self.v
        else:
            self.u = np.array([1.0, 0.0])
        self.n = np.array([-self.u[1], self.u[0]])

        # Crop covering the line points plus the largest default aperture around them
        reach = self.v/2.0 + self.margin + self.halfwidth + 3.0
        nrows, ncols = maxframe.shape
        self.x0, self.x1 = max(int(np.floor(x.min() - reach)), 0), min(int(np.ceil(x.max() + reach)) + 1, ncols)
        self.y0, self.y1 = max(int(np.floor(y.min() - reach)), 0), min(int(np.ceil(y.max() + reach)) + 1, nrows)

        yy, xx = np.mgrid[self.y0:self.y1, self.x0:self.x1]
        self.xx, self.yy = xx.astype(np.float64), yy.astype(np.float64)
        self.mf = maxframe[self.y0:self.y1, self.x0:self.x1]
        self.sig = max_avg_corrected[self.y0:self.y1, self.x0:self.x1].astype(np.float64)

        self.centers = {}
        self._sigma = None


    def _center(self, frame):
        if frame in self.centers:
            return self.centers[frame]

        return (self.ax*frame + self.bx, self.ay*frame + self.by)


    def _aperture(self, frame, inner=0.0):
        """ Pixels of the frame's aperture.

        inner > 0 instead selects the segment's interior at the default width, for the profile:
        |s| <= v/2 - inner along the track (no margin, away from the frame boundaries).
        """

        cx, cy = self._center(frame)
        dx, dy = self.xx - cx, self.yy - cy
        s = dx*self.u[0] + dy*self.u[1]
        d = dx*self.n[0] + dy*self.n[1]

        if inner > 0:
            return (np.abs(d) <= self.halfwidth) & (np.abs(s) <= self.v/2.0 - inner) \
                & (self.mf == frame), d

        hw = self.halfwidth
        s_lo, s_hi = -(self.v/2.0 + self.margin), self.v/2.0 + self.margin

        # Widen to the frame's own line points: bright objects are wider than the default
        own = self.lp[self.lp[:, 2] == frame]
        if len(own):
            ps = (own[:, 0] - cx)*self.u[0] + (own[:, 1] - cy)*self.u[1]
            pd = (own[:, 0] - cx)*self.n[0] + (own[:, 1] - cy)*self.n[1]
            hw = max(hw, np.max(np.abs(pd)) + 2.0)
            s_lo, s_hi = min(s_lo, np.min(ps) - 2.0), max(s_hi, np.max(ps) + 2.0)

        return (np.abs(d) <= hw) & (s >= s_lo) & (s <= s_hi) & (self.mf == frame), d


    def intensity(self, frame, half_frame=None, center=None):
        """ Background-subtracted linear intensity of the meteor in the given frame.

        Arguments:
            frame: [int] FF frame index.

        Keyword arguments:
            half_frame: [int] Field (0 or 1) when deinterlacing, None for progressive.
            center: [tuple] (x, y) of the frame's centroid; the fitted position if None.

        Return:
            [float] Sum of max_avg_corrected over the frame's aperture.
        """

        frame = int(frame)
        if center is not None:
            self.centers[frame] = (float(center[0]), float(center[1]))
            self._sigma = None

        sel, _ = self._aperture(frame)
        if (half_frame is not None) and (self.deinterlace_order >= 0):
            sel &= (self.yy.astype(int) % 2) == ((self.deinterlace_order + half_frame) % 2)

        return float(np.sum(self.sig[sel]))


    def sigma(self):
        """ Gaussian sigma of the meteor's cross-track profile, measured on the FF.

        Second moment of the mean cross-track profile (0.5 px bins over the default aperture width,
        negative bins clipped) of the segment interiors, away from the frame boundaries where the
        split thins the trail. Uses the frames measured so far (their centroids), else every frame of
        the line points. None if the meteor is too slow to have segment interiors (v <= 4 px/frame).
        """

        if self._sigma is not None:
            return self._sigma

        if self.v <= 4.0:
            return None

        frames = sorted(self.centers) if self.centers else \
            range(int(np.min(self.lp[:, 2])), int(np.max(self.lp[:, 2])) + 1)

        ds, ws = [], []
        for frame in frames:
            sel, d = self._aperture(frame, inner=2.0)
            ds.append(d[sel])
            ws.append(self.sig[sel])

        d, w = np.concatenate(ds), np.concatenate(ws)
        if not len(d):
            return None

        edges = np.arange(-self.halfwidth, self.halfwidth + 1e-9, 0.5)
        centres = 0.5*(edges[1:] + edges[:-1])
        prof = np.zeros(len(centres))
        for k in range(len(centres)):
            in_bin = (d >= edges[k]) & (d < edges[k + 1])
            if np.any(in_bin):
                prof[k] = np.mean(w[in_bin])
        prof = np.clip(prof, 0, None)

        if prof.sum() <= 0:
            return None

        mu = np.sum(centres*prof)/prof.sum()
        self._sigma = math.sqrt(max(np.sum((centres - mu)**2*prof)/prof.sum(), 0.05))

        return self._sigma


    def splitCorrection(self, max_loss=0.25):
        """ Multiplicative correction for the light lost at the frame boundaries.

        Keyword arguments:
            max_loss: [float] Largest loss fraction the model is applied at (validated range).

        Return:
            (factor, loss): [tuple of floats] factor = 1/(1 - loss); (1.0, loss) when the loss is
                outside the validated range, (1.0, nan) when sigma cannot be measured.
        """

        sigma = self.sigma()
        if sigma is None:
            return 1.0, float('nan')

        loss = splitLossFraction(sigma, self.v)
        if not (0 <= loss <= max_loss):
            return 1.0, loss

        return 1.0/(1.0 - loss), loss
