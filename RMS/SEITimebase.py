""" SEI-derived integration-start timebase: the primary frame-time source when the camera
emits a valid RMSP provenance SEI.

Each frame's timestamp is used RAW -- capture_utc - exp - k -- which is the camera's own
ground-truth start of integration for row 0. Nothing is fitted into the returned value, so
real per-frame structure is preserved: an exposure step moves exactly the frame it applies to,
a dropped frame leaves a real gap, and a VMAX/reinit step is followed instantly.

A short sliding linear fit of capture_utc against PTS is kept for two side jobs only:
  (a) fill a frame whose SEI record is missing or corrupt (flagged as interpolated), and
  (b) monitor the residual of each raw stamp, so a step, a clock slew or a bad parse is
      detected and counted.
The fit never rewrites a good stamp. See [[reference_rmsp_provenance]] and
[[project_rms_timestamp_discipline]].
"""

from __future__ import print_function, division, absolute_import

import time
import threading
from collections import deque

# integration start = capture_utc - K_READOUT - exp. Kept equal to BufferedCapture._K_READOUT_S
# (row-0 readout offset k, measured once by PPS-LED calibration).
K_READOUT_S = 88e-6

_RESID_ANOMALY_S = 300e-6     # a raw stamp this far off the local trend = step / slew / bad parse
_STALE_S = 2.0                # no fresh SEI for this long -> timebase not usable
_MIN_SPAN_S = 1.0             # need at least this much time base before the fit is trusted
_DISCONT_GAP_S = 2.0          # PTS gap larger than this -> reset the fit (reconnect / sensor reinit)
_MAX_ANOM_RATE = 0.10         # recent fraction of anomalous residuals above which we stand down
_MAX_OFFSET_S = 2.0           # |median(SEI - legacy)| beyond this -> camera clock is wrong
                             # (e.g. chrony not synced); the internal fit stays perfect, so
                             # legacy wallclock is the ONLY thing that can catch it -> gate on it
_MIN_OFFSET_SAMPLES = 10     # need this many SEI-vs-legacy samples before trusting the gate


class SEITimebase(object):
    """ Thread-safe: fed from the GStreamer streaming thread (feed), queried from the capture
        thread (estimate / ready / health). Present-frame stamps do NOT go through here -- the
        caller stores feed()'s return in a plain pts-keyed dict and pops it lock-free, exactly
        like the arrival-time probe. Only the rare miss path (estimate) and the per-block
        readiness check take the lock. """

    def __init__(self, fps, window_s=20.0):
        self._lock = threading.Lock()
        self._fps = float(fps) if fps else 25.0
        maxlen = max(4, int(window_s*self._fps))
        self._pts = deque(maxlen=maxlen)          # PTS [s], referenced per-fit for numerical stability
        self._utc = deque(maxlen=maxlen)          # capture_utc [s]
        self._last_pts_s = None
        self._last_exp_s = None
        self._last_feed_wall = None
        self._n_fed = 0
        self._anom = deque(maxlen=200)            # recent residual-anomaly flags (1/0)
        self._last_resid_s = 0.0
        self._off = deque(maxlen=200)             # recent SEI-minus-legacy offsets [s] (from the caller)

    # -- fit helpers (call with the lock held) --
    def _fit(self):
        n = len(self._pts)
        if n < 2:
            return None
        x0 = self._pts[0]
        sx = sy = sxx = sxy = 0.0
        for xp, yp in zip(self._pts, self._utc):
            x = xp - x0
            sx += x; sy += yp; sxx += x*x; sxy += x*yp
        d = n*sxx - sx*sx
        if d == 0.0:
            return None
        b = (n*sxy - sx*sy)/d
        a = (sy - b*sx)/n
        return (x0, a, b)

    def _eval(self, pts_s):
        f = self._fit()
        if f is None:
            return None
        x0, a, b = f
        return a + b*(pts_s - x0)

    def _int_start(self, cap_utc, exp_s):
        e = exp_s if exp_s is not None else (self._last_exp_s if self._last_exp_s is not None else 0.0)
        return cap_utc - K_READOUT_S - e

    # -- streaming thread --
    def feed(self, pts_ns, cap_utc, exp_s, frame_seq=None):
        """ Add one frame's SEI, update the fit and the residual monitor, and return this
            frame's RAW integration-start timestamp (capture_utc - exp - k). """
        pts_s = pts_ns/1e9
        with self._lock:
            # Discontinuity (reconnect, PTS wrap, sensor reinit): drop the stale fit.
            if self._last_pts_s is not None and \
               (pts_s < self._last_pts_s or (pts_s - self._last_pts_s) > _DISCONT_GAP_S):
                self._pts.clear(); self._utc.clear(); self._anom.clear()
            # Residual of the RAW stamp vs the local trend, BEFORE adding this point.
            pred = self._eval(pts_s)
            if pred is not None:
                self._last_resid_s = cap_utc - pred
                self._anom.append(1 if abs(self._last_resid_s) > _RESID_ANOMALY_S else 0)
            self._pts.append(pts_s); self._utc.append(cap_utc)
            self._last_pts_s = pts_s
            if exp_s is not None:
                self._last_exp_s = exp_s
            self._last_feed_wall = time.time()
            self._n_fed += 1
            return self._int_start(cap_utc, exp_s)

    # -- capture thread --
    def note_offset(self, sei_minus_legacy_s):
        """ Record SEI-minus-legacy for this frame. Legacy wallclock is anchored to the host
            clock, so a large sustained offset means the camera's own clock is wrong (unsynced
            chrony) even when the SEI is internally perfect -- the gate for that lives here. """
        with self._lock:
            self._off.append(sei_minus_legacy_s)

    def _median_off(self):
        if not self._off:
            return None
        o = sorted(self._off); m = len(o)//2
        return o[m] if len(o) % 2 else 0.5*(o[m - 1] + o[m])

    def estimate(self, pts_ns):
        """ Fit-based integration-start for a frame whose SEI record is missing/corrupt, or
            None if the fit is not usable. Flagged as interpolated by the caller. """
        pts_s = pts_ns/1e9
        with self._lock:
            utc = self._eval(pts_s)
            if utc is None:
                return None
            return self._int_start(utc, None)

    def ready(self):
        """ True when the SEI timebase is trustworthy as the primary source: fresh, spanning
            enough time, and with a low recent residual-anomaly rate. """
        with self._lock:
            if self._last_feed_wall is None or (time.time() - self._last_feed_wall) > _STALE_S:
                return False
            if len(self._pts) < 2 or (self._pts[-1] - self._pts[0]) < _MIN_SPAN_S:
                return False
            if self._anom and (sum(self._anom)/len(self._anom)) > _MAX_ANOM_RATE:
                return False
            # Absolute-time gate: SEI must agree with the host-anchored legacy clock. This is
            # what stops an unsynced camera (SEI hours off but internally consistent) from
            # being trusted as the primary timebase.
            if len(self._off) < _MIN_OFFSET_SAMPLES:
                return False
            med = self._median_off()
            if med is None or abs(med) > _MAX_OFFSET_S:
                return False
            return True

    def health(self):
        with self._lock:
            span = (self._pts[-1] - self._pts[0]) if len(self._pts) >= 2 else 0.0
            rate = (sum(self._anom)/len(self._anom)) if self._anom else 0.0
            stale = None if self._last_feed_wall is None else (time.time() - self._last_feed_wall)
            med = self._median_off()
            return {'n_fed': self._n_fed, 'span_s': round(span, 2),
                    'anom_rate': round(rate, 3), 'last_resid_us': round(self._last_resid_s*1e6, 1),
                    'off_med_ms': None if med is None else round(med*1e3, 1),
                    'n_off': len(self._off),
                    'stale_s': None if stale is None else round(stale, 2)}
