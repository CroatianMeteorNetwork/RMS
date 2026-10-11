""" Per-FF-block photometric provenance from the RMSP camera SEI.

An FF file is a 256-frame summary and photometry assumes the block is stationary, so besides
block means this records the within-block exposure extremes and a stability flag (exposure and
all gains constant), plus the encoder's mean/max QP as a codec-quality indicator (under the
fixed-QP-with-ceiling rate control, QP near the floor = clean, toward the cap = bitrate-ceiling
limited, i.e. faint-star flux quantised). Also: the per-gain extremes, how many frames carried
frame-exact gains (the rest were read from the ISP at encode time, which can lag an exposure
change), the white balance gains (colour-term provenance), the sensor line period and lines per
frame (rolling-shutter row timing) and the camera frame sequence range with the number of camera
frames missing from it (frame_seq gaps: lost between the encoder and RMS). Accumulated from each
frame's record as BufferedCapture's block loop reads the frame, snapshotted at every block
boundary into a lock-free shared array, and read by the Compressor when the block is ready.
A zero frame count means no SEI was seen (e.g. XM cameras) and no cards are written.
"""

from __future__ import print_function, division, absolute_import

import threading


# Field order of the shared array. 'seq' (last) is a block sequence number written after the
# fields so a reader can detect a torn read; 'nfrm' == 0 means "no data".
SEI_META_FIELDS = ('nfrm', 'exp_mean', 'exp_min', 'exp_max', 'again', 'dgain', 'ispdgain',
                   'stable', 'qp_mean', 'qp_max', 'wb_r', 'wb_b',
                   'again_min', 'again_max', 'dgain_min', 'dgain_max', 'ispdgain_min', 'ispdgain_max',
                   'gains_exact_n', 'wb_g', 't_row_ns', 'vmax',
                   # camera frame_seq of the block's first/last record (-1 = none) and the frames
                   # missing before/between them
                   'seq_first', 'seq_last', 'seq_lost',
                   # timing provenance (set by the capture loop, not the accumulator):
                   # time_src 1=SEI integration-start / 0=legacy; time_off_ms = SEI-minus-legacy
                   # block median [ms]; interp_n = frames with interpolated SEI time
                   'time_src', 'time_off_ms', 'interp_n', 'seq')
SEI_META_N = len(SEI_META_FIELDS)
_SEQ = SEI_META_N - 1


class SEIBlockAccumulator(object):
    """ Thread-safe per-block accumulator fed from the GStreamer streaming thread. """

    def __init__(self):
        self._lock = threading.Lock()
        # Previous record's frame_seq, kept across blocks so a gap at a block boundary counts too
        self._seq_prev = None
        self._reset()

    def _reset(self):
        self.n = 0
        self.exp_sum = 0.0; self.exp_min = None; self.exp_max = None
        self.ag_sum = 0.0; self.ag_min = None; self.ag_max = None
        self.dg_sum = 0.0; self.dg_min = None; self.dg_max = None
        self.ig_sum = 0.0; self.ig_min = None; self.ig_max = None
        self.qp_sum = 0.0; self.qp_n = 0; self.qp_max = 0
        self.exact_n = 0
        self.wb_r = None; self.wb_b = None; self.wb_g = None
        self.t_row_ns = None; self.vmax = None
        self.seq_first = None; self.seq_last = None; self.seq_lost = 0

    def add(self, meta):
        """ meta: [dict] from the SEI parser: exp_s [s], again/dgain/ispdgain (x, linear),
            gains_exact, qp, wb_r/wb_b/wb_g, t_row_ns, vmax, frame_seq; any value may be None
            when the SEI flags it invalid. """
        if not meta:
            return
        with self._lock:
            e = meta.get('exp_s')
            if e is not None:
                self.n += 1
                self.exp_sum += e
                self.exp_min = e if self.exp_min is None else min(self.exp_min, e)
                self.exp_max = e if self.exp_max is None else max(self.exp_max, e)
                for key, a in (('again', 'ag'), ('dgain', 'dg'), ('ispdgain', 'ig')):
                    v = meta.get(key)
                    if v is None:
                        continue
                    setattr(self, a + '_sum', getattr(self, a + '_sum') + v)
                    mn = getattr(self, a + '_min'); mx = getattr(self, a + '_max')
                    setattr(self, a + '_min', v if mn is None else min(mn, v))
                    setattr(self, a + '_max', v if mx is None else max(mx, v))
                if meta.get('gains_exact'):
                    self.exact_n += 1
            q = meta.get('qp')
            if q:
                self.qp_sum += q; self.qp_n += 1
                if q > self.qp_max:
                    self.qp_max = q
            if meta.get('wb_r') is not None:
                self.wb_r = meta['wb_r']
            if meta.get('wb_b') is not None:
                self.wb_b = meta['wb_b']
            if meta.get('wb_g') is not None:
                self.wb_g = meta['wb_g']
            if meta.get('t_row_ns') is not None:
                self.t_row_ns = meta['t_row_ns']
            if meta.get('vmax') is not None:
                self.vmax = meta['vmax']
            s = meta.get('frame_seq')
            if s is not None:
                if self.seq_first is None:
                    self.seq_first = s
                # A lower number is a venc restart (counter from 0): no gap is known there
                if (self._seq_prev is not None) and (s > self._seq_prev):
                    self.seq_lost += s - self._seq_prev - 1
                self.seq_last = s
                self._seq_prev = s

    def snapshotAndReset(self, seq):
        """ Return this block's values (list, SEI_META_FIELDS order) and start a new block. """
        with self._lock:
            n = self.n
            vals = [0.0]*SEI_META_N
            if n > 0:
                stable = (self.exp_min == self.exp_max and self.ag_min == self.ag_max
                          and self.dg_min == self.dg_max and self.ig_min == self.ig_max)
                vals = [float(n), self.exp_sum/n, self.exp_min, self.exp_max,
                        self.ag_sum/n if self.ag_min is not None else 0.0,
                        self.dg_sum/n if self.dg_min is not None else 0.0,
                        self.ig_sum/n if self.ig_min is not None else 0.0,
                        1.0 if stable else 0.0,
                        (self.qp_sum/self.qp_n) if self.qp_n else 0.0, float(self.qp_max),
                        self.wb_r if self.wb_r is not None else 0.0,
                        self.wb_b if self.wb_b is not None else 0.0]
                for a in ('ag', 'dg', 'ig'):
                    mn = getattr(self, a + '_min')
                    vals += [mn, getattr(self, a + '_max')] if mn is not None else [0.0, 0.0]
                vals += [float(self.exact_n),
                         self.wb_g if self.wb_g is not None else 0.0,
                         float(self.t_row_ns) if self.t_row_ns is not None else 0.0,
                         float(self.vmax) if self.vmax is not None else 0.0,
                         float(self.seq_first) if self.seq_first is not None else -1.0,
                         float(self.seq_last) if self.seq_last is not None else -1.0,
                         float(self.seq_lost),
                         0.0, 0.0, 0.0, 0.0]   # time_src, time_off_ms, interp_n, seq (set later)
            vals[_SEQ] = float(seq)
            self._reset()
        return vals


def publishSeiMeta(shared, vals):
    """ Writer side: fields first, sequence number last. No-op when the array is absent. """
    if shared is None:
        return
    for i in range(SEI_META_N - 1):
        shared[i] = vals[i]
    shared[_SEQ] = vals[_SEQ]


def readSeiMeta(shared):
    """ Reader side: dict keyed by SEI_META_FIELDS, or None when absent, empty (no SEI frames)
        or if the read never stabilises. """
    if shared is None:
        return None
    for _ in range(3):
        s1 = shared[_SEQ]
        vals = [shared[i] for i in range(SEI_META_N)]
        if shared[_SEQ] == s1 and vals[_SEQ] == s1:
            if vals[0] <= 0:
                return None
            return dict(zip(SEI_META_FIELDS, vals))
    return None
