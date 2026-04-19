# RPi Meteor Station
# Copyright (C) 2015  Dario Zubovic
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function, division, absolute_import

import gc
import os
import sys
import ctypes
import traceback

import re
import time
import datetime
import copy
import os.path
from multiprocessing import Process, Event, Value, Array
import threading
from collections import deque
import os
import signal

import cv2
import numpy as np
import socket
import errno
import json
import logging


from RMS.Misc import obfuscatePassword
from RMS.Routines.GstreamerCapture import GstVideoFile, getStructureValue
from RMS.Routines.VencCalibration import calibrate_epoch_offset
from RMS.Formats.ObservationSummary import getObsDBConn, addObsParam
from RMS.RawFrameSave import RawFrameSaver
from RMS.Misc import RmsDateTime, mkdirP, UTCFromTimestamp
from RMS.Formats import FTfile, FTStruct
from RMS.Logger import LoggingManager, getLogger, gstDebugLogger
from RMS.CaptureModeSwitcher import switchCameraMode
import Utils.CameraControl as cc

# Get the logger from the main module
log = getLogger("rmslogger")

if sys.version_info[0] < 3:
    # py2
    from urlparse import urlparse
else:
    # py3
    from urllib.parse import urlparse


GST_IMPORTED = False
try:
    import gi
    gi.require_version('Gst', '1.0')
    gi.require_version('GstRtp', '1.0')
    from gi.repository import Gst, GstRtp
    GST_IMPORTED = True

except ImportError as e:
    log.info('Could not import gi: {}. Using OpenCV.'.format(e))

except ValueError as e:
    log.info('Could not import Gst: {}. Using OpenCV.'.format(e))


# Define probe result constants
class RtspProbeResult:
    """
    Constants representing possible RTSP probe results.
    
    SUCCESS: Connection successful
    NETWORK_DOWN: Local network interface is down or unreachable
    HOST_UNREACHABLE: Network up but target host cannot be reached
    CONNECTION_REFUSED: Host is up but actively refusing RTSP connections
    TIMEOUT: Connection attempt exceeded specified timeout
    DNS_ERROR: Unable to resolve hostname to IP address
    UNKNOWN_ERROR: Other unspecified connection errors
    """
    SUCCESS = "SUCCESS"
    NETWORK_DOWN = "NETWORK_DOWN"          # No network connectivity
    HOST_UNREACHABLE = "HOST_UNREACHABLE"  # Can't reach the host  
    CONNECTION_REFUSED = "CONNECTION_REFUSED" # Host reachable but RTSP port closed
    TIMEOUT = "TIMEOUT"                    # Connection attempt timed out
    DNS_ERROR = "DNS_ERROR"                # Can't resolve hostname
    UNKNOWN_ERROR = "UNKNOWN_ERROR"        # Other connection errors


class VencMetadataReader(object):
    """Background thread that reads per-frame metadata from the camera's
    metadata stream (port 9602).  Stores the latest values for association
    with captured frames.  No encoder ioctls — pure TCP, ~25Hz push.

    Format v3 (8 fields): pts_90k raw_pts_us frame_seq exp_us again dgain ispdgain temp
    Format v2 (7 fields): pts_90k raw_pts_us exp_us again dgain ispdgain temp
    Format v1 (6 fields): pts_90k exp_us again dgain ispdgain temp

    In v3 (ring buffer mode), there is exactly one metadata line per video
    frame — true 1:1 correspondence.  frame_seq is a monotonic counter
    (0 means polled fallback, no ring buffer).

    The raw_pts_us field gives 1µs VENC register precision for FT science
    timestamps, bypassing the 11.1µs quantization of the 90kHz RTP pipeline.
    A rolling lookup table maps pts_90k → raw_pts_us so video frames (which
    carry 90kHz PTS) can be matched to their µs-precise VENC timestamp."""

    # Keep last N entries in the pts lookup table (~2 min at 25fps)
    PTS_TABLE_SIZE = 3000

    def __init__(self, camera_ip, port=9602):
        self.camera_ip = camera_ip
        self.port = port
        self._latest = {}
        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        # Rolling lookup: pts_90k → raw_pts_us (deque for bounded size)
        self._pts_table = deque(maxlen=self.PTS_TABLE_SIZE)
        # Wrap tracking for raw_pts_us (32-bit µs wraps every ~71.6 min)
        self._pts_us_prev = 0
        self._pts_us_wraps = 0

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    @property
    def latest(self):
        with self._lock:
            return dict(self._latest)

    def lookup_pts_us(self, rtp_ts_32):
        """Look up raw µs PTS for a 32-bit RTP timestamp.

        Matches against the low 32 bits of the daemon's wrap-corrected PTS.
        Tolerance of ±20 ticks handles small GUARD residual shift.
        Returns (raw_pts_us_unwrapped, True) if found, (None, False) if not."""
        target = rtp_ts_32 & 0xFFFFFFFF
        with self._lock:
            best_diff = 20  # ticks — well within one frame period (3602)
            best_us = None
            for p90k, p_us_unwrapped in reversed(self._pts_table):
                p32 = p90k & 0xFFFFFFFF
                diff = abs(p32 - target)
                if diff > 0x7FFFFFFF:  # 32-bit wrap
                    diff = 0x100000000 - diff
                if diff < best_diff:
                    best_diff = diff
                    best_us = p_us_unwrapped
                    if diff == 0:
                        break
        return (best_us, True) if best_us is not None else (None, False)

    def _run(self):
        log = logging.getLogger("logger")
        buf = b''
        while not self._stop.is_set():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(5)
                s.connect((self.camera_ip, self.port))
                log.info("VencMetadata: connected to %s:%d", self.camera_ip, self.port)
                buf = b''
                while not self._stop.is_set():
                    s.settimeout(2)
                    try:
                        data = s.recv(512)
                    except socket.timeout:
                        continue
                    if not data:
                        break
                    buf += data
                    while b'\n' in buf:
                        line, buf = buf.split(b'\n', 1)
                        parts = line.decode(errors='replace').strip().split()
                        if len(parts) >= 8:
                            # v3: pts_90k raw_pts_us frame_seq exp again dgain ispdgain temp
                            pts_90k = int(parts[0])
                            raw_pts_us = int(parts[1])
                            frame_seq = int(parts[2])
                            # Track 32-bit µs wraps (~71.6 min period)
                            if raw_pts_us < self._pts_us_prev - 2000000000:
                                self._pts_us_wraps += 1
                            self._pts_us_prev = raw_pts_us
                            pts_us_unwrapped = raw_pts_us + self._pts_us_wraps * 4294967296
                            with self._lock:
                                self._pts_table.append((pts_90k, pts_us_unwrapped))
                                self._latest = {
                                    'pts_90k': pts_90k,
                                    'raw_pts_us': pts_us_unwrapped,
                                    'frame_seq': frame_seq,
                                    'exposure_us': int(parts[3]),
                                    'analog_gain': int(parts[4]) / 1024.0,
                                    'digital_gain': int(parts[5]) / 1024.0,
                                    'isp_dgain': int(parts[6]) / 1024.0,
                                    'soc_temp_c': int(parts[7]),
                                }
                        elif len(parts) >= 7:
                            # v2: pts_90k raw_pts_us exp again dgain ispdgain temp
                            pts_90k = int(parts[0])
                            raw_pts_us = int(parts[1])
                            if raw_pts_us < self._pts_us_prev - 2000000000:
                                self._pts_us_wraps += 1
                            self._pts_us_prev = raw_pts_us
                            pts_us_unwrapped = raw_pts_us + self._pts_us_wraps * 4294967296
                            with self._lock:
                                self._pts_table.append((pts_90k, pts_us_unwrapped))
                                self._latest = {
                                    'pts_90k': pts_90k,
                                    'raw_pts_us': pts_us_unwrapped,
                                    'exposure_us': int(parts[2]),
                                    'analog_gain': int(parts[3]) / 1024.0,
                                    'digital_gain': int(parts[4]) / 1024.0,
                                    'isp_dgain': int(parts[5]) / 1024.0,
                                    'soc_temp_c': int(parts[6]),
                                }
                        elif len(parts) >= 6:
                            # v1 format (backward compat): pts_90k exp again dgain ispdgain temp
                            with self._lock:
                                self._latest = {
                                    'pts_90k': int(parts[0]),
                                    'exposure_us': int(parts[1]),
                                    'analog_gain': int(parts[2]) / 1024.0,
                                    'digital_gain': int(parts[3]) / 1024.0,
                                    'isp_dgain': int(parts[4]) / 1024.0,
                                    'soc_temp_c': int(parts[5]),
                                }
                s.close()
            except Exception as e:
                if not self._stop.is_set():
                    log.debug("VencMetadata: %s, reconnecting in 5s", e)
                    self._stop.wait(5)


class PtsStreamReader(object):
    """Background thread reading fresh VENC PTS from pts_stream (port 9603).

    pts_stream reads the VENC PTS register via /dev/mem at each register
    transition — guaranteed fresh, no stale race.

    Frame association uses block-level jitter fingerprinting: after a 256-
    frame block is captured, call align_block() with the block's RTP
    timestamps.  Cross-correlation of the delta patterns finds the alignment,
    and every frame gets a fresh UTC timestamp via a linear clock model.

    Clock model: linear regression of (venc_pts_us, host_utc) pairs from
    periodic VencCalibration probes (gettime port 9601).  This references
    all timestamps to the HOST's GPS/PPS-disciplined clock, bypassing
    camera NTP entirely.  Probes run every PROBE_INTERVAL seconds in a
    background thread.
    """

    TABLE_SIZE = 12000   # ~8 min at 25fps
    MATCH_TOL_US = 50    # per-delta correlation tolerance
    PROBE_INTERVAL = 120 # seconds between VencCalibration probes

    def __init__(self, camera_ip, port=9603, gettime_port=9601, drift_freq=None,
                 drift_freq_path=None):
        self.camera_ip = camera_ip
        self.port = port
        self.gettime_port = gettime_port
        self._seed_drift = drift_freq if drift_freq else 0.0
        self._drift_freq_path = drift_freq_path
        self._lock = threading.Lock()
        self._thread = None
        self._probe_thread = None
        self._stop = threading.Event()
        # Ordered entries: (venc_pts_us_unwrapped,)
        self._entries = deque(maxlen=self.TABLE_SIZE)
        self._deltas = deque(maxlen=self.TABLE_SIZE)
        self._pts_prev = 0
        self._wraps = 0
        # Clock PLL: C = offset from VENC µs to UTC, drift_rate = dC/dt
        # UTC = (C_last + drift_rate * elapsed) + venc_us * 1e-6
        # C is in the unwrapped domain (compensated for 32-bit PTS wraps)
        self._C = None           # last measured C (unwrapped domain)
        self._C_raw = None       # fresh C from last probe (32-bit domain)
        self._drift_rate = 0.0   # dC/dt (seconds per second, ~ppm)
        self._C_time = 0.0       # host time when C was last measured
        # Stats
        self.block_aligns = 0
        self.block_fails = 0
        self.drops_detected = 0
        self.probe_count = 0

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._probe_thread = threading.Thread(target=self._run_probes, daemon=True)
        self._probe_thread.start()

    def stop(self):
        self._stop.set()
        sock = getattr(self, '_socket', None)
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=3)
        if self._probe_thread:
            self._probe_thread.join(timeout=3)

    def restart(self):
        """Full stop + fresh start. Clears all stale state."""
        log = logging.getLogger("logger")
        log.info("PtsStream: restarting (full state reset)")
        self.stop()
        # Reset all state
        self._entries.clear()
        self._deltas.clear()
        self._C_raw = None
        self._C = None
        self._C_time = 0.0
        self._pts_prev = 0
        self._wraps = 0
        self._force_reconnect = False
        self._force_probe = False
        if hasattr(self, '_rt_base_offset'):
            del self._rt_base_offset
        self.start()

    def align_block(self, rtp_90k_list, block_start_time=None):
        """Match each frame to a pts_stream entry by PTS value.

        Simple, proven matching.  No phase correction, no residual
        classification — those are deferred to offline reanalysis.
        Saves raw data (K indices, ref_pts) for offline use.

        Stale frames (monotonicity filter delta≤2) get the previous
        frame's timestamp.

        Args:
            rtp_90k_list: list of RTP timestamps (90kHz, may contain None
                for firmware-anomalous frames flagged by GUARD).
            block_start_time: unused (reserved for offline reanalysis).

        Returns:
            (utc_list, raw_pts_list, K_list):
                utc_list: UTC timestamps per frame (None if unmatched).
                raw_pts_list: matched ref_pts µs per frame (None if unmatched).
                K_list: matched entry index per frame (None if unmatched).
        """
        import bisect
        log = logging.getLogger("logger")
        n = len(rtp_90k_list)

        with self._lock:
            ref_pts = [e[0] for e in self._entries]
            ref_camwall = [e[2] if len(e) > 2 else None for e in self._entries]
            C = self._C
            drift_rate = self._drift_rate
            C_time = self._C_time

        if len(ref_pts) < 50 or C is None:
            self.block_fails += 1
            return [None] * n, [None] * n, [None] * n

        import time as _time
        C_now = C + drift_rate * (_time.time() - C_time)

        # PTS-value match every frame
        K = [None] * n
        result = [None] * n
        raw_pts_out = [None] * n
        matched = 0
        n_stale = 0
        last_idx = -1

        for i in range(n):
            # Firmware anomaly (flagged by GUARD): skip
            if rtp_90k_list[i] is None:
                continue

            # Stale detection (monotonicity filter: delta≤2)
            if i > 0 and rtp_90k_list[i - 1] is not None:
                delta = (int(rtp_90k_list[i]) - int(rtp_90k_list[i - 1])) \
                        & 0xFFFFFFFF
                if delta <= 2:
                    n_stale += 1
                    K[i] = K[i - 1]
                    if i > 0 and result[i - 1] is not None:
                        result[i] = result[i - 1]
                        raw_pts_out[i] = raw_pts_out[i - 1]
                        matched += 1
                    continue

            # PTS-value match with multi-wrap unwrap
            rtp_us = (int(rtp_90k_list[i]) * 100) // 9
            if last_idx >= 0:
                ref_base = ref_pts[last_idx]
            else:
                ref_base = ref_pts[len(ref_pts) // 2]
            diff = rtp_us - ref_base
            n_wraps = round(diff / 4294967296)
            rtp_uw = rtp_us - n_wraps * 4294967296

            lo = max(0, last_idx)
            idx = bisect.bisect_left(ref_pts, rtp_uw, lo=lo)
            best_i, best_d = None, 20000
            for c in (idx - 1, idx, idx + 1):
                if 0 <= c < len(ref_pts) and c > last_idx:
                    d = abs(ref_pts[c] - rtp_uw)
                    if d < best_d:
                        best_d, best_i = d, c

            if best_i is not None:
                K[i] = best_i
                last_idx = best_i
                # Prefer cam_wall (camera chrony-disciplined UTC at FE_START,
                # computed camera-side via GetCurPTS anchor — pipeline-immune).
                # Fall back to C_now + venc_us (host-side VencCalibration,
                # subject to network-roundtrip estimation error).
                cw = ref_camwall[best_i]
                if cw is not None and cw > 0:
                    result[i] = cw
                else:
                    result[i] = C_now + ref_pts[best_i] * 1e-6
                raw_pts_out[i] = float(ref_pts[best_i])
                matched += 1

        # Fill unmatched frames by interpolating from nearest matched
        # neighbor using exact frame period (40ms after VMAX=1350 fix).
        # Two sources of unmatched frames:
        #   1. First ~4 frames of a session: RTP arrived before pts_stream
        #      received any entries (back-extrapolate from earliest match).
        #   2. Periodic pts_stream skips (camera emits ~230 entries per 256
        #      frames due to firmware FE_START misses every ~2s).
        # Drift error over a 4-frame interpolation is <4µs at 100ppm.
        interpolated = 0
        if matched > 0:
            matched_i = [i for i in range(n) if result[i] is not None]
            for i in range(n):
                if result[i] is not None:
                    continue
                # Nearest matched neighbor (by frame index)
                j = min(matched_i, key=lambda m: abs(m - i))
                result[i] = result[j] + (i - j) * self._FRAME_PERIOD
                interpolated += 1

        if matched > 0:
            self.block_aligns += 1
        else:
            self.block_fails += 1
        log.info("PtsStream: %d/%d matched (%d stale, %d interpolated)",
                 matched, n, n_stale, interpolated)

        return result, raw_pts_out, K

    # Pipeline delay: VENC PTS register captures the currently-reading
    # frame, D frames ahead of the frame being encoded.
    # Calibrated from GPS PPS LED (2026-04-04).
    _PIPELINE_D = 0
    _FRAME_PERIOD = 4400 * 1350 / 148.5e6  # 0.040000000 s (VMAX bug fixed)
    _WRAP_US = 4294967296

    def lookupFrameUtc(self, rtp_90k, guard_shift=0, exposure_us=0.0):
        """Real-time per-frame UTC from PTS value matching.

        Matches the frame's RTP PTS value to the closest pts_stream
        entry via binary search, then computes UTC from the latest
        C_raw probe with pipeline delay and exposure correction.

        Args:
            rtp_90k: raw RTP timestamp in 90kHz ticks (from pad probe).
            guard_shift: cumulative GUARD shift to undo firmware anomalies.
            exposure_us: sensor exposure time in microseconds.

        Returns:
            (utc, raw_pts_us) or (None, None) if no match.
        """
        import bisect

        with self._lock:
            if not self._entries or self._C_raw is None:
                return None, None
            ref_pts = [e[0] for e in self._entries]
            c_raw = self._C_raw

        # Convert RTP 90kHz to µs (undo guard shift first)
        rtp_us = ((int(rtp_90k) + guard_shift) * 100) // 9

        # Bootstrap base_offset on first call.
        # base_offset = rtp_us - ref_pts[match] (full domain, not unwrapped).
        if not hasattr(self, '_rt_base_offset'):
            mid = ref_pts[len(ref_pts) // 2]
            diff = rtp_us - mid
            n_wraps = round(diff / self._WRAP_US)
            rtp_uw = rtp_us - n_wraps * self._WRAP_US
            idx = bisect.bisect_left(ref_pts, rtp_uw)
            best_c, best_d = -1, 1e18
            for c in (idx - 1, idx, idx + 1):
                if 0 <= c < len(ref_pts):
                    d = abs(ref_pts[c] - rtp_uw)
                    if d < best_d:
                        best_d, best_c = d, c
            if best_c >= 0 and best_d < 20000:
                self._rt_base_offset = rtp_us - ref_pts[best_c]
            else:
                return None, None

        # Binary search
        target = rtp_us - self._rt_base_offset
        # Handle wrap: target may need wrap adjustment
        diff_from_last = target - ref_pts[-1]
        if diff_from_last > self._WRAP_US // 2:
            target -= self._WRAP_US
        elif diff_from_last < -self._WRAP_US // 2:
            target += self._WRAP_US

        idx = bisect.bisect_left(ref_pts, target)
        best_i, best_d = -1, 20000
        for c in (idx - 1, idx, idx + 1):
            if 0 <= c < len(ref_pts):
                d = abs(ref_pts[c] - target)
                if d < best_d:
                    best_d, best_i = d, c

        if best_i < 0:
            return None, None

        matched_pts = ref_pts[best_i]

        # Prefer camera wallclock (chrony-disciplined) if available
        with self._lock:
            entry = self._entries[best_i] if best_i < len(self._entries) else None
        cam_wall = entry[2] if entry and len(entry) > 2 else None

        if cam_wall is not None and cam_wall > 0:
            # cam_wall was read at FE_START + 15ms; subtract to get capture time
            utc = cam_wall
            utc -= exposure_us * 1e-6
        else:
            # Fallback: C_raw + PTS conversion (old path)
            raw_pts_mod = matched_pts % self._WRAP_US
            utc = c_raw + raw_pts_mod * 1e-6
            utc -= self._PIPELINE_D * self._FRAME_PERIOD
            utc -= exposure_us * 1e-6

        return utc, float(matched_pts)

    def _run(self):
        """Read pts_stream entries — VENC PTS + camera wallclock (if available)."""
        log = logging.getLogger("logger")
        buf = b''
        while not self._stop.is_set():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(5)
                s.connect((self.camera_ip, self.port))
                self._socket = s  # expose for forced close
                log.info("PtsStream: connected to %s:%d", self.camera_ip, self.port)
                # Clear stale state from previous session (camera may
                # have rebooted — PTS epoch, C_raw all invalid now)
                with self._lock:
                    self._entries.clear()
                    self._deltas.clear()
                    self._C_raw = None
                    self._C = None
                self._pts_prev = 0
                self._wraps = 0
                self._force_reconnect = False
                # Force immediate re-probe of C_raw
                self._force_probe = True
                buf = b''
                while not self._stop.is_set():
                    if getattr(self, '_force_reconnect', False):
                        self._force_reconnect = False
                        log.info("PtsStream: forced reconnect (camera may have rebooted)")
                        break  # close socket → reconnect with fresh state
                    s.settimeout(2)
                    try:
                        data = s.recv(512)
                    except socket.timeout:
                        continue
                    if not data:
                        break
                    buf += data
                    while b'\n' in buf:
                        line, buf = buf.split(b'\n', 1)
                        parts = line.decode(errors='replace').strip().split()
                        if len(parts) >= 2:
                            raw_us = int(parts[1])
                            if raw_us < self._pts_prev - 2000000000:
                                self._wraps += 1
                            self._pts_prev = raw_us
                            venc_us = raw_us + self._wraps * 4294967296
                            # Use camera wallclock if available (chrony-disciplined),
                            # fall back to host time.time() otherwise
                            if len(parts) >= 4:
                                cam_wall = int(parts[2]) + int(parts[3]) / 1e6
                            else:
                                cam_wall = None
                            arrival = time.time()
                            with self._lock:
                                if self._entries:
                                    self._deltas.append(venc_us - self._entries[-1][0])
                                if len(self._entries) == self._entries.maxlen:
                                    pass  # deque auto-evicts
                                self._entries.append((venc_us, arrival, cam_wall))
                s.close()
            except Exception as e:
                if not self._stop.is_set():
                    log.debug("PtsStream: %s, reconnecting in 5s", e)
                    self._stop.wait(5)

    def _run_probes(self):
        """PLL clock discipline via periodic VencCalibration probes.

        Measures C (VENC-to-UTC offset) against host GPS clock every
        PROBE_INTERVAL seconds.  Tracks drift_rate = dC/dt from consecutive
        measurements.  Between probes, interpolates smoothly:
            C_now = C_last + drift_rate * elapsed
            UTC = C_now + venc_us * 1e-6

        C_raw from calibrate_epoch_offset is in the 32-bit PTS domain
        and jumps at wraps.  We compensate using _wraps from the
        pts_stream reader (same register, same wraps).
        """
        from RMS.Routines.VencCalibration import calibrate_epoch_offset
        log = logging.getLogger("logger")
        import time as _time

        WRAP_STEP = 4294967296e-6  # 2^32 µs in seconds

        # Initial calibration
        C_raw = calibrate_epoch_offset(self.camera_ip, self.gettime_port, n_samples=20)
        if C_raw is not None:
            with self._lock:
                self._C_raw = C_raw  # fresh measurement, no wraps
                self._C = C_raw - self._wraps * WRAP_STEP
                self._drift_rate = self._seed_drift
                self._C_time = _time.time()
            self.probe_count = 1
            log.info("PtsStream: initial C=%.6f (raw=%.6f, wraps=%d), "
                     "seeded drift=%.1f ppm",
                     self._C, C_raw, self._wraps, self._drift_rate * 1e6)
        else:
            log.warning("PtsStream: initial calibration failed, retrying...")
            # Don't return — fall through to the steady-state loop which
            # will keep probing until it succeeds.

        # Steady-state PLL: measure C, update drift_rate
        self._force_probe = False
        while not self._stop.is_set():
            # Wait for next probe interval, or wake early on force
            for _ in range(self.PROBE_INTERVAL):
                if self._stop.is_set():
                    return
                if getattr(self, '_force_probe', False):
                    self._force_probe = False
                    log.info("PtsStream: forced re-probe (reconnection)")
                    break
                self._stop.wait(1)

            C_raw = calibrate_epoch_offset(self.camera_ip, self.gettime_port,
                                           n_samples=10)
            if C_raw is None:
                continue

            self.probe_count += 1
            t_now = _time.time()

            with self._lock:
                self._C_raw = C_raw  # fresh measurement, no wraps
                C_new = C_raw - self._wraps * WRAP_STEP
                C_old = self._C
                dt = t_now - self._C_time if self._C_time > 0 else 0
                if C_old is None:
                    # First successful probe (initial failed)
                    C_predicted = C_new
                    pred_err_ms = 0
                    self._drift_rate = self._seed_drift
                    log.info("PtsStream: initial C=%.6f (delayed)", C_new)
                else:
                    C_predicted = C_old + self._drift_rate * dt
                    pred_err_ms = (C_new - C_predicted) * 1000

                if dt > 60.0:
                    new_rate = (C_new - C_old) / dt
                    if abs(new_rate) < 0.001:
                        alpha = 0.5 if self.probe_count <= 5 else 0.2
                        self._drift_rate = (1 - alpha) * self._drift_rate + alpha * new_rate
                self._C = C_new
                self._C_time = t_now
            log.info("PtsStream: probe #%d, C=%.6f, drift=%.1f ppm, "
                     "pred_err=%.2fms",
                     self.probe_count, C_new,
                     self._drift_rate * 1e6, pred_err_ms)
            # Save PLL drift_rate to .drift_freq for next session
            if self._drift_freq_path:
                try:
                    with open(self._drift_freq_path, 'w') as f:
                        f.write("{:.10e}\n".format(self._drift_rate))
                except Exception:
                    pass


class BufferedCapture(Process):
    """ Capture from device to buffer in memory.
    """
    
    running = False
    
    def __init__(self, array1, start_time1, array2, start_time2, config, video_file=None, night_data_dir=None,
                 saved_frames_dir=None, daytime_mode=None, camera_mode_switch_trigger=None):
        """ Populate arrays with (startTime, frames) after startCapture is called.
        
        Arguments:
            array1: numpy array in shared memory that is going to be filled with frames
            start_time1: float in shared memory that holds time of first frame in array1
            array2: second numpy array in shared memory
            start_time2: float in shared memory that holds time of first frame in array2

        Keyword arguments:
            video_file: [str] Path to the video file, if it was given as the video source. None by default.
            night_data_dir: [str] Path to the directory where night data is stored. None by default.
            saved_frames_dir: [str] Path to the directory where saved frames are stored. None by default.
            daytime_mode: [multiprocessing.Value] Shared boolean variable to communicate camera mode switch
                direction (daytime or nighttime).
            camera_mode_switch_trigger: [multiprocessing.Value] Shared boolean variable to trigger camera 
                mode switch at the right time.
        """
        
        super(BufferedCapture, self).__init__()
        
        # Store configuration and paths (immutable data is safe to pass to child process)
        self.config = config
        self.video_file = video_file
        self.night_data_dir = night_data_dir
        self.saved_frames_dir = saved_frames_dir

        # make sure the flags are always real shared Values
        if daytime_mode is None:
            self.daytime_mode = Value(ctypes.c_bool, False)       # default: "night"
        else:
            self.daytime_mode = daytime_mode

        if camera_mode_switch_trigger is None:
            self.camera_mode_switch_trigger = Value(ctypes.c_bool, False)
        else:
            self.camera_mode_switch_trigger = camera_mode_switch_trigger

        # Store shared memory arrays and values for compressor (these are designed for multiprocessing)
        self.array1 = array1
        self.start_time1 = start_time1
        self.array2 = array2
        self.start_time2 = start_time2
        self.start_time1.value = 0
        self.start_time2.value = 0

        # Initialize shared values for raw frame saving (these are designed for multiprocessing)

        # Raw-frame infrastructure: always create, even if save_frames=False
        self.raw_frame_saver = None    # avoids AttributeError in releaseRawArrays
        self.shared_raw_array = None   # primary raw-frame buffer
        self.shared_raw_array2 = None  # secondary raw-frame buffer

        if self.config.save_frames:

            # Frame saving block size - these many raw frames are written to buffer before saving to disk
            self.num_raw_frames = 10

            self.start_raw_time1 = Value('d', 0.0)
            self.start_raw_time2 = Value('d', 0.0)
            self.shared_timestamps_base = Array(ctypes.c_double, self.num_raw_frames)
            self.shared_timestamps_base2 = Array(ctypes.c_double, self.num_raw_frames)

        # Initialize shared counter for dropped frames
        self.dropped_frames = Value('i', 0)
        self.last_daytime_mode = None  # Track day/night transitions
        self.dropped_frames_timestamps = deque()  # Track when frames were dropped for 10-min window

        # Flag for process control
        self.exit = Event()

        # Heartbeat timestamp for watchdog - updated every frame block to detect hangs
        self.heartbeat = Value('d', 0.0)

        # Initialize sync tick
        self.last_sync_tick = -1
        self.sync_tick_reference = 0  # reference epoch for sync ticks

        # Timestamp correction between smoothed PTS and pipeline running time
        self.last_pts_correction_ns = 0
        self.last_running_time_ns = None

        # Storage branch PTS guard: fix stale-PTS duplicates before splitmuxsink
        self._storage_last_pts_ns = 0

        # handle for the Gst bus-poller thread
        self._bus_should_exit = False
        self._bus_thread = None


    def startCapture(self, cameraID=0):
        """ Start capture using specified camera.
        
        Arguments:
            cameraID: ID of video capturing device (ie. ID for /dev/video3 is 3). Default is 0.
            
        """
        
        self.cameraID = cameraID
        self.exit = Event()

        self.start()
    

    def stopCapture(self):
        """ Stop capture.
        """
        
        self.exit.set()

        time.sleep(1)

        log.info("Joining capture...")

        # Wait for the capture to join for 60 seconds, then terminate
        waiting_to_join = 60
        log.info("Waiting up to {} seconds for capture to join...".format(waiting_to_join))
        
        # Track how many seconds we actually waited
        seconds_waited = 0
        
        for i in range(waiting_to_join):
            seconds_waited = i + 1
            if self.is_alive():
                time.sleep(1)
            else:
                break
        
        # Log the outcome based on final state
        if not self.is_alive():
            log.info("Capture joined successfully after {} seconds".format(seconds_waited))
        else:
            log.info("Timed out after waiting {} seconds, capture thread still alive".format(seconds_waited))
            log.info("Sending interrupt signal for graceful shutdown...")
            
            try:
                # Send SIGINT to allow child process to clean up gracefully
                if self.pid:
                    os.kill(self.pid, signal.SIGINT)
                
                # Wait a few seconds for graceful shutdown
                self.join(5)
                
                if self.is_alive():
                    log.warning("Process still alive after interrupt, forcing termination")
                    self.terminate()
                else:
                    log.info("Process exited gracefully after interrupt")
                    
            except ProcessLookupError:
                log.info("Process already terminated")
            except Exception as e:
                log.error("Error during graceful shutdown: {}".format(e))
                log.info("Falling back to terminate()")
                self.terminate()
            
            # Always join to reap zombie (returns instantly if already dead)
            self.join()
            
            # Note: RTSP connections are cleaned up by releaseResources() in the child process
            
            # Clean up raw frame arrays after process termination
            if hasattr(self, 'raw_frame_saver') and self.raw_frame_saver:
                self.releaseRawArrays()

        return self.dropped_frames.value


    def deviceIsOpened(self):
        """ Return True if media backend is opened.
        """

        if self.device is None:
            return False
        
        try:
            # OpenCV
            if self.video_device_type == "cv2":

                return self.device.isOpened()
            
            # GStreamer
            else:

                if GST_IMPORTED:

                    # Use a 10-second timeout to avoid indefinite blocking while checking if the device is in the PLAYING state
                    state = self.device.get_state(Gst.SECOND * 10).state

                    if state == Gst.State.PLAYING:
                        return True
                    else:
                        return False
                    
                else:
                    return False
                
        except Exception as e:
            log.error('Error checking device status: {}'.format(e))
            return False


    def calculatePTSRegressionParams(self, y):
        """ Add pts and perform an online linear regression on pts.
            smoothed_pts = m*frame_count + b
            m is the slope in ns per frame (1e9/fps)
            Adjust b so that the line passes through the earliest frames.

        Arguments:
            y: [float] pts of the frame

        Return:
            m: [float] slope in ns per frame
            b: [float] y-intercept in ns

        """
        self.n += 1
        x = self.n
        self.sum_x += x
        self.sum_y += y
        self.sum_xx += x*x
        self.sum_xy += x*y

        # Update regression parameters
        if x > 1:
            m = (self.n*self.sum_xy - self.sum_x*self.sum_y)/(self.n*self.sum_xx - self.sum_x**2)
        
        # First frame
        else:
            m = self.expected_m
            self.b = y - m*x
        
        ## STARTUP ##
        # On startup, use expected fps until calculate fps stabilizes
        if (self.n <= self.startup_frames) and self.startup_flag:

            # Exit startup if calculated m doesn't converge with expected m

            # Check error at increasingly longer intervals
            if x < self.startup_frames/32:
                sample_interval = 128
            elif x < self.startup_frames/16:
                sample_interval = 512
            elif x < self.startup_frames/8:
                sample_interval = 1024
            elif x < self.startup_frames/4:
                sample_interval = 2048
            else:
                sample_interval = 4096

            # Determine if the values converge. Skipping the first few noisy frames
            if ((x - 25)%sample_interval == 0) or (x == self.startup_frames):

                m_err = abs(m - self.expected_m)
                delta_m_err = (m_err - self.last_m_err)/(x - self.last_m_err_n)
                startup_remaining = self.startup_frames - x
                final_m_err = m_err + startup_remaining*delta_m_err
                self.last_m_err = m_err
                self.last_m_err_n = x

                # If end is reached, or error does not converge to zero, exit startup
                if (final_m_err > 0) or (x == self.startup_frames):

                    # If residual error on exit is too large, the expected m is probably wrong.
                    if m_err > 2000:

                        # Reset debt and b as they were probably wrong, and permanently disable startup
                        self.startup_flag = False
                        self.b_error_debt = 0
                        self.b = y - m*x
                        self.m_jump_error = 0

                        log.debug("Check config FPS! Startup sequence exited early probably due to inaccurate FPS value. "
                                 "Startup is disabled for the remainder of the run")

                    # On normal exit, calculate residual error for smooth transition to calculate m
                    else:

                        # calculate the jump error
                        self.m_jump_error = x*(m - self.expected_m) # ns

                    log.debug("Exiting startup logic at {:.1f}% of startup sequence, Expected fps: {:.6f}, "
                             "calculated fps at this point: {:.6f}, residual m error: {:.1f} ns, sample interval: {}"
                             .format(100*x/self.startup_frames, 1e9/self.expected_m, 1e9/m, m_err, sample_interval))

                    # This will temporarily exit startup
                    self.startup_frames = 0

            # Use expected value during startup
            if self.startup_frames > 0:
                m = self.expected_m

        ### LEAST DELAYED FRAME LOGIC ###
                
        # The code attempts to smoothly distribute presentation timestamps (pts) on a line that passes
        # through the least-delayed frame. The idea is that the least-delayed frames are thought to
        # be the least affected by network and other delays, and should therefore offer the most
        # consistent points of reference.
        # When a new least-delayed frame is detected, the time delta is smoothly distributed over
        # time.
        # The line has a slope m (ns per frame) that passes through the least delayed frame by
        # adjusting b in: y = m*x + b
        # where y is the pts, and x is the frame number.
        # A slow positive bias is introduce to keep the line in contact with a slowly accelerating
        # frame rate.
        # Finally, the small jump error at the completion of the startup sequence, when
        # transitioning from expected fps to calculated fps (linear regression), is smoothly
        # distributed over time.
                
        # Calculate the delta between the lowest point and current point
        delta_b = self.b - (y - m*x)

        # Adjust b error debt to the max of current debt or new delta b
        self.b_error_debt = max(self.b_error_debt, delta_b)
        
        # Skew b, if due
        if self.b_error_debt > 0 or self.m_jump_error != 0:

            # Don't limit changes to b for the first few blocks of frames
            if x <= 256*3:
                max_adjust = float('inf')

            # Then adjust b aggressively for the first few minutes
            elif x <= 256*6*10: # first ~10 min
                max_adjust = 100*1000/256 # 0.1 ms per block

            # Then only allow small changes for the remainder of the run
            else:
                max_adjust = 25*1000/256 # 0.025 ms per block
            
            # Determine the correction factor
            b_corr = min(self.b_error_debt, max_adjust) # ns

            # Update the lowest b and adjust the debt
            self.b -= b_corr
            self.b_error_debt -= b_corr

            # Update m jump error debt
            if self.m_jump_error > 0:
                self.m_jump_error = max(self.m_jump_error - max_adjust, 0)
            else:
                self.m_jump_error = min(self.m_jump_error + max_adjust, 0)

        else:
            # Introduce a very small positive bias
            self.b += 25 # ns
        
        return m, self.b - self.m_jump_error


    def smoothPTS(self, new_pts):
        """ Smooth pts using linear regression.

        Arguments:
            new_pts: [float] pts of the frame

        Return:
            smoothed_pts: [float] smoothed pts

        """

        # Disable smoothing if too many resets are detected
        if self.reset_count >= 50:
            if self.reset_count == 50:
                log.info("Too many resets. Disabling smoothing function!")
                self.reset_count += 1
            return new_pts

        # Calculate linear regression params
        m, b = self.calculatePTSRegressionParams(new_pts)

        # Store last calculated fps for the longest run so far
        if self.n > self.last_calculated_fps_n:
            self.last_calculated_fps = 1e9/m
            self.last_calculated_fps_n = self.n

        # On initial run or after a reset
        if self.n == 1:
            smoothed_pts = new_pts

        # Calculate smoothed pts from regression parameters
        else:
            smoothed_pts = m*self.n + b

            # Reset regression on dropped frame (raw pts is more than 1 frame late)
            if new_pts - smoothed_pts > self.expected_m:

                self.reset_count += 1
                self.n = 0
                self.sum_x = 0
                self.sum_y = 0
                self.sum_xx = 0
                self.sum_xy = 0
                self.startup_frames = 25*60*10 # 10 minutes
                self.m_jump_error = 0
                self.b_error_debt = 0
                self.last_m_err = float('inf')
                self.last_m_err_n = 0
                log.info('smooth_pts detected dropped frame. Resetting regression parameters.')

                return new_pts
        
        return smoothed_pts


    def read(self):
        """ Retrieve frames and timestamp.

        Return:
        (tuple): (ret, frame, timestamp, gst_pts_ns) where ret is a boolean indicating
                 success, frame is the captured frame, timestamp is the frame UTC timestamp,
                 and gst_pts_ns is the raw GStreamer buffer PTS in nanoseconds (0 for non-GStreamer).
                 gst_pts_ns is the shared key between MKV frames and FT entries.
        """
        ret, frame, timestamp, gst_pts_ns = False, None, None, 0

        # Read Video file frame
        if self.video_file is not None:
            ret, frame = self.device.read()
            if ret:
                timestamp = None # assigned later
        
        # Read capture device frame
        else:

            # GStreamer
            if GST_IMPORTED and (self.config.media_backend == 'gst') and (not self.media_backend_override):

                # Pull a frame from the GStreamer pipeline with a .5 sec timeout
                sample = self.device.emit("try-pull-sample", 500 * Gst.MSECOND)
                if not sample:
                    log.info("GStreamer pipeline did not emit a sample.")
                    return False, None, None, 0

                # Extract the frame buffer and timestamp
                buffer = sample.get_buffer()
                if not buffer:
                    log.error("Failed to get buffer from sample.")
                    return False, None, None, 0

                gst_timestamp_ns = buffer.pts  # GStreamer timestamp in nanoseconds
                gst_pts_ns = gst_timestamp_ns  # shared key for MKV↔FT lookup

                # Sanity check for pts value
                max_expected_ns = 24*60*60*1e9  # 24 hours in nanoseconds
                if not (0 < gst_timestamp_ns <= max_expected_ns):
                    log.info("Unexpected PTS value: {}.".format(gst_timestamp_ns))
                    return False, None, None, 0

                ret, map_info = buffer.map(Gst.MapFlags.READ)
                if not ret:
                    log.info("GStreamer Buffer did not contain a frame.")
                    return False, None, None, 0

                try:
                    # Convert to np.ndarray
                    frame = np.ndarray(shape=self.frame_shape, buffer=map_info.data, dtype=np.uint8)

                    # Deferred VENC calibration: if pre-pipeline calibration failed,
                    # compute C now that frames are flowing and combine with
                    # clock_base from the pad probe for sub-ms accuracy.
                    if not self._venc_post_start_calibrated \
                            and self._rtp_clock_base is not None \
                            and hasattr(self, '_pts_stream') \
                            and self._pts_stream is not None \
                            and self._pts_stream._C_raw is not None \
                            and getattr(self, '_last_delivery_rtp', 0) != 0:
                        try:
                            # Correct epoch_offset using PtsStream's C.
                            # C maps raw VENC PTS (µs) to UTC: UTC = C + venc_us * 1e-6
                            # epoch_offset was time.time() at first packet arrival,
                            # biased by ~90ms of GStreamer pipeline latency.
                            # Fix: get first frame's raw VENC PTS from PtsStream
                            # entries (no wrap_offset contamination), then compute
                            # correct epoch_offset = C + first_venc_us * 1e-6.
                            C = self._pts_stream._C_raw
                            WRAP_US = 2**32
                            first_venc_us = None
                            with self._pts_stream._lock:
                                if self._pts_stream._entries:
                                    # Oldest entry ≈ first frame's VENC PTS
                                    first_venc_us = self._pts_stream._entries[0][0]
                            if C is not None:
                                # Measure pipeline latency automatically.
                                # The PtsStream entry that arrived closest to
                                # epoch_offset (first appsink delivery) is the
                                # first video frame.  Its raw VENC PTS gives
                                # the true capture time: C + venc_us * 1e-6.
                                # Pipeline latency = epoch_offset - true_capture.
                                # Measure pipeline latency directly.
                                # The appsink delivered a frame at T_del.
                                # The same frame's PtsStream entry arrived
                                # ~(pipeline - 15ms) earlier.  Try entries
                                # near T_del - 75ms, compute true capture
                                # via C, pipeline_latency = T_del - capture.
                                WRAP_US = 2**32
                                pipeline_latency = -1
                                T_del = getattr(self, '_last_delivery_time', 0)
                                # Only use T_del if recent (not stale from pre-reboot)
                                if T_del > 0 and abs(T_del - self.venc_epoch_offset) < 5.0:
                                    with self._pts_stream._lock:
                                        entries = list(self._pts_stream._entries)
                                    # Find entries with arrival near T_del - 75ms
                                    # (±50ms window to catch the right one)
                                    candidates = []
                                    for e in entries:
                                        if len(e) < 2:
                                            continue
                                        dt = T_del - e[1]  # time since entry arrived
                                        if 0.025 < dt < 0.125:  # 25-125ms before delivery
                                            capture = C + (e[0] % WRAP_US) * 1e-6
                                            pl = T_del - capture
                                            candidates.append(pl)
                                    log.info("VENC cal: T_del=%.3f C_raw=%.6f n_entries=%d n_cand=%d",
                                             T_del, C, len(entries), len(candidates))
                                    if candidates:
                                        candidates.sort()
                                        pipeline_latency = candidates[len(candidates) // 2]
                                        log.info("VENC cal candidates: %s",
                                                 [round(c*1000, 1) for c in candidates])
                                if 0.01 < pipeline_latency < 0.5:
                                    old_offset = self.venc_epoch_offset
                                    self.venc_epoch_offset -= pipeline_latency
                                    self._venc_post_start_calibrated = True
                                    log.info("VENC calibration: pipeline=%.1fms, "
                                             "epoch_offset %.6f -> %.6f (corrected by %.1fms)",
                                             pipeline_latency * 1000,
                                             old_offset, self.venc_epoch_offset,
                                             (self.venc_epoch_offset - old_offset) * 1000)
                                else:
                                    log.debug("VENC calibration: pipeline=%.1fms, retrying...",
                                              pipeline_latency * 1000)
                                self._venc_drift_baseline_age = None
                                self._venc_drift_total = 0.0
                            else:
                                log.warning("VENC calibration: C or entries not ready")
                        except Exception as e:
                            log.warning("Deferred VENC calibration failed: {}".format(e))

                    if self.venc_epoch_offset is not None:
                        # VENC PTS mode: use raw RTP timestamps (VENC hardware
                        # sensor PTS) directly, bypassing GStreamer's clock
                        # skew correction which adds ~0.9ms σ of jitter.
                        # Falls back to GStreamer PTS if raw RTP unavailable.

                        # Drift correction: disabled when cam_wall re-anchoring
                        # is active (PtsStream with chrony handles drift at µs
                        # level).  Only accumulate if no cam_wall available.
                        if not (hasattr(self, '_pts_stream') and self._pts_stream is not None):
                            VENC_FRAME_PERIOD = 4400 * 1350 / 148.5e6
                            self._drift_total_correction += (
                                self._drift_freq * VENC_FRAME_PERIOD)

                        # Try raw RTP timestamp (from pad probe, no GStreamer
                        # clock skew).  Fall back to GStreamer PTS.
                        #
                        # Raw RTP ts includes clock_base, and epoch_offset
                        # already includes clock_base/90000, so subtract it
                        # to avoid double-counting.  The result is equivalent
                        # to gst_running_time but without jitter buffer skew.
                        _entry = self._rtp_ts_by_pts.get(gst_timestamp_ns, None)

                        if _entry is not None:
                            if isinstance(_entry, tuple):
                                rtp_ts_raw, _frame_anomalous = _entry
                            else:
                                rtp_ts_raw = _entry
                                _frame_anomalous = False
                            self._rtp_dict_hits = getattr(self, '_rtp_dict_hits', 0) + 1
                            if _frame_anomalous:
                                log.info("Side-door: anomalous frame detected at block pos %d",
                                         len(getattr(self, '_block_rtp_ts', [])))

                            # Sequential validation: detect wrong-frame dict
                            # lookups and stale duplicates.
                            if not hasattr(self, '_rtp_prev_raw'):
                                self._rtp_prev_raw = 0
                            if self._rtp_prev_raw != 0:
                                expected_rtp = (self._rtp_prev_raw + 3600) & 0xFFFFFFFF
                                fwd = (rtp_ts_raw - expected_rtp) & 0xFFFFFFFF
                                bwd = (expected_rtp - rtp_ts_raw) & 0xFFFFFFFF
                                off = min(fwd, bwd)
                                if off > 3200:
                                    rtp_ts_raw = expected_rtp
                            # Track RTP 32-bit wraps (~13.3h cycle) for elapsed unwrap.
                            # Detect wrap: backward jump > 2^31 from previous frame.
                            if not hasattr(self, '_rtp_wrap_accum'):
                                self._rtp_wrap_accum = 0
                            if self._rtp_prev_raw and rtp_ts_raw + (1 << 31) < self._rtp_prev_raw:
                                self._rtp_wrap_accum += (1 << 32)
                                log.info("RTP 32-bit wrap at frame (accum=%d)",
                                         self._rtp_wrap_accum)
                            self._rtp_prev_raw = rtp_ts_raw

                            elapsed_ticks = ((rtp_ts_raw - self._rtp_clock_base) & 0xFFFFFFFF) + self._rtp_wrap_accum
                            timestamp = self.venc_epoch_offset + elapsed_ticks / 90000.0

                            if not hasattr(self, '_block_diag'):
                                self._block_diag = []
                            self._block_diag.append({
                                'hit': True,
                                'gst_pts': gst_timestamp_ns,
                                'rtp_raw': rtp_ts_raw,
                                'stale_corrected': (rtp_ts_raw != (_entry[0] if isinstance(_entry, tuple) else _entry)),
                                'timestamp': timestamp,
                            })

                            if hasattr(self, '_pts_stream') and self._pts_stream is not None:
                                if not hasattr(self, '_block_rtp_ts'):
                                    self._block_rtp_ts = []
                                if _frame_anomalous:
                                    self._block_rtp_ts.append(None)
                                else:
                                    guard = getattr(self, '_rtp_probe_shift', 0)
                                    raw_rtp = (rtp_ts_raw + guard) & 0xFFFFFFFF
                                    self._block_rtp_ts.append(raw_rtp)
                        else:
                            # Dict miss: use sequential prediction
                            self._rtp_dict_misses = getattr(self, '_rtp_dict_misses', 0) + 1
                            if not hasattr(self, '_rtp_prev_raw'):
                                self._rtp_prev_raw = 0
                            if self._rtp_prev_raw != 0:
                                rtp_ts_raw = (self._rtp_prev_raw + 3600) & 0xFFFFFFFF
                                # Synthetic +3600 shouldn't wrap within a frame,
                                # but use accumulator for consistency
                                if rtp_ts_raw + (1 << 31) < self._rtp_prev_raw:
                                    self._rtp_wrap_accum = getattr(self, '_rtp_wrap_accum', 0) + (1 << 32)
                                self._rtp_prev_raw = rtp_ts_raw
                                wrap_accum = getattr(self, '_rtp_wrap_accum', 0)
                                elapsed_ticks = ((rtp_ts_raw - self._rtp_clock_base) & 0xFFFFFFFF) + wrap_accum
                                timestamp = self.venc_epoch_offset + elapsed_ticks / 90000.0
                            else:
                                timestamp = self.venc_epoch_offset + gst_timestamp_ns / 1e9
                            if not hasattr(self, '_block_diag'):
                                self._block_diag = []
                            self._block_diag.append({
                                'hit': False,
                                'gst_pts': gst_timestamp_ns,
                                'rtp_raw': rtp_ts_raw if self._rtp_prev_raw != 0 else 0,
                                'stale_corrected': False,
                                'timestamp': timestamp,
                            })
                            if self._rtp_dict_misses <= 3 or self._rtp_dict_misses % 1000 == 0:
                                log.info("RTP dict miss #%d: gst_pts=%d dict_size=%d",
                                         self._rtp_dict_misses, gst_timestamp_ns,
                                         len(self._rtp_ts_by_pts))

                        if hasattr(self, '_venc_meta') and self._venc_meta is not None:
                            meta = self._venc_meta.latest
                            if 'exposure_us' in meta:
                                timestamp -= meta['exposure_us'] / 1e6

                        self._last_venc_gst_pts = gst_timestamp_ns
                        self._last_venc_timestamp = timestamp
                        self._last_delivery_time = time.time()
                        self._last_delivery_rtp = getattr(self, '_rtp_prev_raw', 0)

                        # Side-door RTP collection (dict hit path handled above)
                        if _entry is None and getattr(self, '_rtp_prev_raw', 0) != 0:
                            if hasattr(self, '_pts_stream') and self._pts_stream is not None:
                                if not hasattr(self, '_block_rtp_ts'):
                                    self._block_rtp_ts = []
                                guard = getattr(self, '_rtp_probe_shift', 0)
                                raw_rtp = (self._rtp_prev_raw + guard) & 0xFFFFFFFF
                                self._block_rtp_ts.append(raw_rtp)

                        # (exposure correction and state update already done above)

                        # Store timestamp for drift measurement.
                        self._venc_raw_timestamp = timestamp

                    else:
                        # Legacy mode: smooth PTS via linear regression and correct
                        # with pipeline running time.
                        smoothed_pts = self.smoothPTS(gst_timestamp_ns)
                        running_time_ns = None

                        if self.pipeline is not None:
                            try:
                                clock = self.pipeline.get_clock()
                                base_time = self.pipeline.get_base_time()

                                if clock is not None and base_time != Gst.CLOCK_TIME_NONE:
                                    clock_time = clock.get_time()
                                    if clock_time != Gst.CLOCK_TIME_NONE and clock_time >= base_time:
                                        running_time_ns = clock_time - base_time
                            except Exception as clock_exc:
                                log.debug("Failed to query pipeline clock/base time: %s", clock_exc)

                        if running_time_ns is not None:
                            self.last_running_time_ns = running_time_ns
                            self.last_pts_correction_ns = smoothed_pts - running_time_ns
                            corrected_running_time_ns = running_time_ns + self.last_pts_correction_ns
                            timestamp = self.start_timestamp + (corrected_running_time_ns/1e9)
                        else:
                            self.last_running_time_ns = None
                            self.last_pts_correction_ns = 0
                            timestamp = self.start_timestamp + (smoothed_pts/1e9)

                finally:
                    # Always unmap buffer to prevent memory leaks
                    buffer.unmap(map_info)

            # OpenCV
            else:
                ret, frame = self.device.read()
                if ret:
                    timestamp = time.time()

        return ret, frame, timestamp, gst_pts_ns


    def _driftFreqPath(self):
        """Path to per-station drift frequency file."""
        return os.path.join(self.config.config_file_path, '.drift_freq')

    def _loadDriftFreq(self):
        """Load persisted drift frequency, or return 0."""
        try:
            with open(self._driftFreqPath(), 'r') as f:
                freq = float(f.read().strip())
            # Sanity check: VENC PTS counter rate offset plus crystal
            # drift can reach ~500 ppm.  Reject obviously corrupted values.
            if abs(freq) > 600e-6:
                log.warning("Ignoring corrupted drift freq: %.1f ppm from %s",
                            freq * 1e6, self._driftFreqPath())
                return 0.0
            log.info("Loaded drift freq: %.1f ppm from %s",
                     freq * 1e6, self._driftFreqPath())
            return freq
        except Exception:
            return 0.0

    def _saveDriftFreq(self):
        """Persist current drift frequency estimate."""
        try:
            with open(self._driftFreqPath(), 'w') as f:
                f.write("{:.10e}\n".format(self._drift_freq))
        except Exception:
            pass


    def extractRtspUrl(self, input_string):
        """
        Return validated camera url
        """

        # Define a regular expression pattern for RTSP URLs
        pattern = r'rtsp://[^\s]+'

        # Search for the pattern in the input string
        match = re.search(pattern, input_string)

        # Extract, format, and return the RTSP URL
        if match:

            rtsp_url = match.group(0)

            # Add '/' if it's missing from '.sdp' URL
            if rtsp_url.endswith('.sdp'):
                rtsp_url += '/'

            return rtsp_url

        # If no match is found, return None or handle as appropriate
        else:
            log.error("No RTSP URL found in the input string: {}".format(input_string))
            raise ValueError("No RTSP URL found in the input string: {}".format(input_string))
    

    def probeRtspService(self, max_attempts=720, probe_interval=10, timeout=1):
        """
        Test RTSP service availability by attempting TCP connection to the service port.
        Uses TCP connection only - does not validate RTSP protocol
        
        Performs a thorough connection test by:
        1. Resolving hostname via DNS
        2. Creating a TCP socket connection to the RTSP port
        3. Analyzing any connection failures
        4. Retrying with backoff if connection fails
        
        Args:
            max_attempts (int, optional): Maximum number of connection attempts before giving up.
                Defaults to 720.
            probe_interval (int, optional): Time in seconds between connection attempts.
                Defaults to 10 seconds.
            timeout (int, optional): Socket connection timeout in seconds.
                Defaults to 1 second.
        
        Returns:
            tuple: A pair (success, status) where:
                - success (bool): True if connection was successful, False otherwise
                - status (str): One of the RtspProbeResult status strings:
                    - SUCCESS: Connection successful
                    - NETWORK_DOWN: Local network interface is down
                    - HOST_UNREACHABLE: Cannot reach the target host
                    - CONNECTION_REFUSED: Host up but RTSP port is closed
                    - TIMEOUT: Connection attempt timed out
                    - DNS_ERROR: Cannot resolve hostname
                    - UNKNOWN_ERROR: Other connection failures
                
        """
        try:
            # Parse RTSP URL to get host and port
            device_url = self.extractRtspUrl(self.config.deviceID)
            parsed = urlparse(device_url)
            host = parsed.hostname
            port = parsed.port or 554

            # VENC cameras: probe the gettime service port instead of
            # RTSP port. XM RTSP servers create stale sessions from bare
            # TCP connects, blocking subsequent GStreamer connections.
            # The gettime service (rtp_patch daemon) is safe to probe.
            # Restart PtsStream on reconnection (camera may have rebooted)
            if hasattr(self, '_pts_stream') and self._pts_stream is not None:
                self._pts_stream.restart()

            venc_port = getattr(self.config, 'venc_gettime_port', 0)
            if venc_port > 0:
                probe_port = venc_port
                log.info("Using VENC gettime port %d for probe (avoiding RTSP session)", probe_port)
            else:
                probe_port = port

            last_error = None
            stop_event = getattr(self, "exit", None)

            for attempt in range(max_attempts):

                if stop_event is not None and stop_event.is_set():
                    log.info("RTSP probe aborted - shutdown requested")
                    return False, RtspProbeResult.UNKNOWN_ERROR

                # Update heartbeat during probe attempts to show we're still alive
                if hasattr(self, 'heartbeat'):
                    self.heartbeat.value = time.time()

                try:
                    # Try to resolve hostname first
                    try:
                        socket.gethostbyname(host)
                    except socket.gaierror:
                        last_error = RtspProbeResult.DNS_ERROR
                        raise

                    # Create socket with timeout
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(timeout)

                    # Try to connect
                    result = sock.connect_ex((host, probe_port))
                    sock.close()

                    if result == 0:
                        if venc_port > 0:
                            # VENC probe: gettime daemon is up, but the RTSP
                            # server in App may still be stabilizing after
                            # ptrace attach/detach by rtp_patch.  Give it a
                            # few seconds before the pipeline tries RTSP.
                            log.info("VENC gettime service ready after {} attempts, waiting 15s for RTSP stabilization...".format(attempt + 1))
                            time.sleep(15)
                        else:
                            # First probe succeeded, wait 10s and verify with second probe
                            log.info("First probe successful, waiting 10s for verification probe...")
                            time.sleep(10)

                            # Second verification probe
                            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            sock.settimeout(timeout)
                            result = sock.connect_ex((host, probe_port))
                            sock.close()

                        if result == 0:
                            log.info("RTSP service ready after {} attempts (verified with 2 probes)".format(attempt + 1))
                            return True, RtspProbeResult.SUCCESS

                        log.info("Second probe failed, continuing retry loop...")

                    # Analyze specific connection errors
                    if result in (errno.ENETUNREACH, errno.ENETDOWN):
                        last_error = RtspProbeResult.NETWORK_DOWN
                    elif result in (errno.EHOSTUNREACH, errno.EHOSTDOWN):
                        last_error = RtspProbeResult.HOST_UNREACHABLE
                    elif result == errno.ECONNREFUSED:
                        last_error = RtspProbeResult.CONNECTION_REFUSED
                    elif result == errno.ETIMEDOUT:
                        last_error = RtspProbeResult.TIMEOUT
                    else:
                        last_error = RtspProbeResult.UNKNOWN_ERROR
                        
                except socket.gaierror:
                    last_error = RtspProbeResult.DNS_ERROR
                except socket.timeout:
                    last_error = RtspProbeResult.TIMEOUT
                except socket.error as e:
                    if e.errno in (errno.ENETUNREACH, errno.ENETDOWN):
                        last_error = RtspProbeResult.NETWORK_DOWN
                    elif e.errno in (errno.EHOSTUNREACH, errno.EHOSTDOWN):
                        last_error = RtspProbeResult.HOST_UNREACHABLE
                    else:
                        last_error = RtspProbeResult.UNKNOWN_ERROR
                    log.debug("RTSP probe attempt {} failed: {}".format(attempt + 1, e))
                
                error_messages = {
                    RtspProbeResult.NETWORK_DOWN: "Network appears to be down",
                    RtspProbeResult.HOST_UNREACHABLE: "Camera is unreachable",
                    RtspProbeResult.CONNECTION_REFUSED: "RTSP service not accepting connections",
                    RtspProbeResult.TIMEOUT: "Connection attempt timed out",
                    RtspProbeResult.DNS_ERROR: "Cannot resolve camera hostname",
                    RtspProbeResult.UNKNOWN_ERROR: "Unknown connection error"
                }
                
                print('Trying to connect to camera RTSP service... (attempt {}) - {}'.format(
                    attempt + 1, error_messages[last_error]))
                time.sleep(probe_interval)
                
            log.error("RTSP service not responding after all attempts. Last error: {}".format(
                error_messages[last_error]))
            return False, last_error
            
        except Exception as e:
            log.error("Error probing RTSP service: {}".format(e))
            return False, RtspProbeResult.UNKNOWN_ERROR
                

    def isGrayscale(self, frame, stride=64):
        """
        Quickly check if a frame is grayscale by sampling pixels along the diagonal.
        If all three channels match on those diagonal samples, return True.
        If an IndexError is raised (i.e., frame is single-channel), also return True.
        This trades completeness for speed, as only the diagonal is checked.

        Args:
            frame (numpy.ndarray): The image frame to check (usually BGR or GRAY).
            stride (int): Spacing for diagonal sampling, skipping many pixels for efficiency.

        Returns:
            bool or None: True if all sampled channels match (or frame is single-channel),
                False if channels differ (color), or None if inconclusive (all pixels are
                the same value, e.g. all white or all black).
        """
        if frame is None:
            raise ValueError("isGrayscale() called with frame=None")

        # We don't explicitly check frame.shape first; instead we rely on an IndexError
        # if 'frame' is single-channel (which is inherently grayscale).
        # This is faster than an extra dimension check for most BGR GMN stations

        try:
            # If diagonal samples are not identical, frame is color
            sampled = frame[::stride, ::stride]
            is_gray = np.all(sampled[..., 0] == sampled[..., 1]) and \
                    np.all(sampled[..., 1] == sampled[..., 2])

            # If all sampled pixels have the same value (e.g. all white or all black),
            # the check is inconclusive — channels match trivially
            if is_gray and np.all(sampled[..., 0] == sampled[0, 0, 0]):
                return None

        except IndexError:
             # If IndexError, frame is grayscale
            is_gray = True

        return is_gray


    def handleGrayscaleConversion(self, frame):
        """Handle conversion of frame to grayscale if necessary.

            Camera outputs BGR (3 channels) even in night mode. For efficiency, we save raw frames in:
            - Grayscale (1 channel) when all channels are identical
            - Full BGR (3 channels) when they differ

            Note: While raw frames are saved in color when available, frames are converted
            to grayscale before compression in the processing pipeline

        Args:
            frame: a numpy.ndarray frame

        Returns:
            numpy.ndarray: Frame data either as grayscale (2D) or BGR (3D) array
        """

        # First check if frame is None to prevent NoneType subscript error
        if frame is None:
            return None

        # We don't explicitly check frame.shape first; instead we rely on an IndexError
        # if 'frame' is single-channel (which is inherently grayscale).
        # This is faster than an extra dimension check for most BGR GMN stations 

        try:
            # If frame channels are not identical (color), return all 3 channels
            if not self.convert_to_gray:
                return frame

            try:
                # If frame channels are identical (gray), extract green channel for grayscale
                return frame[:, :, 1]
        
            except IndexError:
                # If IndexError occurs, frame is already grayscale (single-channel)
                return frame
            
        except Exception as e:
            log.error('Error in grayscale conversion: {}'.format(e))
            log.debug('Frame shape: {}'.format(frame.shape if frame is not None else None))
            return None


    def moveSegment(self, splitmuxsink, fragment_id, first_sample=None):
        """
        Custom callback for splitmuxsink's format-location signal to name and move each segment as its
        created. Generates a timestamp-based folder structure: Year/Day-Of-Year/Hour/ per video segment.

        Arguments:
          splitmuxsink [GstElement]: The splitmuxsink object itself, included in arguments as GStreamer expects it.
          fragment_id [int]: Fragment / segment number of the new clip
          first_sample [GstSample]: First sample in the fragment when connected to the
              ``format-location-full`` signal. Optional.

        Returns:
          full_path [str]: Full path to save this new video segment to
        """

        segment_timestamp = None
        corrected_running_time_ns = None

        if first_sample is not None:
            try:
                buffer = first_sample.get_buffer()
                segment = first_sample.get_segment()
                running_time_ns = None

                if buffer is not None:
                    buffer_pts = buffer.pts
                    if buffer_pts != Gst.CLOCK_TIME_NONE:
                        running_time_ns = buffer_pts

                        if segment is not None:
                            converted = segment.to_running_time(Gst.Format.TIME, buffer_pts)
                            if converted != Gst.CLOCK_TIME_NONE:
                                running_time_ns = converted

                if running_time_ns is not None:
                    # Prefer side-door cam_wall for the segment's first frame
                    # (chrony-disciplined UTC, pipeline-immune). Match the
                    # first buffer's GStreamer PTS to a pts_stream entry via
                    # the raw-RTP dict, then read the matched entry's cam_wall.
                    # Falls back to venc_epoch_offset-based timestamp only if
                    # side-door is unavailable — that path is subject to the
                    # VencCalibration candidate-grid quantization and shifts
                    # by up to ±40ms across RTSP reconnects.
                    sd_ts = None
                    if (hasattr(self, '_pts_stream') and self._pts_stream is not None
                            and hasattr(self, '_rtp_ts_by_pts')):
                        try:
                            # Look up the RTP timestamp corresponding to buffer_pts
                            _entry = self._rtp_ts_by_pts.get(buffer_pts, None)
                            if _entry is not None:
                                rtp_raw = _entry[0] if isinstance(_entry, tuple) else _entry
                                ps = self._pts_stream
                                WRAP32 = 4294967296
                                rtp_us_mod = (int(rtp_raw) * 100 // 9) % WRAP32
                                with ps._lock:
                                    # Walk recent entries (bounded deque)
                                    best_cw, best_d = None, 20000
                                    for e in ps._entries:
                                        if len(e) < 3 or not e[2] or e[2] <= 0:
                                            continue
                                        d = abs(int(e[0] % WRAP32) - int(rtp_us_mod))
                                        if d > WRAP32 // 2:
                                            d = WRAP32 - d
                                        if d < best_d:
                                            best_d, best_cw = d, e[2]
                                if best_cw is not None:
                                    sd_ts = best_cw
                                    # No -0.040 correction here (unlike the
                                    # fallback path below). cam_wall is the
                                    # sensor-side FE_START UTC looked up from
                                    # the buffer's GStreamer PTS. GStreamer
                                    # preserves buffer.pts across the pipeline,
                                    # so the PTS at splitmuxsink == PTS at
                                    # appsink for the same frame. Both sinks
                                    # therefore end up anchored to the same
                                    # sensor-time domain → FF filenames and
                                    # MKV filenames stay byte-for-byte
                                    # alignable without extra compensation.
                                    # The legacy 1-frame bandaid only existed
                                    # because venc_epoch_offset was wall-clock
                                    # at first FF-delivery, 40 ms after the
                                    # same frame reached splitmuxsink.
                        except Exception as sd_exc:
                            log.debug("MKV cam_wall lookup failed: %s", sd_exc)

                    if sd_ts is not None:
                        segment_timestamp = sd_ts
                    elif self.venc_epoch_offset is not None:
                        # Fallback: venc_epoch_offset path (candidate-grid biased).
                        # -0.040 aligns MKV (post-decode) with FF (pre-decode).
                        segment_timestamp = self.venc_epoch_offset + running_time_ns / 1e9 - 0.040
                        if hasattr(self, '_venc_meta') and self._venc_meta is not None:
                            meta = self._venc_meta.latest
                            if 'exposure_us' in meta:
                                segment_timestamp -= meta['exposure_us'] / 1e6
                    else:
                        corrected_running_time_ns = running_time_ns + self.last_pts_correction_ns
                        segment_timestamp = self.start_timestamp + (corrected_running_time_ns / 1e9)

            except Exception as sample_exc:
                log.debug("Failed to derive running time from splitmux sample: %s", sample_exc)

        if segment_timestamp is None and corrected_running_time_ns is None and self.last_running_time_ns is not None:
            corrected_running_time_ns = self.last_running_time_ns + self.last_pts_correction_ns
            segment_timestamp = self.start_timestamp + (corrected_running_time_ns / 1e9)

        if segment_timestamp is not None:
            segment_time = UTCFromTimestamp.utcfromtimestamp(segment_timestamp)
            self.last_segment_savetime = segment_timestamp
        else:
            # Fallback to previous behaviour using wall-clock time
            segment_time = UTCFromTimestamp.utcfromtimestamp(self.last_segment_savetime)
            self.last_segment_savetime = time.time()

        segment_filename = segment_time.strftime("{}_%Y%m%d_%H%M%S_%f_video.mkv".format(self.config.stationID))
        segment_subpath = os.path.join(self.config.data_dir, self.config.video_dir, segment_time.strftime("%Y/%Y%m%d-%j/%Y%m%d-%j_%H"))

        # Create full path for the segment
        mkdirP(segment_subpath)
        segment_full_path = os.path.join(segment_subpath, segment_filename)
        log.info("Created new video segment #%d at: %s",
                 fragment_id, segment_full_path)

        # Return full path to splitmux's callback
        return segment_full_path

      
    def handleStateChange(self, pipeline, target_state, timeout=60):
        """Handle GStreamer pipeline state changes with proper synchronization.
        
        Transitions pipeline through state sequence (NULL->READY->PAUSED->PLAYING),
        ensuring each state change is complete before proceeding. Uses explicit synchronization
        to prevent race conditions.

        For live sources like RTSP, accepts both SUCCESS and NO_PREROLL as valid state changes.

        Args:
            pipeline: The GStreamer pipeline to change state
            target_state: The target state to reach (usually Gst.State.PLAYING)
            timeout: Maximum seconds to wait for each state change (default 60)

        tuple: (success, start_time) where:
            - success (bool): True if state change succeeded, False if any step failed
            - start_time (float or None): Timestamp when PAUSED state was initiated, or None if not reached
        """

        try:
            # Initialize start time
            start_time = None

            # Get current pipeline state
            ret, current, pending = pipeline.get_state(0)
            log.debug("Current pipeline state: {}, pending: {}".format(current.value_nick, pending.value_nick))

            # Define the sequence of states we need to go through
            target_sequence = [Gst.State.READY, Gst.State.PAUSED, Gst.State.PLAYING]

            # Find where we are in the sequence (-1 if current state isn't in sequence)
            current_index = target_sequence.index(current) if current in target_sequence else -1
            target_index = target_sequence.index(target_state)
            
            # Step through each state change needed to reach target
            for state in target_sequence[current_index + 1:target_index + 1]:
                log.debug("Transitioning to {} state...".format(state.value_nick))
                
                # Force synchronization before state change to prevent race conditions
                if not pipeline.sync_children_states():
                    log.warning("Sync failed before {}".format(state.value_nick))

                # Capture time just before camera starts capture
                if state == Gst.State.PAUSED:
                    start_time = time.time()

                # Request state change and wait for completion
                ret = pipeline.set_state(state)
                ret, new_state, pending = pipeline.get_state(Gst.SECOND * timeout)
                
                # Both SUCCESS and NO_PREROLL are valid (NO_PREROLL happens with live sources)
                if ret not in (Gst.StateChangeReturn.SUCCESS, Gst.StateChangeReturn.NO_PREROLL):
                    log.error("Failed to change to state {}".format(state.value_nick))
                    return False, None
                
                # Force synchronization after state change
                if not pipeline.sync_children_states():
                    log.warning("Sync failed after {}".format(state.value_nick))
                
                log.debug("Successfully transitioned to {} state".format(state.value_nick))
                    
            return True, start_time
            
        except Exception as e:
            log.error("State change error: {}".format(str(e)))
            import traceback
            log.debug(traceback.format_exc())
            return False, None


    def shouldSaveFrame(self, ts):
        """
        Return True for the first frame that lies within half a frame of
        each universal sync tick (ref + k·dt).

        Arguments:
            ts : [float]  epoch timestamp of current frame (seconds)
        """
        dt   = self.config.frame_save_aligned_interval  # seconds
        ref  = self.sync_tick_reference                 # epoch offset
        tol  = 0.5 / self.config.fps                    # half-frame

        tick = int((ts - ref) / dt + 0.5)
        tick_time = ref + tick * dt

        if abs(ts - tick_time) <= tol and tick != self.last_sync_tick:
            self.last_sync_tick = tick
            return True
        return False


    def _onFirstRtpBuffer(self, pad, info):
        """Pad probe callback on depayloader sink: capture raw RTP timestamps.

        First call: capture clock_base for epoch_offset calibration.
        Subsequent calls: track per-frame RTP timestamps and detect anomalies
        before GStreamer's jitterbuffer processing.  RTP marker bit indicates
        the last packet of a frame — log the RTP timestamp at each new frame.
        """
        buf = info.get_buffer()
        try:
            success, rtp_buf = GstRtp.RTPBuffer.map(buf, Gst.MapFlags.READ)
            if success:
                rtp_ts = rtp_buf.get_timestamp()
                marker = rtp_buf.get_marker()
                rtp_ssrc = rtp_buf.get_ssrc()
                rtp_buf.unmap()

                # First packet: capture clock_base and epoch_offset.
                # epoch_offset = host time at stream start.  NOT C + clock_base/90000:
                # C (raw VENC domain) and clock_base (trampoline-filtered domain)
                # diverge after PTS wraps.  The trampoline offset cancels in
                # elapsed_ticks = (rtp - clock_base), so epoch_offset + elapsed/90000
                # gives correct capture time.  Sub-ms accuracy comes from the
                # side-door (align_block + PLL), not from epoch_offset.
                if self._rtp_clock_base is None:
                    # Skip obviously-bogus initial values: rtp_ts=0 can come
                    # from RTSP session init or pre-trampoline packets.
                    # Wait for a real frame with non-zero rtp_ts.
                    if rtp_ts == 0:
                        log.debug("RTP probe: ignoring rtp_ts=0 (pre-trampoline)")
                        return Gst.PadProbeReturn.OK
                    self._rtp_clock_base = rtp_ts
                    self._rtp_probe_last_ts = rtp_ts
                    self._rtp_probe_anomalies = 0
                    self._sender_ssrc = rtp_ssrc
                    self._rtp_probe_frames = 0
                    import time as _t
                    self.venc_epoch_offset = _t.time()
                    log.info("RTP clock_base=%u, epoch_offset=%.6f",
                             self._rtp_clock_base, self.venc_epoch_offset)

                # Track per-frame RTP timestamps (marker bit = last packet of frame)
                if marker and rtp_ts != self._rtp_probe_last_ts:
                    delta = (rtp_ts - self._rtp_probe_last_ts) & 0xFFFFFFFF
                    self._rtp_probe_frames += 1
                    if delta < 1800 or delta > 7200:  # not ~3603 ticks
                        self._rtp_probe_anomalies += 1
                        delta_ms = delta / 90.0
                        log.debug("RTP_PROBE anomaly: rtp_ts=%u delta=%u (%.3fms) frame=%d",
                                  rtp_ts, delta, delta_ms, self._rtp_probe_frames)

                    # GUARD: correct firmware BIG/NEAR anomalies in the
                    # probe (before leaky queue which drops NEAR frames).
                    # Accumulate shift in ticks; BIG adds ~+3603, NEAR
                    # adds ~-3601 — pairs cancel to ~+2 tick residual.
                    #
                    # When side-door is active: the monotonicity filter
                    # produces delta=1 for stale frames.  The GUARD must
                    # NOT correct delta=1 (it's a real stale marker, not
                    # a firmware anomaly).  But other firmware anomalies
                    # (delta=0, delta~7200) still need correction —
                    # uncorrected anomalies cause false phase-transition
                    # detection in align_block.
                    guard_shift = getattr(self, '_rtp_probe_shift', 0)
                    _sidedoor_active = (hasattr(self, '_pts_stream')
                                        and self._pts_stream is not None)
                    # Track anomaly: this frame AND the next have bad PTS
                    # (the next is clamped by the monotonicity filter).
                    is_anomalous = getattr(self, '_guard_next_anomalous', False)
                    self._guard_next_anomalous = False
                    if delta < 1800:
                        if not (_sidedoor_active and delta == 1):
                            # NEAR = stale PTS read or same-frame NALU.
                            # Do NOT accumulate guard_shift — stale reads
                            # are camera-side artifacts, not real anomalies.
                            # Accumulating causes timestamps to drift and
                            # buffer fill to drop to 0%.
                            log.debug("GUARD_PROBE NEAR d=%u (no shift) shift=%d ticks",
                                      delta, guard_shift)
                    elif delta > 5400 and delta < 9000:
                        # BIG = transport drop (one frame lost), NOT a firmware
                        # anomaly.  The next frame's RTP timestamp is correct.
                        # Do NOT accumulate guard_shift — that makes timestamps
                        # drift and buffer fill climb to 100%.
                        log.debug("GUARD_PROBE BIG d=%u (transport drop, no shift) shift=%d ticks",
                                  delta, guard_shift)
                    self._rtp_probe_shift = guard_shift
                    self._rtp_frame_anomalous = is_anomalous

                    # Log summary every 1000 frames
                    if self._rtp_probe_frames % 1000 == 0:
                        hits = getattr(self, '_rtp_dict_hits', 0)
                        misses = getattr(self, '_rtp_dict_misses', 0)
                        total = hits + misses
                        hit_pct = 100.0 * hits / total if total > 0 else 0
                        sd_hits = getattr(self, '_sidedoor_hits', 0)
                        sd_misses = getattr(self, '_sidedoor_misses', 0)
                        sd_total = sd_hits + sd_misses
                        sd_pct = 100.0 * sd_hits / sd_total if sd_total > 0 else 0
                        pts_str = ""
                        if hasattr(self, '_pts_stream') and self._pts_stream is not None:
                            ps = self._pts_stream
                            pts_str = ", pts_blocks=%d/%d, drops=%d" % (
                                ps.block_aligns,
                                ps.block_aligns + ps.block_fails,
                                ps.drops_detected)
                        log.info("RTP_PROBE: %d frames, %d anomalies (%.2f%%), "
                                 "dict hit=%d/%d (%.1f%%), guard_shift=%d ticks, "
                                 "sidedoor=%d/%d (%.1f%%)%s",
                                 self._rtp_probe_frames, self._rtp_probe_anomalies,
                                 100 * self._rtp_probe_anomalies / self._rtp_probe_frames,
                                 hits, total, hit_pct, guard_shift,
                                 sd_hits, sd_total, sd_pct, pts_str)
                    self._rtp_probe_last_ts = rtp_ts

                    # Store CORRECTED RTP timestamp keyed by GStreamer PTS.
                    # Store on EVERY packet with a new rtp_ts (not just markers)
                    # because the skip counter's FORCE_NEW can change the RTP
                    # timestamp mid-frame. h264parse takes PTS from the first
                    # NALU, so we need the dict entry for that PTS too.
                    corrected_rtp = (rtp_ts - guard_shift) & 0xFFFFFFFF
                    buf_pts = buf.pts
                    if buf_pts is not None and buf_pts != Gst.CLOCK_TIME_NONE:
                        self._rtp_ts_by_pts[buf_pts] = (corrected_rtp,
                                                         is_anomalous)
                        if is_anomalous:
                            log.info("PAD_PROBE: anomaly flag stored for "
                                     "buf_pts=%d rtp=%u", buf_pts, rtp_ts)
                        # Prevent unbounded growth from dropped frames
                        if len(self._rtp_ts_by_pts) > 200:
                            oldest = min(self._rtp_ts_by_pts)
                            del self._rtp_ts_by_pts[oldest]

        except Exception as e:
            log.warning("Failed to read RTP timestamp: %s", e)
        return Gst.PadProbeReturn.OK

    def _storageGuardProbe(self, pad, info):
        """Pad probe on storage branch: enforce monotonic PTS before splitmuxsink.

        Stale shared-page VENC PTS reads produce occasional duplicate RTP
        timestamps that fail splitmuxsink's running-time conversion, causing
        matroskamux to drop buffers.  This probe enforces strictly monotonic
        PTS/DTS so every buffer reaches the muxer with a valid timestamp.
        """
        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK

        # One frame period in nanoseconds: HMAX=4400, VMAX=1350, 148.5MHz
        # (exact 25 fps after VMAX=1350 fix; was 40029630 for VMAX=1351).
        FRAME_NS = 40000000

        pts = buf.pts

        # If PTS is NONE, synthesize from last known PTS
        if pts == Gst.CLOCK_TIME_NONE:
            if self._storage_last_pts_ns > 0:
                pts = self._storage_last_pts_ns + FRAME_NS
                buf.pts = pts
                buf.dts = pts
                self._storage_last_pts_ns = pts
            return Gst.PadProbeReturn.OK

        if pts <= self._storage_last_pts_ns and self._storage_last_pts_ns > 0:
            # Duplicate or backward PTS — bump forward by one frame period
            fixed = self._storage_last_pts_ns + FRAME_NS
            buf.pts = fixed
            buf.dts = fixed
            self._storage_last_pts_ns = fixed
        else:
            self._storage_last_pts_ns = pts
            # Ensure DTS is set (matroskamux may require it)
            if buf.dts == Gst.CLOCK_TIME_NONE:
                buf.dts = pts

        return Gst.PadProbeReturn.OK

    def _busPoller(self):
        """Poll the GStreamer bus and drain queued messages.

        Runs in a background daemon thread:
                - Wakes every 1 s via ``bus.timed_pop_filtered``.
                - Logs any ``ERROR`` or ``WARNING`` message for visibility.
                - Silently discards all other message types to keep the queue small.

        The loop exits when ``_bus_should_exit`` is set or EOS is received.

        Arguments:
            None

        Return:
            None

        """
        if not GST_IMPORTED:
            return

        try:
            # Get bus reference - this keeps the bus (and indirectly the pipeline)
            # alive until this thread exits, preventing premature garbage collection
            pipeline = self.pipeline
            if pipeline is None:
                log.debug("_busPoller: No pipeline, exiting immediately")
                return

            bus = pipeline.get_bus()
            if bus is None:
                log.debug("_busPoller: No bus, exiting immediately")
                return

            mask = Gst.MessageType.ANY

            # Use shorter timeout (1 second) so thread can check exit flag more frequently
            while not self._bus_should_exit:
                msg = bus.timed_pop_filtered(1 * Gst.SECOND, mask)
                if not msg:
                    continue

                if msg.type == Gst.MessageType.ERROR:
                    err, dbg = msg.parse_error()
                    log.error("GST ERROR from %s: %s", msg.src.get_name(), err)
                elif msg.type == Gst.MessageType.WARNING:
                    warn, dbg = msg.parse_warning()
                    log.warning("GST WARN  from %s: %s", msg.src.get_name(), warn)
                elif msg.type == Gst.MessageType.EOS:
                    log.debug("_busPoller: Received EOS, exiting")
                    break

        except Exception as e:
            log.debug("_busPoller: Exception occurred: %s", str(e))
        finally:
            # Explicitly release local references to allow GC to free the pipeline
            # These would go out of scope anyway, but being explicit ensures cleanup
            # happens promptly even if this thread is slow to fully exit
            pipeline = None
            bus = None
            log.debug("_busPoller: Thread exiting")


    def createGstreamDevice(self, video_format, gst_decoder='decodebin', 
                            video_file_dir=None, segment_duration_sec=30, max_retries=5, retry_interval=5):
        """
        Creates a GStreamer pipeline for capturing video from an RTSP source and 
        initializes playback with specific configurations.

        The method also sets an initial timestamp for the pipeline's operation.

        Arguments:
            video_format: [str] The desired video format for the conversion, 
                e.g., 'BGR', 'GRAY8', etc.
            
        Keyword arguments:
            gst_decoder: [str] The gst_decoder to use for the Gstreamer video stream. Default is 'decodebin'.
            video_file_dir: [str] The directory where the raw video stream should be saved. 
                If None, the raw stream will not be saved to disk. Default is None.
            segment_duration_sec: [int] The duration of each video segment in seconds. 
                Default is 30.
            max_retries: [int] The maximum number of retry attempts
            retry_interval: [float] The number of seconds to wait between retries

        Returns:
            Gst.Element: The appsink element of the created GStreamer pipeline, 
                which can be used for further processing of the captured video frames.
        """

        device_url = self.extractRtspUrl(self.config.deviceID)

        # VENC PTS calibration is deferred until after the RTSP stream starts.
        # The VENC PTS register is a 32-bit µs counter that wraps every ~71.6 min.
        # Pre-pipeline calibration can catch PTS near a wrap boundary; by the time
        # the RTSP stream starts (hundreds of ms later), PTS may have wrapped,
        # putting calibration C and clock_base in different wrap cycles (~71 min
        # epoch_offset error).  Deferred calibration runs after frames are flowing,
        # guaranteeing calibration PTS and clock_base are in the same wrap cycle.
        self.venc_epoch_offset = None
        self._venc_pts_clock_offset = None  # C: PTS-to-wallclock constant
        self._rtp_clock_base = None         # First RTP timestamp from pipeline
        self._venc_post_start_calibrated = False
        self._venc_drift_baseline_age = None  # Drift correction baseline
        self._venc_drift_total = 0.0          # Cumulative correction for logging
        if getattr(self.config, 'venc_gettime_port', 0) > 0:
            log.info("VENC calibration will run after stream starts (deferred)")

            # Start metadata stream reader (port 9602) for per-frame
            # exposure/gain data.  Replaces SEI in-band metadata with
            # out-of-band TCP stream — no encoder ioctls needed.
            from urllib.parse import urlparse
            camera_ip = urlparse(device_url).hostname
            meta_port = getattr(self.config, 'venc_meta_port', 9602)
            if not hasattr(self, '_venc_meta') or self._venc_meta is None:
                self._venc_meta = VencMetadataReader(camera_ip, port=meta_port)
                self._venc_meta.start()
                log.info("VENC metadata reader started on port %d", meta_port)

            # Start PTS stream reader (port 9603) for fresh VENC hardware
            # timestamps — independent /dev/mem reads, no stale PTS race.
            pts_port = getattr(self.config, 'venc_pts_stream_port', 9603)
            if not hasattr(self, '_pts_stream') or self._pts_stream is None:
                gt_port = getattr(self.config, 'venc_gettime_port', 9601)
                drift = getattr(self, '_drift_freq', 0.0)
                drift_path = self._driftFreqPath()
                self._pts_stream = PtsStreamReader(camera_ip, port=pts_port,
                                                    gettime_port=gt_port,
                                                    drift_freq=drift,
                                                    drift_freq_path=drift_path)
                self._pts_stream.start()
                log.info("PTS stream reader started on port %d", pts_port)

        # Common timeout settings for both UDP and TCP
        # All streams need tcp-timeout since RTSP control always uses TCP
        common_timeouts = "retry=5 timeout=5000000 tcp-timeout=5000000 teardown-timeout=3000000"
        
        if self.config.protocol == 'udp':
            protocol_str = f"protocols=udp udp-buffer-size=16777216 {common_timeouts}"
        else:
            protocol_str = f"protocols=tcp {common_timeouts}"

        # Define the source up to the point where we want to branch off
        source_to_tee = (
            "rtspsrc name=src buffer-mode=0 ntp-sync=false latency=500 {:s} "
            "location=\"{:s}\" ! "
            "rtph264depay name=depay ! video/x-h264,stream-format=byte-stream,alignment=nal ! h264parse ! tee name=t"
            ).format(protocol_str, device_url)

        # Branch for processing
        processing_branch = (
            "t. ! queue ! {:s} ! "
            "queue leaky=downstream max-size-buffers=100 max-size-bytes=0 max-size-time=0 ! "
            "videoconvert ! video/x-raw,format={:s} ! "
            "queue max-size-buffers=100 max-size-bytes=0 max-size-time=0 ! "
            "appsink max-buffers={:d} drop=true sync=0 name=appsink"
            ).format(gst_decoder, video_format, 100)
        self._appsink_max_buffers = 100
        
         # Branch for storage - if video_file_dir is not None, save the raw stream to a file
        if video_file_dir is not None:
            
            # The video will be split into segments of segment_duration_sec seconds
            # The splitmuxsink will save the segments to video_file_dir
            # The splitmuxsink will use the matroskamux muxer
            # The splitmuxsink will use the format-location-full signal to name and move each segment
            # queue2 smooths out the writes, but doesn't wait until the buffers fill up for writing
            storage_branch = (
                "t. ! queue2 max-size-buffers=150 max-size-bytes=2097152 max-size-time=5000000000 ! "
                "h264parse name=storageparse ! "
                "splitmuxsink name=splitmuxsink0 async-finalize=true max-size-time={:d} muxer-factory=matroskamux"
                ).format(int(segment_duration_sec*1e9))

        # Otherwise, skip saving the raw stream to disk
        else:
            storage_branch = ""

         # Combine all parts of the pipeline
        pipeline_str = "{:s} {:s} {:s}".format(source_to_tee, processing_branch, storage_branch)

        # Obfuscate the password in the pipeline string before logging
        obfuscated_pipeline_str = obfuscatePassword(pipeline_str)

        log.info("GStreamer pipeline string: {:s}".format(obfuscated_pipeline_str))

        # Set the pipeline to PLAYING state with retries
        for attempt in range(max_retries):
            try:
                log.info("Attempt {}: transitioning Pipeline to PLAYING state.".format(attempt + 1))
                
                # Make sure any previous pipeline is cleaned up
                if hasattr(self, 'pipeline') and self.pipeline:
                    self.releaseResources()

                # Parse and create the pipeline
                self._bus_should_exit = False
                self.pipeline = Gst.parse_launch(pipeline_str)
                if not self.pipeline:
                    raise ValueError("Could not create pipeline")
                
                # Start a daemon thread that drains the GstBus so it never fills
                self._bus_thread = threading.Thread(target=self._busPoller, daemon=True)
                self._bus_thread.start()
                
                # Add pad probe to capture first RTP timestamp for precise
                # PTS-to-wallclock mapping (sub-ms accuracy).
                # Always add when VENC gettime is configured — clock_base is
                # needed by both pre-pipeline and deferred calibration paths.
                if getattr(self.config, 'venc_gettime_port', 0) > 0:
                    depay = self.pipeline.get_by_name("depay")
                    if depay:
                        sink_pad = depay.get_static_pad("sink")
                        sink_pad.add_probe(
                            Gst.PadProbeType.BUFFER,
                            self._onFirstRtpBuffer
                        )

                    # RTCP SR disabled: XM firmware truncates NTP fractional
                    # seconds. cam_wall re-anchoring via PtsStream is used instead.

                # If raw video saving is enabled, connect the "format-location-full" signal to the
                # moveSegment function
                if video_file_dir is not None:
                    
                    splitmuxsink = self.pipeline.get_by_name("splitmuxsink0")
                    splitmuxsink.connect("format-location-full", self.moveSegment)

                    # Add PTS guard probe before splitmuxsink to fix
                    # stale-PTS duplicates before they reach matroskamux
                    storageparse = self.pipeline.get_by_name("storageparse")
                    if storageparse:
                        storageparse.get_static_pad("src").add_probe(
                            Gst.PadProbeType.BUFFER,
                            self._storageGuardProbe
                        )
                        log.info("Storage PTS guard probe attached to storageparse src pad")

                # Transition through states
                log.info("Starting pipeline state transitions...")

                # First transition to PAUSED to capture start_time
                success, start_time = self.handleStateChange(self.pipeline, Gst.State.PAUSED)
                if not success:
                    raise ValueError("Failed to transition pipeline to PAUSED state")

                # Calculate start timestamp BEFORE going to PLAYING
                # This ensures splitmuxsink has correct timing reference when it creates first segment
                if start_time is not None:
                    self.start_timestamp = start_time - (self.config.camera_buffer/self.config.fps + self.config.camera_latency)

                # Now transition to PLAYING
                success, _ = self.handleStateChange(self.pipeline, Gst.State.PLAYING)
                if not success:
                    raise ValueError("Failed to transition pipeline to PLAYING state")

                # Log start time
                start_time_str = (UTCFromTimestamp.utcfromtimestamp(self.start_timestamp)
                                    .strftime('%Y-%m-%d %H:%M:%S.%f'))

                log.info("Start time is {:s}".format(start_time_str))

                # Get appsink for frame retrieval
                appsink = self.pipeline.get_by_name("appsink")
                if not appsink:
                    raise ValueError("Could not get appsink from pipeline")

                log.info("Pipeline successfully created and started")
                return appsink
            
            except Exception as e:
                log.error("Attempt {} failed: {}".format(attempt + 1, str(e)))
                # Clean up any partial pipeline that was created
                if hasattr(self, 'pipeline') and self.pipeline:
                    try:
                        self.pipeline.set_state(Gst.State.NULL)
                        self.pipeline = None
                    except Exception as cleanup_e:
                        log.error("Error cleaning up failed pipeline: {}".format(cleanup_e))
                        
                if attempt < max_retries - 1:
                    log.info("Waiting {} seconds before next attempt...".format(retry_interval))
                    time.sleep(retry_interval)
                    continue
                else:
                    log.error("All attempts to create pipeline failed")
                    self.releaseResources()
                    return False
        return False


    def initVideoDevice(self):
        """ Initialize the video device. """

        # Assume OpenCV as the default video device type, which will be overridden if GStreamer is used
        self.video_device_type = "cv2"

        # Use a file as the video source
        if self.video_file is not None:

            # If the video file is a GStreamer file, use the GstVideoFile class
            if GST_IMPORTED and (self.config.media_backend == 'gst'):

                self.device = GstVideoFile(self.video_file, decoder=self.config.gst_decoder,
                                           video_format=self.config.gst_colorspace)

            # Fall back to OpenCV if GStreamer is not available
            else:
                self.device = cv2.VideoCapture(self.video_file)

        # Use a device as the video source
        else:

            reprobe = False

            # If an analog camera is used, skip the probe
            if "rtsp" in str(self.config.deviceID):
                success, probe_result = self.probeRtspService()
                if not success:
                    error_messages = {
                        RtspProbeResult.NETWORK_DOWN: 
                            "Cannot connect to camera - Please check your network connection",
                        RtspProbeResult.HOST_UNREACHABLE: 
                            "Cannot reach camera - Please check if camera is powered on and connected to network",
                        RtspProbeResult.CONNECTION_REFUSED: 
                            "Camera is reachable but RTSP service is not responding - Camera might still be booting",
                        RtspProbeResult.TIMEOUT: 
                            "Connection timeout - Network might be slow or unstable",
                        RtspProbeResult.DNS_ERROR: 
                            "Cannot resolve camera hostname - Please check network DNS settings",
                        RtspProbeResult.UNKNOWN_ERROR: 
                            "Unknown connection error - Please check logs for details"
                    }
                    log.error("Camera connection failed: {}".format(error_messages[probe_result]))
                    return False
                else:
                    # After camera connection is established, if necessary inititliaze camera settings
                    # and/or perform camera mode change

                    # initialize flag to indicate if camera should be reprobed after mode change
                    reprobe = False

                    # ------------------------------------------------------------------
                    # One-time camera initialization with flag file in rms_root_dir
                    # ------------------------------------------------------------------
                    root_dir  = self.config.rms_root_dir

                    # e.g.  "XX0001.camera_init.done"
                    flag_file = os.path.join(root_dir, "{}.camera_init.done".format(self.config.stationID))

                    if self.config.initialize_camera and not os.path.exists(flag_file):
                        log.info("Running camera init sequence ...")
                        reprobe = True
                        try:
                            mode_name = "init"
                            mode_path = self.config.camera_settings_path

                            if not os.path.exists(mode_path):
                                raise FileNotFoundError("Mode file {} not found.".format(mode_path))

                            with open(mode_path, 'r') as f:
                                modes = json.load(f)

                            if mode_name not in modes:
                                raise KeyError("Mode '{}' not defined in {}.".format(mode_name, mode_path))

                            try:
                                cc.cameraControlV2(self.config, "SwitchMode", mode_name)

                                # create empty sentinel file
                                open(flag_file, "a").close()
                                log.info("Init complete - flag written to %s", flag_file)

                            except Exception as e:
                                raise RuntimeError("Failed to switch camera mode: {}".format(e))

                        except Exception as e:
                            log.warning("Camera switch to %s mode failed: %s. Will retry later.", mode_name, e)

                    # -------------------------------------------
                    # Day/night switching
                    # -------------------------------------------
                    if self.config.continuous_capture and self.config.switch_camera_modes:
                        if self.camera_mode_switch_trigger.value:
                            reprobe = True
                            switchCameraMode(self.config, self.daytime_mode, self.camera_mode_switch_trigger)

            if reprobe:
                # Wait 5 seconds for the camera to register all commands after mode switching / reboot
                log.info("Waiting for camera to register all commands...")
                time.sleep(5)
                success, probe_result = self.probeRtspService()
                if not success:
                    log.error("Camera connection failed after switching modes: {}".format(probe_result))
                    return False

            # Init the video device
            log.info("Initializing the video device...")
            log.info("Device: " + str(self.config.deviceID))

            # If media backend is set to gst, but GStreamer is not available, switch to openCV
            if (self.config.media_backend == 'gst') and (not GST_IMPORTED):
                log.info("GStreamer is not available. Switching to alternative.")
                self.media_backend_override = True

            if (self.config.media_backend == 'gst') and GST_IMPORTED and (self.media_backend_override == False):
                
                log.info("Initialize GStreamer Standalone Device.")
                
                # Initialize Smoothing parameters
                self.reset_count += 1
                self.n = 0
                self.sum_x = 0
                self.sum_y = 0
                self.sum_xx = 0
                self.sum_xy = 0
                self.startup_frames = 25*60*10 # 10 minutes
                self.b = 0
                self.b_error_debt = 0
                self.m_jump_error = 0
                self.last_m_err = float('inf')
                self.last_m_err_n = 0
                self.last_pts_correction_ns = 0
                self.last_running_time_ns = None


                try:

                    # Initialize GStreamer (only if not already initialized)
                    if not Gst.is_initialized():
                        Gst.init(None)

                    # Determine if which directory to save the raw video, if any
                    if self.config.raw_video_save:
                        raw_video_dir = os.path.join(self.config.data_dir, self.config.video_dir)
                    else:
                        raw_video_dir = None

                    # Create and start a GStreamer pipeline
                    log.info("Creating GStreamer pipeline...")
                    venc_retry = 15 if getattr(self.config, 'venc_gettime_port', 0) > 0 else 5
                    self.device = self.createGstreamDevice(
                        self.config.gst_colorspace, gst_decoder=self.config.gst_decoder,
                        video_file_dir=raw_video_dir, segment_duration_sec=self.config.raw_video_duration,
                        retry_interval=venc_retry
                        )

                    if not self.device:
                        raise ValueError("Could not create GStreamer pipeline.")
                    
                    log.info("GStreamer pipeline created!")   
                    
                    # Reset presentation time stamp buffer
                    self.pts_buffer = []

                    # Attempt to get a sample and determine the frame shape
                    # Use a longer timeout for the initial sample - at high bitrate (e.g. 51 Mbps
                    # GOP=1), the pipeline needs more time for RTSP negotiation + jitter buffer
                    # fill + first I-frame decode
                    sample = self.device.emit("try-pull-sample", 5000 * Gst.MSECOND)
                    if not sample:
                        raise ValueError("Could not obtain sample.")

                    buffer = sample.get_buffer()
                    ret, map_info = buffer.map(Gst.MapFlags.READ)
                    if not ret:
                        raise ValueError("Could not obtain frame.")

                    # Extract video information from caps
                    caps = sample.get_caps()
                    if not caps:
                        raise ValueError("Sample caps are None.")
                        
                    structure = caps.get_structure(0)
                    if not structure:
                        raise ValueError("Could not determine frame shape.")
                    
                    # Extract width, height, and format, and create frame
                    width = getStructureValue(structure, 'width')
                    height = getStructureValue(structure, 'height')

                    if self.config.gst_colorspace == 'GRAY8':
                        self.frame_shape = (height, width)
                    else:
                        self.frame_shape = (height, width, 3)

                    frame = np.ndarray(shape=self.frame_shape, buffer=map_info.data, dtype=np.uint8)

                    # Unmap the buffer
                    buffer.unmap(map_info)
                    
                    # Check if frame is grayscale and set flag
                    # gray_result = self.isGrayscale(frame)
                    # if gray_result is not None:
                    #     self.convert_to_gray = gray_result
                    pass
                    log.info("Video format: {}, {}P, color: {}".format(self.config.gst_colorspace, height, 
                                                                       not self.convert_to_gray))

                    # Set the video device type
                    self.video_device_type = "gst"

                    conn = getObsDBConn(self.config)
                    try:
                        addObsParam(conn, "media_backend", self.video_device_type)
                    finally:
                        conn.close()

                    return True

                except Exception as e:
                    log.info("Error initializing GStreamer, switching to alternative. Error: {}".format(e))
                    self.media_backend_override = True
                    self.releaseResources()

                    conn = getObsDBConn(self.config)
                    try:
                        addObsParam(conn, "media_backend", self.video_device_type)
                    finally:
                        conn.close()

            if self.config.media_backend == 'v4l2':
                try:
                    log.info("Initialize OpenCV Device with v4l2.")
                    self.device = cv2.VideoCapture(self.config.deviceID, cv2.CAP_V4L2)
                    self.device.set(cv2.CAP_PROP_CONVERT_RGB, 0)

                    return True
                
                except Exception as e:
                    log.info("Could not initialize OpenCV with v4l2. Initialize "
                             "OpenCV Device without v4l2 instead. Error: {}".format(e))
                    self.media_backend_override = True
                    self.releaseResources()


            elif (self.config.media_backend == 'cv2') or self.media_backend_override:
                log.info("Initialize OpenCV Device.")
                self.device = cv2.VideoCapture(self.config.deviceID)

                return True

            else:
                error_msg  = "Invalid media backend: {}\n".format(self.config.media_backend)
                error_msg += "Or GStreamer is not available but is set as the media_backend."
                raise ValueError(error_msg)

        return False


    def releaseResources(self):
        """Tear everything down in the right order so no FD survives."""
        
        # Prevent multiple simultaneous calls
        if hasattr(self, '_releasing_resources') and self._releasing_resources:
            log.debug("releaseResources: Already in progress, skipping duplicate call")
            return
        self._releasing_resources = True

        log.debug("releaseResources: Starting")

        # Stop metadata reader if running
        if hasattr(self, '_venc_meta') and self._venc_meta is not None:
            self._venc_meta.stop()
            self._venc_meta = None

        # Reset timestamp correction whenever resources are released
        self.last_pts_correction_ns = 0
        self.last_running_time_ns = None

        def _timedCall(fn, timeout_s=2):
            """Run *fn()* in a daemon thread and wait *timeout_s*.
            Returns True if the call finished in time."""
            th = threading.Thread(target=fn, daemon=True)
            th.start()
            th.join(timeout_s)
            return not th.is_alive()

        # stop frame-saver children
        log.debug("releaseResources: Calling releaseRawArrays()")
        self.releaseRawArrays()
        log.debug("releaseResources: releaseRawArrays() completed")

        # gracefully drain & stop the pipeline
        if self.pipeline:
            log.debug("releaseResources: Pipeline exists, starting shutdown")
            
            # Disconnect any signal handlers first
            try:
                splitmuxsink = self.pipeline.get_by_name("splitmuxsink0")
                if splitmuxsink:
                    log.debug("releaseResources: Disconnecting splitmuxsink signals")
                    splitmuxsink.disconnect_by_func(self.moveSegment)
            except Exception as e:
                log.debug("releaseResources: Error disconnecting signals: %s", e)
            
            bus = self.pipeline.get_bus()
            
            # Try graceful shutdown first
            log.debug("releaseResources: Attempting graceful shutdown")
            try:
                log.debug("releaseResources: Sending EOS event")
                self.pipeline.send_event(Gst.Event.new_eos())
                
                log.debug("releaseResources: Waiting for EOS/ERROR (2 second timeout)")
                msg = bus.timed_pop_filtered(2*Gst.SECOND,
                                    Gst.MessageType.EOS | Gst.MessageType.ERROR)
                log.debug(f"releaseResources: timed_pop_filtered returned: {msg}")

                log.debug("releaseResources: Setting pipeline to NULL state")
                ret = self.pipeline.set_state(Gst.State.NULL)
                log.debug(f"releaseResources: set_state returned: {ret}")
                
                log.debug("releaseResources: Getting pipeline state (2 second timeout)")
                ret, state, pending = self.pipeline.get_state(2*Gst.SECOND)
                log.debug(f"releaseResources: get_state returned: ret={ret}, state={state}, pending={pending}")
                
                # Check if we actually reached NULL state
                if state != Gst.State.NULL:
                    log.warning("releaseResources: Graceful shutdown failed, pipeline stuck in state %s", state)
                    raise Exception("Pipeline stuck, forcing shutdown")
                    
                log.debug("releaseResources: Graceful shutdown successful")
                
            except Exception as e:
                log.warning("releaseResources: Graceful shutdown failed (%s), forcing pipeline shutdown", e)
                
                # Force shutdown - just set to NULL without waiting
                log.debug("releaseResources: Force setting pipeline to NULL state")
                self.pipeline.set_state(Gst.State.NULL)
                # Don't wait for state change - just proceed with cleanup

            # wake poller
            if bus:
                log.debug("releaseResources: Posting EOS to wake poller")
                bus.post(Gst.Message.new_eos(None))

        else:
            log.debug("releaseResources: No pipeline to shutdown")

        # Clear pipeline reference. Note: the bus poller thread holds its own local
        # reference to the pipeline, so it won't be garbage collected until that
        # thread exits. We set _bus_should_exit=True below to signal the thread to stop.
        self.pipeline = None

        # shut down the poller
        if self._bus_thread and self._bus_thread.is_alive():
            log.debug(f"releaseResources: Bus thread is alive (thread={self._bus_thread})")
            self._bus_should_exit = True

            log.debug("releaseResources: Joining bus thread (6 second timeout)")
            self._bus_thread.join(timeout=6)

            if self._bus_thread.is_alive():
                log.debug("releaseResources: WARNING - Bus thread still alive after timeout!")
            else:
                log.debug("releaseResources: Bus thread joined successfully")
        self._bus_thread = None

        # NOTE: Do NOT call pipeline.unref() manually! Python's GI bindings handle
        # reference counting automatically. Calling unref() manually causes double-free
        # crashes when Python's garbage collector also tries to free the object.
        # The pipeline will be cleaned up when all Python references go out of scope.

        # device section
        if self.device:
            log.debug("releaseResources: Releasing %s", type(self.device).__name__)

            try:
                if hasattr(self.device, "release"):                      # OpenCV branch
                    if not _timedCall(self.device.release):
                        log.warning("releaseResources: cap.release() hung - fd dropped")

                # For GStreamer devices (AppSink), cleanup happens automatically when
                # the pipeline is garbage collected. Don't try to access the device
                # as it may have already been freed with the pipeline.
                elif self.video_device_type == "gst":
                    log.debug("releaseResources: GStreamer device cleaned up with pipeline")

                else:                                                    # Fallback
                    log.debug("releaseResources: Unknown device type - just dropping ref")

            finally:
                self.device = None

        log.debug("releaseResources: Completed")

        # Force garbage collection to break any reference cycles and ensure
        # GStreamer resources are freed. This replaces the manual unref() call
        # which caused double-free crashes.
        gc.collect()

        # Reset the flag so future calls can proceed
        self._releasing_resources = False


    def releaseRawArrays(self):
        """Clean up raw frame arrays and saver."""
        if self.raw_frame_saver:
            try:
                self.raw_frame_saver.stop()
                self.raw_frame_saver.join(5)
                if self.raw_frame_saver.is_alive():
                    log.warning("RawFrameSaver still busy. Sending interrupt signal...")
                    try:
                        if self.raw_frame_saver.pid:
                            os.kill(self.raw_frame_saver.pid, signal.SIGINT)
                        
                        # Wait for graceful shutdown
                        self.raw_frame_saver.join(3)
                        
                        if self.raw_frame_saver.is_alive():
                            log.warning("RawFrameSaver still alive after interrupt, forcing termination")
                            self.raw_frame_saver.terminate()
                            self.raw_frame_saver.join()
                        else:
                            log.info("RawFrameSaver exited gracefully after interrupt")
                            
                    except ProcessLookupError:
                        log.info("RawFrameSaver already terminated")
                    except Exception as e:
                        log.error("Error during graceful RawFrameSaver shutdown: {}".format(e))
                        self.raw_frame_saver.terminate()
                        self.raw_frame_saver.join()
            finally:
                self.raw_frame_saver = None

        # Clean up array resources
        self.current_raw_frame_shape = None
        self.shared_raw_array = None
        
        # Safely delete shared memory arrays
        for name in ("shared_raw_array_base", "shared_raw_array", "shared_raw_array_base2", "shared_raw_array2"):
            if hasattr(self, name):
                delattr(self, name)


    def initRawFrameArrays(self, frame_shape):
        """Initialize raw frame arrays based on current frame shape.
        
        Arguments:
            frame_shape: tuple of frame dimensions
        """
        try:
            # Clean up any existing arrays first
            self.releaseRawArrays()

            # Calculate buffer size based on actual dimensions
            if len(frame_shape) == 3:
                buffer_size = self.num_raw_frames * frame_shape[0] * frame_shape[1] * frame_shape[2]
                array_shape = (self.num_raw_frames, frame_shape[0], frame_shape[1], frame_shape[2])
            else:
                buffer_size = self.num_raw_frames * frame_shape[0] * frame_shape[1]
                array_shape = (self.num_raw_frames, frame_shape[0], frame_shape[1])

            log.debug("Creating shared arrays with shape: {}".format(array_shape))

            # Initialize shared memory arrays
            self.shared_raw_array_base = Array(ctypes.c_uint8, buffer_size)
            self.shared_raw_array = np.ctypeslib.as_array(self.shared_raw_array_base.get_obj())
            self.shared_raw_array = self.shared_raw_array.reshape(array_shape)

            self.shared_raw_array_base2 = Array(ctypes.c_uint8, buffer_size)
            self.shared_raw_array2 = np.ctypeslib.as_array(self.shared_raw_array_base2.get_obj())
            self.shared_raw_array2 = self.shared_raw_array2.reshape(array_shape)

            # Store current array configuration
            self.current_raw_frame_shape = frame_shape
            self.current_mode = self.daytime_mode.value if self.daytime_mode is not None else False
            
            return True

        except Exception as e:
            log.error("Failed to initialize raw frame arrays: {}".format(e))
            log.debug(repr(traceback.format_exception(*sys.exc_info())))
            return False


    def run(self):
        """ Main process function - initializes all process-specific resources and runs capture loop.
        """
        try:
            log.debug("Initializing process-specific resources...")

            # Initialize heartbeat for watchdog
            self.heartbeat.value = time.time()

            # GStreamer debug setup
            if GST_IMPORTED:
                try:
                    # Activate debug system
                    Gst.debug_set_active(True)

                    # Set debug level from environment or default given value
                    # The Gst debug level is set in Logger.py
                    debug_env = os.environ.get("GST_DEBUG", "2")
                    Gst.debug_set_default_threshold(int(debug_env))

                    # Route all GStreamer debug output through our Python handler
                    # (which filters noisy warnings like "decreasing timestamp").
                    # Remove the default stderr handler first to avoid duplicates.
                    Gst.debug_remove_log_function(None)
                    Gst.debug_add_log_function(gstDebugLogger, None)

                    log.info("GStreamer logging initialized at level: {}".format(debug_env))

                except Exception as e:
                    log.error("Failed to initialize GStreamer logging: {}".format(e))

            # Initialize process-specific variables
            self.media_backend_override = False
            self.video_device_type = "cv2"
            self.time_for_drop = 1.5*(1.0/self.config.fps)
            # VENC mode uses raw GStreamer PTS (no smoothPTS regression), which
            # exposes real frame losses that smoothPTS hides by interpolation.
            # At high bitrates (61Mbps), femac TX FIFO overflows cause occasional
            # 3-8 frame losses (~120-320ms gaps). These are real network drops,
            # not timestamp artifacts. 10x frame period threshold flags only
            # sustained outages (>400ms = likely RTSP failure, not transient loss).
            self.time_for_drop_venc = 10.0*(1.0/self.config.fps)
            self.device = None
            self.pipeline = None
            self.start_timestamp = 0
            self.frame_shape = None
            self.convert_to_gray = True  # Force grayscale — IMX307 IR cameras are always mono
            self.last_pts_correction_ns = 0
            self.last_running_time_ns = None

            # VENC PTS calibration: if set, bypasses smoothPTS and pipeline clock
            # correction entirely, using hardware sensor timestamps instead.
            self.venc_epoch_offset = None
            self._venc_pts_clock_offset = None
            self._rtp_clock_base = None
            self._venc_post_start_calibrated = False
            self._rtcp_calibrated = False
            self._last_venc_timestamp = 0.0
            self._last_venc_gst_pts = 0
            self._last_gst_delta_ms = 40.0
            self._rtp_ts_by_pts = {}
            self._drift_freq = self._loadDriftFreq()
            # Reset delivery tracking for fresh calibration
            self._last_delivery_time = 0
            self._last_delivery_rtp = 0
            self._rtp_prev_raw = 0
            # Reset drift discipline state
            self._venc_drift_baseline_age = None
            self._venc_drift_total = 0.0
            # Full PtsStream restart — kills threads, clears all stale
            # state (entries, C_raw, wraps), reconnects fresh.
            if hasattr(self, '_pts_stream') and self._pts_stream is not None:
                self._pts_stream.restart()
            self._drift_samples = []
            self._drift_baseline_age = None
            self._drift_t0 = None
            self._drift_log_accum = 0.0
            self._drift_p_correction_per_frame = 0.0
            self._drift_total_correction = 0.0

            # Initialize smoothing variables
            self.startup_flag = True
            self.last_calculated_fps = 0
            self.last_calculated_fps_n = 0
            self.expected_m = 1e9/self.config.fps
            self.reset_count = -1
            self.n = 0
            self.sum_x = 0
            self.sum_y = 0
            self.sum_xx = 0
            self.sum_xy = 0
            self.startup_frames = 25*60*10  # 10 minutes
            self.b = 0
            self.b_error_debt = 0
            self.m_jump_error = 0
            self.last_m_err = float('inf')
            self.last_m_err_n = 0
            self.current_raw_frame_shape = None
            self.current_mode = None

            # Initialize raw frame handling if enabled
            if self.config.save_frames:
                self.raw_frame_count = 0
                
                # Convert shared timestamp arrays to numpy arrays
                self.sharedTimestamps = np.ctypeslib.as_array(self.shared_timestamps_base.get_obj())
                self.sharedTimestamps2 = np.ctypeslib.as_array(self.shared_timestamps_base2.get_obj())

                # Raw frame arrays will be initialized after we know the frame shape
                self.shared_raw_array_base = None
                self.shared_raw_array = None
                self.shared_raw_array_base2 = None
                self.shared_raw_array2 = None
                self.raw_frame_saver = None

            # Initialize timestamp array for ft file buffer
            if self.config.save_frame_times:
                self.timestamp_buffer = []
                # For testing ft files
                # self.ft_test_time = time.time()

            # Initialize segment saving time for raw video saving
            if self.config.raw_video_save:
                self.last_segment_savetime = time.time()

            log.debug("Process-specific initialization complete")

            # Main capture loop
            while not self.exit.is_set() and not self.initVideoDevice():
                # Update heartbeat during connection attempts to show we're still alive
                self.heartbeat.value = time.time()
                log.info('Waiting for the video device to be connected...')
                time.sleep(5)

            if self.device is None:
                log.info('The video source could not be opened!')
                self.exit.set()
                return False

            # Continue with main capture loop
            self.captureFrames()

        except KeyboardInterrupt:
            log.info("Capture process received interrupt signal. Shutting down gracefully...")
            self.exit.set()
        except Exception as e:
            log.error("Error in capture process: {}".format(e))
            log.debug(repr(traceback.format_exception(*sys.exc_info())))
            self.exit.set()
        finally:
            self.releaseResources()



    def captureFrames(self):
        """ Main frame capture loop - moved from run() for clarity """

        # Keep track of the total number of frames
        total_frames = 0

        # Timestamp of the very first good frame - becomes the run's origin
        run_start_ts = None

        # For video devices only (not files), throw away the first 10 frames
        if (self.video_file is None) and (self.video_device_type == "cv2"):

            first_skipped_frames = 10
            for i in range(first_skipped_frames):
                _, _, ts, _ = self.read()
                if run_start_ts is None:
                    run_start_ts = ts  

            total_frames = first_skipped_frames

        # If a video file was used, set the time of the first frame to the time read from the file name
        if self.video_file is not None:
            time_stamp = "_".join(os.path.basename(self.video_file).split("_")[1:4])
            time_stamp = time_stamp.split(".")[0]
            video_first_time = datetime.datetime.strptime(time_stamp, "%Y%m%d_%H%M%S_%f")
            log.info("Using a video file: " + self.video_file)
            log.info("Setting the time of the first frame to: " + str(video_first_time))

            # Convert the first time to a UNIX timestamp
            video_first_timestamp = (video_first_time - datetime.datetime(1970, 1, 1)).total_seconds()

        # Use the first frame buffer to start - it will be flip-flopped between the first and the second
        #   buffer during capture, to prevent any data loss
        buffer_one = True

        wait_for_reconnect = False

        last_frame_timestamp = False

        # Setup additional timing variables for memory share with RawFrameSaver
        if self.config.save_frames:
            raw_buffer_one = True
            first_raw_frame_timestamp = False


        # Run until stopped from the outside
        while not self.exit.is_set():

            # Wait until the compression is done (only when a video file is used)
            if self.video_file is not None:
                
                wait_for_compression = False

                if buffer_one:
                    if self.start_time1.value == -1:
                        wait_for_compression = True
                else:
                    if self.start_time2.value == -1:
                        wait_for_compression = True

                if wait_for_compression:
                    log.debug("Waiting for the {:d}. compression thread to finish...".format(int(not buffer_one) + 1))
                    time.sleep(0.1)
                    continue

            
            if buffer_one:
                self.start_time1.value = 0
            else:
                self.start_time2.value = 0
            

            # If the video device was disconnected, wait 5s for reconnection
            if wait_for_reconnect:

                print('Reconnecting...')

                while not self.exit.is_set() and not self.initVideoDevice():

                    # Update heartbeat during reconnection attempts to show we're still alive
                    self.heartbeat.value = time.time()

                    log.info('Waiting for the video device to be reconnected...')

                    time.sleep(5)

                    if self.device is None:
                        print("The video device couldn't be connected! Retrying...")
                        continue


                    if self.exit.is_set():
                        break

                    # Read the frame
                    log.info("Reading frame...")
                    ret, frame, _, _ = self.read()
                    log.info("Frame read!")

                    # If the connection was made and the frame was retrieved, continue with the capture
                    if ret:
                        log.info('Video device reconnected successfully!')
                        wait_for_reconnect = False
                        break


                wait_for_reconnect = False


            t_frame = 0
            t_assignment = 0
            t_convert = 0
            t_block = time.time()
            max_frame_interval_normalized = 0.0
            max_frame_age_seconds = 0.0
            first_frame_timestamp = None

            # running totals for mean calculations
            sum_frame_interval_norm = 0.0
            sum_frame_age_seconds   = 0.0

            # Capture a block of 256 frames
            block_frames = 256

            # Check if camera needs switching
            if self.config.continuous_capture and self.config.switch_camera_modes:

                # Check that the camera mode switch is triggered
                if self.camera_mode_switch_trigger.value:
                    
                    # If the camera mode switch trigger is set, switch the camera mode
                    switchCameraMode(self.config, self.daytime_mode, self.camera_mode_switch_trigger)


            log.info('Grabbing a new block of {:d} frames...'.format(block_frames))

            # Update heartbeat timestamp for watchdog to detect hangs
            self.heartbeat.value = time.time()

            for i in range(block_frames):

                # Read the frame (keep track how long it took to grab it), and check for color if saving raw frame
                t1_frame = time.time()
                ret, frame, frame_timestamp, frame_gst_pts_ns = self.read()
                t_frame = time.time() - t1_frame

                # If the video device was disconnected, wait for reconnection
                if (self.video_file is None) and (not ret):
                    log.info('Frame grabbing failed, video device is probably disconnected!')
                    self.releaseResources()
                    # Kill PtsStream so run() creates a fresh one
                    if hasattr(self, '_pts_stream') and self._pts_stream is not None:
                        self._pts_stream.stop()
                        self._pts_stream = None
                    # Return to run() for full re-init (identical to first start)
                    return

                # Set flag to save a raw frame
                save_this_frame = (self.config.save_frames and
                                   self.video_file is None and
                                   self.shouldSaveFrame(frame_timestamp)
                                   )

                # Check if frame contains color information
                if save_this_frame:
                    # gray_result = self.isGrayscale(frame)
                    # if gray_result is not None:
                    #     self.convert_to_gray = gray_result
                    pass

                # Handling for grayscale conversion
                frame = self.handleGrayscaleConversion(frame)



                # If a video file is used, compute the time using the time from the file timestamp
                if self.video_file is not None:
                
                    frame_timestamp = video_first_timestamp + total_frames/self.config.fps

                    # print("tot={:6d}, i={:3d}, fps={:.2f}, t={:.8f}".format(total_frames, i, self.config.fps, frame_timestamp))

                    
                # Set the time of the first frame
                if i == 0:

                    # Initialize last frame timestamp if it's not set
                    if not last_frame_timestamp:
                        last_frame_timestamp = frame_timestamp
                    
                    # Always set first frame timestamp in the beginning of the block
                    first_frame_timestamp = frame_timestamp

                # Real-time timestamp refinement — disabled pending
                # lock contention investigation.  The offline solver
                # (SidedoorCorrect) handles sub-ms correction instead.
                # if (hasattr(self, '_pts_stream') and self._pts_stream is not None
                #         and hasattr(self, '_block_rtp_ts') and self._block_rtp_ts
                #         and self._block_rtp_ts[-1] is not None):
                #     sd_utc, _ = self._pts_stream.lookupFrameUtc(
                #         self._block_rtp_ts[-1], guard_shift=0, exposure_us=0.0)
                #     if sd_utc is not None:
                #         frame_timestamp = sd_utc

                # Append current timestamp to ft file buffer
                if self.config.save_frame_times:
                    self.timestamp_buffer.append((total_frames, frame_timestamp, frame_gst_pts_ns))

                # If save_frames is set and a video device is used, save a frame every nth frames
                if save_this_frame:

                    # Check if frame shape (color or grayscale) or capture mode changed (day or night)
                    if (frame.shape != self.current_raw_frame_shape) or \
                        (self.current_mode != (self.daytime_mode.value if self.daytime_mode is not None else False)) or \
                        (self.shared_raw_array is None):

                        log.info("Frame shape/mode changed, reinitializing arrays...")

                        # First signal the raw frame saver to finish saving current block
                        if raw_buffer_one:
                            self.start_raw_time1.value = first_raw_frame_timestamp
                        else:
                            self.start_raw_time2.value = first_raw_frame_timestamp

                        # Clean up existing frame saver before creating a new one
                        if hasattr(self, 'raw_frame_saver') and self.raw_frame_saver:
                            log.info("Cleaning up existing raw frame saver before mode change")
                            self.releaseRawArrays()

                        if not self.initRawFrameArrays(frame.shape):
                            log.error("Failed to reinitialize arrays after mode change")

                        else:
                            # Initialize new frame saver
                            self.raw_frame_saver = RawFrameSaver(
                                self.saved_frames_dir,
                                self.shared_raw_array, self.start_raw_time1,
                                self.shared_raw_array2, self.start_raw_time2,
                                self.sharedTimestamps, self.sharedTimestamps2,
                                self.daytime_mode.value,
                                self.config
                            )
                            self.raw_frame_saver.start()
                            self.raw_frame_count = 0
                            log.info("Successfully reinitialized raw frame handling")


                    # reset start time values everytime the buffers are switched
                    if self.raw_frame_count == 0:

                        if raw_buffer_one:
                            self.start_raw_time1.value = 0
                        else:
                            self.start_raw_time2.value = 0

                        # Always set first raw frame timestamp in the beginning of the block
                        first_raw_frame_timestamp = frame_timestamp 


                    # Write raw frame and timestamp to one of the two corresponding buffers
                    # Use appropriate indexing based on frame dimensions
                    if len(frame.shape) == 3:
                        # Color frame - use 4D indexing
                        if raw_buffer_one:
                            self.shared_raw_array[self.raw_frame_count, :, :, :] = frame
                            self.sharedTimestamps[self.raw_frame_count] = frame_timestamp
                        else:
                            self.shared_raw_array2[self.raw_frame_count, :, :, :] = frame
                            self.sharedTimestamps2[self.raw_frame_count] = frame_timestamp
                    else:
                        # Grayscale frame - use 3D indexing
                        if raw_buffer_one:
                            self.shared_raw_array[self.raw_frame_count, :, :] = frame
                            self.sharedTimestamps[self.raw_frame_count] = frame_timestamp
                        else:
                            self.shared_raw_array2[self.raw_frame_count, :, :] = frame
                            self.sharedTimestamps2[self.raw_frame_count] = frame_timestamp

                    self.raw_frame_count += 1

                    # switch buffers arrays every (self.num_raw_frames) frames
                    if self.raw_frame_count == self.num_raw_frames:

                        if raw_buffer_one:
                            self.start_raw_time1.value = first_raw_frame_timestamp
                        else:
                            self.start_raw_time2.value = first_raw_frame_timestamp
                        
                        self.raw_frame_count = 0
                        raw_buffer_one = not raw_buffer_one


                # If the end of the video file was reached, stop the capture
                if self.video_file is not None: 
                    if (frame is None) or (not self.deviceIsOpened()):

                        log.info("End of video file!")
                        log.debug("Video end status:")
                        log.debug("Frame:" + str(frame))
                        log.debug("Device open:" + str(self.deviceIsOpened()))

                        self.exit.set()
                        time.sleep(0.1)
                        break


                # Check if frame is dropped if it has been more than 1.5 frames than the last frame
                # Skip drop detection during first 250 frames (~10s) — GStreamer pipeline
                # startup causes transient PTS gaps from jitterbuffer/format negotiation
                elif total_frames > 250 and (frame_timestamp - last_frame_timestamp) >= self.time_for_drop:

                    # In VENC mode, use higher threshold to avoid false positives
                    # from hardware PTS jitter (daemon/writeback race produces
                    # 1-frame timestamp gaps while frames arrive on time).
                    drop_threshold = self.time_for_drop_venc if self.venc_epoch_offset is not None else self.time_for_drop
                    gap = frame_timestamp - last_frame_timestamp

                    if gap >= drop_threshold:
                        # Calculate the number of dropped frames
                        n_dropped = int(gap*self.config.fps)

                        self.dropped_frames.value += n_dropped

                        # Record timestamp for 10-minute window tracking
                        current_time = time.time()
                        # Add n_dropped timestamps efficiently
                        self.dropped_frames_timestamps.extend([current_time] * n_dropped)

                        # Clean up old timestamps efficiently (remove from left)
                        ten_min_ago = current_time - 600
                        while self.dropped_frames_timestamps and self.dropped_frames_timestamps[0] <= ten_min_ago:
                            self.dropped_frames_timestamps.popleft()

                        # Safety limit to prevent unbounded memory growth
                        if len(self.dropped_frames_timestamps) > 20000:
                            log.warning("Dropped frames timestamp queue exceeded safety limit, trimming to recent 10000 entries")
                            # Keep only most recent half
                            recent_timestamps = list(self.dropped_frames_timestamps)[-10000:]
                            self.dropped_frames_timestamps = deque(recent_timestamps)

                        gap_ms = gap * 1000
                        log.info("DROP: %d frames, gap=%.1fms (threshold=%.1fms), t_frame=%.3f",
                                 n_dropped, gap_ms, drop_threshold * 1000, t_frame)

                        if self.config.report_dropped_frames:
                            log.info("{}/{} frames dropped or late! Time for frame: {:.3f}, convert: {:.3f}, assignment: {:.3f}".format(
                                str(n_dropped), str(self.dropped_frames.value), t_frame, t_convert, t_assignment))


                # If cv2:
                if (self.config.media_backend != 'gst') and not self.media_backend_override and last_frame_timestamp is not False:
                    # Calculate the normalized frame interval between the current and last frame read, normalized by frames per second (fps)
                    frame_interval_normalized = (frame_timestamp - last_frame_timestamp)*self.config.fps
                    # Update max_frame_interval_normalized for this cycle
                    max_frame_interval_normalized = max(max_frame_interval_normalized, frame_interval_normalized)
                    sum_frame_interval_norm += frame_interval_normalized

                # If GStreamer:
                else:
                    # Use drift-corrected VENC timestamp for age so
                    # buffer fill and drift measurement track real time.
                    # FT timestamps are raw VENC clock (no drift correction).
                    age_ts = getattr(self, '_venc_raw_timestamp', None) or frame_timestamp
                    drift_corr = getattr(self, '_drift_total_correction', 0.0)
                    frame_age_seconds = time.time() - (age_ts + drift_corr)
                    # Update max_frame_age_seconds for this cycles
                    max_frame_age_seconds = max(max_frame_age_seconds, frame_age_seconds)
                    sum_frame_age_seconds += frame_age_seconds

                # On the last loop, report late or dropped frames
                if i == block_frames - 1:

                    # For cv2, show elapsed time since frame read to assess loop performance
                    if self.config.media_backend != 'gst' and not self.media_backend_override:
                        mean_interval_norm = sum_frame_interval_norm/block_frames

                        # running late-frame total since the start of capture
                        if run_start_ts is not None:
                            elapsed_run = last_frame_timestamp - run_start_ts
                            expected_run = int(round(elapsed_run*self.config.fps))
                            run_late_frames = max(0, expected_run - total_frames)
                        else:
                            run_start_ts = last_frame_timestamp
                            run_late_frames = 0

                        log.info("Block interval: mean %.3f, max %.3f (normalized). Dropped frames: %d",
                                 mean_interval_norm, max_frame_interval_normalized, run_late_frames)
                    
                    # For GStreamer, show elapsed time since frame capture to assess sink fill level
                    else:
                        # Check for any day/night mode transition to reset counters
                        current_daytime = self.daytime_mode.value if self.daytime_mode is not None else False
                        if self.last_daytime_mode is not None and self.last_daytime_mode != current_daytime:
                            # Transition detected (either day→night or night→day)
                            transition_type = "Day→Night" if not current_daytime else "Night→Day"
                            log.info(f"{transition_type} transition detected, resetting counters and media backend")

                            # Update last_daytime_mode BEFORE breaking to prevent detecting same transition again
                            self.last_daytime_mode = current_daytime

                            # Reset dropped frames counter for new session
                            self.dropped_frames.value = 0
                            self.dropped_frames_timestamps.clear()

                            # Reset PTS smoothing reset counter for new session
                            self.reset_count = -1

                            # Reset media backend override to allow GStreamer retry
                            self.media_backend_override = False

                            # Force device re-initialization by releasing and reconnecting
                            log.info("Releasing resources to re-initialize video device with GStreamer")
                            self.releaseResources()
                            wait_for_reconnect = True
                            break

                        self.last_daytime_mode = current_daytime

                        # Calculate buffer fill percentage based on max frame age
                        # The appsink has max-buffers=100, so at fps rate, max capacity is ~100/fps seconds
                        max_buffer_time = 100.0 / self.config.fps  # Theoretical max buffer time in seconds
                        buffer_fill_frames = max_frame_age_seconds * self.config.fps

                        # VENC clock discipline: estimate frequency offset
                        # between VENC crystal and host clock, then apply a
                        # continuous per-frame slew.  Like chrony: regression
                        # gives drift rate, PI loop tracks residual offset.
                        #
                        # _drift_freq  = estimated freq offset (s/s, ~92 ppm)
                        # _drift_samples = list of (elapsed_s, offset_s) pairs
                        #
                        # Offset = mean_age - baseline.  Positive = timestamps
                        # behind wallclock (VENC clock slow).
                        if (self.venc_epoch_offset is not None and block_frames > 0
                                and not (hasattr(self, '_pts_stream') and self._pts_stream is not None)):
                            mean_age = sum_frame_age_seconds / block_frames
                            now = time.time()

                            if self._drift_baseline_age is None:
                                # Delay baseline capture by 60s to skip
                                # GStreamer startup transient (~50ms offset
                                # settles within first minute).
                                if not hasattr(self, '_drift_warmup_t0'):
                                    self._drift_warmup_t0 = now
                                elif now - self._drift_warmup_t0 >= 60:
                                    self._drift_baseline_age = mean_age
                                    self._drift_t0 = now
                            else:
                                offset = mean_age - self._drift_baseline_age
                                elapsed = now - self._drift_t0

                                # Collect sample for regression
                                self._drift_samples.append((elapsed, offset))

                                # Keep last 2 minutes of samples (shorter
                                # window reduces phase lag → damps oscillation)
                                max_age = 120
                                self._drift_samples = [
                                    (t, o) for t, o in self._drift_samples
                                    if elapsed - t < max_age
                                ]

                                # Estimate frequency from regression (need
                                # at least 60s of post-baseline data)
                                if len(self._drift_samples) >= 6 and elapsed > 60:
                                    ts_arr = [s[0] for s in self._drift_samples]
                                    os_arr = [s[1] for s in self._drift_samples]
                                    n = len(ts_arr)
                                    t_mean = sum(ts_arr) / n
                                    o_mean = sum(os_arr) / n
                                    num = sum((t - t_mean) * (o - o_mean)
                                              for t, o in zip(ts_arr, os_arr))
                                    den = sum((t - t_mean) ** 2 for t in ts_arr)
                                    if den > 0:
                                        freq = num / den  # residual rate (s/s)
                                        # Additive: adjust drift_freq to cancel
                                        # the observed residual rate.
                                        alpha = 0.05
                                        self._drift_freq += alpha * freq
                                        # Clamp to sanity range
                                        self._drift_freq = max(-600e-6,
                                            min(600e-6, self._drift_freq))

                                # Log periodically
                                self._drift_log_accum += abs(offset)
                                if self._drift_log_accum > 0.050:
                                    log.info("VENC clock discipline: "
                                             "freq=%.1f ppm, offset=%.1fms, "
                                             "drift_corr=%.3fms",
                                             self._drift_freq * 1e6,
                                             offset * 1000,
                                             self._drift_total_correction * 1000)
                                    self._drift_log_accum = 0.0
                                    # Don't save drift_freq when side-door is active —
                                    # old controller's freq estimate is unreliable
                                    if not (hasattr(self, '_pts_stream') and self._pts_stream is not None):
                                        self._saveDriftFreq()

                        # Re-anchor epoch_offset from cam_wall (GetCurPTS anchor,
                        # ±3µs).  Match by PTS value: rtp_90k * 100/9 mod 2^32
                        # equals PtsStream venc_us mod 2^32 (±11µs quantization,
                        # unambiguous at 40ms frame spacing).  Wrap-immune.
                        if (hasattr(self, '_pts_stream') and self._pts_stream is not None
                                and self.venc_epoch_offset is not None
                                and self._rtp_clock_base is not None
                                and hasattr(self, '_rtp_prev_raw') and self._rtp_prev_raw):
                            try:
                                ps = self._pts_stream
                                WRAP32 = 4294967296
                                rtp_us_mod = (self._rtp_prev_raw * 100 // 9) % WRAP32
                                with ps._lock:
                                    entries = [(e[0], e[2]) for e in ps._entries
                                              if len(e) > 2 and e[2] and e[2] > 0]
                                # Find entry whose venc_us mod 2^32 matches rtp_us_mod
                                best_cw = None
                                best_d = 20000  # max 20ms tolerance
                                for venc_us, cw in entries:
                                    d = abs(int(venc_us % WRAP32) - int(rtp_us_mod))
                                    if d > WRAP32 // 2:
                                        d = WRAP32 - d  # wrap-around distance
                                    if d < best_d:
                                        best_d = d
                                        best_cw = cw
                                if best_cw is not None:
                                    # Unwrapped elapsed (handles overnight >13.3h)
                                    wrap_accum = getattr(self, '_rtp_wrap_accum', 0)
                                    elapsed = ((self._rtp_prev_raw - self._rtp_clock_base) & 0xFFFFFFFF) + wrap_accum
                                    expected = self.venc_epoch_offset + elapsed / 90000.0
                                    error = expected - best_cw
                                    if abs(error) < 2.0:
                                        self.venc_epoch_offset -= error
                                    if abs(error) > 0.001:
                                        log.info("cam_wall reanchor: error=%.3fms "
                                                 "(match_d=%dus wrap_accum=%d)",
                                                 error * 1000, best_d, wrap_accum)
                            except Exception as e:
                                log.warning("cam_wall reanchor exception: %s", e)

                        # Calculate dropped frames in last 10 minutes
                        current_time = time.time()
                        ten_min_ago = current_time - 600  # 10 minutes in seconds
                        recent_dropped = len([t for t in self.dropped_frames_timestamps if t > ten_min_ago])

                        meta_str = ""
                        if hasattr(self, '_venc_meta') and self._venc_meta is not None:
                            m = self._venc_meta.latest
                            if m:
                                meta_str = " | exp={}us ag={:.4f}x dg={:.4f}x ig={:.4f}x temp={}C".format(
                                    m.get('exposure_us', 0),
                                    m.get('analog_gain', 0),
                                    m.get('digital_gain', 0),
                                    m.get('isp_dgain', 0),
                                    m.get('soc_temp_c', '?'))
                        log.info("Buffer fill: {:.2f}/{:d} frames. Dropped frames: {} (last 10 min), {} this session{}".format(
                            buffer_fill_frames, self._appsink_max_buffers,
                            recent_dropped, self.dropped_frames.value, meta_str))

                last_frame_timestamp = frame_timestamp
                

                ### Convert the frame to grayscale ###  (Not to be done in case of daytime mode)
                if not self.daytime_mode.value:

                    t1_convert = time.time()

                    # Convert the frame to grayscale
                    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Convert the frame to grayscale
                    if len(frame.shape) == 3:

                        # If a color image is given, take the green channel
                        if frame.shape[2] == 3:

                            gray = frame[:, :, 1]

                        # If UYVY image given, take luma (Y) channel
                        elif self.config.uyvy_pixelformat and (frame.shape[2] == 2):
                            gray = frame[:, :, 1]

                        # Otherwise, take the first available channel
                        else:
                            gray = frame[:, :, 0]

                    else:
                        gray = frame


                    # Cut the frame to the region of interest (ROI)
                    gray = gray[self.config.roi_up:self.config.roi_down, \
                        self.config.roi_left:self.config.roi_right]

                    # Track time for frame conversion
                    t_convert = time.time() - t1_convert


                    ### ###




                    # Assign the frame to shared memory (track time to do so)
  
                    t1_assign = time.time()
                    if buffer_one:
                        self.array1[i, :gray.shape[0], :gray.shape[1]] = gray
                    else:
                        self.array2[i, :gray.shape[0], :gray.shape[1]] = gray

                    t_assignment = time.time() - t1_assign



                # Keep track of all captured frames
                total_frames += 1


            if self.exit.is_set():
                wait_for_reconnect = False
                log.info('Capture exited!')

                # Don't signal RawFrameSaver here - let releaseRawArrays() handle final flush
                # to avoid double flushing. The RawFrameSaver.stop() method will flush any
                # remaining frames automatically.

                break


            # Side-door block alignment: replace timestamps in the buffer
            # Side-door: capture raw data for offline reanalysis.
            # Do NOT replace timestamps — let the old-path handle real-time.
            # Save both RTP and pts_stream data so the alignment algorithm
            # can be developed and tested offline.
            if (hasattr(self, '_pts_stream') and self._pts_stream is not None
                    and hasattr(self, '_block_rtp_ts') and self._block_rtp_ts):
                try:
                    import numpy as _np
                    ps = self._pts_stream

                    # PTS-value matching: each RTP PTS → unique pts_stream entry
                    # (venc_us, cam_wall). cam_wall is chrony-GPS-disciplined
                    # UTC at FE_START, so mapping is unambiguous — no pipeline
                    # candidate grid, no median heuristics. Use align_block's
                    # utc output to replace coarse epoch_offset timestamps.
                    utc_list, raw_pts_list, _ = ps.align_block(
                        list(self._block_rtp_ts),
                        block_start_time=first_frame_timestamp)
                    self._block_raw_pts = raw_pts_list

                    # Overwrite timestamp_buffer entries with aligned UTCs.
                    # Falls back to epoch-based timestamp when a frame didn't
                    # match (e.g., firmware anomaly flagged by GUARD).
                    if (self.config.save_frame_times
                            and hasattr(self, 'timestamp_buffer')
                            and self.timestamp_buffer
                            and utc_list
                            and len(utc_list) == len(self.timestamp_buffer)):
                        aligned_count = 0
                        for i in range(len(self.timestamp_buffer)):
                            if utc_list[i] is not None:
                                fn, _old_ts, gst_pts_ns = self.timestamp_buffer[i]
                                self.timestamp_buffer[i] = (fn, utc_list[i], gst_pts_ns)
                                aligned_count += 1
                        if aligned_count > 0:
                            # Re-derive first_frame_timestamp from the first
                            # aligned frame (compression block start).
                            first_frame_timestamp = self.timestamp_buffer[0][1]
                        log.info("Side-door aligned %d/%d frame timestamps",
                                 aligned_count, len(self.timestamp_buffer))

                    with ps._lock:
                        ref_snapshot = [e[0] for e in ps._entries]
                        ref_camwall = [e[2] if len(e) > 2 and e[2] else 0.0
                                       for e in ps._entries]
                        _C = ps._C if ps._C else 0.0
                        # ps._C_raw may be None before the first probe has
                        # landed — coerce to a numeric fallback so savez
                        # doesn't serialize a Python-object array (which
                        # would then require allow_pickle=True to load).
                        _C_raw = ps._C_raw if (
                            hasattr(ps, '_C_raw') and ps._C_raw is not None
                        ) else _C
                        _dr = ps._drift_rate if ps._drift_rate is not None else 0.0
                        _wraps = ps._wraps if ps._wraps is not None else 0

                    rtp_arr = _np.array(
                        [int(x) if x is not None else 0
                         for x in self._block_rtp_ts], dtype=_np.uint32)
                    guard_arr = _np.array(
                        [x is None for x in self._block_rtp_ts],
                        dtype=_np.bool_)
                    # Matched raw PTS per frame (µs in full unwrapped domain).
                    # Value-based — offline solver can re-derive K indices.
                    raw_pts_arr = _np.array(
                        [p if p is not None else 0.0 for p in raw_pts_list],
                        dtype=_np.float64)
                    # Save the last 512 pts_stream entries (covers this
                    # block + margin for alignment search)
                    ref_arr = _np.array(ref_snapshot[-512:], dtype=_np.float64)
                    camwall_arr = _np.array(ref_camwall[-512:], dtype=_np.float64)

                    npz_dir = os.path.join(
                        self.config.data_dir, self.config.times_dir,
                        'sidedoor_raw')
                    mkdirP(npz_dir)
                    base = UTCFromTimestamp.utcfromtimestamp(
                        first_frame_timestamp)
                    npz_name = base.strftime(
                        "SD_{}_%Y%m%d_%H%M%S.npz".format(
                            self.config.stationID))
                    # Exposure time from ISP metadata (µs)
                    _exp_us = 0.0
                    if hasattr(self, '_venc_meta') and self._venc_meta is not None:
                        meta = self._venc_meta.latest
                        if 'exposure_us' in meta:
                            _exp_us = float(meta['exposure_us'])

                    # Per-frame diagnostics: dict hit/miss, stale correction
                    _diag = getattr(self, '_block_diag', [])
                    _diag_hit = _np.array([d['hit'] for d in _diag], dtype=bool)
                    _diag_gst = _np.array([d['gst_pts'] for d in _diag], dtype=_np.uint64)
                    _diag_rtp = _np.array([d['rtp_raw'] for d in _diag], dtype=_np.uint32)
                    _diag_stale = _np.array([d['stale_corrected'] for d in _diag], dtype=bool)
                    _diag_ts = _np.array([d['timestamp'] for d in _diag], dtype=_np.float64)

                    _np.savez_compressed(
                        os.path.join(npz_dir, npz_name),
                        rtp_90k=rtp_arr,
                        ref_pts_us=ref_arr,
                        ref_camwall=camwall_arr,
                        guard_flags=guard_arr,
                        matched_pts_us=raw_pts_arr,
                        C=_C, C_raw=_C_raw,
                        drift_rate=_dr, wraps=_wraps,
                        exposure_us=_exp_us,
                        block_start_time=first_frame_timestamp,
                        guard_shift=getattr(self, '_rtp_probe_shift', 0),
                        n_ref_total=len(ref_snapshot),
                        diag_hit=_diag_hit,
                        diag_gst_pts=_diag_gst,
                        diag_rtp_raw=_diag_rtp,
                        diag_stale_corrected=_diag_stale,
                        diag_timestamp=_diag_ts)
                    log.info("Side-door: saved %d RTP + %d ref_pts + %d diag to %s",
                             len(rtp_arr), len(ref_arr), len(_diag), npz_name)
                except Exception as e:
                    log.debug("Failed to save sidedoor .npz: %s", e)

                self._block_rtp_ts.clear()
                if hasattr(self, '_block_diag'):
                    self._block_diag.clear()

            # Re-derive first_frame_timestamp from the buffer in case
            # the stale-PTS guard or side-door corrected frame 0.
            if (self.config.save_frame_times
                    and hasattr(self, 'timestamp_buffer')
                    and len(self.timestamp_buffer) > 0):
                first_frame_timestamp = self.timestamp_buffer[0][1]

            if (not wait_for_reconnect
                and not self.daytime_mode.value
                and first_frame_timestamp is not None):

                # Set the starting value of the frame block, which indicates to the compression that the
                # block is ready for processing
                if buffer_one:
                    self.start_time1.value = first_frame_timestamp

                else:
                    self.start_time2.value = first_frame_timestamp

                log.debug('New block of raw frames available for compression with starting time: {:s}'
                         .format(str(first_frame_timestamp)))

            
            # Switch the frame block buffer flags
            buffer_one = not buffer_one
            if self.config.report_dropped_frames:
                log.info('Estimated FPS: {:.3f}'.format(block_frames/(time.time() - t_block)))
        

            # Save current timestamp buffer to ft file
            # Construct FTStruct, record timestamps, and reset the timestamp array in memory
            if (self.config.save_frame_times and first_frame_timestamp is not None):
                ft = FTStruct.FTStruct()
                # timestamp_buffer has 3-tuples: (frame_num, utc, gst_pts_ns)
                ft.timestamps = [(fn, ts) for fn, ts, _ in self.timestamp_buffer]
                ft.gst_pts_ns = [pts for _, _, pts in self.timestamp_buffer]
                # Attach raw PTS if available (from side-door align_block)
                raw_pts = getattr(self, '_block_raw_pts', [])
                n_nonnull = sum(1 for p in raw_pts if p is not None) if raw_pts else 0
                log.debug("FT raw_pts: len=%d ts_len=%d nonnull=%d",
                          len(raw_pts), len(ft.timestamps), n_nonnull)
                if raw_pts and len(raw_pts) == len(ft.timestamps):
                    ft.raw_pts_us = [p if p is not None else 0.0
                                     for p in raw_pts]

                # Clear the timestamp buffer list
                del self.timestamp_buffer[:]

                # Clear raw PTS list so stale data can't leak into the next block
                if hasattr(self, '_block_raw_pts'):
                    self._block_raw_pts = []

                base_time = UTCFromTimestamp.utcfromtimestamp(first_frame_timestamp)
                ft_filename = base_time.strftime("FT_{}_%Y%m%d_%H%M%S.bin".format(self.config.stationID))
                ft_subpath = os.path.join(self.config.data_dir, self.config.times_dir, base_time.strftime("%Y/%Y%m%d-%j/%Y%m%d-%j_%H"))

                mkdirP(ft_subpath)
                FTfile.write(ft, ft_subpath, ft_filename)
                log.debug("Created FT file {} for block starting at {}".format(os.path.join(ft_subpath, ft_filename), first_frame_timestamp))

                # For Testing: 
                # Print first and last 10 timestamps, array length, average time difference and time difference from last block
                # Enable self.ft_test_time in __init__
                
                # print("\n\n --- FT file data --- \nFirst 10 timestamps: {}\n\nLast 10 timestamps: {}\n\nArray length: {}\n\n".format(
                #       ft.timestamps[:11], 
                #       ft.timestamps[-10:],
                #       len(ft.timestamps),
                # ),
                #       "Average per-frame time difference: {}\n\nLast segment time difference: {}\n\n ---------------- \n\n".format(
                #       sum(ft.timestamps[i+1][1] - ft.timestamps[i][1] for i in range(len(ft.timestamps) - 1)) / (len(ft.timestamps) - 1),
                #       ft.timestamps[0][1] - self.ft_test_time
                # ), end='')
                # self.ft_test_time = ft.timestamps[-1][1]


        log.info('Releasing video device...')
        self.releaseResources()


if __name__ == "__main__":

    import argparse
    import ctypes

    import multiprocessing

    import RMS.ConfigReader as cr

    ###

    arg_parser = argparse.ArgumentParser(description='Test capturing frames from a video source defined in the config file. ')

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
        help="Path to a config file which will be used instead of the default one.")
    
    arg_parser.add_argument('--video_file', metavar='VIDEO_FILE', type=str, \
        help="Path to a video file to be used as a video source instead of a camera.")
    

     # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    ###
    
    # Load the config file
    config = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('.'))

    # Initialize the logger
    log_manager = LoggingManager()
    log_manager.initLogging(config)

    # Get the logger handle
    log = getLogger("rmslogger")

    # Print the kind of media backend
    print("Station code: {}".format(config.stationID))
    print('Media backend: {}'.format(config.media_backend))


    # Init dummy shared memory
    sharedArrayBase = multiprocessing.Array(ctypes.c_uint8, 256*(config.width)*(config.height))
    sharedArray = np.ctypeslib.as_array(sharedArrayBase.get_obj())
    sharedArray = sharedArray.reshape(256, (config.height), (config.width))
    startTime = multiprocessing.Value('d', 0.0)


    # If a video is given, use it as the video source
    if cml_args.video_file:

        print("Using video file: {}".format(cml_args.video_file))

        bc = BufferedCapture(sharedArray, startTime, sharedArray, startTime, config, 
                             video_file=cml_args.video_file)
        
        bc.initVideoDevice()
        

        # Read at least 256 frames from the video file
        for i in range(256):
            ret, frame = bc.device.read()

            print('Frame read: {}'.format(i))
            if not ret:
                print("End of video file!")
                break
                
        # Close the device
        bc.releaseResources()

        
    
    # Capture from a camera
    else:

        # Init the BufferedCapture object
        bc = BufferedCapture(sharedArray, startTime, sharedArray, startTime, config)

        device = bc.createGstreamDevice('BGR', video_file_dir=None, segment_duration_sec=config.raw_video_duration)

        print('GStreamer device created!')

        ### TEST
        print("Pulling a sample...", end=' ')
        sample = device.emit("pull-sample")
        print('Sample pulled!')

        print('Mapping buffer...', end=' ')
        buffer = sample.get_buffer()
        ret, map_info = buffer.map(Gst.MapFlags.READ)
        print('Buffer mapped!')

        print('Getting caps...', end=' ')
        caps = sample.get_caps()
        print('Caps obtained!')

        print('Getting structure...', end=' ')
        structure = caps.get_structure(0)
        print('Structure obtained!')

        print('Extracting width and height...', end=' ')
        width = getStructureValue(structure, 'width')
        height = getStructureValue(structure, 'height')
        print('Width and height extracted!')

        print('Creating frame...', end=' ')
        frame_shape = (height, width, 3)
        frame = np.ndarray(shape=frame_shape, buffer=map_info.data, dtype=np.uint8)
        print('Frame created!')

        print('Unmapping buffer...', end=' ')
        buffer.unmap(map_info)
        print('Buffer unmapped!')
        ###

        # Close the device
        bc.releaseResources()
