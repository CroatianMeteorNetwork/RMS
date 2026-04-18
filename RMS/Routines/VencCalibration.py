"""VENC PTS calibration for cameras running the gettime service.

Cameras with rtp_patch + gettime provide hardware VENC PTS timestamps
(sensor capture time, sub-ms accuracy) in the RTP stream. This module
calibrates the constant C that maps those PTS values to absolute
wallclock (UTC):

    frame_wallclock = C + rtp_timestamp / 90000.0

The gettime service on the camera spin-polls the VENC PTS register and
snaps gettimeofday() at the exact frame transition, returning both
values atomically. Multiple samples are collected and averaged.

Clock offset correction:
    The camera's gettimeofday (busybox ntpd, ~5ms accuracy) is the
    dominant absolute timing error. To eliminate it, each gettime TCP
    query measures the network one-way delay from the TCP handshake
    (SYN → SYN-ACK = 2× one-way). The host's wallclock at the PTS
    transition is estimated as t_response_received - one_way_delay.
    This references C to the HOST clock (sub-ms if running chrony/PPS)
    instead of the camera's ntpd clock.

Protocol (TCP, one response per connection):
    "<epoch_seconds.usec> <pts_90khz_int> <pts_microseconds>\\n"

Usage:
    C = calibrate_epoch_offset("192.168.42.121", port=9601)
    if C is not None:
        # After capturing clock_base from first RTP packet:
        epoch_offset = C + clock_base / 90000.0
        frame_wallclock = epoch_offset + buffer_pts_seconds
"""

import socket
import time
import logging

log = logging.getLogger("logger")


def _query_gettime(ip, port, timeout=5.0):
    """Single TCP query to the gettime service.

    Measures the TCP connect time (SYN → SYN-ACK) to estimate the
    network one-way delay.  The camera sends the response immediately
    after gettimeofday(), so the host's estimate of the snap time is:
        host_at_snap = t_response_received - one_way_delay

    Returns:
        (cam_time, pts_90k, pts_us, host_at_snap, one_way_ms)
        or None on failure.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)

        # One absolute timestamp at the start; all intervals via
        # perf_counter (monotonic, immune to NTP adjustments).
        t0_abs = time.time()
        t0_pc = time.perf_counter()

        # Measure TCP handshake for one-way delay: connect() blocks
        # for SYN → SYN-ACK (2 × network one-way transit)
        s.connect((ip, port))
        t1_pc = time.perf_counter()

        # Server now polls for PTS transition, then sends response
        resp = s.recv(128).decode().strip()
        t2_pc = time.perf_counter()
        s.close()

        parts = resp.split()
        if len(parts) < 3:
            return None

        one_way = (t1_pc - t0_pc) / 2.0
        host_at_snap = t0_abs + (t2_pc - t0_pc) - one_way

        return (float(parts[0]), int(parts[1]), int(parts[2]),
                host_at_snap, one_way * 1000)
    except Exception:
        return None


def calibrate_epoch_offset(ip, port=9601, n_samples=20, timeout_per_sample=5.0):
    """Calibrate C: the PTS-to-wallclock constant.

    Collects n_samples from the gettime service (each waits for a PTS
    frame transition on the camera, ~40ms per sample).

    For each sample the host estimates its own wallclock at the PTS
    transition using the TCP-handshake one-way delay.  C is computed
    relative to the HOST clock, bypassing camera NTP error entirely.

    Args:
        ip: Camera IP address.
        port: gettime TCP port (default 9601).
        n_samples: Number of samples to collect (default 20, ~0.8s).
        timeout_per_sample: TCP timeout per query in seconds.

    Returns:
        C (float): the PTS-to-wallclock constant such that
            frame_wallclock = C + rtp_timestamp / 90000.0
        Referenced to the host clock.  Returns None if calibration failed.
    """
    samples = []
    for _ in range(n_samples):
        result = _query_gettime(ip, port, timeout_per_sample)
        if result is not None:
            samples.append(result)

    if len(samples) < 3:
        log.warning("VencCalibration: only got %d/%d samples from %s:%d",
                     len(samples), n_samples, ip, port)
        return None

    # Compute camera-host clock offset for diagnostics.
    # cam_time and host_at_snap should represent the same instant.
    clock_offsets = [s[0] - s[3] for s in samples]
    clock_offset = sum(clock_offsets) / len(clock_offsets)

    # Compute C using host_at_snap (host-referenced wallclock).
    #
    # C = host_at_snap - pts_90k / 90000
    #
    # This is the constant that maps any VENC PTS (in 90kHz ticks)
    # to absolute wallclock on the HOST clock:
    #     wallclock = C + pts_90k / 90000
    offsets = []
    for cam_time, pts_90k, pts_us, host_at_snap, ow_ms in samples:
        offsets.append(host_at_snap - pts_90k / 90000.0)

    # Trim outliers (top/bottom 10%) and take mean
    offsets.sort()
    trim = max(1, len(offsets) // 10)
    trimmed = offsets[trim:-trim] if len(offsets) > 2 * trim else offsets
    C = sum(trimmed) / len(trimmed)

    spread_ms = (offsets[-1] - offsets[0]) * 1000
    trimmed_spread_ms = (trimmed[-1] - trimmed[0]) * 1000
    ow_mean = sum(s[4] for s in samples) / len(samples)

    log.info("VencCalibration: %d samples from %s:%d, spread=%.2fms "
             "(trimmed=%.2fms), C=%.6f, cam-host=%.2fms, one_way=%.2fms",
             len(samples), ip, port, spread_ms, trimmed_spread_ms, C,
             clock_offset * 1000, ow_mean)

    return C


def probe_clock_sample(ip, port=9601, timeout=5.0):
    """Single host-referenced (venc_pts_us, host_utc) measurement.

    Returns (pts_us, host_utc_float, one_way_ms) or None.
    The host_utc is referenced to the HOST clock (chrony/GPS PPS),
    bypassing camera NTP entirely.
    """
    result = _query_gettime(ip, port, timeout)
    if result is None:
        return None
    cam_time, pts_90k, pts_us, host_at_snap, ow_ms = result
    return (pts_us, host_at_snap, ow_ms)
