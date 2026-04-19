"""Verify pipeline timing accuracy using GPS-PPS LED flashes.

For each LED pulse, measure the on/off edge time by rolling-shutter
row counting. Uses FT per-frame cam_wall as the anchor (truth) and
compares against the known GPS-PPS integer second.

The LED rig:
    Pulse width ≈ 100 ms (> 40 ms frame period, > 8 ms VBI)
    LED-on at integer UTC second + ~1 ms rig delay (driver ramp)
    Physical LED localized to a narrow row band in the scene

Rolling shutter:
    HMAX=4400 → line time = 29.63 µs/row
    1080 active rows → readout = 32.0 ms
    Frame period = 40.0 ms → VBI = 8.0 ms (row-counting blind spot)

Three edge cases per pulse (from reference_rolling_shutter_blind_spot.md):
    1. Leading edge in readout + trailing edge in readout  → 2 measurements
    2. Leading edge in readout + trailing edge in VBI       → 1 (leading)
    3. Leading edge in VBI       + trailing edge in readout → 1 (trailing)
    Case "both in VBI" is impossible because pulse > frame period.

An edge IS row-resolvable when its frame has a partial gradient across
the LED row band (some rows dark, some bright in the same frame).
An edge is in VBI when its "transition" frame is uniformly bright
(leading lit the whole next frame) or uniformly dark (trailing left the
previous frame fully lit), with no internal gradient.

Usage:
    python3 -m Utils.VerifyLEDTiming --mkv <mkv> --ft-dir <ftdir>

Dependencies: ffmpeg/ffprobe, numpy.
"""
from __future__ import print_function, division
import argparse, bisect, datetime, os, re, subprocess, sys

import numpy as np

from RMS.Formats import FTfile


LINE_TIME_S = 4400.0 / 148.5e6   # 29.6296 µs (HMAX=4400, pixel_clk=148.5 MHz)
FRAME_PERIOD_S = 0.040           # Exact 25 fps after VMAX=1350 fix
ACTIVE_ROWS = 1080
VBI_S = FRAME_PERIOD_S - ACTIVE_ROWS * LINE_TIME_S  # ≈ 8.0 ms
LED_RIG_DELAY_S = 1.0e-3         # GPS-PPS → LED-light aggregate delay
LED_PULSE_S = 0.100              # LED on-time (100 ms)
MKV_W, MKV_H = 1920, 1080


# ──────────────────────────────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────────────────────────────
def parse_mkv_filename_ts(fname):
    base = os.path.basename(fname)
    m = re.search(r'_(\d{8})_(\d{6})_(\d{6})_video\.mkv$', base)
    if not m:
        raise ValueError("Filename doesn't match MKV pattern: %s" % base)
    dt = datetime.datetime.strptime(
        m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(
        tzinfo=datetime.timezone.utc)
    return dt.timestamp() + int(m.group(3)) / 1e6


def get_mkv_pts_seconds(mkv_path, max_frames=None):
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v',
           '-show_entries', 'frame=pts_time', '-of', 'csv=p=0', mkv_path]
    out = subprocess.check_output(cmd, text=True).strip().splitlines()
    values = [float(line) for line in out if line]
    return values[:max_frames] if max_frames else values


def extract_frames_gray(mkv_path, n_frames):
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error',
           '-i', mkv_path, '-frames:v', str(n_frames),
           '-vf', 'format=gray', '-f', 'rawvideo', '-']
    raw = subprocess.check_output(cmd)
    frame_bytes = MKV_W * MKV_H
    n = len(raw) // frame_bytes
    return np.frombuffer(raw[:n*frame_bytes], dtype=np.uint8).reshape(
        n, MKV_H, MKV_W)


def build_ft_timeline(ft_dir, time_window=None):
    """Returns list of (utc, gst_pts_ns, fname, local_idx, frame_num)
    sorted by utc. If time_window=(t0,t1) is given, only entries in
    [t0,t1] are kept."""
    entries = []
    for f in sorted(os.listdir(ft_dir)):
        if not f.startswith('FT_') or not f.endswith('.bin'):
            continue
        ft = FTfile.read(ft_dir, f)
        has_gst = hasattr(ft, 'gst_pts_ns') and len(ft.gst_pts_ns) == len(ft.timestamps)
        for i, (fn, utc) in enumerate(ft.timestamps):
            if time_window and not (time_window[0] <= utc <= time_window[1]):
                continue
            gst = ft.gst_pts_ns[i] if has_gst else None
            entries.append((utc, gst, f, i, fn))
    entries.sort()
    return entries


# ──────────────────────────────────────────────────────────────────────
# LED pulse analysis
# ──────────────────────────────────────────────────────────────────────
def locate_led_band(frames):
    """Find the rows where the LED physically illuminates the scene.

    Strategy: pick a handful of "bright" frames (LED-on) and compare
    their max-per-row to max-per-row of non-bright frames.  Rows where
    the LED saturates (>= +80 boost vs ambient) are the LED band.
    Returns (row_start, row_end) of the longest contiguous run, or
    None if no band is found.
    """
    # Identify LED-on frames by overall brightness spike
    mids = frames[:, 400:800, :].mean(axis=(1, 2))
    thresh = mids.mean() + 1.5 * mids.std()
    bright = np.where(mids > thresh)[0]
    dark = np.setdiff1d(np.arange(len(frames)), bright)
    if len(bright) == 0 or len(dark) == 0:
        return None
    led_row_max = frames[bright].max(axis=0).mean(axis=1)
    bg_row_max = frames[dark].max(axis=0).mean(axis=1)
    diff = led_row_max - bg_row_max
    band_rows = np.where(diff > 80)[0]
    if len(band_rows) < 10:
        return None
    gaps = np.where(np.diff(band_rows) > 5)[0]
    starts = np.concatenate(([0], gaps + 1))
    ends = np.concatenate((gaps + 1, [len(band_rows)]))
    best = max(range(len(starts)), key=lambda k: ends[k] - starts[k])
    r0 = int(band_rows[starts[best]])
    r1 = int(band_rows[ends[best] - 1]) + 1
    return (r0, r1)


def detect_pulses(frames, band, bright_thresh=150, dark_thresh=80):
    """Classify each frame as 'dark', 'bright', or 'partial' within
    the LED band. Group into pulses (runs of not-dark frames)."""
    r0, r1 = band
    band_img = frames[:, r0:r1, :]
    # For each frame, examine row means within the band
    # state heuristics:
    #   all rows bright (mean > bright_thresh) → 'bright' (fully exposed)
    #   all rows dark (mean < dark_thresh)     → 'dark'
    #   otherwise (gradient)                   → 'partial'
    states = []
    per_frame_row_means = []
    for i in range(len(band_img)):
        row_means = band_img[i].mean(axis=1)
        per_frame_row_means.append(row_means)
        mn, mx = row_means.min(), row_means.max()
        if mn > bright_thresh:
            state = 'bright'
        elif mx < dark_thresh:
            state = 'dark'
        else:
            state = 'partial'
        states.append(state)

    # Group into pulses: a pulse is a run from first non-dark frame to
    # last non-dark frame before returning to dark.
    pulses = []
    i = 0
    while i < len(states):
        if states[i] == 'dark':
            i += 1
            continue
        start = i
        while i < len(states) and states[i] != 'dark':
            i += 1
        end = i - 1
        pulses.append({'start': start, 'end': end, 'states': states[start:end+1]})
    return pulses, states, per_frame_row_means


def find_transition_row(row_means, direction):
    """Row where brightness rises (direction='up') or falls ('down').
    Returns row index or None if no clean transition visible.

    Uses a THRESHOLD CROSSING, not argmax(diff), because on a dim or
    slow-ramping LED the dark→bright transition spans ~20 rows and
    each per-row step is only ~10 units — argmax(diff) catches it but
    argmax(diff) >20 threshold rejects it. Threshold-crossing finds
    the FIRST row whose brightness is clearly above ambient (or below
    saturation for trailing), which is where the physical edge landed.
    """
    rm = row_means.astype(np.int32)
    # Ambient from top 5% rows (assumed dark for leading frame).
    n_top = max(5, len(rm) // 20)
    if direction == 'up':
        ambient_top = rm[:n_top].mean()
        # Find first row above ambient + significant margin
        threshold = ambient_top + 15
        above = np.where(rm > threshold)[0]
        if len(above) == 0:
            return None
        # Require at least 5 consecutive rows above for stability
        return int(above[0]) if (rm.max() - ambient_top) > 30 else None
    else:
        # For trailing: find first row where brightness drops from top plateau
        top_level = rm[:n_top].mean()
        threshold = top_level - 30
        below = np.where(rm < threshold)[0]
        if len(below) == 0:
            return None
        return int(below[0]) if (top_level - rm.min()) > 30 else None


def measure_edge(frames_row_means, frame_idx, ft_for_frame, band_r0,
                 direction, exposure_s=0.040):
    """Measure the LED on/off edge time from a partial frame.

    direction: 'on' (leading) or 'off' (trailing).

    Leading and trailing edges expose DIFFERENT row boundaries.

    LEADING (LED turns on):
        Rows below r_a: exposure ended BEFORE LED on → dark
        Rows above r_a: progressively more LED → saturated quickly
        Visible transition: r_a (sharp dark → bright ramp)
        Formula: T_on = FE_START + r_a × line_time

    TRAILING (LED turns off):
        Rows below r_a (hidden by saturation): full LED → saturated
        Rows r_a..r_b: ramp saturated → dark
        Rows above r_b: exposure STARTED after LED off → dark
        With bright saturating LED the r_a→ramp region is saturated-flat
        and the only visible sharp transition is r_b (partial → dark).
        Formula: T_off = FE_START + r_b × line_time − exposure

    Returns (T_edge, sensor_row) or (None, None) if no transition found.
    """
    rm = frames_row_means[frame_idx]
    # For leading, find brightness RISE; for trailing, find DROP.
    row = find_transition_row(rm, 'up' if direction == 'on' else 'down')
    if row is None:
        return None, None
    sensor_row = band_r0 + row
    fe_start = ft_for_frame
    # With pts_stream2 publishing cam_wall = CLOCK_REALTIME snapshot at
    # ioctl(FE_START) return, cam_wall is the UTC of row-0 READOUT
    # start (FE_START), NOT the row-0 integration start (T0).
    # T_readout(r) = cam_wall + r × line_time.
    # exposure_start(r) = T_readout(r) − exposure = cam_wall + r×LT − exp
    if direction == 'on':
        # r_a (first lit row): T_readout(r_a) = T_on
        T_edge = fe_start + sensor_row * LINE_TIME_S
    else:
        # r_b (first dark row after partial): exposure_start(r_b) = T_off
        T_edge = fe_start + sensor_row * LINE_TIME_S - exposure_s
    return T_edge, sensor_row


# ──────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────
def verify(mkv_path, ft_dir, probe_frames=250, verbose=True):
    mkv_t0 = parse_mkv_filename_ts(mkv_path)
    pts_s = get_mkv_pts_seconds(mkv_path, max_frames=probe_frames)
    frames = extract_frames_gray(mkv_path, probe_frames)

    band = locate_led_band(frames)
    if band is None:
        print("ERROR: no LED band detected in scene", file=sys.stderr)
        return []
    r0, r1 = band

    pulses, states, row_means = detect_pulses(frames, band)

    # FT timeline covering the MKV window
    seg_start = mkv_t0 - 1.0
    seg_end = mkv_t0 + probe_frames * FRAME_PERIOD_S + 1.0
    ft_tl = build_ft_timeline(ft_dir, time_window=(seg_start, seg_end))
    ft_utcs = [e[0] for e in ft_tl]

    # For a given MKV frame index, pick the FT entry by matching gst_pts
    # (appsink buffer.pts = MKV buffer.pts in ns). ffprobe gives pts_time
    # in seconds with ms resolution; use the raw integer pts to key lookup.
    pts_base = pts_s[0] if pts_s else 0.0
    # FT gst_pts_ns is absolute pipeline running time; MKV pts_s is also
    # absolute (in seconds). Match by nearest.
    def ft_utc_for_mkv(mkv_i):
        if mkv_i >= len(pts_s):
            return None, None
        target_s = pts_s[mkv_i]
        target_ns = int(target_s * 1e9)
        # Find closest ft entry by gst_pts_ns (or by utc if gst not avail)
        best = None
        best_d = 1 << 62
        for j, e in enumerate(ft_tl):
            if e[1] is None:
                continue
            d = abs(e[1] - target_ns)
            if d < best_d:
                best_d = d
                best = j
        if best is None or best_d > 50_000_000:  # >50 ms → missing
            return None, None
        return ft_tl[best][0], best

    results = []
    if verbose:
        print(f"MKV: {os.path.basename(mkv_path)}")
        print(f"MKV t0 = {mkv_t0:.6f} ({datetime.datetime.fromtimestamp(mkv_t0, datetime.timezone.utc).isoformat()})")
        print(f"LED band: rows {r0}–{r1} ({r1-r0} rows)")
        print(f"Line time = {LINE_TIME_S*1e6:.3f} µs/row, VBI = {VBI_S*1e3:.2f} ms")
        print(f"Pulses detected: {len(pulses)}")
        print(f"\n{'pulse':>5} {'lead_i':>6} {'lead_row':>8} {'trail_i':>7} {'trail_row':>9} "
              f"{'T_on':>16} {'T_off':>16} {'pps':>16} {'on-pps_ms':>10} {'off-pps_ms':>11} {'case':>5}")

    for p_idx, p in enumerate(pulses):
        lead_i = p['start']
        trail_i = p['end']
        lead_state = p['states'][0]
        trail_state = p['states'][-1]

        # Leading edge: row-resolvable only if the first pulse frame is 'partial'.
        # If it's 'bright', leading edge was in VBI before it.
        T_on = T_off = None
        lead_row = trail_row = None
        lead_ft_utc = None
        trail_ft_utc = None

        # Leading edge is row-resolvable when the FIRST pulse frame is
        # 'partial' (LED turned on during this frame's readout, leaving
        # a dark-top/bright-bottom gradient).  If the first pulse frame
        # is already 'bright', LED turned on in the VBI before it and
        # the leading edge is unresolvable.
        if lead_state == 'partial':
            lead_ft_utc, _ = ft_utc_for_mkv(lead_i)
            if lead_ft_utc is not None:
                T_on, lead_row = measure_edge(row_means, lead_i, lead_ft_utc, r0, 'on')

        # Trailing edge is row-resolvable when the LAST pulse frame is
        # 'partial' (LED turned off during this frame's readout, leaving
        # a bright-top/dim-bottom gradient).  If the last pulse frame
        # is 'bright', LED turned off in the VBI after it (→ next frame
        # is 'dark', trailing edge is unresolvable).
        if trail_state == 'partial':
            trail_ft_utc, _ = ft_utc_for_mkv(trail_i)
            if trail_ft_utc is not None:
                T_off, trail_row = measure_edge(
                    row_means, trail_i, trail_ft_utc, r0, 'off')

        # Derive T_on from T_off if needed
        if T_on is None and T_off is not None:
            T_on_derived = T_off - LED_PULSE_S
        else:
            T_on_derived = T_on

        if T_on_derived is None:
            continue

        # Expected PPS: nearest integer UTC second to T_on
        pps_utc = round(T_on_derived)
        on_err_ms = (T_on_derived - pps_utc) * 1000.0
        off_err_ms = None
        if T_off is not None:
            off_err_ms = (T_off - (pps_utc + LED_PULSE_S)) * 1000.0

        case = ('1both' if (T_on is not None and T_off is not None) else
                '2lead' if T_on is not None else
                '3trail')
        results.append({
            'pulse': p_idx, 'case': case,
            'T_on': T_on_derived, 'T_off': T_off,
            'on_err_ms': on_err_ms, 'off_err_ms': off_err_ms,
            'lead_row': lead_row, 'trail_row': trail_row,
            'lead_i': lead_i if T_on is not None else None,
            'trail_i': trail_i if T_off is not None else None,
            'pps_utc': pps_utc,
        })

        if verbose:
            lead_i_s = str(lead_i) if T_on is not None else '-'
            lead_row_s = str(lead_row) if lead_row is not None else '-'
            trail_i_s = str(trail_i) if T_off is not None else '-'
            trail_row_s = str(trail_row) if trail_row is not None else '-'
            t_on_s = f"{T_on_derived:.6f}" if T_on_derived is not None else '-'
            t_off_s = f"{T_off:.6f}" if T_off is not None else '-'
            off_s = f"{off_err_ms:+.3f}" if off_err_ms is not None else '-'
            print(f"{p_idx:>5} {lead_i_s:>6} {lead_row_s:>8} {trail_i_s:>7} {trail_row_s:>9} "
                  f"{t_on_s:>16} {t_off_s:>16} {pps_utc:>16} {on_err_ms:>+10.3f} {off_s:>11} {case:>5}")

    if verbose and results:
        on_errs = [r['on_err_ms'] for r in results]
        corrected = [e - LED_RIG_DELAY_S * 1000.0 for e in on_errs]
        print(f"\n=== Summary (n={len(results)}) ===")
        print(f"T_on − PPS:  mean={np.mean(on_errs):+.3f} ms  std={np.std(on_errs):.3f} ms")
        print(f"After LED-rig correction (−1.0 ms): mean={np.mean(corrected):+.3f} ms")
        # Case breakdown
        cases = {}
        for r in results:
            cases[r['case']] = cases.get(r['case'], 0) + 1
        print(f"Cases: {cases}")
    return results, frames, pulses, row_means


def plot_results(results, frames, pulses, row_means, band, mkv_t0):
    """GUI: for each pulse, show the actual video frame with the detected
    transition row drawn across it, plus the computed timestamp and
    residual vs GPS-PPS. User can eyeball whether the row-detection
    matches where the LED edge actually is.
    """
    try:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; install with: pip install matplotlib",
              file=sys.stderr)
        return

    r0, r1 = band
    n = min(len(results), 10)
    if n == 0:
        print("No pulses to plot.", file=sys.stderr)
        return

    # Two columns: leading frame, trailing frame
    fig, axes = plt.subplots(n, 2, figsize=(13, 3.2*n), squeeze=False)
    fig.suptitle("LED leading/trailing edge detection — visual verification\n"
                 "Green line = detected leading-edge row,  Red line = detected trailing-edge row",
                 fontsize=11)

    for row, r in enumerate(results[:n]):
        p_idx = r['pulse']
        lead_i = r.get('lead_i')
        trail_i = r.get('trail_i')
        lead_row = r.get('lead_row')
        trail_row = r.get('trail_row')
        case = r['case']
        T_on = r['T_on']
        T_off = r['T_off']
        pps = r['pps_utc']
        on_err = r['on_err_ms']
        off_err = r['off_err_ms']

        # Column 0: leading frame
        ax = axes[row, 0]
        if lead_i is not None and lead_i < len(frames):
            ax.imshow(frames[lead_i], cmap='gray', vmin=0, vmax=255, aspect='auto')
            if lead_row is not None:
                ax.axhline(lead_row, color='lime', linestyle='-', linewidth=2)
                ax.text(30, lead_row - 25, f"LEADING row r_a = {lead_row}",
                        color='lime', fontsize=11, fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.7, pad=2))
            # Timestamp overlay
            info = (f"Pulse {p_idx} LEADING\n"
                    f"Frame idx in MKV: {lead_i}\n"
                    f"T_on (measured)  = {T_on:.6f}\n"
                    f"Nearest GPS PPS  = {pps}\n"
                    f"Residual to PPS  = {on_err:+.1f} ms")
            ax.text(0.02, 0.98, info, transform=ax.transAxes,
                    color='yellow', fontsize=9, fontweight='bold',
                    verticalalignment='top', family='monospace',
                    bbox=dict(facecolor='black', alpha=0.75, pad=4))
        else:
            ax.text(0.5, 0.5, f"Pulse {p_idx} LEADING in VBI\n(case {case})\n"
                              f"T_on derived from T_off: {T_on:.6f}\n"
                              f"Residual to PPS: {on_err:+.1f} ms",
                    transform=ax.transAxes, ha='center', va='center',
                    color='white', fontsize=10, family='monospace',
                    bbox=dict(facecolor='darkred', alpha=0.8, pad=6))
            ax.set_facecolor('#222')
        ax.set_ylabel(f"P{p_idx}", fontsize=10, fontweight='bold')
        ax.set_yticks([0, 270, 540, 810, 1079])
        ax.set_xticks([])

        # Column 1: trailing frame
        ax = axes[row, 1]
        if trail_i is not None and trail_i < len(frames):
            ax.imshow(frames[trail_i], cmap='gray', vmin=0, vmax=255, aspect='auto')
            if trail_row is not None:
                ax.axhline(trail_row, color='red', linestyle='-', linewidth=2)
                ax.text(30, trail_row - 25, f"TRAILING row r_b = {trail_row}",
                        color='red', fontsize=11, fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.7, pad=2))
            off_info = f"+?"
            if T_off is not None:
                off_info = f"{T_off:.6f}"
            info = (f"Pulse {p_idx} TRAILING\n"
                    f"Frame idx in MKV: {trail_i}\n"
                    f"T_off (measured) = {off_info}\n"
                    f"Expected (PPS+0.1): {pps + LED_PULSE_S}\n"
                    f"Residual to PPS+100ms = {off_err:+.1f} ms" if off_err is not None else
                    f"Trailing edge not row-resolvable")
            ax.text(0.02, 0.98, info, transform=ax.transAxes,
                    color='yellow', fontsize=9, fontweight='bold',
                    verticalalignment='top', family='monospace',
                    bbox=dict(facecolor='black', alpha=0.75, pad=4))
        else:
            ax.text(0.5, 0.5, f"Pulse {p_idx} TRAILING in VBI\n(case {case})",
                    transform=ax.transAxes, ha='center', va='center',
                    color='white', fontsize=10, family='monospace',
                    bbox=dict(facecolor='darkred', alpha=0.8, pad=6))
            ax.set_facecolor('#222')
        ax.set_yticks([0, 270, 540, 810, 1079])
        ax.set_xticks([])

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    print(f"Showing {n} pulses. Close the plot window to exit.", file=sys.stderr)
    plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mkv', required=True)
    ap.add_argument('--ft-dir', required=True)
    ap.add_argument('--probe-frames', type=int, default=250)
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--plot', action='store_true',
                    help='Show matplotlib GUI with leading/trailing frames + '
                         'row profiles + detected transition rows for each pulse')
    args = ap.parse_args()

    if not os.path.isfile(args.mkv):
        print("MKV not found: %s" % args.mkv, file=sys.stderr)
        return 2
    if not os.path.isdir(args.ft_dir):
        print("FT dir not found: %s" % args.ft_dir, file=sys.stderr)
        return 2

    results, frames, pulses, row_means = verify(
        args.mkv, args.ft_dir,
        probe_frames=args.probe_frames, verbose=not args.quiet)

    if args.plot and results:
        mkv_t0 = parse_mkv_filename_ts(args.mkv)
        band = locate_led_band(frames)
        if band is not None:
            plot_results(results, frames, pulses, row_means, band, mkv_t0)

    return 0 if results else 1


if __name__ == '__main__':
    sys.exit(main())
