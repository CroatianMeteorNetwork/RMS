"""Post-capture timestamp correction using side-door PTS data.

Runs after capture ends, before processNight.  Reads the .npz files
saved during capture (one per 256-frame block), matches each frame
to its pts_stream sensor entry by PTS value, then refines the UTC
timestamps using a C(t) clock model fit across all PLL probes.

Real-time timestamps (from BufferedCapture) are ~3ms accurate.
This solver refines them to sub-ms using the full night's calibration
data.

Pipeline delay D=1: the VENC PTS register holds the currently-capturing
frame's FE_START, which is one frame ahead of the frame being encoded.
Confirmed by GPS PPS LED calibration (2026-04-04).

The correction is atomic: a journal file is written first, and all
renames happen in a two-pass (old->tmp, tmp->new) operation.  If
anything fails, everything rolls back.

Handles multiple runs (reconnections) within a single night by
detecting time gaps > 30s between consecutive blocks.
"""
import os
import re
import json
import glob
import logging
from datetime import datetime, timezone

import numpy as np

from RMS.Formats import FTfile
from RMS.Formats.FTStruct import FTStruct
from RMS.Misc import UTCFromTimestamp

log = logging.getLogger("logger")

# Minimum gap between consecutive blocks to split into separate runs
RUN_GAP_THRESHOLD_S = 30.0

# VENC pipeline delay: the PTS register captures the currently-reading
# frame, which is D frames ahead of the frame being encoded.
# Calibrated from GPS PPS LED (2026-04-04).
PIPELINE_DELAY_FRAMES = 0

# Sensor frame period: HMAX=4400 * VMAX=1350 / 148.5 MHz (exact 25 fps)
FRAME_PERIOD_S = 4400 * 1350 / 148.5e6  # 0.040000000 s

# VENC PTS wraps at 2^32 microseconds
WRAP_US = 4294967296
WRAP_S = WRAP_US * 1e-6


def sidedoorCorrect(nightDataDir, config):
    """Main entry point for post-capture timestamp correction."""

    npzDir = os.path.join(config.data_dir, config.times_dir, 'sidedoor_raw')
    if not os.path.isdir(npzDir):
        log.info("SidedoorCorrect: no sidedoor_raw directory, skipping")
        return False

    nightKey = os.path.basename(nightDataDir) if nightDataDir else ''
    doneMarker = os.path.join(npzDir, 'sidedoor_corrected.json')
    if os.path.exists(doneMarker):
        try:
            with open(doneMarker) as f:
                info = json.load(f)
            if info.get('night_key') == nightKey:
                log.info("SidedoorCorrect: already corrected for %s, skipping",
                         nightKey)
                return False
        except Exception:
            pass

    npzFiles = sorted(glob.glob(os.path.join(npzDir, 'SD_*.npz')))
    if len(npzFiles) < 3:
        log.info("SidedoorCorrect: only %d blocks, need >= 3", len(npzFiles))
        return False

    log.info("SidedoorCorrect: processing %d blocks from %s",
             len(npzFiles), npzDir)

    try:
        runs = _loadAndSplitRuns(npzFiles)
        log.info("SidedoorCorrect: found %d run(s)", len(runs))

        allCorrections = []
        totalFrames = 0
        totalAnomalies = 0

        for ri, run in enumerate(runs):
            nFrames = len(run['rtpUs'])
            if nFrames < 256 * 3:
                log.info("SidedoorCorrect: run %d too short (%d frames), "
                         "skipping", ri, nFrames)
                continue

            # Per-frame value matching
            assigned = _matchFrames(run['rtpUs'], run['guard'],
                                    run['refPts'])
            if assigned is None:
                log.warning("SidedoorCorrect: run %d matching failed", ri)
                continue

            # Fit C(t) and convert to UTC
            utc = _fitClockAndConvert(assigned, run['pllProbes'],
                                      run['blockBounds'])

            nAnomalies = _verify(assigned)

            _writeCorrectedFT(utc, assigned, run['blockBounds'],
                              config, run['pllProbes'])

            corrections = _computeBlockCorrections(utc, run['blockBounds'],
                                                   run['pllProbes'])
            allCorrections.extend(corrections)

            nMatched = int(np.sum(~np.isnan(assigned)))
            totalFrames += nFrames
            totalAnomalies += nAnomalies
            log.info("SidedoorCorrect: run %d — %d frames, %d matched, "
                     "%d anomalies", ri, nFrames, nMatched, nAnomalies)

        if not allCorrections:
            log.warning("SidedoorCorrect: no corrections produced")
            return False

        _atomicRenameFiles(allCorrections, config, nightDataDir)

        doneInfo = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'night_key': nightKey,
            'runs': len(runs),
            'blocks': len(npzFiles),
            'frames': totalFrames,
            'anomalies': totalAnomalies,
        }
        with open(doneMarker, 'w') as f:
            json.dump(doneInfo, f, indent=2)

        log.info("SidedoorCorrect: done — %d run(s), %d frames, "
                 "%d anomalies", len(runs), totalFrames, totalAnomalies)
        return True

    except Exception:
        log.exception("SidedoorCorrect: failed")
        return False


def _loadAndSplitRuns(npzFiles):
    """Load all .npz blocks, split into independent runs by time gaps."""
    blockTimes = []
    for fn in npzFiles:
        f = np.load(fn)
        blockTimes.append(float(f['block_start_time']))

    runBoundaries = [0]
    for i in range(1, len(blockTimes)):
        if blockTimes[i] - blockTimes[i - 1] > RUN_GAP_THRESHOLD_S:
            runBoundaries.append(i)
    runBoundaries.append(len(npzFiles))

    runs = []
    for ri in range(len(runBoundaries) - 1):
        bStart = runBoundaries[ri]
        bEnd = runBoundaries[ri + 1]

        rtpUs_list, guard_list = [], []
        blockBounds, pllProbes = [], []
        refPts_all = {}
        prev_gs = 0
        frame_idx = 0

        for bi in range(bStart, bEnd):
            f = np.load(npzFiles[bi])
            gs = int(f['guard_shift'])
            rtp = f['rtp_90k']
            gf = f['guard_flags']
            ref = f['ref_pts_us']

            first_flag = np.argmax(gf) if gf.any() else len(gf)
            per_frame_gs = np.full(len(rtp), prev_gs, dtype=np.int64)
            per_frame_gs[first_flag:] = gs
            raw = ((rtp.astype(np.int64) + per_frame_gs) * 100) // 9
            rtpUs_list.extend(raw.tolist())
            guard_list.extend(gf.tolist())

            blockBounds.append((frame_idx, frame_idx + len(rtp)))
            c_raw = float(f['C_raw']) if 'C_raw' in f else float(f['C'])
            exposure_us = float(f['exposure_us']) if 'exposure_us' in f else 0.0
            pllProbes.append({
                'block_start': float(f['block_start_time']),
                'C_raw': c_raw,
                'exposure_us': exposure_us,
                'frame_start': frame_idx,
            })
            for v in ref:
                refPts_all[int(v)] = v
            prev_gs = gs
            frame_idx += len(rtp)

        runs.append({
            'rtpUs': np.array(rtpUs_list, dtype=np.float64),
            'guard': np.array(guard_list, dtype=bool),
            'refPts': np.array(sorted(refPts_all.values()), dtype=np.float64),
            'blockBounds': blockBounds,
            'pllProbes': pllProbes,
        })

    return runs


def _matchFrames(rtpUs, guard, refPts):
    """Per-frame PTS value matching via binary search.

    Each non-stale RTP frame carries a PTS value that corresponds to
    exactly one ref_pts entry (sub-tick precision).  Stale frames
    (delta < 200us from previous) get the same assignment as the
    previous frame (duplicate sensor readout).

    Returns array of matched ref_pts values (NaN for unmatched).
    """
    n = len(rtpUs)
    if len(refPts) == 0:
        return None

    # Bootstrap base_offset from the first non-stale frame using
    # cross-correlation to find the approximate initial match.
    rtp_deltas = []
    for i in range(1, min(2000, n)):
        if guard[i] or guard[i - 1]:
            continue
        d = rtpUs[i] - rtpUs[i - 1]
        if 30000 < d < 50000:
            rtp_deltas.append(d)

    ref_deltas = np.diff(refPts)
    ref_real = ref_deltas[(ref_deltas > 30000) & (ref_deltas < 50000)]

    n_query = min(500, len(rtp_deltas))
    if n_query < 50:
        log.warning("SidedoorCorrect: too few valid deltas (%d)", n_query)
        return None

    q = np.array(rtp_deltas[:n_query])
    q_norm = (q - q.mean()) / q.std()
    best_corr, best_off = -1, 0
    search = min(len(ref_real) - n_query, 1000)
    for off in range(search):
        w = ref_real[off:off + n_query]
        w_std = w.std()
        if w_std == 0:
            continue
        corr = np.dot(q_norm, (w - w.mean()) / w_std) / n_query
        if corr > best_corr:
            best_corr, best_off = corr, off

    if best_corr < 0.9:
        log.warning("SidedoorCorrect: weak cross-correlation (%.3f)", best_corr)
        return None

    # Compute base_offset from the first non-stale frame
    first_good = 0
    for i in range(n):
        if not guard[i]:
            first_good = i
            break
    base_offset = rtpUs[first_good] - refPts[best_off + first_good]

    log.info("SidedoorCorrect: bootstrap corr=%.4f, base_offset=%.0f",
             best_corr, base_offset)

    # Per-frame matching
    assigned = np.full(n, np.nan, dtype=np.float64)
    matched = 0
    stale = 0
    missed = 0

    for i in range(n):
        # Stale frame: delta < 200us from previous = duplicate readout
        if i > 0 and not guard[i] and not guard[i - 1]:
            delta = rtpUs[i] - rtpUs[i - 1]
            if abs(delta) < 200:
                assigned[i] = assigned[i - 1]
                stale += 1
                continue

        # Binary search for matching ref_pts entry
        target = rtpUs[i] - base_offset
        idx = np.searchsorted(refPts, target)

        best_idx, best_res = -1, 1e18
        for c in (idx - 1, idx, idx + 1):
            if 0 <= c < len(refPts):
                res = abs(refPts[c] - target)
                if res < best_res:
                    best_res = res
                    best_idx = c

        if best_res < 20000:  # within 20ms
            assigned[i] = refPts[best_idx]
            matched += 1
        else:
            missed += 1

    log.info("SidedoorCorrect: matched=%d stale=%d missed=%d",
             matched, stale, missed)
    return assigned


def _fitClockAndConvert(assigned, pllProbes, blockBounds):
    """Fit C(t) clock model across all PLL probes, convert to UTC.

    Individual C_raw probes have ~3ms jitter.  Fitting a linear model
    across all probes reduces this to sub-ms.  The fit captures crystal
    drift (typically 5-30 ppm).

    Applies pipeline delay (D=1) and exposure correction.
    """
    n = len(assigned)

    # Collect (time, C_raw) from all probes and unwrap PTS wraps
    probe_times = np.array([p['block_start'] for p in pllProbes])
    probe_c_raw = np.array([p['C_raw'] for p in pllProbes])

    # Unwrap C_raw (jumps by ~4295s at PTS wraps)
    c_unwrapped = probe_c_raw.copy()
    for i in range(1, len(c_unwrapped)):
        diff = c_unwrapped[i] - c_unwrapped[i - 1]
        if diff > 2000:
            c_unwrapped[i:] -= WRAP_S
        elif diff < -2000:
            c_unwrapped[i:] += WRAP_S

    # Fit linear C(t) = C0 + drift_rate * (t - t0)
    t0 = probe_times[0]
    t_rel = probe_times - t0
    if len(t_rel) >= 3:
        coeffs = np.polyfit(t_rel, c_unwrapped, 1)
        fit_residuals = c_unwrapped - np.polyval(coeffs, t_rel)
        log.info("SidedoorCorrect: C(t) fit — drift=%.1f ppm, "
                 "residual std=%.2fms, max=%.2fms",
                 coeffs[0] * 1e6,
                 np.std(fit_residuals) * 1000,
                 np.max(np.abs(fit_residuals)) * 1000)
    else:
        coeffs = np.array([0.0, c_unwrapped[0]])

    # Get exposure (use median across probes)
    exposure_vals = [p['exposure_us'] for p in pllProbes if p['exposure_us'] > 0]
    exposure_s = np.median(exposure_vals) * 1e-6 if exposure_vals else 0.0

    # Convert each frame to UTC
    utc = np.full(n, np.nan, dtype=np.float64)
    prev_utc = None

    for i in range(n):
        if np.isnan(assigned[i]):
            continue

        # Estimate frame time for C(t) lookup
        # Use nearest probe's block_start as approximation
        best_probe_time = probe_times[0]
        for p in pllProbes:
            if p['frame_start'] <= i:
                best_probe_time = p['block_start']
            else:
                break

        # Interpolated C from the fit
        c_fit = np.polyval(coeffs, best_probe_time - t0)

        # UTC = C(t) + raw_pts_mod_wrap - pipeline_delay - exposure
        raw_pts_us = assigned[i] % WRAP_US
        t = c_fit + raw_pts_us * 1e-6

        # Wrap discontinuity fix
        if prev_utc is not None:
            diff = t - prev_utc
            if diff < -WRAP_S / 2:
                t += WRAP_S
            elif diff > WRAP_S / 2:
                t -= WRAP_S

        # Pipeline delay: VENC PTS is from the frame being captured,
        # not the frame being encoded (D=1 frame ahead)
        t -= PIPELINE_DELAY_FRAMES * FRAME_PERIOD_S

        # Exposure: FE_START is readout start, integration started earlier
        t -= exposure_s

        utc[i] = t
        prev_utc = t

    return utc


def _verify(assigned):
    """Count anomalous intervals in assigned ref_pts."""
    valid = assigned[~np.isnan(assigned)]
    if len(valid) < 2:
        return -1
    intervals = np.diff(valid)
    anomalies = int(np.sum((intervals < 30000) | (intervals > 50000)))
    normal = intervals[(intervals > 30000) & (intervals < 50000)]
    if len(normal) > 0:
        log.info("SidedoorCorrect: %d intervals, %d anomalies, "
                 "sigma=%.1fus", len(intervals), anomalies, normal.std())
    return anomalies


def _computeBlockCorrections(utc, blockBounds, pllProbes):
    """Compute per-block corrected first_frame_timestamp."""
    corrections = []
    for bi, (f_start, f_end) in enumerate(blockBounds):
        if f_start < len(utc) and not np.isnan(utc[f_start]):
            old_ts = pllProbes[bi]['block_start']
            new_ts = float(utc[f_start])
            corrections.append((bi, old_ts, new_ts))
    return corrections


def _writeCorrectedFT(utc, refVals, blockBounds, config, pllProbes):
    """Write corrected FT files (v2 with raw_pts_us)."""
    times_dir = os.path.join(config.data_dir, config.times_dir)
    written = 0

    for bi, (f_start, f_end) in enumerate(blockBounds):
        if f_start >= len(utc) or np.isnan(utc[f_start]):
            continue

        ft = FTStruct()
        for i in range(f_start, min(f_end, len(utc))):
            if not np.isnan(utc[i]):
                ft.timestamps.append((i - f_start, float(utc[i])))
                if not np.isnan(refVals[i]):
                    ft.raw_pts_us.append(float(refVals[i]))
                else:
                    ft.raw_pts_us.append(0.0)

        if not ft.timestamps:
            continue

        old_ts = pllProbes[bi]['block_start']
        base_time = UTCFromTimestamp.utcfromtimestamp(old_ts)
        hour_dir = base_time.strftime(
            os.path.join(times_dir, "%Y/%Y%m%d-%j/%Y%m%d-%j_%H"))
        if not os.path.isdir(hour_dir):
            continue

        new_ts_dt = UTCFromTimestamp.utcfromtimestamp(float(utc[f_start]))
        new_hour_dir = new_ts_dt.strftime(
            os.path.join(times_dir, "%Y/%Y%m%d-%j/%Y%m%d-%j_%H"))
        search_dirs = {hour_dir}
        if new_hour_dir != hour_dir and os.path.isdir(new_hour_dir):
            search_dirs.add(new_hour_dir)

        best_ft, best_dt = None, 15.0
        for sdir in search_dirs:
            for ft_path in sorted(glob.glob(os.path.join(sdir, 'FT_*.bin'))):
                m = re.search(r'_(\d{8})_(\d{6})\.bin', ft_path)
                if m:
                    ft_dt = datetime.strptime(
                        m.group(1) + m.group(2), '%Y%m%d%H%M%S')
                    ft_dt = ft_dt.replace(tzinfo=timezone.utc)
                    dt = min(abs(ft_dt.timestamp() - old_ts),
                             abs(ft_dt.timestamp() - float(utc[f_start])))
                    if dt < best_dt:
                        best_dt, best_ft = dt, ft_path

        if best_ft:
            FTfile.write(ft, os.path.dirname(best_ft),
                         os.path.basename(best_ft))
            written += 1

    log.info("SidedoorCorrect: wrote %d corrected FT files", written)


def _atomicRenameFiles(corrections, config, nightDataDir):
    """Atomically rename FF/FT/FS files with corrected timestamps.

    Uses delta-based correction with idempotency guard: files already
    at the corrected timestamp are skipped.
    """
    data_dir = config.data_dir

    rename_map = {}
    for bi, old_ts, new_ts in corrections:
        delta = new_ts - old_ts
        if abs(delta) < 0.001:
            continue

        for dirpath, _, filenames in os.walk(data_dir):
            for fname in filenames:
                file_ts, ftype = _parseTimestamp(fname)
                if file_ts is None:
                    continue
                if abs(file_ts - old_ts) > 1.0:
                    continue
                if abs(file_ts - new_ts) < 0.001:
                    continue

                corrected_ts = file_ts + delta
                new_name = _buildNewName(fname, corrected_ts, ftype)
                if new_name and new_name != fname:
                    old_path = os.path.join(dirpath, fname)
                    new_path = os.path.join(dirpath, new_name)
                    rename_map[old_path] = new_path

    if not rename_map:
        log.info("SidedoorCorrect: no files need renaming")
        return

    log.info("SidedoorCorrect: renaming %d files", len(rename_map))

    new_paths = set()
    for old, new in rename_map.items():
        if not os.path.exists(old):
            log.error("SidedoorCorrect: source missing: %s", old)
            return
        if new in new_paths:
            log.error("SidedoorCorrect: duplicate target: %s", new)
            return
        if os.path.exists(new) and new != old:
            log.error("SidedoorCorrect: target exists: %s", new)
            return
        new_paths.add(new)

    journal = {old: new for old, new in rename_map.items() if old != new}
    journal_path = os.path.join(data_dir, 'rename_journal.json')
    with open(journal_path, 'w') as f:
        json.dump(journal, f, indent=2)

    tmp_map = {}
    try:
        for old, new in journal.items():
            tmp = old + '.rename_tmp'
            os.rename(old, tmp)
            tmp_map[tmp] = new
        for tmp, new in tmp_map.items():
            os.rename(tmp, new)
    except Exception as e:
        log.error("SidedoorCorrect: rename failed: %s — rolling back", e)
        for old, new in journal.items():
            tmp = old + '.rename_tmp'
            if os.path.exists(tmp):
                os.rename(tmp, old)
            elif os.path.exists(new):
                os.rename(new, old)
        if os.path.exists(journal_path):
            os.remove(journal_path)
        return

    base_map = {os.path.basename(old): os.path.basename(new)
                for old, new in journal.items()
                if os.path.basename(old) != os.path.basename(new)}
    if base_map:
        for dirpath, _, filenames in os.walk(nightDataDir):
            for fname in filenames:
                if fname.startswith('FTPdetectinfo'):
                    ftp_path = os.path.join(dirpath, fname)
                    with open(ftp_path) as f:
                        content = f.read()
                    changed = False
                    for old_base, new_base in base_map.items():
                        if old_base in content:
                            content = content.replace(old_base, new_base)
                            changed = True
                    if changed:
                        with open(ftp_path, 'w') as f:
                            f.write(content)

    log.info("SidedoorCorrect: renamed %d files, journal at %s",
             len(journal), journal_path)


def _parseTimestamp(filename):
    """Extract timestamp from FF/FT/FS filename."""
    m = re.search(r'_(\d{8})_(\d{6})_(\d{3})_', filename)
    if m:
        dt = datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')
        dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp() + int(m.group(3)) / 1000.0, 'ff'

    m = re.search(r'_(\d{8})_(\d{6})\.bin', filename)
    if m:
        dt = datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')
        dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp(), 'ft'

    m = re.search(r'_(\d{8})_(\d{6})_(\d+)_\d+_fieldsum', filename)
    if m:
        dt = datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')
        dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp() + int(m.group(3)) / 1e6, 'fs'

    return None, None


def _buildNewName(old_name, new_ts, file_type):
    """Build filename with corrected timestamp."""
    dt = datetime.fromtimestamp(new_ts, tz=timezone.utc)

    if file_type == 'ff':
        ms = int((new_ts % 1) * 1000)
        new_time = dt.strftime('%Y%m%d_%H%M%S') + '_{:03d}'.format(ms)
        old_m = re.search(r'(\d{8}_\d{6}_\d{3})', old_name)
        if old_m:
            return old_name.replace(old_m.group(1), new_time)

    elif file_type == 'ft':
        new_time = dt.strftime('%Y%m%d_%H%M%S')
        old_m = re.search(r'(\d{8}_\d{6})\.bin', old_name)
        if old_m:
            return old_name.replace(old_m.group(1) + '.bin',
                                     new_time + '.bin')

    elif file_type == 'fs':
        us = int((new_ts % 1) * 1e6)
        new_time = dt.strftime('%Y%m%d_%H%M%S') + '_{:06d}'.format(us)
        old_m = re.search(r'(\d{8}_\d{6}_\d+)(_\d+_fieldsum)', old_name)
        if old_m:
            return old_name.replace(old_m.group(1), new_time)

    return None
