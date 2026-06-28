""" Lightweight, dependency-free memory profiler for diagnosing RMS capture OOM/leak issues.

Walks the RMS process tree (rooted at StartCapture) and periodically logs per-process
memory broken down into the categories that distinguish the competing leak hypotheses:

  - RssAnon       : anonymous private memory (malloc/GLib/GStreamer native buffers,
                    decoder/jitterbuffer/rtspsrc state). Growth here in the *capture*
                    process => native in-process leak across pipeline rebuilds.
                    A multiprocessing start-method change (forkserver) will NOT help this.

  - RssShmem      : shared-memory pages (the 256*W*H mp.Array frame buffers). High but
                    flat is just the baseline cost (~2 buffers per camera).

  - Private_Dirty : pages this PID has actually written and now owns. Growth concentrated
                    in *child* PIDs (RawFrameSaver/Compressor/pool workers) on top of the
                    shared buffers => fork() COW dirtying of inherited buffers.
                    forkserver/spawn WOULD help this.

  - Pss           : proportional set size. Shared pages are divided among the processes
                    mapping them, so summed Pss is the true, non-double-counted physical
                    footprint of the whole tree. THIS is the headline number: if total Pss
                    climbs monotonically across the night, the box has a real leak; which
                    column it concentrates in tells you where.

Also logs /dev/shm usage, system MemAvailable/Committed_AS, and any recent kernel
oom-kill lines so the killed PID and its RSS-at-death are captured.

Pure stdlib, Linux /proc only. Designed to be a no-op import failure away from never
touching the capture hot path. Sampling cost at a 60 s interval is negligible.

Enable from StartCapture by setting the environment variable before launch:

    export RMS_MEMPROFILE=60      # sample interval in seconds (any positive number)
    python -m RMS.StartCapture -c .config

Output goes to the standard RMS log under the 'memprofile' logger name, so it lands in
the same log file you already collect, interleaved with the capture/reconnect lines.
"""

from __future__ import print_function

import os
import glob
import threading
import time
import logging


PAGE = 4096


def _read_status(pid):
    """Read /proc/<pid>/status into a dict of the VmRSS/Rss* fields (in bytes).

    Cheap (single small file), always present. Returns {} if the pid vanished.
    """
    out = {}
    try:
        with open("/proc/{}/status".format(pid)) as f:
            for line in f:
                if line.startswith(("VmRSS:", "RssAnon:", "RssFile:", "RssShmem:",
                                    "VmSwap:", "VmSize:")):
                    key, val = line.split(":", 1)
                    # value looks like "  123456 kB"
                    out[key.strip()] = int(val.strip().split()[0]) * 1024
    except (IOError, OSError, ValueError, IndexError):
        return {}
    return out


def _read_rollup(pid):
    """Read /proc/<pid>/smaps_rollup for Pss/Private_Dirty/Shared_Dirty (bytes).

    smaps_rollup is a single pre-aggregated file (kernel >= 4.14) so it is cheap.
    Falls back to summing /proc/<pid>/smaps on older kernels. Returns {} on failure.
    """
    fields = ("Pss:", "Private_Dirty:", "Shared_Dirty:", "Swap:")
    out = {}
    path = "/proc/{}/smaps_rollup".format(pid)
    if not os.path.exists(path):
        path = "/proc/{}/smaps".format(pid)
    try:
        with open(path) as f:
            for line in f:
                for fld in fields:
                    if line.startswith(fld):
                        key = fld.rstrip(":")
                        out[key] = out.get(key, 0) + int(line.split()[1]) * 1024
                        break
    except (IOError, OSError, ValueError, IndexError):
        return {}
    return out


def _cmdline(pid):
    try:
        with open("/proc/{}/cmdline".format(pid), "rb") as f:
            return f.read().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
    except (IOError, OSError):
        return ""


def _ppid(pid):
    try:
        with open("/proc/{}/stat".format(pid)) as f:
            # field 4 is ppid; comm (field 2) may contain spaces/parens, so split on ')'
            data = f.read()
            after = data[data.rfind(")") + 2:].split()
            return int(after[1])
    except (IOError, OSError, ValueError, IndexError):
        return None


def _label(pid, root_pid):
    """Best-effort human label for a PID from its cmdline / thread name."""
    if pid == root_pid:
        return "StartCapture(root)"
    cmd = _cmdline(pid)
    low = cmd.lower()
    # Python multiprocessing children set their process name in argv on spawn/forkserver,
    # but under plain fork they share the parent argv. Fall back to comm.
    table = [
        ("bufferedcapture", "BufferedCapture"),
        ("rawframe", "RawFrameSaver"),
        ("compress", "Compressor"),
        ("queuedpool", "DetectionPool"),
        ("pool", "DetectionPool"),
        ("liveview", "LiveViewer"),
        ("upload", "UploadManager"),
        ("eventmonitor", "EventMonitor"),
    ]
    for needle, name in table:
        if needle in low:
            return name
    try:
        with open("/proc/{}/comm".format(pid)) as f:
            comm = f.read().strip()
    except (IOError, OSError):
        comm = "?"
    return "{}[{}]".format(comm, "fork-child" if cmd else "")


def _tree_pids(root_pid):
    """All PIDs in the tree rooted at root_pid (root + all descendants)."""
    children = {}
    for entry in glob.glob("/proc/[0-9]*"):
        try:
            pid = int(os.path.basename(entry))
        except ValueError:
            continue
        pp = _ppid(pid)
        if pp is not None:
            children.setdefault(pp, []).append(pid)

    seen = []
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.append(pid)
        stack.extend(children.get(pid, []))
    return seen


def _meminfo():
    out = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(("MemTotal:", "MemAvailable:", "Committed_AS:",
                                    "Shmem:", "SwapFree:")):
                    key, val = line.split(":", 1)
                    out[key.strip()] = int(val.strip().split()[0]) * 1024
    except (IOError, OSError, ValueError):
        pass
    return out


def _dev_shm_bytes():
    total = 0
    try:
        for root, _dirs, files in os.walk("/dev/shm"):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _recent_oom(seen_marker):
    """Return any new kernel oom-kill lines from dmesg since last call.

    Best-effort: reads /var/log/kern.log if dmesg is not readable. Returns ([], marker)
    if nothing is available (no permissions). marker is the last line index consumed.
    """
    lines = []
    try:
        with open("/var/log/kern.log") as f:
            content = f.readlines()
        for ln in content[seen_marker:]:
            low = ln.lower()
            if "out of memory" in low or "oom-kill" in low or "killed process" in low:
                lines.append(ln.rstrip())
        return lines, len(content)
    except (IOError, OSError):
        return [], seen_marker


def _mb(n):
    return n / (1024.0 * 1024.0)


def sample(root_pid):
    """Build one human-readable report block for the whole RMS tree."""
    pids = _tree_pids(root_pid)
    rows = []
    tot = dict(Pss=0, RssAnon=0, RssShmem=0, RssFile=0, Private_Dirty=0, Swap=0)

    for pid in pids:
        st = _read_status(pid)
        if not st:
            continue
        ru = _read_rollup(pid)
        anon = st.get("RssAnon", 0)
        shmem = st.get("RssShmem", 0)
        rfile = st.get("RssFile", 0)
        pss = ru.get("Pss", st.get("VmRSS", 0))
        pdirty = ru.get("Private_Dirty", 0)
        swap = st.get("VmSwap", 0)
        tot["Pss"] += pss
        tot["RssAnon"] += anon
        tot["RssShmem"] += shmem
        tot["RssFile"] += rfile
        tot["Private_Dirty"] += pdirty
        tot["Swap"] += swap
        rows.append((pss, pid, _label(pid, root_pid), anon, shmem, pdirty, swap))

    rows.sort(reverse=True)  # largest Pss first

    mi = _meminfo()
    lines = []
    lines.append("MEMPROFILE  tree_pss={:.0f}MB  anon={:.0f}MB  shmem={:.0f}MB  "
                 "priv_dirty={:.0f}MB  swap={:.0f}MB  | MemAvail={:.0f}MB  "
                 "Committed={:.0f}MB  /dev/shm={:.0f}MB".format(
                     _mb(tot["Pss"]), _mb(tot["RssAnon"]), _mb(tot["RssShmem"]),
                     _mb(tot["Private_Dirty"]), _mb(tot["Swap"]),
                     _mb(mi.get("MemAvailable", 0)), _mb(mi.get("Committed_AS", 0)),
                     _mb(_dev_shm_bytes())))
    lines.append("  {:>7} {:<20} {:>9} {:>9} {:>9} {:>9} {:>8}".format(
        "PID", "role", "Pss(MB)", "Anon(MB)", "Shmem(MB)", "PrivD(MB)", "Swap(MB)"))
    for pss, pid, label, anon, shmem, pdirty, swap in rows:
        lines.append("  {:>7} {:<20} {:>9.1f} {:>9.1f} {:>9.1f} {:>9.1f} {:>8.1f}".format(
            pid, label[:20], _mb(pss), _mb(anon), _mb(shmem), _mb(pdirty), _mb(swap)))
    return "\n".join(lines)


def start_background_logger(root_pid=None, interval=60.0, logger=None):
    """Spawn a daemon thread that logs a memory report every *interval* seconds.

    Safe to call once from the root StartCapture process. Never raises into the caller.

    Arguments:
        root_pid: [int] Root of the tree to profile. Default: this process.
        interval: [float] Seconds between samples.
        logger: [logging.Logger] Where to log. Default: 'memprofile' logger.
    """
    if root_pid is None:
        root_pid = os.getpid()
    if logger is None:
        logger = logging.getLogger("memprofile")

    def _run():
        oom_marker = 0
        # Prime the oom marker so we only report kills that happen after we start.
        _, oom_marker = _recent_oom(0)
        logger.info("MEMPROFILE started: root_pid=%d interval=%.0fs page=%dB",
                    root_pid, interval, PAGE)
        while True:
            try:
                report = sample(root_pid)
                logger.info("\n" + report)
                oom_lines, oom_marker = _recent_oom(oom_marker)
                for ol in oom_lines:
                    logger.warning("MEMPROFILE OOM-KILL: %s", ol)
            except Exception as e:  # never let the profiler crash capture
                logger.debug("MEMPROFILE sample failed: %s", e)
            time.sleep(interval)

    th = threading.Thread(target=_run, name="MemoryProfiler", daemon=True)
    th.start()
    return th


if __name__ == "__main__":
    # Standalone mode: profile an arbitrary pid tree from the shell, no RMS needed.
    #   python -m Utils.MemoryProfiler <root_pid> [interval_s]
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rpid = int(sys.argv[1]) if len(sys.argv) > 1 else os.getpid()
    iv = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    print("Profiling pid tree under {} every {:.0f}s (Ctrl+C to stop)".format(rpid, iv))
    while True:
        print(sample(rpid))
        print("-" * 100)
        time.sleep(iv)
