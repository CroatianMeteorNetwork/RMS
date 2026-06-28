""" Dependency-free, whole-box memory profiler for diagnosing RMS capture OOM/leak issues
in a SINGLE overnight run.

It is built to be decisive on the first night: it captures, simultaneously, the evidence
for every competing hypothesis so you never have to go back for "more data".

WHAT IT MEASURES, AND WHICH HYPOTHESIS EACH METRIC SETTLES
---------------------------------------------------------
Per process (all RMS processes on the box, not just one camera):

  Pss            Proportional set size. Shared pages divided among the processes mapping
                 them, so the SUM over all RMS PIDs is the true, non-double-counted
                 physical footprint of the whole station. Headline number: if total Pss
                 climbs across the night, the box has a real leak.

  RssAnon        Anonymous private memory = native malloc/GLib/GStreamer state (rtspsrc
                 16MB UDP buffer, avdec_h264, jitterbuffer, queues). Climbing in the
                 BufferedCapture PID with rebuild count => native in-process leak.
                 *** A forkserver/spawn start-method change will NOT help this. ***

  RssShmem       Shared-memory pages = the 256*W*H mp.Array frame buffers (~225 MiB each
                 at 720p). High but flat is just the per-camera baseline.

  Private_Dirty  Pages a PID has written and now owns. Growth concentrated in *child*
                 PIDs (RawFrameSaver/Compressor/pool) on top of the shared buffers =>
                 fork() COW dirtying of inherited buffers. *** forkserver WOULD help. ***

  Threads        /proc Threads count. Monotonic growth across pipeline rebuilds =>
                 leaked bus/rtspsrc threads (the "abandoned bus thread" path). Decisive
                 and impossible to confuse with anything else.

  FDs            Open file descriptors. Monotonic growth => leaked sockets/files.

In-process native probe (logged by BufferedCapture itself, glibc mallinfo2):

  uordblks       Bytes currently handed out by malloc and still referenced. Growing =>
                 a GENUINE leak (live, reachable allocations).
  fordblks       Free bytes retained inside arenas (not returned to OS). Large/growing
                 while uordblks is flat => arena fragmentation/retention, NOT a leak.
                 Fixed by MALLOC_ARENA_MAX / malloc_trim, NOT by forkserver. This is the
                 mechanism every other instrument is blind to, so it is the one most
                 likely to have made previous nights inconclusive.
  hblkhd         Bytes in mmap'd large allocations (the big GStreamer/decoder buffers).
                 Growing => large-buffer leak.

System: MemTotal/MemAvailable/Committed_AS/SwapFree/Shmem and /dev/shm size, so the OOM
moment and any kernel oom-kill line (killed PID + RSS-at-death) are captured too.

OUTPUTS
-------
  - Human-readable blocks to the RMS log (interleaved with capture/reconnect lines).
  - A machine CSV (one row per PID per sample) at $RMS_MEMPROFILE_CSV (default
    ./rms_memprofile.csv) so the whole night plots in one command.
  - Adaptive cadence: samples every RMS_MEMPROFILE seconds normally, but switches to a
    fast 5 s burst whenever MemAvailable drops below RMS_MEMPROFILE_LOWMB (default 400 MB)
    so the minutes before an OOM kill are captured at high resolution.

ENABLE
------
  export RMS_MEMPROFILE=60            # base sample interval, seconds
  export RMS_MEMPROFILE_CSV=/home/fireballs360/memprofile.csv   # optional
  export RMS_MEMPROFILE_LOWMB=400     # optional low-mem fast-burst threshold (MB)
  # start RMS as usual; one deployed instance observes the WHOLE box (all cameras).

Pure stdlib, Linux /proc only. Never raises into the caller.
"""

from __future__ import print_function

import os
import glob
import threading
import time
import logging
import ctypes


# ----------------------------------------------------------------------------------------
# glibc native allocator introspection (in-process only)
# ----------------------------------------------------------------------------------------

class _Mallinfo2(ctypes.Structure):
    _fields_ = [(n, ctypes.c_size_t) for n in (
        "arena", "ordblks", "smblks", "hblks", "hblkhd",
        "usmblks", "fsmblks", "uordblks", "fordblks", "keepcost")]


_libc = None
try:
    _libc = ctypes.CDLL("libc.so.6", use_errno=True)
    _libc.mallinfo2.restype = _Mallinfo2          # glibc >= 2.33, size_t fields (no overflow)
    _libc.malloc_trim.argtypes = [ctypes.c_size_t]
    _libc.malloc_trim.restype = ctypes.c_int
except (OSError, AttributeError):
    _libc = None


def mallinfo():
    """Return the glibc mallinfo2 struct as a dict of bytes, or {} if unavailable."""
    if _libc is None:
        return {}
    try:
        mi = _libc.mallinfo2()
        return {f: getattr(mi, f) for f, _ in _Mallinfo2._fields_}
    except Exception:
        return {}


def malloc_trim_probe():
    """Measure how much RSS malloc_trim(0) can reclaim RIGHT NOW (in-process).

    Returns (rss_before, rss_after, reclaimed) in bytes. A large 'reclaimed' means the
    growth is arena retention, not a leak. Returns None if unavailable.

    Note: this mutates allocator state, so it is only fired on demand (SIGUSR2), never on
    the periodic path, so it can't mask the natural growth we are trying to observe.
    """
    if _libc is None:
        return None
    before = _read_status(os.getpid()).get("VmRSS", 0)
    try:
        _libc.malloc_trim(0)
    except Exception:
        return None
    after = _read_status(os.getpid()).get("VmRSS", 0)
    return before, after, before - after


# ----------------------------------------------------------------------------------------
# /proc readers
# ----------------------------------------------------------------------------------------

def _read_status(pid):
    out = {}
    try:
        with open("/proc/{}/status".format(pid)) as f:
            for line in f:
                if line.startswith(("VmRSS:", "RssAnon:", "RssFile:", "RssShmem:",
                                    "VmSwap:", "VmSize:", "Threads:")):
                    key, val = line.split(":", 1)
                    parts = val.strip().split()
                    # Threads has no kB unit; memory fields do
                    out[key.strip()] = int(parts[0]) * (1024 if len(parts) > 1 else 1)
    except (IOError, OSError, ValueError, IndexError):
        return {}
    return out


def _read_rollup(pid):
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


def _fd_count(pid):
    try:
        return len(os.listdir("/proc/{}/fd".format(pid)))
    except (IOError, OSError):
        return 0


def _cmdline(pid):
    try:
        with open("/proc/{}/cmdline".format(pid), "rb") as f:
            return f.read().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
    except (IOError, OSError):
        return ""


def _cwd_base(pid):
    try:
        return os.path.basename(os.readlink("/proc/{}/cwd".format(pid)))
    except (IOError, OSError):
        return ""


def _comm(pid):
    try:
        with open("/proc/{}/comm".format(pid)) as f:
            return f.read().strip()
    except (IOError, OSError):
        return "?"


# ----------------------------------------------------------------------------------------
# RMS process discovery (whole box, all cameras)
# ----------------------------------------------------------------------------------------

_RMS_HINTS = ("startcapture", "bufferedcapture", "rms/", "rms.", "rawframesave",
              "compression", "queuedpool", "meteordetect", "reprocess",
              "liveviewer", "uploadmanager", "eventmonitor")

_ROLE_TABLE = [
    ("startcapture", "StartCapture(root)"),
    ("bufferedcapture", "BufferedCapture"),
    ("rawframe", "RawFrameSaver"),
    ("compress", "Compressor"),
    ("queuedpool", "DetectionPool"),
    ("meteordetect", "DetectionPool"),
    ("liveview", "LiveViewer"),
    ("upload", "UploadManager"),
    ("eventmonitor", "EventMonitor"),
    ("reprocess", "Reprocess"),
]


def _all_pids():
    pids = []
    for entry in glob.glob("/proc/[0-9]*"):
        try:
            pids.append(int(os.path.basename(entry)))
        except ValueError:
            pass
    return pids


def _rms_pids():
    """Every RMS-related PID on the box, with (pid, role, station) labels.

    Matches by cmdline hint so a single deployed profiler instance sees all cameras,
    including fork children that inherit the parent argv.
    """
    found = []
    for pid in _all_pids():
        cmd = _cmdline(pid).lower()
        if not cmd or "python" not in cmd and not any(h in cmd for h in _RMS_HINTS):
            continue
        if not any(h in cmd for h in _RMS_HINTS):
            continue
        role = "RMS"
        for needle, name in _ROLE_TABLE:
            if needle in cmd:
                role = name
                break
        station = _cwd_base(pid) or "?"
        found.append((pid, role, station))
    return found


# ----------------------------------------------------------------------------------------
# system + reporting
# ----------------------------------------------------------------------------------------

def _meminfo():
    out = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(("MemTotal:", "MemAvailable:", "Committed_AS:",
                                    "Shmem:", "SwapFree:", "SwapTotal:")):
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
    for path in ("/var/log/kern.log", "/var/log/syslog", "/var/log/messages"):
        try:
            with open(path) as f:
                content = f.readlines()
        except (IOError, OSError):
            continue
        lines = []
        for ln in content[seen_marker:]:
            low = ln.lower()
            if "out of memory" in low or "oom-kill" in low or "killed process" in low:
                lines.append(ln.rstrip())
        return lines, len(content)
    return [], seen_marker


def _mb(n):
    return n / (1024.0 * 1024.0)


def collect():
    """Collect one full snapshot. Returns (rows, totals, meminfo, devshm).

    rows: list of dicts per RMS PID. totals: aggregate dict. Safe; returns best-effort.
    """
    rows = []
    tot = dict(Pss=0, RssAnon=0, RssShmem=0, RssFile=0, Private_Dirty=0, Swap=0,
               Threads=0, FDs=0)
    for pid, role, station in _rms_pids():
        st = _read_status(pid)
        if not st:
            continue
        ru = _read_rollup(pid)
        row = dict(
            pid=pid, role=role, station=station,
            Pss=ru.get("Pss", st.get("VmRSS", 0)),
            RssAnon=st.get("RssAnon", 0),
            RssShmem=st.get("RssShmem", 0),
            RssFile=st.get("RssFile", 0),
            Private_Dirty=ru.get("Private_Dirty", 0),
            Swap=st.get("VmSwap", 0),
            Threads=st.get("Threads", 0),
            FDs=_fd_count(pid),
        )
        rows.append(row)
        for k in tot:
            tot[k] += row[k]
    rows.sort(key=lambda r: r["Pss"], reverse=True)
    return rows, tot, _meminfo(), _dev_shm_bytes()


def format_report(rows, tot, mi, devshm):
    lines = []
    lines.append("MEMPROFILE  rms_pss={:.0f}MB  anon={:.0f}MB  shmem={:.0f}MB  "
                 "priv_dirty={:.0f}MB  threads={}  fds={}  | MemAvail={:.0f}MB  "
                 "Committed={:.0f}MB  SwapFree={:.0f}MB  /dev/shm={:.0f}MB".format(
                     _mb(tot["Pss"]), _mb(tot["RssAnon"]), _mb(tot["RssShmem"]),
                     _mb(tot["Private_Dirty"]), tot["Threads"], tot["FDs"],
                     _mb(mi.get("MemAvailable", 0)), _mb(mi.get("Committed_AS", 0)),
                     _mb(mi.get("SwapFree", 0)), _mb(devshm)))
    lines.append("  {:>7} {:<18} {:<10} {:>8} {:>8} {:>9} {:>9} {:>8} {:>7} {:>6}".format(
        "PID", "role", "station", "Pss", "Anon", "Shmem", "PrivD", "Swap", "Thr", "FDs"))
    for r in rows:
        lines.append("  {:>7} {:<18} {:<10} {:>8.0f} {:>8.0f} {:>9.0f} {:>9.0f} "
                     "{:>8.0f} {:>7} {:>6}".format(
                         r["pid"], r["role"][:18], r["station"][:10],
                         _mb(r["Pss"]), _mb(r["RssAnon"]), _mb(r["RssShmem"]),
                         _mb(r["Private_Dirty"]), _mb(r["Swap"]), r["Threads"], r["FDs"]))
    return "\n".join(lines)


_CSV_COLS = ["ts", "pid", "role", "station", "Pss", "RssAnon", "RssShmem", "RssFile",
             "Private_Dirty", "Swap", "Threads", "FDs",
             "sys_MemAvailable", "sys_Committed_AS", "sys_SwapFree", "dev_shm"]


def _csv_append(path, rows, mi, devshm, ts):
    try:
        new = not os.path.exists(path)
        with open(path, "a") as f:
            if new:
                f.write(",".join(_CSV_COLS) + "\n")
            for r in rows:
                f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    ts, r["pid"], r["role"], r["station"], r["Pss"], r["RssAnon"],
                    r["RssShmem"], r["RssFile"], r["Private_Dirty"], r["Swap"],
                    r["Threads"], r["FDs"], mi.get("MemAvailable", 0),
                    mi.get("Committed_AS", 0), mi.get("SwapFree", 0), devshm))
    except (IOError, OSError):
        pass


def start_background_logger(root_pid=None, interval=60.0, logger=None,
                            csv_path=None, low_mb=400.0):
    """Spawn a daemon thread logging whole-box RMS memory until process exit.

    Adaptive: drops to a 5 s burst whenever MemAvailable < low_mb so the run-up to an
    OOM kill is captured at high resolution. Never raises into the caller.
    """
    if logger is None:
        logger = logging.getLogger("memprofile")
    if csv_path is None:
        csv_path = os.environ.get("RMS_MEMPROFILE_CSV",
                                  os.path.join(os.getcwd(), "rms_memprofile.csv"))

    def _run():
        oom_marker = 0
        _, oom_marker = _recent_oom(0)
        logger.info("MEMPROFILE started: interval=%.0fs low_mem_burst<%.0fMB csv=%s",
                    interval, low_mb, csv_path)
        while True:
            slp = interval
            try:
                rows, tot, mi, devshm = collect()
                ts = time.strftime("%Y/%m/%d %H:%M:%S")
                logger.info("\n" + format_report(rows, tot, mi, devshm))
                _csv_append(csv_path, rows, mi, devshm, ts)

                oom_lines, oom_marker = _recent_oom(oom_marker)
                for ol in oom_lines:
                    logger.warning("MEMPROFILE OOM-KILL: %s", ol)

                # Adaptive: fast burst when memory is getting tight
                if _mb(mi.get("MemAvailable", 1 << 60)) < low_mb:
                    slp = 5.0
                    logger.warning("MEMPROFILE low memory (<%.0fMB) - fast 5s sampling",
                                   low_mb)
            except Exception as e:
                logger.debug("MEMPROFILE sample failed: %s", e)
            time.sleep(slp)

    th = threading.Thread(target=_run, name="MemoryProfiler", daemon=True)
    th.start()
    return th


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    iv = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    print("Whole-box RMS memory profile every {:.0f}s (Ctrl+C to stop)".format(iv))
    while True:
        rows, tot, mi, devshm = collect()
        print(format_report(rows, tot, mi, devshm))
        print("-" * 110)
        time.sleep(iv)
