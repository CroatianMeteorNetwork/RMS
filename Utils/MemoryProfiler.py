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
import re
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


# Load libc unconditionally; then probe each function separately so that a glibc too old
# for mallinfo2 (e.g. < 2.33, common on Ubuntu 20.04 / Py3.8) still gets malloc_info and
# malloc_trim instead of disabling everything.
_libc = None
_has_mallinfo2 = False
_has_malloc_info = False
try:
    _libc = ctypes.CDLL("libc.so.6", use_errno=True)
    try:
        _libc.malloc_trim.argtypes = [ctypes.c_size_t]
        _libc.malloc_trim.restype = ctypes.c_int
    except AttributeError:
        pass
    try:
        _libc.mallinfo2.restype = _Mallinfo2      # size_t fields, no >2GB overflow
        _has_mallinfo2 = True
    except AttributeError:
        pass
    try:
        # int open_memstream-backed capture of malloc_info(0, FILE*)
        _libc.open_memstream.restype = ctypes.c_void_p
        _libc.open_memstream.argtypes = [ctypes.POINTER(ctypes.c_char_p),
                                         ctypes.POINTER(ctypes.c_size_t)]
        _libc.malloc_info.argtypes = [ctypes.c_int, ctypes.c_void_p]
        _libc.fclose.argtypes = [ctypes.c_void_p]
        _has_malloc_info = True
    except AttributeError:
        pass
except OSError:
    _libc = None


def _mallinfo_via_xml():
    """Parse glibc malloc_info() XML into the same keys mallinfo2 exposes (bytes).

    malloc_info emits per-arena blocks then grand totals. We take the LAST occurrence of
    each total so we get the whole-process aggregate:
        arena    <- <system type="current" size=..>   (heap obtained from OS)
        hblkhd   <- <total  type="mmap"    size=..>    (large mmap'd allocations)
        fordblks <- fast + rest free                   (freed but retained in arenas)
        uordblks <- arena - fordblks                   (in use; a real leak grows this)
    Returns {} if unavailable.
    """
    if _libc is None or not _has_malloc_info:
        return {}
    buf = ctypes.c_char_p()
    size = ctypes.c_size_t()
    fp = _libc.open_memstream(ctypes.byref(buf), ctypes.byref(size))
    if not fp:
        return {}
    try:
        _libc.malloc_info(0, fp)
        _libc.fclose(fp)
        xml = ctypes.string_at(buf) if buf.value else b""
    finally:
        if buf.value:
            # free the open_memstream buffer
            try:
                ctypes.CDLL("libc.so.6").free(buf)
            except Exception:
                pass
    text = xml.decode("ascii", "replace")

    def _last(kind, typ):
        vals = re.findall(r'<%s type="%s"[^>]*size="(\d+)"' % (kind, typ), text)
        return int(vals[-1]) if vals else 0

    arena = _last("system", "current")
    hblkhd = _last("total", "mmap")
    fordblks = _last("total", "fast") + _last("total", "rest")
    uordblks = max(0, arena - fordblks)
    return {"arena": arena, "hblkhd": hblkhd, "fordblks": fordblks,
            "uordblks": uordblks}


def mallinfo():
    """Return glibc allocator stats as a dict of bytes (arena/uordblks/fordblks/hblkhd).

    Prefers mallinfo2 (exact). Falls back to malloc_info() XML on older glibc. Returns {}
    only if neither is available.
    """
    if _has_mallinfo2:
        try:
            mi = _libc.mallinfo2()
            return {f: getattr(mi, f) for f, _ in _Mallinfo2._fields_}
        except Exception:
            pass
    return _mallinfo_via_xml()


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


# Per-process throttle state for the in-process native probe (fresh in each fork child).
_NATIVE_LAST = {}


def logNativeStats(log, role, extra=""):
    """Log this process's native memory stats, for any RMS worker to call from its loop.

    Opt-in via RMS_MEMPROFILE (interval seconds); inert otherwise. Throttled per role so
    it can be called every loop iteration cheaply. Emits VmRSS/RssAnon/threads/fds plus
    the glibc mallinfo split (uordblks = in-use -> real leak; fordblks = retained free ->
    arena retention; hblkhd = mmap'd large allocs). Uniform "MEMPROFILE-NATIVE role=..."
    format across all processes so the log is greppable. Never raises into the caller.

    Arguments:
        log: logger to write to.
        role: [str] short process role, e.g. "RMS-Compress".
        extra: [str] optional extra field, e.g. "rebuilds=12".
    """
    iv = os.environ.get("RMS_MEMPROFILE")
    if not iv:
        return
    try:
        iv = float(iv)
    except (TypeError, ValueError):
        iv = 60.0
    now = time.time()
    if now - _NATIVE_LAST.get(role, 0.0) < iv:
        return
    _NATIVE_LAST[role] = now
    try:
        pid = os.getpid()
        st = _read_status(pid)
        mi = mallinfo()
        mb = 1024.0 * 1024.0
        log.info("MEMPROFILE-NATIVE role=%s pid=%d %s VmRSS=%.0fMB RssAnon=%.0fMB "
                 "threads=%d fds=%d | malloc uordblks=%.0fMB fordblks=%.0fMB "
                 "hblkhd=%.0fMB arena=%.0fMB" % (
                     role, pid, extra,
                     st.get("VmRSS", 0) / mb, st.get("RssAnon", 0) / mb,
                     st.get("Threads", 0), _fd_count(pid),
                     mi.get("uordblks", 0) / mb, mi.get("fordblks", 0) / mb,
                     mi.get("hblkhd", 0) / mb, mi.get("arena", 0) / mb))
    except Exception as e:
        try:
            log.debug("MEMPROFILE-NATIVE failed: %s", e)
        except Exception:
            pass


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


_STATION_RE = re.compile(r'[Ss]tations/([^/ ]+)/')


def _station(pid):
    """Which camera this PID belongs to, from the -c .../Stations/<ID>/.config arg.

    On a multicam box every camera runs from the same cwd (~/source/RMS), so cwd can't
    tell them apart - the station ID is only in the config path on the command line.
    Fork children inherit the parent argv, so they carry it too. Falls back to cwd
    basename for non-RMS children (ffmpeg etc.) that have no config arg.
    """
    m = _STATION_RE.search(_cmdline(pid))
    if m:
        return m.group(1)
    return _cwd_base(pid) or "?"


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
    ("reprocess", "Reprocess/processNight"),
    ("processnight", "Reprocess/processNight"),
    ("timelapse", "Timelapse"),
    ("generatemp4", "GenerateMP4"),
    ("runexternal", "ExternalScript"),
    ("liveview", "LiveViewer"),
    ("upload", "UploadManager"),
    ("eventmonitor", "EventMonitor"),
]

# Post-processing children spawned by RMS that DON'T carry an "RMS" hint in their
# command line (ffmpeg for timelapse/MP4, user external scripts). We catch these by
# walking the descendants of the RMS seed processes and labelling by comm.
_COMM_ROLE = {
    "ffmpeg": "ffmpeg(postproc)",
    "convert": "ImageMagick(postproc)",
}


def _all_pids():
    pids = []
    for entry in glob.glob("/proc/[0-9]*"):
        try:
            pids.append(int(os.path.basename(entry)))
        except ValueError:
            pass
    return pids


def _ppid(pid):
    try:
        with open("/proc/{}/stat".format(pid)) as f:
            data = f.read()
            after = data[data.rfind(")") + 2:].split()
            return int(after[1])
    except (IOError, OSError, ValueError, IndexError):
        return None


def _rms_pids():
    """Every RMS-related PID on the box, with (pid, role, station) labels.

    Two-step discovery so a single deployed instance sees the whole station AND any
    post-processing burst:
      1. Seed = processes whose cmdline carries an RMS hint (all cameras, all children
         that inherit the parent argv).
      2. Expand to every descendant of a seed, so ffmpeg/timelapse/external scripts
         forked during post-processing are included even though they aren't "RMS".
    """
    all_pids = _all_pids()

    # Build child map once.
    children = {}
    for pid in all_pids:
        pp = _ppid(pid)
        if pp is not None:
            children.setdefault(pp, []).append(pid)

    # Seeds by cmdline hint OR a prctl-set RMS- process name (comm).
    seeds = []
    for pid in all_pids:
        cmd = _cmdline(pid).lower()
        if (cmd and any(h in cmd for h in _RMS_HINTS)) or _comm(pid).startswith("RMS-"):
            seeds.append(pid)

    # Expand to all descendants of seeds.
    members = set(seeds)
    stack = list(seeds)
    while stack:
        pid = stack.pop()
        for ch in children.get(pid, []):
            if ch not in members:
                members.add(ch)
                stack.append(ch)

    found = []
    for pid in members:
        comm = _comm(pid)
        role = None
        # 1. A prctl-set RMS- name is the only reliable label: fork children inherit the
        #    parent's argv, so cmdline cannot tell BufferedCapture/Compressor/RawSave/
        #    workers apart - comm (set via setProcName) can.
        if comm.startswith("RMS-"):
            role = comm
        else:
            # 2. Fall back to cmdline hint (catches the StartCapture root and any process
            #    not yet name-tagged) then comm for non-RMS children (ffmpeg etc.).
            cmd = _cmdline(pid).lower()
            for needle, name in _ROLE_TABLE:
                if needle in cmd:
                    role = name
                    break
            if role is None:
                role = _COMM_ROLE.get(comm, "child:" + comm)
        station = _station(pid)
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


def _recent_oom(seen_offset):
    """Return new kernel oom-kill lines since byte offset *seen_offset* (best-effort).

    Reads in binary and decodes with errors='replace' so non-UTF-8 bytes in the kernel
    log can't crash the profiler. Tracks a byte offset (not a line count) so each call
    only reads appended bytes rather than re-reading the whole file every sample.
    """
    for path in ("/var/log/kern.log", "/var/log/syslog", "/var/log/messages"):
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                # First call (prime to current end) or log rotated/truncated: start fresh.
                if seen_offset <= 0 or seen_offset > size:
                    return [], size
                f.seek(seen_offset)
                data = f.read()
        except (IOError, OSError):
            continue
        lines = []
        for ln in data.decode("utf-8", "replace").splitlines():
            low = ln.lower()
            if "out of memory" in low or "oom-kill" in low or "killed process" in low:
                lines.append(ln.rstrip())
        return lines, size
    return [], seen_offset


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
        try:
            _, oom_marker = _recent_oom(0)
        except Exception as e:
            logger.debug("MEMPROFILE oom-scan init failed (continuing): %s", e)
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
    # Standalone decoupled monitor: run as its own long-lived process (screen/tmux/
    # systemd/cron) so it keeps sampling across capture -> post-processing -> upload and
    # survives even if RMS itself is the OOM victim. Catches a post-processing burst
    # (ffmpeg/timelapse/detection) and attributes it to the owning PID/role.
    #
    #   python -m Utils.MemoryProfiler [interval_s] [csv_path] [low_mb]
    import sys
    iv = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    csv = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.getcwd(),
                                                             "rms_memprofile.csv")
    low = float(sys.argv[3]) if len(sys.argv) > 3 else 400.0

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("memprofile")
    print("Whole-box RMS memory monitor: every {:.0f}s, csv={}, "
          "fast-burst<{:.0f}MB. Ctrl+C to stop.".format(iv, csv, low))
    start_background_logger(interval=iv, logger=log, csv_path=csv, low_mb=low)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("stopped")
