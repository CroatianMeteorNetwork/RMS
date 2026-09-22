""" Regression tests for the second capture-pipeline review: child logging on a full queue, bounded
    exits with an abandoned logging queue, polar captureDuration, the reprocess-once guard, the NTP
    offset estimate, config parsing of empty values, float FITS conversion, FFfits/FRbin I/O, the FOV
    KML polygon guard and the updater shell helpers.

    Multiprocessing tests pass their targets as module-level functions so they run under both the
    'fork' and 'forkserver' start methods, and every wait is bounded.
"""

from __future__ import print_function, division, absolute_import

import os
import sys
import time
import logging
import multiprocessing

import pytest


RMS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

sys.path.insert(0, RMS_ROOT)

posix_only = pytest.mark.skipif(os.name != 'posix', reason='POSIX process semantics')


def _startMethods():
    """ Return the multiprocessing start methods available on this platform that stations use. """

    methods = multiprocessing.get_all_start_methods()
    return [m for m in ('fork', 'forkserver') if m in methods]


def _joinOrKill(p, timeout):
    """ Join a process with a timeout and SIGKILL it if it is still alive.

    Arguments:
        p: [Process] Process to join.
        timeout: [float] Seconds to wait.

    Return:
        alive: [bool] True if the process had to be killed.
    """

    p.join(timeout)
    alive = p.is_alive()
    if alive:
        p.kill()
        p.join(5)

    return alive


# ---------------------------------------------------------------------------
# Item 1: initChildLogging must install the drop-on-full handler

def _childLogOnFullQueue(q):
    """ Child side: redirect stderr into logging like a forked RMS child, attach child logging to a
        full queue, and exit 0 only if three log calls return quickly.
    """

    from RMS.Logger import initChildLogging, LoggerWriter, _DroppingQueueHandler

    # Silence the real stderr so a storm (without the fix) does not flood the test output
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)

    # A forked RMS child inherits sys.stderr as a LoggerWriter into the same queue
    sys.stderr = LoggerWriter(logging.getLogger('rmslogger'), logging.WARNING)

    initChildLogging(q, None)

    t_beg = time.monotonic()
    for i in range(3):
        logging.getLogger('rmslogger').info('record %d', i)
    elapsed = time.monotonic() - t_beg

    ok = isinstance(logging.getLogger().handlers[0], _DroppingQueueHandler) and (elapsed < 2.0)
    os._exit(0 if ok else 1)


@posix_only
@pytest.mark.parametrize('method', _startMethods())
def testChildLoggingOnFullQueueDropsRecords(method):
    """ A child logging into a full queue must drop the records instead of recursing through
        handleError -> stderr -> logging.
    """

    ctx = multiprocessing.get_context(method)
    q = ctx.Queue(5)
    for i in range(5):
        q.put(i)

    # Let the feeder push the items so the queue is really full
    time.sleep(0.3)

    p = ctx.Process(target=_childLogOnFullQueue, args=(q,))
    p.start()
    killed = _joinOrKill(p, 15)

    q.cancel_join_thread()
    q.close()

    assert not killed, 'child log calls did not return on a full queue'
    assert p.exitcode == 0


# ---------------------------------------------------------------------------
# Item 2: exits must not hang joining the feeder of an unread logging queue

_SHUTDOWN_SCRIPT = """
import sys, time, multiprocessing
sys.path.insert(0, {root!r})
from RMS.Logger import LoggingManager


def slowListener(q):
    time.sleep(100)


if __name__ == '__main__':
    multiprocessing.set_start_method({method!r}, force=True)

    # A manager whose listener is wedged, with a backlog larger than the pipe buffer
    mgr = LoggingManager()
    mgr.logging_queue = multiprocessing.Queue(30000)
    mgr.listener_process = multiprocessing.Process(target=slowListener, args=(mgr.logging_queue,))
    mgr.listener_process.daemon = True
    mgr.listener_process.start()
    mgr.is_initialized = True
    for i in range(2000):
        mgr.logging_queue.put_nowait('x'*200)

    mgr.shutdownLogging()
    print('shutdown returned', flush=True)
"""


@posix_only
@pytest.mark.parametrize('method', _startMethods())
def testMainExitAfterShutdownWithWedgedListener(method, tmp_path):
    """ After shutdownLogging terminated a wedged listener the interpreter must still exit, instead
        of joining a queue feeder blocked on the dead listener's full pipe.
    """

    import subprocess

    script = tmp_path/'shutdown_exit.py'
    script.write_text(_SHUTDOWN_SCRIPT.format(root=RMS_ROOT, method=method))

    try:
        res = subprocess.run([sys.executable, str(script)], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=40)
    except subprocess.TimeoutExpired:
        pytest.fail('interpreter exit hung after shutdownLogging')

    assert b'shutdown returned' in res.stdout
    assert res.returncode == 0


def _childLogsIntoAbandonedQueue(q):
    """ Child side: attach child logging to a queue nobody reads, log more than a pipe buffer holds
        and return normally.
    """

    from RMS.Logger import initChildLogging

    initChildLogging(q, None)
    for i in range(2000):
        logging.getLogger('rmslogger').warning('x'*200)


@posix_only
@pytest.mark.parametrize('method', _startMethods())
def testChildExitWithAbandonedQueueIsBounded(method):
    """ A child still logging into a queue whose listener is gone (abandoned by a pipeline restart)
        must exit within the bounded flush instead of hanging in the feeder join.
    """

    ctx = multiprocessing.get_context(method)
    q = ctx.Queue(30000)

    p = ctx.Process(target=_childLogsIntoAbandonedQueue, args=(q,))
    t_beg = time.monotonic()
    p.start()
    killed = _joinOrKill(p, 20)
    elapsed = time.monotonic() - t_beg

    q.cancel_join_thread()
    q.close()

    assert not killed, 'child hung at exit joining the logging queue feeder'
    assert p.exitcode == 0
    assert elapsed < 15


# ---------------------------------------------------------------------------
# Item 3: capture children must die with StartCapture

def _exitIfParentGoneChild(parent_pid):
    """ Child side: run the orphan check against the given PID; exit 3 if it returns. """

    from RMS.Misc import exitIfParentGone

    exitIfParentGone(parent_pid, 'test')
    os._exit(3)


def _shortLived():
    """ A process that exits at once. """

    pass


@posix_only
@pytest.mark.parametrize('method', _startMethods())
def testExitIfParentGone(method):
    """ exitIfParentGone exits 0 when the parent PID is gone and returns while it is alive. """

    ctx = multiprocessing.get_context(method)

    # A PID that certainly belonged to a process that is gone now
    dead = ctx.Process(target=_shortLived)
    dead.start()
    dead.join(10)

    p = ctx.Process(target=_exitIfParentGoneChild, args=(dead.pid,))
    p.start()
    assert not _joinOrKill(p, 15)
    assert p.exitcode == 0

    p = ctx.Process(target=_exitIfParentGoneChild, args=(os.getpid(),))
    p.start()
    assert not _joinOrKill(p, 15)
    assert p.exitcode == 3


def _guardedSleeper(pid_queue):
    """ Grandchild side: arm the parent death signal like the capture children, then sleep. """

    from RMS.Misc import setParentDeathSignal

    setParentDeathSignal()
    pid_queue.put(os.getpid())
    time.sleep(60)


def _middleParent(pid_queue):
    """ Child side: start a guarded grandchild and wait to be killed. """

    ctx = multiprocessing.get_context('fork')
    p = ctx.Process(target=_guardedSleeper, args=(pid_queue,))
    p.start()
    time.sleep(60)


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='PR_SET_PDEATHSIG is Linux-only')
def testParentDeathSignalKillsOrphan():
    """ A child that armed setParentDeathSignal dies when its parent is SIGKILLed. """

    import signal

    ctx = multiprocessing.get_context('fork')
    pid_queue = ctx.Queue()
    middle = ctx.Process(target=_middleParent, args=(pid_queue,))
    middle.start()
    grandchild = pid_queue.get(timeout=15)

    os.kill(middle.pid, signal.SIGKILL)
    middle.join(10)

    # The grandchild is not our child, so poll its existence
    deadline = time.monotonic() + 10
    alive = True
    while alive and (time.monotonic() < deadline):
        try:
            os.kill(grandchild, 0)
            time.sleep(0.1)
        except ProcessLookupError:
            alive = False

    if alive:
        os.kill(grandchild, signal.SIGKILL)

    assert not alive, 'orphaned child survived its parent'


@pytest.mark.parametrize('rel_path, class_name', [
    ('RMS/BufferedCapture.py', 'BufferedCapture'),
    ('RMS/Compression.py', 'Compressor'),
    ('RMS/UploadManager.py', 'UploadManager'),
    ('RMS/EventMonitor.py', 'EventMonitor'),
    ])
def testCaptureChildrenArmOrphanProtection(rel_path, class_name):
    """ The long-running children of StartCapture record the parent PID in __init__ and arm the
        orphan protection in run(). Checked on the source (AST) to avoid the modules' heavy imports.
    """

    import ast

    with open(os.path.join(RMS_ROOT, rel_path)) as f:
        tree = ast.parse(f.read())

    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and (n.name == class_name)][0]
    methods = dict((n.name, n) for n in cls.body if isinstance(n, ast.FunctionDef))

    def calledNames(fn):
        return set(n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name))

    init_src = ast.dump(methods['__init__'])
    assert "attr='parent_pid'" in init_src

    run_calls = calledNames(methods['run'])
    assert 'setParentDeathSignal' in run_calls
    assert 'exitIfParentGone' in run_calls


# ---------------------------------------------------------------------------
# Item 4: stopCapture must not join an unkillable capture process without a bound

class _UnkillableCapture(object):
    """ Stand-in for a BufferedCapture stuck in D-state: alive through terminate and SIGKILL. """

    def __init__(self):

        from RMS.Misc import AtomicFlag

        self.exit = AtomicFlag()
        self.pid = 999999999
        self.raw_frame_saver = None
        self.dropped_frames = multiprocessing.Value('i', 7, lock=False)
        self.join_timeouts = []

    def is_alive(self):
        return True

    def terminate(self):
        pass

    def join(self, timeout=None):
        self.join_timeouts.append(timeout)


def testStopCaptureAbandonsUnkillableProcess(monkeypatch):
    """ stopCapture must give up on a process that survives SIGKILL instead of joining forever. """

    import RMS.BufferedCapture as bcm

    # Skip the real waits and the real SIGKILL
    monkeypatch.setattr(bcm.time, 'sleep', lambda s: None)
    monkeypatch.setattr(bcm.os, 'kill', lambda pid, sig: None)

    fake = _UnkillableCapture()
    assert bcm.BufferedCapture.stopCapture(fake) == 7

    assert fake.join_timeouts, 'stopCapture never joined'
    assert all(t is not None for t in fake.join_timeouts), 'unbounded join: {}'.format(fake.join_timeouts)


# ---------------------------------------------------------------------------
# Item 5: captureDuration on the last night before polar day

# (lat, lon, first time, last time, step in minutes, continuous_capture). The windows contain the
# times where the next sunrise exists but next_setting() raised AlwaysUpError/NeverUpError
_POLAR_WINDOWS = [
    (72.0, 15.0, (2026, 4, 20, 22, 0), (2026, 4, 21, 0, 0), 2, None),
    (74.0, 15.0, (2026, 4, 15, 22, 30), (2026, 4, 15, 23, 30), 2, None),
    (76.0, 15.0, (2026, 4, 9, 22, 10), (2026, 4, 9, 23, 50), 2, None),
    (80.0, 15.0, (2026, 3, 30, 22, 30), (2026, 3, 30, 23, 30), 2, None),
    (80.0, 15.0, (2026, 3, 21, 22, 10), (2026, 3, 21, 23, 58), 2, True),
    (-77.85, 166.67, (2026, 10, 8, 11, 40), (2026, 10, 8, 13, 30), 2, None),
    (89.9, 0.0, (2026, 3, 5, 9, 0), (2026, 3, 5, 10, 0), 2, None),
    (89.9, 0.0, (2026, 10, 6, 23, 30), (2026, 10, 8, 6, 30), 10, None),
    ]


@pytest.mark.parametrize('lat, lon, t_beg, t_end, step, continuous', _POLAR_WINDOWS)
def testCaptureDurationLastNightBeforePolarDay(lat, lon, t_beg, t_end, step, continuous):
    """ captureDuration must not raise when the next sunset does not exist, and must return a sane
        start time and duration throughout the window.
    """

    import datetime
    from RMS.CaptureDuration import captureDuration

    t = datetime.datetime(*t_beg)
    t_end = datetime.datetime(*t_end)
    while t <= t_end:

        start_time, duration = captureDuration(lat, lon, 0, current_time=t, continuous_capture=continuous)

        assert 0 < duration <= 23*3600, (t, start_time, duration)
        if not isinstance(start_time, bool):
            assert start_time >= t - datetime.timedelta(minutes=1), (t, start_time)
        else:
            assert start_time is True

        t += datetime.timedelta(minutes=step)


# ---------------------------------------------------------------------------
# Item 6: reprocess_if_archive_missing must reprocess a night only once

def testArchiveMissingNightReprocessedOnce(tmp_path, monkeypatch):
    """ A processed night whose archive keeps disappearing (quota/retention) is reprocessed once,
        not on every start.
    """

    import RMS.StartCapture as sc
    import RMS.ConfigReader as cr
    from RMS.Reprocess import updateProcessingStatus

    config = cr.Config()
    config.data_dir = str(tmp_path)
    config.stationID = 'XX0001'
    config.reprocess_if_archive_missing = True
    config.auto_reprocess_max_attempts = 3
    config.continuous_capture = True
    config.external_script_run = False

    # A captured night which finished processing, and whose archive was deleted
    night = 'XX0001_20260101_000000_000000'
    captured_dir_path = os.path.join(str(tmp_path), config.captured_dir, night)
    os.makedirs(captured_dir_path)
    open(os.path.join(captured_dir_path, 'FF_XX0001_20260101_000000_000_0000000.fits'), 'w').close()
    updateProcessingStatus(config, captured_dir_path, archiving='completed')

    calls = []

    def fakeProcessNight(dir_path, cfg):

        # Recreate the archive and mark the night complete, like processNight
        calls.append(dir_path)
        updateProcessingStatus(cfg, dir_path, archiving='completed')
        return os.path.join(cfg.data_dir, cfg.archived_dir, night), None, None, None, None

    monkeypatch.setattr(sc, 'processNight', fakeProcessNight)

    # The module logger is only created in StartCapture's __main__ block
    monkeypatch.setattr(sc, 'log', logging.getLogger('rmslogger'), raising=False)

    # Several starts; the quota logic deletes the archive again each time (it is never created here)
    for _ in range(3):
        sc.processIncompleteCaptures(config, None)

    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Item 7: NTP clock offset estimate

class _FakeNTPSocket(object):
    """ UDP socket stand-in answering one NTP request with the given remote timestamps. """

    def __init__(self, t_remote_receive, t_remote_transmit, fail=None):
        self.t_remote_receive = t_remote_receive
        self.t_remote_transmit = t_remote_transmit
        self.fail = fail
        self.closed = False

    def settimeout(self, t):
        pass

    def sendto(self, data, addr):
        if self.fail is not None:
            raise self.fail

    def recvfrom(self, n):

        import struct

        def ntp(t):
            t = t + 2208988800
            return int(t), int(round((t - int(t))*2**32))

        words = [0]*12
        words[8], words[9] = ntp(self.t_remote_receive)
        words[10], words[11] = ntp(self.t_remote_transmit)
        return struct.pack('!12I', *words), ('127.0.0.1', 123)

    def close(self):
        self.closed = True


def _patchNTP(monkeypatch, osm, sock, local_times):
    """ Route timestampFromNTP's socket and clock through fakes. """

    import types
    import socket

    fake_socket_module = types.SimpleNamespace(socket=lambda *a: sock, AF_INET=socket.AF_INET,
        SOCK_DGRAM=socket.SOCK_DGRAM, timeout=socket.timeout)
    times = list(local_times)
    fake_time_module = types.SimpleNamespace(time=lambda: times.pop(0))

    monkeypatch.setattr(osm, 'socket', fake_socket_module)
    monkeypatch.setattr(osm, 'time', fake_time_module)


@pytest.mark.parametrize('delay, processing', [(0.0, 0.0), (0.1, 0.01)])
def testNTPOffsetLocalClockOneSecondAhead(monkeypatch, delay, processing):
    """ Local clock 1 s ahead with a symmetric network delay: the remote time estimate at the
        receive instant must be 1 s behind the local receive time (clock_ahead_ms = 1000).
    """

    import RMS.Formats.ObservationSummary as osm

    t_local_tx = 1.7e9
    t_local_rx = t_local_tx + delay + processing

    # Remote clock = local clock - 1 s, with the given processing time on the server
    t_remote_rx = t_local_tx + delay/2 - 1.0
    t_remote_tx = t_remote_rx + processing

    sock = _FakeNTPSocket(t_remote_rx, t_remote_tx)
    _patchNTP(monkeypatch, osm, sock, [t_local_tx, t_local_rx])

    remote_time, network_delay, addr = osm.timestampFromNTP('ntp.example')

    ahead_ms = (t_local_rx - remote_time)*1000
    assert abs(ahead_ms - 1000) < 1e-3
    assert abs(network_delay - delay) < 1e-6
    assert addr == 'ntp.example'
    assert sock.closed


def testNTPSocketFailureReturnsThreeValues(monkeypatch):
    """ A socket failure must return (None, None, addr), which is what the caller unpacks. """

    import socket
    import RMS.Formats.ObservationSummary as osm

    sock = _FakeNTPSocket(0, 0, fail=socket.timeout())
    _patchNTP(monkeypatch, osm, sock, [1.7e9, 1.7e9])

    assert osm.timestampFromNTP('ntp.example') == (None, None, 'ntp.example')
    assert sock.closed


# ---------------------------------------------------------------------------
# Item 8: a reprocess after finalize must not lose the start-of-session summary values

def testObservationSummaryReseededFromFinalJson(tmp_path):
    """ With the working JSON removed by finalize, the next working dict starts from the final
        summary instead of from an empty dict.
    """

    import RMS.ConfigReader as cr
    import RMS.Formats.ObservationSummary as osm
    from RMS.Misc import getRMSStyleFileName

    config = cr.Config()
    night_dir = str(tmp_path/'XX0001_20260101_000000_000000')
    os.makedirs(night_dir)

    # Start-of-session values recorded during capture
    d = osm.getObservationSummaryDict(night_dir)
    osm.addObsParam(d, 'start_time', '2026-01-01T00:00:00+00:00')
    osm.addObsParam(d, 'stationID', 'XX0001')
    osm.addObsParam(d, 'media_backend', 'gst')

    # Finalize writes the final JSON and removes the working one
    osm.writeToJSON(config, getRMSStyleFileName(night_dir, osm.OBSERVATION_SUMMARY_NAME_JSON), night_dir)
    os.unlink(getRMSStyleFileName(night_dir, osm.OBSERVATION_SUMMARY_WORKING_NAME_JSON))

    # Reprocessing loads the working dict again
    d = osm.getObservationSummaryDict(night_dir)

    assert d.get('start_time') == '2026-01-01T00:00:00+00:00'
    assert d.get('stationID') == 'XX0001'
    assert d.get('media_backend') == 'gst'
    assert d.get('night_data_dir') == night_dir

    # And it was persisted as the working JSON
    assert os.path.isfile(getRMSStyleFileName(night_dir, osm.OBSERVATION_SUMMARY_WORKING_NAME_JSON))


# ---------------------------------------------------------------------------
# Item 9: UploadManager.run must not sleep while holding its runtime locks

class _UploadLoopStub(object):
    """ Minimal stand-in for the UploadManager state used by run(). """

    def __init__(self):

        from RMS.Misc import AtomicFlag, BoundedLock

        self.exit = AtomicFlag()
        self.last_runtime_lock = BoundedLock('last')
        self.next_runtime_lock = BoundedLock('next')
        self.last_runtime = multiprocessing.Value('d', 0.0, lock=False)
        self.next_runtime = multiprocessing.Value('d', 0.0, lock=False)
        self.logging_queue = None
        self.config = None
        self.parent_pid = None
        self.uploads = 0

    def loadQueue(self):
        pass

    def uploadData(self):
        self.uploads += 1


@pytest.mark.parametrize('which', ['last', 'next'])
def testUploadManagerLoopReleasesLocksWhileWaiting(monkeypatch, which):
    """ While run() waits for the next upload, the parent must get the runtime locks at once. """

    import threading
    import RMS.UploadManager as um

    # Keep the process-wide side effects out of the test process
    monkeypatch.setattr(um, 'setParentDeathSignal', lambda: None)
    monkeypatch.setattr(um, 'initChildProcess', lambda *a, **k: None)
    monkeypatch.setattr(um, 'exitIfParentGone', lambda *a, **k: None)

    stub = _UploadLoopStub()

    # Make the loop wait: on the 15-minute interval after one upload, or on an upload delay
    if which == 'next':
        stub.next_runtime.value = time.time() + 3600

    loop = threading.Thread(target=um.UploadManager.run, args=(stub,))
    loop.daemon = True
    loop.start()

    try:
        time.sleep(0.5)
        lock = stub.last_runtime_lock if which == 'last' else stub.next_runtime_lock

        for _ in range(3):
            t_beg = time.monotonic()
            got = lock._lock.acquire(timeout=2.0)
            elapsed = time.monotonic() - t_beg
            if got:
                lock._lock.release()

            assert got and (elapsed < 0.5), 'lock held by the waiting loop ({:.2f} s)'.format(elapsed)
            time.sleep(0.3)

    finally:
        stub.exit.set()
        loop.join(5)

    assert stub.uploads == (1 if which == 'last' else 0)
