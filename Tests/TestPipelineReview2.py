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
    assert 'startParentWatch' in run_calls


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

    # A value only written when a step succeeds, which the reprocess must recompute
    osm.addObsParam(d, 'photometry_good', 'True')

    # Finalize writes the final JSON and removes the working one
    osm.writeToJSON(config, getRMSStyleFileName(night_dir, osm.OBSERVATION_SUMMARY_NAME_JSON), night_dir)
    os.unlink(getRMSStyleFileName(night_dir, osm.OBSERVATION_SUMMARY_WORKING_NAME_JSON))

    # Reprocessing loads the working dict again
    d = osm.getObservationSummaryDict(night_dir)

    assert d.get('start_time') == '2026-01-01T00:00:00+00:00'
    assert d.get('stationID') == 'XX0001'
    assert d.get('media_backend') == 'gst'
    assert d.get('night_data_dir') == night_dir

    # Only capture-time keys are carried over, so a stale result cannot survive the reprocess
    assert 'photometry_good' not in d
    assert set(d) <= set(osm.OBSERVATION_SUMMARY_SESSION_KEYS) | {'night_data_dir'}

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


# ---------------------------------------------------------------------------
# Item 10: the nightly detector pool gets the config (log filter in the workers)

@pytest.mark.parametrize('rel_path', ['RMS/StartCapture.py', 'RMS/DetectStarsAndMeteors.py'])
def testQueuedPoolCallsPassConfig(rel_path):
    """ Every QueuedPool created by the capture/detection entry points passes config=, so the
        workers attach the InRmsFilter to their queue handler.
    """

    import ast

    with open(os.path.join(RMS_ROOT, rel_path)) as f:
        tree = ast.parse(f.read())

    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and (n.func.id == 'QueuedPool')]

    assert calls
    for call in calls:
        assert 'config' in [kw.arg for kw in call.keywords], 'line {:d}'.format(call.lineno)


# ---------------------------------------------------------------------------
# Item 11: runWithTimeout late-completion race

def testRunWithTimeoutLateCompletionRace(monkeypatch):
    """ If the function completes right between the caller's completion check and the caller giving
        up, the late cleanup must still run (the caller reports a timeout).
    """

    import types
    import threading
    import RMS.Misc as misc

    gate = threading.Event()
    was_set = threading.Event()

    class _RacyEvent(threading.Event):
        """ Event whose first is_set() lets the function finish and then reports the stale state. """

        checks = 0

        def set(self):
            super(_RacyEvent, self).set()
            was_set.set()

        def is_set(self):
            _RacyEvent.checks += 1
            if _RacyEvent.checks == 1:

                # Let the function finish now, as if it completed during the check
                gate.set()
                was_set.wait(0.5)
                return False

            return super(_RacyEvent, self).is_set()

    # Swap the Event class only for Misc's own threading references
    fake_threading = types.SimpleNamespace(Event=_RacyEvent, Thread=threading.Thread, Lock=threading.Lock)
    monkeypatch.setattr(misc, 'threading', fake_threading)

    cleaned = threading.Event()

    success, _, _ = misc.runWithTimeout(lambda: gate.wait(5), timeout=0.05,
        on_late_completion=cleaned.set)

    assert success is False
    assert cleaned.wait(5), 'caller reported a timeout but the late cleanup never ran'


# ---------------------------------------------------------------------------
# Item 12: continuous mode keeps capturing until the reboot is issued

class _StoppableCapture(object):
    """ BufferedCapture stand-in counting stopCapture calls. """

    def __init__(self):
        self.stops = 0

    def stopCapture(self):
        self.stops += 1
        return 5


def testStopPendingCaptureStopsOnceAndRecordsDroppedFrames(tmp_path, monkeypatch):
    """ The capture handed over for the reboot is stopped exactly once, with its dropped frame count
        recorded, however many times stopPendingCapture is called.
    """

    import RMS.StartCapture as sc

    recorded = {}
    monkeypatch.setattr(sc, 'log', logging.getLogger('rmslogger'), raising=False)
    monkeypatch.setattr(sc, 'getObservationSummaryDict', lambda night_dir: {'night_data_dir': night_dir})
    monkeypatch.setattr(sc, 'addObsParam', lambda d, k, v: recorded.update({(d['night_data_dir'], k): v}))

    bc = _StoppableCapture()
    monkeypatch.setattr(sc, 'REBOOT_PENDING_CAPTURE', (bc, str(tmp_path)))

    sc.stopPendingCapture()
    sc.stopPendingCapture()

    assert bc.stops == 1
    assert sc.REBOOT_PENDING_CAPTURE is None
    assert recorded == {(str(tmp_path), 'dropped_frames'): 5}


def testMainLoopStopsPendingCaptureBeforeResuming():
    """ After tryRebootAfterProcessing returns (no reboot), the main loop stops the pending capture
        before runCapture can start a second one.
    """

    with open(os.path.join(RMS_ROOT, 'RMS', 'StartCapture.py')) as f:
        src = f.read()

    main_src = src[src.index('if __name__ == "__main__":'):]
    i_call = main_src.index('tryRebootAfterProcessing(')
    i_resume = main_src.index('Reboot did not happen, resuming capture')

    assert 'stopPendingCapture()' in main_src[i_call:i_resume]


class _Uploading(object):
    """ Upload manager stand-in whose upload_in_progress flag is controlled by the test. """

    def __init__(self):

        import threading

        self.upload_in_progress = threading.Event()


def _rebootHarness(monkeypatch, tmp_path, uploading_iterations, dusk_after=None):
    """ Drive tryRebootAfterProcessing with fakes for the capture, sleep and os.system.

    Arguments:
        uploading_iterations: [int] Number of 1-minute waits during which an upload is in progress.

    Keyword arguments:
        dusk_after: [int] Number of 1-minute waits after which the mode switcher turns to night, or
            None for never.

    Return:
        events: [list] Ordered log of 'stop', 'reboot' and 'wait' events.
    """

    import types
    import RMS.StartCapture as sc

    events = []

    class _Capture(object):
        def stopCapture(self):
            events.append('stop')
            return 0

    monkeypatch.setattr(sc, 'log', logging.getLogger('rmslogger'), raising=False)
    monkeypatch.setattr(sc, 'getObservationSummaryDict', lambda night_dir: {'night_data_dir': night_dir})
    monkeypatch.setattr(sc, 'addObsParam', lambda d, k, v: None)
    monkeypatch.setattr(sc, 'REBOOT_PENDING_CAPTURE', (_Capture(), str(tmp_path)))

    upload_manager = _Uploading()
    daytime_mode = multiprocessing.Value('b', True, lock=False)
    waits = [0]

    def fakeSleep(seconds):

        # Only the 1-minute retry waits count; the uploads and dusk follow them
        if seconds != 60:
            return

        waits[0] += 1
        events.append('wait')
        if waits[0] >= uploading_iterations:
            upload_manager.upload_in_progress.clear()
        if (dusk_after is not None) and (waits[0] >= dusk_after):
            daytime_mode.value = False

    def fakeSystem(cmd):

        # A failed reboot command: the helper gives up at once
        events.append('reboot')
        return 256

    monkeypatch.setattr(sc, 'time', types.SimpleNamespace(sleep=fakeSleep, time=time.time))
    monkeypatch.setattr(sc.os, 'system', fakeSystem)

    if uploading_iterations:
        upload_manager.upload_in_progress.set()

    config = types.SimpleNamespace(continuous_capture=True, data_dir=str(tmp_path),
        reboot_lock_file='.reboot_lock')
    sc.tryRebootAfterProcessing(config, upload_manager, True, daytime_mode=daytime_mode)

    return events


def testRebootKeepsCapturingUntilTheRebootIsIssued(tmp_path, monkeypatch):
    """ The capture keeps running through the upload wait and is stopped right before the reboot. """

    events = _rebootHarness(monkeypatch, tmp_path, uploading_iterations=3)

    assert events == ['wait', 'wait', 'wait', 'stop', 'reboot']


def testRebootWaitStopsCaptureAtDusk(tmp_path, monkeypatch):
    """ If night mode starts while the reboot waits, the daytime capture is stopped then, once. """

    events = _rebootHarness(monkeypatch, tmp_path, uploading_iterations=5, dusk_after=2)

    assert events == ['wait', 'wait', 'stop', 'wait', 'wait', 'wait', 'reboot']


def testRebootNeverIssuedLeavesCaptureToMainLoop(tmp_path, monkeypatch):
    """ If the retries run out in daylight, the helper leaves the capture running (the main loop
        stops it before resuming) and never issues the reboot.
    """

    import RMS.StartCapture as sc

    events = _rebootHarness(monkeypatch, tmp_path, uploading_iterations=10**6)

    assert events.count('wait') == 4*60
    assert ('stop' not in events) and ('reboot' not in events)
    assert sc.REBOOT_PENDING_CAPTURE is not None


# ---------------------------------------------------------------------------
# Item 13: Compressor.stop goes straight to terminate

class _StuckCompressor(object):
    """ Compressor stand-in whose run() never exits until it is terminated. """

    def __init__(self):

        from RMS.Misc import AtomicFlag

        self.exit = AtomicFlag()
        self.run_exited = AtomicFlag()
        self.run_exited.set()
        self.pid = 999999999
        self.detector = 'detector'
        self.terminated = False
        self.join_timeouts = []

    def is_alive(self):
        return not self.terminated

    def terminate(self):
        self.terminated = True

    def join(self, timeout=None):
        self.join_timeouts.append(timeout)


def testCompressorStopSkipsSigint(monkeypatch):
    """ The compressor ignores SIGINT, so stop() must not send it and wait 5 s for nothing. """

    import RMS.Compression as comp

    signals = []
    monkeypatch.setattr(comp.os, 'kill', lambda pid, sig: signals.append(sig))

    fake = _StuckCompressor()
    assert comp.Compressor.stop(fake) == 'detector'

    assert fake.terminated
    assert signals == []


# ---------------------------------------------------------------------------
# Item 14: a stale station lock file owned by another user

@posix_only
def testStationLockFallsBackToReadOnlyOnEacces():
    """ If the lock file cannot be opened for appending (owned by another user), the lock is taken
        through a read-only descriptor instead of refusing forever.
    """

    import ast
    import errno
    import fcntl
    import builtins
    import tempfile

    src_path = os.path.join(RMS_ROOT, 'RMS', 'StartCapture.py')
    with open(src_path) as f:
        tree = ast.parse(f.read())

    wanted = {'_closeStationLockInChild', '_takeStationLock'}
    mod = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted],
        type_ignores=[])

    # Simulate the other user's file: opening it for appending fails with EACCES
    def fakeOpen(path, mode='r', *args, **kwargs):
        if ('rms_startcapture_' in str(path)) and ('a' in mode):
            raise PermissionError(errno.EACCES, 'Permission denied', path)
        return builtins.open(path, mode, *args, **kwargs)

    ns = {'os': os, 'sys': sys, 'open': fakeOpen}
    exec(compile(mod, 'StartCapture_lock', 'exec'), ns)

    station = 'TESTEACCES1'
    lock_path = os.path.join(tempfile.gettempdir(), 'rms_startcapture_{:s}.lock'.format(station))
    with open(lock_path, 'w') as f:
        f.write('12345')

    try:
        lock_file = ns['_takeStationLock'](station)
        try:
            # The lock is really held
            with open(lock_path) as f2:
                with pytest.raises((IOError, OSError)):
                    fcntl.flock(f2, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            lock_file.close()

    finally:
        os.remove(lock_path)


# ---------------------------------------------------------------------------
# Item 15: empty config values keep the defaults

@pytest.mark.parametrize('option', ['udp_buffer_size', 'video_scale_width', 'video_scale_height',
    'auto_reprocess_max_attempts', 'ml_model_file'])
def testEmptyConfigValueKeepsDefault(tmp_path, option):
    """ An empty value ("option:" with nothing after it) must not make the config unparseable. """

    import re
    import RMS.ConfigReader as cr

    with open(os.path.join(RMS_ROOT, '.config')) as f:
        text = f.read()

    # Blank the option where it is set, or add it blank after the (commented) example
    pattern = re.compile(r'^(;\s*)?{:s}\s*[:=].*$'.format(re.escape(option)), re.MULTILINE)
    assert pattern.search(text), option
    text = pattern.sub('{:s}:'.format(option), text, count=1)

    config_path = str(tmp_path/'.config')
    with open(config_path, 'w') as f:
        f.write(text)

    config = cr.parse(config_path)
    default = cr.Config()

    assert getattr(config, option) == getattr(default, option)
    if option == 'ml_model_file':
        assert config.ml_model_path == default.ml_model_path


# ---------------------------------------------------------------------------
# Item 16: float FITS images in an image directory

def _floatFitsInput():
    """ An InputTypeImages stand-in carrying only the float conversion state. """

    from RMS.Formats.FrameInterface import InputTypeImages

    return InputTypeImages.__new__(InputTypeImages)


def testFloatFitsOffsetFixedForSequence():
    """ The same pixel value maps to the same level in every image of a sequence, whatever the
        minimum of each image.
    """

    import numpy as np
    from RMS.Formats.FrameInterface import InputTypeImages

    inp = _floatFitsInput()

    frame1 = np.array([[100.0, 1100.0], [600.0, 5000.0]])
    frame2 = np.array([[300.0, 1100.0], [600.0, 5000.0]])

    out1 = InputTypeImages._floatFitsToUint16(inp, frame1)
    out2 = InputTypeImages._floatFitsToUint16(inp, frame2)

    assert out1.dtype == np.uint16
    assert out1[0, 1] == out2[0, 1] == 1000
    assert out2[0, 0] == 200


def testFloatFitsNormalisedImageKeepsLevels():
    """ A normalised [0, 1] image is rescaled instead of collapsing to 0 and 1, and NaN pixels
        become 0.
    """

    import numpy as np
    from RMS.Formats.FrameInterface import InputTypeImages

    inp = _floatFitsInput()
    frame = np.array([[0.0, 0.25], [1.0, np.nan]], dtype=np.float32)

    out = InputTypeImages._floatFitsToUint16(inp, frame)

    assert out[0, 0] == 0
    assert abs(int(out[0, 1]) - 16384) <= 1
    assert abs(int(out[1, 0]) - 65535) <= 1
    assert out[1, 1] == 0
    assert len(np.unique(out)) == 3


def testFloatFitsWideRangeRescaled():
    """ A range beyond 65535 is compressed into uint16 instead of clipped. """

    import numpy as np
    from RMS.Formats.FrameInterface import InputTypeImages

    inp = _floatFitsInput()
    frame = np.array([[0.0, 65535.0], [131070.0, 200000.0]])

    out = InputTypeImages._floatFitsToUint16(inp, frame)

    assert out[1, 1] == 65535
    assert 0 < out[1, 0] < 65535
    assert out[0, 1] < out[1, 0]


# ---------------------------------------------------------------------------
# Item 17: 16-bit FF FITS files read with the default memmap=True

def testFFfits16BitReadWithMemmap(tmp_path):
    """ A 16-bit FF file (stored with BZERO) must be readable with the default memmap=True. """

    import numpy as np
    from RMS.Formats import FFfits
    from RMS.Formats.FFStruct import FFStruct

    ff = FFStruct()
    ff.nrows, ff.ncols, ff.nbits, ff.nframes = 4, 5, 16, 256
    ff.first, ff.camno, ff.fps = 0, 1, 25.0
    ff.starttime = '2026-01-01T00:00:00.000000'
    for name in ('maxpixel', 'maxframe', 'avepixel', 'stdpixel'):
        setattr(ff, name, (np.arange(20).reshape(4, 5)*3000).astype(np.uint16))

    file_name = 'FF_XX0001_20260101_000000_000_0000000.fits'
    FFfits.write(ff, str(tmp_path), file_name)

    ff_read = FFfits.read(str(tmp_path), file_name, memmap=True)

    assert ff_read.maxpixel.dtype == np.uint16
    assert np.array_equal(ff_read.maxpixel, ff.maxpixel)
    assert np.array_equal(ff_read.stdpixel, ff.stdpixel)


# ---------------------------------------------------------------------------
# Item 18: FRbin write/read round trip without leaking the file handle

def testFRbinWriteReadRoundTrip(tmp_path):
    """ write() must produce a file read() parses back, and read() must close its file. """

    import numpy as np
    from RMS.Formats import FRbin

    fr = FRbin.fr_struct()
    fr.lines = 1
    fr.frameNum = [2]
    fr.yc = [[10, 11]]
    fr.xc = [[20, 21]]
    fr.t = [[5, 6]]
    fr.size = [[2, 2]]
    fr.frames = [[np.array([[1, 2], [3, 4]], dtype=np.uint8), np.array([[5, 6], [7, 8]], dtype=np.uint8)]]

    file_name = 'FR_XX0001_20260101_000000_000_0000000.bin'
    FRbin.write(fr, str(tmp_path), file_name)

    def openFds():
        return len(os.listdir('/proc/self/fd')) if os.path.isdir('/proc/self/fd') else 0

    n_fds = openFds()
    fr_read = FRbin.read(str(tmp_path), file_name)
    assert openFds() == n_fds

    assert fr_read.lines == 1
    assert list(fr_read.frameNum) == [2]
    assert fr_read.yc == [[10, 11]]
    assert fr_read.xc == [[20, 21]]
    assert fr_read.t == [[5, 6]]
    assert np.array_equal(fr_read.frames[0][1], fr.frames[0][1])


# ---------------------------------------------------------------------------
# Item 19: FOV KML with empty polygon sides

@pytest.mark.parametrize('sides', [
    [],
    [[], [], [], []],
    [[[45.0, 15.0, 100000.0], [45.1, 15.1, 100000.0]], [], [[45.2, 15.0, 100000.0]], []],
    ])
@pytest.mark.parametrize('plot_station', [True, False])
def testFovKMLEmptyPolygon(tmp_path, monkeypatch, sides, plot_station):
    """ A fully or partly masked FOV (empty sides from fovArea) must still produce a KML file. """

    import types
    import Utils.FOVKML as fovkml

    monkeypatch.setattr(fovkml, 'fovArea', lambda *a, **k: [list(side) for side in sides])

    platepar = types.SimpleNamespace(station_code='XX0001', lat=45.0, lon=15.0, elev=100.0)
    kml_path = fovkml.fovKML(str(tmp_path), platepar, plot_station=plot_station)

    with open(kml_path) as f:
        kml = f.read()

    assert kml.rstrip().endswith('</kml>')


# ---------------------------------------------------------------------------
# Item 20: GRMSUpdater.sh kernel-mismatch reboot fallback

def _extractShellFunction(src, name):
    """ Return the text of a top-level bash function from a script. """

    lines = src.split('\n')
    start = lines.index('{:s}() {{'.format(name))
    end = lines.index('}', start)
    return '\n'.join(lines[start:end + 1])


@posix_only
@pytest.mark.parametrize('running, installed, expect_reboot, target', [

    # Raspberry Pi 5 with all Raspberry Pi OS flavours installed and nothing newer
    ('6.6.31+rpt-rpi-2712', ['6.6.31+rpt-rpi-2712', '6.6.31+rpt-rpi-v7', '6.6.31+rpt-rpi-v7l',
        '6.6.31+rpt-rpi-v8'], False, ''),

    # A newer kernel of the running flavour is installed
    ('6.6.31+rpt-rpi-2712', ['6.6.31+rpt-rpi-2712', '6.6.51+rpt-rpi-2712', '6.6.51+rpt-rpi-v8'], True,
        '6.6.51+rpt-rpi-2712'),

    # Older Raspberry Pi OS naming
    ('5.10.103-v7l+', ['5.10.103+', '5.10.103-v7+', '5.10.103-v7l+', '5.10.103-v8+'], False, ''),

    # Debian: the ABI number changes between versions of one flavour
    ('6.1.0-18-amd64', ['6.1.0-18-amd64', '6.1.0-21-amd64'], True, '6.1.0-21-amd64'),

    # Early Bookworm names carry an ABI revision (-rpiN) that changes with every kernel update
    ('6.1.0-rpi4-rpi-v8', ['6.1.0-rpi4-rpi-v8', '6.1.0-rpi7-rpi-v8', '6.1.0-rpi7-rpi-2712'], True,
        '6.1.0-rpi7-rpi-v8'),

    # ... and the later +rpt naming of the same flavour counts as an update too
    ('6.1.0-rpi7-rpi-v8', ['6.1.0-rpi7-rpi-v8', '6.6.31+rpt-rpi-v8', '6.6.31+rpt-rpi-2712'], True,
        '6.6.31+rpt-rpi-v8'),

    # Nothing newer of the running flavour among early Bookworm names
    ('6.1.0-rpi7-rpi-2712', ['6.1.0-rpi7-rpi-2712', '6.1.0-rpi7-rpi-v8', '6.1.0-rpi7-rpi-v7l'], False, ''),

    # Ubuntu
    ('5.15.0-91-generic', ['5.15.0-91-generic', '5.15.0-94-generic'], True, '5.15.0-94-generic'),
    ])
def testUpdaterKernelFallbackComparesRunningFlavour(tmp_path, running, installed, expect_reboot, target):
    """ should_reboot's kernel fallback only compares kernels of the running flavour. """

    import subprocess

    with open(os.path.join(RMS_ROOT, 'Scripts', 'MultiCamLinux', 'GRMSUpdater.sh')) as f:
        src = f.read()

    modules_dir = tmp_path/'modules'
    modules_dir.mkdir()
    for kver in installed:
        (modules_dir/kver).mkdir()

    # Keep the host's reboot flag file out of the test
    functions = _extractShellFunction(src, 'should_reboot').replace('/var/run/reboot-required',
        str(tmp_path/'no-reboot-required'))
    if 'kernel_flavour() {' in src:
        functions = _extractShellFunction(src, 'kernel_flavour') + '\n' + functions

    script = '\n'.join([
        'set -Eeuo pipefail',
        'log_message() { :; }',
        'uname() {{ echo {:s}; }}'.format(running),
        'ls() {{ command ls {:s}; }}'.format(str(modules_dir)),
        'REBOOT_MODE=if-needed',
        'REBOOT_STAMP_FILE={:s}'.format(str(tmp_path/'stamp')),
        functions,
        'if should_reboot; then echo "REBOOT $REBOOT_KERNEL_TARGET"; else echo "NOREBOOT"; fi',
        ])

    res = subprocess.run(['bash', '-c', script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=30)
    out = res.stdout.decode().strip()

    assert res.returncode == 0, out
    if expect_reboot:
        assert out == 'REBOOT {:s}'.format(target)
    else:
        assert out == 'NOREBOOT'


# ---------------------------------------------------------------------------
# Item 21: arithmetic increments under set -e in the update scripts

@posix_only
def testPostIncrementFromZeroAbortsUnderErrexit():
    """ The bash behaviour behind the fix: ((n++)) with n=0 returns status 1 and aborts a set -e
        script, while n=$((n + 1)) does not.
    """

    import subprocess

    bad = subprocess.run(['bash', '-c', 'set -Eeuo pipefail; n=0; ((n++)); echo reached'],
        stdout=subprocess.PIPE, timeout=30)
    good = subprocess.run(['bash', '-c', 'set -Eeuo pipefail; n=0; n=$((n + 1)); echo reached $n'],
        stdout=subprocess.PIPE, timeout=30)

    assert bad.returncode != 0 and b'reached' not in bad.stdout
    assert good.returncode == 0 and good.stdout.strip() == b'reached 1'


@pytest.mark.parametrize('rel_path', ['Scripts/RMS_Update.sh', 'Scripts/MultiCamLinux/GRMSUpdater.sh'])
def testUpdateScriptsHaveNoBarePostIncrement(rel_path):
    """ The errexit update scripts must not use bare ((var++)) / ((var--)) statements. """

    import re

    with open(os.path.join(RMS_ROOT, rel_path)) as f:
        src = f.read()

    assert 'set -Eeuo pipefail' in src
    bare = re.findall(r'^\s*\(\(\s*\w+\s*(?:\+\+|--)\s*\)\)\s*$', src, re.MULTILINE)
    assert bare == []


class _ExitingCompressorMixin(object):
    """ Mimics the end of Compressor.run(): run_exited is set, then the exit flush takes a while
        before os._exit(0).
    """

    def setUp(self):

        from RMS.Misc import AtomicFlag

        self.exit = AtomicFlag()
        self.run_exited = AtomicFlag()
        self.detector = None
        self.terminated = False

    def run(self):

        while not self.exit.is_set():
            time.sleep(0.01)

        self.run_exited.set()

        # flushChildLogging() on a busy queue
        time.sleep(0.3)
        os._exit(0)

    def terminate(self):

        self.terminated = True
        super(_ExitingCompressorMixin, self).terminate()


# One module-level class per start method, so the process pickles under forkserver
_EXITING_COMPRESSORS = {}
for _method in _startMethods():
    _cls_name = '_ExitingCompressor_' + _method
    globals()[_cls_name] = type(_cls_name, (_ExitingCompressorMixin,
        multiprocessing.get_context(_method).Process), {})
    _EXITING_COMPRESSORS[_method] = globals()[_cls_name]


@posix_only
@pytest.mark.parametrize('method', _startMethods())
def testCompressorStopWaitsForNormalExit(method):
    """ A compressor that set run_exited and is still flushing its logs must be allowed to exit on
        its own, not be terminated mid-flush.
    """

    import RMS.Compression as comp

    fake = _EXITING_COMPRESSORS[method]()
    fake.setUp()
    fake.start()
    time.sleep(0.3)

    comp.Compressor.stop(fake)
    _joinOrKill(fake, 10)

    assert not fake.terminated, 'a normally exiting compressor was terminated'
    assert fake.exitcode == 0


# ---------------------------------------------------------------------------
# Review 2, item 2: UploadManager.stop must not join an unkillable process without a bound

class _UnkillableUploader(object):
    """ Stand-in for an UploadManager that survives terminate and SIGKILL. """

    def __init__(self):

        from RMS.Misc import AtomicFlag

        self.exit = AtomicFlag()
        self.pid = 999999999
        self._mgr = None
        self.join_timeouts = []

    def is_alive(self):
        return True

    def terminate(self):
        pass

    def join(self, timeout=None):
        self.join_timeouts.append(timeout)

    def _shutdownManager(self):
        pass


def testUploadManagerStopAbandonsUnkillableProcess(monkeypatch):
    """ stop() escalates to SIGKILL and gives up instead of joining forever. """

    import RMS.UploadManager as um

    signals = []
    monkeypatch.setattr(um.os, 'kill', lambda pid, sig: signals.append(sig))

    fake = _UnkillableUploader()
    um.UploadManager.stop(fake, timeout=0.01)

    assert all(t is not None for t in fake.join_timeouts), fake.join_timeouts
    assert signals == [9]


def testStopPendingCaptureSkipsFinalizedSummary(tmp_path, monkeypatch):
    """ Once the night was finalized, stopping the pending capture must not recreate a working
        observation summary for the dropped frame count.
    """

    import RMS.StartCapture as sc
    import RMS.Formats.ObservationSummary as osm
    from RMS.Misc import getRMSStyleFileName

    night_dir = str(tmp_path/'XX0001_20260101_000000_000000')
    os.makedirs(night_dir)
    with open(getRMSStyleFileName(night_dir, osm.OBSERVATION_SUMMARY_NAME_JSON), 'w') as f:
        f.write('{}')

    monkeypatch.setattr(sc, 'log', logging.getLogger('rmslogger'), raising=False)
    bc = _StoppableCapture()
    monkeypatch.setattr(sc, 'REBOOT_PENDING_CAPTURE', (bc, night_dir))

    sc.stopPendingCapture()

    assert bc.stops == 1
    assert not os.path.exists(getRMSStyleFileName(night_dir, osm.OBSERVATION_SUMMARY_WORKING_NAME_JSON))


# ---------------------------------------------------------------------------
# Review 2, item 6: a child exiting normally delivers all its records through the exit hook

def _childLogsAndReturns(q, n):
    """ Child side: attach child logging, log n records and return normally. """

    from RMS.Logger import initChildLogging

    initChildLogging(q, None)
    for i in range(n):
        logging.getLogger('rmslogger').warning('record %d %s', i, 'y'*100)


@posix_only
@pytest.mark.parametrize('method', _startMethods())
@pytest.mark.parametrize('n', [10, 5000])
def testChildNormalExitDeliversAllRecords(method, n):
    """ The bounded exit flush must not cut off records a live listener is still reading. """

    import queue

    ctx = multiprocessing.get_context(method)
    q = ctx.Queue(100000)

    p = ctx.Process(target=_childLogsAndReturns, args=(q, n))
    p.start()

    # Read like the listener until everything arrived or the child is gone and the queue is empty
    got = 0
    deadline = time.monotonic() + 30
    while (got < n) and (time.monotonic() < deadline):
        try:
            q.get(timeout=1.0)
            got += 1
        except queue.Empty:
            if not p.is_alive():
                break

    killed = _joinOrKill(p, 10)

    assert not killed
    assert p.exitcode == 0
    assert got == n


# ---------------------------------------------------------------------------
# Codex review, item 1: orphan protection under forkserver

_FORKSERVER_ORPHAN_SCRIPT = """
import os, sys, time, multiprocessing
sys.path.insert(0, {root!r})
from RMS.Misc import AtomicFlag, setParentDeathSignal, exitIfParentGone, startParentWatch


class Child(multiprocessing.Process):
    # Stand-in for a capture child: the same orphan protection as BufferedCapture.run

    def __init__(self, pid_path, done_path):
        super(Child, self).__init__()
        self.exit = AtomicFlag()
        self.parent_pid = os.getpid()
        self.pid_path = pid_path
        self.done_path = done_path

    def run(self):
        setParentDeathSignal()
        exitIfParentGone(self.parent_pid, 'Child')
        startParentWatch(self.parent_pid, 'Child', self.exit, grace=10.0, interval=0.2)

        with open(self.pid_path, 'w') as f:
            f.write(str(os.getpid()))

        # The main loop, leaving through the normal exit path when the exit flag is set
        while not self.exit.is_set():
            time.sleep(0.05)

        with open(self.done_path, 'w') as f:
            f.write('clean')


if __name__ == '__main__':
    multiprocessing.set_start_method({method!r}, force=True)
    Child(sys.argv[1], sys.argv[2]).start()
    time.sleep(60)
"""


def _pidAlive(pid):
    """ True if the PID exists and is not a zombie. """

    try:
        with open('/proc/{:d}/stat'.format(pid)) as f:
            return f.read().split(')')[-1].split()[0] != 'Z'
    except (IOError, OSError):
        return False


@pytest.mark.skipif(not os.path.isdir('/proc'), reason='needs /proc')
@pytest.mark.parametrize('method', _startMethods())
def testOrphanStopsAfterMainSigkill(tmp_path, method):
    """ A capture child must stop within seconds of StartCapture being SIGKILLed, also under
        forkserver where the parent death signal never fires, and the fork server must not linger.
    """

    import signal
    import subprocess

    script = tmp_path/'orphan.py'
    script.write_text(_FORKSERVER_ORPHAN_SCRIPT.format(root=RMS_ROOT, method=method))
    pid_path = str(tmp_path/'child.pid')
    done_path = str(tmp_path/'child.done')

    main = subprocess.Popen([sys.executable, str(script), pid_path, done_path])
    child_pid = None
    try:
        # Wait for the child to be up
        deadline = time.monotonic() + 30
        while (not os.path.isfile(pid_path)) and (time.monotonic() < deadline):
            time.sleep(0.05)
        time.sleep(0.2)
        child_pid = int(open(pid_path).read())

        with open('/proc/{:d}/stat'.format(child_pid)) as f:
            os_parent = int(f.read().split(')')[-1].split()[1])

        # Kill the main process the hard way
        main.kill()
        main.wait(10)

        # The orphan must stop through its normal exit path, and the fork server with it
        deadline = time.monotonic() + 8
        while (_pidAlive(child_pid) or ((os_parent != main.pid) and _pidAlive(os_parent))) \
                and (time.monotonic() < deadline):
            time.sleep(0.1)

        assert not _pidAlive(child_pid), 'orphaned child survived the main process'

        # Under fork the death signal (SIGKILL) takes the child down at once. Under forkserver the
        #   watcher stops it through its normal exit path, and the fork server then exits too
        if os_parent != main.pid:
            assert os.path.isfile(done_path), 'orphan did not leave through its normal exit path'
            assert not _pidAlive(os_parent), 'fork server lingered after the orphan exited'

    finally:
        if main.poll() is None:
            main.kill()
        if (child_pid is not None) and _pidAlive(child_pid):
            os.kill(child_pid, signal.SIGKILL)


def _ignoresExitFlag(dead_pid):
    """ Child side: watch a parent that is already gone, but never honour the exit flag. """

    from RMS.Misc import AtomicFlag, startParentWatch

    startParentWatch(dead_pid, 'test', AtomicFlag(), grace=0.5, interval=0.1)
    time.sleep(30)
    os._exit(3)


@posix_only
@pytest.mark.parametrize('method', _startMethods())
def testParentWatchExitsHardAfterGrace(method):
    """ An orphan that does not leave through its normal exit path is exited after the grace period. """

    ctx = multiprocessing.get_context(method)

    # A PID that certainly belonged to a process that is gone now
    dead = ctx.Process(target=_shortLived)
    dead.start()
    dead.join(10)

    p = ctx.Process(target=_ignoresExitFlag, args=(dead.pid,))
    t_beg = time.monotonic()
    p.start()
    killed = _joinOrKill(p, 10)

    assert not killed
    assert p.exitcode == 0
    assert time.monotonic() - t_beg < 8


@pytest.mark.parametrize('first', ['constant', 'nan'])
def testFloatFitsMappingWaitsForAFrameWithLevels(first):
    """ A constant or all-NaN first image must not fix the mapping: a later normalised [0, 1] image
        still maps to the 16-bit range.
    """

    import numpy as np
    from RMS.Formats.FrameInterface import InputTypeImages

    inp = _floatFitsInput()

    if first == 'constant':
        frame1 = np.full((2, 2), 0.5)
    else:
        frame1 = np.full((2, 2), np.nan)

    out1 = InputTypeImages._floatFitsToUint16(inp, frame1)
    assert np.all(out1 == 0)

    frame2 = np.array([[0.0, 0.25], [0.5, 1.0]])
    out2 = InputTypeImages._floatFitsToUint16(inp, frame2)

    assert out2[1, 1] == 65535
    assert abs(int(out2[0, 1]) - 16384) <= 1
    assert len(np.unique(out2)) == 4
