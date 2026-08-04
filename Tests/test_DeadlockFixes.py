""" Tests for the multiprocessing deadlock-cascade fixes (PR #935).

Covers the review-requested cases: the drop-on-full logging handler, SafeValue /
BoundedLock behavior with an orphaned lock, AtomicFlag semantics, the per-station
flock guard (including the SIGKILLed-holder and orphaned-child cases), the
QueuedPool graceful-shutdown paths, and torn-read tolerance of the lock-free
start-time doubles.

The station-lock tests exercise the REAL functions from RMS/StartCapture.py,
extracted via AST so the test does not pay (or depend on) the module's heavy
imports - the same reason the lock itself runs before them.
"""

import ast
import logging
import logging.handlers
import multiprocessing
import os
import signal
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

posix_only = pytest.mark.skipif(os.name != 'posix', reason='POSIX flock/fork semantics')


@pytest.fixture(autouse=True)
def _forceForkStartMethod():
    """ Stations run Linux where fork is the default; QueuedPool on this branch
        predates the spawn-pickling support, and the flock semantics under test
        are fork semantics. Force fork for this module (POSIX only). """
    if os.name == 'posix':
        try:
            multiprocessing.set_start_method('fork', force=True)
        except RuntimeError:
            pass
    yield


def _forkCtx():
    return multiprocessing.get_context('fork')


# ---------------------------------------------------------------------------
# AtomicFlag

def testAtomicFlagBasics():
    from RMS.Misc import AtomicFlag

    flag = AtomicFlag()
    assert not flag.is_set()
    flag.set()
    assert flag.is_set()
    flag.clear()
    assert not flag.is_set()


def testAtomicFlagWaitTimeout():
    from RMS.Misc import AtomicFlag

    flag = AtomicFlag()
    t_beg = time.monotonic()
    assert flag.wait(timeout=0.5, poll_interval=0.05) is False
    elapsed = time.monotonic() - t_beg
    assert 0.4 < elapsed < 2.0

    flag.set()
    assert flag.wait(timeout=0.5, poll_interval=0.05) is True


@posix_only
def testAtomicFlagCrossProcess():
    from RMS.Misc import AtomicFlag

    ctx = _forkCtx()
    flag = AtomicFlag()

    def setter(f):
        time.sleep(0.2)
        f.set()

    p = ctx.Process(target=setter, args=(flag,))
    p.start()
    assert flag.wait(timeout=5, poll_interval=0.05) is True
    p.join()


# ---------------------------------------------------------------------------
# Drop-on-full logging handler

def testDroppingHandlerNeverBlocksAndCounts():
    from RMS.Logger import _DroppingQueueHandler

    ctx = _forkCtx() if os.name == 'posix' else multiprocessing
    q = ctx.Queue(2)
    handler = _DroppingQueueHandler(q)

    record = logging.LogRecord('t', logging.INFO, __file__, 1, 'msg', None, None)

    t_beg = time.monotonic()
    for _ in range(50):
        handler.enqueue(record)
    elapsed = time.monotonic() - t_beg

    # Never blocked (a blocking put on the full queue would take >> this)
    assert elapsed < 2.0

    # Drops were counted, and the counter resets on read
    assert handler.dropped >= 40
    n = handler.takeDroppedCount()
    assert n >= 40
    assert handler.takeDroppedCount() == 0

    q.cancel_join_thread()
    q.close()


# ---------------------------------------------------------------------------
# SafeValue / BoundedLock with an orphaned lock

def testSafeValueOrphanedLockBounded(monkeypatch):
    import RMS.QueuedPool as QP

    monkeypatch.setattr(QP, 'SAFE_VALUE_LOCK_TIMEOUT', 0.5)

    sv = QP.SafeValue(0)

    # Simulate a process dying while holding the lock
    assert sv.lock.acquire(timeout=1)

    t_beg = time.monotonic()
    sv.increment()
    first = time.monotonic() - t_beg
    assert 0.4 < first < 3.0
    assert sv.value() >= 1

    # Memoized: subsequent operations are fast (short timeout, no full wait)
    t_beg = time.monotonic()
    sv.increment()
    assert (time.monotonic() - t_beg) < 1.5


def testBoundedLockOrphaned():
    from RMS.Misc import BoundedLock

    bl = BoundedLock('test', timeout=0.5)
    assert bl._lock.acquire(timeout=1)      # orphan it

    t_beg = time.monotonic()
    with bl:
        pass
    assert (time.monotonic() - t_beg) < 3.0

    # Memoized short path
    t_beg = time.monotonic()
    with bl:
        pass
    assert (time.monotonic() - t_beg) < 1.5


# ---------------------------------------------------------------------------
# Torn-read tolerant double

def testStableDoubleRead():
    from RMS.Misc import stableDoubleRead

    val = multiprocessing.Value('d', 0.0, lock=False)
    assert stableDoubleRead(val) == 0.0
    val.value = 1234567890.5
    assert stableDoubleRead(val) == 1234567890.5


# ---------------------------------------------------------------------------
# Station single-instance lock (real code, AST-extracted)

def _loadLockFunctions():
    src_path = os.path.join(os.path.dirname(__file__), os.pardir, 'RMS', 'StartCapture.py')
    tree = ast.parse(open(src_path).read())
    wanted = {'_closeStationLockInChild', '_takeStationLock'}
    mod = ast.Module(
        body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted],
        type_ignores=[])
    ns = {'os': os, 'sys': sys}
    exec(compile(mod, 'StartCapture_lock', 'exec'), ns)
    return ns['_takeStationLock']


def _lockPath(station):
    import tempfile
    return os.path.join(tempfile.gettempdir(),
        'rms_startcapture_{:s}.lock'.format(station))


@posix_only
def testStationLockRefusesDuplicate():
    import fcntl

    takeLock = _loadLockFunctions()
    station = 'TESTDUP1'
    lock_file = takeLock(station)
    try:
        # A second flock attempt from another fd must fail while we hold it
        f2 = open(_lockPath(station), 'a+')
        with pytest.raises((IOError, OSError)):
            fcntl.flock(f2, fcntl.LOCK_EX | fcntl.LOCK_NB)
        f2.close()
    finally:
        lock_file.close()
        os.remove(_lockPath(station))


@posix_only
def testStationLockReleasedOnSigkill(tmp_path):
    import fcntl

    ctx = _forkCtx()
    station = 'TESTKILL1'
    pid_file = str(tmp_path/'pid')

    def holder():
        takeLock = _loadLockFunctions()
        takeLock(station)
        open(pid_file, 'w').write(str(os.getpid()))
        time.sleep(30)

    p = ctx.Process(target=holder)
    p.start()
    for _ in range(100):
        if os.path.isfile(pid_file):
            break
        time.sleep(0.1)
    os.kill(int(open(pid_file).read()), signal.SIGKILL)
    p.join(5)
    time.sleep(0.2)

    f = open(_lockPath(station), 'a+')
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)   # must not raise
    f.close()
    os.remove(_lockPath(station))


@posix_only
def testStationLockSurvivesOrphanedChild(tmp_path):
    """ The review-blocking case: a child forked after lock acquisition must not
        keep the lock alive once the main process dies. Runs in a fresh
        interpreter: forking from the pytest process is unreliable on macOS
        once other tests have started threads. """

    import fcntl
    import subprocess

    station = 'TESTORPH1'
    pid_file = str(tmp_path/'pids')
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    script = (
        "import multiprocessing, os, sys, time, ast\n"
        "multiprocessing.set_start_method('fork', force=True)\n"
        "sys.path.insert(0, {root!r})\n"
        "src = open(os.path.join({root!r}, 'RMS', 'StartCapture.py')).read()\n"
        "tree = ast.parse(src)\n"
        "mod = ast.Module(body=[n for n in tree.body if getattr(n, 'name', '') in\n"
        "    ('_closeStationLockInChild', '_takeStationLock')], type_ignores=[])\n"
        "ns = {{'os': os, 'sys': sys}}\n"
        "exec(compile(mod, 'lock', 'exec'), ns)\n"
        "ns['_takeStationLock']({station!r})\n"
        "child = multiprocessing.Process(target=time.sleep, args=(30,))\n"
        "child.start()\n"
        "open({pid_file!r}, 'w').write('{{:d}} {{:d}}'.format(os.getpid(), child.pid))\n"
        "time.sleep(30)\n"
    ).format(root=repo_root, station=station, pid_file=pid_file)

    proc = subprocess.Popen([sys.executable, '-c', script])
    try:
        for _ in range(100):
            if os.path.isfile(pid_file) and open(pid_file).read().strip():
                break
            time.sleep(0.1)
        main_pid, child_pid = map(int, open(pid_file).read().split())
        os.kill(main_pid, signal.SIGKILL)
        proc.wait(5)
        time.sleep(0.3)

        # The orphan child is still alive...
        os.kill(child_pid, 0)

        try:
            # ...but a new instance must be able to take the lock
            f = open(_lockPath(station), 'a+')
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)   # must not raise
            f.close()
        finally:
            os.kill(child_pid, signal.SIGKILL)
            os.remove(_lockPath(station))
    finally:
        if proc.poll() is None:
            proc.kill()


# ---------------------------------------------------------------------------
# QueuedPool graceful shutdown

def _double(x):
    time.sleep(0.02)
    return x*2


@posix_only
def testQueuedPoolJobsAndClose():
    from RMS.QueuedPool import QueuedPool

    qp = QueuedPool(_double, cores=2, log=None, print_state=False)
    qp.startPool()
    for i in range(10):
        qp.addJob([i])
    time.sleep(0.3)

    t_beg = time.monotonic()
    qp.closePool()
    assert (time.monotonic() - t_beg) < 60

    assert sorted(qp.getResults()) == [2*i for i in range(10)]
    qp.shutdownManager()


@posix_only
def testQueuedPoolZeroJobClose():
    """ With no jobs and a start delay, active_workers can still be 0 at close
        time - the pill count must come from the real worker list so shutdown
        stays graceful (review finding). """

    from RMS.QueuedPool import QueuedPool

    qp = QueuedPool(_double, cores=2, log=None, delay_start=2, print_state=False)
    qp.startPool()

    t_beg = time.monotonic()
    qp.closePool()
    assert (time.monotonic() - t_beg) < 30
    qp.shutdownManager()


# ---------------------------------------------------------------------------
# Logging stall detection


class _FakeQueue(object):
    """ Serves a scripted sequence of qsize() readings, holding the last one. """

    def __init__(self, sizes):
        self.sizes = list(sizes)

    def qsize(self):
        return self.sizes.pop(0) if len(self.sizes) > 1 else self.sizes[0]


class _FakeCounter(object):
    """ Serves a scripted sequence of listener progress counts. """

    def __init__(self, counts):
        self.counts = list(counts)

    @property
    def value(self):
        return self.counts.pop(0) if len(self.counts) > 1 else self.counts[0]


class _AliveListener(object):
    def is_alive(self):
        return True


def _stallManager(backlogs, processed):
    """ A LoggingManager wired to scripted readings, with the restart stubbed out. """

    from RMS.Logger import LoggingManager

    mgr = LoggingManager()
    mgr.is_initialized = True
    mgr.listener_process = _AliveListener()
    mgr.logging_queue = _FakeQueue(backlogs)
    mgr._records_processed = _FakeCounter(processed)

    mgr.restarts = []
    mgr._restartLogging = lambda reason: mgr.restarts.append(reason)

    return mgr


def _sample(mgr):
    """ Take one strike-eligible sample (pretend enough time has passed since the last). """

    mgr._last_sample_time = None
    mgr.checkLoggingHealth()


def testLoggingStallIgnoresGrowingBacklogWhileListenerConsumes():
    """ A backlog that keeps GROWING is not a stall if the listener is still taking records
        off the queue - producers can legitimately outrun it during a burst, and restarting
        then discards everything queued. """

    mgr = _stallManager(backlogs=[1000, 5000, 12000, 20000, 25000],
                        processed=[500, 4000, 9000, 15000, 21000])

    for _ in range(5):
        _sample(mgr)

    assert mgr.restarts == []
    assert mgr._stall_strikes == 0


def testLoggingStallRestartsOnFrozenListener():
    """ A high backlog with a listener that consumes nothing is the real wedge. """

    mgr = _stallManager(backlogs=[5000], processed=[77])

    for _ in range(1 + 3):      # first sample only establishes the baseline
        _sample(mgr)

    assert len(mgr.restarts) == 1
    assert 'no listener progress' in mgr.restarts[0]


def testLoggingStallSamplesAreSpacedOut():
    """ Both the capture watchdog and the always-on health thread call the check, so
        back-to-back calls must not each count as a strike. """

    mgr = _stallManager(backlogs=[5000], processed=[77])

    _sample(mgr)                    # baseline
    for _ in range(10):             # no clock advance: all of these must be ignored
        mgr.checkLoggingHealth()

    assert mgr._stall_strikes == 0
    assert mgr.restarts == []

    _sample(mgr)
    assert mgr._stall_strikes == 1


class _DeadListener(object):
    def is_alive(self):
        return False


def testLoggingRestartBackoffPacesDeadListenerRestarts():
    """ A listener that dies instantly on every respawn (e.g. unwritable log dir) must be
        restarted on the backoff schedule, not on every check - each unpaced cycle leaks
        the abandoned queue's fds. Runs the real _restartLogging with only the spawn
        stubbed out. """

    from RMS.Logger import LoggingManager

    mgr = LoggingManager()
    mgr.is_initialized = True
    mgr.listener_process = _DeadListener()
    mgr.logging_queue = _FakeQueue([0])

    spawns = []
    mgr._spawnListener = lambda: spawns.append(1)

    # First check: dead listener, no backoff armed yet - must restart immediately
    assert mgr.checkLoggingHealth() is False
    assert len(spawns) == 1
    assert mgr._restart_count == 1
    assert mgr._next_restart_allowed > time.monotonic()

    # The spawn was stubbed so the listener is still dead - checks inside the backoff
    # window must report failure without restarting again
    for _ in range(5):
        assert mgr.checkLoggingHealth() is False
    assert len(spawns) == 1
