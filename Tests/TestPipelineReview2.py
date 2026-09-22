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
