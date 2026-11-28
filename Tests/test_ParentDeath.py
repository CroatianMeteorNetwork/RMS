""" Tests of the child processes of StartCapture (capture, compression, upload manager, event monitor) stopping
    when StartCapture is gone.

    When StartCapture dies without stopping its children (killed by the OOM killer, a crash, SIGTERM or SIGKILL),
    the children must not keep running: the station lock is released with the dead process, and a restarted
    StartCapture would run a second capture on the same camera and a second upload manager on the same upload
    queue file next to them. Under the 'fork' start method the parent death signal stops them. Under
    'forkserver' (the default on Linux from Python 3.14) the OS parent of a child is the fork server, which keeps
    running while any of its children runs, so the death signal never comes and the children also watch the
    PID of their logical parent.

    The child targets are module-level functions so they run under both the 'fork' and the 'forkserver' start
    methods, and every wait is bounded.
"""

from __future__ import print_function, division, absolute_import

import os
import sys
import ast
import time
import signal
import subprocess
import multiprocessing

import pytest


RMS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
REPO_CONFIG = os.path.join(RMS_ROOT, '.config')

sys.path.insert(0, RMS_ROOT)

posix_only = pytest.mark.skipif(os.name != 'posix', reason='POSIX process semantics')
linux_only = pytest.mark.skipif(not sys.platform.startswith('linux'), reason='the death signal is Linux only')
proc_only = pytest.mark.skipif(not os.path.isdir('/proc'), reason='needs /proc')


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

    # Wait for the process, and kill it if it did not end in time
    p.join(timeout)
    alive = p.is_alive()
    if alive:
        p.kill()
        p.join(5)

    return alive


def _shortLived():
    """ A process which ends at once. """


def _deadPid(ctx):
    """ Return the PID of a process which certainly ended.

    Arguments:
        ctx: [context] Multiprocessing context.

    Return:
        [int] PID of the ended process.
    """

    # Run a process to its end and reap it
    dead = ctx.Process(target=_shortLived)
    dead.start()
    dead.join(10)

    return dead.pid


def _pidAlive(pid):
    """ Tell whether a process exists and is not a zombie, from /proc.

    Arguments:
        pid: [int] Process ID.

    Return:
        [bool] True if the process runs.
    """

    # The state follows the command name in brackets, which may contain spaces
    try:
        with open('/proc/{:d}/stat'.format(pid)) as f:
            return f.read().split(')')[-1].split()[0] != 'Z'
    except (IOError, OSError):
        return False


def _osParent(pid):
    """ Return the PID of the OS parent of a process, from /proc.

    Arguments:
        pid: [int] Process ID.

    Return:
        [int] PID of the parent.
    """

    # The parent PID follows the state
    with open('/proc/{:d}/stat'.format(pid)) as f:
        return int(f.read().split(')')[-1].split()[1])


def _waitForFile(path, timeout):
    """ Wait until a file exists and return its content.

    Arguments:
        path: [str] Path of the file.
        timeout: [float] Seconds to wait.

    Return:
        [str] Content of the file, None if it did not appear in time.
    """

    # Poll for the file, then give its writer a moment to finish
    deadline = time.monotonic() + timeout
    while (not os.path.isfile(path)) and (time.monotonic() < deadline):
        time.sleep(0.05)

    if not os.path.isfile(path):
        return None

    time.sleep(0.2)

    with open(path) as f:
        return f.read()


def _killMainAndWatch(main, child_pid, os_parent, timeout):
    """ SIGKILL the stand-in of StartCapture and wait until its child (and a fork server) are gone.

    Arguments:
        main: [Popen] The stand-in of StartCapture.
        child_pid: [int] PID of its child.
        os_parent: [int] PID of the OS parent of the child (the main process or the fork server).
        timeout: [float] Seconds to wait.
    """

    # Kill the main process the hard way, like the OOM killer
    main.kill()
    main.wait(10)

    # Wait for the child, and for the fork server if the child came from one
    deadline = time.monotonic() + timeout
    while (_pidAlive(child_pid) or ((os_parent != main.pid) and _pidAlive(os_parent))) \
            and (time.monotonic() < deadline):
        time.sleep(0.1)


def _cleanUp(main, pids):
    """ Kill what a test started, if it is still running.

    Arguments:
        main: [Popen] The stand-in of StartCapture.
        pids: [list] PIDs of its children, None entries are skipped.
    """

    # The main process first, then its children which outlived it
    if main.poll() is None:
        main.kill()
        main.wait(10)

    for pid in pids:
        if (pid is not None) and _pidAlive(pid):
            os.kill(pid, signal.SIGKILL)


# ---------------------------------------------------------------------------
# The parent death signal

def _guardedSleeper(pid_queue):
    """ Grandchild side: arm the parent death signal like the capture children, report it, then sleep.

    Arguments:
        pid_queue: [Queue] Gets the PID of this process and whether the death signal was armed.
    """

    from RMS.Misc import setParentDeathSignal

    # Arm the death signal and report it
    armed = setParentDeathSignal()
    pid_queue.put((os.getpid(), armed))
    time.sleep(60)


def _middleParent(pid_queue):
    """ Child side: start a guarded grandchild and wait to be killed.

    Arguments:
        pid_queue: [Queue] Handed to the grandchild.
    """

    # The grandchild is forked from the main thread, like the capture children of StartCapture
    ctx = multiprocessing.get_context('fork')
    p = ctx.Process(target=_guardedSleeper, args=(pid_queue,))
    p.start()
    time.sleep(60)


@linux_only
def testParentDeathSignalKillsOrphan():
    """ setParentDeathSignal reports that it armed the death signal, and a child which armed it dies when its
        parent is SIGKILLed (the dying part is a guard of behaviour that must not change).
    """

    # A child whose own child armed the death signal
    ctx = multiprocessing.get_context('fork')
    pid_queue = ctx.Queue()
    middle = ctx.Process(target=_middleParent, args=(pid_queue,))
    middle.start()
    grandchild, armed = pid_queue.get(timeout=15)

    # Kill the middle process the hard way
    os.kill(middle.pid, signal.SIGKILL)
    middle.join(10)

    # The grandchild is not a child of this process, so poll its existence
    deadline = time.monotonic() + 10
    while _pidAlive(grandchild) and (time.monotonic() < deadline):
        time.sleep(0.1)

    # Kill a survivor
    alive = _pidAlive(grandchild)
    if alive:
        os.kill(grandchild, signal.SIGKILL)

    assert armed is True
    assert not alive, 'orphaned child survived its parent'


# ---------------------------------------------------------------------------
# The check of the logical parent at the start of run()

def _exitIfParentGoneChild(parent_pid):
    """ Child side: run the check against the given PID, and exit with 3 if it returns.

    Arguments:
        parent_pid: [int] PID to check.
    """

    from RMS.Misc import exitIfParentGone

    # The check exits the process with 0 if the parent is gone
    exitIfParentGone(parent_pid, 'test')
    os._exit(3)


@posix_only
@pytest.mark.parametrize('method', _startMethods())
def testExitIfParentGone(method):
    """ exitIfParentGone exits the child when the parent PID is gone, and returns while the parent runs, also
        when the parent belongs to another user (PID 1, EPERM for a normal user).
    """

    ctx = multiprocessing.get_context(method)

    # Each PID with the exit code the child must end with: 0 if it exited in the check, 3 if the check returned
    for parent_pid, exitcode in [(_deadPid(ctx), 0), (os.getpid(), 3), (1, 3)]:
        p = ctx.Process(target=_exitIfParentGoneChild, args=(parent_pid,))
        p.start()

        assert not _joinOrKill(p, 15)
        assert p.exitcode == exitcode, 'parent PID {:d}'.format(parent_pid)


# ---------------------------------------------------------------------------
# The watch of the logical parent

def _parentWatchStarted(parent_pid, arm):
    """ Child side: arm the death signal or not, and exit with 10 if a parent watch was started, 11 if not.

    Arguments:
        parent_pid: [int] PID of the logical parent.
        arm: [bool] True to arm the death signal first.
    """

    from RMS.Misc import AtomicFlag, setParentDeathSignal, startParentWatch

    # Arm the death signal or not, and report whether the watch started
    death_signal = setParentDeathSignal() if arm else False
    watcher = startParentWatch(parent_pid, 'test', AtomicFlag(), death_signal=death_signal)

    os._exit(10 if watcher is not None else 11)


@linux_only
@pytest.mark.parametrize('arm', [True, False])
@pytest.mark.parametrize('method', _startMethods())
def testParentWatchOnlyWhereDeathSignalMissesParent(method, arm):
    """ The parent watch thread is started under 'forkserver', where the OS parent is the fork server, and when
        the death signal is not armed, but not under 'fork' with the death signal armed (the signal covers it).
    """

    ctx = multiprocessing.get_context(method)

    # Run the child, which reports through its exit code
    p = ctx.Process(target=_parentWatchStarted, args=(os.getpid(), arm))
    p.start()

    assert not _joinOrKill(p, 15)
    assert p.exitcode in (10, 11)

    # The watch starts unless the death signal sees the parent
    started = (p.exitcode == 10)
    assert started == ((method != 'fork') or (not arm))


def _ignoresExitFlag(dead_pid):
    """ Child side: watch a parent which is already gone, but never leave through the normal exit path.

    Arguments:
        dead_pid: [int] PID of a process which ended.
    """

    from RMS.Misc import AtomicFlag, startParentWatch

    # Watch the parent which is gone, with a short grace period
    exit_flag = AtomicFlag()
    startParentWatch(dead_pid, 'test', exit_flag, grace=0.5, interval=0.1)

    # Only the parent watch may end this process, with the exit flag set first
    time.sleep(30)
    os._exit(3)


@posix_only
@pytest.mark.parametrize('method', _startMethods())
def testParentWatchExitsAfterGrace(method):
    """ An orphan which does not leave through its normal exit path is exited by the parent watch after the grace
        period.
    """

    ctx = multiprocessing.get_context(method)

    # Run the orphan and time its exit
    p = ctx.Process(target=_ignoresExitFlag, args=(_deadPid(ctx),))
    t_beg = time.monotonic()
    p.start()
    killed = _joinOrKill(p, 10)

    assert not killed
    assert p.exitcode == 0
    assert (time.monotonic() - t_beg) < 8


# ---------------------------------------------------------------------------
# StartCapture SIGKILLed with a running child

_STAND_IN_SCRIPT = """
import os, sys, time, multiprocessing
sys.path.insert(0, {root!r})
from RMS.Misc import AtomicFlag, setParentDeathSignal, exitIfParentGone, startParentWatch


class Child(multiprocessing.Process):
    # Stand-in for a capture child, with the same protection as BufferedCapture.run()

    def __init__(self, pid_path, done_path):
        super(Child, self).__init__()
        self.exit = AtomicFlag()
        self.parent_pid = os.getpid()
        self.pid_path = pid_path
        self.done_path = done_path

    def run(self):

        # Stop when the parent is gone
        death_signal = setParentDeathSignal()
        exitIfParentGone(self.parent_pid, 'Child')
        startParentWatch(self.parent_pid, 'Child', self.exit, death_signal=death_signal, grace=10.0,
            interval=0.2)

        # Tell the test that the child runs
        with open(self.pid_path, 'w') as f:
            f.write(str(os.getpid()))

        # The main loop, which ends when the exit flag is set
        while not self.exit.is_set():
            time.sleep(0.05)

        # Mark the normal exit path
        with open(self.done_path, 'w') as f:
            f.write('normal exit')


if __name__ == '__main__':

    # Start the child with the start method of the test, and wait to be killed
    multiprocessing.set_start_method({method!r}, force=True)
    Child(sys.argv[1], sys.argv[2]).start()
    time.sleep(60)
"""


@proc_only
@pytest.mark.parametrize('method', _startMethods())
def testOrphanStopsAfterMainSigkill(tmp_path, method):
    """ A child must stop within seconds of its parent being SIGKILLed: under 'fork' through the death signal,
        under 'forkserver' (where the death signal does not come) through its normal exit path, and the fork
        server must not stay behind.
    """

    # Run the stand-in of StartCapture
    script = tmp_path/'stand_in.py'
    script.write_text(_STAND_IN_SCRIPT.format(root=RMS_ROOT, method=method))
    pid_path = str(tmp_path/'child.pid')
    done_path = str(tmp_path/'child.done')

    main = subprocess.Popen([sys.executable, str(script), pid_path, done_path])
    child_pid = None
    os_parent = None
    try:

        # Wait for the child to run
        content = _waitForFile(pid_path, 30)
        assert content is not None, 'the child did not start'
        child_pid = int(content)
        os_parent = _osParent(child_pid)

        # Kill StartCapture and give the child a few seconds
        _killMainAndWatch(main, child_pid, os_parent, 8)

        assert not _pidAlive(child_pid), 'orphaned child survived the main process'

        # Under 'forkserver' the child left through its normal exit path, and the fork server ended with it
        if os_parent != main.pid:
            assert os.path.isfile(done_path), 'the orphan did not leave through its normal exit path'
            assert not _pidAlive(os_parent), 'the fork server stayed after the orphan exited'

    finally:
        _cleanUp(main, [child_pid, os_parent if (os_parent != main.pid) else None])


_COMPRESSOR_SCRIPT = """
import os, sys, time, ctypes, multiprocessing
sys.path.insert(0, {root!r})


if __name__ == '__main__':
    multiprocessing.set_start_method({method!r}, force=True)

    import numpy as np
    import RMS.ConfigReader as cr
    from RMS.Compression import Compressor
    from RMS.Misc import frameBufferShape

    # A config with a small frame size, so the compressor starts in a moment
    config = cr.parse({config!r})
    config.width = 64
    config.height = 48
    config.data_dir = sys.argv[2]
    config.live_jpg = False
    config.enable_fireball_detection = False

    # Shared frame buffers like the capture's, with no block waiting
    buffer_len = int(np.prod(frameBufferShape(config)))
    array1 = multiprocessing.Array(ctypes.c_uint8, buffer_len)
    array2 = multiprocessing.Array(ctypes.c_uint8, buffer_len)
    start_time1 = multiprocessing.Value('d', 0.0, lock=False)
    start_time2 = multiprocessing.Value('d', 0.0, lock=False)

    # Start the compressor like StartCapture does, and wait to be killed
    compressor = Compressor(sys.argv[2], array1, start_time1, array2, start_time2, config)
    compressor.start()

    # Tell the test that the compressor runs
    with open(sys.argv[1], 'w') as f:
        f.write(str(compressor.pid))

    time.sleep(60)
"""


@proc_only
@pytest.mark.parametrize('method', _startMethods())
def testCompressorStopsAfterMainSigkill(tmp_path, method):
    """ The compressor of a SIGKILLed StartCapture must not keep running, under 'fork' and under 'forkserver'.
    """

    # Run the stand-in of StartCapture with a real compressor
    script = tmp_path/'start_capture.py'
    script.write_text(_COMPRESSOR_SCRIPT.format(root=RMS_ROOT, method=method, config=REPO_CONFIG))
    pid_path = str(tmp_path/'compressor.pid')
    night_dir = tmp_path/'night'
    night_dir.mkdir()

    main = subprocess.Popen([sys.executable, str(script), pid_path, str(night_dir)])
    child_pid = None
    os_parent = None
    try:

        # Wait for the compressor to run
        content = _waitForFile(pid_path, 60)
        assert content is not None, 'the compressor did not start'
        child_pid = int(content)
        time.sleep(1.0)
        os_parent = _osParent(child_pid)

        # Kill StartCapture and give the compressor a few seconds
        _killMainAndWatch(main, child_pid, os_parent, 10)

        assert not _pidAlive(child_pid), 'the compressor kept running after StartCapture was killed'
        if os_parent != main.pid:
            assert not _pidAlive(os_parent), 'the fork server stayed after the compressor exited'

    finally:
        _cleanUp(main, [child_pid, os_parent if (os_parent != main.pid) else None])


# ---------------------------------------------------------------------------
# Every long-running child of StartCapture is protected

@pytest.mark.parametrize('rel_path, class_name', [
    ('RMS/BufferedCapture.py', 'BufferedCapture'),
    ('RMS/Compression.py', 'Compressor'),
    ('RMS/UploadManager.py', 'UploadManager'),
    ('RMS/EventMonitor.py', 'EventMonitor'),
    ])
def testCaptureChildrenStopWithParent(rel_path, class_name):
    """ The long-running children of StartCapture record the parent PID in __init__, and in run() arm the death
        signal, check the parent and watch it. Checked on the source (AST), without the heavy imports.
    """

    # Parse the source of the module
    with open(os.path.join(RMS_ROOT, rel_path)) as f:
        tree = ast.parse(f.read())

    # The methods of the class
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and (n.name == class_name)][0]
    methods = dict((n.name, n) for n in cls.body if isinstance(n, ast.FunctionDef))

    def calledNames(fn):
        """ Return the names of the plain functions called in a function.

        Arguments:
            fn: [FunctionDef] Function node.

        Return:
            [set] Names of the called functions.
        """

        return set(n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name))

    # The parent PID is recorded in the parent
    assert "attr='parent_pid'" in ast.dump(methods['__init__'])

    # The child arms the protection
    run_calls = calledNames(methods['run'])
    assert 'setParentDeathSignal' in run_calls
    assert 'exitIfParentGone' in run_calls
    assert 'startParentWatch' in run_calls
