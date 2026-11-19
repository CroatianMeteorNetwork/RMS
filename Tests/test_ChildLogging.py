""" Tests of the logging in child processes (capture, compression, extraction, upload, ...), which forward their
    records through a bounded multiprocessing queue to the listener process.

    A child logging into a full queue must drop the records and count them instead of blocking or recursing. At
    its exit, a child must wait for its buffered records to be read while the listener may still read them (a
    child which exits in the middle of a write leaves the queue's shared write lock taken for good), and exit at
    once when the queue was abandoned (the logging watchdog replaced it, or the listener was stopped at shutdown).
    The records a child logs right before it exits, through os._exit() or normally, must reach the listener.

    The child targets are module-level functions so they run under both the 'fork' and the 'forkserver' start
    methods (the default on Linux from Python 3.14), and every wait is bounded.
"""

from __future__ import print_function, division, absolute_import

import os
import sys
import time
import queue
import ctypes
import logging
import threading
import multiprocessing

import numpy as np
import pytest


RMS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
REPO_CONFIG = os.path.join(RMS_ROOT, '.config')

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

    # Wait for the process, and kill it if it did not end in time
    p.join(timeout)
    alive = p.is_alive()
    if alive:
        p.kill()
        p.join(5)

    return alive


def _discardQueue(q):
    """ Release a test queue without waiting for its feeder at the exit of the test process. """

    q.cancel_join_thread()
    q.close()


def _readRecords(q, p, until, timeout=30):
    """ Read log records from a queue like the listener does, until a condition is met or the child is gone and
        the queue stays empty.

    Arguments:
        q: [Queue] Logging queue.
        p: [Process] Child process writing into the queue.
        until: [callable] Called with the list of messages read so far, True ends the reading.

    Keyword arguments:
        timeout: [float] Upper bound of the reading in seconds.

    Return:
        messages: [list] Messages of the records read.
    """

    # Take records off the queue until the condition is met, the time is up, or the child is gone and nothing
    #   more arrives
    messages = []
    deadline = time.monotonic() + timeout
    while (not until(messages)) and (time.monotonic() < deadline):
        try:
            item = q.get(timeout=1.0)
        except queue.Empty:
            if not p.is_alive():
                break
            continue

        messages.append(item.getMessage() if isinstance(item, logging.LogRecord) else str(item))

    return messages


@pytest.fixture
def startMethod(request):
    """ Make the start method of the test the default one (multiprocessing.Process subclasses such as the
        Compressor use the default), and restore the previous default afterwards.
    """

    previous = multiprocessing.get_start_method(allow_none=True)
    multiprocessing.set_start_method(request.param, force=True)

    yield request.param

    multiprocessing.set_start_method(previous, force=True)


# ---------------------------------------------------------------------------
# A full queue: the records are dropped and counted

def _childLogOnFullQueue(q):
    """ Child side: redirect stderr into logging like a forked RMS child, attach child logging to a full queue,
        and exit 0 only if three log calls return quickly through the drop-on-full handler.
    """

    from RMS.Logger import initChildLogging, LoggerWriter, _DroppingQueueHandler

    # Silence the real stderr, so a recursion (without the fix) does not flood the test output
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)

    # A forked RMS child inherits sys.stderr as a LoggerWriter into the same queue
    sys.stderr = LoggerWriter(logging.getLogger('rmslogger'), logging.WARNING)

    initChildLogging(q, None)

    # Time three log calls into the full queue
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
        handleError -> stderr -> logging, which never returned and stopped the child's work.
    """

    ctx = multiprocessing.get_context(method)

    # Fill the queue, and let the feeder push the items so the queue is really full
    q = ctx.Queue(5)
    for i in range(5):
        q.put(i)
    time.sleep(0.3)

    # Run the child, which ends by itself only if its log calls return
    p = ctx.Process(target=_childLogOnFullQueue, args=(q,))
    p.start()
    killed = _joinOrKill(p, 15)

    _discardQueue(q)

    assert not killed, 'child log calls did not return on a full queue'
    assert p.exitcode == 0


def _childDropsThenLogs(q, dropped, drained):
    """ Child side: log three records into a full queue, tell the parent, wait until it emptied the queue and
        log one more record, which must carry the count of the dropped ones with it.
    """

    from RMS.Logger import initChildLogging

    initChildLogging(q, None)

    # These three find the queue full
    for i in range(3):
        logging.getLogger('rmslogger').info('lost record %d', i)
    dropped.set()

    # The queue has room again
    drained.wait(10)
    logging.getLogger('rmslogger').info('record after the drops')


@posix_only
@pytest.mark.parametrize('method', _startMethods())
def testChildReportsDroppedRecordsWhenQueueHasRoom(method):
    """ The records a child dropped on a full queue are counted, and the count reaches the log as soon as the
        queue has room again, so the loss is visible.
    """

    ctx = multiprocessing.get_context(method)

    # Fill the queue, and let the feeder push the items so the queue is really full
    q = ctx.Queue(5)
    for i in range(5):
        q.put(i)
    time.sleep(0.3)

    # Run the child, which waits for the parent between its drops and its last record
    dropped = ctx.Event()
    drained = ctx.Event()
    p = ctx.Process(target=_childDropsThenLogs, args=(q, dropped, drained))
    p.start()

    # Empty the queue once the child dropped its records, then read what it logs afterwards
    assert dropped.wait(15), 'child log calls did not return on a full queue'
    for i in range(5):
        q.get(timeout=5)
    drained.set()

    messages = _readRecords(q, p, lambda m: len(m) >= 2)
    killed = _joinOrKill(p, 10)

    _discardQueue(q)

    assert not killed
    assert p.exitcode == 0
    assert 'record after the drops' in messages
    assert any(('3 log records from process {:d} were dropped'.format(p.pid) in m) for m in messages), messages


# ---------------------------------------------------------------------------
# The exit of a child: wait while the queue may be read, give up once it was abandoned

def _childLogsIntoUnreadQueue(q):
    """ Child side: attach child logging to a queue nobody reads, log more than a pipe holds and return
        normally.
    """

    from RMS.Logger import initChildLogging

    # More records than the pipe holds stay in the feeder's buffer
    initChildLogging(q, None)
    for i in range(2000):
        logging.getLogger('rmslogger').warning('x'*200)


@posix_only
@pytest.mark.parametrize('method', _startMethods())
def testChildExitWithAbandonedQueueIsBounded(method):
    """ A child still logging into a queue which nobody reads waits at its exit while the queue was not abandoned,
        and exits at once when it is (as the logging watchdog does when it replaces the queue), instead of hanging
        in the join of the queue's feeder thread, which also hung the parent's join() of the child.
    """

    from RMS.Logger import _LoggingQueue

    ctx = multiprocessing.get_context(method)
    q = _LoggingQueue(30000, ctx=ctx)

    # Run the child; nobody reads the queue
    p = ctx.Process(target=_childLogsIntoUnreadQueue, args=(q,))
    p.start()

    # The child waits for its records to be read, longer than the earlier 2 s limit
    time.sleep(3.0)
    waited = p.is_alive()

    # Abandon the queue, after which the child exits by itself
    t_beg = time.monotonic()
    q.abandon()
    killed = _joinOrKill(p, 20)
    elapsed = time.monotonic() - t_beg

    _discardQueue(q)

    assert waited, 'the child gave up its records on a queue which was not abandoned'
    assert not killed, 'child hung at exit joining the logging queue feeder'
    assert p.exitcode == 0
    assert elapsed < 5


def _childSeesAbandonedFlag(q, started):
    """ Child side: exit 0 only if the queue's abandoned flag set by the parent is seen within 10 s. """

    started.set()

    # Wait for the flag
    t_end = time.monotonic() + 10
    while (not q.isAbandoned()) and (time.monotonic() < t_end):
        time.sleep(0.05)

    os._exit(0 if q.isAbandoned() else 1)


@posix_only
@pytest.mark.parametrize('method', [m for m in ('fork', 'forkserver', 'spawn')
                                    if m in multiprocessing.get_all_start_methods()])
def testAbandonedFlagTravelsWithTheQueue(method):
    """ The abandoned flag reaches a child which got the queue as an argument under every start method (pickled
        with the queue under forkserver and spawn).
    """

    from RMS.Logger import _LoggingQueue

    ctx = multiprocessing.get_context(method)
    q = _LoggingQueue(100, ctx=ctx)
    started = ctx.Event()

    # Run the child, and set the flag once it runs
    p = ctx.Process(target=_childSeesAbandonedFlag, args=(q, started))
    p.start()
    assert started.wait(30)
    q.abandon()
    killed = _joinOrKill(p, 20)

    _discardQueue(q)

    assert not killed
    assert p.exitcode == 0


def _childFlushesUnreadQueueAndHardExits(q, timeout):
    """ Child side: log more than a pipe holds into a queue nobody reads, flush like the compressor does before
        os._exit() and exit 0 only if the flush gave up within its time limit.
    """

    from RMS.Logger import initChildLogging, flushChildLogging

    # More records than the pipe holds stay in the feeder's buffer
    initChildLogging(q, None)
    for i in range(2000):
        logging.getLogger('rmslogger').warning('x'*200)

    # Time the flush, which cannot finish
    t_beg = time.monotonic()
    flushChildLogging(timeout=timeout)
    elapsed = time.monotonic() - t_beg

    os._exit(0 if (elapsed < timeout + 1.0) else 1)


@posix_only
@pytest.mark.parametrize('method', _startMethods())
def testFlushChildLoggingIsBoundedOnUnreadQueue(method):
    """ When no watchdog ever abandons a queue which nobody reads, the flush before os._exit() gives up at its
        time limit.
    """

    ctx = multiprocessing.get_context(method)
    q = ctx.Queue(30000)

    # Run the child with a flush limit of 1 s
    p = ctx.Process(target=_childFlushesUnreadQueueAndHardExits, args=(q, 1.0))
    p.start()
    killed = _joinOrKill(p, 20)

    _discardQueue(q)

    assert not killed, 'the flush of the logging queue did not return'
    assert p.exitcode == 0


def _pausedListener(q, resume, got_other):
    """ Listener side: start reading only when told to, then read until the record of another process arrives. """

    resume.wait(60)

    # Read until the record which this test's own process put
    while True:
        item = q.get()
        if item == 'record of another process':
            got_other.set()
            return


def _childLogsShortRecords(q, n):
    """ Child side: attach child logging, log n short records and return normally. """

    from RMS.Logger import initChildLogging

    initChildLogging(q, None)
    for i in range(n):
        logging.getLogger('rmslogger').warning('r%d', i)


@posix_only
@pytest.mark.parametrize('method', _startMethods())
def testPausedListenerKeepsQueueUsable(method):
    """ A child which exits while the listener is paused (longer than the earlier 2 s flush limit) must not give
        up its records in the middle of a write: its feeder held the queue's shared write lock while it was blocked
        on the full pipe, the lock stayed taken after the child exited, and once the listener resumed no record of
        any other process got through any more.
    """

    ctx = multiprocessing.get_context(method)
    q = ctx.Queue(30000)

    # A listener which is paused to start with
    resume = ctx.Event()
    got_other = ctx.Event()
    listener = ctx.Process(target=_pausedListener, args=(q, resume, got_other))
    listener.start()

    # A child logs more than the pipe holds and returns while the listener is paused
    child = ctx.Process(target=_childLogsShortRecords, args=(q, 2000))
    child.start()
    time.sleep(4.0)

    # The listener resumes, reads the child's records, and the child exits
    resume.set()
    killed = _joinOrKill(child, 30)

    # Another process (this one) logs a record, which needs the write lock
    q.put('record of another process')
    delivered = got_other.wait(20)
    lock_free = q._wlock.acquire(timeout=5)
    if lock_free:
        q._wlock.release()

    _joinOrKill(listener, 5)
    _discardQueue(q)

    assert not killed
    assert child.exitcode == 0
    assert delivered, 'a record of another process did not get through after the listener resumed'
    assert lock_free, 'the write lock of the queue stayed taken after the child exited'


def _sentinelListener(queue, records_processed, *args):
    """ Listener side for the manager tests: read until the shutdown sentinel. """

    while True:
        item = queue.get()
        records_processed.value += 1
        if item is None:
            return


@posix_only
def testWatchdogRestartAbandonsTheOldQueue(monkeypatch):
    """ The logging watchdog abandons the old queue when it restarts the logging pipeline, so the children which
        still write into it may exit; the new queue is not abandoned.
    """

    import RMS.Logger as Logger

    monkeypatch.setattr(Logger, '_listener_process', _sentinelListener)

    # A manager with a running listener
    mgr = Logger.LoggingManager()
    mgr._listener_args = ()
    mgr._spawnListener()
    old_queue, old_listener = mgr.logging_queue, mgr.listener_process

    # Restart the pipeline like the watchdog does
    mgr._restartLogging('test')
    new_queue, new_listener = mgr.logging_queue, mgr.listener_process

    abandoned = (old_queue.isAbandoned(), new_queue.isAbandoned())

    # Stop the new listener
    new_queue.put(None)
    _joinOrKill(new_listener, 10)
    _joinOrKill(old_listener, 10)
    _discardQueue(new_queue)

    assert isinstance(old_queue, Logger._LoggingQueue)
    assert abandoned == (True, False)


@posix_only
def testShutdownAbandonsTheQueue(monkeypatch):
    """ Once the listener was stopped at shutdown (it took the sentinel), nobody reads the queue any more, and
        the children which still write into it may exit.
    """

    import RMS.Logger as Logger

    monkeypatch.setattr(Logger, '_listener_process', _sentinelListener)

    # A manager with a running listener
    mgr = Logger.LoggingManager()
    mgr._listener_args = ()
    mgr._spawnListener()
    mgr.is_initialized = True
    q, listener = mgr.logging_queue, mgr.listener_process

    # Shut down: the listener takes the sentinel and stops
    mgr.shutdownLogging()
    killed = _joinOrKill(listener, 10)
    _discardQueue(q)

    assert not killed
    assert listener.exitcode == 0
    assert q.isAbandoned()


# ---------------------------------------------------------------------------
# A listener which reads: the last records of a child arrive

def testFlushChildLoggingWithoutQueueHandlerReturns():
    """ Without a queue handler on the root logger there is nothing to flush, and the call returns at once. """

    from RMS.Logger import flushChildLogging

    # Take the handlers off the root logger for the call
    root = logging.getLogger()
    saved = root.handlers[:]
    root.handlers = [logging.NullHandler()]
    try:
        t_beg = time.monotonic()
        flushChildLogging(timeout=5.0)
        elapsed = time.monotonic() - t_beg
    finally:
        root.handlers = saved

    assert elapsed < 1.0


def _childLogsFlushesAndHardExits(q):
    """ Child side: log one record, flush and os._exit() like the compressor. """

    from RMS.Logger import initChildLogging, flushChildLogging

    initChildLogging(q, None)
    logging.getLogger('rmslogger').warning('last record before os._exit')
    flushChildLogging(timeout=5.0)
    os._exit(0)


@posix_only
@pytest.mark.parametrize('method', _startMethods())
def testFlushChildLoggingDeliversRecordsBeforeHardExit(method):
    """ A record logged right before os._exit() in a child must still reach the listener. """

    ctx = multiprocessing.get_context(method)
    q = ctx.Queue(30000)

    # Run the child and read its record like the listener
    p = ctx.Process(target=_childLogsFlushesAndHardExits, args=(q,))
    p.start()
    messages = _readRecords(q, p, lambda m: len(m) >= 1)
    killed = _joinOrKill(p, 10)

    _discardQueue(q)

    assert not killed
    assert p.exitcode == 0
    assert messages == ['last record before os._exit']


def _childLogsAndReturns(q, n):
    """ Child side: attach child logging, log n records and return normally. """

    from RMS.Logger import initChildLogging

    initChildLogging(q, None)
    for i in range(n):
        logging.getLogger('rmslogger').warning('record %d %s', i, 'y'*100)


@posix_only
@pytest.mark.parametrize('method', _startMethods())
@pytest.mark.parametrize('n', [10, 5000, 50000])
def testChildNormalExitDeliversAllRecords(method, n):
    """ The bounded flush at a normal child exit must not cut off the records a listener is still reading. """

    ctx = multiprocessing.get_context(method)
    q = ctx.Queue(100000)

    # Read like the listener while the child logs and exits
    p = ctx.Process(target=_childLogsAndReturns, args=(q, n))
    p.start()
    messages = _readRecords(q, p, lambda m: len(m) >= n, timeout=60)
    killed = _joinOrKill(p, 10)

    _discardQueue(q)

    assert not killed
    assert p.exitcode == 0
    assert len(messages) == n


@pytest.fixture
def smallConfig(tmp_path):
    """ A config with a small frame size, so the compressor starts in a moment. """

    import RMS.ConfigReader as cr

    config = cr.parse(REPO_CONFIG)
    config.width = 64
    config.height = 48
    config.data_dir = str(tmp_path)
    config.live_jpg = False
    config.enable_fireball_detection = False

    return config


def _startCompressor(config, tmp_path, q, stop_while_waiting):
    """ Start a compressor which logs into the given queue, and bring it to the point where it is stopped.

    Arguments:
        config: [Config] Config with a small frame size.
        tmp_path: [Path] Directory of the test.
        q: [Queue] Logging queue.
        stop_while_waiting: [bool] True to let the compressor compress one block and wait for the next one
            (the exit path in the wait for frames), False to request the stop before it runs (the exit at the end
            of the compression loop).

    Return:
        compressor: [Compressor] The running compressor.
    """

    from RMS.Compression import Compressor
    from RMS.Misc import frameBufferShape

    # Shared frame buffers like the capture's, with no block waiting to start with
    buffer_len = int(np.prod(frameBufferShape(config)))
    array1 = multiprocessing.Array(ctypes.c_uint8, buffer_len)
    array2 = multiprocessing.Array(ctypes.c_uint8, buffer_len)
    start_time1 = multiprocessing.Value('d', 0.0, lock=False)
    start_time2 = multiprocessing.Value('d', 0.0, lock=False)

    out_dir = tmp_path/"night"
    out_dir.mkdir()

    # The compressor logs into the given queue
    compressor = Compressor(str(out_dir), array1, start_time1, array2, start_time2, config)
    compressor.logging_queue = q

    # Let the compressor compress one block and get back to waiting for frames, or request the stop before it
    #   runs so it leaves the compression loop at once
    if stop_while_waiting:
        start_time1.value = time.time()
        compressor.start()

        # Wait for the FF file of the block, then for the compressor to get back to waiting for frames
        deadline = time.monotonic() + 30
        while (time.monotonic() < deadline) and (not any(f.startswith('FF_') for f in os.listdir(str(out_dir)))):
            time.sleep(0.1)
        time.sleep(0.5)

    else:
        compressor.exit_time.value = time.time()
        compressor.exit.set()
        compressor.start()

    return compressor


def _listenAfter(q, delay, wanted, messages, timeout=40):
    """ Listener side, in a thread: start reading after a delay, and read until the wanted messages arrived.

    Arguments:
        q: [Queue] Logging queue.
        delay: [float] Seconds before the reading starts.
        wanted: [list] Messages to read up to.
        messages: [list] The messages read are appended here.

    Keyword arguments:
        timeout: [float] Upper bound of the reading in seconds.
    """

    time.sleep(delay)

    # Read until every wanted message arrived or the time is up
    deadline = time.monotonic() + timeout
    while (not all(w in messages for w in wanted)) and (time.monotonic() < deadline):
        try:
            item = q.get(timeout=1.0)
        except queue.Empty:
            continue
        messages.append(item.getMessage() if isinstance(item, logging.LogRecord) else str(item))


@posix_only
@pytest.mark.parametrize('startMethod', _startMethods(), indirect=True)
@pytest.mark.parametrize('stop_while_waiting', [True, False])
def testCompressorExitRecordsReachBusyListener(startMethod, stop_while_waiting, smallConfig, tmp_path):
    """ The compressor ends through os._exit(), which skips the logging queue's feeder thread: its last record
        ('Compression run exit') must be flushed first, also when the listener is behind and the pipe is still
        full at the compressor's exit. Both exit paths: stopped while waiting for frames, and the end of the
        compression loop (which waits 1 s before os._exit(), so its record also got out before the fix).
    """

    # A listener which is behind: more records wait than the pipe holds
    q = multiprocessing.Queue(30000)
    for i in range(100):
        q.put('backlog record ' + 'x'*1000)

    compressor = _startCompressor(smallConfig, tmp_path, q, stop_while_waiting)

    # Stop the compressor while the listener catches up, starting half a second after the stop request
    messages = []
    listener = threading.Thread(target=_listenAfter, args=(q, 0.5, ['Compression run exit'], messages))
    listener.start()

    compressor.stop()
    listener.join(45)
    killed = _joinOrKill(compressor, 10)

    _discardQueue(q)

    assert not killed
    assert compressor.exitcode == 0
    assert 'Compression run exit' in messages


@posix_only
@pytest.mark.parametrize('startMethod', _startMethods(), indirect=True)
def testCompressorStopWaitsForFlushToPausedListener(startMethod, smallConfig, tmp_path):
    """ A compressor stopped while the listener is paused for longer than the grace period of stop() (5 s) must
        be allowed to finish writing out its last records instead of being terminated in the middle of a write:
        its exit is clean, its last record arrives when the listener resumes, and the queue stays usable for the
        other processes.
    """

    # A paused listener: more records wait than the pipe holds
    q = multiprocessing.Queue(30000)
    for i in range(100):
        q.put('backlog record ' + 'x'*1000)

    compressor = _startCompressor(smallConfig, tmp_path, q, True)

    # The listener resumes 8 s after the stop request
    wanted = ['Compression run exit', 'record of another process']
    messages = []
    listener = threading.Thread(target=_listenAfter, args=(q, 8.0, wanted, messages))
    listener.start()

    compressor.stop()
    killed = _joinOrKill(compressor, 10)

    # Another process (this one) logs a record afterwards
    q.put('record of another process')
    listener.join(45)

    _discardQueue(q)

    assert not killed
    assert compressor.exitcode == 0, 'the compressor was terminated while it was writing out its records'
    assert 'Compression run exit' in messages
    assert 'record of another process' in messages
