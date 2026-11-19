""" Tests of the logging in child processes (capture, compression, extraction, upload, ...), which forward their
    records through a bounded multiprocessing queue to the listener process.

    A child logging into a full queue must drop the records and count them instead of blocking or recursing, a
    child must exit in bounded time when nobody reads its logging queue any more (a stopped listener, or the
    queue the logging watchdog abandoned), and the records a child logs right before it exits, through
    os._exit() or normally, must still reach a listener which reads.

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
# Nobody reads the queue: the exit is bounded

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
    """ A child still logging into a queue whose listener does not read any more (stopped, or the queue was
        abandoned by a logging pipeline restart) must exit within the bounded flush, instead of hanging in the
        join of the queue's feeder thread, which also hung the parent's join() of the child.
    """

    ctx = multiprocessing.get_context(method)
    q = ctx.Queue(30000)

    # Run the child and time it until it exited
    p = ctx.Process(target=_childLogsIntoUnreadQueue, args=(q,))
    t_beg = time.monotonic()
    p.start()
    killed = _joinOrKill(p, 20)
    elapsed = time.monotonic() - t_beg

    _discardQueue(q)

    assert not killed, 'child hung at exit joining the logging queue feeder'
    assert p.exitcode == 0
    assert elapsed < 15


def _childFlushesUnreadQueueAndHardExits(q, timeout):
    """ Child side: log more than a pipe holds into a queue nobody reads, flush like the compressor does before
        os._exit() and exit 0 only if the flush gave up within its bound.
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
    """ The flush before os._exit() must give up after its timeout when nobody reads the queue. """

    ctx = multiprocessing.get_context(method)
    q = ctx.Queue(30000)

    # Run the child with a flush limit of 1 s
    p = ctx.Process(target=_childFlushesUnreadQueueAndHardExits, args=(q, 1.0))
    p.start()
    killed = _joinOrKill(p, 20)

    _discardQueue(q)

    assert not killed, 'the flush of the logging queue did not return'
    assert p.exitcode == 0


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


@posix_only
@pytest.mark.parametrize('startMethod', _startMethods(), indirect=True)
@pytest.mark.parametrize('stop_while_waiting', [True, False])
def testCompressorExitRecordsReachBusyListener(startMethod, stop_while_waiting, smallConfig, tmp_path):
    """ The compressor ends through os._exit(), which skips the logging queue's feeder thread: its last record
        ('Compression run exit') must be flushed first, also when the listener is behind and the pipe is still
        full at the compressor's exit. Both exit paths: stopped while waiting for frames, and the end of the
        compression loop (which waits 1 s before os._exit(), so its record also got out before the fix).
    """

    from RMS.Compression import Compressor
    from RMS.Misc import frameBufferShape

    # Shared frame buffers like the capture's, with no block waiting to start with
    buffer_len = int(np.prod(frameBufferShape(smallConfig)))
    array1 = multiprocessing.Array(ctypes.c_uint8, buffer_len)
    array2 = multiprocessing.Array(ctypes.c_uint8, buffer_len)
    start_time1 = multiprocessing.Value('d', 0.0, lock=False)
    start_time2 = multiprocessing.Value('d', 0.0, lock=False)

    out_dir = tmp_path/"night"
    out_dir.mkdir()

    # A listener which is behind: more records wait than the pipe holds
    q = multiprocessing.Queue(30000)
    for i in range(100):
        q.put('backlog record ' + 'x'*1000)

    # The compressor logs into that queue
    compressor = Compressor(str(out_dir), array1, start_time1, array2, start_time2, smallConfig)
    compressor.logging_queue = q

    # Stop the compressor while it waits for frames after compressing one block, or request the stop before it
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

    # The listener starts reading half a second after the stop request, well inside the exit flush limit
    messages = []
    def _listen():
        time.sleep(0.5)
        deadline = time.monotonic() + 20
        while ('Compression run exit' not in messages) and (time.monotonic() < deadline):
            try:
                item = q.get(timeout=1.0)
            except queue.Empty:
                continue
            messages.append(item.getMessage() if isinstance(item, logging.LogRecord) else str(item))

    # Stop the compressor while the listener catches up
    listener = threading.Thread(target=_listen)
    listener.start()

    compressor.stop()
    listener.join(30)
    killed = _joinOrKill(compressor, 10)

    _discardQueue(q)

    assert not killed
    assert compressor.exitcode == 0
    assert 'Compression run exit' in messages
