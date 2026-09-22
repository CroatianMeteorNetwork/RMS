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
