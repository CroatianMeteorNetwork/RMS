""" Regression tests for the capture-pipeline review fixes: video_crop validation, the thread-safe
    BoundedLock, the bounded child-logging flush, the captureDuration return contract, the single
    horizon-constant definition, QueuedPool Manager shutdown in closePool, and the GRMSUpdater.sh
    option-argument checks.
"""

from __future__ import print_function, division, absolute_import

import os
import stat
import logging
import logging.handlers
import datetime
import threading
import subprocess
import multiprocessing

import pytest


RMS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


# ---------------------------------------------------------------------------
# validVideoCrop

def testValidVideoCropAcceptsWellFormed():
    from RMS.BufferedCapture import validVideoCrop

    assert validVideoCrop("top=8 bottom=8 left=0 right=16")
    assert validVideoCrop("top=8")
    assert validVideoCrop("  left=2   right=2  ")


def testValidVideoCropRejectsEmpty():
    from RMS.BufferedCapture import validVideoCrop

    # An empty spec would put a bare 'videocrop !' into the GStreamer pipeline
    assert not validVideoCrop("")
    assert not validVideoCrop("   ")
    assert not validVideoCrop(None)


def testValidVideoCropRejectsMalformed():
    from RMS.BufferedCapture import validVideoCrop

    assert not validVideoCrop("top=-1")
    assert not validVideoCrop("top=8 middle=2")
    assert not validVideoCrop("top")
    assert not validVideoCrop("top=8; rm -rf /")


# ---------------------------------------------------------------------------
# BoundedLock

def testBoundedLockBasicAcquireRelease():
    from RMS.Misc import BoundedLock

    bl = BoundedLock('test', timeout=0.5)
    with bl:
        # Held: a non-blocking acquire from the same thread must fail
        assert not bl._lock.acquire(block=False)

    # Released after the with block
    assert bl._lock.acquire(block=False)
    bl._lock.release()


def testBoundedLockOtherThreadCannotReleaseHolders():
    """ Two threads sharing one instance: a thread that timed out must not release the lock the
        other thread is still holding (the per-instance flag bug).
    """
    from RMS.Misc import BoundedLock

    bl = BoundedLock('test', timeout=0.2)
    holder_entered = threading.Event()
    release_holder = threading.Event()

    def holder():
        with bl:
            holder_entered.set()
            release_holder.wait(5)

    t = threading.Thread(target=holder)
    t.daemon = True
    t.start()
    assert holder_entered.wait(5)

    # This thread times out, proceeds without the lock and must NOT release on exit
    with bl:
        pass

    assert not bl._lock.acquire(block=False), "second thread released a lock it never acquired"

    release_holder.set()
    t.join(5)

    # The holder's exit released it
    assert bl._lock.acquire(block=False)
    bl._lock.release()


def testBoundedLockHasNoThreadLocalState():
    """ The lock lives on Process subclasses that are pickled under spawn/forkserver, and a
        threading.local cannot be pickled - the per-thread bookkeeping must be a plain container.
    """
    from RMS.Misc import BoundedLock

    bl = BoundedLock('test', timeout=0.5)
    with bl:
        assert threading.get_ident() in bl._holders

    assert bl._holders == {}
    assert not any(isinstance(v, threading.local) for v in vars(bl).values())


# ---------------------------------------------------------------------------
# flushChildLogging

def testFlushChildLoggingNoHandlersReturnsQuickly():
    from RMS.Logger import flushChildLogging

    root = logging.getLogger()
    saved = root.handlers[:]
    root.handlers = []
    try:
        flushChildLogging(timeout=1.0)
    finally:
        root.handlers = saved


def _logFlushAndHardExit(q):
    """ Child side: log one record through a QueueHandler, flush, and os._exit like Compressor. """
    from RMS.Logger import flushChildLogging

    root = logging.getLogger()
    root.handlers = [logging.handlers.QueueHandler(q)]
    logging.getLogger('flush_test').warning('buffered record')
    flushChildLogging(timeout=5.0)
    os._exit(0)


def testFlushChildLoggingDeliversBufferedRecords():
    """ A record logged right before os._exit() in a child must still reach the parent's queue. """

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_logFlushAndHardExit, args=(q,))
    p.start()
    p.join(15)
    assert p.exitcode == 0

    record = q.get(timeout=5)
    assert 'buffered record' in record.getMessage()


# ---------------------------------------------------------------------------
# captureDuration return contract and horizon constants

def testCaptureDurationPolarNightReturnsBoolStart():
    from RMS.CaptureDuration import captureDuration

    # Svalbard, mid winter: the Sun never rises, capture starts right away
    start_time, duration = captureDuration(78.2, 15.6, 0,
        current_time=datetime.datetime(2023, 12, 21, 12, 0, 0))

    assert isinstance(start_time, bool) and start_time is True
    assert isinstance(duration, (int, float)) and duration > 0


def testCaptureDurationMidLatitudeReturnsDatetimeOrTrue():
    from RMS.CaptureDuration import captureDuration

    start_time, duration = captureDuration(43.0, -81.0, 265,
        current_time=datetime.datetime(2023, 6, 21, 12, 0, 0))

    assert isinstance(start_time, (bool, datetime.datetime))
    assert 0 < duration <= 23*3600


def testHorizonConstantsSingleDefinition():
    import RMS.CaptureDuration as cd
    import RMS.CaptureModeSwitcher as cms
    import RMS.Formats.ObservationSummary as obs

    assert cms.SWITCH_HORIZON_DEG is cd.SWITCH_HORIZON_DEG
    assert cms.CAPTURE_HORIZON_DEG is cd.CAPTURE_HORIZON_DEG
    assert obs.SWITCH_HORIZON_DEG is cd.SWITCH_HORIZON_DEG


# ---------------------------------------------------------------------------
# QueuedPool Manager shutdown

def _square(x):
    return x*x


def testQueuedPoolClosePoolShutsDownManagerKeepsResults():
    from RMS.QueuedPool import QueuedPool

    qp = QueuedPool(_square, cores=1, log=None, backup_dir=None, print_state=False)
    qp.startPool()
    for i in range(5):
        qp.addJob([i])

    qp.closePool()

    # closePool released the Manager server process on its own...
    assert qp.manager is None

    # ...and the results survived the shutdown; a second call yields nothing new
    assert sorted(qp.getResults()) == [i*i for i in range(5)]
    assert qp.getResults() == []

    # Idempotent for callers that still call it explicitly
    qp.shutdownManager()


def testQueuedPoolWorkersFallbackWithoutPrivateList():
    from RMS.QueuedPool import QueuedPool

    qp = QueuedPool(_square, cores=1, log=None, backup_dir=None, print_state=False)

    class _NoPool(object):
        pass

    # No pool yet, or a pool object without the private worker list: graceful empty list
    assert qp._poolWorkers() == []
    qp.pool = _NoPool()
    assert qp._poolWorkers() == []
    qp.pool = None
    qp.shutdownManager()


# ---------------------------------------------------------------------------
# GRMSUpdater.sh option-argument checks

def _runUpdater(args, tmp_path):
    """ Run the updater with a no-op `logger` on PATH so the test does not write to syslog. """

    script = os.path.join(RMS_ROOT, 'Scripts', 'MultiCamLinux', 'GRMSUpdater.sh')
    fake_bin = tmp_path/'bin'
    fake_bin.mkdir()
    fake_logger = fake_bin/'logger'
    fake_logger.write_text('#!/bin/sh\nexit 0\n')
    fake_logger.chmod(fake_logger.stat().st_mode | stat.S_IXUSR)

    env = dict(os.environ)
    env['PATH'] = str(fake_bin) + os.pathsep + env.get('PATH', '')

    proc = subprocess.run(['bash', script] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, timeout=60)
    return proc.returncode, proc.stdout.decode('utf-8', 'replace')


@pytest.mark.parametrize('option', ['--profile', '--term'])
def testUpdaterOptionWithoutArgumentFails(option, tmp_path):

    code, out = _runUpdater([option], tmp_path)

    if 'Another GRMSUpdater instance' in out:
        pytest.skip('a real GRMSUpdater instance holds the lock')

    assert code != 0
    assert 'requires' in out
    assert 'Usage: GRMSUpdater.sh' in out


def testUpdaterHelpExitsZero(tmp_path):

    code, out = _runUpdater(['--help'], tmp_path)

    assert code == 0
    assert 'Usage: GRMSUpdater.sh' in out
