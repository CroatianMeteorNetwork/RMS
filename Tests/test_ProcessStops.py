""" Tests of stopping the processes StartCapture runs during a night: the compression, the capture, the upload
    manager and the event monitor.

    Every wait of a stop must be bounded. A process stuck in uninterruptible (D-state) I/O survives even SIGKILL
    until the I/O completes, and an unbounded join() on it wedges the end of the night (the night is never
    processed, and StartCapture keeps the station lock so no restart can run). Such a process cannot be made in
    a test, so the stops are run on stand-ins which stay alive through every signal and record the joins and
    signals. The compressor ignores SIGINT, so its stop must not wait for an interrupt to work, but a compressor
    which finishes late must still get its grace. The upload loop must not sleep while it holds the locks which
    the parent takes when it queues files or delays the upload.
"""

from __future__ import print_function, division, absolute_import

import os
import sys
import time
import ctypes
import signal
import threading
import multiprocessing

import numpy as np
import pytest


RMS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
REPO_CONFIG = os.path.join(RMS_ROOT, '.config')

sys.path.insert(0, RMS_ROOT)

from RMS.Misc import AtomicFlag, BoundedLock


# PID of the stand-ins, of no real process
FAKE_PID = 999999999


class _FastClock(object):
    """ Stand-in for the time module of the module under test: every time() call is a minute later and sleep()
        returns at once, so the waits of a stop end without waiting.
    """

    def __init__(self):

        self.now = time.time()


    def time(self):
        """ Return a time a minute after the one returned before. """

        self.now += 61.0

        return self.now


    def sleep(self, seconds):
        """ Return at once. """



class _SignalRecorder(object):
    """ Stand-in for os.kill which records the signals instead of sending them. """

    def __init__(self):

        self.signals = []


    def __call__(self, pid, sig):
        """ Record the signal.

        Arguments:
            pid: [int] Process ID.
            sig: [int] Signal number.
        """

        self.signals.append(sig)



class _ProcessStandIn(object):
    """ Stand-in for a child process, for calling the stop method of its class without starting a process. It
        records the timeouts of the joins and whether it was terminated, and stays alive through every signal
        unless the scenario of a subclass ends it.
    """

    def __init__(self):

        self.pid = FAKE_PID
        self.exit = AtomicFlag()
        self.join_timeouts = []
        self.terminated = False
        self.alive = True


    def is_alive(self):
        """ Return True while the stand-in is alive. """

        return self.alive


    def terminate(self):
        """ Record the terminate, the stand-in does not die from it. """

        self.terminated = True


    def join(self, timeout=None):
        """ Record the timeout of the join and return at once.

        Keyword arguments:
            timeout: [float] Timeout of the join, None for a join without a bound.
        """

        self.join_timeouts.append(timeout)



class _CompressorStandIn(_ProcessStandIn):
    """ Stand-in for a Compressor which does not exit by itself.

    Arguments:
        run_exited: [bool] True if its run() finished its work and the process is writing out its last log
            records, False if it is still compressing.
    """

    def __init__(self, run_exited):

        super(_CompressorStandIn, self).__init__()

        self.exit_time = multiprocessing.Value('d', 0.0, lock=False)
        self.run_exited = AtomicFlag()
        if run_exited:
            self.run_exited.set()

        self.detector = 'detector'



class _StuckCompressor(_CompressorStandIn):
    """ Stand-in for a Compressor which only a terminate ends. """

    def terminate(self):
        """ The terminate ends the process. """

        self.terminated = True
        self.alive = False



class _LateCompressor(_CompressorStandIn):
    """ Stand-in for a Compressor which finishes its work only after the stop gave up waiting for it: it
        finishes during the first join and is still writing out its last log records at its end, then exits
        during the next join.
    """

    def __init__(self):

        super(_LateCompressor, self).__init__(False)


    def join(self, timeout=None):
        """ Finish the work during the first join, and exit during the second one. """

        self.join_timeouts.append(timeout)

        if len(self.join_timeouts) == 1:
            self.run_exited.set()

        else:
            self.alive = False



@pytest.fixture
def compression(monkeypatch):
    """ The Compression module with a fast clock and the signals recorded instead of sent.

    Return:
        (comp, recorder): [module, _SignalRecorder] The module and the record of the signals.
    """

    import RMS.Compression as comp

    recorder = _SignalRecorder()
    monkeypatch.setattr(comp, 'time', _FastClock())
    monkeypatch.setattr(comp.os, 'kill', recorder)

    return comp, recorder



# ---------------------------------------------------------------------------
# Compressor.stop()

def testCompressorStopSendsNoInterrupt(compression):
    """ The compressor ignores SIGINT (initChildProcess), so a compressor which does not exit is terminated
        without an interrupt first, which only delayed the stop and logged that an interrupt was sent.
    """

    comp, recorder = compression

    stand_in = _StuckCompressor(True)
    assert comp.Compressor.stop(stand_in) == 'detector'

    assert stand_in.terminated
    assert recorder.signals == []
    assert all(t is not None for t in stand_in.join_timeouts), stand_in.join_timeouts


def testLateCompressorGetsItsGraceAndFlush(compression):
    """ A compressor which finishes its work only after the wait of stop() ran out gets a few more seconds, and
        once it finished, the time for writing out its last log records: it is neither interrupted nor
        terminated in the middle of the write (which would leave the logging queue's write lock taken).
    """

    from RMS.Logger import LOG_CHILD_EXIT_FLUSH_TIMEOUT

    comp, recorder = compression

    stand_in = _LateCompressor()
    comp.Compressor.stop(stand_in)

    assert not stand_in.terminated, 'the compressor was terminated while it was writing out its records'
    assert recorder.signals == []
    assert stand_in.join_timeouts[:2] == [5, LOG_CHILD_EXIT_FLUSH_TIMEOUT + 5]


def testCompressorStopLeavesUnkillableProcess(compression):
    """ A compressor stuck in uninterruptible I/O survives SIGKILL: stop() sends SIGKILL, then gives up on the
        process after a bounded join instead of waiting for it forever, and returns the detector.
    """

    comp, recorder = compression

    stand_in = _CompressorStandIn(True)
    assert comp.Compressor.stop(stand_in) == 'detector'

    assert stand_in.join_timeouts, 'stop() never joined'
    assert all(t is not None for t in stand_in.join_timeouts), 'unbounded join: {}'.format(
        stand_in.join_timeouts)
    assert stand_in.terminated
    assert recorder.signals == [signal.SIGKILL]


def testStopOfCompressorWhichNeverStarted(compression, tmp_path):
    """ stop() on a compressor which was never started returns without an error (join() raises on a process
        which was never started).
    """

    import RMS.ConfigReader as cr
    from RMS.Misc import frameBufferShape

    comp, recorder = compression

    # A real compressor with small frame buffers, never started
    config = cr.parse(REPO_CONFIG)
    config.width = 64
    config.height = 48
    config.data_dir = str(tmp_path)

    buffer_len = int(np.prod(frameBufferShape(config)))
    array1 = multiprocessing.Array(ctypes.c_uint8, buffer_len)
    array2 = multiprocessing.Array(ctypes.c_uint8, buffer_len)
    start_time1 = multiprocessing.Value('d', 0.0, lock=False)
    start_time2 = multiprocessing.Value('d', 0.0, lock=False)

    compressor = comp.Compressor(str(tmp_path), array1, start_time1, array2, start_time2, config,
        detector='detector')

    assert compressor.stop() == 'detector'
    assert recorder.signals == []



# ---------------------------------------------------------------------------
# BufferedCapture.stopCapture()

class _UnkillableCapture(_ProcessStandIn):
    """ Stand-in for a BufferedCapture stuck in uninterruptible V4L2 or USB I/O. """

    def __init__(self):

        super(_UnkillableCapture, self).__init__()

        self.raw_frame_saver = None
        self.dropped_frames = multiprocessing.Value('i', 7, lock=False)



def testStopCaptureLeavesUnkillableProcess(monkeypatch):
    """ stopCapture() sends SIGKILL to a capture which survives the terminate, then gives up on it after a
        bounded join instead of wedging the end of the night, and returns the number of dropped frames.
    """

    import RMS.BufferedCapture as bcm

    recorder = _SignalRecorder()
    monkeypatch.setattr(bcm, 'time', _FastClock())
    monkeypatch.setattr(bcm.os, 'kill', recorder)

    stand_in = _UnkillableCapture()
    assert bcm.BufferedCapture.stopCapture(stand_in) == 7

    assert stand_in.join_timeouts, 'stopCapture() never joined'
    assert all(t is not None for t in stand_in.join_timeouts), 'unbounded join: {}'.format(
        stand_in.join_timeouts)
    assert stand_in.terminated
    assert recorder.signals == [signal.SIGKILL]



# ---------------------------------------------------------------------------
# UploadManager.stop() and the upload loop

class _UnkillableUploader(_ProcessStandIn):
    """ Stand-in for an UploadManager stuck in uninterruptible I/O (an NFS share, a wedged USB disk). """

    def __init__(self):

        super(_UnkillableUploader, self).__init__()

        self.killed = False
        self.manager_shut_down = False


    def kill(self):
        """ Record the SIGKILL, the stand-in does not die from it. """

        self.killed = True


    def _shutdownManager(self):
        """ Record the shutdown of the Manager server. """

        self.manager_shut_down = True



def testUploadManagerStopLeavesUnkillableProcess():
    """ stop() sends SIGKILL to an upload manager which survives the terminate, gives up on it after a bounded
        join instead of wedging the shutdown of StartCapture, and still shuts down the Manager server.
    """

    pytest.importorskip("paramiko")
    import RMS.UploadManager as um

    stand_in = _UnkillableUploader()
    um.UploadManager.stop(stand_in, timeout=0.01)

    assert stand_in.join_timeouts, 'stop() never joined'
    assert all(t is not None for t in stand_in.join_timeouts), 'unbounded join: {}'.format(
        stand_in.join_timeouts)
    assert stand_in.terminated
    assert stand_in.killed
    assert stand_in.manager_shut_down



class _UploadLoopStandIn(object):
    """ Stand-in for the state of an UploadManager used by its run() loop, without the upload. """

    def __init__(self):

        self.exit = AtomicFlag()
        self.last_runtime_lock = BoundedLock('upload last_runtime')
        self.next_runtime_lock = BoundedLock('upload next_runtime')
        self.last_runtime = multiprocessing.Value('d', 0.0, lock=False)
        self.next_runtime = multiprocessing.Value('d', 0.0, lock=False)
        self.logging_queue = None
        self.config = None
        self.parent_pid = None
        self.uploads = 0


    def loadQueue(self):
        """ No queue file. """


    def uploadData(self):
        """ Count the uploads instead of uploading. """

        self.uploads += 1



@pytest.mark.parametrize('which', ['last', 'next'])
def testUploadLoopLeavesLocksFreeWhileWaiting(monkeypatch, which):
    """ While the upload loop waits, for the 15 minutes after an upload or for a delayed upload, the parent gets
        the runtime locks at once. The loop used to sleep while holding them, so the parent's addFiles() and
        delayNextUpload() waited on them and timed out with a warning that a process had died holding the lock.
    """

    pytest.importorskip("paramiko")
    import RMS.UploadManager as um

    # Keep the logging set-up of a child process out of the test process
    monkeypatch.setattr(um, 'initChildProcess', lambda *args, **kwargs: None)
    monkeypatch.setattr(um, 'setParentDeathSignal', lambda *args, **kwargs: False)

    stand_in = _UploadLoopStandIn()

    # Make the loop wait: on the 15 minutes after one upload, or on an upload delayed by an hour
    if which == 'next':
        stand_in.next_runtime.value = time.time() + 3600

    loop = threading.Thread(target=um.UploadManager.run, args=(stand_in,))
    loop.daemon = True
    loop.start()

    try:

        # Take the lock of the wait a few times like the parent does, each must be free at once
        time.sleep(0.5)
        lock = stand_in.last_runtime_lock if (which == 'last') else stand_in.next_runtime_lock

        for _ in range(3):

            t_beg = time.monotonic()
            acquired = lock._lock.acquire(timeout=2.0)
            elapsed = time.monotonic() - t_beg
            if acquired:
                lock._lock.release()

            assert acquired and (elapsed < 0.5), 'the lock was held by the waiting loop ({:.2f} s)'.format(
                elapsed)

            time.sleep(0.3)

    finally:
        stand_in.exit.set()
        loop.join(5)

    assert not loop.is_alive()
    assert stand_in.uploads == (1 if (which == 'last') else 0)



# ---------------------------------------------------------------------------
# EventMonitor.stop() and the start delay

class _EventMonitorStandIn(_ProcessStandIn):
    """ Stand-in for an EventMonitor whose check is stuck in network or file I/O, and which a terminate does
        not end either.
    """

    def __init__(self):

        super(_EventMonitorStandIn, self).__init__()

        self.db_conn = None



def testEventMonitorStopIsBounded(monkeypatch):
    """ stop() waits for a check in progress with a bound, terminates the process after it, and gives up on a
        process which survives the terminate instead of wedging the shutdown of StartCapture.
    """

    pytest.importorskip("paramiko")
    import RMS.EventMonitor as em

    monkeypatch.setattr(em, 'time', _FastClock())

    stand_in = _EventMonitorStandIn()
    em.EventMonitor.stop(stand_in)

    assert stand_in.exit.is_set()
    assert stand_in.join_timeouts, 'stop() never joined'
    assert all(t is not None for t in stand_in.join_timeouts), 'unbounded join: {}'.format(
        stand_in.join_timeouts)
    assert stand_in.terminated



class _EventLoopStandIn(object):
    """ Stand-in for the state of an EventMonitor used by its run() loop up to the first check. """

    def __init__(self):

        self.exit = AtomicFlag()
        self.logging_queue = None
        self.config = None
        self.parent_pid = None


    def getConnectionToEventMonitorDB(self):
        """ No database. """

        return None



def testEventMonitorStartDelayEndsOnStop(monkeypatch):
    """ A stop requested during the one-minute delay at the start of the event monitor ends the delay at once,
        instead of holding up the stop of StartCapture for the rest of the minute.
    """

    pytest.importorskip("paramiko")
    import RMS.EventMonitor as em

    # Keep the logging set-up of a child process out of the test process
    monkeypatch.setattr(em, 'initChildProcess', lambda *args, **kwargs: None)
    monkeypatch.setattr(em, 'setParentDeathSignal', lambda *args, **kwargs: False)

    stand_in = _EventLoopStandIn()

    loop = threading.Thread(target=em.EventMonitor.run, args=(stand_in,))
    loop.daemon = True
    loop.start()

    # Request the stop during the delay
    time.sleep(0.5)
    t_stop = time.monotonic()
    stand_in.exit.set()
    loop.join(10)

    assert not loop.is_alive(), 'the delay did not end on the stop'
    assert (time.monotonic() - t_stop) < 5
