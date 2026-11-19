""" Tests of the hard timeouts around network operations in RMS.Misc.

    runWithTimeout runs a function in a daemon thread and gives up waiting after a deadline. A function which
    finishes after the caller gave up has its late cleanup run, so that a connection it opened is not leaked.

    hostReachable is the cheap reachability check made before every network git operation (e.g. the git lag
    measured for the observation summary). Its name lookup must not be able to block it past its deadline.
"""

from __future__ import print_function, division, absolute_import

import socket
import threading
import time
import types

import RMS.Misc as misc
from RMS.Misc import hostReachable


def testRunWithTimeoutLateCompletionRace(monkeypatch):
    """ If the function completes right between the caller's completion check and the caller giving up,
        the late cleanup must still run (the caller reports a timeout).
    """

    gate = threading.Event()
    workers = []

    class _RecordedThread(threading.Thread):
        """ Thread which is kept in a list when started, so the test can wait for the worker. """

        def start(self):
            workers.append(self)
            super(_RecordedThread, self).start()


    class _RacyEvent(threading.Event):
        """ Event whose first is_set() lets the function finish and then reports the stale state. """

        def __init__(self):
            super(_RacyEvent, self).__init__()
            self.checks = 0

        def is_set(self):

            self.checks += 1
            if self.checks == 1:

                # Let the function finish now, as if it completed during the check, and give the worker
                # the time to run to its end before the caller goes on
                gate.set()
                workers[0].join(0.5)

                return False

            return super(_RacyEvent, self).is_set()


    # Swap the Event and Thread classes only for the threading references of RMS.Misc
    fake_threading = types.SimpleNamespace(Event=_RacyEvent, Thread=_RecordedThread, Lock=threading.Lock)
    monkeypatch.setattr(misc, 'threading', fake_threading)

    cleaned = threading.Event()

    # The function waits for the gate, which only opens during the caller's completion check
    success, _, _ = misc.runWithTimeout(lambda: gate.wait(5), timeout=0.05, on_late_completion=cleaned.set)

    assert success is False
    assert cleaned.wait(5), 'the caller reported a timeout but the late cleanup never ran'


class _FakeSocket(object):
    """ Minimal stand-in for a socket, only records whether it was closed. """

    def __init__(self):
        self.closed = threading.Event()

    def close(self):
        self.closed.set()


def testUnreachableWhenHostIsNone():
    """ No host means nothing to connect to. """

    assert hostReachable(None) is False


def testReturnsTrueWhenConnectionSucceeds(monkeypatch):
    """ A connection which opens straight away is reported as reachable and the socket is closed. """

    # The connection opens straight away
    fake_sock = _FakeSocket()
    monkeypatch.setattr(socket, "create_connection", lambda address, timeout=None: fake_sock)

    assert hostReachable("example.invalid", 443, timeout=1.0) is True
    assert fake_sock.closed.is_set()


def testReturnsFalseOnConnectionError(monkeypatch):
    """ A refused connection or a DNS failure is reported as unreachable. """

    # The name lookup fails straight away
    def _raise(address, timeout=None):
        raise socket.gaierror("Name or service not known")

    monkeypatch.setattr(socket, "create_connection", _raise)

    assert hostReachable("example.invalid", 443, timeout=1.0) is False


def testStalledLookupIsBoundedByDeadline(monkeypatch):
    """ A blocking name lookup must not hold the caller past the deadline, and a connection which comes
        through after the deadline must be closed.
    """

    # The lookup stalls for much longer than the deadline
    deadline = 0.3
    stall = 5*deadline
    fake_sock = _FakeSocket()

    def _stall(address, timeout=None):

        # Simulate getaddrinfo() hanging on a dead resolver, then the connection coming through late
        time.sleep(stall)
        return fake_sock

    monkeypatch.setattr(socket, "create_connection", _stall)

    # Time the check with the stalled lookup
    t_start = time.time()
    reachable = hostReachable("example.invalid", 443, timeout=deadline)
    elapsed = time.time() - t_start

    assert reachable is False
    assert elapsed < stall

    # The late connection is released by the cleanup hook once the stalled call returns
    assert fake_sock.closed.wait(5*stall)
