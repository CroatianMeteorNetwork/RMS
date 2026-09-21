""" Tests for RMS.Misc.hostReachable, in particular that a stalled name lookup cannot block it. """

from __future__ import print_function, division, absolute_import

import socket
import time

import pytest

import RMS.Misc
from RMS.Misc import hostReachable


class _FakeSocket(object):
    """ Minimal stand-in for a socket, only records whether it was closed. """

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def testUnreachableWhenHostIsNone():
    """ No host means nothing to connect to. """

    assert hostReachable(None) is False


def testReturnsTrueWhenConnectionSucceeds(monkeypatch):
    """ A connection which opens straight away is reported as reachable and the socket is closed. """

    fake_sock = _FakeSocket()
    monkeypatch.setattr(socket, "create_connection", lambda address, timeout=None: fake_sock)

    assert hostReachable("example.invalid", 443, timeout=1.0) is True
    assert fake_sock.closed is True


def testReturnsFalseOnConnectionError(monkeypatch):
    """ A refused connection or a DNS failure is reported as unreachable. """

    def _raise(address, timeout=None):
        raise socket.gaierror("Name or service not known")

    monkeypatch.setattr(socket, "create_connection", _raise)

    assert hostReachable("example.invalid", 443, timeout=1.0) is False


def testStalledLookupIsBoundedByDeadline(monkeypatch):
    """ A blocking name lookup must not hold the caller past the deadline. """

    deadline = 0.3
    stall = 5*deadline
    fake_sock = _FakeSocket()

    def _stall(address, timeout=None):

        # Simulate getaddrinfo() hanging on a dead resolver, then the connection coming through late
        time.sleep(stall)
        return fake_sock

    monkeypatch.setattr(socket, "create_connection", _stall)

    t_start = time.time()
    reachable = hostReachable("example.invalid", 443, timeout=deadline)
    elapsed = time.time() - t_start

    assert reachable is False
    assert elapsed < stall

    # The late connection is released by the cleanup hook once the stalled call returns
    time.sleep(stall)
    assert fake_sock.closed is True
