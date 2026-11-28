""" Unit tests for NTP clock offset estimation in RMS.Formats.ObservationSummary. """

from __future__ import print_function, division, absolute_import

import socket
import struct
import types

import pytest

import RMS.Formats.ObservationSummary as osm


class _FakeNTPSocket(object):
    """ UDP socket stand-in answering one NTP request with the given remote timestamps. """

    def __init__(self, t_remote_receive, t_remote_transmit, fail=None):
        self.t_remote_receive = t_remote_receive
        self.t_remote_transmit = t_remote_transmit
        self.fail = fail
        self.closed = False

    def settimeout(self, t):
        return None

    def sendto(self, data, addr):
        if self.fail is not None:
            raise self.fail

    def recvfrom(self, n):

        # Pack the remote receive and transmit timestamps into the standard 48-byte NTP reply
        def ntp(t):
            t = t + 2208988800
            return int(t), int(round((t - int(t))*2**32))

        words = [0]*12
        words[8], words[9] = ntp(self.t_remote_receive)
        words[10], words[11] = ntp(self.t_remote_transmit)
        return struct.pack('!12I', *words), ('127.0.0.1', 123)

    def close(self):
        self.closed = True


def _patchNTP(monkeypatch, sock, local_times):
    """ Route timestampFromNTP's socket and clock through deterministic stand-ins. """

    fake_socket_module = types.SimpleNamespace(
        socket=lambda *a: sock, AF_INET=socket.AF_INET,
        SOCK_DGRAM=socket.SOCK_DGRAM, timeout=socket.timeout
    )
    times = list(local_times)
    fake_time_module = types.SimpleNamespace(time=lambda: times.pop(0))

    monkeypatch.setattr(osm, 'socket', fake_socket_module)
    monkeypatch.setattr(osm, 'time', fake_time_module)


@pytest.mark.parametrize('delay, processing', [(0.0, 0.0), (0.1, 0.01)])
def testNTPOffsetLocalClockOneSecondAhead(monkeypatch, delay, processing):
    """ Local clock 1 s ahead with a symmetric network delay: the remote time estimate at the
        receive instant must be 1 s behind the local receive time (clock_ahead_ms = 1000).
    """

    t_local_tx = 1.7e9
    t_local_rx = t_local_tx + delay + processing

    # Remote clock = local clock - 1 s, with the given processing time on the server
    t_remote_rx = t_local_tx + delay/2 - 1.0
    t_remote_tx = t_remote_rx + processing

    sock = _FakeNTPSocket(t_remote_rx, t_remote_tx)
    _patchNTP(monkeypatch, sock, [t_local_tx, t_local_rx])

    remote_time, network_delay, addr = osm.timestampFromNTP('ntp.example')

    ahead_ms = (t_local_rx - remote_time)*1000
    assert abs(ahead_ms - 1000) < 1e-3
    assert abs(network_delay - delay) < 1e-6
    assert addr == 'ntp.example'
    assert sock.closed


@pytest.mark.parametrize('exc', [socket.timeout(), OSError("network unreachable")])
def testNTPSocketFailureReturnsThreeValuesAndClosesSocket(monkeypatch, exc):
    """ A socket failure must close the socket and return (None, None, addr) for the caller's 3-way unpack. """

    sock = _FakeNTPSocket(0, 0, fail=exc)
    _patchNTP(monkeypatch, sock, [1.7e9, 1.7e9])

    assert osm.timestampFromNTP('ntp.example') == (None, None, 'ntp.example')
    assert sock.closed
