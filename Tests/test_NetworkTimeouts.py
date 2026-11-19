""" Tests of the hard timeouts around network operations in RMS.Misc.

    runWithTimeout runs a function in a daemon thread and gives up waiting after a deadline. A function which
    finishes after the caller gave up has its late cleanup run, so that a connection it opened is not leaked.
"""

from __future__ import print_function, division, absolute_import

import threading
import types

import RMS.Misc as misc


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
