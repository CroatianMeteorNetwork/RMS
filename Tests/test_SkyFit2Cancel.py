""" Tests for SkyFit2's Stop-button cancellation scopes (Utils.SkyFit2). """

from __future__ import absolute_import, division, print_function

import os
import threading
import time

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
SkyFit2 = pytest.importorskip("Utils.SkyFit2")
PlateTool = SkyFit2.PlateTool
OperationCancelled = SkyFit2.OperationCancelled


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class FakeGui(object):
    """ Just the pieces the cancellation helpers touch, borrowing the real implementations. """

    _requestCancel = PlateTool._requestCancel
    _beginCancellableOperation = PlateTool._beginCancellableOperation
    _checkCancelled = PlateTool._checkCancelled
    _endCancellableOperation = PlateTool._endCancellableOperation

    def __init__(self):
        self._cancel_requested = False
        self.stop_button = QtWidgets.QPushButton()
        self.messages = []
        self.status_bar = self
        self.stop_button.setVisible(False)
        self._background_thread = None

    def showMessage(self, message):
        self.messages.append(message)


def test_scope_shows_and_hides_the_stop_button(qapp):
    gui = FakeGui()

    PlateTool._beginCancellableOperation(gui)
    assert gui.stop_button.isVisible()

    PlateTool._endCancellableOperation(gui)
    assert not gui.stop_button.isVisible()


def test_cancel_inside_a_scope_raises(qapp):
    gui = FakeGui()
    PlateTool._beginCancellableOperation(gui)

    PlateTool._checkCancelled(gui)  # no request yet - must not raise

    PlateTool._requestCancel(gui)
    with pytest.raises(OperationCancelled):
        PlateTool._checkCancelled(gui)


def test_cancel_between_stages_is_not_swallowed(qapp):
    # A multi-stage operation opens one scope per stage. A Stop click landing in the gap
    # between two stages used to be discarded when the next stage opened its scope, so the
    # user pressed Stop and the operation ran on regardless.
    gui = FakeGui()

    PlateTool._beginCancellableOperation(gui)
    PlateTool._endCancellableOperation(gui)

    PlateTool._requestCancel(gui)          # the click lands in the gap
    PlateTool._beginCancellableOperation(gui)   # next stage starts

    with pytest.raises(OperationCancelled):
        PlateTool._checkCancelled(gui)


def test_a_completed_operation_leaves_no_stale_request(qapp):
    gui = FakeGui()

    PlateTool._beginCancellableOperation(gui)
    PlateTool._requestCancel(gui)
    PlateTool._endCancellableOperation(gui)

    PlateTool._beginCancellableOperation(gui)
    PlateTool._checkCancelled(gui)  # must not raise


def test_a_cooperative_worker_is_told_to_stop(qapp):
    # Abandoning the thread is not enough for a polling worker (the astrometry.net solve polls
    # every 5 s): it has to be handed a stop event, or it keeps working - and printing - long
    # after the user pressed Stop
    gui = FakeGui()
    observed = {'event': None, 'exited': False}

    def worker(stop_event=None):
        observed['event'] = stop_event
        gui._cancel_requested = True          # stands in for the Stop click
        while not stop_event.wait(0.01):
            pass
        observed['exited'] = True
        return "unused"

    with pytest.raises(OperationCancelled):
        PlateTool.runInBackground(gui, worker)

    assert observed['event'] is not None, "worker was never handed a stop event"
    assert observed['event'].is_set(), "stop event was not set on cancel"

    gui._background_thread.join(timeout=2.0)
    assert observed['exited'], "worker did not wind itself up"
    assert not gui._background_thread.is_alive()


def test_a_plain_worker_is_still_abandoned(qapp):
    # A function that knows nothing about stopping must not be handed the keyword (it would
    # raise TypeError) - it is simply left running with its result discarded
    gui = FakeGui()
    started = threading.Event()

    def worker():
        started.set()
        time.sleep(0.5)
        return "unused"

    def cancel_once():
        started.wait(2.0)
        gui._cancel_requested = True

    threading.Thread(target=cancel_once, daemon=True).start()

    with pytest.raises(OperationCancelled):
        PlateTool.runInBackground(gui, worker)

    assert gui._background_thread.is_alive(), "plain worker should be abandoned, not stopped"
    gui._background_thread.join(timeout=2.0)
