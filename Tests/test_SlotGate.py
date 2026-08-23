""" The machine-wide flux slot gate.

The gate exists because six co-located stations reach the heavy part of their
morning processing within minutes of each other and their combined peak
OOM-kills the box. That origin fixes what has to be tested: the cap has to
actually hold across separate processes, and - since the failure the gate
guards against is a SIGKILL - a holder that is killed must not take a slot to
the grave with it. A gate that deadlocked a pod after one OOM would be worse
than no gate at all.

Everything else here is about the gate never being able to block the night's
work: an exhausted gate, a disabled one and an unusable lock directory all have
to fall through to running ungated.
"""

from __future__ import absolute_import, division, print_function

import os
import subprocess
import sys
import time

import pytest

from RMS.SlotGate import slotGate


# fcntl.flock is the whole mechanism
pytestmark = pytest.mark.skipif(sys.platform.startswith("win"),
    reason="the gate degrades to ungated where fcntl is unavailable")


# Child that takes a slot, announces the fact with timestamps, and holds it
WORKER = """
import os, sys, time
sys.path.insert(0, {repo!r})
from RMS.SlotGate import slotGate
with slotGate("t", int(sys.argv[2]), slot_dir=sys.argv[1], poll_interval=0.02) as idx:
    sys.stdout.write("E %f\\n" % time.time())
    sys.stdout.flush()
    time.sleep(float(sys.argv[3]))
    sys.stdout.write("X %f\\n" % time.time())
    sys.stdout.flush()
"""

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _spawn(slot_dir, slots, hold):
    return subprocess.Popen(
        [sys.executable, "-c", WORKER.format(repo=REPO_ROOT),
         str(slot_dir), str(slots), str(hold)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        universal_newlines=True)


def _intervals(procs):
    """ (enter, exit) wall-clock pairs for children that ran to completion. """

    spans = []

    for p in procs:
        out, _ = p.communicate(timeout=60)
        enter = exit_ = None

        for line in out.splitlines():
            if line.startswith("E "):
                enter = float(line.split()[1])
            elif line.startswith("X "):
                exit_ = float(line.split()[1])

        if enter is not None and exit_ is not None:
            spans.append((enter, exit_))

    return spans


def _maxOverlap(spans):
    """ Largest number of spans overlapping at any instant. """

    events = [(t, +1) for t, _ in spans] + [(t, -1) for _, t in spans]

    # Exits sort before entries at equal timestamps, so a slot handed straight
    # over does not read as two concurrent holders
    events.sort(key=lambda e: (e[0], -e[1]))

    peak = running = 0
    for _, delta in events:
        running += delta
        peak = max(peak, running)

    return peak


def test_capHoldsAcrossProcesses(tmpdir):
    """ Six stations, two slots: never three at once. This is the whole point. """

    slot_dir = str(tmpdir.join("slots"))
    procs = [_spawn(slot_dir, 2, 0.4) for _ in range(6)]
    spans = _intervals(procs)

    assert len(spans) == 6
    assert _maxOverlap(spans) <= 2


def test_killedHolderReleasesItsSlot(tmpdir):
    """ The gate guards against OOM kills, so it must survive one.

    flock is released by the kernel when the descriptor closes, including on
    SIGKILL - a holder that is killed must not wedge the remaining stations.
    """

    slot_dir = str(tmpdir.join("slots"))

    victim = _spawn(slot_dir, 1, 60)

    # Wait for it to actually hold the only slot
    assert victim.stdout.readline().startswith("E ")

    victim.kill()
    victim.wait(timeout=30)

    survivor = _spawn(slot_dir, 1, 0.05)
    spans = _intervals([survivor])

    assert len(spans) == 1, "the killed holder leaked its slot"


def test_exhaustedGateProceedsUngated(tmpdir):
    """ A night that cannot get a slot must still be processed, not skipped. """

    slot_dir = str(tmpdir.join("slots"))

    with slotGate("t", 1, slot_dir=slot_dir, poll_interval=0.02) as held:
        assert held == 0

        t_start = time.time()
        with slotGate("t", 1, slot_dir=slot_dir, timeout=0.2,
                      poll_interval=0.02) as second:

            # None means the body runs, just without a slot
            assert second is None
            assert time.time() - t_start >= 0.2


def test_zeroSlotsDisablesTheGate(tmpdir):
    """ 0 is the documented off switch and must not even need a lock directory. """

    slot_dir = str(tmpdir.join("nonexistent", "deeper"))

    with slotGate("t", 0, slot_dir=slot_dir) as held:
        assert held is None

    assert not os.path.exists(slot_dir)


def test_unusableSlotDirDoesNotRaise(tmpdir):
    """ Infrastructure trouble costs the gate, never the night's products. """

    # A regular file where the slot directory should be
    blocker = tmpdir.join("blocked")
    blocker.write("")

    with slotGate("t", 2, slot_dir=str(blocker), timeout=0.2,
                  poll_interval=0.02) as held:
        assert held is None


def test_slotIsReusableAfterRelease(tmpdir):
    """ Releasing must actually free the slot for the next caller in-process. """

    slot_dir = str(tmpdir.join("slots"))

    with slotGate("t", 1, slot_dir=slot_dir, poll_interval=0.02) as first:
        assert first == 0

    with slotGate("t", 1, slot_dir=slot_dir, poll_interval=0.02) as second:
        assert second == 0
