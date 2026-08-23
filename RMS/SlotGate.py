""" A counting semaphore shared by every RMS process on one machine.

Co-located stations run independent capture processes that know nothing about each
other, and their morning pipelines are driven by the same sunrise - so the heaviest
phase of every station's night processing starts within minutes of every other's.
Each one's peak is legitimate working set proportional to the night's star-record
count, which does not shrink with camera resolution: on a six-camera pod the same
~1 GB per station lands six times over, and the kernel kills whichever process is
largest. Serializing that phase is the only measure that bounds the concurrent peak;
releasing memory sooner and running the stage in its own process cap growth but do
not stop six peaks from coinciding.

The gate is built on fcntl.flock over a small set of lock files, one per slot:

- The kernel releases a flock when the file descriptor closes, including when the
  holder is killed. A station that is OOM-killed mid-stage therefore cannot leak a
  slot, and there is no stale-lock reaper to get wrong. This is the property a
  PID-file or directory-based lock would not give us, and it matters precisely
  because the failure this gate exists to prevent is a SIGKILL.
- Slots are acquired non-blocking and retried, rather than blocking on one chosen
  slot, so a waiter takes whichever frees first instead of queueing behind a
  specific holder.

Waiting is bounded. The gate is a memory optimization, not a correctness
requirement, and the pipeline's standing invariant is that a night must still be
archived and uploaded even when a stage fails - so a waiter that cannot get a slot
in time proceeds ungated and says so at WARNING rather than skipping the work.
"""

from __future__ import print_function, division, absolute_import

import errno
import os
import time

from contextlib import contextmanager

from RMS.Logger import getLogger
from RMS.Misc import rssSuffix

log = getLogger("rmslogger")


# Shared by every station on the box, so it cannot live under any one station's data
# directory. /tmp is per-boot, which is what we want: slot state must never outlive
# the processes holding it.
DEFAULT_SLOT_DIR = "/tmp/.rms_flux_slots"

# Seconds between sweeps of the slot files. Long enough that a multi-hour wait costs
# nothing measurable, short enough that a freed slot is picked up promptly.
POLL_INTERVAL = 5.0

# Cap on how long a station will wait before giving up and running ungated. Sized
# against the caller's own ceiling: RMS.Reprocess.FLUX_STAGE_TIMEOUT kills a wedged
# flux stage at 4 h, so a slot cannot credibly stay held longer than that.
DEFAULT_TIMEOUT = 4*3600


@contextmanager
def slotGate(name, slots, slot_dir=None, timeout=DEFAULT_TIMEOUT,
             poll_interval=POLL_INTERVAL):
    """ Hold one of `slots` machine-wide slots for the duration of the block.

    Never raises on account of the gate itself: any failure to acquire - a full set
    of slots, an unwritable lock directory, or a platform without fcntl - falls
    through to running the block ungated, because the work inside matters more than
    the scheduling around it.

    Arguments:
        name: [str] Gate name, used for the lock file names. Separate names are
            separate semaphores.
        slots: [int] Number of concurrent holders allowed. 0 or less disables the
            gate entirely.

    Keyword arguments:
        slot_dir: [str] Directory holding the lock files. DEFAULT_SLOT_DIR if None.
        timeout: [float] Seconds to wait for a slot before proceeding ungated.
        poll_interval: [float] Seconds between sweeps of the slot files.

    Yield:
        [int | None] Index of the slot held, or None when running ungated.
    """

    if slots is None or slots <= 0:
        yield None
        return

    # Windows has no fcntl. Stations there run one camera, so there is nothing to
    # serialize; degrade to ungated rather than making the import a hard dependency.
    try:
        import fcntl

    except ImportError:
        yield None
        return

    if slot_dir is None:
        slot_dir = DEFAULT_SLOT_DIR

    handle = None
    index = None
    t_start = time.time()
    waited_logged = False

    try:
        try:
            # exist_ok is Python 3 only, and the tree targets 2.7 as well
            if not os.path.isdir(slot_dir):
                os.makedirs(slot_dir)

        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        while True:

            for i in range(slots):

                path = os.path.join(slot_dir, "{:s}.{:d}.lock".format(name, i))

                try:
                    f = open(path, "a+")

                except IOError:
                    continue

                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                except (IOError, OSError):
                    f.close()
                    continue

                handle = f
                index = i
                break

            if handle is not None:
                break

            waited = time.time() - t_start

            if waited >= timeout:
                log.warning("%s gate: no slot free after %.0f s of %d - running "
                            "ungated", name, waited, slots)
                break

            if not waited_logged:
                log.info("%s gate: all %d slots busy, waiting%s", name, slots,
                         rssSuffix())
                waited_logged = True

            time.sleep(poll_interval)

    except Exception as e:
        # An unusable lock directory must not cost the night its products
        log.warning("%s gate: unavailable (%s) - running ungated", name, e)

    if handle is not None:
        log.info("%s gate: slot %d/%d acquired after %.0f s%s", name, index + 1,
                 slots, time.time() - t_start, rssSuffix())

    try:
        yield index

    finally:
        if handle is not None:
            log.info("%s gate: slot %d/%d released after %.0f s%s", name, index + 1,
                     slots, time.time() - t_start, rssSuffix())
            try:
                # Closing the descriptor drops the flock; doing it explicitly keeps
                # the release at a known point rather than at garbage collection
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

            except Exception:
                pass

            try:
                handle.close()

            except Exception:
                pass
