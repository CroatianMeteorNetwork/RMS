""" Guards against Python 3.14 forkserver/spawn regressions.

From Python 3.14 the default multiprocessing start method on Linux changes from 'fork' to
'forkserver'. Under 'fork' a child inherits the parent's memory by copy, so objects that
cross a process boundary are never pickled and these code paths are never exercised. Under
'forkserver'/'spawn' the object (a Process's self, or a Pool initializer) is pickled and
sent to a fresh interpreter.

These tests exercise the pickling paths under the 'spawn' context, which is the
picklability-equivalent of 'forkserver', so the regressions are caught on today's Python
instead of only surfacing on 3.14. Note that multiprocessing primitives (Value/Event/Queue)
can only be pickled through the spawn machinery itself, so faithful tests must spawn a real
process rather than calling pickle.dumps() directly.
"""

import multiprocessing

import pytest

from RMS.Misc import setMultiprocessingStartMethod
from RMS.QueuedPool import QueuedPool


def _dummy_worker(x):
    """ Module-level worker so it is importable by reference in spawned processes. """
    return x*2


def _exercise_pool_in_child(pool, result_queue):
    """ Runs in a spawned child: the pool arrived via QueuedPool.__getstate__. Verify the
        non-picklable handles were dropped and the picklable proxies/primitives still work.
    """
    result_queue.put({
        'manager_is_none': pool.manager is None,
        'pool_is_none': pool.pool is None,
        'cores': pool.cores.value(),        # SafeValue -> mp.Value + mp.Lock across spawn
    })
    pool.input_queue.put(['hello'])         # Manager Queue proxy reconnects on the child side
    pool.kill_workers.set()                 # plain mp.Event shared across processes


def test_setMultiprocessingStartMethod_returns_valid_method():
    method = setMultiprocessingStartMethod()
    assert method in multiprocessing.get_all_start_methods()

    # forkserver is preferred where available (Linux/macOS)
    if 'forkserver' in multiprocessing.get_all_start_methods():
        assert method == 'forkserver'


def test_queuedpool_getstate_is_clean():
    """ __getstate__ must drop the non-picklable SyncManager and Pool handles. """
    pool = QueuedPool(_dummy_worker, cores=1, backup_dir=None)
    try:
        state = pool.__getstate__()
        assert state['manager'] is None
        assert state['pool'] is None
        # The logger is reduced to its name (a str) or None - never a Logger object
        assert state['log'] is None or isinstance(state['log'], str)
    finally:
        pool.shutdownManager()


def test_queuedpool_roundtrips_through_spawn():
    """ A QueuedPool built under the spawn context must survive being sent to a child. """

    # Force the spawn context so the Manager/Value/Queue inside QueuedPool and the child
    # process all share one start method (the forkserver-equivalent for picklability).
    multiprocessing.set_start_method('spawn', force=True)

    pool = QueuedPool(_dummy_worker, cores=1, backup_dir=None)
    try:
        result_queue = multiprocessing.Queue()
        proc = multiprocessing.Process(target=_exercise_pool_in_child, args=(pool, result_queue))
        proc.start()

        info = result_queue.get(timeout=60)
        proc.join(timeout=60)

        assert proc.exitcode == 0
        assert info['manager_is_none'] is True   # __getstate__ dropped the SyncManager
        assert info['pool_is_none'] is True       # __getstate__ dropped the Pool handle
        assert info['cores'] == 1                 # SafeValue survived the spawn boundary

        # The child put an item on the Manager-backed input queue; the parent should see it
        assert pool.input_queue.get(timeout=30) == ['hello']
        assert pool.kill_workers.is_set()         # Event state is shared across processes
    finally:
        pool.shutdownManager()


def test_uploadmanager_getstate_drops_manager():
    """ UploadManager.__getstate__ must drop the SyncManager but keep the queue proxies. """
    pytest.importorskip("paramiko")  # UploadManager import chain
    from RMS.UploadManager import UploadManager

    inst = object.__new__(UploadManager)
    inst.__dict__ = {'_mgr': object(), 'file_queue': 'QUEUE_PROXY', 'config': 'CONFIG'}

    state = inst.__getstate__()
    assert state['_mgr'] is None
    assert state['file_queue'] == 'QUEUE_PROXY'   # picklable proxy retained
    assert state['config'] == 'CONFIG'


def test_eventmonitor_getstate_drops_connections():
    """ EventMonitor.__getstate__ must drop the sqlite connections. """
    pytest.importorskip("paramiko")  # EventMonitor import chain
    from RMS.EventMonitor import EventMonitor

    inst = object.__new__(EventMonitor)
    inst.__dict__ = {'conn': object(), 'db_conn': object(), 'event_monitor_db_path': '/tmp/x.db'}

    state = inst.__getstate__()
    assert state['conn'] is None
    assert state['db_conn'] is None
    assert state['event_monitor_db_path'] == '/tmp/x.db'   # path retained for reopening in run()
