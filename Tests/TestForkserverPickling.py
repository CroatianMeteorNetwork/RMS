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


def _logcheck_worker(x):
    """ Worker that reports the root logger handlers present in its process. Used to confirm
        the worker re-attached logging (a QueueHandler) under spawn.
    """
    import logging
    return [type(h).__name__ for h in logging.getLogger().handlers]


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
        # The logging queue must be preserved so workers can re-attach logging in _workerFunc
        assert 'logging_queue' in state
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


def test_queuedpool_worker_reattaches_logging():
    """ A QueuedPool worker must re-attach a QueueHandler under spawn (handlers are not
        inherited), and the real Pool(initializer=self._workerFunc) path must pickle the
        whole pool successfully.
    """
    import time

    multiprocessing.set_start_method('spawn', force=True)

    pool = QueuedPool(_logcheck_worker, cores=1, backup_dir=None, print_state=False)
    # Simulate an initialized logging queue (getLoggingQueue() is None without initLogging)
    pool.logging_queue = multiprocessing.Queue()
    try:
        pool.addJob([1])
        pool.startPool()

        deadline = time.time() + 30
        while not pool.allDone() and time.time() < deadline:
            time.sleep(0.1)

        pool.closePool()
        results = pool.getResults()

        assert results, "worker produced no result"
        assert 'QueueHandler' in results[0]   # logging was re-attached in the worker
    finally:
        pool.shutdownManager()


def test_getLoggingQueue_reads_root_queuehandler():
    """ getLoggingQueue() must find the queue from the root logger's QueueHandler even when
        the global LoggingManager was never used. StartCapture/Reprocess create their own
        LoggingManager() instance, so the global manager's queue stays None; without this
        fallback every child process would receive None and silently stop logging under
        forkserver/spawn.
    """
    import logging
    import logging.handlers
    from RMS.Logger import getLoggingQueue

    root = logging.getLogger()
    saved = root.handlers[:]
    try:
        q = multiprocessing.Queue()
        root.handlers = [logging.handlers.QueueHandler(q)]
        assert getLoggingQueue() is q
    finally:
        root.handlers = saved


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
