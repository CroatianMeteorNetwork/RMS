""" The upload manager started with watch_parent stops when the process which started it is gone. """

import multiprocessing
import os
import signal
import time

import pytest

import RMS.ConfigReader as cr
from RMS.UploadManager import UploadManager


def _parentWithUploadManager(data_dir, pid_queue):
    config = cr.Config()
    config.data_dir = data_dir

    upload_manager = UploadManager(config, watch_parent=True)
    upload_manager.start()
    pid_queue.put((upload_manager.pid, upload_manager._mgr._process.pid))

    time.sleep(60)


def _alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False

    # A process which ended but was not reaped yet is gone too
    try:
        with open('/proc/{:d}/stat'.format(pid)) as f:
            return f.read().split(')')[-1].split()[0] != 'Z'
    except IOError:
        return False


@pytest.mark.skipif(not os.path.isdir('/proc'), reason="needs /proc")
def test_upload_manager_stops_when_its_parent_is_killed(tmp_path):

    ctx = multiprocessing.get_context('fork')
    pid_queue = ctx.Queue()

    parent = ctx.Process(target=_parentWithUploadManager, args=(str(tmp_path), pid_queue))
    parent.start()
    upload_pid, manager_pid = pid_queue.get(timeout=30)

    os.kill(parent.pid, signal.SIGKILL)
    parent.join()

    # The upload manager and the server process of its queue end within a few seconds
    deadline = time.time() + 10
    while (time.time() < deadline) and (_alive(upload_pid) or _alive(manager_pid)):
        time.sleep(0.2)

    assert not _alive(upload_pid)
    assert not _alive(manager_pid)
