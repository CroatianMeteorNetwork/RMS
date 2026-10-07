""" Tests of the lifetime of the logging listener process. """

from __future__ import print_function, division, absolute_import

import glob
import multiprocessing
import os
import signal
import sys
import time

import pytest

import RMS.ConfigReader as cr
from RMS.Logger import LoggingManager, getLogger


REPO_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.config')


def _loggingProcess(data_dir, pid_queue, exit_mode):
    """ Start logging, log a few records, report the PID of the listener, then end as given. """

    config = cr.parse(REPO_CONFIG)
    config.data_dir = data_dir
    config.log_dir = 'logs'

    manager = LoggingManager()
    manager.initLogging(config, 'listener_test_')

    log = getLogger("logger")
    for i in range(5):
        log.info("record {:d}".format(i))

    pid_queue.put(manager.listener_process.pid)

    if exit_mode == 'normal':
        return

    # Wait to be killed
    time.sleep(60)


def _listenerAlive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False

    # A zombie child of the test process is not running anymore
    try:
        with open('/proc/{:d}/stat'.format(pid)) as f:
            return f.read().split()[2] != 'Z'
    except IOError:
        return False


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason="Uses Linux process states")
@pytest.mark.parametrize('start_method', ['fork', 'forkserver'])
@pytest.mark.parametrize('exit_mode', ['normal', signal.SIGTERM, signal.SIGKILL])
def test_listener_stops_with_the_logging_process(tmp_path, start_method, exit_mode):

    ctx = multiprocessing.get_context(start_method)
    pid_queue = ctx.Queue()

    proc = ctx.Process(target=_loggingProcess, args=(str(tmp_path), pid_queue,
                                                      'normal' if exit_mode == 'normal' else 'wait'))
    proc.start()

    listener_pid = pid_queue.get(timeout=60)

    # Records are sent to the listener by a feeder thread of the queue. Those which were not sent yet when the
    #   process is killed are lost, so the process is killed after they were sent
    if exit_mode != 'normal':
        time.sleep(1)
        os.kill(proc.pid, exit_mode)

    proc.join(30)

    # The listener stops shortly after the logging process is gone
    t0 = time.time()
    while _listenerAlive(listener_pid) and (time.time() - t0 < 10):
        time.sleep(0.2)

    assert not _listenerAlive(listener_pid)

    # The records logged before were all written
    log_text = "".join(open(path).read() for path in glob.glob(os.path.join(str(tmp_path), 'logs', '*.log')))
    for i in range(5):
        assert "record {:d}".format(i) in log_text


def _killedDuringListenerStart(data_dir, pid_queue):
    """ Start logging and get killed at once, while the listener is still starting. """

    config = cr.parse(REPO_CONFIG)
    config.data_dir = data_dir
    config.log_dir = 'logs'

    manager = LoggingManager()
    manager.initLogging(config, 'listener_test_')

    pid_queue.put(manager.listener_process.pid)
    os.kill(os.getpid(), signal.SIGKILL)


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason="Uses Linux process states")
@pytest.mark.parametrize('start_method', ['fork', 'forkserver'])
def test_listener_stops_if_the_logging_process_dies_during_its_start(tmp_path, start_method):

    # The PID is written synchronously, as the process is killed right after sending it
    ctx = multiprocessing.get_context(start_method)
    pid_queue = ctx.SimpleQueue()

    proc = ctx.Process(target=_killedDuringListenerStart, args=(str(tmp_path), pid_queue))
    proc.start()

    listener_pid = pid_queue.get()

    # The dead process stays a zombie (it is not joined), which must not keep the listener alive
    t0 = time.time()
    while _listenerAlive(listener_pid) and (time.time() - t0 < 10):
        time.sleep(0.2)

    alive = _listenerAlive(listener_pid)
    proc.join(30)

    assert not alive
