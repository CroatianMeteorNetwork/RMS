""" RMS needs the 'fork' start method of multiprocessing on Linux (Python 3.14 changed the default to
    'forkserver'), and its processes which don't depend on it can be started with any start method.
"""

import os
import subprocess
import sys

import pytest


REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(code, *args):
    """ Run the code in a new Python interpreter (so the start method isn't set yet), return its output. """

    result = subprocess.run([sys.executable, '-c', code] + list(args), cwd=REPO_DIR, capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stderr

    return result.stdout.strip().splitlines()[-1]


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason="the start method is only set on Linux")
def test_importing_rms_sets_fork():

    assert _run("import multiprocessing, RMS; print(multiprocessing.get_start_method())") == 'fork'


def test_start_method_chosen_by_the_program_is_kept():

    code = ("import multiprocessing; multiprocessing.set_start_method('spawn'); import RMS; "
            "print(multiprocessing.get_start_method())")

    assert _run(code) == 'spawn'


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason="forkserver is not available everywhere")
def test_upload_manager_starts_with_forkserver(tmp_path):

    code = """
import multiprocessing, sys, time
if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    import RMS.ConfigReader as cr
    from RMS.UploadManager import UploadManager
    config = cr.Config()
    config.data_dir = sys.argv[1]
    upload_manager = UploadManager(config)
    upload_manager.start()
    time.sleep(2)
    alive = upload_manager.is_alive()
    upload_manager.stop(timeout=20)
    print(alive, upload_manager.exitcode)
"""

    assert _run(code, str(tmp_path)) == 'True 0'


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason="forkserver is not available everywhere")
def test_event_monitor_can_be_pickled_for_forkserver(tmp_path):

    code = """
import multiprocessing, sys
from multiprocessing import context, reduction
if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    import RMS.ConfigReader as cr
    from RMS.EventMonitor import EventMonitor
    config = cr.Config()
    config.data_dir = sys.argv[1]
    event_monitor = EventMonitor(config)
    context.set_spawning_popen(object())
    reduction.ForkingPickler.dumps(event_monitor)
    context.set_spawning_popen(None)
    event_monitor.db_conn.close()
    print('pickled')
"""

    assert _run(code, str(tmp_path)) == 'pickled'
