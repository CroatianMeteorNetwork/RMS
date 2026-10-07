# RMS shares data between processes in ways which need the 'fork' start method of multiprocessing, e.g. numpy
#   views of shared memory arrays (capture and compression), and pools with bound methods (QueuedPool). Python
#   3.14 changed the default start method on Linux to 'forkserver', with which the processes would silently
#   work on copies of the shared arrays. 'fork' is therefore set on Linux, unless the program already chose a
#   start method.
import multiprocessing as _multiprocessing
import sys as _sys

if _sys.platform.startswith('linux'):
    try:
        _multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass
