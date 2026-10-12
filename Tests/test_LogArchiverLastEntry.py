""" The time of the last entry of a log file (Utils.LogArchiver.getTimeOfLastLogEntry), also when the file ends with
lines which are not entries (e.g. the continuation of a multi-line warning). """

import datetime
import os
import types

from Utils.LogArchiver import getTimeOfLastLogEntry


def _config(tmp_path):
    """ A config with the log directory in a temporary directory. """

    return types.SimpleNamespace(data_dir=str(tmp_path), log_dir='logs')


def test_last_entry_time(tmp_path):
    """ A log which ends with the continuation of a multi-line warning gives the time of the warning. """

    # Two entries, the second one with a continuation line
    os.makedirs(tmp_path/'logs')
    with open(tmp_path/'logs'/'a.log', 'w') as f:
        f.write('2026/10/12 01:39:30-1-INFO-Module-line:1 - first\n')
        f.write('2026/10/12 01:39:34-1-WARNING-Logger-line:105 - /RMS/ExtractStars.py:702: RuntimeWarning: maxfev\n')
        f.write('  popt, ier = opt.leastsq(twoDGaussianResidual, np.array(initial_guess, dtype=np.float64),\n')

    # The time of the last line which is an entry
    assert getTimeOfLastLogEntry(_config(tmp_path), 'a.log') == datetime.datetime(2026, 10, 12, 1, 39, 34)


def test_no_entries(tmp_path):
    """ A log without entries gives the default time (2000-01-01). """

    # A log with a line which is no entry
    os.makedirs(tmp_path/'logs')
    with open(tmp_path/'logs'/'b.log', 'w') as f:
        f.write('no entries here\n')

    # The default time
    assert getTimeOfLastLogEntry(_config(tmp_path), 'b.log') == datetime.datetime(2000, 1, 1, 0, 0)
