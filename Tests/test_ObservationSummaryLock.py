""" Tests of the locking of the working observation summary (RMS.Formats.ObservationSummary.saveObservationSummaryDict)
    on platforms with and without fcntl, and of its temporary files. """

from __future__ import print_function, division, absolute_import

import json
import os

import RMS.Formats.ObservationSummary as osum


class _FakeMsvcrt(object):
    """ Stand-in for the Windows msvcrt module which records the lock calls. """

    LK_NBLCK = 2
    LK_UNLCK = 0

    def __init__(self):
        self.calls = []

    def locking(self, fd, mode, nbytes):
        self.calls.append((mode, nbytes))


def test_summary_is_locked_without_fcntl(tmp_path, monkeypatch):
    """ Without fcntl (Windows) the summary is still locked, with msvcrt, and unlocked afterwards. """

    fake = _FakeMsvcrt()
    monkeypatch.setattr(osum, 'fcntl', None)
    monkeypatch.setattr(osum, 'msvcrt', fake)

    night_dir = str(tmp_path/'XX0001_20251225_010203_000000')
    os.makedirs(night_dir)
    osum.saveObservationSummaryDict({'night_data_dir': night_dir, 'media_backend': 'gst'}, night_dir)

    assert [mode for mode, _ in fake.calls] == [fake.LK_NBLCK, fake.LK_UNLCK]


def test_each_writer_uses_its_own_temporary_file(tmp_path, monkeypatch):
    """ The temporary file is unique to the writer, so two writers never write into or replace the same one. """

    replaced = []
    real_replace = os.replace
    monkeypatch.setattr(osum.os, 'replace', lambda src, dst: (replaced.append(src), real_replace(src, dst)))

    night_dir = str(tmp_path/'XX0001_20251225_010203_000000')
    os.makedirs(night_dir)
    osum.saveObservationSummaryDict({'night_data_dir': night_dir, 'a': 1}, night_dir)

    assert len(replaced) == 1
    assert str(os.getpid()) in os.path.basename(replaced[0])
    assert not os.path.exists(replaced[0])


def test_writers_merge_their_keys(tmp_path):
    """ Two saves with different keys both end up in the summary. """

    night_dir = str(tmp_path/'XX0001_20251225_010203_000000')
    os.makedirs(night_dir)
    osum.saveObservationSummaryDict({'night_data_dir': night_dir, 'a': 1}, night_dir)
    osum.saveObservationSummaryDict({'night_data_dir': night_dir, 'b': 2}, night_dir)

    path = osum.getRMSStyleFileName(night_dir, osum.OBSERVATION_SUMMARY_WORKING_NAME_JSON)
    with open(os.path.join(night_dir, path)) as f:
        saved = json.load(f)
    assert (saved['a'], saved['b']) == (1, 2)
