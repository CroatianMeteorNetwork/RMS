""" Tests for StartCapture startup robustness: archive-missing marker, EACCES lock fallback, and QueuedPool config. """

from __future__ import print_function, division, absolute_import

import ast
import builtins
import errno
import json
import logging
import os
from types import SimpleNamespace

import pytest

from RMS.Formats.ObservationSummary import OBSERVATION_SUMMARY_NAME_JSON
from RMS.Misc import getRMSStyleFileName


def _makeStartCaptureConfig(tmp_path, max_attempts=3, reprocess_missing=True):
    """ Build a minimal Config stand-in rooted in tmp_path. """

    data_dir = str(tmp_path)
    captured_dir = "CapturedFiles"
    archived_dir = "ArchivedFiles"
    os.makedirs(os.path.join(data_dir, captured_dir), exist_ok=True)
    os.makedirs(os.path.join(data_dir, archived_dir), exist_ok=True)

    return SimpleNamespace(
        stationID="XX0001",
        data_dir=data_dir,
        captured_dir=captured_dir,
        archived_dir=archived_dir,
        processing_status_file=".processing_status.json",
        auto_reprocess_max_attempts=max_attempts,
        reprocess_if_archive_missing=reprocess_missing,
        external_script_run=False,
        auto_reprocess_external_script_run=False,
        continuous_capture=True,
        latitude=43.0,
        longitude=-81.0,
        elevation=250.0,
    )


def _createProcessedNightWithoutArchive(config, subdir_name="XX0001_20261127_000000_000000"):
    """ Create a captured night folder that looks processed (final obs summary present) with no ArchivedFiles dir. """

    night_dir = os.path.join(config.data_dir, config.captured_dir, subdir_name)
    os.makedirs(night_dir, exist_ok=True)

    # One FF file so processIncompleteCaptures considers the folder
    ff_path = os.path.join(night_dir, "FF_XX0001_20261127_000000_000_0000000.fits")
    with open(ff_path, "wb") as f:
        f.write(b"0")

    # Final observation summary marks the night as NIGHT_PROCESSED
    final_summary = getRMSStyleFileName(night_dir, OBSERVATION_SUMMARY_NAME_JSON)
    with open(final_summary, "w") as f:
        json.dump({"stationID": "XX0001"}, f)

    return night_dir


def testArchiveMissingNightReprocessedOnce(tmp_path, monkeypatch):
    """ A processed night whose archive stays absent must be reprocessed only once, not on every start. """

    import RMS.StartCapture as sc

    config = _makeStartCaptureConfig(tmp_path, max_attempts=3, reprocess_missing=True)
    _createProcessedNightWithoutArchive(config)

    calls = []

    def _fakeProcessNight(captured_dir_path, cfg):
        calls.append(captured_dir_path)
        return (None, None, None, None, None)

    monkeypatch.setattr(sc, "log", logging.getLogger("test_startcapture"), raising=False)
    monkeypatch.setattr(sc, "processNight", _fakeProcessNight)

    # Simulate three starts where the archive is never present (e.g. quota purged it)
    for _ in range(3):
        sc.processIncompleteCaptures(config, upload_manager=None)

    assert len(calls) == 1


def testArchiveMissingNightRetriesFailuresUpToMaxAttempts(tmp_path, monkeypatch):
    """ Failing archive-missing reprocess attempts must still retry up to auto_reprocess_max_attempts. """

    import RMS.StartCapture as sc

    config = _makeStartCaptureConfig(tmp_path, max_attempts=2, reprocess_missing=True)
    _createProcessedNightWithoutArchive(config)

    calls = []

    def _failingProcessNight(captured_dir_path, cfg):
        calls.append(captured_dir_path)
        raise RuntimeError("transient error")

    monkeypatch.setattr(sc, "log", logging.getLogger("test_startcapture"), raising=False)
    monkeypatch.setattr(sc, "processNight", _failingProcessNight)

    for _ in range(4):
        sc.processIncompleteCaptures(config, upload_manager=None)

    assert len(calls) == 2


@pytest.mark.skipif(os.name != "posix", reason="POSIX fcntl flock test")
def testStationLockFallsBackToReadOnlyOnEacces(tmp_path, monkeypatch):
    """ _takeStationLock must acquire a stale lock via read-only fd on EACCES and still reject when held. """

    import fcntl
    import tempfile
    import RMS.StartCapture as sc

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    station_id = "XX9999"
    lock_path = os.path.join(str(tmp_path), "rms_startcapture_{:s}.lock".format(station_id))
    with open(lock_path, "w") as f:
        f.write("12345")

    real_open = builtins.open

    def _eaccesOnAppend(path, mode="r", *args, **kwargs):
        if (os.path.abspath(str(path)) == os.path.abspath(lock_path)) and (mode == "a+"):
            raise PermissionError(errno.EACCES, "Permission denied", path)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _eaccesOnAppend)

    # First acquisition succeeds through the read-only fallback
    lock_fh = sc._takeStationLock(station_id)
    try:
        assert lock_fh.mode == "r"

        # A second attempt while the exclusive flock is held must still refuse and exit
        with pytest.raises(SystemExit) as exc_info:
            sc._takeStationLock(station_id)
        assert exc_info.value.code == 1
    finally:
        fcntl.flock(lock_fh, fcntl.LOCK_UN)
        lock_fh.close()


@pytest.mark.parametrize("rel_path", [
    "RMS/StartCapture.py",
    "RMS/DetectStarsAndMeteors.py",
    "RMS/ExtractStars.py",
])
def testQueuedPoolCallsProvideConfig(rel_path):
    """ Every QueuedPool call in capture/detection/star-extraction modules must provide config=. """

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_path = os.path.join(repo_root, rel_path)
    with open(src_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=rel_path)

    pool_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else (
                func.attr if isinstance(func, ast.Attribute) else None
            )
            if name == "QueuedPool":
                kw_names = [kw.arg for kw in node.keywords if kw.arg is not None]
                pool_calls.append((node.lineno, kw_names))

    assert len(pool_calls) > 0, "Expected at least one QueuedPool call in {}".format(rel_path)
    for lineno, kw_names in pool_calls:
        assert "config" in kw_names, "{}:{} QueuedPool() call is missing config=".format(rel_path, lineno)
