""" Tests for reseeding the working observation summary from the finalized JSON on reprocess. """

from __future__ import print_function, division, absolute_import

import json
import os

from RMS.Formats.ObservationSummary import (
    OBSERVATION_SUMMARY_NAME_JSON,
    OBSERVATION_SUMMARY_SESSION_KEYS,
    OBSERVATION_SUMMARY_WORKING_NAME_JSON,
    getObservationSummaryDict,
)
from RMS.Misc import getRMSStyleFileName


def testObservationSummaryReseededFromFinalJson(tmp_path):
    """ Recreating a missing working summary must carry over capture-time session keys only. """

    night_dir = str(tmp_path/"XX0001_20261127_000000_000000")
    os.makedirs(night_dir, exist_ok=True)

    final_data = {
        "night_data_dir": "/old/moved/path",
        "start_time": "2026-11-27T00:00:00",
        "duration_from_start_of_observation": 36000.0,
        "stationID": "XX0001",
        "hardware_version": "Pi4",
        "commit_date": "20261127_120000",
        "commit_hash": "abcdef012345",
        "storage_total_gb": 128.0,
        "storage_used_gb": 40.0,
        "storage_free_gb": 88.0,
        "captured_directories": 12,
        "camera_information": "IMX291",
        "camera_firmware_version": "V4.0",
        "camera_firmware_build_date": "2024-01-01",
        "media_backend": "gst",
        "dropped_frames": 3,
        # Recomputed keys that must NOT be carried over from the earlier run
        "photometry_good": True,
        "astrometry_good": True,
        "total_fits": 1234,
    }

    final_path = getRMSStyleFileName(night_dir, OBSERVATION_SUMMARY_NAME_JSON)
    with open(final_path, "w") as f:
        json.dump(final_data, f)

    working_path = getRMSStyleFileName(night_dir, OBSERVATION_SUMMARY_WORKING_NAME_JSON)
    assert not os.path.exists(working_path)

    seeded = getObservationSummaryDict(night_dir)

    # All capture-time session keys survived
    for key in OBSERVATION_SUMMARY_SESSION_KEYS:
        assert seeded[key] == final_data[key]

    # Recomputed keys from the earlier run were dropped
    assert "photometry_good" not in seeded
    assert "astrometry_good" not in seeded
    assert "total_fits" not in seeded

    # Directory path updated to the current absolute path and persisted to the working JSON
    assert seeded["night_data_dir"] == os.path.abspath(night_dir)
    assert os.path.isfile(working_path)
    with open(working_path, "r") as f:
        saved = json.load(f)
    assert saved["stationID"] == "XX0001"
    assert "photometry_good" not in saved


def testObservationSummaryReseedHandlesCorruptFinalJson(tmp_path):
    """ A corrupt or non-dict final JSON must fall back cleanly to a fresh working summary. """

    night_dir = str(tmp_path/"XX0001_20261127_010000_000000")
    os.makedirs(night_dir, exist_ok=True)

    final_path = getRMSStyleFileName(night_dir, OBSERVATION_SUMMARY_NAME_JSON)
    with open(final_path, "w") as f:
        f.write("{not valid json")

    fresh = getObservationSummaryDict(night_dir)
    assert fresh == {"night_data_dir": os.path.abspath(night_dir)}
