""" Regression tests for defects found in the second review of SkyFit2.

    The tests that need Qt run offscreen and are skipped cleanly when Qt or pyqtgraph is not available.
"""

from __future__ import print_function, division, absolute_import

import copy
import functools
import os
import shutil
import types

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip("pyqtgraph")

try:
    import Utils.SkyFit2 as SF
except SystemExit:
    pytest.skip("SkyFit2 could not import its Qt dependencies", allow_module_level=True)

import RMS.ConfigReader as cr


REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_CONFIG = os.path.join(REPO_DIR, ".config")


###################################################################################################
# CONFIG WRITING
###################################################################################################

def _fakePlateTool(config_path, **overrides):
    """ Build the minimal object that the config writer and the File Manager config save read.

    Arguments:
        config_path: [str] Path of the "loaded" config file.

    Keyword arguments:
        overrides: Star detection override values to use instead of the defaults.

    Return:
        pt: [SimpleNamespace] Stand-in for PlateTool.
    """

    pt = types.SimpleNamespace()
    pt.config = cr.parse(config_path)
    pt.override_intensity_threshold = overrides.get('intensity_threshold', 41)
    pt.override_neighborhood_size = overrides.get('neighborhood_size', 7)
    pt.override_max_stars = overrides.get('max_stars', 1200)
    pt.override_segment_radius = overrides.get('segment_radius', 5)
    pt.override_max_feature_ratio = overrides.get('max_feature_ratio', 0.77)
    pt.override_roundness_threshold = overrides.get('roundness_threshold', 0.33)
    pt.tuned_cat_lim_mag = overrides.get('tuned_cat_lim_mag', 6.3)
    pt.platepar_modified = False
    pt.configMaxStars = functools.partial(SF.PlateTool.configMaxStars, pt)
    pt._writeStarDetectionConfig = functools.partial(SF.PlateTool._writeStarDetectionConfig, pt)
    pt._updateConfigSaveButtonState = lambda: None
    pt.updateFileManagerButton = lambda: None

    return pt


@pytest.fixture
def quietMessages(monkeypatch):
    """ Record the SkyFit2 message boxes instead of showing them. """

    messages = []
    monkeypatch.setattr(SF, "qmessagebox", lambda *a, **k: messages.append(k))

    return messages


@pytest.mark.parametrize("spelling", ["{k}: {v}", "{k}:{v}", "{k}={v}", "{k} = {v}", "{k} : {v}  ; note"])
def testWriteConfigAnyKeySpelling(tmp_path, quietMessages, spelling):
    """ Every key spelling configparser accepts is updated in place, never appended a second time. """

    # Respell the StarExtraction keys of the template config
    keys = ["intensity_threshold", "neighborhood_size", "max_stars", "segment_radius", "max_feature_ratio",
            "roundness_threshold"]
    lines = []
    section = None
    with open(TEMPLATE_CONFIG) as f:
        for line in f:
            if line.strip().startswith('['):
                section = line.strip()
            key = line.split(':')[0].strip()
            if section == "[StarExtraction]" and key in keys and not line.lstrip().startswith(';'):
                value = line.split(':', 1)[1].split(';')[0].strip()
                line = spelling.format(k=key, v=value) + "\n"
            lines.append(line)

    cfg_path = str(tmp_path / ".config")
    with open(cfg_path, 'w') as f:
        f.writelines(lines)

    pt = _fakePlateTool(cfg_path)
    pt._writeStarDetectionConfig(cfg_path, catalog_mag_limit=6.3)

    # The strict parser RMS starts with must accept the result, with the new values
    config = cr.parse(cfg_path, strict=True)
    assert config.intensity_threshold == 41
    assert config.max_stars == 1200
    assert config.max_feature_ratio == pytest.approx(0.77)
    assert config.catalog_mag_limit == pytest.approx(6.3)

    # The spelling and the inline comment are kept
    with open(cfg_path) as f:
        text = f.read()
    assert spelling.format(k="intensity_threshold", v="41") in text


def testUpdateConfigLinesAddsMissing():
    """ Missing keys go at the end of their section, missing sections at the end of the file. """

    lines = ["[StarExtraction]\n", "max_stars=100\n", "\n", "[Other]\n", "x: 1"]
    new = SF.updateConfigLines(lines, {"StarExtraction": {"max_stars": "5", "segment_radius": "4"},
                                       "Calibration": {"catalog_mag_limit": "6.0"}})

    assert new[:3] == ["[StarExtraction]\n", "max_stars=5\n", "segment_radius: 4\n"]
    assert "x: 1\n" in new
    assert new[-2:] == ["\n[Calibration]\n", "catalog_mag_limit: 6.0\n"]


def testSaveConfigKeepsTargetStation(tmp_path, quietMessages):
    """ Saving the config to another station patches that station's config instead of replacing it. """

    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"
    src_dir.mkdir()
    dst_dir.mkdir()

    src_path = str(src_dir / ".config")
    shutil.copy(TEMPLATE_CONFIG, src_path)

    # The target config has its own station ID and a line the source does not have
    with open(TEMPLATE_CONFIG) as f:
        text = f.read()
    text = text.replace("stationID: XX0001", "stationID: ZZ0009")
    text = text.replace("[StarExtraction]", "[StarExtraction]\n; target only comment line")
    dst_path = str(dst_dir / ".config")
    with open(dst_path, 'w') as f:
        f.write(text)

    pt = _fakePlateTool(src_path)
    dialog = types.SimpleNamespace(plate_tool=pt, _refreshAll=lambda: None)
    SF.CalibrationFilesDialog._saveFile(dialog, "Config", [str(dst_dir)])

    # The target keeps its identity and gets the new detection settings
    with open(dst_path) as f:
        new_text = f.read()
    assert "stationID: ZZ0009" in new_text
    assert "; target only comment line" in new_text
    assert cr.parse(dst_path).intensity_threshold == 41

    # The backup is of the original target, not of a copy of the source
    backups = [f for f in os.listdir(str(dst_dir)) if ".bak." in f]
    assert len(backups) == 1
    with open(str(dst_dir / backups[0])) as f:
        assert f.read() == text

    # The loaded config was not written, so its in-memory values stay as loaded
    assert pt.config.intensity_threshold == cr.parse(src_path).intensity_threshold


def testSaveConfigFailureNotReported(tmp_path, quietMessages, monkeypatch):
    """ A failed config write is reported as failed and leaves the in-memory config alone. """

    cfg_path = str(tmp_path / ".config")
    shutil.copy(TEMPLATE_CONFIG, cfg_path)
    pt = _fakePlateTool(cfg_path)
    orig_threshold = pt.config.intensity_threshold

    def failingWrite(*args, **kwargs):
        raise IOError("disk full")

    monkeypatch.setattr(SF, "updateConfigLines", failingWrite)

    dialog = types.SimpleNamespace(plate_tool=pt, _refreshAll=lambda: None)
    SF.CalibrationFilesDialog._saveFile(dialog, "Config", [str(tmp_path)])

    assert pt.config.intensity_threshold == orig_threshold
    assert any(m.get("message_type") == "error" and "failed" in m.get("message", "") for m in quietMessages)
    assert not any(m.get("title") == "Settings Saved" for m in quietMessages)
