""" Tests of the SkyFit2 config save (File Manager > Config > Save, CalibrationFilesDialog._saveFile) and of
the config writer behind it (PlateTool._writeStarDetectionConfig, updateConfigLines).

Saving the config into a folder that already has one (e.g. the station folder, from an archived night) must
only patch the star detection keys of that config and keep everything else (station ID, coordinates, camera
and upload settings, comments). Every key spelling configparser accepts must be updated in place, never
appended a second time, and the line endings and encoding of the file must stay as they are. A write that
fails, or a config that is not writable, must be reported as failed and leave the config and the in-memory
values as they were.

The tests run without a display and without station data, on the template .config of the repository.
"""

from __future__ import print_function, division, absolute_import

import functools
import os
import shutil
import stat
import types

import pytest

import RMS.ConfigReader as cr


REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_CONFIG = os.path.join(REPO_DIR, ".config")

# The star detection keys the config save writes
STAR_KEYS = ["intensity_threshold", "neighborhood_size", "max_stars", "segment_radius", "max_feature_ratio",
             "roundness_threshold"]


@pytest.fixture(scope="module")
def SF():
    """ The SkyFit2 module, skipped cleanly when its Qt dependencies are not available. """

    # No display is needed
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    pytest.importorskip("pyqtgraph")

    try:
        import Utils.SkyFit2 as skyfit2
    except SystemExit:
        pytest.skip("SkyFit2 could not import its Qt dependencies")

    return skyfit2


@pytest.fixture
def quietMessages(SF, monkeypatch):
    """ Record the SkyFit2 message boxes instead of showing them. """

    # The keyword arguments of every message box
    messages = []
    monkeypatch.setattr(SF, "qmessagebox", lambda *args, **kwargs: messages.append(kwargs))

    return messages


def fakePlateTool(SF, config_path, **overrides):
    """ Build the minimal object that the config writer and the File Manager config save read.

    Arguments:
        SF: [module] The SkyFit2 module.
        config_path: [str] Path of the loaded config file.

    Keyword arguments:
        overrides: Star detection override values to use instead of the defaults.

    Return:
        pt: [SimpleNamespace] Stand-in for PlateTool.
    """

    # The loaded config
    pt = types.SimpleNamespace()
    pt.config = cr.parse(config_path)

    # The values on the Star Detection tab
    pt.override_intensity_threshold = overrides.get('intensity_threshold', 41)
    pt.override_neighborhood_size = overrides.get('neighborhood_size', 7)
    pt.override_max_stars = overrides.get('max_stars', 1200)
    pt.override_segment_radius = overrides.get('segment_radius', 5)
    pt.override_max_feature_ratio = overrides.get('max_feature_ratio', 0.77)
    pt.override_roundness_threshold = overrides.get('roundness_threshold', 0.33)
    pt.tuned_cat_lim_mag = overrides.get('tuned_cat_lim_mag', 6.3)

    # The methods under test, and the GUI updates they call
    pt.configMaxStars = functools.partial(SF.PlateTool.configMaxStars, pt)
    pt._writeStarDetectionConfig = functools.partial(SF.PlateTool._writeStarDetectionConfig, pt)
    pt._updateConfigSaveButtonState = lambda: None
    pt.updateFileManagerButton = lambda: None

    return pt


def saveConfig(SF, pt, target_dir, save_lm=False):
    """ Save the config into a folder the way the File Manager does after its save dialog.

    Arguments:
        SF: [module] The SkyFit2 module.
        pt: [SimpleNamespace] Stand-in for PlateTool, see fakePlateTool.
        target_dir: [str] Folder to save into.

    Keyword arguments:
        save_lm: [bool] Also write the tuned catalog limiting magnitude. False by default.
    """

    # The save only needs the plate tool and the refresh of the dialog
    dialog = types.SimpleNamespace(plate_tool=pt, _refreshAll=lambda: None)
    SF.CalibrationFilesDialog._saveFile(dialog, "Config", [target_dir], save_lm=save_lm)


def failureReported(messages):
    """ Check that a failed save was reported in an error message box.

    Arguments:
        messages: [list] Keyword arguments of the recorded message boxes.

    Return:
        [bool] True if an error box reports a failure.
    """

    return any((m.get("message_type") == "error") and ("failed" in m.get("message", "")) for m in messages)


def readBytes(path):
    """ Read a file as bytes.

    Arguments:
        path: [str] File path.

    Return:
        [bytes] File contents.
    """

    with open(path, 'rb') as f:
        return f.read()


@pytest.mark.parametrize("spelling", ["{k}: {v}", "{k}:{v}", "{k}={v}", "{k} = {v}", "{k} : {v}  ; note"])
def testWriteConfigAnyKeySpelling(SF, tmp_path, quietMessages, spelling):
    """ Every key spelling configparser accepts is updated in place, never appended a second time. """

    # Respell the star detection keys of the template config
    lines = []
    section = None
    with open(TEMPLATE_CONFIG) as f:
        for line in f:
            if line.strip().startswith('['):
                section = line.strip()
            key = line.split(':')[0].strip()
            if (section == "[StarExtraction]") and (key in STAR_KEYS) and (not line.lstrip().startswith(';')):
                value = line.split(':', 1)[1].split(';')[0].strip()
                line = spelling.format(k=key, v=value) + "\n"
            lines.append(line)

    # Write the respelled config
    cfg_path = str(tmp_path / ".config")
    with open(cfg_path, 'w') as f:
        f.writelines(lines)

    # Write the new values and a catalog LM into it
    pt = fakePlateTool(SF, cfg_path)
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


def testUpdateConfigLinesAddsMissing(SF):
    """ Missing keys go at the end of their section, missing sections at the end of the file. """

    # A key of another spelling, a key and a section that are missing, a last line without a line ending
    lines = ["[StarExtraction]\n", "max_stars=100\n", "\n", "[Other]\n", "x: 1"]
    new = SF.updateConfigLines(lines, {"StarExtraction": {"max_stars": "5", "segment_radius": "4"},
                                       "Calibration": {"catalog_mag_limit": "6.0"}})

    # Updated in place, added after the last key of its section, and appended in a new section
    assert new[:3] == ["[StarExtraction]\n", "max_stars=5\n", "segment_radius: 4\n"]
    assert "x: 1\n" in new
    assert new[-2:] == ["\n[Calibration]\n", "catalog_mag_limit: 6.0\n"]


@pytest.mark.parametrize("text, check", [
    # An empty value must not swallow the line ending
    ("[StarExtraction]\nmax_stars: \nsegment_radius: 3\n", None),
    # configparser does not strip section names, so this is not the StarExtraction section
    ("[ StarExtraction ]\nmax_stars: 1\n", "[ StarExtraction ]\nmax_stars: 1\n"),
    # An indented line is the continuation of the value above it
    ("[StarExtraction]\nmax_stars: 200\n  intensity_threshold: 5\n", None),
])
def testUpdateConfigLinesParserEdgeCases(SF, text, check):
    """ The updated lines parse with the strict parser to exactly the new values. """

    import configparser

    # Update the lines
    updates = {"StarExtraction": {"max_stars": "300", "segment_radius": "6", "intensity_threshold": "25"}}
    out = "".join(SF.updateConfigLines(text.splitlines(True), updates))

    # Parse the result like RMS does
    parser = configparser.RawConfigParser(inline_comment_prefixes=(';',), strict=True)
    parser.read_string(out)
    for key, value in updates["StarExtraction"].items():
        assert parser.get("StarExtraction", key) == value

    # The lines that must stay as they are
    if check is not None:
        assert out.startswith(check)


def testSaveConfigKeepsTargetStation(SF, tmp_path, quietMessages):
    """ Saving the config to another station patches that station's config instead of replacing it. """

    # The loaded config
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

    # Save into the target folder
    pt = fakePlateTool(SF, src_path)
    saveConfig(SF, pt, str(dst_dir))

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


def testSaveConfigToNewFolderCopiesLoadedConfig(SF, tmp_path, quietMessages):
    """ A folder without a config gets a copy of the loaded config with the new values, and no backup. """

    # The loaded config, and a target folder without a config
    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"
    src_dir.mkdir()
    dst_dir.mkdir()

    src_path = str(src_dir / ".config")
    shutil.copy(TEMPLATE_CONFIG, src_path)

    # Save into the target folder
    pt = fakePlateTool(SF, src_path)
    saveConfig(SF, pt, str(dst_dir))

    # The new config is the loaded one with the new detection settings
    dst_path = str(dst_dir / ".config")
    assert cr.parse(dst_path).stationID == cr.parse(src_path).stationID
    assert cr.parse(dst_path).intensity_threshold == 41

    # A copy made by the save itself needs no backup
    assert not [f for f in os.listdir(str(dst_dir)) if ".bak." in f]

    # The loaded config was not written, so its in-memory values stay as loaded
    assert pt.config.intensity_threshold == cr.parse(src_path).intensity_threshold


def testSaveConfigFailureNotReported(SF, tmp_path, quietMessages, monkeypatch):
    """ A failed config write is reported as failed and leaves the in-memory config alone. """

    # The loaded config
    cfg_path = str(tmp_path / ".config")
    shutil.copy(TEMPLATE_CONFIG, cfg_path)
    pt = fakePlateTool(SF, cfg_path)
    orig_threshold = pt.config.intensity_threshold

    # Every file SkyFit2 opens for writing fails
    real_open = open

    def failingOpen(path, mode='r', *args, **kwargs):
        if 'w' in mode:
            raise IOError("disk full")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(SF, "open", failingOpen, raising=False)

    # Save it into its own folder
    saveConfig(SF, pt, str(tmp_path))

    # Reported as failed, not as saved, and the in-memory values are the ones of the file
    assert pt.config.intensity_threshold == orig_threshold
    assert failureReported(quietMessages)
    assert not any(m.get("title") == "Settings Saved" for m in quietMessages)


def testConfigWriteKeepsCRLFAndEncoding(SF, tmp_path, quietMessages):
    """ A CRLF config with non-ASCII text keeps every byte except the updated lines. """

    # A CRLF config with a UTF-8 comment
    with open(TEMPLATE_CONFIG, encoding='utf-8') as f:
        text = f.read()
    text = text.replace("stationID: XX0001", "stationID: XX0001 ; Kri\u017eevci, \u010cakovec")
    raw = text.replace("\n", "\r\n").encode('utf-8')

    cfg_path = str(tmp_path / ".config")
    with open(cfg_path, 'wb') as f:
        f.write(raw)

    # Write the new values into it
    pt = fakePlateTool(SF, cfg_path)
    pt._writeStarDetectionConfig(cfg_path)

    new_raw = readBytes(cfg_path)

    # Only CRLF line endings, and the same number of lines
    old_lines = raw.split(b"\r\n")
    new_lines = new_raw.split(b"\r\n")
    assert b"\n" not in new_raw.replace(b"\r\n", b"")
    assert len(new_lines) == len(old_lines)

    # Only the star detection lines changed
    changed = [(a, b) for a, b in zip(old_lines, new_lines) if a != b]
    changed_keys = {b.split(b":")[0] for _, b in changed}
    assert changed_keys <= {key.encode('ascii') for key in STAR_KEYS}
    assert cr.parse(cfg_path).intensity_threshold == 41


def testConfigWriteKeepsSymlinkedConfig(SF, tmp_path, quietMessages):
    """ Writing a symlinked config updates the file it points to and keeps the link a link. """

    # The real station config
    real_dir = tmp_path / "station"
    real_dir.mkdir()
    real_cfg = str(real_dir / ".config")
    shutil.copy(TEMPLATE_CONFIG, real_cfg)

    # The config the tool is pointed at is a symlink to the station's real config
    link_cfg = str(tmp_path / ".config")
    try:
        os.symlink(real_cfg, link_cfg)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("Symbolic links cannot be made here")

    # Write the new values through the link
    pt = fakePlateTool(SF, link_cfg)
    pt._writeStarDetectionConfig(link_cfg)

    # The link is still a link to the real config, which holds the new values
    assert os.path.islink(link_cfg)
    assert os.path.realpath(link_cfg) == os.path.realpath(real_cfg)
    assert cr.parse(real_cfg).intensity_threshold == 41


def testSaveConfigWriteFailureKeepsConfigAndBackup(SF, tmp_path, quietMessages, monkeypatch):
    """ A config write that fails part way through keeps the config and its backup and is reported. """

    # The loaded config
    cfg_path = str(tmp_path / ".config")
    shutil.copy(TEMPLATE_CONFIG, cfg_path)
    original = readBytes(cfg_path)

    pt = fakePlateTool(SF, cfg_path)

    # A file opened for writing takes the first 100 characters, then the disk is full
    real_open = open

    class FailingFile(object):
        def __init__(self, f):
            self.f = f

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.f.close()

        def writelines(self, lines):
            self.f.write("".join(lines)[:100])
            raise IOError("No space left on device")

    def failingOpen(path, mode='r', *args, **kwargs):
        f = real_open(path, mode, *args, **kwargs)
        if 'w' in mode:
            return FailingFile(f)
        return f

    monkeypatch.setattr(SF, "open", failingOpen, raising=False)

    # Save it into its own folder
    saveConfig(SF, pt, str(tmp_path))

    # The config is untouched and the backup holds it
    assert readBytes(cfg_path) == original
    backups = [f for f in os.listdir(str(tmp_path)) if ".bak." in f]
    assert len(backups) == 1
    assert readBytes(str(tmp_path / backups[0])) == original

    # No temporary file is left behind, and the failure is reported
    assert not [f for f in os.listdir(str(tmp_path)) if f.endswith(".tmp")]
    assert failureReported(quietMessages)


@pytest.mark.skipif(hasattr(os, 'geteuid') and (os.geteuid() == 0),
                    reason="root can write to a read-only file")
def testSaveConfigReadOnlyConfigRefused(SF, tmp_path, quietMessages):
    """ A read-only config in a writable folder is reported as failed and left as it is, not replaced. """

    # The loaded config
    cfg_path = str(tmp_path / ".config")
    shutil.copy(TEMPLATE_CONFIG, cfg_path)
    original = readBytes(cfg_path)

    pt = fakePlateTool(SF, cfg_path)
    orig_threshold = pt.config.intensity_threshold

    # Make it read-only and save it into its own folder
    os.chmod(cfg_path, stat.S_IRUSR)
    try:
        saveConfig(SF, pt, str(tmp_path))
        mode_after = stat.S_IMODE(os.stat(cfg_path).st_mode)
    finally:
        os.chmod(cfg_path, stat.S_IRUSR | stat.S_IWUSR)

    # The failure is reported, and the in-memory config keeps the values the file holds
    assert failureReported(quietMessages)
    assert not any(m.get("title") == "Settings Saved" for m in quietMessages)
    assert pt.config.intensity_threshold == orig_threshold

    # The file is the same, still read-only, and not backed up
    assert readBytes(cfg_path) == original
    assert mode_after == stat.S_IRUSR
    assert not [f for f in os.listdir(str(tmp_path)) if ".bak." in f]
