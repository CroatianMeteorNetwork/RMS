"""
Tests of Find Best Frame in SkyFit2 (PlateTool.findBestFrame) when the best frame of the CALSTARS has no FF file
in the data folder.

The auto fit then runs on a generated placeholder image. The placeholder only exists in memory: a file written to
the data folder was left behind, and was taken for an FF file by the other tools which scan the folder. The
placeholder gets the stars detected on the best frame, and the re-detected ones too.

The tests run offscreen on a copy of the example station AU000A, and are skipped when Qt is not available.
"""

import os
import shutil
import tarfile

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

pytest.importorskip("pyqtgraph")

try:
    import Utils.SkyFit2 as SF
except SystemExit:
    pytest.skip("SkyFit2 could not import its Qt dependencies", allow_module_level=True)

from pyqtgraph.Qt import QtWidgets

import RMS.ConfigReader as cr
import RMS.Routines.CustomPyqtgraphClasses as CPC


REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIONS_ARCHIVE = os.path.join(REPO_DIR, "Tests", "ExampleStationData", "stations.tar.bz2")


@pytest.fixture(scope="module")
def qapp():
    """ The offscreen QApplication shared by the tests. """

    app = QtWidgets.QApplication.instance()
    if app is None:
        try:
            app = QtWidgets.QApplication([])
        except Exception as e:
            pytest.skip("No Qt application could be created: {}".format(e))

    return app


@pytest.fixture(scope="module")
def stationArchive(tmp_path_factory):
    """ The example station AU000A, extracted once per module. """

    if not os.path.isfile(STATIONS_ARCHIVE):
        pytest.skip("The example station archive is missing")

    out_dir = str(tmp_path_factory.mktemp("stations"))
    with tarfile.open(STATIONS_ARCHIVE, "r:bz2") as tar:
        members = [m for m in tar.getmembers() if m.name.startswith("Stations/AU000A")]
        tar.extractall(out_dir, members=members)

    return os.path.join(out_dir, "Stations", "AU000A")


@pytest.fixture
def stationDir(stationArchive, tmp_path):
    """ A fresh copy of the example station for each test. """

    dir_path = str(tmp_path / "AU000A")
    shutil.copytree(stationArchive, dir_path)

    return dir_path


@pytest.fixture
def quietMessages(monkeypatch):
    """ Do not show the SkyFit2 message boxes, and answer questions with Yes. """

    monkeypatch.setattr(SF, "qmessagebox", lambda *a, **k: None)
    monkeypatch.setattr(CPC, "qmessagebox", lambda *a, **k: None)

    yes = QtWidgets.QMessageBox.StandardButton.Yes
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", staticmethod(lambda *a, **k: yes))
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QtWidgets.QMessageBox, name, staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", lambda self: 0, raising=False)
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec_", lambda self: 0, raising=False)


@pytest.fixture
def plateTool(qapp, stationDir, quietMessages):
    """ A SkyFit2 PlateTool on a fresh copy of the example station. """

    config = cr.loadConfigFromDirectory('.config', stationDir)
    pt = SF.PlateTool(stationDir, config)
    qapp.processEvents()

    yield pt

    pt.close()
    pt.deleteLater()
    qapp.processEvents()


def testBestFramePlaceholderNotLeftInDataFolder(plateTool, monkeypatch):
    """ The auto fit on a frame without an image uses a placeholder which is not written to the data folder, and
        gets the re-detected stars of that frame. """

    pt = plateTool
    real_ff = pt.img_handle.name()
    missing_ff = "FF_AU000A_20250817_200000_000_0600000.fits"
    files_before = sorted(os.listdir(pt.dir_path))

    # The stars of the frame without an image, in the CALSTARS and re-detected
    stars = pt.calstars[real_ff]
    pt.calstars[missing_ff] = stars
    pt.star_detection_override_data[missing_ff] = stars

    # The frame without an image is the best one
    monkeypatch.setattr(SF, "selectBestFrame",
                        lambda data, *a, **k: (missing_ff if missing_ff in data else real_ff, 1.0, {}))

    # Choose the auto fit in the dialog, and record the image of the fit instead of running it
    def clickedButton(self):
        for button in self.buttons():
            if button.text().startswith("Auto Fit"):
                return button
        return None

    monkeypatch.setattr(QtWidgets.QMessageBox, "clickedButton", clickedButton)
    fits = []
    pt.autoFitAstrometryNet = lambda: fits.append(pt.img_handle.name())

    pt.findBestFrame()

    # The fit ran on the placeholder, which has the re-detected stars and is not in the data folder
    placeholder_name = "FF_AU000A_20250817_200000_000_0600000_placeholder.png"
    assert fits == [placeholder_name]
    assert sorted(os.listdir(pt.dir_path)) == files_before
    assert pt.calstars[placeholder_name] is stars
    assert pt.star_detection_override_data[placeholder_name] is stars

    # The placeholder can be shown again after navigating away
    pt.nextImg(n=1)
    assert pt.img_handle.name() == real_ff
    pt.nextImg(n=-1)
    assert pt.img_handle.name() == placeholder_name
    assert np.all(pt.img_handle.loadChunk().avepixel <= 40)
