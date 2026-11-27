""" Headless tests for SkyFit2 File Manager platepar naming and mask save handling. """

from __future__ import print_function, division, absolute_import

import os
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def skyfit2():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pg = pytest.importorskip("pyqtgraph")
    QtWidgets = pg.Qt.QtWidgets
    if QtWidgets.QApplication.instance() is None:
        QtWidgets.QApplication([])
    return pytest.importorskip("Utils.SkyFit2")


def _makeDialogStub(skyfit2, pt):
    dlg = skyfit2.CalibrationFilesDialog.__new__(skyfit2.CalibrationFilesDialog)
    dlg.plate_tool = pt
    dlg._refreshAll = lambda: None
    return dlg


def testFileManagerPlateparSaveNamePolicy(skyfit2, monkeypatch, tmp_path):
    """ Platepar loaded from target folder keeps its filename; foreign/new platepar uses config.platepar_name. """

    folder_a = tmp_path/"station_a"
    folder_b = tmp_path/"station_b"
    folder_a.mkdir()
    folder_b.mkdir()

    written = []
    errors = []
    pp = SimpleNamespace(write=lambda dest, fmt=None, fov=None: written.append(dest))
    pt = SimpleNamespace(
        config=SimpleNamespace(platepar_name="platepar_cmn2010.cal"),
        platepar=pp,
        platepar_fmt="json",
        platepar_file=str(folder_a/"custom_pick.cal"),
        platepar_modified=True,
        updateFileManagerButton=lambda: None,
    )

    monkeypatch.setattr(skyfit2, "computeFOVSize", lambda p: (60.0, 35.0))
    monkeypatch.setattr(skyfit2, "qmessagebox", lambda **kw: errors.append(kw))

    dlg = _makeDialogStub(skyfit2, pt)

    # Saving back to folder_a preserves custom_pick.cal
    dlg._saveFile("Platepar", [str(folder_a)])
    assert written[-1] == str(folder_a/"custom_pick.cal")
    assert pt.platepar_modified is False
    assert errors == []

    # Saving the same foreign platepar into folder_b uses config.platepar_name
    dlg._saveFile("Platepar", [str(folder_b)])
    assert written[-1] == str(folder_b/"platepar_cmn2010.cal")
    assert errors == []


def testFileManagerMaskSaveUpdatesDetectionMaskAndChecksImwrite(skyfit2, monkeypatch, tmp_path):
    """ Mask save must update pt.mask, refilter catalog stars, and report imwrite failure instead of saved. """

    mask_arr = np.full((64, 64), 255, dtype=np.uint8)
    mask_arr[:10, :10] = 0
    unsaved_calls = []
    star_updates = []
    errors = []

    pt = SimpleNamespace(
        generateMaskImage=lambda: mask_arr,
        mask_source_path="/old/mask.bmp",
        mask=None,
        updateStars=lambda only_update_catalog=False: star_updates.append(only_update_catalog),
        updateFileManagerButton=lambda: None,
        tab=SimpleNamespace(mask=SimpleNamespace(setUnsaved=lambda v: unsaved_calls.append(v))),
    )

    monkeypatch.setattr(skyfit2, "qmessagebox", lambda **kw: errors.append(kw))
    dlg = _makeDialogStub(skyfit2, pt)

    # 1. Failed imwrite -> reported via error dialog, does not update mask_source_path or clear unsaved
    monkeypatch.setattr(skyfit2.cv2, "imwrite", lambda p, img: False)
    dlg._saveFile("Mask", [str(tmp_path)])
    assert len(errors) == 1
    assert "failed" in errors[-1]["message"].lower()
    assert pt.mask_source_path == "/old/mask.bmp"
    assert pt.mask is None
    assert unsaved_calls == []
    assert star_updates == []

    # 2. Successful imwrite -> no error dialog, updates pt.mask, refilters catalog stars, clears unsaved
    monkeypatch.setattr(skyfit2.cv2, "imwrite", lambda p, img: True)
    dlg._saveFile("Mask", [str(tmp_path)])
    assert len(errors) == 1
    assert pt.mask_source_path == os.path.join(str(tmp_path), "mask.bmp")
    assert pt.mask is not None
    assert np.array_equal(pt.mask.img, mask_arr)
    assert star_updates == [True]
    assert unsaved_calls == [False]
