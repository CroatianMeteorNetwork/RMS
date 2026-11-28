""" Headless tests for SkyFit2 mask editing tab guard, brush undo clearing, and unreadable mask loads. """

from __future__ import print_function, division, absolute_import

from types import SimpleNamespace

import cv2
import numpy as np
import pytest


@pytest.fixture(scope="module")
def skyfit2():
    pg = pytest.importorskip("pyqtgraph")
    QtWidgets = pg.Qt.QtWidgets
    if QtWidgets.QApplication.instance() is None:
        QtWidgets.QApplication([])
    return pytest.importorskip("Utils.SkyFit2")


def _makeMaskTool(skyfit2, on_mask_tab):
    """ Build a lightweight PlateTool stand-in bound to the real mask mouse and load methods. """

    PlateTool = skyfit2.PlateTool

    class _StubTool(object):
        isMaskTabCurrent = PlateTool.isMaskTabCurrent
        updatePanningEnabled = PlateTool.updatePanningEnabled
        onMousePressed = PlateTool.onMousePressed
        loadMaskFromFile = PlateTool.loadMaskFromFile
        loadMaskImage = PlateTool.loadMaskImage

        def updateMaskDisplay(self):
            return None

        def _updateMaskStatus(self):
            return None

        def updateMaskFromPolygons(self):
            return None

    pt = _StubTool()
    pt.mode = "astrometry"
    pt.cursor = None
    pt.mask_brush_mode = False
    pt.mask_draw_mode = False
    pt.mask_source_path = "/existing/mask.bmp"
    pt.mask_polygons = [[(10.0, 10.0), (100.0, 10.0), (100.0, 100.0), (10.0, 100.0)]]
    pt.mask_current_polygon = []
    pt.mask_paint_layer = None
    pt.mask_brush_stroke_history = ["stale_stroke"]
    pt.mask_dragging_vertex = None
    pt.img_frame = SimpleNamespace(
        panning_enabled=True,
        mapSceneToView=lambda pos: SimpleNamespace(x=lambda: pos.x(), y=lambda: pos.y()),
    )
    pt.tab = SimpleNamespace(
        mask=SimpleNamespace(setUndoEnabled=lambda v: None, setUnsaved=lambda v: None),
        currentIndex=lambda: 1 if on_mask_tab else 0,
        indexOf=lambda w: 1,
    )
    pt.brush_calls = []
    pt.brushStrokeBegin = lambda: pt.brush_calls.append("begin")
    pt.brushPaintAt = lambda x, y: pt.brush_calls.append((x, y))
    pt.findNearestMaskVertex = lambda x, y, threshold=15: (0, 0)
    pt.deleted_vertices = []
    pt.deleteMaskVertex = lambda v: pt.deleted_vertices.append(v)
    return pt


def _fakeMouseEvent(skyfit2, button, x=10.5, y=10.5):
    return SimpleNamespace(
        button=lambda: button,
        scenePos=lambda: SimpleNamespace(x=lambda: x, y=lambda: y),
    )


def testMaskVerticesOnlyEditedOnMaskTab(skyfit2):
    """ Clicking near a mask vertex or with brush mode on a non-Mask tab must not edit the mask. """

    QtCore = skyfit2.QtCore

    # Off the Mask tab: right-clicking near vertex (0, 0) must not delete it or start brush paint
    pt_other = _makeMaskTool(skyfit2, on_mask_tab=False)
    pt_other.mask_brush_mode = True
    pt_other.onMousePressed(_fakeMouseEvent(skyfit2, QtCore.Qt.MouseButton.RightButton))
    assert pt_other.deleted_vertices == []
    assert pt_other.brush_calls == []

    # Off the Mask tab with brush off: right-clicking near vertex must not delete it either
    pt_other.mask_brush_mode = False
    pt_other.onMousePressed(_fakeMouseEvent(skyfit2, QtCore.Qt.MouseButton.RightButton))
    assert pt_other.deleted_vertices == []

    # On the Mask tab: brush click paints, left-click starts vertex drag, right-click deletes vertex
    pt_mask = _makeMaskTool(skyfit2, on_mask_tab=True)
    pt_mask.mask_brush_mode = True
    pt_mask.onMousePressed(_fakeMouseEvent(skyfit2, QtCore.Qt.MouseButton.LeftButton))
    assert len(pt_mask.brush_calls) == 2

    pt_mask.mask_brush_mode = False
    pt_mask.onMousePressed(_fakeMouseEvent(skyfit2, QtCore.Qt.MouseButton.LeftButton))
    assert pt_mask.mask_dragging_vertex == (0, 0)

    pt_mask.onMousePressed(_fakeMouseEvent(skyfit2, QtCore.Qt.MouseButton.RightButton))
    assert pt_mask.deleted_vertices == [(0, 0)]


def testMaskLoadClearsBrushHistoryForCleanPolygonMask(skyfit2, tmp_path):
    """ Loading a clean polygon mask with no raster residuals must still clear brush undo history. """

    pt = _makeMaskTool(skyfit2, on_mask_tab=True)
    assert pt.mask_brush_stroke_history == ["stale_stroke"]

    # Clean rectangle mask (exact polygon match -> paint_layer stays None)
    mask_img = np.full((120, 160), 255, dtype=np.uint8)
    mask_img[20:80, 30:130] = 0
    mask_path = str(tmp_path/"clean_mask.bmp")
    cv2.imwrite(mask_path, mask_img)

    assert pt.loadMaskFromFile(mask_path) is True
    assert pt.mask_source_path == mask_path
    assert pt.mask_brush_stroke_history == []


def testUnreadableMaskNotReportedLoadedAndKeepsSource(skyfit2, tmp_path):
    """ Missing or unreadable mask files must return False and preserve the previous mask_source_path. """

    pt = _makeMaskTool(skyfit2, on_mask_tab=True)

    # Missing file
    assert pt.loadMaskFromFile(str(tmp_path/"missing.bmp")) is False
    assert pt.mask_source_path == "/existing/mask.bmp"

    # Corrupt/unreadable file
    bad_path = str(tmp_path/"corrupt.bmp")
    with open(bad_path, "wb") as f:
        f.write(b"not a valid image")

    assert pt.loadMaskFromFile(bad_path) is False
    assert pt.mask_source_path == "/existing/mask.bmp"
