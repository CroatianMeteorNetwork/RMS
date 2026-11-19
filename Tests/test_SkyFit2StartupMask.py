""" Tests of the initialization of the SkyFit2 mask editor (PlateTool.initMaskFromFile) with a mask given at
startup (--mask, or found in the RMS root or the working directory).

When the data directory has no mask.bmp, the startup mask must be edited instead of being replaced with an
empty mask, which made star detection and the catalog filter run unmasked. A mask.bmp in the data directory
still takes precedence, and a station change still clears the mask of the previous station.
"""

from __future__ import print_function, division, absolute_import

from types import SimpleNamespace

import numpy as np
import pytest


# Image size (px) of the data in the tests
IMG_WIDTH = 320
IMG_HEIGHT = 180


def maskImage(width, height, masked_rows):
    """ A mask image with the given band of rows masked.

    Arguments:
        width: [int] Image width (px).
        height: [int] Image height (px).
        masked_rows: [tuple] First and last row (px) of the masked band.

    Return:
        mask_img: [ndarray] Grayscale mask image (0 = masked, 255 = unmasked).
    """

    mask_img = np.full((height, width), 255, dtype=np.uint8)
    mask_img[masked_rows[0]:masked_rows[1] + 1, :] = 0

    return mask_img


@pytest.fixture(scope="module")
def skyfit2():
    return pytest.importorskip("Utils.SkyFit2")


@pytest.fixture
def plateTool(skyfit2, tmp_path):
    """ A stand-in with the mask editor methods of PlateTool and no GUI, on a data directory without a
        mask.bmp. """

    PlateTool = skyfit2.PlateTool

    # The mask editor methods under test, taken from PlateTool
    class MaskEditor(object):
        initMaskFromFile = PlateTool.initMaskFromFile
        loadMaskFromFile = PlateTool.loadMaskFromFile
        generateMaskImage = PlateTool.generateMaskImage
        maskFitsImage = getattr(PlateTool, 'maskFitsImage', None)
        loadMaskImage = getattr(PlateTool, 'loadMaskImage', None)

        # The GUI updates are not needed
        def updateMaskDisplay(self):
            pass

        def _updateMaskStatus(self):
            pass

        def _exitBrushMode(self):
            pass

    pt = MaskEditor()
    pt.dir_path = str(tmp_path)
    pt.mask_brush_mode = False
    pt.mask_source_path = None
    pt.mask_polygons = []
    pt.mask_current_polygon = []
    pt.mask_paint_layer = None
    pt.mask_brush_stroke_history = []
    pt.tab = SimpleNamespace(mask=SimpleNamespace(setUndoEnabled=lambda enabled: None,
                                                  setUnsaved=lambda unsaved: None))

    # The image data is indexed as (x, y)
    pt.img = SimpleNamespace(data=np.zeros((IMG_WIDTH, IMG_HEIGHT), dtype=np.uint8))

    return pt


def testStartupMaskKeptWithoutMaskFile(skyfit2, plateTool):
    """ Without a mask.bmp in the data directory, the mask given at startup becomes the edited mask. """

    startup_img = maskImage(IMG_WIDTH, IMG_HEIGHT, (150, 179))
    plateTool.mask = skyfit2.MaskStructure(startup_img.copy())
    plateTool.startup_mask = plateTool.mask

    plateTool.initMaskFromFile()

    # The editor holds the startup mask, and the detection mask is the same image
    assert len(plateTool.mask_polygons) > 0
    assert np.array_equal(plateTool.generateMaskImage(), startup_img)
    assert np.array_equal(plateTool.mask.img, startup_img)


def testStationChangeClearsStartupMask(skyfit2, plateTool):
    """ The startup mask is only used once: a later initialization (a station change) clears the mask. """

    plateTool.mask = skyfit2.MaskStructure(maskImage(IMG_WIDTH, IMG_HEIGHT, (150, 179)))
    plateTool.startup_mask = plateTool.mask

    plateTool.initMaskFromFile()
    plateTool.initMaskFromFile()

    assert plateTool.mask_polygons == []
    assert np.all(plateTool.mask.img == 255)


def testMaskFileTakesPrecedence(skyfit2, plateTool):
    """ A mask.bmp in the data directory is loaded instead of the startup mask. """

    import cv2

    file_img = maskImage(IMG_WIDTH, IMG_HEIGHT, (0, 29))
    cv2.imwrite(plateTool.dir_path + "/mask.bmp", file_img)

    plateTool.mask = skyfit2.MaskStructure(maskImage(IMG_WIDTH, IMG_HEIGHT, (150, 179)))
    plateTool.startup_mask = plateTool.mask

    plateTool.initMaskFromFile()

    assert np.array_equal(plateTool.mask.img, file_img)


def testStartupMaskOfOtherSizeIgnored(skyfit2, plateTool):
    """ A startup mask of another size than the image is not applied, the mask is cleared. """

    plateTool.mask = skyfit2.MaskStructure(maskImage(2*IMG_WIDTH, 2*IMG_HEIGHT, (300, 359)))
    plateTool.startup_mask = plateTool.mask

    plateTool.initMaskFromFile()

    assert plateTool.mask_polygons == []
    assert plateTool.mask.img.shape == (IMG_HEIGHT, IMG_WIDTH)
    assert np.all(plateTool.mask.img == 255)


def testStateWithoutStartupMaskClearsMask(skyfit2, plateTool):
    """ A state saved by an earlier version has no startup mask attribute: the mask is cleared as before. """

    plateTool.mask = skyfit2.MaskStructure(maskImage(IMG_WIDTH, IMG_HEIGHT, (150, 179)))

    plateTool.initMaskFromFile()

    assert plateTool.mask_polygons == []
    assert np.all(plateTool.mask.img == 255)
