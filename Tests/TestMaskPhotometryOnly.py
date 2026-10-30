"""Tests for the two mask modes of SkyFit2: excluding stars entirely, or from photometry only."""

import types

import numpy as np
import pytest

from RMS.Routines.MaskImage import MaskStructure


def makeMask(x_res, y_res, box):
    """Mask of the given resolution with the box (x0, x1, y0, y1) masked."""
    img = np.full((y_res, x_res), 255, dtype=np.uint8)
    x0, x1, y0, y1 = box
    img[y0:y1, x0:x1] = 0
    return MaskStructure(img)


def makePlateTool(mask, photometry_only, x_res=1280, y_res=720):
    """Object carrying the PlateTool mask methods and only the state they read."""
    PlateTool = pytest.importorskip("Utils.SkyFit2").PlateTool

    tool = types.SimpleNamespace(mask=mask, mask_photometry_only=photometry_only,
                                 platepar=types.SimpleNamespace(X_res=x_res, Y_res=y_res))
    for name in ('astrometryMask', 'pointsBehindMask', 'filterCatalogStarsByMask'):
        setattr(tool, name, types.MethodType(getattr(PlateTool, name), tool))

    return tool


# A star behind the masked box, one in clear sky, and one in the clear column just left of the box
STAR_X = np.array([150.0, 150.0, 99.9])
STAR_Y = np.array([60.0, 10.0, 60.0])
BOX = (100, 200, 50, 80)


class TestPlateToolMaskModes:

    def test_full_exclusion_masks_astrometry_and_photometry(self):
        tool = makePlateTool(makeMask(1280, 720, BOX), photometry_only=False)

        assert tool.astrometryMask() is tool.mask
        assert tool.pointsBehindMask(STAR_X, STAR_Y).tolist() == [True, False, False]
        assert tool.pointsBehindMask(STAR_X, STAR_Y, photometry=True).tolist() == [True, False, False]

    def test_photometry_only_keeps_masked_stars_for_astrometry(self):
        tool = makePlateTool(makeMask(1280, 720, BOX), photometry_only=True)

        assert tool.astrometryMask() is None
        assert tool.pointsBehindMask(STAR_X, STAR_Y).tolist() == [False, False, False]
        assert tool.filterCatalogStarsByMask(STAR_X, STAR_Y).tolist() == [True, True, True]
        assert tool.pointsBehindMask(STAR_X, STAR_Y, photometry=True).tolist() == [True, False, False]

    def test_full_exclusion_filters_the_catalog_stars(self):
        tool = makePlateTool(makeMask(1280, 720, BOX), photometry_only=False)
        catalog_stars = np.array([[10.0, 20.0, 3.0], [11.0, 21.0, 4.0], [12.0, 22.0, 5.0]])

        kept, not_masked = tool.filterCatalogStarsByMask(STAR_X, STAR_Y, catalog_stars)

        assert not_masked.tolist() == [False, True, True]
        assert kept.tolist() == catalog_stars[1:].tolist()

    def test_points_outside_take_the_border_pixel(self):
        tool = makePlateTool(makeMask(1280, 720, (0, 10, 0, 720)), photometry_only=False)
        points_behind = tool.pointsBehindMask(np.array([-5.0, 1300.0]), np.array([360.0, 360.0]), photometry=True)
        assert points_behind.tolist() == [True, False]

    def test_mask_of_another_resolution_is_ignored(self):
        tool = makePlateTool(makeMask(1920, 1080, BOX), photometry_only=False)

        assert tool.pointsBehindMask(STAR_X, STAR_Y).tolist() == [False, False, False]
        assert tool.pointsBehindMask(STAR_X, STAR_Y, photometry=True).tolist() == [False, False, False]

    def test_no_mask(self):
        for mask in (None, MaskStructure(None)):
            tool = makePlateTool(mask, photometry_only=False)
            assert tool.pointsBehindMask(STAR_X, STAR_Y, photometry=True).tolist() == [False, False, False]
