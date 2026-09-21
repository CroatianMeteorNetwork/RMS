""" Tests for the SkyFit2 mask editing helpers in RMS.Routines.MaskImage (no GUI dependencies). """

from __future__ import print_function, division, absolute_import

import numpy as np
import cv2

from RMS.Routines.MaskImage import compositeMaskLayers, maskRasterResiduals, decomposeMaskImage, \
    resampleMaskLayers, paintBrushSegment, PAINT_MASKED, PAINT_UNMASKED


class TestPaintLayerCompositing:
    """ The paint layer must override the polygon fill. """

    def setup_method(self):
        self.width = 100
        self.height = 80

    def test_polygon_only(self):
        polygons = [[(10, 10), (50, 10), (50, 50), (10, 50)]]
        mask = compositeMaskLayers(polygons, None, self.width, self.height)
        assert mask.shape == (self.height, self.width)
        assert mask.dtype == np.uint8
        assert mask[10, 10] == 0
        assert mask[30, 30] == 0
        assert mask[0, 0] == 255
        assert mask[70, 90] == 255

    def test_brush_mask_only(self):
        paint = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.circle(paint, (60, 40), 10, PAINT_MASKED, -1)
        mask = compositeMaskLayers([], paint, self.width, self.height)
        assert mask[40, 60] == 0
        assert mask[0, 0] == 255

    def test_brush_erase_inside_polygon(self):
        polygons = [[(0, 0), (99, 0), (99, 79), (0, 79)]]
        paint = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.circle(paint, (50, 40), 10, PAINT_UNMASKED, -1)

        mask = compositeMaskLayers(polygons, paint, self.width, self.height)
        assert mask[40, 50] == 255
        assert mask[0, 0] == 0

    def test_brush_overrides_polygon(self):
        polygons = [[(20, 20), (40, 20), (40, 40), (20, 40)]]
        paint = np.zeros((self.height, self.width), dtype=np.uint8)
        paint[25, 25] = PAINT_UNMASKED

        mask = compositeMaskLayers(polygons, paint, self.width, self.height)
        assert mask[25, 25] == 255
        assert mask[30, 30] == 0

    def test_empty_paint_layer_no_effect(self):
        polygons = [[(10, 10), (50, 10), (50, 50), (10, 50)]]
        paint = np.zeros((self.height, self.width), dtype=np.uint8)
        mask_with = compositeMaskLayers(polygons, paint, self.width, self.height)
        mask_without = compositeMaskLayers(polygons, None, self.width, self.height)
        np.testing.assert_array_equal(mask_with, mask_without)

    def test_mismatched_paint_layer_is_resampled(self):
        """ A paint layer saved for a smaller frame is scaled up with nearest-neighbour interpolation. """

        paint = np.zeros((40, 50), dtype=np.uint8)
        paint[10:20, 10:20] = PAINT_MASKED
        mask = compositeMaskLayers([], paint, self.width, self.height)
        assert mask.shape == (self.height, self.width)
        assert mask[30, 30] == 0
        assert mask[10, 10] == 255


class TestOverlayConsistency:
    """ The overlay (1 = masked) must show exactly the pixels the saved mask (0 = masked) masks. """

    def test_overlay_matches_mask(self):
        w, h = 100, 80
        polygons = [[(10, 10), (50, 10), (50, 50), (10, 50)]]
        paint = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(paint, (70, 60), 8, PAINT_MASKED, -1)
        paint[30, 30] = PAINT_UNMASKED

        mask = compositeMaskLayers(polygons, paint, w, h)
        overlay = compositeMaskLayers(polygons, paint, w, h, masked_value=1, unmasked_value=0)

        np.testing.assert_array_equal(mask == 0, overlay == 1)
        assert set(np.unique(overlay).tolist()) <= {0, 1}


class TestUndoSystem:
    """ Test the undo snapshot logic. """

    def test_undo_restores_previous_state(self):
        h, w = 80, 100
        history = []

        # Initial state: no paint
        paint = None
        history.append(None if paint is None else paint.copy())

        # First stroke
        paint = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(paint, (50, 40), 10, 1, -1)
        history.append(paint.copy())

        # Second stroke adds more
        cv2.circle(paint, (20, 20), 5, 1, -1)

        # Undo second stroke
        paint = history.pop()
        assert paint[40, 50] == 1
        assert paint[20, 20] == 0

        # Undo first stroke
        paint = history.pop()
        assert paint is None

    def test_max_undo_depth(self):
        max_undo = 5
        history = []
        for i in range(10):
            history.append(i)
            if len(history) > max_undo:
                history.pop(0)
        assert len(history) == max_undo
        assert history[0] == 5


class TestResidualDetection:
    """ Loading a mask must recover polygons plus the raster residuals so the round trip is lossless. """

    def test_pure_polygon_mask_no_residual(self):
        w, h = 100, 80
        polygons = [[(10, 10), (50, 10), (50, 50), (10, 50)]]
        mask = compositeMaskLayers(polygons, None, w, h)

        loaded_polygons, residual = decomposeMaskImage(mask)

        # Rectangle should round-trip perfectly
        assert len(loaded_polygons) == 1
        assert residual is None
        assert maskRasterResiduals(mask, loaded_polygons) is None

    def test_brush_strokes_create_residual(self):
        w, h = 100, 80
        paint = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(paint, (50, 40), 15, PAINT_MASKED, -1)
        mask = compositeMaskLayers([], paint, w, h)

        # Round-trip via residual detection must recover the original mask exactly
        loaded_polygons, residual = decomposeMaskImage(mask)
        remask = compositeMaskLayers(loaded_polygons, residual, w, h)
        np.testing.assert_array_equal(mask, remask,
            err_msg="Brush stroke round-trip via residual detection lost pixels")

    def test_erase_inside_polygon_creates_residual(self):
        w, h = 100, 80
        polygons = [[(0, 0), (99, 0), (99, 79), (0, 79)]]
        paint = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(paint, (50, 40), 10, PAINT_UNMASKED, -1)
        mask = compositeMaskLayers(polygons, paint, w, h)

        # The erased hole boundary may not be reproduced by the polygon approximation, but the overall
        #   mask must be reconstructable
        loaded_polygons, residual = decomposeMaskImage(mask)
        remask = compositeMaskLayers(loaded_polygons, residual, w, h)
        np.testing.assert_array_equal(mask, remask)


class TestResampleMaskLayers:
    """ A mask saved for another frame size is rescaled, polygons included. """

    def test_same_size_is_passthrough(self):
        polygons = [[(10.0, 10.0), (50.0, 10.0), (50.0, 50.0)]]
        paint = np.zeros((80, 100), dtype=np.uint8)
        out_polygons, out_paint = resampleMaskLayers(polygons, paint, (100, 80), (100, 80))
        assert out_polygons is polygons
        assert out_paint is paint

    def test_polygons_and_paint_scale_together(self):
        polygons = [[(10, 10), (50, 10), (50, 50), (10, 50)]]
        paint = np.zeros((80, 100), dtype=np.uint8)
        paint[60:70, 60:70] = PAINT_MASKED

        # Double the size in both axes
        out_polygons, out_paint = resampleMaskLayers(polygons, paint, (100, 80), (200, 160))

        assert out_polygons == [[(20, 20), (100, 20), (100, 100), (20, 100)]]
        assert out_paint.shape == (160, 200)
        assert out_paint[130, 130] == PAINT_MASKED
        assert out_paint[110, 110] == 0

        # The rescaled layers render the same picture, only larger
        small = compositeMaskLayers(polygons, paint, 100, 80)
        large = compositeMaskLayers(out_polygons, out_paint, 200, 160)
        assert large[40, 40] == small[20, 20] == 0
        assert large[130, 130] == small[65, 65] == 0
        assert large[150, 10] == small[75, 5] == 255

    def test_none_paint_stays_none(self):
        _, out_paint = resampleMaskLayers([], None, (100, 80), (50, 40))
        assert out_paint is None


class TestBrushSegment:
    """ The brush footprint must be the same disc along the whole stroke. """

    def test_first_point_is_a_disc(self):
        paint = np.zeros((80, 100), dtype=np.uint8)
        center = paintBrushSegment(paint, None, (50, 40), 5, PAINT_MASKED)
        assert center == (50, 40)
        expected = np.zeros_like(paint)
        cv2.circle(expected, (50, 40), 5, PAINT_MASKED, -1)
        np.testing.assert_array_equal(paint, expected)

    def test_segment_ends_match_the_disc(self):
        """ Every point of a stroke gets the full disc, the start and end included. """

        radius = 6
        paint = np.zeros((80, 100), dtype=np.uint8)
        pos = paintBrushSegment(paint, None, (20, 40), radius, PAINT_MASKED)
        paintBrushSegment(paint, pos, (70, 40), radius, PAINT_MASKED)

        disc_start = np.zeros_like(paint)
        cv2.circle(disc_start, (20, 40), radius, 1, -1)
        disc_end = np.zeros_like(paint)
        cv2.circle(disc_end, (70, 40), radius, 1, -1)

        assert np.all(paint[disc_start == 1] == PAINT_MASKED)
        assert np.all(paint[disc_end == 1] == PAINT_MASKED)

        # The gap between the two positions is filled
        assert np.all(paint[40, 20:71] == PAINT_MASKED)

        # Nothing far away is touched
        assert paint[10, 10] == 0
        assert paint[70, 90] == 0

    def test_coordinates_are_clamped(self):
        paint = np.zeros((80, 100), dtype=np.uint8)
        center = paintBrushSegment(paint, None, (1e6, -1e6), 3, PAINT_MASKED)
        assert center == (99, 0)
        assert paint[0, 99] == PAINT_MASKED

    def test_erase_value(self):
        paint = np.zeros((80, 100), dtype=np.uint8)
        paint[:] = PAINT_MASKED
        paintBrushSegment(paint, None, (50, 40), 4, PAINT_UNMASKED)
        assert paint[40, 50] == PAINT_UNMASKED
        assert paint[0, 0] == PAINT_MASKED


class TestCoordinateConsistency:
    """ Paint layer indexing is row = y, col = x. """

    def test_paint_at_specific_point(self):
        w, h = 200, 150  # Non-square to catch axis swaps
        paint = np.zeros((h, w), dtype=np.uint8)
        paintBrushSegment(paint, None, (180, 10), 5, PAINT_MASKED)
        assert paint[10, 180] == PAINT_MASKED
        assert paint[10, 0] == 0
        assert paint.shape == (h, w)
