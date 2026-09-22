""" Tests for the SkyFit2 mask editing helpers in RMS.Routines.MaskImage (no GUI dependencies). """

from __future__ import print_function, division, absolute_import

import numpy as np
import cv2

from RMS.Routines.MaskImage import compositeMaskLayers, maskRasterResiduals, decomposeMaskImage, \
    paintBrushSegment, PAINT_MASKED, PAINT_UNMASKED


class TestPaintLayerCompositing:
    """ The paint layer must override the polygon fill. """

    def setup_method(self):
        """ Set the size of the test image. """

        self.width = 100
        self.height = 80


    def test_polygon_only(self):
        """ A polygon alone masks its interior and leaves the rest of the image unmasked. """

        polygons = [[(10, 10), (50, 10), (50, 50), (10, 50)]]
        mask = compositeMaskLayers(polygons, None, self.width, self.height)
        assert mask.shape == (self.height, self.width)
        assert mask.dtype == np.uint8
        assert mask[10, 10] == 0
        assert mask[30, 30] == 0
        assert mask[0, 0] == 255
        assert mask[70, 90] == 255


    def test_brush_mask_only(self):
        """ A brush painted disc masks its pixels when there are no polygons. """

        paint = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.circle(paint, (60, 40), 10, PAINT_MASKED, -1)
        mask = compositeMaskLayers([], paint, self.width, self.height)
        assert mask[40, 60] == 0
        assert mask[0, 0] == 255


    def test_brush_erase_inside_polygon(self):
        """ A brush erased disc unmasks pixels which are inside a polygon. """

        polygons = [[(0, 0), (99, 0), (99, 79), (0, 79)]]
        paint = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.circle(paint, (50, 40), 10, PAINT_UNMASKED, -1)

        mask = compositeMaskLayers(polygons, paint, self.width, self.height)
        assert mask[40, 50] == 255
        assert mask[0, 0] == 0


    def test_brush_overrides_polygon(self):
        """ A single erased pixel overrides the polygon underneath it. """

        polygons = [[(20, 20), (40, 20), (40, 40), (20, 40)]]
        paint = np.zeros((self.height, self.width), dtype=np.uint8)
        paint[25, 25] = PAINT_UNMASKED

        mask = compositeMaskLayers(polygons, paint, self.width, self.height)
        assert mask[25, 25] == 255
        assert mask[30, 30] == 0


    def test_empty_paint_layer_no_effect(self):
        """ An all-zero paint layer renders the same mask as no paint layer at all. """

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
        """ The 1/0 overlay marks exactly the pixels which the 0/255 mask masks. """

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
        """ Popping the history restores the paint layer as it was before each stroke. """

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
        """ The history is capped at the maximum undo depth, dropping the oldest snapshots. """

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
        """ A mask made only of a rectangle decomposes back into a polygon with no residuals. """

        w, h = 100, 80
        polygons = [[(10, 10), (50, 10), (50, 50), (10, 50)]]
        mask = compositeMaskLayers(polygons, None, w, h)

        loaded_polygons, residual = decomposeMaskImage(mask)

        # Rectangle should round-trip perfectly
        assert len(loaded_polygons) == 1
        assert residual is None
        assert maskRasterResiduals(mask, loaded_polygons) is None


    def test_brush_strokes_create_residual(self):
        """ A brush painted mask survives the decompose/composite round trip without losing pixels. """

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
        """ A polygon with a brush erased hole survives the decompose/composite round trip. """

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


class TestBrushSegment:
    """ The brush footprint must be the same disc along the whole stroke. """

    def test_first_point_is_a_disc(self):
        """ The first point of a stroke stamps a full disc, not a single pixel. """

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
        """ A brush position far outside the image is clamped to the image edge. """

        paint = np.zeros((80, 100), dtype=np.uint8)
        center = paintBrushSegment(paint, None, (1e6, -1e6), 3, PAINT_MASKED)
        assert center == (99, 0)
        assert paint[0, 99] == PAINT_MASKED


    def test_erase_value(self):
        """ Painting with the erase value unmasks the pixels under the brush. """

        paint = np.zeros((80, 100), dtype=np.uint8)
        paint[:] = PAINT_MASKED
        paintBrushSegment(paint, None, (50, 40), 4, PAINT_UNMASKED)
        assert paint[40, 50] == PAINT_UNMASKED
        assert paint[0, 0] == PAINT_MASKED


class TestCoordinateConsistency:
    """ Paint layer indexing is row = y, col = x. """

    def test_paint_at_specific_point(self):
        """ The (x, y) brush position lands at the [y, x] index of the paint layer. """

        # A non-square image is used to catch axis swaps
        w, h = 200, 150

        paint = np.zeros((h, w), dtype=np.uint8)
        paintBrushSegment(paint, None, (180, 10), 5, PAINT_MASKED)
        assert paint[10, 180] == PAINT_MASKED
        assert paint[10, 0] == 0
        assert paint.shape == (h, w)


class TestMaskResizeConsistency(object):
    """ A mask loaded at a different size must be resized as a raster before it is decomposed. """

    def testDecomposingTheResizedRasterReproducesIt(self):
        """ Polygons plus residuals of the resized raster composite back to exactly that raster. """

        # A rectangle whose edges do not land on a scale boundary
        mask = np.full((80, 100), 255, dtype=np.uint8)
        mask[10:51, 10:51] = 0

        # This is the order loadMaskFromFile uses: resize the raster, then decompose it
        resized = cv2.resize(mask, (200, 160), interpolation=cv2.INTER_NEAREST)
        polygons, paint_layer = decomposeMaskImage(resized)

        composite = compositeMaskLayers(polygons, paint_layer, 200, 160)

        # The editable overlay and the mask handed to star detection agree everywhere
        assert np.array_equal(composite, resized)


    def testScalingPolygonsInsteadLosesBoundaryPixels(self):
        """ Scaling the vertices of the source-size polygons does not reproduce the resized raster.

            This is why the raster is resized first. Kept as a test so the order is not swapped back.
        """

        mask = np.full((80, 100), 255, dtype=np.uint8)
        mask[10:51, 10:51] = 0

        resized = cv2.resize(mask, (200, 160), interpolation=cv2.INTER_NEAREST)

        # Decompose at the source size and scale the polygon vertices, as the old load path did
        polygons, _ = decomposeMaskImage(mask)
        scaled_polygons = [[(x*2.0, y*2.0) for x, y in polygon] for polygon in polygons]

        composite = compositeMaskLayers(scaled_polygons, None, 200, 160)

        assert not np.array_equal(composite, resized)
