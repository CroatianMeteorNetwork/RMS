"""Tests for brush mask painting logic (no GUI dependencies)."""

import numpy as np
import cv2
import pytest


def generate_mask_image(mask_polygons, mask_paint_layer, img_width, img_height):
    """Reimplementation of SkyFit2.generateMaskImage for testing."""
    mask = np.full((img_height, img_width), 255, dtype=np.uint8)

    for polygon in mask_polygons:
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 0)

    if mask_paint_layer is not None:
        mask[mask_paint_layer == 1] = 0
        mask[mask_paint_layer == 2] = 255

    return mask


def overlay_from_mask(mask_polygons, mask_paint_layer, img_width, img_height):
    """Reimplementation of SkyFit2.updateMaskOverlayImage logic for testing."""
    mask_img = np.zeros((img_height, img_width), dtype=np.uint8)

    for polygon in mask_polygons:
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask_img, [pts], 1)

    if mask_paint_layer is not None:
        mask_img[mask_paint_layer == 1] = 1
        mask_img[mask_paint_layer == 2] = 0

    return mask_img


def compute_residuals(mask_img, mask_polygons):
    """Reimplementation of loadMaskFromFile residual detection."""
    polygon_mask = np.full_like(mask_img, 255)
    for polygon in mask_polygons:
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(polygon_mask, [pts], 0)

    img_height, img_width = mask_img.shape[:2]
    paint_layer = np.zeros((img_height, img_width), dtype=np.uint8)
    paint_layer[(mask_img == 0) & (polygon_mask == 255)] = 1
    paint_layer[(mask_img == 255) & (polygon_mask == 0)] = 2

    if np.any(paint_layer != 0):
        return paint_layer
    return None


class TestPaintLayerCompositing:
    """Test that paint layer correctly overrides polygon mask."""

    def setup_method(self):
        self.width = 100
        self.height = 80

    def test_polygon_only(self):
        polygons = [[(10, 10), (50, 10), (50, 50), (10, 50)]]
        mask = generate_mask_image(polygons, None, self.width, self.height)
        assert mask[10, 10] == 0
        assert mask[30, 30] == 0
        assert mask[0, 0] == 255
        assert mask[70, 90] == 255

    def test_brush_mask_only(self):
        paint = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.circle(paint, (60, 40), 10, 1, -1)
        mask = generate_mask_image([], paint, self.width, self.height)
        assert mask[40, 60] == 0
        assert mask[0, 0] == 255

    def test_brush_erase_inside_polygon(self):
        polygons = [[(0, 0), (99, 0), (99, 79), (0, 79)]]
        paint = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.circle(paint, (50, 40), 10, 2, -1)

        mask = generate_mask_image(polygons, paint, self.width, self.height)
        assert mask[40, 50] == 255
        assert mask[0, 0] == 0

    def test_brush_overrides_polygon(self):
        polygons = [[(20, 20), (40, 20), (40, 40), (20, 40)]]
        paint = np.zeros((self.height, self.width), dtype=np.uint8)
        paint[25, 25] = 2

        mask = generate_mask_image(polygons, paint, self.width, self.height)
        assert mask[25, 25] == 255
        assert mask[30, 30] == 0

    def test_empty_paint_layer_no_effect(self):
        polygons = [[(10, 10), (50, 10), (50, 50), (10, 50)]]
        paint = np.zeros((self.height, self.width), dtype=np.uint8)
        mask_with = generate_mask_image(polygons, paint, self.width, self.height)
        mask_without = generate_mask_image(polygons, None, self.width, self.height)
        np.testing.assert_array_equal(mask_with, mask_without)


class TestOverlayConsistency:
    """Test that overlay display matches the saved mask."""

    def test_overlay_matches_mask(self):
        w, h = 100, 80
        polygons = [[(10, 10), (50, 10), (50, 50), (10, 50)]]
        paint = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(paint, (70, 60), 8, 1, -1)
        paint[30, 30] = 2

        mask = generate_mask_image(polygons, paint, w, h)
        overlay = overlay_from_mask(polygons, paint, w, h)

        masked_pixels = (mask == 0)
        overlay_pixels = (overlay == 1)
        np.testing.assert_array_equal(masked_pixels, overlay_pixels)


class TestUndoSystem:
    """Test the undo snapshot logic."""

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
    """Test loadMaskFromFile residual logic."""

    def test_pure_polygon_mask_no_residual(self):
        w, h = 100, 80
        polygons = [[(10, 10), (50, 10), (50, 50), (10, 50)]]
        mask = generate_mask_image(polygons, None, w, h)

        # Simulate load: find contours, simplify
        inverted = cv2.bitwise_not(mask)
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        loaded_polygons = []
        for contour in contours:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
            if len(points) >= 3:
                loaded_polygons.append(points)

        residual = compute_residuals(mask, loaded_polygons)
        # Rectangle should round-trip perfectly
        assert residual is None

    def test_brush_strokes_create_residual(self):
        w, h = 100, 80
        paint = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(paint, (50, 40), 15, 1, -1)
        mask = generate_mask_image([], paint, w, h)

        inverted = cv2.bitwise_not(mask)
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        loaded_polygons = []
        for contour in contours:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
            if len(points) >= 3:
                loaded_polygons.append(points)

        # Round-trip via residual detection must recover the original mask exactly
        residual = compute_residuals(mask, loaded_polygons)
        remask = generate_mask_image(loaded_polygons, residual, w, h)
        np.testing.assert_array_equal(mask, remask,
            err_msg="Brush stroke round-trip via residual detection lost pixels")

    def test_erase_inside_polygon_creates_residual(self):
        w, h = 100, 80
        polygons = [[(0, 0), (99, 0), (99, 79), (0, 79)]]
        paint = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(paint, (50, 40), 10, 2, -1)
        mask = generate_mask_image(polygons, paint, w, h)

        # Load and reconstruct
        inverted = cv2.bitwise_not(mask)
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        loaded_polygons = []
        for contour in contours:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
            if len(points) >= 3:
                loaded_polygons.append(points)

        residual = compute_residuals(mask, loaded_polygons)
        # The erased hole will be captured as a contour, but the polygon
        # approximation of the hole boundary may differ slightly
        # In either case, the overall mask should be reconstructable
        remask = generate_mask_image(loaded_polygons, residual, w, h)
        np.testing.assert_array_equal(mask, remask)


class TestCoordinateConsistency:
    """Test that paint layer coordinates match the image frame."""

    def test_paint_at_specific_point(self):
        w, h = 200, 150  # Non-square to catch axis swaps
        paint = np.zeros((h, w), dtype=np.uint8)
        # Paint at x=180, y=10 (near right edge, near top)
        cv2.circle(paint, (180, 10), 5, 1, -1)
        assert paint[10, 180] == 1
        assert paint[10, 0] == 0
        # Verify array indexing: row=y, col=x
        assert paint.shape == (h, w)

    def test_line_connects_points(self):
        w, h = 100, 80
        paint = np.zeros((h, w), dtype=np.uint8)
        cv2.line(paint, (10, 40), (90, 40), 1, thickness=4)
        assert paint[40, 50] == 1
        assert paint[0, 50] == 0
