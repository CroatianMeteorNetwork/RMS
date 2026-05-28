"""Clear Sky Area Detection Utility.

Detects clear sky areas in video frames by checking if expected catalog stars are visible.
Uses a catalog-first approach with Voronoi-style regions where each star "controls" a cell.
"""

from __future__ import absolute_import, division, print_function

import argparse
import datetime
import os
import re
import sys

import cv2
import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter
from scipy.spatial import Voronoi

from RMS.Astrometry.Conversions import date2JD
from RMS.ConfigReader import findConfigInDir
from RMS.Formats import StarCatalog
from RMS.Formats.Platepar import Platepar, getCatalogStarsImagePositions


class ClearSkyResult:
    """Detection result for a single frame."""

    def __init__(self):
        self.jd = None
        self.frame_num = None

        # Overall metrics
        self.clear_sky_ratio = 0.0
        self.area_weighted_ratio = 0.0
        self.detected_count = 0
        self.expected_count = 0
        self.is_clear = False

        # Per-star details
        self.star_detected = None
        self.star_intensities = None
        self.star_positions = None
        self.star_mags = None

        # Regional info
        self.region_clear = None
        self.region_areas = None

        # Lazy Voronoi computation data
        self._voronoi = None
        self._valid_regions = None
        self._x_valid = None
        self._y_valid = None
        self._img_shape = None
        self._detector = None

    def _ensureVoronoi(self):
        """Build Voronoi tessellation lazily on first mask request."""
        if self._voronoi is not None:
            return True

        if self._x_valid is None or len(self._x_valid) < 4:
            return False

        if self._detector is None:
            return False

        vor, valid_regions, region_areas = self._detector._buildVoronoi(
            self._x_valid, self._y_valid, self._img_shape
        )

        if vor is None:
            return False

        self._voronoi = vor
        self._valid_regions = valid_regions
        self.region_areas = region_areas
        self.region_clear = self.star_detected

        # Compute area-weighted ratio now
        if region_areas is not None and np.sum(region_areas) > 0:
            detected_area = np.sum(region_areas[self.star_detected])
            total_area = np.sum(region_areas)
            self.area_weighted_ratio = detected_area / total_area

        return True

    def getClearRegionMask(self, img_shape):
        """Generate binary mask of clear sky regions.

        Args:
            img_shape: Tuple (height, width) of the output mask

        Returns:
            2D numpy array (uint8) where 255 = clear, 0 = not clear
        """
        if self.star_detected is None:
            return np.zeros(img_shape[:2], dtype=np.uint8)

        self._ensureVoronoi()

        if self._voronoi is None:
            return np.zeros(img_shape[:2], dtype=np.uint8)

        return self._generateRegionMask(img_shape, self.star_detected)

    def getCloudyRegionMask(self, img_shape):
        """Generate binary mask of cloudy regions.

        Args:
            img_shape: Tuple (height, width) of the output mask

        Returns:
            2D numpy array (uint8) where 255 = cloudy, 0 = clear
        """
        if self.star_detected is None:
            return np.ones(img_shape[:2], dtype=np.uint8) * 255

        self._ensureVoronoi()

        if self._voronoi is None:
            return np.ones(img_shape[:2], dtype=np.uint8) * 255

        return self._generateRegionMask(img_shape, ~self.star_detected)

    def _generateRegionMask(self, img_shape, region_flags):
        """Generate binary mask for specified regions.

        Args:
            img_shape: Tuple (height, width)
            region_flags: Boolean array indicating which regions to include

        Returns:
            2D numpy array (uint8)
        """
        import cv2

        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if self._voronoi is None or self._valid_regions is None:
            return mask

        for i, region_idx in enumerate(self._valid_regions):
            if not region_flags[i]:
                continue

            region = self._voronoi.regions[region_idx]
            if -1 in region or len(region) == 0:
                continue

            # Get polygon vertices
            vertices = self._voronoi.vertices[region]

            # Clip to image bounds and convert to integer
            vertices = np.clip(vertices, 0, [w - 1, h - 1])
            pts = vertices.astype(np.int32).reshape((-1, 1, 2))

            cv2.fillPoly(mask, [pts], 255)

        return mask


class ClearSkyDetector:
    """High-performance clear sky detection using catalog star matching."""

    def __init__(
        self,
        platepar,
        config,
        mask=None,
        lim_mag=None,
        intensity_threshold=15,
        min_snr=3.0,
        clear_threshold=0.7,
    ):
        """Initialize the clear sky detector.

        Args:
            platepar: Platepar with astrometric calibration
            config: RMS config object
            mask: Optional MaskStructure for excluding regions
            lim_mag: Expected limiting magnitude (stars brighter than this should be visible)
            intensity_threshold: Min pixel intensity above background to count as detected
            min_snr: Minimum signal-to-noise ratio for detection
            clear_threshold: Ratio threshold to consider sky "clear" (default 0.7)
        """
        self.platepar = platepar
        self.config = config
        self.mask = mask
        self.lim_mag = lim_mag if lim_mag is not None else config.catalog_mag_limit
        self.intensity_threshold = intensity_threshold
        self.min_snr = min_snr
        self.clear_threshold = clear_threshold

        # Load and cache catalog stars
        self.catalog_stars = self._loadCatalog()

        # Voronoi tessellation (built on first use with actual positions)
        self.voronoi = None
        self.region_areas = None
        self._last_jd = None
        self._last_positions = None

    def _loadCatalog(self):
        """Load catalog stars brighter than limiting magnitude.

        Returns:
            ndarray of shape (N, 3) with columns (RA, Dec, Mag) in degrees
        """
        catalog_file = self.config.star_catalog_file
        catalog_path = self.config.star_catalog_path

        # Find the actual catalog file (handle different naming conventions)
        if not os.path.isfile(os.path.join(catalog_path, catalog_file)):
            # Try with .bin extension
            for ext in ['.bin', '_LM9.0_v1.bin', '_LM12.0.bin']:
                test_file = catalog_file + ext
                if os.path.isfile(os.path.join(catalog_path, test_file)):
                    catalog_file = test_file
                    break

        # Read the catalog
        result = StarCatalog.readStarCatalog(
            catalog_path, catalog_file, lim_mag=self.lim_mag
        )

        # Handle different return formats
        if len(result) == 3:
            catalog_stars, _, _ = result
        else:
            catalog_stars = result[0]

        return catalog_stars

    def _buildVoronoi(self, x_cat, y_cat, img_shape):
        """Build Voronoi tessellation from projected star positions.

        Args:
            x_cat: X coordinates of catalog stars
            y_cat: Y coordinates of catalog stars
            img_shape: Image shape (height, width) for bounding

        Returns:
            Voronoi object, valid_regions list, and region_areas array
        """
        if len(x_cat) < 4:
            return None, None, None

        # Stack coordinates
        points = np.column_stack([x_cat, y_cat])

        # Add bounding box points to ensure all regions are finite
        h, w = img_shape[:2]
        margin = max(h, w) * 2
        bbox_points = np.array([
            [-margin, -margin],
            [-margin, h + margin],
            [w + margin, -margin],
            [w + margin, h + margin],
        ])
        all_points = np.vstack([points, bbox_points])

        try:
            vor = Voronoi(all_points)
        except Exception:
            return None, None, None

        # Only consider regions for actual stars (not bounding box points)
        n_stars = len(x_cat)
        valid_regions = []
        region_areas = []

        for i in range(n_stars):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]

            if -1 in region or len(region) == 0:
                # Unbounded region, use a default area
                valid_regions.append(region_idx)
                region_areas.append(0.0)
                continue

            # Compute area of the polygon
            vertices = vor.vertices[region]
            # Clip to image bounds
            vertices = np.clip(vertices, 0, [w - 1, h - 1])
            area = self._polygonArea(vertices)

            valid_regions.append(region_idx)
            region_areas.append(area)

        return vor, valid_regions, np.array(region_areas)

    def _polygonArea(self, vertices):
        """Compute area of a polygon using the shoelace formula.

        Args:
            vertices: Nx2 array of (x, y) coordinates

        Returns:
            Area of the polygon
        """
        n = len(vertices)
        if n < 3:
            return 0.0

        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i, 0] * vertices[j, 1]
            area -= vertices[j, 0] * vertices[i, 1]

        return abs(area) / 2.0

    def detectFrame(self, frame, jd, background=None):
        """Detect clear sky in a single frame.

        Args:
            frame: 2D numpy array (grayscale image)
            jd: Julian date of the frame
            background: Optional background image or scalar to subtract

        Returns:
            ClearSkyResult with detection results
        """
        return detectFrame(self, frame, jd, background)

    def _parseVideoTime(self, video_path):
        """Parse beginning datetime from video filename."""
        filename = os.path.basename(video_path)
        filename_noext = os.path.splitext(filename)[0]

        patterns = [
            r'(\d{8})_(\d{6})(?:\.(\d+))?',
            r'(\d{8})-(\d{6})',
            r'[A-Z0-9]+_(\d{8})_(\d{6})_(\d+)',
            r'[A-Z0-9]+_(\d{8})-(\d{6})',
        ]

        for pattern in patterns:
            match = re.search(pattern, filename_noext)
            if match:
                groups = match.groups()
                date_str = groups[0]
                time_str = groups[1]
                microseconds = int(groups[2]) if len(groups) > 2 and groups[2] else 0
                dt = datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
                dt = dt.replace(microsecond=microseconds)
                return dt

        # Fallback
        from RMS.Astrometry.Conversions import jd2Date
        dt_tuple = jd2Date(self.platepar.JD)
        return datetime.datetime(*[int(x) for x in dt_tuple[:6]])

    def processVideo(self, video_path, start_frame=0, end_frame=None,
                     frame_step=1, callback=None, frametimes=None):
        """Process entire video file.

        Args:
            video_path: Path to MP4/AVI video file
            start_frame: First frame to process
            end_frame: Last frame (None = all)
            frame_step: Process every Nth frame
            callback: Optional function(frame_num, result) called per frame
            frametimes: Optional dict mapping frame_num -> datetime

        Returns:
            List of ClearSkyResult, one per processed frame
        """
        return processVideo(self, video_path, start_frame, end_frame,
                           frame_step, callback, frametimes)


def clipPolygonToRect(polygon, x_min, y_min, x_max, y_max):
    """Clip a polygon to a rectangle using Sutherland-Hodgman algorithm.

    Args:
        polygon: Nx2 array of (x, y) vertices
        x_min, y_min, x_max, y_max: Rectangle bounds

    Returns:
        Clipped polygon as Nx2 array, or None if completely outside
    """
    def inside(p, edge):
        x, y = p
        if edge == 'left':
            return x >= x_min
        elif edge == 'right':
            return x <= x_max
        elif edge == 'bottom':
            return y >= y_min
        elif edge == 'top':
            return y <= y_max

    def intersection(p1, p2, edge):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1

        if edge == 'left':
            t = (x_min - x1) / dx if dx != 0 else 0
            return (x_min, y1 + t * dy)
        elif edge == 'right':
            t = (x_max - x1) / dx if dx != 0 else 0
            return (x_max, y1 + t * dy)
        elif edge == 'bottom':
            t = (y_min - y1) / dy if dy != 0 else 0
            return (x1 + t * dx, y_min)
        elif edge == 'top':
            t = (y_max - y1) / dy if dy != 0 else 0
            return (x1 + t * dx, y_max)

    def clipEdge(poly, edge):
        if len(poly) == 0:
            return []

        result = []
        for i in range(len(poly)):
            curr = poly[i]
            prev = poly[i - 1]

            if inside(curr, edge):
                if not inside(prev, edge):
                    result.append(intersection(prev, curr, edge))
                result.append(curr)
            elif inside(prev, edge):
                result.append(intersection(prev, curr, edge))

        return result

    # Convert to list of tuples
    poly = [(p[0], p[1]) for p in polygon]

    # Clip against each edge
    for edge in ['left', 'right', 'bottom', 'top']:
        poly = clipEdge(poly, edge)
        if len(poly) == 0:
            return None

    if len(poly) < 3:
        return None

    return np.array(poly)


def processVideo(
    detector,
    video_path,
    start_frame=0,
    end_frame=None,
    frame_step=1,
    callback=None,
    frametimes=None,
):
    """Process entire video file.

    Args:
        detector: ClearSkyDetector instance
        video_path: Path to MP4/AVI video file
        start_frame: First frame to process
        end_frame: Last frame (None = all)
        frame_step: Process every Nth frame (for speed)
        callback: Optional function(frame_num, result) called per frame
        frametimes: Optional dict mapping frame_num -> datetime

    Returns:
        List of ClearSkyResult, one per processed frame
    """
    # Open video with cv2 directly
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if end_frame is None:
        end_frame = total_frames

    # Determine time source
    if frametimes:
        use_frametimes = True
    else:
        use_frametimes = False
        start_dt, end_dt = _parseTimelapseRange(video_path)
        if start_dt and end_dt:
            total_duration = (end_dt - start_dt).total_seconds()
            seconds_per_frame = total_duration / total_frames
        else:
            start_dt = detector._parseVideoTime(video_path)
            seconds_per_frame = 1.0 / fps

    results = []

    for frame_num in range(start_frame, min(end_frame, total_frames), frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = frame[:, :, 1]
        else:
            gray = frame

        # Get JD for this frame
        if use_frametimes and frame_num in frametimes:
            frame_time = frametimes[frame_num]
        else:
            frame_time = start_dt + datetime.timedelta(seconds=frame_num * seconds_per_frame)

        jd = frametimeToJD(frame_time)

        # Detect clear sky
        result = detector.detectFrame(gray, jd)
        result.frame_num = frame_num

        results.append(result)

        if callback:
            callback(frame_num, result)

    cap.release()
    return results


# Continue ClearSkyDetector class methods as module-level functions that take detector as first arg
# This was refactored - the class methods are defined in the class above

def _parseTimelapseRange(video_path):
    """Parse start and end times from timelapse filename.

    Expected format: StationID_YYYYMMDD-HHMMSS_to_YYYYMMDD-HHMMSS_..._timelapse.mp4
    """
    filename = os.path.basename(video_path)

    match = re.search(r'(\d{8})-(\d{6})_to_(\d{8})-(\d{6})', filename)
    if match:
        start_date, start_time, end_date, end_time = match.groups()
        start_dt = datetime.datetime.strptime(start_date + start_time, "%Y%m%d%H%M%S")
        end_dt = datetime.datetime.strptime(end_date + end_time, "%Y%m%d%H%M%S")
        return start_dt, end_dt

    return None, None


def detectFrame(detector, frame, jd, background=None):
    """Detect clear sky in a single frame.

    Args:
        detector: ClearSkyDetector instance
        frame: 2D numpy array (grayscale image)
        jd: Julian date of the frame
        background: Optional background image or scalar to subtract

    Returns:
        ClearSkyResult with detection results
    """
    result = ClearSkyResult()
    result.jd = jd

    # Project catalog stars to image coordinates
    x_cat, y_cat, mag_cat = getCatalogStarsImagePositions(
        detector.catalog_stars, jd, detector.platepar
    )

    # Filter to stars within image bounds
    h, w = frame.shape[:2]
    valid = (x_cat >= 0) & (x_cat < w) & (y_cat >= 0) & (y_cat < h)

    # Apply mask if available
    if detector.mask is not None and detector.mask.img is not None:
        x_int = np.clip(x_cat.astype(int), 0, w - 1)
        y_int = np.clip(y_cat.astype(int), 0, h - 1)
        valid &= detector.mask.img[y_int, x_int] > 0

    x_valid = x_cat[valid]
    y_valid = y_cat[valid]
    mag_valid = mag_cat[valid]

    result.expected_count = len(x_valid)
    result.star_positions = np.column_stack([x_valid, y_valid])
    result.star_mags = mag_valid

    if result.expected_count == 0:
        return result

    # Fast background estimation if not provided
    if background is None:
        corners = [
            frame[0, 0], frame[0, w-1],
            frame[h-1, 0], frame[h-1, w-1],
            frame[h//2, w//2]
        ]
        background = float(np.median(corners))

    # Star detection using LOCAL CONTRAST (fast vectorized version)
    x_int = np.clip(x_valid.astype(np.intp), 0, w - 1)
    y_int = np.clip(y_valid.astype(np.intp), 0, h - 1)

    # Compute local max (3x3) and local background (9x9 mean) for whole image ONCE
    frame_f = frame.astype(np.float32)
    local_max = maximum_filter(frame_f, size=3)
    local_bg = uniform_filter(frame_f, size=9)

    # Sample at star positions (vectorized - very fast)
    center_vals = local_max[y_int, x_int]
    bg_vals = local_bg[y_int, x_int]

    # Contrast: how much brighter is center than local background
    contrast = center_vals - bg_vals
    result.star_intensities = contrast

    # Detection: star must have significant contrast above local background
    result.star_detected = contrast > detector.intensity_threshold
    result.detected_count = np.sum(result.star_detected)
    result.clear_sky_ratio = result.detected_count / result.expected_count

    # Store data for lazy Voronoi computation
    result._x_valid = x_valid
    result._y_valid = y_valid
    result._img_shape = frame.shape
    result._detector = detector

    result.is_clear = result.clear_sky_ratio > detector.clear_threshold

    return result


def loadFrametimes(frametimes_path):
    """Load frame timestamps from JSON file.

    Args:
        frametimes_path: Path to JSON file with frame_num -> timestamp mapping
                        Format: {"0": "YYYYMMDD_HHMMSS_mmm_suffix", ...}

    Returns:
        Dict mapping frame number (int) to datetime object
    """
    import json

    with open(frametimes_path, 'r') as f:
        data = json.load(f)

    frametimes = {}
    for frame_str, timestamp_str in data.items():
        frame_num = int(frame_str)

        # Parse format: YYYYMMDD_HHMMSS_mmm_suffix
        # Example: 20260125_013450_005_n
        parts = timestamp_str.split('_')
        date_str = parts[0]  # YYYYMMDD
        time_str = parts[1]  # HHMMSS
        ms_str = parts[2]    # milliseconds

        dt = datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
        dt = dt.replace(microsecond=int(ms_str) * 1000)
        frametimes[frame_num] = dt

    return frametimes


def frametimeToJD(dt):
    """Convert datetime to Julian Date."""
    return date2JD(
        dt.year, dt.month, dt.day,
        dt.hour, dt.minute, dt.second,
        dt.microsecond / 1000
    )


def createOverlayVideo(
    detector,
    video_path,
    output_path,
    start_frame=0,
    end_frame=None,
    frame_step=1,
    show_text=True,
    cell_alpha=0.3,
    frametimes=None,
):
    """Create video with Voronoi cell overlay.

    Args:
        detector: ClearSkyDetector instance
        video_path: Input video path
        output_path: Output video path
        start_frame: First frame to process
        end_frame: Last frame (None = all)
        frame_step: Process every Nth frame
        show_text: Whether to show detection stats text
        cell_alpha: Transparency of cell overlay (0-1)
        frametimes: Optional dict mapping frame_num -> datetime (from loadFrametimes)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if end_frame is None:
        end_frame = total_frames

    # Determine time source
    if frametimes:
        print(f"Using frametimes JSON ({len(frametimes)} entries)")
        use_frametimes = True
    else:
        use_frametimes = False
        # Check if this is a timelapse video with time range in filename
        start_dt, end_dt = _parseTimelapseRange(video_path)
        if start_dt and end_dt:
            total_duration = (end_dt - start_dt).total_seconds()
            seconds_per_frame = total_duration / total_frames
            print(f"Timelapse detected: {start_dt} to {end_dt} ({total_duration/3600:.1f} hours)")
        else:
            start_dt = detector._parseVideoTime(video_path)
            seconds_per_frame = 1.0 / fps
            print(f"Regular video starting at {start_dt}")

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_fps = fps / frame_step
    out = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))

    # Prepare mask overlay if available (darken masked regions)
    mask_overlay = None
    if detector.mask is not None and detector.mask.img is not None:
        # Create a 3-channel mask: masked areas will be darkened
        mask_img = detector.mask.img
        mask_overlay = np.ones((height, width, 3), dtype=np.uint8) * 255
        # Set masked areas (where mask is 0) to dark gray
        mask_overlay[mask_img == 0] = [40, 40, 40]
        print(f"Mask overlay enabled - masked regions will be darkened")

    print(f"Creating overlay video: {output_path}")
    print(f"Processing frames {start_frame} to {end_frame}, step {frame_step}")

    frame_count = 0
    for frame_num in range(start_frame, min(end_frame, total_frames), frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        # Get grayscale for detection
        gray = frame[:, :, 1] if len(frame.shape) == 3 else frame

        # Get frame time and JD
        if use_frametimes and frame_num in frametimes:
            frame_time = frametimes[frame_num]
            jd = frametimeToJD(frame_time)
        else:
            frame_time = start_dt + datetime.timedelta(seconds=frame_num * seconds_per_frame)
            jd = frametimeToJD(frame_time)

        # Detect
        result = detector.detectFrame(gray, jd)

        # Build Voronoi for this frame and draw cells
        if result.star_positions is not None and len(result.star_positions) >= 4:
            result._ensureVoronoi()

            if result._voronoi is not None:
                # Create overlay for cells
                overlay = frame.copy()

                for i, region_idx in enumerate(result._valid_regions):
                    region = result._voronoi.regions[region_idx]
                    if -1 in region or len(region) == 0:
                        continue

                    vertices = result._voronoi.vertices[region]

                    # Proper polygon clipping to image bounds
                    clipped = clipPolygonToRect(vertices, 0, 0, width - 1, height - 1)
                    if clipped is None or len(clipped) < 3:
                        continue

                    pts = clipped.astype(np.int32).reshape((-1, 1, 2))

                    if result.star_detected[i]:
                        color = (0, 180, 0)  # Green for clear (star detected)
                    else:
                        color = (0, 0, 180)  # Red for cloudy (star missing)

                    cv2.fillPoly(overlay, [pts], color)

                # Apply mask to overlay - clear the overlay in masked regions
                if mask_overlay is not None:
                    # Where mask is 0 (masked), use original frame instead of overlay
                    mask_3ch = np.stack([detector.mask.img] * 3, axis=-1)
                    overlay = np.where(mask_3ch > 0, overlay, frame)

                # Blend overlay with original frame
                frame = cv2.addWeighted(overlay, cell_alpha, frame, 1 - cell_alpha, 0)

        # Apply mask darkening on top
        if mask_overlay is not None:
            # Darken masked regions
            mask_3ch = np.stack([detector.mask.img] * 3, axis=-1)
            frame = np.where(mask_3ch > 0, frame, (frame * 0.3).astype(np.uint8))

            # Draw star markers on top of cells (only for unmasked stars)
            for i, (x, y) in enumerate(result.star_positions):
                x, y = int(x), int(y)
                if result.star_detected[i]:
                    # Green filled circle for detected star
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                else:
                    # Red hollow circle for catalog position (star not detected)
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), 1)
        elif result.star_positions is not None:
            # Draw star markers on top of cells
            for i, (x, y) in enumerate(result.star_positions):
                x, y = int(x), int(y)
                if result.star_detected[i]:
                    # Green filled circle for detected star
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                else:
                    # Red hollow circle for catalog position (star not detected)
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), 1)

        # Add text overlay
        if show_text:
            status = "CLEAR" if result.is_clear else "CLOUDY"
            time_str = frame_time.strftime("%H:%M:%S")
            text = f"{time_str} - {result.clear_sky_ratio:.1%} ({result.detected_count}/{result.expected_count}) - {status}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        out.write(frame)
        frame_count += 1

        if frame_count % 50 == 0:
            print(f"  Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Done. Wrote {frame_count} frames to {output_path}")


def findClearIntervals(results, min_consecutive=1):
    """Find intervals of consecutive clear frames.

    Args:
        results: List of ClearSkyResult objects
        min_consecutive: Minimum number of consecutive clear frames

    Returns:
        List of tuples (start_frame, end_frame) for each clear interval
    """
    intervals = []
    start = None

    for i, result in enumerate(results):
        if result.is_clear:
            if start is None:
                start = result.frame_num
        else:
            if start is not None:
                # End of clear interval
                end = results[i - 1].frame_num
                if (end - start) >= (min_consecutive - 1):
                    intervals.append((start, end))
                start = None

    # Handle case where video ends during clear interval
    if start is not None:
        end = results[-1].frame_num
        if (end - start) >= (min_consecutive - 1):
            intervals.append((start, end))

    return intervals


def formatFrameAsTime(frame_num, fps):
    """Convert frame number to HH:MM:SS format.

    Args:
        frame_num: Frame number
        fps: Frames per second

    Returns:
        String in HH:MM:SS format
    """
    total_seconds = frame_num / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def discoverFiles(directory):
    """Auto-discover required files in a directory.

    Looks for:
    - platepar_cmn2010.cal or platepar*.cal
    - .config
    - mask.bmp
    - *_timelapse.mp4 video file
    - matching *_frametimes.json file

    Args:
        directory: Path to directory containing the files

    Returns:
        Dict with keys: video_path, platepar, config, mask, frametimes
        Values are None if not found
    """
    import glob

    result = {
        'video_path': None,
        'platepar': None,
        'config': None,
        'mask': None,
        'frametimes': None,
    }

    # Find platepar (prefer platepar_cmn2010.cal)
    platepar_path = os.path.join(directory, 'platepar_cmn2010.cal')
    if os.path.isfile(platepar_path):
        result['platepar'] = platepar_path
    else:
        # Try any platepar*.cal file
        platepars = glob.glob(os.path.join(directory, 'platepar*.cal'))
        if platepars:
            result['platepar'] = platepars[0]

    # Find config
    config_path = findConfigInDir(directory)
    if config_path is not None:
        result['config'] = config_path

    # Find mask
    mask_path = os.path.join(directory, 'mask.bmp')
    if os.path.isfile(mask_path):
        result['mask'] = mask_path

    # Find timelapse video
    timelapse_videos = glob.glob(os.path.join(directory, '*_timelapse.mp4'))
    if timelapse_videos:
        result['video_path'] = timelapse_videos[0]

        # Find matching frametimes JSON
        video_basename = os.path.basename(result['video_path'])
        # Extract the base name pattern (StationID_YYYYMMDD-HHMMSS_to_YYYYMMDD-HHMMSS)
        match = re.match(r'(.+?)_frames_timelapse\.mp4$', video_basename)
        if match:
            base_pattern = match.group(1)
            frametimes_path = os.path.join(directory, f'{base_pattern}_frametimes.json')
            if os.path.isfile(frametimes_path):
                result['frametimes'] = frametimes_path

    return result


def main():
    """Command-line interface for clear sky detection."""
    parser = argparse.ArgumentParser(
        description="Detect clear sky areas in video frames using catalog star matching.",
        epilog="Example: python -m RMS.ClearSkyDetector /path/to/directory/"
    )
    parser.add_argument(
        "path",
        help="Directory containing video and calibration files, or path to video file"
    )
    parser.add_argument(
        "-p", "--platepar",
        help="Path to platepar (auto-discovered if directory given)"
    )
    parser.add_argument(
        "-c", "--config",
        help="Path to .config file (auto-discovered if directory given)"
    )
    parser.add_argument(
        "-m", "--mask",
        help="Path to mask.bmp (auto-discovered if directory given)"
    )
    parser.add_argument(
        "-f", "--frametimes",
        help="Path to frametimes JSON (auto-discovered if directory given)"
    )
    parser.add_argument(
        "--lim-mag",
        type=float,
        default=4.0,
        help="Limiting magnitude for star detection (default: 4.0)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Clear sky ratio threshold (default: 0.7)",
    )
    parser.add_argument(
        "--intensity",
        type=float,
        default=15,
        help="Minimum intensity above background (default: 15)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1, all frames)",
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Starting frame number (default: 0)"
    )
    parser.add_argument(
        "--end", type=int, default=None, help="Ending frame number (default: all)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output per frame"
    )
    parser.add_argument(
        "-o", "--overlay", action="store_true",
        help="Create overlay video (saved as clearsky_overlay.mp4 in same directory)"
    )

    args = parser.parse_args()

    # Determine if path is a directory or video file
    if os.path.isdir(args.path):
        # Auto-discover files from directory
        discovered = discoverFiles(args.path)
        work_dir = args.path

        video_path = discovered['video_path']
        platepar_path = args.platepar or discovered['platepar']
        config_path = args.config or discovered['config']
        mask_path = args.mask or discovered['mask']
        frametimes_path = args.frametimes or discovered['frametimes']

        if video_path is None:
            print(f"Error: No *_timelapse.mp4 video found in {args.path}")
            sys.exit(1)
        if platepar_path is None:
            print(f"Error: No platepar*.cal file found in {args.path}")
            sys.exit(1)
    else:
        # Treat as video file path
        video_path = args.path
        work_dir = os.path.dirname(video_path)
        platepar_path = args.platepar
        config_path = args.config
        mask_path = args.mask
        frametimes_path = args.frametimes

        if platepar_path is None:
            # Try to auto-discover in video directory
            discovered = discoverFiles(work_dir)
            platepar_path = discovered['platepar']
            if config_path is None:
                config_path = discovered['config']
            if mask_path is None:
                mask_path = discovered['mask']
            if frametimes_path is None:
                frametimes_path = discovered['frametimes']

        if platepar_path is None:
            print("Error: No platepar found. Use -p to specify.")
            sys.exit(1)

    # Print discovered files
    print(f"Camera settings file: {config_path or '(default)'}")
    if mask_path:
        print(f"Mask file: {mask_path}")
    if frametimes_path:
        print(f"Frametimes file: {frametimes_path}")

    # Load config
    from RMS.ConfigReader import Config, parse as parseConfig

    if config_path and os.path.isfile(config_path):
        config = parseConfig(config_path)
    else:
        config = Config()

    # Override limiting magnitude
    config.catalog_mag_limit = args.lim_mag

    # Load platepar
    platepar = Platepar()
    platepar.read(platepar_path)

    # Load mask if found
    mask = None
    if mask_path:
        from RMS.Routines.MaskImage import loadMask
        mask = loadMask(mask_path)
        if mask is not None:
            print(f"Loaded mask: {mask_path} ({mask.width}x{mask.height})")
        else:
            print(f"Warning: Could not load mask from {mask_path}")

    # Initialize detector
    detector = ClearSkyDetector(
        platepar,
        config,
        mask=mask,
        lim_mag=args.lim_mag,
        intensity_threshold=args.intensity,
        clear_threshold=args.threshold,
    )

    # Get video info for reporting using cv2 directly
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        sys.exit(1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    frames_to_process = len(
        range(args.start, args.end or total_frames, args.step)
    )

    print(f"Processing {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps:.1f}")
    print(f"Processing every {args.step}th frame ({frames_to_process} frames)")
    print(f"Limiting magnitude: {args.lim_mag}, Clear threshold: {args.threshold}")
    print()

    # Load frametimes if available
    frametimes = None
    if frametimes_path:
        frametimes = loadFrametimes(frametimes_path)

    # Process video
    def progress_callback(frame_num, result):
        if args.verbose:
            status = "CLEAR" if result.is_clear else "CLOUDY"
            print(
                f"Frame {frame_num}: {result.clear_sky_ratio:.1%} clear "
                f"({result.detected_count}/{result.expected_count} stars) - {status}"
            )

    results = detector.processVideo(
        video_path,
        start_frame=args.start,
        end_frame=args.end,
        frame_step=args.step,
        callback=progress_callback,
    )

    # Summary
    if not results:
        print("No frames processed.")
        return

    clear_frames = [r for r in results if r.is_clear]
    clear_count = len(clear_frames)
    total_count = len(results)

    print()
    print(f"Summary: {clear_count}/{total_count} frames clear ({100*clear_count/total_count:.1f}%)")

    # Find and report clear intervals
    intervals = findClearIntervals(results)
    if intervals:
        print("Clear intervals:")
        for start, end in intervals:
            start_time = formatFrameAsTime(start, fps)
            end_time = formatFrameAsTime(end, fps)
            print(f"  {start_time}-{end_time}")
    else:
        print("No clear intervals found.")

    # Create overlay video if requested
    if args.overlay:
        print()
        overlay_path = os.path.join(work_dir, "clearsky_overlay.mp4")
        createOverlayVideo(
            detector,
            video_path,
            overlay_path,
            start_frame=args.start,
            end_frame=args.end,
            frame_step=args.step,
            frametimes=frametimes,
        )


if __name__ == "__main__":
    main()
