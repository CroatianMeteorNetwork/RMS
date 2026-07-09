""" Unit tests for the fireball-detector recall improvements:

    - RMS.VideoExtraction.decontaminateStdpixel: the stdpixel decontamination that
      stops bright fireball trails from suppressing themselves through the threshold.
    - RMS.Routines.Grouping3D.findCoefficients: the FOV-aware deg/s -> px/frame
      velocity cap that replaced the hard-coded "total > 2" filter.

    These are pure-numpy / pure-python paths, so no capture pipeline, Cython threshold
    or Extractor process is needed.
"""

import pytest

np = pytest.importorskip("numpy")

from RMS.VideoExtraction import (
    decontaminateStdpixel,
    BRIGHT_PERCENTILE,
    CONTAMINATION_STD_FACTOR,
    MIN_BACKGROUND_PIXELS,
)
from RMS.Routines.Grouping3D import findCoefficients


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_compressed(maxpix, stdpix):
    """ Build a (4, H, W) uint8 FTP-compressed array from maxpixel/stdpixel planes.
        maxframe (plane 1) and avepixel (plane 2) are filled with zeros; the
        decontamination only reads planes 0 (maxpixel) and 3 (stdpixel). """

    h, w = maxpix.shape
    compressed = np.zeros((4, h, w), dtype=np.uint8)
    compressed[0] = maxpix
    compressed[3] = stdpix
    return compressed


def _line(p1, p2, first_frame, last_frame):
    """ Build a line_list entry as consumed by findCoefficients: endpoints at
        indices 0/1 (each a (y, x, z) triple) and first/last frame at indices 4/5. """

    return [p1, p2, 0, 0, first_frame, last_frame]


class _Cfg(object):
    """ Minimal stand-in for the RMS Config object (only the attributes
        findCoefficients reads). """

    def __init__(self, **kwargs):
        self.fireball_max_ang_vel = 60.0
        self.fps = 25.0
        self.fov_h = 35.0
        self.fov_w = 64.0
        self.height = 720
        self.width = 1280
        self.f = 16
        for key, val in kwargs.items():
            setattr(self, key, val)


# ---------------------------------------------------------------------------
# decontaminateStdpixel
# ---------------------------------------------------------------------------

class TestDecontaminateStdpixel:

    def test_clean_image_is_noop(self):
        """ With uniform stdpixel there is no contamination, so the same array object
            is returned untouched (zero-copy fast path). """

        maxpix = np.full((100, 100), 30, dtype=np.uint8)
        maxpix[:20, :] = 60  # a bright but clean region (low, uniform std)
        stdpix = np.full((100, 100), 10, dtype=np.uint8)
        compressed = _make_compressed(maxpix, stdpix)

        result = decontaminateStdpixel(compressed)

        assert result is compressed
        assert np.array_equal(result[3], stdpix)

    def test_contaminated_trail_is_replaced(self):
        """ A bright trail with inflated stdpixel gets its stdpixel replaced by the
            background median; every other plane and pixel is left untouched, and the
            input array is not mutated. """

        maxpix = np.full((100, 100), 20, dtype=np.uint8)
        maxpix[:20, :] = 40                    # bright-but-clean band (not contaminated)
        maxpix[50, 25:75] = 255                # the fireball trail
        stdpix = np.full((100, 100), 12, dtype=np.uint8)
        stdpix[50, 25:75] = 200                # self-contaminated stdpixel on the trail
        compressed = _make_compressed(maxpix, stdpix)

        result = decontaminateStdpixel(compressed)

        # A copy is returned when contamination is present
        assert result is not compressed

        # Trail stdpixel is pulled down to the background median (12)
        assert np.all(result[3][50, 25:75] == 12)

        # The clean bright band and background are unchanged
        assert np.all(result[3][:20, :] == 12)
        assert np.all(result[3][60:, :] == 12)

        # maxpixel plane is untouched
        assert np.array_equal(result[0], compressed[0])

        # The original array was not mutated in place
        assert np.all(compressed[3][50, 25:75] == 200)

    def test_binned_mask_shape_mismatch_does_not_crash(self):
        """ Regression for the blocking bug: on stations with detection_binning_factor > 1
            the mask is binned to a smaller shape than the capture-resolution frames.
            The decontamination must skip the mask instead of raising a broadcast error. """

        maxpix = np.full((100, 100), 20, dtype=np.uint8)
        maxpix[50, 25:75] = 255
        stdpix = np.full((100, 100), 12, dtype=np.uint8)
        stdpix[50, 25:75] = 200
        compressed = _make_compressed(maxpix, stdpix)

        # Half-resolution mask, as produced by binImageCalibration() at binning factor 2
        binned_mask = np.full((50, 50), 255, dtype=np.uint8)

        # Must not raise, and must still decontaminate as if no mask were present
        result = decontaminateStdpixel(compressed, mask_img=binned_mask)

        assert result is not compressed
        assert np.all(result[3][50, 25:75] == 12)

    def test_matching_mask_excludes_masked_region(self):
        """ Pixels where the mask is 0 (camera borders/obstructions) must be excluded
            from the contamination set, so a contaminated pixel under the mask keeps its
            original stdpixel. """

        maxpix = np.full((100, 100), 20, dtype=np.uint8)
        maxpix[50, 25:75] = 255   # trail in the visible region
        maxpix[70, 25:75] = 255   # trail in the masked-out region
        stdpix = np.full((100, 100), 12, dtype=np.uint8)
        stdpix[50, 25:75] = 200
        stdpix[70, 25:75] = 200
        compressed = _make_compressed(maxpix, stdpix)

        mask_img = np.full((100, 100), 255, dtype=np.uint8)
        mask_img[60:, :] = 0      # mask out the bottom 40 rows

        result = decontaminateStdpixel(compressed, mask_img=mask_img)

        # Visible trail is decontaminated
        assert np.all(result[3][50, 25:75] == 12)

        # Masked-out trail is left untouched
        assert np.all(result[3][70, 25:75] == 200)

    def test_replacement_is_clamped_to_at_least_one(self):
        """ The replacement value is always a valid uint8 >= 1 even when the background
            median is 0, so a decontaminated pixel never gets a zero threshold offset. """

        maxpix = np.full((100, 100), 20, dtype=np.uint8)
        maxpix[50, 25:75] = 255
        stdpix = np.zeros((100, 100), dtype=np.uint8)   # background median std == 0
        stdpix[50, 25:75] = 200
        compressed = _make_compressed(maxpix, stdpix)

        result = decontaminateStdpixel(compressed)

        assert np.all(result[3][50, 25:75] == 1)

    def test_tuning_constants_have_expected_defaults(self):
        """ Pin the documented tuning constants so a future edit is a conscious choice. """

        assert BRIGHT_PERCENTILE == 90
        assert CONTAMINATION_STD_FACTOR == 3.0
        assert MIN_BACKGROUND_PIXELS == 100


# ---------------------------------------------------------------------------
# findCoefficients velocity cap
# ---------------------------------------------------------------------------

class TestFindCoefficientsVelocityCap:

    def _slope_line(self, velocity_px):
        """ A one-frame line whose total subsampled speed equals velocity_px:
            dz = 1, dx = velocity_px, dy = 0 -> total = sqrt(v^2 + 0) = v. """

        p1 = (0.0, 0.0, 0.0)
        p2 = (0.0, float(velocity_px), 1.0)
        return _line(p1, p2, first_frame=0, last_frame=1)

    def test_fallback_cap_without_config(self):
        """ With no config the legacy ~2.0 px/frame cap applies. """

        kept = findCoefficients([self._slope_line(1.5)], config=None)
        dropped = findCoefficients([self._slope_line(2.5)], config=None)

        assert len(kept) == 1
        assert len(dropped) == 0

    def test_default_config_widens_cap(self):
        """ At the default 720p/25 fps/f=16 config, 60 deg/s converts to ~3.04
            subsampled px/frame -- wider than the legacy 2.0, so a 3.0 px/frame line is
            kept while a 3.1 px/frame line is dropped. """

        cfg = _Cfg()

        assert len(findCoefficients([self._slope_line(3.0)], config=cfg)) == 1
        assert len(findCoefficients([self._slope_line(3.1)], config=cfg)) == 0

    def test_line_faster_than_cap_would_be_kept_under_default(self):
        """ A 2.5 px/frame line is dropped by the fallback but kept by the default
            config -- the exact behavior that rescues slower fireballs. """

        assert len(findCoefficients([self._slope_line(2.5)], config=None)) == 0
        assert len(findCoefficients([self._slope_line(2.5)], config=_Cfg())) == 1

    def test_missing_config_attribute_falls_back(self):
        """ If a required attribute is missing/zero, the conversion is skipped and the
            conservative 2.0 px/frame fallback is used. """

        cfg = _Cfg(fov_h=None)

        assert len(findCoefficients([self._slope_line(2.5)], config=cfg)) == 0

    def test_points_on_same_frame_are_skipped(self):
        """ Degenerate lines with both endpoints on the same frame are ignored rather
            than dividing by zero. """

        same_frame = _line((0.0, 0.0, 5.0), (0.0, 1.0, 5.0), first_frame=5, last_frame=5)

        assert findCoefficients([same_frame], config=_Cfg()) == []
