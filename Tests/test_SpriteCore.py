""" Tests for the sprite detector core: RMS/Sprite/Detection.py, RMS/Sprite/Astrometry.py and
RMS/Sprite/Filter.py.

All FF files are synthetic and built in memory, no disk and no model are needed, except for one inference
smoke test that is skipped without a TFLite backend or the model file.
"""

from __future__ import print_function, division, absolute_import

import ast
import datetime
import os
import random
import sys
import time
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from PIL import Image

from RMS.ConfigReader import Config
from RMS.Formats.FFStruct import FFStruct
from RMS.Formats.FFfile import reconstructFrame
import RMS.Sprite.Astrometry as Astrometry
import RMS.Sprite.Detection as Detection
import RMS.Sprite.Filter as Filter


REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
FF_NAME = "FF_XX0001_20250101_020520_353_0005120.fits"
CORE_MODULES = ["RMS/Sprite/Filter.py", "RMS/Sprite/Detection.py", "RMS/Sprite/Astrometry.py"]


# ###########################################################################
#                       Helpers
# ###########################################################################

def makeFF(nrows=64, ncols=64, nframes=256, fps=25.0):
    """ Build an empty in-memory FF. """

    ff = FFStruct()
    ff.nrows = nrows
    ff.ncols = ncols
    ff.nframes = nframes
    ff.fps = fps
    ff.maxpixel = np.zeros((nrows, ncols), dtype=np.uint8)
    ff.maxframe = np.zeros((nrows, ncols), dtype=np.uint8)
    ff.avepixel = np.zeros((nrows, ncols), dtype=np.uint8)
    ff.stdpixel = np.zeros((nrows, ncols), dtype=np.uint8)

    return ff


def makeConfig():
    """ Default config with the attributes the core reads. """

    config = Config()
    config.stationID = "XX0001"
    config.fps = 25.0
    config.width = 1280
    config.height = 720
    config.sprite_confidence = 0.386

    return config


def prArtifactLoop(ff, x0, y0, x1, y1, k=1):
    """ The PR's tle_artifact_filter, verbatim in what it computes, returning d as well. """

    nframes = int(ff.nframes)
    fieldsums = np.empty(nframes, dtype=np.float64)

    fieldsum = 0
    for i in range(nframes):
        frame = reconstructFrame(ff, i)
        roi = frame[y0:y1, x0:x1]
        fieldsum += roi.sum()
        fieldsums[i] = fieldsum

    d = np.diff(fieldsums, prepend=0)
    share = np.sort(d)[-k:].sum()/d.sum()

    return share > 0.1, int(np.argmax(d)), share, d


class FakeInterpreter(object):
    """ Stand-in for a TFLite interpreter: records the input and returns a fixed output. """

    def __init__(self, output):
        self.output = output
        self.inputs = []

    def set_tensor(self, index, data):
        self.inputs.append(np.array(data, copy=True))

    def invoke(self):
        pass

    def get_output_details(self):
        return [{"index": 1}]

    def get_tensor(self, index):
        return self.output


INPUT_DETAILS = {"shape": np.array([1, 3, 320, 320]), "index": 0, "dtype": np.float32}


def modelOutput(boxes, n_anchors=50):
    """ Build a raw (1, 6, n_anchors) model output from (cx, cy, w, h, class_id, conf) tuples. """

    out = np.zeros((1, 6, n_anchors), dtype=np.float32)
    for i, (cx, cy, w, h, cid, conf) in enumerate(boxes):
        out[0, :4, i] = [cx, cy, w, h]
        out[0, 4 + int(cid), i] = conf

    return out


@pytest.fixture
def fakeModel(monkeypatch):
    """ Patch the interpreter loader; set .output on the returned object before running detection. """

    fake = FakeInterpreter(modelOutput([]))
    monkeypatch.setattr(Detection, "getSpriteInterpreter", lambda path: (fake, INPUT_DETAILS))
    monkeypatch.setattr(Detection, "_resized_mask_cache", {"source": None, "shape": None,
        "resized": None})

    return fake


# ###########################################################################
#                       Artifact filter
# ###########################################################################

def test_artifact_histogram_equals_reconstruct_loop():
    rng = np.random.RandomState(1)
    ff = makeFF()
    ff.maxpixel = rng.randint(0, 256, size=(64, 64)).astype(np.uint8)
    ff.maxframe = rng.randint(0, 256, size=(64, 64)).astype(np.uint8)

    x0, y0, x1, y1 = 5, 7, 50, 40
    is_loop, idx_loop, share_loop, d_loop = prArtifactLoop(ff, x0, y0, x1, y1)

    d = Detection.boxLightPerFrame(ff, x0, y0, x1, y1)
    np.testing.assert_array_equal(d, d_loop)

    is_sprite, idx, share = Detection.spriteArtifactFilter(ff, x0, y0, x1, y1)
    assert is_sprite == is_loop
    assert idx == idx_loop
    assert share == share_loop


def test_artifact_single_frame_flash_passes_spread_fails():
    ff = makeFF()
    ff.maxpixel[15:25, 15:25] = 200
    ff.maxframe[15:25, 15:25] = 42

    is_sprite, idx, share = Detection.spriteArtifactFilter(ff, 10, 10, 30, 30)
    assert is_sprite and idx == 42 and share == 1.0

    # Same light over 100 frames, 1% per frame
    ff.maxframe[15:25, 15:25] = np.arange(100).reshape(10, 10)
    is_sprite, idx, share = Detection.spriteArtifactFilter(ff, 10, 10, 30, 30)
    assert not is_sprite
    assert abs(share - 0.01) < 1e-12


def test_artifact_zero_roi_no_warning():
    ff = makeFF()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert Detection.spriteArtifactFilter(ff, 10, 10, 30, 30) == (False, 0, 0.0)


def test_artifact_nframes_unknown():
    ff = makeFF(nframes=-1)
    ff.maxpixel[20, 20] = 100
    ff.maxframe[20, 20] = 7

    is_sprite, idx, share = Detection.spriteArtifactFilter(ff, 10, 10, 30, 30)
    assert is_sprite and idx == 7


def test_artifact_histogram_truncated_to_nframes():
    ff = makeFF(nframes=100)
    ff.maxpixel[20, 20] = 250
    ff.maxframe[20, 20] = 150
    ff.maxpixel[21, 21] = 10
    ff.maxframe[21, 21] = 3

    d = Detection.boxLightPerFrame(ff, 10, 10, 30, 30)
    assert len(d) == 100
    assert Detection.spriteArtifactFilter(ff, 10, 10, 30, 30)[1] == 3


def test_artifact_box_clipped_not_raised():
    ff = makeFF()
    ff.maxpixel[60, 60] = 100
    ff.maxframe[60, 60] = 9

    # Partly outside the frame
    assert Detection.spriteArtifactFilter(ff, 50, 50, 100, 100)[:2] == (True, 9)

    # Reversed corners
    assert Detection.spriteArtifactFilter(ff, 100, 100, 50, 50)[:2] == (True, 9)

    # Entirely outside and degenerate, still at least one pixel and no exception
    Detection.spriteArtifactFilter(ff, 200, 200, 300, 300)
    Detection.spriteArtifactFilter(ff, -20, -20, -10, -10)
    Detection.spriteArtifactFilter(ff, 30, 30, 30, 30)


def test_artifact_brightest_frame_wins():
    ff = makeFF()
    ff.maxpixel[12:15, 12:15] = 100
    ff.maxframe[12:15, 12:15] = 10
    ff.maxpixel[25:28, 25:28] = 250
    ff.maxframe[25:28, 25:28] = 200

    assert Detection.spriteArtifactFilter(ff, 10, 10, 30, 30)[1] == 200


def test_artifact_no_maxframe_and_background():
    ff = makeFF()
    ff.maxframe = None
    assert Detection.spriteArtifactFilter(ff, 0, 0, 10, 10) == (True, 0, 1.0)

    # A static star (maxpixel == avepixel) dominates raw maxpixel but not the background-subtracted light
    ff = makeFF()
    ff.maxpixel[20:24, 20:24] = 255
    ff.avepixel[20:24, 20:24] = 255
    ff.maxframe[20:24, 20:24] = 77
    ff.maxpixel[15, 15] = 60
    ff.maxframe[15, 15] = 5
    assert Detection.spriteArtifactFilter(ff, 10, 10, 30, 30)[1] == 77
    assert Detection.spriteArtifactFilter(ff, 10, 10, 30, 30, subtract_background=True)[1] == 5


# ###########################################################################
#                       Timing
# ###########################################################################

def test_frame_time_uses_milliseconds():
    t = Detection.spriteFrameTime(FF_NAME, 128, 25.0)
    assert t == datetime.datetime(2025, 1, 1, 2, 5, 25, 473000)
    assert t.tzinfo is None

    # The PR dropped the milliseconds field
    parts = FF_NAME.split("_")
    pr_time = datetime.datetime.strptime(parts[2] + parts[3], "%Y%m%d%H%M%S") \
        + datetime.timedelta(seconds=128/25.0)
    assert (t - pr_time) == datetime.timedelta(milliseconds=353)

    assert Detection.formatIsoTimestamp(t) == "2025-01-01T02:05:25.473000Z"


def test_detection_uses_ff_fps(fakeModel):
    ff = makeFF(fps=20.0)
    ff.maxpixel[28:36, 28:36] = 200
    ff.maxframe[28:36, 28:36] = 100
    fakeModel.output = modelOutput([(0.5, 0.5, 0.25, 0.25, 1, 0.9)])

    config = makeConfig()
    config.fps = 25.0

    dets = Detection.detectSpritesInFF(ff, FF_NAME, config, "sprite_detector.tflite")
    assert len(dets) == 1
    assert dets[0]["frame_index"] == 100
    assert dets[0]["event_time"] == datetime.datetime(2025, 1, 1, 2, 5, 25, 353000)

    # Without a header rate the config rate is used
    ff.fps = -1
    dets = Detection.detectSpritesInFF(ff, FF_NAME, config, "sprite_detector.tflite")
    assert dets[0]["event_time"] == datetime.datetime(2025, 1, 1, 2, 5, 24, 353000)


def test_detections_get_own_julian_dates(fakeModel):
    ff = makeFF()
    ff.maxpixel[5:10, 5:10] = 200
    ff.maxframe[5:10, 5:10] = 0
    ff.maxpixel[45:50, 45:50] = 200
    ff.maxframe[45:50, 45:50] = 255
    fakeModel.output = modelOutput([(0.12, 0.12, 0.2, 0.2, 1, 0.9), (0.74, 0.74, 0.2, 0.2, 1, 0.8)])

    dets = Detection.detectSpritesInFF(ff, FF_NAME, makeConfig(), "m.tflite")
    assert [d["frame_index"] for d in dets] == [0, 255]
    assert [d["detection_index"] for d in dets] == [0, 1]
    assert abs((dets[1]["jd"] - dets[0]["jd"]) - 255/25.0/86400) < 1e-9


# ###########################################################################
#                       Detection record, boxes, mask, preprocessing
# ###########################################################################

def test_detection_record_and_box_scaling(fakeModel):
    ff = makeFF()
    ff.maxpixel[28:36, 28:36] = 200
    ff.maxframe[28:36, 28:36] = 128
    fakeModel.output = modelOutput([(0.5, 0.5, 0.25, 0.25, 0, 0.9)])

    config = makeConfig()
    dets = Detection.detectSpritesInFF(ff, FF_NAME, config, "/x/sprite_detector.tflite")
    assert len(dets) == 1
    det = dets[0]

    # Scaled with the FF's 64x64, not the config's 1280x720
    assert (det["box_x1"], det["box_y1"], det["box_x2"], det["box_y2"]) == (24, 24, 40, 40)
    assert (det["centroid_x"], det["centroid_y"]) == (32.0, 32.0)
    assert det["detection_type"] == "elve"
    assert det["model"] == "sprite_detector"
    assert det["ff_name"] == FF_NAME
    assert det["timestamp"] == "2025-01-01T02:05:25.473000Z"
    assert abs(det["confidence"] - 0.9) < 1e-6
    assert det["artifact_share"] == 1.0

    expected_keys = set(["ff_name", "detection_index", "detection_type", "model", "confidence", "centroid_x",
        "centroid_y", "box_x1", "box_y1", "box_x2", "box_y2", "frame_index", "artifact_share", "event_time",
        "timestamp", "jd"])
    assert set(det.keys()) == expected_keys


def test_box_pixel_units_and_clipping(fakeModel):
    ff = makeFF()
    ff.maxpixel[28:36, 28:36] = 200
    ff.maxframe[28:36, 28:36] = 128

    # Boxes in 320 px model input units give the same FF box
    fakeModel.output = modelOutput([(160.0, 160.0, 80.0, 80.0, 1, 0.9)])
    det = Detection.detectSpritesInFF(ff, FF_NAME, makeConfig(), "m.tflite")[0]
    assert (det["box_x1"], det["box_y1"], det["box_x2"], det["box_y2"]) == (24, 24, 40, 40)

    # A box hanging over the edge is clipped, a zero-size box gets one pixel
    boxes = Detection._scaleBoxes(np.array([[0.9, -0.1, 1.2, 0.2], [0.5, 0.5, 0.5, 0.5]]),
        INPUT_DETAILS, 64, 64)
    assert boxes[0] == (57, 0, 64, 12)
    assert boxes[1] == (32, 32, 33, 33)


def test_confidence_from_config(fakeModel):
    ff = makeFF()
    ff.maxpixel[28:36, 28:36] = 200
    ff.maxframe[28:36, 28:36] = 128
    fakeModel.output = modelOutput([(0.5, 0.5, 0.25, 0.25, 1, 0.5)])

    config = makeConfig()
    config.sprite_confidence = 0.6
    assert Detection.detectSpritesInFF(ff, FF_NAME, config, "m.tflite") == []

    config.sprite_confidence = 0.4
    assert len(Detection.detectSpritesInFF(ff, FF_NAME, config, "m.tflite")) == 1


def test_artifact_rejected_detection_dropped(fakeModel):
    ff = makeFF()
    ff.maxpixel[28:38, 28:38] = 200
    ff.maxframe[28:38, 28:38] = np.arange(100).reshape(10, 10)
    fakeModel.output = modelOutput([(0.5, 0.5, 0.3, 0.3, 1, 0.9)])

    assert Detection.detectSpritesInFF(ff, FF_NAME, makeConfig(), "m.tflite") == []


def test_mask_is_applied_and_resized(fakeModel):
    rng = np.random.RandomState(3)
    ff = makeFF()
    ff.maxpixel = rng.randint(50, 256, size=(64, 64)).astype(np.uint8)
    maxpixel_orig = ff.maxpixel.copy()
    config = makeConfig()

    Detection.detectSpritesInFF(ff, FF_NAME, config, "m.tflite")
    unmasked_input = fakeModel.inputs[-1]

    # Left half masked out
    mask = np.full((64, 64), 255, dtype=np.uint8)
    mask[:, :32] = 0
    Detection.detectSpritesInFF(ff, FF_NAME, config, "m.tflite", mask=mask)
    masked_input = fakeModel.inputs[-1]

    assert not np.array_equal(unmasked_input, masked_input)

    # The masked half is flat (filled with the mean of the unmasked pixels), the other half untouched
    assert np.ptp(masked_input[0, :, :, :150]) == 0
    np.testing.assert_array_equal(masked_input[0, :, :, 170:], unmasked_input[0, :, :, 170:])

    # The FF itself is not modified
    np.testing.assert_array_equal(ff.maxpixel, maxpixel_orig)

    # A half-resolution mask is resized with nearest-neighbour, not ignored
    small_mask = np.full((32, 32), 255, dtype=np.uint8)
    small_mask[:, :16] = 0
    Detection.detectSpritesInFF(ff, FF_NAME, config, "m.tflite", mask=small_mask)
    np.testing.assert_array_equal(fakeModel.inputs[-1], masked_input)

    # The resized mask is cached and reused for the same mask object
    cached = Detection._resized_mask_cache["resized"]
    Detection.detectSpritesInFF(ff, FF_NAME, config, "m.tflite", mask=small_mask)
    assert Detection._resized_mask_cache["resized"] is cached


def test_preprocessing_identical_to_pr(fakeModel):
    rng = np.random.RandomState(4)
    ff = makeFF(nrows=72, ncols=128)
    ff.maxpixel = rng.randint(0, 256, size=(72, 128)).astype(np.uint8)

    Detection.detectSpritesInFF(ff, FF_NAME, makeConfig(), "m.tflite")

    # The PR's preprocessing, channels-first
    image = Image.fromarray(ff.maxpixel).convert("RGB").convert("RGB")
    image = image.resize((320, 320))
    expected = np.array(image, dtype=np.float32)
    expected /= 255
    expected = np.transpose(expected, (2, 0, 1))[None]

    assert fakeModel.inputs[-1].dtype == np.float32
    np.testing.assert_array_equal(fakeModel.inputs[-1], expected)


def test_process_predictions_nms_and_caps():
    # Three overlapping sprites collapse to one, four separate sprites keep four of five, one elve max
    boxes = [(0.1, 0.1, 0.1, 0.1, 1, 0.9), (0.105, 0.1, 0.1, 0.1, 1, 0.8), (0.11, 0.1, 0.1, 0.1, 1, 0.7)]
    boxes += [(0.3 + 0.12*i, 0.5, 0.05, 0.05, 1, 0.6 - 0.01*i) for i in range(5)]
    boxes += [(0.5, 0.9, 0.05, 0.05, 0, 0.95), (0.8, 0.9, 0.05, 0.05, 0, 0.5)]
    out = Detection.processPredictions(modelOutput(boxes))

    assert int(np.sum(out[:, 5] == 1)) == 4
    assert int(np.sum(out[:, 5] == 0)) == 1
    assert np.all(np.diff(out[:, 4]) <= 0)
    assert Detection.processPredictions(modelOutput([])).shape == (0, 6)


def test_class_names():
    assert Detection.CLASS_NAMES == {0: "elve", 1: "sprite"}


def test_inference_smoke():
    if Detection.SPRITE_TFLITE_BACKEND == "litert":
        pytest.importorskip("ai_edge_litert.interpreter")
    elif Detection.SPRITE_TFLITE_BACKEND == "tf_full":
        pytest.importorskip("tensorflow")
    else:
        pytest.importorskip("tflite_runtime.interpreter")

    model_path = os.path.join(REPO_DIR, "share", "sprite_detector.tflite")
    if not os.path.isfile(model_path):
        pytest.skip("model file not available")

    rng = np.random.RandomState(5)
    ff = makeFF(nrows=720, ncols=1280)
    ff.maxpixel = rng.randint(0, 40, size=(720, 1280)).astype(np.uint8)
    ff.maxframe = rng.randint(0, 256, size=(720, 1280)).astype(np.uint8)

    dets = Detection.detectSpritesInFF(ff, FF_NAME, makeConfig(), model_path)
    assert isinstance(dets, list)

    interpreter, input_details = Detection.getSpriteInterpreter(model_path)
    assert Detection.getSpriteInterpreter(model_path)[0] is interpreter


# ###########################################################################
#                       Astrometry
# ###########################################################################

def makePlatepar(**kwargs):
    from RMS.Formats.Platepar import Platepar

    pp = Platepar()
    pp.X_res = 64
    pp.Y_res = 64
    pp.lat = 45.0
    pp.lon = 15.0
    pp.RA_d = 100.0
    pp.dec_d = 40.0
    pp.F_scale = 64/30.0
    for key, value in kwargs.items():
        setattr(pp, key, value)

    return pp


def makeDetection(jd, x1=10, y1=12, x2=30, y2=40):
    return {"ff_name": FF_NAME, "box_x1": x1, "box_y1": y1, "box_x2": x2, "box_y2": y2,
        "centroid_x": (x1 + x2)/2.0, "centroid_y": (y1 + y2)/2.0, "jd": jd}


def test_astrometry_call_arguments(monkeypatch):
    calls = {}

    def fakeXY(time_data, x_data, y_data, level_data, platepar, **kwargs):
        calls["xy"] = (np.array(time_data), np.array(x_data), np.array(y_data), kwargs)
        n = len(x_data)
        return np.array(time_data), np.arange(n, dtype=float), np.arange(n, dtype=float) - 10, np.ones(n)

    def fakeAltAz(ra, dec, jd, lat, lon, **kwargs):
        calls["altaz"] = (np.array(jd), kwargs)
        return ra + 100.0, dec + 50.0

    monkeypatch.setattr(Astrometry, "xyToRaDecPP", fakeXY)
    monkeypatch.setattr(Astrometry, "trueRaDec2ApparentAltAz", fakeAltAz)

    dets = [makeDetection(2460000.1), makeDetection(2460000.2, 1, 2, 3, 4)]
    Astrometry.calibrateSpriteDetections(dets, makePlatepar(), True, None)

    jd_arr, x_arr, y_arr, kwargs = calls["xy"]
    assert kwargs == {"extinction_correction": False, "measurement": True, "jd_time": True,
        "precompute_pointing_corr": False}
    np.testing.assert_array_equal(jd_arr, [2460000.1]*5 + [2460000.2]*5)
    np.testing.assert_array_equal(x_arr[5:], [2.0, 1, 1, 3, 3])
    np.testing.assert_array_equal(y_arr[5:], [3.0, 2, 4, 2, 4])

    altaz_jd, altaz_kwargs = calls["altaz"]
    assert altaz_kwargs == {"refraction": False}
    np.testing.assert_array_equal(altaz_jd, jd_arr)

    # Results distributed five points per detection, rounded as the PR
    assert dets[1]["astrometry_ok"] and dets[1]["astrometry_reason"] is None
    assert dets[1]["ra_j2000"] == 5.0 and dets[1]["box_ra_j2000_4"] == 9.0
    assert dets[1]["dec_j2000"] == -5.0 and dets[1]["box_altitude_4"] == 49.0
    assert dets[0]["box_azimuth_2"] == 102.0


def test_astrometry_real_platepar():
    pp = makePlatepar()
    dets = [makeDetection(2460000.5), makeDetection(2460000.5 + 1/24.0)]
    Astrometry.calibrateSpriteDetections(dets, pp, True, None)

    for det in dets:
        assert det["astrometry_ok"]
        for n in range(1, 5):
            assert np.isfinite(det["box_azimuth_{:d}".format(n)])
            assert -90 <= det["box_altitude_{:d}".format(n)] <= 90

    # Same pixels an hour apart: same alt/az, different RA
    assert abs(dets[0]["altitude"] - dets[1]["altitude"]) < 1e-3
    assert abs(dets[0]["ra_j2000"] - dets[1]["ra_j2000"]) > 10


def test_astrometry_failure_is_per_ff(monkeypatch):
    real_xy = Astrometry.xyToRaDecPP

    def flakyXY(time_data, *args, **kwargs):
        if time_data[0] < 2460000.3:
            raise ValueError("boom")
        return real_xy(time_data, *args, **kwargs)

    monkeypatch.setattr(Astrometry, "xyToRaDecPP", flakyXY)
    pp = makePlatepar()

    ff1 = [makeDetection(2460000.1), makeDetection(2460000.1)]
    ff2 = [makeDetection(2460000.5)]
    Astrometry.calibrateSpriteDetections(ff1, pp, True, None)
    Astrometry.calibrateSpriteDetections(ff2, pp, True, None)

    for det in ff1:
        assert det["astrometry_ok"] is False
        assert "boom" in det["astrometry_reason"]
        assert "ra_j2000" not in det
    assert ff2[0]["astrometry_ok"] is True
    assert "box_dec_j2000_3" in ff2[0]


def test_astrometry_not_usable_marks_all():
    dets = [makeDetection(2460000.1), makeDetection(2460000.2)]
    Astrometry.calibrateSpriteDetections(dets, None, False, "no platepar")
    assert all(d["astrometry_ok"] is False and d["astrometry_reason"] == "no platepar" for d in dets)


def test_platepar_usable():
    config = makeConfig()

    assert Astrometry.plateparUsable(None, 64, 64, config)[0] is False

    ok, reason = Astrometry.plateparUsable(makePlatepar(station_code="YY0002"), 64, 64, config)
    assert not ok and "YY0002" in reason

    # Unset station code (the Platepar default is the string "None") and a lower-case match are fine
    assert Astrometry.plateparUsable(makePlatepar(station_code="None", auto_recalibrated=True), 64, 64,
        config) == (True, None)
    assert Astrometry.plateparUsable(makePlatepar(station_code="xx0001", auto_recalibrated=True), 64,
        64, config) == (True, None)

    ok, reason = Astrometry.plateparUsable(makePlatepar(), 1280, 720, config)
    assert not ok and "resolution" in reason

    # Refraction is always taken out, explicitly or inside the fitted distortion, so a platepar fitted with
    #   refraction off is used like any other
    ok, reason = Astrometry.plateparUsable(makePlatepar(refraction=False,
        measurement_apparent_to_true_refraction=False, auto_check_fit_refined=True), 64, 64, config)
    assert ok and reason is None

    ok, reason = Astrometry.plateparUsable(makePlatepar(refraction=False,
        measurement_apparent_to_true_refraction=True, auto_check_fit_refined=True), 64, 64, config)
    assert ok and reason is None

    # Neither automatic refinement flag: usable, with a caveat
    ok, reason = Astrometry.plateparUsable(makePlatepar(auto_check_fit_refined=False,
        auto_recalibrated=False), 64, 64, config)
    assert ok and reason


def test_platepar_provenance():
    pp = makePlatepar(auto_check_fit_refined=True)
    prov = Astrometry.plateparProvenance(pp, "/data/platepar_cmn2010.cal")
    assert prov == {"JD": float(pp.JD), "source_path": "/data/platepar_cmn2010.cal", "X_res": 64,
        "Y_res": 64, "refraction": True, "measurement_apparent_to_true_refraction": False,
        "auto_check_fit_refined": True, "auto_recalibrated": False}

    prov_none = Astrometry.plateparProvenance(None, None)
    assert set(prov_none.keys()) == set(prov.keys())


# ###########################################################################
#                       False-positive filter
# ###########################################################################

def runFilter(events, window_sec=60.0, max_detections=3):
    """ Feed (timestamp, ff_name or None) events, None meaning a tick; return confirmed names in order. """

    confirmed = []
    fp = Filter.SpriteFalsePositiveFilter(lambda c: confirmed.append(c.ff_name), window_sec=window_sec,
        max_detections=max_detections)
    for ts, name in events:
        if name is None:
            fp.tick(ts)
        else:
            fp.addCandidate(ts, name, [])
    fp.flush()

    return confirmed


def test_filter_burst_confirms_none():
    t0 = 1.7e9
    events = [(t0 + 10*i, "ff{:d}".format(i)) for i in range(5)]
    events += [(t0 + 10*i, None) for i in range(5, 30)]
    assert runFilter(events) == []


def test_filter_isolated_confirm_and_latency():
    t0 = 1.7e9
    confirmed = []
    fp = Filter.SpriteFalsePositiveFilter(lambda c: confirmed.append((c.ff_name, c.detections)))
    fp.addCandidate(t0, "a", ["det"])

    # Still pending exactly at the window edge, confirmed once past it
    fp.tick(t0 + 60.0)
    assert confirmed == []
    fp.tick(t0 + 61.44)
    assert confirmed == [("a", ["det"])]

    fp.addCandidate(t0 + 600, "b", [])
    fp.tick(t0 + 671.68)
    assert [c[0] for c in confirmed] == ["a", "b"]


def test_filter_independent_of_wall_clock(monkeypatch):
    t0 = 1.7e9
    events = []
    rng = random.Random(7)
    for i in range(200):
        ts = t0 + 10.24*i
        name = "ff{:d}".format(i) if rng.random() < 0.15 else None
        events.append((ts, name))

    first = runFilter(events)

    # Wall clock jumping around and real delays between the events must not change anything
    fake_now = [0.0]

    def jumpyTime():
        fake_now[0] += rng.random()*1000
        return fake_now[0]

    monkeypatch.setattr(time, "time", jumpyTime)
    monkeypatch.setattr(time, "monotonic", jumpyTime)

    confirmed = []
    fp = Filter.SpriteFalsePositiveFilter(lambda c: confirmed.append(c.ff_name))
    for i, (ts, name) in enumerate(events):
        if i % 50 == 0:
            time.sleep(0.001)
        if name is None:
            fp.tick(ts)
        else:
            fp.addCandidate(ts, name, [])
    fp.flush()

    assert confirmed == first
    assert len(first) > 0


def test_filter_flush_once():
    t0 = 1.7e9
    confirmed = []
    fp = Filter.SpriteFalsePositiveFilter(lambda c: confirmed.append(c.ff_name))
    fp.addCandidate(t0, "a", [])
    fp.addCandidate(t0 + 20, "b", [])
    assert fp.pending() == 2

    fp.flush()
    fp.flush()
    fp.tick(t0 + 1000)
    assert confirmed == ["a", "b"]
    assert fp.pending() == 0


def test_filter_clock_monotonic():
    t0 = 1.7e9
    confirmed = []
    fp = Filter.SpriteFalsePositiveFilter(lambda c: confirmed.append(c.ff_name))
    fp.addCandidate(t0, "a", [])
    fp.tick(t0 + 100)
    assert confirmed == ["a"]

    # A tick back in time does not rewind the clock
    fp.addCandidate(t0 + 90, "b", [])
    fp.tick(t0)
    fp.tick(t0 + 151)
    assert confirmed == ["a", "b"]


def test_filter_rejected_callback_exactly_once():
    t0 = 1.7e9
    confirmed = []
    rejected = []
    fp = Filter.SpriteFalsePositiveFilter(lambda c: confirmed.append(c.ff_name),
        on_rejected=lambda c: rejected.append(c.ff_name))

    # A burst of five, then an isolated one ten minutes later, resolved by ticks
    burst = ["b{:d}".format(i) for i in range(5)]
    for i, name in enumerate(burst):
        fp.addCandidate(t0 + 10*i, name, [])
    fp.addCandidate(t0 + 600, "iso", [])
    fp.tick(t0 + 700)
    assert rejected == burst
    assert confirmed == ["iso"]

    # flush() drains both kinds, still exactly once
    for i in range(4):
        fp.addCandidate(t0 + 1000 + i, "c{:d}".format(i), [])
    fp.addCandidate(t0 + 2000, "iso2", [])
    fp.flush()
    fp.flush()
    assert rejected == burst + ["c0", "c1", "c2", "c3"]
    assert confirmed == ["iso", "iso2"]
    assert fp.pending() == 0

    # Without on_rejected a burst is simply dropped
    fp = Filter.SpriteFalsePositiveFilter(lambda c: confirmed.append(c.ff_name))
    for i in range(5):
        fp.addCandidate(t0 + i, "d{:d}".format(i), [])
    fp.flush()
    assert confirmed == ["iso", "iso2"]


def test_filter_candidate_slots():
    c = Filter.SpriteCandidate(1.0, "x", [])
    assert not hasattr(c, "__dict__")
    assert c.tainted is False


def test_deterministic_matches_streaming_and_order():
    rng = np.random.RandomState(11)
    for trial in range(20):

        # Clustered timestamps so that both bursts and isolated FFs occur
        n = rng.randint(1, 40)
        t = np.sort(1.7e9 + np.round(rng.exponential(40.0, size=n).cumsum(), 2))

        # Some identical timestamps, the streaming filter must agree on ties too
        ties = np.where(rng.rand(n) < 0.15)[0]
        ties = ties[ties > 0]
        t[ties] = t[ties - 1]

        accepted = Filter.filterCandidatesDeterministic(t)

        confirmed = runFilter([(ts, i) for i, ts in enumerate(t)])
        streamed = np.zeros(n, dtype=bool)
        streamed[confirmed] = True
        np.testing.assert_array_equal(accepted, streamed)

        perm = rng.permutation(n)
        np.testing.assert_array_equal(Filter.filterCandidatesDeterministic(t[perm]), accepted[perm])


def test_deterministic_simple_cases():
    t0 = 1.7e9
    burst = [t0 + 10*i for i in range(5)]
    assert not Filter.filterCandidatesDeterministic(burst).any()
    assert Filter.filterCandidatesDeterministic([t0, t0 + 600]).all()
    assert Filter.filterCandidatesDeterministic([]).shape == (0,)

    # Exactly max_detections in the window is still fine
    assert Filter.filterCandidatesDeterministic([t0, t0 + 10, t0 + 20]).all()


# ###########################################################################
#                       House style
# ###########################################################################

@pytest.mark.parametrize("rel_path", CORE_MODULES)
def test_house_style(rel_path):
    path = os.path.join(REPO_DIR, rel_path)
    with open(path, "rb") as f:
        raw = f.read()

    # ASCII only, lines within 110 characters
    raw.decode("ascii")
    for i, line in enumerate(raw.decode("ascii").splitlines()):
        assert len(line) <= 110, "{:s}:{:d} is too long".format(rel_path, i + 1)

    tree = ast.parse(raw.decode("ascii"))
    for node in ast.walk(tree):
        assert not isinstance(node, ast.JoinedStr), "f-string in {:s}:{:d}".format(rel_path, node.lineno)
        assert not isinstance(node, ast.AnnAssign), "annotation in {:s}".format(rel_path)
        assert not isinstance(node, ast.NamedExpr), "walrus in {:s}".format(rel_path)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert node.returns is None, "return annotation in {:s}".format(rel_path)
            all_args = node.args.args + node.args.kwonlyargs + node.args.posonlyargs
            if node.args.vararg:
                all_args.append(node.args.vararg)
            if node.args.kwarg:
                all_args.append(node.args.kwarg)
            for arg in all_args:
                assert arg.annotation is None, "argument annotation in {:s}".format(rel_path)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] + [getattr(node, "module", None) or ""]
            assert "dataclasses" not in names
