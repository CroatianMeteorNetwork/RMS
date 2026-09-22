""" Tests of the sprite detector's integration layer: configuration, products, model download, and the live
    and offline detection runs. The model itself is replaced by a fake, so nothing here needs TFLite.
"""

from __future__ import print_function, division, absolute_import

import datetime
import hashlib
import json
import os
import re
import sys
import threading
import time

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import RMS.ConfigReader as cr  # noqa: E402
from RMS.Formats import FFfile  # noqa: E402
from RMS.Formats.FFStruct import FFStruct  # noqa: E402
from RMS import DownloadSpriteModel  # noqa: E402
from RMS import SpriteDetector  # noqa: E402
from RMS import SpriteProducts  # noqa: E402


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# The file name patterns the sprite server accepts (spritemap/storage.py), copied so that a change on either
#   side shows up here
SERVER_IMAGE_RE = re.compile(
    r"^(FF_[A-Z0-9]{3,8}_\d{8}_\d{6}_\d{3,6}_\d{7})_(marked|unmarked)\.(jpg|jpeg|png)$")

FF_NAME = "FF_XX0001_20260828_031422_123_0000512.fits"


### Helpers ###


def makeConfig(tmp_path):
    """ A configuration with every path pointing into the test's directory. """

    config = cr.Config()

    config.stationID = "XX0001"
    config.latitude = 43.19
    config.longitude = -81.32
    config.elevation = 324.0
    config.fps = 25.0
    config.width = 64
    config.height = 48

    config.data_dir = str(tmp_path/"data")
    config.config_file_path = str(tmp_path/"config")
    config.rsa_private_key = str(tmp_path/"config"/"id_rsa")
    config.sprite_model_path = str(tmp_path/"share"/config.sprite_model_file)

    for path in (config.data_dir, config.config_file_path, os.path.dirname(config.sprite_model_path)):
        os.makedirs(path)

    return config


def ffName(seconds_after):
    """ FF file name for a block starting the given number of seconds after 2026-08-28 03:14:22. """

    t = datetime.datetime(2026, 8, 28, 3, 14, 22) + datetime.timedelta(seconds=seconds_after)

    return "FF_XX0001_{:s}_{:03d}_{:07d}.fits".format(t.strftime("%Y%m%d_%H%M%S"), t.microsecond//1000,
                                                     int(seconds_after*25))


def writeFF(directory, ff_name, nrows=48, ncols=64):
    """ Write a small synthetic FF file and return it. """

    ff = FFStruct()
    ff.nrows = nrows
    ff.ncols = ncols
    ff.nframes = 256
    ff.fps = 25.0
    ff.first = 0
    ff.camno = 1
    ff.nbits = 8
    ff.starttime = FFfile.filenameToDatetime(ff_name).strftime("%Y-%m-%dT%H:%M:%S.%f")
    ff.maxpixel = np.full((nrows, ncols), 20, dtype=np.uint8)
    ff.maxframe = np.zeros((nrows, ncols), dtype=np.uint8)
    ff.avepixel = np.full((nrows, ncols), 10, dtype=np.uint8)
    ff.stdpixel = np.ones((nrows, ncols), dtype=np.uint8)

    FFfile.write(ff, str(directory), ff_name, fmt="fits")

    return ff


def detection(ff_name, astrometry_ok=True, confidence=0.9, detection_type="sprite"):
    """ A detection record as RMS.SpriteDetection and RMS.SpriteAstrometry produce it. """

    event = FFfile.filenameToDatetime(ff_name) + datetime.timedelta(seconds=128/25.0)

    det = {
        "ff_name": ff_name, "detection_index": 0, "detection_type": detection_type, "model": "fake",
        "confidence": confidence, "centroid_x": 30.0, "centroid_y": 20.0,
        "box_x1": 25, "box_y1": 10, "box_x2": 35, "box_y2": 30,
        "frame_index": 128, "artifact_share": 0.6, "event_time": event,
        "timestamp": event.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z", "jd": 2461280.63,
        "astrometry_ok": astrometry_ok, "astrometry_reason": None if astrometry_ok else "no platepar",
    }

    if astrometry_ok:
        det.update({"ra_j2000": 310.2, "dec_j2000": -15.1, "azimuth": 250.5, "altitude": 12.3})
        for n in range(1, 5):
            det.update({"box_ra_j2000_{:d}".format(n): 310.0, "box_dec_j2000_{:d}".format(n): -15.0,
                        "box_azimuth_{:d}".format(n): 250.0 + n, "box_altitude_{:d}".format(n): 12.0 + n})

    return det


class FakeUploader(object):
    """ Records what would have been uploaded. """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.enqueued = []
        self.started = False
        self.stopped = False

    def isEnabled(self):
        return self.enabled

    def start(self):
        self.started = True

    def enqueueDetections(self, ff_name, payload_path, files=None):
        self.enqueued.append((ff_name, payload_path, files))

    def stop(self, drain_timeout=120):
        self.stopped = True


### Configuration ###


def test_every_sprite_option_is_documented_in_dot_config():

    # The config audit reports any option that ConfigReader reads but .config lacks, on every station, every
    #   night. The first version of the sprite detector added seven such options.
    from Utils.AuditConfig import extractConfigOptions

    options = extractConfigOptions(os.path.join(REPO_ROOT, "RMS", "ConfigReader.py"))
    sprite_options = sorted(option for option in options if "sprite" in option)

    with open(os.path.join(REPO_ROOT, ".config")) as f:
        documented = f.read()

    assert len(sprite_options) >= 10
    missing = [option for option in sprite_options if "\n{:s}:".format(option) not in documented]
    assert missing == []


def test_sprite_options_parse_from_dot_config():

    config = cr.parse(os.path.join(REPO_ROOT, ".config"))

    assert config.detect_sprites is False
    assert config.sprite_confidence == pytest.approx(0.386)
    assert config.sprite_upload_url == ""
    assert config.sprite_upload_ff is True
    assert config.sprite_max_detections_per_night == 150
    assert config.sprite_model_path.endswith(os.path.join("share", "sprite_detector.tflite"))

    # The directory and retention options of the first version are gone
    assert not hasattr(config, "sprite_dir")
    assert not hasattr(config, "sprite_days_to_keep")


### Products ###


def test_image_names_are_accepted_by_the_server(tmp_path):

    # The first version named them <ff_name>_marked.png, with .fits still in the name, which the server
    #   refuses
    marked, unmarked = SpriteProducts.imagePaths(str(tmp_path), FF_NAME)

    for path in (marked, unmarked):
        name = os.path.basename(path)
        assert SERVER_IMAGE_RE.match(name), name
        assert ".fits" not in name


def test_payload_keeps_only_detections_with_astrometry(tmp_path):

    config = makeConfig(tmp_path)
    ff_start = FFfile.filenameToDatetime(FF_NAME)
    dets = [detection(FF_NAME, astrometry_ok=True), detection(FF_NAME, astrometry_ok=False)]

    payload, reason = SpriteProducts.buildSpritePayload(config, FF_NAME, ff_start, 25.0, dets, "XX0001_night")

    assert reason is None
    assert len(payload["detections"]) == 1
    assert payload["detections"][0]["azimuth"] == 250.5


def test_payload_is_none_without_any_astrometry(tmp_path):

    config = makeConfig(tmp_path)
    ff_start = FFfile.filenameToDatetime(FF_NAME)

    payload, reason = SpriteProducts.buildSpritePayload(
        config, FF_NAME, ff_start, 25.0, [detection(FF_NAME, astrometry_ok=False)], "night")

    assert payload is None
    assert "no platepar" in reason


def test_payload_matches_the_server_contract(tmp_path):

    config = makeConfig(tmp_path)
    config.stationID = "xx0001"
    ff_start = FFfile.filenameToDatetime(FF_NAME)

    payload, _ = SpriteProducts.buildSpritePayload(config, FF_NAME, ff_start, 25.0, [detection(FF_NAME)],
                                                   "night")

    # Station upper-cased, the FF block start with its milliseconds, the frame rate the indices refer to
    assert payload["station_id"] == "XX0001"
    assert payload["timestamp"] == "2026-08-28T03:14:22.123000Z"
    assert payload["fps"] == 25.0

    # No images inside the JSON (the first version embedded two base64 PNGs, over the server's 1 MiB limit)
    body = json.dumps(payload)
    assert "marked_image" not in body
    assert "unmarked_image" not in body
    assert len(body) < 4096

    # Only fields the server knows, and every one it requires
    det = payload["detections"][0]
    assert set(det) <= set(SpriteProducts.PAYLOAD_DETECTION_FIELDS)
    assert {"azimuth", "altitude", "timestamp", "frame_index"} <= set(det)
    assert all("box_azimuth_{:d}".format(n) in det for n in range(1, 5))


def test_payload_caps_detections_at_twenty(tmp_path):

    config = makeConfig(tmp_path)
    dets = [detection(FF_NAME, confidence=i/100.0) for i in range(30)]

    payload, _ = SpriteProducts.buildSpritePayload(config, FF_NAME, FFfile.filenameToDatetime(FF_NAME),
                                                   25.0, dets, "night")

    assert len(payload["detections"]) == 20
    assert min(det["confidence"] for det in payload["detections"]) == pytest.approx(0.10)


def test_upload_payload_bytes_are_final(tmp_path):

    # The worker posts these bytes on every attempt; they must not change once written, or every retry would
    #   be a new submission to the server
    config = makeConfig(tmp_path)
    payload, _ = SpriteProducts.buildSpritePayload(config, FF_NAME, FFfile.filenameToDatetime(FF_NAME),
                                                   25.0, [detection(FF_NAME)], "night")

    path = str(tmp_path/"payload.json")
    assert SpriteProducts.writeUploadPayload(payload, path)

    with open(path, "rb") as f:
        first = f.read()

    written = json.loads(first.decode("utf-8"))
    assert written["sent_at"].endswith("Z")
    assert not os.path.exists(path + ".tmp")

    # Reading it again gives the same bytes, and the payload dict itself was not modified
    with open(path, "rb") as f:
        assert f.read() == first
    assert "sent_at" not in payload


def test_csv_header_written_once_and_status_recorded(tmp_path):

    path = str(tmp_path/"night_sprites.csv")

    SpriteProducts.appendSpriteCSV(path, [detection(FF_NAME)], "confirmed")
    SpriteProducts.appendSpriteCSV(path, [detection(FF_NAME, astrometry_ok=False)], "burst_rejected")

    with open(path) as f:
        lines = f.read().splitlines()

    assert len(lines) == 3
    assert lines[0].split(";") == SpriteProducts.CSV_COLUMNS
    assert lines[1].split(";")[1] == "confirmed"
    assert lines[2].split(";")[1] == "burst_rejected"


def test_writers_do_not_raise_on_an_unwritable_path(tmp_path):

    missing_dir = str(tmp_path/"missing"/"x.csv")

    assert SpriteProducts.appendSpriteCSV(missing_dir, [detection(FF_NAME)], "confirmed") is False
    assert SpriteProducts.appendSpriteJSONL(missing_dir, {"a": 1}) is False
    assert SpriteProducts.writeUploadPayload({"a": 1}, missing_dir) is False


def test_image_writer_always_returns_a_pair(tmp_path):

    # The first version returned None on error and a pair on success, and its caller unpacked two values
    ff = FFStruct()
    ff.maxpixel = None

    result = SpriteProducts.writeDetectionImages(ff, [detection(FF_NAME)], str(tmp_path/"m.jpg"),
                                                 str(tmp_path/"u.jpg"))

    assert result == (None, None)


def test_images_written_as_jpeg(tmp_path):

    ff = writeFF(tmp_path, FF_NAME)
    marked, unmarked = SpriteProducts.imagePaths(str(tmp_path), FF_NAME)

    result = SpriteProducts.writeDetectionImages(ff, [detection(FF_NAME)], marked, unmarked)

    assert result == (marked, unmarked)
    for path in result:
        with open(path, "rb") as f:
            assert f.read(3) == b"\xff\xd8\xff"


def test_jsonl_serialises_datetimes_and_numpy(tmp_path):

    path = str(tmp_path/"night_sprites.json")
    record = {"when": datetime.datetime(2026, 8, 28, 3, 14, 22), "n": np.int64(3), "x": np.float32(1.5)}

    assert SpriteProducts.appendSpriteJSONL(path, record)

    with open(path) as f:
        loaded = json.loads(f.readline())

    assert loaded == {"when": "2026-08-28T03:14:22", "n": 3, "x": 1.5}


def test_old_upload_directories_are_pruned(tmp_path):

    config = makeConfig(tmp_path)

    old = SpriteProducts.uploadDir(config, str(tmp_path/"XX0001_20260701_000000_000000"))
    new = SpriteProducts.uploadDir(config, str(tmp_path/"XX0001_20260827_000000_000000"))

    long_ago = time.time() - 40*86400
    os.utime(old, (long_ago, long_ago))

    assert SpriteProducts.pruneUploadDirs(config) == 1
    assert not os.path.exists(old)
    assert os.path.isdir(new)


### Model download ###


def test_model_ready_checks_the_checksum(tmp_path, monkeypatch):

    config = makeConfig(tmp_path)

    with open(config.sprite_model_path, "wb") as f:
        f.write(b"model bytes")

    good = hashlib.sha256(b"model bytes").hexdigest()

    monkeypatch.setattr(DownloadSpriteModel, "KNOWN_MODEL_SHA256", {config.sprite_model_file: good})
    assert DownloadSpriteModel.spriteModelReady(config)

    monkeypatch.setattr(DownloadSpriteModel, "KNOWN_MODEL_SHA256", {config.sprite_model_file: "0"*64})
    assert not DownloadSpriteModel.spriteModelReady(config)

    # A model this code does not know is taken as it is
    monkeypatch.setattr(DownloadSpriteModel, "KNOWN_MODEL_SHA256", {})
    assert DownloadSpriteModel.spriteModelReady(config)


def test_known_model_hash_is_the_published_model():

    assert DownloadSpriteModel.KNOWN_MODEL_SHA256["sprite_detector.tflite"] == \
        "a73da954ab97a7b68683edc249eafe457f1085bfd8d619f05c9e0d94cdad831a"


class FakeSFTP(object):
    """ Serves one remote file, or none. """

    def __init__(self, content=None, fail=False):
        self.content = content
        self.fail = fail
        self.closed = False

    def lstat(self, path):
        if self.content is None:
            raise IOError("no such file")

    def get(self, remote_path, local_path):
        if self.fail:
            raise OSError("connection dropped")
        with open(local_path, "wb") as f:
            f.write(self.content)

    def close(self):
        self.closed = True


class FakeSSH(object):

    def close(self):
        pass


def fakeConnection(monkeypatch, sftp):
    """ Replace the SFTP connection with a fake one. """

    monkeypatch.setattr(DownloadSpriteModel, "getSSHAndSFTP", lambda *args, **kwargs: (FakeSSH(), sftp))


def withKey(config):
    with open(config.rsa_private_key, "w") as f:
        f.write("key")


def test_download_puts_a_verified_model_in_place(tmp_path, monkeypatch):

    config = makeConfig(tmp_path)
    withKey(config)
    content = b"the real model"
    monkeypatch.setattr(DownloadSpriteModel, "KNOWN_MODEL_SHA256",
                        {config.sprite_model_file: hashlib.sha256(content).hexdigest()})

    sftp = FakeSFTP(content)
    fakeConnection(monkeypatch, sftp)

    assert DownloadSpriteModel.downloadSpriteModel(config)

    with open(config.sprite_model_path, "rb") as f:
        assert f.read() == content
    assert not os.path.exists(config.sprite_model_path + ".download")
    assert sftp.closed


def test_download_with_the_wrong_checksum_is_discarded(tmp_path, monkeypatch):

    config = makeConfig(tmp_path)
    withKey(config)
    monkeypatch.setattr(DownloadSpriteModel, "KNOWN_MODEL_SHA256", {config.sprite_model_file: "0"*64})
    fakeConnection(monkeypatch, FakeSFTP(b"tampered"))

    assert not DownloadSpriteModel.downloadSpriteModel(config)
    assert not os.path.exists(config.sprite_model_path)
    assert not os.path.exists(config.sprite_model_path + ".download")


def test_download_fails_cleanly(tmp_path, monkeypatch):

    config = makeConfig(tmp_path)

    # No key at all
    assert not DownloadSpriteModel.downloadSpriteModel(config)

    # Not published on the server
    withKey(config)
    fakeConnection(monkeypatch, FakeSFTP(None))
    assert not DownloadSpriteModel.downloadSpriteModel(config)

    # Connection drops mid-transfer
    fakeConnection(monkeypatch, FakeSFTP(b"x", fail=True))
    assert not DownloadSpriteModel.downloadSpriteModel(config)
    assert not os.path.exists(config.sprite_model_path + ".download")


### Detector ###


def test_unix_time_ignores_the_local_timezone(monkeypatch):

    # datetime.timestamp() on a naive value assumes local time; on a station set to Toronto that is 4 hours
    #   off
    monkeypatch.setenv("TZ", "America/Toronto")
    time.tzset()

    dt = datetime.datetime(2026, 7, 15, 3, 0, 0)

    try:
        assert SpriteDetector.unixTime(dt) == 1784084400.0
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()


def test_platepar_found_in_the_config_directory(tmp_path):

    # config_file_path is the directory of the config file. The first version took its dirname, looking one
    #   level too high, so on a real station it never found the platepar and live astrometry never ran.
    config = makeConfig(tmp_path)
    night = tmp_path/"night"
    os.makedirs(str(night))

    from RMS.Formats.Platepar import Platepar
    pp = Platepar()
    pp.station_code = "XX0001"
    pp.write(os.path.join(config.config_file_path, config.platepar_name))

    platepar, path = SpriteDetector.loadStationPlatepar(config, str(night))

    assert platepar is not None
    assert path == os.path.join(config.config_file_path, config.platepar_name)


def test_platepar_in_the_night_directory_wins(tmp_path):

    config = makeConfig(tmp_path)
    night = tmp_path/"night"
    os.makedirs(str(night))

    from RMS.Formats.Platepar import Platepar
    for directory in (config.config_file_path, str(night)):
        Platepar().write(os.path.join(directory, config.platepar_name))

    _, path = SpriteDetector.loadStationPlatepar(config, str(night))

    assert path == os.path.join(str(night), config.platepar_name)


def test_no_platepar(tmp_path):

    config = makeConfig(tmp_path)

    assert SpriteDetector.loadStationPlatepar(config, str(tmp_path)) == (None, None)


def fakeDetections(monkeypatch, hits):
    """ Replace the model: FF names in hits get one detection, others none. Calibration passes through. """

    def detect(ff, ff_name, config, model_path, mask=None):
        return [detection(ff_name)] if ff_name in hits else []

    def calibrate(dets, platepar, usable, reason):
        for det in dets:
            det["astrometry_ok"] = True

    monkeypatch.setattr(SpriteDetector, "detectSpritesInFF", detect)
    monkeypatch.setattr(SpriteDetector, "calibrateSpriteDetections", calibrate)


def makeNight(tmp_path, n_ff, spacing=600):
    """ A night directory with n_ff FF files spaced apart in time. """

    night = tmp_path/"XX0001_20260828_031400_000000"
    os.makedirs(str(night))

    names = [ffName(i*spacing) for i in range(n_ff)]
    for name in names:
        writeFF(night, name)

    return str(night), names


def test_recorder_writes_products_and_hands_payload_to_the_uploader(tmp_path, monkeypatch):

    config = makeConfig(tmp_path)
    night, names = makeNight(tmp_path, 3)
    fakeDetections(monkeypatch, {names[1]})
    uploader = FakeUploader()

    recorder = SpriteDetector.NightRecorder(config, night, uploader, {"JD": None})
    detector = SpriteDetector.NightDetector(config, night, recorder, None, None)

    for name in names:
        detector.processFF(name, "model")
    detector.finish()

    assert recorder.n_confirmed == 1

    # CSV and JSON record in the night directory, the images under the server's names
    with open(recorder.csv_path) as f:
        assert "confirmed" in f.read()
    with open(recorder.jsonl_path) as f:
        assert json.loads(f.readline())["ff_name"] == names[1]

    marked, unmarked = SpriteProducts.imagePaths(night, names[1])
    assert os.path.isfile(marked) and os.path.isfile(unmarked)

    # One payload queued, with the files the server may ask for later
    assert len(uploader.enqueued) == 1
    ff_name, payload_path, files = uploader.enqueued[0]
    assert ff_name == names[1]
    assert os.path.isfile(payload_path)
    assert files == {"ff": os.path.join(night, names[1]), "marked": marked, "unmarked": unmarked}


def test_detection_without_astrometry_is_recorded_but_not_sent(tmp_path, monkeypatch, caplog):

    config = makeConfig(tmp_path)
    night, names = makeNight(tmp_path, 4)

    def detect(ff, ff_name, config, model_path, mask=None):
        return [detection(ff_name, astrometry_ok=False)] if ff_name in names[1:3] else []

    monkeypatch.setattr(SpriteDetector, "detectSpritesInFF", detect)
    monkeypatch.setattr(SpriteDetector, "calibrateSpriteDetections", lambda *args: None)

    uploader = FakeUploader()
    recorder = SpriteDetector.NightRecorder(config, night, uploader, {})
    detector = SpriteDetector.NightDetector(config, night, recorder, None, "no platepar")

    for name in names:
        detector.processFF(name, "model")
    detector.finish()

    assert recorder.n_confirmed == 2
    assert uploader.enqueued == []
    assert recorder.n_not_sent == 2

    # Said once, then counted
    warnings = [r for r in caplog.records if "not sent to the server" in r.getMessage()]
    assert len(warnings) == 1


def test_burst_is_recorded_as_rejected(tmp_path, monkeypatch):

    config = makeConfig(tmp_path)
    night, names = makeNight(tmp_path, 6, spacing=10)
    fakeDetections(monkeypatch, set(names[:5]))
    uploader = FakeUploader()

    recorder = SpriteDetector.NightRecorder(config, night, uploader, {})
    detector = SpriteDetector.NightDetector(config, night, recorder, None, None)

    for name in names:
        detector.processFF(name, "model")
    detector.finish()

    assert recorder.n_confirmed == 0
    assert recorder.n_rejected == 5
    assert uploader.enqueued == []


def test_images_are_capped_per_night(tmp_path, monkeypatch):

    config = makeConfig(tmp_path)
    config.sprite_max_images = 2
    night, names = makeNight(tmp_path, 4)
    fakeDetections(monkeypatch, set(names))

    recorder = SpriteDetector.NightRecorder(config, night, FakeUploader(), {})
    detector = SpriteDetector.NightDetector(config, night, recorder, None, None)

    for name in names:
        detector.processFF(name, "model")
    detector.finish()

    assert recorder.n_confirmed == 4
    assert recorder.n_images == 2
    assert len([n for n in os.listdir(night) if n.endswith("_marked.jpg")]) == 2


def test_nightly_cap_stops_inference_and_still_decides_pending(tmp_path, monkeypatch):

    config = makeConfig(tmp_path)
    config.sprite_max_detections_per_night = 2
    night, names = makeNight(tmp_path, 5)

    calls = []

    def detect(ff, ff_name, config, model_path, mask=None):
        calls.append(ff_name)
        return [detection(ff_name)]

    monkeypatch.setattr(SpriteDetector, "detectSpritesInFF", detect)
    monkeypatch.setattr(SpriteDetector, "calibrateSpriteDetections", lambda *args: None)

    recorder = SpriteDetector.NightRecorder(config, night, FakeUploader(), {})
    detector = SpriteDetector.NightDetector(config, night, recorder, None, None)

    for name in names:
        detector.processFF(name, "model")
    detector.finish()

    # The model ran on the first two only, and both of those were still decided
    assert calls == names[:2]
    assert detector.capped
    assert recorder.n_confirmed == 2


def test_live_loop_exits_promptly_on_a_quiet_night(tmp_path, monkeypatch):

    # The previous version only left its loop once more than 150 FF files had detections. On a normal night it
    #   never left, so stop() timed out, the process was killed and the last minute of detections was lost.
    config = makeConfig(tmp_path)
    night, names = makeNight(tmp_path, 3)
    fakeDetections(monkeypatch, {names[0]})

    uploader = FakeUploader()
    monkeypatch.setattr(SpriteDetector, "SpriteUploadWorker", lambda config: uploader)
    monkeypatch.setattr(SpriteDetector.SpriteDetector, "prepareModel", lambda self, path: True)

    sprite_detector = SpriteDetector.SpriteDetector(night, config)
    for name in names:
        sprite_detector.input_queue.put((night, name))

    # Queue feeding is asynchronous; wait until the items are visible
    deadline = time.time() + 5
    while sprite_detector.input_queue.empty() and time.time() < deadline:
        time.sleep(0.01)

    sprite_detector.exit.set()

    # Run the loop in this process, in a thread, so it can be timed
    runner = threading.Thread(target=sprite_detector.runDetection)
    t_beg = time.time()
    runner.start()
    runner.join(timeout=30)

    assert not runner.is_alive(), "the detection loop did not exit"
    assert time.time() - t_beg < 20

    # It confirmed the one detection on the way out and stopped the uploader
    assert uploader.started and uploader.stopped
    with open(SpriteProducts.csvPath(night)) as f:
        assert "confirmed" in f.read()


def test_disabled_detector_keeps_the_queue_empty(tmp_path):

    # Without a model the process keeps draining its queue, otherwise the Compressor warns all night that the
    #   detector is behind
    config = makeConfig(tmp_path)
    sprite_detector = SpriteDetector.SpriteDetector(str(tmp_path), config)

    for i in range(3):
        sprite_detector.input_queue.put((str(tmp_path), ffName(i)))

    runner = threading.Thread(target=sprite_detector.discardQueue)
    runner.start()

    deadline = time.time() + 5
    while (not sprite_detector.input_queue.empty()) and time.time() < deadline:
        time.sleep(0.05)

    sprite_detector.exit.set()
    runner.join(timeout=5)

    assert not runner.is_alive()
    assert sprite_detector.input_queue.empty()


class CrashingDetector(SpriteDetector.SpriteDetector):
    """ A detector whose work fails at once. """

    def runDetection(self):
        raise RuntimeError("boom")


def test_stop_returns_quickly_when_the_process_died(tmp_path):

    # The previous stop() waited for a flag that a crashed process never set, burning its full timeout
    config = makeConfig(tmp_path)
    sprite_detector = CrashingDetector(str(tmp_path), config)
    sprite_detector.start()
    sprite_detector.join(timeout=20)

    t_beg = time.time()
    sprite_detector.stop()

    assert time.time() - t_beg < 10
    assert not sprite_detector.is_alive()


def test_offline_run_refuses_to_record_a_night_twice(tmp_path, monkeypatch):

    config = makeConfig(tmp_path)
    night, names = makeNight(tmp_path, 2)
    fakeDetections(monkeypatch, {names[0]})
    monkeypatch.setattr(SpriteDetector, "SPRITE_TFLITE_AVAILABLE", True)
    monkeypatch.setattr(SpriteDetector, "spriteModelReady", lambda config: True)

    first = SpriteDetector.runSpriteDetectionDirectory(night, config)
    assert first.n_confirmed == 1

    # Again without --overwrite: refused, nothing duplicated
    assert SpriteDetector.runSpriteDetectionDirectory(night, config) is None
    with open(SpriteProducts.csvPath(night)) as f:
        assert f.read().count("confirmed") == 1

    # With --overwrite: replaced, still one record
    again = SpriteDetector.runSpriteDetectionDirectory(night, config, overwrite=True)
    assert again.n_confirmed == 1
    with open(SpriteProducts.csvPath(night)) as f:
        assert f.read().count("confirmed") == 1
