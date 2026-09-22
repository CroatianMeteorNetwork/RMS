""" Tests for RMS/SpriteUpload.py, the signed upload client of the spritemap server.

    Every test that talks HTTP uses Tests/SpriteFakeServer.py on 127.0.0.1; there is no network access.
    Keys are generated in the tests and written to tmp_path, both as PEM and as OpenSSH files.
"""

from __future__ import print_function, division, absolute_import

import hashlib
import importlib
import io
import json
import logging
import os
import random
import re
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

pytest.importorskip("cryptography")

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa

import RMS.ConfigReader as cr
import RMS.SpriteUpload as su
from Tests.SpriteFakeServer import FakeSpriteServer, verifySignature


STATION = "XX0001"
FF_NAME = "FF_XX0001_20260828_031422_123_0000512.fits"
FF_NAME_2 = "FF_XX0001_20260828_031530_456_0000768.fits"

VECTORS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                            "spritemap_canonical_vectors.json")

# Copied from spritemap/storage.py of spritemap-data-pipeline
SERVER_IMAGE_FILE_RE = re.compile(
    r"^(FF_[A-Z0-9]{3,8}_\d{8}_\d{6}_\d{3,6}_\d{7})_(marked|unmarked)\.(jpg|jpeg|png)$")



# ###########################################################################
#                       Fixtures and helpers
# ###########################################################################

# Keys are expensive to generate, so one of each per test session
_KEYS = {}


def _key(kind):
    if kind not in _KEYS:
        if kind == "rsa":
            _KEYS[kind] = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        elif kind == "rsa1024":
            _KEYS[kind] = rsa.generate_private_key(public_exponent=65537, key_size=1024)
        elif kind == "ed25519":
            _KEYS[kind] = ed25519.Ed25519PrivateKey.generate()
        elif kind == "p256":
            _KEYS[kind] = ec.generate_private_key(ec.SECP256R1())
        elif kind == "p384":
            _KEYS[kind] = ec.generate_private_key(ec.SECP384R1())
    return _KEYS[kind]


def _writeKey(tmp_path, key, fmt="pem", password=None, name=None):
    """ Write a private key as a PEM (TraditionalOpenSSL or PKCS8) or OpenSSH file. """

    if password is None:
        enc = serialization.NoEncryption()
    else:
        enc = serialization.BestAvailableEncryption(password)

    if fmt == "openssh":
        data = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.OpenSSH, enc)
    elif isinstance(key, rsa.RSAPrivateKey):
        data = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                 enc)
    else:
        data = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, enc)

    path = str(tmp_path / (name or "id_{}_{}".format(fmt, id(key))))
    with open(path, "wb") as f:
        f.write(data)
    return path


def _makeConfig(tmp_path, key_path, url, **kwargs):
    config = cr.Config()
    config.stationID = STATION
    config.data_dir = str(tmp_path / "data")
    config.reboot_lock_file = ".reboot_lock"
    config.rsa_private_key = key_path
    config.sprite_upload_url = url
    config.sprite_upload_allow_insecure = True
    config.sprite_upload_timeout = 5.0
    config.sprite_upload_ff = True
    for k, v in kwargs.items():
        setattr(config, k, v)
    if not os.path.isdir(config.data_dir):
        os.makedirs(config.data_dir)
    return config


class _ListHandler(logging.Handler):
    def __init__(self):
        logging.Handler.__init__(self, level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)

    def messages(self, level=None):
        return [r.getMessage() for r in self.records if level is None or r.levelno == level]


@pytest.fixture
def logs():
    logger = logging.getLogger("rmslogger")
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    handler = _ListHandler()
    logger.addHandler(handler)
    yield handler
    logger.removeHandler(handler)
    logger.setLevel(old_level)


@pytest.fixture
def rsa_key_path(tmp_path):
    return _writeKey(tmp_path, _key("rsa"), name="id_rsa")


@pytest.fixture
def server():
    srv = FakeSpriteServer(_key("rsa").public_key(), STATION)
    yield srv
    srv.close()


@pytest.fixture
def worker(tmp_path, rsa_key_path, server):
    w = su.SpriteUploadWorker(_makeConfig(tmp_path, rsa_key_path, server.url))
    assert w.isEnabled(), w.disabled_reason
    return w


def _writeFile(path, data):
    d = os.path.dirname(path)
    if not os.path.isdir(d):
        os.makedirs(d)
    with open(path, "wb") as f:
        f.write(data)
    return path


def _payload(tmp_path, ff_name=FF_NAME, extra=""):
    # Odd spacing on purpose: the bytes must go out exactly as written, not re-encoded
    body = '[ {"station_id":"XX0001",  "ff_name": "%s", "sent_at": "2026-08-28T03:15:00Z"%s} ]\n' % (
        ff_name, extra)
    return _writeFile(str(tmp_path / "night" / (ff_name + ".json")), body.encode("utf-8"))


def _products(tmp_path, ff_name=FF_NAME, ff_size=4096, img_size=1000, bz2=False, img_names=None):
    """ Write an FF file and its two images, return the files dict for enqueueDetections(). """

    night = tmp_path / "night"
    ff_path = _writeFile(str(night / ff_name), os.urandom(ff_size))
    if bz2:
        _writeFile(ff_path + ".bz2", b"BZh9" + os.urandom(100))

    if img_names is None:
        stem = ff_name[:-len(".fits")]
        img_names = (stem + "_marked.jpg", stem + "_unmarked.png")

    marked = _writeFile(str(night / img_names[0]), os.urandom(img_size))
    unmarked = _writeFile(str(night / img_names[1]), os.urandom(img_size))
    return {"ff": ff_path, "marked": marked, "unmarked": unmarked}


def _detections(rec):
    return rec.method == "POST" and rec.path == "/api/v1/detections"



# ###########################################################################
#                       Canonical string and signatures
# ###########################################################################

def testCanonicalMatchesGoldenVectors():
    with open(VECTORS_PATH) as f:
        vectors = json.load(f)

    assert vectors["protocol_version"] == su.PROTOCOL_VERSION
    assert len(vectors["vectors"]) >= 5

    for v in vectors["vectors"]:

        # The digest in the file belongs to the body in the file
        assert su.hashBody(bytes.fromhex(v["body_hex"])) == v["body_sha256"]

        canonical = su.buildCanonicalString(v["method"], v["path"], v["station_id"], v["timestamp"],
                                            v["body_sha256"])
        assert canonical == v["canonical"].encode("ascii"), v["name"]
        assert not canonical.endswith(b"\n")


def testEmptyBodyDigest():
    empty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert su.hashBody(b"") == empty

    canonical = su.buildCanonicalString("get", "/api/v1/stations/XX0001/requests", "xx0001", 1788000000.7,
                                        empty)
    assert canonical == ("SPRITEMAP-V1\nGET\n/api/v1/stations/XX0001/requests\nXX0001\n1788000000\n"
                         + empty).encode("ascii")


def testCanonicalCrossCheckWithServer():
    repo = os.environ.get("SPRITEMAP_REPO", "/home/dvida/source/spritemap-data-pipeline")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    signing = pytest.importorskip("spritemap.signing")

    rng = random.Random(1234)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

    for _ in range(300):
        method = rng.choice(["GET", "POST", "PUT", "get", "Post", "delete"])
        path = "/api/v1/" + "".join(rng.choice(alphabet + "_./-") for _ in range(rng.randint(0, 60)))
        station = "".join(rng.choice(alphabet) for _ in range(rng.randint(3, 8)))
        timestamp = rng.choice([rng.randint(0, 2**33), rng.uniform(0, 2**33)])
        body = os.urandom(rng.randint(0, 200))
        digest = hashlib.sha256(body).hexdigest()
        if rng.random() < 0.3:
            digest = digest.upper()

        assert su.hashBody(body) == signing.hashBody(body)
        assert (su.buildCanonicalString(method, path, station, timestamp, digest)
                == signing.buildCanonicalString(method, path, station, timestamp, digest))

    # The server must accept what the client signs, with every key type
    for kind in ("rsa", "ed25519", "p256", "p384"):
        key = _key(kind)
        canonical = su.buildCanonicalString("POST", "/api/v1/detections", STATION, 1788000000,
                                            su.hashBody(b"x"))
        assert signing.verifyCanonical(key.public_key(), canonical, su.signCanonical(key, canonical))


@pytest.mark.parametrize("kind", ["rsa", "ed25519", "p256", "p384"])
def testSignaturesVerify(kind):
    key = _key(kind)
    canonical = su.buildCanonicalString("PUT", "/api/v1/files/" + FF_NAME, STATION, 1788000000,
                                        su.hashBody(b"data"))
    signature = su.signCanonical(key, canonical)
    public = key.public_key()

    # Verify with the algorithm spelled out, independently of the client code
    if kind == "rsa":
        public.verify(signature, canonical, padding.PKCS1v15(), hashes.SHA256())
    elif kind == "ed25519":
        public.verify(signature, canonical)
    elif kind == "p256":
        public.verify(signature, canonical, ec.ECDSA(hashes.SHA256()))
    else:
        public.verify(signature, canonical, ec.ECDSA(hashes.SHA384()))

    assert verifySignature(public, canonical, signature)
    assert not verifySignature(public, canonical + b"x", signature)


@pytest.mark.parametrize("kind,fmt", [("rsa", "pem"), ("rsa", "openssh"), ("ed25519", "pem"),
                                      ("ed25519", "openssh"), ("p256", "openssh"), ("p384", "pem")])
def testKeyFormatsLoad(tmp_path, kind, fmt):
    key, reason = su.loadPrivateKey(_writeKey(tmp_path, _key(kind), fmt=fmt))
    assert reason is None
    assert key is not None



# ###########################################################################
#                       Disabling on bad setup
# ###########################################################################

@pytest.mark.parametrize("fmt", ["pem", "openssh"])
def testPassphraseKeyDisables(tmp_path, logs, fmt):
    path = _writeKey(tmp_path, _key("rsa"), fmt=fmt, password=b"secret")
    w = su.SpriteUploadWorker(_makeConfig(tmp_path, path, "https://sprites.example.org"))

    assert not w.isEnabled()
    assert "passphrase" in w.disabled_reason

    # Nothing raises afterwards either
    w.start()
    w.enqueueDetections(FF_NAME, str(tmp_path / "missing.json"), {})
    w.stop(drain_timeout=1)
    assert len([m for m in logs.messages(logging.WARNING) if "disabled" in m]) == 1


def testMissingKeyDisables(tmp_path):
    w = su.SpriteUploadWorker(_makeConfig(tmp_path, str(tmp_path / "nope"), "https://sprites.example.org"))
    assert not w.isEnabled()
    assert "cannot be read" in w.disabled_reason


def testShortRsaKeyDisables(tmp_path):
    path = _writeKey(tmp_path, _key("rsa1024"))
    w = su.SpriteUploadWorker(_makeConfig(tmp_path, path, "https://sprites.example.org"))
    assert not w.isEnabled()
    assert "1024 bits" in w.disabled_reason


def testGarbageKeyDisables(tmp_path):
    path = _writeFile(str(tmp_path / "garbage"), b"not a key at all")
    w = su.SpriteUploadWorker(_makeConfig(tmp_path, path, "https://sprites.example.org"))
    assert not w.isEnabled()
    assert "cannot be parsed" in w.disabled_reason


def testNoCryptographyDisables(tmp_path, rsa_key_path, monkeypatch):

    # Make every cryptography import fail, then import the module afresh
    for name in list(sys.modules):
        if name == "cryptography" or name.startswith("cryptography."):
            monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "cryptography", None)
    monkeypatch.delitem(sys.modules, "RMS.SpriteUpload")

    fresh = importlib.import_module("RMS.SpriteUpload")
    assert fresh is not su

    w = fresh.SpriteUploadWorker(_makeConfig(tmp_path, rsa_key_path, "https://sprites.example.org"))
    assert not w.isEnabled()
    assert "cryptography" in w.disabled_reason


def testLowerCaseStationDisables(tmp_path, rsa_key_path):
    w = su.SpriteUploadWorker(_makeConfig(tmp_path, rsa_key_path, "https://sprites.example.org",
                                          stationID="xx0001"))
    assert not w.isEnabled()
    assert "upper case" in w.disabled_reason
    assert "XX0001" in w.disabled_reason


def testEmptyUrlDisables(tmp_path, rsa_key_path):
    assert not su.SpriteUploadWorker(_makeConfig(tmp_path, rsa_key_path, "")).isEnabled()


def testUrlChecks(tmp_path, rsa_key_path):
    assert su.checkBaseUrl("https://sprites.example.org/")[0] == "https://sprites.example.org"
    assert su.checkBaseUrl("https://sprites.example.org:8443")[0] == "https://sprites.example.org:8443"

    # Plain http is refused except for a loopback literal with the flag
    assert su.checkBaseUrl("http://sprites.example.org", allow_insecure=True)[0] is None
    assert su.checkBaseUrl("http://localhost:8000", allow_insecure=True)[0] is None
    assert su.checkBaseUrl("http://127.0.0.1:8000")[0] is None
    assert su.checkBaseUrl("http://127.0.0.1:8000", allow_insecure=True)[0] == "http://127.0.0.1:8000"
    assert su.checkBaseUrl("http://[::1]:8000", allow_insecure=True)[0] == "http://[::1]:8000"

    # The signature binds the path, so a path prefix would break it
    assert su.checkBaseUrl("https://example.org/sprites")[0] is None
    assert su.checkBaseUrl("https://example.org/?a=1")[0] is None
    assert su.checkBaseUrl("ftp://example.org")[0] is None
    assert su.checkBaseUrl("https://user:pw@example.org")[0] is None

    # And the worker uses the same rules
    assert not su.SpriteUploadWorker(_makeConfig(tmp_path, rsa_key_path,
                                                 "http://sprites.example.org")).isEnabled()
    assert not su.SpriteUploadWorker(_makeConfig(tmp_path, rsa_key_path, "http://127.0.0.1:9",
                                                 sprite_upload_allow_insecure=False)).isEnabled()
    assert su.SpriteUploadWorker(_makeConfig(tmp_path, rsa_key_path, "http://127.0.0.1:9")).isEnabled()
    assert not su.SpriteUploadWorker(_makeConfig(tmp_path, rsa_key_path, "https://x.org/api")).isEnabled()



# ###########################################################################
#                       Detections
# ###########################################################################

def testDetectionsPostSignedExactBytes(tmp_path, worker, server):
    path = _payload(tmp_path)
    worker.enqueueDetections(FF_NAME, path, _products(tmp_path))
    worker._runCycle(time.time())

    posts = [r for r in server.requests if _detections(r)]
    assert len(posts) == 1
    rec = posts[0]

    with open(path, "rb") as f:
        assert rec.body == f.read()

    assert rec.signature_ok
    assert rec.headers["x-station-id"] == STATION
    assert rec.headers["content-type"] == "application/json"
    assert rec.headers["content-length"] == str(len(rec.body))
    assert worker._items == []

    # No FF is ever sent without the server asking for it
    assert server.byPath("PUT") == []


def testDuplicateResponseRemovesItem(tmp_path, worker, server, logs):
    server.script("POST", "/api/v1/detections", 200, {"success": True, "duplicate": True, "submission_id": 3})
    worker.enqueueDetections(FF_NAME, _payload(tmp_path), {})
    worker._runCycle(time.time())

    assert worker._items == []
    assert any("already on the server" in m for m in logs.messages(logging.INFO))


def testRedirectNotFollowed(tmp_path, worker, server, logs):
    server.script("POST", "/api/v1/detections", 302, b"", headers={"Location": server.url + "/api/v1/other"})
    worker.enqueueDetections(FF_NAME, _payload(tmp_path), {})
    worker._runCycle(time.time())

    # Exactly the one request; nothing went to the redirect target
    assert [r.path for r in server.requests if r.method == "POST"] == ["/api/v1/detections"]
    assert not any(r.path == "/api/v1/other" for r in server.requests)
    assert worker._items == []
    assert any("HTTP 302" in m for m in logs.messages(logging.ERROR))



# ###########################################################################
#                       File requests and uploads
# ###########################################################################

def testRequestedFilesUploadedRaw(tmp_path, worker, server):
    files = _products(tmp_path)
    server.wanted.append(FF_NAME)
    worker.enqueueDetections(FF_NAME, _payload(tmp_path), files)
    worker._runCycle(time.time())
    worker._runCycle(time.time())

    puts = server.byPath("PUT")
    stem = FF_NAME[:-5]
    assert [r.path for r in puts] == ["/api/v1/files/" + FF_NAME, "/api/v1/files/" + stem + "_marked.jpg",
                                      "/api/v1/files/" + stem + "_unmarked.png"]

    for rec, key in zip(puts, ("ff", "marked", "unmarked")):
        with open(files[key], "rb") as f:
            data = f.read()
        assert rec.body == data
        assert rec.signature_ok
        assert rec.headers["content-length"] == str(len(data))
        assert "transfer-encoding" not in rec.headers
        assert rec.headers["content-type"] == "application/octet-stream"
        assert b"boundary" not in rec.body and "boundary" not in rec.headers["content-type"]

    assert server.wanted == []
    assert worker._items == []


def testCompressedFFPreferred(tmp_path, worker, server):
    files = _products(tmp_path, bz2=True)
    server.wanted.append(FF_NAME)
    worker.enqueueDetections(FF_NAME, _payload(tmp_path), files)
    worker._runCycle(time.time())

    ff_puts = [r for r in server.byPath("PUT") if ".fits" in r.path]
    assert [r.path for r in ff_puts] == ["/api/v1/files/" + FF_NAME + ".bz2"]
    with open(files["ff"] + ".bz2", "rb") as f:
        assert ff_puts[0].body == f.read()


def testOversizeFilesNotSent(tmp_path, worker, server, logs):
    files = _products(tmp_path, ff_size=10, img_size=10)

    # Sparse files, so the test does not write 35 MiB
    with open(files["ff"], "r+b") as f:
        f.truncate(su.MAX_FF_BYTES + 1)
    with open(files["marked"], "r+b") as f:
        f.truncate(su.MAX_IMAGE_BYTES + 1)

    server.wanted.append(FF_NAME)
    worker.enqueueDetections(FF_NAME, _payload(tmp_path), files)
    worker._runCycle(time.time())

    names = [r.path.rsplit("/", 1)[1] for r in server.byPath("PUT")]
    assert FF_NAME not in names
    assert not any("_marked" in n for n in names)
    assert any("above the server limit" in m for m in logs.messages(logging.WARNING))


def testImageNamesMatchServerPattern(tmp_path, worker, server):

    # Local names the products code might produce, including a .fits in the middle
    files = _products(tmp_path, img_names=(FF_NAME + "_marked.JPG", "sprite_unmarked_0.png"))
    server.wanted.append(FF_NAME)
    worker.enqueueDetections(FF_NAME, _payload(tmp_path), files)
    worker._runCycle(time.time())

    images = [r.path.rsplit("/", 1)[1] for r in server.byPath("PUT") if "marked" in r.path]
    assert len(images) == 2
    for name in images:
        assert SERVER_IMAGE_FILE_RE.match(name), name
        assert ".fits" not in name


def testRequestsGetHasNoBody(tmp_path, worker, server):
    worker._runCycle(time.time())

    gets = server.byPath("GET")
    assert len(gets) == 1
    rec = gets[0]
    assert rec.path == "/api/v1/stations/XX0001/requests"
    assert rec.headers["x-station-id"] == "XX0001"
    assert rec.body == b""
    assert "content-length" not in rec.headers
    assert "transfer-encoding" not in rec.headers
    assert rec.signature_ok
    assert rec.body_sha256 == hashlib.sha256(b"").hexdigest()


def testUnknownRequestIgnored(tmp_path, worker, server, logs):
    server.wanted.append(FF_NAME_2)
    worker.enqueueDetections(FF_NAME, _payload(tmp_path), _products(tmp_path))
    worker._runCycle(time.time())

    assert server.byPath("PUT") == []
    assert any(FF_NAME_2 in m and "does not know" in m for m in logs.messages(logging.WARNING))

    # The warning is not repeated at every poll
    worker._pollRequests(time.time())
    assert len([m for m in logs.messages(logging.WARNING) if FF_NAME_2 in m]) == 1


def testUploadFfFalseNeverPolls(tmp_path, rsa_key_path, server):
    w = su.SpriteUploadWorker(_makeConfig(tmp_path, rsa_key_path, server.url, sprite_upload_ff=False))
    server.wanted.append(FF_NAME)
    w.enqueueDetections(FF_NAME, _payload(tmp_path), _products(tmp_path))
    w._runCycle(time.time())

    assert server.byPath("GET") == []
    assert server.byPath("PUT") == []
    assert len([r for r in server.requests if _detections(r)]) == 1



# ###########################################################################
#                       Error handling
# ###########################################################################

@pytest.mark.parametrize("status,body", [
    (502, b"<html><head><title>502 Bad Gateway</title></head><body>nginx</body></html>"),
    (500, b""),
    (500, b"\xff\xfe\xfa not utf-8 \x80"),
    (400, b"\xff\xfe"),
])
def testNonJsonErrorBodies(tmp_path, worker, server, status, body):
    server.script("POST", "/api/v1/detections", status, body, headers={"X-Request-Id": "abc123"})
    worker.enqueueDetections(FF_NAME, _payload(tmp_path), {})
    worker._runCycle(time.time())

    if status >= 500:
        assert len(worker._items) == 1 and worker._items[0]["attempts"] == 1
    else:
        assert worker._items == []


def testParseErrorBodyNeverRaises():
    assert su.parseErrorBody(b"") == (None, None, None)
    assert su.parseErrorBody(b"\xff") == (None, None, None)
    assert su.parseErrorBody(b"[1, 2]") == (None, None, None)
    assert su.parseErrorBody(b'{"error": "x"}', {"X-Request-Id": "r1"}) == (None, None, "r1")
    code, message, rid = su.parseErrorBody(
        b'{"error": {"code": "bad_payload", "message": "line1\\nline2"}, "request_id": "r2"}')
    assert (code, rid) == ("bad_payload", "r2")
    assert "\n" not in message


class _CountingReader(io.RawIOBase):
    """ A 10 MiB body that counts how much was read from it. """

    def __init__(self, size):
        self.size = size
        self.read_bytes = 0

    def readable(self):
        return True

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.read_bytes
        n = max(0, min(n, self.size - self.read_bytes))
        self.read_bytes += n
        return b"x"*n

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def testLargeErrorBodyReadBounded(tmp_path, worker, monkeypatch):
    import urllib.error

    reader = _CountingReader(10*1024*1024)

    def fakeOpen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 500, "boom", {"X-Request-Id": "big1"}, reader)

    monkeypatch.setattr(worker._opener, "open", fakeOpen)
    result = worker._request("POST", "/api/v1/detections", body=b"{}")

    assert result.status == 500
    assert result.request_id == "big1"
    assert reader.read_bytes <= su.ERROR_BODY_LIMIT


def testLargeErrorBodyOverHttp(tmp_path, worker, server):
    server.script("POST", "/api/v1/detections", 500, b"y"*(10*1024*1024))
    worker.enqueueDetections(FF_NAME, _payload(tmp_path), {})
    worker._runCycle(time.time())
    assert worker._items[0]["attempts"] == 1


@pytest.mark.parametrize("status,code", [(400, "bad_payload"), (400, "too_many_frames"),
                                         (403, "station_mismatch"),
                                         (404, "not_found"), (405, None), (409, "content_mismatch"),
                                         (413, "payload_too_large")])
def testPermanentStatusDropped(tmp_path, worker, server, logs, status, code):
    body = {"error": {"code": code, "message": "refused"}, "request_id": "rid{}".format(status)}
    server.script("POST", "/api/v1/detections", status, body)
    worker.enqueueDetections(FF_NAME, _payload(tmp_path), {})
    worker._runCycle(time.time())
    worker._runCycle(time.time() + 7200)

    assert len([r for r in server.requests if _detections(r)]) == 1
    assert worker._items == []

    errors = logs.messages(logging.ERROR)
    assert len(errors) == 1
    assert "HTTP {:d}".format(status) in errors[0]
    assert "rid{}".format(status) in errors[0]
    if code:
        assert code in errors[0]


def testTransientBackoff(tmp_path, rsa_key_path, server, monkeypatch):
    w = su.SpriteUploadWorker(_makeConfig(tmp_path, rsa_key_path, server.url, sprite_upload_ff=False))
    server.script("POST", "/api/v1/detections", 503, b"busy", times=10)
    w.enqueueDetections(FF_NAME, _payload(tmp_path), {})

    now = time.time()
    expected = [60, 120, 240, 480, 960, 1800, 1800, 1800]
    for attempt, base in enumerate(expected, 1):
        w._runCycle(now)
        rec = w._items[0]
        assert rec["attempts"] == attempt
        delay = rec["next_attempt"] - now
        assert 0.8*base - 1e-6 <= delay <= 1.2*base + 1e-6

        # Not retried before it is due
        n_before = len(server.requests)
        w._runCycle(now + 0.5*base)
        assert len(server.requests) == n_before

        # The absolute time is on disk
        with open(w.queue_path) as f:
            on_disk = [json.loads(line) for line in f if line.strip()]
        assert [r["next_attempt"] for r in on_disk if r["kind"] == "detections"] == [rec["next_attempt"]]

        now = rec["next_attempt"]


def testBackoffFormula():
    class Fixed(object):
        def __init__(self, v):
            self.v = v

        def uniform(self, a, b):
            return self.v

    assert su.backoffDelay(1, Fixed(1.0)) == 60
    assert su.backoffDelay(3, Fixed(1.0)) == 240
    assert su.backoffDelay(50, Fixed(1.0)) == 1800
    assert su.backoffDelay(1, Fixed(0.8)) == pytest.approx(48)


def testUnauthorizedKeepsItemAndLogsOnce(tmp_path, rsa_key_path, server, logs):
    w = su.SpriteUploadWorker(_makeConfig(tmp_path, rsa_key_path, server.url, sprite_upload_ff=False))
    body = {"error": {"code": "unauthorized", "message": "Authentication failed"}, "request_id": "a1"}
    server.script("POST", "/api/v1/detections", 401, body, times=3)
    w.enqueueDetections(FF_NAME, _payload(tmp_path), {})

    now = time.time()
    for i in range(3):
        w._runCycle(now)
        assert w._items[0]["next_attempt"] == pytest.approx(now + 3600)

        # Nothing is sent during the hold
        n_before = len(server.requests)
        w._runCycle(now + 1800)
        assert len(server.requests) == n_before

        now += 3601

    assert len([r for r in server.requests if _detections(r)]) == 3
    assert len(w._items) == 1 and w._items[0]["attempts"] == 3
    assert len(logs.messages(logging.ERROR)) == 1

    # Once the key is registered the backlog goes out
    w._runCycle(now)
    assert w._items == []


def testClockSkewIsAuthFailure(tmp_path, worker, server):
    server.script("POST", "/api/v1/detections", 401,
                  {"error": {"code": "stale_timestamp", "message": "Request timestamp outside"}})
    worker.enqueueDetections(FF_NAME, _payload(tmp_path), {})
    now = time.time()
    worker._runCycle(now)
    assert worker._items[0]["next_attempt"] == pytest.approx(now + 3600)


def testConnectionErrorRetried(tmp_path, rsa_key_path):

    # A closed port on loopback: connection refused, no network involved
    w = su.SpriteUploadWorker(_makeConfig(tmp_path, rsa_key_path, "http://127.0.0.1:9",
                                          sprite_upload_ff=False))
    w.enqueueDetections(FF_NAME, _payload(tmp_path), {})
    w._runCycle(time.time())
    assert len(w._items) == 1 and w._items[0]["attempts"] == 1


def testNoSecretsInLogs(tmp_path, rsa_key_path, server, logs):
    w = su.SpriteUploadWorker(_makeConfig(tmp_path, rsa_key_path, server.url))
    server.wanted.append(FF_NAME)
    server.script("POST", "/api/v1/detections", 500, b"oops")
    server.script("POST", "/api/v1/detections", 401, {"error": {"code": "unauthorized", "message": "no"}})
    w.enqueueDetections(FF_NAME, _payload(tmp_path), _products(tmp_path))

    now = time.time()
    for k in range(4):
        w._runCycle(now + k*4000)

    assert server.byPath("PUT")

    with open(rsa_key_path, "rb") as f:
        pem = f.read().decode("ascii")
    signatures = [r.headers["x-signature"] for r in server.requests if "x-signature" in r.headers]
    canonicals = [r.canonical.decode("ascii") for r in server.requests if r.canonical]
    assert signatures

    text = "\n".join(logging.Formatter().format(r) for r in logs.records)
    assert "PRIVATE KEY" not in text
    for sig in signatures:
        assert sig not in text
    for canonical in canonicals:
        assert canonical not in text

    # Any 16 byte window of the key body
    body = "".join(pem.splitlines()[1:-1])
    for i in range(0, len(body) - 16):
        assert body[i:i + 16] not in text

    # And no request body either
    assert '"station_id":"XX0001"' not in text


def testRedactHeaders():
    safe = su._redactHeaders({"X-Signature": "deadbeef", "X-Station-Id": "XX0001"})
    assert safe == {"X-Signature": "<redacted>", "X-Station-Id": "XX0001"}



# ###########################################################################
#                       Queue persistence
# ###########################################################################

def testQueueRoundTrip(tmp_path, rsa_key_path):
    config = _makeConfig(tmp_path, rsa_key_path, "http://127.0.0.1:9")
    w = su.SpriteUploadWorker(config)
    files = _products(tmp_path)
    path = _payload(tmp_path)
    w.enqueueDetections(FF_NAME, path, files)
    w._runCycle(time.time())

    w2 = su.SpriteUploadWorker(config)
    w2._ensureLoaded(time.time())
    assert [(r["kind"], r["name"], r["path"]) for r in w2._items] == [("detections", FF_NAME, path)]
    assert w2._items[0]["attempts"] == 1
    assert w2._known[FF_NAME]["files"] == files

    # No bodies in the queue file, only paths
    with open(w.queue_path) as f:
        text = f.read()
    assert "sent_at" not in text
    for line in text.splitlines():
        rec = json.loads(line)
        assert set(rec) == {"kind", "name", "path", "files", "attempts", "next_attempt", "queued_at",
                            "version"}


def testCorruptLineSkipped(tmp_path, rsa_key_path, logs):
    config = _makeConfig(tmp_path, rsa_key_path, "http://127.0.0.1:9")
    path = _payload(tmp_path)
    good = {"kind": "detections", "name": FF_NAME, "path": path, "files": {}, "attempts": 0,
            "next_attempt": 0, "queued_at": time.time(), "version": 1}
    with open(os.path.join(config.data_dir, su.QUEUE_FILE_NAME), "w") as f:
        f.write('{"kind": "detections", "name": "trunc')
        f.write("\n" + json.dumps(good) + "\n")
        f.write("[1, 2, 3]\n\n")

    w = su.SpriteUploadWorker(config)
    w._ensureLoaded(time.time())
    assert [r["name"] for r in w._items] == [FF_NAME]
    assert len([m for m in logs.messages(logging.WARNING) if "corrupt line" in m]) == 2


def testRestartPostsIdenticalBytes(tmp_path, rsa_key_path, server):
    config = _makeConfig(tmp_path, rsa_key_path, server.url, sprite_upload_ff=False)
    server.script("POST", "/api/v1/detections", 503, b"")
    w = su.SpriteUploadWorker(config)
    w.enqueueDetections(FF_NAME, _payload(tmp_path), {})
    w._runCycle(time.time())

    # A new process picks the item up from disk
    w2 = su.SpriteUploadWorker(config)
    w2._runCycle(time.time() + 3600)

    posts = [r for r in server.requests if _detections(r)]
    assert len(posts) == 2
    assert posts[0].body == posts[1].body
    assert posts[0].body_sha256 == posts[1].body_sha256
    assert w2._items == []

    # The same body a third time is a duplicate on the server, and that counts as done
    w2.enqueueDetections(FF_NAME, _payload(tmp_path), {})
    w2._runCycle(time.time() + 3600)
    assert w2._items == []
    assert len([r for r in server.requests if _detections(r)]) == 3
    assert len(server.seen_bodies) == 1


def testOldItemsDropped(tmp_path, rsa_key_path):
    config = _makeConfig(tmp_path, rsa_key_path, "http://127.0.0.1:9")
    now = time.time()
    old = {"kind": "detections", "name": FF_NAME, "path": _payload(tmp_path), "files": {}, "attempts": 30,
           "next_attempt": 0, "queued_at": now - 22*86400, "version": 1}
    fresh = dict(old, name=FF_NAME_2, path=_payload(tmp_path, FF_NAME_2), queued_at=now - 20*86400)
    known = {"kind": "known", "name": FF_NAME, "path": None, "files": {"ff": "/x"}, "attempts": 0,
             "next_attempt": 0, "queued_at": now - 22*86400, "version": 1}
    with open(os.path.join(config.data_dir, su.QUEUE_FILE_NAME), "w") as f:
        for rec in (old, fresh, known):
            f.write(json.dumps(rec) + "\n")

    w = su.SpriteUploadWorker(config)
    w._ensureLoaded(now)
    assert [r["name"] for r in w._items] == [FF_NAME_2]
    assert w._known == {}


def testVanishedFileDropped(tmp_path, rsa_key_path):
    config = _makeConfig(tmp_path, rsa_key_path, "http://127.0.0.1:9")
    w = su.SpriteUploadWorker(config)
    path = _payload(tmp_path)
    w.enqueueDetections(FF_NAME, path, {})
    os.remove(path)
    w._runCycle(time.time())
    assert w._items == []


def testQueueBounded(tmp_path, rsa_key_path):
    config = _makeConfig(tmp_path, rsa_key_path, "http://127.0.0.1:9")
    w = su.SpriteUploadWorker(config)
    path = _payload(tmp_path)

    now = time.time()
    w._ensureLoaded(now)
    for i in range(su.QUEUE_MAX_ITEMS + 20):
        w.enqueueDetections("FF_XX0001_20260828_031422_123_{:07d}.fits".format(i), path, {})
    w._ingest(now)

    assert len(w._items) == su.QUEUE_MAX_ITEMS

    # The same FF again replaces its item instead of adding one
    w.enqueueDetections("FF_XX0001_20260828_031422_123_{:07d}.fits".format(su.QUEUE_MAX_ITEMS + 19), path, {})
    w._ingest(now)
    assert len(w._items) == su.QUEUE_MAX_ITEMS



# ###########################################################################
#                       Thread, stop and reboot lock
# ###########################################################################

@pytest.mark.parametrize("started", [False, True])
def testStopHoldsAndReleasesLock(tmp_path, worker, monkeypatch, started):
    seen = {}

    def failingDrain(deadline):
        seen["lock"] = os.path.isfile(worker.lock_path)
        raise RuntimeError("drain exploded")

    monkeypatch.setattr(worker, "_drainOnStop", failingDrain)
    if started:
        worker.start()

    worker.stop(drain_timeout=5)

    assert seen["lock"] is True
    assert worker.lock_path.endswith(".reboot_lock.sprite")
    assert not os.path.exists(worker.lock_path)


def testStopDrainsQueue(tmp_path, worker, server):
    server.wanted.append(FF_NAME)
    worker.enqueueDetections(FF_NAME, _payload(tmp_path), _products(tmp_path))
    worker.stop(drain_timeout=20)

    assert len([r for r in server.requests if _detections(r)]) == 1
    assert len(server.byPath("PUT")) == 3
    assert not os.path.exists(worker.lock_path)


def testStaleLockRemovedAtStart(tmp_path, worker):
    with open(worker.lock_path, "w") as f:
        f.write("1\n")

    # A fresh lock belongs to someone who is still working
    worker.start()
    assert os.path.exists(worker.lock_path)
    worker.stop(drain_timeout=1)

    with open(worker.lock_path, "w") as f:
        f.write("1\n")
    old = time.time() - 7*3600
    os.utime(worker.lock_path, (old, old))
    worker._removeStaleLock()
    assert not os.path.exists(worker.lock_path)


def testEnqueueNeverBlocks(tmp_path, worker, server):
    release = threading.Event()
    server.script("GET", "/api/v1/stations", 200, {"requests": []}, hang=release)
    server.script("POST", "/api/v1/detections", 200, {"success": True}, hang=release)
    worker.start()

    try:
        # Wait until the worker is stuck inside a request
        deadline = time.time() + 5
        while not server.requests and time.time() < deadline:
            time.sleep(0.01)
        assert server.requests

        t0 = time.time()
        for i in range(50):
            worker.enqueueDetections(FF_NAME, _payload(tmp_path), {})
        assert time.time() - t0 < 0.5

    finally:
        release.set()
        worker.stop(drain_timeout=10)

    assert not worker._thread.is_alive()
