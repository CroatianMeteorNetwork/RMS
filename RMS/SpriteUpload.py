# RPi Meteor Station
# Copyright (C) 2025  Dino Grzinic
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" Upload client for the spritemap server (spritemap-data-pipeline).

    The station signs every request with the SSH private key it already uses for SFTP uploads. The canonical
    string and the signature algorithms below MUST match spritemap/signing.py on the server byte for byte:

        SPRITEMAP-V1\\n<METHOD>\\n<PATH>\\n<STATION_ID>\\n<TIMESTAMP>\\n<sha256 hex of the body>

    If the layout ever changes, bump PROTOCOL_VERSION on both sides and regenerate
    Tests/data/spritemap_canonical_vectors.json from the server's buildCanonicalString().

    The worker runs in a daemon thread. It posts detection payloads as they are queued, polls the server
    for the FF files it would like (the server only asks once a second station confirmed the event) and
    uploads those. Pending work is kept in a JSON Lines file in the data directory, so it survives restarts.
"""

from __future__ import print_function, division, absolute_import

import binascii
import hashlib
import json
import os
import random
import re
import ssl
import sys
import threading
import time

from RMS.Logger import getLogger

# Python 2/3 compatibility for urllib and the thread-safe queue
if sys.version_info[0] < 3:

    import urllib2 as urllib_request
    import urllib2 as urllib_error
    from urlparse import urlparse
    import Queue as queue_module

else:

    import urllib.request as urllib_request
    import urllib.error as urllib_error
    from urllib.parse import urlparse
    import queue as queue_module

# Text type of decoded JSON on both Pythons
try:
    STRING_TYPES = (str, unicode)
except NameError:
    STRING_TYPES = (str,)

log = getLogger("rmslogger")


# Bumped whenever the canonical string layout changes (must equal spritemap/signing.py)
PROTOCOL_VERSION = "SPRITEMAP-V1"

# Authentication header names
HEADER_STATION = "X-Station-Id"
HEADER_TIMESTAMP = "X-Timestamp"
HEADER_SIGNATURE = "X-Signature"

# Smallest RSA key the server accepts
MIN_RSA_BITS = 2048

# Endpoints
DETECTIONS_PATH = "/api/v1/detections"
FILES_PATH_PREFIX = "/api/v1/files/"
REQUESTS_PATH_TEMPLATE = "/api/v1/stations/{:s}/requests"

# Server side limits, checked before sending so nothing is sent that the server must refuse
MAX_DETECTIONS_BYTES = 1024*1024
MAX_FF_BYTES = 32*1024*1024
MAX_IMAGE_BYTES = 2*1024*1024

# Timeouts in seconds; file uploads get a long one because stations may be on slow links
FILE_TIMEOUT = 300.0
DEFAULT_TIMEOUT = 30.0

# How much of a response body is ever read
ERROR_BODY_LIMIT = 8*1024
RESPONSE_BODY_LIMIT = 1024*1024

# Polling of the file requests endpoint
POLL_INTERVAL = 30*60.0
POLL_JITTER = 5*60.0

# Queue bounds. The age bound is longer than the server's 14 day request expiry
QUEUE_FILE_NAME = "SPRITE_UPLOADS.json"
QUEUE_VERSION = 1
QUEUE_MAX_ITEMS = 500
QUEUE_MAX_KNOWN = 5000
QUEUE_MAX_AGE = 21*24*3600.0

# Retry backoff
BACKOFF_BASE = 60.0
BACKOFF_CAP = 1800.0
AUTH_BACKOFF = 3600.0

# A .sprite reboot lock older than this was left behind by a power cut
STALE_LOCK_AGE = 6*3600.0

# Longest the idle worker sleeps between checks
MAX_IDLE_WAIT = 60.0

# File names the server accepts (copied from spritemap/validation.py and spritemap/storage.py)
FF_NAME_RE = re.compile(r"^FF_([A-Z0-9]{3,8})_(\d{8})_(\d{6})_(\d{3,6})_(\d{7})\.fits$")
FF_FILE_RE = re.compile(r"^FF_([A-Z0-9]{3,8})_(\d{8})_(\d{6})_(\d{3,6})_(\d{7})\.fits(\.bz2)?$")
IMAGE_FILE_RE = re.compile(
    r"^(FF_[A-Z0-9]{3,8}_\d{8}_\d{6}_\d{3,6}_\d{7})_(marked|unmarked)\.(jpg|jpeg|png)$")

# Error codes that mean "not authorised yet or clock wrong": keep the item and wait
AUTH_CODES = ("unauthorized", "stale_timestamp", "bad_timestamp")

# Statuses that will never succeed on a retry
PERMANENT_STATUSES = (400, 403, 404, 405, 409, 413)

# Headers whose values must never reach a log
SECRET_HEADERS = (HEADER_SIGNATURE.lower(), "authorization", "cookie")



# ###########################################################################
#                       Canonical string and signatures
# ###########################################################################

def hashBody(body):
    """ Compute the hex SHA-256 digest of a request body.

    Arguments:
        body: [bytes] Raw body, may be empty.

    Return:
        [str] 64 lower case hex characters.
    """

    # A request without a body still gets a digest, that of the empty string
    return hashlib.sha256(body).hexdigest()


def hashFile(path, chunk_size=1024*1024):
    """ Compute the hex SHA-256 digest of a file without loading it into memory.

    Arguments:
        path: [str] Path to the file.

    Keyword arguments:
        chunk_size: [int] Read size in bytes. 1 MiB by default.

    Return:
        [str] 64 lower case hex characters.
    """

    digest = hashlib.sha256()

    # FF files are several megabytes, so hash them in pieces
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def buildCanonicalString(method, path, station_id, timestamp, body_sha256):
    """ Assemble the bytes that get signed. Must match spritemap/signing.py byte for byte.

    Arguments:
        method: [str] HTTP method, e.g. "POST".
        path: [str] Request path without the query string, e.g. "/api/v1/detections".
        station_id: [str] Station code; upper-cased here so both sides agree.
        timestamp: [int] Unix time in seconds.
        body_sha256: [str] Hex digest of the body from hashBody().

    Return:
        [bytes] Canonical string encoded as ASCII.
    """

    # Fixed order and fixed case, joined by newlines with no trailing newline
    lines = [
        PROTOCOL_VERSION,
        method.upper(),
        path,
        station_id.upper(),
        str(int(timestamp)),
        body_sha256.lower(),
    ]

    return "\n".join(lines).encode("ascii")


def _ecdsaHash(key):
    """ Pick the hash that goes with an elliptic curve key, following the SSH convention.

    Arguments:
        key: [EllipticCurvePrivateKey or EllipticCurvePublicKey]

    Return:
        [HashAlgorithm] SHA-256 for P-256, SHA-384 for P-384, SHA-512 for P-521.
    """

    from cryptography.hazmat.primitives import hashes

    curve_name = key.curve.name

    # SSH pairs every NIST curve with the hash of matching strength
    if curve_name == "secp384r1":
        return hashes.SHA384()

    if curve_name == "secp521r1":
        return hashes.SHA512()

    return hashes.SHA256()


def signCanonical(private_key, canonical):
    """ Sign the canonical string with the station's private key.

    Arguments:
        private_key: [object] RSA, Ed25519 or EC private key from the cryptography library.
        canonical: [bytes] Output of buildCanonicalString().

    Return:
        [bytes] Raw signature (DER encoded for ECDSA).
    """

    # Imported here so that RMS works on stations without the cryptography package
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa

    # RSA with PKCS#1 v1.5 padding over SHA-256, as the rsa-sha2-256 SSH signature
    if isinstance(private_key, rsa.RSAPrivateKey):
        return private_key.sign(canonical, padding.PKCS1v15(), hashes.SHA256())

    # Ed25519 hashes the message itself
    if isinstance(private_key, ed25519.Ed25519PrivateKey):
        return private_key.sign(canonical)

    # ECDSA over the hash that matches the curve, DER encoded
    if isinstance(private_key, ec.EllipticCurvePrivateKey):
        return private_key.sign(canonical, ec.ECDSA(_ecdsaHash(private_key)))

    raise TypeError("Unsupported private key type: {:s}".format(type(private_key).__name__))


def loadPrivateKey(path):
    """ Load and vet the station's private key. Never raises.

    Arguments:
        path: [str] Path to the key file, PEM or OpenSSH format, without a passphrase.

    Return:
        (key, reason):
            key: [object] Private key, or None if it cannot be used.
            reason: [str] Why the key cannot be used, None if it can.
    """

    # Without the cryptography package nothing can be signed
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa

    except Exception:
        return None, "the Python package 'cryptography' is not installed"

    if not path:
        return None, "no private key configured (rsa_private_key)"

    path = os.path.expanduser(path)

    # Both the read and the parse are inside the try, so no key problem can stop the process
    try:
        with open(path, "rb") as f:
            data = f.read()

    except Exception as e:
        return None, "private key {:s} cannot be read ({:s})".format(path, type(e).__name__)

    key = None
    try:

        # The classic PEM container first; a TypeError means it is encrypted
        try:
            key = serialization.load_pem_private_key(data, password=None)

        except TypeError:
            return None, "private key {:s} is passphrase-protected, which is not supported".format(path)

        except ValueError:

            # Not PEM, so try the OpenSSH container that ssh-keygen writes by default
            try:
                key = serialization.load_ssh_private_key(data, password=None)

            except (TypeError, ValueError) as e:
                if "password" in str(e).lower() or "encrypted" in str(e).lower():
                    return None, "private key {:s} is passphrase-protected, which is not supported".format(
                        path)
                raise

    except Exception as e:
        return None, "private key {:s} cannot be parsed ({:s})".format(path, type(e).__name__)

    # RSA must be long enough for the server to accept it
    if isinstance(key, rsa.RSAPrivateKey):
        if key.key_size < MIN_RSA_BITS:
            return None, "RSA private key {:s} has {:d} bits, at least {:d} are required".format(
                path, key.key_size, MIN_RSA_BITS)
        return key, None

    if isinstance(key, ed25519.Ed25519PrivateKey):
        return key, None

    # Only the NIST curves that SSH uses have a defined hash
    if isinstance(key, ec.EllipticCurvePrivateKey):
        if key.curve.name in ("secp256r1", "secp384r1", "secp521r1"):
            return key, None
        return None, "EC private key {:s} uses the unsupported curve {:s}".format(path, key.curve.name)

    return None, "private key {:s} has the unsupported type {:s}".format(path, type(key).__name__)



# ###########################################################################
#                       URL and response helpers
# ###########################################################################

def _isLoopbackLiteral(host):
    """ Tell whether a host name is a loopback IP literal (not a name that might resolve elsewhere).

    Arguments:
        host: [str] Host part of a URL.

    Return:
        [bool]
    """

    if not host:
        return False

    try:
        import ipaddress
        return ipaddress.ip_address(u"{}".format(host)).is_loopback

    except Exception:
        return host == "127.0.0.1"


def checkBaseUrl(url, allow_insecure=False):
    """ Validate the server base URL.

    Arguments:
        url: [str] Base URL from the config, e.g. https://sprites.example.org

    Keyword arguments:
        allow_insecure: [bool] Allow plain http for a loopback host, for tests. False by default.

    Return:
        (base, reason):
            base: [str] Normalised base URL without a trailing slash, or None if unusable.
            reason: [str] Why the URL is unusable, None if it is fine.
    """

    try:
        parsed = urlparse(url.strip())
    except Exception:
        return None, "sprite_upload_url cannot be parsed"

    scheme = (parsed.scheme or "").lower()

    # The signature is bound to the path, so the server must live at the root of the host
    if parsed.path not in ("", "/") or parsed.params or parsed.query or parsed.fragment:
        return None, "sprite_upload_url must not contain a path, query or fragment"

    if not parsed.hostname or "@" in parsed.netloc:
        return None, "sprite_upload_url must be a plain https://host[:port] URL"

    # Plain http only against this machine, and only when explicitly allowed
    if scheme == "http":
        if not (allow_insecure and _isLoopbackLiteral(parsed.hostname)):
            return None, "sprite_upload_url must use https"

    elif scheme != "https":
        return None, "sprite_upload_url must use https"

    return "{:s}://{:s}".format(scheme, parsed.netloc), None


def _redactHeaders(headers):
    """ Copy headers for logging with every secret value replaced. The only way headers reach a log.

    Arguments:
        headers: [dict] Header names and values.

    Return:
        [dict] Safe copy.
    """

    safe = {}
    for name, value in dict(headers or {}).items():
        if str(name).lower() in SECRET_HEADERS:
            safe[name] = "<redacted>"
        else:
            safe[name] = value

    return safe


def _cleanText(text, limit=200):
    """ Make server supplied text safe to put in a log line (printable ASCII, bounded).

    Arguments:
        text: [object] Anything.

    Keyword arguments:
        limit: [int] Maximum length. 200 by default.

    Return:
        [str]
    """

    if text is None:
        return ""

    try:
        text = "{}".format(text)
    except Exception:
        return ""

    # A newline in a server message must not be able to forge log lines
    return "".join(c if 32 <= ord(c) < 127 else "?" for c in text[:limit])


def _readBounded(fp, limit):
    """ Read at most limit bytes from a response or error body. Never raises.

    Arguments:
        fp: [file-like] Response object.
        limit: [int] Maximum number of bytes.

    Return:
        [bytes] What could be read, possibly empty.
    """

    try:
        data = fp.read(limit)
        if not isinstance(data, bytes):
            return b""
        return data[:limit]

    except Exception:
        return b""


def parseErrorBody(raw, headers=None):
    """ Extract code, message and request id from an error response. Never raises.

    Arguments:
        raw: [bytes] Response body (possibly HTML from a proxy, empty or invalid UTF-8).

    Keyword arguments:
        headers: [object] Response headers with a .get() method, for X-Request-Id.

    Return:
        (code, message, request_id): [str or None] each.
    """

    code = None
    message = None
    request_id = None

    # The server's own error shape: {"error": {"code": ..., "message": ...}, "request_id": ...}
    try:
        data = json.loads(raw.decode("utf-8"))
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict):
                code = err.get("code")
                message = err.get("message")
            request_id = data.get("request_id")

    except Exception:
        pass

    # A proxy page has no JSON, but the header may still be there
    if not request_id and headers is not None:
        try:
            request_id = headers.get("X-Request-Id")
        except Exception:
            request_id = None

    code = _cleanText(code, 64) or None
    message = _cleanText(message) or None
    request_id = _cleanText(request_id, 64) or None

    return code, message, request_id


def backoffDelay(attempts, rng=random):
    """ Retry delay after a transient failure.

    Arguments:
        attempts: [int] Number of failed attempts so far, at least 1.

    Keyword arguments:
        rng: [object] Source of uniform(), for tests. The random module by default.

    Return:
        [float] Delay in seconds: min(60*2^(attempts - 1), 1800), jittered by +-20 %.
    """

    exponent = max(0, min(int(attempts) - 1, 20))
    return min(BACKOFF_BASE*(2**exponent), BACKOFF_CAP)*rng.uniform(0.8, 1.2)


class _NoRedirectHandler(urllib_request.HTTPRedirectHandler):
    """ Refuse every redirect: following one would re-send a signature bound to another method and path. """

    def redirect_request(self, req, fp, code, msg, headers, newurl):

        # Returning None makes urllib raise HTTPError with the 3xx status
        return None


class _Result(object):
    """ Outcome of one HTTP exchange. """

    __slots__ = ("status", "data", "code", "message", "request_id", "error")

    def __init__(self, status=None, data=None, code=None, message=None, request_id=None, error=None):
        self.status = status
        self.data = data
        self.code = code
        self.message = message
        self.request_id = request_id
        self.error = error


def classifyResult(result):
    """ Decide what to do after a request.

    Arguments:
        result: [_Result]

    Return:
        [str] "ok", "permanent" (drop, never retry), "auth" (keep, wait an hour) or "retry".
    """

    # No status at all: connection error, timeout, TLS failure
    if result.status is None:
        return "retry"

    # Success needs a JSON object; anything else (a captive portal page) is retried
    if 200 <= result.status < 300:
        if isinstance(result.data, dict):
            return "ok"
        return "retry"

    # The key is not registered yet, or the clock is wrong: both fix themselves or need the operator
    if result.status == 401 or result.code in AUTH_CODES:
        return "auth"

    # A redirect is never followed and never retried
    if 300 <= result.status < 400:
        return "permanent"

    if result.status in PERMANENT_STATUSES:
        return "permanent"

    return "retry"



# ###########################################################################
#                       Upload worker
# ###########################################################################

class SpriteUploadWorker(object):
    """ Background uploader for sprite detections and the FF files the server asks for.

    Arguments:
        config: [Config] RMS configuration.
    """

    def __init__(self, config):

        self.config = config
        self.enabled = False
        self.disabled_reason = None

        self.station_id = str(getattr(config, "stationID", "") or "")
        self.timeout = float(getattr(config, "sprite_upload_timeout", DEFAULT_TIMEOUT) or DEFAULT_TIMEOUT)
        self.upload_ff = bool(getattr(config, "sprite_upload_ff", True))

        data_dir = os.path.expanduser(str(getattr(config, "data_dir", "") or "."))
        self.queue_path = os.path.join(data_dir, QUEUE_FILE_NAME)
        self.lock_path = os.path.join(data_dir, str(getattr(config, "reboot_lock_file", ".reboot_lock"))
                                      + ".sprite")

        self.base_url = None
        self._private_key = None
        self._opener = None

        # Everything below is owned by the worker thread once it runs
        self._inbox = queue_module.Queue()
        self._wake = threading.Event()
        self._stop_event = threading.Event()
        self._thread = None
        self._drain_deadline = None

        self._items = []
        self._known = {}
        self._loaded = False
        self._dirty = False
        self._next_poll = 0.0
        self._auth_hold_until = 0.0
        self._auth_logged = set()
        self._warned_requests = set()

        # Any failure while setting up disables the worker; the station keeps capturing either way
        try:
            self._configure()
        except Exception as e:
            self._disable("unexpected setup error ({:s})".format(repr(e)))


    def _disable(self, reason, level="warning"):
        """ Mark the worker unusable and say why, once.

        Arguments:
            reason: [str] Human readable reason.

        Keyword arguments:
            level: [str] Log level name. "warning" by default.
        """

        self.enabled = False
        self.disabled_reason = reason
        self._private_key = None
        getattr(log, level)("Sprite uploads disabled: {:s}".format(reason))


    def _configure(self):
        """ Validate the configuration, load the key once and build the HTTP opener. """

        url = str(getattr(self.config, "sprite_upload_url", "") or "").strip()

        # Empty URL is the normal "not configured" state
        if not url:
            self._disable("sprite_upload_url is empty", level="info")
            return

        # FF names on disk carry the station id as configured; the server only knows upper case
        if not self.station_id:
            self._disable("stationID is empty")
            return

        if self.station_id != self.station_id.upper():
            self._disable("stationID '{:s}' must be upper case; set stationID: {:s} in the config file "
                          "so that FF names match what the server expects".format(
                              self.station_id, self.station_id.upper()))
            return

        allow_insecure = bool(getattr(self.config, "sprite_upload_allow_insecure", False))
        base, reason = checkBaseUrl(url, allow_insecure=allow_insecure)
        if base is None:
            self._disable(reason)
            return

        key, reason = loadPrivateKey(getattr(self.config, "rsa_private_key", None))
        if key is None:
            self._disable(reason)
            return

        # One opener for the life of the worker: verified TLS and no redirects
        context = ssl.create_default_context()
        self._opener = urllib_request.build_opener(urllib_request.HTTPSHandler(context=context),
                                                   _NoRedirectHandler())

        self.base_url = base
        self._private_key = key
        self.enabled = True
        log.info("Sprite uploads enabled to {:s} as station {:s}".format(base, self.station_id))


    def isEnabled(self):
        """ Tell whether the worker will upload anything.

        Return:
            [bool]
        """

        return self.enabled


    # -----------------------------------------------------------------------
    # Public API used by the detector
    # -----------------------------------------------------------------------

    def start(self):
        """ Start the daemon thread. Removes a stale .sprite reboot lock first. Never raises. """

        try:
            self._removeStaleLock()

            if not self.enabled or (self._thread is not None and self._thread.is_alive()):
                return

            self._thread = threading.Thread(target=self._run, name="SpriteUploadWorker")
            self._thread.daemon = True
            self._thread.start()

        except Exception as e:
            log.error("Sprite upload worker could not start: {:s}".format(repr(e)))


    def enqueueDetections(self, ff_name, payload_path, files=None):
        """ Queue a detections payload. Returns immediately; the network is only touched by the worker.

        Arguments:
            ff_name: [str] FF file name the payload describes, .fits included.
            payload_path: [str] JSON file whose bytes are posted verbatim.

        Keyword arguments:
            files: [dict] Optional keys "ff", "marked", "unmarked" with absolute paths of files the server
                may request later. None by default.
        """

        try:
            if not self.enabled:
                return

            # Keep only the keys the server can ask for
            clean_files = {}
            for kind in ("ff", "marked", "unmarked"):
                if files and files.get(kind):
                    clean_files[kind] = str(files[kind])

            self._inbox.put((str(ff_name), str(payload_path), clean_files, time.time()))
            self._wake.set()

        except Exception as e:
            log.error("Sprite upload: could not queue {}: {:s}".format(ff_name, repr(e)))


    def stop(self, drain_timeout=120):
        """ Drain the queue for a bounded time while holding the .sprite reboot lock. Never raises.

        Keyword arguments:
            drain_timeout: [float] Longest time to spend uploading, in seconds. 120 by default.
        """

        lock_created = False
        try:
            if not self.enabled:
                return

            # Hold off a reboot while the last uploads go out; not the shared lock, which others delete
            try:
                with open(self.lock_path, "w") as f:
                    f.write("{:d}\n".format(os.getpid()))
                lock_created = True
            except Exception as e:
                log.warning("Sprite upload: could not create reboot lock {:s}: {:s}".format(
                    self.lock_path, repr(e)))

            deadline = time.time() + max(0.0, float(drain_timeout))

            # The worker thread does the drain itself, so its state is never touched from two threads
            if self._thread is not None and self._thread.is_alive():
                self._drain_deadline = deadline
                self._stop_event.set()
                self._wake.set()
                self._thread.join(max(0.0, float(drain_timeout)) + 5.0)

                if self._thread.is_alive():
                    log.warning("Sprite upload: drain did not finish in {:.0f} s, pending items stay "
                                "queued for the next run".format(float(drain_timeout)))

            else:
                self._stop_event.set()
                self._drainSafely(deadline)

        except Exception as e:
            log.error("Sprite upload: error while stopping: {:s}".format(repr(e)))

        finally:

            # Always release the lock, even if the drain failed
            if lock_created:
                try:
                    os.remove(self.lock_path)
                except Exception as e:
                    log.warning("Sprite upload: could not remove reboot lock {:s}: {:s}".format(
                        self.lock_path, repr(e)))


    # -----------------------------------------------------------------------
    # Thread body
    # -----------------------------------------------------------------------

    def _run(self):
        """ Thread body. No exception escapes it. """

        try:
            while not self._stop_event.is_set():

                # One failed cycle must not end the thread; wait a bit so a persistent error does not spin
                failed = False
                try:
                    self._runCycle(time.time())
                except Exception as e:
                    log.error("Sprite upload: worker cycle failed: {:s}".format(repr(e)))
                    failed = True

                timeout = MAX_IDLE_WAIT if failed else self._nextWake(time.time())
                self._wake.wait(timeout)
                self._wake.clear()

            # Stop was requested; the caller holds the reboot lock while this runs
            if self._drain_deadline is not None:
                self._drainSafely(self._drain_deadline)

        except BaseException as e:
            try:
                log.error("Sprite upload: worker thread ended with an error: {:s}".format(repr(e)))
            except BaseException:
                pass


    def _drainSafely(self, deadline):
        """ Run the stop drain, logging instead of raising.

        Arguments:
            deadline: [float] Unix time after which no new request is started.
        """

        try:
            self._drainOnStop(deadline)
        except Exception as e:
            log.error("Sprite upload: drain failed: {:s}".format(repr(e)))

        # Whatever happened, keep what is left for the next run
        try:
            self._saveQueue()
        except Exception as e:
            log.error("Sprite upload: could not save queue: {:s}".format(repr(e)))


    def _drainOnStop(self, deadline):
        """ Poll the requests once and send whatever is due until the deadline.

        Arguments:
            deadline: [float] Unix time after which no new request is started.
        """

        now = time.time()
        self._ensureLoaded(now)
        self._ingest(now)

        # One last look at what the server wants, so a reboot does not delay it by a night
        if now < deadline:
            self._pollRequests(now)

        # Keep going while requests succeed; a failure puts items into backoff and ends the loop
        while time.time() < deadline:
            if self._processDue(time.time(), deadline=deadline) == 0:
                break

        self._saveQueue()


    def _runCycle(self, now):
        """ One pass of the worker: take new items, poll if it is time, send what is due, save.

        Arguments:
            now: [float] Unix time used for scheduling.
        """

        self._ensureLoaded(now)
        self._ingest(now)

        if now >= self._next_poll:
            self._pollRequests(now)

        self._processDue(now)
        self._saveQueue()


    def _nextWake(self, now):
        """ How long the idle worker may sleep.

        Arguments:
            now: [float] Unix time.

        Return:
            [float] Seconds, between 0.1 and MAX_IDLE_WAIT.
        """

        wake = now + MAX_IDLE_WAIT

        if self.upload_ff:
            wake = min(wake, self._next_poll)

        for rec in self._items:
            wake = min(wake, max(rec["next_attempt"], self._auth_hold_until))

        return min(MAX_IDLE_WAIT, max(0.1, wake - now))


    # -----------------------------------------------------------------------
    # Queue
    # -----------------------------------------------------------------------

    def _newRecord(self, kind, name, path, files, now):
        """ Build a queue record.

        Arguments:
            kind: [str] "detections", "file" or "known".
            name: [str] FF name for detections and known, upload name for file.
            path: [str] Local file path (None for known).
            files: [dict] Requestable files (detections and known), None for file.
            now: [float] Unix time.

        Return:
            [dict]
        """

        return {"kind": kind, "name": name, "path": path, "files": files, "attempts": 0,
                "next_attempt": now, "queued_at": now, "version": QUEUE_VERSION}


    def _ensureLoaded(self, now):
        """ Load the queue from disk the first time the worker needs it.

        Arguments:
            now: [float] Unix time.
        """

        if not self._loaded:
            self._loaded = True
            self._loadQueue(now)


    def _loadQueue(self, now):
        """ Read the JSON Lines queue, skipping corrupt lines and dropping stale items.

        Arguments:
            now: [float] Unix time.
        """

        if not os.path.isfile(self.queue_path):
            return

        try:
            with open(self.queue_path, "rb") as f:
                lines = f.read().splitlines()

        except Exception as e:
            log.warning("Sprite upload: could not read queue {:s}: {:s}".format(self.queue_path, repr(e)))
            return

        n_loaded = 0
        for line_no, line in enumerate(lines, 1):

            if not line.strip():
                continue

            # A torn write or a manual edit must not lose the rest of the queue
            try:
                rec = json.loads(line.decode("utf-8"))
                rec = self._validateRecord(rec)

            except Exception:
                rec = None

            if rec is None:
                log.warning("Sprite upload: skipping corrupt line {:d} in {:s}".format(
                    line_no, self.queue_path))
                continue

            self._addRecord(rec, replace=True)
            n_loaded += 1

        self._enforceBounds(now)
        self._dirty = True

        if n_loaded:
            log.info("Sprite upload: loaded {:d} queued item(s) from {:s}".format(
                len(self._items), self.queue_path))


    def _validateRecord(self, rec):
        """ Check the shape of a record read from disk.

        Arguments:
            rec: [object] Decoded JSON line.

        Return:
            [dict] Normalised record, or None if unusable.
        """

        if not isinstance(rec, dict) or rec.get("kind") not in ("detections", "file", "known"):
            return None

        name = rec.get("name")
        if not isinstance(name, STRING_TYPES) or not name:
            return None

        path = rec.get("path")
        if rec["kind"] != "known" and (not isinstance(path, STRING_TYPES) or not path):
            return None

        files = rec.get("files")
        if files is not None and not isinstance(files, dict):
            return None

        if files is not None:
            files = dict((str(k), str(v)) for k, v in files.items())

        return {"kind": rec["kind"], "name": name, "path": path, "files": files,
                "attempts": int(rec.get("attempts", 0)),
                "next_attempt": float(rec.get("next_attempt", 0)),
                "queued_at": float(rec.get("queued_at", 0)),
                "version": int(rec.get("version", QUEUE_VERSION))}


    def _addRecord(self, rec, replace=False):
        """ Put a record in memory, deduplicated on (kind, name).

        Arguments:
            rec: [dict] Record.

        Keyword arguments:
            replace: [bool] Replace an existing record with the same key. False by default.

        Return:
            [bool] True if the record was added or replaced.
        """

        if rec["kind"] == "known":
            if rec["name"] in self._known and not replace:
                return False
            self._known[rec["name"]] = rec
            return True

        for i, old in enumerate(self._items):
            if old["kind"] == rec["kind"] and old["name"] == rec["name"]:
                if not replace:
                    return False
                self._items[i] = rec
                return True

        self._items.append(rec)
        return True


    def _enforceBounds(self, now):
        """ Drop items that are too old, whose file is gone, or that exceed the count bound.

        Arguments:
            now: [float] Unix time.
        """

        kept = []
        for rec in self._items:

            if now - rec["queued_at"] > QUEUE_MAX_AGE:
                log.warning("Sprite upload: dropping {:s} {:s}, queued more than 21 days ago".format(
                    rec["kind"], rec["name"]))
                self._dirty = True
                continue

            if not os.path.isfile(rec["path"]):
                log.warning("Sprite upload: dropping {:s} {:s}, file {:s} no longer exists".format(
                    rec["kind"], rec["name"], rec["path"]))
                self._dirty = True
                continue

            kept.append(rec)

        # Oldest first out when the queue is full
        if len(kept) > QUEUE_MAX_ITEMS:
            kept.sort(key=lambda r: r["queued_at"])
            n_drop = len(kept) - QUEUE_MAX_ITEMS
            log.warning("Sprite upload: queue full, dropping the {:d} oldest item(s)".format(n_drop))
            kept = kept[n_drop:]
            self._dirty = True

        self._items = kept

        # The list of requestable files ages out the same way, a bit after the server stops asking
        for name in list(self._known):
            if now - self._known[name]["queued_at"] > QUEUE_MAX_AGE:
                del self._known[name]
                self._dirty = True

        if len(self._known) > QUEUE_MAX_KNOWN:
            ordered = sorted(self._known.values(), key=lambda r: r["queued_at"])
            for rec in ordered[:len(self._known) - QUEUE_MAX_KNOWN]:
                del self._known[rec["name"]]
            self._dirty = True


    def _saveQueue(self, force=False):
        """ Rewrite the queue file atomically, only if something changed.

        Keyword arguments:
            force: [bool] Write even if nothing changed. False by default.
        """

        if not (self._dirty or force) or not self._loaded:
            return

        tmp_path = self.queue_path + ".tmp"

        try:
            with open(tmp_path, "w") as f:
                for rec in list(self._items) + list(self._known.values()):
                    f.write(json.dumps(rec, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())

            # tmp + replace, so a power cut leaves either the old or the new file
            if hasattr(os, "replace"):
                os.replace(tmp_path, self.queue_path)
            else:
                os.rename(tmp_path, self.queue_path)

            self._dirty = False

        except Exception as e:
            log.error("Sprite upload: could not save queue {:s}: {:s}".format(self.queue_path, repr(e)))


    def _ingest(self, now):
        """ Move newly enqueued detections from the thread-safe inbox into the queue.

        Arguments:
            now: [float] Unix time.
        """

        while True:
            try:
                ff_name, payload_path, files, _ = self._inbox.get_nowait()
            except queue_module.Empty:
                break

            rec = self._newRecord("detections", ff_name, payload_path, files, now)
            self._addRecord(rec, replace=True)

            # Remember which files could be asked for, independently of the detections item
            if files and self.upload_ff:
                self._addRecord(self._newRecord("known", ff_name, None, files, now), replace=True)

            self._dirty = True

        self._enforceBounds(now)


    # -----------------------------------------------------------------------
    # HTTP
    # -----------------------------------------------------------------------

    def _signedHeaders(self, method, path, body_sha256):
        """ Build the authentication headers for one request.

        Arguments:
            method: [str] HTTP method.
            path: [str] Request path.
            body_sha256: [str] Hex digest of the body.

        Return:
            [dict] Headers.
        """

        # The wall clock at sending time, since the server checks it against its own
        timestamp = int(time.time())
        canonical = buildCanonicalString(method, path, self.station_id, timestamp, body_sha256)
        signature = signCanonical(self._private_key, canonical)

        log.debug("Sprite upload: signing {:s} {:s} station={:s} ts={:d} body_sha256={:s}...".format(
            method, path, self.station_id.upper(), timestamp, body_sha256[:12]))

        return {
            HEADER_STATION: self.station_id.upper(),
            HEADER_TIMESTAMP: str(timestamp),
            HEADER_SIGNATURE: binascii.hexlify(signature).decode("ascii"),
        }


    def _request(self, method, path, body=None, file_path=None, content_type=None, timeout=None):
        """ Send one signed request and collect the outcome. Never raises.

        Arguments:
            method: [str] "GET", "POST" or "PUT".
            path: [str] Request path, starting with /.

        Keyword arguments:
            body: [bytes] Body in memory. None for no body.
            file_path: [str] File streamed as the body instead of body. None by default.
            content_type: [str] Content-Type header. None by default.
            timeout: [float] Timeout in seconds. The configured timeout by default.

        Return:
            [_Result]
        """

        if timeout is None:
            timeout = self.timeout

        fh = None
        try:

            # Streamed file bodies are hashed from disk first, then sent from an open handle
            if file_path is not None:
                size = os.path.getsize(file_path)
                body_sha256 = hashFile(file_path)
            else:
                size = len(body) if body is not None else 0
                body_sha256 = hashBody(body if body is not None else b"")

            headers = self._signedHeaders(method, path, body_sha256)

            # An explicit length keeps urllib from falling back to chunked, which the server refuses
            if file_path is not None or body is not None:
                headers["Content-Length"] = str(size)
            if content_type:
                headers["Content-Type"] = content_type

            if file_path is not None:
                fh = open(file_path, "rb")
                data = fh
            else:
                data = body

            log.debug("Sprite upload: {:s} {:s} headers={}".format(method, path, _redactHeaders(headers)))

            req = urllib_request.Request(self.base_url + path, data=data, headers=headers)
            req.get_method = lambda: method

            resp = self._opener.open(req, timeout=timeout)
            try:
                raw = _readBounded(resp, RESPONSE_BODY_LIMIT)
                status = resp.getcode()
                resp_headers = resp.info()
            finally:
                resp.close()

            # A success without a JSON object is treated as a failure by classifyResult()
            try:
                data_obj = json.loads(raw.decode("utf-8"))
            except Exception:
                data_obj = None

            request_id = None
            try:
                request_id = _cleanText(resp_headers.get("X-Request-Id"), 64) or None
            except Exception:
                pass

            return _Result(status=status, data=data_obj, request_id=request_id)

        except urllib_error.HTTPError as e:

            # Error bodies can be anything, e.g. an nginx HTML page; read little and never raise here
            try:
                raw = _readBounded(e, ERROR_BODY_LIMIT)
                hdrs = getattr(e, "headers", None)
                code, message, request_id = parseErrorBody(raw, hdrs)
                return _Result(status=e.code, code=code, message=message, request_id=request_id)

            except Exception:
                return _Result(status=getattr(e, "code", None))

            finally:
                try:
                    e.close()
                except Exception:
                    pass

        except Exception as e:

            # Connection refused, DNS, timeouts, TLS verification: all worth retrying later
            return _Result(error=_cleanText(repr(e)))

        finally:
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass


    def _describe(self, result):
        """ One log fragment describing a failed request.

        Arguments:
            result: [_Result]

        Return:
            [str]
        """

        if result.status is None:
            return "no response ({:s})".format(result.error or "unknown error")

        return "HTTP {:d} code={:s} message={:s} request_id={:s}".format(
            result.status, result.code or "-", result.message or "-", result.request_id or "-")


    # -----------------------------------------------------------------------
    # Sending queued items
    # -----------------------------------------------------------------------

    def _processDue(self, now, deadline=None):
        """ Try every item that is due, detections first, then files in queue order.

        Arguments:
            now: [float] Unix time for scheduling.

        Keyword arguments:
            deadline: [float] Unix time after which no new request is started. None for no limit.

        Return:
            [int] Number of requests made.
        """

        # After a 401 nothing is sent for an hour: it would only produce more 401s
        if now < self._auth_hold_until:
            return 0

        ordered = [r for r in self._items if r["kind"] == "detections"] + \
                  [r for r in self._items if r["kind"] == "file"]

        n_requests = 0
        for rec in ordered:

            if rec["next_attempt"] > now:
                continue

            if deadline is not None and time.time() >= deadline:
                break

            # Images wait for their FF, which is what clears the request on the server
            if rec["kind"] == "file" and self._waitsForFF(rec):
                continue

            outcome = self._attempt(rec)
            if outcome is None:
                continue

            n_requests += 1
            if not self._handleOutcome(rec, outcome, now):
                break

        return n_requests


    def _waitsForFF(self, rec):
        """ Tell whether an image upload should wait because its FF is still queued.

        Arguments:
            rec: [dict] File record.

        Return:
            [bool]
        """

        m = IMAGE_FILE_RE.match(rec["name"])
        if m is None:
            return False

        ff_name = m.group(1) + ".fits"
        for other in self._items:
            if other["kind"] == "file" and other["name"] in (ff_name, ff_name + ".bz2"):
                return True

        return False


    def _removeItem(self, rec):
        """ Take a record out of the queue.

        Arguments:
            rec: [dict]
        """

        try:
            self._items.remove(rec)
        except ValueError:
            pass

        self._dirty = True


    def _attempt(self, rec):
        """ Send one queued item.

        Arguments:
            rec: [dict] Queue record.

        Return:
            [_Result] or None if the item was dropped before sending.
        """

        # The file may have been removed by the disk cleanup in the meantime
        if not os.path.isfile(rec["path"]):
            log.warning("Sprite upload: dropping {:s} {:s}, file {:s} no longer exists".format(
                rec["kind"], rec["name"], rec["path"]))
            self._removeItem(rec)
            return None

        if rec["kind"] == "detections":
            return self._sendDetections(rec)

        return self._sendFile(rec)


    def _sendDetections(self, rec):
        """ POST the payload file's bytes unchanged, so every retry has the same digest.

        Arguments:
            rec: [dict] Detections record.

        Return:
            [_Result] or None if dropped.
        """

        try:
            size = os.path.getsize(rec["path"])
            if size > MAX_DETECTIONS_BYTES:
                log.warning("Sprite upload: dropping detections {:s}, payload is {:d} bytes, above the "
                            "server limit of {:d}".format(rec["name"], size, MAX_DETECTIONS_BYTES))
                self._removeItem(rec)
                return None

            with open(rec["path"], "rb") as f:
                body = f.read()

        except Exception as e:
            log.warning("Sprite upload: dropping detections {:s}, payload unreadable: {:s}".format(
                rec["name"], repr(e)))
            self._removeItem(rec)
            return None

        return self._request("POST", DETECTIONS_PATH, body=body, content_type="application/json")


    def _sendFile(self, rec):
        """ PUT a file as the raw request body, streamed from disk.

        Arguments:
            rec: [dict] File record.

        Return:
            [_Result] or None if dropped.
        """

        name = rec["name"]

        # Names are checked here too, so nothing goes out that the server would refuse by name
        if FF_FILE_RE.match(name):
            limit = MAX_FF_BYTES
        elif IMAGE_FILE_RE.match(name):
            limit = MAX_IMAGE_BYTES
        else:
            log.warning("Sprite upload: dropping file {:s}, the server does not accept that name".format(
                name))
            self._removeItem(rec)
            return None

        try:
            size = os.path.getsize(rec["path"])
        except Exception as e:
            log.warning("Sprite upload: dropping file {:s}: {:s}".format(name, repr(e)))
            self._removeItem(rec)
            return None

        if size > limit:
            log.warning("Sprite upload: not sending {:s}, {:d} bytes is above the server limit "
                        "of {:d}".format(name, size, limit))
            self._removeItem(rec)
            return None

        return self._request("PUT", FILES_PATH_PREFIX + name, file_path=rec["path"],
                             content_type="application/octet-stream", timeout=FILE_TIMEOUT)


    def _handleOutcome(self, rec, result, now):
        """ Remove, drop or reschedule an item after a request.

        Arguments:
            rec: [dict] Queue record.
            result: [_Result]
            now: [float] Unix time for scheduling.

        Return:
            [bool] True if the worker should continue with the next item in this pass.
        """

        verdict = classifyResult(result)

        if verdict == "ok":
            duplicate = bool(result.data.get("duplicate"))
            log.info("Sprite upload: {:s} {:s} {:s}".format(
                rec["kind"], rec["name"], "already on the server" if duplicate else "uploaded"))
            self._removeItem(rec)
            self._auth_logged.clear()
            return True

        if verdict == "permanent":
            log.error("Sprite upload: dropping {:s} {:s}, the server refused it: {:s}".format(
                rec["kind"], rec["name"], self._describe(result)))
            self._removeItem(rec)
            return True

        rec["attempts"] += 1
        self._dirty = True

        if verdict == "auth":
            self._authFailure(result, now)
            rec["next_attempt"] = now + AUTH_BACKOFF
            return False

        # Transient: back off, stored as an absolute time so it survives a restart
        delay = backoffDelay(rec["attempts"])
        rec["next_attempt"] = now + delay

        msg = "Sprite upload: {:s} {:s} failed (attempt {:d}), retrying in {:.0f} s: {:s}".format(
            rec["kind"], rec["name"], rec["attempts"], delay, self._describe(result))
        if rec["attempts"] == 1 or rec["attempts"]%10 == 0:
            log.warning(msg)
        else:
            log.debug(msg)

        # The network or the server is probably down, so the other items can wait too
        return False


    def _authFailure(self, result, now):
        """ Pause all uploads for an hour after a 401 or clock error, logging once per cause.

        Arguments:
            result: [_Result]
            now: [float] Unix time.
        """

        self._auth_hold_until = now + AUTH_BACKOFF
        cause = result.code or "http_{}".format(result.status)

        if cause in self._auth_logged:
            log.debug("Sprite upload: still not authorised ({:s}), waiting".format(cause))
            return

        self._auth_logged.add(cause)

        if cause in ("stale_timestamp", "bad_timestamp"):
            hint = "the station clock is probably wrong; check NTP"
        else:
            hint = "the station key is probably not registered on the server yet"

        log.error("Sprite upload: server rejected the signature ({:s}): {:s}. Queued items are kept and "
                  "retried every hour.".format(self._describe(result), hint))


    # -----------------------------------------------------------------------
    # File requests
    # -----------------------------------------------------------------------

    def _pollRequests(self, now):
        """ Ask the server which FF files it wants and queue the ones this station has.

        Arguments:
            now: [float] Unix time for scheduling.
        """

        self._next_poll = now + POLL_INTERVAL + random.uniform(0.0, POLL_JITTER)

        # With FF uploads off the station never answers, so it does not ask either
        if not self.upload_ff or now < self._auth_hold_until:
            return

        path = REQUESTS_PATH_TEMPLATE.format(self.station_id.upper())
        result = self._request("GET", path)
        verdict = classifyResult(result)

        if verdict == "auth":
            self._authFailure(result, now)
            return

        if verdict != "ok":
            log.warning("Sprite upload: could not fetch file requests: {:s}".format(self._describe(result)))
            return

        self._auth_logged.clear()

        entries = result.data.get("requests")
        if not isinstance(entries, list):
            log.warning("Sprite upload: file requests response has no list")
            return

        for entry in entries[:1000]:
            if isinstance(entry, dict) and isinstance(entry.get("ff_name"), STRING_TYPES):
                self._answerRequest(entry["ff_name"], now)


    def _warnRequestOnce(self, ff_name, message):
        """ Warn about a request that cannot be answered, once per FF per run (it is listed every poll).

        Arguments:
            ff_name: [str]
            message: [str]
        """

        if ff_name in self._warned_requests:
            return

        self._warned_requests.add(ff_name)
        log.warning("Sprite upload: server requested {:s}, {:s}".format(_cleanText(ff_name, 80), message))


    def _answerRequest(self, ff_name, now):
        """ Queue the FF and its images for one request: FF first, then marked, then unmarked.

        Arguments:
            ff_name: [str] Requested FF name.
            now: [float] Unix time.
        """

        m = FF_NAME_RE.match(ff_name)
        if m is None or m.group(1) != self.station_id.upper():
            self._warnRequestOnce(ff_name, "which is not an FF name of this station; ignoring")
            return

        known = self._known.get(ff_name)
        if known is None:
            self._warnRequestOnce(ff_name, "which this station does not know about; ignoring")
            return

        files = known.get("files") or {}

        # The server takes the compressed copy too, so prefer it when one sits next to the FF
        ff_path = None
        if files.get("ff"):
            for candidate in (files["ff"] + ".bz2", files["ff"]):
                if os.path.isfile(candidate):
                    ff_path = candidate
                    break

        if ff_path is None:
            self._warnRequestOnce(ff_name, "but the FF file is no longer on disk; ignoring")
            return

        upload_name = ff_name + (".bz2" if ff_path.endswith(".bz2") else "")
        new = [(upload_name, ff_path)]

        # Image names are rebuilt from the FF stem, so they always match the server's pattern
        stem = ff_name[:-len(".fits")]
        for kind in ("marked", "unmarked"):
            path = files.get(kind)
            if not path or not os.path.isfile(path):
                continue

            ext = os.path.splitext(path)[1].lower().lstrip(".")
            if ext not in ("jpg", "jpeg", "png"):
                continue

            new.append(("{:s}_{:s}.{:s}".format(stem, kind, ext), path))

        for name, path in new:
            if self._addRecord(self._newRecord("file", name, path, None, now)):
                log.info("Sprite upload: server requested {:s}, queued {:s}".format(ff_name, name))
                self._dirty = True


    # -----------------------------------------------------------------------
    # Reboot lock
    # -----------------------------------------------------------------------

    def _removeStaleLock(self):
        """ Remove a .sprite reboot lock left behind by a power cut. """

        try:
            if os.path.isfile(self.lock_path):
                age = time.time() - os.path.getmtime(self.lock_path)
                if age > STALE_LOCK_AGE:
                    os.remove(self.lock_path)
                    log.info("Sprite upload: removed stale reboot lock {:s} ({:.1f} h old)".format(
                        self.lock_path, age/3600.0))

        except Exception as e:
            log.warning("Sprite upload: could not check reboot lock {:s}: {:s}".format(
                self.lock_path, repr(e)))
