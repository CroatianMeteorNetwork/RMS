""" In-process fake of the spritemap server for the sprite upload tests.

    Listens on 127.0.0.1 on a free port, records every request and verifies each signature on its own,
    with the station's public key and a canonical string rebuilt from what actually arrived (method, path,
    headers and the hash of the received body). It does not share code with RMS/SpriteUpload.py, so a bug
    in the client's canonical string shows up as a failed verification here.

    Responses can be scripted per method and path prefix; otherwise it answers like the real server:
    detections are acknowledged (a repeated body as a duplicate), files are stored, and the requests
    endpoint lists whatever is in .wanted until the FF file is uploaded.
"""

from __future__ import print_function, division, absolute_import

import hashlib
import json
import threading
import time

try:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
except ImportError:
    ThreadingHTTPServer = None
    BaseHTTPRequestHandler = object


class RecordedRequest(object):
    """ One request as the fake server saw it. """

    def __init__(self, method, path, query, headers, body, chunked):
        self.method = method
        self.path = path
        self.query = query
        self.headers = headers
        self.body = body
        self.chunked = chunked
        self.body_sha256 = hashlib.sha256(body).hexdigest()
        self.signature_ok = False
        self.canonical = None


def _independentCanonical(method, path, station_id, timestamp, body_sha256):
    """ The canonical string from docs/SECURITY.md, written out again on purpose. """

    return ("SPRITEMAP-V1\n" + method + "\n" + path + "\n" + station_id.upper() + "\n" + timestamp
            + "\n" + body_sha256).encode("ascii")


def verifySignature(public_key, canonical, signature):
    """ Verify a signature with a public key of any supported type.

    Arguments:
        public_key: [object] RSA, Ed25519 or EC public key.
        canonical: [bytes] Signed bytes.
        signature: [bytes] Raw signature.

    Return:
        [bool]
    """

    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa

    try:
        if isinstance(public_key, rsa.RSAPublicKey):
            public_key.verify(signature, canonical, padding.PKCS1v15(), hashes.SHA256())

        elif isinstance(public_key, ed25519.Ed25519PublicKey):
            public_key.verify(signature, canonical)

        elif isinstance(public_key, ec.EllipticCurvePublicKey):
            hash_alg = {"secp256r1": hashes.SHA256(), "secp384r1": hashes.SHA384(),
                        "secp521r1": hashes.SHA512()}[public_key.curve.name]
            public_key.verify(signature, canonical, ec.ECDSA(hash_alg))

        else:
            return False

        return True

    except (InvalidSignature, ValueError):
        return False


class FakeSpriteServer(object):
    """ Threaded HTTP server that behaves like the spritemap ingest API.

    Arguments:
        public_key: [object] Public key of the station.
        station_id: [str] Station code the key belongs to.
    """

    def __init__(self, public_key, station_id):

        self.public_key = public_key
        self.station_id = station_id.upper()
        self.requests = []
        self.wanted = []
        self.stored = {}
        self.seen_bodies = set()
        self.scripts = []
        self.lock = threading.Lock()
        self.hang_events = []
        self.max_skew = 300

        server = self

        # The handler reads the body exactly as the real server would, by Content-Length
        class Handler(BaseHTTPRequestHandler):

            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _handle(self):
                server._handle(self)

            do_GET = _handle
            do_POST = _handle
            do_PUT = _handle

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.url = "http://127.0.0.1:{:d}".format(self.port)
        self.thread = threading.Thread(target=self.httpd.serve_forever)
        self.thread.daemon = True
        self.thread.start()


    def close(self):
        """ Release hanging handlers and stop the server. """

        for ev in self.hang_events:
            ev.set()

        self.httpd.shutdown()
        self.httpd.server_close()


    def script(self, method, path_prefix, status=200, body=None, headers=None, hang=None, times=1):
        """ Queue a canned response for the next matching request(s).

        Arguments:
            method: [str] HTTP method.
            path_prefix: [str] Path prefix to match.

        Keyword arguments:
            status: [int] Status code.
            body: [bytes, dict or list] Body; dicts and lists are JSON encoded.
            headers: [dict] Extra response headers.
            hang: [threading.Event] Wait for this event before answering.
            times: [int] How many requests this answers.
        """

        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")

        if hang is not None:
            self.hang_events.append(hang)

        with self.lock:
            for _ in range(times):
                self.scripts.append((method, path_prefix, status, body or b"", headers or {}, hang))


    def byPath(self, method, path_prefix=""):
        """ Recorded requests of one method under a path prefix. """

        return [r for r in self.requests if r.method == method and r.path.startswith(path_prefix)]


    def _handle(self, h):

        path, _, query = h.path.partition("?")
        headers = dict((k.lower(), v) for k, v in h.headers.items())
        chunked = "chunked" in headers.get("transfer-encoding", "").lower()

        # A chunked body is refused like the real server does, without reading it
        body = b""
        if not chunked and "content-length" in headers:
            body = h.rfile.read(int(headers["content-length"]))

        rec = RecordedRequest(h.command, path, query, headers, body, chunked)

        # Verify from what arrived, not from what the client meant to send
        try:
            canonical = _independentCanonical(h.command, path, headers.get("x-station-id", ""),
                                              headers.get("x-timestamp", ""), rec.body_sha256)
            signature = bytes.fromhex(headers.get("x-signature", ""))
            rec.canonical = canonical
            rec.signature_ok = (verifySignature(self.public_key, canonical, signature)
                                and headers.get("x-station-id", "").upper() == self.station_id
                                and abs(time.time() - int(headers.get("x-timestamp", "0"))) <= self.max_skew)
        except Exception:
            rec.signature_ok = False

        with self.lock:
            self.requests.append(rec)
            script = None
            for i, s in enumerate(self.scripts):
                if s[0] == h.command and path.startswith(s[1]):
                    script = self.scripts.pop(i)
                    break

        if script is not None:
            _, _, status, resp_body, resp_headers, hang = script
            if hang is not None:
                hang.wait(30)
            return self._send(h, status, resp_body, resp_headers)

        if chunked:
            return self._sendJson(h, 411, {"error": {"code": "length_required", "message": "Length"},
                                           "request_id": "req411"})

        if not rec.signature_ok:
            return self._sendJson(h, 401, {"error": {"code": "unauthorized",
                                                     "message": "Authentication failed"},
                                           "request_id": "req401"})

        return self._default(h, rec)


    def _default(self, h, rec):

        # Detections: the same body again is a duplicate, as on the real server
        if rec.method == "POST" and rec.path == "/api/v1/detections":
            duplicate = rec.body_sha256 in self.seen_bodies
            self.seen_bodies.add(rec.body_sha256)
            resp = {"success": True, "submission_id": len(self.seen_bodies), "frames": 1, "detections": 1}
            if duplicate:
                resp["duplicate"] = True
            return self._sendJson(h, 200, resp)

        # Files: an FF upload clears its request
        if rec.method == "PUT" and rec.path.startswith("/api/v1/files/"):
            name = rec.path[len("/api/v1/files/"):]
            self.stored[name] = rec.body
            ff_name = name[:-4] if name.endswith(".bz2") else name
            if ff_name in self.wanted:
                self.wanted.remove(ff_name)
            return self._sendJson(h, 200, {"success": True, "file": name, "size_bytes": len(rec.body),
                                           "sha256": rec.body_sha256, "duplicate": False})

        if rec.method == "GET" and rec.path == "/api/v1/stations/{:s}/requests".format(self.station_id):
            return self._sendJson(h, 200, {"station_id": self.station_id,
                                           "requests": [{"ff_name": n, "event_id": 1} for n in self.wanted]})

        return self._sendJson(h, 404, {"error": {"code": "not_found", "message": "Not found"},
                                       "request_id": "req404"})


    def _sendJson(self, h, status, obj):
        return self._send(h, status, json.dumps(obj).encode("utf-8"), {"Content-Type": "application/json"})


    def _send(self, h, status, body, headers):

        try:
            h.send_response(status)
            for k, v in headers.items():
                h.send_header(k, v)
            h.send_header("Content-Length", str(len(body)))
            h.send_header("Connection", "close")
            h.end_headers()
            h.wfile.write(body)
        except Exception:
            pass

        h.close_connection = True
