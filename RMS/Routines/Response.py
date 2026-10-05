""" Camera response: the mapping between the 8-bit codes a camera outputs and linear light.

RMS has always assumed a pure power law, linear = code**(1/gamma) (config.gamma). A camera's
real curve can differ, and where it does the power law misreads light. The ContrailCast/OpenIPC
science cameras apply gamma 0.5 as a NODE TABLE with straight lines between nodes (Goke 1025
nodes, Hi3516CV300 257), so their curve is a straight line below the first node (CV300 codes
0-16, Goke 0-8) and a chord between every pair of nodes above it. Decoding those codes as
code**2 biases star photometry by tenths of a magnitude on a CV300 night sky -- in a direction
that depends on the sky level, so a photometric fit cannot absorb it (A/B with
ExtractStars.extractStarsFF, 2026-10-05). These cameras report their real decode table
(`gamma decode` on the control port, 256 linear values); this module carries it.

A ResponseCurve built from a gamma (ResponseCurve.power) reproduces the old arithmetic exactly,
so every camera without a table, and every caller that still passes a float gamma, is unchanged.
"""

from __future__ import print_function, division, absolute_import

import hashlib
import json
import os
import re

import numpy as np


CODES = np.arange(256, dtype=np.float64)


class ResponseCurve(object):
    """ code (0..255 at 8 bits, scaled for other white points) -> relative linear light (0..1). """

    def __init__(self, linear, source, gamma=None, info=None):
        """
        Arguments:
            linear: [array] 256 linear values for codes 0..255, any scale (normalised to 0..1).
            source: [str] 'power' or 'camera'.

        Keyword arguments:
            gamma: [float] The nominal gamma (power curves; metadata for camera tables).
            info: [dict] Provenance (camera ip, soc, image, decode kind, nodes).
        """
        lin = np.asarray(linear, dtype=np.float64).copy()
        if lin.shape != (256,):
            raise ValueError("a response table needs 256 values, got %s" % (lin.shape,))
        if not np.all(np.isfinite(lin)) or lin[-1] <= lin[0]:
            raise ValueError("response table is not increasing")
        lin = (lin - lin[0])/(lin[-1] - lin[0])
        # strictly increasing, so the inverse (encode) is well defined
        lin = np.maximum.accumulate(lin)
        for i in range(1, 256):
            if lin[i] <= lin[i - 1]:
                lin[i] = lin[i - 1] + 1e-9
        self.linear = lin
        self.source = source
        self.gamma = gamma
        self.info = dict(info or {})


    @classmethod
    def power(cls, gamma):
        """ The classic RMS assumption: linear = (code/255)**(1/gamma). """
        return cls((CODES/255.0)**(1.0/float(gamma)), "power", gamma=float(gamma))


    @classmethod
    def fromCameraReply(cls, reply, gamma=0.5, info=None):
        """ Parse a `gamma decode` reply: 'gamma_decode decode=<kind> nodes=<n> v0 ... v255'. """
        if not reply:
            raise ValueError("empty gamma decode reply")
        m = re.search(r"gamma_decode\s+decode=(\w+)\s+nodes=(\d+)\s+(.*)", reply, re.S)
        if not m:
            raise ValueError("not a gamma decode reply: %r" % reply[:80])
        vals = [float(v) for v in m.group(3).split()]
        if len(vals) != 256:
            raise ValueError("gamma decode reply has %d values, expected 256" % len(vals))
        meta = dict(info or {})
        meta.update({"decode": m.group(1), "nodes": int(m.group(2))})
        return cls(vals, "camera", gamma=gamma, info=meta)


    @property
    def isPower(self):
        return self.source == "power"


    def ident(self):
        """ Short fingerprint of the table: the same curve gives the same id on every camera. """
        return hashlib.md5(np.round(self.linear, 9).tobytes()).hexdigest()[:12]


    def decode(self, codes, wp=255):
        """ Codes (any shape, fractional allowed, white point wp) -> relative linear light 0..1. """
        x = np.clip(np.asarray(codes, dtype=np.float64), 0, wp)*(255.0/wp)
        if self.isPower:
            return (x/255.0)**(1.0/self.gamma)
        return np.interp(x, CODES, self.linear)


    def encode(self, linear, wp=255):
        """ Relative linear light 0..1 -> code (fractional) at white point wp. The inverse of decode. """
        y = np.clip(np.asarray(linear, dtype=np.float64), 0.0, 1.0)
        if self.isPower:
            return wp*y**self.gamma
        return np.interp(y, self.linear, CODES)*(wp/255.0)


    def decodeLUT(self, wp=255):
        """ decode() for every integer code 0..wp, in RMS's 'linear code' units (0..wp). Used by the
        Cython compressors, which index a table per pixel. """
        return (wp*self.decode(np.arange(wp + 1), wp=wp)).astype(np.float64)


    def toDict(self):
        return {"source": self.source, "gamma": self.gamma, "id": self.ident(), "info": self.info,
                "linear": [round(float(v), 9) for v in self.linear]}


    @classmethod
    def fromDict(cls, d):
        if d.get("source") == "power":
            return cls.power(d["gamma"])
        return cls(d["linear"], d.get("source", "camera"), gamma=d.get("gamma"), info=d.get("info"))


    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.toDict(), f, indent=1)


    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.fromDict(json.load(f))



def responseOf(config):
    """ The response RMS should decode with: the camera's table when one was loaded for this
    station (config.response), else the power law of config.gamma. """
    r = getattr(config, "response", None)
    return r if isinstance(r, ResponseCurve) else ResponseCurve.power(config.gamma)


RESPONSE_FILE = "camera_response.json"


def loadNightResponse(dir_path):
    """ The response saved with a night of captured data (camera_response.json), or None. """
    p = os.path.join(dir_path, RESPONSE_FILE)
    if os.path.isfile(p):
        try:
            return ResponseCurve.load(p)
        except (ValueError, KeyError, OSError):
            return None
    return None


def _cameraIP(config):
    m = re.search(r'(?:\d{1,3}\.){3}\d{1,3}', str(getattr(config, "deviceID", "")))
    return m.group() if m else None


def _ask(ip, cmd, port=9600, timeout=3.0):
    import socket
    s = socket.create_connection((ip, port), timeout=timeout)
    try:
        s.settimeout(timeout)
        s.sendall((cmd + "\n").encode())
        s.shutdown(socket.SHUT_WR)
        buf = b""
        while True:
            d = s.recv(65536)
            if not d:
                break
            buf += d
        return buf.decode("latin1")
    finally:
        s.close()


def fetchCameraResponse(config, timeout=3.0):
    """ Ask the camera for its real decode table (`gamma decode` on the science firmware's
    control port). None when disabled (response_table: off), when the camera has no control
    server (stock/XM, file input) or when it does not answer -- RMS then keeps the power law. """
    if str(getattr(config, "response_table", "auto")).lower() != "auto":
        return None
    ip = _cameraIP(config)
    if ip is None:
        return None
    try:
        reply = _ask(ip, "gamma decode", timeout=timeout)
        info = {"ip": ip}
        try:
            si = _ask(ip, "sysinfo", timeout=timeout)
            info.update(dict(re.findall(r"(\w+)=(\S+)", si)))
        except OSError:
            pass
        return ResponseCurve.fromCameraReply(reply, gamma=getattr(config, "gamma", None), info=info)
    except (OSError, ValueError):
        return None


def prepareNightResponse(config, night_dir, log=None):
    """ At capture start: fetch the camera's table, save it with the night's data
    (camera_response.json) and set config.response, so the compressor and every later
    reduction of this night decode with the same curve. Without a table config.response is
    None (power law of config.gamma, unchanged behaviour). """
    r = fetchCameraResponse(config)
    config.response = r
    if r is not None and night_dir and os.path.isdir(night_dir):
        try:
            r.save(os.path.join(night_dir, RESPONSE_FILE))
        except OSError as e:
            if log:
                log.warning("Cannot save the camera response table: %s", e)
    if log:
        if r is None:
            log.info("Camera response: power law, gamma %s", getattr(config, "gamma", None))
        else:
            log.info("Camera response: camera table %s (%s, %s nodes)", r.ident(),
                     r.info.get("decode"), r.info.get("nodes"))
    return r


def useNightResponse(config, dir_path):
    """ Reducing a night (live or reprocessed, any tool): decode with the table saved with that
    night, or the power law when there is none, so the same night always reduces the same way.
    Returns config.response. """
    if str(getattr(config, "response_table", "auto")).lower() != "auto":
        config.response = None
    else:
        config.response = loadNightResponse(dir_path) if dir_path else None
    return config.response
