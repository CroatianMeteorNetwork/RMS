""" Per-frame exposure metadata embedded in saved raw frames.

On cameras that send the RMSP v5 provenance SEI (OpenIPC science cameras), every frame's
record carries its capture time, exposure, analog/sensor-digital/ISP-digital gains, WB
gains, SoC temperature and frame sequence number. BufferedCapture copies the record of each
frame that goes into the raw-frame buffer into a shared per-frame row (fillRow), and
RawFrameSaver embeds it -- plus the highlight-rebuild info when a frame was rebuilt -- as
one JSON line in the saved image (config.save_frame_metadata):

    PNG:  a tEXt chunk with keyword "RMS" right after IHDR
    JPEG: a COM (comment) segment right after SOI

The image data is unchanged; exiftool, Pillow (Image.info / .text) and ImageMagick read both.
With it a frame converts to light on a fixed scale: linear = (v/255)^2 (pure gamma 0.5),
light = linear / (exp_us * again * dgain * ispdgain), per channel also / its WB gain.

Caveat: the exposure is the one the ISP reported when the frame was encoded; right after an
exposure change the sensor may still have integrated the previous one for a frame or two.
"""

from __future__ import print_function, division, absolute_import

import json
import struct
import zlib

import cv2
import numpy as np


# Layout of one shared-memory row (float64). "valid" is 1.0 when the row holds a record.
FIELDS = ("valid", "capture_utc", "exp_us", "again", "dgain", "ispdgain", "wb_r", "wb_b", "temp_c", "frame_seq")
NFIELDS = len(FIELDS)


def fillRow(row, cu):
    """ Fill a shared row from a parsed RMSP record (BufferedCapture._rmspCapUtc tuple:
        capture_utc, exp_s, temp_c, meta dict, frame_seq, ...), or mark it empty.
    """
    row[:] = np.nan
    row[0] = 0.0
    if cu is None:
        return
    try:
        meta = cu[3] or {}
        exp_s = meta.get("exp_s")
        row[0] = 1.0
        row[1] = cu[0]
        row[2] = exp_s*1e6 if exp_s is not None else np.nan
        for i, k in ((3, "again"), (4, "dgain"), (5, "ispdgain"), (6, "wb_r"), (7, "wb_b")):
            v = meta.get(k)
            row[i] = v if v is not None else np.nan
        row[8] = cu[2] if cu[2] is not None else np.nan
        row[9] = cu[4] if len(cu) > 4 and cu[4] is not None else np.nan
    except Exception:
        row[0] = 0.0


def rowToDict(row):
    """ The metadata dict for one shared row, or None if the row is empty. """
    if row is None or not row[0] > 0:
        return None
    d = {"rmsp": 5}
    for i, k in enumerate(FIELDS[1:], 1):
        v = float(row[i])
        if v == v:                      # skip NaN
            if k in ("exp_us", "frame_seq"):
                d[k] = int(round(v))
            elif k == "capture_utc":
                d[k] = round(v, 6)
            else:
                d[k] = round(v, 4)
    return d


def _pngWithText(png_bytes, keyword, text):
    """ PNG bytes with a tEXt chunk inserted right after IHDR. """
    data = keyword.encode("latin-1") + b"\x00" + text.encode("latin-1", "replace")
    chunk = struct.pack(">I", len(data)) + b"tEXt" + data + struct.pack(">I", zlib.crc32(b"tEXt" + data) & 0xffffffff)
    ihdr_end = 8 + 4 + 4 + 13 + 4       # signature + IHDR length/type/data/crc
    return png_bytes[:ihdr_end] + chunk + png_bytes[ihdr_end:]


def _jpegWithComment(jpg_bytes, text):
    """ JPEG bytes with a COM segment inserted right after SOI. """
    data = text.encode("utf-8")[:65533]
    seg = b"\xff\xfe" + struct.pack(">H", len(data) + 2) + data
    return jpg_bytes[:2] + seg + jpg_bytes[2:]


def writeImage(path, frame, params, meta):
    """ Encode frame by the extension of path with cv2 params, embedding meta (dict or None)
        as one JSON line, and write it. Returns True on success.
    """
    ext = ".png" if path.lower().endswith(".png") else ".jpg"
    ok, buf = cv2.imencode(ext, frame, params)
    if not ok:
        return False
    data = buf.tobytes()
    if meta:
        text = json.dumps(meta, separators=(",", ":"))
        data = _pngWithText(data, "RMS", text) if ext == ".png" else _jpegWithComment(data, text)
    with open(path, "wb") as f:
        f.write(data)
    return True


def readImageMeta(path):
    """ The embedded metadata dict of a saved frame, or None. """
    with open(path, "rb") as f:
        data = f.read()
    try:
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            i = 8
            while i + 8 <= len(data):
                n = struct.unpack(">I", data[i:i + 4])[0]
                typ = data[i + 4:i + 8]
                if typ == b"tEXt":
                    body = data[i + 8:i + 8 + n]
                    k, _, v = body.partition(b"\x00")
                    if k == b"RMS":
                        return json.loads(v.decode("latin-1"))
                if typ == b"IDAT":
                    break
                i += 12 + n
        elif data[:2] == b"\xff\xd8":
            i = 2
            while i + 4 <= len(data) and data[i] == 0xff:
                marker = data[i + 1]
                n = struct.unpack(">H", data[i + 2:i + 4])[0]
                if marker == 0xfe:
                    return json.loads(data[i + 4:i + 2 + n].decode("utf-8"))
                if marker == 0xda:
                    break
                i += 2 + n
    except Exception:
        return None
    return None
