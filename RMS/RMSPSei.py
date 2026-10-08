""" RMSP v5 provenance SEI: the per-frame record OpenIPC science cameras embed in the H.264/H.265
stream (user-data-unregistered SEI, UUID "science-frame-v5"), carried into the saved mkv segments.

Live capture (BufferedCapture) parses it from each access unit as it arrives. readVideoFrameTimes
does the same for a saved video file, so an offline reduction (SkyFit2) times every frame exactly
as capture did: row-0 start of integration, capture_utc - exp - k.

Kept free of GStreamer/camera-control imports so the readers can use it on any machine.
"""

from __future__ import print_function, division, absolute_import

import shutil
import struct
import subprocess
from collections import deque

import numpy as np

from RMS.Logger import getLogger

log = getLogger("rmslogger")


# Fixed sensor-readout offset (raw_pts stamp point -> true row-0 readout), established
# +88 us on the IMX307 (VMAX/HMAX/SHS1 timing + PPS-LED cal); see reference_rmsp_provenance.
K_READOUT_S = 88e-6


def rmspCapUtc(data):
    """Parse the first checksum-valid RMSP v5 provenance SEI in a raw (escaped) H.264 access
    unit. Returns (capture_utc_s, exp_s, soc_temp_c or None, meta dict, frame_seq) or None.
    A v5 record describes the frame it rides in (venc emits it in the frame's own access unit).
    Earlier versions rode a LATER frame and are not accepted. Layout matches
    venc/main.c build_rmsp_payload (fields XOR-0xFF after the 'RMSP' magic).
    capture_utc = (sec+usec) - (mono_pts_us - raw_pts_us): host-clock at emit minus the
    camera-side capture->emit delay, both from the same back-to-back cal. The raw
    (utc_s, mono_us, raw_pts_us) triple is returned too, for ClockPairFilter.
    Cheap enough to run per frame: locate the magic in the escaped bytes (memchr speed) and
    de-escape only a short slice from there. The magic has no zero bytes, so the emulation-
    prevention state is known (clean) at that point; the SEI precedes slice data, so the first
    checksum-valid magic is the record."""
    def deesc(b):
        o = bytearray(); z = 0
        for x in b:
            if z >= 2 and x == 3:
                z = 0; continue
            o.append(x); z = z + 1 if x == 0 else 0
        return bytes(o)
    i = 0
    while True:
        j = data.find(b"RMSP", i)
        if j < 0:
            return None
        i = j + 4
        u = deesc(data[j:j + 96])          # 61-byte record plus room for emulation-prevention bytes
        if len(u) < 6:
            continue
        v = u[4] ^ 0xFF
        L = 61
        if len(u) < L:
            continue
        u = u[:4] + bytes(x ^ 0xFF for x in u[4:L])
        ck = 0
        for x in u[4:L - 1]:
            ck ^= x
        if v != 5 or ck != u[L - 1]:
            continue
        sec = struct.unpack('<I', u[6:10])[0]; usec = struct.unpack('<I', u[10:14])[0]
        frame_seq = struct.unpack('<I', u[14:18])[0]
        raw_pts = struct.unpack('<I', u[44:48])[0]; mono = struct.unpack('<I', u[48:52])[0]
        exp_us = struct.unpack('<I', u[18:22])[0]
        fl = u[5]
        temp = struct.unpack('<h', u[42:44])[0]/10.0 if (fl & 0x01) else None   # flags b0 = temp valid
        delay = (mono - raw_pts) & 0xffffffff
        # Per-block photometric provenance (RMS.SEIBlockMeta): gains are HI_MPI_ISP_QueryExposureInfo
        # units, x1024 = 1x; WB gains x256 = 1x; mean_qp = this frame's actual coded QP.
        # Flags: b1 wb valid, b3 exposure/gains valid, b6 qp valid
        exp_ok = bool(fl & 0x08)
        meta = {
            'exp_s': exp_us/1e6 if exp_ok else None,
            'again': struct.unpack('<I', u[22:26])[0]/1024.0 if exp_ok else None,
            'dgain': struct.unpack('<I', u[26:30])[0]/1024.0 if exp_ok else None,
            'ispdgain': struct.unpack('<I', u[30:34])[0]/1024.0 if exp_ok else None,
            'wb_r': struct.unpack('<H', u[34:36])[0]/256.0 if (fl & 0x02) else None,
            'wb_b': struct.unpack('<H', u[36:38])[0]/256.0 if (fl & 0x02) else None,
            'wb_g': struct.unpack('<H', u[38:40])[0]/256.0 if (fl & 0x02) else None,
            'qp': u[58] if (fl & 0x40) else None,
        }
        return ((sec + usec/1e6) - delay/1e6, exp_us/1e6, temp, meta, frame_seq,
                (sec + usec/1e6, mono, raw_pts))


class ClockPairFilter(object):
    """Denoise the camera's UTC/MPP clock pair; never the frame's own hardware stamp.

    capture_utc = utc - (mono - raw_pts) = (utc - mono) + raw_pts. raw_pts is the frame's
    hardware PTS and carries all real per-frame timing (exposure steps, VMAX, drops); it is
    used as is. utc - mono is only the offset between two camera clocks, read back to back
    when the encoder thread handles the frame. It truly moves only as chrony slews the
    clock (a few ppm), but each reading scatters ~16 us, and a preemption between the two
    reads (single-core CV300) makes one frame's offset 3-5 ms late -- measured on US005F,
    2026-09-26: 59 single-frame glitches/hour, each exactly a jump in utc - mono. The
    offset used is the median of the last `n` frames' (causal: no added latency; one bad
    pair cannot move it). A jump > reset_s clears the window: the 32-bit us MPP counter
    wraps every ~71.6 min (offset +4294.97 s, expected), or the camera clock was stepped
    (logged)."""

    _WRAP_S = 4294.967296

    def __init__(self, n=25, reset_s=1.0):
        self._off = deque(maxlen=n)
        self._reset_s = reset_s
        self.steps = 0

    def capture_utc(self, utc_s, mono_us, raw_pts_us):
        delay_s = ((mono_us - raw_pts_us) & 0xffffffff)/1e6
        off = utc_s - mono_us/1e6
        if self._off:
            jump = off - self._off[-1]
            if abs(jump) > self._reset_s:
                if abs(abs(jump) - self._WRAP_S) > self._reset_s:
                    self.steps += 1
                    log.info("SEI clock pair: camera clock stepped {:+.3f} s -- offset window reset"
                             .format(jump))
                self._off.clear()
        self._off.append(off)
        med = sorted(self._off)[len(self._off)//2]
        return med + mono_us/1e6 - delay_s


def _pairOffsetMedian(recs, n=25, reset_s=1.0):
    """ Offline counterpart of ClockPairFilter: the utc - mono offset of every frame replaced by
        the CENTRED median of its n neighbours (the whole file is known, so no causal warm-up at
        the start of a segment), split at jumps > reset_s (MPP wrap / clock step).
        recs: list of (utc_s, mono_us, raw_pts_us). Returns the filtered capture_utc per frame. """
    utc = np.array([r[0] for r in recs], dtype=np.float64)
    mono = np.array([r[1] for r in recs], dtype=np.float64)
    raw = np.array([r[2] for r in recs], dtype=np.int64)
    off = utc - mono/1e6
    delay = ((mono.astype(np.int64) - raw) & 0xffffffff)/1e6
    breaks = np.concatenate([[0], np.nonzero(np.abs(np.diff(off)) > reset_s)[0] + 1, [len(off)]])
    med = np.empty_like(off)
    h = n//2
    for a, b in zip(breaks[:-1], breaks[1:]):
        for i in range(a, b):
            med[i] = np.median(off[max(a, i - h):min(b, i + h + 1)])
    return med + mono/1e6 - delay


def _iterNals(stream, chunk_size=1 << 22):
    """ Yield the NAL units (header byte onwards) of an Annex-B byte stream, read in chunks. """
    buf = b""
    while True:
        data = stream.read(chunk_size)
        buf += data
        starts = []
        i = 0
        while True:
            j = buf.find(b"\x00\x00\x01", i)
            if j < 0:
                break
            starts.append(j + 3)
            i = j + 3
        # Emit every NAL that is followed by another start code; keep the last one open
        last = None if data else len(buf)
        for k in range(len(starts) - (1 if data else 0)):
            end = starts[k + 1] - 3 if k + 1 < len(starts) else last
            nal = buf[starts[k]:end]
            # Strip the zero of a following 4-byte start code
            yield nal.rstrip(b"\x00") if k + 1 < len(starts) else nal
        if not data:
            return
        if starts:
            buf = buf[starts[-1] - 3:]


def readVideoFrameTimes(file_path):
    """ Per-frame RMSP SEI times of a video file, in decode order (= display order: the camera
        encodes without B-frames).

    Demuxes the elementary stream with ffmpeg (stream copy, no decoding) and assigns each RMSP
    record to the picture whose access unit carries it. Frames without a valid record are
    interpolated from their neighbours by frame index.

    Arguments:
        file_path: [str] Path to the video file (mkv/mp4...).

    Return:
        None if ffmpeg is unavailable or the stream has fewer than 2 RMSP records, otherwise a dict:
            int_start: [ndarray] Row-0 start of integration of each frame [unix s].
            exp_s: [ndarray] Exposure of each frame [s] (nan where unknown).
            frame_seq: [ndarray] Camera frame sequence number (-1 where no record).
            n_interp: [int] Number of frames whose time was interpolated.
    """

    if (shutil.which("ffmpeg") is None) or (shutil.which("ffprobe") is None):
        print("ffmpeg/ffprobe not found, cannot read the per-frame SEI timestamps")
        return None

    try:
        codec = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                         "-show_entries", "stream=codec_name", "-of", "csv=p=0",
                                         file_path]).decode().strip()
    except (subprocess.CalledProcessError, OSError):
        return None

    if codec == "h264":
        hevc = False
    elif codec == "hevc":
        hevc = True
    else:
        return None

    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", file_path, "-map", "0:v:0", "-c", "copy",
                             "-bsf:v", "{:s}_mp4toannexb".format(codec), "-f", codec, "-"],
                            stdout=subprocess.PIPE)

    # One entry per picture: the record of its access unit, or None
    frame_recs = []
    pending = None
    try:
        for nal in _iterNals(proc.stdout):

            if len(nal) < 3:
                continue

            if hevc:
                nal_type = (nal[0] >> 1) & 0x3f
                is_sei = (nal_type == 39)
                # VCL NAL with first_slice_segment_in_pic_flag set
                new_pic = (nal_type < 32) and bool(nal[2] & 0x80)
            else:
                nal_type = nal[0] & 0x1f
                is_sei = (nal_type == 6)
                # Slice with first_mb_in_slice == 0 (ue(v) '1')
                new_pic = (nal_type in (1, 5)) and bool(nal[1] & 0x80)

            if is_sei:
                rec = rmspCapUtc(nal)
                if rec is not None:
                    pending = rec

            elif new_pic:
                frame_recs.append(pending)
                pending = None

    finally:
        proc.stdout.close()
        proc.wait()

    good = np.array([r is not None for r in frame_recs], dtype=bool)
    if np.count_nonzero(good) < 2:
        return None

    idx = np.arange(len(frame_recs))
    recs = [r for r in frame_recs if r is not None]

    # Integration start of the frames with a record, as capture computes it
    cap_utc = _pairOffsetMedian([r[5] for r in recs])
    exp_good = np.array([r[1] for r in recs])
    int_start_good = cap_utc - K_READOUT_S - exp_good

    int_start = np.interp(idx, idx[good], int_start_good)

    exp_s = np.full(len(frame_recs), np.nan)
    exp_s[good] = [r[3]['exp_s'] if r[3]['exp_s'] is not None else np.nan for r in recs]

    frame_seq = np.full(len(frame_recs), -1, dtype=np.int64)
    frame_seq[good] = [r[4] for r in recs]

    return {"int_start": int_start, "exp_s": exp_s, "frame_seq": frame_seq,
            "n_interp": int(np.count_nonzero(~good))}
