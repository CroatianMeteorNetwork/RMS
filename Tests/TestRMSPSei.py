""" RMSP v5 provenance SEI: the parser reads every field of the record venc builds
(venc/main.c build_rmsp_payload), and the per-frame row and the per-block accumulator keep them.

Run with: python -m pytest Tests/TestRMSPSei.py
"""

from __future__ import print_function, division, absolute_import

import struct

import numpy as np

from RMS.RMSPSei import rmspCapUtc
from RMS.FrameMetadata import NFIELDS, fillRow, rowToDict
from RMS.SEIBlockMeta import SEIBlockAccumulator, SEI_META_FIELDS


def makeRecord(frame_seq=1000, flags=0xFF, exp_us=40000, again=16384, dgain=1024, ispdgain=1088,
               wb=(256, 256, 256), colortemp=5000, temp_x10=586, raw_pts=10000000, mono=10060000,
               t_row_ns=29629, vmax=1350, mean_qp=19, start_qp=21):
    """ One record as venc emits it: 'RMSP' | fields XOR-0xFF | checksum, inside an SEI NAL with
        emulation prevention. """
    raw = bytearray(61)
    raw[0:4] = b"RMSP"
    raw[4] = 5
    raw[5] = flags
    struct.pack_into('<IIIIIII', raw, 6, 1790000000, 500000, frame_seq, exp_us, again, dgain, ispdgain)
    struct.pack_into('<HHHHh', raw, 34, wb[0], wb[2], wb[1], colortemp, temp_x10)
    struct.pack_into('<IIIH', raw, 44, raw_pts, mono, t_row_ns, vmax)
    raw[58] = mean_qp
    raw[59] = start_qp
    ck = 0
    for x in raw[4:60]:
        ck ^= x
    raw[60] = ck
    payload = bytes(raw[:4]) + bytes(x ^ 0xFF for x in raw[4:])
    rbsp = bytes([5, 16 + len(payload)]) + b"science-frame-v5" + payload + b"\x80"

    # Emulation prevention: 00 00 0x (x <= 3) -> 00 00 03 0x
    out = bytearray([0x06]); z = 0
    for x in rbsp:
        if z >= 2 and x <= 3:
            out.append(3); z = 0
        out.append(x); z = z + 1 if x == 0 else 0
    return b"\x00\x00\x00\x01" + bytes(out)


def testParserReadsEveryField():
    cu = rmspCapUtc(makeRecord())
    assert cu is not None
    meta = cu[3]
    assert meta['exp_s'] == 0.04
    assert meta['again'] == 16.0 and meta['dgain'] == 1.0 and meta['ispdgain'] == 1088/1024.0
    assert meta['gains_exact'] is True
    assert meta['colortemp'] == 5000
    assert meta['t_row_ns'] == 29629 and meta['vmax'] == 1350
    assert meta['qp'] == 19 and meta['start_qp'] == 21
    assert meta['temp_c'] == 58.6 and meta['frame_seq'] == 1000
    assert cu[2] == 58.6 and cu[4] == 1000


def testParserHonoursFlags():
    # b7 clear: gains polled, not frame-exact; b5/b6 clear: no row timing, no QP; b1 clear: no WB
    meta = rmspCapUtc(makeRecord(flags=0xFF & ~0x80 & ~0x20 & ~0x40 & ~0x02))[3]
    assert meta['gains_exact'] is False
    assert meta['t_row_ns'] is None and meta['vmax'] is None
    assert meta['qp'] is None and meta['start_qp'] is None
    assert meta['wb_g'] is None and meta['colortemp'] is None
    # b3 clear: no exposure, so whether the gains are exact is unknown too
    assert rmspCapUtc(makeRecord(flags=0xFF & ~0x08))[3]['gains_exact'] is None


def testFrameRowKeepsNewFields():
    row = np.zeros(NFIELDS)
    fillRow(row, rmspCapUtc(makeRecord()))
    d = rowToDict(row)
    assert d['qp'] == 19 and d['start_qp'] == 21
    assert d['gains_exact'] is True
    assert d['colortemp'] == 5000 and d['t_row_ns'] == 29629 and d['vmax'] == 1350
    assert d['frame_seq'] == 1000 and d['exp_us'] == 40000


def testBlockAccumulator():
    acc = SEIBlockAccumulator()

    # Block 1: frames 100..109 with 103 and 104 missing; the gain steps at frame 107
    for s in (100, 101, 102, 105, 106, 107, 108, 109):
        acc.add(rmspCapUtc(makeRecord(frame_seq=s, again=16384 if s < 107 else 32768,
                                      flags=0xFF if s != 105 else 0x7F))[3])
    b = dict(zip(SEI_META_FIELDS, acc.snapshotAndReset(1)))
    assert b['nfrm'] == 8
    assert b['again_min'] == 16.0 and b['again_max'] == 32.0 and b['stable'] == 0.0
    assert b['gains_exact_n'] == 7
    assert b['t_row_ns'] == 29629 and b['vmax'] == 1350 and b['wb_g'] == 1.0
    assert (b['seq_first'], b['seq_last'], b['seq_lost']) == (100, 109, 2)

    # Block 2: a gap at the block boundary counts in the new block; a venc restart (counter back
    # to 0) is not a gap
    for s in (112, 113, 0, 1):
        acc.add(rmspCapUtc(makeRecord(frame_seq=s))[3])
    b = dict(zip(SEI_META_FIELDS, acc.snapshotAndReset(2)))
    assert (b['seq_first'], b['seq_last'], b['seq_lost']) == (112, 1, 2)
    assert b['stable'] == 1.0

    # Empty block: no data
    b = dict(zip(SEI_META_FIELDS, acc.snapshotAndReset(3)))
    assert b['nfrm'] == 0 and b['seq'] == 3
