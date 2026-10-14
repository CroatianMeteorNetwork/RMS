""" Replay a station's raw video through the capture-time compressor, block by block.

Capture (RMS.BufferedCapture.createGstreamDevice) splits the camera stream after h264parse: one
branch decodes frames for the compressor, the other muxes the undecoded H.264 into the raw-video
mkv segments. Decoding the mkv through the same chain (h264parse ! decoder ! videoconvert ! BGR,
green channel, as BufferedCapture fills the shared block buffer) gives back the frames the
compressor saw, and each frame's RMSP SEI record gives its camera frame_seq and integration
start (RMS.RMSPSei.readVideoFrameTimes).

A station FF carries SEQFIRST/SEQLAST/SEQLOST (frame_seq of its first/last frame and the frames
missing in between), so its block is the run of mkv frames with frame_seq in [SEQFIRST, SEQLAST].
The 'check' command rebuilds those blocks, runs RMS.CompressionCy.compressFrames on them with the
station's gamma and the night's camera response table, writes the FF with the capture code
(RMS.Compression.Compressor.saveFF) and compares it with the station's FF.

Two things stand between a replay and a byte-identical FF (US05E1, 2026-10-14):
- camera_response.json stores the decode table rounded to 9 decimals, while capture compressed
  with the camera's unrounded reply; the rounding flips AVERESID on ~80 px per block. Pass the
  camera's raw 'gamma decode' reply (--camera-reply) to decode exactly.
- compressFrames draws MAXFRAME of a pixel whose maximum is tied over several frames from a
  generator seeded with an uninitialised byte (randomN[0] of np.empty). One seed (240 on the
  station's compressor, constant over the blocks tested; varying in a replay process) reproduces
  the station's MAXFRAME exactly; 'check' verifies the tied pixels instead.

H.264 decoding is bit exact, so the CPU decoder (avdec_h264) gives the same frames as the
station's vaapih264dec (same caps and colorimetry, BGR identical on US05E1 2026-10-14) and the
replay does not compete with the live captures for the GPU decoder.
"""

from __future__ import print_function, division, absolute_import

import argparse
import datetime
import glob
import os
import shutil
import sys
import tempfile
import time
import types

import numpy as np

from astropy.io import fits

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import RMS.ConfigReader as cr
from RMS.Compression import Compressor
from RMS.CompressionCy import compressFrames
from RMS.Formats import FTfile
from RMS.Formats.FFfits import splitAvepixel16, splitStdpixel16
from RMS.RMSPSei import readVideoFrameTimes
from RMS.Routines.Response import ResponseCurve, useNightResponse


# Cards that capture derives from the SEI block statistics and the SoC temperature, not from the
# frames; the replay does not reconstruct them
SEI_CARDS = ('SOCTEMP', 'EXPTIME', 'EXPMIN', 'EXPMAX', 'AGAIN', 'DGAIN', 'ISPDGAIN', 'AGAINMIN',
             'AGAINMAX', 'DGAINMIN', 'DGAINMAX', 'ISPDGMIN', 'ISPDGMAX', 'GAINEXCT', 'SEISTABL',
             'QPMEAN', 'QPMAX', 'WBR', 'WBB', 'WBG', 'TROWNS', 'VMAX', 'SEQFIRST', 'SEQLAST',
             'SEQLOST', 'SEINFRM', 'TIMESRC', 'TIMEOFFS', 'TIMEITRP')


def decodeFrames(path, threads=2):
    """ Yield the frames of a video file as capture hands them to the compressor: decoded,
    converted to BGR by videoconvert and reduced to the green channel.

    Arguments:
        path: [str] Video file (mkv).

    Keyword arguments:
        threads: [int] Decoder threads.

    Return:
        Generator of [2D ndarray uint8] frames, in decode order.
    """

    Gst.init(None)
    pipe = Gst.parse_launch(
        'filesrc location="{:s}" ! matroskademux ! h264parse ! avdec_h264 max-threads={:d} ! '
        'videoconvert ! video/x-raw,format=BGR ! appsink name=sink sync=false max-buffers=4'
        .format(path, threads))
    sink = pipe.get_by_name('sink')
    pipe.set_state(Gst.State.PLAYING)

    try:
        while True:
            sample = sink.emit('pull-sample')
            if sample is None:
                break

            st = sample.get_caps().get_structure(0)
            w, h = st.get_value('width'), st.get_value('height')
            buf = sample.get_buffer()
            ok, info = buf.map(Gst.MapFlags.READ)
            if not ok:
                raise IOError("could not map a decoded frame of {:s}".format(path))
            try:
                if info.size != w*h*3:
                    raise IOError("unexpected decoded frame size {:d} for {:d}x{:d} BGR"
                                  .format(info.size, w, h))
                frame = np.ndarray((h, w, 3), dtype=np.uint8, buffer=info.data)[:, :, 1].copy()
            finally:
                buf.unmap(info)

            yield frame

        msg = pipe.get_bus().pop_filtered(Gst.MessageType.ERROR)
        if msg is not None:
            err, dbg = msg.parse_error()
            raise IOError("decoding {:s} failed: {:s}".format(path, str(err)))

    finally:
        pipe.set_state(Gst.State.NULL)


def indexVideo(paths, cache_dir):
    """ Per-frame SEI index of a list of video files, cached as one npz per file.

    Frames without an RMSP record get their frame_seq from the neighbours when the neighbours
    agree on it (consecutive seq for consecutive frames); otherwise they keep -1 and no block
    that needs them can be rebuilt.

    Arguments:
        paths: [list] Video files, in recording order.
        cache_dir: [str] Directory for the per-file index caches.

    Return:
        [dict] Concatenated per-frame arrays: file (np.intp index into paths), frame (np.intp
            index in its file), seq (int64), int_start (float64, unix s), exp_s (float64); and
            t_row_ns, a per-file list.
    """

    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    out = {'file': [], 'frame': [], 'seq': [], 'int_start': [], 'exp_s': []}
    t_row_ns = []
    for i, path in enumerate(paths):

        cache = os.path.join(cache_dir, os.path.basename(path) + '.sei.npz')
        if os.path.isfile(cache) and (os.path.getmtime(cache) >= os.path.getmtime(path)):
            c = np.load(cache)
            seq, int_start, exp_s, trow = c['seq'], c['int_start'], c['exp_s'], int(c['t_row_ns'])

        else:
            r = readVideoFrameTimes(path)
            if r is None:
                print("No SEI frame times in {:s}, skipped".format(path))
                t_row_ns.append(0)
                continue
            seq = fillSeq(r['frame_seq'])
            int_start, exp_s = r['int_start'], r['exp_s']
            trow = r['t_row_ns'] or 0
            np.savez(cache, seq=seq, int_start=int_start, exp_s=exp_s, t_row_ns=trow)

        n = len(seq)
        out['file'].append(np.full(n, i, dtype=np.intp))
        out['frame'].append(np.arange(n, dtype=np.intp))
        out['seq'].append(seq)
        out['int_start'].append(int_start)
        out['exp_s'].append(exp_s)
        t_row_ns.append(trow)

    idx = {k: np.concatenate(v) if v else np.empty(0) for k, v in out.items()}
    idx['t_row_ns'] = t_row_ns

    return idx


def fillSeq(seq):
    """ Fill frame_seq = -1 (frame without an RMSP record) from the nearest recorded neighbours
    when both imply the same value. """

    seq = np.asarray(seq, dtype=np.int64).copy()
    good = np.flatnonzero(seq >= 0)
    for k in np.flatnonzero(seq < 0):
        j = np.searchsorted(good, k)
        if (j == 0) or (j == len(good)):
            continue
        a, b = good[j - 1], good[j]
        if seq[b] - seq[a] == b - a:
            seq[k] = seq[a] + (k - a)

    return seq


def ffBlocks(ff_paths):
    """ The capture blocks of a list of FF files, from their SEQFIRST/SEQLAST/SEQLOST cards.

    Return:
        [list] of dicts (path, seq_first, seq_last, seq_lost, start_time, n), sorted by seq_first.
            FFs without the cards are left out.
    """

    blocks = []
    for path in ff_paths:
        h = fits.getheader(path, 0)
        if 'SEQFIRST' not in h:
            continue
        t = datetime.datetime.strptime(h['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f')
        t = t.replace(tzinfo=datetime.timezone.utc).timestamp()
        name = os.path.basename(path)
        blocks.append({'path': path, 'seq_first': int(h['SEQFIRST']), 'seq_last': int(h['SEQLAST']),
                       'seq_lost': int(h['SEQLOST']), 'start_time': t,
                       'n': int(name.split('_')[-1].split('.')[0])})

    blocks.sort(key=lambda b: b['seq_first'])

    return blocks


def iterBlocks(paths, index, blocks, threads=2, frames_buf=None):
    """ Rebuild the frame blocks of the given FFs from the video, in seq order.

    Only the files that hold frames of a wanted block are decoded. A block is complete when
    exactly 256 video frames have a frame_seq in [seq_first, seq_last]; otherwise it is reported
    with frames=None (the compressor saw a set of frames the video cannot reproduce).

    Arguments:
        paths: [list] Video files, in recording order (as given to indexVideo).
        index: [dict] indexVideo of paths.
        blocks: [list] ffBlocks.

    Keyword arguments:
        threads: [int] Decoder threads.
        frames_buf: [3D ndarray] Reusable (256, H, W) uint8 buffer, None to allocate one.

    Return:
        Generator of (block, frames, sel): frames is the (256, H, W) buffer (overwritten by the
            next block) or None, sel the indices into index of its frames.
    """

    if not blocks:
        return

    seq = index['seq']
    lo, hi = blocks[0]['seq_first'], blocks[-1]['seq_last']
    want = (seq >= lo) & (seq <= hi)
    files = np.unique(index['file'][want])

    b = 0
    pos = 0
    sel = []
    for fi in files:
        rows = np.flatnonzero(index['file'] == fi)
        k = -1
        for k, frame in enumerate(decodeFrames(paths[fi], threads=threads)):

            if k >= len(rows):
                raise IOError("{:s}: more decoded frames than SEI-indexed pictures".format(paths[fi]))
            r = rows[k]
            s = seq[r]

            # Close every block this frame has moved past
            while (b < len(blocks)) and (s > blocks[b]['seq_last']):
                if pos:
                    yield blocks[b], None, np.array(sel, dtype=np.intp)
                b += 1
                pos = 0
                sel = []

            if b == len(blocks):
                return

            if s < blocks[b]['seq_first']:
                continue

            if pos < 256:
                if frames_buf is None:
                    frames_buf = np.zeros((256,) + frame.shape, dtype=np.uint8)
                frames_buf[pos] = frame
            sel.append(r)
            pos += 1

            if s == blocks[b]['seq_last']:
                yield blocks[b], (frames_buf if pos == 256 else None), np.array(sel, dtype=np.intp)
                b += 1
                pos = 0
                sel = []

        if k + 1 != len(rows):
            raise IOError("{:s}: decoded {:d} frames, SEI index has {:d}".format(paths[fi], k + 1,
                                                                                 len(rows)))

    if pos and (b < len(blocks)):
        yield blocks[b], None, np.array(sel, dtype=np.intp)


def compareFF(ff_station, ff_replay):
    """ Compare a replayed FF with the station's: the image HDUs byte by byte (header and data
    as stored, compressed when the FF is), their pixels, and the primary header cards.

    Return:
        [dict] hdus_identical (bool), plane_mismatch {extname: n differing pixels},
            header_diff [list of (card, station, replay)] (SEI-derived cards excluded).
    """

    res = {'hdus_identical': True, 'plane_mismatch': {}, 'header_diff': []}

    with open(ff_station, 'rb') as f:
        raw_s = f.read()
    with open(ff_replay, 'rb') as f:
        raw_r = f.read()

    with fits.open(ff_station) as hs, fits.open(ff_replay) as hr:

        names_s = [h.name for h in hs]
        names_r = [h.name for h in hr]
        if names_s != names_r:
            res['hdus_identical'] = False
            res['header_diff'].append(('HDUS', names_s, names_r))
            return res

        for i in range(1, len(hs)):
            a, b = hs.fileinfo(i), hr.fileinfo(i)
            seg_s = raw_s[a['hdrLoc']:a['datLoc'] + a['datSpan']]
            seg_r = raw_r[b['hdrLoc']:b['datLoc'] + b['datSpan']]
            if seg_s != seg_r:
                res['hdus_identical'] = False
            res['plane_mismatch'][hs[i].name] = int(np.count_nonzero(hs[i].data != hr[i].data))

        h_s, h_r = hs[0].header, hr[0].header
        for key in sorted(set(h_s.keys()) | set(h_r.keys())):
            if key in SEI_CARDS:
                continue
            if h_s.get(key) != h_r.get(key):
                res['header_diff'].append((key, h_s.get(key), h_r.get(key)))

    return res


def findFT(ft_root, start_time, station_id):
    """ The FT file capture wrote for the block starting at start_time, or None. """

    t = datetime.datetime.fromtimestamp(start_time, tz=datetime.timezone.utc)
    name = t.strftime("FT_{:s}_%Y%m%d_%H%M%S.bin".format(station_id))
    hits = glob.glob(os.path.join(ft_root, '**', name), recursive=True)

    return hits[0] if hits else None


def check(args):
    """ Rebuild the station's FFs from the video and compare them (see the module docstring). """

    config = cr.parse(args.config)
    useNightResponse(config, args.night_dir)
    if args.camera_reply:
        with open(args.camera_reply) as f:
            config.response = ResponseCurve.fromCameraReply(f.read(), gamma=config.gamma)
    print("Station {:s}: gamma {:.3f}, response {:s}, deinterlace_order {:d}".format(
        config.stationID, config.gamma,
        config.response.ident() if config.response is not None else 'power law',
        config.deinterlace_order))

    paths = sorted(p for d in args.video for p in (glob.glob(os.path.join(d, '*.mkv'))
                                                    if os.path.isdir(d) else [d]))
    t0 = time.time()
    index = indexVideo(paths, os.path.join(args.work, 'sei_index'))
    print("Indexed {:d} files, {:d} frames, seq {:d}..{:d} ({:.1f} s)".format(
        len(paths), len(index['seq']), int(index['seq'].min()), int(index['seq'].max()),
        time.time() - t0))

    ff_paths = sorted(glob.glob(os.path.join(args.ff_dir, 'FF_{:s}_*.fits'.format(config.stationID))))
    blocks = [b for b in ffBlocks(ff_paths)
              if (b['seq_first'] >= index['seq'].min()) and (b['seq_last'] <= index['seq'].max())]
    if args.first:
        blocks = [b for b in blocks if os.path.basename(b['path']) >= args.first]
    blocks = blocks[:args.max_blocks] if args.max_blocks else blocks
    print("{:d} FF blocks inside the video".format(len(blocks)))

    tmp_dir = tempfile.mkdtemp(prefix='ffreplay_', dir=args.work)
    stub = types.SimpleNamespace(config=config, data_dir=tmp_dir)

    n_ok = 0
    n_ties = 0
    n_bad = 0
    n_skip = 0
    try:
        for blk, frames, sel in iterBlocks(paths, index, blocks, threads=args.threads):

            name = os.path.basename(blk['path'])
            if frames is None:
                n_skip += 1
                print("{:s}  seq {:d}..{:d} lost {:d}: {:d} video frames in range, not 256 -> "
                      "not reproducible".format(name, blk['seq_first'], blk['seq_last'],
                                                blk['seq_lost'], len(sel)))
                continue

            t1 = time.time()
            ftp, ave16, std16, _ = compressFrames(frames, config.deinterlace_order, config.gamma,
                                                  response=config.response)
            t_comp = time.time() - t1

            h = fits.getheader(blk['path'], 0)
            soc = h.get('SOCTEMP')
            fn_ms, _ = Compressor.saveFF(stub, ftp[:, :config.height, :config.width], blk['start_time'],
                                         blk['n'], ave16=ave16[:config.height, :config.width],
                                         std16=std16[:config.height, :config.width], soc_temp=soc)
            replay = os.path.join(tmp_dir, 'FF_' + fn_ms + '.' + config.ff_format)
            if not os.path.isfile(replay):
                raise IOError("replay FF {:s} not written (name mismatch?)".format(replay))

            res = compareFF(blk['path'], replay)

            # MAXFRAME of a pixel whose maximum is reached in several frames is drawn at random by
            # compressFrames, from a generator seeded with an uninitialised byte (randomN[0] of
            # np.empty), so only the frames with a unique maximum are reproducible. A tied pixel is
            # consistent when the station's frame also holds the maximum
            if res['plane_mismatch'].get('MAXFRAME'):
                with fits.open(blk['path']) as hs:
                    mf_s = hs['MAXFRAME'].data
                yy, xx = np.nonzero(mf_s != ftp[1])
                ties = np.count_nonzero(frames == ftp[0][None], axis=0)
                res['maxframe_unique_max_mismatch'] = int(np.count_nonzero(ties[yy, xx] == 1))
                res['maxframe_station_not_a_max'] = int(np.count_nonzero(
                    frames[mf_s[yy, xx].astype(np.intp), yy, xx] != ftp[0][yy, xx]))

            # Frame times: replayed SEI integration start vs the FF start and the FT file
            t_rep = index['int_start'][sel]
            dt_ff_us = (t_rep[0] - blk['start_time'])*1e6
            ft_path = findFT(args.ft_dir, blk['start_time'], config.stationID) if args.ft_dir else None
            dt_ft = ''
            if ft_path is not None:
                ft = FTfile.read(os.path.dirname(ft_path), os.path.basename(ft_path))
                t_ft = np.array([t for _, t in ft.timestamps])
                if len(t_ft) == len(t_rep):
                    d = (t_rep - t_ft)*1e6
                    dt_ft = "  FT-replay {:+.1f}..{:+.1f} us".format(-d.max(), -d.min())

            ok = res['hdus_identical'] and not res['header_diff']
            ties_only = (not ok) and (not res['header_diff']) \
                and all(v == 0 for k, v in res['plane_mismatch'].items() if k != 'MAXFRAME') \
                and (res.get('maxframe_unique_max_mismatch', 1) == 0) \
                and (res.get('maxframe_station_not_a_max', 1) == 0)
            if ok:
                verdict = 'IDENTICAL'
            elif ties_only:
                verdict = 'IDENTICAL except MAXFRAME of {:d} tied px'.format(
                    res['plane_mismatch']['MAXFRAME'])
            else:
                verdict = 'DIFFERS {}'.format(res)
            n_ok += ok
            n_ties += ties_only
            n_bad += not (ok or ties_only)
            print("{:s}  seq {:d}..{:d}  {:s}  compress {:.2f} s  t0 replay-FF {:+.1f} us{:s}".format(
                name, blk['seq_first'], blk['seq_last'], verdict, t_comp, dt_ff_us, dt_ft))

            os.remove(replay)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("identical {:d}, identical except MAXFRAME ties {:d}, different {:d}, not reproducible {:d}"
          .format(n_ok, n_ties, n_bad, n_skip))

    return n_bad == 0


def referencePhotometry(frames, lut, x, y):
    """ Brute-force reference of RMS.StarLightCurves photometry: each full frame decoded in float64,
    each star's 5x5 px box integrated over the pixels it overlaps (overlap computed per pixel),
    the ring median at the 8 PatchPhotometry offsets around the half-up rounded position.

    Return:
        (flux, bg): [2D ndarray float64] (N, S).
    """

    from RMS.PatchPhotometry import APERTURE_HALF, RING_OFFSETS

    lut64 = np.asarray(lut, dtype=np.float64)
    half = APERTURE_HALF + 0.5
    n_frames, n_stars = x.shape
    flux = np.empty((n_frames, n_stars))
    bg = np.empty((n_frames, n_stars))

    def overlap(c, lo, hi):
        p = np.arange(lo, hi + 1)
        return np.clip(np.minimum(p + 0.5, c + half) - np.maximum(p - 0.5, c - half), 0, None)

    for n in range(n_frames):
        img = lut64[frames[n]]
        for s in range(n_stars):
            xc, yc = x[n, s], y[n, s]
            i_lo, i_hi = int(np.floor(xc - half)), int(np.ceil(xc + half))
            j_lo, j_hi = int(np.floor(yc - half)), int(np.ceil(yc + half))
            w = np.outer(overlap(yc, j_lo, j_hi), overlap(xc, i_lo, i_hi))
            flux[n, s] = np.sum(w*img[j_lo:j_hi + 1, i_lo:i_hi + 1])

            xi, yi = int(np.floor(xc + 0.5)), int(np.floor(yc + 0.5))
            bg[n, s] = np.median([img[yi + oy, xi + ox] for oy, ox in RING_OFFSETS])

    return flux, bg


def run(args):
    """ Replay blocks through the compressor and the light-curve pass, one output file per block. """

    from RMS.Formats.Platepar import Platepar
    from RMS.StarLightCurves import SCHEMA_VERSION, StarLightCurves, saveBlock

    config = cr.parse(args.config)
    useNightResponse(config, args.night_dir)
    if args.camera_reply:
        with open(args.camera_reply) as f:
            config.response = ResponseCurve.fromCameraReply(f.read(), gamma=config.gamma)

    pp = Platepar()
    pp.read(args.platepar, use_flat=config.use_flat)

    lim_mag = args.lim_mag if args.lim_mag else config.catalog_mag_limit
    lc = StarLightCurves(config, pp, lim_mag)
    print("Catalog {:s}, lim mag {:.2f}: {:d} stars".format(lc.field.catalog_name, lim_mag,
                                                             len(lc.field.oid)))

    paths = sorted(p for d in args.video for p in (glob.glob(os.path.join(d, '*.mkv'))
                                                    if os.path.isdir(d) else [d]))
    index = indexVideo(paths, os.path.join(args.work, 'sei_index'))
    ff_paths = sorted(glob.glob(os.path.join(args.ff_dir, 'FF_{:s}_*.fits'.format(config.stationID))))
    blocks = [b for b in ffBlocks(ff_paths)
              if (b['seq_first'] >= index['seq'].min()) and (b['seq_last'] <= index['seq'].max())]
    if args.first:
        blocks = [b for b in blocks if os.path.basename(b['path']) >= args.first]
    blocks = blocks[:args.max_blocks] if args.max_blocks else blocks
    print("{:d} FF blocks inside the video".format(len(blocks)))

    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    t_lc = []
    for nb, (blk, frames, sel) in enumerate(iterBlocks(paths, index, blocks, threads=args.threads)):

        name = os.path.basename(blk['path'])
        if frames is None:
            print("{:s}: not reproducible from the video, skipped".format(name))
            continue

        ftp, ave16, std16, _ = compressFrames(frames, config.deinterlace_order, config.gamma,
                                              response=config.response)
        with fits.open(blk['path']) as hs:
            same = all(np.array_equal(hs[k].data, v) for k, v in
                       (('MAXPIXEL', ftp[0]), ('AVEPIXEL', ftp[2]), ('STDPIXEL', ftp[3])))

        t_frames = index['int_start'][sel]
        t1 = time.time()
        res = lc.measureBlock(frames, ave16, t_frames)
        dt = time.time() - t1
        t_lc.append(dt)

        header = {'schema_version': SCHEMA_VERSION, 'stationID': config.stationID, 'ff': name,
                  'catalog': lc.field.catalog_name, 'lim_mag': lim_mag,
                  'platepar': os.path.basename(args.platepar), 'response': lc.response_id,
                  'gamma': config.gamma, 'aperture': '5x5 box at the fractional position',
                  'ring': 'median of 8 px at +/-7 px', 'offset': res['offset'],
                  't_row_ns': int(index['t_row_ns'][int(index['file'][sel[0]])]),
                  'exp_s': float(np.nanmedian(index['exp_s'][sel])),
                  'seq_first': blk['seq_first'], 'seq_last': blk['seq_last']}
        out = os.path.join(args.out, 'LC_' + name.replace('FF_', '').replace('.fits', '.npz'))
        saveBlock(out, res, t_frames, index['seq'][sel], header)

        ref = ''
        if args.reference_every and (nb % args.reference_every == 0):
            t2 = time.time()
            # The stored first/last positions and the frame times define every frame's position;
            # the reference measures there, independently, and is compared with the stored curves
            n_ref = min(args.reference_frames, len(sel))
            frac = (t_frames - t_frames[0])/(t_frames[-1] - t_frames[0])
            x = res['x_first'].astype(np.float64)[None, :] \
                + frac[:, None]*(res['x_last'].astype(np.float64) - res['x_first'])[None, :]
            y = res['y_first'].astype(np.float64)[None, :] \
                + frac[:, None]*(res['y_last'].astype(np.float64) - res['y_first'])[None, :]
            rf, rb = referencePhotometry(frames[:n_ref], lc.lut, x[:n_ref], y[:n_ref])
            df = np.abs(res['flux'][:n_ref] - rf)/np.maximum(np.abs(rf), 1.0)
            db = np.abs(res['bg'][:n_ref] - rb)
            ref = "  reference ({:d} frames, {:.0f} s): max rel dflux {:.2e}, max dbg {:.2e}".format(
                n_ref, time.time() - t2, df.max(), db.max())

        o = res['offset']
        print("{:s}  stars {:d}  offset ({:+.2f}, {:+.2f}) px from {:d}/{:d} rms {:.2f}  "
              "pass {:.3f} s  FF planes {:s}{:s}".format(name, len(res['oid']), o['dx'], o['dy'],
              o['n_used'], o['n_cand'], o['rms'], dt, 'same' if same else 'DIFFER', ref), flush=True)

    if t_lc:
        print("pass time per block: median {:.3f} s, max {:.3f} s over {:d} blocks".format(
            float(np.median(t_lc)), float(np.max(t_lc)), len(t_lc)))

    return True


def main(argv=None):

    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd')
    sub.required = True

    c = sub.add_parser('check', help="rebuild the station's FFs from the video and compare them")
    c.add_argument('--config', required=True, help="station .config")
    c.add_argument('--night-dir', required=True, help="night directory (camera_response.json)")
    c.add_argument('--camera-reply', help="the camera's raw 'gamma decode' reply, used instead of "
                   "camera_response.json (which stores the table rounded to 9 decimals)")
    c.add_argument('--ff-dir', required=True, help="directory of the station's FF files")
    c.add_argument('--ft-dir', help="TimeFiles directory (FT files), to compare frame times")
    c.add_argument('--video', required=True, nargs='+', help="mkv files or directories")
    c.add_argument('--work', required=True, help="work directory (SEI index cache)")
    c.add_argument('--first', help="first FF file name to check")
    c.add_argument('--max-blocks', type=int, default=0)
    c.add_argument('--threads', type=int, default=2, help="decoder threads")
    c.set_defaults(func=check)

    r = sub.add_parser('run', help="replay blocks through the light-curve pass")
    r.add_argument('--config', required=True, help="station .config")
    r.add_argument('--night-dir', required=True, help="night directory (camera_response.json)")
    r.add_argument('--camera-reply', help="the camera's raw 'gamma decode' reply")
    r.add_argument('--platepar', required=True, help="station platepar")
    r.add_argument('--lim-mag', type=float, help="catalog limiting magnitude (default: "
                   "catalog_mag_limit of the config)")
    r.add_argument('--ff-dir', required=True, help="directory of the station's FF files")
    r.add_argument('--video', required=True, nargs='+', help="mkv files or directories")
    r.add_argument('--work', required=True, help="work directory (SEI index cache)")
    r.add_argument('--out', required=True, help="output directory (one file per block)")
    r.add_argument('--first', help="first FF file name")
    r.add_argument('--max-blocks', type=int, default=0)
    r.add_argument('--threads', type=int, default=2, help="decoder threads")
    r.add_argument('--reference-every', type=int, default=0,
                   help="check the pass against the brute-force reference every N blocks")
    r.add_argument('--reference-frames', type=int, default=16,
                   help="frames of a block checked against the reference")
    r.set_defaults(func=run)

    args = ap.parse_args(argv)

    return 0 if args.func(args) else 1


if __name__ == '__main__':
    sys.exit(main())
