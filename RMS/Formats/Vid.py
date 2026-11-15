""" Loading and handling *.vid files. """

from __future__ import print_function, division, absolute_import

import os
import numpy as np


class VidStruct(object):
    """ Structure for storing vid file info. """

    def __init__(self):

        # List of video frames - this is used only in the parent Vid structure, while the child structures
        # have this empty, but they contain image data in the img_data variable
        self.frames = None

        self.magic = 0

        # Bytes for a single image
        self.seqlen = 0

        # Header length in bytes
        self.headlen = 0

        self.flags = 0
        self.seq = 0

        # UNIX time
        self.ts = 0
        self.tu = 0

        # Station number
        self.station_id = 0

        # Image dimensions in pixels
        self.wid = 0
        self.ht = 0

        # Image depth in bits
        self.depth = 0

        # Mirror pointing for centre of frame
        self.hx = 0
        self.hy = 0

        # Stream number
        self.str_num = 0
        self.reserved0 = 0

        # Exposure time in milliseconds
        self.exposure = 0

        self.reserved2 = 0

        self.text = 0

        # Image data
        self.img_data = None




def readFrame(st, fid, metadata_only=False):
    """ Read in the information from the next frame, save them to the given structure and return the image 
        data.

    Arguments:
        st: [Vid structure]
        fid: [file handle] File handle to the vid file. Make sure it was open in the 'rb' mode.

    Keyword arguments:
        metadata_only: [bool] Only read the metadata, but not the whole frame. False by default
    """

    # Get the current position in the file
    file_pos = fid.tell()

    # Check if the end of file (EOF) is reached
    if not fid.read(1):
        return None

    fid.seek(file_pos)


    #### Read the header ###
    ##########################################################################################################

    st.magic = int(np.fromfile(fid, dtype=np.uint32, count=1)[0])

    # Size of one frame in bytes
    st.seqlen = int(np.fromfile(fid, dtype=np.uint32, count=1)[0])

    # Header length in bytes
    st.headlen = int(np.fromfile(fid, dtype=np.uint32, count=1)[0])

    st.flags = int(np.fromfile(fid, dtype=np.uint32, count=1)[0])
    st.seq = int(np.fromfile(fid, dtype=np.uint32, count=1)[0])

    # Beginning UNIX time
    st.ts = int(np.fromfile(fid, dtype=np.int32, count=1)[0])
    st.tu = int(np.fromfile(fid, dtype=np.int32, count=1)[0])

    # Station number
    st.station_id = int(np.fromfile(fid, dtype=np.int16, count=1)[0])

    # Image dimensions
    st.wid = int(np.fromfile(fid, dtype=np.int16, count=1)[0])
    st.ht = int(np.fromfile(fid, dtype=np.int16, count=1)[0])

    # Image depth
    st.depth = int(np.fromfile(fid, dtype=np.int16, count=1)[0])

    st.hx = int(np.fromfile(fid, dtype=np.uint16, count=1)[0])
    st.hy = int(np.fromfile(fid, dtype=np.uint16, count=1)[0])

    # Camera stream identifier, where 0 = 'A', 1 = 'B', etc
    st.str_num = int(np.fromfile(fid, dtype=np.uint16, count=1)[0])

    st.reserved0 = int(np.fromfile(fid, dtype=np.uint16, count=1)[0])
    st.exposure = int(np.fromfile(fid, dtype=np.uint32, count=1)[0])
    st.reserved2 = int(np.fromfile(fid, dtype=np.uint32, count=1)[0])
    
    st.text = np.fromfile(fid, dtype=np.uint8, count=64).tobytes().decode("ascii").replace('\0', '')

    ##########################################################################################################

    if not metadata_only:

        # Rewind the file to the beginning of the frame
        fid.seek(file_pos)

        # Read one whole frame
        fr = np.fromfile(fid, dtype=np.uint16, count=st.seqlen//2)

        # Set the values of the first row to 0
        fr[:st.ht] = 0

        # Reshape the frame to the proper image size
        img = fr.reshape(st.ht, st.wid)

    else:

        # If only the metadata was read, set the image to None
        img = None


    return img




# The longest run of consecutive frames with the seconds field off which is taken as such (the longest seen
#   in the recordings is 75 frames, most are under 12). A longer run is a step in the timestamps
MAX_OFF_RUN_FRAMES = 128

# Finest spacing (frames) of the headers read between sampled frames off the model when the samples alone
#   can't give it (see denseTimeModel). A run of frames with the seconds field off is then found to within
#   this many frames on each side, so every run up to 98 frames (MAX_OFF_RUN_FRAMES + 2 - 2*DENSE_STEP_FRAMES)
#   is taken as such, and the frames on the model between the runs of a burst (over 20) are found
DENSE_STEP_FRAMES = 16

# Most headers read besides the sampled frames. The header of a vid frame is a small read at a known position
#   in the file
MAX_DENSE_HEADERS = 1024


def frameTimeModel(fid, n_samples=64, tolerance=0.1, max_run=3, max_run_frames=MAX_OFF_RUN_FRAMES,
                   step_frames=DENSE_STEP_FRAMES, max_dense=MAX_DENSE_HEADERS):
    """ A model of the frame timestamps of a vid file from its frame counter. The timestamps are written as the
        time of the first frame plus the counter times the frame period, so the time of any frame follows from
        its counter. A few frames can have a seconds field off by whole seconds (the microseconds stay right),
        which the model shows (see repairFrameTime). The model is built from the headers of frames spread over
        the file (see sampleTimeModel), and the position in the file is kept.

    Arguments:
        fid: [file handle] File handle to the vid file, open in the 'rb' mode.

    Keyword arguments:
        n_samples: [int] Number of frames sampled. 64 by default.
        tolerance: [float] Largest difference (s) between a sampled timestamp and the model, apart from whole
            seconds. 0.1 by default (a few lost frames).
        max_run: [int] Most consecutive sampled frames off the model by whole seconds. Frames with the seconds
            field off come in short runs, far shorter than the spacing of the samples in a long file, so more
            consecutive samples off the model are a step in the timestamps. 3 by default.
        max_run_frames: [int] Longest run of frames with the seconds field off. In a short file the samples
            are dense and such a run covers more than max_run of them, so the number of consecutive samples
            allowed off the model is raised to the most which a run of this many frames can cover.
            MAX_OFF_RUN_FRAMES by default.
        step_frames: [int] Finest spacing (frames) of the headers read between sampled frames off the model
            when the samples alone can't give it. DENSE_STEP_FRAMES by default.
        max_dense: [int] Most headers read besides the sampled frames. MAX_DENSE_HEADERS by default.

    Return:
        [tuple] (counter, seconds, offset, period, counters, residuals) of the model (see fitTimeModel): the time
            of a frame with the counter c is seconds + offset + period*(c - counter), plus the residual of a sampled
            frame next to it. None if the counter doesn't give the timestamps of the sampled frames.
    """

    file_pos = fid.tell()

    try:

        # The number of frames, from the size of a frame in the header of the first one
        fid.seek(0, os.SEEK_END)
        file_size = fid.tell()
        fid.seek(0)
        header = np.fromfile(fid, dtype=np.uint32, count=7)
        if len(header) < 7 or header[1] == 0:
            return None
        seqlen = int(header[1])
        n_frames = file_size//seqlen
        if n_frames < 8:
            return None

        # The counter and the timestamp in the header of a frame, at the start of the frame
        def readHeader(k):
            fid.seek(k*seqlen)
            words = np.fromfile(fid, dtype=np.uint32, count=7)
            return int(words[4]), int(np.int32(words[5])), int(np.int32(words[6]))

        return sampleTimeModel(n_frames, readHeader, n_samples=n_samples, tolerance=tolerance,
                               max_run=max_run, max_run_frames=max_run_frames, step_frames=step_frames,
                               max_dense=max_dense)

    finally:
        fid.seek(file_pos)


def sampleTimeModel(n_frames, readHeader, n_samples=64, tolerance=0.1, max_run=3,
                    max_run_frames=MAX_OFF_RUN_FRAMES, step_frames=DENSE_STEP_FRAMES,
                    max_dense=MAX_DENSE_HEADERS):
    """ The model of the frame timestamps of an input from the headers of its frames (see fitTimeModel): from
        frames spread over the input, and when those can't give it, also from frames between the sampled
        frames off it (see denseTimeModel). Shared by the vid files and the UWO PNG frame dumps, whose headers
        carry the same fields.

    Arguments:
        n_frames: [int] Number of frames of the input.
        readHeader: [function] Gives the header of the frame with the given index in the input (0 for the
            first) as a tuple (counter, seconds, microseconds), or None if the frame has no such header.

    Keyword arguments:
        n_samples: [int] Number of frames sampled. 64 by default.
        tolerance: [float] Largest difference (s) between a timestamp and the model, apart from whole seconds.
            0.1 by default (a few lost frames).
        max_run: [int] Most consecutive sampled frames off the model by whole seconds. 3 by default.
        max_run_frames: [int] Longest run of frames with the seconds field off. MAX_OFF_RUN_FRAMES by default.
        step_frames: [int] Finest spacing (frames) of the headers read between sampled frames off the model.
            DENSE_STEP_FRAMES by default.
        max_dense: [int] Most headers read besides the sampled frames. MAX_DENSE_HEADERS by default.

    Return:
        [tuple] The model (see fitTimeModel), or None if the counter doesn't give the timestamps.
    """

    # The headers of frames spread over the input
    pos, seqs, secs, usecs = [], [], [], []
    for k in np.unique(np.linspace(0, n_frames - 1, min(n_samples, n_frames)).astype(np.int64)):
        header = readHeader(int(k))
        if header is None:
            continue
        pos.append(int(k))
        seqs.append(header[0])
        secs.append(header[1])
        usecs.append(header[2])

    # The model from the sampled frames alone, which takes a few consecutive samples off it as frames with the
    #   seconds field off
    model = fitTimeModel(seqs, secs, usecs, tolerance=tolerance, max_run=max_run,
                         max_run_frames=max_run_frames)
    if model is not None:
        return model

    # Dense bursts of runs of frames with the seconds field off leave more consecutive samples off the model
    #   than allowed (or most of the samples off it), so the headers between them are read
    return denseTimeModel(readHeader, pos, seqs, secs, usecs, tolerance=tolerance,
                          max_run_frames=max_run_frames, step_frames=step_frames, max_dense=max_dense)


# A timestamp with its seconds field off reads exactly whole seconds early: the stamps are written to the
#   microsecond from the counter, and the model gives them to a fraction of a millisecond, while a real gap in the
#   timestamps (lost frames) is a multiple of the frame period, which is not a whole number of seconds to within
#   this tolerance (32 frames of 31.119 ms are 4.2 ms short of a second)
WHOLE_SECOND_TOLERANCE = 0.002


def wholeSecondsEarly(seq, res, seq_on, res_on, whole_tolerance=WHOLE_SECOND_TOLERANCE):
    """ The whole seconds by which a timestamp reads early against the samples on the model next to it.

    A few lost frames make steps of a frame period in the residuals of the line, between two samples, so the
    residual of the frame is compared with the residual of the sample on each side of it (not interpolated
    across the step): the seconds field is off when the frame reads exactly whole seconds early against either.

    Arguments:
        seq: [int] Frame counter.
        res: [float] Residual of the timestamp from the line of the model (s).
        seq_on: [ndarray] Counters of the sampled frames on the model (increasing).
        res_on: [ndarray] Their residuals.

    Keyword arguments:
        whole_tolerance: [float] Largest difference (s) from exactly whole seconds early. WHOLE_SECOND_TOLERANCE
            by default.

    Return:
        [int] The whole seconds (negative) the timestamp reads early, 0 if it is not off.
    """

    k = int(np.searchsorted(seq_on, seq))
    for j in (k - 1, k):
        if 0 <= j < len(seq_on):
            d = res - res_on[j]
            n = int(round(d))
            if (n < 0) and (abs(d - n) < whole_tolerance):
                return n

    return 0


def fitTimeModel(seqs, secs, usecs, tolerance=0.1, max_run=3, whole_tolerance=WHOLE_SECOND_TOLERANCE,
                 max_run_frames=MAX_OFF_RUN_FRAMES):
    """ The model of the frame timestamps from the frame counter, fitted to the headers of sampled frames (see
        frameTimeModel). Shared by the vid files and the UWO PNG frame dumps, whose headers carry the same fields.

    Arguments:
        seqs: [list] Frame counters of the sampled frames, in the order of the frames.
        secs: [list] Seconds fields of their timestamps.
        usecs: [list] Microseconds fields of their timestamps.

    Keyword arguments:
        tolerance: [float] Largest difference (s) between a sampled timestamp and the model for the sample to be on
            the model (a few lost frames). 0.1 by default.
        max_run: [int] Most consecutive sampled frames off the model by whole seconds. 3 by default.
        whole_tolerance: [float] Largest difference (s) from exactly whole seconds early of a sample off the model.
            WHOLE_SECOND_TOLERANCE by default.
        max_run_frames: [int] Longest run of frames with the seconds field off. Dense samples (a short input)
            may have more than max_run consecutive samples off the model within a run this long, which is then
            allowed. MAX_OFF_RUN_FRAMES by default.

    Return:
        [tuple] (counter, seconds, offset, period, counters, residuals) of the model: the line of frameTimeModel
            and the residuals of the samples on it (see wholeSecondsEarly); or None.
    """

    if len(seqs) < 8:
        return None

    # The counter has to increase from frame to frame
    seqs = np.array(seqs, dtype=np.int64)
    if np.any(np.diff(seqs) <= 0):
        return None

    # The period from the median step between the sampled frames (a sample with a seconds field off changes
    #   only its two steps), then the offset of the timestamps from the median
    sec0, seq0 = int(secs[0]), int(seqs[0])
    t = (np.array(secs, dtype=np.int64) - sec0) + np.array(usecs)/1e6
    period = float(np.median(np.diff(t)/np.diff(seqs)))
    if not (period > 0):
        return None
    offset = float(np.median(t - period*(seqs - seq0)))

    # Most sampled frames have to be on the model
    on = np.abs(t - offset - period*(seqs - seq0)) < tolerance
    if np.sum(on) < 0.5*len(t):
        return None

    # The line refined on the samples on the model, which every sample off it has to read whole seconds early
    #   against. A long series of samples off the model is a step too
    fit = refineTimeLine(seqs, t, on, seq0, period, offset, whole_tolerance=whole_tolerance)
    if fit is None:
        return None
    period, offset, res = fit
    off = ~on

    # A run of frames with the seconds field off covers more samples when they are dense (a short input), so
    #   the number of consecutive samples allowed off the model is at least the most which a run of
    #   max_run_frames frames can cover, at the smallest spacing of the samples. Sparse samples (a long input)
    #   keep max_run
    max_run = max(max_run, (max_run_frames - 1)//int(np.min(np.diff(seqs))) + 1)
    run = 0
    for o in off:
        run = run + 1 if o else 0
        if run > max_run:
            return None

    return seq0, sec0, float(offset), float(period), seqs[on].astype(np.float64), res[on]


def refineTimeLine(seqs, t, on, seq0, period, offset, whole_tolerance=WHOLE_SECOND_TOLERANCE):
    """ The line of the timestamps against the frame counter refined by a fit to the headers on it, and the
        residuals of all headers from it. Every header off the line has to read exactly whole seconds early
        (the seconds field low) against a header on it next to it: a header off by anything else, or late, is
        a real gap or step in the timestamps (e.g. many lost frames near an end of the input, which leave few
        samples after them), and the timestamps are then kept as they are.

    Arguments:
        seqs: [ndarray] Frame counters of the headers (increasing).
        t: [ndarray] Their timestamps (s), from the seconds of the first header of the model.
        on: [ndarray] Whether each header is on the line (bool).
        seq0: [int] Counter of the first header of the model.
        period: [float] Frame period (s) of the line.
        offset: [float] Timestamp (s) of the line at the counter seq0.

    Keyword arguments:
        whole_tolerance: [float] Largest difference (s) from exactly whole seconds early of a header off the
            line. WHOLE_SECOND_TOLERANCE by default.

    Return:
        [tuple] (period, offset, residuals) of the refined line, or None if a header off it is not whole
            seconds early.
    """

    # The period and the offset refined by a line fit to the headers on the model
    if np.sum(on) >= 3:
        period, offset = np.polyfit(seqs[on] - seq0, t[on], 1)

    # The residuals of the headers: a few lost frames make steps of a frame period in the timestamps, which
    #   the line leaves in its residuals
    res = t - offset - period*(seqs - seq0)
    seq_on, res_on = seqs[on], res[on]

    # Every header off the model whole seconds early against one on the model next to it
    for i in np.nonzero(~on)[0]:
        if wholeSecondsEarly(seqs[i], res[i], seq_on, res_on, whole_tolerance) == 0:
            return None

    return period, offset, res


def wholeSecondLine(seqs, t, tolerance=0.1):
    """ The line of the timestamps against the frame counter, taken modulo whole seconds so that it is not
        thrown off by frames with the seconds field off, even when they are many of the samples (dense bursts
        of runs).

    Arguments:
        seqs: [ndarray] Frame counters of the sampled frames, from the first one (increasing).
        t: [ndarray] Their timestamps (s), from the seconds of the first one.

    Keyword arguments:
        tolerance: [float] Largest difference (s) between a timestamp and the line, apart from whole seconds.
            0.1 by default.

    Return:
        [tuple] (period, offset) of the line t = offset + period*seqs, or None if there is none.
    """

    # The period from the median step between the samples, then again from the steps with the whole seconds
    #   by which each differs from it taken out (a step into or out of a run of frames with the seconds off)
    dc, dt = np.diff(seqs), np.diff(t)
    period = float(np.median(dt/dc))
    if not (period > 0):
        return None
    period = float(np.median((dt - np.round(dt - period*dc))/dc))
    if not (period > 0):
        return None

    # The fraction of a second of the line: the median of the fractions of the samples from it, taken around
    #   their circular mean so that they are not split at the wrap of whole seconds
    r = t - period*seqs
    ref = float(np.angle(np.mean(np.exp(2j*np.pi*r))))/(2*np.pi)
    phase = ref + float(np.median((r - ref + 0.5)%1.0)) - 0.5

    # The whole seconds of the line: the level shared by most samples on the fraction of the line (the latest
    #   of equally shared levels, as frames with the seconds field off read early)
    levels = np.round(r - phase).astype(np.int64)
    near = np.abs(r - phase - levels) < tolerance
    if not np.any(near):
        return None
    values, counts = np.unique(levels[near], return_counts=True)
    level = int(np.max(values[counts == np.max(counts)]))

    return period, phase + level


def longOffStretches(seqs, on, max_run_frames):
    """ The stretches of consecutive headers off the model which may hold a run of frames off it longer than
        max_run_frames: the headers on the model on either side of them (or the ends of the input) are further
        apart.

    Arguments:
        seqs: [ndarray] Frame counters of the headers (increasing).
        on: [ndarray] Whether each header is on the model (bool).
        max_run_frames: [int] Longest run of frames with the seconds field off.

    Return:
        [list] (first, last, counter) of each stretch: the indices of its first and last headers, and the
            counter of the header on the model before it (one less than the first counter at the start of the
            input).
    """

    # The first and the last header of each stretch of consecutive headers off the model
    edges = np.flatnonzero(np.diff(np.concatenate([[0], (~on).astype(np.int64), [0]])))

    # The stretches whose headers on the model on either side are more than max_run_frames frames apart (the
    #   frames before the first header and after the last one taken as on the model)
    stretches = []
    for first, last in zip(edges[::2], edges[1::2] - 1):
        prev_seq = seqs[first - 1] if first > 0 else seqs[0] - 1
        next_seq = seqs[last + 1] if last + 1 < len(seqs) else seqs[-1] + 1
        if next_seq - prev_seq - 1 > max_run_frames:
            stretches.append((int(first), int(last), int(prev_seq)))

    return stretches


def denseTimeModel(readHeader, pos, seqs, secs, usecs, tolerance=0.1,
                   whole_tolerance=WHOLE_SECOND_TOLERANCE, max_run_frames=MAX_OFF_RUN_FRAMES,
                   step_frames=DENSE_STEP_FRAMES, max_dense=MAX_DENSE_HEADERS):
    """ The model of the frame timestamps (see fitTimeModel) when the sampled frames alone can't give it: runs
        of frames with the seconds field off come in dense bursts (a run every 64 frames or so), so
        consecutive samples land in different runs and more of them are off the model than a single run could
        explain. The line is taken from the samples modulo whole seconds (see wholeSecondLine), the headers
        between the samples off it are read (on a grid of frames, made finer down to every step_frames frames
        where needed), and the runs of frames off the model they show have to be at most max_run_frames long.
        A longer run is a step in the timestamps (one which lasts to an end of the input too), and so is a
        header off by anything else than whole seconds early; there is then no model.

    Arguments:
        readHeader: [function] Gives the header of a frame of the input (see sampleTimeModel).
        pos: [list] Indices in the input of the sampled frames (increasing).
        seqs: [list] Their frame counters.
        secs: [list] Seconds fields of their timestamps.
        usecs: [list] Microseconds fields of their timestamps.

    Keyword arguments:
        tolerance: [float] Largest difference (s) between a timestamp and the model, apart from whole seconds.
            0.1 by default.
        whole_tolerance: [float] Largest difference (s) from exactly whole seconds early of a header off the
            model. WHOLE_SECOND_TOLERANCE by default.
        max_run_frames: [int] Longest run of frames with the seconds field off. MAX_OFF_RUN_FRAMES by default.
        step_frames: [int] Spacing (frames) of the finest grid of headers read between the samples.
            DENSE_STEP_FRAMES by default.
        max_dense: [int] Most headers read besides the samples; with more needed there is no model.
            MAX_DENSE_HEADERS by default.

    Return:
        [tuple] The model (see fitTimeModel), or None.
    """

    # Too few samples for a model
    if len(seqs) < 8:
        return None

    # The counter has to increase from frame to frame
    pos = np.array(pos, dtype=np.int64)
    seqs = np.array(seqs, dtype=np.int64)
    if np.any(np.diff(seqs) <= 0):
        return None

    # The line modulo whole seconds from the samples, most of which have to be on it
    sec0, seq0 = int(secs[0]), int(seqs[0])
    t = (np.array(secs, dtype=np.int64) - sec0) + np.array(usecs)/1e6
    line = wholeSecondLine(seqs - seq0, t, tolerance=tolerance)
    if line is None:
        return None
    period, offset = line
    on = np.abs(t - offset - period*(seqs - seq0)) < tolerance
    if np.sum(on) < 0.5*len(t):
        return None

    # The line refined on the samples on it, which every sample off it has to read whole seconds early against
    #   (else a real step, and no more headers are read)
    fit = refineTimeLine(seqs, t, on, seq0, period, offset, whole_tolerance=whole_tolerance)
    if fit is None:
        return None
    period, offset = fit[:2]

    # The headers read so far, by their index in the input, the frames without a header, and the number read
    headers = {int(k): (int(q), int(s), int(u)) for k, q, s, u in zip(pos, seqs, secs, usecs)}
    no_header = set()
    n_read = [0]

    # Whether a header is on the line refined on the samples
    def onLine(h, period=period, offset=offset):
        return abs((h[1] - sec0) + h[2]/1e6 - offset - period*(h[0] - seq0)) < tolerance

    # The header of a frame, read if it has not been, as long as no more than max_dense have been read (False
    #   past that, None if the frame has no header)
    def getHeader(k):
        if (k not in headers) and (k not in no_header):
            n_read[0] += 1
            if n_read[0] > max_dense:
                return False
            h = readHeader(k)
            if h is None:
                no_header.add(k)
            else:
                headers[k] = tuple(int(v) for v in h)
        return headers.get(k)

    # The headers on a grid of frames between two frames (False past max_dense headers read)
    def readGrid(a, b, step):
        return all(getHeader(k) is not False for k in range(a - a%step + step, b, step))

    # Each stretch of samples off the model which may hold a run of frames longer than max_run_frames is
    #   walked through in the order of the frames, reading every max_run_frames/2 frames (which finds most
    #   runs of a burst bounded, with few reads). Where the run of headers off the model since the last one on
    #   it grows past max_run_frames, or ends longer, the frames between are read every step_frames frames,
    #   and a run still too long then is a step in the timestamps (the rest is not read)
    coarse = max(step_frames, max_run_frames//2)
    for first, last, prev_seq in longOffStretches(seqs, on, max_run_frames):
        start = pos[first - 1] if first > 0 else pos[first] - 1
        end = pos[last + 1] if last + 1 < len(pos) else pos[last] + 1
        next_seq = seqs[last + 1] if last + 1 < len(seqs) else seqs[last] + 1
        grid = range(int(start) - int(start)%coarse + coarse, int(end), coarse)
        last_on, last_k, run_off = int(prev_seq), int(start), False
        for k in sorted(set(grid) | set(pos[first:last + 1].tolist())) + [int(end)]:

            # The header of the frame, on the model or not (the end of the stretch is on it)
            if k == end:
                q, on_k = int(next_seq), True
            else:
                h = getHeader(k)
                if h is False:
                    return None
                if h is None:
                    continue
                q, on_k = h[0], onLine(h)

            # A run off the model grown or ended too long: the frames since the last header on it read on the
            #   fine grid, and the last header on it found there
            grown = (not on_k) and (q - last_on > max_run_frames)
            ended = on_k and run_off and (q - last_on - 1 > max_run_frames)
            if grown or ended:
                if not readGrid(last_k, k, step_frames):
                    return None
                found = [j for j in headers if (last_k < j < k) and onLine(headers[j])]
                if found:
                    last_k = max(found)
                    last_on = headers[last_k][0]

            # A header on the model ends the run, and a run still longer than max_run_frames on the fine grid
            #   is a step
            if on_k:
                last_on, last_k, run_off = q, k, False
            elif q - last_on > max_run_frames:
                return None
            else:
                run_off = True

    # All headers read, which the counter has to increase over too
    order = sorted(headers)
    seqs = np.array([headers[k][0] for k in order], dtype=np.int64)
    if np.any(np.diff(seqs) <= 0):
        return None
    t = np.array([headers[k][1] - sec0 + headers[k][2]/1e6 for k in order])

    # The line refined on all headers on the line of the samples, which every header off it has to read whole
    #   seconds early against
    on = np.array([onLine(headers[k]) for k in order])
    fit = refineTimeLine(seqs, t, on, seq0, period, offset, whole_tolerance=whole_tolerance)
    if fit is None:
        return None
    period, offset, res = fit

    # Every run of headers off the model, between the headers on it on either side (or the ends of the input),
    #   at most max_run_frames frames long
    if longOffStretches(seqs, on, max_run_frames):
        return None

    return seq0, sec0, float(offset), float(period), seqs[on].astype(np.float64), res[on]


def wholeSecondsOff(ts, tu, seq, model, tolerance=WHOLE_SECOND_TOLERANCE):
    """ The whole seconds by which a timestamp reads early compared to the time its frame counter gives (see
        fitTimeModel).

    Arguments:
        ts: [int] Seconds field of the timestamp.
        tu: [int] Microseconds field.
        seq: [int] Frame counter.
        model: [tuple] The model of the timestamps. 0 is returned if None.

    Keyword arguments:
        tolerance: [float] Largest difference (s) from exactly whole seconds early. WHOLE_SECOND_TOLERANCE by
            default.

    Return:
        [int] The whole seconds the seconds field is off (negative: to be subtracted), 0 if it is not off.
    """

    if model is None:
        return 0

    seq0, sec0, offset, period, seq_on, res_on = model

    # The difference of the timestamp from the line of the model, against the samples on the model next to the
    #   frame: only exactly whole seconds early is the seconds field off
    res = (ts - sec0) + tu/1e6 - offset - period*(seq - seq0)

    return wholeSecondsEarly(seq, res, seq_on, res_on, tolerance)


def repairFrameTime(st, model, tolerance=WHOLE_SECOND_TOLERANCE):
    """ Correct the seconds field of the timestamp of a frame read into the given structure (see readFrame) if
        it differs from the time given by the frame counter by whole seconds.

    Arguments:
        st: [Vid structure] Frame read by readFrame.
        model: [tuple] The model of the timestamps of the file (see frameTimeModel). Nothing is done if None.

    Keyword arguments:
        tolerance: [float] Largest difference (s) from exactly whole seconds early. WHOLE_SECOND_TOLERANCE by
            default.

    Return:
        [int] The whole seconds added to the seconds field, 0 if the timestamp was not changed.
    """

    n = wholeSecondsOff(st.ts, st.tu, st.seq, model, tolerance=tolerance)
    st.ts -= n

    return -n



def readVid(dir_path, file_name):
    """ Read in a *.vid file. 
    
    Arguments:
        dir_path: [str] path to the directory where the *.vid file is located
        file_name: [str] name of the *.vid file

    Return:
        [VidStruct object]
    """

    # Open the file for binary reading
    fid = open(os.path.join(dir_path, file_name), 'rb')

    # Init the vid struct
    vid = VidStruct()

    # Read the info from the first frame
    readFrame(vid, fid, metadata_only=True)

    # Reset the file pointer to the beginning
    fid.seek(0)
    
    vid.frames = []

    # Read in the frames
    while True:

        # Init a new frame structure
        frame = VidStruct()

        # Read one frame
        #fr = np.fromfile(fid, dtype=np.uint16, count=vid.seqlen//2)
        frame.img_data = readFrame(frame, fid)

        # Check if we have reached the end of file
        if frame.img_data is None:
            break

        # Reshape the frame and add it to the frame list
        vid.frames.append(frame)

    fid.close()


    return vid



if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Vid file path
    dir_path = "../../MirfitPrepare/20160929_050928_mir"
    file_name = "ev_20160929_050928A_01T.vid"

    # Read in the *.vid file
    vid = readVid(dir_path, file_name)

    frame_num = 125

    # Show one frame of the vid file
    plt.imshow(vid.frames[frame_num].img_data, cmap='gray', vmin=0, vmax=255)
    plt.show()


