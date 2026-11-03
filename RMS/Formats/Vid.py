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




def frameTimeModel(fid, n_samples=64, tolerance=0.1, max_run=3):
    """ A model of the frame timestamps of a vid file from its frame counter. The timestamps are written as the
        time of the first frame plus the counter times the frame period, so the time of any frame follows from
        its counter. A few frames can have a seconds field off by whole seconds (the microseconds stay right),
        which the model shows (see repairFrameTime). The model is built from the headers of frames spread over
        the file, and the position in the file is kept.

    Arguments:
        fid: [file handle] File handle to the vid file, open in the 'rb' mode.

    Keyword arguments:
        n_samples: [int] Number of frames sampled. 64 by default.
        tolerance: [float] Largest difference (s) between a sampled timestamp and the model, apart from whole
            seconds. 0.1 by default (a few lost frames).
        max_run: [int] Most consecutive sampled frames off the model by whole seconds. Frames with the seconds
            field off come in short runs, far shorter than the spacing of the samples, so more consecutive
            samples off the model are a step in the timestamps. 3 by default.

    Return:
        [tuple] (counter, seconds, offset, period, counters, residuals) of the model (see fitTimeModel): the time
            of a frame with the counter c is seconds + offset + period*(c - counter), plus the residuals of the
            sampled frames interpolated at c. None if the counter doesn't give the timestamps of the sampled frames.
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

        # The counter and the timestamp of frames spread over the file
        seqs, secs, usecs = [], [], []
        for k in np.unique(np.linspace(0, n_frames - 1, min(n_samples, n_frames)).astype(np.int64)):
            fid.seek(int(k)*seqlen)
            words = np.fromfile(fid, dtype=np.uint32, count=7)
            seqs.append(int(words[4]))
            secs.append(int(np.int32(words[5])))
            usecs.append(int(np.int32(words[6])))

    finally:
        fid.seek(file_pos)

    return fitTimeModel(seqs, secs, usecs, tolerance=tolerance, max_run=max_run)


# A timestamp with its seconds field off reads exactly whole seconds early: the stamps are written to the
#   microsecond from the counter, and the model gives them to a fraction of a millisecond, while a real gap in the
#   timestamps (lost frames) is a multiple of the frame period, which is not a whole number of seconds to within
#   this tolerance (32 frames of 31.119 ms are 4.2 ms short of a second)
WHOLE_SECOND_TOLERANCE = 0.002


def fitTimeModel(seqs, secs, usecs, tolerance=0.1, max_run=3, whole_tolerance=WHOLE_SECOND_TOLERANCE):
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

    Return:
        [tuple] (counter, seconds, offset, period, counters, residuals) of the model: the line of frameTimeModel
            and the residuals of the samples on it, between which it is interpolated; or None.
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

    # The period and the offset refined by a line fit to the samples on the model
    if np.sum(on) >= 3:
        period, offset = np.polyfit(seqs[on] - seq0, t[on], 1)

    # The residuals of the samples on the model, between which the model is interpolated: a few lost frames make
    #   steps of a frame period in the timestamps, which the line leaves in its residuals
    res = t - offset - period*(seqs - seq0)
    seq_on, res_on = seqs[on], res[on]

    # Every sample off the model has to read exactly whole seconds early (the seconds field low) against the
    #   samples on the model around it: a sample off by anything else, or late, is a real gap or step in the
    #   timestamps (e.g. many lost frames near an end of the input, which leave few samples after them), and the
    #   timestamps are then kept as they are. A long series of samples off the model is a step too
    res = res - np.interp(seqs, seq_on, res_on)
    whole = np.round(res)
    off = ~on
    if np.any(off & ((whole >= 0) | (np.abs(res - whole) >= whole_tolerance))):
        return None
    run = 0
    for o in off:
        run = run + 1 if o else 0
        if run > max_run:
            return None

    return seq0, sec0, float(offset), float(period), seq_on.astype(np.float64), res_on


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

    # The difference of the timestamp from the model, interpolated between the samples on the model around the
    #   frame: only exactly whole seconds early is the seconds field off
    res = (ts - sec0) + tu/1e6 - offset - period*(seq - seq0) - float(np.interp(seq, seq_on, res_on))
    n = int(round(res))
    if (n >= 0) or (abs(res - n) >= tolerance):
        return 0

    return n


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


