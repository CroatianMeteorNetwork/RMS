""" A segment of the frames of an input held in memory, which stands in for its frame interface.

The frames of a segment are read from the input once, in order, and every pass over them (the star extraction,
the background, the search, the measurement) takes them from memory instead of reading the input again. The
segment slides along the input: the frames which the next segment shares with it stay in memory and are not read
again. The frame numbers are those of the whole input.
"""

from __future__ import print_function, division, absolute_import

import datetime

import numpy as np

from RMS.Routines.DynamicFTPCompressionCy import FFMimickInterface


# Input types whose frames and chunks a segment gives exactly as the input itself does (their chunks are made of
#   the frames which loadFrame returns): .vid files and frame blocks
SEGMENT_INPUT_TYPES = ('vid', 'fblk')


def segmentFrames(img_handle, config, memory_gb):
    """ The number of frames of a segment which fit in the given memory, in whole blocks of the matched filter
        (which are whole chunks of the star extraction too), or 0 if the input can't be read in segments.

    Arguments:
        img_handle: [FrameInterface] Frame interface of the input (detection=True).
        config: [Config] Configuration object.
        memory_gb: [float] Memory of the segment (GB).

    Return:
        [int] Number of frames of a segment, 0 for none.
    """

    # Without the memory, or for other inputs, the frames are read from the input for every pass
    if (memory_gb <= 0) or (img_handle.input_type not in SEGMENT_INPUT_TYPES):
        return 0

    # The frames are kept as they are read (binned, uint16 for the .vid files), as many as fit, in whole units
    frame_bytes = img_handle.nrows*img_handle.ncols*np.dtype(np.uint16).itemsize
    unit = segmentUnit(config, img_handle)
    segment_frames = int(memory_gb*1024**3//frame_bytes)//unit*unit

    # A segment holds at least twice the blocks which the slow search compares, so that the last segment can
    #   begin early enough to hold them too (see segmentRanges). A smaller memory is not used
    if segment_frames < 2*segmentMinimum(config, img_handle):

        # Except a frame block (e.g. 30 s of frames) which fits in the memory as a whole: it is one segment of all
        #   its frames, nothing is cut, so the minimum of the segments of a long input doesn't apply
        total_frames = int(getattr(img_handle, 'total_frames', 0) or 0)
        if (img_handle.input_type == 'fblk') and (total_frames > 0) \
                and (total_frames*frame_bytes <= memory_gb*1024**3):
            return total_frames

        return 0

    return segment_frames


def segmentUnit(config, img_handle):
    """ Segments begin and end on whole blocks of the matched filter (as it uses them, see
        matchedFilterBlockFrames) and whole chunks of the star extraction. """

    # Imported here, as the matched filter uses this module
    from RMS.MatchedFilterDetection import matchedFilterBlockFrames

    return int(np.lcm(matchedFilterBlockFrames(config), int(getattr(img_handle, 'chunk_frames', 1) or 1)))


def segmentMinimum(config, img_handle):
    """ The fewest frames of a segment, in whole units: the slow search of the matched filter compares every block
        with blocks a few blocks before and after it, and can find nothing in fewer blocks (see
        slowSearchMinBlocks). """

    # Imported here, as the matched filter uses this module
    from RMS.MatchedFilterDetection import matchedFilterBlockFrames, slowSearchMinBlocks

    unit = segmentUnit(config, img_handle)

    return -(-slowSearchMinBlocks()*matchedFilterBlockFrames(config)//unit)*unit


def segmentOverlap(segment_frames, overlap, unit):
    """ The overlap of neighbouring segments: at most half a segment, in whole units, so only neighbouring
        segments share frames (and every segment adds new frames).

    Arguments:
        segment_frames: [int] Number of frames of a segment.
        overlap: [int] Requested overlap (frames).
        unit: [int] Frames which segments begin and end on.

    Return:
        [int] The overlap (frames).
    """

    return max(0, min(overlap, (segment_frames//2)//unit*unit))


def segmentRanges(total_frames, segment_frames, overlap, unit, minimum=0):
    """ The frame ranges of the segments of an input: every segment begins overlap frames before the end of the
        previous one, and the last one ends at the end of the input. A last segment which would add less than a
        unit of new frames is taken into the one before it, if that still fits in the memory, and a last segment
        shorter than the minimum begins earlier. The overlap is at most half a segment (see segmentOverlap), so
        only neighbouring segments share frames.

    Arguments:
        total_frames: [int] Number of frames of the input.
        segment_frames: [int] Number of frames of a segment (a multiple of unit).
        overlap: [int] Number of frames which neighbouring segments share (a multiple of unit).
        unit: [int] Frames which segments begin and end on.

    Keyword arguments:
        minimum: [int] Fewest frames of a segment (a multiple of unit, at most half a segment, see
            segmentMinimum). 0 by default.

    Return:
        [list] Segments as (first frame, end frame), one for the whole input if it fits in one.
    """

    # An input which fits in one segment is one segment
    if (segment_frames <= 0) or (total_frames <= segment_frames):
        return [(0, total_frames)]

    # Every segment begins a step after the previous one, the step being the new frames of a segment (at least
    #   half a segment)
    overlap = segmentOverlap(segment_frames, overlap, unit)
    step = segment_frames - overlap

    # The whole segments before the end of the input
    ranges = []
    first = 0
    while first + segment_frames < total_frames:
        ranges.append((first, first + segment_frames))
        first += step

    # The last segment ends at the end of the input. If it would add little more than the overlap, the segment
    #   before it is extended to the end instead (if it still fits)
    if ((total_frames - first) - overlap < unit) and (total_frames - ranges[-1][0] <= segment_frames):
        ranges[-1] = (ranges[-1][0], total_frames)
    else:

        # A short last segment begins earlier, on a whole unit, to hold the minimum. It shares more frames with the
        #   segment before it, but at most half a segment (the minimum is at most that)
        if total_frames - first < minimum:
            first = max((total_frames - minimum)//unit*unit, ranges[-1][1] - (segment_frames//2)//unit*unit)
        ranges.append((first, total_frames))

    return ranges


class SegmentBuffer(object):
    def __init__(self, img_handle, capacity):
        """ Frames of a segment of an input in memory, in place of the frame interface of the input for them.
            Frames outside the segment are read from the input.

        Arguments:
            img_handle: [FrameInterface] Frame interface of the input (detection=True).
            capacity: [int] Largest number of frames of a segment.
        """

        self.img_handle = img_handle
        self.capacity = capacity

        # The frames in memory (allocated with the first segment, when their type is known), their times, and
        #   the range of the input they are
        self.frames = None
        self.times = []
        self.first = 0
        self.end = 0

        # The current frame and chunk, numbered as in the whole input
        self.current_frame = 0
        self.current_frame_chunk = 0
        self.current_frame_time = None
        self.frame_chunk_unix_times = []
        self.current_fr_chunk_size = 0
        self.ff = None

        # Number of frames which were read from the input (once each) and outside of the segment
        self.frames_read = 0
        self.frames_outside = 0


    def __getattr__(self, name):
        """ Everything else (the size of the frames, the frame rate, the number of frames, the times of the
            input...) is that of the input. """

        return getattr(self.__dict__['img_handle'], name)


    def load(self, first, end):
        """ Make the frames from first to end the segment in memory. The frames it shares with the current
            segment are kept, the others are read from the input, in order.

        Arguments:
            first: [int] First frame of the segment.
            end: [int] End of the segment (the frame after its last one).
        """

        # The memory holds the longest segment
        if end - first > self.capacity:
            raise ValueError("A segment of {:d} frames doesn't fit in {:d}".format(end - first, self.capacity))

        # The frames which the new segment shares with the current one are moved to its beginning
        keep_first, keep_end = max(first, self.first), min(end, self.end)
        if (self.frames is not None) and (keep_end > keep_first):
            n_keep = keep_end - keep_first
            self.frames[:n_keep] = self.frames[keep_first - self.first:keep_end - self.first]
            self.times = self.times[keep_first - self.first:keep_end - self.first]
            read_from = keep_end
        else:
            n_keep = 0
            self.times = []
            read_from = first

        # The other frames are read from the input, in order, with their times. The memory is allocated for the
        #   longest segment with the first frame, when the size and the type of the frames are known
        for i in range(read_from, end):
            self.img_handle.setFrame(i)
            frame = self.img_handle.loadFrame()
            if self.frames is None:
                self.frames = np.empty((self.capacity,) + frame.shape, dtype=frame.dtype)
            self.frames[n_keep + i - read_from] = frame
            self.times.append(self.img_handle.currentFrameTime(frame_no=i, dt_obj=True))
        self.frames_read += end - read_from

        # The range of the input which is in memory now
        self.first, self.end = first, end


    ### The frame interface for the frames of the segment ###

    def setFrame(self, fr_num):
        """ Set the current frame (as in the whole input). """

        self.current_frame = fr_num%self.img_handle.total_frames


    def loadFrame(self, avepixel=False):
        """ The current frame, from memory, or read from the input outside of the segment. The frame in memory is
            returned itself, it must not be changed. """

        # A frame of the segment, from memory
        n = self.current_frame
        if self.first <= n < self.end:
            self.current_frame_time = self.times[n - self.first]
            return self.frames[n - self.first]

        # A frame outside of the segment is read from the input (and counted)
        self.frames_outside += 1
        self.img_handle.setFrame(n)
        frame = self.img_handle.loadFrame()
        self.current_frame_time = self.img_handle.currentFrameTime(frame_no=n, dt_obj=True)

        return frame


    def currentFrameTime(self, frame_no=None, dt_obj=False):
        """ The time of the given frame (of the current one if not given). """

        # The time of a frame of the segment from memory, of the other frames from the input
        n = self.current_frame if frame_no is None else frame_no
        if self.first <= n < self.end:
            dt = self.times[n - self.first]
        else:
            dt = self.img_handle.currentFrameTime(frame_no=n, dt_obj=True)

        # As a datetime, or as a tuple like the frame interfaces
        if dt_obj:
            return dt

        return (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond/1000)


    def nextChunk(self):
        """ Go to the next chunk of frames (as in the whole input). """

        self.current_frame_chunk = (self.current_frame_chunk + 1)%self.img_handle.total_fr_chunks
        self.current_frame = self.current_frame_chunk*self.img_handle.chunk_frames


    def loadChunk(self, first_frame=None, read_nframes=None):
        """ The maximum, average and the other images of a chunk of frames, made from the frames as the input
            makes them (see InputTypeUWOVid.loadChunk).

        Keyword arguments:
            first_frame: [int] First frame of the chunk. The current chunk by default.
            read_nframes: [int] Number of frames. chunk_frames by default.

        Return:
            [FFMimickInterface] The images of the chunk.
        """

        # The frames of the chunk, at most until the end of the input
        chunk_frames = self.img_handle.chunk_frames
        if first_frame is None:
            first_frame = self.current_frame_chunk*chunk_frames
        n = chunk_frames if (read_nframes is None) else read_nframes
        n = max(min(n, self.img_handle.total_frames - first_frame), 0)

        # The images of the chunk are made from its frames as the input makes them: the type of the images is
        #   set from the first frame, and the time of every frame is kept (as UNIX time)
        ff = None
        self.frame_chunk_unix_times = []
        for i in range(n):
            self.setFrame(first_frame + i)
            frame = self.loadFrame()
            if ff is None:
                ff = FFMimickInterface(frame.shape[0], frame.shape[1], self.img_handle.getTargetDtype(),
                                       res_size=self.img_handle.config.background_reservoir_size,
                                       median_threads=self.img_handle.config.median_threads)
            ff.addFrame(frame.astype(np.uint16))
            if i == 0:
                ff.dtype = self.img_handle.getTargetDtype(frame)
            t = self.current_frame_time
            self.frame_chunk_unix_times.append((t - datetime.datetime(1970, 1, 1, tzinfo=t.tzinfo)).total_seconds())

        # The finished images are the current chunk
        ff.finish()
        self.current_fr_chunk_size = n
        self.ff = ff

        return ff
