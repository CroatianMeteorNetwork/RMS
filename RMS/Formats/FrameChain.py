""" An input preceded by the last frames of the input recorded just before it, so that a search which follows objects
    from frame to frame (the matched filter) continues across the boundary between consecutive files of a
    recording, as in one long file.

The chained input numbers its frames from the first borrowed frame: frames 0 to n_context - 1 are the last frames of
the previous input, frames n_context onwards are those of the input. It begins at its first frame (the times of the
frames relative to the beginning follow its frames), and the results are named after the beginning of the input
(name_datetime).
"""

from __future__ import print_function, division, absolute_import

import datetime

import numpy as np


class ChainedInput(object):
    def __init__(self, previous, n_context, current):
        """ The input current, preceded by the last n_context frames of the input previous.

        Arguments:
            previous: [FrameInterface] The input recorded just before (opened like current).
            n_context: [int] Number of its last frames which are put in front (at most all of its frames).
            current: [FrameInterface] The input.
        """

        # The two inputs, and the borrowed frames: the last n_context frames of the previous input
        self.previous = previous
        self.current = current
        self.n_context = int(min(max(n_context, 0), previous.total_frames))
        self.context_first = previous.total_frames - self.n_context

        # The chained input is the input for everything but its frames: its type, size, frame rate, chunks and
        #   beginning
        self.input_type = current.input_type
        self.nrows, self.ncols = current.nrows, current.ncols
        self.fps = current.fps
        self.chunk_frames = current.chunk_frames

        # The chained input begins at its first frame, the results are named after the input
        self.name_datetime = current.beginning_datetime
        self.beginning_datetime = previous.currentFrameTime(frame_no=self.context_first, dt_obj=True) \
            if self.n_context > 0 else current.beginning_datetime

        # The directory and the properties of the images are those of the input, and the frames those of both
        self.dir_path = getattr(current, 'dir_path', None)
        self.byteswap = getattr(current, 'byteswap', False)
        self.ff = getattr(current, 'ff', None)
        self.total_frames = self.n_context + current.total_frames
        self.current_frame = 0

        # The chunks of frames: those of the borrowed frames (from the first one), then those of the input, so the
        #   chunks of the input (e.g. of its stars) are chunks of the chained input as well
        self.chunk_first_frames = list(range(0, self.n_context, self.chunk_frames)) \
            + [self.n_context + k*self.chunk_frames for k in range(max(1, current.total_frames//self.chunk_frames))]
        self.total_fr_chunks = len(self.chunk_first_frames)

        # The times of the frames (unix s), from the frame tables of both inputs if they have them
        prev_times = getattr(previous, 'frame_unix_times', None)
        cur_times = getattr(current, 'frame_unix_times', None)
        if (prev_times is not None) and (cur_times is not None):
            self.frame_unix_times = np.concatenate([np.asarray(prev_times, dtype=np.float64)[self.context_first:],
                                                    np.asarray(cur_times, dtype=np.float64)])
        else:
            self.frame_unix_times = None


    def _source(self, frame_no):
        """ The input and its frame for a frame of the chained input. """

        if frame_no < self.n_context:
            return self.previous, self.context_first + frame_no

        return self.current, frame_no - self.n_context


    def setFrame(self, frame_no):
        """ Go to a frame of the chained input. """

        self.current_frame = int(frame_no)
        handle, i = self._source(self.current_frame)
        handle.setFrame(i)


    def loadFrame(self, avepixel=False):
        """ Load the current frame, from the input it belongs to. """

        handle, _ = self._source(self.current_frame)
        frame = handle.loadFrame(avepixel=avepixel)
        self.current_frame_time = getattr(handle, 'current_frame_time', None)

        return frame


    def currentFrameTime(self, frame_no=None, dt_obj=False):
        """ The time of a frame of the chained input (the current one by default). """

        frame_no = self.current_frame if frame_no is None else int(frame_no)
        handle, i = self._source(frame_no)

        return handle.currentFrameTime(frame_no=i, dt_obj=dt_obj)


    def hasAccurateTime(self):
        """ Both inputs need the time of every frame for their frames to follow one another in time. """

        return bool(getattr(self.previous, 'hasAccurateTime', lambda: False)()) \
            and bool(getattr(self.current, 'hasAccurateTime', lambda: False)())


    def name(self, beginning=False):
        """ The name of the input. """

        return self.current.name(beginning=beginning)


    def toCurrentFrames(self, frames):
        """ Frames of the chained input as frames of the input (negative for the borrowed frames). """

        return np.asarray(frames, dtype=np.float64) - self.n_context



def contiguous(previous, current, factor=1.5):
    """ Whether an input follows another one without missing frames: its first frame comes at most factor frame
        intervals after the last frame of the other one, at the same frame rate (within 1%) and with frames of the same
        size.

    Arguments:
        previous: [FrameInterface] The earlier input.
        current: [FrameInterface] The later input.

    Keyword arguments:
        factor: [float] Largest step, in frame intervals of the later input. 1.5 by default.

    Return:
        [bool] True if the inputs are contiguous (and both have the time of every frame).
    """

    # Both inputs need the time of every frame
    for handle in (previous, current):
        if (not hasattr(handle, 'hasAccurateTime')) or (not handle.hasAccurateTime()):
            return False

    # A recording which changed the frame size or the frame rate in between can't be searched as one
    if (previous.nrows, previous.ncols) != (current.nrows, current.ncols):
        return False
    if abs(previous.fps - current.fps) > 0.01*current.fps:
        return False

    # The step from the last frame of the earlier input to the first frame of the later one
    t_last = previous.currentFrameTime(frame_no=previous.total_frames - 1, dt_obj=True)
    t_first = current.currentFrameTime(frame_no=0, dt_obj=True)
    step = (t_first.replace(tzinfo=None) - t_last.replace(tzinfo=None)).total_seconds()

    return 0 < step <= factor/current.fps
