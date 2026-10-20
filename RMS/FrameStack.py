""" Burst stacking of the saved frames (config.frame_stack).

A saved frame is normally the one video frame at each save tick (ref + k*frame_save_aligned_interval).
With frame_stack = n > 1 it is the mean of the n consecutive frames centred on the tick instead:
n//2 frames before the tick frame, the tick frame, and the rest after it (n = 8: offsets -4..+3,
0.32 s at 25 fps). Night sky noise in a single frame is almost all temporal, so the mean of 8 frames
has ~1/sqrt(8) of it while faint extended objects (contrails) keep their amplitude. Benchmarked
2026-10-17/20 on US05 cameras with synthetic contrails: the faintest detectable contrail after the
timelapse codec is 53 % fainter at night and 19 % fainter by day, at equal video size.

The mean is saved as a 16-bit PNG in 8.8 fixed point: value = round(256 * mean 8-bit code), exact for
n = 8 (the sum times 32). 8.8 rather than full-scale 16-bit (x257) because it is exact and its PNG is
26 % smaller (5.2 vs 7.1 MB for a 1080p colour day frame). Divide by STACK_SCALE to get 8-bit codes.

The capture process only sums (uint16, one cv2.add per frame, ~5 ms at 1080p colour); the frame saver
process does the scaling and the PNG.
"""

from __future__ import print_function, division, absolute_import

import math

import numpy as np
import cv2


# Saved value = STACK_SCALE * mean 8-bit code
STACK_SCALE = 256

# Largest supported stack: the uint16 sum must hold (n + 2)*255
MAX_STACK = 32


def stackWindow(n):
    """ Frame offsets from the tick frame that a stack of n frames covers.

    Arguments:
        n: [int] Frames per stack.

    Return:
        (lo, hi): [tuple[int, int]] First and last offset, inclusive: n//2 frames before the tick
            frame, the tick frame, n - n//2 - 1 frames after it.
    """
    return -(n//2), n - n//2 - 1


def stackToImage(frame_sum, count):
    """ The saved 16-bit image of a stack: round(STACK_SCALE * sum/count), saturating.

    Arguments:
        frame_sum: [ndarray] uint16 sum of the stacked 8-bit frames.
        count: [int] Number of frames in the sum.

    Return:
        image: [ndarray] uint16 image, STACK_SCALE per 8-bit code.
    """
    return cv2.addWeighted(frame_sum, STACK_SCALE/float(max(1, count)), frame_sum, 0.0, 0.0, dtype=cv2.CV_16U)


class FrameStack(object):
    """ A finished stack: the sum of its frames and where it sits in time. """

    def __init__(self, frame_sum, count, timestamp, meta, t_first, t_last):
        """
        Arguments:
            frame_sum: [ndarray] uint16 sum of the frames (same shape as the frames).
            count: [int] Number of frames summed.
            timestamp: [float] Time of the tick frame (the frame a single-frame save would have
                taken), or the tick time when that frame was dropped.
            meta: Exposure record of the tick frame (or of the first frame when it was dropped), as
                the capture keeps it, or None.
            t_first, t_last: [float] Times of the first and last frame in the sum.
        """
        self.frame_sum = frame_sum
        self.count = count
        self.timestamp = timestamp
        self.meta = meta
        self.t_first = t_first
        self.t_last = t_last


class FrameStacker(object):
    """ Sums the n consecutive frames centred on each save tick.

    Frames are placed by their timestamps: a frame belongs to tick k = round((t - ref)/dt) at offset
    round((t - tick time)*fps) frames, and is summed if that offset is inside stackWindow(n). A stack
    is finished by the frame at the last offset, or by the first frame after the window when that
    one was dropped. Dropped frames simply leave the sum with fewer frames (the count says how
    many). A frame of a different shape (grayscale <-> colour switch) discards the stack in progress.
    """

    def __init__(self, n, dt, fps, ref=0.0):
        """
        Arguments:
            n: [int] Frames per stack, 2..MAX_STACK.
            dt: [float] Save interval (s), config.frame_save_aligned_interval.
            fps: [float] Frame rate.

        Keyword arguments:
            ref: [float] Epoch of tick 0 (s). 0 by default.
        """
        if not 2 <= n <= MAX_STACK:
            raise ValueError("frame_stack must be 2..{:d}, got {}".format(MAX_STACK, n))
        self.n = int(n)
        self.lo, self.hi = stackWindow(self.n)
        self.dt = float(dt)
        self.fps = float(fps)
        self.ref = float(ref)

        self.tick = None            # tick of the stack in progress
        self.done_tick = None       # last tick whose stack was finished (or abandoned)
        self.frame_sum = None
        self.count = 0
        self.centre = None          # (timestamp, meta) of the tick frame
        self.first_meta = None
        self.t_first = None
        self.t_last = None
        self.discarded = 0          # stacks abandoned on a frame shape change


    def position(self, ts):
        """ Tick and offset of a frame time.

        Arguments:
            ts: [float] Frame time (s).

        Return:
            (tick, offset): [tuple[int, int | None]] offset is None outside the stack window.
        """
        tick = int(math.floor((ts - self.ref)/self.dt + 0.5))
        off = int(round((ts - (self.ref + tick*self.dt))*self.fps))
        return tick, (off if self.lo <= off <= self.hi else None)


    def startsStack(self, ts):
        """ True if a frame at this time would start a new stack (the frame to check for colour). """
        tick, off = self.position(ts)
        return (off is not None) and (tick != self.tick) and (tick != self.done_tick)


    def _finish(self):
        """ Close the stack in progress and return it. """
        if self.centre is not None:
            timestamp, meta = self.centre
        else:
            timestamp, meta = self.ref + self.tick*self.dt, self.first_meta
        out = FrameStack(self.frame_sum, self.count, timestamp, meta, self.t_first, self.t_last)
        self.done_tick, self.tick = self.tick, None
        self.frame_sum = None
        return out


    def add(self, frame, ts, meta=None):
        """ Add one video frame.

        Arguments:
            frame: [ndarray] uint8 frame (H x W or H x W x C).
            ts: [float] Frame time (s).

        Keyword arguments:
            meta: The frame's exposure record (passed through to the finished stack), or None.

        Return:
            stack: [FrameStack | None] A stack finished by this frame, else None. Its frame_sum is
                not touched again by the stacker.
        """
        tick, off = self.position(ts)
        out = None

        # The window of the stack in progress has passed (its last frame was dropped)
        if (self.tick is not None) and ((tick != self.tick) or (off is None)):
            out = self._finish()

        if (off is None) or (tick == self.done_tick):
            return out

        # Grayscale <-> colour switch in the middle of a stack: drop the partial stack
        if (self.tick is not None) and (frame.shape != self.frame_sum.shape):
            self.discarded += 1
            self.tick = None
            self.frame_sum = None

        if self.tick is None:
            self.tick = tick
            self.frame_sum = np.zeros(frame.shape, np.uint16)
            self.count = 0
            self.centre = None
            self.first_meta = meta
            self.t_first = ts

        # Broken timestamps could pile up frames on one tick; the sum must not overflow
        if self.count < self.n + 2:
            cv2.add(self.frame_sum, frame, dst=self.frame_sum, dtype=cv2.CV_16U)
            self.count += 1
            self.t_last = ts
            if off == 0:
                self.centre = (ts, meta)

        if off == self.hi:
            out = self._finish()

        return out
