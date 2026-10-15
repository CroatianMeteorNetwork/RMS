""" Compiled kernels of the matched-filter detection: the shift-and-add stacks along a grid of velocities, and
the joint fit of a moving PSF to a short run of frames.

The kernels are compiled with numba. The velocity stacks also have a CUDA version, used when numba can reach a
GPU.


The idea of the velocity stack (a matched filter for a moving point source):

An object moving with the velocity v = (vx, vy) px/frame, which is at the position p at the middle time of a
run of N frames, is at p + v*dt_k in the frame k, where dt_k = k - (N - 1)/2 is the time of the frame relative
to the middle of the run. If every frame k is shifted back by v*dt_k and the frames are averaged, the object
adds up at p, while the noise of the frames is independent and averages out:

    S_v(p) = (1/N) * sum_k z_k(p + v*dt_k)

With frames normalized to unit noise (z = (frame - background)/noise), a constant object of peak amplitude A
per frame gives S_v(p) = A at its position, and the noise of S_v is 1/sqrt(N), so the signal-to-noise ratio of
the stack is A*sqrt(N): an object at 1.5 sigma per frame is at 6 sigma in a stack of 16 frames.

The velocity of a faint object is not known, so the stack is computed for a grid of velocities, and every
pixel keeps the largest mean over the velocities and the index of that velocity. A wrong velocity smears the
object over the run, so the grid has to be fine enough that the object stays within about one PSF width over
the run (see velocityGrid in RMS.MatchedFilterDetection).
"""

from __future__ import print_function, division, absolute_import

import math

import numpy as np

import numba

# The GPU version needs numba's CUDA support (the numba-cuda package) and a GPU. Without them, everything runs on
#   the CPU
try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except Exception:
    cuda = None
    CUDA_AVAILABLE = False


def velocityShifts(velocities, n_frames):
    """ Integer pixel shifts of every frame of a stack for every velocity. The object which is at p at the
        middle time of the stack is at p + v*(t - t_mid) in a frame at time t.

    Arguments:
        velocities: [ndarray] Velocities (vx, vy) in px per frame, shape (n_vel, 2).
        n_frames: [int] Number of frames in the stack.

    Return:
        [ndarray] int32 shifts (dx, dy) of shape (n_vel, n_frames, 2).
    """

    # Time of every frame relative to the middle of the stack (frames), e.g. -3.5, -2.5, ..., 3.5 for 8 frames
    dt = np.arange(n_frames) - (n_frames - 1)/2.0

    # The shift of the frame k for the velocity v is v*dt_k, rounded to whole pixels: the stack adds pixels and
    #   doesn't interpolate. The rounding error is at most 0.5 px per frame, well within the PSF smoothing of the
    #   frames which are stacked
    shifts = np.rint(velocities[:, None, :]*dt[None, :, None]).astype(np.int32)

    return np.ascontiguousarray(shifts)


def velocitySamples(velocities, n_frames, streak=False, smooth_sigma=0.0):
    """ The pixels which the stack of every velocity sums: a sample (dx, dy, k) is the frame k shifted by
        (dx, dy). Without the streak, every frame is one sample, at the shift of velocityShifts. With the streak,
        an object which moves L px during a frame is a streak of that length in every frame, and the frame is
        sampled round(L) times along it (at the middles of equal parts of the frame time), so the stack collects
        the light of the whole streak and not only of its middle.

    The sum of a velocity is weighted by one over its noise relative to one sample per frame, which keeps the
    noise of all velocities the same, so their largest sum is taken at the same noise level and the thresholds of
    the search don't change. The samples of a frame are neighbouring pixels of frames smoothed by a Gaussian, so
    their noise is correlated: the variance of their sum is the sum of the correlations of all pairs of samples,
    exp(-d^2/(4*sigma^2)) for samples d px apart (1 for a sample with itself).

    Arguments:
        velocities: [ndarray] Velocities (vx, vy) in px per frame, shape (n_vel, 2).
        n_frames: [int] Number of frames in the stack.

    Keyword arguments:
        streak: [bool] Sample the streak of every frame. False by default (one sample per frame).
        smooth_sigma: [float] Sigma of the Gaussian smoothing of the frames (px), for the correlation of the noise
            of neighbouring samples. 0 by default (independent pixels).

    Return:
        (samples, counts, weights): int32 samples (dx, dy, k) of shape (n_vel, n_max, 3), the int32 number of
            samples of every velocity (n_vel,), and the float32 weight of every velocity (n_vel,). The samples of
            a velocity are ordered by frame.
    """

    # The velocities as rows (vx, vy)
    velocities = np.asarray(velocities, dtype=np.float64).reshape(-1, 2)
    n_vel = len(velocities)

    # The samples per frame: the length of the streak in px (the motion during a frame), at least 1
    speed = np.hypot(velocities[:, 0], velocities[:, 1])
    per_frame = np.maximum(1, np.rint(speed)).astype(np.int32) if streak else np.ones(n_vel, dtype=np.int32)
    n_max = n_frames*int(per_frame.max()) if n_vel else n_frames

    # The samples of every velocity (a slower velocity uses only the first of them, the rest are not summed), their
    #   number, and the weights (1 for one sample per frame)
    samples = np.zeros((n_vel, n_max, 3), dtype=np.int32)
    counts = (n_frames*per_frame).astype(np.int32)
    weights = np.ones(n_vel, dtype=np.float32)

    # The shift of the middle of every frame, as in velocityShifts
    shifts = velocityShifts(velocities, n_frames)

    for v in range(n_vel):

        # The offsets of the samples of a frame from its middle: the motion at the middles of per_frame equal
        #   parts of the frame, in whole pixels (rounded half up, so the samples of a frame are different
        #   pixels, symmetric around the middle when that is possible)
        m = per_frame[v]
        tau = (np.arange(m) + 0.5)/m - 0.5
        off_x = np.floor(velocities[v, 0]*tau + 0.5).astype(np.int32)
        off_y = np.floor(velocities[v, 1]*tau + 0.5).astype(np.int32)

        # The samples frame by frame: the shift of the frame plus the offsets, and the frame of every sample
        samples[v, :n_frames*m, 0] = (shifts[v, :, 0][:, None] + off_x[None, :]).ravel()
        samples[v, :n_frames*m, 1] = (shifts[v, :, 1][:, None] + off_y[None, :]).ravel()
        samples[v, :n_frames*m, 2] = np.repeat(np.arange(n_frames), m)

        # The variance of the sum of the samples of a frame, relative to one sample: the correlations of all pairs
        #   (pixels further apart than a few sigmas are independent, and without smoothing only a sample with
        #   itself counts)
        if m > 1:
            d2 = (off_x[:, None] - off_x[None, :])**2 + (off_y[:, None] - off_y[None, :])**2
            if smooth_sigma > 0:
                corr = np.exp(-d2/(4.0*smooth_sigma**2))
            else:
                corr = (d2 == 0).astype(np.float64)
            weights[v] = 1.0/math.sqrt(float(np.sum(corr)))

    return np.ascontiguousarray(samples), counts, weights


@numba.njit(parallel=True, fastmath=True, cache=True)
def velocityStackMaxCPU(frames, pad, samples, counts, weights, out_max, out_idx):
    """ For every pixel, the largest mean of the frames along the velocities, and the index of that velocity.
        The frames are zero-padded, so whole rows are summed for every velocity without bounds checks (pixels
        shifted outside the image add zero).

    Arguments:
        frames: [ndarray] float32 frames of the stack, padded by pad px on every side, shape
            (n_frames, height + 2*pad, width + 2*pad).
        pad: [int] Padding (px), at least the largest shift.
        samples: [ndarray] int32 samples (dx, dy, k) of every velocity, shape (n_vel, n_max, 3), as given by
            velocitySamples.
        counts: [ndarray] int32 number of samples of every velocity.
        weights: [ndarray] float32 weight of the sum of every velocity.
        out_max: [ndarray] float32 output image of the largest mean (height, width).
        out_idx: [ndarray] int32 output image of the index of the velocity with the largest mean.
    """

    n_frames = frames.shape[0]
    height, width = out_max.shape
    n_vel = samples.shape[0]

    # Every row of the output is computed by one thread. For a row, the stacks of all velocities are computed one
    #   after the other, a whole row at a time, which reads the frames in memory order
    for y in numba.prange(height):

        # The largest stack of every pixel of the row so far, and the index of its velocity
        best = np.full(width, -1e30, dtype=np.float32)
        best_idx = np.zeros(width, dtype=np.int32)

        # The stack of the current velocity for the whole row
        acc = np.empty(width, dtype=np.float32)

        for v in range(n_vel):

            # Sum of the frames along the velocity: the pixel (x, y) of the stack is the sum over the frames of the
            #   pixels (x + dx_k, y + dy_k), where the object which is at (x, y) at the middle time is in the frame
            #   k. A row of the stack is therefore a shifted row of every frame (in the padded frames, the shift
            #   never goes outside the array)
            acc[:] = 0.0
            for s in range(counts[v]):
                k = samples[v, s, 2]
                ys = y + pad + samples[v, s, 1]
                x0 = pad + samples[v, s, 0]
                for x in range(width):
                    acc[x] += frames[k, ys, x0 + x]

            # Keep the largest weighted sum over the velocities for every pixel (the weight keeps the noise of
            #   the sums of all velocities the same, see velocitySamples)
            w = weights[v]
            for x in range(width):
                if acc[x]*w > best[x]:
                    best[x] = acc[x]*w
                    best_idx[x] = v

        # The mean instead of the sum, so the stacks of runs of different lengths are on the same scale (the
        #   amplitude of the object per frame)
        for x in range(width):
            out_max[y, x] = best[x]/n_frames
            out_idx[y, x] = best_idx[x]


# The threads of a block of the stack kernel on the GPU (_velocityStackRunsKernel): GPU_TX x GPU_TY threads, every
#   thread computing GPU_PX pixels of a row, GPU_TX px apart. The 32 threads of a warp are neighbours in a row, so
#   at every step they read 32 neighbouring pixels at once (one memory transaction). A block covers GPU_TX*GPU_PX
#   x GPU_TY pixels, and the pixels of a thread share the reading of the shifts and the computing of the addresses
GPU_TX = 32
GPU_TY = 8
GPU_PX = 4


if CUDA_AVAILABLE:

    @cuda.jit(cache=True)
    def _padKernel(src, dst, pad):
        """ Copy the frames into the zero-padded frames on the GPU: pad px on the top, bottom and left, and
            pad + GPU_TX*GPU_PX px on the right (the extra pixels of the last threads of a row).
        """

        # The pixel of the padded frames written by this thread (one thread per pixel, the threads beyond the
        #   padded frames do nothing)
        x, y, k = cuda.grid(3)
        if (k >= dst.shape[0]) or (y >= dst.shape[1]) or (x >= dst.shape[2]):
            return

        # The pixel of the frames it corresponds to, shifted by the padding
        ys = y - pad
        xs = x - pad

        # Inside the frames the pixel is copied, in the padding it is zero (a pixel shifted outside the image adds
        #   nothing to a stack)
        if (ys >= 0) and (ys < src.shape[1]) and (xs >= 0) and (xs < src.shape[2]):
            dst[k, y, x] = src[k, ys, xs]
        else:
            dst[k, y, x] = 0.0


    @cuda.jit(cache=True)
    def _velocityStackRunsKernel(frames, pad, run_starts, n_frames, samples, counts, weights, out_max, out_idx):
        """ The velocity stacks of several runs of a block of padded frames on the GPU: the run is the z index
            of the grid, and every thread computes GPU_PX pixels of a row, GPU_TX px apart. The sums are in single
            precision as on the CPU (velocityStackMaxCPU), and the padding removes the bounds checks.
        """

        # The run of this block of threads (the z index of the grid), the row of this thread, and the first of its
        #   pixels in the row: the thread computes the pixels x0, x0 + GPU_TX, ..., x0 + (GPU_PX - 1)*GPU_TX
        run = cuda.blockIdx.z
        y = cuda.blockIdx.y*GPU_TY + cuda.threadIdx.y
        x0 = cuda.blockIdx.x*GPU_TX*GPU_PX + cuda.threadIdx.x
        _, height, width = out_max.shape

        # The row in the padded frames. The threads beyond the last row compute the last row again, and don't
        #   write it
        yy = min(y, height - 1) + pad

        # The first frame of the run in the block
        k0 = run_starts[run]

        # For every pixel of the thread: the largest sum so far and the index of its velocity, and the sum of the
        #   current velocity (small arrays of a fixed size, kept in the registers of the thread)
        best = cuda.local.array(GPU_PX, numba.float32)
        best_idx = cuda.local.array(GPU_PX, numba.int32)
        acc = cuda.local.array(GPU_PX, numba.float32)

        # Before the first velocity, every sum is larger than the largest so far
        for j in range(GPU_PX):
            best[j] = numba.float32(-1e30)
            best_idx[j] = 0

        # Sum of the frames along every velocity, keeping the largest sum and its velocity for every pixel
        for v in range(samples.shape[0]):

            # The sums of this velocity start at zero (in single precision: a float literal would make the sums
            #   double precision, which consumer GPUs compute many times slower)
            for j in range(GPU_PX):
                acc[j] = numba.float32(0.0)

            # Every sample is a frame k of the run read shifted by the motion of the velocity v at its time (see
            #   velocitySamples), so an object moving with v adds up at its position at the middle of the run. The
            #   padding is at least the largest shift, so the shifted pixels are always inside the array (pixels
            #   shifted outside the image read zeros) and no bounds have to be checked
            for s in range(counts[v]):
                ys = yy + samples[v, s, 1]
                xs = x0 + pad + samples[v, s, 0]
                k = k0 + samples[v, s, 2]
                for j in range(GPU_PX):
                    acc[j] += frames[k, ys, xs + j*GPU_TX]

            # Keep the largest weighted sum of every pixel and its velocity (see velocitySamples)
            w = weights[v]
            for j in range(GPU_PX):
                if acc[j]*w > best[j]:
                    best[j] = acc[j]*w
                    best_idx[j] = v

        # The mean over the frames, as on the CPU, written for the pixels of the thread which are in the image (the
        #   last threads of a row also compute pixels in the padding on the right)
        if y < height:
            for j in range(GPU_PX):
                x = x0 + j*GPU_TX
                if x < width:
                    out_max[run, y, x] = best[j]/n_frames
                    out_idx[run, y, x] = best_idx[j]


    @cuda.jit(cache=True)
    def _velocityStackMaxKernel(frames, samples, counts, weights, out_max, out_idx):
        """ The GPU version of velocityStackMaxCPU: every thread computes one pixel of the output, for all
            velocities. The frames are not padded, pixels shifted outside the image are skipped (they add zero,
            as in the padded frames of the CPU version).
        """

        # The pixel of this thread (threads beyond the image do nothing)
        x, y = cuda.grid(2)
        n_frames, height, width = frames.shape
        if (x >= width) or (y >= height):
            return

        # Sum of the frames along every velocity, keeping the largest sum and its velocity
        best = -1e30
        best_idx = 0
        for v in range(samples.shape[0]):
            acc = 0.0
            for s in range(counts[v]):
                xs = x + samples[v, s, 0]
                ys = y + samples[v, s, 1]
                if (xs >= 0) and (xs < width) and (ys >= 0) and (ys < height):
                    acc += frames[samples[v, s, 2], ys, xs]
            acc *= weights[v]
            if acc > best:
                best = acc
                best_idx = v

        # The mean over the frames, as on the CPU
        out_max[y, x] = best/n_frames
        out_idx[y, x] = best_idx


class DeviceFrames(object):
    def __init__(self, frames, pad):
        """ Frames copied to the GPU once and zero-padded there, for the stacks of all runs and velocities of a
            block (see VelocityStacker.stackMaxRuns).

        Arguments:
            frames: [ndarray] float32 frames, shape (n_frames, height, width).
            pad: [int] Padding (px), at least the largest shift of the stacks which use the frames.
        """

        # The frames as one contiguous float32 array, and their size
        frames = np.ascontiguousarray(frames, dtype=np.float32)
        self.n_frames, self.height, self.width = frames.shape
        self.pad = int(pad)

        # The frames are copied to the GPU as they are, and padded there: pad px of zeros around them, and
        #   GPU_TX*GPU_PX px more on the right, which the last threads of a row of the stack kernel read (copying
        #   the padded frames would copy mostly zeros)
        d_src = cuda.to_device(frames)
        self.d_frames = cuda.device_array((self.n_frames, self.height + 2*self.pad,
                                           self.width + 2*self.pad + GPU_TX*GPU_PX), dtype=np.float32)

        # One thread per pixel of the padded frames, in blocks of 32x8 threads, a layer of blocks per frame
        threads = (32, 8, 1)
        blocks = ((self.d_frames.shape[2] + 31)//32, (self.d_frames.shape[1] + 7)//8, self.n_frames)
        _padKernel[blocks, threads](d_src, self.d_frames, self.pad)

        # The unpadded copy on the GPU is not needed any more
        del d_src


class VelocityStacker(object):
    def __init__(self, velocities, n_frames, use_gpu=False, streak=False, smooth_sigma=0.0):
        """ Velocity stacks of runs of n_frames frames, on the CPU or on the GPU.

        Arguments:
            velocities: [ndarray] Velocities (vx, vy) in px per frame, shape (n_vel, 2).
            n_frames: [int] Number of frames in a stack.

        Keyword arguments:
            use_gpu: [bool] Run on the GPU (only if CUDA is available). False by default.
            streak: [bool] Sum the streak of a fast object in every frame (see velocitySamples). False by
                default.
            smooth_sigma: [float] Sigma of the Gaussian smoothing of the frames (px), for the weights of the streak
                samples (see velocitySamples). 0 by default.
        """

        self.velocities = np.asarray(velocities, dtype=np.float64)
        self.n_frames = n_frames

        # The shifts of every frame for every velocity (at the middles of the frames), and the samples which the
        #   stacks sum, are the same for all runs, so they are computed once
        self.shifts = velocityShifts(self.velocities, n_frames)
        self.streak = streak
        self.samples, self.counts, self.weights = velocitySamples(self.velocities, n_frames, streak=streak,
                                                                  smooth_sigma=smooth_sigma)

        # The padding of the frames on the CPU: the largest shift, so no shifted pixel is outside the array
        self.pad = int(np.abs(self.samples[:, :, :2]).max()) if self.samples.size else 0

        self.use_gpu = use_gpu and CUDA_AVAILABLE

        # On the GPU, the samples are copied to the GPU memory once
        if self.use_gpu:
            self.d_samples = cuda.to_device(self.samples)
            self.d_counts = cuda.to_device(self.counts)
            self.d_weights = cuda.to_device(self.weights)


    def setSmoothing(self, smooth_sigma):
        """ Set the sigma of the Gaussian smoothing of the frames, which the weights of the streak samples depend on
            (see velocitySamples). The samples don't change, only their weights.

        Arguments:
            smooth_sigma: [float] Sigma of the smoothing (px).
        """

        # The weights of the same samples for the new smoothing
        _, _, self.weights = velocitySamples(self.velocities, self.n_frames, streak=self.streak,
                                             smooth_sigma=smooth_sigma)

        # On the GPU, the weights are copied to the GPU memory again
        if self.use_gpu:
            self.d_weights = cuda.to_device(self.weights)


    def stackMax(self, frames):
        """ The largest mean along the velocities of every pixel, and the index of that velocity.

        Arguments:
            frames: [ndarray] float32 frames, shape (n_frames, height, width).

        Return:
            (stack_max, vel_idx): [tuple of ndarrays] float32 and int32 images.
        """

        frames = np.ascontiguousarray(frames, dtype=np.float32)
        _, height, width = frames.shape
        out_max = np.empty((height, width), dtype=np.float32)
        out_idx = np.empty((height, width), dtype=np.int32)

        if self.use_gpu:

            # Copy the frames to the GPU, run one thread per pixel in blocks of 16x16 threads (enough blocks to
            #   cover the image), and copy the results back
            d_frames = cuda.to_device(frames)
            d_max = cuda.device_array((height, width), dtype=np.float32)
            d_idx = cuda.device_array((height, width), dtype=np.int32)
            threads = (16, 16)
            blocks = ((width + threads[0] - 1)//threads[0], (height + threads[1] - 1)//threads[1])
            _velocityStackMaxKernel[blocks, threads](d_frames, self.d_samples, self.d_counts, self.d_weights, d_max,
                                                     d_idx)
            d_max.copy_to_host(out_max)
            d_idx.copy_to_host(out_idx)

        else:

            # Pad the frames with zeros, so the shifted rows never leave the array (see velocityStackMaxCPU)
            pad = self.pad
            padded = np.zeros((frames.shape[0], height + 2*pad, width + 2*pad), dtype=np.float32)
            padded[:, pad:pad + height, pad:pad + width] = frames
            velocityStackMaxCPU(padded, pad, self.samples, self.counts, self.weights, out_max, out_idx)

        return out_max, out_idx


    def stackMaxRuns(self, frames, run_starts):
        """ stackMax of several runs of n_frames frames of a block at once.

        Arguments:
            frames: [ndarray or DeviceFrames] float32 frames of the block, shape (n, height, width), or the
                block already on the GPU (padded by at least the largest shift of this stacker).
            run_starts: [list] Index of the first frame of every run.

        Return:
            (stack_max, vel_idx): [tuple of ndarrays] float32 and int32 images of every run, shape
                (n_runs, height, width).
        """

        # The number of runs to stack
        n_runs = len(run_starts)

        # On the CPU (or without runs), run by run, as stackMax computes them. The frames are then an array on the
        #   CPU (frames on the GPU are only given to stackers on the GPU)
        if (not self.use_gpu) or (n_runs == 0):
            frames = frames if isinstance(frames, np.ndarray) else None
            results = [self.stackMax(frames[r0:r0 + self.n_frames]) for r0 in run_starts]

            # Without runs, empty stacks
            if not results:
                return np.empty((0, 0, 0), np.float32), np.empty((0, 0, 0), np.int32)

            # The stacks of the runs as one array of every output
            return np.stack([r[0] for r in results]), np.stack([r[1] for r in results])

        # On the GPU: the frames are copied once (unless they are already there), all runs are stacked by one
        #   launch, and the results are copied back once
        d_frames = frames if isinstance(frames, DeviceFrames) else DeviceFrames(frames, self.pad)

        # The padding of the frames on the GPU has to hold the largest shift of this stacker
        if d_frames.pad < self.pad:
            raise ValueError('The frames on the GPU are padded by {} px, the stacks need {}'.format(d_frames.pad,
                                                                                                   self.pad))

        # The first frames of the runs, and the outputs of all runs, on the GPU
        height, width = d_frames.height, d_frames.width
        d_starts = cuda.to_device(np.asarray(run_starts, dtype=np.int32))
        d_max = cuda.device_array((n_runs, height, width), dtype=np.float32)
        d_idx = cuda.device_array((n_runs, height, width), dtype=np.int32)

        # Blocks of GPU_TX x GPU_TY threads, every thread computing GPU_PX pixels of a row: enough blocks to cover
        #   the image, and a layer of blocks per run
        blocks = ((width + GPU_TX*GPU_PX - 1)//(GPU_TX*GPU_PX), (height + GPU_TY - 1)//GPU_TY, n_runs)
        _velocityStackRunsKernel[blocks, (GPU_TX, GPU_TY)](d_frames.d_frames, d_frames.pad, d_starts, self.n_frames,
                                                            self.d_samples, self.d_counts, self.d_weights, d_max,
                                                            d_idx)

        # The stacks of all runs copied back at once (which waits for the kernel to finish)
        return d_max.copy_to_host(), d_idx.copy_to_host()



@numba.njit(cache=True, nogil=True)
def _movingPSFModel(params, frames, skip, x_pred, y_pred, dt, vx, vy, sigma, n_sub, radius, jac, residuals,
                    compute):
    """ Residuals (data - model) and Jacobian of a moving Gaussian PSF over the patches of the frames.
        Returns the number of pixels used. See fitMovingPSF for the arguments and the model.

    The model of the pixel (i, j) in the frame k is

        m_k(i, j) = bg + amp*G_k(i, j)

    where G_k is the PSF smeared along the motion during the frame: the mean of n_sub circular Gaussians of
    sigma, with peak 1, at the positions along the segment the object moved along during the frame,

        G_k(i, j) = (1/n_sub) * sum_s exp(-((i - x_ks)^2 + (j - y_ks)^2)/(2 sigma^2)),
        x_ks = x_mid + vx*(dt_k + f_s), y_ks = y_mid + vy*(dt_k + f_s), f_s from -0.5 to 0.5 (frames)

    The derivatives of the model by the parameters (x_mid, y_mid, amp, bg), the columns of the Jacobian, are

        dm/dx_mid = amp * (1/n_sub) * sum_s e_s*(i - x_ks)/sigma^2      (e_s is the Gaussian of the step s)
        dm/dy_mid = amp * (1/n_sub) * sum_s e_s*(j - y_ks)/sigma^2
        dm/damp   = G_k(i, j)
        dm/dbg    = 1

    With compute=False, only the pixels are counted (to allocate the arrays).
    """

    x_mid, y_mid, amp, bg = params[0], params[1], params[2], params[3]
    n_frames, height, width = frames.shape
    inv_s2 = 1.0/(sigma*sigma)
    n = 0

    for k in range(n_frames):

        # The patch of the frame is centred on the predicted position of the object in this frame (it doesn't
        #   move with the fitted position, so the same pixels are used in every iteration); the model is centred
        #   on the fitted position
        xc = x_pred + vx*dt[k]
        yc = y_pred + vy*dt[k]
        xm = x_mid + vx*dt[k]
        ym = y_mid + vy*dt[k]

        # The square patch of (2*radius + 1) px around the predicted position
        x0 = int(math.floor(xc - radius + 0.5))
        y0 = int(math.floor(yc - radius + 0.5))
        size = 2*radius + 1

        for iy in range(y0, y0 + size):
            if (iy < 0) or (iy >= height):
                continue
            for ix in range(x0, x0 + size):

                # Pixels outside the image and the skipped pixels (e.g. saturated) are not used
                if (ix < 0) or (ix >= width) or (skip[k, iy, ix] != 0):
                    continue

                # The streak is the mean of n_sub Gaussians along the motion during the frame (see above): the
                #   value g of the smeared PSF at the pixel, and the sums gx, gy of its derivatives by the
                #   position (without the amplitude)
                g = 0.0
                gx = 0.0
                gy = 0.0
                for s in range(n_sub):

                    # Time of the step within the frame, from -0.5 to 0.5 (only the middle for one step)
                    f = 0.0 if n_sub == 1 else (s/(n_sub - 1.0) - 0.5)

                    # Distance of the pixel from the position of the object at this step
                    dx = ix - (xm + vx*f)
                    dy = iy - (ym + vy*f)

                    # The Gaussian, and its derivative by the centre: d/dx exp(-(dx^2 + dy^2)/(2 s^2)) =
                    #   exp(...)*dx/s^2 (dx = ix - x_centre, so moving the centre by +1 increases the term)
                    e = math.exp(-0.5*(dx*dx + dy*dy)*inv_s2)
                    g += e
                    gx += e*dx
                    gy += e*dy
                g /= n_sub
                gx *= inv_s2/n_sub
                gy *= inv_s2/n_sub

                if compute:

                    # Residual of the pixel (data - model) and the derivatives of the model by
                    #   (x_mid, y_mid, amp, bg)
                    residuals[n] = frames[k, iy, ix] - (bg + amp*g)
                    jac[n, 0] = amp*gx
                    jac[n, 1] = amp*gy
                    jac[n, 2] = g
                    jac[n, 3] = 1.0

                n += 1

    return n


@numba.njit(cache=True, nogil=True)
def fitMovingPSF(frames, skip, x_pred, y_pred, dt, vx, vy, sigma, radius, max_iter=20):
    """ Fit a moving Gaussian PSF jointly to patches of a run of frames.

    The object is modelled in frame k at (x_mid, y_mid) + v*dt[k], smeared along the motion during the frame,
    with a fixed PSF width. The free parameters are the position at the middle time, the peak amplitude of
    the PSF and the background. The frames are normalized to unit noise, so the fit is unweighted and the
    covariance is in the units of the data.

    Fitting all frames of the run jointly with the known velocity uses all the light of the object, like the
    velocity stack, but at the full resolution of the frames and without rounding the shifts: the position error
    of an object of amplitude A (in the noise of a frame) over N frames is about sigma/(A*sqrt(N)) times a factor
    of order one.

    The fit is a Levenberg-Marquardt minimization of the sum of the squared residuals (data - model) over the
    pixels of all patches (see _movingPSFModel for the model and its derivatives).

    Arguments:
        frames: [ndarray] float32 frames normalized to unit noise, shape (n_frames, height, width).
        skip: [ndarray] uint8 mask of the pixels left out of the fit (e.g. saturated), same shape as frames.
        x_pred, y_pred: [float] Predicted position at the middle time (px).
        dt: [ndarray] Time of every frame relative to the middle time (frames).
        vx, vy: [float] Velocity (px per frame).
        sigma: [float] PSF sigma (px).
        radius: [int] Half size of the patch around the predicted position in every frame (px).

    Keyword arguments:
        max_iter: [int] Maximum number of Levenberg-Marquardt iterations. 20 by default.

    Return:
        result: [ndarray] [x_mid, y_mid, amp, bg, sigma_x, sigma_y, sigma_amp, reduced chi2, n_pixels,
            converged].
    """

    result = np.full(10, np.nan)

    # The streak during a frame is modelled with steps at most 0.5 px apart along the motion; an object moving
    #   less than 0.5 px per frame is modelled as a point
    speed = math.sqrt(vx*vx + vy*vy)
    n_sub = max(1, int(math.ceil(speed/0.5)) + 1) if speed > 0.5 else 1

    # Count the pixels of the patches (outside the image and skipped pixels are left out). Too few pixels can't
    #   constrain four parameters
    dummy_jac = np.empty((1, 4))
    dummy_res = np.empty(1)
    params = np.array([x_pred, y_pred, 0.0, 0.0])
    n_pix = _movingPSFModel(params, frames, skip, x_pred, y_pred, dt, vx, vy, sigma, n_sub, radius, dummy_jac,
                            dummy_res, False)
    if n_pix < 10:
        return result

    jac = np.empty((n_pix, 4))
    res = np.empty(n_pix)


    ### Initial parameters ###

    # With amp = 0 and bg = 0 the residuals are the data: the background is their median, and the amplitude the
    #   linear least squares amplitude of the PSF at the predicted position, amp = sum(G*(d - bg))/sum(G^2)
    #   (jac[:, 2] is G)
    _movingPSFModel(params, frames, skip, x_pred, y_pred, dt, vx, vy, sigma, n_sub, radius, jac, res, True)
    bg0 = np.median(res)
    params[3] = bg0
    num = 0.0
    den = 0.0
    for i in range(n_pix):
        num += jac[i, 2]*(res[i] - bg0)
        den += jac[i, 2]*jac[i, 2]
    params[2] = max(num/den, 0.1) if den > 0 else 1.0

    ### ###


    ### Levenberg-Marquardt iterations ###

    # The residuals and the Jacobian at the initial parameters, and the cost (sum of the squared residuals)
    _movingPSFModel(params, frames, skip, x_pred, y_pred, dt, vx, vy, sigma, n_sub, radius, jac, res, True)
    cost = np.sum(res*res)
    lam = 1e-3
    converged = False

    for _ in range(max_iter):

        # The Gauss-Newton step solves (J^T J) step = J^T r (r = data - model, so the step moves the model towards
        #   the data). Levenberg-Marquardt damps it by scaling the diagonal: (J^T J + lam*diag(J^T J)) step = J^T r,
        #   a short gradient step for a large lam, the Gauss-Newton step for a small one
        jtj = jac.T @ jac
        jtr = jac.T @ res
        a = jtj.copy()
        for i in range(4):
            a[i, i] *= (1.0 + lam)

        try:
            step = np.linalg.solve(a, jtr)
        except Exception:
            break

        new_params = params + step

        # The position has to stay within the patch around the prediction, otherwise the fit would move to
        #   another source: the step is rejected, and a more damped one is tried
        if (abs(new_params[0] - x_pred) > radius) or (abs(new_params[1] - y_pred) > radius):
            lam *= 10.0
            continue

        # The residuals and the Jacobian at the new parameters
        new_jac = np.empty((n_pix, 4))
        new_res = np.empty(n_pix)
        _movingPSFModel(new_params, frames, skip, x_pred, y_pred, dt, vx, vy, sigma, n_sub, radius, new_jac, new_res,
                        True)
        new_cost = np.sum(new_res*new_res)

        if new_cost < cost:

            # The step improves the fit: accept it and trust the quadratic model more (smaller damping). The fit
            #   has converged when the position changes by less than 0.001 px
            small = (abs(step[0]) < 1e-3) and (abs(step[1]) < 1e-3)
            params = new_params
            jac = new_jac
            res = new_res
            cost = new_cost
            lam = max(lam/10.0, 1e-7)
            if small:
                converged = True
                break
        else:

            # The step doesn't improve the fit: more damping. When even very short steps don't improve it, the
            #   fit is at the minimum
            lam *= 10.0
            if lam > 1e6:
                converged = True
                break

    ### ###


    ### Uncertainties ###

    # The covariance of the parameters is (J^T J)^-1 scaled by the reduced chi^2 (the variance of the residuals
    #   per degree of freedom), so the uncertainties follow the actual scatter of the data around the model (for
    #   frames normalized to unit noise, the reduced chi^2 is about 1 if the model fits)
    dof = max(n_pix - 4, 1)
    chi2 = cost/dof
    jtj = jac.T @ jac
    try:
        cov = np.linalg.inv(jtj)*chi2
    except Exception:
        return result

    ### ###

    result[0] = params[0]
    result[1] = params[1]
    result[2] = params[2]
    result[3] = params[3]
    result[4] = math.sqrt(max(cov[0, 0], 0.0))
    result[5] = math.sqrt(max(cov[1, 1], 0.0))
    result[6] = math.sqrt(max(cov[2, 2], 0.0))
    result[7] = chi2
    result[8] = n_pix
    result[9] = 1.0 if converged else 0.0

    return result



@numba.njit(cache=True, nogil=True)
def forcedTrackSignal(frames, frame_idx, xs, ys, sigma, radius, excluded, vx=0.0, vy=0.0):
    """ PSF-weighted sums along a fixed track, for the combined significance of a detection. For every frame,
        the sum of the PSF-weighted data and the sum of the squared weights at the given position.

    For frames normalized to unit noise (z), the optimal (matched-filter) estimate of the amplitude of a PSF g at a
    known position is A = sum(g*z)/sum(g^2), with the noise 1/sqrt(sum(g^2)), so the signal-to-noise ratio of the
    object summed over all frames of a track is sum(g*z)/sqrt(sum(g^2)). The two sums are returned separately,
    so sums of different sets of positions (on the track and off it) can be combined (see
    MatchedFilterDetector.verify).

    Arguments:
        frames: [ndarray] float32 frames normalized to unit noise, shape (n, height, width), e.g. a block.
        frame_idx: [ndarray] Indices of the frames of the track in frames (the frames are not copied).
        xs, ys: [ndarray] Position of the object in every frame of frame_idx (px).
        sigma: [float] PSF sigma (px).
        radius: [int] Half size of the patch (px).
        excluded: [ndarray] uint8 image, pixels which are not 0 are left out of the sums (e.g. stars).

    Keyword arguments:
        vx, vy: [float] Motion of the object during a frame (px). An object which moves during the frame is a
            streak: the PSF is then smeared along the segment it moves along, centred on the position (the
            radius has to include half the segment). 0 by default (a point source).

    Return:
        (sum_gz, sum_gg): [tuple of float] Sums over all frames.
    """

    n_frames = len(frame_idx)
    height, width = frames.shape[1], frames.shape[2]
    inv_s2 = 1.0/(sigma*sigma)
    sum_gz = 0.0
    sum_gg = 0.0

    # The streak of a moving object: its length and direction. The PSF smeared along a segment of length L is a
    #   Gaussian across the segment times the difference of the error functions at its ends along it (shorter
    #   than 0.1 px it is the PSF itself)
    length = math.sqrt(vx*vx + vy*vy)
    streak = length > 0.1
    ux = vx/length if streak else 1.0
    uy = vy/length if streak else 0.0
    erf_scale = 1.0/(math.sqrt(2.0)*sigma)

    for k in range(n_frames):

        # The frame of the k-th position in the frames (e.g. a frame of the block)
        f = frame_idx[k]

        # The patch of (2*radius + 1) px around the position in this frame
        x0 = int(math.floor(xs[k] + 0.5))
        y0 = int(math.floor(ys[k] + 0.5))
        for iy in range(y0 - radius, y0 + radius + 1):
            if (iy < 0) or (iy >= height):
                continue
            for ix in range(x0 - radius, x0 + radius + 1):
                if (ix < 0) or (ix >= width) or (excluded[iy, ix] != 0):
                    continue

                # The weight of the pixel is the PSF (a Gaussian with peak 1) centred on the position, or the
                #   PSF smeared along the streak (the scale of the weights doesn't change the signal-to-noise
                #   ratio)
                dx = ix - xs[k]
                dy = iy - ys[k]
                if streak:
                    along = dx*ux + dy*uy
                    across = -dx*uy + dy*ux
                    g = math.exp(-0.5*across*across*inv_s2)*0.5*(math.erf((0.5*length - along)*erf_scale)
                                                                - math.erf((-0.5*length - along)*erf_scale))
                else:
                    g = math.exp(-0.5*(dx*dx + dy*dy)*inv_s2)
                sum_gz += g*frames[f, iy, ix]
                sum_gg += g*g

    return sum_gz, sum_gg



@numba.njit(cache=True, nogil=True)
def streakAperture(frames, saturated, frame_idx, noise, xs, ys, vx, vy, radius):
    """ Photometry of a moving object: in every frame, the sum of the background-subtracted pixels within radius
        of the segment the object moved along during the frame, and the number of saturated pixels there.

    The aperture is a "stadium": the set of the pixels whose distance from the segment A-B (the positions of the
    object at the beginning and at the end of the frame) is at most the radius. For a moving object, a circular
    aperture would cut off the ends of the streak.

    Arguments:
        frames: [ndarray] float32 frames normalized to unit noise (background subtracted), (n, height, width),
            e.g. a block.
        saturated: [ndarray] uint8 mask of the saturated pixels in the raw frames, same shape.
        frame_idx: [ndarray] Indices of the frames of the object in frames (the frames are not copied).
        noise: [ndarray] float32 noise image (ADU), to convert the normalized frames back to ADU.
        xs, ys: [ndarray] Position of the object in the middle of every frame of frame_idx (px).
        vx, vy: [float] Velocity (px per frame).
        radius: [float] Radius of the aperture around the segment (px).

    Return:
        (sums, n_saturated): [tuple of ndarrays] Sum in ADU and saturated pixel count per frame.
    """

    n_frames = len(frame_idx)
    height, width = frames.shape[1], frames.shape[2]
    sums = np.zeros(n_frames)
    n_sat = np.zeros(n_frames, dtype=np.int64)

    # Squared length of the segment, and the farthest a pixel of the aperture can be from the middle position (half
    #   the segment plus the radius), which bounds the box of pixels to look at
    seg2 = vx*vx + vy*vy
    reach = radius + 0.5*math.sqrt(seg2)

    for k in range(n_frames):

        # The frame of the k-th position in the frames (e.g. a frame of the block)
        f = frame_idx[k]

        # The beginning of the segment: the position half a frame before the middle of the frame
        ax = xs[k] - 0.5*vx
        ay = ys[k] - 0.5*vy

        for iy in range(int(math.floor(ys[k] - reach)), int(math.ceil(ys[k] + reach)) + 1):
            if (iy < 0) or (iy >= height):
                continue
            for ix in range(int(math.floor(xs[k] - reach)), int(math.ceil(xs[k] + reach)) + 1):
                if (ix < 0) or (ix >= width):
                    continue

                # Distance of the pixel from the segment: the projection of the pixel on the line through the
                #   segment is at the fraction t = ((P - A).v)/|v|^2 of the segment, limited to the segment
                #   (0 <= t <= 1), and the distance is from the pixel to that point
                t = 0.0
                if seg2 > 0:
                    t = ((ix - ax)*vx + (iy - ay)*vy)/seg2
                    t = min(max(t, 0.0), 1.0)
                dx = ix - (ax + t*vx)
                dy = iy - (ay + t*vy)
                if dx*dx + dy*dy > radius*radius:
                    continue

                # The normalized pixel times the noise is the background-subtracted pixel in ADU
                sums[k] += frames[f, iy, ix]*noise[iy, ix]
                if saturated[f, iy, ix] != 0:
                    n_sat[k] += 1

    return sums, n_sat
