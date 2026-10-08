""" Compiled kernels of the matched-filter detection: the shift-and-add stacks along a grid of velocities, and
the joint fit of a moving PSF to a short run of frames.

The kernels are compiled with numba. The velocity stacks also have a CUDA version, used when numba can reach a
GPU.
"""

from __future__ import print_function, division, absolute_import

import math

import numpy as np

import numba

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

    dt = np.arange(n_frames) - (n_frames - 1)/2.0
    shifts = np.rint(velocities[:, None, :]*dt[None, :, None]).astype(np.int32)

    return np.ascontiguousarray(shifts)


@numba.njit(parallel=True, fastmath=True, cache=True)
def velocityStackMaxCPU(frames, pad, shifts, out_max, out_idx):
    """ For every pixel, the largest mean of the frames along the velocities, and the index of that velocity.
        The frames are zero-padded, so whole rows are summed for every velocity without bounds checks (pixels
        shifted outside the image add zero).

    Arguments:
        frames: [ndarray] float32 frames of the stack, padded by pad px on every side, shape
            (n_frames, height + 2*pad, width + 2*pad).
        pad: [int] Padding (px), at least the largest shift.
        shifts: [ndarray] int32 shifts of every frame for every velocity, shape (n_vel, n_frames, 2), as
            given by velocityShifts.
        out_max: [ndarray] float32 output image of the largest mean (height, width).
        out_idx: [ndarray] int32 output image of the index of the velocity with the largest mean.
    """

    n_frames = frames.shape[0]
    height, width = out_max.shape
    n_vel = shifts.shape[0]

    for y in numba.prange(height):

        best = np.full(width, -1e30, dtype=np.float32)
        best_idx = np.zeros(width, dtype=np.int32)
        acc = np.empty(width, dtype=np.float32)

        for v in range(n_vel):

            acc[:] = 0.0
            for k in range(n_frames):
                ys = y + pad + shifts[v, k, 1]
                x0 = pad + shifts[v, k, 0]
                for x in range(width):
                    acc[x] += frames[k, ys, x0 + x]

            for x in range(width):
                if acc[x] > best[x]:
                    best[x] = acc[x]
                    best_idx[x] = v

        for x in range(width):
            out_max[y, x] = best[x]/n_frames
            out_idx[y, x] = best_idx[x]


if CUDA_AVAILABLE:

    @cuda.jit
    def _velocityStackMaxKernel(frames, shifts, out_max, out_idx):

        x, y = cuda.grid(2)
        n_frames, height, width = frames.shape
        if (x >= width) or (y >= height):
            return

        best = -1e30
        best_idx = 0
        for v in range(shifts.shape[0]):
            acc = 0.0
            for k in range(n_frames):
                xs = x + shifts[v, k, 0]
                ys = y + shifts[v, k, 1]
                if (xs >= 0) and (xs < width) and (ys >= 0) and (ys < height):
                    acc += frames[k, ys, xs]
            if acc > best:
                best = acc
                best_idx = v

        out_max[y, x] = best/n_frames
        out_idx[y, x] = best_idx


class VelocityStacker(object):
    def __init__(self, velocities, n_frames, use_gpu=False):
        """ Velocity stacks of runs of n_frames frames, on the CPU or on the GPU.

        Arguments:
            velocities: [ndarray] Velocities (vx, vy) in px per frame, shape (n_vel, 2).
            n_frames: [int] Number of frames in a stack.

        Keyword arguments:
            use_gpu: [bool] Run on the GPU (only if CUDA is available). False by default.
        """

        self.velocities = np.asarray(velocities, dtype=np.float64)
        self.n_frames = n_frames
        self.shifts = velocityShifts(self.velocities, n_frames)
        self.pad = int(np.abs(self.shifts).max()) if self.shifts.size else 0
        self.use_gpu = use_gpu and CUDA_AVAILABLE

        if self.use_gpu:
            self.d_shifts = cuda.to_device(self.shifts)


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
            d_frames = cuda.to_device(frames)
            d_max = cuda.device_array((height, width), dtype=np.float32)
            d_idx = cuda.device_array((height, width), dtype=np.int32)
            threads = (16, 16)
            blocks = ((width + threads[0] - 1)//threads[0], (height + threads[1] - 1)//threads[1])
            _velocityStackMaxKernel[blocks, threads](d_frames, self.d_shifts, d_max, d_idx)
            d_max.copy_to_host(out_max)
            d_idx.copy_to_host(out_idx)

        else:
            pad = self.pad
            padded = np.zeros((frames.shape[0], height + 2*pad, width + 2*pad), dtype=np.float32)
            padded[:, pad:pad + height, pad:pad + width] = frames
            velocityStackMaxCPU(padded, pad, self.shifts, out_max, out_idx)

        return out_max, out_idx



@numba.njit(cache=True)
def _movingPSFModel(params, frames, x_pred, y_pred, dt, vx, vy, sigma, n_sub, radius, jac, residuals, compute):
    """ Residuals (data - model) and Jacobian of a moving Gaussian PSF over the patches of the frames.
        Returns the number of pixels used. See fitMovingPSF for the arguments. """

    x_mid, y_mid, amp, bg = params[0], params[1], params[2], params[3]
    n_frames, height, width = frames.shape
    inv_s2 = 1.0/(sigma*sigma)
    n = 0

    for k in range(n_frames):

        # Predicted centre of the patch of this frame, and the modelled centre
        xc = x_pred + vx*dt[k]
        yc = y_pred + vy*dt[k]
        xm = x_mid + vx*dt[k]
        ym = y_mid + vy*dt[k]

        x0 = int(math.floor(xc - radius + 0.5))
        y0 = int(math.floor(yc - radius + 0.5))
        size = 2*radius + 1

        for iy in range(y0, y0 + size):
            if (iy < 0) or (iy >= height):
                continue
            for ix in range(x0, x0 + size):
                if (ix < 0) or (ix >= width):
                    continue

                # The streak is the mean of n_sub Gaussians along the motion during the frame
                g = 0.0
                gx = 0.0
                gy = 0.0
                for s in range(n_sub):
                    f = 0.0 if n_sub == 1 else (s/(n_sub - 1.0) - 0.5)
                    dx = ix - (xm + vx*f)
                    dy = iy - (ym + vy*f)
                    e = math.exp(-0.5*(dx*dx + dy*dy)*inv_s2)
                    g += e
                    gx += e*dx
                    gy += e*dy
                g /= n_sub
                gx *= inv_s2/n_sub
                gy *= inv_s2/n_sub

                if compute:
                    residuals[n] = frames[k, iy, ix] - (bg + amp*g)
                    jac[n, 0] = amp*gx
                    jac[n, 1] = amp*gy
                    jac[n, 2] = g
                    jac[n, 3] = 1.0

                n += 1

    return n


@numba.njit(cache=True)
def fitMovingPSF(frames, x_pred, y_pred, dt, vx, vy, sigma, radius, max_iter=20):
    """ Fit a moving Gaussian PSF jointly to patches of a run of frames.

    The object is modelled in frame k at (x_mid, y_mid) + v*dt[k], smeared along the motion during the frame,
    with a fixed PSF width. The free parameters are the position at the middle time, the peak amplitude of
    the PSF and the background. The frames are normalized to unit noise, so the fit is unweighted and the
    covariance is in the units of the data.

    Arguments:
        frames: [ndarray] float32 frames normalized to unit noise, shape (n_frames, height, width).
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

    # The number of sub-steps along the motion during a frame, at most 0.5 px apart
    speed = math.sqrt(vx*vx + vy*vy)
    n_sub = max(1, int(math.ceil(speed/0.5)) + 1) if speed > 0.5 else 1

    dummy_jac = np.empty((1, 4))
    dummy_res = np.empty(1)
    params = np.array([x_pred, y_pred, 0.0, 0.0])
    n_pix = _movingPSFModel(params, frames, x_pred, y_pred, dt, vx, vy, sigma, n_sub, radius, dummy_jac,
                            dummy_res, False)
    if n_pix < 10:
        return result

    jac = np.empty((n_pix, 4))
    res = np.empty(n_pix)

    # Initial amplitude and background from the data
    _movingPSFModel(params, frames, x_pred, y_pred, dt, vx, vy, sigma, n_sub, radius, jac, res, True)
    bg0 = np.median(res)
    params[3] = bg0
    num = 0.0
    den = 0.0
    for i in range(n_pix):
        num += jac[i, 2]*(res[i] - bg0)
        den += jac[i, 2]*jac[i, 2]
    params[2] = max(num/den, 0.1) if den > 0 else 1.0

    _movingPSFModel(params, frames, x_pred, y_pred, dt, vx, vy, sigma, n_sub, radius, jac, res, True)
    cost = np.sum(res*res)
    lam = 1e-3
    converged = False

    for _ in range(max_iter):

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

        # Keep the position near the prediction
        if (abs(new_params[0] - x_pred) > radius) or (abs(new_params[1] - y_pred) > radius):
            lam *= 10.0
            continue

        new_jac = np.empty((n_pix, 4))
        new_res = np.empty(n_pix)
        _movingPSFModel(new_params, frames, x_pred, y_pred, dt, vx, vy, sigma, n_sub, radius, new_jac, new_res,
                        True)
        new_cost = np.sum(new_res*new_res)

        if new_cost < cost:
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
            lam *= 10.0
            if lam > 1e6:
                converged = True
                break

    dof = max(n_pix - 4, 1)
    chi2 = cost/dof
    jtj = jac.T @ jac
    try:
        cov = np.linalg.inv(jtj)*chi2
    except Exception:
        return result

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



@numba.njit(cache=True)
def forcedTrackSignal(frames, xs, ys, sigma, radius, excluded):
    """ PSF-weighted sums along a fixed track, for the combined significance of a detection. For every frame,
        the sum of the PSF-weighted data and the sum of the squared weights at the given position.

    Arguments:
        frames: [ndarray] float32 frames normalized to unit noise, shape (n_frames, height, width).
        xs, ys: [ndarray] Position of the object in every frame (px).
        sigma: [float] PSF sigma (px).
        radius: [int] Half size of the patch (px).
        excluded: [ndarray] uint8 image, pixels which are not 0 are left out of the sums (e.g. stars).

    Return:
        (sum_gz, sum_gg): [tuple of float] Sums over all frames.
    """

    n_frames, height, width = frames.shape
    inv_s2 = 1.0/(sigma*sigma)
    sum_gz = 0.0
    sum_gg = 0.0

    for k in range(n_frames):
        x0 = int(math.floor(xs[k] + 0.5))
        y0 = int(math.floor(ys[k] + 0.5))
        for iy in range(y0 - radius, y0 + radius + 1):
            if (iy < 0) or (iy >= height):
                continue
            for ix in range(x0 - radius, x0 + radius + 1):
                if (ix < 0) or (ix >= width) or (excluded[iy, ix] != 0):
                    continue
                dx = ix - xs[k]
                dy = iy - ys[k]
                g = math.exp(-0.5*(dx*dx + dy*dy)*inv_s2)
                sum_gz += g*frames[k, iy, ix]
                sum_gg += g*g

    return sum_gz, sum_gg
