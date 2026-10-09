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
            for k in range(n_frames):
                ys = y + pad + shifts[v, k, 1]
                x0 = pad + shifts[v, k, 0]
                for x in range(width):
                    acc[x] += frames[k, ys, x0 + x]

            # Keep the largest sum over the velocities for every pixel
            for x in range(width):
                if acc[x] > best[x]:
                    best[x] = acc[x]
                    best_idx[x] = v

        # The mean instead of the sum, so the stacks of runs of different lengths are on the same scale (the
        #   amplitude of the object per frame)
        for x in range(width):
            out_max[y, x] = best[x]/n_frames
            out_idx[y, x] = best_idx[x]


if CUDA_AVAILABLE:

    @cuda.jit
    def _velocityStackMaxKernel(frames, shifts, out_max, out_idx):
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

        # The mean over the frames, as on the CPU
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

        # The shifts of every frame for every velocity are the same for all runs, so they are computed once
        self.shifts = velocityShifts(self.velocities, n_frames)

        # The padding of the frames on the CPU: the largest shift, so no shifted pixel is outside the array
        self.pad = int(np.abs(self.shifts).max()) if self.shifts.size else 0

        self.use_gpu = use_gpu and CUDA_AVAILABLE

        # On the GPU, the shifts are copied to the GPU memory once
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

            # Copy the frames to the GPU, run one thread per pixel in blocks of 16x16 threads (enough blocks to
            #   cover the image), and copy the results back
            d_frames = cuda.to_device(frames)
            d_max = cuda.device_array((height, width), dtype=np.float32)
            d_idx = cuda.device_array((height, width), dtype=np.int32)
            threads = (16, 16)
            blocks = ((width + threads[0] - 1)//threads[0], (height + threads[1] - 1)//threads[1])
            _velocityStackMaxKernel[blocks, threads](d_frames, self.d_shifts, d_max, d_idx)
            d_max.copy_to_host(out_max)
            d_idx.copy_to_host(out_idx)

        else:

            # Pad the frames with zeros, so the shifted rows never leave the array (see velocityStackMaxCPU)
            pad = self.pad
            padded = np.zeros((frames.shape[0], height + 2*pad, width + 2*pad), dtype=np.float32)
            padded[:, pad:pad + height, pad:pad + width] = frames
            velocityStackMaxCPU(padded, pad, self.shifts, out_max, out_idx)

        return out_max, out_idx



@numba.njit(cache=True)
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


@numba.njit(cache=True)
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



@numba.njit(cache=True)
def forcedTrackSignal(frames, frame_idx, xs, ys, sigma, radius, excluded):
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

    Return:
        (sum_gz, sum_gg): [tuple of float] Sums over all frames.
    """

    n_frames = len(frame_idx)
    height, width = frames.shape[1], frames.shape[2]
    inv_s2 = 1.0/(sigma*sigma)
    sum_gz = 0.0
    sum_gg = 0.0

    for k in range(n_frames):
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

                # The weight of the pixel is the PSF (a Gaussian with peak 1) centred on the position
                dx = ix - xs[k]
                dy = iy - ys[k]
                g = math.exp(-0.5*(dx*dx + dy*dy)*inv_s2)
                sum_gz += g*frames[f, iy, ix]
                sum_gg += g*g

    return sum_gz, sum_gg



@numba.njit(cache=True)
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
