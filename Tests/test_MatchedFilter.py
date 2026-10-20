"""
Tests for the matched-filter detection of faint moving objects (RMS.MatchedFilterDetection and
RMS.Routines.MatchedFilterKernels), on synthetic frames: a constant sky with noise, static stars, and objects
moving along straight or curved tracks well below the single-frame detection limit.

The tests are in groups:
    - Kernels: the velocity grid, the binning, the velocity stack (against a direct sum and between the CPU and
      the GPU), the joint fit of the moving PSF (bias and realistic errors), the smoothing of the track.
    - Detection: the whole detector on synthetic inputs (_SynthHandle): faint, bright, fast and curved objects,
      noise and stars only, stars whose brightness changes with the transparency of the sky, short tracks.
    - Output, photometry, special cases and the command line: the FTPdetectinfo, the intensities of every
      frame, saturation and spilled charge, flashing objects, trails of bright objects, the monitor options.

The positions of the objects are known exactly, so the measured positions are compared with the truth. The
noise is NOISE ADU per pixel, and the brightness of an object is given as its peak signal-to-noise ratio in a
single frame (the peak of its PSF over the noise of a pixel).
"""

import os
import math
import datetime

import cv2
import numpy as np
import pytest
from astropy.io import fits

import RMS.ConfigReader as cr
from RMS.Formats import FTPdetectinfo
from RMS.DetectStarsAndMeteors import saveResultsFrameInterface
from RMS.Routines.MaskImage import MaskStructure
import RMS.MatchedFilterDetection as mfd
from RMS.MatchedFilterDetection import velocityGrid, binFrames, MatchedFilterDetector, smoothTrack, smoothPositions
from RMS.Routines.MatchedFilterKernels import CUDA_AVAILABLE, VelocityStacker, fitMovingPSF


# The time of the first frame of the synthetic inputs
BEG_TIME = datetime.datetime(2026, 10, 8, 3, 0, 0)

# The synthetic frames: size (px), sky level and its noise (ADU), and the PSF sigma of the stars and objects (px)
SIZE = 128
SKY = 100.0
NOISE = 5.0
PSF_SIGMA = 1.0


def renderObject(img, x, y, peak, sigma=PSF_SIGMA, vx=0.0, vy=0.0, n_sub=5):
    """ Add a Gaussian PSF with the given peak (ADU) at (x, y), smeared over the frame by the motion. """

    # The motion during the frame is approximated by n_sub copies of the PSF, each with 1/n_sub of the peak,
    #   at the positions of the middles of n_sub equal parts of the frame (relative times -0.5 .. 0.5)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    for s in range(n_sub):
        d = (s + 0.5)/n_sub - 0.5
        img += peak/n_sub*np.exp(-0.5*((xx - x - vx*d)**2 + (yy - y - vy*d)**2)/sigma**2)


class _SynthHandle(object):
    """ Frame-based image handle of synthetic frames: sky, noise, static stars, and moving objects whose
        positions are functions of the frame number.
    """

    input_type = 'vid'
    byteswap = False

    def __init__(self, objects, total_frames=512, fps=25.0, seed=1, n_stars=12, n_faint_stars=0,
                 star_gain=None, clip=None):

        self.objects = objects
        self.total_frames = total_frames
        self.nrows = self.ncols = SIZE
        self.fps = fps
        self.beginning_datetime = BEG_TIME
        self.dir_path = None
        self.seed = seed
        self.current_frame = 0

        # Brightness of the stars in every frame relative to the average (e.g. the transparency of the sky)
        self.star_gain = star_gain

        # Saturation level of the frames (pixel values are clipped there), None for no saturation
        self.clip = clip

        # The image of the static stars, the same in every frame (times the gain)
        rng = np.random.default_rng(seed + 1000)
        self.stars = np.zeros((SIZE, SIZE))

        # Positions and fluxes of the bright stars (x, y, flux)
        self.star_list = []
        for _ in range(n_stars):
            x, y, peak = rng.uniform(10, SIZE - 10), rng.uniform(10, SIZE - 10), rng.uniform(5, 50)*NOISE
            renderObject(self.stars, x, y, peak)
            self.star_list.append((x, y, peak*2*np.pi*PSF_SIGMA**2))
        # Faint stars, with peaks of 0.5 - 2 times the noise: too faint to be masked as stars
        for _ in range(n_faint_stars):
            renderObject(self.stars, rng.uniform(5, SIZE - 5), rng.uniform(5, SIZE - 5),
                         rng.uniform(0.5, 2)*NOISE)

    def setFrame(self, fr_num):
        self.current_frame = fr_num

    def loadFrame(self, avepixel=False):
        """ The current frame. The noise of every frame is seeded with its number, so a frame is the same
            every time it is read (the detector reads the frames several times).
        """

        # The sky, the stars (scaled by the transparency) and the noise
        f = self.current_frame
        rng = np.random.default_rng(self.seed*100003 + f)
        gain = self.star_gain(f) if self.star_gain is not None else 1.0
        img = SKY + gain*self.stars + rng.normal(0, NOISE, (SIZE, SIZE))

        # The moving objects visible in this frame (frames f0 .. f1 - 1)
        for obj in self.objects:
            if obj['f0'] <= f < obj['f1']:

                # The position in the middle of the frame, and the motion during the frame (from the positions
                #   half a frame before and after)
                x, y = obj['pos'](f)
                x2, y2 = obj['pos'](f + 0.5)
                x1, y1 = obj['pos'](f - 0.5)

                # The brightness can change with time (e.g. flashes)
                snr = obj['snr'](f) if callable(obj['snr']) else obj['snr']
                renderObject(img, x, y, snr*NOISE, vx=x2 - x1, vy=y2 - y1)

                # Charge spilled along the row and the column of a very bright object: the given fraction of
                #   its flux spread over 4 - 10 px on both sides along the row and the column (28 pixels)
                if obj.get('spill'):
                    xi, yi = int(round(x)), int(round(y))
                    flux = snr*NOISE*2*np.pi*PSF_SIGMA**2*obj['spill']
                    for d in list(range(-10, -3)) + list(range(4, 11)):
                        if 0 <= xi + d < SIZE:
                            img[yi, xi + d] += flux/28
                        if 0 <= yi + d < SIZE:
                            img[yi + d, xi] += flux/28

                # Trails of a bright source along its whole column, and along its row on one side
                if obj.get('trail'):
                    xi, yi = int(round(x)), int(round(y))
                    img[:, xi] += obj['trail']*NOISE
                    img[yi, :max(xi - 3, 0)] += 2*obj['trail']*NOISE

        # Saturation
        if self.clip is not None:
            img = np.minimum(img, self.clip)

        return img.astype(np.float32)

    def currentFrameTime(self, frame_no=None, dt_obj=False):
        """ The time of a frame, at a constant frame rate. """

        if frame_no is None:
            frame_no = self.current_frame

        return BEG_TIME + datetime.timedelta(seconds=frame_no/self.fps)


def linearObject(snr, x0, y0, vx, vy, f0, f1):
    """ An object moving at a constant velocity, at (x0, y0) in the frame f0. """

    return {'snr': snr, 'f0': f0, 'f1': f1, 'pos': lambda f: (x0 + vx*(f - f0), y0 + vy*(f - f0))}


@pytest.fixture
def config():
    """ The config of the synthetic inputs: the size of the frames, no binning, the PSF sigma given (it is not
        measured on the synthetic stars) and the CPU only (the results are the same on the GPU).
    """

    config = cr.Config()
    config.stationID = 'XX0001'
    config.width = config.height = SIZE
    config.fps = 25.0

    # 0.04 deg/px: the speeds of 0.01 - 2 deg/s are 0.01 - 2 px/frame
    config.fov_w = config.fov_h = 0.04*SIZE
    config.detection_binning_factor = 1
    config.mf_psf_sigma = PSF_SIGMA
    config.mf_gpu = 'off'

    return config


def truthError(cent, obj):
    """ Distances of the centroids from the true positions (px). """

    truth = np.array([obj['pos'](f) for f in cent[:, 0]])
    return np.hypot(cent[:, 1] - truth[:, 0], cent[:, 2] - truth[:, 1])


### Kernels ###

def test_velocity_grid_sorted_and_bounded():
    """ The velocity grid of runs of 8 frames binned 2x2, up to 2 px/frame: sorted by speed from 0, and not
        faster than the largest speed (in binned pixels) plus one step.
    """

    vel = velocityGrid(2.0, 8, 2)
    speed = np.hypot(vel[:, 0], vel[:, 1])

    # Sorted by speed, starting with the static stack
    assert np.all(np.diff(speed) >= 0)
    assert speed[0] == 0

    # 2 px/frame is 1 binned px/frame, and the grid covers it within one step of 1/(n - 1) = 1/7
    assert speed.max() <= 2.0/2 + 1.0/7 + 1e-9

    # The spacing is a binned pixel over the run: the largest position error at the run end is half a pixel
    assert np.isclose(np.min(np.abs(np.diff(np.unique(vel[:, 0])))), 1.0/7)


def test_bin_frames_keeps_unit_noise():
    """ Binned frames of unit noise have unit noise again (the sum of b^2 pixels divided by b), so the
        thresholds of the search are in units of the noise.
    """

    rng = np.random.default_rng(0)
    binned = binFrames(rng.normal(0, 1, (4, 128, 128)).astype(np.float32), 2)

    assert binned.shape == (4, 64, 64)
    assert abs(np.std(binned) - 1) < 0.03


@pytest.mark.parametrize('bin_factor', [2, 3, 4, 8])
def test_bin_frames_is_the_scaled_sum_of_the_bins(bin_factor):
    """ Every binned pixel is the sum of its bin divided by the bin factor, with the frames cropped to whole bins
        (up to the rounding of the last bits of a float32, as the pixels are added in another order).
    """

    # Frames whose size is not a multiple of the bins, and the sums of their whole bins
    rng = np.random.default_rng(1)
    frames = rng.normal(0, 1, (3, 101, 135)).astype(np.float32)
    h2, w2 = 101//bin_factor, 135//bin_factor
    expected = frames[:, :h2*bin_factor, :w2*bin_factor].reshape(3, h2, bin_factor, w2, bin_factor).sum(axis=(2, 4))

    # float32 frames of the whole bins, every pixel the sum divided by the bin factor
    binned = binFrames(frames, bin_factor)
    assert binned.dtype == np.float32
    assert np.allclose(binned, expected/bin_factor, rtol=1e-6, atol=1e-6)


def test_velocity_stack_finds_moving_object():
    """ A single-pixel object of 3 sigma per frame moving across 8 frames of unit noise: the maximum of the
        velocity stack is at the position of the object in the middle of the run, at the right velocity.
    """

    # The object is in the pixel nearest to its position in every frame, the times relative to the middle of
    #   the run
    rng = np.random.default_rng(1)
    n = 8
    frames = rng.normal(0, 1, (n, 64, 64)).astype(np.float32)
    vel_true = np.array([1.0, -0.5])
    for k in range(n):
        dt = k - (n - 1)/2.0
        x, y = 30 + vel_true[0]*dt, 32 + vel_true[1]*dt
        frames[k, int(round(y)), int(round(x))] += 3.0

    # The maximum over the velocities in every pixel, and the index of the best velocity
    vel = velocityGrid(1.5, n, 1)
    out_max, out_idx = VelocityStacker(vel, n, use_gpu=False).stackMax(frames)

    # The brightest pixel is the object, and its velocity is within a step of the grid of the true one
    y, x = np.unravel_index(np.argmax(out_max), out_max.shape)
    assert (x, y) == (30, 32)
    assert np.hypot(*(vel[out_idx[y, x]] - vel_true)) <= 1.0/(n - 1) + 1e-9

    # The mean along the right velocity is the object signal plus the noise of the mean
    assert out_max[y, x] > 2.0


def test_velocity_stack_matches_direct_sum():
    """ The stack of every velocity is the mean of the shifted frames, with pixels outside the image as zero. """

    rng = np.random.default_rng(4)
    n = 6
    frames = rng.normal(0, 1, (n, 20, 30)).astype(np.float32)
    vel = velocityGrid(2.0, n, 1)
    stacker = VelocityStacker(vel, n, use_gpu=False)
    out_max, out_idx = stacker.stackMax(frames)

    # The direct sum: every frame shifted by the integer shift of the velocity at its time (the frames are
    #   padded with 10 px of zeros, more than the largest shift)
    padded = np.pad(frames, ((0, 0), (10, 10), (10, 10)))
    sums = np.zeros((len(vel), 20, 30))
    for v in range(len(vel)):
        for k in range(n):
            dx, dy = stacker.shifts[v, k]
            sums[v] += padded[k, 10 + dy:30 + dy, 10 + dx:40 + dx]
    # The maximum is the mean of the best sum, and the index points to a velocity with the maximum sum (equal
    #   sums can have different indices)
    assert np.allclose(out_max, sums.max(axis=0)/n, atol=1e-5)
    assert np.all(sums[out_idx, np.arange(20)[:, None], np.arange(30)] >= sums.max(axis=0) - 1e-4)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_velocity_stack_cpu_gpu_agree():
    """ The CPU and GPU velocity stacks give the same maxima, and the same best velocities except for rare
        ties (sums equal within the float precision).
    """

    rng = np.random.default_rng(2)
    frames = rng.normal(0, 1, (8, 96, 80)).astype(np.float32)
    vel = velocityGrid(2.0, 8, 2)

    max_cpu, idx_cpu = VelocityStacker(vel, 8, use_gpu=False).stackMax(frames)
    max_gpu, idx_gpu = VelocityStacker(vel, 8, use_gpu=True).stackMax(frames)

    assert np.allclose(max_cpu, max_gpu, atol=1e-5)
    assert np.mean(idx_cpu == idx_gpu) > 0.999


@pytest.mark.parametrize("vx, vy", [(0.0, 0.0), (1.5, -0.7)])
def test_moving_psf_fit_unbiased_with_realistic_errors(vx, vy):
    """ The joint fit of the moving PSF to runs of 8 frames at a peak SNR of 2 per frame: no bias, and the
        scatter of the positions matches the uncertainties of the fit.
    """

    rng = np.random.default_rng(3)
    n = 8
    dt = np.arange(n) - (n - 1)/2.0
    radius = int(math.ceil(3*PSF_SIGMA + 0.5*math.hypot(vx, vy) + 1))

    # 300 runs of frames of unit noise with the object at a random subpixel position in the middle of the run
    errs, sig = [], []
    for _ in range(300):
        xt, yt = 20 + rng.uniform(-0.5, 0.5), 20 + rng.uniform(-0.5, 0.5)
        frames = rng.normal(0, 1, (n, 40, 40))
        for k in range(n):
            renderObject(frames[k], xt + vx*dt[k], yt + vy*dt[k], 2.0, vx=vx, vy=vy)

        # Predicted 0.7 px off the truth
        res = fitMovingPSF(frames.astype(np.float32), np.zeros(frames.shape, np.uint8), xt + 0.5, yt - 0.5, dt, vx, vy,
                           PSF_SIGMA, radius)
        # The errors of the converged fits, and their formal uncertainties
        if res[9] and np.isfinite(res[0]):
            errs.append((res[0] - xt, res[1] - yt))
            sig.append((res[4], res[5]))

    errs = np.array(errs)
    sig = np.array(sig)

    # Nearly all fits converge, without a bias (the mean error is much smaller than the scatter of ~0.3 px)
    assert len(errs) > 280
    assert np.all(np.abs(np.mean(errs, axis=0)) < 0.05)

    # The scatter of the errors is the uncertainty of the fit within 20-30% (the uncertainties are realistic,
    #   so the adaptive number of frames per measurement can be chosen from them)
    ratio = np.std(errs, axis=0)/np.median(sig, axis=0)
    assert np.all((ratio > 0.8) & (ratio < 1.3))


def test_smooth_track_follows_curve():
    """ The smooth track (used in the verification) follows a slightly curved track exactly, also across the
        border of its segments of 512 frames.
    """

    # Exact positions every 4 frames along a quadratic in x and a line in y
    frames = np.arange(0, 800, 4.0)
    cent = np.column_stack([frames, 10 + 0.1*frames + 2e-5*frames**2, 50 - 0.05*frames])
    xs, ys = smoothTrack(cent, np.arange(0, 797))
    f = np.arange(0, 797)

    assert np.max(np.abs(xs - (10 + 0.1*f + 2e-5*f**2))) < 0.05
    assert np.max(np.abs(ys - (50 - 0.05*f))) < 0.05


def test_smooth_track_short_accelerating_track_with_gap():
    """ A bright fast object whose apparent speed changes (6.25 to 7 px/frame over 108 frames, a short track) and
        which was not measured in the middle of its track: the smooth track follows its curvature, also in the
        gap, instead of a line which misses it by several px in the middle and at the ends.
    """

    # Positions along a line with a constant acceleration, measured in every frame but 37 - 71
    f_all = np.arange(0, 108.0)
    accel = (7.0 - 6.25)/107
    s_all = 6.25*f_all + 0.5*accel*f_all**2
    u = np.array([-0.33, -0.94])/np.hypot(0.33, 0.94)
    measured = (f_all < 37) | (f_all > 71)
    rng = np.random.default_rng(4)
    cent = np.column_stack([f_all[measured], 1364 + s_all[measured]*u[0] + rng.normal(0, 0.05, measured.sum()),
                            722 + s_all[measured]*u[1] + rng.normal(0, 0.05, measured.sum())])

    xs, ys = smoothTrack(cent, f_all.astype(int))
    err = np.hypot(xs - (1364 + s_all*u[0]), ys - (722 + s_all*u[1]))
    assert np.max(err) < 0.2


def test_curvature_of_noisy_short_tracks_not_fitted():
    """ Faint straight short tracks (position errors of 0.5 px): the curvature is not significant, so the smooth
        track stays a line and doesn't follow the noise; a clearly curved bright one is found curved.
    """

    rng = np.random.default_rng(5)
    t = np.arange(0, 100.0, 2)
    n_curved = 0
    for _ in range(200):
        x = 10 + 0.3*t + rng.normal(0, 0.5, len(t))
        y = 40 - 0.2*t + rng.normal(0, 0.5, len(t))
        n_curved += mfd.curvatureSignificant(t, x, y)

    # A false curvature at the 5 sigma level is very rare
    assert n_curved <= 1

    # 1 px of curvature over the track, measured to 0.05 px
    x = 10 + 0.3*t + 4.0/100**2*(t - 50)**2 + rng.normal(0, 0.05, len(t))
    y = 40 - 0.2*t + rng.normal(0, 0.05, len(t))
    assert mfd.curvatureSignificant(t, x, y)


def test_hits_with_another_velocity_removed_from_a_track():
    """ Hits linked to a track by their position but found with a very different velocity (the sums of other
        velocities through a bright object) are removed; a track whose speed changes keeps its hits, and a track
        whose hits mostly disagree is kept as it is.
    """

    f = np.arange(0, 120, 4.0)
    good = np.column_stack([f, 100 + 3.0*f, 50 - 0.4*f, np.full(len(f), 50.0), np.full(len(f), 3.0),
                            np.full(len(f), -0.4), np.full(len(f), 16.0), np.full(len(f), 2.0)])
    ghosts = np.array([[8.0, 125.0, 30.0, 30.0, 4.6, -10.3, 8.0, 4.0], [40.0, 220.0, 70.0, 30.0, 5.7, 12.0, 8.0, 4.0]])
    hits = np.vstack([good, ghosts])
    hits = hits[np.argsort(hits[:, 0], kind='stable')]
    cleaned = mfd.cleanTrackHits(hits)
    assert len(cleaned) == len(good) and not np.any(np.isin(cleaned[:, 5], [-10.3, 12.0]))

    # A track whose speed changes by 25% keeps all its hits
    accel = good.copy()
    accel[:, 4] = np.linspace(2.7, 3.4, len(f))
    assert len(mfd.cleanTrackHits(accel)) == len(accel)

    # As many hits of other velocities as of the object, but much weaker: they are still removed
    many = np.vstack([good, np.tile(ghosts, (len(good)//2 + 1, 1))])
    many = many[np.argsort(many[:, 0], kind='stable')]
    assert len(mfd.cleanTrackHits(many)) == len(good)

    # A track is not cut below the smallest number of hits of a track: then it is kept as it is
    assert len(mfd.cleanTrackHits(hits, min_hits=len(good) + 1)) == len(hits)

    # Hits which mostly disagree by their weight: the track is kept as it is
    mixed = np.vstack([good[:5], np.tile(ghosts*[1, 1, 1, 3, 1, 1, 1, 1], (4, 1))])
    assert len(mfd.cleanTrackHits(mixed)) == len(mixed)


def test_faint_hits_linked_along_a_line():
    """ The hits of a faint object in a few runs far apart (where the noise added to it) are linked into one chain
        along its line, among the hits of the noise as dense as at a low threshold on real frames.
    """

    rng = np.random.default_rng(7)

    # Hits of the noise: random positions and velocities, 1000 in every run of 8 frames over 512 frames of a
    #   1000 x 1000 px image, as rows [frame, x, y, z, vx, vy, run, bin]
    noise = []
    for t in np.arange(3.5, 512, 8.0):
        noise.append(np.column_stack([np.full(1000, t), rng.uniform(0, 1000, (1000, 2)), rng.uniform(3, 4, 1000),
                                      rng.uniform(-3, 3, (1000, 2)), np.full(1000, 8.0), np.full(1000, 2.0)]))
    noise = np.vstack(noise)

    # The object in 6 of the 64 runs, 7 to 13 runs apart, at its position and velocity with errors like those of
    #   the search (1 px, and 0.07 px per frame along each axis)
    t = np.arange(3.5, 512, 8.0)[[5, 13, 20, 29, 37, 50]]
    obj = np.column_stack([t, 200 + 1.2*t + rng.normal(0, 1, len(t)), 300 - 0.5*t + rng.normal(0, 1, len(t)),
                           rng.uniform(3, 4, len(t)), 1.2 + rng.normal(0, 0.07, len(t)),
                           -0.5 + rng.normal(0, 0.07, len(t)), np.full(len(t), 8.0), np.full(len(t), 2.0)])
    hits = np.vstack([noise, obj])
    chains = mfd.linkFaintHits(hits)

    # One chain holds all hits of the object, and at most one hit of the noise
    def isObject(rows):
        return np.isin(rows[:, 1], obj[:, 1])

    with_object = [ch for ch in chains if isObject(ch).any()]
    assert len(with_object) == 1
    assert isObject(with_object[0]).sum() == len(obj)
    assert (~isObject(with_object[0])).sum() <= 1


def test_smooth_positions_reduce_errors_without_bias():
    """ Noisy positions along a curved track (a bend of 4 px over 400 frames, as across a distorted field):
        combining them over +-64 frames reduces the error several times and does not follow the noise.
    """

    rng = np.random.default_rng(8)
    f = np.arange(0, 400, 4.0)
    true_x = 20 + 0.3*f
    true_y = 30 + 0.1*f + 8*4.0/400**2*0.5*(f - 200)**2
    cent = np.column_stack([f, true_x + rng.normal(0, 0.4, len(f)), true_y + rng.normal(0, 0.4, len(f)),
                            np.zeros(len(f)), np.zeros(len(f)), np.full(len(f), 3.0)])

    # One measurement far off (e.g. a fit pulled to a noise peak at the end of the track)
    cent[0, 1] += 8.0

    # The errors of the measurements and of the smoothed positions, of the kept measurements
    smooth, keep = smoothPositions(cent, 64)
    raw_err = np.hypot(cent[keep, 1] - true_x[keep], cent[keep, 2] - true_y[keep])
    err = np.hypot(smooth[:, 0] - true_x[keep], smooth[:, 1] - true_y[keep])

    # The outlier is removed and nearly all others kept; the error is at least 2.5 times smaller (it would be
    #   about sqrt(33) times smaller for a straight line fit to the 33 measurements of a window, the quadratic
    #   costs some of that), and there is no bias from the curvature
    assert not keep[0] and keep[1:].sum() >= len(f) - 3
    assert np.sqrt(np.mean(err**2)) < np.sqrt(np.mean(raw_err**2))/2.5
    assert abs(np.mean(smooth[:, 1] - true_y[keep])) < 0.05

    # The ends of the track are not worse than the raw measurements
    assert np.all(err[:5] < 0.6) and np.all(err[-5:] < 0.6)


### Detection ###

def runDetector(handle, config):
    """ Run the whole detector on an image handle, return the detector and its detections. """

    det = MatchedFilterDetector(handle, config)
    return det, det.run()


def test_detects_faint_object_below_single_frame_limit(config):
    """ An object with a peak SNR of 1.5 per frame (far below the normal detection) moving at 0.3 px/frame is
        found and measured to about half a pixel. Its positions are measured on several frames together, but
        there is a row for every frame (with the intensity of that frame).
    """

    obj = linearObject(1.5, 25, 30, 0.3, 0.12, 40, 470)
    det, detections = runDetector(_SynthHandle([obj]), config)

    # One detection (not broken into pieces), with the RMS position error below 0.6 px
    assert len(detections) == 1
    cent = detections[0][2]

    err = truthError(cent, obj)
    assert np.sqrt(np.mean(err**2)) < 0.6

    # Most of the track is measured, every frame
    assert cent[-1, 0] - cent[0, 0] > 0.7*(obj['f1'] - obj['f0'])
    assert np.all(np.diff(cent[:, 0]) >= 1) and (np.median(np.diff(cent[:, 0])) == 1)


def test_bright_object_measured_every_frame(config):
    """ A bright object (peak SNR 12 per frame) is measured on single frames (the adaptive number of frames per
        measurement is 1), to better than 0.25 px.
    """

    obj = linearObject(12.0, 20, 100, 0.8, -0.3, 100, 220)
    _, detections = runDetector(_SynthHandle([obj]), config)

    assert len(detections) == 1
    cent = detections[0][2]
    assert np.median(np.diff(cent[:, 0])) == 1
    assert np.sqrt(np.mean(truthError(cent, obj)**2)) < 0.25


def test_detects_fast_faint_object_with_velocity_search(config):
    """ A faint object (peak SNR 2.5 per frame) moving at 1.8 px/frame for 55 frames is found by the velocity
        search. Without it, the stacks of static positions smear it out over many pixels.
    """

    obj = linearObject(2.5, 15, 20, 1.6, 0.9, 200, 255)
    _, detections = runDetector(_SynthHandle([obj]), config)

    assert len(detections) == 1
    assert np.sqrt(np.mean(truthError(detections[0][2], obj)**2)) < 0.5

    # Without the velocity search, the object is too fast to be found
    config.mf_velocity_search = False
    _, detections = runDetector(_SynthHandle([obj]), config)
    assert len(detections) == 0


def test_follows_curved_track(config):
    """ A track bent like an object seen through a distorting lens: the measurements follow the curve. """

    def pos(f):
        t = f - 50
        return 20 + 0.22*t, 20 + 0.02*t + 0.0004*t**2

    obj = {'snr': 2.5, 'f0': 50, 'f1': 480, 'pos': pos}
    _, detections = runDetector(_SynthHandle([obj]), config)

    # One detection which follows the curve (a straight line would be off by up to ~10 px) over most of the
    #   track
    assert len(detections) == 1
    cent = detections[0][2]
    assert np.sqrt(np.mean(truthError(cent, obj)**2)) < 0.5
    assert cent[-1, 0] - cent[0, 0] > 0.7*(obj['f1'] - obj['f0'])


def test_no_detections_in_noise_and_stars(config):
    """ No false detections in frames of noise and static stars (two different noise realizations). """

    for seed in (5, 6):
        _, detections = runDetector(_SynthHandle([], seed=seed), config)
        assert len(detections) == 0


@pytest.mark.parametrize('correct', [True, False])
def test_transparency_changes_are_removed(config, monkeypatch, correct):
    """ Faint stars (too faint to be masked) whose brightness changes with the transparency of the sky leave
        no residuals in the normalized frames, which would make faint static sources varying in time.
    """

    # Without the correction: the star template needs more pixels than there are, so it is never used
    if not correct:
        monkeypatch.setattr(mfd, 'MIN_STAR_TEMPLATE_PX', 10**9)

    # 150 faint stars whose brightness changes by +-80% with a period of 300 frames
    handle = _SynthHandle([], n_stars=0, n_faint_stars=150,
                          star_gain=lambda f: 1.0 + 0.8*math.sin(2*math.pi*f/300.0))
    # Only the background of the two blocks of frames, without the search
    det = MatchedFilterDetector(handle, config)
    det.block_starts, det.block_ends = [0, 256], [256, 512]
    det.backgroundPass()

    # Residuals of the stars relative to their brightness, in the mean of 32 frames when the stars are
    #   brightest (1.8 times the average)
    z = det.normalize(det.readFrames(0, 256), 0)
    star_px = handle.stars > 0.5*NOISE
    resid = np.mean(np.mean(z[60:92], axis=0)[star_px]/(handle.stars[star_px]/NOISE))

    # With the correction the residuals are below 10% of the stars, without it they are most of the change of
    #   their brightness
    if correct:
        assert abs(resid) < 0.1
    else:
        assert resid > 0.5


def test_faint_object_found_with_changing_transparency(config):
    """ A faint object (peak SNR 2 per frame) among faint stars whose brightness changes with the transparency
        is found and measured, without false detections on the stars.
    """

    obj = linearObject(2.0, 25, 30, 0.2, 0.1, 40, 470)
    handle = _SynthHandle([obj], n_stars=4, n_faint_stars=150,
                          star_gain=lambda f: 1.0 + 0.8*math.sin(2*math.pi*f/300.0))
    _, detections = runDetector(handle, config)

    assert len(detections) == 1
    assert np.sqrt(np.mean(truthError(detections[0][2], obj)**2)) < 0.6


def test_short_track_rejected(config):
    """ Tracks shorter than mf_min_frames are not kept. """

    # A bright object visible for 14 frames
    obj = linearObject(15.0, 40, 40, 1.0, 0.0, 300, 314)
    config.mf_min_frames = 20
    _, detections = runDetector(_SynthHandle([obj]), config)

    assert len(detections) == 0


def test_rolling_shutter_corrects_the_frames(config):
    """ With a rolling shutter (deinterlace_order -1), the frames of the centroids of the whole input are later by
        the share of the readout time of their row; with a global shutter, or for a segment of the input (corrected
        with the other segments), they are unchanged.
    """

    # The detection with a global shutter
    obj = linearObject(3.0, 20, 100, 0.3, -0.15, 100, 400)
    _, global_shutter = runDetector(_SynthHandle([obj]), config)

    # A sensor read in 32 ms at 25 fps: 0.8 frames from the top row to the bottom row
    config.deinterlace_order = -1
    config.rolling_shutter_readout = 0.032
    rolling = mfd.detectMatchedFilter(_SynthHandle([obj]), config)
    assert (len(global_shutter) == 1) and (len(rolling) == 1)
    cent_g, cent_r = global_shutter[0][2], rolling[0][2]
    assert np.allclose(cent_r[:, 1:], cent_g[:, 1:])
    assert np.allclose(cent_r[:, 0] - cent_g[:, 0], 0.8*cent_g[:, 2]/SIZE)

    # A segment of the input is not corrected on its own
    segment = mfd.detectMatchedFilter(_SynthHandle([obj]), config, frame_range=(0, 512))
    assert np.allclose(segment[0][2][:, 0], cent_g[:, 0])


### Output ###

def test_ftpdetectinfo_round_trip(config, tmp_path):
    """ The detections saved as an FTPdetectinfo with the suffix "mf" (as the normal detection saves its
        detections) are read back with the same positions and times.
    """

    obj = linearObject(4.0, 25, 30, 0.3, 0.15, 40, 300)
    handle = _SynthHandle([obj])
    _, detections = runDetector(handle, config)
    assert len(detections) == 1

    # Save without stars, with the suffix of the matched filter
    _, _, ftp_name = saveResultsFrameInterface([], detections, handle, config, output_suffix='mf',
                                               output_dir=str(tmp_path))
    assert ftp_name.endswith('_mf.txt')

    # Read it back in the format of the writer
    _, _, meteors = FTPdetectinfo.readFTPdetectinfo(str(tmp_path), ftp_name, ret_input_format=True)
    assert len(meteors) == 1

    # Centroid rows: frame (relative to the time in the FF name), x, y, ...
    picks = np.array(meteors[0][4], dtype=np.float64)
    cent = detections[0][2]
    assert len(picks) == len(cent)
    assert np.allclose(picks[:, 1], cent[:, 1], atol=0.01)
    assert np.allclose(picks[:, 2], cent[:, 2], atol=0.01)

    # The times are kept, the frames are counted from the time in the name
    assert np.allclose(np.diff(picks[:, 0]), np.diff(cent[:, 0]), atol=0.01)


def test_faint_object_found_by_faint_linking(config):
    """ An object with a peak SNR of 1 per frame moving at 0.85 px/frame for 120 frames is below the threshold of
        the search in most runs and is not found. The chains of the hits down to the faint threshold find it, and
        it is measured to better than half a pixel over most of its track.
    """

    # Without the faint linking: not found, and no faint hits kept
    obj = linearObject(1.0, 10, 20, 0.8, 0.3, 300, 420)
    det, detections = runDetector(_SynthHandle([obj]), config)
    assert len(detections) == 0
    assert (not det.faint) and (len(det.faint_hits) == 0)

    # With it: found and measured over most of its track
    config.mf_faint_threshold = 3.0
    det, detections = runDetector(_SynthHandle([obj]), config)
    assert len(detections) == 1
    cent = detections[0][2]
    assert np.sqrt(np.mean(truthError(cent, obj)**2)) < 0.5
    assert cent[-1, 0] - cent[0, 0] > 0.7*(obj['f1'] - obj['f0'])
    assert det.summary()['faint_measured'] >= 1


def test_no_faint_detections_in_noise_and_stars(config):
    """ The chains of the noise and of faint stars are not significant: no detections with the faint linking. A
        faint threshold at or above the threshold turns it off.
    """

    # Noise and faint stars only, two noise realizations
    config.mf_faint_threshold = 3.0
    for seed in (5, 6):
        det, detections = runDetector(_SynthHandle([], seed=seed, n_faint_stars=30), config)
        assert len(detections) == 0

    # The option is off when the faint threshold is not below the threshold
    config.mf_faint_threshold = config.mf_threshold
    assert not MatchedFilterDetector(_SynthHandle([]), config).faint


def test_faint_significance_with_the_streak_search(config, monkeypatch):
    """ With the streak search, the signal along the line of a faint chain is summed with the PSF smeared along
        the motion of the object during a frame, as in verify; without it, as a point source. """

    # An object and a chain of hits on its line
    obj = linearObject(1.0, 10, 20, 0.8, 0.3, 300, 420)
    frames = np.arange(300, 420, 10.0)
    chain = np.column_stack([frames, 10 + 0.8*(frames - 300), 20 + 0.3*(frames - 300)])

    # The motion passed to the sums of the signal
    motions = []
    signal = mfd.forcedTrackSignal

    def spy(*args):
        motions.append(args[-2:] if len(args) > 7 else (0.0, 0.0))
        return signal(*args)

    # Without the streak search a point source, with it the motion of the chain per frame
    for streak, expected in ((False, (0.0, 0.0)), (True, (0.8, 0.3))):
        config.mf_faint_threshold = 3.0
        config.mf_streak_search = streak
        det, _ = runDetector(_SynthHandle([obj]), config)
        monkeypatch.setattr(mfd, 'forcedTrackSignal', spy)
        del motions[:]
        det.faintSignificance([chain])
        monkeypatch.setattr(mfd, 'forcedTrackSignal', signal)
        assert motions and all(np.allclose(m, expected, atol=1e-6) for m in motions)


def test_gap_pieces_count_the_faint_chains_of_all_parts():
    """ An input with a gap in the recording is searched in parts: the faint chains measured are counted over
        all parts, not only the main one. """

    import types

    # The detectors of two parts, with their counts of faint chains
    def part(measured):
        return types.SimpleNamespace(psf_sigma=1.0, timing={}, cloud_rows_removed=0, cloud_removed_row_frames=[],
                                     cloud_detections_removed=0, cloud_removed_frames=[], tracks_dropped=0,
                                     tracks_measured=0, significances=[], deviation_rejected=0, photometry={},
                                     sky_map=None, faint_measured=measured)

    # The whole range counts the chains of both parts
    run = mfd.GapPieces([part(2), part(3)], [(0, 100), (120, 200)], [120], [0.5])
    assert run.faint_measured == 5


def test_config_parse(tmp_path):
    """ The options of the [MatchedFilter] section of the config are parsed (booleans, lists, numbers and
        strings).
    """

    path = os.path.join(str(tmp_path), '.config')
    with open(path, 'w') as f:
        f.write("[System]\nstationID: XX0001\n\n[MatchedFilter]\nmf_enable: true\nmf_run_frames: 4, 8\n"
                "mf_threshold: 6.5\nmf_gpu: off\nmf_min_frames: 30\nmf_faint_threshold: 3.5\nmf_faint_max_tracks: 20\n")

    config = cr.parse(path)

    assert config.mf_enable is True
    assert config.mf_run_frames == [4, 8]
    assert config.mf_threshold == 6.5
    assert config.mf_gpu == 'off'
    assert config.mf_min_frames == 30
    assert (config.mf_faint_threshold == 3.5) and (config.mf_faint_max_tracks == 20)


### Operation ###

def test_ff_input_is_skipped(config):
    """ FF files have no frames, so there is nothing to detect and no detector. """

    class _FF(object):
        input_type = 'ff'

    assert mfd.detectMatchedFilter(_FF(), config) == []
    assert mfd.detectMatchedFilter(_FF(), config, return_detector=True) == ([], None)


def test_summary_lists_tracks_with_significance(config):
    """ The summary of a run (saved as the done file) lists every candidate track with its significance and
        whether it was detected, and the processing time of every step.
    """

    obj = linearObject(2.0, 25, 30, 0.3, 0.12, 40, 470)
    det, detections = runDetector(_SynthHandle([obj]), config)
    summary = det.summary()

    # The tracks marked as detected are the ones above the significance limit, and the only one is the object
    assert summary['total_frames'] == 512
    assert sum(t['detected'] for t in summary['tracks']) == len(detections) == 1
    assert all((t['significance'] >= config.mf_track_significance) == t['detected'] for t in summary['tracks'])
    assert 'measure' in summary['timing_s']


def test_find_input_files(tmp_path):
    """ The video files are found in directories and their subdirectories (the extension in any case), other
        files are ignored, and a file given both directly and in its directory is listed once. A directory of
        FITS frames is one input, but not a directory of FF files or of .fit frames.
    """

    # A directory of FITS frames, a directory of the FF files of the normal processing, and FRIPON .fit frames
    (tmp_path/'night'/'sub').mkdir(parents=True)
    for name in ('20260424_054022', 'XX0001_20260424_010000_000000', 'fripon'):
        (tmp_path/'night'/name).mkdir()
    for name in ('a.vid', 'sub/b.vid', 'sub/c.MKV', 'notes.txt', '20260424_054022/f1.fits',
                 '20260424_054022/f2.fits', 'XX0001_20260424_010000_000000/FF_XX0001_20260424_010000_000_0000000.fits',
                 'fripon/f1.fit'):
        (tmp_path/'night'/name).write_text('x')

    files = mfd.findInputFiles([str(tmp_path/'night'), str(tmp_path/'night'/'a.vid')])

    assert [os.path.basename(f) for f in files] == ['20260424_054022', 'a.vid', 'b.vid', 'c.MKV']

    # A directory of FITS frames given directly
    assert mfd.findInputFiles([str(tmp_path/'night'/'20260424_054022')]) == [str(tmp_path/'night'/'20260424_054022')]


def test_fits_directory_input(config, tmp_path):
    """ A directory of FITS frames (one frame per file, with the time of the frame in DATE-OBS) is processed
        like a video file: the object in it is detected and saved.
    """

    # The synthetic frames saved as 16-bit FITS files, as the cameras which save FITS frames do
    obj = linearObject(3.0, 25, 30, 0.3, 0.12, 40, 470)
    handle = _SynthHandle([obj])
    fits_dir = tmp_path/'20261008_030000'
    fits_dir.mkdir()
    for f in range(handle.total_frames):
        handle.setFrame(f)
        frame = np.clip(np.round(handle.loadFrame()), 0, 65535).astype(np.uint16)
        hdu = fits.PrimaryHDU(frame)
        hdu.header['DATE-OBS'] = handle.currentFrameTime(f).isoformat()
        hdu.writeto(str(fits_dir/'XX0001_{:04d}.fits'.format(f)))

    # The frame rate of FITS frames is measured from their times, also with a wrong frame rate in the config
    config.fps = 2*handle.fps
    summary = mfd.processFile(str(fits_dir), config, str(tmp_path/'out'), extract_stars=False)

    assert summary['detections'] == 1
    assert os.path.isfile(str(tmp_path/'out'/summary['ftpdetectinfo']))


def test_process_files_resumes_and_continues_after_failure(config, tmp_path, monkeypatch):
    """ Files with a done file are skipped, a failing file doesn't stop the others, and the detections of all
        files are merged.
    """

    calls = []

    # A replacement of processFile which records the files it is called on, fails on the "bad" file, and
    #   otherwise writes an FTPdetectinfo with one detection and the done file
    def _process(path, config, out, **kwargs):
        calls.append(os.path.basename(path))
        if 'bad' in path:
            raise RuntimeError('broken file')
        os.makedirs(out)
        picks = [[f, 10.0 + f, 20.0, 1.0, 2.0, 3.0, 4.0, 100, 9.0, 50, 5.0, 0] for f in (1.0, 2.0)]
        # The recalibration leaves a backup of the uncalibrated file next to the calibrated one
        for name in ('FTPdetectinfo_{:s}_mf.txt', 'FTPdetectinfo_{:s}_mf_backup_20261008_184146.444121.txt'):
            FTPdetectinfo.writeFTPdetectinfo([['FF_XX0001_20261008_030000_000_0000000.fits', 1, 10.0, 45.0,
                                               picks, 25.0]], out, name.format(os.path.basename(out)), out,
                                             'XX0001', 25.0, celestial_coords_given=True)
        return mfd.saveSummary(out, None, detections=1, processing_time_s=1.0)

    monkeypatch.setattr(mfd, 'processFile', _process)
    files = [str(tmp_path/name) for name in ('a.vid', 'bad.vid', 'c.vid')]
    out = str(tmp_path/'out')

    # The failed file is returned, and the file after it is still processed
    assert mfd.processFiles(files, config, out) == [files[1]]
    assert calls == ['a.vid', 'bad.vid', 'c.vid']

    # One merged file with the detections of the two good files (the backups of the recalibration are not
    #   merged)
    merged = [f for f in os.listdir(out) if f.startswith('FTPdetectinfo')]
    assert merged == ['FTPdetectinfo_out_mf.txt']
    assert len(FTPdetectinfo.readFTPdetectinfo(out, merged[0])) == 2

    # Started again: only the failed file is processed
    calls.clear()
    assert mfd.processFiles(files, config, out) == [files[1]]
    assert calls == ['bad.vid']


def test_output_names_unique_for_same_file_names(tmp_path):
    """ Files with the same name in different directories get different output directories (the name and a
        hash of the path), a unique name is kept as it is, and the names are the same in every run.
    """

    files = [str(tmp_path/'cam1'/'a.vid'), str(tmp_path/'cam2'/'a.vid'), str(tmp_path/'cam1'/'b.vid')]
    names = mfd.outputNames(files)

    assert names[2] == 'b'
    assert len(set(names)) == 3 and all(n.startswith('a_') for n in names[:2])
    assert mfd.outputNames(files) == names


@pytest.mark.parametrize('method, factor', [('avg', 4), ('sum', 1)])
def test_binned_frames_speed_limits_and_intensity(config, method, factor):
    """ With 2x2 detection binning the speed limits are in binned pixels, and the intensity is scaled to the
        unbinned image only for averaged bins.
    """

    config.detection_binning_factor = 2
    config.detection_binning_method = method

    # The speeds in px/frame of the binned image are half of the unbinned ones
    opts = mfd.MatchedFilterOptions(config, det_bin=2)
    unbinned = mfd.MatchedFilterOptions(config)
    assert np.isclose(opts.speed_max, unbinned.speed_max/2)
    assert np.isclose(opts.speed_min, unbinned.speed_min/2)

    # The same object detected on the binned frames (the synthetic frames are not binned, so this only tests
    #   the scaling of the intensity) and without binning
    obj = linearObject(12.0, 20, 100, 0.8, -0.3, 100, 220)
    det = MatchedFilterDetector(_SynthHandle([obj]), config)
    detections = det.run()
    config.detection_binning_factor = 1
    det1 = MatchedFilterDetector(_SynthHandle([obj]), config)
    detections1 = det1.run()

    # Averaged bins hold the mean of 4 pixels, so their intensities are multiplied by 4; summed bins are not
    #   scaled
    assert len(detections) == len(detections1) == 1
    ratio = np.median(detections[0][2][:, 3])/np.median(detections1[0][2][:, 3])
    assert np.isclose(ratio, factor, rtol=0.1)


### Photometry and saturation ###

@pytest.mark.parametrize('snr, vx', [(3.0, 0.3), (12.0, 0.8), (12.0, 1.8)])
def test_intensity_is_the_flux_per_frame(config, snr, vx):
    """ The intensity of a measurement is the background-subtracted sum of the pixels of the object in a frame,
        as for the normal detection: the flux of the PSF (peak*2*pi*sigma^2), within a few percent.
    """

    # The object moves 80 px, so the track is long enough at every speed
    obj = linearObject(snr, 20, 60, vx, 0.1, 60, 60 + int(80/vx))
    config.mf_smooth_frames = 0
    _, detections = runDetector(_SynthHandle([obj], n_stars=4), config)

    # The flux of a Gaussian PSF is its peak times 2*pi*sigma^2; faint objects have a larger scatter
    assert len(detections) == 1
    flux = snr*NOISE*2*np.pi*PSF_SIGMA**2
    intensity = np.median(detections[0][2][:, 3])
    assert abs(intensity/flux - 1) < (0.06 if snr > 5 else 0.15)


def test_saturated_object_flagged_and_measured(config):
    """ An object whose core is clipped by saturation: the measurements are flagged, and leaving the clipped
        pixels out of the fit keeps the positions accurate.
    """

    # The frames saturate at 25 times the noise above the sky, the peak of the object is at 60 times
    obj = linearObject(60.0, 20, 60, 0.6, 0.2, 60, 200)
    clip = SKY + 25*NOISE
    config.mf_saturation_level = clip - 1
    config.mf_smooth_frames = 0
    _, detections = runDetector(_SynthHandle([obj], n_stars=4, clip=clip), config)

    # The frames have saturated pixels, and the positions are accurate (a fit to the clipped core would be
    #   pulled around by the flat top)
    assert len(detections) == 1
    cent = detections[0][2]
    assert np.median(cent[:, 6]) >= 1
    assert np.sqrt(np.mean(truthError(cent, obj)**2)) < 0.15

    # Without a saturation level, nothing is flagged
    config.mf_saturation_level = 0
    _, detections = runDetector(_SynthHandle([obj], n_stars=4, clip=clip), config)
    assert np.all(detections[0][2][:, 6] == 0)


@pytest.mark.parametrize('mask_slow', [True, False])
def test_slow_object_photometry_uses_background_without_it(config, monkeypatch, mask_slow):
    """ A slow object stays on the same pixels for a large part of the frames of the background, which then
        contains part of it; the background is estimated again without it, so its intensity is the flux.
    """

    # Without the masking: no track is slower than a speed of 0, so no background is estimated again
    if not mask_slow:
        monkeypatch.setattr(mfd, 'SLOW_SPEED', 0.0)

    # 1536 frames: the object stays on its pixels for a tenth of them (in a 10-minute file, under 2%)
    obj = linearObject(6.0, 30, 60, 0.03, 0.015, 0, 1536)
    config.mf_smooth_frames = 0
    _, detections = runDetector(_SynthHandle([obj], n_stars=4, total_frames=1536), config)

    # With the masking the intensity is the flux, without it more than 15% of the flux is in the background
    assert len(detections) == 1
    flux = 6.0*NOISE*2*np.pi*PSF_SIGMA**2
    ratio = np.median(detections[0][2][:, 3])/flux
    if mask_slow:
        assert abs(ratio - 1) < 0.1
    else:
        assert ratio < 0.85


@pytest.mark.parametrize('trail_level, line_concentration', [(25.0, 0.0), (0.0, 0.5), (0.0, 0.0)])
def test_trails_of_bright_object_removed(config, trail_level, line_concentration):
    """ A very bright object which leaves faint trails along its column and its row: the trails move with it
        and make tracks along them, unless they are removed from the search, or the tracks rejected as lying on
        a linear structure.
    """

    # Trails of 1 sigma along the column and 2 sigma along the row, moving with the object
    obj = linearObject(400.0, 110, 50, -0.6, 0.1, 100, 260)
    obj['trail'] = 1.0
    config.mf_trail_level = trail_level
    config.mf_line_concentration = line_concentration
    det, detections = runDetector(_SynthHandle([obj], n_stars=4), config)

    # With the trails removed, or their tracks rejected, only the object is detected (at its position); with
    #   neither the trails are detected too
    if (trail_level > 0) or (line_concentration > 0):
        assert len(detections) == 1
        assert np.median(truthError(detections[0][2], obj)) < 0.3
        assert (det.summary()['linear_structure_rejected'] > 0) == (trail_level == 0)
    else:
        assert len(detections) > 1


def test_extent_line_tells_points_from_lines():
    """ The concentration along the brightest axis through an object: about 1 for a point source (also one smeared
        by its motion during a frame), low for an object on a thin line through it, and the axis of noise alone is
        not significant.
    """

    r = mfd.EXTENT_RADIUS
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(float)
    rng = np.random.default_rng(3)
    n = 200
    noise = rng.normal(0, 1/np.sqrt(n), xx.shape)

    # A point source, and the same smeared by 3.6 px/frame along a diagonal (the fastest searched motions)
    point = 2.0*np.exp(-0.5*(xx**2 + yy**2)/PSF_SIGMA**2)
    smeared = sum(0.4*np.exp(-0.5*((xx - 2.5*d)**2 + (yy - 2.5*d)**2)/PSF_SIGMA**2)
                  for d in np.linspace(-0.5, 0.5, 5))
    for stack in (point, smeared):
        concentration, _ = mfd.extentLine(stack + noise, PSF_SIGMA, n)
        assert concentration > 0.8

    # A point on a line through it as bright as its peak, along any direction (also between the axes): the
    #   concentration along the line is low and the line significant, while the concentration of the whole ring
    #   is still high
    point_line = point/2.0
    for angle in (0.0, 0.2, 0.4, 1.2, 2.0, 2.9):
        across = -xx*np.sin(angle) + yy*np.cos(angle)
        stack = point_line + np.exp(-0.5*across**2/1.5**2) + noise
        concentration, significance = mfd.extentLine(stack, PSF_SIGMA, n)
        assert (concentration < 0.5) and (significance > mfd.LINE_MIN_SIGNIFICANCE)
        assert mfd.extentConcentration(stack, PSF_SIGMA) > 0.6

    # Noise alone: the brightest axis is not significant
    significances = [mfd.extentLine(rng.normal(0, 1/np.sqrt(n), xx.shape), PSF_SIGMA, n)[1] for _ in range(50)]
    assert max(significances) < mfd.LINE_MIN_SIGNIFICANCE


def test_extent_concentration_fast_streak():
    """ An object moving fast enough to be a long streak in every frame (up to 22 px/frame) lights the ring of the
        concentration along its motion: with the motion given, its streak is left out and it is a point source,
        while a cloud-like structure moving the same way is still extended.
    """

    r = mfd.EXTENT_RADIUS
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(float)
    for speed in (6.0, 10.0, 15.0, 22.0):
        motion = speed*np.array([np.cos(0.3), np.sin(0.3)])

        # The PSF smeared over the frame along the motion
        streak = sum(np.exp(-0.5*((xx - motion[0]*d)**2 + (yy - motion[1]*d)**2)/PSF_SIGMA**2)
                     for d in np.linspace(-0.5, 0.5, 41))/41
        assert mfd.extentConcentration(streak, PSF_SIGMA, motion=motion) > 0.95

        # A smooth structure as wide as the stack, with the streak on top of it
        blob = np.exp(-0.5*(xx**2 + yy**2)/6.0**2)
        assert mfd.extentConcentration(blob + 0.3*streak, PSF_SIGMA, motion=motion) < 0.6

    # A PSF so wide that its streak covers the whole ring: the whole ring is used, the concentration is a number
    wide = np.exp(-0.5*(xx**2 + yy**2)/5.0**2)
    assert np.isfinite(mfd.extentConcentration(wide, 5.0, motion=(3.0, 1.0)))

    # Without the motion, the long streak reaches the ring (as before the motion was used)
    motion = 22.0*np.array([np.cos(0.3), np.sin(0.3)])
    streak = sum(np.exp(-0.5*((xx - motion[0]*d)**2 + (yy - motion[1]*d)**2)/PSF_SIGMA**2)
                 for d in np.linspace(-0.5, 0.5, 41))/41
    assert mfd.extentConcentration(streak, PSF_SIGMA) < 0.85


def test_pieces_of_the_same_object_on_its_path():
    """ A track on the path of another (a piece of the same object, also ahead of it along the path) is on its
        path; a track moving the same way 15 px beside it is not.
    """

    f = np.arange(0, 100)
    path = (f, 100 + 6.0*f, 50 + 2.0*f)
    piece = (f[20:40], 100 + 6.0*f[20:40] + 0.4, 50 + 2.0*f[20:40] - 0.3)
    ahead = (f[20:40], 100 + 6.0*(f[20:40] + 3), 50 + 2.0*(f[20:40] + 3))
    beside = (f[20:40], 100 + 6.0*f[20:40] - 15*2.0/np.hypot(6, 2), 50 + 2.0*f[20:40] + 15*6.0/np.hypot(6, 2))
    assert mfd.onPath(piece, path, 5.0) and mfd.onPath(path, piece, 5.0)
    assert mfd.onPath(ahead, path, 5.0) and mfd.onPath(path, ahead, 5.0)
    assert not mfd.onPath(beside, path, 5.0) and not mfd.onPath(path, beside, 5.0)

    # Positions 22 px apart along the path (a frame apart): a track 12 px beside it is not on it
    fast = (f, 100 + 22.0*f, 50 + 0.0*f)
    beside_fast = (f[20:40], 100 + 22.0*f[20:40], 62 + 0.0*f[20:40])
    assert not mfd.onPath(beside_fast, fast, 6.0)


def test_extent_line_fast_object_not_a_line():
    """ An object moving fast enough to be a streak in every frame lights the ring along its motion: that axis is
        left out when the motion is given, so it is not taken for a linear structure, while a line across the
        motion still is.
    """

    r = mfd.EXTENT_RADIUS
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(float)
    n = 200

    # A streak of 12 px per frame along the direction 0.5 rad
    angle = 0.5
    vx, vy = 12*math.cos(angle), 12*math.sin(angle)
    streak = sum(0.1*np.exp(-0.5*((xx - vx*d)**2 + (yy - vy*d)**2)/PSF_SIGMA**2) for d in np.linspace(-0.5, 0.5, 41))

    # Without the motion it looks like a line, with it it doesn't
    assert mfd.extentLine(streak, PSF_SIGMA, n)[0] < 0.5
    assert mfd.extentLine(streak, PSF_SIGMA, n, motion=(vx, vy))[0] > 0.5

    # A line across the motion (as bright as the streak) is still found
    across = -xx*math.sin(angle + np.pi/2) + yy*math.cos(angle + np.pi/2)
    stack = streak + 1.5*streak.max()*np.exp(-0.5*across**2/1.5**2)
    assert mfd.extentLine(stack, PSF_SIGMA, n, motion=(vx, vy))[0] < 0.5

    # A slow object: the axis along the motion is kept, a line along it is found
    along = -xx*math.sin(angle) + yy*math.cos(angle)
    stack = np.exp(-0.5*(xx**2 + yy**2)/PSF_SIGMA**2) + np.exp(-0.5*along**2/1.5**2)
    assert mfd.extentLine(stack, PSF_SIGMA, n, motion=(0.1*math.cos(angle), 0.1*math.sin(angle)))[0] < 0.5


def test_track_left_without_motion_after_the_faint_ends_are_cut_is_dropped(config):
    """ A joined track which moves enough only thanks to its faint head (a piece over empty sky joined to an
        object which barely moves) is not a detection once the head is cut. """

    det = MatchedFilterDetector(_SynthHandle([], total_frames=256), config)
    det.psf_sigma = PSF_SIGMA

    # A faint piece of 80 frames, then the bright object for 30 frames, moving less than a detection has to
    v = 0.4*det.minDisplacement()/30
    def piece(f0, n, snr):
        """ Rows of n frames from frame f0 on the line of the track, at the given SNR. """
        cent = np.zeros((n, 7))
        cent[:, 0] = f0 + np.arange(n)
        cent[:, 1] = 40.0 + v*cent[:, 0]
        cent[:, 2] = 60.0
        cent[:, 5] = snr
        return cent
    head, obj = piece(0, 80, 1.0), piece(80, 30, 30.0)
    assert len(det.joinDetections([[0.0, 0.0, obj]], connected=lambda *args: None)) == 1

    assert det.joinDetections([[0.0, 0.0, head], [0.0, 0.0, obj]], connected=lambda *args: None) == []


def test_measured_track_is_not_cut_again_when_not_joined(config):
    """ The faint ends of a track are cut once: a track already cut (when it was measured) is not cut again by the
        joining of the tracks (the level of the cut follows the track, so a second cut could shorten it further),
        while a track made since (e.g. by the stitching of segments) is cut. """

    det = MatchedFilterDetector(_SynthHandle([], total_frames=128), config)
    det.psf_sigma = PSF_SIGMA

    # A faint head of 25 rows, then the object for 40 rows
    n = 65
    cent = np.zeros((n, 7))
    cent[:, 0] = np.arange(n)
    cent[:, 1] = 20.0 + 2.0*np.arange(n)
    cent[:, 2] = 60.0
    cent[:, 5] = np.r_[np.full(25, 1.0), np.full(n - 25, 30.0)]

    joined = det.joinDetections([[0.0, 0.0, cent]], connected=lambda *args: None, cut={id(cent)})
    assert (len(joined) == 1) and (joined[0][2] is cent)

    joined = det.joinDetections([[0.0, 0.0, cent]], connected=lambda *args: None, cut=set())
    assert (len(joined) == 1) and (joined[0][2][0, 0] >= 20)


def test_linear_structure_flags_per_track(config):
    """ The flag of a track rejected on a linear structure belongs to that track, also when another track has the
        same first and last frame. """

    det = MatchedFilterDetector(_SynthHandle([], total_frames=64), config)
    det.significances = [(0.0, 100.0, 20.0, 20.0, 21.0, 0.95, True), (0.0, 100.0, 20.0, 20.0, 21.0, 0.95, False)]
    summary = det.summary()
    assert [t['linear_structure'] for t in summary['tracks']] == [True, False]
    assert summary['linear_structure_rejected'] == 1


class _LineHandle(_SynthHandle):
    """ Synthetic frames with a faint line across the field which drifts slowly sideways, its brightness changing
        gently along it: the light left for a while along the path of a bright source. """

    def __init__(self, objects, amp=0.5, drift=0.06, angle=1.2, x0=40.0, y0=64.0, width=1.5, period=30.0,
                 modulation=0.3, **kwargs):

        super(_LineHandle, self).__init__(objects, **kwargs)
        self.line = (amp, drift, angle, x0, y0, width, period, modulation)

    def loadFrame(self, avepixel=False):

        # The distances of the pixels along the line and across it, which drifts by the given px per frame
        amp, drift, angle, x0, y0, width, period, modulation = self.line
        yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(float)
        along = (xx - x0)*math.cos(angle) + (yy - y0)*math.sin(angle)
        across = -(xx - x0 - drift*self.current_frame)*math.sin(angle) + (yy - y0)*math.cos(angle)
        line = amp*NOISE*(1.0 + modulation*np.cos(2*np.pi*along/period))*np.exp(-0.5*(across/width)**2)

        return super(_LineHandle, self).loadFrame(avepixel) + line.astype(np.float32)


@pytest.mark.parametrize('line_concentration', [0.5, 0.0])
def test_tracks_on_a_drifting_line_rejected(config, line_concentration):
    """ The matched filter makes tracks along a faint line which drifts slowly across the field. They are
        rejected by their concentration along the line, while a faint object elsewhere is kept.
    """

    obj = linearObject(1.5, 95, 20, 0.02, 0.18, 40, 470)
    config.mf_line_concentration = line_concentration
    det, detections = runDetector(_LineHandle([obj], n_stars=6), config)

    # The object is detected in both cases; with the test only the object, without it tracks along the line too
    real = [d for d in detections if np.median(truthError(d[2], obj)) < 1.5]
    assert len(real) == 1
    if line_concentration > 0:
        assert len(detections) == 1
        assert det.summary()['linear_structure_rejected'] > 0
    else:
        assert len(detections) > 1
        assert det.summary()['linear_structure_rejected'] == 0


def test_duplicate_along_the_motion_removed_when_the_light_is_connected():
    """ A detection which follows a longer one at a constant offset along the direction of motion, further than the
        usual limit, is a duplicate only if the image is lit all the way between them (e.g. a second measurement on
        the wing or the trail of a bright object), not if there is no light between them (two objects). """

    from RMS.Detection import removeDuplicateDetections

    # Two detections moving the same way, the second one 7 px behind the first along the motion and shorter
    frames = np.arange(100, 300, 1.0)
    first = np.c_[frames, 10 + 1.0*(frames - 100), 50 + 0.0*frames]
    second = first.copy()
    second[:, 1] -= 7.0
    second = second[5:]
    dets = [[0, 0, first], [0, 0, second]]

    # The light between them: all lit, or none
    lit = lambda frame, x1, y1, x2, y2: 1.0
    dark = lambda frame, x1, y1, x2, y2: 0.0

    # Not a duplicate with the usual limits, nor with the light unknown or not connected
    assert len(removeDuplicateDetections(dets, 1.4, 4.2)) == 2
    assert len(removeDuplicateDetections(dets, 1.4, 4.2, connected=dark, max_connected_along=14.0)) == 2

    # A duplicate when the light is connected, within the larger limit along the motion
    kept = removeDuplicateDetections(dets, 1.4, 4.2, connected=lit, max_connected_along=14.0)
    assert (len(kept) == 1) and (kept[0][2] is first)
    assert len(removeDuplicateDetections(dets, 1.4, 4.2, connected=lit, max_connected_along=5.0)) == 2


def test_duplicate_which_runs_into_the_object_removed():
    """ A shorter detection which begins away from an object and runs into it (a piece of the sums of other
        velocities through a bright object) is a duplicate once it is on the object in a third of its frames,
        while two objects which cross are not. """

    from RMS.Detection import removeDuplicateDetections
    from RMS.MatchedFilterDetection import DUPLICATE_COINCIDENT

    # The object, and a detection which joins it from 12 px across over 18 frames, then follows it for 11 frames
    #   (as in a recording of a bright object)
    frames = np.arange(100, 241, 1.0)
    obj = np.c_[frames, 900 + 0.6*(frames - 100), 10 + 2.1*(frames - 100)]
    ghost = obj[48:77].copy()
    off = np.clip(1 - np.arange(29)/18.0, 0, None)
    ghost[:, 1] -= 12*off
    dets = [[0, 0, obj], [0, 0, ghost]]

    # The medians are too far apart for the usual limits
    assert len(removeDuplicateDetections(dets, 2.2, 6.6)) == 2
    kept = removeDuplicateDetections(dets, 2.2, 6.6, min_coincident=DUPLICATE_COINCIDENT)
    assert (len(kept) == 1) and (kept[0][2] is obj)

    # A shallow crossing: a shorter object which passes across it, close to it in a third of its frames in the
    #   middle, is not a duplicate
    shallow = obj[60:81].copy()
    shallow[:, 1] += np.linspace(-6.0, 6.0, len(shallow))
    assert len(removeDuplicateDetections([[0, 0, obj], [0, 0, shallow]], 2.2, 6.6,
                                         min_coincident=DUPLICATE_COINCIDENT)) == 2

    # An object crossing it at another direction is close to it only in a few frames
    cross = np.c_[frames[30:70], 920 - 1.0*(frames[30:70] - 130), 73 + 2.1*(frames[30:70] - 130)]
    assert len(removeDuplicateDetections([[0, 0, obj], [0, 0, cross]], 2.2, 6.6,
                                         min_coincident=DUPLICATE_COINCIDENT)) == 2


def test_connected_light_along_a_trail(config):
    """ The fraction of a segment which is lit (well above the background) in a frame: along the trail of a very
        bright object from its position it is lit, towards the empty sky it is not. """

    # A very bright object with a trail, searched
    obj = linearObject(400.0, 64, 50, 0.0, 0.1, 0, 512)
    obj['trail'] = 5.0
    det = MatchedFilterDetector(_SynthHandle([obj], total_frames=512, n_stars=2), config)
    det.run()

    # Its position in a frame
    f = 300
    x, y = obj['pos'](f)

    # Along its row (the trail is on the side of the lower columns) and its column, and towards the empty sky
    assert det.connectedLight(f, x, y, x - 7, y) > 0.9
    assert det.connectedLight(f, x, y, x + 7, y + 7) < 0.5


def test_connected_light_over_the_segments():
    """ The light between two points in a frame is taken from the detector of the segment of the input whose
        blocks contain the frame (the detections joined over the segments span several of them). """

    class _Det(object):
        def __init__(self, first, end, value):
            """ A detector of the frames first .. end, whose light between any two points is value. """
            self.block_starts, self.block_ends, self.value = [first], [end], value

        def connectedLight(self, frame, x1, y1, x2, y2):
            """ The light between two points, the same everywhere. """
            return self.value

    # Three segments, the second one not searched
    connected = mfd.connectedOver([_Det(0, 1000, 0.1), None, _Det(900, 2000, 0.9)])

    # Every frame from the first segment which contains it, none outside them
    assert connected(500, 0, 0, 1, 1) == 0.1
    assert connected(1500, 0, 0, 1, 1) == 0.9
    assert connected(5000, 0, 0, 1, 1) is None


def test_bright_object_along_a_column_detected(config):
    """ A bright object moving along a pixel column lights the column with its own trail, which makes the column
        bright in the median of a block and masked as a bright line: the line is not masked near the object, so
        it is detected over its whole passage. """

    # A bright object with a trail, moving along a column
    obj = linearObject(200.0, 60, 20, 0.0, 0.12, 20, 620)
    obj['trail'] = 1.0
    _, detections = runDetector(_SynthHandle([obj], total_frames=640, n_stars=6), config)

    # One detection, accurate, over the whole passage
    assert len(detections) == 1
    cent = detections[0][2]
    assert np.median(truthError(cent, obj)) < 0.3
    assert (cent[0, 0] < 60) and (cent[-1, 0] > 580)


def test_line_keep_only_sources_above_the_line():
    """ On a bright column which flickers as a whole above the level, no pixel is kept; a compact source on it is
        kept with its surroundings, also on a column brighter near one end (e.g. near the star of its trail). """

    # A frame of noise, and a bright column
    rng = np.random.default_rng(3)
    frame = rng.normal(0, 1, (SIZE, SIZE)).astype(np.float32)
    lines = np.zeros((SIZE, SIZE), dtype=bool)
    lines[:, 70] = True

    # The whole column 15 sigma above the sky: nothing is kept
    frame[:, 70] += 15
    assert not mfd.lineKeep(frame, lines).any()

    # The column brighter towards its top end (a trail), still nothing
    frame[:, 70] += np.linspace(20, 0, SIZE).astype(np.float32)
    assert not mfd.lineKeep(frame, lines).any()

    # A compact source of 60 sigma on it is kept, with TRAIL_KEEP px around it
    frame[80, 70] += 60
    keep = mfd.lineKeep(frame, lines)
    assert keep[80, 70] and keep[80 + mfd.TRAIL_KEEP, 70] and not keep[80 + mfd.TRAIL_KEEP + 2, 70]


class _BrightColumnHandle(_SynthHandle):
    """ Synthetic frames with a column brighter than the sky by a constant level (e.g. a bad column). """

    def __init__(self, objects, column=60, level=8.0, **kwargs):
        """ The frames of _SynthHandle with the column brighter by level times the noise. """

        super(_BrightColumnHandle, self).__init__(objects, **kwargs)
        self.column, self.level = column, level

    def loadFrame(self, avepixel=False):
        """ The current frame, with the bright column. """

        img = super(_BrightColumnHandle, self).loadFrame(avepixel)
        img[:, self.column] += np.float32(self.level*NOISE)

        return img


def test_bright_object_along_a_strong_bright_column(config):
    """ A column bright enough to be above the level of the masked stars in the median is not masked as a star
        along its length (only compact sources on it are), so a bright object moving along it is detected. """

    # A column 8 times the noise above the sky, and one 100 times (whose median spreads into the neighbouring
    #   columns when the brightness of the stars is smoothed)
    for level in (8.0, 100.0):
        obj = linearObject(200.0, 60, 5, 0.0, 0.2, 20, 620)
        _, detections = runDetector(_BrightColumnHandle([obj], level=level, total_frames=640, n_stars=6), config)

        # The object is found over its whole passage
        found = [d for d in detections if np.median(truthError(d[2], obj)) < 0.5]
        assert len(found) == 1
        assert (found[0][2][0, 0] < 60) and (found[0][2][-1, 0] > 560)


def test_bright_line_masked_before_the_trails_are_removed(config):
    """ A bright line which is above the level of the trail removal in a frame is masked before the trails of the
        bright sources are removed, so the line is not taken for a source whose trails cover the whole frame: the
        other pixels of the frame are kept. """

    # A frame of noise with a bright column, and no static sources
    det = MatchedFilterDetector(_SynthHandle([], total_frames=64), config)
    rng = np.random.default_rng(1)
    frame = rng.normal(0, 1, (SIZE, SIZE)).astype(np.float32)
    lines = np.zeros((SIZE, SIZE), dtype=bool)
    lines[:, 90] = True
    static = np.zeros((SIZE, SIZE), dtype=bool)

    # The line 40 sigma above the sky over its whole length, and a faint object elsewhere
    frame[:, 90] += 40
    frame[30, 20] += 8
    det.prepareFrame(frame, static, lines)

    # The object is kept, the line masked
    assert frame[30, 20] > 4
    assert np.all(frame[:, 90] == 0)


def test_line_keep_long_streak(config):
    """ The streak of a fast object along a bright column is kept when the running median along the line is longer
        than twice the streak (see lineWindow), not when the streak sets the level of the line. """

    # A bright column with a streak of 20 px along it
    rng = np.random.default_rng(4)
    frame = rng.normal(0, 1, (SIZE, SIZE)).astype(np.float32)
    lines = np.zeros((SIZE, SIZE), dtype=bool)
    lines[:, 70] = True
    frame[:, 70] += 5
    frame[50:70, 70] += 60

    # A window of the default length is mostly the streak, a long enough one is not
    assert not mfd.lineKeep(frame, lines)[60, 70]
    assert mfd.lineKeep(frame, lines, window=61)[60, 70]

    # A streak which reaches the edge of the image along the line is kept too (the line is not padded with it)
    edge = rng.normal(0, 1, (SIZE, SIZE)).astype(np.float32)
    edge[:, 70] += 5
    edge[0:20, 70] += 60
    assert mfd.lineKeep(edge, lines, window=61)[5, 70]

    # The window of the detector covers the fastest searched streaks
    config.mf_ang_vel_max = 30.0
    det = MatchedFilterDetector(_SynthHandle([], total_frames=64), config)
    det.psf_sigma = PSF_SIGMA
    assert det.lineWindow() > 2*(det.opts.speed_max + 2*PSF_SIGMA)


class _DriftHandle(_SynthHandle):
    """ Synthetic frames whose stars drift across the field by the given px per frame (e.g. the diurnal motion
        for a fixed camera). """

    def __init__(self, objects, drift=(0.008, 0.003), **kwargs):
        """ The frames of _SynthHandle with the stars drifting by drift (px per frame, x and y). """

        # The stars at frame 0, which are shifted for every frame
        super(_DriftHandle, self).__init__(objects, **kwargs)
        self.drift = drift
        self.base_stars = self.stars.astype(np.float32)

    def loadFrame(self, avepixel=False):
        """ The current frame, with its stars shifted by the drift. """

        # The stars of this frame, shifted by the drift (with subpixel interpolation)
        f = self.current_frame
        M = np.float32([[1, 0, self.drift[0]*f], [0, 1, self.drift[1]*f]])
        self.stars = cv2.warpAffine(self.base_stars, M, (SIZE, SIZE), flags=cv2.INTER_CUBIC)

        return super(_DriftHandle, self).loadFrame(avepixel)


def test_drifting_faint_stars_make_no_tracks(config):
    """ Faint stars (below the masking of the stars) which drift with the sky leave no residuals which make tracks:
        the medians of the neighbouring blocks are aligned with the motion of the stars before they are averaged,
        and the background is extrapolated with that motion beyond the middle of the first and the last block. """

    det, detections = runDetector(_DriftHandle([], total_frames=1100, n_stars=20, n_faint_stars=200, seed=2),
                                  config)

    # The motion of the stars is measured, and nothing is detected
    assert np.allclose(det.drift(), (0.008, 0.003), atol=0.001)
    assert len(detections) == 0


def test_star_drift_from_the_middles_of_the_blocks(config):
    """ The drift of the stars is the shift between the medians of two blocks over the time between their middles
        (which the medians represent), also when the last block is shorter than the others. """

    # Four blocks, the last one shorter
    det = MatchedFilterDetector(_SynthHandle([], total_frames=896), config)
    det.block_starts = [0, 256, 512, 768]
    det.block_ends = [256, 512, 768, 896]

    # The medians of a field of stars shifted by the drift to the middle of every block
    rng = np.random.default_rng(5)
    field = np.zeros((SIZE, SIZE), dtype=np.float32)
    for _ in range(60):
        renderObject(field, rng.uniform(10, SIZE - 10), rng.uniform(10, SIZE - 10), rng.uniform(5, 30)*NOISE)
    drift = (0.01, -0.004)
    for k, (first, last) in enumerate(zip(det.block_starts, det.block_ends)):
        mid = (first + last - 1)/2.0
        M = np.float32([[1, 0, drift[0]*mid], [0, 1, drift[1]*mid]])
        bg = mfd.BlockBackground(np.zeros_like(field), np.ones_like(field), 1)
        bg.raw_median = SKY + cv2.warpAffine(field, M, (SIZE, SIZE), flags=cv2.INTER_CUBIC)
        det.backgrounds[k] = bg

    # The pair of blocks 0 and 3 (3 blocks apart): their middles are 704 frames apart, their starts 768
    assert np.allclose(det.starDrift(step=3), drift, rtol=0, atol=0.0006)

    # The drift is also measured on the first use when the backgrounds were set without the background pass
    assert np.allclose(det.drift(), drift, rtol=0, atol=0.0006)


def test_background_around_a_track_aligned_with_the_stars(config):
    """ The background which is estimated again around a track without the object (see maskObjects) takes the
        samples of the neighbouring blocks shifted by the motion of the stars, as the rest of the background of
        the block, so the stars in it are where they are at the middle of the block. """

    # Many drifting stars, and the backgrounds of five blocks
    drift = (0.01, 0.0)
    handle = _DriftHandle([], total_frames=1280, n_stars=400, drift=drift, seed=4)
    det = MatchedFilterDetector(handle, config)
    det.frame_first, det.frame_end = 0, 1280
    det.block_starts, det.block_ends = [0, 256, 512, 768, 1024], [256, 512, 768, 1024, 1280]
    det.psf_sigma = PSF_SIGMA
    det.backgroundPass()

    # A track across the middle block, and the background around it estimated again
    k = 2
    mid = (det.block_starts[k] + det.block_ends[k] - 1)/2.0
    track = np.array([[f, 30 + 0.05*(f - 512), 64.0] for f in range(512, 768, 8)], dtype=np.float64)
    det.maskObjects([track])
    after = det.backgrounds[k].median

    # The stars at the middle of the block
    M = np.float32([[1, 0, drift[0]*mid], [0, 1, drift[1]*mid]])
    truth = SKY + cv2.warpAffine(handle.base_stars, M, (SIZE, SIZE), flags=cv2.INTER_CUBIC)

    # In the region estimated again, away from the object, the background is the stars at the middle of the block
    #   within the noise of the median (unaligned samples blur the stars by the motion over three blocks)
    region = (slice(57, 72), slice(23, 48))
    yy, xx = np.mgrid[region[0], region[1]]
    away = np.abs(yy - 64) > 5.5
    assert np.std((after[region] - truth[region])[away]) < NOISE


def test_noise_around_a_track_not_reduced_by_the_alignment(config):
    """ The noise of the background estimated again around a track (see maskObjects) is the noise of the frames,
        although the samples are shifted with subpixel interpolation, which smooths their noise. """

    # Pure noise, with a drift of fractions of a pixel (so the samples are interpolated), in five blocks
    handle = _DriftHandle([], total_frames=1280, n_stars=0, drift=(0.0031, 0.0023), seed=6)
    det = MatchedFilterDetector(handle, config)
    det.frame_first, det.frame_end = 0, 1280
    det.block_starts, det.block_ends = [0, 256, 512, 768, 1024], [256, 512, 768, 1024, 1280]
    det.psf_sigma = PSF_SIGMA
    det.backgroundPass()
    det._drift = (0.0031, 0.0023)

    # A track across the middle block, and the background around it estimated again
    k = 2
    track = np.array([[f, 30 + 0.05*(f - 512), 64.0] for f in range(512, 768, 8)], dtype=np.float64)
    det.maskObjects([track])

    # The noise in the region estimated again, away from the object, is the noise of the frames
    noise = det.backgrounds[k].noise[57:72, 23:48]
    yy = np.mgrid[57:72, 23:48][0]
    assert abs(np.median(noise[np.abs(yy - 64) > 5.5])/NOISE - 1) < 0.1


def test_star_drift_of_a_short_input(config):
    """ The drift of the stars, with which the backgrounds of the blocks are aligned, is measured also for an input
        of fewer blocks than the separation used for longer inputs (e.g. a short segment). """

    # Three blocks of drifting stars
    handle = _DriftHandle([], total_frames=768, n_stars=60, drift=(0.01, -0.004), seed=8)
    det = MatchedFilterDetector(handle, config)
    det.frame_first, det.frame_end = 0, 768
    det.block_starts, det.block_ends = [0, 256, 512], [256, 512, 768]
    det.psf_sigma = PSF_SIGMA
    det.backgroundPass()

    # The drift is measured over the blocks there are
    assert np.allclose(det.drift(), (0.01, -0.004), rtol=0, atol=0.001)


def test_background_around_a_track_without_the_object(config):
    """ The region estimated again around a track (see maskObjects) is large enough for the light of the object in
        the medians of the neighbouring blocks after they are aligned with the motion of the stars, so none of it
        is left in the background of the block. """

    # A bright object which stays in place while the sky drifts fast, in five blocks
    drift = (0.04, 0.0)
    obj = linearObject(30.0, 60, 64, 0.0, 0.0, 0, 1280)
    handle = _DriftHandle([obj], total_frames=1280, n_stars=0, drift=drift, seed=9)
    det = MatchedFilterDetector(handle, config)
    det.frame_first, det.frame_end = 0, 1280
    det.block_starts, det.block_ends = [0, 256, 512, 768, 1024], [256, 512, 768, 1024, 1280]
    det.psf_sigma = PSF_SIGMA

    # The motion of the stars given (the synthetic frames have no stars to measure it on)
    det.starDrift = lambda step=mfd.SLOW_REF_BLOCKS: drift
    det.backgroundPass()

    # Its track, and the background of the middle block estimated again around it
    k = 2
    track = np.array([[f, *obj['pos'](f)] for f in range(0, 1280, 8)], dtype=np.float64)
    det.maskObjects([track])

    # No light of the object above the sky anywhere near its path in the background of the block
    region = det.backgrounds[k].median[50:79, 20:110]
    assert np.max(region) - SKY < 3*NOISE


def test_track_deviation_from_one_smooth_curve():
    """ The deviation of a track across its motion from one smooth path (a quadratic in time) over its whole
        length: about 0 for a straight or a uniformly curved track, also with a little noise or a changing speed
        along the path, and large for a track which wanders sideways. """

    # Straight, curved, curved with noise, and wandering tracks
    t = np.arange(0, 600, 1.0)
    rng = np.random.default_rng(2)
    straight = np.c_[t, 100 + 0.3*t, 50 + 0.1*t]
    curved = np.c_[t, 100 + 0.3*t + 2e-4*t**2, 50 + 0.1*t - 1e-4*t**2]
    noisy = curved.copy()
    noisy[:, 1:3] += rng.normal(0, 0.1, (len(t), 2))
    wandering = np.c_[t, 100 + 0.3*t + 3*np.sin(2*np.pi*t/200), 50 + 0.1*t + 3*np.cos(2*np.pi*t/170)]

    # The speed along a straight path changing more than a quadratic follows (e.g. a long pass)
    speeding = np.c_[t, 100 + 0.3*t + 20*np.sin(2*np.pi*t/600)*0.3, 50 + 0.1*t + 20*np.sin(2*np.pi*t/600)*0.1]

    # Only the wandering track deviates
    assert mfd.trackDeviation(straight) < 0.01
    assert mfd.trackDeviation(speeding) < 0.01
    assert mfd.trackDeviation(curved) < 0.01
    assert mfd.trackDeviation(noisy) < 0.15
    assert mfd.trackDeviation(wandering) > 2

    # Too few rows to judge: no deviation
    assert mfd.trackDeviation(wandering[:4]) == 0.0

    # The detections deviating more than the limit (a fraction of the FWHM plus of the length) are removed, none
    #   with the fraction of the FWHM 0
    dets = [[0, 0, straight], [0, 0, wandering]]
    kept, rejected = mfd.removeDeviatingDetections(dets, 1.6, 0.6, 0.005)
    assert (len(kept) == 1) and (kept[0][2] is straight) and (len(rejected) == 1)
    assert len(mfd.removeDeviatingDetections(dets, 1.6, 0.0, 0.005)[0]) == 2

    # The allowed deviation grows with the length of the track: the same wander on a track 10 times longer (as a
    #   long track across a wide field bends) is kept
    long_wandering = np.c_[t, 100 + 3.0*t + 3*np.sin(2*np.pi*t/200), 50 + 1.0*t + 3*np.cos(2*np.pi*t/170)]
    assert len(mfd.removeDeviatingDetections([[0, 0, long_wandering]], 1.6, 0.6, 0.005)[0]) == 1


def test_track_deviation_with_the_fwhm_of_each_track():
    """ The FWHM which the deviation of a track is scaled with can be given per track (e.g. the PSF of the segment
        of an input which the track belongs to). """

    # The same wandering track early and late in the input
    t = np.arange(0, 600, 1.0)
    wandering = np.c_[t, 100 + 0.3*t + 1.5*np.sin(2*np.pi*t/200), 50 + 0.1*t + 1.5*np.cos(2*np.pi*t/170)]
    late = wandering.copy()
    late[:, 0] += 5000
    dets = [[0, 0, wandering], [0, 0, late]]

    # A narrow PSF early in the input and a wide one later: only the early track is rejected
    fwhm = lambda cent: 1.0 if cent[0, 0] < 5000 else 4.0
    kept, rejected = mfd.removeDeviatingDetections(dets, fwhm, 0.6, 0.0)
    assert (len(kept) == 1) and (kept[0][2] is late) and (len(rejected) == 1)


@pytest.mark.parametrize('deviation_fwhm', [0.3, 0.0])
def test_wandering_track_rejected(config, deviation_fwhm):
    """ A track which wanders sideways from one smooth path (by 4 px back and forth) is rejected by the whole detector,
        a straight object at the same time is kept; with the test off both are detected. """

    # A straight object, and one which wanders sideways at the same time
    straight = linearObject(4.0, 20, 30, 0.2, 0.05, 40, 470)
    wander = {'snr': 4.0, 'f0': 40, 'f1': 470,
              'pos': lambda f: (60 + 4*np.sin(2*np.pi*(f - 40)/200), 110 - 0.2*(f - 40))}

    # The PSF of the synthetic frames is wide (FWHM 2.4 px), and the measured wander about 1.4 px RMS, so the test
    #   is run with a smaller fraction of the FWHM than the default
    config.mf_track_deviation_fwhm = deviation_fwhm
    det, detections = runDetector(_SynthHandle([straight, wander], n_stars=4), config)

    # The straight object is always found, the wandering one only with the test off
    found_straight = [d for d in detections if np.median(truthError(d[2], straight)) < 1.5]
    found_wander = [d for d in detections if np.median(truthError(d[2], wander)) < 3]
    assert len(found_straight) == 1
    if deviation_fwhm > 0:
        assert len(found_wander) == 0
        assert det.summary()['deviation_rejected'] >= 1
    else:
        assert len(found_wander) >= 1


### Light curves, bright objects, photometric scale ###

def test_flashes_kept_in_light_curve(config):
    """ An object below the single-frame limit which flashes for one frame every 24 frames: it is found, and
        the light curve has the flashes at their full brightness (the positions of faint objects are measured on
        several frames, but the intensity of every frame is kept).
    """

    # A peak SNR of 1.5 between the flashes and 40 in the flashes
    flash_snr, base_snr = 40.0, 1.5
    flashes = set(range(110, 460, 24))
    snr = lambda f: flash_snr if int(f) in flashes else base_snr
    obj = linearObject(1.0, 25, 30, 0.25, 0.1, 100, 460)
    obj['snr'] = snr
    _, detections = runDetector(_SynthHandle([obj]), config)

    assert len(detections) == 1
    cent = detections[0][2]
    flux = lambda s: s*NOISE*2*np.pi*PSF_SIGMA**2

    # Most flashes have a row, with the flux of the flash; the frames between them have the flux of the object
    #   between the flashes (with the larger scatter of the faint intensities)
    at_flash = np.isin(np.round(cent[:, 0]).astype(int), list(flashes))
    assert at_flash.sum() >= 0.8*len(flashes)
    assert abs(np.median(cent[at_flash, 3])/flux(flash_snr) - 1) < 0.15
    assert abs(np.median(cent[~at_flash, 3])/flux(base_snr) - 1) < 0.3


def test_spilled_charge_flagged_and_counted(config):
    """ A very bright object whose charge spills along its row and column (without saturated pixels): the
        measurements are flagged as saturated and the intensity includes the spilled charge.
    """

    # The spilled charge is as large as the flux of the object itself
    obj = linearObject(400.0, 30, 64, 0.5, 0.05, 100, 200)
    obj['spill'] = 1.0
    config.mf_smooth_frames = 0

    # No pixel reaches the saturation level
    config.mf_saturation_level = 10**6
    _, detections = runDetector(_SynthHandle([obj], n_stars=4), config)

    # Flagged, and the intensity is the flux of the object plus the spilled charge (twice the flux)
    assert len(detections) == 1
    cent = detections[0][2]
    total = 400.0*NOISE*2*np.pi*PSF_SIGMA**2*2.0
    assert np.median(cent[:, 6]) >= 1
    assert abs(np.median(cent[:, 3])/total - 1) < 0.1

    # A bright object without spill is not flagged
    obj['spill'] = 0.0
    _, detections = runDetector(_SynthHandle([obj], n_stars=4), config)
    assert np.all(detections[0][2][:, 6] == 0)


def test_intensities_on_the_scale_of_the_stars(config):
    """ The intensities are scaled to the intensities of the stars of the calibration: here the stars are
        listed 10% brighter than their flux, so the intensity of the object is 10% above its flux.
    """

    # The stars of two chunks (the same stars) with intensities 1.1 times their flux, as the star extraction
    #   lists them: (y, x, intensity, amplitude, fwhm, background, snr, saturated pixels)
    obj = linearObject(8.0, 20, 40, 0.4, 0.1, 100, 300)
    handle = _SynthHandle([obj], n_stars=30)
    stars = [(y, x, 1.1*flux, 0, 2.0, SKY, 50.0, 0) for x, y, flux in handle.star_list]
    star_list = [['FF_XX0001_20261008_030000_000_0000000.fits', stars],
                 ['FF_XX0001_20261008_030010_000_0000256.fits', stars]]
    config.mf_smooth_frames = 0

    # With the stars and without them
    detections, detector = mfd.detectMatchedFilter(handle, config, star_list=star_list, return_detector=True)
    plain = mfd.detectMatchedFilter(_SynthHandle([obj], n_stars=30), config)

    # The aperture correction is the 1.1 of the stars, the intensities are scaled by it, and it is in the
    #   summary
    assert len(detections) == len(plain) == 1
    ratio = np.median(detections[0][2][:, 3])/np.median(plain[0][2][:, 3])
    assert abs(detector.aperture_correction/1.1 - 1) < 0.05
    assert abs(ratio/detector.aperture_correction - 1) < 0.01
    assert detector.summary()['aperture_correction'] == detector.aperture_correction


### Robustness ###

def test_strong_trail_is_not_restored(config):
    """ The trails of a very bright object which are themselves above trail_level are removed as well (only
        the object itself is kept).
    """

    # Trails of 30 and 60 sigma, above the default trail level of 25
    obj = linearObject(4000.0, 100, 50, -0.6, 0.1, 100, 260)
    obj['trail'] = 30.0
    _, detections = runDetector(_SynthHandle([obj], n_stars=4), config)

    assert len(detections) == 1


def test_no_edge_margin(config):
    """ With no edge margin, the image is not masked. """

    config.mf_edge_margin = 0
    config.detection_border = 0
    obj = linearObject(2.5, 25, 30, 0.3, 0.12, 40, 470)
    _, detections = runDetector(_SynthHandle([obj]), config)

    assert len(detections) == 1


def test_too_few_frames(config):
    """ An input shorter than the longest run of the search has no detections (and doesn't fail). """

    _, detections = runDetector(_SynthHandle([], total_frames=4), config)
    assert detections == []


def test_calibration_of_the_caller_not_binned(config):
    """ With detection binning, the caller's mask is not binned again (the normal detection may have binned it
        already in place).
    """

    config.detection_binning_factor = 2
    mask = MaskStructure(np.full((SIZE, SIZE), 255, dtype=np.uint8))

    # An input which gives frames binned 2x2, as the frame interface does with detection binning
    class _Binned(_SynthHandle):
        def loadFrame(self, avepixel=False):
            img = super(_Binned, self).loadFrame()
            return img.reshape(SIZE//2, 2, SIZE//2, 2).mean(axis=(1, 3)).astype(np.float32)

    # The mask given to the detection keeps its unbinned size
    handle = _Binned([linearObject(4.0, 20, 30, 0.4, 0.1, 50, 300)])
    handle.nrows = handle.ncols = SIZE//2
    mfd.detectMatchedFilter(handle, config, mask=mask)
    assert mask.img.shape == (SIZE, SIZE)


def test_slow_object_kept_after_background_without_it(config):
    """ A slow object is measured again on a background estimated without it; the star template and the masks
        made with it in the background must be updated too, or its own path is left out of the verification.
    """

    # An object moving 27 px over 2048 frames (0.013 px/frame), at two brightnesses
    for snr in (6.0, 10.0):
        obj = linearObject(snr, 30, 60, 0.012, 0.006, 0, 2048)
        _, detections = runDetector(_SynthHandle([obj], n_stars=4, total_frames=2048), config)
        assert len(detections) == 1, snr


def test_smoothing_keeps_faint_measurements():
    """ Measurements of a track whose brightness changes: the faint ones scatter more, in proportion to their
        errors, and are not outliers.
    """

    # A measurement every frame, bright (SNR 20) for 340 frames and then faint (SNR 3), with position errors
    #   inversely proportional to the SNR
    rng = np.random.default_rng(3)
    t = np.arange(400, dtype=float)
    snr = np.where(t < 340, 20.0, 3.0)
    err = 0.5/snr
    cent = np.column_stack([t, 10 + 0.1*t + rng.normal(0, err), 20 + 0.05*t + rng.normal(0, err),
                            np.ones_like(t), np.ones_like(t), snr])
    # Nearly all faint measurements are kept, as their distances from the track are measured in units of their
    #   errors
    _, keep = smoothPositions(cent, 64)

    assert keep[t >= 340].mean() > 0.95


def test_merge_tracks_through_a_bridge(config):
    """ Pieces of one track: A and C are too far apart to be merged directly, B fills the gap; and a sparse
        track of 32-frame runs which overlaps a track of 8-frame runs in its gap.
    """

    # Hits along one straight track, rows [frame, x, y, z, vx, vy, run]
    det = MatchedFilterDetector(_SynthHandle([]), config)
    row = lambda f, run: [f, 20 + 0.1*f, 30 + 0.05*f, 6.0, 0.1, 0.05, run]

    # A ends at frame 400 and C begins at 600, a gap longer than the merging allows; B and B2 fill it
    a = np.array([row(f, 8) for f in np.arange(3.5, 400, 8)])
    b = np.array([row(f, 8) for f in np.arange(403.5, 460, 8)])
    c = np.array([row(f, 8) for f in np.arange(600 + 3.5, 1000, 8)])
    b2 = np.array([row(f, 8) for f in np.arange(459.5, 600, 8)])
    assert len(det.mergeTracks([a, c, b, b2])) == 1

    # Hits of 32-frame runs with a gap, and hits of 8-frame runs within the gap
    sparse = np.array([row(f, 32) for f in (15.5, 47.5, 175.5, 207.5)])
    dense = np.array([row(f, 8) for f in np.arange(59.5, 164, 8)])
    assert len(det.mergeTracks([sparse, dense])) == 1


def test_merged_track_stays_one_object(config):
    """ Tracks are merged pairwise: a track A, two hits which continue A and then turn away (e.g. noise, or a
        piece of another object), and a track B which continues those two hits. A and the two hits pass the tests
        of the merging, and the merged track would then be continued by B, giving one track of two objects. The
        union of A and the two hits doesn't move as one object, so they are not merged, and A and B stay apart.
    """

    det = MatchedFilterDetector(_SynthHandle([]), config)
    row = lambda f, x, y: [f, x, y, 8.0, 1.0, 0.0, 8, 2]

    # A moves along x at 1 px per frame; the two hits continue it and then turn away by 20 px
    a = np.array([row(f, 100 + f, 200.0) for f in np.arange(3.5, 100, 8)])
    n = np.array([row(107.5, 207.5, 200.0), row(115.5, 215.5, 220.0)])

    # B continues the turn of the two hits, at the velocity of the end of the track A and the two hits would have
    vx, vy = det._endVelocity(det._union(a, n))
    b = np.array([row(f, 215.5 + vx*(f - 115.5), 220.0 + vy*(f - 115.5)) for f in np.arange(123.5, 220, 8)])

    merged = det.mergeTracks([a, n, b])

    # No track contains hits of both A and B
    for tr in merged:
        assert not ((tr[:, 0] < 100).any() and (tr[:, 0] > 120).any())

    # The pieces of one straight track are still merged
    c = np.array([row(f, 100 + f, 200.0) for f in np.arange(107.5, 220, 8)])
    assert len(det.mergeTracks([a, c])) == 1


def test_merge_with_shared_neighbour_times(config):
    """ Two 2-hit tracks of one object from two bins of the search share both of their times: the neighbours of a
        hit are then at one time only, which gives no path, and the tracks are still merged.
    """

    det = MatchedFilterDetector(_SynthHandle([]), config)
    a = np.array([[20.0, 120.0, 200.0, 8.0, 1.0, 0.0, 8, 2], [28.0, 128.0, 200.0, 8.0, 1.0, 0.0, 8, 2]])
    b = np.array([[20.0, 120.5, 200.3, 8.0, 1.0, 0.0, 8, 4], [28.0, 128.4, 199.8, 8.0, 1.0, 0.0, 8, 4]])
    assert det._validMerge(a, b, 4.0)
    assert len(det.mergeTracks([a, b])) == 1


def test_ghosts_of_bright_objects_removed(config):
    """ The sums of the search along other velocities through a bright object give tracks along its path at other
        speeds (ghosts), which are removed. A fainter object on the same path at the same speed (e.g. a train of
        objects) and an object crossing the path are kept.
    """

    det = MatchedFilterDetector(_SynthHandle([]), config)
    row = lambda f, x, y, z: [f, x, y, z, 2.0, 0.0, 8, 2]

    # The bright object, at 2 px per frame along x
    bright = np.array([row(f, 100 + 2*f, 300.0, 100.0) for f in np.arange(3.5, 200, 8)])

    # Ghosts: on its path at 3 px per frame and at 1 px per frame, during its time
    ghost_fast = np.array([row(f, 150 + 3*(f - 40), 300.5, 20.0) for f in (43.5, 51.5, 59.5)])
    ghost_slow = np.array([row(f, 300 + 1*(f - 100), 299.5, 20.0) for f in (99.5, 107.5, 115.5, 123.5)])

    # A fainter object following on the path at the same speed, and one crossing the path
    follower = np.array([row(f, 40 + 2*f, 300.0, 20.0) for f in np.arange(35.5, 200, 8)])
    crossing = np.array([row(f, 250.0, 200 + 2*f, 20.0) for f in np.arange(3.5, 100, 8)])

    tracks = [bright, ghost_fast, ghost_slow, follower, crossing]
    kept = det.removeGhosts(tracks)

    assert len(kept) == 3
    assert any(tr is bright for tr in kept)
    assert any(tr is follower for tr in kept)
    assert any(tr is crossing for tr in kept)


def test_streak_duplicates_removed():
    """ A fast object (10 px per frame) measured twice, at the middle of its streak and 11 px ahead of it, with the
        same light curve, is one object; two objects moving together 15 px apart with their own light curves, and
        two slow objects close together, are two. Objects of steady brightness moving together are one object only
        with light between them.
    """

    from RMS.MatchedFilterDetection import removeStreakDuplicates

    f = np.arange(100.0, 140.0)
    rng = np.random.default_rng(3)
    light = 1000*(1.5 + np.sin(f/3.0))
    det = lambda x, y, inten: [0.0, 0.0, np.column_stack([f, x, y, inten, np.zeros_like(f), np.full_like(f, 10.0),
                                                         np.zeros_like(f)])]

    # The object, and its second measurement 11 px ahead and 5 px across, a little shorter
    obj = det(10*f, 0.15*10*f, light)
    second = det(10*f + 11 + rng.normal(0, 0.3, len(f)), 0.15*10*f - 5, 0.8*light)
    second[2] = second[2][3:]
    assert len(removeStreakDuplicates([obj, second], 2.4)) == 1

    # Two objects moving together, 15 px apart (within the reach of a second measurement), each with its own light
    #   curve
    other = det(10*f + 15, 0.15*10*f, 1000*(1.5 + np.cos(f/5.0)))
    assert len(removeStreakDuplicates([obj, other], 2.4)) == 2

    # Two slow objects close together (e.g. two objects at nearly the same place) with flat light curves
    slow_a = det(500 + 0.05*f, 300.0 + 0*f, np.full_like(f, 800.0))
    slow_b = det(505 + 0.05*f, 302.0 + 0*f, np.full_like(f, 790.0))
    assert len(removeStreakDuplicates([slow_a, slow_b], 2.4)) == 2

    # Objects of steady brightness moving together 12 px apart: two objects without light between them, one object
    #   with its streak between them
    steady = det(10*f, 0.15*10*f, np.full_like(f, 900.0))
    steady_ahead = det(10*f + 12, 0.15*10*f, np.full_like(f, 880.0))
    dark = lambda frame, x1, y1, x2, y2: 0.0
    lit = lambda frame, x1, y1, x2, y2: 1.0
    assert len(removeStreakDuplicates([steady, steady_ahead], 2.4)) == 2
    assert len(removeStreakDuplicates([steady, steady_ahead], 2.4, connected=dark)) == 2
    assert len(removeStreakDuplicates([steady, steady_ahead], 2.4, connected=lit)) == 1


def test_flashes_without_light_between_are_linked(config):
    """ An object seen only in its flashes (2 frames every 16 frames): the hits of the runs with a flash have a
        poorly known velocity, so they are linked by their positions. The flashes count fully in the search only
        with a higher clipping level of the frames.
    """

    # Flashes of a peak SNR of 60, nothing between them
    flashes = {f for f in range(100, 320) if (f - 100) % 16 < 2}
    obj = linearObject(1.0, 15, 20, 0.4, 0.3, 100, 320)
    obj['snr'] = lambda f: 60.0 if int(f) in flashes else 0.0
    config.mf_clip = 10.0
    config.mf_saturation_level = 10**6
    _, detections = runDetector(_SynthHandle([obj], n_stars=4), config)

    # One detection, with the positions in the frames of the flashes accurate
    assert len(detections) == 1
    cent = detections[0][2]
    assert np.median(truthError(cent[np.isin(np.round(cent[:, 0]).astype(int), list(flashes))], obj)) < 0.3


@pytest.mark.parametrize('measurable_from_block', [1, None])
def test_psf_measured_on_a_later_block(config, monkeypatch, measurable_from_block):
    """ The PSF sigma is measured on the first block in which it can be measured (e.g. the first blocks are
        clouded), and 1 px is used if it can't be measured on any block.
    """

    # The PSF is measured, not given
    config.mf_psf_sigma = 0

    # The measurement fails on the blocks before measurable_from_block, and gives 0.8 px from it on
    calls = []

    def _estimate(image, noise, **kwargs):
        calls.append(len(calls))
        if (measurable_from_block is None) or (len(calls) - 1 < measurable_from_block):
            return None
        return 0.8

    monkeypatch.setattr(mfd, 'estimatePSFSigma', _estimate)

    obj = linearObject(3.0, 25, 30, 0.3, 0.12, 40, 470)
    det, detections = runDetector(_SynthHandle([obj]), config)

    # Measured on the second block and not again; or tried on every block and 1 px used
    if measurable_from_block is None:
        assert len(calls) == len(det.block_starts)
        assert det.psf_sigma == 1.0
    else:
        assert len(calls) == measurable_from_block + 1
        assert det.psf_sigma == 0.8

    # The object is detected either way
    assert len(detections) == 1


def test_search_tiers(config):
    """ The speeds are searched in tiers of binning: a tier with the bin B covers the speeds up to tier_speed*B
        px per frame, and the last tier (at most max_bin) covers the speeds up to the largest one. Without tiers,
        or when the largest speed is in the first tier, all speeds are searched on the bins of the search.
    """

    det = MatchedFilterDetector(_SynthHandle([], total_frames=64), config)

    # The largest speed of the config fixture is 2 px/frame: one tier
    assert det.searchTiers() == [(2, 0.0, det.opts.speed_max)]

    # Up to 14 px/frame with tiers every 2 binned px per frame: 2x2 up to 4 px/frame, 4x4 up to 8, 8x8 the rest
    det.opts.speed_max = 14.0
    assert det.searchTiers() == [(2, 0.0, 4.0), (4, 4.0, 8.0), (8, 8.0, 14.0)]

    # At most 4x4 bins: the second tier covers the rest
    det.opts.max_bin = 4
    assert det.searchTiers() == [(2, 0.0, 4.0), (4, 4.0, 14.0)]

    # No tiers
    det.opts.tier_speed = 0
    assert det.searchTiers() == [(2, 0.0, 14.0)]

    # The velocity grid of a faster tier starts one grid step below its smallest speed, so the tiers overlap
    vel = velocityGrid(8.0, 8, 4, speed_min=4.0)
    speed = np.hypot(vel[:, 0], vel[:, 1])
    assert speed.min() >= 4.0/4 - 1.0/7 - 1e-9 and speed.min() < 4.0/4
    assert speed.max() <= 8.0/4 + 1.0/7 + 1e-9


def test_fast_object_found_on_coarse_tier(config):
    """ A fast object, long in every frame (3 px/frame, smeared during the frame), is found by a coarse tier of
        the search (4x4 bins) and measured accurately, without false detections.
    """

    # 0.04 deg/px at 25 FPS: 6 deg/s is 6 px/frame, searched on 2x2 bins up to 2 px/frame and on 4x4 bins above
    #   (tiers every binned px per frame, at most 4x4 bins: the synthetic frames are small)
    config.mf_ang_vel_max = 6.0
    config.mf_tier_speed = 1.0
    config.mf_max_bin = 4

    # A bright object crossing the image diagonally in 40 frames
    obj = linearObject(12.0, 12, 14, 2.4, 1.8, 100, 140)
    det, detections = runDetector(_SynthHandle([obj], total_frames=256), config)

    assert [t[0] for t in det.searchTiers()] == [2, 4]
    assert len(detections) == 1
    assert np.sqrt(np.mean(truthError(detections[0][2], obj)**2)) < 0.3

    # The object is faster than the first tier: without the coarse tier (speeds up to 2 px/frame), it is not found
    config.mf_ang_vel_max = 2.0
    _, detections = runDetector(_SynthHandle([obj], total_frames=256), config)
    assert len(detections) == 0


def test_fast_accelerating_object_kept(config):
    """ A bright fast object whose speed changes along its short track (1.2 to 1.8 px/frame over 80 frames) is
        kept and measured accurately: the smooth track used to test its shape follows the change of speed.
    """

    config.mf_ang_vel_max = 3.0
    accel = (1.8 - 1.2)/80
    obj = {'snr': 15.0, 'f0': 100, 'f1': 180,
           'pos': lambda f: (10 + (1.2*(f - 100) + 0.5*accel*(f - 100)**2)*0.8,
                             15 + (1.2*(f - 100) + 0.5*accel*(f - 100)**2)*0.6)}
    det, detections = runDetector(_SynthHandle([obj], total_frames=256), config)

    assert len(detections) == 1
    assert np.sqrt(np.mean(truthError(detections[0][2], obj)**2)) < 0.3


def test_coarse_tier_keeps_nearby_peaks(config):
    """ The local maxima of the hits are found within +-4 unbinned px on every tier: 5x5 pixels of 2x2 bins, the
        3x3 neighbours on 4x4 bins, so two objects 8 px apart give two hits on 4x4 bins (a 5x5 neighbourhood of
        4x4 bins, +-8 px, would keep only the brighter one).
    """

    # Two static point sources 8 px (two 4x4 bins) apart, at the centres of their bins, in frames of unit noise,
    #   searched as one run on 4x4 bins
    config.mf_ang_vel_max = 0.0
    det = MatchedFilterDetector(_SynthHandle([], total_frames=64), config)
    det.psf_sigma = PSF_SIGMA
    det.persistence_counts, det.persistence_runs = {}, {}
    rng = np.random.default_rng(5)
    z = rng.normal(0, 1, (8, SIZE, SIZE)).astype(np.float32)
    for x, amp in ((49.5, 6.0), (57.5, 4.0)):
        for k in range(8):
            renderObject(z[k], x, 65.5, amp)
    stackers = [(8, 4, VelocityStacker(velocityGrid(0.0, 8, 4), 8, use_gpu=False))]
    hits = det.searchBlock(z, np.zeros((SIZE, SIZE), dtype=bool), 0, stackers)

    # Both sources have a hit on the 4x4 bins
    xs = sorted(h.x for h in hits if abs(h.y - 65.5) < 6)
    assert any(abs(x - 49.5) < 2 for x in xs) and any(abs(x - 57.5) < 2 for x in xs)


def test_end_of_the_input_in_the_summary():
    """ The summary gives the end of the input: the time of its last frame and the length of a frame. """

    import types
    from RMS.MatchedFilterDetection import MatchedFilterDetector

    begin = datetime.datetime(2025, 12, 25, 3, 0, 0)

    class _Handle(object):
        fps = 25.0

        def currentFrameTime(self, frame_no=None, dt_obj=False):
            return begin + datetime.timedelta(seconds=frame_no/25.0)

    detector = types.SimpleNamespace(img_handle=_Handle(), total_frames=1000)
    detector.frameTimes = lambda frames: MatchedFilterDetector.frameTimes(detector, frames)
    end = MatchedFilterDetector.inputEnd(detector)
    assert end == (begin + datetime.timedelta(seconds=1000/25.0)).strftime('%Y-%m-%dT%H:%M:%S.%f')


@pytest.mark.parametrize('calibrated', [(True, True), (True, False)])
def test_merged_detections_keep_their_calibration(tmp_path, calibrated):
    """ The merged FTPdetectinfo of the matched filter is calibrated if all the merged files are, so the
        magnitudes of the detections stay usable (an uncalibrated file has placeholder magnitudes).
    """

    config = cr.Config()
    config.stationID = 'XX0001'
    dirs = []
    for i, cal in enumerate(calibrated):
        mf_dir = tmp_path/'file{:d}'.format(i)/mfd.MATCHED_FILTER_DIR
        mf_dir.mkdir(parents=True)
        (mf_dir/mfd.DONE_NAME).write_text('{}')
        picks = [[f, 10.0 + f, 20.0, 170.0, 50.0, 60.0, 70.0, 1000, 5.5 if cal else 0.0, 50, 8.0, 0]
                 for f in (3.0, 4.0, 5.0)]
        ff_name = 'FF_XX0001_20251225_0300{:02d}_000_0000000.fits'.format(10*i)
        FTPdetectinfo.writeFTPdetectinfo([[ff_name, 1, 10.0, 45.0, picks, 25.0]], str(mf_dir),
            'FTPdetectinfo_file{:d}_mf.txt'.format(i), str(mf_dir), 'XX0001', 25.0,
            calibration=('Recalibrated with RMS' if cal else None), celestial_coords_given=True)
        dirs.append(str(mf_dir.parent))

    merged = mfd.mergeNightMatchedFilter(str(tmp_path/'night'), dirs, config)
    entries = FTPdetectinfo.readFTPdetectinfo(os.path.dirname(merged), os.path.basename(merged))
    assert len(entries) == 2
    assert all(meas[0] == int(all(calibrated)) for entry in entries for meas in entry[11])


def test_streak_samples():
    """ Without the streak every frame is one sample at the shift of the middle of the frame, with the weight 1.
        With it, a velocity of L px per frame samples every frame round(L) times along the motion, centred on the
        middle of the frame, with the weight 1/sqrt(samples per frame).
    """

    from RMS.Routines.MatchedFilterKernels import velocityShifts, velocitySamples
    vel = np.array([[0.0, 0.0], [0.4, 0.2], [3.0, 0.0], [2.0, -2.0]])
    shifts = velocityShifts(vel, 8)
    samples, counts, weights = velocitySamples(vel, 8)
    assert np.all(counts == 8) and np.all(weights == 1.0)
    assert np.array_equal(samples[:, :8, :2], shifts) and np.array_equal(samples[0, :8, 2], np.arange(8))

    samples, counts, weights = velocitySamples(vel, 8, streak=True)
    per_frame = [1, 1, 3, 3]
    assert list(counts) == [8*m for m in per_frame]
    assert np.allclose(weights, 1/np.sqrt(per_frame))

    # On smoothed frames the noise of neighbouring samples is correlated, and their sum is weighted less
    _, _, weights_smooth = velocitySamples(vel, 8, streak=True, smooth_sigma=0.5)
    assert np.allclose(weights_smooth[:2], 1.0) and np.all(weights_smooth[2:] < weights[2:])

    # The 3 samples of a frame of the velocity (3, 0) are 1 px apart around the middle of the frame
    v = 2
    for k in range(8):
        dx = samples[v, 3*k:3*k + 3, 0]
        assert np.all(samples[v, 3*k:3*k + 3, 2] == k)
        assert np.array_equal(dx - dx[1], [-1, 0, 1]) and dx[1] == shifts[v, k, 0]


def test_streak_search_collects_the_streak():
    """ A fast object, a streak of 4 px in every frame: summing the whole streak raises its stack above the noise
        by about sqrt(4)/2 more than the middle alone (the streak sampling adds the light of the whole streak, and
        the noise of the stack stays the same).
    """

    rng = np.random.default_rng(11)
    n = 8
    v = np.array([4.0, 0.0])
    frames = rng.normal(0, 1, (n, 64, 64)).astype(np.float32)
    clean = np.zeros_like(frames)
    for k in range(n):
        renderObject(clean[k], 20 + v[0]*(k - (n - 1)/2.0) + 12, 32, 3.0, vx=v[0], vy=v[1], n_sub=21)
    frames += clean

    vel = np.array([[4.0, 0.0]])
    plain, _ = VelocityStacker(vel, n).stackMax(frames)
    streak, _ = VelocityStacker(vel, n, streak=True).stackMax(frames)
    noise_plain, _ = VelocityStacker(vel, n).stackMax(frames - clean)
    noise_streak, _ = VelocityStacker(vel, n, streak=True).stackMax(frames - clean)

    # The noise of the two stacks is the same, the signal of the streak stack higher
    assert abs(np.std(noise_streak)/np.std(noise_plain) - 1) < 0.1
    snr_plain = plain[32, 32]/np.std(noise_plain)
    snr_streak = streak[32, 32]/np.std(noise_streak)
    assert snr_streak > 1.4*snr_plain

    # The same on smoothed frames (correlated noise of neighbouring pixels), with the weights of that smoothing
    sigma = 0.7
    sm = np.array([cv2.GaussianBlur(f, (0, 0), sigma) for f in frames])
    sm_noise = np.array([cv2.GaussianBlur(f, (0, 0), sigma) for f in frames - clean])
    noise_plain, _ = VelocityStacker(vel, n).stackMax(sm_noise)
    noise_streak, _ = VelocityStacker(vel, n, streak=True, smooth_sigma=sigma).stackMax(sm_noise)
    assert abs(np.std(noise_streak)/np.std(noise_plain) - 1) < 0.1


def test_streak_weights_follow_the_smoothing():
    """ The weights of the streak samples can be set again for another smoothing (e.g. after the PSF was
        measured), and are then the same as for a stacker made with that smoothing.
    """

    vel = velocityGrid(6.0, 8, 2)
    stacker = VelocityStacker(vel, 8, streak=True, smooth_sigma=0.5)
    stacker.setSmoothing(0.8)
    assert np.array_equal(stacker.weights, VelocityStacker(vel, 8, streak=True, smooth_sigma=0.8).weights)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_gpu_streak_stacks_agree_with_the_cpu():
    """ With the streak samples, the stacks of the GPU (single and runs) agree with the CPU. The velocity of the
        largest sum may differ where several velocities give nearly the same sum (many fast velocities sample
        nearly the same pixels, and the CPU rounds differently), so a small fraction of the indices may differ.
    """

    rng = np.random.default_rng(12)
    frames = rng.normal(0, 1, (24, 48, 80)).astype(np.float32)
    vel = velocityGrid(6.0, 8, 2)
    cpu = VelocityStacker(vel, 8, use_gpu=False, streak=True)
    gpu = VelocityStacker(vel, 8, use_gpu=True, streak=True)
    m_cpu, i_cpu = cpu.stackMax(frames[:8])
    m_gpu, i_gpu = gpu.stackMax(frames[:8])
    assert np.allclose(m_cpu, m_gpu, atol=1e-4) and np.mean(i_cpu == i_gpu) > 0.99
    r_cpu, _ = cpu.stackMaxRuns(frames, [0, 8, 16])
    r_gpu, _ = gpu.stackMaxRuns(frames, [0, 8, 16])
    assert np.allclose(r_cpu, r_gpu, atol=1e-4)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason='No GPU')
def test_gpu_runs_agree_with_the_cpu():
    """ The stacks of several runs computed at once on the GPU (frames copied there once) are the stacks of every
        run on the CPU, also when the frames on the GPU are padded more than the stacker needs. """

    from RMS.Routines.MatchedFilterKernels import DeviceFrames, VelocityStacker

    # Random motions, noise frames whose width is not a multiple of the pixels of a block of GPU threads, and
    #   four runs of 8 frames
    rng = np.random.default_rng(2)
    angles = rng.uniform(0, 2*np.pi, 60)
    speeds = rng.uniform(0, 1.5, 60)
    velocities = np.c_[speeds*np.cos(angles), speeds*np.sin(angles)]
    frames = rng.normal(0, 1, (40, 70, 150)).astype(np.float32)
    starts = [0, 8, 16, 24]

    # The same motions stacked on the CPU and on the GPU
    cpu = VelocityStacker(velocities, 8, use_gpu=False)
    gpu = VelocityStacker(velocities, 8, use_gpu=True)

    # The frames given from the CPU, and already on the GPU with a larger padding
    for source in (frames, DeviceFrames(frames, gpu.pad + 3)):
        stack_max, vel_idx = gpu.stackMaxRuns(source, starts)
        assert stack_max.shape == (4, 70, 150)

        # Every run as on the CPU (the sums in another order differ in the last bits, which can change the best
        #   motion of a very few pixels where two motions give the same sum)
        for i, r0 in enumerate(starts):
            expected_max, expected_idx = cpu.stackMax(frames[r0:r0 + 8])
            assert np.allclose(stack_max[i], expected_max, atol=1e-5)
            assert np.mean(vel_idx[i] == expected_idx) > 0.999


@pytest.mark.parametrize('case', ['faint', 'slow'])
def test_kept_normalized_blocks_give_the_same_detections(config, monkeypatch, case):
    """ With the normalized blocks kept from the search (mf_cache_memory), the detections and every measurement
        are the same as with the blocks read and normalized again, fewer frames are read, and a block whose
        background is estimated again without a slow object is normalized again.
    """

    # A faint object (the blocks are kept to the end), or a slow one, whose background is estimated again without
    #   it after the search (its blocks are read and normalized again)
    if case == 'faint':
        objects, kwargs = [linearObject(1.5, 25, 30, 0.3, 0.12, 40, 470)], {}
    else:
        objects, kwargs = [linearObject(6.0, 30, 60, 0.03, 0.015, 0, 1536)], dict(n_stars=4, total_frames=1536)
        config.mf_smooth_frames = 0

    # The frames read by every run of the detector are counted
    reads = []
    read = MatchedFilterDetector.readFrames

    def countingRead(self, first, n, step=1, saturation=False):
        """ Read the frames as the detector does, counting them for the current run of the detector. """

        reads[-1] += n
        return read(self, first, n, step=step, saturation=saturation)

    # The detector reads its frames through the counting function
    monkeypatch.setattr(MatchedFilterDetector, 'readFrames', countingRead)

    # The detector without and with the kept blocks
    results = []
    for memory in (0.0, 10.0):
        config.mf_cache_memory = memory
        reads.append(0)
        det, detections = runDetector(_SynthHandle(objects, **kwargs), config)
        results.append(detections)

    # The same detection with the same measurements, from fewer frames read
    assert len(results[0]) == len(results[1]) == 1
    for a, b in zip(results[0], results[1]):
        assert np.array_equal(a[2], b[2])
    assert reads[1] < reads[0]

    # The kept blocks are released at the end (a segmented input keeps the detector of every segment)
    assert det.block_cache == {}


def test_cpu_search_streams_the_runs(config, monkeypatch):
    """ On the CPU, the runs are stacked and searched one after the other (as before the GPU batches), not all
        held in memory at once: the stacks of all runs at once are only computed on the GPU. """

    from RMS.Routines.MatchedFilterKernels import VelocityStacker

    # The stacks of all runs at once are not used on the CPU
    def allRuns(self, frames, run_starts):
        """ Fail the test if the stacks of all runs at once are asked for. """

        raise AssertionError('All runs stacked at once on the CPU')

    # Any use of the stacks of all runs at once fails
    monkeypatch.setattr(VelocityStacker, 'stackMaxRuns', allRuns)

    # The faint object is found as with the stacks of all runs
    obj = linearObject(1.5, 25, 30, 0.3, 0.12, 40, 470)
    _, detections = runDetector(_SynthHandle([obj]), config)
    assert len(detections) == 1


def test_faint_ends_of_a_track_are_cut():
    """ A track whose leading part is much fainter than the rest (a piece of track over empty sky joined to the
        object's track) loses that part; a track which is faint all along, or briefly faint in its middle, is kept
        as it is. """

    rng = np.random.default_rng(3)
    n = 60
    cent = np.zeros((n, 7))
    cent[:, 0] = np.arange(n)

    # 25 rows at SNR ~1 (empty sky), then the object at SNR ~25
    cent[:, 5] = np.r_[rng.normal(1.0, 0.8, 25), rng.normal(25.0, 4.0, n - 25)]
    kept = mfd.trimFaintEnds(cent)
    assert 22 <= kept[0, 0] <= 28 and kept[-1, 0] == n - 1

    # Faint all along (near the limit): kept
    cent[:, 5] = rng.normal(1.5, 0.8, n)
    assert len(mfd.trimFaintEnds(cent)) == n

    # A short dip in the middle (e.g. behind a star) is not an end
    cent[:, 5] = np.full(n, 20.0)
    cent[30:36, 5] = 0.5
    assert len(mfd.trimFaintEnds(cent)) == n

    # Flashing (e.g. tumbling): bright every third frame and faint between, all along or after a steady part. It
    #   is kept at least from its first to its last flash
    cent[:, 5] = np.where(np.arange(n)%3 == 0, 100.0, 1.0)
    kept = mfd.trimFaintEnds(cent)
    assert (kept[0, 0] == 0) and (kept[-1, 0] >= 57)
    cent[:30, 5] = 50.0
    kept = mfd.trimFaintEnds(cent)
    assert (kept[0, 0] == 0) and (kept[-1, 0] >= 57)

    # A faint head which is most of the track (a long piece over empty sky joined to the object)
    long = np.zeros((120, 7))
    long[:, 0] = np.arange(120)
    long[:, 5] = np.r_[rng.normal(1.0, 0.5, 100), rng.normal(25.0, 4.0, 20)]
    assert mfd.trimFaintEnds(long)[0, 0] >= 95

    # The bright part exactly a tenth of the track
    tenth = np.zeros((210, 7))
    tenth[:, 0] = np.arange(210)
    tenth[:, 5] = np.r_[np.full(189, 1.0), np.full(21, 25.0)]
    assert mfd.trimFaintEnds(tenth)[0, 0] >= 185

    # Flashes every fifth frame, at both ends too, the shortest track a detection can be
    short = np.zeros((21, 7))
    short[:, 0] = np.arange(21)
    short[:, 5] = np.where(np.arange(21)%5 == 0, 100.0, 1.0)
    assert len(mfd.trimFaintEnds(short)) == 21

    # An object which brightens by magnitudes keeps its fainter part (above the noise), and a saturated bright part
    #   (whose SNR is not a measure of its brightness) doesn't make the rest of the track faint
    bright = np.zeros((80, 7))
    bright[:, 0] = np.arange(80)
    bright[:, 5] = np.r_[np.full(40, 18.0), np.full(40, 1000.0)]
    assert len(mfd.trimFaintEnds(bright)) == 80
    bright[:, 5] = np.r_[np.full(30, 90.0), np.full(20, 1.1e5), np.full(30, 90.0)]
    bright[30:50, 6] = 5
    assert len(mfd.trimFaintEnds(bright)) == 80

    # A saturated object with a continuation over empty sky: the continuation is cut
    bright[:, 5] = np.r_[np.full(50, 1.1e5), np.full(30, 1.0)]
    bright[:, 6] = np.r_[np.full(50, 5), np.zeros(30)]
    kept = mfd.trimFaintEnds(bright)
    assert (kept[0, 0] == 0) and (kept[-1, 0] <= 52)

    # A single bright noise row at the end of a faint head doesn't keep the head
    cent[:, 5] = np.r_[rng.normal(1.0, 0.5, 25), rng.normal(25.0, 4.0, n - 25)]
    cent[0, 5] = 30.0
    assert mfd.trimFaintEnds(cent)[0, 0] >= 20


### Gaps in the recording ###

class _GapHandle(_SynthHandle):
    """ Synthetic frames with the time of every frame, with frames missing after the frame gap_frame: the time
        jumps there by gap_missing frame intervals more. The objects move with the time, so after the gap they
        are where the time puts them, not where the frame number would.
    """

    def __init__(self, objects, gap_frame, gap_missing, table=False, **kwargs):

        self.gap_frame = gap_frame
        self.gap_missing = gap_missing

        # The positions of the objects are functions of the time in frame intervals (the time axis)
        timed = []
        for obj in objects:
            timed.append(dict(obj, pos=self._timed(obj['pos'])))
        _SynthHandle.__init__(self, timed, **kwargs)

        # The time table of the frames (as of frame block files), if asked for
        if table:
            self.frame_unix_times = np.array([self.frameTime(f) for f in range(self.total_frames)]) + 1.8e9

    def frameTime(self, f):
        """ The time of a frame in frame intervals from the first one. """

        return f + (self.gap_missing if f > self.gap_frame else 0)

    def _timed(self, pos):
        """ The position function of the time, as a function of the frame (between frames by the nearest one's
            time, so the motion during a frame is the usual one). """

        def timedPos(f):
            base = int(math.floor(f + 0.5))
            return pos(self.frameTime(base) + (f - base))

        return timedPos

    def currentFrameTime(self, frame_no=None, dt_obj=False):
        """ The time of a frame, with the gap. """

        if frame_no is None:
            frame_no = self.current_frame

        return BEG_TIME + datetime.timedelta(seconds=self.frameTime(frame_no)/self.fps)

    def hasAccurateTime(self):
        return True


def test_time_gaps_found_from_the_frame_times():
    """ The gaps in the recording are where the time between two frames is much longer than usual, from the time
        of every frame or from the time table of the input; none without the time of every frame. """

    handle = _GapHandle([], 250, 30, total_frames=400)
    assert mfd.timeGaps(handle) == [251]
    assert mfd.timeGaps(handle, frame_range=(260, 400)) == []
    assert mfd.timeGaps(_GapHandle([], 250, 30, total_frames=400, table=True)) == [251]

    # Without the time of every frame the frames are at a constant interval
    assert mfd.timeGaps(_SynthHandle([], total_frames=400)) == []


def test_object_measured_at_its_times_across_a_gap(config):
    """ An object moving through a gap in the recording (30 frame intervals missing) is measured where it is on
        both sides of the gap: the track is not followed across the gap as if no time had passed (which put
        phantom points on empty sky after it), and the object is not found twice. """

    obj = linearObject(3.0, 15, 30, 0.25, 0.1, 40, 500)
    handle = _GapHandle([obj], 250, 30, total_frames=512)
    detections, detector = mfd.detectMatchedFilter(handle, config, return_detector=True)

    # The parts on both sides of the gap, without overlap
    assert len(detections) == 2
    cents = sorted((det[2] for det in detections), key=lambda c: c[0, 0])
    assert cents[0][-1, 0] <= 250 < 251 <= cents[1][0, 0]

    # Every measurement is near the true position at the time of its frame
    for cent in cents:
        err = truthError(cent, handle.objects[0])
        assert np.sqrt(np.mean(err**2)) < 0.6
        assert err.max() < 2.0

    # The summary has the gap
    summary = detector.summary()
    assert summary['time_gaps'] == [{'frame': 251, 'gap_s': round(31/config.fps, 3)}]
    assert len(summary['tracks']) >= 2


def test_input_without_gaps_unchanged(config):
    """ An input with the time of every frame but without gaps is searched as before, in one piece. """

    obj = linearObject(3.0, 15, 30, 0.25, 0.1, 40, 500)
    handle = _GapHandle([obj], 10000, 0, total_frames=512)
    detections, detector = mfd.detectMatchedFilter(handle, config, return_detector=True)
    _, plain = runDetector(_GapHandle([obj], 10000, 0, total_frames=512), config)

    assert isinstance(detector, MatchedFilterDetector)
    assert len(detections) == len(plain) == 1
    assert np.array_equal(detections[0][2], plain[0][2])


def test_segment_with_a_gap_carries_the_persistence(config):
    """ A segment of an input with a gap: its parts are searched separately, the persistence counts reach the
        next segment (counted up to its first frame, here at the end of the block of the first part which holds
        it), and the light between points is taken from the part which the frame belongs to. """

    obj = linearObject(3.0, 15, 30, 0.25, 0.1, 40, 500)
    handle = _GapHandle([obj], 250, 30, total_frames=512)
    detections, detector = mfd.detectMatchedFilter(handle, config, return_detector=True, frame_range=(0, 512),
                                                   persistence_until=200, track_deviation=False)

    assert isinstance(detector, mfd.GapPieces)
    assert [tuple(r) for r in detector.ranges] == [(0, 251), (251, 512)]
    assert detector.persistence_carry is not None
    assert (detector.persistence_carry_frames, detector.persistence_full_frames) == (251, 512)
    assert detector.connectedLight(100, 10, 10, 20, 20) is not None
    assert detector.connectedLight(300, 10, 10, 20, 20) is not None


class _StampedHandle(object):
    """ An input with a table of frame times only (see timeGaps). """

    def __init__(self, times):
        self.frame_unix_times = np.asarray(times, dtype=np.float64)
        self.total_frames = len(times)

    def hasAccurateTime(self):
        return True


def test_time_gaps_with_repeated_time_stamps():
    """ An input whose time stamps repeat for pairs of frames (steps of zero) still has its gaps found: the usual
        interval is the median of the positive steps. """

    # Pairs of frames share a stamp every 0.08 s (as many steps of zero as positive ones); 0.8 s are missing before
    #   frame 200
    times = np.repeat(np.arange(200)*0.08, 2)
    times[200:] += 0.8
    assert mfd.timeGaps(_StampedHandle(times)) == [200]

    # Without a gap nothing is found, and an input whose stamps never change has no usual interval
    assert mfd.timeGaps(_StampedHandle(np.repeat(np.arange(200)*0.08, 2))) == []
    assert mfd.timeGaps(_StampedHandle(np.zeros(100))) == []


def test_detections_cut_and_joined_only_within_the_parts_between_gaps(config):
    """ Detections stitched over a gap are cut at it, and pieces on the two sides of a gap are not joined again,
        however well one continues the other. """

    def det(frames):
        frames = np.asarray(frames, dtype=float)
        cent = np.c_[frames, 100 + 2*frames, 200 + frames, np.full(len(frames), 500.0), np.zeros((len(frames), 3))]
        return [0.0, 0.0, cent]

    # A detection across the gap at frame 100 is cut into its two pieces
    pieces = mfd.splitAtGaps([det(range(60, 140))], [100])
    assert [(p[2][0, 0], p[2][-1, 0]) for p in pieces] == [(60, 99), (100, 139)]
    assert mfd.splitAtGaps([det(range(60, 90))], [100])[0][2].shape[0] == 30

    # Two pieces of one straight track, one on each side of the gap, stay two detections; without the gap they
    #   are joined
    class _Joiner(object):
        def joinDetections(self, detections, connected=None, cut=None):
            # Joins everything it is given into one detection, as the joining of continuous pieces would
            if len(detections) < 2:
                return detections
            return [[0.0, 0.0, np.concatenate([d[2] for d in detections])]]

    two = [det(range(60, 99)), det(range(101, 140))]
    assert len(mfd.joinWithinGaps(_Joiner(), two, [100])) == 2
    assert len(mfd.joinWithinGaps(_Joiner(), two, [])) == 1
