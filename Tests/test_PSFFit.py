""" The compiled PSF model gives the same fits as the NumPy model used before. """

import numpy as np
import scipy.optimize as opt

import RMS.ExtractStars as es
from RMS.Math import twoDGaussian
from RMS.Routines.PSFFitCy import twoDGaussianResidual


def _segment(shape=(8, 8), params=(900.0, 3.7, 4.2, 1.3, 1.6, 0.4, 100.0), noise=5.0, seed=0):
    y_ind, x_ind = np.indices(shape, dtype=np.float64)
    model = twoDGaussian((y_ind, x_ind, np.full(shape, 65535.0)), *params).reshape(shape)
    return model + np.random.default_rng(seed).normal(0, noise, shape), y_ind.ravel(), x_ind.ravel()


def test_residual_matches_the_numpy_model():

    seg, y_ind, x_ind = _segment()
    params = np.array([850.0, 3.5, 4.0, 1.2, 1.8, 0.3, 90.0])

    for saturation in [65535.0, 500.0]:
        expected = twoDGaussian((y_ind.reshape(8, 8), x_ind.reshape(8, 8), np.full((8, 8), saturation)),
                                *params) - seg.ravel()
        residual = twoDGaussianResidual(params, y_ind, x_ind, saturation, seg.ravel())
        assert np.allclose(residual, expected, rtol=1e-12, atol=1e-9)


def test_fit_matches_curve_fit():

    for seed in range(20):
        seg, y_ind, x_ind = _segment(seed=seed)
        p0 = (float(seg.max() - np.median(seg)), 4.0, 4.0, 1.0, 1.0, 0.0, float(np.median(seg)))

        popt_ref, _ = opt.curve_fit(twoDGaussian, (y_ind.reshape(8, 8), x_ind.reshape(8, 8),
                                                   np.full((8, 8), 65535)), seg.ravel(), p0=p0, maxfev=200)
        popt, ier = opt.leastsq(twoDGaussianResidual, np.array(p0), args=(y_ind, x_ind, 65535.0, seg.ravel()),
                                maxfev=200)

        assert ier in (1, 2, 3, 4)
        assert np.allclose(popt, popt_ref, rtol=1e-7, atol=1e-7)


def test_fitPSF_finds_the_star():

    seg, _, _ = _segment(shape=(40, 40), params=(900.0, 20.3, 17.6, 1.3, 1.5, 0.4, 100.0))
    result = es.fitPSF(seg, 100.0, [17], [20], segment_radius=4, bit_depth=16)

    x_fitted, y_fitted = result[0], result[1]
    assert len(x_fitted) == 1
    assert abs(x_fitted[0] - 17.6) < 0.1 and abs(y_fitted[0] - 20.3) < 0.1
