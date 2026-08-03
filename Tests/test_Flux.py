

def test_fit_moon_gain_recovers_truth():
    # Synthetic night: detections generated at a known gain must be recovered
    import numpy as np
    from Utils.Flux import fitMoonGain, MOON_GAIN_MIN_FRAMES

    rng = np.random.RandomState(3)
    dome_s = 0.4
    g_true = 0.6

    samples = []
    for _ in range(MOON_GAIN_MIN_FRAMES + 30):
        n = 120
        logit = rng.uniform(-2.0, 3.0, n).astype(np.float32)
        q = rng.uniform(0.5, 6.0, n).astype(np.float32)
        pen = 1.25*np.log10(1.0 + g_true*q.astype(np.float64))
        p_true = 1.0/(1.0 + np.exp(-(logit.astype(np.float64) - pen/dome_s)))
        det = int(round(p_true.sum()))     # expectation, noise-free
        samples.append((logit, q, det))

    g_fit = fitMoonGain(samples, 1.0, dome_s)
    assert g_fit is not None
    assert abs(g_fit - g_true) < 0.1

    # A night whose moonlit envelope already matches dark fits gain 0
    samples0 = []
    for logit, q, _ in samples:
        p0 = 1.0/(1.0 + np.exp(-logit.astype(np.float64)))
        samples0.append((logit, q, int(round(p0.sum()))))
    assert fitMoonGain(samples0, 1.0, dome_s) == 0.0

    # Too few frames: no fit
    assert fitMoonGain(samples[:10], 1.0, dome_s) is None


def test_moon_gain_applied_warmup(tmp_path):
    # No history -> gain 0; enough fitted nights -> bounded median
    import json, os
    import numpy as np
    from Utils.Flux import moonGainApplied, recordMoonGain, MOON_GAIN_MAX

    class Cfg(object):
        data_dir = str(tmp_path)
        stationID = "US005X"

    assert moonGainApplied(Cfg()) == 0.0

    for i, gv in enumerate((0.5, 0.7, 5.0)):
        recordMoonGain(Cfg(), "US005X_2026080{:d}_010203_000000".format(i + 1),
            gv, "night")
    applied = moonGainApplied(Cfg())
    assert 0.0 < applied <= MOON_GAIN_MAX
    assert applied == 0.7    # median, and the 5.0 outlier neither wins nor breaks the bound
