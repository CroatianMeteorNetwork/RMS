"""
Tests of the sky quality from the stars of the chunks of frames (RMS.Routines.SkyQuality): clear sky is clear
everywhere, a region of dimmed stars or of missing stars is clouded in its chunks, a chunk with too few stars is
clouded everywhere, and the measurements of detections on clouded sky are removed.
"""

import os
import datetime

import numpy as np
import pytest

import RMS.ConfigReader as cr
import RMS.Routines.SkyQuality as sq
from RMS.Astrometry.ApplyAstrometry import raDecToXYPP
from RMS.Astrometry.Conversions import datetime2JD
from RMS.Formats.Platepar import Platepar
from RMS.Routines.SkyQuality import (CalibrationError, SkyQualityMap, SkyReference, chunkFirstFrames,
                                     filterClouded)


# Size of the image, frames per second and frames per chunk of the synthetic inputs
WIDTH, HEIGHT = 512, 512
FPS = 25.0
CHUNK_FRAMES = 128
N_CHUNKS = 12
BEGIN = datetime.datetime(2025, 12, 25, 10, 0, 0)


def _chunkName(i, begin=BEGIN):
    """ FF name of the chunk i (its first frame) of an input beginning at begin. """

    t = begin + datetime.timedelta(seconds=i*CHUNK_FRAMES/FPS)
    return 'FF_XX0001_{:s}_{:03d}_0000000.fits'.format(t.strftime('%Y%m%d_%H%M%S'), t.microsecond//1000)


def _skyMap(n_stars=500, cloud=None, opaque=None, few_stars_chunk=None, seed=0, skipped=(), total_frames=None,
            begin=BEGIN, reference=None, n_chunks=N_CHUNKS):
    """ Sky quality of synthetic chunks with uniformly distributed matched stars with residuals of 0.05 mag.

    Keyword arguments:
        n_stars: [int] Matched stars per chunk.
        cloud: [tuple] (x, y, radius, chunks, dimming): the stars in the circle are fainter by the dimming (mag) in
            the given chunks.
        opaque: [tuple] (x, y, radius, chunks): no stars in the circle in the given chunks.
        few_stars_chunk: [int] A chunk with only 20 stars.
        skipped: [tuple] Chunks left out of the star list (e.g. skipped by the star extraction).
        total_frames: [int] Number of frames of the input.
        begin: [datetime] Beginning of the input.
        reference: [SkyReference] Clear-sky reference of the camera.
        n_chunks: [int] Number of chunks.
    """

    rng = np.random.default_rng(seed)

    # The chunks need ff_min_stars stars to be usable
    config = cr.Config()
    config.ff_min_stars = 100

    # Only the size of the image is used, as the matches are given
    platepar = Platepar()
    platepar.X_res, platepar.Y_res = WIDTH, HEIGHT

    # The same stars in every chunk (a fixed field, the same in every input), of which 10% are randomly missed in a
    #   chunk
    field = np.random.default_rng(100)
    field_x, field_y = field.uniform(0, WIDTH, n_stars), field.uniform(0, HEIGHT, n_stars)

    star_list, matched = [], []
    for i in range(n_chunks):

        # The stars found in this chunk (only 20 in the chunk with few stars), with their index as the key
        found = rng.random(n_stars) > 0.1
        if i == few_stars_chunk:
            found[np.nonzero(found)[0][20:]] = False
        x, y, keys = field_x[found], field_y[found], np.nonzero(found)[0].astype(np.float64)
        n = len(x)

        # Their residuals: the scatter of a star from chunk to chunk on clear sky, plus the dimming of the cloud
        res = rng.normal(0, 0.05, n)
        if (cloud is not None) and (i in cloud[3]):
            res[np.hypot(x - cloud[0], y - cloud[1]) < cloud[2]] += cloud[4]

        # The stars behind an opaque cloud are not found
        keep = np.ones(n, dtype=bool)
        if (opaque is not None) and (i in opaque[3]):
            keep = np.hypot(x - opaque[0], y - opaque[1]) >= opaque[2]

        # The stars as from the star extraction, (y, x, intensity, amplitude, fwhm, background, snr, saturated)
        stars = [(yy, xx, 1000.0, 100.0, 2.0, 500.0, 20.0, 0) for xx, yy in zip(x[keep], y[keep])]

        # The skipped chunks are not in the star list
        if i in skipped:
            continue
        star_list.append([_chunkName(i, begin), stars])
        matched.append(np.c_[x[keep], y[keep], res[keep], keys[keep]])

    sky_map = SkyQualityMap(star_list, platepar, config, begin, FPS, CHUNK_FRAMES, matched=matched,
                            total_frames=total_frames, reference=reference)

    # For building other maps of the same stars
    sky_map.platepar_for_test, sky_map.matched_for_test = platepar, matched

    return sky_map


def test_clear_sky_is_clear():
    """ Without clouds, the sky is clear everywhere and in every chunk. """

    sky_map = _skyMap()

    assert sky_map.clear.all()
    assert not sky_map.cloudMask(0, N_CHUNKS*CHUNK_FRAMES - 1, HEIGHT//2, WIDTH//2, bin_factor=2).any()


def test_dimmed_stars_are_clouded():
    """ Stars dimmed by 0.5 mag in a region are a cloud: clouded in its chunks and the neighbouring ones (the cloud
        moves during a chunk), clear elsewhere and at other times.
    """

    sky_map = _skyMap(cloud=(150, 300, 80, (5, 6), 0.5))

    # In the cloud, during its chunks
    frame_in = 5*CHUNK_FRAMES + 10
    assert not sky_map.isClear(np.array([150.0]), np.array([300.0]), np.array([frame_in]))[0]

    # The neighbouring chunks are not clear, the farther ones are
    assert not sky_map.isClear(np.array([150.0]), np.array([300.0]), np.array([4*CHUNK_FRAMES + 10]))[0]
    assert sky_map.isClear(np.array([150.0]), np.array([300.0]), np.array([1*CHUNK_FRAMES]))[0]
    assert sky_map.isClear(np.array([150.0]), np.array([300.0]), np.array([10*CHUNK_FRAMES]))[0]

    # Far from the cloud, the sky is clear
    assert sky_map.isClear(np.array([450.0]), np.array([50.0]), np.array([frame_in]))[0]

    # The search mask of a block covering the cloud, 2x2 binned
    mask = sky_map.cloudMask(5*CHUNK_FRAMES, 6*CHUNK_FRAMES, HEIGHT//2, WIDTH//2, bin_factor=2)
    assert mask[300//2, 150//2] and not mask[50//2, 450//2]


def test_small_dimming_is_clear():
    """ A dimming below mf_cloud_max_offset (0.15 mag) is not a cloud. """

    sky_map = _skyMap(cloud=(150, 300, 80, (5, 6), 0.05))
    assert sky_map.clear.all()


def test_uniform_haze_is_clear():
    """ A haze dimming the whole image uniformly is calibrated by the photometric offset of the chunk, it is not a
        cloud.
    """

    sky_map = _skyMap(cloud=(WIDTH/2, HEIGHT/2, 2*WIDTH, (5, 6), 0.5))
    assert sky_map.clear.all()


def test_missing_stars_are_clouded():
    """ A region without stars (an opaque cloud) is clouded. """

    sky_map = _skyMap(opaque=(300, 200, 100, (7,)))
    assert not sky_map.isClear(np.array([300.0]), np.array([200.0]), np.array([7*CHUNK_FRAMES + 50]))[0]
    assert sky_map.isClear(np.array([60.0]), np.array([460.0]), np.array([7*CHUNK_FRAMES + 50]))[0]


def test_chunk_with_few_stars_is_clouded():
    """ A chunk with fewer stars than ff_min_stars is clouded everywhere. """

    sky_map = _skyMap(few_stars_chunk=3)
    assert not sky_map.clear[3].any()
    assert sky_map.clear[[0, 1, 5, 6]].all()


def test_filter_clouded_measurements():
    """ The measurements of a detection on clouded sky are removed, and a detection left with too few
        measurements is removed.
    """

    sky_map = _skyMap(cloud=(150, 300, 80, (5, 6), 0.5))

    # A detection crossing the cloud: 100 measurements from x = 0 to 400 at y = 300 during chunks 5 and 6, and one
    #   far from it
    frames = np.linspace(5*CHUNK_FRAMES, 7*CHUNK_FRAMES - 1, 100)
    crossing = np.c_[frames, np.linspace(0, 400, 100), np.full(100, 300.0), np.ones(100)]
    clear = np.c_[frames, np.linspace(380, 480, 100), np.full(100, 40.0), np.ones(100)]

    kept, n_rows, n_dets = filterClouded([[0.0, 0.0, crossing], [0.0, 0.0, clear]], sky_map, min_rows=6)
    assert n_dets == 0
    assert len(kept[1][2]) == 100
    assert 0 < n_rows < 100
    assert np.all(np.abs(kept[0][2][:, 1] - 150) > 60)

    # With a higher minimum number of measurements, the crossing detection is removed
    kept, n_rows, n_dets = filterClouded([[0.0, 0.0, crossing], [0.0, 0.0, clear]], sky_map, min_rows=99)
    assert n_dets == 1 and len(kept) == 1


@pytest.mark.parametrize('frame', [-50, 10**6])
def test_frames_outside_the_chunks(frame):
    """ Frames before the first and after the last chunk use the nearest chunk. """

    sky_map = _skyMap()
    assert sky_map.isClear(np.array([200.0]), np.array([200.0]), np.array([frame]))[0]


def test_missing_chunks_are_clouded():
    """ Chunks missing from the star list (at the beginning, in the middle and at the end of the input) are
        clouded, and the frames of the chunks present are clear.
    """

    # The first, a middle and the last chunk missing: they are added back as clouded chunks
    sky_map = _skyMap(skipped=(0, 5, 11), total_frames=N_CHUNKS*CHUNK_FRAMES)
    assert len(sky_map.chunk_first) == N_CHUNKS

    x, y = np.array([200.0]), np.array([200.0])
    for chunk in (0, 5, 11):
        assert not sky_map.isClear(x, y, np.array([chunk*CHUNK_FRAMES + 60]))[0]
    assert sky_map.isClear(x, y, np.array([8*CHUNK_FRAMES + 60]))[0]

    # Every other chunk missing: with the chunk size given, the missing ones are clouded
    sky_map = _skyMap(skipped=(1, 3, 5, 7, 9, 11), total_frames=N_CHUNKS*CHUNK_FRAMES)
    assert len(sky_map.chunk_first) == N_CHUNKS
    assert not sky_map.isClear(x, y, np.array([3*CHUNK_FRAMES + 60]))[0]

    # A few frames after the last full chunk (which the star extraction doesn't use) are in a missing chunk
    sky_map = _skyMap(total_frames=N_CHUNKS*CHUNK_FRAMES + 20)
    assert len(sky_map.chunk_first) == N_CHUNKS + 1
    assert not sky_map.isClear(x, y, np.array([N_CHUNKS*CHUNK_FRAMES + 10]))[0]

    # Without the number of frames, the frames after the last chunk belong to it
    sky_map = _skyMap(skipped=(11,))
    assert len(sky_map.chunk_first) == N_CHUNKS - 1


def test_no_catalog_is_unknown(tmp_path):
    """ Without the star catalog the sky quality is not known (CalibrationError), rather than clouded. """

    # A catalog which doesn't exist
    config = cr.Config()
    config.star_catalog_path = str(tmp_path)
    config.star_catalog_file = 'missing_catalog.bin'

    platepar = Platepar()
    platepar.X_res, platepar.Y_res = WIDTH, HEIGHT

    # Any stars: the catalog is read before they are matched
    stars = [(100.0, 100.0, 1000.0, 100.0, 2.0, 500.0, 20.0, 0)]*200
    with pytest.raises(CalibrationError):
        SkyQualityMap([[_chunkName(i), stars] for i in range(3)], platepar, config, BEGIN, FPS, CHUNK_FRAMES)


class _GapInput(object):
    """ An input with timestamped frames and a gap of 10 s in the recording before the chunk 6. """

    def __init__(self):
        self.chunk_frames = CHUNK_FRAMES
        self.total_frames = N_CHUNKS*CHUNK_FRAMES
        self.fps = FPS
        self.beginning_datetime = BEGIN

    def currentFrameTime(self, frame_no=None, dt_obj=False):

        # The frames from the chunk 6 on are 10 s later than the frame rate would put them
        gap = 10.0 if frame_no >= 6*CHUNK_FRAMES else 0.0
        return BEGIN + datetime.timedelta(seconds=frame_no/FPS + gap)


def test_first_frames_after_a_recording_gap():
    """ The chunks after a gap in the recording are at their frames, not where their times would put them, and
        they are clear.
    """

    # The chunks named by their times (10 s later from the chunk 6 on), and a chunk at a time of no chunk of the
    #   input
    img_handle = _GapInput()
    names = [_chunkName(i) for i in range(6)] + [_chunkName(i + 10.0*FPS/CHUNK_FRAMES) for i in range(6, N_CHUNKS)]
    star_list = [[name, []] for name in names] + [['FF_XX0001_20251225_120000_000_0000000.fits', []]]

    # The chunks are at their sequential frames, also after the gap
    first_frames = chunkFirstFrames(star_list, img_handle)
    assert first_frames[:N_CHUNKS] == [i*CHUNK_FRAMES for i in range(N_CHUNKS)]

    # A chunk at a time which is not the time of a chunk of the input is not known
    assert first_frames[-1] == -1

    # The sky quality with these first frames: no missing chunks after the gap, and clear sky there
    clear_map = _skyMap()
    stars = [[name, [(0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0)]*500] for name in names]
    sky_map = SkyQualityMap(stars, clear_map.platepar_for_test, clear_map.config, BEGIN, FPS, CHUNK_FRAMES,
                            matched=clear_map.matched_for_test, total_frames=N_CHUNKS*CHUNK_FRAMES,
                            first_frames=first_frames[:N_CHUNKS])
    assert len(sky_map.chunk_first) == N_CHUNKS
    assert sky_map.isClear(np.array([200.0]), np.array([200.0]), np.array([8*CHUNK_FRAMES + 10]))[0]


def _patchCalibration(monkeypatch, catalog, offset):
    """ Replace the catalog, the recalibration and the projection: the catalog stars are projected to their
        (ra, dec) as (x, y), shifted by offset px (a platepar off by that much).
    """

    # The recalibration always succeeds, with a platepar of the size of the image
    def recalibrate(platepar, ff_names, calstars, catalog_stars, config, **kwargs):
        pp = Platepar()
        pp.X_res, pp.Y_res = WIDTH, HEIGHT
        pp.auto_recalibrated = True
        return {ff_names[0]: pp}

    # The given catalog, the projection to (ra + offset, dec), all catalog stars near the field, and no extinction
    monkeypatch.setattr(sq.StarCatalog, 'readStarCatalog', lambda *args, **kwargs: (catalog, None, None))
    monkeypatch.setattr(sq, 'recalibratePlateparsForFF', recalibrate)
    monkeypatch.setattr(sq, 'raDecToXYPP', lambda ra, dec, jd, pp: (ra + offset, dec))
    monkeypatch.setattr(sq, 'catalogNearField', lambda catalog, pp, jds: np.ones(len(catalog), dtype=bool))
    monkeypatch.setattr(sq, 'extinctionCorrectionTrueToApparent', lambda mags, ra, dec, jd, pp: mags)


@pytest.mark.parametrize('offset, trusted', [(0.0, True), (5.0, False)])
def test_calibration_matching_few_stars_is_not_trusted(monkeypatch, offset, trusted):
    """ A calibration which matches dozens of the stars of the clearest chunk is used; one which matches only a
        few (e.g. a platepar off by a few pixels, accepted by the recalibration) leaves the sky quality unknown
        instead of making every chunk clouded.
    """

    # 300 stars at their catalog positions
    rng = np.random.default_rng(1)
    x, y = rng.uniform(20, WIDTH - 20, 300), rng.uniform(20, HEIGHT - 20, 300)
    catalog = np.c_[x, y, np.full(300, 8.0)]

    # 10 stars which the offset platepar still matches (catalog entries 5 px to their left, projected onto them)
    extra = np.c_[x[:10] - offset, y[:10]]
    catalog = np.vstack([catalog, np.c_[extra, np.full(10, 8.0)]]) if offset else catalog

    _patchCalibration(monkeypatch, catalog, offset)

    config = cr.Config()
    config.ff_min_stars = 100
    platepar = Platepar()
    platepar.X_res, platepar.Y_res = WIDTH, HEIGHT

    # The same stars in 6 chunks
    stars = [(yy, xx, 1000.0, 100.0, 2.0, 500.0, 20.0, 0) for xx, yy in zip(x, y)]
    star_list = [[_chunkName(i), stars] for i in range(6)]

    # Without the offset nearly all stars match; with it, only the 10 extra ones
    if trusted:
        sky_map = SkyQualityMap(star_list, platepar, config, BEGIN, FPS, CHUNK_FRAMES)
        assert np.all(sky_map.n_matched > 250)
    else:
        with pytest.raises(CalibrationError, match='only 1[0-9] of the 300'):
            SkyQualityMap(star_list, platepar, config, BEGIN, FPS, CHUNK_FRAMES)


def _residuals(keys, value):
    """ The same visit value for all stars. """

    return np.full(len(keys), value), np.full(len(keys), 1.0)


def test_reference_order_and_reprocessing(tmp_path):
    """ The visits of the inputs are kept by the time of the observation: in any order of the processing, an input
        processed again replaces its visit, and only the latest MAX_VISITS visits are kept.
    """

    # Three stars, and the inputs of every 10 minutes (more than the visits which are kept)
    keys = np.array([11, 22, 33])
    times = [BEGIN + datetime.timedelta(minutes=10*i) for i in range(SkyReference.MAX_VISITS + 2)]

    reference = SkyReference(str(tmp_path), 'XX0001')

    # Out of order: the second input first
    reference.update(times[1], keys, *_residuals(keys, 0.2), 0.0, [])
    reference.update(times[0], keys, *_residuals(keys, 0.1), 0.0, [])
    values, _ = SkyReference(str(tmp_path), 'XX0001').lookup(keys)
    assert sorted(values[0][np.isfinite(values[0])]) == pytest.approx([0.1, 0.2])

    # Processed again: replaced, not added
    reference.update(times[1], keys, *_residuals(keys, 0.3), 0.0, [])
    values, _ = reference.lookup(keys)
    assert sorted(values[0][np.isfinite(values[0])]) == pytest.approx([0.1, 0.3])

    # Processed again with fewer stars and an unknown zero point: the stars and the zero point which are not in
    #   the new processing are removed too
    reference.update(times[1], keys[:1], *_residuals(keys[:1], 0.4), np.nan, [])
    values, _ = reference.lookup(keys)
    assert sorted(values[0][np.isfinite(values[0])]) == pytest.approx([0.1, 0.4])
    assert values[1][np.isfinite(values[1])] == pytest.approx([0.1])
    assert len(reference.zeroPoints()) == 1
    reference.update(times[1], keys, *_residuals(keys, 0.3), 0.0, [])

    # The earlier processing of an input can be left out
    values, _ = reference.lookup(keys, exclude_time=(times[1] - datetime.datetime(1970, 1, 1)).total_seconds())
    assert values[0][np.isfinite(values[0])] == pytest.approx([0.1])

    # Only the latest visits are kept
    for i, t in enumerate(times):
        reference.update(t, keys, *_residuals(keys, float(i)), 0.0, [])
    values, _ = reference.lookup(keys)
    assert sorted(values[0][np.isfinite(values[0])]) == pytest.approx(list(range(2, SkyReference.MAX_VISITS + 2)))


def test_conditions_log(tmp_path):
    """ The conditions of every chunk are appended to the log of the date of the chunk, with the time of the
        processing.
    """

    reference = SkyReference(str(tmp_path), 'XX0001')
    sky_map = _skyMap(cloud=(150, 300, 80, (5, 6), 0.5), reference=reference)
    sky_map.updateReference()

    # The log of the date of the input: the description, the header and a row per chunk
    path = tmp_path/'sky_conditions_XX0001_20251225.csv'
    lines = [line for line in path.read_text().splitlines() if not line.startswith('#')]
    header = lines[0].split(',')
    rows = [dict(zip(header, line.split(','))) for line in lines[1:]]
    assert len(rows) == N_CHUNKS

    # The time of the first chunk, the cloud in the chunk 5, and the transparency of clear sky
    assert rows[0]['chunk_utc'] == '2025-12-25T10:00:00.000'
    assert float(rows[5]['clear_fraction']) < 1.0 and float(rows[0]['clear_fraction']) == 1.0
    assert abs(float(rows[0]['transparency_mag'])) < 0.05


def test_stationary_cloud_found_with_reference(tmp_path):
    """ A cloud over the same region during a whole input: with the stars of that input alone it is the
        reference (not found), with the clear-sky reference of an earlier input it is found.
    """

    cloud = (150, 300, 80, tuple(range(N_CHUNKS)), 0.5)
    later = BEGIN + datetime.timedelta(minutes=10)

    # Alone
    sky_map = _skyMap(cloud=cloud, begin=later)
    assert sky_map.clear.all()

    # With the reference of an earlier, clear input
    reference = SkyReference(str(tmp_path), 'XX0001')
    _skyMap(reference=reference).updateReference()
    sky_map = _skyMap(cloud=cloud, begin=later, reference=SkyReference(str(tmp_path), 'XX0001'))
    assert not sky_map.isClear(np.array([150.0]), np.array([300.0]), np.array([5*CHUNK_FRAMES]))[0]
    assert sky_map.isClear(np.array([450.0]), np.array([60.0]), np.array([5*CHUNK_FRAMES]))[0]

    # A dimming of 0.2 mag (just above the limit) with a single earlier input: the input itself doesn't enter the
    #   references of the stars which have an earlier one
    cloud = (150, 300, 80, tuple(range(N_CHUNKS)), 0.2)
    sky_map = _skyMap(cloud=cloud, begin=later, reference=SkyReference(str(tmp_path), 'XX0001'))
    assert not sky_map.isClear(np.array([150.0]), np.array([300.0]), np.array([5*CHUNK_FRAMES]))[0]


def test_missing_reliable_stars_are_clouded(monkeypatch, tmp_path):
    """ Where the stars which are reliably seen on clear sky are missing (an opaque cloud), the sky is clouded. """

    # 400 stars at their catalog positions, matched through the patched calibration
    rng = np.random.default_rng(3)
    x, y = rng.uniform(20, WIDTH - 20, 400), rng.uniform(20, HEIGHT - 20, 400)
    catalog = np.c_[x, y, np.full(400, 8.0)]
    _patchCalibration(monkeypatch, catalog, 0.0)

    config = cr.Config()
    config.ff_min_stars = 100
    platepar = Platepar()
    platepar.X_res, platepar.Y_res = WIDTH, HEIGHT

    # The stars of every chunk, without the ones within 90 px of (300, 200) in the hidden chunk (or in all chunks)
    def starList(begin, hidden_chunk=None, hidden_always=False):
        out = []
        for i in range(N_CHUNKS):
            keep = np.ones(len(x), dtype=bool)
            if (i == hidden_chunk) or hidden_always:
                keep = np.hypot(x - 300, y - 200) > 90
            out.append([_chunkName(i, begin), [(yy, xx, 1000.0, 100.0, 2.0, 500.0, 20.0, 0)
                                               for xx, yy in zip(x[keep], y[keep])]])
        return out

    # A clear input makes the stars reliable, then an input with an opaque cloud in the chunk 7
    reference = SkyReference(str(tmp_path), 'XX0001')
    SkyQualityMap(starList(BEGIN), platepar, config, BEGIN, FPS, CHUNK_FRAMES, reference=reference).updateReference()

    later = BEGIN + datetime.timedelta(minutes=10)
    sky_map = SkyQualityMap(starList(later, hidden_chunk=7), platepar, config, later, FPS, CHUNK_FRAMES,
                            reference=SkyReference(str(tmp_path), 'XX0001'))
    assert sky_map.reliable_seen[0] == pytest.approx(1.0, abs=0.05)
    assert sky_map.reliable_seen[7] < 0.9
    assert not sky_map.isClear(np.array([300.0]), np.array([200.0]), np.array([7*CHUNK_FRAMES + 10]))[0]
    assert sky_map.isClear(np.array([60.0]), np.array([460.0]), np.array([7*CHUNK_FRAMES + 10]))[0]

    # An opaque cloud during the whole input: the stars behind it are not matched in it at all, but they are
    #   expected from the reference
    sky_map = SkyQualityMap(starList(later, hidden_always=True), platepar, config, later, FPS, CHUNK_FRAMES,
                            reference=SkyReference(str(tmp_path), 'XX0001'))
    assert not sky_map.isClear(np.array([300.0]), np.array([200.0]), np.array([7*CHUNK_FRAMES + 10]))[0]
    assert sky_map.isClear(np.array([60.0]), np.array([460.0]), np.array([7*CHUNK_FRAMES + 10]))[0]


def test_saturated_reliable_stars_are_seen(monkeypatch, tmp_path):
    """ Reliable stars which are extracted but not matched (saturated pixels, left out of the photometry) are
        seen: the region around them is not clouded.
    """

    # 400 stars at their catalog positions, matched through the patched calibration
    rng = np.random.default_rng(4)
    x, y = rng.uniform(20, WIDTH - 20, 400), rng.uniform(20, HEIGHT - 20, 400)
    catalog = np.c_[x, y, np.full(400, 8.0)]
    _patchCalibration(monkeypatch, catalog, 0.0)

    config = cr.Config()
    config.ff_min_stars = 100
    platepar = Platepar()
    platepar.X_res, platepar.Y_res = WIDTH, HEIGHT

    # The stars of every chunk, the ones within 90 px of (300, 200) with saturated pixels in the saturated chunk
    def starList(begin, saturated_chunk=None):
        out = []
        for i in range(N_CHUNKS):
            saturated = (np.hypot(x - 300, y - 200) < 90) & (i == saturated_chunk)
            out.append([_chunkName(i, begin), [(yy, xx, 1000.0, 100.0, 2.0, 500.0, 20.0, int(sat))
                                               for xx, yy, sat in zip(x, y, saturated)]])
        return out

    # A clear input makes the stars reliable, then an input with the saturated stars in the chunk 7
    reference = SkyReference(str(tmp_path), 'XX0001')
    SkyQualityMap(starList(BEGIN), platepar, config, BEGIN, FPS, CHUNK_FRAMES, reference=reference).updateReference()

    later = BEGIN + datetime.timedelta(minutes=10)
    sky_map = SkyQualityMap(starList(later, saturated_chunk=7), platepar, config, later, FPS, CHUNK_FRAMES,
                            reference=SkyReference(str(tmp_path), 'XX0001'))
    assert sky_map.reliable_seen[7] == pytest.approx(1.0, abs=0.05)
    assert sky_map.isClear(np.array([300.0]), np.array([200.0]), np.array([7*CHUNK_FRAMES + 10]))[0]


def test_short_input(tmp_path):
    """ An input with fewer chunks than needed for the reference of its stars: without a reference of the camera the
        sky quality is not known (not clouded everywhere), with one it is.
    """

    # Three chunks: fewer than MIN_STAR_OBS observations of every star
    with pytest.raises(CalibrationError):
        _skyMap(n_chunks=3)

    # With the reference of an earlier input, a cloud in the short input is found
    reference = SkyReference(str(tmp_path), 'XX0001')
    _skyMap(reference=reference).updateReference()
    later = BEGIN + datetime.timedelta(minutes=10)
    sky_map = _skyMap(n_chunks=3, begin=later, cloud=(150, 300, 80, (0, 1, 2), 0.5),
                      reference=SkyReference(str(tmp_path), 'XX0001'))
    assert not sky_map.isClear(np.array([150.0]), np.array([300.0]), np.array([CHUNK_FRAMES]))[0]
    assert sky_map.isClear(np.array([450.0]), np.array([60.0]), np.array([CHUNK_FRAMES]))[0]


def test_stars_entering_during_the_input(monkeypatch):
    """ With the sky moving across the image during the input (twice the image width), the stars which are in the
        image only at its beginning or its end are matched too.
    """

    # Stars over three image widths in x
    rng = np.random.default_rng(4)
    n = 1500
    ra, dec = rng.uniform(-WIDTH, 2*WIDTH, n), rng.uniform(20, HEIGHT - 20, n)
    catalog = np.c_[ra, dec, np.full(n, 8.0)]
    _patchCalibration(monkeypatch, catalog, 0.0)

    # The sky moves by 2*WIDTH px over the input, so a star at ra is at x = ra - shift(t)
    duration = N_CHUNKS*CHUNK_FRAMES/FPS
    jd0 = datetime2JD(BEGIN)

    def project(r, d, jd, pp):
        return r - 2*WIDTH*(jd - jd0)*86400/duration + WIDTH/2, d

    monkeypatch.setattr(sq, 'raDecToXYPP', project)

    config = cr.Config()
    config.ff_min_stars = 50
    platepar = Platepar()
    platepar.X_res, platepar.Y_res = WIDTH, HEIGHT

    # The stars in the image in every chunk, at their positions at the middle of the chunk
    star_list = []
    for i in range(N_CHUNKS):
        t = BEGIN + datetime.timedelta(seconds=i*CHUNK_FRAMES/FPS)
        x, y = project(ra, dec, datetime2JD(t) + CHUNK_FRAMES/(2*FPS)/86400, None)
        inside = (x > 15) & (x < WIDTH - 15)
        star_list.append([_chunkName(i), [(yy, xx, 1000.0, 100.0, 2.0, 500.0, 20.0, 0)
                                          for xx, yy in zip(x[inside], y[inside])]])

    # The first and the last chunk see stars which are far from the field in the middle of the input
    sky_map = SkyQualityMap(star_list, platepar, config, BEGIN, FPS, CHUNK_FRAMES)
    assert sky_map.n_matched[0] > 0.8*len(star_list[0][1])
    assert sky_map.n_matched[-1] > 0.8*len(star_list[-1][1])


def test_catalog_near_field():
    """ Every catalog star which is in the image at some time of an input is selected near the field, and most of
        the sky is not.
    """

    # A template platepar of the repository
    platepar = Platepar()
    platepar.read(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'share',
                               'platepar_templates', 'template_generic_720p_6mm.cal'))

    # Stars over the whole sky
    rng = np.random.default_rng(5)
    n = 20000
    catalog = np.c_[rng.uniform(0, 360, n), np.degrees(np.arcsin(rng.uniform(-1, 1, n))), np.full(n, 5.0)]

    # An input of two hours (the sky moves by 30 deg), the epochs of the selection over it
    jd0 = datetime2JD(BEGIN)
    jds = list(jd0 + np.linspace(0, 2/24, 21))
    near = sq.catalogNearField(catalog, platepar, jds)

    # The stars in the image at any of many times of the input
    inside = np.zeros(n, dtype=bool)
    for jd in jd0 + np.linspace(0, 2/24, 61):
        x, y = raDecToXYPP(catalog[:, 0], catalog[:, 1], jd, platepar)
        inside |= (x >= 0) & (x < platepar.X_res) & (y >= 0) & (y < platepar.Y_res)

    # All stars in the image are near the field, and most of the sky is not
    assert inside.sum() > 100
    assert np.all(near[inside])
    assert near.mean() < 0.5
