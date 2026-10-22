"""
Tests for the flags of the matched filter (RMS.MatchedFilterFlags) and its rejection of wandering tracks
(MatchedFilterDetection.removeWanderingDetections), on synthetic tracks:

    - Earth shadow: the first sunlit point along a line of sight, and the flag of a source which can't be sunlit.
    - Aircraft: bright detections at the same time, and detections moving alongside each other.
    - Wandering: long tracks of barely moving light which stray from a smooth path are removed, smooth ones and fast
      ones are kept.
    - Files: the flagged detections are saved under a name which no tool looking for FTPdetectinfo files picks up,
      and the files of a night are merged.
"""

import datetime
import fnmatch
import types

import numpy as np

from RMS.Astrometry.Conversions import datetime2JD, JD2LST
from RMS.Formats import FTPdetectinfo
from RMS.Formats.Platepar import Platepar
import RMS.MatchedFilterFlags as F
from RMS.MatchedFilterDetection import removeWanderingDetections, trackWander


# Observer at longitude 0, so 0 h UTC is local midnight; the Sun is near RA 206 deg, Dec -11 deg on this date
LAT, LON, ELEV = 45.0, 0.0, 300.0
MIDNIGHT = datetime.datetime(2026, 10, 22, 0, 0, 0)
NOON = datetime.datetime(2026, 10, 22, 12, 0, 0)


def makeConfig(**changes):
    """ The options of the flags (the defaults of RMS.ConfigReader) for a 512x512 camera with 0.0175 deg/px. """

    config = types.SimpleNamespace(fov_h=512*0.0175, fov_w=512*0.0175, width=512, height=512, fps=32.0,
                                   stationID='XX0001', mf_flag_sunlit_speed=8.0, mf_flag_sunlit_factor=2.0,
                                   mf_flag_aircraft_mag=4.0, mf_flag_aircraft_gap=1.0, mf_flag_parallel_angle=3.0,
                                   mf_flag_parallel_speed=0.15, mf_flag_parallel_offset=0.7)
    for name, value in changes.items():
        setattr(config, name, value)

    return config


def antiSun(dt):
    """ RA and Dec (deg) opposite to the Sun. """

    sun = F.sunDirection(datetime2JD(dt))
    ra = np.degrees(np.arctan2(-sun[1], -sun[0])) % 360
    dec = np.degrees(np.arcsin(-sun[2]))

    return ra, dec


def makeTrack(dt, ra, dec, ang_vel, duration=1.0, mag=3.0, x0=100.0, y0=100.0, direction=(1.0, 0.0),
              offset=(0.0, 0.0), n=32):
    """ A straight track starting at (ra, dec) and moving along the declination at ang_vel (deg/s), and the same
        motion in the image at 0.0175 deg/px, starting at (x0, y0) + offset. """

    t = np.linspace(0, duration, n)
    jd = datetime2JD(dt) + t/86400.0
    px = ang_vel*t/0.0175
    d = np.asarray(direction)/np.hypot(*direction)

    return F.Track(jd, x0 + offset[0] + px*d[0], y0 + offset[1] + px*d[1], np.full(n, ra), dec + ang_vel*t,
                   np.full(n, mag))



### Earth shadow ###

def test_sunlit_range_day_and_night():
    """ In daylight the line of sight is sunlit from the observer on; opposite to the Sun at midnight it is in the
        shadow all the way. """

    jd_noon = datetime2JD(NOON)
    lst = JD2LST(jd_noon, LON)[0]
    assert F.firstSunlitRange(jd_noon, lst, LAT, LAT, LON, ELEV) == 0.0

    ra, dec = antiSun(MIDNIGHT)
    assert F.firstSunlitRange(datetime2JD(MIDNIGHT), ra, dec, LAT, LON, ELEV) == np.inf


def test_sunlit_range_twilight_zenith():
    """ Straight up at midnight the line of sight leaves the shadow at some thousand km. """

    jd = datetime2JD(MIDNIGHT)
    r = F.firstSunlitRange(jd, JD2LST(jd, LON)[0], LAT, LAT, LON, ELEV)

    assert 1000 < r < 20000


def test_not_sunlit_flag():
    """ A fast source opposite to the Sun at midnight can't be sunlit and is flagged; in daylight it is not. """

    config = makeConfig(mf_flag_aircraft_mag=0)
    ra, dec = antiSun(MIDNIGHT)

    flags, values = F.flagTracks([makeTrack(MIDNIGHT, ra, dec, 1.0)], LAT, LON, ELEV, config)
    assert flags[0] == ['not_sunlit']

    flags, values = F.flagTracks([makeTrack(NOON, ra, dec, 1.0)], LAT, LON, ELEV, config)
    assert flags[0] == []
    assert values[0]['sunlit_ratio'] == 0.0


def test_slow_source_can_be_sunlit():
    """ A very slow source can be far away, where the line of sight up from midnight is sunlit: not flagged. """

    jd = datetime2JD(MIDNIGHT)
    track = makeTrack(MIDNIGHT, JD2LST(jd, LON)[0], LAT, 0.01, duration=20.0)

    flags, values = F.flagTracks([track], LAT, LON, ELEV, makeConfig(mf_flag_aircraft_mag=0))

    assert flags[0] == []
    assert values[0]['sunlit_ratio'] < 1.0



### Aircraft ###

def test_aircraft_bright_pair():
    """ Two bright detections at the same time where nothing can be sunlit are aircraft; a single one is not, and
        neither are two faint ones. """

    ra, dec = antiSun(MIDNIGHT)
    config = makeConfig(mf_flag_sunlit_factor=1e9)

    a = makeTrack(MIDNIGHT, ra, dec, 1.0, mag=2.5)
    b = makeTrack(MIDNIGHT, ra + 2, dec, 1.0, mag=3.0, y0=300)
    flags, _ = F.flagTracks([a, b], LAT, LON, ELEV, config)
    assert flags == [['aircraft'], ['aircraft']]

    flags, _ = F.flagTracks([a], LAT, LON, ELEV, config)
    assert flags == [[]]

    a_faint = makeTrack(MIDNIGHT, ra, dec, 1.0, mag=8.0)
    b_faint = makeTrack(MIDNIGHT, ra + 2, dec, 1.0, mag=8.0, y0=300)
    flags, _ = F.flagTracks([a_faint, b_faint], LAT, LON, ELEV, config)
    assert flags == [[], []]


def test_aircraft_parallel_companion():
    """ A bright detection with a fainter one moving alongside it is flagged; one moving in another direction or
        far to the side is not. """

    ra, dec = antiSun(MIDNIGHT)
    config = makeConfig(mf_flag_sunlit_factor=1e9)
    bright = makeTrack(MIDNIGHT, ra, dec, 1.0, mag=3.0)

    alongside = makeTrack(MIDNIGHT, ra, dec, 1.0, mag=7.0, offset=(0, 20))
    assert F.flagTracks([bright, alongside], LAT, LON, ELEV, config)[0][0] == ['aircraft']

    crossing = makeTrack(MIDNIGHT, ra, dec, 1.0, mag=7.0, offset=(0, 20), direction=(0.0, 1.0))
    assert F.flagTracks([bright, crossing], LAT, LON, ELEV, config)[0][0] == []

    far = makeTrack(MIDNIGHT, ra, dec, 1.0, mag=7.0, offset=(0, 200))
    assert F.flagTracks([bright, far], LAT, LON, ELEV, config)[0][0] == []


def test_sunlit_group_not_aircraft():
    """ Bright objects moving together where they can be sunlit (in daylight here) are not taken for aircraft. """

    ra, dec = antiSun(MIDNIGHT)
    a = makeTrack(NOON, ra, dec, 1.0, mag=2.5)
    b = makeTrack(NOON, ra, dec, 1.0, mag=2.5, offset=(0, 20))

    flags, _ = F.flagTracks([a, b], LAT, LON, ELEV, makeConfig())

    assert flags == [[], []]



### Wandering tracks ###

def wanderingDetection(speed, jitter, frames=200, seed=1):
    """ A detection moving speed px/frame along x, its positions scattered by jitter px RMS, with the columns of a
        detection of the matched filter [frame, x, y, intensity, background, snr, saturated]. """

    rng = np.random.default_rng(seed)
    f = np.arange(frames, dtype=np.float64)
    cent = np.column_stack([f, 100 + speed*f + rng.normal(0, jitter, frames),
                            100 + rng.normal(0, jitter, frames), np.full(frames, 1000.0), np.full(frames, 50.0),
                            np.full(frames, 10.0), np.zeros(frames)])

    return [0.0, 0.0, cent]


def test_wandering_removed():
    """ A slow long track scattered by more than the limit is removed; a smooth one and a fast scattered one are
        kept, and so is a short one. """

    fwhm = 1.6
    smooth = wanderingDetection(0.05, 0.1)
    wandering = wanderingDetection(0.05, 1.0, seed=2)
    fast = wanderingDetection(2.0, 1.0, seed=3)
    short = wanderingDetection(0.05, 1.0, frames=20, seed=4)

    kept, rejected = removeWanderingDetections([smooth, wandering, fast, short], fwhm, 0.3, 0.27, 96)

    assert [det is wandering for det in kept] == [False, False, False]
    assert len(rejected) == 1
    assert trackWander(wandering[2]) > 0.3*fwhm > trackWander(smooth[2])


def test_wandering_off():
    """ With the factor 0 everything is kept. """

    dets = [wanderingDetection(0.05, 1.0)]
    kept, rejected = removeWanderingDetections(dets, 1.6, 0.0, 0.27, 96)

    assert (len(kept), len(rejected)) == (1, 0)



### Files ###

class _Handle(object):
    """ Minimal frame-based input for writing the results. """

    input_type = 'vid'

    def __init__(self, beginning, fps=32.0):
        self.beginning_datetime = beginning
        self.fps = fps

    def currentFrameTime(self, frame_no=0, dt_obj=False):
        """ Time of a frame (datetime), at a constant frame rate. """

        return self.beginning_datetime + datetime.timedelta(seconds=frame_no/self.fps)


def makePlatepar():
    """ A simple calibration of the 512x512 camera, pointing straight up at the observer at midnight. """

    pp = Platepar()
    pp.lat, pp.lon, pp.elev = LAT, LON, ELEV
    pp.X_res, pp.Y_res = 512, 512
    pp.F_scale = 1.0/0.0175
    pp.JD = datetime2JD(MIDNIGHT)
    pp.Ho = JD2LST(pp.JD, 0)[0]
    pp.RA_d, pp.dec_d, pp.pos_angle_ref = JD2LST(pp.JD, LON)[0], LAT, 0.0
    pp.setDistortionType("radial3-odd", reset_params=True)
    pp.mag_lev = 15.0

    return pp


def test_flagged_file_name_and_merge(tmp_path):
    """ The flagged detections are saved with their flags, under a name which neither the RMS check of the results
        files nor a glob for FTPdetectinfo files matches, and the files of a night are merged. """

    config = makeConfig()
    handle = _Handle(MIDNIGHT)
    flagged = [(wanderingDetection(2.0, 0.1), ['aircraft'], {'peak_mag': 2.5, 'sunlit_ratio': 3.0}),
               (wanderingDetection(3.0, 0.1, seed=5), ['not_sunlit'], {'peak_mag': 6.0, 'sunlit_ratio': 5.0})]

    results = tmp_path/'results'
    results.mkdir()
    path = F.saveFlaggedDetections(flagged, handle, config, str(results), makePlatepar())

    name = path.split('/')[-1]
    assert name.startswith(F.FLAGGED_PREFIX)
    assert not FTPdetectinfo.validDefaultFTPdetectinfo(name)
    assert not fnmatch.fnmatch(name, 'FTPdetectinfo*')
    assert [p.name for p in results.iterdir()] and all(not p.name.startswith('FTPdetectinfo')
                                                       for p in results.iterdir())

    entries = FTPdetectinfo.readFTPdetectinfo(str(results), name)
    assert len(entries) == 2
    rows = (results/name.replace('.txt', '.csv')).read_text().splitlines()
    assert len(rows) == 3 and {'aircraft', 'not_sunlit'} == {row.split(',')[2] for row in rows[1:]}

    # Saving again with nothing flagged removes the old files of the input
    assert F.saveFlaggedDetections([], handle, config, str(results), makePlatepar()) is None
    assert not list(results.iterdir())

    # The merge of a night
    F.saveFlaggedDetections(flagged, handle, config, str(results), makePlatepar())
    night = tmp_path/'XX0001_20261022_000000_000000'
    night.mkdir()
    merged = F.mergeNightFlagged(str(night), [str(results)], config)
    assert merged.endswith('FlaggedDetections_XX0001_20261022_000000_000000.txt')
    assert len(FTPdetectinfo.readFTPdetectinfo(str(night), merged.split('/')[-1])) == 2



### Options and times ###

def test_flags_off():
    """ mf_flag_sunlit_factor 0 turns only the not-sunlit flag off; mf_flag_sunlit_speed 0 turns both off, so a
        bright sunlit group is never taken for aircraft. """

    ra, dec = antiSun(MIDNIGHT)
    a = makeTrack(MIDNIGHT, ra, dec, 1.0, mag=2.5)
    b = makeTrack(MIDNIGHT, ra + 2, dec, 1.0, mag=3.0, y0=300)

    flags, _ = F.flagTracks([a, b], LAT, LON, ELEV, makeConfig(mf_flag_sunlit_factor=0))
    assert flags == [['aircraft'], ['aircraft']]

    flags, values = F.flagTracks([a, b], LAT, LON, ELEV, makeConfig(mf_flag_sunlit_speed=0))
    assert flags == [[], []]
    assert values == [{}, {}]

    sunlit = [makeTrack(NOON, ra, dec, 1.0, mag=2.5), makeTrack(NOON, ra, dec, 1.0, mag=2.5, offset=(0, 20))]
    flags, _ = F.flagTracks(sunlit, LAT, LON, ELEV, makeConfig(mf_flag_sunlit_speed=0))
    assert flags == [[], []]


class _GapHandle(_Handle):
    """ Frame-based input whose recording stops after a frame and resumes an hour later. """

    def __init__(self, beginning, gap_frame, fps=32.0):
        super(_GapHandle, self).__init__(beginning, fps=fps)
        self.gap_frame = gap_frame

    def currentFrameTime(self, frame_no=0, dt_obj=False):
        """ Time of a frame (datetime), an hour later from the gap on. """

        t = super(_GapHandle, self).currentFrameTime(frame_no=frame_no, dt_obj=dt_obj)

        return t + datetime.timedelta(hours=1) if frame_no >= self.gap_frame else t


def brightDetection(first, x0, y0):
    """ A bright detection of 32 frames moving 1.8 px/frame along x (about 1 deg/s). """

    f = np.arange(first, first + 32, dtype=np.float64)
    n = len(f)
    cent = np.column_stack([f, x0 + 1.8*(f - first), np.full(n, y0), np.full(n, 1e6), np.full(n, 50.0),
                            np.full(n, 100.0), np.zeros(n)])

    return [0.0, 0.0, cent]


def test_times_from_recorded_frames():
    """ Two bright detections a few frames apart are aircraft; across a gap of the recording they are an hour
        apart, and are not. """

    # Only the aircraft flag
    config = makeConfig(mf_flag_sunlit_factor=0)
    dets = [brightDetection(0, 100, 200), brightDetection(40, 100, 350)]

    kept, flagged = F.splitFlagged(dets, _Handle(MIDNIGHT), config, makePlatepar())
    assert (len(kept), [f[1] for f in flagged]) == (0, [['aircraft'], ['aircraft']])

    kept, flagged = F.splitFlagged(dets, _GapHandle(MIDNIGHT, 35), config, makePlatepar())
    assert (len(kept), flagged) == (2, [])


def test_flags_csv_pairs_by_detection(tmp_path):
    """ Two flagged detections which start at the same position at different times keep their own flags in the
        CSV. """

    config = makeConfig()
    first = wanderingDetection(2.0, 0.0)
    later = wanderingDetection(2.0, 0.0)
    later[2] = later[2].copy()
    later[2][:, 0] += 1000
    flagged = [(first, ['aircraft'], {'peak_mag': 2.0, 'sunlit_ratio': 3.0}),
               (later, ['not_sunlit'], {'peak_mag': 7.0, 'sunlit_ratio': 9.0})]

    path = F.saveFlaggedDetections(flagged, _Handle(MIDNIGHT), config, str(tmp_path), makePlatepar())
    rows = [row.split(',') for row in open(path.replace('.txt', '.csv')).read().splitlines()[1:]]

    assert [(row[2], row[3]) for row in rows] == [('aircraft', '2.00'), ('not_sunlit', '7.00')]
