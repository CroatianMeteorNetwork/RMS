"""
Tests for the night reports of RMS.MonitorProcessFrameInterface (RMS.MonitorNightReport): night naming, the
report state, merging the results of a night, the best platepar of the night, the report steps, the cleanup of
old data and the scheduling of the reports.
"""

import collections
import datetime
import json
import os
import shutil

import numpy as np
import pytest

import RMS.ConfigReader as cr
import RMS.MonitorNightReport as mnr
from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, xyToRaDecPP
from RMS.Formats import CALSTARS, FFfile, FFpng, FTPdetectinfo
from RMS.Formats.Platepar import Platepar


REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_CONFIG = os.path.join(REPO_DIR, '.config')
TEMPLATE_PLATEPAR = os.path.join(REPO_DIR, 'share', 'platepar_templates', 'template_generic_720p_6mm.cal')

BEG_TIME = datetime.datetime(2025, 12, 25, 3, 0, 0)
NIGHT = 'XX0001_20251224_222000_000000'


def _config(lat=43.19, lon=-81.32, elev=300.0):
    config = cr.parse(REPO_CONFIG)
    config.stationID = 'XX0001'
    config.latitude = lat
    config.longitude = lon
    config.elevation = elev
    config.width = 64
    config.height = 48
    config.timelapse_generate_captured = False
    config.thumb_bin = 1
    config.upload_split = False
    return config


def _pairName(seconds, frame):
    return FFpng.pairNames(FFfile.constructFFName('XX0001', BEG_TIME + datetime.timedelta(seconds=seconds),
                                                  frame=frame, ext=None))[0]


def _makeResults(output_dir, config, name, n_chunks=2, fps=25.0, meteor_fps=25.0, start_s=0):
    """ Write the results of one processed input file and its chunk images, like processFile does: CALSTARS
        with one star per chunk, an FTPdetectinfo with one calibrated meteor, recalibrated platepars and the
        done flag. Return the results directory relative to the output directory.
    """

    night_dir = mnr.nightDirPath(output_dir, NIGHT, config)
    os.makedirs(night_dir, exist_ok=True)

    chunk_names = []
    for chunk in range(n_chunks):
        ff_name = _pairName(start_s + chunk*128/fps, 128*chunk)
        FFpng.writePair(night_dir, ff_name, np.full((48, 64), 100 + chunk, np.uint8),
                        np.full((48, 64), 50, np.uint8))
        chunk_names.append(ff_name)

    results_dir = os.path.join('2025', '202512', '20251225', name)
    results_path = os.path.join(output_dir, results_dir)
    os.makedirs(results_path)

    star_list = [[ff_name, [(10.0, 20.0, 1000, 50, 2.0, 7, 9.0, 0)]] for ff_name in chunk_names]
    CALSTARS.writeCALSTARS(star_list, results_path, 'CALSTARS_{:s}.txt'.format(name), 'XX0001', 48, 64,
                           chunk_frames=128, fps=fps)

    # Calibrated picks: frame, x, y, ra, dec, azim, elev, intensity, mag, background, snr, saturated
    picks = [[f, 10.0 + f, 20.0, 170.0 + f/100, 50.0, 60.0, 70.0, 1000, 2.5, 50, 8.0, 0] for f in (3.0, 4.0)]
    FTPdetectinfo.writeFTPdetectinfo([[chunk_names[0], 1, 10.0, 45.0, picks, meteor_fps]], results_path,
        'FTPdetectinfo_{:s}.txt'.format(name), results_path, 'XX0001', fps, calibration='Calibrated',
        celestial_coords_given=True)

    with open(os.path.join(results_path, config.platepars_recalibrated_name), 'w') as f:
        json.dump({ff_name: _recalibratedPlatepar(0, auto_recalibrated=False) for ff_name in chunk_names}, f)

    shutil.copy(TEMPLATE_PLATEPAR, os.path.join(results_path, config.platepar_name))
    shutil.copy(REPO_CONFIG, os.path.join(results_path, 'test.config'))

    mnr.writeDoneFlag(results_path, {'night': NIGHT, 'mask_path': None, 'dark_applied': False,
                                     'flat_applied': False})

    return results_dir


### Night naming ###

def test_night_contains_evening_and_morning():

    config = _config()

    # Local time is UTC-5: 21:00 local in the evening, and 05:00 local the next morning
    evening = datetime.datetime(2025, 12, 25, 2, 0, 0)
    morning = datetime.datetime(2025, 12, 25, 10, 0, 0)

    name_evening, start, end = mnr.nightInfo(config, evening)
    name_morning, _, _ = mnr.nightInfo(config, morning)

    assert name_evening == name_morning
    assert name_evening.startswith('XX0001_20251224_')
    assert start < evening < morning < end


def test_dusk_belongs_to_the_coming_night():

    config = _config()
    _, start, _ = mnr.nightInfo(config, datetime.datetime(2025, 12, 25, 2, 0, 0))

    # Shortly before the night starts (the Sun is above the capture horizon at dusk)
    assert mnr.nightInfo(config, start - datetime.timedelta(minutes=10))[1] == start


def test_consecutive_nights_differ():

    config = _config()

    name1, _, end1 = mnr.nightInfo(config, datetime.datetime(2025, 12, 25, 2, 0, 0))
    name2, start2, _ = mnr.nightInfo(config, datetime.datetime(2025, 12, 26, 2, 0, 0))

    assert name1 != name2
    assert end1 < start2


@pytest.mark.parametrize('dt', [datetime.datetime(2025, 12, 21, 12, 0, 0),
                                datetime.datetime(2025, 6, 21, 12, 0, 0)])
def test_polar_night_and_day(dt):

    # 78 deg N has polar night in December and polar day in June
    config = _config(lat=78.2, lon=15.6, elev=10.0)

    name, start, end = mnr.nightInfo(config, dt)

    assert start <= dt < end
    assert (end - start) == datetime.timedelta(days=1)
    assert mnr.nightInfo(config, dt + datetime.timedelta(hours=6))[0] == name


def test_nightBoundsFromName():

    config = _config()
    name, start, end = mnr.nightInfo(config, datetime.datetime(2025, 12, 25, 2, 0, 0))

    start_name, end_name = mnr.nightBoundsFromName(config, name)

    assert abs((start_name - start).total_seconds()) < 1
    assert end_name == end


### Done flags and report state ###

def test_done_flag_round_trip_and_older_flags(tmp_path):

    info = {'night': NIGHT, 'mask_path': None, 'dark_applied': True, 'flat_applied': False}
    mnr.writeDoneFlag(str(tmp_path), info)

    assert mnr.readDoneFlag(str(tmp_path)) == info

    # Results from before the night reports have an empty flag
    open(str(tmp_path/'done.flag'), 'w').close()
    assert mnr.readDoneFlag(str(tmp_path)) == {}


def test_scan_nights(tmp_path):

    config = _config()
    output_dir = str(tmp_path)

    results_dir = _makeResults(output_dir, config, 'file1')

    # Older results without a night, and done flags in the output directories of the monitor are ignored
    os.makedirs(str(tmp_path/'2025'/'old'))
    open(str(tmp_path/'2025'/'old'/'done.flag'), 'w').close()
    os.makedirs(str(tmp_path/'logs'/'x'))
    mnr.writeDoneFlag(str(tmp_path/'logs'/'x'), {'night': 'other'})

    nights = mnr.scanNights(output_dir, config)

    assert list(nights) == [NIGHT]
    assert list(nights[NIGHT]) == [results_dir]
    assert nights[NIGHT][results_dir]['dark_applied'] is False


def test_report_state(tmp_path):

    output_dir = str(tmp_path)
    results = {'2025/a': {}, '2025/b': {}}

    states = mnr.readReportStates(output_dir)
    assert mnr.unreportedResults(states, NIGHT, results) == ['2025/a', '2025/b']

    mnr.updateReportState(output_dir, NIGHT, files=['2025/a'], failed_steps=[])
    mnr.updateReportState(output_dir, latest_platepar_night=NIGHT)

    states = mnr.readReportStates(output_dir)
    assert mnr.unreportedResults(states, NIGHT, results) == ['2025/b']
    assert states['latest_platepar_night'] == NIGHT


def test_deleted_night_dir_does_not_make_the_night_pending(tmp_path):

    config = _config()
    output_dir = str(tmp_path)

    results_dir = _makeResults(output_dir, config, 'file1')
    mnr.updateReportState(output_dir, NIGHT, files=[results_dir])

    # The RMS data management deletes old night directories, the report state is kept in the output directory
    shutil.rmtree(mnr.nightDirPath(output_dir, NIGHT, config))

    states = mnr.readReportStates(output_dir)
    nights = mnr.scanNights(output_dir, config)

    assert mnr.unreportedResults(states, NIGHT, nights[NIGHT]) == []


### Merging the results of a night ###

def test_merge_night(tmp_path):

    config = _config()
    output_dir = str(tmp_path)

    # Two files with slightly different frame rates
    results_dirs = [_makeResults(output_dir, config, 'file1', fps=32.0, meteor_fps=32.13),
                    _makeResults(output_dir, config, 'file2', fps=32.0, meteor_fps=32.11, start_s=600)]

    # ECSV files of detections which begin in the same second, in one file and in both files
    ecsv_name = '2025-12-25T20_00_00_RMS_XX0001'
    for results_dir, file_names in zip(results_dirs, [(ecsv_name, ecsv_name + '_2'), (ecsv_name,)]):
        for file_name in file_names:
            with open(os.path.join(output_dir, results_dir, file_name + '.ecsv'), 'w') as f:
                f.write(results_dir + file_name)

    night_dir = mnr.nightDirPath(output_dir, NIGHT, config)
    results = mnr.scanNights(output_dir, config)[NIGHT]

    merged = mnr.mergeNightResults(night_dir, output_dir, results, config)

    # One CALSTARS with all chunks, named by the image pairs in the night directory
    assert [f for f in os.listdir(night_dir) if f.startswith('CALSTARS')] == [merged.calstars_name]
    star_list, chunk_frames, fps = CALSTARS.readCALSTARS(night_dir, merged.calstars_name, return_fps=True)
    night_ffs = set(f for f in os.listdir(night_dir) if FFfile.validFFName(f))
    assert set(entry[0] for entry in star_list) == night_ffs
    assert (chunk_frames, fps, merged.chunk_frames, merged.fps) == (128, 32.0, 128, 32.0)

    # One FTPdetectinfo with both meteors, keeping their frame rates and celestial coordinates
    entries = FTPdetectinfo.readFTPdetectinfo(night_dir, merged.ftpdetectinfo_name)
    assert [entry[4] for entry in entries] == [32.13, 32.11]
    assert entries[0][11][0][4:6] == [170.03, 50.0]
    assert merged.n_meteors == 2
    assert merged.ff_detected == sorted(entry[0] for entry in entries)

    # All ECSV files are collected under unique names
    ecsv_dir = os.path.join(night_dir, mnr.ECSV_DIR_NAME)
    assert sorted(os.listdir(ecsv_dir)) == [ecsv_name + suffix + '.ecsv' for suffix in ('', '_2', '_3')]
    with open(os.path.join(ecsv_dir, ecsv_name + '_3.ecsv')) as f:
        assert f.read() == results_dirs[1] + ecsv_name

    # Recalibrated platepars of both files, and the platepar and the config are copied
    assert len(merged.recalibrated) == 4
    assert os.path.isfile(os.path.join(night_dir, config.platepar_name))
    assert os.path.isfile(os.path.join(night_dir, 'test.config'))

    # Merging again doesn't duplicate the merged files
    mnr.mergeNightResults(night_dir, output_dir, results, config)
    assert len([f for f in os.listdir(night_dir) if f.startswith('FTPdetectinfo')]) == 1
    assert len(os.listdir(ecsv_dir)) == 3


def test_detected_images_cover_the_meteor_tracks():

    # Images every 5.12 s (128 frames at 25 fps) of two files, the second one starting after a gap
    image_names = [_pairName(seconds, frame) for seconds, frame in ((0, 0), (5.12, 128), (10.24, 256),
                                                                    (60, 0), (65.12, 128))]
    meas = lambda frames: [[frame, 10.0, 20.0] for frame in frames]

    meteor_list = [
        # Within one image
        [image_names[0], 1, 0, 0, meas([3, 4, 5]), 25.0],
        # Starting in the second image and ending in the third one
        [image_names[1], 1, 0, 0, meas([120, 150, 200]), 25.0],
        # Ending in the frames after the last chunk of the first file, which belong to its last image
        [image_names[2], 1, 0, 0, meas([100, 200]), 25.0],
        # No image of the meteor
        [_pairName(30, 0), 1, 0, 0, meas([3, 4]), 25.0],
        ]

    assert mnr.detectedImageNames(meteor_list, image_names[::-1]) == image_names[:3]

    # A track over both images of the second file
    meteor_list = [[image_names[3], 1, 0, 0, meas([10, 200, 300]), 25.0]]
    assert mnr.detectedImageNames(meteor_list, image_names) == image_names[3:]


### Best platepar of the night ###

def _recalibratedPlatepar(n_stars, tag=0, auto_recalibrated=True):
    """ Recalibrated platepar as stored in the JSON file. The tag is stored as the star intensity, so the
        platepar can be identified.
    """
    pp = Platepar()
    pp.read(TEMPLATE_PLATEPAR)
    pp.auto_recalibrated = auto_recalibrated
    pp.star_list = [[2461034.8, 10.0 + i, 20.0, tag, 170.0, 50.0, 8.0] for i in range(n_stars)]
    return json.loads(pp.jsonStr())


def test_plateparResidual():

    platepar = Platepar()
    platepar.read(TEMPLATE_PLATEPAR)

    # Catalog stars in the field of view, and their exact image coordinates
    time_data = [(2025, 12, 25, 3, 0, 0, 0)]*4
    jd, ra, dec, _ = xyToRaDecPP(time_data, [100.0, 640.0, 1000.0, 300.0], [100.0, 360.0, 600.0, 500.0],
                                 [1]*4, platepar, extinction_correction=False)
    x, y = raDecToXYPP(ra, dec, np.mean(jd), platepar)

    # The star list stores the image coordinates as Y X
    platepar.star_list = [[j, yy, xx, 1000, r, d, 5.0] for j, yy, xx, r, d in zip(jd, y, x, ra, dec)]
    assert mnr.plateparResidual(platepar) < 0.01

    # Swapped image coordinates are far off
    platepar.star_list = [[j, xx, yy, 1000, r, d, 5.0] for j, yy, xx, r, d in zip(jd, y, x, ra, dec)]
    assert mnr.plateparResidual(platepar) > 10


def test_select_best_night_platepar(monkeypatch):

    recalibrated = {
        'FF_a': _recalibratedPlatepar(150, tag=1),
        'FF_b': _recalibratedPlatepar(197, tag=2),
        'FF_c': _recalibratedPlatepar(197, tag=3),
        # Most stars, but not recalibrated successfully
        'FF_d': _recalibratedPlatepar(250, tag=4, auto_recalibrated=False),
    }

    # FF_c fits better than FF_b with the same number of stars, FF_a fits best but has fewer stars
    residuals = {1: 0.03, 2: 0.05, 3: 0.04, 4: 0.01}
    computed = []

    def _residual(pp):
        computed.append(pp.star_list[0][3])
        return residuals[pp.star_list[0][3]]

    monkeypatch.setattr(mnr, 'plateparResidual', _residual)

    ff_name, platepar, n_stars, residual = mnr.selectBestNightPlatepar(recalibrated)

    assert (ff_name, n_stars, residual) == ('FF_c', 197, 0.04)

    # Only the platepars with the most stars are compared
    assert sorted(computed) == [2, 3]

    assert mnr.selectBestNightPlatepar({'FF_a': _recalibratedPlatepar(100, auto_recalibrated=False)}) is None


def test_older_night_does_not_overwrite_the_latest_platepar(tmp_path):

    config = _config()
    output_dir = str(tmp_path)
    night_dir = str(tmp_path/'night')
    os.makedirs(night_dir)

    platepar = Platepar()
    platepar.read(TEMPLATE_PLATEPAR)

    later_night = 'XX0001_20251225_222000_000000'
    mnr.saveNightPlatepar(platepar, night_dir, later_night, output_dir, config)
    latest_mtime = os.path.getmtime(mnr.latestPlateparPath(output_dir, config))

    # Reporting an older night again saves its platepar in its night directory only
    platepar.RA_d += 1
    mnr.saveNightPlatepar(platepar, night_dir, NIGHT, output_dir, config)

    latest = Platepar()
    latest.read(mnr.latestPlateparPath(output_dir, config))

    assert latest.RA_d == platepar.RA_d - 1
    assert os.path.getmtime(mnr.latestPlateparPath(output_dir, config)) == latest_mtime
    assert mnr.readReportStates(output_dir)['latest_platepar_night'] == later_night


### Night report ###

def test_generate_night_report(tmp_path):

    config = _config()
    output_dir = str(tmp_path)
    results_dir = _makeResults(output_dir, config, 'file1', n_chunks=3)

    # A file left in the archive by a previous report
    archived_dir = os.path.join(output_dir, config.archived_dir, NIGHT)
    os.makedirs(archived_dir)
    open(os.path.join(archived_dir, 'stale.txt'), 'w').close()

    night_state = mnr.generateNightReport(output_dir, NIGHT, config)

    assert night_state['files'] == [results_dir]
    assert 'archive' in night_state['ok_steps']
    assert night_state['upload_files'] == [os.path.join(output_dir, config.archived_dir,
                                                        NIGHT + '_detected.tar.bz2')]

    night_files = os.listdir(mnr.nightDirPath(output_dir, NIGHT, config))
    assert NIGHT + '_captured_stack.jpg' in night_files
    assert any(f.endswith('_CAPTURED_thumbs.jpg') for f in night_files)

    # The archive was made from scratch
    assert 'stale.txt' not in os.listdir(archived_dir)


def test_failed_optional_product_still_reports_the_night(tmp_path, monkeypatch):

    import Utils.ShowerAssociation

    config = _config()
    config.monitor_shower_association = True

    def _fail(*args, **kwargs):
        raise RuntimeError("no showers")

    monkeypatch.setattr(Utils.ShowerAssociation, 'showerAssociation', _fail)

    output_dir = str(tmp_path)
    results_dir = _makeResults(output_dir, config, 'file1')

    night_state = mnr.generateNightReport(output_dir, NIGHT, config, archive=False)

    assert 'shower_association' in night_state['failed_steps']
    assert night_state['files'] == [results_dir]


def test_failed_essential_step_keeps_the_night_pending(tmp_path, monkeypatch):

    import RMS.ArchiveDetections

    def _fail(*args, **kwargs):
        raise RuntimeError("archiving failed")

    monkeypatch.setattr(RMS.ArchiveDetections, 'archiveDetections', _fail)

    config = _config()
    output_dir = str(tmp_path)
    _makeResults(output_dir, config, 'file1')

    night_state = mnr.generateNightReport(output_dir, NIGHT, config)

    assert 'archive' in night_state['failed_steps']
    assert 'files' not in night_state

    states = mnr.readReportStates(output_dir)
    assert mnr.unreportedResults(states, NIGHT, mnr.scanNights(output_dir, config)[NIGHT])


def test_report_runs_only_enabled_products(tmp_path, monkeypatch):

    import Utils.ShowerAssociation

    config = _config()
    config.monitor_shower_association = True

    calls = []
    monkeypatch.setattr(Utils.ShowerAssociation, 'showerAssociation',
                        lambda config, ftp_list, **kwargs: calls.append(ftp_list))

    output_dir = str(tmp_path)
    _makeResults(output_dir, config, 'file1')

    night_state = mnr.generateNightReport(output_dir, NIGHT, config, archive=False)

    night_dir = mnr.nightDirPath(output_dir, NIGHT, config)
    assert calls == [[os.path.join(night_dir, 'FTPdetectinfo_{:s}.txt'.format(NIGHT))]]

    for step in ['fov_kml', 'flux', 'observation_summary']:
        assert step not in night_state['ok_steps'] + night_state['failed_steps']


@pytest.mark.parametrize('thumb_stack, chunk_frames, expected', [(5, 128, 10), (5, 256, 5), (5, 512, 2),
                                                                   (1, 1024, 1), (5, None, 5), (3, 100, 8)])
def test_scaled_thumb_stack(thumb_stack, chunk_frames, expected):
    assert mnr.scaledThumbStack(thumb_stack, chunk_frames) == expected


def test_report_scales_thumbnail_stacking(tmp_path, monkeypatch):

    import RMS.ArchiveDetections

    stacks = []
    monkeypatch.setattr(RMS.ArchiveDetections, 'generateThumbsAndStacks',
                        lambda night_dir, config, ff_detected: stacks.append(config.thumb_stack))

    config = _config()
    config.thumb_stack = 5

    output_dir = str(tmp_path)
    _makeResults(output_dir, config, 'file1')

    mnr.generateNightReport(output_dir, NIGHT, config, archive=False)

    # The chunks have 128 frames, so twice as many are stacked as with 256-frame FF files, and the given
    #   config is not changed
    assert stacks == [10]
    assert config.thumb_stack == 5


def test_observation_summary_counts_image_pairs(tmp_path):

    from RMS.Formats.ObservationSummary import nightSummaryData

    config = _config()
    config.fps = 25.0

    # Three consecutive 128-frame chunks (5.12 s each)
    for i in range(3):
        FFpng.writePair(str(tmp_path), _pairName(5.12*i, 128*i), np.zeros((4, 4), np.uint8),
                        np.zeros((4, 4), np.uint8))

    result = nightSummaryData(config, str(tmp_path), frames_per_file=128)
    capture_duration, fits_count, total_expected = result[0], result[4], result[11]

    assert fits_count == 3
    assert capture_duration == pytest.approx(3*5.12, abs=0.01)
    assert total_expected == 3


### Cleanup ###

def test_cleanup_keeps_unreported_nights(tmp_path, monkeypatch):

    config = _config()
    config.capt_dirs_to_keep = 2
    output_dir = str(tmp_path)

    # Five night directories, the second oldest has unreported files
    captured_dir = os.path.join(output_dir, config.captured_dir)
    nights = ['XX0001_2025122{:d}_222000_000000'.format(i) for i in range(5)]
    for night_name in nights:
        os.makedirs(os.path.join(captured_dir, night_name))

    results_path = tmp_path/'2025'/'r1'
    os.makedirs(str(results_path))
    mnr.writeDoneFlag(str(results_path), {'night': nights[1]})

    calls = []

    def _deleteOldObservations(data_dir, captured_dir, archived_dir, config, duration=None,
                               needed_bytes=None):
        calls.append((data_dir, captured_dir, archived_dir, config.capt_dirs_to_keep, config.log_dir))
        return True

    monkeypatch.setattr(mnr, 'deleteOldObservations', _deleteOldObservations)

    assert mnr.cleanupOldData(output_dir, config)

    # The night directories back to the unreported night are kept
    assert calls == [(output_dir, config.captured_dir, config.archived_dir, 4, 'logs')]

    config.monitor_delete_old_data = False
    calls.clear()
    mnr.cleanupOldData(output_dir, config)
    assert calls == []


def test_cleanup_needs_the_space_of_the_largest_recent_night(tmp_path, monkeypatch):

    config = _config()
    output_dir = str(tmp_path)

    # Four nights of different sizes, only the last three are considered
    captured_dir = os.path.join(output_dir, config.captured_dir)
    for i, size in enumerate([5000, 1000, 3000, 2000]):
        night_dir = os.path.join(captured_dir, 'XX0001_2025122{:d}_222000_000000'.format(i))
        os.makedirs(night_dir)
        with open(os.path.join(night_dir, 'FF_pair.png'), 'wb') as f:
            f.write(b'0'*size)

    needed = []
    monkeypatch.setattr(mnr, 'deleteOldObservations', lambda *args, **kwargs: needed.append(
        kwargs['needed_bytes']) or True)

    mnr.cleanupOldData(output_dir, config)
    assert needed == [config.extra_space_gb*1024**3 + 3000]


def test_delete_old_night_images(tmp_path):

    config = _config()
    config.monitor_delete_images_days = 2
    output_dir = str(tmp_path)

    results_dir = _makeResults(output_dir, config, 'file1')
    night_dir = mnr.nightDirPath(output_dir, NIGHT, config)

    reported_at = datetime.datetime(2025, 12, 25, 12, 0, 0)
    mnr.updateReportState(output_dir, NIGHT, files=[results_dir],
                          reported_at=reported_at.strftime(mnr.JSON_TIME_FORMAT))

    def _deleteImages(days_later):
        return mnr.deleteOldNightImages(output_dir, config, mnr.readReportStates(output_dir),
                                        mnr.scanNights(output_dir, config),
                                        now=reported_at + datetime.timedelta(days=days_later))

    # Not old enough
    assert _deleteImages(1) == []

    # A late file of the night was not reported yet
    late_dir = _makeResults(output_dir, config, 'file2', start_s=600)
    assert _deleteImages(3) == []

    # Once everything is reported, only the image pairs are deleted
    mnr.updateReportState(output_dir, NIGHT, files=[results_dir, late_dir])
    assert _deleteImages(3) == [NIGHT]
    assert not [f for f in os.listdir(night_dir) if FFpng.isPairName(f)]


### Scheduling ###

class _Process(object):
    """ Stands in for multiprocessing.Process. """

    def __init__(self, target=None, args=()):
        self.alive = True
        self.exitcode = 0
        self.args = args

    def start(self):
        pass

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        pass


Scheduling = collections.namedtuple('Scheduling', ['output_dir', 'config_path', 'night_end', 'last_result',
                                                   'results'])


@pytest.fixture
def scheduling(tmp_path, monkeypatch):
    """ A night with one processed file, which was finished 30 minutes before the end of the night. """

    monkeypatch.setattr(mnr.multiprocessing, 'Process', _Process)

    config_path = str(tmp_path/'test.config')
    shutil.copy(REPO_CONFIG, config_path)
    config = cr.parse(config_path)

    output_dir = str(tmp_path/'out')
    night_name, _, night_end = mnr.nightInfo(config, BEG_TIME)

    results_path = os.path.join(output_dir, '2025', 'r1')
    os.makedirs(results_path)
    mnr.writeDoneFlag(results_path, {'night': night_name})

    last_result = night_end - datetime.timedelta(minutes=30)
    epoch = (last_result - datetime.datetime(1970, 1, 1)).total_seconds()
    os.utime(os.path.join(results_path, 'done.flag'), (epoch, epoch))

    return Scheduling(output_dir, config_path, night_end, last_result, mnr.scanNights(output_dir, config))


def _reporter(scheduling, mode, **kwargs):
    """ Night reporter which already ran its startup cleanup. """

    reporter = mnr.NightReporter(scheduling.output_dir, scheduling.config_path, report_mode=mode, **kwargs)
    reporter.cleanup_due = False
    reporter.last_cleanup = scheduling.night_end + datetime.timedelta(days=10)

    return reporter


def _nightAndResults(scheduling):
    return list(scheduling.results.items())[0]


def test_sunrise_mode_reports_after_the_night_without_waiting_for_idle(scheduling):

    reporter = _reporter(scheduling, 'sunrise')
    night, results = _nightAndResults(scheduling)

    assert not reporter.isDue(night, results, True, scheduling.night_end - datetime.timedelta(minutes=1))

    # The camera may already be processing the next night
    assert reporter.isDue(night, results, False, scheduling.night_end + datetime.timedelta(minutes=1))


def test_partialReportTime():

    config = _config()
    assert mnr.partialReportTime(config, NIGHT) is None

    night_start, night_end = mnr.nightBoundsFromName(config, NIGHT)

    # The first given time of day after the beginning of the night, in the morning or still in the evening
    for report_time in [night_end + datetime.timedelta(hours=2),
                        night_start + datetime.timedelta(minutes=30)]:
        config.monitor_partial_report_time = report_time.strftime("%H:%M")
        assert mnr.partialReportTime(config, NIGHT) == report_time.replace(second=0, microsecond=0)


def test_partial_report_while_the_night_is_processed(scheduling):

    reporter = _reporter(scheduling, 'sunrise')
    night, results = _nightAndResults(scheduling)

    # The partial report time is before the end of the night, and files are still being processed
    partial_time = (scheduling.night_end - datetime.timedelta(hours=2)).replace(second=0, microsecond=0)
    reporter.config.monitor_partial_report_time = partial_time.strftime("%H:%M")
    reporter.states = mnr.readReportStates(scheduling.output_dir)

    def _isDue(now):

        # The latest result was just finished
        epoch = (now - datetime.timedelta(minutes=1) - datetime.datetime(1970, 1, 1)).total_seconds()
        os.utime(os.path.join(scheduling.output_dir, '2025', 'r1', 'done.flag'), (epoch, epoch))

        return reporter.isDue(night, results, False, now)

    assert _isDue(partial_time - datetime.timedelta(minutes=1)) is None
    assert _isDue(partial_time + datetime.timedelta(minutes=1)) == 'partial'

    # Only once per night: not after a report which was made after the partial report time
    mnr.updateReportState(scheduling.output_dir, night, reported_at=(partial_time + datetime.timedelta(
        minutes=5)).strftime(mnr.JSON_TIME_FORMAT))
    reporter.states = mnr.readReportStates(scheduling.output_dir)
    assert _isDue(partial_time + datetime.timedelta(minutes=10)) is None

    # Not for old nights, e.g. a backlog after a downtime, which get only the normal report
    mnr.updateReportState(scheduling.output_dir, night, reported_at=None)
    reporter.states = mnr.readReportStates(scheduling.output_dir)
    assert _isDue(partial_time + datetime.timedelta(days=2)) is None


def test_idle_mode_waits_for_idle_and_quiet_time(scheduling):

    reporter = _reporter(scheduling, 'idle')
    night, results = _nightAndResults(scheduling)
    quiet = datetime.timedelta(minutes=reporter.config.monitor_report_quiet_min)

    assert not reporter.isDue(night, results, True, scheduling.last_result + quiet/2)
    assert not reporter.isDue(night, results, False, scheduling.last_result + 2*quiet)
    assert reporter.isDue(night, results, True, scheduling.last_result + 2*quiet)


@pytest.mark.parametrize('mode, reported', [('external', True), ('none', False)])
def test_trigger_file(scheduling, mode, reported):

    reporter = _reporter(scheduling, mode)
    now = scheduling.night_end + datetime.timedelta(days=1)

    reporter.poll(True, now=now)
    assert reporter.active is None

    trigger_path = os.path.join(scheduling.output_dir, mnr.REPORT_TRIGGER_FILE_NAME)
    open(trigger_path, 'w').close()
    reporter.poll(True, now=now)

    assert (reporter.active is not None) == reported
    assert os.path.exists(trigger_path) != reported


def test_failed_report_retry_and_give_up(scheduling):

    reporter = _reporter(scheduling, 'idle', fail_wait_time=300)
    night, results = _nightAndResults(scheduling)
    now = scheduling.night_end + datetime.timedelta(hours=1)

    # Wait before retrying a failed report
    reporter.failed[night] = {'count': 1, 'time': now, 'n_files': len(results)}
    assert not reporter.isDue(night, results, True, now + datetime.timedelta(seconds=100))
    assert reporter.isDue(night, results, True, now + datetime.timedelta(seconds=400))

    # Give up after the second failure, also if the night was triggered...
    reporter.failed[night] = {'count': 2, 'time': now, 'n_files': len(results)}
    reporter.triggered = {night}
    reporter.poll(True, now=now + datetime.timedelta(days=1))
    assert reporter.active is None
    assert reporter.triggered == set()

    # ... until new files arrive
    new_results_path = os.path.join(scheduling.output_dir, '2025', 'r2')
    os.makedirs(new_results_path)
    mnr.writeDoneFlag(new_results_path, {'night': night})
    epoch = (now - datetime.datetime(1970, 1, 1)).total_seconds()
    os.utime(os.path.join(new_results_path, 'done.flag'), (epoch, epoch))
    more_results = mnr.scanNights(scheduling.output_dir, reporter.config)[night]
    assert reporter.isDue(night, more_results, True, now + datetime.timedelta(days=1))


def test_report_start_and_finish(scheduling):

    reporter = _reporter(scheduling, 'idle')
    night, results = _nightAndResults(scheduling)

    # The archives of a previous report are not uploaded again if this report fails
    mnr.updateReportState(scheduling.output_dir, night, upload_files=['/old_archive.tar.bz2'])

    reporter._startReport(night, results, 'sunrise')
    assert mnr.readReportStates(scheduling.output_dir)['nights'][night]['upload_files'] == []
    assert reporter.report_lock.owner is reporter

    # A finished report releases the lock and makes the reporter scan the results again
    reporter.active[0].alive = False
    reporter._finishReport(scheduling.night_end)

    assert reporter.active is None
    assert reporter.report_lock.owner is None
    assert reporter.pending is None


def test_report_finished_during_shutdown_is_uploaded(scheduling):

    reporter = _reporter(scheduling, 'idle')
    night, results = _nightAndResults(scheduling)

    added = []

    class _UploadManager(object):
        def addFiles(self, file_list):
            added.append(file_list)
        def delayNextUpload(self, delay=0):
            pass
        def stop(self, timeout=None):
            pass

    reporter.upload_manager = _UploadManager()
    reporter._startReport(night, results, 'sunrise')
    mnr.updateReportState(scheduling.output_dir, night, upload_files=['/archive_detected.tar.bz2'])

    # The report finishes while the monitor is stopping
    reporter.active[0].alive = False
    reporter.stop()

    assert added == [['/archive_detected.tar.bz2']]


def test_only_one_camera_reports_at_a_time(scheduling):

    lock = mnr.ReportLock()
    camera1 = _reporter(scheduling, 'idle', report_lock=lock)
    camera2 = _reporter(scheduling, 'idle', report_lock=lock)

    now = scheduling.night_end + datetime.timedelta(days=1)

    camera1.poll(True, now=now)
    camera2.poll(True, now=now)

    assert camera1.active is not None
    assert camera2.active is None


def test_cleanup_runs_at_startup_and_after_reports(scheduling):

    reporter = mnr.NightReporter(scheduling.output_dir, scheduling.config_path, report_mode='idle')
    now = scheduling.night_end + datetime.timedelta(days=1)

    # A cleanup runs at startup, and no report is started while it runs
    reporter.poll(True, now=now)
    reporter.poll(True, now=now)
    assert (reporter.cleanup_proc is not None) and (reporter.active is None)

    # After the cleanup the due report runs
    reporter.cleanup_proc.alive = False
    reporter.poll(True, now=now)
    assert (reporter.cleanup_proc is None) and (reporter.active is not None)

    # After the report another cleanup runs
    reporter.active[0].alive = False
    reporter.poll(True, now=now)
    assert (reporter.active is None) and (reporter.cleanup_proc is not None)


### Output directory lock ###

def _holdLock(output_dir, locked, release):
    lock = mnr.lockOutputDir(output_dir)
    locked.set()
    release.wait(30)


def test_only_one_process_locks_the_output_dir(tmp_path):

    import multiprocessing

    output_dir = str(tmp_path)
    locked, release = multiprocessing.Event(), multiprocessing.Event()

    proc = multiprocessing.Process(target=_holdLock, args=(output_dir, locked, release))
    proc.start()
    assert locked.wait(30)

    # Locked by the other process, whose PID is noted
    assert mnr.lockOutputDir(output_dir) is None
    assert mnr.lockOwner(output_dir) == str(proc.pid)

    # The lock is released when the process ends, also if it is killed
    proc.kill()
    proc.join()
    assert mnr.lockOutputDir(output_dir) is not None


def test_damaged_state_file_is_moved_aside(tmp_path):

    state_path = os.path.join(str(tmp_path), mnr.REPORT_STATE_FILE_NAME)
    with open(state_path, 'w') as f:
        f.write('{"nights": {"XX')

    # The monitor starts over instead of failing
    assert mnr.readReportStates(str(tmp_path)) == {'nights': {}, 'latest_platepar_night': None}
    assert os.path.isfile(state_path + '.corrupt') and not os.path.exists(state_path)

    mnr.updateReportState(str(tmp_path), NIGHT, files=['a'])
    assert mnr.readReportStates(str(tmp_path))['nights'][NIGHT]['files'] == ['a']
