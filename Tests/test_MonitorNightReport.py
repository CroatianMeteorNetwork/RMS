"""
Tests for the night reports of RMS.MonitorProcessFrameInterface (RMS.MonitorNightReport): night naming, the
report state, merging the results of a night, the best platepar of the night, the report steps, the cleanup of
old data and the scheduling of the reports.
"""

import collections
import datetime
import glob
import json
import os
import shutil
import time

import numpy as np
import pytest

import RMS.ConfigReader as cr
import RMS.MonitorNightReport as mnr
from RMS.Astrometry.ApplyAstrometry import raDecToXYPP, xyToRaDecPP
from RMS.Formats import CALSTARS, FFfile, FFpng, FTPdetectinfo
from RMS.Formats.Platepar import Platepar
from RMS.MatchedFilterDetection import MATCHED_FILTER_DIR, DONE_NAME


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


def _makeResults(output_dir, config, name, n_chunks=2, fps=25.0, meteor_fps=25.0, start_s=0,
                 pick_frames=(3.0, 4.0)):
    """ Write the results of one processed input file and its chunk images, like processFile does: CALSTARS
        with one star per chunk, an FTPdetectinfo with one calibrated meteor (with picks at pick_frames,
        relative to the first chunk), recalibrated platepars and the done flag. Return the results directory
        relative to the output directory.
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
    picks = [[f, 10.0 + f, 20.0, 170.0 + f/100, 50.0, 60.0, 70.0, 1000, 2.5, 50, 8.0, 0] for f in pick_frames]
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
    assert mnr.unreportedResults(output_dir, states, NIGHT, results) == ['2025/a', '2025/b']

    mnr.updateReportState(output_dir, NIGHT, files=['2025/a'], failed_steps=[])
    mnr.updateReportState(output_dir, latest_platepar_night=NIGHT)

    states = mnr.readReportStates(output_dir)
    assert mnr.unreportedResults(output_dir, states, NIGHT, results) == ['2025/b']
    assert states['latest_platepar_night'] == NIGHT


def test_reprocessed_file_makes_the_night_pending(tmp_path):

    config = _config()
    output_dir = str(tmp_path)

    results_dir = _makeResults(output_dir, config, 'file1')
    done_flag = os.path.join(output_dir, results_dir, mnr.DONE_FLAG_NAME)
    os.utime(done_flag, (time.time() - 60, time.time() - 60))

    mnr.generateNightReport(output_dir, NIGHT, config, archive=True)
    states = mnr.readReportStates(output_dir)
    assert not mnr.isPending(output_dir, config, states, NIGHT, mnr.scanNights(output_dir, config)[NIGHT])

    # The file is processed again (e.g. with --force) into the same results directory
    os.utime(done_flag, None)

    assert mnr.isPending(output_dir, config, states, NIGHT, mnr.scanNights(output_dir, config)[NIGHT])


def test_deleted_night_dir_does_not_make_the_night_pending(tmp_path):

    config = _config()
    output_dir = str(tmp_path)

    results_dir = _makeResults(output_dir, config, 'file1')
    mnr.updateReportState(output_dir, NIGHT, files=[results_dir])

    # The RMS data management deletes old night directories, the report state is kept in the output directory
    shutil.rmtree(mnr.nightDirPath(output_dir, NIGHT, config))

    states = mnr.readReportStates(output_dir)
    nights = mnr.scanNights(output_dir, config)

    assert mnr.unreportedResults(output_dir, states, NIGHT, nights[NIGHT]) == []


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


def test_merge_keeps_uncalibrated_detections_uncalibrated(tmp_path):
    """ The detections of a file processed without a platepar (the automatic calibration had none yet) stay
        uncalibrated in the merged FTPdetectinfo, the others are recalibrated. """

    config = _config()
    output_dir = str(tmp_path)
    results_dirs = [_makeResults(output_dir, config, 'file1'), _makeResults(output_dir, config, 'file2',
                                                                            start_s=600)]

    # The second file has no calibration: its detections have no celestial coordinates
    results_path = os.path.join(output_dir, results_dirs[1])
    ftp_name = 'FTPdetectinfo_file2.txt'
    entry = FTPdetectinfo.readFTPdetectinfo(results_path, ftp_name)[0]
    picks = [[line[1], line[2], line[3], 0.0, 0.0, 0.0, 0.0] + list(line[8:]) for line in entry[11]]
    FTPdetectinfo.writeFTPdetectinfo([[entry[0], 1, 10.0, 45.0, picks, entry[4]]], results_path, ftp_name,
                                     results_path, 'XX0001', 32.0, calibration=None, celestial_coords_given=True)

    night_dir = mnr.nightDirPath(output_dir, NIGHT, config)
    merged = mnr.mergeNightResults(night_dir, output_dir, mnr.scanNights(output_dir, config)[NIGHT], config)

    # The calibrated meteor is marked recalibrated, the other one uncalibrated
    entries = FTPdetectinfo.readFTPdetectinfo(night_dir, merged.ftpdetectinfo_name)
    assert [entry[11][0][0] for entry in entries] == [1, 0]
    with open(os.path.join(night_dir, merged.ftpdetectinfo_name)) as f:
        lines = f.read().splitlines()
    status = [lines[i + 1] for i, line in enumerate(lines) if FFfile.validFFName(line)]
    assert status[0].startswith('Recalibrated with RMS') and (status[1] == 'Uncalibrated')


def test_matched_filter_results_are_merged_apart(tmp_path):
    """ The night report merges the finished results of the matched filter of every file into their own
        file in the matched-filter directory of the night, and the normal merged results only have the normal
        detections.
    """

    config = _config()
    config.mf_enable = True
    output_dir = str(tmp_path)

    # Results of two files, each with two detections of the matched filter in its own directory
    results_dirs = [_makeResults(output_dir, config, 'file1'), _makeResults(output_dir, config, 'file2',
                                                                            start_s=600)]
    for results_dir in results_dirs:
        # The detections of the matched filter are made from the normal detection of the file (number 1) and
        #   a copy of it (number 2), converted from the format of the reader to the format of the writer
        results_path = os.path.join(output_dir, results_dir)
        mf_path = os.path.join(results_path, MATCHED_FILTER_DIR)
        os.makedirs(mf_path)
        ftp_name = [f for f in os.listdir(results_path) if f.startswith('FTPdetectinfo_')][0]
        meteors = [[e[0], 1, e[9], e[10], [line[1:] for line in e[11]], e[4]]
                   for e in FTPdetectinfo.readFTPdetectinfo(results_path, ftp_name)]
        meteors += [[m[0], 2] + m[2:] for m in meteors]
        FTPdetectinfo.writeFTPdetectinfo(meteors, mf_path, ftp_name.replace('.txt', '_mf.txt'), mf_path,
            'XX0001', 25.0, celestial_coords_given=True)

        # The backup which the recalibration leaves is not merged
        FTPdetectinfo.writeFTPdetectinfo(meteors, mf_path, ftp_name.replace('.txt', '_mf_backup_1.txt'), mf_path,
            'XX0001', 25.0, celestial_coords_given=True)

        # Finished results have a done file
        with open(os.path.join(mf_path, DONE_NAME), 'w') as f:
            f.write('{}')

    # Results of a third file whose matched filter didn't finish (no done file) are not merged
    unfinished = _makeResults(output_dir, config, 'file3', start_s=1200)
    unfinished_path = os.path.join(output_dir, unfinished)
    os.makedirs(os.path.join(unfinished_path, MATCHED_FILTER_DIR))
    ftp_name = [f for f in os.listdir(unfinished_path) if f.startswith('FTPdetectinfo_')][0]
    meteors = [[e[0], 1, e[9], e[10], [line[1:] for line in e[11]], e[4]]
               for e in FTPdetectinfo.readFTPdetectinfo(unfinished_path, ftp_name)]
    FTPdetectinfo.writeFTPdetectinfo(meteors, os.path.join(unfinished_path, MATCHED_FILTER_DIR),
        ftp_name.replace('.txt', '_mf.txt'), unfinished_path, 'XX0001', 25.0, celestial_coords_given=True)

    # The merging of the matched filter is a step of the report, and it succeeded
    night_state = mnr.generateNightReport(output_dir, NIGHT, config, archive=True)
    assert 'matched_filter_merge' in night_state['ok_steps']

    # The normal results only have the normal detections (of the three files)
    night_dir = mnr.nightDirPath(output_dir, NIGHT, config)
    ftp_names = [f for f in os.listdir(night_dir) if f.startswith('FTPdetectinfo')]
    assert len(ftp_names) == 1
    assert len(FTPdetectinfo.readFTPdetectinfo(night_dir, ftp_names[0])) == 3

    # The finished detections of the matched filter are merged into their own directory
    mf_night = os.path.join(night_dir, MATCHED_FILTER_DIR)
    assert os.listdir(mf_night) == ['FTPdetectinfo_{:s}_mf.txt'.format(NIGHT)]
    # Two detections of each of the two finished files, with the coordinates of the original detections
    entries = FTPdetectinfo.readFTPdetectinfo(mf_night, os.listdir(mf_night)[0])
    assert len(entries) == 4
    assert entries[0][11][0][4:6] == [170.03, 50.0]


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


def test_night_platepar_only_from_the_newest_platepar(tmp_path):
    """ With the automatic calibration, the platepar of the night is selected only from the files calibrated with
        the newest platepar of the night (a platepar which was replaced during the night is not carried forward). """

    config = _config()
    output_dir = str(tmp_path)

    # Two files of the night: one calibrated with an older platepar, one with the newest
    results = {}
    for name, mtime in [('old', 1000.0), ('new', 2000.0)]:
        results_dir = os.path.join('2025', name)
        os.makedirs(os.path.join(output_dir, results_dir))
        with open(os.path.join(output_dir, results_dir, config.platepars_recalibrated_name), 'w') as f:
            json.dump({'FF_' + name: {}}, f)
        results[results_dir] = {'platepar_mtime': mtime}

    recalibrated = {'FF_old': {'a': 1}, 'FF_new': {'b': 2}}
    assert mnr.newestPlateparRecalibrations(output_dir, results, recalibrated, config) == {'FF_new': {'b': 2}}

    # Without the times (results processed before they were recorded), all are kept
    results = {results_dir: {} for results_dir in results}
    assert mnr.newestPlateparRecalibrations(output_dir, results, recalibrated, config) == recalibrated


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


@pytest.mark.parametrize('update, auto, newest, expected', [
    (True, True, 'auto', 'auto'), (True, True, 'latest', 'latest'), (True, True, 'given', 'given'),
    (False, True, 'latest', 'auto'), (True, False, 'auto', 'latest'), (False, False, 'auto', 'given')])
def test_select_platepar_takes_the_newest_enabled(tmp_path, update, auto, newest, expected):
    """ The newest of the given platepar, the carried forward one (monitor_update_platepar) and the automatically
        fitted one (monitor_auto_calibration), each only if enabled. """

    # The platepars which are enabled
    config = _config()
    config.monitor_update_platepar = update
    config.monitor_auto_calibration = auto

    # The given, the carried forward and the automatically fitted platepar
    output_dir = str(tmp_path)
    paths = {'given': str(tmp_path/'given.cal'), 'latest': mnr.latestPlateparPath(output_dir, config),
             'auto': mnr.autoPlateparPath(output_dir, config)}

    # All exist, the newest one as given, the others older in this order
    for i, name in enumerate(['given', 'latest', 'auto']):
        open(paths[name], 'w').close()
        os.utime(paths[name], (1000 + i, 1000 + i))
    os.utime(paths[newest], (5000, 5000))

    assert mnr.selectPlatepar(output_dir, config, paths['given']) == paths[expected]


def test_select_platepar_without_any(tmp_path):
    """ Without a given platepar (missing or not given) and none fitted, there is no platepar. """

    config = _config()
    config.monitor_auto_calibration = True

    assert mnr.selectPlatepar(str(tmp_path), config, str(tmp_path/'missing.cal')) is None
    assert mnr.selectPlatepar(str(tmp_path), config, None) is None


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


def test_night_report_without_a_platepar(tmp_path):
    """ The files processed before the camera was calibrated automatically have no platepar. The night is still
        reported, the products which need the platepar are skipped, and the platepar of the night is the one of
        the latest file which has one. """

    # The products which need the platepar are enabled
    config = _config()
    config.monitor_fov_kml = True
    config.monitor_flux = True
    output_dir = str(tmp_path)

    # A night of one file without the calibration
    results_dir = _makeResults(output_dir, config, 'file1')
    for name in [config.platepar_name, config.platepars_recalibrated_name]:
        os.remove(os.path.join(output_dir, results_dir, name))

    night_state = mnr.generateNightReport(output_dir, NIGHT, config)
    night_dir = mnr.nightDirPath(output_dir, NIGHT, config)

    # Reported without failures, the products which need the platepar skipped, no platepar of the night
    assert 'archive' in night_state['ok_steps']
    assert night_state['failed_steps'] == []
    for step in ['fov_kml', 'flux', 'calibration_report', 'calibration_variation']:
        assert step not in night_state['ok_steps']
    assert not os.path.exists(os.path.join(night_dir, config.platepar_name))

    # A later file of the night was calibrated, its platepar is the one of the night
    _makeResults(output_dir, config, 'file2', start_s=600)
    mnr.generateNightReport(output_dir, NIGHT, config)
    assert os.path.isfile(os.path.join(night_dir, config.platepar_name))


def test_detected_stack_named_with_the_number_of_meteors(tmp_path):
    """ A meteor longer than a chunk is on several chunk images, which are all stacked, but the stack is
        named with the number of meteors and not with the number of images.
    """

    config = _config()
    output_dir = str(tmp_path)

    # One meteor from the first to the third chunk of 128 frames
    _makeResults(output_dir, config, 'file1', n_chunks=3, pick_frames=(3.0, 200.0, 300.0))

    night_state = mnr.generateNightReport(output_dir, NIGHT, config)
    assert 'archive' in night_state['ok_steps']

    night_files = os.listdir(mnr.nightDirPath(output_dir, NIGHT, config))
    assert NIGHT + '_stack_1_meteors.jpg' in night_files
    assert not any(f.startswith(NIGHT + '_stack_3_') for f in night_files)


def test_failed_optional_product_still_reports_the_night(tmp_path, monkeypatch):

    import Utils.ShowerAssociation

    config = _config()
    config.monitor_shower_association = True

    def _fail(*args, **kwargs):
        raise RuntimeError("no showers")

    monkeypatch.setattr(Utils.ShowerAssociation, 'showerAssociation', _fail)

    output_dir = str(tmp_path)
    results_dir = _makeResults(output_dir, config, 'file1')

    night_state = mnr.generateNightReport(output_dir, NIGHT, config, archive=True)

    assert 'shower_association' in night_state['failed_steps']
    assert night_state['files'] == [results_dir]

    # A report without the archive (e.g. from the command line) doesn't mark the night as reported
    mnr.updateReportState(output_dir, NIGHT, files=[])
    assert mnr.generateNightReport(output_dir, NIGHT, config, archive=False)['files'] == []


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
    assert mnr.unreportedResults(output_dir, states, NIGHT, mnr.scanNights(output_dir, config)[NIGHT])


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
                        lambda night_dir, config, ff_detected, **kwargs: stacks.append(config.thumb_stack))

    config = _config()
    config.thumb_stack = 5

    output_dir = str(tmp_path)
    _makeResults(output_dir, config, 'file1')

    mnr.generateNightReport(output_dir, NIGHT, config, archive=False)

    # The chunks have 128 frames, so twice as many are stacked as with 256-frame FF files, and the given
    #   config is not changed
    assert stacks == [10]
    assert config.thumb_stack == 5


def test_report_writes_the_observation_summary(tmp_path):
    """ With monitor_observation_summary, the night report writes the observation summary of the night. """

    config = _config()
    config.monitor_observation_summary = True
    output_dir = str(tmp_path)
    _makeResults(output_dir, config, 'file1')

    night_state = mnr.generateNightReport(output_dir, NIGHT, config, archive=False)

    assert 'observation_summary' in night_state['ok_steps']
    night_dir = mnr.nightDirPath(output_dir, NIGHT, config)
    summary_names = [f for f in os.listdir(night_dir) if f.endswith('observation_summary.json')]
    assert len(summary_names) == 1

    # The parameters filled in by the monitor are in the summary
    with open(os.path.join(night_dir, summary_names[0])) as f:
        summary = json.load(f)
    assert summary['camera_information'] == 'Processed from recorded data'
    assert 'detections_after_ml' in summary


def test_observation_summaries_of_the_nights_are_kept(tmp_path):
    """ The observation database keeps the summaries of the previous nights, which e.g. the days since the last
        detection are computed from. """

    import sqlite3

    import RMS.Formats.ObservationSummary as obs

    config = _config()
    config.monitor_observation_summary = True
    config.data_dir = str(tmp_path)
    output_dir = str(tmp_path)

    # The summary of an earlier night is in the database
    conn = obs.getObsDBConn(config)
    obs.storeDictInDB(conn, {'night_data_dir': 'XX0001_20251223_222000_000000', 'detections_after_ml': '3',
                             'time_last_detection': '2025-12-24 05:00:00'})
    conn.close()

    _makeResults(output_dir, config, 'file1')
    night_state = mnr.generateNightReport(output_dir, NIGHT, config, archive=False)
    assert 'observation_summary' in night_state['ok_steps']

    conn = sqlite3.connect(os.path.join(output_dir, 'observation.db'))
    nights = [row[0] for row in conn.execute("SELECT night_data_dir FROM {:s}".format(obs.OBSERVATIONS_TABLE_NAME))]
    conn.close()

    assert 'XX0001_20251223_222000_000000' in nights
    assert len(nights) == 2


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


def test_cleanup_frees_the_space_of_the_largest_recent_night(tmp_path, monkeypatch):

    config = _config()
    config.extra_space_gb = 0
    output_dir = str(tmp_path)

    # Four nights of different sizes, the oldest one also has an archive
    captured_dir = os.path.join(output_dir, config.captured_dir)
    archived_dir = os.path.join(output_dir, config.archived_dir)
    nights = ['XX0001_2025122{:d}_222000_000000'.format(i) for i in range(4)]
    for night_name, size in zip(nights, [5000, 1000, 3000, 2000]):
        os.makedirs(os.path.join(captured_dir, night_name))
        with open(os.path.join(captured_dir, night_name, 'FF_pair.png'), 'wb') as f:
            f.write(b'0'*size)
    os.makedirs(os.path.join(archived_dir, nights[0]))
    open(os.path.join(archived_dir, nights[0] + '_imgdata.tar.bz2'), 'w').close()

    monkeypatch.setattr(mnr, 'deleteOldObservations', lambda *args, **kwargs: True)

    # The free space grows with the deleted nights
    def _availableSpace(free_base):
        return lambda path: free_base + sum(size for night_name, size in zip(nights, [5000, 1000, 3000, 2000])
                                            if not os.path.isdir(os.path.join(captured_dir, night_name)))

    # The largest of the last three nights (3000) is needed, deleting the oldest night frees enough
    monkeypatch.setattr(mnr, 'availableSpace', _availableSpace(100))
    assert mnr.cleanupOldData(output_dir, config)
    assert sorted(os.listdir(captured_dir)) == nights[1:]
    assert os.listdir(archived_dir) == []

    # If other data takes the disk, so deleting the old nights wouldn't free enough space, nothing is deleted
    config.extra_space_gb = 1
    monkeypatch.setattr(mnr, 'availableSpace', lambda path: 100)
    assert not mnr.cleanupOldData(output_dir, config)
    assert sorted(os.listdir(captured_dir)) == nights[1:]

    # Otherwise the old nights are deleted, but never the latest one
    assert not mnr.freeSpace(output_dir, config, 3000)
    assert os.listdir(captured_dir) == [nights[-1]]


def test_done_flags_of_deleted_inputs_are_deleted(tmp_path):

    config = _config()
    output_dir = str(tmp_path/'out')
    input_dir = tmp_path/'in'
    input_dir.mkdir()
    night_dir = mnr.nightDirPath(output_dir, NIGHT, config)
    os.makedirs(night_dir)

    def _results(name, **info):
        results_path = os.path.join(output_dir, '2025', '202512', '20251225', name)
        os.makedirs(results_path)
        info.setdefault('input_file', str(input_dir/(name + '.vid')))
        mnr.writeDoneFlag(results_path, dict(night=NIGHT, **info))
        return results_path

    kept_input = _results('kept_input')
    (input_dir/'kept_input.vid').write_text('')
    gone_input = _results('gone_input')
    no_input_dir = _results('no_input_dir', input_file=str(tmp_path/'unmounted'/'x.vid'))
    old_flag = _results('old_flag', input_file=None)
    skipped = _results('skipped', skipped=True)

    # While the night exists, only the done flag of a skipped file whose input is gone is deleted
    assert mnr.deleteDoneFlags(output_dir, config) == 1
    assert not os.path.exists(skipped)

    # Once the night was deleted, the done flags of the deleted inputs are not needed anymore. Inputs which
    #   still exist, inputs on a directory which can't be found and old done flags without the input are kept
    shutil.rmtree(night_dir)
    assert mnr.deleteDoneFlags(output_dir, config) == 1
    assert not os.path.exists(gone_input)
    assert all(os.path.exists(path) for path in [kept_input, no_input_dir, old_flag])

    # Empty date directories are deleted too
    for path in [kept_input, no_input_dir, old_flag]:
        shutil.rmtree(path)
    (input_dir/'kept_input.vid').unlink()
    _results('last')
    assert mnr.deleteDoneFlags(output_dir, config) == 1
    assert os.listdir(output_dir) == [config.captured_dir]


def test_done_flags_of_older_versions_are_deleted_with_their_input(tmp_path):

    config = _config()
    output_dir = str(tmp_path/'out')
    input_dir = tmp_path/'in'
    input_dir.mkdir()

    # Results of an older version: an empty done flag, named after the input file without the extension
    def _oldResults(name):
        results_path = os.path.join(output_dir, '2026', '202608', '20260808', name)
        os.makedirs(results_path)
        open(os.path.join(results_path, mnr.DONE_FLAG_NAME), 'w').close()
        return results_path

    kept = _oldResults('dump_kept_02F')
    (input_dir/'dump_kept_02F.vid').write_text('')
    gone = _oldResults('dump_gone_02F')

    # Without the input directory they are kept
    assert mnr.deleteDoneFlags(output_dir, config) == 0

    # The results of the input files which are gone are deleted, with the empty date directories
    assert mnr.deleteDoneFlags(output_dir, config, input_dir=str(input_dir), input_ext='.vid') == 1
    assert os.path.exists(kept) and not os.path.exists(gone)

    # Nothing is deleted if the input directory can't be found (e.g. an unmounted disk)
    os.remove(str(input_dir/'dump_kept_02F.vid'))
    assert mnr.deleteDoneFlags(output_dir, config, input_dir=str(tmp_path/'unmounted'), input_ext='.vid') == 0
    assert mnr.deleteDoneFlags(output_dir, config, input_dir=str(input_dir), input_ext='.vid') == 1
    assert os.listdir(output_dir) == []


def test_done_flags_of_older_versions_find_inputs_in_subdirectories_and_any_case(tmp_path):

    config = _config()
    output_dir = str(tmp_path/'out')
    input_dir = tmp_path/'in'
    (input_dir/'2026-08-08').mkdir(parents=True)

    def _oldResults(name):
        results_path = os.path.join(output_dir, '2026', '202608', '20260808', name)
        os.makedirs(results_path)
        open(os.path.join(results_path, mnr.DONE_FLAG_NAME), 'w').close()
        return results_path

    # An input file with an upper case extension, and one in a subdirectory (--recursive)
    upper = _oldResults('21-00-00')
    (input_dir/'21-00-00.MKV').write_text('')
    nested = _oldResults('22-00-00')
    (input_dir/'2026-08-08'/'22-00-00.mkv').write_text('')

    # Both are found, only the results of the file in the subdirectory are deleted if not recursive
    assert mnr.deleteDoneFlags(output_dir, config, input_dir=str(input_dir), input_ext='.mkv',
                               recursive=True) == 0
    assert mnr.deleteDoneFlags(output_dir, config, input_dir=str(input_dir), input_ext='.mkv') == 1
    assert os.path.exists(upper) and not os.path.exists(nested)


def test_free_space_keeps_protected_nights(tmp_path, monkeypatch):

    config = _config()
    output_dir = str(tmp_path)
    captured_dir = os.path.join(output_dir, config.captured_dir)
    nights = ['XX0001_2025122{:d}_222000_000000'.format(i) for i in range(4)]
    for night_name in nights:
        os.makedirs(os.path.join(captured_dir, night_name))
        with open(os.path.join(captured_dir, night_name, 'FF_pair.png'), 'wb') as f:
            f.write(b'0'*1000)

    # The night being reported is not deleted, also when the space is needed
    monkeypatch.setattr(mnr, 'availableSpace', lambda path: 0)
    assert not mnr.freeSpace(output_dir, config, 1, protected_nights=[nights[1]])
    assert sorted(os.listdir(captured_dir)) == [nights[1], nights[-1]]


def test_delete_old_night_images(tmp_path):

    config = _config()
    config.monitor_delete_images_days = 2
    output_dir = str(tmp_path)

    results_dir = _makeResults(output_dir, config, 'file1')
    night_dir = mnr.nightDirPath(output_dir, NIGHT, config)

    reported_at = datetime.datetime(2025, 12, 25, 12, 0, 0)
    mnr.updateReportState(output_dir, NIGHT, files=[results_dir],
                          reported_at=reported_at.strftime(mnr.JSON_TIME_FORMAT))

    def _finishedBeforeReport(results_dir):
        finished = reported_at - datetime.timedelta(hours=1)
        finished = finished.replace(tzinfo=datetime.timezone.utc).timestamp()
        os.utime(os.path.join(output_dir, results_dir, mnr.DONE_FLAG_NAME), (finished, finished))

    _finishedBeforeReport(results_dir)

    def _deleteImages(days_later):
        return mnr.deleteOldNightImages(output_dir, config, mnr.readReportStates(output_dir),
                                        mnr.scanNights(output_dir, config),
                                        now=reported_at + datetime.timedelta(days=days_later))

    # Not old enough
    assert _deleteImages(1) == []

    # A late file of the night was not reported yet
    late_dir = _makeResults(output_dir, config, 'file2', start_s=600)
    _finishedBeforeReport(late_dir)
    assert _deleteImages(3) == []

    # Once everything is reported, only the image pairs are deleted
    mnr.updateReportState(output_dir, NIGHT, files=[results_dir, late_dir])
    assert _deleteImages(3) == [NIGHT]
    assert not [f for f in os.listdir(night_dir) if FFpng.isPairName(f)]

    # The night is not reported again, a report without the images would replace its products
    _makeResults(output_dir, config, 'file3', start_s=1200)
    assert not mnr.isPending(output_dir, config, mnr.readReportStates(output_dir), NIGHT,
                             mnr.scanNights(output_dir, config)[NIGHT])


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

    def terminate(self):
        self.alive = False
        self.exitcode = -15

    kill = terminate


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

    # The worker creates the night directory
    os.makedirs(mnr.nightDirPath(output_dir, night_name, config))

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

    # A failed report doesn't count, the partial report is made again (after the wait for failed reports)
    report_time = (partial_time + datetime.timedelta(minutes=5)).strftime(mnr.JSON_TIME_FORMAT)
    mnr.updateReportState(scheduling.output_dir, night, reported_at=report_time)
    reporter.states = mnr.readReportStates(scheduling.output_dir)
    assert _isDue(partial_time + datetime.timedelta(minutes=10)) == 'partial'

    # Only once per night: not after a successful report which was made after the partial report time
    mnr.updateReportState(scheduling.output_dir, night, files=[], succeeded_at=report_time)
    reporter.states = mnr.readReportStates(scheduling.output_dir)
    assert _isDue(partial_time + datetime.timedelta(minutes=10)) is None

    # Not for old nights, e.g. a backlog after a downtime, which get only the normal report
    mnr.updateReportState(scheduling.output_dir, night, reported_at=None, succeeded_at=None)
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


def test_night_deleted_by_the_cleanup_is_not_reported_again(scheduling):

    reporter = _reporter(scheduling, 'idle')
    night, _ = _nightAndResults(scheduling)
    now = scheduling.night_end + datetime.timedelta(hours=1)

    # E.g. the report state was lost, or a late file of an old night came in
    shutil.rmtree(mnr.nightDirPath(scheduling.output_dir, night, reporter.config))

    reporter.poll(True, now)
    assert reporter.pending == {} and reporter.active is None


def test_stuck_report_is_stopped(scheduling):

    reporter = _reporter(scheduling, 'idle')
    night, results = _nightAndResults(scheduling)
    now = scheduling.night_end + datetime.timedelta(hours=1)

    reporter._startReport(night, results, 'sunrise')
    proc = reporter.active[0]

    # Within the time limit the report keeps running
    reporter.poll(True, now)
    assert reporter.active is not None

    # Over the time limit it is stopped, counted as failed, and the reporter is free again
    proc.start_time -= mnr.REPORT_TIMEOUT + 1
    reporter.poll(True, now)
    assert reporter.active is None and (not proc.alive)
    assert reporter.failed[night]['count'] == 1
    assert reporter.report_lock.owner is None


def test_report_finished_during_shutdown_is_uploaded(scheduling):

    reporter = _reporter(scheduling, 'idle')
    night, results = _nightAndResults(scheduling)

    added = []

    class _UploadManager(object):
        def addFiles(self, file_list):
            added.append(file_list)
        def delayNextUpload(self, delay=0):
            pass

        class exit(object):
            @staticmethod
            def set():
                pass

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return False

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


def test_cleanup_runs_while_a_report_runs(scheduling):

    reporter = _reporter(scheduling, 'idle')
    night, _ = _nightAndResults(scheduling)
    now = scheduling.night_end + datetime.timedelta(days=1)

    reporter.poll(True, now=now)
    assert reporter.active is not None

    # A cleanup requested during the report (e.g. the disk is full) starts right away, and keeps the night of
    #   the report
    reporter.cleanup_due = True
    reporter.poll(True, now=now)
    assert (reporter.active is not None) and (reporter.cleanup_proc is not None)
    assert reporter.cleanup_proc.args[2] == [night]

    # No report starts while the cleanup runs
    reporter.active[0].alive = False
    reporter.poll(True, now=now)
    assert reporter.active is None
    reporter.poll(True, now=now)
    assert reporter.active is None


def test_process_which_fails_to_start_is_tried_again(scheduling, monkeypatch):

    reporter = mnr.NightReporter(scheduling.output_dir, scheduling.config_path, report_mode='idle')
    now = scheduling.night_end + datetime.timedelta(days=1)

    def _failingStart(self):
        raise OSError(12, "Cannot allocate memory")

    # The cleanup can't be started (e.g. no memory for the fork). The reporter keeps working
    monkeypatch.setattr(_Process, 'start', _failingStart)
    reporter.poll(True, now=now)
    assert (reporter.cleanup_proc is None) and (reporter.active is None)

    # It is tried again after a minute
    monkeypatch.setattr(_Process, 'start', lambda self: None)
    reporter.poll(True, now=now)
    assert reporter.cleanup_proc is None

    # The cleanup stays due until it was started
    reporter.start_retry_time = 0
    reporter.poll(True, now=now)
    assert reporter.cleanup_proc is not None


def test_scan_reads_only_changed_results(tmp_path, monkeypatch):

    config = _config()
    output_dir = str(tmp_path)
    results_dirs = [_makeResults(output_dir, config, name) for name in ['file1', 'file2']]

    reads = []
    read_done_flag = mnr.readDoneFlag
    monkeypatch.setattr(mnr, 'readDoneFlag', lambda path: reads.append(path) or read_done_flag(path))

    cache = {}
    assert sorted(mnr.scanNights(output_dir, config, cache=cache)[NIGHT]) == sorted(results_dirs)
    assert len(reads) == 2

    # Nothing changed
    del reads[:]
    assert sorted(mnr.scanNights(output_dir, config, cache=cache)[NIGHT]) == sorted(results_dirs)
    assert reads == []

    # A file processed again: its done flag is written again
    results_path = os.path.join(output_dir, results_dirs[0])
    mnr.writeDoneFlag(results_path, {'night': NIGHT, 'again': True})
    os.utime(results_path, ns=(0, os.stat(results_path).st_mtime_ns + 10**9))
    nights = mnr.scanNights(output_dir, config, cache=cache)
    assert len(reads) == 1
    assert nights[NIGHT][results_dirs[0]]['again']

    # The output directory of another monitor inside this one is not scanned
    nested = os.path.join(output_dir, 'other')
    os.makedirs(nested)
    open(os.path.join(nested, mnr.MONITOR_LOCK_FILE_NAME), 'w').close()
    _makeResults(nested, config, 'file3')
    assert sorted(mnr.scanNights(output_dir, config, cache=cache)[NIGHT]) == sorted(results_dirs)


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


def test_new_results_are_scanned_at_most_every_minute(scheduling, monkeypatch):

    reporter = _reporter(scheduling, 'idle')
    now = scheduling.night_end + datetime.timedelta(hours=1)

    scans = []
    scan_nights = mnr.scanNights
    monkeypatch.setattr(mnr, 'scanNights', lambda *args, **kwargs: scans.append(1) or scan_nights(*args, **kwargs))

    reporter.poll(False, now)
    assert len(scans) == 1

    # New results are scanned after the rescan interval, and no report starts before they are scanned
    reporter.resultsChanged()
    reporter.poll(True, now + datetime.timedelta(seconds=30))
    assert (len(scans) == 1) and (reporter.active is None)

    reporter.poll(True, now + datetime.timedelta(seconds=61))
    assert (len(scans) == 2) and (reporter.active is not None)


def _ignoreSigterm():
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(60)


def test_stopProcess_kills_a_process_which_ignores_sigterm():

    import multiprocessing

    proc = multiprocessing.Process(target=_ignoreSigterm)
    proc.start()
    time.sleep(0.5)

    t0 = time.time()
    assert mnr.stopProcess(proc, "Test", timeout=1)
    assert (not proc.is_alive()) and (time.time() - t0 < 5)


def test_results_of_deleted_nights_are_pruned(tmp_path):

    config = _config()
    output_dir = str(tmp_path)

    results_dir = _makeResults(output_dir, config, 'file1')
    nights = mnr.scanNights(output_dir, config)

    # Nothing is pruned while the night directory exists
    assert mnr.pruneResults(output_dir, config, nights) == 0

    # Once the cleanup deleted the night, only the done flag is kept, so the file stays processed, and the
    #   report state of the night is reduced to a mark
    mnr.updateReportState(output_dir, NIGHT, files=[results_dir])
    shutil.rmtree(mnr.nightDirPath(output_dir, NIGHT, config))
    assert mnr.pruneResults(output_dir, config, nights) == 1
    assert os.listdir(os.path.join(output_dir, results_dir)) == [mnr.DONE_FLAG_NAME]
    assert mnr.readReportStates(output_dir)['nights'] == {NIGHT: {'deleted': True}}

    # The pruned night is not pending, so it doesn't keep the following nights from being deleted
    assert not mnr.isPending(output_dir, config, mnr.readReportStates(output_dir), NIGHT,
                             mnr.scanNights(output_dir, config)[NIGHT])
    assert mnr.pruneResults(output_dir, config, nights) == 0

    # A late file of the deleted night creates its directory again, the night is still not reported (its
    #   archive would replace the full one with a single file)
    late_dir = _makeResults(output_dir, config, 'file2', start_s=600)
    assert not mnr.isPending(output_dir, config, mnr.readReportStates(output_dir), NIGHT,
                             mnr.scanNights(output_dir, config)[NIGHT])
    assert late_dir in mnr.scanNights(output_dir, config)[NIGHT]


@pytest.mark.parametrize('replace', [True, False])
def test_night_report_plots_sky_conditions(tmp_path, monkeypatch, replace):
    """ With the matched filter, the night report plots the sky conditions of the files of the night from the log
        of the conditions, with the files which were not searched as they had too few stars, and saves the
        conditions of the night. When the matched filter is a separate pass, a file with too few stars has no
        summary of it, and a summary of an earlier processing with the matched filter replacing the normal
        detection is not read.
    """

    import RMS.Routines.SkyQuality as sq
    from RMS.Routines.SkyQuality import CONDITION_COLUMNS

    config = _config()
    config.mf_enable = True
    config.mf_replace_detection = replace
    output_dir = str(tmp_path)

    # The arguments of the plot
    plotted = {}
    plot = sq.plotSkyConditions
    def _plot(rows, plot_path, title, **kwargs):
        plotted.update(kwargs, rows=rows)
        return plot(rows, plot_path, title, **kwargs)
    monkeypatch.setattr(sq, 'plotSkyConditions', _plot)

    # A searched file with conditions in the log, and a file of one chunk which was not searched (one star per
    #   chunk)
    searched = _makeResults(output_dir, config, 'file1', n_chunks=3)
    skipped = _makeResults(output_dir, config, 'file2', n_chunks=1, start_s=60)

    # A file which was not searched whose first chunk is half a chunk before the night
    night_start, _ = mnr.nightBoundsFromName(config, NIGHT)
    early = _makeResults(output_dir, config, 'file3', n_chunks=3,
                         start_s=(night_start - BEG_TIME).total_seconds() - 0.5*128/25.0)

    summary_dir = '' if replace else MATCHED_FILTER_DIR
    for results_dir, summary in [(searched, {'total_frames': 384, 'cloud_detections_removed': 2,
                                             'clear_sky_fraction': 0.67}),
                                 (skipped, {'max_stars': 1} if replace else None),
                                 (early, {'max_stars': 1} if replace else None)]:
        results_path = os.path.join(output_dir, results_dir)
        info = mnr.readDoneFlag(results_path)
        info['input_file'] = '/data/a, b/{:s}.vid'.format(os.path.basename(results_dir))
        mnr.writeDoneFlag(results_path, info)
        if summary is not None:
            os.makedirs(os.path.join(results_path, summary_dir), exist_ok=True)
            with open(os.path.join(results_path, summary_dir, DONE_NAME), 'w') as f:
                json.dump(summary, f)

        # A summary of an earlier processing in the other mode is not read
        if not replace:
            with open(os.path.join(results_path, DONE_NAME), 'w') as f:
                json.dump({'total_frames': 384, 'cloud_detections_removed': 5}, f)

    # The conditions of the searched file (logged by its path), of a file of another night with the same name, of
    #   another file, and of an earlier processing of the file which was not searched now
    with open(os.path.join(output_dir, 'sky_conditions_XX0001_{:s}.csv'.format(BEG_TIME.strftime('%Y%m%d'))),
              'w') as f:
        f.write(','.join(CONDITION_COLUMNS) + ',processed_utc\n')
        for i in range(3):
            t = BEG_TIME + datetime.timedelta(seconds=i*128/25.0)
            f.write('{:s},"/data/a, b/file1.vid",500,300,{:.3f},0.020,0.950,2025-12-26T00:00:00\n'.format(
                t.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3], 1.0 if i < 2 else 0.0))
        f.write('2025-12-25T00:00:00.000,other.vid,500,300,1.000,0.020,0.950,2025-12-26T00:00:00\n')

        # A row of an older log, which named the files by their names, is not taken for a file with rows by its
        #   path
        f.write('2025-12-25T03:00:00.000,file1.vid,500,300,0.000,0.020,0.950,2025-12-27T00:00:00\n')
        f.write('2025-12-25T03:01:00.000,"/data/a, b/file2.vid",500,300,1.000,0.020,0.950,2025-12-25T12:00:00\n')
    with open(os.path.join(output_dir, 'sky_conditions_XX0001_20251226.csv'), 'w') as f:
        f.write(','.join(CONDITION_COLUMNS) + ',processed_utc\n')
        f.write('2025-12-26T03:00:00.000,file1.vid,500,300,1.000,0.020,0.950,2025-12-27T00:00:00\n')

    night_state = mnr.generateNightReport(output_dir, NIGHT, config)
    assert 'sky_conditions' in night_state['ok_steps']

    night_dir = mnr.nightDirPath(output_dir, NIGHT, config)
    assert os.path.isfile(os.path.join(night_dir, NIGHT + '_sky_conditions.png'))

    # The conditions of the files of the night only
    with open(os.path.join(night_dir, NIGHT + '_sky_conditions.csv')) as f:
        lines = f.read().splitlines()
    assert lines[0] == ','.join(CONDITION_COLUMNS)
    assert len(lines) == 4 and all(line.startswith('2025-12-25T03:00') and ',"/data/a, b/file1.vid",' in line
                                   for line in lines[1:])

    # The files with too few stars are shown as not searched for the time of their chunks in the night (also the
    #   chunk which begins before it, the plot takes the time within the night), and the removed detections are
    #   counted
    assert len(plotted['not_searched']) == 2
    early_span, skipped_span = sorted(plotted['not_searched'], key=lambda span: span[0])
    start, end, times, stars = early_span
    assert abs((start - night_start).total_seconds() + 0.5*128/25.0) < 0.01 and stars == [1, 1, 1]
    assert plotted['bounds'][0] == night_start
    start, end, times, stars = skipped_span
    assert start == BEG_TIME + datetime.timedelta(seconds=60) and stars == [1]
    assert (end - start).total_seconds() == pytest.approx(128/25.0)
    assert plotted['removed_detections'] == 2


def test_night_sky_conditions_with_results_which_cant_be_read(tmp_path, monkeypatch):
    """ Results of a file which can't be read (a broken summary, stars or detections) leave the file out of the
        plot instead of failing it, the products of a previous report are removed when there are no conditions,
        and the detections outside the night are not shown.
    """

    import RMS.Routines.SkyQuality as sq

    config = _config()
    config.mf_enable = True
    config.mf_replace_detection = True
    output_dir = str(tmp_path)
    night_dir = mnr.nightDirPath(output_dir, NIGHT, config)

    results = {}
    for name, start_s in [('file1', 0), ('file2', 60), ('file3', 120)]:
        results_dir = _makeResults(output_dir, config, name, n_chunks=2, start_s=start_s)
        results[results_dir] = mnr.readDoneFlag(os.path.join(output_dir, results_dir))
    file1, file2, file3 = sorted(results)

    # A broken summary, and broken stars of a file which was not searched
    with open(os.path.join(output_dir, file1, DONE_NAME), 'w') as f:
        f.write('{"total_frames": ')
    with open(os.path.join(output_dir, file2, DONE_NAME), 'w') as f:
        json.dump({'max_stars': 1}, f)
    with open(glob.glob(os.path.join(output_dir, file2, 'CALSTARS_*.txt'))[0], 'w') as f:
        f.write('broken')

    # No conditions: the products of a previous report are removed
    for ext in ('csv', 'png'):
        open(os.path.join(night_dir, '{:s}_sky_conditions.{:s}'.format(NIGHT, ext)), 'w').close()
    broken = {file1: results[file1], file2: results[file2]}
    assert not mnr.plotNightSkyConditions(night_dir, NIGHT, output_dir, broken, config, '')
    assert not any(f.startswith(NIGHT + '_sky_conditions') for f in os.listdir(night_dir))

    # A file which was not searched, and detections: one in the night, one with no frame rate, one before the
    #   night
    with open(os.path.join(output_dir, file3, DONE_NAME), 'w') as f:
        json.dump({'max_stars': 1}, f)
    plotted = {}
    plot = sq.plotSkyConditions
    def _plot(rows, plot_path, title, **kwargs):
        plotted.update(kwargs)
        return plot(rows, plot_path, title, **kwargs)
    monkeypatch.setattr(sq, 'plotSkyConditions', _plot)
    picks = [[3.0, 10.0, 20.0, 170.0, 50.0, 60.0, 70.0, 1000, 2.5, 50, 8.0, 0]]
    ff_name = os.path.basename(glob.glob(os.path.join(night_dir, '*.png'))[0])
    early = 'FF_XX0001_20251224_120000_000_0000000.fits'
    FTPdetectinfo.writeFTPdetectinfo([[ff_name, 1, 10.0, 45.0, picks, 25.0], [ff_name, 2, 10.0, 45.0, picks, 0.0],
                                      [early, 3, 10.0, 45.0, picks, 25.0]], night_dir, 'FTPdetectinfo_test.txt',
                                     night_dir, 'XX0001', 25.0, celestial_coords_given=True)
    assert mnr.plotNightSkyConditions(night_dir, NIGHT, output_dir, results, config,
                                      os.path.join(night_dir, 'FTPdetectinfo_test.txt'))
    assert len(plotted['not_searched']) == 1
    assert len(plotted['detection_times']) == 1


def test_night_sky_conditions_file_not_searched_with_a_pause(tmp_path, monkeypatch):
    """ A file which was not searched and whose recording paused is not searched only for the time of its
        chunks, not during the pause.
    """

    import RMS.Routines.SkyQuality as sq

    config = _config()
    config.mf_enable = True
    config.mf_replace_detection = True
    output_dir = str(tmp_path)
    night_dir = mnr.nightDirPath(output_dir, NIGHT, config)

    # Two chunks, then one more after a pause of 10 minutes
    results_dir = _makeResults(output_dir, config, 'file1', n_chunks=2)
    results_path = os.path.join(output_dir, results_dir)
    star_list = CALSTARS.readCALSTARS(results_path, 'CALSTARS_file1.txt')[0]
    late = FFfile.constructFFName('XX0001', BEG_TIME + datetime.timedelta(minutes=10), frame=0, ext='fits')
    star_list.append([late, star_list[0][1]])
    CALSTARS.writeCALSTARS(star_list, results_path, 'CALSTARS_file1.txt', 'XX0001', 48, 64, chunk_frames=128,
                           fps=25.0)
    with open(os.path.join(results_path, DONE_NAME), 'w') as f:
        json.dump({'max_stars': 1}, f)

    plotted = {}
    plot = sq.plotSkyConditions
    def _plot(rows, plot_path, title, **kwargs):
        plotted.update(kwargs)
        return plot(rows, plot_path, title, **kwargs)
    monkeypatch.setattr(sq, 'plotSkyConditions', _plot)

    results = {results_dir: mnr.readDoneFlag(results_path)}
    assert mnr.plotNightSkyConditions(night_dir, NIGHT, output_dir, results, config, '')
    assert [len(span[2]) for span in plotted['not_searched']] == [2, 1]
    assert all((end - start).total_seconds() <= 2*128/25.0 + 0.01 for start, end, _, _ in plotted['not_searched'])


def test_night_report_removes_sky_conditions_without_the_matched_filter(tmp_path):
    """ The sky conditions of a previous report are removed when the matched filter is not enabled anymore. """

    config = _config()
    output_dir = str(tmp_path)
    _makeResults(output_dir, config, 'file1', n_chunks=2)
    night_dir = mnr.nightDirPath(output_dir, NIGHT, config)
    for ext in ('csv', 'png'):
        open(os.path.join(night_dir, '{:s}_sky_conditions.{:s}'.format(NIGHT, ext)), 'w').close()

    night_state = mnr.generateNightReport(output_dir, NIGHT, config, archive=False)
    assert 'sky_conditions' not in night_state['ok_steps']
    assert not any(f.startswith(NIGHT + '_sky_conditions') for f in os.listdir(night_dir))


def test_night_sky_conditions_counts_the_removed_detections_within_the_night(tmp_path, monkeypatch):
    """ Only the detections removed on clouded sky within the night are counted, from the times of their first
        frames, and all of them for a summary which has only their number. The time of the chunks of the files is
        from their stars.
    """

    import RMS.Routines.SkyQuality as sq

    config = _config()
    config.mf_enable = True
    config.mf_replace_detection = True
    output_dir = str(tmp_path)
    night_dir = mnr.nightDirPath(output_dir, NIGHT, config)
    night_start, _ = mnr.nightBoundsFromName(config, NIGHT)

    # A file which begins 10 s before the night, with a detection removed before the night, one in it and one whose
    #   time is not known (counted), and a file of an older processing with the number of the removed detections
    #   only
    early = _makeResults(output_dir, config, 'file1', n_chunks=8,
                         start_s=(night_start - BEG_TIME).total_seconds() - 10)
    older = _makeResults(output_dir, config, 'file2', n_chunks=2)
    removed_utc = [(night_start + datetime.timedelta(seconds=dt)).strftime('%Y-%m-%dT%H:%M:%S.%f')
                   for dt in (-9, 10)] + [None]
    for results_dir, summary in [(early, {'total_frames': 1024, 'clear_sky_fraction': 0.5,
                                          'cloud_detections_removed': 3, 'cloud_removed_utc': removed_utc}),
                                 (older, {'total_frames': 256, 'clear_sky_fraction': 0.5,
                                          'cloud_detections_removed': 3})]:
        with open(os.path.join(output_dir, results_dir, DONE_NAME), 'w') as f:
            json.dump(summary, f)

    plotted = {}
    plot = sq.plotSkyConditions
    def _plot(rows, plot_path, title, **kwargs):
        plotted.update(kwargs)
        return plot(rows, plot_path, title, **kwargs)
    monkeypatch.setattr(sq, 'plotSkyConditions', _plot)

    # A file which was not searched, so there is something to plot
    skipped = _makeResults(output_dir, config, 'file3', n_chunks=2, start_s=60)
    with open(os.path.join(output_dir, skipped, DONE_NAME), 'w') as f:
        json.dump({'max_stars': 1}, f)

    results = {}
    for results_dir in (early, older, skipped):
        results[results_dir] = mnr.readDoneFlag(os.path.join(output_dir, results_dir))
        results[results_dir]['input_file'] = '/data/{:s}.vid'.format(os.path.basename(results_dir))
    assert mnr.plotNightSkyConditions(night_dir, NIGHT, output_dir, results, config, '')
    assert plotted['removed_detections'] == 2 + 3

    # The time of the chunks of the files is from their stars
    assert plotted['chunk_seconds']['/data/file1.vid'] == pytest.approx(128/25.0)


def test_median_magnitude():
    """ The median magnitude is taken on the intensities: the two middle values of an even number are averaged as
        intensities, not as magnitudes, and the magnitudes which are not known are left out.
    """

    assert mnr.medianMagnitude([3.0, 1.0, 2.0]) == pytest.approx(2.0)
    assert mnr.medianMagnitude([1.0, 2.0, 3.0, 4.0]) \
        == pytest.approx(-2.5*np.log10((10**(-0.4*2.0) + 10**(-0.4*3.0))/2))
    assert mnr.medianMagnitude([1.0, 2.0, 3.0, 4.0]) < 2.5
    assert mnr.medianMagnitude([None, float('nan'), 5.0]) == pytest.approx(5.0)
    assert mnr.medianMagnitude([None, float('nan')]) is None


def test_magnitude_of_a_detection():
    """ The magnitude of a detection is the median magnitude of its calibrated measurements: the measurements of
        an uncalibrated detection have a placeholder magnitude, which is not taken.
    """

    def _meas(calibrated, mag):
        return [int(calibrated), 10.0, 100.0, 100.0, 170.0, 50.0, 60.0, 70.0, 1000, mag, 50, 8.0, 0]

    assert mnr.detectionMagnitude([_meas(True, 3.0), _meas(True, 5.0), _meas(True, 4.0)]) == pytest.approx(4.0)
    assert mnr.detectionMagnitude([_meas(False, 0.0), _meas(False, 0.0)]) is None
    assert mnr.detectionMagnitude([_meas(True, float('nan'))]) is None


@pytest.mark.parametrize('calibration, status', [(None, 0), ('Recalibrated with RMS on: 2025-12-25', 1)])
def test_calibration_status_of_the_detections(tmp_path, calibration, status):
    """ The calibration status of the measurements read from an FTPdetectinfo follows the calibration line of
        every detection: 0 for an uncalibrated one, 1 for a calibrated one.
    """

    picks = [[f, 10.0 + f, 20.0, 170.0, 50.0, 60.0, 70.0, 1000, 2.5, 50, 8.0, 0] for f in (3.0, 4.0)]
    FTPdetectinfo.writeFTPdetectinfo([['FF_XX0001_20251225_030000_000_0000000.fits', 1, 10.0, 45.0, picks, 25.0]],
        str(tmp_path), 'FTPdetectinfo_test.txt', str(tmp_path), 'XX0001', 25.0, calibration=calibration,
        celestial_coords_given=True)

    entries = FTPdetectinfo.readFTPdetectinfo(str(tmp_path), 'FTPdetectinfo_test.txt')
    assert [meas[0] for meas in entries[0][11]] == [status, status]


def test_conditions_until_the_end_of_the_input_at_the_beginning_of_the_night(tmp_path):
    """ The last chunk of an input which began before the night is read if the end of the input is in the night,
        also when the chunk itself ended before it.
    """

    from RMS.Routines.SkyQuality import CONDITION_COLUMNS

    config = _config()
    night_start = datetime.datetime(2025, 12, 25, 1, 0, 0)
    log_path = tmp_path/'sky_conditions_{:s}_20251225.csv'.format(config.stationID)
    with open(str(log_path), 'w') as f:
        f.write(','.join(CONDITION_COLUMNS) + ',processed_utc\n')
        for seconds in (-14, -10, -6):
            t = night_start + datetime.timedelta(seconds=seconds)
            f.write('{:s},/data/a.vid,500,300,1.000,0.020,0.950,2025-12-25T04:00:00\n'.format(
                t.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]))

    night_end = night_start + datetime.timedelta(hours=10)
    ends = {'/data/a.vid': night_start + datetime.timedelta(seconds=0.5)}
    rows = mnr.nightConditionRows(str(tmp_path), config, ['/data/a.vid'], {'/data/a.vid': 4.0}, night_start,
                                  night_end, input_ends=ends)
    assert [row['time'] for row in rows] == [night_start - datetime.timedelta(seconds=6)]
    assert mnr.nightConditionRows(str(tmp_path), config, ['/data/a.vid'], {'/data/a.vid': 4.0}, night_start,
                                  night_end) == []
