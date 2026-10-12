""" The start of the observation session in the observation summary, also when a night is reprocessed. """

import json
import os

import RMS.ConfigReader as cr
import RMS.Formats.ObservationSummary as obs


REPO_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.config')


def _night(tmp_path):
    """ A config with its data in tmp_path, and a night directory of it. """

    config = cr.parse(REPO_CONFIG)
    config.data_dir = str(tmp_path)
    night_dir = os.path.join(str(tmp_path), config.captured_dir,
                             '{:s}_20261011_230000_000000'.format(config.stationID))
    os.makedirs(night_dir)

    return config, night_dir


def _summary(night_dir):
    """ The working observation summary of the night. """

    return obs.getObservationSummaryDict(night_dir)


def test_reprocessing_keeps_the_start_of_the_session(tmp_path, monkeypatch):
    """ When the night is reprocessed, the start time and the duration recorded by the capture must be kept, not
        replaced by the time of the reprocessing. """

    # The camera is not queried
    monkeypatch.setattr(obs, 'gatherCameraInformation', lambda config: ('sensor', 'firmware', 'date'))

    config, night_dir = _night(tmp_path)

    # The capture recorded the start of the session (hours before the reprocessing) and its duration
    obs.startObservationSummaryReport(config, night_dir, 36000)
    obs.addObsParam(_summary(night_dir), 'start_time', '2026-10-11T23:00:00+00:00')
    captured = _summary(night_dir)

    # The night is reprocessed later
    obs.startObservationSummaryReport(config, night_dir, None, reprocessing=True)
    reprocessed = _summary(night_dir)

    assert reprocessed['start_time'] == captured['start_time']
    assert reprocessed['duration_from_start_of_observation'] == '36000'


def test_start_without_an_earlier_session_records_it(tmp_path, monkeypatch):
    """ Without a recorded start (e.g. a night which was never captured on this station), the start is recorded. """

    monkeypatch.setattr(obs, 'gatherCameraInformation', lambda config: ('sensor', 'firmware', 'date'))

    config, night_dir = _night(tmp_path)
    obs.startObservationSummaryReport(config, night_dir, None, reprocessing=True)

    assert 'start_time' in _summary(night_dir)


def test_capture_records_a_new_start(tmp_path, monkeypatch):
    """ A capture (not a reprocessing) records its own start, e.g. continuous capture which has no duration. """

    monkeypatch.setattr(obs, 'gatherCameraInformation', lambda config: ('sensor', 'firmware', 'date'))

    config, night_dir = _night(tmp_path)
    obs.startObservationSummaryReport(config, night_dir, 36000)
    obs.addObsParam(_summary(night_dir), 'start_time', '2026-10-11T23:00:00+00:00')

    obs.startObservationSummaryReport(config, night_dir, None)

    assert _summary(night_dir)['start_time'] != '2026-10-11T23:00:00+00:00'


def test_days_since_last_detection_without_detection_columns(tmp_path, monkeypatch):
    """ With --nodetect on a fresh database, the columns of the detections were never created. There is no last
        detection to measure to, which is not an error. """

    monkeypatch.setattr(obs, 'gatherCameraInformation', lambda config: ('sensor', 'firmware', 'date'))

    config, night_dir = _night(tmp_path)
    obs.startObservationSummaryReport(config, night_dir, 36000)
    d = _summary(night_dir)
    obs.addObsParam(d, 'time_last_fits_file', '2026-10-12 05:00:00')
    obs.addObsParam(d, 'time_last_detection', None)

    # The values are stored before the query, as at the end of the observation (finalizeObservationSummary). The
    #   detections after the machine learning are never added without the detection (processNight)
    conn = obs.getObsDBConn(config)
    obs.storeDictInDB(conn, d)
    conn.close()

    assert obs.getDaysSinceLastDetection(config, night_dir, d=d) == "Unknown"


def test_summary_with_a_relative_night_directory(tmp_path, monkeypatch):
    """ Reprocess accepts a relative night directory. The working summary is written into it (not into a nested
        copy of the relative path), and read back from there. """

    monkeypatch.setattr(obs, 'gatherCameraInformation', lambda config: ('sensor', 'firmware', 'date'))

    config, night_dir = _night(tmp_path)
    monkeypatch.chdir(str(tmp_path))
    relative_dir = os.path.relpath(night_dir, str(tmp_path))

    obs.startObservationSummaryReport(config, relative_dir, 36000)
    obs.addObsParam(obs.getObservationSummaryDict(relative_dir), 'time_last_fits_file', '2026-10-12 05:00:00')

    working = [f for f in os.listdir(night_dir) if f.endswith('_working.json')]
    assert len(working) == 1
    assert obs.getObservationSummaryDict(relative_dir)['time_last_fits_file'] == '2026-10-12 05:00:00'


def test_reprocessing_a_finalized_night_keeps_its_start(tmp_path, monkeypatch):
    """ At the end of the observation the working summary is replaced by the final summary. Reprocessing the night
        later takes the start and the duration of the session from the final summary. """

    from RMS.Misc import getRMSStyleFileName

    monkeypatch.setattr(obs, 'gatherCameraInformation', lambda config: ('sensor', 'firmware', 'date'))

    config, night_dir = _night(tmp_path)
    final = {'start_time': '2026-10-11T23:00:00+00:00', 'duration_from_start_of_observation': '36000'}
    with open(getRMSStyleFileName(night_dir, obs.OBSERVATION_SUMMARY_NAME_JSON), 'w') as f:
        json.dump(final, f)

    obs.startObservationSummaryReport(config, night_dir, None, reprocessing=True)

    summary = _summary(night_dir)
    assert summary['start_time'] == final['start_time']
    assert summary['duration_from_start_of_observation'] == '36000'
