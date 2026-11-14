""" Tests of the processing state of a captured night (RMS.Reprocess.nightProcessingState), for nights processed
    before the processing status record existed. """

from __future__ import print_function, division, absolute_import

import os

import pytest

import RMS.ConfigReader as cr
from RMS.Reprocess import NIGHT_INCOMPLETE, NIGHT_LEGACY_PROCESSED, nightProcessingState


REPO_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.config')
NIGHT = 'XX0001_20251225_010203_000000'


@pytest.fixture
def night(tmp_path):
    """ A captured night of a station with no processing status record, and a function which adds files to it. """

    config = cr.parse(REPO_CONFIG)
    config.data_dir = str(tmp_path)
    captured = os.path.join(config.data_dir, config.captured_dir, NIGHT)
    os.makedirs(captured)

    def _add(*names, archive=False):
        """ Create empty files in the captured night, and its archive directory if asked. """

        for name in names:
            open(os.path.join(captured, name), 'w').close()
        if archive:
            os.makedirs(os.path.join(config.data_dir, config.archived_dir, NIGHT), exist_ok=True)

    return config, captured, _add


def test_legacy_night_with_detections_is_processed(night):
    """ A night from before the status record with detections and no backups was processed. """

    config, captured, add = night
    add('FTPdetectinfo_{:s}.txt'.format(NIGHT))

    assert nightProcessingState(config, captured).state == NIGHT_LEGACY_PROCESSED


def test_legacy_night_with_backups_and_no_archive_is_incomplete(night):
    """ A night from before the status record which still has its detection backups, and was never archived (e.g.
        it crashed after writing the detections), is incomplete: it is resumed from the backups. """

    config, captured, add = night
    add('FTPdetectinfo_{:s}.txt'.format(NIGHT), 'rms_queue_bkup_1_abc.pickle')

    assert nightProcessingState(config, captured).state == NIGHT_INCOMPLETE


def test_legacy_night_with_backups_and_an_archive_is_processed(night):
    """ Leftover backups don't make a night incomplete when its archive shows it was finished (issue #418). """

    config, captured, add = night
    add('FTPdetectinfo_{:s}.txt'.format(NIGHT), 'rms_queue_bkup_1_abc.pickle', archive=True)

    assert nightProcessingState(config, captured).state == NIGHT_LEGACY_PROCESSED


def test_legacy_night_with_final_summary_is_processed(night):
    """ The final observation summary is written near the end of the processing, so the night was processed even
        with leftover backups. """

    config, captured, add = night
    add('{:s}_observation_summary.json'.format(NIGHT), 'rms_queue_bkup_1_abc.pickle')

    assert nightProcessingState(config, captured).state == NIGHT_LEGACY_PROCESSED
