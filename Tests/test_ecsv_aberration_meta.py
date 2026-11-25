""" Unit tests for recording platepar.aberration in ECSV metadata headers. """

import datetime

from astropy.table import Table

from RMS.Formats.ECSV import writeECSV
from RMS.Formats.Platepar import Platepar


def _makePlatepar(aberration):
    """ Build a minimal Platepar with the requested aberration flag. """

    pp = Platepar()
    pp.lat = 43.19
    pp.lon = -81.32
    pp.elev = 329.0
    pp.X_res = 512
    pp.Y_res = 512
    pp.RA_d = 120.0
    pp.dec_d = 45.0
    pp.JD = 2461000.5
    pp.pos_angle_ref = 0.0
    pp.F_scale = 56.0
    pp.mag_lev_stddev = 0.1
    pp.aberration = aberration

    return pp


def _samplePick():
    """ Return one calibrated meteor pick dict for writeECSV. """

    return {
        'datetime': datetime.datetime(2026, 11, 25, 4, 0, 0, 123456),
        'ra': 120.1,
        'dec': 45.2,
        'azim': 180.0,
        'alt': 60.0,
        'x': 256.0,
        'y': 256.0,
        'intensity': 1000,
        'background': 50,
        'saturated': False,
        'mag': 3.5,
        'snr': 25.0,
    }


def test_write_ecsv_records_astrometry_aberration(tmp_path):
    """ writeECSV writes astrometry_aberration matching platepar.aberration as a YAML bool. """

    for flag in (False, True):
        pp = _makePlatepar(flag)
        fname = "aberr_{:d}.ecsv".format(int(flag))
        path = writeECSV(str(tmp_path), fname, "TEST01", pp, [_samplePick()], n_stars=42)

        table = Table.read(path, delimiter=',', format='ascii.ecsv', guess=False)
        assert table.meta['astrometry_aberration'] is flag


def test_write_ecsv_defaults_missing_aberration_to_false(tmp_path):
    """ Legacy Platepar instances without an aberration attribute write astrometry_aberration=False. """

    pp = _makePlatepar(False)
    delattr(pp, 'aberration')
    path = writeECSV(str(tmp_path), "legacy.ecsv", "TEST01", pp, [_samplePick()], n_stars=10)

    table = Table.read(path, delimiter=',', format='ascii.ecsv', guess=False)
    assert table.meta['astrometry_aberration'] is False
