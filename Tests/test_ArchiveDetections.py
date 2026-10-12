# Tests for ArchiveDetections

# Mark McIntyre, 2026-


import os
import pytest

from RMS.ArchiveDetections import selectFiles
import RMS.ConfigReader as cr


frDataDir = f'{os.path.dirname(__file__)}/ExampleStationData/Stations/FR0012'
ukDataDir = f'{os.path.dirname(__file__)}/ExampleStationData/Stations/UK0006'
plDataDir = f'{os.path.dirname(__file__)}/ExampleStationData/Stations/PL000P'


def checkDataExists():
    if not os.path.isdir(f'{frDataDir}'):
        return False
    return True


def test_selectFiles_fr():
    if not checkDataExists():
        pytest.skip("Test data not available - did you expand the archive?")
    ff_detected = ['FF_FR0012_20261011_233326_818_0542976.fits']
    cfg = cr.parse(f'{frDataDir}/.config')
    sel_files = selectFiles(cfg, frDataDir, ff_detected)
    assert ff_detected[0] in sel_files
    assert 'FR_FR0012_20261011_211813_064_0306688.bin' in sel_files
    assert 'mask.bmp' in sel_files


def test_selectFiles_uk():
    if not checkDataExists():
        pytest.skip("Test data not available - did you expand the archive?")
    ff_detected = ['FF_PL000P_20260927_004524_976_0651264.fits']
    cfg = cr.parse(f'{plDataDir}/.config')
    sel_files = selectFiles(cfg, plDataDir, ff_detected)
    assert ff_detected[0] in sel_files
    assert 'FR_PL000P_20260927_004555_719_0652032.bin' in sel_files
    assert 'CALSTARS_PL000P.txt' in sel_files
