# tests for FRbin.py
# Note: 
# specific test cases for French data since these cameras are prefixed "FR"
# which can be confused with the "FR_" naming convention for fr-bin files

import numpy as np
import glob
import os

from RMS.Formats.FRbin import read, writeArray, validFRName

frDataDir = f'{os.path.dirname(__file__)}/ExampleStationData/Stations/FR0012'
ukDataDir = f'{os.path.dirname(__file__)}/ExampleStationData/Stations/UK0006'


def createSyntheticClips(srcdir, srcfile):
    fr = read(srcdir, srcfile)
    arr = []
    for i in range(fr.lines):
        frames = []
        sizepos = None
        for z in range(fr.frameNum[i]):
            try:
                sizepos = np.vstack([sizepos, [fr.yc[i][z], fr.xc[i][z], fr.t[i][z], fr.size[i][z]]])
            except Exception:
                sizepos = np.array([fr.yc[i][z], fr.xc[i][z], fr.t[i][z], fr.size[i][z]])
            frames.append(fr.frames[i][z])
        arr.append([frames, sizepos])
    return arr, fr

# arr,_ = createSyntheticClips('./ExampleStationData/Stations/UK0006', 'FR_UK0006_20261011_220419_944_0375808.bin')

def test_read_uk():
    fr = read(ukDataDir,'FR_UK0006_20261011_220419_944_0375808.bin')
    assert fr.lines == 1
    assert len(fr.frames) == 1
    assert len(fr.frames[0]) == 19


def test_read_fr():
    fr = read(frDataDir,'FR_FR0012_20261011_211813_064_0306688.bin')
    assert fr.lines == 1
    assert len(fr.frames) == 1
    assert len(fr.frames[0]) == 12


def test_writeArray_uk():
    clips=[]
    origname = 'UK0006_20261011_220419_944_0375808'
    fname = 'UKTEST_20261011_220419_944_0375808'
    clips, origdata = createSyntheticClips(ukDataDir, origname)
    writeArray(clips, ukDataDir, fname)
    newdata = read(ukDataDir, fname)
    assert newdata.yc == origdata.yc

    # check that the correct filename was written
    newFRname = os.path.join(ukDataDir, f'FR_{fname}.bin')
    frfiles = glob.glob(newFRname)
    assert os.path.split(frfiles[0])[1] == f'FR_{fname}.bin'
    os.remove(frfiles[0])


def test_writeArray_fr():
    clips=[]
    origname = 'FR0012_20261011_211813_064_0306688'
    fname = 'FRTEST_20261011_211813_064_0306688'
    clips, origdata = createSyntheticClips(frDataDir, origname)
    writeArray(clips, frDataDir, fname)
    newdata = read(frDataDir, fname)
    assert newdata.yc == origdata.yc
    assert len(newdata.frames) == len(origdata.frames)

    # check that the correct filename was written
    newFRname = os.path.join(frDataDir, f'FR_{fname}.bin')
    frfiles = glob.glob(newFRname)
    assert os.path.split(frfiles[0])[1] == f'FR_{fname}.bin'
    os.remove(frfiles[0])

def test_validFRName():
    assert validFRName('FR_FRtest.bin')
    assert validFRName('FR_UKtest.bin')
    assert validFRName('UKtest.bin') is False
    assert validFRName('FR_UKtest.fits') is False

    fr_list = [fr for fr in os.listdir(frDataDir) if validFRName(fr)]
    assert 'FR_FR0012_20261011_211813_064_0306688.bin' in fr_list
