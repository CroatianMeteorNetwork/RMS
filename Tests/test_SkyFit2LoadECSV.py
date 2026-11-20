"""
Test of the loading of the picks from an ECSV file in SkyFit2 (PlateTool.loadECSV).

The saturated_pixels column is a boolean, which saveECSV writes as True or False, so a loaded pick must be
saturated only where the column reads True.
"""

import datetime
from types import SimpleNamespace

import pytest


FRAME_TIME_0 = datetime.datetime(2026, 10, 21, 20, 15, 0)
FRAME_DT = datetime.timedelta(seconds=0.04)

# The header line and two rows as saveECSV writes them, the first pick not saturated and the second saturated
ECSV_LINES = [
    "# %ECSV 0.9",
    "# schema: astropy-2.0",
    "datetime,ra,dec,azimuth,altitude,x_image,y_image,integrated_pixel_value,background_pixel_value,"
    "saturated_pixels,mag_data,err_minus_mag,err_plus_mag,snr",
    "2026-10-21T20:15:00.040000,123.456789, +12.345678, 45.678901, +34.567890,  500.000,  300.000,"
    "      1000,        20,False,  +2.50, -0.10, +0.10,     25.00",
    "2026-10-21T20:15:00.080000,123.466789, +12.355678, 45.688901, +34.577890,  510.000,  305.000,"
    "     90000,        20,True ,  -1.50, -0.05, +0.05,    400.00",
]


@pytest.fixture(scope="module")
def loadECSV():
    skyfit2 = pytest.importorskip("Utils.SkyFit2")
    return skyfit2.PlateTool.loadECSV


def test_saturated_pixels_column_is_parsed_as_boolean(loadECSV, tmp_path):

    ecsv_path = tmp_path / "picks.ecsv"
    ecsv_path.write_text("\n".join(ECSV_LINES) + "\n")

    # A stand-in with only the frame times which loadECSV reads
    pt = SimpleNamespace(img_handle=SimpleNamespace(
        total_frames=5,
        currentFrameTime=lambda frame_no, dt_obj: FRAME_TIME_0 + frame_no*FRAME_DT
    ))

    pick_list = loadECSV(pt, str(ecsv_path))

    assert sorted(pick_list) == [1, 2]
    assert pick_list[1]['saturated'] is False
    assert pick_list[2]['saturated'] is True
