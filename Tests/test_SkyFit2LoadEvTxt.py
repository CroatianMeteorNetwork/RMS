"""
Tests of the loading of the picks from a UWO event file (ev*.txt) in SkyFit2 (PlateTool.loadEvTxt), which the
ASTRA dialog of Manual Reduction offers (SELECT ECSV/TXT FILE).

The column names are in the commented header line above the first row of data. Every row of data is a pick,
the first one too. A file without a line above its data has no column names, and is reported as a format error.
"""

from types import SimpleNamespace

import pytest


# The header of an ev file as the UWO software writes it: metadata comments, then the column names
EV_HEADER = [
    "#",
    "#   version : test",
    "#    num_fr : 3",
    "#      unix : 1601098143.715272",
    "#      site : 01",
    "#    stream : A",
    "#",
    "#  fr       time    sum     seq          cx          cy         th         phi     lsp    mag  flag   bak    max",
]

# Three rows of data: frame, time, sum, seq, cx, cy, th, phi, lsp, mag, flag, bak, max
EV_ROWS = [
    "   32  -0.404544   2864  114407  383.156424  174.520245  46.851100   88.740210  -8.642   7.25  0000   0.0    0.0",
    "   33  -0.373425   4266  114408  387.808106  167.702448  47.036236   88.536686  -9.075   6.81  0000   0.0    0.0",
    "   35  -0.311189   5471  114410  397.082478  154.109345  47.406286   88.134738  -9.345   6.54  0000   0.0    0.0",
]


@pytest.fixture(scope="module")
def skyfit2():
    return pytest.importorskip("Utils.SkyFit2")


@pytest.fixture
def messages(skyfit2, monkeypatch):
    """ Record the SkyFit2 message boxes instead of showing them. """

    shown = []
    monkeypatch.setattr(skyfit2, "qmessagebox", lambda *a, **k: shown.append(k))

    return shown


def writeEv(tmp_path, lines):
    """ Write an ev file with the given lines.

    Arguments:
        tmp_path: [Path] Directory of the file.
        lines: [list] Lines of the file.

    Return:
        [str] Path to the file.
    """

    ev_path = tmp_path / "ev_20200926_052903A_01A.txt"
    ev_path.write_text("\n".join(lines) + "\n")

    return str(ev_path)


def testEvTxtLoadsEveryRow(skyfit2, messages, tmp_path):
    """ Every row of data of an ev file is a pick, the first one too. """

    ev_path = writeEv(tmp_path, EV_HEADER + EV_ROWS)

    pick_list = skyfit2.PlateTool.loadEvTxt(SimpleNamespace(), ev_path)

    # All frames are loaded with their centroids
    assert messages == []
    assert sorted(pick_list) == [32, 33, 35]
    assert pick_list[32]['x_centroid'] == pytest.approx(383.156424)
    assert pick_list[32]['y_centroid'] == pytest.approx(174.520245)
    assert pick_list[35]['x_centroid'] == pytest.approx(397.082478)


def testEvTxtWithoutHeaderReportsFormatError(skyfit2, messages, tmp_path):
    """ An ev file whose data starts on its first line has no column names: the format error is reported,
        instead of an error reading the column names before they are set. """

    ev_path = writeEv(tmp_path, EV_ROWS)

    pick_list = skyfit2.PlateTool.loadEvTxt(SimpleNamespace(), ev_path)

    # Nothing is loaded, and the message names the missing columns
    assert pick_list is None
    assert [message['title'] for message in messages] == ["TXT File Format Error"], messages
