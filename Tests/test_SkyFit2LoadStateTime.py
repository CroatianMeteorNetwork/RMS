""" The beginning time of a video when a state is loaded from a running session. """

import datetime
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest

from pyqtgraph.Qt import QtWidgets

from RMS.Formats.FrameInterface import beginningTimeFromVideoName
from Utils.SkyFit2 import PlateTool


SESSION_TIME = datetime.datetime(2025, 9, 13, 22, 24, 50)


class Station(object):

    def __init__(self, station_id):
        self.stationID = station_id


class Session(object):
    """ Stand-in for a PlateTool with a video open. """

    findDataFileOfState = PlateTool.findDataFileOfState
    sameInputAsSession = PlateTool.sameInputAsSession
    beginningTimeOfState = PlateTool.beginningTimeOfState

    def __init__(self):
        self.input_path = "/here/bolido.mp4"
        self.beginning_time = SESSION_TIME
        self.config = Station("BB0001")


def answers(monkeypatch, *replies):
    """ Answer the dialog with the given replies, and return the list of the warnings shown. """

    replies = list(replies)
    warnings = []

    def getText(*args, **kwargs):
        reply = replies.pop(0)
        return (reply, bool(reply))

    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", staticmethod(getText))
    monkeypatch.setattr("Utils.SkyFit2.qmessagebox", lambda **k: warnings.append(k))

    return warnings


@pytest.mark.parametrize("name, expected", [
    ("20250913_222450.mp4", datetime.datetime(2025, 9, 13, 22, 24, 50)),
    ("20250913-222450.5.mp4", datetime.datetime(2025, 9, 13, 22, 24, 50, 500000)),
    ("CAWE01_20240611_020430_954145_video.mkv", datetime.datetime(2024, 6, 11, 2, 4, 30, 954145)),
    ("bolido.mp4", None),
    ("192.168.0.108_ch6_20250913192440_20250913192534_BOLIDO_shape.avi", None),
    ("Meteor_30_JAN_26_11_26_PM.mp4", None),
])
def test_time_from_video_name(name, expected):
    assert beginningTimeFromVideoName(name) == expected


def test_same_video_takes_the_time_of_the_session():
    """ A state of the video open in the session, from the same station, needs no question. """

    state = {"input_path": "/there/bolido.mp4", "config": Station("BB0001")}

    assert Session().beginningTimeOfState(state) == SESSION_TIME


def test_same_name_of_another_station_is_asked(monkeypatch):
    """ The same file name from another camera is another video, it doesn't take the time of the session. """

    answers(monkeypatch, "20250913_222500")
    state = {"input_path": "/there/bolido.mp4", "config": Station("WI0001")}

    assert Session().beginningTimeOfState(state) == datetime.datetime(2025, 9, 13, 22, 25, 0)


def test_another_video_is_asked_again_after_a_wrong_answer(monkeypatch):

    warnings = answers(monkeypatch, "nonsense", "20250913_222500")
    state = {"input_path": "/there/other.mp4", "config": Station("BB0001")}

    assert Session().beginningTimeOfState(state) == datetime.datetime(2025, 9, 13, 22, 25, 0)
    assert len(warnings) == 1


def test_cancelling(monkeypatch):
    """ Cancelling is told apart from a time which isn't needed. """

    answers(monkeypatch, "")
    state = {"input_path": "/there/other.mp4", "config": Station("BB0001")}

    assert Session().beginningTimeOfState(state) is False


@pytest.mark.parametrize("state", [
    {"input_path": "/there/other.mp4", "beginning_time": SESSION_TIME},
    {"input_path": "/there/20250913_222500.mp4"},
    {"input_path": "/there/FF_folder"},
    {},
])
def test_not_needed(state):
    """ The time is in the state or in the name, or the input isn't a video: nothing is asked. """

    assert Session().beginningTimeOfState(state) is None


def fileDialog(monkeypatch, reply):

    asked = []

    def getOpenFileName(*args, **kwargs):
        asked.append(args)
        return (reply, '')

    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", staticmethod(getOpenFileName))

    return asked


def test_data_file_next_to_the_state_comes_first(tmp_path, monkeypatch):

    asked = fileDialog(monkeypatch, "")
    (tmp_path/"video.mp4").write_text("")

    found = Session().findDataFileOfState(str(tmp_path), "/other_machine/video.mp4", "/here/video.mp4")

    assert found == str(tmp_path/"video.mp4")
    assert not asked


def test_data_file_of_the_session_when_the_saved_path_does_not_exist(tmp_path, monkeypatch):

    asked = fileDialog(monkeypatch, "")
    session_file = tmp_path/"video.mp4"
    session_file.write_text("")
    state_folder = tmp_path/"state"
    state_folder.mkdir()

    found = Session().findDataFileOfState(str(state_folder), "/other_machine/video.mp4", str(session_file))

    assert found == str(session_file)
    assert not asked


def test_data_file_is_asked_for_and_cancelling_gives_none(tmp_path, monkeypatch):

    asked = fileDialog(monkeypatch, "")

    assert Session().findDataFileOfState(str(tmp_path), "/other_machine/video.mp4") is None
    assert len(asked) == 1
