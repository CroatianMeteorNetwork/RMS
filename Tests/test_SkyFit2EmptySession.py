""" SkyFit2 started with no input has no image handle until something is loaded. """

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from Utils.SkyFit2 import PlateTool


class EmptySession(object):
    """ Stand-in for PlateTool with no data loaded and nothing else, so anything else that is used fails. """

    changeMode = PlateTool.changeMode

    def hasData(self):
        return False


def test_change_mode_without_data():
    """ The mode buttons do nothing with no data loaded. """

    EmptySession().changeMode('manualreduction')
    EmptySession().changeMode('skyfit')
