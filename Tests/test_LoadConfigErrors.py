""" The errors of a config file which can't be loaded say which file it is, and what is wrong with it. """

import pytest

import RMS.ConfigReader as cr


def writeConfig(tmp_path, text):
    path = tmp_path/"station.config"
    path.write_text(text)

    return str(path)


def test_config_is_loaded(tmp_path):
    """ A config with the station ID in the [System] section is loaded. """

    config = cr.loadConfigFromDirectory([writeConfig(tmp_path, "[System]\nstationID: AB0001\n")], str(tmp_path))

    assert config.stationID == "AB0001"


def test_missing_file(tmp_path):
    """ A config file which doesn't exist is reported as such, not as not configured. """

    path = str(tmp_path/"nothing_here.config")

    with pytest.raises(FileNotFoundError, match="nothing_here.config"):
        cr.loadConfigFromDirectory([path], str(tmp_path))


@pytest.mark.parametrize("text", ["", "[Capture]\nfps: 25\n"])
def test_no_system_section(tmp_path, text):
    """ A file without the [System] section says so, and which file it is. """

    path = writeConfig(tmp_path, text)

    with pytest.raises(RuntimeError, match=r"no \[System\] section.*station.config"):
        cr.loadConfigFromDirectory([path], str(tmp_path))


def test_no_station_id(tmp_path):
    """ A [System] section without the station ID says so, and which file it is. """

    path = writeConfig(tmp_path, "[System]\nlatitude: 45.0\n")

    with pytest.raises(RuntimeError, match="no stationID.*station.config"):
        cr.loadConfigFromDirectory([path], str(tmp_path))
