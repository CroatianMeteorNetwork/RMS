""" Tests of the parsing of the enum-like options in the [Capture] section of the config file (protocol,
media_backend, gst_colorspace, frame_file_type, frame_cleanup): a value written with other letter case or
surrounding spaces must select the same behaviour as the canonical value.
"""

from __future__ import print_function, division, absolute_import

import pytest

from RMS import ConfigReader as cr


def _parseCaptureOptions(monkeypatch, tmp_path, options):
    """ Parse a minimal [Capture] section with the given options into a default config.

    Arguments:
        monkeypatch: [pytest fixture] Used to skip the ffmpeg probe of the capture parser.
        tmp_path: [pathlib.Path] Directory used as the location of the config file.
        options: [dict] Option names and the values as they would be written in the config file.

    Return:
        config: [Config] Default config with the [Capture] section parsed.
    """

    # Saving the frames is off, so the parser does not need the frame directories
    parser = cr.RawConfigParser()
    parser.add_section('Capture')
    parser.set('Capture', 'save_frames', 'false')

    for option, value in options.items():
        parser.set('Capture', option, value)

    # The ffmpeg probe is not relevant to the parsing of these options
    config = cr.Config()
    config.config_file_path = str(tmp_path)
    monkeypatch.setattr(cr, 'isFfmpegWorking', lambda: False)
    cr.parseCapture(config, parser)

    return config


def testCaptureEnumOptionsAreNormalized(monkeypatch, tmp_path):
    """ Values with other letter case and surrounding spaces are read as the canonical values, which is what
        the capture code compares them with (e.g. 'UDP' must select UDP, not fall back to TCP).
    """

    config = _parseCaptureOptions(monkeypatch, tmp_path, {
        'protocol': ' UDP ',
        'media_backend': ' GST ',
        'gst_colorspace': ' gray8 ',
        'frame_file_type': ' PNG ',
        'frame_cleanup': ' Delete ',
    })

    assert config.protocol == 'udp'
    assert config.media_backend == 'gst'
    assert config.gst_colorspace == 'GRAY8'
    assert config.frame_file_type == 'png'
    assert config.frame_cleanup == 'delete'


@pytest.mark.parametrize('gst_colorspace', ['RGBx', 'xRGB', 'v210', 'r210'])
def testCapturePreservesCanonicalGstColorspaceCase(monkeypatch, tmp_path, gst_colorspace):
    """ GStreamer formats other than the ones RMS handles keep their case, which GStreamer requires, and only
        lose the surrounding spaces.
    """

    config = _parseCaptureOptions(monkeypatch, tmp_path, {
        'gst_colorspace': ' {} '.format(gst_colorspace),
    })

    assert config.gst_colorspace == gst_colorspace
