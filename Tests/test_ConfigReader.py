"""Focused tests for capture option parsing."""

import pytest

from RMS import ConfigReader as cr


def _parseCaptureOptions(monkeypatch, tmp_path, options):
    """Parse a minimal Capture section with the supplied options."""
    parser = cr.RawConfigParser()
    parser.add_section('Capture')
    parser.set('Capture', 'save_frames', 'false')

    for option, value in options.items():
        parser.set('Capture', option, value)

    config = cr.Config()
    config.config_file_path = str(tmp_path)
    monkeypatch.setattr(cr, 'isFfmpegWorking', lambda: False)
    cr.parseCapture(config, parser)

    return config


def testCaptureEnumOptionsAreNormalized(monkeypatch, tmp_path):
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
    config = _parseCaptureOptions(monkeypatch, tmp_path, {
        'gst_colorspace': ' {} '.format(gst_colorspace),
    })

    assert config.gst_colorspace == gst_colorspace
