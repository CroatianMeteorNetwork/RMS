""" Unit tests for blank option values in RMS.ConfigReader. """

from __future__ import print_function, division, absolute_import

import os
import re

import pytest

import RMS.ConfigReader as cr


RMS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize('option', [
    'udp_buffer_size', 'video_scale_width', 'video_scale_height',
    'auto_reprocess_max_attempts', 'ml_model_file'
])
def testEmptyConfigValueKeepsDefault(tmp_path, option):
    """ An empty value ("option:" with nothing after it) must keep the default instead of failing to parse. """

    with open(os.path.join(RMS_ROOT, '.config')) as f:
        text = f.read()

    # Blank the option where it is set, or uncomment and blank the commented example
    pattern = re.compile(r'^(;\s*)?{:s}\s*[:=].*$'.format(re.escape(option)), re.MULTILINE)
    assert pattern.search(text), option
    text = pattern.sub('{:s}:'.format(option), text, count=1)

    config_path = str(tmp_path/'.config')
    with open(config_path, 'w') as f:
        f.write(text)

    config = cr.parse(config_path)
    default = cr.Config()

    assert getattr(config, option) == getattr(default, option)
    if option == 'ml_model_file':
        assert config.ml_model_path == default.ml_model_path
