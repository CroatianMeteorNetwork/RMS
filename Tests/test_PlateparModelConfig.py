"""
Tests of the model of a platepar fitted from scratch set in the config file (platepar_distortion_type and the flags
after it in the [Calibration] section): parsed from the config, used by AutoPlatepar unless a setting is given, and
set on a platepar with the coefficients moved to its layout.
"""

import logging
import types

import numpy as np
import pytest

try:
    from configparser import RawConfigParser
except ImportError:
    from ConfigParser import RawConfigParser

import RMS.ConfigReader as cr
from RMS.Astrometry import AutoPlatepar
from RMS.Astrometry.AutoPlatepar import applyPlateparModel, plateparModel
from RMS.Formats.Platepar import Platepar


def parseCalibration(text):
    """ A default config with the given [Calibration] options parsed. """

    config = cr.Config()
    parser = RawConfigParser()
    parser.read_string("[Calibration]\n" + text)
    cr.parseCalibration(config, parser)

    return config


def test_defaults_are_the_ones_of_autoplatepar():

    # Without the options, the config has the model AutoPlatepar falls back to
    config = cr.Config()
    assert plateparModel(config) == plateparModel(None)
    assert plateparModel(None) == {'distortion_type': AutoPlatepar.DEFAULT_DISTORTION_TYPE,
                                   'equal_aspect': AutoPlatepar.DEFAULT_EQUAL_ASPECT,
                                   'asymmetry_corr': AutoPlatepar.DEFAULT_ASYMMETRY_CORR,
                                   'decentering_corr': AutoPlatepar.DEFAULT_DECENTERING_CORR,
                                   'force_distortion_centre': AutoPlatepar.DEFAULT_FORCE_DISTORTION_CENTRE}

    # Every distortion type of a platepar is an option of the config
    assert list(cr.PLATEPAR_DISTORTION_TYPES) == Platepar().distortion_type_list


def test_options_are_parsed():

    config = parseCalibration("platepar_distortion_type: radial5-odd\n"
                              "platepar_equal_aspect: false\n"
                              "platepar_asymmetry_corr: false\n"
                              "platepar_decentering_corr: true\n"
                              "platepar_force_distortion_centre: true\n")

    assert plateparModel(config) == {'distortion_type': 'radial5-odd', 'equal_aspect': False,
                                     'asymmetry_corr': False, 'decentering_corr': True,
                                     'force_distortion_centre': True}


def test_unknown_distortion_type_keeps_the_default(caplog):

    with caplog.at_level(logging.WARNING, logger=cr.log.name):
        config = parseCalibration("platepar_distortion_type: radial7\n")

    assert config.platepar_distortion_type == cr.Config().platepar_distortion_type
    assert "platepar_distortion_type" in caplog.text


def test_given_settings_override_the_config():

    config = parseCalibration("platepar_distortion_type: radial5-odd\nplatepar_decentering_corr: true\n")

    # The given settings are kept, the others come from the config
    model = plateparModel(config, distortion_type='radial9-odd', equal_aspect=False)
    assert model == {'distortion_type': 'radial9-odd', 'equal_aspect': False, 'asymmetry_corr': True,
                     'decentering_corr': True, 'force_distortion_centre': False}

    # A setting given as False is not replaced by the config's True
    assert plateparModel(config, decentering_corr=False)['decentering_corr'] is False


def test_model_applied_to_a_platepar():

    # A radial5-odd platepar with some distortion
    pp = Platepar()
    pp.equal_aspect, pp.asymmetry_corr, pp.decentering_corr, pp.force_distortion_centre = True, False, False, False
    pp.setDistortionType("radial5-odd", reset_params=True)
    pp.x_poly_fwd = np.array([0.1, -0.08, -0.5, 0.003, -0.0001])
    pp.x_poly_rev = pp.x_poly_fwd.copy()
    k_fwd = pp.extractRadialCoeffs(pp.x_poly_fwd)

    model = plateparModel(None, distortion_type='radial7-odd', decentering_corr=True)
    applyPlateparModel(pp, model)

    # The platepar has the model: radial7-odd with the asymmetry and the decentering correction (centre 2,
    #   radial 3, asymmetry 2, decentering 4)
    assert pp.distortion_type == 'radial7-odd'
    assert pp.asymmetry_corr and pp.decentering_corr and pp.equal_aspect and not pp.force_distortion_centre
    assert len(pp.x_poly_fwd) == pp.poly_length == 2 + 3 + 2 + 4

    # The distortion centre and the radial coefficients of the old model are kept, the new ones are zero
    k_new = pp.extractRadialCoeffs(pp.x_poly_fwd)
    for name in ('x0', 'y0', 'k1', 'k2'):
        assert k_new[name] == pytest.approx(k_fwd[name])
    for name in ('k3', 'a1', 'a2', 'p1', 'p2', 'p3', 'p4'):
        assert k_new[name] == 0

    # Reset, the coefficients are the ones of a new platepar with the model
    applyPlateparModel(pp, model, reset_params=True)
    fresh = Platepar()
    applyPlateparModel(fresh, model, reset_params=True)
    assert np.array_equal(pp.x_poly_fwd, fresh.x_poly_fwd)
    assert np.array_equal(pp.y_poly_rev, fresh.y_poly_rev)
    assert not np.any(pp.x_poly_fwd[2:])


def test_positional_arguments_of_autofitplatepar_unchanged():

    # The new decentering setting is after the older arguments, which keep their positions
    try:
        from inspect import getfullargspec as argspec
    except ImportError:
        from inspect import getargspec as argspec
    args = argspec(AutoPlatepar.autoFitPlatepar).args
    assert args == ['dir_path', 'config', 'catalog_stars', 'platepar_template', 'fov_w_hint', 'ff_name',
                    'distortion_type', 'equal_aspect', 'asymmetry_corr', 'force_distortion_centre', 'refraction',
                    'photometric_sigma', 'fwhm_mult', 'wide_fov_search', 'final_catalog_stars', 'verbose',
                    'decentering_corr']


def test_skyfit2_restores_the_model_of_the_config():

    pytest.importorskip("pyqtgraph")
    from RMS.Routines.CustomPyqtgraphClasses import PlateparParameterManager

    config = parseCalibration("platepar_distortion_type: radial9-odd\nplatepar_decentering_corr: true\n")
    manager = types.SimpleNamespace(gui=types.SimpleNamespace(config=config))

    model = PlateparParameterManager.defaultModel(manager)
    assert model['distortion_type'] == 'radial9-odd'
    assert model['decentering_corr'] is True
