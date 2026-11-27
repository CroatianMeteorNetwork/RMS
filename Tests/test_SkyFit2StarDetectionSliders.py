""" Headless tests that StarDetectionWidget.loadFromConfig displays config values without altering overrides. """

from __future__ import print_function, division, absolute_import

from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def qt_app():
    pg = pytest.importorskip("pyqtgraph")
    QtWidgets = pg.Qt.QtWidgets
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def testStarDetectionSlidersKeepConfigValues(qt_app):
    """ Loading a config with out-of-range or unrounded float values must not clamp or quantise overrides. """

    from RMS.Routines.CustomPyqtgraphClasses import StarDetectionWidget
    from Utils.SkyFit2 import PlateTool

    widget = StarDetectionWidget(SimpleNamespace(openHelpTopic=lambda topic: None))

    config = SimpleNamespace(
        bit_depth=8,
        intensity_threshold=25,
        neighborhood_size=57,        # above default slider max 40
        max_stars=8000,              # above default slider max 5000
        gamma=0.57999,               # int(0.57999*100) used to truncate to 57 -> 0.57
        segment_radius=25,           # above default slider max 20
        max_feature_ratio=0.7999,
        roundness_threshold=0.5599,
    )

    signal_calls = []
    widget.sigIntensityThresholdChanged.connect(lambda v: signal_calls.append(('intensity_threshold', v)))
    widget.sigNeighborhoodSizeChanged.connect(lambda v: signal_calls.append(('neighborhood_size', v)))
    widget.sigMaxStarsChanged.connect(lambda v: signal_calls.append(('max_stars', v)))
    widget.sigGammaChanged.connect(lambda v: signal_calls.append(('gamma', v)))
    widget.sigSegmentRadiusChanged.connect(lambda v: signal_calls.append(('segment_radius', v)))
    widget.sigMaxFeatureRatioChanged.connect(lambda v: signal_calls.append(('max_feature_ratio', v)))
    widget.sigRoundnessThresholdChanged.connect(lambda v: signal_calls.append(('roundness_threshold', v)))

    save_btn_updates = []

    class _StubTool(object):
        initStarDetectionOverrides = PlateTool.initStarDetectionOverrides

        def _updateConfigSaveButtonState(self):
            save_btn_updates.append(True)

    pt = _StubTool()
    pt.config = config
    pt.tab = SimpleNamespace(star_detection=widget)

    pt.initStarDetectionOverrides()

    # No slider change signal fired during load, and the save button state was refreshed
    assert signal_calls == []
    assert len(save_btn_updates) == 1

    # Overrides keep the exact config values
    assert pt.override_intensity_threshold == 25
    assert pt.override_neighborhood_size == 57
    assert pt.override_max_stars == 8000
    assert pt.override_gamma == pytest.approx(0.57999)
    assert pt.override_segment_radius == 25
    assert pt.override_max_feature_ratio == pytest.approx(0.7999)
    assert pt.override_roundness_threshold == pytest.approx(0.5599)

    # Sliders widened rather than clamping, and float sliders rounded instead of truncating
    assert widget.neighborhood_size_slider.value() == 57
    assert widget.max_stars_slider.value() == 8000
    assert widget.segment_radius_slider.value() == 25
    assert widget.gamma_slider.value() == 58
    assert widget.max_feature_ratio_slider.value() == 80
    assert widget.roundness_threshold_slider.value() == 56
    assert widget.slider_labels['gamma'].text() == '0.58'
    assert widget.slider_labels['max_stars'].text() == '8000'

    # Below-minimum values widen downward and 12-bit intensity thresholds keep headroom without clamping
    high_bit_config = SimpleNamespace(
        bit_depth=12,
        intensity_threshold=450,
        neighborhood_size=2,         # below default slider min 5
        max_stars=20,                # below default slider min 50
        gamma=0.35,                  # below default slider min 45
        segment_radius=1,            # below default slider min 2
        max_feature_ratio=0.30,      # below default slider min 50
        roundness_threshold=0.15,    # below default slider min 30
    )
    pt.config = high_bit_config
    pt.initStarDetectionOverrides()

    assert signal_calls == []
    assert len(save_btn_updates) == 2
    assert widget.intensity_threshold_slider.value() == 450
    assert widget.intensity_threshold_slider.maximum() >= 450
    assert widget.neighborhood_size_slider.value() == 2
    assert widget.max_stars_slider.value() == 20
    assert widget.gamma_slider.value() == 35
    assert widget.segment_radius_slider.value() == 1
    assert widget.max_feature_ratio_slider.value() == 30
    assert widget.roundness_threshold_slider.value() == 15
