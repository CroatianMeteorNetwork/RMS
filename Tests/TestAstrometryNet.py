import datetime
import inspect
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

import RMS.Astrometry.AstrometryNet as AstrometryNet
import RMS.Astrometry.AstrometryNetNova as AstrometryNetNova
import RMS.Astrometry.AutoPlatepar as AutoPlatepar


class _NoMatchSolution(object):
    def has_match(self):
        return False


class _NewSolver(object):
    call = None

    def __init__(self, index_files):
        pass

    def solve(self, stars, size_hint, position_hint, solution_parameters):
        _NewSolver.call = {
            'stars': stars,
            'position_hint': position_hint,
        }
        return _NoMatchSolution()


class _LegacySolver(object):
    call = None

    def __init__(self, index_files):
        pass

    def solve(self, stars_xs, stars_ys, size_hint, position_hint, solution_parameters):
        _LegacySolver.call = {
            'stars_xs': stars_xs,
            'stars_ys': stars_ys,
            'position_hint': position_hint,
        }
        return _NoMatchSolution()


def _fakeAstrometry(solver_class):
    return SimpleNamespace(
        Solver=solver_class,
        PositionHint=lambda **kwargs: SimpleNamespace(**kwargs),
        SolutionParameters=lambda **kwargs: SimpleNamespace(**kwargs),
        Action=SimpleNamespace(STOP='stop', CONTINUE='continue'),
        series_4100=SimpleNamespace(index_files=lambda **kwargs: []),
    )


@pytest.mark.parametrize('solver_class', [_NewSolver, _LegacySolver])
def test_local_solver_forwards_position_hint_for_supported_apis(monkeypatch, solver_class):
    """ Position hints reach both the current and legacy astrometry solver APIs. """

    solver_class.call = None
    monkeypatch.setattr(AstrometryNet, 'astrometry', _fakeAstrometry(solver_class))

    result = AstrometryNet.astrometryNetSolveLocal(
        x_data=np.array([10.0, 20.0, 30.0]),
        y_data=np.array([15.0, 25.0, 35.0]),
        position_hint=(123, 45, 20),
    )

    assert result is None
    assert solver_class.call is not None
    hint = solver_class.call['position_hint']
    assert (hint.ra_deg, hint.dec_deg, hint.radius_deg) == (123.0, 45.0, 20.0)


def test_local_solver_preserves_blind_solve_default(monkeypatch):
    """ Omitting a position hint keeps the original blind-solve behavior. """

    _NewSolver.call = None
    monkeypatch.setattr(AstrometryNet, 'astrometry', _fakeAstrometry(_NewSolver))

    result = AstrometryNet.astrometryNetSolveLocal(
        x_data=np.array([10.0, 20.0, 30.0]),
        y_data=np.array([15.0, 25.0, 35.0]),
    )

    assert result is None
    assert _NewSolver.call['position_hint'] is None


def test_remote_coordinate_and_image_fallbacks_preserve_position_hint(monkeypatch):
    """ A failed coordinate solve keeps the hint when retrying with the image. """

    calls = []

    def fakeNovaSolve(**kwargs):
        calls.append(kwargs)
        return None if len(calls) == 1 else 'remote solution'

    monkeypatch.setattr(AstrometryNet, 'ASTROMETRY_NET_AVAILABLE', False)
    monkeypatch.setattr(AstrometryNet, 'novaAstrometryNetSolve', fakeNovaSolve)

    position_hint = (123.0, 45.0, 20.0)
    result = AstrometryNet.astrometryNetSolve(
        img=np.zeros((10, 10), dtype=np.uint8),
        x_data=np.array([1.0, 2.0]),
        y_data=np.array([3.0, 4.0]),
        position_hint=position_hint,
    )

    assert result == 'remote solution'
    assert [call['position_hint'] for call in calls] == [position_hint, position_hint]
    assert calls[0]['img'] is None
    assert calls[1]['img'] is not None


def test_local_exception_remote_fallback_preserves_position_hint(monkeypatch):
    """ A local solver error does not turn the remote retry into a blind solve. """

    calls = []

    def failLocalSolve(**kwargs):
        raise RuntimeError('local failure')

    def fakeNovaSolve(**kwargs):
        calls.append(kwargs)
        return 'remote solution'

    monkeypatch.setattr(AstrometryNet, 'ASTROMETRY_NET_AVAILABLE', True)
    monkeypatch.setattr(AstrometryNet, 'astrometryNetSolveLocal', failLocalSolve)
    monkeypatch.setattr(AstrometryNet, 'novaAstrometryNetSolve', fakeNovaSolve)

    position_hint = (123.0, 45.0, 20.0)
    result = AstrometryNet.astrometryNetSolve(
        x_data=np.array([1.0, 2.0]),
        y_data=np.array([3.0, 4.0]),
        position_hint=position_hint,
    )

    assert result == 'remote solution'
    assert calls[0]['position_hint'] == position_hint


def test_remote_solver_translates_position_hint_to_upload_fields(monkeypatch):
    """ Remote submissions use the positional fields understood by Astrometry.net. """

    class FakeClient(object):
        upload_args = None

        def __init__(self, apiurl=None):
            pass

        def login(self, api_key):
            pass

        def upload(self, **kwargs):
            FakeClient.upload_args = kwargs
            return {'status': 'failure'}

    monkeypatch.setattr(AstrometryNetNova, 'Client', FakeClient)

    result = AstrometryNetNova.novaAstrometryNetSolve(
        x_data=[1.0, 2.0],
        y_data=[3.0, 4.0],
        api_url='https://example.invalid/api/',
        position_hint=(123, 45, 20),
    )

    assert result is False
    assert FakeClient.upload_args['center_ra'] == 123.0
    assert FakeClient.upload_args['center_dec'] == 45.0
    assert FakeClient.upload_args['radius'] == 20.0


def test_auto_fit_preserves_api_order_and_wide_retry_hint(monkeypatch):
    """ Existing positional slots stay fixed and the wide retry keeps its position hint. """

    parameters = list(inspect.signature(AutoPlatepar.autoFitPlatepar).parameters)
    assert parameters[-4:] == [
        'wide_fov_search', 'final_catalog_stars', 'verbose', 'position_hint']

    ff_name = 'FF_test.bin'
    star_data = np.array([
        [float(i), float(i + 1), 100.0, 1.0, 2.5, 0.0, 5.0, 0.0]
        for i in range(12)
    ])

    monkeypatch.setattr(AutoPlatepar.os, 'listdir', lambda path: ['CALSTARS_test.txt'])
    monkeypatch.setattr(AutoPlatepar.CALSTARS, 'readCALSTARS',
                        lambda path, name: ([(ff_name, star_data)], None))
    monkeypatch.setattr(AutoPlatepar, 'getMaskFile', lambda path, config: None)
    monkeypatch.setattr(AutoPlatepar, 'filenameToDatetime',
                        lambda name: datetime.datetime(2026, 1, 1))
    monkeypatch.setattr(AutoPlatepar, 'date2JD', lambda *args: 2460000.0)
    monkeypatch.setattr(AutoPlatepar, 'JD2HourAngle', lambda jd: 123.0)

    class FakePlatepar(object):
        def __init__(self):
            self.lat = 0.0
            self.lon = 0.0
            self.elev = 0.0
            self.X_res = 0
            self.Y_res = 0
            self.station_code = ''

        def addVignettingCoeff(self, use_flat=False):
            pass

    monkeypatch.setattr(AutoPlatepar, 'Platepar', FakePlatepar)

    solver_hints = []
    monkeypatch.setattr(AutoPlatepar, 'astrometryNetSolve',
                        lambda **kwargs: solver_hints.append(kwargs['position_hint']))

    recursive_calls = []
    retry_result = object()

    def fakeAutoFit(*args, **kwargs):
        recursive_calls.append(kwargs)
        return retry_result

    original_auto_fit = AutoPlatepar.autoFitPlatepar
    monkeypatch.setattr(AutoPlatepar, 'autoFitPlatepar', fakeAutoFit)

    config = SimpleNamespace(
        latitude=45.0,
        longitude=16.0,
        elevation=100.0,
        width=1920,
        height=1080,
        stationID='TEST',
        fov_w=75.0,
    )
    position_hint = (123.0, 45.0, 20.0)

    result = original_auto_fit(
        '/unused', config, np.empty((0, 3)), ff_name=ff_name,
        verbose=False, position_hint=position_hint)

    assert result is retry_result
    assert solver_hints == [position_hint]
    assert len(recursive_calls) == 1
    assert recursive_calls[0]['wide_fov_search'] is True
    assert recursive_calls[0]['position_hint'] == position_hint
