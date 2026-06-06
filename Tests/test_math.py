import pytest

np = pytest.importorskip("numpy")

from RMS.Math import angularSeparationVect


def test_angular_separation_vect_handles_non_unit_vectors():
    vect1 = np.array([2.0, 0.0, 0.0])
    vect2 = np.array([0.0, 3.0, 0.0])

    angle = angularSeparationVect(vect1, vect2)

    assert np.isclose(angle, np.pi / 2)
