import numpy as np
from mahotas.moments import moments

def _slow(A, p0, p1, cm):
    c0,c1 = cm
    I,J = np.meshgrid(np.arange(A.shape[1],dtype=float), np.arange(A.shape[0], dtype=float))
    I -= c1
    J -= c0
    I **= p0
    J **= p1
    return (I * J * A).sum()

def test_smoke():
    assert moments(np.zeros((100,23)), 2, 2) == 0.0
    assert moments(np.ones((100,23)), 2, 2) != 0.0
    assert moments(np.ones((100,23)), 0, 0) == 100*23
    assert moments(np.ones((100,23)), 2, 2) != moments(np.ones((100,23)), 2, 2, cm=(50,12))

def test_against_slow():
    def perform(p0, p1, cm, A):
        assert moments(A, p0, p1, cm) == _slow(A, p0,p1,cm)

    A = (np.arange(2048) % 14).reshape((32, -1))
    yield perform, 2, 2, (22, 22), A
    yield perform, 2, 2, (20, 22), A
    yield perform, 2, 2, (0, 0), A
    yield perform, 1, 2, (0, 0), A
    yield perform, 1, 0, (0, 0), A

