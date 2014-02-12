import numpy as np
import mahotas as mh
from mahotas.features.moments import moments

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


def test_normalize():
    A,B = np.meshgrid(np.arange(128),np.arange(128))
    for p0,p1 in [(1,1), (1,2), (2,1), (2,2)]:
        def f(im):
            return moments(im, p0, p1, cm=mh.center_of_mass(im), normalize=1)
        im = A+B
        fs = [f(im), f(im[::2]), f(im[:,::2]), f(im[::2, ::2])]
        assert np.var(fs) < np.mean(np.abs(fs))/10
