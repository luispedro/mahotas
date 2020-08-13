import numpy as np
import mahotas as mh
from mahotas.features.moments import moments
import pytest

def _slow(A, p0, p1, cm):
    c0,c1 = cm
    I,J = np.meshgrid(np.arange(A.shape[0],dtype=float), np.arange(A.shape[1], dtype=float))
    I -= c0
    J -= c1
    I **= p0
    J **= p1
    return (I.T * J.T * A).sum()

def test_smoke():
    assert moments(np.zeros((100,23)), 2, 2) == 0.0
    assert moments(np.ones((100,23)), 2, 2) != 0.0
    assert moments(np.ones((100,23)), 0, 0) == 100*23
    assert moments(np.ones((100,23)), 2, 2) != moments(np.ones((100,23)), 2, 2, cm=(50,12))

@pytest.mark.parametrize('p0, p1, cm', [
     (2, 2, (22, 22)),
     (2, 2, (20, 22)),
     (2, 2, (0, 0)),
     (1, 2, (0, 0)),
     (1, 0, (0, 0))])
def test_against_slow(p0, p1, cm):
    A = (np.arange(2048) % 14).reshape((32, -1))
    assert moments(A, p0, p1, cm) == _slow(A, p0,p1,cm)


def test_normalize():
    A,B = np.meshgrid(np.arange(128),np.arange(128))
    for p0,p1 in [(1,1), (1,2), (2,1), (2,2)]:
        def f(im):
            return moments(im, p0, p1, cm=mh.center_of_mass(im), normalize=1)
        im = A+B
        fs = [f(im), f(im[::2]), f(im[:,::2]), f(im[::2, ::2])]
        assert np.var(fs) < np.mean(np.abs(fs))/10

def test_moments01():
    im = np.zeros((16,16))
    im += np.arange(16)
    im -= im.mean()
    assert np.abs(mh.moments(im , 1, 0)) < 0.1

def test_expression():
    import mahotas as mh
    import numpy as np
    im = (np.zeros((12,16)) + np.arange(16))
    im -= im.mean()
    c0,c1 = 4,7
    p0,p1 = 2, 3
    for p0,p1 in [(0,1),
                    (1,0),
                    (1,2),
                    (3,2),
                    (2,0)]:
        sum_r = sum((im[i,j] * (i - c0)**p0 * (j - c1)**p1) for i in range(im.shape[0]) for j in range(im.shape[1]))
        mm_r = mh.moments(im , p0, p1, cm=(c0,c1))
        assert np.abs(mm_r - sum_r) < 1.
