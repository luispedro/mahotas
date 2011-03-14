import numpy as np
from mahotas.thresholding import otsu, rc

def test_thresholding():
    np.random.seed(123)
    A = np.random.rand(128,128)
    A[24:48,24:48] += 4 * np.random.rand(24,24)
    A *= 255//A.max()
    A = A.astype(np.uint8)
    def tm(method):
        T = method(A)
        assert (A > T)[24:48,24:48].mean() > .5
        assert (A > T)[:24,:24].mean() < .5
        assert (A > T)[48:,:].mean() < .5
        assert (A > T)[:,48:].mean() < .5
    yield tm, otsu
    yield tm, rc


def test_nozeros():
    np.seterr(all='raise')
    np.random.seed(22)
    A = (np.random.rand(100,100)*50).astype(np.uint8)+201
    assert rc(A) > 200
    assert otsu(A) > 200

