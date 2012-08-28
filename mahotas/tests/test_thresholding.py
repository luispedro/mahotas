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

def test_ignore_zeros():
    np.seterr(all='raise')
    np.random.seed(22)
    A = np.zeros((1024,24), np.uint8)
    A[:24,:24] = np.random.random_integers(100, 200, size=(24,24))
    assert rc(A) < 100
    assert otsu(A) < 100
    assert rc(A, ignore_zeros=1) > 100
    assert otsu(A, ignore_zeros=1) > 100

def test_zero_image():
    A = np.zeros((16,16), np.uint8)
    def tm(method):
        assert method(A, ignore_zeros=0) == 0
        assert method(A, ignore_zeros=1) == 0
    yield tm, rc
    yield tm, otsu

def test_soft_threhold():
    from mahotas.thresholding import soft_threshold

    np.random.seed(223)
    for i in range(4):
        f = np.random.randint(-256,256, size=(128,128,4))
        fo = f.copy()
        t = soft_threshold(f, 16)

        assert not np.all(fo == t)
        assert np.all(t[np.abs(f) < 16] == 0)
        assert t.max() == f.max()-16
        assert t.min() == f.min()+16
        assert np.all( (np.abs(f) <= 16) | (np.abs(f)-16 == np.abs(t)))
