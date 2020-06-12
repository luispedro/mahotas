import numpy as np
from mahotas.histogram import fullhistogram
import pytest

def test_fullhistogram():
    A100 = np.arange(100).reshape((10,10)).astype(np.uint32)
    assert fullhistogram(A100).shape == (100,)
    assert np.all(fullhistogram(A100) == np.ones(100))

    A1s = np.ones((12,12), np.uint8)
    assert fullhistogram(A1s).shape == (2,)
    assert np.all(fullhistogram(A1s) == np.array([0,144]))

    A1s[0] = 0
    A1s[1] = 2
    assert fullhistogram(A1s).shape == (3,)
    assert np.all(fullhistogram(A1s) == np.array([12,120,12]))

def test_fullhistogram_random():
    np.random.seed(122)
    A = np.random.rand(12,3,44,33)*1000
    A = A.astype(np.uint16)
    hist = fullhistogram(A)
    for i in range(len(hist)):
        assert hist[i] == (A == i).sum()
    assert len(hist.shape) == 1

    A = A[::2,:,2::3,1:-2:2]
    hist = fullhistogram(A)
    for i in range(len(hist)):
        assert hist[i] == (A == i).sum()
    assert hist.sum() == A.size
    assert len(hist.shape) == 1

def test_fullhistogram_boolean():
    np.random.seed(123)
    A = (np.random.rand(128,128) > .5)
    H = fullhistogram(A)
    assert H[0] == (~A).sum()
    assert H[1] == A.sum()

def test_types():
    A100 = np.arange(100).reshape((10,10)).astype(np.uint32)
    assert np.all(fullhistogram(A100.astype(np.uint8)) == fullhistogram(A100))
    assert np.all(fullhistogram(A100.astype(np.uint16)) == fullhistogram(A100))
    assert np.all(fullhistogram(A100.astype(np.uint32)) == fullhistogram(A100))
    assert np.all(fullhistogram(A100.astype(np.uint64)) == fullhistogram(A100))

def test_float():
    with pytest.raises(TypeError):
        fullhistogram(np.arange(16.*4., dtype=float).reshape((16,4)))

