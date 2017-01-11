import numpy as np
from mahotas.convolve import median_filter, rank_filter
from nose.tools import raises

def test_median_filter():
    A = np.zeros((128,128), bool)
    A[3::3,3::3] = 1
    Am = median_filter(A)
    assert not Am.any()
    assert Am.shape == A.shape


def _slow_rank_filter(A,r):
    B = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            s0 = max(0, i - 1)
            s1 = max(0, j - 1)
            e0 = min(A.shape[0], i + 2)
            e1 = min(A.shape[1], j + 2)
            pixels = list(A[s0:e0, s1:e1].ravel())
            pixels.extend([0] * (9-len(pixels)))
            pixels.sort()
            B[i,j] = pixels[r]
    return B

def test_rank_filter():
    np.random.seed(22)
    A = np.random.randint(0, 256, (32,32))
    Bc = np.ones((3,3))
    for r in range(9):
        B1 = rank_filter(A, Bc, r, mode='constant')
        B2 = _slow_rank_filter(A,r)
        assert np.all(B1 == B2)

def test_uint8():
    # This used to raise an exception in 0.7.1
    f = np.arange(64*4).reshape((16,-1))
    median_filter(f.astype(np.uint8), np.ones((5,5)))

@raises(ValueError)
def test_mismatched_ndim():
    a = np.zeros((8,8))
    a[:2] = 2
    a = np.array([a])
    median_filter(a > 0, np.ones((3,3)))
