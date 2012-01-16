import numpy as np
from mahotas.convolve import median_filter, rank_filter
def test_median_filter():
    A = np.zeros((128,128), bool)
    A[3::3,3::3] = 1
    Am = median_filter(A)
    assert not Am.any()
    assert Am.shape == A.shape


def _slow_rank_filter(A,r):
    B = np.zeros_like(A)
    for i in xrange(1,A.shape[0]-1):
        for j in xrange(1, A.shape[1]-1):
            pixels = A[i-1:i+2,j-1:j+2].ravel()
            pixels = pixels.copy()
            pixels.sort()
            B[i,j] = pixels[r]
    return B[1:-1,1:-1]
def test_rank_filter():
    np.random.seed(22)
    A = np.random.random_integers(0,255, (32,32))
    Bc = np.ones((3,3))
    for r in xrange(9):
        B1 = rank_filter(A,Bc,r)
        B2 = _slow_rank_filter(A,r)
        assert np.all(B1[1:-1,1:-1] == B2)

def test_uint8():
    # This used to raise an exception in 0.7.1
    f = np.arange(64*4).reshape((16,-1))
    median_filter(f.astype(np.uint8), np.ones((5,5)))

