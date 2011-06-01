import numpy as np
from mahotas.convolve import median_filter
def test_median_filter():
    A = np.zeros((128,128), bool)
    A[3::3,3::3] = 1
    Am = median_filter(A)
    assert not Am.any()
    assert Am.shape == A.shape

