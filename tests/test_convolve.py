import numpy as np
import mahotas.convolve
import mahotas._filters

def test_compare_w_ndimage():
    from scipy import ndimage
    A = np.arange(34*340).reshape((34,340))%3
    B = np.ones((3,3), A.dtype)
    for mode in mahotas._filters.modes:
        assert np.all(mahotas.convolve.convolve(A, B, mode=mode) == ndimage.convolve(A, B, mode=mode))
