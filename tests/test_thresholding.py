import numpy as np
from mahotas.thresholding import otsu

def test_otsu():
    np.random.seed(123)
    A = np.random.rand(128,128)
    A[24:48,24:48] += 4 * np.random.rand(24,24)
    A *= 255//A.max()
    A = A.astype(np.uint8)
    T = otsu(A)
    assert (A > T)[24:48,24:48].mean() > .5
    assert (A > T)[:24,:24].mean() < .5
    assert (A > T)[48:,:].mean() < .5
    assert (A > T)[:,48:].mean() < .5
