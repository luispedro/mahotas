import mahotas.labeled
import numpy as np
def test_label():
    A = np.zeros((128,128), np.int)
    L,n = mahotas.labeled.label(A)
    assert not L.max()
    assert n == 0

    A[2:5, 2:5] = 34
    A[10:50, 10:50] = 34
    L,n = mahotas.labeled.label(A)
    assert L.max() == 2
    assert L.max() == n
    assert np.sum(L > 0) == (40*40 + 3*3)

