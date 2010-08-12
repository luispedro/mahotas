import numpy as np
import mahotas.thin
import mahotas

def test_thin():
    A = np.zeros((100,100), bool)
    A[20:40] = 1
    W = mahotas.thin(A)
    assert mahotas.erode(W).sum() == 0

