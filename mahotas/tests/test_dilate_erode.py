import numpy as np
import mahotas

def test_dilate_erode():
    A = np.zeros((100,100))
    Bc = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]], bool)
    A[30,30] = 1
    A = (A!=0)
    orig = A.copy()
    for i in xrange(12):
        A = mahotas.dilate(A, Bc)
    for i in xrange(12):
        A = mahotas.erode(A, Bc)
    assert np.all(A == orig)

