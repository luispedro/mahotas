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
    assert np.all( (L > 0) == (A > 0) )
    assert set(L.ravel()) == set([0,1,2])


def test_all_ones():
    labeled, nr = mahotas.labeled.label(np.ones((32,32)))
    assert nr == 1
    assert np.all(labeled == 1)

def test_random():
    np.random.seed(33)
    A = np.random.rand(128,128) > .8
    labeled,nr = mahotas.labeled.label(A)
    assert len(set(labeled.ravel())) == (nr+1)
    assert labeled.max() == nr
