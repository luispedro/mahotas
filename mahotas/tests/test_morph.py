import numpy as np
from mahotas.morph import get_structuring_elem
from nose.tools import raises

def test_get_structuring_elem():
    A = np.zeros((10,10), np.bool)
    Bc = np.ones((4,4), dtype=np.bool)
    Bc[0,2] = 0

    assert np.all(get_structuring_elem(A, None) == [[0,1,0],[1,1,1],[0,1,0]])
    assert np.all(get_structuring_elem(A, 4) == [[0,1,0],[1,1,1],[0,1,0]])
    assert np.all(get_structuring_elem(A, 4) == get_structuring_elem(A, 1))
    assert np.all(get_structuring_elem(A, 8) == get_structuring_elem(A, 2))
    assert np.all(get_structuring_elem(A, 8) == np.ones((3,3), dtype=np.bool))
    assert np.all(get_structuring_elem(A, Bc) == Bc)
    assert np.all(get_structuring_elem(A, Bc.T) == Bc.T)
    assert get_structuring_elem(A, Bc.T).flags['C_CONTIGUOUS']
    assert np.all(get_structuring_elem(A, Bc.astype(np.float).T).flags['C_CONTIGUOUS'])
    assert np.all(get_structuring_elem(A, Bc.astype(np.float).T) == Bc.T)

