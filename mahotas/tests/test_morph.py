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

    @raises(ValueError)
    def bad_dims():
        Bc = np.ones((3,3,3), dtype=np.bool)
        get_structuring_elem(A, Bc)

    bad_dims()
        

def test_open():
    from mahotas.morph import open
    np.random.seed(123)
    A = np.random.random_sample((16,16)) > .345
    assert open(A).shape == (16,16)

def test_close():
    from mahotas.morph import close
    np.random.seed(123)
    A = np.random.random_sample((16,16)) > .345
    assert close(A).shape == (16,16)


def slow_reg(A, agg):
    def get(i, j):
        vals = []
        def try_this(i,j):
            if 0 <= i < A.shape[0] and \
                0 <= j < A.shape[1]:
                    vals.append( A[i,j] )
        try_this(i,j)
        try_this(i-1,j)
        try_this(i+1,j)
        try_this(i,j-1)
        try_this(i,j+1)
        return vals


    res = np.zeros(A.shape, bool)
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            res[i,j] = (A[i,j] == agg(get(i,j)))
    return res

def test_locmin_max():
    from mahotas.morph import locmax, locmin
    np.random.seed(123)
    for i in range(4):
        A = np.random.random_sample((64,64))
        A *= 255
        A = A.astype(np.uint8)
        fast = locmax(A)
        assert np.all(fast == slow_reg(A, max))

        fast = locmin(A)
        assert np.all(fast == slow_reg(A, min))


def test_regmax_min():
    from mahotas.morph import locmax, locmin, regmax, regmin
    np.random.seed(123)
    for i in range(4):
        A = np.random.random_sample((64,64))
        A *= 255
        A = A.astype(np.uint8)

        loc = locmax(A)
        reg = regmax(A)
        assert not np.any(reg & ~loc)

        loc = locmin(A)
        reg = regmin(A)
        assert not np.any(reg & ~loc)

def test_dilate_crash():
    # There was a major bug in dilate, that caused this to crash
    from mahotas.morph import dilate
    large = np.random.random_sample((512,512)) > .5
    small = large[128:256,128:256]
    dilate(small)

