import mahotas as mh
import numpy as np
from mahotas.morph import get_structuring_elem, subm, tophat_open, tophat_close
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
    for i in range(8):
        A = np.random.random_sample((64,64))
        A *= 255
        if (i % 2) == 0:
            A = A.astype(np.uint8)
        fast = locmax(A)
        assert np.all(fast == slow_reg(A, max))

        fast = locmin(A)
        assert np.all(fast == slow_reg(A, min))


def test_regmax_min():
    from mahotas.morph import locmax, locmin, regmax, regmin
    np.random.seed(123)
    for i in range(8):
        A = np.random.random_sample((64,64))
        A *= 255
        if (i % 2) == 1:
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

def slow_subm_uint8(a, b):
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    c = a - b
    return np.clip(c, 0, 255).astype(np.uint8)

def slow_subm_uint16(a, b):
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    c = a - b
    return np.clip(c, 0, 2**16-1).astype(np.uint16)

def slow_subm_int16(a, b):
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    c = a - b
    return np.clip(c, -2**15, 2**15-1).astype(np.int16)

def test_subm():
    np.random.seed(34)
    for j in range(8):
        s = (128, 256)
        a = np.random.randint(0,255, size=s)
        b = np.random.randint(0,255, size=s)
        a = a.astype(np.uint8)
        b = b.astype(np.uint8)
        assert np.all(slow_subm_uint8(a,b) == subm(a,b))

        a = 257*np.random.randint(0,255, size=s)
        b = 257*np.random.randint(0,255, size=s)
        a = a.astype(np.uint16)
        b = b.astype(np.uint16)
        assert np.all(slow_subm_uint16(a,b) == subm(a,b))

        a2 = 257*np.random.randint(0,255, size=s)
        b2 = 257*np.random.randint(0,255, size=s)
        a = a.astype(np.int16)
        b = b.astype(np.int16)
        a -= a2
        b -= b2
        assert np.all(slow_subm_int16(a,b) == subm(a,b))



def test_subm_out():
    np.random.seed(32)
    for j in range(8):
        s = (128, 256)
        a = np.random.randint(0,255, size=s)
        b = np.random.randint(0,255, size=s)

        c = subm(a,b)
        assert c is not a
        assert c is not b
        assert not np.all(c == a)


        c = subm(a,b, out=a)
        assert c is  a
        assert c is not b
        assert np.all(c == a)

def test_tophat():
    np.random.seed(32)
    for j in range(8):
        s = (128, 256)
        f = np.random.randint(0,255, size=s)
        f = f.astype(np.uint8)
        g = tophat_close(f)
        assert f.shape == g.shape

        g = tophat_open(f)
        assert f.shape == g.shape


def test_circle_se():
    from mahotas.morph import circle_se
    for r in (4,5):
        c = circle_se(r)
        assert len(c) == (2*r + 1)
        assert len(c) == len(c.T)
        assert not c.all()
        assert c.any()

    @raises(ValueError)
    def circle_1():
        circle_se(-1)
    circle_1()

def test_distance_multi():
    import mahotas._morph
    np.random.seed(20)
    binim = np.random.random((12,18)) > .1
    f = (binim * 0 + 26 *27).astype(float)
    Bc = np.ones((3,3), bool)
    mahotas._morph.distance_multi(f, binim, Bc)
    f2 = mahotas.distance(binim)
    assert np.all(f == f2)



def test_disk():
    from mahotas.morph import disk
    D2 = disk(2)
    assert D2.shape[0] == D2.shape[1]
    assert D2.shape == (5,5)
    assert not D2[0,0]
    assert len(D2.shape) == 2
    D3 = disk(2,3)

    assert np.all(D3[2] == D2)

    D3 = disk(4,3)
    assert len(D3.shape) == 3
    assert D3.shape[0] == D3.shape[1]
    assert D3.shape[0] == D3.shape[2]

    # Simple regression
    D = disk(32, 2)
    assert D[32,2]

    @raises(ValueError)
    def test_negative_dim(dim):
        disk(3, dim)

    test_negative_dim(-2)
    test_negative_dim(-1)
    test_negative_dim(0)


@raises(ValueError)
def test_close_holes_3d():
    'Close holes should raise exception with 3D inputs'
    f = np.random.rand(100,100,3) > .9
    mh.close_holes(f)
