import numpy as np
import mahotas as mh
import mahotas

def test_grey_erode():
    from mahotas.tests.pymorph_copy import erode as slow_erode
    from mahotas.tests.pymorph_copy import dilate as slow_dilate
    np.random.seed(334)
    for i in range(8):
        f = np.random.random_sample((128,128))
        f *= 255
        f = f.astype(np.uint8)
        B = (np.random.random_sample((3,3))*255).astype(np.uint8)
        B //= 4
        fast = mahotas.erode(f,B)
        slow = slow_erode(f,B)
        # mahotas & pymorph use different border conventions.
        assert np.all(fast[1:-1,1:-1] == slow[1:-1,1:-1])

        fast = mahotas.dilate(f,B)
        slow = slow_dilate(f,B)
        # mahotas & pymorph use different border conventions.
        assert np.all(fast[1:-1,1:-1] == slow[1:-1,1:-1])


def test_dilate_erode():
    A = np.zeros((128,128), dtype=bool)
    Bc = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]], bool)
    A[32,32] = True
    origs = []
    for i in range(12):
        origs.append(A.copy())
        A = mahotas.dilate(A, Bc)
    for i in range(12):
        A = mahotas.erode(A, Bc)
        assert np.all(A == origs[-i-1])



def test_dilate_1():
    A = np.zeros((16,16), dtype=np.uint8)
    B = np.array([
        [0,1,0],
        [2,2,1],
        [1,3,0]], dtype=np.uint8)
    A[8,8] = 1
    D = mahotas.dilate(A, B)
    assert np.sum(D) == np.sum(B+(B>0))

def test_signed():
    A = np.array([0,0,1,1,1,0,0,0], dtype=np.int32)
    B = np.array([0,1,0])
    assert np.min(mahotas.erode(A,B)) == -1


def test_cerode():
    from mahotas.tests.pymorph_copy import erode as slow_erode
    from mahotas.tests.pymorph_copy import dilate as slow_dilate
    np.random.seed(334)
    f = np.random.random_sample((128,128))
    f = (f > .9)
    assert np.all(mahotas.erode(f) == mahotas.cerode(f, np.zeros_like(f)))

def test_cdilate():
    from mahotas.tests.pymorph_copy import cdilate as slow_cdilate
    np.random.seed(332)
    Bc = np.zeros((3,3),bool)
    Bc[1] = 1
    Bc[:,1] = 1
    for n in range(8):
        f = np.random.random_sample((128,128))
        f = (f > .9)
        g = np.random.random_sample((128,128))
        g = (g > .7)
        assert np.all(mahotas.cdilate(f,g,Bc,n+1) == slow_cdilate(f,g,Bc,n+1))


def test_cdilate_large():
    f = np.zeros((8,8),bool)
    f[4,4] = 1
    assert np.all(mh.cdilate(f, 12))


def test_erode_slice():
    np.random.seed(30)
    for i in range(16):
        f = (np.random.random_sample((256,256))*255).astype(np.uint8)
        assert np.all(mahotas.erode(f[:3,:3]) == mahotas.erode(f[:3,:3].copy()))

def test_dilate_slice():
    np.random.seed(30)
    for i in range(16):
        f = (np.random.random_sample((256,256))*255).astype(np.uint8)
        assert np.all(mahotas.dilate(f[:3,:3]) == mahotas.dilate(f[:3,:3].copy()))

def test_fast_binary():
    # This test is based on an internal code decision: the fast code is only triggered for CARRAYs
    # Therefore, we test to see if both paths lead to the same result
    np.random.seed(34)
    for i in range(8):
        f = np.random.random((128,128)) > .9
        f2 = np.dstack([f,f,f])

        SEs = [
            np.ones((3,3)),
            np.ones((5,5)),
            np.array([
                    [0,1,0],
                    [0,0,0],
                    [0,0,0]]),
            np.array([
                    [0,0,0],
                    [1,0,0],
                    [0,0,0]]),
            np.array([
                    [1,0,0],
                    [1,0,0],
                    [0,0,0]]),
            np.array([
                    [1,1,1],
                    [1,1,1],
                    [1,1,0]]),
            np.array([
                    [1,1,1],
                    [0,1,1],
                    [1,1,0]]),
            ]
        for Bc in SEs:
            assert np.all(mahotas.erode(f,Bc=Bc) == mahotas.erode(f2[:,:,1],Bc=Bc))
            # For dilate, the border conditions are different;
            # This is not great, but it's actually the slow implementation
            # which has the most unsatisfactory behaviour:
            assert np.all(mahotas.dilate(f,Bc=Bc)[1:-1,1:-1] == mahotas.dilate(f2[:,:,1],Bc=Bc)[1:-1,1:-1])

def test_se_zeros():
    np.random.seed(35)
    f = np.random.random((128,128)) > .9
    f2 = np.dstack([f,f,f])
    mahotas.erode(f, np.zeros((3,3)))
    mahotas.dilate(f, np.zeros((3,3)))
    mahotas.erode(f2[:,:,1], np.zeros((3,3)))
    mahotas.dilate(f2[:,:,1], np.zeros((3,3)))
