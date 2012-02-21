import numpy as np
import mahotas



def test_grey_erode():
    from mahotas.tests.pymorph_copy import erode as slow_erode
    np.random.seed(334)
    for i in xrange(8):
        f = np.random.random_sample((128,128))
        f *= 255
        f = f.astype(np.uint8)
        B = (np.random.random_sample((3,3))*255).astype(np.uint8)
        B //= 4
        fast = mahotas.erode(f,B)
        slow = slow_erode(f,B)
        # mahotas & pymorph use different border conventions.
        assert np.all(fast[1:-1,1:-1] == slow[1:-1,1:-1])


def test_dilate_erode():
    A = np.zeros((100,100))
    Bc = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]], bool)
    A[30,30] = 1
    A = (A!=0)
    origs = []
    for i in xrange(12):
        origs.append(A.copy())
        A = mahotas.dilate(A, Bc)
    for i in xrange(12):
        A = mahotas.erode(A, Bc)
        assert np.all(A == origs[-i-1])



