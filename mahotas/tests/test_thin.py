import numpy as np
import mahotas.thin

def slow_thin(binimg, n=-1):
    """
    This was the old implementation
    """
    from mahotas.bbox import bbox
    from mahotas._morph import hitmiss

    _struct_elems = []
    _struct_elems.append([
            [0,0,0],
            [2,1,2],
            [1,1,1]])
    _struct_elems.append([
            [2,0,0],
            [1,1,0],
            [1,1,2]])
    _struct_elems.append([
            [1,2,0],
            [1,1,0],
            [1,2,0]])
    _struct_elems.append([
            [1,1,2],
            [1,1,0],
            [2,0,0]])
    _struct_elems.append([
            [1,1,1],
            [2,1,2],
            [0,0,0]])
    _struct_elems.append([
            [2,1,1],
            [0,1,1],
            [0,0,2]])
    _struct_elems.append([
            [0,2,1],
            [0,1,1],
            [0,2,1]])
    _struct_elems.append([
            [0,0,2],
            [0,1,1],
            [2,1,1]])

    _struct_elems = [np.array(elem, np.uint8) for elem in _struct_elems]
    res = np.zeros_like(binimg)
    min0,max0,min1,max1 = bbox(binimg)

    r,c = (max0-min0,max1-min1)

    image_exp = np.zeros((r+2, c+2), np.uint8)
    imagebuf = np.zeros((r+2,c+2), np.uint8)
    prev = np.zeros((r+2,c+2), np.uint8)
    image_exp[1:r+1, 1:c+1] = binimg[min0:max0,min1:max1]
    n_iter = 0
    while True:
        prev[:] = image_exp[:]
        for elem in _struct_elems:
            newimg = hitmiss(image_exp, elem, imagebuf)
            image_exp -= newimg
        if np.all(prev == image_exp):
            break
        n_iter += 1
        if (n > 0) and (n_iter == n):
            break
    res[min0:max0,min1:max1] = image_exp[1:r+1, 1:c+1]
    return res


def test_thin():
    A = np.zeros((100,100), bool)
    A[20:40] = 1
    W = mahotas.thin(A)
    assert mahotas.erode(W).sum() == 0
    assert (W & A).sum() == W.sum()

def test_compare():
    def compare(A):
        W = mahotas.thin(A)
        W2 = slow_thin(A)
        assert np.all(W == W2)
    A = np.zeros((100,100), bool)
    yield compare, A
    A[20:40] = 1
    yield compare, A
    A[:,20:40] = 1
    yield compare, A

    A[60:80,60:80] = 1
    yield compare, A

    W = mahotas.thin(A, 2)
    W2 = slow_thin(A, 2)
    assert np.all(W == W2)


