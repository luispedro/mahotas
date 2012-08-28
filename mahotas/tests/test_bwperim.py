from mahotas import bwperim
import numpy as np

def _neighbours(bwimg, y, x, n):
    s0 = max(0, y-1)
    e0 = min(y+2, bwimg.shape[0])
    s1 = max(0, x-1)
    e1 = min(x+2, bwimg.shape[1])
    if n == 8:
        return bwimg[s0:e0, s1:e1]
    return np.concatenate([bwimg[s0:e0,x], bwimg[y,s1:e1]])

def _slow_bwperim(bwimg, n=4):
    r,c = bwimg.shape
    res = np.zeros_like(bwimg)
    for y in range(r):
        for x in range(c):
            res[y,x] = bwimg[y,x] and np.any(~_neighbours(bwimg,y,x,n))
    return res

def _compare_slow(img):
    for n in (4,8):
        assert np.all(_slow_bwperim(img, n) == bwperim(img, n))

def test_bwperim():
    img = np.zeros((8,8), np.bool)
    img[3:7,3:7] = 1
    _compare_slow(img)

    assert img[3:7,3:7].all()
    assert img[3:7,3:7].sum() == img.sum()
    img[2,2:4] = 1
    _compare_slow(img)
