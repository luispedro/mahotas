import numpy as np
import mahotas.morph
from scipy import ndimage

def slow_majority(img, N):
    r,c = img.shape
    output = np.zeros_like(img)
    for y in xrange(r-N):
        for x in xrange(c-N):
            count = 0
            for dy in xrange(N):
                for dx in xrange(N):
                    count += img[y+dy, x+dx]
            if count >= (N*N)//2:
                output[y+dy//2,x+dx//2] = 1
    return output

def test_majority():
    np.random.seed(22)
    R = np.random.rand(64, 64) > .88
    assert np.all(mahotas.morph.majority_filter(R, 3) == slow_majority(R, 3))

    R = np.random.rand(32, 64) > .88
    assert np.all(mahotas.morph.majority_filter(R, 3) == slow_majority(R, 3))

    R = np.random.rand(64, 64) > .88
    assert np.all(mahotas.morph.majority_filter(R[:32,:], 3) == slow_majority(R[:32,:], 3))

    R = np.random.rand(64, 64) > .88
    assert np.all(mahotas.morph.majority_filter(R[:23,:], 3) == slow_majority(R[:23,:], 3))

