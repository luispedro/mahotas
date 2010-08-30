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
    def compare_w_slow(R):
        assert np.all(mahotas.morph.majority_filter(R, 3) == slow_majority(R, 3))

    np.random.seed(22)
    R = np.random.rand(64, 64) > .88
    yield compare_w_slow, R

    R = np.random.rand(32, 64) > .88
    yield compare_w_slow, R

    R = np.random.rand(64, 64) > .88
    yield compare_w_slow, R[:32,:]

    R = np.random.rand(64, 64) > .88
    yield compare_w_slow, R[:23,:]

