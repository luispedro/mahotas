import numpy as np
import mahotas.morph
from scipy import ndimage
import pytest

def slow_majority(img, N):
    img = (img > 0)
    r,c = img.shape
    output = np.zeros_like(img)
    for y in range(r-N):
        for x in range(c-N):
            count = 0
            for dy in range(N):
                for dx in range(N):
                    count += img[y+dy, x+dx]
            if count >= (N*N)//2:
                output[y+dy//2,x+dx//2] = 1
    return output

def gen_R():

    np.random.seed(22)
    R = np.random.rand(64, 64) > .68
    yield R
    yield R*24.

    R = np.random.rand(32, 64) > .68
    yield R

    R = np.random.rand(64, 64) > .68
    yield R[:32,:]

    R = np.random.rand(64, 64) > .68
    yield R[:23,:]

@pytest.mark.parametrize('R', list(gen_R()))
def test_compare_w_slow(R):
    for N in (3,5,7):
        assert np.all(mahotas.morph.majority_filter(R, N) == slow_majority(R, N))


def test_N0():
    with pytest.raises(ValueError):
        mahotas.morph.majority_filter(np.zeros((20,20), np.bool_), 0)



