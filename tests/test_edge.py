from mahotas.edge import sobel
import numpy as np
def test_sobel():
    I = np.array([
            [0,0,0,1,0,0],
            [0,0,0,1,0,0],
            [0,0,1,1,1,0],
            [0,0,1,1,1,0],
            [0,0,0,0,0,0]])
    E = sobel(I)
    r,c = I.shape
    for y,x in zip(*np.where(E)):
        N = [I[y,x]]
        if y > 0: N.append(I[y-1,x])
        if x > 0: N.append(I[y,x-1])
        if y < (r-1): N.append(I[y+1,x])
        if x < (c-1): N.append(I[y,x+1])
        assert len(set(N)) > 1

def test_zero_images():
    assert np.isnan(sobel(np.zeros((16,16)))).sum() == 0

