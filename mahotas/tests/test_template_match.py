import numpy as np
import mahotas.convolve
from mahotas.convolve import template_match

def test_template_match():
    np.random.seed(33)
    A = 255*np.random.random((1024, 512))
    t = A[8:12,8:12]+1
    m = template_match(A, t)

    assert m[10,10] == 16
    # I had tried testing over the whole image, but that took too long.
    for i in xrange(100):
        y = np.random.randint(m.shape[0]-4)
        x = np.random.randint(m.shape[1]-4)
        assert np.allclose(m[y+2,x+2], np.sum( (A[y:y+4, x:x+4] - t) ** 2))
