import numpy as np
import mahotas as mh
from mahotas.convolve import mean_filter
def test_smoke():
    f = np.random.random((512,1024))
    se = np.ones((2,2))
    ff = mean_filter(f, se)
    for _ in range(128):
        a = np.random.randint(1, f.shape[0] - 1)
        b = np.random.randint(1, f.shape[1] - 1)
        assert np.allclose(ff[a,b], np.mean(f[a-1:a+1, b-1:b+1]))
