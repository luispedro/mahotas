import numpy as np
import mahotas.surf
def test_integral():
    f = np.arange(8*16).reshape((8,16)) % 8
    fi = mahotas.surf.integral(f.copy())
    fi[-1,-1] == f.sum()
    for y,x in np.indices(f.shape).reshape((2,-1)).T:
        assert fi[y,x] == f[:y+1,:x+1].sum()
