import numpy as np
import mahotas.surf
def test_integral():
    f = np.arange(8*16).reshape((8,16)) % 8
    fi = mahotas.surf.integral(f.copy())
    assert fi[-1,-1] == f.sum()
    for y,x in np.indices(f.shape).reshape((2,-1)).T:
        assert fi[y,x] == f[:y+1,:x+1].sum()

def test_integral2():
    f = np.arange(80*16).reshape((80,16)) % 7
    fi = mahotas.surf.integral(f.copy())
    assert fi[-1,-1] == f.sum()
    for y,x in np.indices(f.shape).reshape((2,-1)).T:
        assert fi[y,x] == f[:y+1,:x+1].sum()
