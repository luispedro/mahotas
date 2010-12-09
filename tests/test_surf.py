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


def test_sum_rect():
    import mahotas._surf
    f = np.arange(800*160).reshape((800,160)) % 7
    fi = mahotas.surf.integral(f.copy()) 

    np.random.seed(22)
    for i in xrange(100):
        y0 = np.random.randint(1,780)
        y1 = np.random.randint(y0+1,799)
        x0 = np.random.randint(1,150)
        x1 = np.random.randint(x0+1, 159)
        assert mahotas._surf.sum_rect(fi, y0, x0, y1, x1) == f[y0:y1, x0:x1].sum()
