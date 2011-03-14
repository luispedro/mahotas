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

def test_surf_guassians():
    f = np.zeros((1024,1024))
    Y,X = np.indices(f.shape)
    Y -= 768
    X -= 768
    f += 120*np.exp(-Y**2/2048.-X**2/480.)
    Y += 512
    X += 512
    f += 120*np.exp(-Y**2/2048.-X**2/480.)
    spoints = mahotas.surf.surf(f, 1, 24, 2)

    YX = np.array([spoints[:,0],spoints[:,1]]).T
    is_256 = False
    is_768 = False
    for y,x in YX:
        if (np.abs(y-256) < 8 and np.abs(x-256) < 8): is_256 = True
        if (np.abs(y-768) < 8 and np.abs(x-768) < 8): is_768 = True
    assert is_256
    assert is_768

def test_interest_points_descriptors():
    np.random.seed(22)
    f = np.random.rand(256,256)*230
    f = f.astype(np.uint8)
    fi = mahotas.surf.integral(f)
    spoints = mahotas.surf.surf(f, 6, 24, 1)
    for arr, is_integral in zip([f,fi], [False, True]):
        points = mahotas.surf.interest_points(arr, 6, 24, 1, is_integral=is_integral)
        points = list(points)
        points.sort(key=(lambda p: -p[3]))
        points = np.array(points, dtype=np.float64)
        descs = mahotas.surf.descriptors(arr, points, is_integral)
        assert np.all(descs[:len(spoints)] == spoints)


