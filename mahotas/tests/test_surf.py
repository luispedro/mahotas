import numpy as np
import mahotas as mh
import mahotas.features.surf as surf
from mahotas.features import _surf
from .utils import luispedro_jpg
from nose.tools import raises

def test_integral():
    f = np.arange(8*16).reshape((8,16)) % 8
    fi = surf.integral(f.copy())
    assert fi[-1,-1] == f.sum()
    for y,x in np.indices(f.shape).reshape((2,-1)).T:
        assert fi[y,x] == f[:y+1,:x+1].sum()

def test_integral2():
    f = np.arange(80*16).reshape((80,16)) % 7
    fi = surf.integral(f.copy())
    assert fi[-1,-1] == f.sum()
    for y,x in np.indices(f.shape).reshape((2,-1)).T:
        assert fi[y,x] == f[:y+1,:x+1].sum()


def test_sum_rect():
    f = np.arange(800*160).reshape((800,160)) % 7
    fi = surf.integral(f.copy())

    np.random.seed(22)
    for i in range(100):
        y0 = np.random.randint(1,780)
        y1 = np.random.randint(y0+1,799)
        x0 = np.random.randint(1,150)
        x1 = np.random.randint(x0+1, 159)
        assert _surf.sum_rect(fi, y0, x0, y1, x1) == f[y0:y1, x0:x1].sum()

def test_sum_rect_edge():
    # regression
    # https://github.com/luispedro/mahotas/issues/58
    f = np.arange(80*60).reshape((80,60)) % 7
    fi = surf.integral(f.copy())
    _surf.sum_rect(fi, 0, 0, 81, 61)

def test_surf_guassians():
    f = np.zeros((1024,1024))
    Y,X = np.indices(f.shape)
    Y -= 768
    X -= 768
    f += 120*np.exp(-Y**2/2048.-X**2/480.)
    Y += 512
    X += 512
    f += 120*np.exp(-Y**2/2048.-X**2/480.)
    spoints = surf.surf(f, 1, 24, 2)

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
    fi = surf.integral(f)
    spoints = surf.surf(f, 6, 24, 1)
    for arr, is_integral in zip([f,fi], [False, True]):
        points = surf.interest_points(arr, 6, 24, 1, is_integral=is_integral)
        points = list(points)
        points.sort(key=(lambda p: -p[3]))
        points = np.array(points, dtype=np.float64)
        descs = surf.descriptors(arr, points, is_integral)
        assert np.all(descs[:len(spoints)] == spoints)


def test_show_surf():
    np.random.seed(22)
    f = np.random.rand(256,256)*230
    f = f.astype(np.uint8)
    spoints = surf.surf(f, 6, 24, 1)
    f2 = surf.show_surf(f, spoints)
    assert f2.shape == (f.shape + (3,))


def test_interest_points_descriptor_only():
    np.random.seed(22)
    f = np.random.rand(256,256)*230
    f = f.astype(np.uint8)
    full = surf.surf(f, 6, 24, 1)
    only = surf.surf(f, 6, 24, 1, descriptor_only=True)
    assert full.size > only.size

def test_descriptors_descriptor_only():
    np.random.seed(22)
    f = np.random.rand(256,256)*230
    f = f.astype(np.uint8)
    points = surf.interest_points(f, 6, 24, 1)
    full = surf.descriptors(f, points)
    only = surf.descriptors(f, points, descriptor_only=True)
    assert full.size > only.size

@raises(ValueError)
def test_3d_image():
    surf.surf(np.arange(8*8*16).reshape((16,8,8)), 6, 24, 1)

@raises(TypeError)
def test_integral_intested_points():
    np.random.seed(22)
    f = np.random.rand(16,16)*230
    f = f.astype(np.uint8)
    f = surf.integral(f)
    surf.interest_points(f.astype(np.int32), is_integral=True)


@raises(TypeError)
def test_integral_descriptors():
    np.random.seed(22)
    f = np.random.rand(16,16)*230
    f = f.astype(np.uint8)
    f = surf.integral(f)
    points = surf.interest_points(f, is_integral=True)
    surf.descriptors(f.astype(np.int32), points, is_integral=True)

def test_dense():
    f = np.arange(280*360).reshape((280,360)) % 25
    d16 = surf.dense(f, 16)
    d16_s = surf.dense(f, 16, 3.)
    d32 = surf.dense(f, 32)

    assert len(d16) > len(d32)
    assert d16.shape[1] == d32.shape[1]
    assert d16.shape[1] == d16_s.shape[1]


def test_dense_scale():
    im = luispedro_jpg(True)
    surf.dense(im, spacing=32)
    s5 = surf.dense(im, spacing=32, scale=5)
    s51 = surf.dense(im, spacing=32, scale=5.1)
    assert not np.all(s5 == s51)
