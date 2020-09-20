import numpy as np
import mahotas as mh
import mahotas.polygon
from mahotas.polygon import fill_polygon, fill_convexhull
import pytest

def test_polygon():
    polygon = [(10,10), (10,20), (20,20)]
    canvas = np.zeros((40,40), np.bool)
    fill_polygon(polygon, canvas)
    assert canvas.sum() == (10*10+10)/2
    canvas2 = canvas.copy()
    fill_polygon([], canvas2)
    assert np.all(canvas == canvas2)


def test_convex():
    polygon = [(100,232), (233,222), (234,23), (555,355), (343,345), (1000,800)]
    canvas = np.zeros((1024, 1024), np.bool)
    mahotas.polygon.fill_polygon(polygon, canvas)
    canvas2 = mahotas.polygon.fill_convexhull(canvas)
    # The overlap isn't perfect. There is a slight sliver. Fixing it is not
    # worth the trouble for me (LPC), but I'd take a patch
    assert (canvas & ~canvas2).sum() < 1024

def test_convex3():
    f = np.array([
        [False, False, False, False],
        [False,  True,  True, False],
        [False,  True, False, False],
        [False, False, False, False]], dtype=bool)
    assert np.all(fill_convexhull(f) == f)

def test_fill3():
    canvas = np.zeros((4,4), bool)
    # This polygon also has a horizontal and a vertical edge
    polygon = [(1, 1), (1, 2), (2, 1)]
    mahotas.polygon.fill_polygon(polygon, canvas)
    assert canvas.sum()

def test_line():
    canvas = np.zeros((32,32), int)
    polygon = [(8,8), (8,16),(16,16),(16,8), (8,8)]
    for p0,p1 in zip(polygon[:-1], polygon[1:]):
        mahotas.polygon.line(p0,p1, canvas, color=2)
    assert set(canvas.ravel()) == set([0,2])
    assert canvas.sum() == 2*(8*4) # 8*4 is perim size, 2 is value

def test_line_non_square():
    A = np.zeros((128, 64))
    mahotas.polygon.line((0,0),(127,63), A)
    assert A.sum()



def test_fill_line():
    # This is a regression test
    # https://github.com/luispedro/mahotas/issues/3
    canvas = np.zeros((50,30))
    poly = [( 0,10),
            (10, 0),
            (40,20),
            ( 0,10)]
    fill_polygon(poly, canvas)
    assert np.all(canvas[10,1:10])

def test_convex_in_3d():
    canvas = np.zeros((12,8,8))
    canvas[3,4,2] = 1
    canvas[5,4,2] = 1
    canvas[3,6,2] = 1
    with pytest.raises(ValueError):
        mahotas.polygon.fill_convexhull(canvas)


def test_border():
    canvas = np.zeros((32,32))
    polygon = np.array([(0,0),(0,32),(32,0)])
    fill_polygon(polygon, canvas)
    assert not np.all(canvas)
