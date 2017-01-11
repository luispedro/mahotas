from mahotas.stretch import stretch
import mahotas
import mahotas as mh
from nose.tools import raises
import numpy as np

def test_stretch():
    np.random.seed(2323)
    A = np.random.randint(12, 121, size=(100,100))
    A = stretch(A, 255)
    assert A.max() > 250
    assert A.min() == 0
    A = stretch(A,20)
    assert A.max() <= 20
    A = stretch(A, 10, 20)
    assert A.min() >= 10
    A = stretch(A * 0, 10, 20)
    assert A.min() >= 10

def test_neg_numbers():
    A = np.arange(-10,10)
    scaled = stretch(A, 255)
    assert scaled.shape == A.shape
    assert scaled.min() <= 1
    assert scaled.max() >= 254



def test_as_rgb():
    np.random.seed(2323)
    r = np.random.randint(12, 121, size=(8,8))
    g = np.random.randint(12, 121, size=(8,8))
    b = np.random.randint(12, 121, size=(8,8))
    assert mahotas.as_rgb(r,g,b).max() >= 254
    assert mahotas.as_rgb(r,None,b).shape == (8,8,3)
    assert mahotas.as_rgb(r,None,b)[:,:,1].sum() == 0


@raises(ValueError)
def test_as_rgb_Nones():
    mahotas.as_rgb(None,None,None)

@raises(ValueError)
def test_as_rgb_shape_mismatch():
    np.random.seed(2323)
    r = np.random.randint(12, 121, size=(8,8))
    g = np.random.randint(12, 121, size=(8,8))
    b = np.random.randint(12, 121, size=(8,6))
    mahotas.as_rgb(r,g,b)



def test_as_rgb_integer():
    int_rgb = mh.as_rgb(1,2,np.zeros((8,6)))
    assert int_rgb.dtype == np.uint8
    assert int_rgb.shape == (8,6,3)
    assert np.all( int_rgb[0,0] == (1,2,0) )
    assert np.all( int_rgb[-1,3] == (1,2,0) )
    assert np.all( int_rgb[-2,4] == (1,2,0) )

def test_stretch_rgb():
    r = np.arange(256).reshape((32,-1))
    g = 255-r
    b = r/2
    s = mh.stretch(np.dstack([r,g,b]))
    s_rgb = mh.stretch_rgb(np.dstack([r,g,b]))
    assert not np.all(s == s_rgb)
    assert np.all(s[:,:,0] == s_rgb[:,:,0])
    assert np.all(mh.stretch(b) == mh.stretch_rgb(b))

@raises(ValueError)
def test_stretch_rgb4():
    mh.stretch_rgb(np.zeros((8,8,3,2)))


def test_overlay():
    im = mh.demos.load('luispedro', as_grey=1)
    im = mh.stretch(im)
    assert np.all(mh.overlay(im).max(2) == im)
    edges = mh.sobel(im)

    im3 = mh.overlay(im, green=edges)
    assert np.all(im3[:,:,0] == im)
    assert np.all(im3[:,:,2] == im)
    assert np.all(im3[:,:,1] >= im )
