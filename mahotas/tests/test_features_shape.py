import mahotas.features.shape
import numpy as np
import mahotas as mh
from mahotas.features.shape import roundness, eccentricity

def test_eccentricity():
    D = mh.disk(32, 2)
    ecc = mahotas.features.shape.eccentricity(D)
    assert 0 <= ecc < .01

    Index = np.indices((33,33)).astype(float)
    Index -= 15
    X,Y = Index
    ellipse = ((X**2+2*Y**2) < 12**2)
    assert 0 < mahotas.features.shape.eccentricity(ellipse) < 1

def test_roundness():
    Y,X = -24 + np.indices((48,48)).astype(float)
    r = roundness( (Y ** 2. + X**2.) < 4**2. )
    assert r > 0
    r2 = roundness( (Y ** 2. + 2* X**2.) < 4**2. )
    assert r2 > 0
    assert r2 < r

def test_zeros():
    assert roundness(np.zeros((10,10))) == 0
    assert eccentricity(np.zeros((10,10))) == 0
    I = np.zeros((16,16))
    I[8:4:12] = 1
    assert eccentricity(I) == 0

def test_ellipse_axes():
    Y,X = np.mgrid[:1024,:1024]
    Y = Y/1024.
    X = X/1024.
    im = ((2.*(Y - .5)**2 + (X - .5)**2) < .2)
    maj,min = mh.features.ellipse_axes(im)
    assert np.abs(2 - (maj/min)**2) < .01

    maj2,min2 = mh.features.ellipse_axes(im.T)

    assert np.abs(maj - maj2) < .001
    assert np.abs(min - min2) < .001

    im = (((Y - .5)**2 + (X - .5)**2) < .2)
    maj,min = mh.features.ellipse_axes(im)
    assert np.abs(1 - (maj/min)**2) < .01
