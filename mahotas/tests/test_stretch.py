from mahotas.stretch import stretch
import mahotas
from nose.tools import raises
import numpy as np

def test_stretch():
    np.random.seed(2323)
    A = np.random.random_integers(12, 120, size=(100,100))
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
    r = np.random.random_integers(12, 120, size=(8,8))
    g = np.random.random_integers(12, 120, size=(8,8))
    b = np.random.random_integers(12, 120, size=(8,8))
    assert mahotas.as_rgb(r,g,b).max() >= 254
    assert mahotas.as_rgb(r,None,b).shape == (8,8,3)
    assert mahotas.as_rgb(r,None,b)[:,:,1].sum() == 0


@raises(ValueError)
def test_as_rgb_Nones():
    mahotas.as_rgb(None,None,None)

@raises(ValueError)
def test_as_rgb_shape_mismatch():
    np.random.seed(2323)
    r = np.random.random_integers(12, 120, size=(8,8))
    g = np.random.random_integers(12, 120, size=(8,8))
    b = np.random.random_integers(12, 120, size=(8,6))
    mahotas.as_rgb(r,g,b)


