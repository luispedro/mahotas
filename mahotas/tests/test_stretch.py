from mahotas.stretch import stretch
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

