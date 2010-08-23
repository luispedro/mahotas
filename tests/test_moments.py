import numpy as np
from mahotas.moments import moments

def test_moments():
    assert moments(np.zeros((100,23)), 2, 2) == 0.0
    assert moments(np.ones((100,23)), 2, 2) != 0.0
    assert moments(np.ones((100,23)), 0, 0) == 100*23
    assert moments(np.ones((100,23)), 2, 2) != moments(np.ones((100,23)), 2, 2, cm=(50,12))

