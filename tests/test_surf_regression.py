import mahotas
import numpy as np
import mahotas.surf
from os import path

def test_determinant_zero():
    img = mahotas.imread(path.join(
        path.abspath(path.dirname(__file__)),
                    'data',
                    'determinant_zero.png'))
    points = mahotas.surf.surf(img, threshold=.0)
    assert type(points) == np.ndarray

