import mahotas
import numpy as np
from mahotas.features.surf import surf
from os import path

# Originally, the file `determinant_zero.png` contained a PNG which FreeImage
# was opening incorrectly (it returned the palette instead of mapping through
# to the colours).
#
# If correctly mapped, the image was actually full of zeros (i.e., all colours
# in the palette were (0,0,0)!)
#
# Therefore, there are actually two tests here:
#     with a zero image
#     with the values of the palette in determinant_zero.png
#
# The file `determinant_zero.png` now contains what was originally the palette
# values.

def test_determinant_zero():
    img = mahotas.imread(path.join(
        path.abspath(path.dirname(__file__)),
                    'data',
                    'determinant_zero.png'))
    points = surf(img, threshold=.0)
    assert type(points) == np.ndarray

def test_determinant_zero2():
    img = np.zeros((128,28), np.uint8)
    points = surf(img, threshold=.0)
    assert type(points) == np.ndarray

