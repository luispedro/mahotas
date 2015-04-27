import numpy as np
import mahotas
import sys

def test_close_holes_simple():
    img = np.zeros((64,64),bool)
    img[16:48,16:48] = True
    holed =  np.logical_xor(img, mahotas.erode(mahotas.erode(img)))
    assert np.all( mahotas.close_holes(holed) == img)
    holed[12,12] = True
    img[12,12] = True
    assert np.all( mahotas.close_holes(holed) == img)
    assert sys.getrefcount(holed) == 2
