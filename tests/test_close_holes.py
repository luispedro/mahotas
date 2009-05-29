import numpy as np
import morph.morph

def test_close_holes_simple():
    img = np.zeros((64,64),bool)
    img[16:48,16:48] = True
    holed =  (img - morph.erode(morph.erode(img)))
    assert np.all( morph.close_holes(holed) == img)
    holed[12,12] = True
    img[12,12] = True
    assert np.all( morph.close_holes(holed) == img)
