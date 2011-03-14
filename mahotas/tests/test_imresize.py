from mahotas import imresize
import numpy as np
def test_imresize():
    img = np.repeat(np.arange(100), 10).reshape((100,10))
    assert imresize(img, (1024,104)).shape == (1024,104)
    assert imresize(img, (10.,10.)).shape == (1000,100)
    assert imresize(img, .2,).shape == (20,2)
    assert imresize(img, (10.,2.)).shape == (1000,20)
