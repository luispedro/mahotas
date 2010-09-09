import numpy as np
import mahotas
from nose.tools import raises

def test_croptobbox():
    X,Y = np.meshgrid(np.arange(16)-8,np.arange(16)-8)
    ball = ((X**2+Y**2) < 32).astype(np.uint8)
    assert mahotas.croptobbox(ball).sum() == ball.sum()
    assert mahotas.croptobbox(ball,border=2).sum() == ball.sum()
    assert mahotas.croptobbox(ball,border=256).sum() == ball.sum()
    assert mahotas.croptobbox(ball,border=256).size == ball.size

def test_bbox_empty():
    assert mahotas.bbox(np.zeros((), np.bool)).shape == (0,)

@raises(NotImplementedError)
def test_bbox_3():
    mahotas.bbox(np.arange(3*3*3).reshape((3,3,3)))

