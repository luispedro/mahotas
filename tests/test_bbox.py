import numpy as np
import mahotas

def test_croptobbox():
    X,Y = np.meshgrid(np.arange(16)-8,np.arange(16)-8)
    ball = ((X**2+Y**2) < 32).astype(np.uint8)
    assert mahotas.croptobbox(ball).sum() == ball.sum()
    assert mahotas.croptobbox(ball,border=2).sum() == ball.sum()
    assert mahotas.croptobbox(ball,border=256).sum() == ball.sum()
    assert mahotas.croptobbox(ball,border=256).size == ball.size

