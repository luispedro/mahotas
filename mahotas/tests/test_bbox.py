import numpy as np
import mahotas
from mahotas import bbox
from nose.tools import raises

def test_croptobbox():
    X,Y = np.meshgrid(np.arange(16)-8,np.arange(16)-8)
    ball = ((X**2+Y**2) < 32).astype(np.uint8)
    assert mahotas.croptobbox(ball).sum() == ball.sum()
    assert mahotas.croptobbox(ball,border=2).sum() == ball.sum()
    assert mahotas.croptobbox(ball,border=256).sum() == ball.sum()
    assert mahotas.croptobbox(ball,border=256).size == ball.size
    assert mahotas.croptobbox(ball.T).sum() == ball.sum()

    assert mahotas.croptobbox(ball[::2]).sum() == ball[::2].sum()
    assert mahotas.croptobbox(ball[::2].T).sum() == ball[::2].sum()
    assert mahotas.croptobbox(ball.T, border=2).sum() == ball.sum()
    assert mahotas.croptobbox(ball.T, border=256).sum() == ball.sum()
    assert mahotas.croptobbox(ball.T, border=256).size == ball.size

def test_bbox_empty():
    assert mahotas.bbox(np.zeros((), np.bool)).shape == (0,)

def test_bbox_3():
    YXZ = np.indices((32,32,64), float)
    YXZ -= 8
    Y,X,Z = YXZ
    ball = ((X**2+Y**2+Z**2) < 64).astype(np.uint8)
    m0,M0,m1,M1,m2,M2 = mahotas.bbox(ball)

    Y,X,Z = np.where(ball)
    assert np.all(m0 <= Y)
    assert np.all(m1 <= X)
    assert np.all(m2 <= Z)
    assert np.all(M0 > Y)
    assert np.all(M1 > X)
    assert np.all(M2 > Z)


def test_bbox():
    img = np.zeros((10,10), np.uint16)
    
    a0,b0,a1,b1 = bbox(img)
    assert a0 == b0
    assert a1 == b1

    img[4,2]=1
    a0,b0,a1,b1=bbox(img)
    assert a0 == 4
    assert b0 == 5
    assert a1 == 2
    assert b1 == 3

    img[6,8]=1
    a0,b0,a1,b1=bbox(img)
    assert a0 == 4 
    assert b0 == 7 
    assert a1 == 2 
    assert b1 == 9 

    img[7,7]=1
    a0,b0,a1,b1=bbox(img)
    assert a0 == 4
    assert b0 == 8
    assert a1 == 2
    assert b1 == 9

