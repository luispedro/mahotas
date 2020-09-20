import numpy as np
import mahotas
import mahotas as mh
from mahotas import bbox

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

def test_croptobbox_3d():
    YXZ = np.indices((32,32,64), float)
    YXZ -= 8
    Y,X,Z = YXZ
    ball = ((X**2+Y**2+Z**2) < 64).astype(np.uint8)
    assert np.sum(ball) == np.sum(mh.croptobbox(ball))


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

    c0,d0,c1,d1=bbox(img, 0)
    assert c0 == a0
    assert b0 == d0
    assert c1 == a1
    assert b1 == d1

    c0,d0,c1,d1=bbox(img, 1)
    assert c0 != a0
    assert b0 != d0
    assert c1 != a1
    assert b1 != d1

def test_as_slice():
    YXZ = np.indices((32,32,64), float)
    YXZ -= 8
    Y,X,Z = YXZ
    ball = ((X**2+Y**2+Z**2) < 64).astype(np.uint8)
    s = bbox(ball, as_slice=True)
    assert ball[s].sum() == ball.sum()

def test_slice_border():
    'Test bbox(slice=True, border=6) in 2D & 3D'
    f = np.zeros((32,32), bool)
    f[8:8] = 1
    m0,M0, m1,M1 = mh.bbox(f, border=6, as_slice=False)
    sl = mh.bbox(f, border=6, as_slice=True)

    assert np.all(f[sl] == f[m0:M0, m1:M1])

    f = np.zeros((32,32, 32), bool)
    f[8:8,12:15] = 1
    m0,M0, m1,M1, m2, M2 = mh.bbox(f, border=6, as_slice=False)
    sl = mh.bbox(f, border=6, as_slice=True)

    assert np.all(f[sl] == f[m0:M0, m1:M1, m2:M2])

