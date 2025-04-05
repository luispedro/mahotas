from mahotas.edge import sobel
from mahotas.edge import sujoy
import pytest
import mahotas as mh
import numpy as np

def test_sobel_shape():
    A = np.arange(100*100)
    A = (A % 15)
    A = A.reshape((100,100))
    assert sobel(A).shape == A.shape
    assert sobel(A, just_filter=True).shape == A.shape

def test_sobel_zeros():
    A = np.zeros((15,100))
    assert sobel(A).shape == A.shape
    assert sobel(A).sum() == 0

def test_sobel():
    I = np.array([
            [0,0,0,0,0,0],
            [0,0,0,1,0,0],
            [0,0,0,1,0,0],
            [0,0,0,1,0,0],
            [0,0,0,1,0,0],
            [0,0,0,0,0,0]])
    E = sobel(I)
    r,c = I.shape
    for y,x in zip(*np.where(E)):
        N = [I[y,x]]
        if y > 0: N.append(I[y-1,x])
        if x > 0: N.append(I[y,x-1])
        if y < (r-1): N.append(I[y+1,x])
        if x < (c-1): N.append(I[y,x+1])
        assert len(set(N)) > 1

def test_zero_images():
    assert np.isnan(sobel(np.zeros((16,16)))).sum() == 0
    assert sobel(np.zeros((16,16)), just_filter=True).sum() == 0


def test_sobel_pure():
    f = np.random.random((64, 128))
    f2 = f.copy()
    _ = mh.sobel(f)
    assert np.all(f == f2)


def test_3d_error():
    f = np.zeros((32,16,3))
    with pytest.raises(ValueError):
        sobel(f)

def test_sujoy_shape():
    A = np.arange(100*100)
    A = (A % 15)
    A = A.reshape((100,100))
    assert sujoy(A).shape == A.shape
    assert sujoy(A, kernel_nhood=1).shape == A.shape
    assert sujoy(A, just_filter=True).shape == A.shape

def test_sujoy_zeros():
    A = np.zeros((15,100))
    assert sujoy(A).shape == A.shape
    assert sujoy(A).sum() == 0
    assert sujoy(A, kernel_nhood=1).sum() == 0

def test_sujoy():
    I = np.array([
        [0,0,0,0,0,0],
        [0,0,0,1,0,0],
        [0,0,0,1,0,0],
        [0,0,0,1,0,0],
        [0,0,0,1,0,0],
        [0,0,0,0,0,0]])
    E = sujoy(I)
    r,c = I.shape
    for y,x in zip(*np.where(E)):
        N = [I[y,x]]
        if y > 0: N.append(I[y-1,x])
        if x > 0: N.append(I[y,x-1])
        if y < (r-1): N.append(I[y+1,x])
        if x < (c-1): N.append(I[y,x+1])
        assert len(set(N)) > 1
    E = sujoy(I, kernel_nhood=1)
    r,c = I.shape
    for y,x in zip(*np.where(E)):
        N = [I[y,x]]
        if y > 0: N.append(I[y-1,x])
        if x > 0: N.append(I[y,x-1])
        if y < (r-1): N.append(I[y+1,x])
        if x < (c-1): N.append(I[y,x+1])
        assert len(set(N)) > 1

def test_zero_images1():
    assert np.isnan(sujoy(np.zeros((16,16)))).sum() == 0
    assert np.isnan(sujoy(np.zeros((16,16)), kernel_nhood=1)).sum() == 0
    assert sujoy(np.zeros((16,16)), just_filter=True).sum() == 0
    assert sujoy(np.zeros((16,16)), just_filter=True, kernel_nhood=1).sum() == 0

def test_sujoy_pure():
    f = np.random.random((64, 128))
    f2 = f.copy()
    _ = sujoy(f)
    assert np.all(f == f2)

def test_3d_error1():
    f = np.zeros((32,16,3))
    with pytest.raises(ValueError):
        sujoy(f)

def test_dog():
    im = mh.demos.load('lena')
    im = im.mean(2)
    edges = mh.dog(im)
    assert edges.shape == im.shape
    assert edges.any()
    edges1 = mh.dog(im, sigma1=1.)
    assert np.any(edges != edges1)
