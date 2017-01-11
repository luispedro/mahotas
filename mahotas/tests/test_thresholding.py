# Copyright 2011-2013 Luis Pedro Coelho <luis@luispedro.org>
# License: MIT

import numpy as np
from mahotas.thresholding import otsu, rc, bernsen, gbernsen
from mahotas.histogram import fullhistogram

def slow_otsu(img, ignore_zeros=False):
    hist = fullhistogram(img)
    hist = hist.astype(np.double)
    Hsum = img.size - hist[0]
    if ignore_zeros:
        hist[0] = 0
    if Hsum == 0:
        return 0
    Ng = len(hist)

    nB = np.cumsum(hist)
    nO = nB[-1]-nB
    mu_B = 0
    mu_O = np.dot(np.arange(Ng), hist)/ Hsum
    best = nB[0]*nO[0]*(mu_B-mu_O)*(mu_B-mu_O)
    bestT = 0

    for T in range(1, Ng):
        if nB[T] == 0: continue
        if nO[T] == 0: break
        mu_B = (mu_B*nB[T-1] + T*hist[T]) / nB[T]
        mu_O = (mu_O*nO[T-1] - T*hist[T]) / nO[T]
        sigma_between = nB[T]*nO[T]*(mu_B-mu_O)*(mu_B-mu_O)
        if sigma_between > best:
            best = sigma_between
            bestT = T
    return bestT

def test_otsu_fast():
    np.random.seed(120)
    for i in range(12):
        A = 32*np.random.rand(128,128)
        A = A.astype(np.uint8)
        fast = otsu(A)
        slow = slow_otsu(A)
        assert fast == slow

def test_thresholding():
    np.random.seed(123)
    A = np.random.rand(128,128)
    A[24:48,24:48] += 4 * np.random.rand(24,24)
    A *= 255//A.max()
    A = A.astype(np.uint8)
    def tm(method):
        T = method(A)
        assert (A > T)[24:48,24:48].mean() > .5
        assert (A > T)[:24,:24].mean() < .5
        assert (A > T)[48:,:].mean() < .5
        assert (A > T)[:,48:].mean() < .5
    yield tm, otsu
    yield tm, rc


def test_nozeros():
    np.seterr(all='raise')
    np.random.seed(22)
    A = (np.random.rand(100,100)*50).astype(np.uint8)+201
    assert rc(A) > 200
    assert otsu(A) > 200

def test_ignore_zeros():
    np.seterr(all='raise')
    np.random.seed(22)
    A = np.zeros((1024,24), np.uint8)
    A[:24,:24] = np.random.randint(100, 200, size=(24,24))
    assert rc(A) < 100
    assert otsu(A) < 100
    assert rc(A, ignore_zeros=1) > 100
    assert otsu(A, ignore_zeros=1) > 100

def test_zero_image():
    A = np.zeros((16,16), np.uint8)
    def tm(method):
        assert method(A, ignore_zeros=0) == 0
        assert method(A, ignore_zeros=1) == 0
    yield tm, rc
    yield tm, otsu

def test_soft_threhold():
    from mahotas.thresholding import soft_threshold

    np.random.seed(223)
    for i in range(4):
        f = np.random.randint(-256,256, size=(128,128,4))
        fo = f.copy()
        t = soft_threshold(f, 16)

        assert not np.all(fo == t)
        assert np.all(t[np.abs(f) < 16] == 0)
        assert t.max() == f.max()-16
        assert t.min() == f.min()+16
        assert np.all( (np.abs(f) <= 16) | (np.abs(f)-16 == np.abs(t)))


def test_bernsen():
    np.random.seed(120)
    for i in range(4):
        f = 32*np.random.rand(40,68)
        f = f.astype(np.uint8)
        b = bernsen(f, 8, 15)
        assert f.shape == b.shape
        b = bernsen(f, 8, 15, 34)
        assert f.shape == b.shape

def test_gbernsen():
    np.random.seed(120)
    for i in range(4):
        f = 32*np.random.rand(64,96)
        f = f.astype(np.uint8)
        b = gbernsen(f, np.ones((3,3), bool), 15, 145)
        assert f.shape == b.shape
