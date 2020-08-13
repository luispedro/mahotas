import numpy as np
from mahotas.features import _lbp
import mahotas.thresholding
from mahotas.features import lbp
from mahotas.features.lbp import lbp_transform, lbp_names
from mahotas.tests.utils import luispedro_jpg
import pytest

def test_shape():
    A = np.arange(32*32).reshape((32,32))
    B = np.arange(64*64).reshape((64,64))
    features0 = lbp(A, 3, 12)
    features1 = lbp(B, 3, 12)
    assert features0.shape == features1.shape

def test_nonzero():
    A = np.arange(32*32).reshape((32,32))
    features = lbp(A, 3, 12)
    features_ignore_zeros = lbp(A * (A> 256), 3, 12, ignore_zeros=True)
    assert features.sum() > 0
    assert not np.all(features == features_ignore_zeros)

def test_histogram():
    A = np.arange(32*32).reshape((32,32))
    for r in (2,3,4,5):
        assert lbp(A,r,8).sum() == A.size

def test_histogram_large():
    A = np.arange(32*32).reshape((32,32))
    for r in (2,3,4,5):
        assert lbp(A,r,12).sum() == A.size


def test_map():
    assert len(set(_lbp.map(np.arange(256,dtype=np.uint32), 8))) == 36


def test_positives():
    np.random.seed(23)
    f = np.random.random_sample((256,256))
    lbps = lbp(f, 4, 8)
    assert len(np.where(lbps == 0)[0]) < 2
    assert lbps.sum() == f.size

def test_lbp_transform():

    im = luispedro_jpg().max(2)
    transformed = lbp_transform(im, 8, 4, preserve_shape=True)
    assert transformed.shape == im.shape
    assert transformed.min() >= 0
    assert transformed.max() < 2**4
    transformed = lbp_transform(im, 8, 4, preserve_shape=False)
    assert len(transformed.shape) == 1
    assert transformed.size == im.size

    np.random.seed(234)
    im *= np.random.random(im.shape) > .1
    transformed = lbp_transform(im, 8, 4, preserve_shape=False, ignore_zeros=True)
    assert len(transformed.shape) == 1
    assert transformed.size == (im.size - (im==0).sum())


def test_count_binary1s():
    from mahotas.features.lbp import count_binary1s
    assert np.all(count_binary1s(np.zeros((23,23), np.uint8)) == np.zeros((23,23)))

    np.random.seed(3499)
    arr = np.random.randint(45,size=(23,23))
    c = count_binary1s(arr)
    assert np.all((c == 0) == (arr == 0))
    assert c.shape == arr.shape
    assert np.all(c[arr == 5] == 2)
    assert np.all(c[arr == 7] == 3)
    assert np.all(c[arr == 8] == 1)
    assert np.all(c[arr == 32] == 1)

    assert np.all(count_binary1s([128]) == [1])

def test_lbp_3d():
    im = np.arange(10*20*3).reshape((10,20,3))
    with pytest.raises(ValueError):
        lbp_transform(im, 1, 8)

def test_lbp_names():
    f = np.random.random(size=(64,72))
    f *= 255
    f = f.astype(np.uint8)

    for radius,points in [(8,6),
            (8,8),(6,6),(8,4),(12,6)]:
        assert len(lbp(f, radius, points)) == len(lbp_names(radius, points))
