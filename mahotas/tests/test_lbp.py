import numpy as np
import mahotas._lbp
import mahotas.thresholding
from mahotas.lbp import lbp

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
    assert len(set(mahotas._lbp.map(np.arange(256,dtype=np.uint32), 8))) == 36


def test_positives():
    np.random.seed(23)
    f = np.random.random_sample((256,256))
    lbps = mahotas.lbp.lbp(f, 4, 8)
    assert len(np.where(lbps == 0)[0]) < 2
    assert lbps.sum() == f.size
