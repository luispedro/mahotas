import numpy as np
import mahotas.texture
import mahotas._texture
from nose.tools import raises

def test__cooccurence():
    cooccurence = mahotas._texture.cooccurence
    f = np.array([
          [0,1,1,1],
          [0,0,1,1],
          [2,2,2,2],
        ])
    Bc = np.zeros((3,3), f.dtype)
    Bc[1,2] = 1
    res = np.zeros((5,5), np.int32)
    cooccurence(f, res, Bc, 0)
    assert res[0,0] == 1
    assert res[0,1] == 2
    assert res[1,0] == 0
    assert res[1,1] == 3
    assert res[2,2] == 3
    assert not np.any(res[2,:2])
    assert not np.any(res[:2,2])
    res[:3,:3] = 0
    assert not np.any(res)

    res = np.zeros((5,5), np.int32)
    Bc = np.zeros((3,3), f.dtype)
    Bc[2,2] = 1
    cooccurence(f, res, Bc, 0)
    assert res[0,0] == 1
    assert res[0,1] == 0
    assert res[0,2] == 2
    assert res[1,0] == 0
    assert res[1,1] == 2
    assert res[1,2] == 1
    res[:3,:3] = 0
    assert not np.any(res)


def brute_force(f, dy, dx):
    res = np.zeros((f.max()+1, f.max() + 1), np.double)
    for y in xrange(f.shape[0]):
        for x in xrange(f.shape[1]):
            if 0 <= y + dy < f.shape[0] and \
                0 <= x + dx < f.shape[1]:
                res[f[y,x], f[y +dy,x+dx]] += 1
    return res

def brute_force3(f, dy, dx, dz):
    res = np.zeros((f.max()+1, f.max() + 1), np.double)
    for y in xrange(f.shape[0]):
        for x in xrange(f.shape[1]):
            for z in xrange(f.shape[2]):
                if 0 <= y + dy < f.shape[0] and \
                    0 <= x + dx < f.shape[1] and \
                    0 <= z + dz < f.shape[2]:
                    res[f[y,x,z], f[y +dy,x+dx,z+dz]] += 1
    return res


def brute_force_sym(f, dy, dx):
    cmat = brute_force(f, dy, dx)
    return (cmat + cmat.T)

def test_cooccurence():
    np.random.seed(222)
    f = np.random.rand(32, 32)
    f = (f * 255).astype(np.int32)

    assert np.all(mahotas.texture.cooccurence(f, 0, symmetric=False) == brute_force(f, 0, 1))
    assert np.all(mahotas.texture.cooccurence(f, 1, symmetric=False) == brute_force(f, 1, 1))
    assert np.all(mahotas.texture.cooccurence(f, 2, symmetric=False) == brute_force(f, 1, 0))
    assert np.all(mahotas.texture.cooccurence(f, 3, symmetric=False) == brute_force(f, 1, -1))

    assert np.all(mahotas.texture.cooccurence(f, 0, symmetric=1) == brute_force_sym(f, 0, 1))
    assert np.all(mahotas.texture.cooccurence(f, 1, symmetric=1) == brute_force_sym(f, 1, 1))
    assert np.all(mahotas.texture.cooccurence(f, 2, symmetric=1) == brute_force_sym(f, 1, 0))
    assert np.all(mahotas.texture.cooccurence(f, 3, symmetric=1) == brute_force_sym(f, 1, -1))

def test_cooccurence3():
    np.random.seed(222)
    f = np.random.rand(32, 32, 8)
    f = (f * 255).astype(np.int32)

    for di, (d0,d1,d2) in enumerate(mahotas.texture._3d_deltas):
        assert np.all(mahotas.texture.cooccurence(f, di, symmetric=False) == brute_force3(f, d0, d1, d2))

def test_haralick():
    np.random.seed(123)
    f = np.random.rand(1024, 1024)
    f = (f * 255).astype(np.int32)
    feats = mahotas.texture.haralick(f)
    assert not np.any(np.isnan(feats))

def test_haralick3():
    np.random.seed(123)
    f = np.random.rand(34, 12, 8)
    f = (f * 255).astype(np.int32)
    feats = mahotas.texture.haralick(f)
    assert not np.any(np.isnan(feats))


def test_single_point():
    A = np.zeros((5,5), np.uint8)
    A[2,2]=12
    assert not np.any(np.isnan(mahotas.texture.cooccurence(A,0)))

@raises(TypeError)
def test_float_cooccurence():
    A = np.zeros((5,5), np.float32)
    A[2,2]=12
    mahotas.texture.cooccurence(A,0)

@raises(TypeError)
def test_float_haralick():
    A = np.zeros((5,5), np.float32)
    A[2,2]=12
    mahotas.texture.haralick(A)

def test_haralick3d():
    np.random.seed(22)
    img = mahotas.stretch(255*np.random.rand(20,20,4))
    features = mahotas.texture.haralick(img)
    assert features.shape == (13,13)

    features = mahotas.texture.haralick(img[:,:,0])
    assert features.shape == (4,13)


