import numpy as np
from mahotas.features import texture
import mahotas as mh
import mahotas.features._texture
import pytest

def test__cooccurence():
    cooccurence = mahotas.features._texture.cooccurence
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

def test_cooccurence_errors():
    f2 = np.zeros((6,6), np.uint8)
    f3 = np.zeros((6,6,6), np.uint8)
    f4 = np.zeros((6,6,6,6), np.uint8)

    with pytest.raises(ValueError):
        texture.cooccurence(f2, -2, distance=1)

    with pytest.raises(ValueError):
        texture.cooccurence(f3, -2, distance=1)

    with pytest.raises(ValueError):
        texture.cooccurence(f2, 10, distance=1)

    with pytest.raises(ValueError):
        texture.cooccurence(f3, 17, distance=1)

    with pytest.raises(ValueError):
        texture.cooccurence(f4, 1, distance=1)



def brute_force(f, dy, dx):
    res = np.zeros((f.max()+1, f.max() + 1), np.double)
    for y in range(f.shape[0]):
        for x in range(f.shape[1]):
            if 0 <= y + dy < f.shape[0] and \
                0 <= x + dx < f.shape[1]:
                res[f[y,x], f[y +dy,x+dx]] += 1
    return res

def brute_force3(f, dy, dx, dz):
    res = np.zeros((f.max()+1, f.max() + 1), np.double)
    for y in range(f.shape[0]):
        for x in range(f.shape[1]):
            for z in range(f.shape[2]):
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

    assert np.all(texture.cooccurence(f, 0, distance=1, symmetric=False) == brute_force(f, 0, 1))
    assert np.all(texture.cooccurence(f, 1, distance=1, symmetric=False) == brute_force(f, 1, 1))
    assert np.all(texture.cooccurence(f, 2, distance=1, symmetric=False) == brute_force(f, 1, 0))
    assert np.all(texture.cooccurence(f, 3, distance=1, symmetric=False) == brute_force(f, 1, -1))

    assert np.all(texture.cooccurence(f, 0, distance=1, symmetric=1) == brute_force_sym(f, 0, 1))
    assert np.all(texture.cooccurence(f, 1, distance=1, symmetric=1) == brute_force_sym(f, 1, 1))
    assert np.all(texture.cooccurence(f, 2, distance=1, symmetric=1) == brute_force_sym(f, 1, 0))
    assert np.all(texture.cooccurence(f, 3, distance=1, symmetric=1) == brute_force_sym(f, 1, -1))

def test_cooccurence3():
    np.random.seed(222)
    f = np.random.rand(32, 32, 8)
    f = (f * 255).astype(np.int32)

    for di, (d0,d1,d2) in enumerate(texture._3d_deltas):
        assert np.all(texture.cooccurence(f, di, distance=1, symmetric=False) == brute_force3(f, d0, d1, d2))

def test_haralick():
    np.random.seed(123)
    f = np.random.rand(1024, 1024)
    f = (f * 255).astype(np.int32)
    feats = texture.haralick(f)
    assert not np.any(np.isnan(feats))

def test_haralick3():
    np.random.seed(123)
    f = np.random.rand(34, 12, 8)
    f = (f * 255).astype(np.int32)
    feats = texture.haralick(f)
    assert not np.any(np.isnan(feats))


def test_single_point():
    A = np.zeros((5,5), np.uint8)
    A[2,2]=12
    assert not np.any(np.isnan(texture.cooccurence(A, 0, distance=1)))

def test_float_cooccurence():
    A = np.zeros((5,5), np.float32)
    A[2,2]=12
    with pytest.raises(TypeError):
        texture.cooccurence(A, 0, distance=1)

def test_float_haralick():
    A = np.zeros((5,5), np.float32)
    A[2,2]=12
    with pytest.raises(TypeError):
        texture.haralick(A)

def test_haralick3d():
    np.random.seed(22)
    img = mahotas.stretch(255*np.random.rand(20,20,4))
    features = texture.haralick(img)
    assert features.shape == (13,13)

    features = texture.haralick(img[:,:,0])
    assert features.shape == (4,13)

    features = texture.haralick(img.max(0), ignore_zeros=True, preserve_haralick_bug=True, compute_14th_feature=True)
    assert features.shape == (4,14)


def test_zeros():
    zeros = np.zeros((64,64), np.uint8)
    feats = texture.haralick(zeros)
    assert not np.any(np.isnan(feats))

def test_ignore_zeros_raise():
    zeros = np.zeros((64,64), np.uint8)
    with pytest.raises(ValueError):
        texture.haralick(zeros, ignore_zeros=True)

def test_4d_image():
    with pytest.raises(ValueError):
        texture.haralick(np.arange(4**5).reshape((4,4,4,4,4)))


def rand_haralick():
    f = 255*np.random.random((128,128))
    f = f.astype(np.uint8)
    f = mh.features.haralick(f)
    return f.mean(0)
def test_feature_non_zero():
    np.random.seed(23)
    assert any(np.all(rand_haralick() != 0) for i in range(12))

def test_feature_not_same():
    np.random.seed(26)

    multiple = np.array([rand_haralick() for i in range(8)])
    assert np.all(multiple.ptp(0) > 0)



def test_return_mean_ptp():
    f = 255*np.random.random((128,128))
    f = f.astype(np.uint8)
    fs = mh.features.haralick(f)
    fs_mean = mh.features.haralick(f, return_mean=1)
    fs_mean_ptp = mh.features.haralick(f, return_mean_ptp=1)
    assert np.all(fs.mean(0) == fs_mean)
    assert np.all(fs.mean(0) == fs_mean_ptp[:fs.shape[1]])
    assert np.all(fs.ptp(0) == fs_mean_ptp[fs.shape[1]:])

def test_return_mean_ptp_xor():
    f = 255*np.random.random((128,128))
    f = f.astype(np.uint8)
    with pytest.raises(ValueError):
        mh.features.haralick(f, return_mean=1, return_mean_ptp=1)


def test_x_minus_y():
    f = 255*np.random.random((128,128))
    f = f.astype(np.uint8)
    h = mh.features.haralick(f, use_x_minus_y_variance=False)
    h2 = mh.features.haralick(f, use_x_minus_y_variance=True)
    assert np.sum(h != h2) == 4


def test_negative_values_haralick():
    # https://github.com/luispedro/mahotas/issues/72
    f = 255*np.random.random((16,16))
    f = f.astype(np.int8)
    with pytest.raises(ValueError):
        mh.features.haralick(f)


def test_int8_positive_haralick():
    # https://github.com/luispedro/mahotas/issues/72
    f = 64*np.random.random((16,16))
    f = f.astype(np.int8)
    mh.features.haralick(f) # this should be fine: all the values are positive
