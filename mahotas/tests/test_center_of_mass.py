import numpy as np
from scipy import ndimage
import mahotas.center_of_mass

np.random.seed(2321)
def _mean_out(img, axis):
    if len(img.shape) == 2: return img.mean(1-axis)
    if axis == 0:
        return _mean_out(img.mean(1), 0)
    return _mean_out(img.mean(0), axis - 1)

def slow_center_of_mass(img):
    '''
    Returns the center of mass of img.
    '''
    xs = []
    for axis,si in enumerate(img.shape):
        xs.append(np.mean(_mean_out(img, axis) * np.arange(si)))
    xs = np.array(xs)
    xs /= img.mean()
    return xs


def test_cmp_ndimage():
    R = (255*np.random.rand(128,256)).astype(np.uint16)
    R += np.arange(256, dtype=np.uint16)
    m0,m1 = mahotas.center_of_mass(R)
    n0,n1 = ndimage.center_of_mass(R)
    assert np.abs(n0 - m0) < 1.
    assert np.abs(n1 - m1) < 1.

def test_cmp_ndimage3():
    R = (255*np.random.rand(32,128,8,16)).astype(np.uint16)
    R += np.arange(16, dtype=np.uint16)
    m = mahotas.center_of_mass(R)
    n = ndimage.center_of_mass(R)
    p = slow_center_of_mass(R)
    assert np.abs(n - m).max() < 1.
    assert np.abs(p - m).max() < 1.

def test_simple():
    R = (255*np.random.rand(128,256)).astype(np.uint16)
    R += np.arange(256, dtype=np.uint16)
    m0,m1 = mahotas.center_of_mass(R)

    assert 0 < m0 < 128
    assert 0 < m1 < 256


def test_labels():
    R = (255*np.random.rand(128,256)).astype(np.uint16)
    labels = np.zeros(R.shape, np.intc)
    labels[100:,:] += 1
    labels[100:,100:] += 1
    centres =  mahotas.center_of_mass(R, labels)
    for label,cm in enumerate(centres):
        assert np.all(cm == mahotas.center_of_mass(R * (labels == label)))



def test_zero_sum_label():
    # Labels whose pixels sum to zero should return NaN, not crash or Inf
    img = np.zeros((10, 10), dtype=np.uint8)
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[5:, :] = 1
    img[:5, :] = 10  # label 0 has values, label 1 is all zeros
    cm = mahotas.center_of_mass(img, labels)
    assert cm.shape == (2, 2)
    # label 0 should have a valid center
    assert np.all(np.isfinite(cm[0]))
    # label 1 (all zeros) should be NaN
    assert np.all(np.isnan(cm[1]))

    # Also test signed image where values cancel to zero
    img2 = np.array([[5, -5]], dtype=np.int32)
    cm2 = mahotas.center_of_mass(img2)
    assert np.all(np.isnan(cm2))

def test_labels_not_intc():
    img = np.arange(256).reshape((16,16))
    labels = img.copy()
    labels %= 3
    cm = mahotas.center_of_mass(img, labels)
    assert cm.shape == (3,2)

    labels = labels.T.copy()
    cm = mahotas.center_of_mass(img, labels.T)
    assert cm.shape == (3,2)

    labels = labels.T.copy()
    labels = labels.astype(np.uint16)
    cm = mahotas.center_of_mass(img, labels)
    assert cm.shape == (3,2)

