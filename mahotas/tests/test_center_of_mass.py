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
    R += np.arange(256)
    m0,m1 = mahotas.center_of_mass(R)
    n0,n1 = ndimage.center_of_mass(R)
    assert np.abs(n0 - m0) < 1.
    assert np.abs(n1 - m1) < 1.

def test_cmp_ndimage3():
    R = (255*np.random.rand(32,128,8,16)).astype(np.uint16)
    R += np.arange(16)
    m = mahotas.center_of_mass(R)
    n = ndimage.center_of_mass(R)
    p = slow_center_of_mass(R)
    assert np.abs(n - m).max() < 1.
    assert np.abs(p - m).max() < 1.

def test_simple():
    R = (255*np.random.rand(128,256)).astype(np.uint16)
    R += np.arange(256)
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

