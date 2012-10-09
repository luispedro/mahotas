import numpy as np
import mahotas.labeled
from nose.tools import raises

def test_border():
    labeled = np.zeros((32,32), np.uint8)
    labeled[8:11] = 1
    labeled[11:14] = 2
    labeled[14:17] = 3
    labeled[10,8:] = 0
    b12 = mahotas.labeled.border(labeled, 1, 2)
    YX = np.where(b12)
    YX = np.array(YX).T
    b13 = mahotas.labeled.border(labeled, 1, 3)

    assert not np.any(b13)
    assert np.any(b12)
    assert (11,0) in YX
    assert (11,1) in YX
    assert (12,1) in YX
    assert (12,9) not in YX

    b13 = mahotas.labeled.border(labeled, 1, 3, always_return=0)
    assert b13 is None

def _included(a,b):
    assert np.sum(a&b) == a.sum()

def test_borders():
    labeled = np.zeros((32,32), np.uint8)
    labeled[8:11] = 1
    labeled[11:14] = 2
    labeled[14:17] = 3
    labeled[10,8:] = 0
    borders = mahotas.labeled.borders(labeled)
    _included(mahotas.labeled.border(labeled,1,2), borders)
    _included(mahotas.labeled.border(labeled,1,23), borders)
    _included(mahotas.labeled.border(labeled,1,3), borders)
    _included(mahotas.labeled.border(labeled,2,3), borders)

    union = np.zeros_like(borders)
    for i in range(4):
        for j in range(4):
            if i != j:
                union |= mahotas.labeled.border(labeled, i, j)

    assert np.all(union == borders)


def slow_labeled_sum(array, labeled):
    return np.array([
            np.sum(array * (labeled == i))
            for i in range(labeled.max()+1)
        ])

def slow_labeled_max(array, labeled):
    return np.array([
            np.max(array * (labeled == i))
            for i in range(labeled.max()+1)
        ])

def slow_labeled_min(array, labeled):
    return np.array([
            np.min(array * (labeled == i))
            for i in range(labeled.max()+1)
        ])

def test_sum_labeled():
    np.random.seed(334)
    for i in range(16):
        f = np.random.random_sample((64,128))
        labeled = np.zeros(f.shape, dtype=np.intc)
        labeled += 8 * np.random.random_sample(labeled.shape).astype(np.intc)
        fast = mahotas.labeled.labeled_sum(f, labeled)
        slow = slow_labeled_sum(f, labeled)
        assert np.allclose(fast, slow)

def test_max_labeled():
    np.random.seed(334)
    for i in range(16):
        f = np.random.random_sample((64,128))
        labeled = np.zeros(f.shape, dtype=np.intc)
        labeled += 8 * np.random.random_sample(labeled.shape).astype(np.intc)
        fast = mahotas.labeled.labeled_max(f, labeled)
        slow = slow_labeled_max(f, labeled)
        assert np.allclose(fast, slow)

def test_min_labeled():
    np.random.seed(334)
    for i in range(16):
        f = np.random.random_sample((64,128))
        labeled = np.zeros(f.shape, dtype=np.intc)
        labeled += 8 * np.random.random_sample(labeled.shape).astype(np.intc)
        fast = mahotas.labeled.labeled_min(f, labeled)
        slow = slow_labeled_min(f, labeled)
        assert np.allclose(fast, slow)

def slow_labeled_size(labeled):
    return np.array([
            np.sum(labeled == i)
            for i in range(labeled.max()+1)
        ])


def test_size_labeled():
    np.random.seed(334)
    for i in range(16):
        labeled = np.zeros((64,125), dtype=np.intc)
        labeled += 8 * np.random.random_sample(labeled.shape).astype(np.intc)
        fast = mahotas.labeled.labeled_size(labeled)
        slow = slow_labeled_size(labeled)
        assert np.all(fast == slow)

def test_remove_bordering():
    np.random.seed(343)
    for i in range(4):
        labeled,_ = mahotas.label(np.random.random_sample((128,64)) > .7)
        removed = mahotas.labeled.remove_bordering(labeled)
        assert np.all(removed[0] == 0)
        assert np.all(removed[-1] == 0)
        assert np.all(removed[:,0] == 0)
        assert np.all(removed[:,-1] == 0)

        removed2 = np.zeros_like(removed)
        mahotas.labeled.remove_bordering(labeled, out=removed2)
        assert np.all(removed2 == removed)

@raises(ValueError)
def test_check_array_labeled_not_int():
    arr = np.zeros((4,4))
    lab = np.zeros((4,4), dtype=np.float32)
    mahotas.labeled._check_array_labeled(arr, lab, 'testing')

@raises(ValueError)
def test_check_array_labeled_not_same_shape():
    arr = np.zeros((4,7))
    lab = np.zeros((4,3), dtype=np.intc)
    mahotas.labeled._check_array_labeled(arr, lab, 'testing')
