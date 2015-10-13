import numpy as np
import mahotas as mh
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

def test_sum_minlength_arg():
    for _ in range(8):
        f = mh.dilate(np.random.random_sample((64,64)) > .8)
        labeled, n = mh.labeled.label(f)
        sizes = mh.labeled.labeled_sum(f, labeled)
        sizes2 = mh.labeled.labeled_sum(f, labeled, minlength=(n+23))
        assert len(sizes) == (n+1)
        assert len(sizes2) == (n+23)
        assert np.all(sizes == sizes2[:n+1])


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
def test_check_array_labeled_not_same_shape():
    arr = np.zeros((4,7))
    lab = np.zeros((4,3), dtype=np.intc)
    mahotas.labeled._as_labeled(arr, lab, 'testing')

def _nelems(arr):
    return len(set(map(int, arr.ravel())))

def test_relabel():
    np.random.seed(24)
    for i in range(8):
        f = np.random.random_sample((128,128)) > .8
        labeled, n = mahotas.labeled.label(f)
        labeled *= ( (labeled % 7) != 4)
        relabeled,new_n = mahotas.labeled.relabel(labeled)
        assert relabeled.max() == new_n
        assert (relabeled.max()+1) == _nelems(labeled)
        assert np.all( (relabeled > 0) == (labeled > 0) )
        assert not np.all(labeled == relabeled)

        for a in (8, 23, 35, 13, 213):
            assert _nelems(labeled[relabeled == a]) == 1
            assert _nelems(relabeled[labeled == a]) == 1
        mahotas.labeled.relabel(labeled, inplace=True)
        assert np.all( labeled == relabeled )


def test_remove_regions():
    np.random.seed(34)
    f = np.random.random_sample((128,128)) > .92
    labeled, n = mahotas.labeled.label(f)
    regions = [23,55,8]
    removed = mahotas.labeled.remove_regions(labeled, regions)

    for r in regions:
        assert not  np.any(removed == r)
        assert      np.any(labeled == r)
    mahotas.labeled.remove_regions(labeled, regions, inplace=True)
    assert np.all(labeled == removed)
    removed = mahotas.labeled.remove_regions(labeled, [])
    assert np.all(labeled == removed)

def test_is_same_labeling():
    np.random.seed(143)
    ell = (np.random.random((256,256))*16).astype(np.intc)
    order = np.arange(1,16)
    np.random.shuffle(order)
    order = np.insert(order, 0, 0)
    assert mh.labeled.is_same_labeling(ell, order[ell])
    ell2 = order[ell]
    ell2[ell == 0] = 1
    
    assert not mh.labeled.is_same_labeling(ell, ell2)

def test_perimeter():
    for r in (40,80,160):
        disk = mh.disk(r, 2)
        p = mahotas.labeled.perimeter(disk)
        p_exact = r*np.pi*2
        assert .9 < (p/p_exact)  < 1.1



def test_remove_regions_where():
    np.random.seed(34)
    for _ in range(4):
        f = np.random.random_sample((128,128)) > .82
        labeled, n = mh.labeled.label(f)
        relabeled = mh.labeled.remove_regions_where(labeled, mh.labeled.labeled_size(labeled) < 2)
        relabeled,_ = mh.labeled.relabel(relabeled)
        sizes = mh.labeled.labeled_size(relabeled)
        assert sizes[1:].min() >= 2

def test_remove_bordering_tuple():
    import mahotas as mh
    import numpy as np
    f = np.zeros((32,32))
    f[0,0] = 1
    f[2,4] = 2
    f.T[4,2] = 3
    f[8,8] = 4
    assert np.any(mh.labeled.remove_bordering(f) == 3)
    assert np.any(mh.labeled.remove_bordering(f, (2,4)) == 4)
    assert np.any(mh.labeled.remove_bordering(f, (2,4)) == 3)
    assert not np.any(mh.labeled.remove_bordering(f, (4,2)) == 3)

def test_as_labeled():

    from mahotas.labeled import _as_labeled
    arr = np.zeros((64,64))
    labeled = np.zeros((64,64), dtype=np.intc)
    funcname = 'testing'

    assert _as_labeled(arr, labeled, funcname, inplace=True) is labeled
    assert _as_labeled(arr, labeled, funcname) is labeled

    lab2 = _as_labeled(arr, labeled, funcname, inplace=False)
    assert lab2 is not labeled
    assert np.all(labeled == lab2)

    assert _as_labeled(arr[::2], labeled[::2], funcname, inplace=False) is not labeled
    assert _as_labeled(arr[::2], labeled[::2], funcname, inplace=None) is not labeled
    assert _as_labeled(arr[::2], labeled[::2], funcname) is not labeled

    @raises(ValueError)
    def t():
        _as_labeled(arr[::2], labeled[::2], funcname, inplace=True)
    t()

    @raises(ValueError)
    def t():
        _as_labeled(arr[::2], labeled, funcname)
    t()


def test_labeled_bbox():
    f = np.random.random((128,128))
    f = f > .8
    f,n = mh.label(f)


    result = mh.labeled.bbox(f)
    result_as = mh.labeled.bbox(f, as_slice=True)
    for _ in range(32):
        ix = np.random.randint(n+1)
        assert np.all(result[ix] == mh.bbox(f == ix))
        assert np.all(result_as[ix] == mh.bbox(f == ix, as_slice=True))

def test_labeled_bbox_zeros():
    'Issue #61'
    def nozeros_test(f):
        result = mh.labeled.bbox(f)
        result_as = mh.labeled.bbox(f, as_slice=True)
        assert not np.all(result == 0)
        for ix in range(4):
            assert np.all(result[ix] == mh.bbox(f == ix))
            assert np.all(result_as[ix] == mh.bbox(f == ix, as_slice=True))

    f = np.array([
        [2,1,1],
        [2,2,1],
        [2,2,3]])
    f3 = np.array([f])
    yield nozeros_test, f
    yield nozeros_test, f3


def test_filter_labeled():

    f = np.random.random(size=(256,256)) > .66
    labeled, nr = mh.label(f)
    no_change,no_nr = mh.labeled.filter_labeled(labeled)
    assert no_nr == nr
    assert np.all(no_change == labeled)

    no_border, border_nr = mh.labeled.filter_labeled(labeled, remove_bordering=True)
    assert nr > border_nr
    assert not np.all(no_border == labeled)

    min_size_3,nr3 = mh.labeled.filter_labeled(labeled, min_size=3)
    assert nr > nr3
    assert mh.labeled.labeled_size(min_size_3).min() == 3

    max_size_3,_ = mh.labeled.filter_labeled(labeled, max_size=3)
    assert mh.labeled.labeled_size(max_size_3)[1:].max() == 3


    all_size_3,_ = mh.labeled.filter_labeled(labeled, max_size=3, min_size=3)
    assert np.all(mh.labeled.labeled_size(all_size_3)[1:] == 3)
