import numpy as np
import mahotas
import mahotas as mh
import sys
from nose.tools import raises

def test_watershed():
    S = np.array([
        [0,0,0,0],
        [0,1,2,1],
        [1,1,1,1],
        [0,0,1,0],
        [1,1,1,1],
        [1,2,2,1],
        [1,1,2,2]
        ])
    M = np.array([
        [0,0,0,0],
        [0,0,1,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,2,0,0],
        [0,0,0,0],
        ])
    def cast_test(M,S,dtype):
        M = M.astype(dtype)
        S = S.astype(dtype)
        W = mahotas.cwatershed(2-S,M)
        assert sys.getrefcount(W) == 2
        assert np.all(W == np.array([[1, 1, 1, 1],
               [1, 1, 1, 1],
               [1, 1, 1, 1],
               [2, 2, 1, 1],
               [2, 2, 2, 2],
               [2, 2, 2, 2],
               [2, 2, 2, 2]]))
    for d in [np.uint8, np.int8, np.uint16, np.int16, np.int32, np.uint32,int]:
        yield cast_test, M, S, d


def test_watershed2():
    S = np.zeros((100,10), np.uint8)
    markers = np.zeros_like(S)
    markers[20,2] = 1
    markers[80,2] = 2
    W = mahotas.cwatershed(S, markers)
    assert np.all( (W == 1) | (W == 2) )

@raises(ValueError)
def test_mismatched_array_markers():
    S = np.zeros((10,12), np.uint8)
    markers = np.zeros((8,12), np.uint8)
    markers[2,2] = 1
    markers[6,2] = 2
    mahotas.cwatershed(S, markers)

def test_mix_types():
    f = np.zeros((64,64), np.uint16)
    f += (np.indices(f.shape)[1]**2).astype(np.uint16)
    f += ((np.indices(f.shape)[0]-23)**2).astype(np.uint16)
    markers = np.zeros((64,64), np.int64)
    markers[32,32] = 1
# Below used to force a crash (at least in debug mode)
    a,b = mahotas.cwatershed(f, markers, return_lines=1)


def test_overflow():
    '''Test whether we can force an overflow in the output of cwatershed

    This was reported as issue #41 on github:

    https://github.com/luispedro/mahotas/issues/41
    '''
    f = np.random.random((128,64))
    f *= 255 
    f = f.astype(np.uint8)
    for max_n in [127, 240, 280]:
        markers = np.zeros(f.shape, np.int)
        for i in range(max_n):
            while True:
                a = np.random.randint(f.shape[0])
                b = np.random.randint(f.shape[1])
                if markers[a,b] == 0:
                    markers[a,b] = i + 1
                    break
                
        r = mh.cwatershed(f, markers)
        assert markers.max() == max_n
        assert r.max() == max_n


def test_watershed_labeled():
    import mahotas as mh
    S = np.array([
        [0,0,0,0],
        [0,1,2,1],
        [1,1,1,1],
        [0,0,1,0],
        [1,1,1,1],
        [1,2,2,1],
        [1,1,2,2]
        ])
    M = np.array([
        [0,0,0,0],
        [0,0,1,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0],
        [0,2,0,0],
        [0,0,0,0],
        ])
    labeled = mh.cwatershed(S, M)
    sizes = mh.labeled.labeled_sum(S, labeled)
    assert len(sizes) == labeled.max() + 1
