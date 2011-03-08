import numpy as np
import mahotas
import sys

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


def test_mix_types():
    f = np.zeros((64,64), np.uint16)
    f += np.indices(f.shape)[1]**2
    f += (np.indices(f.shape)[0]-23)**2
    markers = np.zeros((64,64), np.int64)
    markers[32,32] = 1
# Below used to force a crash (at least in debug mode)
    a,b = mahotas.cwatershed(f, markers, return_lines=1)


