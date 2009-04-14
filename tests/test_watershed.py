import morph.morph
import numpy as np 

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
        W = morph.morph.cwatershed(2-S,M)
        assert np.all(W == np.array([[1, 1, 1, 1],
               [1, 1, 1, 1],
               [1, 1, 1, 1],
               [2, 2, 1, 1],
               [2, 2, 2, 2],
               [2, 2, 2, 2],
               [2, 2, 2, 2]]))
    for d in [np.uint8, np.int8, np.uint16, np.int16, np.int32, np.uint32,int]:
        yield cast_test, M, S, d
