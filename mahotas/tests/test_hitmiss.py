import mahotas
import numpy as np


def slow_hitmiss(A, Bc):
    res = np.zeros_like(A)
    for y in range(1,A.shape[0]-1):
        for x in range(1,A.shape[1]-1):
            value = 1
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    ny = y + dy
                    nx = x + dx
                    if Bc[dy+1, dx + 1] != 2 and Bc[dy+1, dx+1] != A[ny, nx]:
                        value = 0
            res[y,x] = value
    return res

def test_hitmiss():
    A = np.zeros((100,100), np.bool_)
    Bc = np.array([
        [0,1,2],
        [0,1,1],
        [2,1,1]])
    mahotas.morph.hitmiss(A,Bc)
    assert not mahotas.morph.hitmiss(A,Bc).sum()

    A[4:7,4:7] = np.array([
        [0,1,1],
        [0,1,1],
        [0,1,1]])
    assert mahotas.morph.hitmiss(A,Bc).sum() == 1
    assert mahotas.morph.hitmiss(A,Bc)[5,5]


def test_hitmiss_against_slow():
    np.random.seed(222)
    for i in range(4):
        A = np.random.rand(100,100)
        A = (A > .3)
        Bc = np.array([
            [0,1,2],
            [0,1,1],
            [2,1,1]])
        W = mahotas.morph.hitmiss(A,Bc)
        assert np.all(W == slow_hitmiss(A, Bc))


def test_hitmiss_types():
    f = np.zeros((16,16), np.uint8)
    f[8:12,8:12] = 1
    Bc = np.array([[1, 1, 2],[1,1,2],[0,0,0]], dtype=np.int32)
    assert np.sum(mahotas.morph.hitmiss(f,Bc))
    Bc = np.array([[1, 1, 2],[1,1,2],[0,0,0]], dtype=np.int64)
    assert np.sum(mahotas.morph.hitmiss(f,Bc))

