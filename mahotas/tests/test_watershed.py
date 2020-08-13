import numpy as np
import mahotas
import mahotas as mh
import sys
import pytest

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

types = [np.uint8, np.int8, np.uint16, np.int16, np.int32, np.uint32,int,
             np.float32, np.float64, float]
if hasattr(np, 'float128'):
    types.append(np.float128)

@pytest.mark.parametrize('dtype', types)
def test_watershed(dtype):
    St = S.astype(dtype)
    Mt = M.astype(int)
    W = mahotas.cwatershed(2-St, Mt)
    if hasattr(sys, 'getrefcount'):
        assert sys.getrefcount(W) == 2
    assert np.all(W == np.array([[1, 1, 1, 1],
           [1, 1, 1, 1],
           [1, 1, 1, 1],
           [2, 2, 1, 1],
           [2, 2, 2, 2],
           [2, 2, 2, 2],
           [2, 2, 2, 2]]))


def test_float16():
    dtype = np.float16
    St = S.astype(dtype)
    Mt = M.astype(int)
    with pytest.raises(TypeError):
        mh.cwatershed(2-St,Mt)


def test_watershed2():
    S = np.zeros((100,10), np.uint8)
    markers = np.zeros_like(S)
    markers[20,2] = 1
    markers[80,2] = 2
    W = mahotas.cwatershed(S, markers)
    assert np.all( (W == 1) | (W == 2) )

def test_mismatched_array_markers():
    S = np.zeros((10,12), np.uint8)
    markers = np.zeros((8,12), np.uint8)
    markers[2,2] = 1
    markers[6,2] = 2
    with pytest.raises(ValueError):
        mahotas.cwatershed(S, markers)

def test_mix_types():
    "[watershed regression]: Mixing types of surface and marker arrays used to cause crash"
    f = np.zeros((64,64), np.uint16)
    f += (np.indices(f.shape)[1]**2).astype(np.uint16)
    f += ((np.indices(f.shape)[0]-23)**2).astype(np.uint16)
    markers = np.zeros((64,64), np.int64)
    markers[32,32] = 1
# Below used to force a crash (at least in debug mode)
    a,b = mahotas.cwatershed(f, markers, return_lines=1)


def test_overflow():
    '''[watershed regression]: Try to force overflow on the output

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


def test_float_input():
    "[watershed]: Compare floating point input with integer input"
    f = np.random.random((128,64))
    f = mh.gaussian_filter(f, 8.)
    f = mh.gaussian_filter(f, 8.)
    markers,_ = mh.label(mh.regmin(f))
    f = np.round(f * 2**30)
    wf = mh.cwatershed(f / 2**30., markers)
    w32 = mh.cwatershed(f.astype(np.int32), markers)
    assert (wf == w32).mean() > .999
