from mahotas import distance
import numpy as np
def _slow_dist(bw, metric):
    sd = np.empty(bw.shape, np.double)
    sd.fill(np.inf)
    Y,X = np.indices(bw.shape)
    for y,x in zip(*np.where(~bw)):     
        sd = np.minimum(sd, (Y-y)**2 + (X-x)**2)
    if metric == 'euclidean':
        sd = np.sqrt(sd)
    return sd

def _slow_dist4d(bw, metric):
    sd = np.empty(bw.shape, np.double)
    sd.fill(np.inf)
    Y,X,W,Z = np.indices(bw.shape)
    for y,x,w,z in zip(*np.where(~bw)):     
        sd = np.minimum(sd, (Y-y)**2 + (X-x)**2 + (W-w)**2 + (Z-z)**2)
    if metric == 'euclidean':
        sd = np.sqrt(sd)
    return sd


def compare_slow(bw):
    for metric in ('euclidean', 'euclidean2'):
        f = distance(bw, metric)
        sd = _slow_dist(bw, metric)
        assert np.all(f == sd)

def test_distance():
    bw = np.ones((256, 256), bool)
    bw[100, 100] = 0
    yield compare_slow, bw

    bw[100:110, 100:110] = 0
    yield compare_slow, bw

    bw[200:210, 200:210] = 0
    yield compare_slow, bw


def test_uint8():
    # This did not work correctly in 0.9.5
    a8 = np.zeros((5,5), dtype=np.uint8)
    ab = np.zeros((5,5), dtype=bool)
    assert np.all(distance(a8) == distance(ab))


def test_4d():
    np.random.seed(324)
    for _ in range(16):
        binim = np.random.random((4,8,4,6)) > .5
        dist = distance(binim)
        assert dist.shape == binim.shape
        assert np.all(dist[~binim] == 0)
        assert np.all(dist == _slow_dist4d(binim, 'euclidean2'))
