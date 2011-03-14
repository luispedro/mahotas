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
