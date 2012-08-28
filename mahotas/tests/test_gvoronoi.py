import numpy as np
from mahotas.segmentation import gvoronoi

def scipy_gvoronoi(labeled):
    from scipy import ndimage
    L1,L2 = ndimage.distance_transform_edt(labeled== 0, return_distances=False, return_indices=True)
    return labeled[L1,L2]


def test_compare_w_scipy():
    np.random.seed(2322)
    for i in range(8):
        labeled = np.zeros((128,128))
        for p in range(16):
            y = np.random.randint(128)
            x = np.random.randint(128)
            labeled[y,x] = p+1
        sp = scipy_gvoronoi(labeled)
        mh = gvoronoi(labeled)
        assert np.all(sp == mh)

def test_gvoronoi():
    labeled = np.zeros((128,128))
    labeled[0,0] = 1
    labeled[-1,-1] = 2
    regions = gvoronoi(labeled)
    Y,X = np.where(regions == 1)
    assert np.all(Y+X < 128)

