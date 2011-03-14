import numpy as np
from mahotas.segmentation import gvoronoi
def test_gvoronoi():
    labeled = np.zeros((128,128))
    labeled[0,0] = 1
    labeled[-1,-1] = 2
    regions = gvoronoi(labeled)
    Y,X = np.where(regions == 1)
    assert np.all(Y+X < 128)

