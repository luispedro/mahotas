import mahotas.segmentation
import numpy as np
import mahotas
def test_slic():
    f = mahotas.imread('mahotas/demos/data/luispedro.jpg')
    segmented, n = mahotas.segmentation.slic(f)
    assert segmented.shape == (f.shape[0], f.shape[1])
    assert segmented.max() == n
