import mahotas
import numpy as np
from mahotas.colors import rgb2xyz, rgb2lab, xyz2rgb

def test_colors():
    f = mahotas.imread('mahotas/demos/data/luispedro.jpg')
    lab = rgb2lab(f)
    assert np.max(np.abs(lab)) <= 100.
    assert np.max(np.abs(xyz2rgb(rgb2xyz(f)) - f)) < 1.
    lab8 = rgb2lab(f, dtype=np.uint8)
    assert lab.dtype != np.uint8
    assert lab8.dtype == np.uint8
