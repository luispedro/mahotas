import numpy as np
from mahotas.polygon import fill_polygon

def test_polygon():
    polygon = [(10,10), (10,20), (20,20)]
    canvas = np.zeros((40,40), np.bool)
    fill_polygon(polygon, canvas)
    assert canvas.sum() == (10*10-10)/2
