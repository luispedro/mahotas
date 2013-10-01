import mahotas.features.shape
import numpy as np
import mahotas as mh
def test_eccentricity():
    D = mh.disk(32, 2)
    ecc = mahotas.features.shape.eccentricity(D)
    assert 0 <= ecc < .01

    Index = np.indices((33,33)).astype(float)
    Index -= 15
    X,Y = Index
    ellipse = ((X**2+2*Y**2) < 12**2)
    assert 0 < mahotas.features.shape.eccentricity(ellipse) < 1
