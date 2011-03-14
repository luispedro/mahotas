import numpy as np
import mahotas.tas
def test_tas():
    np.random.seed(22)
    f = np.random.rand(1024, 1024)
    f = (f * 255).astype(np.uint8)
    assert np.abs(mahotas.tas.tas(f).sum()-6) < 0.0001
    assert np.abs(mahotas.tas.pftas(f).sum()-6) < 0.0001

def test_tas3d():
    np.random.seed(22)
    f = np.random.rand(512, 512, 8)
    f = (f * 255).astype(np.uint8)
    assert np.abs(mahotas.tas.tas(f).sum()-6) < 0.0001
    assert np.abs(mahotas.tas.pftas(f).sum()-6) < 0.0001
