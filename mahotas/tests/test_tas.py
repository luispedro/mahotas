import numpy as np
from mahotas.features import tas, pftas
import pytest

def test_tas():
    np.random.seed(22)
    f = np.random.rand(1024, 1024)
    f = (f * 255).astype(np.uint8)
    assert np.abs(tas(f).sum()-6) < 0.0001
    assert np.abs(pftas(f).sum()-6) < 0.0001

def test_tas3d():
    np.random.seed(22)
    f = np.random.rand(512, 512, 8)
    f = (f * 255).astype(np.uint8)
    assert np.abs(tas(f).sum()-6) < 0.0001
    assert np.abs(pftas(f).sum()-6) < 0.0001

def test_regression():
    np.random.seed(220)
    img = np.random.random_sample((1024,1024))
    img *= 255
    img = img.astype(np.uint8)
    features = pftas(img)
    assert not np.any(features == 0.)

def test_zero_image():
    features = pftas(np.zeros((64,64), np.uint8))
    assert not np.any(np.isnan(features))

def test_4d():
    np.random.seed(22)
    f = np.random.rand(16,16,16,16)
    f = (f * 255).astype(np.uint8)
    with pytest.raises(ValueError):
        tas(f)
