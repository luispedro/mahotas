from mahotas import interpolate
import numpy as np

def test_spline_filter1d_smoke():
    f  = (np.arange(64*64, dtype=np.intc) % 64).reshape((64,64)).astype(np.float64)
    f2 =interpolate.spline_filter1d(f,2,0)
    assert f.shape == f2.shape

def test_spline_filter_smoke():
    f  = (np.arange(64*64, dtype=np.intc) % 64).reshape((64,64)).astype(np.float64)
    f2 = interpolate.spline_filter(f,3)
    assert f.shape == f2.shape
