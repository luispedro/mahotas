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

def test_zoom_ratio():
    f = np.zeros((128,128))
    f[32:64,32:64] = 128
    for z in [.7,.5,.2,.1]:
        output = interpolate.zoom(f,z)
        ratio = output.sum()/f.sum()
        assert np.abs(ratio - z*z) < .1

def test_shift_ratio():
    f = np.zeros((128,128))
    f[32:64,32:64] = 128
    for s in [0,1,2,3]:
        output = interpolate.shift(f,(s,s))
        ratio = output.sum()/f.sum()
        assert np.abs(ratio - 1.) < .01
