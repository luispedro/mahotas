from mahotas import interpolate
import numpy as np
import pytest
from pytest import raises

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

def test_zoom_ratio_2():
    f = np.zeros((128,128))
    f[32:64,32:64] = 128
    z0,z1  = .7,.5
    output = interpolate.zoom(f,[z0,z1])
    ratio = output.sum()/f.sum()
    assert np.abs(ratio - z0*z1) < .1

def test_shift_ratio():
    f = np.zeros((128,128))
    f[32:64,32:64] = 128
    for s in [0,1,2,3]:
        output = interpolate.shift(f,(s,s))
        ratio = output.sum()/f.sum()
        assert np.abs(ratio - 1.) < .01

@pytest.mark.parametrize('func, deg', [(interpolate.spline_filter1d, -6),
                                    (interpolate.spline_filter1d, 6),
                                    (interpolate.spline_filter, 0)])
def test_order(func, deg):
    f = np.arange(16*16).reshape((16,16))
    with raises(ValueError):
        func(f, deg)

@pytest.mark.parametrize('func, deg', [(interpolate.spline_filter1d, 3),
                                        (interpolate.spline_filter, 3)])
def test_complex(func, deg):
    f = -np.arange(16.*16).reshape((16,16))
    f = np.lib.scimath.sqrt(f)

    with raises(TypeError):
        func(f, deg)


def test_maybe_filter_error():
    with raises(ValueError):
        interpolate._maybe_filter(np.array(3), 1, 'testing', False, np.float32)

def test_short_shift():
    im = np.arange(256).reshape((16,4,-1))
    with raises(ValueError):
        interpolate.shift(im, [1,0])

def test_shift_uint8():
    im = np.arange(256).reshape((16,-1))
    im = im.astype(np.uint8)
    interpolate.shift(im, [0, np.pi/2], order=1)
