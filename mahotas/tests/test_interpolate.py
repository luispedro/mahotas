from mahotas import interpolate
import numpy as np
import pytest
from pytest import raises

@pytest.mark.parametrize('order', range(6))
@pytest.mark.parametrize('mode', ['constant', 'nearest', 'mirror', 'reflect', 'wrap', 'ignore'])
def test_zero_shift_identity(order, mode):
    f = np.arange(12., dtype=np.float64).reshape((3, 4))
    shifted = interpolate.shift(f, [0, 0], order=order, mode=mode)
    np.testing.assert_allclose(shifted, f, atol=1e-12)

def test_linear_zoom_coordinates():
    # Endpoints are aligned, so five samples span coordinates 0 through 3.
    f = np.array([2., 5., 8., 11.])
    out = np.empty(5)
    interpolate.zoom(f, 1, out=out, order=1)
    np.testing.assert_allclose(out, [2., 4.25, 6.5, 8.75, 11.])

def test_bilinear_zoom_coordinates():
    y, x = np.mgrid[:4, :5]
    f = 2. + 3.*y + 5.*x
    out = np.empty((7, 9))
    interpolate.zoom(f, 1, out=out, order=1)
    yy = np.arange(7)[:, None] / 2.
    xx = np.arange(9)[None, :] / 2.
    np.testing.assert_allclose(out, 2. + 3.*yy + 5.*xx)

def test_fractional_constant_shift():
    f = np.array([2., 5., 8., 11.])
    shifted = interpolate.shift(f, .25, order=1)
    np.testing.assert_allclose(shifted, [0., 4.25, 7.25, 10.25])

@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('ndim', [1, 2])
def test_shift_rejects_nonfinite(value, ndim, monkeypatch):
    def unexpected_native_call(*args):
        raise AssertionError('nonfinite shift reached native interpolation')

    monkeypatch.setattr(interpolate._interpolate, 'zoom_shift', unexpected_native_call)
    f = np.zeros((4,) * ndim)
    shift = value if ndim == 1 else [0., value]
    with pytest.raises(ValueError, match='finite'):
        interpolate.shift(f, shift, order=1)

@pytest.mark.parametrize('readonly', [False, True])
def test_shift_preserves_coordinate_vector(readonly):
    f = np.arange(20.).reshape(4, 5)
    shift = np.array([.25, -.75])
    original = shift.copy()
    shift.flags.writeable = not readonly
    first = interpolate.shift(f, shift, order=1)
    np.testing.assert_array_equal(shift, original)
    second = interpolate.shift(f, shift, order=1)
    np.testing.assert_array_equal(second, first)

@pytest.mark.parametrize('order', range(6))
def test_zoom_last_endpoint(order):
    f = np.full((49, 55), 7.)
    out = np.empty((32, 48))
    # Constant spline coefficients isolate sampling from prefilter precision.
    interpolate.zoom(f, 1, out=out, order=order, prefilter=False)
    np.testing.assert_allclose(out, 7., atol=1e-12)

@pytest.mark.parametrize('order', range(6))
@pytest.mark.parametrize('shape', [(1, 1), (1, 7), (9, 1)])
def test_zoom_singleton_axis(order, shape):
    f = np.full(shape, 7.)
    out = np.empty((1, 5))
    interpolate.zoom(f, 1, out=out, order=order)
    np.testing.assert_allclose(out, 7., atol=1e-12)

@pytest.mark.parametrize('mode', ['constant', 'ignore', 'nearest', 'mirror', 'reflect', 'wrap'])
@pytest.mark.parametrize('order', range(6))
@pytest.mark.parametrize('prefilter', [False, True])
def test_shift_boundary_coordinates(mode, order, prefilter):
    ndimage = pytest.importorskip('scipy.ndimage')
    f = np.array([2., 5., 8., 11.])
    for shift in [-8.4, -.4, .25, 7.25]:
        expected = ndimage.shift(f, shift, mode='constant' if mode == 'ignore' else mode,
                                 order=order, prefilter=prefilter)
        actual = interpolate.shift(f, shift, mode=mode, order=order, prefilter=prefilter)
        np.testing.assert_allclose(actual, expected, atol=2e-6)

@pytest.mark.parametrize('mode', ['constant', 'ignore', 'nearest', 'mirror', 'reflect', 'wrap'])
@pytest.mark.parametrize('shift', [3e9, -3e9, 1e300, -1e300])
def test_large_finite_shift(mode, shift):
    f = np.array([5., 6., 7., 8.])
    if mode in ('constant', 'ignore'):
        expected = np.zeros_like(f)
    elif mode == 'nearest':
        expected = np.full_like(f, f[0] if shift > 0 else f[-1])
    else:
        # Reduce the periodic shift before calling an independent interpolator.
        ndimage = pytest.importorskip('scipy.ndimage')
        period = {'mirror': 6., 'reflect': 8., 'wrap': 3.}[mode]
        coordinates = np.fmod(np.arange(4.) - shift, period)
        if mode == 'wrap':
            coordinates = np.where(coordinates <= 0, coordinates + period, coordinates) if shift > 0 else coordinates
        expected = ndimage.map_coordinates(f, [coordinates], mode=mode, order=1)
    actual = interpolate.shift(f, shift, mode=mode, order=1)
    np.testing.assert_allclose(actual, expected)

@pytest.mark.parametrize('mode', ['constant', 'ignore', 'nearest', 'mirror', 'reflect', 'wrap'])
@pytest.mark.parametrize('order', [1, 3])
def test_zoom_rejects_empty_input(mode, order):
    with pytest.raises(ValueError, match='empty'):
        interpolate.zoom(np.empty((0, 4)), 1, out=np.empty((3, 4)), mode=mode, order=order)

def test_ignore_shift_outside():
    f = np.arange(2000.)
    actual = interpolate.shift(f, 500.5, order=1, mode='ignore')
    np.testing.assert_array_equal(actual[:501], 0.)
    np.testing.assert_allclose(actual[501:], np.arange(1499.) + .5)

def test_near_integer_constant_boundary():
    actual = interpolate.shift([2., 5., 8., 11.], -1e-12, order=1)
    np.testing.assert_allclose(actual, [2., 5., 8., 0.], atol=4e-12)

def test_native_zoom_retains_requested_coordinates():
    f = np.array([2., 5., 8., 11.])
    out = np.empty(3)
    interpolate._interpolate.zoom_shift(f, np.array([.25]), None, out, 1, 4, 0.)
    np.testing.assert_allclose(out, [2., 2.75, 3.5])

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
                                    (interpolate.spline_filter1d, 0),
                                    (interpolate.spline_filter1d, 1),
                                    (interpolate.spline_filter, 0),
                                    (interpolate.spline_filter, 1)])
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

def test_zoom_order0():
    f = np.zeros((64, 64))
    f[16:48, 16:48] = 200
    out = interpolate.zoom(f, 0.5, order=0)
    assert out.shape == (32, 32)
    # Nearest-neighbor: center block should be preserved
    assert out[12, 12] == 200
    assert out[0, 0] == 0

@pytest.mark.parametrize('order', [0, 1, 3])
def test_shift_non_contiguous(order):
    # https://github.com/luispedro/mahotas/issues/124
    im = np.arange(64*64, dtype=np.float64).reshape((64,64))
    for view in (im[:, ::2], np.asfortranarray(im), im.T):
        expected = interpolate.shift(np.ascontiguousarray(view), [1.5, 2], order=order)
        assert np.allclose(interpolate.shift(view, [1.5, 2], order=order), expected)

@pytest.mark.parametrize('order', [0, 1, 3])
def test_zoom_non_contiguous(order):
    im = np.arange(64*64, dtype=np.float64).reshape((64,64))
    for view in (im[:, ::2], np.asfortranarray(im), im.T):
        expected = interpolate.zoom(np.ascontiguousarray(view), 0.5, order=order)
        assert np.allclose(interpolate.zoom(view, 0.5, order=order), expected)

def test_zoom_out_ndim_mismatch():
    im = np.zeros((49, 55))
    with raises(ValueError):
        interpolate.zoom(im, [1., 1.], out=np.empty((64,)))
    with raises(ValueError):
        interpolate.zoom(im, [1., 1.], out=np.empty((64, 64, 3)))

def test_zoom_shift_out_ndim_mismatch():
    from mahotas import _interpolate
    im = np.zeros((49, 55))
    zoom = np.ones(2)
    with raises(ValueError):
        _interpolate.zoom_shift(im, zoom, None, np.empty((64,)), 1, interpolate.mode2int['nearest'], 0.)
