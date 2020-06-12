import numpy as np
from mahotas.internal import _get_output, _get_axis
from mahotas.internal import _normalize_sequence, _verify_is_integer_type, _verify_is_floatingpoint_type, _as_floating_point_array
from mahotas.internal import _check_3
import pytest

def test_get_output():
    f = np.arange(256).reshape((32,8))
    output = _get_output(f, None, 'test')
    assert output.dtype == f.dtype
    assert output.shape == f.shape
    out2 = _get_output(f, output, 'test')
    assert out2 is output

def test_dtype():
    f = np.arange(256).reshape((32,8))
    output = _get_output(f, None, 'test', np.float32)
    assert output.dtype == np.float32

def test_get_output_bad_shape():
    f = np.arange(256).reshape((32,8))
    output = np.zeros((16,16), f.dtype)
    with pytest.raises(ValueError):
        _get_output(f, output, 'testing')

def test_get_output_non_contiguous():
    f = np.arange(256).reshape((32,8))
    output = np.zeros((32,16), f.dtype)
    output = output[:,::2]
    assert output.shape == f.shape
    with pytest.raises(ValueError):
        _get_output(f, output, 'testing')

def test_get_output_explicit_dtype():
    f = np.arange(256).reshape((32,8))
    output = np.zeros_like(f)
    with pytest.raises(ValueError):
        _get_output(f, output, 'testing', bool)


def test_get_axis_good():
    f = np.zeros((3,4,5,3,2,2,5,3,2,4,1))
    for i in range(len(f.shape)):
        assert i == _get_axis(f, i, 'test')
    for i in range(len(f.shape)):
        assert len(f.shape)-1-i == _get_axis(f, -1-i, 'test')

def test_get_axis_off():
    f = np.zeros((3,4,5,3,2,2,5,3,2,4,1))
    for i in [ 12,
             13,
             14,
             67,
             -67,
             -len(f.shape)-1]:
        with pytest.raises(ValueError):
            _get_axis(f, i, 'test')



def test_normalize():
    f = np.arange(64)
    assert len(_normalize_sequence(f, 1, 'test')) == f.ndim
    f = f.reshape((2,2,2,-1))
    assert len(_normalize_sequence(f, 1, 'test')) == f.ndim
    f = f.reshape((2,2,2,1,1,-1))
    assert len(_normalize_sequence(f, 1, 'test')) == f.ndim

def test_normalize_sequence():
    f = np.arange(64)
    assert len(_normalize_sequence(f, [1], 'test')) == f.ndim
    f = f.reshape((16,-1))
    assert _normalize_sequence(f, [2,4], 'test') == [2,4]

def test_normalize_wrong_size():
    def check(ns, val):
        with pytest.raises(ValueError):
            _normalize_sequence(f.reshape(ns), val, 'test')
    f = np.arange(64)

    check((64,),[1,2])
    check((64,),[1,1])
    check((64,),[1,2,3,4])
    check((4,-1),[1,2,3,4])
    check((4,-1),[1])
    check((4,2,-1),[1,2])

@pytest.mark.parametrize('dtype', [np.float, np.float32, np.float64])
def test_verify_float_arr(dtype):
    arr = np.arange(1., dtype=dtype)
    _verify_is_floatingpoint_type(arr, 'test')
    with pytest.raises(TypeError):
        _verify_is_integer_type(arr, 'test')

@pytest.mark.parametrize('dtype', [np.int32, np.uint16, np.int64])
def test_verify_int_arr(dtype):
    arr = np.arange(1., dtype=dtype)
    _verify_is_integer_type(arr, 'test')
    with pytest.raises(TypeError):
        _verify_is_floatingpoint_type(arr, 'test')


@pytest.mark.parametrize('data', [
      np.arange(8, dtype=np.int8)
    , np.arange(8, dtype=np.int16)
    , np.arange(8, dtype=np.uint32)
    , np.arange(8, dtype=np.double)
    , np.arange(8, dtype=np.float32)
    , [1,2,3]
    , [[1,2],[2,3],[3,4]]
    , [[1.,2.],[2.,3.],[3.,4.]]
    ])
def test_as_floating_point_array(data):
    array = _as_floating_point_array(data)
    assert np.issubdtype(array.dtype, np.floating)

def test_check_3():
    _check_3(np.zeros((14,24,3), np.uint8), 'testing')

def test_check_3_dim4():
    with pytest.raises(ValueError):
        _check_3(np.zeros((14,24,3,5), np.uint8), 'testing')

def test_check_3_not3():
    with pytest.raises(ValueError):
        _check_3(np.zeros((14,24,5), np.uint8), 'testing')

def test_check_3_not3_dim4():
    with pytest.raises(ValueError):
        _check_3(np.zeros((14,24,5,5), np.uint8), 'testing')


def test_make_binary():
    from mahotas.internal import _make_binary
    np.random.seed(34)
    binim = np.random.random_sample((32,64)) > .25
    assert _make_binary(binim) is binim
    assert np.all( _make_binary(binim.astype(int)) == binim )
    assert np.all( _make_binary(binim.astype(int)*4) == binim )
    assert np.all( _make_binary(binim.astype(float)*3.4) == binim )
    assert np.all( _make_binary(binim* np.random.random_sample(binim.shape)) == binim )

