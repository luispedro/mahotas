import numpy as np
from mahotas.internal import _get_output, _get_axis
from mahotas.internal import _normalize_sequence, _verify_is_integer_type, _verify_is_floatingpoint_type, _as_floating_point_array
from nose.tools import raises

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

@raises(ValueError)
def test_get_output_bad_shape():
    f = np.arange(256).reshape((32,8))
    output = np.zeros((16,16), f.dtype)
    _get_output(f, output, 'testing')
    
@raises(ValueError)
def test_get_output_non_contiguous():
    f = np.arange(256).reshape((32,8))
    output = np.zeros((32,16), f.dtype)
    output = output[:,::2]
    assert output.shape == f.shape
    _get_output(f, output, 'testing')

@raises(ValueError)
def test_get_output_explicit_dtype():
    f = np.arange(256).reshape((32,8))
    output = np.zeros_like(f)
    _get_output(f, output, 'testing', bool)
    

def test_get_axis_good():
    f = np.zeros((3,4,5,3,2,2,5,3,2,4,1))
    for i in range(len(f.shape)):
        assert i == _get_axis(f, i, 'test')
    for i in range(len(f.shape)):
        assert len(f.shape)-1-i == _get_axis(f, -1-i, 'test')

def test_get_axis_off():
    f = np.zeros((3,4,5,3,2,2,5,3,2,4,1))
    @raises(ValueError)
    def index(i):
        _get_axis(f, i, 'test')
    yield index, 12
    yield index, 13
    yield index, 14
    yield index, 67
    yield index, -67
    yield index, -len(f.shape)-1
    yield raises



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
    @raises(ValueError)
    def check(ns, val):
        _normalize_sequence(f.reshape(ns), val, 'test')
    f = np.arange(64)

    check((64,),[1,2])
    check((64,),[1,1])
    check((64,),[1,2,3,4])
    check((4,-1),[1,2,3,4])
    check((4,-1),[1])
    check((4,2,-1),[1,2])

def test_verify_int():
    @raises(TypeError)
    def check_fp(arr):
        _verify_is_integer_type(arr, 'test')

    def check_int(arr):
        _verify_is_integer_type(arr, 'test')

    yield check_fp, np.arange(1., dtype=np.float)
    yield check_fp, np.arange(1., dtype=np.float32)
    yield check_fp, np.arange(1., dtype=np.float64)

    yield check_int, np.arange(1, dtype=np.int32)
    yield check_int, np.arange(1, dtype=np.uint16)
    yield check_int, np.arange(1, dtype=np.int64)

def test_verify_fp():
    def check_fp(arr):
        _verify_is_floatingpoint_type(arr, 'test')

    @raises(TypeError)
    def check_int(arr):
        _verify_is_floatingpoint_type(arr, 'test')

    yield check_fp, np.arange(1., dtype=np.float)
    yield check_fp, np.arange(1., dtype=np.float32)
    yield check_fp, np.arange(1., dtype=np.float64)

    yield check_int, np.arange(1, dtype=np.int32)
    yield check_int, np.arange(1, dtype=np.uint16)
    yield check_int, np.arange(1, dtype=np.int64)

def test_as_floating_point_array():
    def check_arr(data):
        array = _as_floating_point_array(data)
        assert np.issubdtype(array.dtype, np.float_)

    yield check_arr, np.arange(8, dtype=np.int8)
    yield check_arr, np.arange(8, dtype=np.int16)
    yield check_arr, np.arange(8, dtype=np.uint32)
    yield check_arr, np.arange(8, dtype=np.double)
    yield check_arr, np.arange(8, dtype=np.float32)
    yield check_arr, [1,2,3]
    yield check_arr, [[1,2],[2,3],[3,4]]
    yield check_arr, [[1.,2.],[2.,3.],[3.,4.]]

