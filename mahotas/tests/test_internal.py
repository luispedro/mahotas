import numpy as np
from mahotas.internal import _get_output, _get_axis
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
    for i in xrange(len(f.shape)):
        assert i == _get_axis(f, i, 'test')
    for i in xrange(len(f.shape)):
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



from mahotas.internal import _normalize_sequence
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

