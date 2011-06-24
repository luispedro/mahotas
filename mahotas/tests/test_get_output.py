import numpy as np
from mahotas.internal import _get_output
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
    
