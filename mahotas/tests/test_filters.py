from mahotas import _filters
from nose.tools import raises

@raises(ValueError)
def test_bad_mode():
    _filters._check_mode('nayrest', 0., 'f')

@raises(NotImplementedError)
def test_cval_not_zero():
    _filters._check_mode('constant', 1.2, 'f')

def test_good_mode():
    for mode in _filters.modes:
        _filters._check_mode(mode, 0., 'f')

