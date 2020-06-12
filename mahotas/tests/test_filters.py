from mahotas import _filters
import pytest

def test_bad_mode():
    with pytest.raises(ValueError):
        _filters._check_mode('nayrest', 0., 'f')

def test_cval_not_zero():
    with pytest.raises(NotImplementedError):
        _filters._check_mode('constant', 1.2, 'f')

def test_good_mode():
    for mode in _filters.modes:
        _filters._check_mode(mode, 0., 'f')

