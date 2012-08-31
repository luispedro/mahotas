# Copyright (C) 2011-2012, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT (see COPYING file)
import numpy as np

def _get_output(array, out, fname, dtype=None, output=None):
    '''
    output = _get_output(array, out, fname, dtype=None, output=None)

    Implements the mahotas output convention:
        (1) if `out` is None, return np.empty(array.shape, array.dtype)
        (2) else verify that output is of right size, shape, and contiguous

    Parameters
    ----------
    array : ndarray
    out : ndarray or None
    fname : str
        Function name. Used in error messages

    Returns
    -------
    output : ndarray
    '''
    if dtype is None:
        dtype = array.dtype
    if output is not None:
        import warnings
        warnings.warn('Using deprecated `output` argument in function `%s`. Please use `out` in the future.' % fname, DeprecationWarning)
        if out is not None:
            warnings.warn('Using both `out` and `output` in function `%s`' % fname)
        else:
            out = output
    if out is None:
        return np.empty(array.shape, dtype)
    if out.dtype != dtype:
        raise ValueError(
            'mahotas.%s: `out` has wrong type (out.dtype is %s; expected %s)' %
                (fname, out.dtype, dtype))
    if out.shape != array.shape:
        raise ValueError('mahotas.%s: `out` has wrong shape' % fname)
    if not out.flags.contiguous:
        raise ValueError('mahotas.%s: `out` is not c-array' % fname)
    return out

def _get_axis(array, axis, fname):
    '''
    axis = _get_axis(array, axis, fname)

    Checks that ``axis`` is a valid axis of ``array`` and normalises it.

    Parameters
    ----------
    array : ndarray
    axis : int
    fname : str
        Function name. Used in error messages

    Returns
    -------
    axis : int
        The positive index of the axis to use
    '''
    if axis < 0:
        axis += len(array.shape)
    if not (0 <= axis < len(array.shape)):
        raise ValueError('mahotas.%s: `axis` is out of bounds' % fname)
    return axis

def _normalize_sequence(array, value, fname):
    '''
    values = _normalize_sequence(array, value, fname)

    If `value` is a sequence, checks that it has an element for each dimension
    of `array`. Otherwise, returns a sequence that repeats `value` once for
    each dimension of array.

    Parameters
    ----------
    array : ndarray
    value : sequence or scalar
    fname : str
        Function name. Used in error messages

    Returns
    -------
    values : sequence
    '''
    try:
        value = list(value)
    except TypeError:
        return [value for s in array.shape]
    if len(value) != array.ndim:
        raise ValueError('mahotas.%s: argument is sequence, but has wrong size (%s for an array of %s dimensions' % (fname, len(value), array.ndim))
    return value

def _verify_is_floatingpoint_type(A, function_name):
    '''
    _verify_is_integer_type(array, "function")

    Checks that ``A`` is a floating-point array. If it is not, it raises
    ``TypeError``.

    Parameters
    ----------
    A : ndarray
    function_name : str
        Used for error messages
    '''
    if not np.issubdtype(A.dtype, np.float):
        raise TypeError('mahotas.%s: This function only accepts floating-point types (passed array of type %s)' % (function_name, A.dtype))

def _verify_is_integer_type(A, function_name):
    '''
    _verify_is_integer_type(array, "function")

    Checks that ``A`` is an integer array. If it is not, it raises
    ``TypeError``.

    Parameters
    ----------
    A : ndarray
    function_name : str
        Used for error messages
    '''
    int_types=[
                np.bool,
                np.uint8,
                np.int8,
                np.uint16,
                np.int16,
                np.uint32,
                np.int32,
                np.int64,
                np.uint64,
                ]
    if A.dtype not in int_types:
        raise TypeError('mahotas.%s: This function only accepts integer types (passed array of type %s)' % (function_name, A.dtype))


def _as_floating_point_array(array):
    '''
    array = _as_floating_point_array(array)

    Returns (possibly a copy) of array as a floating-point array
    '''
    array = np.asanyarray(array)
    if not np.issubdtype(array.dtype, np.float_):
        return array.astype(np.double)
    return array
