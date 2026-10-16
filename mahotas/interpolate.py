# This module was adapted from scipy.ndimage and retains its license
# Copyright (C) 2003-2005 Peter J. Verveer
# Copyright (C) 2011-2026 Luis Pedro Coelho
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. The name of the author may not be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
Interpolation
-------------

This module was adapted from scipy.ndimage
'''

from typing import Any, overload

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from ._typing import Array, BorderMode, ScalarT
from . import internal
from . import _interpolate
from ._filters import mode2int, modes, _check_mode

def _check_interpolate(array: ArrayLike, order: int, funcname: str) -> Array:
    if not (0 <= order <= 5):
        raise ValueError(f'mahotas.interpolate.{funcname}: spline order not supported')

    array = np.asarray(array)
    if np.iscomplexobj(array):
        raise TypeError(f'mahotas.interpolate.{funcname}: Complex type not supported')
    return array

def spline_filter1d(
    array: ArrayLike, order: int = 3, axis: int = -1,
    out: Array | type[Any] | None = None, dtype: DTypeLike = np.float64,
    output: Array | type[Any] | None = None,
) -> Array:
    """
    Calculates a one-dimensional spline filter along the given axis.

    The lines of the array along the given axis are filtered by a
    spline filter. The order of the spline must be >= 2 and <= 5.

    Parameters
    ----------
    array : array_like
        The input array.
    order : int, optional
        The order of the spline, default is 3.
    axis : int, optional
        The axis along which the spline filter is applied. Default is the last
        axis.
    out : ndarray, optional
        The array in which to place the output
    dtype : dtype, optional
        The dtype to use for computation (default: np.float64)

    Returns
    -------
    return_value : ndarray
        The filtered input.

    Notes
    -----
    For compatibility with scipy.ndimage, you can pass a dtype as the
    ``output`` argument. This will work as having passed it as a dtype.
    However, this is deprecated and should not be used in new code.
    """
    array = _check_interpolate(array, order, 'spline_filter1d')
    if order < 2:
        raise ValueError('mahotas.interpolate.spline_filter1d: spline order must be >= 2')
    if isinstance(out, type): # pragma: no cover
        import warnings
        warnings.warn('mahotas.interpolate.spline_filter1d: Use `dtype` for type instead of `out`', DeprecationWarning)
        dtype = out
        out = None
    if isinstance(output, type): # pragma: no cover
        import warnings
        warnings.warn('mahotas.interpolate.spline_filter1d: Use `dtype` for type instead of `output`', DeprecationWarning)
        dtype = output
        output = None
    output = internal._get_output(array, out, 'interpolate.spline_filter1d', dtype=dtype, output=output)
    output[...] = array
    axis = internal._get_axis(array, axis, 'interpolate.spline_filter1d')
    _interpolate.spline_filter1d(output, order, axis)
    return output


def spline_filter(
    array: ArrayLike, order: int = 3, out: Array | type[Any] | None = None,
    dtype: DTypeLike = np.float64, output: Array | type[Any] | None = None,
) -> Array:
    """
    Multi-dimensional spline filter.

    Parameters
    ----------
    array : array_like
        The input array.
    order : int, optional
        The order of the spline, default is 3.
    out : ndarray, optional
        The array in which to place the output
    dtype : dtype, optional
        The dtype to use for computation (default: np.float64)

    Returns
    -------
    return_value : ndarray
        The filtered input.

    See Also
    --------
    spline_filter1d

    Notes
    -----
    For compatibility with scipy.ndimage, you can pass a dtype as the
    ``out`` argument. This will work as having passed it as a dtype.  However,
    this is deprecated and should not be used in new code.

    The multi-dimensional filter is implemented as a sequence of
    one-dimensional spline filters. The intermediate arrays are stored
    in the same data type as the output. Therefore, for output types
    with a limited precision, the results may be imprecise because
    intermediate results may be stored with insufficient precision.

    """
    array = _check_interpolate(array, order, 'spline_filter')
    if order < 2:
        raise ValueError('mahotas.interpolate.spline_filter: spline order must be >= 2')
    if isinstance(out, type): # pragma: no cover
        import warnings
        warnings.warn('mahotas.interpolate.spline_filter: Use `dtype` for type instead of `out`', DeprecationWarning)
        dtype = out
        out = None
    if isinstance(output, type): # pragma: no cover
        import warnings
        warnings.warn('mahotas.interpolate.spline_filter: Use `dtype` for type instead of `output`', DeprecationWarning)
        dtype = output
        output = None
    output = internal._get_output(array, out, 'interpolate.spline_filter', dtype=dtype, output=output)
    output[...] = array
    for axis in range(array.ndim):
        _interpolate.spline_filter1d(output, order, axis)
    return output



def _maybe_filter(array: ArrayLike, order: int, func: str, prefilter: bool, dtype: DTypeLike, mode: BorderMode = 'mirror') -> tuple[Array, int]:
    array = _check_interpolate(array, order, func)
    if array.ndim < 1:
        raise ValueError(func+': array rank must be > 0')
    if 0 in array.shape:
        raise ValueError(func+': input axes must not be empty')
    npad = 0
    if prefilter and order > 1:
        if mode == 'nearest':
            # As in scipy.ndimage, prepad edge samples before filtering so
            # the spline can approach the constant extension smoothly.
            npad = 12
            array = np.pad(array, npad, mode='edge')
        if mode in ('nearest', 'reflect'):
            array = np.array(array, dtype=dtype, order='C', copy=True)
            for axis in range(array.ndim):
                _interpolate.spline_filter1d(array, order, axis, mode2int['reflect'])
            return array, npad
        return spline_filter(array, order, dtype=dtype), npad
    else:
        # _interpolate.zoom_shift requires a C-contiguous array
        return np.ascontiguousarray(array, dtype=dtype), npad

@overload
def zoom(
    array: ArrayLike, zoom: ArrayLike, out: NDArray[ScalarT], order: int = 3,
    mode: BorderMode = 'constant', cval: float = 0.0, prefilter: bool = True,
    output: Array | None = None,
) -> NDArray[ScalarT]: ...

@overload
def zoom(
    array: ArrayLike, zoom: ArrayLike, out: None = None, order: int = 3,
    mode: BorderMode = 'constant', cval: float = 0.0, prefilter: bool = True,
    output: None = None,
) -> NDArray[np.float64]: ...

@overload
def zoom(
    array: ArrayLike, zoom: ArrayLike, out: Array | None = None, order: int = 3,
    mode: BorderMode = 'constant', cval: float = 0.0, prefilter: bool = True,
    output: Array | None = None,
) -> Array: ...

def zoom(
    array: ArrayLike, zoom: ArrayLike, out: Array | None = None, order: int = 3,
    mode: BorderMode = 'constant', cval: float = 0.0, prefilter: bool = True,
    output: Array | None = None,
) -> Array:
    """
    Zoom an array.

    The array is zoomed using spline interpolation of the requested order.

    Parameters
    ----------
    array : ndarray
        The input array.
    zoom : float or sequence
        The zoom factor along the axes. If a float, `zoom` is the same for each
        axis. If a sequence, `zoom` should contain one value for each axis.
    out : ndarray, optional
        The array in which to place the output. It must have the same number
        of dimensions as `array`, but may be of any dtype (the result is
        computed in floating point and then converted). If given, the shape
        of `out` determines the actual zoom factors.
    order : int, optional
        The order of the spline interpolation, default is 3.
        The order has to be in the range 0-5.
    mode : str, optional
        Points outside the boundaries of the input are filled according
        to the given mode: one of ``'constant'``, ``'nearest'``,
        ``'reflect'``, ``'mirror'``, ``'wrap'``, ``'ignore'``.
        Default is ``'constant'``.
    cval : scalar, optional
        Value used for points outside the boundaries of the input if
        ``mode='constant'``. Currently, only 0 is supported (other values
        raise ``NotImplementedError``).
    prefilter : bool, optional
        The parameter prefilter determines if the input is pre-filtered with
        `spline_filter` before interpolation (necessary for spline
        interpolation of order > 1).  If False, it is assumed that the input is
        already filtered. Default is True.

    Returns
    -------
    return_value : ndarray
        The zoomed input. If `out` is not given, this is a new ``np.float64``
        array of shape ``int(array.shape[i] * zoom[i])``.
    """
    _check_mode(mode, cval, 'interpolation.zoom')
    array, npad = _maybe_filter(array, order, 'interpolate.zoom', prefilter, dtype=np.float64, mode=mode)
    input_shape = tuple(s - 2*npad for s in array.shape)
    zoom = np.array(zoom)
    if zoom.ndim == 0:
        zoom = np.array([zoom]*array.ndim)
    elif zoom.ndim != 1:
        raise ValueError('mahotas.interpolation.zoom: zoom should be a 1-d array')
    if len(zoom) != array.ndim:
        raise ValueError('mahotas.interpolation.zoom: zoom should have one element for each dimension of array')

    if out is None and output is not None: # pragma: no cover
        import warnings
        warnings.warn('mahotas.interpolate.zoom: Use `out` for output parameter instead of `output`', DeprecationWarning)
        out = output

    if out is None:
        output_shape = tuple([int(s * z) for s,z in zip(input_shape, zoom)])
        out = np.empty(output_shape, dtype=array.dtype)
    elif out.ndim != array.ndim:
        raise ValueError('mahotas.interpolation.zoom: out should have the same number of dimensions as array')
    zoom_div = np.array(out.shape, float) - 1
    zoom = np.divide(np.array(input_shape) - 1, zoom_div,
                     out=np.ones_like(zoom_div), where=zoom_div != 0)
    zoom = np.ascontiguousarray(zoom)

    # zooming is always done in floats, but the user may want the output in
    # another type, so handle this case here:
    o_out = None
    if out.dtype != array.dtype:
        o_out = out
        out = np.empty(out.shape, array.dtype)
    _interpolate.zoom_shift(array, zoom, None, out, order, mode2int[mode], cval, npad)
    if o_out is not None:
        o_out[:] = out[:]
        out = o_out
    return out


def shift(
    array: ArrayLike, shift: ArrayLike, out: NDArray[np.float64] | None = None,
    order: int = 3, mode: BorderMode = 'constant', cval: float = 0.0,
    prefilter: bool = True, output: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """
    Shift an array.

    The array is shifted using spline interpolation of the requested order.
    Points outside the boundaries of the input are filled according to the
    given mode.

    Parameters
    ----------
    array : ndarray
        The input array.
    shift : sequence of float
        The shift along the axes. Must contain one value for each axis (a
        scalar is only accepted for 1-D arrays). Values must be finite and
        are not modified.
    out : ndarray, optional
        The array in which to place the output. Must be a C-contiguous
        ``np.float64`` array of the same shape as `array`.
    order : int, optional
        The order of the spline interpolation, default is 3.
        The order has to be in the range 0-5.
    mode : str, optional
        Points outside the boundaries of the input are filled according
        to the given mode: one of ``'constant'``, ``'nearest'``,
        ``'reflect'``, ``'mirror'``, ``'wrap'``, ``'ignore'``.
        Default is ``'constant'``.
    cval : scalar, optional
        Value used for points outside the boundaries of the input if
        ``mode='constant'``. Currently, only 0 is supported (other values
        raise ``NotImplementedError``).
    prefilter : bool, optional
        The parameter prefilter determines if the input is pre-filtered with
        `spline_filter` before interpolation (necessary for spline
        interpolation of order > 1).  If False, it is assumed that the input is
        already filtered. Default is True.

    Returns
    -------
    return_value : ndarray of np.float64
        The shifted input.

    Raises
    ------
    ValueError
        If any shift value is not finite.

    """
    _check_mode(mode, cval, 'interpolation.shift')
    array, npad = _maybe_filter(array, order, 'interpolate.shift', prefilter, dtype=np.float64, mode=mode)
    unpadded = array[(slice(npad, -npad),)*array.ndim] if npad else array
    output = internal._get_output(unpadded, out, 'interpolate.shift', dtype=np.float64, output=output)
    shift = np.ascontiguousarray(shift, dtype=np.float64)
    if not np.all(np.isfinite(shift)):
        raise ValueError('mahotas.interpolate.shift: shift values must be finite')
    shift = -shift
    _interpolate.zoom_shift(array, None, shift, output, order, mode2int[mode], cval, npad)
    return output
