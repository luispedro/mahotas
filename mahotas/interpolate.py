# This module was adapted from scipy.ndimage and retains its license
# Copyright (C) 2003-2005 Peter J. Verveer
# Copyright (C) 2011-2014 Luis Pedro Coelho
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

import numpy as np
from . import internal
from . import _interpolate
from ._filters import mode2int, modes, _check_mode

def _check_interpolate(array, order, funcname):
    if not (0 < order < 5):
        raise ValueError('mahotas.interpolate.%s: spline order not supported' % funcname)

    array = np.asarray(array)
    if np.iscomplexobj(array):
        raise TypeError('mahotas.interpolate.%s: Complex type not supported' % funcname)
    return array

def spline_filter1d(array, order=3, axis=-1, out=None, dtype=np.float64, output=None):
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

    For compatibility with scipy.ndimage, you can pass a dtype as the
    ``output`` argument. This will work as having passed it as a dtype.
    However, this is deprecated and should not be used in new code.

    Returns
    -------
    return_value : ndarray or None
        The filtered input.
    """
    array = _check_interpolate(array, order, 'spline_filter1d')
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


def spline_filter(array, order=3, out=None, dtype=np.float64, output=None):
    """
    Multi-dimensional spline filter.

    Parameters
    ----------
    array : array_like
        The input array.
    order : int, optional
        The order of the spline, default is 3.
        axis.
    out : ndarray, optional
        The array in which to place the output
    dtype : dtype, optional
        The dtype to use for computation (default: np.float64)

    For compatibility with scipy.ndimage, you can pass a dtype as the
    ``out`` argument. This will work as having passed it as a dtype.  However,
    this is deprecated and should not be used in new code.

    Returns
    -------
    return_value : ndarray or None
        The filtered input.

    See Also
    --------
    spline_filter1d

    Notes
    -----
    The multi-dimensional filter is implemented as a sequence of
    one-dimensional spline filters. The intermediate arrays are stored
    in the same data type as the output. Therefore, for output types
    with a limited precision, the results may be imprecise because
    intermediate results may be stored with insufficient precision.

    """
    array = _check_interpolate(array, order, 'spline_filter')
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



def _maybe_filter(array, order, func, prefilter, dtype):
    array = _check_interpolate(array, order, func)
    if array.ndim < 1:
        raise ValueError(func+': array rank must be > 0')
    if prefilter and order > 1:
        return spline_filter(array, order, dtype=dtype)
    else:
        return array.astype(dtype)

def zoom(array, zoom, out=None, order=3, mode='constant', cval=0.0, prefilter=True, output=None):
    """
    Zoom an array.

    The array is zoomed using spline interpolation of the requested order.

    Parameters
    ----------
    array : ndarray
        The input array.
    zoom : float or sequence, optional
        The zoom factor along the axes. If a float, `zoom` is the same for each
        axis. If a sequence, `zoom` should contain one value for each axis.
    out : ndarray or dtype, optional
        The array in which to place the output, or the dtype of the returned
        array.
    order : int, optional
        The order of the spline interpolation, default is 3.
        The order has to be in the range 0-5.
    mode : str, optional
        Points outside the boundaries of the input are filled according
        to the given mode ('constant', 'nearest', 'reflect' or 'wrap').
        Default is 'constant'.
    cval : scalar, optional
        Value used for points outside the boundaries of the input if
        ``mode='constant'``. Default is 0.0
    prefilter : bool, optional
        The parameter prefilter determines if the input is pre-filtered with
        `spline_filter` before interpolation (necessary for spline
        interpolation of order > 1).  If False, it is assumed that the input is
        already filtered. Default is True.

    Returns
    -------
    return_value : ndarray
    """
    array = _maybe_filter(array, order, 'interpolate.zoom', prefilter, dtype=np.float64)
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
        output_shape = tuple([int(s * z) for s,z in zip(array.shape, zoom)])
        out = np.empty(output_shape, dtype=array.dtype)
    zoom_div = np.array(out.shape, float) - 1
    zoom = (np.array(array.shape) - 1) / zoom_div
    zoom = np.ascontiguousarray(zoom)

    # Zooming to infinity is unpredictable, so just choose
    # zoom factor 1 instead
    zoom[np.isinf(zoom)] = 1

    _check_mode(mode, cval, 'interpolation.zoom')
    _interpolate.zoom_shift(array, zoom, None, out, order, mode2int[mode], cval)
    return out


def shift(array, shift, out=None, order=3, mode='constant', cval=0.0,
          prefilter=True, output=None):
    """
    Shift an array.

    The array is shifted using spline interpolation of the requested order.
    Points outside the boundaries of the input are filled according to the
    given mode.

    Parameters
    ----------
    array : ndarray
        The input array.
    shift : float or sequence, optional
        The shift along the axes. If a float, `shift` is the same for each
        axis. If a sequence, `shift` should contain one value for each axis.
    out : ndarray or dtype, optional
        The array in which to place the output, or the dtype of the returned
        array.
    order : int, optional
        The order of the spline interpolation, default is 3.
        The order has to be in the range 0-5.
    mode : str, optional
        Points outside the boundaries of the input are filled according
        to the given mode ('constant', 'nearest', 'reflect' or 'wrap').
        Default is 'constant'.
    cval : scalar, optional
        Value used for points outside the boundaries of the input if
        ``mode='constant'``. Default is 0.0
    prefilter : bool, optional
        The parameter prefilter determines if the input is pre-filtered with
        `spline_filter` before interpolation (necessary for spline
        interpolation of order > 1).  If False, it is assumed that the input is
        already filtered. Default is True.

    Returns
    -------
    return_value : ndarray
        The shifted input.

    """
    array = _maybe_filter(array, order, 'interpolate.shift', prefilter, dtype=np.float64)
    _check_mode(mode, cval, 'interpolation.shift')
    output = internal._get_output(array, out, 'interpolate.shift', dtype=np.float64, output=output)
    shift = np.ascontiguousarray(shift, dtype=np.float64)
    shift *= -1
    _interpolate.zoom_shift(array, None, shift, output, order, mode2int[mode], cval)
    return output


