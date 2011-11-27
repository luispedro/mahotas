# This module was adapted from scipy.ndimage and retains its license
# Copyright (C) 2003-2005 Peter J. Verveer
# Copyright (C) 2011 Luis Pedro Coelho
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

def spline_filter1d(array, order=3, axis=-1, output=None, dtype=np.float64):
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
    output : ndarray, optional
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
    if order < 0 or order > 5:
        raise RuntimeError('mahotas.interpolate.spline_filter1d: spline order not supported')
    array = np.asarray(array)
    if np.iscomplexobj(array):
        raise TypeError('mahotas.interpolate.spline_filter1d: Complex type not supported')
    if isinstance(output, type):
        dtype = output
        output = None
    output = internal._get_output(array, output, 'interpolate.spline_filter1d', dtype=dtype)
    output[...] = array
    axis = internal._get_axis(array, axis, 'interpolate.spline_filter1d')
    _interpolate.spline_filter1d(output, order, axis)
    return output


def spline_filter(array, order=3, output=None, dtype=np.float64):
    """
    Multi-dimensional spline filter.

    Parameters
    ----------
    array : array_like
        The input array.
    order : int, optional
        The order of the spline, default is 3.
        axis.
    output : ndarray, optional
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
    array = np.asanyarray(array)
    if not (2 < order < 5):
        raise RuntimeError('mahotas.interpolation.spline_filter: spline order not supported')
    if np.iscomplexobj(array):
        raise TypeError('mahotas.interpolation.spline_filter: Complex type not supported')
    if isinstance(output, type):
        dtype = output
        output = None
    output = internal._get_output(array, output, 'interpolate.spline_filter', dtype=dtype)
    output[...] = array
    for axis in range(array.ndim):
        _interpolate.spline_filter1d(output, order, axis)
    return output



def _maybe_filter(array, order, func, prefilter, dtype):
    if order < 0 or order > 5:
        raise RuntimeError(func+': spline order not supported')
    array = np.asanyarray(array)
    if np.iscomplexobj(array):
        raise TypeError(func+': Complex type not supported')
    if array.ndim < 1:
        raise RuntimeError(func+': array rank must be > 0')
    if prefilter and order > 1:
        return spline_filter(array, order, dtype=dtype)
    else:
        return array

def zoom(array, zoom, output=None, order=3, mode='constant', cval=0.0, prefilter=True):
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
    output : ndarray or dtype, optional
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
    return_value : ndarray or None
        The zoomed input. If `output` is given as a parameter, None is
        returned.

    """
    array = _maybe_filter(array, order, 'interpolate.zoom', prefilter, dtype=np.float64)
    zoom = np.array(zoom)
    if zoom.ndim == 0:
        zoom = np.array([zoom]*array.ndim)
    elif zoom.ndim != 1:
        raise ValueError('mahotas.interpolation.zoom: zoom should be a 1-d array')
    if len(zoom) != array.ndim:
        raise ValueError('mahotas.interpolation.zoom: zoom should have one element for each dimension of array')

    if output is None:
        output_shape = tuple([int(s * z) for s,z in zip(array.shape, zoom)])
        output = np.empty(output_shape, dtype=array.dtype)
    zoom_div = np.array(output.shape, float) - 1
    zoom = (np.array(array.shape) - 1) / zoom_div
    zoom = np.ascontiguousarray(zoom)

    # Zooming to infinity is unpredictable, so just choose
    # zoom factor 1 instead
    zoom[np.isinf(zoom)] = 1

    _check_mode(mode, cval, 'interpolation.zoom')
    _interpolate.zoom_shift(array, zoom, None, output, order, mode2int[mode], cval)
    return output


def shift(array, shift, output=None, order=3, mode='constant', cval=0.0,
          prefilter=True):
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
    output : ndarray or dtype, optional
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
    output = internal._get_output(array, output, 'interpolate.shift', dtype=np.float64)
    shift = np.ascontiguousarray(shift, dtype=np.float64)
    shift *= -1
    _interpolate.zoom_shift(array, None, shift, output, order, mode2int[mode], cval)
    return output


