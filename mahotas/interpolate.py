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
