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

import _interpolate
import numpy as np

def spline_filter1d(array, order=3, axis=-1, output=np.float64):
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
    output : ndarray or dtype, optional
        The array in which to place the output, or the dtype of the returned
        array. Default is `np.float64`.

    Returns
    -------
    return_value : ndarray or None
        The filtered input.
    """
    import internal
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

