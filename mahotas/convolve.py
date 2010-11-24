# Copyright (C) 2010, Luis Pedro Coelho <lpc@cmu.edu>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation; either version 2 of the License,
# or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
# 02110-1301, USA.

from __future__ import division
import numpy as np
from . import _convolve
from ._filters import mode2int, modes

__all__ = ['convolve']

def convolve(f, weights, mode='reflect', cval=0.0, output=None):
    '''
    convolved = convolve(f, weights, mode='reflect', cval=0.0, output={new array})

    Convolution of `f` and `weights`

    Convolution is performed in `doubles` to avoid over/underflow, but the
    result is then cast to `f.dtype`.

    Parameters
    ----------
    f : ndarray
        input. Any dimension is supported
    weights : ndarray
        weight filter. If not of the same dtype as `f`, it is cast
    mode : {'reflect' [default], 'nearest', 'wrap', 'mirror', 'constant'}
        How to handle borders
    cval : double, optional
        If `mode` is constant, which constant to use (default: 0.0)
    output : ndarray, optional
        Output array. Must have same shape and dtype as `f` as well as be
        C-contiguous.

    Returns
    -------
    convolved : ndarray of same dtype as `f`
    '''
    if f.dtype != weights.dtype:
        weights = weights.astype(f.dtype)
    if output is not None:
        if output.dtype != f.dtype: raise ValueError('mahotas.convolve: `output` has wrong type')
        if output.shape != f.shape: raise ValueError('mahotas.convolve: `output` has wrong shape')
        if not output.flags['CONTIGUOUS']: raise ValueError('mahotas.convolve: `output` is not c-array')
    if mode not in modes:
        raise ValueError('mahotas.convolve: `mode` not in %s' % modes)
    if mode == 'constant' and cval != 0.:
        raise NotImplementedError('Please email mahotas developers to get this implemented.')
    return _convolve.convolve(f, weights, output, mode2int[mode])

