# Copyright (C) 2010-2011, Luis Pedro Coelho <luis@luispedro.org>
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
from . import morph
from ._filters import mode2int, modes

__all__ = [
    'convolve',
    'median_filter',
    'rank_filter'
    ]

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

def median_filter(f, Bc=None, mode='reflect', cval=0.0, output=None):
    '''
    median = median_filter(f, Bc={square}, mode='reflect', cval=0.0, output=None)

    Median filter

    Parameters
    ----------
    f : ndarray
        input. Any dimension is supported
    Bc : ndarray or int, optional
        Defines the neighbourhood, default is a square of side 3.
    mode : {'reflect' [default], 'nearest', 'wrap', 'mirror', 'constant'}
        How to handle borders
    cval : double, optional
        If `mode` is constant, which constant to use (default: 0.0)
    output : ndarray, optional
        Output array. Must have same shape and dtype as `f` as well as be
        C-contiguous.

    Returns
    -------
    median : ndarray of same type and shape as ``f``
        median[i,j] is the median value of the points in f close to (i,j)
    '''
    if Bc is None:
        Bc = np.ones((3,) * len(f.shape), f.dtype)
    elif f.dtype != Bc.dtype:
        Bc = Bc.astype(f.dtype)
    rank = Bc.sum()//2
    if output is not None:
        if output.dtype != f.dtype: raise ValueError('mahotas.median_filter: `output` has wrong type')
        if output.shape != f.shape: raise ValueError('mahotas.median_filter: `output` has wrong shape')
        if not output.flags.contiguous: raise ValueError('mahotas.median_filter: `output` is not c-array')
    else:
        output = np.empty(f.shape, f.dtype)
    if mode not in modes:
        raise ValueError('mahotas.median_filter: `mode` not in %s' % modes)
    if mode == 'constant' and cval != 0.:
        raise NotImplementedError('Please email mahotas developers to get this implemented.')
    return _convolve.rank_filter(f, Bc, output, rank, mode2int[mode])

def rank_filter(f, Bc, rank, mode='reflect', cval=0.0, output=None):
    '''
    ranked = rank_filter(f, Bc, rank, mode='reflect', cval=0.0, output=None)

    Rank filter. The value at ``ranked[i,j[`` will be the ``rank``th largest in
    the neighbourhood defined by ``Bc``.

    Parameters
    ----------
    f : ndarray
        input. Any dimension is supported
    Bc : ndarray
        Defines the neighbourhood. Must be explicitly passed, no default.
    rank : integer
    mode : {'reflect' [default], 'nearest', 'wrap', 'mirror', 'constant'}
        How to handle borders
    cval : double, optional
        If `mode` is constant, which constant to use (default: 0.0)
    output : ndarray, optional
        Output array. Must have same shape and dtype as `f` as well as be
        C-contiguous.

    Returns
    -------
    ranked : ndarray of same type and shape as ``f``
        ranked[i,j] is the ``rank``th value of the points in f close to (i,j)

    See Also
    --------
    median_filter : A special case of rank_filter
    '''
    Bc = morph.get_structuring_elem(f, Bc)
    if output is not None:
        if output.dtype != f.dtype: raise ValueError('mahotas.rank_filter: `output` has wrong type')
        if output.shape != f.shape: raise ValueError('mahotas.rank_filter: `output` has wrong shape')
        if not output.flags.contiguous: raise ValueError('mahotas.rank_filter: `output` is not c-array')
    else:
        output = np.empty(f.shape, f.dtype)
    if mode not in modes:
        raise ValueError('mahotas.rank_filter: `mode` not in %s' % modes)
    if mode == 'constant' and cval != 0.:
        raise NotImplementedError('Please email mahotas developers to get this implemented.')
    return _convolve.rank_filter(f, Bc, output, rank, mode2int[mode])

