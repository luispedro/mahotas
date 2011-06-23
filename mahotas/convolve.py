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
from .internal import _get_output
from ._filters import mode2int, modes, _check_mode

__all__ = [
    'convolve',
    'median_filter',
    'rank_filter',
    'template_match',
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
    output = _get_output(f, output, 'convolve')
    _check_mode(mode, cval, 'convolve')
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
    output = _get_output(f, output, 'median_filter')
    _check_mode(mode, cval, 'median_filter')
    return _convolve.rank_filter(f, Bc, output, rank, mode2int[mode])

def rank_filter(f, Bc, rank, mode='reflect', cval=0.0, output=None):
    '''
    ranked = rank_filter(f, Bc, rank, mode='reflect', cval=0.0, output=None)

    Rank filter. The value at ``ranked[i,j]`` will be the ``rank``th largest in
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
    output = _get_output(f, output, 'rank_filter')
    _check_mode(mode, cval, 'rank_filter')
    return _convolve.rank_filter(f, Bc, output, rank, mode2int[mode])


def template_match(f, template, mode='reflect', cval=0., output=None):
    '''
    match = template_match(f, template, mode='reflect', cval=0., output={np.empty_like(f)})

    Match template.

    The value at ``match[i,j]`` will be the difference (in squared euclidean
    terms), between `template` and a same sized window on `f` centered on that
    point.

    Parameters
    ----------
    f : ndarray
        input. Any dimension is supported
    template : ndarray
        Template to match. Must be explicitly passed, no default.
    mode : {'reflect' [default], 'nearest', 'wrap', 'mirror', 'constant'}
        How to handle borders
    cval : double, optional
        If `mode` is constant, which constant to use (default: 0.0)
    output : ndarray, optional
        Output array. Must have same shape and dtype as `f` as well as be
        C-contiguous.

    Returns
    -------
    match : ndarray of same type and shape as ``f``
        match[i,j] is the squared euclidean distance between
        ``f[i-s0:i+s0,j-s1:j+s1]`` and ``template`` (for appropriately defined
        ``s0`` and ``s1``).
    '''
    template = template.astype(f.dtype)
    output = _get_output(f, output, 'template_match')
    _check_mode(mode, cval, 'template_match')
    return _convolve.template_match(f, template, output, mode2int[mode])

