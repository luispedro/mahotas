# Copyright (C) 2008-2010, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# 
# LICENSE: GPLv3

from __future__ import division
import numpy as np
from .morph import get_structuring_elem
from . import _labeled
from .internal import _get_output

__all__ = [
    'borders',
    'border',
    'label',
    'labeled_sum',
    ]

def label(array, Bc=None, output=None):
    '''
    labeled, nr_objects = label(array, Bc={3x3 cross}, output=None)

    Label the array

    Parameters
    ----------
    array : ndarray
        This will be interpreted as an integer array
    Bc : ndarray, optional
        This is the structuring element to use
    output : ndarray, optional
        Output array. Must be a C-array, of type np.int32

    Returns
    -------
    labeled : ndarray
        Labeled result
    nr_objects : int
        Number of objects
    '''
    output = _get_output(array, output, 'labeled.label', np.int32)
    output[:] = (array != 0)
    Bc = get_structuring_elem(output, Bc)
    nr_objects = _labeled.label(output, Bc)
    return output, nr_objects

def border(labeled, i, j, Bc=None, output=None, always_return=True):
    '''
    border_img = border(labeled, i, j, Bc={3x3 cross}, output={np.zeros(labeled.shape, bool)}, always_return=True)

    Compute the border region between `i` and `j` regions.

    A pixel is on the border if it has value `i` (or `j`) and a pixel in its
    neighbourhood (defined by `Bc`) has value `j` (or `i`).

    Parameters
    ----------
    labeled : ndarray of integer type
        input labeled array
    i : integer
    j : integer
    Bc : structure element, optional
    output : ndarray of same shape as `labeled`, dtype=bool, optional
        where to store the output. If ``None``, a new array is allocated
    always_return : bool, optional
        if false, then, in the case where there is no pixel on the border,
        returns ``None``. Otherwise (the default), it always returns an array
        even if it is empty.

    Returns
    -------
    border_img : boolean ndarray
        Pixels are True exactly where there is a border between `i` and `j` in `labeled`
    '''
    Bc = get_structuring_elem(labeled, Bc)
    output = _get_output(labeled, output, 'labeled.border', bool)
    output.fill(False)
    return _labeled.border(labeled, Bc, output, i, j, bool(always_return))

def borders(labeled, Bc=None, output=None):
    '''
    border_img = borders(labeled, Bc={3x3 cross}, output={np.zeros(labeled.shape, bool)})

    Compute border pixels

    A pixel is on a border if it has value `i` and a pixel in its neighbourhood
    (defined by `Bc`) has value `j`, with ``i != j``.

    Parameters
    ----------
    labeled : ndarray of integer type
        input labeled array
    Bc : structure element, optional
    output : ndarray of same shape as `labeled`, dtype=bool, optional
        where to store the output. If ``None``, a new array is allocated

    Returns
    -------
    border_img : boolean ndarray
        Pixels are True exactly where there is a border in `labeled`
    '''
    Bc = get_structuring_elem(labeled, Bc)
    output = _get_output(labeled, output, 'labeled.borders', bool)
    output.fill(False)
    return _labeled.borders(labeled, Bc, output)

def labeled_sum(array, labeled):
    '''
    sums = labeled_sum(array, labeled)

    Labeled sum. sum will be an array of size ``labeled.max() + 1``, where
    ``sum[i]`` is equal to ``np.sum(array[labeled == i])``.

    Parameters
    ----------
    array : ndarray of any type
    labeled : int ndarray
        Label map. This is the same type as returned from ``mahotas.label()``

    Returns
    -------
    sums : 1-d ndarray of ``array.dtype``
    '''
    if labeled.dtype != np.intc or not labeled.flags.carray:
        raise ValueError('mahotas.labeled.labeled_sum: labeled is not as expected')
    if array.shape != labeled.shape:
        raise ValueError('mahotas.labeled.labeled_sum: `array` is not the same size as `labeled`')
    maxv = labeled.max() + 1
    output = np.empty(maxv, dtype=array.dtype)
    _labeled.labeled_sum(array, labeled, output)
    return output

def labeled_size(labeled):
    '''
    sizes = labeled_size(labeled)

    Equivalent to::

        for i in xrange(...):
            sizes[i] = np.sum(labeled == i)

    but, naturally, much faster.

    Parameters
    ----------
    labeled : int ndarray

    Returns
    -------
    sizes : 1-d ndarray of int

    See Also
    --------
    mahotas.fullhistogram : almost same function by another name (the only
    difference is that that function only accepts unsigned integer types).
    '''
    from .histogram import fullhistogram
    return fullhistogram(labeled.astype(np.uint32))


