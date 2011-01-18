# Copyright (C) 2008-2010, Luis Pedro Coelho <lpc@cmu.edu>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# 
# LICENSE: GPLv3

from __future__ import division
import numpy as np
from .morph import get_structuring_elem
from . import _labeled

__all__ = ['border']

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
    if output is None:
        output = np.zeros(labeled.shape, bool)
    else:
        if output.dtype != bool:
            raise TypeError('mahotas.labeled.border: output must be boolean')
        if not output.flags['C_CONTIGUOUS']:
            raise TypeError('mahotas.labeled.border: output must be C-contiguous')
        if output.shape != labeled.shape:
            raise ValueError('mahotas.labeled.border: output must be of same size as `labeled`')
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
    if output is None:
        output = np.zeros(labeled.shape, bool)
    else:
        if output.dtype != bool:
            raise TypeError('mahotas.labeled.borders: output must be boolean')
        if not output.flags['C_CONTIGUOUS']:
            raise TypeError('mahotas.labeled.borders: output must be C-contiguous')
        if output.shape != labeled.shape:
            raise ValueError('mahotas.labeled.borders: output must be of same size as `labeled`')
    return _labeled.borders(labeled, Bc, output)

