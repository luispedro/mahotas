# -*- coding: utf-8 -*-
# Copyright (C) 2008-2012 Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# Carnegie Mellon University
#
# License: MIT (see COPYING file)

from __future__ import division
import numpy as np

from .internal import _check_3
from . import _distance
from . import _segmentation

__all__ = [
    'gvoronoi',
    'slic'
    ]

def gvoronoi(labeled):
    '''
    segmented = gvoronoi(labeled)

    Generalised Voronoi Transform.

    The generalised Voronoi diagram assigns to the pixel (i,j) the label of the
    nearest object (i.e., the value of the nearest non-zero pixel in labeled).

    Parameters
    ----------
    labeled : ndarray
        a labeled array, of a form similar to one returned by
        ``mahotas.label()``

    Returns
    -------
    segmented : is of the same size and type as labeled and
                `segmented[y,x]` is the label of the object at position `y,x`.
    '''
    labeled = np.ascontiguousarray(labeled)
    bw = (labeled == 0)
    f = np.zeros(bw.shape, np.double)
    f[bw] = len(f.shape)*max(f.shape)**2+1
    orig = np.arange(f.size, dtype=np.intc).reshape(f.shape)
    _distance.dt(f, orig)
    return labeled.flat[orig]


def slic(array, spacer=16):
    '''
    segmented, n_segments = slic(array, spacer=16)

    SLIC Superpixels

    Note: This function operates on the array values. In the original
    publication, SLIC was applied in L*a*b* space. See the mahotas.color module
    for color space transformations.

    Parameters
    ----------
    array : ndarray
    spacer : int, optional

    Returns
    -------
    segmented : ndarray
    n_segments : int
        Number of segments

    References
    ----------
    .. [1] Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi,
        Pascal Fua, and Sabine Süsstrunk, SLIC Superpixels Compared to
        State-of-the-art Superpixel Methods, TPAMI, May 2012.
    '''
    array = np.ascontiguousarray(array, dtype=np.float32)
    _check_3(array, 'slic')
    labels = np.zeros((array.shape[0], array.shape[1]), dtype=np.intc)
    labels = labels.copy()
    n = _segmentation.slic(array, labels, int(spacer))
    return labels, n
