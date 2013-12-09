# -*- coding: utf-8 -*-
# Copyright (C) 2008-2013 Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# Carnegie Mellon University
#
# License: MIT (see COPYING file)

from __future__ import division
import numpy as np

from .internal import _check_3
from . import _distance
from . import _labeled

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


def slic(array, spacer=16, m=1.0, max_iters=128):
    '''Compute SLIC superpixel oversegmentation

    Note: This function operates on the array values. In the original
    publication, SLIC was applied in L*a*b* space.

    To use the original mode, use::

        rgb = mh.demos.load('lena')
        lab = mh.colors.rgb2lab(rgb)
        superseg,nr = mh.segmentation.slic(lab)

    See the mahotas.color module for color space transformations

    Parameters
    ----------
    array : ndarray
    spacer : int, optional
        x/y spacing between initial seeds. Initial seeds will be placed at
        ``array[spacer/2::spacer,spacer/::spacer]``
    m : float, optional
        tradeoff between colour space and spatial distance.
    max_iters : int, optional
        Maximum number of k-means iterations. Generally this does not need to
        be very large because the search is only local and convergence is
        very fast, in which case, the algorithm will exit early. (default: 128)

    Returns
    -------
    segmented : ndarray
        A segmented area numbered 1..N. The general mahotas convention is that
        region 0 is background. Thus, 1..N is used here.
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
    n = _labeled.slic(array, labels, int(spacer), float(m), int(max_iters))
    return labels, n
