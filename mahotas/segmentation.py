# -*- coding: utf-8 -*-
# Copyright (C) 2008-2011 Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# Carnegie Mellon University
# 
# License: MIT (see COPYING file)

from __future__ import division
import numpy as np
from . import _distance

__all__ = [
    'gvoronoi',
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
