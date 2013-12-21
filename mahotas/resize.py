# Copyright (C) 2010-2013, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT (see COPYING file)

from __future__ import division
import numpy as np

__all__ = [
    'imresize',
    ]


def imresize(img, nsize, order=3):
    '''Resizes image

    This function works in two ways: if ``nsize`` is a tuple or list of
    integers, then the result will be of this size; otherwise, this function
    behaves the same as ``mh.interpolate.zoom``

    Parameters
    ----------
    img : ndarray
    nsize : float or tuple(float) or tuple(integers)
        Size of return. Meaning depends on the type
            float: img'.shape[i] = nsize * img.shape[i]
            tuple of float: img'.shape[i] = nsize[i] * img.shape[i]
            tuple of int: img'.shape[i] = nsize[i]
    order : integer, optional
        Spline order to use (default: 3)

    Returns
    -------
    img' : ndarray

    See Also
    --------
    zoom : Similar function
    scipy.misc.pilutil.imresize : Similar function
    '''
    from .interpolate import zoom
    if type(nsize) == tuple or type(nsize) == list:
        if type(nsize[0]) == int:
            nsize = np.array(nsize, dtype=float)
            nsize /= img.shape
    return zoom(img, nsize, order=order)
