# Copyright (C) 2010-2013, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT (see COPYING file)

from __future__ import division
import numpy as np

__all__ = [
    'imresize',
    'resize_to',
    'resize_rgb_to',
    ]

def resize_to(im, nsize, order=3):
    '''Resize image to a specified new size

    Parameters
    ----------
    im : ndarray
    nsize : sequence of numbers
        Will be the new size of the array
    order : integer, optional
        Spline order to use (default: 3)

    Returns
    -------
    im' : ndarray

    See Also
    --------
    zoom : Similar function
    imresize : A more flexible, but also confusing, version of this function
    resize_rgb_to : A version appropriate for resize RGB images
    '''
    from .interpolate import zoom
    if len(nsize) != im.ndim:
        raise ValueError('mahotas.resize_to: new size does not have the same dimension as old one')
    out = np.empty(nsize, dtype=im.dtype)
    nsize = np.array(nsize, dtype=float)
    nsize /= im.shape
    return zoom(im, nsize, order=order, out=out)


def resize_rgb_to(im, nsize, order=3):
    '''Resize an RGB image to size ``nsize``

    Parameters
    ----------
    im : ndarray
    nsize : sequence of 2 numbers
        if nsize is ``(h,w)``, the new image will be ``(h,w,3)``
    order : integer, optional
        Spline order to use (default: 3)

    Returns
    -------
    im' : ndarray

    See Also
    --------
    zoom : Similar function
    imresize : A more flexible, but also confusing, version of this function
    resize_to : A generic version of this function
    '''
    from .internal import _check_3
    _check_3(im, 'resize_rgb_to')
    return np.dstack([resize_to(ch, nsize, order) for ch in im.transpose((2,0,1))])

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
