# -*- coding: utf-8 -*-
# Copyright (C) 2009-2019, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

from __future__ import division
import numpy as np
from .internal import _check_2

__all__ = [
        'as_rgb',
        'overlay',
        'stretch',
        'stretch_rgb',
        ]

def stretch_rgb(img, arg0=None, arg1=None, dtype=np.uint8):
    '''Variation of stretch() function that works per-channel on an RGB image

    Parameters
    ----------
    img : ndarray
        input image. It is *not modified* by this function
    min : integer, optional
        minimum value for output [default: 0]
    max : integer, optional
        maximum value for output [default: 255]
    dtype : dtype of output,optional
         [default: np.uint8]

    Returns
    -------
    img': ndarray
        resulting image. ndarray of same shape as `img` and type `dtype`.

    See Also
    --------
    stretch : function
    '''
    if img.ndim == 2:
        return stretch(img, arg0, arg1, dtype)
    elif img.ndim == 3:
        return np.dstack([stretch(img[:,:,i], arg0, arg1, dtype) for i in range(img.shape[2])])
    else:
        raise ValueError('mahotas.stretch_rgb: Only works for RGB images')


def stretch(img, arg0=None, arg1=None, dtype=np.uint8):
    '''
    img' = stretch(img, [dtype=np.uint8])
    img' = stretch(img, max, [dtype=np.uint8])
    img' = stretch(img, min, max, [dtype=np.uint8])

    Contrast stretch the image to the range [0, max] (first form) or [min, max]
    (second form). The method is simple linear stretching according to the
    formula::

        p' = max * (p - img.min())/img.ptp() + min

    Parameters
    ----------
    img : ndarray
        input image. It is *not modified* by this function
    min : integer, optional
        minimum value for output [default: 0]
    max : integer, optional
        maximum value for output [default: 255]
    dtype : dtype of output,optional
         [default: np.uint8]

    Returns
    -------
    img': ndarray
        resulting image. ndarray of same shape as `img` and type `dtype`.

    Notes
    -----
    If max > 255, then it truncates the values if dtype is not specified.
    '''
    if arg0 is None:
        min = 0
        max = 255
    elif arg1 is None:
        min = 0
        max = arg0
    else:
        min = arg0
        max = arg1
    img = img.astype(np.double, copy=True)
    img -= img.min()
    ptp = img.ptp()
    if not ptp:
        img = np.zeros(img.shape, dtype)
        if min:
            img += min
        return img
    img *= float(max - min)/ptp
    if min: img += min
    return img.astype(dtype, copy=False)

def as_rgb(r, g, b):
    '''
    rgb = as_rgb(r, g, b)

    Returns an RGB image with ``r`` in the red channel, ``g`` in the green, and
    ``b`` in the blue. The channels are contrast stretched.

    If any of the channels is `None`, that channel is set to zero. The same can
    be achieved by passing ``0`` as that channels value. In fact, passing a
    number as a channel value will set the whole channel to that value.

    Examples
    --------

    This shows a nice looking picture::

        z1 = np.linspace(0, np.pi)
        X,Y = np.meshgrid(z1, z1)
        red = np.sin(X)
        green = np.cos(4*Y)
        blue = X*Y

        plt.imshow(mahotas.as_rgb(red, green, blue))

    Notice that the scaling on the ``blue`` channel is so different from the
    other channels (from 0..2500 compared with 0..1), but ``as_rgb`` stretches
    each channel independently.

    Parameters
    ----------
    r,g,b : array-like or int, optional
        The channels can be of any type or None.
        At least one must be not None and all must have the same shape.

    Returns
    -------
    rgb : ndarray
        RGB ndarray
    '''
    for c in (r,g,b):
        if c is not None:
            c = np.array(c)
            shape = c.shape
            if shape != ():
                break
    else:
        raise ValueError('mahotas.as_rgb: Not all arguments can be None')
    def s(c):
        if c is None:
            return np.zeros(shape, np.uint8)
        c = np.asanyarray(c)
        if c.shape == ():
            c = np.tile(c, shape)
            return c.astype(np.uint8, copy=False)
        elif c.shape != shape:
            sh = lambda c : (c.shape if c is not None else ' . ')
            raise ValueError('mahotas.as_rgb: Not all arguments have the same shape. Shapes were : %s' % [sh(r), sh(g), sh(b)])
        return stretch(c)
    return np.dstack([s(r), s(g), s(b)])


def overlay(gray, red=None, green=None, blue=None, if_gray_dtype_not_uint8='stretch'):
    '''
    Create an image which is greyscale, but with possible boolean overlays.

    Parameters
    ----------
    gray: ndarray of type np.uint8
        Should be a greyscale image of type np.uint8

    red,green,blue : ndarray, optional
        boolean arrays

    if_gray_dtype_not_uint8 : str, optional
        What to do if ``gray`` is not of type ``np.uint8``, must be one of
            'stretch' (default): the function ``stretch`` is called.
            'error' : in this case, an error is raised

    Returns
    -------
    overlaid : ndarray
        Colour image
    '''
    _check_2(gray, 'overlay')
    if gray.dtype != np.uint8:
        if if_gray_dtype_not_uint8 == 'stretch':
            gray = stretch(gray)
        else:
            raise ValueError('mahotas.overlay: first argument should be of dtype np.uint8')
    def _v(ch):
        if ch is None:
            return gray
        ch = ch.astype(bool, copy=False)
        ch = 255*ch
        return np.maximum(gray, ch)
    r = _v(red)
    g = _v(green)
    b = _v(blue)
    return np.dstack([r,g,b]).astype(gray.dtype, copy=False)
