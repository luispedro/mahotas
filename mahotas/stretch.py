# -*- coding: utf-8 -*-
# Copyright (C) 2009-2012, Luis Pedro Coelho <luis@luispedro.org>
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

__all__ = ['stretch', 'as_rgb']

def stretch(img, arg0=None, arg1=None, dtype=np.uint8):
    '''
    img' = stretch(img, [dtype=np.uint8])
    img' = stretch(img, max, [dtype=np.uint8])
    img' = stretch(img, min, max, [dtype=np.uint8])

    Contrast stretch the image to the range [0, max] (first form) or
        [min, max] (second form).

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

    Bugs
    ----
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
    img = img.astype(np.double)
    img -= img.min()
    ptp = img.ptp()
    if not ptp:
        img = np.zeros(img.shape, dtype)
        if min:
            img += min
        return img
    img *= float(max - min)/ptp
    if min: img += min
    return img.astype(dtype)

def as_rgb(r, g, b):
    '''
    rgb = as_rgb(r, g, b)

    Returns an RGB image

    If any of the channels is `None`, that channel is set to zero.

    Parameters
    ----------
    r,g,b : array-like, optional
        The channels can be of any type or None.
        At least one must be not None and all must have the same shape.

    Returns
    -------
    rgb : ndarray
        RGB ndarray
    '''
    for c in (r,g,b):
        if c is not None:
            shape = c.shape
            break
    else:
        raise ValueError('mahotas.as_rgb: Not all arguments can be None')
    def s(c):
        if c is None:
            return np.zeros(shape, np.uint8)
        if c.shape != shape:
            raise ValueError('mahotas.as_rgb: Not all arguments have the same shape')
        return stretch(np.asanyarray(c))
    return np.dstack([s(r), s(g), s(b)])

